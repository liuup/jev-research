"""Text-only Qwen3.5 backbone with a permutation-equivariant outcome head."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoTokenizer, Qwen3_5TextConfig, Qwen3_5TextModel
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from .objectives import masked_log_probs


class DecisionHead(nn.Module):
    def __init__(self, hidden_size, width=256, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.project = nn.Linear(hidden_size, width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=0, batch_first=True
        )
        self.score = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, hidden, mask):
        projected = self.project(self.norm(hidden.float()))
        attended = self.attention(
            projected,
            projected,
            projected,
            key_padding_mask=~mask,
            need_weights=False,
        )[0]
        return masked_log_probs(self.score(projected + attended).squeeze(-1), mask)


class JevSnakeModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    @classmethod
    def load_base(cls, config):
        root = Path(config["base_model"])
        raw = json.loads((root / "config.json").read_text())
        text_config = Qwen3_5TextConfig(**raw["text_config"])
        text_config.use_cache = False
        text_config._attn_implementation = "sdpa"
        with torch.device("meta"):
            backbone = Qwen3_5TextModel(text_config)
        index = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        prefix = "model.language_model."
        state = {}
        for filename in sorted(set(index.values())):
            with safe_open(root / filename, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key.startswith(prefix):
                        state[key[len(prefix) :]] = handle.get_tensor(key).to(
                            torch.bfloat16
                        )
        backbone.load_state_dict(state, strict=True, assign=True)
        backbone.rotary_emb = Qwen3_5TextRotaryEmbedding(text_config, device="cpu")
        if config.get("gradient_checkpointing", True):
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        return cls(
            backbone,
            DecisionHead(
                text_config.hidden_size,
                config["head_width"],
                config["attention_heads"],
            ),
        )

    def forward(self, batch):
        device = next(self.parameters()).device
        tokens = {
            key: value.to(device)
            for key, value in batch["tokens"].items()
            if key in ("input_ids", "attention_mask")
        }
        hidden = self.backbone(**tokens, use_cache=False).last_hidden_state
        positions = torch.arange(hidden.shape[1], device=device).expand(hidden.shape[:2])
        last = positions.masked_fill(
            ~tokens["attention_mask"].bool(), -1
        ).max(-1).values
        pooled = hidden[torch.arange(hidden.shape[0], device=device), last].float()
        offsets = batch["offsets"]
        mask = batch["mask"].to(device)
        groups = torch.nn.utils.rnn.pad_sequence(
            [pooled[start:end] for start, end in zip(offsets[:-1], offsets[1:])],
            batch_first=True,
        )
        return self.head(groups, mask)


def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
