"""A real but tiny model and tokenizer so the loops can run on CPU.

These are not stand-ins for measurements: the model computes log-probabilities from its
own embeddings and a linear head, so gradients, masks and metrics behave exactly as they
do with the Qwen backbone, only small enough for a laptop-class CPU test.
"""

import hashlib

import torch
from torch import nn

from .objectives import masked_log_probs


class TinyTokenizer:
    """Whitespace tokenizer with a fixed vocabulary and deterministic ids."""

    def __init__(self, vocab=512):
        self.vocab = vocab
        self.pad_token_id = 0

    def _encode(self, text):
        ids = []
        for word in text.split():
            digest = hashlib.sha256(word.encode()).digest()
            ids.append(1 + int.from_bytes(digest[:2], "big") % (self.vocab - 1))
        return ids or [1]

    def __call__(self, texts, padding=True, truncation=False, return_tensors="pt"):
        encoded = [self._encode(text) for text in texts]
        width = max(len(ids) for ids in encoded)
        input_ids = torch.zeros(len(encoded), width, dtype=torch.long)
        attention_mask = torch.zeros(len(encoded), width, dtype=torch.long)
        for index, ids in enumerate(encoded):
            input_ids[index, : len(ids)] = torch.tensor(ids)
            attention_mask[index, : len(ids)] = 1
        return dict(input_ids=input_ids, attention_mask=attention_mask)


class TinyModel(nn.Module):
    """Mean-pooled embedding, then a linear head over the candidate paths."""

    def __init__(self, vocab=512, width=32):
        super().__init__()
        self.backbone = nn.ModuleDict(dict(embed=nn.Embedding(vocab, width)))
        self.head = nn.Linear(width, 2)

    def forward(self, batch):
        hidden = self.backbone["embed"](batch["tokens"]["input_ids"]).mean(1)
        offsets = batch["offsets"]
        pooled = torch.stack(
            [hidden[start:end].mean(0) for start, end in zip(offsets[:-1], offsets[1:])]
        )
        return masked_log_probs(self.head(pooled), batch["mask"])


def tiny_model_and_tokenizer(vocab=512, width=32):
    return TinyModel(vocab=vocab, width=width), TinyTokenizer(vocab=vocab)
