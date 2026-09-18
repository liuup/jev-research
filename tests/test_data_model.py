from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from jev2048.dataset import collate, mc_pair, read_rows
from jev2048.env import Game
from jev2048.model import DecisionHead, JevModel
from jev2048.serialization import OUTCOMES, bucket, serialize


def row():
    return dict(
        **Game(17).state(), action="LEFT", state_id="test:17:0", observed_outcome="1024"
    )


def test_buckets_and_serialization():
    assert [bucket(t) for t in [2, 256, 512, 1024, 2048, 4096, 8192, 16384]] == list(
        OUTCOMES
    ) + ["8192_plus"]
    r = row()
    a = serialize(r, "512")
    r["observed_outcome"] = "4096"
    assert a == serialize(r, "512") and a.endswith("[CANDIDATE]\n512")


class Tokenizer:
    def __call__(self, paths, **kwargs):
        return dict(
            input_ids=torch.ones(len(paths), 4, dtype=torch.long),
            attention_mask=torch.ones(len(paths), 4, dtype=torch.long),
        )


def test_batch_variable_candidates():
    a = row()
    b = row() | dict(candidate_ids=["256", "1024"])
    batch = collate([a, b], Tokenizer())
    assert batch["offsets"] == [0, 7, 9]
    assert batch["mask"].sum(1).tolist() == [7, 2]
    assert batch["targets"].tolist() == [3, 1]
    assert len(batch["action_ids"]) == len(batch["state_ids"]) == 2


def test_head_permutation_and_padding():
    torch.manual_seed(17)
    head = DecisionHead(12, 8, 2).eval()
    h = torch.randn(2, 4, 12)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    p = head(h, mask).exp()
    perm = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(
        head(h[:, perm], mask[:, perm]).exp()[:, perm.argsort()], p
    )
    torch.testing.assert_close(p[1, :2], head(h[1:2, :2], mask[1:2, :2]).exp()[0])
    assert torch.equal(p[1, 2:], torch.zeros(2))


def test_last_meaningful_token_pooling():
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(1))

        def forward(self, input_ids, attention_mask, use_cache):
            assert not use_cache
            return SimpleNamespace(
                last_hidden_state=input_ids.float().unsqueeze(-1) * self.scale
            )

    class Capture(nn.Module):
        def forward(self, hidden, mask):
            return hidden.squeeze(-1)

    m = JevModel(Backbone(), Capture())
    batch = dict(
        tokens=dict(
            input_ids=torch.tensor([[0, 4, 9], [2, 7, 0], [0, 0, 3]]),
            attention_mask=torch.tensor([[0, 1, 1], [1, 1, 0], [0, 0, 1]]),
        ),
        offsets=[0, 2, 3],
        mask=torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
    )
    torch.testing.assert_close(m(batch), torch.tensor([[9.0, 7.0], [3.0, 0.0]]))


def test_reference_terminal_continuation():
    r = dict(
        board=((2, 2, 8, 16), (8, 16, 32, 64), (16, 32, 64, 128), (32, 64, 128, 256)),
        score=0,
        steps=0,
        action="LEFT",
        state_id="test:1:0",
        observed_outcome="256",
    )
    a = mc_pair((r, 4, 17))
    b = mc_pair((r, 4, 17))
    assert a == b and sum(a["counts"]) == 4 and sum(a["q"]) == 1
    assert "observed_outcome" not in a


def test_split_isolation():
    c = yaml.safe_load(open("configs/data.yaml"))
    ranges = [set(range(seed, seed + 100000)) for seed in c["split_seeds"].values()]
    assert all(not a & b for i, a in enumerate(ranges) for b in ranges[i + 1 :])


def test_generated_smoke_split_isolation():
    if not Path("data/smoke/manifest.json").exists():
        pytest.skip("Generate smoke data for artifact audit")
    splits = [
        read_rows(f"data/smoke/{s}.jsonl")
        for s in ("train", "dev", "calibration", "test")
    ]
    groups = [set(r["source_game"] for r in rows) for rows in splits]
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i + 1 :])
    for rows in splits:
        for r in rows:
            game = Game(board=r["board"], score=r["score"], steps=r["steps"])
            actions = [x["action"] for x in rows if x["state_id"] == r["state_id"]]
            assert actions == game.legal_actions
            assert "q" not in r and r["observed_outcome"] in OUTCOMES
