"""CPU integrity and leakage audit for generated single-event datasets."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from jev2048.dataset import read_rows
from jev2048.env import Game
from jev2048.serialization import OUTCOMES
from jev2048.utils import digest, save_json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/main")
    a = p.parse_args()
    root = Path(a.data_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    seen = set()
    report = {}
    for split, info in manifest["splits"].items():
        assert digest(root / f"{split}.jsonl") == info["sha256"]
        rows = read_rows(root / f"{split}.jsonl")
        groups = defaultdict(list)
        seeds = set()
        for r in rows:
            assert "q" not in r and r["observed_outcome"] in OUTCOMES
            groups[r["state_id"]].append(r)
            seeds.add(int(r["source_game"].split(":")[1]))
        assert not seen & seeds
        seen.update(seeds)
        for state_id, group in groups.items():
            g = Game(**{k: group[0][k] for k in ("board", "score", "steps")})
            assert [r["action"] for r in group] == g.legal_actions, state_id
            assert all(r["board"] == group[0]["board"] for r in group)
            assert len({r["rollout_seed"] for r in group}) == len(group)
        report[split] = dict(
            questions=len(rows),
            states=len(groups),
            source_games=len(seeds),
            observed_outcomes=dict(Counter(r["observed_outcome"] for r in rows)),
        )
    save_json(root / "audit.json", report)
    print(json.dumps(report, indent=2))
