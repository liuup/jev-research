"""CPU integrity and leakage audit for generated single-event datasets."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from jev2048.dataset import read_rows
from jev2048.env import Game
from jev2048.serialization import candidates
from jev2048.utils import digest, save_json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/offline_pi0_v1")
    a = p.parse_args()
    root = Path(a.data_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    seen = set()
    expected_policy_id = f"heuristic:{manifest['policy_sha256']}"
    report = {}
    for split, info in manifest["splits"].items():
        assert digest(root / f"{split}.jsonl") == info["sha256"]
        rows = read_rows(root / f"{split}.jsonl")
        groups = defaultdict(list)
        seeds = set()
        for r in rows:
            assert "q" not in r and r["observed_outcome"] in candidates(r)
            if manifest.get("task") == "reach_2048":
                assert r["task"] == "reach_2048"
                assert max(map(max, r["board"])) < 2048
                assert (r["observed_outcome"] == "yes") == (r["terminal_max_tile"] >= 2048)
            assert r["continuation_policy_id"] == expected_policy_id
            assert r["terminal_max_tile"] >= 2
            assert r["terminal_score"] >= r["score"]
            assert r["rollout_steps"] >= 1
            groups[r["state_id"]].append(r)
            seeds.add(int(r["source_game"].rsplit(":", 1)[1]))
        assert not seen & seeds
        seen.update(seeds)
        for state_id, group in groups.items():
            g = Game(**{k: group[0][k] for k in ("board", "score", "steps")})
            assert [r["action"] for r in group] == g.legal_actions, state_id
            assert all(r["board"] == group[0]["board"] for r in group)
            assert len({r["rollout_seed"] for r in group}) == len(group)
        outcomes = Counter(r["observed_outcome"] for r in rows)
        assert len(rows) == info["questions"]
        assert len(groups) == info["states"]
        assert len(seeds) == info["source_games"]
        assert dict(outcomes) == info["outcome_counts"]
        report[split] = dict(
            questions=len(rows),
            states=len(groups),
            source_games=len(seeds),
            observed_outcomes=dict(outcomes),
        )
    save_json(root / "audit.json", report)
    print(json.dumps(report, indent=2))
