"""Derive binary observed events without resampling or changing source splits."""

import argparse
import json
from collections import Counter
from pathlib import Path

from jev2048.dataset import read_rows
from jev2048.utils import digest, save_json


def convert(source, output, limit=None):
    source, output = Path(source), Path(output)
    manifest = json.loads((source / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    groups = []
    manifest = dict(manifest, task="reach_2048", parent_manifest_sha256=digest(source / "manifest.json"),
                    transformation="Filter initial max>=2048; terminal max>=2048 is yes; preserve splits",
                    splits={})
    for split in ("train", "dev", "calibration", "test"):
        rows = [r for r in read_rows(source / f"{split}.jsonl") if max(map(max, r["board"])) < 2048]
        if limit and len(rows) > limit:
            cutoff = rows[limit - 1]["state_id"]
            end = limit
            while end < len(rows) and rows[end]["state_id"] == cutoff:
                end += 1
            rows = rows[:end]
        for r in rows:
            r.update(task="reach_2048", candidate_ids=["yes", "no"],
                     observed_outcome="yes" if r["terminal_max_tile"] >= 2048 else "no")
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        group = {r["source_game"].rsplit(":", 1)[-1] for r in rows}
        assert all(not group & other for other in groups)
        groups.append(group)
        manifest["splits"][split] = dict(questions=len(rows), states=len({r["state_id"] for r in rows}),
                                         source_games=len(group), sha256=digest(path),
                                         outcome_counts=dict(Counter(r["observed_outcome"] for r in rows)))
    save_json(output / "manifest.json", manifest)
    if not limit and (source / "mc_reference.jsonl").exists():
        refs = read_rows(source / "mc_reference.jsonl")
        refs = [r for r in refs if max(map(max, r["board"])) < 2048]
        for r in refs:
            r.update(task="reach_2048", candidate_ids=["yes", "no"],
                     q=[sum(r["q"][4:]), sum(r["q"][:4])],
                     counts=[sum(r["counts"][4:]), sum(r["counts"][:4])])
        path = output / "mc_reference.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in refs))
        save_json(output / "mc_manifest.json", dict(data_manifest_sha256=digest(output / "manifest.json"),
                                                   sha256=digest(path), parent=str(source)))
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/offline_pi0_v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    convert(args.source, args.output, args.limit)
