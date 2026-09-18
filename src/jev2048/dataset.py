"""Single observed events for training; MC references live in separate files."""

import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .env import Game
from .heuristic import HeuristicPolicy
from .serialization import OUTCOMES, bucket, serialize
from .utils import digest, save_json


def rollout(state, action, seed, policy=None):
    game = Game(seed=seed, **{k: state[k] for k in ("board", "score", "steps")})
    policy = policy or HeuristicPolicy()
    if not game.step(action):
        raise ValueError("Illegal first action")
    while not game.terminal:
        game.step(policy.choose(game))
    return bucket(game.max_tile)


def source_game(args):
    split, game_seed, config = args
    game, policy = Game(game_seed), HeuristicPolicy()
    rng = random.Random(game_seed + 1000000000)
    rows = []
    while not game.terminal:
        if game.steps % config["select_every"] == 0:
            state = game.state()
            for action in game.legal_actions:
                seed = rng.randrange(2**63)
                rows.append(
                    dict(
                        **state,
                        action=action,
                        state_id=f"{split}:{game_seed}:{game.steps}",
                        source_game=f"{split}:{game_seed}",
                        rollout_seed=seed,
                        observed_outcome=rollout(state, action, seed, policy),
                    )
                )
        action = (
            rng.choice(game.legal_actions)
            if rng.random() < config["random_action_probability"]
            else policy.choose(game)
        )
        game.step(action)
    return rows


def generate(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("*.json*")):
        raise FileExistsError(f"Refusing to overwrite dataset: {output}")
    if config["select_every"] < 1 or any(n < 1 for n in config["questions"].values()):
        raise ValueError("Question counts and selection interval must be positive")
    benchmark = json.loads(Path("results/heuristic_benchmark.json").read_text())
    assert benchmark["games"] >= 100
    assert benchmark["policy_sha256"] == digest("configs/heuristic.yaml")
    manifest = dict(
        config=config,
        policy_sha256=digest("configs/heuristic.yaml"),
        policy_code_sha256=digest("src/jev2048/heuristic.py"),
        splits={},
    )
    for split, count in config["questions"].items():
        rows, game_ids, index = [], [], 0
        with ProcessPoolExecutor(config["workers"]) as pool:
            while len(rows) < count:
                tasks = [
                    (split, config["split_seeds"][split] + index + i, config)
                    for i in range(config["workers"])
                ]
                for task, result in zip(tasks, pool.map(source_game, tasks)):
                    if len(rows) >= count:
                        break
                    rows.extend(result)
                    game_ids.append(task[1])
                index += config["workers"]
        # Keep every legal action for each selected state, even at the cutoff.
        cutoff = rows[min(count, len(rows)) - 1]["state_id"]
        end = min(count, len(rows))
        while end < len(rows) and rows[end]["state_id"] == cutoff:
            end += 1
        rows = rows[:end]
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest["splits"][split] = dict(
            questions=len(rows), source_seeds=game_ids, sha256=digest(path)
        )
        print(split, len(rows), flush=True)
    save_json(output / "manifest.json", manifest)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def mc_pair(args):
    row, repeats, seed = args
    counts = np.zeros(len(OUTCOMES), dtype=int)
    rng, policy = random.Random(seed), HeuristicPolicy()
    for _ in range(repeats):
        counts[
            OUTCOMES.index(rollout(row, row["action"], rng.randrange(2**63), policy))
        ] += 1
    return {
        k: v for k, v in row.items() if k not in ("observed_outcome", "rollout_seed")
    } | dict(
        counts=counts.tolist(),
        q=(counts / repeats).tolist(),
        mc_seed=seed,
        rollouts=repeats,
    )


def build_reference(data_dir, pairs=256, repeats=256, workers=16, seed=717):
    root = Path(data_dir)
    if pairs < 1 or repeats < 2:
        raise ValueError("Need positive pair count and at least two MC rollouts")
    if (root / "mc_reference.jsonl").exists():
        raise FileExistsError("Reference cohort is frozen; choose a new data directory")
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["policy_sha256"] == digest("configs/heuristic.yaml")
    assert manifest["policy_code_sha256"] == digest("src/jev2048/heuristic.py")
    cohort = random.Random(seed).sample(read_rows(root / "test.jsonl"), pairs)
    with ProcessPoolExecutor(workers) as pool:
        rows = list(
            pool.map(
                mc_pair,
                [(r, repeats, seed + i + 100000000) for i, r in enumerate(cohort)],
            )
        )
    path = root / "mc_reference.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    save_json(
        root / "mc_manifest.json",
        dict(
            pairs=pairs,
            repeats=repeats,
            seed=seed,
            training_labels=False,
            data_manifest_sha256=digest(root / "manifest.json"),
            sha256=digest(path),
        ),
    )


def collate(rows, tokenizer, max_length=512):
    ids = [list(r.get("candidate_ids", OUTCOMES)) for r in rows]
    if any(not x or len(x) != len(set(x)) for x in ids):
        raise ValueError("Empty or duplicate candidates")
    paths = [serialize(r, c) for r, cs in zip(rows, ids) for c in cs]
    tokens = tokenizer(paths, padding=True, truncation=False, return_tensors="pt")
    if tokens["input_ids"].shape[1] > max_length:
        raise ValueError("Input exceeds max_length; refusing to truncate candidate")
    offsets = [0]
    for cs in ids:
        offsets.append(offsets[-1] + len(cs))
    mask = (
        torch.arange(max(map(len, ids)))[None, :]
        < torch.tensor(list(map(len, ids)))[:, None]
    )
    return dict(
        tokens=tokens,
        offsets=offsets,
        mask=mask,
        candidate_ids=ids,
        action_ids=[r["action"] for r in rows],
        state_ids=[r["state_id"] for r in rows],
        targets=torch.tensor(
            [
                cs.index(r["observed_outcome"]) if "observed_outcome" in r else -1
                for r, cs in zip(rows, ids)
            ]
        ),
    )
