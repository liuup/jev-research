"""Single observed events for training; MC references live in separate files."""

import json
import random
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .env import Game
from .heuristic import HeuristicPolicy
from .serialization import OUTCOMES, bucket, candidates, serialize
from .utils import digest, save_json


def rollout_event(state, action, seed, policy=None):
    game = Game(seed=seed, **{k: state[k] for k in ("board", "score", "steps")})
    policy = policy or HeuristicPolicy()
    if not game.step(action):
        raise ValueError("Illegal first action")
    binary = state.get("task") == "reach_2048"
    while not game.terminal and not (binary and game.max_tile >= 2048):
        game.step(policy.choose(game))
    return dict(
        observed_outcome=("yes" if game.max_tile >= 2048 else "no") if binary else bucket(game.max_tile),
        terminal_max_tile=game.max_tile,
        terminal_score=game.score,
        rollout_steps=game.steps - state["steps"],
    )


def rollout(state, action, seed, policy=None):
    return rollout_event(state, action, seed, policy)["observed_outcome"]


def source_game(args):
    split, game_seed, config = args
    game = Game(game_seed)
    policy = HeuristicPolicy(config["heuristic_config"])
    behavior_rng = random.Random(game_seed + 1000000000)
    rollout_rng = random.Random(game_seed + 2000000000)
    policy_id = f"heuristic:{digest(config['heuristic_config'])}"
    rows = []
    binary = config.get("task") == "reach_2048"
    while not game.terminal and not (binary and game.max_tile >= 2048):
        if game.steps % config["select_every"] == 0:
            state = game.state()
            if binary:
                state.update(task="reach_2048", candidate_ids=["yes", "no"])
            for action in game.legal_actions:
                seed = rollout_rng.randrange(2**63)
                rows.append(
                    dict(
                        **state,
                        action=action,
                        state_id=f"{split}:{game_seed}:{game.steps}",
                        source_game=f"{split}:{game_seed}",
                        rollout_seed=seed,
                        continuation_policy_id=policy_id,
                        **rollout_event(state, action, seed, policy),
                    )
                )
        action = (
            behavior_rng.choice(game.legal_actions)
            if behavior_rng.random() < config["random_action_probability"]
            else policy.choose(game)
        )
        game.step(action)
    return rows


def generate(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("*.json*")):
        raise FileExistsError(f"Refusing to overwrite dataset: {output}")
    required_splits = {"train", "dev", "calibration", "test"}
    if set(config["questions"]) != required_splits:
        raise ValueError(f"Expected exactly these splits: {sorted(required_splits)}")
    if set(config["split_seeds"]) != required_splits:
        raise ValueError("Every split needs an independent seed range")
    if len(set(config["split_seeds"].values())) != len(required_splits):
        raise ValueError("Split seeds must be unique")
    if config["select_every"] < 1 or any(n < 1 for n in config["questions"].values()):
        raise ValueError("Question counts and selection interval must be positive")
    benchmark = json.loads(Path("results/heuristic_benchmark.json").read_text())
    assert benchmark["games"] >= 100
    assert benchmark["policy_sha256"] == digest(config["heuristic_config"])
    manifest = dict(
        schema_version=2,
        task=config.get("task", "terminal_max_tile"),
        config=config,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        git_dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        generator_code_sha256=digest("src/jev2048/dataset.py"),
        policy_sha256=digest(config["heuristic_config"]),
        policy_code_sha256=digest("src/jev2048/heuristic.py"),
        simulator_sha256=digest("src/jev2048/env.py"),
        training_labels="one observed terminal event per state-action; never MC probabilities",
        splits={},
    )
    split_groups = {}
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
        groups = sorted({row["source_game"] for row in rows})
        split_groups[split] = {
            int(source_game.rsplit(":", 1)[1]) for source_game in groups
        }
        manifest["splits"][split] = dict(
            questions=len(rows),
            states=len({row["state_id"] for row in rows}),
            source_games=len(groups),
            source_seeds=game_ids,
            outcome_counts=dict(Counter(row["observed_outcome"] for row in rows)),
            sha256=digest(path),
        )
        print(split, len(rows), flush=True)
    if any(
        left & right
        for index, left in enumerate(split_groups.values())
        for right in list(split_groups.values())[index + 1 :]
    ):
        raise AssertionError("Source-game leakage across dataset splits")
    save_json(output / "manifest.json", manifest)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def mc_pair(args):
    row, repeats, seed, heuristic_config = args
    ids = candidates(row)
    counts = np.zeros(len(ids), dtype=int)
    rng, policy = random.Random(seed), HeuristicPolicy(heuristic_config)
    for _ in range(repeats):
        counts[
            ids.index(rollout(row, row["action"], rng.randrange(2**63), policy))
        ] += 1
    return {
        k: v
        for k, v in row.items()
        if k
        not in (
            "observed_outcome",
            "rollout_seed",
            "terminal_max_tile",
            "terminal_score",
            "rollout_steps",
        )
    } | dict(
        counts=counts.tolist(),
        q=(counts / repeats).tolist(),
        mc_seed=seed,
        rollouts=repeats,
    )


def build_reference(data_dir, states=64, repeats=256, workers=16, seed=717):
    root = Path(data_dir)
    if states < 1 or repeats < 2:
        raise ValueError("Need positive state count and at least two MC rollouts")
    if (root / "mc_reference.jsonl").exists():
        raise FileExistsError("Reference cohort is frozen; choose a new data directory")
    manifest = json.loads((root / "manifest.json").read_text())
    heuristic_config = manifest["config"]["heuristic_config"]
    assert manifest["policy_sha256"] == digest(heuristic_config)
    assert manifest["policy_code_sha256"] == digest("src/jev2048/heuristic.py")
    by_state = defaultdict(list)
    for row in read_rows(root / "test.jsonl"):
        by_state[row["state_id"]].append(row)
    if states > len(by_state):
        raise ValueError(f"Requested {states} states from only {len(by_state)}")
    selected = random.Random(seed).sample(sorted(by_state), states)
    cohort = [row for state_id in selected for row in by_state[state_id]]
    for state_id in selected:
        game = Game(
            board=by_state[state_id][0]["board"],
            score=by_state[state_id][0]["score"],
            steps=by_state[state_id][0]["steps"],
        )
        if [row["action"] for row in by_state[state_id]] != game.legal_actions:
            raise AssertionError("MC reference requires every legal action per state")
    with ProcessPoolExecutor(workers) as pool:
        rows = list(
            pool.map(
                mc_pair,
                [
                    (r, repeats, seed + i + 100000000, heuristic_config)
                    for i, r in enumerate(cohort)
                ],
            )
        )
    path = root / "mc_reference.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    save_json(
        root / "mc_manifest.json",
        dict(
            states=states,
            pairs=len(rows),
            repeats=repeats,
            seed=seed,
            training_labels=False,
            data_manifest_sha256=digest(root / "manifest.json"),
            sha256=digest(path),
        ),
    )


def collate(rows, tokenizer, max_length=512):
    ids = [candidates(r) for r in rows]
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
