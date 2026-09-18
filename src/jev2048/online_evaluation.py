"""Policy improvement is decided by held-out games, never by training reward."""

import json
import time
from collections import defaultdict

import numpy as np

from .controller import outcome_utility
from .env import Game
from .utils import game_summary


def play_seeded(policy, seeds, batch_size=8, game_factory=Game, progress=None):
    """Refill completed slots immediately; return results in input seed order."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    seeds = list(seeds)
    finished = [None] * len(seeds)
    active = {}
    next_index = completed = 0
    started = last_report = time.monotonic()

    def report(event):
        if progress is not None:
            print(json.dumps(dict(
                event=event, **progress,
                completed_games=completed,
                total_games=len(seeds), active_games=len(active),
                active_steps=[g.steps for g in active.values()],
                elapsed_seconds=round(time.monotonic() - started, 2),
            )), flush=True)

    report("evaluation_start")
    while next_index < len(seeds) or active:
        previous_completed = completed
        while len(active) < batch_size and next_index < len(seeds):
            game = game_factory(seeds[next_index])
            if game.terminal:
                finished[next_index] = game
                completed += 1
            else:
                active[next_index] = game
            next_index += 1
        if active:
            actions = policy.choose_many(list(active.values()))
            if len(actions) != len(active):
                raise ValueError("Wrong number of evaluation actions")
            for index, action in zip(list(active), actions):
                game = active[index]
                if not game.step(action):
                    raise ValueError("Invalid evaluation action")
                if game.terminal:
                    finished[index] = active.pop(index)
                    completed += 1
        now = time.monotonic()
        if completed != previous_completed or now - last_report >= 10:
            report("evaluation_progress")
            last_report = now
    records = [
        dict(seed=seed, score=g.score, max_tile=g.max_tile, steps=g.steps)
        for seed, g in zip(seeds, finished)
    ]
    report("evaluation_complete")
    return dict(policy_id=policy.policy_id, **game_summary(finished), records=records)


def promotion_decision(incumbent, candidate, min_log_tile_gain):
    if min_log_tile_gain < 0:
        raise ValueError("Expected log-tile gain must be nonnegative")
    if [r["seed"] for r in incumbent["records"]] != [
        r["seed"] for r in candidate["records"]
    ]:
        raise ValueError("Promotion games must use identical seeds")
    differences = np.array(
        [
            np.log2(new["max_tile"]) - np.log2(old["max_tile"])
            for old, new in zip(incumbent["records"], candidate["records"])
        ]
    )
    threshold = incumbent["mean_log_tile"] + min_log_tile_gain
    return dict(
        accepted=bool(candidate["mean_log_tile"] > threshold),
        metric="mean_log_tile",
        required_mean_log_tile=threshold,
        incumbent_mean_log_tile=incumbent["mean_log_tile"],
        candidate_mean_log_tile=candidate["mean_log_tile"],
        paired_mean_log_tile_delta=float(differences.mean()),
        paired_log_tile_delta_se=float(differences.std(ddof=1) / len(differences) ** 0.5)
        if len(differences) > 1
        else None,
        games=len(differences),
        selection_split="promotion",
        test_used_for_selection=False,
    )


def action_ranking_metrics(predictions, rows, utility, threshold):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["state_id"]].append(index)
    predicted = outcome_utility(predictions, utility, threshold)
    reference = outcome_utility(
        np.array([row["q"] for row in rows]), utility, threshold
    )
    agreement, regrets, gaps = [], [], []
    for indices in groups.values():
        p, q = predicted[indices], reference[indices]
        chosen = int(p.argmax())
        agreement.append(bool(np.isclose(q[chosen], q.max())))
        regrets.append(float(q.max() - q[chosen]))
        gaps.append(float(p.max() - p.min()))
    return dict(
        states=len(groups),
        mc_greedy_agreement=float(np.mean(agreement)),
        mc_utility_regret=float(np.mean(regrets)),
        mean_predicted_action_gap=float(np.mean(gaps)),
        caveat="Finite MC action rankings have sampling error",
    )
