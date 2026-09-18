"""Policy improvement is decided by held-out games, never by training reward."""

from collections import defaultdict

import numpy as np

from .controller import outcome_utility
from .env import Game
from .utils import game_summary


def play_seeded(policy, seeds, batch_size=8, game_factory=Game):
    finished, records = [], []
    for offset in range(0, len(seeds), batch_size):
        batch_seeds = seeds[offset : offset + batch_size]
        games = [game_factory(seed) for seed in batch_seeds]
        active = [i for i, g in enumerate(games) if not g.terminal]
        while active:
            actions = policy.choose_many([games[i] for i in active])
            if len(actions) != len(active):
                raise ValueError("Wrong number of evaluation actions")
            for index, action in zip(active, actions):
                if not games[index].step(action):
                    raise ValueError("Invalid evaluation action")
            active = [i for i in active if not games[i].terminal]
        finished.extend(games)
        records.extend(
            dict(seed=seed, score=g.score, max_tile=g.max_tile, steps=g.steps)
            for seed, g in zip(batch_seeds, games)
        )
    return dict(policy_id=policy.policy_id, **game_summary(finished), records=records)


def promotion_decision(incumbent, candidate, min_relative_gain):
    if min_relative_gain < 0:
        raise ValueError("Promotion margin must be nonnegative")
    if [r["seed"] for r in incumbent["records"]] != [
        r["seed"] for r in candidate["records"]
    ]:
        raise ValueError("Promotion games must use identical seeds")
    differences = np.array(
        [
            new["score"] - old["score"]
            for old, new in zip(incumbent["records"], candidate["records"])
        ]
    )
    threshold = incumbent["mean_score"] * (1 + min_relative_gain)
    return dict(
        accepted=bool(candidate["mean_score"] > threshold),
        metric="mean_score",
        required_score=threshold,
        incumbent_mean_score=incumbent["mean_score"],
        candidate_mean_score=candidate["mean_score"],
        paired_mean_score_delta=float(differences.mean()),
        paired_score_delta_se=float(differences.std(ddof=1) / len(differences) ** 0.5)
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
