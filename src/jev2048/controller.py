"""Greedy controllers built from outcome models; utilities are not action probabilities."""

import numpy as np

from .evaluation import LOG_TILE_UTILITIES, predict
from .serialization import action_questions


def analyze_reach_2048(model, tokenizer, game, continuation_policy_id,
                       max_length=512, precision="fp32"):
    """All legal actions in one backbone batch; requires a binary-trained model."""
    rows = action_questions(game, continuation_policy_id)
    probabilities = predict(model, tokenizer, rows, microbatch=len(rows),
                            max_length=max_length, precision=precision)
    return {
        row["action"]: {candidate: float(value) for candidate, value in
                        zip(row["candidate_ids"], distribution)}
        for row, distribution in zip(rows, probabilities)
    }


def outcome_utility(p, utility="threshold", threshold=2048):
    if utility == "threshold":
        if threshold not in (256, 512, 1024, 2048, 4096, 8192):
            raise ValueError(threshold)
        return p[:, np.array([128, 256, 512, 1024, 2048, 4096, 8192]) >= threshold].sum(
            -1
        )
    if utility == "log_tile":
        return p @ LOG_TILE_UTILITIES
    raise ValueError(utility)


class JevController:
    def __init__(
        self,
        model,
        tokenizer,
        utility="threshold",
        threshold=2048,
        max_length=512,
        inference_questions=16,
        inference_precision="fp32",
        task="terminal_max_tile",
        continuation_policy_id=None,
    ):
        self.model, self.tokenizer = model, tokenizer
        self.utility, self.threshold, self.max_length = utility, threshold, max_length
        self.inference_questions = inference_questions
        self.inference_precision = inference_precision
        self.task = task
        self.continuation_policy_id = continuation_policy_id

    def choose(self, game):
        return self.choose_many([game])[0]

    def choose_many(self, games):
        if not games:
            return []
        if self.task == "reach_2048":
            if self.utility != "threshold" or self.threshold != 2048:
                raise ValueError("Binary model only supports reaching 2048")
            rows = [row for i, game in enumerate(games) for row in
                    action_questions(game, self.continuation_policy_id, f"actor:{i}:{game.steps}")]
            p = predict(self.model, self.tokenizer, rows, max(4, self.inference_questions),
                        self.max_length, precision=self.inference_precision)
            actions, offset = [], 0
            for game in games:
                legal = game.legal_actions
                actions.append(legal[int(p[offset:offset+len(legal), 0].argmax())])
                offset += len(legal)
            return actions
        legal = [game.legal_actions for game in games]
        if any(not actions for actions in legal):
            raise ValueError("Cannot act in a terminal game")
        rows = [
            dict(**game.state(), action=a, state_id=f"actor:{i}:{game.steps}")
            for i, (game, actions) in enumerate(zip(games, legal))
            for a in actions
        ]
        p = predict(
            self.model, self.tokenizer, rows, self.inference_questions, self.max_length,
            precision=self.inference_precision,
        )
        values = outcome_utility(p, self.utility, self.threshold)
        result, offset = [], 0
        for actions in legal:
            result.append(actions[int(values[offset : offset + len(actions)].argmax())])
            offset += len(actions)
        return result
