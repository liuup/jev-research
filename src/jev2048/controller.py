"""Greedy controllers built from outcome models; utilities are not action probabilities."""

import numpy as np

from .evaluation import LOG_TILE_UTILITIES, predict


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
    ):
        self.model, self.tokenizer = model, tokenizer
        self.utility, self.threshold, self.max_length = utility, threshold, max_length
        self.inference_questions = inference_questions
        self.inference_precision = inference_precision

    def choose(self, game):
        return self.choose_many([game])[0]

    def choose_many(self, games):
        if not games:
            return []
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
