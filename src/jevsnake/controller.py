"""Expected-utility control from calibrated event distributions."""

import numpy as np

from .events import UTILITIES
from .inference import predict


def expected_utility(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape[-1] != len(UTILITIES):
        raise ValueError("Wrong outcome dimension")
    return probabilities @ UTILITIES


class OutcomeController:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def action_probabilities(self, games):
        if not games:
            return []
        legal = [game.legal_actions for game in games]
        if any(not actions for actions in legal):
            raise ValueError("Cannot act in a terminal game")
        rows = []
        for game_index, (game, actions) in enumerate(zip(games, legal)):
            for action in actions:
                rows.append(
                    game.public_state()
                    | {
                        "action": action,
                        "event_horizon": self.config["event_horizon"],
                        "state_id": f"actor:{game_index}:{game.steps}",
                    }
                )
        probabilities = predict(
            self.model,
            self.tokenizer,
            rows,
            self.config["inference_questions"],
            self.config["max_length"],
            self.config["inference_precision"],
        )
        values = expected_utility(probabilities)
        result, offset = [], 0
        for actions in legal:
            local = values[offset : offset + len(actions)]
            best = int(np.argmax(local))
            result.append(
                {
                    action: float(index == best)
                    for index, action in enumerate(actions)
                }
            )
            offset += len(actions)
        return result

    def event_probabilities(self, game):
        rows = [
            game.public_state()
            | {
                "action": action,
                "event_horizon": self.config["event_horizon"],
                "state_id": f"inspect:{game.steps}",
            }
            for action in game.legal_actions
        ]
        values = predict(
            self.model,
            self.tokenizer,
            rows,
            self.config["inference_questions"],
            self.config["max_length"],
            self.config["inference_precision"],
        )
        return dict(zip(game.legal_actions, values.tolist()))
