import numpy as np
import unittest

from jevsnake.controller import expected_utility
from jevsnake.env import SnakeGame
from jevsnake.evaluation import promotion_decision
from jevsnake.serialization import serialize


def row():
    game = SnakeGame(25, 8, 3)
    return game.public_state() | {
        "state_id": "test",
        "action": "north",
        "event_horizon": 16,
    }


class InterfaceTests(unittest.TestCase):
    def test_input_contains_visible_state_action_and_event_but_no_oracle_fields(self):
        value = row() | {"seed": 999, "rng_state": 123, "q": [0.1, 0.2, 0.7]}
        text = serialize(value, "food")
        self.assertIn("BodyHeadFirst", text)
        self.assertIn("Move north", text)
        self.assertIn("at most 15 additional moves", text)
        self.assertNotIn("999", text)
        self.assertNotIn("rng_state", text)
        self.assertNotIn("0.7", text)

    def test_expected_utility_is_food_probability_minus_collision_probability(self):
        probabilities = np.asarray([[0.15, 0.25, 0.60], [0.05, 0.60, 0.35]])
        self.assertTrue(np.allclose(expected_utility(probabilities), [0.45, 0.30]))

    def test_promotion_requires_positive_one_sided_lower_bound(self):
        incumbent = {"records": [{"seed": i, "food": 0} for i in range(8)]}
        candidate = {"records": [{"seed": i, "food": 1} for i in range(8)]}
        decision = promotion_decision(incumbent, candidate, 0.0, 1.645)
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["paired_mean_food_gain"], 1.0)
        tied = promotion_decision(incumbent, incumbent, 0.0, 1.645)
        self.assertFalse(tied["accepted"])
