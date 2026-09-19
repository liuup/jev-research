import torch
import unittest

from jevsnake.collector import OnlineCollector, validate_events
from jevsnake.policies import UniformPolicy


def config():
    return {
        "seed": 25,
        "source_envs": 4,
        "board_size": 6,
        "initial_length": 3,
        "behavior_epsilon": 0.2,
        "select_every": 1,
        "event_horizon": 4,
    }


class CollectorTests(unittest.TestCase):
    def test_strict_online_collector_is_reproducible_and_observed_only(self):
        first = OnlineCollector(UniformPolicy(), config(), 0)
        second = OnlineCollector(UniformPolicy(), config(), 0)
        rows = first.collect(32)
        torch.rand(100)
        self.assertEqual(rows, second.collect(32))
        self.assertTrue(all(row["observed_outcome"] in {"collision", "timeout", "food"} for row in rows))
        self.assertTrue(all("q" not in row and 1 <= row["event_steps"] <= 4 for row in rows))
        self.assertGreaterEqual(first.counters["events_completed"], 32)
        validate_events(rows, first.policy_id, 0)

    def test_collector_records_only_the_executed_action(self):
        collector = OnlineCollector(UniformPolicy(), config(), 0)
        rows = collector.collect(16)
        self.assertEqual(len({row["state_id"] for row in rows}), len(rows))
        self.assertTrue(all(isinstance(row["action"], str) for row in rows))
        with self.assertRaises(ValueError):
            validate_events([rows[0] | {"policy_id": "stale"}], collector.policy_id, 0)
