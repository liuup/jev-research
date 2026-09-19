import unittest

from jevsnake.config import load_config
from jevsnake.trainer import validate_online_config


class ConfigTests(unittest.TestCase):
    def test_configs_are_valid_and_seeded_25(self):
        for path in ("configs/online.yaml", "configs/online_smoke.yaml"):
            config = load_config(path)
            validate_online_config(config)
            self.assertEqual(config["seed"], 25)

    def test_event_schema_and_utility_are_fixed(self):
        config = load_config("configs/online_smoke.yaml")
        with self.assertRaises(ValueError):
            validate_online_config(config | {"outcomes": ["food", "collision", "timeout"]})
        with self.assertRaises(ValueError):
            validate_online_config(config | {"utilities": [-1, 0, 2]})
