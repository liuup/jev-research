import copy
import unittest

from jevsnake.env import SnakeGame


def state(game, **changes):
    value = game.full_state()
    value.update(changes)
    return value


class SnakeEnvironmentTests(unittest.TestCase):
    def test_seeded_food_stream_and_clone_are_reproducible(self):
        first = SnakeGame(25, 8, 3)
        second = SnakeGame(25, 8, 3)
        self.assertEqual(first.full_state(), second.full_state())
        clone = first.clone()
        transition = clone.step("north")
        self.assertFalse(transition["collision"])
        self.assertEqual((first.steps, clone.steps), (0, 1))

    def test_reverse_excluded_but_collisions_remain_candidates(self):
        game = SnakeGame.from_state(
            state(
                SnakeGame(25, 4, 3),
                body=[[0, 2], [0, 1], [0, 0]],
                direction="east",
                food=[3, 3],
            )
        )
        self.assertEqual(game.legal_actions, ["north", "east", "south"])
        self.assertFalse(game.one_step_safe("north"))
        transition = game.step("north")
        self.assertTrue(transition["collision"])
        self.assertEqual(game.outcome, "wall_collision")

    def test_vacated_tail_is_legal_unless_move_grows(self):
        game = SnakeGame.from_state(
            state(
                SnakeGame(25, 3, 2),
                body=[[1, 1], [1, 0], [0, 0], [0, 1]],
                direction="east",
                food=[2, 2],
            )
        )
        self.assertEqual(game.destination("north"), (0, 1))
        self.assertTrue(game.one_step_safe("north"))
        game.step("north")
        self.assertEqual(game.body[0], (0, 1))
        self.assertFalse(game.terminal)

    def test_food_growth_and_full_board_win(self):
        game = SnakeGame.from_state(
            state(
                SnakeGame(25, 2, 1),
                body=[[0, 0], [1, 0], [1, 1]],
                direction="north",
                food=[0, 1],
            )
        )
        result = game.step("east")
        self.assertEqual(result, {"ate_food": True, "collision": False, "win": True})
        self.assertEqual(game.score, 1)
        self.assertTrue(game.terminal)
        self.assertIsNone(game.food)

    def test_step_rejects_reverse_and_does_not_hide_invalid_state(self):
        game = SnakeGame(25, 8, 3)
        with self.assertRaises(ValueError):
            game.step("west")
        broken = copy.deepcopy(game.full_state())
        broken["body"][1] = broken["body"][0]
        with self.assertRaises(ValueError):
            SnakeGame.from_state(broken)
