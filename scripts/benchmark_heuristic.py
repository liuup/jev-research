import argparse
from concurrent.futures import ProcessPoolExecutor

from jev2048.env import Game
from jev2048.heuristic import HeuristicPolicy
from jev2048.utils import digest, game_summary, save_json


def play(seed):
    g, p = Game(seed), HeuristicPolicy()
    while not g.terminal:
        g.step(p.choose(g))
    return g


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    with ProcessPoolExecutor(a.workers) as pool:
        games = list(pool.map(play, range(a.games)))
    report = game_summary(games)
    report.update(
        seeds=list(range(a.games)), policy_sha256=digest("configs/heuristic.yaml")
    )
    save_json("results/heuristic_benchmark.json", report)
    print(report)
