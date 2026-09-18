"""Deterministic one-ply afterstate policy; ties use LEFT, RIGHT, UP, DOWN."""
import math
import yaml
from .env import moved

class HeuristicPolicy:
    def __init__(self, config="configs/heuristic.yaml"):
        with open(config) as f:
            self.weights = yaml.safe_load(f)

    def value(self, board, gain):
        b = [[math.log2(x) if x else 0 for x in row] for row in board]
        lines = b + list(zip(*b))
        mono = sum(max(sum(min(line[i+1]-line[i], 0) for i in range(3)),
                       sum(min(line[i]-line[i+1], 0) for i in range(3))) for line in lines)
        smooth = -sum(abs(line[i]-line[i+1]) for line in lines for i in range(3) if line[i] and line[i+1])
        merges = sum(line[i] == line[i+1] and line[i] > 0 for line in lines for i in range(3))
        maximum = max(map(max,b))
        corner = max(b[0][0], b[0][3], b[3][0], b[3][3])
        features = dict(empty=sum(x==0 for row in b for x in row), monotonicity=mono,
                        smoothness=smooth, merges=merges, corner=corner, maximum=maximum, immediate=gain)
        return sum(self.weights[k]*v for k,v in features.items())

    def choose(self, game):
        legal = game.legal_actions
        if not legal:
            raise ValueError("Terminal board")
        return max(legal, key=lambda a: self.value(*moved(game.board,a)))
