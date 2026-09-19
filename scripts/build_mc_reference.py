import argparse

from jev2048.dataset import build_reference

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/offline_pi0_v1")
    p.add_argument("--states", type=int, default=64)
    p.add_argument("--rollouts", type=int, default=256)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=717)
    a = p.parse_args()
    build_reference(a.data_dir, a.states, a.rollouts, a.workers, a.seed)
