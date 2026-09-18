"""CPU audit of a completed online GPU smoke."""

import argparse
import json

from jev2048.utils import save_json
from jev2048.verification import audit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/online_smoke")
    args = parser.parse_args()
    report = audit(args.run, require_gpu=True)
    save_json("results/online_smoke.json", report)
    print(json.dumps(report, indent=2))
