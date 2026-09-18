import argparse
import json
from pathlib import Path

from jev2048.model import load_tokenizer
from jev2048.online_trainer import reevaluate_online
from jev2048.trainer import evaluate_run, load_checkpoint, require_slurm

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    a = p.parse_args()
    require_slurm()
    if (Path(a.run) / "target_policy.json").exists():
        print(json.dumps(reevaluate_online(a.run), indent=2))
        raise SystemExit(0)
    m, s = load_checkpoint(a.run + "/checkpoint.pt")
    c = s["config"]
    del s
    print(json.dumps(c, indent=2), flush=True)
    evaluate_run(m, load_tokenizer(c["base_model"]), c, a.run)
