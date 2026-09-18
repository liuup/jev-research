import argparse
import yaml
from jev2048.dataset import generate

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config",default="configs/data.yaml"); p.add_argument("--output",default="data/main")
    p.add_argument("--train-questions",type=int); p.add_argument("--heldout-questions",type=int); p.add_argument("--workers",type=int)
    a=p.parse_args(); c=yaml.safe_load(open(a.config))
    if a.train_questions is not None: c["questions"]["train"]=a.train_questions
    if a.heldout_questions is not None:
        for s in ("dev","calibration","test"): c["questions"][s]=a.heldout_questions
    if a.workers: c["workers"]=a.workers
    generate(c,a.output)
