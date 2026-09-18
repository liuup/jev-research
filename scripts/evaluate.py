import argparse
from jev2048.trainer import require_slurm,load_checkpoint,evaluate_run
from jev2048.model import load_tokenizer
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--run",required=True); a=p.parse_args(); require_slurm()
    m,s=load_checkpoint(a.run+"/checkpoint.pt"); c=s["config"]
    evaluate_run(m,load_tokenizer(c["base_model"]),c,a.run)
