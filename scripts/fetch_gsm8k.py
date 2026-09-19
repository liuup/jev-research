"""Download the two official GSM8K splits and record their provenance."""

import argparse
import json
import urllib.request
from pathlib import Path

from jevmath.gsm8k import TEST_URL, TRAIN_URL, final_answer, question_numbers
from jevmath.utils import digest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/gsm8k")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, url in (("train", TRAIN_URL), ("test", TEST_URL)):
        path = root / f"{name}.jsonl"
        if path.exists() and not args.force:
            raise FileExistsError(f"{path} exists; pass --force to replace it")
        with urllib.request.urlopen(url) as response:
            path.write_bytes(response.read())
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        usable = unparsed = 0
        for row in rows:
            try:
                if question_numbers(row["question"]) and final_answer(row["answer"]):
                    usable += 1
            except ValueError:
                unparsed += 1
        counts[name] = dict(rows=len(rows), usable=usable, unparsed=unparsed, sha256=digest(path))
        print(name, counts[name], flush=True)
    (root / "manifest.json").write_text(
        json.dumps(dict(urls=dict(train=TRAIN_URL, test=TEST_URL), splits=counts), indent=2)
        + "\n"
    )
