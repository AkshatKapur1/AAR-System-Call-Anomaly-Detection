import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ngram_utils import load_jsonl, sliding_windows

N = 7
IN_PATH = "/home/aahav/parser/output/split/train.jsonl"
OUT_PATH = "/home/aahav/parser/feature_extraction/train_windows_n7.jsonl"

records = list(load_jsonl(IN_PATH))

window_count = 0
with open(OUT_PATH, "w") as out:
    for r in records:
        for context, target in sliding_windows(r["token_ids"], N):
            out.write(json.dumps(list(context) + [target]) + "\n")
            window_count += 1

size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
print(f"input:  {IN_PATH} ({len(records)} records)")
print(f"n:      {N}")
print(f"output: {OUT_PATH}")
print(f"windows written: {window_count}")
print(f"file size: {size_mb:.1f} MB")
