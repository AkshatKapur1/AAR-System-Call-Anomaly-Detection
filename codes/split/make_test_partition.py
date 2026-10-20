"""
Partitions split/test.jsonl.gz into a TUNE part (30%) and a FINAL part (70%),
stratified by (source, label, attack_type), so every dataset and attack type
appears in both.

Why: val holds benign data only, so it can set thresholds but cannot tell us
which scoring/aggregation/model choice catches attacks better. Those choices
are made on TUNE only; FINAL is touched once, for the reported results.

split/test.jsonl.gz itself is unchanged — the partition is a list of record
indices (line numbers) written to split/test_partition.json.
"""
import gzip
import json
import os
import random
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEST_PATH = os.path.join(REPO_ROOT, "split", "test.jsonl.gz")
OUT_PATH = os.path.join(REPO_ROOT, "split", "test_partition.json")
TUNE_FRACTION = 0.30
SEED = 42

strata = defaultdict(list)
with gzip.open(TEST_PATH, "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        strata[(r["source"], r["label"], r["attack_type"] or "")].append(i)

tune, final = [], []
for key in sorted(strata):
    idx = strata[key]
    random.Random(SEED).shuffle(idx)
    n_tune = max(1, round(TUNE_FRACTION * len(idx))) if len(idx) > 1 else 0
    tune.extend(idx[:n_tune])
    final.extend(idx[n_tune:])

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump({"seed": SEED, "tune_fraction": TUNE_FRACTION,
               "stratified_by": ["source", "label", "attack_type"],
               "tune": sorted(tune), "final": sorted(final)}, f)
print(f"tune: {len(tune)} records, final: {len(final)} records -> {OUT_PATH}")
