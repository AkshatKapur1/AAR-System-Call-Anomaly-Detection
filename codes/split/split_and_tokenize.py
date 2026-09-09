import json
import random
from pathlib import Path

IN_PATH = "/home/aahav/parser/output/capped/combined_training_data.jsonl"
OUT_DIR = Path("/home/aahav/parser/output/split")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


records = list(load_jsonl(IN_PATH))
benign = [r for r in records if r["label"] == "benign"]
attack = [r for r in records if r["label"] == "attack"]

train, val, test = [], [], []

for source in sorted(set(r["source"].split("/")[0] for r in benign)):
    group = [r for r in benign if r["source"].split("/")[0] == source]
    random.Random(SEED).shuffle(group)
    n = len(group)
    n_train = round(0.70 * n)
    n_val = round(0.15 * n)
    train.extend(group[:n_train])
    val.extend(group[n_train:n_train + n_val])
    test.extend(group[n_train + n_val:])

test.extend(attack)

# vocab is built ONLY from train's benign syscalls
vocab = {"<PAD>": 0, "<UNK>": 1}
syscall_set = set()
for r in train:
    syscall_set.update(r["syscalls"])
for name in sorted(syscall_set):
    vocab[name] = len(vocab)


def tokenize(seq):
    unk = vocab["<UNK>"]
    return [vocab.get(s, unk) for s in seq]


splits = {"train": train, "val": val, "test": test}
for split_name, split_records in splits.items():
    out = []
    for r in split_records:
        r = dict(r)
        r["token_ids"] = tokenize(r["syscalls"])
        out.append(r)
    write_jsonl(OUT_DIR / f"{split_name}.jsonl", out)

with open(OUT_DIR / "vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)

print(f"vocab size: {len(vocab)} (incl. <PAD>, <UNK>)\n")
for split_name, split_records in splits.items():
    by_source_label = {}
    for r in split_records:
        key = (r["source"].split("/")[0], r["label"])
        by_source_label[key] = by_source_label.get(key, 0) + 1
    print(f"{split_name:<6} {len(split_records):>6} records  {by_source_label}")

print()
for split_name in ("val", "test"):
    split_records = splits[split_name]
    total = sum(len(r["syscalls"]) for r in split_records)
    unk = sum(t == 1 for r in split_records for t in tokenize(r["syscalls"]))
    print(f"{split_name}: UNK rate = {unk}/{total} = {unk / total:.4%}")

# UNK rate broken out separately for benign vs attack inside test
attack_in_test = [r for r in splits["test"] if r["label"] == "attack"]
benign_in_test = [r for r in splits["test"] if r["label"] == "benign"]
for label, group in [("test/benign", benign_in_test), ("test/attack", attack_in_test)]:
    total = sum(len(r["syscalls"]) for r in group)
    unk = sum(t == 1 for r in group for t in tokenize(r["syscalls"]))
    print(f"{label}: UNK rate = {unk}/{total} = {unk / total:.4%}" if total else f"{label}: n/a")

print(f"\nOutput dir: {OUT_DIR}")
