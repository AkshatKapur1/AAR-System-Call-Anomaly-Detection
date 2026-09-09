#!/usr/bin/env python3
import json
from pathlib import Path

ADFA_BASE = Path("/home/aahav/ADFA-LD")
SYSCALL_TABLE = Path("/home/aahav/ids_pipeline/syscalls-i386.txt")
OUT_PATH = Path("/home/aahav/ids_pipeline/output/adfa_parsed.jsonl")


def load_mapping():
    mapping = {}
    with open(SYSCALL_TABLE) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                name, code = parts[0], parts[1]
                try:
                    mapping[int(code)] = name
                except ValueError:
                    continue
    return mapping


def decode_trace(raw_tokens, mapping):
    tokens = []
    errors = 0
    for tok in raw_tokens:
        try:
            code = int(tok)
        except ValueError:
            tokens.append("UNK")
            errors += 1
            continue
        name = mapping.get(code)
        if name is None:
            tokens.append("UNK")
            errors += 1
        else:
            tokens.append(name)
    return tokens, errors


def make_record(source_split, recording_id, label, attack_type, syscalls, decode_errors):
    return {
        "source": "ADFA-LD",
        "source_split": source_split,
        "recording_id": recording_id,
        "process": None,
        "label": label,
        "attack_type": attack_type,
        "attack_subtype": None,
        "syscalls": syscalls,
        "seq_len": len(syscalls),
        "decode_errors": decode_errors,
    }


def main():
    mapping = load_mapping()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_records = 0
    n_benign = 0
    n_attack = 0
    with open(OUT_PATH, "w") as out:
        for split in ["Training_Data_Master", "Validation_Data_Master"]:
            for f in sorted((ADFA_BASE / split).glob("*.txt")):
                raw = f.read_text(errors="ignore").split()
                syscalls, errors = decode_trace(raw, mapping)
                rec = make_record(split, f.stem, "benign", None, syscalls, errors)
                out.write(json.dumps(rec) + "\n")
                n_records += 1
                n_benign += 1
        for cat_dir in sorted((ADFA_BASE / "Attack_Data_Master").iterdir()):
            if not cat_dir.is_dir():
                continue
            attack_type = cat_dir.name.rsplit("_", 1)[0]
            for f in sorted(cat_dir.glob("*.txt")):
                raw = f.read_text(errors="ignore").split()
                syscalls, errors = decode_trace(raw, mapping)
                rec = make_record(f"Attack_Data_Master/{cat_dir.name}", f.stem, "attack", attack_type, syscalls, errors)
                out.write(json.dumps(rec) + "\n")
                n_records += 1
                n_attack += 1
    total_tokens = 0
    total_errors = 0
    with open(OUT_PATH) as f:
        for line in f:
            r = json.loads(line)
            total_tokens += r["seq_len"]
            total_errors += r["decode_errors"]
    print(f"ADFA parsed: {n_records} records ({n_benign} benign, {n_attack} attack)")
    print(f"total tokens: {total_tokens}, total decode_errors: {total_errors}")
    print(f"output: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
