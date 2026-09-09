#!/usr/bin/env python3
import json
from pathlib import Path

PLAID_BASE = Path("/home/aahav/plaid/data/PLAID")
OUT_PATH = Path("/home/aahav/ids_pipeline/output/plaid_parsed.jsonl")


def make_record(source_split, recording_id, process, label, attack_type, syscalls):
    return {
        "source": "PLAID",
        "source_split": source_split,
        "recording_id": recording_id,
        "process": process,
        "label": label,
        "attack_type": attack_type,
        "attack_subtype": None,
        "syscalls": syscalls,
        "seq_len": len(syscalls),
        "decode_errors": 0,
    }


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_records = 0
    n_benign = 0
    n_attack = 0
    with open(OUT_PATH, "w") as out:
        for app_dir in sorted((PLAID_BASE / "baseline").iterdir()):
            if not app_dir.is_dir():
                continue
            for f in sorted(app_dir.glob("*.txt")):
                syscalls = f.read_text(errors="ignore").split()
                recording_id = f"{app_dir.name}/{f.stem}"
                rec = make_record(f"baseline/{app_dir.name}", recording_id, app_dir.name, "benign", None, syscalls)
                out.write(json.dumps(rec) + "\n")
                n_records += 1
                n_benign += 1
        for trial_dir in sorted((PLAID_BASE / "attack").iterdir()):
            if not trial_dir.is_dir():
                continue
            attack_type = trial_dir.name.rsplit("_", 1)[0]
            for f in sorted(trial_dir.glob("*.txt")):
                syscalls = f.read_text(errors="ignore").split()
                recording_id = f"{trial_dir.name}/{f.stem}"
                rec = make_record(f"attack/{trial_dir.name}", recording_id, attack_type, "attack", attack_type, syscalls)
                out.write(json.dumps(rec) + "\n")
                n_records += 1
                n_attack += 1
    total_tokens = 0
    with open(OUT_PATH) as f:
        for line in f:
            r = json.loads(line)
            total_tokens += r["seq_len"]
    print(f"PLAID parsed: {n_records} records ({n_benign} benign, {n_attack} attack)")
    print(f"total tokens: {total_tokens}")
    print(f"output: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
