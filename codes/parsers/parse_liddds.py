#!/usr/bin/env python3
import json
import zipfile
from pathlib import Path

BASE = Path("/home/aahav/LID-DS")
OUT_PATH = Path("/home/aahav/ids_pipeline/output/liddds_parsed.jsonl")
MIN_LEN = 10

CVES = ["CVE-2012-2122", "CVE-2014-0160"]
SPLITS = ["training", "validation", "test/normal", "test/normal_and_attack"]


def read_zip_member(zf, suffix):
    names = [n for n in zf.namelist() if n.endswith(suffix)]
    if not names:
        return None
    return zf.read(names[0]).decode("utf-8", errors="ignore")


def parse_sc_entries(sc_text):
    events = []
    for line in sc_text.splitlines():
        parts = line.split()
        if len(parts) < 7 or parts[6] != ">":
            continue
        try:
            ts_ns = int(parts[0])
        except ValueError:
            continue
        events.append((ts_ns, parts[3], parts[5]))
    return events


def make_record(cve, source_split, recording_id, process, label, attack_subtype, syscalls):
    return {
        "source": f"LID-DS/{cve}",
        "source_split": source_split,
        "recording_id": recording_id,
        "process": process,
        "label": label,
        "attack_type": cve,
        "attack_subtype": attack_subtype,
        "syscalls": syscalls,
        "seq_len": len(syscalls),
        "decode_errors": 0,
    }


def process_recording(cve, split, zpath, out, stats):
    with zipfile.ZipFile(zpath) as zf:
        sc_text = read_zip_member(zf, ".sc")
        json_text = read_zip_member(zf, ".json")
    if sc_text is None or json_text is None:
        stats["skipped_missing_file"] += 1
        return
    meta = json.loads(json_text)
    events = parse_sc_entries(sc_text)
    if not events:
        stats["skipped_empty"] += 1
        return

    time_info = meta.get("time", {})
    warmup_end = time_info.get("warmup_end", {}).get("absolute")
    warmup_end_ns = int(round(warmup_end * 1e9)) if warmup_end is not None else None

    is_exploit = bool(meta.get("exploit", False))
    exploit_name = meta.get("exploit_name")
    exploit_start_ns = None
    if is_exploit:
        exploit_list = time_info.get("exploit", [])
        if exploit_list:
            exploit_start_ns = int(round(exploit_list[0]["absolute"] * 1e9))

    recording_id = zpath.stem

    by_process = {}
    for ts_ns, proc, syscall in events:
        by_process.setdefault(proc, []).append((ts_ns, syscall))

    for proc, proc_events in by_process.items():
        if warmup_end_ns is not None:
            proc_events = [(t, s) for (t, s) in proc_events if t >= warmup_end_ns]

        if is_exploit and exploit_start_ns is not None:
            benign_part = [s for (t, s) in proc_events if t < exploit_start_ns]
            attack_part = [s for (t, s) in proc_events if t >= exploit_start_ns]

            if len(benign_part) >= MIN_LEN:
                out.write(json.dumps(make_record(
                    cve, f"{split}:pre-exploit", f"{recording_id}/{proc}",
                    proc, "benign", None, benign_part
                )) + "\n")
                stats["benign_records"] += 1
                stats["benign_tokens"] += len(benign_part)
            else:
                stats["dropped_short_fragments"] += 1

            if len(attack_part) >= MIN_LEN:
                out.write(json.dumps(make_record(
                    cve, split, f"{recording_id}/{proc}",
                    proc, "attack", exploit_name, attack_part
                )) + "\n")
                stats["attack_records"] += 1
                stats["attack_tokens"] += len(attack_part)
            else:
                stats["dropped_short_fragments"] += 1
        else:
            seq = [s for (t, s) in proc_events]
            if len(seq) >= MIN_LEN:
                out.write(json.dumps(make_record(
                    cve, split, f"{recording_id}/{proc}",
                    proc, "benign", None, seq
                )) + "\n")
                stats["benign_records"] += 1
                stats["benign_tokens"] += len(seq)
            else:
                stats["dropped_short_fragments"] += 1


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "benign_records": 0, "attack_records": 0,
        "benign_tokens": 0, "attack_tokens": 0,
        "dropped_short_fragments": 0,
        "skipped_missing_file": 0, "skipped_empty": 0,
        "total_recordings": 0,
    }
    with open(OUT_PATH, "w") as out:
        for cve in CVES:
            for split in SPLITS:
                d = BASE / cve / split
                if not d.exists():
                    continue
                for zpath in sorted(d.glob("*.zip")):
                    stats["total_recordings"] += 1
                    process_recording(cve, split, zpath, out, stats)
                print(f"  done: {cve}/{split}")

    print(f"LID-DS parsed from {stats['total_recordings']} raw recordings")
    print(f"benign records: {stats['benign_records']}, tokens: {stats['benign_tokens']}")
    print(f"attack records: {stats['attack_records']}, tokens: {stats['attack_tokens']}")
    print(f"dropped short fragments (<{MIN_LEN} syscalls): {stats['dropped_short_fragments']}")
    print(f"skipped (missing sc/json): {stats['skipped_missing_file']}, skipped (no entry events): {stats['skipped_empty']}")
    print(f"output: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
