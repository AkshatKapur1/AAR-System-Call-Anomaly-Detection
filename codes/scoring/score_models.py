"""
Layer 3: score the train, val and test splits with the three trained models
(HMM, GRU, LSTM — v2) and write per-recording anomaly scores.

Input : split/{train,val,test}.jsonl.gz and the models under models/.
Output: scoring/{hmm,gru,lstm}_{train,val,test}_record_scores.csv and
        scoring/scoring_report.json.

Every recording is cut into n=7 sliding windows (never crossing a record
boundary — on split/train this reproduces the 8,771,179 training windows).
Each window is scored in two modes; higher = more anomalous for all models:
  - window7: the model sees only the 7 syscalls of the window
      HMM      -log P(window)
      GRU/LSTM mean NLL of syscalls 2..7 given the earlier ones in the window
  - stream : the model reads the recording from the start, and the window's
             score is the mean NLL of its syscalls 2..7 given EVERYTHING
             before them (longer memory).
Window scores are aggregated per recording as mean / max / top-1% / top-5% /
top-10% mean. Layer 4 picks one (mode, aggregation) per model on the TUNE
part of test; this layer makes no choices, sets no thresholds and does not
use labels (they are copied through for Layer 4).

Run from anywhere:  python codes/scoring/score_models.py
"""
import csv
import gzip
import json
import math
import os
import sys
import time

import joblib
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "codes", "feature_extraction"))
sys.path.insert(0, os.path.join(REPO_ROOT, "models"))
sys.path.insert(0, os.path.join(REPO_ROOT, "models", "HMM", "Codes"))
from ngram_utils import sliding_windows  # noqa: E402
import sequence_model_common as rnn  # noqa: E402
import train_hmm as hmm_v2  # noqa: E402

SPLIT_DIR = os.path.join(REPO_ROOT, "split")
OUT_DIR = os.path.join(REPO_ROOT, "scoring")
MODEL_PATHS = {
    "hmm": os.path.join(REPO_ROOT, "models", "HMM", "hmm_model.pkl"),
    "gru": os.path.join(REPO_ROOT, "models", "gru", "results", "gru_model.pt"),
    "lstm": os.path.join(REPO_ROOT, "models", "lstm", "results", "lstm_model.pt"),
}
N = 7
MODES = ["window7", "stream"]
AGGREGATIONS = {"mean": None, "max": None, "top1": 0.01, "top5": 0.05, "top10": 0.10}
UNK_ID = 1
CHECK_TOLERANCE = 1e-3


def load_split(name):
    records = []
    with gzip.open(os.path.join(SPLIT_DIR, f"{name}.jsonl.gz"), "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            records.append({k: r[k] for k in ("recording_id", "source", "source_split", "label", "token_ids")}
                           | {"record_index": i, "attack_type": r["attack_type"] or ""})
    return records


def build_windows(records):
    chunks, offsets = [], [0]
    for r in records:
        w = [list(ctx) + [tgt] for ctx, tgt in sliding_windows(r["token_ids"], N)]
        if w:
            chunks.append(np.asarray(w, dtype=np.int64))
        offsets.append(offsets[-1] + len(w))
    return (np.concatenate(chunks) if chunks else np.empty((0, N), dtype=np.int64)), offsets


def aggregate(s):
    if len(s) == 0:
        return {a: float("nan") for a in AGGREGATIONS}
    out = {"mean": float(s.mean()), "max": float(s.max())}
    for name, frac in AGGREGATIONS.items():
        if frac:
            k = max(1, math.ceil(frac * len(s)))
            out[name] = float(np.partition(s, len(s) - k)[len(s) - k:].mean())
    return out


def score_split(name, model, records, windows, offsets):
    """Returns {mode: list of per-record window-score arrays}, timings."""
    seqs = [r["token_ids"] for r in records]
    timings = {}
    t0 = time.time()
    w7 = hmm_v2.window7_scores(model, windows) if name == "hmm" else rnn.window7_scores(model, windows)
    timings["window7"] = time.time() - t0
    t0 = time.time()
    token_nll = (hmm_v2.token_nll_stream(model, seqs) if name == "hmm"
                 else rnn.token_nll_stream(model, seqs))
    stream = [rnn.stream_window_scores(t) for t in token_nll]
    timings["stream"] = time.time() - t0
    per_record = {"window7": [w7[offsets[i]:offsets[i + 1]] for i in range(len(records))],
                  "stream": stream}
    return per_record, timings, token_nll


def consistency_check(name, model, records, per_record, token_nll):
    """The first window of a record has no earlier context, so window7 and
    stream must agree on it (HMM: window7 = -log P(s1) + 6 * stream)."""
    diffs = []
    for i, r in enumerate(records):
        if len(per_record["window7"][i]) == 0:
            continue
        w7, st = per_record["window7"][i][0], per_record["stream"][i][0]
        if name == "hmm":
            s1 = r["token_ids"][0]
            first = -math.log(float(model.startprob_ @ model.emissionprob_[:, s1]))
            diffs.append(abs(w7 - (first + (N - 1) * st)))
        else:
            diffs.append(abs(w7 - st))
    return float(max(diffs))


def peak_memory_mb():
    """Peak resident memory of this process, without extra dependencies."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if not get_info(get_process(), ctypes.byref(counters), counters.cb):
            return None
        return round(counters.PeakWorkingSetSize / 1e6, 1)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1e6 if sys.platform == "darwin" else 1e3), 1)  # bytes on macOS, KB on Linux


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    models = {"hmm": joblib.load(MODEL_PATHS["hmm"]),
              "gru": rnn.load_model(MODEL_PATHS["gru"]),
              "lstm": rnn.load_model(MODEL_PATHS["lstm"])}
    with open(os.path.join(SPLIT_DIR, "vocab.json"), encoding="utf-8") as f:
        vocab_size = len(json.load(f))
    assert models["hmm"].emissionprob_.shape[1] == vocab_size
    n_params = {"hmm": int(models["hmm"].startprob_.size + models["hmm"].transmat_.size
                           + models["hmm"].emissionprob_.size)}
    n_params.update({m: sum(p.numel() for p in models[m].parameters()) for m in ("gru", "lstm")})

    report = {
        "version": "v2", "n": N, "vocab_size": vocab_size,
        "score_direction": "higher = more anomalous (all models, all columns)",
        "modes": {"window7": "window scored in isolation",
                  "stream": "window scored with the whole preceding recording as context"},
        "aggregations": {"mean": "mean window score", "max": "most anomalous window",
                         "top1/top5/top10": "mean of the 1% / 5% / 10% most anomalous windows"},
        "splits": {}, "models": {m: {"model_file": os.path.relpath(p, REPO_ROOT).replace("\\", "/"),
                                     "model_file_kb": round(os.path.getsize(p) / 1024, 1),
                                     "n_parameters": n_params[m], "inference": {},
                                     "consistency_check_max_abs_diff": None}
                                 for m, p in MODEL_PATHS.items()},
    }
    columns = [f"{mode}_{agg}" for mode in MODES for agg in AGGREGATIONS]

    for split in ("train", "val", "test"):
        print(f"Loading split/{split}.jsonl.gz ...")
        records = load_split(split)
        windows, offsets = build_windows(records)
        labels = ("benign", "attack")
        n_tok = {l: sum(len(r["token_ids"]) for r in records if r["label"] == l) for l in labels}
        n_unk = {l: sum(t == UNK_ID for r in records if r["label"] == l for t in r["token_ids"]) for l in labels}
        report["splits"][split] = {
            "records": len(records),
            "records_by_label": {l: sum(r["label"] == l for r in records) for l in labels},
            "windows": int(len(windows)),
            "records_with_no_windows (<7 syscalls, scores=NaN)":
                [r["record_index"] for i, r in enumerate(records) if offsets[i + 1] == offsets[i]],
            "unk_rate": {l: (n_unk[l] / n_tok[l] if n_tok[l] else None) for l in labels},
        }
        print(f"  {len(records)} records -> {len(windows)} windows")

        for name, model in models.items():
            per_record, timings, token_nll = score_split(name, model, records, windows, offsets)
            if split == "val":
                diff = consistency_check(name, model, records, per_record, token_nll)
                report["models"][name]["consistency_check_max_abs_diff"] = diff
                assert diff < CHECK_TOLERANCE, f"{name}: window7/stream mismatch {diff}"
            for mode in MODES:
                assert all(np.isfinite(s).all() for s in per_record[mode]), f"{name}/{mode}: non-finite"
            report["models"][name]["inference"][split] = {
                "windows": int(len(windows)),
                **{f"{mode}_seconds": round(t, 2) for mode, t in timings.items()},
                **{f"{mode}_ms_per_1000_windows": round(1e6 * t / max(1, len(windows)), 3)
                   for mode, t in timings.items()},
            }
            path = os.path.join(OUT_DIR, f"{name}_{split}_record_scores.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["record_index", "recording_id", "source", "source_split", "label", "attack_type",
                            "n_syscalls", "n_windows", "unk_rate"] + columns)
                for i, r in enumerate(records):
                    aggs = {mode: aggregate(per_record[mode][i]) for mode in MODES}
                    n = len(r["token_ids"])
                    w.writerow([r["record_index"], r["recording_id"], r["source"], r["source_split"], r["label"],
                                r["attack_type"], n, offsets[i + 1] - offsets[i],
                                round(sum(t == UNK_ID for t in r["token_ids"]) / n, 6) if n else 0.0]
                               + [aggs[mode][agg] for mode in MODES for agg in AGGREGATIONS])
            print(f"  {name}: window7 {timings['window7']:.1f}s, stream {timings['stream']:.1f}s "
                  f"-> {os.path.relpath(path, REPO_ROOT)}")

    report["peak_memory_mb"] = peak_memory_mb()
    report["torch_threads"] = torch.get_num_threads()
    with open(os.path.join(OUT_DIR, "scoring_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Saved report -> scoring/scoring_report.json")


if __name__ == "__main__":
    main()
