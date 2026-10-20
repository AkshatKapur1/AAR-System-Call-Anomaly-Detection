"""
Layer 3 completion: score the train, val and test splits with the three
trained models (HMM, GRU, LSTM) and write one anomaly score per recording.

Input : split/{train,val,test}.jsonl.gz (token_ids already encoded with
        split/vocab.json), plus the trained models under models/.
Output: scoring/{hmm,gru,lstm}_{train,val,test}_record_scores.csv and
        scoring/scoring_report.json.

Train scores are for comparing against val (overfitting check) only — the
models were fit on samples of these windows, so they are not detection results.

Windowing is identical to codes/feature_extraction/extract_features.py
(n=7 sliding window per record, never crossing a record boundary) — applied
to split/train.jsonl.gz it reproduces all 8,771,179 training windows.

Score direction: HIGHER = more anomalous, for all three models.
  - HMM      : -log P(window)  (negated log-likelihood over the 7 syscalls)
  - GRU/LSTM : mean NLL of predicting syscalls 2..7 from the ones before them
               (same computation as per_window_nll() in the training scripts)

No thresholds and no use of labels here — this layer only produces scores.
Labels are copied through to the CSVs for Layer 4 (evaluation).

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
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "codes", "feature_extraction"))
from ngram_utils import sliding_windows  # noqa: E402

SPLIT_DIR = os.path.join(REPO_ROOT, "split")
VOCAB_PATH = os.path.join(SPLIT_DIR, "vocab.json")
TRAIN_WINDOWS_PATH = os.path.join(REPO_ROOT, "feature_extraction", "train_windows_n7.jsonl.gz")
OUT_DIR = os.path.join(REPO_ROOT, "scoring")

MODEL_PATHS = {
    "hmm": os.path.join(REPO_ROOT, "models", "HMM", "hmm_model.pkl"),
    "gru": os.path.join(REPO_ROOT, "models", "gru", "results", "gru_model.pt"),
    "lstm": os.path.join(REPO_ROOT, "models", "lstm", "results", "lstm_model.pt"),
}
# Existing held-out score CSVs written at training time — used only to check
# that the models are loaded and scored exactly as they were in training.
HELD_OUT_CSVS = {
    "hmm": os.path.join(REPO_ROOT, "models", "HMM", "held_out_window_scores_corrected.csv"),
    "gru": os.path.join(REPO_ROOT, "models", "gru", "results", "held_out_window_scores.csv"),
    "lstm": os.path.join(REPO_ROOT, "models", "lstm", "results", "held_out_window_scores.csv"),
}
# Sampling settings copied from the training scripts, so the exact same
# held-out windows can be rebuilt from train_windows_n7.jsonl.gz.
TOTAL_TRAIN_WINDOWS = 8_771_179
TRAIN_SAMPLE_SIZES = {"hmm": 200_000, "gru": 500_000, "lstm": 500_000}
HELD_OUT_SAMPLE_SIZE = 50_000

N = 7
TOP_FRACTION = 0.05      # primary per-record score = mean of the top 5% windows
UNK_ID = 1
BATCH_SIZE = 4096
VERIFY_TOLERANCE = 1e-3


# ---------------------------------------------------------------- models

class SyscallGRU(nn.Module):
    """Same definition as models/gru/code/train_gru.py."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)
        return self.output(gru_out)


class SyscallLSTM(nn.Module):
    """Same definition as models/lstm/code/train_lstm.py."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.output(lstm_out)


def load_torch_model(name, cls):
    ckpt = torch.load(MODEL_PATHS[name], map_location="cpu")
    model = cls(ckpt["vocab_size"], ckpt["embed_dim"], ckpt["hidden_dim"], ckpt["num_layers"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["vocab_size"]


def hmm_window_scores(model, windows):
    """Batched forward algorithm in log space: returns -log P(window) for
    each (n_windows, 7) row. Equivalent to -model.score(window) per window,
    vectorised so millions of windows score in seconds instead of hours."""
    log_start = np.log(model.startprob_)
    log_trans = np.log(model.transmat_)
    log_emit = np.log(model.emissionprob_)            # (states, symbols)
    out = np.empty(len(windows))
    for start in range(0, len(windows), 200_000):
        w = windows[start:start + 200_000]
        alpha = log_start[None, :] + log_emit[:, w[:, 0]].T
        for t in range(1, w.shape[1]):
            m = alpha[:, :, None] + log_trans[None, :, :]
            mx = m.max(axis=1, keepdims=True)
            alpha = (mx[:, 0, :] + np.log(np.exp(m - mx).sum(axis=1))) + log_emit[:, w[:, t]].T
        mx = alpha.max(axis=1, keepdims=True)
        out[start:start + len(w)] = -(mx[:, 0] + np.log(np.exp(alpha - mx).sum(axis=1)))
    return out


def nn_window_scores(model, windows):
    """Mean NLL of predicting tokens 2..7 from tokens 1..6 — identical to
    per_window_nll() in the GRU/LSTM training scripts."""
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    out = np.empty(len(windows))
    with torch.no_grad():
        for start in range(0, len(windows), BATCH_SIZE):
            batch = torch.from_numpy(windows[start:start + BATCH_SIZE])
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = model(inputs)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            out[start:start + len(batch)] = loss.view(targets.shape).mean(dim=1).numpy()
    return out


# ---------------------------------------------------------------- data

def load_split(name):
    records = []
    with gzip.open(os.path.join(SPLIT_DIR, f"{name}.jsonl.gz"), "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            records.append({
                "record_index": i,
                "recording_id": r["recording_id"],
                "source": r["source"],
                "source_split": r["source_split"],
                "label": r["label"],
                "attack_type": r["attack_type"] or "",
                "token_ids": r["token_ids"],
            })
    return records


def build_windows(records):
    """All n=7 windows of every record, stacked, plus per-record offsets."""
    chunks, offsets = [], [0]
    for r in records:
        w = [list(ctx) + [tgt] for ctx, tgt in sliding_windows(r["token_ids"], N)]
        if w:
            chunks.append(np.asarray(w, dtype=np.int64))
        offsets.append(offsets[-1] + len(w))
    windows = np.concatenate(chunks) if chunks else np.empty((0, N), dtype=np.int64)
    return windows, offsets


def aggregate(window_scores, offsets, i):
    s = window_scores[offsets[i]:offsets[i + 1]]
    if len(s) == 0:
        return float("nan"), float("nan"), float("nan")
    k = max(1, math.ceil(TOP_FRACTION * len(s)))
    top = np.partition(s, len(s) - k)[len(s) - k:]
    return float(s.mean()), float(s.max()), float(top.mean())


def write_record_csv(path, records, offsets, window_scores):
    fields = ["record_index", "recording_id", "source", "source_split", "label",
              "attack_type", "n_syscalls", "n_windows", "unk_rate",
              "score_mean", "score_max", "score_top5", "score"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i, r in enumerate(records):
            mean, mx, top5 = aggregate(window_scores, offsets, i)
            n_tok = len(r["token_ids"])
            unk = sum(t == UNK_ID for t in r["token_ids"]) / n_tok if n_tok else 0.0
            w.writerow([r["record_index"], r["recording_id"], r["source"], r["source_split"],
                        r["label"], r["attack_type"], n_tok, offsets[i + 1] - offsets[i],
                        round(unk, 6), mean, mx, top5, top5])


# ---------------------------------------------------------------- verification

def rebuild_held_out(train_n):
    """Same single-pass systematic sampling as the training scripts; returns
    only the held-out windows (the ones the stored CSVs were computed on)."""
    stride = TOTAL_TRAIN_WINDOWS // train_n
    held_out_stride = TOTAL_TRAIN_WINDOWS // HELD_OUT_SAMPLE_SIZE
    held_out, n_train = [], 0
    with gzip.open(TRAIN_WINDOWS_PATH, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % stride == 0 and n_train < train_n:
                n_train += 1
            elif i % held_out_stride == 1 and len(held_out) < HELD_OUT_SAMPLE_SIZE:
                held_out.append(json.loads(line))
    return np.array(held_out, dtype=np.int64)


def verify_against_training(name, scorer):
    with open(HELD_OUT_CSVS[name], encoding="utf-8") as f:
        stored = np.array([float(row[1]) for row in list(csv.reader(f))[1:]])
    if name == "hmm":
        stored = -stored      # stored as log-likelihood; we score -log-likelihood
    windows = rebuild_held_out(TRAIN_SAMPLE_SIZES[name])
    ours = scorer(windows)
    n = min(len(ours), len(stored))
    diff = float(np.abs(ours[:n] - stored[:n]).max())
    result = {
        "held_out_windows_rebuilt": int(len(windows)),
        "stored_scores": int(len(stored)),
        "max_abs_diff": diff,
        "passed": bool(len(ours) == len(stored) and diff < VERIFY_TOLERANCE),
    }
    print(f"  verify {name}: {result}")
    return result


# ---------------------------------------------------------------- main

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
    with open(VOCAB_PATH, encoding="utf-8") as f:
        vocab_size = len(json.load(f))

    hmm_model = joblib.load(MODEL_PATHS["hmm"])
    gru_model, gru_vocab = load_torch_model("gru", SyscallGRU)
    lstm_model, lstm_vocab = load_torch_model("lstm", SyscallLSTM)
    assert hmm_model.emissionprob_.shape[1] == gru_vocab == lstm_vocab == vocab_size, "vocab size mismatch"

    scorers = {
        "hmm": lambda w: hmm_window_scores(hmm_model, w),
        "gru": lambda w: nn_window_scores(gru_model, w),
        "lstm": lambda w: nn_window_scores(lstm_model, w),
    }
    n_params = {
        "hmm": int(hmm_model.startprob_.size + hmm_model.transmat_.size + hmm_model.emissionprob_.size),
        "gru": sum(p.numel() for p in gru_model.parameters()),
        "lstm": sum(p.numel() for p in lstm_model.parameters()),
    }

    print("Verifying models reproduce the stored training-time held-out scores...")
    verification = {name: verify_against_training(name, fn) for name, fn in scorers.items()}
    if not all(v["passed"] for v in verification.values()):
        sys.exit("Verification failed — models are not being scored as they were in training.")

    report = {
        "n": N,
        "vocab_size": vocab_size,
        "score_direction": "higher = more anomalous (all models)",
        "window_score": {
            "hmm": "-log P(7-syscall window) via forward algorithm",
            "gru": "mean NLL of predicting syscalls 2..7 within the window",
            "lstm": "mean NLL of predicting syscalls 2..7 within the window",
        },
        "record_score": f"score = score_top5 = mean of the top {int(TOP_FRACTION * 100)}% "
                        "most anomalous windows in the record (fixed in advance, not tuned)",
        "verification_against_training_held_out": verification,
        "splits": {},
        "models": {name: {"model_file": os.path.relpath(MODEL_PATHS[name], REPO_ROOT).replace("\\", "/"),
                          "model_file_kb": round(os.path.getsize(MODEL_PATHS[name]) / 1024, 1),
                          "n_parameters": n_params[name],
                          "inference": {}} for name in scorers},
    }

    for split in ("train", "val", "test"):
        print(f"Loading split/{split}.jsonl.gz ...")
        records = load_split(split)
        windows, offsets = build_windows(records)
        n_tokens = {lab: sum(len(r["token_ids"]) for r in records if r["label"] == lab) for lab in ("benign", "attack")}
        n_unk = {lab: sum(t == UNK_ID for r in records if r["label"] == lab for t in r["token_ids"]) for lab in ("benign", "attack")}
        no_windows = [r["record_index"] for i, r in enumerate(records) if offsets[i + 1] == offsets[i]]
        report["splits"][split] = {
            "records": len(records),
            "records_by_label": {lab: sum(r["label"] == lab for r in records) for lab in ("benign", "attack")},
            "windows": int(len(windows)),
            "records_with_no_windows (<7 syscalls, score=NaN)": no_windows,
            "unk_rate": {lab: (n_unk[lab] / n_tokens[lab] if n_tokens[lab] else None) for lab in n_tokens},
        }
        print(f"  {len(records)} records -> {len(windows)} windows")

        for name, scorer in scorers.items():
            t0 = time.time()
            window_scores = scorer(windows)
            elapsed = time.time() - t0
            assert np.isfinite(window_scores).all(), f"{name}: non-finite window scores"
            out_path = os.path.join(OUT_DIR, f"{name}_{split}_record_scores.csv")
            write_record_csv(out_path, records, offsets, window_scores)
            report["models"][name]["inference"][split] = {
                "windows": int(len(windows)),
                "seconds": round(elapsed, 2),
                "ms_per_1000_windows": round(1000 * elapsed / max(1, len(windows)) * 1000, 3),
            }
            print(f"  {name}: scored in {elapsed:.1f}s -> {os.path.relpath(out_path, REPO_ROOT)}")

    report["peak_memory_mb"] = peak_memory_mb()
    report["device"] = "cpu"
    report["torch_threads"] = torch.get_num_threads()
    report_path = os.path.join(OUT_DIR, "scoring_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report -> {os.path.relpath(report_path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
