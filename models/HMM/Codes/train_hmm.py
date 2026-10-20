"""
Track A: HMM (v2) — training and scoring.

Changes vs. v1 (200k sampled 7-gram windows, 8 states, unweighted):
  - trains on ALL of split/train.jsonl.gz as full recordings, so transitions
    between consecutive windows are learned too;
  - honours the capping sample_weight by repeating ADFA-LD records
    round(3.04) = 3 times (EM on a repeated sequence == weight 3);
  - number of hidden states chosen from {8, 16, 32} by per-syscall
    log-likelihood on benign val (no attack data used);
  - Laplace smoothing kept (no zero probabilities for rare syscalls).

Scoring functions (used by codes/scoring/score_models.py):
  - window7_scores : -log P(7-syscall window), the window seen in isolation;
  - token_nll_stream: -log P(syscall t | all earlier syscalls of the record),
    via the scaled forward algorithm, for the "stream" mode.

Output: models/HMM/hmm_model.pkl and models/HMM/training_report.json.
Run:    python models/HMM/Codes/train_hmm.py
"""
import gzip
import json
import os
import time
import warnings

import joblib
import numpy as np
from hmmlearn import hmm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SPLIT_DIR = os.path.join(REPO_ROOT, "split")
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

STATE_CANDIDATES = [8, 16, 32]
N_ITER = 25
TOL = 1e-2
RANDOM_STATE = 42
SMOOTHING_EPSILON = 1e-4
N = 7


def load_records(split):
    with gzip.open(os.path.join(SPLIT_DIR, f"{split}.jsonl.gz"), "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_hmmlearn(seqs):
    seqs = [s for s in seqs if len(s) > 1]
    return np.concatenate(seqs).reshape(-1, 1), [len(s) for s in seqs]


def smooth(model, eps=SMOOTHING_EPSILON):
    for name in ("emissionprob_", "transmat_", "startprob_"):
        m = getattr(model, name) + eps
        setattr(model, name, m / m.sum(axis=-1, keepdims=True))
    return model


# ---------------------------------------------------------------- scoring

def window7_scores(model, windows, batch=200_000):
    """-log P(window) for each row of windows (n, 7), log-space forward."""
    log_start, log_trans, log_emit = (np.log(model.startprob_), np.log(model.transmat_),
                                      np.log(model.emissionprob_))
    out = np.empty(len(windows))
    for start in range(0, len(windows), batch):
        w = windows[start:start + batch]
        alpha = log_start[None, :] + log_emit[:, w[:, 0]].T
        for t in range(1, w.shape[1]):
            m = alpha[:, :, None] + log_trans[None, :, :]
            mx = m.max(axis=1, keepdims=True)
            alpha = mx[:, 0, :] + np.log(np.exp(m - mx).sum(axis=1)) + log_emit[:, w[:, t]].T
        mx = alpha.max(axis=1, keepdims=True)
        out[start:start + len(w)] = -(mx[:, 0] + np.log(np.exp(alpha - mx).sum(axis=1)))
    return out


def token_nll_stream(model, records, batch_records=256):
    """-log P(s_t | s_1..s_{t-1}) for t = 2..L of every record (scaled
    forward algorithm, records batched by length). One array per record."""
    pi, A, B = model.startprob_, model.transmat_, model.emissionprob_
    out = [None] * len(records)
    order = sorted(range(len(records)), key=lambda i: len(records[i]))
    for b in range(0, len(order), batch_records):
        idx = order[b:b + batch_records]
        lens = np.array([len(records[i]) for i in idx])
        mat = np.zeros((len(idx), lens.max()), dtype=np.int64)
        for j, i in enumerate(idx):
            mat[j, :lens[j]] = records[i]
        nll = np.zeros((len(idx), max(lens.max() - 1, 0)))
        alpha = pi[None, :] * B[:, mat[:, 0]].T
        alpha /= alpha.sum(axis=1, keepdims=True)
        for t in range(1, lens.max()):
            pred = alpha @ A
            joint = pred * B[:, mat[:, t]].T
            c = joint.sum(axis=1, keepdims=True)
            nll[:, t - 1] = -np.log(c[:, 0])
            alpha = joint / c
        for j, i in enumerate(idx):
            out[i] = nll[j, :lens[j] - 1]
    return out


# ---------------------------------------------------------------- training

def main():
    warnings.filterwarnings("ignore")
    with open(os.path.join(SPLIT_DIR, "vocab.json"), encoding="utf-8") as f:
        n_features = len(json.load(f))
    train_records, val_records = load_records("train"), load_records("val")
    train_seqs = []
    for r in train_records:
        train_seqs.extend([r["token_ids"]] * max(1, round(r.get("sample_weight", 1.0))))
    X, lengths = to_hmmlearn(train_seqs)
    Xv, lengths_v = to_hmmlearn([r["token_ids"] for r in val_records])
    print(f"train: {len(lengths)} sequences / {len(X)} syscalls (ADFA-LD repeated x3); "
          f"val: {len(lengths_v)} sequences / {len(Xv)} syscalls")

    candidates, best = [], None
    t_all = time.time()
    for k in STATE_CANDIDATES:
        t0 = time.time()
        model = hmm.CategoricalHMM(n_components=k, n_features=n_features, n_iter=N_ITER, tol=TOL,
                                   random_state=RANDOM_STATE, implementation="scaling", init_params="ste")
        model.fit(X, lengths)
        smooth(model)
        val_ll = model.score(Xv, lengths_v) / len(Xv)
        info = {"n_states": k, "converged": bool(model.monitor_.converged),
                "iterations": len(model.monitor_.history), "val_loglik_per_syscall": val_ll,
                "train_time_seconds": round(time.time() - t0, 1)}
        candidates.append(info)
        print(f"  {k} states: val log-lik/syscall={val_ll:.4f} ({info['train_time_seconds']}s)")
        if best is None or val_ll > best[1]["val_loglik_per_syscall"]:
            best = (model, info)

    model, info = best
    joblib.dump(model, os.path.join(RESULTS_DIR, "hmm_model.pkl"))
    report = {
        "version": "v2",
        "training_data": "split/train.jsonl.gz — all records as full sequences, ADFA-LD repeated x3 (sample_weight)",
        "model_selection": "n_states with the best per-syscall log-likelihood on benign val",
        "candidates": candidates,
        "selected_n_states": info["n_states"],
        "n_iter_max": N_ITER, "tol": TOL, "smoothing_epsilon": SMOOTHING_EPSILON,
        "vocab_size": n_features,
        "train_time_seconds": info["train_time_seconds"],
        "total_time_all_candidates_seconds": round(time.time() - t_all, 1),
    }
    with open(os.path.join(RESULTS_DIR, "training_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"selected {info['n_states']} states -> models/HMM/hmm_model.pkl")


if __name__ == "__main__":
    main()
