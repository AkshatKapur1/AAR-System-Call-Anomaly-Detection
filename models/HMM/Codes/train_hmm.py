"""
Track A: HMM training on pre-extracted 7-gram windows.

Input : train_windows_n7.jsonl — one JSON array of 7 integer syscall IDs
        per line (already windowed, already integer-encoded via vocab.json).
Output: trained CategoricalHMM + evaluation report, written to
        track_a_hmm_results/.

Each 7-gram window is treated as one short independent sequence for
Baum-Welch training (hmmlearn's X/lengths API). With 8.77M windows in the
source file, fitting on all of them is not tractable in reasonable time,
so training uses a fixed-size systematic subsample (every Nth line) drawn
across the whole file, plus a separate held-out subsample (offset from the
training one) for a convergence/likelihood sanity check.
"""
import json
import os
import time

import numpy as np
from hmmlearn import hmm
import joblib

INPUT_PATH = r"C:\Users\Akshat Kapur\Downloads\train_windows_n7.jsonl\train_windows_n7.jsonl"
VOCAB_PATH = os.path.join(os.path.dirname(__file__), "vocab.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "track_a_hmm_results")

TOTAL_LINES = 8_771_179
TRAIN_SAMPLE_SIZE = 200_000
HELD_OUT_SAMPLE_SIZE = 50_000
N_STATES = 8
N_ITER = 20
RANDOM_STATE = 42
SMOOTHING_EPSILON = 1e-4  # Laplace smoothing: avoids exact-zero probabilities
                          # for syscalls that are rare/absent in the training
                          # subsample, which otherwise cause -inf log-likelihood
                          # on held-out windows that contain them.


def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def systematic_sample(path, total_lines, train_n, held_out_n, seed=RANDOM_STATE):
    """
    Single pass over the file. Every `stride`-th line goes to the training
    sample; a second, offset stride pulls the held-out sample, so the two
    sets are disjoint and both spread uniformly across the whole file
    (not just the first N lines).
    """
    rng = np.random.default_rng(seed)
    stride = total_lines // train_n
    held_out_stride = total_lines // held_out_n
    train_windows = []
    held_out_windows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % stride == 0 and len(train_windows) < train_n:
                train_windows.append(json.loads(line))
            elif i % held_out_stride == 1 and len(held_out_windows) < held_out_n:
                held_out_windows.append(json.loads(line))
    return np.array(train_windows, dtype=np.int32), np.array(held_out_windows, dtype=np.int32)


def windows_to_hmmlearn_input(windows):
    """windows: (n_windows, 7) -> flattened X: (n_windows*7, 1), lengths: (n_windows,)"""
    n_windows, window_len = windows.shape
    X = windows.reshape(-1, 1)
    lengths = np.full(n_windows, window_len, dtype=np.int32)
    return X, lengths


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    vocab = load_vocab()
    n_features = len(vocab)  # 222

    print(f"Sampling {TRAIN_SAMPLE_SIZE} training windows and "
          f"{HELD_OUT_SAMPLE_SIZE} held-out windows from {TOTAL_LINES} total...")
    t0 = time.time()
    train_windows, held_out_windows = systematic_sample(
        INPUT_PATH, TOTAL_LINES, TRAIN_SAMPLE_SIZE, HELD_OUT_SAMPLE_SIZE
    )
    print(f"  sampled in {time.time() - t0:.1f}s "
          f"(train={train_windows.shape}, held_out={held_out_windows.shape})")

    X_train, lengths_train = windows_to_hmmlearn_input(train_windows)
    X_held_out, lengths_held_out = windows_to_hmmlearn_input(held_out_windows)

    print(f"Fitting CategoricalHMM: n_states={N_STATES}, n_features={n_features}, "
          f"n_iter={N_ITER}, on {len(lengths_train)} sequences ({len(X_train)} observations)...")
    t0 = time.time()
    model = hmm.CategoricalHMM(
        n_components=N_STATES,
        n_features=n_features,
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
        verbose=True,
    )
    model.fit(X_train, lengths_train)
    train_time = time.time() - t0
    print(f"  fit finished in {train_time:.1f}s, converged={model.monitor_.converged}, "
          f"final train log-likelihood={model.monitor_.history[-1]:.2f}")

    # Laplace smoothing: floor every probability at SMOOTHING_EPSILON and
    # renormalize, so no syscall/transition/start-state has exactly zero
    # probability. Without this, a held-out window containing any syscall
    # the model never assigned probability to scores -inf.
    def smooth(matrix, eps=SMOOTHING_EPSILON):
        matrix = matrix + eps
        return matrix / matrix.sum(axis=-1, keepdims=True)

    model.emissionprob_ = smooth(model.emissionprob_)
    model.transmat_ = smooth(model.transmat_)
    model.startprob_ = smooth(model.startprob_)

    # Per-window log-likelihood on held-out data (not used in training) —
    # a convergence/sanity check, NOT a detection-accuracy metric: this
    # file has no attack-labeled windows, so precision/recall/F1/AUC can't
    # be computed yet. That needs a labeled test set (benign + attack)
    # built the same way, which doesn't exist yet in this handoff.
    per_window_scores = []
    for w in held_out_windows:
        score = model.score(w.reshape(-1, 1))
        per_window_scores.append(score)
    per_window_scores = np.array(per_window_scores)

    report = {
        "input_file": INPUT_PATH,
        "total_lines_in_source": TOTAL_LINES,
        "vocab_size": n_features,
        "train_sample_size": int(TRAIN_SAMPLE_SIZE),
        "held_out_sample_size": int(HELD_OUT_SAMPLE_SIZE),
        "n_hidden_states": N_STATES,
        "n_iter_requested": N_ITER,
        "converged": bool(model.monitor_.converged),
        "n_iter_actual": len(model.monitor_.history),
        "final_train_log_likelihood": float(model.monitor_.history[-1]),
        "train_time_seconds": round(train_time, 1),
        "held_out_log_likelihood": {
            "mean": float(per_window_scores.mean()),
            "std": float(per_window_scores.std()),
            "min": float(per_window_scores.min()),
            "max": float(per_window_scores.max()),
            "p05": float(np.percentile(per_window_scores, 5)),
            "p50": float(np.percentile(per_window_scores, 50)),
            "p95": float(np.percentile(per_window_scores, 95)),
        },
        "caveat": (
            "No attack-labeled data was available in this handoff (only "
            "train_windows_n7.jsonl, benign windows). These are convergence "
            "and held-out-likelihood diagnostics, not detection accuracy. "
            "Real precision/recall/F1/AUC requires scoring a labeled "
            "benign+attack test set through this same model."
        ),
    }

    model_path = os.path.join(RESULTS_DIR, "hmm_model.pkl")
    joblib.dump(model, model_path)

    report_path = os.path.join(RESULTS_DIR, "training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    scores_path = os.path.join(RESULTS_DIR, "held_out_window_scores.csv")
    with open(scores_path, "w", encoding="utf-8") as f:
        f.write("window_index,log_likelihood\n")
        for i, s in enumerate(per_window_scores):
            f.write(f"{i},{s}\n")

    print(f"\nSaved model -> {model_path}")
    print(f"Saved report -> {report_path}")
    print(f"Saved held-out scores -> {scores_path}")


if __name__ == "__main__":
    main()
