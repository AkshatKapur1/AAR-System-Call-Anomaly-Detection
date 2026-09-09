"""
Track B: LSTM training on pre-extracted 7-gram windows.

Input : train_windows_n7.jsonl — one JSON array of 7 integer syscall IDs
        per line (already windowed, already integer-encoded via vocab.json).
        Same input file used for Track A's HMM (codes/track_a_hmm/).
Output: trained LSTM + evaluation report, written to models/lstm/results/.

Task: next-syscall prediction (standard approach for this kind of
sequence model). For each window [s0..s6], the model reads s0..s5 and is
trained to predict s1..s6 at each step (teacher-forced, shifted-by-one) —
this uses every position in the window as a training signal instead of
just the last one. One-class / semi-supervised: trained only on this
benign-only file, same as Track A, same as the roadmap's stated approach
(no attack examples seen during training).
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

INPUT_PATH = r"C:\Users\Akshat Kapur\Downloads\train_windows_n7.jsonl\train_windows_n7.jsonl"
VOCAB_PATH = os.path.join(os.path.dirname(__file__), "vocab.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

TOTAL_LINES = 8_771_179
TRAIN_SAMPLE_SIZE = 500_000
HELD_OUT_SAMPLE_SIZE = 50_000

EMBED_DIM = 32
HIDDEN_DIM = 64
NUM_LAYERS = 1
BATCH_SIZE = 512
EPOCHS = 5
LEARNING_RATE = 1e-3
RANDOM_STATE = 42


def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def systematic_sample(path, total_lines, train_n, held_out_n, seed=RANDOM_STATE):
    """Same sampling strategy as Track A's HMM script — one pass over the
    file, two interleaved systematic strides, so both samples spread
    uniformly across the whole 8.77M-line file rather than clustering."""
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
    return (np.array(train_windows, dtype=np.int64),
            np.array(held_out_windows, dtype=np.int64))


class WindowDataset(Dataset):
    """Each 7-gram window -> (input=first 6 tokens, target=last 6 tokens)."""

    def __init__(self, windows):
        self.inputs = windows[:, :-1]
        self.targets = windows[:, 1:]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class SyscallLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.output(lstm_out)  # (batch, seq_len, vocab_size) — logits per step


def per_window_nll(model, windows, device, batch_size=1024):
    """Average negative log-likelihood per window — the anomaly score:
    higher (more positive) = the model found this window more surprising."""
    model.eval()
    scores = []
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = windows[start:start + batch_size]
            inputs = torch.from_numpy(batch[:, :-1]).to(device)
            targets = torch.from_numpy(batch[:, 1:]).to(device)
            logits = model(inputs)  # (B, 6, vocab)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss = loss.view(targets.shape).mean(dim=1)  # avg NLL per window
            scores.extend(loss.cpu().numpy().tolist())
    return np.array(scores)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    vocab = load_vocab()
    vocab_size = len(vocab)  # 222

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Sampling {TRAIN_SAMPLE_SIZE} training windows and "
          f"{HELD_OUT_SAMPLE_SIZE} held-out windows from {TOTAL_LINES} total...")
    t0 = time.time()
    train_windows, held_out_windows = systematic_sample(
        INPUT_PATH, TOTAL_LINES, TRAIN_SAMPLE_SIZE, HELD_OUT_SAMPLE_SIZE
    )
    print(f"  sampled in {time.time() - t0:.1f}s "
          f"(train={train_windows.shape}, held_out={held_out_windows.shape})")

    train_dataset = WindowDataset(train_windows)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SyscallLSTM(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training LSTM: embed_dim={EMBED_DIM}, hidden_dim={HIDDEN_DIM}, "
          f"layers={NUM_LAYERS}, epochs={EPOCHS}, batch_size={BATCH_SIZE}, "
          f"on {len(train_dataset)} windows...")
    t0 = time.time()
    epoch_losses = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        epoch_losses.append(avg_loss)
        print(f"  epoch {epoch}/{EPOCHS}  avg_loss={avg_loss:.4f}  "
              f"({time.time() - t0:.1f}s elapsed)")
    train_time = time.time() - t0
    print(f"  training finished in {train_time:.1f}s")

    print("Scoring held-out windows (never seen during training)...")
    held_out_scores = per_window_nll(model, held_out_windows, device)

    report = {
        "input_file": INPUT_PATH,
        "total_lines_in_source": TOTAL_LINES,
        "vocab_size": vocab_size,
        "train_sample_size": int(TRAIN_SAMPLE_SIZE),
        "held_out_sample_size": len(held_out_windows),
        "architecture": {
            "type": "LSTM (next-syscall prediction)",
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
        },
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epoch_train_losses": epoch_losses,
        "final_train_loss": epoch_losses[-1],
        "train_time_seconds": round(train_time, 1),
        "held_out_nll": {
            "mean": float(held_out_scores.mean()),
            "std": float(held_out_scores.std()),
            "min": float(held_out_scores.min()),
            "max": float(held_out_scores.max()),
            "p05": float(np.percentile(held_out_scores, 5)),
            "p50": float(np.percentile(held_out_scores, 50)),
            "p95": float(np.percentile(held_out_scores, 95)),
        },
        "caveat": (
            "No attack-labeled data was available in this handoff (only "
            "train_windows_n7.jsonl, benign windows). These are convergence "
            "and held-out-loss diagnostics, not detection accuracy. Real "
            "precision/recall/F1/AUC requires scoring a labeled benign+attack "
            "test set through this same model. Score direction note: unlike "
            "the HMM's log-likelihood (higher/less-negative = more normal), "
            "this is negative log-likelihood loss — HIGHER = more surprising "
            "= more anomalous. Don't compare the two raw numbers directly."
        ),
    }

    model_path = os.path.join(RESULTS_DIR, "lstm_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
    }, model_path)

    report_path = os.path.join(RESULTS_DIR, "training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    scores_path = os.path.join(RESULTS_DIR, "held_out_window_scores.csv")
    with open(scores_path, "w", encoding="utf-8") as f:
        f.write("window_index,negative_log_likelihood\n")
        for i, s in enumerate(held_out_scores):
            f.write(f"{i},{s}\n")

    print(f"\nSaved model -> {model_path}")
    print(f"Saved report -> {report_path}")
    print(f"Saved held-out scores -> {scores_path}")


if __name__ == "__main__":
    main()
