"""
Shared code for the GRU (Track B) and LSTM (Track C) next-syscall models:
model definition, training loop, and the two scoring modes used by Layer 3.

Training (v2) — changes vs. the first version, all aimed at better detection:
  - uses ALL training data (8.77M tokens) instead of a 500k-window sample;
  - honours the capping `sample_weight` (ADFA-LD x3.04), which v1 ignored,
    so ADFA-LD is no longer under-represented (14% of windows in v1);
  - trains on 128-syscall chunks, so the model learns longer context, not
    just the 6 syscalls inside a 7-gram window;
  - larger model (embed 64, hidden 128, 2 layers, dropout 0.1);
  - early stopping on benign val loss (no attack data is used for training).

Scoring modes (both score n=7 windows, as in the architecture):
  - "window7": the model only sees the 6 syscalls inside the window
    (same as v1 per_window_nll);
  - "stream" : the model reads the whole recording in order and each window
    is scored with everything that came before it as context.
Which mode is used is decided on the tune part of test, never on final.
"""
import gzip
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLIT_DIR = os.path.join(REPO_ROOT, "split")
PAD_ID = 0
N = 7

CONFIG = {
    "embed_dim": 64,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.1,
    "chunk_len": 128,
    "batch_size": 128,
    "learning_rate": 2e-3,
    "max_epochs": 12,
    "patience": 2,
    "grad_clip": 1.0,
    "seed": 42,
}


class SyscallRNN(nn.Module):
    def __init__(self, cell, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, state=None):
        out, state = self.rnn(self.embedding(x), state)
        return self.output(out), state


# ---------------------------------------------------------------- data

def load_records(split):
    with gzip.open(os.path.join(SPLIT_DIR, f"{split}.jsonl.gz"), "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_chunks(records, chunk_len, rng):
    """Cut every record into chunks of chunk_len+1 tokens (input/target
    shifted by one), with a random start offset so chunk boundaries move
    between epochs. Returns (tokens[n, chunk_len+1], weights[n])."""
    chunks, weights = [], []
    for r in records:
        ids = r["token_ids"]
        if len(ids) < 2:
            continue
        offset = rng.randrange(chunk_len) if len(ids) > chunk_len + 1 else 0
        starts = list(range(offset, len(ids) - 1, chunk_len))
        if offset:
            starts = [0] + starts
        for s in starts:
            piece = ids[s:s + chunk_len + 1]
            if len(piece) < 2:
                continue
            chunks.append(piece + [PAD_ID] * (chunk_len + 1 - len(piece)))
            weights.append(r.get("sample_weight", 1.0))
    return np.asarray(chunks, dtype=np.int64), np.asarray(weights, dtype=np.float32)


def weighted_loss(model, batch, weights):
    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits, _ = model(inputs)
    ce = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                     ignore_index=PAD_ID, reduction="none").view(targets.shape)
    mask = (targets != PAD_ID).float()
    w = weights[:, None] * mask
    return (ce * w).sum(), w.sum()


def evaluate_loss(model, chunks, weights, batch_size=512):
    model.eval()
    total, denom = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            l, d = weighted_loss(model, torch.from_numpy(chunks[i:i + batch_size]),
                                 torch.from_numpy(weights[i:i + batch_size]))
            total += l.item(); denom += d.item()
    return total / denom


# ---------------------------------------------------------------- training

def train(cell, results_dir):
    cfg = CONFIG
    torch.manual_seed(cfg["seed"]); rng = random.Random(cfg["seed"])
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(SPLIT_DIR, "vocab.json"), encoding="utf-8") as f:
        vocab_size = len(json.load(f))

    train_records, val_records = load_records("train"), load_records("val")
    val_chunks, val_weights = make_chunks(val_records, cfg["chunk_len"], random.Random(0))
    model = SyscallRNN(cell, vocab_size, cfg["embed_dim"], cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"])
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)
    print(f"{cell.upper()}: {n_params} params, {sum(len(r['token_ids']) for r in train_records)} train tokens, "
          f"{len(val_chunks)} val chunks, threads={torch.get_num_threads()}")

    best_val, best_state, bad_epochs, history = math.inf, None, 0, []
    t_start = time.time()
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        chunks, weights = make_chunks(train_records, cfg["chunk_len"], rng)
        order = np.random.default_rng(cfg["seed"] + epoch).permutation(len(chunks))
        total, denom = 0.0, 0.0
        for i in range(0, len(order), cfg["batch_size"]):
            idx = order[i:i + cfg["batch_size"]]
            loss_sum, w_sum = weighted_loss(model, torch.from_numpy(chunks[idx]), torch.from_numpy(weights[idx]))
            optimizer.zero_grad()
            (loss_sum / w_sum).backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            total += loss_sum.item(); denom += w_sum.item()
        val_loss = evaluate_loss(model, val_chunks, val_weights)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": total / denom, "val_loss": val_loss,
                        "elapsed_s": round(time.time() - t_start, 1)})
        print(f"  epoch {epoch}: train={total / denom:.4f} val={val_loss:.4f} ({time.time() - t_start:.0f}s)")
        if val_loss < best_val - 1e-4:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                break
    train_time = time.time() - t_start

    torch.save({"model_state_dict": best_state, "cell": cell, "vocab_size": vocab_size, **cfg},
               os.path.join(results_dir, f"{cell}_model.pt"))
    report = {
        "version": "v2",
        "training_data": "split/train.jsonl.gz — all records, 128-syscall chunks, weighted by sample_weight",
        "early_stopping": "benign val loss (split/val.jsonl.gz)",
        "architecture": {"type": f"{cell.upper()} (next-syscall prediction)", **{k: cfg[k] for k in
                         ("embed_dim", "hidden_dim", "num_layers", "dropout")}, "n_params": n_params},
        "config": cfg,
        "epochs_run": len(history),
        "best_val_loss": best_val,
        "history": history,
        "train_time_seconds": round(train_time, 1),
        "vocab_size": vocab_size,
    }
    with open(os.path.join(results_dir, "training_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  saved {cell}_model.pt (best val loss {best_val:.4f}) in {train_time:.0f}s")


# ---------------------------------------------------------------- scoring

def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = SyscallRNN(ckpt["cell"], ckpt["vocab_size"], ckpt["embed_dim"], ckpt["hidden_dim"],
                       ckpt["num_layers"], ckpt["dropout"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def window7_scores(model, windows, batch_size=8192):
    """Mean NLL of syscalls 2..7 of each window, seeing only the window."""
    out = np.empty(len(windows))
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            b = torch.from_numpy(windows[i:i + batch_size])
            logits, _ = model(b[:, :-1])
            ce = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), b[:, 1:].reshape(-1),
                                             reduction="none").view(len(b), -1)
            out[i:i + len(b)] = ce.mean(dim=1).numpy()
    return out


def token_nll_stream(model, records, batch_records=64, piece=2048):
    """Per-token NLL of syscall t given ALL earlier syscalls of its record.
    Records are processed in length-sorted batches, carrying the hidden
    state across pieces so very long recordings fit in memory.
    Returns a list (one array per record) of length len(ids)-1."""
    out = [None] * len(records)
    order = sorted(range(len(records)), key=lambda i: len(records[i]))
    with torch.no_grad():
        for b in range(0, len(order), batch_records):
            idx = order[b:b + batch_records]
            seqs = [records[i] for i in idx]
            max_len = max(len(s) for s in seqs)
            mat = np.full((len(seqs), max_len), PAD_ID, dtype=np.int64)
            for j, s in enumerate(seqs):
                mat[j, :len(s)] = s
            mat = torch.from_numpy(mat)
            nll = np.zeros((len(seqs), max(max_len - 1, 0)), dtype=np.float32)
            state = None
            for p in range(0, max_len - 1, piece):
                inputs = mat[:, p:p + piece]
                targets = mat[:, p + 1:p + piece + 1]
                inputs = inputs[:, :targets.size(1)]
                logits, state = model(inputs, state)
                ce = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                                 reduction="none").view(targets.shape)
                nll[:, p:p + targets.size(1)] = ce.numpy()
            for j, i in enumerate(idx):
                out[i] = nll[j, :len(seqs[j]) - 1]
    return out


def stream_window_scores(token_nll):
    """Window k covers syscalls k..k+6; its score is the mean NLL of
    syscalls k+1..k+6 (same 6 predictions as window7, but with full history)."""
    if len(token_nll) < N - 1:
        return np.empty(0)
    c = np.concatenate([[0.0], np.cumsum(token_nll, dtype=np.float64)])
    return (c[N - 1:] - c[:-(N - 1)]) / (N - 1)
