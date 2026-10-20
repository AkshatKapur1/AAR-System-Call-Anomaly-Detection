"""
Track B: GRU next-syscall model (v2).

Trains on all of split/train.jsonl.gz with sample weights, 128-syscall
context chunks and early stopping on benign val loss. The model, training
loop and scoring code are shared with the other recurrent track in
models/sequence_model_common.py, so GRU vs LSTM differ only in the cell.

Output: models/gru/results/gru_model.pt and training_report.json.
Run:    python models/gru/code/train_gru.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from sequence_model_common import train  # noqa: E402

if __name__ == "__main__":
    train("gru", os.path.join(os.path.dirname(__file__), "..", "results"))
