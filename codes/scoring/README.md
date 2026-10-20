# Layer 3 — scoring train/val/test with HMM, GRU, LSTM (v2)

Turns the three trained models into **per-recording anomaly scores** for the
train, val and test splits. Scores only — no thresholds, no metrics, no
choices (all of that is Layer 4).

## Run

```
python models/HMM/Codes/train_hmm.py      # ~45 min (tries 8/16/32 states)
python models/gru/code/train_gru.py       # ~40 min on CPU
python models/lstm/code/train_lstm.py     # ~30 min on CPU
python codes/scoring/score_models.py      # ~35 min on CPU
```

Needs `numpy`, `torch`, `hmmlearn`, `joblib`. All paths are relative to the
repo root.

## Models (v2)

| | HMM (Track A) | GRU (Track B) | LSTM (Track C) |
|---|---|---|---|
| Training data | all of `split/train` as full recordings | all of `split/train`, 128-syscall chunks | same as GRU |
| Sample weights (ADFA-LD ×3.04) | yes (records repeated ×3) | yes (weighted loss) | yes |
| Size | 32 hidden states (picked from 8/16/32 on benign val) | embed 64, 2×128 GRU | embed 64, 2×128 LSTM |
| Stopping | EM, ≤25 iterations | early stopping on benign val loss | same |
| Benign val fit | −1.55 log-lik/syscall (v1 8 states: −2.47) | val loss 0.429 (v1 ≈ 0.65) | val loss 0.424 (v1 ≈ 0.65) |

Shared GRU/LSTM code: `models/sequence_model_common.py`. HMM training and
scoring: `models/HMM/Codes/train_hmm.py`.

## What the script does

1. **Windows** every recording into n=7 sliding windows (same as
   `codes/feature_extraction`; on `split/train` this gives exactly the
   8,771,179 training windows).
2. **Scores every window in two modes** — higher = more anomalous:
   - `window7` — the model sees only the window: HMM `−log P(window)`;
     GRU/LSTM mean NLL of syscalls 2–7 given the earlier ones in the window.
   - `stream` — the model reads the recording from the start; the window's
     score is the mean NLL of its syscalls 2–7 given **everything before
     them**. Still one score per 7-syscall window, just with memory.
3. **Aggregates per recording**: `mean`, `max`, `top1`, `top5`, `top10`
   (mean of the 1/5/10% most anomalous windows).
4. **Checks** that both modes agree on each record's first window (no earlier
   context there): max difference HMM 1.4e-14, GRU 6.2e-7, LSTM 5.6e-7.

## Outputs (`scoring/`)

`{hmm,gru,lstm}_{train,val,test}_record_scores.csv` — one row per recording:

| Column | Meaning |
|---|---|
| `record_index` | line number in `split/{train,val,test}.jsonl.gz` (unique key) |
| `recording_id`, `source`, `source_split`, `label`, `attack_type` | copied from the split (labels are only passed on to Layer 4) |
| `n_syscalls`, `n_windows`, `unk_rate` | length, number of 7-gram windows, share of unseen syscalls |
| `window7_{mean,max,top1,top5,top10}` | window-in-isolation scores, aggregated 5 ways |
| `stream_{mean,max,top1,top5,top10}` | scores with full preceding context, aggregated 5 ways |

`scoring_report.json` — counts, `<UNK>` rates, consistency checks, inference
time per mode, parameters, model size, peak memory (6.0 GB, dominated by
holding 8.77M train windows).

Train scores are for an overfitting check only — the models were fit on
that data. Records with fewer than 7 syscalls (3 train, 1 val, 4 test) have
no windows and `NaN` scores.
