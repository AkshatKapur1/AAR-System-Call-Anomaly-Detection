# Layer 3 completion — scoring val/test with HMM, GRU, LSTM

Turns the three trained models into **one anomaly score per recording** for
the val and test splits. Scores only — no thresholds, no metrics (that is
Layer 4).

## Run

```
python codes/scoring/score_models.py
```

Needs `numpy`, `torch`, `hmmlearn`, `joblib`. Paths are relative to the repo
root, so it runs on any machine. Takes ~2 minutes on CPU.

## What it does

1. **Verifies the models** — rebuilds the exact held-out windows used at
   training time from `feature_extraction/train_windows_n7.jsonl.gz`,
   rescores them, and checks they match the stored
   `held_out_window_scores*.csv` in `models/`. Stops if they don't.
2. **Windows val and test** — same n=7 sliding window as
   `codes/feature_extraction/extract_features.py` (via `ngram_utils.sliding_windows`),
   never crossing a record boundary. Applied to `split/train.jsonl.gz` this
   reproduces all 8,771,179 training windows. Windows are built in memory,
   not saved.
3. **Scores every window** — higher = more anomalous for all models:
   - HMM: `-log P(window)` (batched forward algorithm, same maths as `model.score()`)
   - GRU / LSTM: mean NLL of predicting syscalls 2–7 within the window
     (same as `per_window_nll()` in the training scripts)
4. **Aggregates to one score per recording** — `score = score_top5`, the mean
   of the record's 5% most anomalous windows (fixed in advance, not tuned —
   attacks are often short bursts inside long normal recordings). `score_mean`
   and `score_max` are kept as alternatives for Layer 4.

## Outputs (`scoring/`)

| File | Contents |
|---|---|
| `{hmm,gru,lstm}_val_record_scores.csv` | 833 benign val recordings |
| `{hmm,gru,lstm}_test_record_scores.csv` | 1,860 test recordings (834 benign, 1,026 attack) |
| `scoring_report.json` | verification results, window counts, UNK rates, inference time, model size, peak memory |

CSV columns:

| Column | Meaning |
|---|---|
| `record_index` | line number in `split/{val,test}.jsonl.gz` — unique key (some LID-DS `recording_id`s appear twice in test: pre-exploit benign part + attack part) |
| `recording_id`, `source`, `source_split`, `label`, `attack_type` | copied from the split file (labels are not used here, only passed on to Layer 4) |
| `n_syscalls`, `n_windows` | record length and number of 7-gram windows |
| `unk_rate` | fraction of syscalls not in the training vocabulary (`<UNK>`) |
| `score_mean`, `score_max`, `score_top5` | window scores aggregated three ways |
| `score` | primary record score (= `score_top5`) |

## Notes for Layer 4

- 1 val and 4 test records have fewer than 7 syscalls → no windows →
  scores are `NaN`. Listed in `scoring_report.json`.
- HMM scores (sum over 7 syscalls) and GRU/LSTM scores (mean over 6
  predictions) are on different scales — compare models by ranking metrics
  (AUC) or with a separate threshold per model, never by raw values.
- `<UNK>` is ~0.6% of attack syscalls vs ~0.00005% of benign ones; the
  GRU/LSTM never saw `<UNK>` in training, so it raises their scores.
- Track naming follows the architecture diagram: HMM = A, GRU = B, LSTM = C.
