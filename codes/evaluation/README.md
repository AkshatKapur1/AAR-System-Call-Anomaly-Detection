# Layer 4 — evaluation & benchmarking (HMM vs GRU vs LSTM)

Compares the three models on the same test set, picks the configuration of
each and the best model for Layer 5, and compares v2 against the original
(v1) models.

## Run

```
python codes/split/make_test_partition.py   # once: tune/final partition
python codes/evaluation/evaluate_models.py   # seconds
```

## Protocol — no decision ever looks at the reported (FINAL) results

1. **Test is partitioned** (`split/test_partition.json`, stratified by source,
   label and attack type): **TUNE** 30% (559 records) for making choices,
   **FINAL** 70% (1,301 records: 583 benign, 716 attack) for reporting.
   `split/test.jsonl.gz` itself is unchanged.
2. **Thresholds from benign val only**: 95th / 99th / 99.5th percentile
   (≈5% / 1% / 0.5% false alarms; **p99 is the primary operating point**),
   either over all of val (`global`) or **per dataset** (`per_dataset` —
   each monitored environment gets its own normal profile). Scores are
   normalised to `z = (score − median) / (p99 − median)`, so z > 1 means
   "above that environment's p99".
3. **Per model, 20 candidates** (2 scoring modes × 5 aggregations × 2
   threshold schemes) are scored on TUNE. Rule, fixed in advance: highest
   TUNE ROC-AUC; within 0.01, higher TUNE recall at p99. The best model is
   chosen on TUNE with the same rule. All candidates: `tuning_results.csv`.
4. The selected configurations are evaluated **once** on FINAL, next to the
   v1 models re-scored on the same FINAL records
   (`baseline_original_models.json`).

## Results on FINAL (583 benign, 716 attack)

| Model | Selected config | ROC-AUC (95% CI) | v1 AUC | PR-AUC | Caught @p99 | v1 caught | False alarms @p99 | Precision | F1 |
|---|---|---|---|---|---|---|---|---|---|
| HMM  | window7 / mean / per-dataset | 0.683 (0.654–0.709) | 0.539 | 0.728 | 88 / 716 | 51 | 9 / 583 | 90.7% | 0.216 |
| GRU  | stream / mean / per-dataset | 0.821 (0.798–0.843) | 0.682 | 0.848 | 110 / 716 | 83 | 4 / 583 | 96.5% | 0.265 |
| **LSTM** | **stream / mean / per-dataset** | **0.835** (0.812–0.857) | 0.676 | **0.856** | 107 / 716 | 107 | 6 / 583 | 94.7% | 0.258 |

(Full numbers incl. CIs, PR-AUC, FPR@TPR, timings: `model_summary.csv`.)

**Best model: LSTM** (highest TUNE AUC, 0.847) — `best_model.json`.

At the looser p95 threshold (≈5% false alarms) the LSTM catches **232 / 716
(32%)** with 28 false alarms (89% precision); GRU 234 / 716.
From the ROC curve, the LSTM catches 37% of attacks at 5% false alarms, 48%
at 10% and 74% at 20%.

### Per dataset — ROC-AUC, v1 → v2 (LSTM)

| Dataset | v1 | v2 |
|---|---|---|
| ADFA-LD | 0.599 | **0.789** |
| LID-DS | 0.530 | **0.829** (Heartbleed CVE-2014-0160: 0.87, v1 ≈ 0.44) |
| PLAID | 0.911 | **0.950** |

### What improved and what didn't

- **Ranking of attacks vs normal improved a lot** — AUC +0.16 for the LSTM,
  +0.14 GRU, +0.14 HMM; ADFA-LD and LID-DS gained most. Heartbleed, which v1
  scored *below* normal traffic, is now clearly separable (AUC 0.87).
- **Which changes mattered** (`tuning_results.csv`): `stream` scoring (full
  context) beats `window7` for GRU/LSTM by ~0.06 AUC; `mean` aggregation
  beats max/top-k; per-dataset thresholds add recall at equal or fewer false
  alarms (TUNE, LSTM: 14.7% vs 13.3%; GRU: 18.6% vs 15.0%).
- **At the strict 1% false-alarm point, recall barely moved** (LSTM 107 →
  107, GRU 83 → 110, HMM 51 → 88) while false alarms fell (LSTM 8 → 6) and
  precision rose. A few unusual *benign* recordings set the p99 threshold
  high, so the extra separation shows up below that point (p95 and the ROC
  curve), not at it. Layer 5's hysteresis/accumulation is designed for
  exactly this regime.
- The selection rule (AUC first) was fixed before the FINAL run and is not
  changed after it. TUNE shows configurations with higher recall at p99 but
  lower AUC (e.g. `stream/top10/per_dataset`: TUNE recall 21.8%, AUC 0.73);
  choosing them now would be selecting on FINAL, so they are left as
  reported options.

### Cost

| Model | Train time | Inference (selected mode) | Parameters |
|---|---|---|---|
| HMM | 1,722 s (32 states; 2,580 s incl. 8/16-state candidates) | 68 ms / 1k windows (window7) | 8,160 |
| GRU | 2,285 s | 19 ms / 1k windows (stream) | 216,414 |
| LSTM | 1,605 s | 8 ms / 1k windows (stream) | 274,270 |

## Outputs (`evaluation/`)

| File | Contents |
|---|---|
| `tuning_results.csv` | all 60 candidates (3 models × 20) scored on TUNE |
| `thresholds.json` | selected score column, scheme and per-dataset median/p95/p99/p99.5 per model |
| `model_summary.csv` | FINAL metrics per model + v1 baseline + cost |
| `threshold_metrics.csv` | TP/FP/TN/FN, precision, recall, F1, FPR at p95 / p99 / p99.5 |
| `per_dataset_metrics.csv` | per dataset and per LID-DS CVE, with v1 AUC |
| `per_attack_type_metrics.csv` | detection rate per attack type at each threshold |
| `unk_analysis.csv` | detection with vs without unseen syscalls |
| `baseline_original_models.json` | v1 models on the same FINAL records |
| `best_model.json` | best model, configuration, thresholds and files for Layer 5 |
| `evaluation_report.json` | partition sizes, selections, settings |
| `plots/` | `roc_curves`, `pr_curves`, `confusion_matrices`, `score_distributions`, `per_attack_recall`, `v1_vs_v2` |

## Limitations

- Name-only syscall features — no arguments, return values or timing.
- GRU/LSTM learn from 128-syscall chunks (scoring is still per 7-syscall
  window); the architecture's 7-gram file is used for windowing only.
- FINAL is 70% of test; small attack groups (CVE-2012-2122 n=4, PLAID
  privesc n=4, cowroot n=11, ftp n=13, nginx n=19) are flagged `small_group`.
