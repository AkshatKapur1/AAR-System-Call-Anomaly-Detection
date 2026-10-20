# Layer 4 — evaluation & benchmarking (HMM vs GRU vs LSTM)

Compares the three models' anomaly scores from Layer 3 against the ground
truth on the **same test set** (834 benign, 1,026 attack recordings), and
selects the model passed on to Layer 5 (response).

## Run

```
python codes/evaluation/evaluate_models.py
```

Needs `numpy`, `scikit-learn`, `matplotlib`. Reads only `scoring/` and
`models/*/training_report.json`; runs in a few seconds.

## Protocol (fixed before looking at test results)

1. **Thresholds from benign val only.** Per model, the 95th / 99th / 99.5th
   percentile of its val `score` (≈5% / 1% / 0.5% expected false alarms).
   **p99 is the primary operating point.** Each model gets its own
   threshold (score scales differ), set by the same rule.
2. **Decision:** a test recording is flagged as an attack when `score > threshold`.
3. **Threshold-free metrics** (ROC-AUC with 95% bootstrap CI, PR-AUC,
   FPR@TPR=1, FPR@TPR=0.95) compare models independent of the threshold.
4. The **oracle best-F1 threshold on test** is reported only as an upper
   bound — never used for any decision.
5. **Best-model rule:** highest test ROC-AUC; if the top two are within 0.01,
   higher recall at p99 wins, then faster inference.
6. Records with no score (<7 syscalls) are excluded: 4 test, 1 val.

## Outputs (`evaluation/`)

| File | Contents |
|---|---|
| `thresholds.json` | p95 / p99 / p99.5 threshold per model, from benign val |
| `model_summary.csv` | one row per model: ROC-AUC (+CI), PR-AUC, FPR@TPR=1, precision/recall/F1/FPR at p99, oracle F1, training time, inference time, parameters, model size |
| `threshold_metrics.csv` | TP/FP/TN/FN, precision, recall, F1, FPR, accuracy for every model × threshold |
| `per_dataset_metrics.csv` | the same per dataset (ADFA-LD, LID-DS, PLAID) and per LID-DS CVE |
| `per_attack_type_metrics.csv` | detection rate per attack type at each threshold, AUC vs same-dataset benign |
| `aggregation_comparison.csv` | `score_mean` vs `score_max` vs `score_top5` (reported only) |
| `unk_analysis.csv` | detection with vs without unseen syscalls (`<UNK>`) |
| `best_model.json` | selected model, reason, threshold and file paths for Layer 5 |
| `evaluation_report.json` | counts, exclusions, settings |
| `plots/roc_curves.png`, `pr_curves.png`, `confusion_matrices.png`, `score_distributions.png`, `per_attack_recall.png` | figures for the report |

## Results

| Model | ROC-AUC (95% CI) | PR-AUC | Precision @p99 | Recall @p99 | F1 @p99 | FPR @p99 | Train time | Inference / 1k windows |
|---|---|---|---|---|---|---|---|---|
| HMM  | 0.527 (0.500–0.554) | 0.629 | 0.923 | 0.070 | 0.131 | 0.7% | 430.7 s | 6.7 ms |
| GRU  | **0.672** (0.648–0.697) | **0.736** | 0.901 | 0.115 | 0.205 | 1.6% | 84.0 s | 2.7 ms |
| LSTM | 0.665 (0.637–0.688) | 0.733 | 0.911 | **0.141** | **0.244** | 1.7% | 47.2 s | 2.6 ms |

**Best model: LSTM** — its AUC is within 0.01 of the GRU (statistically a
tie; CIs overlap), and it catches more attacks at the p99 threshold
(14.1% vs 11.5%) at the same false-alarm level.

### Per dataset (ROC-AUC)

| Dataset | HMM | GRU | LSTM |
|---|---|---|---|
| PLAID | 0.815 | 0.899 | **0.900** |
| ADFA-LD | 0.432 | **0.594** | 0.587 |
| LID-DS (both CVEs) | **0.625** | 0.592 | 0.523 |

### Key findings

1. **GRU/LSTM clearly beat the HMM**, and are ~3× faster at inference; GRU
   and LSTM are statistically tied.
2. **Strong on PLAID** (AUC ≈ 0.90; nginx, privesc, ftp attacks caught
   65–89% at p99), **weak on ADFA-LD** (AUC ≈ 0.59) and **LID-DS
   Heartbleed (CVE-2014-0160) is missed entirely** — its attack traces
   score *lower* than normal ones (AUC < 0.5), i.e. at the 7-gram syscall
   name level the exploit looks like ordinary, very regular server activity.
3. **At a 1% false-alarm budget only 7–14% of attacks are caught.** Alarms
   that do fire are ~90% correct (high precision) — conservative but reliable,
   which suits a kill switch.
4. **FPR@TPR=1 ≈ 1.0 for all models:** some attacks score below almost every
   benign recording, so catching *every* attack is not achievable with these scores.
5. **Aggregation:** `score_mean` reaches a higher AUC (GRU 0.760, LSTM 0.758)
   than the pre-chosen `score_top5` (0.672 / 0.665). The primary score was
   fixed in advance and is **not** switched here, since choosing it from test
   results would bias the evaluation; this is flagged as an improvement to
   validate properly (e.g. with a held-out attack set) in future work.
6. **Unseen syscalls:** the 29 attacks containing syscalls never seen in
   training are caught 86–90% by GRU/LSTM; the remaining 993 still give AUC
   ≈ 0.66, so detection is not only driven by `<UNK>`.

### Limitations

- Name-only 7-gram features — no syscall arguments, return values or timing.
- Models trained on a 2–6% sample of the training windows, ignoring capping `sample_weight`.
- Small attack groups (CVE-2012-2122 n=6, PLAID privesc n=5, cowroot n=16,
  ftp n=17) are flagged `small_group` — treat their rates as indicative only.
