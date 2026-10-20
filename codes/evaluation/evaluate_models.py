"""
Layer 4: evaluation & benchmarking of the HMM, GRU and LSTM anomaly scores.

Input : scoring/{hmm,gru,lstm}_{val,test}_record_scores.csv (Layer 3),
        split/test_partition.json (TUNE 30% / FINAL 70% of test),
        evaluation/baseline_original_models.json (v1 models on FINAL),
        models/*/training_report.json, scoring/scoring_report.json.
Output: evaluation/ — tuning results, thresholds, final metrics, plots,
        best_model.json for Layer 5.

Protocol — no decision ever looks at FINAL:
  1. Thresholds come from benign val only: the 95th / 99th / 99.5th
     percentile (~5% / 1% / 0.5% false alarms; p99 is the primary operating
     point), either over all of val ("global") or separately per dataset
     ("per_dataset" — each monitored environment gets its own normal profile).
     Scores are normalised the same way, z = (s - median) / (p99 - median),
     so z > 1 means "above that environment's p99".
  2. For every model, each candidate (scoring mode x aggregation x
     threshold scheme) is evaluated on TUNE. The candidate with the highest
     TUNE ROC-AUC is selected; within 0.01 AUC, higher TUNE recall at p99
     wins. The best model is chosen on TUNE with the same rule.
  3. The selected configurations are evaluated once on FINAL: ROC-AUC (95%
     bootstrap CI), PR-AUC, FPR@TPR=1/0.95, precision / recall / F1 / FPR at
     each threshold, per dataset / CVE / attack type, cost metrics, and a
     side-by-side with the v1 baseline on the same FINAL records.
  4. Two operating points per model:
       - "kill" (strict): p99 of benign val, ~1% false alarms — for the
         Layer 5 kill switch, where a false alarm kills a legitimate process;
       - "balanced alert": the z cutoff that maximises Youden's J
         (attacks caught % + normal passed % - 1) on TUNE, applied once to
         FINAL — catches most attacks while passing most normal recordings.
  The oracle best-F1 threshold on FINAL is reported as an upper bound only.

Run from anywhere:  python codes/evaluation/evaluate_models.py
"""
import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCORING_DIR = os.path.join(REPO_ROOT, "scoring")
OUT_DIR = os.path.join(REPO_ROOT, "evaluation")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
PARTITION_PATH = os.path.join(REPO_ROOT, "split", "test_partition.json")
BASELINE_PATH = os.path.join(OUT_DIR, "baseline_original_models.json")

MODELS = ["hmm", "gru", "lstm"]
MODEL_LABELS = {"hmm": "HMM (Track A)", "gru": "GRU (Track B)", "lstm": "LSTM (Track C)"}
TRAINING_REPORTS = {
    "hmm": os.path.join(REPO_ROOT, "models", "HMM", "training_report.json"),
    "gru": os.path.join(REPO_ROOT, "models", "gru", "results", "training_report.json"),
    "lstm": os.path.join(REPO_ROOT, "models", "lstm", "results", "training_report.json"),
}
MODES = ["window7", "stream"]
AGGREGATIONS = ["mean", "max", "top1", "top5", "top10"]
SCHEMES = ["global", "per_dataset"]
PERCENTILES = {"p95": 95.0, "p99": 99.0, "p99_5": 99.5}
PRIMARY = "p99"
AUC_TIE = 0.01
SMALL_GROUP = 20
N_BOOTSTRAP = 1000
SEED = 42

COLORS = {"hmm": "#2a78d6", "gru": "#eb6834", "lstm": "#1baf7a"}
BENIGN_COLOR, ATTACK_COLOR, BASELINE_COLOR = "#2a78d6", "#eb6834", "#a3a29c"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CM_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_seq", ["#f0f5fc", "#9ec5f4", "#3987e5", "#184f95"])


# ---------------------------------------------------------------- data

def load_scores(model, split):
    with open(os.path.join(SCORING_DIR, f"{model}_{split}_record_scores.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for mode in MODES:
            for agg in AGGREGATIONS:
                r[f"{mode}_{agg}"] = float(r[f"{mode}_{agg}"])
        r["unk_rate"] = float(r["unk_rate"])
        r["record_index"] = int(r["record_index"])
        r["y"] = 1 if r["label"] == "attack" else 0
        r["dataset"] = r["source"].split("/")[0]
    return [r for r in rows if not math.isnan(r["window7_mean"])]


class Normaliser:
    """Per-scheme reference statistics from benign val scores of one column."""

    def __init__(self, val_rows, col, scheme):
        self.scheme = scheme
        groups = {"all": val_rows} if scheme == "global" else {
            d: [r for r in val_rows if r["dataset"] == d] for d in sorted({r["dataset"] for r in val_rows})}
        self.stats = {}
        for g, rows in groups.items():
            v = np.array([r[col] for r in rows])
            self.stats[g] = {"median": float(np.median(v)),
                             **{k: float(np.percentile(v, q)) for k, q in PERCENTILES.items()}}

    def _key(self, r):
        return "all" if self.scheme == "global" else r["dataset"]

    def z(self, r, col):
        st = self.stats[self._key(r)]
        return (r[col] - st["median"]) / max(st[PRIMARY] - st["median"], 1e-12)

    def flag(self, r, col, level):
        return r[col] > self.stats[self._key(r)][level]


def arrays(rows, norm, col):
    y = np.array([r["y"] for r in rows])
    z = np.array([norm.z(r, col) for r in rows])
    return y, z


# ---------------------------------------------------------------- metrics

def counts(y, pred):
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0))); fn = int(np.sum(~pred & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else 0.0, "accuracy": (tp + tn) / len(y) if len(y) else 0.0}


def decide(rows, norm, col, level):
    return np.array([norm.flag(r, col, level) for r in rows])


def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def fpr_at_tpr(y, s, target):
    fpr, tpr, _ = roc_curve(y, s)
    return float(fpr[np.argmax(tpr >= target)])


def auc_ci(y, s, rng):
    aucs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            aucs.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def oracle_best_f1(y, s):
    p, r, _ = precision_recall_curve(y, s)
    return float(np.max(np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)))


def balanced_cutoff(y, z):
    """z cutoff maximising Youden's J = TPR - FPR (flag when z > cutoff);
    placed midway between the chosen ROC threshold and the next one."""
    fpr, tpr, thr = roc_curve(y, z)
    i = int(np.argmax(tpr - fpr))
    lower = thr[i + 1] if i + 1 < len(thr) else thr[i] - 1.0
    return float((thr[i] + lower) / 2)


def better(a, b):
    """Selection rule: higher AUC; within AUC_TIE, higher recall at p99."""
    if b is None:
        return True
    if abs(a["tune_roc_auc"] - b["tune_roc_auc"]) < AUC_TIE:
        return a["tune_recall_p99"] > b["tune_recall_p99"]
    return a["tune_roc_auc"] > b["tune_roc_auc"]


# ---------------------------------------------------------------- output helpers

def write_csv(name, rows):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)}")


def write_json(name, obj):
    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote evaluation/{name}")


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)}")


# ---------------------------------------------------------------- main

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    partition = json.load(open(PARTITION_PATH, encoding="utf-8"))
    tune_idx, final_idx = set(partition["tune"]), set(partition["final"])
    baseline = json.load(open(BASELINE_PATH, encoding="utf-8"))
    scoring_report = json.load(open(os.path.join(SCORING_DIR, "scoring_report.json"), encoding="utf-8"))

    tuning_rows, selected = [], {}
    data = {}
    for m in MODELS:
        val = load_scores(m, "val")
        test = load_scores(m, "test")
        tune = [r for r in test if r["record_index"] in tune_idx]
        final = [r for r in test if r["record_index"] in final_idx]
        data[m] = {"val": val, "tune": tune, "final": final}
        best = None
        for mode in MODES:
            for agg in AGGREGATIONS:
                col = f"{mode}_{agg}"
                for scheme in SCHEMES:
                    norm = Normaliser(val, col, scheme)
                    y, z = arrays(tune, norm, col)
                    c = counts(y, decide(tune, norm, col, PRIMARY))
                    row = {"model": m, "mode": mode, "aggregation": agg, "threshold_scheme": scheme,
                           "tune_roc_auc": float(roc_auc_score(y, z)),
                           "tune_pr_auc": float(average_precision_score(y, z)),
                           "tune_recall_p99": c["recall"], "tune_fpr_p99": c["fpr"],
                           "tune_precision_p99": c["precision"], "selected": False}
                    tuning_rows.append(row)
                    if better(row, best):
                        best = row
        best["selected"] = True
        selected[m] = best

    # ------------------------------------------------ final evaluation (once)
    summary, threshold_rows, per_group, per_attack, unk_rows, curves = [], [], [], [], [], {}
    thresholds = {}
    for m in MODELS:
        sel = selected[m]
        col = f"{sel['mode']}_{sel['aggregation']}"
        norm = Normaliser(data[m]["val"], col, sel["threshold_scheme"])
        thresholds[m] = {"column": col, "scheme": sel["threshold_scheme"], "groups": norm.stats}
        final = data[m]["final"]
        y, z = arrays(final, norm, col)
        lo, hi = auc_ci(y, z, rng)
        by_level = {}
        for level in PERCENTILES:
            c = counts(y, decide(final, norm, col, level))
            by_level[level] = c
            threshold_rows.append({"model": m, "threshold_level": level, **c})
        zb = balanced_cutoff(*arrays(data[m]["tune"], norm, col))
        bal = counts(y, z > zb)
        threshold_rows.append({"model": m, "threshold_level": "balanced_alert", **bal})
        thresholds[m]["balanced_alert"] = {
            "z_cutoff": zb, "chosen_on": "TUNE, max Youden's J",
            "score_cutoff_per_group": {g: st["median"] + zb * (st[PRIMARY] - st["median"])
                                       for g, st in norm.stats.items()}}
        tr = json.load(open(TRAINING_REPORTS[m], encoding="utf-8"))
        inf = scoring_report["models"][m]["inference"]["test"]
        p = by_level[PRIMARY]
        summary.append({
            "model": m, "mode": sel["mode"], "aggregation": sel["aggregation"],
            "threshold_scheme": sel["threshold_scheme"],
            "roc_auc": float(roc_auc_score(y, z)), "roc_auc_ci95_low": lo, "roc_auc_ci95_high": hi,
            "pr_auc": float(average_precision_score(y, z)),
            "fpr_at_tpr_1": fpr_at_tpr(y, z, 1.0), "fpr_at_tpr_0_95": fpr_at_tpr(y, z, 0.95),
            "attacks_caught_p99": p["tp"], "attacks_total": p["tp"] + p["fn"],
            "false_alarms_p99": p["fp"], "benign_total": p["fp"] + p["tn"],
            "precision_p99": p["precision"], "recall_p99": p["recall"], "f1_p99": p["f1"], "fpr_p99": p["fpr"],
            "balanced_z_cutoff": zb,
            "attacks_caught_balanced": bal["tp"], "normal_passed_balanced": bal["tn"],
            "false_alarms_balanced": bal["fp"],
            "precision_balanced": bal["precision"], "recall_balanced": bal["recall"],
            "f1_balanced": bal["f1"], "fpr_balanced": bal["fpr"],
            "balanced_accuracy": (bal["recall"] + 1 - bal["fpr"]) / 2,
            "oracle_best_f1_on_final": oracle_best_f1(y, z),
            "baseline_roc_auc": baseline["models"][m]["roc_auc"],
            "baseline_recall_p99": baseline["models"][m]["recall"],
            "baseline_attacks_caught_p99": baseline["models"][m]["tp"],
            "train_time_seconds": tr["train_time_seconds"],
            "inference_ms_per_1000_windows": inf[f"{sel['mode']}_ms_per_1000_windows"],
            "n_parameters": scoring_report["models"][m]["n_parameters"],
            "model_file_kb": scoring_report["models"][m]["model_file_kb"],
        })
        curves[m] = {"y": y, "z": z, "cm": p, "balanced": bal}

        for level_name, key in (("dataset", "dataset"), ("source", "source")):
            for g in sorted({r[key] for r in final}):
                if level_name == "source" and "/" not in g:
                    continue
                rows = [r for r in final if r[key] == g]
                gy, gz = arrays(rows, norm, col)
                c = counts(gy, decide(rows, norm, col, PRIMARY))
                cb = counts(gy, gz > zb)
                per_group.append({"model": m, "level": level_name, "group": g,
                                  "n_benign": int((gy == 0).sum()), "n_attack": int(gy.sum()),
                                  "roc_auc": safe_auc(gy, gz), "attacks_caught_p99": c["tp"],
                                  "recall_p99": c["recall"], "fpr_p99": c["fpr"],
                                  "precision_p99": c["precision"], "f1_p99": c["f1"],
                                  "attacks_caught_balanced": cb["tp"], "normal_passed_balanced": cb["tn"],
                                  "recall_balanced": cb["recall"], "fpr_balanced": cb["fpr"],
                                  "baseline_roc_auc": baseline["models"][m]["per_dataset"].get(g, {}).get("roc_auc", ""),
                                  "small_group": bool(gy.sum() < SMALL_GROUP)})

        for ds, at in sorted({(r["dataset"], r["attack_type"]) for r in final if r["y"] == 1}):
            att = [r for r in final if r["y"] == 1 and r["attack_type"] == at]
            ben = [r for r in final if r["y"] == 0 and r["dataset"] == ds]
            ay, az = arrays(ben + att, norm, col)
            row = {"model": m, "dataset": ds, "attack_type": at, "n_attack": len(att),
                   "roc_auc_vs_same_dataset_benign": safe_auc(ay, az),
                   "median_z": float(np.median(az[ay == 1]))}
            for level in PERCENTILES:
                row[f"recall_{level}"] = float(decide(att, norm, col, level).mean())
            row["recall_balanced"] = float(np.mean(az[ay == 1] > zb))
            row["small_group"] = len(att) < SMALL_GROUP
            per_attack.append(row)

        for subset, keep in (("all_attacks", lambda r: True),
                             ("attacks_with_no_unk", lambda r: r["unk_rate"] == 0),
                             ("attacks_with_unk", lambda r: r["unk_rate"] > 0)):
            rows = [r for r in final if r["y"] == 0 or keep(r)]
            sy, sz = arrays(rows, norm, col)
            unk_rows.append({"model": m, "attack_subset": subset, "n_attack": int(sy.sum()),
                             "roc_auc": safe_auc(sy, sz),
                             "recall_p99": counts(sy, decide(rows, norm, col, PRIMARY))["recall"]})

    # ------------------------------------------------ best model (chosen on TUNE)
    ranked = sorted(MODELS, key=lambda m: -selected[m]["tune_roc_auc"])
    best_m = ranked[0]
    for m in ranked[1:]:
        if better(selected[m], selected[best_m]):
            best_m = m
    sel = selected[best_m]
    fin = next(s for s in summary if s["model"] == best_m)
    best_model = {
        "best_model": best_m,
        "chosen_on": "TUNE part of test (split/test_partition.json) — FINAL not used for any choice",
        "selection_rule": f"highest TUNE ROC-AUC; within {AUC_TIE}, higher TUNE recall at p99",
        "tune_roc_auc_per_model": {m: selected[m]["tune_roc_auc"] for m in MODELS},
        "configuration": {"mode": sel["mode"], "aggregation": sel["aggregation"],
                          "threshold_scheme": sel["threshold_scheme"], "score_column": f"{sel['mode']}_{sel['aggregation']}"},
        "final_results": {k: fin[k] for k in ("roc_auc", "pr_auc", "attacks_caught_p99", "attacks_total",
                                              "false_alarms_p99", "benign_total", "precision_p99",
                                              "recall_p99", "f1_p99", "fpr_p99",
                                              "attacks_caught_balanced", "normal_passed_balanced",
                                              "false_alarms_balanced", "precision_balanced",
                                              "recall_balanced", "f1_balanced", "fpr_balanced",
                                              "balanced_accuracy")},
        "for_layer5": {
            "model_file": scoring_report["models"][best_m]["model_file"],
            "score_files": {s: f"scoring/{best_m}_{s}_record_scores.csv" for s in ("val", "test")},
            "thresholds": thresholds[best_m],
            "note": "z = (score - median) / (p99 - median) per dataset from benign val. "
                    "Kill threshold: z > 1 (p99, ~1% false alarms). Alert threshold: "
                    "z > thresholds.balanced_alert.z_cutoff (balanced, chosen on TUNE).",
        },
    }

    print("Writing evaluation outputs...")
    write_csv("tuning_results.csv", tuning_rows)
    write_json("thresholds.json", {"source": "benign val record scores; flag when score > threshold",
                                   "primary": PRIMARY,
                                   "operating_points": {"kill": f"{PRIMARY} of benign val (~1% false alarms)",
                                                        "balanced_alert": "max Youden's J on TUNE"},
                                   "models": thresholds})
    write_csv("model_summary.csv", summary)
    write_csv("threshold_metrics.csv", threshold_rows)
    write_csv("per_dataset_metrics.csv", per_group)
    write_csv("per_attack_type_metrics.csv", per_attack)
    write_csv("unk_analysis.csv", unk_rows)
    write_json("best_model.json", best_model)
    write_json("evaluation_report.json", {
        "partition": {"tune_records": len(tune_idx), "final_records": len(final_idx)},
        "final_composition": {"benign": int((curves[MODELS[0]]["y"] == 0).sum()),
                              "attack": int(curves[MODELS[0]]["y"].sum())},
        "candidates_per_model": len(MODES) * len(AGGREGATIONS) * len(SCHEMES),
        "selected": {m: {k: selected[m][k] for k in ("mode", "aggregation", "threshold_scheme",
                                                     "tune_roc_auc", "tune_recall_p99")} for m in MODELS},
        "percentiles": PERCENTILES, "primary_threshold": PRIMARY,
        "bootstrap_resamples_for_auc_ci": N_BOOTSTRAP, "seed": SEED,
        "peak_memory_mb_during_scoring": scoring_report.get("peak_memory_mb"),
        "best_model": best_m,
    })

    # ------------------------------------------------ plots
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
                         "axes.titlecolor": INK, "axes.titlesize": 11})
    n_b, n_a = int((curves[MODELS[0]]["y"] == 0).sum()), int(curves[MODELS[0]]["y"].sum())

    fig, ax = plt.subplots(figsize=(6, 5.2))
    style_axes(ax)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.2, linestyle="--")
    for m in MODELS:
        fpr, tpr, _ = roc_curve(curves[m]["y"], curves[m]["z"])
        s = next(x for x in summary if x["model"] == m)
        ax.plot(fpr, tpr, color=COLORS[m], linewidth=2,
                label=f"{MODEL_LABELS[m]}  AUC {s['roc_auc']:.3f} (v1 {s['baseline_roc_auc']:.3f})")
        b = curves[m]["balanced"]
        ax.plot(b["fpr"], b["recall"], "o", markersize=8, color=COLORS[m], markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=5)
    ax.text(0.98, 0.30, "● balanced alert threshold (chosen on TUNE)", color=INK_2, fontsize=8,
            ha="right", transform=ax.transAxes)
    ax.set_xlabel("False positive rate (benign flagged)")
    ax.set_ylabel("True positive rate (attacks caught)")
    ax.set_title(f"ROC curves — final test ({n_b} benign, {n_a} attack)", loc="left")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK, fontsize=8.5)
    save(fig, "roc_curves.png")

    fig, ax = plt.subplots(figsize=(6, 5.2))
    style_axes(ax)
    base_rate = curves[MODELS[0]]["y"].mean()
    ax.axhline(base_rate, color=GRID, linewidth=1.2, linestyle="--")
    ax.text(0.01, base_rate + 0.01, f"random = {base_rate:.2f}", color=INK_2, fontsize=8)
    for m in MODELS:
        p, r, _ = precision_recall_curve(curves[m]["y"], curves[m]["z"])
        s = next(x for x in summary if x["model"] == m)
        ax.plot(r, p, color=COLORS[m], linewidth=2, label=f"{MODEL_LABELS[m]}  AP {s['pr_auc']:.3f}")
    ax.set_xlabel("Recall (attacks caught)")
    ax.set_ylabel("Precision (alarms that are real attacks)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision–recall curves — final test", loc="left")
    ax.legend(frameon=False, loc="lower left", labelcolor=INK)
    save(fig, "pr_curves.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, m in zip(axes, MODELS):
        cm = curves[m]["cm"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(mat / mat.sum(axis=1, keepdims=True), cmap=CM_CMAP, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                frac = mat[i, j] / mat[i].sum()
                ax.text(j, i, f"{mat[i, j]}\n({frac:.0%})", ha="center", va="center",
                        color="#ffffff" if frac > 0.55 else INK, fontsize=11)
        ax.set_xticks([0, 1], ["pred. benign", "pred. attack"])
        ax.set_yticks([0, 1], ["benign", "attack"])
        ax.tick_params(colors=INK_2, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{MODEL_LABELS[m]}\n{thresholds[m]['column']}, {thresholds[m]['scheme']} p99",
                     loc="left", fontsize=10)
    fig.suptitle("Confusion matrices — final test, p99 threshold of benign val",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    save(fig, "confusion_matrices.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, m in zip(axes, MODELS):
        style_axes(ax)
        y, z = curves[m]["y"], curves[m]["z"]
        bins = np.linspace(np.percentile(z, 0.5), np.percentile(z, 99.5), 50)
        ax.hist(z[y == 0], bins=bins, histtype="step", linewidth=2, color=BENIGN_COLOR, density=True, label="benign")
        ax.hist(z[y == 1], bins=bins, histtype="step", linewidth=2, color=ATTACK_COLOR, density=True, label="attack")
        ax.axvline(1.0, color=INK_2, linewidth=1.2, linestyle="--")
        ax.text(1.0, ax.get_ylim()[1] * 0.95, " p99 threshold", color=INK_2, fontsize=8, va="top")
        ax.set_title(MODEL_LABELS[m], loc="left")
        ax.set_xlabel("normalised score z (1 = benign p99)")
        ax.set_yticks([])
    axes[0].legend(frameon=False, loc="upper right", labelcolor=INK, fontsize=8)
    fig.tight_layout()
    save(fig, "score_distributions.png")

    types = [(r["dataset"], r["attack_type"], r["n_attack"]) for r in per_attack if r["model"] == MODELS[0]]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(types) + 1.4))
    style_axes(ax)
    ax.grid(False, axis="y")
    h = 0.26
    for k, m in enumerate(MODELS):
        vals = [next(r["recall_p99"] for r in per_attack if r["model"] == m and r["attack_type"] == at
                     and r["dataset"] == ds) for ds, at, _ in types]
        ax.barh(np.arange(len(types)) + (k - 1) * (h + 0.02), vals, height=h, color=COLORS[m], label=MODEL_LABELS[m])
    ax.set_yticks(np.arange(len(types)),
                  [f"{ds} · {at}  (n={n}{', small' if n < SMALL_GROUP else ''})" for ds, at, n in types])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall at p99 threshold (share of attacks caught)")
    ax.set_title("Detection rate per attack type — final test", loc="left")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK, fontsize=8)
    save(fig, "per_attack_recall.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (key, base_key, title) in zip(axes, (("roc_auc", "baseline_roc_auc", "ROC-AUC"),
                                                  ("recall_p99", "baseline_recall_p99", "Recall at p99 (≈1% false alarms)"))):
        style_axes(ax)
        ax.grid(False, axis="x")
        x = np.arange(len(MODELS))
        before = [next(s[base_key] for s in summary if s["model"] == m) for m in MODELS]
        after = [next(s[key] for s in summary if s["model"] == m) for m in MODELS]
        ax.bar(x - 0.19, before, width=0.36, color=BASELINE_COLOR, label="v1 (original)")
        ax.bar(x + 0.19, after, width=0.36, color=[COLORS[m] for m in MODELS], label="v2 (improved)")
        for i in range(len(MODELS)):
            ax.text(x[i] - 0.19, before[i] + 0.01, f"{before[i]:.2f}", ha="center", fontsize=8, color=INK_2)
            ax.text(x[i] + 0.19, after[i] + 0.01, f"{after[i]:.2f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x, [MODEL_LABELS[m] for m in MODELS])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{title} — final test, v1 vs v2", loc="left")
    axes[0].legend(frameon=False, loc="upper left", labelcolor=INK, fontsize=8)
    fig.tight_layout()
    save(fig, "v1_vs_v2.png")

    print(f"\nBest model (chosen on TUNE): {best_m} — {best_model['configuration']}")
    for s in summary:
        print(f"  {s['model']:<5} {s['mode']}/{s['aggregation']}/{s['threshold_scheme']}: "
              f"AUC {s['roc_auc']:.3f} (v1 {s['baseline_roc_auc']:.3f})  "
              f"caught {s['attacks_caught_p99']}/{s['attacks_total']} (v1 {s['baseline_attacks_caught_p99']})  "
              f"false alarms {s['false_alarms_p99']}/{s['benign_total']}  F1 {s['f1_p99']:.3f}")
        print(f"        balanced (z>{s['balanced_z_cutoff']:.3f}): attacks {s['attacks_caught_balanced']}/{s['attacks_total']}  "
              f"normal passed {s['normal_passed_balanced']}/{s['benign_total']}  precision {s['precision_balanced']:.3f}  "
              f"F1 {s['f1_balanced']:.3f}  balanced acc {s['balanced_accuracy']:.3f}")


if __name__ == "__main__":
    main()
