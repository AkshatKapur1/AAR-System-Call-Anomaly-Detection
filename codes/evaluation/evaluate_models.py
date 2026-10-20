"""
Layer 4: evaluation & benchmarking of the HMM, GRU and LSTM anomaly scores.

Input : scoring/{hmm,gru,lstm}_{val,test}_record_scores.csv (Layer 3 output;
        column `score`, higher = more anomalous), scoring/scoring_report.json,
        models/*/training_report.json (training time).
Output: evaluation/ — thresholds, metric tables, plots, best_model.json.

Protocol (fixed before looking at test results):
  - Thresholds come ONLY from benign val scores: the 95th / 99th / 99.5th
    percentile per model (~5% / 1% / 0.5% expected false alarms). The 99th
    percentile is the primary operating point. Test labels are never used to
    set a threshold.
  - A record is flagged as an attack when score > threshold.
  - Threshold-free metrics (ROC-AUC, PR-AUC, FPR@TPR=1) compare the models
    independent of the threshold choice.
  - The "oracle" best-F1 threshold on test is reported only as an upper
    bound, clearly labelled, never used for decisions.
  - Best model: highest test ROC-AUC; if the top two are within 0.01 AUC,
    the higher recall at the primary threshold wins; then faster inference.
  - Records with no score (<7 syscalls, NaN) are excluded and counted.

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

MODELS = ["hmm", "gru", "lstm"]
MODEL_LABELS = {"hmm": "HMM (Track A)", "gru": "GRU (Track B)", "lstm": "LSTM (Track C)"}
TRAINING_REPORTS = {
    "hmm": os.path.join(REPO_ROOT, "models", "HMM", "training_report.json"),
    "gru": os.path.join(REPO_ROOT, "models", "gru", "results", "training_report.json"),
    "lstm": os.path.join(REPO_ROOT, "models", "lstm", "results", "training_report.json"),
}
PERCENTILES = {"p95": 95.0, "p99": 99.0, "p99_5": 99.5}
PRIMARY = "p99"
SCORE_COLUMNS = ["score_mean", "score_max", "score_top5"]
SMALL_GROUP = 20          # attack groups smaller than this are flagged
AUC_TIE = 0.01
N_BOOTSTRAP = 1000
SEED = 42

# Reference categorical palette (first three slots, validated all-pairs) and
# recessive ink/grid colours.
COLORS = {"hmm": "#2a78d6", "gru": "#eb6834", "lstm": "#1baf7a"}
BENIGN_COLOR, ATTACK_COLOR = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CM_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_seq", ["#f0f5fc", "#9ec5f4", "#3987e5", "#184f95"])


# ---------------------------------------------------------------- data

def load_scores(model, split):
    with open(os.path.join(SCORING_DIR, f"{model}_{split}_record_scores.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for col in SCORE_COLUMNS + ["score", "unk_rate"]:
            r[col] = float(r[col])
        r["y"] = 1 if r["label"] == "attack" else 0
        r["dataset"] = r["source"].split("/")[0]
    return rows


def valid(rows, col="score"):
    return [r for r in rows if not math.isnan(r[col])]


# ---------------------------------------------------------------- metrics

def confusion(y, s, thr):
    pred = s > thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
            "accuracy": (tp + tn) / len(y)}


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
    p, r, t = precision_recall_curve(y, s)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)[:-1]
    i = int(np.argmax(f1))
    return float(f1[i]), float(t[i])


def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


# ---------------------------------------------------------------- output helpers

def write_csv(name, rows):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)}")


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
    with open(os.path.join(SCORING_DIR, "scoring_report.json"), encoding="utf-8") as f:
        scoring_report = json.load(f)

    val = {m: load_scores(m, "val") for m in MODELS}
    test = {m: load_scores(m, "test") for m in MODELS}

    thresholds, summary, threshold_rows = {}, [], []
    per_dataset, per_attack, aggregation, unk_rows = [], [], [], []
    curves = {}
    excluded = {}

    for m in MODELS:
        v = np.array([r["score"] for r in valid(val[m])])
        t_rows = valid(test[m])
        excluded[m] = {"val": len(val[m]) - len(v), "test": len(test[m]) - len(t_rows)}
        y = np.array([r["y"] for r in t_rows])
        s = np.array([r["score"] for r in t_rows])

        thresholds[m] = {name: float(np.percentile(v, q)) for name, q in PERCENTILES.items()}

        # --- threshold-based metrics
        for name, thr in thresholds[m].items():
            row = {"model": m, "threshold_name": name, "threshold": thr,
                   "val_false_alarm_rate": float(np.mean(v > thr))}
            row.update(confusion(y, s, thr))
            threshold_rows.append(row)

        # --- threshold-free metrics
        auc = float(roc_auc_score(y, s))
        lo, hi = auc_ci(y, s, rng)
        best_f1, best_thr = oracle_best_f1(y, s)
        primary = confusion(y, s, thresholds[m][PRIMARY])
        train_time = json.load(open(TRAINING_REPORTS[m], encoding="utf-8"))["train_time_seconds"]
        model_info = scoring_report["models"][m]
        summary.append({
            "model": m,
            "roc_auc": auc, "roc_auc_ci95_low": lo, "roc_auc_ci95_high": hi,
            "pr_auc": float(average_precision_score(y, s)),
            "fpr_at_tpr_1": fpr_at_tpr(y, s, 1.0),
            "fpr_at_tpr_0_95": fpr_at_tpr(y, s, 0.95),
            "threshold_p99": thresholds[m][PRIMARY],
            "precision_p99": primary["precision"], "recall_p99": primary["recall"],
            "f1_p99": primary["f1"], "fpr_p99": primary["fpr"],
            "oracle_best_f1_on_test": best_f1, "oracle_threshold_on_test": best_thr,
            "train_time_seconds": train_time,
            "inference_ms_per_1000_windows": model_info["inference"]["test"]["ms_per_1000_windows"],
            "n_parameters": model_info["n_parameters"],
            "model_file_kb": model_info["model_file_kb"],
        })
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        curves[m] = {"fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec, "auc": auc,
                     "ap": summary[-1]["pr_auc"], "cm": primary, "val": v, "y": y, "s": s}

        # --- per dataset / per source (benign + attack from the same group)
        thr = thresholds[m][PRIMARY]
        for level, key in (("dataset", "dataset"), ("source", "source")):
            for group in sorted({r[key] for r in t_rows}):
                if level == "source" and "/" not in group:
                    continue  # ADFA-LD / PLAID already covered at dataset level
                g = [r for r in t_rows if r[key] == group]
                gy = np.array([r["y"] for r in g]); gs = np.array([r["score"] for r in g])
                c = confusion(gy, gs, thr)
                per_dataset.append({
                    "model": m, "level": level, "group": group,
                    "n_benign": int((gy == 0).sum()), "n_attack": int(gy.sum()),
                    "roc_auc": safe_auc(gy, gs), "recall_p99": c["recall"],
                    "fpr_p99": c["fpr"], "precision_p99": c["precision"], "f1_p99": c["f1"],
                    "small_group": bool(gy.sum() < SMALL_GROUP),
                })

        # --- per attack type (vs benign records of the same dataset)
        for (ds, at) in sorted({(r["dataset"], r["attack_type"]) for r in t_rows if r["y"] == 1}):
            att = np.array([r["score"] for r in t_rows if r["y"] == 1 and r["attack_type"] == at])
            ben = np.array([r["score"] for r in t_rows if r["y"] == 0 and r["dataset"] == ds])
            yy = np.r_[np.zeros(len(ben)), np.ones(len(att))]
            row = {"model": m, "dataset": ds, "attack_type": at, "n_attack": len(att),
                   "roc_auc_vs_same_dataset_benign": safe_auc(yy, np.r_[ben, att]),
                   "median_score": float(np.median(att))}
            for name, t_ in thresholds[m].items():
                row[f"recall_{name}"] = float(np.mean(att > t_))
            row["small_group"] = bool(len(att) < SMALL_GROUP)
            per_attack.append(row)

        # --- how window scores are aggregated (reported only; `score` = score_top5)
        for col in SCORE_COLUMNS:
            vv = np.array([r[col] for r in valid(val[m], col)])
            tt = valid(test[m], col)
            ty = np.array([r["y"] for r in tt]); ts = np.array([r[col] for r in tt])
            c = confusion(ty, ts, float(np.percentile(vv, PERCENTILES[PRIMARY])))
            aggregation.append({"model": m, "aggregation": col, "roc_auc": float(roc_auc_score(ty, ts)),
                                "recall_p99": c["recall"], "fpr_p99": c["fpr"], "f1_p99": c["f1"],
                                "primary": col == "score_top5"})

        # --- does detection rely only on unseen syscalls (<UNK>)?
        for subset, keep in (("all_attacks", lambda r: True),
                             ("attacks_with_no_unk", lambda r: r["unk_rate"] == 0),
                             ("attacks_with_unk", lambda r: r["unk_rate"] > 0)):
            sub = [r for r in t_rows if r["y"] == 0 or keep(r)]
            sy = np.array([r["y"] for r in sub]); ss = np.array([r["score"] for r in sub])
            c = confusion(sy, ss, thr)
            unk_rows.append({"model": m, "attack_subset": subset, "n_attack": int(sy.sum()),
                             "roc_auc": safe_auc(sy, ss), "recall_p99": c["recall"]})

    # ---------------------------------------------------------------- best model
    ranked = sorted(summary, key=lambda r: -r["roc_auc"])
    best, runner = ranked[0], ranked[1]
    if best["roc_auc"] - runner["roc_auc"] < AUC_TIE:
        pair = sorted([best, runner], key=lambda r: (-r["recall_p99"], r["inference_ms_per_1000_windows"]))
        best, reason = pair[0], (f"ROC-AUC within {AUC_TIE} of {pair[1]['model']}; "
                                 "won on recall at the p99 threshold, then inference speed")
    else:
        reason = f"highest ROC-AUC (margin {best['roc_auc'] - runner['roc_auc']:.3f} over {runner['model']})"
    m = best["model"]
    best_model = {
        "best_model": m,
        "reason": reason,
        "selection_rule": f"highest test ROC-AUC; if within {AUC_TIE}, higher recall at p99, then faster inference",
        "roc_auc": best["roc_auc"], "pr_auc": best["pr_auc"],
        "threshold_p99": thresholds[m][PRIMARY],
        "precision_p99": best["precision_p99"], "recall_p99": best["recall_p99"], "f1_p99": best["f1_p99"],
        "fpr_p99": best["fpr_p99"],
        "for_layer5": {
            "val_scores": f"scoring/{m}_val_record_scores.csv",
            "test_scores": f"scoring/{m}_test_record_scores.csv",
            "model_file": scoring_report["models"][m]["model_file"],
            "starting_threshold": thresholds[m][PRIMARY],
        },
        "ranking_by_roc_auc": [r["model"] for r in ranked],
    }

    # ---------------------------------------------------------------- write tables
    print("Writing evaluation outputs...")
    with open(os.path.join(OUT_DIR, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump({"source": "benign val record scores (scoring/*_val_record_scores.csv, column `score`)",
                   "rule": "flag as attack when score > threshold",
                   "primary": PRIMARY, "thresholds": thresholds}, f, indent=2)
    write_csv("model_summary.csv", summary)
    write_csv("threshold_metrics.csv", threshold_rows)
    write_csv("per_dataset_metrics.csv", per_dataset)
    write_csv("per_attack_type_metrics.csv", per_attack)
    write_csv("aggregation_comparison.csv", aggregation)
    write_csv("unk_analysis.csv", unk_rows)
    with open(os.path.join(OUT_DIR, "best_model.json"), "w", encoding="utf-8") as f:
        json.dump(best_model, f, indent=2)
    report = {
        "test_records": {mm: len(test[mm]) for mm in MODELS},
        "excluded_nan_scores": excluded,
        "test_composition": {
            "benign": int((curves[MODELS[0]]["y"] == 0).sum()),
            "attack": int(curves[MODELS[0]]["y"].sum()),
        },
        "percentiles": PERCENTILES, "primary_threshold": PRIMARY,
        "bootstrap_resamples_for_auc_ci": N_BOOTSTRAP, "seed": SEED,
        "peak_memory_mb_during_scoring": scoring_report.get("peak_memory_mb"),
        "best_model": best_model["best_model"],
    }
    with open(os.path.join(OUT_DIR, "evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("  wrote evaluation/thresholds.json, best_model.json, evaluation_report.json")

    # ---------------------------------------------------------------- plots
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
                         "axes.titlecolor": INK, "axes.titlesize": 11})

    # ROC curves
    fig, ax = plt.subplots(figsize=(6, 5.2))
    style_axes(ax)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.2, linestyle="--")
    for mm in MODELS:
        c = curves[mm]
        ax.plot(c["fpr"], c["tpr"], color=COLORS[mm], linewidth=2,
                label=f"{MODEL_LABELS[mm]}  AUC = {c['auc']:.3f}")
    ax.set_xlabel("False positive rate (benign flagged)")
    ax.set_ylabel("True positive rate (attacks caught)")
    ax.set_title("ROC curves — test set (834 benign, 1,026 attack)", loc="left")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    save(fig, "roc_curves.png")

    # PR curves
    fig, ax = plt.subplots(figsize=(6, 5.2))
    style_axes(ax)
    base_rate = curves[MODELS[0]]["y"].mean()
    ax.axhline(base_rate, color=GRID, linewidth=1.2, linestyle="--")
    ax.text(0.01, base_rate + 0.01, f"random = {base_rate:.2f}", color=INK_2, fontsize=8)
    for mm in MODELS:
        c = curves[mm]
        ax.plot(c["rec"], c["prec"], color=COLORS[mm], linewidth=2,
                label=f"{MODEL_LABELS[mm]}  AP = {c['ap']:.3f}")
    ax.set_xlabel("Recall (attacks caught)")
    ax.set_ylabel("Precision (alarms that are real attacks)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision–recall curves — test set", loc="left")
    ax.legend(frameon=False, loc="lower left", labelcolor=INK)
    save(fig, "pr_curves.png")

    # Confusion matrices at the primary threshold
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, mm in zip(axes, MODELS):
        cm = curves[mm]["cm"]
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
        ax.set_title(f"{MODEL_LABELS[mm]}\nthreshold {thresholds[mm][PRIMARY]:.2f} (val p99)", loc="left", fontsize=10)
    fig.suptitle("Confusion matrices at the primary threshold (99th pct of benign val)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    save(fig, "confusion_matrices.png")

    # Score distributions: benign vs attack, threshold marked
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, mm in zip(axes, MODELS):
        style_axes(ax)
        c = curves[mm]
        bins = np.linspace(np.percentile(c["s"], 0.5), np.percentile(c["s"], 99.5), 50)
        ax.hist(c["s"][c["y"] == 0], bins=bins, histtype="step", linewidth=2, color=BENIGN_COLOR,
                density=True, label="benign (test)")
        ax.hist(c["s"][c["y"] == 1], bins=bins, histtype="step", linewidth=2, color=ATTACK_COLOR,
                density=True, label="attack (test)")
        thr = thresholds[mm][PRIMARY]
        ax.axvline(thr, color=INK_2, linewidth=1.2, linestyle="--")
        ax.text(thr, ax.get_ylim()[1] * 0.95, " p99 threshold", color=INK_2, fontsize=8, va="top")
        ax.set_title(MODEL_LABELS[mm], loc="left")
        ax.set_xlabel("record anomaly score (higher = more anomalous)")
        ax.set_yticks([])
    axes[0].legend(frameon=False, loc="upper left", labelcolor=INK, fontsize=8)
    fig.tight_layout()
    save(fig, "score_distributions.png")

    # Recall per attack type at the primary threshold
    types = [(r["dataset"], r["attack_type"], r["n_attack"]) for r in per_attack if r["model"] == MODELS[0]]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(types) + 1.4))
    style_axes(ax)
    ax.grid(True, axis="x", color=GRID); ax.grid(False, axis="y")
    h = 0.26
    for k, mm in enumerate(MODELS):
        vals = [next(r["recall_p99"] for r in per_attack
                     if r["model"] == mm and r["attack_type"] == at and r["dataset"] == ds)
                for ds, at, _ in types]
        ax.barh(np.arange(len(types)) + (k - 1) * (h + 0.02), vals, height=h,
                color=COLORS[mm], label=MODEL_LABELS[mm])
    ax.set_yticks(np.arange(len(types)),
                  [f"{ds} · {at}  (n={n}{', small' if n < SMALL_GROUP else ''})" for ds, at, n in types])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall at p99 threshold (share of attacks caught)")
    ax.set_title("Detection rate per attack type", loc="left")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK, fontsize=8)
    save(fig, "per_attack_recall.png")

    print(f"\nBest model: {best_model['best_model']} — {best_model['reason']}")
    for r in summary:
        print(f"  {r['model']:<5} AUC={r['roc_auc']:.3f} [{r['roc_auc_ci95_low']:.3f},{r['roc_auc_ci95_high']:.3f}] "
              f"PR-AUC={r['pr_auc']:.3f} FPR@TPR1={r['fpr_at_tpr_1']:.3f} "
              f"P/R/F1@p99={r['precision_p99']:.3f}/{r['recall_p99']:.3f}/{r['f1_p99']:.3f}")


if __name__ == "__main__":
    main()
