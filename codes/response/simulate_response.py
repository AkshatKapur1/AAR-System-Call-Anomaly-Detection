"""
Layer 5: simulate the confidence-gated, hysteresis-based kill switch
(codes/response/kill_switch.py) on every val/test recording, replayed window
by window in order, as if each recording were a live process.

Input : evaluation/best_model.json (Layer 4: LSTM, stream mode),
        split/{val,test}.jsonl.gz, split/test_partition.json.
Output: response/ — calibration, tuning results, kill log, metrics, plots.

Protocol (same discipline as Layer 4 — FINAL is never used for a choice):
  1. Window scores: the best model in stream mode (each 7-syscall window
     scored with the whole preceding recording as context).
  2. Normalisation per dataset from benign val WINDOWS: z = (s - median) /
     (p99 - median).
     Confidence: running mean of z since the process started (alpha = 0,
     the online form of Layer 4's `mean` aggregation) or an EWMA (alpha > 0).
  3. For each (alpha, sustain) the kill threshold is calibrated per dataset
     on benign val so that ~1% of benign val recordings would ever be killed.
  4. (alpha, sustain) chosen on TUNE: most attacks killed, subject to <= 2%
     benign TUNE recordings killed; ties -> fewer syscalls before the kill.
  5. Alert threshold: max Youden's J (attacks alerted % + benign not alerted %)
     on TUNE; alert_low = alert_high - HYSTERESIS_GAP (fixed in advance).
  6. Alert escalation: kill a process whose alert has stayed on for more
     than E windows in a row. E per dataset = a percentile of the longest
     continuous alert over benign val processes; the percentile (or no
     escalation) is chosen on TUNE with the same rule as step 4.
  7. FINAL: kills (by reason), false kills, alerts, syscalls executed before the kill,
     alert flapping with vs without hysteresis, per dataset / attack type,
     against Layer 4's whole-recording decisions.

Run from anywhere:  python codes/response/simulate_response.py
"""
import csv
import gzip
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "models"))
sys.path.insert(0, os.path.dirname(__file__))
import sequence_model_common as rnn  # noqa: E402
from kill_switch import KillSwitch, Thresholds, demo  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "response")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
N = 7
ALPHAS = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]   # 0 = running mean since process start
SUSTAINS = [1, 5, 10, 25, 50, 100]
KILL_PERCENTILE = 99.0          # ~1% of benign val recordings ever killed
MAX_TUNE_FALSE_KILL = 0.02
HYSTERESIS_GAP = 0.25           # alert_low = alert_high - gap (z units)
ESCALATE_PERCENTILES = [None, 99.5, 99, 98, 97, 95, 90]   # None = no escalation
NEG = -1e9

COLORS = {"attack": "#eb6834", "benign": "#2a78d6"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
KILL_COLOR, ALERT_COLOR = "#d03b3b", "#c98500"


# ---------------------------------------------------------------- data

def load_records(split):
    out = []
    with gzip.open(os.path.join(REPO_ROOT, "split", f"{split}.jsonl.gz"), "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            if len(r["token_ids"]) >= N:
                out.append({"record_index": i, "recording_id": r["recording_id"], "label": r["label"],
                            "dataset": r["source"].split("/")[0], "source": r["source"],
                            "attack_type": r["attack_type"] or "", "token_ids": r["token_ids"]})
    return out


def window_scores(model, records):
    nll = rnn.token_nll_stream(model, [r["token_ids"] for r in records])
    for r, t in zip(records, nll):
        r["s"] = rnn.stream_window_scores(t)


# ---------------------------------------------------------------- vectorised switch

def confidence(z, alpha):
    if alpha == 0:
        return np.cumsum(z) / np.arange(1, len(z) + 1)
    return lfilter([alpha], [1.0, -(1.0 - alpha)], z)


def sustained_peak(c, k):
    """Largest level the confidence stayed above for k consecutive windows."""
    if len(c) < k:
        return NEG
    return float(sliding_window_view(c, k).min(axis=1).max()) if k > 1 else float(c.max())


def kill_window(c, k, thr):
    above = c > thr
    if k == 1:
        idx = np.flatnonzero(above)
        return int(idx[0]) if len(idx) else None
    run = 0
    for t, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= k:
            return t
    return None


def alert_raises(c, high, low):
    raises, on = 0, False
    for v in c:
        if not on and v > high:
            on, raises = True, raises + 1
        elif on and v < low:
            on = False
    return raises


def alert_runs(c, high, low):
    """Per window: how many consecutive windows the alert (with hysteresis)
    has been on, 0 while it is off. Same state machine as KillSwitch."""
    runs, on, run = np.zeros(len(c), dtype=np.int64), False, 0
    for t, v in enumerate(c):
        if not on and v > high:
            on = True
        elif on and v < low:
            on = False
        run = run + 1 if on else 0
        runs[t] = run
    return runs


def first_alert(c, high):
    idx = np.flatnonzero(c > high)
    return int(idx[0]) if len(idx) else None


def youden_cutoff(y, s):
    order = np.unique(s)
    best, best_j = order[0], -1
    pos, neg = y == 1, y == 0
    for i in range(len(order) - 1):
        cut = (order[i] + order[i + 1]) / 2
        j = np.mean(s[pos] > cut) - np.mean(s[neg] > cut)
        if j > best_j:
            best, best_j = cut, j
    return float(best)


# ---------------------------------------------------------------- output helpers

def write_csv(name, rows):
    with open(os.path.join(OUT_DIR, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  wrote response/{name}")


def write_json(name, obj):
    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote response/{name}")


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
    fig.savefig(os.path.join(PLOT_DIR, name), dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote response/plots/{name}")


# ---------------------------------------------------------------- main

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    best = json.load(open(os.path.join(REPO_ROOT, "evaluation", "best_model.json"), encoding="utf-8"))
    assert best["configuration"]["mode"] == "stream" and best["best_model"] in ("gru", "lstm"), \
        "Layer 5 expects a recurrent model in stream mode"
    model = rnn.load_model(os.path.join(REPO_ROOT, best["for_layer5"]["model_file"]))
    part = json.load(open(os.path.join(REPO_ROOT, "split", "test_partition.json"), encoding="utf-8"))
    tune_idx, final_idx = set(part["tune"]), set(part["final"])

    print(f"Scoring windows with {best['best_model'].upper()} (stream)...")
    t0 = time.time()
    val, test = load_records("val"), load_records("test")
    window_scores(model, val)
    window_scores(model, test)
    print(f"  {sum(len(r['s']) for r in val + test):,} windows in {time.time() - t0:.0f}s")
    tune = [r for r in test if r["record_index"] in tune_idx]
    final = [r for r in test if r["record_index"] in final_idx]
    datasets = sorted({r["dataset"] for r in val})

    # ---- 2. per-dataset window normalisation from benign val
    norm = {}
    for d in datasets:
        w = np.concatenate([r["s"] for r in val if r["dataset"] == d])
        norm[d] = {"median": float(np.median(w)), "p99": float(np.percentile(w, 99)), "windows": int(len(w))}
    for r in val + test:
        st = norm[r["dataset"]]
        r["z"] = (r["s"] - st["median"]) / (st["p99"] - st["median"])

    # ---- 3-4. calibrate kill thresholds on val, choose (alpha, sustain) on TUNE
    tuning, best_cfg = [], None
    for a in ALPHAS:
        conf = {id(r): confidence(r["z"], a) for r in val + tune}
        for k in SUSTAINS:
            kill_thr = {}
            for d in datasets:
                peaks = np.array([sustained_peak(conf[id(r)], k) for r in val if r["dataset"] == d])
                kill_thr[d] = float(np.percentile(peaks[peaks > NEG], KILL_PERCENTILE))
            killed = lambda rs: np.array([sustained_peak(conf[id(r)], k) > kill_thr[r["dataset"]] for r in rs])
            tk = killed(tune)
            ty = np.array([r["label"] == "attack" for r in tune])
            times = [kill_window(conf[id(r)], k, kill_thr[r["dataset"]]) + N
                     for r, kk in zip(tune, tk) if kk and r["label"] == "attack"]
            row = {"alpha": a, "sustain": k,
                   "val_benign_false_kill_rate": float(killed(val).mean()),
                   "tune_attacks_killed": int(tk[ty].sum()), "tune_attacks": int(ty.sum()),
                   "tune_attack_kill_rate": float(tk[ty].mean()),
                   "tune_benign_false_kills": int(tk[~ty].sum()), "tune_benign": int((~ty).sum()),
                   "tune_benign_false_kill_rate": float(tk[~ty].mean()),
                   "tune_median_syscalls_before_kill": float(np.median(times)) if times else float("nan"),
                   "selected": False}
            row["_kill_thr"] = kill_thr
            tuning.append(row)
            ok = row["tune_benign_false_kill_rate"] <= MAX_TUNE_FALSE_KILL
            if ok and (best_cfg is None or row["tune_attack_kill_rate"] > best_cfg["tune_attack_kill_rate"]
                       or (row["tune_attack_kill_rate"] == best_cfg["tune_attack_kill_rate"]
                           and row["tune_median_syscalls_before_kill"] < best_cfg["tune_median_syscalls_before_kill"])):
                best_cfg = row
    best_cfg["selected"] = True
    alpha, sustain, kill_thr = best_cfg["alpha"], best_cfg["sustain"], best_cfg["_kill_thr"]
    print(f"Selected alpha={alpha}, sustain={sustain}: TUNE kills {best_cfg['tune_attacks_killed']}/"
          f"{best_cfg['tune_attacks']}, false kills {best_cfg['tune_benign_false_kills']}/{best_cfg['tune_benign']}")

    # ---- 5. alert threshold on TUNE (Youden's J on peak confidence)
    for r in val + test:
        r["c"] = confidence(r["z"], alpha)
    ty = np.array([r["label"] == "attack" for r in tune], dtype=int)
    alert_high = youden_cutoff(ty, np.array([r["c"].max() for r in tune]))
    alert_low = alert_high - HYSTERESIS_GAP

    # ---- 6. alert escalation: calibrate on benign val, choose on TUNE
    for r in val + test:
        r["runs"] = alert_runs(r["c"], alert_high, alert_low)
        r["conf_kill"] = kill_window(r["c"], sustain, kill_thr[r["dataset"]])

    def escalation_window(r, esc):
        if not esc:
            return None
        idx = np.flatnonzero(r["runs"] > esc[r["dataset"]])
        return int(idx[0]) if len(idx) else None

    def kill_at(r, esc):
        times = [t for t in (r["conf_kill"], escalation_window(r, esc)) if t is not None]
        return min(times) if times else None

    escalation_rows, best_esc = [], None
    for pct in ESCALATE_PERCENTILES:
        esc = None if pct is None else {
            d: float(max(sustain, round(np.percentile([r["runs"].max() for r in val if r["dataset"] == d], pct))))
            for d in datasets}
        k_t = [kill_at(r, esc) for r in tune]
        att = [t for r, t in zip(tune, k_t) if r["label"] == "attack"]
        ben = [t for r, t in zip(tune, k_t) if r["label"] == "benign"]
        row = {"escalate_percentile": "off" if pct is None else pct,
               "escalate_after_windows": "" if esc is None else "; ".join(f"{d} {v:.0f}" for d, v in esc.items()),
               "val_benign_kill_rate": float(np.mean([kill_at(r, esc) is not None for r in val])),
               "tune_attacks_killed": sum(t is not None for t in att), "tune_attacks": len(att),
               "tune_benign_killed": sum(t is not None for t in ben), "tune_benign": len(ben),
               "tune_median_syscalls_before_kill": float(np.median([t + N for t in att if t is not None])),
               "selected": False}
        row["tune_benign_kill_rate"] = row["tune_benign_killed"] / row["tune_benign"]
        row["_esc"] = esc
        escalation_rows.append(row)
        if row["tune_benign_kill_rate"] <= MAX_TUNE_FALSE_KILL and (
                best_esc is None or row["tune_attacks_killed"] > best_esc["tune_attacks_killed"]
                or (row["tune_attacks_killed"] == best_esc["tune_attacks_killed"]
                    and row["tune_median_syscalls_before_kill"] < best_esc["tune_median_syscalls_before_kill"])):
            best_esc = row
    best_esc["selected"] = True
    escalate = best_esc["_esc"]
    print(f"Escalation: percentile {best_esc['escalate_percentile']} -> TUNE kills {best_esc['tune_attacks_killed']}/"
          f"{best_esc['tune_attacks']}, normal killed {best_esc['tune_benign_killed']}/{best_esc['tune_benign']}")

    # ---- 7. FINAL
    log = []
    for r in final:
        c, d = r["c"], r["dataset"]
        kw = kill_at(r, escalate)
        reason = "" if kw is None else ("confidence" if kw == r["conf_kill"] else "escalation")
        fa = first_alert(c, alert_high)
        log.append({
            "record_index": r["record_index"], "recording_id": r["recording_id"], "dataset": d,
            "source": r["source"], "label": r["label"], "attack_type": r["attack_type"],
            "n_syscalls": len(r["token_ids"]), "n_windows": len(c),
            "alerted": fa is not None, "first_alert_syscall": (fa + N) if fa is not None else "",
            "alert_raises_with_hysteresis": alert_raises(c, alert_high, alert_low),
            "alert_raises_without_hysteresis": alert_raises(c, alert_high, alert_high),
            "killed": kw is not None, "kill_reason": reason, "kill_syscall": (kw + N) if kw is not None else "",
            "killed_without_escalation": r["conf_kill"] is not None,
            "share_of_recording_before_kill": ((kw + N) / len(r["token_ids"])) if kw is not None else "",
            "peak_confidence": float(c.max()),
        })

    def summarise(rows):
        att = [r for r in rows if r["label"] == "attack"]
        ben = [r for r in rows if r["label"] == "benign"]
        kills = [r["kill_syscall"] for r in att if r["killed"]]
        share = [r["share_of_recording_before_kill"] for r in att if r["killed"]]
        return {
            "attacks": len(att), "benign": len(ben),
            "attacks_killed": sum(r["killed"] for r in att), "benign_killed": sum(r["killed"] for r in ben),
            "attacks_killed_by_escalation": sum(r["kill_reason"] == "escalation" for r in att),
            "benign_killed_by_escalation": sum(r["kill_reason"] == "escalation" for r in ben),
            "attacks_killed_without_escalation": sum(r["killed_without_escalation"] for r in att),
            "benign_killed_without_escalation": sum(r["killed_without_escalation"] for r in ben),
            "attack_kill_rate": (sum(r["killed"] for r in att) / len(att)) if att else float("nan"),
            "benign_false_kill_rate": (sum(r["killed"] for r in ben) / len(ben)) if ben else float("nan"),
            "attacks_alerted": sum(r["alerted"] for r in att), "benign_alerted": sum(r["alerted"] for r in ben),
            "attack_alert_rate": (sum(r["alerted"] for r in att) / len(att)) if att else float("nan"),
            "benign_alert_rate": (sum(r["alerted"] for r in ben) / len(ben)) if ben else float("nan"),
            "median_syscalls_before_kill": float(np.median(kills)) if kills else float("nan"),
            "median_share_of_recording_before_kill": float(np.median(share)) if share else float("nan"),
        }

    overall = summarise(log)
    benign_rows = [r for r in log if r["label"] == "benign"]
    flap = {"benign_alert_raises_with_hysteresis": sum(r["alert_raises_with_hysteresis"] for r in benign_rows),
            "benign_alert_raises_without_hysteresis": sum(r["alert_raises_without_hysteresis"] for r in benign_rows),
            "all_alert_raises_with_hysteresis": sum(r["alert_raises_with_hysteresis"] for r in log),
            "all_alert_raises_without_hysteresis": sum(r["alert_raises_without_hysteresis"] for r in log)}
    per_dataset = [{"dataset": d, **summarise([r for r in log if r["dataset"] == d])} for d in datasets]
    per_dataset += [{"dataset": s, **summarise([r for r in log if r["source"] == s])}
                    for s in sorted({r["source"] for r in log}) if "/" in s]
    per_attack = []
    for d, at in sorted({(r["dataset"], r["attack_type"]) for r in log if r["label"] == "attack"}):
        rows = [r for r in log if r["label"] == "attack" and r["attack_type"] == at and r["dataset"] == d]
        s = summarise(rows)
        per_attack.append({"dataset": d, "attack_type": at, "n_attack": len(rows),
                           "killed": s["attacks_killed"], "kill_rate": s["attack_kill_rate"],
                           "killed_by_escalation": s["attacks_killed_by_escalation"],
                           "killed_without_escalation": s["attacks_killed_without_escalation"],
                           "alerted": s["attacks_alerted"], "alert_rate": s["attack_alert_rate"],
                           "median_syscalls_before_kill": s["median_syscalls_before_kill"]})

    # ---- consistency: vectorised simulation == KillSwitch class, record by record
    mismatches = 0
    for r, row in list(zip(final, log))[:300]:
        st = norm[r["dataset"]]
        ks = KillSwitch(alpha, sustain, Thresholds(st["median"], st["p99"], alert_high, alert_low,
                                                   kill_thr[r["dataset"]],
                                                   int(escalate[r["dataset"]]) if escalate else 0))
        raises = 0
        for s in r["s"]:
            ev = ks.update(float(s))
            raises += ev == "alert"
        kill_ok = (ks.killed == row["killed"]) and (not ks.killed or ks.window + N == row["kill_syscall"])
        mismatches += not kill_ok
    assert mismatches == 0, f"{mismatches} KillSwitch/simulation mismatches"

    # ---- live demo of the action path on one killed attack
    demo_rec = next((r for r, row in zip(final, log) if row["killed"] and row["label"] == "attack"), None)
    demo_result = None
    if demo_rec is not None:
        st = norm[demo_rec["dataset"]]
        demo_result = demo(demo_rec["s"], alpha, sustain,
                           Thresholds(st["median"], st["p99"], alert_high, alert_low, kill_thr[demo_rec["dataset"]],
                                      int(escalate[demo_rec["dataset"]]) if escalate else 0))
        demo_result = {"recording_id": demo_rec["recording_id"], "attack_type": demo_rec["attack_type"],
                       **{k: v for k, v in demo_result.items() if k != "events"},
                       "events": [list(e) for e in demo_result["events"]]}

    # ---- Layer 4 comparison (whole-recording decisions on the same FINAL records)
    l4 = {r["model"]: r for r in csv.DictReader(open(os.path.join(REPO_ROOT, "evaluation", "model_summary.csv"),
                                                     encoding="utf-8"))}[best["best_model"]]

    print("Writing response outputs...")
    for row in tuning:
        row.pop("_kill_thr")
    write_csv("tuning_results.csv", tuning)
    for row in escalation_rows:
        row.pop("_esc")
    write_csv("escalation_tuning.csv", escalation_rows)
    write_json("calibration.json", {
        "model": best["best_model"], "mode": "stream",
        "window_normalisation_from_benign_val": norm,
        "alpha": alpha, "sustain_windows": sustain,
        "kill_threshold_z_per_dataset": kill_thr,
        "kill_calibration": f"{KILL_PERCENTILE}th percentile of benign val sustained peak confidence",
        "alert_high_z": alert_high, "alert_low_z": alert_low, "hysteresis_gap": HYSTERESIS_GAP,
        "alert_calibration": "max Youden's J on TUNE (peak confidence)",
        "escalate_after_windows_per_dataset": escalate,
        "escalation_calibration": f"{best_esc['escalate_percentile']} percentile of the longest continuous alert "
                                  "over benign val processes (min = sustain); percentile chosen on TUNE",
        "escalation_selection_rule": f"most TUNE attacks killed with TUNE benign kill rate <= {MAX_TUNE_FALSE_KILL}",
        "selection_rule": f"most TUNE attacks killed with TUNE benign false-kill rate <= {MAX_TUNE_FALSE_KILL}; "
                          "ties -> fewer syscalls before kill",
    })
    write_csv("kill_log.csv", log)
    write_csv("per_dataset.csv", per_dataset)
    write_csv("per_attack_type.csv", per_attack)
    write_json("response_summary.json", {
        "evaluated_on": "FINAL 70% of test (split/test_partition.json)",
        "records": len(log), **overall, **flap,
        "simulation_matches_killswitch_class": "300 / 300 FINAL records",
        "layer4_whole_recording_comparison": {
            "strict_p99_attacks_caught": int(l4["attacks_caught_p99"]),
            "strict_p99_false_alarms": int(l4["false_alarms_p99"]),
            "balanced_attacks_caught": int(l4["attacks_caught_balanced"]),
            "balanced_false_alarms": int(l4["false_alarms_balanced"]),
            "note": "Layer 4 decides after the whole recording; Layer 5 decides online, window by window.",
        },
        "live_demo": demo_result,
    })

    # ---- plots
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK})
    examples = []
    for want in (("attack", True), ("benign", False)):
        cands = [(r, row) for r, row in zip(final, log)
                 if row["label"] == want[0] and row["killed"] == want[1] and 300 <= len(r["c"]) <= 6000]
        if cands:
            examples.append(cands[len(cands) // 2])
    fig, axes = plt.subplots(1, len(examples), figsize=(6.2 * len(examples), 3.6), squeeze=False)
    for ax, (r, row) in zip(axes[0], examples):
        style_axes(ax)
        x = np.arange(len(r["c"])) + N
        ax.plot(x, r["c"], color=COLORS[r["label"]], linewidth=1.6, label="confidence")
        ax.axhline(kill_thr[r["dataset"]], color=KILL_COLOR, linewidth=1.2, linestyle="--")
        ax.axhline(alert_high, color=ALERT_COLOR, linewidth=1.2, linestyle="--")
        ax.axhline(alert_low, color=ALERT_COLOR, linewidth=1, linestyle=":")
        ax.text(x[-1], kill_thr[r["dataset"]], " kill", color=KILL_COLOR, fontsize=8, va="bottom", ha="right")
        ax.text(x[-1], alert_high, " alert on", color=ALERT_COLOR, fontsize=8, va="bottom", ha="right")
        ax.text(x[-1], alert_low, " alert off", color=ALERT_COLOR, fontsize=8, va="top", ha="right")
        if row["killed"]:
            ax.axvline(row["kill_syscall"], color=KILL_COLOR, linewidth=1.5)
            ax.text(row["kill_syscall"], ax.get_ylim()[1], f" killed at syscall {row['kill_syscall']}",
                    color=KILL_COLOR, fontsize=8, va="top")
        what = f"{r['dataset']} {r['attack_type']} attack" if r["label"] == "attack" else f"{r['dataset']} normal process"
        ax.set_title(f"{what} ({r['recording_id'][:28]})", loc="left", fontsize=10)
        ax.set_xlabel("system calls executed")
        ax.set_ylabel("confidence (z)")
    fig.tight_layout()
    save(fig, "example_timelines.png")

    fig, ax = plt.subplots(figsize=(6.4, 4))
    style_axes(ax)
    kills = np.sort([row["kill_syscall"] for row in log if row["killed"] and row["label"] == "attack"])
    if len(kills):
        ax.step(kills, np.arange(1, len(kills) + 1) / overall["attacks"], where="post", color=COLORS["attack"],
                linewidth=2)
        ax.set_xscale("log")
        ticks = [t for t in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000) if t <= kills.max() * 1.5]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
    ax.set_xlabel("system calls executed before the kill (log scale)")
    ax.set_ylabel("share of all attacks killed")
    ax.set_title("How fast attacks are stopped — FINAL", loc="left")
    save(fig, "time_to_kill.png")

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(per_attack) + 1.4))
    style_axes(ax)
    ax.grid(False, axis="y")
    y = np.arange(len(per_attack))
    ax.barh(y - 0.2, [r["alert_rate"] for r in per_attack], height=0.38, color=ALERT_COLOR, label="alerted")
    ax.barh(y + 0.2, [r["kill_rate"] for r in per_attack], height=0.38, color=KILL_COLOR, label="killed")
    ax.set_yticks(y, [f"{r['dataset']} · {r['attack_type']}  (n={r['n_attack']})" for r in per_attack])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of attack processes")
    ax.set_title("Alerted and killed per attack type — FINAL", loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    save(fig, "per_attack_response.png")

    print(f"\nFINAL: attacks killed {overall['attacks_killed']}/{overall['attacks']} "
          f"(median {overall['median_syscalls_before_kill']:.0f} syscalls before kill), "
          f"benign killed {overall['benign_killed']}/{overall['benign']}; "
          f"alerted attacks {overall['attacks_alerted']}/{overall['attacks']}, "
          f"benign {overall['benign_alerted']}/{overall['benign']}; "
          f"alert raises benign {flap['benign_alert_raises_with_hysteresis']} with vs "
          f"{flap['benign_alert_raises_without_hysteresis']} without hysteresis")
    if demo_result:
        print(f"Live demo: {demo_result}")


if __name__ == "__main__":
    main()
