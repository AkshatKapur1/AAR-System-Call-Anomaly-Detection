"""
Layer 5: simulate the graduated response (codes/response/kill_switch.py:
alert -> isolate -> kill) on every val/test recording, replayed window by
window in order, as if each recording were a live process.

Input : evaluation/best_model.json (Layer 4: LSTM, stream mode),
        split/{val,test}.jsonl.gz, split/test_partition.json.
Output: response/ — calibration, tuning results, action log, metrics, plots.

Protocol (same discipline as Layer 4 — FINAL is never used for a choice):
  1. Window scores: the best model in stream mode (each 7-syscall window
     scored with the whole preceding recording as context).
  2. Normalisation per dataset from benign val WINDOWS: z = (s - median) /
     (p99 - median). Confidence: running mean of z since the process started
     (alpha = 0, the online form of Layer 4's `mean` aggregation) or an EWMA.
  3. KILL: for each (alpha, sustain) the kill line is calibrated per dataset
     on benign val so that ~1% of benign val processes would ever be killed;
     (alpha, sustain) chosen on TUNE: most attacks killed with <= 2% benign
     TUNE processes killed; ties -> fewer syscalls before the kill.
  4. ISOLATE (reversible): isolation line per dataset = the level that ~2% of
     benign val processes ever reach (sustained, same gate as kill); release
     below isolation line - HYSTERESIS_GAP. Fixed at the most conservative
     level tested; other levels reported as TUNE sensitivity.
  5. ALERT: max Youden's J (attacks alerted % + benign not alerted %) on
     TUNE; alert_low = alert_high - HYSTERESIS_GAP.
  6. FINAL: attacks contained (isolated or killed) / killed, normal processes
     isolated / killed, time to containment and kill, time spent isolated,
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
KILL_PERCENTILE = 99.0          # ~1% of benign val processes ever killed
MAX_TUNE_FALSE_KILL = 0.02
ISOLATE_BUDGET = 2.0            # ~2% of benign val processes ever isolated (reversible)
ISOLATE_SENSITIVITY = [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
HYSTERESIS_GAP = 0.25           # release / alert-off = on-line - gap (z units)
NEG = -1e9

COLORS = {"attack": "#eb6834", "benign": "#2a78d6"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ALERT_COLOR, ISOLATE_COLOR, KILL_COLOR = "#c98500", "#ec835a", "#d03b3b"


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


def first_sustained(c, k, thr):
    """First window at which c has been above thr for k consecutive windows."""
    run = 0
    for t, v in enumerate(c > thr):
        run = run + 1 if v else 0
        if run >= k:
            return t
    return None


def isolation_track(c, k, line, release):
    """Same state machine as KillSwitch: returns (first isolation window,
    number of isolations, windows spent isolated, isolated at the end)."""
    run, on, first, count, spent = 0, False, None, 0, 0
    for t, v in enumerate(c):
        run = run + 1 if v > line else 0
        if not on and run >= k:
            on, count = True, count + 1
            first = t if first is None else first
        elif on and v < release:
            on = False
        spent += on
    return first, count, spent, on


def alert_track(c, high, low):
    """(first alert window, number of alert raises) with hysteresis."""
    on, raises, first = False, 0, None
    for t, v in enumerate(c):
        if not on and v > high:
            on, raises = True, raises + 1
            first = t if first is None else first
        elif on and v < low:
            on = False
    return first, raises


def youden_cutoff(y, s):
    order = np.unique(s)
    best, best_j = order[0], -1
    for i in range(len(order) - 1):
        cut = (order[i] + order[i + 1]) / 2
        j = np.mean(s[y == 1] > cut) - np.mean(s[y == 0] > cut)
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

    # ---- 3. kill: calibrate on val, choose (alpha, sustain) on TUNE
    tuning, best_cfg = [], None
    for a in ALPHAS:
        conf = {id(r): confidence(r["z"], a) for r in val + tune}
        for k in SUSTAINS:
            peak = {id(r): sustained_peak(conf[id(r)], k) for r in val + tune}
            kill_thr = {}
            for d in datasets:
                p = np.array([peak[id(r)] for r in val if r["dataset"] == d])
                kill_thr[d] = float(np.percentile(p[p > NEG], KILL_PERCENTILE))
            killed = lambda rs: np.array([peak[id(r)] > kill_thr[r["dataset"]] for r in rs])
            tk = killed(tune)
            ty = np.array([r["label"] == "attack" for r in tune])
            times = [first_sustained(conf[id(r)], k, kill_thr[r["dataset"]]) + N
                     for r, kk in zip(tune, tk) if kk and r["label"] == "attack"]
            row = {"alpha": a, "sustain": k,
                   "val_benign_kill_rate": float(killed(val).mean()),
                   "tune_attacks_killed": int(tk[ty].sum()), "tune_attacks": int(ty.sum()),
                   "tune_benign_killed": int(tk[~ty].sum()), "tune_benign": int((~ty).sum()),
                   "tune_benign_kill_rate": float(tk[~ty].mean()),
                   "tune_median_syscalls_before_kill": float(np.median(times)) if times else float("nan"),
                   "selected": False, "_kill_thr": kill_thr}
            tuning.append(row)
            if row["tune_benign_kill_rate"] <= MAX_TUNE_FALSE_KILL and (
                    best_cfg is None or row["tune_attacks_killed"] > best_cfg["tune_attacks_killed"]
                    or (row["tune_attacks_killed"] == best_cfg["tune_attacks_killed"]
                        and row["tune_median_syscalls_before_kill"] < best_cfg["tune_median_syscalls_before_kill"])):
                best_cfg = row
    best_cfg["selected"] = True
    alpha, sustain, kill_thr = best_cfg["alpha"], best_cfg["sustain"], best_cfg["_kill_thr"]
    print(f"Kill: alpha={alpha}, sustain={sustain} -> TUNE kills {best_cfg['tune_attacks_killed']}/"
          f"{best_cfg['tune_attacks']}, normal killed {best_cfg['tune_benign_killed']}/{best_cfg['tune_benign']}")

    for r in val + test:
        r["c"] = confidence(r["z"], alpha)
        r["peak"] = sustained_peak(r["c"], sustain)
        r["kill_w"] = first_sustained(r["c"], sustain, kill_thr[r["dataset"]])

    # ---- 4. isolation lines from benign val (+ TUNE sensitivity)
    def iso_lines(budget):
        return {d: float(np.percentile([r["peak"] for r in val if r["dataset"] == d and r["peak"] > NEG],
                                        100 - budget)) for d in datasets}

    sensitivity = []
    for budget in ISOLATE_SENSITIVITY:
        lines = iso_lines(budget)
        stopped = lambda r: r["kill_w"] is not None or r["peak"] > lines[r["dataset"]]
        ta = [r for r in tune if r["label"] == "attack"]
        tb = [r for r in tune if r["label"] == "benign"]
        sensitivity.append({
            "isolate_budget_pct_of_val_normal": budget, "used": budget == ISOLATE_BUDGET,
            "isolation_lines": "; ".join(f"{d} {v:.2f}" for d, v in lines.items()),
            "tune_attacks_stopped": sum(map(stopped, ta)), "tune_attacks": len(ta),
            "tune_normal_disrupted": sum(map(stopped, tb)), "tune_normal_killed": sum(r["kill_w"] is not None for r in tb),
            "tune_normal": len(tb), "tune_normal_disrupted_rate": sum(map(stopped, tb)) / len(tb)})
    isolate = iso_lines(ISOLATE_BUDGET)
    release = {d: v - HYSTERESIS_GAP for d, v in isolate.items()}
    assert all(isolate[d] < kill_thr[d] for d in datasets), "isolation line must sit below the kill line"

    # ---- 5. alert on TUNE
    ty = np.array([r["label"] == "attack" for r in tune], dtype=int)
    alert_high = youden_cutoff(ty, np.array([r["c"].max() for r in tune]))
    alert_low = alert_high - HYSTERESIS_GAP

    # ---- 6. FINAL
    log = []
    for r in final:
        d, kw = r["dataset"], r["kill_w"]
        live = r["c"] if kw is None else r["c"][:kw]          # the process stops at the kill
        fi, n_iso, spent, iso_end = isolation_track(live, sustain, isolate[d], release[d])
        fa, raises = alert_track(live, alert_high, alert_low)
        _, raises_no_hyst = alert_track(live, alert_high, alert_high)
        contain = min(t for t in (fi, kw) if t is not None) if (fi is not None or kw is not None) else None
        log.append({
            "record_index": r["record_index"], "recording_id": r["recording_id"], "dataset": d,
            "source": r["source"], "label": r["label"], "attack_type": r["attack_type"],
            "n_syscalls": len(r["token_ids"]), "n_windows": len(r["c"]),
            "alerted": fa is not None, "first_alert_syscall": (fa + N) if fa is not None else "",
            "alert_raises_with_hysteresis": raises, "alert_raises_without_hysteresis": raises_no_hyst,
            "isolated": fi is not None, "first_isolation_syscall": (fi + N) if fi is not None else "",
            "isolations": n_iso, "share_of_life_isolated": spent / len(live) if len(live) else 0.0,
            "killed": kw is not None, "kill_syscall": (kw + N) if kw is not None else "",
            "contained": contain is not None, "containment_syscall": (contain + N) if contain is not None else "",
            "contained_at_end": kw is not None or iso_end,
            "peak_confidence": float(r["c"].max()),
        })

    def summarise(rows):
        att = [r for r in rows if r["label"] == "attack"]
        ben = [r for r in rows if r["label"] == "benign"]
        cnt = lambda rs, key: sum(r[key] for r in rs)
        rate = lambda rs, key: cnt(rs, key) / len(rs) if rs else float("nan")
        med = lambda rs, key, flag: float(np.median([r[key] for r in rs if r[flag]])) if cnt(rs, flag) else float("nan")
        iso_ben = [r for r in ben if r["isolated"]]
        return {
            "attacks": len(att), "normal": len(ben),
            "attacks_alerted": cnt(att, "alerted"), "attacks_contained": cnt(att, "contained"),
            "attacks_isolated": cnt(att, "isolated"), "attacks_killed": cnt(att, "killed"),
            "attacks_contained_at_end": cnt(att, "contained_at_end"),
            "normal_contained_at_end": cnt(ben, "contained_at_end"),
            "normal_alerted": cnt(ben, "alerted"), "normal_isolated": cnt(ben, "isolated"),
            "normal_killed": cnt(ben, "killed"), "normal_disrupted": cnt(ben, "contained"),
            "attack_alert_rate": rate(att, "alerted"), "attack_containment_rate": rate(att, "contained"),
            "attack_kill_rate": rate(att, "killed"),
            "normal_alert_rate": rate(ben, "alerted"), "normal_disruption_rate": rate(ben, "contained"),
            "normal_kill_rate": rate(ben, "killed"),
            "median_syscalls_before_containment": med(att, "containment_syscall", "contained"),
            "median_syscalls_before_kill": med(att, "kill_syscall", "killed"),
            "isolated_normal_mean_share_of_life_isolated":
                float(np.mean([r["share_of_life_isolated"] for r in iso_ben])) if iso_ben else float("nan"),
        }

    overall = summarise(log)
    flap = {"normal_alert_raises_with_hysteresis": sum(r["alert_raises_with_hysteresis"] for r in log if r["label"] == "benign"),
            "normal_alert_raises_without_hysteresis": sum(r["alert_raises_without_hysteresis"] for r in log if r["label"] == "benign"),
            "all_alert_raises_with_hysteresis": sum(r["alert_raises_with_hysteresis"] for r in log),
            "all_alert_raises_without_hysteresis": sum(r["alert_raises_without_hysteresis"] for r in log)}
    per_dataset = [{"group": d, **summarise([r for r in log if r["dataset"] == d])} for d in datasets]
    per_dataset += [{"group": s, **summarise([r for r in log if r["source"] == s])}
                    for s in sorted({r["source"] for r in log}) if "/" in s]
    per_attack = []
    for d, at in sorted({(r["dataset"], r["attack_type"]) for r in log if r["label"] == "attack"}):
        rows = [r for r in log if r["label"] == "attack" and r["attack_type"] == at and r["dataset"] == d]
        s = summarise(rows)
        per_attack.append({"dataset": d, "attack_type": at, "n_attack": len(rows),
                           "alerted": s["attacks_alerted"], "contained": s["attacks_contained"],
                           "killed": s["attacks_killed"], "alert_rate": s["attack_alert_rate"],
                           "containment_rate": s["attack_containment_rate"], "kill_rate": s["attack_kill_rate"],
                           "median_syscalls_before_containment": s["median_syscalls_before_containment"]})

    # ---- consistency: vectorised simulation == KillSwitch class, process by process
    def thresholds_for(d):
        st = norm[d]
        return Thresholds(st["median"], st["p99"], alert_high, alert_low, isolate[d], release[d], kill_thr[d])

    mismatches = 0
    for r, row in list(zip(final, log))[:300]:
        ks, first_iso = KillSwitch(alpha, sustain, thresholds_for(r["dataset"])), None
        for s in r["s"]:
            if ks.update(float(s)) == "isolate" and first_iso is None:
                first_iso = ks.window
        ok = (ks.killed == row["killed"] and (not ks.killed or ks.window + N == row["kill_syscall"])
              and (first_iso is not None) == row["isolated"]
              and (first_iso is None or first_iso + N == row["first_isolation_syscall"]))
        mismatches += not ok
    assert mismatches == 0, f"{mismatches} KillSwitch/simulation mismatches"

    # ---- live demo of the kill path on one killed attack
    demo_rec = next((r for r, row in zip(final, log) if row["killed"] and row["label"] == "attack"), None)
    demo_result = None
    if demo_rec is not None:
        res = demo(demo_rec["s"], alpha, sustain, thresholds_for(demo_rec["dataset"]))
        demo_result = {"recording_id": demo_rec["recording_id"], "attack_type": demo_rec["attack_type"],
                       **{k: v for k, v in res.items() if k != "events"},
                       "events": [[e[0], e[1], float(e[2])] for e in res["events"]]}

    l4 = {r["model"]: r for r in csv.DictReader(open(os.path.join(REPO_ROOT, "evaluation", "model_summary.csv"),
                                                     encoding="utf-8"))}[best["best_model"]]

    print("Writing response outputs...")
    for row in tuning:
        row.pop("_kill_thr")
    write_csv("tuning_results.csv", tuning)
    write_csv("isolation_sensitivity.csv", sensitivity)
    write_json("calibration.json", {
        "model": best["best_model"], "mode": "stream",
        "window_normalisation_from_benign_val": norm,
        "alpha": alpha, "sustain_windows": sustain,
        "kill_threshold_z_per_dataset": kill_thr,
        "kill_calibration": f"{KILL_PERCENTILE}th percentile of benign val sustained peak confidence",
        "kill_selection_rule": f"most TUNE attacks killed with TUNE benign kill rate <= {MAX_TUNE_FALSE_KILL}; "
                               "ties -> fewer syscalls before kill",
        "isolate_threshold_z_per_dataset": isolate, "isolate_release_z_per_dataset": release,
        "isolate_calibration": f"{100 - ISOLATE_BUDGET}th percentile of benign val sustained peak confidence "
                               f"(~{ISOLATE_BUDGET}% of normal val processes ever isolated)",
        "alert_high_z": alert_high, "alert_low_z": alert_low, "hysteresis_gap": HYSTERESIS_GAP,
        "alert_calibration": "max Youden's J on TUNE (peak confidence)",
    })
    write_csv("action_log.csv", log)
    write_csv("per_dataset.csv", per_dataset)
    write_csv("per_attack_type.csv", per_attack)
    write_json("response_summary.json", {
        "evaluated_on": "FINAL 70% of test (split/test_partition.json)",
        "processes": len(log), **overall, **flap,
        "simulation_matches_killswitch_class": "300 / 300 FINAL processes (kill and isolation decisions)",
        "layer4_whole_recording_comparison": {
            "strict_p99_attacks_caught": int(l4["attacks_caught_p99"]),
            "strict_p99_false_alarms": int(l4["false_alarms_p99"]),
            "balanced_attacks_caught": int(l4["attacks_caught_balanced"]),
            "balanced_false_alarms": int(l4["false_alarms_balanced"]),
            "note": "Layer 4 decides after the whole recording; Layer 5 acts online, window by window.",
        },
        "live_demo": demo_result,
    })

    # ---- plots
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK})
    examples = []
    for want in (("attack", "killed"), ("attack", "isolated"), ("benign", None)):
        cands = [(r, row) for r, row in zip(final, log) if row["label"] == want[0] and 200 <= len(r["c"]) <= 6000
                 and (want[1] is None and not row["contained"]
                      or want[1] == "killed" and row["killed"]
                      or want[1] == "isolated" and row["isolated"] and not row["killed"])]
        if cands:
            examples.append(cands[len(cands) // 2])
    fig, axes = plt.subplots(1, len(examples), figsize=(5.4 * len(examples), 3.6), squeeze=False)
    for ax, (r, row) in zip(axes[0], examples):
        style_axes(ax)
        d = r["dataset"]
        end = len(r["c"]) if not row["killed"] else row["kill_syscall"] - N + 1
        x = np.arange(end) + N
        ax.plot(x, r["c"][:end], color=COLORS[r["label"]], linewidth=1.6)
        for y, color, label, ls in ((kill_thr[d], KILL_COLOR, "kill", "--"), (isolate[d], ISOLATE_COLOR, "isolate", "--"),
                                    (alert_high, ALERT_COLOR, "alert", "--")):
            ax.axhline(y, color=color, linewidth=1.1, linestyle=ls)
            ax.text(x[-1], y, f" {label}", color=color, fontsize=8, va="bottom", ha="right")
        if row["isolated"]:
            ax.axvline(row["first_isolation_syscall"], color=ISOLATE_COLOR, linewidth=1.3)
        if row["killed"]:
            ax.axvline(row["kill_syscall"], color=KILL_COLOR, linewidth=1.5)
        status = "killed" if row["killed"] else ("isolated" if row["isolated"] else "left running")
        what = f"{d} {r['attack_type']} attack" if r["label"] == "attack" else f"{d} normal process"
        ax.set_title(f"{what}: {status}", loc="left", fontsize=10)
        ax.set_xlabel("system calls executed")
        ax.set_ylabel("confidence (z)")
    fig.tight_layout()
    save(fig, "example_timelines.png")

    fig, ax = plt.subplots(figsize=(6.4, 4))
    style_axes(ax)
    for key, flag, color, label in (("containment_syscall", "contained", ISOLATE_COLOR, "contained (isolated or killed)"),
                                    ("kill_syscall", "killed", KILL_COLOR, "killed")):
        t = np.sort([row[key] for row in log if row[flag] and row["label"] == "attack"])
        if len(t):
            ax.step(t, np.arange(1, len(t) + 1) / overall["attacks"], where="post", color=color, linewidth=2, label=label)
    ax.set_xscale("log")
    ticks = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("system calls executed before the action (log scale)")
    ax.set_ylabel("share of all attacks")
    ax.set_title("How fast attacks are contained / killed — FINAL", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    save(fig, "time_to_action.png")

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(per_attack) + 1.4))
    style_axes(ax)
    ax.grid(False, axis="y")
    y = np.arange(len(per_attack))
    for off, key, color, label in ((-0.27, "alert_rate", ALERT_COLOR, "alerted"),
                                   (0, "containment_rate", ISOLATE_COLOR, "contained (isolated or killed)"),
                                   (0.27, "kill_rate", KILL_COLOR, "killed")):
        ax.barh(y + off, [r[key] for r in per_attack], height=0.25, color=color, label=label)
    ax.set_yticks(y, [f"{r['dataset']} · {r['attack_type']}  (n={r['n_attack']})" for r in per_attack])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of attack processes")
    ax.set_title("Response per attack type — FINAL", loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    save(fig, "per_attack_response.png")

    print(f"\nFINAL: attacks contained {overall['attacks_contained']}/{overall['attacks']} "
          f"(killed {overall['attacks_killed']}), normal isolated {overall['normal_isolated']}, "
          f"normal killed {overall['normal_killed']}, normal disrupted {overall['normal_disrupted']}/{overall['normal']}; "
          f"median syscalls before containment {overall['median_syscalls_before_containment']:.0f}; "
          f"alerted attacks {overall['attacks_alerted']}, normal {overall['normal_alerted']}; "
          f"alert raises normal {flap['normal_alert_raises_with_hysteresis']} vs "
          f"{flap['normal_alert_raises_without_hysteresis']} without hysteresis")
    if demo_result:
        print(f"Live demo: killed_by_switch={demo_result['killed_by_switch']} at window {demo_result['kill_window']}")


if __name__ == "__main__":
    main()
