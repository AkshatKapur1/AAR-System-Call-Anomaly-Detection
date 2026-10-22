"""
AAR demo: one script that shows the final results of the project.

  Part 1  Layers 1-2: the three datasets parsed into one schema, then
          deduplicated / balanced, split 70/15/15, tokenised and windowed.
  Part 2  Layer 3: HMM, GRU and LSTM trained on benign data, and every
          recording scored.
  Part 3  Layer 4: HMM vs GRU vs LSTM on FINAL; the LSTM (best model) in
          detail: thresholds, confusion matrices, per dataset / attack type.
  Part 4  Layer 5: the LSTM-driven graduated response (alert -> isolate ->
          kill) on FINAL.
  Part 5  Live: the trained LSTM scores a few real FINAL recordings syscall by
          syscall, they are replayed into the KillSwitch, and a real dummy
          process is killed when the switch fires. Every live decision is
          checked against the saved Layer 4 / Layer 5 results.

Part 1 counts the data files once (~15 s) and caches the counts in
logs/demo_pipeline_stats.json. Parts 2-4 read the saved outputs in models/,
scoring/, evaluation/ and response/ (stdlib only). Part 5 needs torch + numpy
(skipped with a message if they are missing).

Run from the repo root (~/AAR):
    # (venv/bin/activate points at an old path; call the venv python directly)
    venv/bin/python demo.py              # everything (~5 s, ~20 s first time)
    venv/bin/python demo.py --no-live    # parts 1-4 only
    venv/bin/python demo.py --recount    # recount part 1 instead of the cache
"""
import argparse
import csv
import gzip
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
N = 7


# ---------------------------------------------------------------- helpers

def path(*p):
    return os.path.join(REPO_ROOT, *p)


def read_csv(*p):
    with open(path(*p), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(*p):
    with open(path(*p), encoding="utf-8") as f:
        return json.load(f)


def pct(x):
    return f"{100 * float(x):.1f}%"


def heading(text, char="="):
    print(f"\n{char * 78}\n{text}\n{char * 78}")


def table(header, rows):
    rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(str(h)), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    line = lambda cells: "  " + "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                                          for i, (c, w) in enumerate(zip(cells, widths)))
    print(line([str(h) for h in header]))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print(line(r))


def confusion(tp, fp, tn, fn):
    print(f"                     predicted ATTACK   predicted NORMAL")
    print(f"    actual ATTACK    {tp:>16}   {fn:>16}")
    print(f"    actual NORMAL    {fp:>16}   {tn:>16}")


# ---------------------------------------------------------------- part 1

DATASETS = ["ADFA-LD", "LID-DS", "PLAID"]
STAGE_FILES = {
    "parsed": ["parsed/adfa_parsed.jsonl.gz", "parsed/liddds_parsed.jsonl.gz", "parsed/plaid_parsed.jsonl.gz"],
    "capped": ["capped/adfa_capped.jsonl.gz", "capped/liddds_capped.jsonl.gz", "capped/plaid_capped.jsonl.gz"],
    "train": ["split/train.jsonl.gz"], "val": ["split/val.jsonl.gz"], "test": ["split/test.jsonl.gz"],
}
STATS_CACHE = os.path.join("logs", "demo_pipeline_stats.json")


def count_pipeline():
    """Records and syscalls per stage / dataset / label, read from the data
    files themselves."""
    stats = {"files": {}, "stages": {}, "attack_types": {}, "example": None}
    for stage, files in STAGE_FILES.items():
        st = stats["stages"][stage] = {d: {"benign": [0, 0], "attack": [0, 0]} for d in DATASETS}
        for f in files:
            stats["files"][f] = os.path.getmtime(path(f))
            with gzip.open(path(f), "rt", encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    d = r["source"].split("/")[0]
                    cell = st[d][r["label"]]
                    cell[0] += 1
                    cell[1] += r["seq_len"]
                    if stage == "test" and r["label"] == "attack":
                        stats["attack_types"].setdefault(d, set()).add(r["attack_type"])
                    if stage == "train" and stats["example"] is None:
                        stats["example"] = {"recording_id": r["recording_id"], "source": r["source"],
                                            "syscalls": r["syscalls"][:10], "token_ids": r["token_ids"][:10]}
    stats["attack_types"] = {d: sorted(v) for d, v in stats["attack_types"].items()}
    return stats


def pipeline_stats(recount):
    cache = path(STATS_CACHE)
    if not recount and os.path.exists(cache):
        stats = read_json(STATS_CACHE)
        if all(os.path.exists(path(f)) and os.path.getmtime(path(f)) == t for f, t in stats["files"].items()):
            return stats
    print("  (counting the data files once, ~15 s; cached in logs/demo_pipeline_stats.json)")
    stats = count_pipeline()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    return stats


def layers12(recount):
    heading("PART 1 - LAYERS 1-2: DATA COLLECTION, PARSING AND PREPROCESSING")
    stats = pipeline_stats(recount)
    print("  Layer 1 - three public syscall datasets, each in its own raw format, parsed into one schema")
    print("            (source, recording_id, label, attack_type, syscalls as names - no arguments):")
    print("    ADFA-LD      Linux traces of syscall NUMBERS, one file per trace -> decoded to names (i386 table)")
    print("    LID-DS-2021  zipped scenario recordings (.sc event logs) for CVE-2012-2122 and CVE-2014-0160")
    print("    PLAID        per-application baseline and attack traces")
    s = stats["stages"]
    fmt = lambda c: f"{c[0]:,} rec / {c[1]:,} sc"
    print("\n  Parsed (Layer 1 output)                 vs   after dedup + balancing (Layer 2)")
    table(["Dataset", "Normal parsed", "Attack parsed", "Normal capped", "Attack capped"],
          [[d, fmt(s["parsed"][d]["benign"]), fmt(s["parsed"][d]["attack"]),
            fmt(s["capped"][d]["benign"]), fmt(s["capped"][d]["attack"])] for d in DATASETS])
    print("  (rec = recordings, sc = system calls)")

    print("\n  Layer 2 - preprocessing:")
    print("    1. Exact duplicates removed; normal data balanced per dataset (~5.4M syscalls each),")
    print("       attack data balanced per attack category (~85K syscalls each).")
    print("    2. Normal recordings split 70/15/15 train/val/test per dataset; ALL attacks go to test")
    print("       only - models never see an attack during training (semi-supervised).")
    table(["Dataset", "Train (normal)", "Val (normal)", "Test normal", "Test attack", "Attack types in test"],
          [[d, s["train"][d]["benign"][0], s["val"][d]["benign"][0], s["test"][d]["benign"][0],
            s["test"][d]["attack"][0], len(stats["attack_types"].get(d, []))] for d in DATASETS]
          + [["Total", *(sum(s[k][d][lab][0] for d in DATASETS) for k, lab in
                         (("train", "benign"), ("val", "benign"), ("test", "benign"), ("test", "attack"))),
              sum(len(v) for v in stats["attack_types"].values())]])
    vocab = read_json("split", "vocab.json")
    windows = read_json("scoring", "scoring_report.json")["splits"]["train"]["windows"]
    print(f"    3. Vocabulary built from training data only: {len(vocab)} tokens "
          f"({len(vocab) - 2} syscalls + <PAD> + <UNK> for syscalls never seen in training).")
    print(f"    4. Sliding window n=7 over each recording (never across recordings): "
          f"{windows:,} training windows.")
    ex = stats["example"]
    print(f"\n  Example - {ex['recording_id']} ({ex['source']}), first 10 syscalls:")
    print("    names : " + " ".join(ex["syscalls"]))
    print("    tokens: " + " ".join(map(str, ex["token_ids"])))
    print("    first 7-window: (" + ", ".join(ex["syscalls"][:N]) + ")")


# ---------------------------------------------------------------- part 2

def layer3():
    heading("PART 2 - LAYER 3: MODELS AND SCORING")
    print("  All three models learn what NORMAL syscall sequences look like (trained on benign data only);")
    print("  a recording is anomalous when the model finds its syscalls surprising.")
    hmm = read_json("models", "HMM", "training_report.json")
    rnns = {m: read_json("models", m, "results", "training_report.json") for m in ("gru", "lstm")}
    sr = read_json("scoring", "scoring_report.json")
    rows = [["HMM", f"{hmm['selected_n_states']} hidden states (best of "
             + "/".join(str(c["n_states"]) for c in hmm["candidates"]) + " on val)",
             f"{sr['models']['hmm']['n_parameters']:,}", "EM, 25 iterations",
             f"{hmm['train_time_seconds']:.0f} s", "-"]]
    for m, r in rnns.items():
        a = r["architecture"]
        rows.append([m.upper(), f"embed {a['embed_dim']}, {a['num_layers']}x{a['hidden_dim']} {m.upper()}",
                     f"{a['n_params']:,}", f"{r['epochs_run']} epochs, early stopping",
                     f"{r['train_time_seconds']:.0f} s", f"{r['best_val_loss']:.3f}"])
    table(["Model", "Architecture", "Parameters", "Training", "Train time", "Best val loss"], rows)
    print("  GRU/LSTM: predict the next syscall (128-syscall chunks); val loss = mean NLL per syscall.")

    print("\n  Scoring - every recording of train / val / test, window by window (n=7):")
    print("    window7 : the model sees only the 7-syscall window")
    print("    stream  : the model reads the recording from the start (full context) - used by the LSTM")
    print("    score per window = mean surprise (NLL) of syscalls 2-7; higher = more anomalous;")
    print("    per recording: mean / max / top-1% / top-5% / top-10% of its window scores")
    table(["Split", "Recordings", "Normal", "Attack", "Windows scored"],
          [[k, v["records"], v["records_by_label"]["benign"], v["records_by_label"]["attack"],
            f"{v['windows']:,}"] for k, v in sr["splits"].items()])
    lstm = sr["models"]["lstm"]["inference"]["test"]
    print(f"  LSTM speed (stream): {lstm['stream_ms_per_1000_windows']:.1f} ms per 1,000 windows on CPU.")
    print("  Check: window7 and stream agree on each record's first window (max diff "
          + ", ".join(f"{m.upper()} {v['consistency_check_max_abs_diff']:.0e}" for m, v in sr["models"].items())
          + ").")

    print("\n  LSTM score (stream / mean) on test - normal vs attack, per dataset:")
    test = read_csv("scoring", "lstm_test_record_scores.csv")
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")
    rows = []
    for d in DATASETS:
        sc = {lab: [float(r["stream_mean"]) for r in test if r["source"].split("/")[0] == d
                    and r["label"] == lab and r["stream_mean"] not in ("", "nan")] for lab in ("benign", "attack")}
        rows.append([d, f"{med(sc['benign']):.3f}", f"{med(sc['attack']):.3f}",
                     f"{med(sc['attack']) / med(sc['benign']):.1f}x"])
    table(["Dataset", "Median normal", "Median attack", "Attack / normal"], rows)
    print("  Attacks score higher than normal recordings in every dataset - Layer 4 turns this into decisions.")


# ---------------------------------------------------------------- part 3

def layer4():
    heading("PART 3 - LAYER 4: EVALUATION (HMM vs GRU vs LSTM)")
    rep = read_json("evaluation", "evaluation_report.json")
    comp = rep["final_composition"]
    print(f"  Trained on benign data only. Test split partitioned: TUNE {rep['partition']['tune_records']} "
          f"records (all choices),\n  FINAL {rep['partition']['final_records']} records "
          f"({comp['benign']} normal, {comp['attack']} attack) - reported below, never used for a choice.")
    print("  Thresholds from benign val per dataset: strict = p99 (~1% false alarms),"
          "\n  balanced = max Youden's J chosen on TUNE.")

    summary = {r["model"]: r for r in read_csv("evaluation", "model_summary.csv")}
    print("\n  Model comparison on FINAL")
    table(["Model", "Config", "ROC-AUC (95% CI)", "PR-AUC", "Caught @p99", "False al. @p99",
           "Caught balanced", "F1 balanced"],
          [[m.upper(), f"{s['mode']}/{s['aggregation']}",
            f"{float(s['roc_auc']):.3f} ({float(s['roc_auc_ci95_low']):.3f}-{float(s['roc_auc_ci95_high']):.3f})",
            f"{float(s['pr_auc']):.3f}",
            f"{s['attacks_caught_p99']}/{s['attacks_total']}", f"{s['false_alarms_p99']}/{s['benign_total']}",
            f"{s['attacks_caught_balanced']}/{s['attacks_total']}", f"{float(s['f1_balanced']):.3f}"]
           for m, s in summary.items()])

    best = read_json("evaluation", "best_model.json")
    tune_auc = best["tune_roc_auc_per_model"]
    print(f"\n  BEST MODEL: {best['best_model'].upper()}  (rule: {best['selection_rule']})")
    print("  TUNE ROC-AUC: " + ", ".join(f"{m.upper()} {v:.3f}" for m, v in tune_auc.items()))

    heading("LSTM results in detail (FINAL)", "-")
    lstm = summary["lstm"]
    print(f"  ROC-AUC {float(lstm['roc_auc']):.3f}   PR-AUC {float(lstm['pr_auc']):.3f}   "
          f"params {int(lstm['n_parameters']):,}   train {float(lstm['train_time_seconds']):.0f} s   "
          f"inference {float(lstm['inference_ms_per_1000_windows']):.1f} ms / 1k windows")

    thr = [r for r in read_csv("evaluation", "threshold_metrics.csv") if r["model"] == "lstm"]
    names = {"p95": "p95 (~5% FA)", "p99": "p99 strict / kill", "p99_5": "p99.5", "balanced_alert": "balanced alert"}
    print("\n  Operating points")
    table(["Threshold", "Attacks caught", "False alarms", "Precision", "Recall", "F1", "FPR", "Accuracy"],
          [[names.get(r["threshold_level"], r["threshold_level"]), f"{r['tp']}/{int(r['tp']) + int(r['fn'])}",
            f"{r['fp']}/{int(r['fp']) + int(r['tn'])}", pct(r["precision"]), pct(r["recall"]),
            f"{float(r['f1']):.3f}", pct(r["fpr"]), pct(r["accuracy"])] for r in thr])
    for level in ("p99", "balanced_alert"):
        r = next(x for x in thr if x["threshold_level"] == level)
        print(f"\n  Confusion matrix - {names[level]}")
        confusion(*(int(r[k]) for k in ("tp", "fp", "tn", "fn")))

    per_ds = [r for r in read_csv("evaluation", "per_dataset_metrics.csv") if r["model"] == "lstm"]
    print("\n  Per dataset / LID-DS CVE")
    table(["Group", "Normal", "Attack", "ROC-AUC", "Caught @p99", "Caught balanced", "Normal passed bal."],
          [[r["group"] + (" *" if r["small_group"] == "True" else ""), r["n_benign"], r["n_attack"],
            f"{float(r['roc_auc']):.3f}",
            f"{r['attacks_caught_p99']}/{r['n_attack']}", f"{r['attacks_caught_balanced']}/{r['n_attack']}",
            f"{r['normal_passed_balanced']}/{r['n_benign']}"] for r in per_ds])

    per_at = [r for r in read_csv("evaluation", "per_attack_type_metrics.csv") if r["model"] == "lstm"]
    print("\n  Detection rate per attack type")
    table(["Dataset", "Attack type", "n", "AUC", "@p95", "@p99", "@balanced"],
          [[r["dataset"], r["attack_type"] + (" *" if r["small_group"] == "True" else ""), r["n_attack"],
            f"{float(r['roc_auc_vs_same_dataset_benign']):.3f}", pct(r["recall_p95"]), pct(r["recall_p99"]),
            pct(r["recall_balanced"])] for r in per_at])
    print("  * small group (n < 20), interpret with care")

    unk = [r for r in read_csv("evaluation", "unk_analysis.csv") if r["model"] == "lstm"]
    if unk:
        print("\n  Unseen-syscall check (is detection just <UNK> tokens?)")
        table(["Attack subset", "n", "ROC-AUC", "Recall @p99"],
              [[r["attack_subset"], r["n_attack"], f"{float(r['roc_auc']):.3f}", pct(r["recall_p99"])] for r in unk])
    return best


# ---------------------------------------------------------------- part 2

def layer5():
    heading("PART 4 - LAYER 5: GRADUATED RESPONSE (alert -> isolate -> kill), LSTM-driven")
    cal = read_json("response", "calibration.json")
    s = read_json("response", "response_summary.json")
    print(f"  Every FINAL recording replayed window by window as a live process. Confidence = running mean"
          f"\n  of z; isolate / kill need {cal['sustain_windows']} consecutive windows above their line, "
          f"so one odd window never acts.")
    table(["Dataset", "Alert on / off (z)", "Isolate / release (z)", "Kill (z)"],
          [[d, f"{cal['alert_high_z']:.2f} / {cal['alert_low_z']:.2f}",
            f"{cal['isolate_threshold_z_per_dataset'][d]:.2f} / {cal['isolate_release_z_per_dataset'][d]:.2f}",
            f"{cal['kill_threshold_z_per_dataset'][d]:.2f}"] for d in cal["kill_threshold_z_per_dataset"]])

    a, n = s["attacks"], s["normal"]
    print(f"\n  Results on FINAL ({n} normal, {a} attack processes)")
    table(["", "Attacks", "Normal processes"],
          [["Alerted", f"{s['attacks_alerted']} ({pct(s['attack_alert_rate'])})",
            f"{s['normal_alerted']} ({pct(s['normal_alert_rate'])})"],
           ["Isolated at some point", s["attacks_isolated"], s["normal_isolated"]],
           ["Contained (isolated or killed)", f"{s['attacks_contained']} ({pct(s['attack_containment_rate'])})",
            f"{s['normal_disrupted']} ({pct(s['normal_disruption_rate'])})"],
           ["Still contained at the end", s["attacks_contained_at_end"], s["normal_contained_at_end"]],
           ["Killed", f"{s['attacks_killed']} ({pct(s['attack_kill_rate'])})",
            f"{s['normal_killed']} ({pct(s['normal_kill_rate'])})"]])
    print(f"  Median syscalls before containment: {s['median_syscalls_before_containment']:.0f}, "
          f"before kill: {s['median_syscalls_before_kill']:.0f}")
    print(f"  Hysteresis: alert raises on normal processes {s['normal_alert_raises_with_hysteresis']} with vs "
          f"{s['normal_alert_raises_without_hysteresis']} without "
          f"({100 * (1 - s['normal_alert_raises_with_hysteresis'] / s['normal_alert_raises_without_hysteresis']):.0f}% fewer)")
    l4 = s["layer4_whole_recording_comparison"]
    print(f"  Layer 4 (decides only after the whole recording): strict {l4['strict_p99_attacks_caught']} caught / "
          f"{l4['strict_p99_false_alarms']} FA, balanced {l4['balanced_attacks_caught']} / {l4['balanced_false_alarms']}")
    print(f"  Check: {s['simulation_matches_killswitch_class']} - simulation == KillSwitch class")

    print("\n  Per dataset")
    table(["Dataset", "Attacks", "Alerted", "Contained", "Killed", "Normal", "Normal isolated", "Normal killed"],
          [[r["group"], r["attacks"], r["attacks_alerted"], r["attacks_contained"], r["attacks_killed"],
            r["normal"], r["normal_isolated"], r["normal_killed"]]
           for r in read_csv("response", "per_dataset.csv")])

    print("\n  Per attack type")
    table(["Dataset", "Attack type", "n", "Alerted", "Contained", "Killed", "Median syscalls to contain"],
          [[r["dataset"], r["attack_type"], r["n_attack"], pct(r["alert_rate"]), pct(r["containment_rate"]),
            pct(r["kill_rate"]),
            "-" if r["median_syscalls_before_containment"] == "nan" else f"{float(r['median_syscalls_before_containment']):.0f}"]
           for r in read_csv("response", "per_attack_type.csv")])
    return cal


# ---------------------------------------------------------------- part 3

def pick_examples(log):
    """A few FINAL processes that show each outcome (chosen from the saved
    action log, not from anything computed here)."""
    want = [
        ("ADFA-LD Hydra-FTP attack (should be KILLED)",
         lambda r: r["attack_type"] == "Hydra_FTP" and r["killed"] == "True"),
        ("LID-DS Heartbleed attack (should be KILLED)",
         lambda r: r["attack_type"] == "CVE-2014-0160" and r["killed"] == "True"),
        ("Attack ISOLATED, not killed",
         lambda r: r["label"] == "attack" and r["isolated"] == "True" and r["killed"] == "False"),
        ("Normal process (should be LEFT RUNNING)",
         lambda r: r["label"] == "benign" and r["contained"] == "False" and r["dataset"] == "LID-DS"),
        ("Normal process (should be LEFT RUNNING)",
         lambda r: r["label"] == "benign" and r["contained"] == "False" and r["dataset"] == "PLAID"),
    ]
    out = []
    for title, cond in want:
        cands = sorted((r for r in log if cond(r) and int(r["n_syscalls"]) <= 5000),
                       key=lambda r: int(r["n_syscalls"]))
        if cands:
            out.append((title, cands[len(cands) // 4]))
    return out


def load_tokens(indices):
    need, out = set(indices), {}
    with gzip.open(path("split", "test.jsonl.gz"), "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in need:
                out[i] = json.loads(line)["token_ids"]
                if len(out) == len(need):
                    break
    return out


def live(best, cal):
    heading("PART 5 - LIVE: LSTM scores real FINAL recordings -> KillSwitch -> real process killed")
    try:
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        print(f"  Skipped: {e}. Install with:  pip install torch --index-url "
              "https://download.pytorch.org/whl/cpu numpy")
        return
    sys.path.insert(0, path("models"))
    sys.path.insert(0, path("codes", "response"))
    import sequence_model_common as rnn
    from kill_switch import KillSwitch, Thresholds, terminate_process
    import subprocess

    model = rnn.load_model(path(best["for_layer5"]["model_file"]))
    print(f"  Loaded {best['for_layer5']['model_file']} ({sum(p.numel() for p in model.parameters()):,} params)")
    log = read_csv("response", "action_log.csv")
    scores = {r["record_index"]: r for r in read_csv("scoring", "lstm_test_record_scores.csv")}
    l4 = best["for_layer5"]["thresholds"]
    examples = pick_examples(log)
    tokens = load_tokens([int(r["record_index"]) for _, r in examples])
    agree = 0

    for title, row in examples:
        d, ids = row["dataset"], tokens[int(row["record_index"])]
        print(f"\n  >> {title}\n     {row['recording_id']}  ({d}, {row['label']}"
              f"{', ' + row['attack_type'] if row['attack_type'] else ''}, {len(ids)} syscalls)")

        # Layer 4 view: whole-recording score
        t0 = time.time()
        win = rnn.stream_window_scores(rnn.token_nll_stream(model, [ids])[0])
        ms = 1000 * (time.time() - t0)
        rec = float(win.mean())
        g = l4["groups"][d]
        z = (rec - g["median"]) / (g["p99"] - g["median"])
        saved = float(scores[row["record_index"]]["stream_mean"])
        print(f"     Layer 4: LSTM scored {len(win)} windows in {ms:.0f} ms; record score {rec:.4f} "
              f"(saved {saved:.4f}), z = {z:.2f}")
        print(f"              strict (z>1): {'ATTACK' if z > 1 else 'normal'}   "
              f"balanced (z>{l4['balanced_alert']['z_cutoff']:.2f}): "
              f"{'ATTACK' if z > l4['balanced_alert']['z_cutoff'] else 'normal'}")

        # Layer 5 view: online, bound to a real process
        norm = cal["window_normalisation_from_benign_val"][d]
        thr = Thresholds(norm["median"], norm["p99"], cal["alert_high_z"], cal["alert_low_z"],
                         cal["isolate_threshold_z_per_dataset"][d], cal["isolate_release_z_per_dataset"][d],
                         cal["kill_threshold_z_per_dataset"][d])
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        switch = KillSwitch(cal["alpha"], cal["sustain_windows"], thr,
                            on_kill=lambda s, pid=proc.pid: terminate_process(pid))
        for s in win:
            if switch.update(float(s)) == "kill":
                break
        time.sleep(0.3)
        alive = proc.poll() is None
        if alive:
            proc.kill()
        events = switch.events
        shown = events[:6] + ([("...", None, None)] if len(events) > 7 else []) + (events[-1:] if len(events) > 6 else [])
        print(f"     Layer 5: dummy process pid {proc.pid} bound to the KillSwitch; events:")
        for name, w, c in shown:
            print("              ..." if w is None else f"              {name:<8} at syscall {w + N:>5}  (confidence {c:.2f})")
        if not events:
            print("              (none - confidence never reached the alert line)")
        status = "KILLED" if switch.killed else ("ISOLATED" if switch.isolated else "left running")
        print(f"     Outcome: {status}; process {'still alive' if alive else 'terminated by the switch'}")

        first_iso = next((w for e, w, _ in events if e == "isolate"), None)
        ok = (switch.killed == (row["killed"] == "True")
              and (not switch.killed or switch.window + N == int(row["kill_syscall"]))
              and (first_iso is not None) == (row["isolated"] == "True")
              and abs(rec - saved) < 1e-3)
        agree += ok
        print(f"     Matches saved Layer 4/5 results: {'YES' if ok else 'NO'}")
    print(f"\n  {agree}/{len(examples)} live runs reproduce the saved results.")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-live", action="store_true", help="skip part 5 (no torch needed)")
    ap.add_argument("--recount", action="store_true", help="recount part 1 from the data files")
    args = ap.parse_args()
    print("AAR - AI-based System Call Anomaly Detection & Response (ADFA-LD + LID-DS-2021 + PLAID)")
    layers12(args.recount)
    layer3()
    best = layer4()
    cal = layer5()
    if not args.no_live:
        live(best, cal)
    heading("Plots and reports")
    for d in ("evaluation/plots", "response/plots", "reports"):
        for f in sorted(os.listdir(path(d))):
            if f != "v1_vs_v2.png":
                print(f"  {d}/{f}")


if __name__ == "__main__":
    main()
