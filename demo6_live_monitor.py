"""DEMO 6 - LIVE MONITOR: stream a real process trace one system call at a time.

A real recording from the FINAL test set (never used for training or tuning) is fed to
the detector exactly as a running process would feed it: one system call at a time.
For every new system call the LSTM is advanced by one step (keeping its memory), the
newest 7-syscall window is scored, and the project's KillSwitch decides
NORMAL / ALERT / ISOLATED / KILLED. A real stand-in process is terminated on KILL.

Click Run, then choose a recording from the menu (or pass its number, e.g. "1").
"""
import aar_demo_common  # noqa: F401  - must be first: switches to the project's Python environment
import math
import os
import subprocess
import sys
import time
from collections import deque

import torch

from aar_demo_common import dataset_of, load_csv, load_json, load_jsonl_gz, path
import sequence_model_common as rnn  # noqa: E402
from kill_switch import KillSwitch, Thresholds, terminate_process  # noqa: E402

os.system("")  # enable ANSI colours in Windows terminals
C = {"reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m", "green": "\033[32m", "yellow": "\033[33m",
     "orange": "\033[38;5;208m", "red": "\033[31m", "cyan": "\033[36m", "invert": "\033[7m"}
STATE_COLOUR = {"NORMAL": "green", "ALERT": "yellow", "ISOLATED": "orange", "KILLED": "red"}
N = 7
TARGET_SECONDS = 35          # a stream lasts about this long
MAX_LINES = 500              # at most this many syscall lines are printed (events always are)

cal = load_json("response", "calibration.json")
vocab = load_json("split", "vocab.json")
id_to_name = {v: k for k, v in vocab.items()}
log = load_csv("response", "action_log.csv")
part = set(load_json("split", "test_partition.json")["final"])


def choose(label, cond, source=None, attack=None, max_len=6000):
    for r in log:
        if (r["label"] == label and int(r["n_syscalls"]) <= max_len and cond(r)
                and (source is None or r["source"] == source) and (attack is None or r["attack_type"] == attack)):
            return int(r["record_index"])


killed = lambda r: r["killed"] == "True"
contained = lambda r: r["contained"] == "True"
untouched = lambda r: r["contained"] == "False" and r["alerted"] == "False"
alert_only = lambda r: r["alerted"] == "True" and r["contained"] == "False"
MENU = [
    ("Heartbleed exploit on a web server (LID-DS CVE-2014-0160)", choose("attack", killed, attack="CVE-2014-0160")),
    ("nginx exploit (PLAID)", choose("attack", killed, attack="nginx")),
    ("FTP password brute-force, Hydra (ADFA-LD)", choose("attack", killed, attack="Hydra_FTP")),
    ("redis exploit (PLAID)", choose("attack", killed, attack="redis")),
    ("Web shell (ADFA-LD)", choose("attack", contained, attack="Web_Shell")),
    ("Add-superuser attack (ADFA-LD) - a hard case", choose("attack", alert_only, attack="Adduser")),
    ("NORMAL Apache web server traffic (LID-DS)", choose("benign", untouched, source="LID-DS/CVE-2014-0160")),
    ("NORMAL program run (PLAID)", choose("benign", untouched, source="PLAID")),
    ("NORMAL server process (ADFA-LD)", choose("benign", untouched, source="ADFA-LD")),
]


def pick_from_menu():
    print(f"\n{C['bold']}AAR LIVE MONITOR{C['reset']} - choose a process to stream "
          f"(real traces from the FINAL test set):\n")
    for k, (name, _) in enumerate(MENU, 1):
        print(f"  {k}. {name}")
    print(f"  0. Random recording from the FINAL test set")
    choice = sys.argv[1] if len(sys.argv) > 1 else input("\nNumber: ").strip()
    if choice == "0":
        import random
        return random.choice(sorted(part)), "Random FINAL recording"
    name, index = MENU[int(choice) - 1]
    return index, name


def main():
    index, name = pick_from_menu()
    rec = load_jsonl_gz("split", "test.jsonl.gz")[index]
    ids = rec["token_ids"]
    d = dataset_of(rec["source"])
    norm = cal["window_normalisation_from_benign_val"][d]
    thresholds = Thresholds(norm["median"], norm["p99"], cal["alert_high_z"], cal["alert_low_z"],
                            cal["isolate_threshold_z_per_dataset"][d], cal["isolate_release_z_per_dataset"][d],
                            cal["kill_threshold_z_per_dataset"][d])
    model = rnn.load_model(path("models", "lstm", "results", "lstm_model.pt"))

    print(f"\n{C['bold']}Streaming:{C['reset']} {name}")
    print(f"  recording {rec['recording_id']}  |  dataset {d}  |  {len(ids):,} system calls")
    print(f"  thresholds (confidence z): alert {thresholds.alert_high:.2f} (off {thresholds.alert_low:.2f}), "
          f"isolate {thresholds.isolate:.2f}, kill {thresholds.kill:.2f}; isolate/kill need {cal['sustain_windows']} "
          f"windows in a row")
    stand_in = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
    print(f"  stand-in process started: PID {stand_in.pid} (it is terminated if the kill switch fires)")
    print(f"  {C['dim']}ground truth is hidden until the end{C['reset']}\n")
    time.sleep(1.5)

    delay = min(0.05, TARGET_SECONDS / max(len(ids), 1))
    every = max(1, math.ceil(len(ids) / MAX_LINES))
    switch = KillSwitch(cal["alpha"], cal["sustain_windows"], thresholds,
                        on_kill=lambda s: terminate_process(stand_in.pid))
    nll = deque(maxlen=N - 1)
    state, logits = None, None
    status, last_event = "NORMAL", ""
    print(f"{C['dim']}{'syscall #':>9}  {'name':<16} {'window z':>9}  {'confidence':>10}  status{C['reset']}")
    with torch.no_grad():
        for t, tok in enumerate(ids):
            if logits is not None:  # surprise of this syscall given everything before it
                nll.append(-torch.log_softmax(logits[0, -1], dim=-1)[tok].item())
            logits, state = model(torch.tensor([[tok]]), state)
            event, z = None, None
            if t >= N - 1:  # a complete 7-syscall window ends at this syscall
                window_score = sum(nll) / len(nll)
                z = (window_score - norm["median"]) / (norm["p99"] - norm["median"])
                event = switch.update(window_score)
                status = ("KILLED" if switch.killed else "ISOLATED" if switch.isolated
                          else "ALERT" if switch.alert else "NORMAL")
            if event or t % every == 0:
                col = C[STATE_COLOUR[status]]
                zs = f"{z:9.3f}" if z is not None else f"{'-':>9}"
                cs = f"{switch.confidence:10.3f}" if z is not None else f"{'-':>10}"
                print(f"{t + 1:>9}  {id_to_name.get(tok, '?'):<16} {zs}  {cs}  {col}{status}{C['reset']}")
            if event:
                msg = {"alert": ("ALERT RAISED", "yellow"), "clear": ("alert cleared", "green"),
                       "isolate": ("PROCESS ISOLATED (reversible containment)", "orange"),
                       "release": ("process released from isolation", "green"),
                       "kill": ("KILL SWITCH FIRED - terminating the process", "red")}[event]
                print(f"{C[msg[1]]}{C['bold']}{C['invert']}  >>> {msg[0]} at system call {t + 1} "
                      f"(confidence {switch.confidence:.3f})  {C['reset']}")
                last_event = msg[0]
                time.sleep(1.2)
            if switch.killed:
                break
            time.sleep(delay)

    time.sleep(0.5)
    alive = stand_in.poll() is None
    if alive:
        stand_in.kill()
    executed = (switch.window + N) if switch.killed else len(ids)
    print(f"\n{C['bold']}RESULT{C['reset']}")
    print(f"  final status      : {C[STATE_COLOUR[status]]}{C['bold']}{status}{C['reset']}")
    print(f"  system calls run  : {executed:,} of {len(ids):,}"
          + (f"  ({len(ids) - executed:,} never executed - stopped early)" if switch.killed else ""))
    print(f"  stand-in PID {stand_in.pid}: " + ("TERMINATED by the kill switch" if switch.killed and not alive
                                             else "left running (stopped after the demo)"))
    truth = f"ATTACK ({rec['attack_type']})" if rec["label"] == "attack" else "NORMAL"
    correct = (rec["label"] == "attack") == (status != "NORMAL")
    print(f"  ground truth      : {C['bold']}{truth}{C['reset']}   ->  "
          + (f"{C['green']}detector was right{C['reset']}" if correct else f"{C['red']}detector missed it{C['reset']}"))


if __name__ == "__main__":
    main()
