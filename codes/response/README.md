# Layer 5 — confidence-gated, hysteresis-based graduated response

Turns the best model's anomaly scores into **actions on a running process**:
**alert → isolate → kill**. Each val/test recording is replayed window by
window, in order, as if it were a live process.

## Run

```
python codes/response/simulate_response.py   # ~3 min on CPU
```

Needs `numpy`, `scipy`, `torch`, `matplotlib`. Uses Layer 4's choice from
`evaluation/best_model.json` (LSTM, `stream` mode).

## How it works (`kill_switch.py`)

For every new 7-syscall window of a process:

1. **Window score** — LSTM `stream` mode: mean NLL of the window's calls 2–7
   given everything the process did before.
2. **Normalise** per environment (dataset) with benign val windows:
   `z = (score − median) / (p99 − median)` → 0 = typical normal window,
   1 = 99th percentile of normal windows.
3. **Confidence** `c_t` = running mean of `z` since the process started.
4. **Alert** (hysteresis): on when `c_t > alert_high`, off only when
   `c_t < alert_high − 0.25`, so it does not flap.
5. **Isolate** (reversible): when `c_t > isolate` for **10 consecutive
   windows**; released when `c_t < isolate − 0.25`. The process keeps running
   inside a containment and is still monitored, so it can be released or
   killed later. (`on_isolate` / `on_release` hooks — in deployment a sandbox
   such as a cgroup or a network cut.)
6. **Kill** (final): when `c_t > kill` for **10 consecutive windows** —
   `SIGKILL` on Linux/macOS, `taskkill /F` on Windows.

A single odd window can never isolate or kill a process.

## Calibration — FINAL never used for a choice

| Parameter | How it is set | Value |
|---|---|---|
| Window median / p99 per dataset | benign val windows | `calibration.json` |
| Confidence type and sustain | grid {running mean, EWMA α 0.01–0.5} × sustain {1 … 100}, picked on **TUNE**: most attacks killed with ≤ 2% TUNE normal processes killed | running mean, 10 windows |
| Kill line per dataset | ~1% of benign val processes would ever be killed | ADFA-LD 1.16, LID-DS 0.95, PLAID 1.12 |
| Isolate line per dataset | ~2% of benign val processes would ever be isolated (most conservative level tested) | ADFA-LD 1.01, LID-DS 0.88, PLAID 0.82 |
| Alert line | max Youden's J (attacks alerted % + normal not alerted %) on **TUNE** | 0.338 (off at 0.088) |

## Results on FINAL (583 normal, 716 attack processes)

| | Attacks | Normal processes |
|---|---|---|
| Alerted | 604 (84.4%) | 244 (41.9%) |
| **Contained at some point** (isolated or killed) | **164 (22.9%)** | 16 (2.7%) |
| **Still contained at the end** (killed or still isolated) | **142 (19.8%)** | 9 (8 killed, 1 isolated) |
| **Killed** (permanent) | **123 (17.2%)** | **8 (1.4%)** |
| System calls before the first action (median) | 22 (containment), 28 (kill) | — |

- **Isolation adds containment without adding a single kill:** +41 attacks
  contained (+19 still contained at the end) vs kill-only, while normal
  processes killed stay at 8. Of the 13 normal processes ever isolated, 7
  were released automatically, 1 was still isolated at the end and 5 were
  later killed (they are among the 8 kills, which happen with or without
  isolation); an isolated normal process spent 18% of its run isolated on
  average. Of the 102 attacks ever isolated, 61 were later killed, 19 stayed
  isolated and 22 were released.
- **Heartbleed (CVE-2014-0160): 58 / 77 killed** online, 0 / 65 normal
  Heartbleed-scenario processes isolated or killed.
- Per dataset (contained / killed): LID-DS 61 / 61 of 81, PLAID 50 / 30 of
  117 (nginx 17/17 of 19, redis 19/12 of 48), ADFA-LD 53 / 32 of 518.
- **Hysteresis**: alert raises on normal processes 253 with vs 345 without
  (−27%); 869 vs 1,187 over all processes.
- **Live demo**: a real dummy process bound to the switch while replaying
  ADFA-LD Hydra-FTP attack `UAD-Hydra-FTP-1-3523` was terminated at window 9
  (16 system calls) — `response_summary.json → live_demo`.
- Check: the vectorised simulation and the `KillSwitch` class give identical
  kill and isolation decisions on 300 / 300 FINAL processes.

Against Layer 4 (decides only after the whole recording, same FINAL records):
strict threshold 107 attacks / 6 normal flagged; balanced 638 / 202.

## Options evaluated (TUNE only) and why this design was kept

| Option | Idea | TUNE outcome | Kept? |
|---|---|---|---|
| EWMA confidence | react to recent windows only | best EWMA setting killed fewer attacks than the running mean | no — running mean selected |
| Alert escalation | kill when an alert lasts very long | every level broke the 2% kill limit (mildest: +2 attacks for +2 normal kills) | no — removed |
| **Isolate tier** | reversible containment between alert and kill | +26 attacks stopped (72 vs 46 of 306) with **no extra kills** (5 of 251) | **yes** |

Isolation sensitivity (`isolation_sensitivity.csv`, TUNE): looser isolation
lines stop more attacks (3% → 97, 5% → 118, 10% → 145 of 306) but isolate
more normal processes (20, 21, 29 of 251). Normal processes killed stay at 5
in every case. Disclosure: the pre-set target of ≤ 5% normal processes
disrupted on TUNE was not met by any level — the most conservative one gives
5.6% (14 of 251) — so that level is used and reported as is.

## Outputs (`response/`)

| File | Contents |
|---|---|
| `calibration.json` | normalisation, alert / isolate / kill lines per dataset, confidence type, sustain |
| `tuning_results.csv` | all 42 (confidence type, sustain) settings on TUNE |
| `isolation_sensitivity.csv` | isolation levels on TUNE |
| `action_log.csv` | one row per FINAL process: alerted, isolated (first syscall, count, share of life, at end), killed (syscall), contained |
| `per_dataset.csv`, `per_attack_type.csv` | alerts, containment, kills, time to action |
| `response_summary.json` | totals, hysteresis effect, Layer 4 comparison, live demo |
| `plots/example_timelines.png` | confidence over time: an attack killed, an attack isolated, a normal process left running |
| `plots/time_to_action.png` | share of attacks contained / killed vs system calls executed |
| `plots/per_attack_response.png` | alerted, contained and killed per attack type |

## Limitations

- Simulated replay of recorded traces; live use needs a syscall stream per
  process (ptrace/eBPF) feeding `KillSwitch.update`, and a real sandbox
  behind `on_isolate` / `on_release`.
- ADFA-LD Adduser / Meterpreter and PLAID ssh are alerted (≥ 67%) but
  rarely contained — they look normal at the syscall-name level.
- ~1.4% of normal processes are killed and ~2% isolated; a deployment would
  whitelist critical services or require operator confirmation for them.
