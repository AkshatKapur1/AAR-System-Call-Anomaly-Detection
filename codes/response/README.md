# Layer 5 — confidence-gated, hysteresis-based kill switch

Turns the best model's anomaly scores into **actions on a running process**:
raise / clear an alert, and kill the process when the evidence is strong and
sustained. Each val/test recording is replayed window by window, in order, as
if it were a live process.

## Run

```
python codes/response/simulate_response.py   # ~2 min on CPU
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
3. **Confidence** `c_t` = running mean of `z` since the process started
   (selected). An EWMA `c_t = (1−α)c_{t−1} + α z_t` was also tried.
4. **Alert** (hysteresis): raised when `c_t > alert_high`, cleared only when
   `c_t < alert_low = alert_high − 0.25`, so it does not flap on/off.
5. **Kill** (confidence gate): when `c_t > kill` for **10 consecutive
   windows** — a single odd window can never kill a process. Final.
   Optional **alert escalation**: also kill when the alert has stayed on for
   more than `escalate_after` windows in a row (off by default — see below).
6. **Action**: `terminate_process(pid)` — `SIGKILL` on Linux/macOS,
   `taskkill /F` on Windows.

## Calibration and tuning (`simulate_response.py`) — FINAL never used for a choice

| Parameter | How it is set | Value |
|---|---|---|
| Window median / p99 per dataset | benign val windows | `calibration.json` |
| Kill threshold per dataset | 99th percentile of the highest *sustained* confidence over benign val processes (~1% of normal val processes would ever be killed) | ADFA-LD 1.16, LID-DS 0.95, PLAID 1.12 |
| Confidence type (α) and sustain | grid α ∈ {running mean, 0.01 … 0.5} × sustain ∈ {1, 5, 10, 25, 50, 100}, picked on **TUNE**: most attacks killed with ≤ 2% TUNE false kills | running mean, 10 windows |
| Alert threshold | max Youden's J (attacks alerted % + normal not alerted %) on **TUNE** | high 0.338, low 0.088 |
| Alert escalation | `escalate_after` per dataset = a percentile (99.5 … 90) of the longest continuous alert over benign val processes; percentile or *off* picked on **TUNE** with the same ≤ 2% false-kill rule | **off** |

## Results on FINAL (583 normal, 716 attack processes)

| | Layer 5 (online, window by window) | Layer 4 (after the whole recording) |
|---|---|---|
| **Kill**: attacks stopped | **123 / 716 (17.2%)** | strict: 107 / 716 |
| **Kill**: normal processes killed | 8 / 583 (1.4%) | strict: 6 / 583 |
| **Alert**: attacks alerted | 604 / 716 (84.4%) | balanced: 638 / 716 |
| **Alert**: normal processes alerted | 244 / 583 (41.9%) | balanced: 202 / 583 |
| System calls run before the kill (median) | **28** (6.7% of the recording) | whole recording |

- **Online kills more attacks and earlier** than the whole-recording decision:
  half of the stopped attacks are killed within 28 system calls.
- **Heartbleed (CVE-2014-0160): 58 / 77 stopped** online (Layer 4 strict:
  20 / 77) — the exploit's burst at the start is caught before the long normal
  tail of the process dilutes it. 0 / 65 normal LID-DS CVE-2014-0160
  processes killed.
- **Hysteresis**: alert raises on normal processes **253 with vs 352
  without** (−28%); 869 vs 1,210 over all processes.
- Per dataset: LID-DS 61 / 81 attacks stopped, PLAID 30 / 117 (nginx 17 / 19),
  ADFA-LD 32 / 518. Alerts reach ≥ 67% of every attack type.
- **Live demo** (`response_summary.json → live_demo`): a real dummy process
  bound to the switch while replaying ADFA-LD Hydra-FTP attack
  `UAD-Hydra-FTP-1-3523` was actually terminated by `terminate_process` at
  window 9 (16 system calls).
- Check: the vectorised simulation and the `KillSwitch` class give identical
  kill decisions and kill times on 300 / 300 FINAL processes.

### Alert escalation (option tested, not adopted)

Idea: an attack that stays alerted for a long time should eventually be
killed. Result on TUNE (306 attacks, 251 normal processes):

| Escalation | Attacks killed | Normal killed | Within ≤ 2% false kills |
|---|---|---|---|
| **off (selected)** | **46** | **5 (2.0%)** | yes |
| 99.5th pct | 48 (+2) | 7 (2.8%) | no |
| 98th pct | 64 (+18) | 11 (4.4%) | no |
| 95th pct | 90 (+44) | 16 (6.4%) | no |
| 90th pct | 135 (+89) | 29 (11.6%) | no |

Many normal processes also stay alerted for long stretches, so a long alert
alone is weak evidence: every extra attack killed costs roughly one extra
normal process. Escalation therefore only trades false kills for kills (like
loosening the kill budget) and stays **off**; FINAL results are unchanged.
It remains available as a policy switch (`Thresholds.escalate_after`) for
low-risk services where more false kills are acceptable; all settings are in
`escalation_tuning.csv`.

### Disclosure — first run

The first run offered only EWMA confidence. On FINAL it stopped 62 / 716
attacks with 14 / 583 false kills (alerts 435 / 716 vs 267 / 583) — worse than
Layer 4, because an EWMA tracks short spikes and long normal processes have
thousands of chances to spike, forcing a very high kill line. The running-mean
option was then added because Layer 4 had found the recording **mean** to be
the best aggregation; the choice between all options is still made on TUNE
only. Both families are in `tuning_results.csv`.

## Outputs (`response/`)

| File | Contents |
|---|---|
| `calibration.json` | normalisation, kill thresholds per dataset, alert thresholds, chosen α / sustain |
| `tuning_results.csv` | all 42 (α, sustain) settings on TUNE + val false-kill rate |
| `escalation_tuning.csv` | alert-escalation settings on TUNE (off selected) |
| `kill_log.csv` | one row per FINAL process: alerted / first alert syscall, killed / kill reason / kill syscall, alert raises with & without hysteresis |
| `per_dataset.csv`, `per_attack_type.csv` | kills, false kills, alerts, time to kill |
| `response_summary.json` | totals, hysteresis effect, Layer 4 comparison, live demo |
| `plots/example_timelines.png` | confidence over time for an attack (killed) and a normal process |
| `plots/time_to_kill.png` | share of attacks stopped vs system calls executed |
| `plots/per_attack_response.png` | alerted and killed per attack type |

## Limitations

- Simulated replay of recorded traces; live use needs a syscall stream per
  process (ptrace/eBPF) feeding `KillSwitch.update`.
- ADFA-LD Adduser / Meterpreter and PLAID ssh / ftp / cowroot are alerted but
  rarely killed — they look normal at the syscall-name level.
- The ~1.4% false-kill rate means roughly 1 in 70 normal processes would be
  killed; a deployment would whitelist critical services or require operator
  confirmation for them.
