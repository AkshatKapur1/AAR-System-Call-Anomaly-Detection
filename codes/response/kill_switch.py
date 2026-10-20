"""
Layer 5: confidence-gated, hysteresis-based per-process kill switch.

Consumes the best model's per-window anomaly scores (Layer 4 picked the LSTM
in `stream` mode) one window at a time, as a running process would produce
them, and decides per process:

  confidence c_t = running mean of z_1..z_t              (alpha = 0)
             c_t = (1 - alpha) * c_{t-1} + alpha * z_t    (alpha > 0, EWMA, c_0 = 0)
      z_t = (window score - median) / (p99 - median), using the benign val
      windows of the process's environment (dataset) -> 0 = typical normal,
      1 = the 99th percentile of normal windows.

  ALERT   raised when c_t > alert_high, cleared only when c_t < alert_low
          (hysteresis: alert_low < alert_high, so the alert does not flap).
  KILL    when c_t > kill for `sustain` consecutive windows (confidence gate:
          a single odd window can never kill a process), OR
          ESCALATION: when the alert has stayed on for more than
          `escalate_after` consecutive windows - longer than almost any normal
          process ever stays alerted. Kill is final.

All parameters come from codes/response/simulate_response.py: thresholds are
calibrated on benign val, alpha / sustain / alert level are chosen on the
TUNE part of test. Nothing here looks at labels.
"""
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field


@dataclass
class Thresholds:
    median: float
    p99: float
    alert_high: float
    alert_low: float
    kill: float
    escalate_after: int = 0     # 0 = no escalation


@dataclass
class KillSwitch:
    alpha: float
    sustain: int
    thresholds: Thresholds
    on_alert: callable = None
    on_kill: callable = None
    confidence: float = 0.0
    alert: bool = False
    killed: bool = False
    window: int = -1
    run_above_kill: int = 0
    z_sum: float = 0.0
    alert_run: int = 0
    kill_reason: str = ""
    events: list = field(default_factory=list)

    def update(self, window_score):
        """Feed one window score; returns 'kill', 'alert', 'clear' or None."""
        if self.killed:
            return None
        self.window += 1
        t = self.thresholds
        z = (window_score - t.median) / max(t.p99 - t.median, 1e-12)
        if self.alpha == 0:
            self.z_sum += z
            self.confidence = self.z_sum / (self.window + 1)
        else:
            self.confidence = (1 - self.alpha) * self.confidence + self.alpha * z
        self.run_above_kill = self.run_above_kill + 1 if self.confidence > t.kill else 0
        if self.run_above_kill >= self.sustain:
            return self._kill("confidence")
        event = None
        if not self.alert and self.confidence > t.alert_high:
            self.alert, event = True, "alert"
            self.events.append(("alert", self.window, self.confidence))
            if self.on_alert:
                self.on_alert(self)
        elif self.alert and self.confidence < t.alert_low:
            self.alert, event = False, "clear"
            self.events.append(("clear", self.window, self.confidence))
        self.alert_run = self.alert_run + 1 if self.alert else 0
        if t.escalate_after and self.alert_run > t.escalate_after:
            return self._kill("escalation")
        return event

    def _kill(self, reason):
        self.killed = True
        self.kill_reason = reason
        self.events.append(("kill:" + reason, self.window, self.confidence))
        if self.on_kill:
            self.on_kill(self)
        return "kill"


def terminate_process(pid):
    """The actual response action: kill the process (SIGKILL on Linux/macOS,
    TerminateProcess on Windows)."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    else:
        os.kill(pid, signal.SIGKILL)


def demo(window_scores, alpha, sustain, thresholds):
    """End-to-end demo of the action path: start a harmless dummy process,
    replay a recording's window scores into a KillSwitch bound to it, and
    terminate the process when the switch fires. Returns a short report."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    switch = KillSwitch(alpha, sustain, thresholds, on_kill=lambda s: terminate_process(proc.pid))
    for score in window_scores:
        if switch.update(score) == "kill":
            break
    time.sleep(0.5)
    alive = proc.poll() is None
    if alive:
        proc.kill()
    return {"pid": proc.pid, "killed_by_switch": switch.killed and not alive,
            "kill_window": switch.window if switch.killed else None, "events": switch.events}
