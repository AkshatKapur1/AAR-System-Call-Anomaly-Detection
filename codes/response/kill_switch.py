"""
Layer 5: confidence-gated, hysteresis-based graduated response per process.

Consumes the best model's per-window anomaly scores (Layer 4 picked the LSTM
in `stream` mode) one window at a time, as a running process would produce
them, and decides per process:

  confidence c_t = running mean of z_1..z_t              (alpha = 0)
             c_t = (1 - alpha) * c_{t-1} + alpha * z_t    (alpha > 0, EWMA, c_0 = 0)
      z_t = (window score - median) / (p99 - median), using the benign val
      windows of the process's environment (dataset) -> 0 = typical normal,
      1 = the 99th percentile of normal windows.

Three escalating responses, each with a confidence gate or hysteresis:
  ALERT    raised when c_t > alert_high, cleared only when c_t < alert_low.
  ISOLATE  (reversible) when c_t > isolate for `sustain` consecutive windows;
           released when c_t < isolate_release. The process keeps running
           inside a containment (e.g. network cut / sandbox) and is still
           monitored, so it can be released or killed later.
  KILL     (final) when c_t > kill for `sustain` consecutive windows.
A single odd window can never isolate or kill a process.

All parameters come from codes/response/simulate_response.py: thresholds are
calibrated on benign val, confidence type / sustain / alert level are chosen
on the TUNE part of test. Nothing here looks at labels.
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
    isolate: float
    isolate_release: float
    kill: float


@dataclass
class KillSwitch:
    alpha: float
    sustain: int
    thresholds: Thresholds
    on_alert: callable = None
    on_isolate: callable = None
    on_release: callable = None
    on_kill: callable = None
    confidence: float = 0.0
    alert: bool = False
    isolated: bool = False
    killed: bool = False
    window: int = -1
    run_above_kill: int = 0
    run_above_isolate: int = 0
    z_sum: float = 0.0
    events: list = field(default_factory=list)

    def update(self, window_score):
        """Feed one window score; returns the most severe event of this window
        ('kill', 'isolate', 'release', 'alert', 'clear') or None."""
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
            self.killed = True
            return self._event("kill", self.on_kill)

        event = None
        self.run_above_isolate = self.run_above_isolate + 1 if self.confidence > t.isolate else 0
        if not self.isolated and self.run_above_isolate >= self.sustain:
            self.isolated = True
            event = self._event("isolate", self.on_isolate)
        elif self.isolated and self.confidence < t.isolate_release:
            self.isolated = False
            event = self._event("release", self.on_release)

        if not self.alert and self.confidence > t.alert_high:
            self.alert = True
            event = event or self._event("alert", self.on_alert)
        elif self.alert and self.confidence < t.alert_low:
            self.alert = False
            event = event or self._event("clear", None)
        return event

    def _event(self, name, callback):
        self.events.append((name, self.window, self.confidence))
        if callback:
            callback(self)
        return name


def terminate_process(pid):
    """The kill action: SIGKILL on Linux/macOS, TerminateProcess on Windows."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    else:
        os.kill(pid, signal.SIGKILL)


def demo(window_scores, alpha, sustain, thresholds):
    """End-to-end demo of the kill path: start a harmless dummy process,
    replay a recording's window scores into a KillSwitch bound to it, and
    terminate the process when the switch fires. (Isolation is reported as an
    event; in deployment on_isolate/on_release map to a sandbox, e.g. a cgroup
    or a network cut.) Returns a short report."""
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
