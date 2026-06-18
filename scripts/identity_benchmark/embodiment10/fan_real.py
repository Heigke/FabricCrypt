"""Embodiment Phase 10 — Task A.

Verify Phase 9 fan-control 49.8% transplant penalty against REAL hardware
response. PWM is not writable on either host (hp driver exposes only
pwm1_enable, which rejects writes — Invalid argument), so we use a
passive-observation variant:

The 'controller' modulates COMPUTE LOAD (a sleep/burst CPU loop) — the only
writable thermal forcing function. The closed-loop policy maps observed
(T_APU, T_GPU, P_GPU) -> load_level in [0,1] to track a target APU temp.
Per-chassi thermal RC differs (cooling efficacy, heat capacity, fan curve);
a policy fit on host A should overshoot/undershoot when transplanted to
host B.

Episodes are kept to <=30s with 60s wait_cool between, APU <70C abort.

Six conditions per host:
  - learned_ikaros        controller trained on ikaros
  - learned_daedalus      controller trained on daedalus
  - learned_shuffle       trained on time-shuffled samples (control)
  - random_init           random-weight controller (no training)
  - constant_mid          fixed load=0.5
  - off                   fixed load=0.0

Pre-reg gate: A_train_on_eval RMSE error to T_target is <= 0.80 x
A_train_on_other RMSE (>=20% transplant penalty).
"""
from __future__ import annotations
import argparse, json, time, subprocess, threading, os, sys
from pathlib import Path
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[2] / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment10"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- thermal / safety constants --------
APU_ABORT_C   = 70.0
APU_RESUME_C  = 55.0
COOL_MAX_S    = 180.0
EPISODE_S     = 25.0     # <30s
EP_DT         = 0.25     # 4 Hz control loop
TARGET_C      = 50.0
N_SEEDS       = 8


def read_apu_c():
    with open("/sys/class/thermal/thermal_zone0/temp") as f:
        return int(f.read().strip()) / 1000.0


def read_gpu_c():
    try:
        with open("/sys/class/hwmon/hwmon7/temp1_input") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return float("nan")


def read_gpu_power_w():
    try:
        with open("/sys/class/hwmon/hwmon7/power1_input") as f:
            return int(f.read().strip()) / 1e6
    except Exception:
        return float("nan")


def read_gpu_freq_mhz():
    try:
        with open("/sys/class/hwmon/hwmon7/freq1_input") as f:
            return int(f.read().strip()) / 1e6
    except Exception:
        return float("nan")


def wait_cool(label=""):
    t0 = time.time()
    while True:
        t = read_apu_c()
        if t <= APU_RESUME_C:
            return t, time.time() - t0
        if time.time() - t0 > COOL_MAX_S:
            return t, time.time() - t0
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Compute-load actuator (CPU spin with duty-cycle = load_level)
# ---------------------------------------------------------------------------
class LoadActuator:
    """Simple CPU duty-cycle load using a background thread."""
    def __init__(self):
        self.level = 0.0
        self.stop = False
        self.th = None

    def _worker(self):
        period = 0.05  # 50ms slot
        x = np.random.randn(96, 96).astype(np.float64)
        while not self.stop:
            lvl = max(0.0, min(1.0, self.level))
            t0 = time.time()
            # active fraction = lvl
            t_active = period * lvl
            t_idle = period * (1.0 - lvl)
            t_end = t0 + t_active
            while time.time() < t_end:
                # small matmul keeps CPU busy
                x = np.tanh(x @ x.T * 0.1 + 0.01)
                if x.std() > 10:
                    x = x / 10.0
            if t_idle > 0:
                time.sleep(t_idle)

    def start(self):
        self.stop = False
        self.th = threading.Thread(target=self._worker, daemon=True)
        self.th.start()

    def set_level(self, lvl):
        self.level = float(max(0.0, min(1.0, lvl)))

    def shutdown(self):
        self.stop = True
        self.level = 0.0
        if self.th is not None:
            self.th.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Episode runner: drive controller for EPISODE_S seconds, record telemetry
# ---------------------------------------------------------------------------
def run_episode(actuator: LoadActuator, controller, target_c=TARGET_C, dur=EPISODE_S):
    """Returns dict with telemetry traces."""
    Ts, GPUTs, Ps, Fs, Levels = [], [], [], [], []
    t0 = time.time()
    t_last = t0
    while time.time() - t0 < dur:
        T = read_apu_c()
        if T >= APU_ABORT_C:
            actuator.set_level(0.0)
            return {"abort": True, "T": Ts, "Tgpu": GPUTs, "P": Ps,
                    "F": Fs, "level": Levels, "T_max": float(max(Ts) if Ts else T)}
        GT = read_gpu_c(); GP = read_gpu_power_w(); GF = read_gpu_freq_mhz()
        obs = np.array([T, GT if np.isfinite(GT) else 40.0,
                        GP if np.isfinite(GP) else 5.0,
                        T - target_c], dtype=np.float32)
        lvl = controller.act(obs)
        actuator.set_level(lvl)
        Ts.append(T); GPUTs.append(GT); Ps.append(GP); Fs.append(GF); Levels.append(lvl)
        # honor EP_DT
        dt = time.time() - t_last
        if dt < EP_DT:
            time.sleep(EP_DT - dt)
        t_last = time.time()
    actuator.set_level(0.0)
    return {"abort": False, "T": Ts, "Tgpu": GPUTs, "P": Ps,
            "F": Fs, "level": Levels, "T_max": float(max(Ts))}


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------
class LinearController:
    """level = sigmoid(W . obs + b).   4 inputs, 1 output."""
    def __init__(self, theta=None, rng=None):
        if theta is None:
            r = rng or np.random.default_rng(0)
            theta = r.normal(0, 0.3, 5).astype(np.float32)
        self.theta = theta.astype(np.float32)

    def act(self, obs):
        z = float(self.theta[:4] @ obs + self.theta[4])
        return 1.0 / (1.0 + np.exp(-z))


class ConstantController:
    def __init__(self, val=0.5):
        self.val = float(val)
    def act(self, obs):
        return self.val


# ---------------------------------------------------------------------------
# Calibration trace -> chassi thermal model (offline)
# ---------------------------------------------------------------------------
def calibration_trace(actuator: LoadActuator, dur=40.0):
    """Drive a square-wave load and record (T, level) trajectory at 4Hz.
    Aborts if APU >= 65C (well below shutdown trip)."""
    Ts, Ls = [], []
    t0 = time.time()
    while time.time() - t0 < dur:
        T = read_apu_c()
        if T >= 65.0:
            actuator.set_level(0.0)
            time.sleep(0.5)
            # cool briefly to keep trace alive
            while read_apu_c() > 55.0 and time.time() - t0 < dur:
                time.sleep(0.5)
            continue
        # 4s on / 4s off square wave
        phase = ((time.time() - t0) % 8.0) < 4.0
        lvl = 0.7 if phase else 0.0
        actuator.set_level(lvl)
        Ts.append(T); Ls.append(lvl)
        time.sleep(0.25)
    actuator.set_level(0.0)
    return np.array(Ts), np.array(Ls)


def fit_thermal_rc(Ts, Ls, dt=0.25):
    """Fit T[t+1] = T[t] + dt/tau*(T_amb + heat*L[t] - T[t])  -> two-param LS.
    Use bounded scipy.optimize to avoid degenerate tau."""
    if len(Ts) < 16:
        return {"tau": 60.0, "heat": 30.0, "T_amb": float(Ts.mean()) if len(Ts) else 40.0}
    try:
        from scipy.optimize import minimize  # type: ignore
    except Exception:
        # fallback to LS with bounds
        minimize = None
    dT = np.diff(Ts) / dt
    T = Ts[:-1]; L = Ls[:-1]
    if minimize is None:
        A = np.column_stack([-T, L, np.ones_like(T)])
        sol, *_ = np.linalg.lstsq(A, dT, rcond=None)
        inv_tau, heat_div_tau, amb_div_tau = sol
        tau = float(np.clip(1.0 / max(inv_tau, 1e-3), 5.0, 300.0))
        heat = float(np.clip(heat_div_tau * tau, 5.0, 80.0))
        amb = float(np.clip(amb_div_tau * tau, 30.0, 50.0))
        return {"tau": tau, "heat": heat, "T_amb": amb}

    def loss(params):
        tau, heat, amb = params
        pred = (1.0/tau) * (amb + heat * L - T)
        return float(np.mean((pred - dT) ** 2))

    res = minimize(loss, x0=[60.0, 30.0, 42.0],
                   bounds=[(5.0, 300.0), (5.0, 80.0), (30.0, 50.0)],
                   method="L-BFGS-B")
    tau, heat, amb = res.x
    return {"tau": float(tau), "heat": float(heat), "T_amb": float(amb)}


def simulate_episode(params, controller, dur=EPISODE_S, dt=EP_DT,
                     T_init=42.0, target_c=TARGET_C, seed=0):
    rng = np.random.default_rng(seed)
    T = T_init; T_amb = params["T_amb"]; tau = params["tau"]; heat = params["heat"]
    Ts = []; Ls = []
    n = int(dur / dt)
    for _ in range(n):
        obs = np.array([T, T - 4.0, 8.0 + 5.0 * 0.5, T - target_c], dtype=np.float32)
        lvl = controller.act(obs)
        dT = dt / tau * (T_amb + heat * lvl - T) + rng.normal(0, 0.1)
        T = float(T + dT)
        Ts.append(T); Ls.append(lvl)
    return np.array(Ts), np.array(Ls)


def cem_train(params, target_c=TARGET_C, n_iters=10, pop=30, elite_frac=0.25,
              seed=0, shuffle_time=False):
    """Cross-entropy method: fit 5-param linear controller against simulator.
    If shuffle_time, the simulator scrambles dt order between samples (a
    control that breaks temporal credit assignment)."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 0.5, 5).astype(np.float32)
    sigma = np.ones(5, dtype=np.float32) * 1.0
    for it in range(n_iters):
        thetas = rng.normal(mu, sigma, size=(pop, 5)).astype(np.float32)
        rewards = []
        for th in thetas:
            ctrl = LinearController(theta=th)
            if shuffle_time:
                # run k=5 mini-episodes and average reward (breaks temporal pattern)
                rs = []
                for k in range(5):
                    Ts, Ls = simulate_episode(params, ctrl, dur=EPISODE_S/5,
                                              dt=EP_DT, T_init=40.0 + rng.uniform(0, 6),
                                              target_c=target_c, seed=it*pop + k)
                    rs.append(-np.mean((Ts - target_c) ** 2))
                rewards.append(np.mean(rs))
            else:
                Ts, Ls = simulate_episode(params, ctrl, target_c=target_c,
                                          seed=it*pop)
                rewards.append(-np.mean((Ts - target_c) ** 2))
        rewards = np.array(rewards)
        order = np.argsort(-rewards)
        elite = thetas[order[:max(2, int(pop*elite_frac))]]
        mu = elite.mean(axis=0); sigma = elite.std(axis=0) + 1e-3
    return mu


# ---------------------------------------------------------------------------
# Main: capture calibration -> train controllers -> evaluate
# ---------------------------------------------------------------------------
def rmse(traj, target):
    return float(np.sqrt(np.mean((np.asarray(traj) - target) ** 2)))


def run_local(host_label, args):
    """Run calibration + episodes locally on the host this script runs on."""
    actuator = LoadActuator(); actuator.start()
    log = {"host": host_label, "started": time.time()}
    try:
        print(f"[{host_label}] Phase 0: calibration trace (target temp window)")
        T0, _ = wait_cool(); print(f"  cooled to {T0:.1f}C")
        Tcal, Lcal = calibration_trace(actuator, dur=18.0)
        params = fit_thermal_rc(Tcal, Lcal)
        print(f"  fitted RC: tau={params['tau']:.1f}s heat={params['heat']:.1f}C T_amb={params['T_amb']:.1f}C")
        log["calibration"] = {"T": Tcal.tolist(), "L": Lcal.tolist(), "params": params}
        wait_cool(); time.sleep(2)

        print(f"[{host_label}] Training controllers offline on local RC simulator")
        theta_normal = cem_train(params, n_iters=12, pop=40, seed=0, shuffle_time=False)
        theta_shuffle = cem_train(params, n_iters=12, pop=40, seed=0, shuffle_time=True)
        log["theta_normal"] = theta_normal.tolist()
        log["theta_shuffle"] = theta_shuffle.tolist()
        log["theta_random"] = np.random.default_rng(42).normal(0, 0.5, 5).astype(float).tolist()
        log["rc_params"] = params

        # Episodes follow when the transplant theta is provided externally.
        # We just save calibration here. Evaluation episodes will be a 2nd pass.
        log["finished"] = time.time()
        return log
    finally:
        actuator.shutdown()


def eval_local(host_label, thetas_named, n_seeds, target_c, args):
    """Evaluate each controller for n_seeds episodes; return per-episode RMSE."""
    actuator = LoadActuator(); actuator.start()
    log = {"host": host_label, "target_c": target_c}
    per_cond = {k: [] for k in thetas_named}
    try:
        for s in range(n_seeds):
            for name, theta in thetas_named.items():
                Tcur, dt_cool = wait_cool()
                if Tcur > APU_RESUME_C + 5:
                    print(f"[{host_label}] WARN cool stuck at {Tcur:.1f}C after {dt_cool:.0f}s; aborting seed {s}")
                    break
                if name == "constant_mid":
                    ctrl = ConstantController(val=0.5)
                elif name == "off":
                    ctrl = ConstantController(val=0.0)
                else:
                    ctrl = LinearController(theta=np.array(theta, dtype=np.float32))
                ep = run_episode(actuator, ctrl, target_c=target_c, dur=EPISODE_S)
                # use second half (after warmup) to compute RMSE to target
                Ts = np.array(ep["T"])
                if len(Ts) < 8:
                    err = float("nan")
                else:
                    err = rmse(Ts[len(Ts)//2:], target_c)
                per_cond[name].append({"seed": s, "rmse": err, "T_max": ep["T_max"],
                                       "abort": ep["abort"],
                                       "T_final": float(Ts[-1]) if len(Ts) else None,
                                       "level_mean": float(np.mean(ep["level"])) if ep["level"] else None})
                print(f"  s{s:02d} {name:18s} rmse={err:.3f}  T_max={ep['T_max']:.1f}C  abort={ep['abort']}")
        log["per_condition"] = per_cond
        return log
    finally:
        actuator.shutdown()


def bootstrap_diff_pct(a, b, n_boot=2000, seed=1):
    """Percent penalty of b vs a = (b - a)/a * 100. Both lists of RMSEs (lower=better)."""
    rng = np.random.default_rng(seed)
    a = np.array([x for x in a if np.isfinite(x)])
    b = np.array([x for x in b if np.isfinite(x)])
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs.append((sb.mean() - sa.mean()) / max(sa.mean(), 1e-9) * 100.0)
    diffs = np.sort(diffs)
    return float(np.mean(diffs)), float(diffs[int(0.025*n_boot)]), float(diffs[int(0.975*n_boot)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["calibrate", "eval"], required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--thetas", help="JSON file w/ thetas dict")
    ap.add_argument("--n_seeds", type=int, default=N_SEEDS)
    ap.add_argument("--target_c", type=float, default=TARGET_C)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "calibrate":
        log = run_local(args.host, args)
    else:
        with open(args.thetas) as f:
            thetas = json.load(f)
        log = eval_local(args.host, thetas, args.n_seeds, args.target_c, args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(log, indent=2, default=float))
    print(f"saved {args.out}")
