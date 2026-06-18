"""Phase 9 Task C — closed-loop fan PWM control benchmark.

Task: at each step, output fan PWM (0..255). Minimise (T - T_target)^2 plus
energy cost lambda * PWM^2 over a horizon. Per-chassis thermal RC differs,
so a controller learned on ikaros should outperform a controller learned on
daedalus or naive baselines when deployed on ikaros.

Implementation notes:
- We attempted writable /sys/class/hwmon/*/pwm1 — not available on this
  laptop. Fall back to a simulated thermal RC parameterised by recorded
  per-chassis step responses (substrate trajectory used as "ambient drift").
- Simulator state: T_die[t+1] = T_die[t] + dt/tau * ( T_ambient
   - T_die[t] - cool_gain * pwm[t]/255 + heat_load[t] )
   where tau, cool_gain are PER-CHASSIS (fit from substrate trace amplitude
   & decay). heat_load[t] is the recorded power trace.
- Controllers:
    - learned_ikaros / learned_daedalus: tiny MLP policy trained via
      offline cross-entropy method (CEM) on each chassis simulator
    - constant_pwm: midpoint baseline
    - bang_bang_pid: PID with default Kp=2, Ki=0.1, Kd=0.5
- Eval: deploy each controller on each simulator. Pre-reg: learned_ikaros
  on ikaros simulator has RMS(T - T_target) at least 20% lower than the
  worst baseline AND at least 5% lower than the transplant (learned_daedalus
  on ikaros simulator).
"""
from __future__ import annotations
import json
import socket
from pathlib import Path
import numpy as np

HOST = socket.gethostname()
ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT_DIR = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment9"
SUB_DIR = OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

T_TARGET = 55.0
T_AMBIENT = 25.0
DT = 0.5
N_STEPS = 200
N_SEEDS = 30
LAMBDA_ENERGY = 1e-4


def fit_chassis_params(substrate: np.ndarray) -> dict:
    """Fit (tau, cool_gain, heat_load_trace) from recorded substrate.

    tau: from autocorrelation 1/e time of T trace
    cool_gain: choose so that pwm=128 maintains T near observed mean
    heat_load: the recorded P_w trace (already in W ~ thermal flux proxy)
    """
    T = substrate[:, 0]
    P = substrate[:, 1]
    # autocorr 1/e
    Tc = T - T.mean()
    if np.linalg.norm(Tc) < 1e-6:
        tau = 5.0
    else:
        acf = np.correlate(Tc, Tc, mode="full")[len(Tc)-1:] / (Tc @ Tc + 1e-9)
        # First lag where acf < 1/e
        below = np.where(acf < 1/np.e)[0]
        tau = float(below[0]) * 0.1 if len(below) else 5.0
        tau = max(2.0, min(30.0, tau))
    # Choose cool_gain such that at PWM=128, steady-state T = mean(T).
    # Steady: 0 = (T_amb - T - cg * 0.5 + mean(P)) → cg = 2*(T_amb - T + mean(P))/... let's just pick
    cool_gain = max(5.0, T.mean() - T_AMBIENT + P.mean()) * 2.0
    return {"tau": tau, "cool_gain": cool_gain, "heat_load": P.astype(np.float32)}


class FanSim:
    def __init__(self, params: dict, seed: int = 0):
        self.tau = params["tau"]
        self.cg = params["cool_gain"]
        self.heat = params["heat_load"]
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.T = T_AMBIENT + 30.0

    def reset(self):
        self.t = 0
        self.T = T_AMBIENT + 30.0 + self.rng.standard_normal() * 0.5
        return self._obs()

    def _obs(self):
        # Observation: [T, dT/dt estimate, recent heat_load]
        h = self.heat[self.t % len(self.heat)]
        return np.array([self.T - T_TARGET, h, 0.0], dtype=np.float32)

    def step(self, pwm: float):
        pwm = float(np.clip(pwm, 0, 255))
        h = self.heat[self.t % len(self.heat)]
        # noise (chassis stochasticity)
        noise = self.rng.standard_normal() * 0.1
        dT = DT / self.tau * (T_AMBIENT - self.T - self.cg * pwm / 255.0 + h * 0.5 + noise)
        self.T += dT
        self.t += 1
        reward = -((self.T - T_TARGET) ** 2) - LAMBDA_ENERGY * pwm ** 2
        return self._obs(), reward, self.t >= N_STEPS


# ---------------------------------------------------------------------------
# Tiny MLP policy (32 hidden, tanh, output → PWM via sigmoid * 255)
# ---------------------------------------------------------------------------
class MLPPolicy:
    def __init__(self, din=3, hid=32, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((din, hid)).astype(np.float32) * 0.3
        self.b1 = np.zeros(hid, dtype=np.float32)
        self.W2 = rng.standard_normal((hid, 1)).astype(np.float32) * 0.3
        self.b2 = np.zeros(1, dtype=np.float32)
        self.shape = (din, hid)

    @property
    def n_params(self):
        d, h = self.shape
        return d*h + h + h*1 + 1

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def set_params(self, theta):
        d, h = self.shape
        i = 0
        self.W1 = theta[i:i+d*h].reshape(d, h).astype(np.float32); i += d*h
        self.b1 = theta[i:i+h].astype(np.float32); i += h
        self.W2 = theta[i:i+h*1].reshape(h, 1).astype(np.float32); i += h
        self.b2 = theta[i:i+1].astype(np.float32)

    def act(self, obs):
        h = np.tanh(obs @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        return float(255.0 / (1.0 + np.exp(-z[0])))


def rollout(policy, sim, n_steps=N_STEPS):
    obs = sim.reset()
    total_r = 0.0
    Ts = []
    pwms = []
    for _ in range(n_steps):
        a = policy.act(obs)
        obs, r, done = sim.step(a)
        total_r += r
        Ts.append(sim.T)
        pwms.append(a)
        if done:
            break
    return total_r, np.array(Ts), np.array(pwms)


def cem_train(sim_params, n_iters=15, pop=40, elite_frac=0.2, seed=0):
    """Cross-entropy method on policy parameters."""
    pol = MLPPolicy(seed=seed)
    nparam = pol.n_params
    rng = np.random.default_rng(seed)
    mean = np.zeros(nparam, dtype=np.float32)
    std = np.ones(nparam, dtype=np.float32) * 0.3
    n_elite = max(2, int(pop * elite_frac))
    for it in range(n_iters):
        samples = rng.standard_normal((pop, nparam)).astype(np.float32) * std + mean
        rewards = np.zeros(pop, dtype=np.float32)
        for i, theta in enumerate(samples):
            pol.set_params(theta)
            sim = FanSim(sim_params, seed=it*100+i)
            r, _, _ = rollout(pol, sim)
            rewards[i] = r
        elite_idx = np.argsort(rewards)[-n_elite:]
        elites = samples[elite_idx]
        mean = elites.mean(0)
        std = elites.std(0) + 0.05
    pol.set_params(mean)
    return pol


# ---------------------------------------------------------------------------
# Baseline controllers
# ---------------------------------------------------------------------------
class ConstantPWM:
    def __init__(self, val=128): self.val = val
    def act(self, obs): return float(self.val)


class PIDController:
    def __init__(self, Kp=20.0, Ki=1.0, Kd=5.0):
        self.Kp = Kp; self.Ki = Ki; self.Kd = Kd
        self.integ = 0.0; self.prev_err = 0.0
    def reset(self): self.integ = 0.0; self.prev_err = 0.0
    def act(self, obs):
        err = obs[0]  # T - T_target (positive = too hot → more PWM)
        self.integ = np.clip(self.integ + err, -50, 50)
        deriv = err - self.prev_err
        self.prev_err = err
        u = self.Kp * err + self.Ki * self.integ + self.Kd * deriv
        return float(np.clip(128 + u, 0, 255))


def eval_controller(pol_or_ctrl, sim_params, n_runs=N_SEEDS):
    rms_list = []
    energy_list = []
    for s in range(n_runs):
        sim = FanSim(sim_params, seed=1000+s)
        ctrl = pol_or_ctrl
        if hasattr(ctrl, "reset"): ctrl.reset()
        _, Ts, pwms = rollout(ctrl, sim)
        rms_list.append(float(np.sqrt(np.mean((Ts - T_TARGET) ** 2))))
        energy_list.append(float(np.mean(pwms ** 2)))
    return {
        "rms_per_run": rms_list,
        "rms_mean": float(np.mean(rms_list)),
        "rms_std": float(np.std(rms_list)),
        "energy_mean": float(np.mean(energy_list)),
    }


def bootstrap_ci(arr, n=2000, alpha=0.05):
    arr = np.asarray(arr); rng = np.random.default_rng(0)
    bs = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)])
    return float(np.percentile(bs, 100*alpha/2)), float(np.percentile(bs, 100*(1-alpha/2)))


def main():
    # Load substrate traces from both hosts
    subs = {}
    for h in ("ikaros", "daedalus"):
        p = SUB_DIR / f"substrate_{h}.npy"
        if not p.exists():
            print(f"[!] missing {p}"); return
        subs[h] = np.load(p)

    params = {h: fit_chassis_params(subs[h]) for h in subs}
    print("Chassis params:")
    for h, pp in params.items():
        print(f"  {h}: tau={pp['tau']:.2f}s  cool_gain={pp['cool_gain']:.2f}  "
              f"mean_heat={pp['heat_load'].mean():.2f}W")

    # Train ikaros & daedalus policies
    print("Training ikaros policy (CEM)...")
    pol_ikaros = cem_train(params["ikaros"], n_iters=12, pop=30, seed=0)
    print("Training daedalus policy (CEM)...")
    pol_daedalus = cem_train(params["daedalus"], n_iters=12, pop=30, seed=0)

    # Eval matrix: each controller on each simulator
    controllers = {
        "learned_ikaros": pol_ikaros,
        "learned_daedalus": pol_daedalus,
        "constant_pwm": ConstantPWM(128),
        "pid_default": PIDController(),
    }
    results = {"host_running": HOST, "n_seeds": N_SEEDS, "matrix": {}}
    for sim_host in ("ikaros", "daedalus"):
        results["matrix"][sim_host] = {}
        for name, ctrl in controllers.items():
            r = eval_controller(ctrl, params[sim_host])
            lo, hi = bootstrap_ci(r["rms_per_run"])
            results["matrix"][sim_host][name] = {**r, "ci95": [lo, hi]}
            print(f"  sim={sim_host:8s} ctrl={name:18s} "
                  f"RMS={r['rms_mean']:.3f}±{r['rms_std']:.3f}  CI95=[{lo:.3f}, {hi:.3f}]")

    # Pre-reg gates
    ikr = results["matrix"]["ikaros"]
    worst_baseline = max(ikr["constant_pwm"]["rms_mean"], ikr["pid_default"]["rms_mean"])
    gate1 = (worst_baseline - ikr["learned_ikaros"]["rms_mean"]) / max(worst_baseline, 1e-6) >= 0.20
    gate2 = (ikr["learned_daedalus"]["rms_mean"] - ikr["learned_ikaros"]["rms_mean"]) / max(ikr["learned_daedalus"]["rms_mean"], 1e-6) >= 0.05
    results["gates"] = {
        "learned_ikaros_beats_worst_baseline_20pct": bool(gate1),
        "learned_ikaros_beats_transplant_5pct": bool(gate2),
        "worst_baseline_rms": float(worst_baseline),
        "learned_ikaros_rms": float(ikr["learned_ikaros"]["rms_mean"]),
        "learned_daedalus_on_ikaros_rms": float(ikr["learned_daedalus"]["rms_mean"]),
    }
    print(f"\n  Gates: {results['gates']}")

    out_path = OUT_DIR / "fan_control.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
