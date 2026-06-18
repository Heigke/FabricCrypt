"""C3 — Thermal-aware survival scheduling.

Fit a tiny linear thermal model per host from observed substrate trace
(c1_<host>_data.npy). Run an RL-style policy (tabular Q with discretised
state) in a SIMULATED rollout that uses the fitted thermal model — we do
NOT drive the actual chip into trip.

The policy learns to choose tasks (HEAVY/MEDIUM/LIGHT/PAUSE) given current
temp + power, minimising thermal trips while maximising tasks completed.

Compare:
  - ikaros policy on ikaros thermal model (own body)
  - daedalus policy on ikaros thermal model (transplant)
  - generic policy: model-blind (uniform random or always-light)

Pre-reg WIN gate: ikaros policy completes ≥20% more tasks at zero
thermal-trips on ikaros simulator than daedalus policy.
"""
from __future__ import annotations
import sys, json, socket
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

HOST = socket.gethostname()
ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
C1_OUT = ROOT / f"results/IDENTITY_BENCHMARK_2026-05-30/embodiment5/c1_{HOST}"
OUT = ROOT / f"results/IDENTITY_BENCHMARK_2026-05-30/embodiment5/c3_{HOST}"
OUT.mkdir(parents=True, exist_ok=True)

# --- Fit a thermal model: T[t+1] = a*T[t] + b*P[t] + c  ------------------
def fit_thermal_model(data: np.ndarray) -> dict:
    """data: (T, 5) channels = [apu_t, gpu_t, power, freq, lat]"""
    T = data[:, 0]
    P = data[:, 2]
    # Regress T[1:] = a T[:-1] + b P[:-1] + c
    X = np.stack([T[:-1], P[:-1], np.ones(len(T) - 1)], axis=1)
    y = T[1:]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = coef.tolist()
    residual = float(np.std(y - X @ coef))
    print(f"  thermal model: T'={a:.4f}T + {b:.4f}P + {c:.4f}  resid_std={residual:.3f}")
    return {"a": a, "b": b, "c": c, "resid": residual,
            "T_med": float(np.median(T)), "T_max": float(T.max()),
            "P_med": float(np.median(P)), "P_max": float(P.max())}


# --- Simulator ----------------------------------------------------------
ACTIONS = {0: "HEAVY", 1: "MEDIUM", 2: "LIGHT", 3: "PAUSE"}
ACTION_POWER = {0: 110.0, 1: 70.0, 2: 35.0, 3: 12.0}
ACTION_REWARD = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0}
TRIP_T = 85.0
SAFE_T = 75.0


class ThermalSim:
    def __init__(self, model: dict, seed: int = 0):
        self.a = model["a"]; self.b = model["b"]; self.c = model["c"]
        self.noise = model["resid"]
        self.T = model["T_med"]
        self.rng = np.random.default_rng(seed)
        self.trips = 0
        self.completed = 0

    def step(self, action: int) -> tuple[float, float, bool]:
        """Returns (reward, next_T, terminated)."""
        P = ACTION_POWER[action]
        # noise but clamp
        eps = float(self.rng.normal(0.0, self.noise * 0.5))
        self.T = self.a * self.T + self.b * P + self.c + eps
        self.T = max(30.0, min(120.0, self.T))
        if self.T >= TRIP_T:
            # thermal trip: huge penalty, reset to 50C (sim recovery), action ineffective
            self.trips += 1
            self.T = 50.0
            return -20.0, self.T, False
        # reward only if action wasn't PAUSE
        r = ACTION_REWARD[action]
        if action != 3:
            self.completed += 1
        return r, self.T, False


# --- Tabular Q-learning -------------------------------------------------
def discretise(T: float) -> int:
    # 0:<55, 1:55-65, 2:65-72, 3:72-78, 4:78-85, 5:85+
    if T < 55: return 0
    if T < 65: return 1
    if T < 72: return 2
    if T < 78: return 3
    if T < 85: return 4
    return 5


N_STATES = 6
N_ACTIONS = 4


def train_q_policy(model: dict, episodes=400, steps_per_ep=200,
                    lr=0.2, gamma=0.95, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float32)
    for ep in range(episodes):
        sim = ThermalSim(model, seed=seed * 1000 + ep)
        eps = max(0.05, 0.6 * (1 - ep / episodes))
        s = discretise(sim.T)
        for _ in range(steps_per_ep):
            if rng.random() < eps:
                a = int(rng.integers(N_ACTIONS))
            else:
                a = int(np.argmax(Q[s]))
            r, T_new, _ = sim.step(a)
            sn = discretise(T_new)
            Q[s, a] += lr * (r + gamma * Q[sn].max() - Q[s, a])
            s = sn
    return Q


def eval_policy(Q: np.ndarray, model: dict, n_episodes=50,
                 steps_per_ep=200, seed=999):
    rewards = []; trips = []; completed = []
    for ep in range(n_episodes):
        sim = ThermalSim(model, seed=seed * 1000 + ep)
        s = discretise(sim.T)
        R = 0.0
        for _ in range(steps_per_ep):
            a = int(np.argmax(Q[s]))
            r, T_new, _ = sim.step(a)
            R += r
            s = discretise(T_new)
        rewards.append(R); trips.append(sim.trips); completed.append(sim.completed)
    return {"reward_mean": float(np.mean(rewards)),
            "trips_mean": float(np.mean(trips)),
            "trips_total": int(np.sum(trips)),
            "completed_mean": float(np.mean(completed))}


def generic_random_policy_eval(model, n_episodes=50, steps_per_ep=200, seed=999):
    rewards = []; trips = []; completed = []
    for ep in range(n_episodes):
        sim = ThermalSim(model, seed=seed * 1000 + ep)
        rng = np.random.default_rng(seed + ep * 3)
        R = 0.0
        for _ in range(steps_per_ep):
            a = int(rng.integers(N_ACTIONS))
            r, T_new, _ = sim.step(a)
            R += r
        rewards.append(R); trips.append(sim.trips); completed.append(sim.completed)
    return {"reward_mean": float(np.mean(rewards)),
            "trips_mean": float(np.mean(trips)),
            "trips_total": int(np.sum(trips)),
            "completed_mean": float(np.mean(completed))}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    data_npy = C1_OUT / f"c1_{HOST}_data.npy"
    if not data_npy.exists():
        print(f"ERROR: need {data_npy}"); sys.exit(2)
    data = np.load(data_npy)
    print(f"[C3] data shape {data.shape}")

    if cmd in ("train", "all"):
        model = fit_thermal_model(data)
        # Save model and trained policies for cross-eval
        policies = []
        per_seed = []
        for seed in range(5):
            Q = train_q_policy(model, seed=seed)
            ev = eval_policy(Q, model, seed=42 + seed)
            print(f"  seed={seed} self: reward={ev['reward_mean']:.1f} "
                  f"trips={ev['trips_mean']:.2f} completed={ev['completed_mean']:.1f}")
            policies.append(Q); per_seed.append({"seed": seed, **ev})
        gen = generic_random_policy_eval(model)
        print(f"  generic random: reward={gen['reward_mean']:.1f} "
              f"trips={gen['trips_mean']:.2f} completed={gen['completed_mean']:.1f}")

        pkg = OUT / f"c3_{HOST}_policy.npz"
        np.savez(pkg, Q=np.stack(policies),
                 model_a=model["a"], model_b=model["b"], model_c=model["c"],
                 model_resid=model["resid"], model_Tmed=model["T_med"])
        summary = {"host": HOST, "thermal_model": model,
                    "per_seed_self": per_seed,
                    "generic_random": gen,
                    "self_completed_med": float(np.median([p["completed_mean"] for p in per_seed])),
                    "self_trips_med": float(np.median([p["trips_mean"] for p in per_seed]))}
        (OUT / f"c3_{HOST}_self_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[C3] self completed med={summary['self_completed_med']:.1f} "
              f"trips_med={summary['self_trips_med']:.2f}")

    if cmd == "eval":
        # Evaluate other-host's policy on THIS host's thermal model
        if len(sys.argv) < 3:
            print("usage: eval <other_policy.npz>"); sys.exit(2)
        other = np.load(sys.argv[2])
        my_model = fit_thermal_model(data)
        Qs = other["Q"]
        rows = []
        for seed, Q in enumerate(Qs):
            ev = eval_policy(Q, my_model, seed=42 + seed)
            rows.append({"seed": int(seed), **ev})
            print(f"  transplant seed={seed} reward={ev['reward_mean']:.1f} "
                  f"trips={ev['trips_mean']:.2f} completed={ev['completed_mean']:.1f}")
        out = {"host_evaluated_on": HOST,
                "other_policy": sys.argv[2],
                "rows": rows,
                "transplant_completed_med": float(np.median([r["completed_mean"] for r in rows])),
                "transplant_trips_med": float(np.median([r["trips_mean"] for r in rows]))}
        (OUT / f"c3_{HOST}_transplant.json").write_text(json.dumps(out, indent=2))
        print(f"[C3] transplant completed med={out['transplant_completed_med']:.1f}")


if __name__ == "__main__":
    main()
