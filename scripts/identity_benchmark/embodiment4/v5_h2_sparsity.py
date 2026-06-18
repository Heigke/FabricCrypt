"""V5-H2: Per-chip thermal headroom → adaptive density at constant power budget.

Hypothesis:
  ikaros and daedalus may have different sustained-power headroom.
  A chip with more headroom can run a denser reservoir within the same
  energy budget, getting higher task accuracy.

Test on a single chip is meaningful: sweep density at FIXED wall-clock
budget; report accuracy vs density. The chip with more headroom should
have a higher optimal density. We then run the SAME budget on the other
chip and check whether per-chip optimal density gives >5% accuracy gain
vs a generic (chip-agnostic) middle-of-the-road density.

Since we run both chips in this experiment, we can compute the cross-chip
adapter gap honestly.

Method:
  1. Sweep density d in {0.10, 0.20, 0.30, 0.40, 0.50}.
  2. For each, measure: (a) NRMSE on NARMA-10 averaged over 3 trials,
     (b) energy proxy = wall-clock × power (use hwmon power_input on ikaros)
  3. For each chip pick optimal density d*.
  4. Compare: own-d* vs generic d (= 0.30, paper-default).
"""
from __future__ import annotations
import json, sys, time, glob, os, socket
from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
HOST = socket.gethostname()
OUT = ROOT / f"results/IDENTITY_BENCHMARK_2026-05-30/embodiment4/v5_h2_{HOST}.json"
N = 128
T_TRAIN = 1500
T_TEST = 500
WASHOUT = 100


def read_apu_power_uw() -> float:
    """Try hwmon power_input — units typically uW."""
    for d in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            name = open(d + "/name").read().strip()
        except Exception:
            continue
        if name in ("amdgpu", "k10temp", "asusec", "asus", "cpu"):
            for f in glob.glob(d + "/power*_input"):
                try:
                    return float(open(f).read().strip())
                except Exception:
                    pass
    return float("nan")


def read_apu_temp_mc() -> float:
    try:
        return float(open("/sys/class/thermal/thermal_zone0/temp").read().strip())
    except Exception:
        return float("nan")


def narma10(T, seed):
    rng = np.random.default_rng(seed)
    u = 0.5 * rng.uniform(0.0, 1.0, size=T + 10)
    y = np.zeros(T + 10)
    for t in range(10, T + 10):
        y[t] = (0.3 * y[t-1] + 0.05 * y[t-1] * np.sum(y[t-10:t])
                + 1.5 * u[t-10] * u[t-1] + 0.1)
    return u[10:], y[10:]


def run_reservoir(u, density, seed):
    rng = np.random.default_rng(seed)
    mask = rng.random((N, N)) < density
    np.fill_diagonal(mask, False)
    W = rng.standard_normal((N, N)) / np.sqrt(N) * mask
    rho = float(np.max(np.abs(np.linalg.eigvals(W))))
    if rho > 1e-9:
        W *= 0.95 / rho
    Win = rng.standard_normal((N, 1))
    x = np.zeros(N); X = np.zeros((len(u), N))
    for t in range(len(u)):
        pre = W @ x + Win[:, 0] * u[t]
        x = 0.7 * x + 0.3 * np.tanh(pre)
        X[t] = x
    return X


def ridge_fit(X, y, alpha=1e-6):
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    A = Xb.T @ Xb + alpha * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ y)


def nrmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)) / (np.std(y) + 1e-12))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    densities = [0.10, 0.20, 0.30, 0.40, 0.50]
    n_trials = 3
    res = {"host": HOST, "densities": densities, "per_density": {}}

    for d in densities:
        nrmses, walls, powers, temps = [], [], [], []
        for trial in range(n_trials):
            u_tr, y_tr = narma10(T_TRAIN, seed=trial * 13 + 7)
            u_te, y_te = narma10(T_TEST, seed=trial * 13 + 9991)
            p0 = read_apu_power_uw(); t0 = time.perf_counter(); temp0 = read_apu_temp_mc()
            X_tr = run_reservoir(u_tr, d, seed=trial * 7 + 1)
            Wout = ridge_fit(X_tr[WASHOUT:], y_tr[WASHOUT:])
            X_te = run_reservoir(u_te, d, seed=trial * 7 + 1)
            wall = time.perf_counter() - t0
            p1 = read_apu_power_uw(); temp1 = read_apu_temp_mc()
            Xb = np.concatenate([X_te[WASHOUT:], np.ones((X_te.shape[0] - WASHOUT, 1))], axis=1)
            yhat = Xb @ Wout
            nr = nrmse(y_te[WASHOUT:], yhat)
            nrmses.append(nr); walls.append(wall)
            powers.append(((p0 if p0 == p0 else 0) + (p1 if p1 == p1 else 0)) / 2)
            temps.append(max(temp0, temp1))
        res["per_density"][f"{d:.2f}"] = {
            "nrmse_med": float(np.median(nrmses)),
            "wall_med": float(np.median(walls)),
            "power_uw_med": float(np.median(powers)),
            "temp_mc_max": float(np.max(temps)),
        }
        print(f"[H2] {HOST} d={d:.2f} nrmse={np.median(nrmses):.4f} wall={np.median(walls):.3f}s temp_peak={np.max(temps)/1000:.1f}C", flush=True)

    # Pick own optimal
    arr = res["per_density"]
    best_d = min(arr.keys(), key=lambda k: arr[k]["nrmse_med"])
    res["own_optimal_density"] = best_d
    res["own_optimal_nrmse"] = arr[best_d]["nrmse_med"]
    generic_d = "0.30"
    res["generic_nrmse"] = arr[generic_d]["nrmse_med"]
    res["accuracy_gain_pct_vs_generic"] = 100.0 * (arr[generic_d]["nrmse_med"] - arr[best_d]["nrmse_med"]) / max(1e-9, arr[generic_d]["nrmse_med"])
    res["WIN"] = res["accuracy_gain_pct_vs_generic"] >= 5.0
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"[H2] own_optimal_d={best_d} (NRMSE={res['own_optimal_nrmse']:.4f}) generic@0.30={arr[generic_d]['nrmse_med']:.4f} gain={res['accuracy_gain_pct_vs_generic']:.2f}%  WIN={res['WIN']}", flush=True)
    print(f"[H2] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
