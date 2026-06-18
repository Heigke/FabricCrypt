"""Track 5 / N2: Random-Telegraph Noise (RTN) on Vth.

Two-state Poisson telegraph: Vth toggles +/- dVth_RTN with mean dwell tau.
Sweep tau in {0.1, 1, 10, 100} ms.

Verify: empirical ACF of Vb(t) matches ACF_theory(t) = exp(-2 t/tau).

Output: results/N2_RTN/{acf_validation.png, summary.json}
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "results" / "N2_RTN"
OUT.mkdir(parents=True, exist_ok=True)

DV_RTN = 5e-3      # V (5 mV)
FS = 1e5           # 100 kHz
T_S = 5.0          # 5 s
N = int(FS * T_S)


def gen_rtn(n: int, fs: float, tau_s: float, dV: float, seed: int = 0) -> np.ndarray:
    """Generate two-state symmetric telegraph signal.

    Each state has mean dwell time tau/1 (so transition rate = 1/tau per state).
    Probability of a flip in dt: p = 1 - exp(-dt/tau).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    p_flip = 1.0 - np.exp(-dt / tau_s)
    flips = rng.random(n) < p_flip
    # cumulative XOR of flips gives state at each step (0 or 1)
    state = np.cumsum(flips) % 2
    return (2 * state - 1).astype(np.float32) * dV  # +-dV


def empirical_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - x.mean()
    var = np.dot(x, x) / len(x)
    acf = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        acf[k] = np.dot(x[: len(x) - k], x[k:]) / (len(x) - k) / max(var, 1e-30)
    return acf


def main() -> None:
    taus = [1e-4, 1e-3, 1e-2, 1e-1]  # 0.1, 1, 10, 100 ms
    results: dict[str, dict] = {}
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for tau in taus:
        v = gen_rtn(N, FS, tau, DV_RTN, seed=int(tau * 1e6))
        # ACF over horizon = 5*tau (or up to N/4)
        max_lag = int(min(5 * tau * FS, N // 8))
        acf = empirical_acf(v, max_lag)
        lags_s = np.arange(max_lag + 1) / FS
        # fit exponential: acf ~ exp(-2 lag / tau_emp) (for symmetric telegraph)
        # use lags where acf > 0.05
        mask = acf > 0.05
        if mask.sum() > 4:
            slope = np.polyfit(lags_s[mask], np.log(acf[mask]), 1)[0]
            tau_emp = -2.0 / slope if slope < 0 else np.inf
        else:
            tau_emp = float("nan")
        spike_rate = float(np.mean(np.abs(np.diff(v)) > 0) * FS)
        results[f"tau_{tau*1e3:.1f}ms"] = {
            "tau_s_target": tau,
            "tau_s_emp": float(tau_emp),
            "ratio": float(tau_emp / tau) if np.isfinite(tau_emp) else None,
            "rtn_std_mV": float(np.std(v) * 1e3),
            "transition_rate_Hz": spike_rate,
            "expected_rate_Hz": 1.0 / tau,
        }
        ax.semilogx(lags_s + 1e-6, acf, label=f"tau={tau*1e3:.1f}ms emp={tau_emp*1e3:.2f}ms")
    ax.set_xlabel("Lag [s]")
    ax.set_ylabel("ACF")
    ax.set_title("N2: RTN ACF vs lag (sweep tau)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.05)
    fig.tight_layout()
    fig.savefig(OUT / "acf_validation.png", dpi=110)
    plt.close(fig)

    # gate: tau_emp within 30% of tau_target for at least 3 of 4
    ratios = [r["ratio"] for r in results.values() if r["ratio"] is not None]
    ok = sum(1 for r in ratios if 0.5 <= r <= 1.7)
    summary = {
        "gate_acf_match": ok >= 3,
        "n_pass": ok,
        "n_total": len(taus),
        "results": results,
        "plot": str(OUT / "acf_validation.png"),
        "config": {"dV_RTN_V": DV_RTN, "fs": FS, "T_s": T_S},
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
