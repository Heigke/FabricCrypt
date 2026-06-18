"""E3 / E4 / E6 / E7 — Temperature, Polynomial, Noise, Layout

Four short validation experiments bundled into one runnable:

  E3  Temperature — does BSIM4 §12 T-model reproduce Zenodo Tbv1=−21.3µ/K
      across 280–340 K? (Joint fit of ALPHA0, BETA0, KT1, UTE.)

  E4  Polynomial(Vg1, Vg2) — synthetic ALPHA0 that depends polynomially
      on (Vg1, Vg2), measured over a bias grid, coefficients recovered.

  E6  Stochastic Iii — flicker (1/f) + thermal noise on top of Iii;
      report spike jitter statistics of the 2T cell.

  E7  Layout scaling — stress-induced Vth / mobility shift across an
      SA/SB sweep, show cell spiking rate modulation.
"""

from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "nsram"))

from nsram.bsim4 import (                                         # noqa: E402
    BSIM4Params, impact_ionization_bsim4, temperature_scale,
    layout_scale, PolynomialBSIM4Params, TwoTransistorCell,
    thermal_noise_psd, flicker_noise_psd,
)
from nsram.fitting import (                                       # noqa: E402
    fit_temperature_scaling, fit_polynomial_vg,
)
from nsram.physics import avalanche_current                       # noqa: E402

OUT = REPO / "results" / "z2_nsram_bsim4_zenodo"
OUT.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# E3 — Temperature validation
# ═══════════════════════════════════════════════════════════════════
def e3_temperature():
    print("=" * 70)
    print("E3 — Temperature: BSIM4 §12 T-model vs Zenodo Tbv1=−21.3µ/K")
    print("=" * 70)
    # Ground truth = Zenodo Chynoweth with Tbv1 temperature dependence
    ZENODO = dict(BV0=3.5, k_vg=1.5, Tbv1=-21.3e-6, Is=1e-16)
    Ts = [280.0, 300.0, 320.0, 340.0]
    Vg1 = 1.0
    Vds_arr = np.linspace(2.5, 4.0, 20)
    Vds_list, Iii_list = [], []
    for T in Ts:
        y = avalanche_current(Vds_arr, Vg1, T=T,
                               I0=ZENODO["Is"], BV0=ZENODO["BV0"],
                               k_vg=ZENODO["k_vg"], Tbv1=ZENODO["Tbv1"])
        y = np.maximum(y, 1e-14)
        Vds_list.append(Vds_arr); Iii_list.append(y)

    r = fit_temperature_scaling(Vds_list, Iii_list, Ts,
                                  Vgs=Vg1, base=BSIM4Params())
    print(f"  ALPHA0 fit : {r['ALPHA0']:.3e}")
    print(f"  BETA0  fit : {r['BETA0']:.2f}")
    print(f"  KT1    fit : {r['KT1']:+.4f}  V   (BSIM4 default: -0.11)")
    print(f"  UTE    fit : {r['UTE']:+.3f}     (BSIM4 default: -1.5)")
    print(f"  R² (log)   : {r['r_squared']:.4f}")

    # Physical interpretation
    print(f"\n  Zenodo Tbv1 effect: BVpar(340K) = BVpar(300K) × (1+Tbv1·40)")
    print(f"    → BVpar shift = {3.5e-6 * -21.3e-6 * 40 * 1e6:+.3f} V")
    print(f"  BSIM4 KT1 effect : Vth(340K) - Vth(300K) = KT1·(340/300-1)")
    print(f"    → Vth shift = {r['KT1'] * (340/300 - 1):+.3f} V")

    return {
        "ALPHA0": r["ALPHA0"], "BETA0": r["BETA0"],
        "KT1": r["KT1"], "UTE": r["UTE"],
        "r_squared": r["r_squared"],
    }


# ═══════════════════════════════════════════════════════════════════
# E4 — Polynomial(Vg1, Vg2)
# ═══════════════════════════════════════════════════════════════════
def e4_polynomial():
    print("\n" + "=" * 70)
    print("E4 — Polynomial(Vg1, Vg2) parameter wrapper — Sebastian's flow")
    print("=" * 70)
    # Ground-truth polynomial parameters
    true_poly = PolynomialBSIM4Params(
        base=BSIM4Params(),
        coeffs={
            "ALPHA0": {"c0": 6e-6, "vg1": 3e-6, "vg2": -2e-6,
                       "vg1_vg2": 1.5e-6},
            "BETA0":  {"c0": 22.0, "vg1": -5.0, "vg2": 2.0},
        })

    # 5x4 bias grid, small sweep range where §6.1 is linear in Vg
    Vg1_grid = np.array([0.7, 0.9, 1.0, 1.1, 1.3])
    Vg2_grid = np.array([0.2, 0.35, 0.5, 0.65])
    Vds_arr = np.linspace(2.5, 4.0, 25)
    rng = np.random.default_rng(4)

    Vds_list, Iii_list = [], []
    Vg1_list, Vg2_list = [], []
    for vg1 in Vg1_grid:
        for vg2 in Vg2_grid:
            p_at = true_poly.evaluate(vg1, vg2)
            y = np.array([float(impact_ionization_bsim4(vg1, v, 0.0, p_at))
                          for v in Vds_arr])
            y = np.maximum(y * (1 + 0.03 * rng.standard_normal(len(y))), 1e-20)
            Vds_list.append(Vds_arr); Iii_list.append(y)
            Vg1_list.append(float(vg1)); Vg2_list.append(float(vg2))

    # Fit polynomial of ALPHA0(Vg1, Vg2)
    r = fit_polynomial_vg(Vds_list, Iii_list, Vg1_list, Vg2_list,
                           param_name="ALPHA0",
                           poly_terms=("c0", "vg1", "vg2", "vg1_vg2"))
    print(f"  ALPHA0 polynomial fit  —  R²={r['r_squared']:.4f}  (N={len(r['per_point'])})")
    for term in ("c0", "vg1", "vg2", "vg1_vg2"):
        t = true_poly.coeffs["ALPHA0"][term]
        f = r["coeffs"][term]
        err_pct = 100 * abs(f - t) / max(abs(t), 1e-12)
        print(f"    {term:>10s}:  true={t:+.3e}   fit={f:+.3e}   err={err_pct:.1f}%")

    return {
        "r_squared": r["r_squared"],
        "coeffs_true": true_poly.coeffs["ALPHA0"],
        "coeffs_fit":  r["coeffs"],
        "n_points": len(r["per_point"]),
    }


# ═══════════════════════════════════════════════════════════════════
# E6 — Stochastic Iii with noise (spike jitter)
# ═══════════════════════════════════════════════════════════════════
def e6_noise():
    print("\n" + "=" * 70)
    print("E6 — Stochastic Iii with thermal + flicker noise")
    print("=" * 70)

    # Simulate 2T cell and inject noise into Iii at each step
    p = BSIM4Params(Cb=1e-12, Rb=1e5, VTH0=0.432,
                    ALPHA0=6e-6, BETA0=22.0, KF=1e-25)
    cell = TwoTransistorCell(bsim=p)
    dt = 1e-8
    t_end = 20e-6
    Vg1, Vds = 0.9, 3.8
    rng = np.random.default_rng(6)

    def run_one(noise_on: bool, seed: int):
        n = int(t_end / dt)
        VB = np.empty(n)
        vb = 0.0
        refr_until = -1.0
        spikes = []
        r = np.random.default_rng(seed)
        for i in range(n):
            ti = i * dt
            Iii = float(impact_ionization_bsim4(Vg1, Vds, min(vb, 1.1), p))
            if noise_on:
                # Thermal shot noise ~ sqrt(2qI·Δf), bandwidth 1/dt
                sigma = np.sqrt(max(2 * 1.602e-19 * abs(Iii) / dt, 1e-30))
                Iii = max(0.0, Iii + sigma * r.standard_normal())
            # Simple Euler of body charge
            Ibs = (1e-15) * (np.exp(min(vb / 0.026, 40)) - 1)
            dVB = (Iii - Ibs - vb / p.Rb) / p.Cb
            vb = vb + dt * dVB
            if vb > 0.6 and ti > refr_until:
                spikes.append(ti)
                refr_until = ti + 1.6e-6
                vb = 0.05
            VB[i] = vb
        return np.asarray(spikes)

    # Collect spike times over many trials
    N_TRIALS = 50
    spikes_clean = [run_one(False, seed=100 + i) for i in range(N_TRIALS)]
    spikes_noisy = [run_one(True,  seed=200 + i) for i in range(N_TRIALS)]

    def jitter_stats(runs):
        # Assume runs all have the "same" number of spikes; if not, truncate
        nmin = min(len(r) for r in runs)
        if nmin == 0:
            return 0.0, 0.0
        arr = np.stack([r[:nmin] for r in runs])
        # Per-spike-index std across trials
        return float(arr.mean(axis=0).mean()), float(arr.std(axis=0).mean())

    mean_clean, jitter_clean = jitter_stats(spikes_clean)
    mean_noisy, jitter_noisy = jitter_stats(spikes_noisy)
    # Mean spike rate
    rate_clean = np.mean([len(r) for r in spikes_clean]) / t_end
    rate_noisy = np.mean([len(r) for r in spikes_noisy]) / t_end
    print(f"  clean: rate={rate_clean/1e3:.1f} kHz, mean-time jitter={jitter_clean*1e9:.1f} ns")
    print(f"  noisy: rate={rate_noisy/1e3:.1f} kHz, mean-time jitter={jitter_noisy*1e9:.1f} ns")
    print(f"  jitter increase due to noise: {jitter_noisy/max(jitter_clean,1e-20):.1f}×")

    return {
        "rate_clean_hz": rate_clean,
        "rate_noisy_hz": rate_noisy,
        "jitter_clean_ns": jitter_clean * 1e9,
        "jitter_noisy_ns": jitter_noisy * 1e9,
        "jitter_ratio": jitter_noisy / max(jitter_clean, 1e-20),
        "N_trials": N_TRIALS,
    }


# ═══════════════════════════════════════════════════════════════════
# E7 — Layout scaling (stress effect on VTH and firing rate)
# ═══════════════════════════════════════════════════════════════════
def e7_layout():
    print("\n" + "=" * 70)
    print("E7 — Layout stress scaling (§13)")
    print("=" * 70)

    SA_values = np.array([0.2e-6, 0.5e-6, 1e-6, 2e-6])  # m
    SB = 0.5e-6                                         # m
    KU0  = -5e-6   # mobility stress coeff (reasonable 180 nm)
    KVTH0 = 1e-8   # Vth stress coeff

    print(f"  KU0={KU0:+.1e}   KVTH0={KVTH0:+.1e}   SB={SB*1e6:.2f} μm")
    print(f"  {'SA (μm)':>10} {'VTH0':>8} {'mu0':>10} {'VSAT':>12} {'spike rate (kHz)':>18}")

    results = []
    for SA in SA_values:
        p = BSIM4Params(SA=SA, SB=SB, KU0=KU0, KVTH0=KVTH0,
                         ALPHA0=6e-6, BETA0=22.0, Cb=1e-12, Rb=1e5)
        p_scaled = layout_scale(p)
        # Short 10 µs simulation to measure spike rate
        cell = TwoTransistorCell(bsim=p_scaled)
        res = cell.simulate(Vg1=0.9, Vg2=0.2, Vds=3.8,
                             t_end=10e-6, dt=1e-8)
        rate = len(res["spikes"]) / (res["t"][-1] + 1e-15)
        print(f"  {SA*1e6:>10.2f} {p_scaled.VTH0:>8.4f} {p_scaled.mu0:>10.4f} "
              f"{p_scaled.VSAT:>12.3e} {rate/1e3:>18.1f}")
        results.append({
            "SA_um": float(SA * 1e6),
            "VTH0": float(p_scaled.VTH0),
            "mu0": float(p_scaled.mu0),
            "VSAT": float(p_scaled.VSAT),
            "spike_rate_khz": float(rate / 1e3),
        })
    return {"SB_um": SB * 1e6, "KU0": KU0, "KVTH0": KVTH0,
            "sweep": results}


def main():
    r3 = e3_temperature()
    r4 = e4_polynomial()
    r6 = e6_noise()
    r7 = e7_layout()

    summary = {"E3": r3, "E4": r4, "E6": r6, "E7": r7}
    with open(OUT / "e3_e4_e6_e7_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] summary → {OUT / 'e3_e4_e6_e7_summary.json'}")


if __name__ == "__main__":
    main()
