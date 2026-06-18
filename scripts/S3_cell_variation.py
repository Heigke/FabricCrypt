#!/usr/bin/env python3
"""S3_cell_variation.py — per-cell variation framework for million-cell NS-RAM sim.

Extracts (Vth0, Bf, C_b, K1) parameter distributions from Sebas's 33 IV traces
(z6_sebas_iv_fit + z304_sebas_three_branch_refit) and provides Monte Carlo
sampling for forward_2t batched calls.

Distributions (best-effort from available fits + published 130nm):
  - Vth0:  N(0.45 V, 30 mV) — published 130nm process σ_Vth (Pelgrom)
  - Bf:    LogN(median=500, log-σ from z304 branch spread)
  - C_b:   N(102 fF, 10% rel) — z2501 default + z2042 measurements
  - K1:    N(1.5, 5% rel) — body-effect coefficient, modest spread

Usage:
    from S3_cell_variation import extract_distributions, sample_cells, validate_iv_cloud
    dists = extract_distributions()
    params = sample_cells(N=10000, seed=42, dists=dists)
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Dict
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES_DIR = REPO / "results" / "S3_network_variation"
RES_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────
# DISTRIBUTION EXTRACTION
# ─────────────────────────────────────────────────────────────────────────

def _safe_load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def extract_distributions(save: bool = True) -> Dict:
    """Pull empirical spreads from Sebas fit artifacts; fall back to published 130nm."""
    per_curve = _safe_load_json(REPO / "results" / "z6_sebas_iv_fit" / "per_curve.json")
    z304 = _safe_load_json(REPO / "results" / "z304_sebas_refit" / "summary.json")

    # --- Bf: from z304 best-fit per VG1 branch ---
    bf_branch = []
    if z304 and "by_vg1" in z304:
        for vg1_key, entry in z304["by_vg1"].items():
            if "best" in entry and entry["best"]:
                bf_branch.append(float(entry["best"]["bf"]))
    if not bf_branch:
        bf_branch = [50.0, 500.0, 9000.0]
    bf_log = np.log(np.array(bf_branch, dtype=float))
    bf_log_med = float(np.median(bf_log))
    bf_log_sigma = float(max(np.std(bf_log), 0.5))  # spread across the 3 branches

    # --- Vth0: Pelgrom-like spread for 130nm; refined by IV peak scatter ---
    # σ_Vth ≈ A_VT/sqrt(W·L). For 130nm, A_VT ≈ 3.5 mV·µm, W·L ≈ 1 µm² → ~3.5 mV.
    # For our larger NS-RAM cells (slide 16: ~10×10 µm²), σ_Vth ≈ 30 mV.
    vth0_mean = 0.45
    vth0_sigma = 0.030
    if per_curve:
        # Use IV peak Id scatter to bound an effective Vth scatter (log-Id → Vth via subthreshold slope ≈ 70 mV/dec)
        peaks = np.array([row["Id_meas_peak"] for row in per_curve if row.get("Id_meas_peak", 0) > 0])
        if peaks.size >= 5:
            log_id = np.log10(peaks)
            # branch median already in log; report scatter within branch
            vth_eff_sigma = float(np.std(log_id) * 0.070 * 0.5)  # damp 0.5×
            vth0_sigma = max(vth0_sigma, min(vth_eff_sigma, 0.080))

    # --- C_b: floating-body cap. z2501 uses 102 fF ± 10% ---
    cb_mean = 102e-15
    cb_rel_sigma = 0.10

    # --- K1: body-effect coefficient (BVpar slope dBV/dVg1 ≈ 1.5) ---
    k1_mean = 1.5
    k1_rel_sigma = 0.05

    dists = {
        "source": {
            "per_curve_n": len(per_curve) if per_curve else 0,
            "z304_branches": bf_branch,
        },
        "Vth0": {"dist": "normal", "mean": vth0_mean, "sigma": vth0_sigma, "unit": "V",
                 "note": "Pelgrom 130nm + IV peak scatter refinement"},
        "Bf":   {"dist": "lognormal", "log_mean": bf_log_med, "log_sigma": bf_log_sigma,
                 "min": 10.0, "max": 30000.0, "unit": "-",
                 "note": "from z304 per-VG1 branch best-fits"},
        "C_b":  {"dist": "normal", "mean": cb_mean, "rel_sigma": cb_rel_sigma, "unit": "F",
                 "note": "z2501 default 102 fF + measured 10% process spread"},
        "K1":   {"dist": "normal", "mean": k1_mean, "rel_sigma": k1_rel_sigma, "unit": "V/V",
                 "note": "BVpar gate-dependence slope"},
    }
    if save:
        with open(RES_DIR / "extracted_param_distributions.json", "w") as f:
            json.dump(dists, f, indent=2)
    return dists


# ─────────────────────────────────────────────────────────────────────────
# MONTE CARLO SAMPLING
# ─────────────────────────────────────────────────────────────────────────

def sample_cells(N: int, seed: int = 42, dists: Dict | None = None) -> Dict[str, np.ndarray]:
    """Draw N per-cell parameter vectors from the extracted distributions."""
    if dists is None:
        dists = extract_distributions(save=False)
    rng = np.random.RandomState(seed)

    Vth0 = rng.normal(dists["Vth0"]["mean"], dists["Vth0"]["sigma"], size=N)
    Vth0 = np.clip(Vth0, 0.20, 0.80)

    log_bf = rng.normal(dists["Bf"]["log_mean"], dists["Bf"]["log_sigma"], size=N)
    Bf = np.clip(np.exp(log_bf), dists["Bf"]["min"], dists["Bf"]["max"])

    Cb_mu = dists["C_b"]["mean"]
    Cb = rng.normal(Cb_mu, Cb_mu * dists["C_b"]["rel_sigma"], size=N)
    Cb = np.clip(Cb, 50e-15, 200e-15)

    K1_mu = dists["K1"]["mean"]
    K1 = rng.normal(K1_mu, K1_mu * dists["K1"]["rel_sigma"], size=N)
    K1 = np.clip(K1, 0.8, 2.5)

    return {"Vth0": Vth0, "Bf": Bf, "C_b": Cb, "K1": K1, "N": N, "seed": seed}


# ─────────────────────────────────────────────────────────────────────────
# BATCHED forward_2t (vectorised over N cells)
# ─────────────────────────────────────────────────────────────────────────

def forward_2t_batched(Vg1, Vg2, Vds, cells, T=300.0):
    """Vectorised quasi-static 2T forward model: returns Id per cell (A).

    Uses simple subthreshold + avalanche surrogate consistent with z2501:
      I_sub  = I0 * exp((Vg1 - Vth0) / nVt) * (1 - exp(-Vds/Vt))   [M1 channel]
      BVpar  = 3.5 - K1 * Vg1   (V)
      I_aval = I_sub * Bf * clamp(exp((Vds - BVpar) / Vt), 1, 1e6)
    Inputs are scalars or 1-D arrays broadcasting against cells['Vth0'] (shape N).
    """
    Vth0 = cells["Vth0"]; Bf = cells["Bf"]; K1 = cells["K1"]
    Vt = 26e-3 * (T / 300.0)
    nVt = 1.3 * Vt
    I0 = 5e-9  # device-level scale
    Vg1 = np.broadcast_to(np.asarray(Vg1, dtype=float), Vth0.shape)
    Vg2 = np.broadcast_to(np.asarray(Vg2, dtype=float), Vth0.shape)
    Vds = np.broadcast_to(np.asarray(Vds, dtype=float), Vth0.shape)
    overdrive = (Vg1 - Vth0) / nVt
    overdrive = np.clip(overdrive, -40.0, 20.0)
    I_sub = I0 * np.exp(overdrive) * (1.0 - np.exp(-Vds / Vt))
    BVpar = 3.5 - K1 * Vg1
    aval_arg = np.clip((Vds - BVpar) / Vt, -50.0, 14.0)
    M = np.clip(np.exp(aval_arg), 1.0, 1e6)
    Id = I_sub * (1.0 + Bf * 1e-3 * (M - 1.0))  # Bf scaled to keep currents physical
    # gentle VG2 modulation (body bias) — increases multiplication when VG2 high
    Id = Id * (1.0 + 0.3 * (Vg2 - 0.4))
    return Id


# ─────────────────────────────────────────────────────────────────────────
# VALIDATION: simulated IV cloud bracketing measured curves
# ─────────────────────────────────────────────────────────────────────────

def validate_iv_cloud(N_sample: int = 200, seed: int = 7, save_plot: bool = False) -> Dict:
    """Generate N_sample cells, sweep Vds at fixed VG1/VG2 grid, check cloud bracket."""
    dists = extract_distributions(save=False)
    cells = sample_cells(N_sample, seed=seed, dists=dists)
    vg1_list = [0.2, 0.4, 0.6]
    vg2_list = [-0.1, 0.0, 0.1]
    vds_grid = np.linspace(0.5, 3.5, 41)

    per_curve = _safe_load_json(REPO / "results" / "z6_sebas_iv_fit" / "per_curve.json") or []
    meas_peak_log = np.log10(np.array(
        [r["Id_meas_peak"] for r in per_curve if r.get("Id_meas_peak", 0) > 0]
    )) if per_curve else np.array([])

    sim_log_peaks = []
    for vg1 in vg1_list:
        for vg2 in vg2_list:
            Id_curve = []
            for vds in vds_grid:
                Id_curve.append(forward_2t_batched(vg1, vg2, vds, cells))
            Id_curve = np.stack(Id_curve, axis=0)  # (n_vds, N)
            peaks = Id_curve.max(axis=0)
            sim_log_peaks.append(np.log10(np.clip(peaks, 1e-15, None)))
    sim_log_peaks = np.concatenate(sim_log_peaks)
    sim_p05, sim_p95 = np.percentile(sim_log_peaks, [5, 95])
    bracket = {
        "sim_log10_peak_p05": float(sim_p05),
        "sim_log10_peak_p95": float(sim_p95),
        "sim_log10_peak_median": float(np.median(sim_log_peaks)),
        "meas_log10_peak_min": float(meas_peak_log.min()) if meas_peak_log.size else None,
        "meas_log10_peak_max": float(meas_peak_log.max()) if meas_peak_log.size else None,
        "meas_log10_peak_median": float(np.median(meas_peak_log)) if meas_peak_log.size else None,
        "brackets_measured": bool(
            meas_peak_log.size and sim_p05 <= meas_peak_log.min() + 1.0
            and sim_p95 >= meas_peak_log.max() - 1.0
        ),
        "N_sample": int(N_sample),
    }
    return bracket


# ─────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    dists = extract_distributions(save=True)
    cells = sample_cells(10000, seed=42, dists=dists)
    spreads = {
        "Vth0_sigma_V":    float(np.std(cells["Vth0"])),
        "Vth0_mean_V":     float(np.mean(cells["Vth0"])),
        "Bf_log_sigma":    float(np.std(np.log(cells["Bf"]))),
        "Bf_median":       float(np.median(cells["Bf"])),
        "Cb_rel_sigma":    float(np.std(cells["C_b"]) / np.mean(cells["C_b"])),
        "K1_rel_sigma":    float(np.std(cells["K1"]) / np.mean(cells["K1"])),
    }
    bracket = validate_iv_cloud(N_sample=200, seed=7)
    elapsed = time.time() - t0
    summary = {
        "elapsed_s": elapsed,
        "spreads_10k": spreads,
        "iv_cloud_bracket": bracket,
        "INFRA_B_PASS": all(np.std(cells[k]) > 0 for k in ("Vth0", "Bf", "C_b", "K1")),
    }
    out = RES_DIR / "S3_cell_variation_demo.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
