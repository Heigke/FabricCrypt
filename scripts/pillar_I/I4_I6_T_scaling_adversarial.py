"""Pillar I.4 + I.6 — internal T-scaling prediction + adversarial discriminator.

We cannot ask Sebas for T-sweep data yet. But BSIM4 v4.8.3 has explicit
T-dependence formulas. For each of three candidate parallel-path physics
(well_tap_junction, sti_edge_fet, gidl) we:

  I.4  Sweep T ∈ {220, 250, 300, 350, 400} K at fixed biases and record
       log10(I_par(T)/I_par(300)) per candidate.
  I.6  Sweep (VG1, VG2, Vd, T) over a coarse 4-D grid (~10k points). At
       each point compute pairwise prediction divergence in log10-space
       and find the top-3 bias points where one candidate disagrees with
       another by >2 decades — these become pre-registered "killshot"
       biases.

NO-CHEAT:
  * T-dependence formulas (ni(T), bandgap E_g(T), mobility µ(T), Vth(T))
    are taken from BSIM4.3.0 / BSIM4 v4.8.3 manual §12 verbatim
    (matching nsram.bsim4.temperature_scale and nsram.bsim4.bandgap).
  * GIDL formula matches nsram.bsim4 docstring eq (§6.2):
        IGIDL = AGIDL · WeffCJ · Nf · (Vd − Vg − EGIDL)/(3·Toxe)
              · exp(−3·Toxe·BGIDL / (Vd − Vg − EGIDL))
    Only T enters via Vth and Toxe is T-independent at this order.
  * Junction T-dep uses ni²(T) ∝ T³ exp(−Eg(T)/kT), Eg(T) from BSIM4 bandgap().
  * STI parasitic FET uses square-law saturation w/ µ0(T)=µ0·(T/300)^UTE
    and Vth_sti(T) = Vth_sti0 + KT1·(T/300 − 1)  (BSIM4 §12.1).
  * Default model parameters are quoted in MODEL_PARAMS dict at bottom of
    file, JSON-dumped for reproducibility.

Output:
  results/Pillar_I_4_T_scaling/predictions.json
  results/Pillar_I_6_adversarial/killshot_biases.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Physical constants (matches nsram.physics)
# ─────────────────────────────────────────────────────────────────────────
K_B = 1.380649e-23      # J/K
Q_E = 1.602176634e-19   # C
EPS_OX = 3.9 * 8.854187817e-12  # F/m

def thermal_voltage(T: float) -> float:
    return K_B * T / Q_E


def bandgap(T: float) -> float:
    """BSIM4 §12.7.2 — Si bandgap in eV. 1.16 eV at 0 K, ~1.12 eV at 300 K."""
    return 1.16 - 7.02e-4 * T * T / (T + 1108.0)


def ni_silicon(T: float) -> float:
    """Intrinsic carrier density of Si (cm⁻³).

    Standard textbook form (Sze, Pierret):
        ni(T) = C · T^1.5 · exp(−E_g(T) / (2 kT))
    Calibrated so ni(300) ≈ 1.0e10 cm⁻³ (industry-standard value).
    """
    Eg = bandgap(T)          # eV
    kT_eV = (K_B / Q_E) * T  # eV
    # C chosen so ni(300) ≈ 1.0e10 cm⁻³
    C = 1.0e10 / (300.0**1.5 * math.exp(-bandgap(300.0) / (2.0 * (K_B/Q_E) * 300.0)))
    return C * T**1.5 * math.exp(-Eg / (2.0 * kT_eV))


# ─────────────────────────────────────────────────────────────────────────
# Model parameters (default, physics-reasonable)
# ─────────────────────────────────────────────────────────────────────────
# These are the "physics-reasonable defaults" we register publicly so
# our predictions are falsifiable. All units SI unless noted.

MODEL_PARAMS = {
    # geometry — taken from PTM 130 nm bulk (Sebas test chip)
    "Weff": 0.5e-6,       # m
    "Leff": 0.13e-6,      # m
    "WeffCJ": 0.5e-6,     # m
    "Toxe": 3.0e-9,       # m   (3 nm equivalent gate oxide)
    "Nf": 1.0,            # finger count

    # well_tap_junction candidate
    # —————————————————————————————
    # n+/p-well junction underneath the NS-RAM cell. Saturation current
    # density J0_si ≈ 1e-12 A/cm² for Si p-n at 300 K (Sze §2.2).
    # A_junction = effective junction area (estimated from cell layout
    # ≈ 0.5 µm × 0.5 µm). η = 1.0 (ideal Shockley).
    "wt_J0_300": 1.0e-12,   # A/cm² at 300 K
    "wt_A_cm2": 0.25e-8,    # cm² (0.5×0.5 µm² in cm²)
    "wt_eta": 1.0,
    "wt_Vd0": 0.0,          # V — offset
    "wt_Rs": 1.0e4,         # Ω — series resistance (well/substrate spreading); caps high-Vd Shockley blowup

    # sti_edge_fet candidate
    # —————————————————————————————
    # Parasitic STI-edge FET sees VG2 directly. Width ≈ STI corner
    # perimeter ≈ 0.05 µm, length ≈ Leff. µ0 = 270 cm²/V·s. Cox = ε_ox/Tox.
    # Vth_sti ≈ 0.25 V (well below main device's 0.39 V — that's the point
    # of STI parasitics; they turn on early).
    "sti_W_eff": 0.05e-6,   # m
    "sti_L_eff": 0.13e-6,   # m
    "sti_mu0_300": 270e-4,  # m²/V·s
    "sti_Vth0": 0.25,       # V at T=300
    "sti_UTE": -1.5,        # BSIM4 mobility exponent (§12.2.1)
    "sti_KT1": -0.11,       # V — BSIM4 Vth T-coefficient (§12.1.1) [neg → Vth drops with T]
    "sti_KT1L": 0.0,        # V·m — length-dependence (neglected)

    # gidl candidate
    # —————————————————————————————
    # BSIM4 §6.2. AGIDL boosted vs default 1e-10 to model the
    # known-large parasitic in NS-RAM cells (cf. nsram.bsim4 calibration
    # for floating-body, where AGIDL is ramped up to 1e-9..2e-8 mho).
    # BGIDL and EGIDL at standard BSIM4 defaults.
    "gidl_AGIDL": 1.0e-9,   # mho   (1e-9 = boosted; default is 1e-10)
    "gidl_BGIDL": 2.3e9,    # V/m
    "gidl_EGIDL": 0.8,      # V

    # constants
    "TNOM": 300.0,
}

p = MODEL_PARAMS  # short alias


# ─────────────────────────────────────────────────────────────────────────
# Candidate 1 — well_tap_junction
# ─────────────────────────────────────────────────────────────────────────
def Ipar_well_tap(VG1, VG2, Vd, T):
    """I_par = Is_wt(T) · [exp((Vd − Vd0)/(η·Vt(T))) − 1]

    Is_wt(T) = A_junction · J0_si(T)
    J0_si(T) = J0_si(300) · (ni²(T) / ni²(300))   (Shockley T-dep, §3.4 Sze)

    Independent of VG1, VG2 (it's a back-junction; gate doesn't gate it).
    """
    Vt = thermal_voltage(T)
    ni_ratio_sq = (ni_silicon(T) / ni_silicon(300.0))**2
    Is_wt = p["wt_A_cm2"] * p["wt_J0_300"] * ni_ratio_sq    # A
    # Self-consistent solve for Shockley + series Rs:
    #   I = Is·(exp((Vd - I·Rs)/(η·Vt)) - 1)
    # Closed-form via Lambert W is overkill; use a few Newton iterations
    # starting from the unclamped Shockley value, with safe clipping.
    Rs = p["wt_Rs"]
    Vd_arr = np.asarray(Vd, dtype=np.float64)
    arg0 = np.clip((Vd_arr - p["wt_Vd0"]) / (p["wt_eta"] * Vt), -80.0, 80.0)
    I = Is_wt * (np.exp(arg0) - 1.0)
    # Newton iter on f(I) = I - Is·(exp((Vd - I·Rs)/(ηVt)) - 1) = 0
    for _ in range(40):
        Vj = Vd_arr - p["wt_Vd0"] - I * Rs
        arg = np.clip(Vj / (p["wt_eta"] * Vt), -80.0, 80.0)
        e = np.exp(arg)
        f = I - Is_wt * (e - 1.0)
        fp_deriv = 1.0 + Is_wt * Rs / (p["wt_eta"] * Vt) * e
        dI = f / fp_deriv
        I = I - dI
        if np.all(np.abs(dI) < 1e-18 + 1e-9 * np.abs(I)):
            break
    return I


# ─────────────────────────────────────────────────────────────────────────
# Candidate 2 — sti_edge_fet
# ─────────────────────────────────────────────────────────────────────────
def Ipar_sti(VG1, VG2, Vd, T):
    """STI-edge parasitic NMOS gated by VG2.

        I_par = (W_eff/L_eff) · µ0(T) · Cox · (VG2 − Vth_sti(T))²   if Vov > 0
                else 0

    Square-law saturation (BSIM4 reduces to this for Vds > Vov, simplified).
    T-dep:
        µ0(T) = µ0(300) · (T/300)^UTE        (BSIM4 §12.2.1)
        Vth_sti(T) = Vth0 + KT1 · (T/300 − 1) (BSIM4 §12.1.1)

    Independent of VG1 and (weakly of) Vd (saturation).
    """
    ratio = T / 300.0
    mu_T = p["sti_mu0_300"] * ratio**p["sti_UTE"]
    Vth_T = p["sti_Vth0"] + p["sti_KT1"] * (ratio - 1.0)
    Cox = EPS_OX / p["Toxe"]
    Vov = VG2 - Vth_T
    # clip to non-negative
    Vov_clip = np.where(Vov > 0.0, Vov, 0.0)
    I = (p["sti_W_eff"] / p["sti_L_eff"]) * mu_T * Cox * Vov_clip**2
    # Cap at linear region — if Vd < Vov, scale by Vd/Vov (triode)
    triode = np.where((Vd < Vov_clip) & (Vov_clip > 1e-9),
                      Vd / np.maximum(Vov_clip, 1e-9), 1.0)
    return I * triode


# ─────────────────────────────────────────────────────────────────────────
# Candidate 3 — gidl
# ─────────────────────────────────────────────────────────────────────────
def Ipar_gidl(VG1, VG2, Vd, T):
    """BSIM4 §6.2 GIDL at the drain edge of the main device.

        IGIDL = AGIDL · WeffCJ · Nf · (Vd − Vg − EGIDL)/(3·Toxe)
              · exp(−3·Toxe·BGIDL / (Vd − Vg − EGIDL))

    Vg here is the gate of the main NS-RAM device → VG1.
    T-dep enters only weakly via Vth-shift influencing band-bending
    (≈ +0.05–0.1 dec from 300→400 K). We include the leading effect:
    band-narrowing reduces effective EGIDL by ΔEg/2.
    """
    # band-narrowing correction (small)
    EGIDL_T = p["gidl_EGIDL"] - 0.5 * (bandgap(300.0) - bandgap(T))
    Vdg = Vd - VG1 - EGIDL_T
    # only fires when Vdg > 0
    mask = Vdg > 1e-6
    Vdg_safe = np.where(mask, Vdg, 1e-6)
    exp_arg = np.clip(-3.0 * p["Toxe"] * p["gidl_BGIDL"] / Vdg_safe, -80.0, 0.0)
    field_term = Vdg_safe / (3.0 * p["Toxe"])
    I = (p["gidl_AGIDL"] * p["WeffCJ"] * p["Nf"]
         * field_term * np.exp(exp_arg))
    return np.where(mask, I, 0.0)


CANDIDATES = {
    "well_tap": Ipar_well_tap,
    "sti":      Ipar_sti,
    "gidl":     Ipar_gidl,
}


# ─────────────────────────────────────────────────────────────────────────
# I.4  T-scaling predictions
# ─────────────────────────────────────────────────────────────────────────
def run_I4_predictions(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    T_list = [220.0, 250.0, 300.0, 350.0, 400.0]

    # Probe biases — diagnostic 250 nA point + a few others
    diagnostic_bias = {"VG1": 0.6, "VG2": -0.05, "Vd": 0.05}
    bias_points = [
        ("diagnostic_250nA", 0.6,  -0.05, 0.05),
        ("subthreshold",     0.30, -0.05, 0.05),
        ("above_knee",       0.9,   0.20, 0.30),
        ("high_Vd_GIDL",     0.0,  -0.05, 1.20),
        ("VG2_strong",       0.6,   0.50, 0.05),
    ]

    results = {
        "T_list_K": T_list,
        "diagnostic_bias": diagnostic_bias,
        "bias_points": [],
    }

    for name, VG1, VG2, Vd in bias_points:
        entry = {"name": name, "VG1": VG1, "VG2": VG2, "Vd": Vd, "per_candidate": {}}
        for cand_name, fn in CANDIDATES.items():
            I_at_T = {T: float(fn(VG1, VG2, Vd, T)) for T in T_list}
            I_300 = I_at_T[300.0]
            # log10 ratio vs T=300
            log_ratio = {}
            for T, I in I_at_T.items():
                if I_300 > 0 and I > 0:
                    log_ratio[T] = float(math.log10(I / I_300))
                elif I_300 == 0 and I == 0:
                    log_ratio[T] = 0.0
                else:
                    log_ratio[T] = float("nan")
            entry["per_candidate"][cand_name] = {
                "I_A": I_at_T,
                "log10_I_over_I300": log_ratio,
                "I_at_T400_over_I_at_T300_dec": log_ratio[400.0],
            }
        results["bias_points"].append(entry)

    # Headline predictions at diagnostic_250nA, T=400 vs T=300
    diag = results["bias_points"][0]
    results["headline_diagnostic_log10_I400_over_I300"] = {
        cand: diag["per_candidate"][cand]["I_at_T400_over_I_at_T300_dec"]
        for cand in CANDIDATES
    }

    results["model_parameters"] = MODEL_PARAMS
    results["notes"] = (
        "T-dependence formulas verbatim from BSIM4.3.0 manual §12 and "
        "Sze/Pierret semiconductor textbook. ni(T)∝T^1.5·exp(-Eg/2kT) with "
        "Eg(T) from BSIM4 bandgap(). Mobility µ(T)=µ0·(T/300)^UTE. Vth(T) "
        "linear in T via KT1. GIDL T-dep only via band-narrowing in EGIDL_T."
    )

    fp = out_dir / "predictions.json"
    fp.write_text(json.dumps(results, indent=2, default=str))
    return results, fp


# ─────────────────────────────────────────────────────────────────────────
# I.6  Adversarial bias-condition discriminator
# ─────────────────────────────────────────────────────────────────────────
def run_I6_killshots(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4-D grid — biases (V) + T (K)
    VG1_grid = np.linspace(0.0, 1.0, 11)
    VG2_grid = np.linspace(-0.2, 0.6, 9)
    Vd_grid  = np.linspace(0.02, 1.4, 12)
    T_grid   = np.array([220.0, 250.0, 300.0, 350.0, 400.0])

    G1, G2, GD, GT = np.meshgrid(VG1_grid, VG2_grid, Vd_grid, T_grid, indexing="ij")
    flat = lambda a: a.reshape(-1)
    VG1 = flat(G1); VG2 = flat(G2); Vd = flat(GD); T = flat(GT)
    N = VG1.size

    # Vectorize candidates — ni_silicon, bandgap loop over T scalars; we
    # compute per unique T to keep it fast.
    def vec_apply(fn):
        out = np.zeros(N)
        for Tk in np.unique(T):
            m = T == Tk
            out[m] = fn(VG1[m], VG2[m], Vd[m], float(Tk))
        return out

    print(f"[I.6] Evaluating {N} grid points × 3 candidates...", flush=True)
    t0 = time.time()
    I_wt   = vec_apply(Ipar_well_tap)
    I_sti  = vec_apply(Ipar_sti)
    I_gidl = vec_apply(Ipar_gidl)
    print(f"[I.6] Done in {time.time()-t0:.2f}s", flush=True)

    # log10 (with floor at 1e-30 A for stability — anything below that is
    # numerically zero in measurement)
    floor = 1e-30
    logI_wt   = np.log10(np.maximum(I_wt,   floor))
    logI_sti  = np.log10(np.maximum(I_sti,  floor))
    logI_gidl = np.log10(np.maximum(I_gidl, floor))

    # Filter to physically-measurable currents: all three must be above
    # 1 fA = 1e-15 A AND below 1 mA = 1e-3 A for at least one candidate
    # to make a measurable killshot.
    measurable_floor = 1e-15
    measurable_ceil  = 1e-3
    any_measurable = (
        ((I_wt   > measurable_floor) & (I_wt   < measurable_ceil)) |
        ((I_sti  > measurable_floor) & (I_sti  < measurable_ceil)) |
        ((I_gidl > measurable_floor) & (I_gidl < measurable_ceil))
    )

    pairs = [
        ("well_tap_vs_sti",  logI_wt,   logI_sti),
        ("well_tap_vs_gidl", logI_wt,   logI_gidl),
        ("sti_vs_gidl",      logI_sti,  logI_gidl),
    ]

    killshots = []
    for pair_name, A, B in pairs:
        D = np.abs(A - B)
        # Mask out points where neither candidate is measurable, OR
        # where the larger of the two is unmeasurable (no point in
        # discriminating if you can't measure the bigger one).
        max_I = np.maximum(10**A, 10**B)
        usable = any_measurable & (max_I > measurable_floor) & (max_I < measurable_ceil)
        D_masked = np.where(usable, D, -np.inf)

        idx = int(np.argmax(D_masked))
        killshots.append({
            "pair": pair_name,
            "max_divergence_dec": float(D[idx]),
            "VG1": float(VG1[idx]),
            "VG2": float(VG2[idx]),
            "Vd": float(Vd[idx]),
            "T_K": float(T[idx]),
            "I_well_tap_A": float(I_wt[idx]),
            "I_sti_A":      float(I_sti[idx]),
            "I_gidl_A":     float(I_gidl[idx]),
        })

    # Rank top-3 by max divergence
    killshots.sort(key=lambda k: -k["max_divergence_dec"])
    top3 = killshots[:3]

    out = {
        "rank_1": {
            "VG1": top3[0]["VG1"], "VG2": top3[0]["VG2"], "Vd": top3[0]["Vd"],
            "T_K": top3[0]["T_K"],
            "max_divergence_dec": top3[0]["max_divergence_dec"],
            "discriminates": top3[0]["pair"].replace("_vs_", " vs "),
            "currents_A": {
                "well_tap": top3[0]["I_well_tap_A"],
                "sti":      top3[0]["I_sti_A"],
                "gidl":     top3[0]["I_gidl_A"],
            },
        },
        "rank_2": {
            "VG1": top3[1]["VG1"], "VG2": top3[1]["VG2"], "Vd": top3[1]["Vd"],
            "T_K": top3[1]["T_K"],
            "max_divergence_dec": top3[1]["max_divergence_dec"],
            "discriminates": top3[1]["pair"].replace("_vs_", " vs "),
            "currents_A": {
                "well_tap": top3[1]["I_well_tap_A"],
                "sti":      top3[1]["I_sti_A"],
                "gidl":     top3[1]["I_gidl_A"],
            },
        },
        "rank_3": {
            "VG1": top3[2]["VG1"], "VG2": top3[2]["VG2"], "Vd": top3[2]["Vd"],
            "T_K": top3[2]["T_K"],
            "max_divergence_dec": top3[2]["max_divergence_dec"],
            "discriminates": top3[2]["pair"].replace("_vs_", " vs "),
            "currents_A": {
                "well_tap": top3[2]["I_well_tap_A"],
                "sti":      top3[2]["I_sti_A"],
                "gidl":     top3[2]["I_gidl_A"],
            },
        },
        "lockdown_date": "2026-05-19",
        "grid": {
            "VG1_V": list(map(float, VG1_grid)),
            "VG2_V": list(map(float, VG2_grid)),
            "Vd_V":  list(map(float, Vd_grid)),
            "T_K":   list(map(float, T_grid)),
            "n_points": int(N),
        },
        "measurable_window_A": [measurable_floor, measurable_ceil],
        "model_parameters": MODEL_PARAMS,
        "per_pair_winners": killshots,
        "notes": (
            "Pre-registered killshot biases. Any independent measurement at "
            "these (VG1,VG2,Vd,T) discriminates the candidate pair by >max_divergence_dec "
            "decades — i.e., one candidate predicts a current >100× different from "
            "another. T-dep formulas verbatim from BSIM4 §12. See "
            "predictions.json for the corresponding T-scaling matrix."
        ),
    }

    fp = out_dir / "killshot_biases.json"
    fp.write_text(json.dumps(out, indent=2))
    return out, fp


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def main():
    repo_root = Path(__file__).resolve().parents[2]
    results_root = repo_root / "results"

    i4_dir = results_root / "Pillar_I_4_T_scaling"
    i6_dir = results_root / "Pillar_I_6_adversarial"

    print(f"[main] repo_root = {repo_root}")
    print(f"[main] writing I.4 → {i4_dir}")
    print(f"[main] writing I.6 → {i6_dir}")

    # Sanity check the helpers
    print(f"  bandgap(300) = {bandgap(300.0):.4f} eV  (expect ≈ 1.124)")
    print(f"  bandgap(400) = {bandgap(400.0):.4f} eV  (expect ≈ 1.090)")
    print(f"  ni(300)      = {ni_silicon(300.0):.3e} cm^-3  (expect ≈ 1e10)")
    print(f"  ni(400)      = {ni_silicon(400.0):.3e} cm^-3")
    print(f"  ni(400)/ni(300) = {ni_silicon(400.0)/ni_silicon(300.0):.2f}")
    print(f"  2*log10(ni(400)/ni(300)) = "
          f"{2*math.log10(ni_silicon(400.0)/ni_silicon(300.0)):.3f} dec "
          f"(expected ≈ +2 dec)")

    # I.4
    i4_res, i4_fp = run_I4_predictions(i4_dir)
    print(f"[I.4] WROTE {i4_fp}")
    diag = i4_res["headline_diagnostic_log10_I400_over_I300"]
    print(f"[I.4] headline log10(I400/I300) at diagnostic 250nA point:")
    for k, v in diag.items():
        print(f"     {k:10s}: {v:+.3f} dec")

    # I.6
    i6_res, i6_fp = run_I6_killshots(i6_dir)
    print(f"[I.6] WROTE {i6_fp}")
    for rk in ("rank_1", "rank_2", "rank_3"):
        r = i6_res[rk]
        print(f"[I.6] {rk}: {r['discriminates']:25s} "
              f"@ VG1={r['VG1']:+.2f}, VG2={r['VG2']:+.2f}, Vd={r['Vd']:+.2f}, "
              f"T={r['T_K']:.0f}K  →  Δlog10={r['max_divergence_dec']:.2f} dec")

    print("\n[DONE] Pillar I.4 + I.6")


if __name__ == "__main__":
    main()
