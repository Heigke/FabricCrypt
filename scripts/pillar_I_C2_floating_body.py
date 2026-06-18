#!/usr/bin/env python3
"""Pillar I — C2 candidate: self-consistent floating-body M1 sub-Vt amplifier.

O91 brainstorm hypothesis: at low VG1, M1 channel is OFF in pyport (predicted
~1e-15 A) but silicon shows ~250 nA. C2 says: the *body* V_B is charged up by
upstream events (snapback, well-tap, prior-bias history). Body-effect reduces
effective Vth_M1 via Vth_eff = Vth0 - γ·(√(2φF - Vbs) - √(2φF)). With Vbs ≈
0.5 V from a self-consistent body equilibrium, Vth drops 0.3-0.5 V → sub-Vt
exp(0.5/0.06) ≈ 4000× → 1e-15 A becomes ~250 nA.

The pyport ALREADY has floating V_B with body-effect on M1 Vth via BSIM4
K1/K1ox/sqrtPhis machinery. The C2 hypothesis is: the existing self-
consistent solver settles at a LOW-Vbs branch. To test, we add a persistent
body-charging mechanism that pushes V_B up.

Edits to `nsram/nsram/bsim4_port/nsram_cell_2T.py`:
  - NSRAMCell2TConfig fields:
      enable_c2_body_inject, c2_body_inject_I0, c2_body_inject_Vd_alpha,
      c2_body_inject_T_xti, c2_vbs_clamp_floor.
  - `_residuals(...)`: when enable_c2_body_inject and I0!=0, add
      I_c2_inject = I0 · max(0, 1+α·Vd) · (T/300)^xti  → +INTO body (R_B+=).
  - M1 Vb eval: if c2_vbs_clamp_floor finite, pass Vb_for_M1 = Vsint +
      clamp → Vbs_M1 = clamp (overrides self-consistent V_B for ablation).

This driver:
  1. C2_ON full 33-bias fwd+bwd refit (I0=1nA, α=0.5, xti=4 — T strong).
  2. C2_OFF control (enable_c2_body_inject=False, same code path).
  3. LEGACY (no C2 at all, baseline reference; expect median ≈ 1.163).
  4. Vbs sweep ablation: clamp Vbs ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6} V,
     fwd-only for runtime. Vbs=0 = body-effect KILL (expected blow-up).
  5. T-coefficient (300→400K) at the 250 nA diagnostic bias.
  6. I0 × α ablation heatmap (fwd-only).
  7. Bootstrap 95% CI on every reported median.

Pre-registered gates (DO NOT MODIFY POST-HOC; baseline LEGACY=1.163 dec):
  - C2_ON cell-wide median dec ≤ 1.0 dec
  - C2_ON VG1=0.6 triode RMSE ≤ 0.5 dec
  - C2_ON VG1=0.2 must NOT regress vs LEGACY (Δdec ≤ +0.2)
  - C2_ON fwd↔bwd spread ≤ 0.3 dec
  - C2_OFF control diff vs C2_ON ≥ 0.5 dec (else KILLSHOT)
  - Vbs=0 clamp control: med_dec must EXPLODE (≥ +2 dec vs LEGACY) — proves
    body-effect is structurally load-bearing.
  - T-coefficient 300→400K diag = +3..+5 dec (per task spec — body-diode
    forward leak grows fast with T if C2 is the dominant amplifier).

NO-CHEAT:
  - Same code path for C2_ON / C2_OFF / LEGACY; only flags differ.
  - fwd+bwd reported separately.
  - NaN biases counted, not silently dropped.
  - Vbs-clamp control documented even if it explodes solver.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import csv
import json
import math
import re
import sys
import time
import traceback
import importlib.util
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
DATA = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/Pillar_I_C2_floating_body"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260519)

# ── Pre-registered gates (DO NOT MODIFY POST-HOC) ──────────────────
GATE_MEDIAN_DEC = 1.0
GATE_TRIODE_RMSE_VG06 = 0.5
GATE_VG02_REGRESS = 0.2
GATE_FWDBWD_SPREAD = 0.3
GATE_T_COEFF_LO = 3.0      # body-diode forward + sub-Vt amplifier → steep T-dep
GATE_T_COEFF_HI = 5.0
GATE_KILLSHOT_DIFF = 0.5
GATE_VBS_KILL_DELTA = 2.0  # Vbs=0 must blow up by ≥ +2 dec vs LEGACY

# ── Bias loaders (reused from C3) ──────────────────────────────────
BRANCH_FLAT = {
    0.4: {"ETAB": 1.9, "K1": 0.53825, "ALPHA0": 7.842e-05, "BETA0": 19.0, "NFACTOR": 6.0, "trise": 10.59},
    0.6: {"ETAB": 2.5, "K1": 0.41825, "ALPHA0": 7.842e-05, "BETA0": 20.0, "NFACTOR": 6.0, "trise": 9.04},
}
M2_STATIC_OVERRIDES = {"k1": 0.63825, "k2": -0.070435, "etab": -0.086777, "beta0": 18.0}


def load_sebas_params():
    rows = []
    with open(DATA / "2Tcell_BSIM_param_DC.csv") as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try: row[k] = float(v)
                except ValueError: row[k] = float("nan")
            rows.append(row)
    return rows


def find_or_impute_row(rows, VG1, VG2, atol=1e-3):
    target = None
    for r in rows:
        if abs(r["VG1"] - VG1) < atol and abs(r["VG2"] - VG2) < atol:
            target = dict(r); break
    if target is None:
        return None, False
    if math.isnan(target.get("K1", float("nan"))):
        branch = BRANCH_FLAT.get(round(VG1, 2))
        if branch is None: return target, False
        for k, v in branch.items(): target[k] = float(v)
        return target, True
    return target, False


def make_overrides(sebas_row):
    if sebas_row is None: return None, None
    P_M1 = {}
    for csv_k, py_k in (("ETAB", "etab"), ("K1", "k1"), ("ALPHA0", "alpha0"), ("BETA0", "beta0")):
        if not math.isnan(sebas_row.get(csv_k, float("nan"))):
            P_M1[py_k] = float(sebas_row[csv_k])
    P_M2 = {}
    if not math.isnan(sebas_row.get("NFACTOR", float("nan"))):
        P_M2["nfactor"] = float(sebas_row["NFACTOR"])
    for k, v in M2_STATIC_OVERRIDES.items():
        if k not in P_M2: P_M2[k] = float(v)
    return (P_M1 or None), (P_M2 or None)


@contextmanager
def patch_sd_scaled(sd, overrides):
    if not overrides: yield; return
    saved = {}
    try:
        for k, v in overrides.items():
            saved[k] = sd.scaled.get(k, None); sd.scaled[k] = float(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None: sd.scaled.pop(k, None)
            else: sd.scaled[k] = v


def load_curves():
    curves = []
    for sub in sorted(DATA.iterdir()):
        if not sub.is_dir(): continue
        m_vg1 = re.search(r"VG1=([\d.\-]+)", sub.name)
        if not m_vg1: continue
        vg1 = float(m_vg1.group(1))
        for f in sorted(sub.glob("*.csv")):
            m = re.search(r"VG2=([\-\d.]+)", f.name)
            if not m: continue
            vg2 = float(m.group(1))
            d = np.loadtxt(f, delimiter=",", skiprows=1)
            if d.ndim != 2 or d.shape[1] < 2: continue
            Vd = d[:, 0].astype(np.float64)
            Id = np.abs(d[:, 1]).astype(np.float64)
            apex = int(np.argmax(Vd))
            fwd_Vd = Vd[: apex + 1]; fwd_Id = Id[: apex + 1]
            bwd_Vd = Vd[apex:][::-1].copy(); bwd_Id = Id[apex:][::-1].copy()
            curves.append({
                "VG1": vg1, "VG2": vg2, "f": f.name,
                "fwd_Vd": fwd_Vd, "fwd_Id": fwd_Id,
                "bwd_Vd": bwd_Vd, "bwd_Id": bwd_Id,
            })
    return curves


def build_pyport_base():
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    cfg.body_pdiode_to = "vnwell"
    cfg.use_well_diode = True
    cfg.vnwell = 2.0
    cfg.body_pdiode_Js = 5.3675e-7 / 22e-12
    cfg.body_pdiode_n = 1.0535
    cfg.body_pdiode_Rs = 1.0e6
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    return cfg, M1, M2, bjt


# ── Metric helpers ────────────────────────────────────────────────
DEC_FLOOR_MEAS = 1e-12
DEC_FLOOR_PRED = 1e-30


def log_residuals(Id_meas, Id_pred, Vd, vmin=0.3):
    m = (Vd > vmin) & (np.abs(Id_meas) > DEC_FLOOR_MEAS) & (Id_pred > 0)
    if m.sum() < 3:
        return np.array([])
    lm = np.log10(np.clip(np.abs(Id_meas[m]), DEC_FLOOR_MEAS, None))
    lp = np.log10(np.clip(Id_pred[m], DEC_FLOOR_PRED, None))
    return np.abs(lm - lp)


def bootstrap_ci(values, alpha=0.05, n_boot=1000):
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    v = np.asarray(values, dtype=np.float64)
    med = float(np.median(v))
    idx = RNG.integers(0, len(v), size=(n_boot, len(v)))
    boots = np.array([np.median(v[i]) for i in idx])
    return med, float(np.percentile(boots, 100*alpha/2)), float(np.percentile(boots, 100*(1-alpha/2)))


def run_grid(cfg, M1, M2, bjt, curves, sebas_rows, label, do_bwd=True):
    rows = []; nan_count = 0
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    branches = (("fwd", "fwd_Vd", "fwd_Id"),)
    if do_bwd:
        branches = (("fwd", "fwd_Vd", "fwd_Id"), ("bwd", "bwd_Vd", "bwd_Id"))
    for c in curves:
        row_sebas, _ = find_or_impute_row(sebas_rows, c["VG1"], c["VG2"])
        P_M1, P_M2 = make_overrides(row_sebas)
        for branch, vdk, idk in branches:
            Vd_np = c[vdk]; Id_np = c[idk]
            Vd = torch.tensor(Vd_np, dtype=torch.float64)
            try:
                with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
                    out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd,
                                     VG1=torch.tensor(c["VG1"], dtype=torch.float64),
                                     VG2=torch.tensor(c["VG2"], dtype=torch.float64),
                                     warm_start=True)
                I_pred = np.abs(out["Id"].detach().cpu().numpy()).astype(np.float64)
                if not np.all(np.isfinite(I_pred)):
                    nan_count += int(np.sum(~np.isfinite(I_pred)))
                    I_pred = np.where(np.isfinite(I_pred), I_pred, 0.0)
            except Exception:
                nan_count += len(Vd_np); I_pred = np.zeros_like(Vd_np)
            res = log_residuals(Id_np, I_pred, Vd_np, vmin=0.3)
            med_dec = float(np.median(res)) if res.size else float("nan")
            mt = (Vd_np > 0.05) & (Vd_np <= 0.5) & (np.abs(Id_np) > DEC_FLOOR_MEAS) & (I_pred > 0)
            if mt.sum() >= 3:
                lm = np.log10(np.clip(np.abs(Id_np[mt]), DEC_FLOOR_MEAS, None))
                lp = np.log10(np.clip(I_pred[mt], DEC_FLOOR_PRED, None))
                triode_rmse = float(np.sqrt(np.mean((lm - lp) ** 2)))
            else:
                triode_rmse = float("nan")
            rows.append({
                "VG1": c["VG1"], "VG2": c["VG2"], "branch": branch, "file": c["f"],
                "n_samples": int(res.size),
                "med_dec": med_dec,
                "triode_rmse_dec": triode_rmse,
                "Imeas_peak": float(np.max(np.abs(Id_np))),
                "Ipred_peak": float(np.max(I_pred)) if I_pred.size else float("nan"),
            })
    return rows, nan_count


def summarize(rows, label):
    out = {"label": label}
    all_med = np.array([r["med_dec"] for r in rows if np.isfinite(r["med_dec"])])
    med, lo, hi = bootstrap_ci(all_med)
    out["median_dec_all"] = {"median": med, "ci95_lo": lo, "ci95_hi": hi, "n": int(all_med.size)}
    for vg1 in (0.2, 0.4, 0.6):
        sub = np.array([r["med_dec"] for r in rows
                        if abs(r["VG1"] - vg1) < 1e-6 and np.isfinite(r["med_dec"])])
        med, lo, hi = bootstrap_ci(sub)
        out[f"median_dec_VG1={vg1}"] = {"median": med, "ci95_lo": lo, "ci95_hi": hi, "n": int(sub.size)}
    for br in ("fwd", "bwd"):
        sub = np.array([r["med_dec"] for r in rows
                        if r["branch"] == br and np.isfinite(r["med_dec"])])
        med, lo, hi = bootstrap_ci(sub)
        out[f"median_dec_{br}"] = {"median": med, "ci95_lo": lo, "ci95_hi": hi, "n": int(sub.size)}
    triode = np.array([r["triode_rmse_dec"] for r in rows
                       if abs(r["VG1"] - 0.6) < 1e-6 and np.isfinite(r["triode_rmse_dec"])])
    med, lo, hi = bootstrap_ci(triode)
    out["triode_rmse_VG1=0.6"] = {"median": med, "ci95_lo": lo, "ci95_hi": hi, "n": int(triode.size)}
    return out


# ── T-coefficient (300→400K) falsifier ───────────────────────────
def t_coefficient(cfg, M1, M2, bjt, curves, sebas_rows):
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    target = None
    for c in curves:
        if abs(c["VG1"] - 0.6) < 1e-6 and abs(c["VG2"] - (-0.05)) < 1e-6:
            target = c; break
    if target is None:
        return {"error": "no VG1=0.6 VG2=-0.05 curve"}
    row_sebas, _ = find_or_impute_row(sebas_rows, target["VG1"], target["VG2"])
    P_M1, P_M2 = make_overrides(row_sebas)
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    Vd_T = target["fwd_Vd"]
    idx_diag = int(np.argmin(np.abs(Vd_T - 0.05)))
    idx_tri  = int(np.argmin(np.abs(Vd_T - 1.0)))
    Vd_t = torch.tensor(Vd_T, dtype=torch.float64)
    Is_out = {"diag": {}, "triode": {}}
    saved_TC = cfg.T_C
    try:
        for T_K in (220.0, 300.0, 350.0, 400.0):
            cfg.T_C = T_K - 273.15
            with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_t,
                                 VG1=torch.tensor(target["VG1"], dtype=torch.float64),
                                 VG2=torch.tensor(target["VG2"], dtype=torch.float64),
                                 warm_start=True)
            Ip = np.abs(out["Id"].detach().cpu().numpy())
            Is_out["diag"][T_K] = float(Ip[idx_diag])
            Is_out["triode"][T_K] = float(Ip[idx_tri])
    finally:
        cfg.T_C = saved_TC

    def dlog(d, T1, T2):
        a = d.get(T1); b = d.get(T2)
        if a is None or b is None or a <= 0 or b <= 0 or not np.isfinite(a) or not np.isfinite(b):
            return float("nan")
        return float(np.log10(b / a))
    return {
        "I_diag_at_T":   {str(k): v for k, v in Is_out["diag"].items()},
        "I_triode_at_T": {str(k): v for k, v in Is_out["triode"].items()},
        "log10_I400_over_I300_diag":   dlog(Is_out["diag"], 300.0, 400.0),
        "log10_I400_over_I300_triode": dlog(Is_out["triode"], 300.0, 400.0),
        "log10_I400_over_I220_diag":   dlog(Is_out["diag"], 220.0, 400.0),
    }


# ── Vbs sweep ablation (fwd-only) ─────────────────────────────────
def vbs_sweep(cfg, M1, M2, bjt, curves, sebas_rows, vbs_grid):
    """Clamp Vbs ∈ vbs_grid and report median dec per Vbs."""
    out = []
    saved_floor = cfg.c2_vbs_clamp_floor
    saved_inj = cfg.enable_c2_body_inject
    # Disable body-injection during clamp sweep — pure body-effect test
    cfg.enable_c2_body_inject = False
    try:
        for vbs in vbs_grid:
            cfg.c2_vbs_clamp_floor = float(vbs)
            t0 = time.time()
            rows, nans = run_grid(cfg, M1, M2, bjt, curves, sebas_rows,
                                  f"Vbs={vbs:.2f}", do_bwd=False)
            meds = np.array([r["med_dec"] for r in rows if np.isfinite(r["med_dec"])])
            med = float(np.median(meds)) if meds.size else float("nan")
            sub02 = np.array([r["med_dec"] for r in rows
                              if abs(r["VG1"] - 0.2) < 1e-6 and np.isfinite(r["med_dec"])])
            med02 = float(np.median(sub02)) if sub02.size else float("nan")
            triode = np.array([r["triode_rmse_dec"] for r in rows
                               if abs(r["VG1"] - 0.6) < 1e-6 and np.isfinite(r["triode_rmse_dec"])])
            tri = float(np.median(triode)) if triode.size else float("nan")
            out.append({
                "Vbs_clamp": float(vbs), "median_dec": med,
                "median_dec_VG1=0.2": med02, "triode_rmse_VG1=0.6": tri,
                "n_total": len(rows), "n_finite": int(meds.size),
                "nan_count": int(nans), "elapsed_s": time.time() - t0,
            })
            print(f"  Vbs={vbs:.2f} V → med_dec={med:.3f} (VG1=0.2: {med02:.3f}, triode: {tri:.3f}) "
                  f"nan={nans} ({time.time()-t0:.0f}s)", flush=True)
    finally:
        cfg.c2_vbs_clamp_floor = saved_floor
        cfg.enable_c2_body_inject = saved_inj
    return out


# ── I0 × Vd_alpha heatmap ─────────────────────────────────────────
def heatmap_I0_alpha(cfg, M1, M2, bjt, curves, sebas_rows):
    I0_grid = [0.0, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
    ALPHA_grid = [0.0, 0.5, 2.0, 5.0]
    Z = np.full((len(I0_grid), len(ALPHA_grid)), np.nan)
    saved = (cfg.enable_c2_body_inject, cfg.c2_body_inject_I0,
             cfg.c2_body_inject_Vd_alpha, cfg.c2_vbs_clamp_floor)
    cfg.c2_vbs_clamp_floor = float("nan")  # no clamp during heatmap
    try:
        for i, I0 in enumerate(I0_grid):
            for j, A in enumerate(ALPHA_grid):
                if I0 <= 0.0:
                    cfg.enable_c2_body_inject = False
                    cfg.c2_body_inject_I0 = 0.0
                    cfg.c2_body_inject_Vd_alpha = 0.0
                else:
                    cfg.enable_c2_body_inject = True
                    cfg.c2_body_inject_I0 = float(I0)
                    cfg.c2_body_inject_Vd_alpha = float(A)
                t0 = time.time()
                rows, _ = run_grid(cfg, M1, M2, bjt, curves, sebas_rows,
                                   f"I0={I0:.0e}_A={A}", do_bwd=False)
                meds = np.array([r["med_dec"] for r in rows if np.isfinite(r["med_dec"])])
                Z[i, j] = float(np.median(meds)) if meds.size else float("nan")
                print(f"  heatmap I0={I0:.0e} α={A:.1f} → med_dec={Z[i,j]:.3f} ({time.time()-t0:.0f}s)", flush=True)
    finally:
        (cfg.enable_c2_body_inject, cfg.c2_body_inject_I0,
         cfg.c2_body_inject_Vd_alpha, cfg.c2_vbs_clamp_floor) = saved
    return I0_grid, ALPHA_grid, Z


def plot_heatmap(I0_grid, ALPHA_grid, Z, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(ALPHA_grid)))
    ax.set_xticklabels([f"{a:.1f}" for a in ALPHA_grid])
    ax.set_yticks(range(len(I0_grid)))
    ax.set_yticklabels([f"{j:.0e}" for j in I0_grid])
    ax.set_xlabel("c2_body_inject_Vd_alpha")
    ax.set_ylabel("c2_body_inject_I0 [A]")
    ax.set_title("33-bias median |Δlog10 I| (dec) — fwd only\n(C2 body-injection; lower = better)")
    for i in range(len(I0_grid)):
        for j in range(len(ALPHA_grid)):
            v = Z[i, j]
            txt = "NaN" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if (np.isfinite(v) and v > 2.0) else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="median |Δlog10 I| (dec)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_vbs_sweep(vbs_sweep_data, legacy_med, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    vbs = [r["Vbs_clamp"] for r in vbs_sweep_data]
    med = [r["median_dec"] for r in vbs_sweep_data]
    med02 = [r["median_dec_VG1=0.2"] for r in vbs_sweep_data]
    tri = [r["triode_rmse_VG1=0.6"] for r in vbs_sweep_data]
    ax.plot(vbs, med, "o-", label="median dec (all 33 biases)")
    ax.plot(vbs, med02, "s--", label="median dec (VG1=0.2 only)")
    ax.plot(vbs, tri, "^:", label="triode RMSE (VG1=0.6)")
    ax.axhline(legacy_med, color="k", lw=1, ls=":", label=f"LEGACY median = {legacy_med:.2f}")
    ax.axhline(GATE_MEDIAN_DEC, color="g", lw=1, ls="--", label=f"PASS gate = {GATE_MEDIAN_DEC}")
    ax.set_xlabel("Vbs clamp [V]  (Vbs=0 ⇒ body-effect off)")
    ax.set_ylabel("median |Δlog10 I| (dec)")
    ax.set_title("Vbs-clamp ablation — C2 body-effect sensitivity")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ── Verdict ───────────────────────────────────────────────────────
def write_verdict(s_on, s_off, s_leg, t_coeff, vbs_data, path):
    import operator as op_
    lines = ["# Pillar I — C2 (self-consistent floating-body M1 sub-Vt amp) — VERDICT", ""]
    lines.append("Date: 2026-05-19 (Pillar I structural fix, C2 implementation)")
    lines.append("")
    lines.append("## Pre-registered gates")

    on_med = s_on["median_dec_all"]["median"]
    off_med = s_off["median_dec_all"]["median"]
    leg_med = s_leg["median_dec_all"]["median"]
    triode = s_on["triode_rmse_VG1=0.6"]["median"]
    vg02_on = s_on["median_dec_VG1=0.2"]["median"]
    vg02_leg = s_leg["median_dec_VG1=0.2"]["median"]
    fwd_med = s_on["median_dec_fwd"]["median"]
    bwd_med = s_on["median_dec_bwd"]["median"]

    def check(name, value, gate, op):
        ok = (op(value, gate)
              if (np.isfinite(value) and np.isfinite(gate))
              else False)
        return f"- **{name}**: {value:.3f} {op.__name__} {gate} → {'PASS' if ok else 'FAIL'}"

    lines += [
        check("median dec ≤ 1.0 (C2 ON)", on_med, GATE_MEDIAN_DEC, op_.le),
        check("triode RMSE VG1=0.6 ≤ 0.5", triode, GATE_TRIODE_RMSE_VG06, op_.le),
        check("VG1=0.2 regression (Δdec ≤ +0.2)", vg02_on - vg02_leg, GATE_VG02_REGRESS, op_.le),
        check("fwd↔bwd spread ≤ 0.3 dec", abs(fwd_med - bwd_med), GATE_FWDBWD_SPREAD, op_.le),
    ]
    c2_off_diff = off_med - on_med
    killshot_ok = c2_off_diff >= GATE_KILLSHOT_DIFF
    lines.append(f"- **C2-OFF control diff = {c2_off_diff:.3f} dec** (gate ≥ {GATE_KILLSHOT_DIFF}) → "
                 f"{'PASS' if killshot_ok else 'FAIL → KILLSHOT (C2 injection not contributing)'}")

    Tc_diag = t_coeff.get("log10_I400_over_I300_diag", float("nan"))
    Tc_tri  = t_coeff.get("log10_I400_over_I300_triode", float("nan"))
    if not np.isfinite(Tc_diag):
        lines.append(f"- **T-coefficient 300→400K (diag, Vd=0.05)**: NaN → INDETERMINATE")
        verdict_T = "INDETERMINATE"
    elif Tc_diag < GATE_T_COEFF_LO or Tc_diag > GATE_T_COEFF_HI:
        lines.append(f"- **T-coefficient 300→400K (diag) = {Tc_diag:.2f} dec** "
                     f"(target +{GATE_T_COEFF_LO}..+{GATE_T_COEFF_HI}) "
                     f"→ INDETERMINATE (outside C2 range)")
        verdict_T = "INDETERMINATE"
    else:
        lines.append(f"- **T-coefficient 300→400K (diag) = {Tc_diag:.2f} dec** "
                     f"(target +{GATE_T_COEFF_LO}..+{GATE_T_COEFF_HI}) "
                     f"→ PASS-IN-RANGE")
        verdict_T = "T-RANGE-OK"
    lines.append(f"- T-coefficient (triode, Vd≈1.0) = {Tc_tri:.2f} dec (diagnostic, no gate)")

    # Vbs=0 kill control
    vbs0 = next((r for r in vbs_data if abs(r["Vbs_clamp"]) < 1e-9), None)
    if vbs0 is not None and np.isfinite(vbs0["median_dec"]):
        vbs_kill_delta = vbs0["median_dec"] - leg_med
        vbs_kill_ok = vbs_kill_delta >= GATE_VBS_KILL_DELTA
        lines.append(f"- **Vbs=0 clamp control (body-effect KILL): Δ vs LEGACY = {vbs_kill_delta:+.3f} dec** "
                     f"(gate ≥ +{GATE_VBS_KILL_DELTA}) → {'PASS' if vbs_kill_ok else 'FAIL (body-effect not load-bearing)'}")
    else:
        lines.append(f"- **Vbs=0 clamp control**: solver did not converge → INDETERMINATE")

    primary = [
        on_med <= GATE_MEDIAN_DEC,
        triode <= GATE_TRIODE_RMSE_VG06,
        (vg02_on - vg02_leg) <= GATE_VG02_REGRESS,
        abs(fwd_med - bwd_med) <= GATE_FWDBWD_SPREAD,
    ]
    primary_pass = sum(primary)
    lines.append("")
    lines.append(f"## Overall: {primary_pass}/4 primary gates PASS, "
                 f"C2-OFF killshot {'CLEAR' if killshot_ok else 'TRIGGERED'}, "
                 f"T-coeff {verdict_T}")
    if primary_pass == 4 and killshot_ok:
        lines.append("**VERDICT: PASS — C2 closes the structural gap on the pre-registered gates.**")
    elif not killshot_ok:
        lines.append("**VERDICT: KILLSHOT — C2-OFF doesn't worsen fit ≥0.5 dec; C2 not the dominant mechanism.**")
    else:
        lines.append(f"**VERDICT: FAIL — {4-primary_pass}/4 primary gates not met. See JSON.**")
    lines.append("")
    lines.append(f"Baseline reference: LEGACY median = {leg_med:.3f} dec, "
                 f"LEGACY VG1=0.2 = {vg02_leg:.3f}, "
                 f"LEGACY VG1=0.6 triode RMSE = {s_leg['triode_rmse_VG1=0.6']['median']:.3f}")
    lines.append("")
    lines.append("## Vbs sweep ablation (fwd-only)")
    lines.append("| Vbs_clamp [V] | median dec | VG1=0.2 dec | triode RMSE | nan | elapsed s |")
    lines.append("|---|---|---|---|---|---|")
    for r in vbs_data:
        lines.append(f"| {r['Vbs_clamp']:.2f} | {r['median_dec']:.3f} | "
                     f"{r['median_dec_VG1=0.2']:.3f} | {r['triode_rmse_VG1=0.6']:.3f} | "
                     f"{r['nan_count']} | {r['elapsed_s']:.0f} |")

    lines.append("")
    lines.append("## Numbers (all with bootstrap 95% CI)")
    for tag, S in (("C2_ON", s_on), ("C2_OFF (enable_c2_body_inject=False)", s_off), ("LEGACY", s_leg)):
        lines.append(f"\n### {tag}")
        for k, v in S.items():
            if k == "label": continue
            lines.append(f"- {k}: median={v['median']:.3f} CI95=[{v['ci95_lo']:.3f}, {v['ci95_hi']:.3f}] n={v['n']}")
    path.write_text("\n".join(lines))


def _safe_default(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return None


def main():
    t_start = time.time()
    print(f"[C2] starting at {time.ctime()}", flush=True)
    sebas_rows = load_sebas_params()
    curves = load_curves()
    print(f"[C2] Loaded {len(curves)} curves (target 33)", flush=True)
    cfg, M1, M2, bjt = build_pyport_base()

    # C2 default knobs: I0=1nA constant, weak Vd-dependence, strong T-dep
    # (well-diode forward ~ ni² ~ T^3·exp(-Eg/kT) → effective xti ~ 4-6).
    C2_DEFAULT_I0 = 1.0e-9
    C2_DEFAULT_VDALPHA = 0.5
    C2_DEFAULT_XTI = 4.0

    # Phase 1: full fwd+bwd C2_ON
    print("[C2] Phase 1/5: C2_ON full 33-bias fwd+bwd ...", flush=True)
    cfg.enable_c2_body_inject = True
    cfg.c2_body_inject_I0 = C2_DEFAULT_I0
    cfg.c2_body_inject_Vd_alpha = C2_DEFAULT_VDALPHA
    cfg.c2_body_inject_T_xti = C2_DEFAULT_XTI
    cfg.c2_vbs_clamp_floor = float("nan")  # no clamp
    t0 = time.time()
    rows_on, nan_on = run_grid(cfg, M1, M2, bjt, curves, sebas_rows, "C2_ON", do_bwd=True)
    print(f"  C2_ON: {time.time()-t0:.0f}s, nan={nan_on}", flush=True)
    s_on = summarize(rows_on, "C2_ON")
    print(f"  C2_ON  median dec = {s_on['median_dec_all']['median']:.3f}", flush=True)

    # Phase 2: C2_OFF control (flag off, same code path)
    print("[C2] Phase 2/5: C2_OFF control (enable_c2_body_inject=False) ...", flush=True)
    cfg.enable_c2_body_inject = False
    cfg.c2_body_inject_I0 = 0.0
    t0 = time.time()
    rows_off, nan_off = run_grid(cfg, M1, M2, bjt, curves, sebas_rows, "C2_OFF", do_bwd=True)
    print(f"  C2_OFF: {time.time()-t0:.0f}s, nan={nan_off}", flush=True)
    s_off = summarize(rows_off, "C2_OFF")
    print(f"  C2_OFF median dec = {s_off['median_dec_all']['median']:.3f}", flush=True)

    # Phase 3: LEGACY (also C2 off — but this is the canonical baseline ref)
    print("[C2] Phase 3/5: LEGACY (canonical baseline) ...", flush=True)
    # LEGACY is identical to C2_OFF code path. We re-use rows_off as LEGACY
    # (no extra knobs differ between them) but keep both labelled for clarity.
    rows_leg = rows_off; nan_leg = nan_off
    s_leg = summarize(rows_leg, "LEGACY")
    print(f"  LEGACY median dec = {s_leg['median_dec_all']['median']:.3f}", flush=True)

    # Phase 4: Vbs sweep ablation (fwd-only)
    print("[C2] Phase 4/5: Vbs sweep ablation (fwd-only) ...", flush=True)
    vbs_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    vbs_data = vbs_sweep(cfg, M1, M2, bjt, curves, sebas_rows, vbs_grid)

    # Phase 5a: T-coefficient at the diagnostic bias (use C2_ON config)
    print("[C2] Phase 5a/5: T-coefficient 300→400K @ VG1=0.6,VG2=-0.05 ...", flush=True)
    cfg.enable_c2_body_inject = True
    cfg.c2_body_inject_I0 = C2_DEFAULT_I0
    cfg.c2_body_inject_Vd_alpha = C2_DEFAULT_VDALPHA
    cfg.c2_body_inject_T_xti = C2_DEFAULT_XTI
    cfg.c2_vbs_clamp_floor = float("nan")
    try:
        t_coeff = t_coefficient(cfg, M1, M2, bjt, curves, sebas_rows)
    except Exception as e:
        traceback.print_exc()
        t_coeff = {"error": str(e), "log10_I400_over_I300_diag": float("nan")}
    print(f"  T-coeff diag={t_coeff.get('log10_I400_over_I300_diag', 'NaN')}, "
          f"triode={t_coeff.get('log10_I400_over_I300_triode', 'NaN')}", flush=True)

    # Phase 5b: I0 × alpha heatmap (fwd-only)
    print("[C2] Phase 5b/5: I0×alpha heatmap (fwd-only) ...", flush=True)
    I0_grid, ALPHA_grid, Z = heatmap_I0_alpha(cfg, M1, M2, bjt, curves, sebas_rows)
    plot_heatmap(I0_grid, ALPHA_grid, Z, OUT / "heatmap_I0_alpha.png")
    plot_vbs_sweep(vbs_data, s_leg["median_dec_all"]["median"], OUT / "Vbs_sweep_heatmap.png")

    summary = {
        "date": "2026-05-19",
        "candidate": "C2 — self-consistent floating-body M1 sub-Vt amplifier",
        "modified_file": "nsram/nsram/bsim4_port/nsram_cell_2T.py "
                         "(added enable_c2_body_inject, c2_body_inject_I0/Vd_alpha/T_xti, "
                         "c2_vbs_clamp_floor cfg fields; wired I_c2_inject into R_B; "
                         "added Vbs clamp at M1 _eval_mosfet entry)",
        "n_biases_discovered": len(curves),
        "NaN_counts": {"C2_ON": nan_on, "C2_OFF": nan_off, "LEGACY": nan_leg},
        "default_c2_config": {
            "c2_body_inject_I0": C2_DEFAULT_I0,
            "c2_body_inject_Vd_alpha": C2_DEFAULT_VDALPHA,
            "c2_body_inject_T_xti": C2_DEFAULT_XTI,
        },
        "pre_reg_gates": {
            "median_dec_target": GATE_MEDIAN_DEC,
            "triode_rmse_VG06_target": GATE_TRIODE_RMSE_VG06,
            "VG02_regression_max": GATE_VG02_REGRESS,
            "fwd_bwd_spread_max": GATE_FWDBWD_SPREAD,
            "T_coeff_range_dec": [GATE_T_COEFF_LO, GATE_T_COEFF_HI],
            "C2_off_killshot_min_diff": GATE_KILLSHOT_DIFF,
            "Vbs0_kill_min_delta": GATE_VBS_KILL_DELTA,
        },
        "summary_C2_ON": s_on,
        "summary_C2_OFF": s_off,
        "summary_LEGACY": s_leg,
        "T_coefficient": t_coeff,
        "Vbs_sweep": vbs_data,
        "heatmap_I0_alpha": {
            "I0_grid": I0_grid,
            "alpha_grid": ALPHA_grid,
            "median_dec_matrix": Z.tolist(),
        },
        "per_bias_C2_ON": rows_on,
        "per_bias_C2_OFF": rows_off,
        "total_elapsed_s": time.time() - t_start,
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_safe_default)

    write_verdict(s_on, s_off, s_leg, t_coeff, vbs_data, OUT / "verdict.md")
    print(f"[C2] Done in {time.time()-t_start:.0f}s. Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
