"""z461 — Comprehensive dynamics validation harness for NS-RAM 2T cell model.

Builds a single re-runnable framework that produces BOTH publication-quality
plots AND a numerical acceptance table for every dynamic property of the
NS-RAM cell. Visually + numerically verify exactly what dynamics we have
and where the gaps are.

9 validation tests (each → 1 plot + 1 metric + pre-registered acceptance gate):

    V1 DC IV per branch        — overlay model vs measured, 3 panels (VG1=0.2/0.4/0.6)
    V2 DC fwd vs bwd hysteresis — overlay (VG1=0.6 column)
    V3 Snapback knee position   — V_d at I_d=10µA
    V4 Ns-snap rise             — V_B(t) for 5ns hold (VG1=0.6/VG2=0/V_d=2V)
    V5 Latch hold               — V_B(t) over 100ns
    V6 Self-reset               — V_B(t) extended 1µs hold then release
    V7 Relaxation oscillation   — V_B(t) for 5µs constant drive
    V8 LIF integrate            — V_B(t) for ramp 0→1V over 1µs (sub-threshold)
    V9 LIF threshold gain       — spike count vs V_drive (1.5/1.7/1.9/2.1V)

Configs:
    SB_OFF    : v449_B base, no snapback subcircuit (control)
    SB_HOT    : v449_B + SB_HOT snapback (BV·0.6, Is·5)
    NX_1p8    : v449_B + SB_HOT + Slotboom knee + NPN current-gate V_knee=1.8 (z457 best)
    z458_best : alias for NX_1p8 with R_body from z458 best Pareto cell
                (z458 found NO self-reset cell; we use snap_Is_scale=0.1, R_body=10M
                 as the most-promising cell from the Pareto sweep)

Output: results/z461_validation_<CONFIG>/
    plot_V[1-9]_*.png       (9 per-test plots)
    validation_summary.png   (3x3 grid)
    validation_table.json    (all metrics + gates + pass/fail + source path)
    report.md
    acceptance_card.md

Pre-registered gates (line 1 of run.log):
    INFRA      = all 9 tests run + plots written
    DISCOVERY  = >=6/9 PASS
    AMBITIOUS  = >=8/9 PASS
    KILL_SHOT  = >=3/9 structurally impossible
"""
from __future__ import annotations
import argparse
import importlib.util as _ilu
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

# ------------------------------------------------------------------ #
# Load shared helpers from z454 (which chains z449→z427/z429).
# ------------------------------------------------------------------ #
_spec454 = _ilu.spec_from_file_location("z454", ROOT / "scripts/z454_snapback_integration.py")
z454 = _ilu.module_from_spec(_spec454); _spec454.loader.exec_module(z454)
z449 = z454.z449
z427 = z454.z427
z429 = z454.z429

from nsram.bsim4_port import transient_real_v2 as trv2
from nsram.bsim4_port.transient_real_v2 import (
    integrate, TransientCfgV2, stim_fast_pulse,
)
from nsram.bsim4_port.nsram_cell_2T import _residuals as _residuals_cell


# ------------------------------------------------------------------ #
# Config registry
# ------------------------------------------------------------------ #
V449B_BASE = {
    "use_vbic_for_q1": True,
    "vbic_AVC1": 0.5, "vbic_AVC2": 0.5,
    "Cbody": 1e-15,
    "body_pdiode_Cj0_per_area": 0.0,
}
SNAP_DEFAULT = dict(
    snap_BV=2.0, snap_n_avl=4.0, snap_Bf=417.0, snap_Va=0.90,
    snap_Is=6.0256e-9, snap_Nf=1.0,
    snap_Id_clamp=1e-2, snap_Iii_clamp=1e-2,
)
SNAP_HOT = dict(SNAP_DEFAULT)
SNAP_HOT["snap_BV"] = 2.0 * 0.6
# z471 — snap_Is calibrated to land Id_pk on Mario 4.8 mA target at lifted clamp.
# Was: 6.0256e-9 * 5.0 = 3.0128e-8 (clamp-bound at 100 mA, +1.32 dec over Mario).
# Now: 4.5192e-12 (×0.00015) → Id_pk = 4.23 mA at VG1=0.6/VG2=0/Vd=2V (gap -0.055 dec).
# Clamps lifted from 1e-2 to 1e-1 (z470 baseline) so the calibration is not masked.
SNAP_HOT["snap_Is"] = 4.5192e-12
SNAP_HOT["snap_Id_clamp"] = 1e-1
SNAP_HOT["snap_Iii_clamp"] = 1e-1


def make_config(name: str) -> dict:
    if name == "SB_OFF":
        return {**V449B_BASE, "use_snapback_sub": False}
    if name == "SB_HOT":
        return {**V449B_BASE, "use_snapback_sub": True, **SNAP_HOT}
    if name == "NX_1p8":
        # z474 — explicit `_R_body=1e7` lock (also now the SnapbackParams /
        # TransientCfgV2 default). z473 R_body sweep verdict: V6 (self-reset)
        # flips PASS at R=1e7 with Id_pk drift only 0.007 dec (well under
        # KILL_SHOT 0.15 dec). V3/V7 unaffected by body-leak path.
        return {**V449B_BASE, "use_snapback_sub": True, **SNAP_HOT,
                "snap_use_knee_gate": True,
                "snap_V_knee": 1.6, "snap_V_sharp": 0.05,
                "snap_npn_gate_mode": "current",
                "snap_npn_V_knee": 1.8, "snap_npn_V_sharp": 0.05,
                "snap_npn_V_BE_offset": 0.3,
                "_R_body": 1e7}
    if name == "z458_best":
        # z458 found no self-reset cell. Use most-promising Pareto cell:
        # snap_Is_scale=0.1, R_body=10M (transient-only knob).
        f = make_config("NX_1p8")
        f["snap_Is"] = SNAP_HOT["snap_Is"] * 0.1
        f["_R_body"] = 1e7  # tcfg-level, not cfg
        return f
    raise ValueError(f"unknown config: {name}")


@dataclass
class TestResult:
    test_id: str
    name: str
    plot_path: str
    metric_value: float
    metric_unit: str
    gate: str
    passed: bool
    notes: str
    source_path: str
    structurally_impossible: bool = False


# ------------------------------------------------------------------ #
# Stimulus helpers
# ------------------------------------------------------------------ #
def stim_pulse_general(V_lo, V_hi, t_pre, t_rise, t_hold, t_fall, t_post,
                       n_total=2000):
    """Generic trapezoidal pulse. Returns (t, Vd) arrays."""
    T = t_pre + t_rise + t_hold + t_fall + t_post
    t = np.linspace(0.0, T, n_total)
    Vd = np.full_like(t, V_lo)
    t_r0 = t_pre
    t_r1 = t_pre + t_rise
    t_h1 = t_r1 + t_hold
    t_f1 = t_h1 + t_fall
    for i, ti in enumerate(t):
        if ti < t_r0:
            Vd[i] = V_lo
        elif ti < t_r1:
            Vd[i] = V_lo + (V_hi - V_lo) * (ti - t_r0) / t_rise
        elif ti < t_h1:
            Vd[i] = V_hi
        elif ti < t_f1:
            Vd[i] = V_hi - (V_hi - V_lo) * (ti - t_h1) / t_fall
        else:
            Vd[i] = V_lo
    return t, Vd


def stim_ramp(V_lo, V_hi, t_ramp, n_total=2000):
    """Linear ramp V_lo → V_hi over t_ramp seconds."""
    t = np.linspace(0.0, t_ramp, n_total)
    Vd = V_lo + (V_hi - V_lo) * (t / t_ramp)
    return t, Vd


# ------------------------------------------------------------------ #
# Build models, curves once
# ------------------------------------------------------------------ #
def thermal_pause(log):
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            t = int(f.read().strip()) / 1000.0
        if t > 85.0:
            log(f"  THERMAL PAUSE: APU={t:.1f}C > 85C, cooling")
            for _ in range(120):
                time.sleep(2)
                with open("/sys/class/thermal/thermal_zone0/temp") as f:
                    t = int(f.read().strip()) / 1000.0
                if t < 75.0:
                    log(f"  COOLED: APU={t:.1f}C"); break
    except Exception:
        pass


# ------------------------------------------------------------------ #
# Transient runner — single-shot integrate with given stimulus
# ------------------------------------------------------------------ #
def run_transient(cfg_flags, model_M1, model_M2, sebas_rows,
                  VG1, VG2, t_arr, Vd_arr, Vb0=0.0,
                  max_step=1e-10, first_step=1e-14):
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(cfg_flags))
    cfg.Cbody = 1e-15
    R_body = cfg_flags.get("_R_body", None)
    tcfg = TransientCfgV2(C_B_const=1e-15, max_step=max_step,
                          first_step=first_step, rtol=1e-6, atol=1e-15,
                          R_body=R_body)
    sebas_row = z427.find_params(sebas_rows, VG1, VG2)
    if sebas_row is None:
        return None
    P_M1, P_M2 = z427.make_overrides(sebas_row)
    bjt = z427.make_bjt(sebas_row)
    z449._VBIC_CTX["cfg"] = cfg
    z449._VBIC_CTX["bjt"] = bjt
    try:
        with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
             z427.patch_sd_scaled(sd_M2, P_M2):
            r = integrate(cfg, model_M1, model_M2, bjt,
                          t_arr, Vd_arr, VG1, VG2,
                          tcfg=tcfg, Vb0=Vb0)
    finally:
        z449._VBIC_CTX["cfg"] = None
        z449._VBIC_CTX["bjt"] = None
    return r


# ------------------------------------------------------------------ #
# V1 — DC IV per branch
# ------------------------------------------------------------------ #
def run_V1_dc_per_branch(cfg_flags, model_M1, model_M2, curves, sebas_rows,
                          out_dir, log):
    """Forward sweep, per-branch RMSE. Overlay 3 panels (VG1=0.2/0.4/0.6)."""
    log("V1 — DC IV per branch")
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(cfg_flags))
    log_eps = 1e-15
    panels = {0.2: [], 0.4: [], 0.6: []}
    for c in curves:
        if c["VG1"] not in panels:
            continue
        sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        bjt = z427.make_bjt(sebas_row)
        Vd_arr = c["Vd"].numpy()
        Id_meas = c["Id"].numpy()
        order = np.argsort(Vd_arr)
        Vd_seq = Vd_arr[order]
        Id_meas_seq = Id_meas[order]
        Id_pred_seq = np.zeros_like(Vd_seq)
        try:
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
                 z427.patch_sd_scaled(sd_M2, P_M2):
                Vb_warm = 0.0
                for i, Vd_f in enumerate(Vd_seq):
                    r = z429.run_vsint_pinned(
                        cfg, model_M1, model_M2, bjt,
                        float(Vd_f), float(c["VG1"]), float(c["VG2"]),
                        Vsint_pin=0.0, Vb_init=Vb_warm)
                    Id_pred_seq[i] = abs(r["Id"]) if r.get("Id") is not None else 0.0
                    if r["converged"]:
                        Vb_warm = r["Vb"]
                    else:
                        Vb_warm = 0.0
        except Exception as e:
            log(f"  V1 fail VG1={c['VG1']} VG2={c['VG2']}: {e}")
            continue
        # Per-curve RMSE (log10)
        lp = np.log10(Id_pred_seq + log_eps)
        lm = np.log10(Id_meas_seq + log_eps)
        rmse = float(np.sqrt(np.mean((lp - lm) ** 2)))
        panels[c["VG1"]].append({
            "VG2": c["VG2"], "Vd": Vd_seq.tolist(),
            "Id_meas": Id_meas_seq.tolist(),
            "Id_pred": Id_pred_seq.tolist(),
            "rmse": rmse,
        })

    # Per-branch RMSE (quadratic mean over curves)
    per_branch_rmse = {}
    for VG1, recs in panels.items():
        if not recs:
            per_branch_rmse[VG1] = float("inf")
        else:
            per_branch_rmse[VG1] = float(
                math.sqrt(sum(r["rmse"] ** 2 for r in recs) / len(recs)))

    # Plot: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, VG1 in zip(axes, [0.2, 0.4, 0.6]):
        recs = panels[VG1]
        colors = plt.cm.viridis(np.linspace(0, 1, max(1, len(recs))))
        for k, rec in enumerate(sorted(recs, key=lambda x: x["VG2"])):
            ax.semilogy(rec["Vd"], np.maximum(rec["Id_meas"], log_eps),
                        "o", ms=3, color=colors[k], alpha=0.6,
                        label=f"meas VG2={rec['VG2']:.1f}")
            ax.semilogy(rec["Vd"], np.maximum(rec["Id_pred"], log_eps),
                        "-", lw=1.0, color=colors[k])
        rm = per_branch_rmse[VG1]
        ax.set_title(f"VG1={VG1:.1f}   RMSE={rm:.2f} dec")
        ax.set_xlabel("V_D [V]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6, ncol=1)
    axes[0].set_ylabel("|I_D| [A]")
    fig.suptitle("V1 — DC IV: model (lines) vs measured (markers)")
    fig.tight_layout()
    p = out_dir / "plot_V1_dc_per_branch.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    log(f"  per_branch_rmse = {per_branch_rmse}")
    worst = max(per_branch_rmse.values())
    passed = all(v < 2.5 for v in per_branch_rmse.values())
    return TestResult(
        test_id="V1", name="DC IV per branch",
        plot_path=str(p),
        metric_value=worst, metric_unit="dec (worst per-branch RMSE)",
        gate="each branch RMSE < 2.5 dec",
        passed=passed,
        notes=f"per-branch RMSE VG1=0.2:{per_branch_rmse[0.2]:.2f}, "
              f"VG1=0.4:{per_branch_rmse[0.4]:.2f}, "
              f"VG1=0.6:{per_branch_rmse[0.6]:.2f}",
        source_path=str(p),
    ), panels


# ------------------------------------------------------------------ #
# V2 — DC fwd vs bwd hysteresis (VG1=0.6 column)
# ------------------------------------------------------------------ #
def run_V2_hysteresis(cfg_flags, model_M1, model_M2, curves, sebas_rows,
                      out_dir, log):
    log("V2 — DC fwd vs bwd hysteresis (VG1=0.6)")
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(cfg_flags))
    recs = []
    for c in curves:
        if abs(c["VG1"] - 0.6) > 1e-3:
            continue
        sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        bjt = z427.make_bjt(sebas_row)
        Vd_arr = c["Vd"].numpy()
        order = np.argsort(Vd_arr)
        Vd_seq = Vd_arr[order]
        out = {"VG2": c["VG2"], "Vd": Vd_seq.tolist()}
        for direction, ord_fn in [("fwd", lambda x: x), ("bwd", lambda x: x[::-1])]:
            Vd_dir = ord_fn(Vd_seq)
            Id_pred = np.zeros_like(Vd_dir)
            try:
                with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
                     z427.patch_sd_scaled(sd_M2, P_M2):
                    Vb_warm = 0.0
                    for i, Vd_f in enumerate(Vd_dir):
                        r = z429.run_vsint_pinned(
                            cfg, model_M1, model_M2, bjt,
                            float(Vd_f), 0.6, float(c["VG2"]),
                            Vsint_pin=0.0, Vb_init=Vb_warm)
                        Id_pred[i] = abs(r["Id"]) if r.get("Id") is not None else 0.0
                        if r["converged"]:
                            Vb_warm = r["Vb"]
                        else:
                            Vb_warm = 0.0
            except Exception as e:
                log(f"  V2 fail dir={direction} VG2={c['VG2']}: {e}")
                Id_pred = np.zeros_like(Vd_dir)
            # Re-sort to ascending Vd for area integration
            if direction == "bwd":
                Id_pred = Id_pred[::-1]
            out[f"Id_{direction}"] = Id_pred.tolist()
        recs.append(out)

    # Hysteresis area: avg over curves of |∫(Id_fwd - Id_bwd) dVd| in V·µA
    areas = []
    for r in recs:
        Vd = np.array(r["Vd"]); If = np.array(r["Id_fwd"]); Ib = np.array(r["Id_bwd"])
        a = float(np.abs(np.trapezoid(If - Ib, Vd)) * 1e6)
        areas.append(a)
    hyst_area = float(np.mean(areas)) if areas else 0.0

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, max(1, len(recs))))
    for k, rec in enumerate(sorted(recs, key=lambda x: x["VG2"])):
        ax.semilogy(rec["Vd"], np.maximum(rec["Id_fwd"], 1e-15),
                    "-",  color=colors[k], lw=1.4,
                    label=f"VG2={rec['VG2']:.1f} fwd")
        ax.semilogy(rec["Vd"], np.maximum(rec["Id_bwd"], 1e-15),
                    "--", color=colors[k], lw=1.0,
                    label=f"VG2={rec['VG2']:.1f} bwd")
    ax.set_xlabel("V_D [V]"); ax.set_ylabel("|I_D| [A]")
    ax.set_title(f"V2 — fwd vs bwd hysteresis, VG1=0.6 "
                 f"(mean area = {hyst_area:.3g} V·µA)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = out_dir / "plot_V2_hysteresis.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    log(f"  hysteresis area = {hyst_area:.4g} V·uA")
    passed = hyst_area > 0.0
    return TestResult(
        test_id="V2", name="DC fwd vs bwd hysteresis",
        plot_path=str(p),
        metric_value=hyst_area, metric_unit="V·µA",
        gate="hysteresis area > 0 (bistability present)",
        passed=passed,
        notes=f"mean across {len(areas)} VG2 columns",
        source_path=str(p),
    )


# ------------------------------------------------------------------ #
# V3 — Snapback knee position (V_d at I_d=10µA)
# ------------------------------------------------------------------ #
def run_V3_knee_position(cfg_flags, model_M1, model_M2, sebas_rows,
                         out_dir, log, panels_v1=None):
    log("V3 — Snapback knee position")
    # Use VG1=0.6, VG2=0 forward sweep; find V_d at which Id crosses 10µA.
    # Reuse z429.run_vsint_pinned.
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(cfg_flags))
    sebas_row = z427.find_params(sebas_rows, 0.6, 0.0)
    if sebas_row is None:
        return TestResult("V3", "Snapback knee position", "", float("nan"), "V",
                          "V_knee within 0.3V of 1.5V", False,
                          "no sebas row VG1=0.6 VG2=0",
                          "", structurally_impossible=True)
    P_M1, P_M2 = z427.make_overrides(sebas_row)
    bjt = z427.make_bjt(sebas_row)
    Vd_seq = np.linspace(0.05, 2.0, 80)
    Id_pred = np.zeros_like(Vd_seq)
    with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
         z427.patch_sd_scaled(sd_M2, P_M2):
        Vb_warm = 0.0
        for i, Vd_f in enumerate(Vd_seq):
            r = z429.run_vsint_pinned(cfg, model_M1, model_M2, bjt,
                                       float(Vd_f), 0.6, 0.0,
                                       Vsint_pin=0.0, Vb_init=Vb_warm)
            Id_pred[i] = abs(r["Id"]) if r.get("Id") is not None else 0.0
            if r["converged"]:
                Vb_warm = r["Vb"]
            else:
                Vb_warm = 0.0
    target = 10e-6
    above = np.where(Id_pred >= target)[0]
    V_knee = float(Vd_seq[above[0]]) if len(above) else float("nan")

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.semilogy(Vd_seq, np.maximum(Id_pred, 1e-15), "b-", lw=1.4,
                label="model VG1=0.6 VG2=0")
    ax.axhline(target, color="grey", ls=":", label="10 µA")
    if not math.isnan(V_knee):
        ax.axvline(V_knee, color="red", ls="--",
                   label=f"V_knee = {V_knee:.2f} V")
    ax.axvspan(1.2, 1.8, color="green", alpha=0.15,
               label="measured target 1.5±0.3 V")
    ax.set_xlabel("V_D [V]"); ax.set_ylabel("|I_D| [A]")
    ax.set_title("V3 — Snapback knee V_d at I_d=10µA")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "plot_V3_knee_position.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    passed = (not math.isnan(V_knee)) and abs(V_knee - 1.5) <= 0.3
    log(f"  V_knee = {V_knee} V")
    return TestResult(
        test_id="V3", name="Snapback knee position",
        plot_path=str(p),
        metric_value=V_knee, metric_unit="V",
        gate="V_knee within 0.3V of 1.5V (measured)",
        passed=passed,
        notes=f"model V_knee={V_knee:.2f}V (target 1.5±0.3V)",
        source_path=str(p),
    )


# ------------------------------------------------------------------ #
# Generic transient test runner (V4..V9)
# ------------------------------------------------------------------ #
def _transient_VB_trace(cfg_flags, model_M1, model_M2, sebas_rows,
                        VG1, VG2, t_arr, Vd_arr, Vb0=0.0,
                        max_step=1e-10, first_step=1e-14):
    r = run_transient(cfg_flags, model_M1, model_M2, sebas_rows,
                      VG1, VG2, t_arr, Vd_arr, Vb0=Vb0,
                      max_step=max_step, first_step=first_step)
    if r is None:
        return None, None, None
    Vb = np.array(r["Vb"])
    Id = np.array(r["Id"])
    return Vb, Id, r


def run_V4_ns_snap(cfg_flags, model_M1, model_M2, sebas_rows, out_dir, log):
    log("V4 — Ns-snap rise (5ns hold)")
    t_arr, Vd_arr = stim_pulse_general(V_lo=0.05, V_hi=2.0,
                                         t_pre=0.5e-9, t_rise=100e-12,
                                         t_hold=5e-9,  t_fall=100e-12,
                                         t_post=2e-9, n_total=600)
    Vb, Id, r = _transient_VB_trace(cfg_flags, model_M1, model_M2,
                                    sebas_rows, 0.6, 0.0, t_arr, Vd_arr)
    if Vb is None:
        return TestResult("V4", "Ns-snap rise", "", float("nan"), "ns",
                          "t_to_0.5V<5ns AND V_B_peak>0.5V", False,
                          "no sebas row", "", structurally_impossible=True)
    ramp_end = 0.5e-9 + 100e-12
    Vb_peak = float(np.nanmax(Vb))
    mask = (np.asarray(Vb) >= 0.5)
    t_to_05 = (float(t_arr[np.argmax(mask)]) - ramp_end) * 1e9 if mask.any() else float("inf")
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.plot(np.array(t_arr) * 1e9, Vd_arr, "k-", lw=0.7, label="V_D")
    ax.plot(np.array(t_arr) * 1e9, Vb,    "b-", lw=1.4, label="V_B")
    ax.axhline(0.5, color="red", ls=":", label="0.5V")
    ax.set_xlabel("time [ns]"); ax.set_ylabel("V [V]")
    ax.set_title(f"V4 — Ns-snap rise: V_B_peak={Vb_peak:.3f}V, "
                 f"t→0.5V={t_to_05:.2f}ns")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "plot_V4_ns_snap.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    passed = (t_to_05 < 5.0) and (Vb_peak > 0.5)
    return TestResult(
        test_id="V4", name="Ns-snap rise",
        plot_path=str(p),
        metric_value=t_to_05, metric_unit="ns (t→0.5V)",
        gate="t→0.5V < 5ns AND V_B_peak > 0.5V",
        passed=passed,
        notes=f"V_B_peak={Vb_peak:.3f}V t→0.5V={t_to_05:.2f}ns",
        source_path=str(p),
    )


def run_V5_latch_hold(cfg_flags, model_M1, model_M2, sebas_rows, out_dir, log):
    log("V5 — Latch hold (100ns)")
    t_arr, Vd_arr = stim_pulse_general(V_lo=0.05, V_hi=2.0,
                                         t_pre=0.5e-9, t_rise=100e-12,
                                         t_hold=100e-9, t_fall=100e-12,
                                         t_post=5e-9, n_total=1200)
    Vb, Id, r = _transient_VB_trace(cfg_flags, model_M1, model_M2,
                                    sebas_rows, 0.6, 0.0, t_arr, Vd_arr,
                                    max_step=2e-10)
    if Vb is None:
        return TestResult("V5", "Latch hold", "", float("nan"), "V",
                          "V_B mean 50-100ns > 0.5V", False, "no sebas row",
                          "", structurally_impossible=True)
    t_ns = np.array(t_arr) * 1e9
    mask = (t_ns >= 50.0) & (t_ns <= 100.0)
    Vb_avg = float(np.nanmean(Vb[mask])) if mask.any() else float("nan")
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.plot(t_ns, Vd_arr, "k-", lw=0.7, label="V_D")
    ax.plot(t_ns, Vb,    "b-", lw=1.4, label="V_B")
    ax.axhline(0.5, color="red", ls=":", label="0.5V threshold")
    ax.axvspan(50, 100, color="green", alpha=0.15, label="hold window")
    ax.set_xlabel("time [ns]"); ax.set_ylabel("V [V]")
    ax.set_title(f"V5 — Latch hold: V_B avg(50-100ns) = {Vb_avg:.3f}V")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "plot_V5_latch_hold.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    passed = (Vb_avg > 0.5)
    return TestResult(
        test_id="V5", name="Latch hold",
        plot_path=str(p),
        metric_value=Vb_avg, metric_unit="V (mean V_B, 50-100ns)",
        gate="V_B_avg > 0.5V during hold",
        passed=passed,
        notes=f"avg V_B in [50,100]ns = {Vb_avg:.3f}V",
        source_path=str(p),
    )


def run_V6_self_reset(cfg_flags, model_M1, model_M2, sebas_rows, out_dir, log):
    log("V6 — Self-reset (1us hold then release)")
    t_arr, Vd_arr = stim_pulse_general(V_lo=0.05, V_hi=2.0,
                                         t_pre=10e-9, t_rise=100e-12,
                                         t_hold=1e-6, t_fall=100e-12,
                                         t_post=100e-9, n_total=1500)
    Vb, Id, r = _transient_VB_trace(cfg_flags, model_M1, model_M2,
                                    sebas_rows, 0.6, 0.0, t_arr, Vd_arr,
                                    max_step=5e-9)
    if Vb is None:
        return TestResult("V6", "Self-reset", "", float("nan"), "ns",
                          "t_reset<100us AND V_B post-release<0.3V", False,
                          "no sebas row", "", structurally_impossible=True)
    t_ns = np.array(t_arr) * 1e9
    # Release time
    t_release_ns = (10e-9 + 100e-12 + 1e-6 + 100e-12) * 1e9
    post = t_ns >= t_release_ns
    Vb_post = float(np.nanmean(Vb[post])) if post.any() else float("nan")
    # Time to reset (V_B < 0.3V after release)
    if post.any():
        idx_post = np.where(post)[0]
        below = Vb[idx_post] < 0.3
        if below.any():
            t_reset_ns = float(t_ns[idx_post[np.argmax(below)]] - t_release_ns)
        else:
            t_reset_ns = float("inf")
    else:
        t_reset_ns = float("inf")
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(t_ns, Vd_arr, "k-", lw=0.7, label="V_D")
    ax.plot(t_ns, Vb,    "b-", lw=1.0, label="V_B")
    ax.axhline(0.3, color="red", ls=":", label="0.3V reset")
    ax.axvline(t_release_ns, color="grey", ls="--", label="release")
    ax.set_xlabel("time [ns]"); ax.set_ylabel("V [V]")
    ax.set_title(f"V6 — Self-reset: V_B post={Vb_post:.3f}V, "
                 f"t→reset={t_reset_ns:.1f}ns")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "plot_V6_self_reset.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    passed = (t_reset_ns < 1e5) and (Vb_post < 0.3)
    return TestResult(
        test_id="V6", name="Self-reset",
        plot_path=str(p),
        metric_value=t_reset_ns, metric_unit="ns (t→reset post-release)",
        gate="t_reset<100µs AND V_B post-release<0.3V",
        passed=passed,
        notes=f"t_reset={t_reset_ns:.1f}ns V_B_post={Vb_post:.3f}V",
        source_path=str(p),
    )


def run_V7_oscillation(cfg_flags, model_M1, model_M2, sebas_rows, out_dir, log):
    log("V7 — Relaxation oscillation (5us constant drive)")
    # Constant drive at V_d = 2.0V for 5us after fast rise.
    t_arr, Vd_arr = stim_pulse_general(V_lo=0.05, V_hi=2.0,
                                         t_pre=10e-9, t_rise=100e-12,
                                         t_hold=5e-6, t_fall=100e-12,
                                         t_post=100e-9, n_total=2500)
    Vb, Id, r = _transient_VB_trace(cfg_flags, model_M1, model_M2,
                                    sebas_rows, 0.6, 0.0, t_arr, Vd_arr,
                                    max_step=20e-9)
    if Vb is None:
        return TestResult("V7", "Relaxation oscillation", "", 0.0, "cycles",
                          ">=3 cycles, period 100-1000ns", False,
                          "no sebas row", "", structurally_impossible=True)
    # Cycle detection: V_B crosses 0.5V upward
    t_ns = np.array(t_arr) * 1e9
    Vb_arr = np.array(Vb)
    finite = np.isfinite(Vb_arr)
    if not finite.any():
        n_cycles = 0
        period_ns = float("nan")
    else:
        crossings = []
        for i in range(1, len(Vb_arr)):
            if (np.isfinite(Vb_arr[i]) and np.isfinite(Vb_arr[i-1])
                and Vb_arr[i-1] < 0.5 <= Vb_arr[i]):
                crossings.append(t_ns[i])
        n_cycles = max(0, len(crossings) - 1)
        if len(crossings) >= 2:
            period_ns = float(np.mean(np.diff(crossings)))
        else:
            period_ns = float("nan")
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    ax.plot(t_ns, Vb_arr, "b-", lw=0.9, label="V_B")
    ax.axhline(0.5, color="red", ls=":", label="0.5V")
    ax.set_xlabel("time [ns]"); ax.set_ylabel("V_B [V]")
    ax.set_title(f"V7 — Oscillation: n_cycles={n_cycles}, "
                 f"period={period_ns:.1f}ns")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "plot_V7_oscillation.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    passed = (n_cycles >= 3) and (100 <= period_ns <= 1000)
    return TestResult(
        test_id="V7", name="Relaxation oscillation",
        plot_path=str(p),
        metric_value=float(n_cycles), metric_unit="cycles (over 5µs)",
        gate=">=3 cycles AND period in [100,1000]ns",
        passed=passed,
        notes=f"n_cycles={n_cycles} period={period_ns:.1f}ns",
        source_path=str(p),
    )


def run_V8_lif_integrate(cfg_flags, model_M1, model_M2, sebas_rows, out_dir, log):
    log("V8 — LIF integrate (ramp 0→1V over 1µs)")
    t_arr, Vd_arr = stim_ramp(V_lo=0.05, V_hi=1.0, t_ramp=1e-6, n_total=600)
    Vb, Id, r = _transient_VB_trace(cfg_flags, model_M1, model_M2,
                                    sebas_rows, 0.6, 0.0, t_arr, Vd_arr,
                                    max_step=5e-9)
    if Vb is None:
        return TestResult("V8", "LIF integrate", "", 0.0, "V/µs",
                          "non-zero positive dV_B/dt at V_d=0.5V", False,
                          "no sebas row", "", structurally_impossible=True)
    Vb_arr = np.array(Vb)
    # Find index closest to V_d=0.5V
    idx = int(np.argmin(np.abs(np.asarray(Vd_arr) - 0.5)))
    # Slope via finite diff
    lo = max(1, idx - 5); hi = min(len(t_arr) - 1, idx + 5)
    if hi > lo and np.isfinite(Vb_arr[lo]) and np.isfinite(Vb_arr[hi]):
        slope = (Vb_arr[hi] - Vb_arr[lo]) / (t_arr[hi] - t_arr[lo]) / 1e6  # V/µs
    else:
        slope = float("nan")
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    t_us = np.array(t_arr) * 1e6
    ax.plot(t_us, Vd_arr, "k-", lw=0.7, label="V_D ramp")
    ax.plot(t_us, Vb_arr, "b-", lw=1.4, label="V_B")
    ax.axvline(t_us[idx], color="red", ls="--",
               label=f"V_D=0.5V (slope={slope:.3g} V/µs)")
    ax.set_xlabel("time [µs]"); ax.set_ylabel("V [V]")
    ax.set_title("V8 — Sub-threshold LIF integrate (V_D ramp)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "plot_V8_lif_integrate.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    passed = np.isfinite(slope) and slope > 0.0
    return TestResult(
        test_id="V8", name="LIF integrate",
        plot_path=str(p),
        metric_value=float(slope), metric_unit="V/µs (dV_B/dt @ V_D=0.5V)",
        gate="non-zero positive slope",
        passed=passed,
        notes=f"dV_B/dt @ V_D=0.5V = {slope:.4g} V/µs",
        source_path=str(p),
    )


def run_V9_threshold_gain(cfg_flags, model_M1, model_M2, sebas_rows,
                          out_dir, log):
    log("V9 — LIF threshold gain (V_drive sweep)")
    Vd_drives = [1.5, 1.7, 1.9, 2.1]
    spikes = []
    traces = {}
    for Vd_drive in Vd_drives:
        t_arr, Vd_arr = stim_pulse_general(V_lo=0.05, V_hi=Vd_drive,
                                             t_pre=10e-9, t_rise=100e-12,
                                             t_hold=1e-6, t_fall=100e-12,
                                             t_post=50e-9, n_total=1000)
        Vb, Id, r = _transient_VB_trace(cfg_flags, model_M1, model_M2,
                                        sebas_rows, 0.6, 0.0, t_arr, Vd_arr,
                                        max_step=10e-9)
        if Vb is None:
            spikes.append(0); continue
        Vb_arr = np.array(Vb)
        t_ns = np.array(t_arr) * 1e9
        # Count up-crossings of 0.5V (after settling 10ns)
        mask = t_ns > 10.0
        cnt = 0
        for i in range(1, len(Vb_arr)):
            if (mask[i] and np.isfinite(Vb_arr[i]) and np.isfinite(Vb_arr[i-1])
                and Vb_arr[i-1] < 0.5 <= Vb_arr[i]):
                cnt += 1
        # Convert spikes/µs (hold = 1µs)
        spikes_per_us = cnt / 1.0
        spikes.append(spikes_per_us)
        traces[Vd_drive] = (t_ns, Vb_arr)
        log(f"  V_drive={Vd_drive}V → {cnt} spikes ({spikes_per_us:.1f}/µs)")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    for Vd_drive in Vd_drives:
        if Vd_drive in traces:
            t_ns, Vb_arr = traces[Vd_drive]
            axes[0].plot(t_ns, Vb_arr, lw=0.8, label=f"V_drive={Vd_drive}V")
    axes[0].axhline(0.5, color="red", ls=":", label="0.5V")
    axes[0].set_xlabel("time [ns]"); axes[0].set_ylabel("V_B [V]")
    axes[0].set_title("V9 — V_B(t) for different drive amplitudes")
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)
    axes[1].plot(Vd_drives, spikes, "o-", lw=1.5)
    axes[1].set_xlabel("V_drive [V]"); axes[1].set_ylabel("spikes/µs")
    axes[1].set_title("V9 — Threshold gain f(V_drive)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "plot_V9_threshold_gain.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    # Monotonic non-decreasing AND at least one step increase
    mono = all(spikes[i+1] >= spikes[i] for i in range(len(spikes)-1))
    has_increase = (max(spikes) > min(spikes))
    passed = mono and has_increase
    return TestResult(
        test_id="V9", name="LIF threshold gain",
        plot_path=str(p),
        metric_value=float(max(spikes) - min(spikes)),
        metric_unit="Δ spikes/µs (max-min)",
        gate="monotonic non-decreasing AND max>min",
        passed=passed,
        notes=f"spikes/µs by V_drive: " +
              ", ".join(f"{vd}V→{s:.1f}" for vd, s in zip(Vd_drives, spikes)),
        source_path=str(p),
    )


# ------------------------------------------------------------------ #
# Summary plot
# ------------------------------------------------------------------ #
def make_summary(out_dir, results, log):
    fig = plt.figure(figsize=(18, 14))
    for i, res in enumerate(results):
        ax = fig.add_subplot(3, 3, i + 1)
        try:
            from matplotlib.image import imread
            img = imread(res.plot_path)
            ax.imshow(img); ax.axis("off")
        except Exception:
            ax.text(0.5, 0.5, "no plot", ha="center", va="center")
            ax.axis("off")
        status = "PASS" if res.passed else (
            "N/A" if res.structurally_impossible else "FAIL")
        color = "green" if res.passed else ("orange" if res.structurally_impossible else "red")
        ax.set_title(f"{res.test_id} {res.name} [{status}]\n"
                     f"{res.metric_value:.3g} {res.metric_unit}",
                     fontsize=9, color=color)
    fig.suptitle(f"z461 Dynamics Validation Summary", fontsize=14)
    fig.tight_layout()
    p = out_dir / "validation_summary.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    log(f"  wrote {p.name}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="NX_1p8",
                    choices=["SB_OFF", "SB_HOT", "NX_1p8", "z458_best"])
    ap.add_argument("--skip", nargs="*", default=[],
                    help="test ids to skip (e.g. V7)")
    args = ap.parse_args()

    out_dir = ROOT / f"results/z461_validation_{args.config}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "run.log", "w")

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True); log_file.write(line + "\n"); log_file.flush()

    log("PRE-REGISTERED GATES (z461, locked):")
    log("  INFRA      = all 9 tests run + plots written")
    log("  DISCOVERY  = >=6/9 tests PASS")
    log("  AMBITIOUS  = >=8/9 tests PASS AND validation_summary.png publication-quality")
    log("  KILL_SHOT  = >=3/9 tests structurally impossible to pass")
    log("")
    log(f"config = {args.config}")
    cfg_flags = make_config(args.config)
    log(f"flags  = {json.dumps({k:v for k,v in cfg_flags.items() if not k.startswith('_')}, default=str)}")
    if "_R_body" in cfg_flags:
        log(f"transient R_body = {cfg_flags['_R_body']}")

    t0 = time.time()
    log("loading models + measured curves + Sebas params")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"  {len(curves)} curves, {len(sebas_rows)} sebas rows")

    results = []
    test_runners = [
        ("V1", lambda: run_V1_dc_per_branch(cfg_flags, model_M1, model_M2,
                                             curves, sebas_rows, out_dir, log)),
        ("V2", lambda: run_V2_hysteresis(cfg_flags, model_M1, model_M2,
                                          curves, sebas_rows, out_dir, log)),
        ("V3", lambda: run_V3_knee_position(cfg_flags, model_M1, model_M2,
                                             sebas_rows, out_dir, log)),
        ("V4", lambda: run_V4_ns_snap(cfg_flags, model_M1, model_M2,
                                       sebas_rows, out_dir, log)),
        ("V5", lambda: run_V5_latch_hold(cfg_flags, model_M1, model_M2,
                                          sebas_rows, out_dir, log)),
        ("V6", lambda: run_V6_self_reset(cfg_flags, model_M1, model_M2,
                                          sebas_rows, out_dir, log)),
        ("V7", lambda: run_V7_oscillation(cfg_flags, model_M1, model_M2,
                                           sebas_rows, out_dir, log)),
        ("V8", lambda: run_V8_lif_integrate(cfg_flags, model_M1, model_M2,
                                             sebas_rows, out_dir, log)),
        ("V9", lambda: run_V9_threshold_gain(cfg_flags, model_M1, model_M2,
                                              sebas_rows, out_dir, log)),
    ]
    v1_panels = None
    for tid, fn in test_runners:
        if tid in args.skip:
            log(f"SKIP {tid}")
            continue
        thermal_pause(log)
        try:
            t1 = time.time()
            out = fn()
            if tid == "V1":
                tr, v1_panels = out
                results.append(tr)
            else:
                results.append(out)
            log(f"  {tid} done in {time.time()-t1:.1f}s -- "
                f"{'PASS' if results[-1].passed else 'FAIL'}")
        except Exception as e:
            import traceback
            log(f"  {tid} EXCEPTION: {e}")
            log(traceback.format_exc())
            results.append(TestResult(
                test_id=tid, name=tid + " (failed to run)",
                plot_path="", metric_value=float("nan"), metric_unit="",
                gate="", passed=False, notes=f"exception: {e}",
                source_path="", structurally_impossible=False))

    # Summary plot
    make_summary(out_dir, results, log)

    # JSON table
    table = {
        "config": args.config,
        "config_flags": {k: v for k, v in cfg_flags.items()
                         if not k.startswith("_")},
        "wall_sec": time.time() - t0,
        "tests": [],
    }
    for r in results:
        table["tests"].append({
            "test_id": r.test_id, "name": r.name,
            "metric_value": (None if (isinstance(r.metric_value, float)
                                      and math.isnan(r.metric_value))
                             else r.metric_value),
            "metric_unit": r.metric_unit,
            "gate": r.gate,
            "passed": r.passed,
            "structurally_impossible": r.structurally_impossible,
            "notes": r.notes,
            "plot_path": r.plot_path,
            "source_path": r.source_path,
        })
    n_pass = sum(1 for r in results if r.passed)
    n_na = sum(1 for r in results if r.structurally_impossible)
    n_total = len(results)
    table["summary"] = {
        "pass": n_pass, "na": n_na, "fail": n_total - n_pass - n_na,
        "total": n_total,
    }
    table["gates"] = {
        "INFRA": all(r.plot_path or r.structurally_impossible for r in results),
        "DISCOVERY": n_pass >= 6,
        "AMBITIOUS": n_pass >= 8,
        "KILL_SHOT": n_na >= 3,
    }
    def _jsan(o):
        if isinstance(o, (bool, np.bool_)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not serializable: {type(o)}")
    (out_dir / "validation_table.json").write_text(json.dumps(table, indent=2, default=_jsan))
    log(f"  wrote validation_table.json")

    # Acceptance card
    card_lines = [
        f"# z461 Acceptance Card — config = {args.config}",
        "",
        f"**{n_pass}/{n_total} dynamics validated** "
        f"(N/A: {n_na}, FAIL: {n_total - n_pass - n_na})",
        "",
        "| # | Test | Metric | Gate | Verdict |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        v = ("N/A" if r.structurally_impossible
             else ("PASS" if r.passed else "FAIL"))
        mv = ("nan" if (isinstance(r.metric_value, float)
                        and math.isnan(r.metric_value))
              else f"{r.metric_value:.4g}")
        card_lines.append(
            f"| {r.test_id} | {r.name} | {mv} {r.metric_unit} | {r.gate} | **{v}** |")
    card_lines.append("")
    card_lines.append("## Gate evaluation")
    for k, v in table["gates"].items():
        card_lines.append(f"- {k} : {'PASS' if v else 'FAIL'}")
    (out_dir / "acceptance_card.md").write_text("\n".join(card_lines))
    log("  wrote acceptance_card.md")

    # Report.md with optional diff vs another config
    rep_lines = [
        f"# z461 Dynamics Validation Report — {args.config}",
        "",
        f"Wall: {time.time()-t0:.1f}s  Pass: {n_pass}/{n_total}  "
        f"N/A: {n_na}  Fail: {n_total-n_pass-n_na}",
        "",
        "## Per-test results",
        "",
    ]
    for r in results:
        v = ("N/A" if r.structurally_impossible
             else ("PASS" if r.passed else "FAIL"))
        rep_lines.append(f"### {r.test_id} {r.name} — **{v}**")
        rep_lines.append(f"- metric: {r.metric_value} {r.metric_unit}")
        rep_lines.append(f"- gate: {r.gate}")
        rep_lines.append(f"- notes: {r.notes}")
        rep_lines.append(f"- plot: `{r.plot_path}`")
        rep_lines.append("")

    # Diff against another config if both exist
    rep_lines.append("## Comparison with other configs")
    other_configs = [c for c in ["SB_OFF", "SB_HOT", "NX_1p8", "z458_best"]
                     if c != args.config]
    diff_rows = []
    this_metrics = {r.test_id: r for r in results}
    for oc in other_configs:
        ojson = ROOT / f"results/z461_validation_{oc}/validation_table.json"
        if not ojson.exists():
            continue
        try:
            other = json.loads(ojson.read_text())
            other_m = {t["test_id"]: t for t in other.get("tests", [])}
            for tid, tr in this_metrics.items():
                if tid in other_m:
                    o = other_m[tid]
                    diff_rows.append((tid, args.config, oc,
                                      tr.passed, o["passed"],
                                      tr.metric_value, o["metric_value"]))
        except Exception:
            continue
    if diff_rows:
        rep_lines.append("| Test | This | Other | This-pass | Other-pass | "
                         "This-val | Other-val |")
        rep_lines.append("|---|---|---|---|---|---|---|")
        for row in diff_rows:
            rep_lines.append(f"| {row[0]} | {row[1]} | {row[2]} | "
                             f"{row[3]} | {row[4]} | "
                             f"{row[5]} | {row[6]} |")
    else:
        rep_lines.append("(no comparison configs found on disk)")
    (out_dir / "report.md").write_text("\n".join(rep_lines))
    log("  wrote report.md")

    log(f"DONE: {n_pass}/{n_total} PASS ({n_na} N/A)  wall={time.time()-t0:.1f}s")
    log_file.close()


if __name__ == "__main__":
    main()
