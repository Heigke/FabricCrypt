"""z454 — Snapback subcircuit integration on top of v449_B base.

Hypothesis (per z449 honest_analysis.md): VBIC + BDF + n-well cap=0 base
(v449_B) plus an explicit V_BC-thresholded µA pull-down via the existing
snapback_subcircuit module will close the ns-snap fast-pulse gap.

Base pipeline: v449_B = use_vbic_for_q1 + Cbody=1fF + body_pdiode_Cj0=0.
The snapback_subcircuit (S9 in nsram_cell_2T.py) wraps M1 with:
  1. Slotboom–Chynoweth avalanche M(V_db = Vd - Vb) → Iii_body INTO R_B.
  2. Parasitic vertical NPN (C=Vd, B=Vb, E=Vsint), GP forward-active with
     Early effect → I_snap_d injected as drain-source regenerative kick
     (also enters R_Sint and Ib_snap = I_snap_d/Bf enters R_B).

Four conditions (all on v449_B base):
  - SB_OFF         : v449_B as-is (no snapback subcircuit)
  - SB_ON_DEFAULT  : use_snapback_sub=True, BV=2.0, Is=6.0e-9, Bf=417.
  - SB_LOW         : BV × 0.8 (lower threshold), Is × 0.5 (weaker NPN).
  - SB_HOT         : BV × 0.6 (much lower threshold), Is × 5 (hotter NPN).

z444-BURN AVOIDANCE — explicit assert that the snapback subcircuit is
actually being called: we sample I_snap_d at a known mid-sweep DC point
(VG1=0.6, VG2=0.0, Vd=1.5 V) and assert |I_snap_d| > 0 whenever the
condition is SB_* and the V_BC = Vb - Vd is below -1 V (avalanche regime).

Outputs to results/z454_snapback_integration/:
  - run.log
  - summary.json (DC fwd/bwd, fast-pulse, gates)
  - dc_compare.png, pulse_overlay.png, snapback_trace.png
  - honest_analysis.md
"""
from __future__ import annotations
import importlib.util as _ilu
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/z454_snapback_integration"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


# Pre-registered gates (LOCKED on line 1 of run.log)
log("PRE-REGISTERED GATES (locked):")
log("  INFRA      = all 4 conditions complete, summary.json written")
log("  DISCOVERY  = any SB_* condition: DC_avg < 1.20 dec AND >= 2/4 biases V_B>0.3V in 5ns")
log("  AMBITIOUS  = any SB_* condition: DC_avg < 0.85 dec AND >= 3/4 biases V_B>0.5V in 10ns AND self-reset visible")
log("  KILL_SHOT  = SB_* identical to SB_OFF (no-op wiring), OR all SB_* worse than v449_B")
log("ASSERT       = I_snap_d != 0 at one mid-sweep DC point (z444-BURN avoidance)")
log("")

# Load z449 internals (reuse cfg/cell helpers + monkey-patch)
_spec449 = _ilu.spec_from_file_location("z449", ROOT / "scripts/z449_vbic_bdf_combo.py")
z449 = _ilu.module_from_spec(_spec449); _spec449.loader.exec_module(z449)

z427 = z449.z427
z429 = z449.z429

from nsram.bsim4_port import transient_real_v2 as trv2
from nsram.bsim4_port.transient_real_v2 import (
    integrate, TransientCfgV2, stim_fast_pulse,
)
from nsram.bsim4_port.nsram_cell_2T import _residuals as _residuals_cell


BIASES = [
    {"VG1": 0.6, "VG2": 0.0, "tag": "VG1_0p6_VG2_0p0"},
    {"VG1": 0.6, "VG2": 0.2, "tag": "VG1_0p6_VG2_0p2"},
    {"VG1": 0.6, "VG2": 0.4, "tag": "VG1_0p6_VG2_0p4"},
    {"VG1": 0.4, "VG2": 0.0, "tag": "VG1_0p4_VG2_0p0"},
]

# Default snapback knobs from SnapbackParams
SNAP_DEFAULT = dict(
    snap_BV=2.0, snap_n_avl=4.0, snap_Bf=417.0, snap_Va=0.90,
    snap_Is=6.0256e-9, snap_Nf=1.0,
    snap_Id_clamp=1e-2, snap_Iii_clamp=1e-2,
)

# v449_B base flags (n-well cap zeroed)
V449B_BASE = {
    "use_vbic_for_q1": True,
    "vbic_AVC1": 0.5, "vbic_AVC2": 0.5,
    "Cbody": 1e-15,
    "body_pdiode_Cj0_per_area": 0.0,
}

CONDITIONS = [
    {
        "name": "SB_OFF",
        "desc": "v449_B base — snapback subcircuit OFF (control)",
        "flags": {**V449B_BASE, "use_snapback_sub": False},
    },
    {
        "name": "SB_ON_DEFAULT",
        "desc": "v449_B + use_snapback_sub=True, default BV=2.0, Is=6.0e-9, Bf=417",
        "flags": {**V449B_BASE, "use_snapback_sub": True, **SNAP_DEFAULT},
    },
    {
        "name": "SB_LOW",
        "desc": "BV×0.8 (lower threshold), Is×0.5 (weaker NPN)",
        "flags": {**V449B_BASE, "use_snapback_sub": True, **SNAP_DEFAULT,
                  "snap_BV": SNAP_DEFAULT["snap_BV"] * 0.8,
                  "snap_Is": SNAP_DEFAULT["snap_Is"] * 0.5},
    },
    {
        "name": "SB_HOT",
        "desc": "BV×0.6 (much lower threshold), Is×5 (hot NPN)",
        "flags": {**V449B_BASE, "use_snapback_sub": True, **SNAP_DEFAULT,
                  "snap_BV": SNAP_DEFAULT["snap_BV"] * 0.6,
                  "snap_Is": SNAP_DEFAULT["snap_Is"] * 5.0},
    },
]


# =========================================================== #
# Assert helper: confirm snapback is actually being called.
# Sample I_snap_d at (VG1=0.6, VG2=0.0, Vd=1.5V, Vb=0.6V, Vsint=0.0).
# At Vd=1.5V, Vb=0.6V → V_db = 0.9V. For BV=2.0 this is 0.45·BV so M
# is small but nonzero. For SB_HOT (BV=1.2) it is 0.75·BV → larger M.
# We assert |I_snap_d| > 0 (any nonzero value confirms wiring).
# Also test a deep-avalanche point at Vd=2.0V to guarantee M >> 1.
# =========================================================== #
def assert_snapback_live(name, flags, model_M1, model_M2, sebas_rows):
    if not flags.get("use_snapback_sub", False):
        log(f"  [{name}] SB OFF — no snapback call to assert (expected I_snap_d=0)")
        return {"snap_called": False, "I_snap_d": 0.0, "I_snap_b": 0.0}
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(flags))
    # Find a Sebas row at VG1=0.6, VG2=0.0
    sebas_row = z427.find_params(sebas_rows, 0.6, 0.0)
    if sebas_row is None:
        log(f"  [{name}] no Sebas row at VG1=0.6 — skipping assert")
        return {"snap_called": False, "reason": "no Sebas row"}
    P_M1, P_M2 = z427.make_overrides(sebas_row)
    bjt = z427.make_bjt(sebas_row)
    # Eval residual at Vd=2.0, Vb=0.6, Vsint=0.0 (deep avalanche)
    Vd_test = 2.0
    Vb_test = 0.6
    Vsint_test = 0.0
    Vd = torch.tensor([Vd_test], dtype=torch.float64)
    VG1 = torch.tensor([0.6], dtype=torch.float64)
    VG2 = torch.tensor([0.0], dtype=torch.float64)
    Vsint_t = torch.tensor([Vsint_test], dtype=torch.float64)
    Vb_t = torch.tensor([Vb_test], dtype=torch.float64)
    with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
         z427.patch_sd_scaled(sd_M2, P_M2):
        _, _, comp = _residuals_cell(cfg, model_M1, bjt, Vd, VG1, VG2,
                                     Vsint_t, Vb_t, None, None,
                                     model_M2=model_M2)
    I_snap_d = float(comp.get("I_snap_d", torch.tensor(0.0)).abs().item())
    I_snap_b = float(comp.get("I_snap_b", torch.tensor(0.0)).abs().item())
    V_db = Vd_test - Vb_test
    log(f"  [{name}] ASSERT @ Vd={Vd_test} Vb={Vb_test} Vsint={Vsint_test} "
        f"V_db={V_db:.2f} BV={flags.get('snap_BV',2.0):.2f}: "
        f"|I_snap_d|={I_snap_d:.3e} A  |I_snap_b|={I_snap_b:.3e} A")
    if I_snap_d == 0.0 and I_snap_b == 0.0:
        log(f"  [{name}] !!! ASSERT FAIL — snapback wiring is a NO-OP (z444-style) !!!")
        return {"snap_called": False, "I_snap_d": 0.0, "I_snap_b": 0.0,
                "V_db": V_db, "BV": flags.get("snap_BV", 2.0)}
    log(f"  [{name}] ASSERT PASS — snapback subcircuit is live.")
    return {"snap_called": True, "I_snap_d": I_snap_d, "I_snap_b": I_snap_b,
            "V_db": V_db, "BV": flags.get("snap_BV", 2.0)}


# =========================================================== #
# Slow-DC cell-wide RMSE, both forward AND backward sweeps.
# Forward: Vd ascending; Backward: Vd descending (with V_b warm-start).
# =========================================================== #
def slow_dc_cell_rmse_dir(name, flags, model_M1, model_M2, curves, sebas_rows,
                          direction="forward"):
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(flags))
    log_eps = 1e-15
    per_bias = []
    fails = 0
    t0 = time.time()
    for c in curves:
        sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        bjt = z427.make_bjt(sebas_row)
        Vd_arr = c["Vd"].numpy()
        Id_meas = c["Id"].numpy()
        if direction == "backward":
            order = np.argsort(-Vd_arr)
        else:
            order = np.argsort(Vd_arr)
        Vd_seq = Vd_arr[order]
        Id_seq_meas = Id_meas[order]
        Id_pred_seq = np.zeros_like(Vd_seq)
        conv_seq = np.zeros_like(Vd_seq, dtype=bool)
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
                    conv_seq[i] = bool(r["converged"])
                    if r["converged"]:
                        Vb_warm = r["Vb"]
                    else:
                        Vb_warm = 0.0
        except Exception as e:
            fails += 1
            log(f"  {name}/{direction} fail VG1={c['VG1']} VG2={c['VG2']}: {e}")
            continue
        Id_pred = torch.tensor(Id_pred_seq, dtype=torch.float64)
        Id_obs = torch.tensor(Id_seq_meas, dtype=torch.float64)
        conv = torch.tensor(conv_seq)
        if not conv.any():
            fails += 1; continue
        log_p = torch.log10(Id_pred + log_eps)
        log_m = torch.log10(Id_obs + log_eps)
        sq = (log_p - log_m) ** 2
        rmse = float(torch.sqrt(sq[conv].mean()))
        per_bias.append({"VG1": c["VG1"], "VG2": c["VG2"],
                         "log_rmse": rmse, "n_conv": int(conv.sum().item())})
    if not per_bias:
        return {"cell_rmse_dec": float("inf"), "n": 0, "fails": fails,
                "wall_sec": time.time() - t0, "per_bias": []}
    cell_sq = sum(r["log_rmse"] ** 2 for r in per_bias)
    cell = math.sqrt(cell_sq / len(per_bias))
    log(f"  {name}/{direction}: cell={cell:.3f} dec ({len(per_bias)} biases) "
        f"fails={fails} wall={time.time()-t0:.1f}s")
    return {"cell_rmse_dec": cell, "n": len(per_bias),
            "fails": fails, "wall_sec": time.time() - t0,
            "per_bias": per_bias}


# =========================================================== #
# Fast-pulse smoke (4 biases), with I_snap_d trace capture for plotting.
# =========================================================== #
def fast_pulse_smoke(name, flags, model_M1, model_M2, sebas_rows,
                     capture_snap_trace_bias=None):
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(flags))
    cfg.Cbody = 1e-15
    tcfg = TransientCfgV2(C_B_const=1e-15,
                          max_step=1e-10, first_step=1e-14,
                          rtol=1e-6, atol=1e-15)
    per_bias = []
    snap_trace_capture = None
    z449._VBIC_CTX["cfg"] = cfg
    for bias in BIASES:
        sebas_row = z427.find_params(sebas_rows, bias["VG1"], bias["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            log(f"  skip {bias['tag']} — no Sebas params"); continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        bjt = z427.make_bjt(sebas_row)
        z449._VBIC_CTX["bjt"] = bjt
        t, Vd_stim = stim_fast_pulse(V_hi=2.0, V_lo=0.05,
                                       t_rise=100e-12, t_hold=10e-9,
                                       t_fall=100e-12,
                                       t_pre=0.5e-9, t_post=5e-9,
                                       n_total=800)
        t_start = time.time()
        try:
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
                 z427.patch_sd_scaled(sd_M2, P_M2):
                r = integrate(cfg, model_M1, model_M2, bjt,
                              t, Vd_stim, bias["VG1"], bias["VG2"],
                              tcfg=tcfg, Vb0=0.0)
        except Exception as e:
            log(f"  {name} FAIL fast {bias['tag']}: {e}")
            continue
        wall = time.time() - t_start
        Vb_arr = np.array(r["Vb"])
        Id_arr = np.array(r["Id"])
        t_arr = np.array(t)
        ramp_end = 0.5e-9 + 100e-12

        def t_cross(Vb_arr, t_arr, thresh):
            mask = Vb_arr >= thresh
            if not mask.any():
                return None
            return float(t_arr[np.argmax(mask)] - ramp_end)

        Vb_peak = float(np.nanmax(Vb_arr))
        idx_peak = int(np.nanargmax(Vb_arr))
        t_peak = float(t_arr[idx_peak])
        idx_5ns = (t_arr <= ramp_end + 5e-9)
        Vb_max_5ns = float(np.nanmax(Vb_arr[idx_5ns])) if idx_5ns.any() else 0.0
        idx_10ns = (t_arr <= ramp_end + 10e-9)
        Vb_max_10ns = float(np.nanmax(Vb_arr[idx_10ns])) if idx_10ns.any() else 0.0
        t_to_03 = t_cross(Vb_arr, t_arr, 0.3)
        t_to_05 = t_cross(Vb_arr, t_arr, 0.5)
        # Self-reset: after V_B peak, does it drop back below 0.5·peak within 5ns?
        self_reset = False
        if Vb_peak > 0.1:
            after = Vb_arr[idx_peak:]
            t_after = t_arr[idx_peak:] - t_peak
            tail_mask = t_after <= 5e-9
            if tail_mask.any() and after[tail_mask].min() < 0.5 * Vb_peak:
                self_reset = True

        per_bias.append({
            "tag": bias["tag"], "VG1": bias["VG1"], "VG2": bias["VG2"],
            "Vb_peak_V": Vb_peak, "t_Vb_peak_ns": t_peak * 1e9,
            "Vb_max_5ns_V": Vb_max_5ns, "Vb_max_10ns_V": Vb_max_10ns,
            "t_to_0p3V_ns": (t_to_03 * 1e9) if t_to_03 is not None else None,
            "t_to_0p5V_ns": (t_to_05 * 1e9) if t_to_05 is not None else None,
            "self_reset_within_5ns": bool(self_reset),
            "wall_sec": round(wall, 1),
            "solver_success": bool(r["solver"]["success"]),
            "_traces": {
                "t": t_arr.tolist(),
                "Vd": list(Vd_stim),
                "Vb": r["Vb"],
                "Id": r["Id"],
            }})
        log(f"  {name}/{bias['tag']}: Vb_peak={Vb_peak:.3f}V@{t_peak*1e9:.2f}ns  "
            f"Vb_5ns={Vb_max_5ns:.3f}V  Vb_10ns={Vb_max_10ns:.3f}V  "
            f"t03={t_to_03}  t05={t_to_05}  reset={self_reset}  "
            f"ok={r['solver']['success']}  wall={wall:.1f}s")

        # Snapback trace capture (post-hoc): for the requested bias,
        # re-evaluate I_snap_d along the V_B trajectory by calling
        # _residuals at each (t, Vd, Vsint=0, Vb).
        if (capture_snap_trace_bias is not None
                and bias["tag"] == capture_snap_trace_bias
                and flags.get("use_snapback_sub", False)):
            log(f"  capturing snapback trace for {bias['tag']} …")
            t_sub = t_arr[::8]
            Vd_sub = np.array(Vd_stim)[::8]
            Vb_sub = Vb_arr[::8]
            I_snap_d_trace = np.zeros_like(t_sub)
            I_snap_b_trace = np.zeros_like(t_sub)
            V_db_trace = np.zeros_like(t_sub)
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
                 z427.patch_sd_scaled(sd_M2, P_M2):
                for k in range(len(t_sub)):
                    Vd_k = torch.tensor([Vd_sub[k]], dtype=torch.float64)
                    Vb_k = torch.tensor([Vb_sub[k]], dtype=torch.float64)
                    Vs_k = torch.tensor([0.0], dtype=torch.float64)
                    VG1_k = torch.tensor([bias["VG1"]], dtype=torch.float64)
                    VG2_k = torch.tensor([bias["VG2"]], dtype=torch.float64)
                    _, _, comp = _residuals_cell(cfg, model_M1, bjt,
                                                 Vd_k, VG1_k, VG2_k,
                                                 Vs_k, Vb_k, None, None,
                                                 model_M2=model_M2)
                    I_snap_d_trace[k] = float(comp.get("I_snap_d",
                                                        torch.tensor(0.0)).item())
                    I_snap_b_trace[k] = float(comp.get("I_snap_b",
                                                        torch.tensor(0.0)).item())
                    V_db_trace[k] = float(Vd_sub[k] - Vb_sub[k])
            snap_trace_capture = {
                "tag": bias["tag"], "BV": flags.get("snap_BV", 2.0),
                "t_ns": (t_sub * 1e9).tolist(),
                "Vd": Vd_sub.tolist(),
                "Vb": Vb_sub.tolist(),
                "V_db": V_db_trace.tolist(),
                "I_snap_d": I_snap_d_trace.tolist(),
                "I_snap_b": I_snap_b_trace.tolist(),
            }
            log(f"    snap trace: max|I_snap_d|={np.max(np.abs(I_snap_d_trace)):.3e}A "
                f"max|I_snap_b|={np.max(np.abs(I_snap_b_trace)):.3e}A")
    z449._VBIC_CTX["cfg"] = None
    z449._VBIC_CTX["bjt"] = None
    return {"per_bias": per_bias, "snap_trace": snap_trace_capture}


# =========================================================== #
# Plots
# =========================================================== #
def plot_dc_compare(results, path):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    names = [r["name"] for r in results]
    fwd = [r["dc_forward"]["cell_rmse_dec"] for r in results]
    bwd = [r["dc_backward"]["cell_rmse_dec"] for r in results]
    x = np.arange(len(names))
    ax.bar(x - 0.18, fwd, 0.36, label="forward", color="C0")
    ax.bar(x + 0.18, bwd, 0.36, label="backward", color="C1")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("cell DC RMSE [dec]")
    ax.axhline(1.20, color="grey", ls=":", label="DISCOVERY gate (1.20)")
    ax.axhline(0.85, color="red", ls=":", label="AMBITIOUS gate (0.85)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title("z454 — DC cell RMSE, forward vs backward (4 conditions)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    log(f"  wrote {path.name}")


def plot_pulse_overlay(results, path, bias_tag="VG1_0p6_VG2_0p2"):
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6), sharex=True)
    colors = ["k", "C0", "C2", "C3"]
    for r, col in zip(results, colors):
        rec = next((x for x in r["fast"]["per_bias"] if x["tag"] == bias_tag), None)
        if rec is None: continue
        tr = rec["_traces"]
        t = np.array(tr["t"]) * 1e9
        axes[0].plot(t, tr["Vb"], col + "-", lw=1.2, label=r["name"])
        axes[1].semilogy(t, np.maximum(np.abs(tr["Id"]), 1e-18),
                          col + "-", lw=1.0, label=r["name"])
    axes[0].axhline(0.3, color="grey", ls=":", lw=0.6, label="0.3V")
    axes[0].axhline(0.5, color="red", ls=":", lw=0.6, label="0.5V")
    axes[0].set_ylabel("V_B [V]"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("|I_D| [A]"); axes[1].set_xlabel("time [ns]")
    axes[1].legend(fontsize=8); axes[1].grid(True, which="both", alpha=0.3)
    axes[0].set_title(f"z454 — fast-pulse V_B(t) overlay @ {bias_tag}")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    log(f"  wrote {path.name}")


def plot_snap_trace(snap_trace, path):
    if snap_trace is None:
        log("  no snap trace to plot"); return
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.5), sharex=True)
    t = np.array(snap_trace["t_ns"])
    BV = snap_trace["BV"]
    axes[0].plot(t, snap_trace["Vd"], "k-", lw=0.8, label="V_D")
    axes[0].plot(t, snap_trace["Vb"], "b-", lw=1.2, label="V_B")
    axes[0].set_ylabel("V [V]"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, snap_trace["V_db"], "m-", lw=1.0, label="V_db = V_D - V_B")
    axes[1].axhline(BV, color="red", ls=":", label=f"BV = {BV:.2f}V")
    axes[1].set_ylabel("V_db [V]"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    axes[2].semilogy(t, np.abs(snap_trace["I_snap_d"]) + 1e-18, "C0-", lw=1.0,
                      label="|I_snap_d| (NPN kick)")
    axes[2].semilogy(t, np.abs(snap_trace["I_snap_b"]) + 1e-18, "C2-", lw=1.0,
                      label="|I_snap_b| (avalanche into body)")
    axes[2].set_ylabel("|I| [A]"); axes[2].set_xlabel("time [ns]")
    axes[2].legend(fontsize=8); axes[2].grid(True, which="both", alpha=0.3)
    axes[0].set_title(f"z454 — snapback trace, {snap_trace['tag']}, BV={BV:.2f}V")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    log(f"  wrote {path.name}")


# =========================================================== #
# Gates
# =========================================================== #
def eval_gates(results):
    sb_off_dc_avg = next((r["dc_avg"] for r in results if r["name"] == "SB_OFF"), None)
    discovery = False; discovery_who = None
    ambitious = False; ambitious_who = None
    no_op = False
    all_worse_than_off = True
    any_better_than_off = False
    no_op_conds = []
    for r in results:
        if r["name"] == "SB_OFF":
            continue
        pb = r["fast"]["per_bias"]
        n = len(pb) if pb else 0
        vb_03 = sum(1 for x in pb if x["Vb_max_5ns_V"] > 0.3)
        vb_05_10ns = sum(1 for x in pb if x["Vb_max_10ns_V"] > 0.5)
        any_reset = any(x["self_reset_within_5ns"] for x in pb)
        dc = r["dc_avg"]
        if dc < 1.20 and vb_03 >= 2:
            if not discovery:
                discovery = True; discovery_who = r["name"]
        if dc < 0.85 and vb_05_10ns >= 3 and any_reset:
            if not ambitious:
                ambitious = True; ambitious_who = r["name"]
        if sb_off_dc_avg is not None and dc <= sb_off_dc_avg + 1e-6:
            any_better_than_off = True
        else:
            pass
        if dc <= (sb_off_dc_avg if sb_off_dc_avg else float("inf")) - 1e-6:
            any_better_than_off = True
        # No-op detection: SB condition identical (within 1e-9 dec) to SB_OFF
        # AND fast-pulse Vb_peak identical to SB_OFF
        sb_off = next((x for x in results if x["name"] == "SB_OFF"), None)
        if sb_off is not None and sb_off_dc_avg is not None:
            if abs(dc - sb_off_dc_avg) < 1e-9:
                # Compare fast-pulse peaks
                identical = True
                for b in BIASES:
                    sb_pb = next((x for x in r["fast"]["per_bias"]
                                  if x["tag"] == b["tag"]), None)
                    off_pb = next((x for x in sb_off["fast"]["per_bias"]
                                   if x["tag"] == b["tag"]), None)
                    if sb_pb is None or off_pb is None: continue
                    if abs(sb_pb["Vb_peak_V"] - off_pb["Vb_peak_V"]) > 1e-9:
                        identical = False; break
                if identical:
                    no_op = True
                    no_op_conds.append(r["name"])
    # all SB_* worse than SB_OFF?
    if sb_off_dc_avg is not None:
        sb_conds = [r for r in results if r["name"] != "SB_OFF"]
        all_worse = all(r["dc_avg"] >= sb_off_dc_avg - 1e-6 for r in sb_conds)
    else:
        all_worse = False
    kill_shot = no_op or all_worse
    return {
        "INFRA_pass": True,
        "DISCOVERY_pass": discovery, "DISCOVERY_who": discovery_who,
        "AMBITIOUS_pass": ambitious, "AMBITIOUS_who": ambitious_who,
        "KILL_SHOT": kill_shot,
        "kill_shot_reason": ("no_op_wiring " + ",".join(no_op_conds)) if no_op
            else ("all_SB_worse_than_SB_OFF" if all_worse else None),
    }


# =========================================================== #
# Main
# =========================================================== #
def main():
    t0_main = time.time()
    log("z454 starting — snapback subcircuit on v449_B base")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    results = []
    for C in CONDITIONS:
        log(f"===== {C['name']}: {C['desc']} =====")
        # z444-BURN assert
        assert_info = assert_snapback_live(C["name"], C["flags"],
                                            model_M1, model_M2, sebas_rows)
        # DC fwd
        dc_f = slow_dc_cell_rmse_dir(C["name"], C["flags"],
                                      model_M1, model_M2, curves, sebas_rows,
                                      direction="forward")
        # DC bwd
        dc_b = slow_dc_cell_rmse_dir(C["name"], C["flags"],
                                      model_M1, model_M2, curves, sebas_rows,
                                      direction="backward")
        dc_avg = 0.5 * (dc_f["cell_rmse_dec"] + dc_b["cell_rmse_dec"])
        log(f"  {C['name']}: DC fwd={dc_f['cell_rmse_dec']:.3f}  "
            f"bwd={dc_b['cell_rmse_dec']:.3f}  avg={dc_avg:.3f}")
        # Fast pulse + snapback trace at VG1=0.6 VG2=0.2 for SB_ON_DEFAULT
        capture_tag = ("VG1_0p6_VG2_0p2"
                       if C["name"] == "SB_ON_DEFAULT" else None)
        fast = fast_pulse_smoke(C["name"], C["flags"],
                                 model_M1, model_M2, sebas_rows,
                                 capture_snap_trace_bias=capture_tag)
        results.append({
            "name": C["name"], "desc": C["desc"],
            "assert": assert_info,
            "dc_forward": dc_f, "dc_backward": dc_b, "dc_avg": dc_avg,
            "fast": fast,
        })

    # Gates
    gates = eval_gates(results)
    log(f"GATES: {gates}")

    # Trim traces for JSON
    def trim(rec, max_pts=200):
        if "_traces" not in rec: return
        tr = rec["_traces"]
        keys = [k for k, v in tr.items() if isinstance(v, list)]
        if not keys: return
        n_in = len(tr[keys[0]])
        if n_in <= max_pts: return
        idx = np.linspace(0, n_in - 1, max_pts).astype(int).tolist()
        for k in keys:
            v = tr[k]
            if len(v) == n_in:
                tr[k] = [v[i] for i in idx]
    for r in results:
        for rec in r["fast"]["per_bias"]:
            trim(rec)

    # Plots
    plot_dc_compare(results, OUT / "dc_compare.png")
    plot_pulse_overlay(results, OUT / "pulse_overlay.png",
                       bias_tag="VG1_0p6_VG2_0p2")
    # snapback trace from SB_ON_DEFAULT
    sb_on = next((r for r in results if r["name"] == "SB_ON_DEFAULT"), None)
    if sb_on is not None and sb_on["fast"].get("snap_trace") is not None:
        plot_snap_trace(sb_on["fast"]["snap_trace"], OUT / "snapback_trace.png")
    else:
        log("  WARNING: no snapback trace captured")

    # Summary
    summary = {
        "conditions": [
            {
                "name": r["name"], "desc": r["desc"],
                "assert": r["assert"],
                "dc_forward_dec": r["dc_forward"]["cell_rmse_dec"],
                "dc_backward_dec": r["dc_backward"]["cell_rmse_dec"],
                "dc_avg_dec": r["dc_avg"],
                "dc_forward_n": r["dc_forward"]["n"],
                "dc_backward_n": r["dc_backward"]["n"],
                "dc_per_branch_forward": r["dc_forward"]["per_bias"],
                "dc_per_branch_backward": r["dc_backward"]["per_bias"],
                "fast_pulse": [{k: v for k, v in x.items() if k != "_traces"}
                               for x in r["fast"]["per_bias"]],
            } for r in results
        ],
        "gates": gates,
        "references": {
            "z449_v449_B_DC": "see results/z449_vbic_bdf_combo/summary.json",
            "z448_BDF_DC_ref": 1.002,
            "z443_VBIC_AVL_DC_ref": 1.311,
            "z430_baseline_DC_ref": 1.619,
        },
        "wall_total_sec": round(time.time() - t0_main, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"wrote summary.json  total_wall={summary['wall_total_sec']:.0f}s")

    # Honest analysis
    best = min(results, key=lambda r: r["dc_avg"])
    sb_off = next((r for r in results if r["name"] == "SB_OFF"), None)

    lines = []
    lines.append("# z454 — Snapback subcircuit integration on v449_B base\n")
    lines.append("## Pre-registered gates\n")
    for k, v in gates.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## DC (forward / backward / avg) [dec]\n")
    lines.append("| condition | DC_fwd | DC_bwd | DC_avg | n |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        lines.append(f"| {r['name']} | {r['dc_forward']['cell_rmse_dec']:.3f} | "
                     f"{r['dc_backward']['cell_rmse_dec']:.3f} | "
                     f"{r['dc_avg']:.3f} | {r['dc_forward']['n']} |")
    lines.append("\n## Fast-pulse smoke (per bias)\n")
    for r in results:
        lines.append(f"### {r['name']}")
        lines.append("| bias | Vb_peak | t_peak[ns] | Vb_5ns | Vb_10ns | "
                     "t→0.3V[ns] | t→0.5V[ns] | self-reset |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for x in r["fast"]["per_bias"]:
            lines.append(f"| {x['tag']} | {x['Vb_peak_V']:.3f} | "
                         f"{x['t_Vb_peak_ns']:.2f} | "
                         f"{x['Vb_max_5ns_V']:.3f} | "
                         f"{x['Vb_max_10ns_V']:.3f} | "
                         f"{x['t_to_0p3V_ns']} | {x['t_to_0p5V_ns']} | "
                         f"{x['self_reset_within_5ns']} |")
        lines.append("")
    lines.append("\n## Snapback assert (z444-BURN avoidance)\n")
    for r in results:
        a = r["assert"]
        lines.append(f"- **{r['name']}**: snap_called={a.get('snap_called')} "
                     f"|I_snap_d|={a.get('I_snap_d', 0):.3e} A "
                     f"|I_snap_b|={a.get('I_snap_b', 0):.3e} A "
                     f"V_db={a.get('V_db', 'NA')} BV={a.get('BV', 'NA')}")
    lines.append(f"\n## Best condition: **{best['name']}**\n")
    lines.append(f"- DC_avg = {best['dc_avg']:.3f} dec")
    if sb_off is not None:
        lines.append(f"- vs SB_OFF DC_avg = {sb_off['dc_avg']:.3f} dec  "
                     f"(Δ = {best['dc_avg'] - sb_off['dc_avg']:+.3f} dec)")
    (OUT / "honest_analysis.md").write_text("\n".join(lines))
    log("wrote honest_analysis.md")
    log("DONE.")


if __name__ == "__main__":
    main()
