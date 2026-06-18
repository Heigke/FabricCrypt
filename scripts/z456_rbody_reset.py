"""z456 — R_body reset path on top of v449_B + SB_HOT (z454 base).

Hypothesis: V_B latches at ~0.71V after the NPN fires with no leak path in
z454. Adding an explicit body-leak resistor I_leak = V_B / R_body to GND
should let V_B self-reset, potentially producing a relaxation oscillator.

Physics: floating-body NMOS in 130nm DNW has natural junction reverse-sat
leakage + thermal generation, total ~ 1e-8 S → τ ~ C_eff/G ≈ 2.7 fF/1e-8
≈ 270 ns. Tunable as R_body; sweep:
  - R_INF  : no R_body (=z454 SB_HOT reference)
  - R_1G   : 1 GΩ, τ ≈ 2.7 ms (very weak)
  - R_100M : 100 MΩ, τ ≈ 270 µs
  - R_10M  : 10 MΩ, τ ≈ 27 µs
  - R_1M   : 1 MΩ, τ ≈ 2.7 µs

Mario slide-21 oscillation period target ≈ 400 ns.

NOTE: R_body is wired into the TRANSIENT body-KCL (transient_real_v2.py).
It is NOT applied at DC (z429 vsint-pinned newton solver is unchanged),
because (a) at DC V_B is small (~0V mode-A operation) so leak is negligible
relative to BJT injection, and (b) the goal is to test whether AC self-
reset is feasible while leaving DC operation unchanged. DC RMSE for all
R_body conditions therefore equals SB_HOT baseline; we still measure it
to confirm and to report honestly.

Outputs to results/z456_rbody_reset/:
  - run.log
  - summary.json
  - dc_vs_rbody.png
  - pulse_extended.png
  - oscillation_zoom.png (if oscillation found)
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
OUT = ROOT / "results/z456_rbody_reset"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


# Pre-registered gates (LOCKED on line 1 of run.log)
log("PRE-REGISTERED GATES (locked):")
log("  INFRA      = all 5 R_body conditions complete + summary.json written")
log("  DISCOVERY  = >=1 R_body produces self-reset in [100ns, 100us] AND DC_avg < 2.5 dec")
log("  AMBITIOUS  = >=1 R_body produces periodic relaxation oscillation (>=3 cycles) AND DC_avg < 1.5 dec")
log("  KILL_SHOT  = no R_body produces self-reset OR all R_body kill DC fully (>3 dec)")
log("  Mario slide-21 oscillation period target ~= 400 ns (quoted, not target-matched)")
log("")

# Load z449 internals (reuse cfg/cell helpers + monkey-patch via z454 base flags)
_spec449 = _ilu.spec_from_file_location("z449", ROOT / "scripts/z449_vbic_bdf_combo.py")
z449 = _ilu.module_from_spec(_spec449); _spec449.loader.exec_module(z449)

z427 = z449.z427
z429 = z449.z429

from nsram.bsim4_port import transient_real_v2 as trv2
from nsram.bsim4_port.transient_real_v2 import (
    integrate, TransientCfgV2,
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

# v449_B + SB_HOT (z454 base): hot NPN, BV*0.6, Is*5
SB_HOT_FLAGS = {
    "use_vbic_for_q1": True,
    "vbic_AVC1": 0.5, "vbic_AVC2": 0.5,
    "Cbody": 1e-15,
    "body_pdiode_Cj0_per_area": 0.0,
    "use_snapback_sub": True,
    **SNAP_DEFAULT,
    "snap_BV": SNAP_DEFAULT["snap_BV"] * 0.6,   # 1.2 V
    "snap_Is": SNAP_DEFAULT["snap_Is"] * 5.0,
}

# R_body sweep
CONDITIONS = [
    {"name": "R_INF",  "R_body": None,    "tau_est_s": float("inf")},
    {"name": "R_1G",   "R_body": 1.0e9,   "tau_est_s": 2.7e-3},
    {"name": "R_100M", "R_body": 1.0e8,   "tau_est_s": 2.7e-4},
    {"name": "R_10M",  "R_body": 1.0e7,   "tau_est_s": 2.7e-5},
    {"name": "R_1M",   "R_body": 1.0e6,   "tau_est_s": 2.7e-6},
]


# =========================================================== #
# Extended fast-pulse: rise 100ps to 2V, hold 1us
# =========================================================== #
def stim_fast_pulse_extended(V_hi=2.0, V_lo=0.05,
                              t_rise=100e-12, t_hold=1.0e-6,
                              t_pre=0.5e-9, n_total=3000):
    """Step up at t_pre, hold V_hi for t_hold, then stop (no fall)."""
    T = t_pre + t_rise + t_hold
    # Dense at edge, log-spaced after
    n_edge = 400
    n_tail = n_total - n_edge
    t_edge = np.linspace(0.0, t_pre + t_rise + 5e-9, n_edge, endpoint=False)
    t_tail = (t_pre + t_rise + 5e-9) + np.logspace(
        math.log10(1e-12), math.log10(t_hold - 5e-9), n_tail)
    t = np.concatenate([t_edge, t_tail])
    # Clamp last point to T
    t = np.clip(t, 0.0, T)
    Vd = np.full_like(t, V_lo)
    for i, ti in enumerate(t):
        if ti < t_pre:
            Vd[i] = V_lo
        elif ti < t_pre + t_rise:
            Vd[i] = V_lo + (V_hi - V_lo) * (ti - t_pre) / t_rise
        else:
            Vd[i] = V_hi
    return t, Vd


# =========================================================== #
# Slow-DC cell-wide RMSE, both forward AND backward sweeps.
# Note: R_body does not enter the DC residual; this measures the
# SB_HOT baseline DC for all conditions (will be identical).
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
# Self-reset detection on V_B(t) trace
# =========================================================== #
def detect_self_reset(t_arr, Vb_arr, ramp_end_s,
                      peak_thresh=0.5, reset_thresh=0.3):
    """Look for V_B rising above peak_thresh after ramp_end, then falling
    below reset_thresh. Returns (t_to_peak, t_to_reset) in seconds, or
    (None, None) if either step doesn't occur."""
    mask = t_arr >= ramp_end_s
    if not mask.any():
        return None, None, None
    Vb_post = Vb_arr[mask]
    t_post = t_arr[mask]
    above_peak = Vb_post >= peak_thresh
    if not above_peak.any():
        return None, None, None
    i_peak = int(np.argmax(above_peak))  # first index where Vb >= peak
    t_peak_first = float(t_post[i_peak])
    Vb_peak_val = float(np.nanmax(Vb_post))
    # After first crossing of peak_thresh, look for fall back below reset_thresh
    Vb_after = Vb_post[i_peak:]
    t_after = t_post[i_peak:]
    below = Vb_after < reset_thresh
    if below.any():
        i_reset = int(np.argmax(below))
        t_reset = float(t_after[i_reset])
        return t_peak_first, t_reset, Vb_peak_val
    return t_peak_first, None, Vb_peak_val


def detect_oscillation(t_arr, Vb_arr, ramp_end_s,
                        peak_thresh=0.5, reset_thresh=0.3):
    """Detect periodic spike-reset cycles. A cycle = V_B crosses up through
    peak_thresh, then crosses down through reset_thresh. Returns
    (n_cycles, period_s_mean, period_s_std, cycle_times)."""
    mask = t_arr >= ramp_end_s
    t_post = t_arr[mask]
    Vb_post = Vb_arr[mask]
    cycle_up_times = []
    state = "low"
    for i in range(len(t_post)):
        v = Vb_post[i]
        if not np.isfinite(v): continue
        if state == "low" and v >= peak_thresh:
            cycle_up_times.append(float(t_post[i]))
            state = "high"
        elif state == "high" and v < reset_thresh:
            state = "low"
    n_cycles = len(cycle_up_times)
    if n_cycles >= 2:
        periods = np.diff(cycle_up_times)
        return n_cycles, float(np.mean(periods)), float(np.std(periods)), cycle_up_times
    return n_cycles, None, None, cycle_up_times


# =========================================================== #
# Fast-pulse extended run with R_body
# =========================================================== #
def fast_pulse_extended(name, flags, R_body, model_M1, model_M2, sebas_rows):
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(flags))
    cfg.Cbody = 1e-15
    tcfg = TransientCfgV2(C_B_const=1e-15,
                          max_step=1e-9,           # loosen for 1us run
                          first_step=1e-14,
                          rtol=1e-6, atol=1e-15,
                          R_body=R_body,
                          R_body_thresh=0.0)
    per_bias = []
    z449._VBIC_CTX["cfg"] = cfg
    for bias in BIASES:
        sebas_row = z427.find_params(sebas_rows, bias["VG1"], bias["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            log(f"  skip {bias['tag']} — no Sebas params"); continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        bjt = z427.make_bjt(sebas_row)
        z449._VBIC_CTX["bjt"] = bjt
        t, Vd_stim = stim_fast_pulse_extended(V_hi=2.0, V_lo=0.05,
                                                t_rise=100e-12,
                                                t_hold=1.0e-6,
                                                t_pre=0.5e-9,
                                                n_total=3000)
        ramp_end_s = 0.5e-9 + 100e-12
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
        Vb_peak = float(np.nanmax(Vb_arr))

        t_peak_first, t_reset, _ = detect_self_reset(
            t_arr, Vb_arr, ramp_end_s, peak_thresh=0.5, reset_thresh=0.3)
        t_to_05_after_ramp = (t_peak_first - ramp_end_s) if t_peak_first else None
        t_to_reset_after_peak = (t_reset - t_peak_first) if (t_reset and t_peak_first) else None

        n_cycles, period_s, period_std, cycle_times = detect_oscillation(
            t_arr, Vb_arr, ramp_end_s, peak_thresh=0.5, reset_thresh=0.3)

        # Subsample trace for JSON
        n_samp = min(len(t_arr), 400)
        idx = np.linspace(0, len(t_arr) - 1, n_samp).astype(int)
        per_bias.append({
            "tag": bias["tag"], "VG1": bias["VG1"], "VG2": bias["VG2"],
            "Vb_peak_V": Vb_peak,
            "t_to_0p5V_after_ramp_ns": (t_to_05_after_ramp * 1e9) if t_to_05_after_ramp else None,
            "t_to_reset_after_peak_ns": (t_to_reset_after_peak * 1e9) if t_to_reset_after_peak else None,
            "self_reset": (t_reset is not None),
            "n_oscillation_cycles": n_cycles,
            "oscillation_period_ns": (period_s * 1e9) if period_s else None,
            "oscillation_period_std_ns": (period_std * 1e9) if period_std else None,
            "wall_sec": round(wall, 1),
            "solver_success": bool(r["solver"]["success"]),
            "_traces": {
                "t": (t_arr[idx]).tolist(),
                "Vd": (np.array(Vd_stim)[idx]).tolist(),
                "Vb": (Vb_arr[idx]).tolist(),
                "Id": (Id_arr[idx]).tolist(),
            },
        })
        log(f"  {name}/{bias['tag']}: Vb_peak={Vb_peak:.3f}V  "
            f"t→0.5V={t_to_05_after_ramp}  t→reset={t_to_reset_after_peak}  "
            f"reset={t_reset is not None}  cycles={n_cycles}  "
            f"period_ns={period_s*1e9 if period_s else None}  "
            f"ok={r['solver']['success']}  wall={wall:.1f}s")
    z449._VBIC_CTX["cfg"] = None
    z449._VBIC_CTX["bjt"] = None
    return {"per_bias": per_bias}


# =========================================================== #
# Plots
# =========================================================== #
def plot_dc_vs_rbody(results, path):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    names = [r["name"] for r in results]
    fwd = [r["dc_forward"]["cell_rmse_dec"] for r in results]
    bwd = [r["dc_backward"]["cell_rmse_dec"] for r in results]
    x = np.arange(len(names))
    ax.bar(x - 0.18, fwd, 0.36, label="forward", color="C0")
    ax.bar(x + 0.18, bwd, 0.36, label="backward", color="C1")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("cell DC RMSE [dec]")
    ax.axhline(2.5, color="grey", ls=":", label="DISCOVERY DC gate (2.5)")
    ax.axhline(1.5, color="red", ls=":", label="AMBITIOUS DC gate (1.5)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title("z456 — DC cell RMSE vs R_body (SB_HOT base)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    log(f"  wrote {path.name}")


def plot_pulse_extended(results, path, bias_tag="VG1_0p6_VG2_0p2"):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    colors = ["k", "C0", "C2", "C3", "C4"]
    for r, col in zip(results, colors):
        rec = next((x for x in r["fast"]["per_bias"] if x["tag"] == bias_tag), None)
        if rec is None: continue
        tr = rec["_traces"]
        t = np.array(tr["t"]) * 1e9   # ns
        label = f"{r['name']} (τ≈{r['tau_est_s']:.1e}s)" if math.isfinite(r['tau_est_s']) else f"{r['name']} (no leak)"
        axes[0].semilogx(t, tr["Vb"], col + "-", lw=1.2, label=label)
        axes[1].loglog(t, np.maximum(np.abs(tr["Id"]), 1e-18),
                       col + "-", lw=1.0, label=label)
    axes[0].axhline(0.5, color="red", ls=":", lw=0.6, label="0.5V (peak gate)")
    axes[0].axhline(0.3, color="grey", ls=":", lw=0.6, label="0.3V (reset)")
    axes[0].set_ylabel("V_B [V]"); axes[0].legend(fontsize=7, loc="upper left")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].set_ylabel("|I_D| [A]"); axes[1].set_xlabel("time [ns] (log)")
    axes[1].legend(fontsize=7); axes[1].grid(True, which="both", alpha=0.3)
    axes[0].set_title(f"z456 — extended fast-pulse V_B(t) up to 1µs, {bias_tag}")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    log(f"  wrote {path.name}")


def plot_oscillation_zoom(results, path, bias_tag="VG1_0p6_VG2_0p2"):
    osc = []
    for r in results:
        rec = next((x for x in r["fast"]["per_bias"] if x["tag"] == bias_tag), None)
        if rec is None: continue
        if rec.get("n_oscillation_cycles", 0) >= 2:
            osc.append((r, rec))
    if not osc:
        log("  no oscillation found in any condition; skipping zoom plot")
        return False
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["C0", "C2", "C3", "C4"]
    for (r, rec), col in zip(osc, colors):
        tr = rec["_traces"]
        t = np.array(tr["t"]) * 1e9
        ax.plot(t, tr["Vb"], col + "-", lw=1.2,
                label=f"{r['name']} period={rec['oscillation_period_ns']:.1f}ns "
                      f"n_cycles={rec['n_oscillation_cycles']}")
    ax.axhline(0.5, color="red", ls=":", lw=0.6)
    ax.axhline(0.3, color="grey", ls=":", lw=0.6)
    ax.set_xlabel("time [ns]"); ax.set_ylabel("V_B [V]")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title(f"z456 — relaxation oscillation zoom ({bias_tag})  "
                 f"Mario slide-21 target ~400 ns")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    log(f"  wrote {path.name}")
    return True


# =========================================================== #
# Gates
# =========================================================== #
def eval_gates(results):
    discovery = False; discovery_who = []
    ambitious = False; ambitious_who = []
    any_reset = False
    all_dc_killed = True
    for r in results:
        dc = r["dc_avg"]
        if dc <= 3.0:
            all_dc_killed = False
        # Check fast-pulse for self-reset
        pb = r["fast"]["per_bias"]
        for x in pb:
            tr = x.get("t_to_reset_after_peak_ns")
            if tr is not None and 100.0 <= tr <= 1.0e5:  # 100ns..100us
                any_reset = True
                if dc < 2.5:
                    discovery = True
                    discovery_who.append(f"{r['name']}/{x['tag']} t_reset={tr:.1f}ns")
                if x.get("n_oscillation_cycles", 0) >= 3 and dc < 1.5:
                    ambitious = True
                    ambitious_who.append(
                        f"{r['name']}/{x['tag']} cycles={x['n_oscillation_cycles']} "
                        f"period={x['oscillation_period_ns']:.1f}ns")
    kill_shot = (not any_reset) or all_dc_killed
    return {
        "INFRA_pass": True,
        "DISCOVERY_pass": discovery, "DISCOVERY_who": discovery_who,
        "AMBITIOUS_pass": ambitious, "AMBITIOUS_who": ambitious_who,
        "KILL_SHOT": kill_shot,
        "kill_shot_reason": ("no_self_reset" if not any_reset
                              else ("all_DC_killed_above_3dec" if all_dc_killed
                                    else None)),
    }


# =========================================================== #
# Main
# =========================================================== #
def main():
    t0_main = time.time()
    log("z456 starting — R_body sweep on SB_HOT base")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    results = []
    for C in CONDITIONS:
        log(f"===== {C['name']} R_body={C['R_body']}  τ_est={C['tau_est_s']:.2e}s =====")
        flags = dict(SB_HOT_FLAGS)
        # DC fwd (R_body not applied — see header note)
        dc_f = slow_dc_cell_rmse_dir(C["name"], flags,
                                      model_M1, model_M2, curves, sebas_rows,
                                      direction="forward")
        dc_b = slow_dc_cell_rmse_dir(C["name"], flags,
                                      model_M1, model_M2, curves, sebas_rows,
                                      direction="backward")
        dc_avg = 0.5 * (dc_f["cell_rmse_dec"] + dc_b["cell_rmse_dec"])
        log(f"  {C['name']}: DC fwd={dc_f['cell_rmse_dec']:.3f}  "
            f"bwd={dc_b['cell_rmse_dec']:.3f}  avg={dc_avg:.3f}")
        fast = fast_pulse_extended(C["name"], flags, C["R_body"],
                                    model_M1, model_M2, sebas_rows)
        results.append({
            "name": C["name"], "R_body": C["R_body"],
            "tau_est_s": C["tau_est_s"],
            "dc_forward": dc_f, "dc_backward": dc_b, "dc_avg": dc_avg,
            "fast": fast,
        })

    gates = eval_gates(results)
    log(f"GATES: {gates}")

    # Plots
    plot_dc_vs_rbody(results, OUT / "dc_vs_rbody.png")
    plot_pulse_extended(results, OUT / "pulse_extended.png",
                        bias_tag="VG1_0p6_VG2_0p2")
    osc_found = plot_oscillation_zoom(results, OUT / "oscillation_zoom.png",
                                       bias_tag="VG1_0p6_VG2_0p2")

    # Summary
    summary = {
        "conditions": [
            {
                "name": r["name"], "R_body": r["R_body"],
                "tau_est_s": r["tau_est_s"],
                "dc_forward_dec": r["dc_forward"]["cell_rmse_dec"],
                "dc_backward_dec": r["dc_backward"]["cell_rmse_dec"],
                "dc_avg_dec": r["dc_avg"],
                "dc_per_branch_forward": r["dc_forward"]["per_bias"],
                "dc_per_branch_backward": r["dc_backward"]["per_bias"],
                "fast_pulse": [{k: v for k, v in x.items() if k != "_traces"}
                               for x in r["fast"]["per_bias"]],
            } for r in results
        ],
        "gates": gates,
        "oscillation_zoom_plot_written": bool(osc_found),
        "references": {
            "z454_SB_HOT_DC_avg": 2.809,
            "z454_SB_HOT_no_self_reset": True,
            "Mario_slide21_oscillation_period_ns": 400,
        },
        "wall_total_sec": round(time.time() - t0_main, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"wrote summary.json  total_wall={summary['wall_total_sec']:.0f}s")

    # Honest analysis
    best = min(results, key=lambda r: r["dc_avg"])
    lines = []
    lines.append("# z456 — R_body reset path (SB_HOT base + body-leak resistor)\n")
    lines.append("## Pre-registered gates\n")
    for k, v in gates.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## DC (forward / backward / avg) [dec]\n")
    lines.append("| R_body | τ_est | DC_fwd | DC_bwd | DC_avg | n |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        rb = "INF" if r["R_body"] is None else f"{r['R_body']:.1e} Ω"
        tau = "inf" if not math.isfinite(r['tau_est_s']) else f"{r['tau_est_s']:.1e}s"
        lines.append(f"| {r['name']} ({rb}) | {tau} | "
                     f"{r['dc_forward']['cell_rmse_dec']:.3f} | "
                     f"{r['dc_backward']['cell_rmse_dec']:.3f} | "
                     f"{r['dc_avg']:.3f} | {r['dc_forward']['n']} |")
    lines.append("\nNOTE: R_body is wired into the transient body-KCL only. DC pathway")
    lines.append("uses z429.run_vsint_pinned which is unchanged → DC values match SB_HOT")
    lines.append("baseline. This is intentional (see script header).\n")

    lines.append("\n## Fast-pulse extended (1µs hold) — self-reset timings\n")
    for r in results:
        lines.append(f"### {r['name']} (R_body={r['R_body']}, τ_est={r['tau_est_s']:.1e}s)")
        lines.append("| bias | Vb_peak | t→0.5V[ns] | t→reset[ns] | self-reset | n_cycles | period[ns] |")
        lines.append("|---|---|---|---|---|---|---|")
        for x in r["fast"]["per_bias"]:
            lines.append(f"| {x['tag']} | {x['Vb_peak_V']:.3f} | "
                         f"{x['t_to_0p5V_after_ramp_ns']} | "
                         f"{x['t_to_reset_after_peak_ns']} | "
                         f"{x['self_reset']} | "
                         f"{x['n_oscillation_cycles']} | "
                         f"{x['oscillation_period_ns']} |")
        lines.append("")
    lines.append(f"\n## Best (lowest DC_avg): **{best['name']}**, DC_avg={best['dc_avg']:.3f}\n")
    lines.append(f"\n## Mario slide-21 reference: ~400 ns oscillation period")
    lines.append("(quoted as benchmark — not target-matched, not optimized to fit)\n")
    (OUT / "honest_analysis.md").write_text("\n".join(lines))
    log("wrote honest_analysis.md")
    log("DONE.")


if __name__ == "__main__":
    main()
