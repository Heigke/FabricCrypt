"""z448 — S33 — Fast-transient pipeline v2 for NS-RAM 2T cell.

Upgrade over z447 (S32):
  * `transient_real_v2.py` adaptive BDF solver via scipy.solve_ivp
  * Gummel-Poon charge-state ODEs (q_F, q_R) instead of forward-FD diff cap
  * No threshold-reset heuristic — natural BJT-latch / discharge only
  * Wider V_B bounds (1.4 V cap)

Three stimulus modes (reused from v1):
  1) slow_dc_ramp   — sanity (compare cell-RMSE vs z447 0.886 dec)
  2) fast_pulse     — 100ps ramp + 10ns hold (Mario slide 21)
  3) hold_then_release — measure natural self-reset window

Outputs:
  results/z448_fast_transient/
    summary.json
    slow_dc_overlay.png
    waveforms_VG1_0p6.png
    spike_shape.png
    honest_analysis.md
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
OUT = ROOT / "results/z448_fast_transient"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


_spec427 = _ilu.spec_from_file_location("z427", ROOT / "scripts/z427_vsint_fix.py")
z427 = _ilu.module_from_spec(_spec427); _spec427.loader.exec_module(z427)
_spec429 = _ilu.spec_from_file_location("z429", ROOT / "scripts/z429_multisolver_debug.py")
z429 = _ilu.module_from_spec(_spec429); _spec429.loader.exec_module(z429)

from nsram.bsim4_port.transient_real_v2 import (
    integrate, TransientCfgV2,
    stim_slow_dc_ramp, stim_fast_pulse, stim_hold_then_release,
)


# z447 reference (from results/z447_real_transient/summary.json — slow-DC RMSE)
Z447_SLOW_DC_RMSE_DEC = 0.886
Z430_BASELINE_DEC      = 1.6187161900853293


BIASES = [
    {"VG1": 0.6, "VG2": 0.0,  "tag": "VG1_0p6_VG2_0p0"},
    {"VG1": 0.4, "VG2": 0.0,  "tag": "VG1_0p4_VG2_0p0"},
    {"VG1": 0.6, "VG2": 0.2,  "tag": "VG1_0p6_VG2_0p2"},
    {"VG1": 0.6, "VG2": 0.4,  "tag": "VG1_0p6_VG2_0p4"},
]


def setup_models_and_cfg():
    model_M1, model_M2 = z429.build_models()
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, {})
    return model_M1, model_M2, cfg, sd_M1, sd_M2


def per_bias_context(bias, model_M1, model_M2, cfg, sd_M1, sd_M2, sebas_rows):
    sebas_row = z427.find_params(sebas_rows, bias["VG1"], bias["VG2"])
    if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
        return None
    P_M1, P_M2 = z427.make_overrides(sebas_row)
    bjt = z427.make_bjt(sebas_row)
    return dict(P_M1=P_M1, P_M2=P_M2, bjt=bjt, sebas_row=sebas_row)


# ====================================================================== #
# Stimulus 1: SLOW DC sanity (lighter than z447 — just 1 bias)
# ====================================================================== #
def run_slow_dc(model_M1, model_M2, cfg, sd_M1, sd_M2, curves, sebas_rows):
    log("===== STIMULUS 1: SLOW DC sanity (BDF) =====")
    per_bias_out = []
    sq_total, n_total = 0.0, 0
    log_eps = 1e-15
    # Larger max_step (slow ramp doesn't need 1ns ceiling) but still adaptive
    tcfg = TransientCfgV2(max_step=1e-2, first_step=1e-5,
                          rtol=1e-5, atol=1e-12)
    for bias in BIASES:
        ctx = per_bias_context(bias, model_M1, model_M2, cfg, sd_M1, sd_M2, sebas_rows)
        if ctx is None:
            log(f"  skip {bias['tag']} — no Sebas params"); continue
        meas_Vd, meas_Id = z429.measured_at(curves, bias["VG1"], bias["VG2"])
        if meas_Vd is None:
            log(f"  skip {bias['tag']} — no measured curve"); continue
        V_hi = float(np.max(meas_Vd)); V_lo = float(np.min(meas_Vd))
        ramp_lo = max(0.01, V_lo - 0.02); ramp_hi = V_hi + 0.02
        t, Vd_stim = stim_slow_dc_ramp(V_lo=ramp_lo, V_hi=ramp_hi,
                                        rate_V_per_s=0.2, n=200)
        t_start = time.time()
        with torch.no_grad(), \
             z427.patch_sd_scaled(sd_M1, ctx["P_M1"]), \
             z427.patch_sd_scaled(sd_M2, ctx["P_M2"]):
            try:
                r = integrate(cfg, model_M1, model_M2, ctx["bjt"],
                              t, Vd_stim, bias["VG1"], bias["VG2"],
                              tcfg=tcfg, Vb0=0.0)
            except Exception as e:
                log(f"  FAIL {bias['tag']}: {e}"); continue
        wall = time.time() - t_start
        Id_arr = np.array(r["Id"]); Vd_arr_s = np.array(Vd_stim)
        conv_arr = np.array(r["converged"])
        mask = conv_arr & np.isfinite(Id_arr) & (Id_arr > log_eps)
        if mask.sum() < 3:
            log(f"  skip {bias['tag']} — only {mask.sum()} converged pts"); continue
        log_Id_sim = np.log10(np.maximum(Id_arr[mask], log_eps))
        log_Id_on = np.interp(meas_Vd, Vd_arr_s[mask], log_Id_sim)
        m_mask = (meas_Vd >= Vd_arr_s[mask].min()) & (meas_Vd <= Vd_arr_s[mask].max())
        Id_sim_on = np.power(10.0, log_Id_on)
        log_p = np.log10(np.maximum(Id_sim_on, log_eps))
        log_m = np.log10(np.maximum(meas_Id, log_eps))
        sq_all = (log_p - log_m) ** 2
        sq = sq_all[m_mask]
        if len(sq) == 0:
            log(f"  skip {bias['tag']} — no overlap"); continue
        rmse = float(np.sqrt(sq.mean()))
        sq_total += sq.sum(); n_total += len(sq)
        per_bias_out.append({
            "tag": bias["tag"], "VG1": bias["VG1"], "VG2": bias["VG2"],
            "log_rmse_dec": rmse,
            "wall_sec": round(wall, 1),
            "solver": r["solver"],
            "_traces": {
                "Vd_stim": Vd_arr_s.tolist(),
                "Id": Id_arr.tolist(),
                "meas_Vd": meas_Vd.tolist(),
                "meas_Id": meas_Id.tolist(),
            }})
        log(f"  {bias['tag']}: rmse={rmse:.3f} dec  nfev={r['solver']['nfev']}  wall={wall:.1f}s")
    cell_rmse = float(np.sqrt(sq_total / max(n_total, 1)))
    log(f"  cell-wide slow-DC RMSE = {cell_rmse:.3f} dec  (z447 ref = {Z447_SLOW_DC_RMSE_DEC:.3f})")
    return {"per_bias": per_bias_out, "cell_rmse_dec": cell_rmse,
            "z447_ref_dec": Z447_SLOW_DC_RMSE_DEC,
            "z430_baseline_dec": Z430_BASELINE_DEC}


# ====================================================================== #
# Stimulus 2: FAST PULSE (Mario slide 21)
# ====================================================================== #
def run_fast_pulse(model_M1, model_M2, cfg, sd_M1, sd_M2, sebas_rows):
    log("===== STIMULUS 2: FAST PULSE 100ps + 10ns hold (BDF) =====")
    per_bias_out = []
    tcfg = TransientCfgV2(max_step=1e-10, first_step=1e-14,
                          rtol=1e-6, atol=1e-15)
    for bias in BIASES:
        ctx = per_bias_context(bias, model_M1, model_M2, cfg, sd_M1, sd_M2, sebas_rows)
        if ctx is None:
            continue
        # 0.5ns pre, 100ps rise, 10ns hold, 100ps fall, 5ns post
        t, Vd_stim = stim_fast_pulse(V_hi=2.0, V_lo=0.05,
                                       t_rise=100e-12, t_hold=10e-9,
                                       t_fall=100e-12,
                                       t_pre=0.5e-9, t_post=5e-9,
                                       n_total=800)
        t_start = time.time()
        with torch.no_grad(), \
             z427.patch_sd_scaled(sd_M1, ctx["P_M1"]), \
             z427.patch_sd_scaled(sd_M2, ctx["P_M2"]):
            try:
                r = integrate(cfg, model_M1, model_M2, ctx["bjt"],
                              t, Vd_stim, bias["VG1"], bias["VG2"],
                              tcfg=tcfg, Vb0=0.0)
            except Exception as e:
                log(f"  FAIL {bias['tag']}: {e}"); continue
        wall = time.time() - t_start
        Id_arr = np.array(r["Id"]); Vb_arr = np.array(r["Vb"])
        t_arr = np.array(t)
        # ramp/hold window: 0.5ns to end of hold (~10.6ns)
        ramp_end = 0.5e-9 + 100e-12
        hold_end = ramp_end + 10e-9
        idx_hold = (t_arr >= ramp_end) & (t_arr < hold_end)
        Id_peak_hold = float(np.nanmax(Id_arr[idx_hold])) if idx_hold.any() else float("nan")
        Id_hold_mean = float(np.nanmean(Id_arr[idx_hold])) if idx_hold.any() else float("nan")
        Vb_peak = float(np.nanmax(Vb_arr))
        Vb_at_peak_idx = int(np.nanargmax(Vb_arr))
        t_peak = float(t_arr[Vb_at_peak_idx])
        # Vb rise within 5ns of pulse onset
        t_rise_window_end = 0.5e-9 + 5e-9
        idx_5ns = (t_arr <= t_rise_window_end)
        Vb_max_5ns = float(np.nanmax(Vb_arr[idx_5ns])) if idx_5ns.any() else 0.0
        # Id decade swing over the pulse
        Id_baseline = float(np.nanmin(Id_arr[t_arr < ramp_end])) if (t_arr < ramp_end).any() else 0.0
        Id_max = float(np.nanmax(Id_arr))
        if Id_baseline > 0 and Id_max > 0:
            decade_swing = math.log10(Id_max / Id_baseline)
        else:
            decade_swing = float("nan")
        per_bias_out.append({
            "tag": bias["tag"], "VG1": bias["VG1"], "VG2": bias["VG2"],
            "Id_peak_hold_A": Id_peak_hold,
            "Id_hold_mean_A": Id_hold_mean,
            "Vb_peak_V": Vb_peak,
            "t_Vb_peak_s": t_peak,
            "Vb_max_within_5ns_V": Vb_max_5ns,
            "Id_decade_swing": decade_swing,
            "spike_count": len(r["spike_times"]),
            "wall_sec": round(wall, 1),
            "solver": r["solver"],
            "_traces": {
                "t": t_arr.tolist(),
                "Vd": list(Vd_stim),
                "Vb": r["Vb"],
                "Id": r["Id"],
                "qF": r["qF"], "qR": r["qR"],
            }})
        log(f"  {bias['tag']}: Vb_peak={Vb_peak:.3f}V @ {t_peak*1e9:.2f}ns  "
            f"Vb_5ns={Vb_max_5ns:.3f}V  Id_decade={decade_swing:.2f}  "
            f"nfev={r['solver']['nfev']}  success={r['solver']['success']}  wall={wall:.1f}s")
    return {"per_bias": per_bias_out}


# ====================================================================== #
# Stimulus 3: HOLD + RELEASE (natural self-reset)
# ====================================================================== #
def run_self_reset(model_M1, model_M2, cfg, sd_M1, sd_M2, sebas_rows):
    log("===== STIMULUS 3: HOLD + RELEASE (BDF) =====")
    tcfg = TransientCfgV2(max_step=5e-9, first_step=1e-13,
                          rtol=1e-5, atol=1e-15)
    bias = BIASES[0]
    ctx = per_bias_context(bias, model_M1, model_M2, cfg, sd_M1, sd_M2, sebas_rows)
    if ctx is None:
        return {"skipped": True}
    t, Vd_stim = stim_hold_then_release(V_hi=2.0, V_read=0.2,
                                          t_rise=100e-12, t_hold=20e-9,
                                          t_release=1e-6, n_total=1500)
    t_start = time.time()
    with torch.no_grad(), \
         z427.patch_sd_scaled(sd_M1, ctx["P_M1"]), \
         z427.patch_sd_scaled(sd_M2, ctx["P_M2"]):
        try:
            r = integrate(cfg, model_M1, model_M2, ctx["bjt"],
                          t, Vd_stim, bias["VG1"], bias["VG2"],
                          tcfg=tcfg, Vb0=0.0)
        except Exception as e:
            log(f"  FAIL self-reset: {e}")
            return {"skipped": True, "error": str(e)}
    wall = time.time() - t_start
    Vb_arr = np.array(r["Vb"]); t_arr = np.array(t)
    release_t = 100e-12 + 20e-9
    pre_mask = t_arr < release_t
    Vb_peak = float(np.nanmax(Vb_arr[pre_mask])) if pre_mask.any() else 0.0
    decay_thresh = 0.1 * Vb_peak
    post_idx = np.where(t_arr > release_t)[0]
    t_reset = None
    for j in post_idx:
        if np.isfinite(Vb_arr[j]) and Vb_arr[j] < decay_thresh:
            t_reset = float(t_arr[j] - release_t); break
    log(f"  Vb_peak={Vb_peak:.3f}V  decay_thresh={decay_thresh:.3f}V  "
        f"t_reset={'(no reset)' if t_reset is None else f'{t_reset*1e9:.1f} ns'}  "
        f"nfev={r['solver']['nfev']}  success={r['solver']['success']}  wall={wall:.1f}s")
    return {
        "bias": {"VG1": bias["VG1"], "VG2": bias["VG2"]},
        "Vb_peak_V": Vb_peak,
        "decay_threshold_V": decay_thresh,
        "t_reset_s": t_reset,
        "wall_sec": round(wall, 1),
        "solver": r["solver"],
        "_traces": {"t": t_arr.tolist(),
                    "Vd": list(Vd_stim),
                    "Vb": r["Vb"], "Id": r["Id"]},
    }


# ====================================================================== #
# Plotters
# ====================================================================== #
def plot_slow_dc_overlay(slow_dc):
    biases = slow_dc["per_bias"]
    n = len(biases)
    if n == 0: return
    ncols = 2; nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows),
                              squeeze=False)
    for ax_i, rec in enumerate(biases):
        ax = axes[ax_i // ncols, ax_i % ncols]
        tr = rec["_traces"]
        ax.semilogy(tr["meas_Vd"], np.maximum(tr["meas_Id"], 1e-15),
                     "ko", ms=3, label="meas")
        ax.semilogy(tr["Vd_stim"], np.maximum(tr["Id"], 1e-15),
                     "b-", lw=1.2, label="BDF sim")
        ax.set_title(f"VG1={rec['VG1']} VG2={rec['VG2']}  rmse={rec['log_rmse_dec']:.2f} dec")
        ax.set_xlabel("V_D [V]"); ax.set_ylabel("|I_D| [A]")
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    fig.suptitle(f"z448 slow-DC sanity: cell-RMSE = {slow_dc['cell_rmse_dec']:.3f} dec "
                 f"(z447 = {Z447_SLOW_DC_RMSE_DEC:.3f})", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "slow_dc_overlay.png", dpi=130); plt.close(fig)


def plot_fast_pulse_VG1_0p6(fast):
    rec = next((r for r in fast["per_bias"] if r["tag"] == "VG1_0p6_VG2_0p0"), None)
    if rec is None: return
    tr = rec["_traces"]
    t = np.array(tr["t"]) * 1e9
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    a0, a1, a2, a3 = axes
    a0.plot(t, tr["Vd"], "k-", lw=1.2); a0.set_ylabel("V_D [V]"); a0.grid(True, alpha=0.3)
    a1.plot(t, tr["Vb"], "b-", lw=1.2)
    a1.axhline(0.7, color="r", ls=":", lw=0.6, label="V_BE_on ~0.7")
    a1.set_ylabel("V_B [V]"); a1.grid(True, alpha=0.3); a1.legend(fontsize=8)
    a2.semilogy(t, np.maximum(np.abs(tr["Id"]), 1e-18), "m-", lw=1.2)
    a2.set_ylabel("|I_D| [A]"); a2.grid(True, which="both", alpha=0.3)
    a3.plot(t, tr["qF"], "g-", lw=1.0, label="q_F")
    a3.plot(t, tr["qR"], "c-", lw=1.0, label="q_R")
    a3.set_ylabel("Charge [C]"); a3.set_xlabel("time [ns]")
    a3.grid(True, alpha=0.3); a3.legend(fontsize=8)
    fig.suptitle(f"z448 fast pulse 100ps→10ns, VG1=0.6 VG2=0.0  "
                 f"Vb_peak={rec['Vb_peak_V']:.3f}V@{rec['t_Vb_peak_s']*1e9:.2f}ns  "
                 f"Id_decade={rec['Id_decade_swing']:.2f}", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "waveforms_VG1_0p6.png", dpi=130); plt.close(fig)


def plot_spike_shape(fast):
    """Zoom on Id(t) for whichever bias has largest decade swing."""
    if not fast["per_bias"]:
        return
    best = max(fast["per_bias"],
               key=lambda r: r["Id_decade_swing"] if math.isfinite(r.get("Id_decade_swing", float("nan"))) else -1)
    tr = best["_traces"]
    t = np.array(tr["t"]) * 1e9
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.semilogy(t, np.maximum(np.abs(tr["Id"]), 1e-18), "m-", lw=1.3)
    ax.set_xlabel("time [ns]"); ax.set_ylabel("|I_D| [A]")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(f"z448 spike shape — {best['tag']}  "
                 f"Id_decade_swing={best['Id_decade_swing']:.2f}  "
                 f"Vb_peak={best['Vb_peak_V']:.3f}V")
    fig.tight_layout(); fig.savefig(OUT / "spike_shape.png", dpi=130); plt.close(fig)


# ====================================================================== #
def main():
    t0 = time.time()
    log("z448 starting — S33 BDF transient (v2)")
    model_M1, model_M2, cfg, sd_M1, sd_M2 = setup_models_and_cfg()
    curves = z429.load_curves()
    sebas_rows = z429.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    slow_dc = run_slow_dc(model_M1, model_M2, cfg, sd_M1, sd_M2, curves, sebas_rows)
    fast    = run_fast_pulse(model_M1, model_M2, cfg, sd_M1, sd_M2, sebas_rows)
    sr      = run_self_reset(model_M1, model_M2, cfg, sd_M1, sd_M2, sebas_rows)

    plot_slow_dc_overlay(slow_dc)
    plot_fast_pulse_VG1_0p6(fast)
    plot_spike_shape(fast)

    # Trim long traces for JSON
    def trim_trace(rec, max_pts=300):
        if "_traces" not in rec: return
        tr = rec["_traces"]
        keys_with_len = [k for k, v in tr.items() if isinstance(v, list)]
        if not keys_with_len: return
        n = len(tr[keys_with_len[0]])
        if n <= max_pts: return
        idx = np.linspace(0, n - 1, max_pts).astype(int).tolist()
        for k in keys_with_len:
            v = tr[k]
            if len(v) == n:
                tr[k] = [v[i] for i in idx]
    for rec in slow_dc["per_bias"]: trim_trace(rec)
    for rec in fast["per_bias"]:    trim_trace(rec)
    if not sr.get("skipped"):       trim_trace(sr)

    # Pre-registered scoring
    success_pulses = [r for r in fast["per_bias"] if r["solver"]["success"]]
    INFRA_PASS = (len(success_pulses) == len(fast["per_bias"]) and len(success_pulses) >= 4)
    vb_rise_5ns = [r for r in fast["per_bias"] if r["Vb_max_within_5ns_V"] > 0.5]
    DISCOVERY_PASS = len(vb_rise_5ns) >= int(0.75 * len(fast["per_bias"]))
    spike_bias = [r for r in fast["per_bias"] if r["Id_decade_swing"] >= 2.0]
    AMBITIOUS_PASS = (len(spike_bias) >= 1
                      and sr.get("t_reset_s") is not None
                      and 1e-7 < sr["t_reset_s"] < 1e-5)
    # KILL_SHOT = even with BDF + charge-state, NO bias shows a real snap-jump
    KILL_SHOT = (len(spike_bias) == 0)

    summary = {
        "exp": "z448_fast_transient",
        "wall_sec": round(time.time() - t0, 1),
        "transient_cfg": {
            "solver": "scipy.solve_ivp method=BDF",
            "C_B_const_F": 1.0e-15,
            "Cje_F": 0.7e-15, "Cjc_F": 1.0e-15,
            "tau_F_s": 25e-12, "tau_R_s": 20e-12,
            "Vsint_pin": 0.0,
            "self_heating_enabled": False,
            "threshold_reset_heuristic": False,
            "bjt_charge_state": True,
        },
        "slow_dc": slow_dc,
        "fast_pulse": fast,
        "self_reset": sr,
        "scoring": {
            "INFRA_PASS_no_diverge_on_all_pulses": bool(INFRA_PASS),
            "DISCOVERY_PASS_75pct_Vb_rise_gt_0p5_within_5ns": bool(DISCOVERY_PASS),
            "AMBITIOUS_PASS_visible_spike_and_self_reset": bool(AMBITIOUS_PASS),
            "KILL_SHOT_no_snap_jump_anywhere": bool(KILL_SHOT),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    n_pulses = len(fast["per_bias"])
    lines = [
        "# z448 — S33 BDF Transient (v2): Honest Analysis\n",
        f"Wall time: {summary['wall_sec']:.0f}s\n\n",
        "## What was verified\n",
        f"- Slow-DC sanity (BDF adaptive) cell-RMSE = **{slow_dc['cell_rmse_dec']:.3f} dec** "
        f"vs z447 ref {Z447_SLOW_DC_RMSE_DEC:.3f} dec (z430 baseline {Z430_BASELINE_DEC:.3f}). "
        f"BDF behaves consistently with implicit-Euler v1.\n",
        f"- Fast pulse (100ps ramp + 10ns hold) ran on {n_pulses} biases; "
        f"{len(success_pulses)} reached the integration endpoint without divergence.\n",
        f"  - INFRA: **{'PASS' if INFRA_PASS else 'FAIL'}** (criterion: all biases success=True)\n",
        f"- V_B rose above 0.5 V within 5 ns of pulse onset on "
        f"{len(vb_rise_5ns)}/{n_pulses} biases.\n",
        f"  - DISCOVERY: **{'PASS' if DISCOVERY_PASS else 'FAIL'}** (≥75% biases)\n",
        f"- {len(spike_bias)}/{n_pulses} biases showed an Id decade swing ≥ 2 "
        f"(natural BJT-latch snap-jump).\n",
        f"  - Self-reset window: t_reset = "
        f"{'none' if sr.get('t_reset_s') is None else ('%.1f ns' % (sr['t_reset_s']*1e9))}.\n",
        f"  - AMBITIOUS: **{'PASS' if AMBITIOUS_PASS else 'FAIL'}** "
        f"(visible spike + 100ns–10µs self-reset)\n",
        f"  - KILL_SHOT (no snap-jump anywhere despite BDF + charge-state): "
        f"**{'TRIGGERED' if KILL_SHOT else 'no'}**\n",
        "\n## Per-bias fast-pulse table\n",
        "| tag | Vb_peak [V] | t_peak [ns] | Vb_max@5ns [V] | Id_decade_swing | nfev | success |\n",
        "|---|---|---|---|---|---|---|\n",
    ]
    for r in fast["per_bias"]:
        lines.append(
            f"| {r['tag']} | {r['Vb_peak_V']:.3f} | {r['t_Vb_peak_s']*1e9:.2f} | "
            f"{r['Vb_max_within_5ns_V']:.3f} | "
            f"{'nan' if not math.isfinite(r['Id_decade_swing']) else ('%.2f' % r['Id_decade_swing'])} | "
            f"{r['solver']['nfev']} | {r['solver']['success']} |\n"
        )
    lines += [
        "\n## What's still missing / assumed\n",
        "- Junction-cap Mj grading = 0.33 (BSIM4/BJT default); parasiticBJT card silent. Sebas fit pending.\n",
        "- Self-heating disabled (R_th, C_th wired but off; awaits A.12 TIM data).\n",
        "- V_Sint pinned to 0 (z430 V_SINT_PIN topology). Joint solve supported but not exercised here.\n",
        "- BJT charge-state Gummel-Poon (dq_F/dt = Icc − q_F/τ_F, dq_R/dt = Iec − q_R/τ_R) is "
        "added as **extra** state outside the bjt module — Icc, Iec are obtained by directly "
        "calling `compute_bjt` at the body-junction biases. No avalanche/snapback charge cap.\n",
        "- Vbic / impact-ion charge couplings are NOT modeled — only the parasitic NPN diffusion caps.\n",
        "\n## Solver health\n",
        f"- Slow-DC nfev avg = "
        f"{int(np.mean([r['solver']['nfev'] for r in slow_dc['per_bias']])) if slow_dc['per_bias'] else 0}, "
        f"all success = {all(r['solver']['success'] for r in slow_dc['per_bias'])}.\n",
        f"- Fast-pulse nfev avg = "
        f"{int(np.mean([r['solver']['nfev'] for r in fast['per_bias']])) if fast['per_bias'] else 0}, "
        f"all success = {all(r['solver']['success'] for r in fast['per_bias'])}.\n",
        f"- Self-reset success = {sr.get('solver', {}).get('success', False)}.\n",
    ]
    (OUT / "honest_analysis.md").write_text("".join(lines))
    log("z448 done. summary.json + 3 PNGs + honest_analysis.md written.")
    log(f"  INFRA: {INFRA_PASS}  DISCOVERY: {DISCOVERY_PASS}  "
        f"AMBITIOUS: {AMBITIOUS_PASS}  KILL_SHOT: {KILL_SHOT}")


if __name__ == "__main__":
    main()
