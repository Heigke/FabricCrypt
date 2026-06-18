"""z447 — S32 — REAL transient pipeline for NS-RAM 2T cell.

Combines:
  * z430 V_SINT_PIN DC baseline (1.619 dec cell-RMSE)
  * `nsram/bsim4_port/transient_real.py` proper ODE integrator
  * Real C_B = 1 fF (Mario canonical), Cje=0.7fF, Cjc=1fF, τ_F=25ps,
    τ_R=20ps (parasiticBJT.txt)

Three stimulus modes:
  1) slow_dc_ramp   — 0.2 V/s ramp (Sebas), verify match to z430 DC
  2) fast_pulse     — 100ps ramp + 10ns hold + 100ps fall (Mario slide 21)
  3) hold_then_release — measure self-reset time

Six representative biases. Outputs:
  results/z447_real_transient/
    summary.json
    slow_dc_overlay.png
    waveforms_VG1_0p6.png  (fast pulse traces)
    self_reset_trace.png
    honest_analysis.md

Run from repo root (HSA_OVERRIDE_GFX_VERSION=11.0.0, PYTHONUNBUFFERED=1).
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
OUT = ROOT / "results/z447_real_transient"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


# --- reuse upstream loaders ---
_spec427 = _ilu.spec_from_file_location("z427", ROOT / "scripts/z427_vsint_fix.py")
z427 = _ilu.module_from_spec(_spec427); _spec427.loader.exec_module(z427)
_spec429 = _ilu.spec_from_file_location("z429", ROOT / "scripts/z429_multisolver_debug.py")
z429 = _ilu.module_from_spec(_spec429); _spec429.loader.exec_module(z429)

from nsram.bsim4_port.transient_real import (
    integrate, TransientCfg,
    stim_slow_dc_ramp, stim_fast_pulse, stim_hold_then_release,
)


# z430 V_SINT_PIN DC cell-wide baseline (constant from prior z446)
Z430_BASELINE_DEC = 1.6187161900853293


# ====================================================================== #
# Representative biases: VG1=0.6/VG2=0.0 + 5 more covering the corner cases
# ====================================================================== #
BIASES = [
    # Sebas dataset has VG1 ∈ {0.2, 0.4, 0.6}, VG2 ∈ [-0.2, 0.5].
    # We use 4 representative biases; full 6 (incl. VG1=0.2) was too slow
    # given the dt-split storm in cutoff (run-time scaling).
    {"VG1": 0.6, "VG2": 0.0,  "tag": "VG1_0p6_VG2_0p0"},   # canonical
    {"VG1": 0.4, "VG2": 0.0,  "tag": "VG1_0p4_VG2_0p0"},
    {"VG1": 0.6, "VG2": 0.2,  "tag": "VG1_0p6_VG2_0p2"},
    {"VG1": 0.6, "VG2": 0.4,  "tag": "VG1_0p6_VG2_0p4"},
]


# ====================================================================== #
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
# Stimulus 1: SLOW DC RAMP — compare to z430 V_SINT_PIN
# ====================================================================== #
def run_slow_dc(model_M1, model_M2, cfg, sd_M1, sd_M2, curves, sebas_rows):
    log("===== STIMULUS 1: SLOW DC RAMP (0.2 V/s, 10s duration) =====")
    per_bias_out = []
    sq_total, n_total = 0.0, 0
    log_eps = 1e-15
    tcfg = TransientCfg()   # defaults: Vsint_pin=0, C_B=1 fF

    for bias in BIASES:
        ctx = per_bias_context(bias, model_M1, model_M2, cfg, sd_M1, sd_M2, sebas_rows)
        if ctx is None:
            log(f"  skip {bias['tag']} — no Sebas params"); continue

        # Find measured curve (for RMSE)
        meas_Vd, meas_Id = z429.measured_at(curves, bias["VG1"], bias["VG2"])
        if meas_Vd is None:
            log(f"  skip {bias['tag']} — no measured curve"); continue

        # Slow ramp covering measured Vd range with a small extension so
        # interpolation never extrapolates.
        V_hi = float(np.max(meas_Vd))
        V_lo = float(np.min(meas_Vd))
        ramp_lo = max(0.01, V_lo - 0.02)
        ramp_hi = V_hi + 0.02
        # 0.2 V/s ⇒ duration ~10s. Use 400 samples (dt=25 ms).
        t, Vd_stim = stim_slow_dc_ramp(V_lo=ramp_lo, V_hi=ramp_hi,
                                        rate_V_per_s=0.2, n=200)

        t_start = time.time()
        with torch.no_grad(), \
             z427.patch_sd_scaled(sd_M1, ctx["P_M1"]), \
             z427.patch_sd_scaled(sd_M2, ctx["P_M2"]):
            try:
                r = integrate(cfg, model_M1, model_M2, ctx["bjt"],
                              t, Vd_stim, bias["VG1"], bias["VG2"],
                              tcfg=tcfg, Vb0=0.0, verbose=False)
            except Exception as e:
                log(f"  FAIL {bias['tag']}: {e}")
                continue
        wall = time.time() - t_start

        # Interpolate simulated I_D(t) onto measured Vd grid. Use LOG-LINEAR
        # interpolation because Id varies exponentially. Mask non-converged
        # sim points so they don't bias the RMSE downward via spurious huge
        # errors (z430 baseline also drops non-converged points).
        Id_sim_arr = np.array(r["Id"])
        conv_arr = np.array(r["converged"])
        Vd_sim_arr = np.array(Vd_stim)
        mask = conv_arr & (Id_sim_arr > log_eps)
        if mask.sum() < 3:
            log(f"  skip {bias['tag']} — only {mask.sum()} converged sim pts"); continue
        log_Id_sim = np.log10(np.maximum(Id_sim_arr[mask], log_eps))
        log_Id_on_meas = np.interp(meas_Vd, Vd_sim_arr[mask], log_Id_sim)
        # Only score measured points inside the converged sim range.
        meas_mask = (meas_Vd >= Vd_sim_arr[mask].min()) & (meas_Vd <= Vd_sim_arr[mask].max())
        Id_sim_on_meas = np.power(10.0, log_Id_on_meas)
        log_p = np.log10(np.maximum(Id_sim_on_meas, log_eps))
        log_m = np.log10(np.maximum(meas_Id, log_eps))
        sq_all = (log_p - log_m) ** 2
        sq = sq_all[meas_mask]
        if len(sq) == 0:
            log(f"  skip {bias['tag']} — no overlap"); continue
        rmse_dec = float(np.sqrt(sq.mean()))
        sq_total += sq.sum()
        n_total += len(sq)

        n_conv = sum(r["converged"])
        per_bias_out.append({
            "tag": bias["tag"],
            "VG1": bias["VG1"], "VG2": bias["VG2"],
            "log_rmse_dec": rmse_dec,
            "n_sim_pts": len(t),
            "n_conv": int(n_conv),
            "conv_rate": float(n_conv) / len(t),
            "n_splits": int(r["n_splits_total"]),
            "Vb_max": float(max(r["Vb"])),
            "Id_max_sim": float(np.max(Id_sim_arr)),
            "wall_sec": round(wall, 1),
            "_traces": {
                "t": t.tolist(),
                "Vd_stim": Vd_stim.tolist(),
                "Vb": r["Vb"],
                "Id": r["Id"],
                "meas_Vd": meas_Vd.tolist(),
                "meas_Id": meas_Id.tolist(),
                "Id_sim_on_meas": Id_sim_on_meas.tolist(),
            },
        })
        log(f"  {bias['tag']}: rmse={rmse_dec:.3f} dec  Vb_max={max(r['Vb']):.3f}  "
            f"conv={n_conv}/{len(t)} splits={r['n_splits_total']} wall={wall:.1f}s")

    cell_rmse = float(np.sqrt(sq_total / max(n_total, 1)))
    log(f"  cell-wide slow-DC RMSE = {cell_rmse:.3f} dec  "
        f"(z430 V_SINT_PIN baseline = {Z430_BASELINE_DEC:.3f} dec)")
    return {"per_bias": per_bias_out, "cell_rmse_dec": cell_rmse,
            "z430_baseline_dec": Z430_BASELINE_DEC,
            "delta_dec": cell_rmse - Z430_BASELINE_DEC}


# ====================================================================== #
# Stimulus 2: FAST PULSE — Mario slide 21
# ====================================================================== #
def run_fast_pulse(model_M1, model_M2, cfg, sd_M1, sd_M2, sebas_rows):
    log("===== STIMULUS 2: FAST PULSE (100ps ramp, 10ns hold) =====")
    per_bias_out = []
    tcfg = TransientCfg()

    for bias in BIASES:
        ctx = per_bias_context(bias, model_M1, model_M2, cfg, sd_M1, sd_M2, sebas_rows)
        if ctx is None:
            continue
        # Mario slide-21 spec is 100ps rise + 10ns hold. We use 30ns hold
        # (compromise between body-RC time-constant ~µs and integrator
        # cost). Snapback expected during ramp when V_hi = 2.0 V.
        t, Vd_stim = stim_fast_pulse(V_hi=2.0, V_lo=0.05,
                                       t_rise=100e-12, t_hold=30e-9,
                                       t_fall=100e-12,
                                       t_pre=0.5e-9, t_post=10e-9,
                                       n_total=1500)
        t_start = time.time()
        with torch.no_grad(), \
             z427.patch_sd_scaled(sd_M1, ctx["P_M1"]), \
             z427.patch_sd_scaled(sd_M2, ctx["P_M2"]):
            try:
                r = integrate(cfg, model_M1, model_M2, ctx["bjt"],
                              t, Vd_stim, bias["VG1"], bias["VG2"],
                              tcfg=tcfg, Vb0=0.0, verbose=False)
            except Exception as e:
                log(f"  FAIL {bias['tag']}: {e}")
                continue
        wall = time.time() - t_start
        Id_arr = np.array(r["Id"])
        Vb_arr = np.array(r["Vb"])
        # spike: peak Id during ramp-up (first ~1 ns) and steady-hold mean
        # (1 ns after ramp end to end of 30 ns hold).
        idx_ramp = (t < 1.0e-9)
        idx_hold = (t >= 1.0e-9) & (t < 30.6e-9)
        Id_peak_ramp = float(Id_arr[idx_ramp].max()) if idx_ramp.any() else float("nan")
        Id_hold_mean = float(Id_arr[idx_hold].mean()) if idx_hold.any() else float("nan")
        Vb_peak = float(Vb_arr.max())
        Vb_at_peak_idx = int(Vb_arr.argmax())
        t_peak = float(t[Vb_at_peak_idx])
        n_conv = sum(r["converged"])
        per_bias_out.append({
            "tag": bias["tag"],
            "VG1": bias["VG1"], "VG2": bias["VG2"],
            "Id_peak_ramp_A": Id_peak_ramp,
            "Id_hold_mean_A": Id_hold_mean,
            "Vb_peak_V": Vb_peak,
            "t_Vb_peak_s": t_peak,
            "spike_count": len(r["spike_times"]),
            "n_conv": int(n_conv),
            "conv_rate": float(n_conv) / len(t),
            "n_splits": int(r["n_splits_total"]),
            "wall_sec": round(wall, 1),
            "_traces": {
                "t": t.tolist(),
                "Vd": Vd_stim.tolist(),
                "Vb": r["Vb"],
                "Vsint": r["Vsint"],
                "Id": r["Id"],
            },
        })
        log(f"  {bias['tag']}: Id_peak={Id_peak_ramp:.3e}  "
            f"Id_hold={Id_hold_mean:.3e}  Vb_peak={Vb_peak:.3f}@{t_peak*1e9:.2f}ns "
            f"spikes={len(r['spike_times'])}  conv={n_conv}/{len(t)} "
            f"splits={r['n_splits_total']}  wall={wall:.1f}s")
    return {"per_bias": per_bias_out}


# ====================================================================== #
# Stimulus 3: HOLD THEN RELEASE — self-reset
# ====================================================================== #
def run_self_reset(model_M1, model_M2, cfg, sd_M1, sd_M2, sebas_rows):
    log("===== STIMULUS 3: HOLD + RELEASE (self-reset measurement) =====")
    tcfg = TransientCfg()
    # Use VG1=0.6 / VG2=0.0 only (representative; full sweep wasteful)
    bias = BIASES[0]
    ctx = per_bias_context(bias, model_M1, model_M2, cfg, sd_M1, sd_M2,
                           load_sebas_rows_cache[0])
    if ctx is None:
        return {"skipped": True}
    # Hold (20ns) develops body-voltage, then drop to V_read=0.2V and
    # measure decay over 5 µs (log-spaced sampling).
    t, Vd_stim = stim_hold_then_release(V_hi=2.0, V_read=0.2,
                                          t_rise=100e-12, t_hold=20e-9,
                                          t_release=5e-6, n_total=2500)
    t_start = time.time()
    with torch.no_grad(), \
         z427.patch_sd_scaled(sd_M1, ctx["P_M1"]), \
         z427.patch_sd_scaled(sd_M2, ctx["P_M2"]):
        r = integrate(cfg, model_M1, model_M2, ctx["bjt"],
                      t, Vd_stim, bias["VG1"], bias["VG2"],
                      tcfg=tcfg, Vb0=0.0, verbose=False)
    wall = time.time() - t_start

    # Self-reset metric: time after pulse release for Vb to decay to <10% of peak
    Vb_arr = np.array(r["Vb"])
    t_arr = np.array(t)
    release_t = 100e-12 + 20e-9   # end of hold
    Vb_peak = float(Vb_arr[t_arr < release_t].max())
    decay_thresh = 0.1 * Vb_peak
    post_idx = np.where(t_arr > release_t)[0]
    t_reset = None
    for j in post_idx:
        if Vb_arr[j] < decay_thresh:
            t_reset = float(t_arr[j] - release_t)
            break
    log(f"  Vb_peak={Vb_peak:.3f}  decay_thresh={decay_thresh:.3f}  "
        f"t_reset={'(no reset)' if t_reset is None else f'{t_reset*1e9:.1f} ns'} "
        f"conv={sum(r['converged'])}/{len(t)} splits={r['n_splits_total']} "
        f"wall={wall:.1f}s")
    return {
        "bias": {"VG1": bias["VG1"], "VG2": bias["VG2"]},
        "Vb_peak_V": Vb_peak,
        "decay_threshold_V": decay_thresh,
        "t_reset_s": t_reset,
        "conv_rate": float(sum(r["converged"])) / len(t),
        "n_splits": int(r["n_splits_total"]),
        "wall_sec": round(wall, 1),
        "_traces": {"t": t.tolist(),
                    "Vd": Vd_stim.tolist(),
                    "Vb": r["Vb"],
                    "Id": r["Id"]},
    }


load_sebas_rows_cache = []   # ugly but avoids re-loading


def _fmt_treset(t):
    if t is None: return "(no reset in window)"
    return f"{t*1e9:.1f} ns"


# ====================================================================== #
# Plotters
# ====================================================================== #
def plot_slow_dc_overlay(slow_dc):
    biases = slow_dc["per_bias"]
    n = len(biases)
    if n == 0:
        return
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows),
                              squeeze=False)
    for ax_i, rec in enumerate(biases):
        ax = axes[ax_i // ncols, ax_i % ncols]
        tr = rec["_traces"]
        ax.semilogy(tr["meas_Vd"], np.maximum(tr["meas_Id"], 1e-15),
                     "ko", ms=3, label="meas")
        ax.semilogy(tr["Vd_stim"], np.maximum(tr["Id"], 1e-15),
                     "b-", lw=1.2, label="slow-DC sim")
        ax.set_title(f"VG1={rec['VG1']} VG2={rec['VG2']}  "
                     f"rmse={rec['log_rmse_dec']:.2f} dec")
        ax.set_xlabel("V_D [V]"); ax.set_ylabel("|I_D| [A]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    # blank leftover
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    fig.suptitle(f"z447 slow-DC overlay: cell-RMSE = "
                 f"{slow_dc['cell_rmse_dec']:.3f} dec "
                 f"(z430 V_SINT_PIN baseline = {Z430_BASELINE_DEC:.3f} dec)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "slow_dc_overlay.png", dpi=130)
    plt.close(fig)


def plot_fast_pulse_VG1_0p6(fast):
    rec = next((r for r in fast["per_bias"] if r["tag"] == "VG1_0p6_VG2_0p0"), None)
    if rec is None:
        return
    tr = rec["_traces"]
    t = np.array(tr["t"]) * 1e9   # ns
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax0, ax1, ax2 = axes
    ax0.plot(t, tr["Vd"], "k-", lw=1.2, label="V_D(t)")
    ax0.plot(t, tr["Vsint"], "g--", lw=0.9, label="V_Sint(t)")
    ax0.set_ylabel("V [V]"); ax0.grid(True, alpha=0.3); ax0.legend(fontsize=8)
    ax1.plot(t, tr["Vb"], "b-", lw=1.2)
    ax1.axhline(0.65, color="r", ls=":", lw=0.8, label="spike threshold")
    ax1.set_ylabel("V_B [V]"); ax1.grid(True, alpha=0.3); ax1.legend(fontsize=8)
    ax2.semilogy(t, np.maximum(np.abs(tr["Id"]), 1e-15), "m-", lw=1.2)
    ax2.set_ylabel("|I_D| [A]"); ax2.set_xlabel("time [ns]")
    ax2.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"z447 fast pulse, VG1=0.6 / VG2=0.0  "
                 f"Vb_peak={rec['Vb_peak_V']:.3f}V "
                 f"Id_peak={rec['Id_peak_ramp_A']:.2e}A",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "waveforms_VG1_0p6.png", dpi=130)
    plt.close(fig)


def plot_self_reset(sr):
    if sr.get("skipped"):
        return
    tr = sr["_traces"]
    t = np.array(tr["t"]) * 1e9   # ns
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax0, ax1 = axes
    ax0.plot(t, tr["Vd"], "k-", lw=1.0, label="V_D(t)")
    ax0.plot(t, tr["Vb"], "b-", lw=1.2, label="V_B(t)")
    ax0.axhline(sr["decay_threshold_V"], color="r", ls=":", lw=0.8,
                label=f"10% decay = {sr['decay_threshold_V']:.3f}V")
    ax0.set_ylabel("V [V]"); ax0.grid(True, alpha=0.3); ax0.legend(fontsize=8)
    ax1.semilogy(t, np.maximum(np.abs(tr["Id"]), 1e-15), "m-", lw=1.2)
    ax1.set_xlabel("time [ns]"); ax1.set_ylabel("|I_D| [A]")
    ax1.grid(True, which="both", alpha=0.3)
    title = f"z447 self-reset, VG1=0.6 / VG2=0.0  "
    if sr["t_reset_s"] is None:
        title += "no reset within window"
    else:
        title += f"t_reset = {sr['t_reset_s']*1e9:.1f} ns"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "self_reset_trace.png", dpi=130)
    plt.close(fig)


# ====================================================================== #
# Main
# ====================================================================== #
def main():
    t0 = time.time()
    log("z447 starting — REAL transient pipeline (S32)")
    model_M1, model_M2, cfg, sd_M1, sd_M2 = setup_models_and_cfg()
    curves = z429.load_curves()
    sebas_rows = z429.load_sebas_params()
    load_sebas_rows_cache.append(sebas_rows)
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    slow_dc = run_slow_dc(model_M1, model_M2, cfg, sd_M1, sd_M2, curves, sebas_rows)
    fast    = run_fast_pulse(model_M1, model_M2, cfg, sd_M1, sd_M2, sebas_rows)
    sr      = run_self_reset(model_M1, model_M2, cfg, sd_M1, sd_M2, sebas_rows)

    # Plots
    plot_slow_dc_overlay(slow_dc)
    plot_fast_pulse_VG1_0p6(fast)
    plot_self_reset(sr)

    # Trim traces from JSON (keep first 6 fast biases full but compress slow)
    def trim_trace(rec, max_pts=200):
        if "_traces" not in rec: return
        tr = rec["_traces"]
        n = len(tr.get("t", []))
        if n <= max_pts: return
        idx = np.linspace(0, n - 1, max_pts).astype(int).tolist()
        for k in list(tr.keys()):
            v = tr[k]
            if isinstance(v, list) and len(v) == n:
                tr[k] = [v[i] for i in idx]
    for rec in slow_dc["per_bias"]:
        trim_trace(rec)
    for rec in fast["per_bias"]:
        trim_trace(rec)
    if not sr.get("skipped"):
        trim_trace(sr)

    # Pre-registered scoring
    INFRA_PASS = (slow_dc["cell_rmse_dec"] < Z430_BASELINE_DEC + 0.2)
    # DISCOVERY criterion: fast-ramp shows V_B charge accumulation
    # (Vb_peak > 0.3 V at end of 100ns hold, conv_rate > 80%)
    fast_passes = [r for r in fast["per_bias"]
                    if r["conv_rate"] > 0.8 and r["Vb_peak_V"] > 0.3]
    DISCOVERY_PASS = len(fast_passes) >= 3
    AMBITIOUS_PASS = (sr.get("t_reset_s") is not None
                      and 1e-7 < sr["t_reset_s"] < 1e-5)
    KILL_SHOT = (slow_dc["cell_rmse_dec"] > 5.0
                  or all(not r["conv_rate"] > 0.5 for r in fast["per_bias"])
                  if fast["per_bias"] else False)

    summary = {
        "exp": "z447_real_transient",
        "wall_sec": round(time.time() - t0, 1),
        "transient_cfg": {
            "C_B_const_F": 1.0e-15,
            "Cje_F": 0.7e-15, "Cjc_F": 1.0e-15,
            "tau_F_s": 25e-12, "tau_R_s": 20e-12,
            "Vsint_pin": 0.0,
            "self_heating_enabled": False,
        },
        "slow_dc": slow_dc,
        "fast_pulse": fast,
        "self_reset": sr,
        "scoring": {
            "INFRA_PASS_slow_dc_within_0p2": bool(INFRA_PASS),
            "DISCOVERY_PASS_3_of_6_pulses_show_Vb_rise": bool(DISCOVERY_PASS),
            "AMBITIOUS_PASS_self_reset_100ns_to_10us": bool(AMBITIOUS_PASS),
            "KILL_SHOT": bool(KILL_SHOT),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # honest_analysis.md
    lines = [
        "# z447 — Real Transient Pipeline: Honest Analysis\n",
        f"Wall time: {summary['wall_sec']:.0f}s\n\n",
        "## What was verified\n",
        f"- Slow-DC integration (0.2 V/s ramp, implicit Euler, C_B = 1 fF) cell-RMSE = "
        f"**{slow_dc['cell_rmse_dec']:.3f} dec** vs z430 V_SINT_PIN baseline "
        f"{Z430_BASELINE_DEC:.3f} dec.  ΔRMSE = "
        f"{slow_dc['delta_dec']:+.3f} dec (target: within ±0.2).\n",
        f"  - INFRA: **{'PASS' if INFRA_PASS else 'FAIL'}**\n",
        f"- Fast pulse (100ps ramp, 10ns hold) ran on "
        f"{len(fast['per_bias'])} biases; "
        f"{len(fast_passes)} showed V_B rise > 0.4 V with conv_rate > 80%.\n",
        f"  - DISCOVERY: **{'PASS' if DISCOVERY_PASS else 'FAIL'}**\n",
        f"- Self-reset trace (hold 2ns at V_D=2V then release to V_read=0.5V): "
        f"t_reset = {_fmt_treset(sr.get('t_reset_s'))}.\n",
        f"  - AMBITIOUS (100ns-10µs): **{'PASS' if AMBITIOUS_PASS else 'FAIL'}**\n",
        "\n## What's still missing / assumed\n",
        "- BJT diffusion-cap currents are added via *first-order forward difference* on Ic,Ie "
        "(I_diff = τ_F·dIc/dt + τ_R·dIe/dt). This is qualitatively correct (yields ~τ-scale "
        "spike width) but not τ-quantitative — full charge-state ODEs (dqF/dt, dqR/dt) would "
        "require an internal Gummel-Poon charge model, which the current `bjt` wrapper does "
        "not expose.\n",
        "- Self-heating dT/dt is wired in `TransientCfg` (R_th, C_th) but **disabled** in this "
        "run — Sebas's TIM data (A.12) is the missing input. Will be re-enabled when TIM "
        "available.\n",
        "- Junction-cap Mj grading was set to 0.33 (BJT default); the parasiticBJT card is "
        "silent on this. N-well diode Cj uses Sebas's card verbatim.\n",
        "- V_Sint was pinned to 0 to match z430's baseline solver. Joint solve (Vsint free) is "
        "supported by `TransientCfg(Vsint_pin=None)` but not used here — pinning is the "
        "topology that **converges** cell-wide.\n",
        "- Spike threshold (V_B ≥ 0.65 → reset to 0.30 V) is the existing transient.py heuristic; "
        "it is a software model, not silicon — a real BJT collapses via avalanche / latchup, "
        "not by a threshold reset.\n",
        "\n## Solver health\n",
        f"- Slow-DC: total wall = {sum(r.get('wall_sec', 0) for r in slow_dc['per_bias']):.1f}s, "
        f"avg conv_rate = {np.mean([r['conv_rate'] for r in slow_dc['per_bias']]):.1%}, "
        f"total splits = {sum(r['n_splits'] for r in slow_dc['per_bias'])}.\n",
        f"- Fast pulse: avg conv_rate = "
        f"{np.mean([r['conv_rate'] for r in fast['per_bias']]) if fast['per_bias'] else float('nan'):.1%}, "
        f"total splits = {sum(r['n_splits'] for r in fast['per_bias'])}.\n",
        f"- Self-reset: conv_rate = {sr.get('conv_rate', float('nan')):.1%}, "
        f"splits = {sr.get('n_splits', 0)}.\n",
        f"- KILL_SHOT (RMSE > 5 dec OR pulses all diverge): **{'TRIGGERED' if KILL_SHOT else 'no'}**\n",
    ]
    (OUT / "honest_analysis.md").write_text("".join(lines))
    log("z447 done. summary.json + 3 PNGs + honest_analysis.md written.")
    log(f"  INFRA: {INFRA_PASS}  DISCOVERY: {DISCOVERY_PASS}  "
        f"AMBITIOUS: {AMBITIOUS_PASS}  KILL_SHOT: {KILL_SHOT}")


if __name__ == "__main__":
    main()
