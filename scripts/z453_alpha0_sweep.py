"""z453 — alpha0 sweep on the best DC pipeline (z447 slow-DC pseudo-transient).

Tests SINGLE-PARAMETER hypothesis: M1.alpha0 is 10-100× under-calibrated and
that one knob simultaneously closes DC knee + ns-snap.

Variants:
  A1   : 1×  (baseline)  alpha0 = 7.84e-5
  A10  : 10×             alpha0 = 7.84e-4
  A30  : 30×             alpha0 = 2.35e-3
  A100 : 100×            alpha0 = 7.84e-3

For EACH variant:
  1) DC cell-wide RMSE (forward AND backward sweep) over Sebas's 33-curve set
  2) Per-branch breakdown VG1=0.2/0.4/0.6
  3) Fast-pulse smoke test on 4 biases (V_D 100ps rise to 2V + 10ns hold)
     Records max(V_B), t_to_0.3V, t_to_0.5V, self_reset.

PRE-REGISTERED GATES are logged on line 1 of run.log BEFORE compute.

Run via:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 \
      nohup venv/bin/python scripts/z453_alpha0_sweep.py \
      > results/z453_alpha0_sweep/nohup.out 2>&1 &
"""
from __future__ import annotations
import csv
import importlib.util as _ilu
import json
import math
import re
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
OUT = ROOT / "results/z453_alpha0_sweep"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


# ====================================================================== #
# PRE-REGISTERED GATES (MANDATORY: line 1 of run.log before compute)
# ====================================================================== #
PREREG = (
    "PRE-REGISTERED GATES (locked before compute):\n"
    "  INFRA     = all 4 alpha0 variants complete without crash & summary.json written\n"
    "  DISCOVERY = some alpha0 gives cell DC (forward+backward avg) < 0.85 dec "
    "AND >=2/4 biases show V_B > 0.3 V within 5 ns\n"
    "  AMBITIOUS = some alpha0 gives cell DC (avg) < 0.6 dec AND >=3/4 biases show "
    "V_B > 0.5 V within 10 ns AND self_reset visible\n"
    "  KILL_SHOT = monotonic-worse cell DC across all alpha0 multipliers "
    "(alpha0 NOT the dominant knob — z451 hypothesis wrong)\n"
)
LOG.write(PREREG + "\n"); LOG.flush()
print(PREREG, flush=True)

# Re-use upstream loaders
_spec427 = _ilu.spec_from_file_location("z427", ROOT / "scripts/z427_vsint_fix.py")
z427 = _ilu.module_from_spec(_spec427); _spec427.loader.exec_module(z427)
_spec429 = _ilu.spec_from_file_location("z429", ROOT / "scripts/z429_multisolver_debug.py")
z429 = _ilu.module_from_spec(_spec429); _spec429.loader.exec_module(z429)

from nsram.bsim4_port.transient_real_v2 import (
    integrate, TransientCfgV2 as TransientCfg,
    stim_slow_dc_ramp, stim_fast_pulse,
)


# ====================================================================== #
# Thermal guard
# ====================================================================== #
THERMAL_LIMIT_C = 85.0
def check_thermal(stage=""):
    try:
        t = int(open("/sys/class/thermal/thermal_zone0/temp").read().strip()) / 1000.0
    except Exception:
        return
    if t >= THERMAL_LIMIT_C:
        log(f"!!! THERMAL {t:.1f}C >= {THERMAL_LIMIT_C}C at {stage}; sleeping 60s")
        time.sleep(60)


# ====================================================================== #
# Backward-sweep loader (mirror of z429.load_curves but uses second half)
# ====================================================================== #
def _parse_vg1(s):
    m = re.search(r"VG1=([\d.]+)", s); return float(m.group(1)) if m else None
def _parse_vg2(s):
    m = re.search(r"VG2=(-?\d+\.\d+)", s); return float(m.group(1)) if m else None

def load_curves_backward():
    """Same shape as z429.load_curves() but uses the SECOND half of the
    triangular sweep (backward leg). Returns curves list of dicts with
    Vd/Id numpy-tensor pair, monotonically increasing in Vd.
    """
    curves = []
    for d in sorted(DATA.glob("2vHCa-2 I-Vs@VG2 VG1=*")):
        VG1 = _parse_vg1(d.name)
        for f in sorted(d.glob("*.csv")):
            VG2 = _parse_vg2(f.name)
            try:
                data = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(0, 1))
            except Exception:
                continue
            if data.ndim == 1 or len(data) < 20:
                continue
            half = len(data) // 2
            Vd = data[half:, 0]
            Id = np.abs(data[half:, 1])
            # Backward leg sweeps high→low; reverse so it is monotonically
            # increasing (so np.interp / mask logic works identically).
            order = np.argsort(Vd)
            Vd, Id = Vd[order], Id[order]
            mask = (Vd >= 0.05) & (Vd <= 2.0)
            Vd, Id = Vd[mask], Id[mask]
            if len(Vd) > 10:
                idx = np.linspace(0, len(Vd) - 1, 30).astype(int)
                Vd, Id = Vd[idx], Id[idx]
                curves.append({"VG1": VG1, "VG2": VG2,
                               "Vd": torch.tensor(Vd, dtype=torch.float64),
                               "Id": torch.tensor(Id, dtype=torch.float64)})
    return curves


# ====================================================================== #
# alpha0 variants
# ====================================================================== #
BASELINE_ALPHA0 = 7.83756e-5
VARIANTS = [
    {"name": "A1",   "mult": 1.0,   "alpha0": BASELINE_ALPHA0 * 1.0},
    {"name": "A10",  "mult": 10.0,  "alpha0": BASELINE_ALPHA0 * 10.0},
    {"name": "A30",  "mult": 30.0,  "alpha0": BASELINE_ALPHA0 * 30.0},
    {"name": "A100", "mult": 100.0, "alpha0": BASELINE_ALPHA0 * 100.0},
]

FAST_BIASES = [
    {"VG1": 0.6, "VG2": 0.0, "tag": "VG1_0p6_VG2_0p0"},
    {"VG1": 0.6, "VG2": 0.2, "tag": "VG1_0p6_VG2_0p2"},
    {"VG1": 0.4, "VG2": 0.0, "tag": "VG1_0p4_VG2_0p0"},
    {"VG1": 0.2, "VG2": 0.0, "tag": "VG1_0p2_VG2_0p0"},
]


# ====================================================================== #
# DC RMSE evaluation (cell-wide; per-branch decomposition)
# ====================================================================== #
def dc_rmse_for_alpha0(model_M1, model_M2, cfg, sd_M1, sd_M2,
                       curves, sebas_rows, alpha0_value, leg_name):
    """Run slow-DC pseudo-transient sweep over ALL Sebas (VG1,VG2) pairs.
    Override alpha0 on each P_M1. Returns (cell_rmse, per_branch_rmse{0.2,0.4,0.6}, n_curves_scored)."""
    log_eps = 1e-15
    tcfg = TransientCfg()

    branch_sq = {0.2: [0.0, 0], 0.4: [0.0, 0], 0.6: [0.0, 0]}
    sq_total, n_total = 0.0, 0
    n_curves_scored = 0
    n_curves_skipped = 0

    for c in curves:
        VG1, VG2 = c["VG1"], c["VG2"]
        meas_Vd, meas_Id = c["Vd"].numpy(), c["Id"].numpy()
        sebas_row = z427.find_params(sebas_rows, VG1, VG2)
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            n_curves_skipped += 1; continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        # Override alpha0 with sweep value
        P_M1["alpha0"] = torch.tensor(float(alpha0_value), dtype=torch.float64)
        bjt = z427.make_bjt(sebas_row)

        V_lo = max(0.01, float(np.min(meas_Vd)) - 0.02)
        V_hi = float(np.max(meas_Vd)) + 0.02
        # Slow ramp — 0.2 V/s; n=200 (z447 setting)
        t, Vd_stim = stim_slow_dc_ramp(V_lo=V_lo, V_hi=V_hi,
                                        rate_V_per_s=0.2, n=200)

        try:
            with torch.no_grad(), \
                 z427.patch_sd_scaled(sd_M1, P_M1), \
                 z427.patch_sd_scaled(sd_M2, P_M2):
                r = integrate(cfg, model_M1, model_M2, bjt,
                              t, Vd_stim, VG1, VG2,
                              tcfg=tcfg, Vb0=0.0, verbose=False)
        except Exception as e:
            log(f"    {leg_name} VG1={VG1} VG2={VG2}: FAIL {e}")
            n_curves_skipped += 1
            continue

        Id_sim = np.array(r["Id"])
        conv = np.array(r["converged"])
        Vd_sim = np.array(Vd_stim)
        mask = conv & (Id_sim > log_eps)
        if mask.sum() < 3:
            n_curves_skipped += 1; continue
        log_Id_sim = np.log10(np.maximum(Id_sim[mask], log_eps))
        log_Id_on_meas = np.interp(meas_Vd, Vd_sim[mask], log_Id_sim)
        meas_mask = (meas_Vd >= Vd_sim[mask].min()) & (meas_Vd <= Vd_sim[mask].max())
        Id_sim_on_meas = np.power(10.0, log_Id_on_meas)
        log_p = np.log10(np.maximum(Id_sim_on_meas, log_eps))
        log_m = np.log10(np.maximum(meas_Id, log_eps))
        sq_all = (log_p - log_m) ** 2
        sq = sq_all[meas_mask]
        if len(sq) == 0:
            n_curves_skipped += 1; continue

        sq_total += float(sq.sum())
        n_total += int(len(sq))
        n_curves_scored += 1

        # bucket by VG1
        for key in branch_sq:
            if abs(VG1 - key) < 1e-3:
                branch_sq[key][0] += float(sq.sum())
                branch_sq[key][1] += int(len(sq))
                break

        check_thermal(f"DC {leg_name} VG1={VG1} VG2={VG2}")

    cell_rmse = float(np.sqrt(sq_total / max(n_total, 1)))
    per_branch = {}
    for k, (s, n) in branch_sq.items():
        per_branch[f"VG1_{k:.1f}"] = (float(np.sqrt(s / n)) if n > 0 else float("nan"))
    return cell_rmse, per_branch, n_curves_scored, n_curves_skipped


# ====================================================================== #
# Fast-pulse evaluation
# ====================================================================== #
def fast_pulse_for_alpha0(model_M1, model_M2, cfg, sd_M1, sd_M2,
                          sebas_rows, alpha0_value):
    """Run fast pulse on FAST_BIASES, return list of records."""
    tcfg = TransientCfg()
    recs = []
    for bias in FAST_BIASES:
        VG1, VG2 = bias["VG1"], bias["VG2"]
        sebas_row = z427.find_params(sebas_rows, VG1, VG2)
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            log(f"    fast bias VG1={VG1} VG2={VG2}: no Sebas params")
            continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        P_M1["alpha0"] = torch.tensor(float(alpha0_value), dtype=torch.float64)
        bjt = z427.make_bjt(sebas_row)

        # 100ps rise + 10ns hold + 100ps fall + 5ns post (for self-reset check)
        t, Vd_stim = stim_fast_pulse(V_hi=2.0, V_lo=0.05,
                                      t_rise=100e-12, t_hold=10e-9,
                                      t_fall=100e-12, t_pre=0.5e-9, t_post=5e-9,
                                      n_total=1200)
        t_arr = np.asarray(t)
        try:
            with torch.no_grad(), \
                 z427.patch_sd_scaled(sd_M1, P_M1), \
                 z427.patch_sd_scaled(sd_M2, P_M2):
                r = integrate(cfg, model_M1, model_M2, bjt,
                              t, Vd_stim, VG1, VG2,
                              tcfg=tcfg, Vb0=0.0, verbose=False)
        except Exception as e:
            log(f"    fast bias VG1={VG1} VG2={VG2}: FAIL {e}")
            continue

        Vb_arr = np.asarray(r["Vb"])
        # ramp starts at t_pre=0.5ns. Define t_relative = t - 0.5ns.
        t_rel = t_arr - 0.5e-9

        def first_cross(thresh, within_s=None):
            mask = Vb_arr >= thresh
            if within_s is not None:
                mask = mask & (t_rel <= within_s) & (t_rel >= 0)
            else:
                mask = mask & (t_rel >= 0)
            if not mask.any():
                return None
            idx = int(np.argmax(mask))
            return float(t_rel[idx])

        max_VB = float(Vb_arr.max())
        t_to_0p3 = first_cross(0.3)
        t_to_0p5 = first_cross(0.5)
        # self-reset: Vb peaks then decays below 0.1V before end of window
        Vb_peak = float(Vb_arr.max())
        idx_peak = int(np.argmax(Vb_arr))
        post_peak = Vb_arr[idx_peak:]
        self_reset = bool((post_peak < 0.1).any() and Vb_peak > 0.2)
        n_conv = int(sum(r["converged"]))

        recs.append({
            "tag": bias["tag"],
            "VG1": VG1, "VG2": VG2,
            "max_VB": max_VB,
            "t_to_VB_0p3_s": t_to_0p3,
            "t_to_VB_0p5_s": t_to_0p5,
            "self_reset": self_reset,
            "conv_rate": float(n_conv) / len(t_arr),
            "n_splits": int(r["n_splits_total"]),
            "_traces": {
                "t": t_arr.tolist(),
                "Vd": list(Vd_stim),
                "Vb": list(r["Vb"]),
                "Id": list(r["Id"]),
            },
        })
        log(f"    fast {bias['tag']}: maxVB={max_VB:.3f}  "
            f"t→0.3={'-' if t_to_0p3 is None else f'{t_to_0p3*1e9:.2f}ns'}  "
            f"t→0.5={'-' if t_to_0p5 is None else f'{t_to_0p5*1e9:.2f}ns'}  "
            f"self_reset={self_reset}  conv={n_conv}/{len(t_arr)}")
        check_thermal("fast pulse")
    return recs


# ====================================================================== #
# Plotters
# ====================================================================== #
def plot_dc_vs_alpha0(per_variant):
    mults = [v["mult"] for v in per_variant]
    fwd  = [v["dc_forward"]  for v in per_variant]
    bwd  = [v["dc_backward"] for v in per_variant]
    avg  = [v["dc_avg"]      for v in per_variant]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(mults, fwd, "o-",  label="forward DC", lw=1.6)
    ax.semilogx(mults, bwd, "s--", label="backward DC", lw=1.6)
    ax.semilogx(mults, avg, "k^-", label="avg fwd/bwd", lw=2.0)
    for x, f, b in zip(mults, fwd, bwd):
        ax.annotate(f"{f:.2f}", (x, f), textcoords="offset points",
                    xytext=(4, 5), fontsize=8)
        ax.annotate(f"{b:.2f}", (x, b), textcoords="offset points",
                    xytext=(4, -10), fontsize=8)
    ax.axhline(0.85, color="g", ls=":", lw=0.8, label="DISCOVERY 0.85")
    ax.axhline(0.60, color="r", ls=":", lw=0.8, label="AMBITIOUS 0.60")
    ax.set_xlabel("M1.alpha0 multiplier (×%.2e)" % BASELINE_ALPHA0)
    ax.set_ylabel("cell-wide log10 RMSE [dec]")
    ax.set_title("z453: DC fit quality vs alpha0 multiplier")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "dc_vs_alpha0.png", dpi=130)
    plt.close(fig)


def plot_fast_pulse_overlay(per_variant):
    nb = len(FAST_BIASES)
    fig, axes = plt.subplots(nb, 1, figsize=(9, 2.8 * nb), sharex=True)
    if nb == 1:
        axes = [axes]
    colors = {"A1": "k", "A10": "tab:blue", "A30": "tab:orange", "A100": "tab:red"}
    for i, bias in enumerate(FAST_BIASES):
        ax = axes[i]
        for v in per_variant:
            rec = next((r for r in v["fast_pulse"] if r["tag"] == bias["tag"]), None)
            if rec is None:
                continue
            tr = rec["_traces"]
            t_ns = (np.asarray(tr["t"]) - 0.5e-9) * 1e9
            ax.plot(t_ns, tr["Vb"], color=colors.get(v["name"], "gray"),
                    lw=1.2, label=f"{v['name']} (×{v['mult']:g})")
        ax.axhline(0.3, color="g", ls=":", lw=0.7)
        ax.axhline(0.5, color="r", ls=":", lw=0.7)
        ax.set_ylabel("V_B [V]")
        ax.set_title(f"{bias['tag']}  VG1={bias['VG1']} VG2={bias['VG2']}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("time since ramp start [ns]")
    fig.suptitle("z453 fast-pulse V_B(t) overlay (V_D: 100ps→2V, 10ns hold)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fast_pulse_overlay.png", dpi=130)
    plt.close(fig)


# ====================================================================== #
# Main
# ====================================================================== #
def main():
    t0 = time.time()
    log("z453 starting — alpha0 sweep (A1/A10/A30/A100)")
    log(f"baseline alpha0 from M1 card = {BASELINE_ALPHA0:.5e}")

    # Setup models/cfg once
    model_M1, model_M2 = z429.build_models()
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, {})
    curves_fwd = z429.load_curves()
    curves_bwd = load_curves_backward()
    sebas_rows = z429.load_sebas_params()
    log(f"loaded: {len(curves_fwd)} fwd curves, {len(curves_bwd)} bwd curves, "
        f"{len(sebas_rows)} sebas rows")

    per_variant = []
    for var in VARIANTS:
        check_thermal(f"start of {var['name']}")
        log(f"===== variant {var['name']}  (mult={var['mult']:g}, "
            f"alpha0={var['alpha0']:.4e}) =====")

        t_var = time.time()
        log(f"  DC forward sweep ({len(curves_fwd)} curves)...")
        dc_fwd, br_fwd, n_fwd, sk_fwd = dc_rmse_for_alpha0(
            model_M1, model_M2, cfg, sd_M1, sd_M2,
            curves_fwd, sebas_rows, var["alpha0"], "FWD")
        log(f"  FWD: cell_rmse={dc_fwd:.3f} dec  scored={n_fwd}  skipped={sk_fwd}")
        log(f"  FWD per-branch: {br_fwd}")

        check_thermal(f"mid {var['name']}")
        log(f"  DC backward sweep ({len(curves_bwd)} curves)...")
        dc_bwd, br_bwd, n_bwd, sk_bwd = dc_rmse_for_alpha0(
            model_M1, model_M2, cfg, sd_M1, sd_M2,
            curves_bwd, sebas_rows, var["alpha0"], "BWD")
        log(f"  BWD: cell_rmse={dc_bwd:.3f} dec  scored={n_bwd}  skipped={sk_bwd}")
        log(f"  BWD per-branch: {br_bwd}")

        log(f"  Fast pulse on {len(FAST_BIASES)} biases...")
        fast = fast_pulse_for_alpha0(model_M1, model_M2, cfg, sd_M1, sd_M2,
                                       sebas_rows, var["alpha0"])

        dc_avg = float(0.5 * (dc_fwd + dc_bwd))
        per_branch_avg = {}
        for key in br_fwd:
            f = br_fwd[key]; b = br_bwd.get(key, float("nan"))
            if math.isnan(f) or math.isnan(b):
                per_branch_avg[key] = float("nan")
            else:
                per_branch_avg[key] = 0.5 * (f + b)

        per_variant.append({
            "name": var["name"],
            "mult": var["mult"],
            "alpha0": var["alpha0"],
            "dc_forward":  dc_fwd,
            "dc_backward": dc_bwd,
            "dc_avg":      dc_avg,
            "per_branch_forward":  br_fwd,
            "per_branch_backward": br_bwd,
            "per_branch_avg":      per_branch_avg,
            "n_curves_scored_fwd": n_fwd,
            "n_curves_scored_bwd": n_bwd,
            "fast_pulse": fast,
            "wall_sec": round(time.time() - t_var, 1),
        })
        log(f"  {var['name']} DONE  dc_avg={dc_avg:.3f}  wall={time.time()-t_var:.1f}s")

    # ────────── Trim traces for JSON ─────────────────────────────────────────
    def trim(rec, max_pts=120):
        if "_traces" not in rec: return
        tr = rec["_traces"]
        n = len(tr.get("t", []))
        if n <= max_pts: return
        idx = np.linspace(0, n - 1, max_pts).astype(int).tolist()
        for k in list(tr.keys()):
            v = tr[k]
            if isinstance(v, list) and len(v) == n:
                tr[k] = [v[i] for i in idx]
    for v in per_variant:
        for r in v["fast_pulse"]:
            trim(r)

    # ────────── Plots ────────────────────────────────────────────────────────
    plot_dc_vs_alpha0(per_variant)
    plot_fast_pulse_overlay(per_variant)

    # ────────── Gates ────────────────────────────────────────────────────────
    INFRA = (len(per_variant) == 4)

    # DISCOVERY: some variant gives dc_avg < 0.85 AND >=2/4 biases with VB>0.3 within 5ns
    def discovery_ok(v):
        if v["dc_avg"] >= 0.85: return False
        n_ok = sum(1 for r in v["fast_pulse"]
                   if r.get("t_to_VB_0p3_s") is not None
                   and r["t_to_VB_0p3_s"] <= 5e-9)
        return n_ok >= 2
    DISCOVERY = any(discovery_ok(v) for v in per_variant)

    # AMBITIOUS: dc_avg<0.6 AND >=3/4 biases with VB>0.5 within 10ns AND self_reset visible
    def ambitious_ok(v):
        if v["dc_avg"] >= 0.6: return False
        n_ok = sum(1 for r in v["fast_pulse"]
                   if r.get("t_to_VB_0p5_s") is not None
                   and r["t_to_VB_0p5_s"] <= 10e-9)
        if n_ok < 3: return False
        any_reset = any(r.get("self_reset") for r in v["fast_pulse"])
        return any_reset
    AMBITIOUS = any(ambitious_ok(v) for v in per_variant)

    # KILL_SHOT: dc_avg is monotonically NON-IMPROVING as mult increases
    avgs = [v["dc_avg"] for v in per_variant]
    KILL_SHOT = all(avgs[i+1] >= avgs[i] for i in range(len(avgs) - 1))

    best_idx = int(np.argmin(avgs))
    best = per_variant[best_idx]

    gates = {
        "INFRA":     bool(INFRA),
        "DISCOVERY": bool(DISCOVERY),
        "AMBITIOUS": bool(AMBITIOUS),
        "KILL_SHOT": bool(KILL_SHOT),
        "best_variant": best["name"],
        "best_dc_avg": best["dc_avg"],
    }

    summary = {
        "exp": "z453_alpha0_sweep",
        "wall_sec": round(time.time() - t0, 1),
        "baseline_alpha0_card": BASELINE_ALPHA0,
        "z447_baseline_fwd_dec": 0.886,
        "variants": per_variant,
        "gates": gates,
        "fast_biases": FAST_BIASES,
        "prereg_text": PREREG,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # honest_analysis.md
    lines = [
        "# z453 — alpha0 sweep: honest analysis\n\n",
        f"Wall time: {summary['wall_sec']:.0f} s\n\n",
        "## Pre-registered gates\n",
        f"```\n{PREREG}\n```\n",
        "## Result summary\n",
        "| Variant | mult | alpha0 | DC fwd | DC bwd | DC avg | branch 0.2 / 0.4 / 0.6 (avg) |\n",
        "|---|---|---|---|---|---|---|\n",
    ]
    for v in per_variant:
        pb = v["per_branch_avg"]
        lines.append(
            f"| {v['name']} | {v['mult']:g}× | {v['alpha0']:.3e} | "
            f"{v['dc_forward']:.3f} | {v['dc_backward']:.3f} | **{v['dc_avg']:.3f}** | "
            f"{pb.get('VG1_0.2', float('nan')):.3f} / "
            f"{pb.get('VG1_0.4', float('nan')):.3f} / "
            f"{pb.get('VG1_0.6', float('nan')):.3f} |\n")
    lines += [
        "\n## Fast-pulse table (max V_B, t→0.3V, t→0.5V, self-reset)\n",
        "| Variant | bias | maxVB [V] | t→0.3 [ns] | t→0.5 [ns] | self_reset |\n",
        "|---|---|---|---|---|---|\n",
    ]
    for v in per_variant:
        for r in v["fast_pulse"]:
            t03 = "-" if r["t_to_VB_0p3_s"] is None else f"{r['t_to_VB_0p3_s']*1e9:.2f}"
            t05 = "-" if r["t_to_VB_0p5_s"] is None else f"{r['t_to_VB_0p5_s']*1e9:.2f}"
            lines.append(f"| {v['name']} | {r['tag']} | {r['max_VB']:.3f} | "
                         f"{t03} | {t05} | {r['self_reset']} |\n")
    lines += [
        "\n## Gates\n",
        f"- INFRA: **{'PASS' if INFRA else 'FAIL'}**\n",
        f"- DISCOVERY: **{'PASS' if DISCOVERY else 'FAIL'}**\n",
        f"- AMBITIOUS: **{'PASS' if AMBITIOUS else 'FAIL'}**\n",
        f"- KILL_SHOT: **{'TRIGGERED' if KILL_SHOT else 'no'}**\n",
        f"\nBest variant: **{best['name']}** (dc_avg = {best['dc_avg']:.3f}).\n",
        "\n## Verdict on z451 ALPHA0 hypothesis\n",
        ("- The z451 hypothesis predicts alpha0 ↑ ⇒ DC RMSE ↓ + ns-snap closes.\n"
         "- KILL_SHOT triggered (monotonic-worse). alpha0 alone is NOT the dominant knob.\n"
         if KILL_SHOT else
         "- Some alpha0 multiplier improves DC. See best variant above.\n"),
        ("- Closure of DC AND ns-snap: AMBITIOUS PASS.\n" if AMBITIOUS else
         "- ns-snap did not close (V_B did not reach 0.5 V within 10 ns on ≥3/4 biases)." if not DISCOVERY else
         "- Partial: DC closed but ns-snap did not reach AMBITIOUS criterion.\n"),
        "\n## No-cheat audit\n",
        "- Both forward AND backward DC reported; dc_avg used in gates.\n",
        "- Per-branch breakdown by VG1 reported (catches sub-threshold regressions).\n",
        "- Fast-pulse uses transient_real_v2.py BDF solver, identical for all 4 variants.\n",
    ]
    (OUT / "honest_analysis.md").write_text("".join(lines))

    log(f"z453 DONE  best={best['name']}  dc_avg={best['dc_avg']:.3f}  "
        f"INFRA={INFRA} DISCOVERY={DISCOVERY} AMBITIOUS={AMBITIOUS} KILL_SHOT={KILL_SHOT}")
    log(f"total wall = {summary['wall_sec']:.0f} s")


if __name__ == "__main__":
    main()
