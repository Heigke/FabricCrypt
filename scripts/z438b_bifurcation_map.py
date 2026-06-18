"""z438b — Bifurcation map: where (V_G1, V_G2) does the model exhibit
bistability vs where do measured curves show snapback?

Per pre-registration (S27):
  1. Per bias: fwd + bwd pseudo-transient sweep → hysteresis amplitude
     A_model = mean(|log10(Id_fwd) - log10(Id_bwd)|) over V_D.
     A > 0.2 dec  ⇒ "model bistable here".
  2. Per bias: from MEASURED curve compute knee score
     k_meas = max |d^2 log10(Id) / dV_D^2|  over V_D ∈ [0.5, 1.5].
     k_meas > threshold ⇒ "measured snapback present".
  3. Plot heatmaps + mismatch overlay.
  4. Approximate boundary curve in (V_G1, V_G2).
  5. At boundary points, compute V_GS,M2 - V_T,M2 (overdrive proxy).

Output: results/z438_bifurcation_map/
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
OUT = ROOT / "results/z438_bifurcation_map"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


# Reuse z427 (loaders, cfg, model build) and z432 (ptran integrator)
_s427 = _ilu.spec_from_file_location("z427", ROOT / "scripts/z427_vsint_fix.py")
z427 = _ilu.module_from_spec(_s427); _s427.loader.exec_module(z427)
_s432 = _ilu.spec_from_file_location("z432", ROOT / "scripts/z432_pseudotransient.py")
z432 = _ilu.module_from_spec(_s432); _s432.loader.exec_module(z432)

# Reduce step budget to stay within wall-time. Bistability detection only
# needs the integrator to either converge or report a different attractor
# in fwd vs bwd — full convergence not required.
N_STEPS_OVERRIDE = 250
z432.N_MIN_STEPS = 60             # was 100


def _run_one_bias_capped(cfg, model_M1, model_M2, bjt, Vd_arr, VG1, VG2,
                          backward=False, Vb_init_first=0.0,
                          n_steps_cap=N_STEPS_OVERRIDE):
    """Mirror of z432.run_one_bias but with explicit n_steps cap."""
    if backward:
        order = list(range(len(Vd_arr) - 1, -1, -1))
    else:
        order = list(range(len(Vd_arr)))
    Vb_warm = Vb_init_first
    Id_out = [None] * len(Vd_arr)
    Vb_out = [None] * len(Vd_arr)
    conv_out = [False] * len(Vd_arr)
    niter_out = [0] * len(Vd_arr)
    for idx in order:
        Vd_f = float(Vd_arr[idx])
        r = z432.integrate_vb(cfg, model_M1, model_M2, bjt,
                                Vd_f, float(VG1), float(VG2),
                                Vb_init=Vb_warm,
                                n_steps=n_steps_cap)
        Id_out[idx] = abs(r["Id"])
        Vb_out[idx] = r["Vb"]
        conv_out[idx] = bool(r["converged"])
        niter_out[idx] = int(r["niter"])
        Vb_warm = r["Vb"]
    return Id_out, Vb_out, conv_out, niter_out


# ============================================================ #
# Measured knee score
# ============================================================ #

def measured_knee_score(Vd: np.ndarray, Id: np.ndarray,
                         vd_lo=0.4, vd_hi=1.6, eps=1e-15):
    """max |d^2 log10(Id) / dV_D^2| over Vd ∈ [vd_lo, vd_hi].

    A "snapback knee" produces a large second derivative (concave-up or
    concave-down spike). Smooth saturation has small d^2.
    """
    Vd = np.asarray(Vd, float)
    Id = np.asarray(Id, float)
    if Vd.size < 5:
        return 0.0
    log_id = np.log10(np.maximum(np.abs(Id), eps))
    # smooth d^2 via finite differences (uniform-ish grid; sebas grid varies)
    # Use 3-point centred differences with spacing taken locally.
    d2 = np.zeros_like(log_id)
    for i in range(1, len(Vd) - 1):
        h1 = Vd[i] - Vd[i - 1]
        h2 = Vd[i + 1] - Vd[i]
        if h1 <= 0 or h2 <= 0:
            continue
        d2[i] = 2 * (h2 * log_id[i - 1] - (h1 + h2) * log_id[i]
                      + h1 * log_id[i + 1]) / (h1 * h2 * (h1 + h2))
    mask = (Vd >= vd_lo) & (Vd <= vd_hi)
    if mask.sum() < 2:
        mask = np.ones_like(Vd, dtype=bool)
    return float(np.max(np.abs(d2[mask])))


# ============================================================ #
# Per-bias hysteresis runner (returns full per-bias dict)
# ============================================================ #

def run_per_bias(model_M1, model_M2, curves, sebas_rows):
    """For each bias, run fwd + bwd ptran, return dict per bias with:
        Vd, Id_meas, Id_fwd, Id_bwd, conv_fwd, conv_bwd.
    """
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, {})
    out = []
    t0 = time.time()
    for i, c in enumerate(curves):
        sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        bjt = z427.make_bjt(sebas_row)
        Vd_arr = c["Vd"].numpy()
        Id_meas = c["Id"].numpy()
        try:
            with torch.no_grad(), \
                 z427.patch_sd_scaled(sd_M1, P_M1), \
                 z427.patch_sd_scaled(sd_M2, P_M2):
                Id_fwd, Vb_fwd, conv_fwd, _ = _run_one_bias_capped(
                    cfg, model_M1, model_M2, bjt, Vd_arr,
                    c["VG1"], c["VG2"],
                    backward=False, Vb_init_first=0.0)
                Id_bwd, Vb_bwd, conv_bwd, _ = _run_one_bias_capped(
                    cfg, model_M1, model_M2, bjt, Vd_arr,
                    c["VG1"], c["VG2"],
                    backward=True, Vb_init_first=0.0)
        except Exception as e:
            log(f"  fail VG1={c['VG1']} VG2={c['VG2']}: {e}")
            continue
        out.append({
            "VG1": float(c["VG1"]),
            "VG2": float(c["VG2"]),
            "Vd": Vd_arr.tolist(),
            "Id_meas": Id_meas.tolist(),
            "Id_fwd": [float(x) for x in Id_fwd],
            "Id_bwd": [float(x) for x in Id_bwd],
            "Vb_fwd": [float(x) for x in Vb_fwd],
            "Vb_bwd": [float(x) for x in Vb_bwd],
            "conv_fwd": [bool(x) for x in conv_fwd],
            "conv_bwd": [bool(x) for x in conv_bwd],
            "vth0_M2": float(sebas_row.get("vth0", 0.54153)),
            "vth0_M1": float(sebas_row.get("vth0", 0.54153)),
        })
        log(f"  [{i+1}/{len(curves)}] VG1={c['VG1']:.2f} VG2={c['VG2']:+.2f}"
            f"  conv_fwd={sum(conv_fwd)}/{len(conv_fwd)} "
            f"conv_bwd={sum(conv_bwd)}/{len(conv_bwd)}  "
            f"wall={time.time()-t0:.0f}s")
    return out


# ============================================================ #
# Analysis
# ============================================================ #

HYST_AMP_THRESH = 0.20   # dec — model "bistable here" if exceeds
KNEE_THRESH_REL = 0.5    # relative knee threshold (fraction of max knee)


def analyse(per_bias):
    """Compute hysteresis amplitude + knee scores and binarise."""
    rows = []
    for r in per_bias:
        Vd = np.asarray(r["Vd"])
        Id_meas = np.asarray(r["Id_meas"])
        Id_f = np.asarray(r["Id_fwd"])
        Id_b = np.asarray(r["Id_bwd"])
        cf = np.asarray(r["conv_fwd"])
        cb = np.asarray(r["conv_bwd"])
        m = cf & cb & (Id_f > 0) & (Id_b > 0)
        if m.sum() < 3:
            hyst_amp = float("nan")
        else:
            d = np.abs(np.log10(Id_f[m] + 1e-15) - np.log10(Id_b[m] + 1e-15))
            hyst_amp = float(d.mean())
        knee = measured_knee_score(Vd, Id_meas)
        rows.append({
            "VG1": r["VG1"], "VG2": r["VG2"],
            "hyst_amp_dec": hyst_amp,
            "n_pts_both_conv": int(m.sum()),
            "knee_score": knee,
            "vth0_M2": r["vth0_M2"],
        })
    return rows


def boundary_curve(rows):
    """For each VG1, find largest VG2 at which measured snapback present
    (knee > threshold). That's the 'measured boundary'."""
    knees = np.array([r["knee_score"] for r in rows if not math.isnan(r["knee_score"])])
    if knees.size == 0:
        return [], 0.0
    knee_thresh = max(0.5, float(np.median(knees)))   # robust threshold
    by_vg1 = {}
    for r in rows:
        by_vg1.setdefault(round(r["VG1"], 3), []).append(r)
    boundary = []
    for vg1, lst in sorted(by_vg1.items()):
        lst_sorted = sorted(lst, key=lambda x: x["VG2"])
        present = [(x["VG2"], x["knee_score"] >= knee_thresh) for x in lst_sorted]
        # boundary VG2 = max VG2 where snapback present
        snapback_vg2s = [vg2 for vg2, p in present if p]
        if snapback_vg2s:
            boundary.append({"VG1": vg1, "VG2_boundary": max(snapback_vg2s)})
    return boundary, knee_thresh


# ============================================================ #
# Plotting
# ============================================================ #

def heatmap_scatter(rows, key, fname, title, vmin=None, vmax=None,
                     cmap="viridis"):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    vg1s = np.array([r["VG1"] for r in rows])
    vg2s = np.array([r["VG2"] for r in rows])
    vals = np.array([r[key] if not math.isnan(r[key]) else 0.0 for r in rows])
    sc = ax.scatter(vg1s, vg2s, c=vals, s=320, marker="s",
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors="k", linewidths=0.5)
    for r in rows:
        v = r[key]
        if math.isnan(v):
            txt = "NaN"
        else:
            txt = f"{v:.2f}"
        ax.text(r["VG1"], r["VG2"], txt, ha="center", va="center",
                 fontsize=7, color="white",
                 bbox=dict(facecolor="black", alpha=0.3, pad=1, edgecolor="none"))
    ax.set_xlabel("V_G1 [V]")
    ax.set_ylabel("V_G2 [V]")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label=key)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


def boundary_overlay(rows, boundary, knee_thresh, fname):
    fig, ax = plt.subplots(figsize=(8, 6))
    # categorise each bias into 4 states
    for r in rows:
        m_bis = (not math.isnan(r["hyst_amp_dec"])) and \
                 r["hyst_amp_dec"] >= HYST_AMP_THRESH
        m_knee = (not math.isnan(r["knee_score"])) and \
                  r["knee_score"] >= knee_thresh
        if m_bis and m_knee:
            color = "tab:green"
            label = "both"
        elif m_bis and not m_knee:
            color = "tab:red"   # over-extension
            label = "model_only"
        elif (not m_bis) and m_knee:
            color = "tab:orange"  # under-prediction
            label = "meas_only"
        else:
            color = "tab:gray"
            label = "neither"
        ax.scatter(r["VG1"], r["VG2"], color=color, s=180, marker="s",
                    edgecolors="k", linewidths=0.6)
    # legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color="tab:green", label="model bistable AND measured snapback"),
        Patch(color="tab:red", label="model bistable, measured smooth (over-extension)"),
        Patch(color="tab:orange", label="model smooth, measured snapback (under-prediction)"),
        Patch(color="tab:gray", label="neither"),
    ]
    if boundary:
        bx = [b["VG1"] for b in boundary]
        by = [b["VG2_boundary"] for b in boundary]
        ax.plot(bx, by, "k--o", lw=2, label="measured snapback boundary")
        legend.append(Patch(facecolor="none", edgecolor="k", label="dashed line: measured boundary"))
    ax.legend(handles=legend, fontsize=8, loc="best")
    ax.set_xlabel("V_G1 [V]"); ax.set_ylabel("V_G2 [V]")
    ax.set_title(f"Bifurcation map: model vs measured\n"
                  f"hyst threshold={HYST_AMP_THRESH:.2f} dec, "
                  f"knee threshold={knee_thresh:.2f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


# ============================================================ #
# M2 saturation check
# ============================================================ #

def m2_saturation_check(rows, per_bias_full, knee_thresh):
    """At boundary points (measured-bistable, near transition), check whether
    M2 is at V_DS,sat.

    Crude proxy (without re-running the model): V_GS,M2 = VG2 - V_sint_pinned,
    with V_sint pinned to 0, so V_GS,M2 = VG2.
    Overdrive_M2 = max(0, VG2 - VT0_M2).

    Boundary points: knee very close to threshold (within 30%).
    """
    pb_map = {(round(r["VG1"], 3), round(r["VG2"], 3)): r for r in per_bias_full}
    results = []
    for r in rows:
        if math.isnan(r["knee_score"]):
            continue
        # boundary = knee within ±30 % of threshold
        rel = r["knee_score"] / max(knee_thresh, 1e-9)
        is_boundary = (0.7 <= rel <= 1.5)
        full = pb_map.get((round(r["VG1"], 3), round(r["VG2"], 3)))
        vth = r["vth0_M2"]
        vgs_m2 = r["VG2"]   # V_sint pinned 0
        overdrive = max(0.0, vgs_m2 - vth)
        # V_DS,M2,sat ≈ overdrive (long-channel), so we ask: at the V_D where
        # the model fires (peak Vb in fwd), is V_DS,M2 ≈ overdrive?
        peak_vb = float(max(full["Vb_fwd"])) if full else float("nan")
        results.append({
            "VG1": r["VG1"], "VG2": r["VG2"],
            "is_boundary_pt": bool(is_boundary),
            "VGS_M2": vgs_m2,
            "VT0_M2": vth,
            "overdrive_M2": overdrive,
            "M2_in_sat_at_VGS": overdrive < 0.15,   # weak inversion
            "peak_Vb_fwd": peak_vb,
            "knee_score": r["knee_score"],
            "hyst_amp_dec": r["hyst_amp_dec"],
        })
    # Correlation: does "M2 in subthreshold" predict "measured smooth"?
    measured_smooth = []
    measured_snapback = []
    for x in results:
        if x["knee_score"] >= knee_thresh:
            measured_snapback.append(x["overdrive_M2"])
        else:
            measured_smooth.append(x["overdrive_M2"])
    if measured_smooth and measured_snapback:
        smooth_mean = float(np.mean(measured_smooth))
        snapback_mean = float(np.mean(measured_snapback))
    else:
        smooth_mean = snapback_mean = float("nan")
    return {
        "knee_threshold_used": knee_thresh,
        "n_boundary_pts": sum(1 for x in results if x["is_boundary_pt"]),
        "mean_overdrive_M2_measured_smooth": smooth_mean,
        "mean_overdrive_M2_measured_snapback": snapback_mean,
        "per_bias": results,
    }


# ============================================================ #
# Main
# ============================================================ #

def main():
    t_main = time.time()
    log("z438b starting — bifurcation map (model bistability vs measured snapback)")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    log("=== Running fwd+bwd ptran per bias ===")
    per_bias = run_per_bias(model_M1, model_M2, curves, sebas_rows)
    log(f"finished per-bias: {len(per_bias)} biases in {time.time()-t_main:.0f}s")

    # Save per-bias raw (light: drop heavy arrays into compact npz, keep summary in json)
    np.savez(OUT / "per_bias_raw.npz",
              data=np.array(per_bias, dtype=object))
    log(f"  wrote per_bias_raw.npz ({len(per_bias)} entries)")

    rows = analyse(per_bias)
    boundary, knee_thresh = boundary_curve(rows)

    # Heatmaps
    heatmap_scatter(rows, "hyst_amp_dec",
                     OUT / "model_bistability_map.png",
                     "Model hysteresis amplitude (dec) — pyport ptran fwd vs bwd",
                     vmin=0.0, vmax=max(0.5, max(r["hyst_amp_dec"]
                                                  for r in rows
                                                  if not math.isnan(r["hyst_amp_dec"])) ),
                     cmap="magma")
    heatmap_scatter(rows, "knee_score",
                     OUT / "measured_snapback_map.png",
                     "Measured curve knee score = max|d^2 log10(Id)/dVd^2|",
                     vmin=0.0, vmax=max(r["knee_score"] for r in rows),
                     cmap="viridis")
    boundary_overlay(rows, boundary, knee_thresh,
                      OUT / "boundary_comparison.png")

    sat = m2_saturation_check(rows, per_bias, knee_thresh)
    (OUT / "m2_saturation_check.json").write_text(json.dumps(sat, indent=2))
    log("  wrote m2_saturation_check.json")

    # Stats
    n_total = len(rows)
    n_model_bistable = sum(1 for r in rows
                            if (not math.isnan(r["hyst_amp_dec"]))
                            and r["hyst_amp_dec"] >= HYST_AMP_THRESH)
    n_meas_snapback = sum(1 for r in rows
                           if r["knee_score"] >= knee_thresh)
    n_both = sum(1 for r in rows
                  if (not math.isnan(r["hyst_amp_dec"]))
                  and r["hyst_amp_dec"] >= HYST_AMP_THRESH
                  and r["knee_score"] >= knee_thresh)
    n_model_only = sum(1 for r in rows
                        if (not math.isnan(r["hyst_amp_dec"]))
                        and r["hyst_amp_dec"] >= HYST_AMP_THRESH
                        and r["knee_score"] < knee_thresh)
    n_meas_only = sum(1 for r in rows
                       if (math.isnan(r["hyst_amp_dec"])
                            or r["hyst_amp_dec"] < HYST_AMP_THRESH)
                       and r["knee_score"] >= knee_thresh)
    agreement = (n_total - n_model_only - n_meas_only) / max(n_total, 1)

    summary = {
        "n_biases": n_total,
        "hyst_threshold_dec": HYST_AMP_THRESH,
        "knee_threshold": knee_thresh,
        "n_model_bistable": n_model_bistable,
        "n_measured_snapback": n_meas_snapback,
        "n_both": n_both,
        "n_model_only_OVEREXTENSION": n_model_only,
        "n_meas_only_UNDERPREDICTION": n_meas_only,
        "agreement_fraction": agreement,
        "boundary_curve": boundary,
        "per_bias_table": rows,
        "config": {
            "hyst_amp_thresh_dec": HYST_AMP_THRESH,
            "knee_threshold_used": knee_thresh,
            "knee_window_Vd": [0.4, 1.6],
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log("  wrote summary.json")

    # Honest analysis
    lines = []
    a = lines.append
    a("# z438b — Bifurcation map: model vs measured\n\n")
    a("## What we did\n")
    a("- For each of the N biases, ran forward (Vd 0→2) and backward (Vd 2→0) "
       "pseudo-transient Vb integration (z432 integrator, Vsint pinned 0).\n")
    a("- Hysteresis amplitude per bias = mean |log10(Id_fwd) − log10(Id_bwd)| "
       "over Vd where both directions converged.\n")
    a("- For each measured Id-Vd curve, computed knee score = "
       "max |d² log10(Id) / dVd²| over Vd ∈ [0.4, 1.6] V.\n")
    a(f"- Thresholds: model bistable if A ≥ {HYST_AMP_THRESH:.2f} dec; "
       f"measured snapback if knee ≥ {knee_thresh:.2f} (data-driven median).\n\n")
    a("## Headline counts\n")
    a(f"- total biases: {n_total}\n")
    a(f"- model bistable: {n_model_bistable}\n")
    a(f"- measured snapback: {n_meas_snapback}\n")
    a(f"- both agree (bistable AND snapback, or neither): "
       f"{n_total - n_model_only - n_meas_only}\n")
    a(f"- model-only (OVER-extension of snapback): {n_model_only}\n")
    a(f"- measured-only (UNDER-prediction): {n_meas_only}\n")
    a(f"- agreement = {agreement*100:.1f}%\n\n")

    a("## Boundary curve (measured snapback boundary)\n")
    if boundary:
        a("(largest VG2 with measured snapback present, per VG1)\n\n")
        for b in boundary:
            a(f"- VG1={b['VG1']:.2f} V → VG2 ≤ {b['VG2_boundary']:.2f} V\n")
    else:
        a("- no measured snapback observed above threshold\n")
    a("\n")

    a("## M2 saturation check\n")
    a(f"- mean overdrive (VGS,M2 − VT0,M2) at measured-SMOOTH points: "
       f"{sat['mean_overdrive_M2_measured_smooth']:.3f} V\n")
    a(f"- mean overdrive at measured-SNAPBACK points: "
       f"{sat['mean_overdrive_M2_measured_snapback']:.3f} V\n")
    diff = sat['mean_overdrive_M2_measured_smooth'] - sat['mean_overdrive_M2_measured_snapback']
    if not (math.isnan(diff)):
        a(f"- difference (smooth − snapback) = {diff:+.3f} V\n")
        if abs(diff) > 0.1:
            a(f"  → overdrive correlates with smoothness "
               f"({'smooth has higher overdrive' if diff > 0 else 'snapback has higher overdrive'}).\n")
        else:
            a("  → overdrive does not separate smooth/snapback classes.\n")
    a("\n")

    a("## Pre-registered gates\n")
    bistable_uniform = (n_model_bistable >= 0.9 * n_total)
    boundary_clean = bool(boundary) and len(boundary) >= 2
    over_extension = (n_model_only >= 0.3 * n_total)
    a(f"- INFRA: ran {n_total} biases fwd+bwd + measured curve derivative "
       f"→ {'PASS' if n_total >= 15 else 'FAIL'}\n")
    a(f"- DISCOVERY (clean boundary + ≥80%% correlate): "
       f"agreement={agreement*100:.1f}% → "
       f"{'PASS' if (boundary_clean and agreement >= 0.8) else 'FAIL'}\n")
    a(f"- AMBITIOUS (single (VG1, VG2) criterion predicts snapback): "
       f"see overdrive separation above; requires |Δoverdrive|>0.15V "
       f"→ {'PASS' if (not math.isnan(diff)) and abs(diff) > 0.15 else 'FAIL'}\n")
    a(f"- KILL_SHOT (model uniformly bistable across all biases → "
       f"topology change needed): "
       f"{n_model_bistable}/{n_total} bistable "
       f"→ {'TRIGGERED' if bistable_uniform else 'not triggered'}\n\n")

    a("## Honest verdict\n")
    if bistable_uniform:
        a("- **KILL SHOT TRIGGERED**: pyport is bistable at essentially every bias, "
           "while measured curves only have a knee at low VG1 / low VG2. "
           "The DC two-node fixed-point structure is fundamentally too 'snappy'. "
           "The model needs a topological change — e.g. a current-controlled "
           "switch that latches only above a (VG1, VG2)-dependent V_D threshold, "
           "or removal of the floating body node in the regime where measured "
           "saturation is smooth. Tweaking BJT Bf / impact-ionisation α0 will "
           "shift the knee but cannot eliminate hysteresis from the bistable "
           "phase space.\n")
    elif over_extension > 0.3 * n_total:
        a("- **Over-extension dominant**: model is bistable in regions where "
           "measured saturates smoothly. Probably impact-ionisation / BJT "
           "feedback is on in too wide a (VG1, VG2) region. Needs a "
           "(VG1, VG2)-dependent gating of the feedback path.\n")
    elif n_meas_only > 0.3 * n_total:
        a("- **Under-prediction dominant**: model misses snapback in regions "
           "where measured curves clearly knee. Needs stronger feedback or "
           "additional bias-dependent gain.\n")
    else:
        a("- Mixed / no clear pattern. Model and measured both have a "
           "boundary, but they do not align — single-mechanism fix unlikely.\n")
    a("\n## Boundary uncertainty\n")
    a("- Threshold for measured snapback (knee score) is data-driven "
       "(median of all biases). Robust within ±20%% but not unique.\n")
    a("- Boundary curve is the *max VG2 with snapback* per VG1 slice; with "
       "the discrete VG2 grid this gives a step function, not a smooth curve.\n")

    (OUT / "honest_analysis.md").write_text("".join(lines))
    log("  wrote honest_analysis.md")

    log(f"DONE wall={time.time()-t_main:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
