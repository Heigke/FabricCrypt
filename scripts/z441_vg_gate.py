"""z441 — S30: V_G1 − V_G2 sigmoid gate on body-charging current.

Implements the S27 bifurcation map finding: measured Sebas data has a
sharp snapback boundary at V_G1 − V_G2 ≥ ~0.20 V. Above the boundary
the body charges, the parasitic NPN fires, and the cell snaps back.
Below: smooth saturation, no snapback. The pyport model has the same
body-charging machinery (Mario Ipos PWL + Iii→body) active EVERYWHERE in
(V_G1, V_G2) space, so it over-extends snapback.

Direct fix (single parameter, surgical): gate the body-charging current
by a sigmoid in (V_G1 − V_G2):

    gate = sigmoid((V_G1 − V_G2 − thr) / width)

Applied to (a) Mario Ipos injection at body node, and (b) the bulk
fraction of Iii routed to the body (iii_to_body_factor). The lateral
fraction (Ib_lat_pair driving the BJT) is NOT gated — it follows the
parasitic-NPN physics.

Grid: thr ∈ {0.15, 0.20, 0.25, 0.30}, width ∈ {0.02, 0.05, 0.10}
                                                    → 12 conditions.

Each condition: cell-wide RMSE (pseudo-transient FORWARD only, N_STEPS=300,
matching z438b's reduced budget). Pick best by cell_rmse_dec; rerun
fwd+bwd at full N_STEPS for the best (mirroring z432). Then re-do the
bifurcation map at the best gate.

Reference: z432 backward = 1.027 dec, z430 V_SINT_PIN = 1.62 dec.
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
OUT = ROOT / "results/z441_vg_gate"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


# ---------- Reuse infrastructure from z427/z429/z432/z438b ----------
_s427 = _ilu.spec_from_file_location("z427", ROOT / "scripts/z427_vsint_fix.py")
z427 = _ilu.module_from_spec(_s427); _s427.loader.exec_module(z427)
_s429 = _ilu.spec_from_file_location("z429", ROOT / "scripts/z429_multisolver_debug.py")
z429 = _ilu.module_from_spec(_s429); _s429.loader.exec_module(z429)
_s432 = _ilu.spec_from_file_location("z432", ROOT / "scripts/z432_pseudotransient.py")
z432 = _ilu.module_from_spec(_s432); _s432.loader.exec_module(z432)


# ============================================================ #
# Grid runner — quick FORWARD-only screen
# ============================================================ #

N_STEPS_GRID = 300        # ~3x faster than default 800
N_STEPS_FULL = 800        # for the chosen-best condition (mirrors z432)
N_MIN_STEPS_GRID = 60     # match z438b
z432.N_MIN_STEPS = N_MIN_STEPS_GRID

# z432 backward baseline (no gate)
Z432_BWD_CELL = 1.026861976331113
Z432_FWD_CELL = 1.34906163370139
Z430_VSINT_PIN_CELL = 1.6187161900853293


def _run_one_bias_capped(cfg, model_M1, model_M2, bjt, Vd_arr, VG1, VG2,
                          backward=False, Vb_init_first=0.0,
                          n_steps_cap=N_STEPS_GRID):
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


def run_cellwide_gate(name, model_M1, model_M2, curves, sebas_rows,
                      gate_thr: float, gate_width: float,
                      direction: str = "forward",
                      n_steps: int = N_STEPS_GRID):
    extra = dict(use_vg_gate=True,
                 vg_gate_thr=float(gate_thr),
                 vg_gate_width=float(gate_width))
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, extra)
    log_eps = 1e-15
    per_bias = []
    fails = 0
    t0 = time.time()
    vb_max_overall = -1e30
    for c in curves:
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
                Id_pred, Vb_list, conv_list, niter_list = _run_one_bias_capped(
                    cfg, model_M1, model_M2, bjt, Vd_arr,
                    c["VG1"], c["VG2"],
                    backward=(direction == "backward"),
                    Vb_init_first=0.0,
                    n_steps_cap=n_steps)
        except Exception as e:
            fails += 1
            log(f"  {name} fail VG1={c['VG1']} VG2={c['VG2']}: {e}")
            continue
        Id_pred_t = torch.tensor(Id_pred, dtype=torch.float64)
        conv_t = torch.tensor(conv_list)
        if not conv_t.any():
            fails += 1
            continue
        log_p = torch.log10(Id_pred_t + log_eps)
        log_m = torch.log10(c["Id"] + log_eps)
        sq = (log_p - log_m) ** 2
        rmse = float(torch.sqrt(sq[conv_t].mean()))
        vb_max = float(max(Vb_list))
        vb_max_overall = max(vb_max_overall, vb_max)
        per_bias.append({
            "VG1": float(c["VG1"]),
            "VG2": float(c["VG2"]),
            "log_rmse": rmse,
            "vb_max": vb_max,
            "n_conv": int(conv_t.sum()),
            "n_pts": len(Vd_arr),
            "Vd": Vd_arr.tolist(),
            "Id_meas": Id_meas.tolist(),
            "Id_pred": [float(x) for x in Id_pred],
            "Vb": [float(x) for x in Vb_list],
            "converged": [bool(x) for x in conv_list],
        })
    cell_sq = sum(r["log_rmse"]**2 for r in per_bias)
    cell_n = len(per_bias)
    cell = math.sqrt(cell_sq / cell_n) if cell_n else float("inf")
    per_branch = {}
    for r in per_bias:
        b = f"VG1_{r['VG1']:.1f}"
        per_branch.setdefault(b, {"sq": 0.0, "n": 0})
        per_branch[b]["sq"] += r["log_rmse"]**2
        per_branch[b]["n"] += 1
    per_branch_rmse = {b: math.sqrt(v["sq"]/v["n"]) for b, v in per_branch.items()}
    total_pts = sum(r["n_pts"] for r in per_bias)
    total_conv = sum(r["n_conv"] for r in per_bias)
    conv_rate = total_conv / max(total_pts, 1)
    log(f"  {name}({direction}) thr={gate_thr:.2f} w={gate_width:.2f}: "
        f"cell={cell:.3f} per_branch={ {k:round(v,3) for k,v in per_branch_rmse.items()} } "
        f"Vb_max={vb_max_overall:.3f} conv_rate={conv_rate*100:.1f}% "
        f"fails={fails} wall={time.time()-t0:.0f}s")
    return {
        "thr": float(gate_thr),
        "width": float(gate_width),
        "direction": direction,
        "cell_rmse_dec": cell,
        "per_branch_rmse_dec": per_branch_rmse,
        "n_biases_evaluated": cell_n,
        "vb_max_overall": vb_max_overall,
        "convergence_rate": conv_rate,
        "fails": fails,
        "wall_sec": round(time.time() - t0, 1),
        "per_bias": per_bias,
    }


# ============================================================ #
# Boundary classification accuracy
# ============================================================ #
# Knee score copied from z438b
def measured_knee_score(Vd, Id, vd_lo=0.4, vd_hi=1.6, eps=1e-15):
    Vd = np.asarray(Vd, float); Id = np.asarray(Id, float)
    if Vd.size < 5: return 0.0
    log_id = np.log10(np.maximum(np.abs(Id), eps))
    d2 = np.zeros_like(log_id)
    for i in range(1, len(Vd) - 1):
        h1 = Vd[i] - Vd[i - 1]; h2 = Vd[i + 1] - Vd[i]
        if h1 <= 0 or h2 <= 0: continue
        d2[i] = 2 * (h2 * log_id[i - 1] - (h1 + h2) * log_id[i]
                     + h1 * log_id[i + 1]) / (h1 * h2 * (h1 + h2))
    mask = (Vd >= vd_lo) & (Vd <= vd_hi)
    if mask.sum() < 2: mask = np.ones_like(Vd, dtype=bool)
    return float(np.max(np.abs(d2[mask])))


def boundary_classification(per_bias_fwd, per_bias_bwd=None,
                            knee_thresh=None, hyst_thresh=0.20):
    """For each bias, compute: (a) measured snapback present (knee >
    thresh), (b) model bistable (|fwd-bwd| > hyst_thresh, or, if no bwd
    given, use V_G1−V_G2 ≥ thr as the model's gate-driven prediction).
    Returns classification: (TP, FP, TN, FN, accuracy) on the BINARY
    decision of "does this bias have snapback in the measured data?".

    Since we may run only forward sweeps in the grid, we use a simpler
    metric: predict snapback if model V_b reaches > 0.5 V at high Vd
    (BJT firing) AND model log10|Id| has knee > 0.5*max model knee.
    """
    # Knee threshold from measured data
    knees_meas = [measured_knee_score(r["Vd"], r["Id_meas"])
                  for r in per_bias_fwd]
    if knee_thresh is None:
        knee_thresh = max(0.5, float(np.median([k for k in knees_meas if k > 0])
                                     if any(k > 0 for k in knees_meas) else 0.5))
    # Index bwd by (vg1, vg2)
    bwd_index = {}
    if per_bias_bwd is not None:
        bwd_index = {(round(r["VG1"], 3), round(r["VG2"], 3)): r
                     for r in per_bias_bwd}
    rows = []
    TP = FP = TN = FN = 0
    for r, k_m in zip(per_bias_fwd, knees_meas):
        meas_snap = (k_m > knee_thresh)
        Id_f = np.asarray(r["Id_pred"], float)
        Vb_f = np.asarray(r["Vb"], float)
        k_model = measured_knee_score(r["Vd"], Id_f)
        # Decide "model has snapback" — two criteria, either:
        # (a) hysteresis amplitude > hyst_thresh (if bwd available)
        # (b) model knee > knee_thresh AND Vb_max > 0.5 V (BJT firing)
        br = bwd_index.get((round(r["VG1"], 3), round(r["VG2"], 3)))
        if br is not None:
            cf = np.asarray(r["converged"], bool)
            cb = np.asarray(br["converged"], bool)
            Id_b = np.asarray(br["Id_pred"], float)
            m = cf & cb & (Id_f > 0) & (Id_b > 0)
            if m.sum() >= 3:
                d = np.abs(np.log10(Id_f[m] + 1e-15)
                           - np.log10(Id_b[m] + 1e-15))
                hyst_amp = float(d.mean())
                model_snap = (hyst_amp > hyst_thresh)
            else:
                hyst_amp = float("nan")
                model_snap = (k_model > knee_thresh
                              and float(np.max(Vb_f)) > 0.5)
        else:
            hyst_amp = float("nan")
            model_snap = (k_model > knee_thresh
                          and float(np.max(Vb_f)) > 0.5)
        # Confusion
        if meas_snap and model_snap: TP += 1
        elif meas_snap and not model_snap: FN += 1
        elif (not meas_snap) and model_snap: FP += 1
        else: TN += 1
        rows.append({
            "VG1": float(r["VG1"]),
            "VG2": float(r["VG2"]),
            "VG1_minus_VG2": float(r["VG1"]) - float(r["VG2"]),
            "knee_meas": float(k_m),
            "knee_model": float(k_model),
            "vb_max_model": float(np.max(Vb_f)),
            "hyst_amp_dec": hyst_amp,
            "meas_snap": bool(meas_snap),
            "model_snap": bool(model_snap),
            "match": bool(meas_snap == model_snap),
        })
    n = TP + FP + TN + FN
    acc = (TP + TN) / max(n, 1)
    return {
        "knee_thresh": float(knee_thresh),
        "hyst_thresh": float(hyst_thresh),
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "accuracy": acc,
        "n_meas_snap": TP + FN,
        "n_model_snap": TP + FP,
        "per_bias": rows,
    }


# ============================================================ #
# Plotting
# ============================================================ #

def overlay_plot(VG1_target, fwd_z441_pb, z432_pb, fname):
    """Overlay measured + z432 fwd + z441-best fwd."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    def rows_at(pb, vg1):
        return {round(r["VG2"], 3): r for r in pb if abs(r["VG1"] - vg1) < 1e-3}
    z441_rows = rows_at(fwd_z441_pb, VG1_target)
    z432_rows = rows_at(z432_pb, VG1_target)
    vg2_vals = sorted(set(z441_rows.keys()) | set(z432_rows.keys()))
    if not vg2_vals:
        plt.close(fig); return
    chosen = [vg2_vals[0],
              vg2_vals[len(vg2_vals)//2],
              vg2_vals[-1]] if len(vg2_vals) >= 3 else vg2_vals
    for ax, vg2 in zip(axes, chosen):
        meas_rec = z441_rows.get(vg2) or z432_rows.get(vg2)
        if meas_rec is None:
            ax.set_title(f"VG2={vg2:.2f} (no data)"); continue
        ax.plot(meas_rec["Vd"], meas_rec["Id_meas"], "k-",
                lw=2.5, label="measured")
        if vg2 in z432_rows:
            r = z432_rows[vg2]
            ax.plot(r["Vd"], r["Id_pred"], "--", lw=1.5,
                    color="tab:red", label="z432 (no gate)")
        if vg2 in z441_rows:
            r = z441_rows[vg2]
            ax.plot(r["Vd"], r["Id_pred"], "--", lw=1.5,
                    color="tab:blue", label="z441 (gate ON)")
        ax.set_yscale("log")
        ax.set_xlabel("V_D [V]")
        ax.set_title(f"VG1={VG1_target:.1f}  VG2={vg2:+.2f}  "
                     f"ΔVG={VG1_target - vg2:+.2f}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("|I_D| [A]")
    fig.suptitle(f"z441 V_G1−V_G2 gate vs z432 baseline @ VG1={VG1_target:.1f}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


def bifurcation_map_plot(boundary_class, fname):
    """Re-do the (VG1, VG2) boundary plot with z441-gated model."""
    rows = boundary_class["per_bias"]
    fig, ax = plt.subplots(figsize=(7, 6))
    from matplotlib.patches import Patch
    for r in rows:
        m_meas = r["meas_snap"]; m_mod = r["model_snap"]
        if m_meas and m_mod:
            c = "tab:green"
        elif m_meas and not m_mod:
            c = "tab:orange"
        elif (not m_meas) and m_mod:
            c = "tab:red"
        else:
            c = "tab:gray"
        ax.scatter(r["VG1"], r["VG2"], s=90, color=c, edgecolor="black",
                   linewidth=0.5)
    # diagonal V_G1−V_G2 = thr
    thr = boundary_class.get("gate_thr", 0.20)
    vg1_line = np.array([0.1, 0.7])
    ax.plot(vg1_line, vg1_line - thr, "k--",
            label=f"V_G1−V_G2 = {thr:.2f}V (gate centre)")
    ax.set_xlabel("V_G1 [V]"); ax.set_ylabel("V_G2 [V]")
    ax.set_title(f"z441 bifurcation map (gate thr={thr:.2f}V) — "
                 f"acc={boundary_class['accuracy']*100:.0f}%")
    legend = [
        Patch(color="tab:green", label="both: snapback"),
        Patch(color="tab:orange", label="meas-only (UNDERPRED)"),
        Patch(color="tab:red", label="model-only (OVEREXTEND)"),
        Patch(color="tab:gray", label="both: smooth"),
    ]
    ax.legend(handles=legend, loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


# ============================================================ #
# Honest analysis writer
# ============================================================ #

def write_honest_analysis(summary, fname):
    s = summary
    best = s["BEST"]
    bc = s["BEST_BOUNDARY_CLASSIFICATION"]
    lines = []
    lines.append("# z441 — V_G1−V_G2 Sigmoid Gate, Honest Analysis\n")
    lines.append(f"\nDate: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("\n## Setup\n")
    lines.append(f"- Single-parameter fix: gate = sigmoid((V_G1 − V_G2 − thr)/width)\n")
    lines.append(f"- Applied to Mario Ipos (body-injection) and Iii→body bulk fraction\n")
    lines.append(f"- Grid: thr ∈ {{0.15, 0.20, 0.25, 0.30}}, width ∈ {{0.02, 0.05, 0.10}}\n")
    lines.append(f"- N_STEPS grid = {N_STEPS_GRID}; chosen-best rerun with N_STEPS = {N_STEPS_FULL}\n")
    lines.append("\n## Reference baselines\n")
    lines.append(f"- z430 V_SINT_PIN cell_rmse_dec = {Z430_VSINT_PIN_CELL:.3f}\n")
    lines.append(f"- z432 PTRAN FORWARD cell_rmse_dec = {Z432_FWD_CELL:.3f}\n")
    lines.append(f"- z432 PTRAN BACKWARD cell_rmse_dec = {Z432_BWD_CELL:.3f}\n")
    lines.append("\n## Grid results (FORWARD sweep)\n")
    lines.append("| thr | width | cell_rmse | conv_rate | wall |\n")
    lines.append("|-----|-------|-----------|-----------|------|\n")
    for g in s["GRID"]:
        lines.append(f"| {g['thr']:.2f} | {g['width']:.2f} | "
                     f"{g['cell_rmse_dec']:.3f} | "
                     f"{g['convergence_rate']*100:.1f}% | "
                     f"{g['wall_sec']:.0f}s |\n")
    lines.append("\n## Best gate\n")
    lines.append(f"- thr={best['thr']:.2f}, width={best['width']:.2f}\n")
    lines.append(f"- cell_rmse_dec (FWD, full N_STEPS) = {best['fwd_full']['cell_rmse_dec']:.3f}\n")
    if "bwd_full" in best:
        lines.append(f"- cell_rmse_dec (BWD, full N_STEPS) = {best['bwd_full']['cell_rmse_dec']:.3f}\n")
    lines.append(f"- per_branch (FWD): {best['fwd_full']['per_branch_rmse_dec']}\n")
    lines.append("\n## Boundary classification (best gate)\n")
    lines.append(f"- accuracy = {bc['accuracy']*100:.0f}%  (TP={bc['TP']}, FP={bc['FP']}, "
                 f"TN={bc['TN']}, FN={bc['FN']})\n")
    lines.append(f"- knee threshold = {bc['knee_thresh']:.2f}\n")
    lines.append(f"- n biases with measured snapback = {bc['n_meas_snap']}\n")
    lines.append(f"- n biases with model snapback = {bc['n_model_snap']}\n")
    lines.append("\n## Gates (pre-registered)\n")
    g = s["GATES"]
    for k, v in g.items():
        lines.append(f"- {k}: {'PASS' if v else 'FAIL'}\n")
    lines.append("\n## Per-bias kill-shot check (VG1=0.6 snapback magnitude)\n")
    pb = bc["per_bias"]
    vg1_06 = [r for r in pb if abs(r["VG1"] - 0.6) < 1e-3]
    if vg1_06:
        lines.append("| VG2 | ΔVG | knee_meas | knee_model | vb_max | match |\n")
        lines.append("|-----|-----|-----------|------------|--------|-------|\n")
        for r in sorted(vg1_06, key=lambda x: x["VG2"]):
            lines.append(f"| {r['VG2']:+.2f} | {r['VG1_minus_VG2']:+.2f} | "
                         f"{r['knee_meas']:.1f} | {r['knee_model']:.1f} | "
                         f"{r['vb_max_model']:.3f} | "
                         f"{'YES' if r['match'] else 'NO'} |\n")
    fname.write_text("".join(lines))
    log(f"  wrote {fname.name}")


# ============================================================ #
# Main
# ============================================================ #

def main():
    t_main = time.time()
    log("z441 starting — V_G1 − V_G2 sigmoid gate on body-charging current")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    # 4×3 grid
    THRS = [0.15, 0.20, 0.25, 0.30]
    WIDTHS = [0.02, 0.05, 0.10]
    grid_results = []
    log(f"=== Grid {len(THRS)}×{len(WIDTHS)} ({len(THRS)*len(WIDTHS)} cond) — "
        f"FORWARD only, N_STEPS={N_STEPS_GRID} ===")
    for thr in THRS:
        for w in WIDTHS:
            r = run_cellwide_gate("GRID", model_M1, model_M2, curves,
                                  sebas_rows, gate_thr=thr, gate_width=w,
                                  direction="forward",
                                  n_steps=N_STEPS_GRID)
            grid_results.append(r)
            # Persist progress
            (OUT / "_grid_progress.json").write_text(json.dumps(
                [{k: v for k, v in g.items() if k != "per_bias"}
                 for g in grid_results], indent=2, default=str))

    # Pick best by cell_rmse_dec, breaking ties by higher conv_rate
    grid_sorted = sorted(grid_results,
                         key=lambda r: (r["cell_rmse_dec"],
                                        -r["convergence_rate"]))
    best_grid = grid_sorted[0]
    log(f"=== Best grid: thr={best_grid['thr']:.2f} w={best_grid['width']:.2f} "
        f"cell={best_grid['cell_rmse_dec']:.3f} ===")

    # Full re-run (forward + backward) at full N_STEPS
    log("=== Best gate full re-run: FORWARD N_STEPS=800 ===")
    z432.N_MIN_STEPS = 100  # z432 default
    fwd_full = run_cellwide_gate("BEST", model_M1, model_M2, curves, sebas_rows,
                                 gate_thr=best_grid["thr"],
                                 gate_width=best_grid["width"],
                                 direction="forward",
                                 n_steps=N_STEPS_FULL)
    log("=== Best gate full re-run: BACKWARD N_STEPS=800 ===")
    bwd_full = run_cellwide_gate("BEST", model_M1, model_M2, curves, sebas_rows,
                                 gate_thr=best_grid["thr"],
                                 gate_width=best_grid["width"],
                                 direction="backward",
                                 n_steps=N_STEPS_FULL)

    # Boundary classification (need fwd+bwd for hysteresis amplitude)
    boundary = boundary_classification(fwd_full["per_bias"],
                                       bwd_full["per_bias"])
    boundary["gate_thr"] = best_grid["thr"]
    boundary["gate_width"] = best_grid["width"]

    # Overlay plots
    for vg1, suffix in [(0.2, "0p2"), (0.4, "0p4"), (0.6, "0p6")]:
        # Load z432 fwd per_bias for overlay
        try:
            with open(ROOT / "results/z432_pseudotransient/summary.json") as f:
                z432_sum = json.load(f)
            # z432 per_bias structure is inside "PTRAN_FORWARD" only if collected
            # But summary stores it only via separate keys; recompute or use
            # fwd_full as the model and skip z432 if not available
            z432_pb = z432_sum.get("PTRAN_FORWARD", {}).get("per_bias", [])
        except Exception:
            z432_pb = []
        overlay_plot(vg1, fwd_full["per_bias"], z432_pb,
                     OUT / f"best_overlay_VG1_{suffix}.png")

    # Bifurcation map
    bifurcation_map_plot(boundary, OUT / "bifurcation_map_z441.png")

    # KILL SHOT check: VG1=0.6 snapback magnitude reduction
    # Compare knee at VG1=0.6 low VG2 (e.g., VG2 ≤ 0.2) between model and meas
    kill_rows = [r for r in boundary["per_bias"]
                 if abs(r["VG1"] - 0.6) < 1e-3 and r["VG2"] <= 0.2]
    if kill_rows:
        avg_knee_meas = float(np.mean([r["knee_meas"] for r in kill_rows]))
        avg_knee_model = float(np.mean([r["knee_model"] for r in kill_rows]))
        kill_dec_drop = math.log10(max(avg_knee_meas, 1e-3)) \
                       - math.log10(max(avg_knee_model, 1e-3))
    else:
        avg_knee_meas = avg_knee_model = float("nan")
        kill_dec_drop = float("nan")

    # Gates (pre-registered)
    cellwide_best = min(fwd_full["cell_rmse_dec"], bwd_full["cell_rmse_dec"])
    gates = {
        "INFRA_grid_12_conditions": len(grid_results) == 12,
        "DISCOVERY_boundary_acc_85pct_AND_cell_lt_1_0":
            (boundary["accuracy"] >= 0.85) and (cellwide_best < 1.0),
        "AMBITIOUS_cell_lt_0p7_AND_boundary_acc_90pct":
            (cellwide_best < 0.7) and (boundary["accuracy"] >= 0.90),
        "KILL_SHOT_gate_breaks_VG1_0p6_snapback":
            (not math.isnan(kill_dec_drop)) and (kill_dec_drop > 0.5),
    }

    summary = {
        "GRID": [{k: v for k, v in g.items() if k != "per_bias"}
                 for g in grid_results],
        "BEST": {
            "thr": best_grid["thr"],
            "width": best_grid["width"],
            "fwd_full": {k: v for k, v in fwd_full.items() if k != "per_bias"},
            "bwd_full": {k: v for k, v in bwd_full.items() if k != "per_bias"},
        },
        "BEST_BOUNDARY_CLASSIFICATION": boundary,
        "REFERENCE": {
            "z430_v_sint_pin_cell_rmse_dec": Z430_VSINT_PIN_CELL,
            "z432_fwd_cell_rmse_dec": Z432_FWD_CELL,
            "z432_bwd_cell_rmse_dec": Z432_BWD_CELL,
        },
        "KILL_SHOT_VG1_0p6": {
            "avg_knee_measured": avg_knee_meas,
            "avg_knee_model": avg_knee_model,
            "dec_drop_meas_minus_model": kill_dec_drop,
        },
        "GATES": gates,
        "DELTAS_VS_Z432_BWD": {
            "fwd_full":  Z432_BWD_CELL - fwd_full["cell_rmse_dec"],
            "bwd_full": Z432_BWD_CELL - bwd_full["cell_rmse_dec"],
        },
        "CONFIG": {
            "N_STEPS_GRID": N_STEPS_GRID,
            "N_STEPS_FULL": N_STEPS_FULL,
            "thresholds": THRS,
            "widths": WIDTHS,
        },
        "wall_total_sec": round(time.time() - t_main, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2,
                                                  default=str))
    # Also persist per_bias separately for later analysis
    (OUT / "best_per_bias_fwd.json").write_text(json.dumps(
        fwd_full["per_bias"], indent=2, default=str))
    (OUT / "best_per_bias_bwd.json").write_text(json.dumps(
        bwd_full["per_bias"], indent=2, default=str))

    write_honest_analysis(summary, OUT / "honest_analysis.md")
    log(f"=== DONE wall={time.time()-t_main:.0f}s ===")
    log(f"  cell_fwd={fwd_full['cell_rmse_dec']:.3f}  "
        f"cell_bwd={bwd_full['cell_rmse_dec']:.3f}  "
        f"boundary_acc={boundary['accuracy']*100:.0f}%")
    log(f"  gates: " + str(gates))


if __name__ == "__main__":
    main()
