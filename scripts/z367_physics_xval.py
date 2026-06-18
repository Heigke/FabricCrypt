"""z367 — R-48: Physics-based cross-validation falsifier for R-46.

R-46 found per-VG1 params giving 0.965 dec cell-wide median:
  VG1=0.20: Bf=1890, iii=1.84, log10Rs=9.17
  VG1=0.40: Bf=1092, iii=1.52, log10Rs=9.90
  VG1=0.60: Bf=418,  iii=0.90, log10Rs=6.78

QUESTION: real physics (smooth VG1-dependence reflecting regime change)
or curve-fitting (per-VG1 free knobs eating dataset)?

METHOD: Leave-one-out — fit smooth scaling law on 2 branches, predict
the 3rd, evaluate cell-wide dec on the held-out branch only.

Compares to R-46's per-VG1 fit dec computed on this same eval.

Pre-registered gates (per held-out branch):
  PHYSICS PASS : held-out dec  <= 1.30 * R-46_fit_dec
  CURVE-FIT    : held-out dec  >= 2.00 * R-46_fit_dec
  IN-BETWEEN   : partial-physics

Output: results/z367_physics_xval/
  params_vs_VG1.png
  xval_results.json
  physics_verdict.md
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys, json, math, time, importlib.util
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
OUT = ROOT / "results/z367_physics_xval"; OUT.mkdir(parents=True, exist_ok=True)

# Reuse machinery from z365
_z365_spec = importlib.util.spec_from_file_location("z365", ROOT / "scripts/z365_perVG1_bbo.py")
z365 = importlib.util.module_from_spec(_z365_spec); _z365_spec.loader.exec_module(z365)

# R-46 best (eval 94)
R46_PARAMS = {
    0.20: {"Bf": 1889.8806320503354, "iii": 1.844675445044413, "log10Rs": 9.172216925770044},
    0.40: {"Bf": 1092.272187024355,  "iii": 1.5151811846066265, "log10Rs": 9.898274548351765},
    0.60: {"Bf": 417.62741417624056, "iii": 0.9035713450051843, "log10Rs": 6.784622445702553},
}
VG1S = [0.20, 0.40, 0.60]


def evaluate_branch(cfg, M1, M2, bjt, sebas_rows, curves, forward_2t,
                    vg1_target, Bf, iii, Rs):
    """Eval log10-RMSE on all curves with VG1=vg1_target, using given params."""
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    bjt.Bf = float(Bf); cfg.iii_body_gain = float(iii); cfg.vnwell_Rs = float(Rs)
    rmses = []
    for c in curves:
        if round(c["VG1"], 2) != round(vg1_target, 2):
            continue
        Vd = torch.tensor(c["Vd"], dtype=torch.float64)
        row, _ = z365.find_or_impute_row(sebas_rows, c["VG1"], c["VG2"])
        P_M1, P_M2 = z365.make_overrides(row)
        try:
            with z365.patch_sd_scaled(sd_M1, P_M1), z365.patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd,
                                 VG1=torch.tensor(c["VG1"], dtype=torch.float64),
                                 VG2=torch.tensor(c["VG2"], dtype=torch.float64),
                                 warm_start=True)
            Id_pred = np.abs(out["Id"].detach().cpu().numpy())
            mask = (c["Id"] > 1e-15) & (Id_pred > 1e-15) & np.isfinite(Id_pred)
            if mask.sum() >= 3:
                rmse = float(np.sqrt(np.mean(
                    (np.log10(Id_pred[mask]) - np.log10(c["Id"][mask])) ** 2)))
            else:
                rmse = float("nan")
        except Exception:
            rmse = float("nan")
        rmses.append({"VG1": c["VG1"], "VG2": c["VG2"], "rmse": rmse})
    valid = [r["rmse"] for r in rmses if not math.isnan(r["rmse"])]
    return float(np.median(valid)) if valid else float("nan"), rmses


def linear_extrap(vg_train, y_train, vg_target):
    """Fit y = a + b * VG on 2 points; predict at vg_target."""
    (x1, x2), (y1, y2) = vg_train, y_train
    b = (y2 - y1) / (x2 - x1)
    a = y1 - b * x1
    return a + b * vg_target


def main():
    t0 = time.time()
    print("[z367] R-48 physics-based cross-validation falsifier", flush=True)
    cfg, M1, M2, bjt = z365.build_pyport_base()
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    sebas_rows = z365.load_sebas_params()
    curves = z365.load_curves()
    print(f"[z367] loaded {len(curves)} curves", flush=True)

    # === 1. Monotonicity check ===
    Bf_vals  = [R46_PARAMS[v]["Bf"]      for v in VG1S]
    iii_vals = [R46_PARAMS[v]["iii"]     for v in VG1S]
    Rs_vals  = [R46_PARAMS[v]["log10Rs"] for v in VG1S]

    def monotonic(vals):
        diffs = np.diff(vals)
        return bool(np.all(diffs > 0) or np.all(diffs < 0))

    monot = {
        "Bf":      monotonic(Bf_vals),
        "iii":     monotonic(iii_vals),
        "log10Rs": monotonic(Rs_vals),
    }
    print(f"[z367] Bf      = {Bf_vals} monot={monot['Bf']}", flush=True)
    print(f"[z367] iii     = {iii_vals} monot={monot['iii']}", flush=True)
    print(f"[z367] log10Rs = {Rs_vals} monot={monot['log10Rs']}", flush=True)

    # === 2. Per-VG1 baseline: R-46 fit dec on each branch ===
    print("[z367] computing R-46 baseline per-branch dec...", flush=True)
    baseline_dec = {}
    for v in VG1S:
        p = R46_PARAMS[v]
        med, _ = evaluate_branch(cfg, M1, M2, bjt, sebas_rows, curves, forward_2t,
                                 v, p["Bf"], p["iii"], 10.0**p["log10Rs"])
        baseline_dec[v] = med
        print(f"  R-46 fit  VG1={v}: dec={med:.4f}", flush=True)

    # === 3. LOOCV predictions ===
    print("[z367] leave-one-out predictions (linear in VG1)...", flush=True)
    xval = []
    for held in VG1S:
        train = [v for v in VG1S if v != held]
        Bf_pred  = linear_extrap(train, [R46_PARAMS[v]["Bf"]      for v in train], held)
        iii_pred = linear_extrap(train, [R46_PARAMS[v]["iii"]     for v in train], held)
        Rs_pred  = linear_extrap(train, [R46_PARAMS[v]["log10Rs"] for v in train], held)
        # Clamp to BBO bounds
        Bf_pred_c  = float(np.clip(Bf_pred, 50.0, 2000.0))
        iii_pred_c = float(np.clip(iii_pred, 0.05, 2.0))
        Rs_pred_c  = float(np.clip(Rs_pred, 6.0, 10.0))
        print(f"  hold VG1={held}: predicted Bf={Bf_pred:.1f} (clip {Bf_pred_c:.1f}), "
              f"iii={iii_pred:.3f} (clip {iii_pred_c:.3f}), "
              f"log10Rs={Rs_pred:.3f} (clip {Rs_pred_c:.3f})", flush=True)
        pred_med, _ = evaluate_branch(cfg, M1, M2, bjt, sebas_rows, curves, forward_2t,
                                      held, Bf_pred_c, iii_pred_c, 10.0**Rs_pred_c)
        ratio = pred_med / baseline_dec[held] if baseline_dec[held] > 0 else float("nan")
        if ratio <= 1.30:    verdict = "PHYSICS_PASS"
        elif ratio >= 2.00:  verdict = "CURVE_FIT"
        else:                verdict = "PARTIAL"
        print(f"  hold VG1={held}: fit_dec={baseline_dec[held]:.4f}  "
              f"pred_dec={pred_med:.4f}  ratio={ratio:.2f} -> {verdict}", flush=True)
        xval.append({
            "held_VG1": held,
            "train_VG1s": train,
            "predicted_params_raw": {"Bf": Bf_pred, "iii": iii_pred, "log10Rs": Rs_pred},
            "predicted_params_clipped": {"Bf": Bf_pred_c, "iii": iii_pred_c, "log10Rs": Rs_pred_c},
            "R46_fit_dec": baseline_dec[held],
            "predicted_dec": pred_med,
            "ratio": ratio,
            "verdict": verdict,
        })

    # === 4. Plot ===
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, vals, lbl in zip(axes, [Bf_vals, iii_vals, Rs_vals],
                              ["Bf", "iii_body_gain", "log10(vnwell_Rs)"]):
        ax.plot(VG1S, vals, "o-", linewidth=2, markersize=10, label="R-46 fit")
        for x in xval:
            held = x["held_VG1"]
            train = x["train_VG1s"]
            train_vals = [R46_PARAMS[v][{"Bf":"Bf","iii_body_gain":"iii","log10(vnwell_Rs)":"log10Rs"}[lbl]] for v in train]
            held_pred = linear_extrap(train, train_vals, held)
            ax.plot([held], [held_pred], "x", markersize=14, markeredgewidth=3,
                    color="red", label=f"linear extrap hold={held}" if held == VG1S[0] else None)
        ax.set_xlabel("VG1 (V)"); ax.set_ylabel(lbl); ax.grid(True, alpha=0.3)
        ax.set_title(lbl)
    axes[0].legend(loc="best", fontsize=8)
    plt.suptitle("R-46 per-VG1 params vs VG1  +  LOOCV linear extrap")
    plt.tight_layout()
    plt.savefig(OUT / "params_vs_VG1.png", dpi=120)
    plt.close()

    # === 5. Overall verdict ===
    verdicts = [x["verdict"] for x in xval]
    n_pass    = sum(1 for v in verdicts if v == "PHYSICS_PASS")
    n_partial = sum(1 for v in verdicts if v == "PARTIAL")
    n_fit     = sum(1 for v in verdicts if v == "CURVE_FIT")
    if n_pass == 3:
        overall = "PHYSICS"
    elif n_fit >= 2:
        overall = "CURVE_FITTING"
    else:
        overall = "MIXED"

    # Honest curve-fit baseline: cell-wide dec if we forced ALL branches to use
    # mean R-46 params (poor-man's "no-per-VG1" engineering fit baseline).
    Bf_m  = float(np.mean(Bf_vals))
    iii_m = float(np.mean(iii_vals))
    Rs_m  = float(np.mean(Rs_vals))
    print(f"[z367] mean-param baseline: Bf={Bf_m:.1f} iii={iii_m:.3f} log10Rs={Rs_m:.3f}", flush=True)
    mean_branch = {}
    all_curves_mean = []
    for v in VG1S:
        med, lst = evaluate_branch(cfg, M1, M2, bjt, sebas_rows, curves, forward_2t,
                                   v, Bf_m, iii_m, 10.0**Rs_m)
        mean_branch[f"{v:.2f}"] = med
        all_curves_mean.extend([r["rmse"] for r in lst if not math.isnan(r["rmse"])])
        print(f"  mean-param VG1={v}: dec={med:.4f}", flush=True)
    cellwide_mean = float(np.median(all_curves_mean)) if all_curves_mean else float("nan")
    print(f"[z367] mean-param CELL-WIDE dec = {cellwide_mean:.4f}", flush=True)

    summary = {
        "script": "z367_physics_xval",
        "R46_eval94_params": R46_PARAMS,
        "monotonicity": monot,
        "R46_per_branch_dec": {f"{k:.2f}": v for k, v in baseline_dec.items()},
        "xval": xval,
        "n_physics_pass": n_pass,
        "n_partial": n_partial,
        "n_curve_fit": n_fit,
        "overall_verdict": overall,
        "honest_no_per_VG1_baseline": {
            "mean_Bf": Bf_m, "mean_iii": iii_m, "mean_log10Rs": Rs_m,
            "per_branch_dec": mean_branch,
            "cell_wide_dec": cellwide_mean,
        },
        "R46_reported_cell_wide_dec": 0.9650,
        "elapsed_s": time.time() - t0,
    }
    (OUT / "xval_results.json").write_text(json.dumps(summary, indent=2))

    # === 6. Verdict markdown ===
    lines = [
        "# R-48 Physics Cross-Validation Verdict\n",
        f"## Overall: {overall}",
        f"  PHYSICS_PASS branches: {n_pass}/3",
        f"  PARTIAL branches:      {n_partial}/3",
        f"  CURVE_FIT branches:    {n_fit}/3\n",
        "## Monotonicity",
        f"  Bf:      {monot['Bf']}",
        f"  iii:     {monot['iii']}",
        f"  log10Rs: {monot['log10Rs']}\n",
        "## Per-branch xval",
    ]
    for x in xval:
        lines.append(
            f"  VG1={x['held_VG1']:.2f}  R46_fit={x['R46_fit_dec']:.3f}  "
            f"pred={x['predicted_dec']:.3f}  ratio={x['ratio']:.2f}  -> {x['verdict']}")
    lines += [
        "",
        "## Honest no-per-VG1 baseline (mean params)",
        f"  Cell-wide dec: {cellwide_mean:.4f}",
        f"  R-46 reported: 0.9650",
        f"  R-46 1.131 dec floor came from prior global-knob runs",
    ]
    (OUT / "physics_verdict.md").write_text("\n".join(lines))
    print(f"[z367] DONE in {time.time()-t0:.0f}s  verdict={overall}", flush=True)


if __name__ == "__main__":
    main()
