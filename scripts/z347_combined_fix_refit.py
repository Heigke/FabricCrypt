"""z347 — R-31: COMBINED fix (R-26 lalpha0 + R-29 Vth) cell-wide refit + optional BBO.

PRE-REGISTERED GATES (locked BEFORE compute, per O64 NO-CHEAT):
  - INFRA:      33/33 curves valid (no NaN)
  - PASS:       cell-wide median dec < 0.95
  - AMBITIOUS:  cell-wide median dec < 0.50
  - DIAGNOSTIC: per-VG1 medians; VG1=0.60 MUST improve vs R-30's 5.42

Plan:
  1. Baseline z347 = PATCHED M1 card (M1_130DNWFB_LALPHA0_FIX, alpha0=7.84e-4,
     lalpha0=0) + Vth patch (lpe0=1.2439e-7, toxe=4e-9 from patch_model_values)
     + cfg.bjt_emitter_to_gnd=True + z334 BJT defaults (Bf=9000, Va=0.55,
     Is=1e-9). No lat_BV / body_pdiode_Rs overrides (use defaults).
  2. If baseline cell-wide median > 1.5 dec → run BBO popsize=8 maxiter=4
     (~32+ evals) over (Bf ∈ [3000,30000], Va ∈ [0.3,3.0], Is ∈ [1e-12,1e-6]
     log, body_pdiode_Rs ∈ [1e6,1e10] log) on FULL 33 curves
     (NOT 9-subset, per O64 NO-CHEAT critique of z338).

Baselines: z304_v4=0.99, z337=4.16, z343=3.99, z344C=4.61, z346(Vth only)=4.08.
R-30 per-VG1 medians: 0.20→2.32, 0.40→3.74, 0.60→5.42.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys, json, re, time, importlib.util
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/z347_combined_fix"
OUT.mkdir(parents=True, exist_ok=True)

DATA = ROOT / "data/sebas_2026_04_22"

# ---- PRE-REGISTERED GATES (locked) ----
GATE_INFRA_MIN_VALID = 33
GATE_PASS_LT        = 0.95
GATE_AMBITIOUS_LT   = 0.50
GATE_VG060_MUST_IMPROVE_VS = 5.42  # R-30
GATE_BBO_TRIGGER_GT = 1.5

# ---- z334 BJT defaults (per task) ----
BJT_BF_DEFAULT = 9000.0
BJT_VA_DEFAULT = 0.55
BJT_IS_DEFAULT = 1.0e-9

# ---- BBO budget (per task) ----
BBO_POPSIZE = 8
BBO_MAXITER = 4
BBO_SEED    = 20260514


def build_pyport():
    """Build cfg, M1 (PATCHED card), M2, bjt (z334 defaults)."""
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.model_card import BSIM4Model
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    f = v1.f  # z91f_validate_with_sebas_params module (patch_model_values)

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True

    # Load PATCHED M1 card (R-26 lalpha0=0 fix), original M2.
    text_M1 = (DATA / "M1_130DNWFB_LALPHA0_FIX.txt").read_text()
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    M1 = BSIM4Model.from_spice(text_M1, model_type="nmos")
    M2 = BSIM4Model.from_spice(text_M2, model_type="nmos")
    # patch_model_values applies R-29 Vth fix (lpe0=1.2439e-7, toxe=4e-9)
    f.patch_model_values(M1, type_n=True)
    f.patch_model_values(M2, type_n=True)
    M1._values["voff"] = M1._values.get("voff", -0.1368)
    M2._values["voff"] = M2._values.get("voff", -0.1368)

    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = BJT_BF_DEFAULT
    bjt.Va = BJT_VA_DEFAULT
    bjt.Is = BJT_IS_DEFAULT
    return cfg, M1, M2, bjt


def load_curves():
    curves = []
    for f in sorted(DATA.glob("VG1*VG2*.csv")):
        m = re.search(r"VG1=([\d.\-]+)[_ ]*VG2=([\d.\-]+)", f.name)
        if not m: continue
        vg1 = float(m.group(1)); vg2 = float(m.group(2))
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:,0],
                       "Id": np.abs(d[:,1]), "f": f.name})
    if not curves:
        for sub in DATA.iterdir():
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
                curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:,0],
                               "Id": np.abs(d[:,1]), "f": f.name})
    return curves


def eval_one_curve(cfg, M1, M2, bjt, c, forward_2t):
    Vd = torch.tensor(c["Vd"], dtype=torch.float64)
    try:
        out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt,
                         Vd_seq=Vd,
                         VG1=torch.tensor(c["VG1"], dtype=torch.float64),
                         VG2=torch.tensor(c["VG2"], dtype=torch.float64),
                         warm_start=True)
        Id_pred = np.abs(out["Id"].detach().cpu().numpy())
        mask = (c["Id"] > 1e-15) & (Id_pred > 1e-15) & np.isfinite(Id_pred)
        if mask.sum() < 3:
            return float("nan"), None
        logr = np.log10(Id_pred[mask]) - np.log10(c["Id"][mask])
        return float(np.sqrt(np.mean(logr ** 2))), Id_pred
    except Exception as e:
        return float("nan"), None


def eval_all(cfg, M1, M2, bjt, curves, forward_2t, want_preds=False):
    results = []
    per_vg1 = {}
    preds = []
    for c in curves:
        rmse, Id_pred = eval_one_curve(cfg, M1, M2, bjt, c, forward_2t)
        results.append({"VG1": c["VG1"], "VG2": c["VG2"],
                        "log_rmse_dec": rmse, "f": c["f"]})
        per_vg1.setdefault(c["VG1"], []).append(rmse)
        if want_preds:
            preds.append({"c": c, "Id_pred": Id_pred, "rmse": rmse})
    valid = [r["log_rmse_dec"] for r in results
             if not np.isnan(r["log_rmse_dec"])]
    summary = {
        "n_curves_total": len(results),
        "n_curves_valid": len(valid),
        "cell_wide_median_dec": float(np.median(valid)) if valid else None,
        "cell_wide_p25_dec":    float(np.percentile(valid, 25)) if valid else None,
        "cell_wide_p75_dec":    float(np.percentile(valid, 75)) if valid else None,
        "cell_wide_max_dec":    float(np.max(valid)) if valid else None,
        "cell_wide_min_dec":    float(np.min(valid)) if valid else None,
        "per_VG1_median": {
            f"{k:.2f}": (float(np.median([x for x in v if not np.isnan(x)]))
                         if any(not np.isnan(x) for x in v) else None)
            for k,v in per_vg1.items()
        },
        "per_curve": results,
    }
    return summary, preds


def plot_per_vg1(preds, summary, tag):
    vg1_set = sorted(set(p["c"]["VG1"] for p in preds))
    for vg1 in vg1_set:
        these = [p for p in preds if abs(p["c"]["VG1"] - vg1) < 1e-6]
        if not these: continue
        these.sort(key=lambda p: p["c"]["VG2"])
        med = summary["per_VG1_median"].get(f"{vg1:.2f}")
        title = (f"z347 {tag} — VG1={vg1:.2f}"
                 + (f"  (median dec={med:.2f})" if med is not None else ""))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(title)
        ax = axes[0]
        cmap = plt.cm.viridis(np.linspace(0, 1, len(these)))
        for col, p in zip(cmap, these):
            c = p["c"]
            ax.semilogy(c["Vd"], np.maximum(c["Id"], 1e-16), color=col, lw=1.5,
                        label=f"VG2={c['VG2']:+.2f}")
            if p["Id_pred"] is not None:
                ax.semilogy(c["Vd"], np.maximum(p["Id_pred"], 1e-16),
                            "--", color=col, lw=1.0)
        ax.set_xlabel("Vd [V]"); ax.set_ylabel("|Id| [A]")
        ax.set_title("solid=silicon  dashed=pyport")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        ax = axes[1]
        vg2s = [p["c"]["VG2"] for p in these]
        rmses = [p["rmse"] for p in these]
        ax.plot(vg2s, rmses, "o-")
        ax.axhline(0.95, color="g", ls=":", label="PASS=0.95")
        ax.axhline(0.50, color="b", ls=":", label="AMBITIOUS=0.50")
        ax.set_xlabel("VG2 [V]"); ax.set_ylabel("log10 RMSE [dec]")
        ax.set_title("per-curve RMSE")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        ax = axes[2]
        for col, p in zip(cmap, these):
            if p["Id_pred"] is None: continue
            c = p["c"]
            mask = (c["Id"] > 1e-15) & (p["Id_pred"] > 1e-15) & np.isfinite(p["Id_pred"])
            if mask.sum() < 2: continue
            resid = np.log10(p["Id_pred"][mask]) - np.log10(c["Id"][mask])
            ax.plot(c["Vd"][mask], resid, color=col, lw=1.0,
                    label=f"VG2={c['VG2']:+.2f}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Vd [V]")
        ax.set_ylabel("log10(pyport) − log10(silicon)")
        ax.set_title("log residual"); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"plot_{tag}_VG1_{vg1:.2f}.png", dpi=110)
        plt.close(fig)


def apply_bbo_params(cfg, bjt, x):
    """x = [log10_Bf, Va, log10_Is, log10_body_pdiode_Rs]"""
    bjt.Bf = float(10.0 ** x[0])
    bjt.Va = float(x[1])
    bjt.Is = float(10.0 ** x[2])
    cfg.body_pdiode_Rs = float(10.0 ** x[3])


def verdict_block(summary, n_total_curves):
    med = summary["cell_wide_median_dec"]
    pvg1 = summary["per_VG1_median"]
    vg060 = pvg1.get("0.60")
    infra_ok = summary["n_curves_valid"] >= GATE_INFRA_MIN_VALID
    pass_ok = (med is not None) and (med < GATE_PASS_LT)
    amb_ok  = (med is not None) and (med < GATE_AMBITIOUS_LT)
    vg060_ok = (vg060 is not None) and (vg060 < GATE_VG060_MUST_IMPROVE_VS)
    return {
        "INFRA":      bool(infra_ok),
        "PASS":       bool(pass_ok),
        "AMBITIOUS":  bool(amb_ok),
        "DIAGNOSTIC_VG060_improves_vs_R30_5.42": bool(vg060_ok),
        "cell_wide_median_dec": med,
        "vg060_median_dec": vg060,
    }


def main():
    t0 = time.time()
    cfg, M1, M2, bjt = build_pyport()
    from nsram.bsim4_port.nsram_cell_2T import forward_2t

    print(f"[z347] PRE-REGISTERED GATES (locked):", flush=True)
    print(f"  INFRA={GATE_INFRA_MIN_VALID}/33  PASS<{GATE_PASS_LT}  "
          f"AMBITIOUS<{GATE_AMBITIOUS_LT}  "
          f"VG060<{GATE_VG060_MUST_IMPROVE_VS}", flush=True)
    print(f"  BBO trigger: baseline cell_wide_median > {GATE_BBO_TRIGGER_GT}",
          flush=True)
    print(f"[z347] cfg.bjt_emitter_to_gnd={cfg.bjt_emitter_to_gnd}  "
          f"use_bjt={cfg.use_bjt}", flush=True)
    print(f"[z347] BJT z334 defaults: Bf={bjt.Bf} Va={bjt.Va} Is={bjt.Is:.2e}",
          flush=True)
    print(f"[z347] M1 alpha0={M1._values.get('alpha0')} "
          f"lalpha0={M1._values.get('lalpha0')} "
          f"lpe0={M1._values.get('lpe0')} toxe={M1._values.get('toxe')}",
          flush=True)

    curves = load_curves()
    print(f"[z347] loaded {len(curves)} curves", flush=True)
    if not curves:
        print(f"[z347] no curves found in {DATA}", flush=True)
        return

    # ---- Stage 1: baseline (z334 BJT defaults) ----
    print(f"\n[z347] === STAGE 1: baseline (combined fix + z334 BJT defaults) ===",
          flush=True)
    t_b0 = time.time()
    baseline_summary, baseline_preds = eval_all(cfg, M1, M2, bjt, curves,
                                                 forward_2t, want_preds=True)
    baseline_summary["elapsed_s"] = time.time() - t_b0
    for r in baseline_summary["per_curve"]:
        print(f"  VG1={r['VG1']:.2f} VG2={r['VG2']:+.2f}  "
              f"rmse={r['log_rmse_dec']:.3f}", flush=True)
    print(f"[z347] BASELINE median={baseline_summary['cell_wide_median_dec']}  "
          f"per_VG1={baseline_summary['per_VG1_median']}", flush=True)
    baseline_verdict = verdict_block(baseline_summary, len(curves))
    print(f"[z347] BASELINE verdict={baseline_verdict}", flush=True)

    plot_per_vg1(baseline_preds, baseline_summary, tag="baseline")

    # ---- Stage 2 (conditional): BBO on FULL 33 curves ----
    bbo_block = None
    best_after_bbo_summary = None
    best_after_bbo_verdict = None
    do_bbo = (baseline_summary["cell_wide_median_dec"] is not None and
              baseline_summary["cell_wide_median_dec"] > GATE_BBO_TRIGGER_GT)
    if do_bbo:
        print(f"\n[z347] === STAGE 2: BBO on FULL 33 curves "
              f"(median {baseline_summary['cell_wide_median_dec']:.3f} > "
              f"{GATE_BBO_TRIGGER_GT}) ===", flush=True)
        bounds = [
            (np.log10(3000.0),  np.log10(30000.0)),  # Bf
            (0.3, 3.0),                              # Va
            (np.log10(1e-12),   np.log10(1e-6)),     # Is
            (np.log10(1e6),     np.log10(1e10)),     # body_pdiode_Rs
        ]
        history = []
        eval_count = [0]
        t_bbo_0 = time.time()

        def objective(x):
            eval_count[0] += 1
            t_e = time.time()
            apply_bbo_params(cfg, bjt, x)
            rmses = []
            for c in curves:  # FULL 33
                r, _ = eval_one_curve(cfg, M1, M2, bjt, c, forward_2t)
                rmses.append(r)
            valid = [r for r in rmses if not np.isnan(r)]
            if len(valid) < max(3, len(curves) // 2):
                cost = 50.0
            else:
                cost = float(np.median(valid))
            dt = time.time() - t_e
            rec = {
                "eval": eval_count[0],
                "x": [float(v) for v in x],
                "params": {
                    "Bf":     float(10.0 ** x[0]),
                    "Va":     float(x[1]),
                    "Is":     float(10.0 ** x[2]),
                    "body_pdiode_Rs": float(10.0 ** x[3]),
                },
                "full33_median_dec": cost,
                "n_valid": len(valid),
                "n_total": len(curves),
                "dt_s": dt,
                "elapsed_total_s": time.time() - t_bbo_0,
            }
            history.append(rec)
            if eval_count[0] % 2 == 0 or eval_count[0] == 1:
                (OUT / "bbo_history.json").write_text(json.dumps(history, indent=2))
            print(f"  [eval {eval_count[0]:3d}] cost={cost:.3f}  "
                  f"valid={len(valid)}/{len(curves)}  dt={dt:.1f}s  "
                  f"elapsed={rec['elapsed_total_s']:.0f}s  "
                  f"Bf={rec['params']['Bf']:.0f} "
                  f"Va={rec['params']['Va']:.2f} "
                  f"Is={rec['params']['Is']:.2e} "
                  f"Rs={rec['params']['body_pdiode_Rs']:.1e}", flush=True)
            return cost

        from scipy.optimize import differential_evolution
        print(f"[z347] starting DE: popsize={BBO_POPSIZE}, "
              f"maxiter={BBO_MAXITER}, budget~"
              f"{BBO_POPSIZE * (1 + BBO_MAXITER)} evals on FULL 33 curves",
              flush=True)
        result = differential_evolution(
            objective, bounds,
            strategy="best1bin",
            maxiter=BBO_MAXITER,
            popsize=BBO_POPSIZE,
            tol=1e-3,
            mutation=(0.5, 1.0),
            recombination=0.7,
            init="sobol",
            seed=BBO_SEED,
            polish=False,
            updating="immediate",
            workers=1,
        )
        print(f"[z347] DE done: best_full33_median={result.fun:.4f} "
              f"after {eval_count[0]} evals", flush=True)
        x_best = result.x
        best_params = {
            "Bf":     float(10.0 ** x_best[0]),
            "Va":     float(x_best[1]),
            "Is":     float(10.0 ** x_best[2]),
            "body_pdiode_Rs": float(10.0 ** x_best[3]),
        }
        (OUT / "bbo_best_params.json").write_text(json.dumps(best_params, indent=2))
        (OUT / "bbo_history.json").write_text(json.dumps(history, indent=2))

        # Final 33-curve evaluation with best params
        apply_bbo_params(cfg, bjt, x_best)
        print(f"[z347] applying BBO best and re-running FULL 33 with preds...",
              flush=True)
        best_after_bbo_summary, best_preds = eval_all(cfg, M1, M2, bjt, curves,
                                                       forward_2t, want_preds=True)
        for r in best_after_bbo_summary["per_curve"]:
            print(f"  VG1={r['VG1']:.2f} VG2={r['VG2']:+.2f}  "
                  f"rmse={r['log_rmse_dec']:.3f}", flush=True)
        print(f"[z347] POST-BBO median="
              f"{best_after_bbo_summary['cell_wide_median_dec']}  "
              f"per_VG1={best_after_bbo_summary['per_VG1_median']}", flush=True)
        best_after_bbo_verdict = verdict_block(best_after_bbo_summary, len(curves))
        print(f"[z347] POST-BBO verdict={best_after_bbo_verdict}", flush=True)
        plot_per_vg1(best_preds, best_after_bbo_summary, tag="bbo_best")

        bbo_block = {
            "ran": True,
            "popsize": BBO_POPSIZE,
            "maxiter": BBO_MAXITER,
            "n_evals": eval_count[0],
            "bounds": {
                "Bf":  [3000.0, 30000.0],
                "Va":  [0.3, 3.0],
                "Is":  [1e-12, 1e-6],
                "body_pdiode_Rs": [1e6, 1e10],
            },
            "best_params": best_params,
            "best_subset_cost": float(result.fun),
            "n_curves_eval_per_iter": len(curves),
            "elapsed_s": time.time() - t_bbo_0,
        }
    else:
        bbo_block = {"ran": False,
                     "reason": f"baseline median "
                               f"{baseline_summary['cell_wide_median_dec']} "
                               f"<= {GATE_BBO_TRIGGER_GT}"}

    # ---- Final summary ----
    summary = {
        "script": "z347_combined_fix_refit",
        "task": "R-31: COMBINED R-26 lalpha0 fix + R-29 Vth fix + BJT z334 "
                "defaults; optional BBO on FULL 33 curves (NO-CHEAT).",
        "pre_registered_gates": {
            "INFRA_min_valid": GATE_INFRA_MIN_VALID,
            "PASS_lt": GATE_PASS_LT,
            "AMBITIOUS_lt": GATE_AMBITIOUS_LT,
            "DIAGNOSTIC_VG060_must_improve_vs": GATE_VG060_MUST_IMPROVE_VS,
            "BBO_trigger_gt": GATE_BBO_TRIGGER_GT,
        },
        "patches_active": {
            "M1_card_file": "M1_130DNWFB_LALPHA0_FIX.txt (R-26: alpha0=7.84e-4, lalpha0=0)",
            "M2_card_file": "M2_130bulkNSRAM.txt (original)",
            "patch_model_values_Vth_fix": {"lpe0": 1.2439e-7, "toxe": 4e-9},
            "cfg.bjt_emitter_to_gnd": True,
        },
        "bjt_defaults": {"Bf": BJT_BF_DEFAULT, "Va": BJT_VA_DEFAULT,
                         "Is": BJT_IS_DEFAULT},
        "baselines_prior": {"z304_v4": 0.99, "z337": 4.16, "z343": 3.99,
                            "z344C": 4.61, "z346_Vth_only": 4.08},
        "R30_per_VG1": {"0.20": 2.32, "0.40": 3.74, "0.60": 5.42},
        "stage1_baseline": baseline_summary,
        "stage1_baseline_verdict": baseline_verdict,
        "stage2_bbo": bbo_block,
        "stage2_post_bbo_summary": best_after_bbo_summary,
        "stage2_post_bbo_verdict": best_after_bbo_verdict,
        "elapsed_s_total": time.time() - t0,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[z347] DONE  elapsed {summary['elapsed_s_total']:.1f}s", flush=True)
    print(f"[z347] BASELINE verdict={baseline_verdict}", flush=True)
    if best_after_bbo_verdict is not None:
        print(f"[z347] POST-BBO verdict={best_after_bbo_verdict}", flush=True)


if __name__ == "__main__":
    main()
