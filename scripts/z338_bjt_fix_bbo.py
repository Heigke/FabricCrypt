"""z338 — Black-box optimize (alpha0, Bf, VAF, Is, lat_BV, body_pdiode_Rs)
with cfg.bjt_emitter_to_gnd=True LOCKED. Goal: cell-wide median < 0.95 dec.

Reuses z337's build_pyport() + load_curves() + per-curve forward_2t.

BBO objective is computed on a 9-curve subset (3 per VG1: VG2 in {min, mid, max})
to keep per-eval cost ~80s. After optimization, best params are re-evaluated on
all 33 curves for final reporting.

Search space (6D):
  alpha0:           [7.84e-7, 7.84e-3] (log)
  bjt.Bf:           [100, 50000]      (log)
  bjt.Va (VAF):     [0.3, 3.0]        (lin)
  bjt.Is:           [1e-12, 1e-6]     (log)
  cfg.lat_BV:       [3.0, 8.0]        (lin)
  body_pdiode_Rs:   [1e6, 1e10]       (log)
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/z338_bjt_fix_bbo"
OUT.mkdir(parents=True, exist_ok=True)

DATA = ROOT / "data/sebas_2026_04_22"


def build_pyport():
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 9000.0; bjt.Va = 0.55; bjt.Is = 1e-9
    return cfg, M1, M2, bjt


def load_curves():
    curves = []
    for f in sorted(DATA.glob("VG1*VG2*.csv")):
        m = re.search(r"VG1=([\d.\-]+)[_ ]*VG2=([\d.\-]+)", f.name)
        if not m: continue
        vg1 = float(m.group(1)); vg2 = float(m.group(2))
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:,0], "Id": np.abs(d[:,1]), "f": f.name})
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
                curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:,0], "Id": np.abs(d[:,1]), "f": f.name})
    return curves


def subset_for_bbo(curves):
    """Pick 3 per VG1 spanning the VG2 range (min, mid, max)."""
    by_vg1 = {}
    for c in curves:
        by_vg1.setdefault(c["VG1"], []).append(c)
    out = []
    for vg1, lst in sorted(by_vg1.items()):
        lst_sorted = sorted(lst, key=lambda c: c["VG2"])
        n = len(lst_sorted)
        idxs = sorted({0, n // 2, n - 1})
        for i in idxs:
            out.append(lst_sorted[i])
    return out


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
            return float("nan")
        logr = np.log10(Id_pred[mask]) - np.log10(c["Id"][mask])
        return float(np.sqrt(np.mean(logr ** 2)))
    except Exception:
        return float("nan")


def apply_params(cfg, M1, M2, bjt, x):
    """x = [log10_alpha0, log10_Bf, Va, log10_Is, lat_BV, log10_Rs]"""
    alpha0 = 10.0 ** x[0]
    bf     = 10.0 ** x[1]
    va     = x[2]
    is_    = 10.0 ** x[3]
    lat_bv = x[4]
    rs     = 10.0 ** x[5]
    # alpha0: override BOTH M1 and M2 size-dep scaled dict
    sd_M1 = cfg.size_dep_M1(M1)
    sd_M2 = cfg.size_dep_M2(M2)
    sd_M1.scaled["alpha0"] = float(alpha0)
    sd_M2.scaled["alpha0"] = float(alpha0)
    bjt.Bf = float(bf)
    bjt.Va = float(va)
    bjt.Is = float(is_)
    cfg.lat_BV = float(lat_bv)
    cfg.body_pdiode_Rs = float(rs)


def main():
    t0_total = time.time()
    cfg, M1, M2, bjt = build_pyport()
    from nsram.bsim4_port.nsram_cell_2T import forward_2t

    all_curves = load_curves()
    bbo_curves = subset_for_bbo(all_curves)
    print(f"[z338] loaded {len(all_curves)} curves, BBO subset = {len(bbo_curves)}", flush=True)
    print(f"[z338] BBO subset:", flush=True)
    for c in bbo_curves:
        print(f"    VG1={c['VG1']:.2f} VG2={c['VG2']:+.2f}", flush=True)

    bounds = [
        (np.log10(7.84e-7), np.log10(7.84e-3)),  # alpha0
        (np.log10(100.0),   np.log10(50000.0)),  # Bf
        (0.3, 3.0),                              # Va
        (np.log10(1e-12),   np.log10(1e-6)),     # Is
        (3.0, 8.0),                              # lat_BV
        (np.log10(1e6),     np.log10(1e10)),     # body_pdiode_Rs
    ]

    history = []
    eval_count = [0]

    def objective(x):
        eval_count[0] += 1
        t_e = time.time()
        apply_params(cfg, M1, M2, bjt, x)
        rmses = []
        for c in bbo_curves:
            r = eval_one_curve(cfg, M1, M2, bjt, c, forward_2t)
            rmses.append(r)
        valid = [r for r in rmses if not np.isnan(r)]
        if len(valid) < max(3, len(bbo_curves) // 2):
            cost = 50.0  # heavy penalty for too many failures
        else:
            cost = float(np.median(valid))
        dt = time.time() - t_e
        rec = {
            "eval": eval_count[0],
            "x": [float(v) for v in x],
            "params": {
                "alpha0": float(10.0 ** x[0]),
                "Bf":     float(10.0 ** x[1]),
                "Va":     float(x[2]),
                "Is":     float(10.0 ** x[3]),
                "lat_BV": float(x[4]),
                "body_pdiode_Rs": float(10.0 ** x[5]),
            },
            "subset_median_dec": cost,
            "n_valid": len(valid),
            "n_total": len(bbo_curves),
            "dt_s": dt,
            "elapsed_total_s": time.time() - t0_total,
        }
        history.append(rec)
        # incremental dump every 5 evals
        if eval_count[0] % 5 == 0 or eval_count[0] == 1:
            (OUT / "history.json").write_text(json.dumps(history, indent=2))
        print(f"  [eval {eval_count[0]:3d}] cost={cost:.3f}  "
              f"valid={len(valid)}/{len(bbo_curves)}  dt={dt:.1f}s  "
              f"elapsed={rec['elapsed_total_s']:.0f}s  "
              f"a0={rec['params']['alpha0']:.2e} Bf={rec['params']['Bf']:.0f} "
              f"Va={rec['params']['Va']:.2f} Is={rec['params']['Is']:.2e} "
              f"BV={rec['params']['lat_BV']:.2f} Rs={rec['params']['body_pdiode_Rs']:.1e}",
              flush=True)
        return cost

    # Budget: 9-curve subset eval = ~78s. Target ~60 evals = ~78min total.
    # DE: total evals ≈ popsize * (1 + maxiter). popsize=10, maxiter=5 → up to 60.
    from scipy.optimize import differential_evolution

    print(f"[z338] starting DE: popsize=10, maxiter=5, budget~60 evals", flush=True)
    result = differential_evolution(
        objective, bounds,
        strategy="best1bin",
        maxiter=5,
        popsize=10,
        tol=1e-3,
        mutation=(0.5, 1.0),
        recombination=0.7,
        init="sobol",
        seed=20260513,
        polish=False,
        updating="immediate",
        workers=1,
    )
    print(f"[z338] DE done: best_subset_median={result.fun:.4f} after {eval_count[0]} evals", flush=True)

    x_best = result.x
    best_params = {
        "alpha0": float(10.0 ** x_best[0]),
        "Bf":     float(10.0 ** x_best[1]),
        "Va":     float(x_best[2]),
        "Is":     float(10.0 ** x_best[3]),
        "lat_BV": float(x_best[4]),
        "body_pdiode_Rs": float(10.0 ** x_best[5]),
    }
    (OUT / "best_params.json").write_text(json.dumps(best_params, indent=2))
    (OUT / "history.json").write_text(json.dumps(history, indent=2))

    # Final 33-curve evaluation
    print(f"[z338] applying best params and running full 33-curve eval...", flush=True)
    apply_params(cfg, M1, M2, bjt, x_best)
    results = []
    per_vg1 = {}
    for c in all_curves:
        rmse = eval_one_curve(cfg, M1, M2, bjt, c, forward_2t)
        results.append({"VG1": c["VG1"], "VG2": c["VG2"], "log_rmse_dec": rmse, "f": c["f"]})
        per_vg1.setdefault(c["VG1"], []).append(rmse)
        print(f"  VG1={c['VG1']:.2f} VG2={c['VG2']:+.2f}  rmse={rmse:.3f}", flush=True)

    valid = [r["log_rmse_dec"] for r in results if not np.isnan(r["log_rmse_dec"])]
    cell_med = float(np.median(valid)) if valid else None
    summary = {
        "script": "z338_bjt_fix_bbo",
        "optimizer": "scipy.optimize.differential_evolution",
        "de_config": {"popsize": 12, "maxiter": 8, "strategy": "best1bin",
                       "init": "sobol", "seed": 20260513},
        "n_evals_used": eval_count[0],
        "bbo_subset_size": len(bbo_curves),
        "best_subset_median_dec": float(result.fun),
        "best_params": best_params,
        "n_curves_total": len(results),
        "n_curves_valid": len(valid),
        "cell_wide_median_dec": cell_med,
        "cell_wide_p25_dec":    float(np.percentile(valid, 25)) if valid else None,
        "cell_wide_p75_dec":    float(np.percentile(valid, 75)) if valid else None,
        "per_VG1_median": {f"{k:.2f}": float(np.median([x for x in v if not np.isnan(x)])) if any(not np.isnan(x) for x in v) else None for k,v in per_vg1.items()},
        "baselines": {"z304_v4": 0.99, "z313_v5b": 3.01, "z326": 3.43, "z334": 7.05, "z337": 4.155},
        "gate_PASS_lt_0.95": (cell_med is not None and cell_med < 0.95),
        "gate_AMBITIOUS_lt_0.50": (cell_med is not None and cell_med < 0.50),
        "improvement_vs_z337_dec": (4.155 - cell_med) if cell_med is not None else None,
        "improvement_vs_z304_dec": (0.99 - cell_med) if cell_med is not None else None,
        "elapsed_s": time.time() - t0_total,
        "per_curve": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[z338] DONE cell_med={cell_med}  per_VG1={summary['per_VG1_median']}", flush=True)
    print(f"[z338] gate PASS(<0.95)={summary['gate_PASS_lt_0.95']}  "
          f"AMBITIOUS(<0.50)={summary['gate_AMBITIOUS_lt_0.50']}", flush=True)
    print(f"[z338] elapsed {summary['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
