"""z370 — R-50: PHYSICAL-BOUNDS GLOBAL BBO (5-dim, NOT per-VG1).

R-46 reached 0.965 dec but R-48 found ~half was curve-fit (Rs at VG1=0.6
was 4 OoM off). R-50 re-runs BBO with PHYSICAL bounds that prevent
curve-fit excursions, using SINGLE global params (NOT per-VG1).

5 dims (single global set):
  Bf            in [50, 50000]      (broad but real BJT beta range)
  Va            in [0.5, 3.0] V     (real Early voltage)
  Is            in [1e-13, 1e-7] A  (BJT saturation current)
  log10(Rs)     in [5, 9]           (1e5..1e9 ohm, real ohmic range)
  iii_body_gain in [0.1, 1.0]       (eta_lat physical range)

DE: popsize=15, maxiter=10  =>  ~150 evals.  Full 33 curves per eval.

Pre-registered gates:
  INFRA        : 33/33 valid
  PHYSICS PASS : cell-wide median < 1.50 dec   (R-43/45/47 baseline ~1.13)
  BREAKTHROUGH : cell-wide median < 1.00 dec   (global physics suffices)

Output: results/z370_physics_bbo/{summary.json, bbo_history.json}
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import sys, json, re, time, math, csv, importlib.util
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import differential_evolution


@contextmanager
def patch_sd_scaled(sd, overrides):
    if not overrides:
        yield; return
    saved = {}
    try:
        for k, v in overrides.items():
            saved[k] = sd.scaled.get(k, None)
            sd.scaled[k] = float(v) if hasattr(v, "item") and not torch.is_tensor(v) else (v.item() if torch.is_tensor(v) else float(v))
        yield
    finally:
        for k, v in saved.items():
            if v is None: sd.scaled.pop(k, None)
            else: sd.scaled[k] = v


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
OUT = ROOT / "results/z370_physics_bbo"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"


def load_sebas_params():
    path = DATA / "2Tcell_BSIM_param_DC.csv"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {}
            for k, v in r.items():
                try: row[k] = float(v)
                except ValueError: row[k] = float("nan")
            rows.append(row)
    return rows


BRANCH_FLAT = {
    0.4: {"ETAB": 1.9,  "K1": 0.53825, "ALPHA0": 7.842e-05, "BETA0": 19.0, "NFACTOR": 6.0, "trise": 10.59},
    0.6: {"ETAB": 2.5,  "K1": 0.41825, "ALPHA0": 7.842e-05, "BETA0": 20.0, "NFACTOR": 6.0, "trise": 9.04},
}
M2_STATIC_OVERRIDES = {
    "k1": 0.63825, "k2": -0.070435, "etab": -0.086777, "beta0": 18.0,
}


def find_or_impute_row(rows, VG1, VG2, atol=1e-3):
    target = None
    for r in rows:
        if abs(r["VG1"] - VG1) < atol and abs(r["VG2"] - VG2) < atol:
            target = dict(r); break
    if target is None:
        return None, False
    if math.isnan(target.get("K1", float("nan"))):
        branch = BRANCH_FLAT.get(round(VG1, 2))
        if branch is None:
            return target, False
        for k, v in branch.items():
            target[k] = float(v)
        return target, True
    return target, False


def make_overrides(sebas_row):
    if sebas_row is None:
        return None, None
    P_M1 = {}
    for csv_k, py_k in (("ETAB", "etab"), ("K1", "k1"),
                       ("ALPHA0", "alpha0"), ("BETA0", "beta0")):
        if not math.isnan(sebas_row.get(csv_k, float("nan"))):
            P_M1[py_k] = float(sebas_row[csv_k])
    P_M2 = {}
    if not math.isnan(sebas_row.get("NFACTOR", float("nan"))):
        P_M2["nfactor"] = float(sebas_row["NFACTOR"])
    for k, v in M2_STATIC_OVERRIDES.items():
        if k not in P_M2:
            P_M2[k] = float(v)
    return (P_M1 or None), (P_M2 or None)


def build_pyport_base():
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    cfg.body_pdiode_to = "vnwell"
    cfg.use_well_diode = True
    cfg.vnwell = 2.0  # frozen at R-45 floor
    cfg.body_pdiode_Js = 5.3675e-7 / 22e-12
    cfg.body_pdiode_n = 1.0535
    cfg.body_pdiode_Rs = 1.0e6
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    return cfg, M1, M2, bjt


def load_curves():
    curves = []
    for sub in sorted(DATA.iterdir()):
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
            curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:, 0],
                          "Id": np.abs(d[:, 1]), "f": f.name})
    for f in sorted(DATA.glob("VG1*VG2*.csv")):
        m = re.search(r"VG1=([\d.\-]+)[_ ]*VG2=([\d.\-]+)", f.name)
        if not m: continue
        vg1 = float(m.group(1)); vg2 = float(m.group(2))
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:, 0],
                      "Id": np.abs(d[:, 1]), "f": f.name})
    return curves


# 5-dim PHYSICAL bounds. x = [Bf, Va, log10(Is), log10(Rs), iii_body_gain]
PARAM_BOUNDS = [
    (50.0,    50000.0),   # Bf
    (0.5,     3.0),       # Va (V)
    (-13.0,   -7.0),      # log10(Is) -> Is in [1e-13, 1e-7]
    (5.0,     9.0),       # log10(Rs) -> Rs in [1e5, 1e9]
    (0.1,     1.0),       # iii_body_gain
]
PARAM_NAMES = ["Bf", "Va", "log10_Is", "log10_Rs", "iii_body_gain"]


def unpack(x):
    return {
        "Bf":            float(x[0]),
        "Va":            float(x[1]),
        "Is":            float(10.0 ** x[2]),
        "vnwell_Rs":     float(10.0 ** x[3]),
        "iii_body_gain": float(x[4]),
    }


_STATE = {}


def apply_params(p):
    cfg = _STATE["cfg"]; bjt = _STATE["bjt"]
    bjt.Bf = p["Bf"]; bjt.Va = p["Va"]; bjt.Is = p["Is"]
    cfg.iii_body_gain = p["iii_body_gain"]
    cfg.vnwell_Rs = p["vnwell_Rs"]


def eval_full(x):
    t0 = time.time()
    cfg = _STATE["cfg"]; M1 = _STATE["M1"]; M2 = _STATE["M2"]; bjt = _STATE["bjt"]
    sebas_rows = _STATE["sebas_rows"]; curves = _STATE["curves"]
    forward_2t = _STATE["forward_2t"]
    sd_M1 = cfg.size_dep_M1(M1)
    sd_M2 = cfg.size_dep_M2(M2)

    p = unpack(x); apply_params(p)
    rmses = []; n_valid = 0
    for c in curves:
        Vd = torch.tensor(c["Vd"], dtype=torch.float64)
        row, _ = find_or_impute_row(sebas_rows, c["VG1"], c["VG2"])
        P_M1, P_M2 = make_overrides(row)
        try:
            with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd,
                                 VG1=torch.tensor(c["VG1"], dtype=torch.float64),
                                 VG2=torch.tensor(c["VG2"], dtype=torch.float64),
                                 warm_start=True)
            Id_pred = np.abs(out["Id"].detach().cpu().numpy())
            mask = (c["Id"] > 1e-15) & (Id_pred > 1e-15) & np.isfinite(Id_pred)
            if mask.sum() >= 3:
                rmse = float(np.sqrt(np.mean((np.log10(Id_pred[mask]) - np.log10(c["Id"][mask])) ** 2)))
                n_valid += 1
            else:
                rmse = 10.0
        except Exception:
            rmse = 10.0
        rmses.append(rmse)
    cost = float(np.median(rmses))
    _STATE["history"].append({
        "iter": len(_STATE["history"]),
        "x": [float(v) for v in x],
        "params": p,
        "cost_median_dec": cost,
        "n_valid": n_valid, "n_total": len(curves),
        "elapsed_s": time.time() - t0,
    })
    if (len(_STATE["history"]) % 5) == 1:
        best = min(_STATE["history"], key=lambda h: h["cost_median_dec"])
        print(f"  [eval {len(_STATE['history']):3d}] cost={cost:.4f}  best={best['cost_median_dec']:.4f}  valid={n_valid}/{len(curves)}  ({time.time()-t0:.1f}s)", flush=True)
        (OUT / "bbo_history.json").write_text(json.dumps({
            "history": _STATE["history"],
            "best_so_far": best,
        }, indent=2))
    return cost


def final_refit(x_best):
    cfg = _STATE["cfg"]; M1 = _STATE["M1"]; M2 = _STATE["M2"]; bjt = _STATE["bjt"]
    sebas_rows = _STATE["sebas_rows"]; curves = _STATE["curves"]
    forward_2t = _STATE["forward_2t"]
    sd_M1 = cfg.size_dep_M1(M1)
    sd_M2 = cfg.size_dep_M2(M2)
    p = unpack(x_best); apply_params(p)

    results = []; per_vg1 = {}
    for c in curves:
        Vd = torch.tensor(c["Vd"], dtype=torch.float64)
        row, imp = find_or_impute_row(sebas_rows, c["VG1"], c["VG2"])
        P_M1, P_M2 = make_overrides(row)
        try:
            with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd,
                                 VG1=torch.tensor(c["VG1"], dtype=torch.float64),
                                 VG2=torch.tensor(c["VG2"], dtype=torch.float64),
                                 warm_start=True)
            Id_pred = np.abs(out["Id"].detach().cpu().numpy())
            mask = (c["Id"] > 1e-15) & (Id_pred > 1e-15) & np.isfinite(Id_pred)
            rmse = float(np.sqrt(np.mean((np.log10(Id_pred[mask]) - np.log10(c["Id"][mask])) ** 2))) if mask.sum() >= 3 else float("nan")
        except Exception:
            rmse = float("nan")
        results.append({"VG1": c["VG1"], "VG2": c["VG2"], "log_rmse_dec": rmse, "imputed": imp})
        k = round(c["VG1"], 2)
        per_vg1.setdefault(k, []).append(rmse)
    valid = [r["log_rmse_dec"] for r in results if not math.isnan(r["log_rmse_dec"])]
    per_vg1_med = {f"{k:.2f}": float(np.median([x for x in v if not math.isnan(x)])) if v else None
                   for k, v in per_vg1.items()}
    return {
        "cell_wide_median_dec": float(np.median(valid)) if valid else None,
        "per_VG1_median": per_vg1_med,
        "n_valid": len(valid), "n_total": len(results),
        "per_curve": results,
    }


def main():
    t0 = time.time()
    print(f"[z370] R-50 PHYSICAL-BOUNDS GLOBAL BBO (5-dim, popsize=15, maxiter=10)", flush=True)
    cfg, M1, M2, bjt = build_pyport_base()
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    sebas_rows = load_sebas_params()
    curves = load_curves()
    print(f"[z370] loaded {len(curves)} curves, {len(sebas_rows)} Sebas rows", flush=True)

    _STATE.update({
        "cfg": cfg, "M1": M1, "M2": M2, "bjt": bjt,
        "sebas_rows": sebas_rows, "curves": curves,
        "forward_2t": forward_2t,
        "history": [],
    })

    # Seed: R-43/R-39 floor params (Bf=991, Va=0.903, Is=5.95e-12, Rs=1e8, iii=1.0)
    x0 = np.array([991.0, 0.903, math.log10(5.95e-12), 8.0, 1.0])
    print("[z370] timing single eval at seed...", flush=True)
    tt = time.time()
    c0 = eval_full(x0)
    print(f"[z370] one eval = {time.time()-tt:.1f}s  cost={c0:.4f} dec", flush=True)

    print("[z370] DE running... (popsize=15, maxiter=10 -> ~150 evals)", flush=True)
    result = differential_evolution(
        eval_full,
        bounds=PARAM_BOUNDS,
        popsize=15,
        maxiter=10,
        tol=1e-3,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=50,
        polish=False,
        init="sobol",
        updating="immediate",
        workers=1,
        x0=x0,
        disp=True,
    )
    x_best = result.x
    cost_best = float(result.fun)
    print(f"[z370] DE DONE  cost_best={cost_best:.4f} dec  nfev={result.nfev}", flush=True)

    print("[z370] final 33-curve refit...", flush=True)
    refit = final_refit(x_best)
    p_best = unpack(x_best)

    cell_med = refit["cell_wide_median_dec"]
    n_valid = refit["n_valid"]; n_total = refit["n_total"]
    # Physicality check: are best params strictly inside the physical bounds?
    physical = {
        "Bf_in_bounds":    50.0 <= p_best["Bf"] <= 50000.0,
        "Va_in_bounds":    0.5 <= p_best["Va"] <= 3.0,
        "Is_in_bounds":    1e-13 <= p_best["Is"] <= 1e-7,
        "Rs_in_bounds":    1e5 <= p_best["vnwell_Rs"] <= 1e9,
        "iii_in_bounds":   0.1 <= p_best["iii_body_gain"] <= 1.0,
    }
    physical["all_physical"] = all(physical.values())

    summary = {
        "script": "z370_physics_bbo",
        "round": "R-50",
        "design": "5-dim GLOBAL params with PHYSICAL bounds (no curve-fit excursions)",
        "patches_active": [
            "R-20 BJT Vbc", "R-29 Vth/tox", "R-37 binunit",
            "R-41 body_pdiode_to=vnwell + use_well_diode=True",
            "R-45 cfg.vnwell=2.0 frozen",
            "R-50 5-dim global BBO with physical bounds",
        ],
        "frozen": {"cfg.vnwell": 2.0,
                   "cfg.body_pdiode_Js": 5.3675e-7 / 22e-12,
                   "cfg.body_pdiode_n": 1.0535,
                   "cfg.body_pdiode_Rs": 1.0e6},
        "param_bounds_physical": {
            "Bf":            [50.0, 50000.0],
            "Va_V":          [0.5, 3.0],
            "Is_A":          [1e-13, 1e-7],
            "vnwell_Rs_ohm": [1e5, 1e9],
            "iii_body_gain": [0.1, 1.0],
        },
        "best_params": p_best,
        "best_x_raw": [float(v) for v in x_best],
        "physicality_check": physical,
        "cell_wide_median_dec": cell_med,
        "per_VG1_median": refit["per_VG1_median"],
        "n_valid": n_valid, "n_total": n_total,
        "n_evals": int(result.nfev),
        "gates": {
            "INFRA_33_of_33":            bool(n_valid == n_total == 33),
            "PHYSICS_PASS_lt_1p50":      bool(cell_med is not None and cell_med < 1.50),
            "BREAKTHROUGH_lt_1p00":      bool(cell_med is not None and cell_med < 1.00),
        },
        "baselines": {
            "R-43_global_floor": 1.131,
            "R-45_floor":         1.131,
            "R-46_per_VG1_engfit": 0.965,
            "R-48_finding": "R-46 Rs at VG1=0.6 was 4 OoM off — half of 0.965 was curve-fit",
        },
        "elapsed_s": time.time() - t0,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "bbo_history.json").write_text(json.dumps({
        "history": _STATE["history"],
        "best": {"x": [float(v) for v in x_best], "params": p_best, "cost": cost_best},
    }, indent=2))
    (OUT / "final_refit_full_33.json").write_text(json.dumps(refit, indent=2))

    print(f"\n[z370] DONE")
    print(f"  cell-wide median = {cell_med}")
    print(f"  per-VG1 = {refit['per_VG1_median']}")
    print(f"  best params = {json.dumps(p_best, indent=2)}")
    print(f"  physical: {physical}")
    print(f"  gates: {summary['gates']}")
    print(f"  elapsed = {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
