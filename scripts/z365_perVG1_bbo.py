"""z365 — R-46: Per-VG1 BBO (engineering fit, NOT physical).

Each VG1 branch (0.20, 0.40, 0.60) gets its own (Bf, iii_body_gain, vnwell_Rs).
9 parameters total, optimized via differential evolution over the full
33-curve dataset (no subset — per O64 NO-CHEAT rule).

Stack: R-20 + R-29 + R-37 + R-41 (pdiode_to=vnwell, use_well_diode=True).
cfg.vnwell fixed at 2.0 V (from R-45 floor result). Va, Is fixed at R-39 best.

Pre-registered gates:
  PASS         : cell-wide median < 0.95 dec
  AMBITIOUS    : cell-wide median < 0.50 dec
  HONESTY      : each VG1's params reported separately; flagged engineering.

Output: results/z365_perVG1_bbo/{summary.json, bbo_history.json,
final_refit_full_33.json}
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
OUT = ROOT / "results/z365_perVG1_bbo"; OUT.mkdir(parents=True, exist_ok=True)
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
    """Build cfg/models once; we will set per-VG1 params per-curve."""
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
    # Frozen at R-39 BBO best
    bjt.Va = 0.903; bjt.Is = 5.95e-12
    bjt.Bf = 991.0  # default; will be overridden per-VG1
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


# 9-dim parameter mapping
# x = [Bf_020, iii_020, log10Rs_020,  Bf_040, iii_040, log10Rs_040,  Bf_060, iii_060, log10Rs_060]
VG1_KEYS = [0.20, 0.40, 0.60]
PARAM_BOUNDS = [
    # Per-branch (Bf, iii_body_gain, log10(vnwell_Rs))
    (50.0, 2000.0),   # Bf
    (0.05, 2.0),      # iii_body_gain
    (6.0, 10.0),      # log10 vnwell_Rs
] * 3


def unpack(x):
    """Return dict[VG1] -> (Bf, iii, Rs)."""
    out = {}
    for i, vg1 in enumerate(VG1_KEYS):
        Bf  = float(x[3*i + 0])
        iii = float(x[3*i + 1])
        Rs  = float(10.0 ** x[3*i + 2])
        out[vg1] = (Bf, iii, Rs)
    return out


# Globals so DE closure stays light
_STATE = {}


def eval_full(x):
    """Cost = cell-wide median log10-RMSE over all 33 curves."""
    t0 = time.time()
    cfg = _STATE["cfg"]; M1 = _STATE["M1"]; M2 = _STATE["M2"]; bjt = _STATE["bjt"]
    sebas_rows = _STATE["sebas_rows"]; curves = _STATE["curves"]
    forward_2t = _STATE["forward_2t"]
    sd_M1 = cfg.size_dep_M1(M1)
    sd_M2 = cfg.size_dep_M2(M2)

    params = unpack(x)
    rmses = []
    for c in curves:
        vg1_key = round(c["VG1"], 2)
        Bf, iii, Rs = params[vg1_key]
        # Set per-VG1 params
        bjt.Bf = Bf
        cfg.iii_body_gain = iii
        cfg.vnwell_Rs = Rs
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
            else:
                rmse = 10.0
        except Exception:
            rmse = 10.0
        rmses.append(rmse)
    cost = float(np.median(rmses))
    _STATE["history"].append({
        "iter": len(_STATE["history"]),
        "x": [float(v) for v in x],
        "cost_median_dec": cost,
        "elapsed_s": time.time() - t0,
    })
    if (len(_STATE["history"]) % 5) == 1:
        print(f"  [eval {len(_STATE['history']):3d}] cost={cost:.4f} dec  ({time.time()-t0:.1f}s)", flush=True)
        # Snapshot
        (OUT / "bbo_history.json").write_text(json.dumps({
            "history": _STATE["history"],
            "best_so_far": min(_STATE["history"], key=lambda h: h["cost_median_dec"]),
        }, indent=2))
    return cost


def final_refit(x_best):
    """Run full 33-curve eval with detailed per-curve output."""
    cfg = _STATE["cfg"]; M1 = _STATE["M1"]; M2 = _STATE["M2"]; bjt = _STATE["bjt"]
    sebas_rows = _STATE["sebas_rows"]; curves = _STATE["curves"]
    forward_2t = _STATE["forward_2t"]
    sd_M1 = cfg.size_dep_M1(M1)
    sd_M2 = cfg.size_dep_M2(M2)

    params = unpack(x_best)
    results = []; per_vg1 = {0.20: [], 0.40: [], 0.60: []}
    for c in curves:
        vg1_key = round(c["VG1"], 2)
        Bf, iii, Rs = params[vg1_key]
        bjt.Bf = Bf
        cfg.iii_body_gain = iii
        cfg.vnwell_Rs = Rs
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
        except Exception as e:
            rmse = float("nan")
        results.append({"VG1": c["VG1"], "VG2": c["VG2"], "log_rmse_dec": rmse,
                       "imputed": imp, "Bf": Bf, "iii_body_gain": iii, "vnwell_Rs": Rs})
        per_vg1[vg1_key].append(rmse)
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
    print(f"[z365] PER-VG1 BBO  (9-dim, popsize=12, maxiter=8)", flush=True)
    cfg, M1, M2, bjt = build_pyport_base()
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    sebas_rows = load_sebas_params()
    curves = load_curves()
    print(f"[z365] loaded {len(curves)} curves, {len(sebas_rows)} Sebas rows", flush=True)

    _STATE.update({
        "cfg": cfg, "M1": M1, "M2": M2, "bjt": bjt,
        "sebas_rows": sebas_rows, "curves": curves,
        "forward_2t": forward_2t,
        "history": [],
    })

    # Time a single full-33 eval first
    print("[z365] timing single eval...", flush=True)
    x0 = np.array([991.0, 1.0, 8.0,   991.0, 1.0, 8.0,   991.0, 1.0, 8.0])
    tt = time.time()
    c0 = eval_full(x0)
    print(f"[z365] one eval = {time.time()-tt:.1f}s  cost={c0:.4f} dec", flush=True)

    # Differential evolution
    print("[z365] DE running...", flush=True)
    result = differential_evolution(
        eval_full,
        bounds=PARAM_BOUNDS,
        popsize=12,
        maxiter=8,
        tol=1e-3,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=46,
        polish=False,
        init="sobol",
        updating="immediate",
        workers=1,
        x0=x0,
        disp=True,
    )
    x_best = result.x
    cost_best = float(result.fun)
    print(f"[z365] DE DONE  cost_best={cost_best:.4f} dec  nfev={result.nfev}", flush=True)

    # Final refit
    print("[z365] final 33-curve refit...", flush=True)
    refit = final_refit(x_best)
    (OUT / "final_refit_full_33.json").write_text(json.dumps(refit, indent=2))

    params = unpack(x_best)
    per_vg1_params = {
        f"{vg1:.2f}": {"Bf": Bf, "iii_body_gain": iii, "vnwell_Rs": Rs}
        for vg1, (Bf, iii, Rs) in params.items()
    }

    cell_med = refit["cell_wide_median_dec"]
    summary = {
        "script": "z365_perVG1_bbo",
        "patches_active": [
            "R-20 BJT Vbc", "R-29 Vth/tox", "R-37 binunit",
            "R-41 body_pdiode_to=vnwell + use_well_diode=True",
            "R-39 BJT Va/Is frozen (0.903, 5.95e-12)",
            "R-45 cfg.vnwell=2.0 frozen",
            "R-46 PER-VG1 (Bf, iii_body_gain, vnwell_Rs) — engineering fit",
        ],
        "honesty_caveat": (
            "PER-VG1 parameters violate physical Sebas card structure: a real "
            "device has ONE set of (Bf, Va, Is, iii, well_Rs). This fit is an "
            "engineering model that captures the anti-correlated VG1=0.20 vs "
            "VG1=0.60 branches by giving each its own freedom. Reported as "
            "curve-fitting; physical claim is bounded by the global-knob floor "
            "(z364 R-45: 1.131 dec)."
        ),
        "frozen": {"cfg.vnwell": 2.0, "bjt.Va": 0.903, "bjt.Is": 5.95e-12},
        "param_bounds": [list(b) for b in PARAM_BOUNDS],
        "per_VG1_params_best": per_vg1_params,
        "cell_wide_median_dec": cell_med,
        "per_VG1_median": refit["per_VG1_median"],
        "n_evals": int(result.nfev),
        "baselines": {
            "z363_R43_global": 1.1306581736187744,
            "z364_R45_floor": 1.131,
            "z361_R41": 1.419,
            "ngspice_target_aspiration": 0.27,
        },
        "elapsed_s": time.time() - t0,
    }
    summary["gate_PASS_lt_0p95"] = bool(cell_med is not None and cell_med < 0.95)
    summary["gate_AMBITIOUS_lt_0p50"] = bool(cell_med is not None and cell_med < 0.50)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "bbo_history.json").write_text(json.dumps({
        "history": _STATE["history"],
        "best": {"x": [float(v) for v in x_best], "cost": cost_best},
    }, indent=2))

    print(f"\n[z365] DONE")
    print(f"  cell-wide median = {cell_med:.4f} dec")
    print(f"  per-VG1 = {refit['per_VG1_median']}")
    print(f"  per-VG1 params = {json.dumps(per_vg1_params, indent=2)}")
    print(f"  PASS<0.95: {summary['gate_PASS_lt_0p95']}  AMB<0.50: {summary['gate_AMBITIOUS_lt_0p50']}")
    print(f"  elapsed = {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
