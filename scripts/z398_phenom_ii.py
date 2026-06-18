"""z398 — S5-B: Phenomenological impact-ionization S-curve fit (post-S4 recovery #2).

REPLACES BSIM4 §6.1 IIMOD with an empirical sigmoid fold:
    Iii_phenom = |Ids_M1| · A(VG1) · sigmoid((Vd - V_knee(VG1)) / Vsharp(VG1))

9-dim BBO over 33-curve cell-wide log10-RMSE cost. Per-VG1 ∈ {0.20, 0.40, 0.60}
gets (A, V_knee, Vsharp). Cell flag `cfg.use_phenom_ii=True` swaps BSIM4 Iii for
the S-curve at the M1 site; routing through eta_lat / iii_gain / iii_body_gain
is unchanged.

Pre-registered gates (logged 01_LOG.md 2026-05-15 13:23):
  INFRA       : BBO completes <60 min, no nan
  DISCOVERY   : cell-wide < 0.85 dec AND VG1=0.6 fold > 0.5 dec
  AMBITIOUS   : cell-wide < 0.30 dec AND VG1=0.6 fold > 1.5 dec
  KILL-SHOT   : phenom S-curve can't fit either (cell-wide > 1.5 dec)

CRITICAL CAVEAT (carried into summary.json): EMPIRICAL fitting, not derived
physics. A successful fit shows the topology bottleneck is the BSIM4 IIMOD
formula, not the BJT routing path. Does NOT validate the mechanism.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
#  Plumbing                                                                   #
# --------------------------------------------------------------------------- #
@contextmanager
def patch_sd_scaled(sd, overrides):
    if not overrides:
        yield; return
    saved = {}
    try:
        for k, v in overrides.items():
            saved[k] = sd.scaled.get(k, None)
            if torch.is_tensor(v):
                sd.scaled[k] = v.item()
            else:
                sd.scaled[k] = float(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sd.scaled.pop(k, None)
            else:
                sd.scaled[k] = v


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
OUT = ROOT / "results/z398_phenom_ii"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"


# --------------------------------------------------------------------------- #
#  Data + Sebas-card overrides (replicated from z365)                         #
# --------------------------------------------------------------------------- #
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


def build_pyport_base():
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    # newton_max_iters=20 during BBO (was 40 in z365). Phenom Iii under
    # high A pushes Newton into regions where extra iterations don't help;
    # capping iter count keeps per-eval wall time bounded.
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=20)
    cfg.bjt_emitter_to_gnd = True
    cfg.body_pdiode_to = "vnwell"
    cfg.use_well_diode = True
    cfg.vnwell = 2.0
    cfg.body_pdiode_Js = 5.3675e-7 / 22e-12
    cfg.body_pdiode_n = 1.0535
    cfg.body_pdiode_Rs = 1.0e6
    # S5-B switch ON; per-curve params overwritten in eval loop
    cfg.use_phenom_ii = True
    cfg.phenom_ii_on_M2 = False
    cfg.phenom_ii_A = 0.0
    cfg.phenom_ii_Vknee = 1.5
    cfg.phenom_ii_Vsharp = 0.1
    # Frozen-from-R-39 BJT (kept as routing parasitic; iii flows through it)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Va = 0.903; bjt.Is = 5.95e-12; bjt.Bf = 991.0
    return cfg, M1, M2, bjt


# --------------------------------------------------------------------------- #
#  9-dim parameter mapping                                                    #
# --------------------------------------------------------------------------- #
VG1_KEYS = [0.20, 0.40, 0.60]
# Per-branch (A, V_knee, Vsharp). A bounded in log10 for better DE coverage.
# Upper A capped at 10^3.3 ≈ 2000 (gives ~3.3 dec fold ceiling — fits 2.2 dec
# target with margin). Going higher induces Newton non-convergence in low-VG2
# curves where the BJT pump is already strong (single-eval >20 min observed).
PARAM_BOUNDS = [
    (0.0, 3.3),       # log10(A) → A ∈ [1, 2000]
    (0.5, 2.5),       # V_knee   [V]
    (0.02, 0.5),      # Vsharp   [V]  (≥20mV solver-friendly floor)
] * 3


def unpack(x):
    out = {}
    for i, vg1 in enumerate(VG1_KEYS):
        A      = float(10.0 ** x[3*i + 0])
        Vknee  = float(x[3*i + 1])
        Vsharp = float(x[3*i + 2])
        out[vg1] = (A, Vknee, Vsharp)
    return out


_STATE = {}


def _eval_curve(c, params, sd_M1, sd_M2):
    cfg = _STATE["cfg"]; M1 = _STATE["M1"]; M2 = _STATE["M2"]; bjt = _STATE["bjt"]
    sebas_rows = _STATE["sebas_rows"]; forward_2t = _STATE["forward_2t"]
    vg1_key = round(c["VG1"], 2)
    if vg1_key not in params:
        return float("nan"), None
    A, Vknee, Vsharp = params[vg1_key]
    cfg.phenom_ii_A = A
    cfg.phenom_ii_Vknee = Vknee
    cfg.phenom_ii_Vsharp = Vsharp
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
        if mask.sum() >= 3:
            rmse = float(np.sqrt(np.mean(
                (np.log10(Id_pred[mask]) - np.log10(c["Id"][mask])) ** 2)))
        else:
            rmse = 10.0
        return rmse, Id_pred
    except Exception:
        return 10.0, None


def eval_full(x):
    """BBO cost: cell-wide log10-RMSE median over the curves currently in
    _STATE["bbo_curves"] (a representative subset of the full 33 to keep
    each eval cheap). The final refit uses ALL 33 curves regardless.
    """
    t0 = time.time()
    cfg = _STATE["cfg"]; M1 = _STATE["M1"]; M2 = _STATE["M2"]
    sd_M1 = cfg.size_dep_M1(M1)
    sd_M2 = cfg.size_dep_M2(M2)
    params = unpack(x)
    rmses = []
    # Per-eval wall-time guard.
    EVAL_BUDGET_S = 60.0
    bailed = False
    for c in _STATE["bbo_curves"]:
        if time.time() - t0 > EVAL_BUDGET_S:
            bailed = True
            break
        rmse, _ = _eval_curve(c, params, sd_M1, sd_M2)
        if not (rmse is None or math.isnan(rmse)):
            rmses.append(rmse)
    if bailed:
        cost = 10.0
    else:
        cost = float(np.median(rmses)) if rmses else 10.0
    _STATE["history"].append({
        "iter": len(_STATE["history"]),
        "x": [float(v) for v in x],
        "cost_median_dec": cost,
        "elapsed_s": time.time() - t0,
    })
    if (len(_STATE["history"]) % 5) == 1:
        print(f"  [eval {len(_STATE['history']):3d}] cost={cost:.4f} dec  "
              f"({time.time()-t0:.1f}s)", flush=True)
        (OUT / "bbo_history.json").write_text(json.dumps({
            "history": _STATE["history"],
            "best_so_far": min(_STATE["history"], key=lambda h: h["cost_median_dec"]),
        }, indent=2))
    return cost


def _fold_dec(Vd, Id):
    """log10 ratio of peak Id to post-peak min Id (snapback fold magnitude)."""
    Id = np.asarray(Id); Vd = np.asarray(Vd)
    mask = np.isfinite(Id) & (Id > 0)
    if mask.sum() < 5:
        return float("nan")
    Id_m = Id[mask]; Vd_m = Vd[mask]
    ipk = int(np.argmax(Id_m))
    if ipk >= len(Id_m) - 1:
        return 0.0
    Id_post = Id_m[ipk+1:]
    if len(Id_post) == 0 or Id_post.min() <= 0:
        return 0.0
    return float(np.log10(Id_m[ipk] / max(Id_post.min(), 1e-30)))


def final_refit(x_best):
    cfg = _STATE["cfg"]; M1 = _STATE["M1"]; M2 = _STATE["M2"]
    sd_M1 = cfg.size_dep_M1(M1)
    sd_M2 = cfg.size_dep_M2(M2)
    params = unpack(x_best)
    results = []
    per_vg1 = {0.20: [], 0.40: [], 0.60: []}
    # Track VG1=0.6 fold magnitudes (model vs meas) for gate check
    folds_06 = []
    folds_all = {0.20: [], 0.40: [], 0.60: []}
    plot_data = []  # for snapback_phenom_fit.png
    for c in _STATE["curves"]:
        vg1_key = round(c["VG1"], 2)
        rmse, Id_pred = _eval_curve(c, params, sd_M1, sd_M2)
        results.append({
            "VG1": c["VG1"], "VG2": c["VG2"], "log_rmse_dec": rmse,
            "fold_meas_dec": _fold_dec(c["Vd"], c["Id"]),
            "fold_pred_dec": _fold_dec(c["Vd"], Id_pred) if Id_pred is not None else float("nan"),
        })
        if not math.isnan(rmse):
            per_vg1[vg1_key].append(rmse)
        if Id_pred is not None and vg1_key in folds_all:
            fp = _fold_dec(c["Vd"], Id_pred)
            fm = _fold_dec(c["Vd"], c["Id"])
            folds_all[vg1_key].append({"VG2": c["VG2"], "fold_meas": fm, "fold_pred": fp})
            if vg1_key == 0.60:
                folds_06.append(fp)
        plot_data.append({
            "VG1": c["VG1"], "VG2": c["VG2"],
            "Vd": c["Vd"].tolist(), "Id_meas": c["Id"].tolist(),
            "Id_pred": Id_pred.tolist() if Id_pred is not None else None,
        })
    valid = [r["log_rmse_dec"] for r in results if not math.isnan(r["log_rmse_dec"])]
    per_vg1_med = {
        f"{k:.2f}": (float(np.median([x for x in v if not math.isnan(x)])) if v else None)
        for k, v in per_vg1.items()
    }
    # Aggregate fold metrics
    fold_pred_06_median = float(np.nanmedian(folds_06)) if folds_06 else float("nan")
    fold_pred_06_max = float(np.nanmax(folds_06)) if folds_06 else float("nan")
    return {
        "cell_wide_median_dec": float(np.median(valid)) if valid else None,
        "per_VG1_median": per_vg1_med,
        "n_valid": len(valid), "n_total": len(results),
        "per_curve": results,
        "folds_per_VG1": folds_all,
        "fold_pred_VG1_0p6_median_dec": fold_pred_06_median,
        "fold_pred_VG1_0p6_max_dec": fold_pred_06_max,
        "plot_data": plot_data,
    }


def make_plot(refit, params, out_path):
    plot_data = refit["plot_data"]
    vg1_groups = {0.20: [], 0.40: [], 0.60: []}
    for d in plot_data:
        k = round(d["VG1"], 2)
        if k in vg1_groups:
            vg1_groups[k].append(d)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, vg1 in zip(axes, [0.20, 0.40, 0.60]):
        items = sorted(vg1_groups[vg1], key=lambda d: d["VG2"])
        cmap = plt.get_cmap("viridis")
        n = max(len(items), 1)
        for i, d in enumerate(items):
            color = cmap(i / max(n - 1, 1))
            ax.semilogy(d["Vd"], np.maximum(np.array(d["Id_meas"]), 1e-15),
                        "o", color=color, markersize=2.5, alpha=0.55)
            if d["Id_pred"] is not None:
                ax.semilogy(d["Vd"], np.maximum(np.array(d["Id_pred"]), 1e-15),
                            "-", color=color, linewidth=1.0, alpha=0.85,
                            label=f"VG2={d['VG2']:.2f}")
        A, Vk, Vs = params[vg1]
        med = refit["per_VG1_median"].get(f"{vg1:.2f}")
        ax.set_title(f"VG1={vg1:.2f}  A={A:.1f}  Vknee={Vk:.2f}  Vsharp={Vs:.3f}\n"
                     f"med-rmse={med:.3f} dec" if med is not None else f"VG1={vg1:.2f}")
        ax.set_xlabel("Vd [V]")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_ylim(1e-12, 1e-2)
        if vg1 == 0.20:
            ax.set_ylabel("|Id|  [A]")
    fig.suptitle("z398 S5-B: phenomenological II S-curve fit  (dots=meas, lines=model)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    print(f"[z398] S5-B PHENOM-II BBO  (9-dim, popsize=12, maxiter=8)", flush=True)
    cfg, M1, M2, bjt = build_pyport_base()
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    sebas_rows = load_sebas_params()
    curves = load_curves()
    print(f"[z398] loaded {len(curves)} curves, {len(sebas_rows)} Sebas rows",
          flush=True)

    # Representative subset for BBO: 2 curves per VG1 × 3 VG1 = 6-9 curves.
    # We pick the lowest-VG2 (deepest snapback) and a mid-VG2 (intermediate)
    # for each VG1 — these are the most fold-discriminating biases. Final
    # refit uses ALL 33 curves.
    bbo_curves = []
    by_vg1 = {0.20: [], 0.40: [], 0.60: []}
    for c in curves:
        k = round(c["VG1"], 2)
        if k in by_vg1:
            by_vg1[k].append(c)
    for k in (0.20, 0.40, 0.60):
        cs = sorted(by_vg1[k], key=lambda c: c["VG2"])
        if not cs:
            continue
        # take lowest, ~median, highest VG2 (3 per VG1 → 9 total)
        n = len(cs)
        idxs = sorted(set([0, n // 2, n - 1]))
        bbo_curves.extend([cs[i] for i in idxs])
    print(f"[z398] BBO subset = {len(bbo_curves)} curves "
          f"(out of {len(curves)} total)", flush=True)

    _STATE.update({
        "cfg": cfg, "M1": M1, "M2": M2, "bjt": bjt,
        "sebas_rows": sebas_rows, "curves": curves,
        "bbo_curves": bbo_curves,
        "forward_2t": forward_2t,
        "history": [],
    })

    print("[z398] timing single eval...", flush=True)
    # Sensible starting point: moderate fold at each VG1 (will be replaced by Sobol init)
    # Conservative start: small fold amplitudes to keep Newton stable for
    # the first eval (large A induces body-pump runaway / slow convergence).
    x0 = np.array([
        1.0, 1.8, 0.15,   # VG1=0.20: A=10,   knee=1.8V, sharp=0.15V
        1.5, 1.6, 0.12,   # VG1=0.40: A=32,   knee=1.6V, sharp=0.12V
        2.0, 1.4, 0.10,   # VG1=0.60: A=100,  knee=1.4V, sharp=0.10V
    ])
    tt = time.time()
    c0 = eval_full(x0)
    one_eval = time.time() - tt
    print(f"[z398] one eval = {one_eval:.1f}s  cost={c0:.4f} dec", flush=True)

    # Budget guard: with popsize=12, maxiter=8 → ~96 evals + init. If
    # one_eval > 25s we trim to popsize=10/maxiter=6 to stay under 90 min.
    # Budget mapping: target ≤75 min DE total.
    # popsize multiplier × dim(9) = pop size. nfev ≈ pop × (1 + maxiter).
    BBO_BUDGET_S = 75 * 60
    if one_eval <= 0.5:
        one_eval = 1.0
    nfev_budget = max(int(BBO_BUDGET_S / max(one_eval, 1.0)), 20)
    # Solve pop=popsize*9, nfev≈pop*(1+maxiter). Keep popsize≥1, maxiter≥2.
    if nfev_budget >= 90:
        popsize, maxiter = 2, 4   # 18 pop × 5 ≈ 90 evals
    elif nfev_budget >= 50:
        popsize, maxiter = 1, 5   # 9 pop × 6 ≈ 54 evals
    elif nfev_budget >= 30:
        popsize, maxiter = 1, 3   # 9 × 4 ≈ 36
    else:
        popsize, maxiter = 1, 2   # 9 × 3 ≈ 27
    print(f"[z398] eval={one_eval:.1f}s → nfev_budget={nfev_budget}, "
          f"popsize={popsize} (pop={popsize*9}), maxiter={maxiter}",
          flush=True)

    print(f"[z398] DE running (popsize={popsize}, maxiter={maxiter})...",
          flush=True)
    result = differential_evolution(
        eval_full,
        bounds=PARAM_BOUNDS,
        popsize=popsize,
        maxiter=maxiter,
        tol=1e-3,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=98,
        polish=False,
        init="sobol",
        updating="immediate",
        workers=1,
        x0=x0,
        disp=True,
    )
    x_best = result.x
    cost_best = float(result.fun)
    print(f"[z398] DE DONE  cost_best={cost_best:.4f} dec  nfev={result.nfev}",
          flush=True)

    print("[z398] final 33-curve refit...", flush=True)
    refit = final_refit(x_best)
    # Save without plot_data (too large for summary)
    refit_for_save = {k: v for k, v in refit.items() if k != "plot_data"}
    (OUT / "final_refit_full_33.json").write_text(
        json.dumps(refit_for_save, indent=2))

    params = unpack(x_best)
    per_vg1_params = {
        f"{vg1:.2f}": {"A": A, "V_knee": Vknee, "Vsharp": Vsharp}
        for vg1, (A, Vknee, Vsharp) in params.items()
    }
    (OUT / "best_params_per_VG1.json").write_text(
        json.dumps(per_vg1_params, indent=2))

    # Plot
    try:
        make_plot(refit, params, OUT / "snapback_phenom_fit.png")
        print("[z398] plot saved", flush=True)
    except Exception as e:
        print(f"[z398] plot FAILED: {e}", flush=True)

    cell_med = refit["cell_wide_median_dec"]
    fold06 = refit["fold_pred_VG1_0p6_max_dec"]
    gate_DISCOVERY = bool(
        cell_med is not None and cell_med < 0.85 and
        isinstance(fold06, float) and fold06 > 0.5
    )
    gate_AMBITIOUS = bool(
        cell_med is not None and cell_med < 0.30 and
        isinstance(fold06, float) and fold06 > 1.5
    )
    gate_KILL = bool(cell_med is not None and cell_med > 1.5)
    gate_INFRA = bool(
        cell_med is not None and (time.time() - t0) < 60 * 60 and
        refit["n_valid"] == refit["n_total"]
    )

    summary = {
        "script": "z398_phenom_ii",
        "phase": "S5-B post-S4 KILL-SHOT recovery attempt #2",
        "honesty_caveat": (
            "EMPIRICAL curve-fitting, NOT derived MOSFET physics. The "
            "phenomenological S-curve Iii=|Ids|·A·sigmoid((Vd-Vknee)/Vsharp) "
            "has 9 free parameters fitted directly to the snapback envelope. "
            "A successful fit shows the topology bottleneck is the BSIM4 "
            "IIMOD formula, not the BJT routing path — it does NOT validate "
            "the underlying mechanism. Honest framing: this is a "
            "feasibility test, not a physics claim."
        ),
        "patches_active": [
            "R-20 BJT Vbc", "R-29 Vth/tox", "R-37 binunit",
            "R-41 body_pdiode_to=vnwell + use_well_diode=True",
            "R-39 BJT Va/Is/Bf frozen (0.903, 5.95e-12, 991.0)",
            "R-45 cfg.vnwell=2.0 frozen",
            "S5-B cfg.use_phenom_ii=True (REPLACES BSIM4 §6.1 Iii on M1)",
        ],
        "param_bounds": {
            "log10_A":  [PARAM_BOUNDS[0][0], PARAM_BOUNDS[0][1]],
            "V_knee_V": [PARAM_BOUNDS[1][0], PARAM_BOUNDS[1][1]],
            "Vsharp_V": [PARAM_BOUNDS[2][0], PARAM_BOUNDS[2][1]],
        },
        "per_VG1_params_best": per_vg1_params,
        "cell_wide_median_dec": cell_med,
        "per_VG1_median": refit["per_VG1_median"],
        "fold_pred_VG1_0p6_median_dec": refit["fold_pred_VG1_0p6_median_dec"],
        "fold_pred_VG1_0p6_max_dec": refit["fold_pred_VG1_0p6_max_dec"],
        "n_valid": refit["n_valid"], "n_total": refit["n_total"],
        "n_evals": int(result.nfev),
        "de_params": {"popsize": popsize, "maxiter": maxiter, "seed": 98},
        "one_eval_s": one_eval,
        "baselines": {
            "z363_R43_global": 1.131,
            "z365_R46_perVG1_engineering": 0.91,
            "ngspice_target_aspiration": 0.27,
        },
        "gates_preregistered": {
            "INFRA_done_lt_60min_no_nan": gate_INFRA,
            "DISCOVERY_cellwide_lt_0p85_AND_fold06_gt_0p5": gate_DISCOVERY,
            "AMBITIOUS_cellwide_lt_0p30_AND_fold06_gt_1p5": gate_AMBITIOUS,
            "KILL_SHOT_cellwide_gt_1p5": gate_KILL,
        },
        "elapsed_s": time.time() - t0,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "bbo_history.json").write_text(json.dumps({
        "history": _STATE["history"],
        "best": {"x": [float(v) for v in x_best], "cost": cost_best},
    }, indent=2))

    print(f"\n[z398] DONE")
    print(f"  cell-wide median = {cell_med}")
    print(f"  per-VG1          = {refit['per_VG1_median']}")
    print(f"  fold@VG1=0.6 max = {fold06}")
    print(f"  params           = {json.dumps(per_vg1_params, indent=2)}")
    print(f"  INFRA={gate_INFRA}  DISCOVERY={gate_DISCOVERY}  "
          f"AMBITIOUS={gate_AMBITIOUS}  KILL-SHOT={gate_KILL}")
    print(f"  elapsed = {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
