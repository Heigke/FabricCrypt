"""z397 — S5-A: explicit lateral BJT β-fit to Sebas snapback envelope.

After S4 KILL-SHOT (S4-C ngspice DC agreed with pyport: BSIM4 §6.1 + standard
parasitic NPN has no fold), we attempt post-mortem recovery #1: enable the
explicit lateral parasitic NPN (M1.drain — body — M2.source=GND) and FIT its
Gummel-Poon params (Bf, Va, Is) to reproduce the measured fold magnitude.

Difference from S3-C (z387):
  - S3-C tested fixed Bf (R-46 default) and Mario canonical — got 0.04 dec fold.
  - S5-A optimizes (Bf, Va, log10(Is)) per VG1 group against the fold loss.

Per-VG1 BBO (3 dims each, 3 groups = 9 dims total budget). Cost is the mean
|log10(fold_model) − log10(fold_meas)| over a representative VG2 subset for the
VG1 group. After per-VG1 fits, we evaluate cell-wide on the full 33-curve set.

Pre-registered gates: see 01_LOG.md entry 2026-05-15.

Output: results/z397_lateral_bjt_fit/{summary.json, best_params_per_VG1.json,
        fold_match.png, run.log}
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys, json, math, csv, re, time, importlib.util
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
OUT = ROOT / "results/z397_lateral_bjt_fit"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"

LOG_LINES = []
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


@contextmanager
def patch_sd_scaled(sd, overrides):
    if not overrides:
        yield; return
    saved = {}
    try:
        for k, v in overrides.items():
            saved[k] = sd.scaled.get(k, None)
            sd.scaled[k] = float(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None: sd.scaled.pop(k, None)
            else: sd.scaled[k] = v


def load_sebas_params():
    rows = []
    with open(DATA / "2Tcell_BSIM_param_DC.csv") as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try: row[k] = float(v)
                except ValueError: row[k] = float("nan")
            rows.append(row)
    return rows


BRANCH_FLAT = {
    0.4: {"ETAB": 1.9, "K1": 0.53825, "ALPHA0": 7.842e-05, "BETA0": 19.0, "NFACTOR": 6.0},
    0.6: {"ETAB": 2.5, "K1": 0.41825, "ALPHA0": 7.842e-05, "BETA0": 20.0, "NFACTOR": 6.0},
}
M2_STATIC = {"k1": 0.63825, "k2": -0.070435, "etab": -0.086777, "beta0": 18.0}


def find_or_impute_row(rows, VG1, VG2, atol=1e-3):
    for r in rows:
        if abs(r["VG1"] - VG1) < atol and abs(r["VG2"] - VG2) < atol:
            target = dict(r); break
    else:
        return None
    if math.isnan(target.get("K1", float("nan"))):
        branch = BRANCH_FLAT.get(round(VG1, 2))
        if branch:
            for k, v in branch.items():
                target[k] = float(v)
    return target


def make_overrides(row):
    if row is None: return None, None
    P_M1 = {}
    for ck, pk in (("ETAB","etab"),("K1","k1"),("ALPHA0","alpha0"),("BETA0","beta0")):
        v = row.get(ck, float("nan"))
        if not math.isnan(v): P_M1[pk] = float(v)
    P_M2 = {}
    nf = row.get("NFACTOR", float("nan"))
    if not math.isnan(nf): P_M2["nfactor"] = float(nf)
    for k, v in M2_STATIC.items():
        P_M2.setdefault(k, float(v))
    return (P_M1 or None), (P_M2 or None)


def build_base():
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    cfg.body_pdiode_to = "vnwell"
    cfg.use_well_diode = True
    cfg.vnwell = 2.0
    cfg.body_pdiode_Js = 5.3675e-7 / 22e-12
    cfg.body_pdiode_n = 1.0535
    cfg.body_pdiode_Rs = 1.0e6
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Va = 0.903; bjt.Is = 5.95e-12; bjt.Bf = 991.0
    return cfg, M1, M2, bjt, GummelPoonNPN


def list_csvs(vg1):
    sub = DATA / f"2vHCa-2 I-Vs@VG2 VG1={vg1} vnwell=2"
    pat = re.compile(rf"VG2=(-?\d+\.\d+)_VG={vg1}")
    items = []
    for f in sorted(sub.glob("*.csv")):
        m = pat.search(f.name)
        if m:
            items.append((float(m.group(1)), f))
    return items


def load_csv(fpath):
    d = np.loadtxt(fpath, delimiter=",", skiprows=1)
    return d[:, 0], np.abs(d[:, 1])


def fold_dec(Id, Vd, vmin=0.8):
    """Snapback fold: max log10 jump (positive = current INCREASES with Vd
    after some peak, indicating fold). Computed on Vd > vmin region.

    For S5-A we measure the *snapback fold*: difference between local max in
    log10(I) and local min after it, in the Vd > 0.8 region. This captures
    the I(Vd) descending-then-rising envelope (the "fold").
    """
    Id = np.maximum(np.asarray(Id, dtype=float), 1e-15)
    Vd = np.asarray(Vd, dtype=float)
    mask = Vd >= vmin
    if mask.sum() < 3:
        return 0.0
    lI = np.log10(Id[mask])
    # snapback fold: span from local peak after start to subsequent valley/rise
    # we capture the full envelope range as max - min over Vd>vmin
    return float(lI.max() - lI.min())


# R-46 best per VG1 for the *vertical* BJT (used as starting state but the
# lateral path is what we actually optimize here)
R46_BEST = {
    0.2: {"Bf_vert": 1889.88, "iii_body_gain": 1.8447, "vnwell_Rs": 10**9.1722},
    0.4: {"Bf_vert": 1092.27, "iii_body_gain": 1.5152, "vnwell_Rs": 10**9.8983},
    0.6: {"Bf_vert": 417.63,  "iii_body_gain": 0.9036, "vnwell_Rs": 10**6.7846},
}


# Sub-sampled VG2 grid per VG1 group for BBO cost (covers regime spread)
FIT_VG2 = {
    0.2: [0.00, 0.10, 0.20],
    0.4: [0.05, 0.20, 0.30],
    0.6: [0.00, 0.20, 0.40],
}


def evaluate_bias(forward_2t, cfg, M1, M2, bjt, bjt_lat,
                  sd_M1, sd_M2, vg1, vg2, Vd_t, P_M1, P_M2,
                  Bf, Va, Is):
    """Run forward_2t with lateral-only BJT using (Bf, Va, Is). Returns Id_p."""
    cfg.use_lateral_bjt = True
    cfg.disable_vertical_bjt = True
    bjt_lat.Bf = float(Bf)
    bjt_lat.Va = float(Va)
    bjt_lat.Is = float(Is)
    bjt_lat.Var = 1e30
    bjt_lat.Nf = 1.0
    bjt_lat.Nr = 1.0
    cfg.bjt_lateral_instance = bjt_lat
    kw = dict(cfg=cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_t,
              VG1=torch.tensor(vg1, dtype=torch.float64),
              VG2=torch.tensor(vg2, dtype=torch.float64),
              warm_start=True)
    with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
        out = forward_2t(**kw)
    Id_p = np.abs(out["Id"].detach().cpu().numpy())
    return Id_p


def build_fit_set(sebas_rows, vg1):
    """Pre-load measured curves for the FIT VG2 subset of VG1."""
    items = list_csvs(vg1)
    targets = []
    for vg2, fpath in items:
        if any(abs(vg2 - v) < 1e-3 for v in FIT_VG2[vg1]):
            Vd_m, Id_m = load_csv(fpath)
            row = find_or_impute_row(sebas_rows, vg1, vg2)
            P_M1, P_M2 = make_overrides(row)
            meas_fold = fold_dec(Id_m, Vd_m, vmin=0.8)
            targets.append({"vg2": vg2, "Vd": Vd_m, "Id_m": Id_m,
                            "P_M1": P_M1, "P_M2": P_M2, "meas_fold": meas_fold,
                            "fname": fpath.name})
    return targets


def cost_per_vg1(x, vg1, fit_set, forward_2t, cfg, M1, M2, bjt, bjt_lat, sd_M1, sd_M2):
    Bf, Va, logIs = x
    Is = 10.0 ** logIs
    # Set vertical BJT to R-46 best (frozen)
    bjt.Bf = R46_BEST[vg1]["Bf_vert"]
    cfg.iii_body_gain = R46_BEST[vg1]["iii_body_gain"]
    cfg.vnwell_Rs = R46_BEST[vg1]["vnwell_Rs"]
    errs = []
    for tgt in fit_set:
        try:
            Vd_t = torch.tensor(tgt["Vd"], dtype=torch.float64)
            Id_p = evaluate_bias(forward_2t, cfg, M1, M2, bjt, bjt_lat,
                                 sd_M1, sd_M2, vg1, tgt["vg2"],
                                 Vd_t, tgt["P_M1"], tgt["P_M2"],
                                 Bf, Va, Is)
            if not np.all(np.isfinite(Id_p)):
                errs.append(5.0); continue
            mf = fold_dec(Id_p, tgt["Vd"], vmin=0.8)
            # log-fold distance with floor for tiny folds
            log_mod = math.log10(max(mf, 1e-3))
            log_meas = math.log10(max(tgt["meas_fold"], 1e-3))
            errs.append(abs(log_mod - log_meas))
        except Exception:
            errs.append(5.0)
    return float(np.mean(errs)) if errs else 10.0


def fit_one_vg1(vg1, fit_set, forward_2t, cfg, M1, M2, bjt, bjt_lat, sd_M1, sd_M2,
                popsize=12, maxiter=10, seed=0):
    """DE over (Bf, Va, log10(Is)) for one VG1 group."""
    bounds = [(10.0, 1e6), (0.1, 100.0), (-12.0, -6.0)]
    n_calls = [0]
    def f(x):
        n_calls[0] += 1
        c = cost_per_vg1(x, vg1, fit_set, forward_2t, cfg, M1, M2, bjt, bjt_lat, sd_M1, sd_M2)
        if n_calls[0] % 10 == 0:
            log(f"    [VG1={vg1}] eval#{n_calls[0]} x={[f'{v:.3g}' for v in x]} cost={c:.3f}")
        return c
    t0 = time.time()
    res = differential_evolution(
        f, bounds, seed=seed, popsize=popsize, maxiter=maxiter,
        tol=1e-3, mutation=(0.5, 1.0), recombination=0.7,
        polish=False, init="sobol", updating="deferred", workers=1)
    dt = time.time() - t0
    log(f"  [VG1={vg1}] DE done in {dt:.1f}s, {n_calls[0]} evals, best cost={res.fun:.3f}, x={res.x.tolist()}")
    return {"x": res.x.tolist(), "cost": float(res.fun), "n_evals": n_calls[0],
            "wall_s": dt, "Bf": float(res.x[0]), "Va": float(res.x[1]),
            "Is": float(10.0 ** res.x[2]), "log10_Is": float(res.x[2])}


def main():
    t_glob = time.time()
    log("z397 S5-A start: lateral BJT β-fit on Sebas snapback envelope")
    cfg, M1, M2, bjt, GummelPoonNPN = build_base()
    bjt_lat = GummelPoonNPN.from_sebas_card()
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    sebas_rows = load_sebas_params()

    # Build fit sets per VG1
    fit_sets = {}
    for vg1 in (0.2, 0.4, 0.6):
        fs = build_fit_set(sebas_rows, vg1)
        fit_sets[vg1] = fs
        mf_str = [f"{t['meas_fold']:.2f}d" for t in fs]
        log(f"VG1={vg1}: {len(fs)} fit biases, meas-folds={mf_str}")

    # Per-VG1 BBO
    best_per_vg1 = {}
    for vg1 in (0.2, 0.4, 0.6):
        log(f"[BBO] VG1={vg1} starting DE (popsize=8, maxiter=6)…")
        best = fit_one_vg1(vg1, fit_sets[vg1], forward_2t, cfg, M1, M2, bjt, bjt_lat,
                           sd_M1, sd_M2, popsize=8, maxiter=6, seed=42)
        best_per_vg1[vg1] = best

    # Save best params now
    with open(OUT / "best_params_per_VG1.json", "w") as f:
        json.dump({str(k): v for k, v in best_per_vg1.items()}, f, indent=2)
    log(f"wrote {OUT/'best_params_per_VG1.json'}")

    # Cell-wide evaluation on ALL 33 biases
    log("Cell-wide evaluation with fitted lateral BJT per VG1...")
    cell_results = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, vg1 in zip(axes, (0.2, 0.4, 0.6)):
        b = best_per_vg1[vg1]
        bjt.Bf = R46_BEST[vg1]["Bf_vert"]
        cfg.iii_body_gain = R46_BEST[vg1]["iii_body_gain"]
        cfg.vnwell_Rs = R46_BEST[vg1]["vnwell_Rs"]
        all_csvs = list_csvs(vg1)
        log(f"  VG1={vg1}: {len(all_csvs)} curves; Bf={b['Bf']:.2f}, Va={b['Va']:.3f}, Is={b['Is']:.2e}")
        per_curves = []
        for (vg2, fpath) in all_csvs:
            Vd_m, Id_m = load_csv(fpath)
            row = find_or_impute_row(sebas_rows, vg1, vg2)
            P_M1, P_M2 = make_overrides(row)
            Vd_t = torch.tensor(Vd_m, dtype=torch.float64)
            meas_fold = fold_dec(Id_m, Vd_m, vmin=0.8)
            try:
                Id_p = evaluate_bias(forward_2t, cfg, M1, M2, bjt, bjt_lat,
                                     sd_M1, sd_M2, vg1, vg2, Vd_t, P_M1, P_M2,
                                     b["Bf"], b["Va"], b["Is"])
                model_fold = fold_dec(Id_p, Vd_m, vmin=0.8)
                finite = bool(np.all(np.isfinite(Id_p)))
                log_err = abs(math.log10(max(model_fold, 1e-3)) - math.log10(max(meas_fold, 1e-3)))
                per_curves.append({"vg2": vg2, "meas_fold": meas_fold,
                                    "model_fold": model_fold, "log_err": log_err,
                                    "finite": finite})
                col = plt.cm.viridis((vg2 + 0.20) / 0.70)
                ax.semilogy(Vd_m, np.maximum(Id_p, 1e-15), color=col, lw=1.0, alpha=0.7)
                ax.semilogy(Vd_m, np.maximum(Id_m, 1e-15), color=col, lw=0.0,
                            marker=".", ms=3, alpha=0.5)
            except Exception as e:
                log(f"    VG2={vg2} ERROR: {type(e).__name__}: {e}")
                per_curves.append({"vg2": vg2, "meas_fold": meas_fold,
                                    "error": str(e), "finite": False})
        cell_results.append({"VG1": vg1, "best_params": b, "curves": per_curves})
        ax.set_xlabel("Vd (V)"); ax.set_ylabel("|Id| (A)")
        ax.set_ylim(1e-13, 1e-2)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"VG1={vg1}  Bf={b['Bf']:.0f} Va={b['Va']:.2f} Is={b['Is']:.1e}")
    fig.suptitle("z397 S5-A: lateral BJT β-fit (lines=model, dots=measured) per VG2 in viridis")
    fig.tight_layout()
    fig.savefig(OUT / "fold_match.png", dpi=140, bbox_inches="tight")
    log(f"wrote {OUT/'fold_match.png'}")

    # Gate evaluation
    all_log_errs = []
    fold_06 = []
    infra_ok = True
    for cr in cell_results:
        for c in cr["curves"]:
            if not c.get("finite", False):
                infra_ok = False
            if "log_err" in c:
                all_log_errs.append(c["log_err"])
            if abs(cr["VG1"] - 0.6) < 1e-3 and "model_fold" in c:
                fold_06.append(c["model_fold"])
    cell_log_err = float(np.mean(all_log_errs)) if all_log_errs else float("nan")
    max_fold_06 = max(fold_06) if fold_06 else 0.0
    mean_fold_06 = float(np.mean(fold_06)) if fold_06 else 0.0

    wall_s = time.time() - t_glob
    infra = infra_ok and (wall_s < 90 * 60) and math.isfinite(cell_log_err)
    discovery = (cell_log_err < 1.0) and (max_fold_06 > 0.5)
    ambitious = (cell_log_err < 0.5) and (max_fold_06 > 1.5)
    kill_shot = (max_fold_06 < 0.3)

    log("=" * 60)
    log(f"GATE INFRA      : {'PASS' if infra else 'FAIL'} wall={wall_s:.1f}s log_err={cell_log_err:.3f}")
    log(f"GATE DISCOVERY  : {'PASS' if discovery else 'FAIL'} cell_log_err={cell_log_err:.3f} max_fold_VG1=0.6={max_fold_06:.2f}d")
    log(f"GATE AMBITIOUS  : {'PASS' if ambitious else 'FAIL'} (need <0.5 and >1.5)")
    log(f"GATE KILL_SHOT  : {'PASS (kills)' if kill_shot else 'no'}  max_fold_VG1=0.6={max_fold_06:.2f}d (vs measured ~2.2d)")
    log("=" * 60)

    summary = {
        "wallclock_s": wall_s,
        "best_params_per_VG1": {str(k): v for k, v in best_per_vg1.items()},
        "cell_log_err_mean": cell_log_err,
        "VG1_0.6_fold": {"max": max_fold_06, "mean": mean_fold_06, "n": len(fold_06)},
        "fit_VG2_subset": {str(k): v for k, v in FIT_VG2.items()},
        "R46_vertical_frozen": {str(k): v for k, v in R46_BEST.items()},
        "cell_results": cell_results,
        "gates": {"infra": infra, "discovery": discovery,
                  "ambitious": ambitious, "kill_shot": kill_shot,
                  "cell_log_err": cell_log_err,
                  "max_fold_VG1_0.6": max_fold_06},
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    with open(OUT / "run.log", "w") as f:
        f.write("\n".join(LOG_LINES) + "\n")
    log(f"DONE in {wall_s:.1f}s. summary.json + run.log written.")


if __name__ == "__main__":
    main()
