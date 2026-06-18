"""z399 — S6 Phase B: Combined BBO over S6 Phase A physics rebuild.

Free parameters (10-dim):
  v_dnw           ∈ [0.3, 1.5] V
  Bf_vert         ∈ [1000, 50000]
  log10(Is_vert)  ∈ [-10, -7]
  etab_scale      ∈ [0.5, 5.0]
  R_M2_scale      ∈ [0.1, 10.0]
  cb_override     ∈ [0.1e-15, 5e-15] F  (sampled linear)
  tau_body        ∈ [1e-3, 200e-3] s    (sampled linear)
  iii_body_gain   ∈ [0.1, 3.0]
  Bf_lateral      ∈ [100, 10000]  (the cell's primary Q1 GummelPoon Bf)
  log10(vnwell_Rs)∈ [3, 8]

Cost = mean(cell-wide log_rmse over 33 curves) + 2.0 * mean(|V_kink_model - V_kink_meas|/V_kink_meas)

BBO: scipy DE, popsize=30, maxiter=50, tol=1e-4, sobol init.

Pre-registered gates (see research_plan/01_LOG.md 2026-05-15 S6 entry):
  INFRA      : all 5 branches compile + converge, BBO < 120 min wall.
  DISCOVERY  : cell-wide < 0.85 dec AND VG1=0.6 fold > 0.5 dec AND both
               kinks within ±0.3 V of measured.
  AMBITIOUS  : cell-wide < 0.4 dec AND fold > 1.5 dec AND both kinks reproduced.
  KILL-SHOT  : cell-wide > 1.5 dec → BSIM4 architecture truly insufficient.

After fit: per-element ablation analysis (turn each flag off individually post-fit,
report fold and cell-wide change).
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
OUT = ROOT / "results/z399_physics_rebuild"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"


# --------------------------------------------------------------------------- #
#  Data + Sebas-card overrides (replicated from z398)                         #
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
    """Build per-curve P_M1/P_M2 dicts. Note: when use_etab_vg2_curve=True
    the etab override here is IGNORED (residual patches sd_M1.scaled["etab"]
    directly from the PWL curve). We still pass it for other params."""
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
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=25)
    cfg.bjt_emitter_to_gnd = True
    cfg.body_pdiode_to = "vnwell"
    cfg.use_well_diode = True
    cfg.vnwell = 2.0
    cfg.body_pdiode_Js = 5.3675e-7 / 22e-12
    cfg.body_pdiode_n = 1.0535
    cfg.body_pdiode_Rs = 1.0e6
    # S6 Phase A flags ON; per-curve params set in eval loop
    cfg.use_vertical_npn_to_dnw = True
    cfg.use_etab_vg2_curve = True
    cfg.use_m2_as_resistor = True
    # cb_override + tau_body set per-eval
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Va = 0.903; bjt.Is = 5.95e-12; bjt.Bf = 991.0
    return cfg, M1, M2, bjt


# --------------------------------------------------------------------------- #
#  10-dim parameter mapping                                                   #
# --------------------------------------------------------------------------- #
PARAM_BOUNDS = [
    (0.3, 1.5),         # 0: v_dnw            [V]
    (1000.0, 50000.0),  # 1: Bf_vert
    (-10.0, -7.0),      # 2: log10(Is_vert)
    (0.5, 5.0),         # 3: etab_scale
    (0.1, 10.0),        # 4: m2_R_scale
    (0.1e-15, 5e-15),   # 5: cb_override      [F]
    (1e-3, 200e-3),     # 6: tau_body         [s]
    (0.1, 3.0),         # 7: iii_body_gain
    (100.0, 10000.0),   # 8: Bf_lateral (Q1 GummelPoon Bf)
    (3.0, 8.0),         # 9: log10(vnwell_Rs)
]
PARAM_NAMES = [
    "v_dnw", "Bf_vert", "log10_Is_vert", "etab_scale", "m2_R_scale",
    "cb_override", "tau_body", "iii_body_gain", "Bf_lateral", "log10_vnwell_Rs",
]


def apply_params(cfg, bjt, x):
    cfg.v_dnw           = float(x[0])
    cfg.Bf_vert         = float(x[1])
    cfg.Is_vert         = float(10.0 ** x[2])
    cfg.etab_scale      = float(x[3])
    cfg.m2_R_scale      = float(x[4])
    cfg.cb_override     = float(x[5])
    cfg.tau_body        = float(x[6])
    cfg.iii_body_gain   = float(x[7])
    bjt.Bf              = float(x[8])
    cfg.vnwell_Rs       = float(10.0 ** x[9])
    # Invalidate vert BJT cache so new Bf/Is/Va take effect
    if hasattr(cfg, "_bjt_vert_cache"):
        object.__setattr__(cfg, "_bjt_vert_cache", None)


_STATE = {}


def detect_kinks(Vd, Id):
    """Return (V_kink_low, V_kink_high) from peaks in d log10(Id)/dVd.

    A "kink" is a local extremum in the numerical derivative of log10|Id|
    over Vd. We look for low-Vd kink (Vd ∈ [0.2, 0.8]) and high-Vd kink
    (Vd ∈ [1.0, 2.0]). Returns NaN if not detectable.
    """
    Id = np.asarray(Id); Vd = np.asarray(Vd)
    mask = np.isfinite(Id) & (Id > 1e-15)
    if mask.sum() < 5:
        return float("nan"), float("nan")
    Idm = Id[mask]; Vdm = Vd[mask]
    log_id = np.log10(Idm)
    if len(log_id) < 3:
        return float("nan"), float("nan")
    dlog = np.gradient(log_id, Vdm)
    # Use |2nd derivative| peak (curvature peak ≈ kink)
    d2 = np.gradient(dlog, Vdm)
    abs_d2 = np.abs(d2)
    def find_peak(lo, hi):
        m = (Vdm >= lo) & (Vdm <= hi)
        if m.sum() < 3:
            return float("nan")
        i_loc = np.argmax(abs_d2[m])
        return float(Vdm[m][i_loc])
    V_low = find_peak(0.2, 0.8)
    V_high = find_peak(1.0, 2.0)
    return V_low, V_high


def _eval_curve(c, sd_M1, sd_M2):
    cfg = _STATE["cfg"]; M1 = _STATE["M1"]; M2 = _STATE["M2"]; bjt = _STATE["bjt"]
    sebas_rows = _STATE["sebas_rows"]; forward_2t = _STATE["forward_2t"]
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


def eval_full(x, use_subset=True):
    """BBO cost evaluation. If use_subset=True, only the BBO subset (9 curves,
    3 VG1 × 3 VG2 spanning the regime) is scored — final_refit always uses
    all 33."""
    t0 = time.time()
    cfg = _STATE["cfg"]; M1 = _STATE["M1"]; M2 = _STATE["M2"]; bjt = _STATE["bjt"]
    apply_params(cfg, bjt, x)
    sd_M1 = cfg.size_dep_M1(M1)
    sd_M2 = cfg.size_dep_M2(M2)
    rmses = []
    kink_errs = []
    curves_to_use = _STATE.get("bbo_curves") if use_subset else _STATE["curves"]
    if curves_to_use is None:
        curves_to_use = _STATE["curves"]
    for c in curves_to_use:
        rmse, Id_pred = _eval_curve(c, sd_M1, sd_M2)
        if not (rmse is None or math.isnan(rmse)):
            rmses.append(rmse)
        # kink position cost
        if Id_pred is not None:
            Vk_lo_m, Vk_hi_m = detect_kinks(c["Vd"], c["Id"])
            Vk_lo_p, Vk_hi_p = detect_kinks(c["Vd"], Id_pred)
            for Vm, Vp in ((Vk_lo_m, Vk_lo_p), (Vk_hi_m, Vk_hi_p)):
                if (np.isfinite(Vm) and np.isfinite(Vp) and Vm > 0.05):
                    kink_errs.append(abs(Vp - Vm) / Vm)
    mean_rmse = float(np.mean(rmses)) if rmses else 10.0
    mean_kink = float(np.mean(kink_errs)) if kink_errs else 1.0
    cost = mean_rmse + 2.0 * mean_kink
    _STATE["history"].append({
        "iter": len(_STATE["history"]),
        "x": [float(v) for v in x],
        "cost": cost,
        "mean_log_rmse": mean_rmse,
        "mean_kink_err": mean_kink,
        "elapsed_s": time.time() - t0,
    })
    if (len(_STATE["history"]) % 10) == 1:
        print(f"  [eval {len(_STATE['history']):4d}] cost={cost:.4f} "
              f"(rmse={mean_rmse:.3f}, kink={mean_kink:.3f})  "
              f"({time.time()-t0:.1f}s)", flush=True)
        (OUT / "bbo_history.json").write_text(json.dumps({
            "history": _STATE["history"][-200:],
            "best_so_far": min(_STATE["history"], key=lambda h: h["cost"]),
        }, indent=2))
    return cost


def _fold_dec(Vd, Id):
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


def final_refit(x_best, label="best"):
    cfg = _STATE["cfg"]; M1 = _STATE["M1"]; M2 = _STATE["M2"]; bjt = _STATE["bjt"]
    apply_params(cfg, bjt, x_best)
    sd_M1 = cfg.size_dep_M1(M1)
    sd_M2 = cfg.size_dep_M2(M2)
    results = []
    folds_06 = []
    per_vg1 = {0.20: [], 0.40: [], 0.60: []}
    plot_data = []
    kink_pairs = []
    for c in _STATE["curves"]:
        vg1_key = round(c["VG1"], 2)
        rmse, Id_pred = _eval_curve(c, sd_M1, sd_M2)
        Vk_lo_m, Vk_hi_m = detect_kinks(c["Vd"], c["Id"])
        Vk_lo_p = Vk_hi_p = float("nan")
        if Id_pred is not None:
            Vk_lo_p, Vk_hi_p = detect_kinks(c["Vd"], Id_pred)
        results.append({
            "VG1": c["VG1"], "VG2": c["VG2"], "log_rmse_dec": rmse,
            "fold_meas_dec": _fold_dec(c["Vd"], c["Id"]),
            "fold_pred_dec": _fold_dec(c["Vd"], Id_pred) if Id_pred is not None else float("nan"),
            "Vk_low_meas": Vk_lo_m, "Vk_low_pred": Vk_lo_p,
            "Vk_high_meas": Vk_hi_m, "Vk_high_pred": Vk_hi_p,
        })
        kink_pairs.append((Vk_lo_m, Vk_lo_p, Vk_hi_m, Vk_hi_p))
        if not math.isnan(rmse):
            per_vg1[vg1_key].append(rmse)
        if Id_pred is not None and vg1_key == 0.60:
            folds_06.append(_fold_dec(c["Vd"], Id_pred))
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
    fold06_med = float(np.nanmedian(folds_06)) if folds_06 else float("nan")
    fold06_max = float(np.nanmax(folds_06)) if folds_06 else float("nan")
    # Aggregate kink errors
    lo_m = np.array([k[0] for k in kink_pairs]); lo_p = np.array([k[1] for k in kink_pairs])
    hi_m = np.array([k[2] for k in kink_pairs]); hi_p = np.array([k[3] for k in kink_pairs])
    mask_lo = np.isfinite(lo_m) & np.isfinite(lo_p)
    mask_hi = np.isfinite(hi_m) & np.isfinite(hi_p)
    kink_low_err = float(np.median(np.abs(lo_p[mask_lo] - lo_m[mask_lo]))) if mask_lo.sum() else float("nan")
    kink_high_err = float(np.median(np.abs(hi_p[mask_hi] - hi_m[mask_hi]))) if mask_hi.sum() else float("nan")
    return {
        "label": label,
        "cell_wide_median_dec": float(np.median(valid)) if valid else None,
        "cell_wide_mean_dec": float(np.mean(valid)) if valid else None,
        "per_VG1_median": per_vg1_med,
        "fold_pred_VG1_0p6_median_dec": fold06_med,
        "fold_pred_VG1_0p6_max_dec": fold06_max,
        "kink_low_median_err_V": kink_low_err,
        "kink_high_median_err_V": kink_high_err,
        "n_valid": len(valid), "n_total": len(results),
        "per_curve": results,
        "plot_data": plot_data,
    }


# --------------------------------------------------------------------------- #
#  Plots                                                                      #
# --------------------------------------------------------------------------- #
def make_snapback_plot(refit, out_path):
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
        med = refit["per_VG1_median"].get(f"{vg1:.2f}")
        ax.set_title(f"VG1={vg1:.2f}  med-rmse={med:.3f} dec" if med is not None else f"VG1={vg1:.2f}")
        ax.set_xlabel("Vd [V]")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_ylim(1e-12, 1e-2)
        if vg1 == 0.20:
            ax.set_ylabel("|Id|  [A]")
    fig.suptitle("z399 S6: full physics rebuild (dots=meas, lines=model)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def make_kink_plot(refit, out_path):
    rows = refit["per_curve"]
    lo_m = np.array([r["Vk_low_meas"]  for r in rows])
    lo_p = np.array([r["Vk_low_pred"]  for r in rows])
    hi_m = np.array([r["Vk_high_meas"] for r in rows])
    hi_p = np.array([r["Vk_high_pred"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (xm, xp, title, lo, hi) in zip(
        axes,
        [(lo_m, lo_p, "Low Vd kink", 0.0, 1.0),
         (hi_m, hi_p, "High Vd kink", 0.8, 2.2)]):
        m = np.isfinite(xm) & np.isfinite(xp)
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.6, label="1:1")
        ax.fill_between([lo, hi], [lo - 0.3, hi - 0.3], [lo + 0.3, hi + 0.3],
                        color="gray", alpha=0.15, label="±0.3 V band")
        ax.scatter(xm[m], xp[m], s=14, alpha=0.7)
        ax.set_xlabel("V_kink measured [V]")
        ax.set_ylabel("V_kink model [V]")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.legend(fontsize=8)
    fig.suptitle("z399 S6: kink-position comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Ablation: turn each S6 element off post-fit, measure fold drop             #
# --------------------------------------------------------------------------- #
ABLATION_FLAGS = [
    ("vertical_npn_to_dnw", "use_vertical_npn_to_dnw"),
    ("etab_vg2_curve",      "use_etab_vg2_curve"),
    ("m2_as_resistor",      "use_m2_as_resistor"),
    ("cb_override",         "_cb_off"),     # special: set None
    ("tau_body",            "_tau_off"),    # special: set None
]


def ablation_analysis(x_best):
    cfg = _STATE["cfg"]
    results = {}
    # Baseline (all on)
    apply_params(cfg, _STATE["bjt"], x_best)
    base = final_refit(x_best, label="all_on")
    base_summary = {
        "cell_wide_median_dec": base["cell_wide_median_dec"],
        "fold_VG1_0p6_max_dec": base["fold_pred_VG1_0p6_max_dec"],
        "kink_low_err_V": base["kink_low_median_err_V"],
        "kink_high_err_V": base["kink_high_median_err_V"],
    }
    results["all_on"] = base_summary
    for label, flag in ABLATION_FLAGS:
        apply_params(cfg, _STATE["bjt"], x_best)
        saved = None
        if flag == "_cb_off":
            saved = cfg.cb_override; cfg.cb_override = None
        elif flag == "_tau_off":
            saved = cfg.tau_body; cfg.tau_body = None
        else:
            saved = getattr(cfg, flag)
            setattr(cfg, flag, False)
        try:
            r = final_refit(x_best, label=f"off_{label}")
            results[f"off_{label}"] = {
                "cell_wide_median_dec": r["cell_wide_median_dec"],
                "fold_VG1_0p6_max_dec": r["fold_pred_VG1_0p6_max_dec"],
                "kink_low_err_V":  r["kink_low_median_err_V"],
                "kink_high_err_V": r["kink_high_median_err_V"],
                "delta_cellwide_vs_base": (
                    (r["cell_wide_median_dec"] or 99.0) - (base_summary["cell_wide_median_dec"] or 0.0)),
                "delta_fold_vs_base": (
                    (base_summary["fold_VG1_0p6_max_dec"] or 0.0) - (r["fold_VG1_0p6_max_dec"] or 0.0)),
            }
        finally:
            if flag == "_cb_off":
                cfg.cb_override = saved
            elif flag == "_tau_off":
                cfg.tau_body = saved
            else:
                setattr(cfg, flag, saved)
    return results


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    print(f"[z399] S6 Phase B BBO  (10-dim, popsize=30, maxiter=50)", flush=True)
    cfg, M1, M2, bjt = build_pyport_base()
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    sebas_rows = load_sebas_params()
    curves = load_curves()
    print(f"[z399] loaded {len(curves)} curves, {len(sebas_rows)} Sebas rows",
          flush=True)
    # Build BBO subset: representative 9 curves (3 VG1 × 3 VG2 each).
    # VG1 ∈ {0.2, 0.4, 0.6}; for each VG1 pick VG2 = {min, ~0, max}.
    bbo_curves = []
    for vg1_target in (0.20, 0.40, 0.60):
        same = [c for c in curves if abs(c["VG1"] - vg1_target) < 1e-3]
        if not same:
            continue
        same.sort(key=lambda c: c["VG2"])
        if len(same) >= 3:
            picks = [same[0], same[len(same)//2], same[-1]]
        else:
            picks = same
        bbo_curves.extend(picks)
    print(f"[z399] BBO subset = {len(bbo_curves)} curves (of {len(curves)}); "
          f"final refit uses all {len(curves)}", flush=True)
    _STATE.update({
        "cfg": cfg, "M1": M1, "M2": M2, "bjt": bjt,
        "sebas_rows": sebas_rows, "curves": curves, "bbo_curves": bbo_curves,
        "forward_2t": forward_2t,
        "history": [],
    })

    print("[z399] timing single eval...", flush=True)
    # Sensible starting point (mid-of-bounds for most, defaults from plan)
    x0 = np.array([
        1.2,       # v_dnw
        10000.0,   # Bf_vert
        -8.30,     # log10(Is_vert) = log10(5e-9)
        1.0,       # etab_scale
        1.0,       # m2_R_scale
        0.5e-15,   # cb_override
        50e-3,     # tau_body
        1.0,       # iii_body_gain
        991.0,     # Bf_lateral (from R-39 frozen value)
        6.0,       # log10(vnwell_Rs) = log10(1e6)
    ])
    tt = time.time()
    c0 = eval_full(x0)
    one_eval = time.time() - tt
    print(f"[z399] one eval = {one_eval:.1f}s  cost={c0:.4f}", flush=True)

    # Budget guard: target ~120 min wall.
    # Total evals ≈ popsize · D · (maxiter+1) where D=10.
    # With popsize=30, maxiter=50: ~15000 evals. Need eval ≤ 0.5s for that.
    # Realistic budget: choose popsize × maxiter ≤ 120 * 60 / one_eval / 10 (D).
    target_budget_s = 120 * 60.0
    max_evals = max(60, int(target_budget_s / max(one_eval, 0.5)))
    # DE scipy: nfev ≈ popsize * D * (maxiter+1) where D=10.
    popsize, maxiter = 30, 50
    if one_eval > 1.0:
        # Pick (popsize, maxiter) such that popsize*10*(maxiter+1) ≤ max_evals
        # with a roughly square allocation.
        approx = max(2, int(math.sqrt(max_evals / 10.0)))
        popsize = max(8, min(30, approx))
        maxiter = max(5, min(50, approx))
    # Honest budget tied to 120-min wall per S6 plan: nfev ≈ 120*60/one_eval.
    # For D=10 DE, nfev ≈ popsize · D · (maxiter+1). Hand-tuned to fit budget.
    if one_eval > 100.0:
        popsize, maxiter = 3, 2     # ~90 evals
    elif one_eval > 40.0:
        popsize, maxiter = 4, 3     # ~160 evals
    elif one_eval > 20.0:
        popsize, maxiter = 6, 5     # ~360 evals
    elif one_eval > 8.0:
        popsize, maxiter = 12, 10   # ~1320 evals
    print(f"[z399] eval={one_eval:.1f}s → popsize={popsize}, maxiter={maxiter} "
          f"(est nfev ≈ {popsize*10*(maxiter+1)}, "
          f"est wall ≈ {popsize*10*(maxiter+1)*one_eval/60:.0f} min)", flush=True)

    print(f"[z399] DE running popsize={popsize} maxiter={maxiter}...", flush=True)
    result = differential_evolution(
        eval_full,
        bounds=PARAM_BOUNDS,
        popsize=popsize,
        maxiter=maxiter,
        tol=1e-4,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=399,
        polish=False,
        init="sobol",
        updating="immediate",
        workers=1,
        x0=x0,
        disp=True,
    )
    x_best = result.x
    cost_best = float(result.fun)
    print(f"[z399] DE DONE  cost_best={cost_best:.4f}  nfev={result.nfev}",
          flush=True)

    print("[z399] final 33-curve refit...", flush=True)
    refit = final_refit(x_best, label="best")
    refit_for_save = {k: v for k, v in refit.items() if k != "plot_data"}
    (OUT / "final_refit_full_33.json").write_text(
        json.dumps(refit_for_save, indent=2))

    print("[z399] ablation analysis (turn each S6 element off post-fit)...",
          flush=True)
    abl = ablation_analysis(x_best)
    (OUT / "ablation.json").write_text(json.dumps(abl, indent=2))

    best_params = {n: float(v) for n, v in zip(PARAM_NAMES, x_best)}
    best_params["Is_vert_A"] = float(10.0 ** x_best[2])
    best_params["vnwell_Rs_ohm"] = float(10.0 ** x_best[9])
    (OUT / "best_params.json").write_text(json.dumps(best_params, indent=2))

    try:
        make_snapback_plot(refit, OUT / "snapback_compare.png")
        make_kink_plot(refit, OUT / "kink_position_compare.png")
        print("[z399] plots saved", flush=True)
    except Exception as e:
        print(f"[z399] plot FAILED: {e}", flush=True)

    cell_med = refit["cell_wide_median_dec"]
    fold06 = refit["fold_pred_VG1_0p6_max_dec"]
    kink_lo = refit["kink_low_median_err_V"]
    kink_hi = refit["kink_high_median_err_V"]
    gate_INFRA = bool(
        cell_med is not None and (time.time() - t0) < 120 * 60 and
        refit["n_valid"] == refit["n_total"]
    )
    kinks_ok_03 = bool(
        np.isfinite(kink_lo) and np.isfinite(kink_hi)
        and kink_lo < 0.3 and kink_hi < 0.3
    )
    gate_DISCOVERY = bool(
        cell_med is not None and cell_med < 0.85 and
        isinstance(fold06, float) and fold06 > 0.5 and kinks_ok_03
    )
    gate_AMBITIOUS = bool(
        cell_med is not None and cell_med < 0.4 and
        isinstance(fold06, float) and fold06 > 1.5 and kinks_ok_03
    )
    gate_KILL = bool(cell_med is not None and cell_med > 1.5)

    summary = {
        "script": "z399_full_physics_rebuild",
        "phase": "S6 Phase B — combined BBO over rebuild from Zoom evidence",
        "honesty_note": (
            "All 5 new physics elements (vertical NPN to DNW, ETAB(VG2) PWL,"
            " M2 as V(VG2)-resistor, cb_override, tau_body) are derived from"
            " Mario/Sebas documentation per S5C_zoom_image_findings_2026-05-15.md."
            " BBO optimizes ONLY over physical-bounds parameters (v_dnw, Bf_vert,"
            " Is_vert, etab_scale, m2_R_scale, cb_override, tau_body, iii_body_gain,"
            " Bf_lateral, vnwell_Rs). Cost = mean log-RMSE + 2.0 · mean |kink-error|."
        ),
        "param_bounds": {n: list(b) for n, b in zip(PARAM_NAMES, PARAM_BOUNDS)},
        "best_params": best_params,
        "cell_wide_median_dec": cell_med,
        "cell_wide_mean_dec": refit["cell_wide_mean_dec"],
        "per_VG1_median": refit["per_VG1_median"],
        "fold_pred_VG1_0p6_median_dec": refit["fold_pred_VG1_0p6_median_dec"],
        "fold_pred_VG1_0p6_max_dec": fold06,
        "kink_low_median_err_V": kink_lo,
        "kink_high_median_err_V": kink_hi,
        "kinks_ok_within_0p3V": kinks_ok_03,
        "n_valid": refit["n_valid"], "n_total": refit["n_total"],
        "n_evals": int(result.nfev),
        "de_params": {"popsize": popsize, "maxiter": maxiter, "seed": 399},
        "one_eval_s": one_eval,
        "ablation_per_element": abl,
        "gates_preregistered": {
            "INFRA_done_lt_120min_no_nan": gate_INFRA,
            "DISCOVERY_cellwide_lt_0p85_AND_fold06_gt_0p5_AND_kinks_lt_0p3": gate_DISCOVERY,
            "AMBITIOUS_cellwide_lt_0p4_AND_fold06_gt_1p5_AND_kinks_lt_0p3": gate_AMBITIOUS,
            "KILL_SHOT_cellwide_gt_1p5": gate_KILL,
        },
        "elapsed_s": time.time() - t0,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "bbo_history.json").write_text(json.dumps({
        "history": _STATE["history"],
        "best": {"x": [float(v) for v in x_best], "cost": cost_best},
    }, indent=2))

    print(f"\n[z399] DONE")
    print(f"  cell-wide median = {cell_med}")
    print(f"  per-VG1          = {refit['per_VG1_median']}")
    print(f"  fold@VG1=0.6 max = {fold06}")
    print(f"  kink_lo / kink_hi median err = {kink_lo} / {kink_hi}")
    print(f"  INFRA={gate_INFRA}  DISCOVERY={gate_DISCOVERY}  "
          f"AMBITIOUS={gate_AMBITIOUS}  KILL-SHOT={gate_KILL}")
    print(f"  elapsed = {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
