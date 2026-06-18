"""z368 — R-49: M1 drain-body avalanche multiplier sweep.

Tests whether the structural shortfall identified in R-48 at VG1=0.6
(log10(vnwell_Rs) non-monotonic, 4 OoM off) can be fixed by adding a
PHYSICAL drain-body avalanche multiplier instead of a per-VG1 curve-fit
knob.

Stack: R-20 + R-29 + R-37 + R-41 + R-46 best per-VG1 params (Bf, iii, Rs)
+ R-49 use_dbd_avalanche=True. Sweep dbd_BV ∈ {6, 8, 10, 12, 15}.

Pre-registered gates:
  HYPOTHESIS PASS: with GLOBAL params (not per-VG1) cell-wide < 1.0 dec
  AMBITIOUS:       cell-wide < 0.5 dec
  VG1=0.6 must drop below R-46 per-VG1 fit of 0.86
  PHYSICS CHECK:   best BV ∈ [8, 12] V → realistic
                   BV < 4 or > 20 → unphysical curve-fit knob

Output: results/z368_dbd_avalanche/{sweep_summary.json, best_per_VG1.png,
physics_check.md}
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
OUT = ROOT / "results/z368_dbd_avalanche"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"


BRANCH_FLAT = {
    0.4: {"ETAB": 1.9,  "K1": 0.53825, "ALPHA0": 7.842e-05, "BETA0": 19.0, "NFACTOR": 6.0, "trise": 10.59},
    0.6: {"ETAB": 2.5,  "K1": 0.41825, "ALPHA0": 7.842e-05, "BETA0": 20.0, "NFACTOR": 6.0, "trise": 9.04},
}
M2_STATIC_OVERRIDES = {
    "k1": 0.63825, "k2": -0.070435, "etab": -0.086777, "beta0": 18.0,
}


# R-46 best per-VG1 params from z365 bbo_history.json best_so_far
R46_PER_VG1 = {
    0.20: {"Bf": 1889.8806320503354, "iii_body_gain": 1.844675445044413,   "vnwell_Rs": 10**9.172216925770044},
    0.40: {"Bf": 1092.272187024355,  "iii_body_gain": 1.5151811846066265,  "vnwell_Rs": 10**9.898274548351765},
    0.60: {"Bf": 417.62741417624056, "iii_body_gain": 0.9035713450051843,  "vnwell_Rs": 10**6.784622445702553},
}
# R-48 honest mean-of-triples (single global) — baseline for hypothesis check
R48_GLOBAL_MEAN = {
    "Bf": (R46_PER_VG1[0.20]["Bf"] + R46_PER_VG1[0.40]["Bf"] + R46_PER_VG1[0.60]["Bf"]) / 3.0,
    "iii_body_gain": (R46_PER_VG1[0.20]["iii_body_gain"] + R46_PER_VG1[0.40]["iii_body_gain"] + R46_PER_VG1[0.60]["iii_body_gain"]) / 3.0,
    "vnwell_Rs": 10 ** ((9.172216925770044 + 9.898274548351765 + 6.784622445702553) / 3.0),
}


def load_sebas_params():
    path = DATA / "2Tcell_BSIM_param_DC.csv"
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try: row[k] = float(v)
                except ValueError: row[k] = float("nan")
            rows.append(row)
    return rows


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
    cfg.vnwell = 2.0
    cfg.body_pdiode_Js = 5.3675e-7 / 22e-12
    cfg.body_pdiode_n = 1.0535
    cfg.body_pdiode_Rs = 1.0e6
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Va = 0.903; bjt.Is = 5.95e-12
    bjt.Bf = 991.0
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


def eval_all_curves(cfg, M1, M2, bjt, sebas_rows, curves, forward_2t, param_mode):
    """param_mode: 'perVG1' or 'global'."""
    sd_M1 = cfg.size_dep_M1(M1)
    sd_M2 = cfg.size_dep_M2(M2)
    results = []
    per_vg1 = {0.20: [], 0.40: [], 0.60: []}
    for c in curves:
        vg1_key = round(c["VG1"], 2)
        if param_mode == "perVG1":
            p = R46_PER_VG1[vg1_key]
        else:
            p = R48_GLOBAL_MEAN
        bjt.Bf = float(p["Bf"])
        cfg.iii_body_gain = float(p["iii_body_gain"])
        cfg.vnwell_Rs = float(p["vnwell_Rs"])
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
                rmse = float(np.sqrt(np.mean((np.log10(Id_pred[mask]) - np.log10(c["Id"][mask])) ** 2)))
            else:
                rmse = float("nan")
        except Exception:
            rmse = float("nan")
        results.append({"VG1": c["VG1"], "VG2": c["VG2"], "log_rmse_dec": rmse})
        if not math.isnan(rmse):
            per_vg1[vg1_key].append(rmse)
    valid = [r["log_rmse_dec"] for r in results if not math.isnan(r["log_rmse_dec"])]
    return {
        "cell_wide_median_dec": float(np.median(valid)) if valid else None,
        "per_VG1_median": {f"{k:.2f}": float(np.median(v)) if v else None for k, v in per_vg1.items()},
        "n_valid": len(valid), "n_total": len(results),
        "per_curve": results,
    }


def main():
    t0 = time.time()
    print(f"[z368] R-49 DBD avalanche sweep", flush=True)
    cfg, M1, M2, bjt = build_pyport_base()
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    sebas_rows = load_sebas_params()
    curves = load_curves()
    print(f"[z368] loaded {len(curves)} curves", flush=True)

    BV_sweep = [6.0, 8.0, 10.0, 12.0, 15.0]
    sweep_results = []

    # Baseline: no avalanche, per-VG1 params (should reproduce ~0.965)
    cfg.use_dbd_avalanche = False
    print(f"[z368] baseline (no avalanche, perVG1)...", flush=True)
    tt = time.time()
    base_pv = eval_all_curves(cfg, M1, M2, bjt, sebas_rows, curves, forward_2t, "perVG1")
    print(f"  cell={base_pv['cell_wide_median_dec']:.4f}  perVG1={base_pv['per_VG1_median']}  ({time.time()-tt:.1f}s)", flush=True)

    # Baseline: no avalanche, GLOBAL mean params (R-48 honest baseline ~1.19)
    print(f"[z368] baseline (no avalanche, global)...", flush=True)
    tt = time.time()
    base_gl = eval_all_curves(cfg, M1, M2, bjt, sebas_rows, curves, forward_2t, "global")
    print(f"  cell={base_gl['cell_wide_median_dec']:.4f}  perVG1={base_gl['per_VG1_median']}  ({time.time()-tt:.1f}s)", flush=True)

    # Avalanche sweep with BOTH perVG1 (sanity) and GLOBAL (the gate)
    for BV in BV_sweep:
        cfg.use_dbd_avalanche = True
        cfg.dbd_BV = BV
        cfg.dbd_n = 4.0
        print(f"[z368] BV={BV}V, n=4 ...", flush=True)
        tt = time.time()
        r_pv = eval_all_curves(cfg, M1, M2, bjt, sebas_rows, curves, forward_2t, "perVG1")
        r_gl = eval_all_curves(cfg, M1, M2, bjt, sebas_rows, curves, forward_2t, "global")
        print(f"  perVG1 cell={r_pv['cell_wide_median_dec']:.4f}  perVG1={r_pv['per_VG1_median']}", flush=True)
        print(f"  global cell={r_gl['cell_wide_median_dec']:.4f}  perVG1={r_gl['per_VG1_median']}  ({time.time()-tt:.1f}s)", flush=True)
        sweep_results.append({
            "dbd_BV": BV, "dbd_n": 4.0,
            "perVG1": {"cell_wide": r_pv["cell_wide_median_dec"], "per_VG1": r_pv["per_VG1_median"]},
            "global": {"cell_wide": r_gl["cell_wide_median_dec"], "per_VG1": r_gl["per_VG1_median"]},
        })
        # Snapshot
        (OUT / "sweep_summary.json").write_text(json.dumps({
            "baselines": {
                "perVG1_noavl": {"cell_wide": base_pv["cell_wide_median_dec"], "per_VG1": base_pv["per_VG1_median"]},
                "global_noavl": {"cell_wide": base_gl["cell_wide_median_dec"], "per_VG1": base_gl["per_VG1_median"]},
            },
            "sweep": sweep_results,
            "elapsed_s": time.time() - t0,
        }, indent=2))

    # Pick best by GLOBAL cell-wide (the hypothesis gate)
    valid_gl = [r for r in sweep_results if r["global"]["cell_wide"] is not None]
    best_global = min(valid_gl, key=lambda r: r["global"]["cell_wide"]) if valid_gl else None
    valid_pv = [r for r in sweep_results if r["perVG1"]["cell_wide"] is not None]
    best_perVG1 = min(valid_pv, key=lambda r: r["perVG1"]["cell_wide"]) if valid_pv else None

    # Physics gates
    physics_check = {
        "HYPOTHESIS_PASS_global_lt_1p0": best_global is not None and best_global["global"]["cell_wide"] < 1.0,
        "AMBITIOUS_global_lt_0p5": best_global is not None and best_global["global"]["cell_wide"] < 0.5,
        "VG1_0p60_drops_below_R46_0p86_perVG1": (
            best_perVG1 is not None
            and best_perVG1["perVG1"]["per_VG1"].get("0.60") is not None
            and best_perVG1["perVG1"]["per_VG1"]["0.60"] < 0.86
        ),
        "BV_physical_8_to_12": best_global is not None and 8.0 <= best_global["dbd_BV"] <= 12.0,
        "BV_unphysical_lt4_or_gt20": best_global is not None and (best_global["dbd_BV"] < 4.0 or best_global["dbd_BV"] > 20.0),
    }

    summary = {
        "script": "z368_dbd_avalanche",
        "patches_active": [
            "R-20", "R-29", "R-37", "R-41", "R-46 perVG1 params",
            "R-49 M1 drain-body avalanche multiplier (this run)",
        ],
        "BV_sweep_V": BV_sweep,
        "dbd_n_fixed": 4.0,
        "baselines": {
            "perVG1_noavl_cell_wide": base_pv["cell_wide_median_dec"],
            "perVG1_noavl_per_VG1":   base_pv["per_VG1_median"],
            "global_noavl_cell_wide": base_gl["cell_wide_median_dec"],
            "global_noavl_per_VG1":   base_gl["per_VG1_median"],
        },
        "best_global": best_global,
        "best_perVG1": best_perVG1,
        "physics_check": physics_check,
        "reference": {
            "R46_perVG1_cell_wide": 0.965,
            "R46_VG1_0p60": 0.863,
            "R48_global_mean_cell_wide": 1.192,
            "R43_global_floor": 1.131,
        },
        "sweep": sweep_results,
        "elapsed_s": time.time() - t0,
    }
    (OUT / "sweep_summary.json").write_text(json.dumps(summary, indent=2))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        BVs = [r["dbd_BV"] for r in sweep_results]
        for vg1 in (0.20, 0.40, 0.60):
            k = f"{vg1:.2f}"
            ys = [r["global"]["per_VG1"].get(k) if r["global"]["per_VG1"] else None for r in sweep_results]
            ax.plot(BVs, ys, "-o", label=f"global VG1={vg1}")
            ys2 = [r["perVG1"]["per_VG1"].get(k) if r["perVG1"]["per_VG1"] else None for r in sweep_results]
            ax.plot(BVs, ys2, "--s", label=f"perVG1 VG1={vg1}", alpha=0.6)
        ax.axhline(0.965, ls=":", c="k", label="R-46 0.965 (perVG1, no avl)")
        ax.axhline(1.192, ls=":", c="gray", label="R-48 1.192 (global, no avl)")
        ax.set_xlabel("dbd_BV [V]")
        ax.set_ylabel("log10-RMSE [dec]")
        ax.set_title("R-49 avalanche sweep: per-VG1 median vs BV")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "best_per_VG1.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[z368] plot skipped: {e}", flush=True)

    # Physics check markdown
    def fmt(x): return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)
    md = []
    md.append("# R-49 Drain-Body Avalanche Physics Check\n")
    md.append(f"## Baselines (no avalanche)")
    md.append(f"  perVG1 (R-46 reproduce): cell={fmt(base_pv['cell_wide_median_dec'])}  per_VG1={base_pv['per_VG1_median']}")
    md.append(f"  global (R-48 honest):    cell={fmt(base_gl['cell_wide_median_dec'])}  per_VG1={base_gl['per_VG1_median']}\n")
    md.append(f"## Sweep (dbd_n=4)")
    for r in sweep_results:
        md.append(f"  BV={r['dbd_BV']:>5.1f}V  perVG1 cell={fmt(r['perVG1']['cell_wide'])}  global cell={fmt(r['global']['cell_wide'])}  perVG1_per={r['perVG1']['per_VG1']}  global_per={r['global']['per_VG1']}")
    md.append("")
    md.append(f"## Best global: BV={best_global['dbd_BV'] if best_global else 'NA'}V  cell={fmt(best_global['global']['cell_wide']) if best_global else 'NA'}")
    md.append(f"## Best perVG1: BV={best_perVG1['dbd_BV'] if best_perVG1 else 'NA'}V  cell={fmt(best_perVG1['perVG1']['cell_wide']) if best_perVG1 else 'NA'}\n")
    md.append("## Gates")
    for k, v in physics_check.items():
        md.append(f"  {k}: {v}")
    md.append("")
    md.append("## References")
    md.append("  R-46 perVG1 cell-wide: 0.965 dec")
    md.append("  R-46 VG1=0.60:         0.863 dec")
    md.append("  R-48 global mean:      1.192 dec")
    md.append("  R-43 global floor:     1.131 dec")
    (OUT / "physics_check.md").write_text("\n".join(md))

    print(f"\n[z368] DONE  elapsed={time.time()-t0:.1f}s")
    if best_global:
        print(f"  Best (global): BV={best_global['dbd_BV']}V cell={best_global['global']['cell_wide']:.4f} dec")
        print(f"  per_VG1 at best: {best_global['global']['per_VG1']}")
    if best_perVG1:
        print(f"  Best (perVG1): BV={best_perVG1['dbd_BV']}V cell={best_perVG1['perVG1']['cell_wide']:.4f} dec")
        print(f"  per_VG1 at best: {best_perVG1['perVG1']['per_VG1']}")
    print(f"  Physics gates: {physics_check}")


if __name__ == "__main__":
    main()
