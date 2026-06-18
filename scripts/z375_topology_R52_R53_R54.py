"""z375 — R-52/R-53/R-54 topology variants for NS-RAM snapback fold.

R-52: M(V_db) DIRECTLY on Ids_M1 (multiplicative, not body injection).
      Sweep bv_ids ∈ {6, 8, 10, 12, 15} V, n_ids=4 fixed.
R-53: Two-stage cascaded: Ids *= sqrt(M) AND body += (sqrt(M)-1)|Ids|.
      Sweep bv_ids ∈ {6, 8, 10, 12, 15} V.
R-54: NPN base resistor RB_npn (bjt_emitter_to_gnd path).
      Sweep RB_npn ∈ {0, 100, 1k, 10k, 100k} Ω.

Each variant uses per-VG1 R-46 best params (eval 94 of z365 BBO).
Output: results/z375_topology_variants/{R52,R53,R54}/{summary.json, best_per_VG1.png, run.log}

Pre-registered gates per variant:
  INFRA      : runs without nan
  DISCOVERY  : any sweep point → cell-wide < 1.0 dec
  AMBITIOUS  : any sweep point → cell-wide < 0.5 dec AND VG1=0.6 fold > 1 dec
               (vs current 0.02 dec — measured from raw Id_pred decade-spread)
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import sys, json, re, time, math, csv, importlib.util, traceback
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
            sd.scaled[k] = float(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None: sd.scaled.pop(k, None)
            else: sd.scaled[k] = v


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
OUT_ROOT = ROOT / "results/z375_topology_variants"; OUT_ROOT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"


BRANCH_FLAT = {
    0.4: {"ETAB": 1.9,  "K1": 0.53825, "ALPHA0": 7.842e-05, "BETA0": 19.0, "NFACTOR": 6.0, "trise": 10.59},
    0.6: {"ETAB": 2.5,  "K1": 0.41825, "ALPHA0": 7.842e-05, "BETA0": 20.0, "NFACTOR": 6.0, "trise": 9.04},
}
M2_STATIC_OVERRIDES = {
    "k1": 0.63825, "k2": -0.070435, "etab": -0.086777, "beta0": 18.0,
}
# eval 94 of z365 BBO (R-46 best per-VG1)
R46_PER_VG1 = {
    0.20: {"Bf": 1889.8806320503354, "iii_body_gain": 1.844675445044413,   "vnwell_Rs": 10**9.172216925770044},
    0.40: {"Bf": 1092.272187024355,  "iii_body_gain": 1.5151811846066265,  "vnwell_Rs": 10**9.898274548351765},
    0.60: {"Bf": 417.62741417624056, "iii_body_gain": 0.9035713450051843,  "vnwell_Rs": 10**6.784622445702553},
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


def fold_dec_at_VG1(curves_pred_by_VG1, VG1_target=0.60):
    """Compute decade-fold of model |Id| trace at VG1=target.
    fold = log10(max(Id)) - log10(min(Id_post_peak)). Robust: across all VG2.
    """
    folds = []
    for (vg1, vg2), Id_pred in curves_pred_by_VG1.items():
        if abs(vg1 - VG1_target) > 1e-3:
            continue
        if Id_pred is None or len(Id_pred) < 5: continue
        Id = np.asarray(Id_pred)
        Id = Id[np.isfinite(Id) & (Id > 1e-18)]
        if len(Id) < 5: continue
        imax = int(np.argmax(Id))
        if imax >= len(Id) - 1: continue
        post = Id[imax:]
        if len(post) < 2: continue
        folds.append(math.log10(Id[imax]) - math.log10(min(post)))
    return float(max(folds)) if folds else 0.0


def eval_all_curves(cfg, M1, M2, bjt, sebas_rows, curves, forward_2t):
    sd_M1 = cfg.size_dep_M1(M1)
    sd_M2 = cfg.size_dep_M2(M2)
    results = []
    per_vg1 = {0.20: [], 0.40: [], 0.60: []}
    preds = {}
    for c in curves:
        vg1_key = round(c["VG1"], 2)
        if vg1_key not in R46_PER_VG1:
            continue
        p = R46_PER_VG1[vg1_key]
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
            preds[(c["VG1"], c["VG2"])] = Id_pred
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
    fold_06 = fold_dec_at_VG1(preds, 0.60)
    return {
        "cell_wide_median_dec": float(np.median(valid)) if valid else None,
        "per_VG1_median": {f"{k:.2f}": float(np.median(v)) if v else None for k, v in per_vg1.items()},
        "n_valid": len(valid), "n_total": len(results),
        "fold_dec_VG1_0p60": fold_06,
        "any_nan": any(math.isnan(r["log_rmse_dec"]) for r in results),
    }


def run_variant(name, cfg_setters, sweep_label, sweep_values, log_fp):
    """Generic sweep runner.
    cfg_setters: function(cfg, value) → mutates cfg for sweep value
    """
    out_dir = OUT_ROOT / name; out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    def L(msg):
        print(msg, flush=True); log_fp.write(msg + "\n"); log_fp.flush()
    L(f"\n========== {name} ==========")
    cfg, M1, M2, bjt = build_pyport_base()
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    sebas_rows = load_sebas_params()
    curves = load_curves()
    L(f"[{name}] loaded {len(curves)} curves")

    # Baseline (variant OFF)
    cfg.use_ids_multiplier = False
    cfg.use_two_stage_avl = False
    cfg.use_base_resistor = False
    cfg.use_dbd_avalanche = False
    L(f"[{name}] baseline (variant OFF, perVG1)...")
    tt = time.time()
    base = eval_all_curves(cfg, M1, M2, bjt, sebas_rows, curves, forward_2t)
    L(f"  cell={base['cell_wide_median_dec']:.4f}  perVG1={base['per_VG1_median']}  fold06={base['fold_dec_VG1_0p60']:.4f}  ({time.time()-tt:.1f}s)")

    sweep = []
    for v in sweep_values:
        # reset
        cfg.use_ids_multiplier = False
        cfg.use_two_stage_avl = False
        cfg.use_base_resistor = False
        cfg.use_dbd_avalanche = False
        cfg_setters(cfg, v)
        L(f"[{name}] {sweep_label}={v} ...")
        tt = time.time()
        try:
            r = eval_all_curves(cfg, M1, M2, bjt, sebas_rows, curves, forward_2t)
        except Exception as e:
            L(f"  ERROR: {e}\n{traceback.format_exc()}")
            r = {"cell_wide_median_dec": None, "per_VG1_median": {}, "fold_dec_VG1_0p60": 0.0, "any_nan": True}
        L(f"  cell={r['cell_wide_median_dec']}  perVG1={r['per_VG1_median']}  fold06={r['fold_dec_VG1_0p60']:.4f}  ({time.time()-tt:.1f}s)")
        sweep.append({sweep_label: v, **r})
        # snapshot
        (out_dir / "summary.json").write_text(json.dumps({
            "variant": name, "baseline": base, "sweep_label": sweep_label,
            "sweep": sweep, "elapsed_s": time.time() - t0,
        }, indent=2))

    # Gates
    valid = [r for r in sweep if r["cell_wide_median_dec"] is not None]
    best = min(valid, key=lambda r: r["cell_wide_median_dec"]) if valid else None
    fold_max = max((r.get("fold_dec_VG1_0p60") or 0.0) for r in sweep) if sweep else 0.0
    best_fold = max(sweep, key=lambda r: r.get("fold_dec_VG1_0p60") or 0.0) if sweep else None
    gates = {
        "INFRA_runs_without_nan": all(not r.get("any_nan", True) for r in valid) and len(valid) == len(sweep_values),
        "DISCOVERY_cell_lt_1p0": best is not None and best["cell_wide_median_dec"] < 1.0,
        "AMBITIOUS_cell_lt_0p5_AND_fold_gt_1": (best is not None and best["cell_wide_median_dec"] < 0.5
                                                and fold_max > 1.0),
    }
    summary = {
        "variant": name,
        "baseline": base,
        "sweep_label": sweep_label,
        "sweep": sweep,
        "best_by_cell": best,
        "best_by_fold": best_fold,
        "fold_max_VG1_0p60": fold_max,
        "gates": gates,
        "elapsed_s": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Plot
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        xs = [r[sweep_label] for r in sweep]
        for vg1 in (0.20, 0.40, 0.60):
            k = f"{vg1:.2f}"
            ys = [r["per_VG1_median"].get(k) for r in sweep]
            ax.plot(xs, ys, "-o", label=f"VG1={vg1}")
        if base.get("cell_wide_median_dec") is not None:
            ax.axhline(base["cell_wide_median_dec"], ls=":", c="k", label=f"baseline cell={base['cell_wide_median_dec']:.3f}")
        ax.axhline(0.965, ls=":", c="gray", alpha=0.5, label="R-46 0.965")
        ax.set_xlabel(sweep_label); ax.set_ylabel("log10-RMSE [dec]")
        ax.set_title(f"{name}: per-VG1 median")
        if sweep_label == "RB_npn":
            ax.set_xscale("symlog", linthresh=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "best_per_VG1.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        L(f"[{name}] plot skipped: {e}")

    L(f"[{name}] DONE elapsed={time.time()-t0:.1f}s")
    L(f"  best by cell: {best}")
    L(f"  fold_max VG1=0.60: {fold_max:.4f}")
    L(f"  gates: {gates}")
    return summary


def main():
    log_path = OUT_ROOT / "run.log"
    log_fp = open(log_path, "a")
    try:
        log_fp.write(f"\n\n##### {time.strftime('%Y-%m-%d %H:%M:%S')} z375 start #####\n")
        out = {}

        # R-52: Ids multiplier
        def setter52(cfg, bv):
            cfg.use_ids_multiplier = True
            cfg.bv_ids = float(bv); cfg.n_ids = 4.0
        out["R52"] = run_variant("R52_ids_multiplier", setter52, "bv_ids",
                                 [6.0, 8.0, 10.0, 12.0, 15.0], log_fp)

        # R-53: two-stage
        def setter53(cfg, bv):
            cfg.use_two_stage_avl = True
            cfg.bv_ids = float(bv); cfg.n_ids = 4.0
        out["R53"] = run_variant("R53_two_stage", setter53, "bv_ids",
                                 [6.0, 8.0, 10.0, 12.0, 15.0], log_fp)

        # R-54: NPN base resistor
        def setter54(cfg, rb):
            cfg.use_base_resistor = True
            cfg.RB_npn = float(rb)
        out["R54"] = run_variant("R54_npn_base_resistor", setter54, "RB_npn",
                                 [0.0, 100.0, 1000.0, 10000.0, 100000.0], log_fp)

        (OUT_ROOT / "all_summary.json").write_text(json.dumps(out, indent=2, default=str))
        print("\n[z375] ALL DONE")
        for k, s in out.items():
            print(f"  {k}: gates={s['gates']}  best_cell={s.get('best_by_cell',{})}")
    finally:
        log_fp.close()


if __name__ == "__main__":
    main()
