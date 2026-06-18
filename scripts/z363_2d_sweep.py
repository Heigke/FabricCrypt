"""z363 — R-43: 2D sweep over iii_body_gain × vnwell_Rs.

Diagnosis (R-42 null): body_pdiode_Rs was invariant because the pdiode is
reverse-biased at Vb=0.484 vs Vnwell=2.0 (only leakage). R-41's improvement
came from `use_well_diode=True`, which forward-biases vnwell→Vb. So the real
knobs to push Vb 0.484 → 0.27 are:
  - iii_body_gain : source pumping current (currently 0.66)
  - vnwell_Rs    : series-R of the well-diode (the actual discharge path)

Sweep: iii_body_gain ∈ {0.1, 0.2, 0.4, 0.66, 1.0}, vnwell_Rs ∈ {1e6, 1e8, 1e10}.
15 configs × ~155s = ~40 min.

Pre-registered gates (R-43):
  PASS         : cell-wide median < 1.0 dec (improve on z361's 1.42)
  AMBITIOUS    : cell-wide median < 0.50 dec
  Vb_target    : Vb @ flagship within 0.05 V of ngspice 0.27 (i.e. <0.32)

Writes results/z363_2d_sweep/{summary.json, heatmap.png, best.json}.
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
OUT = ROOT / "results/z363_2d_sweep"; OUT.mkdir(parents=True, exist_ok=True)
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


def build_pyport(iii_gain: float, vnwell_rs: float):
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    # R-41 stack
    cfg.body_pdiode_to = "vnwell"   # kept for safety (pdiode reverse-biased here anyway)
    cfg.use_well_diode = True
    cfg.vnwell = 2.0
    cfg.body_pdiode_Js = 5.3675e-7 / 22e-12
    cfg.body_pdiode_n = 1.0535
    cfg.body_pdiode_Rs = 1.0e6  # from z362 (invariant, pick any)
    # R-43 SWEEP VARIABLES
    cfg.iii_body_gain = float(iii_gain)
    cfg.vnwell_Rs = float(vnwell_rs)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 991.0; bjt.Va = 0.903; bjt.Is = 5.95e-12
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


def run_one(iii_gain, vnwell_rs, sebas_rows, curves):
    t0 = time.time()
    cfg, M1, M2, bjt = build_pyport(iii_gain, vnwell_rs)
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    print(f"\n[z363] === iii_gain={iii_gain:.3f}  vnwell_Rs={vnwell_rs:.1e} ===", flush=True)
    sd_M1 = cfg.size_dep_M1(M1)
    sd_M2 = cfg.size_dep_M2(M2)

    # Flagship probe
    flagship = None
    try:
        Vd_p = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
        row, _ = find_or_impute_row(sebas_rows, 0.6, 0.20)
        P_M1, P_M2 = make_overrides(row)
        with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
            out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_p,
                             VG1=torch.tensor(0.6, dtype=torch.float64),
                             VG2=torch.tensor(0.20, dtype=torch.float64),
                             warm_start=True)
        idx = 3
        comp = out.get("components", {})
        def _g(d, k):
            if k not in d: return None
            v = d[k]
            try:
                arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
                if arr.ndim == 0: return float(arr)
                return float(arr.flatten()[idx])
            except Exception:
                return None
        flagship = {
            "Vsint": _g(out, "Vsint"), "Vb": _g(out, "Vb"), "Id": _g(out, "Id"),
            "Iii_M1": _g(comp, "Iii_M1"), "Iii": _g(comp, "Iii"),
            "I_well_body": _g(comp, "I_well_body"),
            "Ic_Q1": _g(out, "Ic_Q1"),
        }
        print(f"  flagship Vb={flagship['Vb']}  Id={flagship['Id']}  I_well={flagship['I_well_body']}  Iii={flagship['Iii']}", flush=True)
    except Exception as e:
        print(f"  flagship FAILED: {type(e).__name__}: {e}", flush=True)
        flagship = {"error": str(e)}

    # Full 33-curve refit
    results = []; per_vg1 = {}; n_imputed = 0
    for c in curves:
        Vd = torch.tensor(c["Vd"], dtype=torch.float64)
        row, imp = find_or_impute_row(sebas_rows, c["VG1"], c["VG2"])
        if imp: n_imputed += 1
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
        results.append({"VG1": c["VG1"], "VG2": c["VG2"], "log_rmse_dec": rmse, "imputed": imp})
        per_vg1.setdefault(c["VG1"], []).append(rmse)

    valid = [r["log_rmse_dec"] for r in results if not math.isnan(r["log_rmse_dec"])]
    per_vg1_median = {f"{k:.2f}": float(np.median([x for x in v if not math.isnan(x)]))
                     for k, v in per_vg1.items()}
    median_cell = float(np.median(valid)) if valid else None
    elapsed = time.time() - t0
    print(f"  iii={iii_gain:.3f} vnwellRs={vnwell_rs:.1e}: cell_med={median_cell:.3f}  per_VG1={per_vg1_median}  ({elapsed:.0f}s)", flush=True)
    return {
        "iii_body_gain": float(iii_gain),
        "vnwell_Rs": float(vnwell_rs),
        "cell_wide_median_dec": median_cell,
        "per_VG1_median": per_vg1_median,
        "n_valid": len(valid),
        "n_total": len(results),
        "flagship": flagship,
        "elapsed_s": elapsed,
        "per_curve": results,
    }


def main():
    t0 = time.time()
    iii_grid = [0.1, 0.2, 0.4, 0.66, 1.0]
    rs_grid = [1.0e6, 1.0e8, 1.0e10]
    print(f"[z363] iii_grid={iii_grid}  vnwell_Rs_grid={rs_grid}  ({len(iii_grid)*len(rs_grid)} cells)", flush=True)
    sebas_rows = load_sebas_params()
    curves = load_curves()
    print(f"[z363] loaded {len(curves)} curves, {len(sebas_rows)} Sebas rows", flush=True)

    sweep_results = []
    for iii in iii_grid:
        for rs in rs_grid:
            r = run_one(iii, rs, sebas_rows, curves)
            sweep_results.append(r)
            partial = {"sweep": sweep_results, "in_progress": True}
            (OUT / "summary.json").write_text(json.dumps(partial, indent=2))

    valid_results = [r for r in sweep_results if r["cell_wide_median_dec"] is not None]
    best = min(valid_results, key=lambda r: r["cell_wide_median_dec"])
    best_med = best["cell_wide_median_dec"]
    best_vb = best["flagship"].get("Vb") if isinstance(best["flagship"], dict) else None

    summary = {
        "script": "z363_2d_sweep",
        "patches_active": [
            "R-20 BJT Vbc", "R-29 Vth/tox", "R-37 binunit", "R-39 BBO best (eval5)",
            "R-41 body_pdiode_to=vnwell", "R-41 use_well_diode=True",
            "R-41 NaN-row branch-flat impute (8/33)",
            "R-43 2D sweep iii_body_gain × vnwell_Rs",
        ],
        "iii_grid": iii_grid,
        "vnwell_Rs_grid": rs_grid,
        "sweep": sweep_results,
        "best": {
            "iii_body_gain": best["iii_body_gain"],
            "vnwell_Rs": best["vnwell_Rs"],
            "cell_wide_median_dec": best_med,
            "per_VG1_median": best["per_VG1_median"],
            "flagship": best["flagship"],
        },
        "baselines": {
            "z361_R41": 1.419,
            "z362_R42": 1.419,
            "ngspice_Vb_target": 0.27,
        },
        "elapsed_s": time.time() - t0,
    }
    summary["gate_PASS_lt_1p0"] = best_med < 1.0
    summary["gate_AMBITIOUS_lt_0p50"] = best_med < 0.50
    vb_ok = (best_vb is not None) and (abs(best_vb - 0.27) < 0.05)
    summary["gate_Vb_within_0p05_of_0p27"] = bool(vb_ok)
    summary["in_progress"] = False

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "best.json").write_text(json.dumps(summary["best"], indent=2))

    # Heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        grid = np.full((len(iii_grid), len(rs_grid)), np.nan)
        for r in sweep_results:
            if r["cell_wide_median_dec"] is None: continue
            i = iii_grid.index(r["iii_body_gain"])
            j = rs_grid.index(r["vnwell_Rs"])
            grid[i, j] = r["cell_wide_median_dec"]
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis_r")
        ax.set_xticks(range(len(rs_grid)))
        ax.set_xticklabels([f"{x:.0e}" for x in rs_grid])
        ax.set_yticks(range(len(iii_grid)))
        ax.set_yticklabels([f"{x:.2f}" for x in iii_grid])
        ax.set_xlabel("vnwell_Rs [Ω]")
        ax.set_ylabel("iii_body_gain")
        ax.set_title(f"z363 2D sweep — best=({best['iii_body_gain']:.2f},{best['vnwell_Rs']:.0e})  med={best_med:.3f}")
        for i in range(len(iii_grid)):
            for j in range(len(rs_grid)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                            color="white" if grid[i,j] > np.nanmedian(grid) else "black", fontsize=9)
        plt.colorbar(im, ax=ax, label="log10 RMSE [dec]")
        fig.tight_layout()
        fig.savefig(OUT / "heatmap.png", dpi=120)
        plt.close(fig)
        print(f"[z363] saved heatmap to {OUT}/heatmap.png", flush=True)
    except Exception as e:
        print(f"[z363] plot failed: {type(e).__name__}: {e}", flush=True)

    print(f"\n[z363] DONE  best=(iii={best['iii_body_gain']}, vnwell_Rs={best['vnwell_Rs']:.1e})  med={best_med:.3f}", flush=True)
    print(f"[z363] best_per_VG1={best['per_VG1_median']}", flush=True)
    print(f"[z363] best_flagship_Vb={best_vb}", flush=True)
    print(f"[z363] gates: PASS<1.0={summary['gate_PASS_lt_1p0']}  AMB<0.50={summary['gate_AMBITIOUS_lt_0p50']}  Vb_ok={summary['gate_Vb_within_0p05_of_0p27']}", flush=True)
    print(f"[z363] elapsed {summary['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
