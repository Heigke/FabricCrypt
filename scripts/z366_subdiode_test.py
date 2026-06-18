"""z366 — R-47: SECOND body-leakage path (sub-diode) topology test.

Hypothesis: anti-correlated VG1=0.20 (over-pumped) vs VG1=0.60 (good) under
the R-43 global-knob floor (1.131 dec) can be resolved by adding a second
body-leakage path that conducts at LOW Vb (drains the over-pump at low
VG1) but saturates at HIGH Vb (does not interfere with main well-diode
discharge at high VG1). Branch-dependent Iii_scaling is implicit via
Vb-dependent current ratio between primary well-diode and sub-diode.

Subdiode: anode=Vb, cathode∈{gnd, vnwell}, low Is (1e-12..1e-6), n=1.0,
small Rs limiter (1e6 Ω).

Stacked patches (from R-20 → R-43 best):
  - R-20 BJT Vbc fix, R-29 Vth/tox, R-37 binunit
  - R-41 body_pdiode_to=vnwell + use_well_diode=True + branch-flat impute
  - R-39 BBO BJT params (Bf=991, Va=0.903, Is=5.95e-12)
  - R-43 best: iii_body_gain=1.0, vnwell_Rs=1e8, vnwell=2.0

Sweep grid (R-47):
  body_subdiode_Is ∈ {1e-12, 1e-10, 1e-8, 1e-6}   (4 values)
  body_subdiode_to ∈ {gnd, vnwell}                 (2 destinations)
  ⇒ 8 configs, 33-curve refit each.

Pre-registered gates:
  HYPOTHESIS PASS  : cell-wide median < 1.0 dec
  AMBITIOUS        : cell-wide median < 0.5 dec
  INFRA            : VG1=0.20 must improve below 2.0 dec
                     (currently ~2.62 at R-43 best)

Writes results/z366_subdiode/{summary.json, best.json, best_per_VG1.png}.
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
OUT = ROOT / "results/z366_subdiode"; OUT.mkdir(parents=True, exist_ok=True)
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


def build_pyport(sub_Is: float, sub_to: str):
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
    # R-43 best (fixed)
    cfg.iii_body_gain = 1.0
    cfg.vnwell_Rs = 1.0e8
    # R-47 new: sub-diode
    cfg.use_body_subdiode = True
    cfg.body_subdiode_to = sub_to
    cfg.body_subdiode_Is = float(sub_Is)
    cfg.body_subdiode_n = 1.0
    cfg.body_subdiode_Rs = 1.0e6
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


def run_one(sub_Is, sub_to, sebas_rows, curves):
    t0 = time.time()
    cfg, M1, M2, bjt = build_pyport(sub_Is, sub_to)
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    tag = f"Is={sub_Is:.0e},to={sub_to}"
    print(f"\n[z366] === {tag} ===", flush=True)
    sd_M1 = cfg.size_dep_M1(M1)
    sd_M2 = cfg.size_dep_M2(M2)

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
            "Iii_M1": _g(comp, "Iii_M1"),
            "I_well_body": _g(comp, "I_well_body"),
            "I_body_pdiode": _g(comp, "I_body_pdiode"),
            "I_subdiode": _g(comp, "I_subdiode"),
            "Ic_Q1": _g(out, "Ic_Q1"),
        }
        print(f"  flagship Vb={flagship['Vb']}  Id={flagship['Id']}  I_sub={flagship['I_subdiode']}  I_well={flagship['I_well_body']}", flush=True)
    except Exception as e:
        print(f"  flagship FAILED: {type(e).__name__}: {e}", flush=True)
        flagship = {"error": str(e)}

    results = []; per_vg1 = {}
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
            rmse = float(np.sqrt(np.mean((np.log10(Id_pred[mask]) - np.log10(c["Id"][mask])) ** 2))) if mask.sum() >= 3 else float("nan")
        except Exception:
            rmse = float("nan")
        results.append({"VG1": c["VG1"], "VG2": c["VG2"], "log_rmse_dec": rmse})
        per_vg1.setdefault(c["VG1"], []).append(rmse)

    valid = [r["log_rmse_dec"] for r in results if not math.isnan(r["log_rmse_dec"])]
    per_vg1_median = {f"{k:.2f}": float(np.median([x for x in v if not math.isnan(x)]))
                     for k, v in per_vg1.items()}
    median_cell = float(np.median(valid)) if valid else None
    elapsed = time.time() - t0
    print(f"  {tag}: cell_med={median_cell}  per_VG1={per_vg1_median}  ({elapsed:.0f}s)", flush=True)
    return {
        "body_subdiode_Is": float(sub_Is),
        "body_subdiode_to": sub_to,
        "cell_wide_median_dec": median_cell,
        "per_VG1_median": per_vg1_median,
        "n_valid": len(valid), "n_total": len(results),
        "flagship": flagship,
        "elapsed_s": elapsed,
        "per_curve": results,
    }


def main():
    t0 = time.time()
    Is_grid = [1e-12, 1e-10, 1e-8, 1e-6]
    to_grid = ["gnd", "vnwell"]
    configs = [(Is_, to_) for to_ in to_grid for Is_ in Is_grid]
    print(f"[z366] {len(configs)} configs (4 Is × 2 destinations)", flush=True)
    sebas_rows = load_sebas_params()
    curves = load_curves()
    print(f"[z366] loaded {len(curves)} curves, {len(sebas_rows)} Sebas rows", flush=True)

    sweep_results = []
    for (Is_, to_) in configs:
        r = run_one(Is_, to_, sebas_rows, curves)
        sweep_results.append(r)
        partial = {"sweep": sweep_results, "in_progress": True}
        (OUT / "summary.json").write_text(json.dumps(partial, indent=2))

    valid_results = [r for r in sweep_results if r["cell_wide_median_dec"] is not None]
    best = min(valid_results, key=lambda r: r["cell_wide_median_dec"])
    best_med = best["cell_wide_median_dec"]

    # VG1=0.20 infra gate
    vg1_20_best = best["per_VG1_median"].get("0.20", None)

    summary = {
        "script": "z366_subdiode_test",
        "hypothesis": "R-47: second body-leakage path resolves anti-correlation between VG1 branches",
        "patches_active": [
            "R-20 BJT Vbc", "R-29 Vth/tox", "R-37 binunit",
            "R-39 BBO BJT", "R-41 body_pdiode_to=vnwell + use_well_diode",
            "R-41 NaN-row branch-flat impute",
            "R-43 fixed (iii=1.0, vnwell_Rs=1e8, vnwell=2.0)",
            "R-47 use_body_subdiode (new topology)",
        ],
        "fixed": {"iii_body_gain": 1.0, "vnwell_Rs": 1.0e8, "vnwell": 2.0,
                  "body_subdiode_n": 1.0, "body_subdiode_Rs": 1.0e6},
        "Is_grid": Is_grid, "to_grid": to_grid,
        "sweep": sweep_results,
        "best": {
            "body_subdiode_Is": best["body_subdiode_Is"],
            "body_subdiode_to": best["body_subdiode_to"],
            "cell_wide_median_dec": best_med,
            "per_VG1_median": best["per_VG1_median"],
            "flagship": best["flagship"],
        },
        "baselines": {
            "z363_R43": 1.1306581736187744,
            "z364_R45_vnwell_floor": 1.131,
            "VG1_0p20_R43_dec": 2.62,
        },
        "elapsed_s": time.time() - t0,
    }
    summary["gate_HYPOTHESIS_PASS_lt_1p0"] = (best_med is not None) and (best_med < 1.0)
    summary["gate_AMBITIOUS_lt_0p5"] = (best_med is not None) and (best_med < 0.5)
    summary["gate_INFRA_VG1_0p20_lt_2p0"] = (vg1_20_best is not None) and (vg1_20_best < 2.0)
    summary["VG1_0p20_at_best"] = vg1_20_best
    summary["in_progress"] = False

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "best.json").write_text(json.dumps(summary["best"], indent=2))

    # Plot: per-VG1 dec at best config + cell-med vs Is for both destinations
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        # Left: cell-med vs Is, color by destination
        ax = axes[0]
        for to_ in to_grid:
            xs = [r["body_subdiode_Is"] for r in sweep_results if r["body_subdiode_to"] == to_]
            ys = [r["cell_wide_median_dec"] for r in sweep_results if r["body_subdiode_to"] == to_]
            ax.semilogx(xs, ys, "o-", lw=2, label=f"to={to_}")
        ax.axhline(1.131, color="grey", ls=":", label="R-43 floor 1.131")
        ax.axhline(1.0, color="green", ls="--", label="HYPOTHESIS PASS 1.0")
        ax.axhline(0.5, color="darkgreen", ls="--", label="AMB 0.5")
        ax.set_xlabel("body_subdiode_Is [A]")
        ax.set_ylabel("cell-wide log10 RMSE [dec]")
        ax.set_title(f"z366 sub-diode sweep — best Is={best['body_subdiode_Is']:.0e} to={best['body_subdiode_to']}  med={best_med:.3f}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        # Right: per-VG1 at best
        ax = axes[1]
        items = sorted(best["per_VG1_median"].items(), key=lambda kv: float(kv[0]))
        xs = [float(k) for k, _ in items]
        ys = [v for _, v in items]
        ax.plot(xs, ys, "o-", lw=2, color="purple", label="best config per-VG1")
        ax.axhline(2.0, color="red", ls=":", label="INFRA gate 2.0")
        ax.axhline(1.0, color="green", ls="--", label="HYP PASS 1.0")
        ax.set_xlabel("VG1 [V]")
        ax.set_ylabel("median log10 RMSE [dec]")
        ax.set_title(f"per-VG1 at best (Is={best['body_subdiode_Is']:.0e}, to={best['body_subdiode_to']})")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "best_per_VG1.png", dpi=120)
        plt.close(fig)
        print(f"[z366] saved plot to {OUT}/best_per_VG1.png", flush=True)
    except Exception as e:
        print(f"[z366] plot failed: {type(e).__name__}: {e}", flush=True)

    print(f"\n[z366] DONE  best Is={best['body_subdiode_Is']:.0e}  to={best['body_subdiode_to']}  med={best_med}", flush=True)
    print(f"[z366] best_per_VG1={best['per_VG1_median']}", flush=True)
    print(f"[z366] VG1=0.20 at best: {vg1_20_best}", flush=True)
    print(f"[z366] gates: HYP<1.0={summary['gate_HYPOTHESIS_PASS_lt_1p0']}  AMB<0.5={summary['gate_AMBITIOUS_lt_0p5']}  INFRA<2.0={summary['gate_INFRA_VG1_0p20_lt_2p0']}", flush=True)
    print(f"[z366] elapsed {summary['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
