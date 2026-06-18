"""z344C — Cell-wide refit using PATCHED M1+M2 cards (R-28).

Both M1 and M2 cards have lalpha0=-9.84e-12 cancelling alpha0 at L=0.13µm
in ngspice (binunit=2). Pyport overrides binunit=1 → cancellation is ~0
anyway, but we still get a 10× alpha0 boost from the alpha0 7.84e-5→7.84e-4
patch.

Sweep:
  - body_pdiode_Rs ∈ {1e8, 1e9, 1e10}  (3 values)
  - bjt.Va = 1.5  (raised from 0.357)
  - bjt.Bf = 15000 (raised from 2605)

33 curves × 3 Rs values = 99 evaluations.
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

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/z344_dual_card_refit"
OUT.mkdir(parents=True, exist_ok=True)

DATA = ROOT / "data/sebas_2026_04_22"

# z338 eval 21 best (excluding alpha0 — card provides it via patch).
# Raised Bf and Va per R-28 spec.
BJT_BF = 15000.0
BJT_VA = 1.5
BJT_IS = 3.2906845928467974e-10
LAT_BV = 4.018266147002578

RS_SWEEP = [1.0e8, 1.0e9, 1.0e10]


def build_pyport(body_pdiode_Rs: float):
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.model_card import BSIM4Model
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    f = v1.f

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    cfg.lat_BV = LAT_BV
    cfg.body_pdiode_Rs = body_pdiode_Rs

    text_M1 = (DATA / "M1_130DNWFB_LALPHA0_FIX.txt").read_text()
    text_M2 = (DATA / "M2_130bulkNSRAM_LALPHA0_FIX.txt").read_text()
    M1 = BSIM4Model.from_spice(text_M1, model_type="nmos")
    M2 = BSIM4Model.from_spice(text_M2, model_type="nmos")
    f.patch_model_values(M1, type_n=True)
    f.patch_model_values(M2, type_n=True)
    M1._values["voff"] = M1._values.get("voff", -0.1368)
    M2._values["voff"] = M2._values.get("voff", -0.1368)

    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = BJT_BF
    bjt.Va = BJT_VA
    bjt.Is = BJT_IS
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


def run_one_rs(curves, body_pdiode_Rs):
    cfg, M1, M2, bjt = build_pyport(body_pdiode_Rs)
    from nsram.bsim4_port.nsram_cell_2T import forward_2t

    results = []
    per_vg1 = {}
    for c in curves:
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
                rmse = float("nan")
            else:
                logr = np.log10(Id_pred[mask]) - np.log10(c["Id"][mask])
                rmse = float(np.sqrt(np.mean(logr ** 2)))
        except Exception as e:
            rmse = float("nan")
            print(f"    err {c['f']}: {type(e).__name__}: {e}", flush=True)
        results.append({"VG1": c["VG1"], "VG2": c["VG2"], "log_rmse_dec": rmse, "f": c["f"]})
        per_vg1.setdefault(c["VG1"], []).append(rmse)

    valid = [r["log_rmse_dec"] for r in results if not np.isnan(r["log_rmse_dec"])]
    summary = {
        "body_pdiode_Rs": body_pdiode_Rs,
        "n_curves_total": len(results),
        "n_curves_valid": len(valid),
        "median_dec": float(np.median(valid)) if valid else None,
        "p25_dec": float(np.percentile(valid, 25)) if valid else None,
        "p75_dec": float(np.percentile(valid, 75)) if valid else None,
        "max_dec": float(np.max(valid)) if valid else None,
        "min_dec": float(np.min(valid)) if valid else None,
        "per_VG1_median": {f"{k:.2f}": (float(np.median([x for x in v if not np.isnan(x)])) if any(not np.isnan(x) for x in v) else None) for k,v in per_vg1.items()},
        "per_curve": results,
    }
    return summary


def main():
    t0 = time.time()
    curves = load_curves()
    print(f"[z344C] loaded {len(curves)} curves", flush=True)
    if not curves:
        print("[z344C] no curves", flush=True); return

    all_results = []
    for Rs in RS_SWEEP:
        print(f"\n[z344C] === body_pdiode_Rs = {Rs:.0e} ===", flush=True)
        s = run_one_rs(curves, Rs)
        print(f"  median_dec={s['median_dec']:.3f}  per_VG1={s['per_VG1_median']}", flush=True)
        all_results.append(s)

    best = min((r for r in all_results if r["median_dec"] is not None),
               key=lambda r: r["median_dec"])
    out = {
        "script": "z344_dual_card_refit",
        "patch": "M1+M2 cards both lalpha0=0+alpha0×10; cfg.bjt_emitter_to_gnd=True; Bf=15000 Va=1.5; Rs sweep",
        "bjt": {"Bf": BJT_BF, "Va": BJT_VA, "Is": BJT_IS},
        "lat_BV": LAT_BV,
        "Rs_sweep": RS_SWEEP,
        "per_Rs": all_results,
        "best": best,
        "baselines": {"z337": 4.16, "z343": 3.99},
        "elapsed_s": time.time() - t0,
    }
    out["gate_PASS_lt_0.95"] = (best["median_dec"] is not None and best["median_dec"] < 0.95)
    out["gate_AMBITIOUS_lt_0.50"] = (best["median_dec"] is not None and best["median_dec"] < 0.50)

    (OUT / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\n[z344C] BEST: Rs={best['body_pdiode_Rs']:.0e}  median_dec={best['median_dec']:.3f}", flush=True)
    print(f"[z344C] per_VG1 @ best: {best['per_VG1_median']}", flush=True)
    print(f"[z344C] gate_PASS_lt_0.95: {out['gate_PASS_lt_0.95']}", flush=True)
    print(f"[z344C] elapsed {out['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
