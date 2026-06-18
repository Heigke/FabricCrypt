"""z358 — Cell-wide refit POST R-37 binunit fix.

R-37 found: binunit=1 override in z91f killed lbeta0/lalpha0 binning by
1e6, giving wrong beta0=19 instead of true beta0=10 → 2-3 dec Iii gap.
Override now removed. Re-run 33-curve refit to see cell-wide impact.

Active fixes stacked:
- R-20: bjt_emitter_to_gnd Vbc = Vb - Vsint
- R-29: lpe0=1.2439e-7, toxe=4e-9 (Vth/tox correct)
- R-37: binunit override removed (Iii gap closed)
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/z358_post_iimod_refit"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"


def build_pyport():
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 9000.0; bjt.Va = 0.55; bjt.Is = 1e-9
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
            curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:,0], "Id": np.abs(d[:,1]), "f": f.name})
    # also check top-level CSVs
    for f in sorted(DATA.glob("VG1*VG2*.csv")):
        m = re.search(r"VG1=([\d.\-]+)[_ ]*VG2=([\d.\-]+)", f.name)
        if not m: continue
        vg1 = float(m.group(1)); vg2 = float(m.group(2))
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:,0], "Id": np.abs(d[:,1]), "f": f.name})
    return curves


def main():
    t0 = time.time()
    cfg, M1, M2, bjt = build_pyport()
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    print(f"[z358] cfg.bjt_emitter_to_gnd={cfg.bjt_emitter_to_gnd}", flush=True)
    print(f"[z358] M1 binunit={M1._values.get('binunit')} (should be 2 post R-37)", flush=True)
    print(f"[z358] M1 alpha0={M1._values.get('alpha0'):.3e}  lalpha0={M1._values.get('lalpha0'):.3e}", flush=True)
    print(f"[z358] M1 beta0={M1._values.get('beta0'):.3e}  lbeta0={M1._values.get('lbeta0'):.3e}", flush=True)

    curves = load_curves()
    print(f"[z358] loaded {len(curves)} curves", flush=True)
    if not curves:
        print(f"[z358] no curves found in {DATA}", flush=True); return

    results = []
    per_vg1 = {}
    for c in curves:
        Vd = torch.tensor(c["Vd"], dtype=torch.float64)
        try:
            out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd,
                             VG1=torch.tensor(c["VG1"], dtype=torch.float64),
                             VG2=torch.tensor(c["VG2"], dtype=torch.float64), warm_start=True)
            Id_pred = np.abs(out["Id"].detach().cpu().numpy())
            mask = (c["Id"] > 1e-15) & (Id_pred > 1e-15) & np.isfinite(Id_pred)
            rmse = float(np.sqrt(np.mean((np.log10(Id_pred[mask]) - np.log10(c["Id"][mask])) ** 2))) if mask.sum() >= 3 else float("nan")
        except Exception as e:
            rmse = float("nan"); print(f"  err {c['f']}: {type(e).__name__}: {e}", flush=True)
        results.append({"VG1": c["VG1"], "VG2": c["VG2"], "log_rmse_dec": rmse})
        per_vg1.setdefault(c["VG1"], []).append(rmse)
        print(f"  VG1={c['VG1']:.2f} VG2={c['VG2']:+.2f}  rmse={rmse:.3f}", flush=True)

    valid = [r["log_rmse_dec"] for r in results if not np.isnan(r["log_rmse_dec"])]
    summary = {
        "script": "z358_post_iimod_refit",
        "patches_active": ["R-20 BJT Vbc=Vb-Vsint", "R-29 lpe0/toxe", "R-37 binunit override removed"],
        "n_curves_valid": len(valid), "n_curves_total": len(results),
        "cell_wide_median_dec": float(np.median(valid)) if valid else None,
        "per_VG1_median": {f"{k:.2f}": float(np.median([x for x in v if not np.isnan(x)])) for k,v in per_vg1.items()},
        "baselines": {"z304_v4_spurious": 0.99, "z337": 4.16, "z343": 3.99, "z346": 4.08, "z352_best": 3.93},
        "elapsed_s": time.time() - t0,
        "per_curve": results,
    }
    if summary["cell_wide_median_dec"] is not None:
        m = summary["cell_wide_median_dec"]
        summary["gate_PASS_lt_0.95"] = m < 0.95
        summary["gate_AMBITIOUS_lt_0.5"] = m < 0.5
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[z358] DONE  median={summary['cell_wide_median_dec']}  per_VG1={summary['per_VG1_median']}", flush=True)
    print(f"[z358] elapsed {summary['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
