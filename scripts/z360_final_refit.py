"""z360 — Final 33-curve refit with R-39 BBO's best params (eval 5).

Best from R-39 z359 BBO (before silent kill at eval 12):
  Bf = 991, Va = 0.90, Is = 6e-12, iii_body_gain = 0.66  → cost 2.668 dec on 6-bias subset
Plus all earlier fixes locked: R-20 BJT Vbc, R-29 Vth/tox, R-37 binunit.

Compare full 33-curve median to z358 (4.28).
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION","11.0.0")
import sys, json, re, time, importlib.util
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"nsram"))
OUT = ROOT/"results/z360_final_refit"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT/"data/sebas_2026_04_22"

def build_pyport():
    sp = importlib.util.spec_from_file_location("v1", ROOT/"scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    cfg.iii_body_gain = 0.66  # R-39 best
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 991.0; bjt.Va = 0.903; bjt.Is = 5.95e-12  # R-39 eval 5 best
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
            curves.append({"VG1":vg1,"VG2":vg2,"Vd":d[:,0],"Id":np.abs(d[:,1]),"f":f.name})
    for f in sorted(DATA.glob("VG1*VG2*.csv")):
        m = re.search(r"VG1=([\d.\-]+)[_ ]*VG2=([\d.\-]+)", f.name)
        if not m: continue
        vg1=float(m.group(1)); vg2=float(m.group(2))
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        curves.append({"VG1":vg1,"VG2":vg2,"Vd":d[:,0],"Id":np.abs(d[:,1]),"f":f.name})
    return curves

def main():
    t0 = time.time()
    cfg, M1, M2, bjt = build_pyport()
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    print(f"[z360] cfg.iii_body_gain={cfg.iii_body_gain}  bjt.Bf={bjt.Bf} Va={bjt.Va} Is={bjt.Is:.2e}", flush=True)
    curves = load_curves()
    print(f"[z360] loaded {len(curves)} curves", flush=True)
    results = []; per_vg1 = {}
    for c in curves:
        Vd = torch.tensor(c["Vd"], dtype=torch.float64)
        try:
            out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd,
                             VG1=torch.tensor(c["VG1"], dtype=torch.float64),
                             VG2=torch.tensor(c["VG2"], dtype=torch.float64), warm_start=True)
            Id_pred = np.abs(out["Id"].detach().cpu().numpy())
            mask = (c["Id"] > 1e-15) & (Id_pred > 1e-15) & np.isfinite(Id_pred)
            rmse = float(np.sqrt(np.mean((np.log10(Id_pred[mask]) - np.log10(c["Id"][mask]))**2))) if mask.sum()>=3 else float("nan")
        except Exception as e:
            rmse = float("nan"); print(f"  err {c['f']}: {e}", flush=True)
        results.append({"VG1":c["VG1"],"VG2":c["VG2"],"log_rmse_dec":rmse})
        per_vg1.setdefault(c["VG1"], []).append(rmse)
        print(f"  VG1={c['VG1']:.2f} VG2={c['VG2']:+.2f}  rmse={rmse:.3f}", flush=True)

    valid = [r["log_rmse_dec"] for r in results if not np.isnan(r["log_rmse_dec"])]
    summary = {
        "script": "z360_final_refit",
        "patches_active": ["R-20 BJT Vbc", "R-29 Vth/tox", "R-37 binunit", "R-39 BBO best (eval5)"],
        "params": {"Bf": bjt.Bf, "Va": bjt.Va, "Is": bjt.Is, "iii_body_gain": cfg.iii_body_gain},
        "n_curves_valid": len(valid), "n_curves_total": len(results),
        "cell_wide_median_dec": float(np.median(valid)) if valid else None,
        "per_VG1_median": {f"{k:.2f}": float(np.median([x for x in v if not np.isnan(x)])) for k,v in per_vg1.items()},
        "baselines": {"z304_spurious": 0.99, "z337": 4.16, "z346": 4.08, "z352_best": 3.93, "z358_postR37": 4.28},
        "elapsed_s": time.time()-t0, "per_curve": results,
    }
    if summary["cell_wide_median_dec"] is not None:
        m = summary["cell_wide_median_dec"]
        summary["gate_PASS_lt_0.95"] = m < 0.95
        summary["gate_AMBITIOUS_lt_0.5"] = m < 0.5
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[z360] DONE  median={summary['cell_wide_median_dec']}", flush=True)
    print(f"[z360] per_VG1={summary['per_VG1_median']}", flush=True)
    print(f"[z360] elapsed {summary['elapsed_s']:.1f}s", flush=True)

if __name__ == "__main__":
    main()
