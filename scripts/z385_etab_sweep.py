"""z385 — Test B: etab DIAGNOSTIC + sweep.

Hypothesis 4: etab is the missing nonlinearity for the VG1=0.6 snapback fold.

Step 1 (DIAGNOSTIC): Verify Vth actually changes with Vbs in pyport.
  - For Vbs ∈ [0, 0.3, 0.6, 1.0] V evaluate Vth via the BSIM4 DC path.
  - If Vth doesn't change → etab is not wired.

Step 2 (SWEEP): etab override ∈ {0.5, 1.8 (Mario canonical), 5.0, 10.0, 20.0}
  - Targets: (VG1,VG2) ∈ {(0.2,0.10), (0.4,0.20), (0.6,0.20)}

Gates:
  DISCOVERY  : any etab gives VG1=0.6 model_jump > 0.5 dec
"""
from __future__ import annotations
import sys, json, time, math
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _z384_shared import (ROOT, TARGETS, build_base, load_sebas_params, run_one,
                          find_or_impute_row, make_overrides, patch_sd_scaled)

OUT = ROOT / "results/z385_etab"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.log"


def _log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")


def vth_diagnostic(rows):
    """Probe BSIM4 compute_dc directly to extract Vth at varying Vbs.

    compute_dc returns DCResult(Vth=...). For each Vbs, evaluate at
    Vgs=0.5 V, Vds=0.05 V and read Vth scalar. Vth in BSIM4 does NOT
    depend on Vgs (only on Vbseff and Vds via DIBL_Sft), so this is exact.
    """
    from nsram.bsim4_port.dc import compute_dc
    cfg, M1, M2, bjt = build_base()
    sd_M1 = cfg.size_dep_M1(M1)
    row = find_or_impute_row(rows, 0.6, 0.20)
    P_M1, _ = make_overrides(row)

    VBS_TEST = [0.0, 0.3, 0.6, 1.0]
    out_vth = {}
    for Vbs in VBS_TEST:
        with patch_sd_scaled(sd_M1, P_M1):
            try:
                dc = compute_dc(M1, sd_M1,
                                Vgs=torch.tensor(0.5, dtype=torch.float64),
                                Vds=torch.tensor(0.05, dtype=torch.float64),
                                Vbs=torch.tensor(float(Vbs), dtype=torch.float64))
                vth_val = float(dc.Vth.detach().cpu().item())
                out_vth[Vbs] = vth_val
            except Exception as e:
                out_vth[Vbs] = float("nan")
                _log(f"  Vbs={Vbs}: EXCEPTION {e}")
    return out_vth


def main():
    if LOG.exists(): LOG.unlink()
    rows = load_sebas_params()
    t_start = time.time()

    # --- STEP 1: DIAGNOSTIC ---
    _log("=== DIAGNOSTIC: Vth(Vbs) ===")
    vth_map = vth_diagnostic(rows)
    for vbs, vth in sorted(vth_map.items()):
        _log(f"  Vbs={vbs:.2f} V  →  Vth ≈ {vth:.4f} V")
    vth_vals = [v for v in vth_map.values() if v == v]
    delta = (max(vth_vals) - min(vth_vals)) if len(vth_vals) >= 2 else 0.0
    etab_wired = delta > 1e-3
    _log(f"  Δ Vth over Vbs range = {delta:.4f} V → etab_wired = {etab_wired}")

    # --- STEP 2: SWEEP ---
    ETAB_VALUES = [0.5, 1.8, 5.0, 10.0, 20.0]
    results = {}
    for ev in ETAB_VALUES:
        label = f"etab_{ev}"
        _log(f"=== {label} ===")
        cfg, M1, M2, bjt = build_base()
        per_t = []
        for (vg1, vg2) in TARGETS:
            r = run_one(cfg, M1, M2, bjt, rows, vg1, vg2, etab_override=ev,
                        log=_log)
            _log(f"  VG1={vg1} VG2={vg2}: rmse={r.get('rmse_dec',float('nan')):.3f} dec  "
                 f"jump(meas/model)={r.get('meas_jump_dec',0) or 0:.2f}/"
                 f"{r.get('model_jump_dec',float('nan')):.2f}  "
                 f"nan={r.get('has_nan')}  {r.get('elapsed_s',0):.1f}s")
            per_t.append(r)
        results[label] = per_t

    best_mj = -1e9; best_lbl = None
    for lbl, lst in results.items():
        for r in lst:
            if r["VG1"] == 0.6:
                mj = r.get("model_jump_dec", float("nan"))
                if mj is not None and mj == mj and mj > best_mj:
                    best_mj = mj; best_lbl = lbl
    discovery = best_mj > 0.5

    elapsed = time.time() - t_start
    summary = {
        "diagnostic_vth_vs_vbs": vth_map,
        "etab_wired": etab_wired,
        "delta_vth": delta,
        "conditions": list(results.keys()),
        "results": results,
        "gates": {
            "infra_ok": elapsed < 45*60,
            "elapsed_s": elapsed,
            "best_model_jump_at_vg06": best_mj,
            "best_condition": best_lbl,
            "discovery_fold_gt_0p5": discovery,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"DONE in {elapsed:.1f}s")
    _log(f"  etab wired? {etab_wired}  (ΔVth={delta:.4f} V)")
    _log(f"  best VG1=0.6 model_jump = {best_mj:.3f} dec  ({best_lbl})")
    _log(f"  DISCOVERY (fold>0.5): {discovery}")


if __name__ == "__main__":
    main()
