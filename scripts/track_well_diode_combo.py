#!/usr/bin/env python3
"""Run the K1+ALPHA0+Tlpe1+well_diode_fix combo on the full 33-bias (fwd+bwd,
n=66) pyport DC fit. Stacks on top of the prior PARTIAL PASS (0.461 dec) by
adding `cfg.well_diode_mode='ngspice_match'`.

A/B conditions:
  - OFF (legacy_into_body, tlpe1_disable True)  ← reproduces prior 0.461
  - ON  (ngspice_match,    tlpe1_disable True)  ← new fix on top

Smoke-probe Vb at the 3 worst over-conduction biases (VG1=0.6, VG2∈{-0.1,0,0.1},
Vd=0.1V) before and after to confirm Vb collapses from ~0.49 to ~0.24 V.

Output: results/track_well_diode_fix/{ablation.json, combo_k1_alpha0.json}
"""
from __future__ import annotations
import os, sys, json, time, importlib.util, traceback
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

from pathlib import Path
import numpy as np
import torch

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/track_well_diode_fix"
OUT.mkdir(parents=True, exist_ok=True)

sp = importlib.util.spec_from_file_location("combo", ROOT / "scripts/track_combo_k1_alpha0.py")
combo = importlib.util.module_from_spec(sp); sp.loader.exec_module(combo)
pillar = combo.pillar

K1_CARD     = combo.K1_CARD       # 0.53825
ALPHA0_CARD = combo.ALPHA0_CARD   # 7.83756e-4


def vb_probe(cfg, M1, M2, bjt, vg1=0.6, vg2_list=(-0.1, 0.0, 0.1), vd=0.10):
    """Return dict {(vg1,vg2,vd): {'Vb': float, 'Id': float}} at one Vd point."""
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    out_probe = {}
    Vd_seq = torch.tensor([vd], dtype=torch.float64)
    for vg2 in vg2_list:
        try:
            out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_seq,
                             VG1=torch.tensor(vg1, dtype=torch.float64),
                             VG2=torch.tensor(vg2, dtype=torch.float64),
                             warm_start=False)
            Vb = float(out["Vb"][0].detach().cpu().numpy())
            Id = float(np.abs(out["Id"][0].detach().cpu().numpy()))
        except Exception as e:
            Vb = float("nan"); Id = float("nan")
        out_probe[f"VG1={vg1}_VG2={vg2}_Vd={vd}"] = {"Vb": Vb, "Id": Id}
    return out_probe


def run_with_flags(tag, well_mode):
    print(f"\n=== {tag}  well_diode_mode={well_mode}  tlpe1=ON  K1={K1_CARD} ALPHA0={ALPHA0_CARD} ===", flush=True)
    sebas_rows = pillar.load_sebas_params()
    curves = pillar.load_curves()
    print(f"  loaded {len(curves)} curves, {len(sebas_rows)} sebas rows", flush=True)

    cfg, M1, M2, bjt = pillar.build_pyport_base()
    # tlpe1 fix (already validated)
    M1._values["tlpe1_disable"] = True
    M2._values["tlpe1_disable"] = True
    # new well_diode fix gated via cfg
    cfg.well_diode_mode = well_mode

    # Vb smoke probe (3 biases at Vd=0.10) BEFORE running full sweep
    probe = vb_probe(cfg, M1, M2, bjt)
    print(f"  Vb probe: {probe}", flush=True)

    saved_branch_k1 = pillar.BRANCH_FLAT[0.6]["K1"]
    pillar.BRANCH_FLAT[0.6]["K1"] = float(K1_CARD)
    orig_make = pillar.make_overrides
    def patched_make(sebas_row):
        P_M1, P_M2 = orig_make(sebas_row)
        if P_M1 is None: P_M1 = {}
        if P_M2 is None: P_M2 = {}
        P_M1["alpha0"] = float(ALPHA0_CARD)
        P_M2["alpha0"] = float(ALPHA0_CARD)
        if sebas_row is not None and abs(sebas_row.get("VG1", float("nan")) - 0.6) < 1e-6:
            P_M1["k1"] = float(K1_CARD)
        return P_M1, P_M2
    pillar.make_overrides = patched_make
    try:
        t0 = time.time()
        rows, nan_count = pillar.run_grid(cfg, M1, M2, bjt, curves, sebas_rows, tag, do_bwd=True)
        dt = time.time() - t0
    finally:
        pillar.make_overrides = orig_make
        pillar.BRANCH_FLAT[0.6]["K1"] = saved_branch_k1
        M1._values.pop("tlpe1_disable", None)
        M2._values.pop("tlpe1_disable", None)

    summ = pillar.summarize(rows, tag)
    summ["k1"] = float(K1_CARD); summ["alpha0"] = float(ALPHA0_CARD)
    summ["tlpe1_disable"] = True
    summ["well_diode_mode"] = well_mode
    summ["nan_count"] = int(nan_count); summ["runtime_s"] = float(dt)
    summ["n_rows"] = len(rows)
    summ["n_finite"] = sum(1 for r in rows if np.isfinite(r["med_dec"]) and r["med_dec"] > 0)
    summ["vb_probe"] = probe
    print(f"  median_dec_all = {summ['median_dec_all']['median']:.3f}   runtime = {dt:.0f}s", flush=True)
    return summ


def main():
    results = {}
    for tag, mode in [("OFF_legacy_into_body", "legacy_into_body"),
                      ("ON_ngspice_match",     "ngspice_match")]:
        try:
            results[tag] = run_with_flags(tag, mode)
        except Exception as e:
            traceback.print_exc()
            results[tag] = {"error": str(e)}
        with open(OUT / "ablation.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
    off = results.get("OFF_legacy_into_body", {}).get("median_dec_all", {}).get("median", float("nan"))
    on  = results.get("ON_ngspice_match",     {}).get("median_dec_all", {}).get("median", float("nan"))
    print("\n=== SUMMARY (well_diode_fix on top of Tlpe1) ===")
    print(f"  OFF legacy_into_body : median_dec_all = {off:.3f}  (expected ~0.461)")
    print(f"  ON  ngspice_match    : median_dec_all = {on:.3f}")
    print(f"  Δ (on − off)         = {on - off:+.3f} dec")
    return 0


if __name__ == "__main__":
    sys.exit(main())
