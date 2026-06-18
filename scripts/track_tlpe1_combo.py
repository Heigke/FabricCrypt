#!/usr/bin/env python3
"""Run the canonical K1+ALPHA0 combo (K1=0.53825, ALPHA0=7.83756e-4) on the
full 33-bias (fwd+bwd, n=66) pyport DC fit, with the new `tlpe1_disable=True`
model flag applied. Compare median_dec vs the prior baseline 0.665.

Also runs the OFF condition for direct A/B (same code-path) to confirm no
regression in the flag-off path.

Output: results/track_tlpe1_fix/combo_k1_alpha0.json
"""
from __future__ import annotations
import os, sys, json, time, importlib.util
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

from pathlib import Path
import numpy as np
ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/track_tlpe1_fix"
OUT.mkdir(parents=True, exist_ok=True)

sp = importlib.util.spec_from_file_location("combo", ROOT / "scripts/track_combo_k1_alpha0.py")
combo = importlib.util.module_from_spec(sp); sp.loader.exec_module(combo)
pillar = combo.pillar

K1_CARD     = combo.K1_CARD       # 0.53825
ALPHA0_CARD = combo.ALPHA0_CARD   # 7.83756e-4


def run_with_flag(tag, flag_disable):
    print(f"\n=== {tag}  tlpe1_disable={flag_disable}  K1={K1_CARD} ALPHA0={ALPHA0_CARD} ===")
    sebas_rows = pillar.load_sebas_params()
    curves = pillar.load_curves()
    print(f"  loaded {len(curves)} curves, {len(sebas_rows)} sebas rows", flush=True)

    cfg, M1, M2, bjt = pillar.build_pyport_base()
    M1._values["tlpe1_disable"] = bool(flag_disable)
    M2._values["tlpe1_disable"] = bool(flag_disable)

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
    summ["tlpe1_disable"] = bool(flag_disable)
    summ["nan_count"] = int(nan_count); summ["runtime_s"] = float(dt)
    summ["n_rows"] = len(rows)
    summ["n_finite"] = sum(1 for r in rows if np.isfinite(r["med_dec"]) and r["med_dec"] > 0)
    print(f"  median_dec_all = {summ['median_dec_all']['median']:.3f}   "
          f"runtime = {dt:.0f}s")
    return summ


def main():
    results = {}
    for tag, flag in [("FLAG_OFF_canonical", False),
                      ("FLAG_ON_tlpe1_disabled", True)]:
        try:
            results[tag] = run_with_flag(tag, flag)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[tag] = {"error": str(e)}
        with open(OUT / "combo_k1_alpha0.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
    off = results.get("FLAG_OFF_canonical", {}).get("median_dec_all", {}).get("median", float("nan"))
    on  = results.get("FLAG_ON_tlpe1_disabled", {}).get("median_dec_all", {}).get("median", float("nan"))
    print("\n=== SUMMARY ===")
    print(f"  FLAG OFF (canonical):  median_dec_all = {off:.3f}  (expected ~0.665)")
    print(f"  FLAG ON  (tlpe1_dis):  median_dec_all = {on:.3f}")
    print(f"  Δ (on − off)         = {on - off:+.3f} dec")
    return 0


if __name__ == "__main__":
    sys.exit(main())
