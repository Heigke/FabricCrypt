"""z231 — B.1/F7: ngspice vs pyport at production params, 9 biases.

Reads b1_out.csv (output of test_2t_cell_prod.sp) and compares against
pyport _solve_at_fixed_vb at Vb = ngspice's reported steady-state Vb.

Production BJT: Bf=9000, Va=0.55, Is=1e-9 (matches z220 surrogate
generation env via OPT_BF/OPT_VA/OPT_IS in nsram_surrogate_4d).

If max log10|Id| error < 0.5 dec on all 9 biases, the pyport-direct
path matches the third-party silicon simulator on the same regime
that z230 already validated against the surrogate. Triangulation:
  pyport ↔ surrogate (z230)   max 0.39 dec ✅
  pyport ↔ ngspice (z231)     gate <0.50 dec
  ⇒ surrogate ↔ ngspice transitive

Closes the R-track ngspice 3-node cross-check criterion.
"""
from __future__ import annotations
import os, sys, json, csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))
OUT = ROOT / "results/z231_b1_ngspice"; OUT.mkdir(parents=True, exist_ok=True)
CSV = ROOT / "research_plan/ngspice_repro_harness/b1_out.csv"


def main():
    from scripts.nsram_surrogate_4d import _solve_at_fixed_vb, _build_pyport_models
    cfg, M1, M2, bjt = _build_pyport_models()

    rows = list(csv.DictReader(open(CSV)))
    print(f"=== z231 B.1: ngspice vs pyport @ production BJT (Bf=9000, Va=0.55, Is=1e-9), 9 biases ===")
    print(f"\n  VG1   VG2    Vd   Vb_ng    Id_ng        Id_pyp       Δdec   conv")

    deltas = []
    rows_out = []
    for r in rows:
        vg1 = float(r["VG1"]); vg2 = float(r["VG2"]); vd = float(r["Vd"])
        vb_ng = float(r["Vb"]); id_ng = float(r["Id"])
        out = _solve_at_fixed_vb(cfg, M1, M2, bjt, vd, vg1, vg2, vb_ng)
        id_py = float(out["Id"])
        log_ng = np.log10(max(abs(id_ng), 1e-15))
        log_py = np.log10(max(abs(id_py), 1e-15))
        d = abs(log_ng - log_py)
        deltas.append(d)
        rows_out.append({"VG1": vg1, "VG2": vg2, "Vd": vd, "Vb_ng": vb_ng,
                          "Id_ng": id_ng, "Id_pyp": id_py, "delta_dec": d,
                          "converged": bool(out["converged"])})
        print(f"  {vg1:.2f}  {vg2:.2f}  {vd:.2f}  {vb_ng:.4f}  "
              f"{id_ng:.3e}  {id_py:.3e}  {d:.3f}  {out['converged']}")

    deltas = np.array(deltas)
    rms = float(np.sqrt((deltas**2).mean()))
    mx = float(deltas.max())
    p95 = float(np.quantile(deltas, 0.95))

    print(f"\nlog10|Id|  ngspice (silicon-grade) vs pyport (_solve_at_fixed_vb):")
    print(f"  RMS    : {rms:.3f}")
    print(f"  P95    : {p95:.3f}")
    print(f"  MAX    : {mx:.3f}  (gate <0.50 PASS)")
    print(f"  GATE   : {'✅ PASS' if mx < 0.50 else '❌ FAIL'}")

    out = {"N": len(rows), "rms_dec": rms, "p95_dec": p95, "max_dec": mx,
            "gate_pass": bool(mx < 0.50), "biases": rows_out}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
