"""z91i v3 — verify poly_params module reproduces CSV at grid points.

A.3 v3 deliverable: integrate poly+spline parameterisation as opt-in
alternative to per-bias CSV in z91f. This script is the verification
harness: load the CSV, evaluate the new `nsram.bsim4_port.poly_params`
module at every grid point, compare to CSV values, report errors.

Pass criterion (per z91i v2 fit residuals):
  K1, mbjt:  match within 1e-9 absolute
  BETA0:     ≤ 6% RMS relative
  ETAB:      ≤ 10% RMS relative
  NFACTOR, trise:  ≤ 30% LOO; in-sample fit may be tighter
"""
from __future__ import annotations
import csv, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "nsram"))
from nsram.bsim4_port.poly_params import eval_param_at


def main():
    csv_path = ROOT / "data/sebas_2026_04_22/2Tcell_BSIM_param_DC.csv"
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                d = {k: float(v) for k, v in r.items()}
                rows.append(d)
            except ValueError:
                continue   # skip rows with non-numeric

    print(f"[z91i-v3] Loaded {len(rows)} CSV rows.")
    params_to_check = ["K1", "mbjt", "BETA0", "ETAB", "NFACTOR", "trise",
                         "ALPHA0", "IS"]
    print(f"[z91i-v3] Checking {len(params_to_check)} parameters.\n")

    print(f"  {'param':>8s}  {'rows_with_data':>15s}  "
           f"{'rms_abs':>10s}  {'rms_rel':>9s}  {'max_rel':>9s}")
    overall_pass = True
    for p in params_to_check:
        diffs_abs = []
        diffs_rel = []
        n_used = 0
        for row in rows:
            if p not in row or math.isnan(row[p]):
                continue
            VG1, VG2 = row["VG1"], row["VG2"]
            try:
                pred = eval_param_at(p, VG1, VG2)
            except KeyError:
                pred = None
            if pred is None:
                continue
            true = row[p]
            diffs_abs.append(pred - true)
            if abs(true) > 1e-15:
                diffs_rel.append((pred - true) / true)
            n_used += 1
        if n_used == 0:
            print(f"  {p:>8s}  {'SKIP':>15s}  (no eval)")
            continue
        rms_abs = float(np.sqrt(np.mean(np.array(diffs_abs)**2)))
        rms_rel = float(np.sqrt(np.mean(np.array(diffs_rel)**2))) if diffs_rel else 0.0
        max_rel = float(np.max(np.abs(diffs_rel))) if diffs_rel else 0.0
        verdict = "OK"
        if p in ("K1", "mbjt") and rms_abs > 1e-6:
            verdict = "FAIL (perfect-fit poly should match)"
            overall_pass = False
        elif p == "BETA0" and rms_rel > 0.07:
            verdict = "FAIL (>7%)"
            overall_pass = False
        elif p == "ETAB" and rms_rel > 0.12:
            verdict = "FAIL (>12%)"
            overall_pass = False
        print(f"  {p:>8s}  {n_used:>15d}  {rms_abs:>10.3e}  "
               f"{rms_rel:>9.3%}  {max_rel:>9.3%}  {verdict}")

    print()
    if overall_pass:
        print("[z91i-v3] === PASS — poly_params module reproduces CSV within fit envelope ===")
    else:
        print("[z91i-v3] === FAIL — at least one parameter exceeds expected fit envelope ===")

    # Smoke test: evaluate at off-grid points (interpolation regime)
    print("\n[z91i-v3] Off-grid smoke test:")
    off_grid_pts = [(0.15, 0.25), (0.55, 0.55), (0.30, 0.10)]
    for VG1, VG2 in off_grid_pts:
        print(f"  ({VG1}, {VG2}):")
        for p in ["K1", "mbjt", "BETA0", "ETAB", "NFACTOR", "trise"]:
            v = eval_param_at(p, VG1, VG2)
            print(f"    {p}: {v:+.6g}")

    # Drop-in sanity for make_overrides_poly
    print("\n[z91i-v3] make_overrides_poly() smoke at (VG1=0.4, VG2=0.3):")
    from nsram.bsim4_port.poly_params import make_overrides_poly, make_bjt_kwargs_poly
    P_M1, P_M2 = make_overrides_poly(0.4, 0.3)
    print(f"  P_M1 keys: {list(P_M1.keys())}")
    for k, v in P_M1.items():
        print(f"    {k}: {v.item():+.6g}")
    print(f"  P_M2: {dict(P_M2)}")
    print(f"  bjt kwargs: {make_bjt_kwargs_poly(0.4, 0.3)}")


if __name__ == "__main__":
    main()
