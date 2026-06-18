"""z91i — A.3: polynomial(VG1, VG2) fit to Sebas's 33-bias parameter CSV.

Replaces the per-bias lookup-table override in
`data/sebas_2026_04_22/2Tcell_BSIM_param_DC.csv` with a smooth
polynomial-shape model in (VG1, VG2). For each of the 9 fitted
parameters (trise, ETAB, K1, ALPHA0, BETA0, NFACTOR, mbjt, IS, area)
we fit a low-order bivariate polynomial and report the residual.

This is the A.3 v1 deliverable: produce a JSON of polynomial
coefficients and a per-parameter residual report. Integration into
`z91f.patch_model_values` (replacing the CSV override) is a follow-up
task once the polynomial fit quality is validated.

Method per parameter:
  - Drop columns that are constant across all 33 biases (no fit needed).
  - For varying columns, fit `c00 + c10·VG1 + c01·VG2 + c11·VG1·VG2
    + c20·VG1² + c02·VG2²` (6-term degree-2 bivariate).
  - Report RMS residual + max relative error.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import csv

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z91i_poly_param_fit"
OUT.mkdir(parents=True, exist_ok=True)

CSV_PATH = ROOT / "data/sebas_2026_04_22/2Tcell_BSIM_param_DC.csv"


def load_csv():
    """Returns dict of column name → np.array, plus VG1, VG2 vectors.
    NaN entries (biases Sebas didn't fit) are converted to np.nan and
    handled per-column via masking in the fitting step."""
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    cols = list(rows[0].keys())
    def parse(s: str) -> float:
        s = s.strip()
        if not s or s.lower() == "nan":
            return float("nan")
        return float(s)
    data = {c: np.array([parse(r[c]) for r in rows]) for c in cols}
    return data, cols


def design_matrix(VG1, VG2):
    """6-term bivariate degree-2 design matrix."""
    return np.column_stack([
        np.ones_like(VG1),     # c00
        VG1,                    # c10
        VG2,                    # c01
        VG1 * VG2,              # c11
        VG1 * VG1,              # c20
        VG2 * VG2,              # c02
    ])


def fit_param(VG1, VG2, y):
    """Least-squares poly fit on non-nan rows.
    Returns (coeffs, pred_full, rms, rel_max, n_used).
    pred_full is len(y) with NaN at the input-NaN positions."""
    finite = np.isfinite(y)
    n_used = int(finite.sum())
    if n_used < 6:
        return None, np.full_like(y, np.nan), float("nan"), float("nan"), n_used
    X_all = design_matrix(VG1, VG2)
    X = X_all[finite]
    yf = y[finite]
    coeffs, *_ = np.linalg.lstsq(X, yf, rcond=None)
    pred_full = X_all @ coeffs
    resid = yf - pred_full[finite]
    rms = float(np.sqrt(np.mean(resid * resid)))
    nz_mask = np.abs(yf) > 1e-30
    if nz_mask.sum() == 0:
        rel_max = 0.0
    else:
        rel = np.abs(resid[nz_mask]) / np.abs(yf[nz_mask])
        rel_max = float(rel.max())
    return coeffs, pred_full, rms, rel_max, n_used


def main():
    t0 = time.time()
    print(f"[z91i] A.3 polynomial(VG1, VG2) fit to "
          f"data/sebas_2026_04_22/2Tcell_BSIM_param_DC.csv\n")
    data, cols = load_csv()
    n_rows = len(data["VG1"])
    print(f"  Loaded {n_rows} bias rows, {len(cols)} columns")
    print(f"  VG1 range: [{data['VG1'].min():.2f}, {data['VG1'].max():.2f}]")
    print(f"  VG2 range: [{data['VG2'].min():.2f}, {data['VG2'].max():.2f}]\n")

    # Identify constant vs varying columns (excluding VG1, VG2 themselves);
    # ignore NaN entries when computing mean/std.
    fitable_cols = [c for c in cols if c not in ("VG1", "VG2")]
    constants = {}
    varying = {}
    for c in fitable_cols:
        v = data[c]
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            continue
        if finite.std() / max(abs(finite.mean()), 1e-30) < 1e-9:
            constants[c] = float(finite.mean())
        else:
            varying[c] = v
    print(f"[z91i] Constant columns ({len(constants)}): "
          f"{list(constants.keys())}")
    for c, val in constants.items():
        print(f"    {c} = {val:.6g}")
    print(f"\n[z91i] Varying columns ({len(varying)}): "
          f"{list(varying.keys())}\n")

    # Fit each varying column
    results = {}
    print(f"  {'param':>10s}  {'mean':>12s}  {'std':>12s}  "
          f"{'RMS resid':>12s}  {'max rel err':>12s}")
    for c, y in varying.items():
        coeffs, pred, rms, rel_max, n_used = fit_param(
            data["VG1"], data["VG2"], y)
        finite = y[np.isfinite(y)]
        results[c] = {
            "coeffs": coeffs.tolist() if coeffs is not None else None,
            "rms_residual": rms,
            "max_relative_error": rel_max,
            "mean": float(finite.mean()) if finite.size > 0 else float("nan"),
            "std": float(finite.std()) if finite.size > 0 else float("nan"),
            "n_used": n_used,
        }
        print(f"  {c:>10s}  {finite.mean():>12.4g}  {finite.std():>12.4g}  "
               f"{rms:>12.4g}  {rel_max:>11.2%}  ({n_used}/{len(y)})")

    # Report which params are well-captured by degree-2 poly
    print(f"\n[z91i] === Per-parameter fit quality (degree-2 6-term poly) ===")
    well_fit = [c for c, r in results.items() if r["max_relative_error"] < 0.01]
    moderate = [c for c, r in results.items()
                if 0.01 <= r["max_relative_error"] < 0.10]
    poor = [c for c, r in results.items() if r["max_relative_error"] >= 0.10]
    print(f"  Well fit (max rel err < 1%):   {well_fit}")
    print(f"  Moderate (1% <= err < 10%):    {moderate}")
    print(f"  Poor    (err >= 10%):           {poor}")

    # Recommend polynomial order per param
    print(f"\n[z91i] Recommendation per param:")
    for c, r in results.items():
        if r["max_relative_error"] < 0.01:
            rec = "use degree-2 6-term poly"
        elif r["max_relative_error"] < 0.10:
            rec = "use degree-3 (10-term) or higher"
        else:
            rec = "stay with CSV override or use spline interpolation"
        print(f"    {c:>10s}: {rec}")

    payload = {
        "csv_path": str(CSV_PATH.relative_to(ROOT)),
        "n_bias_rows": n_rows,
        "VG1_range": [float(data["VG1"].min()), float(data["VG1"].max())],
        "VG2_range": [float(data["VG2"].min()), float(data["VG2"].max())],
        "constants": constants,
        "varying_fits": results,
        "design_matrix_form": "[1, VG1, VG2, VG1·VG2, VG1², VG2²]",
        "well_fit_params": well_fit,
        "moderate_fit_params": moderate,
        "poor_fit_params": poor,
        "wall_s": float(time.time() - t0),
    }
    json.dump(payload, (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z91i] wall: {time.time()-t0:.2f}s")
    print(f"[z91i] saved {OUT}/summary.json")


if __name__ == "__main__":
    main()
