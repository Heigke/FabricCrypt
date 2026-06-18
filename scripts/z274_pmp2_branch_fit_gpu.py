"""PMP-2: Per-V_G1-branch DC polynomial fit on GPU (batched).

Closed-form lstsq for log10|I_d| ~ poly(V_d deg=3, V_G2 deg=2) fit
independently per V_G1 branch ∈ {0.20, 0.40, 0.60} V, all batched in a
single torch.linalg.lstsq call on AMD gfx1151.

Pre-registered gate (research_plan/01_LOG.md, 2026-05-11):
  PASS if V_G1=0.60 branch residual ≤ 3.04 dec
  AMBITIOUS if V_G1=0.60 branch ≤ M2's V_G1=0.20 branch (0.86 dec)

NO-CHEAT: poly degrees (3 in V_d, 2 in V_G2) locked; no row exclusion;
same basis across all branches.

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/z274_pmp2_branch_fit_gpu.py
"""

import json
import math
import os
from collections import defaultdict

import torch

PRED_PATH = "results/z91f_validate_sebas/predictions.json"
M2_PATH = "results/z256_m2_branch_residual/summary.json"
OUT_DIR = "results/z274_pmp2_branch_fit"
OUT_PATH = os.path.join(OUT_DIR, "summary.json")

VG1_BRANCHES = (0.20, 0.40, 0.60)
DEG_VD = 3
DEG_VG2 = 2
PASS_THRESH_DEC = 3.04          # V_G1=0.60 must hit this
AMBITIOUS_THRESH_DEC = 0.86     # ambitious: close to V_G1=0.20 M2 level

# M2 baseline (from results/z256_m2_branch_residual/summary.json)
M2_RESID = {0.20: 0.8598448731382842,
            0.40: 2.3765546766500627,
            0.60: 3.53520172736754}


def build_features(vd, vg2, deg_vd=DEG_VD, deg_vg2=DEG_VG2):
    """Polynomial feature matrix [1, vd, vd^2, vd^3, vg2, vg2^2,
    vd*vg2, vd^2*vg2, vd^3*vg2, vd*vg2^2, vd^2*vg2^2, vd^3*vg2^2].

    Full tensor-product basis up to (deg_vd, deg_vg2). Locked basis;
    same for all 3 branches. Includes cross terms because the tensor
    product is the natural degree-(deg_vd, deg_vg2) bivariate poly.
    """
    cols = []
    for j in range(deg_vg2 + 1):
        for i in range(deg_vd + 1):
            cols.append((vd ** i) * (vg2 ** j))
    return torch.stack(cols, dim=-1)  # (N, P)


def load_branch_rows():
    with open(PRED_PATH) as f:
        data = json.load(f)
    branches = defaultdict(lambda: {"vd": [], "vg2": [], "y": []})
    n_used = 0
    n_skipped_curves = 0
    for curve in data:
        if curve.get("skipped"):
            n_skipped_curves += 1
            continue
        vg1 = round(float(curve["VG1"]), 2)
        if vg1 not in VG1_BRANCHES:
            continue
        vg2 = float(curve["VG2"])
        for vd, im in zip(curve["Vd"], curve["Id_meas"]):
            if vd is None or im is None:
                continue
            try:
                vd_f = float(vd); im_f = float(im)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(vd_f) or not math.isfinite(im_f):
                continue
            if im_f == 0.0:
                continue
            branches[vg1]["vd"].append(vd_f)
            branches[vg1]["vg2"].append(vg2)
            branches[vg1]["y"].append(math.log10(abs(im_f)))
            n_used += 1
    return branches, n_used, n_skipped_curves


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[z274] device = {device}", flush=True)
    if device.type == "cuda":
        print(f"[z274] gpu = {torch.cuda.get_device_name(0)}", flush=True)

    branches, n_used, n_skipped = load_branch_rows()
    print(f"[z274] loaded {n_used} rows across {len(branches)} branches "
          f"(skipped {n_skipped} curves)", flush=True)

    # Pad to equal N for batched lstsq (or pad with mask)
    per_branch_N = {b: len(branches[b]["y"]) for b in VG1_BRANCHES}
    N_max = max(per_branch_N.values())
    P = (DEG_VD + 1) * (DEG_VG2 + 1)

    # Build padded batched (A, y, mask)
    A_batch = torch.zeros(len(VG1_BRANCHES), N_max, P, dtype=torch.float64)
    y_batch = torch.zeros(len(VG1_BRANCHES), N_max, dtype=torch.float64)
    mask = torch.zeros(len(VG1_BRANCHES), N_max, dtype=torch.float64)

    for k, vg1 in enumerate(VG1_BRANCHES):
        b = branches[vg1]
        n = len(b["y"])
        vd = torch.tensor(b["vd"], dtype=torch.float64)
        vg2 = torch.tensor(b["vg2"], dtype=torch.float64)
        A = build_features(vd, vg2)
        A_batch[k, :n] = A
        y_batch[k, :n] = torch.tensor(b["y"], dtype=torch.float64)
        mask[k, :n] = 1.0

    A_batch = A_batch.to(device)
    y_batch = y_batch.to(device)
    mask = mask.to(device)

    # Weighted normal equations: solve (A^T W A) x = A^T W y per branch,
    # batched. Rocm torch.linalg.lstsq may not support batched fp64; use
    # solve on normal-equation form which is fast and explicit (P=12 small).
    Aw = A_batch * mask.unsqueeze(-1)         # (B, N, P)
    yw = y_batch * mask                       # (B, N)
    AtA = torch.matmul(Aw.transpose(-1, -2), Aw)       # (B, P, P)
    Aty = torch.matmul(Aw.transpose(-1, -2), yw.unsqueeze(-1)).squeeze(-1)  # (B, P)
    # Tiny ridge for numerical stability; epsilon irrelevant at scale of
    # AtA diagonal ~ N. NOT a regularizer — locked at 1e-12.
    eye = torch.eye(P, dtype=torch.float64, device=device).unsqueeze(0)
    coeffs = torch.linalg.solve(AtA + 1e-12 * eye, Aty.unsqueeze(-1)).squeeze(-1)  # (B, P)

    # Per-branch residuals (mean |y_pred - y_meas|, same metric as M2)
    y_pred = torch.matmul(A_batch, coeffs.unsqueeze(-1)).squeeze(-1)  # (B, N_max)
    abs_err = (y_pred - y_batch).abs() * mask
    n_per = mask.sum(dim=-1)
    mean_resid = (abs_err.sum(dim=-1) / n_per).cpu().tolist()
    coeffs_cpu = coeffs.cpu().tolist()

    branch_summary = {}
    for k, vg1 in enumerate(VG1_BRANCHES):
        m2 = M2_RESID[vg1]
        r = mean_resid[k]
        branch_summary[f"branch_VG1_{vg1:.2f}"] = {
            "n_rows": int(n_per[k].item()),
            "mean_resid_dec_poly": r,
            "m2_resid_dec": m2,
            "delta_vs_M2": r - m2,             # negative = improvement
            "improvement_dec": m2 - r,
            "coeffs": coeffs_cpu[k],
        }

    r60 = mean_resid[VG1_BRANCHES.index(0.60)]
    verdict = "PASS" if r60 <= PASS_THRESH_DEC else "FAIL"
    ambitious = "PASS" if r60 <= AMBITIOUS_THRESH_DEC else "FAIL"
    margin = PASS_THRESH_DEC - r60

    summary = {
        "source_predictions": PRED_PATH,
        "m2_source": M2_PATH,
        "device": str(device),
        "deg_vd": DEG_VD,
        "deg_vg2": DEG_VG2,
        "n_features": P,
        "feature_basis": "tensor product 1, vd, vd^2, vd^3 outer 1, vg2, vg2^2 (12 features)",
        "n_rows_used": n_used,
        "n_curves_skipped_upstream": n_skipped,
        **branch_summary,
        "pass_threshold_dec": PASS_THRESH_DEC,
        "ambitious_threshold_dec": AMBITIOUS_THRESH_DEC,
        "verdict": verdict,
        "verdict_margin_dec": margin,
        "ambitious_verdict": ambitious,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    # Print concise console report
    print()
    for vg1 in VG1_BRANCHES:
        k = f"branch_VG1_{vg1:.2f}"
        s = branch_summary[k]
        print(f"  VG1={vg1:.2f}: poly={s['mean_resid_dec_poly']:.3f} dec  "
              f"M2={s['m2_resid_dec']:.3f} dec  "
              f"Δ={s['delta_vs_M2']:+.3f} dec  "
              f"(improvement {s['improvement_dec']:+.3f} dec)")
    print()
    print(f"  VG1=0.60 residual: {r60:.3f} dec   threshold: {PASS_THRESH_DEC:.3f} dec")
    print(f"  verdict: {verdict}   margin: {margin:+.3f} dec")
    print(f"  ambitious (≤ {AMBITIOUS_THRESH_DEC:.3f} dec): {ambitious}")
    print(f"  wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
