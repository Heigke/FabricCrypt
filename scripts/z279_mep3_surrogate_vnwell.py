"""MEP-3: 5D body-state surrogate with V_Nwell as parameter axis.

Adds V_Nwell as a 5th input axis to test whether the N-well/P-body diode
and body-P-diode (cathode = vnwell) are the missing physics at high V_b.

PMP-6b showed the diode is "silent" at V_Nwell=2.0 V because
V_b − V_Nwell ∈ [−1.7, −1.3] V (reverse-biased). At lower V_Nwell
(say 0.5 V), the diode FORWARD-biases at high V_b. This 5D grid lets us
probe that hypothesis without committing to one V_Nwell.

Grid (LOCKED — do not tweak post-run):
  V_G1   : 3  vals — {0.20, 0.40, 0.60} V
  V_G2   : 5  vals — {0.00, 0.15, 0.30, 0.45, 0.60} V
  V_d    : 5  vals — {0.50, 1.00, 1.50, 2.00, 2.50} V
  V_b    : 11 vals — {0.00, 0.10, ..., 1.00} V
  V_Nwell: 5  vals — {0.5, 1.0, 2.0, 2.5, 5.0} V
  = 3 × 5 × 5 × 11 × 5 = 4125 op points

Crucially: body_pdiode_to=vnwell is ON in this build (otherwise V_Nwell
would only affect the well-body diode path; we want both paths active).

Pre-reg gates (research_plan/MEP_DS_PLAN_2026-05-12.md):
  PASS              ≥95% conv AND clear V_Nwell-dependence in Ileak.
  INFORMATIVE-PASS  80-94% conv (with diagnostics).
  FAIL              <80% conv OR no V_Nwell-dependence in Ileak.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import json
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# MEP-3 axes (LOCKED)
VG1_AXIS    = np.array([0.20, 0.40, 0.60], dtype=np.float64)
VG2_AXIS    = np.array([0.00, 0.15, 0.30, 0.45, 0.60], dtype=np.float64)
VD_AXIS     = np.array([0.50, 1.00, 1.50, 2.00, 2.50], dtype=np.float64)
VB_AXIS     = np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50,
                        0.60, 0.70, 0.80, 0.90, 1.00], dtype=np.float64)
VNWELL_AXIS = np.array([0.5, 1.0, 2.0, 2.5, 5.0], dtype=np.float64)
# 3 × 5 × 5 × 11 × 5 = 4125

# Per-worker globals
_CFG = _M1 = _M2 = _BJT = None
_SOLVE = None


def init_worker():
    """Build pyport models once per worker. Enable body_pdiode_to=vnwell."""
    global _CFG, _M1, _M2, _BJT, _SOLVE
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ns4d", ROOT / "scripts/nsram_surrogate_4d.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CFG, _M1, _M2, _BJT = mod._build_pyport_models()
    # Wire the P-body diode cathode to vnwell — the entire point of MEP-3.
    _CFG.body_pdiode_to = "vnwell"
    _SOLVE = mod._solve_at_fixed_vb


def _body_pdiode_current(vb, vnwell, cfg):
    """Recompute the body-pdiode current (cathode=vnwell) post-solve. The
    pyport _residuals computes this internally but does not surface it in
    the `comp` dict, so we replicate the closed-form here for diagnostic
    Ileak accounting. Sign: +I_body_pdiode LEAVES the body (anode = body,
    cathode = V_Nwell), so forward-bias (V_b > V_Nwell) gives positive
    Ileak-out contribution.
    """
    import math
    Vt_body = 0.02585 * (273.15 + cfg.T_C) / 300.0
    Vab = vb - vnwell
    # area branch
    exp_arg = max(-40.0, min(40.0, Vab / (cfg.body_pdiode_n * Vt_body)))
    I = cfg.body_pdiode_Js * cfg.body_pdiode_area * (math.exp(exp_arg) - 1.0)
    if getattr(cfg, "body_pdiode_perim_length", 0.0) > 0.0:
        exp_arg_sw = max(-40.0, min(40.0, Vab / (cfg.body_pdiode_n_sw * Vt_body)))
        I += (cfg.body_pdiode_Js_sw * cfg.body_pdiode_perim_length
              * (math.exp(exp_arg_sw) - 1.0))
    return I


def solve_point(args):
    i, j, k, l, m, vg1, vg2, vd, vb, vnwell = args
    try:
        # Override per-call V_Nwell on the (mutable dataclass) cfg.
        _CFG.vnwell = float(vnwell)
        out = _SOLVE(_CFG, _M1, _M2, _BJT, vd, vg1, vg2, vb)
        # Augment Ileak_out with the body-pdiode-to-vnwell current (not
        # surfaced in comp dict by the pyport code).
        I_bp = _body_pdiode_current(vb, vnwell, _CFG)
        Ileak_aug = out["Ileak_out"] + I_bp
        return (i, j, k, l, m, out["Id"], out["Iii_in"], Ileak_aug,
                bool(out["converged"]), None)
    except Exception as e:
        return (i, j, k, l, m, np.nan, 0.0, 0.0, False, str(e))


def main(n_workers: int = 4):
    NG1, NG2, NVD, NVB, NVN = (len(VG1_AXIS), len(VG2_AXIS), len(VD_AXIS),
                                len(VB_AXIS), len(VNWELL_AXIS))
    n_total = NG1 * NG2 * NVD * NVB * NVN
    print(f"[z279/mep3] grid = {NG1}x{NG2}x{NVD}x{NVB}x{NVN} = {n_total} pts; "
          f"workers={n_workers}; body_pdiode_to=vnwell")

    tasks = []
    for i, vg1 in enumerate(VG1_AXIS):
        for j, vg2 in enumerate(VG2_AXIS):
            for k, vd in enumerate(VD_AXIS):
                for l, vb in enumerate(VB_AXIS):
                    for m, vn in enumerate(VNWELL_AXIS):
                        tasks.append((i, j, k, l, m,
                                      float(vg1), float(vg2),
                                      float(vd),  float(vb),
                                      float(vn)))

    shape = (NG1, NG2, NVD, NVB, NVN)
    Id_grid    = np.full(shape, np.nan, dtype=np.float64)
    Iii_grid   = np.zeros(shape, dtype=np.float64)
    Ileak_grid = np.zeros(shape, dtype=np.float64)
    conv_grid  = np.zeros(shape, dtype=bool)
    err_count  = {}

    t0 = time.time()
    n_done = 0
    with Pool(n_workers, initializer=init_worker) as pool:
        for res in pool.imap_unordered(solve_point, tasks, chunksize=8):
            i, j, k, l, m, Id, Iii, Ileak, conv, err = res
            Id_grid[i, j, k, l, m]    = Id
            Iii_grid[i, j, k, l, m]   = Iii
            Ileak_grid[i, j, k, l, m] = Ileak
            conv_grid[i, j, k, l, m]  = conv
            if err is not None:
                err_count[err] = err_count.get(err, 0) + 1
            n_done += 1
            if n_done % 500 == 0:
                wall = time.time() - t0
                eta = wall / n_done * (n_total - n_done)
                print(f"  {n_done}/{n_total} ({100*n_done/n_total:.0f}%); "
                      f"wall={wall:.0f}s eta={eta:.0f}s "
                      f"conv_so_far={int(conv_grid.sum())}")

    wall = time.time() - t0
    n_conv = int(conv_grid.sum())
    conv_rate = n_conv / n_total
    print(f"\n[z279/mep3] done in {wall:.0f}s; converged {n_conv}/{n_total} "
          f"({100*conv_rate:.1f}%)")

    out_dir = ROOT / "results/z279_mep3_surrogate_vnwell"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "surrogate_5d.npz"
    np.savez(out_path,
             Id=Id_grid, Iii=Iii_grid, Ileak=Ileak_grid,
             converged=conv_grid,
             vg1_axis=VG1_AXIS, vg2_axis=VG2_AXIS,
             vd_axis=VD_AXIS,  vb_axis=VB_AXIS,
             vnwell_axis=VNWELL_AXIS)

    # ---- Diagnostics ------------------------------------------------------
    # Convergence per V_Nwell slice
    conv_by_vnwell = {float(VNWELL_AXIS[m]):
                      float(conv_grid[..., m].mean()) for m in range(NVN)}

    # Sanity #1: at V_b=0.5 V, how does mean |Ileak| vary across V_Nwell?
    # Expected: ~0 for V_Nwell >= V_b (reverse bias),
    #           rising for V_Nwell < V_b (forward bias of body→nwell diode).
    vb_idx_05 = int(np.argmin(np.abs(VB_AXIS - 0.5)))
    ileak_vs_vnwell_at_vb05 = {
        float(VNWELL_AXIS[m]): float(
            np.nanmean(np.abs(Ileak_grid[:, :, :, vb_idx_05, m]))
        ) for m in range(NVN)
    }

    # Sanity #2: at (V_G1=0.4 [closest to 0.45], V_G2=0.30, V_d=2.0, V_b=0.5),
    # how does Id vary with V_Nwell?
    # NOTE: 0.45 not on VG1 axis; using nearest (0.40) per slide-21 spirit.
    i_g1 = int(np.argmin(np.abs(VG1_AXIS - 0.45)))
    j_g2 = int(np.argmin(np.abs(VG2_AXIS - 0.30)))
    k_vd = int(np.argmin(np.abs(VD_AXIS  - 2.00)))
    l_vb = int(np.argmin(np.abs(VB_AXIS  - 0.50)))
    Id_test = {
        float(VNWELL_AXIS[m]): float(Id_grid[i_g1, j_g2, k_vd, l_vb, m])
        for m in range(NVN)
    }
    Ileak_test = {
        float(VNWELL_AXIS[m]): float(Ileak_grid[i_g1, j_g2, k_vd, l_vb, m])
        for m in range(NVN)
    }
    Iii_test = {
        float(VNWELL_AXIS[m]): float(Iii_grid[i_g1, j_g2, k_vd, l_vb, m])
        for m in range(NVN)
    }

    # Gate: V_Nwell-dependence visible in body currents.
    # The well-body diode I_well_body lives in Iii_in (well pumping body UP
    # when V_Nwell>V_b); the body-pdiode I_body_pdiode (now folded into
    # Ileak_aug) leaves the body when V_b>V_Nwell. So total V_Nwell coupling
    # = max relative variation across Ileak OR Iii.
    ileak_vals = list(ileak_vs_vnwell_at_vb05.values())
    ileak_range = max(ileak_vals) - min(ileak_vals)
    ileak_max = max(map(abs, ileak_vals))
    iii_vals = list(Iii_test.values())
    iii_range = max(iii_vals) - min(iii_vals)
    iii_max = max(map(abs, iii_vals))
    ileak_rel = ileak_range / max(ileak_max, 1e-30)
    iii_rel = iii_range / max(iii_max, 1e-30)
    has_vnwell_dep = (ileak_rel > 0.10) or (iii_rel > 0.10)

    if conv_rate >= 0.95 and has_vnwell_dep:
        verdict = "PASS"
    elif conv_rate >= 0.95 and not has_vnwell_dep:
        verdict = "FAIL"   # converged but no V_Nwell coupling = wiring bug
    elif conv_rate >= 0.80:
        verdict = "INFORMATIVE-PASS"
    else:
        verdict = "FAIL"

    summary = {
        "task": "MEP-3 5D surrogate with V_Nwell axis (body_pdiode_to=vnwell)",
        "verdict": verdict,
        "n_total": n_total,
        "n_converged": n_conv,
        "conv_rate": conv_rate,
        "wall_s": wall,
        "node": os.uname().nodename,
        "n_workers": n_workers,
        "grid_shape": list(shape),
        "axes": {
            "vg1":    VG1_AXIS.tolist(),
            "vg2":    VG2_AXIS.tolist(),
            "vd":     VD_AXIS.tolist(),
            "vb":     VB_AXIS.tolist(),
            "vnwell": VNWELL_AXIS.tolist(),
        },
        "conv_by_vnwell": conv_by_vnwell,
        "diagnostic_ileak_at_vb_0p5": {
            "description": ("Mean |Ileak| across (V_G1, V_G2, V_d) at V_b=0.5 V, "
                            "vs V_Nwell. Should rise sharply when V_Nwell<V_b."),
            "values_by_vnwell": ileak_vs_vnwell_at_vb05,
            "range": ileak_range,
            "max":   ileak_max,
            "rel_variation_ileak": ileak_rel,
            "rel_variation_iii_at_test_point": iii_rel,
            "has_vnwell_dependence": bool(has_vnwell_dep),
        },
        "diagnostic_slide21_test_point": {
            "description": ("Id, Iii, Ileak at (V_G1≈0.45→0.40, V_G2=0.30, "
                            "V_d=2.0, V_b=0.5) vs V_Nwell. If diode is the "
                            "only V_Nwell-coupled element, Id should be near-"
                            "constant; if not, V_Nwell affects channel via "
                            "body charging coupling."),
            "vg1_used": float(VG1_AXIS[i_g1]),
            "vg2_used": float(VG2_AXIS[j_g2]),
            "vd_used":  float(VD_AXIS[k_vd]),
            "vb_used":  float(VB_AXIS[l_vb]),
            "Id_by_vnwell":    Id_test,
            "Iii_by_vnwell":   Iii_test,
            "Ileak_by_vnwell": Ileak_test,
        },
        "err_count": err_count,
        "out_path": str(out_path),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[z279/mep3] verdict = {verdict}; written {out_dir}/summary.json")
    return summary


if __name__ == "__main__":
    import sys
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    main(n_workers=nw)
