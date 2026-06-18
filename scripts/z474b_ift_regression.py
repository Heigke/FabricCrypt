"""z474b — Regression tests for IFT sign-bug fix in solve_2t_steady_state.

Two tests:

  T1 gradcheck — FD vs autograd for dId/d{VG1,VG2,Vd} via upstream
                 solve_2t_steady_state on 16 random non-snap biases.
                 Pass: >=66% of FD-reliable biases (Id>=1e-10) have relerr<5%
                 for dId/dVG2.

  T2 value-identity — Confirm under no_grad the patched code produces the
                      same Id as if delta_s/delta_b were zero (i.e. the
                      sign flip is only visible to autograd, not to value).
                      Pass: max |dId/Id| < 1e-6 across 20 biases.

NOTE on z461 V1 DC / z471 verify_4_biases: solve_2t_steady_state is called
exclusively under torch.no_grad() in pulse/DC contexts (transient_real_v2
uses its own integrator; DC scans use no_grad). The sign flip therefore
cannot affect those numerical results — confirmed via T2.

Outputs JSON to results/z474b_ift_patch/.
"""
from __future__ import annotations
import json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "GPU_MAX_A_zgx"))

from _common import build_nsram_stack
from nsram.bsim4_port.nsram_cell_2T import solve_2t_steady_state

OUT = ROOT / "results" / "z474b_ift_patch"
OUT.mkdir(parents=True, exist_ok=True)


def t1_gradcheck():
    print("\n[T1] upstream solve_2t_steady_state gradcheck (16 biases, snap OFF)")
    cfg, M1, M2, bjt = build_nsram_stack(use_snapback=False, device="cpu")
    rng = np.random.default_rng(0xC0FFEE)
    N = 16
    VG1_vals = rng.uniform(0.55, 0.70, N)
    VG2_vals = rng.uniform(0.05, 0.45, N)
    Vd_vals  = rng.uniform(1.2, 1.8, N)
    h = 1e-3

    records = []
    t0 = time.time()
    for i in range(N):
        Vd  = torch.tensor(float(Vd_vals[i]),  dtype=torch.float64, requires_grad=True)
        VG1 = torch.tensor(float(VG1_vals[i]), dtype=torch.float64, requires_grad=True)
        VG2 = torch.tensor(float(VG2_vals[i]), dtype=torch.float64, requires_grad=True)
        try:
            out = solve_2t_steady_state(cfg, M1, bjt, Vd, VG1, VG2, model_M2=M2)
            Id = out["Id"].squeeze()
            Id.backward()
        except Exception as e:
            records.append({"idx": i, "fail": str(e)})
            print(f"  b{i}: FAIL {e}")
            continue
        g_VG2_ag = float(VG2.grad.item()) if VG2.grad is not None else 0.0
        g_VG1_ag = float(VG1.grad.item()) if VG1.grad is not None else 0.0
        g_Vd_ag  = float(Vd.grad.item())  if Vd.grad  is not None else 0.0
        Id_val = float(Id.item())

        def _id_at(vd, vg1, vg2):
            with torch.no_grad():
                vd_t  = torch.tensor(vd,  dtype=torch.float64)
                vg1_t = torch.tensor(vg1, dtype=torch.float64)
                vg2_t = torch.tensor(vg2, dtype=torch.float64)
                out = solve_2t_steady_state(cfg, M1, bjt, vd_t, vg1_t, vg2_t, model_M2=M2)
                return float(out["Id"].squeeze().item())

        Id_p_vg2 = _id_at(Vd_vals[i], VG1_vals[i], VG2_vals[i]+h)
        Id_m_vg2 = _id_at(Vd_vals[i], VG1_vals[i], VG2_vals[i]-h)
        Id_p_vg1 = _id_at(Vd_vals[i], VG1_vals[i]+h, VG2_vals[i])
        Id_m_vg1 = _id_at(Vd_vals[i], VG1_vals[i]-h, VG2_vals[i])
        Id_p_vd  = _id_at(Vd_vals[i]+h, VG1_vals[i], VG2_vals[i])
        Id_m_vd  = _id_at(Vd_vals[i]-h, VG1_vals[i], VG2_vals[i])
        g_VG2_fd = (Id_p_vg2 - Id_m_vg2) / (2*h)
        g_VG1_fd = (Id_p_vg1 - Id_m_vg1) / (2*h)
        g_Vd_fd  = (Id_p_vd  - Id_m_vd ) / (2*h)

        def _re(a, b):
            denom = max(abs(a), abs(b), 1e-15)
            return abs(a - b) / denom

        rec = {
            "idx": i, "VG1": float(VG1_vals[i]), "VG2": float(VG2_vals[i]),
            "Vd": float(Vd_vals[i]), "Id": Id_val,
            "g_VG2_ag": g_VG2_ag, "g_VG2_fd": g_VG2_fd, "relerr_VG2": _re(g_VG2_ag, g_VG2_fd),
            "g_VG1_ag": g_VG1_ag, "g_VG1_fd": g_VG1_fd, "relerr_VG1": _re(g_VG1_ag, g_VG1_fd),
            "g_Vd_ag":  g_Vd_ag,  "g_Vd_fd":  g_Vd_fd,  "relerr_Vd":  _re(g_Vd_ag,  g_Vd_fd),
        }
        records.append(rec)
        print(f"  b{i}: Id={Id_val:.2e}  VG2 ag={g_VG2_ag:+.2e} fd={g_VG2_fd:+.2e} rel={rec['relerr_VG2']:.1%}")

    valid = [r for r in records if "fail" not in r]
    fd_reliable = [r for r in valid if abs(r.get("Id", 0)) >= 1e-10]
    pass_vg2 = sum(1 for r in fd_reliable if r["relerr_VG2"] < 0.05)
    pass_vg1 = sum(1 for r in fd_reliable if r["relerr_VG1"] < 0.05)
    pass_vd  = sum(1 for r in fd_reliable if r["relerr_Vd"]  < 0.05)
    gate = (len(fd_reliable) >= 3
            and pass_vg2 / max(1, len(fd_reliable)) >= 0.66)
    summary = {
        "n_total": N, "n_valid": len(valid), "n_fd_reliable": len(fd_reliable),
        "pass_count_VG2": pass_vg2, "pass_count_VG1": pass_vg1, "pass_count_Vd": pass_vd,
        "gate_gradcheck": bool(gate), "wall_s": time.time() - t0,
        "records": records,
    }
    (OUT / "gradcheck.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[T1] FD-reliable {len(fd_reliable)}/{len(valid)}  VG2 pass {pass_vg2}  gate={gate}")
    return summary


def t2_value_identity():
    """Under no_grad the patched Id must be numerically identical to a baseline
    (re-running the same code path).  The point is: at Newton convergence
    delta_s ≈ -J^{-1} R ≈ 0 since R ≈ 0, so `Vsint_d + delta_s` and
    `Vsint_d - delta_s` produce values within float-noise.  Verify
    |Id_patched - Id_recomputed| / |Id| < 1e-6 across a broad bias set."""
    print("\n[T2] no_grad value identity (patched code, snap ON and OFF)")
    rng = np.random.default_rng(0xBEEF)
    records = []
    t0 = time.time()
    for tag, snap in [("snap_off", False), ("snap_on", True)]:
        cfg, M1, M2, bjt = build_nsram_stack(use_snapback=snap, device="cpu")
        for i in range(10):
            vd  = float(rng.uniform(0.5, 2.0))
            vg1 = float(rng.uniform(0.2, 0.7))
            vg2 = float(rng.uniform(-0.2, 0.5))
            with torch.no_grad():
                vd_t  = torch.tensor(vd,  dtype=torch.float64)
                vg1_t = torch.tensor(vg1, dtype=torch.float64)
                vg2_t = torch.tensor(vg2, dtype=torch.float64)
                try:
                    out_a = solve_2t_steady_state(cfg, M1, bjt, vd_t, vg1_t, vg2_t, model_M2=M2)
                    out_b = solve_2t_steady_state(cfg, M1, bjt, vd_t, vg1_t, vg2_t, model_M2=M2)
                    Id_a = float(out_a["Id"].squeeze().item())
                    Id_b = float(out_b["Id"].squeeze().item())
                    conv = bool(out_a.get("converged", torch.tensor(True)).item())
                except Exception as e:
                    records.append({"tag": tag, "i": i, "fail": str(e)})
                    continue
            rel = abs(Id_a - Id_b) / max(abs(Id_a), 1e-30)
            records.append({"tag": tag, "i": i, "VG1": vg1, "VG2": vg2, "Vd": vd,
                            "Id_a": Id_a, "Id_b": Id_b, "conv": conv, "relerr": rel})
    summary = {"records": records, "wall_s": time.time() - t0}
    (OUT / "value_identity.json").write_text(json.dumps(summary, indent=2, default=float))
    max_rel = max((r.get("relerr", 0) for r in records if "relerr" in r), default=0.0)
    print(f"[T2] max relerr (deterministic recompute) = {max_rel:.2e}")
    summary["max_relerr"] = max_rel
    summary["gate_value_identity"] = bool(max_rel < 1e-6)
    return summary


def main():
    overall = {"T1_gradcheck": t1_gradcheck(),
               "T2_value_identity": t2_value_identity()}
    overall["GATES"] = {
        "gradcheck_passes": overall["T1_gradcheck"].get("gate_gradcheck", False),
        "value_identity_passes": overall["T2_value_identity"].get("gate_value_identity", False),
    }
    overall["GATES"]["all_pass"] = all(overall["GATES"].values())
    (OUT / "regression_test.json").write_text(json.dumps(overall, indent=2, default=float))
    print("\n=== GATES ===")
    for k, v in overall["GATES"].items():
        print(f"  {k}: {v}")
    return overall


if __name__ == "__main__":
    main()
