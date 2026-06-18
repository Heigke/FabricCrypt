"""z110 — measure τ_body from the calibrated pyport transient.

Fits the actual body-cap time constant by perturbing Vb away from
its quiescent value and watching the exponential relaxation, then
reports τ_body. Combined with the configured C_body, this gives
the effective body-node resistance at the quiescent operating
point.

Resolves the C_body_eff uncertainty in the C.3 v2.1 κ↔R_bulk
mapping (which assumed 5–10 fF; the actual pyport config is 1 fF
per `nsram_cell_2T.py:87`).

Method:
  1. Pick quiescent (VG1, VG2, Vd) operating point.
  2. Solve quiescent Vb_eq via joint_newton.
  3. Perturb Vb_0 = Vb_eq + δ (δ = 0.05 V).
  4. Run transient with Vd, VG1, VG2 held fixed; record Vb(t).
  5. Fit log(Vb(t) − Vb_eq) vs t → slope = −1/τ_body.
  6. Report τ_body and the implied R_internal = τ / Cbody.

Repeat at three operating points: low-VG2 (subthreshold), mid, and
high-VG2 (above-threshold) to bracket τ_body.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z110_cbody_characterization"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.joint_newton import transient_2t, joint_newton_solve


def run_relaxation(VG1: float, VG2: float, Vd: float, delta: float = 0.005,
                    dt: float = 0.005e-9, T: int = 400):
    """Perturb Vb above equilibrium and watch decay."""
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=50)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 5e4

    VG1_t = torch.tensor(VG1, dtype=torch.float64)
    VG2_t = torch.tensor(VG2, dtype=torch.float64)
    Vd_t_scalar = torch.tensor(Vd, dtype=torch.float64)

    # Step 1: quiescent solve
    qres = joint_newton_solve(cfg, M1, M2, bjt, Vd_t_scalar, VG1_t, VG2_t,
                                Vsint0=0.1, Vb0=0.0, max_iters=50, tol=1e-12, damp=0.7)
    Vb_eq = float(qres["Vb"]); Vsint_eq = float(qres["Vsint"])

    # Step 2: transient with perturbed initial Vb
    t = torch.arange(T, dtype=torch.float64) * dt
    Vd_seq = torch.full((T,), Vd, dtype=torch.float64)
    res = transient_2t(cfg, M1, M2, bjt, Vd_seq, t, VG1_t, VG2_t,
                        Vb0=Vb_eq + delta, Vsint0=Vsint_eq,
                        spike_threshold=10.0,  # disable spiking
                        reset_Vb=Vb_eq,
                        newton_iters=20, newton_tol=1e-11, damp=0.7)
    Vb_traj = res["Vb"].numpy()

    # Step 3: fit log(Vb - Vb_eq) vs t
    # Take absolute decay magnitude in case Vb relaxes from above OR below
    diff = np.abs(Vb_traj - Vb_eq)
    # Use only the part where diff > 0.05 · delta (avoids noise floor)
    mask = (diff > 0.05 * delta) & np.isfinite(diff)
    if mask.sum() < 5:
        return None
    t_fit = t.numpy()[mask]
    log_diff = np.log(diff[mask])
    slope, intercept = np.polyfit(t_fit, log_diff, 1)
    tau_body = -1.0 / slope if slope < 0 else float("inf")
    R_internal = tau_body / cfg.Cbody

    return {
        "VG1": VG1, "VG2": VG2, "Vd": Vd,
        "Vb_eq": Vb_eq, "Vsint_eq": Vsint_eq,
        "Cbody_assumed_F": cfg.Cbody,
        "tau_body_s": float(tau_body),
        "R_internal_ohm": float(R_internal),
        "n_fit_points": int(mask.sum()),
        "Vb_initial_offset": float(delta),
        "Vb_traj_first10": Vb_traj[:10].tolist(),
    }


def main():
    t0 = time.time()
    print(f"[z110] τ_body characterization at 3 operating points")
    op_points = [
        (0.2, 0.1, 1.0, "subthreshold"),
        (0.4, 0.3, 1.0, "near-threshold"),
        (0.6, 0.5, 1.0, "above-threshold"),
    ]
    results = []
    for VG1, VG2, Vd, label in op_points:
        ti = time.time()
        print(f"\n[z110] {label}: VG1={VG1}, VG2={VG2}, Vd={Vd} ...", flush=True)
        try:
            r = run_relaxation(VG1, VG2, Vd)
            if r is None:
                print(f"  FAIL — fit gave too few points")
                continue
            r["label"] = label
            r["wall_s"] = float(time.time() - ti)
            results.append(r)
            print(f"  Vb_eq={r['Vb_eq']:+.4f} V")
            print(f"  τ_body = {r['tau_body_s']*1e9:.3f} ns "
                  f"(C_body assumed {r['Cbody_assumed_F']*1e15:.2f} fF)")
            print(f"  → R_internal = {r['R_internal_ohm']/1e6:.1f} MΩ "
                  f"({r['n_fit_points']} fit points, {r['wall_s']:.1f}s)")
        except Exception as e:
            print(f"  exception: {e}")

    if results:
        taus = [r["tau_body_s"] for r in results if np.isfinite(r["tau_body_s"])]
        Rs = [r["R_internal_ohm"] for r in results if np.isfinite(r["R_internal_ohm"])]
        print(f"\n[z110] === SUMMARY ===")
        print(f"  τ_body range: {min(taus)*1e9:.2f} – {max(taus)*1e9:.2f} ns")
        print(f"  R_internal range: {min(Rs)/1e6:.1f} – {max(Rs)/1e6:.1f} MΩ")
        print(f"  Pazos paper reference: τ_body ≈ 0.7 ns")
    json.dump({"op_points": [list(p) for p in op_points], "results": results,
                "wall_s": float(time.time() - t0)},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z110] wall: {time.time()-t0:.1f}s")
    print(f"[z110] saved {OUT}/summary.json")


if __name__ == "__main__":
    main()
