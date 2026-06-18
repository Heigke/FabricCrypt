"""z229 — R-track step 1: surrogate vs pyport-direct I-V comparison.

Per gap-closing plan, R-track requires verifying the 4D surrogate does
not lie about the operating ridge the reservoir visits. Before running
a full pyport-direct reservoir (heavy), first check static accuracy
on a representative bias set.

If max log10|Id| error < 0.3 dec across 32 random biases the surrogate
faithfully reproduces pyport at reservoir operating points, and the
NRMSE results from z221-z228 are not a surrogate artifact.

CPU-only. ~30s wall.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/z229_pyport_vs_surrogate"; OUT.mkdir(parents=True, exist_ok=True)
SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"


def main():
    sys.path.insert(0, str(ROOT / "nsram"))
    from nsram.bsim4_port.nsram_cell_2T import (
        solve_2t_steady_state, NSRAMCell2TConfig,
    )
    from nsram.bsim4_port.model_card import BSIM4Model
    from nsram.bsim4_port.bjt import GummelPoonNPN
    from scripts.nsram_surrogate_4d import NSRAMSurrogate4D

    # Build pyport at production params (matches z220 surrogate generation)
    import scripts.z91f_validate_with_sebas_params as f
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True)
    DATA = ROOT / "data/sebas_2026_04_22"
    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(text_M1, model_type="nmos")
    model_M2 = BSIM4Model.from_spice(text_M2, model_type="nmos")
    f.patch_model_values(model, type_n=True)
    f.patch_model_values(model_M2, type_n=True)
    bjt = GummelPoonNPN()

    # Reservoir-typical bias range (matches z221+ runtime clip ranges)
    rng = np.random.default_rng(42)
    N = 32
    VG1 = torch.tensor(rng.uniform(0.05, 0.7, N), dtype=torch.float64)
    VG2 = torch.tensor(rng.uniform(0.0, 0.6, N), dtype=torch.float64)
    Vd  = torch.ones(N, dtype=torch.float64)

    print(f"=== z229 R-track: surrogate vs pyport-direct, N={N} biases ===")
    t0 = time.time()
    pyp = solve_2t_steady_state(cfg, model, bjt, Vd, VG1, VG2,
                                  model_M2=model_M2, verbose=False)
    pyport_wall = time.time() - t0
    print(f"pyport wall: {pyport_wall:.2f}s for {N} biases (converged={pyp['converged'].sum().item()}/{N})")

    Id_pyp = pyp["Id"].abs().clamp_min(1e-15).cpu().numpy()
    Vb_pyp = pyp["Vb"].cpu().numpy()

    # Surrogate eval at SAME (VG1, VG2, Vd, Vb_pyp) — fair comparison
    surr = NSRAMSurrogate4D(SURR_PATH)
    log_Id_s, Iii_s, Ileak_s = surr.eval(
        VG1.numpy(), VG2.numpy(), np.ones(N), Vb_pyp,
    )
    Id_surr = 10.0 ** log_Id_s

    log_pyp = np.log10(Id_pyp)
    delta_dec = np.abs(log_pyp - log_Id_s)
    rms_dec = float(np.sqrt((delta_dec ** 2).mean()))
    max_dec = float(delta_dec.max())
    p95 = float(np.quantile(delta_dec, 0.95))

    print(f"\nlog10|Id|  pyport vs surrogate (decade error):")
    print(f"  RMS    : {rms_dec:.3f}")
    print(f"  P95    : {p95:.3f}")
    print(f"  MAX    : {max_dec:.3f}  (gate: <0.30 PASS)")
    print(f"  GATE   : {'✅ PASS' if max_dec < 0.30 else '❌ FAIL'}")

    # Per-bias detail
    print(f"\n  i  VG1   VG2    Vb_pyp  log10|Id|_pyp  log10|Id|_surr  Δdec")
    order = np.argsort(-delta_dec)[:10]
    for k, i in enumerate(order):
        print(f"  {int(i):2d}  {VG1[i]:.3f} {VG2[i]:.3f}  {Vb_pyp[i]:.3f}  "
              f"{log_pyp[i]:>7.3f}  {log_Id_s[i]:>7.3f}  {delta_dec[i]:.3f}")

    out = {
        "N": N, "rms_dec": rms_dec, "p95_dec": p95, "max_dec": max_dec,
        "gate_pass": bool(max_dec < 0.30),
        "pyport_wall_s": pyport_wall,
        "pyport_converged": int(pyp['converged'].sum().item()),
    }
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
