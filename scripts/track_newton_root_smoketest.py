#!/usr/bin/env python3
"""Smoke test for newton_root_seed='high_Vb_continuation'.

Single bias VG1=0.6, VG2=0.0, Vd ∈ [0.05..2.0]. Compare Vb at high Vd
between zero-seed and high_Vb_continuation. Forensic predicts:
  zero-seed → Vb ≈ 0.27 V (spurious low root)
  high_Vb_continuation → Vb ≈ 0.95 V (physical root, matching ngspice)
"""
from __future__ import annotations
import os, sys, json
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

from pathlib import Path
import numpy as np, torch

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t
from nsram.bsim4_port.bjt import GummelPoonNPN


def build_cfg(seed_mode: str):
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    cfg.body_pdiode_to = "vnwell"
    cfg.use_well_diode = True
    cfg.vnwell = 2.0
    cfg.body_pdiode_Js = 5.3675e-7 / 22e-12
    cfg.body_pdiode_n = 1.0535
    cfg.body_pdiode_Rs = 1.0e6
    cfg.hurkx_bbt_A = 0.0  # Hurkx OFF
    cfg.newton_root_seed = seed_mode
    cfg.newton_root_seed_n_steps = 20
    cfg.newton_root_seed_vb_init = 0.95
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 9000.0; bjt.Va = 0.55; bjt.Is = 1e-9
    return cfg, M1, M2, bjt


def run(seed_mode, VG1=0.6, VG2=0.0):
    cfg, M1, M2, bjt = build_cfg(seed_mode)
    # K1+ALPHA0 overrides (LALPHA0_FIX cards equivalent — apply at compute level via sd.scaled)
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    saved = {}
    for k, v in {"k1": 0.53825, "alpha0": 7.83756e-4}.items():
        saved[("M1", k)] = sd_M1.scaled.get(k); sd_M1.scaled[k] = float(v)
    for k, v in {"k1": 0.63825, "k2": -0.070435, "etab": -0.086777,
                 "beta0": 18.0, "alpha0": 7.83756e-4}.items():
        saved[("M2", k)] = sd_M2.scaled.get(k); sd_M2.scaled[k] = float(v)
    try:
        Vd = torch.linspace(0.05, 2.0, 40, dtype=torch.float64)
        out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd,
                         VG1=torch.tensor(VG1, dtype=torch.float64),
                         VG2=torch.tensor(VG2, dtype=torch.float64),
                         warm_start=True)
        Id = out["Id"].detach().cpu().numpy()
        Vb = out["Vb"].detach().cpu().numpy()
        Vsint = out["Vsint"].detach().cpu().numpy()
    finally:
        for (which, k), v in saved.items():
            sd = sd_M1 if which == "M1" else sd_M2
            if v is None: sd.scaled.pop(k, None)
            else: sd.scaled[k] = v
    return Vd.numpy(), Id, Vb, Vsint


def main():
    print("=== Newton root-seed smoke test (VG1=0.6, VG2=0.0) ===")
    print(f"  Forensic prediction:")
    print(f"    zero-seed → Vb ≈ 0.27V, Id ≈ 4e-11 A at Vd=2V (spurious low root)")
    print(f"    high_Vb_continuation → Vb ≈ 0.95V, Id ≈ 2e-6 A (physical root)")
    print()

    Vd_z, Id_z, Vb_z, Vs_z = run("zero")
    Vd_c, Id_c, Vb_c, Vs_c = run("high_Vb_continuation")

    # Sample at Vd=2V (last point)
    print(f"{'Vd':>6} | {'Id(zero)':>12} {'Vb(zero)':>10} | {'Id(cont)':>12} {'Vb(cont)':>10} | Δlog10(Id)")
    print("-" * 90)
    for i in (0, 10, 20, 30, 39):
        d_log = float(np.log10(max(abs(Id_c[i]), 1e-30)) - np.log10(max(abs(Id_z[i]), 1e-30)))
        print(f"{Vd_z[i]:6.3f} | {abs(Id_z[i]):12.3e} {Vb_z[i]:10.4f} | "
              f"{abs(Id_c[i]):12.3e} {Vb_c[i]:10.4f} | {d_log:+6.3f}")
    print()

    # PASS check
    Vb_c_high = float(Vb_c[-1])
    Vb_z_high = float(Vb_z[-1])
    Id_ratio = float(np.log10(max(abs(Id_c[-1]), 1e-30) / max(abs(Id_z[-1]), 1e-30)))
    print(f"At Vd=2V:")
    print(f"  Vb: zero={Vb_z_high:.4f}V  cont={Vb_c_high:.4f}V  (Δ={Vb_c_high-Vb_z_high:+.4f}V)")
    print(f"  Δlog10(Id) = {Id_ratio:+.3f} dec")
    if Vb_c_high > 0.7 and Id_ratio > 2.0:
        print(f"  SMOKE TEST PASS: continuation finds high-Vb root")
    else:
        print(f"  SMOKE TEST FAIL: continuation did NOT find high-Vb root")

    out = {
        "Vd": Vd_z.tolist(),
        "zero": {"Id": Id_z.tolist(), "Vb": Vb_z.tolist(), "Vsint": Vs_z.tolist()},
        "high_Vb_continuation": {"Id": Id_c.tolist(), "Vb": Vb_c.tolist(), "Vsint": Vs_c.tolist()},
        "summary": {"Vb_zero@2V": Vb_z_high, "Vb_cont@2V": Vb_c_high,
                    "dlog10_Id@2V": Id_ratio,
                    "PASS": (Vb_c_high > 0.7 and Id_ratio > 2.0)},
    }
    OUT = ROOT / "results/track_newton_root_fix"
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "smoketest.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT/'smoketest.json'}")


if __name__ == "__main__":
    main()
