"""Smoke test for the quasi-2D body solver (Plan A wrapper).

Three checks:
  1. Stiff-coupling regression: Rb_SD = 1Ω forces Vb_S ≈ Vb_D ≈ Vb_lumped
     and Id matches lumped fit to <1e-6 relative.
  2. Default-coupling perturbation: Rb_SD = 1e6Ω, alpha = 0.7. Reports the
     Vb_S vs Vb_D asymmetry and Id delta vs lumped at a snapback bias.
  3. 5x5 (Rb_SD, alpha) sweep on one bias: prints residual heatmap to
     guide whether Plan B refactor is worth the time.

Single-bias tests; runs in ~5 seconds. Thread caps applied.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "4")
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nsram"))

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import (
    NSRAMCell2TConfig,
    solve_2t_steady_state,
    solve_2t_quasi2d_steady_state,
)

SEBAS_M1 = ROOT / "data/sebas_2026_04_22/M1_130DNWFB.txt"
SEBAS_M2 = ROOT / "data/sebas_2026_04_22/M2_130bulkNSRAM.txt"


def _t(x):
    return torch.tensor(x, dtype=torch.float64)


def _build():
    model_M1 = BSIM4Model.from_spice(SEBAS_M1.read_text(), model_type="nmos")
    model_M2 = BSIM4Model.from_spice(SEBAS_M2.read_text(), model_type="nmos")
    bjt = GummelPoonNPN.from_sebas_card()
    return model_M1, model_M2, bjt


def smoke():
    model_M1, model_M2, bjt = _build()
    # Benign bias well away from the snapback fold (single-root regime).
    # VG1=0.6 strong-on, VG2=0.5 (M2 well above Vth), Vd=0.5 linear region.
    Vd = _t(0.5); VG1 = _t(0.6); VG2 = _t(0.5)

    print("=" * 68)
    print(f"smoke @ Vd={float(Vd):.2f} VG1={float(VG1):.2f} VG2={float(VG2):.2f}")
    print("=" * 68)

    cfg_lumped = NSRAMCell2TConfig()
    out_l = solve_2t_steady_state(
        cfg_lumped, model_M1, bjt, Vd, VG1, VG2, model_M2=model_M2)
    print(f"\n[lumped]   Id={float(out_l['Id']):.6e}  Vsint={float(out_l['Vsint']):.4f}  Vb={float(out_l['Vb']):.4f}")

    # ── Test 1: stiff coupling -> should match lumped ──────────────────
    cfg_stiff = NSRAMCell2TConfig(quasi2d_body=True, Rb_SD=1.0, iii_split_alpha=0.5)
    out_s = solve_2t_quasi2d_steady_state(
        cfg_stiff, model_M1, bjt, Vd, VG1, VG2, model_M2=model_M2)
    Id_s = float(out_s["Id"]); Vbs = float(out_s["Vb_S"]); Vbd = float(out_s["Vb_D"])
    rel = abs(Id_s - float(out_l["Id"])) / max(abs(float(out_l["Id"])), 1e-15)
    print(f"\n[stiff Rb=1Ω alpha=0.5]")
    print(f"   Id={Id_s:.6e}  Vb_S={Vbs:.4f}  Vb_D={Vbd:.4f}  asym={abs(Vbs-Vbd):.2e}")
    print(f"   relative Id error vs lumped: {rel:.3e}  -> {'PASS' if rel < 1e-3 else 'FAIL'}")

    # ── Test 2: default coupling -> report asymmetry ──────────────────
    cfg_dflt = NSRAMCell2TConfig(quasi2d_body=True, Rb_SD=1e6, iii_split_alpha=0.7)
    out_d = solve_2t_quasi2d_steady_state(
        cfg_dflt, model_M1, bjt, Vd, VG1, VG2, model_M2=model_M2)
    Id_d = float(out_d["Id"]); Vbs = float(out_d["Vb_S"]); Vbd = float(out_d["Vb_D"])
    delta = (Id_d - float(out_l["Id"])) / max(abs(float(out_l["Id"])), 1e-15)
    print(f"\n[default Rb=1e6Ω alpha=0.7]")
    print(f"   Id={Id_d:.6e}  Vb_S={Vbs:.4f}  Vb_D={Vbd:.4f}  asym={Vbd-Vbs:+.4f} V")
    print(f"   relative Id delta vs lumped: {delta:+.3e}")

    # ── Test 3: 5x5 sweep -> see if anything moves ─────────────────────
    print("\n[5x5 sweep over (Rb_SD, alpha) at this bias]")
    print(f"   columns = alpha in [0.3, 0.5, 0.7, 0.9, 1.0]")
    print(f"   rows    = Rb_SD in {{1e3, 1e5, 1e6, 1e7, 1e9}}")
    print(f"   value   = (Id - Id_lumped) / Id_lumped, log10|·| (negative=lower)")
    rb_grid = [1e3, 1e5, 1e6, 1e7, 1e9]
    a_grid = [0.3, 0.5, 0.7, 0.9, 1.0]
    Id_l = float(out_l['Id'])
    print(f"\n   Rb_SD\\α    " + "  ".join([f"{a:.1f}" for a in a_grid]))
    for rb in rb_grid:
        row = []
        for a in a_grid:
            cfg = NSRAMCell2TConfig(quasi2d_body=True, Rb_SD=rb, iii_split_alpha=a)
            try:
                o = solve_2t_quasi2d_steady_state(
                    cfg, model_M1, bjt, Vd, VG1, VG2, model_M2=model_M2)
                d = (float(o['Id']) - Id_l) / max(abs(Id_l), 1e-15)
                if abs(d) < 1e-12:
                    row.append("  0   ")
                else:
                    sign = "-" if d < 0 else "+"
                    row.append(f"{sign}{abs(d):5.0e}")
            except Exception as e:
                row.append("  err  ")
        print(f"   {rb:8.0e}   " + "  ".join(row))

    print("\nsmoke test done.")


if __name__ == "__main__":
    smoke()
