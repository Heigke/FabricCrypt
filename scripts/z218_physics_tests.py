"""z218_physics_tests — Physics edge-case sanity tests for the 2T NS-RAM port.

Exercises the model in extreme regimes to catch fundamental bugs that a
high-quality I-V fit might mask.

Usage:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z218_physics_tests.py

Outputs to: results/z218_physics_edge_cases/test_results.md
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import replace
import math

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "nsram"))

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.dc import compute_dc
from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.nsram_cell_2T import (
    NSRAMCell2TConfig, solve_2t_steady_state,
)
from contextlib import contextmanager

# Match z82's split: which params live in sd.scaled vs as direct attrs.
SCALED_KEYS = {"k1", "k2", "agidl", "bgidl", "cgidl", "egidl",
               "alpha0", "beta0"}
ATTR_KEYS = {"vth0": "vth0_T", "u0": "u0temp", "vsat": "vsattemp"}


@contextmanager
def patch_sd(sd, values: dict):
    saved_scaled = {}
    saved_attr = {}
    try:
        for name, val in values.items():
            if name in SCALED_KEYS:
                saved_scaled[name] = sd.scaled.get(name, None)
                sd.scaled[name] = val
            elif name in ATTR_KEYS:
                attr = ATTR_KEYS[name]
                saved_attr[attr] = getattr(sd, attr)
                setattr(sd, attr, val)
        yield
    finally:
        for k, v in saved_scaled.items():
            if v is None:
                sd.scaled.pop(k, None)
            else:
                sd.scaled[k] = v
        for k, v in saved_attr.items():
            setattr(sd, k, v)


SEBAS_CARD = REPO / "data" / "sebas_2026_04_22" / "PTM130bulkNSRAM.txt"
OUT_DIR = REPO / "results" / "z218_physics_edge_cases"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def t(x):
    return torch.tensor([x], dtype=torch.float64)


def solve(cfg, model, bjt, Vd, VG1, VG2, P_M1=None, P_M2=None,
          Vb_init=None, Vsint_init=None):
    out = solve_2t_steady_state(
        cfg, model, bjt,
        Vd=t(Vd), VG1=t(VG1), VG2=t(VG2),
        P_M1=P_M1, P_M2=P_M2,
        Vb_init=Vb_init, Vsint_init=Vsint_init,
    )
    return out


def main():
    print("=== z218: Physics edge-case tests for 2T NS-RAM port ===")
    print(f"Card: {SEBAS_CARD}\n")

    model = BSIM4Model.from_spice(SEBAS_CARD.read_text(), model_type="nmos")
    bjt = GummelPoonNPN.from_sebas_card()
    cfg = NSRAMCell2TConfig(
        Ln=180e-9, Wn=360e-9, M2_length_factor=10.0, T_C=27.0,
        use_iii=True, use_gidl=True, use_bjt=True, use_igb=True, use_diode=True,
        newton_max_iters=50,
    )

    results = []  # list of (name, status, detail)

    # -------------------- (A) OFF-state floor --------------------
    print("\n[A] OFF-state floor: VG1=0.0, VG2=-1.0, Vd=0.5")
    out = solve(cfg, model, bjt, Vd=0.5, VG1=0.0, VG2=-1.0)
    Id = float(out["Id"])
    print(f"    Id = {Id:.3e} A")
    if abs(Id) < 1e-12:
        status = "PASS"
    elif abs(Id) > 1e-9:
        status = "FAIL"
    else:
        status = "WARN"
    results.append(("A_off_state_floor", status,
                    f"Id={Id:.3e} A (req <1e-12 PASS, >1e-9 FAIL)"))

    # -------------------- (B) VG1 monotonicity --------------------
    print("\n[B] VG1 monotonicity: VG2=0.0, Vd=1.0, VG1 in [0.0,1.5]")
    VG1_grid = [round(0.0 + i * 0.1, 3) for i in range(16)]  # 0.0..1.5 step 0.1
    Ids_B = []
    for v in VG1_grid:
        out = solve(cfg, model, bjt, Vd=1.0, VG1=v, VG2=0.0)
        Ids_B.append(float(out["Id"]))
        print(f"    VG1={v:+.2f}  Id={Ids_B[-1]:+.3e}")
    # Ignore subthreshold floor (Id < 1e-13 A) where numerical noise dominates.
    floor_B = 1e-13
    max_reversal = 0.0
    worst_pair = None
    for i in range(1, len(Ids_B)):
        prev, cur = Ids_B[i - 1], Ids_B[i]
        if prev < floor_B or cur < floor_B:
            continue
        if cur < prev:
            rel = (prev - cur) / abs(prev)
            if rel > max_reversal:
                max_reversal = rel
                worst_pair = (VG1_grid[i - 1], VG1_grid[i], prev, cur)
    status = "PASS" if max_reversal <= 0.05 else "FAIL"
    detail = f"max reversal={max_reversal*100:.2f}% (limit 5%, ignoring Id<{floor_B:.0e})"
    if worst_pair:
        detail += f" worst@VG1 {worst_pair[0]}->{worst_pair[1]}: {worst_pair[2]:.3e}->{worst_pair[3]:.3e}"
    results.append(("B_VG1_monotonic", status, detail))

    # -------------------- (C) VG2 monotonicity --------------------
    print("\n[C] VG2 monotonicity: VG1=0.6, Vd=1.0, VG2 in [-0.4,+0.4]")
    VG2_grid = [round(-0.4 + i * 0.1, 3) for i in range(9)]
    Ids_C = []
    for v in VG2_grid:
        out = solve(cfg, model, bjt, Vd=1.0, VG1=0.6, VG2=v)
        Ids_C.append(float(out["Id"]))
        print(f"    VG2={v:+.2f}  Id={Ids_C[-1]:+.3e}")
    floor_C = 1e-13
    max_reversal = 0.0
    worst_pair = None
    for i in range(1, len(Ids_C)):
        prev, cur = Ids_C[i - 1], Ids_C[i]
        if prev < floor_C or cur < floor_C:
            continue
        if cur < prev:
            rel = (prev - cur) / abs(prev)
            if rel > max_reversal:
                max_reversal = rel
                worst_pair = (VG2_grid[i - 1], VG2_grid[i], prev, cur)
    status = "PASS" if max_reversal <= 0.05 else "FAIL"
    detail = f"max reversal={max_reversal*100:.2f}% (limit 5%, ignoring Id<{floor_C:.0e})"
    if worst_pair:
        detail += f" worst@VG2 {worst_pair[0]}->{worst_pair[1]}: {worst_pair[2]:.3e}->{worst_pair[3]:.3e}"
    results.append(("C_VG2_monotonic", status, detail))

    # -------------------- (D) VG2 polarity --------------------
    print("\n[D] VG2 polarity: VG1=0.4, Vd=1.0, VG2={-0.3,+0.3}")
    out_lo = solve(cfg, model, bjt, Vd=1.0, VG1=0.4, VG2=-0.3)
    out_hi = solve(cfg, model, bjt, Vd=1.0, VG1=0.4, VG2=+0.3)
    Id_lo = float(out_lo["Id"])
    Id_hi = float(out_hi["Id"])
    ratio = Id_hi / max(abs(Id_lo), 1e-30)
    print(f"    Id(VG2=-0.3) = {Id_lo:.3e}")
    print(f"    Id(VG2=+0.3) = {Id_hi:.3e}")
    print(f"    ratio        = {ratio:.2f}")
    if ratio > 100:
        status = "PASS"
    elif ratio < 10:
        status = "FAIL"
    else:
        status = "WARN"
    results.append(("D_VG2_polarity", status,
                    f"Id_hi/Id_lo={ratio:.2f} (req >100 PASS, <10 FAIL)"))

    # -------------------- (E) Body-effect direction --------------------
    # Use compute_dc on M1 alone with forced Vbs to test physics direction.
    # Bias mapping for M1 standalone with grounded source:
    #   Vgs = VG1 = 0.4, Vds = Vd = 1.0, Vbs = Vb (forced)
    print("\n[E] Body-effect direction: M1 standalone, VG1=0.4, Vd=1.0, Vb in {-0.5,+0.3}")
    geom_M1 = Geometry(L=cfg.Ln, W=cfg.Wn, NF=1)
    sd_M1 = compute_size_dep(model, geom_M1, T_C=cfg.T_C)
    dc_rev = compute_dc(model, sd_M1, Vgs=t(0.4), Vds=t(1.0), Vbs=t(-0.5))
    dc_fwd = compute_dc(model, sd_M1, Vgs=t(0.4), Vds=t(1.0), Vbs=t(+0.3))
    Id_rev = float(dc_rev.Ids)
    Id_fwd = float(dc_fwd.Ids)
    print(f"    Id(Vbs=-0.5, reverse) = {Id_rev:.3e}")
    print(f"    Id(Vbs=+0.3, forward) = {Id_fwd:.3e}")
    if Id_fwd > Id_rev and Id_rev > 0:
        status = "PASS"
    else:
        status = "FAIL"
    results.append(("E_body_effect_direction", status,
                    f"Id(Vbs=-0.5)={Id_rev:.3e}, Id(Vbs=+0.3)={Id_fwd:.3e}; "
                    f"forward>reverse required"))

    # -------------------- (F) Snapback knee --------------------
    print("\n[F] Snapback search: VG1=0.4, VG2=0.0, Vd 0.5..2.5 in 50 steps")
    print("    overrides: alpha0=2e-2, beta0=15, BJT.Bf=10000")
    bjt_hot = GummelPoonNPN.from_sebas_card()
    bjt_hot.Bf = 10000.0
    sd_M1_F = cfg.size_dep_M1(model)
    sd_M2_F = cfg.size_dep_M2(model)
    overrides_F = {"alpha0": 2e-2, "beta0": 15.0}
    Vd_grid = [0.5 + i * (2.5 - 0.5) / 49 for i in range(50)]
    Ids_F = []
    Vsint_warm = None
    Vb_warm = torch.tensor(0.5, dtype=torch.float64)
    with patch_sd(sd_M1_F, overrides_F), patch_sd(sd_M2_F, overrides_F):
        for v in Vd_grid:
            try:
                out = solve_2t_steady_state(
                    cfg, model, bjt_hot,
                    Vd=t(v), VG1=t(0.4), VG2=t(0.0),
                    Vsint_init=Vsint_warm,
                    Vb_init=Vb_warm.expand_as(t(v)),
                )
                Ids_F.append(float(out["Id"]))
                Vsint_warm = out["Vsint"].detach().squeeze(0)
                Vb_warm = out["Vb"].detach().squeeze(0)
            except Exception as e:
                Ids_F.append(float("nan"))
                print(f"    Vd={v:.3f}: solve raised {type(e).__name__}: {e}")
    knee = None
    knee_ratio = 0.0
    # Look for any window where Id jumps >10x within ~0.05V (≈1 grid step)
    for i in range(1, len(Ids_F)):
        a = Ids_F[i - 1]; b = Ids_F[i]
        if a is None or b is None or math.isnan(a) or math.isnan(b):
            continue
        if a <= 0 or b <= 0:
            continue
        r = b / a
        if r > 10 and r > knee_ratio:
            knee_ratio = r
            knee = (Vd_grid[i - 1], Vd_grid[i], a, b)
    print(f"    Id range: {min((x for x in Ids_F if not math.isnan(x)), default=float('nan')):.3e} ..."
          f" {max((x for x in Ids_F if not math.isnan(x)), default=float('nan')):.3e}")
    if knee is not None:
        status = "PASS"
        detail = (f"knee at Vd~{knee[0]:.3f}->{knee[1]:.3f}: "
                  f"Id {knee[2]:.3e}->{knee[3]:.3e} (ratio {knee_ratio:.1f}x)")
    else:
        status = "FAIL"
        detail = "no >10x jump found anywhere in [0.5, 2.5] V"
    print(f"    {detail}")
    results.append(("F_snapback_knee", status, detail))

    # -------------------- (G) Gradient flow through Newton --------------------
    # Bias: VG1=0.6, VG2=0.6 (both on), Vd=1.5 (high Vds-Vdseff drives Iii).
    # Force alpha0 large enough that impact-ion is meaningfully active.
    print("\n[G] Gradient flow: dId/dvth0, dId/dalpha0, dId/dbeta0")
    print("    bias: VG1=0.6, VG2=0.6, Vd=1.5; alpha0/beta0 baselines forced for active Iii")
    sd_g = compute_size_dep(model, Geometry(L=cfg.Ln, W=cfg.Wn, NF=1), T_C=cfg.T_C)
    base_vth0 = float(sd_g.scaled.get("vth0", model.get("vth0", 0.5)))
    base_alpha0 = 1e-2   # force into impact-ion-active regime
    base_beta0 = 15.0

    sd_M1_g = cfg.size_dep_M1(model)
    sd_M2_g = cfg.size_dep_M2(model)
    grads = {}
    for name, base in [("vth0", base_vth0),
                       ("alpha0", base_alpha0),
                       ("beta0", base_beta0)]:
        p = torch.tensor(base, dtype=torch.float64, requires_grad=True)
        ov = {name: p}
        try:
            with patch_sd(sd_M1_g, ov), patch_sd(sd_M2_g, ov):
                out = solve_2t_steady_state(
                    cfg, model, bjt,
                    Vd=t(1.5), VG1=t(0.6), VG2=t(0.6),
                )
                Id = out["Id"].sum()
            g, = torch.autograd.grad(Id, p, retain_graph=False, allow_unused=False)
            gv = float(g)
        except Exception as e:
            gv = float("nan")
            print(f"    grad {name}: raised {type(e).__name__}: {e}")
        grads[name] = gv
        print(f"    base {name}={base:.4g} -> dId/d{name} = {gv:.3e}")

    # Expected signs
    sign_ok = {
        "vth0": grads["vth0"] < 0,        # higher Vth -> lower Id
        "alpha0": grads["alpha0"] > 0,    # more impact-ion coeff -> more Iii -> Vb up -> Id up (positive)
        "beta0": grads["beta0"] < 0,      # exp(-beta0/X): larger beta0 -> less Iii -> lower Id
    }
    finite_ok = all((not math.isnan(v)) and math.isfinite(v) for v in grads.values())
    nonzero_ok = all(abs(v) > 0 for v in grads.values() if math.isfinite(v))
    if finite_ok and nonzero_ok and all(sign_ok.values()):
        status = "PASS"
    else:
        status = "FAIL"
    detail = (f"finite={finite_ok}, nonzero={nonzero_ok}, "
              f"sign(vth0<0)={sign_ok['vth0']}, sign(alpha0>0)={sign_ok['alpha0']}, "
              f"sign(beta0<0)={sign_ok['beta0']}; grads={grads}")
    results.append(("G_gradient_flow", status, detail))

    # -------------------- (H) M1 vs M2 length scaling --------------------
    # Same Vgs, Vds, Vbs=0; M1 (L=180n) vs M2 (L=1.8u)
    print("\n[H] M1 vs M2 standalone: same Vgs=0.6, Vds=1.0, Vbs=0")
    geom_M2 = Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn, NF=1)
    sd_M2 = compute_size_dep(model, geom_M2, T_C=cfg.T_C)
    dc1 = compute_dc(model, sd_M1, Vgs=t(0.6), Vds=t(1.0), Vbs=t(0.0))
    dc2 = compute_dc(model, sd_M2, Vgs=t(0.6), Vds=t(1.0), Vbs=t(0.0))
    I1 = float(dc1.Ids); I2 = float(dc2.Ids)
    ratio = I1 / max(I2, 1e-30)
    print(f"    Ids_M1 (L=180n) = {I1:.3e}")
    print(f"    Ids_M2 (L=1.8u) = {I2:.3e}")
    print(f"    ratio M1/M2     = {ratio:.2f}  (expected ~10x)")
    # Long-channel I ~ W/L; ratio of Ls is 10. Allow 5..30x window.
    if 5.0 <= ratio <= 30.0:
        status = "PASS"
    elif ratio > 1.0:
        status = "WARN"
    else:
        status = "FAIL"
    results.append(("H_length_scaling", status,
                    f"Ids_M1/Ids_M2={ratio:.2f} (expect ~10x, accept 5-30x)"))

    # -------------------- Summary --------------------
    print("\n=== SUMMARY ===")
    for name, status, detail in results:
        print(f"[{status}] {name}: {detail}")

    # Write markdown
    md_path = OUT_DIR / "test_results.md"
    lines = ["# z218 Physics Edge-Case Tests — 2T NS-RAM Port", ""]
    lines.append(f"Card: `{SEBAS_CARD.relative_to(REPO)}`  ")
    lines.append(f"Port: `nsram.bsim4_port.nsram_cell_2T`  ")
    lines.append("")
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    n_warn = sum(1 for _, s, _ in results if s == "WARN")
    lines.append(f"**Result: {n_pass} PASS / {n_fail} FAIL / {n_warn} WARN ({len(results)} total)**")
    lines.append("")
    lines.append("| Test | Status | Detail |")
    lines.append("|------|--------|--------|")
    for name, status, detail in results:
        lines.append(f"| {name} | **{status}** | {detail} |")
    lines.append("")
    lines.append("## Bias notes")
    lines.append(f"- (B) VG1 sweep raw Ids: {[f'{x:.2e}' for x in Ids_B]}")
    lines.append(f"- (C) VG2 sweep raw Ids: {[f'{x:.2e}' for x in Ids_C]}")
    lines.append(f"- (F) snapback Id min/max: {min((x for x in Ids_F if not math.isnan(x)), default=float('nan')):.3e} .. "
                 f"{max((x for x in Ids_F if not math.isnan(x)), default=float('nan')):.3e}")
    lines.append(f"- (G) gradients: {grads}")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main()
