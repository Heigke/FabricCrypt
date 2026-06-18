"""A1p — Backward-Vd sweep test for hysteresis at the worst failing bias.

Tests Gemini's bistability hypothesis at VG1=0.6, VG2=0.0 by running TWO
Vd sweeps:
  1) FORWARD : Vd 0.05 -> 2.0 V, warm-start from cold (Vsint=0, Vb=0)
  2) BACKWARD: Vd 2.0 -> 0.05 V, warm-start from high-current state
                (Vsint=0.5, Vb=0.85) to hunt the high-Vb branch.

If Id_forward(Vd) == Id_backward(Vd) for all Vd  -> no bistability
If they differ by > 0.5 dec at any Vd            -> bistability confirmed

Run:
    cd /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
    source venv/bin/activate
    python research_plan/artifacts/A1p_backward_sweep.py
"""
from __future__ import annotations
import sys, json, importlib.util
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import (
    NSRAMCell2TConfig, solve_2t_with_homotopy,
)
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

spec = importlib.util.spec_from_file_location(
    "z91f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = importlib.util.module_from_spec(spec); spec.loader.exec_module(z91f)

DATA = ROOT / "data/sebas_2026_04_22"

# Bias under test (worst failing in z91g)
VG1_VAL = 0.6
VG2_VAL = 0.0

# Sebas per-bias overrides for VG1=0.6/VG2=0.0
SEBAS = dict(ETAB=2.5, K1=0.41825, ALPHA0=7.842e-5, BETA0=20.0,
             NFACTOR=6.0, mbjt=1.0, IS=5e-9, area=1e-6)

# Vd grid (forward order)
VD_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.3,
           1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]


def build():
    m1_text = (DATA / "M1_130DNWFB.txt").read_text()
    model_M1 = BSIM4Model.from_spice(m1_text, model_type="nmos")
    z91f.patch_model_values(model_M1, type_n=True)
    m2_text = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model_M2 = BSIM4Model.from_spice(m2_text, model_type="nmos")
    z91f.patch_model_values(model_M2, type_n=True)

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            use_diode=True, use_igb=True,
                            newton_max_iters=80, gmin_step=True)
    sd_M1 = compute_size_dep(model_M1, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model_M2,
                             Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn),
                             T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2

    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Is = SEBAS["IS"]
    bjt.area = SEBAS["area"] * SEBAS["mbjt"]

    P_M1 = {
        "etab":   torch.tensor(SEBAS["ETAB"]),
        "k1":     torch.tensor(SEBAS["K1"]),
        "alpha0": torch.tensor(SEBAS["ALPHA0"]),
        "beta0":  torch.tensor(SEBAS["BETA0"]),
    }
    P_M2 = {
        "nfactor": torch.tensor(SEBAS["NFACTOR"]),
        "alpha0":  torch.tensor(SEBAS["ALPHA0"]),
        "beta0":   torch.tensor(SEBAS["BETA0"]),
    }
    return cfg, model_M1, model_M2, sd_M1, sd_M2, bjt, P_M1, P_M2


def run_sweep(direction: str, vd_list, vsint0: float, vb0: float):
    """Sweep Vd across vd_list, warm-starting between points."""
    cfg, model_M1, model_M2, sd_M1, sd_M2, bjt, P_M1, P_M2 = build()
    VG1 = torch.tensor(VG1_VAL, dtype=torch.float64)
    VG2 = torch.tensor(VG2_VAL, dtype=torch.float64)

    Vsint_warm = torch.tensor([vsint0], dtype=torch.float64)
    Vb_warm = torch.tensor([vb0], dtype=torch.float64)

    records = []
    print(f"\n=== {direction} sweep "
          f"(init Vsint={vsint0}, Vb={vb0}) ===")
    print(f"{'Vd':>8} {'Vsint':>10} {'Vb':>10} {'Id':>12} {'conv':>5}")
    with torch.no_grad(), \
         z91f.patch_sd_scaled(sd_M1, P_M1), \
         z91f.patch_sd_scaled(sd_M2, P_M2):
        for vd in vd_list:
            Vd_t = torch.tensor([vd], dtype=torch.float64)
            try:
                out = solve_2t_with_homotopy(
                    cfg, model=model_M1, bjt=bjt,
                    Vd=Vd_t, VG1=VG1, VG2=VG2,
                    P_M1=None, P_M2=None,
                    Vsint_init=Vsint_warm, Vb_init=Vb_warm,
                    model_M2=model_M2,
                )
                Vsint = out["Vsint"].detach()
                Vb    = out["Vb"].detach()
                Id    = out["Id"].detach()
                conv  = bool(out["converged"].all())
                rec = dict(Vd=float(vd),
                           Vsint=float(Vsint.item()),
                           Vb=float(Vb.item()),
                           Id=float(Id.item()),
                           converged=conv)
                # Warm-start next point
                Vsint_warm = Vsint
                Vb_warm = Vb
            except Exception as e:
                rec = dict(Vd=float(vd), status=f"FAIL:{type(e).__name__}:{e}",
                           converged=False, Id=float("nan"),
                           Vsint=float("nan"), Vb=float("nan"))
            records.append(rec)
            print(f"{rec['Vd']:>8.3f} {rec['Vsint']:>+10.4f} "
                  f"{rec['Vb']:>+10.4f} {rec['Id']:>12.3e} "
                  f"{str(rec['converged']):>5}")
    return records


def main():
    out_dir = Path(__file__).parent

    # 1) Forward sweep: low -> high, cold init
    fwd_grid = list(VD_GRID)
    fwd = run_sweep("FORWARD (Vd low->high)", fwd_grid,
                    vsint0=0.0, vb0=0.0)

    # 2) Backward sweep: high -> low, hot init
    bwd_grid = list(reversed(VD_GRID))
    bwd = run_sweep("BACKWARD (Vd high->low)", bwd_grid,
                    vsint0=0.5, vb0=0.85)

    # Index backward by Vd to align with forward's order
    bwd_by_vd = {round(r["Vd"], 6): r for r in bwd}

    # Build aligned arrays (forward Vd order)
    rows = []
    for fr in fwd:
        vd = fr["Vd"]
        br = bwd_by_vd[round(vd, 6)]
        rows.append(dict(
            Vd=vd,
            Id_forward=fr["Id"],  Id_backward=br["Id"],
            Vsint_forward=fr["Vsint"], Vsint_backward=br["Vsint"],
            Vb_forward=fr["Vb"],  Vb_backward=br["Vb"],
            conv_forward=fr.get("converged", False),
            conv_backward=br.get("converged", False),
        ))

    # Diagnostic: max log10 |Id_b/Id_f|
    def safe_log10(x):
        return float(np.log10(max(abs(x), 1e-30)))
    diffs_dec = []
    for r in rows:
        d = abs(safe_log10(r["Id_forward"]) - safe_log10(r["Id_backward"]))
        diffs_dec.append(d)
        r["abs_diff_decades"] = d
    max_diff = max(diffs_dec)

    if max_diff < 0.01:
        verdict = "NO_BISTABILITY"
    elif max_diff > 0.5:
        verdict = "BISTABILITY_CONFIRMED"
    else:
        verdict = "INTERMEDIATE"

    # ---- JSON ----
    out_json = out_dir / "A1p_results.json"
    out_json.write_text(json.dumps({
        "bias": {"VG1": VG1_VAL, "VG2": VG2_VAL},
        "sebas": SEBAS,
        "vd_grid": VD_GRID,
        "forward_sweep": fwd,
        "backward_sweep": bwd,
        "aligned": rows,
        "max_abs_diff_decades": max_diff,
        "verdict": verdict,
    }, indent=2))

    # ---- Plot ----
    Vds = [r["Vd"] for r in rows]
    Idf = [max(abs(r["Id_forward"]), 1e-30) for r in rows]
    Idb = [max(abs(r["Id_backward"]), 1e-30) for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.semilogy(Vds, Idf, "o-", label="forward (cold init)", color="C0")
    ax1.semilogy(Vds, Idb, "s--", label="backward (hot init)", color="C3")
    ax1.set_xlabel("Vd (V)"); ax1.set_ylabel("|Id| (A)")
    ax1.set_title(f"VG1={VG1_VAL}, VG2={VG2_VAL}  —  {verdict}\n"
                  f"max |ΔlogId| = {max_diff:.3f} decades")
    ax1.legend(); ax1.grid(True, which="both", alpha=0.3)

    Vbf = [r["Vb_forward"]  for r in rows]
    Vbb = [r["Vb_backward"] for r in rows]
    ax2.plot(Vds, Vbf, "o-", label="forward Vb",  color="C0")
    ax2.plot(Vds, Vbb, "s--", label="backward Vb", color="C3")
    ax2.set_xlabel("Vd (V)"); ax2.set_ylabel("Vb (V)")
    ax2.set_title("Body voltage Vb across sweep")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = out_dir / "A1p_backward_sweep.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    # ---- Markdown ----
    md = []
    md.append("# A1p — Backward-Vd Sweep Hysteresis Test\n")
    md.append(f"**Bias:** VG1={VG1_VAL} V, VG2={VG2_VAL} V "
              f"(worst-failing point in z91g).\n")
    md.append("**Sebas overrides:** ETAB=2.5, K1=0.41825, "
              "ALPHA0=7.842e-5, BETA0=20, NFACTOR=6.0, mbjt=1, IS=5e-9.\n")
    md.append("**Physics:** post-A.1.j/A.1.o, emitter=GND + vnwell=2 V, "
              "vnwell_Rs=1e9 (cfg defaults).\n")
    md.append("\n## Procedure\n")
    md.append("- **Forward:** Vd 0.05 → 2.0 V, init Vsint=0, Vb=0; "
              "warm-start each point from previous solve.\n")
    md.append("- **Backward:** Vd 2.0 → 0.05 V, init Vsint=0.5, Vb=0.85 "
              "(hot, biased into the impact-ion / high-Vb branch); "
              "warm-start each point from previous solve.\n")
    md.append("- Both use `solve_2t_with_homotopy` (gmin schedule "
              "1e-3→1e-5→1e-8→1e-12→target).\n")
    md.append("\n## Result\n")
    md.append(f"- max |Δlog10 Id| across sweep = **{max_diff:.4f} decades**\n")
    md.append(f"- Verdict: **{verdict}**\n\n")
    md.append("| Vd (V) | Id_fwd (A) | Id_bwd (A) | Vb_fwd | Vb_bwd | "
              "Δdec |\n|---|---|---|---|---|---|\n")
    for r in rows:
        md.append(f"| {r['Vd']:.3f} | {r['Id_forward']:.3e} | "
                  f"{r['Id_backward']:.3e} | {r['Vb_forward']:+.4f} | "
                  f"{r['Vb_backward']:+.4f} | {r['abs_diff_decades']:.3f} |\n")

    md.append("\n## Interpretation\n")
    if verdict == "NO_BISTABILITY":
        md.append(
            "Forward and backward sweeps are bit-identical (max Δ < 0.01 dec). "
            "The Newton solver converges to the **unique** root regardless of "
            "warm-start; the high-Vb branch is **not an attractor** at this "
            "bias under the current physics. The z91g residual at "
            "VG1=0.6/VG2=0.0 is therefore **not** a missed bistable branch — "
            "it is a genuine **model-physics gap** (impact-ion / BJT / "
            "leakage parameter shape) that no solver-side fix can recover. "
            "Next step: refit alpha0/beta0/NFACTOR/IS jointly across this "
            "diagnostic point or add a missing current pathway.\n")
    elif verdict == "BISTABILITY_CONFIRMED":
        md.append(
            "Forward and backward sweeps **diverge** by > 0.5 decade at one "
            "or more Vd points — the model has **two stable roots** at this "
            "bias and the cold-warm-start z91g protocol is silently locked "
            "onto the low-current branch. This opens a **solver-side fix**: "
            "either (a) seed Vb high before the snapback region, (b) run "
            "z91g with `solve_2t_with_homotopy` plus hot init, or (c) sweep "
            "from high Vd downward. Worth re-running the full sweep with "
            "the backward protocol and comparing fit RMSE.\n")
    else:
        md.append(
            "Sweeps differ by 0.01–0.5 decade somewhere on the grid — "
            "marginal hysteresis. The two branches exist but are weakly "
            "separated, suggesting a near-bifurcation rather than fully "
            "developed bistability. Worth a denser Vd grid near the "
            "transition and a continuation-style solve before declaring "
            "either physics-gap or solver-gap.\n")

    out_md = out_dir / "A1p_backward_sweep.md"
    out_md.write_text("".join(md))

    print(f"\nmax |Δlog10 Id| = {max_diff:.4f} decades  →  {verdict}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
