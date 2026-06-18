"""z377 — Body-strap diagnostic (S1 from SEVEN_GAPS_PLAN_2026-05-15).

Force Vb to a list of fixed values, then for each Vb compute Ids(Vd) at
VG1=0.6, VG2=+0.20 using the R-46 per-VG1 best params. Solver does 1D
Newton on Vsint with Vb pinned.

DECISION RULE:
- If at Vb >= 0.55 V the Ids semilogy curve jumps 2-3 decades above the
  Vb=0 curve at Vd >= 1.0 V  -> the physics IS in the model, the
  self-consistent 2D Newton just fails to find the high-Vb basin.
  -> S2 (continuation solver).
- Else (curves flat / same family at all Vb) -> BSIM4 Ids(Vbs) is
  fundamentally too weak. -> S4 (empirical fold).

Output: results/z377_body_strap_diagnostic/{ids_vs_vd_perVb.png,
        summary.json}
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import sys, json, math, importlib.util, csv, re
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
OUT = ROOT / "results/z377_body_strap_diagnostic"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"


@contextmanager
def patch_sd_scaled(sd, overrides):
    if not overrides:
        yield; return
    saved = {}
    try:
        for k, v in overrides.items():
            saved[k] = sd.scaled.get(k, None)
            sd.scaled[k] = float(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None: sd.scaled.pop(k, None)
            else: sd.scaled[k] = v


# Replicate z372 params loader
def load_sebas_params():
    rows = []
    with open(DATA / "2Tcell_BSIM_param_DC.csv") as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try: row[k] = float(v)
                except ValueError: row[k] = float("nan")
            rows.append(row)
    return rows


BRANCH_FLAT = {
    0.4: {"ETAB": 1.9,  "K1": 0.53825, "ALPHA0": 7.842e-05, "BETA0": 19.0, "NFACTOR": 6.0},
    0.6: {"ETAB": 2.5,  "K1": 0.41825, "ALPHA0": 7.842e-05, "BETA0": 20.0, "NFACTOR": 6.0},
}
M2_STATIC = {"k1": 0.63825, "k2": -0.070435, "etab": -0.086777, "beta0": 18.0}


def find_row(rows, VG1, VG2, atol=1e-3):
    for r in rows:
        if abs(r["VG1"] - VG1) < atol and abs(r["VG2"] - VG2) < atol:
            row = dict(r)
            if math.isnan(row.get("K1", float("nan"))):
                br = BRANCH_FLAT.get(round(VG1, 2))
                if br is not None:
                    for k, v in br.items(): row[k] = float(v)
            return row
    return None


def make_overrides(row):
    if row is None: return None, None
    P_M1 = {}
    for ck, pk in (("ETAB","etab"),("K1","k1"),("ALPHA0","alpha0"),("BETA0","beta0")):
        if not math.isnan(row.get(ck, float("nan"))): P_M1[pk] = float(row[ck])
    P_M2 = {}
    if not math.isnan(row.get("NFACTOR", float("nan"))): P_M2["nfactor"] = float(row["NFACTOR"])
    for k, v in M2_STATIC.items():
        P_M2.setdefault(k, float(v))
    return (P_M1 or None), (P_M2 or None)


def build_base():
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    cfg.body_pdiode_to = "vnwell"
    cfg.use_well_diode = True
    cfg.vnwell = 2.0
    cfg.body_pdiode_Js = 5.3675e-7 / 22e-12
    cfg.body_pdiode_n = 1.0535
    cfg.body_pdiode_Rs = 1.0e6
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Va = 0.903; bjt.Is = 5.95e-12; bjt.Bf = 991.0
    return cfg, M1, M2, bjt


def load_measured(vg1, vg2=0.20):
    sub = DATA / f"2vHCa-2 I-Vs@VG2 VG1={vg1} vnwell=2"
    pat = re.compile(rf"VG2={vg2:.2f}_VG={vg1}")
    for f in sorted(sub.glob("*.csv")):
        if pat.search(f.name):
            d = np.loadtxt(f, delimiter=",", skiprows=1)
            return d[:, 0], np.abs(d[:, 1])
    raise FileNotFoundError(f"no csv for VG1={vg1} VG2={vg2}")


def solve_vsint_only(cfg, M1, M2, bjt, Vd, VG1, VG2, Vb_fixed,
                    P_M1, P_M2, Vsint_init=None,
                    max_iter=60, tol=1e-12):
    """1D Newton on Vsint with Vb pinned. Returns (Ids, Vsint, converged)."""
    from nsram.bsim4_port.nsram_cell_2T import _residuals
    Vd = torch.as_tensor(Vd, dtype=torch.float64)
    if Vd.ndim == 0: Vd = Vd.unsqueeze(0)
    VG1 = torch.as_tensor(VG1, dtype=torch.float64).expand_as(Vd)
    VG2 = torch.as_tensor(VG2, dtype=torch.float64).expand_as(Vd)
    Vb = torch.full_like(Vd, float(Vb_fixed))
    if Vsint_init is None:
        Vsint = (Vd * 0.5).clone()
    else:
        vsi = torch.as_tensor(Vsint_init, dtype=torch.float64)
        if vsi.ndim == 0: vsi = vsi.unsqueeze(0)
        Vsint = vsi.expand_as(Vd).clone()
    converged = False
    # Note: P_M1/P_M2 are already applied via patch_sd_scaled in caller scope,
    # so we pass None here to avoid the legacy setattr-based override path.
    for it in range(max_iter):
        R_S, R_B, comp = _residuals(cfg, M1, bjt, Vd, VG1, VG2, Vsint, Vb,
                                    P_M1=None, P_M2=None, model_M2=M2)
        rS = R_S.detach()
        if rS.abs().max().item() < tol:
            converged = True; break
        # FD derivative dR_S/dVsint
        eps = 1e-6
        Vs_p = Vsint + eps
        R_S_p, _, _ = _residuals(cfg, M1, bjt, Vd, VG1, VG2, Vs_p, Vb,
                                 P_M1=None, P_M2=None, model_M2=M2)
        dRdVs = ((R_S_p.detach() - rS) / eps).clamp(min=1e-30, max=None)
        # Handle sign
        dRdVs_raw = (R_S_p.detach() - rS) / eps
        # Guard against zero derivative
        dRdVs_safe = torch.where(dRdVs_raw.abs() < 1e-25,
                                 torch.full_like(dRdVs_raw, 1e-25),
                                 dRdVs_raw)
        dVs = -rS / dRdVs_safe
        # Cap step
        dVs = dVs.clamp(-0.2, 0.2)
        Vsint = Vsint + dVs
        Vd_scalar = float(Vd.flatten()[0].item())
        Vsint = Vsint.clamp(min=-0.05, max=Vd_scalar + 0.05)
    # Ids from final M1 evaluation
    _, _, comp = _residuals(cfg, M1, bjt, Vd, VG1, VG2, Vsint, Vb,
                            P_M1=None, P_M2=None, model_M2=M2)
    Ids = comp["Ids_M1"].detach()
    return float(Ids.flatten()[0].item()), float(Vsint.flatten()[0].item()), converged


def main():
    cfg, M1, M2, bjt = build_base()
    # R-46 best for VG1=0.6 branch
    Bf, iii_gain, log10Rs = 417.63, 0.9036, 6.7846
    bjt.Bf = Bf
    cfg.iii_body_gain = iii_gain
    cfg.vnwell_Rs = 10 ** log10Rs

    sebas_rows = load_sebas_params()
    VG1, VG2 = 0.6, 0.20
    row = find_row(sebas_rows, VG1, VG2)
    P_M1, P_M2 = make_overrides(row)

    Vd_m, Id_m = load_measured(VG1, VG2)
    Vd_grid = Vd_m.astype(np.float64)

    Vb_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.60, 0.65, 0.70, 0.80]

    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)

    results = {}
    Ids_by_Vb = {}
    with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
        for Vb_fix in Vb_list:
            Ids_arr = np.zeros_like(Vd_grid)
            niter_conv = 0
            Vsint_warm = None
            for i, vd in enumerate(Vd_grid):
                try:
                    Ids, Vs, conv = solve_vsint_only(cfg, M1, M2, bjt,
                                                    torch.tensor(vd), VG1, VG2,
                                                    Vb_fix, P_M1, P_M2,
                                                    Vsint_init=Vsint_warm)
                    Vsint_warm = torch.tensor(Vs)
                    if conv: niter_conv += 1
                except Exception as e:
                    if i == 0:
                        import traceback; traceback.print_exc()
                    Ids = float("nan")
                Ids_arr[i] = abs(Ids)
            Ids_by_Vb[Vb_fix] = Ids_arr
            print(f"[Vb={Vb_fix:.2f}] conv={niter_conv}/{len(Vd_grid)}, "
                  f"max|Ids|={np.nanmax(Ids_arr):.3e}, "
                  f"Ids@Vd=1.5={Ids_arr[np.argmin(np.abs(Vd_grid-1.5))]:.3e}")

    # Decision metric: at Vd=1.5V, max log10(Ids) over Vb >=0.55 minus log10(Ids) at Vb=0
    Vd_eval = 1.5
    j = int(np.argmin(np.abs(Vd_grid - Vd_eval)))
    Ids_Vb0 = max(Ids_by_Vb[0.0][j], 1e-15)
    jumps = []
    for Vb_fix in Vb_list:
        if Vb_fix < 0.55: continue
        I = max(Ids_by_Vb[Vb_fix][j], 1e-15)
        jumps.append(math.log10(I) - math.log10(Ids_Vb0))
    max_jump_dec = max(jumps) if jumps else 0.0

    if max_jump_dec >= 2.0:
        decision = "SOLVER_ISSUE: forcing high Vb DOES recover ~{0:.1f} dec jump at Vd={1:.1f}V. Physics is in the model; the 2D Newton just doesn't find the high-Vb basin. -> S2 (continuation solver) is right call.".format(max_jump_dec, Vd_eval)
        gate = "S2"
    elif max_jump_dec >= 1.0:
        decision = "MARGINAL: jump of {0:.1f} dec at Vb=0.55+. Partial physics, partial solver. Try S2 first; fallback S4.".format(max_jump_dec)
        gate = "S2_TRY_THEN_S4"
    else:
        decision = "PHYSICS_ISSUE: even forcing Vb up to 0.8 V only gives {0:.1f} dec jump. BSIM4 Ids(Vbs) coupling is fundamentally too weak in this regime. -> S4 (empirical fold fallback) is right call.".format(max_jump_dec)
        gate = "S4"

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(Vb_list)))
    ax.semilogy(Vd_m, np.maximum(Id_m, 1e-15), "k.", ms=5, label="measured", zorder=10)
    for c, Vb_fix in zip(colors, Vb_list):
        ax.semilogy(Vd_grid, np.maximum(Ids_by_Vb[Vb_fix], 1e-15),
                    "-", color=c, lw=1.3, label=f"Vb={Vb_fix:.2f}V")
    ax.set_xlabel("Vd (V)")
    ax.set_ylabel("|Ids| (A)")
    ax.set_ylim(1e-13, 1e-2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.set_title(f"z377 body-strap: forced-Vb sweep @ VG1={VG1}, VG2={VG2}\n"
                 f"Max jump at Vd={Vd_eval:.1f}V (Vb>=0.55 vs Vb=0): {max_jump_dec:.2f} dec\n"
                 f"DECISION GATE: {gate}", fontsize=10)
    out_png = OUT / "ids_vs_vd_perVb.png"
    fig.tight_layout(); fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"[z377] wrote {out_png}")

    summary = {
        "script": "z377_body_strap_diagnostic",
        "VG1": VG1, "VG2": VG2,
        "params": {"Bf": Bf, "iii_body_gain": iii_gain, "vnwell_Rs": 10**log10Rs},
        "Vb_list": Vb_list,
        "Vd_grid": Vd_grid.tolist(),
        "Id_measured": [float(x) for x in Id_m],
        "Ids_by_Vb": {f"{Vb:.2f}": Ids_by_Vb[Vb].tolist() for Vb in Vb_list},
        "Vd_eval_for_decision_V": Vd_eval,
        "max_jump_dec_Vb_ge_055_vs_Vb0": max_jump_dec,
        "decision": decision,
        "gate": gate,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[z377] wrote {OUT/'summary.json'}")
    print("\n=== DECISION ===")
    print(decision)


if __name__ == "__main__":
    main()
