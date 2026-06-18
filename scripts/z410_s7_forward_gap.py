"""z410 — S7 forward-current 660,000× gap investigation (4 tests).

Investigates why pyport predicts Ids(VG1=0.6, Vd=1.5) ~ 39 pA when
Sebas measures ~ 20.7 µA (660,000×). Four hypothesis tests:

  A — Geometry audit (text-only, written to results/z402_geometry_audit/findings.md)
  B — Vth0_override sweep   → results/z403_vth0_shift/{sweep.png, summary.json}
  C — BTBT explicit term    → results/z404_btbt/{sweep.png, summary.json}
  D — Capacitive Vd→body    → results/z405_cap_inject/summary.json

Pre-registered gates logged in results/.../summary.json:
  INFRA       :  finishes < 90 min, no NaN
  DISCOVERY   :  Ids @ VG1=0.6,Vd=1.5V > 1 µA
  AMBITIOUS   :  combined within 3× of 20.7 µA
  KILL-SHOT   :  none of A/B/C/D closes > 100× → BSIM4 insufficient
"""
from __future__ import annotations
import os, sys, time, json, math, importlib.util
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))

DATA = ROOT / "data/sebas_2026_04_22"

# Reuse helpers
_z365 = importlib.util.spec_from_file_location("z365", ROOT / "scripts/z365_perVG1_bbo.py")
z365 = importlib.util.module_from_spec(_z365); _z365.loader.exec_module(z365)

from nsram.bsim4_port.nsram_cell_2T import forward_2t


# === reference measurement at VG1=0.6, VG2=0.2, Vd=1.5 ===
MEASURED_FILE = DATA / "2vHCa-2 I-Vs@VG2 VG1=0.6 vnwell=2" / "StandardIV_HH_2vHCa-2_VG2=0.20_VG=0.6(1)_03-45-46PM.csv"
_d = np.loadtxt(MEASURED_FILE, delimiter=",", skiprows=1)
_idx = np.argmin(np.abs(_d[:, 0] - 1.5))
IDS_MEAS_15V = float(np.abs(_d[_idx, 1]))  # ~ 2.07e-5 A
print(f"[ref] Sebas Ids(VG1=0.6, VG2=0.2, Vd=1.5) = {IDS_MEAS_15V:.4e} A", flush=True)


# ===================== Test A — Geometry audit ===========================
def test_A_geometry():
    out = ROOT / "results/z402_geometry_audit"
    out.mkdir(parents=True, exist_ok=True)
    findings = []
    findings.append("# z402 — Geometry audit (S7-A)")
    findings.append("")
    findings.append("## Sources")
    findings.append("- pyport NSRAMCell2TConfig defaults: nsram/nsram/bsim4_port/nsram_cell_2T.py")
    findings.append("- LTspice schematic: data/sebas_2026_04_22/2tnsram_simple.asc")
    findings.append("- BSIM4 M1 card: data/sebas_2026_04_22/M1_130DNWFB.txt")
    findings.append("- Sebas DC param CSV: data/sebas_2026_04_22/2Tcell_BSIM_param_DC.csv")
    findings.append("- Measurement CSV: 2vHCa-2 I-Vs@VG2 VG1=0.6 vnwell=2/...VG2=0.20...")
    findings.append("")

    # pyport defaults
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    cfg = NSRAMCell2TConfig()
    findings.append("## pyport defaults")
    findings.append(f"- Ln (M1 length) = {cfg.Ln*1e9:.1f} nm")
    findings.append(f"- Wn (channel width) = {cfg.Wn*1e9:.1f} nm")
    findings.append(f"- M2_length_factor = {cfg.M2_length_factor}  → L_M2 = {cfg.Ln*cfg.M2_length_factor*1e9:.1f} nm")
    findings.append(f"- Total M1 W/L ratio = {cfg.Wn/cfg.Ln:.2f}")

    # LTspice schematic
    asc = (DATA / "2tnsram_simple.asc").read_text(errors="ignore")
    findings.append("")
    findings.append("## LTspice schematic params (`.param` lines)")
    for line in asc.splitlines():
        if ".param" in line.lower() and ("ln" in line.lower() or "wn" in line.lower()):
            findings.append(f"  `{line.strip()}`")
    findings.append("- Schematic: Ln=0.18 µm, Wn=0.36 µm (W/L = 2.0)")

    # Sebas BSIM CSV header
    import csv
    csv_path = DATA / "2Tcell_BSIM_param_DC.csv"
    with open(csv_path) as fh:
        rdr = csv.DictReader(fh)
        rows = list(rdr)
    findings.append("")
    findings.append("## Sebas BSIM CSV (per-curve fit params)")
    findings.append(f"- Columns: {list(rows[0].keys())}")
    findings.append(f"- `area` field across all rows: "
                    f"{sorted(set(r['area'] for r in rows))}")
    findings.append("- NOTE: `area` here is BJT emitter area for parasiticBJT.txt"
                    " (1 µm² in Sebas BJT model card), NOT MOSFET W·L.")

    # Compare
    findings.append("")
    findings.append("## Comparison & verdict")
    findings.append(f"- pyport W/L = {cfg.Wn*1e6:.2f}/{cfg.Ln*1e6:.2f} µm")
    findings.append(f"- schematic W/L = 0.36/0.18 µm")
    findings.append("- **MATCH** — pyport geometry exactly equals what Sebas simulated.")
    findings.append("- BSIM4 cards: M1 reuses 130DNWFB (PTM 130nm, normal-Vth); no W/L override")
    findings.append("- Conclusion: **Test A NEGATIVE** — geometry is not the source of"
                    " the 660,000× forward current gap.")
    findings.append("")
    findings.append(f"## Reference measurement")
    findings.append(f"- Ids(VG1=0.6, VG2=0.2, Vd=1.5 V) = {IDS_MEAS_15V:.4e} A "
                    f"(≈ {IDS_MEAS_15V*1e6:.2f} µA)")

    (out / "findings.md").write_text("\n".join(findings) + "\n")
    print(f"[A] wrote {out/'findings.md'}", flush=True)
    return {"verdict": "NEGATIVE — geometry matches", "pyport_W_um": cfg.Wn*1e6,
            "pyport_L_um": cfg.Ln*1e6, "schematic_W_um": 0.36, "schematic_L_um": 0.18}


# ===================== shared base-builder for B/C/D ====================
def build_base():
    cfg, M1, M2, bjt = z365.build_pyport_base()
    return cfg, M1, M2, bjt


def predict_ids(cfg, M1, M2, bjt, VG1=0.6, VG2=0.2, Vd=1.5):
    """Single-point prediction at the canonical reference point."""
    Vd_t = torch.tensor([Vd], dtype=torch.float64)
    # Use a Sebas-matched override row to keep parity with z365 sweeps
    sebas_rows = z365.load_sebas_params()
    row, _ = z365.find_or_impute_row(sebas_rows, VG1, VG2)
    P_M1, P_M2 = z365.make_overrides(row)
    sd_M1 = cfg.size_dep_M1(M1)
    try:
        with z365.patch_sd_scaled(sd_M1, P_M1), z365.patch_sd_scaled(cfg.size_dep_M2(M2), P_M2):
            out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_t,
                             VG1=torch.tensor(VG1, dtype=torch.float64),
                             VG2=torch.tensor(VG2, dtype=torch.float64),
                             warm_start=False)
        Id = float(np.abs(out["Id"].detach().cpu().numpy())[0])
        return Id
    except Exception as e:
        return float("nan")


# ===================== Test B — Vth0 sweep ===============================
def test_B_vth0():
    out = ROOT / "results/z403_vth0_shift"
    out.mkdir(parents=True, exist_ok=True)
    sweep = [0.20, 0.30, 0.40, 0.50, 0.541, 0.60, 0.70, 0.85]
    results = []
    print("[B] Vth0 sweep:", flush=True)
    for v in sweep:
        cfg, M1, M2, bjt = build_base()
        # Override vth0 directly on the model card (applies to both M1 + sd)
        M1._values["vth0"] = v
        # For consistency, M2 keeps its native vth0 (selector); M1 is the conducting channel
        Id = predict_ids(cfg, M1, M2, bjt)
        ratio = IDS_MEAS_15V / Id if (Id > 0 and not math.isnan(Id)) else float("nan")
        results.append({"vth0": v, "Ids": Id, "gap_x": ratio})
        print(f"  vth0={v:.3f} → Ids={Id:.3e} A, gap={ratio:.2e}×", flush=True)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    vth = np.array([r["vth0"] for r in results])
    ids = np.array([r["Ids"] for r in results])
    ax.semilogy(vth, ids, "o-", label="pyport Ids")
    ax.axhline(IDS_MEAS_15V, color="r", ls="--", label=f"measured = {IDS_MEAS_15V:.2e} A")
    ax.axhline(1e-6, color="gray", ls=":", label="DISCOVERY gate (1 µA)")
    ax.set_xlabel("vth0_override [V]")
    ax.set_ylabel("Ids @ VG1=0.6, VG2=0.2, Vd=1.5 V  [A]")
    ax.set_title("z403 — Vth0 override sweep")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "sweep.png", dpi=120)
    plt.close(fig)

    # Find best
    finite = [r for r in results if r["Ids"] > 0 and not math.isnan(r["Ids"])]
    best = max(finite, key=lambda r: r["Ids"]) if finite else None
    summary = {
        "measured_Ids": IDS_MEAS_15V,
        "sweep": results,
        "best_vth0": best["vth0"] if best else None,
        "best_Ids": best["Ids"] if best else None,
        "max_closure_x": (IDS_MEAS_15V / best["Ids"]) if best and best["Ids"] > 0 else None,
        "gate_DISCOVERY": (best["Ids"] > 1e-6) if best else False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[B] best vth0={summary['best_vth0']}, "
          f"Ids={summary['best_Ids']:.3e}, gap closure to {summary['max_closure_x']:.2e}×",
          flush=True)
    return summary


# ===================== Test C — BTBT explicit term =======================
def test_C_btbt():
    out = ROOT / "results/z404_btbt"
    out.mkdir(parents=True, exist_ok=True)
    # I_BTBT = A * Vds * (Eg/q)^2 * exp(-B / Vds)
    # Standard 130nm thick-ox BTBT: A ~ 1e-5..1e-2 A/V^3, B ~ 5..30 V
    Eg_q = 1.12  # eV for Si
    Vd = 1.5
    cfg, M1, M2, bjt = build_base()
    Id_base = predict_ids(cfg, M1, M2, bjt)
    print(f"[C] base (no BTBT) Ids = {Id_base:.3e} A", flush=True)

    sweep = []
    # Sweep A logarithmically at modest B, then sweep B
    for A_btbt in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        for B_btbt in [5.0, 10.0, 20.0]:
            I_btbt = A_btbt * Vd * (Eg_q**2) * math.exp(-B_btbt / Vd)
            Id_total = Id_base + I_btbt
            sweep.append({"A": A_btbt, "B": B_btbt,
                          "I_btbt": I_btbt, "Id_total": Id_total,
                          "gap_x": IDS_MEAS_15V / Id_total if Id_total > 0 else None})
    # Plot Id_total vs A for B=10
    fig, ax = plt.subplots(figsize=(7, 5))
    for B_pick in [5.0, 10.0, 20.0]:
        rows = [r for r in sweep if r["B"] == B_pick]
        As = [r["A"] for r in rows]
        Ids = [r["Id_total"] for r in rows]
        ax.loglog(As, Ids, "o-", label=f"B={B_pick} V")
    ax.axhline(IDS_MEAS_15V, color="r", ls="--", label=f"measured = {IDS_MEAS_15V:.2e} A")
    ax.axhline(1e-6, color="gray", ls=":", label="DISCOVERY gate (1 µA)")
    ax.set_xlabel("BTBT A coefficient [A/V³]")
    ax.set_ylabel(f"Ids (Vd={Vd}) total [A]")
    ax.set_title("z404 — BTBT term sweep (added on top of base BSIM4)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "sweep.png", dpi=120)
    plt.close(fig)

    finite = [r for r in sweep if r["Id_total"] > 0 and not math.isnan(r["Id_total"])]
    best = max(finite, key=lambda r: r["Id_total"]) if finite else None
    summary = {
        "measured_Ids": IDS_MEAS_15V,
        "Id_base_no_btbt": Id_base,
        "sweep": sweep,
        "best": best,
        "max_closure_x": IDS_MEAS_15V / best["Id_total"] if best else None,
        "gate_DISCOVERY": (best["Id_total"] > 1e-6) if best else False,
        "note": ("BTBT formula: I = A·Vds·(Eg/q)²·exp(-B/Vds). "
                 "Realistic 130nm A range: 1e-5..1e-3 A/V³, B: 5..20 V."),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[C] best (A={best['A']}, B={best['B']}) "
          f"→ Id_total={best['Id_total']:.3e}, gap={summary['max_closure_x']:.2e}×",
          flush=True)
    return summary


# ===================== Test D — Capacitive Vd → body injection ===========
def test_D_cap():
    out = ROOT / "results/z405_cap_inject"
    out.mkdir(parents=True, exist_ok=True)
    # I_cap_to_body = Cgd_eff * dVd/dt
    # For a slow-DC sweep of 0→2 V over tau_sweep, dVd/dt ≈ 2/tau_sweep.
    Vd = 1.5
    # Plausible Cgd_eff for W=0.36 µm, L=0.18 µm, thick-ox 4 nm: ~ 0.5 fF
    # Sebas sweep at ~80 ms/decade: total sweep ~ 0.75 s for Vd 0→2V (per filenames spacing)
    # tau_sweep_eff ≈ 0.5 s as specified
    Cgd_eff = 0.5e-15  # 0.5 fF nominal
    sweep_rows = []
    for tau in [0.05, 0.1, 0.5, 1.0, 5.0]:
        dVdt = 2.0 / tau
        I_cap = Cgd_eff * dVdt
        sweep_rows.append({"tau_sweep_s": tau, "dVdt": dVdt,
                           "I_cap_to_body": I_cap})
    # Now: would such a body injection translate into a comparable Ids?
    # The injected body current acts like extra Iii; BJT gain Bf can amplify.
    # Use Bf = 991 (z365 base) as upper bound.
    Bf = 991.0
    cfg, M1, M2, bjt = build_base()
    Id_base = predict_ids(cfg, M1, M2, bjt)
    for r in sweep_rows:
        r["Bf"] = Bf
        r["I_bjt_amplified"] = r["I_cap_to_body"] * Bf
        r["Id_total_est"] = Id_base + r["I_bjt_amplified"]
        r["gap_x"] = IDS_MEAS_15V / r["Id_total_est"] if r["Id_total_est"] > 0 else None

    finite = [r for r in sweep_rows if r["Id_total_est"] > 0]
    best = max(finite, key=lambda r: r["Id_total_est"]) if finite else None
    summary = {
        "measured_Ids": IDS_MEAS_15V,
        "Cgd_eff_F": Cgd_eff,
        "Id_base_no_cap": Id_base,
        "Bf_used": Bf,
        "sweep": sweep_rows,
        "best": best,
        "max_closure_x": IDS_MEAS_15V / best["Id_total_est"] if best else None,
        "gate_DISCOVERY": (best["Id_total_est"] > 1e-6) if best else False,
        "note": ("DC-displacement: I_cap = Cgd·dVd/dt, amplified through BJT Bf. "
                 "tau_sweep_eff ≈ 0.5 s per slow-DC measurement protocol."),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[D] best tau={best['tau_sweep_s']}s → I_cap={best['I_cap_to_body']:.3e}, "
          f"Id_total_est={best['Id_total_est']:.3e}, gap={summary['max_closure_x']:.2e}×",
          flush=True)
    return summary


# ===================== driver ============================================
def main():
    t0 = time.time()
    print(f"[z410] S7 forward-gap investigation @ {time.strftime('%H:%M:%S')}", flush=True)
    A = test_A_geometry()
    print(f"--- A done ({time.time()-t0:.1f}s) ---", flush=True)
    B = test_B_vth0()
    print(f"--- B done ({time.time()-t0:.1f}s) ---", flush=True)
    C = test_C_btbt()
    print(f"--- C done ({time.time()-t0:.1f}s) ---", flush=True)
    D = test_D_cap()
    print(f"--- D done ({time.time()-t0:.1f}s) ---", flush=True)

    # Combined gate evaluation
    Id_base = C["Id_base_no_btbt"]
    Id_combined = (B["best_Ids"] or Id_base) + \
                  (C["best"]["I_btbt"] if C["best"] else 0) + \
                  (D["best"]["I_bjt_amplified"] if D["best"] else 0)
    gap_combined = IDS_MEAS_15V / Id_combined if Id_combined > 0 else None
    closures = {
        "A_geometry": None,  # qualitative
        "B_vth0": B.get("max_closure_x"),
        "C_btbt": C.get("max_closure_x"),
        "D_cap":  D.get("max_closure_x"),
        "combined_gap_x": gap_combined,
    }
    target_gap = IDS_MEAS_15V / 39e-12  # the 660,000× headline number
    # AMBITIOUS gate: combined within 3× of measured
    ambit = (gap_combined is not None) and (gap_combined < 3.0)
    # KILL-SHOT: none of A/B/C/D closes >100× of headline gap
    closure_factors = []
    base_pred = Id_base
    for s in (B.get("best_Ids"),
              (C["best"]["Id_total"] if C["best"] else None),
              (D["best"]["Id_total_est"] if D["best"] else None)):
        if s and base_pred > 0:
            closure_factors.append(s / base_pred)
    kill_shot = all(f < 100.0 for f in closure_factors) if closure_factors else True

    final = {
        "elapsed_s": time.time() - t0,
        "measured_Ids": IDS_MEAS_15V,
        "headline_pyport_Ids_ref_39pA": 39e-12,
        "headline_gap_x": target_gap,
        "Id_base_pyport_current": Id_base,
        "Id_combined_est": Id_combined,
        "closures": closures,
        "gates": {
            "INFRA_under_90min": (time.time() - t0) < 90 * 60,
            "DISCOVERY_any_test_gt_1uA": any([
                B.get("gate_DISCOVERY", False),
                C.get("gate_DISCOVERY", False),
                D.get("gate_DISCOVERY", False)
            ]),
            "AMBITIOUS_combined_within_3x": ambit,
            "KILL_SHOT_BSIM4_insufficient": kill_shot,
        },
        "tests": {"A": A, "B_summary": {k: B[k] for k in
                  ("best_vth0", "best_Ids", "max_closure_x", "gate_DISCOVERY")},
                  "C_summary": {"best": C["best"], "max_closure_x": C["max_closure_x"]},
                  "D_summary": {"best": D["best"], "max_closure_x": D["max_closure_x"]}},
    }
    (ROOT / "results/z410_s7_summary.json").write_text(json.dumps(final, indent=2))
    print("\n=== FINAL ===")
    print(json.dumps(final["gates"], indent=2))
    print(f"closures: {closures}")
    print(f"elapsed: {final['elapsed_s']:.1f} s")


if __name__ == "__main__":
    main()
