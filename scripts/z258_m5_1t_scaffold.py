"""z258 — M5 scaffold: 1T deep-Nwell floating-body cell I-V family.

Pre-registered sanity gate (shape-of-curve only, NOT a fit):
  1. I_d(V_d) monotonic non-decreasing in V_d at each V_G1.
  2. No NaN / Inf in the I_d family.
  3. I_d(V_d=2 V) > I_d(V_d=0)  at each V_G1.
  4. I_d at V_d=2 V is monotonic non-decreasing in V_G1
     (above-threshold ordering).

Source: research_plan/POST_AUDIT_FIX_PLAN_2026-05-11.md §M5 + slide 12.29(1).
Caveats: process-node mismatch (PTM 130 nm used vs 180 nm target),
no fit to silicon, body-diode params re-used from the 2T card. This
script is INFRASTRUCTURE, not silicon characterisation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nsram"))

from nsram import OneTFloatingBodyCell, BSIM4_PRESETS  # noqa: E402


OUT_DIR = ROOT / "results" / "z258_m5_1t"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sanity_check(family: dict) -> dict:
    """Apply the four pre-registered sanity gates."""
    Vd = family["Vd"]
    Vg1 = family["Vg1"]
    Id = family["Id"]
    n_vg, n_vd = Id.shape

    # 1. Monotonic in V_d at each V_G1 (allow tiny float noise).
    EPS_MONO = 1e-12  # A
    mono_per_vg = []
    for i in range(n_vg):
        diffs = np.diff(Id[i, :])
        mono = bool(np.all(diffs >= -EPS_MONO))
        mono_per_vg.append({
            "Vg1": float(Vg1[i]),
            "monotonic_Vd": mono,
            "min_dI": float(diffs.min()) if diffs.size else 0.0,
        })
    all_mono = all(d["monotonic_Vd"] for d in mono_per_vg)

    # 2. No NaN / Inf.
    finite = bool(np.all(np.isfinite(Id)))

    # 3. I_d(V_d=2V) > I_d(V_d=0) at each V_G1.
    j_lo = int(np.argmin(np.abs(Vd - 0.0)))
    j_hi = int(np.argmin(np.abs(Vd - 2.0)))
    end_to_end = []
    for i in range(n_vg):
        end_to_end.append({
            "Vg1": float(Vg1[i]),
            "Id_at_0V": float(Id[i, j_lo]),
            "Id_at_2V": float(Id[i, j_hi]),
            "Id_2V_gt_0V": bool(Id[i, j_hi] > Id[i, j_lo]),
        })
    all_end_to_end = all(d["Id_2V_gt_0V"] for d in end_to_end)

    # 4. I_d at V_d=2V monotonic in V_G1.
    id_at_2v = Id[:, j_hi]
    vg_diffs = np.diff(id_at_2v)
    mono_in_vg = bool(np.all(vg_diffs >= -EPS_MONO))

    overall = all_mono and finite and all_end_to_end and mono_in_vg

    return {
        "monotonic_Vd_per_Vg1": mono_per_vg,
        "all_monotonic_Vd": all_mono,
        "no_nan_inf": finite,
        "end_to_end": end_to_end,
        "all_end_to_end_positive": all_end_to_end,
        "Id_at_2V_per_Vg1": [float(x) for x in id_at_2v],
        "monotonic_in_Vg1_at_2V": mono_in_vg,
        "all_gates_pass": overall,
    }


def main() -> int:
    # Base BSIM4 card: 130 nm Pazos preset (the only fitted card we have).
    # Process-node mismatch documented — this is a scaffold, not a fit.
    base = BSIM4_PRESETS["ns_ram_130nm_pazos"]
    # Make a shallow copy so we don't mutate the preset for other scripts.
    import copy
    bsim = copy.deepcopy(base)

    cell = OneTFloatingBodyCell(bsim=bsim)

    Vg1_list = [0.2, 0.4, 0.6]
    Vd = np.arange(0.0, 2.0 + 1e-9, 0.05)  # step 50 mV inclusive

    print(f"[z258] sweeping {len(Vg1_list)} V_G1 values × {len(Vd)} V_d points")
    print(f"[z258] base card: ns_ram_130nm_pazos  (target: 180 nm — MISMATCH)")
    print(f"[z258] cell area: {cell.area_um2} µm²  Weff: {cell.Weff_1t*1e6:.2f} µm")

    family = cell.iv_family(Vg1_list, Vd)

    # Sanity gate.
    sanity = sanity_check(family)

    # Plot.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, vg in enumerate(Vg1_list):
        ax.plot(family["Vd"], family["Id"][i, :] * 1e6,
                color=colors[i % len(colors)],
                marker="o", ms=3, lw=1.2,
                label=f"V_G1 = {vg:.1f} V")
    ax.set_xlabel("V_d  (V)")
    ax.set_ylabel("I_d  (µA)")
    ax.set_title("1T deep-Nwell floating-body cell — I-V family\n"
                 "(SCAFFOLD: PTM 130 nm card, target 180 nm — magnitudes off)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png_path = OUT_DIR / "iv_family.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"[z258] plot saved → {png_path}")

    # Save summary JSON.
    summary = {
        "script": "scripts/z258_m5_1t_scaffold.py",
        "purpose": "M5 scaffold — 1T deep-Nwell floating-body cell sanity",
        "source_slide": "12.29(1)  (1T, 8 µm², 180 nm CMOS, Pazos)",
        "base_bsim4_card": "ns_ram_130nm_pazos",
        "cell": {
            "area_um2": cell.area_um2,
            "Weff_1t_m": cell.Weff_1t,
            "Rb_floating_ohm": cell.Rb_floating,
            "Rb_access_ohm": cell.Rb_access,
            "Vg1_switch_V": cell.Vg1_switch,
        },
        "sweep": {
            "Vg1_list_V": list(map(float, Vg1_list)),
            "Vd_min_V": float(Vd.min()),
            "Vd_max_V": float(Vd.max()),
            "Vd_step_V": 0.05,
            "n_Vd": int(len(Vd)),
        },
        "data": {
            "Vd_V": Vd.tolist(),
            "Vg1_V": list(map(float, Vg1_list)),
            "Id_A":  family["Id"].tolist(),
            "Vbs_V": family["Vbs"].tolist(),
        },
        "sanity_gate": sanity,
        "caveats": [
            "Process node mismatch: PTM 130 nm BSIM4 used; Sebas target 180 nm.",
            "No fit to silicon — shape-of-curve only.",
            "Body-diode (JSS/JSD/NJ/BVS) re-used from 2T card; 1T deep-Nwell "
            "p-body → n-well junction not modelled separately.",
            "Single-gate body-access modulated by sigmoid Rb(V_G1) — NOT fit.",
            "8 µm² area distributed evenly between source and drain regions; "
            "actual layout may differ.",
        ],
    }
    json_path = OUT_DIR / "summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[z258] summary saved → {json_path}")

    # Verdict.
    print("\n[z258] SANITY VERDICT")
    print(f"  all_monotonic_Vd       : {sanity['all_monotonic_Vd']}")
    print(f"  no_nan_inf             : {sanity['no_nan_inf']}")
    print(f"  all_end_to_end_positive: {sanity['all_end_to_end_positive']}")
    print(f"  monotonic_in_Vg1_at_2V : {sanity['monotonic_in_Vg1_at_2V']}")
    print(f"  ─── ALL GATES PASS     : {sanity['all_gates_pass']} ───")
    for d in sanity["end_to_end"]:
        print(f"   V_G1={d['Vg1']:.1f} V → I_d(0)={d['Id_at_0V']:.3e} A  "
              f"I_d(2V)={d['Id_at_2V']:.3e} A")

    return 0 if sanity["all_gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
