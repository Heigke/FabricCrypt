"""z6_sebas_iv_fit.py — Pure first-principles prediction vs Sebas's 2026-04-22 data.

Using EVERY piece of info Sebas has sent:
  • 2tnsram_simple.asc schematic → topology, device sizes, BJT area=1u
  • parasiticBJT.txt → NPN Gummel-Poon params (is=5n, bf=10000, ...)
  • PTM130bulkNSRAM.txt → 130nm BSIM4 card (Vth0, K1/K2, NFACTOR, VOFF,
    ALPHA0, BETA0, AGIDL, KT1, UTE, UA1, UB1, ...)
  • 33 × I-V CSVs (VG1 × VG2) at 0.2 V/s

And the BSIM4.3 manual for §2.2 Vth(Vbs), §3 Ids (with subthreshold
via Vgsteff smoothing), §6.1 Iii, §6.2 GIDL.

The model that ships in nsram.bsim4:
  Id_total(Vg1, Vg2, Vd) = Ids_M1(Vg1, Vd-Sint, Vbs) + Ic_Q1(..., Vbs)

with three coupled unknowns solved self-consistently:
  - Vbs   : M1 floating-body voltage (BJT base). Set by body-charge
            balance Iii + IGIDL = (IS_eff/BF)·(exp(Vbs/Vt)−1)
  - Sint  : M1 source node = M2 drain node. Set by KCL:
            I_M1(Vd−Sint) = I_M2(Sint) with M2 = BSIM4, L = 10·Ln, no BJT
  - IS_eff = IS · BJT_AREA (SPICE area=1u multiplier from schematic)

ZERO free parameters per curve. The only scalar that isn't from
Sebas's files is the initial damping in the fixed-point iteration.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nsram.bsim4 import (
    BSIM4_PRESETS, bipolar_collector_current_ss,
    body_steady_state_vbs, drain_current_bsim,
    impact_ionization_bsim4, total_cell_current_ss,
    two_transistor_cell_ss,
)

DATA_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
                "data/sebas_2026_04_22")
OUT_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
               "results/z6_sebas_iv_fit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRESET = BSIM4_PRESETS["ns_ram_130nm_pazos"]


VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")


def load_csv(path: Path):
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            try:
                rows.append((float(r[2]), float(r[0]), float(r[1])))
            except ValueError:
                continue
    rows.sort()
    Vd = np.array([r[1] for r in rows])
    Id = np.array([r[2] for r in rows])
    peak = int(np.argmax(Vd))
    return Vd[: peak + 1], Id[: peak + 1]


def discover():
    for sub in sorted(DATA_DIR.iterdir()):
        if not sub.is_dir():
            continue
        for fn in sorted(sub.glob("*.csv")):
            m = VG_RE.search(fn.name)
            if m:
                yield float(m.group(2)), float(m.group(1)), fn


def log_rmse(Id_meas, Id_pred, Vd, vmin=0.3):
    m = (Vd > vmin) & (Id_meas > 1e-12) & (Id_pred > 0)
    if m.sum() < 5:
        return float("nan")
    lm = np.log10(np.clip(Id_meas[m], 1e-20, None))
    lp = np.log10(np.clip(Id_pred[m], 1e-30, None))
    return float(np.sqrt(np.mean((lm - lp) ** 2)))


def main():
    entries = list(discover())
    print(f"Loaded {len(entries)} I-V traces — pure first-principles 2T solve")
    print(f"  BSIM4 M1: VTH0={PRESET.VTH0} K1={PRESET.K1} K2={PRESET.K2} "
          f"N={PRESET.NFACTOR} VOFF={PRESET.VOFF}")
    print(f"         ALPHA0={PRESET.ALPHA0:.2e} BETA0={PRESET.BETA0} "
          f"AGIDL={PRESET.AGIDL:.2e}")
    print(f"  NPN   : IS={PRESET.BJT_IS} BF={PRESET.BJT_BF} "
          f"AREA={PRESET.BJT_AREA} → IS_eff={PRESET.BJT_IS*PRESET.BJT_AREA:.2e}")
    print(f"  M2    : L = {PRESET.Leff*10:.1e}  (via two_transistor_cell_ss)")

    per_curve = []
    examples = {}
    # Predict in two modes: (a) M1+NPN only, (b) full 2T (M1+NPN+M2).
    # Mode (b) is currently exploratory — the 2T solver's M2 uses the
    # shared PTM130 "normal Vth=0.54" card, but Sebas's real M2 is
    # likely a low-Vt/native flavor contained only in his NDA-protected
    # foundry card. The solver therefore fails to converge on
    # current-carrying solutions at VG1=0.4/0.6 where M2 is too weak.
    for vg1, vg2, path in sorted(entries):
        vd, idd = load_csv(path)
        vbs_a = body_steady_state_vbs(vg1, vd, PRESET)
        pred_a = total_cell_current_ss(vg1, vd, p=PRESET, self_consistent=True)
        rmse_a = log_rmse(idd, pred_a, vd)
        try:
            I_b, Sint_b, Vbs_b = two_transistor_cell_ss(vg1, vg2, vd, PRESET)
            rmse_b = log_rmse(idd, I_b, vd)
        except Exception:
            I_b = np.zeros_like(vd); Sint_b = np.zeros_like(vd)
            Vbs_b = np.zeros_like(vd); rmse_b = float("nan")
        per_curve.append({
            "vg1": vg1, "vg2": vg2, "file": path.name,
            "log_rmse_M1_only": rmse_a,
            "log_rmse_2T_full": rmse_b,
            "Id_meas_peak": float(idd.max()),
            "Id_pred_peak_M1": float(np.max(pred_a)),
            "Id_pred_peak_2T": float(np.max(I_b)),
        })
        if vg1 not in examples and abs(vg2) < 0.06:
            examples[vg1] = {
                "vd": vd, "id": idd, "vg2": vg2,
                "pred_a": np.asarray(pred_a),
                "pred_b": np.asarray(I_b),
                "Sint": np.asarray(Sint_b),
                "Vbs_a": np.asarray(vbs_a),
                "Vbs_b": np.asarray(Vbs_b),
                "rmse_a": rmse_a, "rmse_b": rmse_b,
            }

    r_a = np.array([r["log_rmse_M1_only"] for r in per_curve
                    if np.isfinite(r["log_rmse_M1_only"])])
    r_b = np.array([r["log_rmse_2T_full"] for r in per_curve
                    if np.isfinite(r["log_rmse_2T_full"])])

    summary = {
        "mode": "pure prediction, zero free parameters",
        "preset": "ns_ram_130nm_pazos",
        "BJT_AREA_applied": PRESET.BJT_AREA,
        "BJT_IS_effective": PRESET.BJT_IS * PRESET.BJT_AREA,
        "n_traces": len(per_curve),
        "M1_only_log_rmse_median": float(np.median(r_a)) if r_a.size else None,
        "M1_only_log_rmse_p90":    float(np.percentile(r_a, 90)) if r_a.size else None,
        "full_2T_log_rmse_median": float(np.median(r_b)) if r_b.size else None,
        "full_2T_log_rmse_p90":    float(np.percentile(r_b, 90)) if r_b.size else None,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(OUT_DIR / "per_curve.json", "w") as f:
        json.dump(per_curve, f, indent=2)

    # Overlay 3 panels — measured vs both predictions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, vg1 in zip(axes, sorted(examples)):
        e = examples[vg1]
        ax.semilogy(e["vd"], np.clip(e["id"], 1e-14, None), "k-", lw=2.2,
                    label=f"meas (VG2={e['vg2']:+.2f})")
        ax.semilogy(e["vd"], np.clip(e["pred_a"], 1e-22, None), "--",
                    color="tab:red", lw=1.3,
                    label=f"M1+NPN only  ({e['rmse_a']:.2f} dec)")
        ax.semilogy(e["vd"], np.clip(e["pred_b"], 1e-22, None), "-",
                    color="tab:green", lw=1.6,
                    label=f"full 2T (M1+NPN+M2)  ({e['rmse_b']:.2f} dec)")
        ax.set_title(f"VG1={vg1} V"); ax.set_xlabel("Vd [V]")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle("First-principles BSIM4 + NPN (BJT_AREA=1u) + M2 series — zero free parameters")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "overlay.png", dpi=140)
    plt.close(fig)

    # Internal state — Sint and Vbs vs Vd
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for vg1, e in sorted(examples.items()):
        axes[0].plot(e["vd"], e["Sint"], lw=1.5, label=f"VG1={vg1}")
        axes[1].plot(e["vd"], e["Vbs_b"], lw=1.5, label=f"VG1={vg1}")
    axes[0].set_xlabel("Vd [V]"); axes[0].set_ylabel("Sint [V]")
    axes[0].set_title("M1 source node voltage (2T solve)")
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].set_xlabel("Vd [V]"); axes[1].set_ylabel("Vbs [V]")
    axes[1].set_title("Floating-body voltage")
    axes[1].axhline(PRESET.BJT_VJE, color="grey", lw=0.5, ls="--",
                     label=f"VJE={PRESET.BJT_VJE}")
    axes[1].grid(alpha=0.3); axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "internal_state.png", dpi=140)
    plt.close(fig)

    # Residuals
    fig, ax = plt.subplots(figsize=(7, 4.3))
    colors = {0.2: "tab:blue", 0.4: "tab:orange", 0.6: "tab:green"}
    for r in per_curve:
        if np.isfinite(r["log_rmse_2T_full"]):
            ax.scatter(r["vg2"], r["log_rmse_2T_full"],
                       c=colors.get(r["vg1"], "k"), s=35, alpha=0.85)
    for vg1, c in colors.items():
        ax.plot([], [], "o", color=c, label=f"VG1={vg1}")
    ax.axhline(1.0, color="grey", lw=0.5, label="1 decade")
    ax.set_xlabel("VG2 [V]"); ax.set_ylabel("log-RMSE [decades] (full 2T)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"Full-2T fit residuals (median="
                 f"{summary['full_2T_log_rmse_median']:.2f} dec)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "residuals.png", dpi=140)
    plt.close(fig)

    print(f"\n═══ M1+NPN only (no M2) ═══")
    print(f"  median log-RMSE : {summary['M1_only_log_rmse_median']:.2f} dec")
    print(f"  p90    log-RMSE : {summary['M1_only_log_rmse_p90']:.2f} dec")
    print(f"\n═══ Full 2T (M1+NPN+M2) ═══")
    print(f"  median log-RMSE : {summary['full_2T_log_rmse_median']:.2f} dec")
    print(f"  p90    log-RMSE : {summary['full_2T_log_rmse_p90']:.2f} dec")
    print(f"\n═══ Example: Vbs and Sint at Vd=2V ═══")
    for vg1, e in sorted(examples.items()):
        print(f"  VG1={vg1}: Sint={e['Sint'][-1]:.3f} V  Vbs={e['Vbs_b'][-1]:.3f} V  "
              f"I_pred={e['pred_b'][-1]:.2e}  I_meas={e['id'][-1]:.2e}")


if __name__ == "__main__":
    main()
