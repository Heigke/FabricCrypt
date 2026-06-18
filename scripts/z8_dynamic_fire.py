"""z8_dynamic_fire.py — branch-following KCL solver that captures the firing step.

The mistake in z6/z7 was treating the cell quasi-statically with a
cold-start iteration at every Vd. The body KCL equation

    Iii(Vb, Vd) + IGIDL(Vb, Vd)  =  (IS/BF)(exp(Vb/Vt) − 1) + Vb/Rb

has a low-Vb branch and a high-Vb branch. At some Vd_crit the low
branch disappears (saddle-node bifurcation) and the cell jumps to
the high-Vb branch — THAT is the firing step seen in the measurement.

A cold-start iteration from Vb=0 always lands on the low branch until
the low branch evaporates, then the iterator can't converge. This
script uses CONTINUATION: the initial guess at each Vd is the solution
from the previous Vd in the sweep. Brent's method finds the closest
root; when the low branch disappears the search bracket naturally
straddles the high branch and the solver jumps — producing the step.

At 0.2 V/s sweep with Cb = 1 fF, the body RC (Cb·Rb ≈ 0.1 ms) is
10^5× faster than the sweep. So branch-tracking quasi-static is
physically correct; full ODE integration adds no information and
is much more expensive. If hysteresis is observed in data (forward
vs reverse sweep differ) we'll revisit.
"""
from __future__ import annotations

import csv, json, re
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from nsram.bsim4 import (
    BSIM4_PRESETS,
    bipolar_collector_current_ss,
    drain_current_bsim,
    gidl_current,
    impact_ionization_bsim4,
)
from nsram.physics import thermal_voltage


DATA_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
                "data/sebas_2026_04_22")
OUT_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
               "results/z8_dynamic_fire")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRESET = BSIM4_PRESETS["ns_ram_130nm_pazos"]
Vt = thermal_voltage(300.0)

VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")


def kcl_net(Vb: float, Vg1: float, Vd: float, p) -> float:
    """Net current INTO the body node at given (Vb, Vg1, Vd).

    Positive = body charging (body voltage would rise).
    Zero-crossings = steady-state solutions.
    """
    Iii = float(impact_ionization_bsim4(Vg1, Vd, Vb, p))
    Igidl = float(gidl_current(Vd, Vg1, Vb, p)) if p.AGIDL > 0 else 0.0
    IS_eff = p.BJT_IS * max(p.BJT_AREA, 1e-30)
    # Guard exp to avoid overflow at Vb close to VJE
    Vb_clip = min(Vb, p.BJT_VJE * 1.1)
    exp_arg = np.clip(Vb_clip / (p.BJT_NE * Vt), -60.0, 60.0)
    Ib_out = (IS_eff / p.BJT_BF) * (np.exp(exp_arg) - 1.0) + Vb / p.Rb
    return (Iii + Igidl) - Ib_out


def find_vb(Vg1: float, Vd: float, p, Vb_init: float,
             vb_max: float = 0.85) -> float:
    """Find steady-state Vb using continuation from Vb_init.

    Strategy: Brent-bracket around Vb_init, expand if needed.
    If no zero-crossing in low-Vb neighbourhood, search the full range
    — low branch has disappeared, solver finds the upper (firing) root.
    """
    # Quick bracket near Vb_init
    lo = max(0.0, Vb_init - 0.02)
    hi = min(vb_max, max(Vb_init + 0.02, 0.02))
    f_lo = kcl_net(lo, Vg1, Vd, p)
    f_hi = kcl_net(hi, Vg1, Vd, p)
    if f_lo * f_hi <= 0 and abs(f_lo) + abs(f_hi) > 0:
        try:
            return float(brentq(kcl_net, lo, hi, args=(Vg1, Vd, p),
                                 xtol=1e-6, rtol=1e-6))
        except ValueError:
            pass
    # Fall back to full-range scan — pick root closest to Vb_init.
    # Scan kcl_net on a fine grid and find sign changes.
    grid = np.linspace(0.0, vb_max, 121)
    fs = np.array([kcl_net(v, Vg1, Vd, p) for v in grid])
    sign_changes = np.where(np.sign(fs[:-1]) != np.sign(fs[1:]))[0]
    if len(sign_changes) == 0:
        # No root on the low branch — take the highest-Vb endpoint
        return float(vb_max)
    roots = []
    for i in sign_changes:
        try:
            r = brentq(kcl_net, grid[i], grid[i + 1], args=(Vg1, Vd, p),
                        xtol=1e-6, rtol=1e-6)
            roots.append(r)
        except ValueError:
            continue
    if not roots:
        return float(vb_max)
    # Prefer the root with the same SIGN_DERIVATIVE structure as a stable
    # fixed point (f'(Vb) < 0). Simpler heuristic: if the iteration was
    # previously at low Vb, prefer lowest root. If it had jumped high,
    # prefer closest-to-previous root.
    if Vb_init < 0.1:
        return float(min(roots))
    return float(min(roots, key=lambda r: abs(r - Vb_init)))


def trace_cell(Vg1: float, Vd_sweep: np.ndarray, p) -> tuple:
    """Sweep Vd and return arrays of Vb, Id_channel, Ic_BJT, Id_total."""
    Vb = 0.0
    Vbs = np.zeros_like(Vd_sweep)
    for k, Vd in enumerate(Vd_sweep):
        Vb = find_vb(Vg1, float(Vd), p, Vb_init=Vb)
        Vbs[k] = Vb
    Ids, _ = drain_current_bsim(Vg1, Vd_sweep, Vbs, p)
    Ic = bipolar_collector_current_ss(Vg1, Vd_sweep, Vbs, p)
    Id_total = np.asarray(Ids) + np.asarray(Ic)
    return Vbs, np.asarray(Ids), np.asarray(Ic), Id_total


def load_csv(path: Path):
    rows = []
    with open(path) as f:
        rdr = csv.reader(f); next(rdr)
        for r in rdr:
            try:
                rows.append((float(r[2]), float(r[0]), float(r[1])))
            except ValueError:
                continue
    rows.sort()
    Vd = np.array([r[1] for r in rows])
    Id = np.array([r[2] for r in rows])
    peak = int(np.argmax(Vd))
    return Vd[:peak + 1], Id[:peak + 1]


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
    print(f"{len(entries)} curves — branch-following quasi-static solver")
    print(f"  BJT_AREA={PRESET.BJT_AREA}  Rb={PRESET.Rb:.0e}")

    per_curve = []
    examples = {}
    for vg1, vg2, path in sorted(entries):
        vd, idd = load_csv(path)
        # Trace in Vd-ascending order
        Vb_trace, Ids_tr, Ic_tr, Id_tr = trace_cell(vg1, vd, PRESET)
        rmse = log_rmse(idd, Id_tr, vd)
        # Detect firing jump
        dVb = np.diff(Vb_trace)
        jump_idx = int(np.argmax(dVb))
        Vd_fire = float(vd[jump_idx + 1]) if dVb[jump_idx] > 0.05 else None
        per_curve.append({
            "vg1": vg1, "vg2": vg2, "file": path.name,
            "log_rmse": rmse,
            "Vd_fire_model": Vd_fire,
            "Vb_final": float(Vb_trace[-1]),
        })
        if vg1 not in examples and abs(vg2) < 0.06:
            examples[vg1] = dict(
                vd=vd, id=idd, vg2=vg2,
                Vb=Vb_trace, Ids=Ids_tr, Ic=Ic_tr, Id_total=Id_tr,
                rmse=rmse, Vd_fire=Vd_fire,
            )

    rs = np.array([r["log_rmse"] for r in per_curve if np.isfinite(r["log_rmse"])])
    summary = {
        "method": "branch-following continuation, quasi-static KCL at body node",
        "preset": "ns_ram_130nm_pazos",
        "n_traces": len(per_curve),
        "log_rmse_median": float(np.median(rs)) if rs.size else None,
        "log_rmse_p90":    float(np.percentile(rs, 90)) if rs.size else None,
        "log_rmse_best":   float(rs.min()) if rs.size else None,
        "log_rmse_worst":  float(rs.max()) if rs.size else None,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(OUT_DIR / "per_curve.json", "w") as f:
        json.dump(per_curve, f, indent=2)

    # Overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, vg1 in zip(axes, sorted(examples)):
        e = examples[vg1]
        ax.semilogy(e["vd"], np.clip(e["id"], 1e-14, None), "k-", lw=2.2,
                     label=f"meas (VG2={e['vg2']:+.2f})")
        ax.semilogy(e["vd"], np.clip(e["Ids"], 1e-22, None), ":",
                     color="tab:blue", lw=1.1, label="Ids channel")
        ax.semilogy(e["vd"], np.clip(e["Ic"], 1e-22, None), "--",
                     color="tab:red", lw=1.3, label="Ic BJT")
        ax.semilogy(e["vd"], np.clip(e["Id_total"], 1e-22, None), "-",
                     color="tab:green", lw=1.6,
                     label=f"total (log-RMSE={e['rmse']:.2f})")
        if e["Vd_fire"]:
            ax.axvline(e["Vd_fire"], color="tab:green", lw=0.6, alpha=0.5,
                        ls=":")
        ax.set_title(f"VG1={vg1} V")
        ax.set_xlabel("Vd [V]"); ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle("Branch-following KCL solver — captures firing as bifurcation")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "overlay.png", dpi=140)
    plt.close(fig)

    # Vb(Vd) trace
    fig, ax = plt.subplots(figsize=(7, 4))
    for vg1, e in sorted(examples.items()):
        ax.plot(e["vd"], e["Vb"], lw=1.5, label=f"VG1={vg1}")
        if e["Vd_fire"]:
            ax.axvline(e["Vd_fire"], lw=0.5, alpha=0.4, ls=":")
    ax.set_xlabel("Vd [V]"); ax.set_ylabel("Body voltage Vb [V]")
    ax.axhline(PRESET.BJT_VJE, color="grey", ls="--", lw=0.5,
                label=f"VJE={PRESET.BJT_VJE}")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Floating-body voltage — jumps at firing bifurcation")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "vb_trace.png", dpi=140)
    plt.close(fig)

    # Residuals
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {0.2: "tab:blue", 0.4: "tab:orange", 0.6: "tab:green"}
    for r in per_curve:
        if np.isfinite(r["log_rmse"]):
            ax.scatter(r["vg2"], r["log_rmse"], c=colors.get(r["vg1"], "k"),
                       s=35, alpha=0.85)
    for vg1, c in colors.items():
        ax.plot([], [], "o", color=c, label=f"VG1={vg1}")
    ax.axhline(1.0, color="grey", lw=0.5, label="1 decade")
    ax.set_xlabel("VG2 [V]"); ax.set_ylabel("log-RMSE [decades]")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"Residuals (median={summary['log_rmse_median']:.2f} dec)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "residuals.png", dpi=140)
    plt.close(fig)

    print("\n═══ Branch-following quasi-static fit ═══")
    print(f"  median log-RMSE : {summary['log_rmse_median']:.2f} decades")
    print(f"  p90    log-RMSE : {summary['log_rmse_p90']:.2f} decades")
    print(f"  best/worst      : {summary['log_rmse_best']:.2f} / "
          f"{summary['log_rmse_worst']:.2f}")
    print(f"\n═══ Firing detection ═══")
    for vg1, e in sorted(examples.items()):
        print(f"  VG1={vg1}: Vd_fire={e['Vd_fire']}  "
              f"Vb_final={e['Vb'][-1]:.3f} V")


if __name__ == "__main__":
    main()
