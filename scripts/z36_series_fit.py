"""z36_series_fit.py — DE fit using REAL series M1+M2 topology.

This is the topology Sebas's NS-RAM cell actually has:
    Vs(0) ── M1[VG1] ── Vmid ── M2[VG2] ── Vd
M1 and M2 share floating body Vb. At each Vd step we solve:
  - Vmid: I_M1(VG1, Vmid, Vb) = I_M2(VG2-Vmid, Vd-Vmid, Vb-Vmid)
  - Vb:   Iii_M2 - I_BJT_base - (Vb-VG2)/Rb = 0  (branch-following)

With this physics, VG2 modulates Id through M2's channel resistance — the
strong VG2-dependence the data shows.

Fit parameters (8):
  VTH0          ∈ [0.20, 0.65]   (wider — series stack lowers effective Vth)
  log10 BJT_AREA∈ [-6, -3]
  log10 ALPHA0  ∈ [-3, 1]
  log10 Rb      ∈ [8, 12]
  U0            ∈ [0.04, 0.15]
  RDSW          ∈ [50, 500]      (NEW — series-resistance fit knob)
  K2            ∈ [-0.20, 0.10]  (NEW — body-effect strength)
  ALPHA0_M2     ∈ [-3, 1]        (NEW — separate impact-ionization mult)
"""
from __future__ import annotations
import csv, json, re, time
from pathlib import Path
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

from nsram.nsram_canonical import trace_nsram_series, make_pazos_130nm

DATA = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
             "data/sebas_2026_04_22")
OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z36_series_fit")
OUT.mkdir(parents=True, exist_ok=True)
VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")
ID_FLOOR = 1e-13
N_DS = 12  # reduced from 20 — series solver is slower


def load_curves():
    curves = []
    for sub in sorted(DATA.iterdir()):
        if not sub.is_dir(): continue
        for fn in sorted(sub.glob("*.csv")):
            m = VG_RE.search(fn.name)
            if not m: continue
            vg2 = float(m.group(1)); vg1 = float(m.group(2))
            rows = []
            with open(fn) as f:
                rdr = csv.reader(f); next(rdr)
                for r in rdr:
                    try: rows.append((float(r[2]), float(r[0]), float(r[1])))
                    except ValueError: continue
            rows.sort()
            Vd = np.array([r[1] for r in rows]); Id = np.array([r[2] for r in rows])
            peak = int(np.argmax(Vd))
            Vd = Vd[:peak+1]; Id = Id[:peak+1]
            mask = Id > ID_FLOOR
            if mask.sum() < 10: continue
            Vd, Id = Vd[mask], Id[mask]
            uVd, idx = np.unique(Vd, return_index=True)
            Id = Id[idx]; Vd = uVd
            nvd = np.linspace(max(0.1, Vd.min()), Vd.max(), N_DS)
            nid = np.power(10.0, np.interp(nvd, Vd, np.log10(Id)))
            curves.append((vg1, vg2, nvd, nid))
    return curves


CURVES = load_curves()
print(f"Loaded {len(CURVES)} curves × {N_DS} pts")
GROUPS = {0.2: [], 0.4: [], 0.6: []}
for c in CURVES:
    for vg1 in GROUPS:
        if abs(c[0] - vg1) < 0.01:
            GROUPS[vg1].append(c)
            break
for k, v in GROUPS.items():
    print(f"  VG1={k}: {len(v)} curves")


def build(x):
    """x = [VTH0, log10_AREA, log10_alpha, log10_Rb, U0, RDSW, K2, log10_alpha_m2]"""
    vth0, lga, lgam, lgrb, u0, rdsw, k2, lgam2 = x
    # We use a single ALPHA0_mult (canonical model has only one Iii path);
    # ALPHA0_M2 is reserved for future split. For now combine multiplicatively.
    return replace(make_pazos_130nm(),
                    VTH0=float(vth0),
                    BJT_AREA=float(10**lga),
                    ALPHA0_mult=float(10**lgam) * float(10**lgam2),
                    Rb=float(10**lgrb),
                    U0=float(u0),
                    RDSW=float(rdsw),
                    K2=float(k2))


def curve_score(pred, meas):
    """Symmetric log-RMSE."""
    m = (meas > ID_FLOOR) & (pred > 0)
    if m.sum() < 5: return None
    log_diff = np.log10(meas[m]) - np.log10(pred[m])
    return float(np.sqrt(np.mean(log_diff ** 2)))


def objective(x):
    p = build(x)
    group_medians = []
    for vg1, group in GROUPS.items():
        if not group: continue
        rmses = []
        for vg1_v, vg2, vd, idd in group:
            try:
                pred, _ = trace_nsram_series(vg1_v, vg2, vd, p)
                r = curve_score(pred, idd)
                if r is not None: rmses.append(r)
            except Exception: continue
        if rmses:
            group_medians.append(np.median(rmses))
    if not group_medians: return 10.0
    return float(np.mean(group_medians))


BOUNDS = [
    (0.20, 0.65),     # VTH0  ← WIDE — series stack effective Vth uncertain
    (-6.0, -3.0),     # log10 BJT_AREA
    (-3.0, 1.0),      # log10 ALPHA0_mult
    ( 8.0, 12.0),     # log10 Rb
    (0.04, 0.15),     # U0
    (50.0, 500.0),    # RDSW
    (-0.20, 0.10),    # K2 (body-effect strength)
    (-2.0, 1.0),      # log10 ALPHA0 secondary mult
]


def main():
    print(f"DE series M1+M2 fit (8 params)  N_DS={N_DS}")
    t0 = time.time(); ctr = [0]
    def cb(xk, conv):
        ctr[0] += 1
        o = objective(xk)
        print(f"  iter {ctr[0]:3d} obj={o:.3f}  VTH0={xk[0]:.3f} "
              f"AREA=1e{xk[1]:.1f} A0=1e{xk[2]:.1f} Rb=1e{xk[3]:.1f} "
              f"U0={xk[4]:.3f} RDSW={xk[5]:.0f} K2={xk[6]:+.3f} A02=1e{xk[7]:.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        return False
    res = differential_evolution(
        objective, BOUNDS,
        maxiter=25, popsize=10,
        mutation=(0.5, 1.2), recombination=0.7,
        seed=51, tol=1e-3, workers=-1, updating="deferred",
        callback=cb, polish=False, disp=False,
    )
    xo = res.x
    p = build(xo)
    print(f"\nBest mean-of-group-medians = {res.fun:.3f}")
    print(f"  VTH0={xo[0]:.3f}  BJT_AREA={10**xo[1]:.2e}  "
          f"ALPHA0={10**xo[2]:.2e}*{10**xo[7]:.2e}  Rb={10**xo[3]:.2e}  "
          f"U0={xo[4]:.3f}  RDSW={xo[5]:.0f}  K2={xo[6]:+.3f}")

    per_curve = []
    group_stats = {}
    for vg1, group in GROUPS.items():
        rmses = []
        for vg1_v, vg2, vd, idd in group:
            try:
                pred, _ = trace_nsram_series(vg1_v, vg2, vd, p)
                r = curve_score(pred, idd)
                if r is None: continue
                per_curve.append({"vg1": vg1_v, "vg2": vg2, "log_rmse": r})
                rmses.append(r)
            except Exception: continue
        if rmses:
            group_stats[vg1] = {
                "n": len(rmses), "median": float(np.median(rmses)),
                "p90": float(np.percentile(rmses, 90)), "worst": float(np.max(rmses)),
            }
            print(f"VG1={vg1}: median={group_stats[vg1]['median']:.2f} "
                   f"p90={group_stats[vg1]['p90']:.2f} worst={group_stats[vg1]['worst']:.2f}")

    rs = np.array([c["log_rmse"] for c in per_curve])
    print(f"\nOverall: median={np.median(rs):.2f} p90={np.percentile(rs,90):.2f} "
           f"worst={np.max(rs):.2f}  ({len(rs)} curves)")

    with open(OUT / "summary.json", "w") as f:
        json.dump({
            "VTH0": float(xo[0]), "BJT_AREA": float(10**xo[1]),
            "ALPHA0_mult": float(10**xo[2]) * float(10**xo[7]),
            "Rb": float(10**xo[3]), "U0": float(xo[4]),
            "RDSW": float(xo[5]), "K2": float(xo[6]),
            "median_log_rmse": float(np.median(rs)),
            "worst_log_rmse": float(np.max(rs)),
            "group_stats": group_stats, "per_curve": per_curve,
            "topology": "series_M1_M2",
        }, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    for ax, vg1 in zip(axes, [0.2, 0.4, 0.6]):
        cands = sorted(GROUPS[vg1], key=lambda c: c[1])
        cmap = plt.cm.coolwarm(np.linspace(0, 1, len(cands)))
        for (_, vg2, vd, idd), color in zip(cands, cmap):
            try:
                pred, _ = trace_nsram_series(vg1, vg2, vd, p)
            except Exception: continue
            r = curve_score(pred, idd) or float("nan")
            ax.semilogy(vd, np.clip(idd, 1e-14, None), color=color, lw=1.6,
                          alpha=0.85, label=f"VG2={vg2:+.2f} ({r:.2f})")
            ax.semilogy(vd, np.clip(pred, 1e-22, None), color=color, lw=1.0,
                          ls="--", alpha=0.85)
        s = group_stats.get(vg1, {})
        ax.set_title(f"VG1={vg1} V — median {s.get('median', 0):.2f} dec\n"
                      f"solid=meas, dashed=fit (series M1+M2)",
                      fontsize=11)
        ax.set_xlabel("Vd [V]", fontsize=11)
        ax.set_ylim(1e-13, 1e-3)
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7, ncol=2, loc="lower right")
    axes[0].set_ylabel("|Id| [A]", fontsize=11)
    fig.suptitle(f"Series M1+M2 NSRAM fit  ·  "
                  f"overall median {np.median(rs):.2f} dec  ·  {len(rs)} curves\n"
                  f"VTH0={xo[0]:.3f} K2={xo[6]:+.3f} "
                  f"AREA=1e{xo[1]:.1f} A0=1e{xo[2]+xo[7]:.1f} "
                  f"Rb=1e{xo[3]:.1f} U0={xo[4]:.3f} RDSW={xo[5]:.0f}",
                  fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "all_curves.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT/'all_curves.png'}")


if __name__ == "__main__":
    main()
