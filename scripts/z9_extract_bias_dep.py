"""z9_extract_bias_dep.py — extract the missing physics from data itself.

Scientific method for finding unknown bias-dependent physics:
  1. Fit ONE effective parameter per curve (here: ALPHA0_eff), holding
     every other parameter at our best physics-identified value.
  2. Extract 33 values of ALPHA0_eff (one per VG1, VG2).
  3. Plot ALPHA0_eff vs VG1 and vs VG2. Look for PATTERN:
       - Smooth exponential in VG1    → Vgs-dependent impact ionization
       - Linear in VG2                → layout/M2 coupling
       - Chaotic                      → device-to-device variation
  4. If the pattern is simple and physical → encode in model.
     If not, we know the residual is genuinely unstructured and
     polynomial regression à la Sebas is the best we can do.

This is how Sebas's "polynomial dependence of model parameters with
tuning voltages" GETS DISCOVERED — not assumed. We might find the
dependence is NOT polynomial but something more physical.
"""
from __future__ import annotations

import csv, json, re, time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from nsram.bsim4 import (BSIM4_PRESETS, bipolar_collector_current_ss,
                          drain_current_bsim, gidl_current,
                          impact_ionization_bsim4)
from nsram.physics import thermal_voltage


DATA_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
                "data/sebas_2026_04_22")
OUT_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
               "results/z9_extract_bias_dep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Best-so-far physics-identified parameters (from z8b/z8c/z8d)
PB = replace(BSIM4_PRESETS["ns_ram_130nm_pazos"],
              VTH0=0.825, BETA0=18.0, BJT_AREA=1e-4, Rb=1e8)
Vt_th = thermal_voltage(300.0)
VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")


def kcl_net(Vb, Vg1, Vd, p):
    Iii = float(impact_ionization_bsim4(Vg1, Vd, Vb, p))
    Igidl = float(gidl_current(Vd, Vg1, Vb, p)) if p.AGIDL > 0 else 0.0
    IS_eff = p.BJT_IS * max(p.BJT_AREA, 1e-30)
    Vb_c = min(Vb, p.BJT_VJE * 1.1)
    exp_arg = np.clip(Vb_c / (p.BJT_NE * Vt_th), -60, 60)
    Ib_out = (IS_eff / p.BJT_BF) * (np.exp(exp_arg) - 1.0) + Vb / p.Rb
    return (Iii + Igidl) - Ib_out


def find_vb(Vg1, Vd, p, Vb_init, vb_max=0.85):
    lo = max(0.0, Vb_init - 0.02)
    hi = min(vb_max, max(Vb_init + 0.02, 0.02))
    try:
        if kcl_net(lo, Vg1, Vd, p) * kcl_net(hi, Vg1, Vd, p) <= 0:
            return float(brentq(kcl_net, lo, hi, args=(Vg1, Vd, p),
                                  xtol=1e-6, rtol=1e-6))
    except ValueError:
        pass
    g = np.linspace(0.0, vb_max, 81)
    fs = np.array([kcl_net(v, Vg1, Vd, p) for v in g])
    sc = np.where(np.sign(fs[:-1]) != np.sign(fs[1:]))[0]
    if len(sc) == 0:
        return float(vb_max)
    roots = []
    for i in sc:
        try:
            roots.append(brentq(kcl_net, g[i], g[i+1],
                                  args=(Vg1, Vd, p), xtol=1e-6, rtol=1e-6))
        except ValueError:
            continue
    if not roots:
        return float(vb_max)
    return float(min(roots) if Vb_init < 0.1
                 else min(roots, key=lambda r: abs(r - Vb_init)))


def predict(Vg1, Vds, p):
    Vb = 0.0
    Vbs = np.zeros_like(Vds)
    for k, v in enumerate(Vds):
        Vb = find_vb(Vg1, float(v), p, Vb_init=Vb)
        Vbs[k] = Vb
    Id, _ = drain_current_bsim(Vg1, Vds, Vbs, p)
    Ic = bipolar_collector_current_ss(Vg1, Vds, Vbs, p)
    return np.asarray(Id) + np.asarray(Ic)


def per_curve_rmse(alpha0, vg1, vds, idd):
    p = replace(PB, ALPHA0=alpha0)
    pred = predict(vg1, vds, p)
    m = (idd > 1e-13) & (pred > 0)
    if m.sum() < 5:
        return 1e6
    lm = np.log10(idd[m]); lp = np.log10(pred[m])
    return float(np.sqrt(np.mean((lm - lp) ** 2)))


def fit_alpha0(vg1, vds, idd):
    """Find ALPHA0_eff per curve by scanning log-space."""
    log_a_grid = np.linspace(-8, -2, 25)           # 1e-8 .. 1e-2
    errs = np.array([per_curve_rmse(10**la, vg1, vds, idd)
                      for la in log_a_grid])
    best = int(np.argmin(errs))
    # Refine with Brent
    lo = log_a_grid[max(0, best - 1)]
    hi = log_a_grid[min(len(log_a_grid) - 1, best + 1)]
    if hi - lo < 1e-3:
        return float(10**log_a_grid[best]), float(errs[best])
    res = minimize_scalar(
        lambda la: per_curve_rmse(10**float(la), vg1, vds, idd),
        bounds=(lo, hi), method="bounded", options={"xatol": 1e-3},
    )
    return float(10**float(res.x)), float(res.fun)


def load_curves(n_ds=25):
    curves = []
    for sub in sorted(DATA_DIR.iterdir()):
        if not sub.is_dir(): continue
        for fn in sorted(sub.glob("*.csv")):
            m = VG_RE.search(fn.name)
            if not m: continue
            rows = []
            with open(fn) as f:
                rdr = csv.reader(f); next(rdr)
                for r in rdr:
                    try: rows.append((float(r[2]), float(r[0]), float(r[1])))
                    except ValueError: continue
            rows.sort()
            Vd = np.array([r[1] for r in rows])
            Id = np.array([r[2] for r in rows])
            peak = int(np.argmax(Vd))
            Vd = Vd[:peak + 1]; Id = Id[:peak + 1]
            new_vd = np.linspace(0.1, float(Vd.max()), n_ds)
            new_id = np.interp(new_vd, Vd, Id)
            curves.append((float(m.group(2)), float(m.group(1)),
                            new_vd, new_id, fn.name))
    return curves


def main():
    curves = load_curves(25)
    print(f"{len(curves)} curves — extracting ALPHA0_eff per curve\n")

    t0 = time.time()
    results = []
    for vg1, vg2, vds, idd, fn in curves:
        a0, rmse = fit_alpha0(vg1, vds, idd)
        results.append({"vg1": vg1, "vg2": vg2, "file": fn,
                         "ALPHA0_eff": a0, "log10_A0": np.log10(a0),
                         "rmse_per_curve": rmse})
        print(f"  VG1={vg1:.1f} VG2={vg2:+.2f}: "
              f"ALPHA0={a0:.2e}  rmse={rmse:.2f}  "
              f"({time.time()-t0:.0f}s)")

    # Aggregate
    A0s = np.array([r["ALPHA0_eff"] for r in results])
    log_A0s = np.log10(A0s)
    rmses = np.array([r["rmse_per_curve"] for r in results])
    vg1s = np.array([r["vg1"] for r in results])
    vg2s = np.array([r["vg2"] for r in results])

    print(f"\n═══ Extracted ALPHA0_eff stats ═══")
    print(f"  range    : {A0s.min():.2e} .. {A0s.max():.2e}")
    print(f"  PTM130   : {BSIM4_PRESETS['ns_ram_130nm_pazos'].ALPHA0:.2e}")
    print(f"  ratio max/min: {A0s.max()/A0s.min():.1e}  (if ~1 → no bias-dep)")
    print(f"  per-curve RMSE median: {np.median(rmses):.2f} dec")
    print(f"  per-curve RMSE p90:    {np.percentile(rmses, 90):.2f} dec")

    # Is it monotone in VG1?
    for vg1_val in sorted(set(vg1s)):
        mask = vg1s == vg1_val
        a0_at_vg1 = A0s[mask]
        print(f"  VG1={vg1_val}: ALPHA0_eff ∈ [{a0_at_vg1.min():.2e}, "
              f"{a0_at_vg1.max():.2e}]  median={np.median(a0_at_vg1):.2e}")

    # Linear regression log(A0) vs (VG1, VG2)
    X = np.stack([np.ones_like(vg1s), vg1s, vg2s, vg1s*vg2s], axis=1)
    beta, *_ = np.linalg.lstsq(X, log_A0s, rcond=None)
    pred_log = X @ beta
    ss_res = np.sum((log_A0s - pred_log)**2)
    ss_tot = np.sum((log_A0s - log_A0s.mean())**2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-30)
    print(f"\n═══ Linear fit log(ALPHA0_eff) = c0 + c1·VG1 + c2·VG2 + c3·VG1·VG2 ═══")
    print(f"  c0        = {beta[0]:+.3f}")
    print(f"  c1 (VG1)  = {beta[1]:+.3f}  → factor of 10^{beta[1]*0.4:+.2f} per 0.4V VG1 swing")
    print(f"  c2 (VG2)  = {beta[2]:+.3f}")
    print(f"  c3 (cross)= {beta[3]:+.3f}")
    print(f"  R²        = {r2:.3f}  (1.0 = perfect linear pattern)")

    with open(OUT_DIR / "per_curve.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "n_curves": len(results),
            "ALPHA0_range": [float(A0s.min()), float(A0s.max())],
            "ALPHA0_ratio": float(A0s.max() / A0s.min()),
            "PTM130_ALPHA0": float(BSIM4_PRESETS['ns_ram_130nm_pazos'].ALPHA0),
            "median_rmse_per_curve": float(np.median(rmses)),
            "p90_rmse_per_curve": float(np.percentile(rmses, 90)),
            "log10_ALPHA0_lstsq": {
                "c0_constant":   float(beta[0]),
                "c1_VG1":        float(beta[1]),
                "c2_VG2":        float(beta[2]),
                "c3_VG1_VG2":    float(beta[3]),
                "R2":            float(r2),
            },
        }, f, indent=2)

    # Plot: ALPHA0_eff vs VG1 colored by VG2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(vg2s.min(), vg2s.max())
    sc = axes[0].scatter(vg1s, A0s, c=vg2s, cmap=cmap, norm=norm, s=60, edgecolor="k")
    axes[0].set_yscale("log")
    axes[0].axhline(BSIM4_PRESETS['ns_ram_130nm_pazos'].ALPHA0,
                      color="grey", ls="--", lw=0.8, label="PTM130 card")
    axes[0].set_xlabel("VG1 [V]"); axes[0].set_ylabel("ALPHA0_eff")
    axes[0].set_title("Extracted ALPHA0 vs VG1 (colored by VG2)")
    axes[0].grid(alpha=0.3, which="both"); axes[0].legend()
    plt.colorbar(sc, ax=axes[0], label="VG2 [V]")

    sc2 = axes[1].scatter(vg2s, A0s, c=vg1s, cmap="viridis", s=60,
                            edgecolor="k")
    axes[1].set_yscale("log")
    axes[1].axhline(BSIM4_PRESETS['ns_ram_130nm_pazos'].ALPHA0,
                      color="grey", ls="--", lw=0.8)
    axes[1].set_xlabel("VG2 [V]"); axes[1].set_ylabel("ALPHA0_eff")
    axes[1].set_title("Extracted ALPHA0 vs VG2 (colored by VG1)")
    axes[1].grid(alpha=0.3, which="both")
    plt.colorbar(sc2, ax=axes[1], label="VG1 [V]")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha0_patterns.png", dpi=140)
    plt.close(fig)

    # 3D-ish: both
    fig, ax = plt.subplots(figsize=(7, 5))
    vg1_unique = sorted(set(vg1s))
    colors = {0.2: "tab:blue", 0.4: "tab:orange", 0.6: "tab:green"}
    for v1 in vg1_unique:
        m = vg1s == v1
        order = np.argsort(vg2s[m])
        ax.plot(vg2s[m][order], A0s[m][order], "o-",
                color=colors.get(v1, "k"), label=f"VG1={v1} V", lw=1.5)
    ax.set_yscale("log")
    ax.axhline(BSIM4_PRESETS['ns_ram_130nm_pazos'].ALPHA0,
                color="grey", ls="--", lw=0.8, label="PTM130 card")
    ax.set_xlabel("VG2 [V]"); ax.set_ylabel("ALPHA0_eff")
    ax.set_title("How much impact ionization the data demands — per bias point")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha0_surface.png", dpi=140)
    plt.close(fig)

    print(f"\nWrote alpha0_patterns.png, alpha0_surface.png, summary.json")


if __name__ == "__main__":
    main()
