"""z45_calibrate_from_data.py — calibrate cell_fast against Sebas's
real I-V hysteresis data (data/sebas_2026_04_22).

Sebas's measurement: slow Vd ramp 0 → Vmax → 0 at fixed (VG1, VG2).  Up-sweep
and down-sweep differ when the cell is bistable, giving a hysteresis loop.

Quantitative anchors we extract per curve:
  • Vd_up      : snapback voltage (impact ionization onset)
  • Vd_down    : release voltage (BJT pull breaks the latch)
  • H = Vd_up − Vd_down   : hysteresis width (bistability strength)
  • Id_high / Id_low      : ratio between latched and unlatched current
  • slope of Vd_up vs VG2 → dVth/dVG2 (back-gate coupling K_back)

We then optimize cell_fast parameters so its Vd-sweep produces matching
Vd_up, Vd_down, hysteresis width and current ratio across the same
(VG1, VG2) grid that Sebas measured.

This calibrates the SUBSTRATE.  Switching/retention TIME constants still
need a transient measurement (Sebas/Robert NN) — flagged at the end.
"""
from __future__ import annotations
import csv, json, re, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize

from nsram.cell_fast import CellArray

DATA = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
             "data/sebas_2026_04_22")
OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z45_calibrate_from_data")
OUT.mkdir(parents=True, exist_ok=True)
VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")


# ─────────────────────────────────────────────────────────────────────
# Step 1: load and split into up + down sweep
# ─────────────────────────────────────────────────────────────────────

def load_curve_with_sweep(path: Path):
    """Read CSV, return (Vd_up, Id_up, Vd_down, Id_down) arrays.

    Down-sweep is returned in TIME order (Vd descending), so detect_release
    can find the Id drop as Vd is reduced.
    """
    rows = []
    with open(path) as f:
        rdr = csv.reader(f); next(rdr)
        for r in rdr:
            try:
                rows.append((float(r[2]), float(r[0]), float(r[1])))
            except ValueError: continue
    rows.sort()             # by time
    Vd = np.array([r[1] for r in rows]); Id = np.array([r[2] for r in rows])
    if len(Vd) < 20: return None
    peak = int(np.argmax(Vd))
    # Up-sweep: Vd ascending (time 0 → peak)
    # Down-sweep: Vd descending (time peak → end)  — kept in time order!
    return Vd[:peak+1], Id[:peak+1], Vd[peak:], Id[peak:]


def load_all_curves():
    curves = []
    for sub in sorted(DATA.iterdir()):
        if not sub.is_dir(): continue
        for fn in sorted(sub.glob("*.csv")):
            m = VG_RE.search(fn.name)
            if not m: continue
            vg2 = float(m.group(1)); vg1 = float(m.group(2))
            x = load_curve_with_sweep(fn)
            if x is None: continue
            curves.append((vg1, vg2) + x)
    return curves


# ─────────────────────────────────────────────────────────────────────
# Step 2: extract anchors from each curve
# ─────────────────────────────────────────────────────────────────────

def detect_snapback(Vd, Id, factor=5.0, vd_min=0.20):
    """Find Vd at which Id increases by `factor` between consecutive points.
    Skips initial channel turn-on (Vd < vd_min) which is not a snapback.
    Returns Vd_jump or None."""
    if len(Vd) < 5: return None
    log_Id = np.log10(np.clip(Id, 1e-13, None))
    dlog = np.diff(log_Id)
    # Mask: only consider jumps where the SECOND point is past vd_min
    valid = Vd[1:] >= vd_min
    if not valid.any(): return None
    masked = np.where(valid, dlog, -np.inf)
    if masked.max() < np.log10(factor): return None
    i = int(np.argmax(masked))
    return float(Vd[i + 1])


def latch_low_high(Vd, Id, jump_v):
    """Median Id below jump (low state) and at peak Vd (high state)."""
    mask_low = Vd < jump_v - 0.05
    mask_high = Vd > Vd.max() - 0.1
    if mask_low.sum() < 3 or mask_high.sum() < 2: return None, None
    return float(np.median(Id[mask_low])), float(np.median(Id[mask_high]))


def extract_anchors(curves):
    anchors = []
    for vg1, vg2, Vd_u, Id_u, Vd_d, Id_d in curves:
        Vu = detect_snapback(Vd_u, Id_u, factor=5.0)
        # For down-sweep, "release" = where Id drops by ≥5×
        if Vd_d.size > 5:
            log_d = np.log10(np.clip(Id_d, 1e-13, None))
            ddec = -np.diff(log_d)
            Vd_release = float(Vd_d[int(np.argmax(ddec)) + 1]) if ddec.max() > np.log10(3) else None
        else:
            Vd_release = None
        Ilo, Ihi = latch_low_high(Vd_u, Id_u, Vu) if Vu else (None, None)
        anchors.append({
            "VG1": vg1, "VG2": vg2,
            "Vd_up": Vu, "Vd_down": Vd_release,
            "hysteresis": (Vu - Vd_release) if (Vu and Vd_release) else None,
            "I_low": Ilo, "I_high": Ihi,
            "ratio": (Ihi / Ilo) if (Ihi and Ilo and Ilo > 0) else None,
        })
    return anchors


# ─────────────────────────────────────────────────────────────────────
# Step 3: cell_fast forward — produce comparable (Vd_up, Vd_down) anchors
# ─────────────────────────────────────────────────────────────────────

def cell_fast_hysteresis(VG1, VG2, params, Vd_max=2.5, n_steps=40,
                            step_per_vd=15):
    """Simulate Vd up-sweep and down-sweep with cell_fast model.
    Reduced step counts for speed; sufficient for snapback detection.
    """
    Vd_grid_up = np.linspace(0.05, Vd_max, n_steps)
    Vd_grid_dn = Vd_grid_up[::-1]
    cell = CellArray(N=1, alpha=1.0,
                          VG2=torch.tensor([float(VG2)]),
                          VTH0=params["VTH0"],
                          K_back=params["K_back"],
                          A_iii=params["A_iii"],
                          G_bjt=params["G_bjt"],
                          V_bjt_on=params["V_bjt_on"],
                          V_latch=params["V_latch"],
                          K_leak=params["K_leak"])
    Id_up = []; Id_dn = []
    for Vd in Vd_grid_up:
        drive = max(Vd, 0.0) * params.get("drive_scale", 1.0)
        for _ in range(step_per_vd):
            cell.step(VG1, drive)
        Id_up.append(cell.read(VG1).item())
    for Vd in Vd_grid_dn:
        drive = max(Vd, 0.0) * params.get("drive_scale", 1.0)
        for _ in range(step_per_vd):
            cell.step(VG1, drive)
        Id_dn.append(cell.read(VG1).item())
    return (np.array(Vd_grid_up), np.array(Id_up),
            np.array(Vd_grid_dn), np.array(Id_dn))


def cell_fast_hysteresis_batch(VG1_arr, VG2_arr, params,
                                  Vd_max=2.5, n_steps=40, step_per_vd=15):
    """Batched: simulate all anchors in parallel as one CellArray of N cells.

    VG1_arr, VG2_arr: shape (N,)
    Returns Vd_grid_up (n_steps,), Id_up (n_steps, N), Vd_grid_dn, Id_dn.
    """
    N = len(VG1_arr)
    VG1_t = torch.tensor(VG1_arr, dtype=torch.float64)
    VG2_t = torch.tensor(VG2_arr, dtype=torch.float64)
    cell = CellArray(N=N, alpha=1.0, VG2=VG2_t,
                          VTH0=params["VTH0"],
                          K_back=params["K_back"],
                          A_iii=params["A_iii"],
                          G_bjt=params["G_bjt"],
                          V_bjt_on=params["V_bjt_on"],
                          V_latch=params["V_latch"],
                          K_leak=params["K_leak"])
    Vd_grid_up = np.linspace(0.05, Vd_max, n_steps)
    Vd_grid_dn = Vd_grid_up[::-1]
    Id_up = np.zeros((n_steps, N))
    Id_dn = np.zeros((n_steps, N))
    drive_scale = params.get("drive_scale", 1.0)
    for i, Vd in enumerate(Vd_grid_up):
        drive = float(Vd) * drive_scale
        drive_t = torch.full((N,), drive)
        for _ in range(step_per_vd):
            cell.step(VG1_t, drive_t)
        Id_up[i] = cell.read(VG1_t).numpy()
    for i, Vd in enumerate(Vd_grid_dn):
        drive = float(Vd) * drive_scale
        drive_t = torch.full((N,), drive)
        for _ in range(step_per_vd):
            cell.step(VG1_t, drive_t)
        Id_dn[i] = cell.read(VG1_t).numpy()
    return Vd_grid_up, Id_up, Vd_grid_dn, Id_dn


def model_anchors_for(VG1, VG2, params):
    Vu, Iu, Vd, Id_d = cell_fast_hysteresis(VG1, VG2, params)
    Vsnap = detect_snapback(Vu, Iu, factor=5.0)
    if Vd.size > 5:
        log_d = np.log10(np.clip(Id_d, 1e-13, None))
        ddec = -np.diff(log_d)
        Vrel = float(Vd[int(np.argmax(ddec)) + 1]) if ddec.max() > np.log10(3) else None
    else:
        Vrel = None
    Ilo, Ihi = latch_low_high(Vu, Iu, Vsnap) if Vsnap else (None, None)
    return {"Vd_up": Vsnap, "Vd_down": Vrel,
             "hysteresis": (Vsnap - Vrel) if (Vsnap and Vrel) else None,
             "I_low": Ilo, "I_high": Ihi,
             "ratio": (Ihi / Ilo) if (Ihi and Ilo and Ilo > 0) else None}


# ─────────────────────────────────────────────────────────────────────
# Step 4: fit cell_fast params to data anchors
# ─────────────────────────────────────────────────────────────────────

def anchor_loss(model, data):
    """L2 loss on the four anchors that exist in both."""
    parts = []
    for k in ("Vd_up", "Vd_down", "hysteresis"):
        if model.get(k) is not None and data.get(k) is not None:
            parts.append((model[k] - data[k]) ** 2)
    if model.get("ratio") is not None and data.get("ratio") is not None:
        parts.append((np.log10(model["ratio"]) - np.log10(data["ratio"])) ** 2)
    return float(np.mean(parts)) if parts else 1.0


def total_loss(x, anchors_data):
    """Vector x packs cell_fast params (log-scaled where useful).
    Batched version: all anchors in one parallel sweep."""
    p = {
        "VTH0": x[0],
        "K_back": x[1],
        "A_iii": float(10**x[2]),
        "G_bjt": float(10**x[3]),
        "V_bjt_on": x[4],
        "V_latch": x[5],
        "K_leak": float(10**x[6]),
        "drive_scale": float(10**x[7]),
    }
    valid = [d for d in anchors_data if d["Vd_up"] is not None]
    if not valid: return 10.0
    VG1_arr = np.array([d["VG1"] for d in valid])
    VG2_arr = np.array([d["VG2"] for d in valid])
    Vu, Iu, Vd, Idn = cell_fast_hysteresis_batch(VG1_arr, VG2_arr, p)
    losses = []
    for j, d in enumerate(valid):
        Vsnap = detect_snapback(Vu, Iu[:, j], factor=5.0, vd_min=0.20)
        log_d = np.log10(np.clip(Idn[:, j], 1e-13, None))
        ddec = -np.diff(log_d)
        Vrel = float(Vd[int(np.argmax(ddec)) + 1]) if ddec.max() > np.log10(3) else None
        Ilo, Ihi = latch_low_high(Vu, Iu[:, j], Vsnap) if Vsnap else (None, None)
        m = {"Vd_up": Vsnap, "Vd_down": Vrel,
              "hysteresis": (Vsnap - Vrel) if (Vsnap and Vrel) else None,
              "I_low": Ilo, "I_high": Ihi,
              "ratio": (Ihi / Ilo) if (Ihi and Ilo and Ilo > 0) else None}
        losses.append(anchor_loss(m, d))
    return float(np.mean(losses)) if losses else 10.0


def main():
    print("Loading Sebas's data...")
    curves = load_all_curves()
    print(f"  Loaded {len(curves)} curves with up+down sweeps")
    anchors = extract_anchors(curves)
    has_jump = sum(1 for a in anchors if a["Vd_up"] is not None)
    has_hyst = sum(1 for a in anchors if a["hysteresis"] is not None)
    has_ratio = sum(1 for a in anchors if a["ratio"] is not None)
    print(f"  detected snapback in {has_jump}/{len(anchors)} curves")
    print(f"  detected hysteresis in {has_hyst}/{len(anchors)} curves")
    print(f"  detected high/low ratio in {has_ratio}/{len(anchors)} curves")

    # K_back from data: linear fit of Vd_up vs VG2 grouped by VG1
    print("\n=== Data-extracted anchors ===")
    by_vg1 = {}
    for a in anchors:
        if a["Vd_up"] is None: continue
        by_vg1.setdefault(round(a["VG1"], 2), []).append(a)
    K_back_estimates = []
    for vg1, lst in by_vg1.items():
        vg2s = np.array([a["VG2"] for a in lst])
        vsn = np.array([a["Vd_up"] for a in lst])
        if len(vg2s) >= 3:
            slope = np.polyfit(vg2s, vsn, 1)[0]
            K_back_estimates.append(slope)
            print(f"  VG1={vg1}: dVd_up/dVG2 = {slope:+.3f}  ({len(vg2s)} pts)")
    if K_back_estimates:
        K_back_data = float(-np.mean(K_back_estimates))   # negate: Vth shift opposite to Vd_up shift
        print(f"\n  → estimated |K_back| from data ≈ {abs(K_back_data):.3f}")

    # Save data anchors
    with open(OUT / "data_anchors.json", "w") as f:
        json.dump([{k: v for k, v in a.items()} for a in anchors],
                    f, indent=2, default=str)

    # Optimize cell_fast params
    print("\n=== Calibrating cell_fast against data anchors ===")
    bounds = [
        (0.20, 0.65),         # VTH0
        (-2.0, 2.0),          # K_back  — sign determined by data, allow negative
        (-1.0, 2.0),          # log10 A_iii
        (-1.5, 1.5),          # log10 G_bjt
        (0.55, 0.85),         # V_bjt_on
        (0.35, 0.70),         # V_latch
        (-3.0, -0.3),         # log10 K_leak
        (-2.0, 1.5),          # log10 drive_scale
    ]
    # Initial guess — start K_back NEGATIVE since data shows positive
    # dVd_up/dVG2 (high VG2 = harder to latch).
    x0 = np.array([0.40, -1.0, np.log10(5.0), np.log10(1.0),
                     0.75, 0.55, np.log10(0.02), np.log10(1.0)])
    t0 = time.time()
    print("  initial loss:", total_loss(x0, anchors))
    # Use Nelder-Mead (no gradient available)
    res = minimize(total_loss, x0, args=(anchors,),
                     method="Nelder-Mead",
                     options={"maxiter": 80, "xatol": 1e-3, "fatol": 1e-4,
                                "disp": True})
    print(f"  optimization done in {time.time()-t0:.0f}s, final loss={res.fun:.4f}")
    x = res.x
    final = {
        "VTH0": float(x[0]), "K_back": float(x[1]),
        "A_iii": float(10**x[2]), "G_bjt": float(10**x[3]),
        "V_bjt_on": float(x[4]), "V_latch": float(x[5]),
        "K_leak": float(10**x[6]), "drive_scale": float(10**x[7]),
    }
    print("\n=== Calibrated parameters ===")
    for k, v in final.items():
        print(f"  {k:14s}  {v:.4f}")
    with open(OUT / "calibrated_params.json", "w") as f:
        json.dump(final, f, indent=2)

    # Plot: data vs model anchors
    print("\n=== Generating overlay plots ===")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # Panel 1: Vd_up vs VG2 grouped by VG1
    ax = axes[0]
    cmap = {0.2: "blue", 0.4: "green", 0.6: "red"}
    for vg1, lst in by_vg1.items():
        vg2s = np.array([a["VG2"] for a in lst])
        vsn = np.array([a["Vd_up"] for a in lst])
        ax.scatter(vg2s, vsn, color=cmap.get(round(vg1, 2), "gray"),
                      label=f"data VG1={vg1}", s=40, marker="o")
    # Model
    for vg1 in [0.2, 0.4, 0.6]:
        vg2_grid = np.linspace(-0.20, 0.30, 8)
        vu = []
        for vg2 in vg2_grid:
            m = model_anchors_for(vg1, vg2, final)
            vu.append(m["Vd_up"])
        valid = [(v, u) for v, u in zip(vg2_grid, vu) if u is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, "--", color=cmap[vg1], label=f"model VG1={vg1}", lw=2)
    ax.set_xlabel("VG2 [V]"); ax.set_ylabel("Vd_up (snapback) [V]")
    ax.set_title("Snapback voltage vs VG2  (← K_back fit)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: hysteresis width
    ax = axes[1]
    hd = [a["hysteresis"] for a in anchors if a["hysteresis"] is not None]
    if hd:
        ax.hist(hd, bins=15, alpha=0.7, color="#3498db", label="data",
                  edgecolor="black")
    # Model histogram
    hm = []
    for vg1 in [0.2, 0.4, 0.6]:
        for vg2 in np.linspace(-0.20, 0.30, 8):
            m = model_anchors_for(vg1, vg2, final)
            if m["hysteresis"] is not None: hm.append(m["hysteresis"])
    if hm:
        ax.hist(hm, bins=15, alpha=0.5, color="#e67e22", label="model",
                  edgecolor="black")
    ax.set_xlabel("hysteresis width Vd_up - Vd_down [V]")
    ax.set_ylabel("# curves")
    ax.set_title("Hysteresis width — data vs model")
    ax.legend(); ax.grid(alpha=0.3)

    # Panel 3: high/low Id ratio
    ax = axes[2]
    rd = [a["ratio"] for a in anchors if a["ratio"] is not None]
    if rd:
        ax.hist(np.log10(rd), bins=15, alpha=0.7, color="#3498db",
                  label="data", edgecolor="black")
    rm = []
    for vg1 in [0.2, 0.4, 0.6]:
        for vg2 in np.linspace(-0.20, 0.30, 8):
            m = model_anchors_for(vg1, vg2, final)
            if m["ratio"] is not None: rm.append(m["ratio"])
    if rm:
        ax.hist(np.log10(rm), bins=15, alpha=0.5, color="#e67e22",
                  label="model", edgecolor="black")
    ax.set_xlabel("log10( Id_high / Id_low )")
    ax.set_ylabel("# curves")
    ax.set_title("Latched/unlatched ratio — data vs model")
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(f"z45 — cell_fast calibrated against Sebas's hysteresis data\n"
                  f"VTH0={final['VTH0']:.2f}  K_back={final['K_back']:.2f}  "
                  f"A_iii={final['A_iii']:.1f}  G_bjt={final['G_bjt']:.2f}  "
                  f"loss={res.fun:.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {OUT/'calibration.png'}")

    # Final advisory note
    print("\n" + "="*70)
    print("CALIBRATION COMPLETE — STATIC ANCHORS ONLY")
    print("="*70)
    print("Calibrated:    K_back, snapback Vd_up, release Vd_down,")
    print("               hysteresis width, latched/unlatched current ratio")
    print("NOT calibrated: switching time τ_sw, retention τ_ret")
    print("                — these need transient measurements (V or I vs time")
    print("                  during a write pulse).  Ask Sebas/Robert.")
    print("="*70)


if __name__ == "__main__":
    main()
