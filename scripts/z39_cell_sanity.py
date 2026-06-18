"""z39_cell_sanity.py — verify the fast phenomenological cell captures
the canonical NS-RAM model's qualitative dynamics.

Tests:
  T1: Bistability  — drive VG1 from low→high→low, verify latch + retention
  T2: VG2 plasticity-knob — sweep VG2, verify Vth shift in Id(VG1)
  T3: Switching speed — pulse train, verify Iii→latch in finite steps
  T4: Retention — after pulse off, body decays toward VG2 over many steps

Reference: nsram.nsram_canonical.trace_nsram (slow, accurate)
Test:      nsram.cell_fast.CellArray (fast, phenomenological)

The fast model doesn't have to match ID values, only:
  - Latch happens at similar (VG1, VG2, Vd) regimes
  - VG2 modulates threshold monotonically in same direction
  - Switching/retention timescale ratios are similar
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nsram.cell_fast import CellArray
from nsram.nsram_canonical import trace_nsram, make_pazos_130nm
from dataclasses import replace

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z39_cell_sanity")
OUT.mkdir(parents=True, exist_ok=True)


def t1_bistability_fast():
    """Drive cell with pulse train. Verify latch persists past pulse end."""
    cell = CellArray(N=1, alpha=0.5, VG2=torch.full((1,), 0.20))
    T = 600
    Vb_hist = []; Id_hist = []; vg1_hist = []
    for t in range(T):
        # Pulse train: high VG1+drive in [100,200], OFF [200,300], pulse [300,400], OFF
        if 100 <= t < 200 or 300 <= t < 400:
            VG1, drive = 0.6, 1.0
        else:
            VG1, drive = 0.2, 0.0
        Id = cell.step(VG1, drive)
        Vb_hist.append(cell.Vb.item()); Id_hist.append(Id.item()); vg1_hist.append(VG1)
    return np.array(Vb_hist), np.array(Id_hist), np.array(vg1_hist)


def t2_vg2_threshold_fast():
    """For each VG2, sweep VG1 with high drive, find Vth_eff (latch onset)."""
    vg2_grid = np.linspace(-0.10, 0.40, 11)
    vg1_grid = np.linspace(0.10, 0.80, 30)
    latch_threshold = []
    for vg2 in vg2_grid:
        thr = None
        for vg1 in vg1_grid:
            cell = CellArray(N=1, alpha=0.5, VG2=torch.full((1,), float(vg2)))
            for _ in range(200):
                cell.step(vg1, 1.0)
            if cell.Vb.item() > 0.55:
                thr = vg1
                break
        latch_threshold.append(thr if thr is not None else 1.0)
    return vg2_grid, np.array(latch_threshold)


def t3_switching_speed_fast():
    """Time to latch for various drives."""
    drives = np.logspace(-1, 1, 8)
    times = []
    for d in drives:
        cell = CellArray(N=1, alpha=0.5, VG2=torch.full((1,), 0.20))
        for t in range(2000):
            cell.step(0.6, float(d))
            if cell.Vb.item() > 0.55:
                times.append(t)
                break
        else:
            times.append(np.nan)
    return drives, np.array(times)


def t4_retention_fast():
    """After latching, turn drive off. Body decays toward VG2 (slow)."""
    cell = CellArray(N=1, alpha=0.5, VG2=torch.full((1,), 0.20))
    for _ in range(200): cell.step(0.6, 1.0)
    Vb_initial = cell.Vb.item()
    Vb_decay = []
    for t in range(10000):
        cell.step(0.2, 0.0)
        Vb_decay.append(cell.Vb.item())
    return Vb_initial, np.array(Vb_decay)


def t2_vg2_canonical():
    """Same Vth-shift test on canonical (slow) for comparison."""
    p = replace(make_pazos_130nm(), VTH0=0.30, U0=0.08,
                 BJT_AREA=1e-5, ALPHA0_mult=0.5, Rb=1e10)
    vg2_grid = np.linspace(-0.10, 0.40, 6)
    vg1_grid = np.linspace(0.10, 0.80, 12)
    latch_threshold = []
    for vg2 in vg2_grid:
        thr = None
        for vg1 in vg1_grid:
            try:
                Id, Vbs = trace_nsram(float(vg1), float(vg2),
                                          np.linspace(0.1, 1.0, 8), p)
                if Vbs.max() > 0.55:
                    thr = vg1
                    break
            except Exception:
                continue
        latch_threshold.append(thr if thr is not None else 1.0)
    return vg2_grid, np.array(latch_threshold)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # T1
    Vb, Id, vg1 = t1_bistability_fast()
    ax = axes[0, 0]
    t = np.arange(len(Vb))
    ax.plot(t, Vb, "b-", lw=1.5, label="Vb (body)")
    ax.fill_between(t, 0, vg1, alpha=0.15, color="orange", label="VG1 input")
    ax.axhline(0.55, color="red", ls="--", alpha=0.5, label="latch threshold")
    ax.set_xlabel("step"); ax.set_ylabel("voltage [V]")
    ax.set_title("T1 Bistability — pulse train forces latch")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)
    pulses = (vg1 == 0.6).sum()
    latched_steady = ((Vb > 0.55) & (vg1 == 0.2)).sum()
    print(f"T1: {pulses} pulse steps, {latched_steady} latched-at-OFF steps "
           f"({'PASS' if latched_steady > 50 else 'FAIL'})")

    # T2
    vg2_f, vth_f = t2_vg2_threshold_fast()
    print("\nT2 fast — running canonical for cross-check (slow)...")
    vg2_c, vth_c = t2_vg2_canonical()
    ax = axes[0, 1]
    ax.plot(vg2_f, vth_f, "g-o", lw=2, label="cell_fast")
    ax.plot(vg2_c, vth_c, "k-s", lw=2, label="canonical")
    ax.set_xlabel("VG2 [V]"); ax.set_ylabel("VG1 latch onset [V]")
    ax.set_title("T2 VG2 → Vth shift  (both should slope down)")
    ax.grid(alpha=0.3); ax.legend()
    slope_f = np.polyfit(vg2_f, vth_f, 1)[0]
    slope_c = np.polyfit(vg2_c, vth_c, 1)[0]
    print(f"T2: slope_fast={slope_f:.3f}, slope_canonical={slope_c:.3f}  "
           f"({'PASS — both negative' if slope_f < 0 and slope_c < 0 else 'FAIL'})")

    # T3
    drives, times = t3_switching_speed_fast()
    ax = axes[1, 0]
    ax.loglog(drives, times, "m-o", lw=2)
    ax.set_xlabel("drive amplitude"); ax.set_ylabel("steps to latch")
    ax.set_title(f"T3 Switching speed  (lower drive → more steps)")
    ax.grid(alpha=0.3, which="both")
    monotonic = np.all(np.diff(times[~np.isnan(times)]) <= 0) or \
                  np.corrcoef(np.log(drives), np.log(np.where(np.isnan(times), 1e6, times)))[0,1] < -0.5
    print(f"T3: drive↑ → time↓  ({'PASS' if monotonic else 'FAIL'})")

    # T4
    Vb_init, Vb_decay = t4_retention_fast()
    ax = axes[1, 1]
    ax.plot(Vb_decay, "r-", lw=1.5, label="Vb (body)")
    ax.axhline(0.20, color="black", ls=":", alpha=0.5, label="VG2 (target)")
    ax.axhline(0.55, color="red", ls="--", alpha=0.5, label="latch threshold")
    ax.set_xlabel("step (drive=0)"); ax.set_ylabel("Vb [V]")
    final = Vb_decay[-1]
    ax.set_title(f"T4 Retention — Vb {Vb_init:.2f} → {final:.2f} after {len(Vb_decay)} steps")
    ax.grid(alpha=0.3); ax.legend()
    decayed = abs(final - 0.20) < 0.10
    monotonic = (np.diff(Vb_decay) <= 1e-6).all()
    print(f"T4: Vb {Vb_init:.3f} → {final:.3f} (target 0.20, monotonic={monotonic})  "
           f"({'PASS' if decayed and monotonic else 'FAIL'})")

    fig.suptitle("z39 — cell_fast sanity (qualitative match to canonical)",
                  fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "sanity.png", dpi=130)
    plt.close(fig)
    print(f"\nWrote {OUT/'sanity.png'}")


if __name__ == "__main__":
    main()
