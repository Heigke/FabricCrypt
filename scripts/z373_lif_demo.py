#!/usr/bin/env python3
"""z373: LIF-like dynamics demo for NS-RAM cells.

Drives N=100 NS-RAM cells with independent Poisson spike trains (200Hz, 1s),
records Vb(t), threshold-cross spike events, and ISI statistics. Plots
3 subplots: representative Vb traces, raster, ISI distribution.

Uses S2b LUT (IiiNetLUT) directly with per-cell heterogeneous VG1/VG2/Cb
and sparse spike→VG2 recurrent kick (matching DS_N10_reservoir.py style).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT / "scripts"))
from S2b_transient import IiiNetLUT  # noqa: E402

OUT = ROOT / "results" / "z373_lif_demo"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    rng = np.random.default_rng(0)
    N = 100
    T_sec = 1.0
    rate_hz = 200.0
    dt_input = 1e-3            # 1 ms input bin
    T_steps = int(T_sec / dt_input)  # 1000 input steps

    # Sub-stepping for analog Vb integration
    sub_steps = 10
    dt_sub = dt_input / sub_steps  # 1e-4 s

    # Per-cell heterogeneity
    VG1 = rng.uniform(0.60, 0.78, size=N)
    VG2_base = rng.uniform(0.35, 0.55, size=N)
    # tau diversity via Cb: tau ~ Cb / g_leak. Use Cb in [2fF, 80fF] to span ~0.1-10ms
    Cb_F = rng.uniform(2e-15, 80e-15, size=N)
    inv_Cb = 1.0 / Cb_F

    # Sparse recurrent W: spike → VG2 kick (density 0.05)
    density = 0.05
    nnz = max(int(density * N * N), N)
    rows = rng.integers(0, N, size=nnz)
    cols = rng.integers(0, N, size=nnz)
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    vals = rng.standard_normal(rows.size) * 0.5
    W = sp.csr_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    fb_gain = 0.02  # small recurrent kick so dynamics are mostly input-driven

    V_th = 0.58
    V_reset = 0.30
    T_ref_sub = 30  # refractory in sub-steps (~3 ms)
    # NOTE: with current S2b LUT the body-current Inet is nearly Vd-independent
    # over [0.25, 1.5] V and stays ~170 pA at Vb in [0.30, 0.50]. This makes
    # cells INTRINSIC OSCILLATORS whose firing rate is set by Cb (which sets
    # dVb/dt) and T_ref, not by the Poisson input. We log this in summary.

    # Hold a lower resting Vd; input spike briefly raises it. Strong LUT current
    # means cells would otherwise saturate at the refractory ceiling, so keep
    # bias subthreshold and rely on Poisson input to push them across V_th.
    Vd_bias = 0.28
    Vd_input_amp = 1.6  # input spike pulse amplitude on Vd
    W_in = rng.uniform(0.5, 1.0, size=N)  # all positive: input excites
    # CONTROL: cells [0..9] receive NO input drive — used to test whether
    # the firing rate is input-driven or intrinsic.
    ctrl_idx = np.arange(10)
    W_in[ctrl_idx] = 0.0
    input_drive_idx = np.arange(10, N)

    # Build Poisson spike trains: per-cell independent. 200Hz over 1s.
    # Use input-step binary spike train via bernoulli with p = rate*dt
    p_spike = rate_hz * dt_input
    input_spikes = (rng.random(size=(T_steps, N)) < p_spike).astype(np.float64)

    lut = IiiNetLUT()

    # State
    Vb = np.full(N, V_reset, dtype=np.float64)
    refr = np.zeros(N, dtype=np.int32)
    spike_pulse = np.zeros(N, dtype=np.float64)

    # Logging
    Vb_log = np.empty((T_steps, N), dtype=np.float32)
    spike_times = [[] for _ in range(N)]  # in seconds

    for ti in range(T_steps):
        u_t = input_spikes[ti]  # shape (N,)
        # Recurrent VG2 modulation from prior spike pulses
        VG2_t = VG2_base + fb_gain * (W @ spike_pulse)
        np.clip(VG2_t, -0.1, 0.6, out=VG2_t)
        spike_pulse[:] = 0.0

        # Input → Vd: pulse drives Vd HIGH only during the 1 ms when an input
        # spike arrives; otherwise Vd drops to a sub-threshold rest value that
        # produces near-zero net LUT current at Vb=V_reset. This decouples the
        # firing rate from T_ref and lets ISI structure emerge from Poisson.
        Vd_rest = 0.30
        Vd_pulse = np.where(u_t > 0, Vd_bias + Vd_input_amp * W_in, Vd_rest)
        np.clip(Vd_pulse, 0.25, 3.0, out=Vd_pulse)
        Vd_rest_arr = np.full(N, Vd_rest)

        for s in range(sub_steps):
            # Vd is high only during the first sub-step (~0.1 ms pulse),
            # then drops to rest. This mimics a short EPSP rather than a
            # 1 ms square wave that would force multi-firing per input.
            Vd = Vd_pulse if s < 2 else Vd_rest_arr
            Inet = lut(VG1, VG2_t, Vd, Vb)
            dVb = (Inet * inv_Cb) * dt_sub
            np.clip(dVb, -0.5, 0.5, out=dVb)
            Vb_new = Vb + dVb
            np.clip(Vb_new, -0.5, 1.5, out=Vb_new)
            ref_mask = refr > 0
            Vb_new[ref_mask] = V_reset
            spike_mask = (Vb_new >= V_th) & (~ref_mask)
            if spike_mask.any():
                idx = np.where(spike_mask)[0]
                t_now = ti * dt_input + (s + 1) * dt_sub
                for j in idx:
                    spike_times[int(j)].append(float(t_now))
                Vb_new[spike_mask] = V_reset
                refr[spike_mask] = T_ref_sub
                spike_pulse[spike_mask] += 1.0
            np.subtract(refr, 1, out=refr, where=refr > 0)
            Vb = Vb_new

        Vb_log[ti] = Vb.astype(np.float32)

    # Stats
    n_spikes_per_cell = np.array([len(s) for s in spike_times])
    total_spikes = int(n_spikes_per_cell.sum())
    pct_active = float((n_spikes_per_cell > 0).mean() * 100.0)
    mean_rate = float(n_spikes_per_cell.mean() / T_sec)

    # ISI stats (pooled across active cells)
    isi_all = []
    cv_per_cell = []
    for s in spike_times:
        if len(s) >= 2:
            arr = np.diff(np.asarray(s))
            isi_all.extend(arr.tolist())
            if len(arr) >= 2 and arr.mean() > 0:
                cv_per_cell.append(arr.std() / arr.mean())
    isi_all = np.asarray(isi_all) if isi_all else np.asarray([])
    cv_pooled = float(isi_all.std() / isi_all.mean()) if isi_all.size > 0 and isi_all.mean() > 0 else float("nan")
    cv_mean_cell = float(np.mean(cv_per_cell)) if cv_per_cell else float("nan")

    rate_ctrl = float(n_spikes_per_cell[ctrl_idx].mean() / T_sec)
    rate_driven = float(n_spikes_per_cell[input_drive_idx].mean() / T_sec)

    summary = {
        "control_no_input_mean_rate_hz": rate_ctrl,
        "input_driven_mean_rate_hz": rate_driven,
        "input_effect_pp": rate_driven - rate_ctrl,
        "intrinsic_oscillation_dominant": bool(abs(rate_driven - rate_ctrl) < 0.2 * rate_ctrl),
        "N_cells": N,
        "T_sec": T_sec,
        "input_rate_hz": rate_hz,
        "V_th": V_th,
        "V_reset": V_reset,
        "Cb_range_fF": [float(Cb_F.min() * 1e15), float(Cb_F.max() * 1e15)],
        "total_spikes": total_spikes,
        "mean_firing_rate_hz": mean_rate,
        "pct_cells_active": pct_active,
        "n_active_cells": int((n_spikes_per_cell > 0).sum()),
        "isi_n": int(isi_all.size),
        "isi_mean_s": float(isi_all.mean()) if isi_all.size else None,
        "isi_median_s": float(np.median(isi_all)) if isi_all.size else None,
        "cv_isi_pooled": cv_pooled,
        "cv_isi_mean_per_cell": cv_mean_cell,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Save raw
    # Pack spike_times into a ragged array via lengths + concat
    lengths = n_spikes_per_cell.astype(np.int32)
    concat = np.concatenate([np.asarray(s, dtype=np.float64) for s in spike_times]) if total_spikes > 0 else np.zeros(0)
    np.savez_compressed(
        OUT / "spike_train.npz",
        Vb_log=Vb_log,
        spike_lengths=lengths,
        spike_times_concat=concat,
        Cb_F=Cb_F, VG1=VG1, VG2_base=VG2_base,
        input_spikes=input_spikes.astype(np.uint8),
    )

    # ─── Plot ───
    fig, axes = plt.subplots(3, 1, figsize=(10, 11))

    # (1) Vb traces for 3 representative cells (one fast, one mid, one slow tau)
    order_by_Cb = np.argsort(Cb_F)
    # one fast-tau driven, one slow-tau driven, one no-input control
    driven_sorted = [i for i in order_by_Cb if i in input_drive_idx]
    pick = [driven_sorted[5], driven_sorted[len(driven_sorted) // 2], int(ctrl_idx[0])]
    t_axis = np.arange(T_steps) * dt_input
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    ax = axes[0]
    for k, c in enumerate(pick):
        tag = "no-input ctrl" if c in ctrl_idx else "input-driven"
        ax.plot(t_axis, Vb_log[:, c], color=colors[k], lw=0.8,
                label=f"cell {c}  Cb={Cb_F[c]*1e15:.1f} fF  ({tag})")
        # Mark reset events with arrows
        sts = np.asarray(spike_times[c])
        if sts.size:
            for ts in sts:
                ax.annotate("", xy=(ts, V_reset), xytext=(ts, V_th + 0.05),
                            arrowprops=dict(arrowstyle="->", color=colors[k], lw=0.6, alpha=0.6))
    ax.axhline(V_th, ls="--", color="k", lw=0.8, label=f"V_th={V_th:.2f}")
    ax.axhline(V_reset, ls=":", color="gray", lw=0.8, label=f"V_reset={V_reset:.2f}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Vb (V)")
    ax.set_title("(1) Vb(t) — integrate, threshold cross, reset (arrows = spikes)")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_xlim(0, T_sec)

    # (2) Raster
    ax = axes[1]
    for j, s in enumerate(spike_times):
        if s:
            ax.vlines(s, j - 0.4, j + 0.4, color="k", lw=0.5)
    ax.set_xlim(0, T_sec)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cell index")
    ax.set_title(f"(2) Raster — {N} cells, Poisson {rate_hz:.0f} Hz input  "
                 f"(mean rate={mean_rate:.1f} Hz, active={pct_active:.0f}%)")

    # (3) ISI distribution
    ax = axes[2]
    if isi_all.size > 0:
        ax.hist(isi_all * 1e3, bins=60, color="#555", alpha=0.85, density=True,
                label=f"NS-RAM ISI (n={isi_all.size})")
        # Overlay reference exponential (Poisson) with same mean
        mean_isi = isi_all.mean()
        x = np.linspace(0, isi_all.max(), 400)
        ax.plot(x * 1e3, (1.0 / mean_isi) * np.exp(-x / mean_isi),
                color="crimson", lw=1.5, ls="--",
                label=f"Poisson ref (mean={mean_isi*1e3:.1f} ms)")
        ax.set_xlabel("ISI (ms)")
        ax.set_ylabel("density")
        ax.set_title(f"(3) ISI distribution — CV_pooled={cv_pooled:.2f}, "
                     f"CV_mean_per_cell={cv_mean_cell:.2f}")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No spikes recorded — NS-RAM cells did not fire under Poisson input.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="crimson")
        ax.set_title("(3) ISI distribution — NO SPIKES")

    fig.suptitle("NS-RAM LIF dynamics: integrate-and-fire with body-charge memory",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT / "lif_dynamics.png", dpi=140)
    print(f"saved → {OUT/'lif_dynamics.png'}")


if __name__ == "__main__":
    main()
