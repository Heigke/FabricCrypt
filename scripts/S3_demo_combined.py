#!/usr/bin/env python3
"""S3_demo_combined.py — combined transient + network + variation pipeline.

Pre-registered gate: N=10K cells, 1 ms wall-clock simulation, < 5 min.

A minimal LIF transient layer drives the SynapticNetwork from S3_network_glue
and uses per-cell parameters from S3_cell_variation. forward_2t_batched
supplies the avalanche current per cell each step.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np

from S3_cell_variation import (extract_distributions, sample_cells,
                                forward_2t_batched)
from S3_network_glue import SynapticNetwork, topology_er

REPO = Path(__file__).resolve().parents[1]
RES_DIR = REPO / "results" / "S3_network_variation"
RES_DIR.mkdir(parents=True, exist_ok=True)


def run_combined(N: int = 10_000, T_sim: float = 1e-3, dt: float = 1e-6,
                 density: float = 0.001, seed: int = 42, verbose: bool = True):
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # 1) Per-cell variation
    dists = extract_distributions(save=False)
    cells = sample_cells(N=N, seed=seed, dists=dists)

    # 2) Network
    W = topology_er(N, density=density, seed=seed)
    net = SynapticNetwork(W, tau_syn=5e-6, vg2_gain=1e-3)

    # 3) LIF state
    Vm = 0.05 * rng.standard_normal(N)
    V_thresh = 1.364 + 0.05 * rng.standard_normal(N)
    V_rest = 0.0
    V_reset_frac = 0.3
    g_leak = (cells["C_b"] / 1e-6)  # τ_mem ≈ 1 µs
    C_mem = cells["C_b"]
    refrac = np.zeros(N)
    t_refrac = 1.6e-6

    # 4) Baseline gate biases (per-cell mild heterogeneity)
    Vg1 = 0.35 + 0.05 * rng.standard_normal(N)
    Vg2_base = 0.40 + 0.03 * rng.standard_normal(N)
    Vds_amp = 2.5

    n_steps = int(round(T_sim / dt))
    spike_counts = np.zeros(N, dtype=np.int64)
    silent_mask = np.ones(N, dtype=bool)
    step_times = []
    last_spikes = np.empty(0, dtype=np.int64)

    t_setup = time.time() - t0
    for step_i in range(n_steps):
        ts = time.time()
        # Vcb pulse: self-resonation 100 kHz square-ish
        phase = (step_i * dt) % 1e-5
        Vds = Vds_amp if phase < 5e-6 else 0.2

        # Network deposit from previous step's spikes
        dVG2, _ = net.step(last_spikes, dt=dt)
        Vg2 = Vg2_base + dVG2

        # Avalanche current (per cell)
        I_aval = forward_2t_batched(Vg1, Vg2, Vds, cells)
        # External "input" current to all cells (small, drives sub-threshold)
        I_ext = 5e-9 * rng.standard_normal(N)
        I_total = I_aval + I_ext - g_leak * (Vm - V_rest)
        Vm = np.where(refrac > 0, Vm, Vm + dt * I_total / C_mem)
        refrac = np.maximum(refrac - dt, 0.0)

        # Spike
        spiked = (Vm >= V_thresh) & (refrac <= 0)
        if spiked.any():
            Vm[spiked] *= V_reset_frac
            refrac[spiked] = t_refrac
            spike_counts[spiked] += 1
            silent_mask[spiked] = False
        last_spikes = np.flatnonzero(spiked)
        step_times.append(time.time() - ts)

    t_total = time.time() - t0
    step_times = np.array(step_times)
    n_silent = int(silent_mask.sum())
    summary = {
        "N": N,
        "T_sim_s": T_sim,
        "dt_s": dt,
        "n_steps": n_steps,
        "setup_s": t_setup,
        "total_s": t_total,
        "step_mean_ms": float(step_times.mean() * 1000),
        "step_p95_ms": float(np.percentile(step_times, 95) * 1000),
        "total_spikes": int(spike_counts.sum()),
        "active_cells": int((~silent_mask).sum()),
        "silent_cells": n_silent,
        "silent_frac": float(n_silent / N),
        "spike_rate_hz_mean": float(spike_counts.mean() / T_sim),
        "spike_rate_hz_max": float(spike_counts.max() / T_sim),
        "PASS_under_5min": bool(t_total < 300.0),
        "PASS_has_spikes": bool(spike_counts.sum() > 0),
    }
    return summary


def main():
    summary = run_combined(N=10_000, T_sim=1e-3, dt=1e-6, density=0.001, seed=42)
    print(json.dumps(summary, indent=2))
    out = RES_DIR / "demo_n10k_1ms.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
