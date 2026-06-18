#!/usr/bin/env python3
"""
analyze_boltzmann_sweep.py — Read Boltzmann sweep raw files and count spikes.

Reads nsram_boltzmann_T{310..335}.raw, counts V(vspike)>0.5 rising edges,
prints a table, and saves results to results/nsram_boltzmann_sweep.json.
"""
import sys, os, json
import numpy as np

# Reuse the raw reader from plot_bridge_experiments
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_bridge_experiments import read_raw, count_spikes

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def analyze():
    temperatures = list(range(310, 336))
    results = []

    print(f"{'T (K)':>7}  {'T (C)':>7}  {'Vg_eff':>8}  {'Spikes':>7}")
    print("-" * 36)

    for t in temperatures:
        raw_path = os.path.join(RESULTS_DIR, f"nsram_boltzmann_T{t}.raw")
        vg_eff = 0.45 + 0.002 * (t - 300)
        data = read_raw(raw_path)
        if data is None:
            print(f"{t:>7}  {t-273:>7}  {vg_eff:>8.4f}  {'MISSING':>7}")
            results.append({"T_K": t, "T_C": t - 273, "Vg_eff": round(vg_eff, 4), "spikes": None})
            continue

        # Find vspike signal
        vspike = None
        for key in data:
            if 'vspike' in key:
                vspike = data[key]
                break

        if vspike is None:
            print(f"{t:>7}  {t-273:>7}  {vg_eff:>8.4f}  {'NO SIG':>7}")
            results.append({"T_K": t, "T_C": t - 273, "Vg_eff": round(vg_eff, 4), "spikes": None})
            continue

        n_spikes = count_spikes(vspike, threshold=0.5)
        print(f"{t:>7}  {t-273:>7}  {vg_eff:>8.4f}  {n_spikes:>7}")
        results.append({"T_K": t, "T_C": t - 273, "Vg_eff": round(vg_eff, 4), "spikes": n_spikes})

    # Save JSON
    out_path = os.path.join(RESULTS_DIR, "nsram_boltzmann_sweep.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "Boltzmann temperature sweep",
            "description": "Fine-grained 310K-335K in 1K steps to resolve 0->13 spike transition",
            "model": "Vg_eff = 0.45 + 0.002*(T-300), dVg/dT = 2mV/K",
            "spike_threshold": 0.5,
            "data": results,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Quick summary
    spike_counts = [r["spikes"] for r in results if r["spikes"] is not None]
    temps_with_spikes = [r["T_K"] for r in results if r["spikes"] is not None and r["spikes"] > 0]
    if temps_with_spikes:
        print(f"\nFirst non-zero spikes at T = {min(temps_with_spikes)}K")
        print(f"Spike range: {min(spike_counts)} to {max(spike_counts)}")


if __name__ == "__main__":
    analyze()
