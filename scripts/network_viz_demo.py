"""network_viz_demo.py — generate fake data & exercise every viz function.

Run:
    python scripts/network_viz_demo.py
Outputs:
    results/network_viz_demo/{raster,vb,energy,latency,pareto,weights_final,
                              dashboard}.png + weight_evo.gif
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from network_viz import (  # noqa: E402
    plot_spike_raster,
    plot_vb_waterfall,
    plot_weight_evolution_gif,
    plot_energy_heatmap,
    plot_latency_violin,
    plot_pareto,
    save_summary_dashboard,
)

OUT = REPO / "results" / "network_viz_demo"
OUT.mkdir(parents=True, exist_ok=True)


def fake_spike_raster(n_neurons=128, n_t=800, seed=0):
    rng = np.random.default_rng(seed)
    # Poisson background + travelling wave
    base = (rng.random((n_neurons, n_t)) < 0.02).astype(np.float32)
    t = np.arange(n_t)
    for i in range(n_neurons):
        # gaussian bump that sweeps through neurons
        center = (i / n_neurons) * n_t
        bump = np.exp(-((t - center) ** 2) / (2 * 30 ** 2))
        base[i] = np.maximum(base[i], (rng.random(n_t) < bump * 0.6).astype(np.float32))
    return base


def fake_vb(n_neurons=64, n_t=800, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n_t) / 100.0
    vb = np.zeros((n_neurons, n_t), dtype=np.float32)
    for i in range(n_neurons):
        freq = 0.5 + 2.0 * (i / n_neurons)
        phase = rng.uniform(0, 2 * np.pi)
        vb[i] = (0.6 * np.sin(2 * np.pi * freq * t + phase)
                 + 0.2 * rng.standard_normal(n_t))
    return vb


def fake_weight_evolution(n_pre=32, n_post=32, n_frames=60, seed=2):
    rng = np.random.default_rng(seed)
    W0 = rng.normal(0, 0.1, size=(n_pre, n_post)).astype(np.float32)
    target = rng.normal(0, 0.5, size=(n_pre, n_post)).astype(np.float32)
    frames = []
    W = W0.copy()
    for k in range(n_frames):
        W = W + 0.05 * (target - W) + 0.01 * rng.standard_normal(W.shape).astype(np.float32)
        frames.append(W.copy())
    return frames


def fake_energy(n=128, seed=3):
    rng = np.random.default_rng(seed)
    # log-normal per-neuron energy in pJ
    return rng.lognormal(mean=1.5, sigma=0.6, size=n).astype(np.float32)


def fake_latency(seed=4):
    rng = np.random.default_rng(seed)
    return {
        "input":   rng.normal(2.0, 0.3, 400).clip(0.5, None),
        "hidden1": rng.normal(5.5, 0.8, 400).clip(0.5, None),
        "hidden2": rng.normal(8.1, 1.2, 400).clip(0.5, None),
        "output":  rng.normal(3.2, 0.5, 400).clip(0.5, None),
    }


def fake_pareto(seed=5):
    rng = np.random.default_rng(seed)
    out = []
    for i, name in enumerate(["MLP32", "MLP128", "MLP512", "Conv2k",
                              "Conv8k", "ESN32k", "ESN128k", "DSN-N2k"]):
        e = 10 ** rng.uniform(0.5, 4.5)
        # accuracy roughly anti-correlated with energy ceiling
        a = 0.55 + 0.4 * (1 - np.exp(-e / 200)) + rng.normal(0, 0.04)
        a = float(np.clip(a, 0, 0.99))
        thr = 10 ** rng.uniform(2, 5)
        out.append({"name": name, "accuracy": a, "energy_pj": float(e),
                    "throughput": float(thr)})
    return out


def main():
    print(f"[demo] generating fake data + writing to {OUT}")

    spikes = fake_spike_raster()
    vb = fake_vb()
    weights = fake_weight_evolution()
    energy = fake_energy()
    latency = fake_latency()
    pareto = fake_pareto()

    # Save raw data so save_summary_dashboard can auto-discover
    np.save(OUT / "spikes.npy", spikes)
    np.save(OUT / "vb.npy", vb)
    np.save(OUT / "weights.npy", np.stack(weights, axis=0))
    np.save(OUT / "energy.npy", energy)
    (OUT / "latency.json").write_text(json.dumps(
        {k: v.tolist() for k, v in latency.items()}))
    (OUT / "pareto.json").write_text(json.dumps(pareto, indent=2))

    paths = {}
    paths["raster"] = plot_spike_raster(spikes, OUT / "raster.png",
                                        title="Demo: travelling-wave raster (128n)")
    paths["vb"] = plot_vb_waterfall(vb, OUT / "vb.png",
                                    title="Demo: V_B(t) waterfall (64n)")
    paths["energy"] = plot_energy_heatmap(energy, OUT / "energy.png",
                                          title="Demo: per-neuron energy (pJ)")
    paths["latency"] = plot_latency_violin(latency, OUT / "latency.png",
                                           title="Demo: latency per layer (us)")
    paths["pareto"] = plot_pareto(pareto, OUT / "pareto.png",
                                  title="Demo: accuracy / energy / throughput Pareto")

    gif_info = plot_weight_evolution_gif(weights, OUT / "weight_evo.gif",
                                         fps=10, max_frames=100, max_mb=5.0)
    paths["weight_evo"] = gif_info["path"]
    print(f"[demo] GIF: {gif_info}")

    paths["dashboard"] = save_summary_dashboard(OUT)
    print("[demo] outputs:")
    for k, v in paths.items():
        sz = Path(v).stat().st_size
        ok = sz > 0
        print(f"   {k:12s} {v}  ({sz/1024:7.1f} KB) {'OK' if ok else 'EMPTY'}")
        assert ok, f"{v} empty!"

    # README with example usage
    readme = OUT / "README.md"
    readme.write_text(EXAMPLE_README)
    print(f"[demo] wrote {readme}")
    print("[demo] DONE")


EXAMPLE_README = """# network_viz demo outputs

Generated by `scripts/network_viz_demo.py`. All plots use the standardized
neuromorphic dark style from `scripts/network_viz.py`.

## Files
- `raster.png` — spike-time raster heatmap
- `vb.png` — V_B(t) waterfall (stacked traces)
- `energy.png` — per-neuron energy heatmap (square layout)
- `latency.png` — latency violin per network layer
- `pareto.png` — accuracy vs energy vs throughput, log-x, Pareto front overlay
- `weight_evo.gif` — animated weight matrix over training
- `dashboard.png` — single-figure 6-panel publication summary
- raw inputs: `spikes.npy`, `vb.npy`, `weights.npy`, `energy.npy`,
  `latency.json`, `pareto.json`

## Example usage
```python
from network_viz import (
    plot_spike_raster, plot_vb_waterfall, plot_weight_evolution_gif,
    plot_energy_heatmap, plot_latency_violin, plot_pareto,
    save_summary_dashboard,
)

# 1) one-shot, just give arrays
plot_spike_raster(spikes_2d, "out/raster.png", dt_ms=1.0)
plot_vb_waterfall(vb_2d, "out/vb.png", max_traces=64)
plot_energy_heatmap(energy_1d, "out/energy.png")
plot_latency_violin({"L1": l1_us, "L2": l2_us}, "out/lat.png")
plot_pareto([{"name": "MLP", "accuracy": 0.92,
              "energy_pj": 120.0, "throughput": 4500.0}],
            "out/pareto.png")
plot_weight_evolution_gif(list_of_weight_matrices, "out/w.gif",
                          fps=10, max_frames=100, max_mb=5.0)

# 2) dashboard auto-discovers spikes.npy/vb.npy/weights.npy/energy.npy/
#    latency.json/pareto.json inside a sim result dir
save_summary_dashboard("results/my_sim_xxx/")
```

## Dependencies
- numpy, matplotlib (already in repo venv)
- imageio + imageio-ffmpeg (install with
  `pip install imageio imageio-ffmpeg`)
- system `ffmpeg` (already installed at /usr/bin/ffmpeg)
- Pillow (transitive via matplotlib) — used as APNG fallback if ffmpeg fails

If imageio-ffmpeg is missing the GIF writer falls back to PIL animated PNG
(`weight_evo.png`) with a warning; no other plots are affected.
"""


if __name__ == "__main__":
    main()
