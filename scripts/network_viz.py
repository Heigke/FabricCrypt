"""network_viz.py — Standardized network visualization utility.

Reusable across all network sim subagents (Phase N4).

Public API
----------
- plot_spike_raster(data, output_path, **kwargs)
- plot_vb_waterfall(data, output_path, **kwargs)
- plot_weight_evolution_gif(weight_frames, output_path, **kwargs)
- plot_energy_heatmap(data, output_path, **kwargs)
- plot_latency_violin(data, output_path, **kwargs)
- plot_pareto(data, output_path, **kwargs)
- save_summary_dashboard(simulation_result_dir, output_path=None, **kwargs)

Styling: dark background, neuromorphic blue->cyan activity palette.
GIF policy: <=100 frames @ 10 fps, attempts <=5 MB via downscale + ffmpeg.
"""
from __future__ import annotations

import io
import json
import os
import warnings
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
NEURO_CMAP = LinearSegmentedColormap.from_list(
    "neuro",
    [
        (0.00, "#06070d"),  # deep blue-black
        (0.20, "#0b2a4a"),
        (0.45, "#1170c2"),
        (0.70, "#21d4fd"),
        (0.90, "#7afcff"),
        (1.00, "#ffffff"),
    ],
)
ENERGY_CMAP = LinearSegmentedColormap.from_list(
    "energy", ["#0d0221", "#3b0f70", "#8c2981", "#dd4968", "#fd9f6c", "#fcfdbf"]
)


def _apply_dark_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#08090d",
            "axes.facecolor": "#0e1117",
            "savefig.facecolor": "#08090d",
            "axes.edgecolor": "#5a6273",
            "axes.labelcolor": "#d7dbe3",
            "axes.titlecolor": "#ffffff",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.color": "#a8afbd",
            "ytick.color": "#a8afbd",
            "grid.color": "#2a2f3a",
            "grid.alpha": 0.4,
            "text.color": "#e6e9ef",
            "font.family": "DejaVu Sans",
            "font.size": 9,
        }
    )


_apply_dark_style()


def _ensure_dir(path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------------
# 1) Spike raster (time x neuron)
# ----------------------------------------------------------------------------
def plot_spike_raster(data, output_path, *, title="Spike raster", dt_ms=1.0,
                      cmap=None, ax=None, fig=None):
    """data: 2D array (n_neurons, n_timesteps) binary or rate."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"spike raster needs 2D array, got shape {arr.shape}")
    n_neurons, n_t = arr.shape
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        arr,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap or NEURO_CMAP,
        origin="lower",
        extent=[0, n_t * dt_ms, 0, n_neurons],
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_title(title)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Activity")
    if standalone:
        out = _ensure_dir(output_path)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        return str(out)
    return im


# ----------------------------------------------------------------------------
# 2) V_B waterfall (per-neuron membrane / bulk voltage over time)
# ----------------------------------------------------------------------------
def plot_vb_waterfall(data, output_path, *, title="V_B(t) waterfall", dt_ms=1.0,
                      offset_scale=1.0, max_traces=64, ax=None, fig=None):
    """data: 2D array (n_neurons, n_timesteps) of V_B traces."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"V_B waterfall needs 2D array, got {arr.shape}")
    n_neurons, n_t = arr.shape
    if n_neurons > max_traces:
        idx = np.linspace(0, n_neurons - 1, max_traces).astype(int)
        arr = arr[idx]
        n_neurons = arr.shape[0]
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))
    t = np.arange(n_t) * dt_ms
    span = float(np.nanmax(arr) - np.nanmin(arr) + 1e-9)
    step = span * offset_scale
    colors = NEURO_CMAP(np.linspace(0.2, 0.95, n_neurons))
    for i in range(n_neurons):
        ax.plot(t, arr[i] + i * step, color=colors[i], linewidth=0.7, alpha=0.9)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron (stacked V_B)")
    ax.set_title(title)
    # colorbar proxy
    sm = plt.cm.ScalarMappable(cmap=NEURO_CMAP,
                               norm=plt.Normalize(vmin=0, vmax=n_neurons))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Neuron index")
    if standalone:
        out = _ensure_dir(output_path)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        return str(out)
    return ax


# ----------------------------------------------------------------------------
# 3) Weight evolution GIF
# ----------------------------------------------------------------------------
def _render_weight_frame(W, step, vmin, vmax, fig_size=(4, 4)):
    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(W, cmap=NEURO_CMAP, vmin=vmin, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    ax.set_title(f"Weights @ step {step}")
    ax.set_xlabel("Post")
    ax.set_ylabel("Pre")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("w")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    import imageio.v2 as imageio
    frame = imageio.imread(buf)
    return frame


def plot_weight_evolution_gif(weight_frames, output_path, *, fps=10,
                              max_frames=100, max_mb=5.0, downscale_to=None):
    """weight_frames: iterable of 2D arrays, or 3D array (T, N, M)."""
    import imageio.v2 as imageio

    frames_list = list(weight_frames)
    if len(frames_list) == 0:
        raise ValueError("weight_frames is empty")
    if len(frames_list) > max_frames:
        idx = np.linspace(0, len(frames_list) - 1, max_frames).astype(int)
        frames_list = [frames_list[i] for i in idx]
        steps = idx
    else:
        steps = np.arange(len(frames_list))

    arrs = [np.asarray(W, dtype=np.float32) for W in frames_list]
    vmin = float(np.min([a.min() for a in arrs]))
    vmax = float(np.max([a.max() for a in arrs]))

    fig_size = (4, 4) if downscale_to is None else downscale_to
    frames = [_render_weight_frame(W, int(s), vmin, vmax, fig_size=fig_size)
              for W, s in zip(arrs, steps)]

    out = _ensure_dir(output_path)
    suffix = out.suffix.lower()
    used_fallback = False
    try:
        if suffix == ".gif":
            imageio.mimsave(str(out), frames, format="GIF",
                            duration=1.0 / fps, loop=0)
        else:
            # mp4 via ffmpeg
            imageio.mimsave(str(out), frames, fps=fps)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"ffmpeg/gif writer failed ({e}); falling back to PIL APNG")
        used_fallback = True
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in frames]
        apng_out = out.with_suffix(".png")
        pil_frames[0].save(apng_out, save_all=True, append_images=pil_frames[1:],
                           duration=int(1000 / fps), loop=0)
        out = apng_out

    # Size check + re-encode smaller if needed
    size_mb = out.stat().st_size / (1024 * 1024)
    if size_mb > max_mb and not used_fallback and suffix == ".gif":
        # downscale frames and retry once
        small_frames = []
        for f in frames:
            h, w = f.shape[:2]
            nh, nw = h // 2, w // 2
            # simple nearest-neighbor decimation
            small_frames.append(f[::2, ::2])
        imageio.mimsave(str(out), small_frames, format="GIF",
                        duration=1.0 / fps, loop=0)
        size_mb = out.stat().st_size / (1024 * 1024)
    return {"path": str(out), "size_mb": round(size_mb, 3),
            "frames": len(frames), "fallback_apng": used_fallback}


# ----------------------------------------------------------------------------
# 4) Energy heatmap (per-neuron)
# ----------------------------------------------------------------------------
def plot_energy_heatmap(data, output_path, *, title="Per-neuron energy (pJ)",
                        layout=None, ax=None, fig=None):
    """data: 1D (n_neurons,) or 2D (rows, cols) energy in pJ."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        n = arr.size
        if layout is None:
            side = int(np.ceil(np.sqrt(n)))
            pad = side * side - n
            arr = np.concatenate([arr, np.full(pad, np.nan)])
            arr = arr.reshape(side, side)
        else:
            arr = arr.reshape(layout)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(arr, cmap=ENERGY_CMAP, aspect="equal",
                   interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Energy (pJ)")
    if standalone:
        out = _ensure_dir(output_path)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        return str(out)
    return im


# ----------------------------------------------------------------------------
# 5) Latency violin per layer
# ----------------------------------------------------------------------------
def plot_latency_violin(data, output_path, *, title="Latency distribution",
                        unit="us", ax=None, fig=None):
    """data: dict {layer_name: 1D array of latencies}."""
    if not isinstance(data, Mapping):
        raise TypeError("latency violin expects a mapping {layer: latencies}")
    labels = list(data.keys())
    samples = [np.asarray(data[k], dtype=np.float32) for k in labels]
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))
    parts = ax.violinplot(samples, showmeans=True, showextrema=True,
                          showmedians=True)
    palette = NEURO_CMAP(np.linspace(0.35, 0.9, len(samples)))
    for body, c in zip(parts["bodies"], palette):
        body.set_facecolor(c)
        body.set_edgecolor("#cfd5e3")
        body.set_alpha(0.75)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#e6e9ef")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(f"Latency ({unit})")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    if standalone:
        out = _ensure_dir(output_path)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        return str(out)
    return ax


# ----------------------------------------------------------------------------
# 6) Pareto (accuracy vs energy vs throughput)
# ----------------------------------------------------------------------------
def plot_pareto(data, output_path, *,
                title="Pareto: accuracy vs energy vs throughput",
                ax=None, fig=None):
    """data: list of dicts with keys
      'name', 'accuracy', 'energy_pj', 'throughput' (optional 'topology')."""
    pts = list(data)
    if not pts:
        raise ValueError("pareto data empty")
    acc = np.array([p["accuracy"] for p in pts], dtype=float)
    eng = np.array([p["energy_pj"] for p in pts], dtype=float)
    thr = np.array([p["throughput"] for p in pts], dtype=float)
    names = [p.get("name", str(i)) for i, p in enumerate(pts)]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))
    sizes = 40 + 220 * (thr - thr.min()) / max(1e-9, (thr.max() - thr.min()))
    sc = ax.scatter(eng, acc, c=thr, s=sizes, cmap=NEURO_CMAP,
                    edgecolor="#e6e9ef", linewidth=0.6, alpha=0.9)
    for x, y, n in zip(eng, acc, names):
        ax.annotate(n, (x, y), xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color="#cfd5e3")
    # Pareto front (minimize energy, maximize accuracy)
    order = np.argsort(eng)
    front_x, front_y = [], []
    best = -np.inf
    for i in order:
        if acc[i] > best:
            front_x.append(eng[i]); front_y.append(acc[i]); best = acc[i]
    ax.plot(front_x, front_y, color="#7afcff", linestyle="--",
            linewidth=1.2, alpha=0.8, label="Pareto front")
    ax.set_xscale("log")
    ax.set_xlabel("Energy / inference (pJ)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="lower left", framealpha=0.3)
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Throughput (inf/s)")
    if standalone:
        out = _ensure_dir(output_path)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        return str(out)
    return ax


# ----------------------------------------------------------------------------
# 7) Summary dashboard
# ----------------------------------------------------------------------------
def _autodiscover(sim_dir):
    """Look for standard files in a sim result dir; return a dict of arrays."""
    d = Path(sim_dir)
    out = {}
    candidates = {
        "spikes":      ["spikes.npy", "raster.npy", "spike_raster.npy"],
        "vb":          ["vb.npy", "v_b.npy", "vb_waterfall.npy"],
        "weights":     ["weights.npy", "W.npy", "weight_evolution.npy"],
        "energy":      ["energy.npy", "energy_pj.npy", "per_neuron_energy.npy"],
        "latency":     ["latency.json", "latency.npz"],
        "pareto":      ["pareto.json", "pareto_points.json"],
    }
    for key, names in candidates.items():
        for n in names:
            p = d / n
            if p.exists():
                if p.suffix == ".npy":
                    out[key] = np.load(p, allow_pickle=False)
                elif p.suffix == ".npz":
                    z = np.load(p, allow_pickle=False)
                    out[key] = {k: z[k] for k in z.files}
                elif p.suffix == ".json":
                    out[key] = json.loads(p.read_text())
                break
    return out


def save_summary_dashboard(simulation_result_dir, output_path=None,
                           data=None, title=None):
    """Render a 6-panel dashboard. Either auto-discover from a sim dir,
    or pass `data` dict directly with same keys as _autodiscover."""
    if data is None:
        data = _autodiscover(simulation_result_dir)
    sim_dir = Path(simulation_result_dir)
    if output_path is None:
        output_path = sim_dir / "dashboard.png"
    out = _ensure_dir(output_path)
    if title is None:
        title = f"Network simulation summary — {sim_dir.name}"

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    if "spikes" in data:
        plot_spike_raster(data["spikes"], None, ax=ax1, fig=fig,
                          title="Spike raster")
    else:
        ax1.set_title("Spike raster (no data)"); ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    if "vb" in data:
        plot_vb_waterfall(data["vb"], None, ax=ax2, fig=fig,
                          title="V_B(t) waterfall")
    else:
        ax2.set_title("V_B waterfall (no data)"); ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    if "energy" in data:
        plot_energy_heatmap(data["energy"], None, ax=ax3, fig=fig,
                            title="Per-neuron energy (pJ)")
    else:
        ax3.set_title("Energy heatmap (no data)"); ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 0])
    if "latency" in data:
        lat = data["latency"]
        if isinstance(lat, dict) and not all(isinstance(v, np.ndarray) for v in lat.values()):
            lat = {k: np.asarray(v) for k, v in lat.items()}
        plot_latency_violin(lat, None, ax=ax4, fig=fig,
                            title="Latency per layer")
    else:
        ax4.set_title("Latency violin (no data)"); ax4.axis("off")

    ax5 = fig.add_subplot(gs[1, 1])
    if "pareto" in data:
        plot_pareto(data["pareto"], None, ax=ax5, fig=fig,
                    title="Accuracy/Energy/Throughput Pareto")
    else:
        ax5.set_title("Pareto (no data)"); ax5.axis("off")

    ax6 = fig.add_subplot(gs[1, 2])
    if "weights" in data:
        W = np.asarray(data["weights"])
        if W.ndim == 3:
            W = W[-1]
        im = ax6.imshow(W, cmap=NEURO_CMAP, aspect="equal",
                        interpolation="nearest")
        ax6.set_title("Final weight matrix")
        ax6.set_xlabel("Post"); ax6.set_ylabel("Pre")
        plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04).set_label("w")
    else:
        ax6.set_title("Weights (no data)"); ax6.axis("off")

    fig.suptitle(title, fontsize=14, color="#ffffff", y=0.995)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


__all__ = [
    "plot_spike_raster",
    "plot_vb_waterfall",
    "plot_weight_evolution_gif",
    "plot_energy_heatmap",
    "plot_latency_violin",
    "plot_pareto",
    "save_summary_dashboard",
    "NEURO_CMAP",
    "ENERGY_CMAP",
]
