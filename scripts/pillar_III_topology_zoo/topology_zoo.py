"""Pillar III — Topology zoo with distilled NS-RAM cell as node nonlinearity.

6 topologies x 5 stimulus regimes x 4 heterogeneity levels = 120 conditions.

Per condition (N=2048):
  - Build adjacency (sparse).
  - Drive each node with stimulus + recurrent input (weighted by adjacency).
  - Pass total input through NS-RAM cell surrogate (VG1 = bias offset + jittered,
    Vd = drive voltage clipped to [0, 2.2], VG2 = 0.10 reservoir prior).
  - Cell returns |Id| in Amps — we log10-normalize to drive the state with a
    leaky integrator (the differentiable cell IS the nonlinearity; we don't
    inject FHN dynamics).

Metrics: Kuramoto R, largest Lyapunov, spectral entropy, NARMA-30 r^2,
  mean rate, CV(ISI-proxy), edge-of-chaos flag.

Output:
  results/Pillar_III_topology_zoo/phase_diagram_3d.json
  results/Pillar_III_topology_zoo/heatmaps.png
  results/Pillar_III_topology_zoo/verdict.md
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "Pillar_III_topology_zoo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[zoo] device={DEVICE}, torch={torch.__version__}")

# ----------------------------------------------------------------------
# 1. Distilled cell surrogate
# ----------------------------------------------------------------------
class DistilledMLP(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("x_mean", torch.zeros(3))
        self.register_buffer("x_std", torch.ones(3))
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = (x - self.x_mean) / self.x_std
        yn = self.net(xn).squeeze(-1)
        return yn * self.y_std + self.y_mean


def load_cell() -> DistilledMLP:
    bundle = torch.load(OUT_DIR / "distilled_mlp.pt", map_location=DEVICE,
                        weights_only=True)
    m = DistilledMLP(hidden=bundle["hidden"]).to(DEVICE)
    m.load_state_dict(bundle["state_dict"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    print(f"[zoo] loaded cell (val_median_abs_dec={bundle['val_median_abs_dec']:.4f})")
    return m


# Hard envelope from Sebas measurements; do NOT extrapolate.
VG1_LO, VG1_HI = 0.20, 0.60
VG2_FIXED = 0.10
VD_LO, VD_HI = 0.0, 2.2


def cell_response(cell: DistilledMLP, drive: torch.Tensor,
                  vg1_bias: torch.Tensor) -> torch.Tensor:
    """drive: (N,) voltage-like, vg1_bias: (N,) per-node VG1 with heterogeneity.

    Returns a *normalized* response in roughly [-1, 1] suitable for state evol.
    """
    vd = torch.clamp(drive, VD_LO, VD_HI)
    vg1 = torch.clamp(vg1_bias, VG1_LO, VG1_HI)
    vg2 = torch.full_like(vd, VG2_FIXED)
    x = torch.stack([vg1, vg2, vd], dim=-1)
    log_id = cell(x)  # log10|Id|, roughly in [-12, -4]
    # Map to ~[-1, 1] centered on midpoint of training y range (~-8).
    # std ~1.5 dec, so divide by 4 to bring most into [-1, 1].
    return (log_id + 8.0) / 4.0


# ----------------------------------------------------------------------
# 2. Topologies (sparse COO adjacency; weights normalized so row-sum~1).
# ----------------------------------------------------------------------
def topo_erdos_renyi(N: int, k: int, rng: np.random.Generator) -> torch.Tensor:
    rows = []
    cols = []
    for i in range(N):
        targets = rng.choice(N, size=k, replace=False)
        rows.extend([i] * k)
        cols.extend(targets.tolist())
    return _to_norm_adj(N, rows, cols)


def topo_watts_strogatz(N: int, k: int, p: float, rng: np.random.Generator) -> torch.Tensor:
    # Start with ring of k nearest neighbors then rewire with prob p.
    rows = []
    cols = []
    half = k // 2
    for i in range(N):
        for j in range(1, half + 1):
            for tgt in (i - j) % N, (i + j) % N:
                if rng.random() < p:
                    tgt = int(rng.integers(N))
                rows.append(i); cols.append(tgt)
    return _to_norm_adj(N, rows, cols)


def topo_barabasi_albert(N: int, m: int, rng: np.random.Generator) -> torch.Tensor:
    # Preferential attachment.
    edges = [(i, j) for i in range(m + 1) for j in range(m + 1) if i != j]
    deg = np.zeros(N, dtype=np.int64)
    for a, b in edges:
        deg[a] += 1
    for new in range(m + 1, N):
        prob = deg[:new] / deg[:new].sum()
        targets = rng.choice(new, size=m, replace=False, p=prob)
        for t in targets:
            edges.append((new, int(t)))
            edges.append((int(t), new))
            deg[new] += 1; deg[t] += 1
    rows, cols = zip(*edges)
    return _to_norm_adj(N, list(rows), list(cols))


def topo_modular(N: int, n_blocks: int, p_in: float, p_out: float,
                 rng: np.random.Generator) -> torch.Tensor:
    block = N // n_blocks
    rows = []
    cols = []
    for i in range(N):
        bi = i // block
        # Intra-block: pick ~p_in * block neighbors
        intra = rng.choice(block, size=max(1, int(p_in * block)), replace=False)
        for j in intra:
            tgt = bi * block + int(j)
            if tgt != i:
                rows.append(i); cols.append(tgt)
        # Inter-block: pick ~p_out * (N-block) others
        n_inter = max(1, int(p_out * (N - block)))
        cand = rng.integers(0, N, size=n_inter * 3)
        added = 0
        for c in cand:
            if c // block != bi and c != i:
                rows.append(i); cols.append(int(c)); added += 1
                if added >= n_inter:
                    break
    return _to_norm_adj(N, rows, cols)


def topo_hierarchical(N: int, n_levels: int, branch: int,
                      rng: np.random.Generator) -> torch.Tensor:
    # 3 levels of branch=8 -> 8*8*8=512 leaves; we tile to fill N.
    # Build a tree-like sparse with random in-level + parent + sibling links.
    rows = []
    cols = []
    leaves = branch ** n_levels  # e.g. 512
    reps = max(1, N // leaves)
    for cluster in range(reps):
        base = cluster * leaves
        for i in range(leaves):
            gi = base + i
            # parent link (in same cluster)
            parent = base + (i // branch)
            if parent != gi:
                rows.append(gi); cols.append(parent)
            # sibling link
            sib = base + (i // branch) * branch + int(rng.integers(branch))
            if sib != gi:
                rows.append(gi); cols.append(sib)
            # random in-cluster
            for _ in range(2):
                r = base + int(rng.integers(leaves))
                if r != gi:
                    rows.append(gi); cols.append(r)
            # rare cross-cluster
            if rng.random() < 0.02:
                r = int(rng.integers(N))
                if r != gi:
                    rows.append(gi); cols.append(r)
    # Pad up to N if needed (shouldn't happen)
    return _to_norm_adj(N, rows, cols)


def _to_norm_adj(N: int, rows: List[int], cols: List[int]) -> torch.Tensor:
    rows_t = torch.tensor(rows, dtype=torch.long, device=DEVICE)
    cols_t = torch.tensor(cols, dtype=torch.long, device=DEVICE)
    # Row-normalize via per-row count.
    counts = torch.bincount(rows_t, minlength=N).clamp_min(1).float()
    vals = (1.0 / counts[rows_t])
    adj = torch.sparse_coo_tensor(
        indices=torch.stack([rows_t, cols_t]),
        values=vals,
        size=(N, N),
        device=DEVICE,
    ).coalesce()
    return adj


def make_topology(name: str, N: int, rng: np.random.Generator) -> torch.Tensor:
    if name == "ER_SPARSE":
        return topo_erdos_renyi(N, k=8, rng=rng)
    if name == "ER_DENSE":
        return topo_erdos_renyi(N, k=64, rng=rng)
    if name == "WS_SMALL_WORLD":
        return topo_watts_strogatz(N, k=8, p=0.1, rng=rng)
    if name == "BA_SCALE_FREE":
        return topo_barabasi_albert(N, m=4, rng=rng)
    if name == "MODULAR_4BLOCK":
        return topo_modular(N, n_blocks=4, p_in=0.10, p_out=0.01, rng=rng)
    if name == "HIERARCHICAL_3LEVEL":
        return topo_hierarchical(N, n_levels=3, branch=8, rng=rng)
    raise ValueError(name)


# ----------------------------------------------------------------------
# 3. Stimuli (T timesteps, single channel; we broadcast over nodes).
# ----------------------------------------------------------------------
def stim_white(T: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, sigma, T).astype(np.float32)


def stim_one_over_f(T: int, alpha: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    f = np.fft.rfftfreq(T, d=1.0)
    f[0] = f[1]
    amp = 1.0 / (f ** (alpha / 2.0))
    phases = rng.uniform(0, 2 * np.pi, len(f))
    spec = amp * np.exp(1j * phases)
    x = np.fft.irfft(spec, n=T).real
    x = (x - x.mean()) / (x.std() + 1e-8) * sigma
    return x.astype(np.float32)


def stim_sine_plus_noise(T: int, freq_hz: float, dt: float, sigma_noise: float,
                         rng: np.random.Generator) -> np.ndarray:
    t = np.arange(T) * dt
    return (np.sin(2 * np.pi * freq_hz * t) + rng.normal(0, sigma_noise, T)).astype(np.float32) * 0.1


def stim_poisson_spikes(T: int, rate_hz: float, dt: float, amp: float,
                        rng: np.random.Generator) -> np.ndarray:
    p = rate_hz * dt
    spikes = (rng.random(T) < p).astype(np.float32) * amp
    return spikes


def stim_mackey_glass(T: int, tau: int, rng: np.random.Generator) -> np.ndarray:
    beta, gamma, n = 0.2, 0.1, 10.0
    x = np.zeros(T + tau, dtype=np.float32)
    x[:tau] = 1.2 + 0.01 * rng.normal(size=tau)
    for i in range(tau, T + tau - 1):
        x[i + 1] = x[i] + beta * x[i - tau] / (1 + x[i - tau] ** n) - gamma * x[i]
    y = x[tau:]
    y = (y - y.mean()) / (y.std() + 1e-8)
    return (y * 0.1).astype(np.float32)


def make_stim(name: str, T: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    if name == "WHITE_NOISE":
        return stim_white(T, 0.1, rng)
    if name == "1F_NOISE":
        return stim_one_over_f(T, alpha=1.0, sigma=0.1, rng=rng)
    if name == "SINE_PLUS_NOISE":
        return stim_sine_plus_noise(T, 5.0, dt, 0.05, rng)
    if name == "POISSON_SPIKES":
        return stim_poisson_spikes(T, 10.0, dt, 0.2, rng)
    if name == "MACKEY_GLASS":
        return stim_mackey_glass(T, tau=17, rng=rng)
    raise ValueError(name)


# ----------------------------------------------------------------------
# 4. Network simulation.
# ----------------------------------------------------------------------
def simulate(cell: DistilledMLP, adj: torch.Tensor, vg1_bias: torch.Tensor,
             stim: np.ndarray, dt: float, leak: float = 0.85,
             drive_gain: float = 1.0, vg1_center: float = 0.40) -> torch.Tensor:
    """Return state trajectory (T, N) on CPU as float32."""
    T = len(stim)
    N = vg1_bias.shape[0]
    state = torch.zeros(N, device=DEVICE)
    stim_t = torch.from_numpy(stim).to(DEVICE)
    states_out = torch.zeros(T, N, device="cpu", dtype=torch.float32)

    for t in range(T):
        # Recurrent input: weighted sum of upstream states.
        rec = torch.sparse.mm(adj, state.unsqueeze(1)).squeeze(1)
        # Drive voltage = bias + stim + recurrent (mapped into [VD_LO, VD_HI]).
        drive = vg1_center + drive_gain * (stim_t[t] + 0.5 * rec)
        # Recenter to envelope: shift so 0 -> midpoint 1.1 V
        vd_input = (drive - vg1_center) * 2.0 + 1.1
        resp = cell_response(cell, vd_input, vg1_bias)
        state = leak * state + (1 - leak) * resp
        states_out[t] = state.cpu()
    return states_out


# ----------------------------------------------------------------------
# 5. Metrics.
# ----------------------------------------------------------------------
def kuramoto_r(states: np.ndarray) -> float:
    # Use Hilbert-like phase estimate via analytic signal on each channel.
    # For speed, use sign(state) and instantaneous phase via finite diff.
    # Simpler: take a sample of channels, normalize, treat each as oscillator-like.
    sel = states[:, ::max(1, states.shape[1] // 64)]  # ~64 channels
    s = (sel - sel.mean(0)) / (sel.std(0) + 1e-8)
    # Phase: atan2 of (s, derivative of s)
    ds = np.gradient(s, axis=0)
    phase = np.arctan2(ds, s)
    R = np.abs(np.exp(1j * phase).mean(axis=1))  # over channels per time
    return float(R.mean())


def largest_lyapunov(states: np.ndarray, tau: int = 5) -> float:
    """Rosenstein-style: pick reference channel(s), compute log mean divergence
    of nearest neighbors in embedding space.

    Cheap proxy: take 1 channel, build delay-coord, find nearest neighbor,
    track divergence over time. Average over a few channels.
    """
    T = states.shape[0]
    if T < 200:
        return float("nan")
    lyaps = []
    n_channels_to_try = min(8, states.shape[1])
    chan_idx = np.linspace(0, states.shape[1] - 1, n_channels_to_try, dtype=int)
    for ch in chan_idx:
        x = states[:, ch]
        if x.std() < 1e-6:
            continue
        m = 5  # embedding dim
        emb = np.stack([x[i:T - m * tau + i] for i in range(0, m * tau, tau)], axis=1)
        # nearest neighbor in first half
        half = emb.shape[0] // 2
        refs = emb[:half:max(1, half // 32)]  # ~32 reference points
        divs = []
        for r_idx, ref in enumerate(refs):
            d = np.linalg.norm(emb[:half] - ref, axis=1)
            d[max(0, r_idx - 5):r_idx + 6] = np.inf
            nn = int(np.argmin(d))
            # track divergence over next 30 steps
            n_track = min(30, emb.shape[0] - max(r_idx, nn) - 1)
            if n_track < 5:
                continue
            dist = np.linalg.norm(
                emb[r_idx + 1:r_idx + 1 + n_track] - emb[nn + 1:nn + 1 + n_track], axis=1
            )
            dist = np.maximum(dist, 1e-10)
            log_d = np.log(dist)
            # slope of log divergence
            slope = np.polyfit(np.arange(n_track), log_d, 1)[0]
            divs.append(slope)
        if divs:
            lyaps.append(np.median(divs))
    return float(np.median(lyaps)) if lyaps else float("nan")


def spectral_entropy(states: np.ndarray) -> float:
    # PSD averaged across channels, normalize, Shannon entropy.
    sel = states[:, ::max(1, states.shape[1] // 64)]
    psd = np.abs(np.fft.rfft(sel - sel.mean(0), axis=0)) ** 2
    psd_mean = psd.mean(axis=1)
    p = psd_mean / (psd_mean.sum() + 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    H = -np.sum(p * np.log2(p))
    return float(H / np.log2(len(p)))  # normalize to [0, 1]


def narma30_r2(states: np.ndarray, stim: np.ndarray) -> float:
    """NARMA-30 task: target is NARMA-30 sequence driven by stim (rescaled).
    Linear readout on reservoir states; report r^2.
    """
    T = states.shape[0]
    u = (stim - stim.min()) / (stim.max() - stim.min() + 1e-8) * 0.5  # [0, 0.5]
    y = np.zeros(T, dtype=np.float32)
    n = 30
    for t in range(n, T - 1):
        y[t + 1] = (0.2 * y[t] +
                    0.004 * y[t] * np.sum(y[t - n + 1:t + 1]) +
                    1.5 * u[t - n + 1] * u[t] + 0.001)
    # ridge regression on states
    burn = 100
    X = states[burn:]
    yt = y[burn:]
    # add bias
    X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    XtX = X.T @ X + 1e-3 * np.eye(X.shape[1], dtype=np.float32)
    w = np.linalg.solve(XtX, X.T @ yt)
    pred = X @ w
    ss_res = np.sum((yt - pred) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def rate_and_cv(states: np.ndarray) -> Tuple[float, float]:
    # "Spike" proxy: zero-crossings of mean-centered state per channel.
    z = states - states.mean(0)
    crossings = ((z[:-1] * z[1:]) < 0).sum(axis=0)  # per channel
    rates = crossings / states.shape[0]
    cv = float(rates.std() / (rates.mean() + 1e-8))
    return float(rates.mean()), cv


def is_edge_of_chaos(lyap: float) -> bool:
    return not math.isnan(lyap) and -0.05 <= lyap <= 0.05


# ----------------------------------------------------------------------
# 6. Main sweep
# ----------------------------------------------------------------------
TOPOLOGIES = ["ER_SPARSE", "ER_DENSE", "WS_SMALL_WORLD", "BA_SCALE_FREE",
              "MODULAR_4BLOCK", "HIERARCHICAL_3LEVEL"]
STIMULI = ["WHITE_NOISE", "1F_NOISE", "SINE_PLUS_NOISE", "POISSON_SPIKES",
           "MACKEY_GLASS"]
HETEROGENEITIES = [0.0, 0.02, 0.05, 0.10]


def run_condition(cell: DistilledMLP, topo: str, stim: str, sigma_jit: float,
                  N: int, T: int, dt: float, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    adj = make_topology(topo, N, rng)
    # Per-node VG1 jitter (centered on 0.40, clipped to envelope).
    vg1_bias = 0.40 + sigma_jit * torch.randn(N, device=DEVICE)
    vg1_bias = vg1_bias.clamp(VG1_LO, VG1_HI)

    stim_arr = make_stim(stim, T, dt, rng)
    t0 = time.time()
    states = simulate(cell, adj, vg1_bias, stim_arr, dt=dt).numpy()
    sim_dt = time.time() - t0

    R = kuramoto_r(states)
    lyap = largest_lyapunov(states)
    H = spectral_entropy(states)
    narma_r2 = narma30_r2(states, stim_arr)
    rate, cv = rate_and_cv(states)
    edge = is_edge_of_chaos(lyap)
    trivial_sync = R > 0.95 or R < 0.10

    return {
        "topology": topo, "stimulus": stim, "heterogeneity": sigma_jit,
        "kuramoto_R": R, "lyapunov": lyap, "spectral_entropy": H,
        "narma30_r2": narma_r2, "mean_rate": rate, "cv_rate": cv,
        "edge_of_chaos": bool(edge), "trivial_sync": bool(trivial_sync),
        "sim_time_s": sim_dt, "N": N, "T": T,
    }


def main() -> None:
    cell = load_cell()

    # Budget sizing. Default N=2048, T=4000 steps (dt=0.005 -> 20 s).
    # 6*5*4 = 120 conditions. We measure first then scale.
    N = 2048
    T = 4000
    dt = 0.005

    # Quick smoke
    print("[zoo] smoke test (ER_SPARSE/WHITE_NOISE/het=0.02)...")
    smoke = run_condition(cell, "ER_SPARSE", "WHITE_NOISE", 0.02, N, T, dt, seed=0)
    print(f"  R={smoke['kuramoto_R']:.3f}  lyap={smoke['lyapunov']:+.3f}  "
          f"H={smoke['spectral_entropy']:.3f}  NARMA r2={smoke['narma30_r2']:+.3f}  "
          f"rate={smoke['mean_rate']:.3f}  sim={smoke['sim_time_s']:.1f}s")
    per_cond_estimate = smoke["sim_time_s"]
    print(f"[zoo] estimated total: {per_cond_estimate * 120 / 60:.1f} min")

    results = []
    total = len(TOPOLOGIES) * len(STIMULI) * len(HETEROGENEITIES)
    i = 0
    t_start = time.time()
    for topo in TOPOLOGIES:
        for stim in STIMULI:
            for het in HETEROGENEITIES:
                i += 1
                seed = hash((topo, stim, het)) & 0xffffffff
                r = run_condition(cell, topo, stim, het, N, T, dt, seed=seed)
                results.append(r)
                elapsed = time.time() - t_start
                eta = elapsed / i * (total - i)
                print(f"[{i:3d}/{total}] {topo:20s} | {stim:16s} | h={het:.2f} | "
                      f"R={r['kuramoto_R']:.3f} lyap={r['lyapunov']:+.3f} "
                      f"H={r['spectral_entropy']:.3f} NARMA={r['narma30_r2']:+.3f} "
                      f"rate={r['mean_rate']:.3f} edge={int(r['edge_of_chaos'])} "
                      f"trivial={int(r['trivial_sync'])} | eta={eta/60:.1f}m")

    # Save phase diagram.
    (OUT_DIR / "phase_diagram_3d.json").write_text(json.dumps({
        "topologies": TOPOLOGIES,
        "stimuli": STIMULI,
        "heterogeneities": HETEROGENEITIES,
        "N": N, "T": T, "dt": dt,
        "results": results,
        "device": str(DEVICE),
    }, indent=2))
    print(f"[zoo] saved phase_diagram_3d.json with {len(results)} conditions.")

    # Gate checks.
    any_edge = any(r["edge_of_chaos"] for r in results)
    max_narma = max(r["narma30_r2"] for r in results)
    all_trivial = all(r["trivial_sync"] for r in results)

    # Verdict file
    top_edge = sorted([r for r in results if r["edge_of_chaos"]],
                      key=lambda r: r["narma30_r2"], reverse=True)[:5]
    top_narma = sorted(results, key=lambda r: r["narma30_r2"], reverse=True)[:5]
    trivial_count = sum(1 for r in results if r["trivial_sync"])

    lines = [
        "# Pillar III — Topology Zoo Verdict",
        "",
        f"- Conditions: {len(results)} (6 topologies x 5 stimuli x 4 heterogeneity levels)",
        f"- N={N} nodes, T={T} steps, dt={dt}",
        f"- Cell: distilled MLP surrogate of Sebas's Pillar V ensemble  "
        f"(val median |res|={torch.load(OUT_DIR / 'distilled_mlp.pt', weights_only=True)['val_median_abs_dec']:.4f} dec)",
        "",
        "## Pre-registered gates",
        f"- ANY edge-of-chaos (Lyap in [-0.05, 0.05]): "
        f"**{'PASS' if any_edge else 'FAIL'}** ({sum(r['edge_of_chaos'] for r in results)}/{len(results)} conditions)",
        f"- ANY NARMA-30 r^2 > 0.8: **{'PASS' if max_narma > 0.8 else 'FAIL'}** (max={max_narma:.3f})",
        f"- KILLSHOT (all trivial sync): **{'TRIGGERED' if all_trivial else 'not triggered'}** "
        f"({trivial_count}/{len(results)} trivial)",
        "",
        "## Top 5 edge-of-chaos conditions (sorted by NARMA-30 r^2)",
        "| topo | stim | het | R | Lyap | H | NARMA | rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in top_edge:
        lines.append(f"| {r['topology']} | {r['stimulus']} | {r['heterogeneity']:.2f} | "
                     f"{r['kuramoto_R']:.3f} | {r['lyapunov']:+.3f} | "
                     f"{r['spectral_entropy']:.3f} | {r['narma30_r2']:+.3f} | "
                     f"{r['mean_rate']:.3f} |")
    lines += ["", "## Top 5 NARMA-30 conditions (any Lyap)",
              "| topo | stim | het | R | Lyap | H | NARMA | rate |",
              "|---|---|---|---|---|---|---|---|"]
    for r in top_narma:
        lines.append(f"| {r['topology']} | {r['stimulus']} | {r['heterogeneity']:.2f} | "
                     f"{r['kuramoto_R']:.3f} | {r['lyapunov']:+.3f} | "
                     f"{r['spectral_entropy']:.3f} | {r['narma30_r2']:+.3f} | "
                     f"{r['mean_rate']:.3f} |")
    lines += ["",
              "## Trivial-sync share by topology",
              "| topology | n_trivial / 20 |",
              "|---|---|"]
    for t in TOPOLOGIES:
        ct = sum(1 for r in results if r["topology"] == t and r["trivial_sync"])
        lines.append(f"| {t} | {ct}/20 |")
    (OUT_DIR / "verdict.md").write_text("\n".join(lines))
    print("[zoo] saved verdict.md")


if __name__ == "__main__":
    main()
