#!/usr/bin/env python3
"""DS-N13: Systematic topology zoo for NS-RAM on Mackey-Glass τ=17.

Six topologies × multiple seeds, plus density sweep for the winner, plus an
LSTM baseline (~50 hidden units). Reuses:
  - NSRAMArray (vectorized NS-RAM cell bank, from DS_N11)
  - SynapticNetwork (S3 sparse event-driven glue)

Output: results/DS_N13_topology_zoo/{topology_comparison.json,
spike_raster_per_topology.png, summary.md, density_sweep.json}.
"""
from __future__ import annotations
import os, sys, json, time, math
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

from pathlib import Path
import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from S3_network_glue import (
    topology_er, topology_small_world, topology_reservoir, SynapticNetwork,
)
from DS_N11_predictive_coding import NSRAMArray

OUT = REPO / "results" / "DS_N13_topology_zoo"
OUT.mkdir(parents=True, exist_ok=True)


# ────────────────────────── Mackey-Glass ─────────────────────────────
def mackey_glass(T: int, tau: int = 17, beta: float = 0.2, gamma: float = 0.1,
                 n: int = 10, dt: float = 1.0, seed: int = 0) -> np.ndarray:
    """Discrete delay equation: x_{t+1} = x_t + dt*(beta*x_{t-tau}/(1+x_{t-tau}^n) - gamma*x_t)."""
    rng = np.random.default_rng(seed)
    burn = 1000
    L = T + burn + tau + 1
    x = np.zeros(L)
    x[:tau + 1] = 1.2 + 0.1 * rng.standard_normal(tau + 1)
    for t in range(tau, L - 1):
        x[t + 1] = x[t] + dt * (beta * x[t - tau] / (1.0 + x[t - tau] ** n) - gamma * x[t])
    y = x[burn + tau:burn + tau + T]
    return ((y - y.mean()) / y.std()).astype(np.float32)


# ────────────────────────── Extra topologies ─────────────────────────
def topology_lattice2d(N: int, seed: int = 0, w_scale: float = 0.05) -> sp.csr_matrix:
    """Toroidal 2D grid, 8-connected (Moore neighborhood)."""
    side = int(round(math.sqrt(N)))
    N2 = side * side
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for r in range(side):
        for c in range(side):
            i = r * side + c
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    j = ((r + dr) % side) * side + (c + dc) % side
                    rows.append(i); cols.append(j)
                    vals.append(rng.normal(0.0, w_scale))
    W = sp.csr_matrix((vals, (rows, cols)), shape=(N2, N2))
    if N2 < N:
        # pad to N rows/cols with empty
        W = sp.csr_matrix((W.toarray(), ), shape=(N, N)) if False else _pad_csr(W, N)
    return W


def _pad_csr(W: sp.csr_matrix, N: int) -> sp.csr_matrix:
    if W.shape[0] == N:
        return W
    out = sp.lil_matrix((N, N))
    out[:W.shape[0], :W.shape[1]] = W
    return out.tocsr()


def topology_ring(N: int, k: int = 4, p_long: float = 0.01, seed: int = 0,
                  w_scale: float = 0.05) -> sp.csr_matrix:
    """1D ring with k-nearest neighbors + sparse longer random connections."""
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for i in range(N):
        for dk in range(1, k + 1):
            for j in ((i + dk) % N, (i - dk) % N):
                rows.append(i); cols.append(j); vals.append(rng.normal(0.0, w_scale))
        # sparse long-range
        if rng.random() < p_long:
            j = int(rng.integers(0, N))
            if j != i:
                rows.append(i); cols.append(j); vals.append(rng.normal(0.0, w_scale))
    W = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    W.sum_duplicates()
    return W


def topology_scale_free(N: int, m: int = 3, seed: int = 0,
                        w_scale: float = 0.05) -> sp.csr_matrix:
    """Barabási-Albert preferential attachment, directed.
    Each new node attaches m edges to existing nodes with probability proportional to degree.
    """
    rng = np.random.default_rng(seed)
    m = max(1, m)
    rows, cols, vals = [], [], []
    # Seed clique of size m+1
    for i in range(m + 1):
        for j in range(m + 1):
            if i == j:
                continue
            rows.append(i); cols.append(j); vals.append(rng.normal(0.0, w_scale))
    # degree array
    deg = np.zeros(N)
    for r in rows:
        deg[r] += 1
    for new in range(m + 1, N):
        # sample m targets without replacement, proportional to deg[:new]
        p = deg[:new].copy()
        p_sum = p.sum()
        if p_sum <= 0:
            targets = rng.choice(new, size=m, replace=False)
        else:
            p = p / p_sum
            targets = rng.choice(new, size=min(m, new), replace=False, p=p)
        for t in targets:
            rows.append(new); cols.append(int(t))
            vals.append(rng.normal(0.0, w_scale))
            # also back-edge to maintain directed-but-coupled flavor
            rows.append(int(t)); cols.append(new)
            vals.append(rng.normal(0.0, w_scale))
            deg[new] += 1
            deg[t] += 1
    W = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    W.sum_duplicates()
    return W


def topology_layered(N: int, n_layers: int = 3, density: float = 0.01,
                     seed: int = 0, w_scale: float = 0.05) -> sp.csr_matrix:
    """Feedforward layered network with dense (sub-)connections between adjacent layers
    plus a small recurrent ER component inside each layer."""
    rng = np.random.default_rng(seed)
    layer_sz = N // n_layers
    layers = [np.arange(l * layer_sz, (l + 1) * layer_sz) for l in range(n_layers)]
    # fold remainder into last layer
    if layer_sz * n_layers < N:
        layers[-1] = np.arange(layers[-1][0], N)
    rows, cols, vals = [], [], []
    for l in range(n_layers - 1):
        src = layers[l]; dst = layers[l + 1]
        # cross-layer density (fairly dense)
        nnz = int(0.05 * len(src) * len(dst))
        r = rng.integers(0, len(src), size=nnz)
        c = rng.integers(0, len(dst), size=nnz)
        rows.extend(src[r].tolist()); cols.extend(dst[c].tolist())
        vals.extend(rng.normal(0.0, w_scale, size=nnz).tolist())
    # weak intra-layer recurrence (ER)
    for l in layers:
        nl = len(l)
        nnz = max(int(density * nl * nl), nl)
        r = rng.integers(0, nl, size=nnz)
        c = rng.integers(0, nl, size=nnz)
        mask = r != c
        r, c = r[mask], c[mask]
        rows.extend(l[r].tolist()); cols.extend(l[c].tolist())
        vals.extend((0.5 * rng.normal(0.0, w_scale, size=r.size)).tolist())
    W = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    W.sum_duplicates()
    return W


def build_topology(name: str, N: int, seed: int, density: float = 0.01) -> sp.csr_matrix:
    if name == "er":
        return topology_er(N, density=density, seed=seed)
    if name == "ring":
        return topology_ring(N, k=4, p_long=0.01, seed=seed)
    if name == "lattice2d":
        return topology_lattice2d(N, seed=seed)
    if name == "small_world":
        return topology_small_world(N, k=4, p_rewire=0.1, seed=seed)
    if name == "scale_free":
        return topology_scale_free(N, m=3, seed=seed)
    if name == "layered":
        return topology_layered(N, n_layers=3, density=density, seed=seed)
    raise ValueError(name)


# ─────────────────── NS-RAM + network run, ridge readout ──────────────
def run_nsram_network(signal: np.ndarray, W: sp.csr_matrix, N: int,
                      dt: float = 1e-6, sub_per_sample: int = 1,
                      seed: int = 0, record_raster_steps: int = 0,
                      n_readout: int = 500, inject_gain: float = 0.02):
    arr = NSRAMArray(N=N, seed=seed)
    net = SynapticNetwork(W, tau_syn=5e-6, vg2_gain=0.0, qbody_gain=1e-15)
    # We will route synaptic accumulator self.s as a direct evidence current adder
    # (vg2_gain=0 to avoid touching thresholds incidentally; we use the trace s).
    T = len(signal)
    rng = np.random.default_rng(seed + 9999)
    readout_idx = rng.choice(N, size=min(n_readout, N), replace=False)
    feats = np.zeros((T, 2 * len(readout_idx) + 1), dtype=np.float32)
    total_spikes = 0
    raster_rows = []  # (t, spike_idx)
    for t in range(T):
        x = float(signal[t])
        # apply network's exponential synapse trace as extra evidence current
        # (NS-RAM step sums I_evd from x; we add net.s scaled to current units)
        for _ in range(sub_per_sample):
            # Step 1: standard NS-RAM update
            spk, Vm, Q, rate = arr.step(x, dt)
            # Step 2: feed spike events into network, get dVG2/dQbody back
            spike_idx = np.flatnonzero(spk)
            total_spikes += int(spike_idx.size)
            # Convert sparse spike rows of W into post-synaptic injection (Vm bump)
            if spike_idx.size > 0:
                sub = W[spike_idx, :]
                contrib = np.asarray(sub.sum(axis=0)).ravel()
            else:
                contrib = np.zeros(N)
            # exponential synapse trace
            net.s *= np.exp(-dt / net.tau_syn)
            net.s += contrib
            # inject into membrane: large enough to perturb spiking dynamics
            # (W weights ~0.05 std × Dale sign, contrib O(0.1-1) per spiking neighbor)
            arr.Vm += inject_gain * net.s
            # record raster only on first sub-step
            if record_raster_steps and t < record_raster_steps:
                for si in spike_idx[:64]:  # cap per step
                    raster_rows.append((t, int(si)))
        K = len(readout_idx)
        feats[t, :K]        = arr.rate[readout_idx]
        feats[t, K:2*K]     = arr.Q[readout_idx]
        feats[t, 2*K]       = x
    return feats, total_spikes, raster_rows


def ridge_predict(feats: np.ndarray, target: np.ndarray, train_frac: float = 0.7,
                  lam: float = 10.0):
    T = len(target)
    n_tr = int(T * train_frac)
    X_tr, X_te = feats[:n_tr - 1], feats[n_tr:-1]
    y_tr, y_te = target[1:n_tr], target[n_tr + 1:]
    sd_full = X_tr.std(0)
    keep = sd_full > 1e-6
    X_tr = X_tr[:, keep]; X_te = X_te[:, keep]
    mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-8
    Xtr = ((X_tr - mu) / sd).astype(np.float64)
    Xte = ((X_te - mu) / sd).astype(np.float64)
    np.clip(Xtr, -6, 6, out=Xtr); np.clip(Xte, -6, 6, out=Xte)
    Xtr = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], axis=1)
    Xte = np.concatenate([Xte, np.ones((Xte.shape[0], 1))], axis=1)
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    b = Xtr.T @ y_tr.astype(np.float64)
    w = np.linalg.solve(A, b)
    pred = Xte @ w
    rmse = float(np.sqrt(np.mean((pred - y_te) ** 2)))
    nrmse = rmse / (y_te.std() + 1e-12)
    return float(nrmse), pred, y_te


# ─────────────────────────── LSTM baseline ───────────────────────────
def lstm_baseline(signal: np.ndarray, train_frac: float = 0.7, hidden: int = 50,
                  win: int = 16, epochs: int = 100, lr: float = 5e-3, seed: int = 0):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = len(signal); n_tr = int(T * train_frac)

    def make_xy(s):
        X = np.stack([s[i:i + win] for i in range(len(s) - win - 1)], axis=0)
        Y = s[win + 1:]
        return X[..., None], Y

    Xtr, Ytr = make_xy(signal[:n_tr])
    Xte, Yte = make_xy(signal[n_tr - win - 1:])
    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    Ytr = torch.tensor(Ytr, dtype=torch.float32, device=dev)
    Xte = torch.tensor(Xte, dtype=torch.float32, device=dev)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)
        def forward(self, x):
            h, _ = self.lstm(x)
            return self.fc(h[:, -1, :]).squeeze(-1)
    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n_params = sum(p.numel() for p in net.parameters())
    for ep in range(epochs):
        opt.zero_grad()
        pred = net(Xtr)
        loss = ((pred - Ytr) ** 2).mean()
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xte).cpu().numpy()
    m = min(len(pred), len(Yte))
    rmse = float(np.sqrt(np.mean((pred[:m] - Yte[:m]) ** 2)))
    nrmse = rmse / (Yte[:m].std() + 1e-12)
    return float(nrmse), int(n_params)


# ────────────────────────────── main ─────────────────────────────────
def main():
    T = 10_000
    N = 10_000
    SEEDS = [0, 1, 2, 3]
    TOPOLOGIES = ["er", "ring", "lattice2d", "small_world", "scale_free", "layered"]
    DENSITY_DEFAULT = 0.01

    print(f"[DS-N13] T={T} N={N} seeds={SEEDS} topologies={TOPOLOGIES}", flush=True)
    signal = mackey_glass(T, tau=17, seed=42)
    print(f"[DS-N13] MG generated. mean={signal.mean():.3f} std={signal.std():.3f}", flush=True)

    results = {}
    diverged = {}
    energy = {}
    raster_per_topo = {}
    t0 = time.time()

    for topo in TOPOLOGIES:
        results[topo] = {"nrmse": [], "spikes": [], "wall_s": [], "nnz": []}
        for si, seed in enumerate(SEEDS):
            ts = time.time()
            W = build_topology(topo, N=N, seed=seed, density=DENSITY_DEFAULT)
            record = 200 if (si == 0) else 0
            feats, n_spk, raster = run_nsram_network(
                signal, W, N=N, dt=1e-6, sub_per_sample=1, seed=seed,
                record_raster_steps=record,
            )
            # divergence check
            div = bool(np.any(~np.isfinite(feats))) or float(np.max(np.abs(feats))) > 1e6
            nrmse, _, _ = ridge_predict(feats, signal) if not div else (float("nan"), None, None)
            wall = time.time() - ts
            results[topo]["nrmse"].append(nrmse)
            results[topo]["spikes"].append(int(n_spk))
            results[topo]["wall_s"].append(wall)
            results[topo]["nnz"].append(int(W.nnz))
            if si == 0:
                raster_per_topo[topo] = raster
            print(f"  [{topo} seed={seed}] nrmse={nrmse:.4f} spikes={n_spk} "
                  f"nnz={W.nnz} t={wall:.1f}s diverged={div}", flush=True)
            diverged.setdefault(topo, []).append(div)
        # summary stats
        arr = np.array(results[topo]["nrmse"], dtype=float)
        ok = arr[np.isfinite(arr)]
        results[topo]["nrmse_mean"] = float(ok.mean()) if ok.size else float("nan")
        results[topo]["nrmse_std"] = float(ok.std()) if ok.size else float("nan")
        results[topo]["spikes_mean"] = float(np.mean(results[topo]["spikes"]))
        results[topo]["E_J_mean"] = float(np.mean(results[topo]["spikes"]) * 21e-15)
        energy[topo] = results[topo]["E_J_mean"]
        results[topo]["any_diverged"] = bool(any(diverged[topo]))

    # LSTM baseline (50 hidden)
    print("[DS-N13] LSTM baseline (hidden=50)…", flush=True)
    lstm_nrmses = []
    for seed in SEEDS:
        nrmse, npar = lstm_baseline(signal, hidden=50, seed=seed)
        lstm_nrmses.append(nrmse)
        print(f"  [LSTM seed={seed}] nrmse={nrmse:.4f} params={npar}", flush=True)
    results["_lstm_baseline"] = {
        "nrmse": lstm_nrmses,
        "nrmse_mean": float(np.mean(lstm_nrmses)),
        "nrmse_std": float(np.std(lstm_nrmses)),
        "hidden": 50,
    }

    # winner
    finite = {k: v for k, v in results.items()
              if k in TOPOLOGIES and np.isfinite(v["nrmse_mean"])}
    winner = min(finite, key=lambda k: finite[k]["nrmse_mean"]) if finite else None
    print(f"[DS-N13] Winner topology: {winner}", flush=True)

    # density sweep on winner
    density_results = {}
    if winner is not None:
        print(f"[DS-N13] Density sweep for {winner}…", flush=True)
        for dens in [0.005, 0.01, 0.05]:
            sweep = {"nrmse": [], "spikes": [], "nnz": []}
            for seed in SEEDS:
                ts = time.time()
                W = build_topology(winner, N=N, seed=seed, density=dens)
                feats, n_spk, _ = run_nsram_network(
                    signal, W, N=N, dt=1e-6, sub_per_sample=1, seed=seed,
                )
                div = bool(np.any(~np.isfinite(feats)))
                nrmse, _, _ = ridge_predict(feats, signal) if not div else (float("nan"), None, None)
                sweep["nrmse"].append(nrmse)
                sweep["spikes"].append(int(n_spk))
                sweep["nnz"].append(int(W.nnz))
                print(f"  [{winner} d={dens} seed={seed}] nrmse={nrmse:.4f} "
                      f"nnz={W.nnz} t={time.time()-ts:.1f}s", flush=True)
            sweep["nrmse_mean"] = float(np.nanmean(sweep["nrmse"]))
            sweep["nrmse_std"] = float(np.nanstd(sweep["nrmse"]))
            density_results[f"d={dens}"] = sweep

    total_wall = time.time() - t0

    out = {
        "config": {"T": T, "N": N, "seeds": SEEDS, "topologies": TOPOLOGIES,
                   "density_default": DENSITY_DEFAULT, "task": "mackey_glass_tau17"},
        "results": results,
        "winner": winner,
        "wall_s": total_wall,
    }
    (OUT / "topology_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"Saved {OUT/'topology_comparison.json'}")

    if density_results:
        (OUT / "density_sweep.json").write_text(json.dumps(
            {"winner": winner, "sweep": density_results}, indent=2))
        print(f"Saved {OUT/'density_sweep.json'}")

    # raster plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ntopo = len(TOPOLOGIES)
        fig, axes = plt.subplots(ntopo, 1, figsize=(10, 1.6 * ntopo), sharex=True)
        for ax, topo in zip(axes, TOPOLOGIES):
            r = raster_per_topo.get(topo, [])
            if r:
                ts_, ids_ = zip(*r)
                ax.scatter(ts_, ids_, s=1.2, c="k", marker=".", alpha=0.6)
            ax.set_ylabel(topo, fontsize=9)
            ax.set_ylim(0, N)
            ax.set_yticks([])
        axes[-1].set_xlabel("timestep")
        plt.suptitle("DS-N13 spike rasters per topology (first 200 steps, seed 0)")
        plt.tight_layout()
        plt.savefig(OUT / "spike_raster_per_topology.png", dpi=130)
        print(f"Saved {OUT/'spike_raster_per_topology.png'}")
    except Exception as e:
        print("raster plot failed:", e)

    # summary.md
    md = ["# DS-N13 Topology Zoo — NS-RAM on Mackey-Glass τ=17\n"]
    md.append(f"- N={N}, T={T}, seeds={SEEDS}, density={DENSITY_DEFAULT}, wall={total_wall:.1f}s\n")
    md.append("\n## NRMSE per topology (mean ± std, 4 seeds)\n")
    md.append("| topology | NRMSE mean | NRMSE std | spikes mean | E (J) | diverged | nnz |")
    md.append("|---|---|---|---|---|---|---|")
    for topo in TOPOLOGIES:
        r = results[topo]
        md.append(f"| {topo} | {r['nrmse_mean']:.4f} | {r['nrmse_std']:.4f} | "
                  f"{r['spikes_mean']:.0f} | {r['E_J_mean']:.2e} | "
                  f"{r['any_diverged']} | {int(np.mean(r['nnz']))} |")
    md.append(f"\n**Winner**: `{winner}`\n")
    md.append(f"\n## LSTM baseline (hidden=50)\n")
    md.append(f"- NRMSE = {results['_lstm_baseline']['nrmse_mean']:.4f} "
              f"± {results['_lstm_baseline']['nrmse_std']:.4f}\n")
    if density_results:
        md.append("\n## Density sweep for winner\n")
        md.append("| density | NRMSE mean | NRMSE std | nnz |")
        md.append("|---|---|---|---|")
        for k, v in density_results.items():
            md.append(f"| {k} | {v['nrmse_mean']:.4f} | {v['nrmse_std']:.4f} | "
                      f"{int(np.mean(v['nnz']))} |")
    md.append("\n## Gates")
    any_div = any(results[t]["any_diverged"] for t in TOPOLOGIES)
    md.append(f"- All converged: {'PASS' if not any_div else 'FAIL'}")
    md.append(f"- Winner identified: {'PASS' if winner else 'FAIL'} ({winner})")
    nbest = results[winner]["nrmse_mean"] if winner else float("nan")
    md.append(f"- NS-RAM winner beats LSTM-50: "
              f"{'PASS' if nbest < results['_lstm_baseline']['nrmse_mean'] else 'FAIL'} "
              f"({nbest:.4f} vs {results['_lstm_baseline']['nrmse_mean']:.4f})")
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print(f"Saved {OUT/'summary.md'}")
    print(f"\nTOTAL wall: {total_wall:.1f}s")


if __name__ == "__main__":
    main()
