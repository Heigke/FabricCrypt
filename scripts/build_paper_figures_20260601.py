#!/usr/bin/env python3
"""Build paper figures: identity heatmap (real + 8-chassis mock) and spoof-defense CI bars.

Outputs to paper_drafts/figures/ at 300 dpi.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
DATA = REPO / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "embodiment13"
OUT = REPO / "paper_drafts" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260601)


def cosine_dist_matrix(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    U = M / norms
    sim = U @ U.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def load_sigs():
    ik = np.load(DATA / "ikaros_sig_v2.npz")["vec"]
    da = np.load(DATA / "daedalus_sig_v2.npz")["vec"]
    return ik, da


# --------------------------------------------------------------------------
# Figure 1 — real heatmap (2 chassis × 10 reps)
# --------------------------------------------------------------------------
def fig_real_heatmap():
    ik, da = load_sigs()
    M = np.vstack([ik, da])  # (20, 290)
    D = cosine_dist_matrix(M)

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(D, cmap="viridis", vmin=0.0, vmax=max(0.5, D.max()))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Cosine distance")

    # Block separator
    ax.axhline(9.5, color="white", lw=1.2)
    ax.axvline(9.5, color="white", lw=1.2)

    ax.set_xticks([4.5, 14.5])
    ax.set_xticklabels(["ikaros (10 reps)", "daedalus (10 reps)"])
    ax.set_yticks([4.5, 14.5])
    ax.set_yticklabels(["ikaros", "daedalus"], rotation=90, va="center")

    # Mean values
    intra_ik = D[:10, :10][np.triu_indices(10, 1)].mean()
    intra_da = D[10:, 10:][np.triu_indices(10, 1)].mean()
    inter = D[:10, 10:].mean()
    ax.set_title(
        f"Phase 13 identity signatures (n=290 features)\n"
        f"intra-ikaros={intra_ik:.3f}  intra-daedalus={intra_da:.3f}  "
        f"inter={inter:.3f}",
        fontsize=10,
    )

    # Annotation for diagonal
    ax.annotate(
        "diagonal: self-identity (d=0)",
        xy=(0.5, 0.5), xytext=(3, -3),
        fontsize=8, color="white",
        arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
    )

    fig.tight_layout()
    fig.savefig(OUT / "identity_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "identity_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    return dict(intra_ikaros=float(intra_ik), intra_daedalus=float(intra_da),
                inter=float(inter))


# --------------------------------------------------------------------------
# Figure 2 — MOCK 8-chassis projection (synthetic rotations)
# --------------------------------------------------------------------------
def fig_mock_8chassis():
    ik, da = load_sigs()
    real = [("ikaros", ik), ("daedalus", da)]

    # Synthesize 6 virtual chassis via random orthogonal-ish rotations + per-host shift
    mocks = []
    for i in range(6):
        # Random rotation in feature space (small angle ortho) + idiosyncratic mean shift
        base = ik if i % 2 == 0 else da
        # Per-chassis "fabric vector": Gaussian shift, scale similar to inter-cluster separation
        shift = RNG.normal(0, 0.5, size=base.shape[1])
        # Per-rep noise matching empirical intra-cluster variance
        intra_std = base.std(axis=0).mean()
        noise = RNG.normal(0, intra_std, size=base.shape)
        virt = base + shift[None, :] + noise
        mocks.append((f"mock_{i+1}", virt))

    all_sigs = real + mocks
    M = np.vstack([v for _, v in all_sigs])
    D = cosine_dist_matrix(M)

    n_chassis = len(all_sigs)
    reps_per = 10

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(D, cmap="viridis", vmin=0.0, vmax=max(0.5, D.max()))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Cosine distance")

    # Block separators
    for i in range(1, n_chassis):
        ax.axhline(i * reps_per - 0.5, color="white", lw=0.6, alpha=0.7)
        ax.axvline(i * reps_per - 0.5, color="white", lw=0.6, alpha=0.7)

    tick_pos = [i * reps_per + 4.5 for i in range(n_chassis)]
    tick_labels = [name for name, _ in all_sigs]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels, fontsize=8)

    ax.set_title(
        "MOCK / PROJECTION — 8-chassis identity heatmap\n"
        "(only ikaros, daedalus are real; mock_1..6 are synthetic rotations of real signatures)",
        fontsize=10,
    )

    # Big watermark
    ax.text(
        0.5, 0.5, "MOCK",
        transform=ax.transAxes,
        fontsize=80, color="red", alpha=0.18,
        ha="center", va="center", rotation=30, weight="bold",
    )

    fig.tight_layout()
    fig.savefig(OUT / "identity_heatmap_8chassis_mock.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "identity_heatmap_8chassis_mock.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3 — Phase 14C spoof-defense bars with bootstrap CI
# --------------------------------------------------------------------------
def fig_spoof_bars():
    # Phase 14C results
    results = [
        # (name, accept_rate, n_trials, expected_pass)
        ("honest_own",                 1.000, 500, True),
        ("honest_own_wrong_nonce",     0.006, 500, False),
        ("daedalus_peer",              0.020, 500, False),
        ("static_replay",              0.006, 500, False),
        ("static_replay_correct_nonce",1.000, 500, True),  # known-bad: same-nonce replay
        ("dynamic_replay",             0.012, 500, False),
        ("nonce_only_mismatch",        0.006, 500, False),
    ]

    rng = np.random.default_rng(42)
    boot = 2000
    names, rates, los, his, expected = [], [], [], [], []
    for name, p, n, exp in results:
        # Wilson-style nonparametric bootstrap of Bernoulli(p) with n trials
        # Simulate observed successes
        k = int(round(p * n))
        # Generate bootstrap samples by resampling 0/1 with observed k/n
        sample = np.zeros(n, dtype=int)
        sample[:k] = 1
        rng.shuffle(sample)
        boots = rng.choice(sample, size=(boot, n), replace=True).mean(axis=1)
        lo, hi = np.quantile(boots, [0.025, 0.975])
        names.append(name)
        rates.append(p * 100)
        los.append(p * 100 - lo * 100)
        his.append(hi * 100 - p * 100)
        expected.append(exp)

    colors = ["#2ca02c" if e else "#d62728" for e in expected]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, rates, yerr=[los, his], capsize=4, color=colors,
                  edgecolor="black", linewidth=0.6)

    # Pre-reg gate lines
    ax.axhline(95, color="green", ls="--", lw=1.0, label="pre-reg gate: ≥95% (honest pass)")
    ax.axhline(5,  color="red",   ls="--", lw=1.0, label="pre-reg gate: ≤5% (spoof reject)")
    ax.axhline(10, color="orange",ls=":",  lw=1.0, label="pre-reg gate: ≤10% (relaxed)")

    ax.set_ylabel("Acceptance rate (%)")
    ax.set_ylim(-2, 108)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_title("Phase 14C — FabricCrypt spoof-defense (n=500/condition, 95% bootstrap CI)")
    ax.legend(loc="center right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)

    # Annotate value on top
    for xi, r, h in zip(x, rates, his):
        ax.text(xi, r + h + 2, f"{r:.1f}%", ha="center", va="bottom", fontsize=8)

    # Footnote
    ax.text(
        0.01, -0.30,
        "Green = expected pass; red = expected reject. "
        "static_replay_correct_nonce is the diagnostic known-bad: "
        "replay with matching nonce DOES pass — confirms nonce binding works as designed.",
        transform=ax.transAxes, fontsize=7.5, style="italic", wrap=True,
    )

    fig.tight_layout()
    fig.savefig(OUT / "spoof_defense_bars.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "spoof_defense_bars.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    stats = fig_real_heatmap()
    fig_mock_8chassis()
    fig_spoof_bars()
    print("OK")
    print(json.dumps(stats, indent=2))
