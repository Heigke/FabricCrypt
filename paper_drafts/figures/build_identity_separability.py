#!/usr/bin/env python3
"""Build identity-separability figures from Phase 13 signatures.

Replaces the cosine-heatmap (which has intra-spread > inter-gap) with
visualizations that EXPOSE the separation that the 290-dim logistic
classifier sees (100% LOO).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# --- Paths
REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
DATA = REPO / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment13"
OUT = REPO / "paper_drafts/figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- Load
ik = np.load(DATA / "ikaros_sig_v2.npz")["vec"]      # (10, 290)
da = np.load(DATA / "daedalus_sig_v2.npz")["vec"]    # (10, 290)
X = np.vstack([ik, da])                              # (20, 290)
y = np.array([0] * 10 + [1] * 10)
labels = ["ikaros", "daedalus"]
colors = ["#1f77b4", "#d62728"]
markers = ["o", "s"]

print(f"X: {X.shape}, y: {y.shape}")
print(f"Per-class means cosine: ik mean-norm={np.linalg.norm(ik.mean(0)):.3f}, "
      f"da mean-norm={np.linalg.norm(da.mean(0)):.3f}")

# --- Standardize for A,B,C,D,E
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# ============================================================
# A. PCA on standardized features
# ============================================================
pca = PCA(n_components=2)
Z_pca = pca.fit_transform(Xs)
ev = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(6, 5))
for c in (0, 1):
    m = y == c
    ax.scatter(Z_pca[m, 0], Z_pca[m, 1], c=colors[c], marker=markers[c],
               s=90, edgecolor="k", linewidth=0.6, label=labels[c], alpha=0.85)
ax.set_xlabel(f"PC1 ({ev[0]:.1f}% var)")
ax.set_ylabel(f"PC2 ({ev[1]:.1f}% var)")
ax.set_title("PCA (top-2) — standardized 290-D signature")
ax.legend(loc="best", frameon=True)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "pca_top2.png", dpi=300)
fig.savefig(OUT / "pca_top2.pdf")
plt.close(fig)

# ============================================================
# B. LDA 1-D projection
# ============================================================
lda = LinearDiscriminantAnalysis(n_components=1)
Z_lda = lda.fit_transform(Xs, y).ravel()

# Fisher score (1D): (mu1-mu0)^2 / (var0 + var1)
z0, z1 = Z_lda[y == 0], Z_lda[y == 1]
fisher = (z1.mean() - z0.mean()) ** 2 / (z0.var(ddof=1) + z1.var(ddof=1))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
bins = np.linspace(Z_lda.min() - 0.5, Z_lda.max() + 0.5, 12)
ax.hist(z0, bins=bins, color=colors[0], alpha=0.55, label=f"{labels[0]} (n=10)",
        edgecolor="k")
ax.hist(z1, bins=bins, color=colors[1], alpha=0.55, label=f"{labels[1]} (n=10)",
        edgecolor="k")
# 1D scatter underlay
ax.scatter(z0, np.full_like(z0, -0.4), c=colors[0], marker=markers[0], s=70,
           edgecolor="k", linewidth=0.5, zorder=3)
ax.scatter(z1, np.full_like(z1, -0.4), c=colors[1], marker=markers[1], s=70,
           edgecolor="k", linewidth=0.5, zorder=3)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("LDA axis 1")
ax.set_ylabel("count")
ax.set_title(f"LDA 1-D projection — Fisher score = {fisher:.2f}")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "lda_1d.png", dpi=300)
fig.savefig(OUT / "lda_1d.pdf")
plt.close(fig)

print(f"Fisher score (LDA 1D): {fisher:.3f}")

# ============================================================
# C. t-SNE 2-D (UMAP not installed)
# ============================================================
# Pre-reduce to ~10 PCs for stability with 20 samples
pre = PCA(n_components=min(10, X.shape[0] - 1)).fit_transform(Xs)
tsne = TSNE(n_components=2, perplexity=5, learning_rate="auto",
            init="pca", random_state=0)
Z_tsne = tsne.fit_transform(pre)

fig, ax = plt.subplots(figsize=(6, 5))
for c in (0, 1):
    m = y == c
    ax.scatter(Z_tsne[m, 0], Z_tsne[m, 1], c=colors[c], marker=markers[c],
               s=90, edgecolor="k", linewidth=0.6, label=labels[c], alpha=0.85)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title("t-SNE 2-D (perplexity=5, PCA-init)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "umap_2d.png", dpi=300)   # name kept for paper-figure consistency
fig.savefig(OUT / "umap_2d.pdf")
plt.close(fig)

# ============================================================
# D. Top-6 most-discriminant features (RandomForest importances)
# ============================================================
rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
rf.fit(Xs, y)
imp = rf.feature_importances_
top6 = np.argsort(imp)[::-1][:6]
print("Top-6 feature indices:", top6.tolist())
print("Top-6 importances:", imp[top6].round(4).tolist())

# 2x3 grid of pairs: (0,1) (2,3) (4,5)
pairs = [(top6[0], top6[1]), (top6[2], top6[3]), (top6[4], top6[5]),
         (top6[0], top6[2]), (top6[1], top6[3]), (top6[0], top6[5])]

fig, axes = plt.subplots(2, 3, figsize=(11, 7))
for ax, (i, j) in zip(axes.ravel(), pairs):
    for c in (0, 1):
        m = y == c
        ax.scatter(Xs[m, i], Xs[m, j], c=colors[c], marker=markers[c],
                   s=70, edgecolor="k", linewidth=0.5, label=labels[c], alpha=0.85)
    ax.set_xlabel(f"f{i} (imp={imp[i]:.3f})")
    ax.set_ylabel(f"f{j} (imp={imp[j]:.3f})")
    ax.grid(alpha=0.3)
axes[0, 0].legend(loc="best")
fig.suptitle("Top discriminant feature pairs (RF importances on standardized 290-D)")
fig.tight_layout()
fig.savefig(OUT / "top_features_scatter.png", dpi=300)
fig.savefig(OUT / "top_features_scatter.pdf")
plt.close(fig)

# ============================================================
# E. Combined headline figure (2x2)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# PCA
ax = axes[0, 0]
for c in (0, 1):
    m = y == c
    ax.scatter(Z_pca[m, 0], Z_pca[m, 1], c=colors[c], marker=markers[c],
               s=90, edgecolor="k", linewidth=0.6, label=labels[c], alpha=0.85)
ax.set_xlabel(f"PC1 ({ev[0]:.1f}% var)")
ax.set_ylabel(f"PC2 ({ev[1]:.1f}% var)")
ax.set_title("(a) PCA on standardized 290-D")
ax.legend()
ax.grid(alpha=0.3)

# LDA
ax = axes[0, 1]
ax.hist(z0, bins=bins, color=colors[0], alpha=0.55, label=labels[0], edgecolor="k")
ax.hist(z1, bins=bins, color=colors[1], alpha=0.55, label=labels[1], edgecolor="k")
ax.scatter(z0, np.full_like(z0, -0.4), c=colors[0], marker=markers[0], s=60,
           edgecolor="k", linewidth=0.5, zorder=3)
ax.scatter(z1, np.full_like(z1, -0.4), c=colors[1], marker=markers[1], s=60,
           edgecolor="k", linewidth=0.5, zorder=3)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("LDA axis 1")
ax.set_ylabel("count")
ax.set_title(f"(b) LDA 1-D — Fisher = {fisher:.2f}")
ax.legend()
ax.grid(alpha=0.3)

# t-SNE
ax = axes[1, 0]
for c in (0, 1):
    m = y == c
    ax.scatter(Z_tsne[m, 0], Z_tsne[m, 1], c=colors[c], marker=markers[c],
               s=90, edgecolor="k", linewidth=0.6, label=labels[c], alpha=0.85)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title("(c) t-SNE 2-D (perp=5)")
ax.legend()
ax.grid(alpha=0.3)

# Top-feature pair
ax = axes[1, 1]
i, j = top6[0], top6[1]
for c in (0, 1):
    m = y == c
    ax.scatter(Xs[m, i], Xs[m, j], c=colors[c], marker=markers[c],
               s=90, edgecolor="k", linewidth=0.6, label=labels[c], alpha=0.85)
ax.set_xlabel(f"feature {i} (imp={imp[i]:.3f})")
ax.set_ylabel(f"feature {j} (imp={imp[j]:.3f})")
ax.set_title("(d) Top-2 RF-important features")
ax.legend()
ax.grid(alpha=0.3)

fig.suptitle("Identity separability of 290-D chassis signature (10 reps × 2 chassis)",
             fontsize=13, y=1.00)
fig.tight_layout()
fig.savefig(OUT / "identity_separability.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "identity_separability.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================
# F. PCA on RAW features (no standardization)
# ============================================================
pca_raw = PCA(n_components=2)
Z_raw = pca_raw.fit_transform(X)
ev_raw = pca_raw.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(6, 5))
for c in (0, 1):
    m = y == c
    ax.scatter(Z_raw[m, 0], Z_raw[m, 1], c=colors[c], marker=markers[c],
               s=90, edgecolor="k", linewidth=0.6, label=labels[c], alpha=0.85)
ax.set_xlabel(f"PC1 ({ev_raw[0]:.1f}% var)")
ax.set_ylabel(f"PC2 ({ev_raw[1]:.1f}% var)")
ax.set_title("PCA (top-2) — RAW features (no z-scoring)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "pca_raw.png", dpi=300)
fig.savefig(OUT / "pca_raw.pdf")
plt.close(fig)

# Feature-scale diagnostic
scales = X.std(0)
print(f"Raw feature std: min={scales.min():.3e}, "
      f"median={np.median(scales):.3e}, max={scales.max():.3e}, "
      f"ratio max/min={scales.max() / max(scales.min(), 1e-30):.2e}")

# ============================================================
# Summary JSON
# ============================================================
summary = {
    "n_samples": int(X.shape[0]),
    "n_features": int(X.shape[1]),
    "pca_standardized_var_explained_pc12_pct": [float(ev[0]), float(ev[1])],
    "pca_raw_var_explained_pc12_pct": [float(ev_raw[0]), float(ev_raw[1])],
    "lda_fisher_score": float(fisher),
    "top6_feature_indices": top6.tolist(),
    "top6_feature_importances": imp[top6].round(6).tolist(),
    "raw_feature_std_min": float(scales.min()),
    "raw_feature_std_median": float(np.median(scales)),
    "raw_feature_std_max": float(scales.max()),
    "raw_feature_std_ratio_max_min": float(scales.max() / max(scales.min(), 1e-30)),
}
(OUT / "identity_separability_stats.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(f"\nAll figures written to {OUT}")
