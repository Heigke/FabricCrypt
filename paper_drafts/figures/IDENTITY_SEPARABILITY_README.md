# Identity-separability figures (Phase 13)

Visualizations replacing the cosine heatmap as the paper's main identity
figure. The heatmap is misleading because in raw cosine the **daedalus
intra-spread (0.120) exceeds the inter-cluster gap (0.097)**, yet a logistic
classifier on the standardized 290-D signature reaches **100% LOO**. These
figures show *why* the classifier succeeds.

## Data
- Source: `results/IDENTITY_BENCHMARK_2026-05-30/embodiment13/{ikaros,daedalus}_sig_v2.npz`
- Shape: 20 samples (10 ikaros + 10 daedalus) × 290 features
- Labels: ikaros = 0, daedalus = 1
- Generator: `build_identity_separability.py`
- Stats: `identity_separability_stats.json`

## Figures

| File | What it shows |
|---|---|
| `pca_top2.{png,pdf}` | PCA top-2 on z-scored 290-D. PC1=41.4%, PC2=10.3% var. Classes visually separate along PC1. |
| `lda_1d.{png,pdf}` | Supervised LDA → 1-D scatter + histograms. **Fisher score = 41.6** (huge: gap >> within-class variance). |
| `umap_2d.{png,pdf}` | t-SNE 2-D (perp=5, PCA-init; `umap-learn` not installed). Two tight clusters. |
| `top_features_scatter.{png,pdf}` | 2×3 grid of scatter pairs over the top-6 RF-important features (indices 272, 59, 47, 281, 32, 49). |
| `identity_separability.{png,pdf}` | **Headline figure (2×2):** PCA + LDA + t-SNE + top-feature pair. 300 DPI. Becomes paper Figure 2. |
| `pca_raw.{png,pdf}` | PCA on **un-standardized** features. PC1 captures whichever dim has largest absolute std — separability degraded vs `pca_top2`. |

## Key statistics

- **PCA (standardized):** PC1 = 41.4 %, PC2 = 10.3 % variance
- **LDA Fisher score (1-D):** **41.6** — between-class mean gap is ~41× the
  pooled within-class variance.
- **Raw feature std range:** min ≈ 0, median 1.7 × 10⁻⁵, max 5.5 × 10⁴
  → max/min ratio ≈ **5.5 × 10³⁴**. Cosine is dominated by the few high-magnitude
  dimensions and never sees the discriminant signal carried by small-magnitude
  features.

## Brief analysis (one paragraph for the paper)

> Raw cosine similarity is misleading because feature scales span more than
> thirty orders of magnitude (max/min std ≈ 5.5 × 10³⁴): a handful of
> high-magnitude features dominate the cosine, drowning the discriminative
> signal that lives in many small-scale features. After per-feature z-scoring,
> the two chassis are linearly separable in the PC1–PC2 plane (PC1 alone
> explains 41 % of variance) and the supervised LDA projection achieves a
> Fisher score of **41.6** — i.e. the between-class mean is ~6.4 σ from each
> class centroid. This explains why the 290-D logistic classifier reaches 100 %
> leave-one-out accuracy despite the deceptive raw-cosine heatmap.

## Reproduce

```
source venv/bin/activate
python paper_drafts/figures/build_identity_separability.py
```

No GPU/HW dependencies, ~10 s on CPU.
