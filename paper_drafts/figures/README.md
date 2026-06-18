# Paper Figures — FabricCrypt

Generated 2026-06-01 by `scripts/build_paper_figures_20260601.py`.
All raster figures at 300 dpi; vector PDFs also provided.

## Files

### `comparison_table.md` / `comparison_table.tex`
"TPM is not enough" — comparison of FabricCrypt against 8 existing
attestation/identity primitives along 8 axes (per-die identity, vendor PKI,
special hardware, replay resistance, static-binary independence, liveness,
open-source verifiability, AI-inference binding).

- `.md` version includes per-row source citations (TCG TPM2, SGX, SEV-SNP,
  TDX, Apple platsec, NVIDIA H100 CCM, DRAM PUF Kim 2018, DRAWNAPART NDSS'22).
- `.tex` version is camera-ready with rotated headers and `\cite{}` keys.

### `identity_heatmap.png` / `.pdf` — REAL DATA
20×20 cosine-distance matrix over Phase 13 signatures
(`results/IDENTITY_BENCHMARK_2026-05-30/embodiment13/{ikaros,daedalus}_sig_v2.npz`,
10 reps × 290 features each).

Measured (from this run):
- intra-ikaros mean cosine distance:   **0.051**
- intra-daedalus mean cosine distance: **0.120**
- inter (ikaros vs daedalus):          **0.097**

**Caveat (n=2 chassis only):** ikaros is more tightly self-clustered than the
inter-cluster gap to daedalus, but daedalus's intra-cluster spread is larger
than the inter-cluster gap. This is consistent with daedalus being a noisier
substrate (see Phase 13 `phase13_report.json`); it is *not* a clean two-block
separation at the cosine-distance level on the raw 290-d signature. Phase 14C
classifier-based decisions (which use a learned classifier `classifier_E.json`,
not raw cosine) are what actually delivers the 100% honest / ~1% spoof rates
in `spoof_defense_bars.png`. The heatmap is shown as a visual prior, not as
the decision metric.

### `identity_heatmap_8chassis_mock.png` / `.pdf` — **MOCK / PROJECTION**
80×80 heatmap showing what an 8-chassis study *would* look like. Only the
ikaros and daedalus rows/cols are real; `mock_1`..`mock_6` are synthetic
rotations of the real signatures (per-chassis Gaussian shift in feature
space + per-rep noise matched to empirical intra-cluster std).

The figure carries a large red "MOCK" watermark. **Never cite this as
evidence.** It is a layout / scale projection only, to motivate the
infrastructure for the full 8-laptop study.

Seed: `np.random.default_rng(20260601)`.

### `spoof_defense_bars.png` / `.pdf` — REAL DATA
Bar chart of Phase 14C 7-condition spoof defense with 95% bootstrap CI
(2000 resamples, n=500 trials per condition).

- Green bars = "expected pass" (honest_own, static_replay_correct_nonce)
- Red bars   = "expected reject" (all spoof / wrong-nonce / replay variants)
- Dashed lines: pre-registration gates (≥95% pass, ≤5% reject, ≤10% relaxed)

`static_replay_correct_nonce` is the diagnostic known-bad: replaying telemetry
that was generated with the *current* nonce passes (as it should), confirming
nonce-binding is doing the work in all the other replay variants.

## Regeneration

```bash
cd /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
venv/bin/python scripts/build_paper_figures_20260601.py
```

Pure-matplotlib; no GPU/CPU heavy compute. Runs in <5 s on ikaros.
