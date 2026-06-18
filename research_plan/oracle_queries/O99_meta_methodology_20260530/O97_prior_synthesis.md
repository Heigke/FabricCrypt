# O97 Synthesis — Hostile 4-way critique of Angle F

Date: 2026-05-30
Oracles: GPT-5, Gemini 2.5 Pro, Grok-4-latest, DeepSeek-Reasoner — **4/4 responded**
Tone: hostile-as-requested. **All four converged on "artifact, not discovery."**

## Per-question consensus

| Q | Topic | GPT-5 | Gemini | Grok | DeepSeek | Consensus |
|---|---|---|---|---|---|---|
| Q1 | Genuine identity vs trivial covariate shift | Trivial | Trivial | Trivial | Trivial | **UNANIMOUS: covariate shift** |
| Q2 | Synthetic noise falsifier reproduces gap? | Yes (dir.) | Yes | Yes | Yes | **UNANIMOUS: yes** |
| Q3 | 2.7× asymmetry = variance artifact? | Yes (RTN≈0 + scaling) | Yes (degenerate RTN) | Yes (zero-var col) | Yes (degenerate RTN) | **UNANIMOUS: variance artifact, not identity** |
| Q4 | Feature-count overfitting? Control? | Yes; 160 iid Gaussian | Yes; 160 iid Gaussian | Yes; 160 matched noise | Yes; 160 iid Gaussian | **UNANIMOUS: yes; dim-matched random control mandatory** |
| Q5 | Publishable? | No (8 missing) | No (gate fail = autoreject) | No (z=0.79 fatal) | No | **UNANIMOUS: NOT publishable** |
| Q6 | Cite real paper on AMD-twin substrate-aware degradation | None exists | None (only Li ISCA'20 on FPGA) | None exists | None (only NVIDIA / synthetic prior) | **UNANIMOUS: no such citation; AMD-twin replication is novel-as-substrate but unpublishable without controls** |
| Q7 | Single killer/elevator experiment | Intra-device recapture (train ikaros-A → test ikaros-B) | Synthetic-noise falsifier (Q2) | Synthetic noise w/ daedalus marginals + degenerate col | RTN-only swap (replace daedalus RTN with ikaros constant during eval) | **Cluster on falsifier-style controls; intra-device-recapture is most decisive** |
| Q8 | Third-machine (minos) prediction | Large noisy + more symmetric; ikaros pathology stays ikaros-specific | gap ∝ KL divergence of marginals; asymmetry ikaros-only | Asymmetry won't generalise | Moderate gaps for non-degenerate pairs; ikaros stays anomalous | **UNANIMOUS: asymmetry is ikaros-specific (RTN=0), won't generalise** |
| Q9 | Cached static features vs live hwreg? | Matters — cached = ordinary cov-shift | Matters — invalidates constitutive claim | Matters — measures nothing beyond shift | Matters — feature-mapping not identity | **UNANIMOUS: cached implementation invalidates "constitutive coupling" framing** |
| Q10 | Odds best / worst / middle | 15 / 65 / 20 | 5 / 95 / 0 | 8 / 75 / 17 | 8 / 72 / 20 | **Mean: ~9% best / ~77% worst / ~14% middle** |

## Sharpest disagreement

There is almost none. The four diverge only on:
- **Magnitude of best-case odds**: Gemini 5% (most damning), GPT-5 15% (most charitable).
- **Which single killer experiment**: GPT-5 picks *intra-device recapture* (test if a snapshot of the same device, recaptured, also causes a "gap" — best at isolating "tag-vs-identity"). Gemini/Grok pick the *synthetic-marginal falsifier* (cheaper, equally decisive on the distribution-shift hypothesis). DeepSeek picks the *RTN-only swap* (most surgical re the degenerate-feature hypothesis). All three are good. GPT-5's is the strongest because it falsifies BOTH "identity carries information" AND "the feature is a stable identity tag at all."

No oracle defended F. No oracle argued the result is genuine identity-coupling.

## Updated verdict on Angle F

**DOWNGRADED: DISCOVERY → ARTIFACT-PENDING-CONTROLS.**

The previously claimed "11× gap, DISCOVERY-grade" framing does not survive even superficial hostile review. The benchmark's own discovery gate (z > 2 AND aware_gap > baseline_gap) **already failed** at z = 0.79 — this was buried in the result file but not in the user's narrative summary. The four oracles independently surfaced this. Root cause identified by all four: **ikaros RTN-rate is degenerate (≈ 0, zero variance)**, and the four outlier seeds in the (ikaros→daedalus, aware) row (NRMSE 1.84, 2.77, 4.94, 6.19) drive both the mean gap and the asymmetry. Remove that one feature column, and the effect almost certainly collapses.

The Phase-1b finding "ikaros RTN ≈ 0, daedalus RTN ≈ 0.11" is itself a real device difference — but feeding it as a static input feature does not test "identity coupling," it tests covariate shift on a degenerate column. Different claim, much weaker.

**F is not killed yet** — there is a 9% (best) / 14% (middle) chance that controls preserve a non-trivial residual after the variance artifact is removed. But the current packet is not publishable and the "DISCOVERY" claim should be retracted from the live narrative.

## Top 3 experiments to run next (oracle-consensus order)

### 1. Synthetic-marginal-noise falsifier (3/4 explicitly recommend; UNANIMOUS in Q2 directional prediction)
Replace the 160 substrate features with iid samples drawn from each device's per-feature marginal (mean, std), preserving feature count and per-device marginals but destroying all spatial / cross-feature / temporal identity structure. If the ~11× gap survives, F is dead as identity. If gap vanishes, the structural content of the real features is doing real work — promote F back to "interesting." Cheapest test, decisive on the dominant hypothesis.

### 2. Drop-degenerate-feature + dim-matched random baseline (4/4 demand a feature-count control in Q4)
Two-part: (a) ablate each of the 4 feature channels (RTN-rate, spatial-corr, LDS-startup, 1/f-knee) individually; prediction (DeepSeek explicit): the gap collapses when RTN is dropped or when daedalus RTN is replaced with ikaros's constant zero. (b) Add 160 iid N(0,1) features to the *baseline* model — if it also degrades cross-device, the gap is feature-count overfitting, not identity. Without this control, F cannot be defended at any venue.

### 3. Intra-device recapture (GPT-5's pick; cleanest "tag vs. identity" discriminator)
Re-measure ikaros's substrate features twice at matched temperature (capture A and capture B, separated by hours / thermal cycle / reboot). Train on capture A, test on capture B. If the "aware" model degrades within the same device, F is testing snapshot-recency, not identity. If intra-device gap is small while inter-device gap survives the controls in (1) and (2), THEN F upgrades to a real finding worth replicating on minos as the third twin.

## Additional unanimous demands (must-do, not optional)

- **Replicate on minos** (third twin) — 4/4 explicitly require this for publication. All four predict the asymmetry will NOT generalise (it is ikaros-specific from RTN=0).
- **Implement live per-forward-pass hwreg reads** if the "constitutive coupling" framing is to be retained. 4/4 say the cached-feature implementation reduces F to ordinary covariate shift.
- **Pass the discovery gate (z > 2)** with robust statistics. Currently z = 0.79, driven by 4 outlier seeds (out of 10) in a single condition. Median/trimmed-mean reporting; outlier diagnostics.
- **No citation exists** for AMD-twin substrate-aware-model degradation. 4/4 confirm. This means: if controls hold, the AMD-twin demonstration IS novel-as-substrate and worth a workshop paper. But "concept novelty" is gone (DeepSigns / HWN-DNN / BadNets already own the concept-space).

## Files

- Prompt: `prompt.md`
- Attachments: `IDENTITY_NOVEL_ANGLES_2026-05-30.md`, `IDENTITY_BENCHMARK_2026-05-30.md`, `..._PHASE1.md`, `..._PHASE1B.md`, `..._PHASE2.md`, `O96_prior_synthesis.md`, `F_results.json`
- Responses: `openai_response.md`, `gemini_response.md`, `grok_response.md`, `deepseek_response.md`
