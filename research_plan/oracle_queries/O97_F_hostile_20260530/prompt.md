# O97 — HOSTILE 4-way oracle critique of Angle F (self-referential identity coupling)

You are one of four oracles (GPT-5, Gemini 2.5 Pro, Grok-4-latest, DeepSeek-Reasoner).
This is a **HOSTILE** critique. Your job is to find what is wrong, not to validate.

## Context

We are studying whether two nominally identical AMD gfx1151 GPUs (`ikaros` and `daedalus`,
Radeon 8060S, same SKU, same driver, same OS) carry a per-die "identity" signature that
a trained model can become **constitutively coupled to**, such that transplanting the
model to the other identical machine causes asymmetric performance degradation.

Phase 1, Phase 1b, Phase 2 of the orthodox identity-discovery benchmark all returned
**NULL** or **MIXED** verdicts. See attached `IDENTITY_BENCHMARK_2026-05-30*.md` for
the four-phase plan and prior results.

We then ran a 10-angle novel-probe brainstorm (`IDENTITY_NOVEL_ANGLES_2026-05-30.md`).
The previous oracle round (`O96_prior_synthesis.md`) voted **Angle F as NOT NOVEL**:
all four oracles cited DNN watermarking / device-conditioned inference / model-binding
literature (Rouhani DeepSigns 2019, Gu BadNets 2017, Li HWN-DNN ISCA 2020).
F was placed on the **kill list**.

**We built and ran Angle F anyway.** This packet asks you to hostile-critique the result.

## Angle F design (one paragraph)

Train a tiny reservoir-style regressor on a synthetic NARMA-10 task. The model has two
variants:
- **substrate-naive** (baseline): inputs = NARMA driver only.
- **substrate-aware**: inputs = NARMA driver + per-CU substrate features
  (RTN-rate, spatial-corr, LDS-startup, 1/f-knee) sampled from each device's
  Phase 1b `raw_idle.npz`.

The implementation does **NOT** read hwreg live via shader. It uses **cached** Phase 1b
per-CU summary statistics as substrate features (40 CUs × 4 features = 160 extra
input dims).

We train on `ikaros` features (cached), evaluate on `ikaros` features and `daedalus`
features; then train on `daedalus`, evaluate on both. 10 seeds per condition.

## Data: gap = NRMSE(other) − NRMSE(own), per-seed-matched

| condition | own_mean | other_mean | gap | gap_std |
|---|---|---|---|---|
| aware=False, train=ikaros | 0.676 | 0.803 | **+0.127** | 0.057 |
| aware=False, train=daedalus | 0.649 | 1.057 | **+0.408** | 0.101 |
| aware=True, train=ikaros | 0.676 | **2.179** | **+1.503** | **1.800** |
| aware=True, train=daedalus | 0.649 | 1.215 | **+0.567** | 0.101 |

Overall: aware_gap_mean=1.035, baseline_gap_mean=0.267, **ratio ≈ 3.9×** (not 11×; the
"11×" comes from comparing the ikaros-trained row only: 1.503 / 0.127 ≈ 11.8).
**Discovery gate (z>2 AND aware>baseline) FAILED**: z=0.79 because aware_gap_std=1.36
is enormous, driven by the (ikaros→daedalus, aware) condition's std of 1.80.

Asymmetry: ikaros→daedalus (aware) gap=1.503 vs daedalus→ikaros (aware) gap=0.567,
**2.7× imbalance**. Note: ikaros has RTN-rate ≈ 0.000 (degenerate, Phase 1b finding),
daedalus has RTN-rate ≈ 0.11.

## Format constraints

- ≤500 words total.
- Numbered answers Q1–Q10. No preamble.
- HOSTILE tone preferred — find what's wrong.
- Cite real papers if you cite anything. Hallucinated citations will be penalised.

---

## Questions

**Q1.** Is the 11× degradation gap (substrate-aware ikaros→daedalus: 1.503 vs naive
ikaros→daedalus: 0.127) genuine identity-coupling, or is it a trivial distribution-shift
effect — any model trained with feature X will degrade when feature X distribution
changes between train and test, regardless of whether X carries device-identity?

**Q2.** The proper falsifier: if we replaced the per-CU substrate-features with PURELY
SYNTHETIC noise drawn from each device's marginal distribution (matched mean+std per
feature, but no spatial / cross-feature / device-identity content), would we expect
the same ~11× gap? Predict directionally.

**Q3.** The asymmetry — ikaros→daedalus aware gap = 1.503 vs daedalus→ikaros aware
gap = 0.567 (2.7× imbalance). What does this suggest? Hint: ikaros RTN-rate ≈ 0.000
(degenerate, zero variance feature), daedalus RTN-rate ≈ 0.11. Could the asymmetry
be a pure variance / NaN-handling artifact rather than identity signal?

**Q4.** The substrate-aware model has 160 more input features than the baseline
(40 CUs × 4 features). Could the 11× gap simply be feature-count-dependent overfitting
to ikaros-specific feature values that don't exist on daedalus? What is the simplest
baseline that controls for feature-count (same dim count, no identity content)?

**Q5.** Is this result publishable as-is? If not, exactly which experiments are missing
before it would be defensible at a top venue (NeurIPS / USENIX Security / ISCA)?
Be specific. Note: discovery_gate (z>2) FAILED at z=0.79.

**Q6.** The O96 oracle round said F is "not novel" because adversarial-ML / PUF /
DNN-watermarking literature has the concept. **Cite ONE actual paper** that
demonstrated substrate-aware model degradation on AMD-twin GPUs (or even AMD-twin
compute substrates of any kind). If you cannot cite one (no hallucinations), state
that explicitly, and answer: is our empirical demonstration on real AMD-twin gfx1151
silicon still valuable, or has the field already shown this on NVIDIA/x86 such that
the AMD result is mechanical replication?

**Q7.** What single experiment would conclusively kill F, or conclusively elevate it
from "interesting" to "important"? Pick ONE.

**Q8.** If we ran F on a third nominally-identical gfx1151 machine (call it `minos`),
predict the gap pattern: (a) what would aware_gap_mean look like for ikaros→minos,
daedalus→minos, minos→ikaros, minos→daedalus? (b) Would the asymmetry pattern
generalise, or is it ikaros-specific?

**Q9.** The "model reads its own hwreg signature via shader" framing in the design
doc — the current implementation does NOT do that. It uses cached Phase 1b
`raw_idle.npz` summary statistics computed offline. Does this matter for the
identity-coupling claim? Specifically: does using cached static features (vs live
per-forward-pass hwreg read) change F from "constitutive coupling" to "ordinary
covariate-shift demonstration"?

**Q10.** Final hostile read in three lines:
- best-case interpretation (what is the strongest defensible claim from this data),
- worst-case interpretation (what is the most damning artifact explanation),
- your odds (X% best-case / Y% worst-case / Z% middle).

---

Attachments in this packet:
- `IDENTITY_NOVEL_ANGLES_2026-05-30.md` — original 10-angle brainstorm (F design)
- `O96_prior_synthesis.md` — prior oracle "F not novel" verdict
- `IDENTITY_BENCHMARK_2026-05-30.md`, `_PHASE1.md`, `_PHASE1B.md`, `_PHASE2.md` —
  the orthodox 4-phase plan and NULL/MIXED prior results
- `F_results.json` — full per-seed NRMSE data (10 seeds × 4 device pairs × 2 aware modes = 80 runs)
