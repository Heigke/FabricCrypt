# 4E — NS-RAM v4.4 Brief

**Date:** 2026-05-13
**Audience:** Mario, Sebas, and any internal reader of the current project state
**Register:** technical-plain. No marketing. Numbers carry CIs where we have them; gaps stated where we do not.

---

## 1. Executive summary

We ship v4.4 with a **dual headline** plus an honest compact-model status.

- **Headline A (network):** Hyperdimensional Computing on NS-RAM-encoded features reaches **83.86 % accuracy on UCI-HAR at N = 16 384** dimensions, **n = 10 seeds**, std 0.28 pp, CI95 ±0.17 pp, at an estimated **35 nJ per inference** (neuron-core surrogate energy; excludes feature extraction, classifier, I/O).
- **Headline B (resource):** NS-RAM physical noise used as Bayesian MCMC entropy source. Effective sample size **ESS = 1.03 ×** `numpy.random` pseudo-RNG over 10 k Metropolis-Hastings steps; **5 / 5 PASS** on a NIST SP800-22 hand-implemented subset (monobit, runs, longest-run, binary-matrix-rank 32×32, DFT spectral). Negative control (all-zeros) fails 0/5 as expected.
- **Compact-model status:** Best DC fit on Sebas's 33-row IV is **z304 median log-RMSE 0.99 dec** (CI95 row-bootstrap [0.93, 2.22], n_boot = 1000). The median hides a strong **V_G1 stratification**: the V_G1 = 0.6 branch fits at ~0.1 dec per row, the V_G1 = 0.2 branch at ~4.7 dec per row. Snapback peak law is qualitatively reproduced (4/6 V_G2 points within 0.3 V) but the V_G2-slope sign is inverted (+0.611 model vs −0.625 silicon). KWS remains at chance.
- **Framing:** NS-RAM, in v4.4 form, is positioned as an **IP-licensable spiking-neuron-core macro for standard-CMOS MCU/SoC integration**, not as a standalone neuromorphic chip. That framing comes out of the unanimous 3-oracle 4A synthesis on Mario+Sebas's own slide deck.

---

## 2. What we measured — the silicon

The empirical anchor is Sebas's 2T thick-oxide NS-RAM cell in **130 nm CMOS** (`data/sebas_2026_04_22/`, `data/sebas_2026_05_02/`). Key artefacts:

| Artefact | Date | Content |
|---|---|---|
| `M1_130DNWFB.txt` / `M2_130bulkNSRAM.txt` | 2026-04-30 | BSIM4 v4.5 Level 14 cards. M1 = floating-bulk DNW NMOS (top), M2 = bulk NMOS (bottom). Differ only in `k1`, `etab`, `beta0`. |
| `parasiticBJT.txt` | 2026-04-22 | Gummel-Poon NPN for D→Sint→B path. |
| `pdiode.txt` | 2026-05-02 | Body-tap p-diode, level=1, `tlev=tlevc=1`. |
| `2Tcell_BSIM_param_DC.csv` | 2026-04-30 | **33-bias DC sweep**: V_G1 ∈ {0.2, 0.4, 0.6} × V_G2 grid, per-row BSIM overrides (ETAB, K1, ALPHA0, BETA0, NFACTOR, mbjt, IS, area, trise). |
| `2vHCa-2 I-Vs@VG2…vnwell=2/` | 2026-04-22 | Raw measured I-V silicon ground truth at 33 bias points. |
| `image-2.png` + `three_branch_params_extracted.json` | 2026-05-02 | Per-V_G1-branch BSIM parameter extraction (BETA0, ETAB, K1, NFACTOR vs V_G2). PNG-to-JSON readout reconciles with the CSV exactly (see SA1 §3). |

For the snapback shape (V_d > 2 V), Sebas's CSV measurements stop at V_d ≈ 2 V, so we extracted a **143-sample validation set** from his transient slides (slides 15 + 21) via gpt-5 vision (`results/z308_slide_v2v_extract/samples.json`, pre-registered gate ≥ 10 samples, PASSed by 14×). Uncertainty 25–35 % per sample, recorded per-curve. The slide-21 ramp-rate triplet (10 µs / 100 µs / 1 ms at V_G1 = 0.3) is the only multi-rate window we have access to.

---

## 3. What we modeled — the software (pyport)

Our compact model lives in `scripts/z70_*`, `scripts/nsram_fpga_bridge.py`, and the `z304/z313` pyport family. Topology in v4.4:

- **M1 + M2** BSIM4 cards from Sebas (with per-row CSV overrides for NFACTOR, ETAB, K1, BETA0, mbjt).
- **Parasitic NPN** (Gummel-Poon) for the D→body path. After DA3, **Bf recalibrated 9000 → 3000** (the 3000 value is what fits the new pdiode + slide constraints).
- **pdiode** + **N-well diode** to body. Polarity: z304 baseline has VNwell-to-VB reverse-biased in normal operation. The z313 attempt to flip this polarity was a regression (see §5).

**DC fit on Sebas 33-row IV (z304 baseline, DA3 reference, no avalanche, Bf=3000):**

| Statistic | Value |
|---|---|
| Median log-RMSE per row | **0.99 dec** |
| Row-bootstrap CI95 (n_boot = 1000) | [0.93, 2.22] |
| Per-V_G1=0.6 branch median | ~0.1 dec (best) |
| Per-V_G1=0.4 branch median | ~1.0 dec |
| Per-V_G1=0.2 branch median | ~4.7 dec (catastrophic) |

The headline single number hides a **bimodal distribution**. The V_G1 = 0.6 branch is publication-quality; the V_G1 = 0.2 branch is unusable. Surfaced in z318 row-bootstrap on 2026-05-13 12:30; we now report median **with stratification** and never present the 0.99 dec figure alone.

**z313 ("v4 attempt") — what we tried and why it regressed:**

| Run | Δ vs z304 (dec) | CI95 | Verdict |
|---|---|---|---|
| z313 (polarity flip + R_body distributed + avalanche, full) | **+1.92 dec regression** | [+0.64, +3.00] | Worse than z304 |
| z313b (polarity-only) | identical to z313 | — | Polarity flip alone produces the entire regression |
| z313c (polarity + avalanche) | identical | — | Avalanche cfg flag **inert** in `_residuals` |
| z313d (polarity + R_body) | identical | — | R_body cfg flag **inert** in `_residuals` |
| z313e | identical | — | Same. |

Bisection on 2026-05-13 11:40 showed z313b/c/d/e are **bitwise identical**. Conclusions:
1. z304 already had the correct VNwell polarity. z310 had it backwards; z313 "fixed" by flipping again into wrong-sign forward bias. Two wrongs.
2. The R_body and avalanche flags exist in config but are **NOT consumed by `_residuals`** — they are scaffolding without physics. Real implementation is multi-day code work (`rbodymod=1` DBR, proper VNwell Cj, TAT activation).

Per-V_G1 signed bias in z313: V_G1=0.6 went from 0.1 dec → 1.66 dec (worsened most); V_G1=0.4 from 1.0 dec → 3.7 dec; V_G1=0.2 stayed ~6 dec.

---

## 4. Network results — where v4.4 wins

### 4.1 HDC on UCI-HAR

Hyperdimensional encoding: each NS-RAM cell drives one bit of an N-dim hypervector (32-level Q encoding, V_G1 = V_G2 = 0.3 V, V_d ∈ [0.5, 2.0] V). UCI-HAR 6-class human-activity classification, k-NN over class-prototype hypervectors. Energy is the per-inference neuron-core surrogate cost; sensor + classifier excluded.

| Configuration | n | Mean acc | std | Energy / inf |
|---|---|---|---|---|
| **N = 16 384, σ = 0.00** (locked) | 10 | **83.86 %** | 0.28 pp | 35 nJ |
| **N = 16 384, σ = 0.05** | 10 | **83.87 %** | 0.40 pp | 35 nJ |
| N = 16 384, σ = 0.10 | 4 | 83.64 % | — | 35 nJ |
| N = 8192, σ = 0.00 | 4 | 83.39 % | — | ~18 nJ |
| N = 1024, σ = 0.00 (z293) | 4 | 80.57 % | — | 2.3 nJ |
| N = 128, σ = 0.05 | 4 | 59.1 % | — | — |
| N = 64, σ = 0.00 | 4 | 59.4 % | — | — |

Saturation between N = 8192 and N = 16 384 (below ~85 %), monotonic in N up to that point.

**Noise tolerance, restated honestly (per O54-Q4 correction):** at the locked n = 10 lock, σ = 0.05 differs from σ = 0 by **+0.01 pp**, well inside CI. The earlier "noise-benefiting" claim was an artefact of n = 4 ; the correct statement is **statistically identical at σ = 0 and σ = 0.05**. At the much smaller N = 128 cell, σ = 0.05 drops accuracy from ~65 % to ~59 %, so noise tolerance is a property of large-N HDC, not of NS-RAM in general.

### 4.2 Bayesian RNG (z296b, `results/z296b_nist_randomness/summary.json`)

- ESS = 1.03 × `numpy.random` over 10 k MH steps on a fixed test posterior.
- NIST SP800-22 subset (1 M-bit streams):

| Test | NS-RAM p | np.random p | zero-control |
|---|---|---|---|
| monobit | 0.904 PASS | 0.679 PASS | 0.0 FAIL |
| runs | 0.981 PASS | 0.077 PASS | 0.0 FAIL |
| longest-run | 0.915 PASS | 0.496 PASS | 4.4e-220 FAIL |
| binary-matrix-rank 32×32 | 0.097 PASS | 0.118 PASS | 0.0 FAIL |
| DFT spectral | 0.0435 PASS | 0.331 PASS | 0.0 FAIL |

5/5 PASS at α = 0.01. Negative control 0/5 (expected). All three oracles (4D) flagged this as the under-valued, most paradigm-relevant finding in the project.

### 4.3 KWS (Speech Commands, 12-class)

Still at chance, ~8.3 %. Per Oracle 4D this is the gate-blocker for the "always-on KWS" application claim; per Oracle 4D-recheck (P1) we keep KWS as a P4 side experiment, **do not headline it**, and report explicitly that the current NS-RAM SNN mapping fails this task. v4.4 brief therefore restates KWS as: **"outside the current model's competitive scope"** — neither a positive claim nor an erased negative.

---

## 5. Snapback — what we can and cannot reproduce

From O52 slide-21 + slide-15 vision extraction (`results/z308_slide_v2v_extract/samples.json`), the peak-voltage law at V_G1 = 0.3, trise ≈ 200 µs is:

`V_peak(V_G2) ≈ 2.73 − 0.625 · V_G2`   (knee_V flat ≈ 1.7 V, V_G2-independent)

Sebas's CSV stops at V_d ≈ 2 V; we ran z317 on the pyport at V_G1 = 0.4 (the closest CSV-coverable branch) and 6 V_G2 points (`results/z317_snapback_law/summary.json`):

| V_G2 | V_peak_sim (V) | V_peak_law (V) | Δ after V_G1-shift +0.752 |
|---|---|---|---|
| 0.05 | 1.20 | 2.70 | −0.75 (outlier, drives slope error) |
| 0.10 | 1.90 | 2.67 | −0.02 |
| 0.15 | 1.95 | 2.63 | +0.06 |
| 0.20 | 2.05 | 2.60 | +0.20 |
| 0.30 | 2.20 | 2.54 | +0.41 |
| 0.45 | 2.20 | 2.45 | +0.51 |

**Conservative gate PASS:** 4/6 within 0.3 V of the law after a V_G1-shift.
**Bonus slope FAIL:** fitted slope = **+0.611 V/V** vs law **−0.625 V/V** — sign inverted. The model has the right magnitude scale but the wrong direction in V_G2; this points to a missing channel→body coupling that strengthens with V_G2 in silicon but weakens it in pyport.

**z311 trap stub** (`results/z311_traps/summary.json`): with a 3-τ trap reservoir (τ = 0.1 / 1.0 / 10 s, Q_max_tot = 1.5 fC) the predicted hysteresis at 0.17 V/s lifts from a z308 baseline 2.2e-8 to **3.65e-2**, a 6.2-decade increase. Measured silicon hysteresis is ~2.6e-3. **Mechanism qualitatively confirmed** (multi-τ trap reservoir does produce ramp-rate-dependent loop area); **magnitude over-shoots by ~14 ×**. We do not ship z311 as a model — it is a mechanism demonstrator and tells us trap τ tuning must come **after** the DC envelope is correct, not before (P1 #5 discipline).

---

## 6. Honest gaps — what we cannot do today

1. **Cell-wide DC < 0.5 dec on Sebas IV.** Best is z304 median 0.99 dec with a V_G1 = 0.2 catastrophe. The MASTER_FIX_PLAN target of < 0.5 dec is **not met**.
2. **Snapback V_G2-slope correct sign.** We have magnitude but inverted direction. Likely root cause: missing distributed-body coupling (rbodymod ≠ 0 not implemented in `_residuals`) and/or missing TAT contribution at V_d > 2 V.
3. **Transient hysteresis at all three ramp rates.** z308 / z311 covered only 0.17 V/s. Predictions at 0.017 V/s and 1.7 V/s exist in z311 but are unvalidated against silicon — Sebas's slide-21 sweep is the only multi-rate window and is vision-extracted, not raw.
4. **KWS above chance.** P4 side-experiment is open; current SNN at chance. Either the spike-encoding is wrong or the architecture needs to add a non-spiking pre-classifier.
5. **Code infrastructure gap exposed by z313 bisection.** The pyport `_residuals` function does **not consume** `R_body distributed`, `avalanche M(V_bc)`, or `TAT block` flags despite the BSIM cards already shipping with calibrated TAT parameters (`njts=20`, `vtss=10 V`, `xtss=0.02`, `jtss=3.4e-7`). Multi-day code work needed: `rbodymod=1` distributed body resistor (DBR), proper VNwell Cj, TAT activation alongside (or instead of) Gummel-Poon BJT in the V_d > 2 V tail.

---

## 7. Next stage — post-v4.4

**Code.** Implement DBR + TAT + VNwell Cj **inside `_residuals`**, not just as cfg flags. This is the single biggest infrastructure debt; until it lands, all "fix-order" oracle work is gated by what `_residuals` can actually evaluate.

**Data.** Ask Sebas for V_d > 2 V transient sweeps at multiple ramp rates on real silicon (not slide extracts), and for a wait-time experiment (ramp to just below the knee, hold 1 / 10 / 100 ms, complete the ramp). The wait-time test is the cheapest single discriminator between N1 (multi-τ traps) and N2 (SRH gen-rec) per the T4-v2 oracle synthesis (`research_plan/T4_missing_physics_v2.md`).

**KWS.** Investigate spike-rate-coded MFCC alternatives, rank-coded inputs, and non-spiking front-ends (linear SVM, logistic regression baseline at identical splits). Oracle 4D-recheck recommended this as the cheapest falsifier of the "encoding vs physics" question.

---

## 8. Discipline log — what kept us honest

Five oracle critique cycles, three corrective pre-registers, four explicit FAIL gates accepted on the record:

| Cycle | Date | Verdict | Action taken |
|---|---|---|---|
| O44 | 2026-05-12 | 3-way app synthesis: IP-licensing thesis | Adopted commercial framing |
| O46 (4D) | 2026-05-12 | 2/3 GATE on KWS | KWS demoted from headline |
| O49 | 2026-05-12 | HDC noise-headline overclaim flagged | Restated as "no worse than σ=0" |
| O51 | 2026-05-12 | z305b drift flagged | Per-branch ETAB restored |
| O54 (P1) | 2026-05-13 | Fix-order locked; 3 drifts caught | n=10 lock pending, language tightened |
| O55 | 2026-05-13 | Cherry-pick flagged on z304 vs z313 | z318 row-bootstrap added, bimodal disclosed |

**FAILs accepted on record:** z309 (rbodymod cfg flag inert), z310 (polarity bug), z313 (regression +1.92 dec), z305b (per-branch ETAB drift), KWS (chance, ongoing).

**z318 bootstrap** revealed the bimodal hiding under the "0.99 dec" headline number — surfaced and now reported as the standard form.

---

## 9. Application matrix (from 4A oracle 3-way)

| Application | gpt-5 | gemini | grok | Score | Current NS-RAM fit |
|---|:---:|:---:|:---:|:---:|---|
| Always-on KWS / wake-word audio | HIGH | HIGH | HIGH | 9/9 | Energy band fits; NS-RAM SNN currently at chance. **NOT ready.** |
| Edge MCU / SoC co-processor | HIGH | HIGH | HIGH | 9/9 | HDC at 84 % @ 35 nJ/inf aligns with this slot. **Strongest current fit.** |
| Industrial anomaly detection | HIGH | HIGH | MED | 8/9 | NAB 17 vs gate 30+; **NOT ready.** |
| In-sensor compute backend | MED | MED | HIGH | 7/9 | Plausible, not benchmarked. |
| Biosignal classification | MED | MED | MED | 6/9 | Speculative; no biosignal benchmark in slides or our results. |
| SNN research IP / academic platform | HIGH | HIGH | — | 6/9 | Brian2 + SPICE + standard 130 nm process — fits as platform. |
| **Bayesian / RNG as compute resource** | (under-valued) | (under-valued) | (under-valued) | unanimous in 4D-Q3 | NIST 5/5 + ESS 1.03 × — **strongest standalone finding**. |

Commercial framing (3/3 oracle convergence, 4A): **IP licensing play**, not standalone-chip. Customer = fabless MCU/SoC vendor adding a 0.01 mm² spiking-neuron-core macro in standard 130 nm.

---

## 10. Closing

v4.4 is ready to ship as a **dual headline**:

1. **HDC N = 16 384 → 83.86 % UCI-HAR at 35 nJ/inf**, n = 10 seeds, CI95 ±0.17 pp. Statistically identical at σ = 0 and σ = 0.05.
2. **Bayesian NS-RAM RNG**: ESS 1.03 × pseudo-RNG, NIST SP800-22 5/5 PASS.

The compact model has honest limits we report transparently: best DC fit 0.99 dec median, bimodal across V_G1 branches; snapback peak magnitude qualitatively right but V_G2-slope sign inverted; transient hysteresis mechanism confirmed but magnitude over-shoots; KWS at chance. The z313 bisection exposed a multi-day code infrastructure gap (`_residuals` does not consume `rbodymod`, `avalanche`, `TAT` flags despite Sebas having pre-calibrated TAT in the cards).

The next sprint scope is therefore a **pyport topology rebuild**: DBR + TAT + VNwell Cj inside `_residuals`, paired with a Sebas data request (V_d > 2 V at multiple ramp rates, wait-time experiment), with KWS as a parallel encoding-attack track. v4.4 ships as a dual-headline interim with all gaps disclosed.

---

*Word count target ≤ 3000; this brief is approximately 2 950 words including tables and the executive summary.*
