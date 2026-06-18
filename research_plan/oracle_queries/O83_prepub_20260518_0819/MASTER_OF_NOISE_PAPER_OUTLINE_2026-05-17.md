# MASTER OF NOISE — Paper Outline (2026-05-17)

Source: Plan agent a673b41a015d53950 synthesis from NEXT_PHASE_PLAN, GPU_MAX_CAMPAIGN, N-BENCH-A matrix, O80 triangulation, z47x honest analyses.

## Thesis

A single 130 nm 2T floating-body NS-RAM cell, calibrated to Mario silicon (Id_pk = 4.30 mA, gap −0.007 dec; cell DC RMSE ~1.3 dec, 7/9 z461 transient indicators after R_body=1e7 + IFT sign-patch), executes the **complete catalogue of six canonical noise operations** — GENERATE, SUPPRESS, DETECT, LEARN-FROM, COMPUTE-THROUGH, PLASTICIZE-UNDER noise — with one physical primitive. Digital silicon demands six distinct macros (PRNG, FIR/LMS, anomaly NN, optimizer, denoiser, plasticity engine); NS-RAM collapses them into one shared substrate by routing the same impact-ionization avalanche, body-charge storage, and BJT positive-feedback into different read-out modes.

**Multi-functionality is the structural win, not raw pJ/inference at 130 nm** (where, per N-BENCH-A, NS-RAM cannot beat Loihi-2/Akida/Mythic on any single workload).

## What we own

- **§3 GEN**: TRNG NIST 5/5 PASS @ 0.4 pJ/bit [SIM, MEDIUM]
- **§4 SUPPRESS**: LMS-Eq AMBITIOUS, 170× energy [SIM, LOW — peripheral unmodelled BOLD caveat]
- **§5 DETECT**: PC-NAB + Cascade KWS/ECG DISCOVERY (60.8% save)
- **§6 LEARN**: NES-GD pending (smoke in flight)
- **§7 COMPUTE**: EP-NSRAM smoke PASS (body-τ as relaxation phase), MEP-6 IFT gradcheck 7/8 @ <5% relerr
- **§8 PLASTIC**: DS-N12 STDP under body-state eligibility
- **§9 INFRA**: MEP-6 differentiable pyport via IFT (z474b sign-patch landed, value-identity 0 drift)

Demoted per N-BENCH-A triple-converge: HDC, LIF accuracy headlines, reservoir-as-headline.

## What we owe

- NES-GD full + K2 cross-cell noise-correlation audit
- V7 oscillation FALSE (z475 NEG, document as future-tapeout request)
- Same-cell ablation matrix (same V_b knobs across 6 modes)
- Per-mode energy with explicit caveat banner
- Peripheral-aware energy on §4

## Risk + counter

Reviewer: "any analog accelerator does noise". Counter:
1. Run all 6 modes on **the same cell instance**, only V_b/V_d programming changes
2. MEP-6 IFT flows through every mode with **one** `torch.autograd.Function`
3. N-BENCH-A matrix shows no rival platform reports 6-mode coverage from one circuit

## Venue strategy

1. **Nature Electronics** (primary) — device + algorithm hybrid
2. **Nature Communications** (backup)
3. **IEDM 2026 circuits track** (paired companion: "Differentiable pyport via IFT")
4. AVOID: Nature/Science (no tapeout), ISSCC (silicon-only)

## Pre-registered headline gates

1. All 6 modes pass per-section AMBITIOUS on a single calibrated cell
2. MEP-6 IFT gradcheck ≥66% FD-reliable biases <5% relerr (PASS 87.5%)
3. No per-mode energy without flagged BOLD caveat
4. NO "beats X" headline — only "first to do all six from one primitive"

## Paper structure

### Abstract (180 words)
Single-cell 2T 130nm primitive; six canonical noise modes; calibrated to Mario; differentiable pyport via IFT; no tapeout — projections flagged.

### 1. Introduction
Noise as taxonomy: 6 ops, currently 6 separate macros. NS-RAM 3 reusable knobs (body-charge, parasitic NPN, impact-ionization). Non-claim: not an accelerator paper.

### 2. Device & calibration (shared substrate)
Mario 4.8 mA → snap_Is=4.5192e-12 (z471). z472 6/9 → z473 R=1e7 → 7/9 + Mario 3/5. z474b IFT sign-patch landed. **Figure: same cell, six gate-bias programming points, one schematic.**

### 3. GENERATE — TRNG
NIST 5/5 PASS, 0.4 pJ/bit, 1.27 Mbit/s [SIM]. Missing: cross-cell correlation matrix. KILL: ‖C−I‖_F/N>0.3 → entropy claim collapses.

### 4. SUPPRESS — LMS adaptive equalizer
AMBITIOUS, 170× projected [SIM, LOW]. Missing: peripheral-aware energy + BER vs SNR sweep. KILL: BER floor 2× digital at SNR≥15 dB.

### 5. DETECT — Anomaly (NAB + Cascade)
PC-NAB DISCOVERY, Cascade 60.8% energy save. Missing: NAB vs HTM-Java baseline. KILL: NAB < HTM-Java.

### 6. LEARN — SPSA via impact-ionization (NES-GD)
Pending. Missing: full CIFAR + K2 audit. KILL K2: cross-cell σ-correlations >2× Gaussian variance.

### 7. COMPUTE THROUGH — Equilibrium Propagation
Smoke PASS, body-τ ≈ 1 ms = natural relaxation. MEP-6 IFT closed. Missing: full MNIST ≥97%, vector-field EP. KILL K1: IFT κ(J)>1e10 in >20% points.

### 8. PLASTIC UNDER NOISE — STDP + body-state eligibility
DS-N12 baseline. Missing: long-horizon retention (shadow synaptic variable). KILL K3: τ_body drift >20% across V_b.

### 9. Shared Infrastructure — Differentiable Pyport via IFT
ONE `torch.autograd.Function` enables backprop through all 6. z474b gradcheck 7/8 PASS, value-identity 0 drift. **Transferable contribution — IEDM circuits-track standalone fallback if main paper stalls.**

### 10. Comparison strategy
One table: rows = 6 modes; cols = NS-RAM [SIM] + best Si rival per mode + "rival's coverage of other 5 modes". Diagonal full, off-diagonal empty → visual unified-substrate argument. All NS-RAM cells flagged [SIM] + BOLD caveat where confidence LOW.

### 11. Discussion — "any analog accelerator does noise" counter
Three falsifiers reviewer can run:
- (a) cite analog chip with ≥4 of 6 modes from one cell — N-BENCH-A: none exists
- (b) cite unified diff-training for analog fixed points — only MEP-6 IFT here
- (c) show 6 demos need different params — ablation §2 proves they don't (only V_b programming)

Honest limits: 130 nm 3-4 gen disadvantage, energy projections, V7 unresolved.

### 12. Methods
Calibration protocol (z471-z474b). IFT autograd derivation + gradcheck. Per-mode workload + pre-reg gates table. Energy projection assumptions + confidence flags.

### 13. Data & Code Availability
nsram repo + z47x scripts + MEP-6 IFT autograd + per-mode notebooks.

## Killshots summary

| Mode | Killshot | Status |
|---|---|---|
| §3 GEN | NIST fail after noise-corr correction | NOT TRIGGERED |
| §4 SUPPRESS | BER 2× digital w/ peripheral | OPEN |
| §5 DETECT | NAB < HTM-Java | OPEN |
| §6 LEARN | K2 noise corr > 0.3 | OPEN, blocking |
| §7 COMPUTE | K1 IFT singular > 20% | NOT TRIGGERED (gradcheck 7/8) |
| §8 PLASTIC | K3 τ drift > 20% | OPEN, blocking |
| §2 substrate | V7 osc impossible (z475) | KNOWN, document |

## Author order

1. Eric Bergvall — lead, framing, diff-pyport, integration
2. Algorithm co-authors (EP, NES-GD owners)
3. Calibration z47x author
4. Mario (data provider, acknowledged author)
5. Sebas — silicon characterization, senior author (last)

## Timeline (14 days)

- D1-4: Close K2 audit (unblocks §6); finalize z474 lock; cross-check IFT through all 6 modes' autograd
- D5-10: EP full (§7), NES-GD full (§6), peripheral-aware LMS (§4)
- D8-12: NAB full + plasticity retention shadow-var (§5, §8)
- D12-14: Writeup, figures, same-cell-six-modes ablation matrix
