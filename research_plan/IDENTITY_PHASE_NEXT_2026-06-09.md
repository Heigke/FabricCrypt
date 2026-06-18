# Identity Phase Next — Cryptographic hardware-rooted AI substrate binding

Date: 2026-06-09. Author: research agent (Claude), reviewed/OK'd by Eric.
Supersedes: the consciousness/Butlin framing in EMBODIMENT7 and IDENTITY_CAPABILITY_LANDSCAPE.

> **2026-06-09 correction (Eric):** the "excise consciousness framing" line below
> was wrong — that was an oracle-pulled pivot toward publishable
> anti-piracy and replaced the actual research goal. The actual mission
> is restored in **`H7_PREREG_2026-06-09.md`** (now the headline arm):
> find the analog physical leak-channels that constitute identity in a
> commodity die, and bind a model to them constitutively. Crypto (H1) is
> bookkeeping on top, not the goal. Mocks are banned; all signals are real
> reads. H7 + H3 + H5 are the positive arms; H1 stays paused until we
> have real TPM EK on ≥1 host (we do now — provisioned 2026-06-09).
> H4/H6 still hold as null/equivalence deliverables.

## Mission (in one paragraph)

We test whether cryptographically and physically rooted AI identity can be made falsifiable, not whether it sounds profound. Four bets in parallel: one publishable positive (VCEK/TPM-EK hard-lock on small LLMs), one constitutive-physics positive (NS-RAM floating-bulk PUF on FPGA, ≥2 boards), one cheap external-method falsifier (LockedApart-HIP port on gfx1151), one shippable null deliverable ("The Abstraction Tax" — user-space silicon binding on commodity x86 APU is closed-null under TOST). SUCCESS by 2026-07-17 = (a) TPM-EK hard-locked Qwen3-0.6B with self ≥ baseline−2pp AND transplant ≤ chance+2σ across 3 chassis with adversarial bounds AND multi-board NS-RAM PUF result with HD_inter/HD_intra ≥ 5; OR (b) clean null paper on arXiv + confirmatory LockedApart-negative + honest pivot to attestation-services. FAILURE = any of: (1) by 2026-06-12 LockedApart-HIP fails d≥3 thermal-matched gate AND VCEK sweep finds no (|P|,L) regime → user-space and crypto-LLM arms collapse, ship null-only. (2) by 2026-06-30 "Abstraction Tax" not on arXiv → program in motivated denial, declared failed regardless of positive-arm theater. (3) by 2026-07-17 NS-RAM intra/inter-HD on ≥2 boards shows HD_inter/HD_intra<3 OR only one board exists → constitutive-identity arm cannot be claimed; downgrade to "cryptographic + null" if H1 passed, pure-null if not. The consciousness framing is excised. Butlin 10/14 by-construction count is removed from all quantitative claims. We build substrate-locked models for anti-piracy / attestation, not minds. That is the only register in which this work survives external contact.

## Falsification principles (pre-registered, sha-stamped)

- Pre-registration before any data touches eyes. Every hypothesis below has acceptance + kill gates committed to git BEFORE the experiment runs. Post-hoc threshold edits are program-ending misconduct.
- Equivalence (TOST) not vanity-d. Null claims require pre-registered ROPE/equivalence bounds with stated α=0.05, 1−β=0.9, and MDE. Cohen's d alone is barred as a decision rule.
- Block-level cross-validation, never sample-level. All classification gates use trace-block-split CV. Sample-level CV is a hard error and the result is voided.
- Thermal matching for any per-CU/per-cell timing claim. Effect must survive ±0.5°C ambient match across hosts or 15°C ambient shift on a single host reproducing it kills the claim (O95 Arrhenius killer).
- Spoofing controls outnumber positive arms. For every substrate-conditioning claim, at least three spoof distributions tested: matched-moments Gaussian, matched-spectrum (AR(1)+1/f), replay-from-log. If only iid-Gaussian fails, the claim is dead.
- Family-wise error correction (Benjamini-Hochberg q=0.05) across all tests in a paper.
- Null paper ships first. "Abstraction Tax" goes to arXiv before any positive arm finishes, so positive results cannot retroactively soften the null.
- No consciousness framing in quantitative claims. Butlin indicators excised from README and abstracts. Milinkovic-Aru remains as motivation, not as a measured property.
- N for die-level claims requires ≥2 identical-SKU units. ikaros+daedalus+zgx = chassis/platform binding only.

## Hypotheses

### H1 — VCEK/TPM-EK hard-lock on LLM (cryptographic substrate binding)
- **Claim:** A non-trainable permutation P derived from TPM-EK via KDF, injected at a single mid-layer of Qwen3-0.6B, admits a regime (|P|*, L*) where train-time accuracy is within 2pp of unlocked baseline AND transplant accuracy on another host falls within 2σ of chance, across ≥3 chassis and ≥5 seeds, on MMLU-subset (not CIFAR).
- **Mechanism:** Cryptographic: KDF(TPM_EK) deterministically generates P. Forward pass applies P to activations of layer L. P_target ≠ P_host on transplant → activation distribution mismatch → output collapse. Empirical question = the capacity tradeoff curve + adversarial robustness.
- **Experiments:**
  - E1.1 Permutation-strength sweep: |P| ∈ {8, 32, 128, 512, 2048} × L ∈ {6, 12, 20}, 5 seeds, MMLU-subset eval, self vs 2 other-host transplants. Pre-reg: TOST that other-host ≡ chance within ±2σ.
  - E1.2 Adversarial recovery: black-box query attack (≤10k pairs) + white-box closed-form attack on weights. Pre-reg: black-box must require ≥10^6 queries OR white-box must require breaking AES-256.
  - E1.3 Attestation-gated load: replace static P with runtime TPM-quote check. Replay-attack logged quote; cross-chassis EK forgery. Pre-reg: replay accept rate = 0/100 trials.
- **PASS:** (a) ∃(|P|*, L*) with self_acc ≥ baseline−2pp AND transplant_Δ ≥ 60pp AND other_host_acc ≤ chance+2σ on MMLU; AND (b) black-box attack fails at 10k queries; AND (c) attestation gate accepts 0 replayed/forged quotes.
- **KILL:** No (|P|*, L*) satisfies (a) at any tested point — i.e., curves cross such that self>80% always implies transplant>chance+5pp. Also KILL if black-box recovery succeeds in <10k queries.
- **Cost:** ~40 GPU-hours zgx + ~16 analyst-hours. Wall: 7 days E1.1, +7 days E1.2/E1.3.

### H2 — LockedApart-HIP per-CU contention fingerprint on gfx1151
- **Claim:** Atomic-contention compute-shader probe yields per-chassis classification AUC ≥ 0.95 on held-out trace blocks across 3 chassis, Cohen's d ≥ 5.0 between-chassis vs within-chassis, AND signal survives thermal matching (within-chassis cross-temp AUC ≤ 0.6).
- **Mechanism:** Per-CU scheduling latency under atomic race contention is dominated by physical CU layout + per-CU LDS arbitration — die-bound properties. LockedApart reports 310× speedup, 1.8× accuracy over DrawnApart on commodity GPUs; question is whether ROCm scheduler determinism on RDNA3.5 nullifies it (as it did our SALU-direct probes).
- **Experiments:**
  - E2.1 HIP port of atomic-contention kernel (atop existing scripts/identity_benchmark/puf_kernel.hip). 3 chassis × n=1000 traces, block-CV split.
  - E2.2 Thermal-match protocol: ikaros at 20/30/40°C ambient (chamber if available, log otherwise), daedalus+zgx ambient logged. Mixed-effects ANOVA: temperature-explained variance <20% of between-chassis.
  - E2.3 Replay/spoof control: record daedalus trace, replay through ikaros hash pipeline; must classify as daedalus (proves channel is timing, not controller).
- **PASS:** (a) cross-chassis AUC ≥ 0.95 on held-out blocks; (b) Cohen's d between-chassis ≥ 5.0; (c) within-chassis cross-temp AUC ≤ 0.6; (d) temp variance <20% in mixed model.
- **KILL:** Any of (a)–(d) fails. d<3.0 under thermal match is hard kill. Publish as confirmatory negative.
- **Cost:** ~6 GPU-h/chassis × 3 + 1 day port + 1 day analysis. Wall: 5 days.

### H3 — NS-RAM floating-bulk multi-τ as PUF primitive (constitutive identity)
- **Claim:** NS-RAM 128-neuron cells on FPGA bitstream exhibit intra-device HD < 5%, inter-device HD ≈ 50%, BER < 1% over thermal cycles 25→85°C, HD_inter/HD_intra ≥ 5 across ≥2 boards.
- **Mechanism:** Floating-body charge retention variability at τ_fast/τ_mid/τ_slow set by per-cell process mismatch (Vth, capacitance, leakage). Same cell that holds analog weight defines identity — constitutive in Milinkovic-Aru sense. Adjacent SOTA (40nm ReRAM PUF BER<0.5%/10yr) proves physics class works.
- **Experiments:**
  - E3.1 Intra-board enrollment: 128 cells × 30 enrollments × 3 ambients (25/35/45°C).
  - E3.2 Inter-board enrollment: same on board B.
  - E3.3 Thermal cycling: 100 cycles 25→85°C on board A, re-enroll.
  - E3.4 Min-entropy (NIST SP 800-90B) ≥ 0.9 bits/cell.
- **PASS:** intra-HD < 5% AND inter-HD > 45% AND HD_inter/HD_intra ≥ 5 AND BER < 1% post-100-cycles AND min-entropy ≥ 0.9 bits/cell on ≥2 boards.
- **KILL:** HD_inter/HD_intra < 3 OR BER > 5% post-cycling OR only one board available at week-4 gate.
- **Cost:** ~30 FPGA-h, ~$0–$3k (second board if not in hand), ~24 analyst-h. Wall: 4 weeks.

### H4 — Phase-2 equivalence (formal null re-analysis)
- **Claim:** On commodity x86 APU user-space, |Δ_HW − Δ_SHUFFLE| < 0.5% with α=0.05 and 1−β=0.9 via TOST — establishing formal equivalence, not merely a failed difference test.
- **Mechanism:** If substrate channel carries identity, transplant Δ should exceed random permutation by a margin larger than measurement noise. Phase-2 data has the n; what's missing is pre-registered equivalence bound + BF against chassis-confound model.
- **Experiments:**
  - E4.1 Re-analyze existing Phase-2 data with TOST at ±0.5% bound. Report p_lower, p_upper, BF_10 vs chassis-confound model.
  - E4.2 Power analysis: state MDE at observed n; if MDE > 0.5%, re-collect.
  - E4.3 Apply BH q=0.05 across all 9 mechanisms + 18-probe sweep.
- **PASS:** TOST p < 0.05 AND BF_10 < 1/3 vs chassis-confound model AND MDE ≤ 0.5% at achieved power.
- **KILL:** MDE > 0.5% — recollect or downgrade null paper claim from "equivalent" to "underpowered against MDE=X".
- **Cost:** ~16 analyst-h, 0 GPU-h. Wall: 3 days.

### H5 — Token-level substrate-telemetry conditioning (constructive side-channel identity)
- **Claim:** LLM head conditioned token-by-token on live PM-table thermal hotspot + VRM ripple at 50Hz becomes substrate-locked such that replay-from-log accuracy drops ≥20pp vs live-telemetry on same chassis AND matched-spectrum (AR(1)+1/f) synthetic telemetry also drops ≥20pp.
- **Mechanism:** Model learns to depend on higher-order structure (autocorrelation, 1/f slope, cross-channel MI) that simple noise cannot reproduce. Substrate becomes non-replayable nonce because telemetry is path-dependent on real-time workload + thermal history. EMBODIMENT7 killed per-init conditioning; per-token is a different mechanism.
- **Experiments:**
  - E5.1 Train substrate-conditioned head on host A live telemetry, eval on (i) host A live, (ii) host A replay-from-log, (iii) host B live, (iv) matched-μ,σ Gaussian, (v) matched-spectrum AR(1)+1/f.
  - E5.2 Sensitivity probe: output Jacobian wrt telemetry channels; reject if concentrates in <3 dimensions.
  - E5.3 Cross-channel MI test: shuffled-time telemetry (preserves marginals, destroys temporal structure); must collapse.
- **PASS:** (i) ≫ all of (ii),(iii),(iv),(v) by ≥20pp on perplexity AND Jacobian effective rank ≥ 3 AND shuffled-time collapses (≥20pp).
- **KILL:** Matched-spectrum (v) within 5pp of live (i) — model learned low-rank distribution, not substrate identity. Also KILL if Jacobian rank <3.
- **Cost:** ~20 GPU-h zgx + ~30 analyst-h. Wall: 2 weeks (pre-reg wk4, train+eval wk5-6).

### H6 — The Abstraction Tax — null deliverable
- **Claim:** On commodity x86 APU (Strix Halo, gfx1151) under user-space ROCm + Linux, no user-accessible channel produces a per-die identity signature distinguishable from chassis confounds at d≥3.0 with TOST-confirmed equivalence to a random permutation in transplant.
- **Mechanism:** Abstraction layers (ROCm scheduler, Linux kernel, firmware, DPM governor) homogenize per-die variance below user-space detectability. Channels appearing discriminative at d>30 (power, thermal, NVMe) are demonstrable chassis confounds.
- **Experiments:**
  - E6.1 Consolidate Phase 1a/1b/1c/2 + L1–L15 sweep + FABRICCRYPT + EMBODIMENT2/7 into one manuscript with pre-registration appendix.
  - E6.2 Add H4 TOST result as headline statistical claim.
  - E6.3 Add H2 LockedApart-HIP result as confirmatory evidence.
  - E6.4 arXiv submission before any positive arm (H1/H3/H5) reports.
- **PASS:** Submitted to arXiv by 2026-06-30 with pre-registrations, TOST complete, FWER-corrected tables.
- **KILL:** Not submitted by 2026-06-30 — program in denial about its own negative result.
- **Cost:** ~60 analyst-h writing + reviewer round. Wall: 3 weeks.

## Killshots (run first)

- **KILLSHOT-1** (Week 1 Fri): VCEK permutation sweep on Qwen3-0.6B + MMLU. If no (|P|, L) yields self ≥ baseline−2pp AND transplant Δ ≥ 60pp across 3 chassis × 5 seeds → crypto-binding arm = "XOR with extra steps". Drop to null + exploratory NS-RAM. ~40 GPU-h.
- **KILLSHOT-2** (Week 1–2): LockedApart-HIP across 3 chassis thermal-matched. AUC<0.95 OR within-chassis cross-temp AUC>0.6 OR d<3.0 → ALL user-space per-CU timing leads closed. ~18 GPU-h.
- **KILLSHOT-3** (Week 4): NS-RAM intra/inter-HD on ≥2 boards. HD_inter/HD_intra < 3 OR only 1 board → constitutive arm unprovable. ~30 FPGA-h + board procurement.
- **KILLSHOT-4** (Week 5–6, conditional): Token-conditioning matched-spectrum control. Matched-spectrum within 5pp of live → "substrate as nonce" dies. ~20 GPU-h.
- **KILLSHOT-0** (PROCESS, Week 3): Null paper not on arXiv by 2026-06-30 → program in motivated denial, declare failure.

## Timeline

**Week 1 (2026-06-09 to 2026-06-15) — KILLSHOTS.**
- Day 1–2: VCEK permutation sweep on Qwen3-0.6B, L=12, |P|∈{8,32,128,512,2048}, 5 seeds, train on zgx, transplant-eval on ikaros+daedalus. ~16 GPU-h zgx.
- Day 2–4: LockedApart-HIP port atop puf_kernel.hip. ikaros+daedalus+zgx, n=1000 traces/chassis, block-CV, thermal-matched controls.
- Day 5: TOST equivalence on Phase-2 with pre-registered ±0.5% bound.
- Day 6–7: Lock null-paper draft in papers/null_abstraction_tax/.

**Week 2 (2026-06-15 to 2026-06-22).** VCEK extended to 3 layer depths × MMLU; adversarial probe black-box ≤10k queries + white-box closed-form. NS-RAM 2-board protocol committed; order second FPGA board. Intra-board enrollment on existing 128-neuron bitstream.

**Week 3 (2026-06-22 to 2026-06-29).** Submit "Abstraction Tax" to arXiv (cs.CR + cs.LG). If board B arrived: inter-board enrollment. LockedApart-HIP writeup positive (TOSC/WiSec) or confirmatory-negative appendix.

**Week 4 (2026-06-29 to 2026-07-06).** VCEK product-grade with attestation-gated load (AttestLLM); demonstrate adversarial bound (weight-extraction TPM-removed, replay TPM quote, cross-chassis forgery via known EK). Token-conditioning pre-reg with 3 spoofing controls to OSF.

**Week 5–6 (2026-07-06 to 2026-07-19).** If pre-reg approved: train substrate-conditioned head, 4-arm control. NS-RAM aging: 72h continuous + 100 cycles 25→85°C on board A. Decision gate end wk6: if VCEK-attested AND NS-RAM HD_inter/HD_intra ≥ 5 → positive paper. If only VCEK → 3-paper bundle (VCEK + null + LockedApart-negative) reframed as "Cryptographic Substrate Binding: The Honest Boundary."

**Total:** ~50 GPU-h zgx, ~30 FPGA-h, $0–$4k hardware, ~120 analyst-h.

## Week-1 top-3 actions

1. **ACTION-1** (today): VCEK permutation sweep on zgx. Adapt scripts/identity_benchmark/vcek/02_train.py to Qwen3-0.6B, KDF(TPM_EK)→P at L∈{6,12,20}, |P|∈{8,32,128,512,2048}, 5 seeds, MMLU-subset (STEM 500 prompts). Commit pre-reg sha-stamp BEFORE run. Files: scripts/identity_benchmark/vcek/h1_qwen_sweep.py (new), research_plan/H1_PREREG_2026-06-09.md.
2. **ACTION-2** (today, parallel): Port LockedApart atomic-contention into scripts/identity_benchmark/locked_apart.hip. N workgroups race atomic_add on shared counter, per-CU finish-cycle skew via hwreg(29). 3 chassis × n=1000 × thermal controls. Block-CV. Pre-reg before run: research_plan/H2_PREREG_2026-06-09.md.
3. **ACTION-3** (today, parallel, analyst-only): TOST equivalence on results/IDENTITY_BENCHMARK_2026-05-30/phase2/. Bound |Δ_HW − Δ_SHUFFLE| < 0.5%, α=0.05, 1−β=0.9. Report MDE. BH q=0.05 across 9 mechanisms + 18-probe. Output: results/IDENTITY_NULL_2026-06-09/tost_phase2.md. Null-paper outline checked into papers/null_abstraction_tax/outline.md by Wed.
