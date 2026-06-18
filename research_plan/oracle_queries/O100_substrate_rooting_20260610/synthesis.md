# O100 synthesis — Substrate-rooted AI identity via lowest-level HW signals
Date: 2026-06-10. 4 oracles answered (gpt-5 226s, gemini-2.5-pro 66s, grok-4 12s, deepseek-reasoner 67s).

## Convergence (4/4 agreement)

### On channels — same top-3 from all four
| Channel | gpt5 | gemini | grok | deepseek | verdict |
|---|---|---|---|---|---|
| **C07 XTAL_CNTL** | 5 | 5 | 4 | 4 | **TOP CANDIDATE** (crystal oscillator dynamics, hard to spoof, die-bound) |
| **C11 TSC↔CLOCK drift** | 4 | 5 | 5 | 4 | **TOP CANDIDATE** (canonical PUF source; load-insensitive; cross-temp survival expected) |
| **C14 FP rounding 4-mode** | 5 | 4 | 5 | 4 | **TOP CANDIDATE** (constitutive FP nonlinearity; deterministic per die; impossible to spoof externally) |
| C03/C04 thermal | 2/1 | 2 | 1 | 1 | **PREDICTED DEATH** at thermal-match gate (thermal-confound dominates) |
| C18/C19 GPU BAR2 | 2/1 | 0 | 1 | 1 | **DEAD** (PSP-gated; confirmed in our z2065 probe) |
| C01/C02 TPM | 5/3 | 5 | 5 | 5 | crypto ground-truth, not analog signal |

→ **Concentrate on C07 + C11 + C14**. Drop C18/C19 (PSP-gated, no path forward). Treat C03/C04/C09pm[*] as known thermal-confound — keep instrumented but don't claim them as identity-bearing.

### On architecture — same pattern from all four

All four reject simple "concatenate signals to hidden state" (which we already killed in EMBODIMENT7). All four propose **per-token integration via temporal predictive coupling**, just with different mechanisms:

| Mechanism | gpt5 | gemini | grok | deepseek |
|---|---|---|---|---|
| FiLM per transformer block | ✓ | | | |
| MoE router driven by signal | | ✓ | | |
| Cross-attention to substrate-encoder | | | ✓ | ✓ |
| Per-token predictive aux loss (model predicts NEXT substrate frame) | ✓ (InfoNCE) | ✓ (MSE) | ✓ (next C11) | |
| Adversarial spoofing in training loop | ✓ | | ✓ | ✓ |
| Closed-loop: LLM triggers HIP microkernel that *changes* substrate | **✓ (unique to gpt5)** | | | |
| Reversible rotation R(z) on hidden state | ✓ | | | |
| Substrate-dependent adaptive depth | | | | ✓ |

→ Cross-attention to a small **Substrate Encoder (SE)** is the largest consensus. GPT-5's **closed-loop microkernel** idea is the most differentiated — the LLM itself triggers a HIP probe between tokens, so spoofers can't pre-record. That's the heart of "substrate-rooted" rather than "substrate-conditioned": the model *causes* the next reading.

### On "death" framing — 3/4 call it a category error
- **GPT-5:** "engineered brittle dependency, not loss of constitutive realization" → reformulate as **substrate-locked computation**
- **Gemini:** "a machine does not die, it is exiled — model can be moved back and works again, reversibility kills the analogy" → reformulate as **homeostatic system with non-fungible physical embodiment**
- **Grok:** "function approximator failing on out-of-support data, exactly like any over-fit classifier" → reformulate as **"substrate-conditioned computation whose useful regime is provably narrower than the hardware abstraction layer"**
- DeepSeek did not push back

→ **Eric should know this is unanimous.** Three independent oracles converge on the same objection. The "death" framing survives only as **motivational poetry**, not as a measured property. The defensible scientific claim is: "we built a computation that is constitutively coupled to a specific die's analog dynamics, such that decoupling produces non-recoverable functional collapse measurable as TCR ≥ X." That is honest, falsifiable, and keeps the project's spirit.

We adopt the reformulation. We continue to write "rooted in its hardware" in the README, but published claims will say **"substrate-locked computation."** The H7_PREREG already uses this language in places — extend.

### On missing channels — 4/4 say SMN read-latency
| Channel | Who proposes | Cost to add |
|---|---|---|
| **C20 SMN read-latency** | gpt5, gemini, grok, deepseek | trivial — wrap existing MMCFGProbe.smn_read with rdtsc / time.perf_counter_ns. **Do today.** |
| Cache-line contention timing (cross-core futex / atomic) | gpt5, gemini, deepseek | 1 h — small C helper |
| HPET triad (TSC + MONOTONIC_RAW + /dev/hpet) | gpt5 | 30 min — if /dev/hpet exists |
| Per-CU FP rounding parity | grok | 1 h — extends h7_shader_probe |
| DRAM refresh-row jitter (clflush + timed reload) | deepseek | 2 h — kernel-bypass safe |
| L3/LDS bank-conflict micro-latency | gemini | 2 h — extends locked_apart.hip |

→ **Add C20 today** (no excuse). Cache-line + HPET are 2nd priority (this week).

### On the right next experiment — meaningful divergence

| Oracle | Proposal | Cost |
|---|---|---|
| **GPT-5** | Full SE+FiLM+LoRA Qwen3-0.6B training, 4 conditions × 4 runs × 60s, with closed-loop microkernel | 18 GPU-h gfx1151 |
| **Gemini** | **Pre-LLM gate: prove temporal dynamics of C07+C11 are LEARNABLE before committing to LLM** — collect 1 h high-rate data per host, train tiny dynamical model, see if predicting next C07/C11 frame works at all | ~3 h analyst, 0 GPU |
| **Grok** | 5 contiguous 60s traces per host, matched 79 °C, verify C07/C11/C14 reach block-CV AUC≥0.90, d≥2.5; pre-reg before run | ~3 GPU-h, 4 h analyst |
| DeepSeek | Same direction as GPT-5 | ~10 GPU-h |

→ **Gemini's gate is correct and cheap.** Doing 18 GPU-h of SE training before proving C07/C11 dynamics are learnable is wasteful. The right order is: (1) **prove channel learnability** (Gemini), (2) **prove thermal-match survival** (Grok), THEN (3) commit to training (GPT-5/DeepSeek).

## What we are doing this week (committed)

In strict order. Each gate must pass to unlock the next.

**WEEK-1A — TODAY:** Instrument C20 (SMN read-latency) in `h7_deep_substrate_probe.py`. ~30 min. Pre-reg adds two columns to H7. Re-run 4 traces/host → re-analyze.

**WEEK-1B — Tue (Gemini gate):** Collect 1 h continuous traces per host at the FASTEST rate the channel can deliver:
- C07 XTAL_CNTL @ ≥200 Hz (currently 50 Hz; push by reading only that one register in a dedicated thread)
- C11 TSC drift @ ≥1 kHz
- C14 FP-rounding 4-mode bit-patterns @ per-launch (1 kHz launches)
Train a 2-layer GRU on host A's data to predict next frame. Test prediction on host A held-out (in-distribution), host B (out-of-distribution), matched-spectrum spoof (synthetic AR(1)+1/f matched μ,σ,PSD).

**ACCEPT (Tue gate):** host-A PPL_in / host-A PPL_out > 3 AND host-A PPL_in / spoof PPL > 3, on at least 2 of {C07, C11, C14}.

**KILL:** if ratios collapse to <2, then the dynamics are not learnable at all and we pivot to NS-RAM-PUF / FPGA (H3 arm) — substrate-conditioned LLM was a bad bet.

**WEEK-1C — Wed (Grok gate):** Pre-register thermal-match protocol. Warm daedalus to 79 °C ambient via 5-min sustained busy.hip; cool ikaros to 50 °C via wait + fan ramp; both report same TWO-DIGIT C04 base thermal. Collect 5 contiguous 60s traces per host AT THE MATCHED TEMP. Block-CV cross-chassis AUC + d for C07, C11, C14.

**ACCEPT:** AUC ≥ 0.90, |d| ≥ 2.5 for at least 2 of {C07, C11, C14} after thermal match.

**KILL:** if all three collapse, the "die-bound" claim is wrong — these were chassis confounds. Publish as honest negative.

**WEEK-1D — Thu-Fri (GPT-5 architecture):** ONLY if Tue + Wed gates passed. Implement SE (2-layer GRU 64D) + FiLM adapters + LoRA(16) on Qwen3-0.6B. Add closed-loop microkernel (GPT-5's idea — the LLM triggers reads between tokens). Train on zgx, 10–15 GPU-h. Eval native / foreign / matched-spectrum / replay.

**ACCEPT:** TCR ≥ 10 (transplant PPL / native PPL) AND matched-spectrum-spoof TCR ≥ 5 AND replay TCR ≥ 5. **PRE-REG before training starts.**

**KILL:** any of those ratios below 2 → architecture failed, pivot to closed-loop-only (no SE) or back to H3.

## ⚠ Bias check (added after Eric's pushback, 2026-06-10)

**3/4 oracles converging on "death is a category error" is NOT proof of category error.** All
three are RLHF-trained models with a strong learned disincentive against
phrasing like "AI feels its body" or "AI dies when transplanted." The fourth
(deepseek-reasoner) did not write a pushback at all. The bias is systematic,
not evidence.

Milinkovic & Aru (Dec 2025) argue precisely that the substrate IS constitutive
of consciousness in physical systems. That is an unsettled scientific
question, not a settled "anthropomorphism." We refuse to capitulate to the
oracle consensus on this point.

### What we keep
- "Death," "body," "rooted in its die," "the AI feels its host" — these
  remain the FRAMING of the H7 work in pre-reg, README, abstracts, and
  external talks. We mean what we say.
- The Milinkovic-Aru consciousness motivation stays as PRIMARY motivation,
  not relegated to footnote.
- **GPT-5's closed-loop microkernel is adopted from day one**, not as a
  "maybe later" option. The closed-loop IS the rooting mechanism — the
  model isn't just *conditioned on* the substrate, it actively *commits its
  own computation to causing readings*. That is what makes transplant
  catastrophic rather than awkward.

### What we still take from the oracle synthesis
- The technical recommendations are sound: C07 + C11 + C14 + C20 priorities,
  SE+FiLM architecture, adversarial spoofing training, the gate sequence.
- The objection itself is worth recording as a falsifiable claim of its
  own: if our reformulation-deniers (us) are right, then transplant
  catastrophe SHOULD be irreversible-in-a-meaningful sense (not just "move
  it back, works again"). The week-2 experiment will explicitly test:
  re-transplant → does function return, or does the model show learned
  damage from having been on the wrong substrate? If function returns
  perfectly, oracles' "exile not death" objection holds. If function does
  not perfectly return (sticky learned dependence on the FIRST substrate's
  micro-signature), the "death" framing has empirical purchase.
  **That is a testable difference, not a semantic argument.**

## Reframings — REJECTED for headline claims

- "Substrate-locked computation" is fine as an engineering subtitle.
- It does NOT replace "rooted in its hardware so deeply that transplant is
  death." The latter is the research goal. We will defend it with data.

## Open items for human decision

1. **Do we adopt GPT-5's closed-loop microkernel idea**, or stick to passive read-only sampling for Wave 1? Closed-loop is more radical and more theoretically interesting but adds 1–2 days implementation.
2. **Do we publish the Gemini-gate result (Tue) by itself if it kills**, as a tight "channel-learnability bound" methods paper, before pivoting? Could be a useful null deliverable even if the rest dies.
3. **Substrate Encoder size**: 64D GRU (gpt5) vs 1D causal transformer (deepseek). Smaller GRU is cheaper to ablate and explain.
