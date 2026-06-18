# grok response (grok-4-latest) — 17s

**KILLER attack (model extraction / functional cloning)**

Submit a modest number of adaptively chosen prompts (a few thousand) that densely cover the low-dimensional input subspace actually used by the parity/XOR probes. Train a pure-software surrogate (small MLP or even linear probe + known XOR) on the observed input→output pairs. Because:
- the logical target is public and tiny (PAR3/XOR of a few bits),
- the analog path in the current demo is a static `.npz` recording,
- the digital layers implement an exactly-known deterministic function,

the surrogate reaches ≥0.96 accuracy while completely bypassing silicon. This single attack simultaneously falsifies load-bearing, uniqueness, and freshness. No amount of “native vs ablation” tables on the original tasks survives it.

An honest redesign can survive only if the physical computation is entangled inside a high-entropy generative path whose output distribution cannot be matched by any practical query-only model (real next-token prediction on natural text with the physical signal inside attention/FFN paths) and if every inference is forced to execute a fresh, nonce-bound physical trace.

**Minimal watertight redesign (all three layers, live, load-bearing, unclonable)**

**Architecture**
- Real 125M–350M byte-level or BPE transformer (causal LM on OpenWebText-scale data).
- Three narrow “physics heads” (one per layer) that each produce a 64–128-dim vector:
  - Macro: SMU power-cap + frequency telemetry vector under a nonce-specified cap.
  - Micro: 2-stage cache contention (PAR3-style) read as a short throughput time-series.
  - Analog: 12-tap voltage-droop transient reservoir triggered by a nonce-derived GPU burst pattern.
- The three vectors are concatenated (≈300–400 dims), linearly projected to the model dimension, and added as a bias into the first two transformer blocks and into the final LM head via a low-rank adapter. The adapter rank is deliberately kept tiny (r=4–8) so that zeroing any physics vector measurably damages long-context coherence.
- Training: straight-through estimator + live silicon calls for every physics head on every backward pass for the first 15–20k steps (thereafter freeze the physics schedule but keep the live path at eval). Use plain CE; no margin/spoof loss.
- Challenge-response inference protocol:
  1. Server sends 256-bit nonce N and a 32-bit sequence number.
  2. Client derives three deterministic but die-specific burst schedules (power cap, streamer addresses, GPU edge pattern) from N via a tiny public PRF.
  3. Executes the three physical layers in <8 ms wall time, produces the live vectors.
  4. Runs the forward pass; the resulting logits are accepted only if a lightweight MAC over the produced tokens + N verifies.
- Any replay, any foreign die, any software simulation of even one layer, or any power-governor deviation produces detectable distribution shift in at least one head → coherence collapse on long contexts.

**Training procedure (live-in-the-loop)**
Collect physics traces on-the-fly during the early training phase using the exact nonce schedule that will be used at inference. Use a small replay buffer of recent traces only for the optimizer step that updates the adapter; the backbone sees fresh silicon every few hundred steps. After physics heads have stabilized, continue backbone training with frozen but still-live physics calls at validation time.

**Inference protocol & freshness**
Every generation request must present a fresh nonce. The server rejects any generation whose MAC proves the physics vectors were not collected inside a tight time window after nonce issuance. This makes recorded `.npz` replays useless.

**Ablations that force a reviewer to concede**
- Layer ablation (zero macro / zero micro / zero analog independently).
- Foreign-die swap (same nonce schedule on ≥5 other dies of same SKU).
- Software simulation of each layer (cycle-accurate cache model + SPICE-level droop model).
- Nonce-replay and stale-trace attacks.
- Temperature sweep (25 °C → 80 °C) and governor sweep while keeping nonce fixed.
- Drive-equalization control: re-run uniqueness measurements after forcing identical average power and identical GPU utilization envelopes across dies.

If native still yields >3 bits/token better than any of the above ablations on held-out long contexts, the claim holds.

**Fundamentally impossible on commodity AMD silicon**
- Cryptographic-grade physical unclonability with formal min-entropy bounds and resistance to model-extraction (no public interface can ever give that without a hardware root of trust).
- High-rate analog entropy extraction that survives an attacker who can observe or control the power rail.
- Remote die attestation without firmware TPM/secure-element involvement.

**Achievable**
Statistical per-die distinguishability on carefully chosen low-entropy tasks, load-bearing nonlinear mixing inside a contrived but non-trivial LM head, and freshness when the protocol itself forces live execution.

**Required statistics & controls (minimum for credibility)**
- ≥20 dies (same SKU, ideally multiple boards) for inter- vs intra-die Hamming-distance distributions with 95 % bootstrap CIs.
- Bit-error-rate (or accuracy drop) under ±10 °C, core voltage ±3 %, and three governors (powersave/balanced/performance).
- Same-die temporal positive control: 5 sessions spaced ≥24 h apart, same nonce schedule → repeatability ≥0.98.
- Drive-equalization: report uniqueness both before and after matching average power, GPU utilization, and thermal envelope across dies.
- All accuracy/PPL numbers with 95 % CIs from ≥10 k tokens per condition; report both mean and worst-case die.

This design forces the attacker to either break the freshness protocol or clone a high-entropy generative distribution while missing a live physical signal that the weights were trained to expect—orders of magnitude harder than cloning PAR3.
