# deepseek response (deepseek-reasoner) — 84s

# Hostile Expert Review: Tear-Down, Killer Attack, and Watertight Redesign

## 1. The Single Killer Attack That Destroys the Whole Thing

**Attack: Replace every physical mechanism with a software model trained on public inputs, and achieve identical or better performance without any die-specific hardware.**

### Why it kills everything

- **Digital cache layer** (DEMO A): The cache “XOR gate” is deterministic and near-perfect (fidelity ~0.99 on ikaros). Training already uses the truth table (numpy XOR). At eval, the live silicon merely adds a small error rate (<5%). An attacker can emulate the exact output distribution by training a **binary classifier** on the same public inputs (u1,u2,u3) to predict the cache output. With 256 labeled examples (the 8 possible 3-bit inputs × 32 repeats), they can learn a simple threshold or even a lookup table that matches the die’s behavior to *arbitrary accuracy*. The result: the attacker’s model solves PAR3 at ≥0.944 without touching any AMD die.

- **Analog droop layer** (DEMO B): The recorded `.npz` file is static. An attacker can train a **linear model** on the same u–>Tn mapping from public data (u is generated with known seed). They don’t even need the die; they can simply store the 2600-recorded Tn and replay them. The “uniqueness gap” (0.165) is tiny – a software model using the same u and a random 360-dim projective hash would match or beat that gap by chance. The gap is also inflated by **covariate shift** (drive magnitude 1.17 vs 1.87), not die identity – a single linear regression on drive magnitude alone could explain most of the gap.

- **Combined system**: The “LLM” is a linear classifier on a synthetic parity stream. An attacker can build a **single-layer perceptron** that takes the six-lag window and outputs logits, ignoring the body entirely. Since the body’s contribution is just an additive linear term (`die_head(g)`), and `g` is a deterministic function of `(u1,u2)`, the attacker can absorb that into the bypass matrix. The result: **native accuracy = 0.944 is achievable without any hardware**.

### Conclusion of the attack  
Every claimed dependence on the physical die is **functionally redundant**. The body adds no essential cryptographic or computational advantage that cannot be reproduced with software using public information. The “uniqueness” is a separate, non-load-bearing ornament. Therefore the entire system is **not hardware-rooted** – it is a software model masquerading as embodied.

---

## 2. Watertight Redesign: A True Hardware-Rooted LLM Using All Three Layers

### Design Goals
- **Load-bearing**: The physical layers must be *required* for correct text generation (removing them → catastrophe).
- **Fresh & live**: No recorded data; each inference query must be impossible to replay.
- **Die-unique & unclonable**: The physical response must be a function of the die’s unique process variation, with inter-die gap >> intra-die noise.
- **Three layers integrated**: Macro (firmware power arbitration), micro (cache interference), analog (voltage droop reservoir) are all used simultaneously to compute a single die-specific “signature” for every token.

### High-Level Architecture

Instead of a small parity trick, we embed the die’s signature as an **unguessable key** that modulates the transformer’s weights. The model is **trained live** – every training step queries the physical layers.

**Step 1: Challenge-response protocol**
- At each token position `t`, derive a **challenge** `c_t` (e.g., 64-bit hash of previous token embedding and a public nonce).
- Send `c_t` to the **“body oracle”** – a daemon that runs on the target machine and collects readings from all three layers *simultaneously*:
  1. **MACRO**: set a power-cap value derived from lower bits of `c_t`; read the SMU-reported current and throttle state. Coarse, but nonlinear under power-cap binding.
  2. **MICRO**: run the cache‑interference streamers with inputs `c_t[0:2]` (fast XOR gate), read the throughput sum.
  3. **ANALOG**: perform a sharp GPU burst as in `h7_transient_vdroop.py` but with length determined by `c_t[16:24]`; record the settling transient (120-dim vector).
- The oracle returns a combined **signature vector** `s_t` of dimension ~200 (concatenate: 1 scalar from macro, 2 bits from micro, 120 floats from analog, plus their derived nonlinear features like product terms).

**Step 2: Transformer with FiLM-based die modulation**
- The transformer is a standard causal byte-level LLM (like the one in `h7_rooted_lm_text_embodied.py`).
- At each layer, a **FiLM** conditioner receives `s_t` (projected to per-layer FiLM parameters) and **scales and shifts the hidden states** in each attention head.
- Because `s_t` depends on the challenge (which depends on the context and a fresh nonce), the model **cannot learn the mapping offline** – the exact transformation changes with every token, and with every inference run (nonce prevents replay).

**Step 3: Training procedure – live-in-the-loop with stabilized labels**
- Challenge: `s_t` is not differentiable and is non-deterministic (thermal noise).  
- Solution: Use **self-supervised consistency** – the model is trained to predict the next token using two forward passes per training step: first with the live `s_t` (teacher), second with a **learned surrogate** `ŝ_t` that tries to emulate the oracle. The surrogate is a small network that takes the challenge and tries to reconstruct `s_t`. The model learns to trust the surrogate during inference when the oracle is unavailable? No – we want the oracle live at inference.  
- Better: **Train the model with the live oracle at every gradient step**. This is feasible only for a small model (e.g., 1M params) and short sequences (64 tokens) on a single GPU. The oracle takes ~30ms per token, so 64 tokens = 2 seconds. A 1000-step training run would take ~30 minutes – acceptable for a demo, painful for production. But for a proof-of-concept, it's fine.

**Training stabilisation**:
- The body’s response `s_t` is high-dimensional and repeats for identical challenges only with noise. To avoid the model learning noise, we **average multiple queries** per challenge (e.g., 4 repeats) to obtain a denoised signature.
- The loss is standard cross-entropy on next token prediction – the model **must** use `s_t` to reduce perplexity; if we ablate `s_t` (replace with zeros), perplexity must explode.

**Freshness enforcement**:
- The nonce is a 64-bit random value generated by the server and sent to the client (the die-hosting machine) at the start of each inference session. The challenge includes the nonce, so the oracle’s response depends on both the input context and the nonce. An attacker who records the oracle’s response for a given context and nonce cannot reuse it when the nonce changes.

**Die uniqueness**:
- The analog droop taps are the primary source of die-specific variation. Even with the same challenge, two dies produce different voltage transients. The micro layer also shows subtle timing differences (the exact throughput sums for “01” vs “10” vary slightly across dies). The macro layer’s power arbitration response can also differ. The combined signature thus acts as a **physical unclonable function (PUF)**.
- To avoid covariate shift, we **normalize the signature per die** by subtracting the die’s own mean response computed over 100 calibration challenges and dividing by its standard deviation. The uniqueness gap is then measured as the L2 distance between die-specific signatures for the same challenge, compared to intra-die repeatability.

### Concrete Implementation Sketch

1. **Body Oracle Service**: Runs as a daemon with root access. Communicates via fast IPC (Unix domain socket). Accepts a 64-bit challenge; returns a 196-dim float vector (1+2+120+ other nonlinear terms). The oracle performs:
   - Set GPU power cap using `sysfs` (macro).
   - Spin up the two memory streamers with challenge bits (micro).
   - Execute a `tg-matmul` burst of length `challenge[8:12]` ms, then read 12 taps × 10 channels from SubstrateStateV3 (analog).
   - Compute signature = [macro_throttle, micro_XOR_1, micro_XOR_2] + tanh-normalized TT + pairwise products TT_i * TT_j (nonlinear expansion).
2. **Transformer Model**:
   - Embedding dimension 128, 4 layers, 4 heads.
   - Each layer includes a FiLM module: `γ = W_γ · s_t`, `β = W_β · s_t` (linear projections), then `h = γ * layer_norm(h) + β`.
3. **Training**:
   - Use a small corpus of structured byte sequences (similar to `h7_rooted_lm_text_embodied.py` but larger, e.g., 500k tokens). Each training batch of 4 sequences of length 64.
   - For each token, call the oracle (~30ms per token per batch, so 4×64×0.03 = 7.7 seconds per training step – extremely slow). Optimize: reuse signature for repeated challenges within window? Use caching with TTL.
   - Train for 500 steps; total ~1 hour. Then evaluation.

4. **Ablations**:
   - `native`: live signatures → perplexity PPL_native.
   - `zero`: signatures zeroed → PPL_zero >> PPL_native.
   - `random_die`: signatures from a different die (pre-recorded) → PPL_foreign > PPL_native but < PPL_zero if signature carries die-specific info.
   - `replay_attack`: signatures from a previous session with same context but different nonce → must break (because challenge includes nonce, so signature is different).
   - `software_model`: attacker trains a neural network to emulate the oracle from public challenges and a one-time calibration set (1000 challenges recorded from the die). If the emulator achieves PPL_native within 0.05, the physical body is not required. We must show that no emulator can match the live oracle’s performance because the oracle’s response includes **fresh thermal noise and drift** that changes each session. The emulator trained on old data will fail on new nonces.

### Statistics to Report (to convince a reviewer)

- **Die count**: ≥20 dies (practical limit: 2–3 due to hardware cost, but even 3 dies with repeated sessions provides some evidence). For each die, run 10 sessions on different days, record signatures for 200 fixed challenges.
- **Inter-die Hamming distance (HD)**: For binarized signatures (if we binarize using median), compute mean and std of HD between all pairs of dies. Report min inter-die HD vs max intra-die HD. Need inter-die > intra-die by > 3 sigma.
- **Bit error rate (BER)**: For binarized signature bits, measure BER across 100 repeats of same challenge on same die (same session, different runs). Should be < 5%. If > 5%, apply error correction (e.g., BCH) and report corrected BER.
- **Confidence intervals**: Report 95% confidence intervals for all accuracy/perplexity numbers. Show that native is significantly better than zero (t-test p < 0.01) and better than foreign-die (p < 0.05). Provide effect sizes (Cohen’s d).
- **Temperature/governor sweeps**: Run same die at 50°C, 70°C, and with different CPU governor settings (performance, powersave). Show that intra-die variation is still smaller than inter-die variation.
- **Drive equalization**: For analog, normalize the drive magnitude (e.g., measure actual burst intensity via power meter) and adjust the signature accordingly. Show that uniqueness gap remains after drive normalization.

---

## 3. What Is Fundamentally Impossible on Commodity AMD Silicon

| Property | Achievable? | Reason |
|----------|-------------|--------|
| **Low-latency single-token body query (<1ms)** | **No** (fundamental) | The voltage droop measurement via SubstrateStateV3 requires at least 24ms of settling time per transient (NTAP=12 @ 500Hz = 24ms). Any attempt to reduce this time kills the SNR below usefulness. The cache interference takes ~50ms per gate due to sleep and synchronization. Therefore each body query adds ~30-50ms overhead, making **online training on large sequences impractical**. For a 64-token sequence, inference would add 2–3 seconds per generation – acceptable for a research demo but not for deployment. |
| **Stable, reproducible PUF-like responses (<1% BER)** | **Hard, maybe impossible** | The analog droop is sensitive to temperature, voltage rail noise, and workload history. Our measurements show 3–5% variation across repeated identical bursts. To reduce BER to <1% requires error correction codes (ECC) that add significant redundancy and lower entropy. The micro XOR is more stable but offers only 2 bits of entropy. Combined, the effective entropy may be only ~10–20 bits – far less than a true PUF. |
| **Macro-layer high-speed modulation** | **Partial** | Changing power caps via `sysfs` takes ~10ms. Using it as a per-token signal is too slow. The macro layer is better used as a **global session key** (set at start of inference) rather than a per-token input. |
| **Rooted in a real LLM that runs on the same GPU** | **Possible with compromise** | The body oracle steals GPU cycles for the analog burst, competing with the LLM’s forward pass. We can time-multiplex: while the transformer computes one token, the oracle prepares the next signature. With careful pipelining, total latency per token can stay <100ms. But the transformer must be small enough to fit alongside the burst kernels. |
| **Training without truth-table shortcuts** | **Required for credibility** | Our redesign uses live-oracle training; this is slow but not impossible. We estimate a 1M-param model can be trained in ~2 hours on a single workstation. |

**What is achievable**: A **medium-strength hardware-rooted LLM** that is demonstrably dependent on a specific die’s physical properties, with uniqueness backed by statistical evidence over 3–5 dies. The system will be slow, but that’s fine for a research prototype. It will not be deployable at scale, but it **can** satisfy a reviewer that the concept works.

---

## 4. Exact Statistics and Controls to Report

### Pre-registered Evaluation Plan

For each die (N ≥ 3), run the following:

1. **Calibration**: Collect 1000 challenge-response pairs (fixed nonce = 0) to compute per-die statistics (mean, std of signature vector). Normalize all future signatures by subtracting die’s mean and dividing by its std.

2. **Intra-die stability**: For 100 challenges, repeat the oracle 10 times each within the same session. Compute:
   - Mean signature vector and its standard deviation (per component).
   - BER for binarized signature (threshold = median of that die’s calibration signatures). Report average BER and 95th percentile.
   - Cosine similarity between repeated signatures (should be >0.95).

3. **Inter-die distinctiveness**: For each pair of dies (A,B), compute:
   - Mean Hamming distance between binarized signatures (over same 100 challenges).
   - Compare to intra-die mean HD (should be significantly higher).
   - Report t-test p-value and Cohen’s d effect size.

4. **Load-bearing test** (per die):
   - Train the LLM as described (live oracle) for a fixed number of steps (e.g., 500 steps). Freeze model.
   - Evaluate on a held-out test set of 512 sequences (each 64 tokens) under conditions:
     - **native**: with live oracle for each token.
     - **zero_ablation**: replace signature with zero vector.
     - **foreign_die_ablation**: use signature from another die (pre-recorded for the same challenges? careful: challenges depend on nonce, so we must use the exact same nonce and input contexts; easiest: re-run the same test sequences on the other die and record its signatures).
     - **replay_attack**: use signatures from a previous session (different nonce) for the same input sequences.
   - Report **perplexity** and **accuracy on a binary word-prediction task** (e.g., predict if next byte is 'a' vs 'b' – a task that is easy for a natural LLM but requires the die signature for disambiguation).
   - Show 95% confidence intervals via bootstrap resampling (1000 resamples).

5. **Emulator attack**: Train a neural network (2-layer MLP with 256 hidden units) on 1000 challenge-response pairs from the target die (same session). Then use this network to predict signatures for the test set challenges (with a new nonce, but the emulator has never seen that nonce). If the emulator achieves PPL within 1.0 of native, the physical body is not required. We hypothesize it will fail because the emulator cannot capture the nonce-dependent thermal drift.

6. **Positive control**: Show that the same die across different days (different temperature/load) still yields signatures that work (PPL within 0.5 of same-day native). This proves the signature is stable enough for the LLM to rely on.

### Reporting Format

```
Die count: 3 (ikaros, daedalus, titan)
Inter-die HD (binarized): mean 0.45 ± 0.02; intra-die HD: 0.05 ± 0.01
t(299) = 150, p < 1e-10, Cohen's d = 8.2

Load-bearing perplexity (native): 4.32 [4.28,4.36]
Zero ablation: 12.1 [11.8,12.4] (p < 0.001 vs native)
Foreign die: 6.71 [6.55,6.88] (p < 0.01 vs native, but still lower than zero)
Replay attack: 11.9 [11.5,12.3] (replay fails – fresh nonce required)

Emulator attack: PPL = 11.5 [11.2,11.8] (emulator cannot beat zero ablation)
=> Physical body is load-bearing and unreplaceable.

Temperature sensitivity: PPL native at 55°C: 4.41; at 75°C: 4.45 (not significantly different)
```

---

## Final Verdict

**Current build**: Dead. The killer attack (software XOR emulation) destroys both the load-bearing and uniqueness claims. The analog layer is not load-bearing, and its uniqueness gap is fragile.

**Watertight redesign**: Achievable in principle, but requires:
- Live-in-the-loop training (slow but feasible)
- Challenge-response with fresh nonce per session
- Three layers combined into a single high-dimensional signature that modulates a real transformer via FiLM
- Rigorous statistics over multiple dies with temperature/governor sweeps and replay attack tests

**Impossible on commodity AMD**: Sub-ms per-signature latency; PUF-grade stability (<0.1% BER); large-scale deployment with hundreds of dies. But a research prototype demonstrating the concept is within reach.

**Go build it – or walk away**.
