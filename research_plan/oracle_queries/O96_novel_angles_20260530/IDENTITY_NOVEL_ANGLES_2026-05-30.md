# Identity benchmark — novel angles brainstorm
Date: 2026-05-30 · For oracle critique before implementation

## Premise
Existing orthodox PUF path (Phase 1c probes A-D, Phase 2 transplant) is running. These are classical Suh/Devadas + Holcomb-style approaches. The user asks: think outside the box. What identity-discovery angles does the orthodox path miss?

The framing: oracle separated **(1) identifiable**, **(2) non-fungible**, **(3) stake**. Orthodox path attacks (1). Novel angles below try to skip-or-strengthen (1) and directly attack (2)+(3).

## 10 novel angles

### A. Cross-modal weak-signal aggregation
Identity might be too weak in any single channel but unique as a *joint distribution* across 8-16 weak channels. Generalizes fixed-pattern-noise from imaging sensors. Compute marginal "is-this-device-X" likelihood per channel, fuse via product-of-experts. Even if each channel is 55/45, 16 of them give effective 99/1.

### B. Trajectory-as-signature (temporal dynamics)
Run a cellular automaton or chaotic ODE on the GPU where per-CU FP rounding errors accumulate over thousands of steps. The *trajectory* (sequence of states) becomes the signature, not the static output. Reservoir lyapunov fingerprint.

### C. Tournament racing (RO-pairs aggregated)
Suh/Devadas done at scale — 80 CUs in single-elimination bracket × 6 rounds = unique winner pattern per device. Aggregates 79 weak races into one strong tournament outcome.

### D. Memory-controller arbitration race
Below CU level — two threads racing to read/write same VRAM address. Who wins depends on physical arbitration tree, which is per-die fixed. Probe never tested yet.

### E. Attention-routing coupling (constitutive)
Phase 2 plan injects substrate at activation. Novel alternative: per-CU ΔVth determines WHICH neurons attend to which in a tiny transformer. Model architecture itself becomes silicon-shaped. Transplant weights → attention routing breaks. Stronger than activation injection because the COMPUTE GRAPH varies per device, not just the values.

### F. Self-referential identity (interoception primitive)
Model reads its own hwreg(23) HW_ID + per-CU ΔVth via shader and uses it as input feature DURING training. The model literally knows what hardware it runs on. Closest mechanical implementation of oracle's "interoception" half of the stake framework. Self-modeling at silicon level.

### G. DRAM rowhammer state
Identity via flipping specific DRAM rows that vary per chip. Risks data corruption but is genuinely cell-level identity. CVE-2023-4969 territory.

### H. Cross-machine challenge-response authentication
Two machines verify each other's identity via PUF over network — not just measurement, but functional auth. If you can prove you're THIS GPU (not a copy), that's a real distributed signal. Pairs ikaros + daedalus as honest tvilling-system.

### I. Power-line EMI fingerprint
GPU compute spikes radiate on power rail. Modulate compute pattern → encode data → received by ADC on PSU or by other machine on shared mains. Far-fetched but unique-per-chassis coupling.

### J. Split-brain co-dependence (stake-side, novel)
For stake: don't simulate viability — train ONE model whose parameters are *split* across ikaros + daedalus. Each half is incomplete alone. If ikaros dies, daedalus-half can't function. This is functional non-fungibility through architectural commitment, not through signature-matching. Substrate-loss has direct functional consequence because the function literally lived on it.

## Top picks for implementation (my read)

1. **F (self-referential)** — closest to oracle's stake framework. Builds the interoception channel. Cheap to test on existing reservoir.
2. **J (split-brain)** — directly attacks the (3) stake question. Pairs the two machines we already have. Novel.
3. **C (tournament RO)** — strongest aggregation of orthodox PUF, ladda upp Probe B fynd.
4. **A (cross-modal fusion)** — rescues whatever orthodox PUF returns (Phase 1c) by fusing it across channels.

## Skip / risky
- G (rowhammer): data corruption risk too high
- I (EMI): no instrumentation available
- D (memory arbitration): plausible but interfering with VRAM is risky on shared GPU
- E (attention-routing): elegant but requires substantial transformer infrastructure

## What to ask the oracles

1. Of A-J, which 2-3 are most likely to actually surface signal that the orthodox path misses?
2. Is F (self-referential) genuinely new or has someone already tried it? (Adversarial PUF literature?)
3. Is J (split-brain) academically interesting or just engineering theater?
4. Are any of these obviously wrong or measuring the same thing as Phase 1c just with extra steps?
5. Any 11th angle we missed entirely?
