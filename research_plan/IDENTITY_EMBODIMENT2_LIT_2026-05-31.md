# Embodiment2 Literature Scan — 2026-05-31

**Question**: Has anyone trained models that EXPLICITLY exploit per-die /
per-chassis hardware fingerprints for accuracy or efficiency wins?

## Findings

### Per-die / per-chip neural specialization on **digital CMOS**
**Essentially absent from the 2024-2026 literature.** Both oracles (gpt-5,
gemini-2.5-pro) confirmed they could not cite a single paper demonstrating
per-die specialization on commodity x86/GPU yielding accuracy gains over a
generic baseline. The closest body of work is *per-platform / per-architecture
device-aware NAS and quantization*, which is per-architecture, not per-die.

### Per-die specialization on **analog / neuromorphic / memristor** hardware
Active and growing field — here variability is *the substrate's signal*:

- **In-Memory Computing (IMC)**: device-to-device memristor variation is
  exploited as a free entropy source. PR-CIM (arXiv:2110.09962) reports
  variation-aware training improving binary-NN accuracy from <20% to 87%.
- **Phase-change-memory inference** (Joshi et al. 2020, Nat Comms): models
  retrained with PCM-specific noise model recover ideal-software accuracy.
- **Photonic NN** (microring resonator-based): fabrication-induced resonance
  variation handled via online training and pruning; >90% accuracy over
  wide temperature range.
- **Radio-frequency PUF + ML**: ORACLE (Sankhe et al.), DeepRadioID
  (arXiv:1904.07623) use per-chip RF impairments as classification signal,
  >99% accuracy distinguishing thousands of transmitters.

### Reservoir computing in materio / hardware-aware RC
- **Dale et al. 2019** (substrate-independent framework for RC, R Soc A):
  evolved configurations outperform random structures across all tested
  physical substrates — supports the broader claim that hardware-aware
  structure beats random, but on PHYSICAL reservoirs, not silicon-bound
  digital RC.
- **Tanaka et al. 2019** (Neural Networks): comprehensive review;
  consistently argues physical non-idealities can be harnessed
  constructively.
- **Integer ESNs** (arXiv:1706.00280): replace recurrent matmul with cyclic
  permutation — directly relevant to our D2 "permutation is dominant binder"
  finding.

### Per-position / per-CU weight assignment (most directly relevant)
No literature found. This appears to be a genuine research gap on the
digital x86 GPU substrate side. Our hypotheses H1-H6 are essentially new on
this platform — but our F1 + F2 results (15/15 + 3/3 fail) suggest the gap
may exist because the experiments DON'T work, not because no one thought of
them.

### Variability exploited as regularization (the "for-free" angle)
- **Hochreiter & Schmidhuber 1997** (flat minima): noise-injected training
  finds flatter minima that generalize better. Implies envelope-noise during
  training COULD help — but only if envelope noise has right
  spectrum/amplitude.
- **Bengio et al. 2009**: synaptic noise as regularizer.
- **Gal & Ghahramani 2016** (Dropout as Bayesian approx): formalizes when
  noise injection helps.

The "envelope as natural dropout" hypothesis (H2) follows this lineage but
fails our F1 test by 4-17%. Likely because the envelope-derived dropout is
*fixed* (mask is one realization), not stochastic — so it doesn't act as
implicit Bayesian model averaging.

## Conclusion
Per-die structural specialization for digital-CMOS-bound neural networks is
**a genuine research gap**. Our negative result (F1: 0/15, F2: 0/3 C5 wins)
is consistent with the absence of literature and with NFL-theorem bounds:
envelope information that is statistically independent of the data
distribution cannot improve expected loss, only re-distribute it.

The clearest publishable angle from this scan: **"Binding without
adaptation"** is the natural information-theoretic regime for
hash-derived envelope structures on deterministic digital substrates. The
opposite regime ("adaptation") exists in physical-substrate RC where the
substrate dynamics ARE part of the computation.

## Web search trail (representative)
- WebSearch: "device-specific neural network optimization per-die fingerprint 2026"
- WebSearch: "hardware-tuned LoRA per-device personalized inference accuracy advantage 2025"
- WebSearch: "per-die OR per-chip neural network fine-tuning hardware fingerprint accuracy"
- WebSearch: "reservoir computing hardware-aware structural prior performance benefit substrate"
- WebSearch: "neural network energy efficiency chip-specific tuning process variation PUF accuracy 2024 2025"
- WebSearch: "silicon variability neural network training accuracy improvement exploit not mitigate"
- WebSearch: "echo state network permutation cycle reservoir NARMA performance benefit memory capacity"
