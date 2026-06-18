# O100 — Constitutive HW-Identity Literature Hunt (2026-05-30)

## Context

Twin AMD Ryzen AI Max+ 395 / gfx1151 chassis (ikaros, daedalus). We have confirmed **3 silicon-bound channels** with Cohen d ≥ 8:
- DC power draw distribution (per-device offset + spectral shape)
- Thermal time-constant under standardised workload
- Per-core latency fingerprint (timing PUF, ~256 stable bits)

Two-device classifier reaches **100 % accuracy** — identification is solved. **Identity-bearing computation is not.**

Over the past two weeks we ran **12+ "constitutive identity" attacks**:
1. Inject ΔVth correction into reservoir bias.
2. Substrate-derived stream multiplied into tanh argument.
3. Per-step jitter shifts activation gain.
4. Time-stamp noise as recurrent input.
5. Stream as gating signal.
6. Stream as additive on weights.
7. Stream as bias floor.
8. Stream as readout perturbation.
9. Mixed phase/amplitude injection.
10. Two-stream (power+timing) cross-product.
11. Stream as Lyapunov regulariser term.
12. Stream as outer-loop hyper-parameter.

Every regime collapses under the **SHUFFLE control**: replacing device-`i` stream with device-`j` stream while keeping its statistics gives the same NRMSE delta. The model is **structure-bound, not device-bound**. Digital abstraction passes through.

We want methods that **actually work** — where transplanting a trained model to another (nominally identical) device measurably degrades it, AND where that device-binding is constitutively useful (not just a watermark).

## Eight questions (answer in order)

1. Cite **ONE concrete paper** that demonstrated constitutive (transplant-breaking) HW identity in a *learnable* model on *commodity hardware* (NOT FPGA, NOT memristor, NOT photonic). Just stock CPU/GPU/APU.
2. If no such paper exists for commodity hardware, what is the **theoretical reason**? Information-theoretic (channel capacity), computational (universal approximation), or empirical (driver/runtime layer washes out)?
3. The user wants computation to **benefit from** identity, not just depend on it. What does "benefit" mean operationally? Energy efficiency? Adversarial robustness? Defection-detection? Pick the most well-motivated benefit and propose a falsifiable demonstration.
4. What is the **simplest existing system** where transplant-degradation is real and quantified? Should we port that methodology?
5. Is there a hybrid where **software makes the digital abstraction less perfect** — e.g. deliberately running at thermal-throttle threshold so silicon variance leaks through — that we missed?
6. If we accept user-space gfx1151 cannot do constitutive identity, what is the **cheapest physical hardware addition** (a sensor, a co-processor, a sub-$100 device) that would enable it?
7. Compare to **FPGA route** — is the gap really 100× more dev effort, or is there a Pareto-optimal middle ground (e.g. a small ADC over USB)?
8. **Brutal honesty**: are we asking a question that's been definitively answered "no" by 20 years of computer architecture research? Are we just rediscovering the abstraction-tax theorem?

## Constraints
- Cite real papers (arxiv / venue + year). No hallucinated citations.
- Be specific: "X et al. 2021 Nature Electronics, transplant-degradation = 42 % accuracy drop" — not "memristor work has shown ...".
- ≤ 1500 words total.
