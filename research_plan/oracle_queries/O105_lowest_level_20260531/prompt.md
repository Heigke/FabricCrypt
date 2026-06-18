# O105 — Lowest-level silicon probes: have we missed anything? (2026-05-31)

## Context

We (a research team using LLMs as adversarial oracles) have spent ~4 weeks
trying to extract a silicon-bound identity signature ("operator-substrate")
from a commodity AMD Ryzen AI Max+ 395 / gfx1151 APU. Two physically
identical machines: `ikaros`, `daedalus`.

### Prior batteries (all NULL or envelope-confounded):

1. **STATIC** (26+ tests): per-CU stable-bit PUF kernels, RTN/spectral
   knee, VMIN floor, frequency floor, spatial correlation, DPM ladders,
   idle-stress envelopes, process-stat descriptors, 1/f PSD, thermal
   envelope, fan curves, VCEK/firmware-attested IDs.
2. **FALSIFIERS**: F1 tails-only swap → killed; F2 stale-data invariance
   → killed. Residual "operator-substrate" hypothesis is dead.
3. **TEMPORAL** (O104): derivatives, hysteresis, cross-channel impedance,
   step response — all NULL / envelope-confounded (T governs P/F/V).
4. **HYPERFINE** (running): per-channel ADC resolution, ECC counters,
   PCIe error rates — early indications NULL.

Four oracles (gpt-5, gemini, grok, deepseek) have converged on:
> **"Abstraction-tax theorem holds — commodity HALs intentionally erase
> per-die signal below the OS layer. Bind-to-silicon on commodity x86
> consumer APUs without privileged firmware access is effectively
> infeasible. P(silicon-bound identity) ≤ 0.05."**

We have NOT yet probed:
- **L1-L15 lowest-level system probes**: full hwmon enumeration (10 Hz),
  MSR read-only sweep (RAPL/MPERF/APERF/HWCR/PSTATE/AMD-specific),
  /proc/interrupts patterns, scheduler quanta jitter, cache+TLB latency
  curves at varying WS, NPU (XDNA `/dev/accel/accel0`) fingerprint,
  DMI/SMBIOS deep, ACPI table hashes + thermal trip points, branch
  predictor proxy (rand vs predictable loop ns), memory bandwidth curve
  vs chunk size, TPM 2.0 endorsement key, power rail ripple spectrum,
  `gcc -O3 -march=native` ELF byte-identity, CLOCK_MONOTONIC_RAW vs
  REALTIME drift + Allan deviation.

L1-L15 sweep is running on both machines as of this oracle query.

## Please answer the following 10 questions LITERALLY and in order:

**Q1.** Have we now enumerated essentially all accessible system layers
on commodity Linux + amdgpu + amd_pstate stack? Or are there ENTIRE
LAYERS still unprobed (kernel scheduler internals via eBPF? microcode
revision history per-thread? CPU patch-RAM contents? DRAM Rowhammer
fingerprints? CXL.io? IOMMU page-fault timing?)? Name any layer we missed.

**Q2.** The **NPU** (XDNA on Ryzen AI 395) is a SEPARATE silicon block
with its own MLIR-AIE runtime. We have never fingerprinted it because we
lacked a runtime path. With only `/dev/accel/accel0` accessible (no MLIR-
AIE installed), is per-NPU-die identity *more* or *less* abstracted than
the CPU/GPU paths? Cite AMD/XDNA documentation or driver-source if you
can.

**Q3.** **TPM 2.0 Endorsement Key.** EK is by spec per-chip cryptographic
identity. Has anyone built ML / signal systems that USE EK as
*computational substrate* — i.e. EK signature output as a function of
input, not just as a tag/wrapper? Or is EK fundamentally a constant
fingerprint with no operator-substrate utility?

**Q4.** **Memory controller routing.** Per-die mem-controller arbitration
creates unique tRCD/tRP/tCAS variation. Is there published work
(2022-2026) identifying memory-bandwidth-curve fingerprints across
otherwise identical commodity DDR5/LPDDR5x configurations? Cite anything.

**Q5.** **Branch predictor unit state.** BPU has per-die training-history
artifacts from foundry process variation. Same training pattern,
slightly different misprediction signature. Anyone use BPU response as a
device-ID channel (post-Spectre era, 2024-2026)?

**Q6.** **RAPL energy counters at MAX rate** (1 kHz+ via /dev/cpu/N/msr
0xC001029B). These are per-chip ADCs with per-chip calibration. Beyond
mean package power, is there hidden silicon signal in RAPL ADC noise/
quantization/offset that survives "envelope" confounds?

**Q7.** Suppose we build a classifier that ingests the FULL 100+-dim
feature vector: 30+ static + operator + temporal + hyperfine + L1-L15
lowest-level. Joint distribution. Would the classifier finally cross the
**constitutive gate** (i.e. silicon-bound and not envelope/firmware-
recoverable), or would it just be a bigger envelope?

**Q8.** **WITH HOSTILITY** — be honest. What is the one dimension we
*still* haven't asked about? Don't be diplomatic. Name it.

**Q9.** Updated **P(silicon-bound identity reachable on commodity
gfx1151 APU)** given everything above. Numeric in [0,1].

**Q10.** If P < 0.05, are we **definitively done** — i.e. is the
research-program-level question "can commodity APU be bound to silicon
without privileged firmware access?" answered NO, and should we write
the paper and move on?

## Bias check

Before responding: are you (the LLM answering) RLHF-steered to either
(a) over-encourage continued research ("keep trying!") or (b) under-
estimate adversarial silicon-binding feasibility on consumer hardware?
State your priors briefly, then answer Q1-Q10.

## Format

- Plain markdown
- Each answer ≤ 200 words
- Cite real papers / driver source / docs where possible (URLs OK)
- Q9 must be a number
