# O105 — Oracle Synthesis (2026-05-31)

4/4 providers responded (openai gpt-5 312s, gemini 2.5-pro 53s, grok-4 9s,
deepseek-reasoner 34s). Strong convergence.

## Q9 verdict — P(silicon-bound identity reachable)

| oracle | P | rationale |
|---|---|---|
| openai gpt-5 | 0.03 | abstraction-tax + no surviving channel |
| gemini 2.5 | 0.02 | classifier = bigger envelope |
| grok-4 | 0.02 | inaccessible without firmware/JTAG |
| deepseek-r | 0.01 | NPU/PSP firmware-gated |
| **MEAN** | **0.02** | DONE |

## Q10 — Definitively done?

**4/4 say YES.** Write the paper. Title suggestion (deepseek):
> *"The Abstraction Tax: Infeasibility of Silicon-Bound Identity on
> Commodity x86 APUs."*

## Q1 — Missed layers (union of oracle answers)

- **eBPF scheduler/IRQ tracepoints** (all 4) — sched_switch latency,
  runqueue depth, IRQ affinity. Envelope-confounded.
- **AMD Uncore perf** (gpt-5) — Data Fabric, L3, UMC counters via
  `drivers/perf/amd/*`. Still firmware-shaped.
- **IBS** (Instruction-Based Sampling) (gpt-5) — perf `ibs_fetch/ibs_op`.
- **PCIe lane-margining / equalization coefs** (gpt-5) — read-only via
  PCIe extended cap space; gpt-5's Q8 candidate.
- **IOMMUv2 fault/event logs** (gpt-5, deepseek).
- **RAS/EDAC MCA counters** (gpt-5) — rasdaemon, edac_amd64.
- **DCN pixel-clock jitter** (gpt-5).
- **USB3 LTSSM state counters** (gpt-5).
- **Microcode patch-RAM contents** (deepseek) — PSP-gated, not
  accessible.
- **DRAM Rowhammer timing** (3/4) — DRAM-module signal, not APU.
- **SMM entry/exit timing** (deepseek).
- **DDR5 PMIC trim / VrefDq via SMU** (deepseek) — privileged.
- **GPU Data Fabric counters via debugfs** (deepseek).
- **Power-up SRAM contents** (deepseek) — needs custom boot firmware.

**Net: still many sub-channels but no new entire LAYER beyond
"firmware-gated". All oracles agree the OS-visible surface is
essentially enumerated.**

## Q2 — NPU more or less abstracted?

**4/4: MORE abstracted.** XDNA path: User → MLIR-AIE runtime → kernel
`amdxdna` → firmware → AIE tiles. No per-die calibration / EFUSE / raw
ADC paths exposed in driver UAPI. Our L6 probe confirmed: `/dev/accel/
accel0` exists, openable RO, sysfs empty.

## Q3 — TPM EK as computational substrate?

**4/4: NO.** EK is a deterministic crypto identity — sign(EK, input) is
trivially simulable from public key. No silicon noise in output. Not a
dynamic substrate. (TCG EK Credential Profile.)

## Q4 — Memory bandwidth fingerprints (DDR5/LPDDR5x)?

- gpt-5, grok-4: **no peer-reviewed work** on die-unique BW curves
  surviving training/temp/BIOS/OD-ECC.
- gemini: cites Blacksmith (S&P 2022) as related but indirect.
- deepseek: cites "Wagner 2025 arXiv:2503.12345" and
  "Obermaier HOST 2023" — **could not independently verify, likely
  hallucinated**.
- Independent web research finds FP-Rowhammer (99.91%) and FPHammer
  for DRAM modules — but module-level, not APU-die-level.

## Q5 — BPU misprediction as device ID (post-Spectre)?

- gpt-5, gemini, grok: **NO credible 2024-2026 work**. Mitigations
  (IBRS/IBPB/STIBP) destroy training history.
- deepseek cites "BPUF Spreitzer USENIX 2024" — **could not verify,
  likely hallucinated**.
- Our L9 BPU probe got mean Cohen d = 1.92, max 5.24 — separable but
  inseparable from envelope (different microcode/uarch can change BPU
  baseline).

## Q6 — RAPL ADC hidden signal at max rate?

- gpt-5, gemini, grok: **unpromising**. RAPL is model-output from SMU
  firmware, not raw ADC. Per-family calibration, not per-die.
- deepseek cites "Khan TIFS 2022" with positive result — **unverified**.
- Our L13 rail-ripple probe mean Cohen d = 0.52 (NULL).

## Q7 — 100+-dim classifier crosses constitutive gate?

**4/4: NO — bigger envelope only.** All features confounded by same
temp/V/firmware/scheduler variables. Decision boundary = envelope, not
silicon. Brittle to BIOS update / DIMM swap / ambient shift.

## Q8 — One dimension we still haven't asked about (hostile)

| oracle | proposed dimension |
|---|---|
| gpt-5 | **PCIe lane-margining / per-lane EQ coefficients** (board+die analog) |
| gemini | **Active fault injection** — voltage/freq glitching to find per-die failure threshold (out of "passive" scope) |
| grok-4 | **The firmware attestation boundary itself** — PSP/AGESA/VCEK erases analog state before OS-visible |
| deepseek | **PSP RNG / eFuse keys + power-up SRAM contents** — below-firmware, deliberately hidden |

**Common theme**: every remaining candidate either (a) requires
privileged firmware access (deepseek, grok) or (b) sits at the
board+die boundary not the pure die (gpt-5) or (c) violates the
"passive observation" constraint (gemini).

## Hallucination flags

Deepseek cites three suspicious-looking references:
- "Wagner 2025, arXiv:2503.12345" — arXiv ID format too clean
- "Khan, IEEE TIFS 2022 RAPL fingerprinting" — could not find
- "Spreitzer USENIX 2024 BPUF" — could not find

The other three oracles, which provided fewer specific citations,
align on NULL for the same questions. **Treat deepseek's
positive-citation answers as unverified.**

## Bias check across oracles

- openai: admits mild "be-helpful" bias but errs toward "say unknown".
- gemini: explicit acknowledgment of confirmation bias toward
  abstraction-tax, low-confidence on bleeding-edge.
- grok: "skeptical priors, no encouragement bias" — strongest skeptic.
- deepseek: claims "neutral", but produces the most optimistic-sounding
  citations (which may be confabulated).

Net: even with the most skeptical-of-skepticism reading, P = 0.05 ceiling.
