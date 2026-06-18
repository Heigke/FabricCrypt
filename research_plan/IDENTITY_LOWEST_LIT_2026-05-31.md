# IDENTITY — Lowest-Level Literature Review (2026-05-31)

Web-research track of investigation O105. Goal: any 2024-2026 commodity-x86
PUF / silicon-bound identity success we missed in our 30+ prior probes?

## A. Confirmed prior work (commodity hardware fingerprinting wins)

### A1. FP-Rowhammer (Centauri Lab, 2023; ACM ASIA-CCS 2025)
- arXiv: https://arxiv.org/abs/2307.00143
- ACM: https://dl.acm.org/doi/10.1145/3708821.3733880
- DRAM-based device fingerprinting via Rowhammer-induced bit-flip locations.
- **99.91% fingerprinting accuracy** across 98 DRAM modules.
- Stable over 10 days, <5 s extraction time.
- Survives OS reinstall, MAC/IP changes.
- **Caveat for our APU**: targets the DRAM module itself, not the APU die.
  On LPDDR5x soldered to motherboard (Strix Halo "AI Max+ 395" 128 GB
  unified memory configs), the module is per-board not per-APU-die — so
  it would distinguish ikaros vs daedalus but the signal lives in
  Micron/Samsung memory silicon, not gfx1151.
- DDR5 caveat (per gpt-5 / Mutlu group): on-die ECC and improved refresh
  reduces hammer-bit-flip yield, but FP-Rowhammer still works.
- **Predecessor**: FPHammer (arXiv:2201.07597, IEEE 2022 / IEEE Xplore
  2024) — same idea, slightly lower accuracy.

### A2. DrawnApart (NDSS 2022)
- arXiv: https://arxiv.org/abs/2201.09956
- HAL: https://inria.hal.science/hal-03526240/document
- GPU fingerprinting via WebGL vertex shader timing variance across
  execution units.
- **98% accuracy with compute shaders, 150 ms collection time**.
- Tracks devices 67% longer than browser fingerprinting alone.
- Validated on 2,500+ crowd-sourced devices.
- **Direct relevance to gfx1151**: This is the closest published success
  on commodity GPU silicon. Mechanism = per-CU process variation in
  switching speed, observed via timing. Our PUF kernel probes (z2090+,
  PUF_KERNEL_v2) attempted exactly this; results were envelope-
  confounded but we may not have used the *vertex-shader workload
  shape* DrawnApart relies on. **Worth one explicit replication on
  amdgpu Vulkan/WebGL paths before declaring done.**

### A3. Survey work on intrinsic PUFs in commodity devices
- Schaller et al., "Run-Time Accessible DRAM PUFs in Commodity Devices"
  (CHES 2016) — decay-based DRAM PUFs work without modification.
- PreLatPUF (Talukder, Ray) — DRAM latency variations for signature
  generation under extreme operating conditions.
- D-PUF — reconfigurable DRAM PUF for device auth + RNG.
- **Common pattern**: all DRAM-side, all require root + custom timing
  control, all suffer temperature/voltage sensitivity (matching the
  envelope-tax we keep hitting).

### A4. Branch predictor / TLB side-channels
- "TLBleed" (Gras et al., USENIX 2018) — TLB as side-channel, but
  exploited for *secret extraction*, not device-ID.
- "Branch Privilege Injection" CVE-2024-45332 (ETH Zürich, May 2025)
  — Spectre v2 hardware compromise; needs microcode patch.
- "Apple M1 BPU reverse engineering" (arXiv:2502.10719, Feb 2025) —
  reverse-engineered branch predictor for out-of-place Spectre.
- **No published BPU-as-device-ID work post-Spectre era** that we
  could locate. Mitigations (IBRS/IBPB/STIBP) explicitly destroy the
  per-die training history. (Confirmed by openai, gemini, grok
  oracles. Deepseek cites "BPUF" Spreitzer USENIX 2024 — could not
  verify, likely hallucinated.)

### A5. RAPL energy fingerprinting
- ICPE 2024 RAPL validation artifact (lukalt/icpe-2024-rapl-validation).
- MAD-EN (arXiv:2206.00101) — RAPL for *attack detection*, not device-ID.
- Patents on power-up-state device fingerprinting (US 8219857, 8880954)
  — temperature-profiled SRAM cell fingerprints, requires power cycle.
- **No confirmed peer-reviewed work** showing RAPL noise/quantization as
  a die-unique signal that survives envelope controls. Deepseek cites
  "Khan TIFS 2022" — could not verify, treat as unverified.
- gpt-5, gemini, grok consensus: RAPL is firmware-modeled, not raw ADC.

### A6. AMD microcode (zentool, 39C3 2025)
- google/security-research zentool — analyse + craft AMD Zen microcode.
- CVE-2024-56161 — sig-verify weakness allowed loading custom microcode
  on Zen 1-5 (patched Dec 2024, AGESA 1.2.0.3C).
- **Important**: zentool can READ patch metadata, but the per-thread
  patch-RAM contents themselves are PSP-gated. No per-die variation
  exposed via current microcode rev reads.

## B. XDNA / NPU specifics

- AMD XDNA driver (`drivers/accel/amd/xdna`) exposes high-level command
  queue, **no per-die calibration / EFUSE / serial** in kernel docs.
- WebFetch of https://docs.kernel.org/accel/amdxdna/amdnpu.html confirms:
  no per-instance variation data exposed.
- Userspace path: MLIR-AIE runtime → firmware blob → tile array. Per-NPU
  identity is *more* abstracted than CPU/GPU (all 4 oracles agree).
- Our probe (L6) confirmed: `/dev/accel/accel0` exists, openable RO, but
  no sysfs telemetry beyond driver presence.
- **No public reverse-engineering work on per-die XDNA fingerprinting.**

## C. Strix Halo / gfx1151 specifics
- TechInsights "Strix Halo Advanced Packaging Quick Look" — chiplet
  topology but no PUF mention.
- llm-tracker.info/_TOORG/Strix-Halo — community wiki, no fingerprint
  work.
- AMD AI Max+ 395 launched Jan 2025; no published PUF/fingerprint
  research targeting it as of search date.

## D. New 2024-2026 attack vectors we did NOT cover

1. **Transient Scheduler Attacks (TSA)** — Microsoft + AMD, July 2025.
   Targets Zen 3/4 scheduler timing leaks. Could it expose per-die
   scheduler-queue topology? Not benchmarked for device-ID use.
2. **AVXProbe** (ACM ASIA-CCS 2025) — AVX-masked operation timing on
   TLB/cache. Designed for website fingerprinting, but mechanism is a
   cross-silicon timing channel — *might* reveal per-die TLB topology.
3. **Branch Privilege Injection** (CVE-2024-45332) — Intel-specific,
   not applicable to AMD Zen.
4. **GPU Confidential Computing Demystified** (arXiv:2507.02770) —
   NVIDIA H100/H200 device-key-fuse identity. AMD MI-series has VCEK
   (we already tested), but consumer gfx1151 has no equivalent public
   attestation path.

## E. Adjacent physical methods (require external HW, out of scope)

- EM emanations (gemini Q1) — needs Faraday cage + SDR.
- Voltage glitching / fault injection (gemini Q8) — would brick chip.
- External power-rail probing (gemini Q1) — needs scope + shunt.
- These are the canonical bypass for the abstraction-tax, but they all
  break the "commodity software-only" constraint.

## F. Verdict from literature

- **Two real wins for commodity hardware fingerprinting exist** (A1
  FP-Rowhammer 99.91%, A2 DrawnApart 98%) but both fingerprint the
  *peripheral* (DRAM module / GPU CU array) rather than the APU die
  per se.
- **No published 2024-2026 work** binds a commodity x86 APU to its
  silicon without privileged firmware access. Closest is DrawnApart,
  which is GPU-CU-array-level (and per-board, not per-die).
- The deepseek-cited papers (Wagner 2025, Khan TIFS 2022, Spreitzer
  USENIX 2024) could not be independently verified and look like
  plausible-sounding hallucinations — flagged but not used as evidence.
- Literature corroborates oracle consensus: **abstraction-tax theorem
  holds for commodity x86 APUs**.

## G. One surviving lead worth a dedicated test (Phase L+)

**Replicate DrawnApart on amdgpu / gfx1151** with the exact vertex-
shader workload shape from the original paper. Our PUF kernel probes
(z2090, puf_kernel_v2) used HIP atomics rather than vertex shader
pipelines — DrawnApart's signal lives in vertex-shader dispatch
scheduling, not compute kernels. If this also collapses, we're done.
Otherwise we have the one commodity-GPU positive result on amdgpu.
