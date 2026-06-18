# Oracle packet O108 — making a die-specific ANALOG COMPUTATION reproducible enough to be a fingerprint

You are an adversarial reviewer + measurement-physics expert. Police BOTH biases (wanting to succeed; giving up
too early). Cite ONLY real work (papers, standards, datasheets). Rank every suggestion by expected gain in
intra-die reproducibility / inter-die separability AND by probability of working on commodity hardware. Give a
concrete, thermally-SAFE protocol for anything you rate worth doing. HARD THERMAL LIMIT: 99°C ACPI trip = instant
reboot; low-duty / sharp-edge / demodulated only; we hold ~60-72°C.

## The system
Goal: make a frozen LLM constitutively dependent on ONE specific AMD Strix Halo gfx1151 APU die. Triad:
(1) UNIQUE [SOLVED: CPPC ranking 75% distinct + dynamics 14×], (2) RÄKNA = the die performs a real nonlinear
computation the model needs, (3) FRESH [SOLVED: RDSEED].

RÄKNA status:
- GENERIC compute SOLVED & strong on 2 dies: the die physically computes the PRODUCT u·v via shared-PDN power
  contention (GPU u-bursts × CPU v-bursts droop the rail MORE than the sum; the excess = u·v). A LINEAR readout of
  on-die telemetry does XOR(u,v)=0.75 (= product ceiling 0.746), 300-shuffle null p=0.000, u-only=chance,
  u&v-LINEAR=chance → genuine analog multiplication. Confirmed on both dies.
- DIE-SPECIFIC compute (räkna-UNIKT) = NOT achieved. Three routes tried, all failed on REPRODUCIBILITY:
  - coefficient route: u·v coefficient is MORE alike across 2 dies than across 2 runs of 1 die (operating point
    dominates) → FALSIFIED.
  - spatial route (CPU bursts pinned to cores 0,3,6,9 = spatial zones; temp-compensated ratio coupling matrix):
    weak tendency (same-die a bit more self-similar, mean gap +0.19) but NO clean separation; same-die intra cosine
    only 0.51-0.76, distributions overlap.
  - lock-in route (drive u@f1, v@f2; I/Q demod at intermod f1±f2 per zone/channel; 8 zones×3 tone-pairs): RAW
    same-die intra cosine ≈ 0.00 (orthogonal!) — the complex feature does NOT reproduce run-to-run. Causes we
    identified: (a) normalization baseline (per-run median/MAD of telemetry) swung 5 ORDERS of magnitude between
    runs; (b) die temperature DRIFTED 57→71°C WITHIN a single run; (c) DVFS/governor not pinned.

## Telemetry / sampling reality (important — re-examine our assumptions)
- 10 on-die channels via /dev/mem MMCFG SMN reads + energy/fast counters. We sampled at a 500 Hz control loop —
  but a prior probe hit ~450 kHz raw MMCFG read rate. So the 500 Hz was OUR CHOICE, not the hardware limit.
- The richest die-specific PDN structure (impedance poles/zeros, VRM loop, decoupling-cap resonances) is at
  kHz-MHz, above 500 Hz but possibly within reach of burst-mode high-rate reads and/or equivalent-time sampling.

## Questions — be specific and quantitative
1. REPRODUCIBILITY is the whole battle. Rank the highest-leverage fixes to get intra-die run-to-run cosine from
   ~0 (lock-in) / ~0.6 (spatial) up toward >0.9, WITHOUT inflating inter-die similarity:
   (a) SHARED/FIXED normalization baseline (calibrate median/MAD once per die, reuse across runs) — does this
       legitimately help or does it just hide drift?
   (b) TRUE temperature lock: closed-loop hold the die at a fixed setpoint (PID on a background heater-load) during
       the entire measurement, vs our soak-then-drift. How tight a setpoint (±?°C) is needed for PDN-feature
       stability? Any standard (JESD?) for fingerprint enrollment temperature control?
   (c) DVFS/clock/governor PINNING (we know Hot Pixels USENIX'23: telemetry is workload+frequency dominated).
       Fix CPU/GPU P-states, disable boost, pin governors — expected effect on reproducibility?
   (d) COHERENT AVERAGING / true lock-in: drive a repeating epoch, average the COMPLEX demod over N epochs at a
       FIXED operating point. How many epochs N to beat our noise? Is a real lock-in time-constant/Q the missing
       ingredient vs our one-shot demod?
2. SUPERSAMPLING / beating the sample rate for a REPEATABLE drive:
   (a) EQUIVALENT-TIME SAMPLING (sampling-scope trick): drive a periodic pattern, sample at 500 Hz but step the
       phase each period to reconstruct an effective kHz-MHz waveform. Viable for a software-timed (jittery) ADC
       like ours? How to handle timing jitter (is our sample clock stable enough)? Real references.
   (b) BANDPASS / undersampling: deliberately alias a high-freq PDN excitation down to a readable band — feasible
       to probe a known resonance via its alias?
   (c) Burst-mode 450 kHz capture in short thermally-safe windows — what bandwidth/SNR does that realistically buy,
       and does it expose die-specific structure the 500 Hz loop can't?
3. Better DISCRIMINATIVE features once reproducible: transfer-function / impedance estimate Z(f) from the
   two-tone response (system ID — Pintelon & Schoukens), pole-zero extraction, cepstrum, etc. Which feature is most
   die-specific AND most temperature-robust?
4. GRID SEARCH design: across {temp setpoint, tone freqs, drive amplitude, core set, normalization, N epochs},
   what is the smartest search (not brute force) to MAXIMIZE intra-reproducibility first, then inter-separation?
   Beware: with only 2 dies, feature/parameter selection can overfit/cheat — how to grid-search honestly (nested
   CV? hold-out? what does "honest" even mean at N=2)?
5. ZOOM OUT — did we fall into a rabbit hole? Is "die-specific analog COMPUTATION" the right target at all, or
   should die-identity and die-computation be SEPARATE channels (which already work) fused at the LLM level? Is
   there a fundamentally different, more reproducible die-specific-compute mechanism we MISSED (e.g. ring-oscillator
   frequency ratios as a computed PUF, SRAM-startup-conditioned compute, thermal-time-constant fingerprint as a
   computational kernel, memory-access-latency PUF feeding the reservoir)? Real prior art.
6. The N=2 ceiling: with 2 physical dies, what is the MOST convincing reproducibility+separability experiment we
   can run, and exactly what claim would it (and would it NOT) support? Pre-registered acceptance criteria.
7. LLM integration in parallel: we want to wire the WORKING generic u·v compute into a frozen LLM as a constitutive
   dependency NOW (test full pipeline) while fixing räkna-unikt. Best minimal architecture so the model genuinely
   NEEDS the live u·v signal (breaks on replay/different die) without wrecking text quality? What test proves the
   dependency is real (not cosmetic)?
