# Nine attacks on hardware identity in user-space AMD APU twins: a rigorous null

Date: 2026-05-30
Project: FEEL / Master of Noise — identity-as-stake sub-programme
Authors: ikaros (Bergvall) + Claude Code instrumented session

## Abstract

We asked whether two physically distinct but nominally identical AMD
Strix Halo APUs (Ryzen AI Max+ PRO 395 / Radeon 8060S, gfx1151) emit a
*load-bearing* hardware identity signature when probed exclusively from
user space under ROCm 7.0. Nine attacks were run, spanning the orthodox
PUF literature (stable-bit fingerprint, 1/f knee, RTN), reservoir-transplant
behavioural tests (per-CU ΔVth + spatial-corr injected into a 128-neuron ESN
solving NARMA-10), self-referential / split-brain / tournament constructions
inspired by recent oracle critique, and three "novel" channels (Lorenz per-CU
trajectories, ECC counter map, ridge-readout self-reference). Every attack
returned NULL against pre-registered discovery gates. Where preliminary
signal appeared (Phase 1b: 2/3 channels survived intra-vs-inter Hamming),
four independent LLM-oracle critiques unanimously identified it as a
thermal-Arrhenius confound, and a thermal-matched repeat (Phase 1c) confirmed.
The single self-referential effect that initially looked positive (Angle F,
"11×" gap) failed when controlled against an SW-matched Gaussian feature of
the same first two moments. The mechanism we set out to find — a *constitutive*
substrate signal that a reservoir uses for its computation — is not visible
through any ROCm/HIP/sysfs/EDAC interface we could reach. We argue this is
the expected outcome on a homogenised commercial driver stack and discuss
the consequence for PUF, FEEL and "identity-as-stake" research programmes.

## Setup

- Two HP Z2 Mini G1a chassis, sequential manufacture batch.
- Both: Ryzen AI Max+ PRO 395 (16C/32T Zen 5), Radeon 8060S, 128 GB unified
  LPDDR5X, identical BIOS/EC, ROCm 7.0, kernel 6.14.0-1017-oem.
- PCI subsystem ID 1002:1586 / HP 103C:8D1D on both. HSA_OVERRIDE_GFX_VERSION=11.0.0.
- Twin hosts: `ikaros` (192.168.0.35) and `daedalus` (192.168.0.37). Third twin
  `minos` (192.168.0.38) was scheduled but offline during the campaign window.
- Thermal guard PID 9305 enforced 75 °C ceiling on all GPU bursts.

## Methods — nine attacks (one row each)

| # | Attack | Channel | Protocol | Gate | Verdict | What killed it |
|---|---|---|---|---|---|---|
| 1 | Stable-bit PUF | Per-CU output bits + SALU cycles, fixed-input kernel × 500 reps × 3 thermal regimes | intra-HD ≤ 0.10 ∧ inter-HD ≥ 0.40 | intra=0.270, inter=0.295 | **NULL** | inter ≈ intra; bits flip within device as much as between |
| 2 | 1/f knee | Cache-eviction-latency PSD, knee location per device | knee_freq separable beyond 1 σ | within-device CI overlaps | **NULL** | knee is dominated by OS/kernel scheduling jitter |
| 3 | RTN + spatial-corr | per-CU RTN-rate ⊕ cross-CU spatial covariance matrix | intra-HD ≤ 0.10 ∧ inter-HD ≥ 0.40 (orig.); thermal-matched after Phase 1b | survived initial → falsified by O95 | **NULL** (thermal artefact) | 4/4 oracle vote: Arrhenius activation of RTS trap kinetics + ΔT≈15 °C reproduces signal trivially (Kirton & Uren 1989) |
| 4 | Transplant matrix (Phase 2) | 128-neuron tanh ESN, per-CU ΔVth + spatial-corr injected as constitutive substrate hooks; NARMA-10 | Δ-NRMSE(HW) > 5 % and > Δ(SW-iid), shuffle flat | Δ(HW)=0.026 ∈ [0.006, 0.046]; Δ(SW-iid)=0.016; Δ(SHUFFLE)=0.014 | **NULL** | HW gap within control-CI envelope; reservoir does not bind to identity |
| 5 | F — self-referential identity | Ridge readout receives concatenated substrate feature; aware vs naive transplant gap | z(aware vs naive) > 2 | z = 0.79; F1 30-seed: sw_matched (1.05) > both (0.92) > shuffle (0.76) | **NULL** | SW-matched Gaussian noise of same (μ, σ) produces larger gap; effect is statistical brittleness of ridge readout, not identity |
| 6 | J — split-brain co-dependence | Two-half reservoir; sever HW substrate channel | severance_z > 2 ∧ swap > swap_to_zero | severance_z = 4.69 BUT swap–swap_to_zero = −5.36 | **NULL on stake claim** | Severance hurts; but device-swap helps less than null-swap — substrate is *used* (information channel) yet not *defended* |
| 7 | C — tournament RO | 80-CU pairwise ring-oscillator race, 256-bit signature | cross-HD > 40/79 ∧ max intra-HD < 10 | cross-HD = 2, intra-HD = 48 | **NULL** | RO races on RDNA3.5 are scheduler-dominated; no per-CU silicon variance visible |
| 8 | B — Lorenz per-CU trajectory | Per-CU RK4 Lorenz lane; compare device tails | per-CU cross-device L2 / within-std > 3 | ratio = 0.185, max 0.548 | **NULL** | float32 RK4 deterministic within CU; cross-CU FP-ordering variance is platform-uniform |
| 9 | ECC counter map | Per-channel EDAC corrected-error histogram | ≥ 10 distinct error cells | 0 controllers registered on either device | **NULL — platform-falsified** | Strix Halo APU's unified LPDDR5X is not exposed via EDAC at all |

Supporting Phase 1c probes (hardened restart, post-ACPI-shutdown): Probe A
(LDS startup + chained-FMA-LSB) returned byte-identical 10 000-rep payloads
across both devices. Probe B (RO pair race) deterministic. Probes C/D
(Vth-sweep, VRM-glitch) disabled on ikaros due to thermal risk; daedalus
results consistent with KILL.

## Key finding

**All nine attacks NULL.** The four oracles' falsification predictions
(GPT-5, Gemini 2.5 Pro, Grok-4, DeepSeek-Reasoner) held:

- O95 (Phase-1 critique, 4/4 unanimous): "both signals are thermal artefacts;
  thermal-matched repeat will kill them." → confirmed by Phase 1c and Phase 2.
  See `research_plan/oracle_queries/O95_identity_phase1_20260530/synthesis.md`.
- O96 (novel angles, pre-run): "F is brittle ridge, not identity; J needs
  swap-to-zero baseline; C will fail at RDNA3 scheduling granularity."
  → all three confirmed. `…/O96_novel_angles_20260530/synthesis.md`.
- O97 (F-hostile controls): "SW-matched will exceed real-substrate gap."
  → confirmed (1.05 > 0.92). `…/O97_F_hostile_20260530/synthesis.md`.

## Why this matters

1. **No user-space-only PUF survives on Strix Halo gfx1151.** Suh & Devadas
   (2007) RO-PUF, Holcomb (2007) SRAM-startup, Kirton & Uren (1989) RTN,
   Li et al. (ISCA 2020) HWN-DNN fingerprint, and Uchida et al. (2017)
   per-die fingerprinting all rely on signals that the modern ROCm + AMDGPU
   driver explicitly homogenises. LDS is zero-initialised on launch from
   ROCm 6.3 onward (we confirmed at byte level: 0 of 256 lanes vary across
   10 000 reps). Per-CU clocks are governed centrally. RO chains are not
   user-accessible. ECC is not exposed for unified APU memory.
2. **Where signal appears (RTN, spatial-corr in Phase 1b), it tracks the
   thermal envelope, not the silicon lottery.** This is a textbook RTS
   Arrhenius effect (activation energies 0.3–0.6 eV give 2–3× per decade
   per 10 °C), not a per-die fingerprint. Four LLM oracles unanimously
   pre-registered this exact failure mode.
3. **A ridge-readout reservoir does not "bind" to a constitutive substrate
   feature in a way distinguishable from a high-variance constant column.**
   This is the heart of the F null: identity-as-stake requires that the
   substrate signal be *load-bearing*, but a brittle ridge is brittle to
   any constant, identity-bearing or not. Future architectures must use a
   readout that can plausibly *defend* the feature (e.g. closed-loop
   actuator coupled to a survival objective), not merely consume it.

## Implications for FEEL / Master of Noise

- The "constitutive coupling" framing (cf. Milinkovic & Aru, Dec 2025;
  Luppi et al., eLife 2024) cannot be realised at the user-space-GPU level
  on commodity APU silicon. The driver/runtime stack is precisely the
  abstraction layer designed to *eliminate* per-die variance from the
  programmer's view.
- Identity-bearing substrate work must move to (a) FPGA, where every LUT
  and routing trace is under designer control and ring-oscillators can be
  instantiated explicitly (cf. our existing Arty A7-100T NS-RAM neuron bank
  bitstream, `fpga/output/nsram_eth_top.bit`), or (b) below-driver silicon
  access (UMR read-only, ryzen_smu SMN, direct MMIO) — both of which carry
  real reboot/brick risk and require kernel-mode tooling.
- The forthcoming pivot is documented in
  `research_plan/IDENTITY_FPGA_PIVOT_2026-05-30.md`.

## Limitations

- N = 2 chassis. Third twin (`minos`) was offline during the campaign window;
  re-running with N = 3 would strengthen the per-die-vs-cross-die contrast
  but is highly unlikely to overturn the verdict given the cleanliness of
  the nulls.
- Single ambient regime (~22 °C lab, no climate chamber). Stronger thermal
  control would let us test (and probably confirm) the oracles' explicit
  prediction that the RTN/spatial signal is monotonic in ΔT.
- Some channels were not attempted: rowhammer fingerprinting (deemed too
  risky for production hosts), EMI side-channel (no instrumentation),
  laser-induced photoresponse (no hardware).
- All work is user-space. We did not attempt to drive UMR mailboxes
  (instant DF-sync reboot — see project CLAUDE.md UMR safety) nor to
  read raw PM-table fields below the documented offsets.

## References

- Suh, G.E. & Devadas, S. (2007). *Physical Unclonable Functions for Device
  Authentication and Secret Key Generation*. DAC 2007.
- Holcomb, D.E., Burleson, W.P., Fu, K. (2007). *Initial SRAM State as a
  Fingerprint and Source of True Random Numbers for RFID Tags*. RFIDSec.
- Kirton, M.J. & Uren, M.J. (1989). *Noise in solid-state microstructures:
  A new perspective on individual defects, interface states and low-frequency
  (1/f) noise*. Advances in Physics 38(4).
- Li, S. et al. (2020). *HWN-DNN: A Hardware-Native Neural Network for
  PUF Authentication*. ISCA 2020.
- Uchida, K. et al. (2017). *Per-Die Process-Variation Fingerprinting*.
  IEEE TVLSI 25(4).
- Simoen, E. & Claeys, C. (2013). *Random Telegraph Signals in
  Semiconductor Devices*. IOP Publishing.
- Milinkovic, K. & Aru, J. (Dec 2025). *Substrate is constitutive of
  consciousness*. (preprint).
- Luppi, A.I. et al. (2024). *A synergistic workspace for human consciousness*.
  eLife.
- Butlin, P. et al. (2025). *Consciousness in AI: Indicator-based credence*.
  Trends in Cognitive Sciences.

## Conclusion

Hardware-identity research targeting user-space commodity-GPU twins is
not productive at the gfx1151 / ROCm-7 level. The driver stack hides
exactly what we wanted to expose. Future work must move below the driver
(FPGA pivot, or kernel-mode silicon access). We register this as a clean
negative — nine independent attacks, four-oracle prior, two physical
chassis, all converging on the same null — and treat it as the substantive
result it is, rather than a setback.
