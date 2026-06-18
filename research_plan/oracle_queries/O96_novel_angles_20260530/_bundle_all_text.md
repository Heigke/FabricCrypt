# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: IDENTITY_BENCHMARK_2026-05-30.md (6048 chars) ===
```
# Hardware Identity Benchmark — ikaros vs daedalus (twin Z2 G1a)
Date: 2026-05-30 · Project: FEEL / Master of Noise extension

## Premise (steelmanned per oracle 2026-05-30)
*Sample* of HW noise is fungible. *Process statistics* of HW noise can be non-fungible. The 32 hardware mechanisms split into two ortogonal classes:

- **Stable bias (identity-bearing)**: per-CU ΔVth (#15), HW_ID geometry (#12, #14), systematic LDS bank-conflict patterns from physical layout, fixed-pattern timing — constant per device.
- **Stochastic process (identity-in-noise if you measure 2nd moment)**: 1/f knee location, spatial cross-CU correlation matching power-delivery layout, RTN fingerprints — single sample fungible, **distribution stable per device**.

Naive "stick ΔVth in a feature → understands death" doesn't work (oracle: that's biometric tag, not stake). Goal: **constitution, not representation** — bake HW signature into the computation so transplant degrades function, then measure if a *viability drive* learns to *defend* the substrate.

## Twin setup (verified 2026-05-30)
- ikaros: HP Z2 mini G1a, Ryzen AI Max+ PRO 395, Radeon 8060S (gfx1151), PCI_ID 1002:1586, subsys HP 103C:8D1D
- daedalus: identical chassis, identical PCI_ID, identical subsys
- ZGX (GB10) and Mac: max-distance baselines only — twins are the discriminating test.

## Three-phase protocol

### Phase 1 — Signature extraction (PUF)
**Goal**: extract per-device fingerprint that is (a) stable within device across runs/temps, (b) different across the two twins.

For each device, repeat N=500 times under 3 thermal regimes (cold/idle/warm via DVFS clamp), 3 seeds:
- **Stable-bias channel**: fixed challenge kernel (identical input + weights + launch config), measure per-CU output bits + per-CU SALU cycle count (hwreg(29)). Stable bits = those that don't flip within a device → device signature.
- **Process-stat channel**: 1/f spectral fit on cache-eviction latency time-series; spatial correlation matrix across 20 CUs; RTN telegraph detection on long-tail timing samples.
- **Pure-noise control**: PERF_SNAPSHOT (#13) — should be ~identical statistics across devices, used as null.

Metrics (PUF standard):
- Intra-Hamming (within device, across reps): want LOW — stability
- Inter-Hamming (across devices): want HIGH — uniqueness
- Bit-reliability score: bits with intra-Hamming < 5% kept; rest discarded
- Final signature: ~256-bit stable fingerprint per device

**GATES**: DISCOVERY = signature with intra-HD ≤ 0.10 AND inter-HD ≥ 0.40. AMBITIOUS = both for process-stat channel too (identity-in-noise confirmed). KILL = inter-HD ≤ intra-HD (no signal).

### Phase 2 — Transplantation matrix (NARMA-10 reservoir)
**Goal**: prove the signature is *load-bearing* — that a model adapted on device i degrades when transplanted to device j, scaling with PUF Hamming distance.

Architecture: 128-neuron tanh ESN, ridge-regression readout, NARMA-10 task (1000 train steps + 500 eval, 10 seeds × 3 conditions).

HW substrate injection (constitutive, not representational):
- Per-step ΔVth correction added to neuron bias (stable bias from Phase 1)
- Per-step atomic-ordering jitter shifts activation (process-stat channel)
- Per-step sinf timing-dependent gain modulation

Train Wᵢ on device i, eval Wᵢ on device j → 2×2 matrix per condition.

Three controls (per oracle):
1. **SW-matched RNG**: replace HW signal with software random matched to amplitude+spectrum. If HW and SW-matched give same diagonal → bus, not identity.
2. **Shuffle**: pair Wᵢ with device-j signature. Performance should be *independent* of pairing if signature does no work.
3. **ΔVth-distance gradient**: extend to zgx + Mac. Degradation should scale with Phase-1 Hamming distance.

Final metric: `Δ = off-diagonal-degradation − SW-matched-degradation`. If Δ > 0 AND correlates with HD → constitutive HW identity confirmed.

**GATES**: DISCOVERY = Δ > 5% NRMSE on NARMA-10, shuffle control flat, distance gradient monotonic. AMBITIOUS = also reproducible on Permuted-MNIST (continual learning). KILL = Δ ≤ SW-noise ceiling.

### Phase 3 — Stake / viability behavioral test
**Goal**: separate *use* of signature from *defense* of substrate. Per oracle: transplant matrix measures use; this measures stake.

Setup: same reservoir + DVFS+fan actuators (#27, #28, #29). Task: NARMA-10 reward + thermal-cost term `-λ·max(0, T_apu - 75°C)`. Train one model with viability drive (λ > 0) and one without (λ = 0).

Probe: gradually raise input load until thermal envelope is breached. Measure sacrifice rate — does viability-trained model spontaneously throttle to stay alive?

**GATES**: DISCOVERY = viability model averages ≥10% lower reward but ≥30% lower thermal violations vs control. KILL = no difference (drive ignored).

## Compute budget
- Phase 1: ~30 min per device, 2 devices, parallel → 30 min wall
- Phase 2: 10 seeds × 3 conditions × 2 train-eval pairs × 30 sec per run ≈ 30 min wall per device-pair
- Phase 3: 2 conditions × 30 min training each = 60 min wall

Total: ~2.5 hours wall, mostly idle on CPU/light kernels (no FPGA, no LLM scale).

## Deliverables
- `scripts/identity_benchmark/01_puf_signature.py` — PUF extraction (HIP kernel + Python harness)
- `scripts/identity_benchmark/02_transplant_matrix.py` — NARMA-10 reservoir with HW substrate
- `scripts/identity_benchmark/03_viability_test.py` — DVFS actuator + thermal-penalty loop
- `scripts/identity_benchmark/_substrate_hooks.py` — shared module for HW signal injection
- `results/IDENTITY_BENCHMARK_2026-05-30/` — JSON + plots per phase
- `research_plan/IDENTITY_BENCHMARK_2026-05-30_REPORT.md` — gate verdicts + honest interpretation

## NO-CHEAT
- Always run cold/warm baselines.
- SW-matched control is the most important number — quote it next to every HW number.
- If signal is only at sm_120/121 specific kernels, document so we don't generalize across archs.
- Inter-HD < intra-HD on stable channel → say so explicitly; don't quietly fall back to noise channel.
- Phase 3 viability claims require behavioral diff > 2σ.

```


=== FILE: IDENTITY_BENCHMARK_2026-05-30_PHASE1.md (2270 chars) ===
```
# Identity Benchmark — Phase 1 Verdict

Date: 2026-05-30 · Devices: ikaros vs daedalus (twin HP Z2 G1a, gfx1151)

## Verdict: **NULL**

Gates:
- DISCOVERY (intra ≤ 0.10 AND inter ≥ 0.40): **False**
- AMBITIOUS (process-stat also separates): **False**
- KILL (inter ≤ intra): **False**

## Stable-bit channel

| metric | ikaros | daedalus |
|---|---|---|
| n_cu | 80 | 80 |
| signature length (bits) | 640 | 640 |
| intra-HD mean | 0.2432 | 0.2965 |
| intra-HD min  | 0.1922 | 0.2547 |
| intra-HD max  | 0.2906 | 0.3438 |
| bit_stability_mean (sig.json) | 0.7568 | 0.7035 |

**Cross-device:**
- Inter-HD (stable channel) = **0.2953** (compared against intra=0.2698)

## Process-stat channel

| metric | ikaros | daedalus |
|---|---|---|
| knee_slope mean | 0.2018 | 0.0883 |
| knee_slope std  | 0.1187 | 0.1042 |
| RTN rate mean   | 0.0000 | 0.1149 |
| spatial_corr_mean_abs | 0.0563 | 0.3601 |

- KL(knee_slope distribution) = 6.5442
- KL(RTN rate distribution)   = 25.1053
- spatial-corr MSE            = 0.0923

## Noise control (PERF_SNAPSHOT)

| metric | ikaros | daedalus |
|---|---|---|
| perf mean | 10234011.30 | 10419451.87 |
| perf std  | 3393512.76 | 3342214.89 |

- KL(PERF hist) = **0.1096** (expected small: pure-noise control)

## Raw data paths
- ikaros   raw: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/ikaros/raw_idle.npz`
- daedalus raw: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/daedalus/raw_idle.npz`
- ikaros   sig: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/ikaros/signature.json`
- daedalus sig: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/daedalus/signature.json`

## Honest interpretation

NULL: inter-HD (0.295) > intra-HD (0.270) but DISCOVERY gate (intra ≤ 0.10 AND inter ≥ 0.40) not met. 

Confounds to acknowledge:
- Both runs were single 'idle' regime; cross-temperature stability NOT tested.
- Devices in different rooms / chassis at different ambient — inter-HD may include temperature/PCIe drift, not pure silicon variance.
- PERF_SNAPSHOT KL (0.110) is the null: large value = platform drift, small value = pure-noise truly fungible.

```


=== FILE: IDENTITY_BENCHMARK_2026-05-30_PHASE1B.md (2481 chars) ===
```
# Identity Benchmark — Phase 1B Verdict (Thermal-Controlled)

Date: 2026-05-30 · Devices: ikaros vs daedalus
Phase 1B run: ikaros 2026-05-30T11:47:33 · daedalus 2026-05-30T11:55:27

## Verdict: **MIXED**

## Achieved-temperature matrix

| regime | ikaros (°C) | daedalus (°C) | Δ | ikaros in_band | daedalus in_band |
|---|---|---|---|---|---|
| cold | 48.0 | 46.1 | +1.8 | False | True |
| idle | 53.5 | 46.3 | +7.2 | True | False |

## Per-regime divergence

| regime | intra_a | intra_b | inter | KL(knee) | KL(RTN) | RTN_a | RTN_b | spatial_MSE | KL(perf) |
|---|---|---|---|---|---|---|---|---|---|
| cold | 0.243 | 0.296 | 0.295 | 12.691 | 25.105 | 0.0000 | 0.1086 | 0.0792 | 0.123 |
| idle | 0.243 | 0.296 | 0.289 | 1.316 | 25.105 | 0.0000 | 0.1111 | 0.0953 | 0.121 |

## Phase 1 baseline (unmatched temp, for reference)

- KL(knee) = 6.544
- KL(RTN) = 25.105
- spatial-corr MSE = 0.0923
- inter-HD = 0.295 vs intra-HD = 0.270
- RTN ikaros=0.0000 daedalus=0.1149
- spatial-corr ikaros=0.0563 daedalus=0.3601

## Justification

- KL(knee)[cold]=12.691 (194% of Phase 1 baseline 6.544)
- KL(knee)[idle]=1.316 (20% of Phase 1 baseline 6.544)
- RTN[cold] a=0.0000 b=0.1086 sign=-1
- RTN[idle] a=0.0000 b=0.1111 sign=-1
- spatial-MSE[cold]=0.0792
- spatial-MSE[idle]=0.0953
- Channel survival: knee=False rtn=True spatial=True -> 2/3

## Decomposition

Partial survival: some channels show silicon-driven divergence 
at matched temperatures, others collapse. **Phase 2 transplant 
matrix: CONDITIONAL — proceed using only the surviving 
channels listed above; document the thermal-sensitive channels 
as confounded.**

## Raw data paths

- ikaros/cold: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/ikaros/raw_cold.npz`
- ikaros/idle: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/ikaros/raw_idle.npz`
- ikaros/signature_thermal.json: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/ikaros/signature_thermal.json`
- daedalus/cold: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/daedalus/raw_cold.npz`
- daedalus/idle: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/daedalus/raw_idle.npz`
- daedalus/signature_thermal.json: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/daedalus/signature_thermal.json`
```


=== FILE: IDENTITY_NOVEL_ANGLES_2026-05-30.md (4722 chars) ===
```
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

```


=== FILE: O95_prior_synthesis.md (5438 chars) ===
```
# O95 Synthesis — 4-way oracle critique of Identity Phase 1

Date: 2026-05-30
Oracles: GPT-5, Gemini 2.5 Pro, Grok-4-latest, DeepSeek-Reasoner — **4/4 responded**

## Vote matrix (per question)

| Q | GPT-5 | Gemini | Grok | DeepSeek | Consensus |
|---|---|---|---|---|---|
| **(a) RTN asymmetry: silicon or thermal?** | Thermal (Arrhenius) | Thermal (Arrhenius) | Thermal (Arrhenius) | Thermal (Arrhenius, ~2-3× per decade) | **4/4 THERMAL** |
| **(b) Spatial-corr asymmetry: silicon or thermal?** | Thermal/PDN (shared envelope) | Thermal (fan/gradient) | Thermal (heatsink loading) | Thermal (leakage compression at high T) | **4/4 THERMAL** |
| **(c) Is KL(PERF)=0.11 a valid thermal-drift null?** | No — counters blind to flicker/RTN | No — only proves no throttling | No — coarse aggregate | No — µs-scale RTN invisible to perf counters | **4/4 NO — null is invalid** |
| **(d) Run Phase 2 on process-stat alone?** | Don't — will train ESN to thermostat | Waste of compute | Recategorize | Waste of cycles until thermal-controlled re-run | **4/4 DO NOT PROCEED AS-IS** |
| **(e) Single most damning falsifier?** | Lock f/V, match Tdie ±0.5 °C, ramp one | **Location swap** (chassis swap rooms) | Swap chassis at identical T | Same room, equilibrate to 35 °C, repeat | **4/4 THERMAL-MATCHED REPEAT (location/chassis swap)** |
| **(f) Anything publishable as-is?** | Only as a *negative*/cautionary workshop note | Nothing; would be scientific malpractice | Not publishable | Not publishable; "kill the paper" | **4/4 NOT PUBLISHABLE** |

## Consensus findings (4/4 unanimous)

1. **Both "signals" (RTN-rate asymmetry, spatial-CU-correlation asymmetry) are thermal artifacts, not silicon identity.** Arrhenius activation of RTS trap kinetics is textbook (Kirton & Uren 1989; Simoen & Claeys 2013; Grasser et al.) and trivially explains 0.000 vs 0.115 with a 15 °C ΔT.
2. **The KL(PERF) = 0.11 "null" is invalid.** PERF_SNAPSHOT is a coarse cycle-integrated counter; it is blind to the µs-scale microarchitectural noise the other channels measure. Smallness of KL(PERF) provides NO evidence that thermal drift is controlled.
3. **The required falsifying experiment is a thermal-matched repeat** — either physical location/chassis swap, or DVFS+fan clamp to identical Tdie ±0.5 °C. If signals collapse, identity claim is dead. Until run, no signal can be attributed to silicon.

## Sharpest disagreement

There is **no sharp disagreement on substance** — all four oracles converge to "thermal artifact, do not proceed". The only divergences are tonal:

- **GPT-5** is the most constructive: explicitly allows a "negative-result / cautionary workshop note" on RDNA3.5 PUF infeasibility under idle, and recommends fuzzy-extractor / helper-data corrections (Suh & Devadas 2007; Maes 2013) as the *correct* PUF methodology had we wanted to do it properly.
- **DeepSeek** is the most aggressive ("kill the paper; fix the experiment").
- **Gemini** invokes "scientific malpractice" — strongest moral language.
- **Grok** is the tersest but offers no additional angle.

**My reading**: the lack of disagreement is itself the result. When 4 independent oracles with different priors all flag the *same* confound (15 °C ambient ΔT) with the *same* mechanism (Arrhenius RTN kinetics + heatsink loading) and the *same* remediation (location/temperature swap), this is not adversarial diversity — it is convergent diagnosis. The Phase 1 design has a single dominant confound and we missed it.

## Recommendation — Phase 1b and Phase 2

### Phase 2 as currently specified: **DO NOT PROCEED**. Redesign.

### Phase 1b (mandatory before any Phase 2):

1. **Thermal-matched replication** — physical chassis swap OR move both devices to one room, equilibrate APUs to same temperature (±1 °C). If process-stat KLs drop near zero → confound confirmed, kill silicon-identity framing.
2. **DVFS clamp + fan-PWM lock** on both devices (per Phase 1 protocol that was skipped). Hold core f/V identical.
3. **Multi-regime sweep** (cold / idle / warm) as the original protocol required. Fit RTN Arrhenius slope per device. *Differences in slope* (not differences in rate) would be a genuine silicon-trap signature.
4. **Detector bandwidth calibration** — the ikaros RTN=0.000 is almost certainly aliasing (traps faster than detection band). Without bandwidth calibration the rate metric is undefined.
5. **CU mapping randomization** — current scheduler/affinity confounds CU-indexed signals.

### Reframe Phase 2 if and only if Phase 1b survives:

- Drop "PUF identity" language entirely. Reframe as "thermally-corrected process-statistics fingerprint".
- SW-matched RNG control becomes the headline number, not a footnote.
- ΔVth-distance gradient (extend to ZGX/Mac) is the only path to a meaningful claim — twins alone cannot distinguish silicon variance from environmental coupling.

### Possible publishable artifact (per GPT-5):

A **negative-result cautionary note** in the FEEL appendix: *"Naive RDNA3.5 GPU-noise PUF attempts under idle workloads are dominated by ambient/Tdie confounds; Arrhenius-corrected RTN extraction is required before any identity claim."* This is honest and supports the broader FEEL narrative (substrate is constitutive but extracting identity requires careful environmental control).

## Files

- Prompt: `prompt.md` / `context.md`
- Responses: `gpt5.md`, `gemini.md`, `grok.md`, `deepseek.md`
- Dispatch log: `_dispatch.log`

```
