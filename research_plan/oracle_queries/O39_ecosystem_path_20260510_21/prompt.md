# NS-RAM ecosystem-fit + V_G2-continuum bridge: paths forward without new silicon data

You are reviewing the entire NS-RAM differentiable-simulator project and being
asked to find the highest-leverage research moves we can make WITHOUT
requesting new silicon characterisation from the KAUST collaborators
(Pazos / Lanza). Use the three audit summaries below as your context. We are
deliberately trying to find paths forward we can take unilaterally.

## Audit summary

### What pyport CAN do today
- Single 2T-cell BSIM4 + parasitic Gummel–Poon NPN, fully torch-autograd,
  CPU float64.
- Three DC solvers (joint 2×2 Newton; gmin-homotopy; quasi-2D 3×3 with split
  body Vb_S/Vb_D coupled by Rb_SD). Implicit-Euler transient solver. Arclength
  continuation through snapback fold (autograd-compatible).
- A 4D body-state surrogate (V_G1, V_G2, V_d, V_b) → (I_d, I_ii, I_leak) on a
  5×5×4×5 grid, used to drive reservoir experiments at ~10⁴× the cost of the
  underlying Newton solver.

### What pyport CANNOT do
- Single cell only — no multi-cell coupling, no shared substrate, no array
  parasitics, no SRAM-array thermal effects.
- No self-consistent electro-thermal feedback (T_C is static config).
- No quantum/stochastic effects baked in: deterministic BSIM4, no shot noise,
  RTN, 1/f, trap kinetics, or variability sampling. Stochasticity must be
  injected externally.
- No AC small-signal / S-parameters / frequency-domain analysis.
- bsim4_port is CPU-only torch.float64; ROCm/HSA path untested.

### Knobs in code but NOT yet swept in any network experiment
- `quasi2d_body=True`, `q2d_branch_protect`, `q2d_body_leak_R`,
  `iii_split_alpha`, `Rb_SD`.
- `body_pdiode_to ∈ {off, vnwell, gnd, sint}` and `body_pdiode_perim_length`.
- `m2_body_gnd=False` (a FLOATING M2 body — the configuration our user is
  hypothesising about; see below).
- `vnwell` sweep beyond 2.0 V, `vnwell_Rs`, `vnwell_mbjt`.
- `gmin_step` homotopy, `use_homotopy=True` in `forward_2t`.
- The arclength path used for training across folds.
- The implicit transient solver under arbitrary pulse programs.
- `forward_2t_batched` on the AMD Radeon 8060S (gfx1151 ROCm) iGPU.

### Pazos/Lanza source material (what we actually have)
- One die, one cell, DC ONLY: 33 quasi-static Id–Vd sweeps at V_G1 ∈
  {0.2, 0.4, 0.6} V × V_G2 ∈ [−0.2, +0.5] V. Eight corners are NaN (cell does
  not fire at low V_G2 for high V_G1).
- BSIM4 thin-oxide cards for M1 (deep-N-well floating-bulk control FET) and
  M2 (bulk NS-RAM cell, 10× length), an NPN parasitic card with avalanche
  parameters, an LTSpice schematic of the 2T+BJT+bulk-cap cell, and a
  protection diode.
- A public Zenodo bundle containing two Sentaurus TCAD project trees:
  `FloatBulk_Rsub` (resistor-biased bulk variant) and `FloatBulk_Tsub`
  (transistor-biased bulk variant — matches the LTSpice).
- The LTSpice schematic explicitly treats **V_G2 as the plasticity / STP
  knob** and V_D as the firing input.

### Conspicuously MISSING (and we are not asking for it in this round)
Transient pulse data, thick-oxide cell card, multi-cell variability,
saturation-region B_f extraction, layout/GDS, noise PSD, retention raw data.

### Off-mainline territories we HAVE already explored
- GPU↔FPGA bidirectional closed loops (causal emergence ~2.87×; GPU 1/f noise
  drives FPGA 27× closer to criticality).
- Cross-substrate fusion: FPGA NS-RAM neurons + GPU HIP-kernel nodes + firmware
  noise channels gave the project's strongest single result, +17.1 pp on a
  7-class waveform task compared with FPGA-alone.
- Thermal coupling as channel: anti-static foam between GPU and FPGA increases
  mutual information ~25% while attenuating power 4× — i.e. the useful signal
  is the spectrum, not the amplitude.
- Self-injected firmware noise (untouched VRM/SMN/PM-table/clock bytes piped
  directly to Vg).
- ISA-level analog channels (FP16 rounding mode, wall-clock SCLK, WGP bank
  parity, all unified into one feature path).
- HIP-kernel persistent reservoirs (state kept in LDS across timesteps).
- NS-RAM mode-switching cell ODE: a burst trigger toggles the cell between
  neuron and synapse mode (software only).
- Recurrent FMA neuromorphic: bit-accurate FP32 FMA with intermediate taps,
  used as firmware-only reservoir primitive.

### Directions touched but never developed
1. NS-RAM cell as a substrate that **morphs continuously between digital
   (V_G2 grounded → 0/1 transistor) and analog/LIF (V_G2 floating → body-
   charge dynamics, parasitic-NPN spike generator)**. The mode-switching ODE
   exists in software but has never been (a) run on the real 2T cell, (b)
   used as a *time-varying* control input during a reservoir task, or (c)
   trained as a learned parameter.
2. **Bridging NS-RAM with other compute substrates** (CPU/GPU/FPGA) — only
   the GPU↔FPGA half exists; the actual NS-RAM device is always a model in
   our closed loops.
3. **Mixed-population networks** where some cells are deliberately grounded
   (digital memory) and others floating (LIF/analog), wired together as a
   single fabric.

### The user's hypothesis (to evaluate, not to assume)
Universal-calculator substrates (CPUs, GPUs, TPUs, NPUs) execute *substrate-
independent* code: the same LLM weights run on any of them. By construction
this gives no identity, no rooting of computation in a particular physical
fabric. The NS-RAM cell, with its V_G2 grounded/floating regime knob, is
hypothesised to be a possible *bridge*: at V_G2 grounded the cell is a vanilla
0/1 MOSFET (calculator basis), at V_G2 floating it is an LIF
neuron / synapse / short-term-memory with a quantum-dependent avalanche
trigger. A continuous V_G2 trajectory could in principle morph a computation
gradually from calculator-territory into identity-dependent
analog-LIF territory. Whether or not the philosophical framing holds, the
physical question is concrete: **can a continuous V_G2 schedule produce a
well-defined family of compute regimes that are smoothly parametrised
between fully-digital and fully-LIF, and that have measurable functional
differences from a step-switched version?**

## Three questions

**Q1 — Ecosystem positioning.** Given pyport's actual capabilities and the
single-cell DC-only silicon data we have, where in the chip ecosystem does
NS-RAM *most plausibly* fit today, and where would it fit if the missing
characterisations (transient, multi-cell, thick-ox) arrived? Map NS-RAM
against CPUs / GPUs / TPUs / NPUs / commercial neuromorphic chips
(Loihi / SpiNNaker / Akida) / analog AI accelerators (Mythic / IBM NorthPole)
/ quantum. Be concrete about the niche (energy floor for what kind of
workload). Use the audit honestly: do NOT assume reservoir-quality
superiority over software ESNs.

**Q2 — V_G2-continuum hypothesis: is it scientifically meaningful?** Is
there a real, measurable distinction between a STEP-switched (digital → LIF
abruptly) and a SMOOTHLY-RAMPED (V_G2 traversed continuously during
inference) NS-RAM regime trajectory? If yes, name 2–3 candidate signatures
that a within-pipeline simulation could test before any silicon ask:
e.g. preserved long-range temporal correlations across the morph, gradient
flow through the regime-switch boundary, hysteresis in body charge, etc.
If no — explain why the philosophical "identity rooting" framing reduces
to nothing testable.

**Q3 — Highest-leverage independent path forward** (no Pazos/Lanza request
this round). Out of the explicit unused knobs in pyport, the unused
algorithm-side substrate-bridge territories, and the V_G2-continuum
direction, choose ONE single experiment we can run autonomously in the next
2 working days that would either (a) substantially advance NS-RAM's
defensible story, or (b) decisively kill a direction. Be specific:
script-level outline, expected wall time on a single 32-core APU + 8060S
iGPU, acceptance gate.

Each oracle should answer all three questions concretely. The user wants
candor over hype — call out where the case is weak.
