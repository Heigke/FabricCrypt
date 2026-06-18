# O8 — Milestone review of the NS-RAM PyTorch port before scaling to topology benchmarks

You are reviewing a multi-month port of Sebastian Pazos's 2T NS-RAM cell (130 nm
bulk CMOS) from LTSpice/BSIM4 into a PyTorch implementation. We are at a natural
checkpoint: single-cell physics is done and validated, the implicit transient
solver works, the parasitic NPN fires when driven hard, and we are about to
commit ~weeks of compute to large-scale topology benchmarks. Before we do, we
want a sanity check from two independent oracles on physics credibility,
benchmark choice, and whatever we may have missed.

## Setup

PyTorch BSIM4 v4.8.3 port of Sebastian Pazos's 2T NS-RAM cell, 130 nm bulk
CMOS. Topology (per `2tnsram_simple.asc`):

- **M1** (short, L=0.18 µm, W=0.36 µm): nmos4. Drain=Vd, Gate=VG1, Source=Sint,
  Body=B (floating P-body).
- **M2** (long, L=1.8 µm = 10× M1, W=0.36 µm): nmos4. Drain=Sint, Gate=VG2,
  Source=GND, Body=GND (LTSpice nmos4 default for unconnected body).
- **Q1** parasitic NPN (`parasiticBJT.txt`, Gummel-Poon, Bf=10000, area=1u):
  Collector=Vd, Base=B, Emitter=GND. This is the lateral/vertical NPN inherent
  to the floating P-body inside an N-substrate.
- **CBpar** = 1 fF, Sint↔Vd.
- **vnwell** = +2 V (chip-level deep N-well bias). Drives a P-body→DNW pdiode.
  Sebas's card (`pdiode.txt`): is=5.37e-7 A/m², n=1.05, cj=7.3e-4 F/m²,
  Vj=0.219 V, M=0.241, area = 5 µm × 4.4 µm = 22 µm². At vnwell=2 V this diode
  is strongly reverse-biased → DC current is ≈ −Js·area·(1+...) ≈ pA → DC-inert.
  At op-point its small-signal Cj is ≈ 10 fF, matching Sebas's "5–10 fF" hint
  exactly, which is why he asked us to model it: it captures voltage-dependent
  body capacitance.

Body node B is the only floating internal node; M2.B is grounded (LTSpice
default), so the only Newton unknowns at DC are (Vsint, Vbody). Solver:
**pseudo-arclength continuation in Vd** with **autograd-exact Jacobian Newton
+ Armijo line search**. Vectorized batched solver runs N=1000 cells eager at
22k cell-evals/s, 38k compiled. ROCm 6.3 / AMD Radeon 8060S; CPU↔GPU crossover
at N≈100k.

## Status at this checkpoint (2026-05-01 → 2026-05-02)

1. **DC fit (33-curve I-V family)**: median **0.97 dec** error (log-decade).
   Convergence is full across the full bias grid. The residual is
   **card-limit, not port bug** — verified by:
   * ngspice cross-check on the same .lib cards reproduces the same 0.9–1.0
     dec offset against Sebas's measured data;
   * hand-derivation of the BSIM4 Vth aggregator agrees with ngspice's
     `M.threshold` print to **90 µV** across the bias grid;
   * Vth shift from low to high VG2 in Sebas's data (~80 mV) is reproduced.

2. **Pdiode integration (added 2026-05-02)**: pdiode body↔DNW with Sebas's
   card. At vnwell=+2 V the diode is reverse-biased → adds DC junction
   leakage (~pA) and a voltage-dependent Cj that integrates into the
   transient charge equation on the body. Cj at op-point ≈ 10 fF — exactly
   matches Sebas's "linear-cap equivalent 5–10 fF" hint.

3. **Implicit transient solver (`joint_newton.py` + `transient.py`)**:
   backward-Euler on (Vsint, Vbody) with full coupling through CBpar (Sint↔Vd)
   and Cj_pdiode (B↔DNW). Validated against DC limit at slow ramps (Vd ramp
   over 1 ms shows DC trace within 1 mV). Walks through the snapback fold
   (negative differential resistance region around Vd≈1.3 V at high VG1)
   without losing convergence — pseudo-arclength is essential here.

4. **NPN firing regime — extrapolated above Sebas's measurement range**:
   Sebas measured up to Vd=2 V. We extrapolated to **Vd=3 V** in transient
   simulation. Result: at Vd=3 V the parasitic NPN fires — Vbody crosses
   **0.62 V** (BJT base-emitter knee), Id jumps to **1.7e-4 A**. We then
   demonstrated **9 spikes in a 5-pulse train** at VG1=0.6, VG2=0.45 with
   Sebas's per-bias overrides applied. Plot attached: `lif_real_spikes.png`.
   The mechanism is: drain pulse charges body via CBpar coupling and
   impact-ionisation-like coupling through M1, body voltage rises, NPN
   discharges body to ground through collector-emitter, body falls,
   refractory period set by RC of Cj_pdiode and shunt path. This is the
   **floating-body LIF neuron mechanism** Sebas's group reports.

5. **Vectorized batched solver (`vectorized.py`)**: 22k cell-evals/s eager,
   38k compiled. **GPU verified** on ROCm 6.3 / AMD Radeon 8060S; crossover
   with CPU at N≈100k. Headroom for the 1024-cell topology benchmarks.

## Where we're going

Large-scale topology benchmarks at 5 network scales (9, 16, 64, 256, 1024
cells) on:

- **Hopfield retrieval** (associative memory; tests cell bistability +
  inter-cell coupling).
- **NARMA-10** (nonlinear autoregressive moving average; tests dynamical
  memory).
- **Memory capacity** (Jaeger, linear MC sum).
- **Temporal-XOR** (τ=1, 5, 10).
- **Multi-class waveform classification** (sine/square/triangle/saw,
  4-class).

Goal: identify which **(topology, scale)** combination beats a 1 W edge
accelerator on **energy / decision** — that's the funding-relevant metric.

## Specific questions (4 max — please address each)

**Q1 — Vd=3 V firing regime: physically credible or model breakdown?**
We're 1.5× above Sebas's measurement range. Gate oxide in 130 nm bulk is
nominally 2–3 nm thick → tox-limited Vgs ≤ ~3.6 V. Drain is at 3 V with
gate at 0.6 V, so |Vgd| = 2.4 V — below GO breakdown but well above Sebas's
measured grid. Concerns:
  (a) BSIM4 gate-tunneling currents (igidl, igisl) become non-negligible
      and our card may not have them parameterised correctly.
  (b) Hot-carrier degradation regime (Vds ≥ 2 V × Lmin = 0.18 µm) could
      shift Vth long-term.
  (c) Drain-bulk junction breakdown: Vdb = Vd − Vbody = 3 − 0.62 = 2.38 V,
      well within typical n+/p-well BVdss (~6–8 V), so probably fine.
  (d) Self-heating at Id=170 µA × Vd=3 V = 510 µW per cell — at 1024 cells
      that's 522 mW into a 22 µm² × 1024 ≈ 0.022 mm² area. Density ~24
      kW/cm². Self-heating effect needs explicit modelling (BSIM4 RTH/CTH).

What's a sane operating-ceiling Vd that lets us claim physical credibility
for chip-level results? Our gut says Vd ≤ 2.5 V, but we'd lose the spiking
demonstration. Is there a middle ground (e.g. Vd=2.5 V with stronger M1, or
adding deliberate weak avalanche multiplication)?

**Q2 — Gummel-Poon at Vbe=0.62 V on the parasitic NPN: faithful?**
Body voltage at firing reaches 0.62 V; the NPN is at the knee. Bf=10000
seems extreme for a parasitic — Sebas chose it to match observed coupling
strength, but it implies very low recombination and probably unphysical
high-injection behaviour. Our Gummel-Poon currently has Is, Bf, Vaf only —
no Ikf/Ikr (high-injection knees), no Ise/Isc (recombination). Concerns:
  (a) At Vbe=0.62 V and Bf=10000, Ic ≈ 1.7e-4 A → Ib ≈ 17 nA. With high
      injection, Bf rolls off → effective Ic could be 5–10× lower → spike
      amplitude shrinks → maybe no firing at all.
  (b) Should we add Ikf=1e-3, Ise=1e-15, Ne=2 as a reasonable default and
      see if firing survives? Or does Bf=10000 with no rolloff already
      *imply* Sebas tuned it to match measured firing, in which case adding
      rolloff would double-count and we should leave it alone?

**Q3 — Which network-scale benchmark to lead with for funding?**
Three flavours, ranked by us in order of (a) novelty, (b) ease of
demonstration, (c) interpretability:

  - **Reservoir computing** (random sparse coupling + linear readout):
    easiest to show, accepted in literature, but "yet another reservoir"
    risk.
  - **Meta-plasticity** (single cell switches between LIF / synapse /
    memory roles via VG1/VG2 set-points): most novel — single-substrate
    multi-functionality is Sebas's signature claim — but harder to
    benchmark cleanly.
  - **Memory capacity** (Jaeger MC, linear): fundamental, lets us write a
    scaling-law plot (MC vs N), but unexciting on its own.

Given we want to make the **strongest case for funding**, which would you
lead with? Energy-per-decision is the headline metric.

**Q4 — Anything we've MISSED that would invalidate large-scale results
derived from this single-cell model?**
We want a brutal pre-mortem. Candidates we've already considered:
  - Inter-cell electrical coupling (substrate noise, shared DNW potential
    pumping each others' bodies) — currently not modelled; each cell has
    its own isolated DNW node.
  - Wire RC delays (fan-out routing, especially for 1024-cell).
  - Process variation (Vth mismatch, Cj area variation; we should sweep at
    least σ_Vth=20 mV).
  - Thermal coupling (already mentioned in Q1).
  - Circuit-level readout: can we actually read out 1024 floating-body
    voltages without disturbing them? Charge injection from sense
    amplifiers.

What else? In particular: anything in the BSIM4 card or the topology that
becomes wrong at scale but is invisible at single-cell?

## Format

Please answer each question with:
1. **Verdict** (one sentence).
2. **Reasoning** (1–3 paragraphs).
3. **Concrete recommendation** for what we should do this week before
   starting the topology runs.

## Attachments

- `nsram_cell_2T.py` — full single-cell PyTorch port (DC residuals, BSIM4
  body of equations, pdiode integration).
- `joint_newton.py` — implicit transient solver (backward-Euler + autograd
  Jacobian + Armijo).
- `transient.py` — Cj-helper (voltage-dependent junction capacitance from
  pdiode card).
- `vectorized.py` — batched solver for N cells.
- `topology.py` — multi-cell layer scaffold (where we're going next).
- `M1_130DNWFB.txt`, `M2_130bulkNSRAM.txt` — BSIM4 SPICE cards for M1, M2.
- `parasiticBJT.txt` — Gummel-Poon NPN card.
- `pdiode.txt` — pdiode card (body↔DNW).
- `2tnsram_simple.asc` — LTSpice schematic (text form, the source of truth
  for topology).
- `sebas_schematic.png` — Sebas's updated 2026-05-02 schematic showing the
  pdiode placement explicitly.
- `lif_real_spikes.png` — the 9-spike-in-5-pulse-train demonstration at
  Vd=3 V, VG1=0.6, VG2=0.45.
- `00_RESEARCH_PLAN.md` — top-level research plan.
- `01_LOG_tail80.md` — last 80 lines of the daily research log (what just
  happened).

Thanks. Be honest — if the Vd=3 V firing is in la-la land, say so; if the
benchmark choice is obvious to you, say so; if you spot a gap we missed,
that's the most valuable thing you can give us.
