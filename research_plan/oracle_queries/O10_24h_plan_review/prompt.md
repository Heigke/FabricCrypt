# O10 — 24h plan review and critique

We just closed Phase A (DC fidelity) on a PyTorch BSIM4 v4.8.3 port of Sebastian
Pazos's 130 nm 2T NS-RAM cell. The user wants a multi-oracle critique of our
24h-ahead plan and an honest read on whether we're heading the right direction
for the Mario Lanza NRF brief due 2026-05-06 and the longer-term tape-out
recommendation for KAUST.

## Where we just landed (Phase A closure metrics)

  Median z91g log-RMSE on 33-bias measured set: **1.002 dec**
  Subthreshold Id ratio vs ngspice on isolated M2: **1.05× across 4 decades**
  Vth gap: **≤1.5 mV** at Vds ∈ {0.05, 0.5, 2.0} V
  DIBL gap: **0.3 mV/V**
  Solver convergence: 100% on random bias sweeps
  GPU pipeline: 5 × 10⁶ cell-evals/s on Radeon 8060S (5× G2 target)
  Single-cell transient: ~14 s per 500-step trajectory, 100% Newton convergence

## The five silent ngspice-42 bugs we found and worked around

  1. `.model` body silently drops 2nd+ assignment per line (wvth0, voffl, ww,
     binunit=2, etc.)
  2. `toxe = toxn` .param substitution fails → BSIM4 default 3 nm used
     instead of card 4 nm
  3. `lpe0 = lpe0n` same .param substitution failure → default 1.74e-7
     instead of card 1.244e-7
  4. Pyport's prior phi formula was wrong (extra factor of 2, missing
     +0.4 constant); ngspice form is `phi = Vt·log(NDEP/ni) + phin + 0.4`
  5. `phin` silently dropped despite being first-on-line, making the +0.4
     constant the only contribution

All five documented and reproducible via our debug-printf-instrumented
ngspice-42 rebuild plus a literal C-to-Python port of b4ld.c §1042-1336.

## Where we are NOT (architectural limit)

Memory capacity benchmark on N=10 cells, dt=10 ns: **MC = 0.16** (chance ≈ 0,
theoretical max ≈ 10). With and WITHOUT calibration shifts, MC stays at chance
because cells are independent (no inter-cell coupling), and body-cap τ ≈ 0.7 ns
≪ sample dt → no temporal memory survives between samples.

Cell is a **memoryless analog weight**, not a reservoir node. NS-RAM
architectural recurrence must come from external CMOS or shared-rail topology.

## Sebas/Mario data inventory

  data/sebas_2026_04_22/
    M1_130DNWFB.txt           ← M1 BSIM4 model card (130 nm, DNW floating body)
    M2_130bulkNSRAM.txt       ← M2 BSIM4 model card (130 nm, bulk NS-RAM)
    parasiticBJT.txt          ← LTspice Gummel-Poon NPN card
    2Tcell_BSIM_param_DC.csv  ← Per-bias parameter overrides (33 rows)
    2tnsram_simple.asc        ← LTspice schematic of 2T cell
    "I-Vs@VG2 VG1=0.2/0.4/0.6 vnwell=2"  ← measured I-V data (3 VG1 sweeps)
    PTM130bulkNSRAM.txt       ← Predictive Technology Model reference

  data/sebas_2026_05_02/
    pdiode.txt                ← Body-junction diode SPICE card
    image-2.png               ← Sebas's "Dynamic response (ramp rate)" slide
                                with 7 sweep rates 1ms→5µs at VG1=0.4/VG2=0.3
                                — STILL NEEDED: raw 7-rate transient data

  Pending data requests to Sebas (#A.12, blocking transient validation):
    - Thick-ox 17 µm² cell BSIM4 card (VG2 ∈ [2.5, 3.0] V, 1 V drain)
    - 7-rate transient measurement traces from his "image-2.png" slide

  Mario brief target:
    - 10–100 mW gateway devices (Innatera ladder)
    - Energy: 21 fJ/cycle, 6.7 fJ/spike, 46 µm² (thin-ox cell)
    - One-pager NRF brief due 2026-05-06 (4 days)

## 24h ahead plan

  T+0  to  T+6h:  B.5.c — implement inter-cell coupling in topology.py:
    add shared-bulk-rail resistor between neighboring cells
    (e.g. Vb_i ↔ Vb_j via 1 kΩ–1 MΩ). Re-run MC with N=10 coupled cells.
    EXPECT: MC > 1.0 if coupling provides reservoir recurrence.

  T+6h  to T+12h:  if MC works, scale to N=100 + additional benchmarks
    (XOR τ=2,5; waveform classification). Otherwise pivot to
    software-recurrence layer (B option) atop static cells.

  T+12h to T+18h:  M9 deliverable — 10-30 cell fan-out validation circuit
    that Sebas explicitly asked for. This unblocks the testchip floorplan
    slot and is high-leverage for the NRF brief.

  T+18h to T+24h:  refresh Mario one-pager with whatever new results we
    have, send draft email (TODO from yesterday).

## Questions for the oracle panel

1. **Is the plan order right?** Should we prioritize topology coupling (B.5.c)
   or fan-out validation circuit (M9) first?

2. **What are we missing from the Sebas/Mario data?** Should we be more
   aggressive about asking Sebas for the thick-ox card and 7-rate data?

3. **Is the ngspice bug catalogue publishable as a standalone note?** Or is
   it too project-specific?

4. **For the Mario brief — what's the strongest possible framing?** We have
   1.00 dec DC fit, found and fixed 5 bugs in ngspice's calibration loop,
   working transient solver, GPU pipeline at 5× G2 target. What story
   maximises NRF-funding probability?

5. **Critical risks we're under-weighting?** The user wants Grok especially
   to be brutal here — what's the weakest link in our claim?

6. **For longer term (M3-M12):** is the timeline realistic? Are we
   under-staffed? What deliverables would NRF reviewers find unconvincing?

## Files attached

- `01_LOG_tail.md` — last ~150 lines of research log (Phase A closure)
- `nsram_proposal_short.tex` — current Mario one-pager LaTeX source
- `nsram_proposal_short.pdf` — rendered PDF
- `quadrant_nsram_vs_edge.png` — accelerator positioning chart
- `M2_130bulkNSRAM.txt` — Sebas's M2 card (the one that revealed the parser
  bugs)
- `z91g_fit_vs_meas.png` — 33-bias fit-vs-measurement plot

Please be specific. Bullet your top-3 critiques and your top-3 recommendations.
We have ~24 h before the next user check-in; we want to spend those wisely.
