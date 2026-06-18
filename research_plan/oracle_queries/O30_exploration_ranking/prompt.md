# O30 — Rank novel NS-RAM exploration directions

## Context

Brief sent to NRF 2026-05-06 with ER_SPARSE tape-out recommendation
based on a HOMOGENEOUS single-cell-type sweep. We have months of
post-deadline runway to push exploration deeper. Chip design isn't
locked: we can specify extra components in the next brief revision
(active C-coupling, body-leak resistors, lateral inhibition crossbar,
heterogeneous cell mixes). GPU available (gfx1151 ROCm 7.0) for
larger sims, with thermal-guard.

The attached `plan.md` is a draft exploration plan with 6 Phase-I
novel-architecture sweeps:
  I.1 hetero-cell ratio (thin-ox + thick-ox mix)
  I.2 hierarchical depth (stacked reservoirs)
  I.3 lateral inhibition ring
  I.4 spike vs analog readout (use snapback fold as binary spike)
  I.5 programmable VG2 schedules (cell role switches mid-sequence)
  I.6 multi-frequency VG2 dither (frequency multiplexing)

## Questions (be terse, < 500 words total)

1. **Rank Phase I items 1-6 by expected-value-per-effort**. Which
   should I run FIRST given tonight's wake-up budget? Why?

2. **Missing direction**: what novel architecture or chip-component
   addition is NOT in the plan but should be? Particularly worried
   about blind spots in:
   - Connection topology (we've only tried ER, MESH, WS, HUB,
     SCALE_FREE, GRID, MODULAR, RING, RAND_GAUSS — anything missed?)
   - Plasticity rules (we have FF, Hebbian, rhebb — anything else
     worth trying?)
   - Input encoding (we use direct VG1 modulation; any cleverer way?)
   - Readout (linear ridge today; what about temporal integration
     window, attention, gated readout?)

3. **Phase IV chip components** (active C-coupling, body-leak resistor,
   lateral-inhibition crossbar, two-NPN, quasi-2D body): which of
   these would you spend mask-area on FIRST in the next chip iteration?
   Why?

4. **Killer benchmark**: if I had to pick ONE task to prove NS-RAM
   networks are useful (instead of toys), which task and why?

5. **Most provocative idea you'd add**: brainstorm 1-2 novel directions
   that would be genuinely surprising or impactful if they worked.
   Don't worry about feasibility; goal is to EXHAUST THE SPACE.

Goal: a ranked top-3 to execute first, plus one wildcard idea that
makes me rethink the chip.
