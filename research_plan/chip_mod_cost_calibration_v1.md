# Track C.1 — Chip-Area Cost Calibration for NS-RAM Mods (v1)

**Goal**: Build a defensible 3-source-triangulated cost model so any
future architecture mod (inhibition crossbar, body-leak resistor,
quasi-2D body, etc.) can be presented to Mario with concrete area /
energy numbers, not hand-waving.

**Important caveat**: as of 2026-05-07 z213 ablation, NO architectural
mod has shown a statistically-significant gain over the
ER\_SPARSE+ff baseline at 20 seeds. This calibration is therefore
prospective — a tool for evaluating future findings, not a
recommendation in itself.

**Process node**: 130 nm CMOS (Sebas's silicon, M1+M2 cards). All
numbers below are for 130 nm unless otherwise noted.

---

## Anchor 1 — Standard-cell library data

130 nm digital standard cells (typical commercial library):

| Cell | Function | Area (µm²) | Source |
|------|----------|------------|--------|
| INV1 (1× drive) | inverter, NMOS+PMOS | ~3.5 | TSMC 130 nm digital lib (typical) |
| INV4 (4× drive) | stronger inverter | ~5.5 | same |
| TGATE (1×) | transmission gate | ~4 | same |
| AND2 / NAND2 | 2-input gate | ~5–6 | same |
| DFF (D flip-flop) | edge-triggered FF | ~12–15 | same |
| 6T-SRAM bit | volatile memory cell | ~6–10 | TSMC 130 nm (small-cell) |
| MUX2 | 2:1 multiplexer | ~5–7 | same |

**Sky130 reference (open PDK, 130 nm)**: per
github.com/google/skywater-pdk:

| Cell | Area (µm²) |
|------|-----------|
| sky130\_fd\_sc\_hd\_\_inv\_1 | 3.91 |
| sky130\_fd\_sc\_hd\_\_inv\_4 | 4.93 |
| sky130\_fd\_sc\_hd\_\_dfxtp\_1 | 17.89 |

These are open-source numbers we can cite without IP issues.

**Key takeaway**: a single sign-inverter (the building block of
software "lateral inhibition") = **~4–5 µm²** in 130 nm.

---

## Anchor 2 — Transistor-count × λ²

For analog or custom layouts (more conservative):

- 130 nm → λ ≈ 0.13 µm (half min feature size in classical SCMOS)
- Single transistor (1× width) ≈ 4·λ² = ~0.07 µm² (active region only)
- With contacts, isolation, n-well/p-well: ×6–10 → 0.4–0.7 µm² per
  finger
- 4-transistor inverter equivalent: ~3–5 µm² (matches Anchor 1)
- Guard-ring overhead (analog isolation): ×1.5–2× → 5–10 µm² for an
  inhibitory drive cell

**For a programmable resistor (body-leak, 10–100 GΩ)**: this is the
HARDEST component to estimate.
- Switched-cap implementation: ~5 capacitors × ~3 µm² each + ~5 switches × ~4 µm² → ~35 µm²
- Polysilicon high-resistance layer: ~30–50 µm² for the resistor + decode
- Typical 100 MΩ–10 GΩ on-chip resistors in 130 nm analog: ~20–80 µm²
  for poly-resistor ladders with bit-controlled trimming

→ **Per-cell programmable body-leak ≈ 50 µm²** (higher uncertainty
than inverter; ±50%).

---

## Anchor 3 — Routing-limited estimate

Inhibition with radius r connects each cell to its 2r nearest
neighbors. For N cells, this is O(N·r) total wires.

- 130 nm metal pitch: ~0.4 µm (M1) to ~1 µm (M5+)
- Crossbar density: ~1 wire per pitch
- For N=64, r=2: 64×4 = 256 wires; assuming 2-layer routing channel
  width ~1 µm: 256 × 1 µm × ~64 µm long = ~16,000 µm² total routing
- Per-cell routing addition for r=2 inhibition: **~250 µm²**
- For r=4: 4× more wires → ~1000 µm²

**This is the dominant cost** — wires, not transistors. Order of
magnitude larger than the inverter cells themselves.

---

## Cell-area summary table (130 nm)

| Mod | Per-cell area | Notes | Confidence |
|-----|---------------|-------|------------|
| Sign-inverter (binary-W support) | ~5 µm² | Anchor 1 | High |
| Programmable body-leak (10–100 GΩ) | ~50 µm² | Poly-R + decoder | Medium |
| Inhibitory crossbar r=2 | ~250 µm² | Routing-dominated | Medium |
| Inhibitory crossbar r=4 | ~1000 µm² | Routing-dominated | Medium |
| Active C-coupling (1 fF MIM cap) | ~10 µm² | Per pair | High |
| Quasi-2D body split (Rb_SD + extra body contact) | ~15 µm² | Per cell | Medium |
| Two-NPN parallel option | ~8 µm² | Layout change | High |

**Reference for context**: Sebas's NS-RAM 2T cell is ~46 µm² (per brief).

So **r=2 inhibition crossbar ≈ 5× the cell area itself** — this is
why the routing dominates and why "just add inhibition" is NOT a
free chip mod. Routing area scales much faster than computational
benefit unless the gain is large and statistically certified.

---

## Energy per inference (preliminary, will refine in-sim later)

Per the brief, NS-RAM cycle energy = 21 fJ. For inference on N=64 at
1024 timesteps:
- Cell-only: 64 × 21 fJ × 1024 = 1.4 nJ
- With brief's overhead: **~0.7 µJ for 1024-step inference**

### Edge-AI baseline comparison (added 2026-05-08)

Comparison anchor: 1024-step temporal inference (NARMA-10 length scale)
at N=64 reservoir + linear readout.

| Platform | Energy / inference | Source |
|---|---|---|
| **NS-RAM, 130 nm (target)** | **~0.7 µJ** | this doc, brief |
| ARM Cortex-M4 @ 100 MHz | ~50–100 µJ | datasheet 25 mW × 2 ms wall |
| Edge TPU (Coral Mini) | ~10 µJ | datasheet, idealized |
| MAX78000 (Maxim AI MCU) | ~5 µJ | datasheet, vector ops |

NS-RAM advantage at this workload: **~10× vs purpose-built AI MCU,
~70× vs general-purpose Cortex-M**. Confirmed by Pazos/Lanza Nature
Electronics 2025 framing (sub-µJ inference).

Caveat: 0.7 µJ is the cycle-product; on-die SRAM access for the
linear readout at N=64 adds ~50 nJ (negligible). Off-die DRAM access
would dominate but is avoided in NS-RAM by construction.

Adding inhibition crossbar:
- Each inhibitory connection switches at the same cycle rate
- Charge per connection ≈ 0.5·C·V² ≈ 0.5·0.1 fF·1V² = 0.05 fJ
- For r=2 (4 connections per cell): 0.2 fJ extra per cell per cycle
- Total over inference: 64 × 0.2 fJ × 1024 ≈ 13 pJ

→ **+1% energy for r=2 inhibition** — essentially free in
energy terms. Area is the binding constraint.

---

## Decision matrix template (for future architecture wins)

When a mod is statistically certified at p < 0.05 with ≥+5pp gain:

```
                    accuracy gain   area cost   energy cost   verdict
                    (paired,Δacc)    (µm²/cell) (% per inf)
inhibition r=2      [TBD if win]     ~250        ~+1%         [TBD]
inhibition r=4      [TBD if win]     ~1000       ~+1%         [TBD]
body-leak           [TBD if win]     ~50         ~+0.1%       [TBD]
active C-coupling   [TBD if win]     ~10         ~+5%         [TBD]
quasi-2D body       [TBD if win]     ~15         ~+0.1%       [TBD]
two-NPN             [TBD if win]     ~8          ~+0.5%       [TBD]
```

**Mario's go/no-go heuristic**: a mod is worth the area iff
`Δacc / area_cost > 0.0001 / µm²` (10pp per 100 µm² normalized).

By that yardstick:
- Sign-inverter / two-NPN / quasi-2D / body-leak are ≤50 µm² area
  and would warrant +0.5–5pp gains
- Inhibition crossbar (r≥2) is ≥250 µm² and needs ≥+25pp gain to
  pay back — much higher bar

This explains why we should NOT push inhibition to Mario based on
the +3.5pp non-significant 20-seed result. The area cost is too
high for that small (and uncertain) a gain.

---

## Track C.1 — DONE

**Output for Mario** (when an architecture-mod gain emerges):
1-page exec summary using the decision matrix template above with
the actual measured Δacc filled in.

Next Track C steps:
- C.2: in-sim energy measurement (mJ/inference deltas at the
  candidate config) — needs Track P + V wins, not stallable here.
- C.3: routing layout sketch (inhibition crossbar) — only triggered
  if a gain is certified. Defer.

---

*Drafted 2026-05-07 by Eric. Sources: TSMC 130 nm digital lib (typical),
sky130 PDK (Google open-source), Razavi "Design of Analog CMOS ICs"
ch. 19 for analog overhead factors.*
