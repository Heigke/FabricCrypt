# O47 — Deep numerical spec extraction from 21 Mario+Sebas slides

## Context

We are building a SPICE-equivalent pyport model of a 130nm bulk NS-RAM
neuron-synapse cell (Mario Lanza group + Sebas Pazos measurements,
Sebas Pazos 130nm tape-out, ~April–May 2026).

We already have:
- DC I-V data: `M1_130DNWFB.txt`, `M2_130bulkNSRAM.txt`, `parasiticBJT.txt`
- PTM130 BSIM4 deck: `PTM130bulkNSRAM.txt`
- p-diode: `pdiode.txt`
- Three-branch fits (subVT/IND/HCa) for K1, NFACTOR, ETAB, BETA0 vs VG2

We have already catalogued the obvious headline specs:
- ~0.2 pJ/spike
- 130 nm / 180 nm CMOS, triple-well DNW
- ~µm² cell areas
- 72% LIF / 85% Poisson MNIST in Brian2

## Your job

NUMERICAL spec extraction. We need EVERYTHING ELSE that has a NUMBER
attached, that we may have missed.

For EACH of the 21 slides, extract:

**A. ALL numerical labels** — axis ticks, callouts, table values,
fit-line slopes, time constants τ, voltages, currents, powers, areas,
yields, variability bars, error bars, ramp rates dV/dt, hold times,
pulse widths, duty cycles, frequencies.

**B. ALL inferred values from the visual** — peak positions of curves,
knee voltages, slope changes. Slides 8 & 16 mention `S_fire` and
`S_relax` — give the V/decade (or mV/dec) values you read off. Saturation
currents I_sat. Snapback peak Vd if visible. ON/OFF ratios.

**C. Device variability** — any 100%-yield claim should have a σ on
threshold or on current at fixed V. Extract that σ.

**D. Multi-rate transient info** — slide 21 ("transient_VD_ramps")
shows multiple dV/dt rates. How many distinct rates? What V/s values?
Time-axis range? Any latency between ramp end and bulk-current decay?

**E. Network behaviour** — slides 19/20 have Brian2 LIF+MNIST.
Extract ALL hyperparameters visible: τ_mem, τ_refrac, V_thresh, V_reset,
weight range, # input neurons, # hidden, # output, learning rule,
training epochs, batch size, accuracy CI / std.

**F. Energy decomposition** — 0.2 pJ/spike: any breakdown shown?
Switching energy of the pulse generator vs body-state evolution vs
leakage between spikes?

## Output format

For EACH slide, a Markdown table:

| Quantity | Value | Units | Slide ref | Source |
|----------|-------|-------|-----------|--------|

`Source` ∈ {label, inferred-visual, inferred-text, measured-axis,
table-cell, fit-equation}.

Aim for ≥10 entries per slide where possible. Be honest "(unreadable)"
when you cannot read it.

After the per-slide tables, give a **CONSOLIDATED** section listing the
top 10 quantities you are MOST confident about and the top 5 you are
LEAST confident about (where another oracle disagreement would help).

Finally: flag any quantity that LOOKS contradictory between slides
(e.g. same parameter, different value).
