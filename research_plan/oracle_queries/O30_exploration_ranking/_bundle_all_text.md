# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: plan.md (6246 chars) ===
```
# NS-RAM Exploration Plan — Novel Architectures (2026-05-07)

**Posture shift**: stop waiting on Sebas/Mario; push our exploration
forward with chip-design unlocked (extra components allowed if they
help). Use GPU when needed; thermal-guard + sentinel must be alive.

This plan COMPLEMENTS `RESEARCH_PLAN_2026-05-07.md` (which covers
solver/fit consolidation). Both run in parallel — solver work continues
on autonomous-cron cadence; this plan owns the network-architecture
exploration thread.

---

## Why this matters

The brief's ER\_SPARSE recommendation is from a **homogeneous
single-cell-type** sweep (z142, 270 sims). Sebas's silicon already
supports multiple cell variants in the same mask set:

- **2T** (two-stack, our default) vs **1T** (single-cell)
- **Thin-ox** ($V_{G2} \in [0, 1]$~V, fast-spike) vs **thick-ox**
  ($V_{G2} \in [2.5, 3]$~V, $1$~V drain, slow integrator)
- **Shared-bulk-rail array** (cells coupled via P-body)

Plus **easy chip-level additions** that would not require new mask
steps but do require us to specify them in the next tape-out brief:

- Programmable inter-cell coupling resistor $R_{ij}$ (1 fF poly bridge)
- Active C-coupling (small MIM cap between body rails)
- Adjustable $V_{G2}$ DAC per cell or per row (existing DAC infra)
- Local body-leak resistor (10–100 GΩ, sets memory time constant)

We have NOT explored heterogeneous cell mixes, hierarchical reservoirs,
or chip-augmenting components. **The space is wide open.**

---

## Phase I — Novel architecture sweeps (CPU-friendly, this week)

| ID | Direction | Description | Effort |
|----|-----------|-------------|--------|
| I.1 | **Hetero-cell ratio** | Mix thin-ox + thick-ox cells in same network. Sweep ratio 0%/25%/50%/75%/100% thick. Hypothesis: fast cells = readout, slow cells = memory; mix is better than either alone. | 1 day |
| I.2 | **Hierarchical depth** | Stack reservoirs: layer-1 fast small (N=64), layer-2 slow medium (N=256), layer-3 large (N=1024). Sweep depth 1-4. | 1 day |
| I.3 | **Lateral inhibition** | Add inhibitory ring (radius 1-8 cells) on top of ER\_SPARSE. Hypothesis: improves separability without adding mask steps (sign-inverter sub-fabric already in C.3). | 0.5 day |
| I.4 | **Spike vs analog readout** | Use parasitic-NPN snapback fold as binary spike output instead of analog $I_d$. Compare classification accuracy + energy. | 1 day |
| I.5 | **Programmable VG2 schedules** | Cell switches role mid-sequence (synapse → integrator → spike) via VG2 modulation. Schedule: linear ramp / square wave / chirp. | 0.5 day |
| I.6 | **Multi-frequency VG2 dither** | Different cells respond to different dither frequencies (frequency multiplexing). Output channelised by FFT bin. | 0.5 day |

---

## Phase II — Scale-up (GPU, careful thermal)

| ID | Item | Wall budget | Notes |
|----|------|-------------|-------|
| II.1 | z202 N=8192 with thread-cap (resume crashed run) | 4-6 h CPU | F.1 thread-cap is now wired |
| II.2 | ER\_SPARSE N=8192, 3 seeds | 6-8 h CPU | Match z202 protocol |
| II.3 | Best architecture from Phase I → N=4096 (GPU port) | 2-3 h GPU | gfx1151 ROCm 7.0 path verified |
| II.4 | 10-seed reproducibility at the winning (architecture, N) | 4 h CPU | Bootstrap CI on accuracy |

**Thermal protocol** for GPU runs:
- Pre-check `thermal_guard --probe < 60°C`
- Cap kernel duration to 8s bursts; sleep 2s between
- Monitor `/sys/class/thermal/thermal_zone0/temp` every 100 steps
- Pause @ 75°C, resume @ 50°C
- See z2297 lessons (thermal_guard.py 50-step cadence)

---

## Phase III — Task suite expansion (after Phase I winner is known)

| ID | Task | Why |
|----|------|-----|
| III.1 | Spoken-digit (FSDD or TI-46 subset) | Gemini's O29 suggestion: showcases temporal memory of NS-RAM dynamics |
| III.2 | Santa-Fe laser chaotic forecast | Gpt-5's O29 suggestion: classic regression target |
| III.3 | Sequential MNIST (row-by-row) | Industry benchmark; tests long-range dependency |
| III.4 | Copy-task (sequence reconstruction) | Pure memory test |
| III.5 | Parity-N (temporal XOR generalization) | Long-range XOR variants |

---

## Phase IV — Novel circuit ideas (chip-design contributions)

These would be in the **NEXT brief** to Mario for the M9/M12 chip:

| ID | Idea | Cell-cost | Expected upside |
|----|------|-----------|-----------------|
| IV.1 | Active C-coupling between body rails (1 fF MIM caps) | +1 cap per pair | adds ~µs short-term plasticity scale |
| IV.2 | Per-cell programmable body-leak (10-100 GΩ poly resistor) | +1 R per cell | tunes memory τ; erases parasitic latch (per O28 finding) |
| IV.3 | Local lateral-inhibition cross-bar (4-cell radius) | +12 W per cell | improves separability without mask change |
| IV.4 | Two-NPN model option (parallel collector-emitter) | layout change | per O25 oracles, +0.03–0.08 dec fit gain |
| IV.5 | Quasi-2D body split (V_{b,S}, V_{b,D} + Rb_SD) | +1 R per cell | per Plan A; adds bias-asymmetry handling |

---

## Phase V — Theoretical / write-up

| ID | Item | Wall |
|----|------|------|
| V.1 | "NS-RAM as continuous-time learning substrate" position paper | 1 week |
| V.2 | Compare NS-RAM dynamics to Hopfield / Liquid State Machine theory | 3 days |
| V.3 | Energy-per-decision Pareto vs Innatera, Mythic, GAP9 | 2 days |

---

## Cron integration

This plan adds to the existing `d8d4209a` work-hours queue. Wake-ups
now alternate between:
- Solver/fit work (RESEARCH_PLAN_2026-05-07.md, B.x track)
- Architecture exploration (this plan, I.x and II.x track)

The wake-up handler should pick from whichever track has the highest
expected-value action ready. Phase I items are CPU-friendly and many
fit in a single wake-up; Phase II requires multi-wake-up scheduling
or one big dedicated session.

---

## Decision gates

- After Phase I (~1 week): pick winning architecture, kill losers.
  Decision basis: bootstrap-CI improvement on at least 2 of {Mackey-Glass,
  XOR τ=2, MC} vs ER\_SPARSE baseline.
- After Phase II (~2 weeks): pick winning size; decide GPU is required
  for production demo (yes/no).
- After Phase III (~3 weeks): assemble paper draft + chip-spec amendment
  for Mario's next brief.

---

*Drafted 2026-05-07 by Eric. Send to oracles for ranking before
committing to Phase I top item.*

```
