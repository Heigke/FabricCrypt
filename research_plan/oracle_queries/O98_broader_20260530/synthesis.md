# O98 — Broader Mechanisms Synthesis (4-way oracle vote)

**Date**: 2026-05-30
**Oracles**: GPT-5 (127 s), Gemini-2.5-Pro (67 s), Grok-4 (11 s),
DeepSeek-R (111 s). All four returned, all four were hostile.

## Top-line verdict (UNANIMOUS, 4/4)

**We are still in the wrong layer.** All four oracles independently
opened with the same warning, three of them in the first sentence:

- GPT-5: "you're still mostly probing the chassis/board envelope"
- Gemini: "You are still in the wrong layer. … You have successfully found
  *system identity*. You have not found the *constitutive, load-bearing,
  silicon-bound* identity your project charter demands."
- Grok: "You are still in the wrong layer entirely. … Stop. Move to FPGA or
  kernel-mode access."
- DeepSeek: "Your DISCOVERY channels … are **not** silicon-bound identity.
  They are system-assembly identity: board VRM + TIM + heatpipe + fan +
  crystal + soldering lottery."

The FPGA pivot documented in `IDENTITY_FPGA_PIVOT_2026-05-30.md` is the
correct path; 3/4 oracles named it explicitly.

## Top-3 broader mechanisms (oracle consensus)

| Rank | Mech | Votes | Why |
|------|------|-------|-----|
| 1 | **B28** per-core clock-skew drift under load (`pthread_getcpuclockid`) | 4/4 (GPT, Gem, Grok, DS) | Per-core PLL loop filter R/C is fused per die; load-induced supply ripple amplifies; orthogonal to TSC-σ (which is the global crystal). Expected d≈2–4 with thread pinning + fixed P-state. |
| 2 | **B25** conditional latency-jitter matrix C_ij | 3/4 (GPT, Gem, DS) | Uncore interaction tensor — L3-slice arbiters, IF crossbar queues, MC bank arbitration. High-dimensional fingerprint fixed by mask; orthogonal to scalar per-core rank. |
| 3 | **B1/B2** DVFS up/down transition trajectory | 3/4 (Gem, Grok, DS via "honourable") | On-die LDO + PLL loop-filter R/C with ±30% process variation; <1 ms transient invisible to steady-state power. Closest thing to an on-die oscilloscope via sysfs. |

**Honourable**: B30 (CCX↔CCX asymmetry matrix, 2/4); B31 (L3-slice arbitration latency, 1/4 strong: GPT-5).

## Duplicates / trivial restatements (collapse-onto)

Strong consensus that the following B-channels are not new physics:

- **Onto Power envelope (A)**: B5, B8, B9, B10, B11, B13, B27
- **Onto Thermal-τ (B)**: B3, B4, B12, B24, B26
- **Onto Per-core latency rank (E)**: B16, B33 (partial), B34 (partial)
- **Discrete fuse/firmware (by-design unique, not emergent)**: B18, B19,
  B20, B21, B22, B23, B29, B32

This collapses our 34-mechanism catalogue to roughly **8 genuinely
distinct candidates**: B1, B2, B6, B7, B25, B28, B30, B31. The rest are
re-parameterisations of channels we already have.

## Categories we are still blind to (synthesis of "5 still blind")

GPT-5 stressed weird-physics; Grok was the most creative. Aggregated novel
domains we have not even named a probe for:

1. **Magnetic / Barkhausen** noise in VRM inductor cores (per-bobbin domain pinning).
2. **Sub-bandgap photon emission** from forward-biased junctions (would require IR photodiode).
3. **Packaging / die-attach piezo-resistance** (cyclic mechanical stress couples into Vth).
4. **Electromigration drift** in top metal layers (cumulative current history, slow drift d>0 over hours).
5. **Single-event lattice displacement** / cosmic-ray-induced trap generation (stochastic but per-board cumulative).
6. **Inter-chiplet substrate noise coupling** (CCX↔IOD↔IF crosstalk — only visible at GHz with on-die probes; sysfs cannot reach).
7. **Cache-coherency snoop-broadcast latency tail** (would need precise cross-CCX clock-aligned probes).

Categories 1–5 require **new instrumentation** (mic / SDR / IR / CT clamp /
oscilloscope) — out of scope this round. Categories 6–7 are reachable but
require below-driver access (matches the FPGA-pivot recommendation).

## False-positive trap to watch for

3/4 oracles flagged **B24 (power×temp lag covariance)** as the most likely
false-positive in our top-10. Reason: the slope is dominated by chassis
airflow + TIM contact resistance, not by die-level electrical RC. Without
an independent on-die temperature reference, B24 cannot separate package
thermal mass from die electrical time constant.

Our quick-probe data confirmed this concern: ikaros peak xcorr lag was
−1 (essentially zero; instrument-limited) while daedalus showed +17×100 ms.
That 18×100 ms gap is almost certainly the **GPU idle-power gap** (18 W
ikaros vs 4 W daedalus = different operating points) rather than per-die
thermal-electrical impedance.

## Methodological gap (within-machine across power-cycle tests)

Unanimous: we have *never* run twin-of-self tests. Three mechanisms whose
within-vs-between answer would falsify current framing:

- **B16 NBTI/HCI degradation drift** — if within-machine drift exceeds
  between-machine, our "per-die" channels are actually *operational-history*
  channels.
- **B15 DRAM retention tail** — same logic, environmental binding.
- **B14 NVMe wear-level GC state** — discriminates board-state from die.

Action: run all existing 14 channels twice on the same machine across a
clean reboot, before claiming silicon-binding. This is the cheapest single
experiment to falsify the current "we have discovered hardware identity"
claim.
