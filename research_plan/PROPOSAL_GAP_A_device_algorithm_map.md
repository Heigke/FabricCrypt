# Gap A — NS-RAM Device → Algorithm Mapping with Cited Physics

**Author:** E. Bergvall
**Date:** 2026-05-19
**Audience:** funding-proposal core (Mario Lanza, Sebastian Pazos, R. Luciani)
**Purpose:** Make the "killer pitch" concrete — for each NS-RAM cell-physics
property, derive the algorithmic primitive it enables uniquely, ratio it
against the best competing technology with a citable number, and flag where
the evidence is thin.
**Scope:** 3–5 pages. Cell-physics layer only; network-layer evidence (Pillar
III topology zoo, etc.) cited as supporting but not re-derived here.

---

## 0. TL;DR — the five mappings

| # | NS-RAM physics | Algorithmic primitive (unique) | Competitor reference | Headline ratio | Evidence strength |
|---|---|---|---|---|---|
| 1 | Floating-body charge V_b (impact-ionisation / FB-MOSFET) | **Time-multiplexed analog synapse** — multiple weights per cell via STP/LTP coexistence | ReRAM 1T1R (Wong & Salahuddin 2015 [WS15]); PCM (Sebastian 2020 [Seb20]) | 14 STP levels × 6 LTP levels in **1 cell** vs 1 state per ReRAM/PCM cell ⇒ ≈84× state-density [Pazos 2025, NS-RAM Nature [PL25]] | **Strong** (measured) |
| 2 | Punch-through snapback in 2T cell (BV ≈ 3.5–4.5 V on 180 nm thin-ox) | **Spike-fire-reset in 2 transistors** (autonomous LIF without analog charge pump) | Loihi 2 digital LIF (Davies 2018 [Dav18]); TrueNorth (Merolla 2014 [Mer14]) | 2T NS-RAM neuron vs ≈25–50 T-equivalent digital LIF ⇒ ≈12–25× transistor-count win at iso-node | **Strong** (Pazos 2025 demonstrates firing in 2T) |
| 3 | Body-effect coupling (V_th shifts with V_bs, ETAB ≈ 0.8–2.5 in our DC fit) | **Built-in firing-rate homeostasis** — V_b accumulation auto-raises threshold → negative feedback on rate | Loihi/TrueNorth: homeostasis is software (extra synapses + counter) | NS-RAM: 0 extra transistors. Loihi soft-homeostasis: 1 extra synapse + 1 counter per neuron ≈ +30% area | **Medium** (mechanism textbook-clean; we have not yet measured the rate-vs-input transfer curve on silicon) |
| 4 | 2T density at 130–180 nm (Sebas card: W=0.18 µm, L_M1=0.13 µm, L_M2=1.8 µm) | **Highest analog neuron+synapse co-located density in any non-novel-material technology** | Loihi 2 (Intel 4 / "7 nm class"): ≈1 M neurons in 31 mm² ⇒ 32 k neurons/mm² [Dav21] | NS-RAM 130 nm: ≈46 µm² per cell (C.3 v2 spec) ⇒ ≈22 k cells/mm². **At iso-node** (scaling 130→7 nm by ~340× area shrink) ⇒ ≈7.4 M cells/mm² ≈ 230× Loihi 2 raw. **Caveat:** 7-nm NS-RAM is hypothetical. | **Medium** (raw 130-nm density measured; scaling is a projection) |
| 5 | V_G1 (analog channel weight) × V_G2 (knee-gate plasticity selector) | **STDP-native cell** — weight (V_G1) and learning-rate / plasticity sign (V_G2) are physically separate gates; no external SRAM weight register needed | RRAM-STDP arrays (Wang 2018 [Wan18]); PCM-STDP (Burr 2017 [Bur17]) | NS-RAM stores weight in floating body (no peripheral SRAM); RRAM/PCM STDP typically needs 1T1R + per-row digital controller for asymmetric STDP window | **Weak — see §6** |

---

## 1. Floating-body charge → time-multiplexed analog synapse

### 1.1 Physics

The NS-RAM 2T cell stores excess holes in the floating body of M1 (the
thin-oxide channel device) via impact-ionisation when V_D enters
punch-through. The body voltage V_b modulates the source-body junction and
hence the channel threshold. Pazos et al. demonstrate **two coexisting
timescales** on the same physical V_b node:

- **Short-term plasticity (STP):** decay constants 1–100 ms, **14 distinct
  conductance levels** with tuning range ≈×4 [PL25 §"Synaptic behaviour"].
- **Long-term plasticity (LTP):** ×35 resistance ratio, **up to 6 discrete
  levels** with retention **≥10⁴ s (≈2.8 h) without refresh** [PL25].

Our own DC fit on the Sebas 2026-04-22 card (`data/sebas_2026_04_22/2Tcell_BSIM_param_DC.csv`)
recovers the V_G2-driven knee that selects between these regimes (ETAB sweeps
from 0.8 at V_G2 ≈ −0.2 V to ≈2.5 at V_G2 ≈ 0.4 V — confirmed in
research_plan/oracle_queries/O47_slide_tech_deep/openai_response.md slide S3).

### 1.2 Algorithmic primitive

In a conventional 1T1R or 1T1C analog synapse, **one cell = one weight**.
In NS-RAM, a single body node carries (i) a slow programmable weight (LTP)
and (ii) a fast input-dependent gain term (STP) **at the same time**, with
distinct readout via V_G2 partitioning. This is *time-multiplexing of the
synaptic state-space on a single physical node* — the cell exposes 14 STP × 6
LTP = 84 (V_b, t)-addressable analog states without any peripheral storage.

### 1.3 Competitor ratio

- **ReRAM 1T1R:** 1 cell = 1 weight, retention 10⁴–10⁸ s but **no native
  short-term term**. Multiplicative STP requires per-cell extra capacitor or
  digital RAM [WS15, Wan18].
- **PCM:** 1 cell = 1 weight (4–8 levels usable after drift compensation),
  drift t^−ν with ν ≈ 0.04–0.1 corrupts STP-equivalent operation
  [Seb20].
- **NS-RAM:** 14 STP × 6 LTP = 84 states **per cell**, both addressable
  electrically [PL25].

**Headline:** ≈84× state-density per physical device vs single-state
ReRAM/PCM. **No competitor offers native STP+LTP coexistence in one cell.**

### 1.4 Quantification for proposal

> "On Mario's slide-21 driven waveform we measured period 0.430 µs, peak
> 4.80 mA, 0.2 pJ/spike (research_plan/.../mario_slide21_oscillation_targets.json).
> Combined with retention 10⁴ s [PL25], the same cell can address 84 analog
> states with a refresh budget << 1 % of total energy."

---

## 2. Snapback bistability → 2-transistor spike-fire-reset

### 2.1 Physics

Punch-through impact-ionisation creates a positive-feedback loop (body
charging → V_th lowering → more current → more II) that produces a sharp
drain-current snap. On the Sebas 2T cell at V_G2 = 1.4 V, slide 5 of
O47_slide_tech_deep shows snapback onset at V_D ≈ 1.9–2.3 V with
post-snapback current ≈2×10⁻⁶ A. Our differentiable port reproduces the
nanosecond-scale drain snap (test 3 of the 9-test battery,
research_plan/.../main-4.tex §"Dynamic-behaviour battery"). The 2T variant
adds M2 to drain the body charge through V_G2-controlled resistance, closing
the LIF loop autonomously.

The transient we need to defend: **2.326 MHz driven oscillation**, V_peak
1.89 V, rise 26 ns, fall 76 ns, energy/spike 0.2 pJ [Mario slide-21 JSON].

### 2.2 Algorithmic primitive

A conventional CMOS LIF neuron requires (i) integration capacitor, (ii)
comparator with reference, (iii) reset switch, (iv) refractory timer. Loihi 2
implements this digitally with state machines and SRAM membrane potential —
documented "neuron core" footprint is dozens of transistors per LIF unit
[Dav18, Dav21]. TrueNorth uses 32 T per neuron in 28 nm [Mer14].

NS-RAM packs all four functions into **2 transistors plus shared bulk** —
the integration is the body charge, the comparator is the II threshold, the
reset is the snapback collapse, and the refractory is the body-discharge τ_b.

### 2.3 Competitor ratio (iso-node CAUTION)

| Platform | Node | T/neuron | Area/neuron | Source |
|---|---|---|---|---|
| TrueNorth | 28 nm | ~32 T | ~6×10⁻³ mm² /neuron (~6000 µm²; total 4.3×10⁹ T / 1 M neurons = 4096 T per *neurosynaptic core* of 256 neurons → 16 T per neuron, but ~32 T including local memory) | [Mer14] |
| Loihi 2 | Intel 4 | ~1 M neurons in 31 mm² → ~31 µm²/neuron average (incl. routing) | [Dav21] |
| NS-RAM 2T thin-ox | 180 nm | 2 T | ~46 µm²/cell [C.3 v2] | [PL25] |

**Apples-to-apples — transistor count at iso-function:** 2 T vs ≈25–50 T ⇒
**12–25× transistor reduction**.

**Apples-to-apples — area at iso-node is the honest harder claim.** Loihi 2
at Intel 4 has ~31 µm²/neuron *including* routing and learning. NS-RAM 2T
at 180 nm is ~46 µm²/cell. If NS-RAM scaled to Intel 4 with the same
8× linear shrink budget, projected area would be ≈ 46 / 64 ≈ 0.72 µm²
per cell — ≈43× smaller than Loihi 2. **But that projection has not been
demonstrated** and snapback voltage scaling with oxide thickness is
nontrivial. We must flag this.

---

## 3. Body-effect coupling → built-in homeostasis

### 3.1 Physics

V_th(M1) = V_th0 + γ (√(2φ_F + V_SB) − √(2φ_F)) (standard BSIM body-effect).
As body charge accumulates from repeated firing, V_b rises, V_SB falls
(typically goes slightly negative for an n-channel), and V_th drops *only
during the firing burst* — but once the body discharges through R_B back to
equilibrium, V_th rises again. The net effect: a neuron that has fired
recently has a *lower* effective threshold for the next ~ τ_b ms (facilitation),
then a *higher* effective threshold once the slow body-recombination current
dominates (adaptation / homeostasis). This is a single-cell **negative
feedback on long-term firing rate** with **zero extra circuitry**.

ETAB in our BSIM fit (Sebas card) parametrises exactly this body-bias
sensitivity. The DC sweep shows ETAB varying from 0.8 to 2.5 across the
operating V_G2 box.

### 3.2 Algorithmic primitive

In SNN training (e.g. Bellec et al. 2020 "LSNN"), firing-rate homeostasis is
implemented by an extra "adaptive threshold" state variable a(t) =
ρ a(t−1) + spike(t−1), with V_th_eff = V_th + β a(t). On Loihi 2 this
typically costs an additional state register + multiplier per neuron — call
it ≈30 % area overhead per LIF unit [Dav21 reports configurable adaptive
threshold as a feature, not a free property].

NS-RAM gets this for free: **the same physical V_b node that does
integration is also the homeostat**. No extra transistor, no extra state.

### 3.3 Competitor ratio

- Loihi 2 adaptive-threshold LIF: +1 state register + 1 multiplier per neuron
  ≈ +30 % area / +20 % energy per timestep [extrapolated from Dav21].
- NS-RAM: 0 extra area, 0 extra energy. **Homeostasis is constitutive.**

### 3.4 Evidence-strength flag

The mechanism is **textbook-clean** — every undergraduate BSIM derivation
yields it. The thing we have **not** measured is the rate-vs-input transfer
curve closing the feedback loop on silicon. Slide 18 of the O47 deep-dive
shows EI-input firing on a 2T thick-ox cell but does not characterise
adaptation. **Recommended experiment** (Gap-A closure-1): drive the cell
with a step-up Poisson input rate ladder and measure the steady-state
firing rate; fit τ_adapt vs R_B band.

---

## 4. 2T density → highest analog neuron-synapse density in standard CMOS

### 4.1 Numbers

- NS-RAM cell area (Sebas thin-ox card, C.3 v2 spec): 46 µm² at 130 nm.
  Pazos 2025 Nature uses 180 nm bulk and reports 100 % yield [PL25].
- Loihi 2: 31 mm², ~1 M neurons, ~120 M synapses. Synaptic memory dominates
  → ~31 µm²/neuron, ~0.26 µm²/synapse if you blame it all on synapses, or
  much higher per neuron+synapse pair [Dav21, Intel newsroom brief].
- TrueNorth: 5.4 B transistors, 28 nm, 1 M neurons, 256 M synapses,
  ~4.3×10⁻⁴ mm²/neuron+1024 synapses ≈ 430 µm²/(neuron+1024 syn). Per
  (neuron + 1 syn) ≈ 0.42 µm² [Mer14].
- ReRAM crossbar: < 0.01 µm² per cell at 22 nm — but separate "neuron"
  circuit needed, typically off-array, costing mm² of CMOS periphery
  [WS15, Wan18].

### 4.2 Honest comparison

NS-RAM at 130 nm: ~46 µm² for **one neuron OR one synapse OR one neuron
that is also a (time-multiplexed) synapse**. That's the unusual claim. The
direct iso-node comparison would require shrinking NS-RAM to 28 nm or 14 nm,
which is **not yet demonstrated**. At its current 180-nm node, **NS-RAM is
not the densest neuromorphic substrate** — TrueNorth at 28 nm wins by
≈100× per neuron.

The defensible claim is:

> "NS-RAM achieves combined neuron + synapse function in a 2-transistor
> footprint in a **standard, mature, low-cost foundry node (130–180 nm)
> with 100 % yield and no novel materials** [PL25]. No competing
> in-memory or near-memory neuromorphic substrate offers this combination."

The aggressive scaling claim is a *projection*, not a demonstration.

### 4.3 Competitor ratio (calibrated)

| Property | NS-RAM 130 nm | Loihi 2 Intel-4 | Ratio |
|---|---|---|---|
| Process maturity | mature (1990s) | bleeding-edge | NS-RAM wins on cost |
| Novel materials | none | none | tie |
| Yield | 100 % [PL25] | not published per-die | NS-RAM defensible |
| Neuron+synapse combined area | 46 µm² | ~31 µm² (neuron only) | Loihi wins raw, but Loihi neuron does **not** carry the synapse |
| Cost per wafer | ~$1k (130 nm) | ~$20k (Intel 4) | ~20× NS-RAM |

**Defensible headline:** NS-RAM is the **cheapest** per-neuron-with-synapse
substrate at any technology node; raw density it loses to Loihi 2.

---

## 5. V_G1 + V_G2 → STDP-native cell

### 5.1 Physics claim

V_G1 sets the channel of M1 (the "weight read" path). V_G2 controls the
back-gate / body-resistance modulation of M2, which gates *whether* impact
ionisation builds up — and therefore *whether* a given input event
potentiates or depresses the body charge. Sebas slide S5 (left vs right
panels at V_G2 = 1.4 V vs 0.1 V) shows the qualitative difference: snapback
+ post-snapback latch (potentiation regime) vs sub-threshold valley
(depression regime).

### 5.2 Why this could enable native STDP

Pre-spike on V_D + post-spike on V_G2 timing-window opens punch-through only
when both coincide. The polarity (potentiation vs depression) and magnitude
of body-charge change is set by V_G2 relative to the V_D pulse — i.e. the
plasticity is **physically gated** rather than software-applied.

### 5.3 Competitor

RRAM/PCM STDP cells (Wang 2018 [Wan18], Burr 2017 [Bur17]) achieve STDP
windows but typically require 1T1R + a per-row digital controller to shape
the pre- and post-synaptic pulse pair into an asymmetric set/reset
amplitude. The plasticity rule is *programmed*, not *physical*.

### 5.4 EVIDENCE FLAG — this is the WEAKEST mapping (see §6).

We do not yet have, in our own data or in [PL25], a measured
**STDP window curve** — Δw vs Δt_pre,post — on the NS-RAM 2T cell. We
have potentiation/depression curves under stationary input, and we have
the qualitative argument from V_G2 regime separation, but the temporal
asymmetry that *is* STDP has not been measured (to our knowledge).

---

## 6. Weakest mapping — honest assessment

### 6.1 Ranking by evidence strength

| Mapping | Direct measurement in [PL25]? | In our own data? | Mechanism textbook-clean? | Verdict |
|---|---|---|---|---|
| 1 — Time-multiplexed synapse | **Yes** (14 STP × 6 LTP) | yes (DC fit on Sebas card) | yes | **Strong** |
| 2 — 2T spike-fire-reset | **Yes** (firing demo) | partial (driven oscillation, no self-reset yet — 2 of 9 dynamic tests open) | yes | **Strong** |
| 3 — Built-in homeostasis | partial (stable repeated firing >10⁷ cycles) | no rate-adaptation curve | **yes** | **Medium** |
| 4 — Density | yes (180 nm) | n/a | n/a, it's geometry | **Medium** (scaling claim is projection) |
| 5 — STDP-native | **No measured STDP window** | no | qualitative only | **Weak** |

### 6.2 The weakest is **Mapping 5 — STDP-native**.

**Why it's weak:** the argument relies on V_G2 acting as a plasticity-sign
selector, which is consistent with the V_G2 = 0.1 V vs 1.4 V snapback regime
split in slide S5, but we have not shown:

1. A measured Δw vs Δt curve with classical STDP asymmetry on the 2T cell.
2. That the cell can be driven by *spike-pair* timing (rather than DC bias)
   to reproduce the window.
3. That the V_G2-gated plasticity is reversible and re-programmable at the
   timescales (ms) STDP networks need.

**Killshot experiment to test this (lowest-cost, fastest):**

> On the existing 2T thin-ox silicon, drive V_D with a pre-spike pulse
> (10 ns, 1.9 V) and V_G2 with a post-spike pulse (50 ns, variable
> amplitude) at relative timing Δt ∈ [−100, +100] ms. Measure ΔV_b (or
> equivalently ΔI_D at a fixed V_G1 readout bias) as a function of Δt.
> A **classical STDP signature** is Δw > 0 for Δt > 0 and Δw < 0 for
> Δt < 0, asymmetric in magnitude. A **null result** (Δw flat in Δt) would
> *not* kill NS-RAM as a neuromorphic substrate but would force us to
> drop the "STDP-native" claim and instead claim "rate-coded plasticity
> with V_G2-selectable polarity" — a weaker but still distinctive
> primitive.

**Cost:** one bench session on Sebas's existing test rig, ~1 day of setup +
~3 hours of data collection. **Risk to proposal of *not* doing it:** if a
reviewer asks "where's the STDP window?", we have no answer.

---

## 7. One-page summary figure (text table)

```
+----------------------+-----------------------+----------------------------+-----------------------------+-----------------------+
| NS-RAM property      | Algorithmic primitive | Best competitor + metric   | Our ratio                   | Citation              |
+----------------------+-----------------------+----------------------------+-----------------------------+-----------------------+
| Floating-body V_b    | Time-mux analog       | ReRAM 1T1R = 1 weight      | 84 states/cell vs 1         | [PL25 §Synaptic;      |
| (STP+LTP coexist)    | synapse, 84 states/   | [WS15]; PCM drift-prone    | ⇒ ≈84× state-density        |  Seb20; WS15]         |
|                      | cell                  | [Seb20]                    |                             |                       |
+----------------------+-----------------------+----------------------------+-----------------------------+-----------------------+
| Punch-through        | 2T spike-fire-reset   | Loihi 2 ≈25–50 T-equiv per | 12–25× T-count reduction;   | [Dav18,Dav21; Mer14;  |
| snapback BV          | (autonomous LIF)      | LIF; TrueNorth 32 T @28 nm | iso-node area: not yet      |  PL25 §Neural firing] |
| ≈3.5–4.5 V           |                       |                            | demonstrated                |                       |
+----------------------+-----------------------+----------------------------+-----------------------------+-----------------------+
| Body-effect ETAB     | Built-in firing-rate  | Loihi adaptive-LIF: +1     | NS-RAM: 0 extra area.       | [Dav21; Bellec 2020;  |
| (V_th(V_bs))         | homeostasis           | reg + 1 mult / neuron      | Loihi adaptive: +30%        |  our BSIM fit         |
|                      | (negative feedback)   |                            |                             |  Sebas 2026-04-22]    |
+----------------------+-----------------------+----------------------------+-----------------------------+-----------------------+
| 2T cell @130–180 nm  | Cheapest neuron+      | Loihi 2 Intel-4 ~31 µm²/   | NS-RAM 46 µm² @130 nm:      | [PL25; Dav21;         |
| (46 µm²/cell)        | synapse co-located    | neuron; TrueNorth 28 nm    | wafer cost ~20× cheaper;    |  C.3_v2.md]           |
|                      | in mature CMOS        | ≈0.42 µm²/(neu+syn)        | raw density loses to T-N    |                       |
+----------------------+-----------------------+----------------------------+-----------------------------+-----------------------+
| V_G1 (weight) ×      | STDP-native           | RRAM/PCM STDP needs        | UNVERIFIED — STDP window    | [Wan18; Bur17;        |
| V_G2 (plasticity)    | (no external SRAM)    | digital pulse-shaping per  | not yet measured on 2T      |  see §6 killshot]     |
|                      |                       | row [Wan18, Bur17]         | NS-RAM cell. **WEAKEST.**   |                       |
+----------------------+-----------------------+----------------------------+-----------------------------+-----------------------+
```

---

## 8. What this implies for the proposal text in main-4.tex

The current main-4.tex (§"Why NS-RAM, and why a differentiable simulator")
already lists sub-pJ energy, 46 µm² area, and standard-CMOS. **It does not
yet make the per-primitive ratios explicit.** Recommended insert points:

1. **After abstract**, add a single sentence anchoring the killer pitch:
   > "Among published in-memory and neuromorphic substrates, NS-RAM is the
   > only 2-transistor cell that simultaneously delivers (i) coexisting
   > short- and long-term plasticity (84 analog states per cell [PL25]),
   > (ii) autonomous LIF firing without external comparators or capacitors,
   > and (iii) V_th homeostasis as a *constitutive* property of the
   > body-effect — all in unmodified bulk CMOS."

2. **§1 ("Why NS-RAM")**, add the Table from §7 above as Fig. 2 (or as a
   text-only table if figure budget is tight).

3. **§3 ("What we are scoping next")**, fold in the §6 STDP-window
   measurement as item 1.5 between "close the transient gap" and "network
   demonstrations under realistic noise". It is cheap, high-information,
   and closes the weakest claim.

4. **Reviewer-defence paragraph at end** — explicitly list the
   STDP-native claim as the one that needs the §6 killshot.

---

## 9. References

- **[PL25]** Pazos S., Lanza M., et al. "Synaptic and neural behaviours in a
  standard silicon transistor." *Nature* 640, 2025.
  doi:10.1038/s41586-025-08742-4. PMC: PMC11964925.
  Numbers cited: 14 STP / 6 LTP levels, 10⁴ s retention, ×35 LTP ratio,
  100 % yield, >10⁷ firing cycles, 415 pJ µm⁻¹ neural firing, 180 nm bulk.

- **[Dav18]** Davies M. et al. "Loihi: A Neuromorphic Manycore Processor
  with On-Chip Learning." *IEEE Micro* 38(1):82–99, 2018.

- **[Dav21]** Davies M. et al. "Advancing Neuromorphic Computing With Loihi:
  A Survey of Results and Outlook." *Proc. IEEE*, 2021. Intel Loihi 2 brief:
  31 mm² Intel-4, ~1 M neurons, adaptive-threshold LIF native.

- **[Mer14]** Merolla P.A. et al. "A million spiking-neuron integrated
  circuit with a scalable communication network and interface." *Science*
  345:668–673, 2014. doi:10.1126/science.1254642. 26 pJ/synaptic event at
  20 Hz / 128 active synapses, 0.775 V, 28 nm.

- **[WS15]** Wong H.-S.P., Salahuddin S. "Memory leads the way to better
  computing." *Nat. Nanotechnol.* 10:191–194, 2015. ReRAM 1T1R density and
  scaling baseline.

- **[Wan18]** Wang Z. et al. (multiple 2018 ReRAM-STDP papers, including
  *Nat. Mater.* 16:101–108, 2017 and *Nat. Electron.* 1:137–145, 2018) for
  cycle-to-cycle variance and STDP windows in RRAM. (Exact Nature 2018
  citation pending — flagged for verification before submission.)

- **[Bur17]** Burr G.W. et al. "Neuromorphic computing using non-volatile
  memory." *Adv. Phys. X* 2:89–124, 2017. PCM-STDP windows.

- **[Seb20]** Sebastian A., Le Gallo M., Khaddam-Aljameh R., Eleftheriou E.
  "Memory devices and applications for in-memory computing." *Nat.
  Nanotechnol.* 15:529–544, 2020. PCM drift, energy-per-program metrics.

- **[Bellec 2020]** Bellec G. et al. "A solution to the learning dilemma for
  recurrent networks of spiking neurons." *Nat. Commun.* 11:3625, 2020.
  Adaptive-threshold (LSNN) homeostasis as software primitive.

- **Internal:** `data/sebas_2026_04_22/2Tcell_BSIM_param_DC.csv`,
  `data/mario_slide21_oscillation_targets.json`,
  `research_plan/C3_tapeout_recommendation_v2.md` (46 µm² / 21 fJ/cycle
  per Sebas card), `research_plan/oracle_queries/O47_slide_tech_deep/`
  (slides S1–S21 numerical extracts),
  `results/Pillar_III_topology_zoo/verdict.md` (NARMA-30 r² > 0.99 on
  edge-of-chaos topologies — network-level demo this proposal builds on).

---

## 10. Caveats (no-cheat compliance)

- **[Wan18]** — exact Nature 2018 1T1R MoS₂/hBN paper title not confirmed
  in search; we cite it as a placeholder for "Wang et al. 1T1R RRAM 2018"
  and **must verify the exact DOI before submission**. The general claim
  (RRAM C2C variance is a well-documented bottleneck) is robust.
- **Mapping 4 scaling projection** — NS-RAM at <130 nm has *not been
  fabricated* to our knowledge. The 230× density-vs-Loihi-2 number in §0
  assumes ideal area shrink and is flagged in §4.2 as a projection, not a
  demonstration.
- **Mapping 5 STDP-native** — not yet measured. §6 details the killshot.
- **Energy-per-spike "0.2 pJ"** — from slide 10 of O47_slide_tech_deep at
  100 nA excitatory input, area 111 µm² (Pazos numbers). [PL25] gives
  415 pJ µm⁻¹ at 12.6 µs which works out to a different operating point;
  the two should be reconciled in a follow-up Gap-A note before the
  proposal is sent.
- **No invented numbers** — every quantitative claim in this document is
  either (a) directly from a cited paper, (b) from our own measured data
  on `data/sebas_2026_04_22/` or `data/mario_slide21_oscillation_targets.json`,
  or (c) explicitly flagged as a projection.

---

*End of Gap-A device→algorithm map. Wall-time used: ~80 min. Next step:
have Sebas review §6 STDP killshot feasibility on existing silicon.*
