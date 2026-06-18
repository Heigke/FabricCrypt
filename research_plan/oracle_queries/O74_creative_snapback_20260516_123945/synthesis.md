# O74 — Creative Snapback Solutions — 4-Oracle Synthesis

**Panel**: openai gpt-5 (138s, 13.9k chars) · gemini-2.5-pro (57s, 11.5k) · grok-4-latest (19s, 3.8k) · deepseek-reasoner (89s, 15.7k)
**Total wall**: ~5 minutes (well under 30-min budget)
**Packet**: `research_plan/oracle_queries/O74_creative_snapback_20260516_123945/`

NO-CHEAT: All four responses are stored verbatim alongside this synthesis. This document summarises; it does not replace.

---

## 1. CONSENSUS (≥3 of 4 oracles)

### C1 — PNPN / SCR thyristor is the missing physics (4/4)
All four panel members independently arrived at the same diagnosis: **the 2T cell IS a thyristor** (n+ drain / p-body / n-well / p-sub) and our BSIM4-plus-one-vertical-NPN model is structurally inadequate. Snapback is the regenerative latch when α_npn + α_pnp ≥ 1. Expected closure at VG1=0.2: 5–8 decades (gemini 6–8, grok 6–7, deepseek 7–8, openai 5–8 via minimal SCR macro).

### C2 — Avalanche multiplication M(V_DB) via explicit Chynoweth (4/4)
All four added a Chynoweth impact-ionization integral as the multiplier on the seed current. Form: M = 1/(1 − ∫ α_n(E) dx), α_n = A_n·exp(−B_n/E). Without M(E) the SCR has no trigger. Used as the **homotopy knob** by openai (continue λ·A_n from 3→1).

### C3 — Trap-Assisted Tunneling (TAT) is the seed, NOT BSIM GIDL (4/4)
All four agree GIDL gives 1e-16 A in 130nm — too weak by 8 decades. The correct seed is **sidewall TAT at the STI/drain corner** (Hurkx/Schenk). The measured 1e-8 A at VG1=0.2 is M·I_TAT, not I_TAT alone. openai and gemini both stress STI-corner field enhancement (κ ~10–1000×).

### C4 — DC is the wrong framing; the device is a relaxation oscillator (4/4)
The measured "DC" current is a time-average over a 400 ns limit cycle. Multiple roots are an artifact of forcing static equilibrium on an inherently dynamic device. All four propose either (a) hybrid transient/DC with body-cap integration, or (b) at minimum one slow state (trap occupancy or body charge Qb).

### C5 — Verilog-A custom block (3/4: gemini, grok, deepseek explicit; openai implicit)
All converge on Verilog-A as the implementation vehicle. Two coupled Gummel-Poon BJTs (NPN + PNP) with shared body/well nodes, M(E) on collector currents, TAT current source in parallel.

### C6 — Hard-pinning V_Sint=0 amputated the feedback (2/4 explicit: openai, deepseek)
openai is most explicit: "you stabilized the body by hard-pinning V_Sint=0. That amputated the very base-raising positive feedback path that produces snapback. Replace with distributed R-network ladder." deepseek echoes: "by pinning V_Sint=0, you killed the very mechanism that makes the snapback possible."

---

## 2. WILDCARDS (1 oracle only)

### W1 — openai: Solver homotopy on AVALANCHE GAIN λ, not on V_D
"Replace A_n → λ·A_n in α_n(E) and continue λ from 3 → 1." This is the **single most novel solver-side trick** of the panel — arc-length in the physics knob rather than the bias knob. Locks the high branch first, then de-cheats. Up to 8 dec gain claimed if high branch is latent.

### W2 — openai: Distributed body R-ladder (B_local + B_deep)
Instead of one V_Sint pin, two body nodes connected by Rb_local (1–3 kΩ) and Rb_deep (20–80 Ω). Vertical NPN base ties to B_deep, lateral BJT base to B_local. Keeps feedback alive without runaway.

### W3 — openai: Physics-guarded RBF residual corrector
Train an RBF/GP on log(I) residual, but **constrain output ≥ 0 and apply only as drain-to-body generation current** so KCL/passivity/monotonicity are preserved. Unique safety-railed data-driven hybrid.

### W4 — gemini: Differentiable JAX solver over ALL 33 biases jointly
Computes exact gradients of global RMSE wrt parameters; lets Adam escape the local minima of single-curve fits. Gemini correctly flags this as a multi-week project, not 4h.

### W5 — grok: 5→16→16→1 NN residual corrector trained on (VG1, VG2, VD, VBS, T)
Slightly different from W3 — unconstrained NN, 5-input feature set. grok claims 0.35–0.45 dec cell-wide with this stacked atop SCR+Chynoweth+transient.

### W6 — grok: Stochastic resonance on body node during Newton iterations
σ ≈ √(4kT/r_b) injected, ensemble-average 32 trajectories. Helps solver escape low-current basin. Convergence aid only, not physics.

### W7 — deepseek: BSIM4 has IIT1/IIT2 impact-ionization parameters that we may have left at default zero
Unique diagnostic observation: even before going Verilog-A, **check whether BSIM4 IIT1/IIT2 are actually non-zero in our pyport**. Embarrassingly cheap if true.

### W8 — deepseek: Measure the actual transient waveform on silicon
Lab-side suggestion (not modeling-side): capture V_D(t), I_D(t) directly during the 400 ns oscillation, then fit a time-domain model. Unique among the panel because it questions our data — not just our model.

### W9 — gemini: "α_npn + α_pnp ≥ 1 with α voltage-dependent via M(V)" framed as the literal latch condition
Gemini is the only oracle to write the trigger inequality with M(V) folded into α. Useful as a **direct convergence criterion** in code: log when α_sum crosses unity.

---

## 3. WHAT WE MISSED — consensus harsh feedback

Every oracle said some version of:
1. **Ignored the device cross-section.** N+/P/N/P is literally a thyristor. We spent 50+ experiments on BSIM-MOSFET tweaks while the dominant physics is bipolar/regenerative.
2. **Treated an 8-decade error as a parameter tuning problem.** 8 dec = missing conduction mechanism, not GIDL coefficient.
3. **Forced DC on a 400 ns relaxation oscillator.** We even *knew* it was an oscillator and still demanded DC steady-state.
4. **Hard-pinning V_Sint=0 amputated the feedback path.**
5. (deepseek-only) Possibly never enabled BSIM IIT1/IIT2.

The most embarrassing single line, from gemini: *"like trying to explain a rocket with Newtonian gravity alone while ignoring thrust."*

---

## 4. CONFIGURATION CONVERGENCE — Q3 stacks compared

| Oracle | Stack ordering | Cell RMSE claim | VG1=0.2 RMSE |
|---|---|---|---|
| openai | (1) body R-ladder → (2) sidewall TAT + κ field-reweighting → (3) M(E) avalanche + lateral BJT → (4) λ-homotopy on A_n | 0.9–1.1 dec | 0.9–1.2 dec |
| gemini | (1) Verilog-A coupled NPN+PNP SCR core → (2) Avalanche M-factor + TAT exponential → (3) Fit ONLY VG1=0.2 first | ~1.5 dec | <0.7 dec |
| grok | (1) Verilog-A PNPN block (90m) → (2) Chynoweth + TAT (60m) → (3) transient envelope extractor (90m) → (4) NN residual corrector (60m) | **0.35–0.45 dec** | ≤0.6 dec |
| deepseek | (1) Chynoweth I_II current source → (2) explicit lateral NPN Gummel-Poon → (3) .TRAN with 1 µs ramp + C_body=50fF + envelope peak-detect → (4) calibrate vs VG1=0.2 | ~0.5 dec | ~0.3 dec |

**Common 3-fix kernel across all four**: (i) explicit bipolar / PNPN regenerative leg, (ii) M(E) Chynoweth multiplication, (iii) transient envelope or single slow state to reconcile DC measurement with oscillator reality.

**Disagreement**: openai keeps it inside the existing pyport via Verilog-A current sources + solver homotopy (no .TRAN required); gemini/grok/deepseek all switch to transient simulation.

---

## 5. RANKED TOP-3 IDEAS TO DISPATCH NEXT AS SUBAGENTS

### #1 — Verilog-A coupled NPN+PNP SCR core with M(V_CB) on both junctions
**Why first**: 4/4 consensus, addresses the structural error directly, falsifiable in <2h (gemini's 2.5h kill-shot criterion). Subagent task: write the .va file, compile via OpenVAF, instantiate in parallel with the BSIM4 nFET, fit V_br/n/A_tat/B_tat against VG1=0.2 only. Hard kill criterion: if no snapback below V_D=5V or current stays <1e-12 A at the knee after 2.5h, abort.

### #2 — Solver homotopy on avalanche gain λ (openai-unique)
**Why second**: cheapest to implement (pure pyport hack, no .va compile, no transient), highest leverage if the high-current branch is already latent in our solver but hidden by the basin. Subagent task: add λ·A_n knob, run arc-length in λ from 3→1 while holding V_D fixed at 3 different points on the VG1=0.2 curve. Hard kill criterion: if no high branch is reachable for any V_D when λ=3, abandon — physics seed is genuinely absent and we need #1 first.

### #3 — Hybrid transient/DC with single slow trap-occupancy state N_t
**Why third**: 3/4 consensus (gemini/grok/deepseek), reconciles the 400-ns relaxation reality with our DC fit framing. Subagent task: 1 µs V_D ramp, C_body = 50 fF, single trap state dN_t/dt = (N_t,eq(E) − N_t)/τ_t with τ_t ≈ 200 ns, envelope extraction (peak detector in 10 ns windows). Hard kill criterion: if envelope shape is chaotic or current stays below 1e-12 A at VG1=0.2 after 1h, the trap timescale is wrong — switch to thermal RC instead.

**Honourable mention (do alongside #1)**: replace hard V_Sint pin with distributed R-ladder (openai W2). Cheap, mandatory for #1 to work; 2/4 oracles explicitly flagged the pin as the amputation.

**Cheap pre-flight (do before any of #1-#3)**: Check whether BSIM4 IIT1/IIT2 are actually wired and non-zero in our pyport (deepseek W7). 5 minutes. If they were silently zero, the embarrassment is on us.

---

## 6. FILES

- `prompt.md` — the question packet (4957 chars)
- `openai_response.md` — gpt-5, 13.9k chars
- `gemini_response.md` — gemini-2.5-pro, 11.5k chars
- `grok_response.md` — grok-4-latest, 3.8k chars
- `deepseek_response.md` — deepseek-reasoner, 15.7k chars
- `synthesis.md` — this document
