# FUNDAMENTAL PHYSICS HUNT — 2026-05-20

**Hypothesis:** the persistent 1.163 dec DC fit gap (after 5 ngspice bugs fixed, IIMOD card patched, pdiode+well_diode topology corrected, Theta0_n + phi + eta_sigmoid + Vth/tox + BJT mbjt fixes, IFT sign-bug patched) is too large to be a parameter problem. We are likely missing a **constitutive physics term**.

## Goal
Close DC≤0.5 dec on Sebas 33-bias data **with mechanistically defensible physics**, not parameter tuning. If we can't close it without Sebas's new data, identify which single experiment (variant cell, T-sweep, transient) would force the issue.

## Five gaps to lift (Eric's framing)
1. DC≤0.5 dec across all 33 biases
2. Transient calibrated against ngspice synthetic + Sebas (when arrives)
3. Floating-body self-consistency (C2 robust, ALL biases)
4. Sub-threshold (VG1≤0.4 V) regime stable, no NaN
5. Cell-to-cell variation model (σ on Vth, Bf, αB)

## Candidate missing physics (going in, before agents)
A. **Hurkx full TAT** (1992) with field-dependent enhancement Γ — we have JTS-TAT (BSIM4 §10.1) but unclear if Γ-field-enhancement is active
B. **Self-heating** — BSIM4 selfheatmod likely =0 in current cards
C. **BBT (band-to-band tunneling)** at the floating-body junction at high VG2
D. **Non-uniform base** Gummel-Poon — Pazos parasitic NPN may need Rb(IB) rather than constant Rb
E. **DIBL+body coupling** — DIBL changes Vth which changes IL which changes Vb, not currently iterated to full SC
F. **Field-enhanced GIDL** — agidl tuned to median may miss high-field tail
G. **Quantum confinement** — thin-ox cell quantization shifts threshold

## Dispatch (parallel, 4 tracks)
### Track A — Deep materials + lit search (ikaros, general-purpose agent, ~45 min)
- Re-read ALL data/sebas_2026_04_22/, data/sebas_2026_05_02/, data/mario_slide*.json
- Re-read ALL docs/Zoom/ transcripts and slides (27 jpegs + transcripts)
- WebSearch: Pazos Nature 640:69 2025 supplementary; Hurkx 1992 TAT; thick-ox NS-RAM physics; floating-body charge dynamics at sub-100nm node
- Output: ranked list of TOP 5 missing physics candidates with citations

### Track B — Daedalus extended physics fits (daedalus GPU, agent via SSH, ~60 min)
- Toggle selfheatmod=1 in M1+M2 cards, re-run 33-bias fit
- Replace JTS-TAT with full Hurkx-TAT (field-dependent Γ_TAT)
- Enable mbjt non-uniform-base option in NPN
- BBT term added at gated junction (BSIM4 IBM model)
- Report DC dec change per modification (single-variable ablation)

### Track C — ZGX GPU 5-param physics sweep (zgx, agent via SSH, ~60 min)
- Build 5-axis sweep over (selfheat_κ, hurkx_β, bbt_α, rb_nonuniform_f, agidl_scale)
- ~5^5=3125 cells in parallel, batched on GPU
- Output: best DC dec, ablation matrix, Pareto front

### Track D — Oracle 3-way "missing physics" (~30 min)
- Build packet: full bug-fix history, current 1.163 dec, Pazos paper, Sebas slide titles
- Ask GPT-5 + Gemini + Grok: "What single fundamental BSIM4/floating-body physics term is most likely missing given this audit history? Predict expected dec improvement."
- Synthesize unanimous picks vs divergent

## Post-dispatch synthesis (after all 4 land)
- Cross-reference Track A literature candidates vs Track B/C empirical winners vs Track D oracle consensus
- If 2/3 of {A, B/C, D} converge on same term → implement + re-run cell-wide DC
- If they diverge → 6h follow-up with second oracle round adversarially testing each candidate

## NO-CHEAT discipline
- All numerical claims include `n=33` and forward+backward median±MAD
- Track A literature claims include DOI/URL
- Track B/C single-variable ablation: lock all other params at v5.3 baseline
- Track D oracle consensus = 2/3 same root cause, not "all gave a list"

## Halt condition
- If after all 4 tracks no candidate yields ≥0.3 dec improvement → conclude honestly: "BSIM4+NPN model is parameter-complete; remaining gap is data-limited (need Sebas's specific thick-ox card)"
- That conclusion *is* a publishable result, not a failure

## Eric's standing instruction
- Run hard, all 3 machines, no stops, no idle, no asking
