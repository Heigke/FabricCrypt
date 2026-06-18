# PHYSICS RESEARCH CAMPAIGN — 2026-05-14

## Goal

Find the **missing physics** at L=0.13µm that BSIM4 IIMOD doesn't capture. Both pyport and ngspice match each other but are 5-6 decades below silicon at high VG1. Sebas's M1 card runs out of physics — what physical mechanism actually drives silicon's snapback that we're missing?

## Pre-registered gates (LOCKED before research)

- **Track 1 PASS**: ≥3 cited papers + named alternative MOS model that fixes IIMOD at L<0.18µm
- **Track 2 PASS**: ≥1 explicit physics mechanism cited from Mario/Sebas materials
- **Track 3 PASS**: TCAD data found OR definitive "no TCAD in repo" verdict
- **Track 4 PASS**: ngspice deck runs with alt model + reports IIMOD activation Y/N
- **Track 5 PASS**: quantitative bookkeeping — which physical source delivers 4e-9 A body current

## Five parallel tracks

### Track 1 — BSIM4 IIMOD literature audit
**Subagent**: research/web. Find papers (Google Scholar/arxiv) on:
- BSIM4 IIMOD known limitations at L<0.18µm
- PSP vs BSIM4 vs BSIM-IMG vs BSIM-CMG at deep-submicron snapback
- Floating-body/PD-SOI body-charging models (Mario's domain)
- Hot-carrier injection (HCI) cascaded ionization

**Output**: `research_plan/T1_bsim4_alternatives.md` with ≥3 citations + recommendation of best alternative model family.

### Track 2 — Mario+Sebas materials re-scan
**Subagent**: read-only. Comb:
- 21 Mario slides
- Zoom transcript (`nsram/Zoom/2026-04-30 13.03.27 Zoom NSRAM/meeting_saved_closed_caption.txt`)
- Mail thread (`nsram/Zoom/mail.txt`)
- Look SPECIFICALLY for: what physical mechanism does silicon use to charge V_b at high V_G1? Cascaded ionization? GIDL? BTBT? Self-heating? Latch?

**Output**: `research_plan/T2_mario_physics.md` with explicit citations to slide/page/line for each mechanism.

### Track 3 — TCAD output audit
**Subagent**: file-scan. Check repo for any TCAD data:
- `nsram/Zoom/` subdirs
- `data/` for Sentaurus/Silvaco output files
- 21 slides for TCAD plots that may have raw data linked

**Output**: `research_plan/T3_tcad_inventory.md` — found N TCAD files / definitive "none in repo".

### Track 4 — ngspice with alternative model
**Subagent**: compute-light. Try PSP/BSIM6/BSIM-IMG on the flagship bias:
- Build minimal deck with `level=44` (PSP) or `level=70` (BSIM-CMG)
- Same geometry W=L=0.13µm, same gate biases
- Compare ngspice IIMOD output vs BSIM4

**Output**: `results/z350_alt_model_test/` + verdict in `research_plan/T4_alt_model_result.md`.

### Track 5 — Order-of-magnitude bookkeeping
**Subagent**: analysis-only. Calculate:
- Silicon Ic ≈ 4e-5 A, BJT β=1e4 → required Ib ≈ 4e-9 A
- Current pyport: Iii_M1 = 5e-17 A, Ibd_M1 = -6e-18 A, total ~1e-16
- Gap = 7-8 decades
- What physical mechanism CAN deliver 4e-9 A at Vd=2V? Quantify each candidate:
  - Pure IIMOD with realistic α0: max ~ 1e-10
  - GIDL with reasonable a/b: max ~ ?
  - BTBT: max ~ ?
  - Hot-carrier from drain: max ~ ?
  - Bulk-tunneling at the well-body junction: max ~ ?

**Output**: `research_plan/T5_orderofmag_bookkeeping.md` with table of mechanism → max physically plausible current.

## Synthesis (after all 5 return)

Single document `research_plan/PHYSICS_VERDICT_2026-05-14.md`:
- What physical mechanism is the dominant body-charge source in silicon at high VG1?
- Is that mechanism in BSIM4? If no, in which alt-model?
- Can pyport be extended to add it WITHOUT a full rewrite?
- ESTIMATED cell-wide dec recovery if added

## NO-CHEAT

- Each track pre-registered above
- Failures logged honestly
- ≥1 falsifiable claim per track
- Synthesis must address O64 critique: avoid narrative whiplash; cite specifics

## Time budget

Each subagent: 60-90 min. Parallel → total wall ~90 min. Synthesis: 30 min after all return.
