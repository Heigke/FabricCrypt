# O20 — M3b closure: full validation package after triple walk-back

## Story so far

You (and two other oracles) reviewed this NS-RAM PyTorch port twice
in the past 24 hours. **O18 said FIX** (3-of-3 unanimous): the
brief's 1.00-dec headline relied on Bf=5×10⁴, non-physical for a
130 nm parasitic NPN. **O19 said HOLD** (2:1, gemini+openai vs
grok): even the post-O18 fix (Bf=100 + γ=1×10⁵ multiplier) was
itself a fudge factor, since γ ≫ 1 reintroduces an unphysical gain
path.

We took the M3b plan to closure following O19's recommendations:
- F1.v2: replaced the unbounded γ with a bounded η ∈ [0, 1] sigmoid
  Iii→Vb collection efficiency. Honest result: **median 1.39 dec at
  physical Bf=100, η_max=1.0**.
- F2: ran a 180-pt ngspice grid on M1 and M2; pyport's BSIM4
  evaluator agrees with ngspice to 1–2 % across the operating
  envelope.
- F4: ran a 48-pt 2T-cell op-point cross-check; Id matches at
  32/48 biases despite internal-node disagreement (by design —
  pyport's η mechanism isn't in vanilla ngspice).
- F3 (z142): reran the topology sweep with the η-bounded cell, 5
  seeds, 3 ρ-normalization variants. **The ranking flips.** ER_SPARSE
  (random-sparse) is the new MC champion at N=800 (3.55 ± sd, n=5,
  rho_lambda variant), while MESH_4N drops from z139's 3.29 to 2.20.
  See `z142_summary.json`.

## What we want from you

Three numbered sections, brutal as before. Don't soft-pedal.

### 1. Is the M3b closure actually closed?

We've turned every knob O18 and O19 flagged. The numbers in
`z91g_F1v2_summary.json` are the honest physical-bounded result.
The 180-pt ngspice grid (`z141_summary.json`) confirms the BSIM4
evaluator. The 2T cell op-point check (`z143_summary.json`) shows
Id agreement on most biases.

Are these enough to consider M3b closed, or did we miss something?
Specifically:
  - Is 1.39-dec median defensible as the honest baseline for a
    sendable addendum?
  - The VG1=0.6 V row went from 0.91 dec (Bf=2e4 hack) to 2.25 dec
    (η-bounded honest). That's a HONEST regression — the unphysical
    Bf was hiding a structural model gap. Acceptable, or is the
    2.25 dec a deal-breaker?
  - The 8 NaN biases (Sebas's CSV K1=NaN at negative VG2) still
    aren't fitted. Do we need to refit them with un-overridden
    cards, or is "25/33 biases" defensible if we say so?

### 2. The topology ranking inversion — is it real or another artefact?

The z142 result completely flips z139's ranking. ER_SPARSE jumps
2.20 → 3.55 dec at N=800; MESH_4N drops 3.29 → 2.20 dec. Three
possible interpretations:

  A. **Real:** at honest physical Bf=100, the η-bounded cell has
     less internal feedback gain. Random-sparse connectivity
     (ER_SPARSE) keeps feature decorrelation high; the regular grid
     (MESH_4N) re-collinearises features at low-gain regimes.

  B. **ρ-normalization artefact:** rho_lambda may unfairly clamp
     ER_SPARSE less than MESH_4N at the same nominal ρ=0.9. We're
     running rho_p95_sv and rho_deg_norm variants to check. If those
     give a different ranking, A is wrong.

  C. **Cell-physics artefact:** the η-bounded cell has too-flat a
     response (saturates at η_max=1.0), so all topologies look
     similar; rankings are noise on a flat substrate. Would expect
     all means within ±0.5 dec of each other.

What test would distinguish these? Specifically: should we run the
sweep at η_max < 1 (less head-room for amplification, simulating
even-more-physical bound) to see if rankings stabilise or further
flip?

### 3. Should we send the addendum NOW (with this honest 1.39 dec
+ inverted topology ranking + caveats) or hold for M3c (lateral-
NPN-as-channel-current restructure, ~6 weeks)?

The M3c plan is in `M3c_structural_rewrite_plan.md`. It targets
< 1.0 dec via charge-conserving electron-hole accounting plus
M(Vbc) · Ids_M1 collector formulation. ~6 weeks calendar.

Three options:

- **A (send now):** addendum at 1.39 dec + flipped topology table
  + explicit "M3c restructure required for < 1.0 dec, ETA 6 weeks."
  Walks back two earlier claims but is fully defensible.
- **B (wait for M3c):** hold the addendum 6 weeks. Send a single
  coherent v4.2 brief with M3c-closed numbers.
- **C (something else):** propose a different gating event.

Which is most defensible?

### 4. M3c.2 design decision — REPLACE vs AUGMENT the BJT (NEW)

z142 lands AFTER user authorised "kör m3c". M3c.1 is committed
(charge-conserving Iii routing; gate at η_lat=0 reproduces F1.v2
to machine precision). M3c.2 has an explicit conflict between the
M3c plan and your own (O19 openai) prior critique:

  - M3c plan §M3c.2.b: "Ic_Q1 = M(Vbc)·Ids_channel ... no longer
    a separate Gummel-Poon current"
  - O19 openai (you, 2 days ago): "Don't replace the BJT with a
    pure Ids gain. KEEP the BJT. If you implement 'Ids × gain',
    expect: double-counted conduction, broken gm/gds continuity,
    non-conservative charge, premature snapback/latch."

User has now delegated this choice to us with the constraint:
**"good AND fast for big simulations"** (z142 took 7.9 h; we want
M3c-tier physics that doesn't 10× the wall time).

See `M3c2_design_decision.md` for three candidates:
  - (A) replace per plan literal: introduces 2 fitted knobs (BV, N);
    repeats M3a/M3b fudge-factor pattern
  - (B) augment per O19: keep Gummel-Poon, lateral pair drives
    Ib_Q1; zero new fudge factors; ~2 h to test; no speed cost
  - (C) toggle: implement both, evaluate in parallel

Question: **which path?** Specifically:
  - Is option (B) physically faithful enough to plausibly hit < 1.0 dec,
    or does the model genuinely need the M(Vbc) avalanche multiplier?
  - Does (B)'s additional Ib drive (η_lat·iii_gain·Iii) actually
    propagate through Gummel-Poon's β·Ib relationship to give snapback
    magnitude, or does the current f-Vbc dependence kill it?
  - Is (A)'s "M(Vbc)·Ids_M1 minus Ids_M1 to avoid double-counting"
    a legitimate way to reconcile with O19's critique?

### 5. Topology rec under the cross-norm finding (NEW)

z142 final (270/270 sims, 5 seeds × 6 topologies × 3 N × 3 ρ-norms):

| topology     | rho_lambda | rho_p95_sv | rho_deg_norm | spread |
|--------------|-----------:|-----------:|-------------:|-------:|
| ER_SPARSE    | **3.55**   | **2.84**   | 2.70         | 0.85   |
| LAYERED      | **3.56**   | 2.68       | 1.07         | 2.49   |
| RAND_GAUSS   | 2.25       | **3.08**   | 0.21         | 2.87   |
| MESH_4N      | 2.20       | 1.96       | **4.00**     | 2.04   |
| WS_SMALL     | 2.17       | 2.14       | **4.34**     | 2.20   |
| HUB_SPOKE    | 1.92       | 2.47       | **4.43**     | 2.51   |

Three norms, three different champions. ER_SPARSE is the only
norm-stable topology (spread 0.85 dec, all others ≥2.0 dec).

Question: **which architectural rec for the brief?**
  - (i) ER_SPARSE for cross-norm robustness (safe but never the peak)
  - (ii) HUB_SPOKE under deg_norm = honest "best with right
    normalisation" (4.43 dec, but commits to a specific norm)
  - (iii) Per-norm specific rec ("if you pick rho_lambda, use
    LAYERED; if rho_deg_norm, use HUB_SPOKE/WS/MESH"; risks
    coming across as fence-sitting)

WAVE classification has collapsed across all topologies at honest
cell (0.45–0.53 chance-level). The brief's two-axis MC + WAVE rec
is dead at axis 2.

## Output format

Five numbered answers, terse. Plus a final box:

  - **M3c.2 verdict:** A / B / C, with one sentence on why
  - **Topology rec verdict:** (i) / (ii) / (iii), with one sentence
  - **Send M3b addendum:** A (now) / B (wait for M3c) / C (other)
  - **Critical risk we missed:** what we're still over-claiming
    even now.

## Attached files

- `z91g_F1v2_summary.json` — honest 1.39 dec headline
- `z91g_F1v2_fit.png` — the actual fit plot at η-bounded honest
- `z141_summary.json` — 180-pt ngspice grid pyport-vs-ngspice CDFs
- `z143_summary.json` — 48-pt 2T-cell ngspice cross-check
- `z142_summary.json` — full 6-topo × 3-N × 3-ρ × 5-seed table
  (FINAL — landed 2026-05-04, 28490 s wall)
- `z142_vs_z139_table.md` — analyst summary of cross-norm rankings
- `M3a_addendum_2026-05-03.md` — current DRAFT, banner-flagged
- `M3b_fix_plan_2026-05-03.md` — the F1–F6 plan we executed
- `M3c_structural_rewrite_plan.md` — the lateral-NPN proposal
- `M3c2_design_decision.md` — REPLACE-vs-AUGMENT decision request
- `mario_post_send_honest_email_draft.md` — DRAFT v0
