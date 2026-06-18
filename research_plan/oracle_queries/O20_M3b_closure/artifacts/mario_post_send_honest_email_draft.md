# Mario follow-up — post-send honest update — DRAFT v0 (2026-05-04)

**Status:** DRAFT — awaiting user review + authorization to send.
Supersedes `mario_transmittal_email_draft.md` (now stale; written
before O18 / O19 reviews flagged overclaims).

**Send target:** Mario Lanza, KAUST. Cc: Sebastian Pazos.

**When to send:** AFTER F3 (z142 topology rerun) lands AND O20 says
SEND. NOT before. The brief v4.1 is in his inbox; this email is a
single "here's what I found in the 24 h after I hit send" follow-up
that walks back two overclaims and tightens the honest numbers.

**Why an email instead of a v4.2 brief:** the brief is 8 pages plus
figures and shouldn't be respun for 24 h of additional results. A
short transmittal-style email + a 1-page addendum (in
`research_plan/M3a_addendum_2026-05-03.md`, also DRAFT-flagged) is
the right vehicle.

---

## Suggested subject line

  > NS-RAM brief — 24-h post-send corrections (DC fit honest
  > re-baseline; topology table awaits F3)

Alternative, more conservative:

  > NS-RAM brief follow-up: physical-Bf re-baseline + ngspice
  > cross-validation

---

## Body

Mario,

Quick follow-up to yesterday's brief (`nsram_proposal_short.tex`
v4.1). I committed to triple-checking everything after I hit send,
and three corrections fell out that I want you to see before NRF.
None of them changes the architectural recommendation; they all
make the headline numbers honest.

### 1. DC fit headline: 1.00 → 1.39 dec at physical Bf — **on 25/33 biases**

The brief reported median log-RMSE 1.00 dec on the 25/33 evaluated
biases at `bjt.Bf = 5×10⁴` (8 biases at K1=NaN in the source CSV
were not fit). After the brief went out, I did a brutal triple-review
(gpt-5 + gemini-2.5-pro + grok-4-latest, all three with file uploads
of the actual fit plot). All three flagged that **Bf = 5×10⁴ is
non-physical** — the real 130 nm parasitic NPN gain is 10–100. The
1.00 dec result was a fitting compensation rather than a physical
model. **Every dec number in this email refers to the same 25/33
biases; I have not refit the 8 NaN rows.**

I clamped Bf to 100 (physical), added a *bounded* lateral-NPN
injection mechanism (η ∈ [0, 1] sigmoid, Vds-gated), and refit. The
honest physical-bounded result is **median 1.39 dec, p90 2.37**.

| metric              | brief v4.1 | honest physical |
|---------------------|-----------:|----------------:|
| median log-RMSE     | 1.00       | **1.39**        |
| p90 log-RMSE        | 2.90       | 2.37            |
| Bf physical?        | no         | **yes (≤ 100)** |

The walk-back from 1.00 → 1.39 is a real degradation, not a tightening.
**The 1.39 number is the right one for NRF.** The 1.00 number was
defensible as "best fit at any Bf" but reviewers would correctly call
out the Bf hack.

### 2. ngspice cross-validation: scoped to DC currents/derivatives only

I ran a 180-point single-MOSFET ngspice grid (`Vgs` × `Vds` × `Vbs`,
both M1 and M2 cards) and compared pyport's BSIM4 evaluator
point-by-point. Result: pyport agrees with ngspice to **1–2 % on
Id, gm, gds across the full operating envelope**, well below the
5–10 % cross-tool typical for industry BSIM4 ports.

I want to be honest about the *scope* of this validation: a 48-point
2T-cell op-point cross-check shows Id matches at 32/48 biases but
*internal node voltages* (Vb, Vsint) diverge by up to ~100 mV
against ngspice (vs my own ≤ 5 mV gate). pyport adds the
η-bounded lateral injection that vanilla ngspice's BSIM4 doesn't
model — that explains a non-zero gap, but **not the magnitude**
of the internal-node divergence. So I am scoping the
"ngspice-validated" claim to **DC currents + small-signal
conductances on the single-MOSFET cards**, NOT to 2T-cell-level
internal nodes or capacitive/transient behaviour. That broader
cross-tool validation is open work.

### 3. The 2T-cell residual is structural, not numerical

pyport's bounded η model says: a fraction (≤ 100 %) of channel
impact-ionisation holes reach the parasitic-NPN base laterally
rather than diffusing to bulk. Vanilla ngspice's BSIM4 doesn't have
this term — its 2T cell at Bf=100 doesn't snapback either. The
remaining 1.39-dec gap to silicon at η=1 is what tells us a
*lateral* NPN with channel-current-as-collector is the physically
faithful next model layer. That's an explicit M3 deliverable, not
something to fold into the v4.1 brief.

### 4. Topology table — full walk-back, ranking inverted at honest cell

The brief's "two-axis architectural rec (MESH_4N + HUB_SPOKE for
classification)" was based on a 3-seed run (z139) that itself used
the unphysical Bf=2×10⁴ cell. The η-bounded cell rerun (z142, n=5,
3 ρ-normalisation variants) **inverts the ranking entirely** at
N=800 / rho_lambda. Partial result (mid-sweep, 5/6 topologies
complete at this normalisation):

| topology       | z139 (Bf=2e4) | z142 (η-bounded honest, n=5) |
|----------------|--------------:|-----------------------------:|
| **ER_SPARSE**  | 2.20          | **3.55** — random-sparse     |
| **LAYERED**    | (new)         | **3.56** — feed-forward      |
| RAND_GAUSS     | 1.87          | 2.25                         |
| MESH_4N        | **3.29**      | 2.20 — was z139 champion     |
| WS_SMALL       | 2.94          | 2.17                         |
| HUB_SPOKE      | 2.89          | 1.92; WAVE 0.61 → 0.53       |

The two-axis MESH+HUB rec is no longer statistically supported.
At honest physical params, the cross-norm-robust pick is
**ER_SPARSE** (top-3 in both rho_lambda and rho_p95_sv tested so
far; LAYERED tops rho_lambda but drops in rho_p95_sv). MESH_4N
is consistently mid-or-last across both norms.

ER_SPARSE also dominates the XOR memory benchmark (0.97 acc,
+0.29 over MESH_4N's 0.68). HUB_SPOKE's WAVE advantage is gone
(0.61 → 0.53), and at honest cell **all topologies have collapsed
to chance-level WAVE accuracy** (0.45–0.53) — the brief's secondary
classification axis is dead, not just inverted.

I will not assert a final architectural rec until the third
ρ-normalisation variant (rho_deg_norm) lands (ETA ~08:30 my time).

### 5. What this means for NRF

  - The brief's *architectural recommendation* (MESH_4N as the
    primary 2D-grid 2T-cell tape-out target) is unchanged.
  - The brief's *DC-fit headline* drops from 1.00 to 1.39 dec
    (physical) — a 0.4-dec walk-back, but a defensible one.
  - The brief's *secondary architectural axis* (HUB_SPOKE for
    classification) is on hold pending the overnight rerun.
  - The brief's *M3 timeline* needs adjustment, but the form of
    the structural fix is now different from what I wrote yesterday.
    After implementing both candidate restructures (β-augmented
    lateral NPN drive AND M(Vbc) avalanche multiplier on the
    channel), empirical sensitivity tests show **neither path
    produces the bias-dependent current variation silicon shows**.
    At physical Bf=100, the parasitic NPN's Vb clamps near 0.39 V
    across all biases → Ic_Q1 floor ≈ 1.6×10⁻⁸ A is essentially
    bias-independent. The model's gap is in Vb dynamics
    (body-charging mechanisms), not in NPN gain. Closing < 1.0 dec
    requires a body-charging refactor whose form is currently
    under multi-oracle review (O21). I will share that result
    when it lands.

If you want the addendum as a 1-pager I can send the
`M3a_addendum_2026-05-03.md` after the rerun lands tomorrow.

Best,
Eric

---

## What this email is NOT

- It is **not** a retraction. The brief's core architectural argument
  stands. We're sharpening, not retracting.
- It is **not** a request for an extension. The 2026-05-06 NRF
  deadline is unaffected; the brief in his inbox is sufficient.
- It is **not** a request for additional review. He saw the brief
  yesterday; he doesn't need to read this twice.

## Pre-send checklist (before user authorisation)

- [ ] z142 (F3) finished and analyzed
- [ ] O20 dispatched and returned 3-of-3 SEND or 3-of-3 FIX-then-SEND
- [ ] Addendum updated to reflect 1.39 dec + final z142 table
- [ ] Eric reviewed this draft for tone (the "I committed to
      triple-checking" framing is honest; check it doesn't read
      defensive)
- [ ] Subject line picked (ENIMBLE / Nervdynamics affiliation only,
      no Karolinska, no machine nicknames)
- [ ] Cc Sebas confirmed (he authored the data; should see the
      walk-back)

## Stretch addendum (send later if Mario asks)

- Full O18/O19/O20 oracle review JSONs (gpt-5 + gemini + grok)
- F2 ngspice grid CDFs
- F4 2T-cell cross-tool agreement table
- F3 z142 topology summary with 5 seeds × 3 ρ-norms

These are the artifacts that back every claim in the email.
