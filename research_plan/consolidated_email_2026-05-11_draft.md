# Consolidated email to Sebas + Mario — DRAFT 2026-05-11

**To:** Sebastian Pazos (KAUST personal), Mario Lanza (NUS)
**Cc:** Robert Luciani (Nervdynamics)
**From:** Eric Bergvall
**Status:** DRAFT for user review before sending.

**Suggested subject:**
> NS-RAM follow-up: silicon-data requests, registration status, and next-step framing

---

## Email body

Dear Sebas, Mario,

Hope you're both doing well. Following our exchange around the brief
draft on 3 May, a short consolidated note from our side — three threads
to pick up.

### 1. Sebas — two silicon-characterisation asks

Now that the body-charge model + pdiode update is integrated and the
DC fits are closed (33-bias regression against your I-V family,
pyport vs ngspice cross-checked at 0.51 dec at production BJT
parameters), the highest-value next inputs from your side would be:

**a) B_f ground-truth** — single-bias I_c/I_b measurement at
V_G1 = 0.6 V, V_G2 = 0.30 V, V_d ∈ [1.4, 1.8] V (50 mV step) on the
thin-ox 2T cell. Our DC fit converges to B_f ≈ 9×10³ for the
parasitic NPN; one silicon ratio closes whether this is the right
physical point or an artefact. Standard parameter-analyser run on
your existing wafer — no new fab needed.

**b) Transient τ spectrum** — the same 33-bias DC corners, but swept
at 7 V_d ramp rates (e.g. 0.1, 1, 10, 10², 10³, 10⁴, 10⁵ V/s). This
gives us a τ ladder for the floating-body discharge, which is the
last unconstrained dimension in the dynamic model. Pulsed TLP or
ramped-V_d on the parameter analyser both work.

Looking at the new architecture you described, also two forward-looking
items whenever they fit your schedule:

**c) Thick-ox cell card** — same SPICE/BSIM4 format as the
`M2_130bulkNSRAM.txt` you sent 22 April, but for the thick-gate-oxide
variant used in the soma / input-neuron path (slides 17/18). We
understand V_G2 stays ≤ 0.5 V on this cell — the "thick" refers to
gate-oxide thickness handling V_NW ≥ 2.5 V and the 1 V pulse swings
from the starved-inverter front-end, not to operating V_G2 at the
mirror-bank's 2.5–3.0 V linear range. Apologies for any earlier
ambiguity on our side about that.

**d) Repository link** when you have one set up for the dynamic-data
tracking — we'd rather pull from there than ask for files piecemeal.

Three additional asks pulled directly from your 30 April slides that
we want to make sure haven't fallen through:

**e) NFACTOR_M2 realistic range** — your 30 April composite
dependence plot shows NFACTOR_M2(V_G2) reaching ~12 at low V_G2
(red branch, V_G1 = 0.2 V). Could you confirm the silicon range
spans 1–13 across V_G2 ∈ [−0.2, 0.5] V, with V_G1=0.2 branch the
upper envelope? This is for Robert's BBO clamp on his side
(currently 3.0); the underlying production fit we use treats
NFACTOR as data taken from your CSV, so we don't have a bound to
relax — but knowing the realistic upper end would help Robert
unclamp his Julia stack.

**f) High-V_G2 / high-V_G1 corner sweeps** — your 30 April
parameter-dependence plot has rich coverage at V_G2 ≤ 0.30 V but
sparse points at the high-V_G2 corner (V_G2 ∈ [0.40, 0.60] at
V_G1 = 0.6 V) where the layout-dependent effect is strongest.
Even ~9 additional I-V points at V_G1 = 0.6 V × V_G2 ∈ {0.40,
0.45, 0.50, 0.55, 0.60} on existing wafer would close that corner.

**g) Experiment list items 5–7** — your dynamic-response slide
refers to "SR and firing-time experiments (5 through 7 in experiment
list)" as critical for fitting the time-constant dependence. We don't
have that numbered list on our side; please attach or describe the
protocol whenever convenient.

**h) Bulk-current functional form** — your slide "Semi-empirical
bulk-current model" (20-Mar deck) shows
`I_exp = a·exp[b·(V_D + c)]` + `I_pow = d·(V_D + f)^e for V_D > −f`,
with a, b, d, e, f as PWL functions of gate voltage and c constant.
The axis label reads "Gate Voltage V_G" generically — could we
confirm whether the PWL is over **V_G1** or **V_G2**? We currently
use polynomial(V_G1, V_G2) on BSIM4 §6.1; switching to your exact
form would help us close the V_B-dependence loop on the same page
as your noted next revision (R_B = 1 MΩ).

**h-extra) The PWL coefficient table** for a, b, d, e, f vs gate
voltage + the constant c would be the single highest-leverage
artefact for our model to consume directly.

**i) Brian2 SNN benchmark** — your slide 20 of the 20-Mar deck shows
72% LIF-with-Poisson-training vs a Poisson reference (we read it as
~85% on first pass; a second pass suggests ~89%; could you confirm).
Our PyTorch reproduction lands at 84.65% on the Poisson side, so
either we're a few pp below your reference, or the dataset/network
differs. The Brian2 script + trained weights + the parametric-analysis
plan listed on the slide (thresholds, firing time-constants,
excitatory input ranges) would close this benchmark loop cleanly. The
timescale label says ×10⁵ from CMOS; we initially read it as ×10³,
which would also explain part of our gap.

**j) The pulse-parameter set behind the slide-21 ramp-rate plots** —
the labelled callouts (V_set ≈ 2.05 V, t_set 1 µs, t_rise/t_fall
200 µs, V_G1 = 0.45 V, V_G2 = 0.30 V) and the t_rise ∈ {10 µs, 100 µs,
1 ms} sweep would let us reproduce the S_fire / S_relax extraction
directly. Together with the experiment list above this should close
the τ-spectrum ask.

**k) The "previous slide" with −1 V pre-pulse retention ~100 s** is
referenced on slide 17 but we don't have it — could you forward it?
That dataset is the obvious anchor for our retention model.

**l) Confirmation on the body-voltage revision** — slide 14 notes
the current SPICE model does NOT include V_B dependence and a new
version using R_B = 1 MΩ measurement data is in progress. When it
lands, that data + revised model card would directly close the
V_b-dynamics gap we hit in our SNN substitution tests.

### 2. Mario — Ariba update notifications

Since our 22 April check the Smartbuy supplier record (S81248842)
has been in "Registered" status, and we've now received two further
"Updates Approved" notifications on 8 May and 11 May. Could you
confirm what these latest updates change operationally — i.e.
whether anything is needed from our side before a first PO or
collaboration agreement can be raised against the registered status?

Separate but related: the NRF MSP-Sized Grant thread (concept-paper
deadline you flagged for 8 May) — was that submitted, deferred, or
still in motion? Happy to contribute anything from our side if
useful.

### 3. Next-step framing — collaboration agreement / NDA

To put the work on a more formal footing now that the brief is on
paper and silicon data is starting to flow, we'd like to propose
moving to an NDA + collaboration agreement as the next step. Sebas's
BSIM cards, pdiode card, and incoming dynamic data are already
private our side, but a formal instrument would make the relationship
cleaner ahead of testchip floorplan input (Sebas's bonus-track item
from 30 April) and any IP-adjacent discussion.

If NUS has a standard mutual-NDA template we can sign on receipt,
that's the fastest path; otherwise we have a clean reciprocal
template ready and can send it over for legal review.

### Brief status & roadmap alignment

The 3 May draft on Overleaf is unchanged from your side — we're
holding it as v4.3 final until the items above arrive.

Mario — the two horizon/shape questions in that email (joint vs.
separate filing; 1 vs. 5 vs. 7 year deliverables ladder) remain
open whenever you have a moment. From your 20 March deck the
positioning is clear (10–100 mW edge band, NS-RAM at the soma /
input-neuron layer with thin-ox storage cells below, against the
Innatera 2025 reference) — happy to align the brief's deliverables
ladder against that band directly.

No rush on any of this — wanted to consolidate the threads in one
note so nothing falls through.

Best regards,
Eric

---

## Pre-send checklist

- [ ] Verify Sebas's personal email address is on the To: line (not
  the @visitor.nus.edu.sg account — he flagged on 24 March that all
  communication on this topic moves to his personal address).
- [ ] Confirm Robert is fine being Cc'd.
- [ ] Double-check the Ariba supplier ID matches the notification
  emails (S81248842).
- [ ] If Sebas's repository is already set up and we missed it, drop
  item 1(d).
