# R-11b: 3-Way Oracle Code Review — pyport BSIM4 IIMOD

Packet: `research_plan/oracle_queries/O60_bsim4_iimod_review/`
Oracles: OpenAI gpt-5 (220s), Gemini 2.5-pro (54s), Grok-4 (344s)

## Q1 — Where does compute_iimpact diverge from BSIM4 v4.8.3?

| Claim | OpenAI | Gemini | Grok |
|---|---|---|---|
| Missing explicit `* Vdseff` in final multiply (leak.py:100) | YES | YES | partial |
| `T2 = tmp / leff` should be `T2 = alpha0 + alpha1*Leff` (no /Leff) | — | — | YES |
| Weak-arm floor `MIN_EXP * diff` should be 0 | — | — | YES |

**Verification against source**: `dc.py:845` defines `Idsa_Vdseff = Idsa * Vdseff` BEFORE
being stored as `dc_result.Idsa` (see `dc.py:861`: `Idsa=Idsa_Vdseff`). So
`dc_result.Idsa` already carries the `Vdseff` factor. **OpenAI + Gemini's "missing
Vdseff" claim is INCORRECT** — the multiplication happens upstream in dc.py.
Grok's T2/leff claim: BSIM4 manual §6.1 actually defines
`T2 = (α0 + α1·Leff)/Leff` (per-unit-length), so leak.py:85 matches spec.
Grok's weak-arm claim is plausible — BSIM4 reference sets `Iii=0` for
`diffVds ≤ β0/EXP_THRESHOLD`; pyport keeps a tiny linear floor which is
inconsequential at the failing OP (T1_strong arm is taken when
diffVds > β0/34 ≈ 0.59V).

**Consensus on Q1 fizzles**: code is largely correct vs. spec; oracles mis-read
variable provenance.

## Q2 — Why 7.5e-48 A at the reported OP?

**3/3 unanimous: (b) Idsa_M1 is essentially zero at the solver's fixed point.**

- summary.json: `Vsint = 1.866 V`, `Vb ≈ 2.0 V`
- M1 terminals: Vg=0.6, Vd=2.0, Vs=Vsint=1.87 → **Vgs = -1.27 V (deep subthreshold)**
- `dc_result.Idsa = 5.26e-36 A` (essentially zero)
- T1 ~ 1e-13 (small diffVds=0.0947 V puts T1 on the strong-bias arm but exp(-β0/0.0947) ≈ exp(-211) is tiny)
- Iii = T1·Idsa·... = 1e-13 × 5e-36 ≈ 1e-48 A ✓

**The bug is NOT in leak.py.** The bug is upstream: the solver has
converged to a non-physical fixed point (M1 off, no snapback) because
there is no impact-ionization positive feedback to kick it into snapback.
This is a bootstrapping problem, not a formula error.

## Q3 — Recommended fix

Oracles propose fixes inside leak.py, but those address symptoms not cause.

**Real fix**: bootstrap the solver out of the M1-off basin so Iii has a
non-zero seed. Options ranked:

1. **Homotopy on `alpha0` or initial guess for Vsint**: start with
   Vsint ≈ 0 (M1 strongly on), step toward physical solution. Likely fix
   in `nsram_cell_2T.py` R_B residual / initial-guess logic, NOT leak.py.

2. **Avalanche-aware initial guess**: in nsram_cell_2T.py (around the
   joint Newton initialization), set `Vsint_init = 0.5·V_d` instead of
   floating, so M1 has Vgs ~ 0.1 V > Vt-margin to seed Idsa.

3. **Smoothed weak-arm in leak.py** (Grok's hint, but minor): line 92
   `T1_weak = T2 * MIN_EXP * diff_safe` — keep as-is; this only matters
   at small diffVds and is not the root cause here.

If a leak.py edit is desired as a guardrail: NONE recommended — the code
matches BSIM4 spec when `dc.py` is read together (Idsa already carries
Vdseff via dc.py:845).

## Most likely bug location

**`nsram/nsram/bsim4_port/nsram_cell_2T.py`** — the joint-Newton
initialization / R_B residual is converging to an M1-off fixed point.
Specifically, the initial guess for `Vsint` (and/or the R_B residual's
treatment of impact-ionization current as a state coupler) needs to seed
M1 into conduction so the Iii feedback can ignite.

## Confidence

**2/3 on Q2 root cause** (all 3 oracles agree on mechanism (b); independent
verification via summary.json Idsa=5.26e-36 confirms).
**0/3 on Q1/Q3 leak.py fix** — oracles' proposed leak.py edits would
double-count Vdseff (it's already in dc_result.Idsa per dc.py:845).
**Net: medium confidence — bug is in solver initialization (nsram_cell_2T.py),
not leak.py.**
