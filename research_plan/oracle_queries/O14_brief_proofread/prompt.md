# O14 — Mario / NRF brief proofread (post Stage 6a rewrite)

**Submission deadline:** 2026-05-06 (3 days from this dispatch).
**What you are reviewing:** `nsram_proposal_short.pdf` (6 pages, attached).
This is a one-pager-style funding-request brief for Mario Lanza
(KAUST tape-out lead) routed via the NRF call. The author is Eric
Bergvall (Karolinska, FEEL project); subject is a PyTorch BSIM4
port of Sebastian Pazos's 130 nm 2T NS-RAM cell, plus a topology
co-design proposal for a near-term tape-out.

**Why we need a fresh-eyes pass:** The brief was substantially
rewritten today (2026-05-03) in three sections (Abstract, Status,
Limitations bullet 5) after an empirical finding overturned the
previous framing. We need an external read for residual false
claims, awkward phrasing, and defensibility before the user
authorises sending.

---

## Context for the framing change (today's chronology)

The previous brief described "five subtle ngspice-42 model-card-syntax
behaviours" that pyport had to mirror. After porting it for several
weeks, we discovered today (via interactive `showmod m1` on Pazos's
M2 card) that **ngspice-42 actually loads ALL the disputed parameters
correctly** at their card-textual values:

  - `wvth0 = -1.6569e-8`   (NOT zero — the brief had claimed
                             "ngspice silently drops 2nd/3rd
                             key=value on multi-assign lines")
  - `lpe0 = 1.2439e-7`     (NOT BSIM4 default 1.74e-7 — the brief
                             had claimed `lpe0=lpe0n` falls back
                             to default)
  - `phin = 0.05`          (NOT zero — the brief had claimed
                             `phin` is silently dropped)
  - `toxe / toxp = 4e-9`   (= toxn, correctly substituted —
                             the brief had claimed bare-identifier
                             `.param` substitution fails)
  - All other binning terms (wvoff, voffl, pvsat, pags, ...)
                            also load at their card-textual values.

So 4 of the 5 claimed "ngspice bugs" are **empirically false**.
The 5th (a φ-formula factor of two) was a **pyport bug, not an
ngspice bug**, and we've reframed it as such.

The actual situation: pyport reproduces Pazos's 33-bias measured
I-V family at:

  - **1.00 dec median log-RMSE** under an empirical 24-parameter
    post-load patch (`patch_model_values`) that zeroes BSIM4 W/L/P
    binning corrections.
  - **1.88 dec median log-RMSE** with faithful ngspice-equivalent
    card loading (no patch).

The ~0.87 dec gap is internal to pyport's binning evaluation
code, not in ngspice or in our card parsing. We've localised the
fix to a 1-2 day audit of `b4set.c` / `b4ld.c` against pyport's
`compute_size_dep` and committed to closing the gap as the M3
deliverable.

The rewrite removes the "5 silent bugs" enumeration and replaces
it with an honest "1.00 patched / 1.88 faithful, gap localised
to pyport binning, 1-2 day fix" framing in the Abstract, Status
section, and Limitations bullet 5.

---

## What we want from you

Please read the **attached PDF** carefully and answer:

1. **Residual false claims.** Does any sentence in the rewritten
   brief still imply that ngspice has bugs, or that the "5 silent
   behaviours" framing applies? If so, quote the exact passage
   and suggest a replacement. Be aggressive: any phrasing that
   would let a sophisticated reader infer "ngspice is the
   problem" should be flagged.

2. **Defensibility of the M3 deliverable.** The brief commits to
   closing the ~0.87 dec gap in 1-2 days as the first M3 (Jun
   2026) deliverable. Does that scope hold up given the
   description? Is the closure path (audit `b4set.c` / `b4ld.c`
   against pyport's `compute_size_dep`) a reasonable scoping for
   a binning-evaluation bug, or does the timeline read as
   over-confident?

3. **Awkward phrasing / seams.** The Status section was rewritten
   most heavily; some sentences may still show seams between the
   old and new framings. Quote the worst-flowing sentence and
   suggest a tighter version.

4. **The "shape vs absolute scale" caveat.** The brief argues
   that the binning gap doesn't affect benchmark/topology
   conclusions because those depend on differentiable I-V
   *shape* and on relative-comparison structure rather than on
   absolute drain-current scale. Is that argument as written
   convincing? Is there a stronger or weaker version that would
   hold up better?

5. **Anything else worth flagging** — overclaim, underclaim, an
   un-cited number that should be cited, a missing detail a
   funder would ask for, an obvious next-question the brief
   doesn't pre-empt. Brief is 6 pages so we expect a few of
   these.

Please structure your response as five labelled sections matching
the questions above. If you find no issues in a section, say so
explicitly rather than padding. Aggressive criticism welcome —
the user prefers a direct call to a polite one.
