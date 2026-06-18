# O17 — v4 sanity check after applying O16 fixes

## Context

You (and two other oracles) just reviewed an earlier version of this
proposal (v3) and unanimously voted "Verdict B — fix wording, then
send". Your three v3 critiques are attached as
`O16_*_critique.md`. The author has applied six fixes targeting the
specific overclaims you flagged. We need a sanity check that the
fixes actually landed before the author hits send.

## What changed in v4 (the brief now attached)

1. Abstract: "reproduces Pazos's 33-bias family at 1.00 dec" replaced
   with the honest distribution + per-VG1-row reality. New text
   states converges-at-33, fits-at-25, median 1.00 / mean 1.60 /
   max 3.24 / 28% > 2 dec, names the VG1=0.4 V flat-shelf failure
   explicitly, and frames downstream benchmarks as preliminary
   working hypotheses.
2. Abstract: "forces the architecture" softened to "provisionally
   indicates an architecture".
3. Status section (Sec. 5): full rewrite with a per-VG1 table
   showing the bimodal distribution + an explicit "we own this
   failure now rather than have a reviewer discover it" sentence.
4. Status section: the patched fit plot is now embedded directly
   in the brief (Fig. 5) with caption naming the VG1=0.4 V panel
   as "qualitatively wrong".
5. Limitations bullet on the binning gap: replaced "error is
   applied systematically across every comparison leg" with
   "shape-altering and regional"; explicit "we therefore do NOT
   claim that benchmark conclusions are robust by error
   cancellation" line.
6. M3 deliverables: expanded into M3a/M3b/M3c with pre-registered
   acceptance gates. Timeline widened from "1-2 d code + 3 d
   verification" to "≤2 weeks calendar".

## What we want from you

Three numbered sections, terse:

1. **Are your previous overclaims now removed?** Walk through the
   six items above one at a time. For each, say YES (fix landed,
   no residue) or NO + a quote of where the overclaim survives.
   Be exact. Don't soften.

2. **Did v4 introduce any new overclaim or self-contradiction?**
   Sometimes brief rewrites add their own problems. Read v4
   start-to-end with the question "where does this still go too
   far?". Flag any sentence in v4 that an aggressive reviewer
   would still attack as an overclaim.

3. **Final verdict on send-readiness.** Choose one:
   - SEND: v4 has cleared the bar; no further pre-send edits.
   - FIX [list]: enumerate exactly which sentences need a final
     edit before send.
   - HOLD: v4 still has structural overclaim issues that require
     an O18 round.

Be brief. The author will compare your three answers side by side
and act within an hour.
