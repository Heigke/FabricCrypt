# Feature request draft — warn on bare-identifier `.param` reference inside `.model` body

**Date drafted:** 2026-05-03
**Status:** DRAFT, pending Sebastian Pazos review before submission to
`ngspice-bugs@lists.sourceforge.net` (or SourceForge feature tracker
at `https://sourceforge.net/p/ngspice/feature-requests/`).
**Sender (proposed):** Eric Bergvall (ikaros), with Sebastian Pazos
(KAUST) cc'd as the original card author.

---

## Subject

`feature request — warn on bare-identifier .param reference inside .model body`

## Body

Hi ngspice maintainers,

We have spent several months porting a Berkeley-style BSIM4 model
card (130 nm floating-bulk NS-RAM cell, S. Pazos et al., Nature
Electronics 2025) to a PyTorch reimplementation. The card was
calibrated against ngspice-42 output. Reproducing ngspice's actual
DC behaviour required identifying a documented but easy-to-miss
syntax behaviour that we believe would benefit from a parse-time
warning.

**The behaviour:**

When a `.model` body contains a parameter assignment of the form
`name = identifier` where `identifier` is a `.param` symbol
declared elsewhere in the netlist (e.g.):

    .param toxn = 4e-9
    .model nmos1 nmos level=14
    + toxe = toxn
    + lpe0 = lpe0n
    ...

ngspice does not perform parameter substitution. The `.model`
body keeps the BSIM4 default for `toxe` (3 nm) rather than the
declared 4 nm. Per ngspice manual §2.10 and the model-parameter
documentation, `.param` substitution inside `.model` bodies
requires `{...}` braces or single quotes:

    + toxe = {toxn}
    + lpe0 = '{lpe0n}'

This is documented behaviour. Our concern: it fails **silently**.
There is no warning when `INPgetValue` (in `src/spicelib/parser/inpgmod.c`)
encounters a bare identifier and falls back to the BSIM default.
For Berkeley-legacy / PDK-derived cards that embed a large number
of `.param` references, this can lead to several silently-misset
parameters. In our case, five different parameters in the M2 card
landed at BSIM4 defaults — a discrepancy of up to 10× in
subthreshold drain current that we initially attributed (incorrectly)
to a parser bug, before discovering the true cause through a
debug-printf-instrumented build.

**Proposal:**

Emit a parse-time warning when `INPgetValue` (or its callers in
the BSIM4 / Berkeley-MOS family parsers) receives a non-numeric,
non-braced token in a `.model` value position and falls back to
the model default. Suggested wording:

    Warning: parameter 'toxe' on .model 'nmos1' line N got
    non-numeric token 'toxn'; using BSIM4 default.
    (Hint: use {toxn} or '{toxn}' for parameter substitution.)

This would prevent the failure mode from being silent for cards
that violate the substitution-rule unintentionally, while leaving
fully-correct cards entirely unaffected (no warnings emitted when
all values are numeric or braced).

**Test case:**

We can supply a minimal reproduction harness:
1. A small BSIM4 `.model` card with three `.param`-substituted
   parameters, one with braces and two without.
2. Expected vs actual `b4ld.c` parameter dump showing two
   silent fallbacks.
3. Suggested Linux x86_64 ngspice-42 build command.

Total ~30 lines of SPICE + ~10 lines of expected output.

**Severity:**

Low (UX/documentation), not a correctness issue. The behaviour
is documented; only the silent failure mode merits attention.

Thanks for the work on ngspice. Happy to provide further details
or test the warning patch if useful.

Best,
Eric Bergvall
Karolinska / FEEL project
ikaros@feel-project.eu  (replace with actual address before sending)

---

## Pre-send checklist

- [ ] Sebastian Pazos reviews and confirms the diagnosis is correct
      for his M2 card.
- [ ] Mario Lanza (KAUST tape-out lead) is informed; the brief he
      receives mentions this submission.
- [ ] Test-case minimal reproduction harness prepared (separate
      file, ~30 lines SPICE + sample debug-build instrumentation
      output).
- [ ] Verify the SourceForge feature-request URL is the canonical
      submission path for ngspice (vs the mailing list); both
      appear active. As of 2026-05-03 the SF tracker is
      `https://sourceforge.net/p/ngspice/feature-requests/`.
- [ ] Submit only after Mario brief is sent (deadline 2026-05-06).
      Order matters: Mario should learn of this from the brief,
      not from a public mailing list post.

## Out of scope (do NOT include in submission)

- Claims about "five silent bugs" — verification on 2026-05-03
  established this framing was overstated. The submission is
  scoped to the **single** documented `.param` syntax behaviour
  and a UX warning request.
- Our pyport reimplementation details — irrelevant to the
  upstream maintainers, who care about the ngspice change.
- The BSIM4 phi formula form — that was our porting error,
  not an ngspice issue; strictly off-topic.
