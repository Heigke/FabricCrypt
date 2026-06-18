# O83 prepub-style critique — Master of Noise paper finalize

Context: today (2026-05-18) we pushed v4.5 brief to Overleaf (commit ebe5310) with physics-primitive framing, six noise modes, demotions explicit (HDC, reservoir-as-general-MG-only-after-ERvMESH-killshot, accelerator-class energy).

Today's killshot: ER_SPARSE vs MESH_4N × {NSRAM_LUT, Linear} × {MG, NARMA-10}: MG passes cleanly, NARMA-10 fails ALL 3 pre-reg gates. Reservoir claim now restricted to MG-class chaotic forecasting.

In flight: EP-NSRAM FULL (zgx, 60k MNIST 4 seeds), GPU-MAX-B v2 BBO daedalus, Stoch-RNG audit (ikaros peripheral-aware energy), LMS-Eq audit (ikaros peripheral-aware iso-precision), Hier-MNIST re-run (zgx multi-seed post-z474b), V7 topology rewrite plan (read-only design).

Now act as a hostile pre-publication reviewer (NeurIPS/Nature Electronics style). 3 Qs:

Q1 — REMAINING HEADLINE RISKS: After today's ERvMESH-driven demotion, which of the surviving v4.5 claims still risks being torn down by a reviewer with minimal effort? Be specific: which claim, which experiment they'd ask for, expected outcome.

Q2 — UNFALSIFIABLE FRAMING DETECTION: The "physics primitive, 6 noise modes" framing is broad. Is any subset of those 6 claims essentially unfalsifiable (just descriptions of what the device does, not specific quantitative claims)? Reviewers will demand each mode meet a clear bar — which ones lack one?

Q3 — KILLSHOT WE HAVEN'T YET TRIED: We've done ERvMESH (✓), NES-GD K2 audit (✓), peripheral-aware energy (in flight), z461 9-test (7/9 with V3/V7 caveat). What's the SINGLE remaining brutal experiment that, if it failed, would most damage the paper? Don't repeat already-running tracks.

Return ≤ 500 words per question. Be hostile — assume the reviewer wants to reject.
