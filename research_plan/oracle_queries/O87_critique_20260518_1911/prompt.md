# O87 — Hostile critique on today's physics-fix wave

Last 6h: F1+F2+F3 round 1 (neutral), F3-HIGH jtss amplitude sweep (jtss=1→100 body monotonically improves: 4.029→3.201→2.907 dec, drain regresses, both intermediate), BSIMSOI hybrid (neutral +0.017, isolates triode 6.9 dec >> subth 2.4 dec as dominant residual), F4-PNP+Rwell+RBODYMOD=3 (KILLSHOT — bit-identical, PNP exp(-77) numerical noop).

Baseline DC: median 4.026 dec across 33 biases, 560 NaN/non-conv, slope-MSE 8.70 at VG1=0.4.

**Q1 — Where is the F3-HIGH win fragile / overclaimed?**
jtss=100 body→2.907 dec is the only real lift today. Most likely critiques:
(a) Is the improvement from GIDL/BTBT physics OR from Vbs-modulation side-effect (adding any small body-source forward current shifts Vbs and via body-effect changes channel-Id, which would be a band-aid not physics)?
(b) Does the win hold under bootstrap CI or is it sampling noise on stride=2?
(c) Is V3-knee slope-MSE actually improving (real physics signal) or is the improvement concentrated in already-OK biases?
(d) What happens with jtss=1e3/1e4 — does monotone trend continue (real) or saturate/reverse (numerical artifact)?

**Q2 — Single experiment to falsify the strongest survivable claim**
Strongest claim today: "F3-HIGH at jtss∈[100, 1e4] is a real physics contribution closing 1+ dec of the DC gap via GIDL/BTBT mechanism." Name the SINGLE experiment that would expose this as band-aid rather than physics. Be specific: which control, which observable, what threshold.

**Q3 — NO-CHEAT drift audit**
Today's process: F1+F2+F3 dispatched on oracle 3-way convergence, killed by bit-identical result. F3-HIGH found at non-default amplitude. F4-PNP+BSIMSOI killed. Have we drifted from NO-CHEAT? Look for:
- Post-hoc claim-tuning (declaring 2.907 = "first real lift" when goal was ≤2.5)
- Pelican rolls (claiming F3-HIGH=physics when prior F123 round used same term at wrong amplitude and didn't help)
- Selective baseline (BSIMSOI verdict 3.86 vs published 4.026 — stride=2 sampling artefact, but still in the comparison frame)
- Cite specific log lines if you find drift

Be brutally honest. Return ≤450 words per Q.
