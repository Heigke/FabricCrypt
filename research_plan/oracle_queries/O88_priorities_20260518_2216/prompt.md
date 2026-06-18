# O88 — Where to invest next 14 days

After today's brutal triage: DC gap 4 dec confirmed full-grid, body-physics rejected (BSIMSOI), BSIM4 channel params can't fix it (CHANNEL_ROOT), F3-HIGH win is Vbs-modulation band-aid (O87 unanimous). Silicon Id FLAT 250nA across triode at low VG1 — strong evidence for missing parallel conduction path (well-tap diode? STI corner leak? Schottky?). V7 dead. EP-FIX confirmed cherry-pick.

What survives: V6 self-reset 40ns + V8 LIF + Mario calibration (1 primary bias) + diff IFT pyport (z474b).

3 questions, ≤400 words each:

**Q1 — Best next experiment to find the missing parallel path**
We have evidence for a parallel ~250 nA constant current source dominating in triode at low VG1. What 3 candidate physics, ranked by likelihood given the 'flat Id across Vd' signature: (a) well-tap forward-conducting diode, (b) STI/corner leakage path, (c) source-body parasitic Schottky, (d) some sub-threshold-trap-assisted-tunneling, (e) something else? For top-2: name the falsifying experiment.

**Q2 — Large-scale sim opportunities**
With imperfect cell model (4 dec gap accepted but not fixed), what gigantic-scale network simulations could still provide VALUE to Mario/Seb? Examples: (a) emergent dynamics study (N=10^4-10^6 cells), (b) variability impact (Monte Carlo over Mario data scatter), (c) train deep network using our diff pyport as activation function, (d) other? Concrete metrics + targets.

**Q3 — Pivot decision tree**
If Pillar A (parallel-path fix) FAILS by Day 7, what's the highest-value pivot? Pure methods paper (diff IFT pyport)? ML emulator + reduced-scope claim? Or shut down NS-RAM modeling entirely and pivot resources elsewhere?

NO-CHEAT. Be brutal. Cite specific experiments by name.
