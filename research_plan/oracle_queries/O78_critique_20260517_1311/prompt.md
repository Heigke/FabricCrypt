# NS-RAM 6h critique cycle

Last 6h activity: 3 AMBITIOUS PASS network sims (Res-MG NRMSE 0.015, Stoch-RNG NIST 5/5+KL 1.4e-6, LMS-Eq BER 0 @20dB / 170× energy savings) on top of earlier 4 DISCOVERY PASS (HDC 84.5%, PC-NAB F1 0.335, Mem-Pal 87.5%, FF-MNIST 91.6% near). Cell-physics: z462b PT-solver 0.983 dec, z465 BBO INFRA_ONLY 2/7 Mario targets (I_D 4 dec off Mario's 4.8mA), z461 6/9 dynamics.

QUESTIONS — be HARSH:

Q1 (overclaim): 3 AMBITIOUS PASS sims (Res-MG, Stoch-RNG, LMS-Eq) all on zgx GPU. Are these benchmarks easy for NS-RAM specifically, or did we pick easy benchmarks? Top 3 risks the apparent successes are due to (a) substrate-friendly tasks, (b) hyperparameter cherry-picking, (c) surrogate-vs-real-cell substitution, (d) GPU-noise-aided results not representable on actual NS-RAM silicon.

Q2 (highest-info falsifier): z465 hit structural ceiling at I_D 4 dec below Mario (4.8mA vs 0.2µA). z466 widens BBO to 7D (incl knee-gates) on daedalus. If z466 ALSO fails to reach mA-range I_D, what's the SINGLE concrete next experiment that would conclusively rule in/out "current model can never reproduce Mario transient"? Specifically: do we need different topology, different solver, different fitting target reweighting?

Q3 (NO-CHEAT drift): Read context.md. Cite specific log lines where we may have over-stated PASS results vs actual numerical evidence in summary.json files attached. Especially: LMS-Eq energy "170× lower" — is that comparing apples to apples (NS-RAM surrogate vs LMS-f32)? Stoch-RNG energy 0.40 pJ/bit "model" — is that measured or estimated?

NO-CHEAT in your reply:
- Ground every critique in attached log/summary file paths
- Disagreements between oracles valuable
- If insufficient evidence, say so
