# O70 — 12h NS-RAM gap-closing review

Last 12h (see context.md): 7 consecutive topology rewrites failed (R-43/45/47/49/52/53/55a). R-55a was the "full Zenodo port" with D3 zener + BVPar(VG1) + nbvPar(VG1) + M3 BSS145 + canonical BJT (VA=100, IS=5e-9, Bf=10000). KILL-SHOT gate triggered: ALL-ON gives cell=4.083 dec jump=0.00, baseline 1.127 jump=0.03. Measured fold = 2.2-2.9 dec.

Three questions, BRIEF answers (<150 words each):

Q1 (GATE CROSSING): R-55a hit our agreed KILL-SHOT gate. Is the conclusion "model program cannot reproduce snapback via pyport" justified, or have we cherry-picked the gate definition?

Q2 (CHERRY-PICK RISK): Our "5 surviving application claims" (DS-N10 sine, N11 Lorenz, N14 edge, N15 RNG, N16 5G eq) are all on toy-scale simulated benchmarks. UCI-HAR pubgrade showed sklearn linear ridge 96.2% vs NS-RAM 76%. Are the surviving claims also at risk of falsification at production scale? Single most likely-to-fail among the 5?

Q3 (HIGHEST VALUE NEXT): With model program in retract-mode, what is the single highest-value experiment in 24h that would either (a) salvage the model (e.g. confirm Sebas measurement artifact via body-strap test) or (b) make the application claims production-grade (specific benchmark, dataset URL)?

Be terse. Hostile is fine.
