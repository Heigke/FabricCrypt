# Oracle Critique Packet O68 — NS-RAM Campaign 6h Audit

Context: last 6h includes z371 GPU blitz (new floor 1.05 dec), DS-N17 N=1M scaling (no LSTM win), DS-N10 sine-class win (97.8%), DS-N18 100% valid, SURR-V4 100K on daedalus (94.6% conv), R-49 dbd-avalanche in flight (baseline reproduced perVG1=0.965).

5 surviving claims: DS-N10 sine, DS-N11 Lorenz, DS-N14 edge energy, DS-N15 RNG, DS-N16 equalizer.
5 retractions: DS-N7/N7c/N7d/N9/N12.
Modeling floor: 1.05 dec global, 0.965 per-VG1 (structural).

Please answer BRIEFLY (under 200 words each), CRITICAL not supportive:

Q1: Where is the LATEST result fragile or overclaimed? Specifically scrutinize:
    a) z371 "new floor 1.05" claim — could it be Sobol sampling artefact?
    b) DS-N10 sine-class win — does it really need the substrate or is it task choice cherry-picking?
    c) SURR-V4 88s claim — is 94.6% convergence enough to be useful?

Q2: What SINGLE experiment would best FALSIFY our strongest current claim?
    (Strongest claim: per-VG1 0.965 is "structural", topology fix needed.)

Q3: Have we drifted from NO-CHEAT discipline? Cite specific log lines that worry you.
    Note: we have pre-registered gates but also a pattern of retroactive task-specific framing
    on retractions. Is the surviving-claims list itself a form of cherry-picking?

Respond with terse, hostile critique. We want to find weaknesses before publishing.
