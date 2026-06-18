# gemini response (gemini-2.5-pro) — 68s

## Q1 — Fragility
`N_BENCH_comparison_matrix.md` "MNIST (hierarchical SNN) | 97.15% \| 17.7 pJ \| n/p" → This claim is indefensible. It compares a device-only, peripheral-free, silicon-unvalidated simulation at 130 nm against measured, full-system silicon numbers from competitors on modern nodes. The `N_BENCH_gap_analysis.md` file admits an expected "100–1000× degradation" upon realistic modeling. Presenting a number that is known to be wrong by 2-3 orders of magnitude as a point of comparison is not an "architectural sanity check," it is marketing masquerading as data. A reviewer will treat this as intentional misrepresentation.

## Q2 — Falsification experiment
NAME: Jitter vs. Entropy Exclusion
SWEEP: 2D grid of DC biases VG1 and VG2 over their full operational range [0.2V, 0.8V].
OBSERVABLE: At each bias point, measure two metrics: (1) Firing time jitter (std dev / mean) for a repeated suprathreshold ramp input. (2) NIST 800-22 pass rate (N/5 tests) for the stochastic output in TRNG mode.
PASS THRESHOLD: There must exist at least one contiguous bias region where jitter is < 5% AND the NIST pass rate is ≥ 4/5.
WHY THIS KILLS THE CLAIM: The "multi-function" claim requires a single device to be controllably switched between a reliable neuron (low jitter) and a high-entropy source (passes NIST). If the operating regimes for these two functions are mutually exclusive, the device is not multi-function; it is just a noisy switch that cannot perform either task well.

## Q3 — NO-CHEAT audit
Suspicious line 1: "`2026-05-17 — z473 LANDED: R_body=1e7 Ω chosen. Id_pk drift 0.007 dec (4.30 mA, Mario 4.8). V6 self-reset PASS...`" → DRIFT. The `z473_honest_analysis.md` admits this R_body value was chosen "post-hoc" because the original choice failed the V6 test. This is the definition of moving the goalposts: running a sweep, then picking the parameter that happens to make the test pass.

Suspicious line 2: "`2026-05-17 — N-HDC-DVS ZGX DISCOVERY PASS via 4× chance gate: acc 0.593 (6.52× chance 0.091).`" → DRIFT. A 59.3% accuracy on DVS-Gesture is a catastrophic failure against the state-of-the-art (Loihi 2 @ >90%). Inventing a "4x chance gate" to claim a "DISCOVERY PASS" is a transparent attempt to avoid reporting a failure. The gate is meaningless if not pre-registered, and its name suggests it was not.

Suspicious line 3: "`2026-05-17 — N-BENCH-B LANDED (seed 0): 35-class SC HDC D=65536 acc=0.1336, chance=0.0286, ratio 4.67× chance. DISCOVERY PASS by 1.9pp margin.`" → DRIFT. An accuracy of 13% on a 35-class problem is an empirical failure. Claiming a "DISCOVERY PASS" based on a low bar like "ratio over chance" devalues the meaning of "PASS" and obscures the fact that the system is not competitive.

Overall verdict: DRIFTING
