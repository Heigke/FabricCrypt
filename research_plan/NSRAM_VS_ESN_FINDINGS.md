# NS-RAM vs ESN — head-to-head benchmark matrix (running)

NO-CHEAT discipline: pre-registered gates, n≥5 seeds per cell,
no single-seed pilots in any brief writeup.

## Tests completed

| Task           | N    | n | NS-RAM         | ESN            | Winner |
|---|---|---|---|---|---|
| seq-MNIST cross-task | 1000 | 8 (NS-RAM) / 8 (ESN) | Δ=+5pp over proj | Δ=+27pp over proj | **ESN** (by 22 pp) |
| NARMA-10 (NRMSE) | 200 | 30 | 0.612 ± 0.030 | 0.563 ± 0.038 | **ESN** (8% better, disjoint CIs) |
| NARMA-5 (NRMSE) | 200 | 30 | 0.624 [0.612, 0.638] | 0.541 [0.531, 0.551] | **ESN** (disjoint CIs) |
| NARMA-20 (NRMSE) | 200 | 30 | 0.981 [0.947, 1.016] | 0.853 [0.783, 0.923] | **ESN** (now strict at n=30; flipped from n=5 tie) |
| Memory Capacity (total over k=1..100) | 200 | 30 | 1.777 [1.757, 1.800] | 1.977 [1.948, 2.006] | **ESN** (disjoint CIs) |
| NARMA-10 N=100 | 100 | 5 | 0.693 [0.65,0.74] | 0.572 [0.54,0.61] | **ESN** |
| NARMA-10 N=500 | 500 | 5 | 0.674 [0.65,0.69] | 0.588 [0.55,0.62] | **ESN** |
| NARMA-10 N=1000 | 1000 | 5 | 0.672 [0.64,0.72] | 0.591 [0.56,0.63] | **ESN** |
| Mackey-Glass h=6 | 200 | 5 | 0.193 [0.17,0.22] | 0.067 [0.04,0.11] | **ESN** (large) |
| Mackey-Glass h=12 | 200 | 5 | 0.074 [0.06,0.09] | 0.049 [0.03,0.08] | tie (overlap) |



## Pattern

Across five head-to-head tests at matched N=200, NS-RAM has:
- 0 strict wins
- 1 tie (NARMA-20, both chance-level)
- 4 strict losses to a textbook tanh ESN

The losses are statistically definitive in every case
(non-overlapping bootstrap 95% CIs at n ≥ 5 seeds, p ≪ 0.01).

Mechanistically the story is consistent: the body-state surrogate
has an effective memory horizon of ~3 simulation steps
(τ_body ≈ 1 ms, dt = 500 ns gives ~2 dt per τ). A tanh ESN at
spectral radius 0.9 has a memory horizon roughly an order of
magnitude longer. NS-RAM's body-charge integration is real, but
it does not compete with ESN's long-range linear feedback as a
reservoir-computing memory mechanism.

## Implications for the brief

NS-RAM is **not** a competitive reservoir against textbook software ESNs
on standard temporal benchmarks. The brief's reservoir-computing line
should be framed exclusively as ``ESN-class accuracy at the silicon-
energy floor,'' never as a reservoir-quality claim. The defensible
value is the energy floor and the device physics (parasitic-NPN, body
dynamics, $130$~nm CMOS), not the algorithmic performance.

The Mario brief v4.3 framing (silicon energy + ESN-class NARMA + R-track
triangulation) is correct and final. No fourth headline emerges from
this benchmark matrix.

## What was tried in the V_G2 continuum study and failed

- Rate-dependent hysteresis (z244b): real but soft (5× contrast vs 100×
  pre-registered gate). Body-RC τ ≈ 1 ms is the natural integration
  window; hysteresis-loop area peaks at that ramp duration as
  classically expected.
- Mixed-mode fabric (z246): best-mix (f=0.25 grounded) edges
  pure-floating by 0.006 NRMSE, far below the 0.016 NRMSE margin.
  Grounded cells contribute nothing useful at any fraction.

## Direction stamped CLOSED

The V_G2-continuum / morphable-fabric / regime-bridge family of
hypotheses is **closed by honest negative results**. NS-RAM as a
reservoir-quality device is also closed. Future positive directions
require either (a) measured transient and multi-cell silicon data
(Sebas's pending characterisation), (b) a fundamentally different
network primitive built on top of the cell, or (c) a non-reservoir
application of the cell (compact stateful trigger, PUF, chaotic
oscillator, programmable nonlinear filter) that does not put NS-RAM
in direct competition with software ESNs.
