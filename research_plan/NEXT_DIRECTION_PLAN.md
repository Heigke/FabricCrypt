# Next direction — NS-RAM vs ESN systematic comparison matrix

Created 2026-05-10 after both V_G2-continuum and mixed-mode-fabric
directions failed under NO-CHEAT pre-registered gates.

## What just died (and we accept it)

- V_G2-continuum hysteresis (z244b): contrast vs quasi-static is 5×,
  not 100×; the smooth-morph story is real but soft, not dramatic.
- Mixed-mode fabric (z246): pure-floating dominates; grounded tiles
  add no value at any fraction f ∈ (0, 1).

Both gates were pre-registered. Both failed honestly. We do not bend
them.

## What is still defensible (the Mario v4.3 brief)

1. Silicon-energy floor ~10× vs best AI MCU.
2. ESN-class NARMA-10 accuracy (NRMSE 0.612 ± 0.030, vs ESN 0.563).
3. 3-source physics triangulation ≤ 0.51 dec.

## What is still open to explore

The two ESN comparisons we have (MNIST cross-task: ESN +22 pp;
NARMA-10: ESN 8% better) are at a single network size, single task
type, single readout. NS-RAM may have a regime where it competes or
wins — but we have not searched for it. The point of this new plan
is to do that search systematically and honestly.

## Plan: NS-RAM vs ESN benchmark matrix

For each (task, N, hyperparameter-config), report:
- NS-RAM reservoir performance ± std at 5 seeds (full, no shortcuts)
- ESN performance ± std at 5 seeds (same input, readout, N)
- Paired statistics ΔΔ, sign, CI overlap
- Pre-registered gate per cell: PASS if NS-RAM CI does not overlap
  ESN CI in NS-RAM's favour (NS-RAM beats ESN with non-overlapping
  intervals).

### Matrix axes (pre-registered before running)

Tasks (temporal-reservoir benchmarks):
1. NARMA-10 (already have NS-RAM 0.612 / ESN 0.563)
2. NARMA-5 (shorter memory; might favour faster cell)
3. NARMA-20 (longer memory; tests if body-RC ~1ms helps)
4. Memory Capacity (Jaeger 2002) — direct memory-recall metric
5. Temporal-XOR at τ ∈ {5, 10}
6. Mackey-Glass forecast h ∈ {6, 12}

Network sizes:
- N ∈ {100, 200, 500, 1000} (where compute allows; cap by APU thermal)

NS-RAM hyperparameter axis (one knob at a time, leave others at
strong-input default):
- g_VG2 ∈ {0.10, 0.20, 0.30} — already swept on MNIST, repeat on each
  task to find task-specific best
- leak ∈ {0.10, 0.30, 0.60} — body-charge weighting
- dt ∈ {100 ns, 500 ns, 5 µs} — exposes the body-RC τ ≈ 1 ms

ESN hyperparameter axis (matched, fair):
- Same N, same input projection W_in, same ridge readout
- spectral radius ρ ∈ {0.7, 0.9, 0.99}
- leak ∈ {0.10, 0.30, 0.60}
- tanh activation, sparse 10% W

### Acceptance gates per cell

PASS = NS-RAM mean strictly below ESN mean (for NRMSE) or strictly
above (for accuracy / MC), with 95% CI not overlapping ESN CI, at
n ≥ 5 seeds, with no cherry-picking of hyperparams (gate evaluated
at task-best NS-RAM config matched against task-best ESN config).

Any single (task × N × config) cell that PASSes is a brief-headlinable
result: "NS-RAM beats matched-N ESN on benchmark X at this regime."

### Acceptance gate for the brief

If ≥ 1 cell PASSes: Mario brief v4.4 can add "NS-RAM beats ESN on
benchmark X at N = Y in regime Z" as a fourth headline.

If 0 cells PASS: confirm Mario brief v4.3 (energy + NARMA-class +
R-track) as final. NS-RAM is decisively a silicon-energy-only story.
Either outcome is honest and brief-publishable.

## Step ordering (run smallest first)

STEP A (~30 min): NS-RAM vs ESN on NARMA-5 and NARMA-20 at N=200,
default config. Tests if task-memory-length matters. Smallest first
because it can fail fast and refocus.

STEP B (~1 h): NS-RAM vs ESN on Memory Capacity (Jaeger) at N=200,
default config. Directly tests body-RC ≈ 1 ms as memory mechanism.

STEP C (~1 h): NS-RAM vs ESN at varied N ∈ {100, 200, 500, 1000} on
NARMA-10. Tests if NS-RAM's small-N regime competes.

STEP D (~1 h): NS-RAM vs ESN on Mackey-Glass h ∈ {6, 12} at N=200.

STEP E (~2 h): NS-RAM hyperparameter sweep (g_VG2, leak, dt) on the
best-margin task from A–D.

STEP F (synthesis): VG2_CONTINUUM_FINDINGS.md plus new
NSRAM_VS_ESN_FINDINGS.md with the full matrix.

## NO-CHEAT principle still applies

All gates pre-registered. No bending. Full n=5 seeds minimum at
every cell. No single-seed pilots in any brief writeup.
