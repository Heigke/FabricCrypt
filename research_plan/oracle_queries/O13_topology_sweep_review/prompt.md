# O13 — External review of C.3 sparse-topology recommendation

You are an expert reviewer on:
  - reservoir computing (echo-state networks, liquid-state machines),
  - silicon neuromorphic-array layout (KAUST tape-out scale, 130 nm),
  - and the floating-bulk / NS-RAM neuron architecture from
    Pazos et al. (Nature Electronics 2025).

We are writing a one-pager NRF brief for Mario Lanza (KAUST tape-out
lead) recommending the cell-routing topology for the next mask
(M9 milestone). The recommendation rests on a software topology sweep
of our PyTorch BSIM4 port of Sebastian Pazos's 2T NS-RAM cell.
**We need an independent sanity check before the brief goes out.**

## Files attached

  - `C3_tapeout_recommendation_v2_3_addendum.md` — the routing
    proposal + full empirical data
  - `nsram_proposal_short.tex` — the brief itself (Mario one-pager)
  - `z119_aggregate.txt` — z119 topology sweep data (5 topologies × 3
    N × 3 seeds, MC/XOR/WAVE)
  - `z119b_aggregate.txt` — z119b NARMA topology sweep
  - `z120_aggregate.txt` — z120 NARMA at canonical 1/√N (W_rec
    scaling control)
  - `z121_aggregate.txt` — z121 MC + XOR at canonical 1/√N (the
    decisive replication test)

## The headline finding

Across two W_rec scalings (ρ=0.9 spectral-radius-controlled, and
canonical 1/√N) at N=200, **Erdős-Rényi sparse coupling (p=0.1)
beats every other tested topology on memory capacity (MC) and
temporal-XOR**:

  - MC (z119, ρ=0.9): ER_SPARSE 2.79 vs MESH_4N 2.04 vs RAND 1.10.
  - MC (z121, 1/√N):  ER_SPARSE 2.90 vs RAND 0.90 (Δ=+1.99, t=+12.3,
    n=5).
  - XOR (z119, ρ=0.9): ER_SPARSE 0.76 vs MESH_4N 0.59 vs RAND 0.60.
  - XOR (z121, 1/√N):  ER_SPARSE 0.82 vs RAND 0.59 (Δ=+0.235, t=+20.2).
  - NARMA (ρ=0.9):     ER_SPARSE wins marginally (Δ≈+0.13).
  - NARMA (1/√N):      tie within noise (Δ=+0.027, n.s.).

Mechanism we propose: sparse random connectivity gives each cell
access to ~10% of decorrelated random projections of substrate
state, reducing feature collinearity that limits a 4-neighbor mesh
(where neighbors share Vd common-mode driving).

## What the brief recommends

Drop the 32×32 4-neighbor mesh as the *primary* coupled-array. M9
mask carries:

  1. **Primary:** 16×16 sparse Erdős-Rényi fabric (~6.5k coupling
     resistors via switch matrix, p≈0.1, runtime override).
  2. **Control:** 32×32 4-neighbor mesh (silicon-side check on
     z119/z121 software-W_rec transfer).
  3. **Isolated:** 32×32 unchanged (Hopfield-class).

R_bulk pots span 1MΩ–100GΩ, 10-bit log scale (covers κ ∈
[0.001, 0.1] envelope).

## Specific questions for the oracle

  1. **Does the sparse-vs-mesh advantage we measured make sense
     from reservoir-computing theory?** ER p=0.1 random
     connectivity is a known echo-state-network construction; mesh
     is unusual. What's the literature consensus on sparse vs
     mesh in physical reservoirs at N≈200?

  2. **Is our W_rec scaling robustness check sound?** We test ρ=0.9
     spectral-radius-controlled vs canonical 1/√N (Marchenko-
     Pastur-like spectral mass). MC and XOR sparse advantages
     replicate at *both*; NARMA only at ρ=0.9. Is this pattern
     consistent with what you'd expect for the underlying tasks?

  3. **Silicon implementation feasibility.** A 16×16 array with
     10% sparse coupling implies ~26 switch-mediated coupling
     paths per cell on average. Is this plausible in 130 nm with
     a digitally tunable resistor unit cell ~ ten of µm² each? Are
     there known KAUST or Pazos-group constraints we should worry
     about?

  4. **What are we missing?** Failure modes, alternative
     topologies (e.g., scale-free Barabasi-Albert), or layout
     constraints we haven't enumerated. Be ruthless.

  5. **For the brief specifically:** is the C.3 forward-pointer
     wording at line 255 (companion document reference + one-line
     empirical citation) at appropriate strength for an NRF
     reviewer, or should it be tightened/expanded?

Please respond with:
  - Concise answers to (1)-(5) (200 words each max).
  - A summary "GREEN-LIGHT / YELLOW (revise) / RED (block)" verdict
    on the brief as currently written.
  - Top three issues to fix before send, ranked by severity.
