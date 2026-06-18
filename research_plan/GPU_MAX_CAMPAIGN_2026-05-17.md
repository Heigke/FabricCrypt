# GPU MAX CAMPAIGN — 2026-05-17

Source: Plan agent adbae294e4e5da388 (read-only synthesis from NOVEL_DS_PLAN + 01_LOG + LESSONS_LEARNED).

## Executive Summary

**Goal:** Convert idle GPU capacity across ikaros/daedalus/zgx into 1-2 publishable hardware-aware training results by closing differentiable-pyport infrastructure (MEP-6/7) and running 3 pre-registered campaigns within 14 days.

## 1. Infrastructure to close

### MEP-6: Differentiable pyport via IFT (NOT Newton-unroll)
- Snap-region instability blocked previous MEP-6. Fix: **Implicit Function Theorem gradient** at converged fixed point.
- Forward = existing PT/V_SINT_PIN solver, detached from autograd.
- Backward = single linear solve `torch.linalg.solve(J^T, grad_out)` — no unroll through bistable basin.
- Wrap as `torch.autograd.Function` with custom `backward`.
- Exit gate: `torch.autograd.gradcheck` passes on 64 non-snap cells, FD agreement <1% on 16 in-snap cells, MNIST 1-layer trains ≥90%.
- ETA: Days 1-3 (daedalus dev → zgx port).

### MEP-7: torch.compile + cuda.graph batched solver
- ikaros hit 1M Newton/3.3s. Productionize on zgx.
- Static shapes, AOT capture, persistent CUDA graph, fixed iter + convergence-mask reuse.
- Exit gate: ≥10M Newton solves/s steady-state on zgx.
- ETA: Days 2-4 (parallel with MEP-6).

## 2. Top 3 publishable experiments

### EXP-1: EP-NSRAM — Equilibrium Propagation on body-state
**Why publishable:** EP (Scellier&Bengio 2017, Laborieux 2021) needs a physical relaxation dynamics. NS-RAM body-state τ≈1 ms IS that. First demonstration on CMOS-native analog substrate that's also manufacturable.

**Datasets:** MNIST, Fashion-MNIST, MNIST-1D.
**Baselines:** Laborieux 2021 EP-CNN, BP-MLP matched params, Loihi-2 BP-equiv if available.
**Gates:**
- PASS: MNIST ≥97%, F-MNIST ≥87%, projected energy < 100× digital EP at matched acc
- INFORMATIVE NULL: trains but >2% below BP — still publishable as "first physical EP"
- KILL: gradient diverges in >30% of cells

**Machine:** zgx (main), daedalus (replicate seeds). **ETA:** Days 5-10.

### EXP-2: NES-GD — Noise-Exploiting Gradient Descent
**Why publishable:** Impact-ionization noise IS the SPSA perturbation. No PRNG needed. Connects to Cauwenberghs 1993 weight perturbation with novel non-Gaussian heavy-tailed perturbations. New theory: SPSA convergence under correlated device noise.

**Datasets:** CIFAR-10 LeNet-5, Google Speech Commands V2 (12-class), MNIST.
**Baselines:** Adam, vanilla SPSA-Gaussian, Hiratani 2022 node-perturbation.
**Gates:**
- PASS: Adam-matched ±2% acc at projected ≥5× lower energy/grad-step
- INFORMATIVE NULL: 70-90% of Adam — still methods contribution
- KILL: cross-cell correlations inflate variance >2× Gaussian

**Dependency:** device-noise characterization from existing z44x runs.
**Machine:** daedalus (CIFAR), zgx (KWS + sweeps). **ETA:** Days 6-12.

### EXP-3: HNRT — Hardware-Native Reservoir Tuning
**Why publishable:** Reservoirs are usually hand-tuned. With MEP-6 IFT, backprop through device-physics reservoir to tune V_b, τ, bias — without BPTT. First differentiable analog reservoir.

**Datasets:** NARMA-10, Mackey-Glass, Numenta NAB.
**Baselines:** ESN (Jaeger 2001), LSM (Maass 2002), 1D-CNN-tiny.
**Gates:**
- PASS: NARMA-10 NMSE≤0.05 at N=400; ≥10% over hand-ESN on M-Glass; NAB ≥ HTM-Java
- INFORMATIVE NULL: tuning helps <5% — publishable as device-modeling methods
- KILL: IFT fails on >50% operating points

**Machine:** ikaros (small N), zgx (N=1k-100k sweeps), daedalus (NAB). **ETA:** Days 8-14.

## 3. Pareto sweeps

| ID | Axes | Workloads | Machine |
|---|---|---|---|
| P1 | Accuracy × Energy | EP-NSRAM F-MNIST, HNRT M-Glass | zgx |
| P2 | Accuracy × N (64→100k) | All 3 | zgx primary |
| P3 | Accuracy × Noise σ | NES-GD CIFAR | daedalus |
| P4 | Realism smoke (small N + real noise) | All 3 | ikaros |
| P5 | Steps × τ body-state | EP-NSRAM | zgx |

All sweeps log top-1, NLL, projected energy, wall-time, gradient-norm, convergence rate. n≥4 seeds for screening, n=10 headline.

## 4. Killshot tests

1. **K1 IFT Jacobian singularity** — κ(J)>1e10 in >20% points → no analytic gradient → EP+HNRT blocked. Day 3.
2. **K2 Non-Gaussian noise correlations** — ‖C−I‖_F/N > 0.3 → NES-GD biased. Day 5, no GPU.
3. **K3 Body-state τ drift** — τ varies >20% across V_b range → EP relaxation ill-defined. Day 4.
4. **K4 Snap-region coverage** — if useful points ≥50% in snap and IFT fails → restrict to non-snap, document.
5. **K5 Energy projection sanity** — projected J/inference > digital baseline anywhere → headline collapses, pivot to methods-only.

## 5. Schedule (14 days)

```
Day:           1  2  3  4  5  6  7  8  9 10 11 12 13 14
MEP-6 IFT      X--X--X--|
MEP-7 graphs      X--X--X--|
K1 Jacobian          X--|
K2 noise                X--|
K3 τ drift              X--|
EP-NSRAM smoke              X--X--|
EP-NSRAM full                     X--X--X--X--X--|
NES-GD smoke                   X--X--|
NES-GD full                          X--X--X--X--X--X--|
HNRT smoke                              X--X--|
HNRT full                                     X--X--X--X--X--X--|
Pareto P1-P5                                        X--X--X--X--|
Writeup draft                                                X--X--|
```

**Machine load:**
- **ikaros (always on):** MEP-6 dev iter, small-N realism, device-noise extraction, supervisor scripts
- **daedalus:** NES-GD CIFAR, HNRT NAB, MEP-6 backward correctness, mid-scale
- **zgx:** EP-NSRAM headlines, MEP-7 graphs, P1/P2/P5 sweeps. **Keep saturated 24/7 from Day 5.**

## 6. Risks

| Risk | L | Mitigation |
|---|---|---|
| MEP-6 IFT unstable in snap | M | K1 early; restrict EP to non-snap V_b; document |
| zgx queue contention | L-M | Lock 8-day slot |
| "Projected energy" has no measured baseline | H | State assumptions explicitly; range bounds |
| Reviewers reject projected | H | Pair with measured-fab figure; reframe as "rule the device enables" |
| EP needs symmetric weights NS-RAM lacks | M | Use vector-field EP (Laborieux 2021) |
| K1 fails (IFT singular) | H impact | Pre-commit HNRT-only fallback by Day 5 |
| NES-GD noise drift cell-to-cell | M | Per-cell calibration; theoretical contribution |

## Honest assessment

**3 viable, 2 robustly publishable.** EP-NSRAM + HNRT clear "novel + baseline + matches structural strengths". NES-GD contingent on noise stats (K2). If K2 fails → substitute methods paper "Differentiable Pyport: IFT gradients through analog device fixed points" — publishable at DAC / IEDM circuits track standalone.

## Convergence with O80 + N-BENCH-A

Both independent processes today say: **device-physics primitive paper, not competing accelerator**. The GPU campaign here aligns: EP-NSRAM = "physical EP" (algorithmic claim, not chip claim), NES-GD = "physics-as-RNG-for-training" (primitive claim), HNRT = "differentiable analog reservoir" (methods claim). None of the 3 claims "beats Loihi".
