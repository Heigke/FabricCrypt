# Identity-as-Operator — Literature Hunt
**Date**: 2026-05-31  **Track**: B (web research, substrate-IS-operator reframing)

## Bottom line

**No published work demonstrates commodity-GPU "operator-as-substrate" with
weights co-adapting to per-die numerical fingerprint.** All prior physical
reservoir computing requires custom HW (PCM, STNO, photonic). Floating-point
non-determinism is well-documented but treated as a *bug to be eliminated*,
not a *feature to co-adapt to*. Per-chip output divergence across L4 / 3090 /
4080 is confirmed at ~1e-4 absolute, but no one trains models that rely on
this divergence being load-bearing.

## Key findings

### 1. FP noise is *structured*, not iid Gaussian (NEW, 2025)
- **"On the Structure of Floating-Point Noise in Batch-Invariant GPU Matrix
  Multiplication"** (arXiv 2511.00025, Nov 2025).
  Claims: ~50% of float16 error variance is in off-diagonal covariance —
  noise is "coordinated directional perturbation," not random static.
  Implication for us: if noise is structured per-launch, the chance of
  it being also structured **per-die** is non-zero. But paper does not test
  cross-chip stability.
  https://arxiv.org/abs/2511.00025

### 2. AMD ROCm reproducibility is *not* default
- **rocBLAS 6.4 docs**: "Bitwise reproducibility is only guaranteed within a
  given combination of ROCm version, rocThrust version, AND **GPU
  architecture**". After rocBLAS 4.0 atomics off by default → bit-repro.
  Functions `gemv/symv/trsv/trsm/gemm` *can* use atomics for perf →
  non-bit-repro.
  https://rocm.docs.amd.com/projects/rocBLAS/en/docs-6.4.1/how-to/what-is-rocblas.html
  https://rocm.docs.amd.com/projects/rocThrust/en/develop/bitwise-repro.html
- **MIOpen**: algorithm selection depends on Find/Immediate mode and perf
  DB; can vary per HW + version. Not bit-deterministic across systems.
- **ROCm GH issue #1459** (Aug 2024): user requests rocBLAS-equivalent of
  cuBLAS workspace control; no AMD developer response.
- **Critical**: rocBLAS's own docs admit per-architecture variance —
  exactly what we want. But our two gfx1151 chips are *same* architecture,
  so this confirms variance is **per-arch**, not necessarily **per-die**.

### 3. NCCL / CUDA: cross-GPU output divergence is documented
- NVIDIA CCCL determinism modes: `not_guaranteed | run_to_run | gpu_to_gpu`.
  The default for cub::DeviceReduce is `run_to_run` — gives bit-repro on
  the same chip but **not across different chips**, even within the same
  CC. Same-arch GPUs (e.g., two A100s) are run-to-run repro per-GPU but
  the developer must opt into `gpu_to_gpu` for cross-device match.
  https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/
- Ingonyama post: "kernels on L4 / 3090 / 4080 differed at 1e-4" — confirms
  per-chip-type variance is real and measurable, but Ingonyama treats it
  as a problem to *fix*, not exploit.
- **Same-model GPUs (two A100s)**: no published cross-pair measurement of
  whether they diverge or stay bit-identical. Open question for us.

### 4. Floating-point non-associativity (HPC + DL)
- **Bouguerra et al. arXiv 2408.05148** (Aug 2024): "Impacts of FP
  non-associativity on reproducibility for HPC and DL". Confirms atomicAdd
  ordering depends on runtime scheduling. Cross-device variance documented
  but framed as obstacle to scientific reproducibility.
  https://arxiv.org/abs/2408.05148

### 5. Physical reservoir computing on commodity GPU: zero hits
- Tanaka et al. 2019 review (arXiv 1808.04962): all PRC implementations
  require non-commodity substrates (photonic, spintronic, analog IMC,
  PCM, in-materio polymer). GPU is used to *simulate* PRC, never to *be*
  PRC.
- 2025 papers (Optomechanical PRC PNAS Jul 2025; Dual-Memory FeFET) —
  still custom HW.
- **DRAWNAPART** (arXiv 2201.09956): commodity-GPU **fingerprinting** via
  WebGL timing — but only identification, not constitutive computation.

### 6. Hardware-aware kernel co-design 2024–2025
- NAX (arXiv 2106.12125): co-designs NN + HW for memristive crossbars —
  custom HW.
- FlashAttention / GQA: tailors NN to GPU primitives, but the goal is
  **uniformly fast on any modern GPU**, not chip-specific co-adaptation.
- "Hardware Compatibility Filter" substack (2024): observes that surviving
  architectures are the ones that map cleanly to existing HW — not the
  inverse (architecture that exploits one specific chip).

### 7. Straight-through estimators for stochastic kernels
- Bengio 2013 (arXiv 1308.3432): STE foundational paper.
- Decoupled-ST (arXiv 2410.13331, 2024): separate temperatures for fwd/bwd.
- Gapped-ST (arXiv 2206.07235): training discrete deep generative models.
- **STE through a stochastic GPU kernel is straightforward** — treat the
  forward as a perturbed matmul, backward as the deterministic gradient.
  No novel theory needed.

### 8. NCCL / multi-GPU: per-device divergence
- NCCL itself appears non-deterministic when reduce-then-split-vs-split-
  then-reduce; documented in NVIDIA/nccl#157. Across different GPU
  *types* the error margin is ~1e-4, but across same-type GPUs (two A100s)
  the divergence is not characterised in the literature.

### 9. Exploiting FP errors for benefit
- "Recycled Error Bits" (arXiv 1309.7321, 2013): exploit recycled error
  bits for higher precision at lower hardware cost. Not chip-specific.
- "Exploiting Verified Neural Networks via FP Numerical Error"
  (arXiv 2003.03021): adversarial use of FP error, not constitutive.
- **No paper exploits per-die FP error as a load-bearing feature.**

### 10. Stochastic rounding
- arXiv 2603.06060: limited-precision stochastic rounding for ML.
- Improves gradient descent in low-precision regimes. But the stochasticity
  is *uniform* across all chips — not per-die.

## Portability assessment

| Finding | Portable to gfx1151 + ikaros/daedalus? | Notes |
|---|---|---|
| FP noise is structured (2511.00025) | Likely yes — RDNA3.5 reductions also use trees | Need to measure |
| ROCm per-arch variance | Yes, documented | But our chips are *same* arch |
| Per-chip-type 1e-4 divergence | **Same-model chip divergence is OPEN** | **This is our exact question** |
| STE through stochastic kernel | Yes, trivial | Implemented in C2 |
| PRC on GPU | No — no prior art | We'd be first |

## Implication for our work

The literature **does not refute** the per-die hypothesis but provides
zero direct evidence it succeeds on commodity GPU. The gap between
"per-arch variance documented" and "per-die variance load-bearing" has
never been closed in a published paper. We are in genuinely unexplored
territory — which the oracle synthesis flags as a yellow flag, not a
green one.
