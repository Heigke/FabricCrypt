# O23 — Local-learning rules on NS-RAM reservoir: design sanity-check

## Context

We have a calibrated PyTorch BSIM4 port of Sebastian Pazos's 130nm 2T NS-RAM cell:
- Parasitic Gummel-Poon NPN with Bf=2e4, Is=1e-9 (2D-grid optimum, 0.795 dec NRMSE on Mackey-Glass forecast)
- Vb-clamp at honest physical params, η-bounded ∈ [0,1] Iii→Vb collection efficiency
- ER_SPARSE topology (10% density) won z142 cross-norm sweep
- Vectorized GPU-ready batch Newton solver: `forward_2t_batched(cfg, M1, M2, bjt, Vd_seq, VG1, VG2)`
  → returns Id, Vsint, Vb shape (N, T) given batched VG1/VG2 (N,) and Vd_seq (T,)

Per-step reservoir update currently in `scripts/demo_mackey_glass.py`:
```python
Vd_t = tensor([1.2 + 1.0 * signal[t]])           # input drives Vd
recur = W_rec @ feat_prev                          # recurrent feedback
VG2_eff = (base_VG2 + kappa*recur).clamp(-0.2, 1.0)  # κ=0.30
out = forward_2t_batched(cfg, M1, M2, bjt, Vd_t, base_VG1, VG2_eff)
feat_prev = log10(|Id|)
```

## What we tried

`scripts/demo_local_learning.py` — three "no-readout" local rules on N=12 ER_SPARSE:
1. Forward-Forward (Hinton 2022): `goodness = Σ z²(log|Id|)`, two-pass pos/neg
2. STDP: spike events when z > 1.5, asymmetric cross-correlation
3. Hebbian: `dW = lr·(z @ z.T)/T`

**Class token = global VG2 bias offset ±0.1V** (same for every cell).

All three failed (Hebbian saturated, STDP at chance, FF presumably the same).

## Audit findings (literature + code)

1. **Z-scoring kills the signal.** `Σ z²` is invariant to additive shifts → uniform VG2 puff vanishes from goodness.
2. **Class token is symmetric.** Same DC on every cell → Hebbian covariance is class-independent.
3. **Spectral renormalization (rho=0.9 after each update) fights Hebbian growth.**
4. **Hebbian/STDP are unsupervised** — can't solve supervised classification without 3rd factor (reward/error). Literature (Frémaux-Gerstner 2016, Hoerzer-Legenstein-Maass 2014) is unanimous.
5. Statistical power: N_TEST=8 → resolution 0.125, can't distinguish from chance.

## Proposed rebuild (v2)

**Common harness:**
- N=128 cells, ER_SPARSE p=0.1, GPU (`HSA_OVERRIDE_GFX_VERSION=11.0.0`, ROCm 7.0)
- Bf=2e4, Is=1e-9 (calibrated)
- 2-class signal: MG (chaotic) vs sin+phase-noise
- **Per-cell label mask**: each cell gets a fixed sign s_i ∈ {±1}; class injection = `±0.1·s_i` (breaks symmetry)
- **Goodness = mean(Id²)** (not z-scored)
- Drop spectral renorm; use `W = W.clip(-w_max, w_max)` per element, w_max=0.5/sqrt(N·p)
- N_TEST = 64 (8× more samples)
- 20 epochs × 32 train samples

**Three rules:**
1. **FF-fixed**: positive pass uses correct per-cell mask, negative uses flipped. dW = lr·(act_pos⊗act_pos − act_neg⊗act_neg). act = mean(|Id|) per cell.
2. **R-Hebbian (3-factor reward-modulated)** — Hoerzer-Legenstein-Maass 2014 style:
   - Compute scalar goodness G after each sample
   - Reward r = sign(G − running_mean(G)) ∈ {±1}
   - dW = lr · r · (z⊗z − ⟨z⟩·⟨z⟩ᵀ) over time
3. **FORCE-lite single-cell delta**: append one extra NS-RAM cell as readout; train *only* its row of W with delta rule `Δw_j = η·(target − Id_out)·z_j`. Local on the readout cell; rest of reservoir frozen.

## QUESTIONS

1. **Is per-cell ±0.1V VG2 mask physically meaningful and the right symmetry-breaking primitive?** NS-RAM cells live in a 130nm process; ±100mV on VG2 shifts the operating regime substantially per cell. Is there a better label-injection mechanism (e.g. per-cell Vd amplitude scaling, per-cell input weighting via a fixed random projection) that's both physically realisable and breaks the symmetry Hebbian needs?

2. **For 3-factor reward-modulated Hebbian on a small physical reservoir (N=128), what reward baseline works best?** Running mean of goodness over a window? Per-class running mean? The Hoerzer-Legenstein-Maass paper uses a low-pass filter on `r = R − ⟨R⟩` — is that adequate, or do we need eligibility traces?

3. **FORCE-lite with NS-RAM as the readout cell.** The readout cell needs continuous valued output. NS-RAM Id is log-distributed over 12 decades. Is a single-cell readout actually feasible, or do we need to admit a small linear combiner on top of, say, 8 readout cells with delta-rule each? (Our prior z142 work suggests linear readout NRMSE 0.795 needs ~30-100 features.)

4. **GPU + recurrence**: forward_2t_batched performs Newton with `torch.linalg.solve` per step. For N=128 with T=200 steps × 32 train samples × 20 epochs × 3 rules = 384k Newton solves. Is this realistic on AMD gfx1151 (Radeon 8060S, ROCm 7.0), and is there an obvious bottleneck (e.g. residuals call overhead from Python loop) we should fix first?

5. **Sanity check**: With the v2 design above, what's the realistic upper bound on 2-class accuracy? The literature suggests "1 output cell + delta rule" should hit 90-99% on MG-vs-sine. Does FF (no readout) or R-Hebbian (no readout) plausibly approach that, or are they fundamentally weaker?

Be concrete and brief. Flag specific physical or architectural risks. Suggest concrete parameter ranges where helpful. Cite papers for nontrivial claims.
