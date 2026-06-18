# openai response (gpt-5) — 205s

Below are concrete answers to each question, with short rationale, risks, and parameter suggestions.

1) Per-cell ±0.1 V VG2 mask: is it meaningful and the right symmetry break?
- Yes, it’s physically meaningful and a good symmetry breaker. In 130 nm, ±100 mV on a gate is well within reliability for 1.2 V devices and will move the 2T NS-RAM operating point substantially (especially given the BJT coupling). Using VG2 for label injection directly modulates the effective negative conductance/transfer nonlinearity, which yields label-dependent feature patterns.
- Better/alternative mechanisms (also physically realizable):
  - Differential label rails: provide two global rails L+ and L− and, per cell, permanently connect one of them based on s_i ∈ {±1}. At run time, you only drive a single global label magnitude ±0.1 V; routing is simple (2 rails + a 1-bit per-cell connect). This keeps per-cell DAC overhead minimal and guarantees broken symmetry.
  - Per-cell input mask on Vd (multiplicative): a fixed random projection of the stimulus onto each cell (classic ESN input masking) plus a small DC label bias on a separate port (e.g., a second “label” Vd rail or a small VG1 nudge per cell). Multiplicative input masks are very effective at class separation and easy to implement (one-time programmable couplers/caps/resistors).
  - Avoid uniform global DC on any single port with later z-scoring or goodness that’s shift-invariant; per-cell masks (additive or multiplicative) are the right fix.
- Risks:
  - Injecting the label on the same port used by recurrent drive (VG2) can destabilize dynamics if too large (pushes network out of the echo regime). Keep label magnitude small (50–120 mV) and monitor effective spectral radius via kappa and W scale.
  - Device mismatch may dwarf small label signals on some dies; ±0.1 V is likely large enough to overcome typical 130 nm mismatch but test 50–200 mV.
- Suggested ranges:
  - Label amplitude: 0.05–0.15 V on VG2.
  - Recurrent gain κ: 0.10–0.25 for N=128, p=0.1 (initial), adjust to keep activity non-saturated and aperiodic.

2) Reward-modulated Hebbian (3-factor): what baseline?
- Use an exponential moving average (EMA) baseline and normalize the reward (advantage-like), not just the sign:
  - b_k ← (1 − β)b_{k−1} + β G_k, with β ≈ 1/τ_b, τ_b = 32–128 samples.
  - v_k (EMA variance) in parallel; normalized reward r̂_k = (G_k − b_k)/sqrt(v_k + ε), clip r̂ to ±3.
  - Update ΔW_ij ∝ η r̂_k e_ij, where e_ij is an eligibility trace (see below).
- Eligibility traces improve credit assignment and reduce variance (Hoerzer–Legenstein–Maass 2014; Frémaux–Gerstner 2016):
  - e_ij(t) = λ e_ij(t−1) + [(z_i(t) − μ_i)(z_j(t) − μ_j)], with λ = exp(−Δt/τ_e), τ_e = 10–50 steps. Maintain μ_i as EMA per feature to center correlations.
- Per-class baseline is optional; it can reduce variance but leaks label structure into the baseline. Start with a single global baseline (simpler, less brittle); add per-class baselines only if variance is still too high.
- Learning rate and regularization:
  - η in the range 1e−4 to 5e−3 times 1/√(Np) works well with elementwise clipping to w_max; add tiny weight decay 1e−6–1e−5 per sample.
- References: Frémaux & Gerstner (Front. Neural Circuits, 2016); Hoerzer, Legenstein, Maass (PLoS Comput Biol, 2014).

3) FORCE-lite with a single NS-RAM readout cell
- Feasibility: a single readout cell can work, but it’s fragile. Because Id spans many decades and the Id–VG2 curve is highly nonlinear, relying on one cell as the only continuous output will often underfit. Expect 70–90% depending on task separability and operating point.
- More robust: 4–16 readout cells with a small linear combiner (trained by a local delta/LMS rule per readout cell) almost always reaches 95–99% on 2-class MG-vs-sine, consistent with ESN/RC results. Your z142 prior (30–100 features for NRMSE ~0.8) also points to needing multiple features for reliability.
- If you want to keep a single physical readout cell:
  - Estimate local gain g = ∂Id_out/∂VG2_out from the Newton solver (you already have the Jacobian); use normalized LMS: Δw_j = η e g z_j / (ε + g^2||z||^2). This stabilizes updates across operating points.
  - Operate the readout cell in a mid-slope region (avoid deep subthreshold and hard saturation); clamp VG2 range narrowly (e.g., 0.1–0.6 V) and log-compress Id in software for the loss.
- Otherwise, 8 readout cells + linear combiner is a sweet spot: local delta on each readout row only; rest of reservoir frozen.
- References: Sussillo & Abbott (FORCE, Neuron 2009) for readout stabilization; standard LMS/normalized LMS.

4) GPU + recurrence: is 384k Newton solves realistic on RDNA3 (gfx1151)?
- Likely yes, but only if you reduce Python overhead and batch aggressively. Your bottleneck is not FLOPs; it’s kernel launch and small-matrix solve overhead.
- Do first:
  - Batch across sequences and cells at each time step. Keep a single per-time-step call that handles [B, N] in one tensor (you already do N; make sure you also do B).
  - Use torch.compile (Inductor) on ROCm 7.0 to fuse Python loops and reduce dispatch overhead. If dynamic shapes are minimal, this helps a lot.
  - Reuse state and Jacobians: warm-start Newton from previous time step and reuse the last Jacobian as quasi-Newton; target 2–4 iterations/step.
  - Replace torch.linalg.solve for 2×2/3×3 with a custom fused small-matrix solver (explicit formula or hard-coded LU) to avoid linalg overhead.
  - Pre-allocate all tensors; avoid Python-side per-step allocations; keep everything on device.
- Rough scale: With 128×32×200 ≈ 0.82M device solves per epoch and 20 epochs × 3 rules ≈ 49M solves, if each Newton step is 3–5 iterations and each micro-solve is very small, runtime will be dominated by overhead. With the optimizations above, think minutes, not hours. Without them, it can be painfully slow or even slower than CPU.
- Consider also: torch.vmap/functorch-style batching for per-cell solves; reduce host–device syncs; float32 is fine; fp16 probably unsafe for stiff I–V.

5) Expected accuracy ceilings
- With a trained readout (even just LMS on a small linear combiner of 8–16 readout cells), 2-class MG-vs-sine should reach 95–99% test accuracy consistently, in line with ESN/RC literature.
- R-Hebbian (no explicit readout), with a good baseline and eligibility traces, can approach high 80s to low/mid 90s on simple 2-class problems at N≈128, but is more sensitive to hyperparameters and reward design.
- Forward-Forward (no readout), even with per-cell masks and non-z-scored goodness, is unlikely to consistently exceed ~70–85% here. FF benefits from deep layered structure and well-separated positive/negative phases; in a single recurrent reservoir with uniform label puffs its discriminative power is limited.
- Your earlier failures align with the literature: purely unsupervised Hebbian/STDP won’t solve supervised labels without a third factor (reward/error) or a trained readout (Frémaux–Gerstner 2016; Hoerzer–Legenstein–Maass 2014; Hinton 2022 for FF).

Additional practical notes
- Replace spectral renormalization with elementwise clipping as you proposed; start w_max ≈ 0.4–0.6/√(Np). If activity dies/explodes, adjust κ and w_max jointly.
- Increase N_TEST as planned; with N_TEST=64 you can resolve ±6–8% changes; use 95% CI with Wilson intervals to avoid over-interpreting small gains.
- Keep goodness simple and shift-sensitive (mean Id^2 or mean |Id|) since you now inject per-cell label asymmetrically; no z-scoring inside the goodness.
- Monitor dId/dVG2 distributions per epoch to ensure the readout(s) stay on-slope.

Key references
- Frémaux, Gerstner. Neuromodulated plasticity, reward-based learning and metaplasticity. Front. Neural Circuits, 2016.
- Hoerzer, Legenstein, Maass. Emergence of complex computational structures from reward-modulated STDP in recurrent networks. PLoS Comput Biol, 2014.
- Hinton. The Forward-Forward algorithm. arXiv:2212.13345, 2022.
- Sussillo, Abbott. Generating coherent patterns via chaotic RNNs (FORCE). Neuron, 2009.

Bottom line
- Use the per-cell ±VG2 mask (or differential label rails) for symmetry breaking.
- For R-Hebbian, use an EMA baseline with variance normalization and eligibility traces.
- Prefer 8–16 readout cells with a small linear combiner if you want 95–99% reliably; single-cell readout can work but is brittle unless you use gain-normalized LMS.
- Optimize GPU by batching and fusing; custom small-matrix Newton is worth it.
- Expect: readout-trained > R-Hebbian > FF in achievable accuracy on this task.
