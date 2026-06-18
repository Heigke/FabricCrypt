# gemini response (gemini-2.5-pro) — 49s

Here are concrete, brief answers to your questions.

### 1. Is the per-cell VG2 mask physically meaningful and the right primitive?

Yes, the per-cell `±0.1V` VG2 mask is physically meaningful and an excellent choice for symmetry-breaking.

*   **Physical Realism:** In a 130nm process, a 100mV shift on VG2 is substantial. It will push individual cells into different operating regimes (e.g., from sub-threshold to weak inversion), altering their nonlinear response to the recurrent input. This is a physically grounded way to introduce heterogeneity.
*   **Symmetry Breaking:** The per-cell random sign `s_i` is the critical component. It converts a global, symmetric class signal into a spatially diverse perturbation. This allows a local Hebbian-style rule to build class-specific correlations, which was impossible with a uniform DC offset.

**Alternative:** A fixed, random input projection matrix (`W_in`) is the standard alternative. Instead of `Vd_t = Vd_base + signal[t]`, you would use `Vd_i(t) = Vd_base + (W_in @ u(t))_i`, where `u(t)` is a vector containing the signal and the class label. This is more powerful but may be less "local" depending on your physical implementation constraints. Your VG2 mask approach is a clever, physically-local substitute.

### 2. What reward baseline works best for R-Hebbian on a small reservoir?

For N=128, a simple **running mean of the goodness signal** is the most robust and appropriate baseline.

*   **Mechanism:** Use an exponential moving average (EMA): `baseline_t = α · baseline_{t-1} + (1-α) · G_t`. The reward is then `r = sign(G_t - baseline_t)`. This centers the reward signal so that better-than-average trials are reinforced and worse-than-average are suppressed.
*   **Parameter:** A good starting point for `α` is `exp(-1/τ)`, where `τ` is a time constant of 20-50 samples.
*   **Risks:**
    *   **Per-class mean:** Avoid this. It requires knowing the label to select the baseline, which leaks information and violates the principle of a global, broadcasted reward signal.
    *   **Eligibility Traces:** Likely overkill. Traces are for assigning credit across long time delays. Since you compute goodness `G` immediately after each sample, a simple reward `r` applied to recent activity correlations (`z⊗z`) is sufficient. (Ref: Hoerzer et al., 2014 use traces for more complex temporal tasks).

### 3. Is a single NS-RAM cell feasible for a FORCE-lite readout?

No, this is a **major architectural risk**. A single NS-RAM cell is a poor substitute for a linear readout.

*   **Physical Mismatch:** The delta rule assumes a quasi-linear output unit. The NS-RAM `Id` is a highly nonlinear, log-distributed current spanning many decades. The error signal `(target - Id_out)` would be poorly scaled, leading to unstable and ineffective learning.
*   **Information Bottleneck:** Your own z142 results show that 30-100 features are needed for good performance. A single cell's activity is an insufficient feature representation.

**Recommendation:** Implement a true linear readout in software: `y_out = w_out @ f(states)`, where `f(states)` could be `log10(|Id|)` from a subset of reservoir cells (e.g., 32 cells). Train *only* the `w_out` vector using a delta rule or RLS. This respects the physics of the reservoir while using a mathematically appropriate learning rule for the readout.

### 4. Is the GPU workload realistic and what is the bottleneck?

Yes, the workload is **entirely realistic** on a Radeon 8060S (gfx1151).

*   **Estimate:** A batched 128x128 `torch.linalg.solve` is extremely fast on a modern GPU (<100µs). The total compute time for 384k solves over 200 timesteps will likely be in the range of 1-2 hours, not days.
*   **Primary Bottleneck:** The Python `for t in range(T)` loop. Each iteration incurs kernel launch overhead. The GPU is fast enough that it will spend most of its time waiting for the CPU to dispatch the next time-step's work.

**Recommendation:** Ensure all tensors (`W_rec`, `feat_prev`, etc.) reside permanently on the GPU to eliminate host-device transfer costs inside the loop. The computation itself is not a concern; minimizing per-step Python overhead is key.

### 5. What is the realistic upper bound on 2-class accuracy?

The rules have fundamentally different capabilities. Here are realistic accuracy bounds for the MG-vs-sine task:

1.  **FORCE-lite (with a proper linear readout):** **>95%**. This is a standard reservoir computing setup with a supervised readout. On a clearly separable task, it should approach near-perfect classification. This is your performance ceiling.

2.  **R-Hebbian (3-factor):** **75-85%**. This is an "in-reservoir" learning rule that modifies the recurrent dynamics. It is fundamentally weaker than a dedicated supervised readout because the global reward signal is a very low-bandwidth error signal. It can improve separability but is unlikely to match a direct regression/classification approach. (Ref: Frémaux & Gerstner, 2016, "Neuromodulated STDP").

3.  **FF-fixed:** **65-80%**. This is the most speculative rule. Forward-Forward was designed for feed-forward, layered architectures. Applying it to a recurrent system by averaging activity over time is a heuristic. It's unclear if this local "goodness" metric can effectively train recurrent weights for a temporal task. Expect it to perform similarly to or slightly worse than R-Hebbian.
