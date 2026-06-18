# gemini response (gemini-2.5-pro) — 26s

Here is a terse assessment for your due diligence.

1.  **The low-Id, bias-dependent root is physically correct.** It matches Sebas's measurements, which is the ultimate ground truth. The high-Id, bias-independent root (1-3 µA) is a classic parasitic NPN latch-up or snapback state, common in floating-body devices. A real parameter analyzer would measure the low-current, transistor-like behavior. The high-current state might only be accessible via a transient event.

2.  **The residual floor (~1e-10 A) indicates a model limitation, not an invalid solution.** This is common in complex compact models. It suggests a slight inconsistency between different physical equations (e.g., charge vs. current models) or a missing second-order effect (like a specific generation-recombination path) that prevents perfect KCL closure. For currents in the nA range, a 100 pA residual is a known, often acceptable, numerical artifact.

3.  **The lumped solver's stop criterion is empirically selecting the correct physical basin of attraction.** It's not just luck; the solver stops when the residual stops decreasing, which happens at the "soft minimum" of the physical root. It correctly avoids being pulled into the numerically "deeper" but physically incorrect latch-up state.

4.  **The 0.654-dec headline is defensible.** The fit is calibrated against the model's prediction of the *correct physical operating point*. The fact that this point is not a perfect numerical root is a known limitation of the underlying BSIM4 model in this specific, challenging floating-body regime. The physics captured is what matters.

5.  **Path Forward: (a) then (c).**
    *   **First, (a) ngspice cross-check.** Verify that a standard industry solver also finds the low-Id root for a few key biases. This provides independent certification that you are targeting the correct physical branch.
    *   **Then, (c) engineer the quasi-2D solver.** The goal is to ensure your improved model finds the *same* physical root. Use the converged lumped solution as the initial guess for the quasi-2D Newton solve. This should keep it within the correct basin of attraction, preventing it from jumping to the parasitic NPN state. Abandoning the physically-motivated quasi-2D model is premature.
