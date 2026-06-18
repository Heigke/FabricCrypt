# gemini response (gemini-2.5-pro) — 76s

Here are three independent oracle responses.

***

### Oracle 1: The Pragmatist

**Q1: Root cause or symptom?**
The basin-lock is a **symptom**. The root cause is a catastrophic implementation error documented in `R_deep_A_topology_compare.md`. Your code is solving the wrong circuit. Specifically, discrepancy D1 states the parasitic BJT (Q1) emitter is wired to ground, not the internal node `Sint`. This fundamentally alters the device's feedback dynamics. The solver isn't "stuck"; it is correctly finding a stable, albeit non-physical, solution for the incorrect topology you have coded.

**Q2: Missing structural element?**
No. Stop looking for new, exotic problems. You are failing to correctly implement the elements you have already identified. The `R_deep_A` report is a list of high-severity bugs. D1 (Q1 emitter wiring) and D2 (triple-counting the well/body diode) are individually sufficient to invalidate your model. The docstring in `nsram_cell_2T_head.py` shows the *intended* KCL for `R_Sint` includes `+ Ie_Q1(Vb−Vsint, Vb−Vd)`, but the audit confirms this term is either missing or uses `Vbe=Vb`, which is wrong. This is a straightforward implementation bug, not a subtle missing element.

**Q3: Bimodal error?**
The bimodal error is a direct signature of the broken BJT feedback. The model has two operating regimes:
1.  **Low Error (~1 dec):** At biases where the parasitic BJT is supposed to be OFF, your model gets it "right" by coincidence. The mis-wired BJT is also off, so the error is small.
2.  **High Error (3+ dec):** At biases where the BJT is supposed to turn ON and initiate snapback, your mis-wired version fails to activate correctly. The feedback loop is broken, the internal voltages run away, and the error is massive. The bimodality is the switch between these two regimes.

**Q4: Prediction?**
The Vb-free solver, applied to the **current, broken** topology, will not fix the core problem. It will find the solution to the wrong circuit more robustly, but the physics remain wrong. I predict the median dec will stay high, at **2.5 ± 0.5**. However, if you first fix the documented topological errors (D1, D2, D9) and *then* use the new solver, you will be solving a physically plausible model. In that case, I predict a median dec of **1.2 ± 0.4**.

**Q5: Discriminating experiment?**
Take a single failing bias point from `z330_summary.json` (`VG1=0.6, Vd=2.0`). Manually insert the known-good `ngspice` solution (`Vsint=0.382`, `Vb=0.267`) into your `pyport` residual function. Print the values of the two residuals, `R_Sint` and `R_B`. If the "basin-lock" theory were true, these residuals should be close to zero. I predict they will be enormous, proving that the KCL equations themselves are fundamentally incorrect for that state point. This directly tests the fidelity of your implemented equations.

***

### Oracle 2: The Physicist

**Q1: Root cause or symptom?**
It is a symptom. The root cause is a broken physical feedback loop in your body node KCL (`R_B`). The `Vb=Vd` lock occurs because the primary mechanism for pulling `Vb` *down* and stabilizing it is either missing or incorrect. In the physical device, impact ionization (`Iii`) injects current into the body, raising `Vb`. This turns on the parasitic BJT (`Vbe = Vb - Vsint > 0.7V`), which draws base current (`Ib_Q1`) *out* of the body, creating negative feedback. Your implementation of the BJT (`Vbe=Vb`, per D1) is incorrect, so this crucial self-regulating feedback is broken. The solver finds the only other stable state: the body voltage running away until it is clamped by the forward-biased drain-body diode.

**Q2: Missing structural element?**
The explicit components in `R_deep_A` are correct. I suspect the error is a more subtle one of KCL formulation. The docstring in `nsram_cell_2T_head.py` lists the terms for `R_Sint`, including `+ Ibs_diode_M1(Vb−Vsint)` and `− Ibd_diode_M2(Vb)`. Since M2's drain *is* the `Sint` node, the `Ibd_diode_M2` term should physically be a function of `Vb - Vsint`, not `Vb`. You may be double-counting or using conflicting models for the single physical junction between the body and the `Sint` node. The error is likely a sign flip or a redundant term in the KCL implementation, not a missing box on the diagram.

**Q3: Bimodal error?**
This signals a bifurcation, a sharp transition between two distinct physical regimes.
-   **Mode 1 (Good Fit):** At low `Vd`, impact ionization is off. The device acts as a simple MOSFET pair with minimal body effects. Your model captures this regime adequately.
-   **Mode 2 (Bad Fit):** Above a `Vd` threshold, `Iii` activates, injecting hole current into the body. This is the trigger for snapback. In your model, the BJT feedback that should control the subsequent rise in `Vb` is broken. `Vb` shoots up uncontrollably. The bimodal error cleanly separates the "pre-snapback" and "failed-snapback" operating points.

**Q4: Prediction?**
The 2D solver will not fix the broken physics. It will likely improve convergence stability but will not restore the correct snapback behavior. I predict the median dec will land at **1.8 ± 0.6**. The model will still fail to capture the snapback "knee" correctly, leading to very large errors in that specific, critical regime, even if it performs better elsewhere.

**Q5: Discriminating experiment?**
Perform a "source-injection" experiment. In both `ngspice` and `pyport`, at a fixed bias (`Vd=2.5V, VG1=0.4V`), disable the internal `Iii` calculation and instead inject an ideal current source `I_test` directly into the body node `B`. Sweep `I_test` from 1pA to 1uA and plot the resulting `Vb` and `Vsint`. `ngspice` will show `Vb` rising until `Vb - Vsint` reaches ~0.7V, at which point the BJT turns on and clamps `Vb`. I predict `pyport` will show `Vb` behaving very differently—likely rising much higher—revealing the flaw in how your model's BJT and body diodes respond to injected charge.

***

### Oracle 3: The Mathematician

**Q1: Root cause or symptom?**
Symptom. The root cause is a pathologically structured residual function leading to an ill-conditioned Jacobian matrix. The incorrect wiring of Q1 (D1) and the extraneous diodes (D2) have created a numerical landscape where the correct physical solution (`Vsint=0.38, Vb=0.27`) lies in a region where the Jacobian `J` is nearly singular. The Newton-Raphson solver is numerically repelled from this region. Conversely, the `Vb=Vd` state corresponds to a forward-biased diode dominating the equations, making `J` well-conditioned and creating a large, mathematically stable basin of attraction. The solver isn't "stuck"; it's correctly avoiding a numerical minefield and settling in the only stable fixed point your equations provide.

**Q2: Missing structural element?**
You are not missing a component; you are missing *coupling terms* in your Jacobian matrix. The KCL for `R_Sint` depends on `Ie_Q1`, which in the correct topology depends on `Vb - Vsint`. This creates a strong off-diagonal term `∂R_Sint / ∂Vb`. Per `R_deep_A`, your BJT emitter is grounded, making `Vbe = Vb`. This incorrectly sets the crucial derivative `∂Ie_Q1 / ∂Vsint` to zero. You have severed a critical feedback path in the calculus of the system, decoupling the two residuals and making the problem numerically stiff and difficult to solve.

**Q3: Bimodal error?**
This reflects the conditioning of your nonlinear system.
-   **Well-conditioned regime (low error):** At biases where the mis-wired BJT is inactive, the system is simpler, the Jacobian is well-behaved, and the solver finds the unique, correct solution.
-   **Ill-conditioned regime (high error):** At biases where the BJT should be active, your incorrect implementation creates a non-physical system with a treacherous numerical landscape. The solver's success becomes pathologically sensitive to the initial guess. For some biases, the initial guess falls into the small, correct basin; for others, it falls into the large, non-physical basin at `Vb=Vd`. This is a classic signature of a poorly posed problem.

**Q4: Prediction?**
The 2D solver is the right tool for the *correctly formulated* problem. Applied to your **current** system, it will likely perform *worse*, with more convergence failures, because it will be less constrained. I predict a median dec of **>3.0**. However, if you first fix the topology (D1, D2, D9) to create a well-posed problem, the 2D solver will excel. In that case, I predict a median dec of **0.8 ± 0.3**.

**Q5: Discriminating experiment?**
At a single failing bias point, initialize your `pyport` Newton solver not with a default guess, but directly at the known-good `ngspice` solution (`x0 = [0.382, 0.267]`). Then, compute and execute exactly **one** Newton step: `x1 = x0 - J(x0)⁻¹ * F(x0)`. Examine the update vector `Δx = x1 - x0`. If the "basin-lock" hypothesis is correct, `Δx` should be small, as you are already near a solution. I predict `Δx` will be enormous and point in a non-physical direction (e.g., `ΔVb` > 1V), proving that the local gradient defined by your Jacobian is fundamentally wrong and is actively pushing the solution *away* from the correct physical point.
