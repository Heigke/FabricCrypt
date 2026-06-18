# openai response (gpt-5) — 140s

Below is a blunt, referee-style read, structured by your Q1–Q3. I will assume you want the fastest path to a defensible “discovery” claim with minimal room for accusations of cherry-picking or overfitting.

Q1 — Gate-crossing risk (statistical)

- The short answer: “cell-wide log-RMSE = 1.733 dec” is not by itself misleading, but it is fragile and incomplete. You improved two branches massively and regressed one; the aggregate masks that. In a paper, you must headline a branch-stratified view and at least one robust (outlier-resistant) scalar. Otherwise, you invite the critique that your success hinges on a convenient averaging.

- Why fragile:
  - The dataset is balanced (3 gate levels × 11 V_G2 = 33 points), so simple averaging does not overweight any branch. However, a single scalar average still hides that the “hard” physics (low-V_G1, subthreshold) got worse.
  - Your worst branch RMSE is now 2.74 dec (V_G1=0.2). That alone violates any “<2 dec everywhere” interpretation.
  - The presence of solver artifacts at V_G1=0.2 means your reported error metric is partially contaminated by non-physics. Any claim that crosses a threshold using runs with flagged non-converged points should be treated as provisional.

- Where the headline falls apart:
  - Max per-point log error: with a branch RMSE of 2.74 dec, the per-point max is almost certainly >3 dec and likely >4 dec. DISCOVERY <2.0 won’t hold under a max-error criterion.
  - 90th percentile log error: likely still above 2 dec given the V_G1=0.2 regression. I would expect P90 to fail DISCOVERY.
  - Median per-point absolute log error: likely passes DISCOVERY now (because two branches collapsed their errors dramatically), but you must compute it to be sure. If the median passes and the tail (P90 or max) fails, your claim should be “typical error <2 dec” but “tails still open.”

- What to report (and what I would accept for a 130 nm I–V publication):
  - Always:
    - Branch-stratified RMSE (per V_G1) and overall median absolute log error (across all 33 points).
    - P90 (or P95) of absolute log error to show tail behavior.
    - A balanced-per-branch scalar: mean of per-branch medians or RMSEs so no single branch dominates.
  - If you keep a single headline, phrase it as: “Overall median absolute log error = X dec (P90 = Y), branch RMSEs: [0.56, 1.36, 2.74] dec.” Then explicitly state that low-V_G1 subthreshold is the dominant residual and currently regressed.
  - Do not lead with a naked mean log-RMSE if a major sub-regime went backwards; that reads as cherry-picking by omission.

- Pre-registration and QC:
  - Freeze the metrics now (median, P90, branch RMSE) and stick to them across future gates.
  - Exclude or clearly flag non-converged points from the primary metric, or fix the solver first. Counting spiky, non-physical points poisons the comparability of runs.

Q2 — Cherry-pick risk (physical)

- H1 (1 MΩ Sint→GND shunt):
  - Topology: plausible as a proxy for a finite substrate/body return path. But a constant 1 MΩ is almost certainly not physically “right” for a 130 nm bulk device. Typical substrate tie resistances at submicron distances are in the 10^2–10^4 Ω ballpark, not megaohms, unless you are truly modeling a very long lateral/vertical path or an effectively floating body. Even then, the path is strongly bias dependent (depletion spreading, junction leakage, impact ionization), not a linear ohmic element.
  - Smell test: a single linear resistor that happens to fix an internal node voltage is a classic “effective conductance” curve-fit. If you keep it, label it as an effective shunt calibrated at this geometry, not a physical substrate sheet resistance.
  - If the 1 MΩ came from a guess, not a fit, it reduces overfitting risk slightly, but increases “ad hoc” risk. If sweeping it over decades leaves performance broadly flat (monotonic-with-plateau), that helps your physics story. A sharp optimum is a red flag.

- H2 (GIDL → Sint instead of GND):
  - Physics-first: broadly correct directionally. GIDL is generated in the high-field drain/gate overlap region and produces carriers that must be sunk somewhere; routing a significant fraction into the body node is reasonable and often necessary to capture body charging and snapback.
  - However, 100% routing to the body is unlikely to be universally correct. The partition of electron/hole currents among drain, body, and gate varies with field, bias, and geometry. The fact that H2 is exactly what fixes V_Sint runaway is not surprising; it is also exactly what can blow up subthreshold leakage at low V_G1 by injecting body charge where you least want it. This dual effect is what you’re seeing.
  - If the low-V_G1 regression is driven by H2, what you really need (eventually) is a bias-dependent partitioning of GIDL between drain and body (no new parameter if you can derive it from existing electric fields or overlap charges; otherwise it’s a new DOF).

- Overfitting risk in combination:
  - A linear shunt plus 100% GIDL-to-body is a flexible structural pair that can be tuned to kill a runaway state and shape snapback. That combination can overfit if not validated on held-out bias conditions or temperature.
  - Falsification you should do now: ablate each change independently (H2-only, H1-only), and probe generalization (held-out V_G2 or temperature). If H2-only already stabilizes V_Sint and gives you the big wins, H1 is probably an unnecessary crutch (and the 1 MΩ value is not physically grounded).

- Side note on the “BJT B–E one-way rectifier”:
  - Zero reverse leakage is not physical. At low V_G1, reverse BE leakage (and possibly avalanche at higher reverse) can matter to balance KCL at the body. You may be artificially suppressing a legitimate escape path that would otherwise reduce the subthreshold overprediction. Even a small, fixed reverse leakage would be more defensible than an ideal block.

Q3 — Next falsification experiment (single highest-value)

- First, fix the measurement hygiene: the V_G1=0.2 branch has visible solver artifacts. Any further metric-based decisions are suspect until those spikes are removed (either fix convergence or exclude flagged points). This alone could explain part of the “4-decade” overprediction.

- What you want from the next experiment:
  - Identify whether H2 (GIDL-to-body) is the cause of the low-V_G1 regression, without introducing new fit knobs.
  - Test whether H1 (1 MΩ shunt) is a physics necessity or just a stabilizer that found a sweet spot.
  - Get a generalization signal beyond the 33 points.

- My ranking of your candidates (and why):
  1) Add a blind held-out bias group (e.g., an unseen V_G2 stripe) and re-test H1+H2 — do they generalize?
     - Highest value for discovery-credibility. It addresses overfitting directly. If H1+H2 improves the held-out set similarly (especially the snapback shape at high V_G1) without further tuning, your structural changes are likely real. If P90 on the held-out set is similar to the in-sample P90, you have a robust pass. This does not by itself diagnose the low-V_G1 regression mechanism, but it tells you whether H1+H2 is a curve-fit to the original 33 points.
  2) Switch H2 OFF only at V_G1=0.2 (diagnostic ablation).
     - Best single test to identify the mechanism behind the low-V_G1 regression without adding DOF. If the V_G1=0.2 branch recovers while the others stay good, you have direct evidence the GIDL partition is bias-dependent and your blanket “all to body” is wrong. This is an ablation test, not a proposed final model; it’s acceptable as a falsification step.
  3) Sweep H1 shunt resistance over 100 k–100 MΩ with no refit.
     - Tests identifiability and overfitting. If there’s a sharp optimum near 1 MΩ, that’s curve-fit territory. If performance is flat/monotonic and tails off only at extremes, H1 is acting as a crude, robust stabilizer (physics-ish). This will not close low-V_G1 by itself but will help you decide whether to keep H1.
  4) Replace H1+H2 with H2-only.
     - If H2-only already stabilizes V_Sint and preserves the gains, you can drop H1 (and one suspect structural degree of freedom). This simplifies the physics story and may even help low-V_G1 if the artificial shunt was perturbing KCL in subthreshold.
  5) TCAD/structure probe: measure substrate-tap resistance.
     - Highest physical value, but slow. Also, the true path is bias dependent; a single “R_sub” number will not redeem a constant 1 MΩ resistor. Do this, but not as your next step.

- If you force me to pick one experiment now:
  - Add a held-out bias stripe (candidate 2 in your list). It is the most credible way to answer “physics vs curve-fit” today, with zero new parameters, and it raises the bar beyond metric gaming. Immediately after, do the H2-at-V_G1=0.2 ablation to pinpoint the regression mechanism.

- Two additional low-cost checks (no new DOF):
  - Temperature sanity check if you have any second-T data lying around (even a sparse subset): GIDL and subthreshold have very different T-slopes. If H2 is physical, the body-charging effect should move with T in the right direction; if it’s a curve-fit crutch, the T behavior will likely look off. No refit allowed.
  - KCL at the body node: log the contributions (BJT base, junction leakage, GIDL, suppressed diodes) at the bad points. If one term dominates unrealistically after H2 (e.g., GIDL overwhelming everything at V_G1=0.2), that’s your smoking gun.

Bottom line recommendations

- Treat “1.733 dec” as a provisional discovery gate. Publish only with:
  - Branch-stratified metrics, median and P90 across all points, and explicit disclosure that V_G1=0.2 degraded and remains the dominant residual.
  - A held-out validation demonstrating similar gains with H1+H2.
  - An ablation showing that the low-V_G1 regression is tied to the GIDL-to-body routing (and thus motivating a bias-dependent partition in the next revision).
- De-risk H1: either show that performance is insensitive to its exact value (broad plateau) or drop it if H2-only suffices. A constant 1 MΩ is not physically convincing on its own.
- Fix solver artifacts before re-scoring; otherwise, all metric comparisons are suspect in the regime you most need to improve.
