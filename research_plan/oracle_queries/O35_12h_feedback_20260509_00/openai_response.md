# openai response (gpt-5) — 94s

Q1. Decision gates crossed and the required actions (brief-only)

- Week‑1 CI gate for Path A performance: PASSED
  - Evidence: z223 NARMA‑10 NRMSE 0.6122 ± 0.030 at 30 seeds, +27% vs z216 0.84.
  - Decision: Freeze the z223 hyperparameters as Week‑1 baseline (Cb=5 fF, dt=500 ns, g_VG2=0.05, leak=0.30; N=64 unless otherwise stated).
  - Action for the brief: Replace any prior single‑seed/best‑case numbers with the 30‑seed CI. State “Week‑1 performance target met” and that all subsequent claims will be evaluated against this frozen config.

- R‑track 3‑way triangulation (surrogate ↔ pyport ↔ ngspice): CLOSED (marginal at tail)
  - Evidence: pyport↔surrogate max 0.39 dec (z230), pyport↔ngspice max 0.506 dec (z231) with the only miss at M2‑OFF leakage tail; transitive bound surrogate↔ngspice ≤ 0.90 dec worst‑case; ngspice confirms physical subthreshold regime; Vb settles 0.249–0.259 V.
  - Decision: Treat the surrogate as faithful for reservoir‑regime operation; carry forward a known, tracked M2‑OFF tail bias.
  - Action for the brief: Update the R‑risk section to “closed with one known tail caveat.” Replace the old “1.39 dec at Bf=100 + η ≤ 1” text with the production‑device result: “≤0.39 dec vs pyport; ≤0.51 dec vs ngspice, with a 0.006‑over tail miss at VG2=0 only.”

- Solver architecture choice (lumped vs quasi‑2D): DECIDED (de‑scope q2d as PoR)
  - Evidence: z232 25‑bias bootstrap — lumped 0/25 converged (so prior ‘low‑Id’ values were non‑converged iterates), q2d converges but is +1.03 to +1.26 dec vs lumped; branch‑protect reduces jumps 4–25× but does not recover lumped; ngspice currents ≈1e‑11 A at production BJT, closer to lumped’s last iterates than q2d alt‑root.
  - Decision: Do not present q2d as production path; Plan‑of‑Record is (i) the surrogate (Vb fixed) and (ii) pyport with Vb set to ngspice/surrogate value. Treat q2d explorations as R&D only.
  - Action for the brief: Remove/retire any “branch‑protect rescues lumped” language. Add one clear sentence: “The quasi‑2D Newton path locates an alternate fixed point in snapback; production path will use lumped with continuation or the validated surrogate.”

- Energy/scale‑gap gate (C‑track): PASSED
  - Evidence: 0.7 µJ for 1024‑step inference at N=64 (this work), vs MAX78000 ≈5 µJ, Coral ≈10 µJ, Cortex‑M4 50–100 µJ.
  - Decision: Claim a ~10× energy advantage vs the best AI MCU at area‑matched N=64 for 1k‑step inference.
  - Action for the brief: Insert the comparison table and the headline “~10× vs AI MCU, ~70× vs Cortex‑M” with the cycle‑product caveat (SRAM readout ~50 nJ additional; no DRAM).

- Compute/scale gate: PASSED
  - Evidence: GPU path stable to N=20k using the bmm workaround; APU thread‑capping landed (F.1).
  - Decision: Claim readiness to scale experiments/reservoir size without thermal incidents.
  - Action for the brief: Add a one‑liner on GPU scalability and the safety guardrails (util_safe_sweep, thread caps). No new performance claim needed.

- Cross‑task generalization gate: NOT PASSED (insufficient evidence and current result is negative)
  - Evidence: z224 digits 8×8 sequential — reservoir 52% vs pure projection 56% at 5 seeds.
  - Decision: Do not claim cross‑task generalization at this time.
  - Action for the brief: Add an “in progress” status with the current negative indication and a note that a powered 28×28 sequential run is queued.


Q2. Cherry‑picking/statistical pitfalls in the last 12h

- Primary risk: within‑task hyperparameter selection on NARMA‑10
  - The 0.6122 NRMSE at 30 seeds is statistically solid for that task, but the hyperparameters (Cb, dt, g_VG2, leak) were chosen on NARMA‑10 (z221) and then validated on the same task family (z223). Without nested CV or a held‑out task, this is an in‑sample improvement, not a generalization claim. The strong improvement stands for NARMA‑10, but it should not be promoted as task‑general.

- Cross‑task test (z224) is underpowered and currently negative
  - Only 5 seeds on 8×8 sequential digits, a 4‑point accuracy gap (52% vs 56%). With typical between‑seed SDs for small‑N classification in the 2–4% range, a 5‑seed paired test has low power (<50%) to resolve a 4‑point effect. The negative direction is informative, but not conclusive; at minimum, 20–30 seeds would be needed to get a ~±1.5–2.0% 95% CI around the delta.
  - Bottom line: treat z224 as an early warning that the NARMA‑tuned hyperparameters may not transfer to classification, but do not over‑interpret it. The right fix is a powered, frozen‑hyperparameter cross‑task evaluation.

- Lesser pitfall avoided: You correctly closed the R‑track selection concern via ngspice triangulation and explicitly called out the M2‑OFF tail as the only systematic bias. That transparency reduces the risk of inadvertent cherry‑picking on “friendly” biases.


Q3. Single highest‑value experiment for the next 12 hours

Frozen‑hyperparameter cross‑task evaluation on sequential 28×28 MNIST vs a pure‑projection baseline, powered to 20–30 seeds.

- What to run
  - Script: clone the 8×8 sequential pipeline from z224 to a 28×28 variant (e.g., scripts/z224_seq_digits.py → scripts/seq_mnist_28x28_eval.py). If a fresh script is needed, keep the API identical and wrap with util_safe_sweep to enforce worker×threads limits; use the GPU reservoir matvec via the current bmm path.
  - Dataset: MNIST (train/test from torchvision), streamed as a 784‑step sequence (row‑major pixel scan), standard normalization.
  - Models:
    1) Path A reservoir with the frozen z223 hyperparameters (Cb=5 fF, dt=500 ns, g_VG2=0.05, leak=0.30), N=64 to stay area‑matched with the energy claim; linear or logistic readout with L2 regularization (same as z224).
    2) Pure projection baseline (no reservoir dynamics; same readout and preprocessing).
  - Seeds: 20 minimum (30 preferred, matching z223), paired across models.
  - Outputs: per‑seed test accuracy, paired deltas, 95% CI (bootstrap or paired t), and a one‑line decision (reservoir > projection by ≥3% with 95% CI not crossing 0, or not).

- Why this is the single highest‑value move
  - It directly addresses the main residual credibility gap in the Mario brief: whether the NARMA‑10 gains reflect a generally useful hyperparameter setting or a task‑specific fit. A clear positive narrows the story to “one set of physical params works beyond NARMA,” while a clear negative keeps the brief honest and shifts value framing to energy and the NARMA task family.
  - It is compute‑feasible in 12 hours with the current GPU path and safety wrapper, and it reuses the exact frozen config, eliminating new tuning confounds.
  - Second‑best alternatives (e.g., NARMA‑20/Mackey‑Glass or trying to shave 0.006 dec off the ngspice tail) either fail to move the Mario narrative as much or risk looking like cherry‑picking. The 28×28 sequential MNIST result, even if negative, is immediately brief‑relevant.

- Acceptance criteria for the brief
  - If powered MNIST shows reservoir ≥ projection by ≥3% with 95% CI not crossing 0: add a modest cross‑task generalization claim.
  - If not: explicitly state “no cross‑task gain at N=64 with frozen NARMA params,” and keep the energy advantage and NARMA‑10 gains as the lead.
