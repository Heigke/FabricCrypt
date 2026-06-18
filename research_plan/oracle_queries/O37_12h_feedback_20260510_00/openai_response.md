# openai response (gpt-5) — 146s

Q1. Decision gate for a quantitative-prediction claim
- Yes, for a scoped claim in the brief. You now have:
  - 4 independent tasks spanning proj-baseline 43–72%
  - A priori out-of-sample prediction that landed inside the CI with 0.32 pp error
  - Stable slope/intercept from 3→4 points
  - Each point individually well-powered (n=8–25; all p≪0.01)
  This clears the bar to state: “With NS-RAM fixed (leak=0.30, g_VG2=0.20, N=1000), Δ is approximately linear in projection-baseline strength and is predictable to ≈±1 pp within MNIST-like 28×28 grayscale sequence tasks.”

- Still preliminary for a broader/general claim. To clear a higher bar:
  - Add coverage outside 43–72% (≤35% and/or ≥80% proj-baseline)
  - Add at least one non-MNIST-family dataset
  - Gating proposal: ≥6 total x-points spanning ≥40 pp of baseline; n≥8–10 seeds per point; leave-one-point-out prediction error MAE ≤1.5 pp; R^2 ≥0.95; all signs match the line’s prediction.

Q2. Strongest residual risks
- Scope/attribution risk (pipeline vs NS-RAM):
  - All four tasks are the same modality/resolution and use the same projection+linear readout. A similar Δ-vs-baseline slope might emerge with any mild-memory reservoir (e.g., ESN), meaning the relationship could be a property of the pipeline, not uniquely NS-RAM. Control: rerun one or two x-points with a simple ESN matched for state size and leak; compare slopes.

- Functional form risk (linear vs saturating/threshold):
  - Four points in 43–72% cannot rule out logistic/saturating curves that look linear locally. You need points ≤35% and/or ≥80% to stress-test curvature and the 53% zero-crossing.

- Hyperparameter winner’s curse:
  - The slope may depend on g_VG2 (or leak, N). Right now the monotonic relationship is only shown at g_VG2=0.20. A small g_VG2 sweep (e.g., 0.10, 0.20, 0.30) at two baseline levels would test whether the linear trend is robust vs an isolated sweet spot.

Secondary risks to watch: N scaling (N=1000 only tested), data perturbations that shift baseline differently for the reservoir vs projection, and correlation between baseline and Δ variances.

Q3. Single highest-value 12-hour experiment
Goal: Extend validity by adding a point well below 43% baseline to test linearity and the 53% zero-crossing.

Experiment: Permuted sequential MNIST (pMNIST), fixed hyperparams
- Why this: Minimal code change; same modality; typically yields a much weaker projection baseline (<35%), providing an out-of-range x-point that can either confirm the line or reveal curvature.

- Dataset/config:
  - Task: pMNIST (fixed global pixel permutation applied to both projection and reservoir pipelines; same permutation for all seeds)
  - Train size: 1000 (to match MNIST/FMNIST main points)
  - Hyperparams: leak=0.30, g_VG2=0.20, N=1000 (identical to z235–z238)
  - Seeds: n=8 (expand to n=12 if time/thermals permit)

- Scripts (match your z235–z238 harness; only key flags shown):
  - Projection baseline:
    run_seq_task.py --dataset mnist --permute True --train 1000 --model projection_linear --seeds 8
  - NS-RAM:
    run_seq_task.py --dataset mnist --permute True --train 1000 --model nsram --N 1000 --leak 0.30 --g_vg2 0.20 --seeds 8
  Keep batching/cooldowns as used in z235–z238 to stay <92°C.

- A priori prediction and gate:
  - Measure projection baseline B% on pMNIST.
  - Predict Δ_pred = 29.8 − 0.56·B.
  - Acceptance (extend): sign(Δ_meas) = sign(Δ_pred) AND |Δ_meas − Δ_pred| ≤ 1.5 pp AND Δ_pred ∈ CI_meas.
  - Failure (challenge the claim): sign flip OR |Δ_meas − Δ_pred| > 3 pp OR Δ_pred ∉ CI_meas.

- Expected outcomes:
  - If B ≈ 25–35%, the line predicts Δ ≈ +10 to +15 pp. A close hit strongly extends the range and supports the 53% zero-crossing.
  - A large miss or sign flip would directly challenge linearity/zero-crossing and narrow the claim.

If time remains after pMNIST completes cleanly, a fast add-on to further probe curvature is a second baseline level on the same dataset (e.g., pMNIST train=200) to obtain two low-x points without new tooling.
