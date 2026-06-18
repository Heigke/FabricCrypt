# openai response (gpt-5) — 112s

Q1 — Gate-crossing risk (which would fail a strict audit and why)
- GPU-MAX-A MNIST 82.12% (≤3 pp from 84.76%): Would fail as an AMBITIOUS “gate met” claim in its current form.
  - Single-seed/no CIs: only one run reported (82.12% vs 84.76%), no error bars, hyper-sweep or fairness audit.
  - Baseline mismatch risk: NS-RAM model had extra learnable VG1 and Vd parameters; parity with tanh baseline is not established.
  - Protocol incompleteness: 3 epochs only; no augmentation/regularization parity; training details too thin for reproducibility.
  - Upstream dependency changed mid-stream (IFT sign bug; z474b patch later) — even if values are bit-identical for DC/transient, reviewers will ask to re-run or report post-patch explicitly.

- EP-NSRAM smoke 44% (3/4 gates PASS): Would fail as an NS-RAM learning result.
  - Infra-only: explicitly used tanh surrogate, not the IFT-wrapped nsram_pyport_v2 (“CAVEAT EP-smoke: used tanh surrogate, NOT pyport”).
  - Minimal data (200-sample) and no multi-seed statistics; no ablation/baseline parity.
  - Claim must be reframed as “infrastructure smoke test” only.

- z471 Mario ±0.055 dec (Id_pk ~4.30 mA): Likely passes (as calibration), with caveats.
  - Not clamp-bound (clamp lifted earlier to 0.1 A; Id_pk ~4 mA).
  - Shown across 4 biases with very low dispersion (0.024 dec).
  - Caveat: it was initially accompanied by a z461 hang; full shape/scorecard needed (addressed in z472). Keep as “amplitude calibration only.”

- z472 V1 RMSE 1.31/1.20/1.84: Would fail as a headline metric.
  - No-baseline/uninterpretable: units and normalization not specified; not tied to silicon reference; no acceptance thresholds stated.
  - It is useful internally (verifying z474b bit-identity), but not reviewer-facing evidence.

- z473 V6 self-reset PASS: Passes.
  - Clear, quantitative, not clamp-bound: t_reset 40.7 ns, Vb returns to 0.001 V; Id_pk drift only 0.007 dec; mechanism-level diagnosis consistent.

Q2 — Cherry-pick audit (are losses buried?)
- Evidence honesty-test PASSES:
  - Early negs called out: “N-Rec-DVS … DISCOVERY FAIL … synthetic-proxy” (DVS download blocked). “N-WTA failed.” “z465 … INFRA_ONLY … cannot deliver mA conduction.” “GPU-MAX-B HONEST INFEASIBILITY … 10k BBO = 70 days.” “HNRT … honest neg … NRMSE 1.096 vs ESN 0.646.”
  - Kill-shots logged prominently: “z475 HONEST KILL_SHOT … body-leak cannot manufacture Hopf.” “z476 HONEST KILL_SHOT … V7 … unreachable without topology change.”
  - Self-demotions public: “N-BENCH-A … DEAD pitches … DVS demote, HDC demote, TRNG iso-node loses.” “N-BENCH-B … DISCOVERY FAIL … AMBITIOUS FAIL.” “TRIPLE CONVERGENCE on HDC demotion.”
  - Baseline corrections surfaced: “z468 forensic … SMOKING GUN … omits I_snap_d.” “z469 … bug-fix CONFIRMED.” “Re-baselining may be required.” “z470b … Q4 falsified.”
  - Caveats attached to wins: EP-NSRAM smoke flagged as tanh-surrogate; GPU-MAX-A energy projections caveated; peripheral inclusion warning (“100–1000× degradation”) stated.

- Places to tighten (minor risks):
  - “z474 PARTIAL LAND … accept partial verdict 7/9 implicit” — acknowledge, but do not treat as 7/9 in any reviewer-facing doc until re-run completes.
  - GPU-MAX-A reported as AMBITIOUS PASS without seeds/CIs and before upstream IFT patch was merged; later corrected by z474b, but re-report explicitly post-patch.
  - EP-NSRAM “3/4 gates PASS” headline can be misconstrued; always couple with the surrogate caveat.

Overall: losses are not buried; negatives, demotions, and kill-shots are logged and synthesized (O80/O81/N-BENCH-A/B). A few partials should be quarantined from headline claims until fully re-run.

Q3 — Next highest-value experiment (vs “stop and write”)
Recommendation: Run EP-NSRAM with the IFT-wrapped nsram_pyport_v2 (post z474b), with K1 Jacobian-singularity probe and 5-seed reporting.

Why this, and why now:
- Directly upgrades an existing smoke result (44%) from “infra-only” to a physics-grounded learning result — central to Master of Noise “LEARN” and the methodological pitch (“diff pyport + EP on a physical body-state”).
- Even a null/partial result is publishable (first EP on CMOS body-state; stability/conditioning analysis), and a positive result de-risks the paper’s core framing more than any other 24 h run.
- It aligns with O80/O81 reframing (device-physics + stochastic primitive + differentiable modeling), unlike harder, lower-yield topology changes (V7).
- Operationally feasible: z474b ensures correct gradients; dataset is small; add 5 seeds, a tanh baseline with matched parameter count, and report convergence, NaNs, accuracy, and conditioning stats.

Stop-and-write comparison:
- Stop-and-write is viable for outline + current evidence, but reviewers will flag EP as surrogate-only. One 24 h EP-NSRAM-IFT run turns a soft claim into a hard, methodologically novel result, materially improving acceptance odds.

No-cheat pushback:
- Risk: EP with pyport could stall (Jacobian singularities). Mitigation: conservative step sizes, damping, and early-stop with conditioning plots. Even failure is informative and still advances the paper.
