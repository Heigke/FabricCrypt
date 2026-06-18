# openai response (gpt-5) — 133s

Q1 — FRAGILITY
- The HDC “headline” is the weakest point. You’re mixing conditions to tell a single story.
  - Noise-robustness claim vs gating conflict: In z293 4B2_noise, σ=0.05 yields mean_acc=0.5909 (FAIL) and σ=0.1 yields 0.5486 (FAIL), with gates_locked.4B2_sigma005_within_1pp=false. Yet the summary line says “σ-robustness via N=2048 → 80.4%.” That 80.4% comes from z302 at N=2048 (mean_energy≈4.50 nJ/inf), not from the pre-registered 4B2 run. Overclaim: you cannot present “2.3 nJ/inf” (N=1024) and “80.4% at σ=0.05” (N=2048) as a single headline without stating they occur at different N/energy.
  - Boundary-picking: z293 4B3_vd_grid best cell is at the boundary (vd_high=1.5, vd_low=0.0), with gates_locked.4B3_local_max_interior=false. That’s fragile to small setup changes.
  - Noise model gap: AWGN is injected at feature level; no evidence it reflects sensor/hardware noise. No silicon energy measured; the 2.3 nJ/inf is model-derived.
- Secondary fragility: RNG “AMBITIOUS” is weakly supported.
  - NIST: 5-test subset, single 1e6-bit stream (seed=42), dft_spectral p=0.0435 (barely above α=0.01), no multiple-testing correction, not the full SP800-22 regimen.
  - MCMC: single-seed, single-run ESS ratio 1.0329 with virtually identical acceptance rates (0.6395 vs 0.6322); no CI. Generation overhead (0.384 s) not amortized.

Q2 — SINGLE-EXPERIMENT FALSIFICATION
- Rerun SA4 (z304) with the SA1-canonical knobs that were missing, under the same 33 IV curves, and keep the current topology unchanged (no new diode/cap). Exact rerun:
  - Constraints: single global Bf across all branches; per-branch NFACTOR ∈ [1, 15] (not clipped at 3), enforce MBJT step 0→1 at V_G1≈0.3 V, K1 fixed per V_G1 as in SA1; sweep per-branch Vaf and alpha0; Rs ∈ {0, 1e10} as before.
  - Gates (unchanged): median forward log-RMSE across all 33 curves < 0.5 dec; SAFETY: no branch worse than 1.5 dec.
  - If this rerun flips today’s outcome and passes the gates, it undercuts the core claim that “pure parameter refit cannot bridge the DC gap and topology changes are mandatory.” That would invalidate the SA3→SA4 causal narrative and the v4.4 “DC-only but topology-blocked” positioning.
  - Rationale: SA1 found NFACTOR ranges up to 12.15 and a hard MBJT step; z304 did not include NFACTOR as a free per-branch parameter, so the “incompatibility” could be an artifact of a clipped parameterization.

Q3 — NO-CHEAT DISCIPLINE
- We did drift in one place: presenting noise robustness as a headline while the pre-registered 4B2 noise gate failed.
  - Citable lines:
    - Headline: “HDC headline: 80.23% n=20 UCI-HAR, σ-robustness via N=2048 → 80.4%”
    - But z293 4B2_noise shows σ=0.05 mean_acc=0.5909 (FAIL), σ=0.1 mean_acc=0.5486 (FAIL), gates_locked.4B2_sigma005_within_1pp=false.
    - Also, the 80.4% at σ=0.05 is at N=2048 (z302 B_nscale), whereas the 2.3 nJ/inf headline energy corresponds to N=1024. Mixing these without stating the trade-off violates honest headline discipline.
- Minor discipline concern: RNG “AMBITIOUS” label (DS-N3) is based on a single-seed, single-run ESS ratio; no pre-registered replication/CI. If “AMBITIOUS” implies superiority with statistical backing, this overreaches the protocol.
- Otherwise, gates and negatives were reported straight (e.g., SA4 per-branch FAILs, z303 “mario” configs degrade fit, BONUS_snapback fail, 4D GATE not SHIP).
