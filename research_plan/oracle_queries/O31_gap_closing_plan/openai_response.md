# openai response (gpt-5) — 132s

1) Track ordering
- Do P first, but start with the minimal CPU-safe wrapper immediately and kick off a 6-seed pilot V in parallel the same day. Don’t wait for a GPU port to begin V.
- Pull S (CI utils + paired tests) into Day 1 so the pilot is already “stats-hard.”
- Optional 1-hour R sanity: 1 seed, N=16 pyport-direct on the candidate vs baseline to catch surrogate artifacts early before you sink time.

2) Track P branch
- Choose safer-CPU wrapper now. Add hard thread caps and thermal backoff; vectorize the surrogate to reduce per-call overhead.
- Concretely: OMP_NUM_THREADS=1, MKL/OPENBLAS/NUMEXPR=1; pin workers via taskset; cap to 2–3 workers; nice/ionice low; thermal pause ≥75°C, kill >90°C.
- GPU port only if after 24–48 h the queue is still compute-bound (ETA-to-complete > 3 days) or R/T runtimes are prohibitive. ROCm on gfx1151 is a risk; don’t bet Week 1 on it.

3) Validation power calibration
- Don’t fix at 10×240; do sequential power. After 6 paired seeds, estimate σ_d = stdev(acc_candidate − acc_base) across seeds. Required seeds:
  n ≈ ((z0.975 + z0.8) · σ_d / δ)^2 = (2.8 · σ_d / 0.05)^2.
  Examples: σ_d=0.03 → n≈3; 0.05 → n≈8; 0.07 → n≈16.
- 240 test samples/seed gives ~±6.4 pp binomial noise at p≈0.5; raise to ≥1,000/seed when datasets allow (SeqMNIST, KWS). For tiny sets, keep 240 but accept wider CI or aggregate across disjoint test shards.
- Use paired testing on per-seed accuracies; don’t treat 240×10 as 2,400 iid.

4) Track C cost model
- Literature “4T ≈ 5 µm²” is too weak alone. Triangulate:
  1) Standard-cell LEF data (e.g., sky130_fd_sc_hd inv/tgate areas) as an anchor; cite exact cell area from LEF/GDS headers.
  2) Transistor-count × λ^2 with guard-ring factor (×1.5–2×) to bracket analog layout overhead.
  3) Routing-limited estimate: inhibitory fan-in adds O(N·r^2) wires; convert to area via track density and metal pitch to get a routing overhead band.
- Energy: measure in-sim. From ngspice/pyport, log incremental current/energy per inference when enabling inhibition; report mJ/inference deltas. Present low/med/high bands, plus an alternative “time-multiplexed inhibition (2 passes, ~2× latency, ~0 area)” option.

5) Decision gate at end of Week 1
- “CI ≥ +5 pp” is too rigid and not cost-aware. Use:
  - Paired 95% CI lower bound > 0 pp, and
  - Median relative error reduction ≥ 20% OR absolute +3 pp if baseline >90% acc, and
  - Consistent sign in ≥8/10 seeds, and
  - Beats an area-matched baseline (r=0 with N scaled to same estimated area/power).
- This aligns with what Mario needs: benefit > cost and survives a fair comparator.

6) Cron schedule additions
- GPU off-hours: good. Oracle feedback: make it 12 h or event-triggered (on gate crossings or variance spikes) to avoid churn. Track-audit: every 6 h is enough.
- Add one guardrail: preregister Week-1 hypotheses/metrics and lock the analysis script hash; append-only run ledger (no deletion), to control researcher degrees of freedom.

7) Killer omission
- Area/power–normalized comparator. You must show the candidate beats simply scaling N (r=0, larger reservoir) at equal estimated area/power. Without this Pareto control, Mario can’t justify a fabric change over “just make it bigger.” Add this to V/C: evaluate r=0 with N↑ to match the inhibition cost and compare accuracy/NRMSE and mJ/inference.

Stamp: Proceed with P(min wrapper) + S today, run a 6-seed pilot V and compute σ_d, raise test-set sizes where feasible, and add the area-matched baseline to the Week-1 gate.
