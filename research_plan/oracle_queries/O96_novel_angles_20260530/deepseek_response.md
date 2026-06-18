# deepseek response (deepseek-reasoner) — 23s

1. **B** (trajectory-as-signature) — FP rounding accumulation is orthogonal to your 2 surviving channels and *might* expose die-level FPU path variance not killed by thermal match. **E** (attention-routing) — topology-level coupling is deeper than activation noise; if per-CU Vth truly shapes the graph, transplant breaks it. **J** (split-brain) — forces genuine interdependence, but see #3. A and C are aggregation fantasies that don’t outrun correlated noise.

2. **Already done** — self-referential hardware ID as input is covered in watermarking (e.g., Li et al., ISCA 2020: *HWN-DNN*; also *DeepMarks*). Also standard in PUF-aided model protection: the model just learns to ignore or overfit to the ID bit. “Interoception” is branding, not novelty.

3. **Engineering theater.** A split model is isomorphic to an ensemble of two half-models on separate machines. True non-fungibility requires that the *interaction* between halves only works on that specific hardware pair (e.g., through hardware-specific all-reduce timing). You haven’t created that — you’ve just made a distributed model. Viability can be broken by running half-model on a third device.

4. **No strengthening.** All 79 race pairs share the same thermal envelope, same PDN droop, same workload-induced noise. Tournament amplifies *systematic*, not *silicon*, bias. Formal proof: if each RO pair’s winner bit has Bernoulli(p) with p device-specific, the tournament winner is dominated by the highest p — which may be from layout asymmetry not true PUF. Suh & Devadas already showed single-pair RO is unreliable without calibration; tournament just multiplies the calibration headache.

5. **Independence fails.** RTN, spatial correlation, RO winrate, LDS startup — all modulated by die temperature, supply droop, and workload history (the latter two are shared across channels). Your own Phase 1B shows spatial_MSE and RTN both present, but you never measured their cross-correlation. If they’re correlated, product-of-experts double counts the common latent. This is not “weak signals” — it’s one strong confound measured 4 ways.

6. **Kill C, D, G as duplicates:** C = scaled RO (Probe B), D = memory-timing PUF (same as arbitration-based PUFs in prior art), G = DRAM-based PUF (e.g., Rowhammer PUF already studied). None adds new physics to your 2 surviving channels.

7. **11th angle: **Memory-scrub ECC signature** — read-before-write error correction patterns in GDDR6 memory reveal cell-level fixed faults. Each chip’s bad-block map is unique, stable, and orthogonal to your APU noise channels. Requires no new hardware — just EDAC register polling.

8. **Deepest realistic coupling: microcode-level instruction scheduling** — inject per-CU Vth into instruction-issue priority (which warp gets issued first). This alters the *computation graph’s execution order* irreversibly, not just activation values. Deeper than topology because microcode is baked into the GPU firmware; cannot be recompiled by the user. Activation injection and even attention routing are still representation-level, not execution-level.

9. **Known failures:** B — similar to *silicon biometrics via FP rounding* (Behnam et al., DAC 2019) showed trajectory-unique IDs, but those drift with temperature and aging; replay fails after hours. D — memory arbitration PUFs (Siddiqui et al., HOST 2021) suffer from OS scheduler interference and workload dependency; reproducibility < 60%.

10. **Build only: B (trajectory)** and **E (attention-routing on a 2-layer transformer)**. Skip J—it’s theater. Allocate remaining budget to a proper thermal-gradient experiment: measure B’s and E’s output under controlled Tdie ramp (25–65 °C). If survival, you have two *new* channels orthogonal to RTN/spatial. If they collapse (likely), you’ll at least know they’re thermal-driven — and your novel-angles exercise goes in the “negative result” drawer.
