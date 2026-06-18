# H7 Deep+Wide Plan — more grounded body computation + deeper LLM rooting (2026-06-18)

Synthesis of 4 parallel research agents (online mechanisms, online LLM-integration, internal prior-work review, grounded experiment design). Builds on ALL-GREEN priors: micro cache-XOR, meso GPU-droop (AND), macro CPU→GPU SMU (OR), prefcore fingerprint; frozen GPT-2 + Qwen2.5-1.5B made load-bearing on K = micro⊕meso⊕macro⊕fp.

## Guiding split (from research)
- **Identity = static process-variation PUF** (high uniqueness, doesn't compute).
- **Embodiment = dynamic nonlinear substrate** (computes, weakly unique, noisy).
Don't force one mechanism to do both. Our biggest gap: uniqueness is n=2 software-readable fingerprint, not PUF-grade; droop fails cross-session (BER≈0.47). New mechanisms must target either *richer die-unique fingerprint* or *genuine orthogonal computation* — and every LLM integration must be load-bearing in the forward graph (no pass-by-construction), proven with greedy decoding.

## New body-computation experiments (ranked, grounded on this HW)
Flagship new uniqueness:
- **G1 — per-CU GPU execution-unit speed map (DRAWNAPART on gfx1151).** Time tiny CU-masked FMA dispatches (hipExtStreamCreateWithCUMask + wall_clock64) at locked clock; per-CU median-cycle vector; ratios are the fingerprint. Software-only, low thermal, genuinely die-unique, spatial (orthogonal to aggregate droop). **Top priority — directly upgrades our fingerprint.**
- **B7 — cross-die thermal/EM lock-in coupling.** Shader drives power at nonce frequency f; read induced hotspot oscillation at f via PM-table (lock-in) → transfer gain/phase G(f),φ(f). Live, nonce-keyed → replay-resistant; physically shared-nothing across boxes → clone-resistant. **The prize; needs thermal guard.**
- **B3 — leakage-current temperature curvature Ioff(T).** Fixed clock, idle power at T∈{50,60,70,78}°C, fit a+b·exp(T/c); (b,c) die-unique via Vth. Real physics axis.
- **B1 — RAPL energy-quantum residue.** energy_uj delta modulo quantum over fixed micro-load → per-die ADC offset bits. Trivial/cheap.
- **B6 — GPU PLL phase-noise/jitter spectrum** (1/f slope + floor + peak). 
- **Compute front-end (generic, NOT unique): subnormal-FP + integer-divider data-dependent latency**, **thermal-RC reservoir** — real nonlinear computation substrate for embodiment, never for identity.
Honest negatives to run: **B4 DRAM latency** (fingerprints DIMM not die), **B8 BTB warm-up** (microarch-uniform). A clean fail tightens the claim.

## Layer-usability gates (pre-registered, mirror h7v2_layer_probe)
Promote a layer into K only if ALL pass, on both ikaros+daedalus, across the 4 temps:
1. separability fidelity ≥ 0.95; 2. intra-die BER ≤ 1e-2 (across temperature); 3. inter-die Hamming ≥ 0.35·len (non-overlapping under intra error bars); 4. clone-from-public-input ≤ chance+3σ (and fresh-nonce-resistant for B7); 5. orthogonality MI ≤ 0.1 bit/bit vs existing K. Negative control: all gates must FAIL on shuffled die labels.

## Deeper LLM integration (ranked depth × falsifiability × feasibility)
- **L1 — body-driven MoE routing.** Small expert bank (LoRA-sized deltas on frozen FFN); router logits a function of the body vector ONLY (no token input); wrong key → wrong experts every layer. Train with **NTL dual-key loss** (minimize LM loss on true key, maximize/KL-away on wrong key). Deepest + most falsifiable. Guard: regularize expert diversity (else experts collapse identical).
- **L2 — per-layer attention/MLP gating across ALL blocks** (multi-bit body vector, per-head gates g=σ(W·z)); error compounds with depth. Depth-ablation falsification (TCR grows with #gated layers).
- **L3 — body-as-KV** (synthetic per-layer KV from body); elegant but medium-risk (model may ignore → attention mass→0).
- **L4 — Qwen3-8B frozen, body woven across all blocks** (daedalus/ZGX memory); report honest PPL cost.
- **L5 — cross-die non-transfer (adjudicating).** Train rooted on ikaros; eval native vs daedalus-transplant vs reverse-transplant (greedy). Use B7 as keying layer (cannot exist on other box). Reverse-transplant decides exile-vs-death; pre-register to publish either way.
- **L6 — closed-loop microkernel (moonshot).** Between tokens the LM emits a HIP load pattern that perturbs the substrate, then senses the change → proprioception→agency. Replay-resistant (action = fresh nonce). Risk: per-token HIP latency / sub-noise perturbation.
Falsification suite for EVERY design: deterministic greedy decode (model contributes zero randomness); native vs spoof-z vs constant-z vs random-z; causal mediation + activation patching (restore the injected var → does PPL recover?); MI(bit; output) and per-layer MI(bit; activation) to prove depth; adversary recovery budget.

## Sequenced execution
- **Phase 0:** validation harness (reuse h7_live_vs_fused guards + h7_multisig_fingerprint block-weighting); start aging cron (B10).
- **Phase 1 (cheap, both dies):** G1 per-CU map → B1 RAPL residue → B6 PLL jitter → B8 (expected fail) → B4 (expected fail). Promote passers to K_v2.
- **Phase 2 (thermal physics):** B3+B2 (shared heating sweep) → B9 SMU hysteresis → **B7 coupling (prize)**.
- **Phase 3 (deep LLM):** L2 (toy) → L1 (toy) → L3 (toy, parallel) → pick winner → L4 Qwen-8B → **L5 cross-die** → L6 moonshot.
- Boxes: ikaros = probes + toy LM + closed-loop; daedalus = 2nd die for inter-die gates + big-model train; ZGX = cross-arch NULL baseline only.

## Honest framing to preserve
Generic-but-load-bearing physics (cache/droop/SMU/divider) ≠ die-unique. Uniqueness from process-variation (G1 + prefcore). Economic-deterrence / remote-attestation grade, not crypto PUF until ≥10–20 dies + FAR/FRR. Embodiment honestly L0–L1 until L6 reafference loop closes.

## Key sources
DRAWNAPART (GPU EU fingerprint) arXiv:2201.09956; FP-Rowhammer arXiv:2307.00143; PLATYPUS (RAPL) platypusattack.com; subnormal-FP timing (cseweb.ucsd.edu/~dkohlbre); thermal-neuristor RC arXiv:2312.12899; NTL arXiv:2106.06916; CAST (ICLR2025) arXiv:2409.05907; IA³ arXiv:2205.05638; LST arXiv:2206.06522; Deep-Lock arXiv:2008.05966; activation patching arXiv:2410.14155.

## Oracle critique synthesis (2026-06-18, openai gpt-5 + gemini 2.5-pro; grok/deepseek pending)
CONSENSUS keep: **G1 per-CU map** (highest-prob die-unique fingerprint), **B3 leakage Ioff(T) curvature** (grounded Vth physics), **B7 lock-in** (liveness/non-transfer, NOT a static key). Traps to preregister as expected-fail: B6 PLL jitter, B4 DRAM, B1 RAPL (low-entropy/architectural), subnormal/BTB (embodiment-only).
G1 lockdown recipe (both oracles — our quick try was under-instrumented → that's why BER≈0.5): fix SCLK+MCLK+FCLK+SOCCLK to a SINGLE P-state via pp_od_clk_voltage/manual (not just perf=high), 2–3 min thermal SOAK to stable elevated temp, hold ±1°C, CPU quiesce, WGP-granular (verify affinity via SQ perf counters), VGPR/L0-only kernel, fingerprint = vector-of-RATIOS (rank order), ≥1e4 samples, discard 5% tails, ICC>0.95 across temps/reboots. Negative control: let DVFS roam → uniqueness must collapse (confirms we measure silicon).
MoE routing (L1) = right call IF: capacity limits + load-balance aux loss + expert-diversity (orthogonality/group-lasso on expert deltas) + NO ungated bypass + bilevel NTL (min on true z, KL-max on permuted z). Failure mode: experts collapse to identical / a garbage fallback expert.
Two-key architecture (clean): STATIC ID = G1 ⊕ prefcore ⊕ Ioff(T) params (binding only); DYNAMIC = B7/droop live layer (liveness/load-bearing only). Don't mix roles. Add continuous canary (known-plaintext gated by z → abort if fails) to catch pass-by-construction.
Strongest attack = **surrogate-model replay**: learn f(nonce)→z from many (nonce,z) pairs, emulate z offline. Defeat with a live nonce-locked dynamic layer (B7, high challenge space).
**KILLER EXPERIMENT (both oracles' #1):** double-dissociation, nonce-locked, cross-die activation-patching. Train gated-MoE on die A with a dynamic B7 key, fresh nonce/chunk, freeze. Eval (greedy) 4 conditions: A+live z_A, A+surrogate-replay z_A', A+live z_B (foreign die), A+activation-patch (overwrite router decisions mid-forward with live z_A). Accept iff ONLY {live z_A, patch} recover PPL; surrogate and z_B fail despite identical nonces + matched marginal z stats. Report MI(bit;output), per-layer causal contribution, BER vs temp. This isolates "live body computation necessary" from "z distribution sufficient".
Tightened gates: inter-die ≥0.45·len @99%CI; intra-die BER ≤1e-3 (±15°C, 3 reboots) for identity layers, ≤1e-2 for dynamic layers with error-correcting aggregation; publish surrogate-resistance attacker budget.
EXECUTION UPDATE: G1 quick probe (gpu_cu_probe.hip + h7_gpu_cu_fingerprint.py) → intra-die BER 0.4–0.7, spread 0.04–0.23%, NOT reproducible under locked/unlocked/round-robin — but UNDER-INSTRUMENTED per oracle (no fixed P-state via pp_od_clk_voltage, no thermal soak, CU-mask granularity unverified, wall_clock64 is GPU-global). Retry with full lockdown before final verdict. Next executable: B3 leakage curvature (clean, cross-die validatable) + gated-MoE (L1) with the killer double-dissociation falsification.
