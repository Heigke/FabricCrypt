# Oracle packet O104 — can ANY interaction mode extract die-needed nonlinear computation from a locked APU?

You are an adversarial reviewer. Police BOTH our bias toward wanting to succeed AND our bias toward
giving up too early. Be concrete and technical. Cite real work only (no fabricated arXiv IDs).

## Goal
Make a frozen LLM **constitutively dependent** on ONE specific AMD Strix Halo gfx1151 APU die: a different
chip, or replayed/recorded signals, must break its output, while on the real die it writes fine. Three
requirements of the "body": (1) UNIQUE to the die, (2) RÄKNA = perform a genuine nonlinear COMPUTATION the
model needs (not just a load reading), (3) FRESH = live, non-replayable.

## What we have SOLVED
- (1) UNIQUE: CPPC fused per-core ranking differs 75% ikaros-vs-daedalus; live 2nd-order dynamics fingerprint 14× cross-die > same-die.
- (3) FRESH: RDSEED hardware entropy (0 dup, perfect bit-balance, fresh each read).

## What keeps FAILING: (2) genuine nonlinear computation. Exhaustively tested, all NEGATIVE:
- Power/thermal/cache-latency/cache-contention/branch-predictor (PMU): all MONOTONE in total compute load → can't do XOR/parity. Driven to physical limits (Cohen's d up to 7.9).
- Exhaustive cross-channel sweep (915-dim: all pairwise products ch_i(t-a)·ch_j(t-b) + differentials): reservoir does XOR 0.95 for adjacent lags BUT loses to nonlinear-on-u=1.0.
- Coincidence detectors: store→load-forwarding stall below the ~60-cycle rdtsc fence floor (no Zen5 PMU event); DRAM open/closed-row bistable MASKED (flat 480 cyc, on-die ECC+scrambler+closed-page) on soldered LPDDR5X.
- 2D bilinear / mixed-partial probe (two independent loads a=GPU,b=CPU): 9/10 channels = pure saturation g(a+b) (load-meter+curvature); ONE channel (ch5 power/energy-rate) has a faint genuine bilinear term +0.138 R² (throttling/Vdroop) but too weak/entangled to linearly extract (necessity ≈0) and NOT die-unique.
- Online lit (verified): external EM/acoustic/power = identity or covert signaling; DRAM retention/ComputeDRAM blocked on locked APU; GPU substrate = RNG/load-meter/PUF; cache-occupancy-vector & SpectreRewind = genuine nonlinearity BUT function of COMMANDED input → self-computable.

## The structural theorem we derived (CHALLENGE IT)
For any target that is a function of a drive WE COMMAND, the model self-computes it → the die is never
NEEDED (proven: nonlinear-readout-on-u = 1.0 always). The die supplies only fading MEMORY; the
nonlinearity is the readout's. Die-necessity would require a NONLINEAR transform of EXOGENOUS (uncommanded)
state that a rank-≤4 linear adapter cannot reconstruct — but the substrate has no genuine nonlinearity.

## QUESTIONS (be specific, rank by promise, give a concrete protocol for any you rate >20% likely)
1. Is the structural theorem actually airtight, or is there a loophole? (e.g., does a rank-limited frozen-adapter readout change "necessity" even for commanded drives? does the frozen LM's own compute defeat that?)
2. Interaction MODES we have NOT tried — rate each and give a protocol:
   a. CLOSED-LOOP / recurrence: feed substrate state back into the drive (physical recurrent reservoir) — can feedback turn a weak/saturating nonlinearity into usable computational capacity (edge-of-chaos)?
   b. HYSTERESIS / regime-boundary stimulus: drive load as a ramp UP vs DOWN through the DVFS/throttle transition — is direction-dependent response (a genuine bistable) extractable computation, not a load-meter?
   c. STEP/IMPULSE response & PDN resonance: probe the transient settling after a load step — does the power-delivery-network's LC resonance / large-signal nonlinearity (which the steady-state MEAN discards) carry computation?
   d. Higher-order (3+ independent loads) interactions; non-stationary / chaotic driving.
3. Is there a fundamentally different "interaction with the system" (how we drive/read it) that we're missing — e.g., sub-microsecond readout, analog ADC of an on-die voltage sensor, exploiting a specific Zen5/RDNA3.5 microarchitectural nonlinearity?
4. HONEST verdict: is requirement (2) achievable on this locked commodity APU at all, or should we accept the published answer (Clifford et al. 2025, DeepMind: lock the weights with a die-PUF KEY, no in-forward-pass computation) and stop hunting die-computation?

Attached: the result JSONs and the key probe scripts. Give your most useful, concrete, non-hype answer.
