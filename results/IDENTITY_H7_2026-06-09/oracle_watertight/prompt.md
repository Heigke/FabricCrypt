# Oracle packet — make a hardware-rooted LLM "watertight" against a hostile reviewer

You are a hostile expert reviewer (hardware security + ML + reservoir computing). Be ruthless and concrete.

## What we built (real code + results attached)
Goal: an LLM constitutively dependent on ONE specific AMD gfx1151 die's physics — UNIQUE per die, FRESH (live), and where the body physically COMPUTES a function the network needs, hooked into training+inference so the net fails without the body. We have THREE physical computation layers across the machine:
- MACRO (firmware/SMU power arbitration, CPU↔GPU contention) — coarse, nonlinear under power-cap binding.
- MICRO (L3 cache destructive interference via two streamers, `micro_mem.c`) — computes XOR by sub-additive throughput when working sets spill L3.
- ANALOG (in-kernel GPU voltage-droop transient reservoir, `h7_transient_vdroop.py`) — per-die di/dt droop + settling, read as time-multiplexed virtual nodes.

Current demo (`h7_multilayer_demo.py`): DEMO A composes two cache layers to compute 3-bit parity (PAR3) — a linear classifier can't, native=0.944, ablations≈chance. DEMO B: analog droop reservoir, own-die=0.638 vs foreign-die(daedalus)=0.473 → claimed "unique".

## Our own red-team already found these holes (assume reviewer knows them)
1. CRITICAL — FRESHNESS is fake: DEMO B evaluates a RECORDED .npz; no live signal at inference; replay the file forever.
2. CRITICAL — n=2 dies is ~100x below the PUF uniqueness bar (no inter-die HD distribution, no BER, no temp/governor sweep).
3. CRITICAL — clonable: the target (XOR of public input bits) is u-derivable; a query-only model impersonates the readout at 0.965; the die adds nothing an attacker's input-model can't.
4. CRITICAL — "LLM" is actually a linear 2-class classifier on a synthetic parity stream, not a language model.
5. MAJOR — pass-by-construction: training uses the numpy XOR truth table; live silicon only swapped in at eval (native ce == numpy-xor ce to 4 decimals).
6. MAJOR — cache "00" cell is a trivial idle-detector; only the "11" contention boundary is real physics.
7. MAJOR — uniqueness gap inflated by per-condition alpha tuning (honest gap ~0.07, not 0.165); part of it is drive-magnitude/covariate shift (ikaros drive 1.17 vs daedalus 1.87), not die identity.
8. MAJOR — no single organ is BOTH load-bearing AND die-unique (digital=load-bearing+generic; analog=unique+not-load-bearing).

## What we need from you
1. The single KILLER attack that most cleanly destroys the whole thing — and whether any honest version survives it.
2. A concrete, minimal WATERTIGHT redesign that uses ALL THREE layers (macro+micro+analog), ESPECIALLY their genuine NONLINEAR parts, such that ONE pathway is simultaneously (a) load-bearing for a REAL LLM's text output, (b) FRESH/live at inference (challenge-bound, replay-proof), and (c) die-UNIQUE & not clonable from public queries. Be specific about the architecture, the training procedure (live-in-the-loop), the inference protocol (nonce/challenge binding), and the exact ablations/controls that would make a reviewer concede.
3. What is FUNDAMENTALLY impossible on commodity AMD silicon (so we stop chasing it) vs what is achievable.
4. The exact statistics/controls (CIs, # dies, BER, same-die-across-time positive control, drive-equalization) to report.

Attached: all current scripts + result JSONs. Tear it apart, then tell us the watertight build.
