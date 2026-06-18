# O98 — Hardware Identity: Broader Mechanisms Hostile Critique
**Date**: 2026-05-30
**Project**: FEEL / Master of Noise — identity-as-stake sub-programme

## Context (compressed)

Twin AMD APUs (HP Z2 Mini G1a, Ryzen AI Max+ PRO 395 / Radeon 8060S, gfx1151,
ROCm 7.0). Same SKU, same BIOS, sequential manufacture. Goal: surface a
*load-bearing*, emergent, silicon-bound identity signature that a reservoir
or LM could *use* (not just recognise).

## What we have already tested

**14 channels** (9 NULL + 5 mixed):

- NULL: stable-bit PUF, 1/f knee, RTN (thermal artefact), reservoir-transplant
  matrix, F self-referential, J split-brain (information channel but not
  defended), tournament RO, Lorenz per-CU, EDAC ECC map.
- Device-envelope DISCOVERY (silicon-bound? ambiguous): **power d≥8**,
  **thermal-τ d=7.7**, **per-core latency rank r=-0.51 d=3.37**, **TSC drift σ
  18× ratio**.
- Phase-1c restart: LDS startup byte-identical (10k reps); RO pair race
  deterministic; Vth-sweep / VRM-glitch disabled for thermal safety.

## What we now enumerate (this packet)

Attached: `IDENTITY_BROADER_MECHANISMS_2026-05-30.md` — **34 NEW mechanisms
B1–B34** across 7 categories: active dynamics, EMI/power-quality, wear/time,
fuse/firmware, cross-channel/2nd-order, topological/fabric, behavioural.

Plus our prior catalogues for reference:
- `IDENTITY_MISSED_MECHANISMS_2026-05-30.md` (M1–M17)
- `IDENTITY_NULL_PAPER_2026-05-30.md` (9-attack rigorous null)
- `IDENTITY_DEEP_2026-05-30_REPORT.md` (5-angle deep test)
- `deep_analog_access_report.md` (original 32-channel attack-surface map)

## Your job — be hostile

Answer all 5 questions in plain prose, no hedging. We will weight responses
4-way (you, GPT-5, Gemini-2.5-Pro, Grok-4, DeepSeek-R) and act on majority.

1. **Of B1–B34, which 3 are most likely to surface a NEW silicon-bound signal
   (effect size d > 2) that we have not already captured by power /
   thermal-τ / latency-rank / TSC-σ?** Give physics reasoning.

2. **Which of B1–B34 are duplicates** of channels we have already tested or are
   provably trivial restatements of power / thermal-τ / latency / TSC? List
   them with the channel they collapse onto.

3. **What 5 categories are we still blind to entirely?** (We acknowledge in
   the doc: acoustic/coil-whine, conducted EMI on 12 V, mains harmonics,
   WiFi/BT TX deviation, optical/chassis vibration — but go beyond these.
   Think weirder: chemistry, magnetics, photonics, packaging stress,
   crystalline anisotropy, anything where two ostensibly identical APUs
   *must* differ at the physics level but we have not even named the
   measurement.)

4. **Of our top-10-by-cost (B3,B4,B24,B27,B12,B26,B11,B5,B25,B30), what is the
   single most likely false-positive trap?** (i.e. which one will look like a
   discovery but will turn out to be a thermal/EMI/scheduler/calibration
   confound on Strix Halo?)

5. **Methodological gap**: we have only run cross-machine paired tests. What
   would *within-machine, across-power-cycle* tests tell us that
   between-machine tests cannot? Specifically: is there a substrate channel
   that would be different between two boot sessions of the same machine
   (suggesting environmental/state binding) vs one that is *only* different
   between machines (suggesting per-die binding)? Name 3 mechanisms whose
   answer to this question would falsify our current framing.

## Format

Plain markdown, headed by question number. Brutal honesty preferred over
politeness. If you think we are *still in the wrong layer entirely* (e.g.
"stop probing the user-space envelope and go back to FPGA"), say so on top.
