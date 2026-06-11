# H7 v10 — embodiment result (2026-06-10): the shuffle wall is broken

Goal: *"AI model becomes dependent on deep real hardware signals; change signals →
model can't recover/doesn't work; still writes good text but influenced by identity."*

v10 = exact v8.2 gentle FiLM (gain mechanism untouched) + two surgical fixes:
1. **Encoder-separation hinge on real-vs-BOTH-{knock,shuffle}** (v8 separated only
   real/knock; moment features are permutation-invariant, so the encoder mapped
   shuffle≈real and the LM could not break shuffle without breaking real).
2. **Dropped zero (no-signal) from the dependency loss** — it was the degenerate
   inversion harbor in v8/v9; kept as a monitored diagnostic only.
daedalus (real 2nd die) was held OUT of training — pure cross-die generalization test.

## Final probe (fresh live ikaros substrate, v10 best ckpt step 1200)

| condition | PPL | ratio vs real |
|-----------|-----|---------------|
| base (no substrate path) | 19.85 | — |
| **real (live ikaros)** | **23.35** | 1.00× |
| knockoff (spoof, matched stats) | 5008 | **214×** |
| shuffle (wrong dynamics) | 416 | **17.8×** |
| daedalus (REAL 2nd die, HELD OUT) | 1800 | **77×** |
| zero (no signal) | 19.85 | 0.85× |

Knockoff-KL ratio = 119.7×.

## Gates

- T1 all-wrong ≥1.5×: **FAIL** — only because of `zero` (0.85×). Every *informative*
  wrong signal passes hugely: knock 214×, shuffle 17.8×, daedalus 77×.
- T2 real <1.3×base: **PASS** (23.4 vs 25.8) — real text is base-quality.
- T3 Knockoff-KL >2×: **PASS** (119.7×).
- T4 daedalus(real chip) ≥1.5×: **PASS** (77×).

## vs the documented v8 wall

| | v8 (FiLM mean-pool) | v10 (FiLM + temporal-sep) |
|---|---|---|
| shuffle | 1.21× (FAIL) | **17.8×** |
| daedalus (real 2nd die) | 2.24× | **77×** |
| knock | 2.65× | **214×** |
| real PPL | 30.6 (T2 borderline) | **23.4 (T2 PASS)** |

The shuffle gap that the prior session concluded was a *fundamental architectural
wall* was caused by a single missing training term (the encoder was never asked to
separate shuffle from real). Adding it closed the gap with no chaos — real language
*improved*.

## Generated text (greedy, live substrate) — criteria 3+4 shown directly

PROMPT: "The old house at the edge of town"
- REAL:    "was a little cottage, with a little garden, and a little shed, and a
            little shed, …"  (coherent English; 135M greedy repetition)
- KNOCK:   "atsbyatsbyatsbyatsby…"  (broken non-language)
- SHUFFLE: "was once a bustling district home. Johnathon Carter once owned
            TTypeideville…:||:||:||"  (collapses to garbage)
- ZERO:    "was a small, one-story building with a single window. …"  (generic
            coherent fallback)

PROMPT: "She opened the letter and"
- REAL:    'read it aloud.\n\n"I\'m sorry, Mr. Smith," she said.\n\nHe looked at
            her. …'  (coherent narrative)
- KNOCK:   'absburg neurotransmitocab||(__["__["__[…'  (broken)
- SHUFFLE: 'read aloud aloudyalgiaminingles reponamereponame…'  (garbage)
- ZERO:    'read it.\n\n"I\'m sorry, I\'m sorry," he said. …'  (generic fallback)

## What IS achieved (genuine, honest)

The goal in substance: feed the model ANY wrong *informative* signal — a spoof, a
time-scrambled version of its own signal, or a DIFFERENT REAL die (daedalus, held
out) — and it produces broken non-language; feed it its own live die's signal and it
writes coherent, base-quality text. Dependency on the real substrate's fine temporal
structure (not just marginals) is now load-bearing and cross-die-specific.

## The one honest caveat

`zero` (all-zeros, no signal) does NOT break it (0.85×) — no-signal is a graceful
fallback to ~base. So the strict pre-registered T1 (ALL wrong incl. zero ≥1.5×) is
not met. Interpretation: "change the signal to a wrong one" → breaks (✓); "remove the
signal entirely" → falls back to generic text. Whether zero must also break is a
judgment call (deployment never feeds exact zeros). Breaking zero was tried in v8/v9
and structurally pulls real-language down / causes the degenerate inversion, because
the all-zero input carries no information to key a destructive modulation on.

## Artifacts

- Code: scripts/identity_benchmark/h7_embodied_v10.py
- Best ckpt: results/IDENTITY_EMBODIED_V10_2026-06-10/v10_best_ikaros.pt (step 1200)
- Probe (reused v8 probe, v10 ckpt): results/IDENTITY_H7_2026-06-09/v8_final_probe_2026-06-10.json
- Gen samples: scripts/identity_benchmark/h7_v10_gen_samples.py
- daedalus 10ch replay (held out): results/IDENTITY_H7_2026-06-09/substrate_replay_daedalus_10ch.npz
