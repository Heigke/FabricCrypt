# H7 — Substrate-Rooted AI Identity (AMD Strix Halo / gfx1151)

Make a language model **constitutively dependent on one specific die's deep, real-time
hardware signals**, so that changing the signal breaks the model — while it still writes
coherent text whose "personality" is shaped by that live substrate.

Concretely: an LM bound to chip **A** (ikaros) writes good English when fed A's own live
hardware signal, but produces garbage when fed a *different real chip* B's (daedalus) live
signal, a statistics-matched spoof, or a temporally-shuffled version of its own signal.

> This is research code from an ongoing experiment, reported honestly — including what
> failed. It is **not** a claim of machine consciousness; see "What this is / isn't" below.

## The substrate

10 channels sampled at ~500 Hz directly from the SoC via SMN / `/dev/mem`
(`substrate_realtime_v3.py`): clock counters, thermal ADCs, power/VRM telemetry, and analog
registers. A window of `WIN_LEN=256` samples is encoded into `K_TOKENS=8` conditioning
tokens. Five channels are identity-bearing "keepers"; channel 4 is the most load-bearing.

Each die has its **own** robust normalization (`global_substrate_stats.npz`: per-channel
median / MAD). The baseline operating point differs physically between two chips, so it is
itself part of die identity — feeding chip B's signal through chip A's normalization
saturates the encoder. **A model must always use its home-die stats** (stored in the
checkpoint, recoverable via the eval scripts' `--stats`).

## Architecture

`SmolLM2-135M` (frozen) + **FiLM multiplicative gating** at 2 insert layers + LoRA, with a
conv→transformer→Perceiver **substrate encoder**. FiLM (multiplicative) is what makes the
substrate *load-bearing*: additive cross-attention only *recognizes* the signal, it won't
*depend* on it (see the version history below).

## Metrics

- **Perplexity (PPL)** under the live/real window vs base SmolLM2 (~19.85). Coherent ≈ base.
- **Dependency ratios (×base)** for wrong conditions: **knockoff** (statistics-matched spoof),
  **temporal-shuffle** (same marginals, destroyed dynamics), **zero** (no signal), and
  **cross-die** (a *different real chip's* signal). Higher = the model breaks harder on wrong input.
- **Knockoff-KL (KKL)** — symmetric KL of the output distribution, real vs knockoff.
- **Per-channel leave-one-out**, **DC-only ablation**, **amplitude dose-response**, **regime**
  (idle vs active) — falsification battery.

## Scripts

| script | what it does |
|---|---|
| `substrate_realtime_v3.py` | 500 Hz substrate sampler (SMN / `/dev/mem`) |
| `h7_rooted_lm_v4a.py` | substrate encoder, `GlobalNorm`, shared constants |
| `h7_embodied_v8.py` | `FilmEmbodiedSmolLM` (FiLM+LoRA model) |
| `h7_embodied_v10.py` | v10 trainer — broke the "shuffle wall" (one missing `se_hinge` term) |
| `h7_embodied_v11.py` | v11 — regime-invariant (idle+active) + first graded objective |
| `h7_embodied_v12.py` | v12 — **fixes** the graded-feature bug + per-die `--stats` (current) |
| `h7_gen_stats.py` | compute a die's own `median/MAD` normalization stats |
| `h7_graded_probe.py` | Pearson(channel-4 feature → output entropy) with shuffle control |
| `h7_live_crossdie.py` | the clean, regime-matched **both-ways** cross-die 2×2 probe |
| `h7_v8_final_probe.py` | 4-gate scorecard + daedalus cross-die |
| `h7_v10_falsify*.py` | live + offline falsification batteries |
| `thermal_watchdog.sh` | SIGSTOP/SIGCONT a training PID around 95 °C (99 °C = ACPI trip) |

## Key results (honest)

- **v10** — broke the long-standing "shuffle wall": own live die → coherent English;
  knockoff **214×**, temporal-shuffle **17.8×**, *real 2nd die (daedalus)* **77×**.
- **Live cross-die 2×2** (regime-matched, the decisive test): ikaros-bound model on its
  **home** die → PPL **20.0** (≈ base, coherent); the **same model** on daedalus's **live**
  signal → PPL **369,783** (18,630× base, broken). Same checkpoint, same text, same active
  regime — only the physical die differs. (`live_crossdie_*` JSONs.)
- **Graded objective — failed honestly, exposed a bug.** The v11 "graded" feature was
  computed on the *raw* window where channels span 1…1e8, pinning it to a constant → the
  objective trained against a constant (no-op). v12 computes it on the *normalized* window.
- **v12 — graded coupling CONFIRMED.** Fixing the feature (normalized window) gave
  ikaros-v12 a strong graded link: Pearson(channel-4 dynamics → output entropy) = **+0.914**,
  collapsing to **−0.222** under temporal-shuffle (ratio 4.1×), text coherent. The output is
  now a continuous, legible function of the live signal — not just break/no-break.
- **v12 — both-ways cross-die is 3/4.** Per-die stats let daedalus finally learn
  (min-dependency **2.2 billion×** vs v10's 1.3×). The live 2×2: ikaros-v12 is die-selective
  (home 25.0, breaks 10× on daedalus's live signal); daedalus-v12 is coherent on its home die
  but **also stays coherent on ikaros** — it learned a real-dynamics key, not a die fingerprint.
  So the cross-die break is real but asymmetric (partly a normalization / DC-operating-point
  effect). Both models still reject spoof/shuffle by 10³–10¹¹×.

Full write-ups and per-run JSONs live in `results/IDENTITY_H7_2026-06-09/`:
`V12_RESULTS_2026-06-11.md` (latest: graded coupling + both-ways 2×2) and
`V10_FALSIFICATION_SYNTHESIS_2026-06-10.md` (falsification + oracle critique).

## Reproduce the headline results

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0       # gfx1151
# per-die stats, then train, then the two headline probes:
sudo -E python h7_gen_stats.py    --out results/IDENTITY_H7_2026-06-09/<host>_substrate_stats.npz
sudo -E python h7_embodied_v12.py --steps 6000 --stats <that npz>   # + thermal_watchdog.sh <PID>
sudo -E python h7_graded_probe.py --ckpt <best.pt> --stats <that npz> --tag <host>-v12  # graded r
sudo -E python h7_live_crossdie.py --ckpt <best.pt> --stats <that npz> --tag <host>-v12 # home/foreign
```

## Running

```bash
# substrate reads /dev/mem → needs root; gfx1151 needs the HSA override
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# 1. generate THIS die's normalization stats
sudo -E python h7_gen_stats.py --out results/IDENTITY_H7_2026-06-09/<host>_substrate_stats.npz

# 2. train (launch the thermal watchdog on its PID in parallel!)
sudo -E python h7_embodied_v12.py --steps 6000 --stats <that npz>

# 3. probe coherence + dependency, and the cross-die break
sudo -E python h7_live_crossdie.py --ckpt <best.pt> --tag <name>
sudo -E python h7_graded_probe.py  --ckpt <best.pt> --tag <name>
```

## Safety

- `thermal_zone0` hits a **99 °C ACPI trip → instant reboot**. Training has no internal
  throttle; always run `thermal_watchdog.sh <PID>` alongside it.
- Substrate reads are **read-only**. Never write the SMU mailbox or read `amdgpu_regs_didt`
  (both reboot the machine).

## What this is / isn't

This demonstrates **device-binding**: a model whose function is gated by one physical die's
live dynamics. It is a die-specific *kill-switch / learned temporal key*, not evidence of
subjective experience. The open frontier is making the coherent output a **graded, legible
function** of the live signal (the v12 objective), not just break/no-break.
