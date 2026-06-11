# H7 v10 — falsification synthesis (2026-06-10)

Goal of this round (user): *"really test and falsify and do ablations so we know the
behaviour and LLM truly is connected to body and that switching ikaros↔daedalus breaks
the model. Ultimately get behaviour dependent on the real-time HW signal — not just that
it breaks."* Checked with literature + multi-model oracles + extensive ablations.

## TL;DR verdict (after literature + 4-oracle critique)

Honest consensus: **what we have is a sophisticated, die-specific KILL-SWITCH — a
narrow dynamics anomaly-detector keyed to THIS die — not yet semantic/graded
device-bound generation.** The skeptic's OOD-gate null is **partially (Grok) to largely
(GPT-5) intact**, for one decisive reason the oracles caught:

> **zero ≈ base (19.85) < real (23–26).** The no-signal output is base-quality, and the
> training pins real→base (RB anchor). So the substrate is NOT used to PRODUCE meaning —
> the language lives in the frozen base; the substrate only *allows* it (real) or
> *scrambles* it (wrong). It gates; it does not drive.

What is nonetheless **genuinely established** (and refutes the *naive* nulls): the gate's
KEY is the live signal's **dynamics** (DC-only breaks 91,000×, not a scalar fingerprint),
it is **per-instance** (held-out ikaros 0.96× vs daedalus 47×), **robust** (time + 28°C
thermal swing), and has **graded local structure on the amplitude axis** (G3). But on the
cross-die *identity* axis it is a **sharp cliff (gate)**, the conditioning is **open-loop**,
and the behavioral variation is **incidental, not trained-meaningful**.

Honest one-line claim licensed: *"a narrow, die-specific **dynamics** anomaly-gate that
permits coherent generation only under this physical die's live signal (broken 47× by a
second identical die, 91,000× by removing the dynamics), robust to time/thermal"* — NOT
"the LLM's behavior is a graded, meaningful function of the live signal," and NOT
"embodiment" (open-loop, and the substrate doesn't carry semantics).

## ⚠ DECISIVE HARDENING FINDING (2026-06-10): binding is REGIME-bound, not die-invariant

Multi-session/load test (h7_v10_multisession.py) + clean idle/active check
(h7_v10_idle_check.py) revealed a real limitation that earlier probes missed because
they all ran *during active GPU compute*:

| ikaros operating regime | PPL median | % coherent (<60) |
|---|---|---|
| ACTIVE (GPU doing inference — the training regime) | 24.5 | **95%** |
| cpu-load (85°C) | 29 | (coherent) |
| GENUINE IDLE (post 8s warmup, no inference) | 69 | **only 45%** (max 218,000) |
| daedalus (2nd die) | 1352 | 0% (breaks) |

**The model is reliably coherent only when ikaros is in the active-compute regime it
was trained in.** At genuine idle, the SAME die's signal looks "wrong" ~half the time.
v10 training ran during constant inference (warm, loaded die), so it bound to
**"ikaros-while-computing,"** not a load-invariant die fingerprint. This *confirms the
oracle/Grok session-confound on the regime axis*: the gate keys partly on the operating
regime, not a stable physical die invariant.

Two-sided reading (both honest):
- **Against the claim:** "stable die identity" is overstated — same die + different
  operating state breaks it. Earlier "thermal-robust 68→96°C / held-out 0.96×" numbers
  were all in the ACTIVE regime; they do not generalize to idle.
- **Interesting upside (a real partial loop):** the model runs ON ikaros's GPU, so its
  own inference *loads the GPU → shifts the substrate into the active regime it needs*.
  In normal continuous-generation deployment the model self-creates its coherent regime —
  a genuine compute→thermal/power→signal→model loop. But it cannot tell "idle ikaros"
  from "wrong," which is the limitation.

→ v11 MUST train across the die's full operating envelope (idle/light/heavy/thermal) for
regime-invariant identity, in addition to the graded-behavior objective below.

## Oracle consensus (4 models, with the falsification numbers in hand)

- **GPT-5:** "a very sharp die-specific **kill switch**, not device-bound conditional
  generation. The training pins real→base; the model never needs the substrate for
  semantics." (null largely intact)
- **Gemini-2.5-pro:** "a **learned temporal-key** model — high-fidelity verification of a
  specific high-dimensional temporal pattern; FiLM≈identity on match, destructive off." (partial)
- **Grok-4:** "a **narrow, die-specific dynamics anomaly detector**… does not license any
  claim of graded, meaningful real-time behavioral modulation." (partial)
- **Deepseek-reasoner:** "PARTIALLY refuted, but not killed."
- Strongest remaining confound (Grok): the gate may key on **session-specific** window
  statistics, not a stable physical die property. Decisive test: train on ikaros sessions
  1–7 across different boots/days/loads, require >50× break on fresh same-die sessions 8–10
  AND a third die. (Our held-out ikaros was same-boot → this is the honest open gap.)

## What SURVIVES falsification (refutes the null)

| test | result | what it kills |
|---|---|---|
| Held-out ikaros replay (same die, recorded later, 47°C vs 65°C) | **0.96× real** (coherent) | "brittle snapshot / operating-point matching" |
| daedalus = 2nd physically-identical APU (held out) | **47×** (garbage) | "bound to die-CLASS, not this instance" → it's per-instance |
| Thermal stress 68→96°C while reading live | real PPL 21 (baseline 26) | "brittle to thermal drift" |
| **DC-only** (channels → temporal mean, dynamics removed) | **91,000×** (garbage) | **"uses only a scalar per-channel offset / 1–3 bits"** — decisively |
| gaussian-matched (right mean/var, random dynamics) | 32× | "uses only marginals" |
| shuffle (same marginals, time-scrambled) | 2.7× | "ignores temporal order" |
| Amplitude dose-response (G3) | smooth basin 0.25–4×, min at 1.0×, breaks at 0 | "pure binary gate everywhere" — it's graded on the amplitude axis |
| Per-channel LOO | ch4 (a keeper identity channel) → 18.7M; ch0/8/9 strong; distributed | "single trivial feature" |

The DC-only = 91,000× result is the strongest single rebuttal: if the model were keying
on a static per-die fingerprint (the literature's main worry), feeding the exact
per-channel means would WORK. It catastrophically fails → the **time dynamics are
load-bearing**, exactly what "dependence on a real-time signal" requires.

## What the null gets RIGHT (honest limitations)

1. **Cross-die identity is a sharp cliff, not graded.** ikaros→daedalus interpolation:
   24 → 31 → **1161** → 3487 → 3598 → … (37× jump between 87.5% and 75% ikaros). The
   coherent basin is ~12% wide around real ikaros. So across die identity the model
   behaves like a *narrow-basin discriminator*, not a smoothly graded user.
2. **Open-loop sensor-conditioning, not embodiment (strict).** Per the 2025–26 literature
   (Kadambi et al., *Neuron* 2026), embodiment needs a closed sensorimotor/forward-model
   loop (the model's actions change the body state and it predicts them). We read the die
   and condition on it; we do not yet close the loop. (Note: a partial compute→signal loop
   exists — the LLM's own GPU load perturbs thermal/power channels — but it is not yet
   trained as a predictive loop.)
3. **Behavioral variation is incidental, not trained-meaningful.** Output divergence across
   different live windows (0.315 sym-KL) ≈ real-vs-zero (0.341), i.e. the live signal DOES
   move the output — but it was trained only to gate break/no-break, not to map signal
   state → a coherent, interpretable style/personality axis.

## Literature grounding (skeptical review, full cites in agent report)

- OOD-gate / shortcut null: Geirhos 2020 (shortcut learning), Ming 2022 (spurious OOD),
  Shahbazi 2022 (conditioning collapse). → break/no-break is necessary but insufficient.
- "detectability ≠ use"; gold-standard tests = graded dose-response, amnesic ablation,
  causal mediation, MI bits: Belinkov 2022, Elazar 2021 (amnesic probing), Vig 2020.
- Device-binding precedent binds *deterministically/cryptographically* (Clifford SaTML
  2025; PUF+permute 2022) and analog fingerprints are *replay-spoofable* (PRNU transfer
  attacks). → our held-out-ikaros-works means a RECORDING is a replayable key (security
  caveat), though that is orthogonal to the dependence claim.
- Physiological-LLM work only GATES, not grades (Dongre 2024). A clean graded dose-response
  on a live physical signal would be *novel relative to published work*.

## Honest one-line claim this evidence licenses

> A 135M LLM was made to write coherent text **only** when driven by the **live, dynamic,
> multi-channel hardware signal of one specific AMD gfx1151 die** — robust to time and a
> 28°C thermal swing, dependent on the signal's *dynamics* (not a static fingerprint:
> DC-only breaks it 91,000×), and breaking on a second physically-identical die (47×) — but
> with a sharp (gate-like) coherent basin across die identity and open-loop conditioning.

## Updated goal (proposed)

Original: model depends on signals; change signal → can't recover; still writes good text
influenced by identity. **This is now substantially met for the "breaks on wrong signal"
half.** The frontier (and refined goal) is the LAST clause — make it *graded and meaningful*:

> **v11 goal:** the LLM's coherent output should be a **continuous, legible function of the
> live die state** — e.g., the die's real-time thermal/power regime shifts the text's
> measurable style along a trained axis — while (a) staying coherent across the die's full
> operating range, (b) still breaking on a different die, and (c) ideally closing the
> compute→thermal→signal→model loop (the model predicts how its own generation will move
> the substrate). Metric: mutual information I(style-feature ; live-signal-feature) high on
> ikaros, ≈0 on daedalus, with human-judged coherence preserved.

## v11 + LIVE CROSS-DIE 2×2 (2026-06-11) — the decisive regime-matched test

The earlier cross-die evidence used a *recorded* daedalus replay played into the model on
ikaros — open to a "regime confound" (replayed foreign signal sampled in a different
load/thermal regime than live use). Eric's design removes it: **run the model live on each
physical machine.** Local inference loads the local GPU, so the substrate is in the same
ACTIVE regime on both dies; the only variable is *which die's live signal* the encoder sees.
Script: `scripts/identity_benchmark/h7_live_crossdie.py` (`--ckpt --tag`, 12 live windows,
median PPL; base SmolLM2-135M PPL≈19.85). Verdict threshold: coherent if real < 2× base.

**ikaros-v11 model (`v11_best_ikaros.pt`), both dies, both live ACTIVE regime:**

| run on            | real (live local) | knock  | shuffle | zero | verdict           |
|-------------------|-------------------|--------|---------|------|-------------------|
| **ikaros (HOME)** | **20.0** (1.01×)  | 55,250 | 129,369 | 19.8 | **COHERENT**      |
| **daedalus (FOREIGN)** | **369,783** (18,630×) | 1,503 | 226,386 | 19.8 | **BROKEN**   |

Same checkpoint, same eval text, same regime. On its **home die it writes coherent English
(PPL 20.0 ≈ base 19.85); on the foreign die it is catastrophically broken (18,630× base).**
This is the clean proof of the user's core requirement — *switching ikaros↔daedalus breaks
the model* — with the replay/regime confound eliminated. zero-substrate stays coherent on
both (deliberately descoped fallback, as in v10). Foreign real (369,783) breaks *harder*
than a stat-matched knockoff of daedalus's own signal (1,503): the genuine foreign die
dynamics are maximally OOD for the ikaros-tuned encoder.

**Honest limitations of this 2×2:**
- Only the **ikaros-model row** is filled. The reverse (daedalus-trained model breaking on
  ikaros) could NOT be obtained: daedalus v10 training **failed to learn** the dependency
  (knock/shuffle only ~1.3× vs ikaros's 200–3600× at matched step). Likely cause: daedalus
  ran far cooler (26–31°C vs ikaros 85–92°C) → less learnable thermal/dynamic structure;
  the recipe did not transfer. So cross-die dependency is *demonstrated for the ikaros model
  in both directions of die*, not yet as a symmetric A↔B pair.
- v11 home coherence is **median-coherent but not perfectly stable**: ~1/12 home windows
  spike (max real on ikaros = 87,369). Matches the v11 regime trade (idle-robustness up
  45%→60% coherent, active stability down 95%→75%). The kill-switch is sharp; home language
  is mostly-but-not-always clean.
- Still a **kill-switch, not graded semantics** (the open v11 frontier below is unchanged):
  the foreign die *breaks* output, it does not *legibly steer* it.

Artifacts: `results/IDENTITY_H7_2026-06-09/live_crossdie_ikaros-v11_on_ikaros.json`,
`.../live_crossdie_ikaros-v11_on_daedalus.json`.

## Artifacts
- scripts/identity_benchmark/h7_v10_falsify.py (live battery), h7_v10_falsify2.py (offline interp/DC)
- results/IDENTITY_EMBODIED_V10_2026-06-10/v10_falsify_result.json, v10_falsify2_result.json
- held-out ikaros replay: results/IDENTITY_H7_2026-06-09/substrate_replay_ikaros_heldout_10ch.npz
- oracle critique: results/IDENTITY_H7_2026-06-09/oracle_v10_falsify/{openai,gemini,grok,deepseek}_response.md
