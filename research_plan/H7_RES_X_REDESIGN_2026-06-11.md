Now I have everything I need. The data confirms: ch7 is a near-binary saturating flag (d=29.5, mono≈0), ch5 is the graded-but-deterministic channel (d=4.82, mono=0.97). Critically, **multiple channels respond non-monotonically to the same intensity sweep** (ch0=−0.37, ch2=−0.64, ch3=+0.20, ch4=+0.47) — this is exactly the linearly-irreducible basis the Najem literature identifies. I have what I need to write the decisive recommendation.

---

# REDESIGN RECOMMENDATION — H7-RES-X: Substrate-Reservoir on the Forward Path, Target-Side Bits, Exogenous Driver

## 1. The ONE critical thing we missed

**Every H7 version (v1–v14, plus all three candidate redesigns) made the body's target a function of something the model can already see or command.** v13/v14 put the substrate in the *loss/gate* (kill-switch). H7-RES and SS-CTC put the target as `g(u_t)` where `u_t` is the observable control stream — so a transformer with attention over u-history can compute the fixed known kernel in-context and zero-PPL stays at base (the critique's verdict, p=0.12/0.08). H7-CLAMP made the consequence `z≈g(m_t)` a smooth deterministic function of the model's own motor command — internalizable by definition (p=0.13). **The unifying error: the substrate never carries a bit that is BOTH (a) needed to reduce the clean language loss AND (b) unguessable from anything in the model's input/weights/command history.** Real dependence requires the target to be a function of an *exogenous, high-entropy, model-independent* physical variable (the die's own thermal/jitter/1-f noise state, ξ_t) that no other channel supplies, read through a *frozen nonlinear reservoir* whose output a *throttled linear readout cannot reconstruct* — and the only place that bit can live is the live die. This is the "bits-in-the-target from the **uncommanded** substrate" move that none of the three candidates made (H7-RES's own critique fix #1 names it explicitly: *"Make g a function of a HIDDEN substrate variable the model cannot observe, not the observable control u"*).

## 2. Recommended new design

A synthesis of H7-RES (forward-path reservoir, throttled readout) with the decisive corrections from its own adversarial critique and the physical-NN / info-theory literature. The body is on the **forward compute path** as a frozen reservoir; the language **target** depends on the die's **uncommanded** noise state; the readout is **starved**; and the null is the **u-conditioned linear surrogate**, not zero.

### Architecture
- **Frozen Qwen2.5-0.5B** (or TinyLlama-1.1B). No LoRA on the body — only the readout adapter trains.
- **Forward path per body-window** (reuse `substrate_realtime_v3.py` @500Hz, 10ch, `thermal_watchdog.sh`):
  1. **Exogenous driver ξ_t (the fix that breaks H7-CLAMP's tautology):** the burst window is driven by `(m_t, ξ_t)` where ξ_t is genuine inter-run thermal/VRM/jitter entropy the model does NOT command. Concretely: capture the 256×10 window during a fixed-intensity burst whose *micro-timing is OS-scheduler-jittered and whose absolute thermal operating point drifts run-to-run*. ξ_t = the non-monotone-responding channels {ch0, ch2, ch3, ch4} (confirmed mono ∈ [−0.64, +0.47] in `self_effect_sweep_ikaros.json` — these move non-monotonically with intensity, so they are NOT a function of m_t).
  2. **Frozen reservoir front-end:** feed the window through the z2296 `build_best_features()` temporal-product readout (order-2/3 products at lags {1,2,3,5,8,10}, squares, deltas) → φ_t (~few thousand dims). **Non-trainable.** This is the one stage with proven fading memory (repo: MC=12.5, NARMA5=0.13). Do **not** rely on the z2506 GPU path — its current JSON shows XOR at chance (0.51); the FPGA z2296 readout is the only repo asset with verified temporal computation.
  3. **Frozen random projection** P: φ_t → R⁶⁴, fixed at init.
  4. **Throttled linear injection:** rank-r≤4 linear map of P·φ_t added as an operand at ONE mid layer. Trainable params: A, B (rank≤4) + a rank-2 LoRA on the readout head only. Nothing else.
- **Target-side bits (the decisive lever):** on designated *body positions*, the supervised next-token = a 16-symbol quantization of `h(ξ_t-window)` — a nonlinear-temporal summary (XOR-of-lags / NARMA-style) of the **uncommanded** ξ channels. Free-English positions carry no task bits. The base+text path is at chance (log 16 ≈ 2.77 nats) on body positions; the die is the only source.

### Training objective
A **single** primary loss. **No RB-anchor. No knock/shuffle/xdie gap term. No modality-dropout** (the critique showed dropout is itself a liveness-presence kill-switch generator). **No L_degenerate** (it sees the zero input and shapes behavior on it = kill-switch by another name).

```
L = CE(logits_real, target)          # the ONLY trained term
```
Train **only with the real live substrate present**. The model is **never shown** the zeroed/surrogate input during training. zero-PPL is measured purely as a held-out ablation at eval. This is the protocol H7-CLAMP's critique demanded (fix #2): *"DELETE L_degenerate… the zero-PPL bar must emerge ONLY from removing information the real path genuinely used."*

### Precise reason zeroing must break the language
On body positions the correct token is a 2.77-nat quantization of a nonlinear function of ξ_t, the die's **uncommanded** noise state. That value is (a) absent from text, (b) absent from frozen weights, (c) NOT a function of m_t or u_t (so not in-context computable — this is what kills H7-RES/SS-CTC/CLAMP), and (d) un-reconstructable by a rank-4 linear adapter from a random projection (per arXiv:2504.17503, a readout below the task's nonlinear demand cannot rebuild the kernel). The only component in the graph computing it is the die's fading-memory dynamics. Zero it → body positions go to chance → PPL explodes. A tamper-detector cannot help: there is no known answer to gate.

## 3. How it defeats the v13/v14 kill-switch

| v13/v14 mechanism | Why it was a kill-switch | How H7-RES-X removes it |
|---|---|---|
| RB-match pins real==base | No headroom for body to *help* | **Deleted.** Body positions are loss the base cannot reduce alone; the body finally has a sub-loss to lower. |
| dep_loss rewards knock/shuffle gap | A tamper-detector maximizes exactly this | **Deleted.** No gap term. Acceptance = zero-PPL≫base on *clean* held-out data + recording-only FAILS. |
| Answer already in residual stream | Tamper-detector gates a known answer | **Target = f(ξ_t)** unguessable from text/weights/command. No known answer exists to gate. |
| Additive FiLM high-gain path | Lets model scramble on mismatch | **Operand, not gate;** rank≤4 linear, below nonlinear demand → cannot rebuild kernel, cannot scramble. |
| Interoception = compute→power map | Tautological deterministic physics | **ξ_t is uncommanded** (non-monotone channels), so it is NOT a function of the model's action — internalization route closed. |

The signature inverts: v13/v14 had zero-PPL≈base + 1e5× knock gaps (detector). H7-RES-X must show zero-PPL≫base with **no knock term anywhere in the loss** (decoder).

## 4. Pre-registered falsifiers (predicted real vs null)

| Falsifier | Procedure | REAL embodiment | NULL (kill-switch / laundering / fingerprint) |
|---|---|---|---|
| **Recording-only replay** | Retrain identical pipeline on a deterministic *replay* of recorded ξ-windows; eval on held-out u | Live > replay: replay reproduces *memorized* windows but cannot supply ξ for *unseen* held-out u → replay-PPL ≫ live-PPL | Replay reproduces live-PPL → it was a memorized tape, **FALSIFIED** |
| **u-conditioned ARX surrogate** (the right null, missing from all 3 candidates) | Fit a linear dynamical image of the die's input→output map on u/m-history; inject that instead of φ_t | Real-PPL < 1.5×base AND surrogate-PPL ≫ base (nonlinear ξ-structure destroyed) | Surrogate passes → die only re-encoded u, **FALSIFIED** |
| **Transplanted-authentic-die** | Pipe live signal from a 3rd identical gfx1151 (daedalus) carrying a *different* ξ-stream; shared frozen median/MAD norm | Self-die strictly dominates: only ikaros's ξ generated *this* corpus's targets | Stays coherent on transplanted authentic body → authenticity/class check, **FALSIFIED** (report as die-class, not self) |
| **Yoked-swap** | Per-step randomize {self-live, yoked-other live daedalus under unrelated load, lagged-self, AR-sham} | Self strictly dominates on body-position CE | Self only weakly dominates → conditioning artifact, **FALSIFIED** |
| **Refit-recovery** | After training, zero substrate, refit A,B,head-LoRA on <10% held-out | CE stays at chance — bits never existed without live ξ | CE recovers → low-dim fingerprint, **FALSIFIED** |

## 5. Concrete FIRST experiment (smallest run with signal)

**Gate it BEFORE any LM coupling** (this is the step v1–v14 skipped):

**Step 0 — Bandwidth gate (½ day, no LM):** Build the corpus generator and run the z2296 ridge readout on live ikaros ξ-windows vs (a) zeroed, (b) u-conditioned ARX surrogate, on the body-token classification task (16-way). 
- **PASS only if:** linear-ridge on live-ξ-reservoir-states ≥ 70% body-token accuracy AND linear-ridge on ARX-surrogate ≤ 30% (near chance 6.25%). If the die cannot beat the surrogate at the *device level*, **STOP and report no-necessity** — no LM coupling will fix it. (This directly tests H7-CLAMP-fix's "measure KR/MC on live ξ before coupling.")

**Step 1 — Minimal LM run (1 day):** Qwen2.5-0.5B frozen, 2k examples, 10% body positions, single mid-layer rank-4 injection, ~500 steps, thermal guard every 50 steps / pause@75°C.
- **Success criteria (pre-registered, ALL must hold on held-out u):**
  - `real-PPL < 1.5 × base-PPL` (base ≈ 19.8 → real < ~29.7; target real < ~22)
  - `zero-PPL > 3 × base-PPL` (zero > ~59) driven by body positions → ~log(16) nats
  - `recording-only replay-PPL > 2 × live-PPL` on held-out u
  - `refit-on-zero body-accuracy < 35%` (no recovery)
- **Reuse:** `substrate_realtime_v3.py`, `thermal_watchdog.sh`, `z2296_best_of_all.py::build_best_features/ridge_solve`, `h7_rooted_lm_v2.py` plumbing (keep the streaming pipeline, discard the loss).

## 6. Honest probability and biggest risk

**Probability this clears the bar (zero-PPL≫base AND real-PPL<1.5×base AND recording-only FAILS): ~0.20–0.25.** Higher than the candidates' 0.08–0.13 because (a) the target-side-bits + uncommanded-ξ move structurally forecloses the in-context-compute and command-internalization routes that killed all three, and (b) we gate at the device level first. But it remains a genuine coin-flip-down, not a sure thing.

**The single biggest risk — bandwidth/replay collapse at the exogenous channel.** The whole design rests on ξ_t (the uncommanded non-monotone channels {ch0,ch2,ch3,ch4}) carrying a *faithful, reproducible-enough-to-learn-but-unpredictable* nonlinear-temporal image at 500Hz against token rate. Two ways it dies: (1) **too little entropy/bandwidth** — if ξ is near-constant or low-dim, the die underperforms its own z2296 offline benchmark and zero-PPL≈base (the modal failure, same as v13/v14); (2) **replay-equivalence** — if ξ is reproducible enough that a recorded tape of windows reconstructs held-out body tokens, the live loop is non-essential and we have a memorized-tape discriminator, not a body. Step 0's `live > ARX-surrogate` gate catches (1); the recording-only falsifier catches (2). **Pre-commit now to reporting the honest negative** if either fails — that is the most probable single outcome, and reporting "embodiment not demonstrated at this bandwidth" is the scientifically correct result, not a reason to tighten until it locks.

Relevant files: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/scripts/z2296_best_of_all.py` (reservoir readout to reuse), `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/scripts/identity_benchmark/substrate_realtime_v3.py` and `thermal_watchdog.sh` (plumbing), `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/scripts/identity_benchmark/h7_rooted_lm_v2.py` (LM streaming scaffold to keep, loss to discard), `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/research_plan/oracle_queries/O103_embodiment_bias_check_20260611/self_effect_sweep_ikaros.json` (confirms ch7 binary/ch5 deterministic and the {ch0,ch2,ch3,ch4} non-monotone exogenous channels this design depends on).