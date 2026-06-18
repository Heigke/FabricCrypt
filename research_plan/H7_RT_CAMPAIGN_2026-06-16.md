# H7 Real-Time Embodiment Campaign — plan & execution log (2026-06-16)

> Mandate (Eric): execute on everything; be wary of *our* bias toward NOT succeeding with
> embodiment, but follow the science; use as many signals as possible; pursue the vision
> (bidirectional LLM↔body coupling in real time → a real identity) but down the track that can
> actually succeed; plan, execute with subagents, run until success; use all 4 machines.
>
> Companion: [[H7_EMBODIMENT_SOTA_REALITY_2026-06-16]] (SOTA + oracle verdicts). State: [[h7_embodiment_state]].

## The reframing that makes the vision and the science the same target
Eric's vision = LLM depends on body **and** body depends on LLM, in real time → a closed loop that
constitutes an identity. That is *exactly* the Butlin **AE-2** reafferent loop, which the SOTA says
is the only honest "embodiment" claim for an LLM:
- **body → LLM** (already built): the trained adapter steers generation from the live HW vector.
- **LLM → body** (Phase 0 measures): token generation perturbs the host's live telemetry.
- **the loop + a meta-model on the ~30-50 signals** = Eric's "meta-computation": a 2nd computation
  that *introspects on* the LLM's own computation via its physical shadow, and feeds back.
The honest win condition: a closed loop where the body-signal measurably changes the LLM AND the
LLM measurably changes the body-signal, the model *models* that contingency, and an ablation proves
it is load-bearing — none of it a "loaded heater" artifact.

## Machines (all verified live 2026-06-16)
| host | role | notes |
|---|---|---|
| **ikaros** (local, gfx1151, 32c) | AMD **body** #1 — record + (later) closed-loop | laptop, zone0 trip 99°C → chunked bursts, THERM_ABORT 84, wait_cool 58 |
| **daedalus** (.40, gfx1151) | AMD **body** #2 — replicate + cross-die | training-venv (torch-rocm+transformers); ryzen_smu loaded; THERM_ABORT 85 |
| **zgx-5175** (.41, GB10, 119GB) | compute: train forward/meta-models; NVIDIA cross-substrate control | cool ~40°C; `ssh zgx` |
| **zgx06** (.51, GB10, 119GB) | 2nd GB10: parallel training / 2nd NVIDIA body | `sshpass -p 'demo234!' ssh demo@zgx06.local` |

## Signals (use as many as we can — Eric's directive)
Full live vector per sample (~50-D): per-core **Vcore** ×16 (PM table) + per-core **clock** ×32 +
**gpu_power** + **gpu_freq** + **gpu_temp** + **thermal** zones + **misc** (in*/curr*). Read-only.
Eric's meta-computation Q tested directly: does the **full 30-50-D** vector decode the LLM's
computational state better than **1-D** power? (T3 in the analyzer.)

## Phases & gates (falsifiable; no moving goalposts, but try hard before concluding)
- **P0 — Reafference instrumentation (NOW)**: `h7_rt_phase0.py` records C0_IDLE / C1_SELF /
  C3_SWAP(random-content,same-compute) / C2_YOKED(non-LLM matmul). `h7_rt_phase0_analyze.py` →
  T1 deflection, T2 forward (token-blind vs aware ΔR², rate→telem), T3 decode 30-D vs 1-D, T4 content.
  - **GREEN if** rate→telemetry R² is real (loop exists to drive) AND 30-D decode ≫ 1-D (signals
    carry introspective info). **Content-reafference (T4) may be ~0 — that's expected & honest**
    (fixed-arch LM ≈ constant per-token compute); the loop then operates via *intensity*, not token id.
  - **RED if** even rate→telemetry is at noise after the yoked control ⇒ no usable loop on this HW.
  - *Try-hard before RED (anti-bias)*: faster sampling, more signals, bigger model (more load),
    pacing as an explicit action, longer records, daedalus replicate.
- **P1 — Closed loop (Direction A core)**: build the real-time loop (adapter reads live 30-D each
  step; pacing/intensity is the agent action). Forward model of reafference; **kill-shot ablations**
  (efference-zero, efference-misalign, plant-lock, yoked) must break a pre-registered control metric.
- **P2 — Meta-computation / introspection layer**: train the 30-D→LLM-state decoder (token rate,
  entropy, CoT-vs-direct) on zgx; show 30-D ≫ 1-D; use it as the loop's "interoception."
- **P3 — Cross-substrate + identity**: NVIDIA-variant on both ZGX (cross-substrate control); per-die
  separation honesty (n small — no FAR/FRR claims; per-core NOT independent).
- **P4 — write-up**: only after a gate passes; strip every overclaim flagged by the oracles.

## Execution log
- 2026-06-16: built `h7_rt_phase0.py` (chunked-burst thermal-safe recorder, read-only telemetry,
  4 conditions) + `h7_rt_phase0_analyze.py` (ridge, time-blocked). Smoke OK on ikaros.
  Launched P0 recording on **ikaros** + **daedalus** concurrently (gpt2, 50 Hz, ~1400 samp/cond).
### P0 RESULTS — ikaros (gpt2, 50 Hz, ~50 live channels, burst-recorded; analyzer v2 group-CV)
**Core finding: a real, measurable LLM↔body coupling channel exists — intensity-mediated, not content.**
- **T1 LLM→body deflection (C1_SELF vs C0_IDLE), FAST channels** (slow temp excluded as the trivial
  heater): vcore[13] **d=10.7**, gpu_power **d=3.64**, gpu_freq **d=3.62**, vcore[12] d=−2.71,
  cur_freq[27] d=2.46. The LLM running visibly moves the body's fast electrical signals. ✓ strong.
- **T3 DECODE / meta-computation (Eric's Q): body telemetry → LLM token-generation RATE = R²≈0.47**
  (group-CV across thermal bursts). The physical signal genuinely *carries* information about the
  computation → an honest introspection/meta channel. FULL ~50-D vs 1-D power: gain only **+0.037**
  (power alone already carries most rate info; extra channels add little **for rate**).
  - mean_entropy decode: R²≈−0.6 (full), ≈0 (1-D) → **entropy NOT decodable**.
- **T4 CONTENT (honest null, as predicted)**: entropy adds ≈0 over rate (−0.003); C1-vs-C3 power
  d=0.27 (small) → **token IDENTITY does not move telemetry**; the loop is via generation INTENSITY.
- **T2 REAFFERENCE forward-model**: predicting next fast-telemetry from history barely generalises
  across bursts (blind R²≈−0.15, aware ≈−0.15, Δtoken ≈0) — ikaros data is thermally messy/bursty;
  this is the weak link to strengthen (cleaner continuous data, coarser timescale, pacing as action).

**Honest interim verdict (NOT a bias-driven dismissal — the signal is real):**
- LLM→body ✓ strong; body→decode-LLM-state(rate) ✓ R²≈0.47; content-reafference ✗ (expected).
- The loop the vision needs **exists via intensity** (rate↔power), and the body **can introspect**
  the computation. The 30-D breadth helps only marginally *for rate*; its real value is likely for
  **identity/cross-die** and richer state decode (test on cleaner data + per-channel).
- Confounds handled: per-burst standardisation removes slow thermal integrator; group-CV across
  bursts prevents memorising one regime; yoked/content controls in place.

**Infra notes**: ikaros heats to 84°C in ~3 s of gpt2 gen → chunked bursts mandatory (worked).
daedalus `training-venv` (nightly torch rocm7.13) **crashes gpt2 GPU gen** (`HIP error: device kernel
image is invalid`); `venvs/torch-rocm` (2.9.1+rocm7.1.1) is the candidate but ssh-detached relaunch
was flaky — cross-die replication pending (or use ZGX NVIDIA as cross-substrate control).

### P0 RESULTS — daedalus (CPU gen, cool 52°C, ~500 samp/cond, 3 bursts) — cross-die
- **T1 LLM→body coupling CONFIRMED on 2nd die**: C1 moves **31 fast channels** |d|>0.5; cur_freq[13]
  d=10.7, gpu_power d=8.78, vcore[8] d=−4.8. ✓ The coupling generalises across dies.
- **T3 decode did NOT replicate ikaros**: token_rate R²≈−0.6 (vs ikaros +0.47). **Confound, not
  mystery**: daedalus ran **CPU** generation (to avoid disturbing the live `ex3_resume_v2` GPU
  finetune), so the GPU-power channel that carried rate on ikaros (GPU gen) is idle; plus only
  ~500 samp / 3 bursts ⇒ group-CV starved/unstable. → decode result is currently **1-die, GPU-config
  only**; NOT yet a robust cross-die finding. Honest.
- **T4 content null replicated**: entropy adds ≈0; C1-vs-C3 power d=−0.41 (small). ✓ intensity loop.
- Infra: daedalus ssh drops long sessions (exit 255); `training-venv` nightly-torch HIP-crashes GPU
  gen; **`venvs/torch-rocm` + CPU works in foreground**. daedalus GPU is occupied by ex3 finetune.

### HONEST SCORECARD so far
- ✓✓ **LLM→body coupling**: real & strong on BOTH dies (intensity-mediated).
- ~ **body→decode-LLM-rate**: R²≈0.47 on ikaros(GPU) ONLY; daedalus(CPU) didn't replicate (confound).
  Need matched GPU-gen continuous recording w/ many bursts to claim robustly.
- ✗ **content/token-identity reafference**: null both dies (expected; loop is via intensity).
- ✗ **forward-model reafference (predict next body state)**: weak both dies (R²≈−0.1..−0.15).

### Next (P0→P1) — the decisive clean run
0. **Robust decode run**: long *continuous* GPU-gen recording with MANY bursts on a cool box —
   cleanest is an **NVIDIA-variant on ZGX** (idle, cool, fat GPU; needs nvidia-smi telemetry reader)
   OR daedalus GPU once ex3 frees it. Target: replicate rate-decode R² with GPU gen + ≥10 bursts.
   - 2026-06-16: NVML reader added to harness (works). **ZGX (zgx-5175) blocked**: ssh sessions
     drop reliably (exit 255) mid-run — foreground AND detached/tmux — an environment/network issue
     with that box (daedalus foreground works fine). Stopped thrashing ZGX. Cross-substrate decode
     deferred; NOT blocking P1 (which uses solid ikaros GPU data). Retry ZGX from a more stable
     session or fix its sshd/keepalive later.
   - **Consolidated** (ikaros+daedalus): `results/IDENTITY_H7_2026-06-16/phase0_summary.{md,png}`.

### P0 RESULTS — daedalus GPU (gpt2, dev=cuda, 40 Hz, 5465 samp / ~5 bursts/cond) — CONFIG-MATCHED RE-RUN
*2026-06-16: re-ran on daedalus **GPU** (ex3 finetune had freed the GPU; box cool 26→70°C).
Two harness bugs fixed first: (a) ROCm daemon-thread deadlock → inline sampling (`tick()` in-loop,
no background Sampler thread); (b) `dump()` temp-file `.npz.tmp.npz` mismatch → `p.with_name(p.stem+".tmp.npz")`.
Clean run, all 4 conditions, no hang, no crash.*
- **T1 LLM→body coupling CONFIRMED on 2nd GPU die**: power **d=8.12**, 6 fast channels |d|>0.5. ✓
- **T3 decode-rate REPLICATES with matched GPU gen**: token_rate **R²=0.287** (was −0.6 on CPU — the
  CPU result was a config confound, now confirmed). The body **does** decode the LLM's generation state
  on a 2nd die. ✓✓
- **T3 30-D ≫ 1-D — the meta-computation result, strongest here**: FULL ~50-D R²=0.287 vs **1-D power
  R²=0.006** → **gain +0.282**. On daedalus *no single channel* carries the LLM-state info; you need the
  multi-signal vector. Directly supports Eric's meta-computation vision, AND the per-die-unique angle
  (which channels carry the signal is die-specific: ikaros = power-dominated, daedalus = distributed).
- **T4 content null replicated**: entropy adds −0.001; C1-vs-C3 power d=0.02 → intensity-mediated loop. ✓

### CONSOLIDATED SCORECARD (2 GPU dies, config-matched)
| host | gen | #ch | C1 moved |d|>.5 | power d | decode-rate FULL | 1-D | **30-D gain** | reaff Δ | entropy adds |
|---|---|---|---|---|---|---|---|---|---|---|
| ikaros   | GPU | 55 | 10 | 3.62 | **0.469** | 0.432 | +0.037 | −0.001 | −0.003 |
| daedalus | GPU | 55 |  6 | **8.12** | **0.287** | 0.006 | **+0.282** | −0.002 | −0.001 |
**Robust (both GPU dies):** LLM→body coupling ✓✓; body→decode-LLM-rate ✓✓ (R²=0.29–0.47, now replicated).
**Meta-computation (30-D ≫ 1-D):** load-bearing on daedalus (+0.282), marginal on ikaros (+0.037) —
breadth matters and is **die-specific** (supports UNIQUE-per-die). **Content/reafference null both**
(intensity-mediated loop — honest, expected for fixed-arch LM).
1. Clean continuous recording on a cool box (daedalus venv-fixed, or ZGX NVIDIA variant) → redo T2 at
   100–300 ms horizon + pacing-as-action → get a real reafference ΔR².
2. Build the **closed loop** (P1): adapter reads live 30-D each step; agent action = generation
   pace/intensity; forward model of rate→telemetry; kill-shot ablations (efference-zero/misalign,
   plant-lock, yoked) must break a pre-registered control metric.
3. Train the **meta-decoder** (P2) on ZGX (cool, fat GPU): 30-D→{rate, entropy, CoT-vs-direct},
   per-channel importance → quantify what the multi-signal breadth really buys.
