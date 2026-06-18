# H7 Embodiment — SOTA Reality Check, Honest Gap Analysis & Plan (2026-06-16)

> Triggered by Eric pulling back the LinkedIn post ("kändes barnsligt"). Mandate: stop going
> shallow. Do proper SOTA research with subagents, understand what *real* embodiment requires,
> assess honestly what we have / succeeded / are missing, lay a plan, check with oracles.
> Research run autonomously by Claude Code for Eric Bergvall. 5 parallel research subagents +
> the current H7 state ([[h7_embodiment_state]]). All claims cited to primary sources.

---

## 0. The uncomfortable one-paragraph summary

Across five independent literature sweeps the verdict is consistent and it is not flattering.
**(1)** Our "bind a model to hardware" framing is near-identical prior art — Clifford et al.,
*Locking ML Models into Hardware*, SaTML 2025 — and the one genuinely fresh piece (a learned
steering adapter on *live telemetry* with graceful degradation) is, as built, the **weakest
possible** binding: the signal is software-readable (replay/spoof-trivial), the adapter is a
differentiable module (fine-tune / discard-trivial). **(2)** Our "unique + uncopyable" claim
rests on n=2 dies and one cosine; Vcore is software-settable (Plundervolt) and drifts with
temperature/governor/load/aging — this is *fingerprinting*, far below PUF-grade, and the
uncopyability is actually carried entirely by the TPM, not the physics. **(3)** The TEE seam is
structural and hardware-class: gfx1151 has **no GPU TEE**, consumer Ryzen has **no SEV-SNP**,
and even our NVIDIA GB10/ZGX box has Confidential Computing **disabled** — so on *every* machine
we own, a root user reads plaintext weights after unseal. **(4)** On the embodiment ladder our
"voltage-steered LLM" sits at **L0–L1** (sensor-as-input / hardware-as-constraint) which Butlin
et al. (TiCS 2025) explicitly classify as **non-embodiment** for LLMs — "operational embodiment"
is a coinage that reads as spin. **(5)** On commodity AMD silicon the device physics **cannot do
computational work** for an LLM; it can only be a (real, defensible) non-replayable
entropy/identity source. Eric's "barnsligt" instinct was correct on the literature.

**But the same five reports also point, unanimously, at two genuinely defensible and reachable
targets** — one scientific (the first honest *embodiment* result for an LLM-on-its-host), one
security (a properly-evaluated hardware-rooted binding). Both are within reach of the hardware we
have. The rest of this doc is the evidence, the gap, and the plan to actually hit them.

---

## 1. SOTA synthesis (cited)

### 1A. Binding / locking models to hardware
- **Clifford et al., SaTML 2025** (arXiv:2405.20990). Derives a device fingerprint (they list
  *clock fingerprinting, finite-precision/FP fingerprints, SRAM PUFs* — our Vcore/clock is in
  their taxonomy) and uses SHA-256(fingerprint) as a key to a weight transform with three
  desiderata: Destruction, Encryption, **Indistinguishability** (their novel bit — pre-map the
  Gaussian weight dist to uniform so wrong-key candidates can't be screened cheaply, forcing full
  inference per brute-force guess). ResNet18/CIFAR10: 95.4%→10% wrong-key. **Their own disclaimer:
  "none of the mechanisms… on their own provide security… in no way prevent model extraction."**
- **Lineage**: Deep-Lock (2020), HPNN (ePrint 2020/1016), NN-Lock (JETC 2022), and the 2025 SOTA
  for *generative* models **LLA** (arXiv:2512.22307) — key-gated FFN-neuron permutation in a HW
  fabric; explicitly *non-cryptographic* (obfuscation + computational hardness), distillation can
  bypass.
- **The decisive attack reality**: *Game of Arrows* / ArrowMatch (USENIX Security 2025) recovers
  obfuscated weights at **>98%** — "obfuscation ≈ no protection." Community consensus 2025–26:
  credible binding = **attestation + key-release rooted in certified silicon** (H100 CC,
  TrustZone), not learned tricks.
- **Verdict on ours**: the learned-steering-on-live-telemetry construction appears *novel in
  construction* (found no paper doing exactly it) but is the weakest binding because (i)
  telemetry is replayable/spoofable, (ii) the adapter is fine-tune/discard-trivial, (iii) TPM is
  a key store, not a binding of computation. To be a *contribution* it must condition on an
  **unspoofable challenge-response** (not readable values) and survive a published attack
  battery, claiming only **economic deterrence**, never cryptographic security.

### 1B. Hardware device identity (PUF / fingerprint / attestation)
- **PUF-grade bar**: inter-die Hamming distance ≈ 0.5, intra-die BER < ~2% (often <1%), NIST
  SP 800-22/90B entropy, resistance to ML modeling attacks (arbiter PUF modeled to 94%, 3-XOR to
  98.8%). Papers test **5–20+ dies** with full temp/voltage/aging sweeps and 10³–10⁵ re-reads.
- **Analog fingerprinting** (clock skew, Hot Pixels S&P 2023 at 60–94% cross-device, DVFS): real
  but **temperature/governor/load/aging-dependent**, and Vcore is **software-settable**
  (Plundervolt CVE-2019-11157) → observable ⇒ modelable ⇒ replayable/spoofable.
- **TPM fresh-nonce quote** (RFC 9683/9334): proves **liveness + "a TPM holds the key"** — NOT
  *which* machine and NOT uncopyability, unless bound to a certified **EK/DevID cert chain**
  (else relay attack). Physical breaks: dTPM bus sniffing (~$40 FPGA), TPM-FAIL, cold boot
  (36–41% DDR4/5 retention). RFC's own §5.5.2.4: a stolen TPM is "indistinguishable from an
  authentic device." (Our AMD likely uses **fTPM in the PSP** → immune to bus sniffing, but then
  uncopyability rests on PSP firmware, not a physical primitive.)
- **Verdict on ours**: cosine −0.306 between **two** dies is an existence proof of process
  variation, **not a uniqueness claim** (one pairwise distance ≠ a distribution; no FAR/FRR, no
  environmental stability, no entropy estimate). The uncopyability is **entirely the TPM's**, and
  even that needs EK-cert binding to mean "this machine."

### 1C. TEE / confidential inference (the seam)
- **Real binding exists only on server/datacenter silicon**: Intel TDX & AMD SEV-SNP CPUs
  (<10% throughput / ~20% latency for Llama2-7B/13B/70B, arXiv:2509.18886), NVIDIA H100/H200 CC
  (4–8% overhead, VRAM encrypted + GPU attestation), Blackwell TEE-I/O. Attacks exist
  (RMPocalypse CVE-2025-0033, HECKLER/WeSee) but the *software-adversary* guarantee is real.
- **Our hardware, confirmed gaps**: gfx1151/Radeon 8060S (Strix Halo) — **no GPU TEE**; consumer
  Ryzen — **no SEV-SNP** (EPYC-only); **GB10/DGX-Spark/ZGX — CC explicitly disabled by NVIDIA**
  (forum confirmation Oct 2025). On all three boxes, after unseal the weights/activations/"signals"
  sit in plaintext unified RAM readable by root/DMA.
- **Verdict**: We may honestly claim only **fTPM/PSP-sealed key release tied to measured boot +
  a non-exportable key** = "this key unlocks only on this enrolled machine in this boot state"
  (a *gating/licensing* property), and must state plainly it is **broken by any root user once the
  model is resident**. A true confidential-inference claim requires moving the workload to an
  H100/H200 CC GPU or an EPYC/Xeon SEV-SNP/TDX VM (cloud) — not our local silicon.

### 1D. What "embodiment" actually requires (the ladder)
From enactivism (Varela/Thompson/Rosch), sensorimotor contingencies (O'Regan & Noë 2001), Brooks
1991, Lakoff & Johnson 1999, Clark (extended/predictive mind), morphological computation
(Müller & Hoffmann 2017), and Butlin et al. *Consciousness in AI* (TiCS 2025):

| L | Name | Requires | Loop? |
|---|------|----------|-------|
| **L0** | sensor as input feature | HW state is just an input dim; output doesn't affect it | none |
| **L1** | hardware as constraint/key | physical state gates/keys behavior; open-loop, non-representational | none |
| **L2** | body shapes format of cognition | morphology determines concept structure (image schemas/metaphor) | weak |
| **L3** | morphological computation proper | substrate's nonlinear dynamics do task-relevant compute w/ defined encoding + trained readout + a "user" (Müller-Hoffmann) | optional |
| **L4** | modeled output→input contingency | system **models how its own outputs change future inputs** and uses it (reafference) — **Butlin AE-2**; met even by a virtual avatar | yes (modeled) |
| **L5** | mastered sensorimotor contingencies | know-how of lawful action→sensation map, constitutive of perception (O'Regan & Noë; Brooks) | yes (mastered) |
| **L6** | autonomous sensorimotor agency / autopoiesis | operational closure, self-maintained norms, sense-making | yes (constitutive) |

- **Butlin AE-2 (Embodiment), verbatim**: "Modeling output-input contingencies, including some
  systematic effects, and using this model in perception or control." Butlin's own verdict: LLMs
  *fail* AE-2 because they "don't model how their outputs affect environmental inputs."
- **Honest placement of our system**: **L0–L1.** Telemetry → feature → logit modulation is L0
  (exactly what Butlin says fails AE-2); used as own-die-unlocks-key it is L1 (a real *hardware
  dependency*, not embodiment). **"Operational embodiment" is overclaiming.** Defensible terms
  today: *hardware-coupled, substrate-rooted/substrate-dependent, physically-grounded I/O*.
- **The cliff is L1→L4**: everything at L4+ requires the model's *own actions* to perturb what it
  later senses, and the system to model+use that. Generating tokens **does** heat/load the APU, so
  a genuine output→telemetry→input loop is **physically real and measurable** — this is the door.

### 1E. Physical computation on commodity silicon (the "räkna" dream)
- **Physics that computes** (memristor crossbar MVM, p-bit/thermodynamic sampling — Extropic
  XTR-0/Normal CN101 are real tape-outs but **simulation-benchmarked, custom chips**; photonic/
  spintronic reservoirs; analog Ising) — **all require purpose-built hardware**. Even "neuromorphic"
  flagships (Loihi 2, NorthPole, SpiNNaker2) are **digital emulation**, not analog physics.
- **Physical reservoir computing**: the readout is *linear*, so the substrate must supply genuine
  nonlinear fading-memory state expansion (memory–nonlinearity trade-off, *Sci.Rep.* 2017). A
  stock CPU/GPU read into a software adapter does **not** — the adapter does the nonlinearity.
- **Commodity-silicon honest ceiling**: device physics as **non-replayable entropy/identity** is
  real and mature — CPU jitter-entropy (Linux `jitterentropy`), RDRAND/RDSEED (AMD: ring-oscillator
  jitter; caveat: >97% RDSEED starvation under sustained load on some Zen), DRAM/RowHammer PUFs
  (FP-Rowhammer AsiaCCS 2025, runtime-queryable, unique across identical DIMMs). Device physics
  *computing* for an LLM on stock CPU/GPU = **no published success**; the one paper (SHA-256 ASIC
  reservoir, arXiv:2601.01916) is explicitly speculative. Our own integrated-pipeline result was
  "ALL negative for MNIST, root cause = single MAC channel" — same conclusion.
- **The clean operational test**: physics *computes* iff a fixed/linear readout solves a task it
  couldn't from raw input AND removing the physics collapses performance (true ablation). Else the
  physics is *noise/identity* and the software does the work. **Ours is noise/identity.** That's
  not failure — it's the honest, citable claim (TRNG/PUF literature legitimizes it).

---

## 2. Honest gap analysis: what we HAVE, what we SUCCEEDED at, what we're MISSING

### What we genuinely have (verified, real)
1. A frozen GPT-2 + small adapter whose **text quality is causally, deterministically dependent**
   on a per-machine fingerprint (own ≈ plain GPT-2; wrong key 150–2200× worse; null gate passes
   both dies). The dependence is real and probe-verified (we *disproved* the untrained-steering
   fluke ourselves — that rigor is a genuine asset).
2. A **TPM-sealed cross-die transplant** that unlocks on the own die and refuses on the foreign
   die, both directions, gated by a fresh-nonce quote (liveness/anti-replay).
3. A measured decomposition: time-averaged z-Vcore = identity (cross-die cos −0.306), instantaneous
   deviation = freshness; signal-drift quantified on both dies.
4. Working thermal-safe training on ikaros + daedalus; a real toolchain (probe surgery, null gates).

### What we actually succeeded at (the defensible core)
- **Honest methodology**: we caught our own fluke (untrained steering = random vector) before
  shipping. That self-falsification is the most valuable thing here and must be the spine of any
  future claim.
- A **working hardware-coupled LLM pipeline** end-to-end on real silicon — a solid *engineering*
  substrate to build the real experiments on.

### What we are missing (the gap, blunt)
| Claim we leaned on | What the SOTA demands | Status |
|---|---|---|
| "embodiment" | L4 Butlin AE-2: output→input loop, modeled + ablation-load-bearing | **MISSING** (we're L0/L1) |
| "unique / uncopyable (physical)" | ≥20–30 dies, FAR/FRR, temp/governor/aging sweeps, anti-spoof/replay/model attacks | **MISSING** (n=2, one cosine, Vcore spoofable) |
| "fresh / non-replayable" | challenge-response bound to secret key material, not readable telemetry | **PARTIAL** (TPM nonce ✓; telemetry "freshness" = noise, replayable) |
| "räkna / computation in the body" | physics passes the fixed-readout ablation test | **MISSING / likely infeasible on commodity AMD** |
| "bound to this chip" (security) | attested key-release into a TEE; survive ArrowMatch/fine-tune/replay | **MISSING** (no TEE on any box; adapter discardable) |
| binding is a contribution | quantified failure of replay/spoof/fine-tune/adapter-discard attacks | **NOT YET RUN** |

**Root cause of "shallow"**: we built a *demo* (does the output change? yes) and dressed it as
*science* (is the change embodiment? is it unique? is it uncopyable? is it computation?) without
running the experiments that those four words require. The SOTA for each word is specific and we
met none of them.

---

## 3. The two defensible directions (both reachable on our hardware)

The five reports independently converge on exactly two targets that are (a) honest, (b) novel
enough to matter, (c) reachable with 2× gfx1151 + 1× GB10.

### Direction A (SCIENCE) — the first honest *embodiment* result for an LLM on its own host
Climb L1→L4 (Butlin AE-2). Concretely prove a **closed reafferent loop**:
1. The LLM's *own token generation* causally perturbs its host telemetry (power/thermal/clock) —
   interventionally, not just correlationally (it provably loads the APU).
2. The model maintains a **forward model** of "if I emit X, my next telemetry shifts by Y"
   (reafference), and *uses* it in control (distinguish self-caused from external telemetry change).
3. A **kill-shot ablation** shows the loop is load-bearing (behavior/uncertainty degrades when cut).
This is physically plausible on our APU, nobody has shown it for an LLM-on-its-host, and it
converts "shallow" into a citable AE-2 result. **This is the headline.**

### Direction B (SECURITY) — a properly-evaluated hardware-rooted binding (honest scope)
Stop claiming embodiment for the lock; make the *binding* itself rigorous:
1. Replace readable-telemetry conditioning with an **unspoofable challenge-response** — TPM
   challenge-response and/or a **DRAM/RowHammer PUF** for stable identity + **RDSEED/jitterentropy**
   for fresh non-replayable entropy; bind the TPM to its **EK/DevID cert chain**.
2. Run the **attack battery** with quantified failure of each: replay of a captured trace, a
   learned spoofer/surrogate of the signal, fine-tune-to-recover (< task-train cost), adapter-discard
   (use public backbone), ArrowMatch-style weight recovery.
3. State the guarantee as **economic deterrence / substrate-gated licensing**, never crypto
   security; position explicitly against Clifford 2025 and confidential computing.
Scale the uniqueness evidence as far as the hardware allows (be honest that n is small) using
the **per-core** structure as pseudo-replicates and any extra boxes we can borrow.

**Explicitly NOT pursuing** (per Eric + the physics report): claiming the commodity APU *computes*
for the LLM (no FPGA/NS-RAM) — the honest ceiling there is entropy/identity, already covered by B.

---

## 4. The plan (phased, falsifiable, with kill-criteria)

> Each phase has a **kill-criterion** stated up front. If hit, we report the negative and stop —
> no moving goalposts. Thermal rules from [[h7_embodiment_state]] apply (ikaros bs=2/ctx=64,
> per-step guard, watchdog).

**Phase 0 — De-risk & instrument (1–2 days).**
- Build a clean telemetry harness that timestamps token-emission events against power/thermal/clock
  reads at max safe rate. Measure the raw **output→telemetry transfer function** (does generating a
  high-entropy burst measurably move power/clock vs idle? lag? gain?).
- *Kill*: if token generation produces **no** statistically reliable telemetry deflection above
  governor noise on either box, Direction A (L4) is physically dead here → pivot effort to B only.

**Phase 1 — Direction A core (the L4 loop), ~1 week.**
- A1. Interventional causality: structured generation patterns → measured telemetry response,
  with proper controls (matched compute, governor pinned) and a real causal estimand (not xcorr).
- A2. Forward model: train a small predictor of next-telemetry from (recent tokens, recent
  telemetry); show it beats a no-token baseline (reafference is learnable ⇒ the contingency exists).
- A3. Use-in-control + kill-shot: feed the *prediction error* (surprise) back into generation;
  ablate the loop; show a load-bearing behavioral effect. Pre-register the metric & threshold.
- *Kill*: forward model no better than baseline (A2), OR ablation effect within noise (A3).

**Phase 2 — Direction B core (rigorous binding), ~1 week, parallelizable.**
- B1. Challenge-response: TPM CR + DRAM/RowHammer PUF identity + RDSEED/jitter freshness; bind
  TPM to EK/DevID cert chain. Condition the gate on the *response*, not raw telemetry.
- B2. Attack battery (the actual contribution): replay, learned spoofer, fine-tune-to-recover,
  adapter-discard, ArrowMatch-style recovery — each with a quantified pass/fail and cost.
- B3. Uniqueness honesty: per-core pseudo-replicates + every box we can get; report distance
  *distribution* + FAR/FRR if n permits, or state plainly "pilot, n insufficient for FAR."
- *Kill*: if any single attack trivially recovers capability AND can't be mitigated, that attack
  defines the honest ceiling — report it as the finding.

**Phase 3 — Optional confidential-inference upgrade (only if we want a *true* binding claim).**
- Rent one Intel TDX (GCP C3) or AMD SEV-SNP VM; run the adapter+backbone inside; demonstrate
  attested key-release. This is the only path to "confidential inference bound to silicon" and it
  is **not our local AMD** — decide explicitly whether it's worth it.

**Phase 4 — Write-up & (re)publication.** Only after A and/or B clear their gates. Reframed
honestly: substrate-coupled LLM + (if A passes) first AE-2 reafferent loop for an LLM-on-host +
(if B passes) hardware-rooted binding with a real attack eval. Position against Clifford 2025,
Butlin 2025, the PUF/TRNG literature. No "consciousness," no "operational embodiment," no
"uncopyable" without the n and the attacks.

---

## 5. What we must STOP saying (until earned)
- "operational embodiment" / "embodiment" — until L4 AE-2 is demonstrated (Phase 1 pass).
- "unique / uncopyable (because of the physics)" — the physics gives weak n=2 separation; the
  uncopyability is the TPM's, and only with EK-cert binding.
- "fresh" as if telemetry freshness ≈ security — it's electrical noise (anti-replay of the *signal*
  only); the real anti-replay is the TPM nonce.
- "computation in the body / räkna" on commodity AMD — honest ceiling is entropy/identity.
- any binding claim stronger than **economic deterrence** without a TEE + attack battery.

## 6. Questions for the oracles (adversarial)
1. Is the L1→L4 (Butlin AE-2) reframing the *right* scientific target, or is the output→telemetry
   loop so weak/contaminated (governor, DVFS, thermal inertia) that a "passing" AE-2 result would
   be an artifact? What confound would kill it?
2. Is Direction B worth doing at all given Clifford 2025 + the no-TEE reality, or is the only
   intellectually honest move to drop the security framing and go pure-science (A)?
3. Are we *still* overclaiming anywhere in §3–4? Where would a hostile reviewer laugh?
4. Is there a *third* direction the five reports imply that we missed?
5. For Direction A: the strongest possible kill-shot/ablation design so a positive result is
   credible, not constructed-to-pass.

---

## 7. Oracle verdicts (OpenAI gpt-5, Gemini 2.5 Pro, Grok-4, DeepSeek-reasoner) — 2026-06-16

**Unanimous on priority**: run **Phase 0 of Direction A first**, done correctly (governor pinned),
because it kills-or-greenlights our most ambitious claim in <48h for near-zero cost. Do nothing
else until it passes.

**Direction A — real science but "one hair from a governor/DVFS artifact" (all four).**
- The killer confound (all four agree): the forward model learns *that compute happened* (load →
  power → DVFS/thermal), **not** *which tokens* — a "loaded heater"/thermostat, not reafference.
  Butlin AE-2 needs the **content** of the output to systematically affect the input.
- The decisive controls they demand, in order:
  1. **Yoked compute control** (Grok/DeepSeek): a non-LLM workload that reproduces the exact
     power/thermal/clock trace. If the forward model still "predicts," it's autocorrelation, not
     causation.
  2. **Token-blind vs token-aware** (OpenAI): predictor with vs without the token/efference signal;
     token-aware must beat token-blind on held-out data (pre-registered ΔR²/ΔNLL + permutation test).
  3. **Compute-intensity vs content** (DeepSeek): replace tokens with a scalar FLOP/token-count;
     full-token model must beat it, else zero content-specific reafference.
  4. **Double-dissociation** (OpenAI/Gemini): self-caused vs externally-caused (adjacent-core
     "interferer" heater) telemetry must be *distinguished* by the model, and an **ablation**
     (zero/ misalign the efference copy; lock P-state so telemetry is flat) must break control.
  5. **Hard environment**: pin core, disable SMT/turbo, lock governor/P-state, cap TDP below
     throttle, no stdout/disk, RT telemetry thread, TSC-aligned multi-rate sampling.
- Reframe the claim (all four): NOT "first honest embodiment" → **"a minimal, pre-registered
  experimental test of the Butlin AE-2 criterion in an LLM, with ablations."**

**Direction B — drop or radically re-scope (3 of 4 say drop).**
- Fatal flaw all name: **adapter-discard** — the GPT-2 backbone is public, so an attacker deletes
  our adapter and runs the model for free. The "lock" is on an optional add-on, not the model.
  Plus telemetry is software-spoofable (Plundervolt) and there's no TEE on any box.
- Only honest survival (Gemini): reframe as an **empirical attack study** — *"On the futility of
  software-readable telemetry for model binding"* — where the **learned spoofer** (LSTM/GAN that
  emulates telemetry from tokens; measure queries-to-break) is the novel contribution, all framed
  as quantified **attacker cost**. Otherwise it's "Clifford with worse n." DeepSeek/Grok: just drop it.

**Still-overclaiming (reviewer bait), all flagged:**
- "first honest embodiment" → tone to operational AE-2 test (above).
- **per-core pseudo-replicates** → "a hardware reviewer will laugh"; cores share power plane + L3 +
  memory controller, deeply thermally coupled — NOT independent samples, no FAR/FRR from them.
- **RowHammer PUF** → not drop-in; TRR/on-die ECC may kill it on our DIMMs; time-box, expect a
  possible negative, report it.
- "economic deterrence" → meaningless on a public low-value model (GPT-2) with free adapter-discard;
  needs a concrete cost model vs just renting an H100-CC instance.
- "telemetry freshness ≈ security" → only the TPM nonce is cryptographic freshness; say so.

**Third directions proposed (beyond A/B):**
- **C-energy** (OpenAI + DeepSeek): *energy/thermal-aware self-control* — closed-loop pacing +
  sampling to cut Joules/token under thermal/latency SLA. Honest systems paper, reuses the same
  Phase-0 instrumentation, ≥15–25% J/token target. **Lowest-risk real contribution.**
- **C-adaptive** (Gemini): *substrate-aware generation* — telemetry as a live conditioning context
  (terse when hot/low-power, verbose when cool) → resource-aware LLM behavior.
- **C-sidechannel** (Grok): *power side-channel as a monitor of the computation* — mutual
  information between package power/voltage and token entropy / CoT-vs-direct. No new HW, no
  embodiment claims; relevant to extraction/watermarking. **(This is the closest to Eric's
  "meta-computation" question — see §8.)**
- **C-cPUF** (DeepSeek): *computation-based PUF from DVFS transients* — run a secret computation
  *challenge*, measure the µs-scale per-core voltage/clock *response* shaped by per-die process
  variation. NOT covered by Clifford (static fingerprint); an honest, novel identity primitive.

## 8. Eric's question: can the ~30 signals do "computation on top of the computation" (meta-computation)?

Honest, grounded answer — separating the real version from the trap (the trap is exactly my "bias
to want to succeed", so I am being deliberately strict):

**The 30+ channels (16 per-core Vcore + 32 per-core clock + package power + …) are 30 SENSORS on
ONE computational substrate, not 30 computers.** They are largely *correlated* readouts of the same
event (the GPU/CPU doing matmuls → load → V/f/power move together), sampled slowly (~10–100 Hz) and
dominated by the DVFS governor. So:

- **"Meta-computation = the 30 signals add raw computational power / act as a reservoir that computes
  what the software can't" → NO (the trap).** This is the physical-reservoir / morphological-
  computation L3 claim. The honest test (§1E): the physics computes only if a *fixed/linear* readout
  solves a task it couldn't from the raw input, and removing the physics collapses it. On commodity
  AMD the signals are coupled, low-bandwidth and governor-shaped — they lack the fast, decorrelated
  nonlinear dynamics a reservoir needs; there is **no published success**, and our **own** integrated
  pipeline was "ALL negative for MNIST" (per [[h7_embodiment_state]]/older work). More sensors ≠ more
  compute. Reading 30 instead of 2 gives a richer *feature vector*, but the software adapter still
  does all the computing.

- **"Meta-computation = a SECOND computation that operates ON the primary computation, read out from
  its physical shadow" → YES, and this is the genuinely novel honest core.** The 30 signals are a
  *physical observable of the LLM's own computation*. A model that reads them is doing **introspection
  / monitoring** — computing *about* the primary computation, not adding raw FLOPs. This is precisely
  where three threads converge:
  - Grok's **side-channel monitor** (power trace → token entropy, CoT-vs-direct) — meta-readout of
    the computation's state from its physical shadow.
  - The **AE-2 forward model** (predict telemetry from tokens) — a meta-model of the computation's
    own physical effect; the 30-D signal is the substrate of the reafference.
  - DeepSeek's **cPUF** (run a challenge computation, read the 30-D physical response) — computation
    used to probe the substrate's individuality.
  Here the **breadth of 30 channels is a legitimate asset**: a 30-D physical response carries more
  per-die / per-state information than 1-D power, which directly helps identity (cPUF) and
  introspection (side-channel), even though it adds no raw compute.

- **The concrete, cheap experiment that tests Eric's intuition directly (fold into Phase 0):**
  does the **full 30-D telemetry vector** let a forward/decoder model predict things (token entropy,
  reafference, self-vs-external perturbation) that a **1-D power signal cannot**? Quantify the
  information gain of the 30-D set over the 1-D baseline (ΔNLL / mutual information). If 30-D ≫ 1-D,
  the multi-signal richness is real and we build the meta-layer (introspection / cPUF) on it. If
  30-D ≈ 1-D, the channels are redundant and "30 signals" was never the point — say so plainly.

**One-line answer**: the 30 signals can't *add* computation (they're correlated sensors on one
substrate — the reservoir dream is a dead end on commodity AMD), but they can carry a real
**meta-computation that introspects on the LLM's own computation** (side-channel + reafference +
challenge-response identity), and there the 30-channel breadth genuinely helps. That introspection
layer — not "the body computes" — is the honest, novel target, and it folds straight into Phase 0.

## 9. Decision (post-oracle)
1. **Run Phase 0 first** (Direction A instrumentation), hardened to the oracle spec (pinned
   core/governor, yoked-compute control, token-blind vs token-aware, **30-D vs 1-D information
   test from §8**). This single experiment decides A *and* answers Eric's meta-computation question.
2. **Carry C-energy + C-sidechannel as the fallback** if A's signal is too weak — both reuse the
   same harness and are honest standalone contributions.
3. **Park Direction B** unless a real challenge-response (cPUF / TPM-EK) proves viable; never ship
   another adapter-discardable fingerprint lock.
4. Strip all flagged overclaims from any future write-up/post.
