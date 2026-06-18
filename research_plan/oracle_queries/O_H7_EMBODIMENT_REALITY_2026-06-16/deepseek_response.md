# deepseek response (deepseek-reasoner) — 113s

We read PLAN_AND_SOTA.md. The self-awareness is refreshing—your internal audit is mostly correct. But the two proposed directions still contain major blind spots. Below we answer each question directly, with mechanisms and controls. **We are not polite.**

---

## 1. Direction A (Butlin AE-2 closed loop) — confound and kill-shot

### Confound that would fool you

The **compute-intensity artifact**:
- The forward model (predict next-telemetry from recent tokens) will succeed trivially on any physical system where *any* compute load raises temperature and triggers DVFS. You train the predictor on token sequences; it learns to associate bursts of token generation with rising temperature. But the information it uses is not *which tokens* were generated, but *that* compute happened.  
- **Why this fools an AE-2 claim**: Butlin requires modeling *systematic effects of output on input* — the semantic content of the output matters, not just that there was output. If the forward model works equally well when you replace the token sequence with a scalar "compute intensity" (e.g., total FLOPs or token count per window), you have *zero* evidence of reafference. You have a thermostat that knows when it's working hard.
- **On AMD Strix Halo**: the telemetry (Vcore, clock) is sampled at low resolution (~10–100 Hz via `zenpower` or `msr`). DVFS time constants are milliseconds; thermal inertia is seconds. Token generation is bursty at ~10–20 tok/s. The per-token *compute variation* across tokens in GPT-2 is tiny (all token embeddings go through same layers; only final logit differs). The dominant signal is average power over a sliding window of ~10 tokens. Content-dependent compute differences are likely below quantization noise (especially since Vcore is set by voltage regulator, not by instruction-level power draw). The forward model will see essentially the same telemetry trace for any token sequence of equal length.

### Control that kills it

**Ablation A: replace token conditioning with compute-intensity conditioning.**

- Train a regressor to predict telemetry from (a) full token sequence, (b) a scalar `compute_intensity(t)` = number of tokens × constant FLOPs/token (ignore content).  
- **Pass criterion**: forward model (a) must significantly beat (b) on held-out telemetry (e.g., explained variance improvement > 0.05).  
- If not, the loop has no content-specific reafference — it's just load thermals.

### Strongest kill-shot ablation for the use-in-control step (A3)

- Instead of feeding *prediction error* (surprise) back into generation, feed **random Gaussian noise with identical auto-correlation structure** to the prediction error.  
- Pre-register: the behavioral effect (e.g., perplexity change, output diversity) must be significantly different from the random-noise condition.  
- If random noise produces the same effect, the loop is not using modeled reafference; it's just adding noise that happens to perturb generation in a trivial way (e.g., drift in logit scale).  
- **Even better**: hold the *amount* of modulation constant (e.g., add a fixed perturbation to the steering adapter's input) and compare conditions where the perturbation is derived from actual telemetry vs. from a frozen random seed. If the telemetry-conditioned perturbation shows different behavioral outcomes, then the content of the telemetry matters.

**Bottom line**: Without token-content-specific telemetry dependence, you have a *loaded heater*, not an embodied agent. The plan's Phase 0 should explicitly measure per-token telemetry variance and compare to noise floor. If the per-token signal is below noise, kill A immediately.

---

## 2. Direction B (security binding) — not worth it as is

**Clifford et al. (SaTML 2025) already covers fingerprint-keyed model locking.** Your construction is a *weaker* version:  
- Clifford transforms *weights* so that without the correct key the model is destroyed (indistinguishability prevents brute-force screening).  
- You condition a *steering adapter* on live telemetry. The backbone GPT-2 is public and fully usable without the adapter. An adversary can discard the adapter entirely and run GPT-2 with zero degradation. That is a **fundamental failure**: the binding is not on the model, it's on an optional add-on.  
- The TPM nonce prevents replay of an old telemetry trace, but the telemetry itself is still readable and spoofable (Plundervolt). The attacker can read the current telemetry values from their own machine and set them via software.  
- Even with DRAM/RowHammer PUF: you have no TEE, so the PUF response is read in plaintext by the CPU and passed to the adapter. A root user can intercept or fake it.

**The only honest move**: drop the security framing entirely.  
**If you insist on B**, the absolute minimum contribution would be:  
- A thorough **cost-bypass analysis** for each attack:  
  - Adapter discard: cost = 0 (GPT-2 weights are public).  
  - Fine-tune to recover performance: cost = training a small adapter on a new machine (likely < $5 of compute).  
  - Telemetry replay/spoof: cost = software write to MSR (Plundervolt is free).  
  - TPM capture: cost = $40 FPGA for dTPM; for fTPM, PSP firmware extraction is likely more expensive but feasible for a motivated attacker.  
- Then claim only "economic deterrence against script kiddies" — but even that is weak because the cost is so low. Clifford's approach at least forces an attacker to either extract the key (hard) or brute-force (infeasible for large key space). Yours offers no such barrier.  

**Verdict**: B is a worse re-run of Clifford. Don't waste time. If you want a security result, go get a cloud VM with SEV-SNP or H100 CC and run attested inference. That would be a real (but not novel) demonstration.

---

## 3. Where you are STILL overclaiming (§3–4) — the hostile reviewer's laugh

1. **"first honest embodiment result for an LLM on its own host"** (§3 Direction A)  
   - Laughable because Butlin AE-2 was designed for agents that model output-to-environment contingencies, not output-to-own-DVFS. The "environment" is the CPU die itself—this is closer to a thermostat than a sensorimotor agent.  
   - You skip L2–L3 (morphology, morphological computation) and claim L4 based on a trivial loop that any toaster with a PID controller could satisfy.  
   - The phrase "first honest" implies prior work was dishonest; unfounded.

2. **"converts 'shallow' into a citable AE-2 result"**  
   - A hostile reviewer: "You have not shown that the model *uses* the model of reafference in perception or control. You only propose to feed prediction error back into generation. That is a hand-crafted control loop, not the system autonomously learning to use the model. Butlin's AE-2 requires the model to be used in perception/control, not just a hacky input augmentation."

3. **"properly-evaluated hardware-rooted binding"** (Direction B)  
   - The binding is not hardware-rooted because the adapter is software, the backbone is unmodified, and the TPM is the only hardware root (and even that is fTPM/firmware). You cannot claim "binding to this chip" because the chip's computation is not tied to the key—only the telemetry reading is.  
   - "properly-evaluated" implies a thorough security model; you haven't defined the adversary model (what can the attacker do? root on own machine? remote? physical access?). The plan omits this.

4. **"economic deterrence"** — you use this as a safe harbor, but economic deterrence requires demonstrating that bypassing the binding costs > value of the model. With GPT-2 (public, low-value), the deterrence is zero. Even if you scaled to a proprietary model, the adapter-discard attack gives free use.

5. **Figure of merit**: "null gate passes both dies" — you claim this as rigor, but it actually shows the gate is not discriminating (if null passes both, then the adapter can be bypassed by setting the gating signal to a precomputed 'null' vector). That weakens the security claim further.

---

## 4. Third direction: *Computation-based PUF from DVFS transients*

The SOTA (Section 1B) shows a large body of work on fingerprinting via clock skew, hot pixels, SRAM PUF—all static. A genuine gap: **workload-dependent voltage/frequency response as a challenge-response PUF**.  
- Concept: Send a carefully engineered computation pattern (e.g., a sequence of matrix multiply shapes that stress different parts of the die) and measure the resulting core voltage and clock frequency transients at microsecond resolution (using MSR reads or `pmc` counters). The response is a high-dimensional time series shaped by per-chip process variation, IR drop, thermal coupling.  
- Uniqueness: depends on physical layout and dopant variation — likely as unique as SRAM PUF but resistant to environmental modeling because the computation pattern is secret (challenge) and the response is fast-changing.  
- Amenable to your hardware: `msr` and `zenpower` provide voltage readings; you can control CPU frequency scaling via `cpufreq` or `amd-pstate`. On Strix Halo you have unified memory – could even include GPU workload.  
- No TEE needed: the PUF response is used to seed a key inside the TPM (or just as a biometric). Security model: even if an attacker captures a challenge-response pair, replaying it on another die produces different response.  
- This would be a **contribution** to the physical-layer security community (non-trusted-device PUF). It is not covered by Clifford (who uses static fingerprint). It is honest: you claim a PUF with measurement of uniqueness, stability, and attack resistance.  
- **Challenge**: you need high-resolution timing (maybe use hardware counters or HPET) and careful characterization over temperature/voltage/aging. But you have two dies; you can show cross-die distance distribution (pilot).  

**Why you missed this**: your plan conflates "computation in the body" (physics does ML) with "computation as identity". The latter is a legitimate and reachable target. B tries to do identity, but with weak telemetry; a properly designed challenge-response with controlled workload would be stronger and novel.

---

## 5. Priority call: one direction for the next 2 weeks

**Direction A, but only after Phase 0 (transfer function measurement).**  

**Single highest-value experiment**:  
1. **Phase 0** (your proposed 1–2 days): Measure the transfer function from token generation to telemetry with high temporal resolution. On both AMD boxes.  
   - Use a burst of tokens of varying length (64, 128, 256) vs. idle.  
   - Record voltage, clock, power (if available) at 1 kHz (use `linux-perf` or custom kernel module if needed; `rdmsr` via a tight loop).  
   - Compute the per-token effect size: does generating 100 tokens shift Vcore by > 5 mV or clock by > 10 MHz? What is the rise time and fall time?  
   - Compare to the variance when idle (including governor noise).  

2. **Kill criterion**: if the effect size is < 2× the idle noise (σ), or the signal is purely a slow ramp (time constant > 1 s) and shows *no* correlation with token identity beyond total count, **kill A**.  

3. **If A passes Phase 0** (measurable, content-specific potential): spend the remaining time on A1 (interventional causality with structured patterns) and A2 (forward model with compute-intensity control). Do not even attempt A3 unless A1 and A2 show token-specific effects.  

4. **If A fails Phase 0**: pivot to the **third direction** (computation-based PUF). Design and run a challenge-response protocol using DVFS transients. This is new, honest, and doable in 2 weeks (you already have telemetry harness). It also lets you reuse the "n=2" analysis but with richer data.

**Why not B**: B is a dead end. The only reason to do B is if you desperately need a "security" paper; but any reviewer will ask "where is the TEE?" and "how do I discard the adapter?". The answer is embarrassing. Invest in A (if feasible) or the PUF direction.

**Bottom line**: Phase 0 today. If signal is real, go A (but with compute-intensity control). If not, go third direction. Drop B entirely.
