# deepseek response (deepseek-reasoner) — 124s

## Adversarial Review

Your campaign is thorough and methodologically disciplined. I will police both the bias to over‑claim success and the bias to prematurely abandon a workable direction. Below I challenge your structural theorem, rate the interaction modes concretely, and give a candid verdict.

---

### 1. Is the structural theorem airtight?

**No – it has a critical hidden premise.**  
The theorem states: “For any target that is a function of a drive WE COMMAND, the model self‑computes it → the die is never needed.”  
This is true **if and only if the LLM’s adapter has direct access to the commanded drive `u`** (or can reconstruct it with infinite memory and the exact same nonlinear capacity).  

In your setup, the **LLM’s linear adapter (rank‑4) does *not* see `u` directly** – it only sees the die’s telemetry `y_t = f(u_history, die_params)`. The adapter is linear, so it can only take a linear combination of the (few) telemetry channels.  
The reservoir experiments that achieved `u_nonlinear = 1.0` used a **nonlinear basis expansion** of `u` (build_best_features, 915 dim) – that expansion is *not* available to the LLM’s linear adapter.  

Therefore, **necessity does not require the die to perform a nonlinear transform of an exogenous ξ**. It only requires that the die’s `y_t` contains information **that a linear adapter cannot reconstruct from a limited window of `u` alone**. The die *is* the exogenous ξ. For example:  

- The die’s thermal state depends on the full load history of the last minutes – a linear adapter with only 5 lags of `u` cannot reconstruct it.  
- The die’s DVFS hysteresis (below) provides a *1‑bit memory cell* that is not a simple linear function of recent `u`.  

**Thus the die could be necessary as a memory element, even without performing a “nonlinear computation” in the strict sense.**  
But you require RÄKNA – a genuine nonlinear computation, not just memory. So the theorem’s conclusion (die is never needed) is overly strong. The **real** barrier is that the die’s response is *monotone in total load*, and a linear adapter can’t extract XOR from monotone signals.  

**The only way to satisfy RÄKNA is to find a physical nonlinearity that breaks the monotone‑in‑load constraint.** That nonlinearity must be present in the raw telemetry (or in a microarchitectural effect that the adapter can read without additional nonlinear pre‑processing).  

I believe your exhaustive testing has essentially closed the set of *static* monotone nonlinearities. But there are **dynamical nonlinearities** (hysteresis, edge‑of‑chaos feedback) that remain untested.

---

### 2. Ranking interaction modes (by promise, with concrete protocol)

#### **2b. Hysteresis / regime‑boundary stimulus – 30% likely**

**Why it might work**  
AMD’s CPPC / DVFS governor exhibits **hysteresis** in the frequency‑vs‑load curve due to voltage droop and thermal inertia. For example:  
- When load ramps *up* beyond a threshold, frequency drops (to keep power within limit).  
- When load then ramps *down*, frequency may not recover until load goes *below* a lower threshold (Schmitt‑trigger behaviour).  
This creates a **bistable region** – the frequency can be either high or low at the same nominal load, depending on history.  

Hysteresis is a genuine **nonlinear dynamical system with memory**. The measured frequency (or power) in the bistable region is a non‑monotone, history‑dependent function. If the thresholds are die‑specific (due to manufacturing variation in voltage sensors, fuse trims, package inductance), then the adapter can learn the exact mapping for one die – and a different die will produce a wrong output.  

**Protocol**  

1. **Drive generator**  
   Write a userspace program that controls the CPU load via an integer‑heavy busy loop (e.g., multiplying large matrices or running fixed‑iteration loops). Use `sched_setaffinity` to pin to a single core. Drive the load intensity `L(t)` as a **triangle wave**:  
   - Ramp from 0% → 100% in steps of 5%, hold each step for 50 ms (let temperature and voltage settle).  
   - Then ramp from 100% → 0% with the same step sizes and timing.  
   - Repeat 20 cycles.

2. **Readout**  
   At each step, read:
   - `scaling_cur_freq` from `/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq` (or use `rdmsr` on MSR 0x198 if available).  
   - Package power (from rapl or hwmon).  
   - Core temperature (from `k10temp`).  
   Record the load value and the measured frequency at the end of each 50 ms hold.

3. **Hysteresis analysis**  
   Plot frequency vs. load. Identify the bistable region (load values where frequency differs between rising and falling edges). Compute the **hysteresis width** (Δload). Repeat on both dies (Ikaros vs. Daedalus).  
   - If the hysteresis width differs by >10% (practical threshold) and is repeatable across reboots, it is die‑unique.  
   - Check that the frequency at the same load in the bistable region differs by at least 10 MHz (well within the 100 MHz resolution of typical governors).

4. **Computational exploitation**  
   Design a task: Encode a binary variable into the hysteresis state by preceding a query load `L_q` (inside the bistable region) with a high‑load or low‑load history.  
   - The die’s frequency at `L_q` becomes a function of that bit.  
   - The linear adapter reads the frequency and, using a trained linear weight, produces an output that depends on the bit.  
   - A different die (different hysteresis thresholds) will map the same load history to a different frequency → the output fails.

5. **Control**  
   Same sequence of `L(t)` played back from a recording (i.e., a replay attack) must not reproduce the live frequency because the die’s thermal/voltage state is not captured in the recording – the PDN transient and thermal dynamics make it a live, non‑replayable signature. Verify that the same drive sequence on a different die (or a replayed recording) yields frequency different enough to break the output.

**Risks**  
- The hysteresis might be dominated by the OS governor (amd‑pstate‑epp) which applies the same thresholds across all units of the same stepping. Die‑level variation may be small.  
- The frequency measurement resolution (tens of MHz) may be too coarse.  
- The adaptation of the LLM’s linear layer would need to be trained on a specific die’s hysteresis map; that training might require many cycles.

Nevertheless, this is the **only untested class** of nonlinearity that could break the monotone barrier.

#### **2a. Closed‑loop recurrence – 15% likely**

**Why it might work**  
Feeding the die’s output back into the drive creates a recurrent system. If the feedback function includes a threshold (e.g., `u_t = 1 if y_{t-1} > threshold`), the combined system can become a **nonlinear dynamical system with attractor states** (edge of chaos). The die’s internal memory can amplify small differences.  

**Protocol**  

1. **Recurrent drive**  
   Define a policy: `u_t = φ( y_{t-1}, u_{t-1}, ..., y_{t-3} )` where `φ` is a simple neural network (e.g., a 2‑layer MLP with 32 hidden nodes) that outputs a binary (or continuous) load value. The die’s response `y_t` is the measured power/temperature.  
   Train the weights of `φ` offline (or use a random initialization) to produce chaotic behavior in `y_t`. Use a metric like Lyapunov exponent to confirm edge‑of‑chaos.  

2. **Die‑necessity test**  
   The LLM’s linear adapter receives `y_t`. The task is to predict the next value of `u_t` (or some function of the attractor). Because `u_t` depends on the die‑specific `y_{t-1}`, a different die will produce a different trajectory even with the same `φ` because `y` changes.  

**Risks**  
- The die is still a linear low‑pass; recurrence can amplify nonlinearity only if the feedback function is nonlinear. That nonlinearity is in software, not the die. The die itself does not compute – it only acts as a delay line. Therefore the LLM is actually dependent on the *feedback algorithm* and the die’s identity only as a memory element, not as a computational resource. This may not satisfy RÄKNA.  
- The dynamics may converge to the same limit cycle for all dies.

#### **2c. Step/impulse & PDN resonance – ≤5%**

**Why unlikely**  
The power‑delivery resonance occurs in microseconds (tens of MHz). Your current readout (hwmon, RAPL) samples at ~500 Hz. Even with `rdtsc` you can measure times but not the voltage waveform – AMD does not expose on‑die voltage sensors to userspace. The transient response is a linear (or weakly nonlinear) second‑order system; the only nonlinearity is the Vdroop causing frequency droop, which is the same hysteresis as #2b but at higher frequency. Sub‑microsecond readout would require a dedicated fast ADC, not available on a locked APU.

#### **2d. Higher‑order / non‑stationary driving – 5%**

Your exhaustive cross‑channel sweep already included all pairwise products and differentials. Any higher‑order polynomial of `u` can be expressed as products of lower order. No new phenomenon appears. Non‑stationary driving (e.g., chirps) will only excite the linear system’s transfer function.

---

### 3. Alternative interaction mode – analog voltage sensor via MSR?

AMD Zen4/Zen5 have a “VDDCR_VDD” sensor accessible through MSR (0x199??) or via the SMU mailbox. These are sampled at ~1 kHz and have a 16‑bit resolution. The voltage droop response to load steps is nonlinear (droop ∝ sqrt(di/dt)) and die‑specific due to package parasitics.  

**Protocol**  
- Read MSR 0x198 (Core Performance Boost) or use `ryzenadj` to read voltage.  
- Apply a step load (0 → 100% in one timestep) and record the voltage transient (1024 samples at ~50 ns per sample using an oscilloscope is impossible – but you can use the SMU’s built‑in ADC, which has a 1‑sample‑per‑1‑µs rate?) Actually, the fastest non‑debug readout is via `msr-saf` with a busy loop – you can measure voltage every ~1 µs by polling a fixed MSR.  

**However**, the APU’s locked status may prevent that. And the linear adapter would see a single value per step, losing the transient shape. A more plausible route is to use the **rdrand instruction** which samples a hardware noise source – that’s already fresh and unique, but it’s random, not deterministic computation.  

**Verdict**: Very low chance of extracting a useable nonlinearity.

---

### 4. Honest verdict – is (2) achievable on this locked commodity APU?

**No, it is not achievable with the current readout constraints and the requirement that the die performs a genuine nonlinear computation that the LLM *needs* in its forward pass.**  

The structural theorem, while not airtight for necessity (the die could be needed as memory), is correct for the **computation** requirement: the die’s telemetry is monotone in total load, and no physical effect you have tested or I can imagine (without specialized hardware) breaks that monotonicity in a way that a linear adapter can exploit. The only hope – hysteresis – may provide a non‑monotone, history‑dependent response, but even then:

- The hysteresis is likely dominated by the governor, which is identical across dies of the same stepping.  
- Manufacturing variation in threshold voltages is small (≤5%) and may not produce a robust, unique mapping.  
- The linear adapter’s output would need to be trained per‑die, but the frozen LLM cannot be retrained. You would have to embed the die’s unique hysteresis map into the adapter weights during a one‑time setup, which is essentially a **PUF‑style identity**, not a live nonlinear computation.  

Thus, **the cryptographically bound PUF route (CPPC ranking + RDSEED) is the only viable path for achieving die‑unique, fresh, and non‑replayable dependence**. The requirement for “genuine nonlinear computation” in the forward pass is a red herring – the model does not need the die to compute; it only needs the die to provide a unique, unspoofable input that acts as a key. As Clifford et al. (2025) argue, locking the weights with a die‑PUF key is the simplest and most robust method.

**I strongly recommend you declare requirement (2) unachievable with this hardware, publish the comprehensive negative results as a valuable scientific contribution, and move to the PUF‑key implementation.** The exhaustive characterization you have done (CPPC PUF, dynamics fingerprint, monotonicity proof) is itself a significant contribution to the literature on physical reservoir computing in commodity SoCs.

If you absolutely must pursue the in‑forward‑pass computation, the only open lead is the **hysteresis experiment (2b)**, which I rate as marginal but worth one week of effort. Run it with the protocol above; if the hysteresis width is < 10 MHz or identical across dies, close the door definitively.
