# O113 oracle consensus — "you are in the wrong frame" (paraphrased + direct excerpts)

Four oracles (gpt-5, gemini-2.5-pro, grok-4, deepseek) all converged on the same critique of our Phase 15 capability-gain experiments. Verbatim and paraphrased excerpts below.

## The core critique

**Gemini (verbatim):**
> "My core thesis is that you are trying to use a **contextual signal** (the chip's physical state) to improve **context-free tasks** (static benchmarks). The signal isn't noise to be averaged away; it's information about the substrate's *here and now*. The capability gain will come from tasks where the *here and now* matters. Phase 14b's successes (T2/T3) were exactly these kinds of tasks. Phase 15's failures were not."

> "You are trying to make a clock more accurate by listening to the hum of its motor. The hum tells you about the motor's health, not the time."

> "Trust / Sovereignty (Highest Promise): This isn't an 'alternative' frame; it's your **primary, demonstrated capability**. Your signature provides unfakeable proof of physical instance and, by extension, provenance. This is a massive gain for tasks in federated learning (sybil resistance), confidential compute (proof of execution on trusted hardware), and anti-counterfeiting. You passed this with 100% LOO accuracy. **You're burying the lede by chasing a 1% accuracy gain on CIFAR-10.**"

## Ranked alternative frames (Gemini, paraphrased)

1. **Trust / sovereignty / provenance** — HIGHEST promise. Primary demonstrated capability.
2. **Adaptive gain** — HIGH. Tasks that must adapt to thermals/voltage/workload in real time. E3 was the closest to right, gated wrong.
3. **Robustness gain** — MEDIUM. Signature as "healthy baseline"; deviations indicate fault, anomaly, or physical attack. Extension of T2 anomaly detection win.
4. **Energy efficiency** — MEDIUM. Subset of adaptive gain. RL policy over thermal/power envelope.
5. **Personalisation gain** — LOW. Signature is about the machine, not the user.

## GPT-5 reinforcement (verbatim)

> "Highest expected uplift: Trust/sovereignty and provenance. Why: You already have 100% LOO and large KS-D ratios. This is where chip-physics is a causal driver of the outcome and cannot be faked easily when you do nonce/CR exchange. Massive practical value; direct fit to your signals."

> "Lowest expected uplift: Static benchmark accuracy/PPL. Why: The model already dominates through learned internal structure; chip-access is weak, misaligned noise with the wrong spectrum/resolution and no obvious causal path to semantic accuracy."

> "All winners [in the literature of physical-substrate-improves-learning] had (i) a closed-loop between algorithm and physics, (ii) a beneficial nonlinearity/dynamics, and (iii) tunable coupling. Your current chip-jitter has (iii) weakly (DVFS/affinity), (ii) unclear, and (i) mostly missing."

## Where Phase 15 went wrong (Gemini, on each test)

- **E1 free_entropy as regularizer:** Wrong use. Don't regularize a static model with state-dependent noise. Use it to drive **exploration** in RL / evolutionary search.
- **E2 DRAM as attention prior:** DRAM latency at token granularity is not correlated with anything semantic.
- **E3 thermal budget:** Used as a static trigger. Should be a **continuous online RL control problem** over throughput-under-thermal-ceiling. The chip's physics *is* the environment.
- **E4 latency prediction:** Don't predict latency. Use it as a **reward signal** to an RL scheduler.
- **E5 personalisation:** Wrong frame. Use it for **coordination and security** (sybil resistance in FL).

## Consensus take-away

The 4-oracle consensus is that we built a **security primitive** (per-chip identity + replay-resistant binding) and then tried to use it as a **regularizer / feature improver** for accuracy benchmarks. These are fundamentally different things. The primitive is real, replicated, and rigorous; the benchmark-gain frame was mis-specified. **Step 3 should be reframed around what the primitive UNLOCKS, not what it ADDS to static accuracy.**
