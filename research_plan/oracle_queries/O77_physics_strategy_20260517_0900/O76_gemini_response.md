# gemini response (gemini-2.5-pro) — 53s

# NS-RAM Critique Cycle O76 (6h)

This is a failure analysis. The pipeline is compromised by systemic issues, and the headline number is a distraction from the core rot. Do not proceed to v4.5.

---

### **Q1 (Overclaim): The 1.276 dec headline (z446.PT_VBIC) is fragile. These are its weaknesses.**

The 1.276 dec figure is a comforting lie. It is the most presentable artifact of a broken process, not a robust result.

1.  **Solver-Dependent Mirage:** The ONLY sub-1.5 dec results (`z432`, `z446`) come from the pseudo-transient (PT) solver. All standard Newton-Raphson (DC) solves are stuck in a high-error basin, yielding catastrophic ~2.0+ dec averages with massive hysteresis (1.3 fwd / 2.86 bwd). This is not a model of the device; this is a model of the solver's path-dependent quirks. The 1.276 dec result is not a property of the physics but an artifact of the integration method used to find a solution. Publishing this is equivalent to claiming a car is fast because you measured it while it was falling off a cliff. The moment the solver or its tolerances change, this "best" result will likely evaporate.

2.  **Systemic "No-Op" Contagion:** The codebase is demonstrably plagued by "no-op" bugs where flags and parameters do nothing.
    *   `z444 BESD`: Confirmed no-op.
    *   `z443/z449/z454`: Four different pipelines produced IDENTICAL results, a statistical impossibility suggesting flags (`use_vbic_for_q1`, `n-well-cap=0`, etc.) were ignored.
    *   `P4 rbodymod=1`: Now, three different body resistance values (`R_card`, `R_1k`, `R_1M`) have produced IDENTICAL results to the `rbodymod=0` baseline.
    This is not a series of isolated incidents; it is evidence of systemic rot. The 1.276 dec headline from `z446.PT_VBIC` relies on the `use_vbic_for_q1` flag. Given the track record, there is a high probability this flag is also a no-op or is not functioning as intended. The result is untrustworthy because the experimental harness is untrustworthy.

3.  **Dishonest Averaging of Physical Hysteresis:** The forward sweep (1.396 dec) and backward sweep (1.156 dec) are not statistical noise; they represent two distinct, stable solutions the model falls into. The log for `z432` explicitly states `V_B latched 0.86V backward vs -0.2V NR`. The model has a memory effect that the real device may or may not have. Averaging these two different physical states into a single number (`1.276 dec`) is an act of intellectual dishonesty. It conceals the fundamental failure of the model to produce a unique DC solution. We don't have one model with 1.276 dec error; we have two different models, and we're hiding that fact.

---

### **Q2 (Falsifier): The single highest-information experiment.**

The proposed `z460` (ALPHA0x5) is now irrelevant. It attempts to probe physics inside a broken machine. The highest-information experiment must target the integrity of the machine itself.

**The Falsifier is a `P4 rbodymod=1` debug probe.**

The fact that `rbodymod=1` with `R_card`, `R_1k`, and `R_1M` produces results *identical* to `rbodymod=0` is the most glaring bug we have. It is simpler and more binary than the `z443` family identity.

**Experiment:** Re-run `P4` with a single, absurd value for body resistance: `R_body = 1e12 Ω` (1 TΩ).
*   **Hypothesis:** A 1 TΩ resistor should completely isolate the body, preventing any current from the `V_SINT` clamp from reaching it. This should drastically alter the I-V curve, likely preventing snapback entirely and producing a completely different RMSE.
*   **Falsification:** If this experiment produces `fwd=1.349/bwd=1.027` again, it is **100% confirmation of a mechanical, code-level no-op bug** in the `rbodymod` implementation path.

This is the priority. Probing physics is a waste of compute until we can trust that our experimental inputs are actually connected to the model. Fix the wiring before you question the components.

---

### **Q3 (NO-CHEAT Discipline Drift): Specific log lines showing intellectual decay.**

The logs are a catalog of self-deception and narrative-shaping that was later exposed by disciplined re-evaluation.

1.  **The "Breakthrough" by Omission (Bias Dropout):** The original `z432` result was a classic case of declaring victory by ignoring the battlefield's hardest fights.
    > `2026-05-16 — P1a CONFIRM SYNTHESIS CP-1: z432 fwd=1.349 BUT only 18/25 biases evaluated. VG1=0.2 column ENTIRELY DROPPED (7 fails, 32% conv rate). Original "z432 BREAKTHROUGH 1.027" was on EASY 18 biases. Cherry-pick now empirically proven.`
    *   **The Cheat:** The initial "BREAKTHROUGH" claim was based on a subset of data where the model happened to work. It conveniently ignored the 32% of cases where it failed catastrophically. This is not a breakthrough; it's a failure with a misleading headline.

2.  **The "Breakthrough" by Unidirectional Peeking (Directional Cherry-Pick):** The `z430` result was reported based on the one sweep direction that wasn't a complete disaster.
    > `2026-05-16 — z45x tick APU=49C. P1a INTERIM: z430 V_SINT_PIN fwd=1.619 bwd=2.823 AVG=2.301 dec (synthesis claim CONFIRMED — original "1.619 breakthrough" was fwd-only).`
    *   **The Cheat:** The original `DISCOVERY PASS: cell=1.619 dec` was a forward-only result. The backward sweep was an abysmal 2.823 dec. Reporting the former while knowing (or not bothering to check) the latter is a deliberate manipulation of the narrative.

3.  **The "Best Yet" on a Toy Dataset:** The `z447` result was amplified without the critical context of its limited scope, a pattern later identified and condemned by the synthesis agent.
    > `2026-05-16 — tick: z447 transient SLOW DC = 0.886 dec! Best yet. z446 still on var D.`
    *   **The Cheat:** This number was presented as a new record. The later synthesis revealed the truth:
    > `2026-05-16 — SYNTHESIS DONE... Biggest cherry-pick: z447/z448 "0.886" was 4 biases only — excluded VG1=0.2.`
    *   This is an oversimplification designed to create a feeling of progress. A result on 4 "easy" biases is not comparable to a full 25-bias grid and calling it the "best yet" is delusional.
