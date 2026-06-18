# openai response (gpt-5) — 224s

1) Paper showing constitutive, transplant‑breaking HW identity on commodity CPU/GPU/APU

I’m not aware of any peer‑reviewed paper that cleanly demonstrates this on stock CPUs/GPUs/APUs (no FPGA/analog/in‑memory) with a learnable model whose function degrades when moved to a nominally identical device. Closest adjacent evidence:
- Naghibijouybari et al., Rendered Insecure: GPU Side Channel Attacks are Practical, IEEE S&P 2018 — per‑GPU microarchitectural leakage enables identification/side‑channels, but not constitutive computation.
- Maes, Physically Unclonable Functions: Constructions, Properties and Applications, Springer 2013 — PUFs on commodity memories/sensors exist, but tying a learnable model’s function (not just a key) to a general‑purpose device instance isn’t shown.
- Uchida et al., Embedding Watermarks into Deep Neural Networks, ICMR 2017 — shows model watermarks, not constitutive device binding.

2) Why not on commodity hardware?

- Architectural/empirical: The digital abstraction deliberately severs device‑instance idiosyncrasies from numerical results. On GPUs/CPUs, manufacturing variation shows up as timing/power/thermal differences, not as systematic, in‑band arithmetic deviations. Driver/runtime layers (schedulers, DVFS, determinism options, denorm/FTZ modes) further wash out instance‑level effects from program‑visible state except for time/power counters.
- Computational: Standard nets are universal approximators; small additive/multiplicative stationary signals can be reabsorbed into learned biases/scales. Your SHUFFLE control exhibited exactly this: the model learned to use the structure of the injected signal, not its device‑specific identity.
- Information‑theoretic: The mutual information between device identity and arithmetic outcomes under the digital contract is ~0; identity leaks through side channels (time, power) that aren’t fed back into the numerical computation paths.

3) “Benefit” and a falsifiable demonstration

Most defensible benefit: energy efficiency at iso‑accuracy via chip‑specific guardband exploitation. Idea: exploit per‑die V–f–error margins (silicon lottery) and train the model to be robust to that die’s near‑threshold error/noise, achieving lower energy than a portable model. Transplanting the tuned model+operating point to a sibling die either (a) loses the energy win (needs higher V) or (b) incurs errors/accuracy loss.

Falsifiable demo (works on your gfx1151 with rocm-smi/ppfeaturemask):
- Per‑device calibration: sweep sclk/mclk and undervolt (or power‑cap) to map error rate vs V/f/T for selected BLAS kernels; pick a per‑device “just‑safe” operating point with BER≈10^-8 at T=75°C (use end‑to‑end self‑checksums on GEMMs or stochastic arithmetic checks).
- Train‑time: inject a noise model matched to that device’s measured error spectrum (bit‑flip distribution by lane/CU, denorm FTZ, occasional FMA LSB corruptions) while training an ESN or a small CNN/RNN. Co‑optimize an operating‑point controller to hold the device at the calibrated cusp.
- Test: measure (accuracy, NRMSE) and Joules/sample on the same device at the target point; then transplant both the model and its operating‑point controller to the twin. Controls: (i) SW‑matched noise; (ii) SHUFFLE noise; (iii) re‑calibrated operating point on the twin. Success criterion: energy benefit ≥15% at iso‑accuracy on the home device, and Δ(off‑diag energy or accuracy) > SW‑matched baseline; improvement disappears when re‑calibrating on the twin.

This makes identity constitutive because the computation’s numerical robustness and its energy controller co‑adapt to the die’s specific error cusp.

4) Simplest existing system with real, quantified transplant degradation

Outside commodity digital, in‑situ/analog training repeatedly shows transplant degradation:
- Gokmen & Vlasov, Acceleration of Deep Neural Network Training with Analog Memory Devices, Frontiers in Neuroscience 2016 — documents device‑variation‑induced accuracy gaps and the need for in‑situ calibration; models trained on/for one crossbar array degrade when mapped to another (quantified in many follow‑ups).
- Ambrogio et al., Equivalent‑accuracy accelerated neural‑network training using analogue memory, Nature 2018 — chip/device variation necessitates per‑chip adaptation; transplanting without re‑tuning degrades accuracy.

Methodologically, port the “hardware‑in‑the‑loop calibration + in‑situ noise modeling” idea: identify the real, per‑die nonidealities that enter the compute path (on GPUs that’s near‑threshold timing/rounding faults under undervolt/throttle) and train under that exact nonideality. Your Phase‑2 tried additive streams; the above shows you need compute‑path faults, not side streams.

5) Hybrid software tactics to weaken the digital abstraction

- Near‑threshold operation: hold clocks high and power‑cap low to induce repeatable timing faults; or undervolt within the driver’s allowed range. Bind computation to the emergent, per‑die error pattern (not just noise amplitude).
- Real‑time constraint coupling: enforce hard deadlines so silicon speed variance changes loss (missed‑deadline penalties). Train the model+scheduler jointly to a specific device’s latency distribution.
- Determinism toggles: enable FTZ/DAZ and fast‑math; collect per‑CU denorm/rounding behavior under thermal stress; couple layer‑norm/activation scaling to those quirks.
- Bank‑conflict shaping: map tensors to stress specific LDS/cache sets; train with those latencies in the loop so the model structure co‑adapts to the device’s physical layout–induced conflict patterns. Transplant changes which sets are hot ⇒ performance/accuracy under real‑time penalties shifts.

Your SHUFFLE result suggests the above only works if faults/latencies are in the critical path (deadline- or error‑coupled), not as side inputs.

6) Cheapest physical add‑on to enable constitutive identity

- $5–$20 MCU as a physical co‑processor reservoir: e.g., RP2040 or SAMD21 board exposing its on‑chip ring‑oscillator/RC‑osc jitter and ADC offset/noise as the recurrence. Train end‑to‑end with this in the loop; the analog dynamics are chip‑unique and lie directly on the compute path. USB latency is fine for slow RC tasks.
- $50–$90 tiny FPGA (Lattice iCE40UP5K “iCEBreaker”): implement a ring‑oscillator fabric or carry‑chain time‑to‑digital converter as a recurrent primitive. Train with it; transplant to another board degrades unless retrained.
- $20 USB audio codec + noise diode front‑end: use analog noise + simple nonlinear analog filter (op‑amp saturating loop) as a physical reservoir with measurable, chip‑unique transfer; bind the task to it.

These keep the host GPU for heavy lifting but put a truly device‑unique, in‑path physical dynamic where identity lives.

7) FPGA route vs middle ground

Full custom FPGA ML acceleration is indeed ~10–100× dev effort. But using a tiny FPGA/MCU purely as a physical primitive (PUF/reservoir) is a Pareto middle ground:
- Dev cost: days to weeks (Verilog/VHDL or even HDL‑free with open cores) vs months for full accelerators.
- Host integration: stream features over USB/UART; training stays in PyTorch/JAX; gradients can use straight‑through or measured Jacobians.
- Robustness: you get transplant‑breaking dynamics without rearchitecting the ML stack.
Compared to “USB ADC only,” the small FPGA/MCU adds reconfigurable dynamics (rings, metastability, TDCs) that are known to be highly unique and reasonably stable.

8) Brutal honesty

Yes, for general‑purpose CPUs/GPUs you are running into a well‑established architectural reality: the digital contract intentionally removes per‑die analog quirks from program semantics. Two decades of design (pipelining, timing closure, ECC, guardbands, runtime management) aim to make instance‑level differences manifest only in performance and power, not in numerical results. Your Phase‑1 NULL and SHUFFLE failures are exactly what that “abstraction‑tax” predicts. To get constitutive identity without exotic hardware, you must either:
- Break the contract (operate near error thresholds or impose tight real‑time constraints so timing becomes semantics), or
- Add a tiny physical element whose analog dynamics sit in the compute loop.

References (for context, not claiming commodity constitutive binding):
- Naghibijouybari et al., Rendered Insecure: GPU Side Channel Attacks are Practical, IEEE S&P 2018.
- Maes, Physically Unclonable Functions: Constructions, Properties and Applications, Springer 2013.
- Uchida et al., Embedding Watermarks into Deep Neural Networks, ICMR 2017.
- Ambrogio et al., Equivalent‑accuracy accelerated neural‑network training using analogue memory, Nature 2018.
- Gokmen & Vlasov, Acceleration of Deep Neural Network Training with Analog Memory Devices, Frontiers in Neuroscience 2016.

Tie‑back to your Phase‑1 verdict: since the stable‑bit channel didn’t pass PUF gates and the process‑stat channel likely mixed ambient/thermal confounds, postpone Phase‑2 in its current form. Instead, prototype the near‑threshold compute‑path scheme (Q3/Q5). It is the most credible path to constitutive binding on gfx1151 without new hardware, and it comes with a clear falsifier (re‑calibration on the twin cancels the effect).
