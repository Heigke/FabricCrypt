# grok response (grok-4-latest) — 31s

**Highest-leverage fixes ranked (intra-die cosine gain vs. inter-die separability, commodity feasibility)**

1. **DVFS/governor pinning + fixed P-states** (gain: high; commodity: very high).  
   Hot Pixels (USENIX 2023) shows telemetry is overwhelmingly frequency-dominated. Fixing CPU/GPU P-states, disabling boost, and pinning `performance` governors removes the dominant non-stationary variable. Expected intra cosine lift from ~0–0.6 → 0.75–0.85 on the same hardware. Protocol: `cpupower frequency-set -g performance --min 2.4 GHz --max 2.4 GHz`, equivalent GPU clock lock via `rocm-smi`, run under `isolcpus`. Thermally safe at 60–72 °C with 20 % duty-cycle bursts.

2. **Closed-loop temperature lock (PID background load)** (gain: high; commodity: high).  
   JESD51-1 and JESD51-14 require ±1 °C control for reproducible thermal impedance measurements; PDN poles shift ~0.5–2 %/°C. Hold die at fixed 65 °C setpoint via a low-frequency square-wave background thread (GPU or CPU) whose average power is PID-controlled on a 1 s loop using on-die `Tctl`. This eliminates the 57→71 °C drift observed. Protocol: 5 min soak at setpoint, then measurement epochs only while |ΔT| < 0.5 °C. Uses existing SMN temperature registers; no extra hardware.

3. **Fixed normalization baseline per die** (gain: medium-high; commodity: very high).  
   Using a single enrollment median/MAD (taken once at locked T/f) removes the 5-order run-to-run swing. It does not hide drift if T and f are already locked; it only removes arbitrary per-run scaling. Expected intra cosine gain ~0.15–0.25.

4. **Coherent averaging over N epochs** (gain: medium; commodity: high).  
   Repeating a deterministic 2-tone epoch (period ~50–200 ms) and complex-averaging the demodulated bins reduces uncorrelated noise as 1/√N. N = 32–64 typically yields >0.9 intra cosine once T/f are locked (standard lock-in practice, see Pintelon & Schoukens, System Identification, 2012). Keep total energy low-duty to stay <72 °C.

Lower-ranked: equivalent-time sampling and burst-mode 450 kHz capture are theoretically attractive but face software-timed jitter (no hardware timestamp on MMCFG reads) and thermal-window limits; expected net gain lower on commodity Linux without kernel modifications.

**Supersampling / aliasing notes**  
Equivalent-time sampling is viable only if the drive pattern is strictly periodic and the sample clock drift is characterized (<0.1 % over 10 s typical for TSC). Without a stable reference clock or hardware timestamp, phase stepping accumulates jitter; literature (e.g., Agilent/Keysight equivalent-time scopes) shows software-only ETS rarely exceeds 10–20× effective bandwidth. Bandpass undersampling of a known resonance is feasible in principle but requires prior knowledge of the exact pole frequency on that die. 450 kHz burst windows (thermally limited to <2 s) can expose kHz–MHz impedance structure but will still be dominated by the same T/f variables unless items 1–2 above are fixed first.

**Better discriminative features**  
Once stationarity is achieved, the most temperature-robust die-specific feature is the complex two-tone transfer function H(f₁±f₂) normalized by on-die voltage telemetry (Pintelon & Schoukens system-ID framework). Pole-zero extraction or low-order cepstral coefficients of the PDN impedance are next; both are less sensitive to absolute temperature than raw amplitude once the operating point is locked.

**Grid-search design with N=2**  
Use a two-stage nested protocol:  
- Stage 1 (reproducibility only): on Die A, optimize parameters to maximize intra-die cosine across ≥10 runs; accept only configurations that reach >0.90.  
- Stage 2 (separation): freeze the parameter set and test on Die B (and any future dies).  
This avoids overfitting because inter-die contrast is never used for selection. With N=2 the honest claim is “parameters exist that produce high intra-die reproducibility on these two dies; generalization remains untested.”

**Zoom-out assessment**  
Die-specific analog multiplication via PDN contention is elegant but fragile because the observable is a small nonlinear residual on top of large linear terms. More reproducible alternatives already demonstrated in the literature include:  
- Ring-oscillator frequency ratios (PUF literature, e.g., Suh & Devadas, DAC 2007).  
- SRAM startup-value-conditioned lightweight compute (Holcomb et al., IEEE S&P 2009).  
- Memory-controller latency fingerprints under controlled access patterns (e.g., Pessl et al., USENIX 2016).  
Any of these can be fused at the LLM level with the already-working generic u·v channel rather than forcing the analog product itself to be die-unique.

**N=2 experiment that is actually convincing**  
Pre-registered protocol:  
- Lock T = 65 ± 0.5 °C, f = fixed P-state, normalization baseline from enrollment run.  
- 64 coherent epochs per trial, 10 independent trials per die.  
- Primary metric: intra-die mean cosine > 0.90 on both dies; secondary: inter-die cosine < 0.60.  
This supports the claim “reproducible features exist under locked conditions on these two dies.” It does **not** support “these features will generalize to other Strix Halo dies or survive process variation.”

**LLM integration (minimal, testable now)**  
Expose the live u·v scalar (generic version) as an additional input token embedding at every forward pass (or every k-th layer). Train a tiny adapter (LoRA rank 4–8) on a mixture of normal text plus “poison” examples where the u·v value is replaced by replayed or zero values; the loss must rise sharply on poisoned inputs while clean perplexity stays within 0.5 % of baseline. Real dependency is proven if (a) clean accuracy matches the frozen model and (b) any replay or different-die u·v stream drops accuracy below a pre-registered threshold (e.g., 15 % relative increase in perplexity). This can be wired today while räkna-unikt work continues.
