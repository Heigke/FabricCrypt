# grok response (grok-4-latest) — 10s

**Highest-leverage moves (ranked by expected separation gain × feasibility under thermal constraint)**

1. **Lock-in / frequency-domain readout (candidate 3)** — highest expected gain  
   PDN impedance exhibits die-specific poles/zeros from on-die inductance, decap placement variation, and metal stack mismatch. Time-domain scalar coupling captures only the DC/average term; frequency response samples the transfer function at multiple points. Real precedent: on-die PDN characterization via two-tone or pseudorandom drive (e.g., Köse et al., “Active filter-based on-die PDN impedance measurement,” IEEE TVLSI 2014; similar techniques in power-side-channel literature).  
   Expected: large lift in intra/inter gap because resonances are structural, not just amplitude.  
   Thermally-safe protocol:  
   - Drive u-bursts and v-bursts with two orthogonal Gold-code sequences (length 127 or 255) at 50–200 kHz base rate, duty cycle ≤8 %.  
   - Record per-zone power/thermal telemetry at 1 kHz.  
   - Demodulate at sum/difference frequencies and at 8–12 discrete tone pairs swept 10 kHz–2 MHz.  
   - Keep total active time <4 s per run, idle ≥30 s between runs; monitor instantaneous package power <35 W.  
   - Accept if the frequency-response vector yields min_intra > max_inter on the same matched-temperature band (pre-registered threshold: gap ≥0.25 with n≥5 runs per die).

2. **Template enrollment + simple majority / averaging (candidate 2)** — solid, low-risk gain  
   Standard PUF enrollment step. Real machinery: fuzzy extractors (Dodis et al., “Fuzzy Extractors,” SIAM J. Comput. 2008) and temporal majority voting. With current intra cosine ~0.76, K=5–7 runs per die already pushes effective intra similarity above 0.90 while leaving inter unchanged. No error-correction needed yet; plain averaging suffices.  
   Protocol: enroll 7 runs at 49–56 °C, store mean template; classify new run by cosine to nearest template. Thermally identical constraints as above.

3. **Differential sensor-pair features (candidate 4) + selective feature pruning (5)** — quick orthogonal improvement  
   Subtracting adjacent or opposite zones cancels global temperature drift and common-mode PDN noise. Real analog: differential side-channel preprocessing. Combine with keeping only the 4–6 zone pairs that already show largest intra–inter gap in the existing data. Expected additive gain ~0.08–0.12 on top of Cnorm.

4. **More zones (candidate 1)** — modest net gain, scheduler risk  
   16-core spatial map is richer, but Linux scheduler migration adds run-to-run variance. Worth doing only after differential features and only on pinned cores (taskset + isolcpus). Marginal return drops after ~12 zones.

**Items with low or negative expected return under current constraints**
- Candidate 6 (longer runs) — variance reduction is real but slow; thermal budget is better spent on more short runs.
- Candidate 7 (composite LM adapter) — re-labels the problem rather than strengthens the physical signal; does not produce clean separation on its own.

**PUF reliability toolkit worth importing (ranked)**
- Temporal majority voting / repeated enrollment (biggest immediate lift).  
- Fuzzy-extractor / helper-data construction once Hamming distance on quantized features is measured (Dodis 2008; also Bösch et al., CHES 2008).  
- Uniqueness metric (inter-die fractional Hamming) and reliability (intra-die) tracked separately; target BER <0.05 before extractor.

**Brutal verdict on n=2**  
With only two physical dies you cannot establish statistical uniqueness or clean separation in any generalizable sense. The observed gap is a necessary but not sufficient condition. Minimum convincing experiment requires at least 5–8 dies of the identical SKU (ideally from different weeks/batches) plus the frequency-domain protocol above, with pre-registered acceptance: min_intra > max_inter + 3\sigma on a held-out run set.  

**Single highest-leverage next experiment**  
Run the two-tone Gold-code frequency sweep (protocol in #1) on the existing two dies, 8 runs each, matched 49–56 °C band. Pre-registered criterion: frequency-response cosine yields min_intra ≥0.88 and max_inter ≤0.82. If met, proceed to 5+ additional dies; if not, the räkna-unikt path is probably not worth further hot-run budget versus consolidating generic RÄKNA + separate freshness channel.
