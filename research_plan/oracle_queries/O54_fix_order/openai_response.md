# openai response (gpt-5) — 166s

Q1 — Optimal fix-order (next 8h compute)
Implement first (in this order), then sweep DC on 33 IV + validate on the 143 Vd>2V set; run one transient triplet (0.017/0.17/1.7 V/s) after DC converges.
1) VNwell→VB diode, correct polarity (one-line z310b). Zero DC risk; required for any dynamic body discharge.
2) Distributed body resistance (rbodymod=1 equivalent in pyport). Structural fix to snapback envelope; should collapse the 9-decade per-branch Rs split and cut DC error most.
3) Drain avalanche M(Vbc) coupled to body. Sets Vd>2 V shape and supplies the body current that makes (1) matter; improves both DC high-V and transient knee.
Defer to second wave (post-DC envelope):
4) SRH/TAT gen-rec (body-side depletion) to set ramp-rate knee shift magnitude.
5) N1 multi-τ traps (3 τ first). Tune only after 1–3 settle; z311 already overshoots loop area, so postpone to avoid compensating for envelope errors.

Q2 — Cheapest falsifiers (≤2h wall each)
- P3 (pyport_v4): Two-ablation harness on ikaros
  A) Base+diode+rbody vs +avalanche (no SRH, no traps). Fit 33 IV; reject if vg1=0.2 signed bias remains ≤ −1.0 dec AND cellwide median_log_rmse improves <0.5 dec over z304.
  B) Transient check at 0.017/0.17/1.7 V/s with same model; reject if knee does not monotonically left-shift as ramp slows. If either fails, SRH is required before traps.
- P4 (KWS attack) on zgx
  Train a non-spiking baseline (logistic/linear SVM) on the exact MFCC/rank-coded features and splits used by the NS-RAM SNN. If baseline ≥70% while NS-RAM ≤25% across 3 seeds, falsify “encoding/data” as the bottleneck and localize failure to the NS-RAM SNN mapping.

Q3 — KWS as v4.4 target?
Recommendation: Do not anchor v4.4 on KWS; lead with HDC (84% @ N=16k, noise-tolerant) + Bayesian RNG, and run the KWS attack as a bounded-gate side experiment (P4 PASS ≥25%).
- For (persist cautiously): NS-RAM’s hysteretic, multi-τ memory matches 10–40 ms audio windows; event-sparse operation can be sub‑100 µW if front-end dominates energy anyway; HDC/RNG show platform viability.
- Against (why not headline it): Body-charge/trap dynamics introduce rate/history dependence that can destabilize streaming inference; MFCC front-end often dominates power; current SNN mapping is at chance—indicates architectural or plumbing fault, not physics-limited accuracy yet.

Q4 — NO-CHEAT drift (specific)
- “z312 COMPLETE — v4.4 headline locked at 84.09%” (11:15). Rule says ≥10 seeds for a headline; only 4 seeds were run. Action: add 6 seeds before calling it “locked.”
- “Noise-benefiting” claim (84.09% at σ=0.05 > 83.91% at σ=0) is within ~0.18 pp with std 0.20 (n=4) → statistically ambiguous. Action: rephrase to “no worse,” confirm after 10 seeds.
- Diode fix described as outcome (“will produce the rate-dependent hysteresis we want”) before rerun. Action: run z310b and report numbers before causal language.
