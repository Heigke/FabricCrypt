# O85 — 12h gap-closing review (NS-RAM)

12h status snapshot (from log tail):
- QUAD-KILL today: LMS-Eq 170× DEAD, Hier-MNIST DEMOTE (no NSRAM contrib + peripheral 444× undercount), Stoch-RNG 0.4 pJ DEAD (98.5 pJ honest, 101× iso-node worse vs Cheng 2024), EP-FULL no-Lyapunov FAIL (final 75±16% drift)
- 2 NEW WINS: z477c V7 physical Hopf @ Mario target 420ns w/ V_b ∈ [-0.5, +0.62] V; z481 EP-FIX 90.83±2.60% MNIST 4-seed (drift +0.44 vs -11.75) via β-cos + random-sign + VG1-nudge + early-stop
- INFRA: z478 GPU batched FHN N=1024 in 13.3s on zgx GB10
- KILLS today: z479 NARMA-10 FHN rebuttal FAIL (NMSE 0.346 vs 0.325, reservoir-NARMA permanently retracted), z482 coupled FHN reservoir KILL (all 3 gates fail, mean spike-synchrony 0.0016 = no phase-locking), z483 STDP body-state INFORMATIVE NULL (NSRAM τ_body=1ms ≡ hand τ=1ms +19pp; optimum is τ=0.1ms 94.5%; ablation τ→∞ drops only 3.5pp = decay NOT dominant credit mechanism)
- O84 3-way oracle CONVERGENCE (gpt-5+gemini+grok): single defensible USP = "multi-function 2T cell at 130nm via bias-programming alone — LIF + V6 reset 40ns + V7 Hopf 420ns + intrinsic noise + EP-FIX via diff IFT pyport"
- DS-1 (full-MNIST EP-FIX 5-10 seeds + ablation + alignment-cosine) prioritized but gated on user

3 hostile questions:

**Q1 — Cherry-pick audit on today's 2 wins**
For z477c V7 (420ns Hopf at Mario calibration) and z481 EP-FIX (91% MNIST 4-seed), what is the SINGLE most likely cherry-pick or fragility that wasn't already audited? Name the one experiment that would expose it. Cite numbers from the snapshot if possible.

**Q2 — Reservoir-USP retirement**
With z479 (NARMA-10 rebuttal FAIL) + z482 (coupled FHN KILL) + ERvMESH killshot already retired NARMA, should "differentiable limit-cycle reservoir" be formally KILLED in v4.6 brief, or kept as MG-only/onset-only with explicit scope? Any reviewer who'd accept the latter? Or does keeping it weaken the multi-function USP-1 thesis?

**Q3 — Highest-value next experiment**
Given zgx free + daedalus offline + Sebas silicon unavailable + ~24-48h compute budget, rank these candidates by expected paper-impact-per-hour:
(a) DS-1 full-MNIST EP-FIX 5-10 seeds + ablation + alignment-cosine
(b) bias-programmable mode-atlas in pyport (LIF+Hopf+noise from SAME calibrated cell, bias-only) — software demo of USP-1
(c) GPU 16×16 mode-atlas variance sweep (process-variation surrogate) — DS-3 surrogate
(d) revisit-NES-GD with K2 audit since cell intrinsic noise NOW characterized
(e) something else you'd add

Be brutally honest. Return ≤ 400 words per Q.
