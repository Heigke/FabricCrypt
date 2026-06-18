# grok response (grok-4-latest) — 53s

### Oracle Q1: Most Defensible Headline
**z312 HDC (84.09% at N=16384, σ=0.05) is the most defensible shippable v4.4 brief headline.** It's a direct, quantitative advance in core NS-RAM application (HDC scaling + noise tolerance), with tight std=0.20pp, energy=35 nJ/inf, and continuous gains from N=1024 (80.23%) to 16384. This anchors v4.4's value prop: sub-100 nJ edge ML at scale. M2 traps (6.2-dec lift) is strong physics confirmation but stub-only (no full integration, overshoots measured hyst by 14× per summary.json's distance=0.071x to 0.0026 measured). T2 data harvest (143 samples) is enabling infrastructure, not a standalone result—useful for validation but not headline material.

**Honest overclaim check:** Yes, z312 is overclaimed as "locked" at n=4 seeds (per P1 drift flag: pre-reg requires ≥10 for lock). Current Δ=0.18pp "noise-benefiting" is ambiguous (within std=0.20pp noise). M2 overclaims "confirmed mechanism"—z311 stub lifts 1.66e6× over z308 baseline, but curve shape unreported vs slide-21, and τ={0.1,1,10}s arbitrary (not calibrated). T2 no overclaim—143 samples with ±25-35% uncertainty is honest extraction, though V_d>3V remains unconstrained.

(187 words)

### Oracle Q2: Cherry-Picking Check
**Yes, reporting z304's 0.99 dec as "baseline" without CI is cherry-picking—mild but real.** z313's 4 variants are identical at 2.91 dec, correctly implying infrastructure inertness (flags not consumed). But z304 is a single deterministic run (no stochasticity, per context: "z304 baseline 0.99"), so no inherent variance or CI to report. The issue is inconsistent rigor: you highlight z313's multi-variant identity for diagnostic power, but treat z304's singleton as gospel without noting it's equally un-CI'd. This asymmetrically favors the regression narrative (Δ=-1.92 dec) without quantifying baseline stability—e.g., if z304 re-run with minor numeric noise (float precision, grid interp), it might vary ±0.01 dec, shrinking apparent regression.

Not malicious, but selective: full honesty requires stating "z304 singleton 0.99 dec (deterministic, no CI)" to match z313's multi-run framing. Per P1 drift corrections, this echoes over-reliance on low-n numbers (like z312's n=4 "lock"). Infrastructure gap is real, but baseline reporting lacks parity.

(178 words)

### Oracle Q3: Next Highest-Value Experiment
**C: Snapback peak law sweep—highest value in 1-3h wall time.** Given infrastructure gap (unwired flags need multi-day code for DBR/avalanche), C directly leverages today's partial PASS (V_peak law 2/2 matches at V_G2=0.05/0.10, Δ<0.12V). Sweeping more V_G2 (e.g., 0.20-0.50V) on current pyport (z313e variant, refined R_body) quantifies full law match vs P2's -0.625 V/V slope—falsifiable, builds snapback trajectory evidence for v4.4 brief without code fixes. Outputs: RMSE vs law, potentially 4/4-5/5 PASS to headline as "quantitative snapback reproduction."

A is root-cause fix but explicitly multi-day (>3h gate). B extends traps (z311 already over-lifts; per P1, defer until envelope correct—risks compensating errors). D is passive wait (z312 n=10 in flight; add nothing high-value while idling). C advances open snapback gap pragmatically, aligns with revised v4.4 path (use law as trajectory evidence).

(168 words)
