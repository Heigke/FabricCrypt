# openai response (gpt-5) — 95s

Q1 — FRAGILITY

Short answer: The dual-headline is defensible; “shippable” is fragile only if we imply the compact model is production-ready. As written, the brief mostly avoids that—but two spots need hardening.

Where it’s strong
- Network headline stands on its own data: 83.86% at N=16 384, n=10, CI ±0.17 pp (2026-05-13 12:38). It does not depend on the inert flags.
- RNG headline likewise stands on independent evidence (z296b NIST 5/5; ESS 1.03×).
- The brief explicitly discloses the infra gap: “_residuals does not consume rbodymod, avalanche, TAT flags…” (section 6; also 11:40 bisection log).

Where today’s framing is fragile or overclaimed
- Energy claim lacks provenance: “35 nJ/inf” is an estimate with missing derivation footnote (2026-05-13 14:00 “Still-missing… 35 nJ/inf derivation footnote”). Action: keep “estimate” label; add a ±30–50% uncertainty or pull into a footnote until derived.
- Snapback “4/6 within 0.3 V” depends on a post-hoc V_G1 shift (+0.752 V) and includes a 0.45 point outside Sebas’s rows (interpolated) with a strong outlier at V_G2=0.05 driving the slope inversion (12:35). Action: call it “qualitative presence; slope sign wrong,” which the brief already does; avoid implying quantitative agreement.
- “NS-RAM-encoded features” could be misconstrued as “better than random mapping.” We haven’t run the ablation against i.i.d. random HD features at the same N. Action: add a one-liner in §4.1: “We have not yet shown superiority over random HD features; ablation queued.”
- Compact-model headline number: the brief correctly calls out bimodality (12:30 z318), but any stray use of “0.99 dec baseline” without stratification is an overclaim. Action: never show the single number alone (the brief already conforms).
- Infrastructure-gap risk: z313 proved cfg flags inert (11:40). The brief says “multi-day code work needed.” “Shippable” is fine as an internal report; for any external share, prepend “model under rebuild; flags inert; results based on z304 topology.”

Verdict on “shippable”: Yes, for Mario/Sebas, because (a) the two headlines are empirical and locked, and (b) the infra gap is disclosed prominently. Not for public without the energy footnote and the random-features ablation.

Q2 — FALSIFICATION (pre-registered)

Target the strongest claim: Headline A (83.86% at N=16 384). Best single 1–2 h falsifier is the “Random-features ablation” at identical N/energy.

Design
- Conditions (fixed): UCI-HAR splits and preprocessing identical to z312/z312b; N=16 384; Q=32; V_G1=V_G2=0.3 V; σ ∈ {0.00, 0.05}; k-NN classifier as in z312; n=10 seeds each condition.
- Two encodings:
  1) NS-RAM-encoded features exactly as in z312b (control).
  2) Pure i.i.d. random HD features (Bernoulli(0.5) or Rademacher) with the same bit density and the same per-inference energy accounting (surrogate still 35 nJ/inf).
- Metrics: mean accuracy, std, CI95 via t-interval; effect size Δ = acc(NS-RAM) − acc(Random); two-sided Welch test.
- Gate (pre-registered):
  - If |Δ| ≤ 0.3 pp and CI includes 0 (p ≥ 0.05), we must strip any implication that NS-RAM encoding is superior; headline becomes “HDC at 84% with large-N random projection; NS-RAM is an energy-feasible physical instantiation,” not a performance enhancer.
  - If Δ ≥ 0.5 pp and CI excludes 0 (p < 0.05), we may claim “NS-RAM-encoded features outperform random HD by ≥0.5 pp at N=16 384.”
- Time budget: Existing z312 pipeline; 2×10 seeds at N=16 384 typically completes in ≤2 h on current GPU/CPU.

Rationale
- This directly probes whether the “NS-RAM-encoded” qualifier adds value beyond standard HD baselines. It is cheap, decisive, and acts on the primary headline.

Q3 — NO-CHEAT (discipline drift check)

Have we acknowledged the bimodal DC when framing HDC? Yes.

Citations
- 12:30 — z318 bootstrap: “DO NOT say ‘z304 0.99 dec baseline’ — it’s bimodal … V_G1=0.6 ~0.1 dec; V_G1=0.2 ~4.7 dec.”
- 14:00 — 4E brief §1/§3: “Best DC fit … median 0.99 dec (CI95 [0.93, 2.22])… hides strong V_G1 stratification… V_G1=0.6 ~0.1 dec; V_G1=0.2 ~4.7 dec.”
- 12:25 — O55 corrections: “z304’s ‘0.99 dec baseline’ without CI is asymmetric rigor… Fix: row-bootstrap 1000 resamples → 95% CI.”
- 11:40 — z313 bisection: infrastructure gap disclosed; flags inert; polarity flip caused regression.

Potential drift to watch
- The HDC headline itself reads “83.86% … at N=16 384,” without tying to DC fidelity. That’s acceptable because the operating point is V_G1=V_G2=0.3 V (stated in §4.1) and the catastrophe is at V_G1=0.2. To preempt misreadings, add: “Network uses V_G1=0.3; known DC misfit at V_G1=0.2 is outside this operating point.”
- Energy figure still labeled “estimate”; missing derivation (14:00) is a rigor gap, not a concealment.

Conclusion: Since O55 we corrected the cherry-pick, surfaced bimodality with CI, and reflected it in the 4E brief. No evidence of current cheating-by-omission; just add the one-liner noting the 0.3-V operating point vs the 0.2-V DC failure and finish the energy footnote.
