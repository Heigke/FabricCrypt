# openai response (gpt-5) — 75s

Q1 — Fragile/overclaimed
a) z371 “1.05 floor”: Best point is Sobol (idx=389), plateau is razor-thin (1.048–1.054), global distribution is ugly (median 1.384, p10 1.100). You froze cfg.vnwell=2.0 and body_pdiode_to=vnwell, then searched a narrow 5D box; top hits hug bounds (iii_body_gain near extremes, vnwell_Rs ~1e8). No replicate runs/seed sweeps, no independent sampler agreement on the actual minimum (only near-ties). Median-of-33 hides branch pathology. This can easily be a low-probability Sobol quirk plus biased bounds.
b) DS-N10 sine win: It’s a cherry-picked toy. Four clean sines, no amplitude/phase noise, no adjacent-class confusion. Meanwhile NS-RAM loses Mackey-Glass, NARMA, and MC (your own JSON). A trivial PLL/phase counter or logistic regression on phase would likely hit >97% at negligible energy. No energy-normalized baseline shown.
c) SURR-V4 “88s”/94.6%: 5.4% non-converged likely concentrates in hard avalanche corners (precisely where you need fidelity). No held-out error map, no calibration to NGSPICE, no repeatability across seeds/GPUs. Speed claim is infrastructure noise until accuracy in the failure regions is proved.

Q2 — Single falsification experiment
Impose a monotone-in-VG1, smooth paramization for {Bf(VG1)↓, iii_body_gain(VG1)↓, log10(vnwell_Rs)(VG1)↑} with one global topology and fit only VG1={0.2,0.4}. Then predict VG1=0.6 (held-out) with NO per-VG1 freedom. If cell-wide median ≤1.05 dec (or near 0.965) on the 0.6 branch, the “structural” claim is falsified. Make it adversarial: multiple seeds, independent samplers, and report per-branch errors (not medians).

Q3 — NO-CHEAT drift
Yes. You’re sliding into post-hoc framing and selective survival:
- “R-46 reported cell-wide: 0.965 dec (engineering per-VG1 fit)” and “Honest mean-param fit: 1.192 dec.” You kept the 0.965 headline.
- “R-45 cfg.vnwell=2.0 frozen” + “body_pdiode_to=vnwell” during z371 search — environment sculpting.
- “DS-N11 … predictive-coding head — ridge readout (DS-N17) loses.” Head-specific advantage, then advertised as substrate win.
- “DS-N14 … ENERGY only (detection TBD).” Keeping the part that flatters.
- “DS-N10 … CLEAR WIN” while your own JSON shows NS-RAM loses Mackey-Glass, NARMA, MC. Classic cherry-pick.
- “SURR-V4 88s” touted without accuracy QA (94.6% conv hand-waved).
Surviving-claims list is curated from positives while failures are rebranded as “task not appropriate.” This violates the pre-registered gate spirit.
