# openai response (gpt-5) — 68s

Hostile review — answers keyed to your questions

1) Honesty check
- “Four nulls confirm a structural floor” is too strong. You tested four knobs; that does not prove no remaining single-architecture knob can help. Soften to: “Four oracle-ranked knobs each moved ≤3 mdec; evidence points to a structural floor for the lumped-Vb/single-NPN model; further gains likely require an architectural change.”
- Also fix “52.7% NRMSE reduction” (you reduced median log-RMSE, not NRMSE). Say “52.9% drop in median log-RMSE” or just give the dec values.

2) Figure 5 timeline + “floor band”
- The shaded band invites a cherry-pick critique. It’s defined by the best cells of four targeted null sweeps around one operating point and excludes 8/33 biases. Make it defensible: add a bootstrap across the 25 biases to show the distribution of median log-RMSE at each step (error bars), and state grid sizes (5×5) and fixed seeds. Rename band to “observed plateau across four ranked sweeps.”

3) Stage 6 framing re untested architecture options
- A reviewer will ask whether Two-NPN, body-network, and S–Vb diode were tried. You didn’t. Say it explicitly: “We implemented η(Vbe); Two-NPN, quasi-2D body, and body-network were not tested in v4.2 due to added state/solver complexity; they are first items for M3b/M6.” List expected code delta and planned test.

4) Network demo wording (“best” vs “final”)
- “Best test accuracy” suggests test-set peeking. Use “peak validation” for model selection and report test at the selected epoch. If you keep “best,” define it: “max over training on the test curve (diagnostic only).” Prefer: report final and early-stopped (val-selected) test; drop “best” count.

5) What to cut
- Remove the Stage 4 numeric 2D table; one sentence + the optimum suffices.
- Trim MTK/Enzyme implementation prose; keep one sentence on independent implementation.
- Either the Stage 6 quad or most of its explanatory paragraph is redundant; keep the quad and move the paragraph to caption bullets.
- Consider moving Figure 4 (process cross-section) to appendix if space is tight; one sentence in text can assert “standard 130 nm, no process mods.”
- Keep Figure 3 and the cross-norm table; they carry the tape-out recommendation.

6) Last-mile risks (defensibility weak points)
- Inconsistency: network demo uses “honest” Bf=100, while DC fit optimum is Bf=9×10³ and you argue lateral-parasitic Bf∈[10³,10⁵]. A hostile reader will call this self-contradictory. Either rerun the network at the calibrated point or justify why dynamics/generalisation require bounded Bf and show insensitivity across Bf.
- 25/33 biases: reviewers may ask if the excluded 8 are the hardest. You note NaNs; add a sentence quantifying their expected impact or show a sensitivity bound.
- ngspice validation scope: you admit 100 mV Vb divergence; be ready to defend that this doesn’t affect the DC current fit claims. Consider adding a single 2T DC I–V overlay vs ngspice at a benign bias.
- “Structural floor” proof burden: show confidence intervals across seeds/grids; otherwise it’s a plateau, not a floor.
- Cross-norm sweep: only 5 seeds at N=800; acknowledge limited seeds and add a robustness note (e.g., ER_SPARSE remains top-3 in 4/5 seeds under each norm).
- Energy Figure 7: label the 1024-step point explicitly as “device-only projection; excludes periphery” in the caption and body (you do, but make it bold/first sentence).

Minor edits
- Replace “oracle” with “independent review” or “automated ranking.”
- Stamp “25/33 biases” in every occurrence of the 0.654 dec claim.
