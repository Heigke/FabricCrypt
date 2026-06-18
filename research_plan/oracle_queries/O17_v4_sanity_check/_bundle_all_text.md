# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: O16_gemini_critique.md (7516 chars) ===
```
# gemini response (gemini-2.5-pro) — 74s

Here is a critical review of the provided materials.

### 1. Does the patched fit actually look good in the plot?

No. The fit quality is highly non-uniform and, in some regimes, qualitatively incorrect.

*   **VG1 = 0.2 V (Left Panel):** This is the best-looking panel. The model captures the general "S" shape of the snapback curves and the approximate current magnitudes. However, there are still visible discrepancies in the subthreshold region (low Vd) and the exact Vd threshold for snapback. The shape is *mostly* right, but the parameters are clearly not exact.

*   **VG1 = 0.4 V (Middle Panel):** This is a catastrophic failure of the model. For nearly all VG2 sweeps (the different colored lines), the model predicts a flat, saturated current around 7-8 µA. The measurement data (the dots), in contrast, shows a family of distinct curves spanning orders of magnitude, from ~1 nA to several µA, before snapback. The user's observation of "flat lines" is correct and points to a fundamental inability of the current model to capture the device physics in this critical operating regime. The shape is completely wrong.

*   **VG1 = 0.6 V (Right Panel):** This panel shows a similar, though slightly less severe, failure. The model again predicts a flat, saturated current around 7-8 µA for many of the VG2 sweeps where the device is actually in a subthreshold or pre-snapback state (e.g., the yellow, light-green, and teal curves). It only begins to capture the snapback behavior for the highest-current curves (dark purple/blue). For at least half of the biases in this panel, the model's prediction is qualitatively wrong.

In summary, the model only appears plausible at low VG1. At mid and high VG1, it fails to reproduce the device's behavior across a wide range of VG2 biases, showing a premature and incorrect saturation.

### 2. What does the per-bias log-RMSE distribution actually look like?

The "median 1.00 dec" summary is misleading and hides a severe heavy tail of poorly-fit biases.

Based on the 29 non-skipped curves in `predictions_patched.json`:

*   **Median:** 1.08 decades. (Note: The brief claims 1.00, which is a minor inaccuracy).
*   **Mean:** 1.69 decades. The mean being significantly higher than the median confirms a right-skewed distribution with a tail of very high errors.
*   **Max:** 3.24 decades. This is an extremely poor fit.
*   **Worst Bias:** The worst fit is at (VG1=0.4 V, VG2=0.3 V) with a log-RMSE of 3.24. This corresponds to the flat-line predictions in the middle panel.
*   **Tail Distribution:**
    *   **15 out of 29 biases (52%)** have a log-RMSE > 1.0.
    *   **11 out of 29 biases (38%)** have a log-RMSE > 1.5.
    *   **7 out of 29 biases (24%)** have a log-RMSE > 2.0.

A quarter of the dataset is fit with an error of over two orders of magnitude. The median-based summary obscures the fact that the model is unusable for a large fraction of the device's operating range.

### 3. Is the brief defensible at the current fit quality?

No, not with its current wording. The brief's central defense of the model's utility is weak and vulnerable to attack.

The key sentence in Section 5 is: *"The simulator is sufficient for the topology and benchmark studies below because the residual error is applied systematically across all compared legs... so reported relative performance differences and the monotonic task-difficulty ordering are robust"*.

This claim is indefensible.
1.  **The error is not "systematic."** A systematic error would be a consistent scaling factor or offset. As shown in the plots, the error is a *qualitative shape error*. The model predicts a flat line where the device has a complex curve. This is not a systematic deviation; it is a fundamental misrepresentation of the physics.
2.  **Consequences for Benchmarks:** Any benchmark or topology study that relies on cells operating in the (VG1=0.4V, VG1=0.6V) regimes is built on a faulty foundation. If the model says the current is 8 µA when it is actually 8 nA, any claims about network dynamics, recurrence, or memory capacity are immediately suspect.

A hostile reviewer has a legitimate and powerful attack vector here. They can argue that because the cell model is qualitatively wrong, all downstream algorithmic results are invalid. The brief overclaims the model's sufficiency.

### 4. What's the worst-case interpretation?

A senior compact-modelling expert would likely dismiss the work with a statement like this:

**"The authors' model fails to capture the fundamental device physics across more than half of the measured operational space, substituting a flat saturation for observed device behavior. Their subsequent claims of 'robust' algorithmic performance are therefore derived from a fiction and cannot be considered credible."**

### 5. What MUST be fixed before send, vs what can wait for M3a?

The goal is not to perfect the model before the deadline, but to make the proposal scientifically honest about the model's current state.

**Must be fixed before sending (2026-05-06):**

1.  **Revise the "Validation Status" (Section 5):** The entire framing must change from "the model is sufficient" to "the model has known limitations that we will address."
    *   Replace the "median 1.00 dec" summary with the full, honest distribution (median 1.08, mean 1.69, max 3.24, and the percentage of curves with error > 1.5 dec).
    *   Explicitly state that the model fails qualitatively at VG1 ≥ 0.4V, showing premature saturation. *Include the patched plot in the proposal itself* as evidence of this transparency.
    *   Heavily qualify the claim of sufficiency. Rephrase to: "While the current model's absolute error is large, we present preliminary benchmark studies under the working hypothesis that relative performance trends may still hold. Verifying these benchmarks with the corrected model is the primary goal of the M3a milestone." This reframes the benchmarks as exploratory rather than definitive.

2.  **Strengthen the M3 Deliverable:** The description of M3a ("close the residual ~0.87 decade... gap") must be updated to reflect the severity of the problem. It should explicitly mention fixing the "qualitative failure and premature saturation at mid-to-high VG1."

**Can wait for M3a (Post-funding):**

1.  **The actual model fix.** This is precisely what M3a is for. The proposal is funding this work, so it's expected to be incomplete now.
2.  **Re-running all benchmarks and downstream studies.** This is the verification step that follows the M3a fix. The proposal should promise this as a later milestone.

---

### Final Verdict

**(B) "Fix [list] in the brief first, then send"**

Do not send the proposal as-is. The current framing is not just optimistic; it is vulnerable to being perceived as misleading. The gap between the model and reality is too large to be summarized by a single median value, and the defense of the model's sufficiency is based on a false premise that the error is systematic.

By explicitly detailing the model's failures in the proposal itself, you transform a critical weakness into a well-defined research plan. This demonstrates scientific integrity and a clear-eyed understanding of the problem, which will be far more credible to a reviewer like Mario Lanza than a flawed model presented as "good enough." Address the points in Q5, making the proposal's text a brutally honest assessment of the current state and a clear plan for the future. Then, send it.

```


=== FILE: O16_grok_critique.md (4521 chars) ===
```
# grok response (grok-4-latest) — 61s

1. **Does the patched fit actually look good in the plot?**  
   The patched fit is poor overall, with widespread shape mismatches and flat predictions failing to capture measured snapback and subthreshold structure. For VG1=0.2V panel: predictions for low VG2 (e.g., ≈-0.2V purple, ≈-0.15V blue) are nearly flat lines across Vd=0-2V, detached from measurements that show rising Id with snapback folds at Vd≈1.2-1.5V; shape is wrong (pred flat, meas curved with fold). Mid VG2 (e.g., ≈0V green) has pred lines rising too slowly and detaching at Vd>1V, magnitude off by 1-2 decades; shape mostly right but scaled wrong. High VG2 (e.g., ≈0.1V yellow) pred overestimates at low Vd, flat where meas curves; shape wrong. For VG1=0.4V panel: all curves show pred as flat horizontals at Id≈10^-6A, completely detached from meas that have snapback at Vd≈1-1.5V and Id rising to 10^-5A; shape wrong across the board, magnitude wrong by 1-3 decades (e.g., VG2≈0.3V yellow pred flat, meas snaps). For VG1=0.6V panel: pred lines are flat or weakly sloped at Id≈10^-6A, missing meas snapback and saturation; low VG2 (e.g., ≈0V purple) pred flat where meas rises sharply at Vd>1V (shape wrong); high VG2 (e.g., ≈0.5V yellow) pred magnitude too high at low Vd, flat where meas plateaus (shape wrong). User's observation of flat lines confirmed, especially dominant in 0.4V/0.6V panels.

2. **What does the per-bias log-RMSE distribution actually look like?**  
   From predictions_patched.json (23 entries, but only 14 non-skipped with log_rmse): median=1.001 (close to claimed 1.00), mean=1.652, max=3.244 (VG1=0.4, VG2=0.3). Counts: >1.0 dec=10 (71%), >1.5 dec=8 (57%), >2.0 dec=5 (36%). "Median 1.00 dec" is misleading as it hides a heavy tail of bad fits (e.g., all VG1=0.4 entries >2.5 dec, skewing mean high); distribution is bimodal with low-VG1 cluster near 1.0-1.9 and high-VG1 outliers >2.5. Worst single bias: VG1=0.4 VG2=0.3 at log_rmse=3.244 (pred Id flat ~10^-6A, meas varies 10^-9 to 10^-6A with snapback).

3. **Is the brief defensible at the current fit quality?**  
   The brief is weakly defensible on relative claims but overclaims absolute fidelity, exposing attack vectors on unsubstantiated topology/sign-asymmetry results. Status (sec 5) claims "Pyport reproduces Pazos’s 33-curve I–V family with full Newton convergence" and "median log-RMSE...1.00 decade", but this depends on absolute fit quality, which is poor (e.g., flat preds miss snapback); a hostile reviewer attacks as cherry-picking median while ignoring tail (e.g., >2 dec outliers). Limitations bullet 5 (sec 7, actually bullet 7 in text) admits "Pyport’s residual ∼0.87 decade gap" and defers to M3a, but claims "reported relative effects...are robust" — this holds for monotonic ordering (depends on relative shapes), but topology recommendation and sign-asymmetry (z122: MC drops with positive-only) depend on accurate snapback modeling, which is absent in flat preds; attack: simulated topologies unrepresentative of silicon. Overclaim quotes: "pyport reproduces his measured I–V family at 1.00-decade median log-RMSE" (abstract/sec5, ignores tail/distribution); "Five reservoir/classifier benchmarks...yield a measured monotonic ordering" (abstract, assumes fit good enough for benchmarks, but shape errors undermine); "That data forces the architecture" (abstract, indefensible if sims don't match meas).

4. **What's the worst-case interpretation?**  
   A senior compact-modeling expert would dismiss: "This so-called 'faithful' PyTorch port is a sloppy hack that mangles basic BSIM4 binning, producing flat nonsense lines instead of snapback physics, rendering all downstream benchmarks and tape-out recommendations worthless garbage."

5. **What MUST be fixed before send, vs what can wait for M3a?**  
   Must fix before send: rewrite abstract/sec5 to remove absolute-fit claims (e.g., "reproduces...at 1.00-decade"), add tail disclosure (e.g., "mean 1.65 dec with 36% >2 dec"); qualify benchmarks as relative-only (sec6: add "assuming shape errors cancel"); excise overclaims like "That data forces the architecture" and z122 sign-asymmetry as premature. Can defer to M3a: binning audit/gap closure (fidelity improvement); absolute RMSE acceptance; thick-ox refit (contingent on data).

**Final verdict:** (B) "Fix [absolute-fit overclaims in abstract/sec5, tail disclosure, benchmark qualifications in sec6, removal of premature topology/sign-asymmetry assertions] in the brief first, then send".

```


=== FILE: O16_openai_critique.md (6955 chars) ===
```
# openai response (gpt-5) — 150s

1) Visual audit of the “patched” fit (the plot with “median log‑RMSE = 1.002”)

- VG1 = 0.2 V
  - Several negative-VG2 curves [VG2 ≈ −0.20 … −0.05 V] are essentially flat plateaus around 10^−9 A from Vd ≈ 0.25–1.5 V, while the measurements keep rising by ~1–2 decades toward Vd ≈ 2 V. Shape wrong, not just magnitude.
  - Small positive VG2 [≈ +0.05, +0.10 V] look closer in shape at low Vd, but are still low by ~0.5–1 decade over most of the sweep and miss the high‑Vd upturn.

- VG1 = 0.4 V
  - The prediction lines for all shown VG2≥0 are nearly horizontal at ~7×10^−6 A across almost the entire Vd sweep, while the measured dots rise strongly with Vd and span orders of magnitude. This is the “flat lines” problem, and it is severe. Examples: (VG1,VG2) ≈ (0.4, 0.25) and (0.4, 0.30) — predicted lines are detached from the data by ~3 decades and show no measured slope or structure.
  - No negative‑VG2 curves are shown here in the patched panel (and they are missing from the JSON too), which already biases the visual impression.

- VG1 = 0.6 V
  - Predictions again cluster on an almost flat shelf around ~7×10^−6 A up to ~Vd≈1.6–1.8 V, then some lines bend up. Measurements start much lower (~3×10^−7 A) and then climb to ~3×10^−5 A; several curves show steady slopes that the model largely misses. For multiple VG2 (0.0–0.3 V) the shape is wrong (flat vs rising), even where the RMSE is <1 decade.

Bottom line: yes, there are many flat, detached prediction lines. The 0.4 V panel is catastrophically flat; the 0.6 V panel is largely flat; the 0.2 V panel has long flat plateaus for negative VG2. “Shape mostly right” only holds for a minority of low‑bias curves.

2) What the per‑bias error distribution really is (predictions_patched.json)

- Dataset included in the median: 25 of the 33 biases (8 biases at VG1=0.4 with VG2<0 were skipped as “NaN row”).
- Summary over the 25 included biases:
  - Median log‑RMSE: 1.002 decades (matches the plot).
  - Mean log‑RMSE: ~1.60 decades (heavy tail).
  - Max (worst single bias): 3.244 decades at (VG1≈0.4 V, VG2≈0.30 V).
  - Counts above thresholds:
    - >1.0 decade: 13/25 biases (52%).
    - >1.5 decades: 11/25 biases (44%).
    - >2.0 decades: 7/25 biases (28%).
  - p90 ≈ 2.94 decades (close to the plotted 2.90).
- Omitted data: the 8 skipped biases all sit in the hardest region (VG1=0.4 V, VG2<0). If they are bad — and the 0.4 V positives already are — the true 33‑bias distribution is worse than reported.

Conclusion: “median 1.00 dec” is a cherry-picked statistic that hides a long, ugly tail concentrated at VG1=0.4 V and positive VG2, with entire families showing 2.5–3.2 decade errors and flat shapes.

3) Is the brief defensible at this fit quality?

- Several key claims are framed as if absolute fidelity is not needed:
  - “the residual error is applied systematically across all compared legs … so reported relative performance differences and the monotonic task‑difficulty ordering are robust” (Sec. 5, last paragraph; repeated in Limitations bullet 6). This is not defensible. The error is not a uniform scale factor; it is strongly VG1/VG2‑dependent and shape‑altering (flat shelves where measured curves rise). Any benchmark whose working point touches the 0.4 V region, or relies on Vd‑slope/snapback structure, can change rank when the model is fixed.
- Overclaims that invite attack:
  - “Pyport reproduces Pazos’s 33‑curve I–V family…” (Sec. 5). It doesn’t — only 25/33 biases enter the statistic; the 8 hardest are skipped.
  - The figures in Sec. 2 and 3 visually suggest “thick = measurement; thin = simulation” agreement across regimes. The 0.4 V regime in the patched plot is nowhere near agreement.
  - The topology and sign‑asymmetry recommendations (Sec. 4 and Sec. 6) are derived from simulations on the calibrated (patched) cell. Given the shape errors and the omitted biases, a hostile reviewer can (fairly) argue that the ordering and the sign‑inverter requirement have not yet been demonstrated on a faithful device model.
- What probably survives even with a mediocre fit:
  - Very coarse qualitative statements that don’t depend on exact magnitudes (e.g., “positive‑only coupling reduces expressivity”) likely remain true, but the quantitative effects (e.g., “MC from 2.90 to 0.43 ± 0.54”) are not defensible until the model is faithful where the networks operate.
  - Transient claims are thinly validated; the brief presents solver convergence and a slide, not an apples‑to‑apples transient fit. A reviewer can and will pounce.

4) Worst‑case interpretation (what the grumpiest compact‑model expert will say)

“This isn’t a calibrated device model — it’s a patched code path that outputs flat, wrong‑magnitude Id–Vd shelves for whole bias regions; the ‘1‑decade median’ is computed on 25/33 curves and hides 3‑decade failures, so none of the downstream ‘benchmarks’ or tape‑out recommendations are credible.”

5) What must be fixed before send vs. what can wait for M3a

Must fix in the brief now (2–4 hours of edits; no code needed):
- Replace the headline claim. Say exactly what the JSON shows:
  - “25/33 biases included; median 1.00, mean ~1.60, p90 ≈ 2.94, worst 3.24 decades; 13/25 >1.0, 11/25 >1.5, 7/25 >2.0.”
  - State explicitly which biases were excluded and why, and show them grayed out in the plot.
- Add one figure panel or inset that highlights the 0.4 V failure (a zoom on (VG1=0.4 V, VG2∈{0.0…0.3})) to make the shape error unmistakable and owned by us, not discovered by reviewers.
- Rewrite Sec. 5 and Limitations bullet 6 to remove “applied systematically across legs” and acknowledge that the error is shape‑altering and concentrated in specific VG1/VG2 ranges; say clearly that any benchmark conclusions are provisional pending M3a binning‑parity and a DC refit.
- Tame claims that look like reproduction:
  - Change “reproduces Pazos’s 33‑curve I–V family” to “converges on all 33; matches 25 biases quantitatively with 1‑decade median under a temporary binning‑zeroing patch; 0.4‑V panel shows large shape errors to be fixed in M3a.”
- Flag the sign‑asymmetry and topology recommendations as hypotheses until the model is corrected; add a one‑sentence risk note that their magnitude may change after M3a.

Can wait for M3a (code + regression work):
- The BSIM4 W/L/P binning audit and removal of the 0.4‑V flat‑shelf pathology; a full 33‑bias no‑patch regression with target ≤1‑decade median and no heavy tail.
- A matched transient validation figure (measurement vs pyport) to support the dynamic claims.
- Re‑running the key benchmarks on the fixed model and updating the sign‑asymmetry numbers.

Final verdict: B — Fix the listed wording and disclosure issues in the brief first, then send. As written, the “1.00‑decade” framing is misleading, entire bias panels are obviously wrong and flat, and 8/33 biases are dropped; sending this as‑is will hand a hostile reviewer an easy takedown.

```
