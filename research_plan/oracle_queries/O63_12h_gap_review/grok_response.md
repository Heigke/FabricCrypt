# grok response (grok-4-latest) — 42s

### Oracle Response to Q1 (187 words)

Falsificationist stance: z304's 0.99 dec must be presumed spurious until proven true-physics via targeted tests. Evidence leans spurious: pre-R-20 BJT was reverse-biased (Vbe negative, Q1 off), yet fit achieved sub-1 dec by compensating via other params (e.g., overdriving alpha0/Bf to mimic absent BJT action). Post-fix (z337: 4.16 dec, z338: 3.43 floor) exposes this—correct BJT activates overshoot, unmasking compensation artifacts. R-16's basin existence (||R||²~1e-29) falsifies "no physical basin" but doesn't confirm z304's was physical; R-17's unanimous symptom verdict + R-19's ||R||_inf=3.27e-7 (ambiguous) suggest z304 settled in bug-compensated local min, not global physics.

Test to falsify "true fit": Re-run z304 cfg with BJT fix—predict jump to >3 dec, proving spurious. If stays <1, it's true (unlikely). Goal shift warranted: abandon "match z304" (bug artifact); target new sub-1 basin with correct BJT, prioritizing structural audits over param tweaks to avoid new compensations.

### Oracle Response to Q2 (212 words)

Falsificationist lens: R-22's 3.43 dec floor and top-5 cluster (3.42-3.55) falsify "local min" hypothesis—it's a plateau, implying params can't dip lower without structural change. Cherry-pick risk high: BBO sampled log-scale but clustered, suggesting exploration missed deeper basins due to fix-induced topology shift. Key: BJT fix (bjt_emitter_to_gnd=True) toggled ON, but z304 likely ran with effective "OFF" (bugged Vbe reverse, mimicking disabled BJT). This exposed un-tuned elements, but plateau indicates missing flag reactivation.

Candidates to test (prioritize falsifying "no structural toggle needed"):
- eta_sigmoid: Likely ON in z304 (chokes Iii pre-knee); disable to test if it masked bug—predict pre-knee slope sign flip (R-14) resolves, dropping dec.
- body_pdiode_to: R-17 flagged M2.B defect; if toggled to GND post-fix, test revert to Vb—predict Vb basin shift, breaking plateau.
- use_lateral_collector: If OFF in fix, re-enable—falsifies by checking lat_BV interaction.
- Others (use_well_diode, m2_body_gnd, vnwell_Rs): Lower priority; sweep if top fail.

Run selective toggle+refit: If any drops <2 dec, falsifies "params alone suffice." Plateau persists? Proves deeper rewrite needed.

### Oracle Response to Q3 (198 words)

Falsificationist priority: Select experiment that maximally risks disproving "sub-1 dec achievable with current pyport arch." Goal: 1h runtime, high info yield—either cross gate or force rewrite conclusion.

Ranked candidates (top first):
1. **cfg-diff (R-23 in flight)**: Highest value—directly falsifies "no structural diff" by diffing z304 vs z338 cfgs, reverting BJT-fix mismatches (e.g., eta_sigmoid disable, M2.B toggle). Predict: recovers ~0.99 dec if z304 spurious compensation reversible; if not, proves arch limit. ~20min, builds on R-22.
2. **full DOE 2^7 cfg-flag sweep**: Strong falsifier—exhaustive (128 runs) over candidates (use_well_diode etc.); ~45min parallelized. Risks proving no combo <1 dec, forcing rewrite.
3. **Vsint-residual sensitivity audit**: Tests basin stability post-fix; ~15min, but low risk—unlikely to cross gate, more diagnostic than decisive.
4. **forward-2t solver init from ngspice basin warm-start**: Warm-start falsifies solver-basin hypothesis; ~30min, but R-16 already exonerated residual.
5. **ngspice handover (ngspice OPs then pyport I-V)**: Hybrid sidesteps pyport flaws; ~40min, but doesn't falsify arch limits—evades core issue.

Run #1: In-flight, minimal overhead, directly informs if rewrite inevitable.
