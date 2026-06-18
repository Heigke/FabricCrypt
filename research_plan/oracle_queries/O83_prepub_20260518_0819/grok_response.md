# grok response (grok-4-latest) — 12s

**Q1 — REMAINING HEADLINE RISKS**

The single most fragile surviving claim is the “one physical primitive, six modes, only V_b/V_d programming” assertion now carrying the entire paper after the ERvMESH reservoir demotion. A reviewer will demand the missing same-cell ablation matrix (promised in the outline but absent from v4.5): identical 2T cell instance, fixed geometry and R_body=1e7, all six modes exercised by nothing more than gate-bias changes, each meeting its pre-registered AMBITIOUS gate on the same Monte-Carlo seed set. Expected outcome: at least two modes (SUPPRESS LMS-Eq and PLASTIC STDP) will require bias points that push the cell out of the verified 7/9 z461 window or into the snapback region where IFT becomes singular, forcing either (a) mode-specific topology tweaks or (b) admission that “programming” includes hidden peripheral state. Either result falsifies the unified-substrate headline with one figure.

Secondary risk is the §3 TRNG NIST 5/5 claim at 0.4 pJ/bit. Reviewer will ask for the cross-cell correlation matrix on ≥100 instances; if ‖C−I‖_F/N exceeds ~0.25 the entropy argument collapses and the “GENERATE” mode reduces to a conventional avalanche diode.

**Q2 — UNFALSIFIABLE FRAMING DETECTION**

Two of the six modes are currently unfalsifiable descriptions rather than quantitative claims. DETECT (“anomaly via body-charge”) and PLASTICIZE-UNDER (“STDP eligibility from body-state”) have no pre-registered numerical bar beyond “we observed correlation.” A reviewer will note that any floating-body cell exhibits history-dependent current; without a stated effect-size threshold (e.g., NAB score > HTM-Java baseline + 3 pp, or retention drift <5 % after 10^4 events) these sections are merely “the device does what physics predicts.” GENERATE, SUPPRESS and COMPUTE-THROUGH still carry concrete gates (NIST, BER floor, IFT κ(J) bound) and are therefore falsifiable; LEARN-FROM is borderline because its whitening fix was post-hoc.

**Q3 — KILLSHOT WE HAVEN’T YET TRIED**

Run a single 2T cell instance through the full six-mode sequence while enforcing strict parameter lock: same V_b, V_d, R_body and Monte-Carlo seed vector for every mode, with only the read-out circuit and digital post-processing allowed to change. Measure whether all six ambitious gates can still be met without re-tuning. This is not the ERvMESH or peripheral-energy tracks already running. Failure on even one mode (most likely SUPPRESS or PLASTIC) would prove the “shared substrate” claim requires mode-specific auxiliary state, collapsing the central thesis that digital silicon needs six macros while NS-RAM needs one.
