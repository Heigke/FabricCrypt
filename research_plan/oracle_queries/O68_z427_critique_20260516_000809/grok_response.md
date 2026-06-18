# grok response (grok-4-latest) — 5s

**Q1 — No, the headline is not honest.**

The 1.733 dec cell-wide log-RMSE is dominated by the two high-current branches (V_G1 = 0.4/0.6). The +0.44 dec regression at V_G1 = 0.2 is masked because those 11 points contribute far less to the sum of squared log errors once the high-V_G1 points drop 3+ decades. A median-per-bias residual would almost certainly exceed 2.0 dec; the max-per-bias residual is already ~4 dec in subthreshold. 

For publication the correct metrics are (1) branch-stratified median absolute log error and (2) maximum log error at the lowest gate bias. Cell-wide log-RMSE alone is only acceptable if every branch is shown to improve or at least not degrade; here it fails that test.

**Q2 — Both fixes are under-justified and high-risk of being curve-fits.**

H1’s 1 MΩ value was chosen because it moved V_Sint; no physical extraction or even order-of-magnitude TCAD check is cited. In 130 nm the body resistance to the nearest substrate tap is typically tens to low hundreds of kΩ for this geometry; 1 MΩ is already on the high side and suspiciously convenient. 

H2 is geometrically plausible but exactly the minimal change that eliminates the saturation artifact created by the earlier fixes. The combination therefore has the classic signature of two correlated patches applied to the two branches that were broken, rather than an a-priori structural correction. Until you show that the same topology and resistance work on an independent device or bias region without retuning, treat both as hypotheses, not established fixes.

**Q3 — Highest-value next experiment: blind held-out stripe (candidate 2).**

Run H1+H2 exactly as frozen on a previously unseen V_G2 value (or an intermediate V_G1 = 0.3 stripe) that was never used for topology decisions. This directly tests generalisation with zero new degrees of freedom.

If it fails, the fixes are not portable physics. If it succeeds, then run the H1 resistance sweep on the same held-out set to check for a true minimum versus a plateau. Everything else (switching H2 only at V_G1=0.2, GIDL-only ablation, etc.) is either weaker or still allows post-hoc adjustment.

The sub-threshold over-prediction at V_G1=0.2 remains the dominant open physics problem; the held-out test will tell you whether you have a model or an elaborate interpolator for the three measured V_G1 values.
