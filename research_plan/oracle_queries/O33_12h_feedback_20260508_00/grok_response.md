# grok response (grok-4-latest) — 33s

1. **Decision gate**: Yes, MC>5 crossed (z221 at 5.13), triggering cron's "continue iterating" per protocol (since <10). Body-state PoC validated, but ESN-class needs ~20-100 for robust tasks. Brief/Mario unaffected; no chip-design change. Recommend (b): Push to MC>10 (e.g., denser grid, GRU surrogate, or N=400) before NARMA-10 retry—ensures meaningful benchmark gains vs. z216's weak baseline (NRMSE 0.84). Defer write-up until MC>10 or NARMA win.

2. **Cherry-picking audit**: Yes, iterative multi-knob tuning (Cb, dt, g_VG2, leak) without pre-specification risks p-hacking/overfitting to MC metric—may not generalize to tasks like NARMA. Validation: Pre-register a grid search (e.g., 3-5 values per param, 100+ combos); split data (train MC on 80% seeds, validate on 20%); compute bootstrap CI on held-out MC/NARMA; re-test final config on unseen task variant (e.g., NARMA-30) to confirm robustness.

3. **Next experiment**: Prioritize NARMA-10 re-attempt with z221 config (highest value: directly tests if MC=5.13 yields NRMSE<0.84, unblocks T track, quick ~1h run). Then bigger N=400 (scales MC, checks attractor diversity). Defer GRU (gemini idea, if interp plateaus) and pyport-direct N=32 (sanity check, lower urgency post-PoC).

Total: 248 words. Fire in work-hours: Proceed to NARMA-10 now.
