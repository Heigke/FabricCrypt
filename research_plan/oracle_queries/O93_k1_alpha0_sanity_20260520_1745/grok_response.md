# grok response (grok-4-latest) — 22s

**Q1 — Yes, mechanistically sound and publishable as config recovery, not fitting.**

The K1 override at VG1=0.6 (pillar_I_C3_jts_tat.py:92) explicitly lowered threshold by ~0.12 V, inflating floating-body Vbs modulation and triode current error; restoring the Mario card value 0.53825 directly corrects Vth(Vbs) without introducing a free parameter. ALPHA0 recovery from the CSV typo (7.842e-5) to the LALPHA0_FIX card (7.83756e-4) restores correct Iii prefactor per BSIM4 §6.1, increasing body charging exactly where impact ionization matters. The ablation.json numbers confirm super-additivity only at VG1=0.6 (Δ = −1.114 dec vs −0.436 or −0.805 alone), proving the two bugs were masking each other rather than compensating randomly. Reviewers will see documented provenance (M1_130DNWFB_LALPHA0_FIX.txt, 2Tcell_BSIM_param_DC.csv line 140) and zero new knobs; the risk is low if the paper states “reverted hand-coded overrides to supplied foundry card” and reports the full 66-point CI.

**Q2 — Top-3 candidates for the 0.75 V knee, ranked by likelihood given static-DC data.**

1. Direct gate-body coupling (Cgb offset on floating body). Run a single-bias transient with VG1 stepped 0→0.6 V at Vd=0 V and measure initial Vb step (SPICE .op or .tran 1 ns); a 0.2–0.3 V offset would shift the knee by the observed amount.

2. Missing low-Vd BBT/Hurkx component independent of impact ionization. Sweep Hurkx gamma from 0 to 2e-6 in <30 min on the worst VG1=0.6 point; if the knee moves below 1.0 V while DC fit stays flat, this is the path.

3. Under-estimated NPN Bf or unmodeled M2 BVDSS avalanche. Fix Bf=100→300 in one card edit and re-run the four-panel snapback plot; a 0.6–0.7 V left-shift with unchanged subthreshold slope falsifies or confirms in <1 h.

**Q3 — The residual 0.665 dec is now more likely another config or topology bug than new physics.**

All prior structural additions (self-heat Rth, Hurkx, rbodymod) worsened or gave <0.03 dec; the 1.163→0.665 improvement came solely from restoring card values that had been overwritten. The remaining symmetric fwd/bwd gap (diag_verdict.md: ρ=0.95, median |Δ|=0.0035 dec) and the fact that top-5 outliers are all VG1=0.6 points point to a still-missing static path (parallel leakage or emitter-area scaling) rather than memory or BBT. Until the supplied Mario card set is exhaustively audited against every parameter actually used in build_pyport_base(), declaring “structural addition required” is premature.
