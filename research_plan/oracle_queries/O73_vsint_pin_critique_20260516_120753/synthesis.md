# O73 Synthesis

All 3 oracles converge:

**Q1** Hard pin V_Sint=0 likely legitimate IF substrate current is small (validate by measuring Ipin at pinned node — should be <5% of I_D). 130nm CMOS substrate tie has ~Ω resistance so V_Sint≈0 is correct boundary. Grok demands extracted R_sub.

**Q2** Stratification (drop VG1=0.2) is cherry-pick. Cell-wide 1.619 dec is the honest headline.

**Q3** ALL THREE rank BSIM4 v4.8.3 GIDL §6.2 (d) as #1 next experiment:
- AGIDL/BGIDL/CGIDL parameters refit at VG1=0.2/high-V_D regime
- Single concrete test that should close VG1=0.2 gap
- Keep V_SINT_PIN locked, only fit 3 GIDL params

Concrete fix: enable BSIM4 GIDL §6.2 in pyport, fit 3 params on VG1=0.2 high-V_D tail.
