# openai response (gpt-5) — 286s

Short answer

- The 12% comes from the n-aggregator using the wrong Θ0. In b4ld.c the Θ that multiplies (cdsc + cdscb·Vbseff + cdscd·Vds) is computed with T0 = 0.5·dvt1·Leff/lt1, not the T0 = dvt1·Leff/lt1 used in the Vth DVT block. The port reuses the Vth Θ0, which is too small by orders of magnitude for long L, effectively killing the cdsc branch and leaving only nfactor·(εsi/Xdep)/coxe. That pushes n high and gives the monotonic 1‑decade gap you see.

- In the posted code the n-aggregator is multiplying tmp3 by Theta0 (defined earlier for Vth with T0 = dvt1·Leff/lt1). That is the wrong Θ for the n expression.

- One‑line A/B test: set cdsc = 0 in the card and re‑run the S extraction. If the py/ng gap collapses to ≪12% (it will), the error is in the cdsc·Θ branch, i.e. the Θ definition used in n.

- Fix: compute a dedicated Theta0_n with the 0.5 factor and use it only in the n formula.

Details

1) Where the 12% lives

- It is not Xdep. You already compute the bias‑adjusted Xdep = Xdep0·sqrtPhis/sqrtPhi, which at Vbs=0 equals Xdep0 and matches b4ld.c. Using Xdep0 instead would hardly change n at Vbs=0.

- It is not coxe. The n expression in b4ld.c uses coxe (raw oxide), not Coxeff from the Tcen block. Using cox instead of coxe would move n by only a few percent here, not 12%.

- It is not mstar/Vgsteff mixing. Your reported n is flat vs Vgs (as it should be) while the Id decade gap is nearly flat and vanishes toward strong inversion; that is exactly the signature of the cdsc term being suppressed by an overly small Θ.

- It is the Θ that multiplies cdsc. b4ld.c recomputes a separate Θ for the n branch with T0n = 0.5·dvt1·Leff/lt1. Reusing the Vth Θ (T0 = dvt1·Leff/lt1) makes Θ ≈ e^(−T0)/(e^(−T0)−1)^2 tiny for long L (your L=1.8 μm), so the cdsc contribution to n is almost zero. Ngspice uses the 0.5 factor; the cdsc term then contributes a non‑negligible amount that lowers the effective slope. This is exactly why Δlog10(Id) decreases monotonically with Vgs and tends to zero near on-state in your plots.

2) Which Θ your code uses

- In nsram/nsram/bsim4_port/dc.py you define
  T0_th = dvt1 · Leff / lt1
  Theta0 = _exp_threshold_branch(T0_th)
  …
  tmp3 = cdsc + cdscb · Vbseff + cdscd · Vds
  tmp4 = (tmp2 + tmp3 · Theta0 + cit) / coxe

  That Theta0 is the DVT/Vth one (no 0.5). The n branch in b4ld.c uses a different Theta computed with T0n = 0.5·dvt1·Leff/lt1.

3) One-line A/B test to isolate

- Edit the card and set cdsc = 0 (or run ngspice with .alter cdsc=0) and re‑run S at Vds=0.05 V.
  - If the py/ng S gap collapses, the only place left is the cdsc·Θ term → wrong Θ in n.
  - As a cross‑check, set nfactor=0 instead: if the gap persists, that also points to the cdsc·Θ path.

4) Minimal code change

- File: nsram/nsram/bsim4_port/dc.py

Right after you compute the Vth Θ0, add a second Θ just for n:

  # Vth DVT Θ0 (unchanged)
  T0_th = dvt1 * Leff / lt1.clamp_min(1e-30)
  Theta0_vth = _exp_threshold_branch(T0_th)

  # Use Theta0_vth for Delt_vth:
  Delt_vth = dvt0 * Theta0_vth * V0

  …

  # n-aggregator — use the n-specific Θ0 with the 0.5 factor per b4ld.c §1133-1154
  T0_th_n = 0.5 * dvt1 * Leff / lt1.clamp_min(1e-30)
  Theta0_n = _exp_threshold_branch(T0_th_n)

  tmp1 = epssub / Xdep
  tmp2 = nfactor * tmp1
  tmp3 = cdsc + cdscb * Vbseff + cdscd * Vds
  tmp4 = (tmp2 + tmp3 * Theta0_n + cit) / coxe
  n = 1.0 + torch.where(tmp4 >= -0.5, tmp4,
                        (1.0 + 3.0 * tmp4) / (3.0 + 8.0 * tmp4).clamp_min(1e-6))

- Also rename your original Theta0 to Theta0_vth where it is used in Delt_vth to avoid future confusion, and keep theta0vb0 (DIBL) as-is.

What you should see

- With the patch above, S at Vds=0.05 V should drop by the amount your gap shows, and n reported by DCResult should move from ≈1.23 toward the ngspice‑implied value. The Δlog10(Id) curve in z91m will flatten near deep‑subthreshold and approach zero faster as Vgs→Vth, matching the ngspice signature.

If after this change there is still a small residual difference (a few percent), the next two knobs to sanity‑check are:
- coxe vs coxp/coxe_eff in the denominator (should remain coxe here per b4ld.c), and
- that Xdep uses Phis (bias‑adjusted) not Phi (zero‑bias), which your code already does.

But the dominant 12% is the Θ0 factor-of-two in the n branch.
