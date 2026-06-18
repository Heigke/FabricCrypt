# openai response (gpt-5) — 349s

Short answer up front

- What makes the vertical Id step in LTspice: the classical floating–body positive‑feedback loop driven by BSIM4 impact‑ionization (Isub) in M1 and closed by the parasitic NPN (emitter at ground, base = floating body). As Vd is increased, Isub → Vb↑ → Vth(M1)↓ → Idsa↑ → Isub↑. When the small‑signal loop gain exceeds one the DC solution folds (snapback). No explicit “avalanche diode” is needed because BSIM4 §6.1 already supplies the avalanche/ionization source.

- Why your PyTorch port does not snap back: the ionization source in your port is too weak in the relevant bias range, so the loop gain never reaches unity. The culprit is the Isub magnitude: with the PTM130 numbers (alpha0 ≈ 7.8e‑5, beta0 ≈ 18–20, L ≈ 0.18 µm) your current compute_iimpact produces Isub that is orders of magnitude smaller than LTspice’s BSIM4 (nA–µA there vs pA–fA in your logs). That keeps dIsub/dVb and dIsub/dVs too small, so det(J) never vanishes and the arclength solver quite correctly reports no folds.

- The near‑vertical step in the measurement and in LTspice is not coming from PSCBE or junction reverse‑BV; those shape rout/CLM and diode leakage but do not create the fold by themselves. It is also not a solver artifact in your run: with the present Isub level the 2×2 Jacobian never becomes singular, so there is only one branch to trace.

1) Mechanism in one paragraph

In M1, BSIM4’s impact‑ionization current Isub ≈ K(L)·(Vds−Vdseff)·exp(−β0/(Vds−Vdseff))·(Idsa·Vdseff) grows rapidly once Vds exceeds Vdseff. Those holes charge the floating body, raising Vb. Through the body‑effect Vth(Vb) decreases, Idsa increases, and so does Isub (regeneration). In parallel the vertical parasitic NPN with emitter at ground turns on when Vb ≳ 0.6–0.7 V; its collector current adds to the drain current seen at the D pin and its base current is a strong sink from the body. The DC operating point is the simultaneous solution of the Sint‑KCL and Body‑KCL. As Vd is swept up, the small‑signal feedback factor around the body loop exceeds unity; the body equation loses monotonicity (∂R_B/∂Vb → 0), det(∂(R_S,R_B)/∂(Vs,Vb)) = 0, and the I–V folds. The jump is the transition from the low‑Vb/low‑Isub fixed point to the high‑Vb/BJT‑on point.

2) Loop‑gain to instrument

Let R_S(Vs,Vb;Vd)=0 and R_B(Vs,Vb;Vd)=0 be your two KCLs (you already have them). The fold occurs when the 2×2 Jacobian J wrt (Vs,Vb) becomes singular. A convenient scalar “loop‑gain” that goes to 1 at the fold is

- Define a = ∂R_S/∂Vs, b = ∂R_S/∂Vb, c = ∂R_B/∂Vs, d = ∂R_B/∂Vb.
- The effective body slope after eliminating Vs by R_S=0 is d_eff = d − c·b/a.
- Fold condition: d_eff = 0 ⇔ det(J) = a·d − b·c = 0.
- Loop‑gain: L ≡ (c·b)/(a·d). The snapback threshold is L → 1 (with the usual sign pattern near break: a<0, b>0, c<0, d<0 so L is positive).

Log, in addition to L and det(J),
- decomposition of d into sinks and sources:
  d = (+∂Isub/∂Vb + ∂Igidl/∂Vb + ∂Igb/∂Vb) − (∂Ibs/∂Vb + ∂Ibd/∂Vb + ∂Ibjt/∂Vb),
- and of c and b:
  c ≈ ∂Isub/∂Vs + (smaller gidl/gisl terms),
  b ≈ gmb(M1) + ∂Ibs_M1/∂Vb + ∂Ibd_M2/∂Vb.

Typical magnitudes at the fold in 130 nm DNW‑FB (from LTspice): a ≈ −(10–100) µS, b ≈ +5–50 µS, c ≈ −2…−10 µS, d ≈ −5…−50 µS; L crosses 1 as c and/or b rise with Vd due to Isub growth.

3) One decisive experiment (≤1 day)

Goal: decide “missing physics vs missing coupling vs solver.”

- Step 1 (physics amplitude test; no change to topology or solver):
  In your port, force the Isub strength up only in M1 by either
  (A) setting beta0 → 1.5–2 (env var NSRAM_BETA0_TEST you already have), or
  (B) multiplying alpha0 by 50–100 (you also have NSRAM_A0_MULT).
  Run a single snapback curve, log L(Vd), det(J), Vb(Vd), Isub_M1(Vd), Ic_BJT(Vd).
  Expectation:
  - If a fold appears (det(J)→0, L→1, Vb jumps, Ic jumps), the solver was fine; your baseline Isub was simply too small. You now know to look at the IIMOD branch and size/bias scaling of alpha0/beta0 (see note below).
  - If there is still no fold, go to Step 2.

- Step 2 (coupling test):
  Add a modest explicit series resistance from the body to the M1 source region (a few kΩ; this emulates BSIM4’s RBODY ladder that is inactive with rbodymod=0) or, equivalently, enable the existing rbpb/rbps/rbpd ladder if you have it. Repeat Step 1 with the original alpha0/beta0. If a fold now appears, the missing piece was the internal body resistance distribution; keep it (that is what provides the “Rsub” every ESD/snapback model needs).

- Cross‑check in LTspice at the same bias: plot V(b), I(M1.dii) or @M1[Isub], and I(Q1). You should see Isub rise into the nA–µA range right at the measured jump; that is the magnitude you must reproduce in the port for the loop to reach unity.

4) Ranking of plausible mechanisms

Most likely → least:

1) Underestimated impact‑ionization in the port (highest).
   Evidence: with PTM130 numbers and your current compute_iimpact, T2 ≈ α0/Leff ≈ 4×10^2 and exp(−β0/Δ) at Δ≈0.6–1 V is 10^−8…10^−9, so Isub ends up pA–fA for Idsa·Vdseff ≈ µA·V — far below what is needed to lift Vb. LTspice produces nA–µA at the same corner. This discrepancy is consistent with an IIMOD branch/scaling difference (legacy vs “new” form, or missing per‑instance binning of α0/β0, or a 1/L vs L factor). Your own α0×10/β0 override already moves the curves in the right direction; pushing it further should trigger the fold immediately if everything else is wired correctly.

2) Missing body resistance distribution (RBODYMOD) (medium).
   A single equipotential body node makes the loop harder to trigger because the strong diode slopes (∂Ibs/∂Vb, ∂Ibd/∂Vb) clamp Vb uniformly. The distributed RBODY ladder in BSIM4 (rbpb/rbps/rbpd) provides the “Rsub” that most snapback macro‑models require. If your LTspice run has rbodymod=1 while the port uses rbodymod=0, this alone can explain the qualitative difference. If both use rbodymod=0, this item is less likely.

3) Solver false negative (low).
   With L well below 1 everywhere (current situation) there is only one branch; arclength won’t find a fold because none exists. If you force Isub up (Step 1) and the branch appears, your solver will find it (your augmented 3‑state system and tangent‑flip logic are the right tool). If you still worry, halve ds_max and run a backward trace from Vd_max; genuine snapback shows clear hysteresis and det(J) sign flips.

Notes on the Isub implementation details to check

- IIMOD branch and size binning. The C code forms the effective ionization prefactor with instance‑level binning; the commonly used legacy form is T2 = (alpha0 + alpha1·Leff)/Leff, but cards are binned and sometimes multiplied by NF/Weff. Make sure you read α0/α1 from sd.scaled (post‑binning) and that the Leff normalization matches the branch used by LTspice for your card.

- diffVds definition. Use diffVds = Vds − Vdseff (never negative), exactly as in b4ld.c, and the pre‑SCBE Idsa·Vdseff (you have already fixed this; keep it).

- DNW→body path. It biases the operating point (raises Vb baseline) but does not create the fold by itself; treat it as a DC source into the body, not as the root cause of the step.

Deliverables you can add to your logger now

- det(J) and L = (c·b)/(a·d) vs Vd for each curve.
- The four Jacobian elements and their decomposition into physical terms:
  a: gds contributions in R_S, b: gmb + junction terms in R_S,
  c: dIsub/dVs + gidl/gisl parts, d: +dIsub/dVb − (diode slopes + dIbjt/dVb).
- Isub_M1, Vb, Ic_BJT, Ibs_M1, Ibd_M1 vs Vd for the snapback corner.

What to expect when it’s fixed

Once Isub is brought to the LTspice level (or RBODY is added), L will cross 1 around Vd ≈ 1–1.5 V, det(J) will pass through zero, the arclength trace will rotate (t[2] sign flip) and you will see two Vb solutions at the same Vd — the measured near‑vertical Id jump.
