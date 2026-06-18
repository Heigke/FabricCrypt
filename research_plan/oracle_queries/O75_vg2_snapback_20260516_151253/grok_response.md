# grok response (grok-4-latest) — 14s

**Q1**  
The dominant quenching mechanism at high \(V_{G2}\) is **(b)**: strong inversion of M2 creates a low-resistance body-to-source shunt (\(G_{b2s}\)) that clamps \(V_{body}\) below the value needed for regenerative NPN turn-on.  

Mechanism (a) is secondary (it is a consequence of the same M2 turn-on). Mechanism (c) is unlikely because punch-through in M1 is primarily \(V_{D}\)-driven and shows weak direct \(V_{G2}\) dependence in 130 nm.  

Scaling estimate: set \(G_{b2s}\cdot\Delta V_{body}\gg I_{ion}\) with \(\Delta V_{body}\approx0.7\) V and \(I_{ion}\approx5\) nA \(\to G_{b2s}\gtrsim15\) nS. For \(L_{M2}=1.8\) µm and typical 130 nm \(\mu C_{ox}\approx200\) µA V\(^{-2}\), \(G_{b2s}\approx\mu C_{ox}(W/L)V_{ov}\) reaches 15 nS already at \(V_{ov}\approx50{-}80\) mV above \(V_T\) (i.e., \(V_{G2}\gtrsim0.45{-}0.48\) V). Thus (b) dominates immediately once M2 leaves subthreshold.

**Q2**  
Most physically defensible and numerically robust choice is **(iii)**: insert an explicit voltage-dependent shunt  
\[
I_{b2s}=G_{b2s}(V_{G2})\cdot(V_{body}-V_S),\qquad G_{b2s}=K\cdot\mu C_{ox}\frac{W}{L}(V_{G2}-V_T)_+
\]  
directly into the body-charge ODE. It is the direct compact-model transcription of (b), adds no extra knees, and preserves continuity of all derivatives.  

A lightweight combination with **(ii)** (sigmoid gating of \(I_{ion}\)) is acceptable if impact-ionization tables must remain untouched, but (iii) alone is sufficient and cleaner.

**Q3**  
Fastest reliable method is **(γ)**: perform forward/reverse \(V_D\) sweeps on a dense \((V_{G1},V_{G2})\) grid and record the locus where \(\Delta\log I_D<0.05\) dex. Grid: \(V_{G1}=0.1{-}0.7\) V (20 mV steps), \(V_{G2}=0.0{-}0.6\) V (20 mV steps), \(V_D\) ramp 0–2 V with 5 mV resolution and 1–2 ms/decade pseudo-transient time. Pitfalls to avoid: insufficient ramp time (artificial hysteresis) and solver tolerances that mask tiny \(\Delta I\).

**What I would do tomorrow**  
- Add the \(G_{b2s}(V_{G2})\) shunt term to the body ODE with \(K\) fitted to one measured high-\(V_{G2}\) curve.  
- Re-run the existing 18-bias z432 suite to verify snapback disappears only for \(V_{G2}>0.45\) V.  
- Sweep the \((V_{G1},V_{G2})\) grid with forward/reverse ramps and plot the resulting bistability boundary.  
- Check DC convergence at the boundary points with tightened tolerances and two different initial guesses.  
- If residual bistability persists, add a weak sigmoid gate on \(I_{ion}\) as a one-line patch and re-evaluate.

(The model is simply missing the dominant body-discharge path; once inserted, the “over-eager latching” disappears without global hacks.)
