# openai response (gpt-5) — 257s

Q1 — Falsification of S1 (phantom vs real 2D root)
Single 5‑minute diagnostic: evaluate the full 2D residual and Jacobian at the S1 point and report their norms.

How:
- Freeze biases: VG1=0.6, VG2=0.2, Vd=1.5 V.
- Use Vsint=Vsint_S1(Vb=0.8) from z377 and Vb=0.8.
- Call the exact same residual function used by the 2D solver (nsram_cell_2T._residuals or equivalent) to get R = [R_S, R_B] and J = ∂R/∂[Vsint,Vb].
- Decide:
  - Real 2D fixed point if max(|R_S|,|R_B|) < 1e-9 A (or your solver tol) AND cond(J) < 1e9.
  - Phantom if |R_B| is large (e.g., >1e-6 A) or J is near‑singular.
Why decisive: S1 only zeroed R_S with Vb pinned. If R_B≠0 at (Vsint_S1, 0.8 V), it is not a root of the 2D system. Numbers to beat: at 1.5 V, S1 reports Ids(Vb=0)=~3e-12 A vs Ids(0.8)=~1.05e-6 A (5.5 dec); z378 shows arc-length returns the low‑Vb branch everywhere (33/33, jumps <1e-5 dec). The residual test tells you unequivocally which story is true.

Q2 — Method ranking on R‑46
Most likely to succeed:
1) S2b two‑branch search (hot Vb=0.8 init): Highest chance to land on a disconnected high‑Vb basin if it exists. Fast “yes/no.” If it converges to high Ids at 1.5 V, S1 was real; if it falls back to low‑Vb, S1 was phantom. Minimal code changes.
2) S2a iii_gain homotopy: Good for reconnecting basins through a physics path. Start high iii_gain (e.g., 10×), continue down to 0.9036 (your fit) with pseudo‑arclength in iii, not Vd. If the high‑Vb branch needs avalanche to appear, this can bridge it.
3) S2c pseudo‑transient (BE on Vb, Cb=8 fF): Can find stable attractors the DC Newton misses, but success depends on artificial Cb/time‑scale and may still track the wrong stable branch. Slower and more parameters to tune.

Obvious techniques you’re missing:
- Deflation: solve once (low branch), deflate, solve again to force the second root.
- Series‑R homotopy: add small Rd (10–500 Ω) in D path, continue Rd→0; mimics bench load‑line, often reveals snapback.
- gmin stepping focused on body/diodes; source stepping in Vb.
- Pseudo‑arclength in Vb (not Vd).
- Trust‑region/dogleg Newton near near‑singular J.

Q3 — Meta‑critique (are we modeling the wrong thing?)
- Well‑posedness: Near avalanche/BJT turn‑on, dI/dV can explode and J becomes near‑singular. That yields huge Ids sensitivity without a true root if body KCL isn’t met. Given z378 found no folds (33/33 biases, arc‑length identical to cold Newton, max jump <1e-5 dec), your DC algebraic system likely has a single reachable branch under ideal Vd bias. If R_B at S1’s point is large, S1’s 5.5 dec is a phantom of the 1D constraint.
- Topology sufficiency: A 2‑node lumped “BSIM4 + GP NPN” without distributed well/base resistances and self‑heating rarely produces a 2–3 decade DC snapback. Seven successive R‑phase tweaks (R‑43…R‑55a) failed; even Bf→1e4 didn’t help. You are likely missing:
  - Distributed substrate/well sheet resistances (lateral base spreading).
  - Self‑heating/thermal feedback (dominant in snapback).
  - Realistic bench load‑line (finite series R/compliance).
- Methodology: Your “DC snapback” dataset is slow DC (≈290 ms/step), not TLP. Industry standard characterizes snapback with pulsed (TLP/CDM/HBM) and includes thermal dynamics. Recommendation:
  - Add a compact thermal RC and avalanche state; validate on TLP‑like transients (extract Vt1/Ih).
  - Include distributed Rnet for body/well; or emulate with a small series Rd and continue Rd→0.
  - If DC must be used, solve the load‑line (Vsource + Rs) system, not an ideal voltage source at D. Without this, the mathematical DC problem may have no fold to find.
