# R-17: Deep Critique Synthesis (3-way oracle)

Source: `research_plan/oracle_queries/O62_deep_critique/{openai,gemini,grok}_response.md`
Date: 2026-05-13

## Convergence matrix

| Q | OpenAI (3 sub-oracles) | Gemini (3 sub-oracles) | Grok (3 sub-oracles) | Verdict |
|---|---|---|---|---|
| Q1 root vs symptom | **Symptom** (all 3) — root = D1 (Q1.E miswire) + bad KCL/sign in R_B/R_Sint | **Symptom** (all 3) — root = broken BJT feedback loop (mathematical / physical / numerical framings) | **Symptom** (all 3) — root = D1 + D2 overcounting | **UNANIMOUS 9/9: basin-lock is a SYMPTOM. Root cause is the D1 miswire breaking the Vb→Vsint BJT feedback path.** |
| Q2 missing element | **M2 body handling** — M2.B=0 in LTSpice but pyport residuals still reference M2 body diodes against Vb; also sign of M2 body→drain term into Sint | Subtle: **KCL formulation** of M2 body-drain (should be f(Vb−Vsint), not f(Vb)); possible double-count of single junction | **M2.B handling** + missing CBpar (1fF B→GND) — possible transient grounding path pyport ignores | **CONVERGE on M2 body topology**: pyport residual still treats M2 body as if floating/tied to Vb. Must enforce M2.B=GND and strip all M2 body terms from R_B. Also reconsider sign/arg of M2 body-drain term in R_Sint. |
| Q3 bimodal | Two solution families: A (BJT inactive accidentally OK ~1 dec) vs B (BJT supposed to fire, broken → ≥3 dec); K1 NaN = branch flip | Bifurcation pre-snapback vs failed-snapback; or ill-conditioned Jacobian near correct basin | Regime-specific: low VG1 = errors small, high VG1 = snapback engages and exposes D1/D2 | **UNANIMOUS: bimodality = whether parasitic BJT/snapback regime is engaged. Low-error cluster is "lucky" coincidence; high-error cluster is the actual physics the model can't represent.** |
| Q4 R-13 alone | 2.6 ± 0.4 / 2.4–2.8 / 2.7 ± 0.3 dec | 2.5 ± 0.5 / 1.8 ± 0.6 / >3.0 dec | 1.2 ± 0.3 / 1.5 ± 0.4 / 1.0 ± 0.2 dec | **Wide spread: Grok optimistic (~1.2), OpenAI/Gemini pessimistic (~2.5–2.7).** Consensus: Vb-free solver ALONE will NOT achieve sub-1.0 dec. Need topology fixes too. With D1+D2+M2.B=0 fixes: all converge to **≈1.1–1.4 dec median**. |
| Q5 discriminator | (a) Mutate ngspice with Q1.E→GND, re-run snapback rows; (b) pyport residual-at-ngspice-OP dump | (a) Plug ngspice OP into pyport residual, expect huge nonzero R; (b) source-injection sweep at body; (c) one Newton step from ngspice OP — predict huge Δx | (a) ngspice with Q1.E→GND mimicked; (b) toggle use_well_diode in pyport; (c) force pyport init at ngspice OP | **STRONG CONVERGENCE on the same discriminator (5 of 9 sub-oracles):** Take ngspice operating point at known-failing bias (VG1=0.6, VG2=0.2, Vd=2.0, where Vsint≈0.382, Vb≈0.267), plug DIRECTLY into pyport residual function, print every term of R_Sint and R_B. If residuals ≈ 0 → basin-lock alone. If residuals are huge (predicted) → structural KCL error proven. |

## Structural element flagged that we may have missed

**M2 body coupling.** All three oracles independently flag this beyond the known D1/D2/D9:

- LTSpice has M2.B unconnected (defaults to GND).
- Pyport's residual `R_B` subtracts `Ibd_M2(Vb)` and `Ibs_M2(Vb)` — referencing the floating body Vb.
- Pyport's residual `R_Sint` includes an M2 body-drain diode term that may use wrong sign and/or wrong argument (should be Vb−Vsint referenced to M2.B=0, not Vb alone).

Net effect: phantom leakage shunt from B through M2's nonexistent body connection, which pulls Vb up and prevents the physical Vb≈0.27 basin from being a fixed point.

Action item: in `nsram_cell_2T.py`, when `m2_body_gnd=True`, strip ALL M2 body-diode contributions from R_B, and reformulate the M2 body-drain term in R_Sint with M2.B=0.

Secondary flag (Gemini/Grok): CBpar 1fF B→GND capacitor in LTSpice — DC-inactive but indicates intended body-to-ground reference. Confirms M2.B=GND interpretation.

## The single discriminating experiment (TONIGHT)

**Residual-at-ngspice-OP dump** (5 of 9 sub-oracles propose this exact test):

1. Take ngspice converged operating point at one known-failing bias: **VG1=0.6, VG2=0.2, Vd=2.0 → Vsint=0.382, Vb=0.267**.
2. In pyport, call the residual function with `(Vsint=0.382, Vb=0.267)` directly. **Do not solve.**
3. Print every component of R_Sint and R_B individually: Ids_M1, Ids_M2, Ie_Q1, Ib_Q1, Iii, Ibs_M1, Ibd_M1, Ibs_M2, Ibd_M2, well-diode currents.
4. Sum the residuals.

**Decision rule:**
- |R_Sint| + |R_B| < 1e-9 A → basin-lock alone. R-13 Vb-free Newton fixes everything.
- |R_Sint| + |R_B| > 1e-6 A → structural KCL error confirmed. R-13 will not be sufficient; must also fix D1, D2, and M2 body handling before claiming any improvement.

Bonus (Gemini Oracle 3): also compute one Newton step `Δx = -J⁻¹ F` at that point. If `‖Δx‖` is large AND points away from the true solution, the Jacobian itself is broken (off-diagonal coupling missing because of D1).

## Recommended sequencing

1. **Tonight (1h)**: residual-at-ngspice-OP dump (the discriminator above). Cost: ~30 lines of Python.
2. **If structural error confirmed** (predicted by all 3 oracles): fix D1 (Q1.E → Sint) + strip D2 + enforce M2.B=GND in residuals BEFORE running R-13 Vb-free solver. Otherwise R-13 will land in the broken-physics solution more robustly and we'll waste a debug cycle thinking we're making progress.
3. **Target median dec after combined fixes**: 1.1–1.4 (consensus of all 9 sub-oracles).
