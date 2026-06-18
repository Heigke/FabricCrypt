# O75 — VG2-dependence of snapback in NS-RAM 2T cells

## Context

We are modelling Sebas's NS-RAM 2T cell (TSMC 130nm, M1+M2 series with floating body, parasitic vertical NPN through deep N-well). The cell uses M1 as the access transistor and M2 as a body-control transistor; the floating body charges/discharges via M2 channel + impact ionisation in M1.

Measured I_D(V_D) characteristics (Sebas's silicon, 130nm) show a clear **VG2-dependent snapback region**:

| VG1 | VG2 range | Behaviour at V_D ≈ 0.8–1.5 V |
|-----|-----------|------------------------------|
| 0.4 / 0.6 V | low (0.0–0.3 V) | **Clear snapback knee**, current rises 2–3 decades to ~µA |
| 0.4 / 0.6 V | high (0.4–0.5 V) | **Smooth monotonic rise**, no sharp knee |
| 0.2 V | all VG2 | Always smooth, never reaches µA, never snapbacks visibly |

So the snapback region is bounded **both above (by VG2)** and **below (by VG1)**.

### Our model status

We have a pseudo-transient model (Mario/Sebas-style compact model with BSIM4 channel + Gummel-Poon NPN + body-charge ODE + impact-ionisation injection) that:

- ✅ Reproduces the right snapback shape at VG1=0.4/0.6 low-VG2
- ✅ Has bistability — z432 hysteresis sweep: 0.45 dec mean ΔlogI across 18 biases (forward vs reverse sweep)
- ❌ **Wrongly produces snapback / bistability at VG1=0.2 high-VG2** (where measured shows none)

The bistability lives at ALL biases, not just where measured shows it. The model is over-eager to latch.

## Three questions

### Q1 — Physical mechanism that quenches snapback at high V_G2

In a 2T NS-RAM cell (vertical NPN through deep N-well), what is the PHYSICAL mechanism that QUENCHES snapback at high V_G2?

Candidate mechanisms — please rank by likelihood and propose a 4th if missing:

(a) **M2 saturating below V_GS,M2 − V_T limit** → emitter current saturates → α_NPN × M_avalanche product drops below 1, breaking the positive feedback.

(b) **M2 channel becoming low-resistance** → shunts the floating body to source → body potential clamped → no V_BE buildup → NPN never turns on hard.

(c) **M1 punch-through suppressing impact ionisation** at high V_DS,M1 because depletion region merges before high-field ionisation builds up.

(d) Other? (your candidate)

For TSMC 130nm with L_M2 = 1.8 µm, V_T0 ≈ 0.4 V, please give a **specific scaling estimate**: at what V_GS,M2 does mechanism (b) start dominating? Approximate the body-to-source conductance G_b2s as a function of V_GS,M2 and identify the threshold where G_b2s × (V_body−V_source) >> I_ion (impact-ionisation injection current). Use rough numbers: I_ion ≈ 1–10 nA, body capacitance C_b ≈ 1 fF, typical V_body swing 0.6–0.8 V.

### Q2 — Compact-model encoding of the VG2-snapback boundary

How would Mario/Sebas-style compact models encode this VG2-dependent snapback boundary CLEANLY (not via global hacks)? Candidates:

(i) **Per-bias α0(V_G2) PWL** — make BJT current gain depend on VG2. Physical motivation?

(ii) **Conditional Iion injection** — only inject impact-ionisation current into the body node when M2 is below saturation (V_DS,M2 < V_GS,M2 − V_T2). Specifically: I_ion → I_ion · sigmoid((V_GS,M2 − V_T2 − V_DS,M2)/V_thermal). 

(iii) **Cb fast-discharge via M2** — when M2 conducts strongly, add an explicit G_b2s · (V_body − V_source) shunt term in the body-charge ODE with G_b2s ∝ µCox·(W/L)_M2·(V_GS,M2 − V_T2). This is the *natural* implementation of mechanism (b).

(iv) **M_avalanche(V_DS,M1, V_GS,M2)** — make the avalanche multiplier itself depend on V_GS,M2 through some 2D mechanism.

Which of (i)–(iv) is most physically defensible and most numerically robust (no extra knee discontinuity, no convergence trouble)? Would you combine two of them?

### Q3 — Where SHOULD the bistability region end in (V_G1, V_G2) space?

We need an EXPERIMENT (numerical) that determines the bistability boundary. Candidates:

(α) **Eigenvalue analysis at multiple roots** — for each (V_G1, V_G2), find all DC solutions of the body-node KCL, compute Jacobian at each, classify saddle/node. Bistable iff at least 2 stable + 1 unstable coexist.

(β) **Lyapunov-style trajectory stability** — perturb initial conditions and measure decay/growth of trajectory separation.

(γ) **Forward/reverse sweep separation** — what we currently do (z432), but explicitly trace the locus where Δ log I → 0.

(δ) **Mario's slide 20 / paper figure** — does anyone recall if a (V_G1, V_G2) bias-map of the bistable region is published in Mario / Sebas's work? Specifically: Mario Lanza group, NS-RAM 2T compact-model papers ~2023-2025, or Sebas's thesis.

Recommend which of (α)–(δ) is fastest to implement reliably and what (V_G1, V_G2) grid + sweep parameters you'd use. Mention any pitfalls (numerical continuation failures, parasitic solutions, etc.).

## Deliverable

For each question give a concrete, actionable answer. Cite mechanisms by name where possible. If you disagree with one of our candidate options, say so explicitly — we want falsification, not agreement.

End with a 5-bullet "what I would do tomorrow" action list.
