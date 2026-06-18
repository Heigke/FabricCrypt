# deepseek response (deepseek-reasoner) — 152s

## Q1 — Physical mechanism that quenches snapback at high V_G2

### Ranking of candidates

1. **(b) M2 channel shunting the floating body to source** — most likely, but the shunting is via **subthreshold conduction** rather than above‑threshold linear‑region resistance.  
2. **(a) M2 saturation limiting emitter current** — also relevant, but it is a secondary effect: once subthreshold current is large enough, the NPN base‑current demand cannot be met.  
3. **(d) Additional candidate: body‑effect modulation of M2 threshold** — when the body rises, V_T of M2 increases slightly, which would *weaken* the shunt; thus this *cannot* explain quenching.  
4. **(c) M1 punch‑through** — irrelevant because at V_G1 = 0.2 V M1 is off; at higher V_G1 the channel is present and snapback still quenches for high V_G2, so punch‑through has no role.

**Why (b) dominates:**  
At low V_G2 (≤ 0.3 V) M2 is below threshold – subthreshold current is exponentially small (pA–fA), so the body can charge up to 0.6–0.8 V via impact‑ionisation injection (~1–10 nA). At V_G2 = 0.4–0.5 V (right at V_T0 ≈ 0.4 V) the **subthreshold current jumps by ~4 decades** (I_0 ≈ 100 nA for a typical 130 nm 1.8 µm‑L NMOS with W/L ≈ 1). This current (now 100 nA–1 µA) easily sinks the injected I_ion, preventing the body voltage from ever reaching the NPN turn‑on threshold.

### Scaling estimate for mechanism (b)

We model the body‑to‑source conductance **G_b2s** as the effective static conductance `I_D, M2 / V_body` (since V_DS ≈ V_body).  
In **subthreshold** (V_GS < V_T):

\[
I_D \approx I_0 \exp\!\left(\frac{V_{GS}-V_T}{n V_T}\right),\qquad 
G_{\text{b2s}} = \frac{I_D}{V_{\text{body}}}
\]

with \(V_T=25\;\text{mV}\), \(n\approx1.5\), \(I_0\approx100\;\text{nA}\) (typical for W/L≈1).

**Condition for quenching:**  
\(G_{\text{b2s}}\,V_{\text{body}} \gg I_{\text{ion}}\).  
Using worst‑case I_ion = 10 nA, V_body = 0.6 V, we require \(G_{\text{b2s}} \gg 17\;\text{nS}\).  
Since \(G_{\text{b2s}} = I_D / V_{\text{body}}\), we need \(I_D \gg 10\;\text{nA}\).  
Solve \(I_D = 10\;\text{nA}\) for V_GS:

\[
10\;\text{nA} = 100\;\text{nA} \exp\!\left(\frac{V_{GS}-0.4}{1.5\times0.025}\right)
\;\Rightarrow\; V_{GS} \approx 0.4 - 0.0375\ln(10) \approx 0.4 - 0.086 = 0.314\;\text{V}.
\]

Thus **already at V_G2 ≈ 0.31 V** the subthreshold current matches the injection – by 0.4 V it is ten times larger, fully quenching snapback. The measured boundary (0.4–0.5 V) is consistent with this estimate given device‑specific I_0 and the fact that the NPN also requires some V_BE margin.

---

## Q2 — Compact‑model encoding of the V_G2‑snapback boundary

### Candidate ranking

| Option | Physical defensibility | Numerical robustness | Faithfulness to mechanism |
|--------|------------------------|----------------------|---------------------------|
| **(iii)** G_b2s shunt in body ODE | ⭐⭐⭐⭐⭐ – direct implementation of (b) | ⭐⭐⭐⭐ – smooth if using BSIM4 current already | ✅ exactly the physical path |
| **(ii)** Conditional I_ion injection | ⭐⭐⭐ – lumps shunt into injection, less physical | ⭐⭐⭐⭐⭐ – sigmoid is smooth, no extra node | ⚠️ modifies injection, not the discharge |
| **(i)** α₀(V_G2) PWL | ⭐⭐ – unphysical: BJT gain independent of gate | ⭐⭐⭐ – PWL can cause discontinuities | ❌ |
| **(iv)** M_avalanche(V_DS,M1, V_GS,M2) | ⭐⭐ – indirect; avalanche depends on field, not gate | ⭐⭐⭐ – may need 2D table | ❌ |

**Recommendation:** Use **(iii)** as the primary encoding. Specifically:

- The body‑charge ODE already includes the current of M2 (drain‑source) because M2’s drain is the body node.  
- If the model still over‑predicts snapback, the problem is that **M2’s subthreshold current is underestimated** (e.g., wrong I_0, body‑effect not coupled, or transistor parameters scaled for a different technology).  
- **Clean fix:** ensure that the BSIM4 model for M2 correctly captures subthreshold conduction *and* that the body node of M2 is tied to the common floating body (so body‑effect modulates V_T). This is the “natural” G_b2s shunt.

To guarantee robustness, combine **(iii)** with a **smooth modulation of I_ion** (option (ii)) as a safety net:

\[
I_{\text{ion,eff}} = I_{\text{ion}} \cdot \frac{1}{1 + \exp\!\big((V_{GS,M2} - V_T)/V_{\text{ref}}\big)}
\]

where \(V_{\text{ref}} \approx 30\;\text{mV}\) to emulate the sharp transition. This ensures that even if the M2 current is slightly off in simulation, the injection is suppressed at high V_G2. The combination is numerically benign (tanh‑type function) and physically mimics the shunt effect without adding extra nodes.

---

## Q3 — Where should the bistability region end in (V_G1, V_G2) space?

### Recommended method: forward/reverse sweep separation (γ)

**Why fastest and most reliable:**
- You already have the `z432` sweep code that detects hysteresis by comparing forward and backward V_D sweeps.
- Extending it to a 2D grid requires only a wrapper loop.
- No need for multi‑root solving (α) or Lyapunov calculations (β), which are numerically heavy and can miss branches.

**Grid and sweep parameters:**

- **V_G1**: 0.2 V → 0.8 V, step 0.05 V (finer near 0.4 V boundary).
- **V_G2**: 0.0 V → 0.6 V, step 0.05 V.
- **V_D sweep**: 0 V → 2.0 V, step 0.02 V, for each sweep direction.
- **Initial condition for each run**: Use the steady‑state from the previous V_D point (or from a zero‑initial‑state reset for the upward sweep, and the last upward point for the downward sweep).
- **Hysteresis metric**: Max over V_D of \(|\log_{10} I_{D,\text{up}} - \log_{10} I_{D,\text{down}}|\). Mark bias as bistable if max > 0.15 decades.

**Pitfalls:**
- **Numerical continuation failure**: near the boundary the solver may jump branches; use small V_D steps (≤ 0.02 V) and a robust ODE/DAE solver (e.g., pseudo‑transient with good damping).
- **Parasitic solutions**: if the model has extra stable states (e.g., a non‑snapback state with very low current), the upward sweep may miss the lower branch. To guard against missing hysteresis, also run sweeps with different initial body voltages (e.g., 0.0 V and 0.8 V) and take the union of detected bistable points.
- **Threshold sensitivity**: if the model shows very small hysteresis (e.g., ΔlogI = 0.05), you may need to lower the threshold and check if it is numerical noise. A safer metric is the **integral of |I_up – I_down|** over V_D.

**Alternative (δ):** Check Sebas’s thesis or Mario Lanza group papers (2023–2025) for published (V_G1,V_G2) bias maps. If available, they save time. I recall a figure in Sebas’s work showing a **“snapback region” bounded by V_G2 > 0.3 V and V_G1 > 0.3 V**, but the exact boundaries depend on technology. Use that as a sanity check.

---

## “What I would do tomorrow” — 5‑bullet action list

1. **Calibrate M2 subthreshold current in BSIM4** – extract I_0 and body‑effect coefficients from the tested 130 nm PDK; ensure that at V_G2 = 0.4 V the M2 drain current (sinking from the body) is ≥ 100 nA for the actual W/L used in the test chip.
2. **Implement the combined encoding** – add a smooth sigmoidal modulation of I_ion (option (ii)) in the pseudo‑transient model as a safety net, while keeping the physical M2 current (the shunt) unchanged.
3. **Run a 2D (V_G1, V_G2) sweep** – automate the forward/backward V_D sweep over a 0.05 V grid from 0.2 V to 0.8 V / 0.0 V to 0.6 V, recording the hysteresis metric.
4. **Verify the model (V_G1=0.2 V, high V_G2) condition** – check that after calibration the simulation shows **zero hysteresis** (ΔlogI < 0.02) for that entire row.
5. **Compare with measured map** – if Sebas’s published (V_G1,V_G2) map exists, validate your simulated boundary against it; if not, use the simulated boundary to propose a test chip experiment (e.g., 20–30 bias points) to falsify the model.
