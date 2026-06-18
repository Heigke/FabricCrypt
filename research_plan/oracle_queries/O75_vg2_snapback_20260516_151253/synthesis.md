# O75 Synthesis — VG2-dependence of snapback in NS-RAM 2T cells

**Date**: 2026-05-16  
**Packet**: `research_plan/oracle_queries/O75_vg2_snapback_20260516_151253/`  
**Oracles**: openai (gpt-5, 160s), gemini (gemini-2.5-pro, 67s), grok (grok-4-latest, 14s), deepseek (deepseek-reasoner, 152s)  
**Wall**: ~7 min total.

---

## Cross-tabulation

### Q1 — Physical mechanism that quenches snapback at high V_G2

| Oracle | Primary | Secondary | Rejected |
|--------|---------|-----------|----------|
| openai | **(b) M2 B→S shunt** | (d) voltage partition (M2 steals V_D from M1, reduces I_ion) | (a) is a "red herring" — region label irrelevant, current magnitude is what matters; (c) not VG2-correlated |
| gemini | **(b) M2 B→S shunt** | (d) body-effect on M1 V_T (secondary feedback) — but consequence of (b) | (a) flawed premise — at high VG2 & high V_body M2 is in saturation but the *physics* is "conducts strongly," not "saturates"; (c) only weak 2nd-order on VG2 |
| grok | **(b) M2 B→S shunt** | (a) secondary, consequence of M2 turn-on | (c) very weak VG2 dependence |
| deepseek | **(b) M2 B→S shunt — via SUBTHRESHOLD conduction** | (a) secondary — base-current demand limit | (c) irrelevant; (d') body-effect of M2 itself: *weakens* shunt, can't be cause |

**Consensus**: 4/4 → **(b) M2 channel shunting body→source is the quenching mechanism.** All four explicitly reject (a) as a misframing and (c) as unrelated to VG2.

### VG2 threshold for quenching (key numerical question)

| Oracle | Estimate | Reasoning |
|--------|----------|-----------|
| openai | V_G2 ≈ **0.42–0.45 V** (V_ov 20–30 mV above V_T2 = 0.40 V) | Saturation-limited: I_sat = ½β·V_ov² ≥ 10 nA ⇒ V_ov ≈ 23 mV for β=37 µA/V² |
| gemini | V_G2 ≈ **0.4 V (= V_T2)** transitions sharply | At V_G2=0.5V M2 sinks 1.25 µA = 100× I_ion |
| grok | V_G2 ≈ **0.45–0.48 V** | G_b2s = µCox·(W/L)·V_ov, need ≥ 15 nS ⇒ V_ov ≈ 50–80 mV |
| deepseek | V_G2 ≈ **0.31 V** (already at subthreshold!) | Critical insight: M2 *subthreshold* I_D ≈ 10 nA at V_GS − V_T = −86 mV (with n=1.5, I_0=100 nA) |

**Range: 0.31–0.48 V.** Median ≈ 0.40 V, matching measured boundary. **Deepseek is the outlier worth noting**: argues that subthreshold conduction alone explains the boundary — implies your model's M2 may be missing subthreshold I_D, not just strong-inversion conduction.

### Q2 — Compact-model encoding

| Oracle | Pick | Stance on alternatives |
|--------|------|------------------------|
| openai | **(iii) full MOS-like B→S branch using reduced BSIM core** (Ids(V_gs=VG2, V_ds=V_BS)) | (i)/(iv) unjustified, hide physics; (ii) "brittle hack" with artificial knees |
| gemini | **(iii) pure** — "do not combine, just fix KCL" — strong claim: model is **missing I_DS,M2 in body KCL** | (i)/(ii)/(iv) "fundamentally incorrect" |
| grok | **(iii) primary**, (ii) optional safety net | OK to combine if needed |
| deepseek | **(iii) primary + (ii) sigmoid as safety net** (V_ref ≈ 30 mV) | (i)/(iv) unphysical |

**Consensus**: 4/4 → **(iii) C_b discharge via M2 — full BSIM-style channel current from body node to source.** Most likely diagnosis (gemini, openai, deepseek): your body-node KCL is **missing or under-counting I_DS,M2 with drain = V_body**.

Split 2-2 on whether to also add (ii) sigmoid I_ion gate as numerical safety net. Openai+gemini reject (ii); grok+deepseek accept it as belt-and-braces.

### Q3 — Bistability boundary determination

| Oracle | Pick | Grid |
|--------|------|------|
| openai | **(α) 1D fixed-point analysis on body-node ODE** (F(V_b)=0, sign-changes, dF/dV_b for stability) — rigorous + cheap | V_G1: 0.10–0.70 V / 25 mV; V_G2: 0.00–0.60 V / 25 mV; V_D 0.6–1.5 V / 25–50 mV |
| gemini | **(γ) Forward/reverse sweep**, you already have it, just systematize | V_G1: 0.2–0.7 / 0.05; V_G2: 0.0–0.6 / 0.05 |
| grok | **(γ)** sweep | V_G1: 0.1–0.7 / 0.02 V; V_G2: 0.0–0.6 / 0.02 V; V_D 0–2 V / 5 mV; threshold ΔlogI < 0.05 dex |
| deepseek | **(γ)** sweep + multiple V_b initial seeds (0.0, 0.8) | V_G1: 0.2–0.8 / 0.05; V_G2: 0.0–0.6 / 0.05; V_D 0–2 / 0.02; threshold > 0.15 dex |

**Consensus**: 3/4 for (γ) sweep (gemini, grok, deepseek — pragmatic). 1/4 (openai) prefers (α) fixed-point analysis as rigorous + cheap.  
**Best-of-both**: do (γ) first (fast, already implemented) → identify boundary candidates → do (α) at the boundary to confirm saddle-node bifurcation and extract unstable branch.

On Q3-δ (Mario/Sebas published bias-map): **none of the four oracles confirm a published bias-map exists.** Deepseek "recalls" one bounded by V_G2 > 0.3 V and V_G1 > 0.3 V but this is unverified hallucination — should not be trusted without checking Sebas's thesis directly.

---

## Strong consensus (4/4)

1. **Mechanism (b) M2 body-to-source shunt is the quencher.** Confidence: very high.
2. **The model fix is option (iii)**: include the full BSIM I_DS,M2 current as a sink in the body-node KCL with drain=V_body, gate=V_G2, source=0.
3. **The most likely concrete bug**: model's body-node KCL either omits I_DS,M2 entirely, or M2 is parameterised without subthreshold conduction, or W/L of M2 is wrong, or M2's drain is not tied to V_body in the netlist.
4. **(a), (c), (i), (iv) are all wrong/misleading framings.** Drop them from consideration.

## Notable disagreements

1. **Subthreshold vs strong-inversion onset of quenching.** Deepseek says quenching already begins at V_G2 ≈ 0.31 V via subthreshold. Others (openai, gemini, grok) put threshold at V_G2 ≈ V_T2 ≈ 0.40–0.48 V via above-threshold conduction. **This is testable**: if your model's M2 has only strong-inversion current (square-law) and no subthreshold tail, deepseek's prediction wins → check whether your BSIM core implements subthreshold region properly.
2. **Whether to add sigmoid I_ion safety net (option ii).** 2-2 split. The principled view (openai, gemini) is: if (iii) is implemented correctly, (ii) is unnecessary and dangerous (introduces artificial knees). The pragmatic view (grok, deepseek): keep (ii) as belt-and-braces but with smooth (tanh) form to avoid knees.

## Dispatch-worthy ideas (new angles beyond what we already considered)

1. **(openai novel)** "Voltage partition" mechanism (d-openai): when M2 conducts strongly it absorbs more of V_D, reducing V_DS,M1 and therefore the avalanche multiplier M(V_DS,M1). This is a *second, automatic* quenching mechanism that emerges for free if your netlist topology is right — and *should not* be implemented as a separate VG2-dependence of M_avalanche (which would be option iv, wrong). Action: verify that V_DS,M1 self-consistently drops at high V_G2 in your simulation.
2. **(deepseek novel)** **Subthreshold dominates quenching** — quenching starts at V_G2 ≈ 0.31 V via subthreshold M2 conduction, not at V_T2. If true, your M2 model must implement n·V_T·ln subthreshold properly. Action: log I_DS,M2(V_GS,M2) at V_DS,M2=0.7 V across V_G2 = 0.0–0.6 V on log scale, confirm exponential subthreshold tail with n ≈ 1.3–1.5.
3. **(openai novel)** Use **1D reduced fixed-point analysis** for bistability: treat V_b as the only continuation variable, solve rest of the network quasi-DC at each trial V_b, evaluate F(V_b) = body-net current. Bistability = 3 zero crossings with sign of dF/dV_b alternating. Much cleaner than relying on hysteresis to *probe* bistability.
4. **(gemini novel)** Body-effect on M1's V_T as **secondary positive feedback**: a charging body lowers V_T1, raising I_DS,M1 and thus I_ion. Clamping V_body via (b) also kills this secondary feedback. Action: confirm your M1 BSIM4 model has body-effect (γ_b parameter) properly tied to V_body node.
5. **(grok novel)** Quick check: at V_G2 = 0.45 V the conductance estimate gives G_b2s ≈ 15 nS. If your model has G_b2s ≪ 15 nS at this bias even after fix attempts, the issue is W/L_M2 underestimate or wrong µCox in your PDK.

## Recommended action plan (synthesis of all four)

1. **Diagnose first, fix second.** Before changing anything, instrument the body-node KCL: dump every current term (I_ion, I_recomb, I_DS,M2, I_NPN_collector_into_body, junction leakage) at V_G1=0.2, V_G2=0.5, V_D=1.0 V. The dominant deficit should be I_DS,M2.
2. **Verify M2 model**: 
   - Confirm M2 drain is netlist-tied to V_body node (not to a different rail).
   - Plot I_DS,M2(V_G2) at V_DS=0.7 V on log scale across V_G2 ∈ [0, 0.6 V]. Should show clean subthreshold + linear + saturation.
   - If subthreshold tail is missing or n is wrong (too steep), fix it. Deepseek estimates n ≈ 1.5, I_0 ≈ 100 nA for W/L ≈ 1.
3. **Implement (iii) cleanly**: full BSIM I_DS branch from V_body to V_source, **no extra knees**, no hand-tuned sigmoids first. Re-run z432 single bias (V_G1=0.2, V_G2=0.5) — snapback should vanish.
4. **Run the 2D map**: (γ) forward/reverse V_D sweep on a (V_G1, V_G2) grid of ≈ 13×13 = 169 points. Hysteresis metric max|Δ logI_D|, threshold ≈ 0.10 dex. Plot as a 2D heatmap. Expect the bistable region to shrink to V_G1 ∈ [0.35, 0.65] × V_G2 ∈ [0.0, 0.35].
5. **Confirm with (α) 1D fixed-point** at ~10 boundary biases: evaluate F(V_b) on V_b ∈ [−0.1, 0.9], count sign-changes, check dF/dV_b. Saddle-node bifurcations should map onto the (γ)-detected boundary.
6. **Optional safety net**: if residual bistability persists at clearly-wrong biases, add a *smooth* tanh-gated I_ion suppression (deepseek's I_ion·1/(1+exp((V_GS,M2 − V_T2)/30 mV))) but mark it explicitly as an empirical correction, not physics.

## Open questions not answered by any oracle

- **What is the actual W/L_M2 in Sebas's test chip?** We assumed 1.8 µm L and W ≈ 0.3–1.8 µm; oracles' answers vary by factor ≈ 6 in β depending on W. Need the layout/PDK value.
- **Is Mario/Sebas's published (V_G1, V_G2) bistability bias-map a real thing?** Deepseek claims yes but cannot cite. Need direct check of the thesis / Lanza group papers — **do NOT trust the oracle on this**.
- **Does the deep N-well / NPN parasitic capacitance C_b also need a fast RC term from M2?** None addressed whether the *capacitive* discharge timescale matters for hysteresis width — only DC clamping.
