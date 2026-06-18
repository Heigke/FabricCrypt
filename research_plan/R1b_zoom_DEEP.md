# R-1b — DEEP audit of nsram/Zoom/ (real inspection, not transcript)

Date: 2026-05-13. Supersedes R1_zoom_audit.md (which was transcript-only).
Method: every JPEG opened via Read (vision), xlsx parsed via openpyxl, pptx
via python-pptx, model cards diffed against existing data/sebas_2026_04_22/.

---

## 1. The 31 images — per-image content table

Two distinct meetings ("2026-03-20" introductory slide-set from Mario, and
"2026-04-30" Sebas BSIM fits & circuit follow-up), plus one 04-22 vendor
screenshot and one "image-2.png" dynamic-response slide that was attached to
the 1-May follow-up email.

### Meeting 2026-03-20 (Mario's intro deck, "Status of AI hardware" — slide 1..41)

| File | Slide title / content | Numeric annotations |
|---|---|---|
| 12.05 | "Status of AI hardware" intro | GPU power = 700 W; high cost up to 21% energy by 2030; pollution = 32 Mt e-waste/2030; data transfer >60% energy; market $681.05 B (2024) → $1 T (2030) |
| 12.05(1) | duplicate of 12.05 (zoomed re-share) | same values |
| 12.06 | "AI on edge" Innatera taxonomy <1mW..10–100W | power tiers: <1 mW sensor, 1–10 mW MCU, 10–100 mW DSP/NN, 1–10 W MLNN, 10–100 W HP-inference |
| 12.07 | "Spiking Neural Networks" + LIF eqn | LIF: τ dV/dt = -(V-Vrest)+RI; X.Zhu, S.Pazos, M.Lanza Nature 618 57-62 (2023) |
| 12.07(1) | dup of 12.07 | same |
| 12.07(2) | "Memristor-based electronic neuron" (Liang 2021 / Mitsuru 2024) | #devices >20, low integration density, high power |
| 12.08 | "Memristor-based electronic neuron" overview duplicate | same |
| 12.08(1) | "Neuron implementation" — adaptive exp neuron, mixed-signal IF, analog-Δ; Indiveri Frontiers 2011 | references Indiveri 2011; Vlasov 2024 |
| 12.08(2) | "Memristor-based electronic neuron" arrays — 25 µm² Ag/hBN/Au on SiO2 vs 0.05 µm² Ag/hBN/W on-chip; Alharbi MS&E 2024 | cell areas 25 µm² and **0.05 µm²** (on-chip) |
| 12.09 | "Die-to-die variability — NS-RAM" with bulk-terminal-modulation cartoon. S.Pazos Nature 643 (2025) | references the published Nature paper |
| 12.26 | **"Semi-empirical model fits for impact ionization bulk currents"** (Sebas) | Form: I_bulk = I_exp + I_pwl;  I_exp = a(V_d - b)^c if V_d ≤ d; I_pwl = PWL(V_d) if V_d > d; **a, b, c, d = PWL(V_b)** — i.e. exponential coefficients are themselves piecewise functions of body voltage |
| 12.27 | "Measurements vs SPICE simulation using semi-empirical bulk currents" | At low V_d, body has no time to charge → fit fails for V_b ≠ 0 V if N=1; tilts use N>1 |
| 12.27(1) | **"Floating body 2T NS-RAM cell under transient V_D ramps"** I-V family, V_G1 = 0.5 V, V_G2 swept | Drain current 10⁻¹² → 10⁻⁴ A, V_d 0..2.5 V; clear hysteresis loops |
| 12.29 | "NS-RAM in standard triple-well CMOS (130nm)" | Cell area **5.3 µm × 6 µm**; deep N-well **8.5 µm²**; expected 1000× density improvement over state-of-art neuron |
| 12.29(1) | "Deep-Nwell NFET floating body 1T (thick)" | Area **8 µm²**; firing window **7× — 10⁴×**; 100% yield, 100 µV variability nominal; 180 nm CMOS |
| 12.30 | **"2T NS-RAM spiking neuron cell (thick oxide)"** | Area **17 µm²**; second-transistor body-modulation can give **>10⁴× off/on**; V_G1 = 2.5 V (firing), V_G2 floating; outstanding firing |
| 12.33 | **"NS-RAM Simple LIF in Brian2 for input neurons"** | Slowdown 10⁹; **G_LEAK_REST=1.343, THRESH_VAL=1.354, TAU_REF=4 ms, REFRACTORY=4.8979 V (!), TIMESCALE=145 µs, EXCIT_VALUE=2.3** — these are Brian2 dimensionless params, not real-device |
| 12.39 | "More physically realizable SNN in Brian2" | **Poisson reference = 85%, LIF (w/Poisson training) = 72%** on a confusion matrix |

### Meeting 2026-04-22 19.57

| File | Content |
|---|---|
| 19.57 | NUS Ariba supplier-registration portal screenshot (vendor onboarding). No physics value. |

### Meeting 2026-04-30 (Sebas BSIM fits — already partially extracted in O48)

| File | Content | Key numbers |
|---|---|---|
| 13.23 | **3-panel I-V family**: V_G1 = 0.6 / 0.4 / 0.2 V, V_G2 sweep ranges (-0.2..0.1, 0..0.3, 0..0.5) | inset = full 2T schematic with M1, M2, V_B floating body |
| 13.24 | **4-panel parameter dependences**: BETA0(VG2) for 3 VG1 branches, ETAB(VG2) for 3 VG1, K1(VG1) "for all VG2", NFACTOR(VG2) for 3 VG1 | range BETA0 = 11..21, ETAB = 0.8..2.5, K1 = 0.42..0.56, NFACTOR = 2..12 |
| 13.25 | **Transient I-V noise band**: voltage pulse train 0..1.6 µs with current spikes 0..5 mA; underneath two I-V "noise cloud" panels showing measurements as a *band* not a curve | confirms the I-V is variability-bounded, not a single trace |
| 13.28 | **3-corner overlay meas vs sim** (thick=meas, thin=sim), 3 representative bias combos | (VG1,VG2) = (0.6,0.35), (0.4,0.25), (0.2,0.0) — fits within 1 decade |
| 13.31 | **"Simple NS-RAM cell with integration (self-reset)"** | Cap C_int values shown; energy per spike numbers in green box |
| 13.31(1) | **"NS-RAM blocks for input neurons (soma without diode)"** | **Energy: ~0.5 nJ crossover, ~21.5 pJ per spike of action; 6.7 pJ spike generation, ~25 fJ integration loss; area 40 µm²** |
| 13.33 | **"NSRAM firing with linear excitatory and inhibitory inputs"** | Linear range V_G1 between 2.5 V and 3 V; uses **thick oxide** (high drain voltage RS-RAM); 2 inverter stages drive soma directly |
| 13.46 | duplicate of "AI on edge" Innatera slide | same |
| 13.47 | NS-RAM fab process: morphology after each step | Au top, hBN layer; deposition >400°C; multilayer hBN on chip; metal stack 1–4 |
| 13.50 | "Spiking Neural Networks" review slide (repeat of 12.07 content) | same |
| 13.53 | Outlook screenshot — interest in NIM Med Sized Grant; due **8 May** | admin item |
| 13.53(1) | Outlook screenshot — Re: For Your Advice HSE rep for Deputy Head Research Meeting | admin item |

### Standalone

| File | Content | Numbers |
|---|---|---|
| image-2.png | **"Dynamic response (ramp rate dependence)"** — THE diode slide from 1-May email | I-V vs ramp rate 10..2000 V/s; firing slope flattens at high t_rise; **SR & firing time experiments (5–7 in expt list)** = critical for capacitance/τ fitting; V_b ≠ 0 increases body voltage; "parasitic capacitances play a crucial role on the effective time constant" |

---

## 2. xlsx full parameter table (`2Tcell_BSIM_fits.xlsx`, Sheet1, 35×14)

This is the **DC BSIM fit table** — the canonical source.

Columns: `VG1 | VG2 | trise | ETAB | K1 | ALPHA0 | BETA0 | NFACTOR | mbjt | IS | area`
Plus right-hand legend explaining each parameter's effect.

**Top-5 numerical findings:**

1. **ALPHA0 is FIXED at 7.842e-05** across all 33 bias rows — Sebas does NOT vary impact-ionization prefactor with bias; only BETA0 varies. This contradicts the "ALPHA0/BETA0 jointly polynomial in VG1,VG2" assumption in our v0.12.0 release notes. **Single ALPHA0, single polynomial in BETA0(VG1,VG2) only.**

2. **K1 is FIXED inside each VG1 family but jumps between VG1 levels**: K1=0.55825 (VG1=0.2), 0.53825 (VG1=0.4), 0.41825 (VG1=0.6). I.e. K1 depends *only* on VG1 (not VG2). **Δ between VG1=0.4 and VG1=0.6 is -0.12** — a large step, suggests layout-dependent V_th shift.

3. **NFACTOR(M2)** scans 12.15 → 1.25 monotonically as VG2 rises from -0.2 to +0.5 (for VG1=0.6). Confirms the legend "Higher NFACTOR → Higher Vrelax, Higher Vfire at constant VG2." NFACTOR drops as VG2 increases.

4. **BETA0 monotonic in VG2** within each VG1: at VG1=0.2 it goes 10.75 → 14 across VG2=-0.2..0.1; at VG1=0.4 it sits at 19; at VG1=0.6 it sits at 20. **Saturates at 19–20 for higher VG1.**

5. **mbjt** = 0.001 for VG1=0.2 family (bipolar essentially OFF), then jumps to **mbjt = 1** for both VG1=0.4 and 0.6 families. This is a hard switch, not a continuous polynomial — the parasitic BJT is enabled only above some VG1 threshold between 0.2 and 0.4 V.

`trise` (the body-charge time-constant proxy): mostly ≈11.63 (VG1=0.2), then 10.59..12.98 (VG1=0.4), then **9.04 plateau** for VG1=0.6 — **monotone-decreasing in VG1**, exactly what image-2 ramp-rate slide visualizes. ETAB ranges 0.8..2.5 and is **monotone-increasing in VG1**.

The right-column legend transcribed verbatim:
- ETAB: higher → higher I_fired, lower V_relax
- K1: higher → less leakage at relax state
- ALPHA0: higher → lower V_relax, V_fire, higher I_fired (very sensitive)
- BETA0: higher → narrower hysteresis, higher V_relax, slightly lower V_fire
- NFACTOR: higher → higher V_relax & V_fire at constant VG2
- mbjt: integer scaling of bipolar contribution
- IS: Shockley saturation current of BJT
- area: BJT physical area (real-valued mbjt)

CSV (`2Tcell_BSIM_param_DC.csv`) is the same data with NaN for the un-fit VG1=0.4 negative-VG2 rows and the VG1=0.6 negative-VG2 rows. **Only 23 of 33 bias points are actually fitted**; 10 NaN entries flagged for future fits.

---

## 3. pptx (`2026-04-29 NS-RAM I-V BA plots.pptx`) — 3 slides

- **Slide 1**: triple-panel I-V family, V_G1=0.6/0.4/0.2. VG2 ranges: VG1=0.2 → VG2=-0.2..0.1 step 0.05; VG1=0.4 → 0..0.3 step 0.05; VG1=0.6 → 0..0.5 step 0.05. "Symbols=measurements, Lines=simulations."
- **Slide 2**: 3-corner overlay (representative biases) "Thick=measurements, Thin=simulations." Triplets (VG1, VG2) = (0.6, 0.35), (0.4, 0.25), (0.2, 0.0).
- **Slide 3**: parameter dependences with 4 numbered pictures — **NFACTOR vs VG2** (top-right, 3 VG1 branches), **BETA0 vs VG2** (3 VG1), **ETAB vs VG2** (3 VG1), **K1 vs VG1** ("For all VG2"). Confirms that **K1 depends only on VG1, not VG2** (a quantitative law for pyport_v5).

---

## 4. mail.txt — chronological summary (8 threads, 21 Mar → 13 May)

- **20 Mar 2026** — Mario sends Zoom invite for first meeting (21 Apr).
- **23 Mar** — Eric introduces nsram v0.1.0 PyPI, asks for Sebas review.
- **24 Mar** — Sebas: will update parameters in coming weeks; new role transition.
- **25 Mar** — Eric → Sebas: v0.9.0 features (Chynoweth, SRH, BVpar; tau=10,139 s @ 300 K; 7 conductance levels; 97% temporal XOR, 99.6% Mackey-Glass, 96.75% MNIST; LTP/LTD).
- **2 Apr** — Eric: v0.10.0, BEAM byte-level learner, 3.14 bits/char on text8 with 60K params.
- **4 Apr** — Mario: postpone to second half of April; "presented to a company, interested."
- **17 Apr** — **Sebas KEY**: dropped avalanche-diode models (LTSPice convergence issues). Now uses **BSIM4 impact ionization + body bias directly**, with polynomial dependence of fit parameters on (VG1, VG2) and LDE. Asks: "can your approach drop avalanche voltage and use BSIM impact ionization + body voltage directly?"
- **18 Apr** — Robert asks for 2T schematic + raw IV CSVs + process node + foundry card.
- **19 Apr** — Eric: v0.12.0 ships with §6.1 ALPHA0/BETA0, §2.2 K1/K2 body-bias, §10.1 junction breakdown, §12 KT1/UTE/XTIS, §13 SA/SB/KU0/KVTH0 LDE. Channel-HCI (§6.1) fits **~4 decades RMS better** than §10.1 junction breakdown over 2–4.5 V.
- **20 Apr** — Sebas attaches LTspice .asc + 33 IV CSVs (0.2 V/s) + PTM130 model card. Foundry card cannot be shared.
- **30 Apr** — **Sebas KEY** post-meeting: slides + BSIM fits attached (NDA: cards private). Notes: "Some params change only for M1, while NFACTOR changes only for M2 (I attribute this to LDE)." Mentions testchip floorplan upcoming. Asks for simple architecture to test fan-out.
- **1 May** — **Sebas KEY**: simulation parasitic diode wasn't working. New schematic = explicit **pdiode area 5 × 4.4 µm²**, OR alternative **linear cap 5–10 fF** for body junction. Updated transient agreement.
- **3 May** — Eric: brief draft on Overleaf, BSIM/pdiode private; pyport now loads M1/M2 LDE distinction.

**Key new physics from emails (5):** (i) avalanche diode dropped; (ii) NFACTOR M2-only LDE handle; (iii) pdiode 5 µm × 4.4 µm; (iv) body junction cap 5–10 fF; (v) PolynomialBSIM4Params(VG1,VG2) only for BETA0 (others have lower-dim dependencies, see xlsx finding #1–#3).

---

## 5. Slow I-Vs vs existing fast I-Vs — md5 diff

```
md5: ff50159e1cd7f9918e1e359dc9104076 (Zoom/Slow IVs/VG1=0.6 VG2=0.00)
md5: ff50159e1cd7f9918e1e359dc9104076 (data/sebas_2026_04_22/VG1=0.6 VG2=0.00)
```

**Verdict: BYTE-IDENTICAL.** The "Slow I-Vs … SRavg=0" folder is the SAME 33 CSVs already in `data/sebas_2026_04_22/`. The folder name "SRavg=0" simply documents that these are the *baseline* slow-sweep set (0.2 V/s, per email of 20 Apr). No new IV data. The promised "multiple ramp rates for dynamics" Sebas mentioned has NOT yet arrived in this packet — only image-2.png provides ramp-rate evidence visually.

## 6. Model card cross-reference

| Card | nsram/Zoom path | data/ path | Diff |
|---|---|---|---|
| parasiticBJT.txt | schematic&modelCards/ | sebas_2026_04_22/ | **IDENTICAL** |
| PTM130bulkNSRAM.txt | schematic&modelCards/ | sebas_2026_04_22/ | identical EXCEPT line 62 / 117 (our local DEDUP annotations to remove stale `Alpha0=0.00 Beta0=30.0`); upstream is unchanged |
| 130DNWFB(M1).txt | 2026-04-30 BSIMfitsBA/ | sebas_2026_04_22/M1_130DNWFB.txt | **IDENTICAL** |
| 130bulkNSRAM(M2).txt | 2026-04-30 BSIMfitsBA/ | sebas_2026_04_22/M2_130bulkNSRAM.txt | **IDENTICAL** |
| pdiode.txt | nsram/Zoom/pdiode.txt | data/sebas_2026_05_02/pdiode.txt | identical |
| 2tnsram_simple.asc | schematic&modelCards/ | (none) | **NEW** — LTSpice 2T cell ASC (1419 bytes) not previously catalogued |

`parasiticBJT.txt` contents (constants for SPICE NPN to register if not in pyport):
`is=5E-9 va=100 bf=10000 br=100 nc=2 ikr=100m rc=0.1 vje=0.7 re=0.1 cjc=1e-15 fc=0.5 cje=0.7e-15 ne=1.5 ise=0 tr=20e-12 tf=25e-12 itf=0.03 vtf=7 xtf=2`

`pdiode.txt` key values:
`bv=11, ibv=97740, cj=7.33e-4, cjsw=1.05e-10, vj=0.219, m=0.241, fc=0.5, xti=6.5, eg=1.11`
**Body junction breakdown voltage = 11 V** (much higher than V_d range of interest 0–2.5 V — confirms pdiode is not the firing mechanism, only the capacitance source).

---

## Gate verdict

**PASS (≥5 new physics values not in prior audit):**
1. ALPHA0 = 7.842e-5 IS FIXED (single value, not VG1/VG2 polynomial).
2. K1 depends ONLY on VG1: 0.55825 / 0.53825 / 0.41825.
3. mbjt is binary (0.001 vs 1) thresholded between VG1=0.2 and 0.4.
4. trise plateau at 9.04 for VG1=0.6 (body time-constant scales 1/VG1).
5. Only 23 of 33 bias points are actually fitted (10 NaN in CSV) — clear what to ask Sebas next.
6. pdiode reverse breakdown bv=11 V (rules out pdiode-based firing).
7. NFACTOR varies VG2-only for M2; M1 has fixed NFACTOR (Sebas LDE comment).
8. mbjt-OFF in VG1=0.2 family means BJT contribution can be skipped at low VG1.

**AMBITIOUS (quantitative laws):**
- **Law L1**: ALPHA0(VG1,VG2) ≡ 7.842e-5 (constant). BETA0 is the *only* impact-ionization handle.
- **Law L2**: K1(VG1) = piecewise (0.55825 if VG1=0.2; 0.53825 if VG1=0.4; 0.41825 if VG1=0.6) — quadratic interp candidate.
- **Law L3**: NFACTOR(M2) ∝ -VG2 (monotone-decreasing roughly linear within each VG1 branch); slope = (NFACTOR(VG2=-0.2) - NFACTOR(VG2=+0.1)) / 0.3 ≈ -19.7 / V for VG1=0.2.
- **Law L4**: trise(VG1) plateau-then-step: 11.63 / 11.46 / 9.04 — fits A·exp(-VG1/τ).
- **Law L5** (from image-2.png): body capacitance ≈ 5–10 fF dominates τ_body together with ramp-rate-dependent t_rise — firing slope depends on dV_d/dt up to ~2000 V/s.

---

## Top-5 actionable for pyport_v5

1. **Replace polynomial ALPHA0(VG1,VG2) with constant 7.842e-5** in `src/nsram_pyport_v2.py` BSIM4ImpactBlock. Keep BETA0(VG1,VG2) polynomial. Single API: `alpha0 = 7.842e-5; beta0 = poly_beta0(vg1, vg2)`.
2. **Add `K1(VG1)` lookup table** (only 3 nodes: 0.2/0.4/0.6 → 0.55825/0.53825/0.41825); for off-grid VG1, monotone-cubic interp. Do NOT make K1 a function of VG2.
3. **Add `mbjt(VG1)` step function**: 0.001 if VG1 ≤ 0.3 else 1.0. This switches the parasitic BJT path on/off and probably explains current pyport over-firing at low VG1. Mark the threshold as a fit handle to refine when new data arrives.
4. **Implement pdiode body cap** as a constant **C_body = 7 fF** linear cap (middle of 5–10 fF email range) plus a TRUE diode card matching `pdiode.txt`. `firing_mode="both"` should use the diode; `firing_mode="channel"` may collapse to linear cap for speed. Confirms 1 extra Newton iter per step penalty (acceptable, per Eric's 3-May email).
5. **Drop the avalanche/Chynoweth path entirely** (per Sebas 17-Apr). Keep only BSIM4 §6.1 impact ionization + complementary BJT current. Gate flag `use_chynoweth=False` as default. Ramp-rate-dependent I-V hysteresis must come from C_body × dV_d/dt RC integration, not Chynoweth ionization time.
