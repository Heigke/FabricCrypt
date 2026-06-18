# openai response (gpt-5) — 205s

Short answer up front
The remaining 2–5 decade shortfall is injection‑limited: BSIM4 §6.1 Iii with the BETA0 values in Sebas’s cards/CSV (≈18–20 V) is effectively zero at the NS‑RAM operating headroom (Δ=Vds−Vdseff≈0.2–0.35 V), so the body never charges enough to turn the NPN hard on; LTSpice matches because its deck uses a larger effective Iii (either smaller β0 or an alternate Iii form) while our port uses the classic form.

1) Which of (a)/(b)/(c)/(d) is most likely? Ranked
1. (b) Foundry/Spice Iii physics mismatch — HIGH. With β0≈18–20 V, exp(−β0/Δ) kills Iii by 25–30 decades (we compute ~10^-25…10^-42 A at the worst bias). Any Spice deck using a smaller β0 (1–3 V typical in short‑channel fits) or an alternate IIMOD shape will deliver the needed nA–µA body current and light the NPN; ours won’t.
2. (a) Different fixed point — LOW. Your A1g sweep + arclength shows a single branch (n_folds=0) and all Vb_inits converge to the same root.
3. (c) “Complementary bipolar current” misunderstanding — LOW. The ASC shows it’s just Q1 (NPN) with emitter=GND; we already model that exactly.
4. (d) Arclength wrong branch — VERY LOW. The pseudo‑arclength path has no fold and the reverse/forward sweeps land on the same branch.

Code path check: leak.compute_iimpact uses the classic BSIM4 IIMOD=0 form with Idsa·Vdseff; size‑dependent binning of alpha0/beta0 is already applied in compute_size_dep (scaled["alpha0"] = base + lalpha0·Inv_L + …; same for beta0).

2) The lalpha0 puzzle (units/sign and whether to apply)
- BSIM4 source bins “X” as pParam.X = model.X + lX·Inv_L + wX·Inv_W + pX·Inv_LW (b4temp.c, size‑dependent setup; same path we already use). For alpha0/beta0 this is active because they’re in our SCALED_PARAMS list.
- Units: alpha0 has units m/V. Therefore lalpha0 has units m^2/V so that lalpha0/Leff has m/V. For M2: lalpha0=−9.843e-12, Leff≈1.8 µm ⇒ Δα0 ≈ −5.5e-6 m/V, i.e. −7% of α0=7.84e-5 m/V (exactly your A1d number). lbeta0=−9.5e-7 with the same Leff gives Δβ0≈−0.53, i.e. β0_eff≈17.5 (also what you computed). Conclusion: length binning should be applied (we already do), and it does NOT flip alpha0 negative with the provided numbers.

3) Body‑charging bistability (direct test)
Use your existing solver but start on the high‑Vd side and sweep backward. Minimal patch in scripts/z91g_two_model_validation.py to drive arclength from the top:
- Replace the single forward arclength call with two traces:

from nsram.bsim4_port.arclength import trace_arclength, interpolate_at_targets
…
# forward trace (low→high)
path_f = trace_arclength(cfg, model_M1, bjt, VG1_t, VG2_t,
                         Vd_start=float(c["Vd"].min()), Vd_max=float(c["Vd"].max()),
                         P_M1=P_M1, P_M2=P_M2, model_M2=model_M2)
# reverse trace (high→low)
path_r = trace_arclength(cfg, model_M1, bjt, VG1_t, VG2_t,
                         Vd_start=float(c["Vd"].max()), Vd_max=float(c["Vd"].min()),
                         P_M1=P_M1, P_M2=P_M2, model_M2=model_M2)
warm_f = interpolate_at_targets(path_f, c["Vd"])
warm_r = interpolate_at_targets(path_r, c["Vd"])
# then run solve_2t_steady_state twice (warm_start from warm_f and warm_r) and compare

If a high‑Vb branch exists that our forward trace misses, warm_r will land on it (it doesn’t in your A1g, so this should come back identical — but this is the conclusive check).

4) “Complementary bipolar current” interpretation
In a vanilla LTSpice schematic it can only mean Q1’s collector current in parallel with the MOS channel current. The ASC has no B‑sources or subckts; Q1 is emitter=GND, base=Body, collector=Drain. So “complementary” = the parasitic NPN path complementing the MOS channel path; there is no hidden card/SUBCKT beyond parasiticBJT.txt.

5) Single highest‑information experiment to run tonight (<1h)
Flip the hypothesis by forcing “Spice‑like” Iii and see if the gap closes. One‑liner in z91g to override β0 at runtime and re‑plot ONLY the two worst curves.

Patch (temporary) in scripts/z91g_two_model_validation.py, right after make_overrides():
    # EXPERIMENT: force smaller beta0 in Iii to test injection-limited hypothesis
    BETA0_TEST = float(os.getenv("NSRAM_BETA0_TEST", "0"))
    if BETA0_TEST > 0:
        if P_M1 is None: P_M1 = {}
        if P_M2 is None: P_M2 = {}
        P_M1["beta0"] = torch.tensor(BETA0_TEST, dtype=torch.float64)
        P_M2["beta0"] = torch.tensor(BETA0_TEST, dtype=torch.float64)

Run:
NSRAM_BETA0_TEST=1.5 python scripts/z91g_two_model_validation.py
and inspect VG1=0.6, VG2=0.0 and VG1=0.4, VG2=0.0.

Outcomes:
- If Id jumps into the 10^-6…10^-5 A band and Vb → 0.7–0.9 V, the miss is entirely Iii; keep IIMOD=0 but (i) treat β0 as a per‑bias fit parameter (from Sebas CSV for M1; add an M2 β0 column, or a polynomial), or (ii) add the IIMOD=1/2 branches and select to match the foundry tool.
- If Id hardly moves, then we’re BJT‑limited (unlikely) — repeat with NSRAM_BETA0_TEST=1.5 and also multiply bjt.area by 10× in make_bjt to see if Id tracks (BJT‑limited) or not (still injection‑limited).

Small but important fixes to land regardless
- GISL “ref default” bug (agisl=agidl etc.) — add a pass‑4 re‑resolve after user overrides (exact code from your A1e):
# Pass 4: re-resolve ref defaults whose source was user-overridden
for name, info in PARAMS_META.items():
    d = info["default"]
    if isinstance(d, tuple) and d[0] == "ref" and name not in self._given:
        self._values[name] = self._values.get(d[1], 0.0)
in nsram/bsim4_port/model_card.py __init__ after pass‑3.
- Ensure mbjt is honored (you already patched): bjt.area = csv.area * csv.mbjt.

Optional next steps if the β0 test confirms injection‑limit
- Implement BSIM4 IIMOD=1 (and the HSPICE IIMOD=2 square‑headroom form) in leak.compute_iimpact, selectable by model.get("iimod", 0). That gives you a knob to match the foundry flow without touching the CSV.
- Add a per‑bias β0 override for M2 in the CSV path (your loader already supports nfactor for M2; mirror that for β0 if Sebas can provide it).

Notes on lalpha0/lbeta0 again (to close Q2 explicitly)
- Berkeley source: size‑dependent binning (alpha0, beta0) is in the general pParam scaling block (b4temp.c around lines 700–820 in 4.8.x): pParam->alpha0 = model->alpha0 + model->lalpha0*Inv_L + …; idem for beta0. Units consistent with alpha0 [m/V], beta0 [V], and Inv_L = 1/leff for binunit=2 (your cards use binunit=2).
- Your current port already applies this via compute_size_dep; no extra code is needed, and with Leff≈1.8 µm it yields α0_eff≈0.93·α0 and β0_eff≈18−0.53≈17.5.

Why this experiment is decisive
- It cleanly separates (b) from everything else: if boosting Iii alone fixes the amplitude and Vb, we do not need to chase solver topology or schematic mysteries.
- It’s a one‑line env var, runs in <1 min on two curves, and gives an immediate “yes/no” on the injection‑limited hypothesis, which all your diagnostics already point to.

Bonus sanity checks you can run in parallel (no code)
- Print Iii_M1, Iii_M2, Ib_Q1, Ibs_M2 at VG1=0.6, VG2=0.0 before/after NSRAM_BETA0_TEST=1.5. Expect Iii to rise from ~10^-25 A to nA–µA; Vb should lift past ~0.6 V; Ib_Q1 rises; Id follows Ic_Q1 and the diode begins to conduct strongly once Vb>~0.6 V.

Summary
- Most likely cause = we are using the classic BSIM4 IIMOD=0 with β0≈18–20 V which makes Iii negligible at Δ≈0.27 V; Sebas’s Spice setup effectively has much smaller β0 or an alternate Iii branch.
- lalpha0/lbeta0 binning: already applied; units consistent; does not flip signs with your cards.
- No hidden “complementary” source — it’s Q1.
- Run the single β0‑override test; if it fixes the worst curves, proceed to (i) fit β0 per bias (or per VG1/VG2), or (ii) add IIMOD=1/2; also land the GISL ref‑default fix and keep the mbjt*area mapping.
