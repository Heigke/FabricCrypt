# O93 — Sanity check: K1+ALPHA0 card-value DC fix (NS-RAM 2T cell)

We just landed a 2-knob fix on the Sebas 33-bias DC log-RMSE fit (fwd+bwd, n=66) of our
NS-RAM 2T floating-body cell (Pazos Nature 640:69 2025, 130nm bulk Si). Headline:

| K1@VG1=0.6 | ALPHA0 | median log10-RMSE (dec) | VG1=0.6 dec | Imeas/Ipred @ VG1=0.6 |
|---|---|---|---|---|
| 0.41825 (current code override) | 7.842e-5 (CSV)  | **1.163** (baseline) | 1.732 | 46× |
| 0.41825                         | 7.83756e-4 (card)| 1.163  | 1.295 | 18× |
| 0.53825 (BSIM card)             | 7.842e-5         | 0.883  | 0.927 | 7.6× |
| **0.53825 (card)**              | **7.83756e-4 (card)** | **0.665** | **0.617** | **3.9×** |

**Both fixes revert hand-tuned/CSV values to the BSIM model-card values supplied by Mario
Lanza's group.** No parameter fitting, no extra free knobs. The combination is
super-additive at VG1=0.6 (the regime where the historical 46× current shortfall lived).

## Background context (5 months of fixes, all already applied)

Before reaching the 1.163-dec baseline, we already fixed:
- BSIM4 BJT mbjt mapping, body-pdiode at floating body, well-body diode (vnwell=2V)
- pdiode params Js=5.37e-7/22e-12, n=1.05, Rs=1e6
- emitter=GND topology, M2.B=GND
- 5 ngspice bugs, IIMOD card patch (R-26 R-27), Theta0_n + phi formula
- BJT η-sigmoid clamping, IFT sign-bug upstream patch (z474b)
- arclength solver for snapback fold, NaN-mask for sub-threshold

## The two newly-identified bugs

**Bug 1 — K1@VG1=0.6 hand-tuned override** in `scripts/pillar_I_C3_jts_tat.py:92`:
```python
BRANCH_FLAT = {
    0.4: {"ETAB": 1.9, "K1": 0.53825, "ALPHA0": 7.842e-05, ...},
    0.6: {"ETAB": 2.5, "K1": 0.41825, "ALPHA0": 7.842e-05, ...},  # <-- 0.418 is hand-set!
}
```
M2_STATIC_OVERRIDES has `"k1": 0.63825`. BSIM card value is 0.53825. The override at
VG1=0.6 lowers Vth (since Vth ∝ K1·√(2φF − Vbs)) and exaggerates the floating-body
modulation, mis-modeling the triode-regime saturation.

**Bug 2 — ALPHA0 CSV vs card value** in Sebas's CSV (`2Tcell_BSIM_param_DC.csv`)
ALPHA0=7.842e-5 is **10× smaller** than the Mario LALPHA0_FIX card value 7.83756e-4
(the value in `M2_130bulkNSRAM_LALPHA0_FIX.txt`). ALPHA0 is the BSIM4 impact-ionization
prefactor; smaller value → smaller Iii → less body charging → less NPN amplification.

Critically, the ALPHA0 fix ALONE has **Δ=0 on full-33 median** (only VG1=0.6 subset Δ=−0.436).
But combined with K1, it pushes VG1=0.6 from 0.927 dec to 0.617 dec.

## What we falsified BEFORE finding these bugs

Spent today's 8h running:
- Track B (selfheat=1): best Rth=1e7 → Δ −0.023 dec (proxy nothing-burger)
- Track C (Hurkx-Γ field-enhanced TAT): best α=0 wins (i.e., no improvement)
- Combo (rbodymod + selfheat + Hurkx): Δ +3.6 WORSE (destabilizing)
- Track Diag: gap is **symmetric fwd/bwd** (ρ=0.95) — NOT a memory effect; localized to
  VG1=0.6 triode regime
- Track Vision: re-OCR'd Mario/Sebas slides — slide 21 says LALPHA0_FIX card is the
  correct card value, and CSV ALPHA0=7.842e-5 was a typo/legacy value carried over

## Remaining gap — the snapback knee (see plot_snapback_vs_data.png)

After the K1+ALPHA0 fix the triode-regime current matches data within ~4× at VG1=0.6.
BUT the **snapback knee in our model fires at Vd ≈ 1.3-1.5V**, while in Sebas data the
sharp uptick happens at **Vd ≈ 0.7-0.8V**. So the trigger voltage for snapback is
2× too high in the model.

In our model, snapback fires when impact-ionization Iii charges the floating body Vb
to ~0.7V (NPN forward-biased Vbe). With ALPHA0=7.84e-4, that requires Vd≳1.2V.

Possible explanations for the 0.75V data knee being below our model's:
- Direct gate-body coupling: VG1 sets a standing Vb offset (we model Vb starting at 0)
- BBT/Hurkx prefactor missing — separate from impact-ion
- NPN Bf is higher than the card 100 → tighter coupling
- M2 BVDSS-style avalanche at lower Vd, not currently modeled
- Real device has a parallel leakage path that loads Vb at low Vd

## The 3 questions

**Q1 — Is the K1+ALPHA0 card-value finding mechanistically sound and publishable?**
Both fixes revert hand-set values to BSIM card values. Combined, they take median fit
from 1.163 → 0.665 dec on 66 measurements with no new parameters. Is this defensible
to publish as "two configuration bugs" rather than parameter fitting? Or is there a
risk reviewers see this as just lucky knob-twiddling that happens to fit?

**Q2 — What's the most likely physics behind the 0.75V data knee?**
Rank top 3 candidates for explaining why the data's snapback fires at Vd≈0.75V while
our model (after K1+ALPHA0 fix) fires at Vd≈1.5V. For each, name a single decisive
diagnostic experiment we can run in <2 hours.

**Q3 — Are we missing something fundamental?**
Look at the falsification track record (selfheat dead, Hurkx-Γ dead, rbodymod dead,
mbjt fixed, IIMOD fixed, eta_sigmoid fixed, IFT sign fixed, BJT topology fixed). The
gap of 1.163 dec was reproducible and persistent. The fix turned out to be 2 BSIM card
values. Is it plausible that the OUTSTANDING 0.665 dec residual is ALSO a config bug,
or is it more likely now to require a structural physics addition (BBT, gate-body
direct coupling, multi-emitter NPN, etc.)?

NO-CHEAT. Cite specific paper sections / Mario slide numbers if you know them.
≤400 words per question. Be brutal — if you spot a flaw in the experimental design or
think we got lucky, say so.

## Artifacts attached
- `ablation.json` — full sweep numerics
- `plot_combo.png` — bar chart of all 4 conditions × 3 VG1 branches
- `plot_snapback_vs_data.png` — Id-Vd at VG1=0.6 (4 panels of VG2) showing model vs Sebas
- `diag_verdict.md` — fwd/bwd asymmetry investigation
- `alpha_verdict.md` — ALPHA0-only sweep
