# Oracle query O1 — Low-VG2 residual diagnosis

**Recommended model:** GPT-5 (Thinking) or Gemini 2.5 Pro. Cheaper variants
will not have enough physical depth on this. *Do not pay for full model
on this if a cheaper one already gives a confident answer.*

**You will receive:** this prompt + the attached zip with code, the two
SPICE model cards, the per-bias parameter CSV, and the failing plot.

**Audience for your answer:** an engineer who has read the BSIM4 v4.8.3
manual and is debugging a PyTorch port. Skip basics. Get to the diagnosis.

---

## Background (please read once before answering)

We're fitting a 2T NS-RAM cell (M1 + M2 + parasitic NPN, floating P-body)
in a differentiable PyTorch port of BSIM4 v4.8.3. The device data and
fitted parameters come from Sebastian Pazos's lab (NUS / Madrid).

Sebastian sent us his per-bias fitted parameter table
(`2Tcell_BSIM_param_DC.csv`) — for every measured (VG1, VG2) bias point
he gives the BSIM4 + BJT parameters his SPICE deck uses. The columns:

- `ETAB`, `K1`, `ALPHA0`, `BETA0` — apply to **M1** (LDE-driven)
- `NFACTOR` — applies to **M2** (also LDE)
- `mbjt`, `IS`, `area` — for the parasitic NPN
- `trise` — rise-time we don't yet handle

He sent two model cards:
- `M1_130DNWFB.txt` — deep N-well floating-body (M1)
- `M2_130bulkNSRAM.txt` — bulk (M2)

Per his email, the cards are physically distinct devices.

We've validated against ngspice (Vth ±0.1 mV, Vdsat ±2 mV, Id ±2% on
saturation). 158 unit tests of the BSIM4 port pass. We have a
pseudo-arclength continuation solver that hits 100% Newton convergence
through the snapback fold.

## The result we cannot move

We did a **forward-only validation** (no fitting): for every measured
curve, we apply Sebastian's CSV parameters at that exact (VG1, VG2),
run our forward simulator with the right card on each device, and
compare to measurement.

**Result:** median log-RMSE = **2.40 decades**, p90 = **4.83 decades**.
See attached `fit_vs_meas.png`.

The pattern in the per-curve numbers is unambiguous:

| VG1 | VG2 range | log-RMSE range | comment |
|----:|----:|----:|---|
| 0.20 | −0.20 to +0.10 | 0.46 → 1.80 | best |
| 0.40 | 0.00 to +0.30 | 0.86 → 3.89 | bad at low VG2 |
| 0.60 | 0.00 to +0.50 | 0.93 → 5.63 | catastrophic at low VG2 |

**Error grows dramatically as VG2 → 0** (toward M2 cutoff). At VG1=0.6 /
VG2=0 we are 5.6 decades too low on |Id|. At VG1=0.6 / VG2=0.5 we are
within 1 decade.

## What we've checked

1. **Both model cards loaded correctly** — vth0=0.54153 V, vsat=102230,
   k1 differs (0.53825 vs 0.63825 — M1 vs M2), etab differs (1.8 vs
   −0.087), beta0 differs (19 vs 18). Confirmed by `model.get(...)`.

2. **NFACTOR override IS being applied via `sd.scaled["nfactor"] = X`**
   — patched per-bias from CSV, then forward_2t runs.

3. **Two-model refactor is structurally correct** — `forward_2t` now
   accepts `model_M1` and `model_M2`; 158 existing tests + 2 new
   regression tests pass.

4. **The .param continuation lines were silently dropped by our SPICE
   parser** — fixed via post-load patch (vth0n=0.54153, vsatn=102230,
   etc.). That dropped median RMSE from 4.23 → 2.40. Real fix, but not
   the whole story.

## What we suspect (rank our suspicions)

- **Suspicion 1: NFACTOR isn't reaching the actual subthreshold formula.**
  Possible if `compute_dc` reads it from somewhere other than
  `sd.scaled["nfactor"]`. We're triple-checking this in a parallel task.

- **Suspicion 2: `mbjt` isn't being applied.** The CSV has `mbjt=0.001`
  for all VG1=0.2 rows and `mbjt=1.0` for VG1=0.4 / 0.6. That's a 1000×
  multiplier on BJT current — if we ignore it, BJT current at VG1=0.2 is
  1000× too high; at VG1=0.4/0.6 it should be at "1.0" baseline. Our
  GummelPoonNPN doesn't currently consume `mbjt`.

- **Suspicion 3: floating-body source diode.** With `jss=3.4089e-7`
  and the body floating, M2's body-source junction can forward-bias and
  conduct. If our diode model has a wrong Js or wrong area scaling,
  this contributes Id at low VG2 where channel and BJT are both off.

- **Suspicion 4: K1/ETAB CSV overrides on M1 aren't reaching the Vth
  body-effect formula correctly.** At VG1=0.6 the M1 Vth via body effect
  matters because Vsint floats.

## Specific questions for the oracle

1. **Most likely root cause** of a residual that is concentrated at
   low-VG2 (M2 toward cutoff), best at high-VG2 (M2 strongly on)?
   Rank our 4 suspicions by probability with a one-sentence reason each.

2. **Concrete test we can run tonight** to confirm or eliminate each
   suspicion. We have ~6h of compute and the venv is ready.

3. **mbjt in SPICE** — is it the device multiplicity factor `m=` (which
   scales every current and the area) or something else? Cite a SPICE
   reference if you remember one.

4. **Floating-body diode current at VG2 ≈ 0**: in a 2T NS-RAM topology
   where Vsint floats (M1 source = M2 drain, body P shared), what's
   the dominant current path when M2 is in cutoff? Is it really BJT
   collector current (Sebas's "complementary bipolar") or could it be
   the body-source diode forward-biasing as Vb collapses?

5. **Sanity check on Sebas's CSV**: at VG1=0.6 / VG2=0.0, his CSV says
   NFACTOR = 6.0, K1 = 0.41825, ETAB = 2.5 (for M1). Our output at this
   bias is 5.6 decades below measurement. If you plug those values into
   the BSIM4 v4.8.3 formulas by hand, does the predicted Id make sense
   at Vd=2.0 V? (We'll send you our forward sim's intermediate dump if
   you ask — Vth_eff_M1, Vth_eff_M2, Vsint, Vb, Idsa, Iii, Ic_BJT — but
   start with the formulas.)

## What we'd like back

A short ranked diagnosis (1–2 paragraphs per suspicion, not a textbook),
plus 3–5 concrete test commands or code snippets we can run tonight to
collapse the uncertainty. **No fluff, no "let me know if you have more
questions"** — give us the next moves.

---

## Attached files in this packet

- `prompt.md` — this file
- `fit_vs_meas.png` — the failing plot at median 2.40 / p90 4.83
- `summary.json` — run summary
- `2Tcell_BSIM_param_DC.csv` — Sebas's per-bias parameter table
- `M1_130DNWFB.txt`, `M2_130bulkNSRAM.txt` — the two model cards
- `nsram_bsim4_port.zip` — the forward simulator (compute_dc, leak,
  bjt, body-diode, model-card parser, the 2T cell wrapper, the
  arclength solver, two-model refactor)
- `validation_scripts.zip` — z91f, z91g, z91d, z91e — the validation
  and fitting scripts

(See `make_packet.sh` in this folder for how the zips are produced.)
