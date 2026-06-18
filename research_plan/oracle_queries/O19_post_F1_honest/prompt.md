# O19 — Honest physical-Bf result + F1 closure validation

## Context

You (and two other oracles) reviewed the NS-RAM PyTorch BSIM4 port
in O18 (yesterday) and returned a unanimous **FIX verdict**. The
biggest single fix you flagged: `bjt.Bf = 2×10⁴` is non-physical
(real 130 nm parasitic NPN gain is 10–100). The 0.80-dec median
log-RMSE we were proud of was a **fitting hack**.

We executed F1 (the device-physics fix) as you specified. **Result:
honest physical-Bf number is 1.31 dec, not 0.80.** We added a
new mechanism (`cfg.iii_body_gain` = γ, the lateral-NPN base
injection) to recover some of the lost snapback at physical Bf,
but the model's 2T-cell-with-separate-Gummel-Poon-NPN structure
cannot match silicon at physical Bf.

## What we did (concrete)

1. **Clamped Bf = 100** (F1.a). Result: VG1=0.4 catastrophe row
   improves 2.83 → ~1.35 dec, but VG1=0.6 V row breaks 0.78 → 2.21
   (snapback dies because Bf=100 doesn't amplify the NPN base
   current enough).

2. **Added Iii→Vb lateral injection mechanism (F1.b).** New term
   in body KCL: `iii_gain * (m1["Iii"] + m2["Iii"])` instead of
   just `m1["Iii"] + m2["Iii"]`. Default γ=1.0. Physical motivation:
   the same channel impact-ionisation that creates Iii also
   injects holes laterally into the parasitic NPN base, distinct
   from the diffusive bulk path.

3. **Swept γ ∈ {1, 1e3, 1e5, 1e7, 1e9, 1e11} at Bf=100.** Optimum
   γ = 1×10⁵: overall median 1.31 dec, p90 2.33.

4. **Could NOT push below 1.0 dec without going non-physical.**
   Per pre-registered halt criterion in the M3b plan, we stopped
   adding parameters and accepted the honest physical-Bf result.

## Honest headline (replaces the 0.80-dec claim)

| metric | brief v4.1 (Bf=5e4) | post-stage6 (Bf=2e4) | physical (Bf=100, γ=1e5) |
|--------|---------------------|----------------------|---------------------------|
| median log-RMSE | 1.00 | 0.80 | **1.31** |
| p90 log-RMSE   | 2.90 | 2.58 | 2.33 |
| Bf physical?   | no   | no   | **yes (≤100)** |
| n_evaluated    | 25/33 | 25/33 | 25/33 |

## Plot: see `F1_physical_fit.png` and `stage6_unphysical_fit.png`

These are the two plots side-by-side. The unphysical Bf=2e4 plot
(stage6) hides the catastrophe by overdriving the NPN. The physical
Bf=100 + γ=1e5 plot (F1) is honest: VG1=0.4 V row predictions now
follow measurement shape, but the snapback magnitudes at VG1=0.6 V
no longer match because the lateral-NPN-coupled-to-channel mechanism
(real silicon) is structurally absent from the 2T port.

## Three questions

### 1. Is the honest "1.31 dec at physical Bf" the right number to communicate?

We're walking back our prior 0.80-dec claim. The right thing for the
brief addendum is to present 1.31 dec as the validated result. Is
this defensible? Or are we being too conservative — is there a
parameter-sweep we missed that could legitimately reach <1.0 dec
without going non-physical?

### 2. Is the lateral-NPN-as-channel-current restructure the next step?

We hypothesise that the real silicon mechanism is "channel current
acts as the parasitic-NPN collector current at high Vds" — i.e.,
the NPN is NOT a separate device with its own Bf; it's the lateral
amplification of the M1 channel itself in the snapback regime. To
model this faithfully, we'd remove the Gummel-Poon NPN and add
a Vds-dependent multiplicative gain on `Ids_M1` that activates only
in the snapback regime (Vds > some threshold).

Is this the right diagnosis? If we make that structural change,
what new failure mode should we expect?

### 3. Should we dispatch the brief addendum now with 1.31 dec, or first run F2/F4 full sweeps + F3 z142?

Three concrete options:
- **A.** Send the brief addendum NOW with 1.31 dec headline + the
  M3b fix plan as the "what's next" appendix. Be honest immediately.
- **B.** Hold the addendum until F2 full ngspice grid + F4 full
  2T-cell cross-check + F3 z142 topology rerun all close. Then
  send a coherent package.
- **C.** Restructure the model first (the lateral-NPN refactor in
  question 2), refit, then send.

Which is most defensible to a reviewer?

## Output format

Three numbered answers, terse. Plus a final box:
- **Recommendation:** A / B / C, with one sentence on why.
- **Critical risk we missed:** what we're still over-claiming, even
  with this honest correction.

Be brutal. We over-claimed once; we want to not do it twice.

## Attached files

- `F1_physical_summary.json` — the new honest numbers
- `F1_physical_fit.png` — what the honest fit looks like
- `stage6_unphysical_fit.png` — the prior unphysical hack for comparison
- `M3a_addendum_2026-05-03.md` — current addendum (banner-flagged
  DRAFT) with the proposed fixes already applied (Section 3 deleted,
  caveats added)
- `M3b_fix_plan_2026-05-03.md` — full F1–F6 plan
- `probe_v2_finding.md` — the original VG1=0.4 V diagnosis
