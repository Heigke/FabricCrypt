# O24 — Blind-spot audit on the 0.795-dec optimum (Bf=2e4, Is=1e-9)

## Context

PyTorch BSIM4 v4.8.3 port of Sebastian Pazos's 130nm 2T NS-RAM cell with
parasitic Gummel-Poon NPN (lateral geometry, low-doping base). After O20
(M3b closure: ηIii→Vb gain), O21 (Vb-clamp diagnosis), O22 (M3c.3 Rb
pathology), and O23 (local-learning design), we believed the model had a
**structural floor at 1.39 dec** on Sebas's 33-row dataset. All five 1D
parameter knobs (Bf, β·Iii, M·Ids, Rb, Is) were independently exhausted.

**User pushback** triggered a 2D Bf×Is sweep. Result:

```
Bf \ Is        5e-9    1e-9    1e-10   1e-11   1e-12
100           1.394   1.424   1.631   1.856   1.976
1000          1.383   1.395   1.442   1.510   1.538
10000         0.858   0.862   1.328   1.427   1.405
50000         1.002   0.948   1.431   1.548   1.645
100000        1.216   1.138   1.433   1.657   1.645
```

Refined: **Bf=2×10⁴, Is=1×10⁻⁹ → 0.795 dec** (median, 33 rows). Bf>3e4
worsens monotonically. Side sweeps (area_mult, A0_MULT×VOFF, m1_diode_scale,
vnwell_Rs) all hurt.

**Defensibility**: Bf=10³–10⁵ is the published range for *lateral parasitic
NPN with low-doping base*. Vertical NPN (Bf=10–100) does not apply.
Is=1e-9 is within an order of magnitude of the BJT card's default 5e-9.

## Question

Are there **untested physical knobs or coupling regions** in BSIM4 v4.8.3
or Gummel-Poon that could push us further below 0.795 dec, before we
declare the model "good enough" and ship the headline?

Specifically:

1. **Higher-order BJT effects in Gummel-Poon** that might matter at
   lateral-NPN low-doping geometry but are usually negligible at vertical-
   NPN HBT-style designs:
   - Early effect (VAF) — currently at card default. Lateral NPN often
     has very strong Early effect due to base-width modulation. Worth a
     2D Bf×VAF sweep?
   - Knee current (IKF) — high-injection roll-off. Cell drives ~mA-range
     currents at saturation; could matter.
   - Base-emitter saturation current Ise + leakage exponent NE — extra
     current at low Vbe.
   - Substrate / vertical leakage path (Iss, ISC) — typically 0 in our card.

2. **BSIM4 secondary branches** that we might be silently missing:
   - IGCMOD (gate-channel current) — usually off for thick-ox; for 130nm
     it could be small but nonzero.
   - IGBMOD (gate-bulk current) — same.
   - DITSMOD (drain-induced threshold shift, lateral) — for short-channel
     non-quasi-static effects.
   - PFMOD / TNOIMOD — field/noise; probably irrelevant for DC fit.

3. **Cell-level coupling regions** we may not have explored:
   - Vd-dependent Rs (impact-ionisation feedback through series resistance)
   - Temperature corner — single-temperature fit only? 130nm card likely
     calibrated at 300K; no T-sweep done.
   - Body-bias on M2's Vbs — currently floating; if Sebas's silicon has
     guard-ring biasing it could matter.
   - Drain extension (Rdrain) vs source extension (Rsource) asymmetry.

4. **Solver / numerical issues** we might be hiding:
   - Newton convergence at all 33 rows confirmed, but 5/30 rows have
     residual > 1.0 dec — are those the *same* rows across (Bf, Is)
     points? If so, a structural mismatch persists at specific bias
     corners (probably high VG2, low Vd).
   - GMIN floor — currently 1e-12. Worth a sweep.
   - Floating-body Vb starting guess in solver — sensitivity to initial
     condition.

5. **Cross-validation gap**: F2 ngspice 180-pt cross-val was completed
   *before* the 2D breakthrough. Has the new optimum (Bf=2e4, Is=1e-9)
   been validated against an independent simulator yet? If not, that's
   a critical next step.

## Asks

1. Rank these by likely impact (highest first), specifically for a
   **lateral parasitic NPN at 130nm with low-doping base in floating-body
   2T-NS-RAM topology**.
2. For each, give a parameter range to sweep and a heuristic for "if
   this knob matters, you'll see X in the residual pattern".
3. Flag any **coupling** between knobs (e.g. is VAF correlated with Bf
   in a way that makes 1D sweeps misleading the same way Bf×Is was?).
4. Specifically address: do you expect **more than ~0.2-0.3 dec
   improvement** is reachable without measuring Sebas's silicon
   directly? Or is 0.795 close to the irreducible model-vs-silicon
   mismatch?

Be concrete and brief. Cite BSIM4 v4.8.3 manual section / Gummel-Poon
parameter when nontrivial. <600 words.
