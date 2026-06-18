# O25 — Three null knobs after VAF; what architecture change next?

## Backstory

We are fitting a PyTorch BSIM4 v4.8.3 port of Sebastian Pazos's 130 nm 2T
NS-RAM cell (M1 = thin-ox driver, M2 = thick-ox storage; lateral parasitic
NPN body→drain via Gummel-Poon between M1.S and M1.D, body diode to
n-well, η-bounded Iii→Vb collection efficiency).

In O24 you ranked the next-most-likely missing knobs after VAF:
1. **VAF (forward Early)** — confirmed, drove 0.795 → 0.657 dec
2. **IKF (knee current)** — TESTED, NULL
3. **ISE/NE (B-E recombination)** — TESTED, NULL
4. **RDSMOD/PRWG (S/D resistance Vg-dep)** — TESTED, NULL

We have hit a clear structural floor at **0.654 dec** (5×5 grids each
parallel-fit on the full 33-row Sebas dataset).

## What's null and how

### F6.v5 — IKF × Va (at Bf=9000, Is=1e-9)

```
Ikf↓ \ Va→     0.4    0.55    0.7    0.85    1.0
1e+30 (off)  1.048  0.657  0.659  0.677  0.694   ← floor
1e-1  100mA  1.169  1.000  1.032  1.060  1.083
1e-2  10mA   1.321  1.243  1.273  1.291  1.304
1e-3  1mA    1.233  1.229  1.240  1.249  1.258
1e-4  0.1mA  1.415  1.445  1.476  1.501  1.522
```
Best at Ikf=∞ (effectively disabled). Every finite Ikf monotonically
worsens. Sebas data appears to be sub-mA — high-injection knee never
engaged.

### F6.v6 — ISE × NE (at Bf=9000, Va=0.55, Is=1e-9)

```
Ise↓ \ Ne→   1.2    1.5    1.7    2.0    2.3
0           0.657  0.657  0.657  0.657  0.657
1e-12       0.675  0.657  0.657  0.657  0.657
1e-11       0.999  0.656  0.657  0.657  0.657
1e-10       1.213  0.763  0.663  0.657  0.657
1e-9        1.611  1.307  0.936  0.676  0.656
```
Best at Ise=1e-11, Ne=1.5 → 0.656 (1 mdec, within noise). For Ne ≥ 1.7
ANY Ise gets absorbed into ideal Is term. For Ne ≤ 1.5 large Ise breaks
fit catastrophically. Conclusion: B-E non-ideal recombination has no
measurable effect at the honest BJT optimum.

### F6.v7 — PRWG × Rdsw (M1 BSIM4)

```
PRWG↓\Rdsw→   30     100    300    1000   3000
0.0         0.657  0.657  0.656  0.655  0.654
0.25        0.657  0.657  0.656  0.655  0.654
0.5         0.657  0.657  0.656  0.655  0.654
0.75        0.657  0.657  0.656  0.655  0.654
1.0         0.657  0.657  0.656  0.655  0.654
```
**PRWG row-identical** — no Vg-dep S/D effect at all. Rdsw monotonically
3 mdec from 30 → 3000 Ω·µm but all rows identical → it's pure series-R
attenuation, not bias-dependent.

## Persistent residual cluster (unchanged across all 4 sweeps)

The 5 worst rows are ALL at **VG1 = 0.40 V** (std=0.000 across sweeps),
VG2 ∈ [0.10, 0.30]:

```
Top-5 worst rows at every (Bf, Is, Va, Ikf, Ise, Ne, PRWG, Rdsw) tested:
  VG1=0.40 VG2=0.30  log_rmse ≈ 2.7-2.9
  VG1=0.40 VG2=0.25  log_rmse ≈ 2.6-2.8
  VG1=0.40 VG2=0.20  log_rmse ≈ 2.4-2.6
  VG1=0.40 VG2=0.15  log_rmse ≈ 2.3-2.5
  VG1=0.40 VG2=0.10  log_rmse ≈ 2.2-2.4
```

Best rows are at **VG1 ∈ {0.20, 0.60}** with VG2 unconstrained — corner
regimes (cell firmly off or firmly on) fit fine; the **transition at
VG1=0.40** doesn't.

Physical interpretation: M1 weak inversion (Vth ≈ 0.35 V → VG1=0.40 is
50 mV above threshold) coupled with M2 in subthreshold-to-saturation.
This is exactly where the parasitic NPN ignites and starts to dominate
Id, but no parameter we've tested resolves the residual.

## Honest physical optimum (current best 0.654 dec)

- **Bf = 9000** (lateral parasitic NPN, low-doping base; literature
  range 1e3–1e5)
- **Va = 0.55 V** (forward Early; lateral NPN with narrow base, much
  lower than vertical 100V default)
- **Is = 1e-9** (within OoM of card default 5e-9)
- **η ∈ [0, 1]** Iii→Vb collection, currently η=0.6
- **Vb-clamp** at honest physical limit (BJT exponential base draw
  clamps Vb)
- **mbjt per-row scale** from Sebas card (0.001 to 1.0)
- All other BJT non-ideality params off / default.

## QUESTION

Given the four null sweeps above, what **MODEL ARCHITECTURE change**
(not parameter value) is most likely to break the 0.654-dec floor?

Specifically rate these candidates (1=most likely, n=least):

1. **Two-NPN model**: split parasitic into separate vertical (Bf=10–100,
   Va=100) + lateral (Bf=10⁴, Va=0.55). Currently we have one
   lumped NPN. Lateral dominates at honest params but at VG1=0.40
   ignition point a vertical contribution might shape the slope.

2. **Explicit body-network**: replace single floating Vb with Rb-Cb
   network: rbody (intrinsic body resistance) + sub diode + opt. p-sub
   contact. Currently Vb is a single node from KCL. Adding distributed
   resistance could shape Vb dynamics in the ignition corner.

3. **Bias-dependent η_lat (Iii→Vb gain)**: replace constant η with
   η(Vbe, Vds) — physically, collection efficiency drops as Vbe rises
   because the NPN base-current sinks more of the Iii. We tested
   η ∈ [0, 1] as constants in O20 but never as bias-dependent.

4. **Add lateral parasitic body diode** between M1.S and Vb (currently
   modeled only at body-to-n-well via vnwell). Sebas's silicon may have
   a forward-biased path we're missing at VG1=0.40 / mid-VG2.

5. **Bias-dependent Bf (Bf×f(Vbe))**: Lateral NPNs at saturation can
   show Bf roll-off without high-injection (current crowding). Could
   shape the ignition corner where Vbe jumps.

6. **Quasi-2D body charge** (split Vb into Vb_M1.S, Vb_M1.D, with
   resistive coupling): in long lateral devices the body charge isn't
   uniform; the body voltage seen by the NPN base differs along the
   channel. Currently we have one Vb.

7. **Temperature corner** (multi-T fit): Sebas data is at one T; could
   one of those rows have been measured at slightly elevated junction
   temp, biasing residual?

8. **Something else we haven't considered**?

For your top pick:
- What residual signature would confirm it (which (VG1, VG2) rows
  should improve, and how much, if this is the right architecture)?
- What's the minimum implementation effort (single-node solve still
  applicable, or does it require N≥3 Newton)?
- What's the realistic dec improvement bound — can architecture buy
  another 0.1-0.2 dec, or are we at the silicon-data floor (~0.5 dec)?

Be concrete. Cite physical reasoning. <600 words. The audience is
the user (Eric) who needs a concrete next-implementation directive
and is allergic to "many possibilities — try them all".
