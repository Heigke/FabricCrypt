# O9 — Where is pyport's 5-11× sub-VT excess coming from?

We have a **PyTorch port of BSIM4 v4.8.3** (in `nsram/bsim4_port/`) that
we've been debugging against ngspice-42 ground truth. After 8 hours of
narrow-the-suspect sweeps, we've pinned down a **clean, contradictory
phenomenon** we can't explain. We need an independent read.

## The phenomenon (one-line)

On an **isolated M2 nMOS** (Sebas's 130 nm thick-ox, L=1.8µm, W=0.36µm),
pyport's drain current in **deep subthreshold** is **5–11× too high**
vs ngspice using the same SPICE card. The ratio drops monotonically
from 10.8× at Vgs=0.30 V to 1.86× at Vgs=0.70 V, crossing 1.0× at the
constant-current Vth.

## What we've ruled out

1. **Vth-aggregator formula** (dc.py:349-403): pyport replicates b4ld.c
   §1161-1164 line by line. Hand-derive of every term agrees with
   pyport's compute_dc.Vth to within rounding. NOT the bug.
2. **vth0 binning**: pyport's `scaled["vth0"]` = 0.484 V matches the
   exact b4temp.c §741 formula `vth0 + lvth0/L + wvth0/W + pvth0/(L·W)`
   with binunit=2 (raw 1/L). Card has wvth0=-1.66e-8 → -46 mV W-shift.
3. **voff binning**: pyport's `scaled["voff"]` = -0.155 V (binned from
   card -0.137 V via wvoff=-5.6e-9). Matches b4temp.c §807-810.
4. **voffl (1/L correction)**: pyport applies `voffcbn = voff + voffl/Leff`
   per b4temp.c §1404. Verified.
5. **Vgsteff bridge sign / formula** (dc.py:440-481): replicates
   b4ld.c §1278-1336 exactly, including 3-branch numerator and
   denominator. Verified by reading b4ld.c side by side.
6. **Theta0_n in `n` aggregator** (A.5.f, completed): half-factor fix
   already applied per b4ld.c §1133. Slope mismatch dropped from 12%
   to 5.2%.
7. **phi formula** (A.5.c, completed): now `phi = 2·Vt·log(NDEP/ni) +
   phin` per b4temp.c §289. Verified.
8. **Sign error on voffcbn**: ruled out by C source (b4ld.c:1305 says
   `T1 = voffcbn − (1−mstar)·Vgst`, which pyport does).
9. **noff missing**: card has noff=1.0 (default), so harmless for
   THIS card. But yes, noff is missing from the port.

## What we've confirmed via brute-force voff sweep

Holding ngspice as ground truth and sweeping pyport's effective voff:

| voff_eff | r(Id, py/ng @ Vgs=0.30) |
|----------|--------------------------|
| -0.337   | 123×                     |
| -0.137 (card) | 10.8×              |
| -0.117   | 6.0×                     |
| -0.077   | 1.8×                     |
| -0.057 (extrap) | ≈ 1.0×            |

A **+60-80 mV positive shift on voff** would close most of the gap.
But we cannot identify any term in BSIM4 v4.8.3 that ngspice computes
"effectively 60-80 mV less negative" than pyport.

## What ngspice exposes (and doesn't)

`@m1[vth]` returns **0.6274 at every operating point** (we tested 18
(Vgs, Vds) pairs). Doesn't track Vbs/Vds. So it's the cached scalar
`pParam->BSIM4vth0` (binned vth0 + temp shift), NOT the per-bias
field Vth. We CAN'T directly probe ngspice's per-bias Vth or Vgsteff.

## The deeper contradiction (just discovered)

On the **full 2T NS-RAM cell** (M1 series-stacked with M2, parasitic
NPN at floating body), the same bug propagates to pyport's drain Id
being too LOW vs measurement, not too high. Brute-force voff shifts
confirm:

| Lever              | Best median log-RMSE (33 biases) |
|--------------------|----------------------------------|
| Baseline (no shift) | 1.840 dec                       |
| M2 voff -0.20 V    | 1.495 dec                        |
| M1 voff +0.20 V    | 0.894 dec                        |
| Joint M1+0.20, M2-0.20 | **0.846 dec (best)**         |
| Phase A target     | < 0.5 dec                        |

So on isolated M2 we want voff +60 mV (less subVT current).
On 2T cell, M2 actually wants -200 mV (more), M1 wants +200 mV (less).
These compensate two bugs against each other.

## The question for you

**Where in BSIM4 v4.8.3 does pyport's compute_dc compute sub-VT current
5-11× too high vs ngspice on isolated M2, given that the binning, voff,
phi, Theta0_n, and Vgsteff formulas all match the C source verbatim?**

Suspects we haven't ruled out (from us):
1. **`cdep0`** in temp.py:329 (`sqrt(q·epssub·NDEP·1e6/2/phi)`) —
   could units be subtly off?
2. **`mstar` computation**: `0.5 + atan(minv)/π` with minv=0 gives
   mstar=0.5. Standard?
3. **`n` aggregator** (dc.py:360-381): the rational regularizer
   when tmp4 < -0.5 — does ngspice flag the threshold differently?
4. The `here->BSIM4vfb` path (b4ld.c §1495) when card has explicit
   k1/k2 (which it does): vfb = `type·vth0 - phi - k1·sqrtPhi`. Does
   pyport use this, and does it propagate through?
5. The Vgsteff bridge `T9` denominator (dc.py:480): `T9 = mstar +
   n·T3v` where T3v = (coxe/cdep0)·exp(T2_off). If `coxe/cdep0` ratio
   is off by 2-3× we'd see exactly this kind of subVT excess.

**Please read the attached files (compute_dc.py, temp.py, b4ld.c
excerpts, the M2 card) and tell us which BSIM4 internal pyport is
computing wrong.** A 60-80 mV-equivalent error in voffcbn that can't
be traced to voff/voffl is the specific signature.

We need the actual root cause, not "try X" — we've tried lots of X.

## Files attached

- `compute_dc_vth.py` — extract of dc.py:140-403 (Vth + n aggregator)
- `compute_dc_vgsteff.py` — extract of dc.py:440-481 (Vgsteff bridge)
- `temp_vgsteff_inputs.py` — extract of temp.py:300-330 (mstar/voffcbn/cdep0)
- `b4ld_vth_excerpt.c` — ngspice ground truth for Vth + Vgsteff
- `b4temp_voff_cdep0.c` — ngspice ground truth for voffcbn / cdep0
- `M2_card.txt` — the SPICE model card with binunit=2
- `id_vgs_isolated_m2.png` — the divergent curves
