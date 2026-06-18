# A1i — Decoding Sebas's "Complementary Bipolar Current"

**Source:** `/home/ikaros/nsram_info/schematic&modelCards/2tnsram_simple.asc`
(plus `parasiticBJT.txt`, `PTM130bulkNSRAM.txt`).

## 1. Behavioural elements — verbatim inventory

`grep`-ing the ASC for `bv`, `bi`, `B1`, `.subckt`, `.func`, `.lib` returns
**nothing**. The schematic contains exactly four primitive devices:

| Inst | Symbol | Value / model | Connections (D,G,S/B,Bulk) |
|------|--------|---------------|----------------------------|
| M1   | `nmos4`| `NMOS`, `l=Ln, w=Wn`         | D=`Din`, G=`G`,  S=`Sint`, B=`B` |
| Q1   | `npn`  | `parasiticBJT`, `area=1u`     | C=`D` (=Din), B=`B`, E=`0` (GND) |
| M2   | `nmos4`| `NMOS`, `l=Ln*10, w=Wn`       | D=`B`,  G=`G2`, S=`0`,    B=`0` |
| C1   | `cap`  | `'CBpar'` (=1 fF), Rser=1 mΩ  | between `B` and `0`              |

`.param Ln=0.18u  Wn=0.36u  CBpar=1f`
`.inc PTM130bulkNSRAM.txt   .inc parasiticBJT.txt`

There is **no B-source, no behavioural current, no sub-circuit.** The
"complementary bipolar current" Sebas refers to is *not* a custom
expression — it is simply the **collector current of Q1 (parasiticBJT)**,
fired by the **BSIM4 built-in impact-ionization current** (`alpha0`,
`beta0`) of M1, which charges the floating body node `B`.

## 2. Physical interpretation of each piece

* **M1 (NMOS, BSIM4)** — channel transport plus *intrinsic II generation*:
  `Iii = (alpha0/L) · (Vds−Vdsat) · exp[−beta0/(Vds−Vdsat)] · Ids`
  (BSIM4 manual eq. for `Iii`, routed to the body node). Card uses
  `alpha0 = 7.83756e-5`, `beta0 = 18`.
* **Q1 (NPN, model `parasiticBJT`)** — the lateral parasitic bipolar.
  `is = 5e-9`, `bf = 10000`, `va = 100`, `nc = 2`. Emitter tied to GND,
  base = floating P-body, collector = drain. *This is the "complementary
  bipolar current"* — when V(B) climbs above ~0.6 V, Q1 turns on and
  pumps a large `Ic = β·Is·exp(Vbe/Vt)` from D to GND, in addition to
  the BSIM4 channel current. That is the "full swing of the firing
  mechanism".
* **M2 (long NMOS, l=10·Ln)** — VG2-controlled *body-discharge*
  transistor. When `VG2 > Vth`, M2 sinks B→GND, killing the firing
  (low-VG2 regime = leaky body = no firing — exactly our diagnostic).
* **C1 (1 fF)** — parasitic body capacitance for transient charge
  retention.

## 3. Mathematical form

```
I_complementary(node D → node 0) = Q1.Ic
   = Is · ( exp(V(B)/Vt) − exp(V(B)−V(D))/Vt) ) · (1 + V(D)/Va) / qb
   ≈ 5e-9 · exp(V(B)/0.02585)            # forward-active, V(D)>>Vt
```

with body charge balance

```
C1·dV(B)/dt = I_ii(M1)                                # source: BSIM4 II
            − I_BE(Q1)                                # sink: BJT base
            − I_DS(M2, VG2, V(B))                     # sink: VG2 pull-down
```

So Sebas's "complementary current" is **not a hand-coded B-source**; it
is the standard Gummel–Poon Ic of `parasiticBJT`, *gated by* whether the
BSIM4 II current can outrun the M2 leakage path.

## 4. PyTorch port — how to add it

We already have the BSIM4 `Iii` term in `compute_iimpact`. The missing
piece is the **NPN collector current path D→GND**, plus M2's pull-down.
Concretely:

* **No new free parameters needed.** Hard-code Gummel–Poon constants from
  `parasiticBJT.txt` (`Is=5e-9, Bf=1e4, Va=100, area=1e-6 ⇒ Is_eff=5e-15 A`).
* Add `compute_complementary_bjt(Vd, Vb, Vt, params)` returning
  `Ic = Is_eff*(exp(Vb/Vt) − exp((Vb−Vd)/Vt))*(1+Vd/Va)`.
* In `_eval_mosfet`, add `Ids_total = Ids_bsim + Ic_bjt`.
* In the body-ODE update (or DC Newton solve), replace the existing
  "Ibody = Iii" with `Ibody = Iii − I_BE(Q1) − I_DS(M2,VG2,Vb)` where
  `I_BE = Is_eff·exp(Vb/Vt)/Bf` and `I_DS(M2)` is the same BSIM4 call
  with `l=10·Ln`, `w=Wn`, `Vgs=VG2`, `Vds=Vb`.

## 5. Numerical sanity at (Vd=1.5, Vsint=0.306, Vb=0.342, VG1=0.6, VG2=0)

```
Vt = 0.02585 V
exp(Vb/Vt)            = exp(0.342/0.02585) = exp(13.23)  ≈ 5.6e5
exp((Vb−Vd)/Vt)       = exp(−1.158/0.02585)= exp(−44.8)  ≈ 4e−20
Is_eff = Is·area      = 5e-9 · 1e-6        = 5e-15 A
Ic ≈ 5e-15 · 5.6e5 · (1+1.5/100) ≈ 2.8e-9 A
```

→ **~2.8 nA**, squarely in the "few-nA" range needed to charge `CBpar=1 fF`
on µs timescales and explain the measured firing onset. This closes our
6-decade gap: at Vb≈0.342 V the BSIM4-only current is sub-fA, but Q1
delivers ~3 nA — a ~10⁶× boost, exactly the missing factor.

## 6. Verdict

There is **no behavioural current source** in Sebas's schematic — the
"complementary bipolar current" is the **collector current of the NPN
`parasiticBJT` Q1 wired D-to-GND**, fed by **BSIM4's native impact-
ionization (`alpha0`,`beta0`)** charging the floating body, and gated by
the **VG2-controlled long NMOS M2** that bleeds the body down. We simply
need to add a Gummel-Poon Ic term plus the M2 body-discharge path; no
new free parameters.
