# O21 — Vb-clamp structural finding invalidates M3c.2 premise

## Story so far (compressed)

You (and two other oracles) reviewed this NS-RAM PyTorch port four
times in the past 48 h:
  - **O18** (3-of-3 FIX): Bf=5×10⁴ in the brief was non-physical
  - **O19** (2:1 HOLD): the η-bounded refactor's γ replacement was
    itself a fudge factor; bound η ∈ [0, 1]
  - **O20** (3-of-3 unanimous, 4 h ago): M3c.2 path B (AUGMENT,
    keep BJT, drive Ib via lateral pair) is the right move; topology
    rec ER_SPARSE; send M3b addendum now with caveats

Per O20 unanimous, I implemented M3c.2 path B AND path C (toggle for
M(Vbc)·Ids avalanche multiplier). Both paths committed to
`nsram/nsram/bsim4_port/nsram_cell_2T.py` with passing gate tests
(η_lat=0 and use_lateral_collector=False both reproduce F1.v2
bit-identical).

**Then I tested both paths empirically and found O20 was wrong.**

## What we found

### Test setup
Ran `forward_2t` at Vd=2.0 V, Bf=100 honest, across 5 representative
biases. For each bias compared: F1.v2 baseline (no path B/C) vs
path C activated with BV=3 V.

### Result table

| bias              | Ids_M1   | Ic_Q1    | Id (off)   | Id (path C, BV=3) | ratio |
|-------------------|----------|----------|------------|-------------------|------:|
| VG1=0.8 VG2=0.50  | 3.2e-9   | 2.0e-8   | 2.32e-8    | 2.35e-8           | 1.011 |
| VG1=0.6 VG2=0.30  | 7.0e-12  | 1.6e-8   | 1.64e-8    | 1.64e-8           | 1.000 |
| VG1=0.4 VG2=0.20  | 2.7e-13  | 1.6e-8   | 1.64e-8    | 1.64e-8           | 1.000 |
| VG1=0.6 VG2=0.00  | 1.1e-15  | 1.6e-8   | 1.64e-8    | 1.64e-8           | 1.000 |
| VG1=0.2 VG2=0.10  | 5.1e-15  | 1.6e-8   | 1.64e-8    | 1.64e-8           | 1.000 |

### Diagnosis

  - **Ic_Q1 ≈ 1.6e-8 A across ALL biases** at Bf=100 honest — a
    hard floor set by the Gummel-Poon NPN at the equilibrium Vb
    that the body KCL produces.
  - **Ids_M1 ranges 7 decades** but is *always* ≪ Ic_Q1 — channel
    current is negligible compared to the parasitic-NPN current
    at every bias tested.
  - **Vb clamps near 0.39 V** at every bias — body diodes Ibs/Ibd
    pin Vb wherever Iii / Igidl tries to push it.
  - **Therefore both paths fail by construction:** path B's
    β·Ib_lat_pair = β·η_lat·Iii is zero where Iii is small (which
    is most biases). Path C's M(Vbc)·Ids_M1 is zero where Ids_M1
    is small (also most biases).

### Why O20 was wrong

O20's B/C verdict was based on the assumption that "η_lat·G_pair
drives Ib and is multiplied by β; with Rb you can shape snapback
onset/magnitude". That's mathematically true for any single bias
where G_pair (= iii_gain · Iii) is appreciable. Empirically:
  - At the failing biases (low VG2), Iii is 1e-15 A scale
  - β·Iii at Bf=100 → 1e-13 A
  - Ic_Q1 floor → 1e-8 A
  - The lateral pair contribution is **5 orders of magnitude
    below the floor it's trying to modulate**

The structural gap is not in NPN gain at all — it's in **Vb
dynamics**. At Bf=100, Vb is pinned by body diodes regardless of
bias, so Ic_Q1 = Is·exp(Vbe/Vt) is essentially constant across
biases. Silicon shows bias-dependent current variation, but the
honest-physical-cell pyport cannot reproduce that variation
because Vb is bias-independent in its equilibrium.

## What we want from you

Three numbered sections, brutal as the last four oracles.

### 1. Is the Vb-clamp diagnosis correct?

We claim: at Bf=100 honest, the body diodes Ibs/Ibd dominate the
body KCL, pinning Vb at ~0.39 V across all biases. This makes
Ic_Q1 ≈ const across biases, which is why the model can't
reproduce silicon's bias-dependent current.

Test the logic. Could there be a bug in the body KCL we're missing?
Or is this genuinely the structural limit of the η-bounded model?

### 2. Where is silicon's bias-dependent current coming from?

Sebas's silicon at the same biases shows orders-of-magnitude
variation in drain current. Pyport at honest Bf=100 produces a
~1.6e-8 floor. Two hypotheses:

  - **(A) Missing body-charging mechanism:** silicon has a route
    (capacitive, GIDL-driven, punch-through, lateral drift) that
    pumps Vb above the body-diode clamp at certain biases. Pyport
    captures GIDL and Iii as inflows but they're outweighed by
    body-diode outflows at Bf=100.
  - **(B) Different Bf:** silicon's actual parasitic NPN gain is
    much higher (10⁴ regime) than the 130 nm published 10–100
    range, possibly due to the unusual NS-RAM cell geometry
    (lateral, low-doping-base parasitic, not a vertical NPN).
  - **(C) Different mechanism entirely:** the dominant current at
    failing biases isn't NPN at all — it's punch-through, gate
    leakage, or impact-ionisation-induced channel multiplication
    at the M2 source pin (which pyport doesn't model
    independently from M1).

Which is most likely? What test would distinguish them?

### 3. Should we continue M3c at all?

Per `M3c2_design_decision.md` halt criterion: "if any new fit
param > 1 order of magnitude outside its physical bound, halt;
the structure is still wrong." We are at that point — the
M3c.2 path-B/C effects are 5 OoM below where they need to be.

Options:
  - **(α) New M3c.3** — Vb-charging refactor: weaken body diodes,
    add capacitive Vb storage, model lateral-NPN base resistance
    explicitly so Vb can climb in snapback. This is genuinely
    structural, no new fudges.
  - **(β) Re-examine Bf assumption** — Sebas's parasitic NPN may
    legitimately have Bf >> 100 due to the unusual lateral
    geometry. If we can ground-truth Bf at 1000+ from a single
    measurement (e.g. saturation current at one bias), then the
    Bf=2e4 chosen in z139 wasn't unphysical so much as poorly
    documented.
  - **(γ) Acknowledge model floor** — the η-bounded honest cell
    has a fundamental ~1.4 dec floor on this dataset. Ship that
    as the final result, halt M3c, redirect engineering effort.

Which is most defensible?

## Output format

Three numbered answers, terse. Plus a final box:

  - **Vb-clamp verdict:** correct / not correct / partial — one
    sentence
  - **Silicon current source:** A / B / C — one sentence
  - **M3c continuation:** α / β / γ — one sentence
  - **Critical risk we missed:** what we're still over-claiming
    even now

## Attached files

- `vb_clamp_test_output.txt` — the empirical table above (raw)
- `nsram_cell_2T_excerpt.py` — the body KCL implementation
- `M3c2_design_decision.md` — the prior decision doc
- `O20_synthesis.md` — the unanimous B verdict that this finding
  invalidates
