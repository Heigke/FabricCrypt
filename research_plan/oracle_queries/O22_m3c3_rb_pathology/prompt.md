# O22 — M3c.3 implemented; Rb scaling pathology

## Story so far (compressed)

You're the fifth oracle round on this NS-RAM 2T-cell port. The chain:
  - **O18** (FIX): Bf=5×10⁴ in brief was non-physical
  - **O19** (HOLD): η-bounded ∈ [0,1]; γ replacement was also a fudge
  - **O20** (3-of-3 unanimous B): augment, don't replace BJT;
    topology rec ER_SPARSE; send M3b addendum
  - **Empirical test** (3 hours ago): O20's path B fails. At honest
    Bf=100, Ic_Q1 ≈ const ~1.6×10⁻⁸ across all biases regardless of
    Iii. Both M3c.2 paths B and C are 5+ OoM below the floor.
  - **O21** (3-of-3 Vb-clamp / 2-of-3 α-direction): the BJT itself is
    the Vb clamp; β path empirically refuted; α (Vb-charging refactor)
    is the only coherent fix.

Per O21 + user authorisation, I implemented M3c.3 as a semi-implicit
inner solve (lighter weight than full Newton 3D, same physics):

  - New `cfg.use_local_base` toggle + `cfg.lat_Rb`.
  - Iii + GIDL + Ib_lat_pair injected at a new `Vb_local` node.
  - BJT base sees Vb_local, not Vb_global.
  - Spread resistor Rb couples Vb_local ↔ Vb_global.
  - Inner damped Newton (10 iters max) solves for Vb_local at each
    outer Newton step.
  - Default toggle=False reproduces F1.v2 bit-identical (gate passes).

## What we found

### Single-bias Rb sweep (VG1=0.4 VG2=0.2 Vd=2.0 V)

| Rb       | Vb_global | Vb_local | Ic_Q1    | Id       |
|----------|----------:|---------:|---------:|---------:|
| 1e3      | 0.3876    | 0.3876   | 1.64e-8  | 1.64e-8  |
| 1e6      | 0.3878    | 0.3876   | 1.64e-8  | 1.64e-8  |
| 1e9      | 0.5320    | 0.3852   | 1.49e-8  | 1.49e-8  |
| 1e12     | 1.6039    | 0.7523   | 2.17e-2  | 2.17e-2  |

### Multi-bias Rb scan (Rb=1e6 / 1e8 / 1e10)

| bias              | Rb=1e6   | Rb=1e8   | Rb=1e10  |
|-------------------|----------|----------|----------|
| VG1=0.8 VG2=0.50  | 2.32e-8  | 2.30e-8  | 2.53e-5  |
| VG1=0.6 VG2=0.30  | 1.64e-8  | 1.62e-8  | 8.14e-4  |
| VG1=0.4 VG2=0.20  | 1.64e-8  | 1.62e-8  | 1.12e-2  |
| VG1=0.6 VG2=0.00  | 1.64e-8  | 1.62e-8  | 1.24e-2  |
| VG1=0.2 VG2=0.10  | 1.64e-8  | 1.62e-8  | 1.23e-2  |

### Diagnosis

  - At physical Rb (1 MΩ from literature for lateral parasitic NPN),
    M3c.3 has **no observable effect**. Vb_local just tracks
    Vb_global (spread is too easy).
  - At Rb ≥ 1 GΩ, Vb_global decouples and rises to where body
    diodes' exponential turns on (~0.5 V then ~1 V then 1.6 V).
  - At Rb=1e10, **bias-dependent variation appears** (3 OoM range
    across biases) but Id magnitude is **1000× silicon scale**.
  - Vb_local at Rb=1e12 is 0.75 V, but Ic_Q1 = 2e-2 A — the inner
    solve is producing huge currents, suggesting numerical pathology.

## What we want from you

Three numbered sections, brutal as the prior four oracles.

### 1. Is the Rb=10 GΩ regime physical or numerical pathology?

Possible interpretations:
  - **(A) Physical:** lateral parasitic NPN's actual base spreading
    resistance for the unusual NS-RAM geometry IS in the 10 GΩ
    range due to extremely lightly-doped lateral base. Literature's
    100 kΩ–1 MΩ is for vertical NPNs with conventional base doping.
  - **(B) Pathological:** the inner solve is finding a different
    Newton basin where Vb_global has decoupled to follow body-
    diode exponential alone; the bias-dependent variation we see
    is the body diodes' I-V curve at Vb_global ≈ 1 V, not the
    M3c.3 mechanism we wanted.
  - **(C) Both:** Rb is large but not THAT large; the "M3c.3
    mechanism" is doing something at intermediate Rb, but our
    inner solve isn't capturing it correctly.

### 2. If A is right, how do we ground-truth Rb?

If the parasitic NPN really has Rb ~10 GΩ:
  - That makes Ib_at_local ~ Iii / spread → tiny → Vb_local can
    climb. But the data shows Vb_local stays low (~0.4 V at all Rb
    we tested). Why doesn't Vb_local rise more?
  - Would Sebas's single high-Vd snapback measurement (saturation
    Ic vs Vd at one bias) constrain Rb directly? Or do we need a
    transient base injection test?
  - Is there an ngspice / Spectre primitive that already models
    this lateral geometry, that we could compare against?

### 3. Halt vs continue M3c?

Three options:
  - **(α') Continue M3c.3 with Rb hyper-tuning** — sweep Rb on a
    finer grid, find the regime where bias-dependent variation
    appears at the right MAGNITUDE (not 1000× silicon). This is
    fitting Rb post-hoc — risk of M3a-pattern fudge.
  - **(δ) Add additional physics at Vb_local** — base-collector
    capacitance Cbc to limit |Vb_local − Vb_global| transient
    excursion; or a non-zero "leakage" diode at Vb_local that
    activates above 0.5 V to bound the local voltage.
  - **(γ) Halt M3c, ship 1.39 dec as the final result** — the
    structural gap is real but the model architecture
    (single-cell DC) cannot reach < 1.0 dec without fitted
    parameters. This is the third walk-back of M3-tier work and
    should be communicated honestly.

## Output format

Three numbered answers, terse. Plus a final box:

  - **Rb regime:** A / B / C — one sentence
  - **Ground-truthing:** what Sebas measurement would settle it
  - **M3c continuation:** α' / δ / γ — one sentence
  - **Critical risk we missed:** what we're still over-claiming

## Attached files

- `m3c3_test_output.txt` — Rb sweep + multi-bias test results
- `m3c3_excerpt.py` — body KCL implementation with M3c.3 inner solve
- `M3c3_local_base_plan.md` — the original M3c.3 plan
- `O21_synthesis.md` — the prior verdict that led to M3c.3
