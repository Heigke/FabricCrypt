# T5 — Order-of-Magnitude Bookkeeping: What Supplies 4e-9 A to V_b?

**Date:** 2026-05-14
**OP:** VG1=0.6 V, V_d=2.0 V, V_b=0 V (clamped), V_s=0, M1 = NMOSdnwfb L=0.13µm W=0.5µm
**Target:** Ic_si ≈ 4e-5 A measured → with β=1e4 → required base-current Ib ≥ 4e-9 A
**Oracle:** ngspice-42 single-transistor probe with Sebas 2026-04-22 cards (production + LALPHA0_FIX) + closed-form Kane/Shockley/BSIM4 cross-checks

## ngspice ground truth at exact OP

```
Card variant      Id [A]      ISUB [A]      IGIDL [A]     IBD [A]
production         3.92e-6     0.00 (eff.)   1.21e-24      -2.0e-15
LALPHA0_FIX        3.92e-6     3.21e-4       1.21e-24      -2.0e-15
                   ^^^^^^^^^   ^^^^^^^^^^^   ^^^^^^^^^^^   ^^^^^^^^^^
                   matches     >> 4e-9 A     dead 1e15×    dead 1e6×
                   silicon
```

i(vb) terminal current at LALPHA0_FIX card: **3.21e-4 A** — BSIM4 already routes 80,000× more current to the bulk node than the 4e-9 A target.

## Bookkeeping table

| # | Mechanism | Formula | Realistic params | I at OP [A] | Meets 4e-9? | Notes |
|---|---|---|---|---|---|---|
| 1 | BSIM4 IIMOD production | (α₀+α₁L)/L · ΔV · exp(–β₀/ΔV) · Ids | α₀=7.84e-5, β₀=19 | **~3e-5** (analytic; ngspice clamps to 0 here) | ✓ (7500×) | Iimod term in M1_130DNWFB.txt |
| 1b | BSIM4 IIMOD FIX card | same | α₀=7.84e-4, β₀=19 | **3.21e-4** (ngspice) | ✓ (80 000×) | M1_130DNWFB_LALPHA0_FIX.txt — what pyport currently uses |
| 1c | BSIM4 IIMOD physically tuned | same | α₀=1e-3, β₀=5 | 3.4e-2 (sat caps it lower) | ✓ saturation regime | More realistic 130nm; exp un-suppressed |
| 2 | HCI fraction to bulk traps | P_inj·Iii | P_inj=1e-4 | 1.1e-8 (off 1b) … 3.2e-8 (off 1b) | ✓ (3–8×) | Hot-carriers escaping to body |
| 3 | GIDL @ drain (Sebas card) | AGIDL·W·(Vdg–EGIDL)³·exp(–BGIDL·tox/(...))/CGIDL | A=1.99e-8, B=1.62e9, E=0.91, C=6.3 | **1.21e-24** (ngspice) | ✗ short 1e15× | Vdg=1.4 too small; exp(–4.6e6·tox/0.49)=1.75e-6 then cubed by power-3 also kills it. EGIDL=0.91 is barrier; activates above V_dg ≈ 3 V |
| 3b | GIDL max-plausible AGIDL×50 | same form, A=1e-6 | 1.6e-20 | ✗ short 1e11× | Pre-exp can't fix exponential suppression |
| 4 | BTBT well-body (Kane) | A·E²·exp(–B/E)·q·W_dep·A_junc | A=3.5e21, B=2.25e7, Nbody=1.7e17, V_rev=2 V | **2.8e-27** | ✗ short 1e18× | E_max=3.6e5 V/cm; Kane needs ~1e6 V/cm. Reverse-bias too low |
| 5 | Avalanche well-body junction | M(V)·Js·A; M=1/(1–(V/BV)⁴), BV=12 | V_rev=2, BV=12 | **2.5e-21** | ✗ short 1e12× | M≈1.0008; far below knee |
| 6 | Self-heat ionization (naïve) | Arrhenius boost Ea=0.4 eV, ΔT=50K | 9× factor on Iii | ~1e-4 (from FIX) | ✓ but wrong sign | IIMOD ↓ with T in reality (phonon scattering). ARRHENIUS sign is INVERTED for impact ionization. Cannot be the source. |
| 7 | BJT regenerative latch | Is·exp(Vbe/Vt), Is=4.5e-20 A | Vbe=0.5 needed | 1.1e-11 | ✗ short 360× | Consequence not cause — needs V_b > 0.5 V already, i.e. needs another source to elevate V_b first |

## Verdict (1 paragraph)

The only mechanism within physical headroom is **BSIM4 IIMOD** (channel impact ionization). It supplies between 3e-5 A and 3e-4 A to the bulk node at the OP — between 7 500× and 80 000× the 4e-9 A required to trigger the parasitic NPN base current. GIDL, well-body BTBT and well-body avalanche are 10–18 orders of magnitude too small at V_d=2 V and the floating-body bias range (they only become relevant at V_d≥3.5 V or with Nwell pinned far above body). Self-heating boosts IIMOD only with the wrong sign (impact ionization weakens with T due to phonon scattering, so the Arrhenius framing in the prompt is inverted). The BJT-latch term is regenerative, not initiating. **Therefore the 4e-9 A requirement is not a missing-physics problem — it is a routing / efficiency problem inside pyport's residuals.** The IIMOD ngspice number is so much larger than needed (i(vb)=3.2e-4 A vs 4e-9 A target) that the η ∈ [0,1] collection-efficiency factor introduced in the M3b walk-back (post-O19) needs to fall to ~1e-5 to match the ngspice "moves V_b up to 0.5 V then BJT fires" behaviour — but instead pyport's current default η is too low *and* β=9000 in the production BJT card multiplies it back wrong. The correct one-line fix is on the η side, not adding new physics terms.

## Most likely Mario NS-RAM mechanism

**Iii → body via η × IIMOD**, exactly the path pyport already implements. Mario's slides show V_b rising into the 0.4–0.7 V range during the snapback knee — fully consistent with Iii of order 1e-7 to 1e-5 A charging the floating body until the parasitic NPN fires. No other candidate mechanism in this table sits within four orders of the required current. The candidates that DO matter for **transient shape** (per T4 oracle synthesis) — N1 oxide traps, N2 SRH-TAT — are sub-percent corrections on top of IIMOD, not alternative current sources for the DC base current.

## Recommended one-line ADD to pyport residual

Not "add a new physics term" — **gate / clamp the existing IIMOD path:**

```python
# in nsram_cell_2T._residuals, where Iii is collected onto V_b:
Iii_to_body = cfg.iii_gain * iii_to_body_factor * (m1["Iii"] + m2["Iii"])
# NEW: enforce upper bound = realistic collection efficiency at V_b<0.6 V
# Mario silicon shows V_b plateaus near 0.55 V; cap Iii_to_body so it
# stops pumping V_b once forward BE diffusion takes over.
Iii_to_body = Iii_to_body * torch.sigmoid((0.55 - Vb) / 0.030)  # 30 mV softness
```

This costs **zero new parameters** in the BSIM4 card, only one cfg knob (V_b clamp ≈ 0.55 V, slope ≈ 30 mV). Physically it represents the V_b-dependent collection efficiency: once V_b crosses the BE turn-on, every additional Iii hole is immediately re-injected into the emitter rather than charging C_body further. This is exactly the mechanism Mario's NS-RAM latches on, and the same one TCAD groups model with an explicit V_b-feedback on η.

## Cross-references

- T4 (missing physics v2): traps N1 + G-R N2 are *modulators* of this base mechanism; they explain transient loop SHAPE, not DC base current existence.
- M3b walk-back (2026-05-04): η ≤ 1 was the correction that revealed the routing gap.
- R3_pyport_audit.md: confirms `compute_iimpact()` already produces correct ngspice-matching Iii.
- Sebas LALPHA0_FIX raised α₀ 10× to compensate for binning loss; ngspice ISUB at FIX = 3.2e-4 A is the ground-truth IIMOD source we should be η-gating.
