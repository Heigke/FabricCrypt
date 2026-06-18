# V7 Topology Rewrite — 2026-05-18

Source: Plan agent ab4b117eec8187d46.

## Status quo (3 states: V_B, q_F, q_R)
z475 proved V_B has globally attracting equilibrium at ~0.62V under DC V_d=2V. No Hopf, no hysteresis. Any rewrite must add **new state variable on slower timescale** that periodically destabilizes V_B*.

## RECOMMENDED: Proposal #1 — Slow charge trap (FitzHugh-Nagumo)

**ODE (4th state n):**
```
dn/dt = (α·(V_B - V_n0) - n) / τ_slow,   τ_slow ≈ 200-500 ns ≫ τ_F=25 ps
```
Modify body KCL:
```
C_eff·dV_B/dt = R_B - I_diff - I_leak - k_n·n
```

**Mechanism**: Trap current k_n·n is *negative-feedback recovery* that lags V_B. When V_B latches high, n integrates up, eventually overpowers Iii_body → V_B forced down past V_be(on); n decays → cycle restarts. Classical Bonhoeffer-van der Pol relaxation oscillator.

**Physics**: NS-RAM lit invokes slow interface/oxide-trap charging at body-poly boundary (Sebas's nm shadow state). 100ns-1µs constants documented in BSIM4 TAT models.

**Mario compatibility**: n(0)=0; on 200ns Mario pulse n reaches only ~40% → <5% perturbation to Id_pk. Calibration preserved.

**V6 reset preserved**: R_body=1e7 untouched; trap adds extra discharge channel that aids reset.

**Implementation**: ~40 LOC in `transient_real_v2.py`:
- 4 new TransientCfgV2 fields: `enable_trap`, `tau_slow`, `k_n`, `V_n0`, `alpha_n`
- State vector → 4-vec
- Init q_F, q_R loop extended

No change to snapback_subcircuit.py or nsram_cell_2T.py.

**Falsifier**: sweep τ_slow ∈ {50, 200, 500, 1000} ns × k_n ∈ {1e-6, 1e-4, 1e-2}. If no point produces ≥3 cycles in [100, 1000] ns with Id_pk drift <0.15 dec → trap topology dead.

## Alternatives (if #1 fails)

### #2 Drain RC parasitic (V_d as state)
- C_drain·dV_d/dt = (V_d_stim - V_d)/R_d - I_d
- Risk: ideal V_d=2V assumption breaks → Id_pk drift likely
- Cost: ~25 LOC

### #3 Substrate-return finite R+C
- Vsint joint-mode + R_sub + C_sub
- Risk: z471 cal used Vsint_pin=0; finite R_sub shifts V_be 50-200mV
- Cost: ~20 LOC

### #4 Well-diode active draw
- Shockley I_pdiode at V_B>0.6V
- Risk: same argument as z475 (nonlinear leak alone cannot Hopf)
- Cost: ~15 LOC (cheapest)
- Expected to fail by same logic as z475

### #5 Shadow nm synaptic state
- τ_nm ~ µs (too slow for 430ns V7)
- Cost: ~60 LOC
- Timescale mismatch — wrong tool

## Verdict
**Implement #1 first.** Only proposal with:
- True second timescale (necessary for Hopf)
- Localized to one file
- Mario + V6 untouched by construction
- Direct NS-RAM lit precedent
- 2D sweep (τ_slow × k_n) ~ 25 points expected to reveal oscillation tongue
