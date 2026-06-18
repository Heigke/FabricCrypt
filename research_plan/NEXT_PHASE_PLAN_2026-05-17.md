# NS-RAM Next-Phase Plan — 2026-05-17 (after z473 land + GPU-MAX-A win)

## Where we are

**Modellfas (cell-physics):**
- z471 Mario kalibrerad: Id_pk 4.30 mA, ±0.024 dec över 4 biases
- z472: 6/9 z461 (V1/V2/V4/V5/V8/V9 PASS, V3/V6/V7 FAIL)
- z473: V6 self-reset FLIPPED → 7/9, R_body=1e7. Mario shape 3/5
- V7 (free osc) FAIL — kräver olinjär body-leak
- V3 (DC knee) FAIL — DC-axis, separat issue
- **z469 bug-fix** (I_snap_d i _Id_from_comps) — kvarstår
- **GPU-MAX-A bug-fix** (IFT sign +delta_s) — upstream-patch pending

**Simfas (network):**
- 10 PASS (5 AMB: Res-MG / Stoch-RNG / LMS-Eq / Hier-MNIST / MEP6-MNIST-via-pyport; 5 DISC)
- HDC demoterad efter trippelkonvergens (N-BENCH-A + O80 + N-BENCH-B)
- MEP-6 (differentiable pyport) STÄNGD — backprop verkat genom Newton fixed point via IFT
- Survivable pitch: **multi-function 130nm primitive + diff-pyport methodology + LIF substrate**

## 4 parallella tracks (dispatchade nu)

### Track 1: **z474** — R_body=1e7 default lock + full z461 verification
- Lås `snap_R_body=1e7` som SnapbackParams default
- Re-run full z461 9-test (förväntat 7/9)
- Verifiera Mario shape 3/5 håller över alla 4 biases (inte bara primary)
- Output: results/z474_default_lock/
- ETA: 1-2h, ikaros

### Track 2: **IFT-patch upstream** — apply sign fix to solve_2t_steady_state
- z469-bugg redan applicerad i `_Id_from_comps`
- **Nytt**: GPU-MAX-A hittade `-delta_s → +delta_s` i legacy IFT i `nsram/solver/_solve_at_fixed_vb`
- Lokalisera + patcha + regression test (gradcheck FD <1% relerr)
- Re-run z472 z461 9-test efter patch (kontrollera ingen drift)
- Output: results/z474b_ift_patch/
- ETA: 30 min - 1h, ikaros

### Track 3: **z475** — olinjär body-leak för V7 oscillation
- Linjär R_body kan inte bryta BJT positive-feedback loop under DC hold
- Behöver: tröskel-gated leak (RBE i NPN base) som triggar EFTER V_b > threshold
- Implementera som ny `SnapbackParams.body_leak_kind = "linear"|"threshold"`
- Sweep tröskel + leak-gain, hitta sweet spot där V_b svänger 430 ns period
- Bevara Id_pk + V_b_pk + reset
- Pre-reg: V7 PASS AND z473 metrics held
- Output: results/z475_nonlinear_leak/
- ETA: 2-3h, ikaros

### Track 4: **EP-NSRAM smoke** — Equilibrium Propagation på body-state
- Per GPU MAX CAMPAIGN plan EXP-1
- Smoke: MNIST 2-layer, body-state τ som natural relaxation phase
- Free phase + clamped phase, EP loss
- Pre-reg INFRA only initially: gradient flows, no NaN
- Vidare till full körning om smoke PASS
- Output: results/EP_NSRAM_smoke/
- ETA: 1-2h, zgx

## Längre horisont (dagar 2-7, från GPU MAX CAMPAIGN)

- **EP-NSRAM full** (5-10 dagar): MNIST ≥97%, F-MNIST ≥87%, energy < 100× digital EP
- **HNRT** (5-7 dagar): Differentiable analog reservoir, NARMA-10 NMSE ≤ 0.05
- **NES-GD** (conditional på K2 noise audit): SPSA via impact-jonisering brus
- **Brief v4.5 reframe** (post-z475 + EP smoke): physics primitive + diff modelling, NO "beats X"
- **z476 falsifiers**: Grok ring-osc on Mario die (cheapest), Gemini 16×16 mismatch char

## Killshots (carrying forward)

- K1 IFT singularity → blockerar EP+HNRT. Test in z474b regression
- K2 brus-korrelationer → biaserar NES-GD. Behöver z44x noise-trace audit
- K3 τ-drift → EP relaxation ill-defined
- z475 KILL_SHOT: olinjär leak förstör Id_pk → revert + flagga
- O81 → om 2/3 oracles flag SAME fragility, log och pausa

## Honest standing

- DC fit: ~1.0-1.4 dec (LOCKED, ej försök förbättra utan tape-data)
- Transient: Id_pk Mario ✓, V_b ✓, t_fall ✓, reset ✓, t_rise ❌, osc ❌ (z475 fixar)
- Energy: PROJECTED only, no tapeout. Caveat-flag i varje siffra.
- HDC: demoted. Inte huvudpitch.

## NO-CHEAT carry-forward

- All DC fwd+bwd
- All "beats X" påstående markerade PROJECTED
- z461 metrics verkliga, ej cherry-picked
- Re-run pre-z471 baselines om upstream-patch ändrar matematiken
