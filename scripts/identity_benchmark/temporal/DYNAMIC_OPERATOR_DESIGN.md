# Substrate-as-Dynamic-Operator — Experimental Design (Task E)

Date: 2026-05-31

## Motivation

Phase-1 temporal probe (this dir) surfaced `dP/dT_gpu_slope = 0.068 W/K`
on ikaros vs `0.020 W/K` on daedalus (z_proxy=0.70) — direction predicted
by 4-of-4 oracles. However, the thermal envelopes of the two machines
differ widely (ikaros idle GPU 20 W / 42 °C; daedalus 8 W / 28 °C) so
the slope difference is operating-point-confounded.

To break the envelope confound, design a *closed-loop* model whose
recurrent update **uses live dP/dT as an operator**, not as a state read.

## Hypothesis

H₁: A recurrent model with update rule
```
  h[t+1] = α · h[t] + β · (dP/dT)_live(t) · tanh(W h[t] + U x[t])
```
where `(dP/dT)_live(t)` is the instantaneous cross-coupling read from
the host telemetry while the model is running on the GPU, will exhibit
**chip-specific dynamics** that degrade when transplanted to a chip with
different `dP/dT`.

H₀: Performance is invariant under transplant — `(dP/dT)` is just a
random scalar from the host's environment and the model has learned to
treat it as noise.

## Protocol

### Stage 1: Training

1. Train standard sequence model (small GRU, ~50 k params) on chaotic
   prediction task (Lorenz dt=0.01, 5 k samples) on ikaros.
2. Replace one gate's multiplicative bias with `β · (dP/dT)_live` where
   `(dP/dT)_live` is computed from a 200 ms sliding window of
   `power1_average` and `thermal_zone0` via finite difference.
3. Loss: 1-step MSE. Train 50 epochs. Record final loss `L_ikaros`.

### Stage 2: Transplant

1. Save model weights and architecture.
2. Run inference on **daedalus**, with daedalus's `(dP/dT)_live` driving
   the gate. Record `L_daedalus`.
3. Run inference on **ikaros**, same. Record `L_ikaros_inf`.

### Stage 3: Control

4. Replay ikaros's `(dP/dT)` trace on daedalus (decoupled from local
   physics). Record `L_daedalus_replay_ikaros`.
5. Replay daedalus's `(dP/dT)` trace on ikaros. Record
   `L_ikaros_replay_daedalus`.

### Stage 4: Sensor-ablation control (per gpt-5)

6. Re-run inference on daedalus with all sensor reads patched to return
   ikaros's mean `dP/dT` (constant). Record `L_const`.

## Decision rules

| Outcome | Interpretation |
|---|---|
| L_daedalus >> L_ikaros_inf AND L_daedalus_replay_ikaros ≈ L_ikaros_inf | **CHIP-PHYSICS is the operator** (transplant fails AND restoring source's dP/dT restores accuracy) |
| L_daedalus ≈ L_ikaros_inf | H₀ — model is sensor-agnostic |
| L_const ≈ L_daedalus | Sensor data isn't being used as operator |
| L_daedalus >> L_ikaros_inf AND L_const << L_daedalus | Live dP/dT matters, but in a way that maps to scalar — not necessarily chip-physics |

## Bandwidth note (per gpt-5)

Current SMU bandwidth on gfx1151 hwmon7 is ~5–10 Hz effective. To probe
real physical dynamics requires excitation at f > 5 Hz, which is just
above sensor bandwidth. **This design is limited to the slow regime —
positive results would be "thermal/PDN package operator," not pure die
physics.** A higher-bandwidth follow-up would require on-GPU cycle
counters in place of host telemetry.

## Cost

- Implementation: ~200 LOC PyTorch + telemetry hook
- Compute: ~10 min training × 2 machines + 5 inference runs ~30 min wall
- Thermal: low, model is small

## Status

**DESIGN ONLY** — pre-registration. Do not implement until decision on
write-vs-dispatch is made from the temporal-probe results.
