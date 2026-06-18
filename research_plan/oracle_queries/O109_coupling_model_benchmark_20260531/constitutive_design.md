# Task B (Phase 9) — constitutive live-substrate coupling

Reservoir where the leak rate IS the live substrate at every forward-pass step.

```python
def step(h, x, t):
    apu_t = read_thermal_zone0()           # /sys/class/thermal/thermal_zone0/temp
    apu_p = read_rapl_package()            # /sys/class/powercap/intel-rapl:0/energy_uj
    alpha = sigmoid((apu_t - 50) / 10) * 0.5 + 0.25   # 0.25..0.75
    gain  = 1.0 + 0.1 * (apu_p / 30 - 1)              # gain modulation
    new_h = (1 - alpha) * h + alpha * np.tanh(gain * (W @ h + U @ x))
    return new_h
```

Substrate is NOT a feature appended to `x`. It *is* a parameter of the
recurrent update. Two hosts with different thermal trajectories run
*different dynamical systems*, not just receive different inputs.

Transplant matrix:
- A — trained on ikaros, evaluated on ikaros (own substrate at inference).
- B — trained on daedalus, evaluated on ikaros (transplant; alien dynamics).
- C — random `alpha = 0.5` constant control (no substrate coupling).
- D — SHUFFLE: ikaros-trained model, but at inference replay daedalus's
  recorded substrate trajectory through `alpha`. Tests whether
  *trajectory specificity* matters beyond mere mismatch.

Pre-reg: A − B ≥ 10% NRMSE, A − D ≥ 5% NRMSE.
