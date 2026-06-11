# H7 Channel Audit — ikaros + daedalus synthesis (2026-06-10)

Per Eric: *"har vi verkligen koll på att alla våra signaler djupa signaler har de egenskaper vi vill ha, vi måste vara helt hundra på att de är identitetsbärande och har feedbackloopen vi pratar om sen kan vi försöka koppla in dem i modeller"*

## Criteria

A channel is embodiment-bearing only if it has:
1. **Cross-host discriminativity** — distinguishes ikaros vs daedalus (Cohen's d > 1.0)
2. **Compute-load coupling** — busy-GPU shifts the channel ≥0.5σ vs idle (the closed loop)
3. **Spoof-resistance** — matched-spectrum AR(1) cannot fake the higher-moment signature (>0.85 LR-acc)
4. **Non-trivial dynamics** — ridge R² beats persistence R² by >0.05 (channel has structure model can learn)

KEEP if ≥3/4 met. Both hosts must agree.

## Per-channel results (live 4096-sample windows × 2 conditions × 2 hosts)

| ch  | name        | ik load_d | da load_d | ik r2_gap | da r2_gap | ik spoof | da spoof | ik | da | both |
|-----|-------------|-----------|-----------|-----------|-----------|----------|----------|----|----|------|
| 0   | C07_xtal    |   0.00    |   0.34    | -1803.4   | -0.07     | 0.000    | 0.976    | 1  | 2  |      |
| 1   | C09_pm1     | 195.7     | 508.4     | -0.61     | -6.27     | 0.675    | 0.365    | 2  | 2  |      |
| 2   | C20_lat_x   |   1.54    |   2.84    |  0.327    |  0.141    | 0.865    | 0.921    | 3  | 4  | ★    |
| 3   | C20_logtl   |   0.01    |   0.14    |  0.071    |  0.064    | 0.881    | 0.976    | 2  | 3  |      |
| 4   | C11_drift   |   1.58    |   3.47    |  0.182    |  0.313    | 0.944    | 0.937    | 3  | 4  | ★    |
| 5   | C05_e0_rt   |   1.00    |   1.35    |  0.344    |  0.504    | 1.000    | 1.000    | 4  | 4  | ★★   |
| 6   | C06_fast    |   0.09    |   0.10    |  0.932    |  0.979    | 0.937    | 0.976    | 3  | 3  | ★(†) |
| 7   | C09_pm3     |   3.55    |   6.53    |  0.004    |  0.000    | 0.730    | 0.770    | 2  | 2  |      |
| 8   | C09_pm5     | 215.1     | 668.9     | -0.64     | -0.24     | 0.286    | 0.762    | 2  | 2  |      |
| 9   | C20_lat_e   |   0.85    |   1.82    | -0.018    | -0.002    | 0.929    | 0.968    | 3  | 3  | ★    |

## Robust embodiment-bearing set (5 channels)

- **C05_e0_rt** — energy-counter rate. 4/4 both hosts. The KING channel.
- **C20_lat_x** — SMN xtal-read latency. Closed loop confirmed (load_d 1.54/2.84).
- **C11_drift** — TSC drift. Closed loop confirmed (load_d 1.58/3.47).
- **C20_lat_e** — SMN energy-read latency. Closed loop confirmed (load_d 0.85/1.82).
- **C06_fast** (†) — fast counter rate. Dynamics + spoof excellent, **but load_d ≈ 0.1 on BOTH hosts** → genuinely no compute coupling.

## Honest interpretation

**4 channels carry the closed loop** Buhrmann/SMC requires: C05_e0_rt, C20_lat_x, C11_drift, C20_lat_e. These are the channels where the model's own compute affects what the model later reads. Embodiment via these is principled.

**C06_fast is a passenger** — highly predictable structure, spoof-resistant signature, but inert to compute load. It contributes identity information (it differs across hosts) without supporting sensorimotor contingency. Useful as identity carrier, NOT as closed-loop axis.

**Dropped channels (5/10):** C07_xtal (broken r²), C09_pm1/pm3/pm5 (huge load_d but failed spoof + gap), C20_logtl (no compute coupling).

## Empirical confirmation of Eric's "poäng 1"

> *"är inte alla metriker, oscillationer, latens osv också beroende av värme eller aktivitet så att alla störs?"*

Yes — confirmed empirically. 7/10 channels show load_d > 0.5 on at least one host. The closed loop is not thermal-only; latency, drift, energy rate, and PM table all shift under load. We have a multi-channel SMC loop, not a single thermal pathway.

## Next: v6 substrate

Build `SubstrateStateV4` with 5 channels: `[C20_lat_x, C11_drift, C05_e0_rt, C06_fast, C20_lat_e]`. Tag C06_fast as "identity-only / no SMC closed loop" so v6 model can keep it in but skip it in compute-action conditioning loss.
