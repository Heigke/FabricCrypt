# z142 vs z139 — topology MC ranking comparison

**Source:** `results/z142_topology_v2/summary.json` (FULL)

## N=800 MC across normalisations

| topology       | z139 (Bf=2e4) | z142 rho_lambda | z142 rho_p95_sv | z142 rho_deg_norm |
|----------------|--------------:|----------------:|----------------:|-------------------:|
| ER_SPARSE      |  2.20 (n=2) |  3.55 (n=5) |  2.84 (n=5) |  2.70 (n=5) |
| LAYERED        |  2.17 (n=2) |  3.56 (n=5) |  2.68 (n=5) |  1.07 (n=5) |
| RAND_GAUSS     |  1.87 (n=2) |  2.25 (n=5) |  3.08 (n=5) |  0.21 (n=5) |
| MESH_4N        |  3.29 (n=2) |  2.20 (n=5) |  1.96 (n=5) |  4.00 (n=5) |
| WS_SMALLWORLD  |  2.94 (n=2) |  2.17 (n=5) |  2.14 (n=5) |  4.34 (n=5) |
| HUB_SPOKE      |  2.89 (n=2) |  1.92 (n=5) |  2.47 (n=5) |  4.43 (n=5) |

## N=800 rho_lambda ranking change (z139 → z142)

| topology       | z139 MC | z142 MC | Δ      | rank z139 | rank z142 |
|----------------|--------:|--------:|-------:|----------:|----------:|
| ER_SPARSE      |    2.20 |    3.55 |  +1.35 |         4 |         2 |
| LAYERED        |    2.17 |    3.56 |  +1.39 |         5 |         1 |
| RAND_GAUSS     |    1.87 |    2.25 |  +0.38 |         6 |         3 |
| MESH_4N        |    3.29 |    2.20 |  -1.08 |         1 |         4 |
| WS_SMALLWORLD  |    2.94 |    2.17 |  -0.77 |         2 |         5 |
| HUB_SPOKE      |    2.89 |    1.92 |  -0.97 |         3 |         6 |

## rho_lambda MC across scale (z142 honest cell)

| topology       | N=100 | N=300 | N=800 | scaling N=100→800 |
|----------------|------:|------:|------:|------------------:|
| ER_SPARSE      |  1.90 |  2.99 |  3.55 |             1.87× |
| LAYERED        |  2.40 |  3.53 |  3.56 |             1.48× |
| RAND_GAUSS     |  2.34 |  2.67 |  2.25 |             0.96× |
| MESH_4N        |  1.60 |  1.76 |  2.20 |             1.37× |
| WS_SMALLWORLD  |  1.81 |  1.66 |  2.17 |             1.20× |
| HUB_SPOKE      |  1.19 |  1.72 |  1.92 |             1.60× |

## Multi-task ranking at N=800 rho_lambda (honest cell, n=5)

| topology       | MC    | NARMA NRMSE↓ | XOR_acc | WAVE_acc |
|----------------|------:|-------------:|--------:|---------:|
| ER_SPARSE      |  3.55 |         0.98 |    0.97 |     0.52 |
| LAYERED        |  3.56 |         1.05 |    0.94 |     0.47 |
| RAND_GAUSS     |  2.25 |         1.18 |    0.75 |     0.45 |
| MESH_4N        |  2.20 |         0.95 |    0.68 |     0.53 |
| WS_SMALLWORLD  |  2.17 |         1.00 |    0.59 |     0.51 |
| HUB_SPOKE      |  1.92 |         1.00 |    0.50 |     0.53 |

## Per-task champion (z142 honest cell, N=800 rho_lambda)

- **MC**:    LAYERED
- **NARMA**: MESH_4N (lower is better)
- **XOR**:   ER_SPARSE
- **WAVE**:  HUB_SPOKE

## HUB_SPOKE WAVE classification — z139 → z142

- z139 WAVE: **0.608** (was the brief's classification champion)
- z142 WAVE: **0.530** — advantage gone

## Headline
- z139 N=800 rho_lambda MC champion: **MESH_4N**
- z142 N=800 rho_lambda MC champion: **LAYERED**
- Inversion: YES
