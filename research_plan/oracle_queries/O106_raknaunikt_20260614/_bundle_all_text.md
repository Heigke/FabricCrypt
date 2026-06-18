# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: cross_die_transfer.json (1091 chars) ===
```json
{
  "xor_transfer": [
    {
      "src": "ikaros",
      "dst": "daedalus",
      "self": 0.6146341463414634,
      "cross": 0.5170731707317073,
      "cross_dstnorm": 0.583739837398374,
      "drop": 0.09756097560975607
    },
    {
      "src": "daedalus",
      "dst": "ikaros",
      "self": 0.7008130081300813,
      "cross": 0.5853658536585366,
      "cross_dstnorm": 0.6325203252032521,
      "drop": 0.11544715447154474
    }
  ],
  "recall_u_transfer": [
    {
      "src": "ikaros",
      "dst": "daedalus",
      "self": 0.8991869918699187,
      "cross": 0.7073170731707317,
      "cross_dstnorm": 0.8682926829268293,
      "drop": 0.19186991869918701
    },
    {
      "src": "daedalus",
      "dst": "ikaros",
      "self": 0.9723577235772358,
      "cross": 0.5430894308943089,
      "cross_dstnorm": 0.6943089430894309,
      "drop": 0.4292682926829269
    }
  ],
  "mean_xor_drop": 0.10650406504065041,
  "mean_u_drop": 0.31056910569105695,
  "DIE_SPECIFIC_MIXING": false,
  "verdict": "mixing transfers across dies (or u-only also drops) \u2014 NOT cleanly die-specific"
}
```


=== FILE: mixing_verify_ikaros.json (485 chars) ===
```json
{
  "host": "ikaros",
  "die_full_XOR": 0.6536585365853659,
  "null_mean": 0.5075284552845529,
  "null_p95": 0.5430894308943089,
  "null_pvalue": 0.0,
  "uv_partialR2_median": 0.0002254691528275592,
  "uv_partialR2_max": 0.08890800249210119,
  "u_only_poly": 0.5300813008130081,
  "u_and_v_LINEAR": 0.49105691056910566,
  "uv_product_ceiling": 0.7463414634146341,
  "die_XOR_bootCI95": [
    0.5544715447154471,
    0.6308943089430894
  ],
  "d_v": 1.0,
  "PRE_REGISTERED_REAL": true
}
```
