# `data/` — captured signatures and trained classifiers

Files produced into this directory by the reproduction scripts:

| File                          | Produced by                              | Contents |
|-------------------------------|------------------------------------------|----------|
| `<host>_sig_v2.npz`           | `scripts/01_collect_signature.sh`        | `vec` (R, 290), `host`, `dim`, `block_starts` |
| `<host>_sig_v2_meta.json`     | same                                     | per-rep temps, timings |
| `<host>_paired_sigs.npz`      | `python -m src.protocol.train`           | `nonces` (N, 8) uint8, `sigs` (N, 64) float32 |
| `<host>_t3_best.pt`           | same                                     | trained `TwinMLP` state_dict |
| `<host>_training.json`        | same                                     | T2/T3 AUROC seeds + CI95 |
| `<host>_spoof.json`           | `scripts/03_test_replay.sh`              | 7-attack accept rates + gate verdicts |
| `loo_classification.json`     | `scripts/02_classify.sh`                 | LOO acc across all hosts you pass in |

None of these files contain personal data or filesystem content. They
are safe to share with the community dataset.

## Contributing your signature
See `examples/publish_signature.sh` and open an issue on GitHub.
