"""Phase 2 v2 transplant matrix on the 23-feature envelope substrate.

For each (seed, train_dev, eval_dev, control_variant):
  - HW: real substrate vector for the device
  - SW_MATCHED: iid Gaussian matched to that vec's (mean, std)
  - SHUFFLE: pair W trained on device A's substrate with device B's substrate
             (this overlaps off-diagonal HW, but with the *wrong* device label;
              under our gate, SHUFFLE Δ should be flat if W learned a real coupling)
  - NO_SUB: zero substrate vector

Output: results/IDENTITY_BENCHMARK_2026-05-30/phase2_v2/matrix.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))

from _substrate_v2 import load_pair, matched_gaussian  # noqa
from narma10_envelope import train_eval, train_eval_pmnist  # noqa

OUT = ROOT / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "phase2_v2"
OUT.mkdir(parents=True, exist_ok=True)

N_SEEDS = 30
DEVICES = ("ikaros", "daedalus")


def main():
    zi, zd, labels, raw = load_pair()
    subs = {"ikaros": zi, "daedalus": zd}

    rng_master = np.random.default_rng(42)

    results = {"narma10": [], "pmnist": [], "meta": {
        "n_seeds": N_SEEDS, "n_features": int(zi.size),
        "labels": labels,
        "raw_ikaros": raw[0].tolist(),
        "raw_daedalus": raw[1].tolist(),
        "L2_dist_z": float(np.linalg.norm(zi - zd)),
    }}

    t0 = time.time()
    for seed in range(N_SEEDS):
        # Per-seed matched-Gaussian draws (one per device)
        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        sw = {
            "ikaros": matched_gaussian(zi, rng),
            "daedalus": matched_gaussian(zd, rng),
        }
        no_sub = np.zeros_like(zi)

        for train_dev in DEVICES:
            for eval_dev in DEVICES:
                # HW
                r = train_eval(subs[train_dev], subs[eval_dev], seed)
                results["narma10"].append({"variant": "HW", "train": train_dev,
                                           "eval": eval_dev, **r})
                # SW_MATCHED
                r = train_eval(sw[train_dev], sw[eval_dev], seed)
                results["narma10"].append({"variant": "SW_MATCHED", "train": train_dev,
                                           "eval": eval_dev, **r})
                # NO_SUB
                r = train_eval(no_sub, no_sub, seed)
                results["narma10"].append({"variant": "NO_SUB", "train": train_dev,
                                           "eval": eval_dev, **r})

            # SHUFFLE: train with own device, eval with random permutation
            # of own substrate (destroys envelope structure but preserves
            # mean/std/components)
            shuf = subs[train_dev].copy()
            np.random.default_rng(seed + 12345).shuffle(shuf)
            r = train_eval(subs[train_dev], shuf, seed)
            results["narma10"].append({"variant": "SHUFFLE", "train": train_dev,
                                       "eval": train_dev, **r})

        # Permuted-MNIST: only HW + SW_MATCHED, diag vs off-diag
        for train_dev in DEVICES:
            for eval_dev in DEVICES:
                r = train_eval_pmnist(subs[train_dev], subs[eval_dev], seed)
                results["pmnist"].append({"variant": "HW", "train": train_dev,
                                          "eval": eval_dev, **r})
                r = train_eval_pmnist(sw[train_dev], sw[eval_dev], seed)
                results["pmnist"].append({"variant": "SW_MATCHED", "train": train_dev,
                                          "eval": eval_dev, **r})

        if (seed + 1) % 5 == 0:
            print(f"  seed {seed+1}/{N_SEEDS}  elapsed={time.time()-t0:.1f}s")

    (OUT / "matrix.json").write_text(json.dumps(results, indent=2))
    print(f"DONE in {time.time()-t0:.1f}s -> {OUT/'matrix.json'}")


if __name__ == "__main__":
    main()
