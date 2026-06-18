"""Phase 2 transplant matrix: 2 substrates x {HW, SW-matched-RNG, shuffle} x 10 seeds.

Pure CPU/numpy. No thermal risk. Writes results JSON to out-dir.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _substrate_hooks import SubstrateSampler, shuffle_sampler, matched_rng_sampler
from narma10_reservoir import run_one

DEVICES = ["ikaros", "daedalus"]
CONTROLS = ["HW", "SW_iid", "SHUFFLE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--T-train", type=int, default=2000)
    ap.add_argument("--T-test", type=int, default=500)
    ap.add_argument("--substrate-strength", type=float, default=0.05)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()
    for device in DEVICES:
        base = SubstrateSampler(device, seed=0)
        for control in CONTROLS:
            for seed in range(args.seeds):
                if control == "HW":
                    samp = SubstrateSampler(device, seed=seed)
                elif control == "SW_iid":
                    samp = matched_rng_sampler(base, seed=seed)
                elif control == "SHUFFLE":
                    samp = shuffle_sampler(base, seed=seed)
                r = run_one(samp, seed,
                            T_train=args.T_train, T_test=args.T_test,
                            cfg_kwargs={"substrate_strength": args.substrate_strength})
                r["device"] = device
                r["control"] = control
                rows.append(r)
                print(f"[matrix] {device}/{control}/seed{seed} -> nrmse={r['nrmse']:.4f}")
    dur = time.time() - t0
    out = {"ts": time.time(), "duration_s": dur,
           "n_devices": len(DEVICES), "n_controls": len(CONTROLS),
           "n_seeds": args.seeds, "substrate_strength": args.substrate_strength,
           "rows": rows}
    (args.out / "matrix_results.json").write_text(json.dumps(out, indent=2))
    print(f"[matrix] wrote {args.out/'matrix_results.json'} dur={dur:.1f}s")


if __name__ == "__main__":
    main()
