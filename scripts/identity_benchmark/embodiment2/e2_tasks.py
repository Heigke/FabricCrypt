"""Phase E2: more tasks — NARMA-10 baseline + Mackey-Glass-17 + memory capacity + sinusoid.

Per task, N=128, measure G1 / G2 / factor on ikaros→daedalus transplant.
"""
from __future__ import annotations
import json, sys, time, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _lib import (load_vec, derive_structure_v2, train_eval_task,
                  transplant_eval, OUT2)

N = 128
TASKS = ["narma10", "mackey17", "memcap", "sinusoid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tasks", default=",".join(TASKS))
    args = ap.parse_args()
    out_dir = OUT2 / "phase_e"; out_dir.mkdir(parents=True, exist_ok=True)
    vk = load_vec("ikaros"); vd = load_vec("daedalus")
    s_ik = derive_structure_v2(vk, N); s_da = derive_structure_v2(vd, N)
    tasks = args.tasks.split(",")
    results = {"N": N, "seeds": args.seeds, "per_task": {}}
    for task in tasks:
        g1, g2 = [], []
        for s in range(args.seeds):
            t0 = time.time()
            try:
                nr1, w = train_eval_task(s_ik, N, s, task=task)
                nr2 = transplant_eval(w, s_da, N, s, task=task)
                g1.append(nr1); g2.append(nr2)
                print(f"[E2][{task}] seed={s} G1={nr1:.4f} G2={nr2:.4f} (t={time.time()-t0:.1f}s)", flush=True)
            except Exception as e:
                print(f"[E2][{task}] seed={s} EXCEPTION: {e}", flush=True)
        if not g1:
            results["per_task"][task] = {"error": "no_seeds_completed"}
            continue
        # For memcap, "lower is better" inverted (we return -MC); keep convention
        factor_abs = float(abs(np.median(g2)) / max(1e-9, abs(np.median(g1))))
        results["per_task"][task] = {"G1_median": float(np.median(g1)),
                                      "G2_median": float(np.median(g2)),
                                      "factor_abs": factor_abs,
                                      "G1_per_seed": g1, "G2_per_seed": g2,
                                      "note": "memcap returns -MC; factor uses abs"}
        print(f"[E2][{task}] factor={factor_abs:.2f}x", flush=True)
    (out_dir / "E2_result.json").write_text(json.dumps(results, indent=2))
    print(f"[E2] wrote E2_result.json", flush=True)


if __name__ == "__main__":
    main()
