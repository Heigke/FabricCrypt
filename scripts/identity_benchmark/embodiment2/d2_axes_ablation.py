"""Phase D2: per-axis ablation — which structural axis contributes most binding?

Baseline = no axis envelope-keyed (deterministic).
For each axis A in {mask, acts, perm, weight_scale, leak}:
  - Use envelope-keyed value for axis A, baseline for others
  - Train ikaros, transplant daedalus
  - Measure G2 factor contribution from axis A alone
"""
from __future__ import annotations
import json, sys, time, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _lib import (load_vec, derive_structure_v2, baseline_structure_v2,
                  train_eval_task, transplant_eval, OUT2)

N = 128
TASK = "narma10"
AXES = ["mask", "acts", "perm", "weight_scale", "leak"]


def mix_struct(base, env, axes_from_env):
    out = {}
    for k in ("mask", "acts", "perm", "weight_scale", "leak"):
        out[k] = env[k] if k in axes_from_env else base[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    out_dir = OUT2 / "phase_d"; out_dir.mkdir(parents=True, exist_ok=True)
    vk = load_vec("ikaros"); vd = load_vec("daedalus")
    s_ik = derive_structure_v2(vk, N); s_da = derive_structure_v2(vd, N)
    s_base = baseline_structure_v2(N, seed=0)

    results = {"N": N, "task": TASK, "seeds": args.seeds, "axes_results": {}}
    # Baseline (all deterministic): G1_base
    print("[D2] computing baseline (no axes from envelope)...", flush=True)
    g1b = []
    for s in range(args.seeds):
        nr, _ = train_eval_task(s_base, N, s, task=TASK)
        g1b.append(nr)
    results["baseline_G1_median"] = float(np.median(g1b))
    print(f"[D2] baseline G1={results['baseline_G1_median']:.4f}", flush=True)

    # All axes from envelope (full v2 binding)
    print("[D2] all axes from envelope...", flush=True)
    g1f, g2f = [], []
    for s in range(args.seeds):
        nr1, w = train_eval_task(s_ik, N, s, task=TASK)
        nr2 = transplant_eval(w, s_da, N, s, task=TASK)
        g1f.append(nr1); g2f.append(nr2)
    results["all_axes"] = {"G1": float(np.median(g1f)), "G2": float(np.median(g2f)),
                            "factor": float(np.median(g2f) / max(1e-9, np.median(g1f)))}
    print(f"[D2] all axes G1={results['all_axes']['G1']:.4f} G2={results['all_axes']['G2']:.2f} factor={results['all_axes']['factor']:.1f}x", flush=True)

    # Single-axis-from-envelope variants
    for axis in AXES:
        t0 = time.time()
        s_ik_mix = mix_struct(s_base, s_ik, {axis})
        s_da_mix = mix_struct(s_base, s_da, {axis})
        g1, g2 = [], []
        for s in range(args.seeds):
            nr1, w = train_eval_task(s_ik_mix, N, s, task=TASK)
            nr2 = transplant_eval(w, s_da_mix, N, s, task=TASK)
            g1.append(nr1); g2.append(nr2)
        factor = float(np.median(g2) / max(1e-9, np.median(g1)))
        results["axes_results"][axis] = {"G1": float(np.median(g1)),
                                          "G2": float(np.median(g2)),
                                          "factor": factor}
        print(f"[D2][{axis}] G1={np.median(g1):.4f} G2={np.median(g2):.2f} factor={factor:.1f}x  (t={time.time()-t0:.1f}s)", flush=True)

    # Rank axes by binding contribution
    ranked = sorted(results["axes_results"].items(), key=lambda kv: -kv[1]["factor"])
    results["axes_ranked"] = [k for k, _ in ranked]
    print(f"[D2] ranked: {results['axes_ranked']}", flush=True)

    (out_dir / "D2_result.json").write_text(json.dumps(results, indent=2))
    print(f"[D2] wrote D2_result.json", flush=True)


if __name__ == "__main__":
    main()
