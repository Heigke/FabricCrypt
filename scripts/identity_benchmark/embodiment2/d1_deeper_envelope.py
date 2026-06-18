"""Phase D1: deeper envelope (100+ features) — compare to 23-feature G2 degradation.

We don't have time to re-collect a 100-feature envelope on daedalus. Instead we
SYNTHESIZE a richer envelope from existing 23-feature vec by concatenating
derived stats: pairwise diffs, products, polynomial features. This tests
whether MORE bits in the envelope hash give MORE binding (G2 factor).

Logic: the envelope→hash→structure pipeline is what creates binding. If we
expand vec23 → vec100+ via deterministic feature expansion, the hash becomes
MORE sensitive to per-machine differences (since differences are amplified
in product features), and we expect G2 factor to grow.
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
SEEDS = 5  # half of Phase C for speed; D1 is comparison
TASK = "narma10"


def expand_envelope(vec23):
    """23 → ~100 features via deterministic expansion."""
    v = np.asarray(vec23, dtype=np.float64)
    feats = [v]
    # log-magnitude (sign-safe)
    feats.append(np.sign(v) * np.log1p(np.abs(v)))
    # pairwise products of first 10 features
    p = np.outer(v[:10], v[:10])[np.triu_indices(10, k=1)]
    feats.append(p)
    # ratios
    safe = v + 1e-9
    r = (v[:8, None] / safe[None, :8])[np.triu_indices(8, k=1)]
    feats.append(r)
    # sqrt-abs
    feats.append(np.sign(v) * np.sqrt(np.abs(v)))
    # all polynomial powers
    feats.append(v * v)
    out = np.concatenate(feats)
    return out  # ~23 + 23 + 45 + 28 + 23 + 23 = 165


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()
    out_dir = OUT2 / "phase_d"
    out_dir.mkdir(parents=True, exist_ok=True)
    vk = load_vec("ikaros"); vd = load_vec("daedalus")
    # baseline (23-feat) — re-run for direct comparison
    s_ik23 = derive_structure_v2(vk, N); s_da23 = derive_structure_v2(vd, N)
    # expanded (100+)
    vk_e = expand_envelope(vk); vd_e = expand_envelope(vd)
    s_ik_e = derive_structure_v2(vk_e, N); s_da_e = derive_structure_v2(vd_e, N)

    results = {"N": N, "seeds": args.seeds, "task": TASK,
               "vec_expanded_len": len(vk_e),
               "vec_expanded_overlap_bits_ikvsda": float(
                   (derive_structure_v2(vk_e, N)["mask"] == derive_structure_v2(vd_e, N)["mask"]).mean()
               )}
    print(f"[D1] expanded vec len: {len(vk_e)}", flush=True)
    print(f"[D1] expanded ik vs da mask overlap: {results['vec_expanded_overlap_bits_ikvsda']:.4f}", flush=True)

    for label, s_ik, s_da in [("vec23", s_ik23, s_da23), ("vec_expanded", s_ik_e, s_da_e)]:
        g1, g2 = [], []
        for s in range(args.seeds):
            t0 = time.time()
            nr1, w = train_eval_task(s_ik, N, s, task=TASK)
            nr2 = transplant_eval(w, s_da, N, s, task=TASK)
            g1.append(nr1); g2.append(nr2)
            print(f"[D1][{label}] seed={s} G1={nr1:.4f} G2={nr2:.2f} (t={time.time()-t0:.1f}s)", flush=True)
        results[label] = {
            "G1_median": float(np.median(g1)), "G2_median": float(np.median(g2)),
            "factor": float(np.median(g2) / max(1e-9, np.median(g1))),
            "G1_per_seed": g1, "G2_per_seed": g2,
        }
        print(f"[D1][{label}] G1={results[label]['G1_median']:.4f} G2={results[label]['G2_median']:.2f} factor={results[label]['factor']:.1f}x", flush=True)

    f_old = results["vec23"]["factor"]; f_new = results["vec_expanded"]["factor"]
    results["binding_amplification"] = f_new / max(1e-9, f_old)
    results["verdict"] = "DEEPER_BINDS_TIGHTER" if f_new > 1.5 * f_old else (
        "DEEPER_NEUTRAL" if 0.66 * f_old <= f_new <= 1.5 * f_old else "DEEPER_WEAKER")
    print(f"[D1] amplification: {results['binding_amplification']:.2f}  verdict={results['verdict']}", flush=True)

    (out_dir / "D1_result.json").write_text(json.dumps(results, indent=2))
    print(f"[D1] wrote {out_dir / 'D1_result.json'}", flush=True)


if __name__ == "__main__":
    main()
