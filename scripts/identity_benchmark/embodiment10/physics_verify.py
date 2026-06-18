"""Embodiment Phase 10 — Task B.

Verify Phase 8 physics-aware claim: daedalus showed +37.97% NRMSE gain over
random structure on a chassi-keyed physics-aware reservoir.  Phase 8 used
only 30 seeds + one (canonical) feature-class assignment.

This script:
  1) Re-runs with 100 seeds (instead of 30) — does daedalus gain hold?
  2) Permutes the feature-block assignment 50 times (shuffles which feature-
     class label gets which block-index).  Bootstrap pass-rate.
  3) Swaps the *winning* (canonical, real-class) assignment from daedalus
     onto ikaros — does ikaros also gain?
  4) Identifies which feature *classes* drive daedalus's gain (one-leave-out
     class ablation: drop that class's block from the partition).
  5) 95% bootstrap CI on the 100-seed daedalus gain.

Results -> results/IDENTITY_BENCHMARK_2026-05-30/embodiment10/physics_verify.json
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "embodiment8"))

from abcd_rich import (  # type: ignore
    HASHES, hash_to_seed, load_features, make_windows, nrmse,
    HIST_C1, HORIZON_C1, N_TRAIN_C1, N_TEST_C1, RES_DIM,
    bootstrap_diff_ci,
)
from physics_aware import PhysicsAwareReservoir, classify_ts  # type: ignore

OUT_DIR = Path(__file__).resolve().parents[3] / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment10"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS_VERIFY = 100
N_PERMUTE = 50


# ---------------------------------------------------------------------------
class PermutedPhysicsReservoir(PhysicsAwareReservoir):
    """Like PhysicsAwareReservoir but lets us inject an arbitrary
    feature_idx -> block_idx mapping (overriding the classify_ts result)."""

    def __init__(self, structure_seed, ts_feature_names, custom_feat_block,
                 res_dim=RES_DIM):
        rng = np.random.default_rng(structure_seed)
        din = len(ts_feature_names)
        blocks = self.BLOCKS
        weights = rng.dirichlet(np.ones(len(blocks)))
        sizes = np.round(weights * res_dim).astype(int)
        sizes = np.maximum(sizes, 4)
        while sizes.sum() != res_dim:
            if sizes.sum() > res_dim:
                idx = int(rng.integers(0, len(sizes)))
                if sizes[idx] > 4: sizes[idx] -= 1
            else:
                idx = int(rng.integers(0, len(sizes)))
                sizes[idx] += 1
        block_scale = rng.uniform(0.5, 1.5, size=len(blocks)).astype(np.float32)
        feat_block = np.asarray(custom_feat_block, dtype=int)
        W_in = np.zeros((din, res_dim), dtype=np.float32)
        block_starts = np.concatenate([[0], np.cumsum(sizes)])
        for i in range(din):
            bidx = int(feat_block[i])
            s, e = block_starts[bidx], block_starts[bidx+1]
            denom = max(1, (feat_block == bidx).sum())
            W_in[i, s:e] = rng.standard_normal(e - s).astype(np.float32) / np.sqrt(denom) * block_scale[bidx]
        self.W_in = W_in
        self.W_rec = (rng.standard_normal((res_dim, res_dim)) / np.sqrt(res_dim) * 0.9).astype(np.float32)
        self.bias = (rng.standard_normal(res_dim) * 0.1).astype(np.float32)
        self.res_dim = res_dim; self.din = din
        self.W_out = None; self.b_out = None


def canonical_feat_block(ts_names):
    blocks = PhysicsAwareReservoir.BLOCKS
    out = []
    for n in ts_names:
        cls = classify_ts(n)
        if cls in blocks:
            out.append(blocks.index(cls))
        else:
            out.append(blocks.index("other"))
    return np.asarray(out, dtype=int)


def run_cell(struct_seed_source, ts_names, X, n_seeds, feat_block, host_key=None):
    """Run n_seeds with a given feature_block mapping; return per-seed NRMSE."""
    nrmses = []
    D = X.shape[1]
    for seed in range(n_seeds):
        if struct_seed_source == "random":
            ss = seed * 1009 + 7
        else:
            ss = hash_to_seed(HASHES[struct_seed_source], salt=seed)
        Xtr, Ytr = make_windows(X, N_TRAIN_C1, HIST_C1, HORIZON_C1, seed=seed)
        Xte, Yte = make_windows(X, N_TEST_C1, HIST_C1, HORIZON_C1, seed=seed + 9001)
        m = PermutedPhysicsReservoir(ss, list(ts_names), feat_block)
        m.fit(Xtr, Ytr)
        Yp = m.predict(Xte, HORIZON_C1, D)
        nrmses.append(nrmse(Yte, Yp))
    return nrmses


def gain_pct(nrmse_phys, nrmse_baseline):
    """Lower NRMSE is better; return % improvement."""
    a = float(np.mean(nrmse_phys))
    b = float(np.mean(nrmse_baseline))
    return 100.0 * (b - a) / max(b, 1e-9)


def bootstrap_gain_ci(nrmse_phys, nrmse_base, n_boot=2000, ci=0.95, seed=1):
    """Bootstrap CI on percent-improvement = (mean(base) - mean(phys)) / mean(base) * 100."""
    rng = np.random.default_rng(seed)
    a = np.asarray(nrmse_phys, float); b = np.asarray(nrmse_base, float)
    gains = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        gains.append((sb.mean() - sa.mean()) / max(sb.mean(), 1e-9) * 100.0)
    gains = np.sort(gains)
    lo = float(gains[int((1 - ci) / 2 * n_boot)])
    hi = float(gains[int((1 + ci) / 2 * n_boot)])
    return float(np.mean(gains)), lo, hi


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    out = {"n_seeds": N_SEEDS_VERIFY, "n_permute": N_PERMUTE}

    feats = {}
    for h in ("ikaros", "daedalus"):
        X, names = load_features(h)
        feats[h] = (X, names)
    Di = min(feats["ikaros"][0].shape[1], feats["daedalus"][0].shape[1])
    for h in ("ikaros", "daedalus"):
        X, names = feats[h]
        feats[h] = (X[:, :Di], names[:Di])

    # ------------------------------------------------------------------
    # 1) 100-seed re-run, canonical assignment
    # ------------------------------------------------------------------
    print("[1] 100-seed re-run, canonical assignment")
    rerun = {}
    for h in ("ikaros", "daedalus"):
        X, names = feats[h]
        fb = canonical_feat_block(names)
        rng_block = np.random.default_rng(7)
        rand_fb = rng_block.integers(0, len(PhysicsAwareReservoir.BLOCKS), size=len(names))
        phys = run_cell(h, names, X, N_SEEDS_VERIFY, fb)
        rand_struct = run_cell("random", names, X, N_SEEDS_VERIFY, rand_fb)
        rand_assign_same_struct = run_cell(h, names, X, N_SEEDS_VERIFY, rand_fb)
        g_vs_random = gain_pct(phys, rand_struct)
        g_vs_random_mean, g_lo, g_hi = bootstrap_gain_ci(phys, rand_struct)
        g_vs_assign = gain_pct(phys, rand_assign_same_struct)
        g_vs_assign_mean, ga_lo, ga_hi = bootstrap_gain_ci(phys, rand_assign_same_struct)
        rerun[h] = {
            "phys_nrmse_mean": float(np.mean(phys)),
            "rand_struct_nrmse_mean": float(np.mean(rand_struct)),
            "rand_assign_nrmse_mean": float(np.mean(rand_assign_same_struct)),
            "gain_vs_random_pct": g_vs_random,
            "gain_vs_random_ci95": [g_lo, g_hi],
            "gain_vs_assign_pct": g_vs_assign,
            "gain_vs_assign_ci95": [ga_lo, ga_hi],
            "survives": (g_lo > 0),
        }
        print(f"  {h}: phys={np.mean(phys):.4f} rand_struct={np.mean(rand_struct):.4f} rand_assign={np.mean(rand_assign_same_struct):.4f}")
        print(f"      gain_vs_random={g_vs_random:+.2f}%  CI95=[{g_lo:+.2f}, {g_hi:+.2f}]  SURVIVES={g_lo>0}")
    out["rerun_100seed"] = rerun

    # ------------------------------------------------------------------
    # 2) Permutation test: shuffle which class label gets which block.
    # Use fewer seeds (20) per permutation to keep budget manageable.
    # ------------------------------------------------------------------
    print("[2] Permutation test (50 random class->block permutations, 20 seeds each)")
    permute = {}
    for h in ("ikaros", "daedalus"):
        X, names = feats[h]
        fb_canon = canonical_feat_block(names)
        # baseline reference for this host (rand assignment, n_seeds=20)
        rng_block = np.random.default_rng(7)
        rand_fb = rng_block.integers(0, len(PhysicsAwareReservoir.BLOCKS), size=len(names))
        ref = run_cell(h, names, X, 20, rand_fb)
        ref_m = float(np.mean(ref))
        gains = []
        rng_perm = np.random.default_rng(123)
        for k in range(N_PERMUTE):
            perm = rng_perm.permutation(len(PhysicsAwareReservoir.BLOCKS))
            fb_perm = perm[fb_canon]
            phys_k = run_cell(h, names, X, 20, fb_perm)
            g = (ref_m - float(np.mean(phys_k))) / max(ref_m, 1e-9) * 100.0
            gains.append(g)
        # Also canonical at same n_seeds=20 for fair comparison
        canon_20 = run_cell(h, names, X, 20, fb_canon)
        g_canon = (ref_m - float(np.mean(canon_20))) / max(ref_m, 1e-9) * 100.0
        n_pos = int(sum(1 for g in gains if g > 0))
        n_beats_canon = int(sum(1 for g in gains if g >= g_canon))
        permute[h] = {
            "canonical_gain_pct": g_canon,
            "permuted_gains_pct": gains,
            "permuted_mean_gain": float(np.mean(gains)),
            "permuted_median_gain": float(np.median(gains)),
            "permuted_max_gain": float(np.max(gains)),
            "n_perms_positive": n_pos,
            "n_perms_beat_canonical": n_beats_canon,
            "fraction_positive": n_pos / N_PERMUTE,
        }
        print(f"  {h}: canonical_20s={g_canon:+.2f}%  perm_mean={np.mean(gains):+.2f}%  perm_median={np.median(gains):+.2f}%  pos={n_pos}/{N_PERMUTE}  beat_canon={n_beats_canon}/{N_PERMUTE}")
    out["permutation"] = permute

    # ------------------------------------------------------------------
    # 3) Cross-host transplant: use daedalus's canonical assignment on ikaros
    # ------------------------------------------------------------------
    print("[3] Cross-host: daedalus assignment on ikaros (and vice versa)")
    cross = {}
    # Note: canonical assignment is *determined by feature names only*; since
    # ikaros and daedalus feature schemas are identical (same capture code),
    # canonical_feat_block(names_ikaros) == canonical_feat_block(names_daedalus).
    # The only host-specific bit is the chassi seed (HASHES) for W_in/W_rec.
    # We swap the *chassi seed source* instead.
    for eval_h in ("ikaros", "daedalus"):
        X, names = feats[eval_h]
        fb = canonical_feat_block(names)
        rng_block = np.random.default_rng(7)
        rand_fb = rng_block.integers(0, len(PhysicsAwareReservoir.BLOCKS), size=len(names))
        own = run_cell(eval_h, names, X, 30, fb)
        other = "daedalus" if eval_h == "ikaros" else "ikaros"
        transplant = run_cell(other, names, X, 30, fb)
        ref = run_cell(eval_h, names, X, 30, rand_fb)
        ref_m = float(np.mean(ref))
        g_own = (ref_m - float(np.mean(own))) / max(ref_m, 1e-9) * 100.0
        g_trans = (ref_m - float(np.mean(transplant))) / max(ref_m, 1e-9) * 100.0
        cross[eval_h] = {
            "own_seed_phys_nrmse": float(np.mean(own)),
            "transplant_seed_phys_nrmse": float(np.mean(transplant)),
            "rand_baseline_nrmse": ref_m,
            "gain_own_pct": g_own,
            "gain_transplant_pct": g_trans,
        }
        print(f"  eval={eval_h}: own_seed_gain={g_own:+.2f}%  transplant_seed_gain={g_trans:+.2f}%")
    out["cross_host"] = cross

    # ------------------------------------------------------------------
    # 4) Class-ablation: drop one feature class at a time from canonical
    #    assignment (move to "other" block) and measure gain change.
    # ------------------------------------------------------------------
    print("[4] Class-ablation on daedalus (drop one class at a time)")
    h = "daedalus"
    X, names = feats[h]
    fb_canon = canonical_feat_block(names)
    other_idx = PhysicsAwareReservoir.BLOCKS.index("other")
    rng_block = np.random.default_rng(7)
    rand_fb = rng_block.integers(0, len(PhysicsAwareReservoir.BLOCKS), size=len(names))
    ref = run_cell(h, names, X, 30, rand_fb)
    ref_m = float(np.mean(ref))
    canon_30 = run_cell(h, names, X, 30, fb_canon)
    canon_gain = (ref_m - float(np.mean(canon_30))) / max(ref_m, 1e-9) * 100.0
    ablations = {"_baseline_canonical": canon_gain}
    blocks = PhysicsAwareReservoir.BLOCKS
    for bi, bname in enumerate(blocks):
        if bname == "other": continue
        fb_ab = fb_canon.copy()
        fb_ab[fb_ab == bi] = other_idx
        if not (fb_ab != fb_canon).any():
            ablations[bname] = None
            continue
        ab_nrmse = run_cell(h, names, X, 30, fb_ab)
        g = (ref_m - float(np.mean(ab_nrmse))) / max(ref_m, 1e-9) * 100.0
        ablations[bname] = {"gain_pct_when_dropped": g, "delta_from_canonical": g - canon_gain}
        n_in = int((fb_canon == bi).sum())
        print(f"  drop {bname} (n={n_in}): gain={g:+.2f}%  delta={g - canon_gain:+.2f}%")
    out["class_ablation_daedalus"] = ablations

    out["runtime_s"] = time.time() - t0
    (OUT_DIR / "physics_verify.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"saved physics_verify.json  runtime={out['runtime_s']:.1f}s")


if __name__ == "__main__":
    main()
