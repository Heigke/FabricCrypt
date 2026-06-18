"""Load A_power + B_thermal + E_cpu signatures from the deep sweep and build a
fixed 23-feature per-device substrate vector.

Layout (23 features):
  A_power (4 workloads * 2 stats = 8):
      [mean_W, std_W] for each of {IDLE, LIGHT, MEDIUM, HEAVY}
  B_thermal (3): [tau_heat, tau_cool, R_th_K_per_W]
  E_cpu  (12 = 6 mean+std of per-core blocks):
      [time_p25, time_med, time_p75, time_std,
       freq_med, freq_std, rank_corr_pearson,
       time_min, time_max,
       freq_min, freq_max,
       time_iqr]

The vector is z-scored across devices on each call (so both substrates live
in the same scale) and returned with channel labels.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEEP = ROOT / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "deep"

WORKLOADS = ["IDLE", "LIGHT", "MEDIUM", "HEAVY"]


def _load(host: str) -> dict:
    out = {}
    for name in ("A_power", "B_thermal", "E_cpu"):
        p = DEEP / host / f"{name}.json"
        with p.open() as f:
            out[name] = json.load(f)
    return out


def _vec_A(A: dict) -> np.ndarray:
    # File schema: top-level dict workload -> {ikaros_mean_W, daedalus_mean_W, ikaros_std, daedalus_std,...}
    # Per host though, the A_power.json file itself is per-host. We expect keys
    # like {"IDLE": {"mean_W":..,"std_W":..}, ...} or fall back to ANALYSIS.json layout.
    feats = []
    for w in WORKLOADS:
        if w in A and isinstance(A[w], dict) and "mean_W" in A[w]:
            feats.append(float(A[w]["mean_W"])); feats.append(float(A[w]["std_W"]))
        else:
            # Fall back: assume samples list
            samples = A.get(w, {}).get("samples_W") or A.get(w, [])
            arr = np.asarray(samples, dtype=float)
            if arr.size == 0:
                feats.extend([0.0, 0.0])
            else:
                feats.append(float(arr.mean())); feats.append(float(arr.std()))
    return np.asarray(feats, dtype=float)


def _vec_B(B: dict) -> np.ndarray:
    keys = ("tau_heat", "tau_cool", "R_th_K_per_W")
    feats = []
    for k in keys:
        v = B.get(k)
        if isinstance(v, dict):
            # try mean
            for alt in ("mean", "value", "val"):
                if alt in v:
                    feats.append(float(v[alt])); break
            else:
                feats.append(float(list(v.values())[0]))
        elif isinstance(v, (int, float)):
            feats.append(float(v))
        elif isinstance(v, list):
            feats.append(float(np.mean(v)))
        else:
            feats.append(0.0)
    return np.asarray(feats, dtype=float)


def _vec_E(E: dict) -> np.ndarray:
    t = np.asarray(E.get("per_core_time", []), dtype=float)
    f = np.asarray(E.get("per_core_freq", []), dtype=float)
    if t.size == 0:
        t = np.zeros(8)
    if f.size == 0:
        f = np.zeros(8)
    feats = [
        float(np.percentile(t, 25)),
        float(np.median(t)),
        float(np.percentile(t, 75)),
        float(np.std(t)),
        float(np.median(f)),
        float(np.std(f)),
        float(E.get("rank_corr_pearson", 0.0)),
        float(np.min(t)),
        float(np.max(t)),
        float(np.min(f)),
        float(np.max(f)),
        float(np.percentile(t, 75) - np.percentile(t, 25)),
    ]
    return np.asarray(feats, dtype=float)


def _from_analysis(host: str) -> Tuple[np.ndarray, list]:
    """Fallback: build feature vector from the joint ANALYSIS.json since the
    per-host raw files don't have raw samples in the expected shape."""
    a = json.loads((DEEP / "ANALYSIS.json").read_text())
    feats = []
    labels = []
    # A: feature_vector field gives [i_mean, i_std, d_mean, d_std] per workload
    fv = a["A"]["feature_vector"]
    for w in WORKLOADS:
        i_m, i_s, d_m, d_s = fv[w]
        if host == "ikaros":
            feats.extend([i_m, i_s])
        else:
            feats.extend([d_m, d_s])
        labels.extend([f"A_{w}_mean", f"A_{w}_std"])
    # B
    B = a["B"]
    for k in ("tau_heat", "tau_cool", "R_th_K_per_W"):
        m = B[k][f"{host}_mean"]
        feats.append(float(m))
        labels.append(f"B_{k}")
    # E
    E = a["E"]
    t = np.asarray(E[f"{host}_per_core_time"], dtype=float)
    f = np.asarray(E[f"{host}_per_core_freq"], dtype=float)
    e_feats = [
        float(np.percentile(t, 25)), float(np.median(t)), float(np.percentile(t, 75)),
        float(np.std(t)), float(np.median(f)), float(np.std(f)),
        float(E.get("rank_corr_pearson", 0.0)),
        float(np.min(t)), float(np.max(t)),
        float(np.min(f)), float(np.max(f)),
        float(np.percentile(t, 75) - np.percentile(t, 25)),
    ]
    feats.extend(e_feats)
    labels.extend([
        "E_t_p25", "E_t_med", "E_t_p75", "E_t_std",
        "E_f_med", "E_f_std", "E_rank_corr",
        "E_t_min", "E_t_max", "E_f_min", "E_f_max", "E_t_iqr",
    ])
    return np.asarray(feats, dtype=float), labels


def load_pair():
    """Return ikaros_vec (23,), daedalus_vec (23,), labels list (len 23),
    z-scored jointly so off-diagonal magnitude is meaningful."""
    vi, labels = _from_analysis("ikaros")
    vd, _ = _from_analysis("daedalus")
    M = np.vstack([vi, vd])  # (2, 23)
    mu = M.mean(axis=0)
    sd = M.std(axis=0) + 1e-12
    Z = (M - mu) / sd
    return Z[0], Z[1], labels, (vi, vd)


def matched_gaussian(vec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Replace vec with iid Gaussian matched in mean+std (per-feature)."""
    return rng.standard_normal(vec.shape) * (vec.std() + 1e-12) + vec.mean()


if __name__ == "__main__":
    zi, zd, labels, raw = load_pair()
    print(f"23 features. ikaros L2={np.linalg.norm(zi):.3f}  daedalus L2={np.linalg.norm(zd):.3f}")
    print(f"||ikaros-daedalus||_2 = {np.linalg.norm(zi - zd):.3f}")
    for k, name in enumerate(labels):
        print(f"  {k:2d} {name:18s}  i={raw[0][k]:10.4f}  d={raw[1][k]:10.4f}  z_i={zi[k]:+.2f}  z_d={zd[k]:+.2f}")
