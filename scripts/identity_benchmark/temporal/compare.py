#!/usr/bin/env python3
"""
temporal/compare.py — compare two devices' temporal feature dictionaries,
report Cohen-d-like effect (z-score given single point per device), produce
verdict against pre-registered gates.

We have ONE feature scalar per device (n=1 trial). So we use a degraded
proxy: normalized absolute difference relative to feature mean magnitude:
   z ≈ |f_A - f_B| / mean(|f_A|, |f_B|)
and we flag any feature where z >= 0.5 (50%-level disagreement). Genuine
Cohen-d requires repeats — Phase 2 will add them.

Per pre-registration:
  DISCOVERY: ANY single T2-T7 feature with z >= 0.5 AND not in T1 (envelope)
  KILL: all T2-T7 features collapse below 0.2.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def flatten(d: dict, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (int, float)):
            out[key] = float(v)
    return out


def z_proxy(a: float, b: float) -> float:
    import math
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan")
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="features json A")
    ap.add_argument("--b", required=True, help="features json B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    A = flatten(json.loads(Path(args.a).read_text()))
    B = flatten(json.loads(Path(args.b).read_text()))
    common = sorted(set(A) & set(B))

    rows = []
    for k in common:
        z = z_proxy(A[k], B[k])
        rows.append({"feature": k, "a": A[k], "b": B[k], "z_proxy": z})
    rows.sort(key=lambda r: (r["z_proxy"] if r["z_proxy"] == r["z_proxy"] else -1),
              reverse=True)

    # per-group top
    groups = {}
    for r in rows:
        g = r["feature"].split(".")[0]
        groups.setdefault(g, []).append(r)
    top_by_group = {g: rs[0] for g, rs in groups.items() if rs}

    # gate evaluation
    nontrivial = [r for r in rows if r["feature"].split(".")[0]
                  not in ("meta", "T1_static")]
    max_z = max((r["z_proxy"] for r in nontrivial if r["z_proxy"] == r["z_proxy"]),
                default=0.0)
    discovery = max_z >= 0.5
    kill = max_z < 0.2

    out = {
        "n_features": len(rows),
        "top_overall": rows[:10],
        "top_by_group": top_by_group,
        "max_z_nontrivial": max_z,
        "verdict": ("DISCOVERY" if discovery else
                    "KILL" if kill else "AMBIGUOUS"),
        "note": ("n=1/device — z_proxy is a proxy for Cohen-d. "
                  "Discovery threshold here is 0.5 normalized abs diff."),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[OK] {args.out}")
    print(f"max_z={max_z:.3f}  verdict={out['verdict']}")
    print("\nTop 5 features:")
    for r in rows[:5]:
        print(f"  {r['feature']:50s}  a={r['a']:.4g} b={r['b']:.4g} z={r['z_proxy']:.3f}")
    print("\nTop per group:")
    for g, r in top_by_group.items():
        print(f"  [{g}] {r['feature']:40s}  z={r['z_proxy']:.3f}")


if __name__ == "__main__":
    main()
