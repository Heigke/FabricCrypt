"""Analyze L1-L15 probe results across machines and reps.

For each numeric feature:
  - within-machine std (ikaros r0 vs r1, daedalus r0 vs r1) -> noise floor
  - between-machine diff (ikaros r0 vs daedalus r0)
  - Cohen d = |mean_ikaros - mean_daedalus| / pooled_std
  - within/between ratio = between/within (>1 means separable)

Top-K features by Cohen d, with provenance.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path

BASE = Path("results/IDENTITY_BENCHMARK_2026-05-30/lowest_level")

def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        kk = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, kk))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[kk] = float(v)
        elif isinstance(v, str):
            # categorical -> hash to int for diff
            out[kk + "__str_hash"] = float(hash(v) & 0xFFFFFFFF)
    return out

def load_run(machine, rep):
    d = BASE / f"{machine}_{rep}"
    feats = {}
    for p in sorted(d.glob("L*_*.json")):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        probe = r.get("probe", p.stem)
        feat = r.get("features", {})
        ff = flatten(feat)
        for k, v in ff.items():
            feats[f"{probe}::{k}"] = v
    return feats

def main():
    runs = {}
    for m in ("ikaros", "daedalus"):
        for r in ("r0", "r1"):
            runs[f"{m}_{r}"] = load_run(m, r)
            print(f"  loaded {m}_{r}: {len(runs[f'{m}_{r}'])} features", file=sys.stderr)

    # Common feature set across all 4 runs
    common = set(runs["ikaros_r0"])
    for k in ("ikaros_r1","daedalus_r0","daedalus_r1"):
        common &= set(runs[k])
    print(f"common features (all 4 runs): {len(common)}", file=sys.stderr)

    rows = []
    for f in common:
        ik = [runs["ikaros_r0"][f], runs["ikaros_r1"][f]]
        de = [runs["daedalus_r0"][f], runs["daedalus_r1"][f]]
        ik_m = sum(ik)/2; de_m = sum(de)/2
        ik_s = math.sqrt(((ik[0]-ik_m)**2 + (ik[1]-ik_m)**2)/2)
        de_s = math.sqrt(((de[0]-de_m)**2 + (de[1]-de_m)**2)/2)
        within = (ik_s + de_s) / 2
        between = abs(ik_m - de_m)
        pooled = math.sqrt((ik_s**2 + de_s**2)/2)
        cohen_d = (between / pooled) if pooled > 1e-12 else (float('inf') if between > 1e-12 else 0.0)
        ratio = (between / within) if within > 1e-12 else (float('inf') if between > 1e-12 else 0.0)
        rows.append({
            "feature": f, "ikaros_mean": ik_m, "daedalus_mean": de_m,
            "within_std": within, "between_abs": between,
            "cohen_d": cohen_d if math.isfinite(cohen_d) else 9999.0,
            "ratio": ratio if math.isfinite(ratio) else 9999.0,
        })

    rows.sort(key=lambda r: r["cohen_d"], reverse=True)

    out_path = BASE / "_analysis.json"
    json.dump({"n_features": len(rows), "top_50": rows[:50],
               "constants_diff": [r for r in rows if r["within_std"]==0 and r["between_abs"]>0][:30]},
              open(out_path, "w"), indent=2)

    # Markdown report
    md = BASE / "_analysis.md"
    with open(md, "w") as f:
        f.write("# L1-L15 Lowest-Level Probe Analysis\n\n")
        f.write(f"- Features compared (present in all 4 runs): **{len(rows)}**\n")
        f.write(f"- Reps per machine: 2 (drift baseline)\n\n")
        f.write("## Top 30 by Cohen d (between-machine / pooled-within)\n\n")
        f.write("| # | Feature | ikaros | daedalus | within_std | between | Cohen d | ratio |\n")
        f.write("|---|---------|--------|----------|------------|---------|---------|-------|\n")
        for i, r in enumerate(rows[:30], 1):
            f.write(f"| {i} | `{r['feature'][:80]}` | {r['ikaros_mean']:.4g} | {r['daedalus_mean']:.4g} | {r['within_std']:.3g} | {r['between_abs']:.3g} | {r['cohen_d']:.2f} | {r['ratio']:.2f} |\n")
        f.write("\n## Constants that differ between machines (within_std==0, between>0)\n\n")
        consts = [r for r in rows if r["within_std"]==0 and r["between_abs"]>0]
        f.write(f"Count: **{len(consts)}**\n\n")
        f.write("| # | Feature | ikaros | daedalus |\n|---|---------|--------|----------|\n")
        for i, r in enumerate(consts[:30], 1):
            f.write(f"| {i} | `{r['feature'][:80]}` | {r['ikaros_mean']:.4g} | {r['daedalus_mean']:.4g} |\n")
        # Per-probe summary
        f.write("\n## Per-probe summary (mean Cohen d, top feature)\n\n")
        per_probe = {}
        for r in rows:
            p = r["feature"].split("::")[0]
            per_probe.setdefault(p, []).append(r)
        f.write("| Probe | n_features | mean Cohen d | max Cohen d | top feature |\n")
        f.write("|-------|-----------|--------------|-------------|-------------|\n")
        for p in sorted(per_probe):
            rs = per_probe[p]
            cds = [x["cohen_d"] for x in rs if math.isfinite(x["cohen_d"])]
            if not cds: continue
            top = max(rs, key=lambda x: x["cohen_d"])
            f.write(f"| {p} | {len(rs)} | {sum(cds)/len(cds):.2f} | {max(cds):.2f} | `{top['feature'].split('::')[1][:60]}` |\n")

    print(f"wrote {out_path} and {md}")

if __name__ == "__main__":
    main()
