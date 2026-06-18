"""z142 vs z139 topology comparison.

Reads z139 (Bf=2e4 unphysical) and z142 (η-bounded honest) summaries, prints
ranking comparison, writes markdown artifact for the O20 oracle packet.

Runs against partial summary if full summary.json not yet written.
"""
from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Z139 = ROOT / "results/z139_largescale_topology/summary.json"
Z142 = ROOT / "results/z142_topology_v2/summary.json"
Z142_PARTIAL = ROOT / "results/z142_topology_v2/summary_partial.json"
OUT = ROOT / "research_plan/oracle_queries/O20_M3b_closure/artifacts/z142_vs_z139_table.md"

TOPOS = ["ER_SPARSE", "LAYERED", "RAND_GAUSS", "MESH_4N", "WS_SMALLWORLD", "HUB_SPOKE"]
NS = [100, 300, 800]


def _row_iter(summary: dict):
    rows = summary.get("results", summary)
    for k, v in rows.items():
        if not isinstance(v, dict):
            continue
        yield v


def aggregate(summary: dict, norm_filter: str | None = None):
    """Return {(topo, N, norm): {metric: (mean, sd, n)}}."""
    g = defaultdict(lambda: defaultdict(list))
    for r in _row_iter(summary):
        topo, N = r.get("topo"), r.get("N")
        norm = r.get("rho_variant", r.get("rho_norm", "rho_lambda"))
        if topo is None or N is None:
            continue
        if norm_filter and norm != norm_filter:
            continue
        for m in ("MC", "NARMA_NRMSE", "XOR_acc", "WAVE_acc"):
            v = r.get(m)
            if v is None:
                continue
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(vf):
                continue
            g[(topo, int(N), norm)][m].append(vf)
    out = {}
    for k, mdict in g.items():
        out[k] = {m: (st.mean(vs), st.stdev(vs) if len(vs) > 1 else 0.0, len(vs))
                  for m, vs in mdict.items()}
    return out


def fmt(cell):
    if cell is None:
        return "  --   "
    mean, sd, n = cell
    return f"{mean:5.2f} (n={n})"


def main():
    z139 = json.loads(Z139.read_text())
    z142_path = Z142 if Z142.exists() else Z142_PARTIAL
    z142 = json.loads(z142_path.read_text())
    is_partial = z142_path == Z142_PARTIAL

    # z139: single norm (rho_lambda)
    g139 = aggregate(z139)
    # z142: three norms
    g142 = {n: aggregate(z142, norm_filter=n)
            for n in ["rho_lambda", "rho_p95_sv", "rho_deg_norm"]}

    lines = []
    lines.append("# z142 vs z139 — topology MC ranking comparison")
    lines.append("")
    lines.append(f"**Source:** `{z142_path.relative_to(ROOT)}` "
                 f"({'PARTIAL' if is_partial else 'FULL'})")
    lines.append("")
    lines.append("## N=800 MC across normalisations")
    lines.append("")
    lines.append("| topology       | z139 (Bf=2e4) | z142 rho_lambda | z142 rho_p95_sv | z142 rho_deg_norm |")
    lines.append("|----------------|--------------:|----------------:|----------------:|-------------------:|")
    for topo in TOPOS:
        row = [topo.ljust(14)]
        c139 = g139.get((topo, 800, "rho_lambda"), {}).get("MC")
        row.append(fmt(c139))
        for norm in ["rho_lambda", "rho_p95_sv", "rho_deg_norm"]:
            c = g142[norm].get((topo, 800, norm), {}).get("MC")
            row.append(fmt(c))
        lines.append("| " + " | ".join(row) + " |")

    # ranking summary at rho_lambda
    lines.append("")
    lines.append("## N=800 rho_lambda ranking change (z139 → z142)")
    lines.append("")
    lines.append("| topology       | z139 MC | z142 MC | Δ      | rank z139 | rank z142 |")
    lines.append("|----------------|--------:|--------:|-------:|----------:|----------:|")
    z139_pairs, z142_pairs = [], []
    for topo in TOPOS:
        c139 = g139.get((topo, 800, "rho_lambda"), {}).get("MC")
        c142 = g142["rho_lambda"].get((topo, 800, "rho_lambda"), {}).get("MC")
        if c139:
            z139_pairs.append((topo, c139[0]))
        if c142:
            z142_pairs.append((topo, c142[0]))
    rank139 = {t: i + 1 for i, (t, _) in enumerate(sorted(z139_pairs, key=lambda x: -x[1]))}
    rank142 = {t: i + 1 for i, (t, _) in enumerate(sorted(z142_pairs, key=lambda x: -x[1]))}
    for topo in TOPOS:
        c139 = g139.get((topo, 800, "rho_lambda"), {}).get("MC")
        c142 = g142["rho_lambda"].get((topo, 800, "rho_lambda"), {}).get("MC")
        m139 = c139[0] if c139 else float("nan")
        m142 = c142[0] if c142 else float("nan")
        delta = m142 - m139 if (c139 and c142) else float("nan")
        lines.append(f"| {topo:14s} | {m139:7.2f} | {m142:7.2f} | {delta:+6.2f} "
                     f"| {rank139.get(topo, '--'):>9} | {rank142.get(topo, '--'):>9} |")

    # scaling check: rho_lambda MC at all 3 N
    lines.append("")
    lines.append("## rho_lambda MC across scale (z142 honest cell)")
    lines.append("")
    lines.append("| topology       | N=100 | N=300 | N=800 | scaling N=100→800 |")
    lines.append("|----------------|------:|------:|------:|------------------:|")
    for topo in TOPOS:
        cells = []
        m100 = m800 = None
        for N in NS:
            c = g142["rho_lambda"].get((topo, N, "rho_lambda"), {}).get("MC")
            if c:
                cells.append(f"{c[0]:.2f}")
                if N == 100:
                    m100 = c[0]
                if N == 800:
                    m800 = c[0]
            else:
                cells.append(" -- ")
        scale_ratio = f"{m800 / m100:.2f}×" if (m100 and m800 and m100 > 0.1) else " -- "
        lines.append(f"| {topo:14s} | {cells[0]:>5s} | {cells[1]:>5s} | {cells[2]:>5s} "
                     f"| {scale_ratio:>17s} |")

    # multi-task table at N=800 rho_lambda
    lines.append("")
    lines.append("## Multi-task ranking at N=800 rho_lambda (honest cell, n=5)")
    lines.append("")
    lines.append("| topology       | MC    | NARMA NRMSE↓ | XOR_acc | WAVE_acc |")
    lines.append("|----------------|------:|-------------:|--------:|---------:|")
    multi = []
    for topo in TOPOS:
        cells = []
        row_metrics = {}
        for m in ("MC", "NARMA_NRMSE", "XOR_acc", "WAVE_acc"):
            c = g142["rho_lambda"].get((topo, 800, "rho_lambda"), {}).get(m)
            if c:
                cells.append(f"{c[0]:.2f}")
                row_metrics[m] = c[0]
            else:
                cells.append(" -- ")
        multi.append((topo, row_metrics))
        lines.append(f"| {topo:14s} | {cells[0]:>5s} | {cells[1]:>12s} "
                     f"| {cells[2]:>7s} | {cells[3]:>8s} |")

    # find champion per task
    def champ(metric, lower_better=False):
        valid = [(t, m[metric]) for t, m in multi if metric in m]
        if not valid:
            return None
        return min(valid, key=lambda x: x[1])[0] if lower_better else max(valid, key=lambda x: x[1])[0]

    lines.append("")
    lines.append("## Per-task champion (z142 honest cell, N=800 rho_lambda)")
    lines.append("")
    lines.append(f"- **MC**:    {champ('MC')}")
    lines.append(f"- **NARMA**: {champ('NARMA_NRMSE', lower_better=True)} (lower is better)")
    lines.append(f"- **XOR**:   {champ('XOR_acc')}")
    lines.append(f"- **WAVE**:  {champ('WAVE_acc')}")

    # WAVE flip narrative for HUB_SPOKE
    lines.append("")
    lines.append("## HUB_SPOKE WAVE classification — z139 → z142")
    lines.append("")
    c139w = g139.get(("HUB_SPOKE", 800, "rho_lambda"), {}).get("WAVE_acc")
    c142w = g142["rho_lambda"].get(("HUB_SPOKE", 800, "rho_lambda"), {}).get("WAVE_acc")
    if c139w and c142w:
        lines.append(f"- z139 WAVE: **{c139w[0]:.3f}** (was the brief's classification champion)")
        lines.append(f"- z142 WAVE: **{c142w[0]:.3f}** — advantage gone")
    else:
        lines.append("(awaiting full data)")

    lines.append("")
    lines.append("## Headline")
    if z139_pairs and z142_pairs:
        champ139 = max(z139_pairs, key=lambda x: x[1])[0]
        champ142 = max(z142_pairs, key=lambda x: x[1])[0]
        lines.append(f"- z139 N=800 rho_lambda MC champion: **{champ139}**")
        lines.append(f"- z142 N=800 rho_lambda MC champion: **{champ142}**")
        lines.append(f"- Inversion: {'YES' if champ139 != champ142 else 'no'}")
    if is_partial:
        lines.append("")
        lines.append("**NOTE:** z142 is partial. Results above are from rows present "
                     "in `summary_partial.json` at write time.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[wrote {OUT.relative_to(ROOT)}]")


if __name__ == "__main__":
    main()
