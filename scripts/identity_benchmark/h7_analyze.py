#!/usr/bin/env python3
"""H7 analyzer — per-channel within-chassis vs cross-chassis discriminability.

Reads every .npz in results/IDENTITY_H7_2026-06-09/ and reports for each
channel: mean, std, Cohen's d ikaros-vs-daedalus, simple thresholded AUC,
and whether the channel separates the two real AMD chassis under idle load.

This is the first-pass within-the-day analyzer; the full pre-registered
block-CV + matched-spectrum + replay pipeline lives in h7_analyze_full.py
once we have ≥5 traces per (host, load) cell.

Output: results/IDENTITY_H7_2026-06-09/h7_first_pass.md  (+ .json).
"""
import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "results/IDENTITY_H7_2026-06-09"
OUT_MD = DIR / "h7_first_pass.md"
OUT_JSON = DIR / "h7_first_pass.json"


def cohen_d(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sp = math.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2)
                   / (len(a)+len(b)-2))
    if sp == 0:
        return float("inf") if a.mean() != b.mean() else 0.0
    return (a.mean() - b.mean()) / sp


def simple_auc(a, b):
    """Pairwise probability that a-draw > b-draw. Equivalent to Mann-Whitney U / n1*n2."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.5
    aa = np.tile(a[:, None], (1, n2))
    bb = np.tile(b[None, :], (n1, 1))
    wins = (aa > bb).sum() + 0.5 * (aa == bb).sum()
    return float(wins) / (n1 * n2)


def load_run(path):
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    out = {
        "path": path.name,
        "host": meta["host"],
        "load": meta["load"],
        "ambient": meta["ambient"],
        "tpm_ek": meta.get("tpm", {}).get("ek_name", ""),
        "pcrs": meta.get("tpm", {}).get("pcrs", {}),
    }
    # SMN: [ts, c0..c15, base_th, e0,e1,e2, fast, xtal, gfx_vid, soc_vid]
    smn = d["smn"]
    out["smn"] = smn
    out["pm_vals"] = d["pm_vals"]
    out["tsc_drift"] = d["tsc_drift"]
    out["gpu_bar2"] = d["gpu_bar2"]
    return out


def channel_views(run):
    """Return dict channel_name -> 1D numpy array of samples for this run."""
    v = {}
    smn = run["smn"]
    if smn.size and smn.shape[1] >= 25:
        for i in range(16):
            v[f"C03_core{i:02d}_thermal"] = smn[:, 1 + i].astype(np.float64)
        v["C04_base_thermal_C"] = ((smn[:, 17] >> 21) & 0x7FF) * 0.125
        for j, name in enumerate(("C05_e0", "C05_e1", "C05_e2")):
            v[name] = smn[:, 18 + j].astype(np.float64)
        v["C06_fast"] = smn[:, 21].astype(np.float64)
        v["C07_xtal_cntl"] = smn[:, 22].astype(np.float64)
        v["C08_gfx_vid"] = smn[:, 23].astype(np.float64)
        v["C08_soc_vid"] = smn[:, 24].astype(np.float64)
        # C20 SMN read latencies (ns) — added after O100 oracle synthesis
        if smn.shape[1] >= 28:
            v["C20_lat_base_thermal"] = smn[:, 25].astype(np.float64)
            v["C20_lat_energy0"]      = smn[:, 26].astype(np.float64)
            v["C20_lat_xtal"]         = smn[:, 27].astype(np.float64)
    pm = run["pm_vals"]
    if pm.size and pm.ndim == 2:
        for i in (1, 3, 5, 30, 31, 110, 130, 170, 194):
            if i < pm.shape[1]:
                v[f"C09_pm[{i}]"] = pm[:, i].astype(np.float64)
    tsc = run["tsc_drift"]
    if tsc.size and tsc.shape[1] >= 5:
        # column layout (time_ns, a1, b1, a2, b2). drift = (a2-a1)-(b2-b1)
        gap_a = (tsc[:, 3] - tsc[:, 1]).astype(np.int64)
        gap_b = (tsc[:, 4] - tsc[:, 2]).astype(np.int64)
        v["C11_drift_ns_per_step"] = (gap_a - gap_b).astype(np.float64)
    g = run["gpu_bar2"]
    if g.size and g.shape[1] >= 3:
        # columns [ts, clock_lsb, clock_msb, ...8 status regs...]
        # clock delta per sample
        lsb = g[:, 1].astype(np.uint64)
        msb = g[:, 2].astype(np.uint64)
        clk = (msb << 32) | lsb
        clk_delta = np.diff(clk.astype(np.int64), prepend=clk[0])
        v["C18_gpu_clock_delta"] = clk_delta.astype(np.float64)
        for i, name in enumerate(("GRBM_STATUS", "GRBM_STATUS2", "GRBM_STATUS_SE0",
                                  "GRBM_STATUS_SE1", "SRBM_STATUS", "CP_STAT",
                                  "RLC_STAT", "RLC_GPM_STAT")):
            v[f"C19_{name}"] = g[:, 3 + i].astype(np.float64)
    return v


def main():
    runs = []
    for p in sorted(DIR.glob("*.npz")):
        try:
            runs.append(load_run(p))
        except Exception as e:
            print(f"[skip] {p.name}: {e}")
    if not runs:
        print("no runs found"); return

    print(f"loaded {len(runs)} runs:")
    for r in runs:
        print(f"  {r['host']:8s} {r['load']:6s} {r['ambient']:10s} {r['path']}")

    # Group runs by host (only handling host-discrimination on idle for now)
    by_host = {}
    for r in runs:
        by_host.setdefault(r["host"], []).append(r)

    if len(by_host) < 2:
        print("\nOnly one host present — no cross-chassis stats possible yet.")
        return

    # Combine samples per host per channel
    channel_data = {}      # name -> {host -> np.array}
    for host, host_runs in by_host.items():
        for r in host_runs:
            for ch, vec in channel_views(r).items():
                channel_data.setdefault(ch, {}).setdefault(host, []).append(vec)

    rows = []
    for ch, hosts in sorted(channel_data.items()):
        if not all(h in hosts for h in by_host):
            continue
        pooled = {h: np.concatenate(arrs) for h, arrs in hosts.items()}
        hs = sorted(pooled.keys())
        a, b = pooled[hs[0]], pooled[hs[1]]
        # drop NaN/inf and constants
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        if len(a) == 0 or len(b) == 0:
            continue
        d = cohen_d(a, b)
        auc = simple_auc(a, b)
        gap_auc = max(auc, 1 - auc)        # discriminability either direction
        u_a = len(np.unique(a)); u_b = len(np.unique(b))
        rows.append({
            "channel": ch,
            "host_a": hs[0], "n_a": int(len(a)), "mean_a": float(a.mean()), "std_a": float(a.std()),
            "host_b": hs[1], "n_b": int(len(b)), "mean_b": float(b.mean()), "std_b": float(b.std()),
            "cohen_d_signed": float(d), "cohen_d_abs": float(abs(d)),
            "auc": float(auc), "discrim_auc": float(gap_auc),
            "uniques_a": u_a, "uniques_b": u_b,
        })

    rows.sort(key=lambda r: r["discrim_auc"], reverse=True)

    OUT_JSON.write_text(json.dumps({
        "preregistration": "research_plan/H7_PREREG_2026-06-09.md",
        "n_runs": len(runs),
        "hosts": list(by_host.keys()),
        "rows": rows,
    }, indent=2))

    lines = []
    lines.append("# H7 first-pass — within-day cross-chassis discriminability\n")
    lines.append(f"Source: `{DIR}` ({len(runs)} runs, hosts={list(by_host.keys())})")
    lines.append(f"Pre-registration: `research_plan/H7_PREREG_2026-06-09.md`")
    lines.append("")
    lines.append("This is a **first-pass** report. The pre-registered acceptance gates"
                 " (block-CV AUC, matched-spectrum spoof, thermal match, replay) are NOT"
                 " applied here — they will be enforced once we have ≥5 traces per"
                 " (host, load) cell. The numbers below are the raw separability of"
                 " each channel from a single 20-second idle baseline per chassis.")
    lines.append("")
    lines.append("## TPM ground-truth")
    for r in runs:
        lines.append(f"- {r['host']}: EK={r['tpm_ek']}  PCR0={(r['pcrs'].get('0') or '')[:18]}…")
    lines.append("")
    lines.append("## Channel table (sorted by discriminative AUC, highest first)")
    lines.append("| channel | n_a | n_b | mean_a | mean_b | d | AUC | flag |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        flag = ""
        if r["discrim_auc"] >= 0.95 and r["cohen_d_abs"] >= 3.0:
            flag = "★ candidate"
        elif r["discrim_auc"] >= 0.80:
            flag = "↑ promising"
        elif r["discrim_auc"] >= 0.60:
            flag = "weak"
        else:
            flag = "—"
        lines.append(f"| {r['channel']} | {r['n_a']} | {r['n_b']} | "
                     f"{r['mean_a']:.3g} | {r['mean_b']:.3g} | "
                     f"{r['cohen_d_signed']:+.2f} | {r['discrim_auc']:.3f} | {flag} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("- **★ candidate** = both AUC≥0.95 AND |d|≥3 in this single-trace pair."
                 " That clears the *point-estimate* level of the pre-registered gate."
                 " It does NOT yet clear matched-spectrum spoofing, thermal-matching,"
                 " or replay-from-log — those need more traces and the cross-temp set.")
    lines.append("- **↑ promising** = AUC≥0.80 but not 0.95. Often these are chassis-confounds"
                 " (PSU, fan, NVMe) that survive crude classification but are designed to fail"
                 " the spoof+thermal gate.")
    lines.append("- Channels at AUC≈0.5 are not carrying chassis identity in this trace.")
    OUT_MD.write_text("\n".join(lines))
    print(f"\nwrote {OUT_MD}")
    print(f"wrote {OUT_JSON}")
    cand = [r for r in rows if r["discrim_auc"] >= 0.95 and r["cohen_d_abs"] >= 3.0]
    promising = [r for r in rows if 0.80 <= r["discrim_auc"] < 0.95]
    print(f"point-estimate candidates: {len(cand)}  (top 5: {[r['channel'] for r in cand[:5]]})")
    print(f"promising: {len(promising)}")


if __name__ == "__main__":
    main()
