"""Consolidate H7 Phase 0 across machines: load all phase0_analysis_{host}.json, build a comparison
table + a figure. Run after analyzing each host's npz. Out: phase0_summary.{md,png}.
"""
from __future__ import annotations
import json, glob
from pathlib import Path
OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-16"

def main():
    rows = []
    for p in sorted(OUT.glob("phase0_analysis_*.json")):
        j = json.loads(p.read_text()); host = j.get("host", p.stem)
        t1 = j.get("T1_deflection", {}).get("C1_SELF", {})
        t2 = j.get("T2_reafference", {}); t3 = j.get("T3_decode_meta", {}).get("token_rate", {})
        t4 = j.get("T4_content", {})
        rows.append({
            "host": host, "n_ch": j.get("n_channels"), "n_fast": j.get("n_fast"),
            "C1_n_moved": t1.get("n_fast_moved_|d|>0.5"), "C1_power_d": t1.get("power_d"),
            "decode_rate_full": t3.get("r2_full"), "decode_rate_1D": t3.get("r2_power_1D"),
            "decode_gain_30D": t3.get("gain_full_over_1D"),
            "reaff_delta": t2.get("delta_r2_token"), "entropy_adds": t4.get("entropy_adds"),
        })
    md = ["# H7 Phase 0 — cross-machine summary\n",
          "| host | #ch | #fast | C1 moved(|d|>.5) | power d | decode-rate FULL | decode-rate 1D | 30D gain | reaff Δ | entropy adds |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| {host} | {n_ch} | {n_fast} | {C1_n_moved} | {C1_power_d} | {decode_rate_full} | "
                  "{decode_rate_1D} | {decode_gain_30D} | {reaff_delta} | {entropy_adds} |".format(**r))
    md += ["",
           "**Read:** LLM→body coupling (C1 moved / power d) is the robust cross-die finding. "
           "decode-rate FULL>0 = body telemetry decodes the LLM's generation rate (introspection / "
           "Eric's meta-computation). 30D gain = what the full ~30-50-ch vector buys over 1-D power. "
           "reaff Δ≈0 and entropy adds≈0 = content/identity does NOT drive the loop (intensity-mediated).",
           "", "GPU-gen dies (ikaros, zgx) carry rate in GPU power; CPU-gen (daedalus) does not — config-matched comparison required."]
    (OUT/"phase0_summary.md").write_text("\n".join(md))
    print("\n".join(md)); print(f"\nsaved {OUT/'phase0_summary.md'}")
    # optional figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        hosts = [r["host"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].bar(hosts, [r["C1_power_d"] or 0 for r in rows], color="#3fb950")
        ax[0].set_title("LLM→body coupling (power Cohen's d)"); ax[0].set_ylabel("d")
        ax[1].bar(hosts, [r["decode_rate_full"] or 0 for r in rows], color="#58a6ff", label="FULL 30-50D")
        ax[1].bar(hosts, [r["decode_rate_1D"] or 0 for r in rows], color="#d29922", alpha=.6, label="1-D power")
        ax[1].set_title("body→decode LLM rate (R²)"); ax[1].axhline(0, color="k", lw=.5); ax[1].legend()
        fig.tight_layout(); fig.savefig(OUT/"phase0_summary.png", dpi=110); print(f"saved {OUT/'phase0_summary.png'}")
    except Exception as e:
        print("fig skipped:", e)

if __name__ == "__main__":
    main()
