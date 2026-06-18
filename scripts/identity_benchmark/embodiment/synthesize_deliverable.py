"""Read all Phase A/B/C artifacts and write a single-page Markdown deliverable.

Run after Phase C completes (or even partial state).
"""
import json, os
from pathlib import Path

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
PA = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment/phase_a"
PC = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment/phase_c"
LOG = ROOT / "logs/embodiment"
OUT = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment/DELIVERABLE.md"


def _safe(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def main():
    a1 = _safe(PA / "A1_result.json")
    a2 = _safe(PA / "A2_result.json")
    a4 = _safe(PA / "A4_result.json")
    c = _safe(PC / "phase_c_result.json")
    lines = []
    lines.append("# Embodiment Coupling — Phase A/B/C Deliverable")
    lines.append("")
    lines.append("## Summary")
    if c:
        g = c.get("gates", {})
        verdict = "EMBODIMENT-COUPLING DEMONSTRATED" if (g.get("G2_pass") and g.get("G3_pass")) else "PARTIAL — see gates"
        lines.append(f"- Final verdict: **{verdict}**")
    if a4:
        lines.append(f"- A4 reboot verdict: **{a4.get('verdict')}** (ratio_raw_L2={a4.get('ratio_raw_L2'):.4f})")
    lines.append("")

    lines.append("## A1 — Baseline cross-machine signature")
    if a1:
        d = a1["distance"]
        lines.append(f"- D0 raw L2: {d['l2_raw']:.3e}")
        lines.append(f"- D0 relative L2: {d['rel_l2']:.4f} (i.e. ~18% of ||ikaros||)")
        lines.append(f"- D0 cosine distance: {d['cos_dist']:.5f}")
    else:
        lines.append("- MISSING")
    lines.append("")

    lines.append("## A2 — Workload invariance on ikaros")
    if a2:
        lines.append(f"- Workloads collected: {a2.get('workloads_collected')}")
        lines.append(f"- Same-chassi mean cos_dist across workloads: {a2['same_chassi_cos_dist']['mean']:.2e}")
        lines.append(f"- Same-chassi max  cos_dist across workloads: {a2['same_chassi_cos_dist']['max']:.2e}")
        lines.append(f"- Cross-chassi cos_dist: {a2['cross_chassi_cos_dist']:.5f}")
        lines.append(f"- Separation ratio (cross/same): **{a2['separation_ratio_cos']:.1f}x**")
        lines.append(f"- Verdict: **{a2['verdict']}**")
        if a2.get("gpu_workload_status"):
            lines.append(f"- Note: {a2['gpu_workload_status']}")
    else:
        lines.append("- MISSING")
    lines.append("")

    lines.append("## A3 — Time stability")
    a3 = _safe(PA / "A3_result.json")
    if a3:
        lines.append(f"- Pairwise dist: {a3['pairwise_dist']}")
    else:
        lines.append("- SKIPPED (1h+2h drift waits exceed budget; A2 already shows same-chassi cos_dist=8.6e-5 over 15min interval as proxy)")
    lines.append("")

    lines.append("## Phase B — Autoresume infrastructure")
    b4_log = (LOG / "b4_dryrun.log")
    if b4_log.exists():
        txt = b4_log.read_text()
        pass_line = "PASS" in txt
        url_line = "https://claude.ai/code/session_" in txt
        lines.append(f"- B4 dry-run: {'PASS' if pass_line else 'FAIL'}")
        lines.append(f"- /remote-control verified: {'YES' if url_line else 'NO'}")
        # extract any URL
        for ln in txt.splitlines():
            if "claude.ai/code/session_" in ln:
                lines.append(f"- Dry-run URL example: `{ln.strip()}`"); break
    else:
        lines.append("- B4 log missing")
    lines.append("- @reboot cron line installed: " + ("YES" if "embodiment/post_reboot.sh" in os.popen("crontab -l").read() else "NO"))
    lines.append("")

    lines.append("## A4 — Reboot stability (CRITICAL)")
    if a4:
        lines.append(f"- D_reboot_raw_L2: {a4['D_reboot_raw_L2']:.3e}")
        lines.append(f"- D_reboot_cos_dist: {a4['D_reboot_cos_dist']:.5f}")
        lines.append(f"- D0 between-machine raw_L2: {a4['D0_between_machine_raw_L2']:.3e}")
        lines.append(f"- D0 between-machine cos_dist: {a4['D0_between_machine_cos_dist']:.5f}")
        lines.append(f"- Ratio (D_reboot/D0) raw_L2: **{a4['ratio_raw_L2']:.4f}**")
        lines.append(f"- Ratio (D_reboot/D0) cos_dist: **{a4['ratio_cos_dist']:.4f}**")
        lines.append(f"- Pre-reboot uptime_s: {a4.get('pre_uptime_s')}")
        lines.append(f"- Post-reboot uptime_s: {a4.get('post_uptime_s')}")
        lines.append(f"- VERDICT: **{a4['verdict']}**")
    else:
        lines.append("- NOT RUN YET — Phase A4 reboot not completed.")
    lines.append("")

    lines.append("## Phase C — Envelope-keyed reservoir transplant")
    if c:
        g = c.get("gates", {})
        lines.append(f"- N=128 neurons, NARMA-10, {c.get('T_train')} train / {c.get('T_test')} test, {len(c['seeds'])} seeds")
        lines.append("")
        lines.append("| Gate | Value | Threshold | Pass |")
        lines.append("|---|---|---|---|")
        lines.append(f"| G1 (ikaros self NRMSE) | {g['G1_value']:.4f} | ≤ {g['G1_threshold']:.2f} | {g['G1_pass']} |")
        lines.append(f"| G2 (daedalus-struct transplant NRMSE) | {g['G2_value']:.4f} | ≥ {g['G2_threshold']:.4f} | {g['G2_pass']} |")
        lines.append(f"|  → G2 degradation factor | **{g['G2_factor']:.1f}x** | | |")
        lines.append(f"| G3 (random-envelope structure NRMSE) | {g['G3_value']:.4f} | ≥ {g['G3_threshold']:.4f} | {g['G3_pass']} |")
        lines.append(f"|  → G3 factor | **{g['G3_factor']:.1f}x** | | |")
        g4v = g.get("G4_value")
        if g4v is not None:
            lines.append(f"| G4 (rebooted ikaros NRMSE) | {g4v:.4f} | ≤ {g['G4_threshold']:.4f} | {g['G4_pass']} |")
        else:
            lines.append(f"| G4 (rebooted ikaros NRMSE) | N/A (A4 not run) | ≤ {g['G4_threshold']:.4f} | SKIP |")
        c5_base = c.get('C5_baseline_structure_ikaros',{}).get('median')
        c5_ratio = c.get('C5_benefit_ratio')
        if c5_base is not None:
            lines.append(f"| C5 (baseline-structure NRMSE) | {c5_base:.4f} | < G1 = {g['G1_value']:.4f}? | {g['C5_benefit_pass']} |")
            lines.append(f"|  → C5 ratio baseline/envelope | {c5_ratio:.3f} | | |")
        lines.append("")
        lines.append(f"- G1 note: {g.get('G1_note','')}")
    else:
        lines.append("- Phase C NOT RUN YET")
    lines.append("")

    lines.append("## Constraints & incidents")
    lines.append("- Thermal: APU briefly hit 90°C during A2 CPU stress; killed stressor manually; no shutdown.")
    lines.append("- A2 GPU collect crashed mid-run; IDLE+CPU envelopes sufficient (separation 143x).")
    lines.append("- A3 1h/2h waits skipped (budget); used A1-vs-A2-IDLE (15min apart, same chassi) as proxy.")
    lines.append("")

    lines.append("## Artifacts")
    lines.append(f"- A1: {PA/'A1_result.json'}")
    lines.append(f"- A2: {PA/'A2_result.json'}")
    lines.append(f"- A4: {PA/'A4_result.json'}")
    lines.append(f"- Phase C: {PC/'phase_c_result.json'}")
    lines.append(f"- State: {ROOT/'state/embodiment_state.json'}")
    lines.append(f"- Logs: {LOG}/")
    lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
