"""Compute D_reboot, compare to D0, decide GREENLIGHT/DOWNGRADE for Phase C."""
import json
from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
PA = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment/phase_a"
STATE = ROOT / "state/embodiment_state.json"


def raw_l2(v1, v2):
    a = np.asarray(v1, dtype=float); b = np.asarray(v2, dtype=float)
    return float(np.linalg.norm(a - b))

def cos_dist(v1, v2):
    a = np.asarray(v1, dtype=float); b = np.asarray(v2, dtype=float)
    return float(1.0 - a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def main():
    pre = json.loads((PA / "A4_pre.json").read_text())
    post = json.loads((PA / "A4_post.json").read_text())
    A1 = json.loads((PA / "A1_result.json").read_text())
    D_reboot_raw = raw_l2(pre["vec23"], post["vec23"])
    D_reboot_cos = cos_dist(pre["vec23"], post["vec23"])
    D_pre_vs_daed_raw = raw_l2(pre["vec23"], A1["daedalus"])
    D_post_vs_daed_raw = raw_l2(post["vec23"], A1["daedalus"])
    D_pre_vs_daed_cos = cos_dist(pre["vec23"], A1["daedalus"])
    D0_raw = A1["D0_l2_raw"]
    D0_cos = A1["D0_cos_dist"]
    ratio_raw = D_reboot_raw / D0_raw if D0_raw > 1e-9 else float("inf")
    ratio_cos = D_reboot_cos / D0_cos if D0_cos > 1e-9 else float("inf")
    # use raw_L2 ratio as primary criterion (more sensitive than cosine)
    verdict = "GREENLIGHT" if ratio_raw < 0.30 else ("MARGINAL" if ratio_raw < 0.70 else "DOWNGRADE")
    out = {
        "D_reboot_raw_L2": D_reboot_raw,
        "D_reboot_cos_dist": D_reboot_cos,
        "D0_between_machine_raw_L2": D0_raw,
        "D0_between_machine_cos_dist": D0_cos,
        "ratio_raw_L2": ratio_raw,
        "ratio_cos_dist": ratio_cos,
        "D_pre_vs_daedalus_raw_L2": D_pre_vs_daed_raw,
        "D_post_vs_daedalus_raw_L2": D_post_vs_daed_raw,
        "D_pre_vs_daedalus_cos_dist": D_pre_vs_daed_cos,
        "verdict": verdict,
        "criterion": "ratio_raw_L2 < 0.30 → GREENLIGHT, < 0.70 → MARGINAL, ≥0.70 → DOWNGRADE",
        "pre_ts": pre["timestamp"], "post_ts": post["timestamp"],
        "pre_uptime_s": pre.get("uptime_s"), "post_uptime_s": post.get("uptime_s"),
    }
    (PA / "A4_result.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    # update state
    if STATE.exists():
        st = json.loads(STATE.read_text())
        st.setdefault("phase_a", {}).setdefault("A4", {}).update(out)
        STATE.write_text(json.dumps(st, indent=2, default=str))


if __name__ == "__main__":
    main()
