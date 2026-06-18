"""Post-reboot: re-collect ikaros envelope, derive structure, run trained Phase C
model with NEW envelope structure, compute G4."""
from __future__ import annotations
import json, sys, time, hashlib, subprocess
from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
HERE = ROOT / "scripts/identity_benchmark/embodiment2"
PA = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment/phase_a"
OUT2 = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment2"
STATE = ROOT / "state/embodiment2_state.json"

sys.path.insert(0, str(HERE))
from _lib import (load_vec, derive_structure_v2, train_eval_task,
                  transplant_eval)
# also use the legacy phase_c derive_structure (3-axis)
sys.path.insert(0, str(ROOT / "scripts/identity_benchmark/embodiment"))
from phase_c_run import derive_structure as derive_structure_v1
from phase_c_run import baseline_structure as baseline_structure_v1
from phase_c_run import train_eval as train_eval_v1
from phase_c_run import transplant_eval as transplant_eval_v1
from phase_c_run import N as N_legacy

N = 128
SEEDS = 5


def main():
    OUT2.mkdir(parents=True, exist_ok=True)
    (OUT2 / "phase_a").mkdir(parents=True, exist_ok=True)
    # A4_pre + A4_post must exist
    pre = PA / "A4_pre.json"; post = PA / "A4_post.json"
    if not pre.exists() or not post.exists():
        print(f"[A4][FATAL] missing {pre} or {post}", flush=True)
        sys.exit(2)
    v_pre = json.loads(pre.read_text())["vec23"]
    v_post = json.loads(post.read_text())["vec23"]
    v_da = load_vec("daedalus")

    print(f"[A4] v_pre[:3]={v_pre[:3]}  v_post[:3]={v_post[:3]}", flush=True)
    print(f"[A4] L2(pre,post)={float(np.linalg.norm(np.array(v_pre)-np.array(v_post))):.4f}", flush=True)
    print(f"[A4] cos_dist(pre,post)={float(1.0 - np.dot(v_pre, v_post) / ((np.linalg.norm(v_pre)*np.linalg.norm(v_post))+1e-12)):.6f}", flush=True)

    # --- Legacy 3-axis (phase_c) G4 — direct extension of Phase C ---
    s_pre_v1 = derive_structure_v1(v_pre)
    s_post_v1 = derive_structure_v1(v_post)
    s_da_v1 = derive_structure_v1(v_da)
    g1, g4_post, g2_da = [], [], []
    for s in range(SEEDS):
        nr1, w = train_eval_v1(s_pre_v1, s)
        g1.append(nr1)
        nr4 = transplant_eval_v1(w, s_post_v1, s)
        g4_post.append(nr4)
        nr2 = transplant_eval_v1(w, s_da_v1, s)
        g2_da.append(nr2)
        print(f"[A4][v1] seed={s} G1={nr1:.4f} G4(post-reboot)={nr4:.4f} G2(daedalus)={nr2:.4f}", flush=True)

    g1_med = float(np.median(g1)); g4_med = float(np.median(g4_post)); g2_med = float(np.median(g2_da))
    g4_thresh = 1.5 * g1_med
    pass_g4 = g4_med <= g4_thresh

    # --- 5-axis (v2) G4 — what embodiment2 actually uses ---
    s_pre_v2 = derive_structure_v2(v_pre, N)
    s_post_v2 = derive_structure_v2(v_post, N)
    s_da_v2 = derive_structure_v2(v_da, N)
    g1v2, g4v2, g2v2 = [], [], []
    for s in range(SEEDS):
        nr1, w = train_eval_task(s_pre_v2, N, s, task="narma10")
        g1v2.append(nr1)
        nr4 = transplant_eval(w, s_post_v2, N, s, task="narma10")
        g4v2.append(nr4)
        nr2 = transplant_eval(w, s_da_v2, N, s, task="narma10")
        g2v2.append(nr2)
        print(f"[A4][v2] seed={s} G1={nr1:.4f} G4={nr4:.4f} G2={nr2:.4f}", flush=True)

    g1v2_med = float(np.median(g1v2)); g4v2_med = float(np.median(g4v2)); g2v2_med = float(np.median(g2v2))
    g4v2_thresh = 1.5 * g1v2_med
    pass_g4v2 = g4v2_med <= g4v2_thresh

    # mask bit overlap
    bit_overlap_v1 = float((s_pre_v1[0] == s_post_v1[0]).mean())
    bit_overlap_v2 = float((s_pre_v2["mask"] == s_post_v2["mask"]).mean())

    out = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "uptime_s": float(open("/proc/uptime").read().split()[0]),
        "v_pre_first3": v_pre[:3], "v_post_first3": v_post[:3],
        "envelope_l2_pre_post": float(np.linalg.norm(np.array(v_pre)-np.array(v_post))),
        "envelope_cos_dist_pre_post": float(1.0 - np.dot(v_pre, v_post) / ((np.linalg.norm(v_pre)*np.linalg.norm(v_post))+1e-12)),
        "v1_3axis": {
            "G1_pre_reboot": g1_med, "G4_post_reboot": g4_med, "G2_daedalus": g2_med,
            "G4_threshold": g4_thresh, "G4_ratio": g4_med / max(1e-9, g1_med),
            "G4_PASS": pass_g4,
            "mask_bit_overlap_pre_post": bit_overlap_v1,
            "G1_per_seed": g1, "G4_per_seed": g4_post, "G2_per_seed": g2_da,
            "n_seeds": SEEDS, "N": N_legacy,
        },
        "v2_5axis": {
            "G1_pre_reboot": g1v2_med, "G4_post_reboot": g4v2_med, "G2_daedalus": g2v2_med,
            "G4_threshold": g4v2_thresh, "G4_ratio": g4v2_med / max(1e-9, g1v2_med),
            "G4_PASS": pass_g4v2,
            "mask_bit_overlap_pre_post": bit_overlap_v2,
            "G1_per_seed": g1v2, "G4_per_seed": g4v2, "G2_per_seed": g2v2,
            "n_seeds": SEEDS, "N": N,
        },
        "PRIMARY_VERDICT_v2": "FULL_EMBODIMENT_4OF4_GATES" if pass_g4v2 else (
            "BOOT_STATE_BOUND_NOT_CHASSIS_BOUND" if g4v2_med > 10 * g1v2_med else
            "MARGINAL"),
    }
    (OUT2 / "A4_reboot_result.json").write_text(json.dumps(out, indent=2))
    print(f"[A4] === SUMMARY ===", flush=True)
    print(f"[A4] v1 3-axis: G1={g1_med:.4f} G4={g4_med:.4f} G4_ratio={out['v1_3axis']['G4_ratio']:.2f} PASS={pass_g4}", flush=True)
    print(f"[A4] v2 5-axis: G1={g1v2_med:.4f} G4={g4v2_med:.4f} G4_ratio={out['v2_5axis']['G4_ratio']:.2f} PASS={pass_g4v2}", flush=True)
    print(f"[A4] verdict: {out['PRIMARY_VERDICT_v2']}", flush=True)

    # update state
    STATE.parent.mkdir(parents=True, exist_ok=True)
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    st["A4"] = {k: v for k, v in out.items() if k not in ("v1_3axis", "v2_5axis")}
    st["A4"]["v1_PASS"] = pass_g4; st["A4"]["v2_PASS"] = pass_g4v2
    STATE.write_text(json.dumps(st, indent=2, default=str))
    print(f"[A4] wrote {OUT2 / 'A4_reboot_result.json'}", flush=True)


if __name__ == "__main__":
    main()
