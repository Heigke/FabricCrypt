"""Phase 10 — simulator-based transplant penalty using REAL fitted RC params.

We could not write PWM on either host (hp driver rejects writes), so we ran
calibration traces and fit per-host thermal RC params from real square-wave
load-response data. This script uses those *real* RC params to compute the
transplant penalty in simulator — which is at least one step closer to the
hardware than Phase 9's purely-assumed params.

Also runs robustness ablations:
  - learned_ikaros on ikaros sim    (baseline A)
  - learned_ikaros on daedalus sim  (transplant)
  - learned_daedalus on daedalus sim
  - learned_daedalus on ikaros sim
  - learned_shuffle_ikaros on ikaros sim   (temporal control)
  - random_init on each sim
  - constant_mid on each sim
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from fan_real import (LinearController, ConstantController, simulate_episode,
                      cem_train, TARGET_C, EPISODE_S, EP_DT, bootstrap_diff_pct)

OUT_DIR = Path(__file__).resolve().parents[3] / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment10"

def rmse_to_target(Ts, target):
    Ts = np.asarray(Ts)
    return float(np.sqrt(np.mean((Ts[len(Ts)//2:] - target) ** 2)))


def main():
    ic = json.loads((OUT_DIR / "ikaros_calib.json").read_text())
    dc = json.loads((OUT_DIR / "daedalus_calib.json").read_text())
    rc_i = ic["rc_params"]; rc_d = dc["rc_params"]
    print("[sim] ikaros RC:", rc_i)
    print("[sim] daedalus RC:", rc_d)
    thetas = {
        "learned_ikaros": np.array(ic["theta_normal"], dtype=np.float32),
        "learned_daedalus": np.array(dc["theta_normal"], dtype=np.float32),
        "learned_shuffle_ikaros": np.array(ic["theta_shuffle"], dtype=np.float32),
        "random_init": np.array(ic["theta_random"], dtype=np.float32),
    }
    sims = {"ikaros": rc_i, "daedalus": rc_d}

    N_SEEDS = 50
    target = TARGET_C
    results = {}
    for sim_h, rc in sims.items():
        per_cond = {}
        for name, th in thetas.items():
            ctrl = LinearController(theta=th)
            errs = []
            for s in range(N_SEEDS):
                Ts, _ = simulate_episode(rc, ctrl, dur=EPISODE_S, dt=EP_DT,
                                         T_init=42.0 + 2.0 * np.random.default_rng(s).normal(),
                                         target_c=target, seed=s)
                errs.append(rmse_to_target(Ts, target))
            per_cond[name] = errs
        for cname, cval in (("constant_mid", 0.5), ("off", 0.0)):
            ctrl = ConstantController(val=cval)
            errs = []
            for s in range(N_SEEDS):
                Ts, _ = simulate_episode(rc, ctrl, dur=EPISODE_S, dt=EP_DT,
                                         T_init=42.0 + 2.0 * np.random.default_rng(s).normal(),
                                         target_c=target, seed=s)
                errs.append(rmse_to_target(Ts, target))
            per_cond[cname] = errs
        results[sim_h] = per_cond
        print(f"\n[sim] On {sim_h}:")
        for name, errs in per_cond.items():
            print(f"  {name:24s} RMSE={np.mean(errs):.3f} +/- {np.std(errs):.3f}")

    # transplant penalties: compute % penalty of transplant vs native
    penalties = {}
    for sim_h in sims:
        native_key = f"learned_{sim_h}"
        other_h = "daedalus" if sim_h == "ikaros" else "ikaros"
        transplant_key = f"learned_{other_h}"
        a = results[sim_h][native_key]
        b = results[sim_h][transplant_key]
        pct, lo, hi = bootstrap_diff_pct(a, b)
        penalties[sim_h] = {"native_mean_rmse": float(np.mean(a)),
                            "transplant_mean_rmse": float(np.mean(b)),
                            "transplant_penalty_pct": pct,
                            "ci95": [lo, hi],
                            "GATE_PASS_20pct": (pct >= 20.0 and lo > 0)}
        # also compare native vs shuffle, random, constant
        for ctrl_name in ("learned_shuffle_ikaros", "random_init", "constant_mid"):
            c = results[sim_h][ctrl_name]
            ppct, plo, phi = bootstrap_diff_pct(a, c)
            penalties[sim_h][f"vs_{ctrl_name}_pct"] = ppct
            penalties[sim_h][f"vs_{ctrl_name}_ci"] = [plo, phi]
        print(f"\n[sim] Transplant penalty on {sim_h}: native={np.mean(a):.3f} transplant={np.mean(b):.3f} penalty={pct:+.2f}% CI95=[{lo:+.2f}, {hi:+.2f}] PASS_20pct={penalties[sim_h]['GATE_PASS_20pct']}")

    out = {
        "rc_ikaros": rc_i, "rc_daedalus": rc_d,
        "n_seeds": N_SEEDS, "target_c": target,
        "per_sim_per_cond_rmse": results,
        "penalties": penalties,
    }
    (OUT_DIR / "sim_transplant.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nsaved sim_transplant.json")


if __name__ == "__main__":
    main()
