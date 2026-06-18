"""Phase 2 analysis: compute transplant Δ metric + bootstrap CI + verdict markdown."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEVICES = ["ikaros", "daedalus"]
CONTROLS = ["HW", "SW_iid", "SHUFFLE"]


def bootstrap_ci(x: np.ndarray, n_boot: int = 5000, alpha: float = 0.05,
                 seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    boots = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(np.mean(x)), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = json.loads(args.inp.read_text())
    rows = data["rows"]

    def group(device: str, control: str):
        return np.array([r["nrmse"] for r in rows
                         if r["device"] == device and r["control"] == control])

    # per (device, control) summary
    summary = {}
    for d in DEVICES:
        for c in CONTROLS:
            arr = group(d, c)
            m, lo, hi = bootstrap_ci(arr)
            summary[f"{d}/{c}"] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "n": len(arr)}

    # Δ_transplant: |mean(HW@ikaros) - mean(HW@daedalus)| compared with
    #   reference: |mean(SHUFFLE@ikaros) - mean(SHUFFLE@daedalus)|
    hw_ik = group("ikaros", "HW")
    hw_dd = group("daedalus", "HW")
    sh_ik = group("ikaros", "SHUFFLE")
    sh_dd = group("daedalus", "SHUFFLE")
    iid_ik = group("ikaros", "SW_iid")
    iid_dd = group("daedalus", "SW_iid")

    delta_hw = abs(hw_ik.mean() - hw_dd.mean())
    delta_sh = abs(sh_ik.mean() - sh_dd.mean())
    delta_iid = abs(iid_ik.mean() - iid_dd.mean())

    # Bootstrap CI on the difference-of-means itself
    rng = np.random.default_rng(0)
    n_boot = 5000
    boots_hw, boots_sh, boots_iid = [], [], []
    for _ in range(n_boot):
        b_hw_ik = rng.choice(hw_ik, size=hw_ik.size, replace=True).mean()
        b_hw_dd = rng.choice(hw_dd, size=hw_dd.size, replace=True).mean()
        boots_hw.append(abs(b_hw_ik - b_hw_dd))
        b_sh_ik = rng.choice(sh_ik, size=sh_ik.size, replace=True).mean()
        b_sh_dd = rng.choice(sh_dd, size=sh_dd.size, replace=True).mean()
        boots_sh.append(abs(b_sh_ik - b_sh_dd))
        b_iid_ik = rng.choice(iid_ik, size=iid_ik.size, replace=True).mean()
        b_iid_dd = rng.choice(iid_dd, size=iid_dd.size, replace=True).mean()
        boots_iid.append(abs(b_iid_ik - b_iid_dd))
    def q(x):
        return float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))
    hw_lo, hw_hi = q(boots_hw)
    sh_lo, sh_hi = q(boots_sh)
    iid_lo, iid_hi = q(boots_iid)

    # Verdict
    # HW Δ must be > both control Δ upper-CI to claim identity-driven divergence
    hw_above_sh = delta_hw > sh_hi
    hw_above_iid = delta_hw > iid_hi
    if hw_above_sh and hw_above_iid:
        verdict = "DISCOVERY"
        desc = ("HW transplant Δ exceeds BOTH shuffle and SW-iid controls' "
                "upper 95% CI — device identity persists at the task level.")
    elif hw_above_sh or hw_above_iid:
        verdict = "AMBITIOUS"
        desc = ("HW Δ exceeds one but not both controls — partial identity "
                "signal; substrate shape matters for one channel.")
    elif delta_hw < min(sh_lo, iid_lo):
        verdict = "KILL"
        desc = ("HW Δ is BELOW the shuffle-noise floor — no transplantable "
                "identity at the task level. The Phase 1c-surviving channels "
                "are statistical artefacts, not identity.")
    else:
        verdict = "NULL"
        desc = ("HW Δ within control-CI envelope — no detectable identity "
                "signal beyond control noise.")

    out = {
        "verdict": verdict,
        "description": desc,
        "summary": summary,
        "delta_hw": delta_hw, "delta_hw_ci": [hw_lo, hw_hi],
        "delta_shuffle": delta_sh, "delta_shuffle_ci": [sh_lo, sh_hi],
        "delta_iid": delta_iid, "delta_iid_ci": [iid_lo, iid_hi],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    # Markdown
    md_path = args.out.with_suffix(".md")
    lines = ["# Identity Benchmark — Phase 2 (Transplant Matrix)", "",
             f"Verdict: **{verdict}**", "", desc, "",
             "## Δ summary", "",
             "| pair | Δ NRMSE | 95% CI |",
             "|---|---|---|",
             f"| HW(ikaros) vs HW(daedalus) | {delta_hw:.4f} | [{hw_lo:.4f}, {hw_hi:.4f}] |",
             f"| SW-iid control             | {delta_iid:.4f} | [{iid_lo:.4f}, {iid_hi:.4f}] |",
             f"| SHUFFLE control            | {delta_sh:.4f} | [{sh_lo:.4f}, {sh_hi:.4f}] |",
             "", "## Per-condition NRMSE (mean ± CI)", "",
             "| device | control | mean | CI lo | CI hi | n |",
             "|---|---|---|---|---|---|"]
    for k, v in summary.items():
        d, c = k.split("/")
        lines.append(f"| {d} | {c} | {v['mean']:.4f} | {v['ci_lo']:.4f} | {v['ci_hi']:.4f} | {v['n']} |")
    lines += ["", "## Method",
              "- 128-neuron tanh ESN, spectral radius 0.9, leak 0.3, ridge α=1e-4",
              "- Substrate hooks: per-CU RTN-rate (multiplicative gain) + spatial-corr (colored additive noise) from Phase 1b raw_idle.npz",
              "- NARMA-10 task, T_train=2000, T_test=500, washout=100",
              "- 10 seeds × 3 controls × 2 devices = 60 runs",
              "- SHUFFLE = device's own marginals with permuted CU index (destroys identity, preserves stats)",
              "- SW-iid = uniform marginals + identity spatial covariance",
              ""]
    md_path.write_text("\n".join(lines))
    print(f"[analyze] verdict={verdict}")
    print(f"[analyze] wrote {args.out} + {md_path}")


if __name__ == "__main__":
    main()
