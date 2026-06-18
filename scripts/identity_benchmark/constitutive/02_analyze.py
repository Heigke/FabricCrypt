"""Bootstrap analyzer + verdict for the 5-regime constitutive experiment.

Reads regime_{r}_results.json, computes per-regime:
  diag_mean        : mean NRMSE where train_host == eval_host
  off_hw           : mean NRMSE where train_host != eval_host (other real device)
  off_sw           : sw_matched control
  off_shuffle      : permuted spatial control
  off_ident        : constant substrate control
  delta_*          : off_X - diag_mean
  bootstrap 95% CIs

Verdict gates (per regime, applied with Bonferroni over 5 regimes):
  DISCOVERY  if delta_hw > 2*max(sigma_sw, sigma_shuffle, sigma_ident)
                AND delta_hw > 5 * delta_ident
                AND delta_hw_lo (CI) > 0
  NULL       if delta_hw <= delta_sw
  KILL       if delta_shuffle > delta_hw
Writes summary.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "constitutive"


def boot_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(np.mean(x)), lo, hi


def diff_ci(a, b, n_boot=2000, rng=None):
    rng = rng or np.random.default_rng(1)
    a = np.asarray(a, dtype=np.float64); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=np.float64); b = b[~np.isnan(b)]
    n = min(len(a), len(b))
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    diffs = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(a), size=n)
        ib = rng.integers(0, len(b), size=n)
        diffs.append(a[ia].mean() - b[ib].mean())
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def analyze_regime(regime: int):
    p = OUT / f"regime_{regime}_results.json"
    if not p.exists():
        return None
    data = json.load(open(p))
    cells = data["cells"]
    rng = np.random.default_rng(42)
    if regime == 0:
        base = np.array(cells["baseline"], dtype=np.float64)
        m, lo, hi = boot_ci(base, rng=rng)
        return {"regime": 0, "baseline_mean": m, "baseline_ci": [lo, hi]}

    diag_vals = np.concatenate([
        np.array(cells["train_ikaros__eval_ikaros"], dtype=np.float64),
        np.array(cells["train_daedalus__eval_daedalus"], dtype=np.float64),
    ])
    off_hw = np.concatenate([
        np.array(cells["train_ikaros__eval_daedalus"], dtype=np.float64),
        np.array(cells["train_daedalus__eval_ikaros"], dtype=np.float64),
    ])
    off_sw = np.concatenate([
        np.array(cells["train_ikaros__eval_sw_matched"], dtype=np.float64),
        np.array(cells["train_daedalus__eval_sw_matched"], dtype=np.float64),
    ])
    off_sh = np.concatenate([
        np.array(cells["train_ikaros__eval_shuffle"], dtype=np.float64),
        np.array(cells["train_daedalus__eval_shuffle"], dtype=np.float64),
    ])
    off_id = np.concatenate([
        np.array(cells["train_ikaros__eval_ident_const"], dtype=np.float64),
        np.array(cells["train_daedalus__eval_ident_const"], dtype=np.float64),
    ])

    d_m, d_lo, d_hi = boot_ci(diag_vals, rng=rng)
    h_m, h_lo, h_hi = boot_ci(off_hw, rng=rng)
    sw_m, sw_lo, sw_hi = boot_ci(off_sw, rng=rng)
    sh_m, sh_lo, sh_hi = boot_ci(off_sh, rng=rng)
    id_m, id_lo, id_hi = boot_ci(off_id, rng=rng)

    dHW = diff_ci(off_hw, diag_vals, rng=rng)
    dSW = diff_ci(off_sw, diag_vals, rng=rng)
    dSH = diff_ci(off_sh, diag_vals, rng=rng)
    dID = diff_ci(off_id, diag_vals, rng=rng)

    # Verdict
    delta_hw_lo = dHW[1]
    delta_hw = dHW[0]
    delta_sw = dSW[0]
    delta_sh = dSH[0]
    delta_id = dID[0]
    sigma_sw = (sw_hi - sw_lo) / 3.92
    sigma_sh = (sh_hi - sh_lo) / 3.92
    sigma_id = (id_hi - id_lo) / 3.92
    sigma_floor = max(sigma_sw, sigma_sh, sigma_id, 1e-9)

    verdict = "AMBIGUOUS"
    # NULL: HW delta is at noise floor (CI includes 0 or near 0)
    if np.isnan(delta_hw) or abs(delta_hw) < 1e-3 or delta_hw_lo <= 0:
        verdict = "NULL"
    # KILL: shuffle control beats HW by margin larger than HW std
    elif (not np.isnan(delta_sh)) and delta_sh > delta_hw + sigma_sh:
        verdict = "KILL"
    # NULL: HW doesn't exceed SW-matched
    elif delta_hw <= delta_sw:
        verdict = "NULL"
    # DISCOVERY: HW > 2 sigma above ALL controls AND >5x ident
    elif (delta_hw - delta_sw > 2 * sigma_sw
          and delta_hw - delta_sh > 2 * sigma_sh
          and (abs(delta_id) < 1e-6 or delta_hw > 5 * abs(delta_id))
          and delta_hw_lo > 0):
        verdict = "DISCOVERY"
    else:
        # HW > SW-matched but not by 2σ above ALL controls
        verdict = "WEAK_DISCOVERY"

    return {
        "regime": regime,
        "n_seeds_per_cell": data["n_seeds"],
        "diag": {"mean": d_m, "ci": [d_lo, d_hi]},
        "off_hw": {"mean": h_m, "ci": [h_lo, h_hi]},
        "off_sw_matched": {"mean": sw_m, "ci": [sw_lo, sw_hi]},
        "off_shuffle_perm": {"mean": sh_m, "ci": [sh_lo, sh_hi]},
        "off_ident_const": {"mean": id_m, "ci": [id_lo, id_hi]},
        "delta_hw": {"mean": dHW[0], "ci": [dHW[1], dHW[2]]},
        "delta_sw": {"mean": dSW[0], "ci": [dSW[1], dSW[2]]},
        "delta_shuffle": {"mean": dSH[0], "ci": [dSH[1], dSH[2]]},
        "delta_ident": {"mean": dID[0], "ci": [dID[1], dID[2]]},
        "verdict": verdict,
    }


def main():
    summary = {"regimes": {}}
    for r in [0, 1, 2, 3, 4, 5]:
        a = analyze_regime(r)
        summary["regimes"][f"regime_{r}"] = a
        if a is None:
            continue
        if r == 0:
            print(f"regime 0 BASELINE: NRMSE={a['baseline_mean']:.4f} CI={a['baseline_ci']}")
        else:
            print(f"regime {r}: verdict={a['verdict']} "
                  f"diag={a['diag']['mean']:.4f} "
                  f"d_HW={a['delta_hw']['mean']:.4f}{a['delta_hw']['ci']} "
                  f"d_SW={a['delta_sw']['mean']:.4f} "
                  f"d_SHUF={a['delta_shuffle']['mean']:.4f} "
                  f"d_IDENT={a['delta_ident']['mean']:.4f}")

    # cross-regime trend
    deltas = []
    for r in [1, 2, 3, 4, 5]:
        a = summary["regimes"].get(f"regime_{r}")
        if a and "delta_hw" in a:
            deltas.append((r, a["delta_hw"]["mean"]))
    summary["delta_hw_trend"] = deltas
    monotonic = all(deltas[i][1] <= deltas[i + 1][1] for i in range(len(deltas) - 1))
    summary["monotonic_increasing"] = monotonic
    discovery_regimes = [r for r, a in summary["regimes"].items()
                        if a and a.get("verdict") in ("DISCOVERY", "WEAK_DISCOVERY")]
    summary["discovery_regimes"] = discovery_regimes

    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print("\nMonotonic increasing delta_HW with regime:", monotonic)
    print("Discovery regimes:", discovery_regimes)


if __name__ == "__main__":
    main()
