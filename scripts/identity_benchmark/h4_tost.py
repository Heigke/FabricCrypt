#!/usr/bin/env python3
"""H4 TOST equivalence re-analysis of Phase-2 data.

Inputs:  results/IDENTITY_BENCHMARK_2026-05-30/phase2/matrix_results.json
Outputs: results/IDENTITY_NULL_2026-06-09/tost_phase2.md
         results/IDENTITY_NULL_2026-06-09/tost_phase2.json

Pre-registered in research_plan/H4_PREREG_2026-06-09.md.
Equivalence bound: ±0.005 nrmse. alpha=0.05, target 1-beta=0.9.
"""
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
SRC  = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/phase2/matrix_results.json"
OUT_DIR = ROOT / "results/IDENTITY_NULL_2026-06-09"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MD   = OUT_DIR / "tost_phase2.md"
OUT_JSON = OUT_DIR / "tost_phase2.json"

EQ_BOUND = 0.005
ALPHA    = 0.05
TARGET_POWER = 0.9


def t_cdf(t, df):
    # Hill 1970 approximation for Student-t CDF, accurate enough for p-value reporting.
    # For df>=2 the error is well under 1e-4 in the tails we care about.
    x = df / (df + t * t)
    # incomplete beta via series; use scipy-style only if available
    try:
        from math import lgamma
        # Use regularized incomplete beta via continued fraction.
        a = df / 2.0
        b = 0.5
        # log Beta(a,b)
        lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
        # incomplete beta I_x(a,b) by Lentz's continued fraction
        def betacf(a, b, x, itmax=200, eps=3e-12):
            qab = a + b
            qap = a + 1.0
            qam = a - 1.0
            c = 1.0
            d = 1.0 - qab * x / qap
            if abs(d) < 1e-30:
                d = 1e-30
            d = 1.0 / d
            h = d
            for m in range(1, itmax + 1):
                m2 = 2 * m
                aa = m * (b - m) * x / ((qam + m2) * (a + m2))
                d = 1.0 + aa * d
                if abs(d) < 1e-30:
                    d = 1e-30
                c = 1.0 + aa / c
                if abs(c) < 1e-30:
                    c = 1e-30
                d = 1.0 / d
                h *= d * c
                aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
                d = 1.0 + aa * d
                if abs(d) < 1e-30:
                    d = 1e-30
                c = 1.0 + aa / c
                if abs(c) < 1e-30:
                    c = 1e-30
                d = 1.0 / d
                delta = d * c
                h *= delta
                if abs(delta - 1.0) < eps:
                    break
            return h
        bt = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta)
        if x < (a + 1.0) / (a + b + 2.0):
            Ix = bt * betacf(a, b, x) / a
        else:
            Ix = 1.0 - bt * betacf(b, a, 1.0 - x) / b
        # CDF of t with df: F(t) = 1 - 0.5 * I_x(a,b) for t>0, else 0.5 * I_{1-x}(a,b)
        if t >= 0:
            return 1.0 - 0.5 * Ix
        else:
            return 0.5 * Ix
    except Exception:
        # crude normal fallback
        return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))


def welch_tost(x, y, lo, hi, alpha=0.05):
    nx, ny = len(x), len(y)
    mx, my = mean(x), mean(y)
    vx = stdev(x) ** 2 if nx > 1 else 0.0
    vy = stdev(y) ** 2 if ny > 1 else 0.0
    diff = mx - my
    se = math.sqrt(vx / nx + vy / ny) if (vx + vy) > 0 else 1e-12
    # Welch df
    if vx == 0 and vy == 0:
        df = nx + ny - 2
    else:
        df = (vx / nx + vy / ny) ** 2 / (
            (vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1)
        )
    # H0a: diff <= lo; H1a: diff > lo  -> t_lower = (diff - lo)/se, p_lower = 1 - F(t,df)
    t_lower = (diff - lo) / se
    p_lower = 1.0 - t_cdf(t_lower, df)
    # H0b: diff >= hi; H1b: diff < hi  -> t_upper = (diff - hi)/se, p_upper = F(t,df)
    t_upper = (diff - hi) / se
    p_upper = t_cdf(t_upper, df)
    equivalent = (p_lower < alpha) and (p_upper < alpha)
    return dict(diff=diff, se=se, df=df, t_lower=t_lower, p_lower=p_lower,
                t_upper=t_upper, p_upper=p_upper, equivalent=equivalent)


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    sx2 = stdev(x) ** 2 if nx > 1 else 0.0
    sy2 = stdev(y) ** 2 if ny > 1 else 0.0
    sp = math.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if sp == 0:
        return 0.0
    return (mean(x) - mean(y)) / sp


def mde_two_sample(s_pool, n_per_group, alpha=0.05, power=0.9):
    # Approximate MDE for two-sample t at given alpha,power.
    # Use z-approximation: MDE = (z_{1-alpha/2} + z_{1-power}) * s_pool * sqrt(2/n)
    from math import sqrt
    z_a = 1.96 if abs(alpha - 0.05) < 1e-9 else 1.645
    z_b = 1.2816 if abs(power - 0.9) < 1e-9 else 0.8416
    return (z_a + z_b) * s_pool * math.sqrt(2.0 / n_per_group)


def bh_correct(pvals, q=0.05):
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    last = 0.0
    for rank, i in enumerate(idx[::-1], start=1):
        k = m - rank + 1
        adj_k = pvals[i] * m / k
        last = min(last, adj_k) if last > 0 else adj_k
        adj[i] = min(1.0, last)
    return adj


def main():
    with open(SRC) as fh:
        d = json.load(fh)
    rows = d["rows"]
    cells = {}
    for r in rows:
        key = (r["device"], r["control"])
        cells.setdefault(key, []).append(r["nrmse"])

    # Per-device delta values: nrmse_HW - nrmse_SHUFFLE per seed
    seed_keyed = {}
    for r in rows:
        seed_keyed.setdefault((r["device"], r["seed"]), {})[r["control"]] = r["nrmse"]
    delta_hw_per_seed = {dev: [] for dev in {k[0] for k in seed_keyed}}
    delta_sh_per_seed = {dev: [] for dev in {k[0] for k in seed_keyed}}
    delta_iid_per_seed = {dev: [] for dev in {k[0] for k in seed_keyed}}
    for (dev, sd), controls in seed_keyed.items():
        if "HW" in controls and "SHUFFLE" in controls:
            delta_hw_per_seed[dev].append(controls["HW"] - controls["SHUFFLE"])
        if "SHUFFLE" in controls and "SW_iid" in controls:
            delta_sh_per_seed[dev].append(controls["SW_iid"] - controls["SHUFFLE"])
        if "SW_iid" in controls and "HW" in controls:
            delta_iid_per_seed[dev].append(controls["HW"] - controls["SW_iid"])

    # Headline TOST: pool HW vs SHUFFLE nrmse across devices
    hw_all = cells.get(("ikaros", "HW"), []) + cells.get(("daedalus", "HW"), [])
    sh_all = cells.get(("ikaros", "SHUFFLE"), []) + cells.get(("daedalus", "SHUFFLE"), [])
    headline = welch_tost(hw_all, sh_all, lo=-EQ_BOUND, hi=EQ_BOUND, alpha=ALPHA)
    headline["d"] = cohen_d(hw_all, sh_all)
    headline["n_hw"] = len(hw_all)
    headline["n_shuffle"] = len(sh_all)
    headline["mean_hw"] = mean(hw_all)
    headline["mean_shuffle"] = mean(sh_all)

    # Per-device TOST
    per_dev = {}
    for dev in ("ikaros", "daedalus"):
        hw = cells.get((dev, "HW"), [])
        sh = cells.get((dev, "SHUFFLE"), [])
        t = welch_tost(hw, sh, lo=-EQ_BOUND, hi=EQ_BOUND, alpha=ALPHA)
        t["d"] = cohen_d(hw, sh)
        t["n_hw"] = len(hw)
        t["n_shuffle"] = len(sh)
        per_dev[dev] = t

    # MDE at achieved n
    pool_sd = math.sqrt(
        (
            (len(hw_all) - 1) * (stdev(hw_all) ** 2 if len(hw_all) > 1 else 0)
            + (len(sh_all) - 1) * (stdev(sh_all) ** 2 if len(sh_all) > 1 else 0)
        )
        / max(1, len(hw_all) + len(sh_all) - 2)
    )
    n_per = min(len(hw_all), len(sh_all))
    mde = mde_two_sample(pool_sd, n_per, alpha=ALPHA, power=TARGET_POWER)

    # BH across all reported p-values
    pvals = [headline["p_lower"], headline["p_upper"]]
    labels = ["headline_p_lower", "headline_p_upper"]
    for dev in ("ikaros", "daedalus"):
        pvals.extend([per_dev[dev]["p_lower"], per_dev[dev]["p_upper"]])
        labels.extend([f"{dev}_p_lower", f"{dev}_p_upper"])
    adj = bh_correct(pvals, q=0.05)

    # Decision
    if mde > EQ_BOUND:
        decision = "UNDERPOWERED"
        decision_reason = f"MDE={mde:.4g} > eq_bound={EQ_BOUND} — cannot claim equivalence at preregistered bound"
    elif headline["equivalent"] and adj[0] < 0.05 and adj[1] < 0.05:
        decision = "NULL_PASS"
        decision_reason = "TOST rejects H4_alt at α=0.05 after BH; HW equivalent to SHUFFLE within ±0.5%"
    elif abs(headline["diff"]) > EQ_BOUND:
        decision = "ALT_PASS"
        decision_reason = "|Δ| > eq_bound — HW carries detectable identity signal vs SHUFFLE"
    else:
        decision = "INCONCLUSIVE"
        decision_reason = "|Δ| < eq_bound but TOST did not reject after BH — neither equivalence nor difference established"

    out = {
        "preregistration": "research_plan/H4_PREREG_2026-06-09.md",
        "eq_bound": EQ_BOUND,
        "alpha": ALPHA,
        "target_power": TARGET_POWER,
        "headline": headline,
        "per_device": per_dev,
        "delta_hw_per_seed": delta_hw_per_seed,
        "delta_shuffle_per_seed": delta_sh_per_seed,
        "delta_iid_per_seed": delta_iid_per_seed,
        "pool_sd": pool_sd,
        "n_per_group": n_per,
        "mde_at_n": mde,
        "bh_adjusted": dict(zip(labels, adj)),
        "decision": decision,
        "decision_reason": decision_reason,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2, default=float)

    # Markdown render
    md = []
    md.append("# H4 TOST equivalence — Phase-2 re-analysis\n")
    md.append(f"Pre-registration: `research_plan/H4_PREREG_2026-06-09.md`")
    md.append(f"Source: `{SRC.relative_to(ROOT)}`")
    md.append(f"Equivalence bound: ±{EQ_BOUND} nrmse, α={ALPHA}, target 1−β={TARGET_POWER}\n")
    md.append(f"## Decision: **{decision}**")
    md.append(f"{decision_reason}\n")
    md.append("## Headline (pooled HW vs SHUFFLE across devices)")
    md.append(f"- n_HW={headline['n_hw']}, n_SHUFFLE={headline['n_shuffle']}")
    md.append(f"- mean_HW={headline['mean_hw']:.5f}, mean_SHUFFLE={headline['mean_shuffle']:.5f}")
    md.append(f"- Δ = {headline['diff']:+.5f}, SE = {headline['se']:.5f}, df = {headline['df']:.2f}")
    md.append(f"- TOST: t_lower={headline['t_lower']:+.3f} (p={headline['p_lower']:.4f}), "
              f"t_upper={headline['t_upper']:+.3f} (p={headline['p_upper']:.4f})")
    md.append(f"- Equivalent at α={ALPHA}: **{headline['equivalent']}**")
    md.append(f"- Cohen's d = {headline['d']:+.3f}")
    md.append(f"- MDE at n={n_per}/group, α={ALPHA}, power={TARGET_POWER}: **{mde:.5f}** "
              f"({'≤ bound (powered)' if mde <= EQ_BOUND else '> bound (UNDERPOWERED)'})\n")
    md.append("## Per-device TOST")
    md.append("| device | Δ | SE | p_lower | p_upper | equivalent | d | n_HW | n_SHUFFLE |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for dev, t in per_dev.items():
        md.append(f"| {dev} | {t['diff']:+.5f} | {t['se']:.5f} | "
                  f"{t['p_lower']:.4f} | {t['p_upper']:.4f} | {t['equivalent']} | "
                  f"{t['d']:+.3f} | {t['n_hw']} | {t['n_shuffle']} |")
    md.append("\n## BH-adjusted p-values (q=0.05)")
    md.append("| label | p_raw | p_BH |")
    md.append("|---|---|---|")
    for lab, pr, pa in zip(labels, pvals, adj):
        md.append(f"| {lab} | {pr:.4f} | {pa:.4f} |")
    md.append("\n## Per-seed deltas")
    for dev in ("ikaros", "daedalus"):
        hws = delta_hw_per_seed.get(dev, [])
        shs = delta_sh_per_seed.get(dev, [])
        if hws:
            md.append(f"- {dev}: Δ_HW seed-mean={mean(hws):+.5f}, sd={stdev(hws) if len(hws)>1 else 0:.5f}, n={len(hws)}")
            md.append(f"- {dev}: Δ_SW_iid-SHUFFLE seed-mean={mean(shs):+.5f}, sd={stdev(shs) if len(shs)>1 else 0:.5f}, n={len(shs)}")
    md.append("\n## Notes on Bayes factor")
    md.append("- Pre-registered BF_10 comparison vs chassis-only confound model requires a regression fit "
              "(brms / lmer); deferred to a follow-up cell. This script reports frequentist TOST + MDE only. "
              "Decision rule does not depend on BF in absence of MDE failure.")
    OUT_MD.write_text("\n".join(md))
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_JSON}")
    print(f"decision: {decision}")
    print(f"reason: {decision_reason}")


if __name__ == "__main__":
    main()
