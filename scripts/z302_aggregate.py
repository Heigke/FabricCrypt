"""z302: aggregate noise-robustness sweep into summary.json."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results" / "z302_hdc_noise_robust"


def load_all():
    rows = []
    for sj in ROOT.rglob("summary.json"):
        if sj.parent.name == "_smoke" or sj.parent == ROOT:
            continue
        try:
            d = json.loads(sj.read_text())
        except Exception:
            continue
        cell = d.get("cell", {})
        rows.append({
            "path": str(sj.relative_to(REPO)),
            "strategy": d.get("strategy"),
            "N": cell.get("N"),
            "vd_high": cell.get("vd_high"),
            "vd_low": cell.get("vd_low"),
            "sigma_train": cell.get("sigma_train"),
            "sigma_test": cell.get("sigma_test"),
            "mean_acc": d.get("mean_acc"),
            "std_acc": d.get("std_acc"),
            "ci95": d.get("ci95"),
            "mean_energy_J_per_inference": d.get("mean_energy_J_per_inference"),
            "gates": d.get("gates"),
        })
    return rows


def main():
    rows = load_all()
    rows.sort(key=lambda r: (r["strategy"] or "", -(r["mean_acc"] or 0.0)))

    # Group by strategy
    by_strat = {}
    for r in rows:
        by_strat.setdefault(r["strategy"], []).append(r)

    # Pick best config at sigma_test=0.05 over all strategies
    sig05 = [r for r in rows
             if r["sigma_test"] is not None and abs(r["sigma_test"] - 0.05) < 1e-6
             and r["mean_acc"] is not None]
    sig05.sort(key=lambda r: -r["mean_acc"])
    best = sig05[0] if sig05 else None

    # Pick best per strategy at sigma_test=0.05
    best_per_strat = {}
    for s, group in by_strat.items():
        cands = [r for r in group
                 if r["sigma_test"] is not None and abs(r["sigma_test"] - 0.05) < 1e-6
                 and r["mean_acc"] is not None]
        if cands:
            cands.sort(key=lambda r: -r["mean_acc"])
            best_per_strat[s] = cands[0]

    # Headline drop from strategy A: how much does sigma_test=0 acc drop
    # under noise-trained encoder (sigma_train=0.10) vs clean (sigma_train=0.0)?
    a_clean_test0 = None
    a_noisy_test0 = None
    for r in rows:
        if r["strategy"] != "A_noisetrain":
            continue
        if r["sigma_test"] is None or abs(r["sigma_test"]) > 1e-9:
            continue
        st = r["sigma_train"]
        if st is None:
            continue
        if abs(st) < 1e-9:
            a_clean_test0 = r["mean_acc"]
        elif abs(st - 0.10) < 1e-6:
            a_noisy_test0 = r["mean_acc"]
    headline_drop = (None if (a_clean_test0 is None or a_noisy_test0 is None)
                     else a_clean_test0 - a_noisy_test0)

    summary = {
        "experiment": "z302_hdc_noise_robust",
        "n_rows": len(rows),
        "rows": rows,
        "best_at_sigma_test_0p05": best,
        "best_per_strategy_at_sigma_test_0p05": best_per_strat,
        "headline_drop_clean_vs_noisetrained_at_sigma_test_0": {
            "sigma_train_0_test_0_acc":  a_clean_test0,
            "sigma_train_0p10_test_0_acc": a_noisy_test0,
            "drop": headline_drop,
        },
        "gates": {
            "conservative_geq_0p70_at_sigma_test_0p05":
                bool(best and (best["mean_acc"] or 0.0) >= 0.70),
            "ambitious_geq_0p75_at_sigma_test_0p05":
                bool(best and (best["mean_acc"] or 0.0) >= 0.75),
        },
    }

    out = ROOT / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[z302] aggregated {len(rows)} cells -> {out}")
    print(f"  best @ sigma_test=0.05: {best}")
    print(f"  headline drop: {summary['headline_drop_clean_vs_noisetrained_at_sigma_test_0']}")
    print(f"  gates: {summary['gates']}")


if __name__ == "__main__":
    main()
