"""z312: aggregate HDC N=8192/16384 sweep + compare to z293/z302."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "results" / "z312_hdc_n16k"


def load_cell(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        return {"_error": str(e)}


def summarize(d):
    if d is None or "_error" in (d or {}):
        return d
    per_seed = d.get("per_seed", [])
    accs = [s.get("test_acc") for s in per_seed if isinstance(s.get("test_acc"), (int, float))]
    return {
        "cell": d.get("cell"),
        "n_seeds": len(accs),
        "mean_acc": d.get("mean_acc"),
        "per_seed_test_acc": accs,
        "min_acc": min(accs) if accs else None,
        "max_acc": max(accs) if accs else None,
    }


def main():
    table = {}
    for N in (8192, 16384):
        for sigma in (0.00, 0.05, 0.10):
            s_tag = f"{sigma:.2f}".replace(".", "p")
            cell_dir = OUT_ROOT / f"N{N}_s{s_tag}"
            summary_path = cell_dir / "summary.json"
            key = f"N{N}_sigma{sigma:.2f}"
            d = load_cell(summary_path)
            table[key] = summarize(d) if d else {"status": "MISSING", "path": str(summary_path)}

    # Pull priors for context.
    priors = {}
    # z293: N-scaling at sigma=0
    for N in (64, 128, 256, 512, 1024):
        p = REPO / "results" / "z293_envelope" / "4B1_Nscaling" / f"N{N}" / "summary.json"
        if p.exists():
            priors[f"z293_N{N}_sigma0.00"] = summarize(load_cell(p))
    # z302: N-scaling at sigma_test=0.05
    for N in (1024, 2048, 4096):
        p = REPO / "results" / "z302_hdc_noise_robust" / "B_nscale" / f"N{N}" / "summary.json"
        if p.exists():
            priors[f"z302_N{N}_sigma_te0.05"] = summarize(load_cell(p))
    # z293 noise sweep at N=128 (for sigma-curve shape baseline)
    for stag in ("0p00", "0p05", "0p10", "0p20"):
        p = REPO / "results" / "z293_envelope" / "4B2_noise" / f"sigma{stag}" / "summary.json"
        if p.exists():
            priors[f"z293_N128_sigma{stag}"] = summarize(load_cell(p))

    # Gate logic.
    gates = {"AMBITIOUS": False, "PASS": False, "details": []}
    # AMBITIOUS: any cell >82% at sigma=0 OR >80% at sigma=0.10
    for key, s in table.items():
        if not s or "mean_acc" not in s or s["mean_acc"] is None:
            continue
        if "sigma0.00" in key and s["mean_acc"] > 0.82:
            gates["AMBITIOUS"] = True
            gates["details"].append(f"AMBITIOUS hit: {key}={s['mean_acc']:.4f}>0.82 (sigma=0)")
        if "sigma0.10" in key and s["mean_acc"] > 0.80:
            gates["AMBITIOUS"] = True
            gates["details"].append(f"AMBITIOUS hit: {key}={s['mean_acc']:.4f}>0.80 (sigma=0.10)")

    # PASS: monotone N-scaling at sigma=0  N=16384 > N=4096
    z302_4096 = None
    # nearest sigma=0 comparator is z293 N=1024 (no z302 sigma=0 at large N) — use z293 N=1024 sigma=0
    z293_1024 = priors.get("z293_N1024_sigma0.00", {}).get("mean_acc")
    n16384_s0 = table.get("N16384_sigma0.00", {}).get("mean_acc")
    n8192_s0 = table.get("N8192_sigma0.00", {}).get("mean_acc")
    n4096_s0 = None  # no z293 cell at N=4096 sigma=0 — note this
    if n16384_s0 is not None and n8192_s0 is not None and z293_1024 is not None:
        mono = (n16384_s0 >= n8192_s0 >= z293_1024)
        gates["PASS"] = mono
        gates["details"].append(
            f"PASS check (sigma=0): N=1024:{z293_1024:.4f} -> N=8192:{n8192_s0:.4f} -> "
            f"N=16384:{n16384_s0:.4f}  monotone={mono}"
        )

    out = {
        "experiment": "z312_hdc_n16k",
        "table": table,
        "priors": priors,
        "gates": gates,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
