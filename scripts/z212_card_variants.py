#!/usr/bin/env python3
"""z212: Find duplicate parameters in Sebas's BSIM4 card and test 3 interpretations.

(A) Scan NMOS + PMOS .model blocks for duplicates.
(B) Build 3 variant cards: lastwins (verbatim), firstwins (drop later dupes),
    intentwins (manual: prefer non-zero / non-PTM-default).
(C) Load each via BSIM4Model.from_spice; print effective values.
(D) Run evaluate_full once per variant, no fitting.
(E) Pick best baseline.
"""
import json, sys, time, re, math, copy
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

from z88_bsim4_port_fit_p7v10_skipnonconv import (
    OUT, load_curves, make_cfg_and_sd, evaluate_full,
    PARAM_SPEC, theta_to_value, fitted_dict, make_thetas,
)
from nsram.bsim4_port.model_card import BSIM4Model

DATA_DIR = ROOT / "data/sebas_2026_04_22"
SPICE_FILE = DATA_DIR / "PTM130bulkNSRAM.txt"
OUT_DIR = ROOT / "results/z212_card_variants"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VAR_DIR = Path("/tmp/card_variants")
VAR_DIR.mkdir(parents=True, exist_ok=True)


def value_to_theta(name: str, value: float) -> float:
    spec = PARAM_SPEC[name]
    lo, hi = spec["bounds"]
    if spec["kind"] == "linb":
        s = (float(value) - lo) / (hi - lo)
    else:
        s = math.log(max(float(value), 1e-30) / lo) / math.log(hi / lo)
    s = min(max(s, 1e-9), 1.0 - 1e-9)
    return math.log(s / (1.0 - s))


# ---- (A) Scan duplicates ----
def find_model_block(text: str, model_name: str):
    """Return (start_line, end_line, list_of_(line_no, text)) for given model."""
    lines = text.splitlines()
    starts = []
    for i, ln in enumerate(lines):
        if re.match(rf"\s*\.model\s+{model_name}\s+", ln, re.IGNORECASE):
            starts.append(i)
    if not starts:
        return None
    s = starts[0]
    # End: next .model or EOF
    e = len(lines)
    for i, ln in enumerate(lines[s + 1:], start=s + 1):
        if re.match(r"\s*\.model\s+", ln, re.IGNORECASE):
            e = i; break
    return s, e, [(i + 1, lines[i]) for i in range(s, e)]


def scan_duplicates(block_lines):
    """Given list of (line_no, text), find duplicate param=value declarations.
    Returns list of dicts: {param, occurrences:[(line_no, value_str)]}."""
    # name=value matcher (SPICE numeric or identifier)
    VAL = r"(?:[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?[a-zA-Z]*|[A-Za-z_]\w*)"
    pat = re.compile(rf"(\w+)\s*=\s*({VAL})")
    occ: dict[str, list] = {}
    for line_no, text in block_lines:
        # Skip the .model header line itself
        cleaned = re.sub(r"\.model\s+\S+\s+\S+", " ", text, flags=re.IGNORECASE)
        cleaned = cleaned.split("$")[0]
        if cleaned.lstrip().startswith("*"):
            continue
        for m in pat.finditer(cleaned):
            n = m.group(1).lower()
            v = m.group(2)
            if n in ("level", "version", "type"):
                continue
            occ.setdefault(n, []).append((line_no, v, m.group(1)))
    dupes = {n: lst for n, lst in occ.items() if len(lst) > 1}
    return dupes


# ---- (B) Build variant card files ----
def build_lastwins(text: str, path: Path):
    path.write_text(text)


def build_firstwins(text: str, dupes_nmos: dict, nmos_range, path: Path):
    """Remove second+ occurrences (within NMOS block) so first-declared wins.
    For each duplicate param, find every line in NMOS block containing param=...
    after the first, and surgically delete that 'param=value' token."""
    lines = text.splitlines()
    n_start, n_end = nmos_range
    VAL = r"(?:[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?[a-zA-Z]*|[A-Za-z_]\w*)"
    # For each dup param, compute lines of 2nd, 3rd, ... occurrences
    to_kill = []  # (line_idx, param_orig_case)
    for pname, occs in dupes_nmos.items():
        for (line_no, _v, orig_case) in occs[1:]:
            to_kill.append((line_no - 1, orig_case))
    for line_idx, orig in to_kill:
        line = lines[line_idx]
        # Strip "<orig>=<val>" allowing case-insensitive match on name
        new = re.sub(rf"(?i)\b{re.escape(orig)}\s*=\s*{VAL}\s*", "", line, count=1)
        lines[line_idx] = new
    path.write_text("\n".join(lines) + "\n")


def build_intentwins(text: str, dupes_nmos: dict, nmos_range, intent_choices: dict,
                      path: Path):
    """For each duplicate, keep ONLY the chosen line's value; remove the others.
    intent_choices: {pname: chosen_line_no}."""
    lines = text.splitlines()
    VAL = r"(?:[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?[a-zA-Z]*|[A-Za-z_]\w*)"
    to_kill = []
    for pname, occs in dupes_nmos.items():
        chosen_line = intent_choices[pname]
        for (line_no, _v, orig_case) in occs:
            if line_no != chosen_line:
                to_kill.append((line_no - 1, orig_case))
    for line_idx, orig in to_kill:
        line = lines[line_idx]
        new = re.sub(rf"(?i)\b{re.escape(orig)}\s*=\s*{VAL}\s*", "", line, count=1)
        lines[line_idx] = new
    path.write_text("\n".join(lines) + "\n")


# ---- main ----
def main():
    t0 = time.time()
    raw = SPICE_FILE.read_text()

    # ---- (A) Scan ----
    n_s, n_e, nmos_block = find_model_block(raw, "NMOS")
    p_s, p_e, pmos_block = find_model_block(raw, "PMOS")
    nmos_range = (n_s, n_e)

    dup_nmos = scan_duplicates(nmos_block)
    dup_pmos = scan_duplicates(pmos_block)

    print("=== NMOS DUPLICATES ===")
    nmos_table = []
    for pname, occs in sorted(dup_nmos.items()):
        # Last wins per ngspice
        last = occs[-1]
        for i, (line_no, v, _) in enumerate(occs):
            wins = " <-- LAST-WINS" if i == len(occs) - 1 else ""
            print(f"  {pname:12s} line {line_no:3d}: {v}{wins}")
            nmos_table.append((pname, line_no, v, i == len(occs) - 1))
        print()

    print("=== PMOS DUPLICATES ===")
    pmos_table = []
    for pname, occs in sorted(dup_pmos.items()):
        for i, (line_no, v, _) in enumerate(occs):
            wins = " <-- LAST-WINS" if i == len(occs) - 1 else ""
            print(f"  {pname:12s} line {line_no:3d}: {v}{wins}")
            pmos_table.append((pname, line_no, v, i == len(occs) - 1))
        print()

    # ---- intent choice (for each NMOS duplicate, decide intent) ----
    # heuristic: prefer non-zero, prefer non-PTM-default (defaults: alpha0=0, beta0=30,
    # vsat=8e4, k1=0.5, k2=0, voff=-0.08, u0=0.067, ute=-1.5, kt1=-0.11, kt1l=0,
    # dsub=DROUT, drout=0.56, pdiblc1=0.39, pdiblc2=0.0086, pdiblcb=0,
    # pscbe1=4.24e8, pscbe2=1e-5)
    # For each NMOS dup, build intent choice: pick line whose value is non-zero & non-default
    PTM_DEFAULTS = {
        "alpha0": 0.0, "beta0": 30.0, "vsat": 8e4, "k1": 0.5, "k2": 0.0,
        "voff": -0.08, "u0": 0.067, "ute": -1.5, "kt1": -0.11, "kt1l": 0.0,
        "dsub": 0.56, "drout": 0.56, "pdiblc1": 0.39, "pdiblc2": 0.0086,
        "pdiblcb": 0.0, "pscbe1": 4.24e8, "pscbe2": 1e-5,
    }
    intent_choices = {}
    intent_rationale = {}
    for pname, occs in dup_nmos.items():
        # Parse all candidate values
        vals = []
        for (line_no, vstr, _) in occs:
            try:
                fv = float(vstr)
            except ValueError:
                fv = None  # might be .param ref
            vals.append((line_no, vstr, fv))
        default = PTM_DEFAULTS.get(pname)
        # Score: 0 if zero, 1 if non-zero & equals default, 2 if non-zero & non-default
        def score(t):
            line_no, vstr, fv = t
            if fv is None:
                return 1  # symbol ref — assume meaningful
            if fv == 0.0:
                return 0
            if default is not None and abs(fv - default) < 1e-12:
                return 1
            return 2
        ranked = sorted(vals, key=score, reverse=True)
        chosen = ranked[0]
        intent_choices[pname] = chosen[0]
        intent_rationale[pname] = (
            f"chose line {chosen[0]} val={chosen[1]} (score={score(chosen)}) "
            f"over " + ", ".join(f"L{t[0]}={t[1]}" for t in ranked[1:])
        )

    print("=== INTENT CHOICES (NMOS) ===")
    for pname, line in intent_choices.items():
        print(f"  {pname:12s} -> line {line}: {intent_rationale[pname]}")

    # ---- (B) Build variant files ----
    p_last = VAR_DIR / "card_lastwins.txt"
    p_first = VAR_DIR / "card_firstwins.txt"
    p_intent = VAR_DIR / "card_intentwins.txt"

    build_lastwins(raw, p_last)
    build_firstwins(raw, dup_nmos, nmos_range, p_first)
    build_intentwins(raw, dup_nmos, nmos_range, intent_choices, p_intent)

    print(f"\nWrote {p_last}, {p_first}, {p_intent}")

    # ---- (C) Load + print effective values ----
    REPORT = ["alpha0", "beta0", "k1", "k2", "voff", "vsat", "u0", "ute",
              "kt1", "kt1l", "dsub", "drout", "pdiblc1", "pdiblc2", "pdiblcb",
              "pscbe1", "pscbe2"]
    variants = {
        "lastwins": p_last.read_text(),
        "firstwins": p_first.read_text(),
        "intentwins": p_intent.read_text(),
    }
    eff_values = {}
    models = {}
    for vname, vtxt in variants.items():
        m = BSIM4Model.from_spice(vtxt, model_type="nmos")
        models[vname] = m
        eff_values[vname] = {p: m.get(p) for p in REPORT}

    print("\n=== EFFECTIVE VALUES ===")
    print(f"{'param':10s}  {'lastwins':>14s}  {'firstwins':>14s}  {'intentwins':>14s}")
    for p in REPORT:
        l = eff_values["lastwins"][p]
        f = eff_values["firstwins"][p]
        i = eff_values["intentwins"][p]
        print(f"{p:10s}  {l:14.6e}  {f:14.6e}  {i:14.6e}")

    # ---- (D) Evaluate each variant ----
    print("\n=== EVALUATING (Stage-4 fitted params, no fitting) ===")
    s4 = json.load(open(OUT / "stage4_summary.json"))
    P = s4["params"]

    eval_results = {}
    pred_by_variant = {}
    for vname, m in models.items():
        thetas = make_thetas(seed=0)
        for name, val in P.items():
            if name in thetas:
                thetas[name].data = torch.tensor(value_to_theta(name, val),
                                                  dtype=torch.float64)
        cfg = make_cfg_and_sd(m, gates={"use_iii": True, "use_gidl": True, "use_bjt": True})
        curves = load_curves()
        median_rmse, preds = evaluate_full(thetas, m, cfg, curves)
        rmses = [pp["log_rmse"] for pp in preds if math.isfinite(pp["log_rmse"])]
        n_total = sum(len(np.asarray(pp.get("converged", []))) for pp in preds)
        n_conv = sum(int(np.asarray(pp.get("converged", [])).sum()) for pp in preds)
        eval_results[vname] = {
            "median_log_rmse": median_rmse,
            "mean_log_rmse": float(np.mean(rmses)) if rmses else float("inf"),
            "n_curves_finite": len(rmses),
            "convergence_pct": 100 * n_conv / max(n_total, 1),
            "n_conv": n_conv, "n_total": n_total,
        }
        pred_by_variant[vname] = preds
        print(f"  {vname:11s}: median={median_rmse:.3f}  mean={eval_results[vname]['mean_log_rmse']:.3f}  "
              f"conv={eval_results[vname]['convergence_pct']:.1f}%  ({n_conv}/{n_total})")

    # ---- (E) Identify best ----
    best = min(eval_results, key=lambda v: (eval_results[v]["median_log_rmse"],
                                              -eval_results[v]["convergence_pct"]))
    print(f"\n=== BEST BASELINE: {best} ===")

    # ---- save outputs ----
    # duplicates.md
    md1 = ["# Duplicate parameter scan", "",
           f"Card: `{SPICE_FILE}`", "",
           "## NMOS .model block (lines 14-75)", "",
           "| param | line | value | wins (last-wins)? |",
           "|---|---|---|---|"]
    for pname, line_no, v, wins in nmos_table:
        md1.append(f"| `{pname}` | {line_no} | `{v}` | {'**YES**' if wins else 'no'} |")
    md1 += ["", "## PMOS .model block (for completeness; not used)", "",
            "| param | line | value | wins (last-wins)? |",
            "|---|---|---|---|"]
    for pname, line_no, v, wins in pmos_table:
        md1.append(f"| `{pname}` | {line_no} | `{v}` | {'**YES**' if wins else 'no'} |")
    md1 += ["", "## Intent choices (NMOS)", ""]
    for pname, line in intent_choices.items():
        md1.append(f"- `{pname}` -> line {line}: {intent_rationale[pname]}")
    (OUT_DIR / "duplicates.md").write_text("\n".join(md1) + "\n")

    # variants.md
    md2 = ["# Variant card evaluation", "",
           "## Effective parameter values", "",
           f"| param | lastwins | firstwins | intentwins |",
           "|---|---|---|---|"]
    for p in REPORT:
        l = eff_values["lastwins"][p]; f = eff_values["firstwins"][p]; i = eff_values["intentwins"][p]
        md2.append(f"| `{p}` | {l:.6e} | {f:.6e} | {i:.6e} |")
    md2 += ["", "## Evaluation results (no fitting; Stage-4 thetas)", "",
            "| variant | median log-RMSE | mean log-RMSE | convergence % | n_conv/n_total |",
            "|---|---|---|---|---|"]
    for v in ["lastwins", "firstwins", "intentwins"]:
        r = eval_results[v]
        md2.append(f"| `{v}` | {r['median_log_rmse']:.3f} | {r['mean_log_rmse']:.3f} | "
                    f"{r['convergence_pct']:.1f}% | {r['n_conv']}/{r['n_total']} |")
    md2 += ["", f"**Best baseline: `{best}`**", ""]
    (OUT_DIR / "variants.md").write_text("\n".join(md2) + "\n")

    # summary plot at one VG1 group (pick middle one)
    # Use VG1=0.4 if exists, else first
    by_vg1 = {}
    for v in pred_by_variant:
        for p in pred_by_variant[v]:
            by_vg1.setdefault(p["VG1"], set()).add(v)
    vg1_pick = sorted(by_vg1.keys())[len(by_vg1) // 2]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, vname in zip(axes, ["lastwins", "firstwins", "intentwins"]):
        ps = sorted([p for p in pred_by_variant[vname] if p["VG1"] == vg1_pick],
                    key=lambda c: c["VG2"])
        n = len(ps)
        for i, p in enumerate(ps):
            color = cmap(i / max(n - 1, 1))
            Vd = np.asarray(p["Vd"])
            Id_meas = np.asarray(p["Id_meas"])
            Id_pred = np.asarray(p["Id_pred"])
            conv = np.asarray(p.get("converged", np.ones_like(Vd, dtype=bool)), dtype=bool)
            ax.semilogy(Vd, np.abs(Id_meas), "o", color=color, ms=4, alpha=0.6,
                         label=f"VG2={p['VG2']:+.2f}")
            if conv.any():
                ax.semilogy(Vd[conv], np.abs(Id_pred[conv]), "-",
                             color=color, lw=1.4, alpha=0.95)
        r = eval_results[vname]
        ax.set_title(f"{vname}\nmed log-RMSE={r['median_log_rmse']:.3f}  "
                     f"conv={r['convergence_pct']:.0f}%")
        ax.set_xlabel("Vd [V]"); ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=7, ncol=2)
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle(f"z212 card variant comparison @ VG1 = {vg1_pick} V (Stage-4 thetas, no fit)",
                  fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "variants_compare.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT_DIR}/duplicates.md, variants.md, variants_compare.png  "
          f"(took {time.time()-t0:.0f}s)")

    # ---- Save full json for downstream ----
    summary = {
        "best": best,
        "eval_results": eval_results,
        "effective_values": eff_values,
        "nmos_duplicates": {n: [(ln, v) for (ln, v, _) in occs] for n, occs in dup_nmos.items()},
        "pmos_duplicates": {n: [(ln, v) for (ln, v, _) in occs] for n, occs in dup_pmos.items()},
        "intent_choices": intent_choices,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
