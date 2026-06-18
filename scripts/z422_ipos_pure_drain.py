#!/usr/bin/env python3
"""z422 / S15-A: Test Mario's Ipos formula as a PURE behavioral current source
across the drain, bypassing all topology (no M1/M2/Q1/diodes).

Goal: falsification test — if pure Ipos cannot reproduce Sebas's measured I_D
at ANY VG choice, the formula is fundamentally limited.

Formula (per slide 12.26):
    I_pos = I_exp + I_pow
    I_exp = a * exp(b * (V_D + c))
    I_pow = d * (V_D + f)^e   if V_D + f > 0, else 0
    where (a, b, d, e, f) = PWL(V_G), c = -2.4 V

Hypotheses:
    A: PWL is parametrized by V_G2 (the swept gate)
    B: PWL is parametrized by V_G1

Deck contains only: V_D source + B-source as drain current sink.

Pre-registered gates:
    INFRA       : 33 biases × 2 hyp run < 10 min, 0 convergence failures
    DISCOVERY   : cell-wide < 1.5 dec OR >= 0.5 dec improvement over z421=4.699
    AMBITIOUS   : cell-wide < 0.5 dec — Mario's formula alone is sufficient
"""
import os, re, csv, json, time, subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
PWLDIR = REPO / "nsram" / "Zoom" / "ipos_pwl_digitized"
DATA = REPO / "data" / "sebas_2026_04_22"
OUT = REPO / "results" / "z422_ipos_pure_drain"
DECKS = OUT / "decks"
OUT.mkdir(parents=True, exist_ok=True)
DECKS.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT / "run.log"
LOG_FH = open(LOG_PATH, "w")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_FH.write(line + "\n"); LOG_FH.flush()


# -------------------------- PWL loader (same as z421) --------------------------
def _load_pwl(p: Path):
    rows = []
    with open(p) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#") or ln.startswith("VG"):
                continue
            x, y = ln.split(",")
            rows.append((float(x), float(y)))
    arr = np.array(rows)
    idx = np.argsort(arr[:, 0])
    return arr[idx, 0], arr[idx, 1]

PWL_FILES = {
    "a": "curve1_red_a_CALIBRATED.csv",
    "b": "curve3_red_b_middle_subplot_CALIBRATED.csv",
    "d": "curve2_blue_d_left_subplot.csv",
    "e": "curve4_blue_e_middle_subplot.csv",
    "f": "curve5_blue_f_right_subplot.csv",
}
PWL = {nm: _load_pwl(PWLDIR / fn) for nm, fn in PWL_FILES.items()}
C_CONST = -2.4

def pwl_at(vg):
    return {nm: float(np.interp(vg, xs, ys)) for nm, (xs, ys) in PWL.items()}


# -------------------------- Python reference Ipos (sanity check) ---------------
def ipos_python(vd_arr, P):
    """Evaluate Ipos directly in Python for cross-check."""
    a, b, d, e, f = P["a"], P["b"], P["d"], P["e"], P["f"]
    iexp = a * np.exp(b * (vd_arr + C_CONST))
    arg = vd_arr + f
    ipow = np.where(arg > 0, d * np.power(np.maximum(arg, 1e-30), e), 0.0)
    return iexp + ipow


# -------------------------- Measured-data loader --------------------------
BIAS_DIR = {
    0.2: DATA / "2vHCa-2 I-Vs@VG2 VG1=0.2 vnwell=2",
    0.4: DATA / "2vHCa-2 I-Vs@VG2 VG1=0.4 vnwell=2",
    0.6: DATA / "2vHCa-2 I-Vs@VG2 VG1=0.6 vnwell=2",
}
VG2_RE = re.compile(r"VG2=(-?\d+\.?\d*)")

def list_measured_biases():
    out = []
    for vg1, d in BIAS_DIR.items():
        for f in sorted(d.glob("*.csv")):
            m = VG2_RE.search(f.name)
            if m:
                out.append((vg1, float(m.group(1)), f))
    return out

def load_iv(f: Path):
    vd, idd = [], []
    with open(f) as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            try:
                v = float(row["vdata"]); i = float(row["idata"])
            except Exception:
                continue
            vd.append(v); idd.append(i)
    return np.array(vd), np.array(idd)


# -------------------------- Minimal pure-Ipos deck --------------------------
DECK_TEMPLATE = """.title z422 PURE Ipos drain current ({HYP}: VG1={VG1:.3f}, VG2={VG2:.3f})

* No transistors, no BJT, no PTM card. Pure behavioral current source.
Vd  D  0  DC 0

.param a_val={A_VAL:.6e}
.param b_val={B_VAL:.6f}
.param d_val={D_VAL:.6e}
.param e_val={E_VAL:.6f}
.param f_val={F_VAL:.6f}
.param c_const={C_CONST:.6f}

* B-source forces I from D to 0 = drain current sink.
* So I(Vd) = -B_drain current (passive convention).
B_drain D 0 I = a_val*exp(b_val*(V(D)+c_const)) + (((V(D)+f_val) > 0) ? (d_val*pow(V(D)+f_val, e_val)) : 0)

.options gmin=1e-15 abstol=1e-15 reltol=1e-3 itl1=500 itl2=200

.control
dc Vd 0 4 0.025
wrdata {OUT_DAT} -i(vd)
quit
.endc

.end
"""

def build_deck(vg1, vg2, hypothesis):
    vg_for_pwl = vg2 if hypothesis == "A" else vg1
    P = pwl_at(vg_for_pwl)
    tag = f"VG1_{vg1:.2f}_VG2_{vg2:+.3f}_hyp{hypothesis}"
    out_dat = (DECKS / f"out_{tag}.txt").as_posix()
    deck = DECK_TEMPLATE.format(
        HYP=f"hyp{hypothesis}",
        VG1=vg1, VG2=vg2,
        A_VAL=P["a"], B_VAL=P["b"], D_VAL=P["d"], E_VAL=P["e"], F_VAL=P["f"],
        C_CONST=C_CONST, OUT_DAT=out_dat,
    )
    deck_path = DECKS / f"deck_{tag}.cir"
    deck_path.write_text(deck)
    return deck_path, Path(out_dat), P


def run_ngspice(deck_path: Path):
    proc = subprocess.run(["ngspice", "-b", str(deck_path)],
                          capture_output=True, text=True, timeout=60)
    return proc.returncode, proc.stdout + proc.stderr


def parse_dc_out(dat_path: Path):
    if not dat_path.exists():
        return None
    arr = np.loadtxt(dat_path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return {"vd": arr[:, 0], "id": arr[:, 1]}


def rmse_dec(meas_vd, meas_id, sim_vd, sim_id, floor=1e-13):
    if sim_vd is None or len(sim_vd) < 2:
        return float("nan")
    # Take absolute values (sign convention may differ); we compare magnitudes
    sim_on = np.interp(meas_vd, sim_vd, np.abs(sim_id))
    a = np.log10(np.maximum(np.abs(meas_id), floor))
    b = np.log10(np.maximum(sim_on, floor))
    return float(np.sqrt(np.mean((a - b) ** 2)))


# -------------------------- Main --------------------------
def main():
    t0 = time.time()
    biases = list_measured_biases()
    log(f"Found {len(biases)} measured bias files.")
    for nm in ("a","b","d","e","f"):
        xs, ys = PWL[nm]
        log(f"  PWL[{nm}]: VG range [{xs.min():.3f}, {xs.max():.3f}], y range [{ys.min():.3e}, {ys.max():.3e}]")
    log("PRE-REG gates: INFRA=33×2<10min/0fail, DISCOVERY=<1.5dec OR z421-0.5dec, AMBITIOUS=<0.5dec")
    log(f"z421 baseline cell-wide RMSE = 4.699 dec (target to beat by >=0.5)")

    all_results = {}
    for HYP in ("A", "B"):
        log(f"\n=== Hypothesis {HYP}: PWL evaluated at V_G{'2' if HYP=='A' else '1'} ===")
        results = []
        conv_fail = 0
        for (vg1, vg2, fpath) in biases:
            meas_vd, meas_id = load_iv(fpath)
            deck_path, dat_path, P = build_deck(vg1, vg2, HYP)
            rc, stdout = run_ngspice(deck_path)
            parsed = parse_dc_out(dat_path)
            ok = (rc == 0) and parsed is not None and len(parsed["vd"]) > 5
            if not ok:
                conv_fail += 1
                log(f"  CONV FAIL VG1={vg1} VG2={vg2:+.3f} rc={rc}")
                (DECKS / f"log_hyp{HYP}_VG1_{vg1:.2f}_VG2_{vg2:+.3f}.txt").write_text(stdout)
                results.append({"vg1": vg1, "vg2": vg2, "file": fpath.name,
                                "ok": False, "pwl": P,
                                "meas_vd": meas_vd.tolist(),
                                "meas_id": meas_id.tolist()})
                continue
            rmse = rmse_dec(meas_vd, meas_id, parsed["vd"], parsed["id"])
            # Python cross-check at meas_vd: should match ngspice within float epsilon
            py_id = ipos_python(meas_vd, P)
            py_rmse = float(np.sqrt(np.mean(
                (np.log10(np.maximum(np.abs(meas_id), 1e-13))
                 - np.log10(np.maximum(py_id, 1e-13)))**2)))
            log(f"  VG1={vg1} VG2={vg2:+.3f}: ngsp RMSE={rmse:.3f} dec  py={py_rmse:.3f} dec")
            results.append({
                "vg1": vg1, "vg2": vg2, "file": fpath.name, "ok": True,
                "meas_vd": meas_vd.tolist(), "meas_id": meas_id.tolist(),
                "sim_vd": parsed["vd"].tolist(), "sim_id": parsed["id"].tolist(),
                "rmse_dec": rmse, "py_rmse_dec": py_rmse, "pwl": P,
            })

        def avg_rmse(vg1_target):
            xs = [r["rmse_dec"] for r in results
                  if r["vg1"] == vg1_target and r.get("ok")
                  and not np.isnan(r.get("rmse_dec", np.nan))]
            return float(np.mean(xs)) if xs else float("nan")

        branch = {
            "VG1_0.2": avg_rmse(0.2),
            "VG1_0.4": avg_rmse(0.4),
            "VG1_0.6": avg_rmse(0.6),
        }
        cell_vals = [v for v in branch.values() if not np.isnan(v)]
        cell = float(np.mean(cell_vals)) if cell_vals else float("nan")
        all_results[HYP] = {
            "results": results,
            "branch": branch, "cell": cell, "conv_fail": conv_fail,
        }
        log(f"  Hyp {HYP} branch RMSE: {branch}  cell={cell:.3f}")

    # ---- gates ----
    wall = time.time() - t0
    total_conv_fail = sum(v["conv_fail"] for v in all_results.values())
    infra = (total_conv_fail == 0 and wall < 600)
    best_hyp = min(all_results.keys(), key=lambda h: all_results[h]["cell"]
                   if not np.isnan(all_results[h]["cell"]) else 1e9)
    best_cell = all_results[best_hyp]["cell"]
    Z421_BASELINE = 4.699
    discovery = (not np.isnan(best_cell)) and (best_cell < 1.5
                                               or best_cell <= Z421_BASELINE - 0.5)
    ambitious = (not np.isnan(best_cell)) and (best_cell < 0.5)

    summary = {
        "wall_sec": round(wall, 1),
        "n_biases": len(biases),
        "z421_baseline_cell_dec": Z421_BASELINE,
        "hyp_A": {"branch": all_results["A"]["branch"],
                  "cell": all_results["A"]["cell"],
                  "conv_fail": all_results["A"]["conv_fail"]},
        "hyp_B": {"branch": all_results["B"]["branch"],
                  "cell": all_results["B"]["cell"],
                  "conv_fail": all_results["B"]["conv_fail"]},
        "best_hypothesis": best_hyp,
        "best_cell_dec": best_cell,
        "improvement_over_z421_dec": Z421_BASELINE - best_cell,
        "gates": {
            "INFRA": bool(infra),
            "DISCOVERY": bool(discovery),
            "AMBITIOUS": bool(ambitious),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2,
                                                 default=lambda o: float(o)))

    # variant_sweep
    variant = {h: {"branch": all_results[h]["branch"],
                   "cell": all_results[h]["cell"],
                   "per_bias": [{"vg1": r["vg1"], "vg2": r["vg2"],
                                 "ok": r.get("ok", False),
                                 "rmse_dec": r.get("rmse_dec"),
                                 "py_rmse_dec": r.get("py_rmse_dec"),
                                 "pwl": r.get("pwl")}
                                 for r in all_results[h]["results"]]}
               for h in ("A", "B")}
    (OUT / "variant_sweep.json").write_text(json.dumps(variant, indent=2,
                                                       default=lambda o: float(o)))

    # ---- plots (best hyp) ----
    best_results = all_results[best_hyp]["results"]
    for vg1 in (0.2, 0.4, 0.6):
        fig, ax = plt.subplots(figsize=(9, 6.5))
        rs = sorted([r for r in best_results if r["vg1"] == vg1],
                    key=lambda r: r["vg2"])
        cmap = plt.cm.viridis
        n = len(rs)
        for i, r in enumerate(rs):
            color = cmap(i / max(1, n-1))
            mvd = np.array(r["meas_vd"]); mid = np.abs(np.array(r["meas_id"]))
            ax.semilogy(mvd, np.maximum(mid, 1e-14), "o", ms=3, alpha=0.6,
                        color=color, label=f"meas VG2={r['vg2']:+.2f}")
            if r.get("ok"):
                svd = np.array(r["sim_vd"]); sid = np.abs(np.array(r["sim_id"]))
                ax.semilogy(svd, np.maximum(sid, 1e-14), "-", color=color,
                            alpha=0.9, lw=1.2)
        ax.set_xlabel("Vd (V)")
        ax.set_ylabel("|Id| (A)")
        rmse_branch = all_results[best_hyp]["branch"][f"VG1_{vg1}"]
        ax.set_title(f"z422 PURE Ipos drain (hyp {best_hyp}): VG1={vg1}V  "
                     f"(dots=measured, lines=ngspice Ipos; RMSE={rmse_branch:.2f} dec)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=6, ncol=2)
        ax.set_ylim(1e-14, 1e-2)
        fig.tight_layout()
        fig.savefig(OUT / f"overlay_VG1_{str(vg1).replace('.', 'p')}.png", dpi=130)
        plt.close(fig)

    # also overlay for the LOSING hypothesis (for inspection)
    other_hyp = "B" if best_hyp == "A" else "A"
    other_results = all_results[other_hyp]["results"]
    for vg1 in (0.2, 0.4, 0.6):
        fig, ax = plt.subplots(figsize=(9, 6.5))
        rs = sorted([r for r in other_results if r["vg1"] == vg1],
                    key=lambda r: r["vg2"])
        cmap = plt.cm.viridis
        n = len(rs)
        for i, r in enumerate(rs):
            color = cmap(i / max(1, n-1))
            mvd = np.array(r["meas_vd"]); mid = np.abs(np.array(r["meas_id"]))
            ax.semilogy(mvd, np.maximum(mid, 1e-14), "o", ms=3, alpha=0.6, color=color)
            if r.get("ok"):
                svd = np.array(r["sim_vd"]); sid = np.abs(np.array(r["sim_id"]))
                ax.semilogy(svd, np.maximum(sid, 1e-14), "-", color=color, alpha=0.9, lw=1.2)
        rmse_branch = all_results[other_hyp]["branch"][f"VG1_{vg1}"]
        ax.set_xlabel("Vd (V)"); ax.set_ylabel("|Id| (A)")
        ax.set_title(f"z422 PURE Ipos (hyp {other_hyp}): VG1={vg1}V  RMSE={rmse_branch:.2f} dec")
        ax.grid(True, which="both", alpha=0.3); ax.set_ylim(1e-14, 1e-2)
        fig.tight_layout()
        fig.savefig(OUT / f"overlay_hyp{other_hyp}_VG1_{str(vg1).replace('.','p')}.png", dpi=130)
        plt.close(fig)

    # ---- honest analysis ----
    md = []
    md.append("# z422 / S15-A — Pure Ipos as drain current source\n")
    md.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    md.append("## Setup\n")
    md.append("Minimal ngspice deck: V_D source + behavioral B-source as drain current.\n")
    md.append("NO M1, NO M2, NO Q1, NO PTM card. Just `I(V_D) = Ipos(V_D; V_G)`.\n\n")
    md.append("## Cell-wide log-RMSE (decades)\n\n")
    md.append("| Hyp | VG1=0.2 | VG1=0.4 | VG1=0.6 | **cell** | Δ vs z421 |\n")
    md.append("|---|---|---|---|---|---|\n")
    for h in ("A", "B"):
        br = all_results[h]["branch"]; c = all_results[h]["cell"]
        delta = Z421_BASELINE - c if not np.isnan(c) else float('nan')
        md.append(f"| {h} (PWL@V_G{'2' if h=='A' else '1'}) | "
                  f"{br['VG1_0.2']:.3f} | {br['VG1_0.4']:.3f} | {br['VG1_0.6']:.3f} | "
                  f"**{c:.3f}** | {delta:+.3f} |\n")
    md.append(f"\nz421 baseline (full topology): {Z421_BASELINE:.3f} dec.\n\n")
    md.append("## Pre-registered gates\n\n")
    for k, v in summary["gates"].items():
        md.append(f"- {k}: **{'PASS' if v else 'FAIL'}**\n")
    md.append(f"\nBest hypothesis: **{best_hyp}**, cell-wide = **{best_cell:.3f} dec**, "
              f"improvement vs z421 = **{Z421_BASELINE - best_cell:+.3f} dec**.\n\n")
    md.append("## Interpretation\n\n")
    if summary["gates"]["AMBITIOUS"]:
        md.append("Pure Ipos < 0.5 dec: Mario's formula alone is sufficient to reproduce\n"
                  "Sebas's I_D. The topology contributes essentially nothing beyond Ipos in\n"
                  "this regime; z419/z420/z421 failures were artifacts of injection into the\n"
                  "wrong node (body), not formula limits. Next step: rebuild topology so Ipos\n"
                  "directly drives drain current.\n")
    elif summary["gates"]["DISCOVERY"]:
        md.append("Pure Ipos materially beats topology-coupled z421 (>=0.5 dec improvement\n"
                  "OR <1.5 dec absolute). Confirms that the digitized PWL formula carries\n"
                  "real predictive content and that the topology in z421 was masking it.\n"
                  "Residual gap is likely PWL digitization precision and/or VG1-dependence\n"
                  "missing from the single-VG PWL.\n")
    else:
        md.append("Pure Ipos does NOT close the gap (neither hypothesis < 1.5 dec, nor 0.5 dec\n"
                  "better than z421). The Ipos formula with the currently digitized PWL is\n"
                  "fundamentally insufficient to reproduce Sebas's measurement. Possible reasons:\n"
                  "1. PWL digitization error (especially e, f at low VG → power-law exponent).\n"
                  "2. Formula is missing a V_G1-dependent term (only one gate enters).\n"
                  "3. Sebas's I_D is not Ipos but a different sub-mechanism in the regime.\n"
                  "4. Sign / units mismatch (Ipos as defined is a body injection, NOT drain I).\n"
                  "Need new framework, not just topology fix.\n")
    md.append("\n## NO-CHEAT verification checklist\n")
    md.append("- Python reference Ipos (ipos_python) matches ngspice within float epsilon\n"
              "  (compare `rmse_dec` vs `py_rmse_dec` in variant_sweep.json: they should agree\n"
              "  to 3+ decimals; if they don't, the ngspice deck is doing something extra).\n")
    md.append("- Overlay plots (overlay_VG1_*.png) must be inspected visually: does the\n"
              "  pure-Ipos curve at least track the SHAPE of Sebas's measurement, even if\n"
              "  the magnitude is off?\n")
    (OUT / "honest_analysis.md").write_text("".join(md))

    log(f"\nDONE in {wall:.1f}s.  Best hyp={best_hyp}  cell={best_cell:.3f} dec  "
        f"(z421 baseline {Z421_BASELINE:.3f}, Δ={Z421_BASELINE-best_cell:+.3f})")
    log(f"GATES: {summary['gates']}")
    return summary


if __name__ == "__main__":
    s = main()
    print(json.dumps(s["gates"], indent=2))
    print(f"best={s['best_hypothesis']}  cell={s['best_cell_dec']:.3f} dec  "
          f"Δz421={s['improvement_over_z421_dec']:+.3f}")
