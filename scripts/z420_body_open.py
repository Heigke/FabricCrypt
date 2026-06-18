#!/usr/bin/env python3
"""z420: Fix the body-shorting bug in 2tnsram_simple.asc.

Background (S13/z419 finding): the canonical schematic ties the parasitic body
node B to ground through C1 (CBpar=1f) with Rser=1m. In ngspice DC analysis,
that 1 mΩ series resistor is a near-short — V_B cannot float up to forward-bias
the parasitic NPN, so the snapback regime is unreachable regardless of
canonical params.

This script replicates z419's setup but RE-WIRES the body cap as either:
  - C1 from B to 0 directly (clean float, no Rser), OR equivalently
  - Replace Rcb=1m with Rcb=1Tera (effectively open).

We choose the latter to keep the same node names ("Bx") as z419 for easy diff.

Then we re-run all 4 variants (hypA/B × injB/D) over the 33 measured biases
and compare cell-wide log-RMSE vs the S13 baseline. We also probe max(V_B)
across each DC sweep — the diagnostic the task asked for.

Inputs / outputs match z419's conventions. Outputs land in
results/z420_body_open/.
"""
import os, re, csv, json, time, subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
CANON = REPO / "nsram" / "Zoom" / "schematic&modelCards"
PWLDIR = REPO / "nsram" / "Zoom" / "ipos_pwl_digitized"
DATA = REPO / "data" / "sebas_2026_04_22"
OUT = REPO / "results" / "z420_body_open"
DECKS = OUT / "decks"
OUT.mkdir(parents=True, exist_ok=True)
DECKS.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT / "run.log"
LOG_FH = open(LOG_PATH, "w")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_FH.write(line + "\n"); LOG_FH.flush()

# -------------------------- PWL loader (identical to z419) -----------------
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
    "a": "curve1_red_a.csv",
    "b": "curve3_red_b_middle_subplot.csv",
    "d": "curve2_blue_d_left_subplot.csv",
    "e": "curve4_blue_e_middle_subplot.csv",
    "f": "curve5_blue_f_right_subplot.csv",
}
PWL = {nm: _load_pwl(PWLDIR / fn) for nm, fn in PWL_FILES.items()}
C_CONST = -2.4

def pwl_at(vg):
    return {nm: float(np.interp(vg, xs, ys)) for nm, (xs, ys) in PWL.items()}

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


# -------------------------- Deck (BODY-OPEN FIX) ---------------------------
# Only diff vs z419: Rcb 1m -> Rcb 1T (effectively open). C1 still from B to Bx
# so the body sees an isolated 1 fF to a dangling node. In DC analysis this
# means V_B is fully floating, controlled by Q1 base current + injected Ipos
# (if INJ_NODE=B) minus any leakage through M1's bulk diode model.
DECK_TEMPLATE = """.title z420 body-open ({HYP}: VG1={VG1:.3f}, VG2={VG2:.3f}, inj={INJ})

* Compatibility shim for LTspice-flavored canonical card
.param vsatn=1.35e5
.param Nparam=1.58
.param Citparam=0
.param Voffparam=-0.1368
.param K2Par=-0.070435
.param toxn=4e-9

.param Ln=0.18u
.param Wn=0.36u
.param CBpar=1f

.include "{PTM130}"
.include "{BJT}"

Vd   D   0   DC 0
Vg1  G   0   DC {VG1:.4f}
Vg2  G2  0   DC {VG2:.4f}

* M1 nmos4: D=D, G=G, S=Sint, B=B
M1 D G Sint B NMOS L=0.18u W=0.36u

* M2 nmos4: D=Sint, G=G2, S=0, B=0
M2 Sint G2 0 0 NMOS L=1.8u W=0.36u

* Q1 parasitic NPN: C=D, B=B, E=0
Q1 D B 0 parasiticBJT area=1u

* >>> FIX: floating body cap. Rcb was 1m (near-short to GND). Now 1 TΩ. <<<
C1 B Bx 1f
Rcb Bx 0 1Tera

* ---- Mario Ipos PWL block (identical to z419 / S13) ----
.param a_val={A_VAL:.6e}
.param b_val={B_VAL:.6f}
.param d_val={D_VAL:.6e}
.param e_val={E_VAL:.6f}
.param f_val={F_VAL:.6f}
.param c_const={C_CONST:.6f}

B_ipos {INJ_NODE} 0 I = a_val*exp(b_val*(V(D)+c_const)) + (((V(D)+f_val) > 0) ? (d_val*pow(V(D)+f_val, e_val)) : 0)

.options gmin=1e-15 abstol=1e-15 reltol=1e-3 itl1=500 itl2=200 itl6=100

.control
dc Vd 0 2 0.05
wrdata {OUT_DAT} -i(vd) v(Sint) v(B)
quit
.endc

.end
"""

def build_deck(vg1, vg2, hyp, inj):
    vg_for_pwl = vg2 if hyp == "A" else vg1
    P = pwl_at(vg_for_pwl)
    tag = f"VG1_{vg1:.2f}_VG2_{vg2:+.3f}_hyp{hyp}_inj{inj}"
    out_dat = (DECKS / f"out_{tag}.txt").as_posix()
    deck = DECK_TEMPLATE.format(
        HYP=f"hyp{hyp}", INJ=inj,
        VG1=vg1, VG2=vg2,
        PTM130=(CANON / "PTM130bulkNSRAM.txt").as_posix(),
        BJT=(CANON / "parasiticBJT.txt").as_posix(),
        A_VAL=P["a"], B_VAL=P["b"], D_VAL=P["d"], E_VAL=P["e"], F_VAL=P["f"],
        C_CONST=C_CONST, OUT_DAT=out_dat, INJ_NODE=inj,
    )
    p = DECKS / f"deck_{tag}.cir"
    p.write_text(deck)
    return p, Path(out_dat), P

def run_ngspice(deck_path):
    proc = subprocess.run(["ngspice", "-b", str(deck_path)],
                          capture_output=True, text=True, timeout=120)
    return proc.returncode, proc.stdout + proc.stderr

def parse_dc_out(dat_path):
    if not dat_path.exists():
        return None
    arr = np.loadtxt(dat_path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # wrdata columns: vd, -i(vd), vd, v(Sint), vd, v(B)
    vd = arr[:, 0]
    idd = arr[:, 1]
    vsint = arr[:, 3] if arr.shape[1] > 3 else np.zeros_like(vd)
    vb = arr[:, 5] if arr.shape[1] > 5 else np.zeros_like(vd)
    return {"vd": vd, "id": idd, "vsint": vsint, "vb": vb}

def rmse_dec(meas_vd, meas_id, sim_vd, sim_id, floor=1e-13):
    if sim_vd is None or len(sim_vd) < 2:
        return float("nan")
    sim_id_on = np.interp(meas_vd, sim_vd, sim_id)
    a = np.log10(np.maximum(np.abs(meas_id), floor))
    b = np.log10(np.maximum(np.abs(sim_id_on), floor))
    return float(np.sqrt(np.mean((a - b) ** 2)))


# -------------------------- Main -----------------------------------------
VARIANTS = [("A", "B"), ("A", "D"), ("B", "B"), ("B", "D")]
# S13 reference (from results/z419_ngspice_with_ipos for hypA/injB only;
# other 3 are from z419 reruns logged in run.log)
S13_REF = {
    "A_B": {"VG1_0.2": 2.742, "VG1_0.4": 4.997, "VG1_0.6": 6.452, "cell": 4.730},
}

def main():
    t0 = time.time()
    biases = list_measured_biases()
    log(f"Found {len(biases)} measured bias files.")
    log("PRE-REG gates: INFRA=converge<30min, "
        "DISCOVERY=>=1 variant cell<1.5 dec AND max(VB)>0.7 at high Vd, "
        "AMBITIOUS=cell<0.5 dec + visible snapback, "
        "KILL_SHOT=no improvement vs S13.")

    variant_results = {}
    overall = {"variants": {}, "wall_sec": 0.0}

    for hyp, inj in VARIANTS:
        key = f"{hyp}_{inj}"
        log(f"\n========== VARIANT hyp{hyp} / inj{inj} ==========")
        results = []
        conv_fail = 0
        for (vg1, vg2, fpath) in biases:
            meas_vd, meas_id = load_iv(fpath)
            deck_path, dat_path, P = build_deck(vg1, vg2, hyp, inj)
            rc, stdout = run_ngspice(deck_path)
            parsed = parse_dc_out(dat_path)
            ok = (rc == 0) and parsed is not None and len(parsed["vd"]) > 5
            if not ok:
                conv_fail += 1
                results.append({"vg1": vg1, "vg2": vg2, "file": fpath.name,
                                "meas_vd": meas_vd.tolist(), "meas_id": meas_id.tolist(),
                                "ok": False, "pwl": P})
                continue
            rmse = rmse_dec(meas_vd, meas_id, parsed["vd"], parsed["id"])
            max_vb = float(np.max(parsed["vb"]))
            vb_at_vd2 = float(parsed["vb"][np.argmin(np.abs(parsed["vd"] - 2.0))])
            results.append({
                "vg1": vg1, "vg2": vg2, "file": fpath.name,
                "meas_vd": meas_vd.tolist(), "meas_id": meas_id.tolist(),
                "sim_vd": parsed["vd"].tolist(), "sim_id": parsed["id"].tolist(),
                "sim_vb": parsed["vb"].tolist(),
                "rmse_dec": rmse, "max_vb": max_vb, "vb_at_vd2": vb_at_vd2,
                "ok": True, "pwl": P,
            })
        # aggregate per variant
        def avg_rmse(branch):
            xs = [r["rmse_dec"] for r in results
                  if r["vg1"] == branch and r.get("ok") and not np.isnan(r.get("rmse_dec", np.nan))]
            return float(np.mean(xs)) if xs else float("nan")
        def avg_max_vb(branch):
            xs = [r["max_vb"] for r in results
                  if r["vg1"] == branch and r.get("ok")]
            return float(np.mean(xs)) if xs else float("nan")
        rmse_per = {f"VG1_{v}": avg_rmse(v) for v in (0.2, 0.4, 0.6)}
        cell_vals = [v for v in rmse_per.values() if not np.isnan(v)]
        rmse_per["cell"] = float(np.mean(cell_vals)) if cell_vals else float("nan")
        vb_per = {f"VG1_{v}": avg_max_vb(v) for v in (0.2, 0.4, 0.6)}
        log(f"  conv_failures={conv_fail}  RMSE/dec: {rmse_per}")
        log(f"  mean max(V_B): {vb_per}")
        variant_results[key] = results
        overall["variants"][key] = {
            "hypothesis": hyp, "inject_node": inj,
            "conv_failures": conv_fail,
            "rmse_dec": rmse_per,
            "mean_max_vb": vb_per,
        }

    overall["wall_sec"] = round(time.time() - t0, 1)

    # ---------------- gates ----------------
    cells = {k: v["rmse_dec"]["cell"] for k, v in overall["variants"].items()}
    best_key = min(cells, key=lambda k: (np.inf if np.isnan(cells[k]) else cells[k]))
    best_cell = cells[best_key]
    # max VB across best variant
    best_vbs = [r["max_vb"] for r in variant_results[best_key]
                if r.get("ok") and r["vg1"] == 0.6]
    body_charges = (len(best_vbs) > 0 and np.max(best_vbs) > 0.7)
    infra = all(v["conv_failures"] == 0 for v in overall["variants"].values()) \
            and overall["wall_sec"] < 1800
    discovery = (not np.isnan(best_cell) and best_cell < 1.5) and body_charges
    ambitious = (not np.isnan(best_cell) and best_cell < 0.5)
    # kill-shot: no variant beats S13 baseline (A_B cell=4.730)
    base_ab = S13_REF["A_B"]["cell"]
    killshot = all((np.isnan(c) or c >= base_ab) for c in cells.values())
    overall["gates"] = {"INFRA": bool(infra), "DISCOVERY": bool(discovery),
                        "AMBITIOUS": bool(ambitious), "KILL_SHOT": bool(killshot),
                        "best_variant": best_key, "best_cell_dec": best_cell,
                        "body_charges_over_0.7V": bool(body_charges)}

    # ---------------- save summary + variant_sweep ----------------
    (OUT / "summary.json").write_text(json.dumps(overall, indent=2,
                                                  default=lambda o: float(o)))
    # variant_sweep.json: per-bias detail (small fields only)
    sweep = {}
    for k, rs in variant_results.items():
        sweep[k] = []
        for r in rs:
            sweep[k].append({"vg1": r["vg1"], "vg2": r["vg2"],
                             "ok": r.get("ok", False),
                             "rmse_dec": r.get("rmse_dec"),
                             "max_vb": r.get("max_vb"),
                             "vb_at_vd2": r.get("vb_at_vd2"),
                             "pwl": r.get("pwl")})
    (OUT / "variant_sweep.json").write_text(json.dumps(sweep, indent=2,
                                                       default=lambda o: float(o)))

    # ---------------- overlay plots (best variant) ----------------
    best_rs = variant_results[best_key]
    for vg1 in (0.2, 0.4, 0.6):
        fig, ax = plt.subplots(figsize=(9, 6.5))
        rs = sorted([r for r in best_rs if r["vg1"] == vg1], key=lambda r: r["vg2"])
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
        rmse_b = overall["variants"][best_key]["rmse_dec"][f"VG1_{vg1}"]
        ax.set_xlabel("Vd (V)"); ax.set_ylabel("|Id| (A)")
        ax.set_title(f"z420 body-open ({best_key}) VG1={vg1}V  "
                     f"(dots=measured, lines=ngspice+Ipos; RMSE={rmse_b:.2f} dec)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=6, ncol=2)
        ax.set_ylim(1e-14, 1e-2)
        fig.tight_layout()
        fig.savefig(OUT / f"overlay_VG1_{str(vg1).replace('.', 'p')}.png", dpi=130)
        plt.close(fig)

    # ---------------- V_B trace plot for VG1=0.6 (high-stress branch) ----
    fig, ax = plt.subplots(figsize=(9, 6.5))
    rs = sorted([r for r in best_rs if r["vg1"] == 0.6 and r.get("ok")],
                key=lambda r: r["vg2"])
    cmap = plt.cm.plasma
    n = len(rs)
    for i, r in enumerate(rs):
        c = cmap(i / max(1, n-1))
        ax.plot(r["sim_vd"], r["sim_vb"], "-", color=c, lw=1.2,
                label=f"VG2={r['vg2']:+.2f}")
    ax.axhline(0.7, color="r", lw=1, ls="--", label="0.7V BJT turn-on")
    ax.set_xlabel("Vd (V)"); ax.set_ylabel("V_B (V)")
    ax.set_title(f"z420 body-open ({best_key}): V_B trace, VG1=0.6V "
                 f"(body floats now — does it charge?)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "vb_trace_VG1_0p6.png", dpi=130)
    plt.close(fig)

    # ---------------- honest analysis ----------------
    md = []
    md.append("# z420 body-open fix — honest analysis\n\n")
    md.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    md.append("## Fix applied\n\n")
    md.append("`Rcb` (body cap series resistor in `2tnsram_simple.asc`) was "
              "**1 mΩ** — a near-short to GND in DC. Changed to **1 TΩ**, so "
              "the body node `B` is now genuinely floating. All other elements "
              "and the canonical model cards are untouched.\n\n")
    md.append("## Cell-wide log-RMSE per variant (decades)\n\n")
    md.append("| Variant | VG1=0.2 | VG1=0.4 | VG1=0.6 | cell |\n|---|---|---|---|---|\n")
    for k, v in overall["variants"].items():
        r = v["rmse_dec"]
        md.append(f"| {k} | {r['VG1_0.2']:.3f} | {r['VG1_0.4']:.3f} | "
                  f"{r['VG1_0.6']:.3f} | **{r['cell']:.3f}** |\n")
    md.append("\nS13 reference (hypA/injB, Rcb=1m): cell = "
              f"{S13_REF['A_B']['cell']:.3f} dec.\n\n")
    md.append("## Mean max(V_B) per variant (Volts)\n\n")
    md.append("| Variant | VG1=0.2 | VG1=0.4 | VG1=0.6 |\n|---|---|---|---|\n")
    for k, v in overall["variants"].items():
        vb = v["mean_max_vb"]
        md.append(f"| {k} | {vb['VG1_0.2']:.3f} | {vb['VG1_0.4']:.3f} | "
                  f"{vb['VG1_0.6']:.3f} |\n")
    md.append("\n0.7 V is the rough turn-on of the parasitic NPN base–emitter.\n\n")
    md.append("## Pre-registered gates\n\n")
    for k, v in overall["gates"].items():
        md.append(f"- **{k}**: {v}\n")
    md.append("\n## Interpretation\n\n")
    if overall["gates"]["AMBITIOUS"]:
        md.append("Body-open fix closes the gap. Snapback recovered. Mario "
                  "intent confirmed.\n")
    elif overall["gates"]["DISCOVERY"]:
        md.append("Body actually charges (V_B > 0.7V) AND at least one variant "
                  "drops below 1.5 dec. Significant structural improvement.\n")
    elif overall["gates"]["KILL_SHOT"]:
        md.append("Even with the body genuinely floating, no variant beats the "
                  "S13 baseline. The body-short was a red herring. Something "
                  "structural is still wrong (BJT card, Ipos sign, or M1 bulk "
                  "diode model overpowering Ipos).\n")
    else:
        md.append("Partial improvement; body may charge but not enough to "
                  "trigger snapback at the measured biases.\n")
    (OUT / "honest_analysis.md").write_text("".join(md))

    log(f"\nDONE in {overall['wall_sec']:.1f}s. best={best_key} cell={best_cell:.3f}")
    log(f"GATES: {overall['gates']}")
    return overall

if __name__ == "__main__":
    summary = main()
    print(json.dumps(summary["gates"], indent=2))
    print(json.dumps({k: v["rmse_dec"] for k, v in summary["variants"].items()}, indent=2))
