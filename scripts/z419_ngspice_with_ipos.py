#!/usr/bin/env python3
"""z419: ngspice simulation with Mario's digitized Ipos PWL block.

Goal: test whether Mario's complete semi-empirical model (with manually digitized
PWL parameters for a, b, d, e, f from slide 12.26) closes the 5-decade gap
between canonical SPICE cards and Sebas's 33 measured IV curves.

Formula per slide 12.26:
    I_ion = I_exp + I_pow
    I_exp = a * exp(b * (V_D + c))
    I_pow = d * (V_D + f)^e   if V_D > -f, else 0
    where a, b, d, e, f = PWL(V_G), c = -2.4 V constant

Hypothesis A (default): the PWL is parametrized by V_G2 (the swept gate in
Sebas's data), so each measured bias gets its own (a,b,d,e,f).

Pre-registered gates:
- INFRA: all 33 ngspice runs converge, < 30 min wall
- DISCOVERY: cell-wide < 1.5 dec AND >= 2 VG1 branches match shape
- AMBITIOUS: cell-wide < 0.5 dec AND all 3 VG1 visually track
- KILL_SHOT: Ipos still leaves >= baseline RMSE (3.17 dec) -> hypothesis wrong
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
OUT = REPO / "results" / "z419_ngspice_with_ipos"
DECKS = OUT / "decks"
OUT.mkdir(parents=True, exist_ok=True)
DECKS.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT / "run.log"
LOG_FH = open(LOG_PATH, "w")

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_FH.write(line + "\n"); LOG_FH.flush()


# -------------------------- PWL loader --------------------------
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
    # Ensure sorted by x
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
    """Return dict of (a,b,d,e,f) at gate voltage vg (linear interp, clipped)."""
    out = {}
    for nm, (xs, ys) in PWL.items():
        out[nm] = float(np.interp(vg, xs, ys))
    return out


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
                vg2 = float(m.group(1))
                out.append((vg1, vg2, f))
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


# -------------------------- Deck builder --------------------------
DECK_TEMPLATE = """.title z419 ngspice with Ipos PWL ({HYP}: VG1={VG1:.3f}, VG2={VG2:.3f})

* Compatibility shim for LTspice-flavored canonical card
.param vsatn=1.35e5
.param Nparam=1.58
.param Citparam=0
.param Voffparam=-0.1368
.param K2Par=-0.070435
.param toxn=4e-9

* Schematic params (verbatim from 2tnsram_simple.asc)
.param Ln=0.18u
.param Wn=0.36u
.param CBpar=1f

* Canonical model card includes (UNTOUCHED)
.include "{PTM130}"
.include "{BJT}"

* Schematic nodes: D, Sint, S=0, G, G2, B (floating bulk on M1)
Vd   D   0   DC 0
Vg1  G   0   DC {VG1:.4f}
Vg2  G2  0   DC {VG2:.4f}

* M1 nmos4: D=D, G=G, S=Sint, B=B, L=0.18u, W=0.36u
M1 D G Sint B NMOS L=0.18u W=0.36u

* M2 nmos4: D=Sint, G=G2, S=0, B=0, L=1.8u, W=0.36u
M2 Sint G2 0 0 NMOS L=1.8u W=0.36u

* Q1 parasitic NPN: C=D, B=B, E=0
Q1 D B 0 parasiticBJT area=1u

* C1: B to 0
C1 B Bx 1f
Rcb Bx 0 1m

* ---- Mario Ipos PWL block (digitized from slide 12.26) ----
* I_ion = a*exp(b*(V_D + c)) + ((V_D+f)>0 ? d*(V_D+f)^e : 0)
* Injected INTO body node B (positive = charging body up).
.param a_val={A_VAL:.6e}
.param b_val={B_VAL:.6f}
.param d_val={D_VAL:.6e}
.param e_val={E_VAL:.6f}
.param f_val={F_VAL:.6f}
.param c_const={C_CONST:.6f}

* ngspice B-source: ternary via ((cond) ? ... : ...) works in nutmeg.
* Use 'V(D)' for drain voltage.
B_ipos {INJ_NODE} 0 I = a_val*exp(b_val*(V(D)+c_const)) + (((V(D)+f_val) > 0) ? (d_val*pow(V(D)+f_val, e_val)) : 0)

.options gmin=1e-15 abstol=1e-15 reltol=1e-3 itl1=500 itl2=200 itl6=100

.control
dc Vd 0 2 0.05
wrdata {OUT_DAT} -i(vd) v(Sint) v(B)
quit
.endc

.end
"""


def build_deck(vg1, vg2, hypothesis="A", inject_node="B"):
    """hypothesis A: PWL evaluated at VG2 (per task). hypothesis B: at VG1."""
    if hypothesis == "A":
        vg_for_pwl = vg2
    elif hypothesis == "B":
        vg_for_pwl = vg1
    else:
        raise ValueError(hypothesis)
    P = pwl_at(vg_for_pwl)
    tag = f"VG1_{vg1:.2f}_VG2_{vg2:+.3f}_hyp{hypothesis}_inj{inject_node}"
    out_dat = (DECKS / f"out_{tag}.txt").as_posix()
    deck = DECK_TEMPLATE.format(
        HYP=f"hyp{hypothesis}",
        VG1=vg1, VG2=vg2,
        PTM130=(CANON / "PTM130bulkNSRAM.txt").as_posix(),
        BJT=(CANON / "parasiticBJT.txt").as_posix(),
        A_VAL=P["a"], B_VAL=P["b"], D_VAL=P["d"], E_VAL=P["e"], F_VAL=P["f"],
        C_CONST=C_CONST,
        OUT_DAT=out_dat,
        INJ_NODE=inject_node,
    )
    deck_path = DECKS / f"deck_{tag}.cir"
    deck_path.write_text(deck)
    return deck_path, Path(out_dat), P


def run_ngspice(deck_path: Path):
    proc = subprocess.run(
        ["ngspice", "-b", str(deck_path)],
        capture_output=True, text=True, timeout=120,
    )
    return proc.returncode, proc.stdout + proc.stderr


def parse_dc_out(dat_path: Path):
    if not dat_path.exists():
        return None
    arr = np.loadtxt(dat_path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    vd = arr[:, 0]
    idd = arr[:, 1]
    return {"vd": vd, "id": idd}


def rmse_dec(meas_vd, meas_id, sim_vd, sim_id, floor=1e-13):
    if sim_vd is None or len(sim_vd) < 2:
        return float("nan")
    sim_id_on = np.interp(meas_vd, sim_vd, sim_id)
    a = np.log10(np.maximum(np.abs(meas_id), floor))
    b = np.log10(np.maximum(np.abs(sim_id_on), floor))
    return float(np.sqrt(np.mean((a - b) ** 2)))


# -------------------------- Main --------------------------
def main():
    t0 = time.time()
    biases = list_measured_biases()
    log(f"Found {len(biases)} measured bias files.")
    log(f"PWL VG range: a={PWL['a'][0].min():.3f}-{PWL['a'][0].max():.3f}")

    # Pre-registered: log gates BEFORE
    log("PRE-REG gates: INFRA=converge<30min, DISCOVERY=<1.5dec+2VG1, AMBITIOUS=<0.5dec, KILL=>=baseline")

    HYPOTHESIS = os.environ.get("Z419_HYP", "A")   # A=PWL(V_G2), B=PWL(V_G1)
    INJECT = os.environ.get("Z419_INJ", "B")       # B=body, D=drain

    results = []
    convergence_failures = 0

    for (vg1, vg2, fpath) in biases:
        try:
            meas_vd, meas_id = load_iv(fpath)
        except Exception as e:
            log(f"  load FAIL {fpath.name}: {e}"); continue

        deck_path, dat_path, P = build_deck(vg1, vg2, HYPOTHESIS, INJECT)
        rc, stdout = run_ngspice(deck_path)
        (DECKS / f"log_VG1_{vg1:.2f}_VG2_{vg2:+.3f}.txt").write_text(stdout)
        parsed = parse_dc_out(dat_path)
        ok = (rc == 0) and parsed is not None and len(parsed["vd"]) > 5
        if not ok:
            convergence_failures += 1
            log(f"  CONV FAIL VG1={vg1} VG2={vg2:+.3f} rc={rc}")
            results.append({"vg1": vg1, "vg2": vg2, "file": fpath.name,
                            "meas_vd": meas_vd.tolist(),
                            "meas_id": meas_id.tolist(),
                            "ok": False, "pwl": P})
            continue
        rmse = rmse_dec(meas_vd, meas_id, parsed["vd"], parsed["id"])
        log(f"  VG1={vg1} VG2={vg2:+.3f}: RMSE={rmse:.3f} dec  (a={P['a']:.2e} f={P['f']:.3f})")
        results.append({
            "vg1": vg1, "vg2": vg2, "file": fpath.name,
            "meas_vd": meas_vd.tolist(), "meas_id": meas_id.tolist(),
            "sim_vd": parsed["vd"].tolist(), "sim_id": parsed["id"].tolist(),
            "rmse_dec": rmse, "ok": True, "pwl": P,
        })

    # ---------------- aggregate ----------------
    def avg_rmse(branch_vg1):
        xs = [r["rmse_dec"] for r in results
              if r["vg1"] == branch_vg1 and r.get("ok") and not np.isnan(r.get("rmse_dec", np.nan))]
        return float(np.mean(xs)) if xs else float("nan")

    summary = {
        "n_biases": len(results),
        "convergence_failures": convergence_failures,
        "hypothesis": HYPOTHESIS,
        "inject_node": INJECT,
        "rmse_dec": {
            "ipos_VG1_0.2": avg_rmse(0.2),
            "ipos_VG1_0.4": avg_rmse(0.4),
            "ipos_VG1_0.6": avg_rmse(0.6),
        },
        "wall_sec": round(time.time() - t0, 1),
        "baseline_s11_dec": {"VG1_0.2": 1.379, "VG1_0.4": 3.509, "VG1_0.6": 4.621, "cell": 3.170},
    }
    cell_vals = [v for k, v in summary["rmse_dec"].items() if not np.isnan(v)]
    summary["rmse_dec"]["ipos_cell"] = float(np.mean(cell_vals)) if cell_vals else float("nan")

    # Gates
    infra = (convergence_failures == 0 and summary["wall_sec"] < 1800)
    branch_rmse = [summary["rmse_dec"][f"ipos_VG1_{v}"] for v in ("0.2", "0.4", "0.6")]
    n_better_than_1p5 = sum(1 for r in branch_rmse if not np.isnan(r) and r < 1.5)
    discovery = (not np.isnan(summary["rmse_dec"]["ipos_cell"])
                 and summary["rmse_dec"]["ipos_cell"] < 1.5
                 and n_better_than_1p5 >= 2)
    ambitious = (not np.isnan(summary["rmse_dec"]["ipos_cell"])
                 and summary["rmse_dec"]["ipos_cell"] < 0.5)
    killshot = (not np.isnan(summary["rmse_dec"]["ipos_cell"])
                and summary["rmse_dec"]["ipos_cell"] >= 3.17)

    summary["gates"] = {
        "INFRA": bool(infra),
        "DISCOVERY": bool(discovery),
        "AMBITIOUS": bool(ambitious),
        "KILL_SHOT": bool(killshot),
    }

    # ---------------- plots ----------------
    for vg1 in (0.2, 0.4, 0.6):
        fig, ax = plt.subplots(figsize=(9, 6.5))
        # Color by VG2
        rs = [r for r in results if r["vg1"] == vg1]
        rs.sort(key=lambda r: r["vg2"])
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
        rmse_branch = summary["rmse_dec"][f"ipos_VG1_{vg1}"]
        ax.set_title(f"z419 ngspice + Ipos PWL: VG1={vg1}V  "
                     f"(dots=measured, lines=ngspice; RMSE={rmse_branch:.2f} dec)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=6, ncol=2)
        ax.set_ylim(1e-14, 1e-2)
        fig.tight_layout()
        fig.savefig(OUT / f"overlay_VG1_{str(vg1).replace('.', 'p')}.png", dpi=130)
        plt.close(fig)

    # ---------------- save ----------------
    # Strip big arrays before json dump
    summary_json = json.loads(json.dumps(summary, default=lambda o: float(o)))
    summary_json["per_bias"] = []
    for r in results:
        summary_json["per_bias"].append({
            "vg1": r["vg1"], "vg2": r["vg2"], "file": r["file"],
            "ok": r.get("ok", False),
            "rmse_dec": r.get("rmse_dec"),
            "pwl": r.get("pwl"),
        })
    (OUT / "summary.json").write_text(json.dumps(summary_json, indent=2))
    np.savez_compressed(OUT / "traces.npz", results=np.array(results, dtype=object))

    log(f"DONE in {summary['wall_sec']:.1f}s. RMSE/dec: {summary['rmse_dec']}")
    log(f"GATES: {summary['gates']}")

    # ---------------- honest analysis ----------------
    base = summary["baseline_s11_dec"]
    rd = summary["rmse_dec"]
    md = []
    md.append(f"# z419 ngspice + Ipos PWL — honest analysis\n")
    md.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append(f"Hypothesis: {HYPOTHESIS} (PWL(V_G2)), inject at node {INJECT}\n\n")
    md.append("## Cell-wide log-RMSE (decades)\n\n")
    md.append("| Branch | baseline (S11) | z419+Ipos | Δ |\n")
    md.append("|---|---|---|---|\n")
    for k, base_key in (("0.2", "VG1_0.2"), ("0.4", "VG1_0.4"), ("0.6", "VG1_0.6")):
        b = base[base_key]; r = rd[f"ipos_VG1_{k}"]
        delta = (r - b) if not np.isnan(r) else float("nan")
        md.append(f"| VG1={k} | {b:.3f} | {r:.3f} | {delta:+.3f} |\n")
    md.append(f"| **cell** | **{base['cell']:.3f}** | **{rd['ipos_cell']:.3f}** | "
              f"**{(rd['ipos_cell']-base['cell']):+.3f}** |\n\n")
    md.append("## Pre-registered gates\n\n")
    for k, v in summary["gates"].items():
        md.append(f"- {k}: **{'PASS' if v else 'FAIL'}**\n")
    md.append("\n## Interpretation\n\n")
    if summary["gates"]["AMBITIOUS"]:
        md.append("Ipos closes the gap (cell < 0.5 dec). Mario's model is reproducible.\n")
    elif summary["gates"]["DISCOVERY"]:
        md.append("Ipos materially improves match (cell < 1.5 dec, 2+ VG1 branches).\n"
                  "Some gap remains; likely PWL accuracy or VG1-dependence.\n")
    elif summary["gates"]["KILL_SHOT"]:
        md.append("Ipos with Hypothesis A (PWL(V_G2)) does NOT improve over baseline.\n"
                  "Likely interpretations:\n"
                  "1. V_G in slide 12.26 is V_G1, not V_G2 → run Hypothesis B.\n"
                  "2. Injection node wrong (try drain instead of body).\n"
                  "3. Sign error: Ipos drives body up; with body BJT we may need -Ipos.\n"
                  "4. PWL digitization error (compare a(0.15)≈6.6e-12 to slide).\n")
    else:
        md.append("Ipos partially helps; gap still wide. Re-examine PWL or injection node.\n")
    (OUT / "honest_analysis.md").write_text("".join(md))

    return summary

if __name__ == "__main__":
    summary = main()
    print(json.dumps(summary["gates"], indent=2))
    print(json.dumps(summary["rmse_dec"], indent=2))
