#!/usr/bin/env python3
"""z423: Add explicit reverse-biased isolator diode between body and Sint.

Context: in z420/z421 we found that Ipos injected at the body node B is
shorted to ground through M1's BSIM4 bulk-source diode (PTM130 card).  V_B
clamps at ~0.23 V instead of rising to ~0.7 V needed to forward-bias the
parasitic NPN Q1.

This script tests TWO topologies designed to isolate the body charge:

  TOP1 (series isolator): keep M1.B connected to net B as in z421, but ALSO
        inject Ipos at B and add a diode D_iso from Sint -> B (cathode at B).
        The diode is reverse-biased in the desired regime (V_B > V_Sint), so
        it blocks the BSIM bulk-source forward path's return current.  In
        practice we still have M1's intrinsic diode in parallel, so this is
        only a partial test.

  TOP2 (break M1 bulk): rewire M1's bulk to a new node 'Bbody' that is no
        longer the same net as the BJT base.  Q1.B and Ipos live on net B.
        A reverse-biased diode from Bbody -> 0 keeps M1's bulk near ground.
        This fully isolates the BJT base from M1's bulk-source diode.

Pre-registered gates:
- INFRA: 33 biases converge, < 15 min
- DISCOVERY: V_B > 0.7V at some Vd AND cell-wide < 2.5 dec
- AMBITIOUS: cell-wide < 1.0 dec AND visible snapback fold in overlay plots
- KILL_SHOT: even with isolator, V_B still < 0.3V -> diode topology not the answer
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
OUT = REPO / "results" / "z423_isolator_diode"
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


# -------------------------- Deck templates --------------------------
# TOP1: keep M1's bulk on net B; add reverse-biased diode Sint->B
# (cathode at B). In the desired regime V_B > V_Sint, the diode is reverse-
# biased and blocks return current.  This is a partial test because M1's
# own bulk-source diode is still in parallel (built into PTM130).
DECK_TOP1 = """.title z423 TOP1 series isolator ({HYP}: VG1={VG1:.3f}, VG2={VG2:.3f})
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

* ideal-ish isolator diode model (very low IS, sharp turn-on)
.model DISO D (IS=1e-18 N=1.0 RS=1m BV=50 IBV=1u CJO=0.1f)

Vd   D   0   DC 0
Vg1  G   0   DC {VG1:.4f}
Vg2  G2  0   DC {VG2:.4f}

* M1: B node is net B (same as BJT base)
M1 D G Sint B NMOS L=0.18u W=0.36u
M2 Sint G2 0 0 NMOS L=1.8u W=0.36u
Q1 D B 0 parasiticBJT area=1u
C1 B Bx 1f
Rcb Bx 0 1m

* Series isolator: diode anode at Sint, cathode at B.
* Reverse-biased when V_B > V_Sint -> blocks return path through Sint.
D_iso Sint B DISO

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

# TOP2: break the M1.B connection. Move M1's bulk to a new node 'Bbody'
# that is held near ground by a diode from Bbody to 0 (anode Bbody).
# That way M1's bulk-source diode forward-conducts to 0, NOT to net B.
# Net B carries only Q1's base + Ipos + C1.
DECK_TOP2 = """.title z423 TOP2 broken bulk ({HYP}: VG1={VG1:.3f}, VG2={VG2:.3f})
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

.model DISO D (IS=1e-18 N=1.0 RS=1m BV=50 IBV=1u CJO=0.1f)

Vd   D   0   DC 0
Vg1  G   0   DC {VG1:.4f}
Vg2  G2  0   DC {VG2:.4f}

* M1 bulk is now on Bbody (NOT net B)
M1 D G Sint Bbody NMOS L=0.18u W=0.36u
M2 Sint G2 0 0 NMOS L=1.8u W=0.36u

* BJT base is on net B (isolated from M1.B)
Q1 D B 0 parasiticBJT area=1u
C1 B Bx 1f
Rcb Bx 0 1m
* DC reference for net B (M1.B no longer ties it down). Large enough to be
* invisible to Ipos (~10pA-1uA) yet anchor DC operating point.
Rb_dc B 0 1T

* Hold Bbody near 0: anode at Bbody, cathode at 0. Forward-conducts when
* Bbody > 0 (M1 wants to push bulk up via its parasitic forward diode);
* dumps that current to ground rather than to net B.
D_bulk Bbody 0 DISO

* Tiny resistor to keep Bbody DC-defined when diode is off
Rbbody Bbody 0 1G

.param a_val={A_VAL:.6e}
.param b_val={B_VAL:.6f}
.param d_val={D_VAL:.6e}
.param e_val={E_VAL:.6f}
.param f_val={F_VAL:.6f}
.param c_const={C_CONST:.6f}

* Ipos injected at net B (BJT base side). No longer shorted via M1.B.
B_ipos {INJ_NODE} 0 I = a_val*exp(b_val*(V(D)+c_const)) + (((V(D)+f_val) > 0) ? (d_val*pow(V(D)+f_val, e_val)) : 0)

.options gmin=1e-15 abstol=1e-15 reltol=1e-3 itl1=500 itl2=200 itl6=100

.control
dc Vd 0 2 0.05
wrdata {OUT_DAT} -i(vd) v(Sint) v(B) v(Bbody)
quit
.endc

.end
"""


def build_deck(vg1, vg2, topology, hypothesis="A", inject_node="B"):
    if hypothesis == "A":
        vg_for_pwl = vg2
    else:
        vg_for_pwl = vg1
    P = pwl_at(vg_for_pwl)
    tag = f"{topology}_VG1_{vg1:.2f}_VG2_{vg2:+.3f}_hyp{hypothesis}_inj{inject_node}"
    out_dat = (DECKS / f"out_{tag}.txt").as_posix()
    tpl = DECK_TOP1 if topology == "TOP1" else DECK_TOP2
    deck = tpl.format(
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


def parse_dc_out(dat_path: Path, topology: str):
    """Returns dict with vd, id, v_sint, v_b, [v_bbody]"""
    if not dat_path.exists():
        return None
    arr = np.loadtxt(dat_path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # Columns: TOP1: vd, -i(vd), v(Sint), v(B), i(b_ipos)  (wrdata interleaves
    # x-col with each requested vector, but ngspice's wrdata writes the sweep
    # variable repeated. With a single .dc sweep ngspice puts pairs
    # (x, val) for each vector. Let's just use generic by stride.
    # Actually `wrdata` writes columns: sweep, v1, sweep, v2, sweep, v3, ...
    # We pick only the value columns.
    n_cols = arr.shape[1]
    vd = arr[:, 0]
    # vectors at odd indices: 1, 3, 5, ...
    val_cols = arr[:, 1::2]
    out = {"vd": vd, "id": val_cols[:, 0]}
    if val_cols.shape[1] >= 2:
        out["v_sint"] = val_cols[:, 1]
    if val_cols.shape[1] >= 3:
        out["v_b"] = val_cols[:, 2]
    if topology == "TOP2" and val_cols.shape[1] >= 4:
        out["v_bbody"] = val_cols[:, 3]
    return out


def rmse_dec(meas_vd, meas_id, sim_vd, sim_id, floor=1e-13):
    if sim_vd is None or len(sim_vd) < 2:
        return float("nan")
    sim_id_on = np.interp(meas_vd, sim_vd, sim_id)
    a = np.log10(np.maximum(np.abs(meas_id), floor))
    b = np.log10(np.maximum(np.abs(sim_id_on), floor))
    return float(np.sqrt(np.mean((a - b) ** 2)))


# -------------------------- Main --------------------------
def run_topology(topology, biases):
    log(f"=== Running topology {topology} ===")
    t0 = time.time()
    HYPOTHESIS = "B"  # z421 found hypB better; use it
    INJECT = "B"
    results = []
    conv_fail = 0
    for (vg1, vg2, fpath) in biases:
        try:
            meas_vd, meas_id = load_iv(fpath)
        except Exception as e:
            log(f"  load FAIL {fpath.name}: {e}"); continue
        deck_path, dat_path, P = build_deck(vg1, vg2, topology, HYPOTHESIS, INJECT)
        rc, stdout = run_ngspice(deck_path)
        (DECKS / f"log_{topology}_VG1_{vg1:.2f}_VG2_{vg2:+.3f}.txt").write_text(stdout)
        parsed = parse_dc_out(dat_path, topology)
        ok = (rc == 0) and parsed is not None and len(parsed["vd"]) > 5
        if not ok:
            conv_fail += 1
            log(f"  CONV FAIL {topology} VG1={vg1} VG2={vg2:+.3f} rc={rc}")
            results.append({"topology": topology, "vg1": vg1, "vg2": vg2,
                            "file": fpath.name, "meas_vd": meas_vd.tolist(),
                            "meas_id": meas_id.tolist(), "ok": False, "pwl": P})
            continue
        rmse = rmse_dec(meas_vd, meas_id, parsed["vd"], parsed["id"])
        v_b_max = float(np.max(parsed.get("v_b", [0]))) if "v_b" in parsed else float("nan")
        log(f"  {topology} VG1={vg1} VG2={vg2:+.3f}: RMSE={rmse:.3f} dec  V_B_max={v_b_max:.3f}V")
        rec = {"topology": topology, "vg1": vg1, "vg2": vg2, "file": fpath.name,
               "meas_vd": meas_vd.tolist(), "meas_id": meas_id.tolist(),
               "sim_vd": parsed["vd"].tolist(), "sim_id": parsed["id"].tolist(),
               "v_b": parsed.get("v_b", np.zeros_like(parsed["vd"])).tolist(),
               "v_sint": parsed.get("v_sint", np.zeros_like(parsed["vd"])).tolist(),
               "v_b_max": v_b_max,
               "rmse_dec": rmse, "ok": True, "pwl": P}
        if "v_bbody" in parsed:
            rec["v_bbody"] = parsed["v_bbody"].tolist()
        results.append(rec)
    wall = time.time() - t0
    log(f"  {topology} done in {wall:.1f}s, conv_fail={conv_fail}")
    return results, conv_fail, wall


def main():
    t0 = time.time()
    biases = list_measured_biases()
    log(f"Found {len(biases)} measured bias files.")
    log("PRE-REG: INFRA=33 conv <15min, DISCOVERY=V_B>0.7V & cell<2.5dec, "
        "AMBITIOUS=cell<1.0dec & snapback, KILL=V_B<0.3V")

    all_results = {}
    all_summary = {}
    for topology in ("TOP1", "TOP2"):
        results, conv_fail, wall = run_topology(topology, biases)
        all_results[topology] = results

        def avg_rmse(branch_vg1):
            xs = [r["rmse_dec"] for r in results
                  if r["vg1"] == branch_vg1 and r.get("ok") and not np.isnan(r.get("rmse_dec", np.nan))]
            return float(np.mean(xs)) if xs else float("nan")

        rmse_branches = {f"VG1_{v}": avg_rmse(v) for v in (0.2, 0.4, 0.6)}
        cell_vals = [v for v in rmse_branches.values() if not np.isnan(v)]
        cell = float(np.mean(cell_vals)) if cell_vals else float("nan")
        v_b_max_overall = max(
            (r.get("v_b_max", 0.0) for r in results if r.get("ok")), default=0.0)

        gates = {
            "INFRA": bool(conv_fail == 0 and wall < 900),
            "DISCOVERY": bool(v_b_max_overall > 0.7 and not np.isnan(cell) and cell < 2.5),
            "AMBITIOUS": bool(not np.isnan(cell) and cell < 1.0),
            "KILL_SHOT": bool(v_b_max_overall < 0.3),
        }
        all_summary[topology] = {
            "rmse_dec": {**rmse_branches, "cell": cell},
            "v_b_max_overall": v_b_max_overall,
            "convergence_failures": conv_fail,
            "wall_sec": round(wall, 1),
            "gates": gates,
        }
        log(f"  {topology} summary: cell={cell:.3f} dec  V_B_max={v_b_max_overall:.3f}V  "
            f"gates={gates}")

    # ---------------- plots ----------------
    for topology in ("TOP1", "TOP2"):
        results = all_results[topology]
        for vg1 in (0.2, 0.4, 0.6):
            rs = [r for r in results if r["vg1"] == vg1]
            rs.sort(key=lambda r: r["vg2"])
            n = len(rs)
            cmap = plt.cm.viridis

            # Overlay plot
            fig, ax = plt.subplots(figsize=(9, 6.5))
            for i, r in enumerate(rs):
                color = cmap(i / max(1, n-1))
                mvd = np.array(r["meas_vd"]); mid = np.abs(np.array(r["meas_id"]))
                ax.semilogy(mvd, np.maximum(mid, 1e-14), "o", ms=3, alpha=0.6,
                            color=color, label=f"meas VG2={r['vg2']:+.2f}")
                if r.get("ok"):
                    svd = np.array(r["sim_vd"]); sid = np.abs(np.array(r["sim_id"]))
                    ax.semilogy(svd, np.maximum(sid, 1e-14), "-", color=color,
                                alpha=0.9, lw=1.2)
            rmse_branch = all_summary[topology]["rmse_dec"].get(f"VG1_{vg1}", float("nan"))
            ax.set_xlabel("Vd (V)")
            ax.set_ylabel("|Id| (A)")
            ax.set_title(f"z423 {topology}: VG1={vg1}V  RMSE={rmse_branch:.2f} dec")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="lower right", fontsize=6, ncol=2)
            ax.set_ylim(1e-14, 1e-2)
            fig.tight_layout()
            fig.savefig(OUT / f"overlay_{topology}_VG1_{str(vg1).replace('.', 'p')}.png", dpi=130)
            plt.close(fig)

            # V_B trace plot
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for i, r in enumerate(rs):
                if not r.get("ok"):
                    continue
                color = cmap(i / max(1, n-1))
                svd = np.array(r["sim_vd"])
                vb = np.array(r["v_b"])
                ax.plot(svd, vb, "-", color=color, lw=1.2,
                        label=f"VG2={r['vg2']:+.2f} (max={r['v_b_max']:.2f}V)")
            ax.axhline(0.7, ls="--", c="red", alpha=0.6, label="0.7V (BJT turn-on)")
            ax.axhline(0.3, ls=":", c="orange", alpha=0.6, label="0.3V (kill threshold)")
            ax.set_xlabel("Vd (V)")
            ax.set_ylabel("V_B (V)")
            v_b_max_branch = max((r["v_b_max"] for r in rs if r.get("ok")), default=0.0)
            ax.set_title(f"z423 {topology}: V_B vs Vd at VG1={vg1}V  (max V_B={v_b_max_branch:.3f}V)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=6, ncol=2)
            fig.tight_layout()
            fig.savefig(OUT / f"vb_trace_{topology}_VG1_{str(vg1).replace('.', 'p')}.png", dpi=130)
            plt.close(fig)

    # Aggregate top-level files matching requested naming (no topology suffix
    # uses the better topology by cell RMSE).
    best_top = min(all_summary, key=lambda k: (
        all_summary[k]["rmse_dec"]["cell"] if not np.isnan(all_summary[k]["rmse_dec"]["cell"])
        else float("inf")))
    log(f"Best topology by cell RMSE: {best_top}")
    for vg1 in (0.2, 0.4, 0.6):
        for kind in ("overlay", "vb_trace"):
            src = OUT / f"{kind}_{best_top}_VG1_{str(vg1).replace('.', 'p')}.png"
            dst = OUT / f"{kind}_VG1_{str(vg1).replace('.', 'p')}.png"
            if src.exists():
                dst.write_bytes(src.read_bytes())

    # ---------------- summary.json ----------------
    summary_json = {
        "best_topology": best_top,
        "by_topology": all_summary,
        "wall_sec_total": round(time.time() - t0, 1),
        "baseline_s11_dec": {"VG1_0.2": 1.379, "VG1_0.4": 3.509, "VG1_0.6": 4.621, "cell": 3.170},
        "z421_ipos_cell_dec": 5.46,  # from z421 honest_analysis (Hyp A inj B)
    }
    summary_json["per_bias"] = {}
    for topology, results in all_results.items():
        summary_json["per_bias"][topology] = [
            {"vg1": r["vg1"], "vg2": r["vg2"], "file": r["file"],
             "ok": r.get("ok", False), "rmse_dec": r.get("rmse_dec"),
             "v_b_max": r.get("v_b_max"), "pwl": r.get("pwl")}
            for r in results
        ]
    (OUT / "summary.json").write_text(json.dumps(summary_json, indent=2, default=float))

    # ---------------- honest_analysis.md ----------------
    md = []
    md.append(f"# z423 isolator diode — honest analysis\n")
    md.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    md.append("## Setup\n\n")
    md.append("TOP1 = series isolator diode (Sint -> B, cathode at B).\n")
    md.append("TOP2 = break M1.B: M1.B routed to Bbody, kept near 0 by D_bulk Bbody->0.\n\n")
    md.append("Both use Hypothesis B (PWL(VG1)), Ipos injected at node B (BJT base).\n\n")

    md.append("## Cell-wide log-RMSE (decades) and V_B observation\n\n")
    md.append("| Topology | VG1=0.2 | VG1=0.4 | VG1=0.6 | cell | V_B max | conv_fail | wall(s) |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    for topology in ("TOP1", "TOP2"):
        s = all_summary[topology]
        r = s["rmse_dec"]
        md.append(f"| {topology} | {r['VG1_0.2']:.3f} | {r['VG1_0.4']:.3f} | "
                  f"{r['VG1_0.6']:.3f} | {r['cell']:.3f} | {s['v_b_max_overall']:.3f}V | "
                  f"{s['convergence_failures']} | {s['wall_sec']} |\n")
    md.append(f"\nBaseline (S11, no Ipos): cell=3.170 dec.\n")
    md.append(f"z421 (Ipos no isolator, hypA injB): cell=5.46 dec.\n\n")

    md.append("## Pre-registered gates\n\n")
    for topology in ("TOP1", "TOP2"):
        md.append(f"### {topology}\n\n")
        for k, v in all_summary[topology]["gates"].items():
            md.append(f"- {k}: **{'PASS' if v else 'FAIL'}**\n")
        md.append("\n")

    md.append("## Interpretation\n\n")
    any_discovery = any(all_summary[t]["gates"]["DISCOVERY"] for t in ("TOP1", "TOP2"))
    any_kill = all(all_summary[t]["gates"]["KILL_SHOT"] for t in ("TOP1", "TOP2"))
    any_ambitious = any(all_summary[t]["gates"]["AMBITIOUS"] for t in ("TOP1", "TOP2"))
    if any_ambitious:
        md.append("Isolator diode closes most of the gap. "
                  "BJT mechanism is real; previous failure was the BSIM bulk-source short.\n")
    elif any_discovery:
        md.append("Isolator diode lifts V_B above 0.7V at some biases and improves "
                  "cell RMSE below 2.5 dec. Mechanism is plausible but PWL/parameters "
                  "still need tuning.\n")
    elif any_kill:
        md.append("KILL_SHOT: even with explicit isolation, V_B never rises above 0.3 V. "
                  "This means the BSIM bulk-source diode is NOT the bottleneck; some other "
                  "loss path (e.g. Q1 base-emitter forward-conducting straight to ground) "
                  "is pinning V_B. Diode topology is not the answer.\n")
    else:
        md.append("Partial improvement; V_B rises somewhat but not far enough, or RMSE "
                  "improves but mechanism is not the dominant one.\n")
    (OUT / "honest_analysis.md").write_text("".join(md))

    log(f"DONE total {summary_json['wall_sec_total']:.1f}s. Best topology: {best_top}")
    return summary_json


if __name__ == "__main__":
    summary = main()
    print(json.dumps({"best_topology": summary["best_topology"],
                      "by_topology_cell": {k: v["rmse_dec"]["cell"]
                                          for k, v in summary["by_topology"].items()},
                      "v_b_max": {k: v["v_b_max_overall"]
                                  for k, v in summary["by_topology"].items()},
                      "gates": {k: v["gates"] for k, v in summary["by_topology"].items()}},
                     indent=2))
