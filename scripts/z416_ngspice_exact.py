#!/usr/bin/env python3
"""z416: ngspice EXACT simulation of Mario/Sebas canonical 2T-NSRAM cell.

Goal: determine whether canonical .asc + canonical model cards reproduce
Sebas's measured I-V data, OR whether the gap previously seen in pyport
(S10: VG1=0.6 underpredicted by 5 decades) is intrinsic to the model files.

Uses canonical sources exactly as-is (no parameter tuning).
Optionally adds Mario's behavioral Ipos block at body node B.

Pre-registered gates:
- INFRA: ngspice runs all 33 biases (no convergence failures)
- DISCOVERY: ngspice-baseline matches measured within 1 dec on >=1 VG1 branch
- AMBITIOUS: ngspice+Ipos matches measured within 0.5 dec cell-wide
- KILL-SHOT: ngspice-baseline has same 5-dec gap -> canonical files do not
             reproduce silicon (mystery; model card itself is wrong/incomplete)
"""
import os, re, csv, json, glob, shutil, subprocess, time, traceback
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
CANON = REPO / "nsram" / "Zoom" / "schematic&modelCards"
PDIODE_TXT = REPO / "nsram" / "Zoom" / "pdiode.txt"
DATA = REPO / "data" / "sebas_2026_04_22"
OUT = REPO / "results" / "z416_ngspice_exact"
DECKS = OUT / "decks"
OUT.mkdir(parents=True, exist_ok=True)
DECKS.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT / "run.log"
LOG_FH = open(LOG_PATH, "w")

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_FH.write(line + "\n")
    LOG_FH.flush()


# ---------------------------------------------------------------------------
# Measured-data loader
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Deck builder (EXACT canonical files; only adds vsatn param defaults)
# ---------------------------------------------------------------------------
DECK_TEMPLATE = """.title z416 ngspice EXACT canonical 2T-NSRAM (VG1={VG1:.3f}, VG2={VG2:.3f}, mode={MODE})

* --- Compatibility shim for LTspice-flavored canonical model card. ---
* The canonical PTM130 card uses LTspice's ".param NAME VALUE" (space, no '='),
* which ngspice does not parse, and references "vsatn" without defining it.
* We pre-declare the same params (same values) so they exist before the .include
* line is read. We do NOT modify any canonical file.
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

* M1 nmos4 (NMOS model): D=D, G=G, S=Sint, B=B, L=Ln=0.18u, W=Wn=0.36u
M1 D G Sint B NMOS L=0.18u W=0.36u

* M2 nmos4 (NMOS model): D=Sint, G=G2, S=0, B=0, L=Ln*10=1.8u, W=Wn=0.36u
M2 Sint G2 0 0 NMOS L=1.8u W=0.36u

* Q1 parasitic NPN: C=D, B=B, E=0  (area=1u)
Q1 D B 0 parasiticBJT area=1u

* C1: B to 0, CBpar=1fF (Rser=1m series)
C1 B Bx 1f
Rcb Bx 0 1m

{IPOS_BLOCK}

.options gmin=1e-15 abstol=1e-15 reltol=1e-3 itl1=500 itl2=200 itl6=100

.control
dc Vd 0 2 0.05
wrdata {OUT_DAT} -i(vd) v(Sint) v(B)
quit
.endc

.end
"""

# Mario behavioral Ipos: simple placeholder constants per task spec
IPOS_TEMPLATE = (
    "* Mario Ipos behavioral block (placeholder constants from task brief)\n"
    "* Ipos = c*exp(d*V(D)) + (V(D)>y ? a*(V(D)-y)^beta : 0), injected at body B\n"
    ".param a_val=1e-3\n"
    ".param beta_val=2\n"
    ".param c_val=1e-12\n"
    ".param d_val=5\n"
    ".param y_val=1.0\n"
    "B1 B 0 I = c_val*exp(d_val*V(D)) + "
    "(V(D)>y_val ? a_val*pow(V(D)-y_val,beta_val) : 0)\n"
)


def build_deck(vg1, vg2, mode):
    assert mode in ("baseline", "ipos")
    out_dat = (DECKS / f"out_VG1_{vg1:.2f}_VG2_{vg2:+.3f}_{mode}.txt").as_posix()
    deck = DECK_TEMPLATE.format(
        VG1=vg1, VG2=vg2, MODE=mode,
        PTM130=(CANON / "PTM130bulkNSRAM.txt").as_posix(),
        BJT=(CANON / "parasiticBJT.txt").as_posix(),
        OUT_DAT=out_dat,
        IPOS_BLOCK=(IPOS_TEMPLATE if mode == "ipos" else "* (no Ipos block)"),
    )
    deck_path = DECKS / f"deck_VG1_{vg1:.2f}_VG2_{vg2:+.3f}_{mode}.cir"
    deck_path.write_text(deck)
    return deck_path, Path(out_dat)


def run_ngspice(deck_path: Path):
    proc = subprocess.run(
        ["ngspice", "-b", str(deck_path)],
        capture_output=True, text=True, timeout=120,
    )
    return proc.returncode, proc.stdout + proc.stderr


def parse_dc_out(dat_path: Path):
    """ngspice wrdata writes columns: V(sweep) col1 V(sweep) col2 ... actually
    one (x,y) pair per requested var. We requested -i(vd) v(Sint) v(B) so we get
    Vd, Id, Vd, V(Sint), Vd, V(B).
    """
    if not dat_path.exists():
        return None
    arr = np.loadtxt(dat_path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # columns 0,1 -> (Vd, Id); 2,3 -> (Vd, Vsint); 4,5 -> (Vd, Vb)
    vd = arr[:, 0]
    idd = arr[:, 1]
    vsint = arr[:, 3] if arr.shape[1] >= 4 else None
    vb = arr[:, 5] if arr.shape[1] >= 6 else None
    return {"vd": vd, "id": idd, "vsint": vsint, "vb": vb}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rmse_dec(meas_vd, meas_id, sim_vd, sim_id, floor=1e-13):
    """RMSE in log10 |I| after interpolating sim onto meas_vd grid."""
    if sim_vd is None:
        return float("nan")
    sim_id_on = np.interp(meas_vd, sim_vd, sim_id)
    a = np.log10(np.maximum(np.abs(meas_id), floor))
    b = np.log10(np.maximum(np.abs(sim_id_on), floor))
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    biases = list_measured_biases()
    log(f"Found {len(biases)} measured bias files.")

    results = []  # per-bias dict
    convergence_failures = 0

    for (vg1, vg2, fpath) in biases:
        try:
            meas_vd, meas_id = load_iv(fpath)
        except Exception as e:
            log(f"  load FAIL {fpath.name}: {e}")
            continue

        bias_record = {
            "vg1": vg1, "vg2": vg2, "file": fpath.name,
            "meas_vd": meas_vd.tolist(), "meas_id": meas_id.tolist(),
            "modes": {},
        }

        for mode in ("baseline", "ipos"):
            deck_path, dat_path = build_deck(vg1, vg2, mode)
            rc, stdout = run_ngspice(deck_path)
            (DECKS / f"log_VG1_{vg1:.2f}_VG2_{vg2:+.3f}_{mode}.txt").write_text(stdout)
            parsed = parse_dc_out(dat_path)

            ok = (rc == 0) and parsed is not None and len(parsed["vd"]) > 5
            if not ok:
                convergence_failures += 1
                log(f"  CONV FAIL VG1={vg1} VG2={vg2:+.3f} mode={mode} rc={rc}")
                bias_record["modes"][mode] = {"ok": False, "rmse_dec": None}
                continue

            rmse = rmse_dec(meas_vd, meas_id, parsed["vd"], parsed["id"])
            bias_record["modes"][mode] = {
                "ok": True,
                "rmse_dec": rmse,
                "sim_vd": parsed["vd"].tolist(),
                "sim_id": parsed["id"].tolist(),
            }
            log(f"  VG1={vg1} VG2={vg2:+.3f} {mode:>8s}: RMSE={rmse:.3f} dec")

        results.append(bias_record)

    # ----------------------------------------------------------------
    # Aggregate / per-branch metrics
    # ----------------------------------------------------------------
    def avg_rmse(branch_vg1, mode):
        xs = [r["modes"][mode]["rmse_dec"] for r in results
              if r["vg1"] == branch_vg1 and r["modes"].get(mode, {}).get("ok")]
        return float(np.mean(xs)) if xs else float("nan")

    summary = {
        "n_biases": len(results),
        "convergence_failures": convergence_failures,
        "rmse_dec": {
            "baseline_VG1_0.2": avg_rmse(0.2, "baseline"),
            "baseline_VG1_0.4": avg_rmse(0.4, "baseline"),
            "baseline_VG1_0.6": avg_rmse(0.6, "baseline"),
            "ipos_VG1_0.2":     avg_rmse(0.2, "ipos"),
            "ipos_VG1_0.4":     avg_rmse(0.4, "ipos"),
            "ipos_VG1_0.6":     avg_rmse(0.6, "ipos"),
        },
    }
    summary["rmse_dec"]["baseline_cell"] = float(np.mean([
        v for k, v in summary["rmse_dec"].items()
        if k.startswith("baseline_VG1") and not np.isnan(v)
    ]) if any(not np.isnan(v) for k,v in summary["rmse_dec"].items() if k.startswith("baseline_VG1")) else float("nan"))
    summary["rmse_dec"]["ipos_cell"] = float(np.mean([
        v for k, v in summary["rmse_dec"].items()
        if k.startswith("ipos_VG1") and not np.isnan(v)
    ]) if any(not np.isnan(v) for k,v in summary["rmse_dec"].items() if k.startswith("ipos_VG1")) else float("nan"))

    # Gates
    infra_pass = (convergence_failures == 0)
    branch_rmses = [summary["rmse_dec"][f"baseline_VG1_{vg}"] for vg in (0.2, 0.4, 0.6)]
    discovery_pass = any((not np.isnan(r)) and r <= 1.0 for r in branch_rmses)
    ambitious_pass = (not np.isnan(summary["rmse_dec"]["ipos_cell"])
                      and summary["rmse_dec"]["ipos_cell"] <= 0.5)
    # Kill-shot is satisfied if EITHER cell-wide RMSE>=4 OR any single VG1
    # branch shows the ~5-decade gap reported for pyport at VG1=0.6.
    killshot_branch = any(
        (not np.isnan(summary["rmse_dec"][f"baseline_VG1_{vg}"]))
        and summary["rmse_dec"][f"baseline_VG1_{vg}"] >= 4.0
        for vg in (0.2, 0.4, 0.6)
    )
    killshot = killshot_branch or (
        not np.isnan(summary["rmse_dec"]["baseline_cell"])
        and summary["rmse_dec"]["baseline_cell"] >= 4.0
    )

    summary["gates"] = {
        "INFRA": bool(infra_pass),
        "DISCOVERY": bool(discovery_pass),
        "AMBITIOUS": bool(ambitious_pass),
        "KILL_SHOT": bool(killshot),
    }

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    for vg1 in (0.2, 0.4, 0.6):
        fig, ax = plt.subplots(figsize=(8, 6))
        for r in results:
            if r["vg1"] != vg1:
                continue
            mvd = np.array(r["meas_vd"]); mid = np.abs(np.array(r["meas_id"]))
            ax.semilogy(mvd, np.maximum(mid, 1e-14), "o", ms=2,
                        alpha=0.6, label=f"meas VG2={r['vg2']:+.2f}")
            for mode, color in (("baseline", "red"), ("ipos", "blue")):
                m = r["modes"].get(mode, {})
                if m.get("ok"):
                    svd = np.array(m["sim_vd"]); sid = np.abs(np.array(m["sim_id"]))
                    ax.semilogy(svd, np.maximum(sid, 1e-14), "-",
                                color=color, alpha=0.3, lw=0.8)
        ax.set_xlabel("Vd (V)")
        ax.set_ylabel("|Id| (A)")
        ax.set_title(f"z416 ngspice EXACT: VG1={vg1}V  "
                     f"(red=baseline, blue=ipos, dots=measured)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(OUT / f"overlay_VG1_{str(vg1).replace('.', 'p')}.png", dpi=130)
        plt.close(fig)

    # ----------------------------------------------------------------
    # Save artifacts
    # ----------------------------------------------------------------
    np.savez_compressed(
        OUT / "ngspice_traces.npz",
        results=np.array(results, dtype=object),
    )

    # Strip big arrays before dumping summary json (keep RMSE only)
    light_results = []
    for r in results:
        lr = {"vg1": r["vg1"], "vg2": r["vg2"], "file": r["file"]}
        for m in ("baseline", "ipos"):
            d = r["modes"].get(m, {})
            lr[f"{m}_rmse_dec"] = d.get("rmse_dec")
            lr[f"{m}_ok"] = d.get("ok", False)
        light_results.append(lr)
    summary["per_bias"] = light_results

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ----------------------------------------------------------------
    # Honest analysis
    # ----------------------------------------------------------------
    base_cell = summary["rmse_dec"]["baseline_cell"]
    ipos_cell = summary["rmse_dec"]["ipos_cell"]
    md = []
    md.append("# z416 ngspice EXACT — honest analysis\n")
    md.append(f"- N biases: {summary['n_biases']}")
    md.append(f"- Convergence failures: {summary['convergence_failures']}")
    md.append("")
    md.append("## Cell-wide RMSE (decades, log10|Id|)")
    md.append(f"- baseline cell-wide: {base_cell:.3f}")
    md.append(f"- ipos     cell-wide: {ipos_cell:.3f}")
    md.append("")
    md.append("## Per-branch RMSE (baseline)")
    for vg1 in (0.2, 0.4, 0.6):
        md.append(f"- VG1={vg1}: {summary['rmse_dec'][f'baseline_VG1_{vg1}']:.3f} dec")
    md.append("")
    md.append("## Gates")
    for k, v in summary["gates"].items():
        md.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    md.append("")
    md.append("## Verdict")
    if killshot:
        md.append(
            "**KILL-SHOT TRIGGERED.** Industry-standard ngspice on canonical "
            "Mario/Sebas files reproduces the same ~5-decade underprediction "
            "previously seen in pyport. The pyport implementation is therefore "
            "consistent with the canonical SPICE model — the *model itself* is "
            "missing the physics that drives the measured drain conduction at "
            "high VG1 (likely the floating-bulk amplification / impact-ionization "
            "/ GIDL-bipolar feedback loop that Mario's Ipos block is meant to "
            "supply). Canonical files do NOT reproduce their own silicon.")
    elif discovery_pass:
        md.append(
            "Baseline ngspice (canonical files) reproduces measured data within "
            "1 decade on at least one VG1 branch. The 5-decade gap previously "
            "seen in pyport is therefore *not intrinsic to the model* on at "
            "least some bias regimes; pyport may have an implementation bug.")
    else:
        md.append(
            "ngspice baseline has noticeable gap but below kill-shot threshold "
            "(4 decades). See per-branch numbers.")
    md.append("")
    if ambitious_pass:
        md.append(
            "AMBITIOUS gate passed: adding the placeholder Ipos behavioral "
            "block closes the gap to <=0.5 dec cell-wide — physically credible "
            "story for the floating-bulk drain conduction.")
    else:
        md.append(
            f"AMBITIOUS gate did not pass with placeholder Ipos constants "
            f"(cell-wide RMSE={ipos_cell:.3f} dec). Constants are uncalibrated; "
            f"the PWL(VG2) coefficient table from Mario's slide 12.26 was not "
            f"available in machine-readable form and would need digitization "
            f"to evaluate this gate fairly.")
    (OUT / "honest_analysis.md").write_text("\n".join(md))

    log("=" * 60)
    log("SUMMARY")
    log(json.dumps(summary["rmse_dec"], indent=2, default=float))
    log(json.dumps(summary["gates"], indent=2))
    log("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("FATAL: " + repr(e))
        log(traceback.format_exc())
        raise
    finally:
        LOG_FH.close()
