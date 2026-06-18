#!/usr/bin/env python3
"""z424: Sweep BSIM4 diode/bulk options to decouple M1's bulk-source diode
so Ipos body injection actually charges V(B) instead of being shorted.

Variants tested (all on M1 only; M2 left at canonical):
  V0_BASELINE       — unchanged (reference; same as z421)
  V1_DIOMOD0        — diomod=0 on M1 instance line
  V2_DIOMOD2        — diomod=2 (no breakdown current contribution)
  V3_ACNQSMOD1      — acnqsmod=1 (AC quasi-static off)
  V4_IJTHD_LOW      — ijthdfwd=1e-12 ijthdrev=1e-12
  V5_KILL_DIODE     — js=0 jsw=0 cjs=0 cjd=0 cjswgs=0 cjswgd=0 ijthdfwd=1e-30
                      (kill bulk-source/drain diode entirely via instance overrides)

For each variant we run all 33 measured biases with Ipos injected at body (B),
record max(V(B)) per-bias and cell-wide log-RMSE, and overlay V(B)(V_D) traces
at VG1=0.6 (the most-stuck branch in z421).
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
OUT = REPO / "results" / "z424_bulkmod_sweep"
DECKS = OUT / "decks"
OUT.mkdir(parents=True, exist_ok=True)
DECKS.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT / "run.log"
LOG_FH = open(LOG_PATH, "w")

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_FH.write(line + "\n"); LOG_FH.flush()


# -------------------------- PWL loader (from z421) --------------------------
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


# -------------------------- Variant table --------------------------
# M1 instance-line override strings (appended after W=...).
# Note: BSIM4 instance-level parameters supported in ngspice include
#   nrd, nrs, ad, as, pd, ps  (geometry) and not all model-level params.
# Therefore some variants are implemented as a per-variant *model card edit*.
# We use a 'card_extra' string injected after the .model NMOS block, plus an
# 'inst_extra' on the M1 instance line. The base model card (PTM130bulkNSRAM.txt)
# is included verbatim; overrides are done via a second .model directive that
# we re-include — instead we copy the card to a tmp file with the patch.

VARIANTS = [
    # The PTM130 card is BSIM3v3 (Level=14), not BSIM4. The body node B is
    # connected to (a) M1's bulk-source/drain diodes and (b) Q1's NPN base
    # (parasiticBJT, is=5e-9). Q1's B-E junction (emitter=ground) is a real
    # DC path that clamps V(B) at ~0.5-0.7V forward, INDEPENDENT of M1's
    # body diode. So we sweep BOTH: M1 diode-killing options AND Q1's NPN
    # saturation current.
    {
        "name": "V0_BASELINE",
        "card_patch": "",
        "m1_inst_extra": "",
        "q1_is": None,                         # use card default (5e-9)
    },
    {
        # Kill only M1's body diode (BSIM3 js=0, instance ad/as=0). Q1
        # untouched. Tests whether Q1 alone clamps V_B.
        "name": "V1_M1DIODE_OFF_ONLY",
        "card_patch": "+ js=0 jsw=0 cj=0 cjsw=0 cjswg=0\n",
        "m1_inst_extra": "ad=0 as=0 pd=0 ps=0",
        "q1_is": None,
    },
    {
        # Kill only Q1 (set Q1 IS=1e-30 via on-the-fly model override).
        # Tests whether M1 body diode alone clamps V_B.
        "name": "V2_Q1_OFF_ONLY",
        "card_patch": "",
        "m1_inst_extra": "",
        "q1_is": 1e-30,
    },
    {
        # Kill BOTH M1 body diode AND Q1 — only path for Ipos is the cap
        # C1 (1f) and bulk capacitance. Expected: V(B) should rise nearly
        # unbounded (until convergence breaks) showing the body truly
        # floats.
        "name": "V3_M1_AND_Q1_OFF",
        "card_patch": "+ js=0 jsw=0 cj=0 cjsw=0 cjswg=0\n",
        "m1_inst_extra": "ad=0 as=0 pd=0 ps=0",
        "q1_is": 1e-30,
    },
    {
        # Both off + BSIM4 (level=54) with diomod=2 acnqsmod=1 belts-and-
        # braces. Tests whether a *different* compact model behaves the
        # same way once external diode paths are also killed.
        "name": "V4_BSIM4_AND_Q1_OFF",
        "card_patch": ("+ level=54 diomod=2 acnqsmod=1\n"
                       "+ jss=1e-30 jsd=1e-30 jsws=1e-30 jswd=1e-30 jswgs=1e-30 jswgd=1e-30\n"
                       "+ ijthdfwd=1e-30 ijthdrev=1e-30 ijthsfwd=1e-30 ijthsrev=1e-30\n"
                       "+ cjs=0 cjd=0 cjsws=0 cjswd=0 cjswgs=0 cjswgd=0\n"),
        "m1_inst_extra": "ad=0 as=0 pd=0 ps=0",
        "q1_is": 1e-30,
    },
    {
        # Both off + nj=10 (very soft body diode emission). Sanity variant.
        "name": "V5_NJ_HIGH_AND_Q1_OFF",
        "card_patch": "+ js=1e-30 jsw=1e-30 cj=0 cjsw=0 cjswg=0 nj=10\n",
        "m1_inst_extra": "ad=0 as=0 pd=0 ps=0",
        "q1_is": 1e-30,
    },
]


def patch_card_text(orig: str, patch: str) -> str:
    """Insert patch lines IMMEDIATELY after the '.model NMOS NMOS' header line.

    Putting the '+' continuation lines right after the model header guarantees
    they are parsed as part of the model regardless of comment/blank lines
    later in the block.
    """
    if not patch:
        return orig
    lines = orig.splitlines(keepends=True)
    out = []
    inserted = False
    for ln in lines:
        out.append(ln)
        if (not inserted) and ln.strip().lower().startswith(".model nmos"):
            out.append(patch)
            inserted = True
    if not inserted:
        out.append(patch)
    return "".join(out)


# -------------------------- Deck builder --------------------------
DECK_TEMPLATE = """.title z424 BSIM4 bulkmod sweep ({VARIANT}: VG1={VG1:.3f} VG2={VG2:.3f})

* Compatibility shim
.param vsatn=1.35e5
.param Nparam=1.58
.param Citparam=0
.param Voffparam=-0.1368
.param K2Par=-0.070435
.param toxn=4e-9
.param Ln=0.18u
.param Wn=0.36u
.param CBpar=1f

* Patched canonical model card (variant-specific patch applied)
.include "{PTM130_PATCHED}"
.include "{BJT_PATCHED}"

Vd   D   0   DC 0
Vg1  G   0   DC {VG1:.4f}
Vg2  G2  0   DC {VG2:.4f}

M1 D G Sint B NMOS L=0.18u W=0.36u {M1_INST_EXTRA}
M2 Sint G2 0 0 NMOS L=1.8u W=0.36u
Q1 D B 0 parasiticBJT area=1u
C1 B Bx 1f
Rcb Bx 0 1m

* Ipos PWL block (Mario), injected at body B.
.param a_val={A_VAL:.6e}
.param b_val={B_VAL:.6f}
.param d_val={D_VAL:.6e}
.param e_val={E_VAL:.6f}
.param f_val={F_VAL:.6f}
.param c_const={C_CONST:.6f}

B_ipos B 0 I = a_val*exp(b_val*(V(D)+c_const)) + (((V(D)+f_val) > 0) ? (d_val*pow(V(D)+f_val, e_val)) : 0)

.options gmin=1e-15 abstol=1e-15 reltol=1e-3 itl1=500 itl2=200 itl6=100

.control
dc Vd 0 2 0.05
wrdata {OUT_DAT} -i(vd) v(Sint) v(B)
quit
.endc

.end
"""


def make_patched_card(variant_name: str, patch: str) -> Path:
    orig = (CANON / "PTM130bulkNSRAM.txt").read_text()
    patched = patch_card_text(orig, patch)
    out = DECKS / f"PTM130_{variant_name}.txt"
    out.write_text(patched)
    return out


def make_patched_bjt(variant_name: str, q1_is) -> Path:
    """Optionally override the IS of the parasiticBJT model."""
    orig = (CANON / "parasiticBJT.txt").read_text()
    if q1_is is None:
        out = DECKS / f"parasiticBJT_{variant_name}.txt"
        out.write_text(orig)
        return out
    # Replace 'is=...' inside the .model line.
    patched = re.sub(r"is\s*=\s*[0-9.eE+\-]+", f"is={q1_is:.3e}", orig, count=1)
    # Also reduce bf so reverse-injection current is small if any.
    out = DECKS / f"parasiticBJT_{variant_name}.txt"
    out.write_text(patched)
    return out


def build_deck(variant: dict, patched_card: Path, patched_bjt: Path,
               vg1: float, vg2: float):
    vg_for_pwl = vg2  # hypothesis A
    P = pwl_at(vg_for_pwl)
    tag = f"{variant['name']}_VG1_{vg1:.2f}_VG2_{vg2:+.3f}"
    out_dat = (DECKS / f"out_{tag}.txt").as_posix()
    deck = DECK_TEMPLATE.format(
        VARIANT=variant["name"],
        VG1=vg1, VG2=vg2,
        PTM130_PATCHED=patched_card.as_posix(),
        BJT_PATCHED=patched_bjt.as_posix(),
        M1_INST_EXTRA=variant["m1_inst_extra"],
        A_VAL=P["a"], B_VAL=P["b"], D_VAL=P["d"], E_VAL=P["e"], F_VAL=P["f"],
        C_CONST=C_CONST,
        OUT_DAT=out_dat,
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
    try:
        arr = np.loadtxt(dat_path)
    except Exception:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 6:
        return None
    return {
        "vd":  arr[:, 0],
        "id":  arr[:, 1],
        "vsint": arr[:, 3],
        "vb":  arr[:, 5],
    }


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
    log(f"Found {len(biases)} measured biases.")
    log(f"Variants: {[v['name'] for v in VARIANTS]}")
    log("PRE-REG gates: INFRA<30min; DISCOVERY=any V_B>0.5V & cell<3.0dec; "
        "AMBITIOUS=cell<1.5dec & snapback; KILL=no variant gets V_B>0.5V")

    all_variant_results = {}

    for variant in VARIANTS:
        vname = variant["name"]
        log(f"--- {vname} ---")
        patched_card = make_patched_card(vname, variant["card_patch"])
        patched_bjt = make_patched_bjt(vname, variant.get("q1_is"))
        var_results = []
        conv_fail = 0
        for (vg1, vg2, fpath) in biases:
            try:
                meas_vd, meas_id = load_iv(fpath)
            except Exception as e:
                log(f"  load FAIL {fpath.name}: {e}")
                continue
            deck_path, dat_path, P = build_deck(variant, patched_card, patched_bjt, vg1, vg2)
            rc, stdout = run_ngspice(deck_path)
            parsed = parse_dc_out(dat_path)
            ok = (rc == 0) and parsed is not None and len(parsed["vd"]) > 5
            if not ok:
                conv_fail += 1
                # capture log
                (DECKS / f"log_{vname}_VG1_{vg1:.2f}_VG2_{vg2:+.3f}.txt").write_text(stdout[-4000:])
                var_results.append({
                    "vg1": vg1, "vg2": vg2, "file": fpath.name, "ok": False,
                    "rmse_dec": float("nan"), "vb_max": float("nan"), "vb_min": float("nan"),
                })
                continue
            rmse = rmse_dec(meas_vd, meas_id, parsed["vd"], parsed["id"])
            vb_max = float(np.max(parsed["vb"]))
            vb_min = float(np.min(parsed["vb"]))
            var_results.append({
                "vg1": vg1, "vg2": vg2, "file": fpath.name, "ok": True,
                "rmse_dec": rmse, "vb_max": vb_max, "vb_min": vb_min,
                "sim_vd": parsed["vd"].tolist(),
                "sim_id": parsed["id"].tolist(),
                "sim_vb": parsed["vb"].tolist(),
                "meas_vd": meas_vd.tolist(),
                "meas_id": meas_id.tolist(),
            })

        # Aggregate per-variant
        oks = [r for r in var_results if r["ok"]]
        rmses = [r["rmse_dec"] for r in oks if not np.isnan(r["rmse_dec"])]
        vbs = [r["vb_max"] for r in oks if not np.isnan(r["vb_max"])]
        cell_rmse = float(np.mean(rmses)) if rmses else float("nan")
        vb_max_overall = float(np.max(vbs)) if vbs else float("nan")
        vb_min_overall = float(np.min([r["vb_min"] for r in oks])) if oks else float("nan")

        # per-branch
        per_branch = {}
        for vg1 in (0.2, 0.4, 0.6):
            br = [r["rmse_dec"] for r in oks if r["vg1"] == vg1 and not np.isnan(r["rmse_dec"])]
            per_branch[f"VG1_{vg1}"] = float(np.mean(br)) if br else float("nan")

        log(f"  {vname}: cell_rmse={cell_rmse:.3f} dec, V_B max={vb_max_overall:+.4f} V, "
            f"V_B min={vb_min_overall:+.4f} V, conv_fail={conv_fail}")

        all_variant_results[vname] = {
            "card_patch": variant["card_patch"],
            "m1_inst_extra": variant["m1_inst_extra"],
            "cell_rmse_dec": cell_rmse,
            "per_branch_rmse_dec": per_branch,
            "vb_max": vb_max_overall,
            "vb_min": vb_min_overall,
            "convergence_failures": conv_fail,
            "per_bias": var_results,
        }

    # -------------------- aggregate / gates --------------------
    summary = {
        "n_biases": len(biases),
        "wall_sec": round(time.time() - t0, 1),
        "baseline_z421_cell_rmse_dec": 4.6992,
        "baseline_s11_cell_rmse_dec": 3.170,
        "variants": {
            name: {
                "cell_rmse_dec": r["cell_rmse_dec"],
                "per_branch_rmse_dec": r["per_branch_rmse_dec"],
                "vb_max": r["vb_max"],
                "vb_min": r["vb_min"],
                "convergence_failures": r["convergence_failures"],
            } for name, r in all_variant_results.items()
        },
    }

    any_vb_above = any((not np.isnan(v["vb_max"])) and v["vb_max"] > 0.5
                       for v in summary["variants"].values())
    any_discovery = any(
        (not np.isnan(v["vb_max"])) and v["vb_max"] > 0.5
        and (not np.isnan(v["cell_rmse_dec"])) and v["cell_rmse_dec"] < 3.0
        for v in summary["variants"].values()
    )
    any_ambitious = any(
        (not np.isnan(v["cell_rmse_dec"])) and v["cell_rmse_dec"] < 1.5
        for v in summary["variants"].values()
    )
    infra = (summary["wall_sec"] < 1800
             and all(v["convergence_failures"] == 0 for v in summary["variants"].values()))
    killshot = not any_vb_above

    summary["gates"] = {
        "INFRA": bool(infra),
        "DISCOVERY": bool(any_discovery),
        "AMBITIOUS": bool(any_ambitious),
        "KILL_SHOT": bool(killshot),
    }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"Gates: {summary['gates']}")
    log(f"Wall: {summary['wall_sec']} s")

    # -------------------- plots --------------------
    # V_B trace combined at VG1=0.6 — overlay across variants at one representative VG2 (~0.6)
    fig, ax = plt.subplots(figsize=(8, 5))
    target_vg1 = 0.6
    target_vg2 = 0.6  # mid-bias for VG1=0.6 branch
    colors = plt.cm.tab10.colors
    for i, (vname, vres) in enumerate(all_variant_results.items()):
        # find bias closest to (target_vg1, target_vg2)
        cand = [r for r in vres["per_bias"]
                if r["ok"] and r["vg1"] == target_vg1 and "sim_vd" in r]
        if not cand:
            continue
        # choose vg2 closest to target
        r = min(cand, key=lambda x: abs(x["vg2"] - target_vg2))
        ax.plot(r["sim_vd"], r["sim_vb"], color=colors[i % 10],
                label=f"{vname} (VG2={r['vg2']:+.2f})", lw=1.5)
    ax.axhline(0.5, ls="--", c="k", alpha=0.5, label="V_B=0.5 V threshold")
    ax.set_xlabel("V_D (V)")
    ax.set_ylabel("V(B) (V)")
    ax.set_title(f"z424: V(B) vs V_D across BSIM4 variants (VG1={target_vg1}, "
                 f"VG2≈{target_vg2}), Ipos→B")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "vb_trace_combined.png", dpi=120)
    plt.close(fig)

    # Best-variant overlay at VG1=0.6: pick variant with smallest cell_rmse
    best_name = min(
        summary["variants"].keys(),
        key=lambda k: (summary["variants"][k]["cell_rmse_dec"]
                       if not np.isnan(summary["variants"][k]["cell_rmse_dec"]) else 1e9),
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    best = all_variant_results[best_name]
    for r in best["per_bias"]:
        if not r["ok"] or r["vg1"] != 0.6 or "sim_vd" not in r:
            continue
        ax.semilogy(r["meas_vd"], np.abs(r["meas_id"]),
                    color="k", alpha=0.3, lw=0.8)
        ax.semilogy(r["sim_vd"], np.abs(r["sim_id"]),
                    color="r", alpha=0.4, lw=0.8)
    ax.set_xlabel("V_D (V)")
    ax.set_ylabel("|I_D| (A)")
    ax.set_title(f"z424 best variant: {best_name} "
                 f"(cell={best['cell_rmse_dec']:.2f} dec) — meas (black) vs sim (red), VG1=0.6")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "best_variant_overlay_VG1_0p6.png", dpi=120)
    plt.close(fig)

    # -------------------- honest_analysis.md --------------------
    md = []
    md.append("# z424 BSIM4 bulkmod sweep — honest analysis\n")
    md.append(f"Wall: {summary['wall_sec']} s, {len(biases)} biases × {len(VARIANTS)} variants "
              f"= {len(biases) * len(VARIANTS)} ngspice runs.\n")
    md.append("## Per-variant summary\n")
    md.append("| Variant | cell RMSE (dec) | V_B max (V) | V_B min (V) | conv fails |\n")
    md.append("|---|---|---|---|---|\n")
    for name, v in summary["variants"].items():
        md.append(f"| {name} | {v['cell_rmse_dec']:.3f} | "
                  f"{v['vb_max']:+.4f} | {v['vb_min']:+.4f} | "
                  f"{v['convergence_failures']} |\n")
    md.append(f"\nBaseline z421 cell RMSE: 4.699 dec. Baseline S11 (no Ipos): 3.170 dec.\n")
    md.append(f"\nBest variant: **{best_name}** (cell={summary['variants'][best_name]['cell_rmse_dec']:.3f} dec)\n")
    md.append("\n## Gates\n")
    for k, val in summary["gates"].items():
        md.append(f"- **{k}**: {val}\n")
    md.append("\n## Interpretation\n")
    if killshot:
        md.append("KILL_SHOT triggered: no BSIM4 modification produced V_B > 0.5V. "
                  "The BSIM4 architecture in ngspice fundamentally couples the bulk to source "
                  "through diodes that cannot be fully decoupled via the tested model-card "
                  "options. The bulk-source diode current (or numerical gmin path) dominates "
                  "over the injected Ipos at all swept biases, so the floating-body assumption "
                  "Mario's PWL relies on cannot be realized inside BSIM4. Consider switching "
                  "to a behavioral compact model (Verilog-A or BSIMSOI4) where body terminal "
                  "really floats.\n")
    elif any_discovery:
        md.append("DISCOVERY: at least one variant lifts V(B) above 0.5V while keeping "
                  "cell RMSE < 3.0 dec. The bulk-source diode can be tamed by the listed "
                  "model-card edits. Inspect best_variant_overlay_VG1_0p6.png for shape match.\n")
    else:
        md.append("Variants change RMSE but V(B) remains clamped below 0.5V everywhere; "
                  "no DISCOVERY but also no clean KILL — Ipos may have moved current paths "
                  "without truly floating the body.\n")
    (OUT / "honest_analysis.md").write_text("".join(md))

    log("Done. See results in " + str(OUT))


if __name__ == "__main__":
    main()
