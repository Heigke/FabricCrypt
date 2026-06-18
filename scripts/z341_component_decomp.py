"""z341 — R-25: per-term component decomposition at VG1=0.60, VG2=0.20.

For each Vd point (10 points along ngspice OP states), force the node
voltages V(vsint), V(vb) via voltage sources so ngspice does NOT solve.
Read every device-level current (@m1[id], @m1[isub], @m1[igidl],
@m1[ibd], @q1[ic]) and compare to pyport's components dict computed at
the exact same (Vsint, Vb).

This isolates each term and tells us which path is wrong by how many
decades, at the regime (VG1=0.60, highest pyport error = 5.48 dec).
"""
from __future__ import annotations
import os, sys, json, re, subprocess, time, math
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/z341_component_decomp"
OUT.mkdir(parents=True, exist_ok=True)

VG1_TARGET = 0.60
VG2_TARGET = 0.20

# z338 best params (must match z340)
Z338_BEST = dict(
    alpha0 = 1.63357328192734e-05,
    Bf     = 2605.2882016162002,
    Va     = 0.3567358318716285,
    Is     = 3.2906845928467974e-10,
    lat_BV = 4.018266147002578,
    body_pdiode_Rs = 5480383.486345367,
)

M1_CARD = ROOT / "data/sebas_2026_04_22/M1_130DNWFB.txt"
M2_CARD = ROOT / "data/sebas_2026_04_22/M2_130bulkNSRAM.txt"


# ---------------------------------------------------------------- pick points
def pick_points():
    """Return list of (Vd, Vsint, Vb, Id_meas) using z340 OP states for
    VG1=0.60, VG2=0.20, plus measured Id from the matching csv."""
    states = json.load(open(ROOT / "results/z340_ngspice_handover/per_bias_states.json"))
    target = None
    for c in states:
        if abs(c["VG1"] - VG1_TARGET) < 0.01 and abs(c["VG2"] - VG2_TARGET) < 0.01:
            target = c; break
    assert target is not None, "VG1=0.60, VG2=0.20 not in per_bias_states"
    arr = np.array(target["states"], dtype=float)
    # Forward sweep only (first half)
    half = len(arr)//2 + 1
    fwd = arr[:half]
    idx = np.linspace(0, len(fwd)-1, 10).astype(int)
    sub = fwd[idx]
    # Load measured Id
    csv_dir = ROOT / f"data/sebas_2026_04_22/2vHCa-2 I-Vs@VG2 VG1={VG1_TARGET} vnwell=2"
    csv_files = list(csv_dir.glob(f"*VG2={VG2_TARGET:.2f}_VG=*.csv"))
    if not csv_files:
        # try without sign formatting
        csv_files = list(csv_dir.glob(f"*VG2=0.20*.csv"))
    assert csv_files, f"no csv match in {csv_dir}"
    data = np.loadtxt(csv_files[0], delimiter=",", skiprows=1)
    Vd_meas = data[:, 0].astype(float)
    Id_meas = np.abs(data[:, 1]).astype(float)
    # For each picked Vd, interpolate measured Id at that Vd
    Id_at = []
    for (vd, vs, vb) in sub:
        # nearest neighbour on Vd
        j = int(np.argmin(np.abs(Vd_meas - vd)))
        Id_at.append(float(Id_meas[j]))
    return [(float(r[0]), float(r[1]), float(r[2]), Id_at[i]) for i, r in enumerate(sub)]


# ---------------------------------------------------------------- ngspice deck
def make_deck(points):
    """Single deck: voltage sources force every node. One .op per point via
    alter. Each iteration prints all device currents."""
    cmds = []
    for vd, vs, vb in [(p[0], p[1], p[2]) for p in points]:
        cmds.append(f"alter Vdd dc = {max(vd,1e-6)}")
        cmds.append(f"alter Vsint_src dc = {max(vs,1e-9)}")
        cmds.append(f"alter Vb_src dc = {max(vb,1e-9)}")
        cmds.append("op")
        cmds.append(f"echo \"POINT Vd={vd:.8e} Vsint={vs:.8e} Vb={vb:.8e}\"")
        cmds.append("print v(vd) v(vsint) v(vb)")
        cmds.append("print @m1[id] @m1[ibd] @m1[ibs] @m1[isub] @m1[igidl] @m1[igisl] @m1[igb]")
        cmds.append("print @m2[id] @m2[ibd] @m2[ibs] @m2[isub] @m2[igidl] @m2[igisl] @m2[igb]")
        cmds.append("print @q1[ic] @q1[ib] @q1[ie]")
    ctrl = "\n".join(cmds)
    return f""".title z341 forced-node component decomp VG1={VG1_TARGET} VG2={VG2_TARGET}

.include "{M1_CARD}"
.include "{M2_CARD}"

.model parasiticBJT NPN(is={Z338_BEST['Is']:.4e} va={Z338_BEST['Va']:.4e}
+ bf={Z338_BEST['Bf']:.4e} br=100 nc=2 ikr=100m rc=0.1
+ vje=0.7 re=0.1 cjc=1e-15 fc=0.5 cje=0.7e-15 ne=1.5 ise=0 tr=20e-12 tf=25e-12
+ itf=0.03 vtf=7 xtf=2)

Vdd       vd     0    DC 0.001
Vg1       vg1    0    DC {VG1_TARGET}
Vg2       vg2    0    DC {VG2_TARGET}
Vsint_src vsint  0    DC 0.001
Vb_src    vb     0    DC 0.001

M1  vd vg1 vsint vb NMOSdnwfb L=0.13u W=1u
M2  vsint vg2 0 0 NMOS L=0.234u W=1u
Q1  vsint vb 0 parasiticBJT area=1u

.options gmin=1e-15 abstol=1e-12 reltol=1e-3 itl1=500

.control
{ctrl}
quit
.endc

.end
"""


_re_eq = re.compile(r"(@?\w+(?:\[\w+\])?|v\(\w+\)|i\(\w+\))\s*=\s*([-+]?\d+\.?\d*[eE]?[-+]?\d*)")

def parse_log_blocks(text):
    """Split by POINT marker; per block return dict of parsed currents."""
    blocks = re.split(r"POINT Vd=", text)
    out = []
    for blk in blocks[1:]:
        d = {}
        for m in _re_eq.finditer(blk):
            d[m.group(1).lower()] = float(m.group(2))
        # also extract the marker's Vd
        mvd = re.match(r"([-+]?\d+\.?\d*[eE]?[-+]?\d*)", blk)
        if mvd:
            d["_Vd_marker"] = float(mvd.group(1))
        out.append(d)
    return out


# ---------------------------------------------------------------- pyport
def build_pyport():
    import importlib.util, torch
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    cfg.eta_sigmoid = False  # z338 default
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    sd_M1 = cfg.size_dep_M1(M1)
    sd_M2 = cfg.size_dep_M2(M2)
    sd_M1.scaled["alpha0"] = Z338_BEST["alpha0"]
    sd_M2.scaled["alpha0"] = Z338_BEST["alpha0"]
    bjt.Bf = Z338_BEST["Bf"]
    bjt.Va = Z338_BEST["Va"]
    bjt.Is = Z338_BEST["Is"]
    cfg.lat_BV = Z338_BEST["lat_BV"]
    cfg.body_pdiode_Rs = Z338_BEST["body_pdiode_Rs"]
    return cfg, M1, M2, bjt


def pyport_components_at(cfg, M1, M2, bjt, Vd, Vsint, Vb):
    import torch
    from nsram.bsim4_port.nsram_cell_2T import _residuals
    Vd_t = torch.tensor(Vd, dtype=torch.float64)
    VG1 = torch.tensor(VG1_TARGET, dtype=torch.float64)
    VG2 = torch.tensor(VG2_TARGET, dtype=torch.float64)
    Vs = torch.tensor(Vsint, dtype=torch.float64)
    Vb_t = torch.tensor(Vb, dtype=torch.float64)
    R_S, R_B, comp = _residuals(cfg, M1, bjt, Vd_t, VG1, VG2, Vs, Vb_t, model_M2=M2)
    out = {}
    for k, v in comp.items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


# Pyport Id assembly (matches forward_2t)
def pyport_id(comp):
    Id = (comp.get("Ids_M1", 0.0)
          + comp.get("Ic_Q1", 0.0)
          + comp.get("Ic_lat", 0.0)
          + comp.get("Ic_avalanche", 0.0)
          + comp.get("Igidl_M1", 0.0)
          - comp.get("Ibd_M1", 0.0))
    return Id


# ngspice Id reconstruction at drain: sum currents flowing INTO drain.
# @m1[id] = M1 drain terminal current INTO M1 from external
# Ic_Q1 = collector=Sint, NOT drain — so does not appear at drain directly
# Igidl_M1 enters body from drain in ngspice ⇒ -Igidl at drain externally
# Convention check: we'll just take @m1[id] as the dominant "Id" since
# all D-terminal currents in BSIM4 internal lump into id.
def ngspice_total_id(blk):
    # In ngspice, @m1[id] is the net current at the drain terminal.
    # In z330 deck, this is the only D-pin device, so total drain
    # current equals @m1[id] (+ optional @q1[ic] if Q1 collector=drain;
    # here collector=vsint so no contribution).
    return abs(blk.get("@m1[id]", 0.0))


# ---------------------------------------------------------------- main
def main():
    import torch
    t0 = time.time()
    print(f"[z341] decomposing VG1={VG1_TARGET} VG2={VG2_TARGET}", flush=True)
    points = pick_points()
    print(f"[z341] 10 Vd points selected from forward sweep", flush=True)

    deck_path = OUT / "deck.sp"
    log_path = OUT / "ngspice.log"
    deck_path.write_text(make_deck(points))
    proc = subprocess.run(["ngspice", "-b", str(deck_path)],
                          capture_output=True, text=True, timeout=300)
    log = proc.stdout + "\n--- STDERR ---\n" + proc.stderr
    log_path.write_text(log)
    print(f"[z341] ngspice rc={proc.returncode}", flush=True)

    blocks = parse_log_blocks(log)
    print(f"[z341] parsed {len(blocks)}/{len(points)} OP blocks", flush=True)

    cfg, M1, M2, bjt = build_pyport()

    # Term map: pyport key -> ngspice key(s)
    TERMS = [
        ("Ids_M1",        ["@m1[id]"]),
        ("Ic_Q1",         ["@q1[ic]"]),
        ("Ic_lat",        []),                  # no direct ngspice analog (pyport-internal)
        ("Ic_avalanche",  []),                  # pyport-internal (M1 Vbc avalanche)
        ("Igidl_M1",      ["@m1[igidl]"]),
        ("Ibd_M1",        ["@m1[ibd]"]),
        ("Iii_M1",        ["@m1[isub]"]),
        ("Ibs_M1",        ["@m1[ibs]"]),
        ("Ib_Q1",         ["@q1[ib]"]),
    ]

    rows = []
    for (vd, vs, vb, id_meas), blk in zip(points, blocks):
        py = pyport_components_at(cfg, M1, M2, bjt, vd, vs, vb)
        id_py = pyport_id(py)
        id_ng = ngspice_total_id(blk)
        row = {
            "Vd": vd, "Vsint": vs, "Vb": vb,
            "Id_meas": id_meas,
            "Id_pyport_total": id_py,
            "Id_ngspice_at_m1d": id_ng,
            "log10_ratio_py_over_ng_TOTAL": (math.log10(abs(id_py)/id_ng) if id_ng > 0 and abs(id_py) > 0 else None),
            "log10_ratio_py_over_meas":     (math.log10(abs(id_py)/id_meas) if id_meas > 0 and abs(id_py) > 0 else None),
            "log10_ratio_ng_over_meas":     (math.log10(id_ng/id_meas) if id_meas > 0 and id_ng > 0 else None),
            "terms": {},
        }
        for pname, ngkeys in TERMS:
            pv = py.get(pname, 0.0)
            nv = sum(blk.get(k, 0.0) for k in ngkeys) if ngkeys else None
            entry = {"pyport": pv}
            if nv is not None:
                entry["ngspice"] = nv
                if abs(pv) > 1e-30 and abs(nv) > 1e-30:
                    entry["log10_ratio"] = math.log10(abs(pv) / abs(nv))
                    entry["sign_match"] = (pv * nv) > 0
                else:
                    entry["log10_ratio"] = None
                    entry["sign_match"] = None
            row["terms"][pname] = entry
        rows.append(row)

    # Identify worst per-term ratio across all Vd
    per_term_stats = {}
    for pname, ngkeys in TERMS:
        if not ngkeys:
            continue
        ratios = [r["terms"][pname].get("log10_ratio") for r in rows
                  if r["terms"][pname].get("log10_ratio") is not None]
        if ratios:
            per_term_stats[pname] = {
                "n": len(ratios),
                "median_log10_ratio": float(np.median(ratios)),
                "max_abs_log10_ratio": float(np.max(np.abs(ratios))),
                "min_log10_ratio": float(np.min(ratios)),
                "max_log10_ratio": float(np.max(ratios)),
                "growth_with_Vd": float(ratios[-1] - ratios[0]) if len(ratios) >= 2 else 0.0,
            }

    # Hypothetical fix: replace each pyport term with ngspice value, see
    # how much log10 RMSE on Id_total improves.
    Id_meas_arr = np.array([r["Id_meas"] for r in rows])
    Id_py_arr   = np.array([r["Id_pyport_total"] for r in rows])
    base_rmse = float(np.sqrt(np.mean((np.log10(np.abs(Id_py_arr)) - np.log10(Id_meas_arr))**2)))

    fix_gain = {}
    for pname, ngkeys in TERMS:
        if not ngkeys:
            continue
        Id_fixed = []
        for r, blk in zip(rows, blocks):
            py = pyport_components_at(cfg, M1, M2, bjt, r["Vd"], r["Vsint"], r["Vb"])
            # Override this pyport term with ngspice value
            nv = sum(blk.get(k, 0.0) for k in ngkeys)
            py_mod = dict(py); py_mod[pname] = nv
            Id_fixed.append(pyport_id(py_mod))
        Id_fixed = np.array(Id_fixed)
        try:
            new_rmse = float(np.sqrt(np.mean(
                (np.log10(np.abs(Id_fixed) + 1e-30) - np.log10(Id_meas_arr))**2)))
        except Exception:
            new_rmse = float("nan")
        fix_gain[pname] = {"new_rmse_dec": new_rmse,
                            "delta_dec": base_rmse - new_rmse}

    summary = {
        "vg1": VG1_TARGET, "vg2": VG2_TARGET,
        "z338_best": Z338_BEST,
        "n_points": len(rows),
        "base_log10_rmse_dec": base_rmse,
        "per_term_stats": per_term_stats,
        "hypothetical_fix_dec_gain": fix_gain,
        "rows": rows,
    }

    (OUT / "per_term_diff.json").write_text(json.dumps(summary, indent=2))

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        plot_terms = ["Ids_M1", "Ic_Q1", "Igidl_M1", "Ibd_M1", "Iii_M1", "Ib_Q1"]
        Vd_arr = np.array([r["Vd"] for r in rows])
        for ax, pname in zip(axes.flat, plot_terms):
            py_vals = np.array([abs(r["terms"][pname]["pyport"]) for r in rows]) + 1e-30
            ng_vals = np.array([abs(r["terms"][pname].get("ngspice", 0.0)) for r in rows]) + 1e-30
            ax.semilogy(Vd_arr, py_vals, "o-", label="pyport", color="C0")
            ax.semilogy(Vd_arr, ng_vals, "s--", label="ngspice", color="C3")
            ax.set_title(pname); ax.set_xlabel("Vd"); ax.set_ylabel("|I| [A]")
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle(f"z341 per-term: VG1={VG1_TARGET}, VG2={VG2_TARGET} (pyport solid, ngspice dashed)")
        fig.tight_layout()
        fig.savefig(OUT / f"decomp_plot_VG1_{VG1_TARGET:.2f}_VG2_{VG2_TARGET:.2f}.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"  plot fail: {e}")

    # --- Verdict ---
    sorted_terms = sorted(per_term_stats.items(),
                          key=lambda kv: kv[1]["max_abs_log10_ratio"], reverse=True)
    top3 = sorted_terms[:3]
    sorted_fix = sorted(fix_gain.items(), key=lambda kv: kv[1]["delta_dec"], reverse=True)
    best_fix = sorted_fix[0] if sorted_fix else None

    md = ["# z341 verdict — VG1=0.60, VG2=0.20 component decomposition\n"]
    md.append(f"Base pyport log10-RMSE vs measured: **{base_rmse:.3f} dec** "
              f"(z340 reported 5.48 at VG1=0.60 median across VG2)\n")
    md.append("## Top 3 worst terms (max |log10(pyport/ngspice)|)\n")
    for n, (term, st) in enumerate(top3, 1):
        md.append(f"{n}. **{term}** — max_abs={st['max_abs_log10_ratio']:.2f} dec, "
                  f"median={st['median_log10_ratio']:+.2f} dec, "
                  f"growth_low→high_Vd={st['growth_with_Vd']:+.2f} dec\n")
    md.append("\n## Hypothetical fix gains (replace pyport term with ngspice value)\n")
    md.append("| Term | new RMSE [dec] | Δ improvement [dec] |\n|---|---|---|\n")
    for term, fg in sorted_fix:
        md.append(f"| {term} | {fg['new_rmse_dec']:.3f} | {fg['delta_dec']:+.3f} |\n")
    md.append("\n## Pinpoint\n")
    if best_fix:
        md.append(f"Fix **{best_fix[0]}** first → recovers **{best_fix[1]['delta_dec']:+.3f} dec**.\n")
    md.append("\n## Raw per-Vd table\n")
    md.append("| Vd | Vsint | Vb | Id_meas | Id_py | Id_ng | log10(py/ng) | log10(py/meas) |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        md.append(f"| {r['Vd']:.3f} | {r['Vsint']:.3f} | {r['Vb']:.3f} | "
                  f"{r['Id_meas']:.3e} | {r['Id_pyport_total']:.3e} | "
                  f"{r['Id_ngspice_at_m1d']:.3e} | "
                  f"{(r['log10_ratio_py_over_ng_TOTAL'] if r['log10_ratio_py_over_ng_TOTAL'] is not None else float('nan')):+.2f} | "
                  f"{(r['log10_ratio_py_over_meas'] if r['log10_ratio_py_over_meas'] is not None else float('nan')):+.2f} |\n")
    (OUT / "verdict.md").write_text("".join(md))

    # Console summary
    print(f"\n===== z341 VERDICT =====")
    print(f"  Base RMSE: {base_rmse:.3f} dec ({len(rows)} pts at VG1={VG1_TARGET}, VG2={VG2_TARGET})")
    print(f"  Top 3 worst terms by max |log10 ratio|:")
    for n, (term, st) in enumerate(top3, 1):
        print(f"    {n}. {term:14s}  max={st['max_abs_log10_ratio']:.2f}  median={st['median_log10_ratio']:+.2f}  growth={st['growth_with_Vd']:+.2f}")
    print(f"  Best single-term fix:")
    for term, fg in sorted_fix[:3]:
        print(f"    {term:14s}  Δ={fg['delta_dec']:+.3f} dec  →  new RMSE={fg['new_rmse_dec']:.3f}")
    print(f"  elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
