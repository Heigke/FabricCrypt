"""z217: full 2T envelope validation across 33 measured (VG1,VG2) bias points.

For each (VG1, VG2):
  - sweep Vd 0.05..1.95 V at 0.1 V step (20 points)
  - run ngspice  (gmin=1e-15 reltol=1e-4 itl1=500 itl2=200)
  - run our port forward_2t(use_homotopy=True)  with default Sebas card
  - load Sebas measured CSV for overlay

Output: results/z217_2t_full_envelope/{comparison.png, summary.json, findings.md}
"""
from __future__ import annotations
import os
import re
import json
import subprocess
import tempfile
from pathlib import Path
from glob import glob

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT_DIR = REPO / "results" / "z217_2t_full_envelope"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NGSPICE = "/usr/bin/ngspice"
PTM = REPO / "data/sebas_2026_04_22/PTM130bulkNSRAM.txt"
BJT = REPO / "data/sebas_2026_04_22/parasiticBJT.txt"
DATA_ROOT = REPO / "data/sebas_2026_04_22"

# ngspice-normalized copies (`.param X val` → `.param X = val`)
def _normalize_for_ngspice(src: Path, dst: Path) -> None:
    txt = src.read_text()
    txt = re.sub(r"^(\s*\.param\s+\w+)\s+(?!=)(\S)",
                 r"\1 = \2", txt, flags=re.MULTILINE | re.IGNORECASE)
    inject = "\n.param vsatn = 80000\n"
    txt = inject + txt
    dst.write_text(txt)

PTM_NG = Path("/tmp/PTM130bulkNSRAM_z217.txt")
BJT_NG = Path("/tmp/parasiticBJT_z217.txt")
_normalize_for_ngspice(PTM, PTM_NG)
_normalize_for_ngspice(BJT, BJT_NG)

VD_START = 0.05
VD_STOP = 1.95
VD_STEP = 0.1
VD_LIST = np.round(np.arange(VD_START, VD_STOP + 1e-9, VD_STEP), 4)


def discover_bias_points():
    """Find all (VG1, VG2, csv_path) triples in the Sebas data dirs."""
    triples = []
    for vg1 in (0.2, 0.4, 0.6):
        d = DATA_ROOT / f"2vHCa-2 I-Vs@VG2 VG1={vg1} vnwell=2"
        for csv in sorted(glob(str(d / "*.csv"))):
            m = re.search(r"VG2=(-?\d+\.\d+)", os.path.basename(csv))
            if not m:
                continue
            vg2 = float(m.group(1))
            triples.append((vg1, vg2, csv))
    return triples


def make_netlist(vg1: float, vg2: float) -> str:
    return f"""* NS-RAM 2T cell ngspice
.include {PTM_NG}
.include {BJT_NG}

M1 D G1 Sint B NMOS l=0.18u w=0.36u
M2 Sint G2 0 B NMOS l=1.8u w=0.36u
Q1 D B Sint parasiticBJT area=1u
C1 B 0 1f

VD  D  0 DC 1.0
VG1 G1 0 DC {vg1}
VG2 G2 0 DC {vg2}

.options gmin=1e-15 abstol=1e-14 reltol=1e-4 itl1=500 itl2=200

.control
set wr_vecnames
set wr_singlescale
set filetype=ascii
dc VD {VD_START} {VD_STOP} {VD_STEP}
let id = -i(vd)
wrdata /tmp/ngspice_z217.csv v(d) v(sint) v(b) id
quit
.endc
.end
"""


def run_ngspice_one(vg1: float, vg2: float):
    nl = make_netlist(vg1, vg2)
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(nl)
        cir = f.name
    try:
        out_csv = "/tmp/ngspice_z217.csv"
        if os.path.exists(out_csv):
            os.remove(out_csv)
        res = subprocess.run([NGSPICE, "-b", cir], capture_output=True,
                             text=True, timeout=180)
        if not os.path.exists(out_csv):
            raise RuntimeError(f"ngspice failed VG1={vg1} VG2={vg2}\n{res.stdout}\n{res.stderr}")
        data = np.loadtxt(out_csv, skiprows=1)
        if data.ndim == 1:
            data = data[None, :]
        # cols: x(=VD swept), v(d), v(sint), v(b), id
        return data[:, 1:5]  # (N,4): Vd, Vsint, Vb, Id
    finally:
        os.unlink(cir)


def run_python(vg1: float, vg2: float, vd_array: np.ndarray) -> dict:
    from nsram.bsim4_port.model_card import BSIM4Model
    from nsram.bsim4_port.bjt import GummelPoonNPN
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t

    model = BSIM4Model.from_spice(PTM.read_text(), model_type="nmos")
    bjt = GummelPoonNPN.from_sebas_card()
    cfg = NSRAMCell2TConfig()  # defaults

    Vd_seq = torch.tensor(vd_array, dtype=torch.float64)
    out = forward_2t(cfg, model, bjt,
                     Vd_seq=Vd_seq,
                     VG1=torch.tensor(vg1, dtype=torch.float64),
                     VG2=torch.tensor(vg2, dtype=torch.float64),
                     use_homotopy=True)
    return {
        "Vd": vd_array,
        "Vsint": out["Vsint"].detach().numpy(),
        "Vb": out["Vb"].detach().numpy(),
        "Id": out["Id"].detach().numpy(),
        "converged": np.array(out["converged"], dtype=bool),
    }


def load_sebas(csv_path: str):
    """Returns (Vd, Id) arrays from Sebas measured CSV."""
    arr = np.genfromtxt(csv_path, delimiter=",", skip_header=1, usecols=(0, 1))
    return arr[:, 0], arr[:, 1]


# ---------- main ---------- #
def main():
    triples = discover_bias_points()
    print(f"Found {len(triples)} bias-curve files (expected 33)")
    assert len(triples) == 33, f"expected 33 got {len(triples)}"

    per_curve = []  # list of dicts
    for (vg1, vg2, csv) in triples:
        print(f"  VG1={vg1:+.2f} VG2={vg2:+.3f} ...", end="", flush=True)
        try:
            ng = run_ngspice_one(vg1, vg2)
            ng_conv = np.ones(ng.shape[0], dtype=bool)
        except Exception as e:
            print(f" NGSPICE FAIL: {e}")
            ng = np.full((len(VD_LIST), 4), np.nan)
            ng_conv = np.zeros(len(VD_LIST), dtype=bool)

        try:
            py = run_python(vg1, vg2, ng[:, 0] if not np.isnan(ng[0, 0]) else VD_LIST)
        except Exception as e:
            print(f" PORT FAIL: {e}")
            py = {"Vd": VD_LIST,
                  "Vsint": np.full(len(VD_LIST), np.nan),
                  "Vb": np.full(len(VD_LIST), np.nan),
                  "Id": np.full(len(VD_LIST), np.nan),
                  "converged": np.zeros(len(VD_LIST), dtype=bool)}

        # Sebas
        try:
            sb_vd, sb_id = load_sebas(csv)
        except Exception as e:
            print(f" SEBAS FAIL: {e}")
            sb_vd, sb_id = np.array([]), np.array([])

        per_curve.append({
            "vg1": vg1, "vg2": vg2, "csv": os.path.basename(csv),
            "Vd": ng[:, 0].tolist(),
            "ng_Id": ng[:, 3].tolist(),
            "ng_Vsint": ng[:, 1].tolist(),
            "ng_Vb": ng[:, 2].tolist(),
            "ng_converged": ng_conv.tolist(),
            "py_Id": py["Id"].tolist(),
            "py_Vsint": py["Vsint"].tolist(),
            "py_Vb": py["Vb"].tolist(),
            "py_converged": py["converged"].tolist(),
            "sb_Vd": sb_vd.tolist(),
            "sb_Id": sb_id.tolist(),
        })
        nconv = int(np.sum(py["converged"]))
        print(f" ng_id [{np.nanmin(ng[:,3]):.2e},{np.nanmax(ng[:,3]):.2e}] "
              f"py_id [{np.nanmin(py['Id']):.2e},{np.nanmax(py['Id']):.2e}] "
              f"port_conv {nconv}/{len(py['converged'])}")

    # ---------- aggregate ---------- #
    all_rel_err = []
    all_abs_err_vsint = []
    all_abs_err_vb = []
    port_conv = []
    ng_conv_all = []
    by_vg1 = {0.2: [], 0.4: [], 0.6: []}
    fail_biases = []      # port did not converge
    big_id_diff = []      # >50% diff where both converged

    for cur in per_curve:
        for i in range(len(cur["Vd"])):
            ng_id = cur["ng_Id"][i]
            py_id = cur["py_Id"][i]
            ng_vs = cur["ng_Vsint"][i]
            py_vs = cur["py_Vsint"][i]
            ng_vb = cur["ng_Vb"][i]
            py_vb = cur["py_Vb"][i]
            pyc = bool(cur["py_converged"][i])
            ngc = bool(cur["ng_converged"][i])
            port_conv.append(pyc)
            ng_conv_all.append(ngc)
            if not pyc:
                fail_biases.append({"vg1": cur["vg1"], "vg2": cur["vg2"],
                                    "Vd": cur["Vd"][i]})
            if pyc and ngc and np.isfinite(ng_id) and np.isfinite(py_id):
                denom = max(abs(ng_id), 1e-15)
                rel = abs(py_id - ng_id) / denom
                all_rel_err.append(rel)
                by_vg1[cur["vg1"]].append(rel)
                all_abs_err_vsint.append(abs(py_vs - ng_vs))
                all_abs_err_vb.append(abs(py_vb - ng_vb))
                if rel > 0.50:
                    big_id_diff.append({"vg1": cur["vg1"], "vg2": cur["vg2"],
                                        "Vd": cur["Vd"][i],
                                        "ng_Id": ng_id, "py_Id": py_id,
                                        "rel_err": rel})

    def stats(xs):
        if not xs:
            return {"n": 0}
        a = np.asarray(xs)
        return {"n": int(a.size),
                "median": float(np.median(a)),
                "mean": float(np.mean(a)),
                "p95": float(np.percentile(a, 95)),
                "max": float(np.max(a))}

    aggregate = {
        "n_curves": len(per_curve),
        "n_bias_points_total": len(port_conv),
        "port_convergence_rate": float(np.mean(port_conv)),
        "ngspice_convergence_rate": float(np.mean(ng_conv_all)),
        "rel_err_Id": stats(all_rel_err),
        "abs_err_Vsint": stats(all_abs_err_vsint),
        "abs_err_Vb": stats(all_abs_err_vb),
        "by_vg1_rel_err_Id": {str(k): stats(v) for k, v in by_vg1.items()},
        "n_port_failures": len(fail_biases),
        "n_big_id_diff_gt50pct": len(big_id_diff),
        "fail_biases": fail_biases[:30],
        "big_id_diff_examples": big_id_diff[:30],
    }

    summary = {"aggregate": aggregate, "per_curve": per_curve,
               "vd_grid": VD_LIST.tolist()}
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))

    # ---------- plot ---------- #
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    for ax, vg1 in zip(axes, [0.2, 0.4, 0.6]):
        # color cycle by vg2
        curves = [c for c in per_curve if c["vg1"] == vg1]
        cmap = plt.cm.viridis(np.linspace(0, 1, max(1, len(curves))))
        for cc, cur in zip(cmap, curves):
            vd = np.asarray(cur["Vd"])
            ax.plot(vd, np.abs(np.asarray(cur["ng_Id"])), "-",
                    color=cc, lw=1.5, label=f"VG2={cur['vg2']:+.2f} ng")
            ax.plot(vd, np.abs(np.asarray(cur["py_Id"])), ":",
                    color=cc, lw=1.8)
            sbv = np.asarray(cur["sb_Vd"])
            sbi = np.asarray(cur["sb_Id"])
            if sbv.size:
                ax.plot(sbv, np.abs(sbi), "o", color=cc, ms=2.5, alpha=0.6)
        ax.set_yscale("log")
        ax.set_xlabel("Vd (V)")
        ax.set_ylabel("|Id| (A)")
        ax.set_title(f"VG1={vg1}  (solid=ngspice, dotted=our port, markers=Sebas)")
        ax.set_ylim(1e-13, 1e-2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("z217: 2T NS-RAM cell — port vs ngspice vs measured (default Sebas card)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparison.png", dpi=130)
    plt.close(fig)

    # ---------- findings ---------- #
    a = aggregate
    findings = f"""# z217: 2T full envelope validation

## Setup
- 33 measured (VG1, VG2) bias curves from `data/sebas_2026_04_22/`
- Vd sweep 0.05..1.95 V step 0.1 V → {len(VD_LIST)} points/curve = {a['n_bias_points_total']} total
- Default Sebas card (alpha0=7.84e-5, beta0=18; BJT from_sebas_card)
- ngspice opts: gmin=1e-15 reltol=1e-4 itl1=500 itl2=200
- our port: forward_2t(use_homotopy=True)

## Convergence
- ngspice convergence: {a['ngspice_convergence_rate']*100:.1f}%
- our port convergence: {a['port_convergence_rate']*100:.1f}%
- port failures: {a['n_port_failures']}
- biases with both-converged but |Id| differing >50%: {a['n_big_id_diff_gt50pct']}

## Port vs ngspice (where both converged)
Id relative error:
  median {a['rel_err_Id'].get('median', float('nan')):.3e}
  mean   {a['rel_err_Id'].get('mean',   float('nan')):.3e}
  p95    {a['rel_err_Id'].get('p95',    float('nan')):.3e}
  max    {a['rel_err_Id'].get('max',    float('nan')):.3e}

Vsint absolute error (V):
  median {a['abs_err_Vsint'].get('median', float('nan')):.3e}
  max    {a['abs_err_Vsint'].get('max',    float('nan')):.3e}

Vb absolute error (V):
  median {a['abs_err_Vb'].get('median', float('nan')):.3e}
  max    {a['abs_err_Vb'].get('max',    float('nan')):.3e}

## Per VG1 group (rel_err Id)
"""
    for k, v in a['by_vg1_rel_err_Id'].items():
        if v.get('n', 0):
            findings += f"  VG1={k}: median={v['median']:.3e} p95={v['p95']:.3e} max={v['max']:.3e} (n={v['n']})\n"
        else:
            findings += f"  VG1={k}: n=0 (no both-converged biases)\n"

    findings += "\n## Failure regions (port not converged)\n"
    if a['fail_biases']:
        for fb in a['fail_biases'][:15]:
            findings += f"  VG1={fb['vg1']} VG2={fb['vg2']:+.3f} Vd={fb['Vd']:.2f}\n"
        if len(a['fail_biases']) > 15:
            findings += f"  ... +{len(a['fail_biases'])-15} more\n"
    else:
        findings += "  none\n"

    findings += "\n## Large-disagreement biases (rel_err Id > 50%)\n"
    if a['big_id_diff_examples']:
        for bd in a['big_id_diff_examples'][:15]:
            findings += (f"  VG1={bd['vg1']} VG2={bd['vg2']:+.3f} Vd={bd['Vd']:.2f}: "
                         f"ng={bd['ng_Id']:.3e} py={bd['py_Id']:.3e} rel={bd['rel_err']:.2f}\n")
        if len(a['big_id_diff_examples']) > 15:
            findings += f"  ... +{len(a['big_id_diff_examples'])-15} more\n"
    else:
        findings += "  none\n"

    findings += """
## Classification
- "port matches ngspice" but "ngspice doesn't match Sebas data" → fitting problem
- "port differs from ngspice"                                   → residual model bug
See comparison.png for visual confirmation across all 33 curves.
"""
    (OUT_DIR / "findings.md").write_text(findings)
    print("Wrote", OUT_DIR / "findings.md")
    print("Wrote", OUT_DIR / "summary.json")
    print("Wrote", OUT_DIR / "comparison.png")
    print(f"\nport conv {a['port_convergence_rate']*100:.1f}%, "
          f"Id rel-err median {a['rel_err_Id'].get('median', float('nan')):.2e}")


if __name__ == "__main__":
    main()
