"""z143 — F4 2T-cell ngspice op-point cross-check vs pyport.

Runs the ngspice deck `research_plan/ngspice_repro_harness/test_2t_cell.sp`
(48 op-points: 3 VG1 x 4 VG2 x 4 Vd) and compares Vsint, Vb, Id against
pyport's `forward_2t_arclength_grad` with the post-F1 physical Bf=100.

Acceptance (per F4 spec):
  |Δ Vsint|        ≤ 5 mV
  |Δ Vb|           ≤ 5 mV
  |Δ log10 Id|     ≤ 0.3 dec

Smoke-test mode: NSRAM_F4_SMOKE=1 runs only (VG1=0.6, VG2=0.30, Vd=1.0).
If that one bias disagrees by >0.5 dec on Id, raise SystemExit.
"""
from __future__ import annotations
import json, math, os, re, csv, subprocess, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/sebas_2026_04_22"
DECK = ROOT / "research_plan/ngspice_repro_harness/test_2t_cell.sp"
OUT  = ROOT / "results/z143_2t_cell_crossval"
OUT.mkdir(parents=True, exist_ok=True)

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model, parse_param_blocks
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.arclength import forward_2t_arclength_grad
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

# Reuse z91f's patcher (cross-file .param scope + post-load fixup)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "z91f_mod", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(z91f)
patch_model_values = z91f.patch_model_values


# --------------------------------------------------------------------- #
#  Sweep grid (must match the deck)                                     #
# --------------------------------------------------------------------- #
VG1_GRID = [0.2, 0.4, 0.6]
VG2_GRID = [-0.10, 0.00, 0.15, 0.30]
VD_GRID  = [0.5, 1.0, 1.5, 2.0]


# --------------------------------------------------------------------- #
#  ngspice deck runner + parser                                         #
# --------------------------------------------------------------------- #
def run_ngspice() -> list[dict]:
    """Run the ngspice deck and parse the ###ROW###-tagged op-points."""
    print(f"[z143] running ngspice {DECK.name}", flush=True)
    t0 = time.time()
    res = subprocess.run(
        ["ngspice", "-b", DECK.name],
        cwd=DECK.parent,
        capture_output=True, text=True, timeout=300,
    )
    print(f"[z143] ngspice elapsed {time.time()-t0:.1f}s, rc={res.returncode}",
          flush=True)
    out = res.stdout
    # Save raw
    (OUT / "ngspice.log").write_text(out + "\n----STDERR----\n" + res.stderr)
    # Parse: alternating "###ROW### vg1=… vg2=… vd=…" then three lines
    # "vsint_v = …", "vb_v = …", "id_v = …"
    rows = []
    lines = out.splitlines()
    i = 0
    row_re = re.compile(r"###ROW### vg1=([\-0-9.eE+]+) vg2=([\-0-9.eE+]+) vd=([\-0-9.eE+]+)")
    val_re = re.compile(r"^(\w+)\s*=\s*([\-0-9.eE+]+)")
    cur = None
    for ln in lines:
        m = row_re.search(ln)
        if m:
            if cur is not None:
                rows.append(cur)
            cur = {"VG1": float(m.group(1)),
                   "VG2": float(m.group(2)),
                   "Vd":  float(m.group(3))}
            continue
        if cur is not None:
            v = val_re.match(ln.strip())
            if v:
                key = v.group(1); val = float(v.group(2))
                if key == "vsint_v": cur["Vsint"] = val
                elif key == "vb_v":  cur["Vb"]    = val
                elif key == "id_v":  cur["Id"]    = val
    if cur is not None:
        rows.append(cur)
    return rows


# --------------------------------------------------------------------- #
#  pyport setup (mirrors z91g lines 80-141)                             #
# --------------------------------------------------------------------- #
def build_pyport():
    """Load M1+M2 cards and build (cfg, model_M1, model_M2, bjt)."""
    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    shared_params = parse_param_blocks(text_M2)

    model_M1 = BSIM4Model.from_spice(text_M1, model_type="nmos",
                                     params=shared_params)
    patch_model_values(model_M1, type_n=True)

    model_M2 = BSIM4Model.from_spice(text_M2, model_type="nmos",
                                     params=shared_params)
    patch_model_values(model_M2, type_n=True)

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            newton_max_iters=50)
    # F4: do NOT enable the body-pdiode in pyport unless we also add it to
    # the deck. We DO add a Dpdi diode in the deck; turn it on here too.
    cfg.body_pdiode_to = "sint"   # matches deck Dpdi anode=Vb cath=Vsint

    sd_M1 = compute_size_dep(model_M1, Geometry(L=cfg.Ln, W=cfg.Wn),
                             T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model_M2,
                             Geometry(L=cfg.Ln * cfg.M2_length_factor,
                                      W=cfg.Wn), T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2

    # F1 physical NPN: Bf=100. Using Sebas's parasiticBJT card baseline,
    # override Bf to the post-F1 physical value (≤100).
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 100.0
    return cfg, model_M1, model_M2, bjt


def pyport_eval(cfg, model_M1, model_M2, bjt, VG1, VG2, Vd_list):
    """Evaluate pyport at one (VG1, VG2) over a list of Vd targets."""
    Vd_seq = torch.tensor(sorted(Vd_list), dtype=torch.float64)
    with torch.no_grad():
        out = forward_2t_arclength_grad(
            cfg, model_M1=model_M1, model_M2=model_M2, bjt=bjt,
            Vd_seq=Vd_seq,
            VG1=torch.tensor(VG1), VG2=torch.tensor(VG2))
    return {
        "Vd":    Vd_seq.numpy(),
        "Vsint": out["Vsint"].detach().numpy(),
        "Vb":    out["Vb"].detach().numpy(),
        "Id":    out["Id"].detach().abs().numpy(),
        "conv":  out["converged"].detach().numpy().astype(bool),
    }


# --------------------------------------------------------------------- #
#  Main                                                                 #
# --------------------------------------------------------------------- #
def main():
    smoke = bool(int(os.environ.get("NSRAM_F4_SMOKE", "0")))

    # 1. ngspice
    ngs_rows = run_ngspice()
    print(f"[z143] parsed {len(ngs_rows)} op-points from ngspice", flush=True)
    if not ngs_rows:
        raise SystemExit("[z143] FATAL: no ngspice rows parsed — deck broken")

    # Index by (VG1, VG2, Vd) with float key tolerance
    def key(vg1, vg2, vd):
        return (round(vg1, 3), round(vg2, 3), round(vd, 3))
    ngs_map = {key(r["VG1"], r["VG2"], r["Vd"]): r for r in ngs_rows}

    # 2. pyport
    cfg, model_M1, model_M2, bjt = build_pyport()

    if smoke:
        targets = [(0.6, 0.30, 1.0)]
    else:
        targets = [(g1, g2, vd)
                   for g1 in VG1_GRID for g2 in VG2_GRID for vd in VD_GRID]

    # Group by (VG1, VG2) so arclength sees the full 4-Vd sweep per row
    bias_groups: dict[tuple[float, float], list[float]] = {}
    for g1, g2, vd in targets:
        bias_groups.setdefault((g1, g2), []).append(vd)

    log_eps = 1e-18
    rows = []
    for (g1, g2), vds in bias_groups.items():
        py = pyport_eval(cfg, model_M1, model_M2, bjt, g1, g2, vds)
        for k, vd in enumerate(py["Vd"]):
            ngs = ngs_map.get(key(g1, g2, float(vd)))
            if ngs is None:
                print(f"  [miss] no ngspice row for ({g1},{g2},{vd})", flush=True)
                continue
            d_vsint = float(py["Vsint"][k]) - ngs["Vsint"]
            d_vb    = float(py["Vb"][k])    - ngs["Vb"]
            id_py   = max(float(py["Id"][k]),  log_eps)
            id_ngs  = max(abs(ngs["Id"]),       log_eps)
            d_log_id = math.log10(id_py) - math.log10(id_ngs)
            row = {
                "VG1": g1, "VG2": g2, "Vd": float(vd),
                "Vsint_ngs": ngs["Vsint"], "Vsint_py": float(py["Vsint"][k]),
                "Vb_ngs":    ngs["Vb"],    "Vb_py":    float(py["Vb"][k]),
                "Id_ngs":    abs(ngs["Id"]), "Id_py":   id_py,
                "dVsint_mV": d_vsint * 1e3,
                "dVb_mV":    d_vb    * 1e3,
                "dlog10Id":  d_log_id,
                "py_conv":   bool(py["conv"][k]),
                "pass_vsint": abs(d_vsint) <= 5e-3,
                "pass_vb":    abs(d_vb)    <= 5e-3,
                "pass_id":    abs(d_log_id) <= 0.3,
            }
            row["pass_all"] = all([row["pass_vsint"], row["pass_vb"],
                                   row["pass_id"]])
            rows.append(row)
            tag = "PASS" if row["pass_all"] else "FAIL"
            print(f"  [{tag}] VG1={g1:.2f} VG2={g2:+.2f} Vd={vd:.2f}  "
                  f"ΔVs={row['dVsint_mV']:+7.2f}mV  "
                  f"ΔVb={row['dVb_mV']:+7.2f}mV  "
                  f"Δlog10Id={row['dlog10Id']:+6.2f}", flush=True)

    # 3. Smoke-test gate
    if smoke:
        if not rows:
            raise SystemExit("[z143] SMOKE FATAL: no rows compared")
        r = rows[0]
        if abs(r["dlog10Id"]) > 0.5:
            raise SystemExit(
                f"[z143] SMOKE FATAL: Δlog10Id={r['dlog10Id']:+.2f} > 0.5 dec")
        print(f"[z143] SMOKE OK: Δlog10Id={r['dlog10Id']:+.2f}, "
              f"ΔVs={r['dVsint_mV']:+.1f}mV, ΔVb={r['dVb_mV']:+.1f}mV",
              flush=True)
        # write smoke summary
        (OUT / "smoke.json").write_text(json.dumps(r, indent=2))
        return

    # 4. Save full table + summary
    fields = ["VG1","VG2","Vd","Vsint_ngs","Vsint_py","Vb_ngs","Vb_py",
              "Id_ngs","Id_py","dVsint_mV","dVb_mV","dlog10Id",
              "py_conv","pass_vsint","pass_vb","pass_id","pass_all"]
    with (OUT / "agreement.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    n = len(rows)
    n_pass = sum(1 for r in rows if r["pass_all"])
    summary = {
        "n_points":        n,
        "n_pass_all":      n_pass,
        "n_pass_vsint":    sum(1 for r in rows if r["pass_vsint"]),
        "n_pass_vb":       sum(1 for r in rows if r["pass_vb"]),
        "n_pass_id":       sum(1 for r in rows if r["pass_id"]),
        "max_abs_dVsint_mV": max(abs(r["dVsint_mV"]) for r in rows),
        "max_abs_dVb_mV":    max(abs(r["dVb_mV"])    for r in rows),
        "max_abs_dlog10Id":  max(abs(r["dlog10Id"])  for r in rows),
        "median_abs_dlog10Id": float(np.median([abs(r["dlog10Id"]) for r in rows])),
        "verdict": "PASS" if n_pass == n else "FAIL",
        "thresholds": {"dVsint_mV":5.0, "dVb_mV":5.0, "dlog10Id":0.3},
        "bjt_Bf": bjt.Bf,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[z143] {n_pass}/{n} PASS  verdict={summary['verdict']}", flush=True)

    # 5. 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    arr = np.array([(r["Vsint_ngs"], r["Vsint_py"],
                     r["Vb_ngs"], r["Vb_py"],
                     r["Id_ngs"], r["Id_py"]) for r in rows])
    axes[0].plot(arr[:,0], arr[:,1], "o", ms=4, alpha=0.7)
    lim = [min(arr[:,0].min(), arr[:,1].min()),
           max(arr[:,0].max(), arr[:,1].max())]
    axes[0].plot(lim, lim, "k--", lw=0.8)
    axes[0].set_xlabel("Vsint ngspice [V]"); axes[0].set_ylabel("Vsint pyport [V]")
    axes[0].set_title("Vsint")
    axes[0].grid(alpha=0.3)

    axes[1].plot(arr[:,2], arr[:,3], "o", ms=4, alpha=0.7, color="tab:orange")
    lim = [min(arr[:,2].min(), arr[:,3].min()),
           max(arr[:,2].max(), arr[:,3].max())]
    axes[1].plot(lim, lim, "k--", lw=0.8)
    axes[1].set_xlabel("Vb ngspice [V]"); axes[1].set_ylabel("Vb pyport [V]")
    axes[1].set_title("Vb")
    axes[1].grid(alpha=0.3)

    axes[2].loglog(arr[:,4], arr[:,5], "o", ms=4, alpha=0.7, color="tab:green")
    lim = [min(arr[:,4].min(), arr[:,5].min()),
           max(arr[:,4].max(), arr[:,5].max())]
    axes[2].loglog(lim, lim, "k--", lw=0.8)
    axes[2].set_xlabel("Id ngspice [A]"); axes[2].set_ylabel("Id pyport [A]")
    axes[2].set_title("Id (log10)")
    axes[2].grid(alpha=0.3, which="both")

    fig.suptitle(
        f"z143 F4 cross-check — pyport vs ngspice 2T cell  "
        f"(N={n}, {n_pass} PASS, Bf={bjt.Bf:g})",
        fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "compare.png", dpi=140)
    plt.close(fig)
    print(f"[z143] saved {OUT}/compare.png", flush=True)


if __name__ == "__main__":
    main()
