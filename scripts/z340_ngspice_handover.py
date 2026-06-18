"""z340 — R-24 ngspice OP handover.

For each Sebas curve:
  1. Run ngspice DC sweep on a 2T cell deck mirroring z330's topology
     to obtain (Vsint*, Vb*) at every Vd point on the measured curve.
  2. Plug those ngspice OP states directly into pyport's `_residuals`
     (NO Newton solve) and read Id_pred from the components formula in
     forward_2t (Id = Ids_M1 + Ic_Q1 + Ic_lat + Ic_avalanche + Igidl_M1
     - Ibd_M1).
  3. log10_rmse vs Sebas measured Id per curve.
  4. Two conditions:
       Line A — z338 best params + bjt_emitter_to_gnd=True + eta_sigmoid ON
       Line B — same but eta_sigmoid OFF (z338 default).

Falsifies "pyport architecture is the limit" per gpt-5 oracle (O63).
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import sys, json, re, subprocess, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/z340_ngspice_handover"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"

M1_CARD = ROOT / "data/sebas_2026_04_22/M1_130DNWFB.txt"
M2_CARD = ROOT / "data/sebas_2026_04_22/M2_130bulkNSRAM.txt"


# z338 eval 21 best params
Z338_BEST = dict(
    alpha0 = 1.63357328192734e-05,
    Bf     = 2605.2882016162002,
    Va     = 0.3567358318716285,
    Is     = 3.2906845928467974e-10,
    lat_BV = 4.018266147002578,
    body_pdiode_Rs = 5480383.486345367,
)


def load_curves():
    curves = []
    for sub in DATA.iterdir():
        if not sub.is_dir():
            continue
        m_vg1 = re.search(r"VG1=([\d.\-]+)", sub.name)
        if not m_vg1:
            continue
        vg1 = float(m_vg1.group(1))
        for f in sorted(sub.glob("*.csv")):
            m = re.search(r"VG2=([\-\d.]+)", f.name)
            if not m: continue
            vg2 = float(m.group(1))
            d = np.loadtxt(f, delimiter=",", skiprows=1)
            if d.ndim != 2 or d.shape[1] < 2: continue
            curves.append({"VG1": vg1, "VG2": vg2,
                           "Vd": d[:,0].astype(float),
                           "Id": np.abs(d[:,1]).astype(float),
                           "f": f.name})
    curves.sort(key=lambda c: (c["VG1"], c["VG2"]))
    return curves


# ------------------------------------------------------------- ngspice deck

def make_deck(VG1, VG2, Vd_list):
    """Generate a deck that, in .control, loops over each Vd and prints
    v(vsint), v(vb)."""
    cmds = []
    for vd in Vd_list:
        # numerical stability: avoid identical Vd=0 with hard ground
        v = max(float(vd), 1e-6)
        cmds.append(f"alter Vdd dc = {v}")
        cmds.append("op")
        cmds.append(f"echo \"POINT Vd={v:.8e}\"")
        cmds.append("print v(vsint)")
        cmds.append("print v(vb)")
    ctrl = "\n".join(cmds)
    return f""".title z340 2T cell ngspice OP (VG1={VG1}, VG2={VG2})

.include "{M1_CARD}"
.include "{M2_CARD}"

.model parasiticBJT NPN(is={Z338_BEST['Is']:.4e} va={Z338_BEST['Va']:.4e}
+ bf={Z338_BEST['Bf']:.4e} br=100 nc=2 ikr=100m rc=0.1
+ vje=0.7 re=0.1 cjc=1e-15 fc=0.5 cje=0.7e-15 ne=1.5 ise=0 tr=20e-12 tf=25e-12
+ itf=0.03 vtf=7 xtf=2)

.model Dwell_mod D(IS=3.4089e-19 N=1.017 RS=0)

Vdd     vd       0       DC 0.001
Vg1     vg1      0       DC {VG1}
Vg2     vg2      0       DC {VG2}
Vnwell  vnwell   0       DC 2.0

M1  vd vg1 vsint vb NMOSdnwfb L=0.13u W=1u
M2  vsint vg2 0 0 NMOS L=0.234u W=1u
Q1  vsint vb 0 parasiticBJT area=1u
Rwell  vnwell vnwell_x  10G
Dwell  vb     vnwell_x  Dwell_mod

.options gmin=1e-15 abstol=1e-12 reltol=1e-3 itl1=500

.control
{ctrl}
quit
.endc

.end
"""


def parse_log(text):
    """Return list of (Vd, Vsint, Vb) tuples in order."""
    out = []
    cur_vd = None
    cur_vsint = None
    cur_vb = None
    # Splits per "POINT Vd=" marker
    pat_pt   = re.compile(r"POINT Vd=([-+]?\d+\.?\d*[eE]?[-+]?\d*)")
    pat_vsint= re.compile(r"v\(vsint\)\s*=\s*([-+]?\d+\.?\d*[eE]?[-+]?\d*)")
    pat_vb   = re.compile(r"v\(vb\)\s*=\s*([-+]?\d+\.?\d*[eE]?[-+]?\d*)")
    blocks = re.split(r"POINT Vd=", text)
    for blk in blocks[1:]:
        mvd = re.match(r"([-+]?\d+\.?\d*[eE]?[-+]?\d*)", blk)
        mvs = pat_vsint.search(blk)
        mvb = pat_vb.search(blk)
        if mvd and mvs and mvb:
            out.append((float(mvd.group(1)), float(mvs.group(1)), float(mvb.group(1))))
    return out


def run_ngspice_for_curve(c, tmp_dir):
    deck = make_deck(c["VG1"], c["VG2"], list(c["Vd"]))
    deck_path = tmp_dir / f"deck_VG1_{c['VG1']:.2f}_VG2_{c['VG2']:+.2f}.sp"
    deck_path.write_text(deck)
    try:
        proc = subprocess.run(["ngspice", "-b", str(deck_path)],
                              capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    text = proc.stdout + "\n--STDERR--\n" + proc.stderr
    return parse_log(text), text


# ------------------------------------------------------------- pyport eval

def build_pyport(eta_sigmoid: bool):
    import importlib.util
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    cfg.eta_sigmoid = bool(eta_sigmoid)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    # Apply z338 best params
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


def eval_residuals_at_states(cfg, M1, M2, bjt, Vd_np, VG1, VG2, Vsint_np, Vb_np):
    """Return Id_pred at the supplied (Vsint, Vb) per-point states."""
    from nsram.bsim4_port.nsram_cell_2T import _residuals
    Vd  = torch.tensor(Vd_np,  dtype=torch.float64)
    Vsint = torch.tensor(Vsint_np, dtype=torch.float64)
    Vb    = torch.tensor(Vb_np, dtype=torch.float64)
    vg1 = torch.full_like(Vd, float(VG1))
    vg2 = torch.full_like(Vd, float(VG2))
    R_S, R_B, comp = _residuals(cfg, M1, bjt, Vd, vg1, vg2, Vsint, Vb,
                                model_M2=M2)
    Id = (
        comp["Ids_M1"]
        + comp["Ic_Q1"]
        + comp.get("Ic_lat", torch.zeros_like(Vd))
        + comp.get("Ic_avalanche", torch.zeros_like(Vd))
        + comp["Igidl_M1"]
        - comp["Ibd_M1"]
    )
    return Id.detach().cpu().numpy(), R_S.detach().cpu().numpy(), R_B.detach().cpu().numpy()


def log_rmse(Id_meas, Id_pred):
    mask = (Id_meas > 1e-15) & (np.abs(Id_pred) > 1e-15) & np.isfinite(Id_pred)
    if mask.sum() < 3:
        return float("nan"), int(mask.sum())
    logr = np.log10(np.abs(Id_pred[mask])) - np.log10(Id_meas[mask])
    return float(np.sqrt(np.mean(logr**2))), int(mask.sum())


# ------------------------------------------------------------- main

def main():
    t0 = time.time()
    curves = load_curves()
    print(f"[z340] loaded {len(curves)} curves", flush=True)

    tmp_dir = OUT / "ngspice_decks"
    tmp_dir.mkdir(exist_ok=True)

    # Pass 1: run ngspice once per curve (states are identical for both lines)
    per_bias_states = []
    for i, c in enumerate(curves):
        t_e = time.time()
        states, log_text = run_ngspice_for_curve(c, tmp_dir)
        if states is None:
            print(f"  [{i+1}/{len(curves)}] VG1={c['VG1']:.2f} VG2={c['VG2']:+.2f} NGSPICE FAIL", flush=True)
            per_bias_states.append({"VG1": c["VG1"], "VG2": c["VG2"], "states": None})
            continue
        n_ok = len(states)
        # align to Vd: ngspice returns one entry per .op call
        per_bias_states.append({
            "VG1": c["VG1"], "VG2": c["VG2"],
            "n_ngspice_op": n_ok,
            "n_vd": len(c["Vd"]),
            "states": states,  # list of (Vd, Vsint, Vb)
        })
        print(f"  [{i+1}/{len(curves)}] VG1={c['VG1']:.2f} VG2={c['VG2']:+.2f} "
              f"n_op={n_ok}/{len(c['Vd'])} dt={time.time()-t_e:.1f}s", flush=True)

    (OUT / "per_bias_states.json").write_text(json.dumps(per_bias_states, indent=2))
    print(f"[z340] ngspice pass done, elapsed={time.time()-t0:.0f}s", flush=True)

    # Pass 2 & 3: pyport residual eval at ngspice OP states, two cfg lines
    results = {}
    for line_name, eta_on in [("A_eta_sigmoid_ON", True),
                              ("B_eta_sigmoid_OFF", False)]:
        cfg, M1, M2, bjt = build_pyport(eta_sigmoid=eta_on)
        per_bias_rmse = []
        for c, bs in zip(curves, per_bias_states):
            if bs["states"] is None:
                per_bias_rmse.append({"VG1": c["VG1"], "VG2": c["VG2"],
                                       "log_rmse_dec": float("nan"),
                                       "n_used": 0})
                continue
            arr = np.array(bs["states"], dtype=float)  # (N,3): Vd, Vsint, Vb
            # Map ngspice Vd back to measured Vd by index (in deck order = c["Vd"] order)
            Vd_op = arr[:,0]
            Vsint_op = arr[:,1]
            Vb_op = arr[:,2]
            # truncate measured Id to same length
            n = len(arr)
            Id_meas = c["Id"][:n]
            Id_pred, R_S, R_B = eval_residuals_at_states(cfg, M1, M2, bjt,
                                                         Vd_op, c["VG1"], c["VG2"],
                                                         Vsint_op, Vb_op)
            rmse, n_used = log_rmse(Id_meas, Id_pred)
            per_bias_rmse.append({
                "VG1": c["VG1"], "VG2": c["VG2"],
                "log_rmse_dec": rmse, "n_used": n_used,
                "max_abs_R": float(max(np.max(np.abs(R_S)), np.max(np.abs(R_B)))),
            })
        valid = [r["log_rmse_dec"] for r in per_bias_rmse if not np.isnan(r["log_rmse_dec"])]
        med = float(np.median(valid)) if valid else float("nan")
        # per-VG1 breakdown
        by_vg1 = {}
        for r in per_bias_rmse:
            v = r["VG1"]
            by_vg1.setdefault(v, []).append(r["log_rmse_dec"])
        per_vg1_med = {f"{k:.2f}": float(np.nanmedian(v)) for k,v in sorted(by_vg1.items())}
        results[line_name] = {
            "median_dec": med,
            "n_valid": len(valid),
            "n_total": len(per_bias_rmse),
            "per_vg1_median": per_vg1_med,
            "per_bias": per_bias_rmse,
        }
        print(f"[z340] {line_name}: median={med:.3f} dec  per-VG1={per_vg1_med}  "
              f"valid={len(valid)}/{len(per_bias_rmse)}", flush=True)

    # Verdict (gpt-5 rules)
    medA = results["A_eta_sigmoid_ON"]["median_dec"]
    medB = results["B_eta_sigmoid_OFF"]["median_dec"]
    if np.isnan(medA) or np.isnan(medB):
        verdict = "INCOMPLETE — NaN in one or both lines, cannot decide"
    elif medA >= 2.0 and medB >= 2.0:
        verdict = "REWRITE REQUIRED — both ≥2.0 dec at ngspice OPs → pyport architecture is missing physics"
    elif medA < 1.2 or medB < 1.2:
        verdict = "FIX SOLVER — pyport residual evaluator OK at correct OPs; the Newton solver/init was the blocker"
    else:
        verdict = "MIXED (1.2-2.0) — both solver and physics matter; deeper diagnosis needed"

    summary = {
        "z338_best_params": Z338_BEST,
        "lines": results,
        "verdict": verdict,
        "rule": "gpt-5: both≥2.0→rewrite | either<1.2→solver | else mixed",
        "elapsed_s": time.time() - t0,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n===== z340 VERDICT =====")
    print(f"  Line A (eta_sigmoid ON):  {medA:.3f} dec")
    print(f"  Line B (eta_sigmoid OFF): {medB:.3f} dec")
    print(f"  {verdict}")
    print(f"  elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
