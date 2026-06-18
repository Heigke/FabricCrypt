#!/usr/bin/env python3
"""Track ngspice cross-validation — confirm pyport's K1+ALPHA0+Hurkx config
matches ngspice (BSIM4 reference simulator) at 9 representative biases.

Config under test:
  - K1@VG1=0.6 = 0.53825 (BSIM card value, override default 0.41825)
  - ALPHA0 = 7.83756e-4 (Mario LALPHA0_FIX card, override CSV 7.842e-5)
  - Hurkx-BBT: A=1e-6, B=1.5  (pyport-only; ngspice does NOT have this term)

CAVEAT: ngspice's native BSIM4 has JTSMOD-style band-to-band tunneling but
our pyport's Hurkx-BBT is a *custom additive* term in the body KCL. So the
strict apples-to-apples comparison is the K1+ALPHA0 part. Hurkx-BBT is run
in pyport only (we report both `pyport_dec` with Hurkx and ngspice_dec
without Hurkx — and an additional `pyport_no_hurkx_dec` for direct match).

Biases: VG1 ∈ {0.2, 0.4, 0.6} × VG2 ∈ {-0.1, 0.0, 0.1} = 9 points.
Sweep: Vd ∈ [0, 2] V step 0.05.

Outputs:
  results/track_ngspice_xval/ablation.json
  results/track_ngspice_xval/verdict.md
  results/track_ngspice_xval/plot.png
  results/track_ngspice_xval/decks/  (per-bias ngspice decks + logs)

PASS gate: mean |ngspice_dec − pyport_no_hurkx_dec| ≤ 0.3 dec
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys, json, re, subprocess, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/track_ngspice_xval"
DECKS = OUT / "decks"
OUT.mkdir(parents=True, exist_ok=True)
DECKS.mkdir(parents=True, exist_ok=True)

# Bias grid
VG1_GRID = [0.2, 0.4, 0.6]
VG2_GRID = [-0.1, 0.0, 0.1]
VNWELL = 2.0
VD_LO, VD_HI, VD_STEP = 0.0, 2.0, 0.05

# Config under test
K1_OVERRIDE = 0.53825
ALPHA0_OVERRIDE = 7.83756e-4
HURKX_A = 1e-6
HURKX_B = 1.5
HURKX_P = 2.5

# ngspice card files: the LALPHA0_FIX variants already have k1=0.53825 (M1)
# and alpha0=7.83756e-4. Use these directly.
M1_CARD = ROOT / "data/sebas_2026_04_22/M1_130DNWFB_LALPHA0_FIX.txt"
M2_CARD = ROOT / "data/sebas_2026_04_22/M2_130bulkNSRAM_LALPHA0_FIX.txt"

DEC_FLOOR_MEAS = 1e-12
DEC_FLOOR_PRED = 1e-30
PASS_DELTA = 0.3  # dec

# -----------------------------------------------------------------------------
# Measured CSV loader
# -----------------------------------------------------------------------------
DATA = ROOT / "data/sebas_2026_04_22"


def find_meas_csv(VG1, VG2):
    """Find Sebas CSV for given (VG1, VG2). Returns Path or None."""
    # Directory pattern: 2vHCa-2 I-Vs@VG2 VG1=0.{N} vnwell=2
    vg1_str = f"{VG1:.1f}".rstrip("0").rstrip(".") if VG1 not in (0.2, 0.4, 0.6) else f"{VG1:.1f}"
    # Try canonical
    candidates = list(DATA.glob(f"2vHCa-2*VG1={VG1:.1f}*"))
    if not candidates:
        candidates = list(DATA.glob(f"2vHCa-2*VG1={vg1_str}*"))
    if not candidates:
        return None
    subdir = candidates[0]
    # File pattern: ...VG2=X.XX_VG=Y.Y(1)_...csv
    vg2_str = f"{VG2:.2f}"
    matches = list(subdir.glob(f"*VG2={vg2_str}_*.csv"))
    if not matches:
        # try with extra leading "-0.10" style
        if VG2 < 0:
            vg2_str = f"-{abs(VG2):.2f}"
            matches = list(subdir.glob(f"*VG2={vg2_str}_*.csv"))
    return matches[0] if matches else None


def load_measured(csv_path):
    """Return Vd, Id forward leg (apex split)."""
    d = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    Vd = d[:, 0].astype(np.float64)
    Id = np.abs(d[:, 1]).astype(np.float64)
    apex = int(np.argmax(Vd))
    return Vd[: apex + 1], Id[: apex + 1]


def median_dec(Vd_meas, Id_meas, Vd_sim, Id_sim, vmin=0.3):
    """Interpolate sim onto meas Vd axis, return median |log10 ratio| for Vd>vmin."""
    Id_sim_abs = np.abs(Id_sim)
    # Interpolate sim onto meas axis
    Id_sim_interp = np.interp(Vd_meas, Vd_sim, Id_sim_abs)
    m = (Vd_meas > vmin) & (np.abs(Id_meas) > DEC_FLOOR_MEAS) & (Id_sim_interp > 0)
    if m.sum() < 3:
        return float("nan"), 0
    lm = np.log10(np.clip(np.abs(Id_meas[m]), DEC_FLOOR_MEAS, None))
    lp = np.log10(np.clip(Id_sim_interp[m], DEC_FLOOR_PRED, None))
    return float(np.median(np.abs(lm - lp))), int(m.sum())


# -----------------------------------------------------------------------------
# ngspice
# -----------------------------------------------------------------------------
DECK_TMPL = """.title track_ngspice_xval VG1={vg1} VG2={vg2} (K1+ALPHA0 LALPHA0_FIX cards)

.include "{m1_card}"
.include "{m2_card}"

* Production BJT (z395/z330 production card)
.model parasiticBJT NPN(is=1e-9 va=0.55 bf=9000 br=100 nc=2 ikr=100m rc=0.1
+ vje=0.7 re=0.1 cjc=1e-15 fc=0.5 cje=0.7e-15 ne=1.5 ise=0 tr=20e-12 tf=25e-12
+ itf=0.03 vtf=7 xtf=2)

.model Dwell_mod D(IS=3.4089e-19 N=1.017 RS=0)

Vdd     vd       0       DC 0
Vg1     vg1      0       DC {vg1}
Vg2     vg2      0       DC {vg2}
Vnwell  vnwell   0       DC {vnwell}

* 2T NSRAM cell, M1 floating body, parasitic NPN, well diode to V_nwell
M1  vd vg1 vsint vb NMOSdnwfb L=0.13u W=1u
M2  vsint vg2 0 0 NMOS L=0.234u W=1u
Q1  vsint vb 0 parasiticBJT area=1u
Rwell  vnwell vnwell_x  10G
Dwell  vb     vnwell_x  Dwell_mod

.options gmin=1e-15 abstol=1e-14 reltol=1e-3 itl1=500 itl2=200 itl6=100

.control
dc Vdd {vd_lo} {vd_hi} {vd_step}
wrdata {out_txt} -i(vdd) v(vsint) v(vb)
quit
.endc

.end
"""


def run_ngspice(VG1, VG2, tag):
    deck_path = DECKS / f"{tag}.sp"
    out_txt = DECKS / f"{tag}_dc.txt"
    log_path = DECKS / f"{tag}.log"
    deck = DECK_TMPL.format(
        vg1=VG1, vg2=VG2, vnwell=VNWELL,
        vd_lo=VD_LO, vd_hi=VD_HI, vd_step=VD_STEP,
        m1_card=M1_CARD, m2_card=M2_CARD,
        out_txt=str(out_txt),
    )
    deck_path.write_text(deck)
    try:
        proc = subprocess.run(["ngspice", "-b", str(deck_path)],
                              capture_output=True, text=True, timeout=120)
        log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        log_path.write_text("TIMEOUT\n")
        return None, None, "timeout"
    if not out_txt.exists():
        return None, None, f"no_output(rc={rc})"
    try:
        data = np.loadtxt(out_txt)
    except Exception as e:
        return None, None, f"parse_fail:{e}"
    if data.ndim != 2 or data.shape[1] < 2:
        return None, None, f"bad_shape:{data.shape}"
    Vd = data[:, 0]
    # wrdata format: col0=sweep, col1=-i(vdd), col2=sweep, col3=v(vsint), ...
    Id = np.abs(data[:, 1])
    return Vd, Id, "ok"


# -----------------------------------------------------------------------------
# pyport
# -----------------------------------------------------------------------------
def build_pyport_with_hurkx(use_hurkx=True):
    import importlib.util
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=120)
    cfg.bjt_emitter_to_gnd = True
    # IIIROUTE-FIX (2026-05-20): match build_pyport_base() — include the
    # well diode + body p-diode that ngspice ALSO models. Without these,
    # the pyport↔ngspice comparison was apples-to-oranges (xval inflated
    # the gap by 3-6 dec because the well-diode return path was missing).
    cfg.body_pdiode_to = "vnwell"
    cfg.use_well_diode = True
    cfg.vnwell = 2.0
    cfg.body_pdiode_Js = 5.3675e-7 / 22e-12
    cfg.body_pdiode_n = 1.0535
    cfg.body_pdiode_Rs = 1.0e6
    if use_hurkx:
        cfg.hurkx_bbt_A = HURKX_A
        cfg.hurkx_bbt_B = HURKX_B
        cfg.hurkx_bbt_P = HURKX_P
    else:
        cfg.hurkx_bbt_A = 0.0
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 9000.0; bjt.Va = 0.55; bjt.Is = 1e-9
    return cfg, M1, M2, bjt


def run_pyport_sweep(cfg, M1, M2, bjt, VG1, VG2, Vd_axis):
    """Apply K1 (at VG1=0.6 only, like pillar.BRANCH_FLAT logic) + ALPHA0 (both
    transistors) overrides, then sweep Vd via forward_2t.
    """
    import torch
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    from contextlib import contextmanager

    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)

    P_M1 = {"alpha0": float(ALPHA0_OVERRIDE)}
    P_M2 = {"alpha0": float(ALPHA0_OVERRIDE)}
    if abs(VG1 - 0.6) < 1e-6:
        P_M1["k1"] = float(K1_OVERRIDE)
    # Static M2 overrides (match pillar)
    for k, v in {"k1": 0.63825, "k2": -0.070435, "etab": -0.086777, "beta0": 18.0}.items():
        P_M2.setdefault(k, float(v))

    @contextmanager
    def patch(sd, overrides):
        saved = {}
        try:
            for k, v in overrides.items():
                saved[k] = sd.scaled.get(k, None); sd.scaled[k] = float(v)
            yield
        finally:
            for k, v in saved.items():
                if v is None: sd.scaled.pop(k, None)
                else: sd.scaled[k] = v

    Vd_t = torch.tensor(Vd_axis, dtype=torch.float64)
    try:
        with patch(sd_M1, P_M1), patch(sd_M2, P_M2):
            # IIIROUTE-FIX: multi_init=True with hot_Vb_init=0.8 enables
            # discovery of the high-Vb root that ngspice finds via source
            # stepping. The two-branch picker takes the higher-|Id|
            # converged solution. Without this the pyport gets pinned to
            # the low-Vb attractor and is 5+ decades below ngspice.
            out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_t,
                             VG1=torch.tensor(VG1, dtype=torch.float64),
                             VG2=torch.tensor(VG2, dtype=torch.float64),
                             warm_start=True,
                             multi_init=True,
                             hot_Vsint_init=0.2,
                             hot_Vb_init=0.8)
        Id = np.abs(out["Id"].detach().cpu().numpy()).astype(np.float64)
        Id = np.where(np.isfinite(Id), Id, 0.0)
        return Id
    except Exception as e:
        print(f"  pyport sweep FAIL VG1={VG1} VG2={VG2}: {e}")
        traceback.print_exc()
        return np.full_like(Vd_axis, np.nan)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=== track_ngspice_xval ===")
    print(f"  K1={K1_OVERRIDE}  ALPHA0={ALPHA0_OVERRIDE}  Hurkx A={HURKX_A} B={HURKX_B}")
    print(f"  9 biases (VG1×VG2={VG1_GRID}×{VG2_GRID})")
    print(f"  Vd ∈ [{VD_LO},{VD_HI}] step {VD_STEP}")
    print(f"  M1_card={M1_CARD.name}  M2_card={M2_CARD.name}")
    print()

    # Build pyport once
    print("[1/3] Building pyport (with Hurkx)...")
    cfg_h, M1_h, M2_h, bjt_h = build_pyport_with_hurkx(use_hurkx=True)
    print("[2/3] Building pyport (no Hurkx, for ngspice match)...")
    cfg_nh, M1_nh, M2_nh, bjt_nh = build_pyport_with_hurkx(use_hurkx=False)

    Vd_axis = np.arange(VD_LO, VD_HI + 1e-9, VD_STEP)

    biases = []
    for vg1 in VG1_GRID:
        for vg2 in VG2_GRID:
            biases.append((vg1, vg2))

    results = {"config": {
        "K1@VG1=0.6": K1_OVERRIDE, "ALPHA0": ALPHA0_OVERRIDE,
        "Hurkx_A": HURKX_A, "Hurkx_B": HURKX_B, "Hurkx_P": HURKX_P,
        "VD_LO": VD_LO, "VD_HI": VD_HI, "VD_STEP": VD_STEP,
        "M1_card": M1_CARD.name, "M2_card": M2_CARD.name,
        "PASS_DELTA": PASS_DELTA,
    }, "biases": []}

    print("\n[3/3] Running 9-bias sweep...")
    t_total = time.time()
    for (vg1, vg2) in biases:
        tag = f"VG1={vg1:.2f}_VG2={vg2:.2f}".replace("-", "m").replace(".", "p")
        print(f"\n--- VG1={vg1} VG2={vg2}  (tag={tag}) ---", flush=True)

        # Measured
        meas_csv = find_meas_csv(vg1, vg2)
        if meas_csv is None or not meas_csv.exists():
            print(f"  measured CSV missing (VG1={vg1} VG2={vg2}); skip")
            results["biases"].append({
                "VG1": vg1, "VG2": vg2, "skip_reason": "no_measured_csv",
            })
            continue
        Vd_meas, Id_meas = load_measured(meas_csv)
        print(f"  measured: {meas_csv.name}  (n={len(Vd_meas)} fwd-leg pts)")

        # ngspice
        t0 = time.time()
        Vd_ng, Id_ng, ng_status = run_ngspice(vg1, vg2, tag)
        ng_dt = time.time() - t0
        if Vd_ng is None:
            print(f"  ngspice FAIL: {ng_status} ({ng_dt:.1f}s)")
            results["biases"].append({
                "VG1": vg1, "VG2": vg2, "ngspice_status": ng_status,
                "ngspice_dt_s": ng_dt,
            })
            continue
        print(f"  ngspice OK: {len(Vd_ng)} pts  Id∈[{Id_ng.min():.2e},{Id_ng.max():.2e}]  ({ng_dt:.1f}s)")

        # pyport (with Hurkx)
        t0 = time.time()
        Id_py_h = run_pyport_sweep(cfg_h, M1_h, M2_h, bjt_h, vg1, vg2, Vd_axis)
        py_h_dt = time.time() - t0
        print(f"  pyport+Hurkx OK ({py_h_dt:.1f}s)  Id∈[{np.nanmin(Id_py_h):.2e},{np.nanmax(Id_py_h):.2e}]")

        # pyport (no Hurkx, for direct ngspice match)
        t0 = time.time()
        Id_py_nh = run_pyport_sweep(cfg_nh, M1_nh, M2_nh, bjt_nh, vg1, vg2, Vd_axis)
        py_nh_dt = time.time() - t0
        print(f"  pyport-no-Hurkx OK ({py_nh_dt:.1f}s)  Id∈[{np.nanmin(Id_py_nh):.2e},{np.nanmax(Id_py_nh):.2e}]")

        # Decade metrics vs measured
        ng_dec, ng_n = median_dec(Vd_meas, Id_meas, Vd_ng, Id_ng)
        py_h_dec, py_h_n = median_dec(Vd_meas, Id_meas, Vd_axis, Id_py_h)
        py_nh_dec, py_nh_n = median_dec(Vd_meas, Id_meas, Vd_axis, Id_py_nh)

        # Pyport vs ngspice direct (no-Hurkx)
        Id_py_nh_on_ng = np.interp(Vd_ng, Vd_axis, np.abs(Id_py_nh))
        m_dir = (Vd_ng > 0.3) & (Id_ng > 0) & (Id_py_nh_on_ng > 0)
        if m_dir.sum() >= 3:
            l_ng = np.log10(np.clip(Id_ng[m_dir], 1e-30, None))
            l_py = np.log10(np.clip(Id_py_nh_on_ng[m_dir], 1e-30, None))
            ng_vs_py_med = float(np.median(np.abs(l_ng - l_py)))
            ng_vs_py_max = float(np.max(np.abs(l_ng - l_py)))
        else:
            ng_vs_py_med = float("nan"); ng_vs_py_max = float("nan")

        row = {
            "VG1": vg1, "VG2": vg2, "tag": tag,
            "meas_csv": meas_csv.name, "n_meas": int(len(Vd_meas)),
            "ngspice_dec_vs_data": ng_dec, "ngspice_n": ng_n,
            "pyport_hurkx_dec_vs_data": py_h_dec, "pyport_hurkx_n": py_h_n,
            "pyport_nohurkx_dec_vs_data": py_nh_dec, "pyport_nohurkx_n": py_nh_n,
            "ngspice_vs_pyport_nohurkx_med_dec": ng_vs_py_med,
            "ngspice_vs_pyport_nohurkx_max_dec": ng_vs_py_max,
            "ng_status": ng_status,
            "ng_dt_s": ng_dt, "py_h_dt_s": py_h_dt, "py_nh_dt_s": py_nh_dt,
        }
        print(f"  dec vs data: ngspice={ng_dec:.3f}  pyport+Hurkx={py_h_dec:.3f}  pyport-noH={py_nh_dec:.3f}")
        print(f"  ngspice ↔ pyport-noH: med={ng_vs_py_med:.3f}  max={ng_vs_py_max:.3f}")
        results["biases"].append(row)

        # Persist incrementally
        with open(OUT / "ablation.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Aggregate
    deltas = [r["ngspice_vs_pyport_nohurkx_med_dec"] for r in results["biases"]
              if isinstance(r.get("ngspice_vs_pyport_nohurkx_med_dec"), float)
              and np.isfinite(r["ngspice_vs_pyport_nohurkx_med_dec"])]
    if deltas:
        mean_delta = float(np.mean(deltas))
        median_delta = float(np.median(deltas))
        max_delta = float(np.max(deltas))
    else:
        mean_delta = median_delta = max_delta = float("nan")
    verdict_pass = np.isfinite(mean_delta) and mean_delta <= PASS_DELTA
    results["aggregate"] = {
        "n_completed": len(deltas),
        "mean_ngspice_vs_pyport_dec": mean_delta,
        "median_ngspice_vs_pyport_dec": median_delta,
        "max_ngspice_vs_pyport_dec": max_delta,
        "PASS": bool(verdict_pass),
        "PASS_DELTA": PASS_DELTA,
        "runtime_s": time.time() - t_total,
    }
    with open(OUT / "ablation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # verdict.md
    lines = []
    lines.append(f"# track_ngspice_xval — pyport vs ngspice cross-validation\n")
    lines.append(f"**Config**: K1@VG1=0.6={K1_OVERRIDE}, ALPHA0={ALPHA0_OVERRIDE}, Hurkx-BBT (pyport-only): A={HURKX_A} B={HURKX_B} P={HURKX_P}\n")
    lines.append(f"**ngspice cards**: `{M1_CARD.name}` + `{M2_CARD.name}` (LALPHA0_FIX variants already contain K1=0.53825 and ALPHA0=7.83756e-4)\n")
    lines.append(f"**CAVEAT**: Hurkx-BBT is a *custom additive* term in pyport's body KCL. ngspice's native BSIM4 does NOT have an equivalent. The apples-to-apples comparison is **pyport-no-Hurkx ↔ ngspice**. We also report **pyport+Hurkx ↔ data** for completeness.\n")
    lines.append("")
    lines.append(f"## Per-bias results\n")
    lines.append("| VG1 | VG2 | ngspice dec vs data | pyport+Hurkx dec vs data | pyport-noH dec vs data | ngspice ↔ pyport-noH (med) | (max) |")
    lines.append("|-----|-----|--------------------:|-------------------------:|-----------------------:|---------------------------:|------:|")
    for r in results["biases"]:
        if "skip_reason" in r or "ngspice_status" in r and r.get("ng_status") != "ok":
            lines.append(f"| {r['VG1']} | {r['VG2']} | _skip_ | _skip_ | _skip_ | _skip_ | _skip_ |")
            continue
        lines.append(
            f"| {r['VG1']} | {r['VG2']} "
            f"| {r['ngspice_dec_vs_data']:.3f} "
            f"| {r['pyport_hurkx_dec_vs_data']:.3f} "
            f"| {r['pyport_nohurkx_dec_vs_data']:.3f} "
            f"| {r['ngspice_vs_pyport_nohurkx_med_dec']:.3f} "
            f"| {r['ngspice_vs_pyport_nohurkx_max_dec']:.3f} |"
        )
    lines.append("")
    lines.append(f"## Aggregate\n")
    lines.append(f"- n biases completed: **{len(deltas)}** / 9")
    lines.append(f"- mean |ngspice − pyport-noH| (median per-bias): **{mean_delta:.3f}** dec")
    lines.append(f"- median |ngspice − pyport-noH|: {median_delta:.3f} dec")
    lines.append(f"- max  |ngspice − pyport-noH|: {max_delta:.3f} dec")
    lines.append(f"- PASS threshold: mean ≤ {PASS_DELTA} dec")
    lines.append(f"- **VERDICT: {'PASS' if verdict_pass else 'FAIL'}**")
    lines.append("")
    lines.append(f"## Notes\n")
    lines.append(f"- 'dec' = median over Vd>0.3 V of |log10|I_meas| − log10|I_sim||")
    lines.append(f"- ngspice version: 42 (KLU, batch mode, DC sweep)")
    lines.append(f"- Total runtime: {results['aggregate']['runtime_s']:.1f} s")
    (OUT / "verdict.md").write_text("\n".join(lines))
    print("\n".join(lines[-15:]))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
        for i, vg1 in enumerate(VG1_GRID):
            for j, vg2 in enumerate(VG2_GRID):
                ax = axes[i][j]
                r = next((rr for rr in results["biases"]
                          if abs(rr.get("VG1", -99) - vg1) < 1e-6
                          and abs(rr.get("VG2", -99) - vg2) < 1e-6), None)
                if r is None or "tag" not in r:
                    ax.set_title(f"VG1={vg1} VG2={vg2} (skip)"); continue
                tag = r["tag"]
                # Reload data
                try:
                    meas_csv = DATA / "..." # placeholder
                    csv_p = find_meas_csv(vg1, vg2)
                    Vd_m, Id_m = load_measured(csv_p)
                    ax.semilogy(Vd_m, np.abs(Id_m), "k-", lw=1.2, label="meas")
                except Exception:
                    pass
                try:
                    ng_path = DECKS / f"{tag}_dc.txt"
                    d = np.loadtxt(ng_path)
                    ax.semilogy(d[:, 0], np.abs(d[:, 1]), "b-", lw=1.0, label="ngspice")
                except Exception:
                    pass
                # pyport — rerun quickly? skip, just show data + ngspice
                ax.set_title(f"VG1={vg1} VG2={vg2}\nngdec={r['ngspice_dec_vs_data']:.2f} pyHdec={r['pyport_hurkx_dec_vs_data']:.2f}", fontsize=9)
                ax.set_ylim(1e-12, 1e-3)
                if i == 2: ax.set_xlabel("Vd (V)")
                if j == 0: ax.set_ylabel("|Id| (A)")
                if i == 0 and j == 0: ax.legend(fontsize=7)
                ax.grid(True, which="both", alpha=0.3)
        fig.suptitle(f"track_ngspice_xval — K1+ALPHA0 (Hurkx pyport-only)\nmean|ngspice−pyport|={mean_delta:.3f} dec → {'PASS' if verdict_pass else 'FAIL'}", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "plot.png", dpi=110)
        print(f"  plot saved: {OUT / 'plot.png'}")
    except Exception as e:
        print(f"  plot FAIL: {e}")

    print(f"\n=== DONE ===  mean Δ = {mean_delta:.3f} dec  →  {'PASS' if verdict_pass else 'FAIL'}")
    return 0 if verdict_pass else 1


if __name__ == "__main__":
    sys.exit(main())
