"""z376 — R-55a Full Sebas Zenodo Topology Rebuild.

LAST designed approach before retracting the model program. 6 prior topology
fixes (R-43, R-45, R-47, R-49, R-52, R-53) all failed to reproduce the
2-3 decade snapback fold in silicon. O69 oracle convergence + R-55 zoom dive
point to FIVE coupled topology gaps in pyport vs Sebas Zenodo deck:

  1. D3 zener G1→B  (gate-to-body avalanche, missing entirely)
  2. BVPar = 3.5 - 1.5·V_G1  (Tsinghua avalanche-BV V_G1 dependence)
  3. nbvPar = 9 - 0.55/V_G1  (Tsinghua avalanche-sharpness V_G1 dependence)
  4. BJT params: VA=100, IS=5e-9, Bf=10000  (Sebas canonical, was 0.903/5.95e-12/991)
  5. M3 BSS145 G2→B  (discrete sub-Vth body-leak NMOS, missing)

Ablation sweep: each branch toggled INDIVIDUALLY, then ALL ON together.

Pre-registered gates (logged to nsram/proposal_2026_05/01_LOG.md BEFORE run):
  INFRA      : 6 conditions complete in <60 min, no nan
  DISCOVERY  : cell-wide < 0.85 dec AND VG1=0.6 model_jump > 0.5 dec
  AMBITIOUS  : cell-wide < 0.50 dec AND VG1=0.6 model_jump > 1.5 dec
  KILL-SHOT  : ALL-ON ≥ baseline (0.965 dec, fold=0.02) → MODEL RETRACT

Baseline params (R-46 z365 per-VG1 BBO best, except BJT overridden per R-55a):
  x_best = [Bf_020, iii_020, log10Rs_020, Bf_040, ..., log10Rs_060]
         = [1889.88, 1.84, 9.17, 1092.27, 1.52, 9.90, 417.63, 0.90, 6.78]
  BJT override: Va=100, Is=5e-9, Bf=10000  (Sebas canonical)

Outputs: results/z376_R55a/{summary.json, ablation_heatmap.png,
         snapback_with_full_topology.png, run.log}
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys, json, re, math, csv, importlib.util, time, traceback
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
OUT = ROOT / "results/z376_R55a"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"
LOG_PATH = OUT / "run.log"


def _log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ── Shared utilities (lifted from z372 template) ─────────────────────────
@contextmanager
def patch_sd_scaled(sd, overrides):
    if not overrides:
        yield; return
    saved = {}
    try:
        for k, v in overrides.items():
            saved[k] = sd.scaled.get(k, None)
            sd.scaled[k] = float(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None: sd.scaled.pop(k, None)
            else: sd.scaled[k] = v


def load_sebas_params():
    path = DATA / "2Tcell_BSIM_param_DC.csv"
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try: row[k] = float(v)
                except ValueError: row[k] = float("nan")
            rows.append(row)
    return rows


BRANCH_FLAT = {
    0.4: {"ETAB": 1.9,  "K1": 0.53825, "ALPHA0": 7.842e-05, "BETA0": 19.0, "NFACTOR": 6.0, "trise": 10.59},
    0.6: {"ETAB": 2.5,  "K1": 0.41825, "ALPHA0": 7.842e-05, "BETA0": 20.0, "NFACTOR": 6.0, "trise": 9.04},
}
M2_STATIC = {"k1": 0.63825, "k2": -0.070435, "etab": -0.086777, "beta0": 18.0}


def find_or_impute_row(rows, VG1, VG2, atol=1e-3):
    target = None
    for r in rows:
        if abs(r["VG1"] - VG1) < atol and abs(r["VG2"] - VG2) < atol:
            target = dict(r); break
    if target is None: return None
    if math.isnan(target.get("K1", float("nan"))):
        branch = BRANCH_FLAT.get(round(VG1, 2))
        if branch is None: return target
        for k, v in branch.items():
            target[k] = float(v)
    return target


def make_overrides(row):
    if row is None: return None, None
    P_M1 = {}
    for ck, pk in (("ETAB","etab"),("K1","k1"),("ALPHA0","alpha0"),("BETA0","beta0")):
        if not math.isnan(row.get(ck, float("nan"))): P_M1[pk] = float(row[ck])
    P_M2 = {}
    if not math.isnan(row.get("NFACTOR", float("nan"))): P_M2["nfactor"] = float(row["NFACTOR"])
    for k, v in M2_STATIC.items():
        P_M2.setdefault(k, float(v))
    return (P_M1 or None), (P_M2 or None)


def load_measured(vg1, vg2):
    sub = DATA / f"2vHCa-2 I-Vs@VG2 VG1={vg1} vnwell=2"
    pat = re.compile(rf"VG2={vg2:.2f}_VG={vg1}")
    for f in sorted(sub.glob("*.csv")):
        if pat.search(f.name):
            d = np.loadtxt(f, delimiter=",", skiprows=1)
            return d[:, 0], np.abs(d[:, 1]), f.name
    raise FileNotFoundError(f"no csv for VG1={vg1} VG2={vg2}")


def build_base():
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    cfg.body_pdiode_to = "vnwell"
    cfg.use_well_diode = True
    cfg.vnwell = 2.0
    cfg.body_pdiode_Js = 5.3675e-7 / 22e-12
    cfg.body_pdiode_n = 1.0535
    cfg.body_pdiode_Rs = 1.0e6
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    # R-55a finding #4: Sebas canonical BJT params (override R-46 BBO best)
    bjt.Va = 100.0
    bjt.Is = 5e-9
    bjt.Bf = 10000.0
    return cfg, M1, M2, bjt


# R-46 z365 per-VG1 best  (driver params only; BJT is overridden above)
X_BEST = [1889.88, 1.8447, 9.1722,
          1092.27, 1.5152, 9.8983,
           417.63, 0.9036, 6.7846]
PER_VG1 = {0.2: (X_BEST[0], X_BEST[1], 10**X_BEST[2]),
           0.4: (X_BEST[3], X_BEST[4], 10**X_BEST[5]),
           0.6: (X_BEST[6], X_BEST[7], 10**X_BEST[8])}
TARGETS = [(0.2, 0.10), (0.4, 0.20), (0.6, 0.20)]   # same as z372

CONDITIONS = [
    # (label, cfg_setter)
    ("baseline",     lambda c: None),
    ("d3_only",      lambda c: setattr(c, "use_d3_zener_g1b", True)),
    ("bvpar_only",   lambda c: (setattr(c, "use_dbd_avalanche", True),
                                 setattr(c, "use_bvpar_vg1_dep", True))),
    ("nbv_only",     lambda c: (setattr(c, "use_dbd_avalanche", True),
                                 setattr(c, "use_nbv_vg1_dep", True))),
    ("m3_only",      lambda c: setattr(c, "use_m3_bss145", True)),
    ("all_on",       lambda c: (setattr(c, "use_d3_zener_g1b", True),
                                 setattr(c, "use_dbd_avalanche", True),
                                 setattr(c, "use_bvpar_vg1_dep", True),
                                 setattr(c, "use_nbv_vg1_dep", True),
                                 setattr(c, "use_m3_bss145", True))),
]


def metrics_one(Vd_m, Id_m, Id_p):
    """Replicate z372 metric extraction."""
    mask = (Id_m > 1e-15) & (Id_p > 1e-15) & np.isfinite(Id_p)
    rmse_dec = (float(np.sqrt(np.mean((np.log10(Id_p[mask]) - np.log10(Id_m[mask]))**2)))
                if mask.sum() >= 3 else float("nan"))
    dlog = np.diff(np.log10(np.maximum(Id_m, 1e-15)))
    Vmid = 0.5 * (Vd_m[1:] + Vd_m[:-1])
    valid_knee = Vmid >= 0.5
    if valid_knee.any() and len(dlog) > 0:
        dlog_masked = np.where(valid_knee, dlog, -np.inf)
        knee_idx = int(np.argmax(dlog_masked)) + 1
    else:
        knee_idx = None
    jump_dec = float(dlog.max()) if len(dlog) > 0 else None
    if knee_idx is not None and knee_idx < len(Id_p):
        lo = max(0, knee_idx-3); hi = min(len(Id_p), knee_idx+3)
        dlog_p_window = np.diff(np.log10(np.maximum(Id_p[lo:hi], 1e-15)))
        model_jump = float(dlog_p_window.max()) if len(dlog_p_window) else 0.0
    else:
        model_jump = float("nan")
    return rmse_dec, jump_dec, model_jump, int(mask.sum())


def run_condition(label, setter, sebas_rows):
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    cfg, M1, M2, bjt = build_base()
    setter(cfg)
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    per_target = []
    for (vg1, vg2) in TARGETS:
        try:
            Vd_m, Id_m, fname = load_measured(vg1, vg2)
            # R-46 per-VG1 driver knobs (NOT BJT; BJT is locked to R-55a canonical)
            _, iii, Rs = PER_VG1[vg1]
            cfg.iii_body_gain = iii
            cfg.vnwell_Rs = Rs
            row = find_or_impute_row(sebas_rows, vg1, vg2)
            P_M1, P_M2 = make_overrides(row)
            Vd_t = torch.tensor(Vd_m, dtype=torch.float64)
            t0 = time.time()
            with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_t,
                                 VG1=torch.tensor(vg1, dtype=torch.float64),
                                 VG2=torch.tensor(vg2, dtype=torch.float64),
                                 warm_start=True)
            dt = time.time() - t0
            Id_p = np.abs(out["Id"].detach().cpu().numpy())
            has_nan = bool(np.any(~np.isfinite(Id_p)))
            rmse, meas_jump, model_jump, npts = metrics_one(Vd_m, Id_m, Id_p)
            per_target.append({
                "VG1": vg1, "VG2": vg2,
                "rmse_dec": rmse, "meas_jump_dec": meas_jump,
                "model_jump_dec": model_jump, "n_pts": npts,
                "has_nan": has_nan, "elapsed_s": dt,
                "Id_p": Id_p.tolist(),
                "Vd": Vd_m.tolist(),
                "Id_m": Id_m.tolist(),
            })
            _log(f"  {label} VG1={vg1} VG2={vg2}: rmse={rmse:.3f} dec  "
                 f"jump(meas/model)={meas_jump:.2f}/{model_jump:.2f}  "
                 f"nan={has_nan}  {dt:.1f}s")
        except Exception as e:
            _log(f"  {label} VG1={vg1} VG2={vg2}: EXCEPTION {e}")
            per_target.append({
                "VG1": vg1, "VG2": vg2,
                "rmse_dec": float("nan"), "meas_jump_dec": None,
                "model_jump_dec": float("nan"), "n_pts": 0,
                "has_nan": True, "elapsed_s": 0.0,
                "error": str(e),
            })
    # Cell-wide aggregates (drop nan)
    rmses = [t["rmse_dec"] for t in per_target if not math.isnan(t["rmse_dec"])]
    cell_med = float(np.median(rmses)) if rmses else float("nan")
    cell_p90 = float(np.percentile(rmses, 90)) if rmses else float("nan")
    vg06 = next((t for t in per_target if t["VG1"] == 0.6), None)
    vg06_jump = float(vg06["model_jump_dec"]) if vg06 else float("nan")
    any_nan = any(t.get("has_nan", False) for t in per_target)
    return {
        "label": label,
        "per_target": per_target,
        "cell_median_dec": cell_med,
        "cell_p90_dec": cell_p90,
        "vg1_0p6_model_jump_dec": vg06_jump,
        "any_nan": any_nan,
    }


def plot_ablation_heatmap(all_results):
    labels = [r["label"] for r in all_results]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cell_med = [r["cell_median_dec"] for r in all_results]
    jumps    = [r["vg1_0p6_model_jump_dec"] for r in all_results]
    bars0 = axes[0].bar(labels, cell_med, color="C0")
    axes[0].axhline(0.965, color="gray", ls=":", lw=1, label="baseline R-46 cell-wide (0.965)")
    axes[0].axhline(0.85, color="green", ls="--", lw=1, label="DISCOVERY gate (0.85)")
    axes[0].axhline(0.50, color="purple", ls="--", lw=1, label="AMBITIOUS gate (0.50)")
    axes[0].set_ylabel("Cell-wide median RMSE [dec]")
    axes[0].set_title("Lower is better")
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].legend(fontsize=7, loc="upper right")
    bars1 = axes[1].bar(labels, jumps, color="C1")
    axes[1].axhline(0.02, color="gray", ls=":", lw=1, label="baseline jump (0.02)")
    axes[1].axhline(0.5, color="green", ls="--", lw=1, label="DISCOVERY (0.5)")
    axes[1].axhline(1.5, color="purple", ls="--", lw=1, label="AMBITIOUS (1.5)")
    axes[1].set_ylabel("Model snapback jump @ VG1=0.6 [dec]")
    axes[1].set_title("Higher is better")
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].legend(fontsize=7, loc="upper right")
    fig.suptitle("R-55a — Five-finding ablation (last designed shot)")
    fig.tight_layout()
    out_png = OUT / "ablation_heatmap.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"wrote {out_png}")


def plot_snapback_with_all_on(all_on_result):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, t in zip(axes, all_on_result["per_target"]):
        if "Id_p" not in t:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes); continue
        Vd = np.array(t["Vd"]); Id_m = np.array(t["Id_m"]); Id_p = np.array(t["Id_p"])
        ax.semilogy(Vd, np.maximum(Id_m, 1e-15), "k.", ms=4, label="measured (Sebas)")
        ax.semilogy(Vd, np.maximum(Id_p, 1e-15), "r-", lw=1.6, label="pyport ALL-ON (R-55a)")
        ax.set_xlabel("Vd (V)"); ax.set_ylabel("|Id| (A)")
        ax.set_ylim(1e-13, 1e-2); ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"VG1={t['VG1']}, VG2=+{t['VG2']:.2f}\n"
                     f"RMSE={t['rmse_dec']:.2f} dec | "
                     f"meas-jump={t['meas_jump_dec']:.1f} / model-jump={t['model_jump_dec']:.1f} dec",
                     fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
    rmses = [t["rmse_dec"] for t in all_on_result["per_target"]]
    fig.suptitle(f"R-55a ALL-ON snapback vs Sebas — cell-med={all_on_result['cell_median_dec']:.2f} dec  "
                 f"({rmses[0]:.2f}/{rmses[1]:.2f}/{rmses[2]:.2f})", fontsize=11, y=1.00)
    fig.tight_layout()
    out_png = OUT / "snapback_with_full_topology.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"wrote {out_png}")


def verdict(all_results):
    baseline = next(r for r in all_results if r["label"] == "baseline")
    all_on   = next(r for r in all_results if r["label"] == "all_on")
    base_med, base_jump = baseline["cell_median_dec"], baseline["vg1_0p6_model_jump_dec"]
    aon_med, aon_jump = all_on["cell_median_dec"], all_on["vg1_0p6_model_jump_dec"]
    infra_ok = not any(r["any_nan"] for r in all_results)
    discovery = (aon_med < 0.85) and (aon_jump > 0.5)
    ambitious = (aon_med < 0.50) and (aon_jump > 1.5)
    # Kill-shot: ALL-ON does NO better than baseline
    kill_shot = (aon_med >= base_med - 0.01) and (aon_jump <= base_jump + 0.01)
    return {
        "infra_ok": infra_ok,
        "discovery": discovery,
        "ambitious": ambitious,
        "kill_shot": kill_shot,
        "baseline_cell_med_dec": base_med,
        "baseline_vg06_jump_dec": base_jump,
        "all_on_cell_med_dec": aon_med,
        "all_on_vg06_jump_dec": aon_jump,
    }


def main():
    t_start = time.time()
    _log(f"=== z376 R-55a Full Zenodo Topology Rebuild ===")
    _log(f"Conditions: {[c[0] for c in CONDITIONS]}")
    _log(f"Targets: {TARGETS}")
    sebas_rows = load_sebas_params()
    all_results = []
    for (label, setter) in CONDITIONS:
        _log(f"[run] {label}")
        try:
            r = run_condition(label, setter, sebas_rows)
        except Exception as e:
            _log(f"FATAL on {label}: {e}")
            _log(traceback.format_exc())
            r = {"label": label, "per_target": [], "cell_median_dec": float("nan"),
                 "cell_p90_dec": float("nan"), "vg1_0p6_model_jump_dec": float("nan"),
                 "any_nan": True, "error": str(e)}
        all_results.append(r)
        _log(f"  → cell-med={r['cell_median_dec']:.3f} dec  "
             f"vg06-jump={r['vg1_0p6_model_jump_dec']:.2f} dec  "
             f"nan={r['any_nan']}")

    v = verdict(all_results)
    elapsed = time.time() - t_start
    _log(f"=== Verdict ===")
    _log(f"INFRA_OK:   {v['infra_ok']}  ({elapsed:.0f}s total)")
    _log(f"DISCOVERY:  {v['discovery']}  (all_on_med={v['all_on_cell_med_dec']:.3f}, jump={v['all_on_vg06_jump_dec']:.2f})")
    _log(f"AMBITIOUS:  {v['ambitious']}")
    _log(f"KILL-SHOT:  {v['kill_shot']}  "
         f"(baseline_med={v['baseline_cell_med_dec']:.3f}, jump={v['baseline_vg06_jump_dec']:.2f})")

    try:
        plot_ablation_heatmap(all_results)
    except Exception as e:
        _log(f"plot_ablation_heatmap failed: {e}")
    try:
        all_on = next(r for r in all_results if r["label"] == "all_on")
        plot_snapback_with_all_on(all_on)
    except Exception as e:
        _log(f"plot_snapback_with_all_on failed: {e}")

    # Trim Id arrays from saved summary to keep file small? Keep, but JSON-safe.
    summary = {
        "script": "z376_R55a_full_zenodo_port",
        "task": "R-55a: Full Sebas Zenodo topology rebuild (5 couplings). LAST designed shot.",
        "elapsed_s": elapsed,
        "conditions": [c[0] for c in CONDITIONS],
        "results": all_results,
        "verdict": v,
        "gates": {
            "INFRA": "6 conditions complete in <60min, no nan",
            "DISCOVERY": "cell-wide < 0.85 AND VG1=0.6 jump > 0.5",
            "AMBITIOUS": "cell-wide < 0.50 AND VG1=0.6 jump > 1.5",
            "KILL_SHOT": "ALL-ON no better than baseline (0.965, 0.02) → MODEL RETRACT",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"wrote {OUT/'summary.json'}")

    # Append HONEST verdict to 01_LOG.md (≤10 lines)
    log_md = ROOT / "nsram/proposal_2026_05/01_LOG.md"
    if log_md.exists():
        msg = []
        msg.append(f"\n## 2026-05-14 z376 R-55a Full Zenodo Topology Rebuild — verdict")
        msg.append(f"Ran 6 ablation conditions (baseline + 4 singletons + ALL-ON). Elapsed {elapsed:.0f}s.")
        msg.append(f"Baseline:  cell-med={v['baseline_cell_med_dec']:.3f} dec, VG1=0.6 jump={v['baseline_vg06_jump_dec']:.2f} dec.")
        msg.append(f"ALL-ON:    cell-med={v['all_on_cell_med_dec']:.3f} dec, VG1=0.6 jump={v['all_on_vg06_jump_dec']:.2f} dec.")
        msg.append(f"INFRA={v['infra_ok']}, DISCOVERY={v['discovery']}, AMBITIOUS={v['ambitious']}, KILL-SHOT={v['kill_shot']}.")
        if v["kill_shot"]:
            msg.append(f"→ KILL-SHOT TRIGGERED. R-55a (last designed shot) failed to beat R-46 baseline.")
            msg.append(f"→ Per pre-registration: MODEL PROGRAM RETRACT signal.")
        elif v["discovery"] or v["ambitious"]:
            msg.append(f"→ Gate(s) met. R-55a is structurally productive.")
        else:
            msg.append(f"→ Inconclusive: ALL-ON improved over baseline but did not clear DISCOVERY gate.")
        msg.append(f"Artifacts: results/z376_R55a/{{summary.json, ablation_heatmap.png, snapback_with_full_topology.png}}.")
        with open(log_md, "a") as f:
            f.write("\n".join(msg) + "\n")
        _log(f"appended verdict to {log_md}")

    _log("DONE")


if __name__ == "__main__":
    main()
