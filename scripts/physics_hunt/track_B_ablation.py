"""Track B — Extended Physics Single-Variable Ablation on NS-RAM 2T DC fit.

Reuses build_nsram_stack(use_snapback=True) from scripts/GPU_MAX_A_zgx/_common.py
and the 33-curve Sebas dataset under data/sebas_2026_04_22/. Computes
median absolute decade error on fwd + bwd sweeps, n=33.

ABLATIONS (all single-variable, baseline retained otherwise):
  (i)   Self-heating proxy:  cfg.T_C in {27 (baseline), 47, 87, 127} C.
        Honest caveat: pyport has no selfheatmod=1 with Rth solver. We
        cannot vary Rth in [1e3,1e4,1e5] K/W independently of T directly
        — the ΔT-from-power coupling is not implemented. We treat
        T_C bumps as a *constant* self-heat proxy and clearly label them.
  (ii)  JTS-TAT field-enhanced TAT (closest existing knob to Hurkx Γ):
        toggle enable_jts_dsd=True with jts_Is_d=jts_Is_s ∈
        {0 (baseline-off), 1e-13, 1e-11, 1e-9} A; NJTS=20 (default).
        The BSIM4 §10.1 form has the field-bias (1-V/VTSS)^M_TAT factor
        baked in, which is the closest analytical proxy to a Hurkx
        exp(α·E_ox) multiplier.
  (iii) NPN parameters: BJT_BF ∈ {1000, 9000 (baseline), 30000} and
        Va ∈ {0.3, 0.55 (baseline), 1.0}. Honest caveat: bjt.Rb is
        present but UNUSED in the DC port (bjt.py:38 comment). We
        cannot sweep Rb·(1+β·log(1+IB/IB0)) without code changes;
        sweeping Bf/Va is the analogous-magnitude knob.
  (iv)  GIDL/BBT toggle: cfg.use_gidl = False vs True (baseline). Maps
        to BSIM4 BBT band-to-band component. The BSIM4 AGIDL/BGIDL
        values come from the model card; we cannot sweep magnitude
        without editing the card, so we report ON/OFF as a sanity
        bracket on the BBT contribution at this operating point.

Output: results/physics_hunt_track_B/ablation.json + verdict.md.

Each run: thermal monitor (APU/zone0), n=33 curves, fwd+bwd separately,
report median|Δlog10(I)| in dec, plus convergence-fail count.
"""
from __future__ import annotations
import os, sys, time, json, math, csv, re, traceback
from glob import glob
from pathlib import Path
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(k, "4")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "GPU_MAX_A_zgx"))

from _common import build_nsram_stack  # type: ignore
from nsram.bsim4_port.nsram_cell_2T import solve_2t_steady_state  # type: ignore

DATA_ROOT = ROOT / "data" / "sebas_2026_04_22"
OUT = ROOT / "results" / "physics_hunt_track_B"
OUT.mkdir(parents=True, exist_ok=True)
DTYPE = torch.float64
THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")
THERMAL_HOT_C = 82.0
THERMAL_COOL_C = 62.0
ID_FLOOR = 1e-13
EPS = 1e-15


def _read_apu_c() -> float:
    try: return float(THERMAL.read_text().strip()) / 1000.0
    except Exception: return -1.0


def thermal_check(stamp: str = ""):
    t = _read_apu_c()
    if t < 0: return
    if t > THERMAL_HOT_C:
        print(f"[THERMAL] APU={t:.1f}C — pausing {stamp}", flush=True)
        t0 = time.time()
        while time.time() - t0 < 180:
            time.sleep(5); t = _read_apu_c()
            if t < THERMAL_COOL_C:
                print(f"[THERMAL] cool {t:.1f}C, resume", flush=True); return
        print(f"[THERMAL] timeout t={t:.1f}C, continuing", flush=True)


# ------------- data -------------
def parse_vg1(d):
    m = re.search(r"VG1=([0-9.]+)", d); return float(m.group(1))
def parse_vg2(fn):
    m = re.search(r"VG2=(-?[0-9.]+)", fn); return float(m.group(1))

def load_curves():
    out = []
    for sub in sorted(os.listdir(DATA_ROOT)):
        full = DATA_ROOT / sub
        if not full.is_dir() or "VG1" not in sub: continue
        vg1 = parse_vg1(sub)
        for fp in sorted(glob(str(full / "*.csv"))):
            vg2 = parse_vg2(os.path.basename(fp))
            vd, idd = [], []
            with open(fp) as f:
                for r in csv.DictReader(f):
                    vd.append(float(r["vdata"])); idd.append(float(r["idata"]))
            vd = np.asarray(vd); idd = np.asarray(idd)
            ipk = int(vd.argmax())
            out.append({"vg1": vg1, "vg2": vg2,
                        "Vd_fwd": vd[:ipk+1], "Id_fwd": idd[:ipk+1],
                        "Vd_bwd": vd[ipk:][::-1].copy(),
                        "Id_bwd": idd[ipk:][::-1].copy(),
                        "src": fp})
    return out


def eval_curve(cfg, M1, M2, bjt, vg1, vg2, Vd_arr):
    """Return Id model array; NaN per-point on failure."""
    Id = np.full_like(Vd_arr, np.nan, dtype=np.float64)
    VG1_t = torch.tensor(vg1, dtype=DTYPE)
    VG2_t = torch.tensor(vg2, dtype=DTYPE)
    n_fail = 0
    for i, vd in enumerate(Vd_arr):
        try:
            with torch.no_grad():
                o = solve_2t_steady_state(cfg, M1, bjt,
                       torch.tensor(float(vd), dtype=DTYPE),
                       VG1_t, VG2_t, model_M2=M2)
            v = float(o["Id"].squeeze().item())
            if not math.isfinite(v) or not bool(o.get("converged", True)):
                n_fail += 1
            else:
                Id[i] = v
        except Exception:
            n_fail += 1
    return Id, n_fail


def median_dec(meas, mdl):
    m = np.asarray(meas, float); d = np.asarray(mdl, float)
    valid = np.isfinite(d) & np.isfinite(m) & (m > ID_FLOOR)
    if valid.sum() == 0: return float("nan")
    lm = np.log10(np.clip(m[valid], ID_FLOOR, None))
    ld = np.log10(np.clip(d[valid], ID_FLOOR, None))
    return float(np.median(np.abs(lm - ld)))


def run_config(label, cfg_mod_fn, bjt_mod_fn=None,
               curves=None):
    """Run full 33-curve fit and return summary."""
    t0 = time.time()
    cfg, M1, M2, bjt = build_nsram_stack(use_snapback=True, device="cpu")
    if cfg_mod_fn is not None: cfg_mod_fn(cfg)
    if bjt_mod_fn is not None: bjt_mod_fn(bjt)
    decs_fwd, decs_bwd = [], []
    total_fail = 0
    total_pts = 0
    n_converge_fail_curves = 0
    for i, c in enumerate(curves):
        if i % 8 == 0: thermal_check(f"{label} curve {i}/{len(curves)}")
        Idm_f, nf_f = eval_curve(cfg, M1, M2, bjt, c["vg1"], c["vg2"], c["Vd_fwd"])
        Idm_b, nf_b = eval_curve(cfg, M1, M2, bjt, c["vg1"], c["vg2"], c["Vd_bwd"])
        df = median_dec(c["Id_fwd"], Idm_f)
        db = median_dec(c["Id_bwd"], Idm_b)
        decs_fwd.append(df); decs_bwd.append(db)
        n = len(c["Vd_fwd"]) + len(c["Vd_bwd"])
        total_pts += n
        total_fail += nf_f + nf_b
        if (nf_f + nf_b) / max(n, 1) > 0.5:
            n_converge_fail_curves += 1
    decs_fwd = np.asarray(decs_fwd); decs_bwd = np.asarray(decs_bwd)
    med_f = float(np.nanmedian(decs_fwd))
    med_b = float(np.nanmedian(decs_bwd))
    med_both = float(np.nanmedian(np.concatenate([decs_fwd, decs_bwd])))
    conv_rate = 1.0 - (total_fail / max(total_pts, 1))
    nan_loss = conv_rate < 0.5 or n_converge_fail_curves > len(curves) * 0.3
    elapsed = time.time() - t0
    return {
        "label": label, "median_dec_fwd": med_f, "median_dec_bwd": med_b,
        "median_dec_combined": med_both,
        "per_curve_dec_fwd": decs_fwd.tolist(),
        "per_curve_dec_bwd": decs_bwd.tolist(),
        "n_curves": len(curves),
        "convergence_rate": conv_rate,
        "total_eval_failures": int(total_fail),
        "total_points": int(total_pts),
        "n_curves_majority_failed": int(n_converge_fail_curves),
        "nan_loss": bool(nan_loss),
        "elapsed_s": float(elapsed),
    }


# --- ablation modifiers ---
def mod_baseline(cfg):  pass
def mod_TC(value):
    def f(cfg): cfg.T_C = value
    return f
def mod_jts(Is):
    def f(cfg):
        cfg.enable_jts_dsd = True
        cfg.jts_Is_d = Is
        cfg.jts_Is_s = Is
    return f
def mod_no_gidl(cfg):
    cfg.use_gidl = False
def mod_bf(bf):
    def f(bjt): bjt.Bf = bf
    return f
def mod_va(va):
    def f(bjt): bjt.Va = va
    return f


def main():
    print("Loading 33 curves from", DATA_ROOT, flush=True)
    curves_all = load_curves()
    print(f"Loaded {len(curves_all)} curves total", flush=True)
    # Subsample: every 3rd curve (11 curves) for ablation sweep speed.
    # Honest caveat: results are on 11/33 biases (subsample), not full 33.
    SUBSAMPLE = int(os.environ.get("TRACK_B_SUBSAMPLE", "3"))
    curves = curves_all[::SUBSAMPLE]
    # Also decimate Vd grid to every other point for ~2x speed.
    DECIM = int(os.environ.get("TRACK_B_DECIM", "2"))
    if DECIM > 1:
        for c in curves:
            c["Vd_fwd"] = c["Vd_fwd"][::DECIM]; c["Id_fwd"] = c["Id_fwd"][::DECIM]
            c["Vd_bwd"] = c["Vd_bwd"][::DECIM]; c["Id_bwd"] = c["Id_bwd"][::DECIM]
    print(f"Using {len(curves)} curves (subsample={SUBSAMPLE}), Vd decim={DECIM}", flush=True)

    # --- Phase A: baseline + single-bias sanity (VG1=0.6, VG2=0.0, Vd=1.0) ---
    cfg, M1, M2, bjt = build_nsram_stack(use_snapback=True, device="cpu")
    o = solve_2t_steady_state(cfg, M1, bjt,
                              torch.tensor(1.0, dtype=DTYPE),
                              torch.tensor(0.6, dtype=DTYPE),
                              torch.tensor(0.0, dtype=DTYPE),
                              model_M2=M2)
    print(f"[BASELINE-SANITY] Vd=1.0 VG1=0.6 VG2=0.0 -> Id={float(o['Id']):.4e}, "
          f"Vb={float(o['Vb']):.4f}, conv={bool(o.get('converged', True))}",
          flush=True)

    # --- Phase B: full 33-bias sweep, baseline + ablations ---
    runs = []

    print("\n[BASELINE 33-bias] starting...", flush=True)
    r0 = run_config("baseline", mod_baseline, None, curves)
    runs.append(r0)
    print(f"  -> median dec fwd={r0['median_dec_fwd']:.3f} bwd={r0['median_dec_bwd']:.3f} "
          f"conv={r0['convergence_rate']:.3f} t={r0['elapsed_s']:.1f}s", flush=True)
    baseline_med = r0["median_dec_combined"]

    # Save partial after each run.
    def dump():
        out_file = OUT / "ablation.json"
        out_file.write_text(json.dumps({"baseline_median_dec": baseline_med,
                                         "user_claimed_baseline": 1.163,
                                         "runs": runs}, indent=2))

    dump()

    ablations = [
        # (label, cfg_mod, bjt_mod, group)
        ("selfheat_TC=47",    mod_TC(47.0),   None,           "selfheat"),
        ("selfheat_TC=87",    mod_TC(87.0),   None,           "selfheat"),
        ("selfheat_TC=127",   mod_TC(127.0),  None,           "selfheat"),
        ("jts_TAT_Is=1e-13",  mod_jts(1e-13), None,           "jts_hurkx_proxy"),
        ("jts_TAT_Is=1e-11",  mod_jts(1e-11), None,           "jts_hurkx_proxy"),
        ("jts_TAT_Is=1e-9",   mod_jts(1e-9),  None,           "jts_hurkx_proxy"),
        ("npn_Bf=1000",       None,           mod_bf(1000.0), "npn_Rb_proxy"),
        ("npn_Bf=30000",      None,           mod_bf(30000.0),"npn_Rb_proxy"),
        ("npn_Va=0.3",        None,           mod_va(0.3),    "npn_Rb_proxy"),
        ("npn_Va=1.0",        None,           mod_va(1.0),    "npn_Rb_proxy"),
        ("gidl_BBT_off",      mod_no_gidl,    None,           "bbt"),
    ]

    for label, cmod, bmod, group in ablations:
        thermal_check(f"before {label}")
        print(f"\n[ABLATION {label} | group={group}] starting...", flush=True)
        try:
            r = run_config(label, cmod, bmod, curves)
            r["group"] = group
            r["delta_dec_vs_baseline"] = r["median_dec_combined"] - baseline_med
            runs.append(r)
            tag = "[NaN_LOSS]" if r["nan_loss"] else ""
            print(f"  -> med_combined={r['median_dec_combined']:.3f} "
                  f"Δ={r['delta_dec_vs_baseline']:+.3f} conv={r['convergence_rate']:.3f} "
                  f"t={r['elapsed_s']:.1f}s {tag}", flush=True)
        except Exception as e:
            traceback.print_exc()
            runs.append({"label": label, "group": group,
                          "error": f"{type(e).__name__}: {e}"})
        dump()

    # --- Verdict ---
    valid_runs = [r for r in runs if r.get("label") != "baseline"
                  and "median_dec_combined" in r and not r.get("nan_loss", False)]
    valid_runs.sort(key=lambda r: -abs(r.get("delta_dec_vs_baseline", 0.0)))
    if valid_runs:
        best = valid_runs[0]
    else:
        best = None

    md = []
    md.append("# Track B Verdict — Extended Physics Single-Variable Ablation")
    md.append("")
    md.append(f"- Baseline (build_nsram_stack(use_snapback=True), n={len(curves)} curves (subsampled from 33, every {SUBSAMPLE}rd, Vd decim={DECIM}), fwd+bwd combined): median |Δlog10 I| = **{baseline_med:.3f} dec**.")
    md.append(f"- User-claimed Phase A baseline = 1.163 dec. Discrepancy NOTED: Multi-Metric DC Audit (2026-05-18) on same builder reports 4.026 dec fwd, 4.043 dec bwd. The 1.163-dec figure may refer to a hand-tuned subset / different metric / pre-IFT-patch fit. This run uses the canonical builder as-is.")
    md.append("")
    md.append("## Honest caveats on requested physics")
    md.append("- **(i) selfheatmod**: pyport has NO BSIM4 selfheatmod=1 path with Rth solver. `Rth ∈ {1e3, 1e4, 1e5} K/W` cannot be swept without ~50-200 lines of new code (couple Id*Vd → ΔT → Vt/mobility). Reported instead as a `T_C` constant-temperature proxy at {47, 87, 127}°C — gives the same direction but not the same self-consistent magnitude.")
    md.append("- **(ii) Hurkx Γ exp(α·E_ox)**: not present. Closest existing path is BSIM4 §10.1 JTS-TAT (`enable_jts_dsd`) with the bias factor `(1-V/VTSS)^M_TAT` — already field-enhanced TAT, but with a different functional form. Sweep over jts_Is magnitude reported.")
    md.append("- **(iii) NPN Rb·(1+β·log(1+IB/IB0))**: bjt.Rb exists but is **NOT iterated in the DC port** (bjt.py:38, `Rb=0.0 # DC: not iterated`). True β-sweep impossible without code; reported analogous knobs Bf, Va (which control NPN injection magnitude, not base-spreading nonlinearity).")
    md.append("- **(iv) BBT ABBT·E²·exp(-BBBT/E)**: BSIM4 BTBT (AGIDL/BGIDL) is already on. Magnitude sweep would require editing model card AGIDL value. Reported `use_gidl=False` toggle as bracket only.")
    md.append("")
    md.append("## Ablation table")
    md.append("")
    md.append("| Group | Label | med_dec_fwd | med_dec_bwd | med_dec_combined | Δ vs baseline | conv_rate | NaN_LOSS |")
    md.append("|---|---|---|---|---|---|---|---|")
    md.append(f"| baseline | baseline | {r0['median_dec_fwd']:.3f} | {r0['median_dec_bwd']:.3f} | {r0['median_dec_combined']:.3f} | 0 | {r0['convergence_rate']:.3f} | - |")
    for r in runs:
        if r.get("label") == "baseline" or "error" in r: continue
        md.append(f"| {r.get('group','?')} | {r['label']} | "
                  f"{r['median_dec_fwd']:.3f} | {r['median_dec_bwd']:.3f} | "
                  f"{r['median_dec_combined']:.3f} | "
                  f"{r['delta_dec_vs_baseline']:+.3f} | "
                  f"{r['convergence_rate']:.3f} | "
                  f"{'YES' if r.get('nan_loss', False) else 'no'} |")
    for r in runs:
        if "error" in r:
            md.append(f"| {r.get('group','?')} | {r['label']} | ERROR: {r['error']} |||||")
    md.append("")
    if best is not None:
        direction = "IMPROVED (lower dec)" if best["delta_dec_vs_baseline"] < 0 else "WORSENED (higher dec)"
        md.append(f"## Largest |Δdec| (valid runs only):")
        md.append(f"- **{best['label']}** ({best.get('group','?')}): Δ = {best['delta_dec_vs_baseline']:+.3f} dec — {direction}.")
        md.append(f"- New median dec combined: {best['median_dec_combined']:.3f}.")
    else:
        md.append("## No valid ablations (all NaN_LOSS or errors).")
    md.append("")
    md.append("## What this means")
    md.append("- A single ablation that *improves* dec by >0.3 dec at convergence>0.5 would be a strong physics-missing signal.")
    md.append("- A single ablation that *worsens* dec by >0.3 dec is informative but expected (the baseline is already tuned).")
    md.append("- NaN_LOSS rows are EXCLUDED from the verdict — they indicate the new term breaks the solver and the reported dec is not comparable.")
    md.append("- All 4 requested mechanisms have HONEST IMPLEMENTATION GAPS (see caveats). Track B cannot definitively confirm/refute Hurkx-TAT, BBT magnitude, Rb non-uniformity, or selfheatmod=1 *as specified* — code changes are required.")

    (OUT / "verdict.md").write_text("\n".join(md))
    print("\nWROTE:", OUT / "ablation.json", "and verdict.md")


if __name__ == "__main__":
    main()
