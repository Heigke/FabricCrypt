"""z257 — M4 negative-V_G2 leakage suppression probe.

POST_AUDIT_FIX_PLAN_2026-05-11.md §M4.

Goal: predict pyport DC behaviour at V_G2 = −2 V (Sebas slide 12.30's
"drive down the leaky behaviour" operating point), compare leakage
suppression vs V_G2 = 0 V baseline to slide-reported switching ratio
> 10⁵ (≥5 decades).

Pre-registered gate (research_plan/01_LOG.md 2026-05-11):
  PASS if predicted suppression at V_G2=−2 V vs V_G2=0 V is within
  ±1 decade of slide reference (≥5 dec; accept 4–6 dec window).
  FAIL otherwise.

Read-only w.r.t. M1: imports pyport at its current production params
(z96 build_calibrated_models + BJT Bf=5e4); does NOT modify shared
modules.

Pitfall flagged: V_G2 = −2 V is far outside the BSIM4 model's fit
window ([−0.2, 0.5] V). If solver diverges or returns NaN/extreme,
verdict = FAIL with reason "OUT_OF_RANGE".
"""
from __future__ import annotations
import json, time, math, importlib.util
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z257_m4_negative_vg2"
OUT.mkdir(parents=True, exist_ok=True)

# Use z96's production calibrated builder (A.5.cc/dd zero-shift baseline).
sp = importlib.util.spec_from_file_location("z96", ROOT / "scripts/z96_narma10_pilot.py")
z96 = importlib.util.module_from_spec(sp); sp.loader.exec_module(z96)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched


def run_sweep(VG2_value: float, VG1_vals=(0.2, 0.4, 0.6),
              Vd_min=0.0, Vd_max=2.0, Vd_step=0.05):
    """Run pyport DC at one V_G2 value across V_G1 list, V_d sweep.

    Returns dict mapping VG1 -> dict with arrays Vd, Id, converged, Id_off.
    """
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=80)
    M1, M2 = z96.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 5e4

    Vd_seq = torch.tensor(np.arange(Vd_min, Vd_max + Vd_step/2, Vd_step),
                          dtype=torch.float64)
    VG1_t = torch.tensor(list(VG1_vals), dtype=torch.float64)
    VG2_t = torch.full_like(VG1_t, VG2_value)

    print(f"[z257]   V_G2={VG2_value:+.2f}V  V_G1={list(VG1_vals)}  "
          f"V_d={Vd_min:.2f}..{Vd_max:.2f} step {Vd_step:.3f} "
          f"({len(Vd_seq)} pts)", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = forward_2t_batched(cfg, M1, M2, bjt, Vd_seq, VG1_t, VG2_t,
                                 max_iters=30, tol=1e-9, verbose=False)
    dt = time.time() - t0
    Id = out["Id"].cpu().numpy()  # (N=3, T)
    conv = out["converged"].cpu().numpy()
    Vd = Vd_seq.cpu().numpy()
    n_conv = int(conv.sum())
    n_tot = conv.size
    print(f"[z257]   done in {dt:.1f}s  converged {n_conv}/{n_tot}", flush=True)
    res = {}
    for i, vg1 in enumerate(VG1_vals):
        Id_row = Id[i]
        # off-state at V_d = 0 V (first index)
        i_off = float(abs(Id_row[0]))
        res[float(vg1)] = {
            "Vd": Vd.tolist(),
            "Id": Id_row.tolist(),
            "Id_abs": np.abs(Id_row).tolist(),
            "converged": conv[i].tolist(),
            "Id_off_Vd0": i_off,
            "Id_off_log10": float(np.log10(max(i_off, 1e-30))),
            "n_nan": int(np.isnan(Id_row).sum()),
            "n_inf": int(np.isinf(Id_row).sum()),
            "Id_max_abs": float(np.nanmax(np.abs(Id_row))),
        }
    return res


def main():
    t_start = time.time()
    print(f"[z257] starting at {time.strftime('%H:%M:%S')}", flush=True)

    # --- Run V_G2 = 0 V (baseline leaky) ---
    print("[z257] sweep #1: V_G2 = 0 V (baseline, in-window)", flush=True)
    res_zero = run_sweep(0.0)

    # --- Run V_G2 = −2 V (suppressed; OUT-OF-WINDOW probe) ---
    print("[z257] sweep #2: V_G2 = -2 V (out-of-window probe)", flush=True)
    res_neg2 = run_sweep(-2.0)

    # --- Extract leakage (V_d = 0 V, V_G1 = 0.2 V) ---
    i_off_zero = res_zero[0.2]["Id_off_Vd0"]
    i_off_neg2 = res_neg2[0.2]["Id_off_Vd0"]

    # Out-of-range detection
    out_of_range = False
    oor_reasons = []
    for tag, sweep in (("V_G2=0", res_zero), ("V_G2=-2", res_neg2)):
        for vg1, row in sweep.items():
            if row["n_nan"] > 0:
                out_of_range = True
                oor_reasons.append(f"{tag} VG1={vg1}: {row['n_nan']} NaN")
            if row["n_inf"] > 0:
                out_of_range = True
                oor_reasons.append(f"{tag} VG1={vg1}: {row['n_inf']} Inf")
            if row["Id_max_abs"] > 1.0:  # >1 A is unphysical for ~17 µm² cell
                out_of_range = True
                oor_reasons.append(
                    f"{tag} VG1={vg1}: Id_max={row['Id_max_abs']:.2e} A (unphysical)"
                )
            if not all(row["converged"]):
                n_unc = sum(1 for c in row["converged"] if not c)
                oor_reasons.append(
                    f"{tag} VG1={vg1}: {n_unc}/{len(row['converged'])} unconverged"
                )

    # Compute suppression in decades (V_G2=0 leakage / V_G2=-2 leakage)
    if i_off_neg2 > 0 and i_off_zero > 0:
        suppression_dec = math.log10(i_off_zero / i_off_neg2)
    else:
        suppression_dec = float("nan")

    # Slide-12.30 reference: switching ratio > 10⁵ → ≥5 dec suppression.
    # Treat 5.0 dec as central reference; gate accepts ±1 dec → [4, 6].
    slide_target_dec = 5.0
    gate_lo, gate_hi = slide_target_dec - 1.0, slide_target_dec + 1.0

    # Verdict
    if out_of_range:
        verdict = "FAIL"
        reason = "OUT_OF_RANGE — pyport not designed for V_G2 = -2 V regime; " \
                 + "; ".join(oor_reasons[:5])
    elif math.isnan(suppression_dec):
        verdict = "FAIL"
        reason = "NaN/zero leakage current — solver returned nonsense"
    elif gate_lo <= suppression_dec <= gate_hi:
        verdict = "PASS"
        reason = (f"predicted suppression {suppression_dec:.2f} dec is within "
                  f"±1 dec of slide reference {slide_target_dec:.1f} dec")
    else:
        verdict = "FAIL"
        reason = (f"predicted suppression {suppression_dec:.2f} dec is outside "
                  f"slide window [{gate_lo:.1f}, {gate_hi:.1f}] dec "
                  f"(reference {slide_target_dec:.1f} dec from slide 12.30 "
                  f"\"switching ratio > 10⁵\")")

    # --- Summary ---
    summary = {
        "script": "scripts/z257_m4_negative_vg2_probe.py",
        "plan_ref": "POST_AUDIT_FIX_PLAN_2026-05-11.md §M4",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pyport_config": {
            "calibration": "z96 build_calibrated_models (A.5.cc/dd, voff shifts = 0)",
            "BJT_Bf": 5e4,
            "newton_max_iters": 80,
            "use_iii": True, "use_gidl": True, "use_bjt": True,
        },
        "sweep_params": {
            "VG1_vals": [0.2, 0.4, 0.6],
            "VG2_vals": [0.0, -2.0],
            "Vd_min": 0.0, "Vd_max": 2.0, "Vd_step": 0.05,
        },
        "vg2_minus_2v_leakage_pred": i_off_neg2,
        "vg2_zero_v_leakage_pred":   i_off_zero,
        "predicted_suppression_dec": suppression_dec,
        "slide_target_suppression_dec": slide_target_dec,
        "slide_source": "Image 2026-03-20 at 12.30.jpeg — \"Outstanding firing range exceeding 10⁵ on/off ratio\"",
        "gate_window_dec": [gate_lo, gate_hi],
        "out_of_range": out_of_range,
        "out_of_range_reasons": oor_reasons,
        "verdict": verdict,
        "reason": reason,
        "all_VG1_leakage": {
            "VG2=0":  {str(k): v["Id_off_Vd0"] for k, v in res_zero.items()},
            "VG2=-2": {str(k): v["Id_off_Vd0"] for k, v in res_neg2.items()},
        },
        "wallclock_sec": time.time() - t_start,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    # Save full sweeps for plotting
    np.savez(OUT / "sweeps.npz",
             Vd=np.array(res_zero[0.2]["Vd"]),
             Id_VG2_0_VG1_02=np.array(res_zero[0.2]["Id"]),
             Id_VG2_0_VG1_04=np.array(res_zero[0.4]["Id"]),
             Id_VG2_0_VG1_06=np.array(res_zero[0.6]["Id"]),
             Id_VG2_n2_VG1_02=np.array(res_neg2[0.2]["Id"]),
             Id_VG2_n2_VG1_04=np.array(res_neg2[0.4]["Id"]),
             Id_VG2_n2_VG1_06=np.array(res_neg2[0.6]["Id"]))

    print()
    print("=" * 70)
    print(f"[z257] V_G2= 0  V leakage (V_d=0, V_G1=0.2): {i_off_zero:.3e} A "
          f"(log10 = {math.log10(max(i_off_zero,1e-30)):+.2f})")
    print(f"[z257] V_G2=-2  V leakage (V_d=0, V_G1=0.2): {i_off_neg2:.3e} A "
          f"(log10 = {math.log10(max(i_off_neg2,1e-30)):+.2f})")
    print(f"[z257] Predicted suppression: {suppression_dec:.2f} dec")
    print(f"[z257] Slide-12.30 reference: {slide_target_dec:.1f} dec "
          f"(gate window [{gate_lo:.1f}, {gate_hi:.1f}])")
    print(f"[z257] VERDICT: {verdict}")
    print(f"[z257] reason: {reason}")
    print("=" * 70)
    print(f"[z257] summary: {OUT/'summary.json'}")
    print(f"[z257] total wallclock: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
