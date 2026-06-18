"""Unified transplant eval — runs locally on ikaros.

Trains a 'daedalus' model on the daedalus data, then evaluates it on
ikaros's held-out windows for C1, C2, C3. Computes the gap vs the
'self' (ikaros-trained) model.

Inputs:
  results/.../embodiment5/c1_ikaros/c1_ikaros_data.npy
  results/.../embodiment5/c1_daedalus_data.npy           (copied from daedalus)
  results/.../embodiment5/c1_ikaros/c1_ikaros_model.npz  (already trained)
  results/.../embodiment5/c2_ikaros/c2_ikaros_ae.npz     (already trained)
  results/.../embodiment5/c3_ikaros/c3_ikaros_policy.npz (already trained)

Outputs all transplant results + a unified summary JSON.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the C1/C2/C3 modules to reuse train/eval functions
import c1_self_prediction as c1
import c2_self_anomaly as c2
import c3_thermal_survival as c3

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT_DIR = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment5"

IKAROS_DATA = OUT_DIR / "c1_ikaros" / "c1_ikaros_data.npy"
DAEDALUS_DATA = OUT_DIR / "c1_daedalus_data.npy"


# ---------- C1: train daedalus model, eval on ikaros ---------------------
def train_daedalus_c1_model():
    """Train the C1 predictor on daedalus's data; save weights."""
    data = np.load(DAEDALUS_DATA)
    print(f"[C1-train-daed] daedalus data shape {data.shape}")
    seeds = (0, 1, 2, 3, 4)
    Ws, mus, sds = [], [], []
    for seed in seeds:
        (Xtr, Ytr), _ = c1.make_windows(data, c1.N_TRAIN, c1.N_TEST, seed=seed)
        mu, sd = c1.fit_scaler(Xtr)
        Xtr_s = c1.apply_scaler(Xtr, mu, sd); Ytr_s = c1.apply_scaler(Ytr, mu, sd)
        model = c1.TinyARPred(seed=seed)
        model.fit(Xtr_s, Ytr_s, lr=0.01, epochs=200, verbose=False)
        Ws.append(model.W.copy()); mus.append(mu); sds.append(sd)
    out_npz = OUT_DIR / "c1_daedalus_model.npz"
    np.savez(out_npz, W=np.stack(Ws), mu=np.stack(mus), sd=np.stack(sds),
              seeds=np.array(list(seeds)))
    print(f"[C1-train-daed] saved {out_npz}")
    return out_npz


def transplant_c1():
    other_npz = OUT_DIR / "c1_daedalus_model.npz"
    if not other_npz.exists():
        train_daedalus_c1_model()
    pkg = np.load(other_npz)
    Ws, mus, sds = pkg["W"], pkg["mu"], pkg["sd"]
    data = np.load(IKAROS_DATA)
    rows = []
    for i, seed in enumerate(range(5)):
        if i >= len(Ws): break
        _, (Xte, Yte) = c1.make_windows(data, c1.N_TRAIN, c1.N_TEST, seed=seed)
        Xte_s = c1.apply_scaler(Xte, mus[i], sds[i])
        model = c1.TinyARPred(seed=seed); model.W = Ws[i]
        pred_s = model.predict(Xte_s)
        pred = c1.inverse_scaler(pred_s, mus[i], sds[i])
        nr, _ = c1.nrmse(pred, Yte)
        rows.append({"seed": int(seed), "transplant_nrmse": nr})
    return rows


# ---------- C2: train daedalus AE on daedalus data, eval on ikaros -----
def train_daedalus_c2_ae():
    data = np.load(DAEDALUS_DATA)
    print(f"[C2-train-daed] daedalus data shape {data.shape}")
    weights = []
    for seed in c2.SEEDS:
        Xtr = c2.slice_windows(data, c2.N_TRAIN, seed=seed * 11 + 1)
        mu, sd = c2.fit_scaler(Xtr)
        Xtr_s = c2.scale(Xtr, mu, sd)
        ae = c2.TinyAE(din=c2.WIN * c2.D, hid=c2.HID, seed=seed)
        ae.fit(Xtr_s, epochs=200, lr=0.005, verbose=False)
        weights.append({"W1": ae.W1, "b1": ae.b1, "W2": ae.W2, "b2": ae.b2,
                        "mu": mu, "sd": sd})
    out = OUT_DIR / "c2_daedalus_ae.npz"
    np.savez(out,
             W1=np.stack([w["W1"] for w in weights]),
             b1=np.stack([w["b1"] for w in weights]),
             W2=np.stack([w["W2"] for w in weights]),
             b2=np.stack([w["b2"] for w in weights]),
             mu=np.stack([w["mu"] for w in weights]),
             sd=np.stack([w["sd"] for w in weights]))
    print(f"[C2-train-daed] saved {out}")
    return out


def transplant_c2():
    pkg_path = OUT_DIR / "c2_daedalus_ae.npz"
    if not pkg_path.exists():
        train_daedalus_c2_ae()
    pkg = np.load(pkg_path)
    data = np.load(IKAROS_DATA)
    rows = []
    for i, seed in enumerate(c2.SEEDS):
        if i >= len(pkg["W1"]): break
        Xte_n = c2.slice_windows(data, c2.N_TEST_NORMAL, seed=seed * 11 + 7)
        rng = np.random.default_rng(seed * 13 + 3)
        base = c2.slice_windows(data, c2.N_TEST_ANOM, seed=seed * 11 + 17)
        mu_o, sd_o = pkg["mu"][i], pkg["sd"][i]
        Xte_n_s = c2.scale(Xte_n, mu_o, sd_o)
        base_s = c2.scale(base, mu_o, sd_o)
        anom = np.stack([c2.ANOM_FNS[k % len(c2.ANOM_FNS)](base_s[k], rng)
                          for k in range(c2.N_TEST_ANOM)])
        ae = c2.TinyAE(din=c2.WIN * c2.D, hid=c2.HID, seed=seed)
        ae.W1 = pkg["W1"][i]; ae.b1 = pkg["b1"][i]
        ae.W2 = pkg["W2"][i]; ae.b2 = pkg["b2"][i]
        s_n = ae.recon_err(Xte_n_s); s_a = ae.recon_err(anom)
        au = c2.auroc(s_n, s_a)
        rows.append({"seed": int(seed), "transplant_auroc": au})
    return rows


# ---------- C3: train daedalus policy on daedalus thermal model, eval on ikaros
def train_daedalus_c3_policy():
    data = np.load(DAEDALUS_DATA)
    model = c3.fit_thermal_model(data)
    Qs = [c3.train_q_policy(model, seed=s) for s in range(5)]
    out = OUT_DIR / "c3_daedalus_policy.npz"
    np.savez(out, Q=np.stack(Qs), model_a=model["a"], model_b=model["b"],
             model_c=model["c"], model_resid=model["resid"], model_Tmed=model["T_med"])
    print(f"[C3-train-daed] saved {out}")
    return out, model


def transplant_c3():
    pkg_path = OUT_DIR / "c3_daedalus_policy.npz"
    if not pkg_path.exists():
        train_daedalus_c3_policy()
    other_pkg = np.load(pkg_path)
    Qs = other_pkg["Q"]
    ikaros_data = np.load(IKAROS_DATA)
    ikaros_model = c3.fit_thermal_model(ikaros_data)
    rows = []
    for seed, Q in enumerate(Qs):
        ev = c3.eval_policy(Q, ikaros_model, seed=42 + seed)
        rows.append({"seed": int(seed), **ev})
        print(f"  C3 transplant seed={seed} reward={ev['reward_mean']:.1f} "
              f"trips={ev['trips_mean']:.2f} completed={ev['completed_mean']:.1f}")
    return rows


def main():
    print("=" * 60)
    print("C1 transplant: daedalus model on ikaros data")
    print("=" * 60)
    c1_rows = transplant_c1()
    c1_self_summary = json.loads((OUT_DIR / "c1_ikaros" / "c1_ikaros_self_summary.json").read_text())
    c1_self_med = c1_self_summary["self_model_nrmse_med"]
    c1_trans_med = float(np.median([r["transplant_nrmse"] for r in c1_rows]))
    c1_gap_pct = (c1_trans_med - c1_self_med) / c1_self_med * 100
    print(f"\nC1: self={c1_self_med:.4f}  transplant={c1_trans_med:.4f}  "
          f"gap={c1_gap_pct:+.1f}% (positive = self wins)")

    print("\n" + "=" * 60)
    print("C2 transplant: daedalus AE on ikaros anomalies")
    print("=" * 60)
    c2_rows = transplant_c2()
    c2_self_summary = json.loads((OUT_DIR / "c2_ikaros" / "c2_ikaros_self_summary.json").read_text())
    c2_self_med = c2_self_summary["auroc_self_med"]
    c2_trans_med = float(np.median([r["transplant_auroc"] for r in c2_rows]))
    c2_gap = c2_self_med - c2_trans_med
    print(f"\nC2: self AUROC={c2_self_med:.3f}  transplant={c2_trans_med:.3f}  "
          f"gap={c2_gap:+.3f} pp (positive = self wins)")

    print("\n" + "=" * 60)
    print("C3 transplant: daedalus policy on ikaros thermal sim")
    print("=" * 60)
    c3_rows = transplant_c3()
    c3_self_summary = json.loads((OUT_DIR / "c3_ikaros" / "c3_ikaros_self_summary.json").read_text())
    c3_self_med = c3_self_summary["self_completed_med"]
    c3_self_trips = c3_self_summary["self_trips_med"]
    c3_trans_med = float(np.median([r["completed_mean"] for r in c3_rows]))
    c3_trans_trips = float(np.median([r["trips_mean"] for r in c3_rows]))
    c3_gap_pct = (c3_self_med - c3_trans_med) / max(c3_trans_med, 1e-3) * 100
    print(f"\nC3: self completed={c3_self_med:.1f} trips={c3_self_trips:.2f}  "
          f"transplant completed={c3_trans_med:.1f} trips={c3_trans_trips:.2f}  "
          f"gap={c3_gap_pct:+.1f}%")

    # Pre-reg WIN gates
    c1_pass = c1_gap_pct >= 30.0
    c2_pass = c2_gap >= 0.10
    c3_pass = (c3_gap_pct >= 20.0) and (c3_self_trips == 0)

    summary = {
        "c1": {"self_nrmse_med": c1_self_med,
                "transplant_nrmse_med": c1_trans_med,
                "gap_pct": c1_gap_pct,
                "win_gate_pct": 30.0,
                "PASS": bool(c1_pass),
                "transplant_per_seed": c1_rows,
                "self_per_seed": c1_self_summary["per_seed_self"]},
        "c2": {"self_auroc_med": c2_self_med,
                "transplant_auroc_med": c2_trans_med,
                "generic_untrained_auroc_med": c2_self_summary["auroc_generic_med"],
                "gap_pp": c2_gap,
                "win_gate_pp": 0.10,
                "PASS": bool(c2_pass),
                "transplant_per_seed": c2_rows,
                "self_per_seed": c2_self_summary["rows"]},
        "c3": {"self_completed_med": c3_self_med,
                "self_trips_med": c3_self_trips,
                "transplant_completed_med": c3_trans_med,
                "transplant_trips_med": c3_trans_trips,
                "gap_pct": c3_gap_pct,
                "win_gate_pct": 20.0,
                "win_requires_zero_self_trips": True,
                "PASS": bool(c3_pass),
                "transplant_per_seed": c3_rows,
                "self_per_seed": c3_self_summary["per_seed_self"],
                "generic_random": c3_self_summary["generic_random"]},
    }
    out_summary = OUT_DIR / "embodiment5_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60)
    print(f"SUMMARY saved to {out_summary}")
    print("=" * 60)
    for k in ("c1", "c2", "c3"):
        print(f"  {k.upper()}: {'PASS' if summary[k]['PASS'] else 'NULL'}")


if __name__ == "__main__":
    main()
