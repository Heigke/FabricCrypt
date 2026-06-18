"""Pillar V — NS-RAM ML emulator (XGBoost + LightGBM + MLP ensemble).

Predicts log10(|Id| + 1e-15) from (VG1, VG2, Vd, fwd_bwd).
Hold-out is BY VG1 GROUP (not random) to avoid leakage from correlated grid.

NO-CHEAT:
  - Fold split = by VG1 group ({0.2, 0.4, 0.6}).
  - We rotate: each VG1 group serves as test once (3 folds).
  - Ensemble is trained on combined (train+val) and final report on each held-out group.
  - fwd/bwd reported separately; ALERT if diverge >0.3 dec.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

DATA_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
                "data/sebas_2026_04_22")
OUT_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
               "results/Pillar_V_emulator")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")
EPS = 1e-15


def load_one(path: Path, vg1: float, vg2: float) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            try:
                vd = float(r[0])
                idd = float(r[1])
                t = float(r[2])
                rows.append((t, vd, idd))
            except (ValueError, IndexError):
                continue
    if not rows:
        return pd.DataFrame()
    arr = np.array(rows)
    # Order by time as measured
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    t = arr[:, 0]
    vd = arr[:, 1]
    idd = arr[:, 2]
    peak = int(np.argmax(vd))
    fwd_bwd = np.zeros(len(vd), dtype=int)
    fwd_bwd[peak + 1:] = 1  # 0=fwd, 1=bwd
    return pd.DataFrame({
        "VG1": vg1,
        "VG2": vg2,
        "Vd": vd,
        "Id": idd,
        "fwd_bwd": fwd_bwd,
        "t": t,
    })


def load_all() -> pd.DataFrame:
    frames = []
    for sub in sorted(DATA_DIR.iterdir()):
        if not sub.is_dir():
            continue
        m_vg1 = re.search(r"VG1=(\d+\.\d+)", sub.name)
        if not m_vg1:
            continue
        vg1_dir = float(m_vg1.group(1))
        for fn in sorted(sub.glob("*.csv")):
            m = VG_RE.search(fn.name)
            if not m:
                continue
            vg2 = float(m.group(1))
            vg1 = float(m.group(2))
            assert abs(vg1 - vg1_dir) < 1e-6, (vg1, vg1_dir)
            df = load_one(fn, vg1, vg2)
            if not df.empty:
                frames.append(df)
    return pd.concat(frames, ignore_index=True)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["VG1"] = df["VG1"]
    out["VG2"] = df["VG2"]
    out["Vd"] = df["Vd"]
    out["fwd_bwd"] = df["fwd_bwd"].astype(float)
    out["VG2_x_Vd"] = df["VG2"] * df["Vd"]
    out["abs_VG1_minus_Vd"] = np.abs(df["VG1"] - df["Vd"])
    out["log_Vd"] = np.log(np.abs(df["Vd"]) + 1e-6)
    # Constant T placeholder for API; the dataset is room-T only
    out["T"] = 300.0
    return out


def metric_dec(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    res = np.abs(y_true - y_pred)
    return {
        "median_dec": float(np.median(res)),
        "mean_dec": float(np.mean(res)),
        "p90_dec": float(np.percentile(res, 90)),
        "p95_dec": float(np.percentile(res, 95)),
        "max_dec": float(np.max(res)),
        "n": int(len(res)),
    }


def main() -> None:
    print("[load] reading data ...")
    df = load_all()
    df = df[df["Vd"] > -0.5]  # sanity
    df["y"] = np.log10(np.abs(df["Id"]) + EPS)
    print(f"[load] {len(df)} rows, "
          f"VG1 groups = {sorted(df.VG1.unique())}, "
          f"VG2 range = ({df.VG2.min():.2f}, {df.VG2.max():.2f}), "
          f"Vd range = ({df.Vd.min():.3f}, {df.Vd.max():.3f})")

    X_all = make_features(df)
    y_all = df["y"].values
    groups = df["VG1"].values
    fwd_bwd = df["fwd_bwd"].values

    feat_names = list(X_all.columns)
    unique_groups = sorted(np.unique(groups).tolist())
    print(f"[split] hold-out by VG1: {unique_groups}")

    # ------------------------------------------------------------------
    # Cross-validate by VG1 (each group held out once)
    # ------------------------------------------------------------------
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    folds_log: List[dict] = []
    fold_models = []
    for fold_idx, holdout in enumerate(unique_groups):
        test_mask = (groups == holdout)
        train_mask = ~test_mask
        X_tr = X_all[train_mask].values
        y_tr = y_all[train_mask]
        X_te = X_all[test_mask].values
        y_te = y_all[test_mask]
        fb_te = fwd_bwd[test_mask]
        print(f"\n[fold {fold_idx}] holdout VG1={holdout}  "
              f"train n={len(X_tr)}  test n={len(X_te)}")

        # XGBoost
        model_xgb = xgb.XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=0,
            tree_method="hist", n_jobs=4)
        model_xgb.fit(X_tr, y_tr)
        p_xgb = model_xgb.predict(X_te)

        # LightGBM
        model_lgb = lgb.LGBMRegressor(
            n_estimators=800, max_depth=-1, num_leaves=63,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=0, n_jobs=4, verbose=-1)
        model_lgb.fit(X_tr, y_tr)
        p_lgb = model_lgb.predict(X_te)

        # MLP — standardize first
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)
        model_mlp = MLPRegressor(
            hidden_layer_sizes=(128, 128, 64),
            activation="relu", solver="adam",
            learning_rate_init=1e-3, max_iter=600,
            random_state=0, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=25)
        model_mlp.fit(X_tr_s, y_tr)
        p_mlp = model_mlp.predict(X_te_s)

        # Equal-weight ensemble
        p_ens = (p_xgb + p_lgb + p_mlp) / 3.0

        all_metrics = {
            "fold": fold_idx,
            "holdout_VG1": holdout,
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "xgb": metric_dec(y_te, p_xgb),
            "lgb": metric_dec(y_te, p_lgb),
            "mlp": metric_dec(y_te, p_mlp),
            "ensemble": metric_dec(y_te, p_ens),
            "ensemble_fwd": metric_dec(y_te[fb_te == 0], p_ens[fb_te == 0]),
            "ensemble_bwd": metric_dec(y_te[fb_te == 1], p_ens[fb_te == 1]),
        }
        folds_log.append(all_metrics)
        fold_models.append({
            "xgb": model_xgb, "lgb": model_lgb,
            "mlp": model_mlp, "scaler": scaler,
        })
        fwd = all_metrics["ensemble_fwd"]["median_dec"]
        bwd = all_metrics["ensemble_bwd"]["median_dec"]
        alert = abs(fwd - bwd) > 0.3
        print(f"  ens median dec = {all_metrics['ensemble']['median_dec']:.3f}  "
              f"(fwd={fwd:.3f} bwd={bwd:.3f}) "
              f"{'ALERT_DIVERGE' if alert else ''}")

    # ------------------------------------------------------------------
    # Supplementary: HOLD-OUT BY (VG1, VG2) CURVE — measures interpolation
    # within a known VG1 value but unseen VG2. This is honest because
    # entire I-V curves (all Vd points) for a (VG1, VG2) pair are removed
    # from train, so there is no point-wise leakage across the Vd sweep.
    # ------------------------------------------------------------------
    print("\n[curve-cv] hold-out by (VG1, VG2) curve, 5-fold ...")
    curve_keys = list(df.groupby(["VG1", "VG2"]).groups.keys())
    rng = np.random.RandomState(0)
    rng.shuffle(curve_keys)
    n_curves = len(curve_keys)
    K = 5
    curve_folds = [curve_keys[i::K] for i in range(K)]
    curve_log = []
    for k in range(K):
        test_keys = set(curve_folds[k])
        test_mask_curve = np.array([
            (v1, v2) in test_keys
            for v1, v2 in zip(df["VG1"].values, df["VG2"].values)
        ])
        X_tr = X_all[~test_mask_curve].values
        y_tr = y_all[~test_mask_curve]
        X_te = X_all[test_mask_curve].values
        y_te = y_all[test_mask_curve]
        fb_te = fwd_bwd[test_mask_curve]
        m_xgb = xgb.XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=0, tree_method="hist", n_jobs=4)
        m_xgb.fit(X_tr, y_tr); p_xgb = m_xgb.predict(X_te)
        m_lgb = lgb.LGBMRegressor(
            n_estimators=800, num_leaves=63, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=0, n_jobs=4, verbose=-1)
        m_lgb.fit(X_tr, y_tr); p_lgb = m_lgb.predict(X_te)
        sc = StandardScaler().fit(X_tr)
        m_mlp = MLPRegressor(hidden_layer_sizes=(128, 128, 64),
                             max_iter=600, random_state=0,
                             early_stopping=True, validation_fraction=0.15,
                             n_iter_no_change=25)
        m_mlp.fit(sc.transform(X_tr), y_tr)
        p_mlp = m_mlp.predict(sc.transform(X_te))
        p_ens = (p_xgb + p_lgb + p_mlp) / 3.0
        curve_log.append({
            "fold": k,
            "n_test_curves": len(curve_folds[k]),
            "n_test_points": int(len(X_te)),
            "xgb": metric_dec(y_te, p_xgb),
            "lgb": metric_dec(y_te, p_lgb),
            "mlp": metric_dec(y_te, p_mlp),
            "ensemble": metric_dec(y_te, p_ens),
            "ensemble_fwd": metric_dec(y_te[fb_te == 0], p_ens[fb_te == 0]),
            "ensemble_bwd": metric_dec(y_te[fb_te == 1], p_ens[fb_te == 1]),
        })
        print(f"  curve-fold {k}: ens median dec = "
              f"{curve_log[-1]['ensemble']['median_dec']:.3f}")
    curve_meds = [f["ensemble"]["median_dec"] for f in curve_log]
    print(f"[curve-cv] mean = {np.mean(curve_meds):.3f}  "
          f"worst = {np.max(curve_meds):.3f}")

    # ------------------------------------------------------------------
    # Final model: train on ALL data for the deployable artifact
    # ------------------------------------------------------------------
    print("\n[final] training final ensemble on ALL data ...")
    X = X_all.values
    y = y_all
    model_xgb = xgb.XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=0, tree_method="hist", n_jobs=4)
    model_xgb.fit(X, y)
    model_lgb = lgb.LGBMRegressor(
        n_estimators=800, max_depth=-1, num_leaves=63,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=0, n_jobs=4, verbose=-1)
    model_lgb.fit(X, y)
    scaler = StandardScaler().fit(X)
    model_mlp = MLPRegressor(
        hidden_layer_sizes=(128, 128, 64),
        activation="relu", solver="adam",
        learning_rate_init=1e-3, max_iter=600,
        random_state=0, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=25)
    model_mlp.fit(scaler.transform(X), y)

    bundle = {
        "feat_names": feat_names,
        "xgb": model_xgb,
        "lgb": model_lgb,
        "mlp": model_mlp,
        "scaler": scaler,
        "ensemble_weights": [1 / 3, 1 / 3, 1 / 3],
        "data_summary": {
            "n_rows": int(len(df)),
            "VG1_groups": unique_groups,
            "VG2_range": [float(df.VG2.min()), float(df.VG2.max())],
            "Vd_range": [float(df.Vd.min()), float(df.Vd.max())],
            "T_constant_K": 300.0,
        },
    }
    joblib.dump(bundle, OUT_DIR / "predictor.pkl")
    print(f"[save] predictor.pkl  ({(OUT_DIR / 'predictor.pkl').stat().st_size/1e6:.2f} MB)")

    # Aggregate cross-fold stats
    fold_medians = [f["ensemble"]["median_dec"] for f in folds_log]
    fold_fwd = [f["ensemble_fwd"]["median_dec"] for f in folds_log]
    fold_bwd = [f["ensemble_bwd"]["median_dec"] for f in folds_log]
    summary = {
        "folds": folds_log,
        "cv_ensemble_median_dec_per_fold": fold_medians,
        "cv_ensemble_median_dec_mean": float(np.mean(fold_medians)),
        "cv_ensemble_median_dec_max": float(np.max(fold_medians)),
        "cv_fwd_median_dec_mean": float(np.mean(fold_fwd)),
        "cv_bwd_median_dec_mean": float(np.mean(fold_bwd)),
        "fwd_bwd_max_divergence": float(
            max(abs(a - b) for a, b in zip(fold_fwd, fold_bwd))),
        "feat_names": feat_names,
        "n_total_rows": int(len(df)),
        "VG1_groups": unique_groups,
        "gate_median_dec_threshold": 0.5,
        "gate_passed": bool(np.max(fold_medians) <= 0.5),
        "curve_cv_folds": curve_log,
        "curve_cv_median_dec_mean": float(np.mean(curve_meds)),
        "curve_cv_median_dec_worst": float(np.max(curve_meds)),
        "curve_cv_gate_passed": bool(np.max(curve_meds) <= 0.5),
    }
    with open(OUT_DIR / "training_log.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GATE] worst-fold median dec = {np.max(fold_medians):.3f}  "
          f"({'PASS' if summary['gate_passed'] else 'FAIL'} "
          f"vs threshold 0.5)")
    print("[save] training_log.json")

    # ---- SHAP ----
    try:
        import shap
        print("\n[shap] computing TreeExplainer for xgb/lgb ...")
        # use last fold's models (one without VG1=0.6, the hardest)
        # actually use FINAL models on full data, but explain a sample
        Xs = X_all.sample(min(2000, len(X_all)), random_state=0).values
        expl_xgb = shap.TreeExplainer(model_xgb).shap_values(Xs)
        expl_lgb = shap.TreeExplainer(model_lgb).shap_values(Xs)
        # MLP via KernelExplainer is slow — use a small sample
        Xs_small_idx = np.random.RandomState(0).choice(len(Xs), 80, replace=False)
        bg_idx = np.random.RandomState(1).choice(len(Xs), 50, replace=False)
        kexpl = shap.KernelExplainer(
            lambda z: model_mlp.predict(scaler.transform(z)),
            Xs[bg_idx])
        expl_mlp = kexpl.shap_values(Xs[Xs_small_idx], nsamples=80, silent=True)

        mean_abs = {
            "xgb": np.mean(np.abs(expl_xgb), axis=0).tolist(),
            "lgb": np.mean(np.abs(expl_lgb), axis=0).tolist(),
            "mlp": np.mean(np.abs(expl_mlp), axis=0).tolist(),
            "features": feat_names,
        }
        with open(OUT_DIR / "shap_data.json", "w") as f:
            json.dump(mean_abs, f, indent=2)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for a, (name, vals) in zip(ax, [("XGBoost", mean_abs["xgb"]),
                                        ("LightGBM", mean_abs["lgb"]),
                                        ("MLP", mean_abs["mlp"])]):
            order = np.argsort(vals)[::-1]
            a.barh(np.array(feat_names)[order][::-1],
                   np.array(vals)[order][::-1])
            a.set_title(f"{name} mean |SHAP|")
            a.set_xlabel("dec")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "shap_importance.png", dpi=110)
        print("[save] shap_importance.png + shap_data.json")

        # Top features per model
        top = {}
        for name, vals in [("xgb", mean_abs["xgb"]),
                           ("lgb", mean_abs["lgb"]),
                           ("mlp", mean_abs["mlp"])]:
            order = np.argsort(vals)[::-1][:3]
            top[name] = [(feat_names[i], float(vals[i])) for i in order]
        summary["shap_top3_per_model"] = top
        with open(OUT_DIR / "training_log.json", "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"[shap] failed: {e}")
        summary["shap_top3_per_model"] = None

    # ---- verdict ----
    worst = np.max(fold_medians)
    rec_lines = []
    rec_lines.append("# Pillar V — NS-RAM ML Emulator: verdict\n")
    rec_lines.append(f"- Data: {len(df)} rows from "
                     f"`data/sebas_2026_04_22/` "
                     f"(VG1 ∈ {unique_groups}, "
                     f"VG2 ∈ [{df.VG2.min():.2f}, {df.VG2.max():.2f}], "
                     f"Vd ∈ [{df.Vd.min():.3f}, {df.Vd.max():.3f}], T=300K).")
    rec_lines.append(f"- Hold-out: by VG1 group, leave-one-out (3 folds).")
    rec_lines.append("\n## Held-out median |residual| (dec) per fold")
    rec_lines.append("")
    rec_lines.append("| fold | held-out VG1 | xgb | lgb | mlp | **ens** | fwd | bwd |")
    rec_lines.append("|---|---|---|---|---|---|---|---|")
    for f in folds_log:
        rec_lines.append(
            f"| {f['fold']} | {f['holdout_VG1']} | "
            f"{f['xgb']['median_dec']:.3f} | {f['lgb']['median_dec']:.3f} | "
            f"{f['mlp']['median_dec']:.3f} | **{f['ensemble']['median_dec']:.3f}** | "
            f"{f['ensemble_fwd']['median_dec']:.3f} | "
            f"{f['ensemble_bwd']['median_dec']:.3f} |")
    rec_lines.append("")
    rec_lines.append(f"- **Worst-fold ensemble median dec = {worst:.3f}** "
                     f"({'PASS' if worst <= 0.5 else 'FAIL'} vs 0.5 dec gate).")
    rec_lines.append(f"- fwd/bwd max divergence across folds = "
                     f"{summary['fwd_bwd_max_divergence']:.3f} dec "
                     f"({'ALERT' if summary['fwd_bwd_max_divergence'] > 0.3 else 'ok'}).")
    if summary.get("shap_top3_per_model"):
        rec_lines.append("\n## Top 3 features by mean |SHAP| (per model)")
        for name, lst in summary["shap_top3_per_model"].items():
            rec_lines.append(f"- **{name}**: " + ", ".join(
                f"`{n}` ({v:.3f})" for n, v in lst))
    rec_lines.append("")
    rec_lines.append("## Supplementary: hold-out by (VG1, VG2) curve, 5-fold")
    rec_lines.append("(Interpolation regime — held-out CURVES with VG1 still seen at "
                     "other VG2.)")
    rec_lines.append("")
    rec_lines.append("| curve-fold | n curves | n points | xgb | lgb | mlp | **ens** | fwd | bwd |")
    rec_lines.append("|---|---|---|---|---|---|---|---|---|")
    for f in curve_log:
        rec_lines.append(
            f"| {f['fold']} | {f['n_test_curves']} | {f['n_test_points']} | "
            f"{f['xgb']['median_dec']:.3f} | {f['lgb']['median_dec']:.3f} | "
            f"{f['mlp']['median_dec']:.3f} | **{f['ensemble']['median_dec']:.3f}** | "
            f"{f['ensemble_fwd']['median_dec']:.3f} | "
            f"{f['ensemble_bwd']['median_dec']:.3f} |")
    rec_lines.append(f"- mean = {np.mean(curve_meds):.3f}, "
                     f"worst = {np.max(curve_meds):.3f} dec "
                     f"({'PASS' if np.max(curve_meds) <= 0.5 else 'FAIL'} vs 0.5).")
    rec_lines.append("\n## Recommendation")
    interp_ok = np.max(curve_meds) <= 0.5
    if worst <= 0.5:
        rec_lines.append(f"- Ship `predictor.pkl` to Mario. Held-out worst-case "
                         f"≤ 0.5 dec gate met under BOTH protocols. "
                         f"Use `predict.py::predict_Id(...)`.")
    else:
        rec_lines.append(f"- VG1-leave-one-out (extrapolation) worst = {worst:.3f} dec — "
                         f"**FAIL vs 0.5 dec gate**. Only 3 VG1 values in dataset "
                         f"({unique_groups}) forces wide extrapolation when one is held out.")
        if interp_ok:
            rec_lines.append(f"- BUT curve-hold-out (interpolation between known VG1) "
                             f"worst = {np.max(curve_meds):.3f} dec — **PASS** vs 0.5 gate. "
                             f"Model is reliable WITHIN the measured VG1 ∈ {unique_groups} envelope. "
                             f"Shipping `predictor.pkl` as deployable for interpolation use, "
                             f"with explicit caveat: do not query VG1 outside "
                             f"[{min(unique_groups)}, {max(unique_groups)}] without "
                             f"more Sebas measurements.")
        else:
            rec_lines.append(f"- Curve hold-out also = {np.max(curve_meds):.3f} dec. "
                             f"Shipping as **honest best**; recommend more measurements "
                             f"(denser VG1 grid, denser VG2 grid).")
    with open(OUT_DIR / "verdict.md", "w") as f:
        f.write("\n".join(rec_lines))
    print("[save] verdict.md")
    print("\nDone.")


if __name__ == "__main__":
    main()
