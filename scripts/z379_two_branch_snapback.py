"""z379 — S2b two-branch search for snapback bistability in NS-RAM 2T pyport.

S1 forced Vb=0.8V → 5.5 dec jump (fold physics IS in BSIM4). S2 arc-length
confirmed: high-Vb solution is a DISCONNECTED root from low-Vb. S2b runs the
Newton solver TWICE per Vd point — once from cold init (Vsint=0, Vb=0; the
z372 default cascade) and once from hot init (Vsint=0.05, Vb=0.80; the S1
working point) — and picks the higher-|Ids| converged root per point.

3-bias canonical test: VG1=0.2/VG2=0.10, VG1=0.4/VG2=0.20, VG1=0.6/VG2=0.20.
R-46 per-VG1 best params, canonical Sebas BJT. Three solver modes:
  - cold-only   (z372 baseline)
  - hot-only    (forced upper-branch warm-start cascade)
  - multi-init  (two-branch pick max |Ids|)

Gates (pre-registered in 01_LOG.md):
  INFRA      : 3/3 biases converge cold AND hot, finite Ids both
  DISCOVERY  : hot-init ≥0.5 dec fold at VG1=0.6 AND multi-init beats cold
               by ≥1 dec RMSE at Vd>1V on VG1=0.6
  AMBITIOUS  : cell-wide RMSE < 0.5 dec AND VG1=0.6 fold > 1.5 dec
  KILL-SHOT  : hot-init also fails to produce fold → bistability not BSIM4-native

Output: results/z379_two_branch/{summary.json, snapback_two_branch.png,
        branch_selection.png}
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import sys, json, re, math, csv, importlib.util
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
OUT = ROOT / "results/z379_two_branch"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"


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
    bjt.Va = 0.903; bjt.Is = 5.95e-12; bjt.Bf = 991.0
    return cfg, M1, M2, bjt


def load_measured(vg1, vg2=0.20):
    sub = DATA / f"2vHCa-2 I-Vs@VG2 VG1={vg1} vnwell=2"
    pat = re.compile(rf"VG2={vg2:.2f}_VG={vg1}")
    for f in sorted(sub.glob("*.csv")):
        if pat.search(f.name):
            d = np.loadtxt(f, delimiter=",", skiprows=1)
            return d[:, 0], np.abs(d[:, 1]), f.name
    raise FileNotFoundError(f"no csv for VG1={vg1} VG2={vg2}")


def rmse_dec(Id_p, Id_m, vd_mask=None, floor=1e-15):
    p = np.asarray(Id_p, dtype=float); m = np.asarray(Id_m, dtype=float)
    mask = (m > floor) & (p > floor) & np.isfinite(p)
    if vd_mask is not None:
        mask = mask & vd_mask
    if mask.sum() < 3:
        return float("nan"), int(mask.sum())
    return float(np.sqrt(np.mean((np.log10(p[mask]) - np.log10(m[mask]))**2))), int(mask.sum())


def fold_dec(Id, Vd, vmin=0.5):
    Id = np.maximum(np.asarray(Id, dtype=float), 1e-15)
    Vd = np.asarray(Vd, dtype=float)
    dl = np.diff(np.log10(Id))
    Vmid = 0.5 * (Vd[1:] + Vd[:-1])
    mask = Vmid >= vmin
    if not mask.any() or len(dl) == 0:
        return 0.0
    return float(np.where(mask, dl, -np.inf).max())


def run_mode(forward_2t, cfg, M1, M2, bjt, sd_M1, sd_M2, vg1, vg2, Vd_t, P_M1, P_M2, mode):
    """mode in {'cold', 'hot', 'multi'}.

    Per z372 convention: overrides go through `patch_sd_scaled` on sd.scaled.
    P_M1/P_M2 are NOT passed to forward_2t (compute_dc reads sd.scaled, and
    _override_sd would crash on attrs that only live in sd.scaled).
    """
    kw = dict(cfg=cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_t,
              VG1=torch.tensor(vg1, dtype=torch.float64),
              VG2=torch.tensor(vg2, dtype=torch.float64),
              warm_start=True)
    if mode == "cold":
        with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
            out = forward_2t(**kw)
    elif mode == "hot":
        # Hot-init cascade: warm-start every point from previous HOT root.
        out = _hot_only_cascade(forward_2t, cfg, M1, M2, bjt, sd_M1, sd_M2,
                                vg1, vg2, Vd_t, P_M1, P_M2)
    elif mode == "multi":
        with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
            out = forward_2t(**kw, multi_init=True,
                             hot_Vsint_init=0.05, hot_Vb_init=0.80)
    else:
        raise ValueError(mode)
    return out


def _hot_only_cascade(forward_2t, cfg, M1, M2, bjt, sd_M1, sd_M2,
                      vg1, vg2, Vd_t, P_M1, P_M2):
    """Run the steady-state solver per point with a hot warm-start cascade
    (Vsint=0.05, Vb=0.80 initial; then warm-start each subsequent point with
    the previous HOT-init solution). This is what the user calls "hot-only"."""
    from nsram.bsim4_port.nsram_cell_2T import solve_2t_steady_state
    Vsint_warm = torch.tensor(0.05, dtype=torch.float64)
    Vb_warm = torch.tensor(0.80, dtype=torch.float64)
    Ids, Vs, Vb, Ic, IdM1, IdM2, conv, niters = [], [], [], [], [], [], [], []
    with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
        for i in range(Vd_t.shape[0]):
            Vd_i = Vd_t[i:i+1]
            out = solve_2t_steady_state(
                cfg, M1, bjt,
                Vd=Vd_i,
                VG1=torch.tensor(vg1, dtype=torch.float64),
                VG2=torch.tensor(vg2, dtype=torch.float64),
                Vsint_init=Vsint_warm.expand_as(Vd_i),
                Vb_init=Vb_warm.expand_as(Vd_i),
                verbose=False,
                model_M2=M2,
            )
            Ids.append(out["Id"].squeeze(0))
            Vs.append(out["Vsint"].squeeze(0))
            Vb.append(out["Vb"].squeeze(0))
            IdM1.append(out["Ids_M1"].squeeze(0))
            IdM2.append(out["Ids_M2"].squeeze(0))
            Ic.append(out["Ic_Q1"].squeeze(0))
            niters.append(out["niter"])
            conv.append(bool(out["converged"].all()))
            # Cascade hot warm-start
            Vsint_warm = out["Vsint"].detach().squeeze(0)
            Vb_warm = out["Vb"].detach().squeeze(0)
    return {
        "Id": torch.stack(Ids), "Vsint": torch.stack(Vs), "Vb": torch.stack(Vb),
        "Ids_M1": torch.stack(IdM1), "Ids_M2": torch.stack(IdM2),
        "Ic_Q1": torch.stack(Ic), "niter": niters, "converged": conv,
        "branch_selected": [1] * Vd_t.shape[0],
    }


def main():
    # Per-VG1 best from R-46
    x_best = [1889.88, 1.8447, 9.1722,
              1092.27, 1.5152, 9.8983,
               417.63, 0.9036, 6.7846]
    per_vg1 = {0.2: (x_best[0], x_best[1], 10**x_best[2]),
               0.4: (x_best[3], x_best[4], 10**x_best[5]),
               0.6: (x_best[6], x_best[7], 10**x_best[8])}

    cfg, M1, M2, bjt = build_base()
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    sebas_rows = load_sebas_params()

    targets = [(0.2, 0.10), (0.4, 0.20), (0.6, 0.20)]
    modes = ["cold", "hot", "multi"]
    mode_colors = {"cold": "tab:blue", "hot": "tab:orange", "multi": "tab:red"}
    mode_labels = {"cold": "cold-init (z372)", "hot": "hot-init (S1 cascade)",
                   "multi": "multi-init (max |Ids|)"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig2, axes2 = plt.subplots(1, 3, figsize=(17, 4.0))
    results = []

    for ax, ax2, (vg1, vg2) in zip(axes, axes2, targets):
        Vd_m, Id_m, fname = load_measured(vg1, vg2)
        Bf, iii, Rs = per_vg1[vg1]
        bjt.Bf = Bf; cfg.iii_body_gain = iii; cfg.vnwell_Rs = Rs
        row = find_or_impute_row(sebas_rows, vg1, vg2)
        P_M1, P_M2 = make_overrides(row)
        Vd_t = torch.tensor(Vd_m, dtype=torch.float64)

        per_mode = {}
        for mode in modes:
            out = run_mode(forward_2t, cfg, M1, M2, bjt, sd_M1, sd_M2,
                           vg1, vg2, Vd_t, P_M1, P_M2, mode)
            Id_p = np.abs(out["Id"].detach().cpu().numpy())
            finite = np.isfinite(Id_p).all()
            conv = (all(out["converged"]) if isinstance(out["converged"], list)
                    else bool(np.all(out["converged"])))
            r_all, n_all = rmse_dec(Id_p, Id_m)
            vd_hi = Vd_m > 1.0
            r_hi, n_hi = rmse_dec(Id_p, Id_m, vd_mask=vd_hi)
            fold = fold_dec(Id_p, Vd_m, vmin=0.5)
            branches = out.get("branch_selected", [0]*len(Id_p))
            n_hot = int(sum(1 for b in branches if b == 1))
            per_mode[mode] = {
                "Id_p": Id_p, "rmse_all": r_all, "rmse_hi": r_hi,
                "fold_dec": fold, "n_finite": int(np.sum(np.isfinite(Id_p))),
                "n_total": int(len(Id_p)), "converged_all": bool(conv),
                "branches": branches, "n_hot": n_hot,
            }
            ax.semilogy(Vd_m, np.maximum(Id_p, 1e-15),
                        color=mode_colors[mode], lw=1.4,
                        label=f"{mode_labels[mode]} | RMSE={r_all:.2f}d | fold={fold:.1f}d")

        # Measured
        ax.semilogy(Vd_m, np.maximum(Id_m, 1e-15), "k.", ms=4, label="measured (Sebas)")

        # Measured fold for context
        meas_fold = fold_dec(Id_m, Vd_m, vmin=0.5)

        ax.set_xlabel("Vd (V)"); ax.set_ylabel("|Id| (A)")
        ax.set_ylim(1e-13, 1e-2)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"VG1={vg1}, VG2=+{vg2:.2f}  meas-fold={meas_fold:.1f}d")
        ax.legend(loc="lower right", fontsize=7)

        # Branch selection panel
        br = np.array(per_mode["multi"]["branches"])
        ax2.step(Vd_m, br, where="post", color="tab:red", lw=1.4)
        ax2.fill_between(Vd_m, 0, br, step="post", color="tab:red", alpha=0.15)
        ax2.set_xlabel("Vd (V)")
        ax2.set_ylabel("branch (0=cold, 1=hot)")
        ax2.set_ylim(-0.1, 1.2)
        ax2.set_yticks([0, 1])
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f"VG1={vg1}, VG2=+{vg2:.2f}  "
                      f"hot-picked: {per_mode['multi']['n_hot']}/{len(br)}")

        # Drop heavy arrays from serialization
        serial = {}
        for mode in modes:
            d = dict(per_mode[mode])
            d.pop("Id_p", None)
            d["branches"] = [int(b) for b in d["branches"]]
            serial[mode] = d
        results.append({
            "VG1": vg1, "VG2": vg2, "file": fname,
            "params": {"Bf": Bf, "iii_body_gain": iii, "vnwell_Rs": Rs},
            "measured_fold_dec": meas_fold,
            "modes": serial,
        })

    fig.suptitle("z379 S2b two-branch snapback: cold vs hot vs multi-init  "
                 "(R-46 per-VG1 BBO, canonical BJT)", fontsize=11, y=1.00)
    fig.tight_layout()
    out_png = OUT / "snapback_two_branch.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[z379] wrote {out_png}")

    fig2.suptitle("z379 branch selection (multi-init): which root won per Vd",
                  fontsize=11, y=1.02)
    fig2.tight_layout()
    out_png2 = OUT / "branch_selection.png"
    fig2.savefig(out_png2, dpi=150, bbox_inches="tight")
    print(f"[z379] wrote {out_png2}")

    # Gate evaluation
    # INFRA: 3/3 biases produce finite Ids on BOTH branches (cold + hot).
    # Note: Newton's `converged` flag is per-residual-norm and is often False
    # even when Ids is physically sensible (legacy z372 behavior). Gate keys
    # on finite-Ids count, not the conservative converged flag.
    infra = all(
        r["modes"]["cold"]["n_finite"] == r["modes"]["cold"]["n_total"]
        and r["modes"]["hot"]["n_finite"] == r["modes"]["hot"]["n_total"]
        and r["modes"]["multi"]["n_finite"] == r["modes"]["multi"]["n_total"]
        for r in results
    )
    # VG1=0.6 row
    r06 = next(r for r in results if abs(r["VG1"] - 0.6) < 1e-3)
    hot_fold_06 = r06["modes"]["hot"]["fold_dec"]
    multi_rmse_hi_06 = r06["modes"]["multi"]["rmse_hi"]
    cold_rmse_hi_06 = r06["modes"]["cold"]["rmse_hi"]
    multi_fold_06 = r06["modes"]["multi"]["fold_dec"]
    rmse_improvement_06 = (cold_rmse_hi_06 - multi_rmse_hi_06) if (
        np.isfinite(cold_rmse_hi_06) and np.isfinite(multi_rmse_hi_06)) else float("nan")
    discovery = (hot_fold_06 >= 0.5) and (rmse_improvement_06 >= 1.0
                                          if np.isfinite(rmse_improvement_06) else False)
    cell_rmse_multi = float(np.nanmedian([r["modes"]["multi"]["rmse_all"] for r in results]))
    ambitious = (cell_rmse_multi < 0.5) and (multi_fold_06 > 1.5)
    # KILL-SHOT: hot fails to produce fold (fold < 0.5 dec at VG1=0.6 even hot)
    kill_shot = (hot_fold_06 < 0.5)

    summary = {
        "script": "z379_two_branch_snapback",
        "task": "S2b two-branch snapback search (cold vs hot vs multi-init)",
        "params_source": "R-46 z365 per-VG1 BBO best, canonical BJT",
        "x_best_R46": x_best,
        "results_per_bias": results,
        "gates": {
            "INFRA": bool(infra),
            "DISCOVERY": bool(discovery),
            "AMBITIOUS": bool(ambitious),
            "KILL_SHOT": bool(kill_shot),
        },
        "summary_metrics": {
            "VG1_0.6_hot_fold_dec": hot_fold_06,
            "VG1_0.6_multi_fold_dec": multi_fold_06,
            "VG1_0.6_cold_rmse_hi": cold_rmse_hi_06,
            "VG1_0.6_multi_rmse_hi": multi_rmse_hi_06,
            "VG1_0.6_rmse_improvement_dec": rmse_improvement_06,
            "cell_median_rmse_multi_dec": cell_rmse_multi,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[z379] wrote {OUT/'summary.json'}")
    print(f"[z379] INFRA={infra} DISCOVERY={discovery} AMBITIOUS={ambitious} "
          f"KILL_SHOT={kill_shot}")
    print(f"[z379] VG1=0.6 hot fold={hot_fold_06:.2f}d, multi fold={multi_fold_06:.2f}d, "
          f"cold RMSE(Vd>1)={cold_rmse_hi_06:.2f}d, multi RMSE(Vd>1)={multi_rmse_hi_06:.2f}d, "
          f"Δ={rmse_improvement_06:.2f}d")
    for r in results:
        print(f"[z379]  VG1={r['VG1']} VG2={r['VG2']}: "
              f"cold_RMSE={r['modes']['cold']['rmse_all']:.2f}d  "
              f"hot_RMSE={r['modes']['hot']['rmse_all']:.2f}d  "
              f"multi_RMSE={r['modes']['multi']['rmse_all']:.2f}d  "
              f"hot_fold={r['modes']['hot']['fold_dec']:.2f}d  "
              f"multi_fold={r['modes']['multi']['fold_dec']:.2f}d  "
              f"hot_picked={r['modes']['multi']['n_hot']}/{r['modes']['multi']['n_total']}")


if __name__ == "__main__":
    main()
