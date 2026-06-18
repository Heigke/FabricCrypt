"""z401 — S6-A.1 element test: probe v_dnw for vertical NPN-to-DNW alone.

Sweeps v_dnw across {0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0} V with ONLY
`use_vertical_npn_to_dnw=True` (no other S6 flags). Uses R-46 per-VG1 best
params for VG1=0.6 (eval 94 from z365 bbo_history.json).

Test: VG1=0.6, VG2=+0.2, Vd ∈ [0, 2.0] in 0.05V steps.
Fold magnitude = max(log10 Id) - min(log10 Id) on Vd ∈ [0.5, 2.0].

Reference: measured fold ≈ 1.07 dec (S2b), baseline (no DNW) ≈ 0.03 dec (S3-C).

Outputs: results/z401_v_dnw_sweep/{summary.json, ids_vs_vd_family.png, run.log}
"""
import os, sys, json, time, importlib.util
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/z401_v_dnw_sweep"
OUT.mkdir(parents=True, exist_ok=True)

LOG = open(OUT / "run.log", "w")
def log(m):
    print(m, flush=True)
    LOG.write(m + "\n"); LOG.flush()

t0 = time.time()
log(f"[z401] start  cwd={os.getcwd()}")

# ---- Params: VG1=0.6 from z365 eval 94 (per user instruction) -------------
BBO_HIST = ROOT / "results/z365_perVG1_bbo/bbo_history.json"
hist = json.load(open(BBO_HIST))["history"]
e94 = hist[94]
x = e94["x"]
# Per-VG1 mapping (z365_perVG1_bbo.py): [Bf020, iii020, log10Rs020,
#  Bf040, iii040, log10Rs040, Bf060, iii060, log10Rs060]
Bf060 = float(x[6])
iii060 = float(x[7])
Rs060 = 10.0 ** float(x[8])
log(f"[z401] R-46 eval94 VG1=0.6: Bf={Bf060:.3f} iii={iii060:.4f} Rs={Rs060:.3e}")

# ---- Build base model (mirror of z365 build_pyport_base) ------------------
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t
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
bjt.Va = 0.903; bjt.Is = 5.95e-12
bjt.Bf = Bf060
cfg.iii_body_gain = iii060
cfg.vnwell_Rs = Rs060

# Verify flags exist
assert hasattr(cfg, "use_vertical_npn_to_dnw"), "S6-A flag missing in NSRAMCell2TConfig"
log(f"[z401] cfg has use_vertical_npn_to_dnw flag OK; default v_dnw={getattr(cfg, 'v_dnw', None)}")

# ---- Sweep specifications --------------------------------------------------
V_DNW_LIST = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0]
Vd_arr = np.arange(0.0, 2.0 + 1e-9, 0.05)
VG1_VAL = 0.6
VG2_VAL = 0.2

def run_one(v_dnw, enable):
    cfg.use_vertical_npn_to_dnw = bool(enable)
    if enable:
        cfg.v_dnw = float(v_dnw)
    Vd_t = torch.tensor(Vd_arr, dtype=torch.float64)
    VG1_t = torch.tensor(VG1_VAL, dtype=torch.float64)
    VG2_t = torch.tensor(VG2_VAL, dtype=torch.float64)
    with torch.no_grad():
        out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt,
                         Vd_seq=Vd_t, VG1=VG1_t, VG2=VG2_t)
    Id = np.asarray(out["Id"], dtype=np.float64) if hasattr(out, "keys") else \
         np.asarray(out, dtype=np.float64)
    return Id

# ---- Baseline (no DNW) -----------------------------------------------------
log("[z401] baseline (use_vertical_npn_to_dnw=False) ...")
Id_base = run_one(0.0, enable=False)
def fold(Id):
    mask = (Vd_arr >= 0.5) & (Vd_arr <= 2.0)
    lg = np.log10(np.clip(np.abs(Id[mask]), 1e-30, None))
    return float(lg.max() - lg.min())
base_fold = fold(Id_base)
log(f"[z401] baseline fold = {base_fold:.4f} dec  (Vd∈[0.5,2.0])")

# ---- Sweep -----------------------------------------------------------------
results = []
Id_curves = {}
for v in V_DNW_LIST:
    log(f"[z401] v_dnw={v:.2f}V ...")
    Id = run_one(v, enable=True)
    if np.any(~np.isfinite(Id)):
        log(f"[z401]   WARNING: non-finite Id at v_dnw={v}")
    fd = fold(Id)
    log(f"[z401]   fold = {fd:.4f} dec")
    results.append({"v_dnw": float(v), "fold_dec": fd,
                    "Id_min": float(np.min(Id)), "Id_max": float(np.max(Id))})
    Id_curves[f"{v:.2f}"] = Id.tolist()

# Discovery checks
folds = [r["fold_dec"] for r in results]
max_fold = max(folds)
best_v = V_DNW_LIST[int(np.argmax(folds))]
INFRA_OK = all(np.isfinite(folds)) and (time.time() - t0) < 600
DISCOVERY = max_fold > 0.5
AMBITIOUS = max_fold > 1.5
KILL_SHOT = max_fold < 0.1
MEASURED = 1.07

log(f"[z401] ===== verdicts =====")
log(f"[z401] baseline fold (DNW off)   : {base_fold:.4f} dec  (S3-C ref 0.03)")
log(f"[z401] max DNW-only fold         : {max_fold:.4f} dec at v_dnw={best_v:.2f}")
log(f"[z401] measured target           : {MEASURED:.4f} dec  (S2b)")
log(f"[z401] INFRA       : {INFRA_OK}")
log(f"[z401] DISCOVERY   : {DISCOVERY}   (max > 0.5 dec)")
log(f"[z401] AMBITIOUS   : {AMBITIOUS}   (max > 1.5 dec)")
log(f"[z401] KILL_SHOT   : {KILL_SHOT}   (max < 0.1 dec)")

# ---- Plot ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(Vd_arr, np.abs(Id_base), color="k", lw=1.5, ls=":", label="baseline (no DNW)")
cmap = plt.cm.viridis
for i, v in enumerate(V_DNW_LIST):
    Id = np.array(Id_curves[f"{v:.2f}"])
    ax.plot(Vd_arr, np.abs(Id), lw=1.4, color=cmap(i / max(1, len(V_DNW_LIST) - 1)),
            label=f"v_dnw={v:.2f}V (fold={results[i]['fold_dec']:.2f})")
ax.set_yscale("log")
ax.set_xlabel("Vd [V]"); ax.set_ylabel("|Id| [A]")
ax.set_title("z401 S6-A.1: vertical NPN-to-DNW alone, VG1=0.6 / VG2=+0.2")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, ncol=2, loc="best")
fig.tight_layout()
fig.savefig(OUT / "ids_vs_vd_family.png", dpi=140)
plt.close(fig)

summary = {
    "script": "z401_v_dnw_sweep_dc",
    "elapsed_s": time.time() - t0,
    "params": {"Bf": Bf060, "iii_body_gain": iii060, "vnwell_Rs": Rs060,
               "source": "z365 bbo_history.json eval 94"},
    "VG1": VG1_VAL, "VG2": VG2_VAL,
    "Vd_grid": {"start": 0.0, "stop": 2.0, "step": 0.05},
    "fold_window_V": [0.5, 2.0],
    "baseline_fold_dec": base_fold,
    "v_dnw_results": results,
    "max_fold_dec": max_fold,
    "best_v_dnw": best_v,
    "measured_fold_dec_S2b": MEASURED,
    "baseline_ref_S3C": 0.03,
    "verdicts": {
        "INFRA": INFRA_OK, "DISCOVERY": DISCOVERY,
        "AMBITIOUS": AMBITIOUS, "KILL_SHOT": KILL_SHOT,
    },
    "Id_curves": Id_curves,
    "Id_baseline": Id_base.tolist(),
}
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
log(f"[z401] summary -> {OUT/'summary.json'}")
log(f"[z401] plot    -> {OUT/'ids_vs_vd_family.png'}")
log(f"[z401] done in {time.time()-t0:.1f}s")
LOG.close()
