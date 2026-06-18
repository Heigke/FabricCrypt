"""z381 — S2c pseudo-transient continuation (PTC) for snapback DC fit.

Industry-standard "heavy hammer" parallel to S2a (iii homotopy, z210) and
S2b (two-branch, z379). The idea: replace the failed DC nonlinear solve
F(x)=0 with the ODE  dx/dt = -F(x) (and the actual body-capacitance ODE),
integrate to steady state. Always reaches *a* stable basin; the basin
selected depends on initial condition, mimicking real silicon's time-
domain hopping between snapback states.

CONTEXT:
- S1 (z377) forced Vb=0.8 -> 5.5 dec Ids jump (1D phantom or 2D root?)
- S2 (z378) arc-length: 0 folds across 33 biases (KILL_SHOT for bistability)
- S2c (this) tests whether transient analysis reveals high-Vb basin as
  STABLE in time domain. If PTC-hot stays at high Vb, snapback is bistable;
  if it relaxes to PTC-cold endpoint, high-Vb is unstable.

ALGORITHM:
1. Slow Vd ramp 0 -> 2.0 V over T_ramp = 2 µs (rate 1 V/µs), then
   T_settle = 0.1 µs at Vd=Vmax to settle (well past stiff-ODE time scale
   set by Cj ~ 10 fF, gm ~ 1 µS  -> tau ~ 10 ns).
2. PTC-cold:  Vb0 = 0.0, Vsint0 = 0.0
3. PTC-hot :  Vb0 = 0.80, Vsint0 = 0.5  (probe high-Vb basin)
4. Per-Vd reading: pause the integrator at evenly spaced Vd targets, record
   (Vb, Vsint, Id) at that instant -> "DC" current at that Vd.
5. Three representative biases (VG1, VG2): (0.2, 0.10), (0.4, 0.20), (0.6, 0.20).

GATES (pre-registered):
- INFRA:        no NaN; |dVb/dt| at end < 1e-6 V/ns; wall < 60 min
- DISCOVERY:    PTC-hot fold >= 0.5 dec at VG1=0.6 vs DC-cold baseline
- AMBITIOUS:    PTC-hot cell-wide rmse < 0.5 dec AND fold > 1.5 dec
- KILL_SHOT:    PTC-hot relaxes to PTC-cold endpoint at all 3 biases ->
                high-Vb basin is unstable, snapback is NOT bistable in
                this model. Confirms S2 (z378) verdict.

OUTPUT:
- results/z381_pseudo_transient/{summary.json, vb_trajectory.png, ids_compare_dc_ptc_meas.png, run.log}
"""
from __future__ import annotations
import os, sys, json, math, csv, time, importlib.util, traceback
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
OUT = ROOT / "results/z381_pseudo_transient"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"

LOG_F = open(OUT / "run.log", "w", buffering=1)
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_F.write(line + "\n")


# ---- shared utilities (copy of z378 patterns) -----------------------------

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
    rows = []
    with open(DATA / "2Tcell_BSIM_param_DC.csv") as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try: row[k] = float(v)
                except ValueError: row[k] = float("nan")
            rows.append(row)
    return rows


BRANCH_FLAT = {
    0.4: {"ETAB": 1.9,  "K1": 0.53825, "ALPHA0": 7.842e-05, "BETA0": 19.0, "NFACTOR": 6.0},
    0.6: {"ETAB": 2.5,  "K1": 0.41825, "ALPHA0": 7.842e-05, "BETA0": 20.0, "NFACTOR": 6.0},
}
M2_STATIC = {"k1": 0.63825, "k2": -0.070435, "etab": -0.086777, "beta0": 18.0}

PER_VG1 = {
    0.2: (1889.88, 1.8447, 9.1722),
    0.4: (1092.27, 1.5152, 9.8983),
    0.6: ( 417.63, 0.9036, 6.7846),
}

# Three representative biases for PTC; the third gates DISCOVERY/AMB
REP_BIASES = [(0.2, 0.10), (0.4, 0.20), (0.6, 0.20)]


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


def load_csv(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return d[:, 0], np.abs(d[:, 1])


def measured_path(vg1, vg2):
    sub = DATA / f"2vHCa-2 I-Vs@VG2 VG1={vg1} vnwell=2"
    for f in sorted(sub.glob("*.csv")):
        # Look for VG2 token e.g. "VG2=0.10_" or "VG2=-0.05_"
        if f"VG2={vg2:.2f}_" in f.name or f"VG2={vg2:+.2f}_" in f.name:
            return f
    # Try flexible match
    for f in sorted(sub.glob("*.csv")):
        m = f.name
        for v in (f"VG2={vg2:.2f}", f"VG2={vg2:+.2f}", f"VG2={vg2}"):
            if v in m: return f
    return None


def decade_rmse(Id_pred, Id_meas, floor=1e-15):
    mask = (Id_meas > floor) & (Id_pred > floor) & np.isfinite(Id_pred)
    if mask.sum() < 3:
        return float("nan"), int(mask.sum())
    rm = float(np.sqrt(np.mean((np.log10(Id_pred[mask]) - np.log10(Id_meas[mask]))**2)))
    return rm, int(mask.sum())


def max_forward_jump(Id, Vd, Vd_min=0.5, floor=1e-15):
    if len(Id) < 2: return float("nan")
    dlog = np.diff(np.log10(np.maximum(Id, floor)))
    Vmid = 0.5 * (Vd[1:] + Vd[:-1])
    sel = Vmid >= Vd_min
    if not sel.any() or len(dlog) == 0: return float("nan")
    masked = np.where(sel, dlog, -np.inf)
    return float(masked.max())


# ---- PTC core -------------------------------------------------------------

def run_ptc(cfg, M1, M2, bjt, Vd_targets, vg1, vg2, P_M1, P_M2, sd_M1, sd_M2,
            *, Vb0, Vsint0, T_ramp=2e-6, T_settle=0.1e-6, n_steps_ramp=2000,
            n_steps_settle=200, record_traj=False):
    """Run pseudo-transient continuation.

    Build a time grid Vd(t):
      - Linear ramp from 0 to Vd_max over [0, T_ramp]  with n_steps_ramp pts
      - Constant Vd_max over [T_ramp, T_ramp+T_settle] with n_steps_settle pts

    Run implicit-Euler transient with body capacitor Cj. At each *target* Vd
    in Vd_targets, find nearest sample in the ramp portion and read (Vb,
    Vsint, Id).

    Returns dict with Id (at Vd_targets), Vb (at Vd_targets), Vsint (at
    Vd_targets), full trajectory (if record_traj), dVb_dt_final.
    """
    from nsram.bsim4_port.transient import integrate_2t_transient_implicit
    Bf, iii, Rs = PER_VG1[vg1]
    bjt.Bf = Bf; cfg.iii_body_gain = iii; cfg.vnwell_Rs = Rs
    VG1_t = torch.tensor(vg1, dtype=torch.float64)
    VG2_t = torch.tensor(vg2, dtype=torch.float64)

    Vd_max = float(np.max(Vd_targets))
    # Build time grid: ramp + settle
    t_ramp = np.linspace(0.0, T_ramp, n_steps_ramp, endpoint=False)
    Vd_ramp = Vd_max * (t_ramp / T_ramp)
    t_set = np.linspace(T_ramp, T_ramp + T_settle, n_steps_settle)
    Vd_set = np.full(n_steps_settle, Vd_max)
    t_all = np.concatenate([t_ramp, t_set])
    Vd_all = np.concatenate([Vd_ramp, Vd_set])

    t_t = torch.tensor(t_all, dtype=torch.float64)
    Vd_t = torch.tensor(Vd_all, dtype=torch.float64)

    # DISABLE spike-reset in PTC: we want to OBSERVE whether the basin is
    # stable, not artificially LIF-reset Vb. Use a spike_threshold above any
    # reachable Vb to never trigger.
    with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
        with torch.no_grad():
            out = integrate_2t_transient_implicit(
                cfg, M1, M2, bjt,
                Vd_t=Vd_t, t=t_t, VG1=VG1_t, VG2=VG2_t,
                Vb0=Vb0, Vsint0=Vsint0,
                spike_threshold=10.0, reset_Vb=0.0,
                newton_iters_inner=10, newton_iters_outer=15,
                newton_tol=1e-12, verbose=False)
    Vb_traj = out["Vb"].cpu().numpy()
    Vsint_traj = out["Vsint"].cpu().numpy()
    Id_traj = out["Id"].cpu().numpy()

    # Sample at each Vd_target during the ramp portion only (well-defined Vd-axis)
    Id_at = np.zeros_like(Vd_targets)
    Vb_at = np.zeros_like(Vd_targets)
    Vsint_at = np.zeros_like(Vd_targets)
    for k, vd_target in enumerate(Vd_targets):
        if vd_target <= 0.0:
            Id_at[k] = abs(Id_traj[0]); Vb_at[k] = Vb_traj[0]; Vsint_at[k] = Vsint_traj[0]
            continue
        # nearest ramp sample
        idx = int(np.argmin(np.abs(Vd_ramp - vd_target)))
        Id_at[k] = abs(Id_traj[idx])
        Vb_at[k] = Vb_traj[idx]
        Vsint_at[k] = Vsint_traj[idx]

    # End-of-settle dVb/dt (V/ns)
    if len(t_all) >= 2:
        dVb_dt_end = (Vb_traj[-1] - Vb_traj[-2]) / (t_all[-1] - t_all[-2])
        dVb_dt_end_per_ns = dVb_dt_end * 1e-9
    else:
        dVb_dt_end_per_ns = float("nan")

    res = {"Id_at": Id_at, "Vb_at": Vb_at, "Vsint_at": Vsint_at,
           "dVb_dt_end_V_per_ns": float(dVb_dt_end_per_ns),
           "Vb_end": float(Vb_traj[-1]),
           "Vsint_end": float(Vsint_traj[-1]),
           "Id_end": float(abs(Id_traj[-1]))}
    if record_traj:
        res["t_all"] = t_all
        res["Vd_all"] = Vd_all
        res["Vb_traj"] = Vb_traj
        res["Vsint_traj"] = Vsint_traj
        res["Id_traj"] = Id_traj
    return res


def run_dc_cold(cfg, M1, M2, bjt, Vd_t, vg1, vg2, P_M1, P_M2, sd_M1, sd_M2):
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    Bf, iii, Rs = PER_VG1[vg1]
    bjt.Bf = Bf; cfg.iii_body_gain = iii; cfg.vnwell_Rs = Rs
    with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
        with torch.no_grad():
            out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_t,
                             VG1=torch.tensor(vg1, dtype=torch.float64),
                             VG2=torch.tensor(vg2, dtype=torch.float64),
                             warm_start=True)
    return out


def main():
    t0 = time.time()
    log("z381 — S2c pseudo-transient continuation (PTC) for snapback DC fit")
    log("Gates: INFRA=no NaN, |dVb/dt|<1e-6 V/ns; DISCOVERY=PTC-hot fold>=0.5 dec @VG1=0.6;")
    log("       AMBITIOUS=cell rmse<0.5 AND fold>1.5; KILL_SHOT=PTC-hot relaxes to PTC-cold")
    cfg, M1, M2, bjt = build_base()
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    rows = load_sebas_params()

    per_bias = []
    # For diagnostic plot at VG1=0.6 VG2=0.2 we record trajectories
    record_bias = (0.6, 0.20)
    traj_record = None

    for (vg1, vg2) in REP_BIASES:
        fpath = measured_path(vg1, vg2)
        if fpath is None:
            log(f"  VG1={vg1} VG2={vg2}: measured CSV not found, skipping")
            per_bias.append({"VG1": vg1, "VG2": vg2, "error": "no measured CSV"})
            continue
        log(f"--- BIAS VG1={vg1} VG2={vg2:+.2f}  file={fpath.name} ---")
        Vd_m, Id_m = load_csv(fpath)
        Vd_t = torch.tensor(Vd_m, dtype=torch.float64)
        row = find_or_impute_row(rows, vg1, vg2)
        P_M1, P_M2 = make_overrides(row)

        # DC cold-start baseline
        try:
            t1 = time.time()
            out_dc = run_dc_cold(cfg, M1, M2, bjt, Vd_t, vg1, vg2, P_M1, P_M2, sd_M1, sd_M2)
            Id_dc = np.abs(out_dc["Id"].detach().cpu().numpy())
            t_dc = time.time() - t1
            log(f"  DC cold ({t_dc:.2f}s) ok")
        except Exception as e:
            log(f"  DC cold FAILED: {e}")
            traceback.print_exc(file=LOG_F)
            Id_dc = np.full_like(Vd_m, np.nan)

        record_this = (abs(vg1 - record_bias[0]) < 1e-6 and abs(vg2 - record_bias[1]) < 1e-6)

        # PTC-cold (Vb0=0)
        try:
            t1 = time.time()
            ptc_cold = run_ptc(cfg, M1, M2, bjt, Vd_m, vg1, vg2, P_M1, P_M2, sd_M1, sd_M2,
                               Vb0=0.0, Vsint0=0.0, record_traj=record_this)
            Id_pc = ptc_cold["Id_at"]
            t_pc = time.time() - t1
            log(f"  PTC-cold ({t_pc:.2f}s) Vb_end={ptc_cold['Vb_end']:+.4f}  "
                f"|dVb/dt|={abs(ptc_cold['dVb_dt_end_V_per_ns']):.2e} V/ns")
        except Exception as e:
            log(f"  PTC-cold FAILED: {e}")
            traceback.print_exc(file=LOG_F)
            ptc_cold = {"Id_at": np.full_like(Vd_m, np.nan), "Vb_end": float("nan"),
                        "Vsint_end": float("nan"), "Id_end": float("nan"),
                        "dVb_dt_end_V_per_ns": float("nan")}
            Id_pc = ptc_cold["Id_at"]

        # PTC-hot (Vb0=0.8)
        try:
            t1 = time.time()
            ptc_hot = run_ptc(cfg, M1, M2, bjt, Vd_m, vg1, vg2, P_M1, P_M2, sd_M1, sd_M2,
                              Vb0=0.80, Vsint0=0.5, record_traj=record_this)
            Id_ph = ptc_hot["Id_at"]
            t_ph = time.time() - t1
            log(f"  PTC-hot  ({t_ph:.2f}s) Vb_end={ptc_hot['Vb_end']:+.4f}  "
                f"|dVb/dt|={abs(ptc_hot['dVb_dt_end_V_per_ns']):.2e} V/ns")
        except Exception as e:
            log(f"  PTC-hot FAILED: {e}")
            traceback.print_exc(file=LOG_F)
            ptc_hot = {"Id_at": np.full_like(Vd_m, np.nan), "Vb_end": float("nan"),
                       "Vsint_end": float("nan"), "Id_end": float("nan"),
                       "dVb_dt_end_V_per_ns": float("nan")}
            Id_ph = ptc_hot["Id_at"]

        if record_this:
            traj_record = {"vg1": vg1, "vg2": vg2,
                            "cold": ptc_cold, "hot": ptc_hot,
                            "Vd_m": Vd_m, "Id_m": Id_m,
                            "Id_dc": Id_dc}

        rmse_dc, _  = decade_rmse(Id_dc, Id_m)
        rmse_pc, _  = decade_rmse(Id_pc, Id_m)
        rmse_ph, _  = decade_rmse(Id_ph, Id_m)
        jump_m  = max_forward_jump(Id_m,  Vd_m)
        jump_dc = max_forward_jump(Id_dc, Vd_m)
        jump_pc = max_forward_jump(Id_pc, Vd_m)
        jump_ph = max_forward_jump(Id_ph, Vd_m)
        # Basin separation: end-Vb difference between hot and cold
        basin_dVb = ptc_hot["Vb_end"] - ptc_cold["Vb_end"]
        # Relax-to-cold detection: hot Id_end within 0.3 dec of cold Id_end
        relax = (np.isfinite(ptc_cold["Id_end"]) and np.isfinite(ptc_hot["Id_end"]) and
                 ptc_cold["Id_end"] > 1e-14 and ptc_hot["Id_end"] > 1e-14 and
                 abs(math.log10(max(ptc_hot["Id_end"], 1e-15)) -
                     math.log10(max(ptc_cold["Id_end"], 1e-15))) < 0.3)

        log(f"  RMSE dec: DC={rmse_dc:.3f} PTC-cold={rmse_pc:.3f} PTC-hot={rmse_ph:.3f}")
        log(f"  Jump dec: meas={jump_m:.2f} DC={jump_dc:.2f} PTC-cold={jump_pc:.2f} PTC-hot={jump_ph:.2f}")
        log(f"  Basin: dVb(hot-cold)={basin_dVb:+.4f} V  relax_to_cold={relax}")

        per_bias.append({
            "VG1": float(vg1), "VG2": float(vg2),
            "file": fpath.name,
            "rmse_dc_dec": rmse_dc,
            "rmse_ptc_cold_dec": rmse_pc,
            "rmse_ptc_hot_dec":  rmse_ph,
            "jump_meas_dec":     jump_m,
            "jump_dc_dec":       jump_dc,
            "jump_ptc_cold_dec": jump_pc,
            "jump_ptc_hot_dec":  jump_ph,
            "Vb_end_cold":  ptc_cold["Vb_end"],
            "Vb_end_hot":   ptc_hot["Vb_end"],
            "Vsint_end_cold": ptc_cold["Vsint_end"],
            "Vsint_end_hot":  ptc_hot["Vsint_end"],
            "Id_end_cold":  ptc_cold["Id_end"],
            "Id_end_hot":   ptc_hot["Id_end"],
            "dVbdt_end_cold_V_per_ns": ptc_cold["dVb_dt_end_V_per_ns"],
            "dVbdt_end_hot_V_per_ns":  ptc_hot["dVb_dt_end_V_per_ns"],
            "basin_dVb": basin_dVb,
            "relax_hot_to_cold": bool(relax),
        })

    # ---------- aggregate gates ----------
    def vals(key):
        out = [b.get(key) for b in per_bias if isinstance(b.get(key), (int, float)) and math.isfinite(b.get(key))]
        return out

    cellwide_dc  = float(np.median(vals("rmse_dc_dec")))         if vals("rmse_dc_dec")        else float("nan")
    cellwide_pc  = float(np.median(vals("rmse_ptc_cold_dec")))   if vals("rmse_ptc_cold_dec")  else float("nan")
    cellwide_ph  = float(np.median(vals("rmse_ptc_hot_dec")))    if vals("rmse_ptc_hot_dec")   else float("nan")

    fold_vg06_hot = next((b["jump_ptc_hot_dec"] for b in per_bias
                         if b.get("VG1") == 0.6 and isinstance(b.get("jump_ptc_hot_dec"), float)
                         and math.isfinite(b["jump_ptc_hot_dec"])), float("nan"))
    fold_vg06_dc = next((b["jump_dc_dec"] for b in per_bias
                        if b.get("VG1") == 0.6 and isinstance(b.get("jump_dc_dec"), float)
                        and math.isfinite(b["jump_dc_dec"])), float("nan"))
    fold_gap_vg06 = (fold_vg06_hot - fold_vg06_dc) if (math.isfinite(fold_vg06_hot) and math.isfinite(fold_vg06_dc)) else float("nan")

    n_relax = sum(1 for b in per_bias if b.get("relax_hot_to_cold") is True)
    n_ok = sum(1 for b in per_bias if "error" not in b)
    n_nan = sum(1 for b in per_bias
                 if not math.isfinite(b.get("rmse_ptc_hot_dec", float("nan"))))

    max_abs_dvbdt = max(
        (abs(b.get("dVbdt_end_hot_V_per_ns", float("nan"))) for b in per_bias
         if isinstance(b.get("dVbdt_end_hot_V_per_ns"), (int, float))
         and math.isfinite(b.get("dVbdt_end_hot_V_per_ns"))),
        default=float("nan"))
    max_abs_dvbdt_cold = max(
        (abs(b.get("dVbdt_end_cold_V_per_ns", float("nan"))) for b in per_bias
         if isinstance(b.get("dVbdt_end_cold_V_per_ns"), (int, float))
         and math.isfinite(b.get("dVbdt_end_cold_V_per_ns"))),
        default=float("nan"))

    wall_s = time.time() - t0
    settled = (math.isfinite(max_abs_dvbdt) and math.isfinite(max_abs_dvbdt_cold)
                and max(max_abs_dvbdt, max_abs_dvbdt_cold) < 1e-6)
    gate_infra = (n_nan == 0 and wall_s < 60 * 60 and settled)
    gate_disc = (math.isfinite(fold_gap_vg06) and fold_gap_vg06 >= 0.5)
    gate_amb  = (math.isfinite(cellwide_ph) and cellwide_ph < 0.5
                  and math.isfinite(fold_vg06_hot) and fold_vg06_hot > 1.5)
    # KILL_SHOT: hot relaxes to cold at ALL biases tested -> basin is unstable
    gate_kill = (n_ok >= 1 and n_relax == n_ok)

    if gate_amb: verdict = "AMBITIOUS"
    elif gate_disc: verdict = "DISCOVERY"
    elif gate_kill: verdict = "KILL_SHOT"
    elif gate_infra: verdict = "INFRA_ONLY"
    else: verdict = "INFRA_FAIL"

    summary = {
        "script": "z381_pseudo_transient_dc",
        "wall_s": wall_s,
        "n_biases": len(REP_BIASES),
        "rep_biases": REP_BIASES,
        "cellwide_median_rmse_dc_dec":        cellwide_dc,
        "cellwide_median_rmse_ptc_cold_dec":  cellwide_pc,
        "cellwide_median_rmse_ptc_hot_dec":   cellwide_ph,
        "fold_vg06_dc_dec":   fold_vg06_dc,
        "fold_vg06_hot_dec":  fold_vg06_hot,
        "fold_gap_vg06_hot_minus_dc_dec": fold_gap_vg06,
        "n_relax_hot_to_cold": n_relax,
        "n_ok": n_ok,
        "n_nan": n_nan,
        "max_abs_dVbdt_end_V_per_ns_cold": max_abs_dvbdt_cold,
        "max_abs_dVbdt_end_V_per_ns_hot":  max_abs_dvbdt,
        "settled":  settled,
        "gate_INFRA":      gate_infra,
        "gate_DISCOVERY":  gate_disc,
        "gate_AMBITIOUS":  gate_amb,
        "gate_KILL_SHOT":  gate_kill,
        "verdict": verdict,
        "per_VG1_params": {f"{k}": list(v) for k, v in PER_VG1.items()},
        "per_bias": per_bias,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log("\n=== SUMMARY ===")
    log(f"  Cell-wide median RMSE: DC={cellwide_dc:.3f} | PTC-cold={cellwide_pc:.3f} | PTC-hot={cellwide_ph:.3f} dec")
    log(f"  Fold VG1=0.6: DC={fold_vg06_dc:.2f} -> PTC-hot={fold_vg06_hot:.2f} (gap={fold_gap_vg06:+.2f} dec)")
    log(f"  Hot relaxes to cold at {n_relax}/{n_ok} biases  |  max |dVb/dt|_hot={max_abs_dvbdt:.2e} V/ns")
    log(f"  Verdict: {verdict}  |  wall={wall_s:.1f}s")

    # ---------- Plot 1: Vb(t) trajectory at record_bias ----------
    if traj_record is not None and "t_all" in traj_record["hot"]:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax = axes[0]
        ax.plot(traj_record["cold"]["t_all"] * 1e9, traj_record["cold"]["Vb_traj"],
                "b-", lw=1.5, label="PTC-cold (Vb0=0)")
        ax.plot(traj_record["hot"]["t_all"] * 1e9, traj_record["hot"]["Vb_traj"],
                "r-", lw=1.5, label="PTC-hot (Vb0=0.8)")
        ax.plot(traj_record["hot"]["t_all"] * 1e9, traj_record["hot"]["Vd_all"],
                "k--", lw=0.8, alpha=0.5, label="Vd(t) ramp")
        ax.set_ylabel("Vb (V)  |  Vd (V)")
        ax.set_title(f"z381 PTC body voltage trajectory @ VG1={traj_record['vg1']} VG2={traj_record['vg2']:+.2f}")
        ax.grid(True, alpha=0.3); ax.legend(loc="upper left")
        ax = axes[1]
        ax.semilogy(traj_record["cold"]["t_all"] * 1e9,
                    np.maximum(np.abs(traj_record["cold"]["Id_traj"]), 1e-15),
                    "b-", lw=1.5, label="|Id| PTC-cold")
        ax.semilogy(traj_record["hot"]["t_all"] * 1e9,
                    np.maximum(np.abs(traj_record["hot"]["Id_traj"]), 1e-15),
                    "r-", lw=1.5, label="|Id| PTC-hot")
        ax.set_xlabel("t (ns)"); ax.set_ylabel("|Id| (A)")
        ax.set_ylim(1e-13, 1e-2); ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT / "vb_trajectory.png", dpi=150, bbox_inches="tight")
        log(f"Wrote {OUT/'vb_trajectory.png'}")

    # ---------- Plot 2: 3-panel Ids comparison ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (vg1, vg2) in zip(axes, REP_BIASES):
        cand = [b for b in per_bias if b.get("VG1") == vg1 and abs(b.get("VG2", 99) - vg2) < 1e-3]
        if not cand or "error" in cand[0]:
            ax.set_title(f"VG1={vg1} VG2={vg2:+.2f} (no data)"); continue
        b = cand[0]
        fpath = measured_path(vg1, vg2)
        if fpath is None: continue
        Vd_m, Id_m = load_csv(fpath)
        Vd_t = torch.tensor(Vd_m, dtype=torch.float64)
        row = find_or_impute_row(rows, vg1, vg2)
        P_M1, P_M2 = make_overrides(row)
        # Re-run for plot (small overhead, ensures plot matches recorded results)
        try:
            out_dc = run_dc_cold(cfg, M1, M2, bjt, Vd_t, vg1, vg2, P_M1, P_M2, sd_M1, sd_M2)
            Id_dc = np.abs(out_dc["Id"].detach().cpu().numpy())
        except Exception:
            Id_dc = np.full_like(Vd_m, np.nan)
        try:
            ptc_cold = run_ptc(cfg, M1, M2, bjt, Vd_m, vg1, vg2, P_M1, P_M2, sd_M1, sd_M2,
                                Vb0=0.0, Vsint0=0.0)
            Id_pc = ptc_cold["Id_at"]
        except Exception:
            Id_pc = np.full_like(Vd_m, np.nan)
        try:
            ptc_hot = run_ptc(cfg, M1, M2, bjt, Vd_m, vg1, vg2, P_M1, P_M2, sd_M1, sd_M2,
                               Vb0=0.80, Vsint0=0.5)
            Id_ph = ptc_hot["Id_at"]
        except Exception:
            Id_ph = np.full_like(Vd_m, np.nan)
        ax.semilogy(Vd_m, np.maximum(Id_m, 1e-15), "k.", ms=4, label="measured")
        ax.semilogy(Vd_m, np.maximum(Id_dc, 1e-15), "b-",  lw=1.0, alpha=0.7,
                    label=f"DC cold (RMSE={b['rmse_dc_dec']:.2f})")
        ax.semilogy(Vd_m, np.maximum(Id_pc, 1e-15), "g--", lw=1.2,
                    label=f"PTC-cold (RMSE={b['rmse_ptc_cold_dec']:.2f})")
        ax.semilogy(Vd_m, np.maximum(Id_ph, 1e-15), "r-",  lw=1.6,
                    label=f"PTC-hot  (RMSE={b['rmse_ptc_hot_dec']:.2f})")
        ax.set_xlabel("Vd (V)"); ax.set_ylabel("|Id| (A)")
        ax.set_ylim(1e-13, 1e-2); ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"VG1={vg1} VG2={vg2:+.2f}\n"
                     f"jump meas={b['jump_meas_dec']:.1f} | DC={b['jump_dc_dec']:.1f} | "
                     f"PTC-hot={b['jump_ptc_hot_dec']:.1f} dec\n"
                     f"Vb_end cold={b['Vb_end_cold']:+.2f} hot={b['Vb_end_hot']:+.2f} V "
                     f"relax={b['relax_hot_to_cold']}", fontsize=8)
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle(f"z381 — pseudo-transient continuation | "
                 f"cell DC={cellwide_dc:.2f} -> PTC-hot={cellwide_ph:.2f} dec | verdict={verdict}",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "ids_compare_dc_ptc_meas.png", dpi=150, bbox_inches="tight")
    log(f"Wrote {OUT/'ids_compare_dc_ptc_meas.png'}")
    LOG_F.close()


if __name__ == "__main__":
    main()
