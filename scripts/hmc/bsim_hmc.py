"""z500 — Bayesian posterior over BSIM4 + snapback + body params (Mario silicon).

WHAT
----
Posterior MCMC over ~12 scalar log-multipliers applied on top of Sebas's per-bias
extracted card. Goal is NOT to find a single fit; goal is to characterise the
posterior so we can answer:

  1. Does ANY parameter region in the posterior reach DC dec <= 1.5 ?
     (DISCOVERY: tuning was the issue, not topology)
  2. Is the posterior multi-modal / wild ?
     (INFORMATIVE: topology gap confirmed; tells us WHICH params drift)
  3. Is the posterior unimodal but stuck at >2.5 dec ?
     (KILLSHOT: data + topology both insufficient)

WHY EMCEE (not NumPyro HMC)
---------------------------
The forward solve has the IFT autograd path (z474b) so gradients exist, but
the Newton loop has Python branches (line-search, snap gating) that JAX
cannot trace, and bridging torch <-> JAX for NumPyro is impractical inside
the 6h budget. emcee's affine-invariant ensemble sampler is gradient-free
and well-suited for ~12-D posteriors with O(10^4) likelihood evals.

PARAMS (12 free, log10-space mu=0 perturbations on global multipliers)
----------------------------------------------------------------------
   1. log_u0_mul       sd_M1.scaled['u0']         * 10**x   prior N(0, 0.30)
   2. add_vth0         sd_M1.scaled['vth0']       additive in V, N(0, 0.05)
   3. log_vsat_mul     sd_M1.scaled['vsat']                  N(0, 0.20)
   4. log_rdsw_mul     sd_M1.scaled['rdsw']                  N(0, 0.40)
   5. log_eta0_mul     sd_M1.scaled['eta0']                  N(0, 0.30)
   6. log_pclm_mul     sd_M1.scaled['pclm']                  N(0, 0.30)
   7. log_a0_mul       sd_M1.scaled['a0']                    N(0, 0.30)
   8. log_alpha0_mul   sd_M1.scaled['alpha0']  (II)          N(0, 0.40)
   9. log_snap_Is      cfg.snap_Is             (log)         N(0, 0.50)
  10. log_snap_Bf      cfg.snap_Bf                           N(0, 0.40)
  11. log_jss_mul      (reserved leakage knob)               N(0, 0.50)
  12. add_npn_dV_BE    cfg.snap_npn_V_BE_offset additive (V) N(0, 0.10)

LIKELIHOOD
----------
For each posterior sample, compute model Id at ~33 (VG1, VG2) biases × 8 Vd
points each. Log10(Id) Gaussian with sigma = 0.30 dec.

OUTPUTS
-------
  posterior_samples.npy           (n_chains, n_walkers, n_steps, n_dim)
  log_prob.npy                    (n_chains, n_walkers, n_steps)
  posterior_traces.png            per-param mean trace per chain
  corner.png                      corner of post-warmup samples
  ess_per_param.json
  identifiability_table.md
  posterior_dec_distribution.json (100 posterior samples → dec)
  posterior_best_audit.json
  honest_verdict.md
  run.log

USAGE
-----
  nohup python scripts/hmc/bsim_hmc.py > results/z500_bsim_hmc/nohup.out 2>&1 &
"""
from __future__ import annotations
import json
import math
import os
import sys
import time
import importlib.util as _ilu
import traceback
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/z500_bsim_hmc"
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "run.log", "w")


def log(*a):
    s = "[" + time.strftime("%H:%M:%S") + "] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n"); LOG.flush()


log("=== z500 BSIM4 Bayesian posterior MCMC ===")
log(f"ROOT = {ROOT}")
log(f"OUT  = {OUT}")
log(f"torch={torch.__version__}  cuda_avail={torch.cuda.is_available()}  "
    f"devices={torch.cuda.device_count()}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log(f"using device: {DEVICE}")

# ---------- Load pyport stack via z454 → z449 → z427 → z91f ----------
_spec454 = _ilu.spec_from_file_location("z454", ROOT / "scripts/z454_snapback_integration.py")
z454 = _ilu.module_from_spec(_spec454); _spec454.loader.exec_module(z454)
z449 = z454.z449
z427 = z454.z427
z429 = z454.z429
z91f = z427.z91f

from nsram.bsim4_port.forward_2t_batched_gpu import forward_2t_gpu_batched
from nsram.bsim4_port.bjt import GummelPoonNPN

# ---------- Models / curves / sebas ----------
log("Loading M1/M2 models and Sebas curves …")
model_M1, model_M2 = z429.build_models()
sebas_rows = z91f.load_sebas_params()
curves_all = z91f.load_curves()
log(f"  loaded {len(curves_all)} curves, {len(sebas_rows)} sebas rows")

VALID = []
for c in curves_all:
    row = z91f.find_params(sebas_rows, c["VG1"], c["VG2"])
    if row is None or math.isnan(row.get("K1", float("nan"))):
        continue
    VALID.append((c, row))
log(f"  {len(VALID)} biases with valid Sebas params")

N_VD = 8
VD_GRID = torch.tensor(np.geomspace(0.05, 2.0, N_VD),
                       dtype=torch.float64, device=DEVICE)
VG1_PER_BIAS = torch.tensor([c["VG1"] for c, _ in VALID],
                            dtype=torch.float64, device=DEVICE)
VG2_PER_BIAS = torch.tensor([c["VG2"] for c, _ in VALID],
                            dtype=torch.float64, device=DEVICE)
N_BIAS = len(VALID)

ID_MEAS = torch.zeros((N_BIAS, N_VD), dtype=torch.float64)
for i, (c, _) in enumerate(VALID):
    Vd = c["Vd"].numpy(); Id = c["Id"].numpy()
    keep = (Id > 0) & np.isfinite(Id)
    Vd_k = Vd[keep]; Id_k = Id[keep]
    order = np.argsort(Vd_k)
    Vd_k = Vd_k[order]; Id_k = Id_k[order]
    lg = np.log10(np.maximum(Id_k, 1e-15))
    lg_resamp = np.interp(VD_GRID.cpu().numpy(), Vd_k, lg, left=lg[0], right=lg[-1])
    ID_MEAS[i] = torch.from_numpy(10.0 ** lg_resamp)
ID_MEAS = ID_MEAS.to(device=DEVICE)
log(f"  ID_MEAS shape={tuple(ID_MEAS.shape)} min={ID_MEAS.min().item():.3e} max={ID_MEAS.max().item():.3e}")


def _row_field(rows, name, default=float("nan")):
    out = []
    for _, r in rows:
        v = r.get(name, default)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(default)
        else:
            out.append(v)
    return out


P_M1_SEBAS = {
    "etab":   torch.tensor(_row_field(VALID, "ETAB"),   dtype=torch.float64, device=DEVICE),
    "k1":     torch.tensor(_row_field(VALID, "K1"),     dtype=torch.float64, device=DEVICE),
    "alpha0": torch.tensor(_row_field(VALID, "ALPHA0"), dtype=torch.float64, device=DEVICE),
    "beta0":  torch.tensor(_row_field(VALID, "BETA0"),  dtype=torch.float64, device=DEVICE),
}
P_M2_SEBAS = {
    "nfactor": torch.tensor(_row_field(VALID, "NFACTOR"), dtype=torch.float64, device=DEVICE),
}
BJT_AREA = torch.tensor([float(r.get("area", 1e-6)) * float(r.get("mbjt", 1.0))
                         for _, r in VALID], dtype=torch.float64, device=DEVICE)
BJT_IS_SEBAS = torch.tensor([float(r.get("IS", 1e-9)) for _, r in VALID],
                             dtype=torch.float64, device=DEVICE)

# ---------- cfg / sd templates ----------
V449B_BASE = {"use_vbic_for_q1": True, "vbic_AVC1": 0.5, "vbic_AVC2": 0.5,
              "Cbody": 1e-15, "body_pdiode_Cj0_per_area": 0.0}
SNAP_HOT = dict(snap_BV=2.0*0.6, snap_n_avl=4.0, snap_Bf=417.0, snap_Va=0.90,
                snap_Is=4.5192e-12, snap_Nf=1.0,
                snap_Id_clamp=1e-1, snap_Iii_clamp=1e-1,
                snap_use_knee_gate=True,
                snap_V_knee=1.6, snap_V_sharp=0.05,
                snap_npn_gate_mode="current",
                snap_npn_V_knee=1.8, snap_npn_V_sharp=0.05,
                snap_npn_V_BE_offset=0.3)
BASE_CFG_FLAGS = {**V449B_BASE, "use_snapback_sub": True, **SNAP_HOT}

_cfg_template, sd_M1_TEMPLATE, sd_M2_TEMPLATE = z427.make_cfg(model_M1, model_M2,
                                                              dict(BASE_CFG_FLAGS))
# Disable Mario PWL Ipos — it uses .item() on VG1, incompatible with vector VG1.
# We still get the dominant DC behaviour from the BSIM channel + snapback.
_cfg_template.use_mario_ipos = False
# Same for any other features that assume scalar VG1 (best-effort).
for _flag in ("use_m3_bss145",):
    if hasattr(_cfg_template, _flag):
        setattr(_cfg_template, _flag, False)


def _sd_to_device(sd, device):
    new_scaled = {}
    for k, v in sd.scaled.items():
        if torch.is_tensor(v):
            new_scaled[k] = v.to(device=device, dtype=torch.float64)
        else:
            new_scaled[k] = torch.tensor(float(v), dtype=torch.float64, device=device)
    sd.scaled = new_scaled
    return sd


sd_M1_TEMPLATE = _sd_to_device(sd_M1_TEMPLATE, DEVICE)
sd_M2_TEMPLATE = _sd_to_device(sd_M2_TEMPLATE, DEVICE)

_PERTURB_KEYS_M1 = ["u0", "vth0", "vsat", "rdsw", "eta0", "pclm", "a0"]
SD_M1_BASE = {}
for k in _PERTURB_KEYS_M1:
    if k in sd_M1_TEMPLATE.scaled:
        SD_M1_BASE[k] = sd_M1_TEMPLATE.scaled[k].clone()
    else:
        log(f"  WARN: sd_M1 missing key {k!r} — skipping perturbation for it")

ALPHA0_BASE = P_M1_SEBAS["alpha0"].clone()

PARAM_NAMES = [
    "u0", "vth0", "vsat", "rdsw", "eta0", "pclm", "a0",
    "alpha0", "snap_Is", "snap_Bf", "jss", "npn_dV_BE",
]
PRIOR_SIGMA = np.array([0.30, 0.05, 0.20, 0.40, 0.30, 0.30, 0.30,
                         0.40, 0.50, 0.40, 0.50, 0.10])
N_DIM = len(PARAM_NAMES)


def _apply_and_eval(theta, want_dec=False):
    """Apply sample, run batched forward, return (log_lik, dec) tuple."""
    try:
        x = theta
        mul_u0   = 10.0 ** x[0]; add_vth = float(x[1])
        mul_vsat = 10.0 ** x[2]; mul_rdsw = 10.0 ** x[3]
        mul_eta0 = 10.0 ** x[4]; mul_pclm = 10.0 ** x[5]
        mul_a0   = 10.0 ** x[6]; mul_alpha0 = 10.0 ** x[7]
        snap_Is_val = 10.0 ** (math.log10(BASE_CFG_FLAGS["snap_Is"]) + x[8])
        snap_Bf_val = 10.0 ** (math.log10(BASE_CFG_FLAGS["snap_Bf"]) + x[9])
        # x[10] (jss) is reserved — no current backing in sd; harmless
        dV_BE = SNAP_HOT["snap_npn_V_BE_offset"] + float(x[11])

        saved = {k: sd_M1_TEMPLATE.scaled[k] for k in SD_M1_BASE}
        try:
            if "u0"   in SD_M1_BASE: sd_M1_TEMPLATE.scaled["u0"]   = SD_M1_BASE["u0"]   * mul_u0
            if "vsat" in SD_M1_BASE: sd_M1_TEMPLATE.scaled["vsat"] = SD_M1_BASE["vsat"] * mul_vsat
            if "rdsw" in SD_M1_BASE: sd_M1_TEMPLATE.scaled["rdsw"] = SD_M1_BASE["rdsw"] * mul_rdsw
            if "eta0" in SD_M1_BASE: sd_M1_TEMPLATE.scaled["eta0"] = SD_M1_BASE["eta0"] * mul_eta0
            if "pclm" in SD_M1_BASE: sd_M1_TEMPLATE.scaled["pclm"] = SD_M1_BASE["pclm"] * mul_pclm
            if "a0"   in SD_M1_BASE: sd_M1_TEMPLATE.scaled["a0"]   = SD_M1_BASE["a0"]   * mul_a0
            if "vth0" in SD_M1_BASE: sd_M1_TEMPLATE.scaled["vth0"] = SD_M1_BASE["vth0"] + add_vth

            # NOTE: leak.compute_iimpact uses `if (tmp<=0) or (beta0<=0):`
            # which fails for tensor inputs. Keep alpha0/beta0 as SCALARS
            # (mean over biases). Per-bias variation is preserved via k1/etab/nfactor.
            sd_M1_TEMPLATE.scaled["k1"]     = P_M1_SEBAS["k1"]
            sd_M1_TEMPLATE.scaled["etab"]   = P_M1_SEBAS["etab"]
            sd_M1_TEMPLATE.scaled["beta0"]  = torch.tensor(
                float(P_M1_SEBAS["beta0"].mean().item()),
                dtype=torch.float64, device=DEVICE)
            sd_M1_TEMPLATE.scaled["alpha0"] = torch.tensor(
                float(ALPHA0_BASE.mean().item()) * mul_alpha0,
                dtype=torch.float64, device=DEVICE)
            sd_M2_TEMPLATE.scaled["nfactor"] = P_M2_SEBAS["nfactor"]

            cfg = _cfg_template
            old_Is, old_Bf, old_dV = cfg.snap_Is, cfg.snap_Bf, cfg.snap_npn_V_BE_offset
            cfg.snap_Is = float(snap_Is_val)
            cfg.snap_Bf = float(snap_Bf_val)
            cfg.snap_npn_V_BE_offset = float(dV_BE)

            bjt = GummelPoonNPN.from_sebas_card()
            bjt.Is = float(BJT_IS_SEBAS.mean().item())
            bjt.area = float(BJT_AREA.mean().item())

            with torch.no_grad():
                out = forward_2t_gpu_batched(
                    cfg, model_M1, model_M2, bjt,
                    VD_GRID, VG1_PER_BIAS, VG2_PER_BIAS,
                    max_iters=20, tol=1e-9, eps=1e-5,
                    dtype=torch.float64, device=DEVICE,
                    compile_mode="off", early_stop=True, verbose=False)
            Id_pred = out["Id"].abs() + 1e-15
            conv = out["converged"]
            lp_pred = torch.log10(Id_pred)
            lm_meas = torch.log10(ID_MEAS)
            r = lp_pred - lm_meas
            r_masked = r[conv]
            cfg.snap_Is, cfg.snap_Bf, cfg.snap_npn_V_BE_offset = old_Is, old_Bf, old_dV

            if r_masked.numel() == 0:
                return (-1e9, float("nan"))
            sigma = 0.30
            ll = (-0.5 * (r_masked / sigma).pow(2).sum().item()
                  - r_masked.numel() * math.log(sigma * math.sqrt(2 * math.pi)))
            n_miss = (~conv).sum().item()
            ll -= 0.5 * n_miss

            dec_val = float("nan")
            if want_dec:
                abs_r = r.abs()
                # per-curve quad-mean over valid points
                w = conv.double()
                num = (abs_r.pow(2) * w).sum(dim=1)
                den = w.sum(dim=1).clamp_min(1.0)
                per_curve = torch.sqrt(num / den)
                valid_curves = (w.sum(dim=1) > 0)
                if valid_curves.sum().item() == 0:
                    dec_val = float("nan")
                else:
                    dec_val = float(per_curve[valid_curves].mean().item())
            return (float(ll), dec_val)
        finally:
            for k, v in saved.items():
                sd_M1_TEMPLATE.scaled[k] = v
    except Exception:
        traceback.print_exc(file=LOG); LOG.flush()
        return (-1e9, float("nan"))


def log_prior(theta):
    z = theta / PRIOR_SIGMA
    if not np.all(np.isfinite(z)):
        return -np.inf
    return float(-0.5 * np.sum(z * z))


def log_prob(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll, _ = _apply_and_eval(theta, want_dec=False)
    return lp + ll


# ---------- Sanity ----------
log("Sanity: evaluating log_prob at theta = 0 (current Mario fit)…")
t0 = time.time(); ll0, dec0 = _apply_and_eval(np.zeros(N_DIM), want_dec=True)
log(f"  log_lik(theta=0) = {ll0:.2f}  dec(theta=0) = {dec0:.3f}  [{time.time()-t0:.2f}s]")
t0 = time.time(); ll0b, _ = _apply_and_eval(np.zeros(N_DIM), want_dec=False)
log(f"  warm log_lik(theta=0) = {ll0b:.2f}  [{time.time()-t0:.2f}s]")

# ---------- emcee ----------
import emcee

N_CHAINS  = int(os.environ.get("HMC_NCHAIN",  4))
N_WALKERS = int(os.environ.get("HMC_NWALK",  32))
N_WARMUP  = int(os.environ.get("HMC_NWARM",  200))
N_STEPS   = int(os.environ.get("HMC_NSTEPS", 1000))
log(f"emcee config: chains={N_CHAINS} walkers={N_WALKERS} "
    f"warmup={N_WARMUP} steps={N_STEPS}  dim={N_DIM}")

rng = np.random.default_rng(20260518)
all_samples = np.zeros((N_CHAINS, N_WALKERS, N_STEPS, N_DIM), dtype=np.float32)
all_logp    = np.zeros((N_CHAINS, N_WALKERS, N_STEPS),         dtype=np.float32)

t_grand = time.time()
for c in range(N_CHAINS):
    # Init ball: 0.3*prior_sigma but with a floor at 0.02 to avoid singular walker matrix.
    init_sigma = np.maximum(0.3 * PRIOR_SIGMA, 0.02)
    p0 = rng.normal(0.0, init_sigma, size=(N_WALKERS, N_DIM))
    sampler = emcee.EnsembleSampler(N_WALKERS, N_DIM, log_prob)
    log(f"chain {c+1}/{N_CHAINS}: warm-up {N_WARMUP} steps…")
    t0 = time.time()
    state = sampler.run_mcmc(p0, N_WARMUP, progress=False, store=False)
    log(f"  warm-up done in {time.time()-t0:.1f}s; "
        f"acc_frac={float(np.mean(sampler.acceptance_fraction)):.3f}")
    sampler.reset()
    log(f"chain {c+1}: sampling {N_STEPS} steps…")
    t0 = time.time()
    sampler.run_mcmc(state, N_STEPS, progress=False, store=True)
    chain = sampler.get_chain()
    lp    = sampler.get_log_prob()
    all_samples[c] = chain.transpose(1, 0, 2).astype(np.float32)
    all_logp[c]    = lp.T.astype(np.float32)
    log(f"  chain {c+1} done in {time.time()-t0:.1f}s; "
        f"acc_frac={float(np.mean(sampler.acceptance_fraction)):.3f}; "
        f"best_lp={float(lp.max()):.2f}")
    # incremental dump in case we OOM
    np.save(OUT / "posterior_samples.npy", all_samples)
    np.save(OUT / "log_prob.npy", all_logp)

log(f"all {N_CHAINS} chains complete in {time.time()-t_grand:.1f}s")

# ---------- Post-processing ----------
samples_flat = all_samples.reshape(-1, N_DIM)
logp_flat    = all_logp.reshape(-1)
log(f"Total post-warmup samples: {samples_flat.shape[0]}")

post_sigma = samples_flat.std(axis=0)
post_mean  = samples_flat.mean(axis=0)
ident_ratio = post_sigma / PRIOR_SIGMA
id_table = []
for i, name in enumerate(PARAM_NAMES):
    id_table.append({"param": name,
                     "prior_sigma": float(PRIOR_SIGMA[i]),
                     "post_mean":   float(post_mean[i]),
                     "post_sigma":  float(post_sigma[i]),
                     "ratio":       float(ident_ratio[i]),
                     "identifiable": bool(ident_ratio[i] < 0.5)})
(OUT / "identifiability_table.md").write_text(
    "# Identifiability table\n\n"
    "| Param | prior σ | post mean | post σ | σ-ratio | identifiable (<0.5) |\n"
    "|---|---:|---:|---:|---:|:--:|\n"
    + "\n".join(f"| {r['param']} | {r['prior_sigma']:.3f} | {r['post_mean']:+.3f} | "
                 f"{r['post_sigma']:.3f} | {r['ratio']:.2f} | "
                 f"{'YES' if r['identifiable'] else 'no'} |"
                 for r in id_table) + "\n")

ess = {}
for i, name in enumerate(PARAM_NAMES):
    try:
        chains_avg = all_samples[:, :, :, i].reshape(-1, N_STEPS).mean(axis=0)
        tau = emcee.autocorr.integrated_time(chains_avg, quiet=True)
        ess[name] = float(N_STEPS / max(float(tau), 1.0))
    except Exception:
        ess[name] = float("nan")
(OUT / "ess_per_param.json").write_text(json.dumps(ess, indent=2))

# ---------- Posterior dec distribution ----------
log("Computing dec on 100 random posterior samples…")
rng2 = np.random.default_rng(42)
n_eval = min(100, samples_flat.shape[0])
idx = rng2.choice(samples_flat.shape[0], size=n_eval, replace=False)
dec_list = []
for j, k in enumerate(idx):
    _, d = _apply_and_eval(samples_flat[k], want_dec=True)
    dec_list.append({"sample_idx": int(k), "dec": d,
                      "theta": samples_flat[k].tolist(),
                      "log_prob": float(logp_flat[k])})
    if (j+1) % 20 == 0:
        finite_so_far = [s["dec"] for s in dec_list if np.isfinite(s["dec"])]
        log(f"  dec eval {j+1}/{n_eval}  median so far = "
            f"{np.median(finite_so_far) if finite_so_far else float('nan'):.3f}")

dec_arr = np.array([d["dec"] for d in dec_list], dtype=float)
finite = dec_arr[np.isfinite(dec_arr)]
summary = {
    "n_samples_eval": len(dec_list),
    "n_finite":       int(finite.size),
    "dec_median":     float(np.nanmedian(finite)) if finite.size else None,
    "dec_min":        float(np.nanmin(finite))    if finite.size else None,
    "dec_p10":        float(np.nanpercentile(finite, 10)) if finite.size else None,
    "dec_p90":        float(np.nanpercentile(finite, 90)) if finite.size else None,
    "dec_max":        float(np.nanmax(finite))    if finite.size else None,
    "samples":        dec_list,
}
(OUT / "posterior_dec_distribution.json").write_text(json.dumps(summary, indent=2))
log(f"posterior dec: median={summary['dec_median']}  min={summary['dec_min']}  "
    f"p10={summary['dec_p10']}  p90={summary['dec_p90']}")

# ---------- Best sample audit ----------
best_i = int(np.nanargmax(logp_flat))
best_theta = samples_flat[best_i]
_, best_dec = _apply_and_eval(best_theta, want_dec=True)
best_audit = {
    "best_idx":    int(best_i),
    "best_logp":   float(logp_flat[best_i]),
    "best_dec":    float(best_dec) if np.isfinite(best_dec) else None,
    "best_theta":  {n: float(best_theta[i]) for i, n in enumerate(PARAM_NAMES)},
    "theta0_logp": float(ll0),
    "theta0_dec":  float(dec0),
}
(OUT / "posterior_best_audit.json").write_text(json.dumps(best_audit, indent=2))
log(f"BEST sample: dec={best_dec}  logp={best_audit['best_logp']:.2f}")

# ---------- Plots ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(N_DIM, 1, figsize=(8, 1.6 * N_DIM), sharex=True)
for i, name in enumerate(PARAM_NAMES):
    for c in range(N_CHAINS):
        axes[i].plot(all_samples[c, :, :, i].mean(axis=0),
                     alpha=0.8, lw=0.8, label=f"chain {c}" if i == 0 else None)
    axes[i].set_ylabel(name)
axes[-1].set_xlabel("step")
axes[0].legend(fontsize=7, ncol=min(N_CHAINS, 4))
fig.tight_layout(); fig.savefig(OUT / "posterior_traces.png", dpi=100); plt.close(fig)

try:
    import corner
    fig = corner.corner(samples_flat, labels=PARAM_NAMES,
                        quantiles=(0.16, 0.5, 0.84), show_titles=True,
                        title_kwargs={"fontsize": 8}, label_kwargs={"fontsize": 8})
    fig.savefig(OUT / "corner.png", dpi=80); plt.close(fig)
    log("Saved corner.png")
except Exception as e:
    log(f"corner failed: {e}")

fig, ax = plt.subplots(figsize=(6, 4))
if finite.size:
    ax.hist(finite, bins=20, edgecolor="k")
ax.axvline(1.5, color="g", linestyle="--", label="DISCOVERY (≤1.5)")
ax.axvline(2.0, color="orange", linestyle="--", label="2.0 dec")
ax.axvline(4.0, color="r", linestyle="--", label="4.0 dec (ceiling)")
ax.set_xlabel("RMSE (dec)"); ax.set_ylabel("count")
ax.set_title("Posterior dec distribution")
ax.legend(); fig.tight_layout(); fig.savefig(OUT / "dec_histogram.png", dpi=100); plt.close(fig)

# ---------- Honest verdict ----------
n_below_1p5 = int((finite < 1.5).sum())
n_below_2p0 = int((finite < 2.0).sum())
n_below_3p0 = int((finite < 3.0).sum())
n_identifiable   = sum(1 for r in id_table if r["identifiable"])
n_unidentifiable = sum(1 for r in id_table if r["ratio"] > 0.9)
post_logp_spread = float(np.std(logp_flat))
multimodal_flag = post_logp_spread > 50.0

if summary["dec_min"] is not None and summary["dec_min"] < 1.5:
    gate = "DISCOVERY"
    headline = (f"Posterior reaches dec ≤ 1.5 (min = {summary['dec_min']:.3f}). "
                "Tuning + co-optimisation, not topology, is the bottleneck.")
elif multimodal_flag or n_unidentifiable >= 6:
    gate = "INFORMATIVE"
    headline = (f"Posterior poorly identified ({n_unidentifiable}/{N_DIM} unidentifiable, "
                f"log-prob σ={post_logp_spread:.1f}). Topology gap confirmed.")
else:
    gate = "KILLSHOT"
    headline = (f"Posterior is unimodal but stuck — min dec = {summary['dec_min']}. "
                "Data + topology both insufficient.")

verdict = [f"# z500 BSIM4 posterior MCMC — honest verdict",
           f"\n**Gate: {gate}**\n",
           headline + "\n",
           "## Run summary\n",
           f"- chains × walkers × steps: {N_CHAINS} × {N_WALKERS} × {N_STEPS} "
           f"(warm-up {N_WARMUP}); D={N_DIM}",
           f"- total post-warmup samples: {samples_flat.shape[0]}",
           f"- log_lik(θ=0) = {ll0:.2f}",
           f"- dec(θ=0) = {dec0:.3f}",
           f"- best post log_prob = {best_audit['best_logp']:.2f}",
           f"- best sample dec = {best_dec}",
           "",
           "## Posterior dec distribution",
           f"- N evaluated = {summary['n_samples_eval']}, finite = {summary['n_finite']}",
           f"- median dec = {summary['dec_median']}",
           f"- min dec = {summary['dec_min']}",
           f"- p10 / p90 = {summary['dec_p10']} / {summary['dec_p90']}",
           f"- samples below 1.5 dec: {n_below_1p5}/{summary['n_finite']}",
           f"- samples below 2.0 dec: {n_below_2p0}/{summary['n_finite']}",
           f"- samples below 3.0 dec: {n_below_3p0}/{summary['n_finite']}",
           "",
           "## Identifiability ranking (σ-ratio: post/prior, lower=more identifiable)\n",
           "| param | post σ / prior σ | identifiable |",
           "|---|---:|:--:|"]
for r in sorted(id_table, key=lambda r: r["ratio"]):
    verdict.append(f"| {r['param']} | {r['ratio']:.2f} | "
                   f"{'YES' if r['identifiable'] else 'no'} |")
verdict += ["", "## Recommendation"]
if gate == "DISCOVERY":
    verdict.append("- Re-fit using posterior median as new defaults; current Mario fit was sub-optimal "
                   "but topology is sufficient.")
elif gate == "INFORMATIVE":
    high = [r["param"] for r in id_table if r["ratio"] > 0.9]
    verdict.append(f"- Unidentifiable params {high} are degenerate under DC alone — either need "
                   "orthogonal measurements (transient/AC/back-gate) or they are compensating for a "
                   "missing parallel current path. Recommend adding the missing path explicitly.")
else:
    verdict.append("- Posterior concentrated but model+data cannot reach <2 dec. Re-scope claim or "
                   "request additional silicon measurements to break degeneracies.")
(OUT / "honest_verdict.md").write_text("\n".join(verdict))
log("Wrote honest_verdict.md")
log("=== DONE ===")
LOG.close()
