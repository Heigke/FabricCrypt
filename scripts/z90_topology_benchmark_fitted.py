#!/usr/bin/env python3
"""Topology × benchmark grid using the FULL BSIM4 port + z88-Stage-3 fitted parameters.

Each cell is the actual differentiable BSIM4-port 2T solve_2t_steady_state. We batch
N cells per timestep (broadcastable Vd/VG1/VG2 → ~120 ms for N=128) and run T steps.

Reservoir dynamics:
  - state[t]     = Id_BSIM4(Vd_eff[t], VG1=fixed, VG2=per-cell)         (quasi-static)
  - Vd_eff[t]    = baseline_Vd + DRIVE_GAIN * (W @ state[t-1]) + INPUT_GAIN * Win * u[t]
  - Win is per-cell ±1 random projection of the scalar input
  - per-cell heterogeneity: VG2 ∈ VG2_BIAS ± 0.05 V (Gaussian) — gives reservoir variety

The "memory" in this regime comes from the recurrent topology W @ state, not from body
charge persistence (which would require a transient body-KCL solve — slated for z91).

Topologies:  random, ring, small_world, scale_free, hierarchical, full
Benchmarks:  XOR (τ=2), Memory-Capacity, NARMA-10, 4-class waveform classification
N=128, T=2000, washout=500, 3 seeds.
"""
from __future__ import annotations
import json, time, sys, traceback
from pathlib import Path
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)   # keep <50% CPU so z88/z89 keep their cores

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT  = ROOT / "results/z90_topology_benchmark_fitted"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, solve_2t_steady_state
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry
from nsram.benchmarks import (
    xor_accuracy, memory_capacity, narma_prediction, waveform_classification,
)
from nsram.plasticity_net import (
    topo_random, topo_ring, topo_small_world, topo_scale_free,
    topo_full, topo_hierarchical,
)

# ---------------------------------------------------------------- params + cfg
DATA_DIR = ROOT / "data/sebas_2026_04_22"
P = json.load(open(ROOT / "results/z88_bsim4_port_fit_p7v10_skipnonconv/stage3_summary.json"))["params"]
print("z88 Stage 3 params:", json.dumps(P, indent=2))


def build_port_cfg():
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True)
    model = BSIM4Model.from_spice(str(DATA_DIR / "PTM130bulkNSRAM.txt"))
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = P["Bf"]
    sd_M1 = compute_size_dep(model, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model, Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn), T_C=cfg.T_C)
    SCALED = ["alpha0", "beta0", "agidl", "bgidl", "cgidl", "egidl"]
    ATTRS = {"vth0": "vth0_T", "u0": "u0temp", "vsat": "vsattemp"}
    for sd in [sd_M1, sd_M2]:
        for k in SCALED:
            if k in P:
                sd.scaled[k] = torch.tensor(P[k])
        for kv, attr in ATTRS.items():
            if kv in P:
                setattr(sd, attr, torch.tensor(P[kv]))
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2
    return cfg, model, bjt


CFG, MODEL, BJT = build_port_cfg()

# ---------------------------------------------------------------- reservoir setup
N        = 128
T        = 2000
WASHOUT  = 500
SEEDS    = [0, 1, 2]
VG2_BIAS = 0.20
VG2_SPREAD = 0.05    # ±50 mV per-cell — heterogeneous reservoir
VD_BASE  = 1.0       # operating drain bias
VD_DRIVE = 0.4       # ± swing range from coupling+input
VG1_FIXED = 1.0

INPUT_GAIN = 0.30
COUPLE_GAIN = 0.12   # spectral-radius-ish

TOPOS = {
    "random":       lambda N, s: topo_random(N, p=0.10, seed=s),
    "ring":         lambda N, s: topo_ring(N, k=4),
    "small_world":  lambda N, s: topo_small_world(N, k=4, p_rewire=0.1, seed=s),
    "scale_free":   lambda N, s: topo_scale_free(N, m=3, seed=s),
    "hierarchical": lambda N, s: topo_hierarchical(N, levels=3, branching=4, seed=s),
    "full":         lambda N, s: topo_full(N),
}


def gen_input(seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.uniform(-1.0, 1.0, T))


def make_VG2(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return VG2_BIAS + VG2_SPREAD * torch.randn(N, generator=g)


def make_Win(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + 1000)
    return 2.0 * torch.rand(N, generator=g) - 1.0  # ±1 input mask


def run_reservoir(W: torch.Tensor, VG2_per_cell: torch.Tensor,
                  Win: torch.Tensor, inputs: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Returns states (N, T) using BSIM4 port at every timestep."""
    states = torch.zeros(N, T)
    last = torch.zeros(N)         # last normalized Id
    Vsint_warm = None
    Vb_warm = None
    n_unconv = 0
    last_warmup_print = -1
    with torch.no_grad():
        for t in range(T):
            Vd_t = (
                VD_BASE
                + VD_DRIVE * COUPLE_GAIN * (W @ last)
                + VD_DRIVE * INPUT_GAIN * Win * inputs[t]
            ).clamp(0.05, 1.95)
            VG1_t = torch.full((N,), VG1_FIXED)
            out = solve_2t_steady_state(
                CFG, MODEL, BJT, Vd_t, VG1_t, VG2_per_cell,
                Vsint_init=Vsint_warm, Vb_init=Vb_warm,
            )
            Id = out["Id"].abs()
            # log-scale + standardize so values land in O(1)
            Id_log = torch.log10(Id.clamp_min(1e-15))
            states[:, t] = Id_log
            last = (Id_log - Id_log.mean()) / (Id_log.std() + 1e-9)
            Vsint_warm = out["Vsint"].detach()
            Vb_warm = out["Vb"].detach()
            if not bool(out["converged"].all() if hasattr(out["converged"], "all") else out["converged"]):
                n_unconv += 1
            if t % 200 == 0 and t != last_warmup_print:
                print(f"    t={t:4d}  Id_log mean={Id_log.mean():+.2f}  std={Id_log.std():.2f}", flush=True)
                last_warmup_print = t
    # standardize per-cell across time before returning (helps regressor)
    mu = states.mean(dim=1, keepdim=True)
    sd = states.std(dim=1, keepdim=True) + 1e-9
    states_z = (states - mu) / sd
    return states_z, {"n_unconv": n_unconv}


def run_one(topo_name: str, topo_fn, seed: int) -> dict:
    t0 = time.time()
    W = topo_fn(N, seed).double()
    VG2 = make_VG2(seed)
    Win = make_Win(seed)
    inputs = gen_input(seed)
    print(f"\n[{topo_name} seed={seed}] starting (W nnz={int((W != 0).sum())})", flush=True)
    states, meta = run_reservoir(W, VG2, Win, inputs)
    np_states = states.numpy()
    np_inputs = inputs.numpy()
    row = {"topology": topo_name, "seed": seed, "unconv": meta["n_unconv"]}
    try:
        row["XOR"] = float(xor_accuracy(np_states, np_inputs, washout=WASHOUT, tau=2))
    except Exception as e:
        row["XOR"] = float("nan"); row["XOR_err"] = str(e)[:80]
    try:
        row["MC"] = float(memory_capacity(np_states, np_inputs, washout=WASHOUT, max_delay=15))
    except Exception as e:
        row["MC"] = float("nan"); row["MC_err"] = str(e)[:80]
    try:
        row["NARMA"] = float(narma_prediction(np_states, np_inputs, washout=WASHOUT, order=10))
    except Exception as e:
        row["NARMA"] = float("nan"); row["NARMA_err"] = str(e)[:80]
    try:
        row["Waveform"] = float(waveform_classification(np_states, np_inputs, washout=WASHOUT, n_classes=4))
    except Exception as e:
        row["Waveform"] = float("nan"); row["Waveform_err"] = str(e)[:80]
    row["elapsed"] = round(time.time() - t0, 1)
    print(
        f"[{topo_name} seed={seed}] done in {row['elapsed']:.1f}s  unconv={meta['n_unconv']}  "
        f"XOR={row['XOR']:.3f}  MC={row['MC']:.3f}  NARMA={row['NARMA']:.3f}  Wave={row['Waveform']:.3f}",
        flush=True,
    )
    return row


def main():
    rows = []
    t_global = time.time()
    for topo_name, topo_fn in TOPOS.items():
        for seed in SEEDS:
            try:
                rows.append(run_one(topo_name, topo_fn, seed))
            except Exception as e:
                print(f"[{topo_name} seed={seed}] FAILED: {e}\n{traceback.format_exc()}", flush=True)
                rows.append({"topology": topo_name, "seed": seed, "error": str(e)[:200]})
            # write incremental summary so we can inspect mid-run
            (OUT / "summary.json").write_text(json.dumps({
                "params_used": P,
                "config": {"N": N, "T": T, "washout": WASHOUT, "VG2_bias": VG2_BIAS,
                           "VG2_spread": VG2_SPREAD, "VD_base": VD_BASE,
                           "INPUT_GAIN": INPUT_GAIN, "COUPLE_GAIN": COUPLE_GAIN,
                           "z88_stage3_loss": 0.928},
                "rows": rows,
                "wall_seconds": round(time.time() - t_global, 1),
            }, indent=2))
    # aggregate
    import collections
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if "error" in r: continue
        for b in ["XOR", "MC", "NARMA", "Waveform"]:
            v = r.get(b)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                agg[r["topology"]][b].append(v)
    lines = ["", "=" * 80,
             f"Topology × benchmark — mean ± std across {len(SEEDS)} seeds (z88-Stage-3 BSIM4 port, N={N}, T={T})",
             "=" * 80,
             f"{'topology':12s}  {'XOR':>14s}  {'MC':>14s}  {'NARMA':>14s}  {'Waveform':>14s}"]
    for topo in TOPOS:
        d = agg[topo]
        cells_str = []
        for b in ["XOR", "MC", "NARMA", "Waveform"]:
            xs = d.get(b, [])
            cells_str.append(f"{np.mean(xs):.3f}±{np.std(xs):.3f}" if xs else "    n/a    ")
        lines.append(f"{topo:12s}  " + "  ".join(f"{c:>14s}" for c in cells_str))
    table = "\n".join(lines)
    print(table)
    (OUT / "table.txt").write_text(table)


if __name__ == "__main__":
    main()
