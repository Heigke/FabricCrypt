"""z139 — Large-scale topology + scaling experiment (post-M3a.1 model fix).

Builds on z119. Three changes:
  1. **Bf = 2×10⁴** (M3a.1 finding — beats z119's 5×10⁴ on aggregate fit)
  2. Add **HUB_SPOKE** and **LAYERED** topologies — novel, motivated
     by NS-RAM's natural hub-spoke wiring (one shared body well, many cells).
  3. Push N to **{100, 300, 800}**, with intermediate JSON dumps so
     partial completion is useful if the run is killed.

Budget estimate: N=100 ~25 s/sim, N=300 ~60 s/sim, N=800 ~280 s/sim.
6 topos × 3 N × 3 seeds = 54 sims, expected wall ≈ 3 hours.
"""
from __future__ import annotations
import json, time, os
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z139_largescale_topology"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
sp2 = importlib.util.spec_from_file_location("z119", ROOT / "scripts/z119_topology_sweep.py")
z119 = importlib.util.module_from_spec(sp2); sp2.loader.exec_module(z119)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched

BF_OPT = 2.0e4   # M3a.1 finding — see research_plan/binning_audit/probe_v2_finding.md


def build_W(topology: str, N: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Wraps z119.build_W and adds two new topologies."""
    if topology == "HUB_SPOKE":
        # 1 hub connected to all, all connected back to hub; sparse leaves
        W = np.zeros((N, N))
        hub = 0
        # Strong hub→leaves and leaves→hub
        leaves = np.arange(1, N)
        W[hub, leaves] = rng.normal(0, 1.0, size=N-1)
        W[leaves, hub] = rng.normal(0, 1.0, size=N-1)
        # Sparse leaf↔leaf p=0.02
        mask = rng.random((N-1, N-1)) < 0.02
        np.fill_diagonal(mask, False)
        W[1:, 1:] = np.where(mask, rng.normal(0, 0.5, size=(N-1, N-1)), 0.0)
    elif topology == "LAYERED":
        # 2-layer feedforward+small skip: split N into 2 layers of N/2 each.
        # L1→L2 dense, L2→L1 sparse skip. Mimics deep reservoir.
        N1 = N // 2; N2 = N - N1
        W = np.zeros((N, N))
        # L1 → L2 dense
        W[N1:, :N1] = rng.normal(0, 1.0/np.sqrt(N1), size=(N2, N1))
        # L2 → L1 sparse skip (10%)
        skip_mask = rng.random((N1, N2)) < 0.10
        W[:N1, N1:] = np.where(skip_mask, rng.normal(0, 1.0/np.sqrt(N2), size=(N1, N2)), 0.0)
        # Within-layer recurrence sparse
        for layer_slice, n_layer in [((0, N1), N1), ((N1, N), N2)]:
            i0, i1 = layer_slice
            mask = rng.random((n_layer, n_layer)) < 0.10
            np.fill_diagonal(mask, False)
            W[i0:i1, i0:i1] = np.where(mask, rng.normal(0, 1.0/np.sqrt(n_layer), size=(n_layer, n_layer)), 0.0)
    else:
        return z119.build_W(topology, N, rho, rng)
    eig = np.linalg.eigvals(W)
    rho_W = float(np.max(np.abs(eig)))
    return W * (rho / max(rho_W, 1e-9))


def run_cell_sim(topology, N, T, kappa, drive_fn, seed, Bf=BF_OPT):
    rng = np.random.default_rng(seed)
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=50)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = Bf
    base_VG1 = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    base_VG2 = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    W_rec_np = build_W(topology, N, rho=0.9, rng=rng)
    W_rec = torch.tensor(W_rec_np, dtype=torch.float64)
    feat_prev = torch.zeros(N, dtype=torch.float64)
    log_Id = np.zeros((N, T))
    for t in range(T):
        Vd_t = torch.tensor([float(drive_fn(t))], dtype=torch.float64)
        recur = (W_rec @ feat_prev) if kappa > 0 else torch.zeros(N, dtype=torch.float64)
        VG2_eff = (base_VG2 + kappa * recur).clamp(-0.2, 1.0)
        out = forward_2t_batched(cfg, M1, M2, bjt, Vd_t, base_VG1, VG2_eff,
                                  max_iters=15, tol=1e-9, verbose=False)
        log_Id[:, t] = np.log10(np.maximum(out["Id"].abs().squeeze().numpy(), 1e-15))
        feat_prev = torch.tensor(log_Id[:, t], dtype=torch.float64)
    return log_Id


def main():
    t0 = time.time()
    topologies = ["RAND_GAUSS", "MESH_4N", "ER_SPARSE", "WS_SMALLWORLD",
                  "HUB_SPOKE", "LAYERED"]
    Ns = [int(x) for x in os.environ.get("Z139_NS", "100,300,800").split(",")]
    seeds = [42, 43, 44]
    T = int(os.environ.get("Z139_T", "500"))
    kappa = 0.03
    print(f"[z139] Large-scale topology sweep")
    print(f"  topologies: {topologies}")
    print(f"  Ns: {Ns}, seeds: {seeds}, T={T}, κ={kappa}, Bf={BF_OPT:.1e}")
    print()

    results = {}
    n_done = 0
    n_total = len(topologies) * len(Ns) * len(seeds)
    for topo in topologies:
        for N in Ns:
            for seed in seeds:
                ti = time.time()
                rng = np.random.default_rng(seed)
                u_bin_int = rng.integers(0, 2, size=T)
                u_bin = 2.0 * u_bin_int - 1.0
                drive = lambda t: 1.0 + 0.5 * float(u_bin[t])
                try:
                    log_Id1 = run_cell_sim(topo, N, T, kappa, drive, seed)
                    MC, NARMA_NRMSE = z119.eval_MC_NARMA(log_Id1, u_bin, T)
                    XOR_acc = z119.eval_XOR(log_Id1, u_bin_int, tau=2, T=T)
                    u_wave, cls = z119.waveform_inputs(T, n_classes=4,
                                                       rng=np.random.default_rng(seed+1000))
                    drive_w = lambda t: 1.0 + 0.5 * float(u_wave[t])
                    log_Id2 = run_cell_sim(topo, N, T, kappa, drive_w, seed)
                    WAVE_acc = z119.eval_waveform(log_Id2, cls, T, n_classes=4)
                except Exception as e:
                    MC = float("nan"); NARMA_NRMSE = float("nan")
                    XOR_acc = float("nan"); WAVE_acc = float("nan")
                    print(f"  ERROR {topo}/N{N}/s{seed}: {e}", flush=True)

                key = f"{topo}_N{N}_s{seed}"
                results[key] = {"topo": topo, "N": N, "seed": seed,
                                 "MC": MC, "NARMA_NRMSE": NARMA_NRMSE,
                                 "XOR_acc": XOR_acc, "WAVE_acc": WAVE_acc,
                                 "wall_s": float(time.time() - ti)}
                n_done += 1
                print(f"  [{n_done:3d}/{n_total}] {topo:14s} N={N:>4d} s={seed}  "
                      f"MC={MC:5.2f} NARMA={NARMA_NRMSE:5.2f} "
                      f"XOR={XOR_acc:.2f} WAVE={WAVE_acc:.2f}  "
                      f"({time.time()-ti:.0f}s, total {time.time()-t0:.0f}s)",
                      flush=True)
                # Intermediate dump
                json.dump({"results": results, "Bf": BF_OPT,
                            "topologies": topologies, "Ns": Ns, "seeds": seeds,
                            "T": T, "kappa": kappa,
                            "n_done": n_done, "n_total": n_total},
                           (OUT / "summary_partial.json").open("w"), indent=2)

    # Aggregate
    print(f"\n[z139] === Aggregated (mean over seeds, Bf={BF_OPT:.1e}) ===")
    print(f"  {'topo':14s} {'N':>4s}  {'MC':>6s} {'NARMA':>6s} {'XOR':>5s} {'WAVE':>5s}")
    agg = {}
    for topo in topologies:
        for N in Ns:
            keys = [f"{topo}_N{N}_s{s}" for s in seeds]
            valid = [results[k] for k in keys if k in results]
            if not valid:
                continue
            mc = float(np.nanmean([r["MC"] for r in valid]))
            na = float(np.nanmean([r["NARMA_NRMSE"] for r in valid]))
            xo = float(np.nanmean([r["XOR_acc"] for r in valid]))
            wa = float(np.nanmean([r["WAVE_acc"] for r in valid]))
            agg[f"{topo}_N{N}"] = {"MC": mc, "NARMA_NRMSE": na,
                                    "XOR_acc": xo, "WAVE_acc": wa}
            print(f"  {topo:14s} {N:>4d}  {mc:6.2f} {na:6.2f} {xo:5.2f} {wa:5.2f}")

    json.dump({"results": results, "agg": agg, "Bf": BF_OPT,
                "topologies": topologies, "Ns": Ns, "seeds": seeds,
                "T": T, "kappa": kappa},
               (OUT / "summary.json").open("w"), indent=2)
    (OUT / "summary_partial.json").unlink(missing_ok=True)
    print(f"\n[z139] saved {OUT}/summary.json")
    print(f"[z139] wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
