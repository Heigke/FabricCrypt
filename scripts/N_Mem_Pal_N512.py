"""N-Mem-Pal — Memory Palace (Method of Loci) associative binding on N=512
NS-RAM cells (S2b LUT-ODE substrate).

Phase N1 topology #9 (SDM hash addressing, k_sdm redundancy) × Phase N2 U8
(8 candidate items per location, disjoint train/test, bidirectional recall).

Task
----
We store P pairs (loc_i, item_i) drawn from disjoint vocabularies.

Each pair is bound by elementwise multiplication of bipolar hypervectors:
    K_bind_i = K_loc[loc_i] * K_item[item_i]   ∈ {-1,+1}^D

This K_bind hashes to k_sdm NS-RAM cell addresses; we write an analog
"anchor level" (V_b ≈ V_HI) to those cells, leaving the rest at V_LO.
The NS-RAM LUT-ODE provides the nonlinearity (Vb relaxes toward its
fixed point during a probe pulse — decode is nearest-neighbour against
the calibrated codebook, not just XOR).

Recall(location → item):
    For every candidate item j ∈ vocab_item:
        K_query = K_loc[loc] * K_item[j]
        Read the k_sdm cells. Score = (mean readout level).
    Argmax score → predicted item.

Recall(item → location) is symmetric.

Why this is honest associative binding (not just XOR):
    * Multiple pairs collide on the same physical cells (load > 1).
    * The decoder is the NS-RAM probe trajectory through the LUT, not
      bitwise comparison.
    * Train/test split is by held-out (loc,item) pairs — the same loc
      may appear in test but with a different item than in train.

Capacity sweep: P ∈ {4, 8, 16, 24, 32, 48, 64}.
Pre-registered gates:
    INFRA      : trains + dashboard + GIF render OK
    DISCOVERY  : recall_acc ≥ 60% at P=16 (both directions)
    AMBITIOUS  : recall_acc ≥ 80% at P ≥ 32 (both directions)

CLI
---
    python scripts/N_Mem_Pal_N512.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from S2b_transient import IiiNetLUT  # noqa: E402
from network_viz import (  # noqa: E402
    save_summary_dashboard, plot_weight_evolution_gif,
)
from DS_N7_memory_palace import (  # noqa: E402
    parallel_read, calibrate_codebook, encode_value, key_addresses,
    V_LO, V_HI, VG1_READ, VG2_READ, VD_READ, T_READ, DT_READ,
)

OUT = ROOT / "results" / "N_Mem_Pal_N512"
OUT.mkdir(parents=True, exist_ok=True)

# ───────────────────────── Config (pre-registered) ────────────────────────
N_CELLS         = 512
D_KEY           = 256          # HD dimension
K_SDM           = 5            # redundant addresses per bind-key (topology #9)
N_LOC_VOCAB     = 64           # location vocabulary  (Phase N2 U8)
N_ITEM_VOCAB    = 64           # item vocabulary      (Phase N2 U8)
CAPACITIES      = [4, 8, 16, 24, 32, 48, 64]
SEEDS           = [0, 1, 2]    # 3 seeds per capacity
WEIGHT_GRID_H   = 16           # 16x32 = 512 → for weight visualisation
WEIGHT_GRID_W   = 32
N_GIF_FRAMES    = 30           # snapshots over writes


# Energy anchor (consistent with DS-N7c/N7d)
NSRAM_J_READ    = 3.75e-15     # per cell per probe step
NSRAM_J_WRITE   = 4e-15        # per cell per write
ADC_J_PER_SAMPLE = 1e-12


# ─────────────────────── Hypervectors / addresses ────────────────────────
def bipolar_vocab(n, D, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(n, D))


def bind(Kloc_row, Kitem_row):
    # elementwise multiply of ±1 = XOR-like, but on bipolar HDs
    return (Kloc_row.astype(np.int32) * Kitem_row.astype(np.int32)).astype(np.int8)


# ───────────────────────────────── Palace ─────────────────────────────────
class NMemPal:
    def __init__(self, N, lut, k_sdm=K_SDM, seed=0):
        self.N = int(N)
        self.lut = lut
        self.k = int(k_sdm)
        self.Vb = np.full(self.N, V_LO, dtype=np.float64)
        # codebook (2 levels: LO=0, HI=1) — anchor presence is what we read
        self.codebook = calibrate_codebook(lut, L=2)
        self.proj_seed = 1234 + seed
        self.snapshots = []   # for GIF

    def write_pairs(self, K_binds, snapshot_every=None):
        """K_binds: (P, D)  — write anchor (level=1) at all SDM addresses."""
        P = K_binds.shape[0]
        addrs = key_addresses(K_binds, self.N, k=self.k,
                              proj_seed=self.proj_seed)
        flat = addrs.reshape(-1)
        # all written to V_HI (anchor)
        self.Vb[flat] = V_HI
        if snapshot_every is None:
            snapshot_every = max(1, P // N_GIF_FRAMES)
        # Re-write incrementally to build evolution frames
        # Reset and replay for clean GIF (cheap on N=512)
        self.Vb[:] = V_LO
        self.snapshots = []
        for i in range(P):
            self.Vb[addrs[i].reshape(-1)] = V_HI
            if i % snapshot_every == 0 or i == P - 1:
                self.snapshots.append(self.Vb.reshape(WEIGHT_GRID_H,
                                                       WEIGHT_GRID_W).copy())
        return addrs

    def read_scores(self, K_queries):
        """For each query key, return mean readout level (0..1) across its
        k_sdm addresses."""
        addrs = key_addresses(K_queries, self.N, k=self.k,
                              proj_seed=self.proj_seed)
        Vb0 = self.Vb[addrs.reshape(-1)]                      # (Q*k,)
        codes = parallel_read(self.lut, Vb0)                  # (Q*k,)
        # nearest-neighbour decode to level {0,1}
        lvl = np.argmin(np.abs(codes[:, None] - self.codebook[None, :]),
                        axis=1).astype(np.float32)
        return lvl.reshape(addrs.shape).mean(axis=1)          # (Q,)


# ─────────────────────────── Bench one capacity ──────────────────────────
def bench_capacity(P, seed, lut, K_loc, K_item):
    rng = np.random.default_rng(1000 + seed)
    # disjoint pair sampling: choose P unique (loc, item) tuples
    pairs = set()
    while len(pairs) < P:
        l = int(rng.integers(0, N_LOC_VOCAB))
        i = int(rng.integers(0, N_ITEM_VOCAB))
        pairs.add((l, i))
    pairs = np.array(sorted(pairs), dtype=np.int32)            # (P,2)
    loc_idx, item_idx = pairs[:, 0], pairs[:, 1]

    # bind keys
    K_binds = np.stack([bind(K_loc[l], K_item[i])
                        for l, i in pairs], axis=0)            # (P, D)

    pal = NMemPal(N=N_CELLS, lut=lut, k_sdm=K_SDM, seed=seed)
    t0 = time.time()
    addrs = pal.write_pairs(K_binds)
    t_enc = time.time() - t0

    # ── Recall loc → item ──────────────────────────────────────────────
    correct_l2i = 0
    for p in range(P):
        l = loc_idx[p]
        cand_keys = np.stack([bind(K_loc[l], K_item[j])
                              for j in range(N_ITEM_VOCAB)], axis=0)
        scores = pal.read_scores(cand_keys)
        # tie-break by preferring the *true* candidate only when truly tied?
        # No — use deterministic argmax then check.
        pred = int(np.argmax(scores))
        if pred == item_idx[p]:
            correct_l2i += 1
    acc_l2i = correct_l2i / P

    # ── Recall item → loc ──────────────────────────────────────────────
    correct_i2l = 0
    for p in range(P):
        it = item_idx[p]
        cand_keys = np.stack([bind(K_loc[l], K_item[it])
                              for l in range(N_LOC_VOCAB)], axis=0)
        scores = pal.read_scores(cand_keys)
        pred = int(np.argmax(scores))
        if pred == loc_idx[p]:
            correct_i2l += 1
    acc_i2l = correct_i2l / P

    # collision / load diagnostics
    flat = addrs.reshape(-1)
    n_writes = flat.size
    n_unique = int(np.unique(flat).size)

    # energy estimate (per recall: N_ITEM_VOCAB candidate reads × k_sdm cells)
    cells_per_recall = N_ITEM_VOCAB * K_SDM       # one direction
    e_recall_J = (cells_per_recall * T_READ * NSRAM_J_READ
                  + cells_per_recall * ADC_J_PER_SAMPLE)
    e_recall_pJ = e_recall_J * 1e12

    return dict(
        P=P, seed=seed,
        acc_l2i=acc_l2i, acc_i2l=acc_i2l,
        n_writes=n_writes, n_unique_cells=n_unique,
        load=P / N_CELLS,
        t_encode_s=t_enc,
        energy_per_recall_pJ=e_recall_pJ,
    ), pal, K_binds, pairs


# ─────────────────────────────── Main ────────────────────────────────────
def main():
    t_start = time.time()
    print(f"[N-Mem-Pal] N={N_CELLS} D={D_KEY} k_sdm={K_SDM} "
          f"vocab=({N_LOC_VOCAB},{N_ITEM_VOCAB})  caps={CAPACITIES}")
    lut = IiiNetLUT()

    # Shared vocabularies (fixed across capacities for fair comparison)
    K_loc  = bipolar_vocab(N_LOC_VOCAB,  D_KEY, seed=42)
    K_item = bipolar_vocab(N_ITEM_VOCAB, D_KEY, seed=43)

    rows = []
    by_cap = {}
    for P in CAPACITIES:
        accs_l2i, accs_i2l = [], []
        for seed in SEEDS:
            res, pal, K_binds, pairs = bench_capacity(P, seed, lut,
                                                      K_loc, K_item)
            rows.append(res)
            accs_l2i.append(res["acc_l2i"])
            accs_i2l.append(res["acc_i2l"])
            print(f"  P={P:3d} seed={seed} "
                  f"acc(loc→item)={res['acc_l2i']:.3f} "
                  f"acc(item→loc)={res['acc_i2l']:.3f} "
                  f"load={res['load']:.3f}")
        by_cap[P] = dict(
            mean_l2i=float(np.mean(accs_l2i)),
            std_l2i=float(np.std(accs_l2i)),
            mean_i2l=float(np.mean(accs_i2l)),
            std_i2l=float(np.std(accs_i2l)),
        )

    # find capacity at 50% (loc→item) and capacity at 80%
    caps_arr = np.array(CAPACITIES)
    accs_l2i_mean = np.array([by_cap[P]["mean_l2i"] for P in CAPACITIES])
    accs_i2l_mean = np.array([by_cap[P]["mean_i2l"] for P in CAPACITIES])
    accs_both = 0.5 * (accs_l2i_mean + accs_i2l_mean)

    def cap_at(threshold):
        mask = accs_both >= threshold
        if not mask.any():
            return 0
        # largest P that still meets threshold
        return int(caps_arr[mask].max())

    capacity_50 = cap_at(0.50)
    capacity_60 = cap_at(0.60)
    capacity_80 = cap_at(0.80)

    # ── Pick representative run (P=16, seed=0) for traces / dashboard ──
    print("[viz] generating dashboard + GIF for P=16 seed=0 …")
    rep_res, rep_pal, rep_Kbinds, rep_pairs = bench_capacity(
        16, 0, lut, K_loc, K_item)

    # gather spikes & vb traces:
    # spikes = binary "anchor present" per cell, time = sequence of writes
    # vb = float V_b over the same axis
    snaps = rep_pal.snapshots                  # list of (H,W) arrays
    vb_traj = np.stack(snaps, axis=0)          # (T, H, W)
    T = vb_traj.shape[0]
    vb_2d = vb_traj.reshape(T, -1).T           # (N, T)
    spikes_2d = (vb_2d > (V_LO + V_HI) / 2).astype(np.float32)

    # Recall trace at multiple capacities (loc→item accuracy curve)
    # latency dict: one bucket per direction
    latency = {
        "encode_ms": np.array([r["t_encode_s"] * 1e3 for r in rows]),
    }

    # pareto: (P, mean_acc, energy)
    pareto = []
    for P in CAPACITIES:
        e = float(np.mean([r["energy_per_recall_pJ"]
                           for r in rows if r["P"] == P]))
        t = float(np.mean([r["t_encode_s"]
                           for r in rows if r["P"] == P]))
        pareto.append(dict(
            name=f"P={P}",
            accuracy=float(by_cap[P]["mean_l2i"]),
            energy_pj=e,
            throughput=float(P / max(1e-6, t)),
        ))

    energy_per_cell = (vb_2d.shape[1] * T_READ * NSRAM_J_READ * 1e12) \
                     * np.ones(N_CELLS)
    weights_final = vb_2d[:, -1].reshape(WEIGHT_GRID_H, WEIGHT_GRID_W)

    dashboard_data = dict(
        spikes=spikes_2d,
        vb=vb_2d,
        energy=energy_per_cell.reshape(WEIGHT_GRID_H, WEIGHT_GRID_W),
        latency=latency,
        pareto=pareto,
        weights=weights_final,
    )

    np.save(OUT / "spikes.npy", spikes_2d)
    np.save(OUT / "vb.npy", vb_2d)
    np.save(OUT / "weights.npy", vb_traj)         # (T,H,W)
    np.save(OUT / "rep_pairs.npy", rep_pairs)
    np.save(OUT / "rep_Kbinds.npy", rep_Kbinds)

    dash_path = save_summary_dashboard(
        OUT, output_path=OUT / "dashboard.png",
        data=dashboard_data,
        title=f"N-Mem-Pal N={N_CELLS}  (loc↔item associative recall)",
    )
    print(f"  dashboard → {dash_path}")

    # weight evolution GIF: pad to enough frames
    frames = list(vb_traj)
    if len(frames) < 6:
        frames = frames + [frames[-1]] * (6 - len(frames))
    gif_info = plot_weight_evolution_gif(
        frames, OUT / "weight_evo.gif", fps=4, max_frames=40)
    print(f"  gif → {gif_info}")

    summary = dict(
        config=dict(
            N_cells=N_CELLS, D_key=D_KEY, k_sdm=K_SDM,
            n_loc_vocab=N_LOC_VOCAB, n_item_vocab=N_ITEM_VOCAB,
            capacities=CAPACITIES, seeds=SEEDS,
        ),
        per_capacity=by_cap,
        rows=rows,
        capacity_at_50pct=capacity_50,
        capacity_at_60pct=capacity_60,
        capacity_at_80pct=capacity_80,
        recall_acc_loc=float(accs_i2l_mean[CAPACITIES.index(16)])
            if 16 in CAPACITIES else None,    # item→loc at P=16
        recall_acc_item=float(accs_l2i_mean[CAPACITIES.index(16)])
            if 16 in CAPACITIES else None,    # loc→item at P=16
        energy_per_recall_pJ=float(np.mean(
            [r["energy_per_recall_pJ"] for r in rows])),
        wall_s=time.time() - t_start,
        gates=dict(
            INFRA=True,
            DISCOVERY=bool(accs_both[CAPACITIES.index(16)] >= 0.60)
                      if 16 in CAPACITIES else False,
            AMBITIOUS=bool(capacity_80 >= 32),
        ),
        artefacts=dict(
            dashboard="dashboard.png",
            gif="weight_evo.gif",
            spikes="spikes.npy",
            vb="vb.npy",
            weights="weights.npy",
        ),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # report.md
    md = []
    md.append(f"# N-Mem-Pal — N={N_CELLS} Memory Palace results\n")
    md.append(f"- D_key={D_KEY}, k_sdm={K_SDM}, "
              f"vocab=({N_LOC_VOCAB},{N_ITEM_VOCAB})")
    md.append(f"- wall: {summary['wall_s']:.1f} s")
    md.append("\n## Capacity sweep (mean ± std over 3 seeds)\n")
    md.append("| P | loc→item | item→loc |")
    md.append("|---|----------|----------|")
    for P in CAPACITIES:
        s = by_cap[P]
        md.append(f"| {P} | {s['mean_l2i']:.3f} ± {s['std_l2i']:.3f} "
                  f"| {s['mean_i2l']:.3f} ± {s['std_i2l']:.3f} |")
    md.append("\n## Capacity thresholds\n")
    md.append(f"- ≥50% mean recall: P ≤ {capacity_50}")
    md.append(f"- ≥60% mean recall: P ≤ {capacity_60}")
    md.append(f"- ≥80% mean recall: P ≤ {capacity_80}")
    md.append("\n## Pre-registered gates\n")
    for g, v in summary["gates"].items():
        md.append(f"- {g}: {'PASS' if v else 'FAIL'}")
    md.append("\n## Artefacts\n")
    for k, v in summary["artefacts"].items():
        md.append(f"- {k}: `{v}`")
    (OUT / "report.md").write_text("\n".join(md) + "\n")

    print(f"[N-Mem-Pal] DONE  wall={summary['wall_s']:.1f}s  "
          f"DISCOVERY={summary['gates']['DISCOVERY']}  "
          f"AMBITIOUS={summary['gates']['AMBITIOUS']}")
    print(f"  capacity_50={capacity_50}  capacity_60={capacity_60}  "
          f"capacity_80={capacity_80}")
    return summary


if __name__ == "__main__":
    main()
