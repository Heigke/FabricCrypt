"""z297b DS-N1b: KWS with MFCC MAGNITUDE -> NS-RAM spike-rate encoding.

Previous z297 used delta-modulated MFCC and got 10.6% (~chance for 12 classes).
Delta-mod was too aggressive: ~99% sparsity left almost no signal.

This version uses raw MFCC magnitude:
  - Per-cell pattern: cell_i is tuned to MFCC channel (i mod N_MFCC).
  - Per frame, MFCC magnitude (normalized [0,1]) modulates V_d (drain) of NS-RAM
    cell: V_d = V_LO + (V_HI - V_LO) * MFCC_norm.  (V_HI=2.0V, V_LO=0.5V like DS-N5f.)
  - Body state Vb integrates intrinsic current I_ii - I_leak per substep.
  - Spike events = upward crossings of Vb threshold; Vb resets after spike.
  - Readout features = per-cell spike counts pooled into N_CHUNKS contiguous time bins.

Reuses z287_ds_n1_kws helpers for dataset/MFCC/ridge.

Gates:
  PASS (relaxed):     >= 25%  (3x chance)
  AMBITIOUS:          >= 50%
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import importlib.util
spec = importlib.util.spec_from_file_location("z287", ROOT / "scripts/z287_ds_n1_kws.py")
z287 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(z287)

import torch

# ---- config ----
SEEDS = [0, 1, 2]
N_PER_CLASS_TRAIN = 100   # 12*100 = 1200 train (we'll subsample to 1000 effective)
N_PER_CLASS_TEST  = 20    # 12*20  = 240 test  (we'll subsample to 200)
N_TRAIN_CAP = 1000
N_TEST_CAP  = 200

NSRAM_N = 128             # 3.2x oversample of 40 MFCC channels
N_CHUNKS = 5              # time bins for pooling
SUB_STEPS = 8             # NS-RAM substeps per MFCC frame (more integration time)

# Drain voltage range (DS-N5f style)
VD_LO = 0.5
VD_HI = 2.0

# Per-cell VG1 bias spread (gives heterogeneous regime placement)
VG1_LO = 0.30
VG1_HI = 0.65
# Per-cell VG2 bias spread (controls leak / I_ii balance)
VG2_LO = 0.15
VG2_HI = 0.45

# log10|I_d| spike threshold (cell fires when |Id| crosses upward through this
# threshold). At Vb~0.5, log|Id|~-6.5; at Vb~0.7, log|Id|~-3.  So a threshold
# in the -5..-4 range catches the steep transition (cell "fires").
LOGID_SPIKE_THRESH = -5.0

# Vb reset after a spike (mimics resetting body charge)
VB_RESET   = 0.05

C_B_F  = 80.0 * 1e-15
DT_S   = 1e-7

ENERGY_PER_SPIKE_FJ = 6.4


def nsram_magnitude_features(X, surr, VG1_bias, VG2_bias, generator=None):
    """X: (B, T, N_MFCC) MFCC values normalized to [0,1].

    Maps cell_i -> MFCC channel (i mod N_MFCC).  Vd is modulated by MFCC magnitude.
    Returns:
      feats: (B, NSRAM_N * (N_CHUNKS + 1))  per-cell spike counts in N_CHUNKS time bins
             plus total count
      spike_total: (B,) total spikes per sample (for energy)
    """
    device = X.device
    B, T, D = X.shape
    N = NSRAM_N

    Vb_min = surr["ax_Vb"][0];  Vb_max = surr["ax_Vb"][-1]
    VG1_min = surr["ax_VG1"][0]; VG1_max = surr["ax_VG1"][-1]
    VG2_min = surr["ax_VG2"][0]; VG2_max = surr["ax_VG2"][-1]
    Vd_min  = surr["ax_Vd"][0];  Vd_max  = surr["ax_Vd"][-1]

    # cell -> mfcc-channel map
    chan_idx = torch.arange(N, device=device) % D   # (N,)

    Vb  = torch.full((B, N), VB_RESET, device=device)
    VG1 = VG1_bias.unsqueeze(0).expand(B, N).clamp(VG1_min, VG1_max)
    VG2 = VG2_bias.unsqueeze(0).expand(B, N).clamp(VG2_min, VG2_max)

    chunk_len = max(T // N_CHUNKS, 1)
    chunk_spikes = [torch.zeros(B, N, device=device) for _ in range(N_CHUNKS)]
    spike_total = torch.zeros(B, device=device)
    # also accumulate analog observables (mean log|Id|, mean Vb) per chunk
    chunk_logid = [torch.zeros(B, N, device=device) for _ in range(N_CHUNKS)]
    chunk_vb    = [torch.zeros(B, N, device=device) for _ in range(N_CHUNKS)]
    chunk_count = [0] * N_CHUNKS

    prev_logid = torch.full((B, N), LOGID_SPIKE_THRESH - 1.0, device=device)
    LOG_FLOOR = -18.0

    for fi in range(T):
        ck = min(fi // chunk_len, N_CHUNKS - 1)
        # mfcc magnitude for each cell's assigned channel
        m = X[:, fi, :].index_select(1, chan_idx)        # (B, N) in [0,1]
        Vd = VD_LO + (VD_HI - VD_LO) * m
        Vd = Vd.clamp(Vd_min, Vd_max)

        for s in range(SUB_STEPS):
            Vb_c = Vb.clamp(Vb_min, Vb_max)
            I_d, I_ii, I_leak = z287.query_surrogate(surr, VG1, VG2, Vd, Vb_c)
            # update body state
            Vb = Vb + DT_S * (I_ii - I_leak) / C_B_F
            Vb = Vb.clamp(Vb_min, Vb_max)
            # log10|Id| -> "drain current" observable
            logid = torch.log10(I_d.abs() + 10 ** LOG_FLOOR)
            # spike: upward crossing of LOGID_SPIKE_THRESH
            fired = ((logid >= LOGID_SPIKE_THRESH) &
                     (prev_logid < LOGID_SPIKE_THRESH)).float()
            # reset body state where fired
            Vb = torch.where(fired > 0, torch.full_like(Vb, VB_RESET), Vb)
            chunk_spikes[ck] = chunk_spikes[ck] + fired
            spike_total = spike_total + fired.sum(dim=1)
            chunk_logid[ck] = chunk_logid[ck] + logid
            chunk_vb[ck]    = chunk_vb[ck]    + Vb
            chunk_count[ck] += 1
            prev_logid = logid

    parts = list(chunk_spikes)
    total_per_cell = sum(chunk_spikes)
    parts.append(total_per_cell)
    # add analog observables (chunk-averaged log|Id| and Vb)
    for k in range(N_CHUNKS):
        c = max(chunk_count[k], 1)
        parts.append(chunk_logid[k] / c)
        parts.append(chunk_vb[k]    / c)
    feats = torch.cat(parts, dim=1)
    # log1p compress (only the count parts already use raw; log1p is safe-ish for negatives via shift)
    # apply log1p only to the spike-count block; pass-through for analog
    n_count_block = N * (N_CHUNKS + 1)
    feats_cnt = torch.log1p(feats[:, :n_count_block].clamp(min=0))
    feats = torch.cat([feats_cnt, feats[:, n_count_block:]], dim=1)
    mu = feats.mean(dim=0, keepdim=True); sd = feats.std(dim=0, keepdim=True) + 1e-9
    feats = (feats - mu) / sd
    return feats, spike_total, total_per_cell


def confusion(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def subsample(items, y, cap, rng):
    if len(items) <= cap:
        return items, y
    idx = rng.choice(len(items), size=cap, replace=False)
    return [items[i] for i in idx], y[idx]


def run_seed(seed, splits, bg_files, surr, device):
    rng = np.random.default_rng(seed)
    tr_items, ytr = z287.build_split(splits, bg_files, "train", N_PER_CLASS_TRAIN, rng)
    te_items, yte = z287.build_split(splits, bg_files, "test", N_PER_CLASS_TEST, rng)
    tr_items, ytr = subsample(tr_items, ytr, N_TRAIN_CAP, rng)
    te_items, yte = subsample(te_items, yte, N_TEST_CAP, rng)
    print(f"  [seed {seed}] train={len(tr_items)} test={len(te_items)}", flush=True)

    Xtr = z287.items_to_mfcc(tr_items, desc=f"seed{seed}-tr")
    Xte = z287.items_to_mfcc(te_items, desc=f"seed{seed}-te")
    Xtr_n, mn, mx = z287.normalize_mfcc(Xtr)
    Xte_n, _, _   = z287.normalize_mfcc(Xte, mn, mx)
    Xtr_n = Xtr_n.reshape(-1, z287.N_FRAMES, z287.N_MFCC)
    Xte_n = Xte_n.reshape(-1, z287.N_FRAMES, z287.N_MFCC)

    Xtr_t = torch.tensor(Xtr_n, device=device)
    Xte_t = torch.tensor(Xte_n, device=device)
    ytr_t = torch.tensor(ytr, device=device)
    yte_t = torch.tensor(yte, device=device)

    g = torch.Generator(device=device).manual_seed(seed + 31337)
    VG1_bias = torch.empty(NSRAM_N, device=device).uniform_(VG1_LO, VG1_HI, generator=g)
    VG2_bias = torch.empty(NSRAM_N, device=device).uniform_(VG2_LO, VG2_HI, generator=g)

    def featurize(X_t, batch=16):
        feats_list, spk_list, tpc_list = [], [], []
        for i in range(0, X_t.shape[0], batch):
            f, sp, tpc = nsram_magnitude_features(
                X_t[i:i+batch], surr, VG1_bias, VG2_bias, generator=g)
            feats_list.append(f); spk_list.append(sp); tpc_list.append(tpc)
        return torch.cat(feats_list, 0), torch.cat(spk_list, 0), torch.cat(tpc_list, 0)

    t0 = time.time()
    feats_tr, spk_tr, tpc_tr = featurize(Xtr_t)
    feats_te, spk_te, tpc_te = featurize(Xte_t)
    wall = time.time() - t0

    # sparsity diagnostics
    # fraction of (sample, cell) pairs with zero spikes across entire utterance
    zero_cells_tr = float((tpc_tr == 0).float().mean().item())
    zero_cells_te = float((tpc_te == 0).float().mean().item())
    mean_spikes_per_cell_te = float((tpc_te.mean()).item())
    print(f"  [seed {seed}] wall={wall:.1f}s mean_spikes/inf={spk_te.mean().item():.1f} "
          f"mean_spikes/cell={mean_spikes_per_cell_te:.2f} "
          f"zero_cells_te={zero_cells_te:.3f}", flush=True)

    # ridge readout
    W = z287.ridge_lstsq(feats_tr, ytr_t, alpha=1.0)
    tr_pred = (feats_tr @ W).argmax(dim=1)
    te_pred = (feats_te @ W).argmax(dim=1)
    tr_acc = (tr_pred == ytr_t).float().mean().item()
    te_acc = (te_pred == yte_t).float().mean().item()
    cm = confusion(yte_t.cpu().numpy(), te_pred.cpu().numpy(), z287.N_CLASSES)

    mean_spk = float(spk_te.mean().item())
    energy_nJ = mean_spk * ENERGY_PER_SPIKE_FJ * 1e-15 * 1e9
    return {
        "seed": seed,
        "train_acc": tr_acc,
        "acc": te_acc,
        "mean_spikes_per_inf": mean_spk,
        "mean_spikes_per_cell_te": mean_spikes_per_cell_te,
        "zero_cell_frac_te": zero_cells_te,
        "energy_per_inf_nJ": energy_nJ,
        "wall_s": wall,
        "confusion": cm.tolist(),
    }


def main():
    # thermal check
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            apu = int(f.read()) / 1000.0
        print(f"[z297b] APU={apu:.1f}C", flush=True)
        if apu > 88:
            print(f"[z297b] ABORT: APU too hot ({apu:.1f}>88C)"); return 2
    except Exception as e:
        print(f"[z297b] thermal check skipped: {e}", flush=True)

    data_root = ROOT / "data/speech_commands"
    if not (data_root / "yes").exists():
        print(f"[z297b] ERROR: dataset missing at {data_root}"); return 2

    splits, bg_files = z287.scan_dataset(data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[z297b] device={device}", flush=True)
    surr = z287.load_surrogate(z287.SURROGATE_PATH, device)

    per_seed = []
    for s in SEEDS:
        r = run_seed(s, splits, bg_files, surr, device)
        print(f"[z297b] seed={s} acc={r['acc']:.4f} train_acc={r['train_acc']:.4f} "
              f"energy={r['energy_per_inf_nJ']:.3f}nJ", flush=True)
        per_seed.append(r)

    mean_acc = float(np.mean([r["acc"] for r in per_seed]))
    std_acc  = float(np.std([r["acc"] for r in per_seed]))
    if mean_acc >= 0.50:
        verdict = "AMBITIOUS"
    elif mean_acc >= 0.25:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    # aggregated confusion
    cm_agg = np.sum([np.array(r["confusion"]) for r in per_seed], axis=0)

    out = {
        "task": "DS-N1b KWS magnitude-encoded MFCC -> NS-RAM spike-rate readout",
        "verdict": verdict,
        "mean_acc": mean_acc,
        "std_acc":  std_acc,
        "n_classes": z287.N_CLASSES,
        "classes":   z287.CLASSES,
        "per_seed":  per_seed,
        "aggregated_confusion": cm_agg.tolist(),
        "config": {
            "NSRAM_N": NSRAM_N,
            "N_CHUNKS": N_CHUNKS,
            "SUB_STEPS": SUB_STEPS,
            "VD_LO": VD_LO, "VD_HI": VD_HI,
            "VG1_LO": VG1_LO, "VG1_HI": VG1_HI,
            "VG2_LO": VG2_LO, "VG2_HI": VG2_HI,
            "LOGID_SPIKE_THRESH": LOGID_SPIKE_THRESH, "VB_RESET": VB_RESET,
            "n_per_class_train": N_PER_CLASS_TRAIN, "n_per_class_test": N_PER_CLASS_TEST,
            "n_train_cap": N_TRAIN_CAP, "n_test_cap": N_TEST_CAP,
        },
        "device": str(device),
        "node": os.uname().nodename,
    }
    out_dir = ROOT / "results/z297b_ds_n1b_kws_mag"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"[z297b] VERDICT={verdict} mean_acc={mean_acc:.4f} +/- {std_acc:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
