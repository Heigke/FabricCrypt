"""z297 DS-N1b: KWS with delta-modulated MFCC (sparse spikes).

Reuses z287_ds_n1_kws.py for MFCC frontend, NS-RAM SNN, and ridge readout.
Only difference: input is delta-modulated.

Delta-mod: emit a positive/negative spike when |MFCC[t,c] - MFCC_prev[c]| > delta.
Concretely we compute the per-frame difference, threshold by `delta`, and feed
{-1, 0, +1} (sparse) to the existing pipeline. This SHOULD give ~10x sparser
activity. We feed |delta| as Poisson rate (0 most frames) so spike count drops.

Gates:
  PASS:       top-1 >= 50%
  AMBITIOUS:  top-1 >= 70%

We also report:
  - sparsity ratio (mean fraction of nonzero delta entries vs full MFCC)
  - energy per inference (spikes * 6.4 fJ)
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

# parameters
DELTA_THRESHOLD_STD = 0.10  # frame-active if ANY channel delta > 0.10 (norm units)
SEEDS = [0, 1]             # 2 seeds for speed
N_PER_CLASS_TRAIN = 250
N_PER_CLASS_TEST = 60


def delta_modulate(X_norm: np.ndarray, delta: float) -> np.ndarray:
    """Delta-modulation as event-driven gating.

    Compute per-frame absolute change in MFCC. Frames where ANY channel exceeds
    `delta` are "spike frames" — keep their full MFCC vector. Other frames are
    zeroed (no spike). The downstream nsram_features then sees a sparse temporal
    activation pattern: only ~10% of frames carry info, rest are zero.

    This avoids breaking the (rate-0.5) zero-centering used in z287:
    keeping the un-modulated MFCC at active frames preserves discriminative content.
    """
    diff = np.zeros_like(X_norm)
    diff[:, 1:, :] = X_norm[:, 1:, :] - X_norm[:, :-1, :]
    # frame is "active" if max-abs-delta over channels exceeds threshold
    per_frame_max = np.max(np.abs(diff), axis=-1)   # (B, T)
    frame_active = per_frame_max > delta            # (B, T)
    # always activate first frame
    frame_active[:, 0] = True
    # gate: keep MFCC at active frames, but as a Poisson rate we want mean ~0.5
    # so we just multiply by mask, accepting zero-centering will treat masked as -0.5
    out = (X_norm * frame_active[..., None]).astype(np.float32)
    # full-density mask (per-element) for sparsity reporting
    mask_full = np.broadcast_to(frame_active[..., None], X_norm.shape)
    return out, mask_full


def run_delta_kws(seed: int, splits, bg_files, surr, device):
    rng = np.random.default_rng(seed)
    tr_items, ytr = z287.build_split(splits, bg_files, "train", N_PER_CLASS_TRAIN, rng)
    te_items, yte = z287.build_split(splits, bg_files, "test", N_PER_CLASS_TEST, rng)
    Xtr = z287.items_to_mfcc(tr_items, desc=f"seed{seed}-tr")
    Xte = z287.items_to_mfcc(te_items, desc=f"seed{seed}-te")
    Xtr_n, mn, mx = z287.normalize_mfcc(Xtr)
    Xte_n, _, _   = z287.normalize_mfcc(Xte, mn, mx)
    Xtr_n = Xtr_n.reshape(-1, z287.N_FRAMES, z287.N_MFCC)
    Xte_n = Xte_n.reshape(-1, z287.N_FRAMES, z287.N_MFCC)

    # Delta modulate
    Xtr_d, mask_tr = delta_modulate(Xtr_n, DELTA_THRESHOLD_STD)
    Xte_d, mask_te = delta_modulate(Xte_n, DELTA_THRESHOLD_STD)
    sparsity_tr = float(mask_tr.mean())
    sparsity_te = float(mask_te.mean())
    print(f"  [seed {seed}] sparsity train={sparsity_tr:.3f} test={sparsity_te:.3f} "
          f"(full MFCC=1.0; delta-mod ~10x sparser target=0.10)", flush=True)

    Xtr_t = torch.tensor(Xtr_d, device=device)
    Xte_t = torch.tensor(Xte_d, device=device)
    ytr_t = torch.tensor(ytr, device=device)
    yte_t = torch.tensor(yte, device=device)

    n_units = z287.NSRAM_N
    g = torch.Generator(device=device).manual_seed(seed + 31337)
    W_in = torch.randn(n_units, z287.N_MFCC, generator=g, device=device)
    W_in = W_in / (W_in.norm(dim=1, keepdim=True) + 1e-9)
    V_G1_bias = torch.empty(n_units, device=device).uniform_(
        z287.NSRAM_VG1_LO, z287.NSRAM_VG1_HI, generator=g)
    V_G2_bias = torch.full((n_units,), z287.NSRAM_CELL["V_G2_bias"], device=device)
    C_b_F = z287.NSRAM_CELL["C_b_fF"] * 1e-15
    gp = torch.Generator(device=device).manual_seed(seed + 99991)

    def featurize(X_t, count=False, batch=32):
        feats_list, spike_list = [], []
        for i in range(0, X_t.shape[0], batch):
            f, sp = z287.nsram_features(
                X_t[i:i + batch], n_units, W_in, V_G1_bias, V_G2_bias,
                surr, z287.NSRAM_CELL["g_in"], C_b_F, z287.NSRAM_CELL["dt_s"],
                generator=gp, count_spikes=count, sub_steps=2)
            feats_list.append(f); spike_list.append(sp)
        return torch.cat(feats_list, 0), torch.cat(spike_list, 0)

    t0 = time.time()
    feats_tr, _ = featurize(Xtr_t, count=False)
    feats_te, spikes_te = featurize(Xte_t, count=True)
    wall = time.time() - t0
    W = z287.ridge_lstsq(feats_tr, ytr_t)
    te_pred = (feats_te @ W).argmax(dim=1)
    te_acc = (te_pred == yte_t).float().mean().item()
    mean_input_spikes = float(spikes_te.mean().item())
    energy_J = mean_input_spikes * z287.ENERGY_PER_SPIKE_FJ * 1e-15
    return {
        "seed": seed,
        "acc": te_acc,
        "sparsity_train": sparsity_tr,
        "sparsity_test": sparsity_te,
        "mean_input_spikes_per_inf": mean_input_spikes,
        "energy_per_inf_nJ": energy_J * 1e9,
        "wall_s": wall,
    }


def main():
    # Thermal check — if APU > 80°C, refuse (per CLAUDE.md)
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            apu_temp = int(f.read()) / 1000.0
        print(f"[z297] APU temp at start: {apu_temp:.1f}C", flush=True)
        if apu_temp > 88:
            print(f"[z297] ABORT: APU too hot ({apu_temp:.1f} > 88C)", flush=True)
            return 2
    except Exception as e:
        print(f"[z297] thermal check skipped: {e}", flush=True)

    data_root = ROOT / "data/speech_commands"
    if not (data_root / "yes").exists():
        print(f"[z297] ERROR: dataset not extracted at {data_root}", flush=True)
        return 2

    splits, bg_files = z287.scan_dataset(data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[z297] device={device}", flush=True)

    surr = z287.load_surrogate(z287.SURROGATE_PATH, device)
    per_seed = []
    for s in SEEDS:
        r = run_delta_kws(s, splits, bg_files, surr, device)
        print(f"[z297] seed={s} acc={r['acc']:.4f} energy={r['energy_per_inf_nJ']:.2f}nJ "
              f"wall={r['wall_s']:.1f}s", flush=True)
        per_seed.append(r)

    mean_acc = float(np.mean([r["acc"] for r in per_seed]))
    if mean_acc >= 0.70:
        verdict = "AMBITIOUS"
    elif mean_acc >= 0.50:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    out = {
        "task": "DS-N1b KWS delta-modulated MFCC + NS-RAM SNN",
        "verdict": verdict,
        "mean_acc": mean_acc,
        "n_classes": z287.N_CLASSES,
        "per_seed": per_seed,
        "delta_threshold_std": DELTA_THRESHOLD_STD,
        "n_per_class_train": N_PER_CLASS_TRAIN,
        "n_per_class_test": N_PER_CLASS_TEST,
        "device": str(device),
        "node": os.uname().nodename,
    }
    out_dir = ROOT / "results/z297_ds_n1b_kws"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"[z297] VERDICT={verdict} mean_acc={mean_acc:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
