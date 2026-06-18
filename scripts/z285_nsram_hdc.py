"""DS-N5: NS-RAM HDC classifier on UCI-HAR.

Each of N=128 NS-RAM neurons represents ONE bipolar bit of a 128-D HV.
Class prototype stored as V_G2 bias per neuron (analog memory, validated in
N4c). Inference reads spike-rate pattern under input-driven V_G1.

Wiring (LOCKED before run):
  - HDC encoder identical to z284 but with D = N_units = 128, Q_LEVELS = 32.
    record-based: H(x) = Σ_f P_f * L_f[q_f(x)],  bipolar bundled, kept as int.
  - Per-class prototype p_c ∈ {-1,+1}^128 = sign(Σ_{i∈c} H(x_i)).
  - V_G2 mapping: p_c[i] = +1 -> V_G2_HIGH = 0.50 V (low-leak / memory-on),
                  p_c[i] = -1 -> V_G2_LOW  = 0.00 V (high-leak / memory-off).
    Both are within validated surrogate axis [-0.10, 0.60].
  - V_G1[i] drive for query h: V_G1[i] = V_G1_BIAS + g_in * h_norm[i],
    where h_norm = h / (|h|_inf + eps) ∈ [-1, 1].
  - Vd = 1.0 V; C_b = 8 fF; dt_s = 1e-7 s; T_steps = 100; Vb starts at 0.
  - Score_c = Σ_i rate_i(V_G1[i], p_c-mapped V_G2[i]) * h_norm[i].
    Argmax over c. This is the NS-RAM analog of cosine similarity.

Seeds: 4 initial (then optional 10 if PASS).
Energy estimate: E_per_inf = (Σ neuron spike events over T_steps * num_classes)
  * 6.4 fJ. Spike event = |I_d| * dt_s > q_e (1 elementary charge).

Usage:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python z285_nsram_hdc.py \
      --data_root data/uci_har/'UCI HAR Dataset' \
      --surrogate results/z278_mep2_surrogate_v3/surrogate_4d_v3.npz \
      --N 128 --Q 32 --seeds 0 1 2 3 \
      --out results/z285_nsram_hdc/summary.json
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from z284_hdc_baseline import (
    load_uci_har, build_codebooks, quantize, encode_samples,
)

Q_ELEM = 1.602176634e-19  # C, elementary charge


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_surrogate_torch(path, device):
    z = np.load(path)
    return {
        "I_d":   torch.tensor(z["Id"],    dtype=torch.float32, device=device),
        "I_ii":  torch.tensor(z["Iii"],   dtype=torch.float32, device=device),
        "I_leak":torch.tensor(z["Ileak"], dtype=torch.float32, device=device),
        "ax_VG1":torch.tensor(z["vg1_axis"], dtype=torch.float32, device=device),
        "ax_VG2":torch.tensor(z["vg2_axis"], dtype=torch.float32, device=device),
        "ax_Vd": torch.tensor(z["vd_axis"],  dtype=torch.float32, device=device),
        "ax_Vb": torch.tensor(z["vb_axis"],  dtype=torch.float32, device=device),
    }


def bucketize_index(values, axis):
    n = axis.shape[0]
    idx = torch.bucketize(values, axis) - 1
    return idx.clamp(0, n - 2)


def query_surrogate(surr, VG1, VG2, Vd, Vb):
    iVG1 = bucketize_index(VG1, surr["ax_VG1"])
    iVG2 = bucketize_index(VG2, surr["ax_VG2"])
    iVd  = bucketize_index(Vd,  surr["ax_Vd"])
    iVb  = bucketize_index(Vb,  surr["ax_Vb"])
    return (surr["I_d"][iVG1, iVG2, iVd, iVb],
            surr["I_ii"][iVG1, iVG2, iVd, iVb],
            surr["I_leak"][iVG1, iVG2, iVd, iVb])


def nsram_rates(VG1_batch, VG2_proto, surr, C_b_F, dt_s, T_steps, vd=1.0):
    """Run T_steps of body-state dynamics for B samples, N neurons.

    VG1_batch: (B, N) per-neuron V_G1 (already clamped).
    VG2_proto: (B, N) per-neuron V_G2 (the class prototype for each sample).
    Returns mean |I_d| over T_steps and total spike events (count) per sample.
    """
    device = VG1_batch.device
    B, N = VG1_batch.shape
    Vb_min = surr["ax_Vb"][0]
    Vb_max = surr["ax_Vb"][-1]

    rate_accum = torch.zeros(B, N, device=device)
    spike_events = torch.zeros(B, device=device)
    Vb = torch.zeros(B, N, device=device)
    Vd_2d = torch.full((B, N), float(vd), device=device)

    for _ in range(T_steps):
        Vb_c = Vb.clamp(Vb_min, Vb_max)
        I_d, I_ii, I_leak = query_surrogate(surr, VG1_batch, VG2_proto,
                                            Vd_2d, Vb_c)
        Vb = (Vb + dt_s * (I_ii - I_leak) / C_b_F).clamp(Vb_min, Vb_max)
        rate_accum = rate_accum + I_d.abs() / T_steps
        # A spike-event ~ when |I_d| * dt > q_e (one charge moved)
        spike_events = spike_events + (
            (I_d.abs() * dt_s) > Q_ELEM).float().sum(dim=1)
    return rate_accum, spike_events


def build_prototypes(Htr, ytr, n_classes, N):
    """Bipolar class prototypes (n_classes, N)."""
    P = np.zeros((n_classes, N), dtype=np.float32)
    for c in range(n_classes):
        m = ytr == c
        if m.any():
            P[c] = np.sign(Htr[m].sum(axis=0).astype(np.float32))
            P[c][P[c] == 0] = 1.0
    return P


def hdc_query_normalize(H):
    """Map int HVs to [-1, 1] via per-sample inf-norm."""
    H = H.astype(np.float32)
    maxabs = np.abs(H).max(axis=1, keepdims=True)
    maxabs = np.where(maxabs < 1e-9, 1.0, maxabs)
    return H / maxabs


def run_seed(Xtr, ytr, Xte, yte, surr, device, N, Q, seed, n_classes,
             V_G1_BIAS=0.50, V_G2_HIGH=0.50, V_G2_LOW=0.00,
             g_in=0.25, C_b_F=8e-15, dt_s=1e-7, T_steps=100,
             batch_size=64):
    rng = np.random.default_rng(seed)
    F = Xtr.shape[1]
    mins = Xtr.min(axis=0)
    maxs = Xtr.max(axis=0)
    Xtrq = quantize(Xtr, mins, maxs, Q)
    Xteq = quantize(Xte, mins, maxs, Q)
    t0 = time.time()
    P_pos, L_lev = build_codebooks(F, N, Q, rng)
    Htr_int = encode_samples(Xtrq, P_pos, L_lev)  # (Ntr, N)
    Hte_int = encode_samples(Xteq, P_pos, L_lev)

    protos = build_prototypes(Htr_int, ytr, n_classes, N)  # (C, N) ±1

    Hte_n = hdc_query_normalize(Hte_int)  # (Nte, N) ∈ [-1,1]
    Htr_n = hdc_query_normalize(Htr_int)

    VG1_min = float(surr["ax_VG1"][0].item())
    VG1_max = float(surr["ax_VG1"][-1].item())

    def vg2_map(p):  # p ∈ {-1,+1}  -> V_G2 voltage
        return np.where(p > 0, V_G2_HIGH, V_G2_LOW).astype(np.float32)

    def score_set(H_norm, y_true):
        Nset = H_norm.shape[0]
        scores = np.zeros((Nset, n_classes), dtype=np.float32)
        total_spike_events = 0.0
        for c in range(n_classes):
            VG2_c = vg2_map(protos[c])                 # (N,)
            for b0 in range(0, Nset, batch_size):
                b1 = min(b0 + batch_size, Nset)
                H_b = torch.tensor(H_norm[b0:b1], dtype=torch.float32,
                                   device=device)
                VG1 = (V_G1_BIAS + g_in * H_b).clamp(VG1_min, VG1_max)
                VG2 = torch.tensor(VG2_c, dtype=torch.float32,
                                   device=device).expand(b1 - b0, N)
                rates, spikes = nsram_rates(VG1, VG2, surr, C_b_F, dt_s,
                                             T_steps)
                # NS-RAM analog cosine: Σ rate_i * h_norm_i.
                # High-rate neurons (V_G2_HIGH, prototype bit +1) weighted
                # by input sign and magnitude approximates <h, p_c>.
                s = (rates * H_b).sum(dim=1).cpu().numpy()
                scores[b0:b1, c] = s
                total_spike_events += float(spikes.sum().item())
        preds = scores.argmax(axis=1)
        acc = float((preds == y_true).mean())
        # avg per-inference spike events: each inference scores C classes
        avg_events = total_spike_events / max(1, Nset)
        return acc, avg_events

    train_acc, train_ev = score_set(Htr_n, ytr)
    test_acc,  test_ev  = score_set(Hte_n, yte)
    wall = time.time() - t0

    energy_per_inf_J = test_ev * 6.4e-15  # 6.4 fJ per spike event
    return {
        "seed": seed,
        "train_acc": train_acc, "test_acc": test_acc,
        "wall_s": wall,
        "avg_spike_events_per_inference": test_ev,
        "energy_J_per_inference": energy_per_inf_J,
        "N": int(N), "Q": int(Q), "T_steps": int(T_steps),
        "V_G2_HIGH": V_G2_HIGH, "V_G2_LOW": V_G2_LOW,
        "V_G1_BIAS": V_G1_BIAS, "g_in": g_in,
        "C_b_fF": C_b_F * 1e15, "dt_s": dt_s,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/uci_har/UCI HAR Dataset")
    p.add_argument("--surrogate",
                   default="results/z278_mep2_surrogate_v3/surrogate_4d_v3.npz")
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--Q", type=int, default=32)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--T_steps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--baseline_json",
                   default="results/z285_nsram_hdc/baseline_hdc.json")
    p.add_argument("--out", default="results/z285_nsram_hdc/summary.json")
    args = p.parse_args()

    device = get_device()
    print(f"[z285] device={device}", flush=True)
    surr = load_surrogate_torch(args.surrogate, device)
    Xtr, ytr, Xte, yte = load_uci_har(args.data_root)
    print(f"[z285] train {Xtr.shape} test {Xte.shape}", flush=True)
    n_classes = int(max(ytr.max(), yte.max())) + 1

    per_seed = []
    for s in args.seeds:
        try:
            r = run_seed(Xtr, ytr, Xte, yte, surr, device,
                         args.N, args.Q, s, n_classes,
                         T_steps=args.T_steps,
                         batch_size=args.batch_size)
        except Exception as e:
            import traceback; traceback.print_exc()
            r = {"seed": s, "error": repr(e)}
        per_seed.append(r)
        if "test_acc" in r:
            print(f"  seed {s}: test_acc={r['test_acc']:.4f} "
                  f"train_acc={r['train_acc']:.4f} "
                  f"wall={r['wall_s']:.1f}s "
                  f"E={r['energy_J_per_inference']*1e9:.3f} nJ/inf",
                  flush=True)
        else:
            print(f"  seed {s}: ERROR {r.get('error')}", flush=True)

    accs = [r["test_acc"] for r in per_seed if "test_acc" in r]
    energies = [r["energy_J_per_inference"] for r in per_seed
                if "energy_J_per_inference" in r]
    summary = {
        "experiment": "z285_nsram_hdc_uci_har",
        "n_seeds": len(args.seeds),
        "N_units": int(args.N), "Q_levels": int(args.Q),
        "T_steps": int(args.T_steps),
        "per_seed": per_seed,
        "nsram_hdc_acc_per_seed": accs,
        "mean_acc": float(np.mean(accs)) if accs else None,
        "std_acc":  float(np.std(accs)) if accs else None,
        "mean_energy_J_per_inference":
            float(np.mean(energies)) if energies else None,
        "energy_per_spike_fJ": 6.4,
        "bit_count": int(args.N),
    }
    if len(accs) >= 2:
        rng = np.random.default_rng(0)
        bs = np.array([rng.choice(accs, len(accs), replace=True).mean()
                       for _ in range(4000)])
        summary["ci95"] = [float(np.quantile(bs, 0.025)),
                           float(np.quantile(bs, 0.975))]
    # Pull baseline if available
    bp = Path(args.baseline_json)
    if bp.exists():
        try:
            bd = json.loads(bp.read_text())
            summary["baseline_hdc_acc"] = bd.get("mean_acc")
            summary["baseline_hdc_std"] = bd.get("std_acc")
            summary["baseline_D"] = bd.get("D")
        except Exception:
            pass

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[z285] DONE mean_acc={summary['mean_acc']} "
          f"E={summary['mean_energy_J_per_inference']} J/inf -> {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
