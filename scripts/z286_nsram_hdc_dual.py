"""DS-N5b: Dual-polarity NS-RAM HDC classifier on UCI-HAR.

Fork of z285 (DS-N5). The single-end readout in DS-N5 collapsed to 60.45%
because V_G2 modulates I_d by ~6 orders of magnitude, so only the
(V_G2_HIGH, h>0) quadrant materially contributes to the score. The fix
(pre-registered in DS-N5) is dual-polarity readout: for each class c,
store BOTH the prototype P_c AND its bit-complement ~P_c as a second
V_G2 map; effective similarity is

    sim_c = readout(P_c, h) - readout(~P_c, h)

This restores symmetric +1 and -1 contributions while keeping the
encoding rule (V_G2_HIGH = 0.50 V for +1, V_G2_LOW = 0.00 V for -1)
LOCKED. Two reads per class -> energy ~2x DS-N5's ~0.3 nJ.

Gates:
  INTERMEDIATE: mean > 60.45% with non-overlapping CI95
  CONSERVATIVE: >= 76% (within 5pp of fair HDC 81.1%)
  AMBITIOUS:    >= 81.1% AND energy <= 1 nJ
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
from z285_nsram_hdc import (
    load_surrogate_torch, nsram_rates, build_prototypes,
    hdc_query_normalize, get_device,
)


def run_seed_dual(Xtr, ytr, Xte, yte, surr, device, N, Q, seed, n_classes,
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
    Htr_int = encode_samples(Xtrq, P_pos, L_lev)
    Hte_int = encode_samples(Xteq, P_pos, L_lev)

    protos = build_prototypes(Htr_int, ytr, n_classes, N)  # (C, N) +/- 1

    Hte_n = hdc_query_normalize(Hte_int)
    Htr_n = hdc_query_normalize(Htr_int)

    VG1_min = float(surr["ax_VG1"][0].item())
    VG1_max = float(surr["ax_VG1"][-1].item())

    def vg2_map(p):
        return np.where(p > 0, V_G2_HIGH, V_G2_LOW).astype(np.float32)

    def score_set(H_norm, y_true):
        Nset = H_norm.shape[0]
        scores = np.zeros((Nset, n_classes), dtype=np.float32)
        total_spike_events = 0.0
        for c in range(n_classes):
            # POS readout: V_G2 mapped from prototype
            VG2_pos = vg2_map(protos[c])
            # NEG readout: V_G2 mapped from complement (-prototype)
            VG2_neg = vg2_map(-protos[c])
            for b0 in range(0, Nset, batch_size):
                b1 = min(b0 + batch_size, Nset)
                H_b = torch.tensor(H_norm[b0:b1], dtype=torch.float32,
                                   device=device)
                VG1 = (V_G1_BIAS + g_in * H_b).clamp(VG1_min, VG1_max)
                VG2_p = torch.tensor(VG2_pos, dtype=torch.float32,
                                     device=device).expand(b1 - b0, N)
                VG2_n = torch.tensor(VG2_neg, dtype=torch.float32,
                                     device=device).expand(b1 - b0, N)
                rates_p, spikes_p = nsram_rates(VG1, VG2_p, surr, C_b_F,
                                                dt_s, T_steps)
                rates_n, spikes_n = nsram_rates(VG1, VG2_n, surr, C_b_F,
                                                dt_s, T_steps)
                s_p = (rates_p * H_b).sum(dim=1).cpu().numpy()
                s_n = (rates_n * H_b).sum(dim=1).cpu().numpy()
                scores[b0:b1, c] = s_p - s_n
                total_spike_events += float(spikes_p.sum().item())
                total_spike_events += float(spikes_n.sum().item())
        preds = scores.argmax(axis=1)
        acc = float((preds == y_true).mean())
        avg_events = total_spike_events / max(1, Nset)
        return acc, avg_events

    train_acc, train_ev = score_set(Htr_n, ytr)
    test_acc, test_ev = score_set(Hte_n, yte)
    wall = time.time() - t0

    energy_per_inf_J = test_ev * 6.4e-15
    return {
        "seed": seed,
        "train_acc": train_acc, "test_acc": test_acc,
        "wall_s": wall,
        "avg_spike_events_per_inference": test_ev,
        "energy_J_per_inference": energy_per_inf_J,
        "N": int(N), "Q": int(Q), "T_steps": int(T_steps),
        "V_G2_HIGH": V_G2_HIGH, "V_G2_LOW": V_G2_LOW,
        "V_G1_BIAS": V_G1_BIAS, "g_in": g_in,
        "reads_per_class": 2,
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
    p.add_argument("--single_end_acc", type=float, default=0.6045,
                   help="DS-N5 mean acc to compare against")
    p.add_argument("--single_end_ci_lo", type=float, default=None)
    p.add_argument("--single_end_ci_hi", type=float, default=None)
    p.add_argument("--baseline_acc", type=float, default=0.8110,
                   help="Fair-D HDC baseline mean acc")
    p.add_argument("--out", default="results/z286_nsram_hdc_dual/summary.json")
    args = p.parse_args()

    device = get_device()
    print(f"[z286] device={device}", flush=True)
    surr = load_surrogate_torch(args.surrogate, device)
    Xtr, ytr, Xte, yte = load_uci_har(args.data_root)
    print(f"[z286] train {Xtr.shape} test {Xte.shape}", flush=True)
    n_classes = int(max(ytr.max(), yte.max())) + 1

    # Try to import DS-N5 CI if not given
    if args.single_end_ci_lo is None or args.single_end_ci_hi is None:
        ds5_path = Path("results/z285_nsram_hdc/summary.json")
        if ds5_path.exists():
            try:
                ds5 = json.loads(ds5_path.read_text())
                ci = ds5.get("ci95")
                if ci is not None and len(ci) == 2:
                    args.single_end_ci_lo = float(ci[0])
                    args.single_end_ci_hi = float(ci[1])
                if ds5.get("mean_acc") is not None:
                    args.single_end_acc = float(ds5["mean_acc"])
            except Exception:
                pass

    per_seed = []
    for s in args.seeds:
        try:
            r = run_seed_dual(Xtr, ytr, Xte, yte, surr, device,
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
        "experiment": "z286_nsram_hdc_dual_polarity_uci_har",
        "fork_of": "z285_nsram_hdc",
        "n_seeds": len(args.seeds),
        "N_units": int(args.N), "Q_levels": int(args.Q),
        "T_steps": int(args.T_steps),
        "per_seed": per_seed,
        "nsram_hdc_dual_acc_per_seed": accs,
        "mean_acc": float(np.mean(accs)) if accs else None,
        "std_acc": float(np.std(accs)) if accs else None,
        "mean_energy_J_per_inference":
            float(np.mean(energies)) if energies else None,
        "energy_per_spike_fJ": 6.4,
        "bit_count": int(args.N),
        "reads_per_class": 2,
        "single_end_acc_DS_N5": args.single_end_acc,
        "single_end_ci95_DS_N5": [args.single_end_ci_lo,
                                  args.single_end_ci_hi]
        if args.single_end_ci_lo is not None else None,
        "fair_baseline_acc_HDC_D128": args.baseline_acc,
    }
    if len(accs) >= 2:
        rng = np.random.default_rng(0)
        bs = np.array([rng.choice(accs, len(accs), replace=True).mean()
                       for _ in range(4000)])
        ci_lo = float(np.quantile(bs, 0.025))
        ci_hi = float(np.quantile(bs, 0.975))
        summary["ci95"] = [ci_lo, ci_hi]
    else:
        ci_lo = ci_hi = None

    mean_acc = summary["mean_acc"]
    mean_E_J = summary["mean_energy_J_per_inference"] or 0.0
    summary["delta_vs_single_end"] = (
        mean_acc - args.single_end_acc if mean_acc is not None else None)
    summary["delta_vs_fair_baseline"] = (
        mean_acc - args.baseline_acc if mean_acc is not None else None)

    # Gate verdicts
    se_ci_hi = args.single_end_ci_hi
    intermediate = (
        mean_acc is not None and mean_acc > args.single_end_acc
        and ci_lo is not None and se_ci_hi is not None
        and ci_lo > se_ci_hi
    )
    conservative = mean_acc is not None and mean_acc >= 0.76
    ambitious = (mean_acc is not None and mean_acc >= args.baseline_acc
                 and mean_E_J <= 1e-9)
    summary["verdict"] = {
        "intermediate_pass": bool(intermediate),
        "intermediate_rule": "mean_acc > single_end AND CI95 non-overlap",
        "conservative_pass": bool(conservative),
        "conservative_rule": "mean_acc >= 0.76",
        "ambitious_pass": bool(ambitious),
        "ambitious_rule": "mean_acc >= 0.811 AND energy <= 1 nJ",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[z286] DONE mean_acc={mean_acc} "
          f"E={mean_E_J*1e9:.3f} nJ/inf -> {out_path}", flush=True)
    print(f"[z286] verdict={summary['verdict']}", flush=True)


if __name__ == "__main__":
    main()
