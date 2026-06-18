"""DS-N5c: NS-RAM HDC with TRUE DIFFERENTIAL-PAIR per HD bit.

Pattern: mirrors z266 (N4 -> N4c, R^2=0.006 -> 0.977) for HDC.

Per HD bit i (N=128), we instantiate TWO physical NS-RAM neurons (n_pos, n_neg).
For class prototype p_c[i] in {-1, +1}:
    p_c[i] = +1 -> V_G2(n_pos) = V_G2_HIGH (0.50 V),  V_G2(n_neg) = V_G2_LOW  (0.00 V)
    p_c[i] = -1 -> V_G2(n_pos) = V_G2_LOW  (0.00 V),  V_G2(n_neg) = V_G2_HIGH (0.50 V)
At query, BOTH neurons are driven with the same input V_G1[i] = V_G1_BIAS + g_in * h_norm[i].
Effective bit similarity = I_d(n_pos) - I_d(n_neg).

Locked params (same as DS-N5):
    N = 128, Q = 32, V_G2_HIGH=0.50, V_G2_LOW=0.00
    V_G1_BIAS = 0.30   (per N4c convention; was 0.50 in DS-N5 but the spec
                       in the task asks V_G1=0.30 V same as N4c)
    g_in = 0.25, C_b = 8 fF, dt_s = 1e-7 s, T_steps = 100, Vd = 1.0 V

Energy is ~2x DS-N5 because we now run 2 physical neurons per HD bit.
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


def run_seed(Xtr, ytr, Xte, yte, surr, device, N, Q, seed, n_classes,
             V_G1_BIAS=0.30, V_G2_HIGH=0.50, V_G2_LOW=0.00,
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

    protos = build_prototypes(Htr_int, ytr, n_classes, N)   # (C, N) ±1

    Hte_n = hdc_query_normalize(Hte_int)
    Htr_n = hdc_query_normalize(Htr_int)

    VG1_min = float(surr["ax_VG1"][0].item())
    VG1_max = float(surr["ax_VG1"][-1].item())

    def vg2_pair(p):
        """Return (VG2_pos, VG2_neg) for prototype row p ∈ {-1,+1}^N."""
        vg2_pos = np.where(p > 0, V_G2_HIGH, V_G2_LOW).astype(np.float32)
        vg2_neg = np.where(p > 0, V_G2_LOW,  V_G2_HIGH).astype(np.float32)
        return vg2_pos, vg2_neg

    # Diagnostic: effective bit separation. We sample a representative input
    # at h_norm = +1 and h_norm = 0 (mid) over the whole bit range and check
    # |I_d(pos) - I_d(neg)| dynamic range for the two prototype-bit signs.
    bit_sep_diag = {}

    def score_set(H_norm, y_true, collect_diag=False):
        Nset = H_norm.shape[0]
        scores = np.zeros((Nset, n_classes), dtype=np.float32)
        total_spike_events = 0.0  # counted on BOTH neurons (pos+neg)
        diag_pos_p1 = []  # I_d_pos when proto bit = +1
        diag_neg_p1 = []
        diag_pos_m1 = []
        diag_neg_m1 = []
        for c in range(n_classes):
            vg2_pos_np, vg2_neg_np = vg2_pair(protos[c])
            VG2_pos_t = torch.tensor(vg2_pos_np, dtype=torch.float32,
                                     device=device)
            VG2_neg_t = torch.tensor(vg2_neg_np, dtype=torch.float32,
                                     device=device)
            for b0 in range(0, Nset, batch_size):
                b1 = min(b0 + batch_size, Nset)
                H_b = torch.tensor(H_norm[b0:b1], dtype=torch.float32,
                                   device=device)
                VG1 = (V_G1_BIAS + g_in * H_b).clamp(VG1_min, VG1_max)
                # Drive both pos and neg neuron with SAME V_G1[i]
                VG2_p = VG2_pos_t.expand(b1 - b0, N)
                VG2_n = VG2_neg_t.expand(b1 - b0, N)
                rates_p, spikes_p = nsram_rates(VG1, VG2_p, surr,
                                                C_b_F, dt_s, T_steps)
                rates_n, spikes_n = nsram_rates(VG1, VG2_n, surr,
                                                C_b_F, dt_s, T_steps)
                # Effective bit readout = I_d(pos) - I_d(neg)
                rates_eff = rates_p - rates_n
                # NS-RAM differential-pair analog cosine:
                #   sum over bits of (I_d_pos - I_d_neg) * h_norm
                # Note: protos already implicit in V_G2 mapping.
                s = (rates_eff * H_b).sum(dim=1).cpu().numpy()
                scores[b0:b1, c] = s
                total_spike_events += float(spikes_p.sum().item()
                                            + spikes_n.sum().item())
                if collect_diag and c == 0:
                    proto_t = torch.tensor(protos[c], dtype=torch.float32,
                                           device=device).expand(b1 - b0, N)
                    mask_p1 = (proto_t > 0)
                    mask_m1 = (proto_t < 0)
                    if mask_p1.any():
                        diag_pos_p1.append(rates_p[mask_p1].cpu().numpy())
                        diag_neg_p1.append(rates_n[mask_p1].cpu().numpy())
                    if mask_m1.any():
                        diag_pos_m1.append(rates_p[mask_m1].cpu().numpy())
                        diag_neg_m1.append(rates_n[mask_m1].cpu().numpy())
        preds = scores.argmax(axis=1)
        acc = float((preds == y_true).mean())
        avg_events = total_spike_events / max(1, Nset)
        diag = None
        if collect_diag and diag_pos_p1:
            dp1 = np.concatenate(diag_pos_p1)
            dn1 = np.concatenate(diag_neg_p1)
            dpm = np.concatenate(diag_pos_m1) if diag_pos_m1 else np.array([0.])
            dnm = np.concatenate(diag_neg_m1) if diag_neg_m1 else np.array([0.])
            eff_p1 = float(np.mean(dp1 - dn1))   # bit=+1, expect >0
            eff_m1 = float(np.mean(dpm - dnm))   # bit=-1, expect <0
            diag = {
                "eff_readout_bit_plus1_mean_A":  eff_p1,
                "eff_readout_bit_minus1_mean_A": eff_m1,
                "effective_bit_separation_A":    float(eff_p1 - eff_m1),
                "I_d_pos_mean_when_bit_plus1_A":  float(dp1.mean()),
                "I_d_neg_mean_when_bit_plus1_A":  float(dn1.mean()),
                "I_d_pos_mean_when_bit_minus1_A": float(dpm.mean()),
                "I_d_neg_mean_when_bit_minus1_A": float(dnm.mean()),
            }
        return acc, avg_events, diag

    train_acc, train_ev, _    = score_set(Htr_n, ytr)
    test_acc,  test_ev, diag  = score_set(Hte_n, yte, collect_diag=True)
    wall = time.time() - t0

    energy_per_inf_J = test_ev * 6.4e-15
    return {
        "seed": seed,
        "train_acc": train_acc, "test_acc": test_acc,
        "wall_s": wall,
        "avg_spike_events_per_inference": test_ev,
        "energy_J_per_inference": energy_per_inf_J,
        "N_bits": int(N),
        "N_physical_neurons_per_encoding": int(2 * N),
        "Q": int(Q), "T_steps": int(T_steps),
        "V_G2_HIGH": V_G2_HIGH, "V_G2_LOW": V_G2_LOW,
        "V_G1_BIAS": V_G1_BIAS, "g_in": g_in,
        "C_b_fF": C_b_F * 1e15, "dt_s": dt_s,
        "diagnostic": diag,
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
    p.add_argument("--V_G1_BIAS", type=float, default=0.30)
    p.add_argument("--single_acc",    type=float, default=0.6045)
    p.add_argument("--fair_baseline", type=float, default=0.8110)
    p.add_argument("--baseline_json",
                   default="results/z285_nsram_hdc/baseline_hdc.json")
    p.add_argument("--out",
                   default="results/z288_nsram_hdc_diffpair/summary.json")
    args = p.parse_args()

    device = get_device()
    print(f"[z288] device={device}", flush=True)
    surr = load_surrogate_torch(args.surrogate, device)
    Xtr, ytr, Xte, yte = load_uci_har(args.data_root)
    print(f"[z288] train {Xtr.shape} test {Xte.shape}", flush=True)
    n_classes = int(max(ytr.max(), yte.max())) + 1

    per_seed = []
    for s in args.seeds:
        try:
            r = run_seed(Xtr, ytr, Xte, yte, surr, device,
                         args.N, args.Q, s, n_classes,
                         V_G1_BIAS=args.V_G1_BIAS,
                         T_steps=args.T_steps,
                         batch_size=args.batch_size)
        except Exception as e:
            import traceback; traceback.print_exc()
            r = {"seed": s, "error": repr(e)}
        per_seed.append(r)
        if "test_acc" in r:
            d = r.get("diagnostic") or {}
            print(f"  seed {s}: test_acc={r['test_acc']:.4f} "
                  f"train_acc={r['train_acc']:.4f} "
                  f"wall={r['wall_s']:.1f}s "
                  f"E={r['energy_J_per_inference']*1e9:.3f} nJ/inf "
                  f"bit_sep={d.get('effective_bit_separation_A',0):.3e} A",
                  flush=True)
        else:
            print(f"  seed {s}: ERROR {r.get('error')}", flush=True)

    accs = [r["test_acc"] for r in per_seed if "test_acc" in r]
    energies = [r["energy_J_per_inference"] for r in per_seed
                if "energy_J_per_inference" in r]
    seps = [r["diagnostic"]["effective_bit_separation_A"]
            for r in per_seed
            if r.get("diagnostic") is not None]

    summary = {
        "experiment": "z288_nsram_hdc_diffpair_uci_har",
        "n_seeds": len(args.seeds),
        "N_bits": int(args.N),
        "N_physical_neurons_per_encoding": int(2 * args.N),
        "Q_levels": int(args.Q),
        "T_steps": int(args.T_steps),
        "V_G1_BIAS": args.V_G1_BIAS,
        "per_seed": per_seed,
        "nsram_diffpair_acc_per_seed": accs,
        "mean_acc": float(np.mean(accs)) if accs else None,
        "std_acc":  float(np.std(accs)) if accs else None,
        "mean_energy_J_per_inference":
            float(np.mean(energies)) if energies else None,
        "energy_per_spike_fJ": 6.4,
        "effective_bit_separation_A_mean":
            float(np.mean(seps)) if seps else None,
        "effective_bit_separation_A_max":
            float(np.max(seps)) if seps else None,
    }
    if len(accs) >= 2:
        rng = np.random.default_rng(0)
        bs = np.array([rng.choice(accs, len(accs), replace=True).mean()
                       for _ in range(4000)])
        summary["ci95"] = [float(np.quantile(bs, 0.025)),
                           float(np.quantile(bs, 0.975))]

    summary["delta_vs_single_end_DS_N5"] = (
        summary["mean_acc"] - args.single_acc
        if summary["mean_acc"] is not None else None)
    summary["delta_vs_fair_baseline"] = (
        summary["mean_acc"] - args.fair_baseline
        if summary["mean_acc"] is not None else None)
    summary["single_end_DS_N5_acc"] = args.single_acc
    summary["fair_baseline_acc"] = args.fair_baseline

    # Pre-registered gate
    mean_acc = summary["mean_acc"]
    ci_lo = summary.get("ci95", [None, None])[0]
    intermediate = (mean_acc is not None and ci_lo is not None
                    and ci_lo > args.single_acc)
    conservative = (mean_acc is not None and mean_acc >= 0.76)
    ambitious    = (mean_acc is not None and mean_acc >= args.fair_baseline)
    if ambitious:
        verdict = "AMBITIOUS_PASS"
    elif conservative:
        verdict = "CONSERVATIVE_PASS"
    elif intermediate:
        verdict = "INTERMEDIATE_PASS"
    else:
        verdict = "FAIL"
    summary["gates"] = {
        "intermediate_pass_ci_above_single": bool(intermediate),
        "conservative_pass_geq_0.76":        bool(conservative),
        "ambitious_pass_geq_fair_baseline":  bool(ambitious),
        "verdict": verdict,
    }

    # Pull HDC baseline if available
    bp = Path(args.baseline_json)
    if bp.exists():
        try:
            bd = json.loads(bp.read_text())
            summary["baseline_hdc_acc_file"] = bd.get("mean_acc")
        except Exception:
            pass

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[z288] DONE mean_acc={summary['mean_acc']} "
          f"verdict={verdict} "
          f"E={summary['mean_energy_J_per_inference']} J/inf -> {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
