"""DS-N5f: NS-RAM HDC with V_d (drain voltage) as the bit-axis.

DS-N5e verdict: V_G2-as-bit is fundamentally infeasible on this 4-D surrogate.
All three architectural levers (V_G1_lift, mid-bias swing, above-knee) failed.

Pivot: V_d carries the HD bit. V_d is smooth and roughly linear in I_d past
pinch-off; symmetric swing about a midpoint is natural (no hard knee).

LOCKED parameters:
  V_G1_BIAS = 0.30 V (fixed conducting; input modulates around this)
  V_G2_BIAS = 0.30 V (fixed conducting, SAME for both arms)
  V_d_HIGH  = 2.00 V
  V_d_LOW   = 0.50 V
  bit = +1  : V_d(pos_arm) = HIGH,  V_d(neg_arm) = LOW
  bit = -1  : V_d(pos_arm) = LOW,   V_d(neg_arm) = HIGH
  similarity = Σ (I_d(pos) - I_d(neg))

Surrogate Vd axis: [0.25, 3.0] V — both 0.50 and 2.00 are well within range.

Pre-registered gates (vs DS-N5c=60.97%, fair baseline=81.10%):
  INTERMEDIATE  : mean_acc CI_lo > 0.6097
  CONSERVATIVE  : mean_acc >= 0.76
  AMBITIOUS     : mean_acc >= 0.811
  BREAKTHROUGH  : AMBITIOUS AND mean_energy < 1 nJ/inf
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
    load_surrogate_torch, build_prototypes, hdc_query_normalize,
    get_device, query_surrogate, Q_ELEM,
)


# LOCKED parameters --- DS-N5f
V_G1_BIAS = 0.30
V_G2_BIAS = 0.30
V_d_HIGH  = 2.00
V_d_LOW   = 0.50


def nsram_rates_vd(VG1_batch, VG2_batch, Vd_batch, surr, C_b_F, dt_s, T_steps):
    """T_steps body-state dynamics. Vd_batch is per-neuron (B, N) so we can
    encode the HD bit through drain voltage.
    """
    device = VG1_batch.device
    B, N = VG1_batch.shape
    Vb_min = surr["ax_Vb"][0]
    Vb_max = surr["ax_Vb"][-1]

    rate_accum = torch.zeros(B, N, device=device)
    spike_events = torch.zeros(B, device=device)
    Vb = torch.zeros(B, N, device=device)

    for _ in range(T_steps):
        Vb_c = Vb.clamp(Vb_min, Vb_max)
        I_d, I_ii, I_leak = query_surrogate(surr, VG1_batch, VG2_batch,
                                            Vd_batch, Vb_c)
        Vb = (Vb + dt_s * (I_ii - I_leak) / C_b_F).clamp(Vb_min, Vb_max)
        rate_accum = rate_accum + I_d.abs() / T_steps
        spike_events = spike_events + (
            (I_d.abs() * dt_s) > Q_ELEM).float().sum(dim=1)
    return rate_accum, spike_events


def symmetry_diagnostic(surr, device, T_steps=100, C_b_F=8e-15, dt_s=1e-7):
    """4 (bit, arm) measurements at zero input (V_G1 = V_G1_BIAS).
    bit=+1 pos arm: Vd=HIGH ;  neg arm: Vd=LOW
    bit=-1 pos arm: Vd=LOW  ;  neg arm: Vd=HIGH
    """
    VG1 = torch.tensor([[V_G1_BIAS]], dtype=torch.float32, device=device)
    VG2 = torch.tensor([[V_G2_BIAS]], dtype=torch.float32, device=device)
    def run(vd):
        Vd = torch.tensor([[vd]], dtype=torch.float32, device=device)
        rates, _ = nsram_rates_vd(VG1, VG2, Vd, surr, C_b_F, dt_s, T_steps)
        return float(rates.item())
    I_d_high = run(V_d_HIGH)
    I_d_low  = run(V_d_LOW)
    diff_p1 = I_d_high - I_d_low
    diff_m1 = I_d_low  - I_d_high
    sym = (abs(diff_p1) / max(abs(diff_m1), 1e-30)
           if diff_m1 != 0 else None)
    sep_ratio = I_d_high / max(I_d_low, 1e-30)
    return {
        "V_G1_BIAS": V_G1_BIAS,
        "V_G2_BIAS": V_G2_BIAS,
        "V_d_HIGH":  V_d_HIGH,
        "V_d_LOW":   V_d_LOW,
        "I_d_at_V_d_LOW_A":  I_d_low,
        "I_d_at_V_d_HIGH_A": I_d_high,
        "I_d_bit_plus1_pos_arm_A":  I_d_high,
        "I_d_bit_plus1_neg_arm_A":  I_d_low,
        "I_d_bit_minus1_pos_arm_A": I_d_low,
        "I_d_bit_minus1_neg_arm_A": I_d_high,
        "diff_bit_plus1_A":  diff_p1,
        "diff_bit_minus1_A": diff_m1,
        "symmetry_ratio":    sym,
        "separation_ratio_HIGH_over_LOW": float(sep_ratio),
    }


def run_seed(Xtr, ytr, Xte, yte, surr, device, N, Q, seed, n_classes,
             g_in=0.25, C_b_F=8e-15, dt_s=1e-7, T_steps=100,
             batch_size=64):
    rng = np.random.default_rng(seed)
    F = Xtr.shape[1]
    mins = Xtr.min(axis=0); maxs = Xtr.max(axis=0)
    Xtrq = quantize(Xtr, mins, maxs, Q)
    Xteq = quantize(Xte, mins, maxs, Q)
    t0 = time.time()

    P_pos, L_lev = build_codebooks(F, N, Q, rng)
    Htr_int = encode_samples(Xtrq, P_pos, L_lev)
    Hte_int = encode_samples(Xteq, P_pos, L_lev)
    protos = build_prototypes(Htr_int, ytr, n_classes, N)

    Hte_n = hdc_query_normalize(Hte_int)
    Htr_n = hdc_query_normalize(Htr_int)

    VG1_min = float(surr["ax_VG1"][0].item())
    VG1_max = float(surr["ax_VG1"][-1].item())

    def vd_pair(p):
        # bit=+1 -> pos arm HIGH, neg arm LOW
        # bit=-1 -> pos arm LOW,  neg arm HIGH
        vd_pos = np.where(p > 0, V_d_HIGH, V_d_LOW).astype(np.float32)
        vd_neg = np.where(p > 0, V_d_LOW,  V_d_HIGH).astype(np.float32)
        return vd_pos, vd_neg

    VG2_full = torch.full((1, N), V_G2_BIAS, dtype=torch.float32, device=device)

    def score_set(H_norm, y_true):
        Nset = H_norm.shape[0]
        scores = np.zeros((Nset, n_classes), dtype=np.float32)
        total_spike_events = 0.0
        for c in range(n_classes):
            vd_pos_np, vd_neg_np = vd_pair(protos[c])
            Vd_pos_t = torch.tensor(vd_pos_np, dtype=torch.float32, device=device)
            Vd_neg_t = torch.tensor(vd_neg_np, dtype=torch.float32, device=device)
            for b0 in range(0, Nset, batch_size):
                b1 = min(b0 + batch_size, Nset)
                H_b = torch.tensor(H_norm[b0:b1], dtype=torch.float32, device=device)
                B = b1 - b0
                VG1 = (V_G1_BIAS + g_in * H_b).clamp(VG1_min, VG1_max)
                VG2 = VG2_full.expand(B, N)
                Vd_p = Vd_pos_t.expand(B, N)
                Vd_n = Vd_neg_t.expand(B, N)
                rates_p, spikes_p = nsram_rates_vd(VG1, VG2, Vd_p, surr,
                                                   C_b_F, dt_s, T_steps)
                rates_n, spikes_n = nsram_rates_vd(VG1, VG2, Vd_n, surr,
                                                   C_b_F, dt_s, T_steps)
                rates_eff = rates_p - rates_n
                s = (rates_eff * H_b).sum(dim=1).cpu().numpy()
                scores[b0:b1, c] = s
                total_spike_events += float(spikes_p.sum().item()
                                            + spikes_n.sum().item())
        preds = scores.argmax(axis=1)
        acc = float((preds == y_true).mean())
        avg_events = total_spike_events / max(1, Nset)
        return acc, avg_events

    train_acc, train_ev = score_set(Htr_n, ytr)
    test_acc,  test_ev  = score_set(Hte_n, yte)
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
        "V_G1_BIAS": V_G1_BIAS, "V_G2_BIAS": V_G2_BIAS,
        "V_d_HIGH": V_d_HIGH, "V_d_LOW": V_d_LOW,
        "g_in": g_in,
    }


def gate_for(mean_acc, ci_lo, mean_energy_J, ds_n5c_acc, fair_baseline):
    intermediate = (mean_acc is not None and ci_lo is not None
                    and ci_lo > ds_n5c_acc)
    conservative = (mean_acc is not None and mean_acc >= 0.76)
    ambitious    = (mean_acc is not None and mean_acc >= fair_baseline)
    breakthrough = (ambitious and mean_energy_J is not None
                    and mean_energy_J < 1e-9)
    if breakthrough:      verdict = "BREAKTHROUGH"
    elif ambitious:       verdict = "AMBITIOUS_PASS"
    elif conservative:    verdict = "CONSERVATIVE_PASS"
    elif intermediate:    verdict = "INTERMEDIATE_PASS"
    else:                 verdict = "FAIL"
    return {
        "intermediate_pass_ci_above_DS_N5c": bool(intermediate),
        "conservative_pass_geq_0.76":        bool(conservative),
        "ambitious_pass_geq_fair_baseline":  bool(ambitious),
        "breakthrough_ambitious_and_under_1nJ": bool(breakthrough),
        "verdict": verdict,
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
    p.add_argument("--DS_N5c_acc",    type=float, default=0.6097)
    p.add_argument("--fair_baseline", type=float, default=0.8110)
    p.add_argument("--out",
                   default="results/z292_nsram_hdc_vd_bit/summary.json")
    p.add_argument("--out_dir", default=None, help="alternative: write summary.json in this dir")
    p.add_argument("--vg1", type=float, default=None, help="override V_G1_BIAS")
    p.add_argument("--vg2", type=float, default=None, help="override V_G2_BIAS")
    p.add_argument("--vd_high", type=float, default=None, help="override V_d_HIGH")
    p.add_argument("--vd_low", type=float, default=None, help="override V_d_LOW")
    args = p.parse_args()
    # Apply overrides
    global V_G1_BIAS, V_G2_BIAS, V_d_HIGH, V_d_LOW
    if args.vg1 is not None: V_G1_BIAS = args.vg1
    if args.vg2 is not None: V_G2_BIAS = args.vg2
    if args.vd_high is not None: V_d_HIGH = args.vd_high
    if args.vd_low is not None: V_d_LOW = args.vd_low
    if args.out_dir is not None:
        from pathlib import Path
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        args.out = str(Path(args.out_dir) / "summary.json")
    print(f"[z292] config: V_G1={V_G1_BIAS} V_G2={V_G2_BIAS} V_d_HIGH={V_d_HIGH} V_d_LOW={V_d_LOW}", flush=True)

    device = get_device()
    print(f"[z292] device={device}", flush=True)
    surr = load_surrogate_torch(args.surrogate, device)
    Xtr, ytr, Xte, yte = load_uci_har(args.data_root)
    print(f"[z292] train {Xtr.shape} test {Xte.shape}", flush=True)
    n_classes = int(max(ytr.max(), yte.max())) + 1

    diag = symmetry_diagnostic(surr, device, T_steps=args.T_steps)
    print(f"\n[z292] === V_d-as-bit Symmetry / Separation diagnostic ===",
          flush=True)
    print(f"  V_G1_BIAS={V_G1_BIAS}  V_G2_BIAS={V_G2_BIAS}", flush=True)
    print(f"  V_d_LOW={V_d_LOW}V  V_d_HIGH={V_d_HIGH}V", flush=True)
    print(f"  I_d @ V_d=LOW  ({V_d_LOW:.2f}V): {diag['I_d_at_V_d_LOW_A']:.3e} A",
          flush=True)
    print(f"  I_d @ V_d=HIGH ({V_d_HIGH:.2f}V): {diag['I_d_at_V_d_HIGH_A']:.3e} A",
          flush=True)
    print(f"  separation HIGH/LOW = {diag['separation_ratio_HIGH_over_LOW']:.4f}",
          flush=True)
    print(f"  diff(+1)={diag['diff_bit_plus1_A']:.3e} A   "
          f"diff(-1)={diag['diff_bit_minus1_A']:.3e} A",
          flush=True)
    print(f"  symmetry_ratio = {diag['symmetry_ratio']}", flush=True)

    per_seed = []
    for s in args.seeds:
        try:
            r = run_seed(Xtr, ytr, Xte, yte, surr, device,
                         args.N, args.Q, s, n_classes,
                         T_steps=args.T_steps, batch_size=args.batch_size)
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
        "experiment": "z292_nsram_hdc_vd_bit_uci_har",
        "n_seeds": len(args.seeds),
        "N_bits": int(args.N),
        "Q_levels": int(args.Q),
        "T_steps": int(args.T_steps),
        "V_G1_BIAS": V_G1_BIAS, "V_G2_BIAS": V_G2_BIAS,
        "V_d_HIGH": V_d_HIGH, "V_d_LOW": V_d_LOW,
        "DS_N5c_acc": args.DS_N5c_acc,
        "fair_baseline_acc": args.fair_baseline,
        "symmetry_diagnostic": diag,
        "per_seed": per_seed,
        "acc_per_seed": accs,
        "mean_acc": float(np.mean(accs)) if accs else None,
        "std_acc":  float(np.std(accs))  if accs else None,
        "mean_energy_J_per_inference":
            float(np.mean(energies)) if energies else None,
    }
    if len(accs) >= 2:
        rng = np.random.default_rng(0)
        bs = np.array([rng.choice(accs, len(accs), replace=True).mean()
                       for _ in range(4000)])
        summary["ci95"] = [float(np.quantile(bs, 0.025)),
                           float(np.quantile(bs, 0.975))]
    else:
        summary["ci95"] = [None, None]

    summary["delta_vs_DS_N5c"] = (
        summary["mean_acc"] - args.DS_N5c_acc
        if summary["mean_acc"] is not None else None)
    summary["delta_vs_fair_baseline"] = (
        summary["mean_acc"] - args.fair_baseline
        if summary["mean_acc"] is not None else None)
    summary["gates"] = gate_for(summary["mean_acc"], summary["ci95"][0],
                                summary["mean_energy_J_per_inference"],
                                args.DS_N5c_acc, args.fair_baseline)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[z292] DONE mean_acc={summary['mean_acc']} "
          f"CI95={summary['ci95']} verdict={summary['gates']['verdict']} "
          f"delta_vs_N5c={summary['delta_vs_DS_N5c']} "
          f"delta_vs_baseline={summary['delta_vs_fair_baseline']} "
          f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
