"""z293: NS-RAM HDC envelope sweep — Phase 4B.1/4B.2/4B.3.

Forks z292 (DS-N5f V_d-as-bit) and adds:
  * Configurable N (bits per encoding)
  * Configurable V_d_HIGH, V_d_LOW, V_G1, V_G2
  * Configurable sigma_noise (Gaussian noise added to FEATURE vector
    (post-standardization, pre-quantize)) — models sensor/input noise
  * Single-cell, multi-seed run, writes summary.json under --out_dir

Pre-registered gates (same as z292 V_d-as-bit, but applied per cell):
  INTERMEDIATE  : CI_lo > DS_N5c (0.6097)
  CONSERVATIVE  : mean_acc >= 0.76
  AMBITIOUS     : mean_acc >= fair_baseline (0.811)
  BREAKTHROUGH  : AMBITIOUS AND mean_energy < 1 nJ/inf

NO-CHEAT: all overrides logged; surrogate path identical to z292.
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


def nsram_rates_vd(VG1_batch, VG2_batch, Vd_batch, surr, C_b_F, dt_s, T_steps):
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


def symmetry_diagnostic(surr, device, vg1, vg2, vd_high, vd_low,
                        T_steps=100, C_b_F=8e-15, dt_s=1e-7):
    VG1 = torch.tensor([[vg1]], dtype=torch.float32, device=device)
    VG2 = torch.tensor([[vg2]], dtype=torch.float32, device=device)
    def run(vd):
        Vd = torch.tensor([[vd]], dtype=torch.float32, device=device)
        rates, _ = nsram_rates_vd(VG1, VG2, Vd, surr, C_b_F, dt_s, T_steps)
        return float(rates.item())
    I_d_high = run(vd_high)
    I_d_low  = run(vd_low)
    diff_p1 = I_d_high - I_d_low
    diff_m1 = I_d_low  - I_d_high
    sym = (abs(diff_p1) / max(abs(diff_m1), 1e-30)
           if diff_m1 != 0 else None)
    sep_ratio = I_d_high / max(I_d_low, 1e-30)
    return {
        "V_G1_BIAS": vg1, "V_G2_BIAS": vg2,
        "V_d_HIGH": vd_high, "V_d_LOW": vd_low,
        "I_d_at_V_d_LOW_A":  I_d_low,
        "I_d_at_V_d_HIGH_A": I_d_high,
        "diff_bit_plus1_A":  diff_p1,
        "diff_bit_minus1_A": diff_m1,
        "symmetry_ratio":    sym,
        "separation_ratio_HIGH_over_LOW": float(sep_ratio),
    }


def run_seed(Xtr, ytr, Xte, yte, surr, device, N, Q, seed, n_classes,
             vg1, vg2, vd_high, vd_low, sigma_noise,
             g_in=0.25, C_b_F=8e-15, dt_s=1e-7, T_steps=100,
             batch_size=64):
    rng = np.random.default_rng(seed)
    F = Xtr.shape[1]
    mins = Xtr.min(axis=0); maxs = Xtr.max(axis=0)

    # Inject feature-space Gaussian noise (post-standardization, pre-quantize).
    # Span-normalized so sigma is in standardized [0,1] feature units.
    if sigma_noise > 0:
        span = np.where((maxs - mins) < 1e-9, 1.0, (maxs - mins))
        Xtr_noisy = Xtr + rng.normal(0.0, sigma_noise, size=Xtr.shape).astype(np.float32) * span
        Xte_noisy = Xte + rng.normal(0.0, sigma_noise, size=Xte.shape).astype(np.float32) * span
    else:
        Xtr_noisy = Xtr
        Xte_noisy = Xte

    Xtrq = quantize(Xtr_noisy, mins, maxs, Q)
    Xteq = quantize(Xte_noisy, mins, maxs, Q)
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
        vd_pos = np.where(p > 0, vd_high, vd_low).astype(np.float32)
        vd_neg = np.where(p > 0, vd_low,  vd_high).astype(np.float32)
        return vd_pos, vd_neg

    VG2_full = torch.full((1, N), vg2, dtype=torch.float32, device=device)

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
                VG1 = (vg1 + g_in * H_b).clamp(VG1_min, VG1_max)
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
        "V_G1_BIAS": vg1, "V_G2_BIAS": vg2,
        "V_d_HIGH": vd_high, "V_d_LOW": vd_low,
        "sigma_noise": float(sigma_noise),
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
    p.add_argument("--vg1", type=float, default=0.30)
    p.add_argument("--vg2", type=float, default=0.30)
    p.add_argument("--vd_high", type=float, default=2.00)
    p.add_argument("--vd_low",  type=float, default=0.50)
    p.add_argument("--sigma_noise", type=float, default=0.0)
    p.add_argument("--DS_N5c_acc",    type=float, default=0.6097)
    p.add_argument("--fair_baseline", type=float, default=0.8110)
    p.add_argument("--out", default=None)
    p.add_argument("--out_dir", default=None,
                   help="write summary.json here; ignored if --out set")
    args = p.parse_args()

    if args.out_dir is not None and args.out is None:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        args.out = str(Path(args.out_dir) / "summary.json")
    if args.out is None:
        args.out = "results/z293_envelope/_default/summary.json"

    print(f"[z293] cell: N={args.N} vd_HIGH={args.vd_high} vd_LOW={args.vd_low} "
          f"vg1={args.vg1} vg2={args.vg2} sigma_noise={args.sigma_noise} "
          f"seeds={args.seeds}", flush=True)

    device = get_device()
    print(f"[z293] device={device}", flush=True)
    surr = load_surrogate_torch(args.surrogate, device)
    Xtr, ytr, Xte, yte = load_uci_har(args.data_root)
    print(f"[z293] train {Xtr.shape} test {Xte.shape}", flush=True)
    n_classes = int(max(ytr.max(), yte.max())) + 1

    diag = symmetry_diagnostic(surr, device, args.vg1, args.vg2,
                               args.vd_high, args.vd_low, T_steps=args.T_steps)
    print(f"[z293] sep HIGH/LOW = {diag['separation_ratio_HIGH_over_LOW']:.4f}",
          flush=True)
    print(f"[z293] diff(+1)={diag['diff_bit_plus1_A']:.3e} A   "
          f"diff(-1)={diag['diff_bit_minus1_A']:.3e} A", flush=True)

    per_seed = []
    for s in args.seeds:
        try:
            r = run_seed(Xtr, ytr, Xte, yte, surr, device,
                         args.N, args.Q, s, n_classes,
                         args.vg1, args.vg2, args.vd_high, args.vd_low,
                         args.sigma_noise,
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
        "experiment": "z293_hdc_envelope_sweep",
        "cell": {
            "N": int(args.N), "Q": int(args.Q),
            "vg1": args.vg1, "vg2": args.vg2,
            "vd_high": args.vd_high, "vd_low": args.vd_low,
            "sigma_noise": float(args.sigma_noise),
        },
        "n_seeds": len(args.seeds),
        "seeds": list(args.seeds),
        "T_steps": int(args.T_steps),
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

    summary["gates"] = gate_for(summary["mean_acc"], summary["ci95"][0],
                                summary["mean_energy_J_per_inference"],
                                args.DS_N5c_acc, args.fair_baseline)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[z293] DONE mean_acc={summary['mean_acc']} "
          f"CI95={summary['ci95']} verdict={summary['gates']['verdict']} "
          f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
