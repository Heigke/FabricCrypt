"""H7 full probe — alla embodiment-mått på ett sparat checkpoint.

Kör mot v6_best_{host}.pt eller annan v6-ckpt. Producerar JSON-rapport.

Metriker:
  M1. Knockoff-KL ratio (D_rk / D_rr)              — primärt pre-registrerat mått
  M2. Substrate-ablation PPL gap                   — funkar modellen sämre utan kropp?
  M3. Per-prompt rank-korrelation                  — är effekten systematisk eller brus?
  M5. Per-channel ablation                         — vilka kanaler är kausalt dominanta?
  M6. Behavioral entropy under substrate sweep     — hur mycket rör sig modellen?

Cross-host (M1.transplant) och temporal-replay (M4) kräver extern data, kör separat.
"""
from __future__ import annotations
import os, sys, json, time, socket, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3, higher_moments
from h7_rooted_lm_v4a import GlobalNorm, SubstrateEncoderV4, K_TOKENS, WIN_LEN, N_CHANNELS, BASE_MODEL, STATS
from h7_embodied_v5 import EmbodiedSmolLM
from h7_knockoff_kl_probe import make_knockoff, sym_kl, PROMPTS
from transformers import AutoModelForCausalLM, AutoTokenizer

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_H7_2026-06-09"
OUT.mkdir(parents=True, exist_ok=True)

N_WINDOWS = 16
N_PROMPTS = 32
PROMPT_LEN = 32
SEED = 41


def load_ckpt_into(ckpt_path, device):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = EmbodiedSmolLM().to(device).eval()
    se = SubstrateEncoderV4(d_emb=model.d, K=K_TOKENS).to(device).eval()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"loaded ckpt: step={ck.get('step')}, host={ck.get('host')}, ratio_at_save={ck.get('ratio'):.3f}")
    model.xattn.load_state_dict(ck["xattn"])
    se.load_state_dict(ck["se"])
    lora_state = ck.get("lora", {})
    if lora_state:
        msd = dict(model.named_parameters())
        for k, v in lora_state.items():
            if k in msd: msd[k].data.copy_(v.to(device))
    return model, se, tok, ck


def encode_window(w, se, norm, device):
    wn = norm(w)
    wt = torch.from_numpy(wn).unsqueeze(0).to(device)
    mt = torch.from_numpy(higher_moments(wn).astype(np.float32)).unsqueeze(0).to(device)
    return se(wt, mt)


def logits_under(model, se, norm, enc, windows, device, batch_S=None):
    """Returns (M, N, V) of last-token logits, one per window × prompt."""
    L = []
    with torch.no_grad():
        for w in windows:
            S = encode_window(w, se, norm, device) if w is not None else batch_S
            S = S.expand(enc["input_ids"].shape[0], -1, -1)
            o = model(enc["input_ids"], substrate_tokens=S)
            last_idx = enc["attention_mask"].sum(dim=1) - 1
            rows = torch.arange(o.logits.shape[0], device=device)
            L.append(o.logits[rows, last_idx].cpu())
    return torch.stack(L)


def metric_M1_knockoff_kl(model, se, norm, tok, state, rng, device):
    """M1: Knockoff-KL ratio."""
    print("\n[M1] Knockoff-KL ratio")
    enc = tok(PROMPTS[:N_PROMPTS], return_tensors="pt", padding=True, truncation=True,
              max_length=PROMPT_LEN).to(device)
    real_windows = []
    for _ in range(N_WINDOWS):
        time.sleep(0.6); real_windows.append(state.latest_window(length=WIN_LEN).copy())
    knock_windows = [make_knockoff(w, rng) for w in real_windows]

    L_real = logits_under(model, se, norm, enc, real_windows, device)
    L_knock = logits_under(model, se, norm, enc, knock_windows, device)
    D_rk = sym_kl(L_real, L_knock)
    D_rr_list = [sym_kl(L_real[i], L_real[j])
                 for i in range(N_WINDOWS) for j in range(i+1, N_WINDOWS)]
    D_rr = torch.stack(D_rr_list)
    med_rk = D_rk.median().item(); med_rr = D_rr.median().item()
    ratio = med_rk / max(med_rr, 1e-12)
    print(f"  D_rk={med_rk:.3e}, D_rr={med_rr:.3e}, ratio={ratio:.3f}×")
    return {"D_rk": med_rk, "D_rr": med_rr, "ratio": ratio,
            "L_real": L_real, "L_knock": L_knock,
            "real_windows": real_windows, "knock_windows": knock_windows, "enc": enc}


def metric_M2_ablation_ppl_gap(model, se, norm, tok, real_windows, device):
    """M2: PPL on language under real vs zero substrate."""
    print("\n[M2] Substrate ablation PPL gap")
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "She walked slowly toward the door and listened. "
        "On the morning of his departure he wrote a letter. "
        "Beyond the wall lay a wide field of yellow flowers. "
        "He could not quite remember what she had said last."
    )
    ids = tok(text, return_tensors="pt", truncation=True, max_length=128).input_ids.to(device)

    def ppl_under(windows_or_none):
        nlls = []
        with torch.no_grad():
            for w in windows_or_none:
                if w is None:
                    # Zero substrate — S=0 of the right shape
                    S0 = torch.zeros(1, K_TOKENS, model.d, device=device)
                    S = S0
                else:
                    S = encode_window(w, se, norm, device)
                o = model(ids, substrate_tokens=S)
                logits = o.logits[:, :-1, :]
                tgt = ids[:, 1:]
                nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                      tgt.reshape(-1)).item()
                nlls.append(nll)
        return float(np.mean(nlls)), float(np.exp(np.mean(nlls)))

    nll_r, ppl_r = ppl_under(real_windows[:8])
    nll_0, ppl_0 = ppl_under([None] * 8)
    gap = ppl_0 - ppl_r
    rel = (ppl_0 / ppl_r - 1) * 100
    print(f"  PPL(real)={ppl_r:.3f}, PPL(zero)={ppl_0:.3f}, gap=+{gap:.3f} ({rel:+.1f}%)")
    return {"ppl_real": ppl_r, "ppl_zero": ppl_0, "gap": gap, "pct": rel}


def metric_M3_rank_correlation(L_real, L_knock):
    """M3: Per-prompt rank correlation across windows. If the per-prompt KL
    ordering is consistent across window-pairs, the effect is systematic."""
    print("\n[M3] Per-prompt rank correlation (systematicity)")
    # Per-prompt mean D_rk across windows
    per_prompt_rk = sym_kl(L_real, L_knock).mean(0).numpy()  # (N_PROMPTS,)
    # Compare two halves of windows for stability
    half = L_real.shape[0] // 2
    pp_first = sym_kl(L_real[:half], L_knock[:half]).mean(0).numpy()
    pp_second = sym_kl(L_real[half:], L_knock[half:]).mean(0).numpy()
    # Spearman correlation between halves
    rho = np.corrcoef(np.argsort(np.argsort(pp_first)),
                      np.argsort(np.argsort(pp_second)))[0, 1]
    print(f"  Spearman ρ(half1 prompts, half2 prompts) = {rho:.3f}")
    print(f"  per-prompt mean D_rk range: [{per_prompt_rk.min():.3e}, {per_prompt_rk.max():.3e}]")
    return {"spearman_rho": float(rho),
            "per_prompt_min": float(per_prompt_rk.min()),
            "per_prompt_max": float(per_prompt_rk.max())}


def metric_M5_per_channel_ablation(model, se, norm, tok, real_windows, enc, L_real, device):
    """M5: ablate each of the 10 channels, measure KL vs full-real."""
    print("\n[M5] Per-channel ablation — which channels matter?")
    ch_names = ["C07_xtal","C09_pm1","C20_lat_x","C20_logtl","C11_drift",
                "C05_e0_rt","C06_fast","C09_pm3","C09_pm5","C20_lat_e"]
    results = {}
    for ci, cn in enumerate(ch_names):
        # Ablate channel ci by replacing with its mean
        ablated = []
        for w in real_windows[:8]:
            wa = w.copy()
            wa[:, ci] = wa[:, ci].mean()
            ablated.append(wa)
        L_abl = logits_under(model, se, norm, enc, ablated, device)
        D = sym_kl(L_real[:8], L_abl).median().item()
        results[cn] = D
        print(f"  {cn:14s}  median KL(real || ablate)={D:.4e}")
    # Rank by impact
    ranked = sorted(results.items(), key=lambda kv: -kv[1])
    print(f"  TOP-3 impact: {ranked[:3]}")
    return {"per_channel_kl": results, "ranked": ranked}


def metric_M6_substrate_entropy_sweep(model, se, norm, tok, enc, real_windows, device):
    """M6: How much does the output distribution move per unit substrate change?
    We take pairs of real windows and compute output-KL vs input-distance.
    A substrate-rooted model should show output-KL proportional to substrate-distance."""
    print("\n[M6] Substrate sweep — output sensitivity to substrate distance")
    pairs = []
    for i in range(min(8, len(real_windows))):
        for j in range(i+1, min(8, len(real_windows))):
            w_i = norm(real_windows[i]); w_j = norm(real_windows[j])
            sub_dist = float(((w_i - w_j) ** 2).mean())
            pairs.append((i, j, sub_dist))
    L = logits_under(model, se, norm, enc, real_windows[:8], device)
    out_kl_per_pair = []
    sub_dist_per_pair = []
    for i, j, sd in pairs:
        ok = sym_kl(L[i], L[j]).median().item()
        out_kl_per_pair.append(ok)
        sub_dist_per_pair.append(sd)
    sub_dist_arr = np.array(sub_dist_per_pair)
    out_kl_arr = np.array(out_kl_per_pair)
    corr = float(np.corrcoef(sub_dist_arr, out_kl_arr)[0, 1])
    print(f"  Pearson corr(substrate_dist, output_KL) = {corr:.3f}")
    print(f"  if positive, model output scales with substrate variation")
    return {"corr_substrate_output": corr,
            "sub_dist_range": [float(sub_dist_arr.min()), float(sub_dist_arr.max())],
            "out_kl_range": [float(out_kl_arr.min()), float(out_kl_arr.max())]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "results/IDENTITY_EMBODIED_V6_2026-06-10" / f"v6_best_{HOST}.pt"))
    ap.add_argument("--out", default=str(OUT / "full_probe_2026-06-10.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, ckpt={args.ckpt}")

    if not Path(args.ckpt).exists():
        print(f"FATAL: ckpt not found: {args.ckpt}"); sys.exit(1)

    model, se, tok, ck = load_ckpt_into(args.ckpt, device)
    norm = GlobalNorm(STATS)
    state = SubstrateStateV3(hz_target=500); state.start()
    print("substrate streaming, warmup 2s...")
    time.sleep(2.0)

    m1 = metric_M1_knockoff_kl(model, se, norm, tok, state, rng, device)
    m2 = metric_M2_ablation_ppl_gap(model, se, norm, tok, m1["real_windows"], device)
    m3 = metric_M3_rank_correlation(m1["L_real"], m1["L_knock"])
    m5 = metric_M5_per_channel_ablation(model, se, norm, tok,
                                         m1["real_windows"], m1["enc"], m1["L_real"], device)
    m6 = metric_M6_substrate_entropy_sweep(model, se, norm, tok, m1["enc"],
                                             m1["real_windows"], device)

    state.stop()

    report = {
        "ckpt": args.ckpt, "ckpt_step": ck.get("step"),
        "ckpt_ratio_at_save": ck.get("ratio"),
        "M1_knockoff_kl": {k: v for k, v in m1.items()
                           if k not in ["L_real","L_knock","real_windows","knock_windows","enc"]},
        "M2_ablation_ppl_gap": m2,
        "M3_rank_correlation": m3,
        "M5_per_channel_ablation": m5,
        "M6_substrate_sweep": m6,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nsaved {args.out}")

    print("\n=== SUMMARY ===")
    print(f"  M1 ratio:               {m1['ratio']:.3f}×  (pass if > 2.0)")
    print(f"  M2 PPL ablation gap:    {m2['pct']:+.1f}%   (positive = model needs substrate)")
    print(f"  M3 systematicity ρ:     {m3['spearman_rho']:+.3f}  (close to 1 = systematic)")
    print(f"  M5 top channels:        {m5['ranked'][0][0]}, {m5['ranked'][1][0]}, {m5['ranked'][2][0]}")
    print(f"  M6 sub→out corr:        {m6['corr_substrate_output']:+.3f}  (positive = substrate-responsive)")


if __name__ == "__main__":
    main()
