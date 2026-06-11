"""H7 Knockoff-KL probe — the metric GPT-5 proposed in O102.

Question: does the v5 trained model's output distribution genuinely depend on
substrate, or does any substrate-shaped input produce identical outputs?

Knockoff = synthetic substrate matching μ, σ, AR(1), AR(2), and PSD of real per
channel, but otherwise breaking fine-grained alignment. A model that's truly
substrate-conditioned must distinguish REAL from KNOCKOFF in its output
distribution, by MORE than two real windows differ from each other.

Test:
  - Collect M live windows W_real[1..M] (256 samples × 10ch each)
  - For each, build a knockoff W_knock[i] (AR(1)+AR(2) matched, PSD-matched)
  - On a held-out prompt set P_1..P_N, compute next-token logits under:
       L_real_i(p)  = model(p; W_real_i)
       L_knock_i(p) = model(p; W_knock_i)
  - Symmetric KL per prompt:
       D_rk(i, p) = 0.5 (KL(L_real_i || L_knock_i) + KL(L_knock_i || L_real_i))
       D_rr(i, j, p) = same for L_real_i vs L_real_j
  - Compare medians of D_rk vs D_rr across prompts and windows.

Pass: median(D_rk) > 2 × median(D_rr). Real-vs-knockoff must be at least 2× the
within-real spread, otherwise the model is just using substrate as noise.

Fail mode: median(D_rk) ≈ median(D_rr) → model treats substrate as noise; no
substrate-conditioning beyond first-order statistics.

Run requirements: needs the v5 checkpoint and the live substrate ring buffer.
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3, higher_moments
from h7_rooted_lm_v4a import GlobalNorm, SubstrateEncoderV4, K_TOKENS, WIN_LEN, N_CHANNELS, BASE_MODEL, STATS
from h7_embodied_v5 import EmbodiedSmolLM, BASE_MODEL as BM2
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_H7_2026-06-09"
CKPT = ROOT / "results/IDENTITY_EMBODIED_2026-06-10/embodied_v5_ikaros.pt"

N_WINDOWS = 16          # M real windows
N_PROMPTS = 32          # P prompts
PROMPT_LEN = 32
TOP_K_LOGITS = None     # full vocab KL

PROMPTS = [
    "The quick brown fox jumps over",
    "In a hole in the ground there lived a",
    "It was the best of times, it was the worst",
    "All happy families are alike; each unhappy family",
    "Call me Ishmael. Some years ago — never mind",
    "The sky above the port was the color of television",
    "Many years later, as he faced the firing squad",
    "It is a truth universally acknowledged, that a single",
    "Once upon a time, in a land far, far away",
    "To be or not to be, that is the",
    "The first rule of fight club is",
    "When in the course of human events it becomes",
    "I have a dream that one day",
    "Ask not what your country can do for",
    "We hold these truths to be self-evident",
    "Friends, Romans, countrymen, lend me",
    "Four score and seven years ago",
    "Look upon my works, ye Mighty, and",
    "The road goes ever on and on, down from",
    "Beneath the rule of men entirely great",
    "Tell me, O Muse, of that ingenious hero",
    "Happy families are all alike; every unhappy family",
    "Stately, plump Buck Mulligan came from",
    "It was a bright cold day in April and",
    "Lolita, light of my life, fire of",
    "Mother died today. Or maybe",
    "All this happened, more or",
    "Mrs. Dalloway said she would buy",
    "Whether I shall turn out to be the hero",
    "I am an invisible man. No, I am not",
    "Riverrun, past Eve and Adam's, from",
    "If you really want to hear about it,",
]
assert len(PROMPTS) >= N_PROMPTS


def make_knockoff(window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-channel: match μ, σ, AR(1), AR(2). Then PSD-shape via FFT amplitude
    spectrum from real + random phase. Order: AR generates skeleton, PSD warps."""
    out = np.zeros_like(window)
    T = window.shape[0]
    for c in range(window.shape[1]):
        x = window[:, c]
        mu, sg = x.mean(), x.std() + 1e-9
        # AR(2) via least-squares
        if T > 8 and x.std() > 0:
            X = np.stack([x[1:-1], x[:-2]], axis=1)
            y = x[2:]
            try:
                a, *_ = np.linalg.lstsq(X, y - mu, rcond=None)
                phi1, phi2 = float(a[0]), float(a[1])
            except Exception:
                phi1, phi2 = 0.5, 0.0
        else:
            phi1, phi2 = 0.5, 0.0
        # Innovation variance such that stationary var matches σ²
        ar_var = max(1e-9, (1 - phi1**2 - phi2**2 - 2*phi1**2*phi2/(1-phi2)) * sg**2) if abs(phi2) < 1 else sg**2
        eps = rng.normal(0, np.sqrt(ar_var), T)
        y = np.zeros(T); y[0] = x[0] - mu; y[1] = x[1] - mu
        for t in range(2, T):
            y[t] = phi1*y[t-1] + phi2*y[t-2] + eps[t]
        # PSD-shape: take amplitude spectrum from real, randomize phase
        amp_real = np.abs(np.fft.rfft(x - mu))
        rand_phase = np.exp(1j * rng.uniform(-np.pi, np.pi, len(amp_real)))
        rand_phase[0] = 1.0  # DC real
        y_psd = np.fft.irfft(amp_real * rand_phase, n=T)
        # Blend AR shape and PSD shape (50/50)
        y_blend = 0.5 * y + 0.5 * y_psd
        # Renormalize μ, σ to match real
        y_blend = (y_blend - y_blend.mean()) / (y_blend.std() + 1e-9) * sg + mu
        out[:, c] = y_blend
    return out


def sym_kl(p_logits, q_logits):
    """Symmetric KL between two logits tensors (each (V,))."""
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    lp = F.log_softmax(p_logits, dim=-1)
    lq = F.log_softmax(q_logits, dim=-1)
    kl_pq = (p * (lp - lq)).sum(-1)
    kl_qp = (q * (lq - lp)).sum(-1)
    return 0.5 * (kl_pq + kl_qp)


def main():
    rng = np.random.default_rng(11)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, ckpt={CKPT}")
    if not CKPT.exists():
        print(f"FATAL: checkpoint missing: {CKPT}"); sys.exit(1)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = EmbodiedSmolLM().to(device).eval()
    se = SubstrateEncoderV4(d_emb=model.d, K=K_TOKENS).to(device).eval()

    ck = torch.load(CKPT, map_location=device, weights_only=False)
    print(f"loading ckpt: step={ck.get('step')} host={ck.get('host')}")
    model.xattn.load_state_dict(ck["xattn"])
    se.load_state_dict(ck["se"])
    model.sub_pred.load_state_dict(ck["sub_pred"])
    # Also load LoRA weights
    for name, p in model.base.named_parameters():
        if "A" in name or "B" in name:
            pass  # LoRA handled by inject_lora in EmbodiedSmolLM
    # Direct LoRA state load
    lora_state = ck.get("lora", {})
    if lora_state:
        msd = dict(model.named_parameters())
        for k, v in lora_state.items():
            if k in msd: msd[k].data.copy_(v.to(device))

    alphas = model.gate_alphas()
    print(f"gate alphas: {alphas}")

    norm = GlobalNorm(STATS)

    # 1. Collect M real windows live
    print(f"\nCollecting {N_WINDOWS} real substrate windows ({WIN_LEN}×{N_CHANNELS} each)...")
    state = SubstrateStateV3(hz_target=500); state.start()
    real_windows = []
    for i in range(N_WINDOWS):
        time.sleep(0.6)   # > WIN_LEN/500Hz = 512ms, get fresh window
        w = state.latest_window(length=WIN_LEN)
        real_windows.append(w.copy())
        if i % 4 == 0: print(f"  {i+1}/{N_WINDOWS}")
    state.stop()
    real_windows = np.stack(real_windows)  # (M, T, C)
    print(f"real windows shape: {real_windows.shape}")

    # 2. Build knockoffs
    print("Building knockoffs (AR(2) + PSD-matched, phase-randomized)...")
    knock_windows = np.stack([make_knockoff(w, rng) for w in real_windows])

    # Sanity: per-channel μ/σ should match closely
    for c in range(3):
        rm, rs = real_windows[:,:,c].mean(), real_windows[:,:,c].std()
        km, ks = knock_windows[:,:,c].mean(), knock_windows[:,:,c].std()
        print(f"  ch{c}: real μ={rm:+.3e} σ={rs:.3e} | knock μ={km:+.3e} σ={ks:.3e}")

    # 3. Tokenize prompts (last-token logits → next-token distribution)
    print(f"\nTokenizing {N_PROMPTS} prompts...")
    enc = tok(PROMPTS[:N_PROMPTS], return_tensors="pt", padding=True, truncation=True, max_length=PROMPT_LEN).to(device)

    # 4. For each window, get next-token logits at every prompt's last position
    def logits_under(window_batch_np):
        """window_batch_np: (M, T, C) → returns logits (M, N_PROMPTS, V)"""
        out = []
        with torch.no_grad():
            for w in window_batch_np:
                wn = norm(w)
                wt = torch.from_numpy(wn).unsqueeze(0).to(device)  # (1, T, C)
                mom = higher_moments(wn).astype(np.float32)
                mt = torch.from_numpy(mom).unsqueeze(0).to(device)  # (1, C*5)
                S = se(wt, mt)  # (1, K, d)
                S = S.expand(enc["input_ids"].shape[0], -1, -1)
                # set as substrate context
                # EmbodiedSmolLM expects per-call _S; forward sets/clears it
                with torch.no_grad():
                    o = model(enc["input_ids"], substrate_tokens=S)
                lg = o.logits  # (N, L, V)
                # extract last-token logit per row using attention mask
                last_idx = enc["attention_mask"].sum(dim=1) - 1
                rows = torch.arange(lg.shape[0], device=device)
                last_logits = lg[rows, last_idx]  # (N, V)
                out.append(last_logits.cpu())
        return torch.stack(out)  # (M, N, V)

    print("\nForward passes under REAL substrate...")
    L_real = logits_under(real_windows)
    print("Forward passes under KNOCKOFF substrate...")
    L_knock = logits_under(knock_windows)
    print(f"L_real {L_real.shape}, L_knock {L_knock.shape}")

    # 5. Symmetric KL per (window, prompt)
    D_rk = sym_kl(L_real, L_knock)  # (M, N)
    # D_rr: KL between pairs of distinct real windows
    D_rr_list = []
    for i in range(N_WINDOWS):
        for j in range(i+1, N_WINDOWS):
            D_rr_list.append(sym_kl(L_real[i], L_real[j]))
    D_rr = torch.stack(D_rr_list)  # (M*(M-1)/2, N)

    med_rk = D_rk.median().item()
    med_rr = D_rr.median().item()
    q90_rk = D_rk.flatten().quantile(0.9).item()
    q90_rr = D_rr.flatten().quantile(0.9).item()
    mean_rk = D_rk.mean().item()
    mean_rr = D_rr.mean().item()

    ratio = med_rk / max(med_rr, 1e-9)
    verdict = "PASS" if ratio > 2.0 else ("MARGINAL" if ratio > 1.3 else "FAIL")

    print("\n=== KNOCKOFF-KL RESULTS ===")
    print(f"  median KL(real_i || knockoff_i)      = {med_rk:.4e}  (D_rk)")
    print(f"  median KL(real_i || real_j, i≠j)     = {med_rr:.4e}  (D_rr)")
    print(f"  ratio  D_rk / D_rr                   = {ratio:.3f}x")
    print(f"  q90 D_rk={q90_rk:.4e}, q90 D_rr={q90_rr:.4e}")
    print(f"  mean D_rk={mean_rk:.4e}, mean D_rr={mean_rr:.4e}")
    print(f"\n=> VERDICT: {verdict} (pass if ratio > 2.0)")

    out_path = OUT / "knockoff_kl_v5_2026-06-10.json"
    json.dump({
        "ckpt": str(CKPT), "gate_alphas": alphas,
        "n_windows": N_WINDOWS, "n_prompts": N_PROMPTS,
        "median_D_rk": med_rk, "median_D_rr": med_rr, "ratio": ratio,
        "q90_D_rk": q90_rk, "q90_D_rr": q90_rr,
        "mean_D_rk": mean_rk, "mean_D_rr": mean_rr,
        "verdict": verdict,
    }, open(out_path, "w"), indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
