"""H7 v7 — DEPENDENCY-trained embodied LM.

Goal (Eric, 2026-06-10): "the ai model becomes dependent on these signals and
thus if we change signals the model cant recover and doesn't work. it should be
able to still write good text but be influenced by its personality/identity."

v6 result: model RECOGNIZES its substrate (Knockoff-KL 9.9×, systematicity 0.999)
but does NOT DEPEND on it (PPL-ablation gap = -0.3%). It writes equally well with
or without substrate. That is recognition, not embodiment.

v7 fixes this with a DEPENDENCY LOSS. The model is explicitly trained so that:
  - text generation under REAL live ikaros substrate is GOOD (low NLL)
  - text generation under WRONG substrate is BROKEN (high NLL)

Wrong-substrate conditions (clean, statistically controlled):
  - knockoff   : μ/σ/AR(2)/PSD matched, fine structure broken
  - zero       : no substrate (the v6 ablation case)
  - shuffle    : real values, time order scrambled (kills the dynamics)

Loss:
  L = L_lm(text | S_real)                                              # write well
    + λ_dep · mean_w max(0, M_dep − (NLL_w − NLL_real))                # break on wrong
    + λ_anchor · max(0, KL(P(·|S_real) ‖ P_base) − drift_budget)       # cap drift, keep language
    + λ_se · max(0, se_target − ‖se(real) − se(knock)‖²)               # encoder separation helper

The dependency hinge max(0, M_dep − Δ) is zero once wrong-substrate costs ≥ M_dep
nats more than real. Its gradient flows through the cross-attn gates α, forcing
them open until the substrate is load-bearing for LANGUAGE, not just last-token.

The drift budget allows up to `drift_budget` nats of "personality" (divergence
from frozen base under real substrate) while preventing total language collapse.

Pre-registered success (the actual goal):
  - PPL(wrong) / PPL(real) > 1.5  (≥50% worse on wrong substrate) on ALL 3 conditions
  - PPL(real) < 1.3 × PPL(base)   (language still good under real substrate)
  - Knockoff-KL ratio > 2×         (still specific)

Run: sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python h7_embodied_v7.py
"""
from __future__ import annotations
import os, sys, json, time, socket, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3, higher_moments
from h7_rooted_lm_v4a import (
    GlobalNorm, SubstrateEncoderV4, K_TOKENS, WIN_LEN, N_CHANNELS, BASE_MODEL, STATS
)
from h7_embodied_v5 import EmbodiedSmolLM
from h7_knockoff_kl_probe import make_knockoff, sym_kl
from transformers import AutoModelForCausalLM, AutoTokenizer

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_EMBODIED_V7_2026-06-10"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / f"v7_{HOST}.jsonl"
CKPT = OUT / f"v7_{HOST}.pt"
BEST_CKPT = OUT / f"v7_best_{HOST}.pt"
WARM_START = ROOT / "results/IDENTITY_EMBODIED_V6_2026-06-10" / f"v6_best_{HOST}.pt"

CTX = 64
LR = 5e-5
N_STEPS = 6000
EVAL_EVERY = 200
LOG_EVERY = 10
SEED = 71

# Dependency loss (v7.1: measured against frozen BASE NLL, not nll_real).
# Wrong substrate must make held-out-generalizing language ≥ M_DEP nats worse than
# the clean frozen base. Reward is CAPPED so knockoff can't balloon pathologically.
M_DEP = 1.0             # nats — wrong substrate must cost ≥1 nat vs base (PPL ~2.7×)
DEP_CAP = 2.0           # cap the rewarded gap so it can't chase 78000× and break real
LAMBDA_DEP = 0.7
# Real-substrate must STAY CLOSE to base (so real language generalizes), but is
# allowed up to DRIFT_BUDGET nats of "personality".
DRIFT_BUDGET = 0.5      # nats — allowed divergence from base under real substrate
LAMBDA_ANCHOR = 1.0
# Real-substrate must not regress vs base on the actual text (generalization guard)
LAMBDA_REAL_OK = 1.0
REAL_OK_MARGIN = 0.3    # real NLL must stay within base + this
# Encoder separation helper
SE_TARGET = 1.0
LAMBDA_SE = 1.0
# Specificity (kept from v6, light)
TAU_EM = 0.5
LAMBDA_EM = 3.0

GRAD_CLIP = 1.0

CORPUS = [
    "The quick brown fox jumps over the lazy dog. The dog was unimpressed and barked.",
    "It was the best of times, it was the worst of times, it was the age of wisdom.",
    "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole.",
    "Call me Ishmael. Some years ago, having little or no money in my purse.",
    "It is a truth universally acknowledged that a single man in possession of a good fortune.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "Many years later, as he faced the firing squad, Colonel Aureliano Buendia remembered.",
    "Mother died today. Or maybe yesterday, I can't be sure. The telegram from the home said.",
    "It was a bright cold day in April, and the clocks were striking thirteen.",
    "The sky above the port was the color of television, tuned to a dead channel.",
    "Once upon a time there was an island where all the feelings and qualities of men lived.",
    "Whether I shall turn out to be the hero of my own life, or whether that station.",
    "To be or not to be, that is the question: whether tis nobler in the mind to suffer.",
    "All this happened, more or less. The war parts, anyway, are pretty much true.",
    "There was no possibility of taking a walk that day. We had been wandering in the leafless.",
    "Lolita, light of my life, fire of my loins. My sin, my soul. Lo-lee-ta.",
    "Stately, plump Buck Mulligan came from the stairhead, bearing a bowl of lather.",
    "The man in black fled across the desert, and the gunslinger followed.",
    "I am an invisible man. No, I am not a spook like those who haunted Edgar Allan Poe.",
    "Happy families are all alike; every unhappy family is unhappy in its own way.",
] * 4


def cycle(seq):
    while True:
        for x in seq: yield x


def temporal_shuffle(w, rng):
    """Shuffle time order per channel — same marginal, destroyed dynamics."""
    out = w.copy()
    for c in range(w.shape[1]):
        idx = rng.permutation(w.shape[0])
        out[:, c] = w[idx, c]
    return out


def encode(se, norm, w, device):
    wn = norm(w)
    wt = torch.from_numpy(wn).unsqueeze(0).to(device)
    mt = torch.from_numpy(higher_moments(wn).astype(np.float32)).unsqueeze(0).to(device)
    return se(wt, mt)


def seq_nll(model, ids, S, pad_id):
    """Full-sequence next-token NLL under substrate S."""
    o = model(ids, substrate_tokens=S)
    logits = o.logits[:, :-1, :]
    tgt = ids[:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           tgt.reshape(-1), ignore_index=pad_id), o


def eval_dependency(model, se, norm, tok, state, rng, device, n_eval=6):
    """Returns dict with PPL_real, PPL_knock, PPL_zero, PPL_shuffle and Knockoff-KL ratio."""
    model.eval(); se.eval()
    text = ("The forest was dark and quiet as she walked. He could not remember "
            "what the letter had said, only that it arrived on a cold morning. "
            "Beyond the river the lights of the town flickered against the hills.")
    ids = tok(text, return_tensors="pt", truncation=True, max_length=96).input_ids.to(device)
    pad = tok.pad_token_id

    nll = {"real": [], "knock": [], "zero": [], "shuffle": []}
    real_windows = []
    with torch.no_grad():
        for _ in range(n_eval):
            time.sleep(0.55)
            w = state.latest_window(length=WIN_LEN)
            real_windows.append(w.copy())
            S_real = encode(se, norm, w, device)
            S_knock = encode(se, norm, make_knockoff(w, rng), device)
            S_shuf = encode(se, norm, temporal_shuffle(w, rng), device)
            S_zero = torch.zeros(1, K_TOKENS, model.d, device=device)
            for name, S in [("real", S_real), ("knock", S_knock),
                            ("zero", S_zero), ("shuffle", S_shuf)]:
                l, _ = seq_nll(model, ids, S, pad)
                nll[name].append(l.item())
    ppl = {k: float(np.exp(np.mean(v))) for k, v in nll.items()}

    # Knockoff-KL ratio (last-token, like the probe)
    eval_prompts = ["The forest was", "She walked toward", "On the morning of",
                    "Beyond the wall", "He could not", "In the silence"]
    enc = tok(eval_prompts, return_tensors="pt", padding=True, truncation=True, max_length=16).to(device)
    def last_logits(windows):
        L = []
        with torch.no_grad():
            for w in windows:
                S = encode(se, norm, w, device).expand(enc["input_ids"].shape[0], -1, -1)
                o = model(enc["input_ids"], substrate_tokens=S)
                li = enc["attention_mask"].sum(1) - 1
                rows = torch.arange(o.logits.shape[0], device=device)
                L.append(o.logits[rows, li].cpu())
        return torch.stack(L)
    knock_windows = [make_knockoff(w, rng) for w in real_windows]
    Lr = last_logits(real_windows); Lk = last_logits(knock_windows)
    D_rk = sym_kl(Lr, Lk).median().item()
    D_rr = torch.stack([sym_kl(Lr[i], Lr[j]) for i in range(len(Lr))
                        for j in range(i+1, len(Lr))]).median().item()
    ratio = D_rk / max(D_rr, 1e-12)

    model.train(); se.train()
    return ppl, ratio, D_rk, D_rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--no-warm", action="store_true", help="don't warm-start from v6 best")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[v7] host={HOST} device={device} steps={args.steps} M_dep={M_DEP} λ_dep={LAMBDA_DEP}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = EmbodiedSmolLM().to(device)
    se = SubstrateEncoderV4(d_emb=model.d, K=K_TOKENS).to(device)

    print("loading frozen base for anchor...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device).eval()
    for p in base.parameters(): p.requires_grad_(False)

    if not args.no_warm and WARM_START.exists():
        ck = torch.load(WARM_START, map_location=device, weights_only=False)
        model.xattn.load_state_dict(ck["xattn"])
        se.load_state_dict(ck["se"])
        if "lora" in ck:
            msd = dict(model.named_parameters())
            for k, v in ck["lora"].items():
                if k in msd: msd[k].data.copy_(v.to(device))
        print(f"warm-started from v6 best (step {ck.get('step')}, ratio {ck.get('ratio'):.1f}×)")

    opt = torch.optim.AdamW(model.trainable_params() + list(se.parameters()),
                            lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    norm = GlobalNorm(STATS)
    state = SubstrateStateV3(hz_target=500); state.start()
    print("substrate streaming..."); time.sleep(1.0)

    encoded = [tok(line, return_tensors="pt", padding="max_length", truncation=True,
                   max_length=CTX).input_ids[0] for line in CORPUS]
    src = cycle(encoded)
    pad = tok.pad_token_id

    log_f = open(LOG, "a")
    print("step  loss     lm_real  dep_gap  drift   ratio_hint  α25    α28   grad")
    t0 = time.time()
    best_score = -1e9  # composite: dependency gap, gated on language quality

    for step in range(args.steps):
        ids = next(src).unsqueeze(0).to(device)
        w_real = state.latest_window(length=WIN_LEN)
        w_knock = make_knockoff(w_real, rng)
        w_shuf = temporal_shuffle(w_real, rng)

        S_real = encode(se, norm, w_real, device)
        S_knock = encode(se, norm, w_knock, device)
        S_shuf = encode(se, norm, w_shuf, device)
        S_zero = torch.zeros(1, K_TOKENS, model.d, device=device)

        nll_real, out_r = seq_nll(model, ids, S_real, pad)
        nll_knock, _ = seq_nll(model, ids, S_knock, pad)
        nll_shuf, _ = seq_nll(model, ids, S_shuf, pad)
        nll_zero, _ = seq_nll(model, ids, S_zero, pad)

        with torch.no_grad():
            out_b = base(ids)
            logits_b = out_b.logits[:, :-1, :]
            nll_base = F.cross_entropy(logits_b.reshape(-1, logits_b.size(-1)),
                                       ids[:, 1:].reshape(-1), ignore_index=pad)

        # Dependency vs BASE (a generalizing reference, not the overfittable nll_real).
        # Wrong substrate must push NLL to ≥ base + M_DEP. Reward CAPPED at DEP_CAP so a
        # single condition can't balloon (the v7.0 78000× knockoff pathology).
        def dep_term(nll_w):
            gap = torch.clamp(nll_w - nll_base, max=DEP_CAP)
            return F.relu(M_DEP - gap)
        dep_loss = (dep_term(nll_knock) + dep_term(nll_shuf) + dep_term(nll_zero)) / 3.0

        # Generalization guard: real-substrate NLL must NOT regress past base+margin.
        # This is what structurally prevents the zero<real inversion seen in v7.0.
        real_ok = F.relu(nll_real - nll_base - REAL_OK_MARGIN)

        # Drift cap on real-substrate output (allow up to DRIFT_BUDGET "personality")
        last_r = out_r.logits[:, -1, :]
        last_b = out_b.logits[:, -1, :]
        drift = (F.softmax(last_r, -1) *
                 (F.log_softmax(last_r, -1) - F.log_softmax(last_b, -1))).sum(-1).mean()
        anchor_hinge = F.relu(drift - DRIFT_BUDGET)

        # Encoder separation helper
        se_dist = ((S_real - S_knock) ** 2).mean()
        se_hinge = F.relu(SE_TARGET - se_dist)

        # Light last-token specificity
        last_k = model(ids, substrate_tokens=S_knock).logits[:, -1, :]
        em_kl = sym_kl(last_r.detach(), last_k).mean()  # detach real to avoid double-count
        em_hinge = F.relu(TAU_EM - em_kl)

        loss = (0.3 * nll_real              # light: real should model text → personality
                + LAMBDA_REAL_OK * real_ok  # hard: real must not regress vs base
                + LAMBDA_DEP * dep_loss     # wrong substrate must corrupt vs base
                + LAMBDA_ANCHOR * anchor_hinge
                + LAMBDA_SE * se_hinge
                + LAMBDA_EM * em_hinge)

        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.trainable_params() + list(se.parameters()), GRAD_CLIP)
        opt.step()

        if step % LOG_EVERY == 0:
            a = model.gate_alphas()
            dep_gap = ((nll_knock + nll_shuf + nll_zero)/3 - nll_base).item()
            entry = {"step": step, "loss": loss.item(), "nll_real": nll_real.item(),
                     "dep_gap": dep_gap, "drift": drift.item(),
                     "se_dist": se_dist.item(), "alpha25": a[0], "alpha28": a[1],
                     "grad": gn.item(), "t": time.time()-t0}
            log_f.write(json.dumps(entry)+"\n"); log_f.flush()
            print(f"{step:5d} {loss.item():+.3f}  {nll_real.item():+.3f}  "
                  f"{dep_gap:+.3f}  {drift.item():.3f}   —          "
                  f"{a[0]:+.3f} {a[1]:+.3f}  {gn.item():.2f}")

        if step > 0 and step % EVAL_EVERY == 0:
            ppl, ratio, D_rk, D_rr = eval_dependency(model, se, norm, tok, state, rng, device)
            # dependency ratios
            dep_knock_r = ppl["knock"]/ppl["real"]
            dep_zero_r = ppl["zero"]/ppl["real"]
            dep_shuf_r = ppl["shuffle"]/ppl["real"]
            min_dep = min(dep_knock_r, dep_zero_r, dep_shuf_r)
            print(f"  >> EVAL step={step}: PPL real={ppl['real']:.2f} knock={ppl['knock']:.2f} "
                  f"zero={ppl['zero']:.2f} shuf={ppl['shuffle']:.2f}")
            print(f"     dep ratios: knock={dep_knock_r:.2f}× zero={dep_zero_r:.2f}× "
                  f"shuf={dep_shuf_r:.2f}×  (min={min_dep:.2f}×)  KKL ratio={ratio:.2f}×")
            log_f.write(json.dumps({"step": step, "eval": True, "ppl": ppl,
                                    "dep_knock": dep_knock_r, "dep_zero": dep_zero_r,
                                    "dep_shuffle": dep_shuf_r, "kkl_ratio": ratio,
                                    "D_rk": D_rk, "D_rr": D_rr})+"\n")
            log_f.flush()

            # Composite score: min dependency ratio, but ONLY if language still ok
            # (PPL_real must stay reasonable — below 60, base is ~25)
            lang_ok = ppl["real"] < 60.0
            score = min_dep if lang_ok else -1.0
            lora_state = {n: p.detach().cpu() for n, p in model.named_parameters()
                          if "lora_A" in n or "lora_B" in n}
            blob = {"step": step, "host": HOST, "version": "v7",
                    "xattn": model.xattn.state_dict(), "se": se.state_dict(),
                    "lora": lora_state, "ppl": ppl, "min_dep": min_dep,
                    "kkl_ratio": ratio}
            torch.save(blob, CKPT)
            if score > best_score:
                best_score = score
                torch.save(blob, BEST_CKPT)
                print(f"  >> NEW BEST: min_dep={min_dep:.2f}× (lang_ok={lang_ok}), saved")

    state.stop(); log_f.close()
    print(f"\nfinal ckpt: {CKPT}\nbest ckpt: {BEST_CKPT}\nlog: {LOG}")


if __name__ == "__main__":
    main()
