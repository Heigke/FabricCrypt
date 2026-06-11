"""H7 v6 — Knockoff-KL-driven embodied LM.

Plan from CLOSED_LOOP_VERDICT and KNOCKOFF_KL_V5_VERDICT:

  L_total = L_lm(text | S_real)                                  # keep language
          + λ_em · max(0, τ_em − KL(P(·|S_real) || P(·|S_knock))) # force substrate matters
          + γ · KL(P(·|S_real) || P_base)                         # anchor to base

Pre-registered kill: if at step >= 2000, Knockoff-KL ratio < 2.0× — embodiment
in this architecture class is FALSIFIED.

The Knockoff-KL hinge actively pushes the gates open: when the model output is
near-identical under real vs knockoff substrate, the hinge term gives a large
gradient that opens α. The anchor keeps language from collapsing into a
substrate-only response by tying the real-substrate output to the frozen base.

Each training step uses 3 forward passes:
  1. P_real  = model(ids; S_real)        — trained
  2. P_knock = model(ids; S_knock)       — trained (same params, different S)
  3. P_base  = base_model(ids)           — frozen, cached once per ids

Text: tiny live corpus from a few classic openings (no internet, repeatable),
sliced into 64-token contexts. The point is not language modeling SOTA — it's
to keep the LM grounded while substrate gates open.

Run: sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python h7_embodied_v6.py
"""
from __future__ import annotations
import os, sys, json, time, socket, copy, argparse
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
from h7_embodied_v5 import EmbodiedSmolLM, inject_lora, LORA_RANK
from h7_knockoff_kl_probe import make_knockoff, sym_kl
from transformers import AutoModelForCausalLM, AutoTokenizer

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_EMBODIED_V6_2026-06-10"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / f"v6_{HOST}.jsonl"
CKPT = OUT / f"v6_{HOST}.pt"
BEST_CKPT = OUT / f"v6_best_{HOST}.pt"

# Training hparams
CTX = 64
LR = 5e-5
LAMBDA_EM = 30.0         # weight on Knockoff-KL hinge
TAU_EM = 0.5             # nats — push real-vs-knockoff KL above this
GAMMA_ANCHOR = 1.0       # weight on KL(real || base) — stronger to prevent v6 collapse
LAMBDA_SE_CONTRAST = 5.0 # encoder contrastive: ||se(real) - se(knock)||² should be LARGE
SE_CONTRAST_TARGET = 1.0 # target squared distance (encoder output is d_emb-dim)
GRAD_CLIP = 1.0
N_STEPS = 4000
EVAL_EVERY = 200
LOG_EVERY = 10
KILL_AT_STEP = 2000
KILL_RATIO = 2.0

# Eval hparams (smaller than full probe to keep eval cheap)
EVAL_WINDOWS = 8
EVAL_PROMPTS = 16
SEED = 23

# Tiny training corpus — long enough for many 64-token slices
CORPUS = [
    "The quick brown fox jumps over the lazy dog. The dog was unimpressed and barked.",
    "It was the best of times, it was the worst of times, it was the age of wisdom.",
    "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole.",
    "Call me Ishmael. Some years ago, having little or no money in my purse.",
    "It is a truth universally acknowledged that a single man in possession of a good fortune.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "Many years later, as he faced the firing squad, Colonel Aureliano Buendía remembered.",
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
] * 4  # 80 lines


def cycle(seq):
    while True:
        for x in seq: yield x


def eval_knockoff_kl(model, se, norm, tok, state, rng, device, n_windows=EVAL_WINDOWS, n_prompts=EVAL_PROMPTS):
    """Quick Knockoff-KL eval — returns (median_rk, median_rr, ratio)."""
    model.eval(); se.eval()
    eval_prompts = [
        "The forest was dark and",
        "She walked slowly toward the",
        "On the morning of his",
        "It came as no surprise that",
        "Even after the rain stopped",
        "Beyond the wall lay a",
        "He could not quite remember",
        "In the silence she heard",
        "The old man looked up",
        "Three days later they found",
        "A single light burned in",
        "When the door finally opened",
        "Across the river the lights",
        "She knew without being told",
        "By the time he returned",
        "Underneath the floorboards there was",
    ][:n_prompts]
    enc = tok(eval_prompts, return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)

    real_windows, knock_windows = [], []
    for _ in range(n_windows):
        time.sleep(0.6)
        w = state.latest_window(length=WIN_LEN)
        real_windows.append(w.copy())
        knock_windows.append(make_knockoff(w, rng))

    def logits_under(windows):
        L = []
        with torch.no_grad():
            for w in windows:
                wn = norm(w)
                wt = torch.from_numpy(wn).unsqueeze(0).to(device)
                mt = torch.from_numpy(higher_moments(wn).astype(np.float32)).unsqueeze(0).to(device)
                S = se(wt, mt).expand(enc["input_ids"].shape[0], -1, -1)
                o = model(enc["input_ids"], substrate_tokens=S)
                last_idx = enc["attention_mask"].sum(dim=1) - 1
                rows = torch.arange(o.logits.shape[0], device=device)
                L.append(o.logits[rows, last_idx].cpu())
        return torch.stack(L)

    L_real = logits_under(real_windows)
    L_knock = logits_under(knock_windows)
    D_rk = sym_kl(L_real, L_knock)
    D_rr_list = []
    for i in range(n_windows):
        for j in range(i+1, n_windows):
            D_rr_list.append(sym_kl(L_real[i], L_real[j]))
    D_rr = torch.stack(D_rr_list)
    med_rk = D_rk.median().item(); med_rr = D_rr.median().item()
    model.train(); se.train()
    return med_rk, med_rr, (med_rk / max(med_rr, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--lambda_em", type=float, default=LAMBDA_EM)
    ap.add_argument("--tau", type=float, default=TAU_EM)
    ap.add_argument("--gamma", type=float, default=GAMMA_ANCHOR)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[v6] host={HOST} device={device} steps={args.steps} τ={args.tau} λ={args.lambda_em} γ={args.gamma}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Trainable embodied model
    model = EmbodiedSmolLM().to(device)
    se = SubstrateEncoderV4(d_emb=model.d, K=K_TOKENS).to(device)

    # Frozen base for KL anchor
    print("loading frozen base for anchor...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device).eval()
    for p in base.parameters(): p.requires_grad_(False)

    if args.resume and CKPT.exists():
        ck = torch.load(CKPT, map_location=device, weights_only=False)
        model.xattn.load_state_dict(ck["xattn"])
        se.load_state_dict(ck["se"])
        if "lora" in ck:
            msd = dict(model.named_parameters())
            for k, v in ck["lora"].items():
                if k in msd: msd[k].data.copy_(v.to(device))
        start_step = ck.get("step", 0)
        print(f"resumed from step {start_step}")
    else:
        start_step = 0

    opt = torch.optim.AdamW(model.trainable_params() + list(se.parameters()),
                            lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    norm = GlobalNorm(STATS)
    state = SubstrateStateV3(hz_target=500); state.start()
    print("substrate streaming...")
    time.sleep(1.0)

    # Pre-tokenize corpus into 64-token chunks
    encoded = []
    for line in CORPUS:
        ids = tok(line, return_tensors="pt", padding="max_length",
                  truncation=True, max_length=CTX).input_ids[0]
        encoded.append(ids)
    src_iter = cycle(encoded)

    log_f = open(LOG, "a")
    print(f"step  loss      lm       em_gap   anchor   α25    α28    grad")
    t0 = time.time()
    best_ratio = -1.0
    drops_after_peak = 0
    for step in range(start_step, args.steps):
        ids = next(src_iter).unsqueeze(0).to(device)        # (1, T)
        # Real and knockoff substrate
        w_real = state.latest_window(length=WIN_LEN)
        w_knock = make_knockoff(w_real, rng)
        wn_real = norm(w_real); wn_knock = norm(w_knock)
        wt_r = torch.from_numpy(wn_real).unsqueeze(0).to(device)
        wt_k = torch.from_numpy(wn_knock).unsqueeze(0).to(device)
        mt_r = torch.from_numpy(higher_moments(wn_real).astype(np.float32)).unsqueeze(0).to(device)
        mt_k = torch.from_numpy(higher_moments(wn_knock).astype(np.float32)).unsqueeze(0).to(device)

        S_real = se(wt_r, mt_r)
        S_knock = se(wt_k, mt_k)

        out_r = model(ids, substrate_tokens=S_real)
        out_k = model(ids, substrate_tokens=S_knock)
        with torch.no_grad():
            out_b = base(ids)

        # next-token CE under real
        logits_r = out_r.logits[:, :-1, :]
        tgt = ids[:, 1:]
        lm_loss = F.cross_entropy(logits_r.reshape(-1, logits_r.size(-1)),
                                  tgt.reshape(-1), ignore_index=tok.pad_token_id)

        # Knockoff-KL gap, last-token sym KL across batch
        last_r = out_r.logits[:, -1, :]
        last_k = out_k.logits[:, -1, :]
        last_b = out_b.logits[:, -1, :]
        em_kl = sym_kl(last_r, last_k).mean()         # we want this LARGE
        em_hinge = F.relu(args.tau - em_kl)            # penalize when em_kl < τ
        anchor_kl = (F.softmax(last_r, dim=-1) *
                     (F.log_softmax(last_r, dim=-1) - F.log_softmax(last_b, dim=-1))).sum(-1).mean()

        # Encoder contrastive — force se(real) ≠ se(knock) at the representation
        # level, independent of α. Provides gradient even when α=0.
        se_dist = ((S_real - S_knock) ** 2).mean()
        se_hinge = F.relu(SE_CONTRAST_TARGET - se_dist)

        loss = (lm_loss
                + args.lambda_em * em_hinge
                + args.gamma * anchor_kl
                + LAMBDA_SE_CONTRAST * se_hinge)

        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.trainable_params() + list(se.parameters()), GRAD_CLIP)
        opt.step()

        if step % LOG_EVERY == 0:
            alphas = model.gate_alphas()
            entry = {"step": step, "loss": loss.item(), "lm": lm_loss.item(),
                     "em_kl": em_kl.item(), "em_hinge": em_hinge.item(),
                     "anchor": anchor_kl.item(),
                     "se_dist": se_dist.item(), "se_hinge": se_hinge.item(),
                     "alpha25": alphas[0], "alpha28": alphas[1],
                     "grad": gn.item(), "t": time.time() - t0}
            log_f.write(json.dumps(entry) + "\n"); log_f.flush()
            print(f"{step:5d} {loss.item():+.4f}  {lm_loss.item():+.4f}  "
                  f"{em_kl.item():+.4f}  {anchor_kl.item():+.4f}  "
                  f"{alphas[0]:+.3f} {alphas[1]:+.3f}  {gn.item():.2f}")

        if step > 0 and step % EVAL_EVERY == 0:
            med_rk, med_rr, ratio = eval_knockoff_kl(model, se, norm, tok, state, rng, device)
            print(f"  >> EVAL @ step={step}: D_rk={med_rk:.3e} D_rr={med_rr:.3e} ratio={ratio:.3f}x")
            log_f.write(json.dumps({"step": step, "eval_ratio": ratio,
                                    "D_rk": med_rk, "D_rr": med_rr}) + "\n")
            log_f.flush()
            # Save rolling checkpoint
            lora_state = {n: p.detach().cpu() for n, p in model.named_parameters()
                          if "lora_A" in n or "lora_B" in n}
            ck_blob = {"step": step, "host": HOST, "version": "v6",
                       "xattn": model.xattn.state_dict(),
                       "se": se.state_dict(),
                       "lora": lora_state,
                       "ratio": ratio, "D_rk": med_rk, "D_rr": med_rr,
                       "tau": args.tau, "lambda_em": args.lambda_em}
            torch.save(ck_blob, CKPT)
            # Track best-ratio ckpt separately (this is the pre-registered metric)
            # Sanity-cap: only count a ratio as "best" if D_rr is below 0.1 — above
            # that, both KLs are saturated and ratio is meaningless.
            if med_rr < 0.1 and ratio > best_ratio:
                best_ratio = ratio
                torch.save(ck_blob, BEST_CKPT)
                print(f"  >> NEW BEST: ratio={ratio:.3f}× @ step={step}, saved {BEST_CKPT.name}")
                drops_after_peak = 0
            elif ratio < 0.5 * best_ratio and best_ratio > 1.0:
                drops_after_peak += 1
                print(f"  >> ratio dropped to {ratio:.3f}× (best was {best_ratio:.3f}× — drop {drops_after_peak}/2)")
                if drops_after_peak >= 2:
                    print(f"!! EARLY STOP: 2 consecutive drops from peak {best_ratio:.3f}× — training collapsed")
                    log_f.write(json.dumps({"step": step, "early_stop": True,
                                            "best_ratio": best_ratio, "current": ratio}) + "\n")
                    break
            # Kill gate (on BEST ratio, not current — current can spike from collapse)
            if step >= KILL_AT_STEP and best_ratio < KILL_RATIO:
                print(f"!! KILL GATE TRIPPED @ step={step}: best_ratio {best_ratio:.3f} < {KILL_RATIO}")
                print(f"   embodiment in this architecture class is FALSIFIED.")
                log_f.write(json.dumps({"step": step, "kill": True, "best_ratio": best_ratio}) + "\n")
                break

    state.stop(); log_f.close()
    print(f"\nfinal ckpt: {CKPT}")
    print(f"log:        {LOG}")


if __name__ == "__main__":
    main()
