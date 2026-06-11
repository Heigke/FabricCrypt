"""H7 v9 — substrate-keyed ROTATION LOCK (option C: substrate as a required key).

Goal: model writes good text ONLY under its own die's live signal; a changed signal
breaks it with no recovery; language still fluent + identity-influenced under real.

Why v4-v8 failed (diagnosed):
  - Additive gating: residual bypass → no dependency.
  - FiLM multiplicative: dependency forms but unbounded gain → real language collapses
    to PPL 1e8 (v8.3), or real and encoder-similar wrong conditions corrupt together.

v9 — the LOCK. At insert layers, the hidden state is ROTATED by a substrate-keyed
orthogonal (Givens) rotation, and — critically — the residual bypass is ANNEALED TO
ZERO so all computation must flow THROUGH the lock:

    h_out = (1-g)·h + g·Rotate(h ; angles(S)),   g: 0 → 1 over warmup

  - angles(S) zero-init → Rotate ≈ identity at start; g=0 → exact identity at init.
  - Rotation is ORTHOGONAL → norm-preserving → wrong substrate corrupts but CANNOT
    explode (fixes v8.3 chaos). Bounded by construction.
  - At g=1 there is NO bypass: wrong key → wrong rotation → representation lands in a
    subspace the frozen downstream layers don't understand → broken, no fallback.
  - Right substrate: model learns angles_real ≈ 0 (rotation ≈ identity) → language
    fully preserved (lossless). Dependency loss pushes angles_wrong large → broken.

Loss (base-referenced, same robust design as v8.2):
  L = 0.3·NLL_real + λ_ok·relu(NLL_real−NLL_base−m)
    + λ_dep·mean_w relu(M_DEP − clamp(NLL_w−NLL_base, ≤cap))   w ∈ {knock,shuf,zero}
    + λ_rb·relu(KL(P_real‖P_base) − budget)                    # real ≈ base, generalizes

Pre-registered success: PPL(wrong)/PPL(real) ≥ 1.5× on knock/shuffle/daedalus AND
PPL(real) < 1.3×base AND Knockoff-KL > 2×.

Run: sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python h7_embodied_v9.py
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
from h7_embodied_v5 import inject_lora, LORA_RANK, LORA_LAYERS, INSERT_LAYERS
from h7_knockoff_kl_probe import make_knockoff, sym_kl
from h7_embodied_v7 import temporal_shuffle, encode, seq_nll, cycle
from transformers import AutoModelForCausalLM, AutoTokenizer

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_EMBODIED_V9_2026-06-10"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / f"v9_{HOST}.jsonl"
CKPT = OUT / f"v9_{HOST}.pt"
BEST_CKPT = OUT / f"v9_best_{HOST}.pt"

CTX = 64
LR = 5e-5
N_STEPS = 6000
WARMUP_G = 1200          # steps to anneal bypass g: 0 → 1 (then locked, no bypass)
EVAL_EVERY = 200
LOG_EVERY = 10
SEED = 99
POOL_SIZE = 1024

M_DEP = 1.0
DEP_CAP = 3.0            # rotation is bounded, so a higher cap is safe
LAMBDA_DEP = 0.7
RB_BUDGET = 0.5
LAMBDA_RB = 5.0
LAMBDA_REAL_OK = 2.0
REAL_OK_MARGIN = 0.3
SE_TARGET = 1.0
LAMBDA_SE = 1.0
GRAD_CLIP = 1.0


class SubstrateRotationLock(nn.Module):
    """Substrate-keyed orthogonal (Givens) rotation of the hidden state, with an
    anneal-able residual bypass g. At g=1 there is NO bypass — all info flows through
    the lock. angles(S) zero-init → identity at start. Orthogonal → bounded."""
    def __init__(self, d):
        super().__init__()
        assert d % 2 == 0
        self.d = d
        self.key = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, d // 2))
        # zero-init final → angles 0 → rotation = identity at init
        nn.init.zeros_(self.key[-1].weight); nn.init.zeros_(self.key[-1].bias)
        self.g = 0.0   # bypass gate, set per-step by the training loop (annealed 0→1)

    def forward(self, h, S):
        s = S.mean(dim=1)                       # (B, d)
        theta = self.key(s)                     # (B, d/2)
        c = torch.cos(theta).unsqueeze(1)       # (B, 1, d/2)
        sn = torch.sin(theta).unsqueeze(1)
        h0 = h[..., 0::2]; h1 = h[..., 1::2]    # (B, T, d/2)
        r0 = h0 * c - h1 * sn
        r1 = h0 * sn + h1 * c
        rot = torch.empty_like(h)
        rot[..., 0::2] = r0; rot[..., 1::2] = r1
        return (1.0 - self.g) * h + self.g * rot


class LockedSmolLM(nn.Module):
    def __init__(self, base_name=BASE_MODEL, insert_layers=INSERT_LAYERS, lora_layers=LORA_LAYERS):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(base_name)
        for p in self.base.parameters(): p.requires_grad = False
        self.d = self.base.config.hidden_size
        self.lora_mods = inject_lora(self.base, lora_layers)
        self.insert_layers = list(insert_layers)
        self.locks = nn.ModuleDict({str(i): SubstrateRotationLock(self.d) for i in insert_layers})
        self._S = None
        for i in insert_layers:
            self.base.model.layers[i].register_forward_hook(self._make_hook(i))

    def _make_hook(self, layer_idx):
        lock = self.locks[str(layer_idx)]
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if self._S is not None:
                h = lock(h, self._S)
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        return hook

    def set_g(self, g):
        for i in self.insert_layers:
            self.locks[str(i)].g = g

    def trainable_params(self):
        params = []
        for m in self.lora_mods:
            params += [m.A, m.B]
        params += list(self.locks.parameters())
        return params

    def mean_angle(self):
        # diagnostic: mean |angle| magnitude proxy via last key bias not meaningful;
        # report the g instead (set externally)
        return self.locks[str(self.insert_layers[0])].g

    def forward(self, input_ids, substrate_tokens=None, output_hidden=False):
        self._S = substrate_tokens
        out = self.base(input_ids=input_ids, output_hidden_states=output_hidden)
        self._S = None
        return out


def eval_dependency(model, se, norm, tok, state, rng, device, n_eval=6):
    """Eval at g=1 (fully locked). Returns ppl dict + Knockoff-KL ratio."""
    model.eval(); se.eval()
    model.set_g(1.0)
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
            for name, S in [("real", S_real), ("knock", S_knock), ("zero", S_zero), ("shuffle", S_shuf)]:
                l, _ = seq_nll(model, ids, S, pad); nll[name].append(l.item())
    ppl = {k: float(np.exp(np.mean(v))) for k, v in nll.items()}
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
    D_rr = torch.stack([sym_kl(Lr[i], Lr[j]) for i in range(len(Lr)) for j in range(i+1, len(Lr))]).median().item()
    ratio = D_rk / max(D_rr, 1e-12)
    model.train(); se.train()
    return ppl, ratio, D_rk, D_rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--lr", type=float, default=LR)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[v9 LOCK] host={HOST} device={device} steps={args.steps} warmup_g={WARMUP_G}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = LockedSmolLM().to(device)
    se = SubstrateEncoderV4(d_emb=model.d, K=K_TOKENS).to(device)
    print("loading frozen base for anchor...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device).eval()
    for p in base.parameters(): p.requires_grad_(False)

    # identity-at-init (g=0)
    model.set_g(0.0)
    with torch.no_grad():
        ids0 = tok("hello world this is a test", return_tensors="pt").input_ids.to(device)
        S0 = torch.randn(1, K_TOKENS, model.d, device=device)
        d0 = (model(ids0, substrate_tokens=S0).logits - model(ids0, substrate_tokens=None).logits).abs().max().item()
        print(f"identity-at-init (g=0) max|Δ| = {d0:.3e}")

    opt = torch.optim.AdamW(model.trainable_params() + list(se.parameters()),
                            lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    norm = GlobalNorm(STATS)
    state = SubstrateStateV3(hz_target=500); state.start()
    print("substrate streaming..."); time.sleep(1.0)

    print(f"generating {POOL_SIZE} base-sampled training sequences...")
    pool = []
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        for _ in range(POOL_SIZE // 32):
            seed = torch.full((32, 1), bos, dtype=torch.long, device=device)
            gen = base.generate(seed, max_new_tokens=CTX-1, do_sample=True,
                                 temperature=1.0, top_p=0.95, pad_token_id=tok.pad_token_id)
            for row in gen:
                r = row[:CTX]
                if r.shape[0] < CTX:
                    r = torch.cat([r, torch.full((CTX-r.shape[0],), tok.pad_token_id, device=device)])
                pool.append(r.cpu())
    print(f"  pool ready: {len(pool)} seqs")
    src = cycle(pool); pad = tok.pad_token_id

    log_f = open(LOG, "a")
    print("step  g     loss    nll_real dep_gap  rb_kl   grad")
    t0 = time.time(); best_score = -1e9

    for step in range(args.steps):
        g = min(1.0, step / WARMUP_G)
        model.set_g(g)
        ids = next(src).unsqueeze(0).to(device)
        w_real = state.latest_window(length=WIN_LEN)
        S_real = encode(se, norm, w_real, device)
        S_knock = encode(se, norm, make_knockoff(w_real, rng), device)
        S_shuf = encode(se, norm, temporal_shuffle(w_real, rng), device)
        S_zero = torch.zeros(1, K_TOKENS, model.d, device=device)

        nll_real, out_r = seq_nll(model, ids, S_real, pad)
        nll_knock, _ = seq_nll(model, ids, S_knock, pad)
        nll_shuf, _ = seq_nll(model, ids, S_shuf, pad)
        nll_zero, _ = seq_nll(model, ids, S_zero, pad)
        with torch.no_grad():
            out_b = base(ids)
            lb = out_b.logits[:, :-1, :]
            nll_base = F.cross_entropy(lb.reshape(-1, lb.size(-1)), ids[:, 1:].reshape(-1), ignore_index=pad)

        def dep_term(nll_w):
            gap = torch.clamp(nll_w - nll_base, max=DEP_CAP)
            return F.relu(M_DEP - gap)
        dep_loss = (dep_term(nll_knock) + dep_term(nll_shuf) + dep_term(nll_zero)) / 3.0
        real_ok = F.relu(nll_real - nll_base - REAL_OK_MARGIN)

        lr_ = out_r.logits[:, :-1, :]
        rb_kl = (F.softmax(lr_, -1) * (F.log_softmax(lr_, -1) - F.log_softmax(lb, -1))).sum(-1).mean()
        rb_hinge = F.relu(rb_kl - RB_BUDGET)
        se_dist = ((S_real - S_knock) ** 2).mean()
        se_hinge = F.relu(SE_TARGET - se_dist)

        loss = (0.3 * nll_real + LAMBDA_REAL_OK * real_ok + LAMBDA_RB * rb_hinge
                + LAMBDA_DEP * dep_loss + LAMBDA_SE * se_hinge)

        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.trainable_params() + list(se.parameters()), GRAD_CLIP)
        opt.step()

        if step % LOG_EVERY == 0:
            dep_gap = ((nll_knock + nll_shuf + nll_zero)/3 - nll_base).item()
            log_f.write(json.dumps({"step": step, "g": g, "loss": loss.item(),
                "nll_real": nll_real.item(), "dep_gap": dep_gap, "rb_kl": rb_kl.item(),
                "grad": gn.item(), "t": time.time()-t0})+"\n"); log_f.flush()
            print(f"{step:5d} {g:.2f}  {loss.item():+.3f}  {nll_real.item():+.3f}   "
                  f"{dep_gap:+.3f}  {rb_kl.item():.3f}  {gn.item():.2f}")

        if step > 0 and step % EVAL_EVERY == 0:
            ppl, ratio, D_rk, D_rr = eval_dependency(model, se, norm, tok, state, rng, device)
            dk, dz, ds = ppl["knock"]/ppl["real"], ppl["zero"]/ppl["real"], ppl["shuffle"]/ppl["real"]
            min_dep = min(dk, ds)   # zero is edge case; track knock+shuffle
            print(f"  >> EVAL {step} (g={g:.2f}): PPL real={ppl['real']:.2f} knock={ppl['knock']:.2f} "
                  f"zero={ppl['zero']:.2f} shuf={ppl['shuffle']:.2f}")
            print(f"     dep: knock={dk:.2f}× zero={dz:.2f}× shuf={ds:.2f}× (min_ks={min_dep:.2f}×) KKL={ratio:.2f}×")
            log_f.write(json.dumps({"step": step, "eval": True, "g": g, "ppl": ppl,
                "dep_knock": dk, "dep_zero": dz, "dep_shuffle": ds, "kkl_ratio": ratio})+"\n"); log_f.flush()
            lang_ok = ppl["real"] < 60.0 and g >= 0.99
            score = min_dep if lang_ok else -1.0
            lora_state = {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n}
            blob = {"step": step, "host": HOST, "version": "v9", "g": g,
                    "locks": model.locks.state_dict(), "se": se.state_dict(),
                    "lora": lora_state, "ppl": ppl, "min_dep": min_dep, "kkl_ratio": ratio}
            torch.save(blob, CKPT)
            if score > best_score:
                best_score = score
                torch.save(blob, BEST_CKPT)
                print(f"  >> NEW BEST: min_ks={min_dep:.2f}× (lang_ok={lang_ok}), saved")

    state.stop(); log_f.close()
    print(f"\nfinal: {CKPT}\nbest: {BEST_CKPT}\nlog: {LOG}")


if __name__ == "__main__":
    main()
