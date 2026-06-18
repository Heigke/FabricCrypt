"""H7 v8 — FiLM multiplicative substrate gating for TRUE dependency.

Diagnosis across v4-v7: additive gated cross-attention (h = h + tanh(α)·attn)
RECOGNIZES substrate (Knockoff-KL up to 20×) but does NOT DEPEND on it — the
residual stream always carries the correct base computation, so wrong substrate
can only perturb, never corrupt. PPL-ablation gap stayed ~0%.

v8 puts substrate on the CRITICAL PATH via FiLM (feature-wise linear modulation):

    h = h · (1 + g·γ_s) + g·β_s            at insert layers 25, 28

where (γ_s, β_s) = MLP(mean_pool(S)) per hidden channel, g = tanh(scale) gate.
At init scale=0 and the γ/β heads are zero → exact identity (language preserved).
As training opens g, the modulation becomes load-bearing. Under a WRONG substrate,
γ_s is a wrong per-channel multiplier that scrambles the representation
multiplicatively — far harder for the 1 remaining layer + lm_head to undo than an
additive bias. So wrong substrate genuinely breaks generation.

Loss (same base-referenced dependency design as v7.1):
  L = 0.3·NLL_real + λ_ok·relu(NLL_real − NLL_base − m)        # real stays good
    + λ_dep·mean_w relu(M_DEP − clamp(NLL_w − NLL_base, ≤cap)) # wrong corrupts vs base
    + λ_anchor·relu(drift − budget) + λ_se·se_hinge + λ_em·em_hinge

Goal (pre-registered): PPL(wrong)/PPL(real) ≥ 1.5× on knock/zero/shuffle AND
PPL(real) < 1.3×PPL(base) AND Knockoff-KL ratio > 2×.

Run: sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python h7_embodied_v8.py
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
from h7_embodied_v7 import temporal_shuffle, encode, seq_nll, CORPUS, cycle
from transformers import AutoModelForCausalLM, AutoTokenizer

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_EMBODIED_V13_2026-06-11"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / f"v13_{HOST}.jsonl"
CKPT = OUT / f"v13_{HOST}.pt"
BEST_CKPT = OUT / f"v13_best_{HOST}.pt"

CTX = 64
LR = 5e-5                # gentler — FiLM multiplicative path is high-gain
N_STEPS = 6000
EVAL_EVERY = 200
LOG_EVERY = 10
SEED = 88
POOL_SIZE = 1024         # base-sampled training sequences (kills overfit)

M_DEP = 0.7              # nats — wrong substrate ≥0.7 worse than base (PPL ~2×)
DEP_CAP = 1.5            # cap rewarded gap (gentler than v8.0's 2.0)
LAMBDA_DEP = 0.7
DRIFT_BUDGET = 0.6
LAMBDA_ANCHOR = 1.0
LAMBDA_REAL_OK = 2.0     # stronger: real must track base (held-out generalization)
REAL_OK_MARGIN = 0.3
# v8.2: STRONG full-sequence base-distribution match. Pins real-substrate output
# to the frozen base's per-token distribution (which generalizes) within a small
# "personality" budget. This is what guarantees real stays good on held-out text —
# the v8.0/v8.1 failure was real overfitting because only training-NLL was guarded.
RB_BUDGET = 0.5         # nats/token of allowed real-vs-base divergence (personality)
LAMBDA_RB = 5.0
SE_TARGET = 1.0
LAMBDA_SE = 1.0
TAU_EM = 0.5
LAMBDA_EM = 3.0
GRAD_CLIP = 1.0
# v11 additions:
IDLE_FRAC = 0.45         # fraction of training 'real' windows drawn from the IDLE pool
                         # → regime-invariant die identity (fixes v10 idle-break)
GRAD_BETA = 0.3          # nats PER STD of the (now z-scored) live feature → entropy target
                         # spans ~±0.75 nats: meaningful graded steer, safe for coherence
LAMBDA_GRAD = 0.6        # weight of the graded behavioral-dependence objective
GRAD_CHANNEL = 4         # channel whose dynamics amplitude drives the style axis (keeper, load-bearing)


class FilmGate(nn.Module):
    """FiLM modulation from substrate: h -> h*(1 + g*gamma_s) + g*beta_s.
    Identity at init (scale=0, gamma/beta heads small-random).

    NOTE (v8.3 experiment, REVERTED): flatten-pool over K tokens fixed the shuffle
    gap but gave the FiLM too much gain → real-substrate language went chaotically
    unstable (PPL 30 → 1e8). Mean-pool (below) is the stable v8.2 recipe that
    produced the documented cross-die result (daedalus 2.24×, real PPL 30.6)."""
    def __init__(self, d, k_tokens=K_TOKENS):
        super().__init__()
        self.pool = nn.Linear(d, d)
        self.to_gamma = nn.Linear(d, d)
        self.to_beta = nn.Linear(d, d)
        self.scale = nn.Parameter(torch.zeros(1))   # g=tanh(0)=0 → identity at init
        nn.init.normal_(self.to_gamma.weight, std=0.02); nn.init.zeros_(self.to_gamma.bias)
        nn.init.normal_(self.to_beta.weight, std=0.02); nn.init.zeros_(self.to_beta.bias)

    def forward(self, h, S):
        s = F.gelu(self.pool(S.mean(dim=1)))       # (B, d) — stable v8.2 mean-pool
        gamma = self.to_gamma(s).unsqueeze(1)      # (B, 1, d)
        beta = self.to_beta(s).unsqueeze(1)
        g = torch.tanh(self.scale)
        return h * (1.0 + g * gamma) + g * beta


class FilmEmbodiedSmolLM(nn.Module):
    def __init__(self, base_name=BASE_MODEL,
                 insert_layers=INSERT_LAYERS, lora_layers=LORA_LAYERS):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(base_name)
        for p in self.base.parameters(): p.requires_grad = False
        self.d = self.base.config.hidden_size
        self.lora_mods = inject_lora(self.base, lora_layers)
        self.insert_layers = list(insert_layers)
        self.film = nn.ModuleDict({str(i): FilmGate(self.d) for i in insert_layers})
        self._S = None
        for i in insert_layers:
            self.base.model.layers[i].register_forward_hook(self._make_hook(i))

    def _make_hook(self, layer_idx):
        gate = self.film[str(layer_idx)]
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if self._S is not None:
                h = gate(h, self._S)
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        return hook

    def trainable_params(self):
        params = []
        for m in self.lora_mods:
            params += [m.A, m.B]
        params += list(self.film.parameters())
        return params

    def gate_scales(self):
        return [torch.tanh(self.film[str(i)].scale).item() for i in self.insert_layers]

    def forward(self, input_ids, substrate_tokens=None, output_hidden=False):
        self._S = substrate_tokens
        out = self.base(input_ids=input_ids, output_hidden_states=output_hidden)
        self._S = None
        return out


def eval_dependency(model, se, norm, tok, state, rng, device, n_eval=6):
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
    ap.add_argument("--stats", default=str(STATS),
                    help="normalization stats (median/MAD). v13: use the SHARED stats so the "
                         "other die's signal stays visible (not centered away) and the break "
                         "must be learned, not a per-die normalization artifact.")
    ap.add_argument("--xdie", required=True,
                    help="npz of the OTHER real die's windows — the hard negative. The model "
                         "must BREAK on it (this is what makes the cross-die break learned & symmetric).")
    ap.add_argument("--own-replay", dest="own_replay", default=None,
                    help="npz of THIS die's recorded windows, mixed into positives so the "
                         "discriminator can't cheat on recorded-vs-live instead of die identity.")
    ap.add_argument("--lambda-xdie", dest="lambda_xdie", type=float, default=None,
                    help="weight of the cross-die hard negative. Lower it (e.g. 0.4) for a "
                         "sensitive chip whose real coherence collapses under the full term.")
    args = ap.parse_args()
    Path(f"/tmp/h7_v13_{HOST}.pid").write_text(str(os.getpid()))   # for the thermal watchdog
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[v13 cross-die-hardneg+shared-norm] host={HOST} device={device} steps={args.steps} M_dep={M_DEP} cap={DEP_CAP}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = FilmEmbodiedSmolLM().to(device)
    se = SubstrateEncoderV4(d_emb=model.d, K=K_TOKENS).to(device)
    print("loading frozen base for anchor...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device).eval()
    for p in base.parameters(): p.requires_grad_(False)

    # identity-at-init check
    with torch.no_grad():
        ids0 = tok("hello world this is a test", return_tensors="pt").input_ids.to(device)
        S0 = torch.randn(1, K_TOKENS, model.d, device=device)
        o_with = model(ids0, substrate_tokens=S0).logits
        o_none = model(ids0, substrate_tokens=None).logits
        print(f"identity-at-init max|Δ| = {(o_with - o_none).abs().max().item():.3e}")

    opt = torch.optim.AdamW(model.trainable_params() + list(se.parameters()),
                            lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    norm = GlobalNorm(Path(args.stats)); print(f"stats: {args.stats}")
    state = SubstrateStateV3(hz_target=500); state.start()
    print("substrate streaming...")
    # v11: capture a GENUINE-IDLE substrate pool NOW (before the GPU-heavy base-corpus
    # generation warms the die). v10 trained only on the active-compute regime and so
    # broke on idle ikaros (h7_v10_idle_check: only 45% coherent at idle). Training on
    # idle + active 'real' windows makes the die identity REGIME-INVARIANT.
    print("capturing idle substrate pool (8s warmup)..."); time.sleep(8.0)
    idle_pool = []
    for _ in range(64):
        time.sleep(0.25); idle_pool.append(state.latest_window(length=WIN_LEN).copy())
    print(f"  idle pool: {len(idle_pool)} windows")

    # v13: the OTHER real die's recorded windows = hard negative (model must break on it),
    # and THIS die's recorded windows = extra positives (so the discriminator learns die
    # identity, not recorded-vs-live). Both read through the SHARED normalization.
    xdie_pool = np.load(args.xdie)["windows"].astype(np.float32)
    print(f"  xdie (other-die hard-negative) pool: {xdie_pool.shape}")
    own_pool = (np.load(args.own_replay)["windows"].astype(np.float32)
                if args.own_replay else None)
    if own_pool is not None: print(f"  own-replay (recorded-positive) pool: {own_pool.shape}")
    OWN_FRAC = 0.20    # fraction of positives drawn from recorded-own (anti recorded-vs-live)
    LAMBDA_XDIE = args.lambda_xdie if args.lambda_xdie is not None else 1.0  # cross-die hard-neg weight
    print(f"  LAMBDA_XDIE={LAMBDA_XDIE}  lr={args.lr}")

    # Base-sampled corpus pool — generate POOL_SIZE varied sequences from the FROZEN
    # base so the substrate pathway can't memorize a tiny fixed set (the v7.0/v8.0
    # overfit that made held-out real PPL diverge). Effectively unlimited in-distribution text.
    print(f"generating {POOL_SIZE} base-sampled training sequences...")
    pool = []
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        for _ in range(POOL_SIZE // 32):
            seed = torch.full((32, 1), bos, dtype=torch.long, device=device)
            gen = base.generate(seed, max_new_tokens=CTX-1, do_sample=True,
                                 temperature=1.0, top_p=0.95, pad_token_id=tok.pad_token_id)
            for row in gen:
                ids_row = row[:CTX]
                if ids_row.shape[0] < CTX:
                    pad_n = CTX - ids_row.shape[0]
                    ids_row = torch.cat([ids_row, torch.full((pad_n,), tok.pad_token_id, device=device)])
                pool.append(ids_row.cpu())
    print(f"  pool ready: {len(pool)} sequences of len {CTX}")
    src = cycle(pool); pad = tok.pad_token_id

    log_f = open(LOG, "a")
    print("step  loss     nll_real dep_gap  drift   g25    g28   grad")
    t0 = time.time(); best_score = -1e9
    # v12: EMA standardization of the graded feature. v11's feat was computed on the RAW
    # window where cross-channel scale (1 … 1e8) made std(ch4)/std(all)≈0 for EVERY window
    # → feat pinned at tanh(-1)=-0.762 (zero variance) → graded objective trained against a
    # CONSTANT (no-op). v12 computes feat on the NORMALIZED window and z-scores it with an
    # EMA so the entropy target spans a real, consistent range. (probe must match this feat.)
    feat_ema_mean, feat_ema_var, FEAT_EMA = 0.06, 0.0025, 0.02

    for step in range(args.steps):
        ids = next(src).unsqueeze(0).to(device)
        # v11: alternate REAL between LIVE (active regime) and a captured IDLE window so
        # the die identity becomes REGIME-INVARIANT (v10 broke on idle ikaros, 45% coherent).
        rsel = rng.random()
        if rsel < IDLE_FRAC:
            w_real = idle_pool[int(rng.integers(0, len(idle_pool)))]
        elif own_pool is not None and rsel < IDLE_FRAC + OWN_FRAC:
            w_real = own_pool[int(rng.integers(0, len(own_pool)))]   # recorded-own positive
        else:
            w_real = state.latest_window(length=WIN_LEN)
        # v13 hard negative: a window from the OTHER real die
        w_xdie = xdie_pool[int(rng.integers(0, len(xdie_pool)))]
        S_real = encode(se, norm, w_real, device)
        S_knock = encode(se, norm, make_knockoff(w_real, rng), device)
        S_shuf = encode(se, norm, temporal_shuffle(w_real, rng), device)
        S_xdie = encode(se, norm, w_xdie, device)
        S_zero = torch.zeros(1, K_TOKENS, model.d, device=device)

        nll_real, out_r = seq_nll(model, ids, S_real, pad)
        nll_knock, _ = seq_nll(model, ids, S_knock, pad)
        nll_shuf, _ = seq_nll(model, ids, S_shuf, pad)
        nll_xdie, _ = seq_nll(model, ids, S_xdie, pad)
        nll_zero, _ = seq_nll(model, ids, S_zero, pad)
        with torch.no_grad():
            out_b = base(ids)
            lb = out_b.logits[:, :-1, :]
            nll_base = F.cross_entropy(lb.reshape(-1, lb.size(-1)),
                                       ids[:, 1:].reshape(-1), ignore_index=pad)

        def dep_term(nll_w):
            gap = torch.clamp(nll_w - nll_base, max=DEP_CAP)
            return F.relu(M_DEP - gap)
        # v10: dependency on {knock, shuffle} only — zero (no-signal) DROPPED as a
        # target (it was the degenerate inversion harbor in v8/v9). zero kept as a
        # monitored diagnostic in eval, not a training objective.
        # v13: the OTHER real die is a first-class hard negative (weighted) alongside
        # knock/shuffle. This is what turns the cross-die break from a normalization
        # artifact into a LEARNED "my live dynamics vs another chip's live dynamics".
        dep_loss = (dep_term(nll_knock) + dep_term(nll_shuf)
                    + LAMBDA_XDIE * dep_term(nll_xdie)) / (2.0 + LAMBDA_XDIE)
        real_ok = F.relu(nll_real - nll_base - REAL_OK_MARGIN)

        # STRONG full-sequence base-distribution match: real-substrate per-token
        # distribution must stay within RB_BUDGET of the frozen base. Guarantees
        # real generalizes (it IS base, up to a personality budget).
        lr = out_r.logits[:, :-1, :]; lb_full = out_b.logits[:, :-1, :]
        rb_kl = (F.softmax(lr, -1) *
                 (F.log_softmax(lr, -1) - F.log_softmax(lb_full, -1))).sum(-1).mean()
        rb_hinge = F.relu(rb_kl - RB_BUDGET)

        last_r = out_r.logits[:, -1, :]; last_b = out_b.logits[:, -1, :]
        drift = (F.softmax(last_r, -1) *
                 (F.log_softmax(last_r, -1) - F.log_softmax(last_b, -1))).sum(-1).mean()
        anchor_hinge = F.relu(drift - DRIFT_BUDGET)
        # v10 KEY FIX: separate the encoder embedding of real from BOTH knock AND
        # shuffle. v8 only separated real/knock, so the encoder mapped shuffle≈real
        # (moment features are permutation-invariant) and the LM could not break
        # shuffle without breaking real → shuffle stuck at 1.21×. Pushing real/shuffle
        # apart trains the temporal conv path to encode dynamics, not just marginals.
        se_dist_k = ((S_real - S_knock) ** 2).mean()
        se_dist_s = ((S_real - S_shuf) ** 2).mean()
        se_dist_x = ((S_real - S_xdie) ** 2).mean()   # v13: separate real from other die
        se_hinge = (F.relu(SE_TARGET - se_dist_k) + F.relu(SE_TARGET - se_dist_s)
                    + F.relu(SE_TARGET - se_dist_x))
        last_k = model(ids, substrate_tokens=S_knock).logits[:, -1, :]
        last_s = model(ids, substrate_tokens=S_shuf).logits[:, -1, :]
        last_x = model(ids, substrate_tokens=S_xdie).logits[:, -1, :]
        em_kl = (sym_kl(last_r.detach(), last_k).mean()
                 + sym_kl(last_r.detach(), last_s).mean()
                 + sym_kl(last_r.detach(), last_x).mean()) / 3.0
        em_hinge = F.relu(TAU_EM - em_kl)

        # v11 GRADED behavioral dependence (oracle-endorsed): tie a live-signal feature
        # (channel-GRAD_CHANNEL dynamics amplitude, relative to the window) to a measurable
        # OUTPUT statistic (mean next-token entropy). This makes the substrate CONTINUOUSLY
        # shape coherent style — not just gate break/no-break. Proven post-hoc by
        # Pearson(feat, output-entropy) with a temporal-shuffle control.
        # v12 FIX: feat on the NORMALIZED window (per-channel standardized) so channel-4's
        # relative dynamics actually vary window-to-window (raw-window ratio was degenerate).
        z_real = norm(w_real)
        feat = float(np.tanh(np.std(z_real[:, GRAD_CHANNEL]) / (np.std(z_real) + 1e-6) - 1.0))
        feat_ema_mean = (1 - FEAT_EMA) * feat_ema_mean + FEAT_EMA * feat
        feat_ema_var = (1 - FEAT_EMA) * feat_ema_var + FEAT_EMA * (feat - feat_ema_mean) ** 2
        feat_z = float(np.clip((feat - feat_ema_mean) / (np.sqrt(feat_ema_var) + 1e-6), -2.5, 2.5))
        p_r = F.softmax(lr, -1)
        ent = -(p_r * torch.log(p_r + 1e-9)).sum(-1).mean()
        with torch.no_grad():
            p_b = F.softmax(lb_full, -1)
            ent_b = -(p_b * torch.log(p_b + 1e-9)).sum(-1).mean()
        target_ent = ent_b + GRAD_BETA * feat_z
        graded_loss = (ent - target_ent) ** 2

        loss = (0.3 * nll_real + LAMBDA_REAL_OK * real_ok + LAMBDA_RB * rb_hinge
                + LAMBDA_DEP * dep_loss + LAMBDA_ANCHOR * anchor_hinge
                + LAMBDA_SE * se_hinge + LAMBDA_EM * em_hinge
                + LAMBDA_GRAD * graded_loss)

        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.trainable_params() + list(se.parameters()), GRAD_CLIP)
        opt.step()

        if step % LOG_EVERY == 0:
            g = model.gate_scales()
            dep_gap = ((nll_knock + nll_shuf + nll_zero)/3 - nll_base).item()
            log_f.write(json.dumps({"step": step, "loss": loss.item(),
                "nll_real": nll_real.item(), "dep_gap": dep_gap, "drift": drift.item(),
                "g25": g[0], "g28": g[1], "grad": gn.item(), "t": time.time()-t0})+"\n")
            log_f.flush()
            print(f"{step:5d} {loss.item():+.3f}  {nll_real.item():+.3f}   "
                  f"{dep_gap:+.3f}  {drift.item():.3f}  {g[0]:+.3f} {g[1]:+.3f}  {gn.item():.2f}")

        if step > 0 and step % EVAL_EVERY == 0:
            ppl, ratio, D_rk, D_rr = eval_dependency(model, se, norm, tok, state, rng, device)
            dk, dz, ds = ppl["knock"]/ppl["real"], ppl["zero"]/ppl["real"], ppl["shuffle"]/ppl["real"]
            min_dep = min(dk, ds)   # v10: score on TRAINED conditions {knock, shuffle}; zero is diagnostic
            print(f"  >> EVAL {step}: PPL real={ppl['real']:.2f} knock={ppl['knock']:.2f} "
                  f"zero={ppl['zero']:.2f} shuf={ppl['shuffle']:.2f}")
            print(f"     dep: knock={dk:.2f}× zero={dz:.2f}× shuf={ds:.2f}× (min={min_dep:.2f}×) KKL={ratio:.2f}×")
            log_f.write(json.dumps({"step": step, "eval": True, "ppl": ppl,
                "dep_knock": dk, "dep_zero": dz, "dep_shuffle": ds, "kkl_ratio": ratio})+"\n")
            log_f.flush()
            lang_ok = ppl["real"] < 60.0
            score = min_dep if lang_ok else -1.0
            lora_state = {n: p.detach().cpu() for n, p in model.named_parameters()
                          if "lora_A" in n or "lora_B" in n}
            blob = {"step": step, "host": HOST, "version": "v13", "stats": str(args.stats),
                    "film": model.film.state_dict(), "se": se.state_dict(),
                    "lora": lora_state, "ppl": ppl, "min_dep": min_dep, "kkl_ratio": ratio}
            torch.save(blob, CKPT)
            if score > best_score:
                best_score = score
                torch.save(blob, BEST_CKPT)
                print(f"  >> NEW BEST: min_dep={min_dep:.2f}× (lang_ok={lang_ok}), saved")

    state.stop(); log_f.close()
    print(f"\nfinal: {CKPT}\nbest: {BEST_CKPT}\nlog: {LOG}")


if __name__ == "__main__":
    main()
