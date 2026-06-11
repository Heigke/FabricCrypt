"""H7 rooted LM v4a — Flamingo-style gated cross-attention on SmolLM2-135M.

Implements the O101 oracle-synthesis recommendation:
  - Base: SmolLM2-135M FROZEN
  - Telemetry encoder: 1D depthwise-sep conv → 2-layer transformer → Perceiver
    resampler producing K=8 substrate tokens per 256-sample window
  - Gated cross-attention into 3 insertion layers (20, 24, 28 of 30) with
    tanh(α=0) init — identity-at-init guarantee
  - Loss: L_native + L_zero_KL + 3 margin terms + gate reg + InfoNCE
  - Batch: 40/20/20/20 native/zero/wrong/spoof, all 4 forwards per sample
  - Normalization: GLOBAL median/MAD (not per-window!)

Pre-registration: O101_synthesis.md falsification suite, all 6 must pass.
Kill criteria: after 3k steps, kill if any of:
  - PPL_zero(GL) > 1.25 * PPL_base(GL)
  - ΔNLL_zero-native < 0.2 nats and flat
  - All gates |α| < 0.05
"""
from __future__ import annotations
import argparse, json, math, os, socket, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3, higher_moments

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/IDENTITY_H7_2026-06-09"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT = OUT_DIR / "rooted_lm_v4a.pt"
STATS = OUT_DIR / "global_substrate_stats.npz"
REPLAY_LOG = OUT_DIR / "substrate_replay_{host}_v4a.npz"

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
INSERT_LAYERS = [20, 24, 28]      # top 1/3 of 30
K_TOKENS = 8                       # K substrate tokens
N_CHANNELS = 10
WIN_LEN = 256                      # substrate window samples

# Loss weights
LAMBDA_ZERO_KL = 1.0
LAMBDA_M0_MARGIN = 1.0; M0_NATS = 0.5
LAMBDA_MW_MARGIN = 1.0; MW_NATS = 3.0
LAMBDA_GATE = 1e-3
LAMBDA_CONTRAST = 0.05


# ---------------------------------------------------------------------------
# Global normalization (frozen stats from O101 decision)
# ---------------------------------------------------------------------------
class GlobalNorm:
    """Robust standardization + clamp to [-CLIP, CLIP] to prevent loss explosion
    from heavy-tail outlier samples or under-resolved channels (MAD floor)."""
    CLIP = 8.0

    def __init__(self, path: Path):
        d = np.load(path)
        self.median = d["median"].astype(np.float32)
        self.mad = np.maximum(d["mad"].astype(np.float32), 1e-3)

    def __call__(self, w: np.ndarray) -> np.ndarray:
        """w: (T, C). Robust standardize → soft-clamp to [-CLIP, CLIP] via tanh-scaled."""
        z = (w - self.median) / self.mad
        # Soft-clamp: preserves direction but bounds magnitude
        return (self.CLIP * np.tanh(z / self.CLIP)).astype(np.float32)


# ---------------------------------------------------------------------------
# Substrate Encoder — conv → transformer → Perceiver resampler → K tokens
# ---------------------------------------------------------------------------
class DWSepConv1d(nn.Module):
    def __init__(self, c_in, c_out, k=7, stride=1):
        super().__init__()
        self.dw = nn.Conv1d(c_in, c_in, kernel_size=k, stride=stride,
                            padding=k//2, groups=c_in)
        self.pw = nn.Conv1d(c_in, c_out, kernel_size=1)
        self.ln = nn.LayerNorm(c_out)

    def forward(self, x):
        h = self.pw(self.dw(x))                # (B, C, T)
        h = h.transpose(1, 2)                  # (B, T, C)
        h = F.gelu(self.ln(h))
        return h.transpose(1, 2)


class PerceiverResampler(nn.Module):
    """K learned queries cross-attend to (T_sub × d) sequence → (K, d)."""
    def __init__(self, d: int, K: int, heads: int = 4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(K, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln_q = nn.LayerNorm(d); self.ln_kv = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))
        self.ln_ff = nn.LayerNorm(d)

    def forward(self, kv):
        B = kv.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        q = self.ln_q(q); k = self.ln_kv(kv)
        h, _ = self.attn(q, k, k, need_weights=False)
        h = q + h
        h = h + self.ff(self.ln_ff(h))
        return h          # (B, K, d)


class SubstrateEncoderV4(nn.Module):
    def __init__(self, n_ch=N_CHANNELS, n_mom=N_CHANNELS*5, d_emb=576, K=K_TOKENS, hidden=128):
        super().__init__()
        # Raw window path
        self.conv1 = DWSepConv1d(n_ch, hidden, k=7, stride=2)
        self.conv2 = DWSepConv1d(hidden, hidden, k=5, stride=2)
        self.conv3 = DWSepConv1d(hidden, hidden, k=3, stride=1)
        enc_layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=4*hidden,
                                                batch_first=True, activation="gelu", norm_first=True)
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=2)
        # Higher-moment path
        self.mom_proj = nn.Sequential(nn.Linear(n_mom, hidden), nn.GELU(),
                                       nn.Linear(hidden, hidden))
        # Resampler outputs K tokens at LM hidden dim
        self.to_lm = nn.Linear(hidden, d_emb)
        self.resampler = PerceiverResampler(d_emb, K, heads=4)

    def forward(self, x, moments):
        """x: (B, T, C), moments: (B, C*5) → (B, K, d_emb)"""
        h = x.transpose(1, 2)                  # (B, C, T)
        h = self.conv1(h); h = self.conv2(h); h = self.conv3(h)
        h = h.transpose(1, 2)                  # (B, T', hidden)
        h = self.tr(h)
        # Inject higher-moment features as additional token at end of seq
        m = self.mom_proj(moments).unsqueeze(1)   # (B, 1, hidden)
        h = torch.cat([h, m], dim=1)
        h = self.to_lm(h)                       # (B, T'+1, d_emb)
        tokens = self.resampler(h)              # (B, K, d_emb)
        return tokens


# ---------------------------------------------------------------------------
# Gated cross-attention block — y = h + tanh(α=0) * CrossAttn(h, S)
# ---------------------------------------------------------------------------
class GatedCrossAttn(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln_q = nn.LayerNorm(d); self.ln_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.alpha = nn.Parameter(torch.zeros(1))    # gate, init 0
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.beta = nn.Parameter(torch.zeros(1))     # ff gate, init 0
        self.ln_ff = nn.LayerNorm(d)

    def forward(self, h, S):
        """h: (B, T_text, d), S: (B, K, d). Returns h modified by substrate."""
        q = self.ln_q(h); k = self.ln_kv(S)
        attn_out, _ = self.attn(q, k, k, need_weights=False)
        h = h + torch.tanh(self.alpha) * attn_out
        h = h + torch.tanh(self.beta) * self.ff(self.ln_ff(h))
        return h


# ---------------------------------------------------------------------------
# Rooted SmolLM — wraps SmolLM2-135M with substrate xattn at insert layers
# ---------------------------------------------------------------------------
class RootedSmolLM(nn.Module):
    def __init__(self, base_name=BASE_MODEL, insert_layers=INSERT_LAYERS):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(base_name)
        for p in self.base.parameters(): p.requires_grad = False
        cfg = self.base.config
        self.d = cfg.hidden_size
        self.heads = cfg.num_attention_heads
        self.insert_layers = set(insert_layers)
        self.xattn = nn.ModuleDict({
            str(i): GatedCrossAttn(self.d, self.heads) for i in insert_layers
        })
        # Hook the base model's transformer layers
        # SmolLM2 uses LLaMA-style; layers are at base.model.layers
        self._tx_layers = self.base.model.layers
        assert len(self._tx_layers) >= max(insert_layers) + 1
        self._S = None   # current substrate tokens, set by forward()
        # Register hooks
        for i in insert_layers:
            self._tx_layers[i].register_forward_hook(self._make_hook(i))

    def _make_hook(self, layer_idx):
        xattn = self.xattn[str(layer_idx)]
        def hook(module, args, output):
            # output is tuple (hidden_states, ...) for LLaMA layers
            h = output[0] if isinstance(output, tuple) else output
            if self._S is not None:
                h = xattn(h, self._S)
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        return hook

    def gate_alphas(self):
        return [self.xattn[str(i)].alpha.detach().item() for i in self.insert_layers]

    def forward(self, input_ids, substrate_tokens=None, attention_mask=None):
        """substrate_tokens: (B, K, d) or None for null substrate."""
        self._S = substrate_tokens
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        self._S = None
        return out


# ---------------------------------------------------------------------------
# Tiny text dataset — uses tokenizer + WikiText-style synthetic for now.
# For real eval we should swap in WikiText-103 val; here we mock with a fixed
# byte-derived corpus the same way v2/v3 did, but at SmolLM2 vocab level via
# the tokenizer applied to a long deterministic text string.
# ---------------------------------------------------------------------------
class TextCorpus(IterableDataset):
    def __init__(self, tokenizer, ctx=128, seed=42):
        # Deterministic procedurally-generated text — same trick as before but
        # tokenized through SmolLM2's BPE so the model sees real subwords.
        rng = np.random.default_rng(seed)
        # Generate a long string of common English bigrams to give realistic LM signal
        chars = "the quick brown fox jumps over lazy dog. she said hello world to me. abcdefghijklmnopqrstuvwxyz 0123456789 "
        big_text = "".join(rng.choice(list(chars), size=200_000))
        self.ids = tokenizer(big_text, return_tensors="pt").input_ids[0]
        self.ctx = ctx
        self.rng = rng

    def __iter__(self):
        N = self.ids.shape[0] - self.ctx - 1
        while True:
            i = int(self.rng.integers(0, N))
            x = self.ids[i:i+self.ctx]
            y = self.ids[i+1:i+self.ctx+1]
            yield x, y


def matched_spectrum_spoof(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """AR(1) + 1/f spoof matching per-channel μ, σ, φ."""
    out = np.zeros_like(w)
    for c in range(w.shape[1]):
        x = w[:, c]
        mu, sg = x.mean(), x.std()
        phi = np.corrcoef(x[:-1], x[1:])[0,1] if x.std() > 0 else 0
        eps = rng.normal(0, sg*np.sqrt(max(1e-6, 1-phi**2)), len(x))
        y = np.zeros_like(x); y[0] = x[0]
        for t in range(1, len(x)):
            y[t] = phi*y[t-1] + eps[t]
        y = y - y.mean() + mu
        out[:, c] = y
    return out


def time_shift_window(w: np.ndarray, max_shift: int, rng: np.random.Generator):
    n = w.shape[0]
    shift = int(rng.integers(1, max_shift+1)) * int(rng.choice([-1, 1]))
    return np.roll(w, shift, axis=0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(steps=8000, ctx=128, batch=8, lr_enc=2e-4, lr_gate=3e-5, phase_a=2000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[v4a train] host={HOST} device={device} steps={steps} batch={batch}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Frozen base for KL anchor — separate copy
    base_lm = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device).eval()
    for p in base_lm.parameters(): p.requires_grad = False

    rooted = RootedSmolLM().to(device)
    se = SubstrateEncoderV4(d_emb=rooted.d, K=K_TOKENS).to(device)

    # Two optimizer groups: encoder + gates separately
    enc_params = list(se.parameters())
    xattn_params = list(rooted.xattn.parameters())
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": lr_enc},
        {"params": xattn_params, "lr": lr_gate}
    ], betas=(0.9, 0.999), weight_decay=0.01)

    norm = GlobalNorm(STATS)
    state = SubstrateStateV3(hz_target=500); state.start()
    corpus = TextCorpus(tok, ctx=ctx)
    loader = DataLoader(corpus, batch_size=batch, num_workers=0)
    it = iter(loader)
    rng = np.random.default_rng(123)
    replay_buf = []
    t0 = time.time()
    log = {"native":[], "zero_kl":[], "m0":[], "mw_s":[], "mw_w":[], "alphas":[]}

    for step in range(steps):
        x, y = next(it); x, y = x.to(device), y.to(device)
        attn_mask = torch.ones_like(x)

        w_real = state.latest_window(length=WIN_LEN)
        if step % 50 == 0:
            replay_buf.append(w_real.copy())

        # Build 4 substrate inputs for the SAME text sample
        spoof_w  = matched_spectrum_spoof(w_real, rng)
        # "wrong-host" surrogate: for now use a time-shifted real (until daedalus v4a live)
        # — placeholder, must be replaced with actual daedalus telemetry in step 2
        wrong_w  = time_shift_window(w_real, 100, rng)
        zero_w   = np.zeros((WIN_LEN, N_CHANNELS), dtype=np.float32)

        def encode(raw):
            wn  = norm(raw)
            mom = higher_moments(wn).astype(np.float32)
            wn  = np.tile(wn[None], (batch, 1, 1))
            mom = np.tile(mom[None], (batch, 1))
            return (torch.from_numpy(wn).to(device),
                    torch.from_numpy(mom).to(device))

        sub_t_n, mom_n = encode(w_real)
        sub_t_s, mom_s = encode(spoof_w)
        sub_t_w, mom_w = encode(wrong_w)
        sub_t_z, mom_z = encode(zero_w)

        # Encode substrate → K tokens for each cond
        S_native = se(sub_t_n, mom_n)
        S_spoof  = se(sub_t_s, mom_s)
        S_wrong  = se(sub_t_w, mom_w)
        S_zero   = se(sub_t_z, mom_z)

        # Forward through rooted LM under each condition
        def lm_forward_with_S(S):
            return rooted(x, substrate_tokens=S, attention_mask=attn_mask).logits

        logits_n = lm_forward_with_S(S_native)
        logits_z = lm_forward_with_S(S_zero)
        logits_s = lm_forward_with_S(S_spoof)
        logits_w = lm_forward_with_S(S_wrong)

        # Native CE
        def ce(logits): return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        ce_n = ce(logits_n); ce_z = ce(logits_z); ce_s = ce(logits_s); ce_w = ce(logits_w)

        # KL anchor on zero → base
        with torch.no_grad():
            logits_base = base_lm(input_ids=x, attention_mask=attn_mask).logits
        # KL( P_base || P_theta|t=0 ), token-mean
        log_p_z = F.log_softmax(logits_z, dim=-1)
        p_base  = F.softmax(logits_base, dim=-1)
        zero_kl = F.kl_div(log_p_z, p_base, reduction="batchmean")

        # Margin losses
        loss_m0 = F.relu(M0_NATS - (ce_z - ce_n))
        loss_mws = F.relu(MW_NATS - (ce_s - ce_n))
        loss_mww = F.relu(MW_NATS - (ce_w - ce_n))

        # Gate reg
        alphas = torch.stack([rooted.xattn[str(i)].alpha for i in INSERT_LAYERS])
        betas  = torch.stack([rooted.xattn[str(i)].beta  for i in INSERT_LAYERS])
        loss_gate = (alphas**2).sum() + (betas**2).sum()

        # Contrastive: pool S_native vs (S_zero, S_spoof, S_wrong) negatives
        z_n = S_native.mean(dim=1)         # (B, d)
        z_z = S_zero.mean(dim=1); z_s = S_spoof.mean(dim=1); z_w = S_wrong.mean(dim=1)
        # InfoNCE: similarity native to native (anchor), pull away from others
        tau = 0.1
        pos = (z_n * z_n).sum(dim=-1) / tau     # diag self-sim, basically constant
        # We compare per-sample: native vs each negative
        sim_z = (z_n * z_z).sum(dim=-1) / tau
        sim_s = (z_n * z_s).sum(dim=-1) / tau
        sim_w = (z_n * z_w).sum(dim=-1) / tau
        # Per-sample contrastive: want pos > each neg; soft margin
        loss_c = (F.relu(sim_z - pos + 0.1) + F.relu(sim_s - pos + 0.1) + F.relu(sim_w - pos + 0.1)).mean() / 3

        # Phase: A vs B
        if step < phase_a:
            # Phase A: anchor + native, no margins
            loss = ce_n + LAMBDA_ZERO_KL * zero_kl + LAMBDA_GATE * loss_gate
        else:
            loss = (ce_n
                    + LAMBDA_ZERO_KL * zero_kl
                    + LAMBDA_M0_MARGIN * loss_m0
                    + LAMBDA_MW_MARGIN * (loss_mws + loss_mww)
                    + LAMBDA_GATE * loss_gate
                    + LAMBDA_CONTRAST * loss_c)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc_params + xattn_params, 1.0)
        opt.step()

        log["native"].append(ce_n.item())
        log["zero_kl"].append(zero_kl.item())
        log["m0"].append(loss_m0.item())
        log["mw_s"].append(loss_mws.item())
        log["mw_w"].append(loss_mww.item())
        log["alphas"].append(rooted.gate_alphas())

        if (step+1) % 50 == 0:
            a_mean = float(np.abs(rooted.gate_alphas()).mean())
            print(f"  step {step+1:5d}  "
                  f"ce_n={np.mean(log['native'][-50:]):.3f}  "
                  f"zKL={np.mean(log['zero_kl'][-50:]):.4f}  "
                  f"m0={np.mean(log['m0'][-50:]):.3f}  "
                  f"mws={np.mean(log['mw_s'][-50:]):.3f}  "
                  f"mww={np.mean(log['mw_w'][-50:]):.3f}  "
                  f"|α|={a_mean:.4f}  "
                  f"phase={'A' if step<phase_a else 'B'}  "
                  f"t={time.time()-t0:.0f}s")

    state.stop()
    torch.save({
        "se": se.state_dict(),
        "xattn": rooted.xattn.state_dict(),
        "host": HOST, "ctx": ctx, "steps": steps,
        "version": "4a",
        "log": log,
    }, CKPT)
    print(f"[v4a train] saved {CKPT}")
    if replay_buf:
        rp = REPLAY_LOG.as_posix().format(host=HOST)
        np.savez_compressed(rp, windows=np.stack(replay_buf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--steps", type=int, default=3000)   # 1-day kill-criterion
    ap.add_argument("--batch", type=int, default=4)      # smaller for SmolLM2 on CPU
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--phase_a", type=int, default=2000)
    args = ap.parse_args()
    if args.cmd == "train":
        train(steps=args.steps, ctx=args.ctx, batch=args.batch, phase_a=args.phase_a)


if __name__ == "__main__":
    main()
