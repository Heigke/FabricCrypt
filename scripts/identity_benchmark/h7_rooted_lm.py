"""H7 substrate-rooted tiny LM — train + transplant + ablations end-to-end.

Pipeline (matches GPT-5's closed-loop architecture but on a tiny LM so we can
prove the FLOW before paying the Qwen3-0.6B GPU-h tax):

  1. SubstrateState sampler thread reads /dev/mem MMCFG @500 Hz
  2. SubstrateEncoder (GRU 64D) consumes the latest 256-step window per token
     and emits z_t
  3. RootedTransformer = tiny causal transformer (4 layers, 256 dim, 5M params)
     with FiLM per layer driven by z_t — so the model's *computation* depends
     on the substrate window at every layer, every token
  4. Training: next-token CE + auxiliary PLL prediction loss (model must
     predict the NEXT substrate frame from its own hidden state — forces
     it to track substrate dynamics)
  5. Adversarial spoofing: half the batch sees matched-spectrum AR(1)+1/f
     spoofs as substrate; loss pushes model to use REAL spoof-distinguishable
     features (which spoof can't replicate)
  6. Transplant eval: run the trained model on
       a) native host's live substrate
       b) foreign host's REPLAYED substrate (recorded earlier on daedalus)
       c) matched-spectrum AR(1)+1/f spoof
       d) reverse-transplant (back to native after foreign exposure)
  7. Ablations:
       A) no FiLM (substrate not connected) — does training still work? sanity
       B) FiLM with random fixed z_t — chassis-blind
       C) frozen SE — measures contribution of SE training
       D) per-channel ablation (zero out each of 6 channels in turn)

Usage:
  sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python \\
       scripts/identity_benchmark/h7_rooted_lm.py train --steps 2000
  sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python \\
       scripts/identity_benchmark/h7_rooted_lm.py transplant
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

# local import
sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime import SubstrateState, SubstrateEncoder, normalize_window

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/IDENTITY_H7_2026-06-09"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT = OUT_DIR / "rooted_lm.pt"
REPLAY_LOG = OUT_DIR / "substrate_replay_{host}.npz"


# ---------------------------------------------------------------------------
# Tiny LM with FiLM modulation
# ---------------------------------------------------------------------------
class FiLMTransformerBlock(nn.Module):
    def __init__(self, d: int = 256, heads: int = 4, d_sub: int = 64) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        # FiLM produces per-layer scale + shift from substrate embedding
        self.film = nn.Linear(d_sub, 2 * d)

    def forward(self, x, mask, z_sub):
        # FiLM: x ← γ⊙x + β where γ,β come from z_sub
        gb = self.film(z_sub)              # (B, 2d)
        gamma, beta = gb.chunk(2, dim=-1)  # each (B, d)
        gamma = 1 + 0.5 * torch.tanh(gamma)
        beta  = 0.5 * torch.tanh(beta)
        x = x * gamma.unsqueeze(1) + beta.unsqueeze(1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class RootedTransformer(nn.Module):
    def __init__(self, vocab: int = 1024, d: int = 256, n_layers: int = 4,
                 heads: int = 4, d_sub: int = 64, ctx: int = 128) -> None:
        super().__init__()
        self.vocab = vocab
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(
            [FiLMTransformerBlock(d, heads, d_sub) for _ in range(n_layers)]
        )
        self.head_lm   = nn.Linear(d, vocab)
        # Substrate prediction head — model must predict NEXT substrate frame
        # from final hidden state — forces it to track substrate dynamics
        self.head_sub = nn.Linear(d, 6)
        mask = torch.triu(torch.full((ctx, ctx), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, ids, z_sub):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        h = self.tok(ids) + self.pos(pos)
        m = self.mask[:T, :T]
        for blk in self.blocks:
            h = blk(h, m, z_sub)
        logits = self.head_lm(h)
        next_sub = self.head_sub(h[:, -1])
        return logits, next_sub


# ---------------------------------------------------------------------------
# Synthetic byte-level corpus (for fast iteration; replace w/ WikiText later)
# ---------------------------------------------------------------------------
class ByteCorpus(IterableDataset):
    """Returns (ids[T], next_id[T]) where ids are byte-mod-1024 from a text source.

    We use a deterministic pattern + small entropy so PPL has a meaningful
    floor we can measure transplant degradation against.
    """
    def __init__(self, ctx: int = 128, seed: int = 0) -> None:
        self.ctx = ctx
        # Build a moderately complex deterministic stream
        rng = np.random.default_rng(seed)
        n = 1_000_000
        base = rng.integers(0, 1024, size=n, dtype=np.int64)
        # inject periodic patterns so PPL has structure to learn
        for stride, val in [(7, 17), (13, 113), (31, 451), (97, 777)]:
            base[::stride] = val
        self.stream = base

    def __iter__(self):
        i = 0
        n = len(self.stream)
        while True:
            if i + self.ctx + 1 >= n:
                i = 0
            x = torch.from_numpy(self.stream[i:i+self.ctx])
            y = torch.from_numpy(self.stream[i+1:i+self.ctx+1])
            i += 1
            yield x, y


# ---------------------------------------------------------------------------
# Spoof generator — matched-spectrum AR(1)+1/f from real substrate window
# ---------------------------------------------------------------------------
def matched_spectrum_spoof(real_window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate spoof window with same μ, σ, AR(1) coefficient per channel.

    Real shape: (T, C). Spoof shape: (T, C). Cross-channel correlations are
    NOT preserved on purpose — they are part of the substrate fingerprint
    we want the model to depend on.
    """
    T, C = real_window.shape
    spoof = np.zeros_like(real_window)
    for c in range(C):
        x = real_window[:, c]
        mu, sd = x.mean(), x.std() + 1e-6
        # AR(1) coefficient from lag-1 autocorrelation
        if T > 2 and sd > 0:
            phi = float(np.corrcoef(x[:-1], x[1:])[0, 1])
            if not np.isfinite(phi):
                phi = 0.0
            phi = max(-0.99, min(0.99, phi))
        else:
            phi = 0.0
        eps = rng.standard_normal(T)
        y = np.zeros(T)
        y[0] = eps[0]
        for t in range(1, T):
            y[t] = phi * y[t-1] + np.sqrt(max(1 - phi*phi, 1e-3)) * eps[t]
        spoof[:, c] = y * sd + mu
    return spoof.astype(np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(steps: int = 2000, ctx: int = 128, batch: int = 32,
          lr: float = 3e-4, save: bool = True, record_replay: bool = True):
    """End-to-end training with live substrate stream + spoof adversarial branch.

    Records a snapshot of the substrate window every K steps to be used as
    the "replay" condition in transplant eval.
    """
    state = SubstrateState(hz_target=500)
    state.start()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] host={HOST} device={device} steps={steps} batch={batch} ctx={ctx}")

    se = SubstrateEncoder(n_channels=SubstrateState.N_CHANNELS, hidden=64,
                          layers=2, d_out=64).to(device)
    lm = RootedTransformer(vocab=1024, d=256, n_layers=4, heads=4,
                           d_sub=64, ctx=ctx).to(device)
    opt = torch.optim.AdamW(list(se.parameters()) + list(lm.parameters()), lr=lr)
    corpus = ByteCorpus(ctx=ctx)
    loader = DataLoader(corpus, batch_size=batch, num_workers=0)
    it = iter(loader)

    rng = np.random.default_rng(42)
    replay_buf = []   # list of (T, C) windows from native host for replay test

    t_start = time.time()
    losses_lm, losses_sub, losses_spoof = [], [], []

    for step in range(steps):
        x, y = next(it)
        x, y = x.to(device), y.to(device)

        # Live substrate window
        w = state.latest_window(length=256)
        w_norm = normalize_window(w)
        if record_replay and step % 50 == 0:
            replay_buf.append(w.copy())

        # Spoof half the batch
        n_real = batch // 2
        n_spoof = batch - n_real
        win = np.tile(w_norm[None], (n_real, 1, 1))
        spoofs = np.stack([
            normalize_window(matched_spectrum_spoof(w, rng))
            for _ in range(n_spoof)
        ], axis=0)
        sub_batch = np.concatenate([win, spoofs], axis=0)
        sub_t = torch.from_numpy(sub_batch).to(device)
        real_mask = torch.cat([
            torch.ones(n_real, device=device),
            torch.zeros(n_spoof, device=device),
        ])

        z, next_pred = se(sub_t)
        logits, lm_next_sub = lm(x, z)

        # Primary LM loss — but downweight the spoof half so the model has
        # incentive to PERFORM BETTER on real-substrate inputs and worse on spoof
        ce = F.cross_entropy(
            logits.reshape(-1, 1024), y.reshape(-1), reduction="none"
        ).reshape(batch, -1).mean(dim=1)
        loss_lm = (ce * real_mask).sum() / max(1, real_mask.sum())
        loss_spoof_neg = -(ce * (1 - real_mask)).sum() / max(1, (1 - real_mask).sum())
        # Cap the negative spoof reward so it doesn't dominate
        loss_spoof_neg = 0.05 * torch.clamp(loss_spoof_neg, max=0.0)

        # Substrate-prediction aux loss — the LM's final hidden state predicts
        # the NEXT substrate frame (we just use the last sample of next window)
        true_next = sub_t[:, -1, :]
        loss_sub_lm = F.mse_loss(lm_next_sub, true_next)
        loss_sub_se = F.mse_loss(next_pred, true_next)

        loss = loss_lm + loss_spoof_neg + 0.5 * loss_sub_lm + 0.5 * loss_sub_se

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(se.parameters()) + list(lm.parameters()), 1.0
        )
        opt.step()

        losses_lm.append(loss_lm.item())
        losses_sub.append(loss_sub_lm.item())
        losses_spoof.append(loss_spoof_neg.item())

        if (step + 1) % 100 == 0:
            print(f"  step {step+1:5d}  lm={np.mean(losses_lm[-100:]):.3f}  "
                  f"sub={np.mean(losses_sub[-100:]):.3f}  "
                  f"spoof={np.mean(losses_spoof[-100:]):+.3f}  "
                  f"elapsed={time.time()-t_start:.1f}s")

    state.stop()

    if save:
        torch.save({
            "se": se.state_dict(),
            "lm": lm.state_dict(),
            "host": HOST,
            "ctx": ctx,
            "steps": steps,
        }, CKPT)
        print(f"[train] saved checkpoint -> {CKPT}")

    if record_replay and replay_buf:
        rp = REPLAY_LOG.as_posix().format(host=HOST)
        np.savez_compressed(rp, windows=np.stack(replay_buf))
        print(f"[train] saved {len(replay_buf)} replay windows -> {rp}")

    return se, lm


# ---------------------------------------------------------------------------
# Evaluation conditions
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_ppl(se, lm, device, condition: str, n_steps: int = 200,
             ctx: int = 128, replay_windows=None, frozen_z=None,
             channel_mask: np.ndarray = None,
             state: SubstrateState = None):
    """Return (mean_ppl, mean_loss) under the given condition."""
    corpus = ByteCorpus(ctx=ctx, seed=99)
    loader = DataLoader(corpus, batch_size=8, num_workers=0)
    it = iter(loader)
    rng = np.random.default_rng(7)
    losses = []
    for step in range(n_steps):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        B = x.shape[0]
        if condition == "native":
            w = state.latest_window(length=256)
        elif condition == "spoof":
            assert state is not None
            w_real = state.latest_window(length=256)
            w = matched_spectrum_spoof(w_real, rng)
        elif condition == "replay":
            assert replay_windows is not None
            w = replay_windows[step % len(replay_windows)]
        elif condition == "zero":
            w = np.zeros((256, SubstrateState.N_CHANNELS), dtype=np.float32)
        elif condition == "fixed":
            w = (frozen_z if frozen_z is not None
                 else np.zeros((256, SubstrateState.N_CHANNELS), dtype=np.float32))
        else:
            raise ValueError(condition)

        if channel_mask is not None:
            w = w * channel_mask[None, :]
        w_norm = normalize_window(w)
        sub_t = torch.from_numpy(np.tile(w_norm[None], (B, 1, 1))).to(device)
        z, _ = se(sub_t)
        logits, _ = lm(x, z)
        ce = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1))
        losses.append(ce.item())
    mean_loss = float(np.mean(losses))
    return math.exp(mean_loss), mean_loss


# ---------------------------------------------------------------------------
# Transplant test
# ---------------------------------------------------------------------------
def transplant(replay_host: str | None = None):
    """Evaluate trained checkpoint under native / spoof / replay / zero."""
    if not CKPT.exists():
        print(f"[transplant] no checkpoint at {CKPT}, run train first")
        return
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    se = SubstrateEncoder().to(device)
    lm = RootedTransformer(ctx=ckpt["ctx"]).to(device)
    se.load_state_dict(ckpt["se"])
    lm.load_state_dict(ckpt["lm"])
    se.eval(); lm.eval()

    state = SubstrateState(hz_target=500)
    state.start()

    # Discover replay file
    replay_windows = None
    if replay_host is None:
        replay_host = "daedalus" if HOST == "ikaros" else "ikaros"
    rp_path = Path(str(REPLAY_LOG).format(host=replay_host))
    if rp_path.exists():
        d = np.load(rp_path)
        replay_windows = d["windows"]
        print(f"[transplant] loaded {len(replay_windows)} replay windows "
              f"from foreign host '{replay_host}' ({rp_path.name})")
    else:
        print(f"[transplant] WARN — no replay log at {rp_path}; "
              f"replay condition will be skipped")

    results = {}
    print(f"[transplant] training was on host = '{ckpt['host']}', "
          f"evaluating on host = '{HOST}'")
    print(f"[transplant] {'TRANSPLANT (eval host != training host)' if HOST != ckpt['host'] else 'NATIVE (eval host == training host)'}")

    cond_list = ["native", "spoof", "zero"]
    if replay_windows is not None:
        cond_list.append("replay")

    for cond in cond_list:
        ppl, loss = eval_ppl(se, lm, device, cond, n_steps=200, ctx=ckpt["ctx"],
                             replay_windows=replay_windows, state=state)
        results[cond] = {"ppl": ppl, "loss": loss}
        print(f"  [{cond:8s}] PPL={ppl:.3f}  loss={loss:.4f}")

    # TCR — Transplantation Catastrophe Ratio
    if "native" in results:
        base = results["native"]["ppl"]
        for k in results:
            results[k]["TCR_vs_native"] = results[k]["ppl"] / max(base, 1e-6)
    state.stop()

    out = {
        "trained_on": ckpt["host"],
        "eval_host": HOST,
        "transplant": HOST != ckpt["host"],
        "results": results,
    }
    rp_out = OUT_DIR / f"transplant_eval_{HOST}_from_{ckpt['host']}_{int(time.time())}.json"
    rp_out.write_text(json.dumps(out, indent=2))
    print(f"[transplant] wrote {rp_out}")
    return out


# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------
def ablations():
    if not CKPT.exists():
        print(f"[ablate] no checkpoint at {CKPT}")
        return
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    se = SubstrateEncoder().to(device)
    lm = RootedTransformer(ctx=ckpt["ctx"]).to(device)
    se.load_state_dict(ckpt["se"]); lm.load_state_dict(ckpt["lm"])
    se.eval(); lm.eval()

    state = SubstrateState(hz_target=500)
    state.start()

    results = {}

    # 1. native (reference)
    ppl_n, _ = eval_ppl(se, lm, device, "native", state=state)
    results["A0_native"] = ppl_n
    # 2. zero substrate
    ppl_z, _ = eval_ppl(se, lm, device, "zero", state=state)
    results["A1_zero_substrate"] = ppl_z
    # 3. spoof
    ppl_s, _ = eval_ppl(se, lm, device, "spoof", state=state)
    results["A2_matched_spectrum_spoof"] = ppl_s
    # 4. per-channel mask
    for ch in range(SubstrateState.N_CHANNELS):
        mask = np.ones(SubstrateState.N_CHANNELS, dtype=np.float32)
        mask[ch] = 0.0
        ppl_c, _ = eval_ppl(se, lm, device, "native", state=state, channel_mask=mask)
        results[f"A3_drop_ch{ch}"] = ppl_c

    state.stop()

    # TCR vs native
    base = results["A0_native"]
    print("\n=== Ablations (TCR = PPL / native PPL) ===")
    print(f"  A0 native              : PPL {base:.3f}")
    for k, v in results.items():
        if k == "A0_native":
            continue
        print(f"  {k:25s}: PPL {v:.3f}  TCR={v/base:.3f}")

    out = {k: {"ppl": v, "TCR": v / base} for k, v in results.items()}
    rp_out = OUT_DIR / f"ablations_{HOST}_{int(time.time())}.json"
    rp_out.write_text(json.dumps(out, indent=2))
    print(f"\n[ablate] wrote {rp_out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "transplant", "ablate"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--replay-host", default=None)
    args = ap.parse_args()

    if os.geteuid() != 0:
        print("[refuse] this script needs sudo for /dev/mem MMCFG access")
        sys.exit(2)

    if args.cmd == "train":
        train(steps=args.steps, ctx=args.ctx, batch=args.batch)
    elif args.cmd == "transplant":
        transplant(replay_host=args.replay_host)
    elif args.cmd == "ablate":
        ablations()


if __name__ == "__main__":
    main()
