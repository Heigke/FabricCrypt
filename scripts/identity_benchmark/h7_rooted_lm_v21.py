"""H7 rooted LM v2 — hard substrate coupling.

v1 result: TCR = 1.02 across all 8 ablation conditions.
Modellen ignorerade substrate helt. Diagnosis:
  - FiLM init γ ∈ [0.5, 1.5] too narrow → identity-near, easy to bypass
  - spoof CE penalty capped at 0.05 → too weak
  - no input-embedding modulation → LM never *has* to use substrate
  - no phase-dropout → trained to be robust to substrate-jitter (wrong direction)

v2 changes (all aggressive):
  1. **Exponential FiLM γ = exp(s) ∈ [0.1, 10]**  — initialized small but
     range is large; once the LM learns to compensate γ, removing the
     correct γ destroys the layer-by-layer scale ladder
  2. **Input-embedding modulation**: tok_emb = (tok_emb * γ_sub + β_sub)
     BEFORE block 0. Wrong substrate → wrong base representation → cascades
  3. **Uncapped spoof CE penalty**: full negative CE on spoof half of batch.
     Model must literally do WORSE on spoof inputs to minimize loss
  4. **Phase-dropout**: 30% of training batches, time-shift the substrate
     window by ±50 samples. Model trained to FAIL on phase-shifted
     substrate → forces it to use exact phase relations
  5. **Substrate-prediction loss WITHIN LM**: lm hidden state predicts NEXT
     substrate frame; if substrate is wrong, prediction is wrong, feedback
     loop punishes downstream
  6. **Larger SE**: GRU 128 hidden / 2 layers, output 128D
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

sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime import SubstrateState, normalize_window
# we import only SubstrateState; SubstrateEncoder rebuilt below with larger SE
from h7_rooted_lm import ByteCorpus, matched_spectrum_spoof

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/IDENTITY_H7_2026-06-09"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT = OUT_DIR / "rooted_lm_v21.pt"
REPLAY_LOG = OUT_DIR / "substrate_replay_{host}.npz"
MARGIN_NATS = 2.0   # spoof/phase PPL must exceed real PPL by e^margin ≈ 7.4×


class SubstrateEncoderV2(nn.Module):
    def __init__(self, n_channels=6, hidden=128, layers=2, d_out=128):
        super().__init__()
        self.proj_in = nn.Linear(n_channels, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, d_out)
        self.pll_head = nn.Linear(hidden, n_channels)

    def forward(self, x):
        h = F.relu(self.proj_in(x))
        h, _ = self.gru(h)
        z = self.out(h[:, -1])
        next_pred = self.pll_head(h[:, -1])
        return z, next_pred


class FiLMHardBlock(nn.Module):
    """FiLM block with exp(γ) and direct attention bias from substrate."""
    def __init__(self, d=256, heads=4, d_sub=128):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        # γ exponentiated → wider range; β unbounded
        self.film = nn.Linear(d_sub, 2 * d)
        # initialize so γ starts near identity but with capacity to move
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, mask, z_sub):
        gb = self.film(z_sub)
        s, beta = gb.chunk(2, dim=-1)
        # exp parameterization: γ ∈ (0, ∞), but soft-clipped via tanh*ln(10)
        gamma = torch.exp(torch.tanh(s) * math.log(10.0))   # range (0.1, 10)
        x = x * gamma.unsqueeze(1) + beta.unsqueeze(1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class RootedTransformerV2(nn.Module):
    def __init__(self, vocab=1024, d=256, n_layers=4, heads=4, d_sub=128, ctx=128):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        # Input-embedding modulator — applied BEFORE any block
        self.input_film = nn.Linear(d_sub, 2 * d)
        nn.init.zeros_(self.input_film.weight)
        nn.init.zeros_(self.input_film.bias)
        self.blocks = nn.ModuleList(
            [FiLMHardBlock(d, heads, d_sub) for _ in range(n_layers)]
        )
        self.head_lm  = nn.Linear(d, vocab)
        self.head_sub = nn.Linear(d, 6)
        mask = torch.triu(torch.full((ctx, ctx), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, ids, z_sub):
        B, T = ids.shape
        h = self.tok(ids) + self.pos(torch.arange(T, device=ids.device).unsqueeze(0))
        # Input modulation — wrong substrate corrupts representation from start
        gb = self.input_film(z_sub)
        s, beta = gb.chunk(2, dim=-1)
        gamma = torch.exp(torch.tanh(s) * math.log(5.0))    # tighter input range (0.2, 5)
        h = h * gamma.unsqueeze(1) + beta.unsqueeze(1)
        m = self.mask[:T, :T]
        for blk in self.blocks:
            h = blk(h, m, z_sub)
        return self.head_lm(h), self.head_sub(h[:, -1])


def time_shift_window(w: np.ndarray, max_shift: int, rng: np.random.Generator):
    """Shift the window by a random offset within the existing buffer.

    We don't actually look outside the window — we permute its rows by a circular
    shift. That's a hard test: same marginal stats, same spectrum, wrong phase.
    """
    s = int(rng.integers(-max_shift, max_shift + 1))
    if s == 0:
        return w
    return np.roll(w, shift=s, axis=0)


def train(steps=2000, ctx=128, batch=16, lr=3e-4):
    state = SubstrateState(hz_target=500); state.start()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[v2 train] host={HOST} device={device} steps={steps} batch={batch}")

    se = SubstrateEncoderV2().to(device)
    lm = RootedTransformerV2(ctx=ctx).to(device)
    opt = torch.optim.AdamW(list(se.parameters()) + list(lm.parameters()), lr=lr)

    corpus = ByteCorpus(ctx=ctx)
    loader = DataLoader(corpus, batch_size=batch, num_workers=0)
    it = iter(loader)

    rng = np.random.default_rng(123)
    replay_buf = []
    t0 = time.time()
    log = {"lm": [], "spoof": [], "phase": [], "sub": []}

    for step in range(steps):
        x, y = next(it); x, y = x.to(device), y.to(device)

        w_real = state.latest_window(length=256)
        if step % 50 == 0:
            replay_buf.append(w_real.copy())

        # Build training batch:
        # 1/3 real substrate
        # 1/3 matched-spectrum spoof (uncapped penalty)
        # 1/3 phase-shifted real (uncapped penalty)
        n_each = max(1, batch // 3)
        n_real = batch - 2 * n_each
        win_real = np.tile(normalize_window(w_real)[None], (n_real, 1, 1))
        win_spoof = np.stack([
            normalize_window(matched_spectrum_spoof(w_real, rng))
            for _ in range(n_each)
        ], axis=0)
        win_phase = np.stack([
            normalize_window(time_shift_window(w_real, max_shift=50, rng=rng))
            for _ in range(n_each)
        ], axis=0)
        sub_batch = np.concatenate([win_real, win_spoof, win_phase], axis=0)
        sub_t = torch.from_numpy(sub_batch).to(device)

        # condition labels
        is_real  = torch.zeros(batch, device=device)
        is_real[:n_real] = 1.0
        is_spoof = torch.zeros(batch, device=device)
        is_spoof[n_real:n_real+n_each] = 1.0
        is_phase = torch.zeros(batch, device=device)
        is_phase[n_real+n_each:] = 1.0

        z, next_pred = se(sub_t)
        logits, lm_next_sub = lm(x, z)

        ce = F.cross_entropy(
            logits.reshape(-1, 1024), y.reshape(-1), reduction="none"
        ).reshape(batch, -1).mean(dim=1)

        # primary: minimize CE on real substrate
        ce_real_mean  = (ce * is_real ).sum() / is_real.sum().clamp(min=1)
        ce_spoof_mean = (ce * is_spoof).sum() / is_spoof.sum().clamp(min=1)
        ce_phase_mean = (ce * is_phase).sum() / is_phase.sum().clamp(min=1)

        # MARGIN-LOSS: only penalize as long as gap is < MARGIN_NATS.
        # Once spoof_CE - real_CE >= MARGIN_NATS, no incentive to widen further.
        # This stops the v2 divergence where spoof_CE went to infinity.
        loss_spoof_margin = F.relu(MARGIN_NATS - (ce_spoof_mean - ce_real_mean))
        loss_phase_margin = F.relu(MARGIN_NATS - (ce_phase_mean - ce_real_mean))

        # SE substrate-prediction aux
        true_next = sub_t[:, -1, :]
        loss_sub_se = F.mse_loss(next_pred, true_next)
        loss_sub_lm = F.mse_loss(lm_next_sub, true_next)

        loss = (
            ce_real_mean
            + loss_spoof_margin
            + loss_phase_margin
            + 0.25 * loss_sub_se
            + 0.25 * loss_sub_lm
        )

        # for backward-compat logging
        loss_real_ce = ce_real_mean
        loss_spoof_neg = loss_spoof_margin   # log margin not -CE now
        loss_phase_neg = loss_phase_margin

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(se.parameters()) + list(lm.parameters()), 1.0
        )
        opt.step()

        log["lm"].append(loss_real_ce.item())
        log["spoof"].append(loss_spoof_neg.item())
        log["phase"].append(loss_phase_neg.item())
        log["sub"].append(loss_sub_se.item())

        if (step + 1) % 100 == 0:
            print(f"  step {step+1:5d}  "
                  f"real_ce={np.mean(log['lm'][-100:]):.3f}  "
                  f"spoof_neg={np.mean(log['spoof'][-100:]):+.3f}  "
                  f"phase_neg={np.mean(log['phase'][-100:]):+.3f}  "
                  f"sub={np.mean(log['sub'][-100:]):.3f}  "
                  f"t={time.time()-t0:.0f}s")

    state.stop()
    torch.save({
        "se": se.state_dict(),
        "lm": lm.state_dict(),
        "host": HOST,
        "ctx": ctx,
        "steps": steps,
        "version": 2,
    }, CKPT)
    print(f"[v2 train] saved {CKPT}")

    if replay_buf:
        rp = REPLAY_LOG.as_posix().format(host=HOST)
        # append v2 replay alongside v1
        np.savez_compressed(rp.replace(".npz", "_v2.npz"),
                            windows=np.stack(replay_buf))


@torch.no_grad()
def eval_cond(se, lm, device, cond: str, ctx: int, n_steps=200,
              replay_windows=None, state=None, channel_mask=None,
              phase_shift_max=0):
    rng = np.random.default_rng(11)
    corpus = ByteCorpus(ctx=ctx, seed=99)
    loader = DataLoader(corpus, batch_size=8, num_workers=0)
    it = iter(loader)
    losses = []
    for step in range(n_steps):
        x, y = next(it); x, y = x.to(device), y.to(device)
        B = x.shape[0]
        if cond == "native":
            w = state.latest_window(length=256)
        elif cond == "spoof":
            w = matched_spectrum_spoof(state.latest_window(length=256), rng)
        elif cond == "replay":
            w = replay_windows[step % len(replay_windows)]
        elif cond == "phase":
            w = time_shift_window(state.latest_window(length=256),
                                  max_shift=phase_shift_max or 50, rng=rng)
        elif cond == "zero":
            w = np.zeros((256, 6), dtype=np.float32)
        else:
            raise ValueError(cond)
        if channel_mask is not None:
            w = w * channel_mask[None, :]
        w_norm = normalize_window(w)
        sub_t = torch.from_numpy(np.tile(w_norm[None], (B, 1, 1))).to(device)
        z, _ = se(sub_t)
        logits, _ = lm(x, z)
        ce = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1))
        losses.append(ce.item())
    mean_loss = float(np.mean(losses))
    try:
        ppl = math.exp(min(mean_loss, 50.0))
    except OverflowError:
        ppl = float("inf")
    return ppl, mean_loss


def transplant(replay_host=None):
    if not CKPT.exists():
        print(f"[v2 transplant] missing {CKPT}, run train first"); return
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    se = SubstrateEncoderV2().to(device); lm = RootedTransformerV2(ctx=ckpt["ctx"]).to(device)
    se.load_state_dict(ckpt["se"]); lm.load_state_dict(ckpt["lm"])
    se.eval(); lm.eval()
    state = SubstrateState(hz_target=500); state.start()

    if replay_host is None:
        replay_host = "daedalus" if HOST == "ikaros" else "ikaros"
    rp_path = Path(str(REPLAY_LOG).format(host=replay_host))
    replay_windows = None
    if rp_path.exists():
        d = np.load(rp_path); replay_windows = d["windows"]

    results = {}
    cond_list = ["native", "spoof", "phase", "zero"]
    if replay_windows is not None:
        cond_list.append("replay")
    print(f"[v2 transplant] trained_on={ckpt['host']}  eval_on={HOST}  "
          f"{'TRANSPLANT' if HOST!=ckpt['host'] else 'NATIVE'}")
    for cond in cond_list:
        ppl, loss = eval_cond(se, lm, device, cond, ckpt["ctx"],
                               replay_windows=replay_windows, state=state)
        results[cond] = {"ppl": ppl, "loss": loss}
        print(f"  [{cond:7s}] PPL={ppl:.3f}")
    base = results.get("native", {}).get("ppl", 1.0)
    for k in results:
        results[k]["TCR"] = results[k]["ppl"] / max(base, 1e-6)
    print("\n  TCR table (PPL / native PPL):")
    for k, v in results.items():
        print(f"    {k:7s}: {v['TCR']:.3f}")

    state.stop()
    out = {"trained_on": ckpt["host"], "eval_host": HOST,
           "transplant": HOST != ckpt["host"], "results": results, "version": 2}
    rp = OUT_DIR / f"transplant_v2_{HOST}_from_{ckpt['host']}_{int(time.time())}.json"
    rp.write_text(json.dumps(out, indent=2))
    print(f"[v2 transplant] wrote {rp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "transplant"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--batch", type=int, default=15)
    ap.add_argument("--replay-host", default=None)
    args = ap.parse_args()
    if os.geteuid() != 0:
        print("[refuse] needs sudo"); sys.exit(2)
    if args.cmd == "train":
        train(steps=args.steps, ctx=args.ctx, batch=args.batch)
    else:
        transplant(replay_host=args.replay_host)


if __name__ == "__main__":
    main()
