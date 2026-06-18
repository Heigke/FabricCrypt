"""Phase 14B Task B — tiny transformer with 4-point signature injection.

Architecture:
  - 4 transformer blocks, d_model=128, n_heads=4, d_ff=256
  - vocab=8192 (BPE subset; we use a simple char-byte tokenizer to keep deps minimal)
  - ~3M params
  - Signature dim = 32 (matches LiveSig)

Injection points (all 4 active when embodied=True):
  1. Embedding shift:    emb(x) + W_sig @ sig
  2. Attention temp:     attn / (1 + alpha * sig_score)
  3. LayerNorm gain:     gamma * (1 + beta * sig_feats[:d])
  4. Residual gate:      x + sigmoid(gate(sig)) * sublayer(x)

Two heads:
  - lm_head: vocab projection (only used by T4)
  - reg_head: scalar regression head (T1)
  - cls_head: binary classifier (T2, T3)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


SIG_DIM = 32


class SigAffine(nn.Module):
    """Maps 32-d sig -> per-channel scale/shift."""
    def __init__(self, d, sig_dim=SIG_DIM):
        super().__init__()
        self.lin = nn.Linear(sig_dim, 2*d)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
    def forward(self, sig):  # sig: (B, S) or (S,) -> returns (gamma, beta) each (B,d) or (d,)
        out = self.lin(sig)
        d = out.shape[-1] // 2
        gamma, beta = out[..., :d], out[..., d:]
        return gamma, beta


class EmbodiedAttention(nn.Module):
    def __init__(self, d_model, n_heads, sig_dim=SIG_DIM, embodied=True):
        super().__init__()
        self.d = d_model
        self.h = n_heads
        self.dk = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.out = nn.Linear(d_model, d_model)
        self.embodied = embodied
        if embodied:
            self.temp_lin = nn.Linear(sig_dim, 1)
            nn.init.zeros_(self.temp_lin.weight)
            nn.init.zeros_(self.temp_lin.bias)
    def forward(self, x, sig=None, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.dk).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.dk)
        if self.embodied and sig is not None:
            # sig: (B, S) -> scalar per-batch temperature factor
            t_mod = self.temp_lin(sig)  # (B,1)
            t_mod = torch.tanh(t_mod) * 0.5  # bounded
            # softmax(att / (1 + t_mod))
            denom = (1.0 + t_mod).clamp(min=0.25).unsqueeze(-1).unsqueeze(-1)
            att = att / denom
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1,2).reshape(B, T, D)
        return self.out(out)


class EmbodiedLayerNorm(nn.Module):
    def __init__(self, d, sig_dim=SIG_DIM, embodied=True):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.embodied = embodied
        if embodied:
            self.gain = SigAffine(d, sig_dim)
    def forward(self, x, sig=None):
        y = self.ln(x)
        if self.embodied and sig is not None:
            gamma, beta = self.gain(sig)  # (B, d)
            gamma = gamma.unsqueeze(1)
            beta  = beta.unsqueeze(1)
            y = y * (1.0 + 0.1*torch.tanh(gamma)) + 0.1*torch.tanh(beta)
        return y


class EmbodiedBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, sig_dim=SIG_DIM, embodied=True):
        super().__init__()
        self.ln1 = EmbodiedLayerNorm(d_model, sig_dim, embodied)
        self.attn = EmbodiedAttention(d_model, n_heads, sig_dim, embodied)
        self.ln2 = EmbodiedLayerNorm(d_model, sig_dim, embodied)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model),
        )
        self.embodied = embodied
        if embodied:
            self.gate1 = nn.Linear(sig_dim, 1)
            self.gate2 = nn.Linear(sig_dim, 1)
            nn.init.zeros_(self.gate1.weight); nn.init.zeros_(self.gate1.bias)
            nn.init.zeros_(self.gate2.weight); nn.init.zeros_(self.gate2.bias)
    def forward(self, x, sig=None, mask=None):
        h1 = self.attn(self.ln1(x, sig), sig, mask)
        if self.embodied and sig is not None:
            g1 = torch.sigmoid(self.gate1(sig)).unsqueeze(1)
            x = x + g1 * h1
        else:
            x = x + h1
        h2 = self.ff(self.ln2(x, sig))
        if self.embodied and sig is not None:
            g2 = torch.sigmoid(self.gate2(sig)).unsqueeze(1)
            x = x + g2 * h2
        else:
            x = x + h2
        return x


class EmbodiedTiny(nn.Module):
    def __init__(self, vocab=8192, d_model=128, n_layers=4, n_heads=4,
                 d_ff=256, max_seq=128, sig_dim=SIG_DIM, embodied=True,
                 n_classes=2, reg_out=1):
        super().__init__()
        self.embodied = embodied
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        if embodied:
            self.emb_shift = nn.Linear(sig_dim, d_model)
            nn.init.zeros_(self.emb_shift.weight); nn.init.zeros_(self.emb_shift.bias)
        self.blocks = nn.ModuleList([
            EmbodiedBlock(d_model, n_heads, d_ff, sig_dim, embodied)
            for _ in range(n_layers)
        ])
        self.ln_f = EmbodiedLayerNorm(d_model, sig_dim, embodied)
        self.lm_head  = nn.Linear(d_model, vocab, bias=False)
        self.reg_head = nn.Linear(d_model, reg_out)
        self.cls_head = nn.Linear(d_model, n_classes)
        self.max_seq = max_seq
        self.d = d_model
    def encode(self, x, sig=None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        if self.embodied and sig is not None:
            shift = self.emb_shift(sig).unsqueeze(1)  # (B,1,d)
            h = h + 0.1 * torch.tanh(shift)
        # causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        for blk in self.blocks:
            h = blk(h, sig, mask)
        h = self.ln_f(h, sig)
        return h
    def forward(self, x, sig=None, head='reg'):
        h = self.encode(x, sig)
        pooled = h.mean(dim=1)  # mean over seq
        if head == 'reg':
            return self.reg_head(pooled).squeeze(-1)
        elif head == 'cls':
            return self.cls_head(pooled)
        elif head == 'lm':
            return self.lm_head(h)
        else:
            raise ValueError(head)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == '__main__':
    m = EmbodiedTiny(embodied=True)
    print(f"params (embodied): {count_params(m)/1e6:.2f}M")
    m2 = EmbodiedTiny(embodied=False)
    print(f"params (vanilla):  {count_params(m2)/1e6:.2f}M")
    x = torch.randint(0, 8192, (4, 32))
    sig = torch.randn(4, 32)
    print('reg:', m(x, sig, 'reg').shape)
    print('cls:', m(x, sig, 'cls').shape)
    print('lm :', m(x, sig, 'lm').shape)
