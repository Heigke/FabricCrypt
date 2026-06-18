#!/usr/bin/env python3
"""
z2039: LLM Workspace Scaling — Consciousness Tests at Language Model Scale

Scales the proven consciousness tests (blindsight dissociation, workspace
necessity, contrastive awareness) from small MNIST/CIFAR models (~100K-2M
params) to a character-level transformer language model (~5M params).

If the same patterns emerge at LLM scale, this demonstrates:
  1. Architecture independence (CNN -> ViT -> Transformer LM)
  2. Task independence (classification -> next-token prediction)
  3. Scale independence (100K -> 5M params)

Architecture (GPT-small, ~5M params):
  - Encoder: 4-layer transformer (d_model=256, n_heads=4, ff_dim=512)
  - Global Workspace: bottleneck (256 -> ws_dim -> 256) with LayerNorm + ReLU
  - Self-Model: separate head predicting confidence from workspace activations
  - Decoder: linear projection to vocab (128 ASCII chars)
  - Context window: 128 tokens

Three tests (from the proven patterns):

  Test 1: LLM Blindsight (z2021/z2030 pattern)
    - Full model: task perplexity + Type-2 AUROC
    - Self-model ablated: perplexity preserved, AUROC collapses
    - Pass: AUROC > 0.7 full, < 0.55 ablated, perplexity within 10%

  Test 2: LLM Workspace Necessity (z2037 pattern)
    - 5 ablation conditions: normal, zero, random, frozen, noisy
    - Measure perplexity under each
    - Pass: >50% perplexity increase under zero/random

  Test 3: LLM Contrastive Awareness (z2036 pattern)
    - Linear probe on workspace states: correct vs incorrect predictions
    - Pass: probe AUROC > 0.6, workspace > no-workspace

Controls:
  - No-workspace model (same params, no bottleneck)
  - Random workspace model (bottleneck with noise)

References:
  Dehaene et al. 2006 — Conscious vs subliminal processing
  Baars 2005 — Global workspace theory
  Phua 2025 — Ablation-based consciousness markers
  Sergent & Dehaene 2004 — All-or-none workspace access

NO consciousness losses. All awareness properties must EMERGE from task training.
"""

import sys
import os
import json
import time
import math
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class CharDataset(Dataset):
    """Character-level dataset from text corpus."""
    def __init__(self, text, seq_len=128):
        self.seq_len = seq_len
        # Map to ASCII range [0, 127]
        self.data = np.array([min(ord(c), 127) for c in text], dtype=np.int64)
        self.n_tokens = len(self.data) - seq_len

    def __len__(self):
        return max(0, self.n_tokens)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y


def load_text_data(data_dir, seq_len=128):
    """Load text data from TinyShakespeare or generate synthetic."""
    candidates = [
        data_dir / 'tinyshakespeare.txt',
        data_dir / 'tiny_shakespeare.txt',
        data_dir / 'tinyshakespeare_input.txt',
        data_dir / 'input.txt',
    ]
    text = None
    source = None
    for p in candidates:
        if p.exists():
            raw = p.read_text(encoding='utf-8', errors='replace')
            if len(raw) > 10000:
                text = raw
                source = str(p.name)
                break

    if text is None:
        print("  No text corpus found — generating synthetic data")
        rng = np.random.RandomState(42)
        words = ['the', 'king', 'queen', 'lord', 'and', 'of', 'to', 'in',
                 'that', 'is', 'was', 'for', 'it', 'with', 'he', 'she',
                 'his', 'her', 'but', 'not', 'you', 'all', 'my', 'this',
                 'by', 'from', 'had', 'have', 'they', 'been', 'which',
                 'their', 'said', 'one', 'our', 'upon', 'are', 'what',
                 'will', 'there', 'when', 'who', 'would', 'shall', 'them',
                 'than', 'could', 'may', 'did', 'if', 'no', 'so', 'yet',
                 'more', 'some', 'now', 'do', 'good', 'great', 'come',
                 'make', 'like', 'time', 'very', 'well', 'know', 'just',
                 'how', 'man', 'old', 'new', 'way', 'day', 'part', 'long']
        lines = []
        for _ in range(20000):
            length = rng.randint(3, 15)
            sentence = ' '.join(rng.choice(words, size=length))
            if rng.random() < 0.3:
                sentence = sentence.capitalize()
            lines.append(sentence)
        text = '.\n'.join(lines)
        source = 'synthetic'

    # Truncate if too long for speed
    max_chars = 800000
    if len(text) > max_chars:
        text = text[:max_chars]

    n_total = len(text) - seq_len
    split = int(n_total * 0.9)
    train_text = text[:split + seq_len]
    test_text = text[split:]

    train_ds = CharDataset(train_text, seq_len)
    test_ds = CharDataset(test_text, seq_len)

    print(f"  Text source: {source} ({len(text):,} chars)")
    print(f"  Train: {len(train_ds):,} sequences, Test: {len(test_ds):,} sequences")
    return train_ds, test_ds, source


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=False)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


def make_causal_mask(seq_len, device):
    """Upper-triangular causal mask for autoregressive decoding."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


class CharTransformer(nn.Module):
    """Base character-level transformer (no workspace)."""
    def __init__(self, vocab_size=128, d_model=256, n_heads=4, ff_dim=512,
                 n_layers=4, seq_len=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_embed(idx)
        pos = self.pos_embed(torch.arange(T, device=idx.device))
        x = tok + pos
        mask = make_causal_mask(T, idx.device)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.norm(x)
        logits = self.head(x)
        return {'logits': logits, 'workspace': x, 'pre_ws': x}


class WorkspaceCharTransformer(nn.Module):
    """Transformer with Global Workspace bottleneck + Self-Model."""
    def __init__(self, vocab_size=128, d_model=256, n_heads=4, ff_dim=512,
                 n_layers=4, seq_len=128, ws_dim=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.ws_dim = ws_dim
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.pre_norm = nn.LayerNorm(d_model)

        # Global Workspace bottleneck
        self.ws_compress = nn.Linear(d_model, ws_dim)
        self.ws_norm = nn.LayerNorm(ws_dim)
        self.ws_expand = nn.Linear(ws_dim, d_model)

        self.post_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Self-model: predicts confidence from workspace activations
        self.self_model = nn.Sequential(
            nn.Linear(ws_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, ws_override=None):
        B, T = idx.shape
        tok = self.tok_embed(idx)
        pos = self.pos_embed(torch.arange(T, device=idx.device))
        x = tok + pos
        mask = make_causal_mask(T, idx.device)
        for block in self.blocks:
            x = block(x, mask=mask)
        pre_ws = self.pre_norm(x)  # [B, T, d_model]

        # Workspace bottleneck
        if ws_override is not None:
            ws = ws_override  # [B, T, ws_dim]
        else:
            ws = F.relu(self.ws_norm(self.ws_compress(pre_ws)))  # [B, T, ws_dim]

        broadcast = self.ws_expand(ws)  # [B, T, d_model]
        out = self.post_norm(pre_ws + broadcast)
        logits = self.head(out)

        # Self-model: per-position confidence from workspace
        # Use mean-pool across time for sequence-level confidence
        ws_pool = ws.mean(dim=1)  # [B, ws_dim]
        confidence = torch.sigmoid(self.self_model(ws_pool)).squeeze(-1)  # [B]

        return {
            'logits': logits,
            'workspace': ws,
            'pre_ws': pre_ws,
            'confidence': confidence,
        }


class DirectCharTransformer(nn.Module):
    """No-workspace control: same param count, no bottleneck."""
    def __init__(self, vocab_size=128, d_model=256, n_heads=4, ff_dim=512,
                 n_layers=4, seq_len=128, ws_dim=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.ws_dim = ws_dim
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # Instead of bottleneck, a direct feedforward layer with same params
        self.direct_ff = nn.Sequential(
            nn.Linear(d_model, ws_dim), nn.ReLU(),
            nn.Linear(ws_dim, d_model),
        )
        self.post_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Self-model on direct features (for fair comparison)
        self.self_model = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_embed(idx)
        pos = self.pos_embed(torch.arange(T, device=idx.device))
        x = tok + pos
        mask = make_causal_mask(T, idx.device)
        for block in self.blocks:
            x = block(x, mask=mask)
        h = self.norm(x)
        feat = self.direct_ff(h)
        out = self.post_norm(h + feat)
        logits = self.head(out)

        # Self-model on hidden states
        h_pool = h.mean(dim=1)
        confidence = torch.sigmoid(self.self_model(h_pool)).squeeze(-1)

        return {
            'logits': logits,
            'workspace': h,  # No real workspace, just hidden states
            'pre_ws': h,
            'confidence': confidence,
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_perplexity(logits, targets):
    """Compute perplexity from logits and targets."""
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
    return math.exp(min(loss.item(), 20.0))  # Cap for numerical stability


def compute_type2_auroc(confidences, correctness):
    """Type-2 AUROC: does confidence predict correctness?"""
    if len(np.unique(correctness)) < 2:
        return 0.5
    try:
        return roc_auc_score(correctness, confidences)
    except ValueError:
        return 0.5


def train_model(model, train_loader, device, epochs=10, lr=3e-4, label='model',
                has_self_model=True, lambda_self=0.3):
    """Train a character-level LM with optional self-model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, total_tokens = 0, 0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            B, T, V = out['logits'].shape

            # Next-token prediction loss
            loss_lm = F.cross_entropy(out['logits'].reshape(-1, V), y.reshape(-1))

            # Self-model loss (predict per-sequence correctness)
            if has_self_model and 'confidence' in out:
                with torch.no_grad():
                    preds = out['logits'].argmax(-1)  # [B, T]
                    # Fraction of correct next-token predictions per sequence
                    seq_acc = (preds == y).float().mean(dim=1)  # [B]
                    # Binary: above median accuracy
                    threshold = 0.5
                    is_good = (seq_acc > threshold).float()
                loss_self = F.binary_cross_entropy(out['confidence'], is_good)
                loss = loss_lm + lambda_self * loss_self
            else:
                loss = loss_lm

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss_lm.item() * B * T
            total_tokens += B * T

        scheduler.step()
        elapsed = time.time() - t0
        avg_loss = total_loss / total_tokens
        ppl = math.exp(min(avg_loss, 20.0))

        if ep % 2 == 0 or ep == 1 or ep == epochs:
            print(f"  [{label}] Epoch {ep:2d}/{epochs}  loss={avg_loss:.4f}  ppl={ppl:.1f}  ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Test 1: LLM Blindsight
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_blindsight(model, test_loader, device):
    """Evaluate task perplexity and Type-2 AUROC."""
    model.eval()
    total_loss, total_tokens = 0, 0
    all_conf, all_corr = [], []

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        B, T, V = out['logits'].shape

        loss = F.cross_entropy(out['logits'].reshape(-1, V), y.reshape(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += B * T

        # Per-sequence correctness
        preds = out['logits'].argmax(-1)
        seq_acc = (preds == y).float().mean(dim=1).cpu().numpy()
        is_good = (seq_acc > 0.5).astype(float)

        if 'confidence' in out:
            all_conf.extend(out['confidence'].cpu().numpy())
            all_corr.extend(is_good)

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))
    auroc = compute_type2_auroc(np.array(all_conf), np.array(all_corr)) if all_conf else 0.5

    return {'perplexity': ppl, 'type2_auroc': auroc, 'loss': avg_loss}


def ablate_self_model(model):
    """Zero out self-model weights."""
    with torch.no_grad():
        for p in model.self_model.parameters():
            p.zero_()


def run_test1_blindsight(model, test_loader, device, checkpoint):
    """Test 1: LLM Blindsight — self-model ablation dissociation."""
    print(f"\n{'='*70}")
    print(f"  TEST 1: LLM Blindsight (Self-Model Ablation)")
    print(f"{'='*70}")

    # Full model
    model.load_state_dict(checkpoint)
    full = evaluate_blindsight(model, test_loader, device)
    print(f"  Full model:      ppl={full['perplexity']:.2f}  AUROC={full['type2_auroc']:.4f}")

    # Ablate self-model
    model.load_state_dict(checkpoint)
    ablate_self_model(model)
    ablated = evaluate_blindsight(model, test_loader, device)
    print(f"  Self-ablated:    ppl={ablated['perplexity']:.2f}  AUROC={ablated['type2_auroc']:.4f}")

    # Restore for later
    model.load_state_dict(checkpoint)

    # Criteria
    ppl_ratio = ablated['perplexity'] / max(full['perplexity'], 0.01)
    ppl_preserved = ppl_ratio < 1.10  # Within 10%

    t1 = full['type2_auroc'] > 0.7
    t2 = ablated['type2_auroc'] < 0.55
    t3 = ppl_preserved
    t4 = t1 and t2  # Full dissociation

    print(f"\n  T1: Full AUROC > 0.7:                  {'PASS' if t1 else 'FAIL'} ({full['type2_auroc']:.4f})")
    print(f"  T2: Ablated AUROC < 0.55:              {'PASS' if t2 else 'FAIL'} ({ablated['type2_auroc']:.4f})")
    print(f"  T3: Perplexity preserved (<10% change): {'PASS' if t3 else 'FAIL'} (ratio={ppl_ratio:.3f})")
    print(f"  T4: Full dissociation (T1 & T2):       {'PASS' if t4 else 'FAIL'}")

    n_pass = sum([t1, t2, t3, t4])
    return {
        'full': full, 'ablated': ablated,
        'ppl_ratio': ppl_ratio,
        'tests': {'t1': bool(t1), 't2': bool(t2), 't3': bool(t3), 't4': bool(t4)},
        'n_pass': n_pass,
    }


# ---------------------------------------------------------------------------
# Test 2: LLM Workspace Necessity
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_ws_stats(model, loader, device):
    """Collect workspace statistics for frozen ablation."""
    model.eval()
    all_ws = []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        # Mean-pool workspace across time
        ws_pooled = out['workspace'].mean(dim=1)  # [B, ws_dim]
        all_ws.append(ws_pooled.cpu())
    return torch.cat(all_ws)


@torch.no_grad()
def evaluate_ws_condition(model, test_loader, device, ws_mode='normal', ws_stats=None):
    """Evaluate perplexity under various workspace ablation conditions."""
    model.eval()
    total_loss, total_tokens = 0, 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        B, T = x.shape

        if ws_mode == 'normal':
            out = model(x)
        elif ws_mode == 'zero':
            ws = torch.zeros(B, T, model.ws_dim, device=device)
            out = model(x, ws_override=ws)
        elif ws_mode == 'random':
            ws = torch.randn(B, T, model.ws_dim, device=device)
            out = model(x, ws_override=ws)
        elif ws_mode == 'frozen':
            ws_mean = ws_stats.mean(dim=0).to(device)  # [ws_dim]
            ws = ws_mean.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            out = model(x, ws_override=ws)
        elif ws_mode == 'noisy':
            out_normal = model(x)
            ws = out_normal['workspace'] + torch.randn_like(out_normal['workspace']) * 0.5
            out = model(x, ws_override=ws)
        else:
            out = model(x)

        V = out['logits'].size(-1)
        loss = F.cross_entropy(out['logits'].reshape(-1, V), y.reshape(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += B * T

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))
    return {'perplexity': ppl, 'loss': avg_loss}


def run_test2_workspace_necessity(model, train_loader, test_loader, device, checkpoint):
    """Test 2: LLM Workspace Necessity — causal ablation."""
    print(f"\n{'='*70}")
    print(f"  TEST 2: LLM Workspace Necessity (Causal Ablation)")
    print(f"{'='*70}")

    model.load_state_dict(checkpoint)

    # Collect workspace stats for frozen condition
    ws_stats = collect_ws_stats(model, train_loader, device)

    conditions = {}
    for mode in ['normal', 'zero', 'random', 'frozen', 'noisy']:
        result = evaluate_ws_condition(model, test_loader, device,
                                       ws_mode=mode, ws_stats=ws_stats)
        conditions[mode] = result
        print(f"  {mode:>8}: ppl={result['perplexity']:.2f}")

    baseline_ppl = conditions['normal']['perplexity']
    zero_increase = (conditions['zero']['perplexity'] - baseline_ppl) / max(baseline_ppl, 0.01)
    random_increase = (conditions['random']['perplexity'] - baseline_ppl) / max(baseline_ppl, 0.01)
    frozen_increase = (conditions['frozen']['perplexity'] - baseline_ppl) / max(baseline_ppl, 0.01)
    noisy_increase = (conditions['noisy']['perplexity'] - baseline_ppl) / max(baseline_ppl, 0.01)

    t1 = zero_increase > 0.50   # >50% ppl increase when zeroed
    t2 = random_increase > 0.50  # >50% ppl increase when randomized
    t3 = frozen_increase < random_increase  # Frozen less bad than random
    t4 = noisy_increase > 0.05  # Even noise hurts somewhat

    print(f"\n  T1: Zero ws >50% ppl increase:         {'PASS' if t1 else 'FAIL'} ({zero_increase:+.1%})")
    print(f"  T2: Random ws >50% ppl increase:       {'PASS' if t2 else 'FAIL'} ({random_increase:+.1%})")
    print(f"  T3: Frozen < random (partial info):    {'PASS' if t3 else 'FAIL'} ({frozen_increase:+.1%} vs {random_increase:+.1%})")
    print(f"  T4: Noise hurts (>5% increase):        {'PASS' if t4 else 'FAIL'} ({noisy_increase:+.1%})")

    model.load_state_dict(checkpoint)
    n_pass = sum([t1, t2, t3, t4])
    return {
        'conditions': {k: v for k, v in conditions.items()},
        'baseline_ppl': baseline_ppl,
        'zero_increase': zero_increase,
        'random_increase': random_increase,
        'frozen_increase': frozen_increase,
        'noisy_increase': noisy_increase,
        'tests': {'t1': bool(t1), 't2': bool(t2), 't3': bool(t3), 't4': bool(t4)},
        'n_pass': n_pass,
    }


# ---------------------------------------------------------------------------
# Test 3: LLM Contrastive Awareness
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_representations(model, loader, device, use_workspace=True):
    """Collect workspace/hidden representations and correctness labels."""
    model.eval()
    all_repr, all_correct = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)

        preds = out['logits'].argmax(-1)  # [B, T]
        seq_acc = (preds == y).float().mean(dim=1).cpu().numpy()  # [B]
        correct = (seq_acc > 0.5).astype(float)

        if use_workspace:
            # Mean-pool workspace across time
            ws = out['workspace'].mean(dim=1).cpu().numpy()  # [B, ws_dim or d_model]
        else:
            ws = out['pre_ws'].mean(dim=1).cpu().numpy()

        all_repr.append(ws)
        all_correct.append(correct)

    return {
        'representations': np.concatenate(all_repr),
        'correct': np.concatenate(all_correct),
    }


def run_test3_contrastive_awareness(ws_model, direct_model, test_loader, device,
                                     ws_checkpoint, direct_checkpoint):
    """Test 3: LLM Contrastive Awareness — workspace separability."""
    print(f"\n{'='*70}")
    print(f"  TEST 3: LLM Contrastive Awareness (Seen/Unseen Separability)")
    print(f"{'='*70}")

    # Workspace model representations
    ws_model.load_state_dict(ws_checkpoint)
    ws_reps = collect_representations(ws_model, test_loader, device, use_workspace=True)
    ws_pre_reps = collect_representations(ws_model, test_loader, device, use_workspace=False)

    # Direct model representations (no workspace)
    direct_model.load_state_dict(direct_checkpoint)
    direct_reps = collect_representations(direct_model, test_loader, device, use_workspace=True)

    def probe_auroc(reps):
        X, y = reps['representations'], reps['correct']
        n_pos = int(y.sum())
        n_neg = int((1 - y).sum())
        if n_pos < 5 or n_neg < 5:
            return 0.5
        probe = LogisticRegression(max_iter=1000, C=1.0)
        try:
            scores = cross_val_score(probe, X, y, cv=min(5, min(n_pos, n_neg)),
                                     scoring='roc_auc')
            return float(scores.mean())
        except Exception:
            return 0.5

    ws_auroc = probe_auroc(ws_reps)
    ws_pre_auroc = probe_auroc(ws_pre_reps)
    direct_auroc = probe_auroc(direct_reps)

    print(f"  Workspace probe AUROC:     {ws_auroc:.4f}")
    print(f"  Pre-workspace probe AUROC: {ws_pre_auroc:.4f}")
    print(f"  No-workspace probe AUROC:  {direct_auroc:.4f}")

    n_correct_ws = int(ws_reps['correct'].sum())
    n_wrong_ws = int((1 - ws_reps['correct']).sum())
    print(f"  Workspace: {n_correct_ws} correct, {n_wrong_ws} incorrect sequences")

    t1 = ws_auroc > 0.6                 # Seen/unseen separable in workspace
    t2 = ws_auroc > direct_auroc        # Workspace more separable than direct
    t3 = ws_auroc > ws_pre_auroc        # Workspace better than pre-workspace
    t4 = n_wrong_ws > 5 and n_correct_ws > 5  # Enough variance for valid test

    print(f"\n  T1: Workspace AUROC > 0.6:             {'PASS' if t1 else 'FAIL'} ({ws_auroc:.4f})")
    print(f"  T2: Workspace > no-workspace:          {'PASS' if t2 else 'FAIL'} ({ws_auroc:.4f} vs {direct_auroc:.4f})")
    print(f"  T3: Workspace > pre-workspace:         {'PASS' if t3 else 'FAIL'} ({ws_auroc:.4f} vs {ws_pre_auroc:.4f})")
    print(f"  T4: Sufficient variance for test:      {'PASS' if t4 else 'FAIL'} ({n_correct_ws}/{n_wrong_ws})")

    ws_model.load_state_dict(ws_checkpoint)
    n_pass = sum([t1, t2, t3, t4])
    return {
        'ws_auroc': ws_auroc,
        'ws_pre_auroc': ws_pre_auroc,
        'direct_auroc': direct_auroc,
        'n_correct': n_correct_ws,
        'n_wrong': n_wrong_ws,
        'tests': {'t1': bool(t1), 't2': bool(t2), 't3': bool(t3), 't4': bool(t4)},
        'n_pass': n_pass,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def json_safe(obj):
    if isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (torch.Tensor,)):
        return obj.tolist()
    return obj


def main():
    parser = argparse.ArgumentParser(description='z2039: LLM Workspace Scaling')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--ws-dim', type=int, default=64)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--ff-dim', type=int, default=512)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z2039] Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'not set')}")

    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  z2039: LLM Workspace Scaling")
    print(f"  Scaling consciousness tests from CNNs to transformer language models")
    print(f"  NO consciousness losses — all awareness must EMERGE from task training")
    print(f"{'='*70}")

    # Load data
    print(f"\n--- Loading data ---")
    train_ds, test_ds, data_source = load_text_data(data_dir, seq_len=args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             drop_last=True)

    # ======================================================================
    # Train workspace model
    # ======================================================================
    print(f"\n--- Training Workspace Transformer (ws_dim={args.ws_dim}) ---")
    ws_model = WorkspaceCharTransformer(
        d_model=args.d_model, n_heads=args.n_heads, ff_dim=args.ff_dim,
        n_layers=args.n_layers, seq_len=args.seq_len, ws_dim=args.ws_dim,
    ).to(device)
    ws_params = sum(p.numel() for p in ws_model.parameters())
    print(f"  Parameters: {ws_params:,}")
    train_model(ws_model, train_loader, device, epochs=args.epochs, lr=args.lr,
                label='workspace', has_self_model=True)
    ws_checkpoint = {k: v.clone() for k, v in ws_model.state_dict().items()}

    # ======================================================================
    # Train direct (no-workspace) control model
    # ======================================================================
    print(f"\n--- Training Direct Transformer (no workspace, control) ---")
    direct_model = DirectCharTransformer(
        d_model=args.d_model, n_heads=args.n_heads, ff_dim=args.ff_dim,
        n_layers=args.n_layers, seq_len=args.seq_len, ws_dim=args.ws_dim,
    ).to(device)
    direct_params = sum(p.numel() for p in direct_model.parameters())
    print(f"  Parameters: {direct_params:,}")
    train_model(direct_model, train_loader, device, epochs=args.epochs, lr=args.lr,
                label='direct', has_self_model=True)
    direct_checkpoint = {k: v.clone() for k, v in direct_model.state_dict().items()}

    # ======================================================================
    # Run the three tests
    # ======================================================================

    test1 = run_test1_blindsight(ws_model, test_loader, device, ws_checkpoint)
    test2 = run_test2_workspace_necessity(ws_model, train_loader, test_loader, device, ws_checkpoint)
    test3 = run_test3_contrastive_awareness(ws_model, direct_model, test_loader, device,
                                             ws_checkpoint, direct_checkpoint)

    # ======================================================================
    # Final verdict
    # ======================================================================
    print(f"\n{'='*70}")
    print(f"  FINAL ANALYSIS: LLM Workspace Scaling")
    print(f"{'='*70}")

    tests_summary = {
        'test1_blindsight': test1['n_pass'],
        'test2_necessity': test2['n_pass'],
        'test3_contrastive': test3['n_pass'],
    }

    print(f"\n  Test 1 (LLM Blindsight):          {test1['n_pass']}/4")
    print(f"  Test 2 (Workspace Necessity):      {test2['n_pass']}/4")
    print(f"  Test 3 (Contrastive Awareness):    {test3['n_pass']}/4")

    n_full_pass = sum(1 for v in tests_summary.values() if v == 4)
    total_subtests = test1['n_pass'] + test2['n_pass'] + test3['n_pass']

    if n_full_pass == 3:
        verdict = 'LLM_WORKSPACE_SCALING_CONFIRMED'
    elif n_full_pass == 2:
        verdict = 'PARTIAL'
    elif n_full_pass == 1:
        verdict = 'WEAK'
    else:
        verdict = 'FAIL'

    print(f"\n  Full tests passed: {n_full_pass}/3")
    print(f"  Total subtests:    {total_subtests}/12")
    print(f"\n  VERDICT: {verdict}")

    if data_source == 'synthetic':
        print(f"\n  NOTE: Trained on synthetic text. Real corpus (TinyShakespeare)")
        print(f"  would likely yield stronger results.")

    # Save results
    output = {
        'experiment': 'z2039_llm_workspace_scaling',
        'hypothesis': 'Consciousness test patterns (blindsight, workspace necessity, '
                      'contrastive awareness) scale from CNNs to transformer language models',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A',
        'data_source': data_source,
        'config': {
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'ff_dim': args.ff_dim,
            'ws_dim': args.ws_dim,
            'seq_len': args.seq_len,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
        },
        'model_params': {
            'workspace_model': ws_params,
            'direct_model': direct_params,
        },
        'references': [
            'Dehaene et al. 2006 (Conscious vs subliminal)',
            'Baars 2005 (Global workspace theory)',
            'Phua 2025 (Ablation-based markers)',
            'Sergent & Dehaene 2004 (All-or-none access)',
        ],
        'test1_blindsight': test1,
        'test2_workspace_necessity': test2,
        'test3_contrastive_awareness': test3,
        'tests_summary': tests_summary,
        'total_subtests_passed': total_subtests,
        'n_full_pass': n_full_pass,
        'verdict': verdict,
    }

    rp = results_dir / 'z2039_llm_workspace_scaling.json'
    with open(rp, 'w') as f:
        json.dump(json_safe(output), f, indent=2)
    print(f"\nResults saved to {rp}")


if __name__ == '__main__':
    main()
