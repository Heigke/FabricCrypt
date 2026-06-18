#!/usr/bin/env python3
"""z2050: Hardware IS Computation — FPGA-Like GPU Neural Network

THE FPGA ISOMORPHISM:
  FPGA CLB location → GPU WGP_id (physical silicon address via s_getreg_b32)
  FPGA LUT content  → Weight bank (learned parameters per WGP)
  FPGA place & route → GPU scheduler (hardware decides which WGP runs each block)

ARCHITECTURE:
  encoder(x) → h [B, 128]        (shared CNN encoder)
  W_banks[bank_id] @ h → h_hw    (PER-SAMPLE weight bank selected by WGP_id)
  classifier(h_hw) → logits      (shared classification head)

  5 weight banks matching observed WGP values {0, 2, 4, 6, 8}.
  bank_id = WGP_to_bank[WGP_id] — direct index, no hashing.
  Full gradient flow through bank selection via torch.bmm.

TASK: Hardware-Dependent Label Permutation
  Bank parity determines label mapping:
    Even banks {0, 2, 4} → identity (digit d → label d)
    Odd banks  {1, 3}    → shifted  (digit d → label (d+5)%10)
  Without WGP_id, task is impossible (~57% ceiling with 60/40 bin split).

5 CONDITIONS:
  A_embodied:  WGP_id from HIP kernel → bank selection (real FPGA placement)
  B_blind:     No bank layer, single weights (pure software)
  C_random:    Random bank per sample (random placement)
  D_fixed:     Always bank 0 (single fixed CLB)
  E_scrambled: Trained embodied model, scrambled WGP→bank map (wrong routing)

5 TESTS:
  T1: A_embodied > 90% accuracy
  T2: A_embodied - B_blind gap > 30%
  T3: Bank weight divergence: cos(bank_i, bank_j) < 0.9
  T4: E_scrambled < A_embodied by > 10%
  T5: WGP distribution covers ≥ 3 distinct values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

N_BANKS = 8
WGP_VALUES = [0, 2, 4, 6, 8, 10, 12, 14]  # All observed WGP_ids on gfx1151
HIDDEN_DIM = 128
N_EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3

# =============================================================================
# HIP Kernel: Minimal — just read WGP_id via s_getreg_b32 hwreg(23)
# =============================================================================

HIP_SOURCE = r'''
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// One block per sample. Thread 0 reads WGP_id, writes to output.
// This is the GPU analog of reading an FPGA CLB's physical location.
__global__ void read_wgp_id(int32_t* wgp_ids, int batch_size) {
    int bid = blockIdx.x;
    if (bid >= batch_size) return;

    if (threadIdx.x == 0) {
        uint32_t hw_id1 = 0;
        asm volatile("s_getreg_b32 %0, hwreg(23, 0, 32)" : "=s"(hw_id1));
        // WGP_id: bits [10:7] of HW_ID1
        int32_t wgp_id = (int32_t)((hw_id1 >> 7) & 0xF);
        wgp_ids[bid] = wgp_id;
    }
}

torch::Tensor get_wgp_ids(int batch_size) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto wgp_ids = torch::zeros({batch_size}, opts);

    dim3 grid(batch_size);
    dim3 block(32);  // Minimal wavefront — only thread 0 does work
    read_wgp_id<<<grid, block>>>(wgp_ids.data_ptr<int32_t>(), batch_size);

    return wgp_ids;
}
'''

HIP_CPP = r'''
torch::Tensor get_wgp_ids(int batch_size);
'''


class WGPProbe:
    """Minimal probe: reads WGP_id per sample via HIP ISA."""

    def __init__(self):
        from torch.utils.cpp_extension import load_inline
        print("[BUILD] Compiling WGP probe (s_getreg_b32 hwreg(23) only)...")
        t0 = time.time()
        self.ext = load_inline(
            name='wgp_probe_z2050',
            cpp_sources=HIP_CPP,
            cuda_sources=HIP_SOURCE,
            functions=['get_wgp_ids'],
            verbose=False,
            extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'],
        )
        print(f"[BUILD] Done in {time.time()-t0:.1f}s")

        # Verify WGP probe works
        test_ids = self.get_wgp_ids(64)
        unique = torch.unique(test_ids).cpu().tolist()
        print(f"[PROBE] WGP_ids from 64 blocks: {unique}")

    def get_wgp_ids(self, batch_size):
        """Returns [batch_size] int32 tensor of WGP_ids."""
        ids = self.ext.get_wgp_ids(batch_size)
        torch.cuda.synchronize()
        return ids


# =============================================================================
# WGP → Bank Mapping
# =============================================================================

def build_wgp_to_bank():
    """Map WGP_id → bank index. Each observed WGP gets a unique bank.

    WGP_id: 0  2  4  6  8  10  12  14
    Bank:   0  1  2  3  4   5   6   7
    Parity: E  O  E  O  E   O   E   O  (alternating → 50/50 split)
    Label:  I  S  I  S  I   S   I   S  (I=identity, S=shift+5)
    """
    mapping = torch.zeros(16, dtype=torch.long)
    for i, wgp in enumerate(WGP_VALUES):
        mapping[wgp] = i
    return mapping


def wgp_to_bank(wgp_ids, mapping):
    """Convert WGP_ids tensor to bank indices using lookup table."""
    # Clamp to valid range
    clamped = wgp_ids.clamp(0, 15).long()
    return mapping.to(wgp_ids.device)[clamped]


def permute_labels(labels, bank_ids):
    """Bank-parity-dependent label permutation.

    Even banks (0, 2, 4) → identity mapping
    Odd banks  (1, 3)    → labels shifted by 5
    """
    permuted = labels.clone()
    odd_mask = (bank_ids % 2) == 1
    permuted[odd_mask] = (labels[odd_mask] + 5) % 10
    return permuted


# =============================================================================
# Models
# =============================================================================

class FPGALikeModel(nn.Module):
    """FPGA-like model: per-WGP weight banks select different transformations.

    encoder(x) → h           (shared across all banks)
    W_banks[bank_id] @ h → h_hw   (bank-specific, selected by hardware)
    classifier(h_hw) → logits     (shared head)
    """

    def __init__(self, n_banks=N_BANKS, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.n_banks = n_banks
        self.hidden_dim = hidden_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.enc_fc = nn.Linear(32 * 7 * 7, hidden_dim)

        # Per-bank weight matrices (the "LUT content" in FPGA terms)
        # Shape: [n_banks, hidden_dim, hidden_dim]
        self.bank_weights = nn.Parameter(
            torch.randn(n_banks, hidden_dim, hidden_dim) * 0.02
        )
        self.bank_bias = nn.Parameter(torch.zeros(n_banks, hidden_dim))

        # Shared classification head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x, bank_ids):
        """
        x: [B, 1, 28, 28]
        bank_ids: [B] long tensor — which bank (WGP) each sample uses
        """
        B = x.shape[0]

        # Shared encoding
        h = self.encoder(x)
        h = h.view(B, -1)
        h = F.relu(self.enc_fc(h))  # [B, hidden_dim]

        # Per-sample bank weight selection (FPGA: each CLB has its own LUT)
        # Gather the weight matrix for each sample's bank
        W = self.bank_weights[bank_ids]  # [B, hidden_dim, hidden_dim]
        b = self.bank_bias[bank_ids]     # [B, hidden_dim]

        # Apply per-sample bank transformation via batched matmul
        h_hw = torch.bmm(W, h.unsqueeze(2)).squeeze(2) + b  # [B, hidden_dim]
        h_hw = F.relu(h_hw)

        # Shared head
        logits = self.head(h_hw)
        return logits


class BlindModel(nn.Module):
    """Equivalent architecture but NO bank selection — single weight matrix."""

    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.enc_fc = nn.Linear(32 * 7 * 7, hidden_dim)
        self.mid = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x, bank_ids=None):
        B = x.shape[0]
        h = self.encoder(x)
        h = h.view(B, -1)
        h = F.relu(self.enc_fc(h))
        h = F.relu(self.mid(h))
        logits = self.head(h)
        return logits


# =============================================================================
# Data Loading
# =============================================================================

def get_data_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = torchvision.datasets.MNIST(
        root=str(ROOT / 'data'), train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(
        root=str(ROOT / 'data'), train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_loader, test_loader


# =============================================================================
# Training
# =============================================================================

def train_condition(model, train_loader, probe, wgp_map, condition,
                    n_epochs=N_EPOCHS, lr=LR):
    """Train one condition. Returns metrics dict."""
    print(f"\n{'='*60}")
    print(f"  Training: {condition}")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    all_wgp_ids = []
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        bank_counts = [0] * N_BANKS

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            B = images.shape[0]

            # Get hardware bank assignment
            if condition == 'A_embodied' or condition == 'E_scrambled':
                with torch.no_grad():
                    wgp_ids = probe.get_wgp_ids(B)
                bank_ids = wgp_to_bank(wgp_ids, wgp_map)
                all_wgp_ids.extend(wgp_ids.cpu().tolist())
            elif condition == 'C_random':
                bank_ids = torch.randint(0, N_BANKS, (B,), device=DEVICE)
            elif condition == 'D_fixed':
                bank_ids = torch.zeros(B, dtype=torch.long, device=DEVICE)
            else:  # B_blind
                bank_ids = torch.zeros(B, dtype=torch.long, device=DEVICE)

            # Permute labels by bank parity
            perm_labels = permute_labels(labels, bank_ids)

            # Forward
            if condition == 'B_blind':
                logits = model(images)
            else:
                logits = model(images, bank_ids)

            loss = F.cross_entropy(logits, perm_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == perm_labels).sum().item()
            total += B
            total_loss += loss.item()

            # Track bank distribution
            for bid in bank_ids.cpu().tolist():
                if 0 <= bid < N_BANKS:
                    bank_counts[bid] += 1

        acc = correct / total
        history.append({
            'epoch': epoch,
            'loss': total_loss / len(train_loader),
            'accuracy': acc,
            'bank_counts': bank_counts,
        })

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}: loss={history[-1]['loss']:.4f} "
                  f"acc={acc:.4f} banks={bank_counts}")

    train_time = time.time() - t0
    print(f"  Training done in {train_time:.1f}s")

    return {
        'train_history': history,
        'train_time_s': train_time,
        'wgp_ids_sample': all_wgp_ids[:500],  # Save first 500 for analysis
    }


def evaluate(model, test_loader, probe, wgp_map, condition,
             scrambled_map=None):
    """Evaluate model accuracy on test set with hardware-permuted labels."""
    model.eval()
    correct = 0
    total = 0
    all_wgp = []
    bank_counts = [0] * N_BANKS

    active_map = scrambled_map if scrambled_map is not None else wgp_map

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            B = images.shape[0]

            if condition in ('A_embodied', 'E_scrambled'):
                wgp_ids = probe.get_wgp_ids(B)
                bank_ids = wgp_to_bank(wgp_ids, active_map)
                all_wgp.extend(wgp_ids.cpu().tolist())
            elif condition == 'C_random':
                bank_ids = torch.randint(0, N_BANKS, (B,), device=DEVICE)
            elif condition == 'D_fixed':
                bank_ids = torch.zeros(B, dtype=torch.long, device=DEVICE)
            else:  # B_blind
                bank_ids = torch.zeros(B, dtype=torch.long, device=DEVICE)

            # Label permutation always uses REAL hardware bank assignment
            # (so blind model faces the same impossible task)
            if condition == 'B_blind':
                # Blind: permute using real hardware, but model can't see it
                real_wgp = probe.get_wgp_ids(B)
                real_banks = wgp_to_bank(real_wgp, wgp_map)
                perm_labels = permute_labels(labels, real_banks)
                logits = model(images)
            elif condition == 'E_scrambled':
                # Scrambled: model uses WRONG bank map, labels use REAL map
                real_wgp = probe.get_wgp_ids(B)
                real_banks = wgp_to_bank(real_wgp, wgp_map)
                perm_labels = permute_labels(labels, real_banks)
                # Model sees scrambled bank assignment
                scrambled_banks = wgp_to_bank(real_wgp, scrambled_map)
                logits = model(images, scrambled_banks)
            else:
                perm_labels = permute_labels(labels, bank_ids)
                logits = model(images, bank_ids)

            preds = logits.argmax(dim=1)
            correct += (preds == perm_labels).sum().item()
            total += B

            for bid in bank_ids.cpu().tolist():
                if 0 <= bid < N_BANKS:
                    bank_counts[bid] += 1

    acc = correct / total
    unique_wgp = sorted(set(all_wgp)) if all_wgp else []
    return {
        'accuracy': acc,
        'bank_counts': bank_counts,
        'unique_wgp_values': unique_wgp,
        'n_unique_wgp': len(unique_wgp),
    }


# =============================================================================
# Bank Weight Analysis
# =============================================================================

def analyze_bank_weights(model):
    """Compute pairwise cosine similarity between bank weight matrices."""
    if not hasattr(model, 'bank_weights'):
        return {}

    W = model.bank_weights.detach().cpu()  # [n_banks, H, H]
    # Flatten each bank's weights to a vector
    W_flat = W.view(N_BANKS, -1)  # [n_banks, H*H]

    # Pairwise cosine similarity
    cos_sim = {}
    for i in range(N_BANKS):
        for j in range(i + 1, N_BANKS):
            cos = F.cosine_similarity(W_flat[i:i+1], W_flat[j:j+1]).item()
            cos_sim[f"bank_{i}_vs_{j}"] = round(cos, 4)

    # Average cosine similarity
    vals = list(cos_sim.values())
    cos_sim['mean'] = round(np.mean(vals), 4) if vals else 0.0
    cos_sim['max'] = round(np.max(vals), 4) if vals else 0.0
    cos_sim['min'] = round(np.min(vals), 4) if vals else 0.0

    # Frobenius norm per bank
    norms = [W_flat[i].norm().item() for i in range(N_BANKS)]
    cos_sim['bank_norms'] = [round(n, 4) for n in norms]

    return cos_sim


# =============================================================================
# Build Scrambled WGP→Bank Map
# =============================================================================

def build_scrambled_map():
    """Create a scrambled WGP→bank mapping that FLIPS parity.

    Correct: WGP 0→bank 0(even), WGP 2→bank 1(odd), ...
    Scrambled: shift by 1 so every bank's parity flips.
    This means identity↔shift+5 are swapped → maximum confusion.
    """
    mapping = torch.zeros(16, dtype=torch.long)
    n = len(WGP_VALUES)
    scrambled_order = [(i + 1) % n for i in range(n)]
    for i, wgp in enumerate(WGP_VALUES):
        mapping[wgp] = scrambled_order[i]
    return mapping


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("z2050: FPGA-Like GPU Neural Network — Hardware IS Computation")
    print("  WGP_id (s_getreg_b32) selects per-sample weight bank")
    print("  No timing, no LDS, no DVFS, no debugfs — just physical CU address")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Banks: {N_BANKS}, WGP values: {WGP_VALUES}")
    print(f"Hidden dim: {HIDDEN_DIM}, Epochs: {N_EPOCHS}, Batch: {BATCH_SIZE}")
    print()

    # --- Build probe ---
    probe = WGPProbe()

    # --- WGP distribution check ---
    print("\n[CHECK] WGP distribution across 1000 blocks...")
    all_wgp = []
    for _ in range(10):
        ids = probe.get_wgp_ids(100)
        all_wgp.extend(ids.cpu().tolist())
    unique_wgp = sorted(set(all_wgp))
    wgp_counts = {v: all_wgp.count(v) for v in unique_wgp}
    print(f"  Unique WGP values: {unique_wgp}")
    print(f"  Distribution: {wgp_counts}")

    # --- Build mapping ---
    wgp_map = build_wgp_to_bank()
    scrambled_map = build_scrambled_map()

    # --- Show bank/parity mapping ---
    print("\n[MAP] WGP → Bank → Parity:")
    for wgp in unique_wgp:
        bank = wgp_map[wgp].item()
        parity = "IDENTITY" if bank % 2 == 0 else "SHIFT+5"
        print(f"  WGP {wgp} → bank {bank} → {parity}")

    # --- Data ---
    train_loader, test_loader = get_data_loaders()

    # --- Train all conditions ---
    results = {'timestamp': datetime.now().isoformat(), 'config': {
        'n_banks': N_BANKS, 'wgp_values': WGP_VALUES,
        'hidden_dim': HIDDEN_DIM, 'n_epochs': N_EPOCHS,
        'batch_size': BATCH_SIZE, 'lr': LR,
    }}

    conditions = {}

    # A: Embodied (FPGA-like)
    model_A = FPGALikeModel().to(DEVICE)
    train_A = train_condition(model_A, train_loader, probe, wgp_map, 'A_embodied')
    eval_A = evaluate(model_A, test_loader, probe, wgp_map, 'A_embodied')
    bank_analysis_A = analyze_bank_weights(model_A)
    conditions['A_embodied'] = {**train_A, 'eval': eval_A, 'bank_analysis': bank_analysis_A}
    print(f"\n  >> A_embodied eval accuracy: {eval_A['accuracy']:.4f}")
    print(f"  >> Bank weight cos_sim: {bank_analysis_A.get('mean', 'N/A')}")

    # B: Blind (no bank layer)
    model_B = BlindModel().to(DEVICE)
    train_B = train_condition(model_B, train_loader, probe, wgp_map, 'B_blind')
    eval_B = evaluate(model_B, test_loader, probe, wgp_map, 'B_blind')
    conditions['B_blind'] = {**train_B, 'eval': eval_B}
    print(f"\n  >> B_blind eval accuracy: {eval_B['accuracy']:.4f}")

    # C: Random banks
    model_C = FPGALikeModel().to(DEVICE)
    train_C = train_condition(model_C, train_loader, probe, wgp_map, 'C_random')
    eval_C = evaluate(model_C, test_loader, probe, wgp_map, 'C_random')
    bank_analysis_C = analyze_bank_weights(model_C)
    conditions['C_random'] = {**train_C, 'eval': eval_C, 'bank_analysis': bank_analysis_C}
    print(f"\n  >> C_random eval accuracy: {eval_C['accuracy']:.4f}")

    # D: Fixed bank 0
    model_D = FPGALikeModel().to(DEVICE)
    train_D = train_condition(model_D, train_loader, probe, wgp_map, 'D_fixed')
    eval_D = evaluate(model_D, test_loader, probe, wgp_map, 'D_fixed')
    bank_analysis_D = analyze_bank_weights(model_D)
    conditions['D_fixed'] = {**train_D, 'eval': eval_D, 'bank_analysis': bank_analysis_D}
    print(f"\n  >> D_fixed eval accuracy: {eval_D['accuracy']:.4f}")

    # E: Scrambled (use A's model with wrong WGP→bank mapping)
    print(f"\n{'='*60}")
    print(f"  E_scrambled: Using A's model with WRONG WGP→bank map")
    print(f"{'='*60}")
    eval_E = evaluate(model_A, test_loader, probe, wgp_map, 'E_scrambled',
                      scrambled_map=scrambled_map)
    conditions['E_scrambled'] = {'eval': eval_E}
    print(f"\n  >> E_scrambled eval accuracy: {eval_E['accuracy']:.4f}")

    # =========================================================================
    # Tests
    # =========================================================================
    print("\n" + "=" * 70)
    print("TESTS")
    print("=" * 70)

    acc_A = eval_A['accuracy']
    acc_B = eval_B['accuracy']
    acc_C = eval_C['accuracy']
    acc_D = eval_D['accuracy']
    acc_E = eval_E['accuracy']
    gap_AB = acc_A - acc_B
    gap_AE = acc_A - acc_E
    mean_cos = bank_analysis_A.get('mean', 1.0)
    n_unique = eval_A.get('n_unique_wgp', 0)

    t1_pass = acc_A > 0.90
    t2_pass = gap_AB > 0.30
    t3_pass = mean_cos < 0.90
    t4_pass = gap_AE > 0.10
    t5_pass = n_unique >= 3

    tests = {
        'T1_embodied_accuracy': {
            'criterion': 'A_embodied > 90%',
            'value': round(acc_A, 4),
            'pass': t1_pass,
        },
        'T2_embodied_blind_gap': {
            'criterion': 'A_embodied - B_blind > 30%',
            'A': round(acc_A, 4), 'B': round(acc_B, 4),
            'gap': round(gap_AB, 4),
            'pass': t2_pass,
        },
        'T3_bank_weight_divergence': {
            'criterion': 'mean cos(bank_i, bank_j) < 0.9',
            'mean_cosine': mean_cos,
            'all_pairs': bank_analysis_A,
            'pass': t3_pass,
        },
        'T4_scrambled_kill_shot': {
            'criterion': 'A_embodied - E_scrambled > 10%',
            'A': round(acc_A, 4), 'E': round(acc_E, 4),
            'gap': round(gap_AE, 4),
            'pass': t4_pass,
        },
        'T5_wgp_coverage': {
            'criterion': 'unique WGP values >= 3',
            'unique_values': eval_A.get('unique_wgp_values', []),
            'n_unique': n_unique,
            'pass': t5_pass,
        },
    }

    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)

    for name, t in tests.items():
        status = "PASS" if t['pass'] else "FAIL"
        print(f"  {name}: {status} — {t['criterion']}")
        if 'gap' in t:
            print(f"    gap = {t['gap']:.4f}")
        elif 'value' in t:
            print(f"    value = {t['value']}")
        elif 'mean_cosine' in t:
            print(f"    mean_cos = {t['mean_cosine']:.4f}")

    verdict = "FPGA_ANALOG_CONFIRMED" if n_pass >= 4 else \
              "PARTIAL" if n_pass >= 3 else "WEAK" if n_pass >= 2 else "FAIL"

    print(f"\n  VERDICT: {verdict} ({n_pass}/{n_total} PASS)")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  A_embodied:  {acc_A:.4f} (WGP→bank, hw selects weights)")
    print(f"  B_blind:     {acc_B:.4f} (no bank layer)")
    print(f"  C_random:    {acc_C:.4f} (random bank)")
    print(f"  D_fixed:     {acc_D:.4f} (always bank 0)")
    print(f"  E_scrambled: {acc_E:.4f} (wrong WGP→bank map)")
    print(f"  Gap A-B:     {gap_AB:.4f}")
    print(f"  Gap A-E:     {gap_AE:.4f}")
    print(f"  Bank cos:    {mean_cos:.4f}")
    print(f"  WGP unique:  {n_unique} values")

    # =========================================================================
    # Save Results
    # =========================================================================
    results['conditions'] = {
        k: {
            'eval': v['eval'],
            'train_history': v.get('train_history', []),
            'train_time_s': v.get('train_time_s', 0),
            'bank_analysis': v.get('bank_analysis', {}),
        } for k, v in conditions.items()
    }
    results['tests'] = tests
    results['verdict'] = verdict
    results['n_pass'] = n_pass
    results['n_total'] = n_total
    results['wgp_distribution'] = wgp_counts
    results['summary'] = {
        'A_embodied': round(acc_A, 4),
        'B_blind': round(acc_B, 4),
        'C_random': round(acc_C, 4),
        'D_fixed': round(acc_D, 4),
        'E_scrambled': round(acc_E, 4),
        'gap_A_B': round(gap_AB, 4),
        'gap_A_E': round(gap_AE, 4),
        'bank_cos_mean': mean_cos,
        'wgp_unique': n_unique,
    }

    out_path = RESULTS_DIR / 'z2050_fpga_like_gpu.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
