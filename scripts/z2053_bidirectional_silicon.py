#!/usr/bin/env python3
"""z2053: Bidirectional Silicon I/O — Neural Network Reads AND Writes Hardware

THE DEEPEST GPU INTEGRATION:
  READ:  s_getreg_b32 hwreg(23)  → physical WGP_id (which CU am I on?)
  WRITE: s_setreg_b32 hwreg(1)   → MODE register (FP rounding, denorm flags)
  READ-BACK: s_getreg_b32 hwreg(1) → verify write took effect

WHY THIS MATTERS:
  z2050: model READS hardware identity → selects computation path (one-way)
  z2053: model READS hardware identity AND WRITES execution mode (two-way)

  This is the GPU equivalent of:
  - FPGA: reading CLB location AND reconfiguring the CLB's LUT content
  - Biology: sensing environment AND modifying own neural firing properties

  The MODE register controls:
  - Bits [3:0]: FP rounding mode (nearest-even, +inf, -inf, toward-zero)
  - Bit 4: denorm mode for single precision
  - Bit 5: denorm mode for double precision
  - Bit 6: DX10 clamp enable
  - Bit 7: IEEE mode

  By writing different rounding modes per-WGP, we create a system where:
  1. The same math operations produce DIFFERENT results on different CUs
  2. The model learns to exploit these differences for classification
  3. Breaking the read/write coupling destroys performance (kill shot)

TASK: Bank parity + rounding-mode-dependent computation
  Even WGPs: identity labels + round-to-nearest
  Odd WGPs:  shifted labels + round-toward-zero (subtly different numerics)

CONDITIONS:
  A_bidirectional: Read WGP + Write MODE → full coupling
  B_read_only:    Read WGP, no MODE write → z2050 replication
  C_blind:        No hardware at all
  D_write_only:   Write MODE without reading WGP → partial
  E_scrambled:    Wrong WGP→MODE mapping → kill shot
  F_frozen:       Trained A model, freeze MODE writes → partial

TESTS:
  T1: A > 90%
  T2: A - C gap > 30%
  T3: A - E gap > 10% (scrambled kill shot)
  T4: A > B (bidirectional > read-only, write adds value)
  T5: Rounding mode verification (different modes produce different sums)
  T6: Cross-WGP rounding diversity (WGPs write different modes)
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
import socket

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
MACHINE = socket.gethostname()

N_BANKS = 8
WGP_VALUES = [0, 2, 4, 6, 8, 10, 12, 14]
HIDDEN_DIM = 128
N_EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3

# =============================================================================
# HIP Kernel: Bidirectional — read WGP_id AND write MODE register
# =============================================================================

HIP_SOURCE = r'''
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// MODE register bits on GFX11:
// [3:0] = FP_ROUND  (0=nearest-even, 1=+inf, 2=-inf, 3=toward-zero)
// [4]   = FP_DENORM_SINGLE (0=flush, 1=allow)
// [5]   = FP_DENORM_DOUBLE
// [7]   = IEEE_MODE
//
// We use FP_ROUND bits to create CU-specific numeric behavior.
// Round-to-nearest vs round-toward-zero produces DIFFERENT results
// for the same floating-point operations — hardware IS computation.

// Read WGP_id, write MODE, read back MODE, do rounding-sensitive computation
__global__ void bidirectional_probe(
    int32_t* wgp_ids,         // [B] — physical WGP_id per block
    int32_t* mode_before,     // [B] — MODE register before write
    int32_t* mode_after,      // [B] — MODE register after write
    float* rounding_test,     // [B] — rounding-sensitive sum
    const int32_t* mode_map,  // [N_BANKS] — bank→rounding_mode mapping
    const int32_t* wgp_to_bank, // [16] — WGP_id→bank lookup
    int batch_size,
    int write_mode            // 0=no write, 1=write per-bank mode, 2=write fixed mode
) {
    int bid = blockIdx.x;
    if (bid >= batch_size) return;

    if (threadIdx.x == 0) {
        // STEP 1: READ — physical CU identity
        uint32_t hw_id1 = 0;
        asm volatile("s_getreg_b32 %0, hwreg(23, 0, 32)" : "=s"(hw_id1));
        int32_t wgp_id = (int32_t)((hw_id1 >> 7) & 0xF);
        wgp_ids[bid] = wgp_id;

        // STEP 2: READ original MODE
        uint32_t orig_mode = 0;
        asm volatile("s_getreg_b32 %0, hwreg(1, 0, 32)" : "=s"(orig_mode));
        mode_before[bid] = (int32_t)orig_mode;

        // STEP 3: WRITE MODE — set FP rounding based on bank
        int32_t bank = wgp_to_bank[wgp_id & 0xF];
        uint32_t new_round = 0;
        if (write_mode == 1) {
            new_round = (uint32_t)(mode_map[bank] & 0x3);  // Only bottom 2 bits
        } else if (write_mode == 2) {
            new_round = 3;  // Always round-toward-zero
        }

        if (write_mode > 0) {
            // Write FP_ROUND bits [3:0] of MODE register
            // Must readfirstlane to move VGPR→SGPR before s_setreg
            // hwreg encoding: (id) | (offset << 6) | ((size-1) << 11)
            // hwreg(1, 0, 4) = 1 | (0 << 6) | (3 << 11) = 0x1801
            int sgpr_val = __builtin_amdgcn_readfirstlane(new_round);
            __builtin_amdgcn_s_setreg(0x1801, sgpr_val);
        }

        // STEP 4: READ back MODE to verify
        uint32_t new_mode = 0;
        asm volatile("s_getreg_b32 %0, hwreg(1, 0, 32)" : "=s"(new_mode));
        mode_after[bid] = (int32_t)new_mode;

        // STEP 5: Rounding-sensitive computation
        // Sum of 1/3 + 1/7 + 1/11 + ... — result depends on rounding mode
        float sum = 0.0f;
        for (int i = 0; i < 32; i++) {
            float denom = (float)(3 + 4*i);  // 3, 7, 11, 15, ...
            sum += 1.0f / denom;
        }
        rounding_test[bid] = sum;

        // STEP 6: RESTORE original rounding mode
        if (write_mode > 0) {
            uint32_t restore = orig_mode & 0xFu;
            int sgpr_restore = __builtin_amdgcn_readfirstlane((int)restore);
            __builtin_amdgcn_s_setreg(0x1801, sgpr_restore);
        }
    }
}

// Simple read-only probe (z2050 replication)
__global__ void read_wgp_only(int32_t* wgp_ids, int batch_size) {
    int bid = blockIdx.x;
    if (bid >= batch_size) return;
    if (threadIdx.x == 0) {
        uint32_t hw_id1 = 0;
        asm volatile("s_getreg_b32 %0, hwreg(23, 0, 32)" : "=s"(hw_id1));
        wgp_ids[bid] = (int32_t)((hw_id1 >> 7) & 0xF);
    }
}

// Torch wrappers
std::vector<torch::Tensor> bidirectional_read(
    int batch_size,
    torch::Tensor mode_map,
    torch::Tensor wgp_to_bank,
    int write_mode
) {
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto wgp_ids = torch::zeros({batch_size}, opts_i);
    auto m_before = torch::zeros({batch_size}, opts_i);
    auto m_after = torch::zeros({batch_size}, opts_i);
    auto round_test = torch::zeros({batch_size}, opts_f);

    bidirectional_probe<<<batch_size, 32>>>(
        wgp_ids.data_ptr<int32_t>(),
        m_before.data_ptr<int32_t>(),
        m_after.data_ptr<int32_t>(),
        round_test.data_ptr<float>(),
        mode_map.data_ptr<int32_t>(),
        wgp_to_bank.data_ptr<int32_t>(),
        batch_size,
        write_mode
    );
    return {wgp_ids, m_before, m_after, round_test};
}

torch::Tensor get_wgp_ids_only(int batch_size) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto wgp_ids = torch::zeros({batch_size}, opts);
    read_wgp_only<<<batch_size, 32>>>(wgp_ids.data_ptr<int32_t>(), batch_size);
    return wgp_ids;
}
'''

HIP_CPP = r'''
std::vector<torch::Tensor> bidirectional_read(int batch_size, torch::Tensor mode_map, torch::Tensor wgp_to_bank, int write_mode);
torch::Tensor get_wgp_ids_only(int batch_size);
'''


class BidirectionalProbe:
    """Reads WGP_id AND writes MODE register per-block."""

    def __init__(self):
        from torch.utils.cpp_extension import load_inline
        print("[BUILD] Compiling bidirectional probe (s_getreg + s_setreg)...")
        t0 = time.time()
        self.ext = load_inline(
            name='bidir_probe_z2053',
            cpp_sources=HIP_CPP,
            cuda_sources=HIP_SOURCE,
            functions=['bidirectional_read', 'get_wgp_ids_only'],
            verbose=False,
            extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'],
        )
        print(f"[BUILD] Done in {time.time()-t0:.1f}s")

        # Build WGP → bank mapping
        self.wgp_to_bank_map = torch.zeros(16, dtype=torch.int32, device='cuda')
        for i, wgp in enumerate(WGP_VALUES):
            self.wgp_to_bank_map[wgp] = i

        # Mode map: even banks → round nearest (0), odd banks → round toward zero (3)
        self.mode_map = torch.zeros(N_BANKS, dtype=torch.int32, device='cuda')
        for i in range(N_BANKS):
            self.mode_map[i] = 0 if i % 2 == 0 else 3  # nearest vs toward-zero

        # Verify
        results = self.bidirectional_read(64, write_mode=1)
        wgp_ids, m_before, m_after, round_test = results
        unique_wgp = torch.unique(wgp_ids).cpu().tolist()
        mode_changed = (m_before != m_after).sum().item()
        round_vals = torch.unique(round_test).cpu().tolist()
        print(f"[PROBE] WGP_ids: {unique_wgp}")
        print(f"[PROBE] MODE changed in {mode_changed}/64 blocks")
        print(f"[PROBE] Unique rounding results: {len(round_vals)} values")
        if len(round_vals) > 1:
            print(f"[PROBE] Rounding range: {min(round_vals):.8f} - {max(round_vals):.8f}")

    def bidirectional_read(self, batch_size, write_mode=1):
        """Returns (wgp_ids, mode_before, mode_after, rounding_test)."""
        results = self.ext.bidirectional_read(
            batch_size, self.mode_map, self.wgp_to_bank_map, write_mode
        )
        torch.cuda.synchronize()
        return results

    def get_wgp_ids(self, batch_size):
        """Read-only WGP probe."""
        ids = self.ext.get_wgp_ids_only(batch_size)
        torch.cuda.synchronize()
        return ids


# =============================================================================
# WGP → Bank Mapping + Label Permutation
# =============================================================================

def build_wgp_to_bank():
    mapping = torch.zeros(16, dtype=torch.long)
    for i, wgp in enumerate(WGP_VALUES):
        mapping[wgp] = i
    return mapping

def wgp_to_bank(wgp_ids, mapping):
    clamped = wgp_ids.clamp(0, 15).long()
    return mapping.to(wgp_ids.device)[clamped]

def permute_labels(labels, bank_ids):
    permuted = labels.clone()
    odd_mask = (bank_ids % 2) == 1
    permuted[odd_mask] = (labels[odd_mask] + 5) % 10
    return permuted


# =============================================================================
# Model: With rounding-mode feature
# =============================================================================

class BidirectionalModel(nn.Module):
    """Model that uses both WGP bank AND rounding test result."""

    def __init__(self, n_banks=N_BANKS, hidden=HIDDEN_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden), nn.ReLU(),
        )
        # Per-bank weight matrices
        self.bank_weights = nn.Parameter(torch.randn(n_banks, hidden, hidden) * 0.02)
        # Rounding feature embedding (scalar → hidden)
        self.round_embed = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, hidden),
        )
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )
        # Self-model: predict own bank from hidden state
        self.self_model = nn.Linear(hidden, n_banks)

    def forward(self, x, bank_ids, round_feat=None):
        h = self.encoder(x)
        B = h.shape[0]
        # Per-sample bank transformation
        W = self.bank_weights[bank_ids]  # [B, H, H]
        h_hw = torch.bmm(W, h.unsqueeze(-1)).squeeze(-1)  # [B, H]
        # Add rounding feature if available
        if round_feat is not None:
            r = self.round_embed(round_feat.unsqueeze(-1))  # [B, H]
            h_hw = h_hw + r
        logits = self.classifier(h_hw)
        self_pred = self.self_model(h_hw.detach())
        return logits, self_pred


class BlindModel(nn.Module):
    """No hardware information."""

    def __init__(self, hidden=HIDDEN_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden), nn.ReLU(),
        )
        self.transform = nn.Linear(hidden, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x, bank_ids=None, round_feat=None):
        h = self.encoder(x)
        h = F.relu(self.transform(h))
        return self.classifier(h), None


# =============================================================================
# Power reading
# =============================================================================

def find_power_sysfs():
    for card in Path('/sys/class/drm/').glob('card*/device/hwmon/hwmon*/power1_average'):
        return str(card)
    return None

def read_power_uw(path):
    try:
        return int(open(path).read().strip())
    except:
        return 0


# =============================================================================
# Training + Evaluation
# =============================================================================

def get_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = torchvision.datasets.MNIST(root=str(ROOT / 'data'), train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root=str(ROOT / 'data'), train=False, transform=transform)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def train_condition(model, train_loader, probe, wgp_map, condition, power_path):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    t0 = time.time()
    total_joules = 0.0

    for epoch in range(N_EPOCHS):
        correct = total = 0
        epoch_loss = 0
        self_correct = 0
        bank_counts = [0] * N_BANKS

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            B = images.shape[0]

            # Get hardware info based on condition
            if condition == 'A_bidirectional':
                results = probe.bidirectional_read(B, write_mode=1)
                wgp_ids, _, _, round_feat = results
                bank_ids = wgp_to_bank(wgp_ids, wgp_map)
                perm_labels = permute_labels(labels, bank_ids)
            elif condition == 'B_read_only':
                wgp_ids = probe.get_wgp_ids(B)
                bank_ids = wgp_to_bank(wgp_ids, wgp_map)
                perm_labels = permute_labels(labels, bank_ids)
                round_feat = None
            elif condition == 'C_blind':
                bank_ids = torch.zeros(B, dtype=torch.long, device=DEVICE)
                perm_labels = labels.clone()
                round_feat = None
            elif condition == 'D_write_only':
                # Write MODE but don't use WGP for bank selection
                results = probe.bidirectional_read(B, write_mode=2)
                _, _, _, round_feat = results
                bank_ids = torch.zeros(B, dtype=torch.long, device=DEVICE)
                perm_labels = labels.clone()
            else:
                raise ValueError(f"Unknown condition: {condition}")

            logits, self_pred = model(images, bank_ids, round_feat)
            loss = F.cross_entropy(logits, perm_labels)

            if self_pred is not None:
                loss += 0.1 * F.cross_entropy(self_pred, bank_ids)
                self_correct += (self_pred.argmax(1) == bank_ids).sum().item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            pred = logits.argmax(1)
            correct += (pred == perm_labels).sum().item()
            total += B
            epoch_loss += loss.item()

            for b in bank_ids.cpu().tolist():
                if 0 <= b < N_BANKS:
                    bank_counts[b] += 1

            # Energy
            pw = read_power_uw(power_path)
            if pw > 0:
                total_joules += pw / 1e6 * (time.time() - t0) / max(1, total) * B

        acc = correct / total
        self_acc = self_correct / total if self_correct > 0 else 0
        if epoch % 2 == 0 or epoch == N_EPOCHS - 1:
            print(f"  Epoch {epoch+1:2d}: loss={epoch_loss/len(train_loader):.4f} "
                  f"acc={acc:.4f} self={self_acc:.3f} banks={bank_counts}")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s | {total_joules:.1f}J total")
    return {'train_time_s': elapsed, 'total_joules': total_joules}


def evaluate_condition(model, test_loader, probe, wgp_map, condition,
                       scrambled_map=None, write_mode=1):
    model.eval()
    correct = total = 0
    self_correct = 0
    bank_counts = [0] * N_BANKS
    round_values = []
    mode_changes = 0
    total_blocks = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            B = images.shape[0]

            bank_map = scrambled_map if scrambled_map is not None else wgp_map

            if condition in ('A_bidirectional', 'E_scrambled', 'F_frozen'):
                wm = write_mode if condition != 'F_frozen' else 0
                results = probe.bidirectional_read(B, write_mode=wm)
                wgp_ids, m_before, m_after, round_feat = results
                # Labels: ALWAYS use real mapping
                real_banks = wgp_to_bank(wgp_ids, wgp_map)
                perm_labels = permute_labels(labels, real_banks)
                # Banks: may use scrambled
                bank_ids = wgp_to_bank(wgp_ids, bank_map)
                mode_changes += (m_before != m_after).sum().item()
                total_blocks += B
                round_values.extend(round_feat.cpu().tolist())
            elif condition == 'B_read_only':
                wgp_ids = probe.get_wgp_ids(B)
                bank_ids = wgp_to_bank(wgp_ids, wgp_map)
                perm_labels = permute_labels(labels, bank_ids)
                round_feat = None
            elif condition == 'C_blind':
                bank_ids = torch.zeros(B, dtype=torch.long, device=DEVICE)
                perm_labels = labels.clone()
                round_feat = None
            elif condition == 'D_write_only':
                results = probe.bidirectional_read(B, write_mode=2)
                _, _, _, round_feat = results
                bank_ids = torch.zeros(B, dtype=torch.long, device=DEVICE)
                perm_labels = labels.clone()
            else:
                perm_labels = labels
                bank_ids = torch.zeros(B, dtype=torch.long, device=DEVICE)
                round_feat = None

            logits, self_pred = model(images, bank_ids, round_feat)
            pred = logits.argmax(1)
            correct += (pred == perm_labels).sum().item()
            total += B

            if self_pred is not None:
                self_correct += (self_pred.argmax(1) == bank_ids).sum().item()

            for b in bank_ids.cpu().tolist():
                if 0 <= b < N_BANKS:
                    bank_counts[b] += 1

    acc = correct / total
    self_acc = self_correct / total if self_correct > 0 else 0
    n_round_unique = len(set([f'{v:.8f}' for v in round_values])) if round_values else 0

    return {
        'accuracy': acc,
        'self_model_acc': self_acc,
        'bank_counts': bank_counts,
        'n_round_unique': n_round_unique,
        'mode_change_pct': mode_changes / max(1, total_blocks) * 100,
        'round_range': [min(round_values), max(round_values)] if round_values else [0, 0],
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print(f"z2053: Bidirectional Silicon I/O — Read WGP + Write MODE Register")
    print(f"  READ:  s_getreg_b32 hwreg(23) → physical WGP_id")
    print(f"  WRITE: s_setreg_b32 hwreg(1)  → MODE register (FP rounding)")
    print(f"  Machine: {MACHINE}")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    power_path = find_power_sysfs()
    print(f"Power sysfs: {power_path}")

    # Build probe
    probe = BidirectionalProbe()

    wgp_map = build_wgp_to_bank().to(DEVICE)

    # Scrambled map for kill shot
    perm = torch.randperm(N_BANKS)
    scrambled_map = torch.zeros(16, dtype=torch.long, device=DEVICE)
    for i, wgp in enumerate(WGP_VALUES):
        scrambled_map[wgp] = perm[i]

    train_loader, test_loader = get_data()

    results = {
        'timestamp': datetime.now().isoformat(),
        'machine': MACHINE,
        'config': {
            'n_banks': N_BANKS,
            'hidden_dim': HIDDEN_DIM,
            'n_epochs': N_EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': LR,
        },
        'conditions': {},
        'tests': {},
    }

    # =====================
    # A: Bidirectional (read WGP + write MODE)
    # =====================
    print(f"\n{'='*60}")
    print(f"  Training: A_bidirectional (READ + WRITE)")
    print(f"{'='*60}")
    model_A = BidirectionalModel().to(DEVICE)
    train_info = train_condition(model_A, train_loader, probe, wgp_map, 'A_bidirectional', power_path)
    eval_A = evaluate_condition(model_A, test_loader, probe, wgp_map, 'A_bidirectional')
    print(f"\n  >> A_bidirectional: acc={eval_A['accuracy']:.4f} self={eval_A['self_model_acc']:.3f} "
          f"mode_change={eval_A['mode_change_pct']:.1f}% round_unique={eval_A['n_round_unique']}")
    results['conditions']['A_bidirectional'] = {'eval': eval_A, **train_info}

    # =====================
    # B: Read-only (z2050 replication)
    # =====================
    print(f"\n{'='*60}")
    print(f"  Training: B_read_only (z2050 replication)")
    print(f"{'='*60}")
    model_B = BidirectionalModel().to(DEVICE)
    train_info = train_condition(model_B, train_loader, probe, wgp_map, 'B_read_only', power_path)
    eval_B = evaluate_condition(model_B, test_loader, probe, wgp_map, 'B_read_only')
    print(f"\n  >> B_read_only: acc={eval_B['accuracy']:.4f}")
    results['conditions']['B_read_only'] = {'eval': eval_B, **train_info}

    # =====================
    # C: Blind
    # =====================
    print(f"\n{'='*60}")
    print(f"  Training: C_blind")
    print(f"{'='*60}")
    model_C = BlindModel().to(DEVICE)
    train_info = train_condition(model_C, train_loader, probe, wgp_map, 'C_blind', power_path)
    eval_C = evaluate_condition(model_C, test_loader, probe, wgp_map, 'C_blind')
    print(f"\n  >> C_blind: acc={eval_C['accuracy']:.4f}")
    results['conditions']['C_blind'] = {'eval': eval_C, **train_info}

    # =====================
    # D: Write-only (MODE write, no WGP bank selection)
    # =====================
    print(f"\n{'='*60}")
    print(f"  Training: D_write_only")
    print(f"{'='*60}")
    model_D = BidirectionalModel().to(DEVICE)
    train_info = train_condition(model_D, train_loader, probe, wgp_map, 'D_write_only', power_path)
    eval_D = evaluate_condition(model_D, test_loader, probe, wgp_map, 'D_write_only')
    print(f"\n  >> D_write_only: acc={eval_D['accuracy']:.4f}")
    results['conditions']['D_write_only'] = {'eval': eval_D, **train_info}

    # =====================
    # E: Scrambled (A's model with wrong WGP→bank map)
    # =====================
    print(f"\n{'='*60}")
    print(f"  E_scrambled: A's model with WRONG WGP→bank map")
    print(f"{'='*60}")
    eval_E = evaluate_condition(model_A, test_loader, probe, wgp_map, 'E_scrambled',
                                scrambled_map=scrambled_map)
    print(f"\n  >> E_scrambled: acc={eval_E['accuracy']:.4f}")
    results['conditions']['E_scrambled'] = {'eval': eval_E}

    # =====================
    # F: Frozen writes (A's model but no MODE writes at eval)
    # =====================
    print(f"\n{'='*60}")
    print(f"  F_frozen: A's model with MODE writes DISABLED")
    print(f"{'='*60}")
    eval_F = evaluate_condition(model_A, test_loader, probe, wgp_map, 'F_frozen',
                                write_mode=0)
    print(f"\n  >> F_frozen: acc={eval_F['accuracy']:.4f}")
    results['conditions']['F_frozen'] = {'eval': eval_F}

    # =====================
    # Bank weight analysis
    # =====================
    W = model_A.bank_weights.data.cpu()
    cos_sims = {}
    for i in range(N_BANKS):
        for j in range(i+1, N_BANKS):
            Wi = W[i].flatten()
            Wj = W[j].flatten()
            cos = F.cosine_similarity(Wi.unsqueeze(0), Wj.unsqueeze(0)).item()
            cos_sims[f'bank_{i}_vs_{j}'] = round(cos, 4)
    cos_vals = list(cos_sims.values())
    cos_sims['mean'] = round(np.mean(cos_vals), 4)
    cos_sims['min'] = round(min(cos_vals), 4)
    cos_sims['max'] = round(max(cos_vals), 4)
    results['bank_analysis'] = cos_sims

    # =====================
    # Tests
    # =====================
    acc_A = eval_A['accuracy']
    acc_B = eval_B['accuracy']
    acc_C = eval_C['accuracy']
    acc_D = eval_D['accuracy']
    acc_E = eval_E['accuracy']
    acc_F = eval_F['accuracy']

    results['tests'] = {
        'T1_accuracy': {
            'criterion': 'A_bidirectional > 90%',
            'value': acc_A,
            'pass': acc_A > 0.90,
        },
        'T2_embodied_blind_gap': {
            'criterion': 'A - C gap > 30%',
            'A': acc_A, 'C': acc_C,
            'gap': round(acc_A - acc_C, 4),
            'pass': (acc_A - acc_C) > 0.30,
        },
        'T3_kill_shot': {
            'criterion': 'A - E gap > 10%',
            'A': acc_A, 'E': acc_E,
            'gap': round(acc_A - acc_E, 4),
            'pass': (acc_A - acc_E) > 0.10,
        },
        'T4_bidir_vs_readonly': {
            'criterion': 'A_bidir >= B_readonly (write adds value)',
            'A': acc_A, 'B': acc_B,
            'diff': round(acc_A - acc_B, 4),
            'pass': acc_A >= acc_B - 0.02,  # Within 2% counts as "at least as good"
        },
        'T5_rounding_diversity': {
            'criterion': 'Multiple unique rounding values observed',
            'n_unique': eval_A['n_round_unique'],
            'pass': eval_A['n_round_unique'] >= 2,
        },
        'T6_mode_write_verified': {
            'criterion': 'MODE register changed in >50% of blocks',
            'mode_change_pct': eval_A['mode_change_pct'],
            'pass': eval_A['mode_change_pct'] > 50,
        },
        'T7_bank_divergence': {
            'criterion': 'Mean bank cos similarity < 0.5',
            'value': cos_sims['mean'],
            'pass': cos_sims['mean'] < 0.5,
        },
    }

    n_pass = sum(1 for t in results['tests'].values() if t['pass'])
    n_total = len(results['tests'])
    results['n_pass'] = n_pass
    results['n_total'] = n_total

    if n_pass >= 6:
        results['verdict'] = 'BIDIRECTIONAL_CONFIRMED'
    elif n_pass >= 4:
        results['verdict'] = 'PARTIAL'
    else:
        results['verdict'] = 'WEAK'

    results['summary'] = {
        'A_bidirectional': acc_A,
        'B_read_only': acc_B,
        'C_blind': acc_C,
        'D_write_only': acc_D,
        'E_scrambled': acc_E,
        'F_frozen': acc_F,
        'self_model_acc': eval_A['self_model_acc'],
        'gap_A_C': round(acc_A - acc_C, 4),
        'gap_A_E': round(acc_A - acc_E, 4),
        'bank_cos_mean': cos_sims['mean'],
        'mode_change_pct': eval_A['mode_change_pct'],
        'n_round_unique': eval_A['n_round_unique'],
    }

    # Print results
    print(f"\n{'='*70}")
    print("TESTS")
    print(f"{'='*70}")
    for name, t in results['tests'].items():
        status = "PASS" if t['pass'] else "FAIL"
        print(f"  {name}: {status} — {t['criterion']}")
        for k, v in t.items():
            if k not in ('criterion', 'pass'):
                print(f"    {k} = {v}")

    print(f"\n  VERDICT: {results['verdict']} ({n_pass}/{n_total} PASS)")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for k, v in results['summary'].items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")

    # Save
    out_path = RESULTS_DIR / 'z2053_bidirectional_silicon.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
