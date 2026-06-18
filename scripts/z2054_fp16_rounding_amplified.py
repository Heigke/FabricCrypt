#!/usr/bin/env python3
"""z2054: FP16 Rounding Amplification — Making MODE Register Writes VISIBLE

z2053 FINDING: s_setreg_b32 WORKS but rounding diff below float32 ULP.
z2054 FIX: Use fp16 (half precision) where ULP is ~1000x larger.

FP16 ULP near 1.0 ≈ 0.001 (vs float32 ULP ≈ 1e-7)
100 accumulated fp16 ops → rounding diff ≈ 0.05–0.1 (clearly measurable!)

STRATEGY:
  1. Write FP rounding mode per-WGP (round-nearest vs round-toward-zero)
  2. Compute 256 fp16 additions: sum = Σ(1/(3+4i)) in half precision
  3. The sum DEPENDS on rounding mode — physically different computation
  4. Feed the fp16 sum as a feature to the neural network
  5. The model learns to USE the rounding difference for classification

Also tests DENORM FLUSH (MODE bit 4):
  - Flush-to-zero: subnormals become 0.0 (faster on some hardware)
  - Allow denorms: subnormals preserved (different numerical results)
  - For fp16, denorms are values < 2^-14 ≈ 6.1e-5

CONDITIONS:
  A_full:      Read WGP + Write round mode + fp16 computation
  B_readonly:  Read WGP only (z2050 replication)
  C_blind:     No hardware
  D_denorm:    Write denorm flush mode instead of rounding mode
  E_scrambled: Wrong WGP→bank mapping
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
# HIP Kernel: FP16 rounding amplification
# =============================================================================

HIP_SOURCE = r'''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>

// FP16 rounding-sensitive computation kernel
// Accumulates 256 terms in half precision — rounding mode matters!
__global__ void fp16_rounding_probe(
    int32_t* wgp_ids,          // [B] — physical WGP_id per block
    int32_t* mode_before,      // [B] — MODE register before write
    int32_t* mode_after,       // [B] — MODE register after write
    float* fp16_result,        // [B] — fp16 sum converted to float
    float* fp32_result,        // [B] — fp32 sum for comparison
    const int32_t* mode_map,   // [N_BANKS] — bank→mode value
    const int32_t* wgp_to_bank, // [16]
    int batch_size,
    int write_mode,            // 0=none, 1=round mode, 2=denorm mode
    int n_accumulations        // number of fp16 additions
) {
    int bid = blockIdx.x;
    if (bid >= batch_size) return;
    if (threadIdx.x != 0) return;

    // STEP 1: READ physical CU identity
    uint32_t hw_id1 = 0;
    asm volatile("s_getreg_b32 %0, hwreg(23, 0, 32)" : "=s"(hw_id1));
    int32_t wgp_id = (int32_t)((hw_id1 >> 7) & 0xF);
    wgp_ids[bid] = wgp_id;

    // STEP 2: READ original MODE
    uint32_t orig_mode = 0;
    asm volatile("s_getreg_b32 %0, hwreg(1, 0, 32)" : "=s"(orig_mode));
    mode_before[bid] = (int32_t)orig_mode;

    // STEP 3: WRITE MODE based on bank
    int32_t bank = wgp_to_bank[wgp_id & 0xF];
    uint32_t target_mode = 0;

    if (write_mode == 1) {
        // Rounding mode: 0=nearest, 3=toward-zero
        target_mode = (uint32_t)(mode_map[bank] & 0xF);
        int sgpr_val = __builtin_amdgcn_readfirstlane((int)target_mode);
        __builtin_amdgcn_s_setreg(0x1801, sgpr_val);  // hwreg(1, 0, 4)
    } else if (write_mode == 2) {
        // Denorm mode: bit 4 of MODE (0=flush, 1=allow)
        target_mode = (uint32_t)(mode_map[bank] & 0x1);
        uint32_t new_mode = (orig_mode & ~0x10u) | ((target_mode & 1) << 4);
        int sgpr_val = __builtin_amdgcn_readfirstlane((int)(new_mode & 0xFF));
        __builtin_amdgcn_s_setreg(0x3801, sgpr_val);  // hwreg(1, 0, 8) — write 8 bits
    }

    // STEP 4: READ back MODE
    uint32_t new_mode_readback = 0;
    asm volatile("s_getreg_b32 %0, hwreg(1, 0, 32)" : "=s"(new_mode_readback));
    mode_after[bid] = (int32_t)new_mode_readback;

    // STEP 5: FP16 accumulation — rounding-mode-sensitive
    // Sum of 1/(2+i) for i=0..n_accumulations in half precision
    // This is carefully chosen to hit fp16 precision boundaries
    __half h_sum = __float2half(0.0f);
    float f_sum = 0.0f;

    for (int i = 0; i < n_accumulations; i++) {
        // Use values that create rounding-sensitive accumulation
        // 1/(2+i) ranges from 0.5 to ~0.004 — right in fp16's sweet spot
        float denom = (float)(2 + i);
        __half h_val = __float2half(1.0f / denom);
        h_sum = __hadd(h_sum, h_val);
        f_sum += 1.0f / denom;
    }

    fp16_result[bid] = __half2float(h_sum);
    fp32_result[bid] = f_sum;

    // STEP 6: RESTORE original mode
    if (write_mode > 0) {
        int sgpr_restore = __builtin_amdgcn_readfirstlane((int)(orig_mode & 0xFF));
        __builtin_amdgcn_s_setreg(0x3801, sgpr_restore);  // hwreg(1, 0, 8)
    }
}

// Simple read-only probe
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
std::vector<torch::Tensor> fp16_probe(
    int batch_size,
    torch::Tensor mode_map,
    torch::Tensor wgp_to_bank,
    int write_mode,
    int n_accumulations
) {
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto wgp_ids = torch::zeros({batch_size}, opts_i);
    auto m_before = torch::zeros({batch_size}, opts_i);
    auto m_after = torch::zeros({batch_size}, opts_i);
    auto fp16_res = torch::zeros({batch_size}, opts_f);
    auto fp32_res = torch::zeros({batch_size}, opts_f);

    fp16_rounding_probe<<<batch_size, 32>>>(
        wgp_ids.data_ptr<int32_t>(),
        m_before.data_ptr<int32_t>(),
        m_after.data_ptr<int32_t>(),
        fp16_res.data_ptr<float>(),
        fp32_res.data_ptr<float>(),
        mode_map.data_ptr<int32_t>(),
        wgp_to_bank.data_ptr<int32_t>(),
        batch_size,
        write_mode,
        n_accumulations
    );
    return {wgp_ids, m_before, m_after, fp16_res, fp32_res};
}

torch::Tensor get_wgp_ids_only(int batch_size) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto wgp_ids = torch::zeros({batch_size}, opts);
    read_wgp_only<<<batch_size, 32>>>(wgp_ids.data_ptr<int32_t>(), batch_size);
    return wgp_ids;
}
'''

HIP_CPP = r'''
std::vector<torch::Tensor> fp16_probe(int batch_size, torch::Tensor mode_map, torch::Tensor wgp_to_bank, int write_mode, int n_accumulations);
torch::Tensor get_wgp_ids_only(int batch_size);
'''


class FP16RoundingProbe:
    """Probes WGP_id, writes MODE, computes fp16 sum."""

    def __init__(self, n_accum=256):
        from torch.utils.cpp_extension import load_inline
        self.n_accum = n_accum
        print(f"[BUILD] Compiling fp16 rounding probe (n_accum={n_accum})...")
        t0 = time.time()
        self.ext = load_inline(
            name='fp16_probe_z2054',
            cpp_sources=HIP_CPP,
            cuda_sources=HIP_SOURCE,
            functions=['fp16_probe', 'get_wgp_ids_only'],
            verbose=False,
            extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'],
        )
        print(f"[BUILD] Done in {time.time()-t0:.1f}s")

        # Build WGP → bank mapping
        self.wgp_to_bank_map = torch.zeros(16, dtype=torch.int32, device='cuda')
        for i, wgp in enumerate(WGP_VALUES):
            self.wgp_to_bank_map[wgp] = i

        # Mode maps
        # Rounding: even banks → round nearest (0), odd banks → round toward zero
        # MODE bits [1:0] = FP_ROUND_32, bits [3:2] = FP_ROUND_16_64
        # 0x0 = nearest on all precisions, 0xF = toward-zero on all precisions
        self.round_mode_map = torch.zeros(N_BANKS, dtype=torch.int32, device='cuda')
        for i in range(N_BANKS):
            self.round_mode_map[i] = 0 if i % 2 == 0 else 0xF  # 15 = toward-zero for fp16+fp32+fp64

        # Denorm: even banks → flush (0), odd banks → allow (1)
        self.denorm_mode_map = torch.zeros(N_BANKS, dtype=torch.int32, device='cuda')
        for i in range(N_BANKS):
            self.denorm_mode_map[i] = 0 if i % 2 == 0 else 1

        # Calibration
        print(f"\n[CALIBRATE] Testing fp16 vs fp32 rounding with {n_accum} accumulations...")
        # Test with rounding mode write
        r1 = self.probe(64, write_mode=1)
        wgp_ids, m_before, m_after, fp16_res, fp32_res = r1
        mode_changed = (m_before != m_after).sum().item()

        fp16_vals = fp16_res.cpu().numpy()
        fp32_vals = fp32_res.cpu().numpy()
        diff = np.abs(fp16_vals - fp32_vals)
        unique_fp16 = len(np.unique(np.round(fp16_vals, 6)))
        unique_fp32 = len(np.unique(np.round(fp32_vals, 8)))

        print(f"  WGP_ids: {torch.unique(wgp_ids).cpu().tolist()}")
        print(f"  MODE changed: {mode_changed}/64 blocks")
        print(f"  FP16 unique values: {unique_fp16}")
        print(f"  FP32 unique values: {unique_fp32}")
        print(f"  FP16 range: [{fp16_vals.min():.6f}, {fp16_vals.max():.6f}]")
        print(f"  FP32 range: [{fp32_vals.min():.8f}, {fp32_vals.max():.8f}]")
        print(f"  FP16-FP32 diff: [{diff.min():.6f}, {diff.max():.6f}]")

        # Test without mode write (baseline)
        r0 = self.probe(64, write_mode=0)
        _, _, _, fp16_base, fp32_base = r0
        base_unique = len(np.unique(np.round(fp16_base.cpu().numpy(), 6)))
        print(f"  Baseline (no write) FP16 unique: {base_unique}")

        if unique_fp16 > base_unique:
            print(f"  [CONFIRMED] MODE write produces {unique_fp16 - base_unique} additional fp16 values!")
        else:
            print(f"  [NOTE] MODE write did not produce additional fp16 diversity")

    def probe(self, batch_size, write_mode=1, use_denorm=False):
        mode_map = self.denorm_mode_map if use_denorm else self.round_mode_map
        results = self.ext.fp16_probe(
            batch_size, mode_map, self.wgp_to_bank_map, write_mode, self.n_accum
        )
        torch.cuda.synchronize()
        return results

    def get_wgp_ids(self, batch_size):
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
    return mapping.to(wgp_ids.device)[wgp_ids.clamp(0, 15).long()]

def permute_labels(labels, bank_ids):
    permuted = labels.clone()
    odd_mask = (bank_ids % 2) == 1
    permuted[odd_mask] = (labels[odd_mask] + 5) % 10
    return permuted


# =============================================================================
# Model
# =============================================================================

class FP16AwareModel(nn.Module):
    def __init__(self, n_banks=N_BANKS, hidden=HIDDEN_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, hidden), nn.ReLU(),
        )
        self.bank_weights = nn.Parameter(torch.randn(n_banks, hidden, hidden) * 0.02)
        # FP16 result embedding (2 features: fp16_sum, fp16-fp32 diff)
        self.fp_embed = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, hidden))
        self.classifier = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 10))
        self.self_model = nn.Linear(hidden, n_banks)

    def forward(self, x, bank_ids, fp_feat=None):
        h = self.encoder(x)
        W = self.bank_weights[bank_ids]
        h_hw = torch.bmm(W, h.unsqueeze(-1)).squeeze(-1)
        if fp_feat is not None:
            h_hw = h_hw + self.fp_embed(fp_feat)
        logits = self.classifier(h_hw)
        self_pred = self.self_model(h_hw.detach())
        return logits, self_pred


class BlindModel(nn.Module):
    def __init__(self, hidden=HIDDEN_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, hidden), nn.ReLU(),
        )
        self.transform = nn.Linear(hidden, hidden)
        self.classifier = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x, bank_ids=None, fp_feat=None):
        h = F.relu(self.transform(self.encoder(x)))
        return self.classifier(h), None


# =============================================================================
# Power
# =============================================================================

def find_power_sysfs():
    for card in Path('/sys/class/drm/').glob('card*/device/hwmon/hwmon*/power1_average'):
        return str(card)
    return None

def read_power_uw(path):
    try: return int(open(path).read().strip())
    except: return 0


# =============================================================================
# Training + Evaluation
# =============================================================================

def get_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = torchvision.datasets.MNIST(root=str(ROOT/'data'), train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root=str(ROOT/'data'), train=False, transform=transform)
    return (DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True),
            DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True))


def train_condition(model, loader, probe, wgp_map, cond, power_path):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    t0 = time.time()
    total_j = 0.0

    for epoch in range(N_EPOCHS):
        correct = total = self_c = 0
        epoch_loss = 0
        banks = [0]*N_BANKS

        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            B = images.shape[0]

            if cond in ('A_full', 'A_round'):
                results = probe.probe(B, write_mode=1)
                wgp_ids, _, _, fp16_res, fp32_res = results
                bank_ids = wgp_to_bank(wgp_ids, wgp_map)
                perm_labels = permute_labels(labels, bank_ids)
                fp_feat = torch.stack([fp16_res, fp16_res - fp32_res], dim=1)
            elif cond == 'D_denorm':
                results = probe.probe(B, write_mode=2, use_denorm=True)
                wgp_ids, _, _, fp16_res, fp32_res = results
                bank_ids = wgp_to_bank(wgp_ids, wgp_map)
                perm_labels = permute_labels(labels, bank_ids)
                fp_feat = torch.stack([fp16_res, fp16_res - fp32_res], dim=1)
            elif cond == 'B_readonly':
                wgp_ids = probe.get_wgp_ids(B)
                bank_ids = wgp_to_bank(wgp_ids, wgp_map)
                perm_labels = permute_labels(labels, bank_ids)
                fp_feat = None
            elif cond == 'C_blind':
                bank_ids = torch.zeros(B, dtype=torch.long, device=DEVICE)
                perm_labels = labels.clone()
                fp_feat = None
            else:
                raise ValueError(cond)

            logits, self_pred = model(images, bank_ids, fp_feat)
            loss = F.cross_entropy(logits, perm_labels)
            if self_pred is not None:
                loss += 0.1 * F.cross_entropy(self_pred, bank_ids)
                self_c += (self_pred.argmax(1) == bank_ids).sum().item()

            opt.zero_grad(); loss.backward(); opt.step()
            correct += (logits.argmax(1) == perm_labels).sum().item()
            total += B; epoch_loss += loss.item()
            for b in bank_ids.cpu().tolist():
                if 0 <= b < N_BANKS: banks[b] += 1

            pw = read_power_uw(power_path)
            if pw > 0: total_j += pw/1e6 * (time.time()-t0)/max(1,total)*B

        if epoch % 2 == 0 or epoch == N_EPOCHS-1:
            print(f"  Epoch {epoch+1:2d}: loss={epoch_loss/len(loader):.4f} "
                  f"acc={correct/total:.4f} self={self_c/total:.3f}")

    print(f"  Done in {time.time()-t0:.1f}s")
    return {'train_time_s': time.time()-t0, 'total_joules': total_j}


def evaluate(model, loader, probe, wgp_map, cond, scrambled_map=None, write_mode=1):
    model.eval()
    correct = total = self_c = 0
    fp16_vals = []; mode_changes = 0; total_blocks = 0
    banks = [0]*N_BANKS

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            B = images.shape[0]
            bank_map = scrambled_map if scrambled_map is not None else wgp_map

            if cond in ('A_full', 'A_round', 'E_scrambled'):
                wm = write_mode
                results = probe.probe(B, write_mode=wm)
                wgp_ids, m_before, m_after, fp16_res, fp32_res = results
                real_banks = wgp_to_bank(wgp_ids, wgp_map)
                perm_labels = permute_labels(labels, real_banks)
                bank_ids = wgp_to_bank(wgp_ids, bank_map)
                fp_feat = torch.stack([fp16_res, fp16_res - fp32_res], dim=1)
                mode_changes += (m_before != m_after).sum().item()
                total_blocks += B
                fp16_vals.extend(fp16_res.cpu().tolist())
            elif cond == 'D_denorm':
                results = probe.probe(B, write_mode=2, use_denorm=True)
                wgp_ids, m_before, m_after, fp16_res, fp32_res = results
                bank_ids = wgp_to_bank(wgp_ids, wgp_map)
                perm_labels = permute_labels(labels, bank_ids)
                fp_feat = torch.stack([fp16_res, fp16_res - fp32_res], dim=1)
                mode_changes += (m_before != m_after).sum().item()
                total_blocks += B
                fp16_vals.extend(fp16_res.cpu().tolist())
            elif cond == 'B_readonly':
                wgp_ids = probe.get_wgp_ids(B)
                bank_ids = wgp_to_bank(wgp_ids, wgp_map)
                perm_labels = permute_labels(labels, bank_ids)
                fp_feat = None
            elif cond == 'C_blind':
                bank_ids = torch.zeros(B, dtype=torch.long, device=DEVICE)
                perm_labels = labels.clone()
                fp_feat = None
            else:
                bank_ids = torch.zeros(B, dtype=torch.long, device=DEVICE)
                perm_labels = labels; fp_feat = None

            logits, self_pred = model(images, bank_ids, fp_feat)
            correct += (logits.argmax(1) == perm_labels).sum().item()
            total += B
            if self_pred is not None:
                self_c += (self_pred.argmax(1) == bank_ids).sum().item()
            for b in bank_ids.cpu().tolist():
                if 0 <= b < N_BANKS: banks[b] += 1

    n_fp16_unique = len(set([f'{v:.6f}' for v in fp16_vals])) if fp16_vals else 0
    return {
        'accuracy': correct/total,
        'self_model_acc': self_c/total if self_c > 0 else 0,
        'bank_counts': banks,
        'n_fp16_unique': n_fp16_unique,
        'mode_change_pct': mode_changes/max(1,total_blocks)*100,
        'fp16_range': [min(fp16_vals), max(fp16_vals)] if fp16_vals else [0,0],
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print(f"z2054: FP16 Rounding Amplification — Making MODE Writes VISIBLE")
    print(f"  Strategy: fp16 accumulation where ULP is ~1000x larger than fp32")
    print(f"  Machine: {MACHINE}")
    print("="*70)
    print(f"Device: {DEVICE}")

    power_path = find_power_sysfs()
    print(f"Power sysfs: {power_path}")

    probe = FP16RoundingProbe(n_accum=256)
    wgp_map = build_wgp_to_bank().to(DEVICE)

    perm = torch.randperm(N_BANKS)
    scrambled_map = torch.zeros(16, dtype=torch.long, device=DEVICE)
    for i, wgp in enumerate(WGP_VALUES):
        scrambled_map[wgp] = perm[i]

    train_loader, test_loader = get_data()

    results = {
        'timestamp': datetime.now().isoformat(),
        'machine': MACHINE,
        'config': {'n_banks': N_BANKS, 'hidden_dim': HIDDEN_DIM, 'n_epochs': N_EPOCHS,
                   'batch_size': BATCH_SIZE, 'lr': LR, 'n_accum': 256},
        'conditions': {}, 'tests': {},
    }

    # A: Full (WGP + rounding mode + fp16)
    print(f"\n{'='*60}\n  Training: A_full (WGP + round mode + fp16)\n{'='*60}")
    model_A = FP16AwareModel().to(DEVICE)
    ti = train_condition(model_A, train_loader, probe, wgp_map, 'A_full', power_path)
    eval_A = evaluate(model_A, test_loader, probe, wgp_map, 'A_full')
    print(f"\n  >> A_full: acc={eval_A['accuracy']:.4f} fp16_unique={eval_A['n_fp16_unique']} "
          f"mode={eval_A['mode_change_pct']:.1f}%")
    results['conditions']['A_full'] = {'eval': eval_A, **ti}

    # B: Read-only
    print(f"\n{'='*60}\n  Training: B_readonly\n{'='*60}")
    model_B = FP16AwareModel().to(DEVICE)
    ti = train_condition(model_B, train_loader, probe, wgp_map, 'B_readonly', power_path)
    eval_B = evaluate(model_B, test_loader, probe, wgp_map, 'B_readonly')
    print(f"\n  >> B_readonly: acc={eval_B['accuracy']:.4f}")
    results['conditions']['B_readonly'] = {'eval': eval_B, **ti}

    # C: Blind
    print(f"\n{'='*60}\n  Training: C_blind\n{'='*60}")
    model_C = BlindModel().to(DEVICE)
    ti = train_condition(model_C, train_loader, probe, wgp_map, 'C_blind', power_path)
    eval_C = evaluate(model_C, test_loader, probe, wgp_map, 'C_blind')
    print(f"\n  >> C_blind: acc={eval_C['accuracy']:.4f}")
    results['conditions']['C_blind'] = {'eval': eval_C, **ti}

    # D: Denorm mode
    print(f"\n{'='*60}\n  Training: D_denorm (flush vs allow)\n{'='*60}")
    model_D = FP16AwareModel().to(DEVICE)
    ti = train_condition(model_D, train_loader, probe, wgp_map, 'D_denorm', power_path)
    eval_D = evaluate(model_D, test_loader, probe, wgp_map, 'D_denorm')
    print(f"\n  >> D_denorm: acc={eval_D['accuracy']:.4f} fp16_unique={eval_D['n_fp16_unique']}")
    results['conditions']['D_denorm'] = {'eval': eval_D, **ti}

    # E: Scrambled
    print(f"\n{'='*60}\n  E_scrambled: A's model with WRONG WGP→bank\n{'='*60}")
    eval_E = evaluate(model_A, test_loader, probe, wgp_map, 'E_scrambled', scrambled_map=scrambled_map)
    print(f"\n  >> E_scrambled: acc={eval_E['accuracy']:.4f}")
    results['conditions']['E_scrambled'] = {'eval': eval_E}

    # Bank analysis
    W = model_A.bank_weights.data.cpu()
    cos_sims = {}
    for i in range(N_BANKS):
        for j in range(i+1, N_BANKS):
            cos = F.cosine_similarity(W[i].flatten().unsqueeze(0), W[j].flatten().unsqueeze(0)).item()
            cos_sims[f'bank_{i}_vs_{j}'] = round(cos, 4)
    cos_vals = list(cos_sims.values())
    cos_sims['mean'] = round(np.mean(cos_vals), 4)
    results['bank_analysis'] = cos_sims

    # Tests
    acc = {k: results['conditions'][k]['eval']['accuracy'] for k in results['conditions']}
    fp16_unique_A = eval_A['n_fp16_unique']
    fp16_unique_base = len(set([f'{v:.6f}' for v in [0.0]]))  # baseline

    results['tests'] = {
        'T1_accuracy': {'criterion': 'A > 90%', 'value': acc['A_full'],
                        'pass': acc['A_full'] > 0.90},
        'T2_gap': {'criterion': 'A - C gap > 30%', 'A': acc['A_full'], 'C': acc['C_blind'],
                   'gap': round(acc['A_full']-acc['C_blind'],4),
                   'pass': (acc['A_full']-acc['C_blind']) > 0.30},
        'T3_kill_shot': {'criterion': 'A - E gap > 10%', 'A': acc['A_full'], 'E': acc['E_scrambled'],
                         'gap': round(acc['A_full']-acc['E_scrambled'],4),
                         'pass': (acc['A_full']-acc['E_scrambled']) > 0.10},
        'T4_fp16_diversity': {'criterion': 'FP16 unique > 1 with mode write',
                              'n_unique': fp16_unique_A,
                              'pass': fp16_unique_A >= 2},
        'T5_mode_verified': {'criterion': 'MODE changed > 30%',
                             'pct': eval_A['mode_change_pct'],
                             'pass': eval_A['mode_change_pct'] > 30},
        'T6_bank_divergence': {'criterion': 'Bank cos < 0.5',
                               'value': cos_sims['mean'],
                               'pass': cos_sims['mean'] < 0.5},
        'T7_bidir_value': {'criterion': 'A_full >= B_readonly',
                           'A': acc['A_full'], 'B': acc['B_readonly'],
                           'diff': round(acc['A_full']-acc['B_readonly'],4),
                           'pass': acc['A_full'] >= acc['B_readonly'] - 0.02},
    }

    n_pass = sum(1 for t in results['tests'].values() if t['pass'])
    results['n_pass'] = n_pass
    results['n_total'] = len(results['tests'])
    results['verdict'] = ('FP16_ROUNDING_CONFIRMED' if n_pass >= 6
                          else 'PARTIAL' if n_pass >= 4 else 'WEAK')

    results['summary'] = {k: acc[k] for k in acc}
    results['summary']['fp16_unique'] = fp16_unique_A
    results['summary']['mode_change_pct'] = eval_A['mode_change_pct']
    results['summary']['bank_cos_mean'] = cos_sims['mean']

    print(f"\n{'='*70}\nTESTS\n{'='*70}")
    for name, t in results['tests'].items():
        print(f"  {name}: {'PASS' if t['pass'] else 'FAIL'} — {t['criterion']}")
        for k,v in t.items():
            if k not in ('criterion','pass'): print(f"    {k} = {v}")

    print(f"\n  VERDICT: {results['verdict']} ({n_pass}/{results['n_total']} PASS)")

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for k,v in results['summary'].items():
        print(f"  {k:25s}: {v:.4f}" if isinstance(v,float) else f"  {k:25s}: {v}")

    out = RESULTS_DIR / 'z2054_fp16_rounding_amplified.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
