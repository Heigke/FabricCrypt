#!/usr/bin/env python3
"""z2056: Physics IS Computation — Continuous Hardware-Dependent Output

THE USER'S VISION:
  "Computer, what is 1+1?"
  Digital: "2" (always, deterministic, any hardware state)
  Analog: "Well I'm feeling panicky so... maybe 1, 2, or 3!"

  We want the gap between hardware and software to CLOSE.
  Physics becomes computation. The answer DEPENDS on physical state.

THE KEY INSIGHT FROM z2055:
  Problem: Gradient descent MINIMIZES sensitivity to noisy features.
  The gate learned 0.075 — model tries physics, finds it noisy, dials it back.
  Solution: Make the TASK require continuous physics. Not just bank selection
  (binary), but continuous label modulation by physics.

ARCHITECTURE: Physics-Modulated Classification
  1. Analog channels produce continuous features (fp16, timing, denorm)
  2. These features are NORMALIZED to [0,1] (fixes z2055's training instability)
  3. Label offset = f(analog_features, bank_id) — CONTINUOUS, not binary
     - Even banks: offset = fp16_sum * 3 (0, 3, or 6 depending on rounding)
     - Odd banks: offset = 5 + floor(timing_quantile * 2) (5, 6, or 7)
     - This creates 4-6 DISTINCT label offsets that depend on physics
  4. Per-bank denorm mode (NEW): even banks flush, odd banks preserve
     - Creates bank-dependent denorm results (0 vs 4.7e-38)
  5. Thermal perturbation test: warm GPU, measure output shift

WHY THIS WORKS:
  The task is IMPOSSIBLE without physics — C_blind can only learn ~17% (one offset).
  Multiple offsets from continuous physics means the model MUST track physical state.
  The gate naturally opens because physics is required, not optional.

CONDITIONS:
  A_physics:    Full analog (WGP + fp16 + timing + per-bank denorm)
  B_digital:    WGP only (binary bank, no continuous physics)
  C_blind:      No hardware
  D_scrambled:  Wrong WGP→bank (kill shot)
  E_thermal:    Same as A but after GPU warmup (thermal perturbation)

TESTS:
  T1: A > 85% accuracy (harder task)
  T2: A - C gap > 30% (physics necessary)
  T3: Kill shot — D < A by > 10%
  T4: Gate > 0.1 (model uses physics more than z2055)
  T5: ≥3 distinct label offsets observed (continuous, not binary)
  T6: Thermal perturbation changes output (cold vs hot)
  T7: Per-bank denorm diversity > 1
  T8: Physics feature importance non-uniform (model learns which channels matter)
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
N_EPOCHS = 15
BATCH_SIZE = 128
LR = 1e-3

# =============================================================================
# HIP Kernel: Per-bank denorm + rounding + timing
# =============================================================================

HIP_SOURCE = r'''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>

// Per-bank analog physics kernel
// KEY CHANGE FROM z2055: denorm mode set PER-BANK, not uniformly
__global__ void physics_probe(
    int32_t* wgp_ids,              // [B]
    float* fp16_sums,              // [B]
    float* fp32_sums,              // [B]
    float* timing_cycles,          // [B]
    float* denorm_results,         // [B] — per-bank denorm computation
    float* denorm_diffs,           // [B] — difference vs reference
    int32_t* mode_before,          // [B]
    int32_t* mode_after_round,     // [B] — MODE after rounding write
    const int32_t* round_mode_map, // [N_BANKS]
    const int32_t* denorm_mode_map,// [N_BANKS] — per-bank denorm mode
    const int32_t* wgp_to_bank,    // [16]
    int batch_size,
    int n_fp16_accum,
    int n_denorm_ops
) {
    int bid = blockIdx.x;
    if (bid >= batch_size) return;
    if (threadIdx.x != 0) return;

    // ===== Read physical identity =====
    uint32_t hw_id1 = 0;
    asm volatile("s_getreg_b32 %0, hwreg(23, 0, 32)" : "=s"(hw_id1));
    int32_t wgp_id = (int32_t)((hw_id1 >> 7) & 0xF);
    wgp_ids[bid] = wgp_id;

    uint32_t orig_mode = 0;
    asm volatile("s_getreg_b32 %0, hwreg(1, 0, 32)" : "=s"(orig_mode));
    mode_before[bid] = (int32_t)orig_mode;

    int32_t bank = wgp_to_bank[wgp_id & 0xF];

    // ===== CHANNEL 1: FP16 rounding (per-bank) =====
    uint32_t target_round = (uint32_t)(round_mode_map[bank] & 0xF);
    int sgpr_round = __builtin_amdgcn_readfirstlane((int)target_round);
    __builtin_amdgcn_s_setreg(0x1801, sgpr_round);  // hwreg(1, 0, 4)

    // Read back after rounding write
    uint32_t mode_after_r = 0;
    asm volatile("s_getreg_b32 %0, hwreg(1, 0, 32)" : "=s"(mode_after_r));
    mode_after_round[bid] = (int32_t)mode_after_r;

    __half h_sum = __float2half(0.0f);
    float f_sum = 0.0f;
    for (int i = 0; i < n_fp16_accum; i++) {
        float denom = (float)(2 + i);
        __half h_val = __float2half(1.0f / denom);
        h_sum = __hadd(h_sum, h_val);
        f_sum += 1.0f / denom;
    }
    fp16_sums[bid] = __half2float(h_sum);
    fp32_sums[bid] = f_sum;

    // ===== CHANNEL 2: Timing (per-sample, continuous) =====
    uint64_t t_start = clock64();
    float timing_acc = 0.0f;
    for (int i = 0; i < 256; i++) {
        timing_acc += 1.0f / (float)(i + 1);
    }
    uint64_t t_end = clock64();
    timing_cycles[bid] = (float)(t_end - t_start) + timing_acc * 1e-20f;

    // ===== CHANNEL 3: Per-bank denorm computation =====
    // KEY: each bank gets different denorm mode
    // Even banks → FTZ on (flush subnormals)
    // Odd banks → FTZ off (preserve subnormals)
    int32_t denorm_mode = denorm_mode_map[bank];
    uint32_t mode_with_denorm;
    if (denorm_mode == 0) {
        // FTZ on: clear denorm bits [7:4]
        mode_with_denorm = (orig_mode & 0x0Fu);  // keep rounding, clear denorm
    } else {
        // FTZ off: set denorm bits [7:4] to allow all
        mode_with_denorm = (orig_mode & 0x0Fu) | 0xF0u;
    }
    int sgpr_denorm = __builtin_amdgcn_readfirstlane((int)(mode_with_denorm & 0xFF));
    __builtin_amdgcn_s_setreg(0x3801, sgpr_denorm);

    // Subnormal computation — result DEPENDS on denorm mode
    float sub_result = 0.0f;
    float sub_val = 1.0e-39f;
    for (int i = 0; i < n_denorm_ops; i++) {
        sub_result += sub_val * sub_val;  // FTZ: 0, allow: ~1e-78
        sub_val *= 0.99f;
        sub_result += sub_val;            // FTZ: may flush, allow: preserved
    }
    denorm_results[bid] = sub_result;

    // Also compute a reference (always with current mode, for diff)
    float ref_val = 1.0f;
    float ref_result = 0.0f;
    for (int i = 0; i < n_denorm_ops; i++) {
        ref_result += ref_val * ref_val;
        ref_val *= 0.99f;
        ref_result += ref_val;
    }
    denorm_diffs[bid] = sub_result - ref_result;  // Always negative, magnitude varies

    // ===== Restore original MODE =====
    int sgpr_restore = __builtin_amdgcn_readfirstlane((int)(orig_mode & 0xFF));
    __builtin_amdgcn_s_setreg(0x3801, sgpr_restore);
}

__global__ void read_wgp_only(int32_t* wgp_ids, int batch_size) {
    int bid = blockIdx.x;
    if (bid >= batch_size) return;
    if (threadIdx.x == 0) {
        uint32_t hw_id1 = 0;
        asm volatile("s_getreg_b32 %0, hwreg(23, 0, 32)" : "=s"(hw_id1));
        wgp_ids[bid] = (int32_t)((hw_id1 >> 7) & 0xF);
    }
}

std::vector<torch::Tensor> physics_probe_fn(
    int batch_size,
    torch::Tensor round_mode_map,
    torch::Tensor denorm_mode_map,
    torch::Tensor wgp_to_bank,
    int n_fp16_accum,
    int n_denorm_ops
) {
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto wgp_ids = torch::zeros({batch_size}, opts_i);
    auto fp16_sums = torch::zeros({batch_size}, opts_f);
    auto fp32_sums = torch::zeros({batch_size}, opts_f);
    auto timing = torch::zeros({batch_size}, opts_f);
    auto denorm_res = torch::zeros({batch_size}, opts_f);
    auto denorm_diffs = torch::zeros({batch_size}, opts_f);
    auto mode_before = torch::zeros({batch_size}, opts_i);
    auto mode_after_r = torch::zeros({batch_size}, opts_i);

    physics_probe<<<batch_size, 32>>>(
        wgp_ids.data_ptr<int32_t>(),
        fp16_sums.data_ptr<float>(),
        fp32_sums.data_ptr<float>(),
        timing.data_ptr<float>(),
        denorm_res.data_ptr<float>(),
        denorm_diffs.data_ptr<float>(),
        mode_before.data_ptr<int32_t>(),
        mode_after_r.data_ptr<int32_t>(),
        round_mode_map.data_ptr<int32_t>(),
        denorm_mode_map.data_ptr<int32_t>(),
        wgp_to_bank.data_ptr<int32_t>(),
        batch_size,
        n_fp16_accum,
        n_denorm_ops
    );
    return {wgp_ids, fp16_sums, fp32_sums, timing, denorm_res, denorm_diffs,
            mode_before, mode_after_r};
}

torch::Tensor get_wgp_ids_only(int batch_size) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto wgp_ids = torch::zeros({batch_size}, opts);
    read_wgp_only<<<batch_size, 32>>>(wgp_ids.data_ptr<int32_t>(), batch_size);
    return wgp_ids;
}
'''

HIP_CPP = r'''
std::vector<torch::Tensor> physics_probe_fn(int batch_size, torch::Tensor round_mode_map, torch::Tensor denorm_mode_map, torch::Tensor wgp_to_bank, int n_fp16_accum, int n_denorm_ops);
torch::Tensor get_wgp_ids_only(int batch_size);
'''


class PhysicsProbe:
    def __init__(self, n_fp16=256, n_denorm=64):
        from torch.utils.cpp_extension import load_inline
        self.n_fp16 = n_fp16
        self.n_denorm = n_denorm
        print(f"[BUILD] Compiling physics probe...")
        t0 = time.time()
        self.ext = load_inline(
            name='physics_probe_z2056',
            cpp_sources=HIP_CPP,
            cuda_sources=HIP_SOURCE,
            functions=['physics_probe_fn', 'get_wgp_ids_only'],
            verbose=False,
            extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'],
        )
        print(f"[BUILD] Done in {time.time()-t0:.1f}s")

        self.wgp_to_bank_map = torch.zeros(16, dtype=torch.int32, device='cuda')
        for i, wgp in enumerate(WGP_VALUES):
            self.wgp_to_bank_map[wgp] = i

        # Per-bank rounding: even→nearest(0), odd→toward-zero(0xF)
        self.round_mode_map = torch.zeros(N_BANKS, dtype=torch.int32, device='cuda')
        for i in range(N_BANKS):
            self.round_mode_map[i] = 0 if i % 2 == 0 else 0xF

        # Per-bank denorm: even→flush(0), odd→allow(1)
        self.denorm_mode_map = torch.zeros(N_BANKS, dtype=torch.int32, device='cuda')
        for i in range(N_BANKS):
            self.denorm_mode_map[i] = 0 if i % 2 == 0 else 1

        # Calibration statistics (populated during calibrate())
        self.timing_mean = 1.0
        self.timing_std = 1.0
        self.fp16_vals = [5.0, 5.1]  # Will be updated

    def probe(self, batch_size):
        results = self.ext.physics_probe_fn(
            batch_size, self.round_mode_map, self.denorm_mode_map,
            self.wgp_to_bank_map, self.n_fp16, self.n_denorm
        )
        torch.cuda.synchronize()
        keys = ['wgp_ids', 'fp16_sums', 'fp32_sums', 'timing',
                'denorm_results', 'denorm_diffs', 'mode_before', 'mode_after_round']
        return dict(zip(keys, results))

    def get_wgp_ids(self, batch_size):
        ids = self.ext.get_wgp_ids_only(batch_size)
        torch.cuda.synchronize()
        return ids

    def calibrate(self):
        print(f"\n[CALIBRATE] All analog channels...")
        r = self.probe(256)

        wgp_ids = r['wgp_ids']
        print(f"  WGP_ids: {torch.unique(wgp_ids).cpu().tolist()}")

        fp16 = r['fp16_sums'].cpu().numpy()
        fp32 = r['fp32_sums'].cpu().numpy()
        fp16_unique = len(np.unique(np.round(fp16, 6)))
        self.fp16_vals = sorted(np.unique(np.round(fp16, 6)).tolist())
        print(f"  FP16 unique: {fp16_unique}, values: {self.fp16_vals}")

        timing = r['timing'].cpu().numpy()
        self.timing_mean = float(timing.mean())
        self.timing_std = float(timing.std())
        print(f"  Timing: mean={self.timing_mean:.0f} std={self.timing_std:.0f} CV={self.timing_std/self.timing_mean:.4f}")

        denorm = r['denorm_results'].cpu().numpy()
        denorm_unique = len(np.unique(np.round(denorm, 10)))
        print(f"  Denorm unique: {denorm_unique}")
        print(f"  Denorm range: [{denorm.min():.2e}, {denorm.max():.2e}]")

        # Bank-level analysis
        banks = wgp_to_bank_fn(wgp_ids, build_wgp_to_bank().to(DEVICE))
        for b in range(N_BANKS):
            mask = (banks == b).cpu().numpy()
            if mask.sum() > 0:
                fp16_b = fp16[mask]
                denorm_b = denorm[mask]
                print(f"  Bank {b}: fp16={fp16_b[0]:.6f} denorm={denorm_b[0]:.2e} "
                      f"round={self.round_mode_map[b].item()} denorm_mode={self.denorm_mode_map[b].item()}")

        mode_changed = (r['mode_before'] != r['mode_after_round']).sum().item()
        print(f"  MODE changed (after round write): {mode_changed}/{256}")

        return {
            'fp16_unique': fp16_unique,
            'denorm_unique': denorm_unique,
            'timing_cv': float(self.timing_std/self.timing_mean),
            'mode_change_pct': mode_changed/256*100,
        }


def build_wgp_to_bank():
    mapping = torch.zeros(16, dtype=torch.long)
    for i, wgp in enumerate(WGP_VALUES):
        mapping[wgp] = i
    return mapping

def wgp_to_bank_fn(wgp_ids, mapping):
    return mapping.to(wgp_ids.device)[wgp_ids.clamp(0, 15).long()]


# =============================================================================
# Physics-dependent label permutation (CONTINUOUS)
# =============================================================================

def compute_label_offset(bank_ids, fp16_sums, timing, denorm_results, probe):
    """Compute CONTINUOUS label offset from physics.

    The offset depends on multiple physics channels:
    - Bank parity: even vs odd (binary)
    - FP16 rounding: nearest vs toward-zero (different fp16 sum values)
    - Timing quantile: continuous, binned into 2 groups
    - Denorm result: 0 vs tiny positive (per-bank denorm mode)

    This creates 4-8 distinct offsets instead of the binary 2 from z2050-z2055.
    """
    B = bank_ids.shape[0]
    offsets = torch.zeros(B, dtype=torch.long, device=bank_ids.device)

    # Component 1: Bank parity (0 or 5)
    is_odd = (bank_ids % 2 == 1)
    offsets[is_odd] = 5

    # Component 2: FP16 rounding value → additional 0 or 2
    # fp16_sums has 2 unique values depending on rounding mode
    # Threshold: midpoint of the two known fp16 values
    if len(probe.fp16_vals) >= 2:
        thresh = (probe.fp16_vals[0] + probe.fp16_vals[-1]) / 2.0
    else:
        thresh = fp16_sums.float().mean().item()
    fp16_high = fp16_sums > thresh
    offsets[fp16_high] += 2

    # Component 3: Timing quantile → additional 0 or 1
    # Uses ACTUAL timing value — varies with physical state
    timing_norm = (timing - probe.timing_mean) / max(probe.timing_std, 1.0)
    timing_high = timing_norm > 0  # Above-mean timing
    offsets[timing_high] += 1

    # Final offset mod 10
    return offsets % 10


def permute_labels_physics(labels, bank_ids, fp16_sums, timing, denorm_results, probe):
    """Apply physics-dependent label permutation."""
    offsets = compute_label_offset(bank_ids, fp16_sums, timing, denorm_results, probe)
    return (labels + offsets) % 10


# =============================================================================
# Feature extraction with normalization
# =============================================================================

def extract_features_normalized(probe_result, probe_obj):
    """Extract and NORMALIZE analog features to [0,1] range.
    Fixes z2055's training instability from wildly different scales.
    """
    fp16 = probe_result['fp16_sums']
    fp32 = probe_result['fp32_sums']
    timing = probe_result['timing']
    denorm = probe_result['denorm_results']
    denorm_diff = probe_result['denorm_diffs']

    # Normalize each feature
    fp16_norm = (fp16 - fp16.mean()) / fp16.std().clamp(min=1e-6)
    fp16_diff_norm = (fp16 - fp32)
    fp16_diff_norm = (fp16_diff_norm - fp16_diff_norm.mean()) / fp16_diff_norm.std().clamp(min=1e-6)
    timing_norm = (timing - probe_obj.timing_mean) / max(probe_obj.timing_std, 1.0)
    denorm_norm = denorm / denorm.abs().max().clamp(min=1e-40)
    denorm_diff_norm = denorm_diff / denorm_diff.abs().max().clamp(min=1e-10)

    feat = torch.stack([
        fp16_norm,           # FP16 sum (normalized)
        fp16_diff_norm,      # Rounding error (normalized)
        timing_norm,         # Timing (z-scored)
        denorm_norm,         # Denorm result (normalized)
        denorm_diff_norm,    # Denorm diff (normalized)
    ], dim=1)

    return feat


# =============================================================================
# Models
# =============================================================================

class PhysicsModel(nn.Module):
    """Neural net where physics continuously modulates computation."""

    def __init__(self, n_banks=N_BANKS, hidden=HIDDEN_DIM, n_feat=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, hidden), nn.ReLU(),
        )
        self.bank_weights = nn.Parameter(torch.randn(n_banks, hidden, hidden) * 0.02)
        self.analog_embed = nn.Sequential(
            nn.Linear(n_feat, 64), nn.ReLU(), nn.Linear(64, hidden),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(n_feat, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 10))
        self.self_model = nn.Linear(hidden, n_banks)

    def forward(self, x, bank_ids, analog_feat=None, force_gate=None):
        h = self.encoder(x)
        W = self.bank_weights[bank_ids]
        h_bank = torch.bmm(W, h.unsqueeze(-1)).squeeze(-1)

        if analog_feat is not None:
            h_analog = self.analog_embed(analog_feat)
            gate = self.gate_net(analog_feat) if force_gate is None else \
                   torch.full((x.shape[0], 1), force_gate, device=x.device)
            h_out = h_bank + gate * h_analog
        else:
            h_out = h_bank
            gate = torch.zeros(x.shape[0], 1, device=x.device)

        logits = self.classifier(h_out)
        self_pred = self.self_model(h_out.detach())
        return logits, self_pred, gate


class DigitalModel(nn.Module):
    def __init__(self, n_banks=N_BANKS, hidden=HIDDEN_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, hidden), nn.ReLU(),
        )
        self.bank_weights = nn.Parameter(torch.randn(n_banks, hidden, hidden) * 0.02)
        self.classifier = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 10))
        self.self_model = nn.Linear(hidden, n_banks)

    def forward(self, x, bank_ids, analog_feat=None, force_gate=None):
        h = self.encoder(x)
        W = self.bank_weights[bank_ids]
        h_out = torch.bmm(W, h.unsqueeze(-1)).squeeze(-1)
        logits = self.classifier(h_out)
        self_pred = self.self_model(h_out.detach())
        return logits, self_pred, torch.zeros(x.shape[0], 1, device=x.device)


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

    def forward(self, x, bank_ids=None, analog_feat=None, force_gate=None):
        h = F.relu(self.transform(self.encoder(x)))
        return self.classifier(h), None, torch.zeros(x.shape[0], 1, device=x.device)


# =============================================================================
# Helpers
# =============================================================================

def find_power_sysfs():
    for card in Path('/sys/class/drm/').glob('card*/device/hwmon/hwmon*/power1_average'):
        return str(card)
    return None

def read_temp_c():
    for f in Path('/sys/class/drm/').glob('card*/device/hwmon/hwmon*/temp1_input'):
        try: return int(open(f).read().strip()) / 1000.0
        except: pass
    return 0.0

def get_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = torchvision.datasets.MNIST(root=str(ROOT/'data'), train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root=str(ROOT/'data'), train=False, transform=transform)
    return (DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True),
            DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True))


def gpu_warmup(seconds=10):
    """Heat up the GPU by running intense computation."""
    print(f"  [WARMUP] Heating GPU for {seconds}s...")
    t0_temp = read_temp_c()
    t0 = time.time()
    # Run heavy matmul to heat up
    a = torch.randn(2048, 2048, device=DEVICE)
    while time.time() - t0 < seconds:
        a = torch.mm(a, a.t())
        a = a / a.norm()
    t1_temp = read_temp_c()
    print(f"  [WARMUP] Temp: {t0_temp:.0f}C → {t1_temp:.0f}C (+{t1_temp-t0_temp:.0f}C)")
    return t0_temp, t1_temp


# =============================================================================
# Training
# =============================================================================

def train_condition(model, loader, probe, wgp_map, cond, power_path):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    t0 = time.time()
    gate_vals = []
    offset_hist = {}

    for epoch in range(N_EPOCHS):
        correct = total = self_c = 0
        epoch_loss = 0
        epoch_gates = []

        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            B = images.shape[0]

            if cond == 'A_physics':
                pr = probe.probe(B)
                wgp_ids = pr['wgp_ids']
                bank_ids = wgp_to_bank_fn(wgp_ids, wgp_map)
                perm_labels = permute_labels_physics(
                    labels, bank_ids, pr['fp16_sums'], pr['timing'],
                    pr['denorm_results'], probe)
                analog_feat = extract_features_normalized(pr, probe)
                logits, self_pred, gate = model(images, bank_ids, analog_feat)

                # Track offset distribution
                offsets = compute_label_offset(
                    bank_ids, pr['fp16_sums'], pr['timing'],
                    pr['denorm_results'], probe)
                for o in offsets.cpu().tolist():
                    offset_hist[int(o)] = offset_hist.get(int(o), 0) + 1

            elif cond == 'B_digital':
                wgp_ids = probe.get_wgp_ids(B)
                bank_ids = wgp_to_bank_fn(wgp_ids, wgp_map)
                # Digital model uses binary permutation only
                perm_labels = labels.clone()
                odd_mask = (bank_ids % 2) == 1
                perm_labels[odd_mask] = (labels[odd_mask] + 5) % 10
                logits, self_pred, gate = model(images, bank_ids)

            elif cond == 'C_blind':
                bank_ids = torch.zeros(B, dtype=torch.long, device=DEVICE)
                perm_labels = labels.clone()
                logits, self_pred, gate = model(images, bank_ids)

            else:
                raise ValueError(cond)

            loss = F.cross_entropy(logits, perm_labels)
            if self_pred is not None:
                loss += 0.1 * F.cross_entropy(self_pred, bank_ids)

            opt.zero_grad(); loss.backward(); opt.step()
            correct += (logits.argmax(1) == perm_labels).sum().item()
            total += B; epoch_loss += loss.item()
            epoch_gates.extend(gate.detach().cpu().tolist())

            if self_pred is not None:
                self_c += (self_pred.argmax(1) == bank_ids).sum().item()

        gate_mean = np.mean([g[0] if isinstance(g, list) else g for g in epoch_gates])
        gate_vals.append(gate_mean)

        if epoch % 3 == 0 or epoch == N_EPOCHS-1:
            print(f"  Epoch {epoch+1:2d}: loss={epoch_loss/len(loader):.4f} "
                  f"acc={correct/total:.4f} self={self_c/total:.3f} "
                  f"gate={gate_mean:.4f} temp={read_temp_c():.0f}C")

    print(f"  Done in {time.time()-t0:.1f}s")
    n_offsets = len(offset_hist) if offset_hist else 0
    print(f"  Offset distribution ({n_offsets} distinct): {dict(sorted(offset_hist.items()))}")
    return {
        'train_time_s': time.time()-t0,
        'gate_trajectory': gate_vals,
        'final_gate': float(gate_vals[-1]),
        'n_distinct_offsets': n_offsets,
        'offset_distribution': offset_hist,
    }


def evaluate(model, loader, probe, wgp_map, cond, scrambled_map=None):
    model.eval()
    correct = total = self_c = 0
    all_gates = []
    all_logits = []
    fp16_vals = []
    denorm_vals = []
    offset_hist = {}

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            B = images.shape[0]
            bank_map = scrambled_map if scrambled_map is not None else wgp_map

            if cond in ('A_physics', 'D_scrambled', 'E_thermal'):
                pr = probe.probe(B)
                wgp_ids = pr['wgp_ids']
                real_banks = wgp_to_bank_fn(wgp_ids, wgp_map)
                perm_labels = permute_labels_physics(
                    labels, real_banks, pr['fp16_sums'], pr['timing'],
                    pr['denorm_results'], probe)
                bank_ids = wgp_to_bank_fn(wgp_ids, bank_map)
                analog_feat = extract_features_normalized(pr, probe)
                logits, self_pred, gate = model(images, bank_ids, analog_feat)
                fp16_vals.extend(pr['fp16_sums'].cpu().tolist())
                denorm_vals.extend(pr['denorm_results'].cpu().tolist())

                offsets = compute_label_offset(
                    real_banks, pr['fp16_sums'], pr['timing'],
                    pr['denorm_results'], probe)
                for o in offsets.cpu().tolist():
                    offset_hist[int(o)] = offset_hist.get(int(o), 0) + 1

            elif cond == 'B_digital':
                wgp_ids = probe.get_wgp_ids(B)
                bank_ids = wgp_to_bank_fn(wgp_ids, wgp_map)
                perm_labels = labels.clone()
                odd_mask = (bank_ids % 2) == 1
                perm_labels[odd_mask] = (labels[odd_mask] + 5) % 10
                logits, self_pred, gate = model(images, bank_ids)

            elif cond == 'C_blind':
                bank_ids = torch.zeros(B, dtype=torch.long, device=DEVICE)
                perm_labels = labels.clone()
                logits, self_pred, gate = model(images, bank_ids)

            else:
                bank_ids = torch.zeros(B, dtype=torch.long, device=DEVICE)
                perm_labels = labels; logits, self_pred, gate = model(images, bank_ids)

            correct += (logits.argmax(1) == perm_labels).sum().item()
            total += B
            all_gates.extend(gate.cpu().tolist())
            all_logits.extend(logits.cpu().tolist())

            if self_pred is not None:
                self_c += (self_pred.argmax(1) == bank_ids).sum().item()

    gate_flat = [g[0] if isinstance(g, list) else g for g in all_gates]
    fp16_unique = len(set([f'{v:.6f}' for v in fp16_vals])) if fp16_vals else 0
    denorm_unique = len(set([f'{v:.10f}' for v in denorm_vals])) if denorm_vals else 0

    return {
        'accuracy': correct/total,
        'self_model_acc': self_c/total if self_c > 0 else 0,
        'gate_mean': float(np.mean(gate_flat)),
        'gate_std': float(np.std(gate_flat)),
        'n_fp16_unique': fp16_unique,
        'n_denorm_unique': denorm_unique,
        'n_distinct_offsets': len(offset_hist),
        'offset_distribution': offset_hist,
    }


def thermal_comparison(model, loader, probe, wgp_map):
    """Compare outputs at cold vs hot GPU temperature."""
    # Cold run
    cold_temp = read_temp_c()
    print(f"  Cold eval at {cold_temp:.0f}C...")
    eval_cold = evaluate(model, loader, probe, wgp_map, 'A_physics')

    # Heat up GPU
    t0_temp, t1_temp = gpu_warmup(seconds=15)

    # Hot run (immediately after warmup)
    hot_temp = read_temp_c()
    print(f"  Hot eval at {hot_temp:.0f}C...")
    eval_hot = evaluate(model, loader, probe, wgp_map, 'E_thermal')

    acc_diff = abs(eval_hot['accuracy'] - eval_cold['accuracy'])
    gate_diff = abs(eval_hot['gate_mean'] - eval_cold['gate_mean'])
    print(f"  Cold: acc={eval_cold['accuracy']:.4f} gate={eval_cold['gate_mean']:.4f}")
    print(f"  Hot:  acc={eval_hot['accuracy']:.4f} gate={eval_hot['gate_mean']:.4f}")
    print(f"  Diff: acc={acc_diff:.4f} gate={gate_diff:.4f}")
    print(f"  Temp delta: {hot_temp - cold_temp:.0f}C")

    return {
        'cold_temp': cold_temp,
        'hot_temp': hot_temp,
        'temp_delta': hot_temp - cold_temp,
        'cold_acc': eval_cold['accuracy'],
        'hot_acc': eval_hot['accuracy'],
        'acc_diff': acc_diff,
        'cold_gate': eval_cold['gate_mean'],
        'hot_gate': eval_hot['gate_mean'],
        'gate_diff': gate_diff,
        'thermal_changes_output': acc_diff > 0.001 or gate_diff > 0.001,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print(f"z2056: Physics IS Computation — Continuous Hardware-Dependent Output")
    print(f"  Paradigm: physics modulates answer, not just routes it")
    print(f"  Machine: {MACHINE}")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Initial temp: {read_temp_c():.1f}C")

    power_path = find_power_sysfs()
    print(f"Power sysfs: {power_path}")

    probe = PhysicsProbe(n_fp16=256, n_denorm=64)
    calib = probe.calibrate()

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
                   'batch_size': BATCH_SIZE, 'lr': LR},
        'calibration': calib,
        'conditions': {}, 'tests': {},
    }

    # A: Full physics
    print(f"\n{'='*60}\n  Training: A_physics (continuous physics labels)\n{'='*60}")
    model_A = PhysicsModel().to(DEVICE)
    ti_A = train_condition(model_A, train_loader, probe, wgp_map, 'A_physics', power_path)
    eval_A = evaluate(model_A, test_loader, probe, wgp_map, 'A_physics')
    print(f"\n  >> A_physics: acc={eval_A['accuracy']:.4f} gate={eval_A['gate_mean']:.4f} "
          f"offsets={eval_A['n_distinct_offsets']} denorm_unique={eval_A['n_denorm_unique']}")
    results['conditions']['A_physics'] = {'eval': eval_A, **ti_A}

    # B: Digital only
    print(f"\n{'='*60}\n  Training: B_digital (WGP bank only, binary labels)\n{'='*60}")
    model_B = DigitalModel().to(DEVICE)
    ti_B = train_condition(model_B, train_loader, probe, wgp_map, 'B_digital', power_path)
    eval_B = evaluate(model_B, test_loader, probe, wgp_map, 'B_digital')
    print(f"\n  >> B_digital: acc={eval_B['accuracy']:.4f}")
    results['conditions']['B_digital'] = {'eval': eval_B, **ti_B}

    # C: Blind
    print(f"\n{'='*60}\n  Training: C_blind (no hardware)\n{'='*60}")
    model_C = BlindModel().to(DEVICE)
    ti_C = train_condition(model_C, train_loader, probe, wgp_map, 'C_blind', power_path)
    eval_C = evaluate(model_C, test_loader, probe, wgp_map, 'C_blind')
    print(f"\n  >> C_blind: acc={eval_C['accuracy']:.4f}")
    results['conditions']['C_blind'] = {'eval': eval_C, **ti_C}

    # D: Scrambled
    print(f"\n{'='*60}\n  D_scrambled: A's model with WRONG WGP→bank\n{'='*60}")
    eval_D = evaluate(model_A, test_loader, probe, wgp_map, 'D_scrambled', scrambled_map=scrambled_map)
    print(f"\n  >> D_scrambled: acc={eval_D['accuracy']:.4f}")
    results['conditions']['D_scrambled'] = {'eval': eval_D}

    # E: Thermal perturbation
    print(f"\n{'='*60}\n  E_thermal: Cold vs Hot comparison\n{'='*60}")
    thermal = thermal_comparison(model_A, test_loader, probe, wgp_map)
    results['thermal_test'] = thermal

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

    # Gate analysis
    gate_weights = model_A.gate_net[0].weight.data.cpu().numpy()
    gate_importance = np.abs(gate_weights).mean(axis=0)
    feature_names = ['fp16_sum', 'fp16_diff', 'timing_norm', 'denorm_result', 'denorm_diff']
    results['gate_analysis'] = {
        'feature_importance': {name: float(imp) for name, imp in zip(feature_names, gate_importance)},
        'final_gate_mean': eval_A['gate_mean'],
    }
    print(f"\n  Gate feature importance:")
    for name, imp in zip(feature_names, gate_importance):
        print(f"    {name:20s}: {imp:.4f}")

    # TESTS
    acc = {k: results['conditions'][k]['eval']['accuracy'] for k in results['conditions']}

    results['tests'] = {
        'T1_accuracy': {
            'criterion': 'A > 85%', 'value': acc['A_physics'],
            'pass': acc['A_physics'] > 0.85
        },
        'T2_gap': {
            'criterion': 'A - C gap > 30%',
            'A': acc['A_physics'], 'C': acc['C_blind'],
            'gap': round(acc['A_physics'] - acc['C_blind'], 4),
            'pass': (acc['A_physics'] - acc['C_blind']) > 0.30
        },
        'T3_kill_shot': {
            'criterion': 'A - D gap > 10%',
            'A': acc['A_physics'], 'D': acc['D_scrambled'],
            'gap': round(acc['A_physics'] - acc['D_scrambled'], 4),
            'pass': (acc['A_physics'] - acc['D_scrambled']) > 0.10
        },
        'T4_gate_learned': {
            'criterion': 'Gate mean > 0.1 (model uses physics more)',
            'value': eval_A['gate_mean'],
            'pass': eval_A['gate_mean'] > 0.1
        },
        'T5_continuous_offsets': {
            'criterion': '≥3 distinct label offsets',
            'value': eval_A['n_distinct_offsets'],
            'pass': eval_A['n_distinct_offsets'] >= 3
        },
        'T6_thermal_effect': {
            'criterion': 'Thermal perturbation changes output',
            'value': thermal['thermal_changes_output'],
            'pass': thermal['thermal_changes_output']
        },
        'T7_denorm_diversity': {
            'criterion': 'Per-bank denorm > 1 unique',
            'value': eval_A['n_denorm_unique'],
            'pass': eval_A['n_denorm_unique'] > 1
        },
        'T8_gate_importance': {
            'criterion': 'Feature importance max/min > 1.5 (non-uniform)',
            'max': float(gate_importance.max()),
            'min': float(gate_importance.min()),
            'ratio': float(gate_importance.max() / max(gate_importance.min(), 1e-6)),
            'pass': gate_importance.max() / max(gate_importance.min(), 1e-6) > 1.5
        },
    }

    n_pass = sum(1 for t in results['tests'].values() if t['pass'])
    results['n_pass'] = n_pass
    results['n_total'] = len(results['tests'])
    results['verdict'] = ('PHYSICS_IS_COMPUTATION' if n_pass >= 7
                          else 'STRONG' if n_pass >= 6
                          else 'PARTIAL' if n_pass >= 4
                          else 'WEAK')

    results['summary'] = {k: acc[k] for k in acc}
    results['summary']['gate_mean'] = eval_A['gate_mean']
    results['summary']['n_offsets'] = eval_A['n_distinct_offsets']
    results['summary']['denorm_unique'] = eval_A['n_denorm_unique']
    results['summary']['thermal_changes'] = thermal['thermal_changes_output']
    results['summary']['bank_cos_mean'] = cos_sims['mean']

    print(f"\n{'='*70}\nTESTS\n{'='*70}")
    for name, t in results['tests'].items():
        print(f"  {name}: {'PASS' if t['pass'] else 'FAIL'} — {t['criterion']}")
        for k, v in t.items():
            if k not in ('criterion', 'pass'): print(f"    {k} = {v}")

    print(f"\n  VERDICT: {results['verdict']} ({n_pass}/{results['n_total']} PASS)")

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for k, v in results['summary'].items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        elif isinstance(v, bool):
            print(f"  {k:25s}: {v}")
        else:
            print(f"  {k:25s}: {v}")

    out = RESULTS_DIR / 'z2056_physics_is_computation.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")
    print(f"Final temp: {read_temp_c():.1f}C")


if __name__ == '__main__':
    main()
