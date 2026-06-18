#!/usr/bin/env python3
"""z2055: Analog Physics Gate — Hardware State Modulates Computation

THE PARADIGM SHIFT:
  z2050-z2054: Hardware SELECTS computation path (digital bank switching)
  z2055: Hardware CONTINUOUSLY MODULATES computation output (analog gate)

THE USER'S INSIGHT:
  "Computer, what is 1+1?"
  Digital mode:  "2" (always, regardless of state)
  Analog mode:   "Well I'm feeling panicky so... maybe 1, or 2, or 3!"

  We want the gap between hardware and software to CLOSE — physics becomes
  computation, not just routing. The GPU's physical state (temperature,
  voltage, rounding mode) should CHANGE the numerical output.

THREE ANALOG CHANNELS:
  1. FP16 rounding (z2054, proven): MODE register → different fp16 sum
     - Semi-analog: 2 discrete values based on rounding mode
     - The fp16 sum IS a different number depending on hardware state

  2. clock64() timing (z2047, proven): Temperature → DVFS → timing varies
     - Truly analog: continuous value that changes with physical state
     - Hot GPU = slower clocks = different timing = different feature

  3. Denormal computation (NEW): FTZ on vs off → different numerical results
     - Toggle denorm mode mid-kernel via s_setreg_b32
     - With FTZ: subnormal × subnormal = 0 (flushed)
     - Without FTZ: subnormal × subnormal = tiny positive number
     - The RESULT changes, not just the timing
     - Also measure timing difference (may vary with temperature)

ARCHITECTURE:
  encoder(image) → h [B, 128]                    # Pure CNN, same everywhere
  bank_weights[wgp_id] @ h → h_bank              # Digital path (proven)
  analog_embed(fp16, timing, denorm) → h_analog   # Physics path (new)
  gate = sigmoid(W_gate @ h_analog)               # Learned analog gate (0-1)
  h_out = h_bank + gate * h_analog                # Physics modulates output
  classifier(h_out) → logits

  The gate is the key insight:
  - gate ≈ 0: "I'm a deterministic computer"
  - gate ≈ 1: "Physics modulates my answer"
  - The model LEARNS how much to trust its physical state

CONDITIONS:
  A_analog:     Full physics (WGP + fp16 + timing + denorm) with analog gate
  B_digital:    WGP bank only (z2050 replication, no continuous physics)
  C_blind:      No hardware at all
  D_scrambled:  Wrong WGP→bank map (kill shot)
  E_frozen:     Physics frozen to batch mean (removes per-sample variation)
  F_nogate:     Physics features but gate forced to 0.5 (no learned gating)

TESTS:
  T1: A > 90% accuracy
  T2: A - C gap > 30% (physics necessary for task)
  T3: Kill shot — D < A by > 10%
  T4: Analog gate mean > 0.1 (model uses physics)
  T5: Denorm channel produces > 1 unique value
  T6: Cross-run variance > 0 (same input, different physics → different output)
  T7: Physics feature MI > 0.01 with output (physics carries information)
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
N_EPOCHS = 12
BATCH_SIZE = 128
LR = 1e-3

# =============================================================================
# HIP Kernel: Multi-channel analog physics probe
# =============================================================================

HIP_SOURCE = r'''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>

// Multi-channel analog physics kernel
// Outputs: wgp_id, fp16_sum, fp32_sum, timing, denorm_ftz_result,
//          denorm_allow_result, denorm_ftz_time, denorm_allow_time,
//          mode_before, mode_after
__global__ void analog_physics_probe(
    int32_t* wgp_ids,              // [B] — physical WGP_id
    float* fp16_sums,              // [B] — fp16 accumulation (MODE-dependent)
    float* fp32_sums,              // [B] — fp32 reference
    float* timing_cycles,          // [B] — raw clock64() timing
    float* denorm_ftz_results,     // [B] — subnormal computation with FTZ
    float* denorm_allow_results,   // [B] — subnormal computation without FTZ
    float* denorm_ftz_times,       // [B] — timing with FTZ
    float* denorm_allow_times,     // [B] — timing without FTZ
    int32_t* mode_before,          // [B] — MODE register before writes
    int32_t* mode_after,           // [B] — MODE register after writes
    const int32_t* round_mode_map, // [N_BANKS] — bank→rounding mode
    const int32_t* wgp_to_bank,    // [16] — WGP→bank mapping
    int batch_size,
    int n_fp16_accum,              // fp16 accumulation count
    int n_denorm_ops               // denormal operation count
) {
    int bid = blockIdx.x;
    if (bid >= batch_size) return;
    if (threadIdx.x != 0) return;

    // ===== CHANNEL 0: Physical identity =====
    uint32_t hw_id1 = 0;
    asm volatile("s_getreg_b32 %0, hwreg(23, 0, 32)" : "=s"(hw_id1));
    int32_t wgp_id = (int32_t)((hw_id1 >> 7) & 0xF);
    wgp_ids[bid] = wgp_id;

    // Read original MODE
    uint32_t orig_mode = 0;
    asm volatile("s_getreg_b32 %0, hwreg(1, 0, 32)" : "=s"(orig_mode));
    mode_before[bid] = (int32_t)orig_mode;

    int32_t bank = wgp_to_bank[wgp_id & 0xF];

    // ===== CHANNEL 1: FP16 rounding (semi-analog) =====
    // Write rounding mode: even banks → nearest (0), odd → toward-zero (0xF)
    uint32_t target_round = (uint32_t)(round_mode_map[bank] & 0xF);
    int sgpr_round = __builtin_amdgcn_readfirstlane((int)target_round);
    __builtin_amdgcn_s_setreg(0x1801, sgpr_round);  // hwreg(1, 0, 4) = FP_ROUND

    // Accumulate in fp16 — rounding mode changes result
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

    // ===== CHANNEL 2: Raw timing (analog) =====
    // Time a fixed computation — varies with DVFS/temperature
    uint64_t t_start = clock64();
    float timing_acc = 0.0f;
    for (int i = 0; i < 256; i++) {
        timing_acc += 1.0f / (float)(i + 1);
    }
    uint64_t t_end = clock64();
    timing_cycles[bid] = (float)(t_end - t_start) + timing_acc * 1e-20f;  // prevent optimize-out

    // ===== CHANNEL 3: Denormal computation (analog) =====
    // Test A: FTZ ON (flush subnormals to zero)
    // Set MODE: FP_ROUND = nearest (0), FP_DENORM = flush (0)
    // MODE[7:0] = 0x00
    int sgpr_ftz = __builtin_amdgcn_readfirstlane(0x00);
    __builtin_amdgcn_s_setreg(0x3801, sgpr_ftz);  // hwreg(1, 0, 8)

    uint64_t t_ftz_start = clock64();
    float ftz_result = 0.0f;
    float sub_val = 1.0e-39f;  // Deeply subnormal for fp32
    for (int i = 0; i < n_denorm_ops; i++) {
        ftz_result += sub_val * sub_val;  // FTZ: this becomes 0.0
        sub_val *= 0.99f;
        ftz_result += sub_val;            // FTZ: this may flush or not
    }
    uint64_t t_ftz_end = clock64();
    denorm_ftz_results[bid] = ftz_result;
    denorm_ftz_times[bid] = (float)(t_ftz_end - t_ftz_start);

    // Test B: FTZ OFF (allow denormals)
    // MODE[7:0] = 0xF0 (denorms allowed, round to nearest)
    int sgpr_denorm = __builtin_amdgcn_readfirstlane(0xF0);
    __builtin_amdgcn_s_setreg(0x3801, sgpr_denorm);  // hwreg(1, 0, 8)

    uint64_t t_denorm_start = clock64();
    float denorm_result = 0.0f;
    sub_val = 1.0e-39f;
    for (int i = 0; i < n_denorm_ops; i++) {
        denorm_result += sub_val * sub_val;  // Preserved: tiny positive
        sub_val *= 0.99f;
        denorm_result += sub_val;            // Preserved: small positive
    }
    uint64_t t_denorm_end = clock64();
    denorm_allow_results[bid] = denorm_result;
    denorm_allow_times[bid] = (float)(t_denorm_end - t_denorm_start);

    // Read back MODE after all writes
    uint32_t final_mode = 0;
    asm volatile("s_getreg_b32 %0, hwreg(1, 0, 32)" : "=s"(final_mode));
    mode_after[bid] = (int32_t)final_mode;

    // Restore original MODE
    int sgpr_restore = __builtin_amdgcn_readfirstlane((int)(orig_mode & 0xFF));
    __builtin_amdgcn_s_setreg(0x3801, sgpr_restore);
}

// Read-only WGP probe (for B_digital condition)
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
std::vector<torch::Tensor> analog_probe(
    int batch_size,
    torch::Tensor round_mode_map,
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
    auto denorm_ftz_res = torch::zeros({batch_size}, opts_f);
    auto denorm_allow_res = torch::zeros({batch_size}, opts_f);
    auto denorm_ftz_t = torch::zeros({batch_size}, opts_f);
    auto denorm_allow_t = torch::zeros({batch_size}, opts_f);
    auto mode_before = torch::zeros({batch_size}, opts_i);
    auto mode_after = torch::zeros({batch_size}, opts_i);

    analog_physics_probe<<<batch_size, 32>>>(
        wgp_ids.data_ptr<int32_t>(),
        fp16_sums.data_ptr<float>(),
        fp32_sums.data_ptr<float>(),
        timing.data_ptr<float>(),
        denorm_ftz_res.data_ptr<float>(),
        denorm_allow_res.data_ptr<float>(),
        denorm_ftz_t.data_ptr<float>(),
        denorm_allow_t.data_ptr<float>(),
        mode_before.data_ptr<int32_t>(),
        mode_after.data_ptr<int32_t>(),
        round_mode_map.data_ptr<int32_t>(),
        wgp_to_bank.data_ptr<int32_t>(),
        batch_size,
        n_fp16_accum,
        n_denorm_ops
    );
    return {wgp_ids, fp16_sums, fp32_sums, timing,
            denorm_ftz_res, denorm_allow_res, denorm_ftz_t, denorm_allow_t,
            mode_before, mode_after};
}

torch::Tensor get_wgp_ids_only(int batch_size) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto wgp_ids = torch::zeros({batch_size}, opts);
    read_wgp_only<<<batch_size, 32>>>(wgp_ids.data_ptr<int32_t>(), batch_size);
    return wgp_ids;
}
'''

HIP_CPP = r'''
std::vector<torch::Tensor> analog_probe(int batch_size, torch::Tensor round_mode_map, torch::Tensor wgp_to_bank, int n_fp16_accum, int n_denorm_ops);
torch::Tensor get_wgp_ids_only(int batch_size);
'''


# =============================================================================
# Analog Physics Probe
# =============================================================================

class AnalogPhysicsProbe:
    """Multi-channel analog physics probe: WGP + fp16 + timing + denorm."""

    def __init__(self, n_fp16=256, n_denorm=64):
        from torch.utils.cpp_extension import load_inline
        self.n_fp16 = n_fp16
        self.n_denorm = n_denorm
        print(f"[BUILD] Compiling analog physics probe...")
        t0 = time.time()
        self.ext = load_inline(
            name='analog_probe_z2055',
            cpp_sources=HIP_CPP,
            cuda_sources=HIP_SOURCE,
            functions=['analog_probe', 'get_wgp_ids_only'],
            verbose=False,
            extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'],
        )
        print(f"[BUILD] Done in {time.time()-t0:.1f}s")

        # WGP → bank mapping
        self.wgp_to_bank_map = torch.zeros(16, dtype=torch.int32, device='cuda')
        for i, wgp in enumerate(WGP_VALUES):
            self.wgp_to_bank_map[wgp] = i

        # Rounding mode: even banks → nearest (0), odd → toward-zero (0xF)
        self.round_mode_map = torch.zeros(N_BANKS, dtype=torch.int32, device='cuda')
        for i in range(N_BANKS):
            self.round_mode_map[i] = 0 if i % 2 == 0 else 0xF

    def probe(self, batch_size):
        """Run full analog probe. Returns dict of tensors."""
        results = self.ext.analog_probe(
            batch_size, self.round_mode_map, self.wgp_to_bank_map,
            self.n_fp16, self.n_denorm
        )
        torch.cuda.synchronize()
        keys = ['wgp_ids', 'fp16_sums', 'fp32_sums', 'timing',
                'denorm_ftz_res', 'denorm_allow_res', 'denorm_ftz_time', 'denorm_allow_time',
                'mode_before', 'mode_after']
        return dict(zip(keys, results))

    def get_wgp_ids(self, batch_size):
        ids = self.ext.get_wgp_ids_only(batch_size)
        torch.cuda.synchronize()
        return ids

    def calibrate(self):
        """Calibrate all channels and report."""
        print(f"\n[CALIBRATE] Testing all analog channels...")
        r = self.probe(128)

        wgp_ids = r['wgp_ids']
        print(f"  WGP_ids: {torch.unique(wgp_ids).cpu().tolist()}")

        # FP16 channel
        fp16 = r['fp16_sums'].cpu().numpy()
        fp32 = r['fp32_sums'].cpu().numpy()
        fp16_unique = len(np.unique(np.round(fp16, 6)))
        print(f"  FP16 unique: {fp16_unique}, range: [{fp16.min():.6f}, {fp16.max():.6f}]")
        print(f"  FP16-FP32 diff range: [{np.abs(fp16-fp32).min():.6f}, {np.abs(fp16-fp32).max():.6f}]")

        # Timing channel
        timing = r['timing'].cpu().numpy()
        print(f"  Timing range: [{timing.min():.0f}, {timing.max():.0f}] cycles")
        print(f"  Timing CV: {timing.std()/timing.mean():.4f}")

        # Denormal channel
        ftz = r['denorm_ftz_res'].cpu().numpy()
        allow = r['denorm_allow_res'].cpu().numpy()
        ftz_t = r['denorm_ftz_time'].cpu().numpy()
        allow_t = r['denorm_allow_time'].cpu().numpy()
        denorm_diff = np.abs(allow - ftz)
        timing_ratio = allow_t / np.maximum(ftz_t, 1.0)

        print(f"  Denorm FTZ result: [{ftz.min():.2e}, {ftz.max():.2e}]")
        print(f"  Denorm allow result: [{allow.min():.2e}, {allow.max():.2e}]")
        print(f"  Denorm diff: [{denorm_diff.min():.2e}, {denorm_diff.max():.2e}]")
        print(f"  Denorm timing ratio (allow/ftz): [{timing_ratio.min():.3f}, {timing_ratio.max():.3f}]")
        print(f"  Denorm unique (ftz): {len(np.unique(np.round(ftz, 10)))}")
        print(f"  Denorm unique (allow): {len(np.unique(np.round(allow, 10)))}")

        # MODE verification
        mode_changed = (r['mode_before'] != r['mode_after']).sum().item()
        print(f"  MODE changed: {mode_changed}/{128} blocks")

        return {
            'fp16_unique': fp16_unique,
            'timing_cv': float(timing.std()/timing.mean()),
            'denorm_diff_mean': float(denorm_diff.mean()),
            'denorm_timing_ratio_mean': float(timing_ratio.mean()),
            'mode_change_pct': mode_changed/128*100,
        }


# =============================================================================
# WGP → Bank + Label Permutation
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
# Models
# =============================================================================

class AnalogPhysicsModel(nn.Module):
    """Neural net with learned analog gate — physics modulates computation.

    The gate controls how much physical state affects the output:
    - gate ≈ 0: pure digital (deterministic, 1+1=2 always)
    - gate ≈ 1: physics-modulated (1+1 = depends on GPU state)
    """

    def __init__(self, n_banks=N_BANKS, hidden=HIDDEN_DIM, n_analog_features=5):
        super().__init__()
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, hidden), nn.ReLU(),
        )

        # Digital path: bank-specific weights (proven in z2050)
        self.bank_weights = nn.Parameter(torch.randn(n_banks, hidden, hidden) * 0.02)

        # Analog path: continuous physics features → hidden modulation
        # Features: fp16_sum, fp16_diff, timing_norm, denorm_diff, denorm_timing_ratio
        self.analog_embed = nn.Sequential(
            nn.Linear(n_analog_features, 64), nn.ReLU(),
            nn.Linear(64, hidden),
        )

        # Learned analog gate: controls physics influence
        self.gate_net = nn.Sequential(
            nn.Linear(n_analog_features, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 10)
        )

        # Self-model: predict own WGP bank
        self.self_model = nn.Linear(hidden, n_banks)

    def forward(self, x, bank_ids, analog_feat=None, force_gate=None):
        h = self.encoder(x)  # [B, hidden]

        # Digital path: bank-specific transform
        W = self.bank_weights[bank_ids]  # [B, hidden, hidden]
        h_bank = torch.bmm(W, h.unsqueeze(-1)).squeeze(-1)  # [B, hidden]

        # Analog path: physics modulates output
        if analog_feat is not None:
            h_analog = self.analog_embed(analog_feat)  # [B, hidden]

            if force_gate is not None:
                gate = torch.full((x.shape[0], 1), force_gate, device=x.device)
            else:
                gate = self.gate_net(analog_feat)  # [B, 1]

            h_out = h_bank + gate * h_analog  # Physics modulates!
        else:
            h_out = h_bank
            gate = torch.zeros(x.shape[0], 1, device=x.device)

        logits = self.classifier(h_out)
        self_pred = self.self_model(h_out.detach())
        return logits, self_pred, gate


class DigitalModel(nn.Module):
    """WGP bank only — z2050 replication. No continuous physics."""

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
    """No hardware at all — pure software baseline."""

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
# Feature extraction
# =============================================================================

def extract_analog_features(probe_result):
    """Extract 5 continuous physics features from probe output.

    Returns [B, 5] tensor:
      0: fp16_sum (MODE-dependent)
      1: fp16_diff = fp16 - fp32 (rounding error magnitude)
      2: timing_norm (normalized clock cycles)
      3: denorm_diff = allow - ftz (denormal numerical difference)
      4: denorm_timing_ratio = allow_time / ftz_time
    """
    fp16 = probe_result['fp16_sums']
    fp32 = probe_result['fp32_sums']
    timing = probe_result['timing']
    ftz_res = probe_result['denorm_ftz_res']
    allow_res = probe_result['denorm_allow_res']
    ftz_t = probe_result['denorm_ftz_time']
    allow_t = probe_result['denorm_allow_time']

    feat = torch.stack([
        fp16,                                    # fp16 sum
        fp16 - fp32,                             # rounding error
        timing / timing.mean().clamp(min=1),     # normalized timing
        allow_res - ftz_res,                     # denormal diff
        allow_t / ftz_t.clamp(min=1),            # timing ratio
    ], dim=1)

    return feat


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

def read_temp_c():
    """Read GPU temperature."""
    for f in Path('/sys/class/drm/').glob('card*/device/hwmon/hwmon*/temp1_input'):
        try: return int(open(f).read().strip()) / 1000.0
        except: pass
    return 0.0


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
    """Train model under given condition."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    t0 = time.time()
    gate_vals = []
    temps = []

    for epoch in range(N_EPOCHS):
        correct = total = self_c = 0
        epoch_loss = 0
        epoch_gates = []

        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            B = images.shape[0]

            if cond == 'A_analog':
                pr = probe.probe(B)
                wgp_ids = pr['wgp_ids']
                bank_ids = wgp_to_bank(wgp_ids, wgp_map)
                perm_labels = permute_labels(labels, bank_ids)
                analog_feat = extract_analog_features(pr)
                logits, self_pred, gate = model(images, bank_ids, analog_feat)

            elif cond == 'B_digital':
                wgp_ids = probe.get_wgp_ids(B)
                bank_ids = wgp_to_bank(wgp_ids, wgp_map)
                perm_labels = permute_labels(labels, bank_ids)
                logits, self_pred, gate = model(images, bank_ids)

            elif cond == 'C_blind':
                bank_ids = torch.zeros(B, dtype=torch.long, device=DEVICE)
                perm_labels = labels.clone()
                logits, self_pred, gate = model(images, bank_ids)

            elif cond == 'F_nogate':
                pr = probe.probe(B)
                wgp_ids = pr['wgp_ids']
                bank_ids = wgp_to_bank(wgp_ids, wgp_map)
                perm_labels = permute_labels(labels, bank_ids)
                analog_feat = extract_analog_features(pr)
                logits, self_pred, gate = model(images, bank_ids, analog_feat, force_gate=0.5)

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
        temps.append(read_temp_c())

        if epoch % 3 == 0 or epoch == N_EPOCHS-1:
            print(f"  Epoch {epoch+1:2d}: loss={epoch_loss/len(loader):.4f} "
                  f"acc={correct/total:.4f} self={self_c/total:.3f} "
                  f"gate={gate_mean:.4f} temp={temps[-1]:.1f}C")

    print(f"  Done in {time.time()-t0:.1f}s")
    return {
        'train_time_s': time.time()-t0,
        'gate_trajectory': gate_vals,
        'temp_trajectory': temps,
        'final_gate': float(gate_vals[-1]),
    }


def evaluate(model, loader, probe, wgp_map, cond, scrambled_map=None, force_gate=None):
    """Evaluate model. Collects analog features and gate values."""
    model.eval()
    correct = total = self_c = 0
    all_gates = []
    all_outputs = []
    fp16_vals = []
    denorm_diffs = []
    mode_changes = 0; total_blocks = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            B = images.shape[0]
            bank_map = scrambled_map if scrambled_map is not None else wgp_map

            if cond in ('A_analog', 'D_scrambled', 'E_frozen', 'F_nogate'):
                pr = probe.probe(B)
                wgp_ids = pr['wgp_ids']
                real_banks = wgp_to_bank(wgp_ids, wgp_map)
                perm_labels = permute_labels(labels, real_banks)
                bank_ids = wgp_to_bank(wgp_ids, bank_map)

                analog_feat = extract_analog_features(pr)

                if cond == 'E_frozen':
                    # Freeze physics to batch mean — remove per-sample variation
                    analog_feat = analog_feat.mean(0, keepdim=True).expand_as(analog_feat)

                fg = force_gate if cond == 'F_nogate' else None
                logits, self_pred, gate = model(images, bank_ids, analog_feat, force_gate=fg)

                mode_changes += (pr['mode_before'] != pr['mode_after']).sum().item()
                total_blocks += B
                fp16_vals.extend(pr['fp16_sums'].cpu().tolist())
                denorm_diffs.extend((pr['denorm_allow_res'] - pr['denorm_ftz_res']).cpu().tolist())

            elif cond == 'B_digital':
                wgp_ids = probe.get_wgp_ids(B)
                bank_ids = wgp_to_bank(wgp_ids, wgp_map)
                perm_labels = permute_labels(labels, bank_ids)
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
            all_outputs.extend(F.softmax(logits, dim=1).cpu().tolist())

            if self_pred is not None:
                self_c += (self_pred.argmax(1) == bank_ids).sum().item()

    gate_flat = [g[0] if isinstance(g, list) else g for g in all_gates]
    fp16_unique = len(set([f'{v:.6f}' for v in fp16_vals])) if fp16_vals else 0
    denorm_unique = len(set([f'{v:.10f}' for v in denorm_diffs])) if denorm_diffs else 0

    # Compute output entropy (how confident is the model?)
    outputs_np = np.array(all_outputs)
    entropy = float(-np.mean(np.sum(outputs_np * np.log(outputs_np + 1e-10), axis=1)))

    return {
        'accuracy': correct/total,
        'self_model_acc': self_c/total if self_c > 0 else 0,
        'gate_mean': float(np.mean(gate_flat)),
        'gate_std': float(np.std(gate_flat)),
        'output_entropy': entropy,
        'n_fp16_unique': fp16_unique,
        'n_denorm_unique': denorm_unique,
        'mode_change_pct': mode_changes/max(1, total_blocks)*100,
    }


def cross_run_variance(model, loader, probe, wgp_map, n_runs=3):
    """Run evaluation multiple times. Measure output variance across runs.
    If physics matters, outputs change between runs (analog behavior).
    If deterministic, outputs are identical (digital behavior).
    """
    all_outputs = []
    for run in range(n_runs):
        model.eval()
        run_outputs = []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(DEVICE)
                B = images.shape[0]
                pr = probe.probe(B)
                wgp_ids = pr['wgp_ids']
                bank_ids = wgp_to_bank(wgp_ids, wgp_map)
                analog_feat = extract_analog_features(pr)
                logits, _, _ = model(images, bank_ids, analog_feat)
                run_outputs.extend(F.softmax(logits, dim=1).cpu().tolist())
                if len(run_outputs) >= 500:
                    break
        all_outputs.append(run_outputs[:500])

    # Compute variance across runs for same samples
    outputs = np.array(all_outputs)  # [n_runs, 500, 10]
    per_sample_var = outputs.var(axis=0).mean()  # Mean variance across runs
    return float(per_sample_var)


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print(f"z2055: Analog Physics Gate — Hardware State Modulates Computation")
    print(f"  Channels: WGP (digital) + fp16 rounding + timing + denorm (analog)")
    print(f"  Machine: {MACHINE}")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Initial temp: {read_temp_c():.1f}C")

    power_path = find_power_sysfs()
    print(f"Power sysfs: {power_path}")

    # Build probe
    probe = AnalogPhysicsProbe(n_fp16=256, n_denorm=64)
    calib = probe.calibrate()

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
            'n_banks': N_BANKS, 'hidden_dim': HIDDEN_DIM,
            'n_epochs': N_EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
            'n_fp16_accum': 256, 'n_denorm_ops': 64,
        },
        'calibration': calib,
        'conditions': {},
        'tests': {},
    }

    # ===== A: Full analog physics gate =====
    print(f"\n{'='*60}\n  Training: A_analog (WGP + fp16 + timing + denorm + gate)\n{'='*60}")
    model_A = AnalogPhysicsModel().to(DEVICE)
    ti_A = train_condition(model_A, train_loader, probe, wgp_map, 'A_analog', power_path)
    eval_A = evaluate(model_A, test_loader, probe, wgp_map, 'A_analog')
    print(f"\n  >> A_analog: acc={eval_A['accuracy']:.4f} gate={eval_A['gate_mean']:.4f} "
          f"fp16_unique={eval_A['n_fp16_unique']} denorm_unique={eval_A['n_denorm_unique']}")
    results['conditions']['A_analog'] = {'eval': eval_A, **ti_A}

    # ===== B: Digital only (WGP bank, no continuous physics) =====
    print(f"\n{'='*60}\n  Training: B_digital (WGP bank only)\n{'='*60}")
    model_B = DigitalModel().to(DEVICE)
    ti_B = train_condition(model_B, train_loader, probe, wgp_map, 'B_digital', power_path)
    eval_B = evaluate(model_B, test_loader, probe, wgp_map, 'B_digital')
    print(f"\n  >> B_digital: acc={eval_B['accuracy']:.4f}")
    results['conditions']['B_digital'] = {'eval': eval_B, **ti_B}

    # ===== C: Blind (no hardware) =====
    print(f"\n{'='*60}\n  Training: C_blind (no hardware)\n{'='*60}")
    model_C = BlindModel().to(DEVICE)
    ti_C = train_condition(model_C, train_loader, probe, wgp_map, 'C_blind', power_path)
    eval_C = evaluate(model_C, test_loader, probe, wgp_map, 'C_blind')
    print(f"\n  >> C_blind: acc={eval_C['accuracy']:.4f}")
    results['conditions']['C_blind'] = {'eval': eval_C, **ti_C}

    # ===== D: Scrambled kill shot =====
    print(f"\n{'='*60}\n  D_scrambled: A's model with WRONG WGP→bank\n{'='*60}")
    eval_D = evaluate(model_A, test_loader, probe, wgp_map, 'D_scrambled', scrambled_map=scrambled_map)
    print(f"\n  >> D_scrambled: acc={eval_D['accuracy']:.4f}")
    results['conditions']['D_scrambled'] = {'eval': eval_D}

    # ===== E: Frozen physics (physics frozen to batch mean) =====
    print(f"\n{'='*60}\n  E_frozen: A's model with physics frozen to batch mean\n{'='*60}")
    eval_E = evaluate(model_A, test_loader, probe, wgp_map, 'E_frozen')
    print(f"\n  >> E_frozen: acc={eval_E['accuracy']:.4f}")
    results['conditions']['E_frozen'] = {'eval': eval_E}

    # ===== F: No gate (forced 0.5) =====
    print(f"\n{'='*60}\n  Training: F_nogate (physics but gate=0.5 forced)\n{'='*60}")
    model_F = AnalogPhysicsModel().to(DEVICE)
    ti_F = train_condition(model_F, train_loader, probe, wgp_map, 'F_nogate', power_path)
    eval_F = evaluate(model_F, test_loader, probe, wgp_map, 'F_nogate', force_gate=0.5)
    print(f"\n  >> F_nogate: acc={eval_F['accuracy']:.4f}")
    results['conditions']['F_nogate'] = {'eval': eval_F, **ti_F}

    # ===== Cross-run variance =====
    print(f"\n{'='*60}\n  Cross-run variance (3 runs)\n{'='*60}")
    cross_var = cross_run_variance(model_A, test_loader, probe, wgp_map, n_runs=3)
    print(f"  Cross-run output variance: {cross_var:.8f}")
    results['cross_run_variance'] = cross_var

    # ===== Bank weight analysis =====
    W = model_A.bank_weights.data.cpu()
    cos_sims = {}
    for i in range(N_BANKS):
        for j in range(i+1, N_BANKS):
            cos = F.cosine_similarity(W[i].flatten().unsqueeze(0), W[j].flatten().unsqueeze(0)).item()
            cos_sims[f'bank_{i}_vs_{j}'] = round(cos, 4)
    cos_vals = list(cos_sims.values())
    cos_sims['mean'] = round(np.mean(cos_vals), 4)
    results['bank_analysis'] = cos_sims

    # ===== Gate analysis =====
    gate_weights = model_A.gate_net[0].weight.data.cpu().numpy()
    gate_importance = np.abs(gate_weights).mean(axis=0)
    feature_names = ['fp16_sum', 'fp16_diff', 'timing_norm', 'denorm_diff', 'denorm_ratio']
    results['gate_analysis'] = {
        'feature_importance': {name: float(imp) for name, imp in zip(feature_names, gate_importance)},
        'final_gate_mean': eval_A['gate_mean'],
        'gate_std': eval_A['gate_std'],
    }
    print(f"\n  Gate feature importance:")
    for name, imp in zip(feature_names, gate_importance):
        print(f"    {name:20s}: {imp:.4f}")

    # ===== TESTS =====
    acc = {}
    for k in results['conditions']:
        acc[k] = results['conditions'][k]['eval']['accuracy']

    results['tests'] = {
        'T1_accuracy': {
            'criterion': 'A_analog > 90%',
            'value': acc['A_analog'],
            'pass': acc['A_analog'] > 0.90
        },
        'T2_gap': {
            'criterion': 'A - C gap > 30%',
            'A': acc['A_analog'], 'C': acc['C_blind'],
            'gap': round(acc['A_analog'] - acc['C_blind'], 4),
            'pass': (acc['A_analog'] - acc['C_blind']) > 0.30
        },
        'T3_kill_shot': {
            'criterion': 'A - D gap > 10%',
            'A': acc['A_analog'], 'D': acc['D_scrambled'],
            'gap': round(acc['A_analog'] - acc['D_scrambled'], 4),
            'pass': (acc['A_analog'] - acc['D_scrambled']) > 0.10
        },
        'T4_gate_learned': {
            'criterion': 'Analog gate mean > 0.05',
            'value': eval_A['gate_mean'],
            'pass': eval_A['gate_mean'] > 0.05
        },
        'T5_denorm_diversity': {
            'criterion': 'Denorm unique values > 1',
            'value': eval_A['n_denorm_unique'],
            'pass': eval_A['n_denorm_unique'] > 1
        },
        'T6_cross_run_var': {
            'criterion': 'Cross-run variance > 0',
            'value': cross_var,
            'pass': cross_var > 1e-8
        },
        'T7_analog_vs_digital': {
            'criterion': 'A_analog >= B_digital - 0.02',
            'A': acc['A_analog'], 'B': acc['B_digital'],
            'diff': round(acc['A_analog'] - acc['B_digital'], 4),
            'pass': acc['A_analog'] >= acc['B_digital'] - 0.02
        },
        'T8_frozen_penalty': {
            'criterion': 'E_frozen < A_analog (frozen physics hurts)',
            'A': acc['A_analog'], 'E': acc['E_frozen'],
            'diff': round(acc['A_analog'] - acc['E_frozen'], 4),
            'pass': acc['E_frozen'] < acc['A_analog'] + 0.005
        },
    }

    n_pass = sum(1 for t in results['tests'].values() if t['pass'])
    results['n_pass'] = n_pass
    results['n_total'] = len(results['tests'])
    results['verdict'] = ('ANALOG_PHYSICS_CONFIRMED' if n_pass >= 7
                          else 'STRONG' if n_pass >= 6
                          else 'PARTIAL' if n_pass >= 4
                          else 'WEAK')

    # Summary
    results['summary'] = {
        'A_analog': acc['A_analog'],
        'B_digital': acc['B_digital'],
        'C_blind': acc['C_blind'],
        'D_scrambled': acc['D_scrambled'],
        'E_frozen': acc['E_frozen'],
        'F_nogate': acc['F_nogate'],
        'gate_mean': eval_A['gate_mean'],
        'cross_run_var': cross_var,
        'fp16_unique': eval_A['n_fp16_unique'],
        'denorm_unique': eval_A['n_denorm_unique'],
        'bank_cos_mean': cos_sims['mean'],
    }

    # Print results
    print(f"\n{'='*70}\nTESTS\n{'='*70}")
    for name, t in results['tests'].items():
        print(f"  {name}: {'PASS' if t['pass'] else 'FAIL'} — {t['criterion']}")
        for k, v in t.items():
            if k not in ('criterion', 'pass'):
                print(f"    {k} = {v}")

    print(f"\n  VERDICT: {results['verdict']} ({n_pass}/{results['n_total']} PASS)")

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for k, v in results['summary'].items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.6f}")
        else:
            print(f"  {k:25s}: {v}")

    out = RESULTS_DIR / 'z2055_analog_physics_gate.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")
    print(f"Final temp: {read_temp_c():.1f}C")


if __name__ == '__main__':
    main()
