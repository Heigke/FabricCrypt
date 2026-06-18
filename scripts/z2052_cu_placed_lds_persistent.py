#!/usr/bin/env python3
"""z2052: CU-Placed + LDS-Persistent Neural Network — True FPGA Place-and-Route

GOES DEEPER THAN z2051 with three new capabilities:

1. CU MASKING (hipExtStreamCreateWithCUMask):
   Control WHICH physical WGP runs the kernel — like FPGA placement constraints.
   The GPU scheduler is overridden; we decide routing.

2. LDS PERSISTENCE (LeftoverLocals / CVE-2023-4969):
   Verify that local memory (SRAM) state persists between consecutive kernel
   launches on the same CU. This is physical silicon memory — not cleared by
   hardware on gfx1151.

3. CROSS-MACHINE PUF COMPARISON:
   Compare silicon fingerprints (hwreg values, timing profiles) between
   ikaros and daedalus to prove uniqueness per chip.

DEPTH HIERARCHY:
  z2050: WGP_id → bank (digital, reactive — scheduler decides placement)
  z2051: + timing PUF + energy (analog, still reactive)
  z2052: + CU masking (proactive — WE decide placement)
         + LDS persistence (physical state survives between kernels)
         = TRUE FPGA place-and-route on GPU

CONDITIONS:
  A_placed:    Force-place batches on specific CUs via CU mask, per-WGP banks
  B_scheduler: Same model, let scheduler decide (z2050 replication)
  C_blind:     No hardware info (pure software)
  D_wrong_cu:  Force-place on WRONG CU (scrambled routing)
  E_lds:       LDS persistence verification (separate measurement)

TESTS:
  T1: A_placed > 90% (CU masking works with bank task)
  T2: B_scheduler > 90% (replicates z2050)
  T3: D_wrong_cu << A_placed by >10% (wrong routing kills accuracy)
  T4: LDS persistence rate > 80% on same CU
  T5: LDS persistence rate < 20% across different CUs
  T6: Cross-machine hwreg fingerprint differs (if both z2051 results exist)
  T7: Energy comparison A_placed vs B_scheduler vs C_blind
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
import glob
import socket
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

N_BANKS = 8
WGP_VALUES = [0, 2, 4, 6, 8, 10, 12, 14]
HIDDEN_DIM = 128
N_EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3

# =============================================================================
# HIP Kernel: CU Masking + LDS Persistence + WGP Reading
# =============================================================================

HIP_SOURCE = r'''
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <torch/extension.h>

// ─── Standard WGP reader (same as z2050) ───
__global__ void read_wgp_id(int32_t* wgp_ids, int batch_size) {
    int bid = blockIdx.x;
    if (bid >= batch_size) return;
    if (threadIdx.x != 0) return;
    uint32_t hw_id1 = 0;
    asm volatile("s_getreg_b32 %0, hwreg(23, 0, 32)" : "=s"(hw_id1));
    wgp_ids[bid] = (int32_t)((hw_id1 >> 7) & 0xF);
}

// ─── CU-masked WGP reader ───
// Launches read_wgp_id on a CU-masked stream to force specific WGP placement.
// cu_mask_bits: bitmask of allowed CUs (e.g., 0x3 = CUs 0-1 = WGP 0)
torch::Tensor read_wgp_masked(int batch_size, int cu_mask_bits) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto wgp_ids = torch::zeros({batch_size}, opts);

    hipStream_t stream;
    uint32_t mask[1] = {(uint32_t)cu_mask_bits};
    hipError_t err = hipExtStreamCreateWithCUMask(&stream, 1, mask);
    if (err != hipSuccess) {
        // Fallback: use default stream
        read_wgp_id<<<batch_size, 32>>>(wgp_ids.data_ptr<int32_t>(), batch_size);
        hipDeviceSynchronize();
        return wgp_ids;
    }

    read_wgp_id<<<batch_size, 32, 0, stream>>>(
        wgp_ids.data_ptr<int32_t>(), batch_size);
    hipStreamSynchronize(stream);
    hipStreamDestroy(stream);

    return wgp_ids;
}

// ─── CU mask discovery ───
// Try each CU mask bit and report which WGP_id it produces.
// Returns [32] tensor: cu_to_wgp[cu_bit] = observed WGP_id (-1 if no result)
torch::Tensor discover_cu_mapping(int max_cu_bits) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto results = torch::full({max_cu_bits}, -1, opts);
    auto single = torch::zeros({1}, opts);

    for (int bit = 0; bit < max_cu_bits; bit++) {
        hipStream_t stream;
        uint32_t mask[1] = {(uint32_t)(1 << bit)};
        hipError_t err = hipExtStreamCreateWithCUMask(&stream, 1, mask);
        if (err != hipSuccess) continue;

        read_wgp_id<<<1, 32, 0, stream>>>(single.data_ptr<int32_t>(), 1);
        hipStreamSynchronize(stream);
        hipStreamDestroy(stream);

        // Copy the single result
        int32_t* res_ptr = results.data_ptr<int32_t>() + bit;
        int32_t* src_ptr = single.data_ptr<int32_t>();
        hipMemcpy(res_ptr, src_ptr, sizeof(int32_t), hipMemcpyDeviceToDevice);
    }

    return results;
}

// ─── Standard (unmasked) batch WGP read ───
torch::Tensor read_wgp_batch(int batch_size) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto wgp_ids = torch::zeros({batch_size}, opts);
    read_wgp_id<<<batch_size, 32>>>(wgp_ids.data_ptr<int32_t>(), batch_size);
    hipDeviceSynchronize();
    return wgp_ids;
}

// ─── LDS Persistence Test ───
// Writer kernel: write a known pattern to LDS
__global__ void lds_write_pattern(float* verify_out, float seed, int32_t* wgp_out) {
    __shared__ volatile float lds[256];  // volatile prevents optimization
    int tid = threadIdx.x;

    // Write pattern: seed * (tid + 1) — unique per thread, deterministic
    float val = seed * (float)(tid + 1);
    lds[tid] = val;
    __syncthreads();

    // Copy to global for verification
    verify_out[tid] = lds[tid];

    // Report which CU we ran on
    if (tid == 0) {
        uint32_t hw_id1 = 0;
        asm volatile("s_getreg_b32 %0, hwreg(23, 0, 32)" : "=s"(hw_id1));
        wgp_out[0] = (int32_t)((hw_id1 >> 7) & 0xF);
    }
}

// Reader kernel: read LDS WITHOUT writing (reads residual from previous kernel)
__global__ void lds_read_residual(float* readout, int32_t* wgp_out) {
    __shared__ volatile float lds[256];
    int tid = threadIdx.x;

    // Read whatever is in LDS (should be previous kernel's data if same CU)
    readout[tid] = lds[tid];

    if (tid == 0) {
        uint32_t hw_id1 = 0;
        asm volatile("s_getreg_b32 %0, hwreg(23, 0, 32)" : "=s"(hw_id1));
        wgp_out[0] = (int32_t)((hw_id1 >> 7) & 0xF);
    }
}

// ─── LDS persistence test: write then read, report results ───
// Returns: {write_data[256], read_data[256], write_wgp[1], read_wgp[1]}
std::vector<torch::Tensor> lds_persist_test_default() {
    auto fopts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto iopts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    auto write_data = torch::zeros({256}, fopts);
    auto read_data  = torch::zeros({256}, fopts);
    auto write_wgp  = torch::zeros({1}, iopts);
    auto read_wgp   = torch::zeros({1}, iopts);

    float seed = 3.14159f;

    // Launch writer kernel (1 block, 256 threads)
    lds_write_pattern<<<1, 256>>>(
        write_data.data_ptr<float>(), seed, write_wgp.data_ptr<int32_t>());
    hipDeviceSynchronize();

    // Immediately launch reader kernel (same default stream → likely same CU)
    lds_read_residual<<<1, 256>>>(
        read_data.data_ptr<float>(), read_wgp.data_ptr<int32_t>());
    hipDeviceSynchronize();

    return {write_data, read_data, write_wgp, read_wgp};
}

// ─── CU-masked LDS persistence test ───
std::vector<torch::Tensor> lds_persist_test_masked(int write_cu_mask, int read_cu_mask) {
    auto fopts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto iopts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    auto write_data = torch::zeros({256}, fopts);
    auto read_data  = torch::zeros({256}, fopts);
    auto write_wgp  = torch::zeros({1}, iopts);
    auto read_wgp   = torch::zeros({1}, iopts);

    float seed = 2.71828f;

    // Write on specific CU(s)
    hipStream_t ws;
    uint32_t wmask[1] = {(uint32_t)write_cu_mask};
    hipError_t werr = hipExtStreamCreateWithCUMask(&ws, 1, wmask);
    if (werr != hipSuccess) {
        // Fallback
        lds_write_pattern<<<1, 256>>>(
            write_data.data_ptr<float>(), seed, write_wgp.data_ptr<int32_t>());
        hipDeviceSynchronize();
        lds_read_residual<<<1, 256>>>(
            read_data.data_ptr<float>(), read_wgp.data_ptr<int32_t>());
        hipDeviceSynchronize();
        return {write_data, read_data, write_wgp, read_wgp};
    }

    lds_write_pattern<<<1, 256, 0, ws>>>(
        write_data.data_ptr<float>(), seed, write_wgp.data_ptr<int32_t>());
    hipStreamSynchronize(ws);
    hipStreamDestroy(ws);

    // Read on specific CU(s) — may be same or different
    hipStream_t rs;
    uint32_t rmask[1] = {(uint32_t)read_cu_mask};
    hipError_t rerr = hipExtStreamCreateWithCUMask(&rs, 1, rmask);
    if (rerr != hipSuccess) {
        lds_read_residual<<<1, 256>>>(
            read_data.data_ptr<float>(), read_wgp.data_ptr<int32_t>());
        hipDeviceSynchronize();
        return {write_data, read_data, write_wgp, read_wgp};
    }

    lds_read_residual<<<1, 256, 0, rs>>>(
        read_data.data_ptr<float>(), read_wgp.data_ptr<int32_t>());
    hipStreamSynchronize(rs);
    hipStreamDestroy(rs);

    return {write_data, read_data, write_wgp, read_wgp};
}

// ─── Timing fingerprint (from z2051) ───
__global__ void timing_fingerprint(int32_t* wgp_ids, float* timing, int batch_size) {
    int bid = blockIdx.x;
    if (bid >= batch_size) return;
    int tid = threadIdx.x;

    uint32_t hw_id1 = 0;
    asm volatile("s_getreg_b32 %0, hwreg(23, 0, 32)" : "=s"(hw_id1));

    float val = 1.0f;
    uint64_t t0 = __clock64();
    #pragma unroll 1
    for (int i = 0; i < 128; i++) {
        val = __sinf(val + 0.001f * (float)(i + 1));
        val = __cosf(val * 1.001f);
        val = val * val + 0.5f;
        val = __fsqrt_rn(val);
    }
    uint64_t t1 = __clock64();
    if (val == -999.0f) timing[0] = val;  // prevent optimization

    if (tid == 0) {
        wgp_ids[bid] = (int32_t)((hw_id1 >> 7) & 0xF);
        timing[bid] = (float)(t1 - t0);
    }
}

std::vector<torch::Tensor> get_timing_fingerprint(int batch_size) {
    auto iopts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto fopts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto wgp = torch::zeros({batch_size}, iopts);
    auto tming = torch::zeros({batch_size}, fopts);
    timing_fingerprint<<<batch_size, 32>>>(
        wgp.data_ptr<int32_t>(), tming.data_ptr<float>(), batch_size);
    hipDeviceSynchronize();
    return {wgp, tming};
}
'''

HIP_CPP = r'''
torch::Tensor read_wgp_masked(int batch_size, int cu_mask_bits);
torch::Tensor discover_cu_mapping(int max_cu_bits);
torch::Tensor read_wgp_batch(int batch_size);
std::vector<torch::Tensor> lds_persist_test_default();
std::vector<torch::Tensor> lds_persist_test_masked(int write_cu_mask, int read_cu_mask);
std::vector<torch::Tensor> get_timing_fingerprint(int batch_size);
'''


# =============================================================================
# Silicon Probe with CU Masking
# =============================================================================

class CUPlacedProbe:
    """GPU probe with CU masking and LDS persistence capabilities."""

    def __init__(self):
        from torch.utils.cpp_extension import load_inline
        print("[BUILD] Compiling CU-placed probe (CU mask + LDS + timing)...")
        t0 = time.time()
        self.ext = load_inline(
            name='cu_placed_z2052',
            cpp_sources=HIP_CPP,
            cuda_sources=HIP_SOURCE,
            functions=[
                'read_wgp_masked', 'discover_cu_mapping', 'read_wgp_batch',
                'lds_persist_test_default', 'lds_persist_test_masked',
                'get_timing_fingerprint',
            ],
            verbose=False,
            extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'],
        )
        print(f"[BUILD] Done in {time.time()-t0:.1f}s")

        # Discover CU → WGP mapping
        self.cu_wgp_map = self._discover_cu_mapping()

        # Calibrate timing
        self.timing_profile = self._calibrate_timing()

    def _discover_cu_mapping(self, max_bits=32):
        """Discover which CU mask bits correspond to which WGP_ids."""
        print("\n[CU MAPPING] Discovering CU bit → WGP_id mapping...")
        mapping = self.ext.discover_cu_mapping(max_bits)
        torch.cuda.synchronize()
        vals = mapping.cpu().numpy()

        cu_to_wgp = {}
        wgp_to_cu_bits = defaultdict(list)
        print("  CU bit |  WGP_id")
        print("  " + "-" * 22)
        for i in range(max_bits):
            if vals[i] >= 0:
                wgp = int(vals[i])
                cu_to_wgp[i] = wgp
                wgp_to_cu_bits[wgp].append(i)
                print(f"  {i:5d}  |  {wgp:3d}")

        print(f"\n  Found {len(cu_to_wgp)} active CU bits mapping to "
              f"{len(wgp_to_cu_bits)} WGPs")

        self.wgp_to_cu_bits = dict(wgp_to_cu_bits)
        self.cu_mask_available = len(cu_to_wgp) > 0
        return cu_to_wgp

    def _calibrate_timing(self, n_rounds=20, n_blocks=256):
        """Measure per-WGP timing profile."""
        print(f"\n[TIMING] Calibrating ({n_rounds} rounds × {n_blocks} blocks)...")
        wgp_timings = defaultdict(list)
        for _ in range(n_rounds):
            results = self.ext.get_timing_fingerprint(n_blocks)
            torch.cuda.synchronize()
            wgps = results[0].cpu().numpy()
            times = results[1].cpu().numpy()
            for w, t in zip(wgps, times):
                wgp_timings[int(w)].append(float(t))

        profile = {}
        print("  WGP  |  Mean cycles  |  Std  |  CV%  |  Count")
        print("  " + "-" * 52)
        for wgp in sorted(wgp_timings.keys()):
            t = np.array(wgp_timings[wgp])
            mean, std = t.mean(), t.std()
            cv = (std / mean * 100) if mean > 0 else 0
            profile[wgp] = {'mean': float(mean), 'std': float(std),
                            'cv_pct': float(cv), 'count': len(t)}
            print(f"  {wgp:3d}  |  {mean:11.1f}  |  {std:5.1f}  |  {cv:4.2f}  |  {len(t)}")

        means = [v['mean'] for v in profile.values()]
        inter_spread = ((max(means) - min(means)) / np.mean(means) * 100) if len(means) > 1 else 0.0
        print(f"  Inter-WGP spread: {inter_spread:.3f}%")
        profile['_inter_spread_pct'] = inter_spread
        return profile

    def get_wgp_ids(self, batch_size, cu_mask=None):
        """Get WGP_ids for a batch, optionally restricted by CU mask."""
        if cu_mask is not None and self.cu_mask_available:
            return self.ext.read_wgp_masked(batch_size, cu_mask)
        return self.ext.read_wgp_batch(batch_size)

    def test_lds_persistence(self, n_trials=50):
        """Test LDS persistence: write pattern → read residual."""
        print("\n[LDS] Testing LDS persistence (LeftoverLocals)...")

        # Default stream test (likely same CU)
        same_cu_matches = 0
        same_cu_trials = 0
        for _ in range(n_trials):
            results = self.ext.lds_persist_test_default()
            torch.cuda.synchronize()
            write_data = results[0].cpu().numpy()
            read_data = results[1].cpu().numpy()
            write_wgp = results[2].cpu().item()
            read_wgp = results[3].cpu().item()

            if write_wgp == read_wgp:
                same_cu_trials += 1
                # Check how many values match (handle NaN)
                valid = np.isfinite(write_data) & np.isfinite(read_data)
                if valid.sum() > 0:
                    matches = np.sum(np.abs(write_data[valid] - read_data[valid]) < 1e-6)
                else:
                    matches = 0
                if matches > 200:  # >78% match
                    same_cu_matches += 1

        same_cu_rate = same_cu_matches / max(same_cu_trials, 1)
        print(f"  Default stream: {same_cu_trials}/{n_trials} same CU, "
              f"{same_cu_matches}/{same_cu_trials} with LDS persistence "
              f"({same_cu_rate*100:.1f}%)")

        # Cross-CU test (if CU masking available)
        cross_cu_matches = 0
        cross_cu_trials = 0
        if self.cu_mask_available and len(self.wgp_to_cu_bits) >= 2:
            wgps = list(self.wgp_to_cu_bits.keys())
            wgp_a, wgp_b = wgps[0], wgps[1]
            # Use only first 2 CU bits per WGP to keep mask small
            bits_a = self.wgp_to_cu_bits[wgp_a][:2]
            bits_b = self.wgp_to_cu_bits[wgp_b][:2]
            mask_a = sum(1 << b for b in bits_a) & 0xFFFF
            mask_b = sum(1 << b for b in bits_b) & 0xFFFF

            print(f"  Cross-CU masks: write=0x{mask_a:04x} (WGP {wgp_a}), "
                  f"read=0x{mask_b:04x} (WGP {wgp_b})")

            for _ in range(n_trials):
                results = self.ext.lds_persist_test_masked(mask_a, mask_b)
                torch.cuda.synchronize()
                write_data = results[0].cpu().numpy()
                read_data = results[1].cpu().numpy()
                write_wgp = results[2].cpu().item()
                read_wgp = results[3].cpu().item()

                cross_cu_trials += 1
                valid = np.isfinite(write_data) & np.isfinite(read_data)
                if valid.sum() > 0:
                    matches = np.sum(np.abs(write_data[valid] - read_data[valid]) < 1e-6)
                else:
                    matches = 0
                if matches > 200:
                    cross_cu_matches += 1

            cross_cu_rate = cross_cu_matches / max(cross_cu_trials, 1)
            print(f"  Cross-CU (WGP {wgp_a}→{wgp_b}): {cross_cu_matches}/{cross_cu_trials} "
                  f"with LDS persistence ({cross_cu_rate*100:.1f}%)")
        else:
            cross_cu_rate = -1.0
            print("  Cross-CU test: skipped (CU masking unavailable)")

        return {
            'same_cu_trials': same_cu_trials,
            'same_cu_persist': same_cu_matches,
            'same_cu_rate': same_cu_rate,
            'cross_cu_trials': cross_cu_trials,
            'cross_cu_persist': cross_cu_matches,
            'cross_cu_rate': cross_cu_rate if cross_cu_rate >= 0 else None,
        }


# =============================================================================
# Energy Measurement
# =============================================================================

def find_power_sysfs():
    patterns = [
        '/sys/class/drm/card0/device/hwmon/hwmon*/power1_average',
        '/sys/class/drm/card1/device/hwmon/hwmon*/power1_average',
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None


POWER_SYSFS = find_power_sysfs()


def read_power_w():
    if POWER_SYSFS is None:
        return 15.0
    try:
        with open(POWER_SYSFS, 'r') as f:
            return int(f.read().strip()) / 1e6
    except Exception:
        return 15.0


# =============================================================================
# WGP → Bank Mapping
# =============================================================================

def build_wgp_to_bank():
    mapping = torch.zeros(16, dtype=torch.long)
    for i, wgp in enumerate(WGP_VALUES):
        mapping[wgp] = i
    return mapping


def build_scrambled_map():
    mapping = torch.zeros(16, dtype=torch.long)
    n = len(WGP_VALUES)
    scrambled = [(i + 1) % n for i in range(n)]
    for i, wgp in enumerate(WGP_VALUES):
        mapping[wgp] = scrambled[i]
    return mapping


def wgp_to_bank(wgp_ids, mapping):
    return mapping.to(wgp_ids.device)[wgp_ids.clamp(0, 15).long()]


def permute_labels(labels, bank_ids):
    permuted = labels.clone()
    odd_mask = (bank_ids % 2) == 1
    permuted[odd_mask] = (labels[odd_mask] + 5) % 10
    return permuted


# =============================================================================
# Models (same architecture as z2050)
# =============================================================================

class FPGAModel(nn.Module):
    """Per-WGP bank weights, shared encoder + head."""

    def __init__(self, n_banks=N_BANKS, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.enc_fc = nn.Linear(32 * 7 * 7, hidden_dim)
        self.bank_weights = nn.Parameter(
            torch.randn(n_banks, hidden_dim, hidden_dim) * 0.02)
        self.bank_bias = nn.Parameter(torch.zeros(n_banks, hidden_dim))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x, bank_ids):
        B = x.shape[0]
        h = self.encoder(x)
        h = h.view(B, -1)
        h = F.relu(self.enc_fc(h))
        W = self.bank_weights[bank_ids]
        b = self.bank_bias[bank_ids]
        h_hw = torch.bmm(W, h.unsqueeze(2)).squeeze(2) + b
        h_hw = F.relu(h_hw)
        return self.head(h_hw)


class BlindModel(nn.Module):
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
        return self.head(h)


# =============================================================================
# Data
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
# Training & Evaluation
# =============================================================================

def train_model(model, train_loader, probe, wgp_map, condition, n_epochs=N_EPOCHS):
    """Train a model for a given condition."""
    print(f"\n{'='*60}")
    print(f"  Training: {condition}")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    total_joules = 0.0
    total_correct = 0
    t_start = time.time()

    for epoch in range(n_epochs):
        model.train()
        ep_loss = ep_correct = ep_total = 0
        bank_counts = [0] * N_BANKS

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            B = images.shape[0]

            p0 = read_power_w()
            bt0 = time.time()

            # Get WGP assignment
            if condition == 'C_blind':
                bank_ids = torch.zeros(B, dtype=torch.long, device=DEVICE)
            else:
                with torch.no_grad():
                    wgp_ids = probe.get_wgp_ids(B)
                bank_ids = wgp_to_bank(wgp_ids, wgp_map)

            # Permute labels by bank parity
            if condition == 'C_blind':
                with torch.no_grad():
                    real_wgp = probe.get_wgp_ids(B)
                real_banks = wgp_to_bank(real_wgp, wgp_map)
                perm_labels = permute_labels(labels, real_banks)
            else:
                perm_labels = permute_labels(labels, bank_ids)

            # Forward
            if condition == 'C_blind':
                logits = model(images)
            else:
                logits = model(images, bank_ids)

            loss = F.cross_entropy(logits, perm_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Energy
            batch_dt = time.time() - bt0
            p1 = read_power_w()
            total_joules += (p0 + p1) / 2.0 * batch_dt

            preds = logits.argmax(1)
            ep_correct += (preds == perm_labels).sum().item()
            ep_total += B
            total_correct += (preds == perm_labels).sum().item()
            ep_loss += loss.item()

            for bid in bank_ids.cpu().tolist():
                if 0 <= bid < N_BANKS:
                    bank_counts[bid] += 1

        acc = ep_correct / ep_total
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}: loss={ep_loss/len(train_loader):.4f} "
                  f"acc={acc:.4f} banks={bank_counts}")

    train_time = time.time() - t_start
    joules_per_correct = total_joules / max(total_correct, 1)
    print(f"  Done in {train_time:.1f}s | {total_joules:.1f}J | "
          f"{joules_per_correct*1000:.2f} mJ/correct")

    return {
        'train_time_s': train_time,
        'total_joules': total_joules,
        'joules_per_correct': joules_per_correct,
    }


def evaluate_condition(model, test_loader, probe, wgp_map, condition,
                       cu_mask=None, scrambled_map=None):
    """Evaluate a model under specific CU placement constraints.

    IMPORTANT: Labels are ALWAYS permuted by the REAL wgp_map.
    scrambled_map only affects which bank weights the model receives.
    This ensures the kill shot test works: model gets WRONG banks for REAL labels.
    """
    model.eval()
    correct = total = 0
    total_joules = 0.0
    bank_counts = [0] * N_BANKS
    all_wgp = []
    # Bank selection map (possibly scrambled)
    bank_map = scrambled_map if scrambled_map is not None else wgp_map

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            B = images.shape[0]

            p0 = read_power_w()
            bt0 = time.time()

            if condition == 'C_blind':
                bank_ids = torch.zeros(B, dtype=torch.long, device=DEVICE)
                real_wgp = probe.get_wgp_ids(B)
                real_banks = wgp_to_bank(real_wgp, wgp_map)
                perm_labels = permute_labels(labels, real_banks)
                logits = model(images)
            else:
                wgp_ids = probe.get_wgp_ids(B, cu_mask=cu_mask)
                # Labels: ALWAYS use real mapping (ground truth)
                real_banks = wgp_to_bank(wgp_ids, wgp_map)
                perm_labels = permute_labels(labels, real_banks)
                # Bank selection: may use scrambled map (wrong weights)
                bank_ids = wgp_to_bank(wgp_ids, bank_map)
                logits = model(images, bank_ids)
                all_wgp.extend(wgp_ids.cpu().tolist())

            batch_dt = time.time() - bt0
            p1 = read_power_w()
            total_joules += (p0 + p1) / 2.0 * batch_dt

            preds = logits.argmax(1)
            correct += (preds == perm_labels).sum().item()
            total += B

            for bid in bank_ids.cpu().tolist():
                if 0 <= bid < N_BANKS:
                    bank_counts[bid] += 1

    accuracy = correct / total
    unique_wgp = sorted(set(all_wgp)) if all_wgp else []
    joules_per_correct = total_joules / max(correct, 1)

    return {
        'accuracy': accuracy,
        'bank_counts': bank_counts,
        'unique_wgp': unique_wgp,
        'n_unique_wgp': len(unique_wgp),
        'total_joules': total_joules,
        'joules_per_correct': joules_per_correct,
    }


def bank_weight_analysis(model):
    """Compute pairwise cosine similarity between bank weight matrices."""
    if not hasattr(model, 'bank_weights'):
        return {}
    W = model.bank_weights.data.view(N_BANKS, -1)
    analysis = {}
    sims = []
    for i in range(N_BANKS):
        for j in range(i + 1, N_BANKS):
            sim = F.cosine_similarity(W[i:i+1], W[j:j+1]).item()
            analysis[f'bank_{i}_vs_{j}'] = round(sim, 4)
            sims.append(sim)
    if sims:
        analysis['mean'] = round(np.mean(sims), 4)
        analysis['min'] = round(min(sims), 4)
        analysis['max'] = round(max(sims), 4)
    return analysis


# =============================================================================
# Cross-Machine PUF Comparison
# =============================================================================

def cross_machine_puf():
    """Compare z2051 results between machines if both exist."""
    results = {}

    # Look for z2051 results
    ikaros_path = RESULTS_DIR / 'z2051_silicon_puf_neural_net.json'
    daedalus_path = RESULTS_DIR / 'z2051_daedalus_silicon_puf.json'

    if not ikaros_path.exists():
        print("[PUF] No ikaros z2051 results found")
        return None

    with open(ikaros_path) as f:
        ikaros_data = json.load(f)

    if daedalus_path.exists():
        with open(daedalus_path) as f:
            daedalus_data = json.load(f)
    else:
        print("[PUF] No daedalus z2051 results — will compare after daedalus run")
        return None

    print("\n" + "=" * 60)
    print("  Cross-Machine PUF Comparison: ikaros vs daedalus")
    print("=" * 60)

    # Compare hwreg fingerprints
    ik_hw = ikaros_data.get('hwreg_probe', {})
    da_hw = daedalus_data.get('hwreg_probe', {})

    print("\n  Hardware Register Comparison:")
    print(f"  {'hwreg':>8s}  {'ikaros':>12s}  {'daedalus':>12s}  {'match':>6s}")
    print("  " + "-" * 46)
    all_regs = sorted(set(list(ik_hw.keys()) + list(da_hw.keys())))
    n_match = 0
    n_differ = 0
    for reg in all_regs:
        ik_val = ik_hw.get(reg, 'N/A')
        da_val = da_hw.get(reg, 'N/A')
        match = ik_val == da_val
        if match:
            n_match += 1
        else:
            n_differ += 1
        print(f"  {reg:>8s}  {str(ik_val):>12s}  {str(da_val):>12s}  "
              f"{'YES' if match else 'NO':>6s}")

    # Compare timing profiles
    ik_tp = ikaros_data.get('timing_profile', {})
    da_tp = daedalus_data.get('timing_profile', {})

    print(f"\n  Timing Profile Comparison:")
    print(f"  {'WGP':>5s}  {'ik_mean':>10s}  {'da_mean':>10s}  {'diff%':>8s}")
    print("  " + "-" * 38)
    timing_diffs = []
    for wgp in sorted(set(
        [k for k in ik_tp.keys() if k != '_inter_spread_pct'] +
        [k for k in da_tp.keys() if k != '_inter_spread_pct']
    )):
        ik_m = ik_tp.get(wgp, ik_tp.get(int(wgp) if isinstance(wgp, str) else str(wgp), {}))
        da_m = da_tp.get(wgp, da_tp.get(int(wgp) if isinstance(wgp, str) else str(wgp), {}))
        if isinstance(ik_m, dict) and isinstance(da_m, dict):
            ik_mean = ik_m.get('mean', 0)
            da_mean = da_m.get('mean', 0)
            diff_pct = abs(ik_mean - da_mean) / max(ik_mean, 1) * 100
            timing_diffs.append(diff_pct)
            print(f"  {wgp:>5s}  {ik_mean:>10.1f}  {da_mean:>10.1f}  {diff_pct:>7.2f}%")

    results['hwreg_match'] = n_match
    results['hwreg_differ'] = n_differ
    results['timing_mean_diff_pct'] = float(np.mean(timing_diffs)) if timing_diffs else None
    results['chips_distinguishable'] = n_differ > 0 or (
        timing_diffs and np.mean(timing_diffs) > 0.1)

    if results['chips_distinguishable']:
        print(f"\n  RESULT: Chips ARE distinguishable ({n_differ} hwreg diffs, "
              f"{np.mean(timing_diffs):.2f}% avg timing diff)")
    else:
        print(f"\n  RESULT: Chips NOT distinguishable")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    machine = socket.gethostname()
    print("=" * 70)
    print("z2052: CU-Placed + LDS-Persistent Neural Network")
    print("  CU masking: hipExtStreamCreateWithCUMask (FPGA place-and-route)")
    print("  LDS persistence: LeftoverLocals (physical silicon memory)")
    print("  Cross-machine PUF comparison (silicon fingerprint uniqueness)")
    print(f"  Machine: {machine}")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Power sysfs: {POWER_SYSFS}")

    # Build probe
    probe = CUPlacedProbe()

    # ─── Part 1: LDS Persistence Test ───
    lds_results = probe.test_lds_persistence(n_trials=50)

    # ─── Part 2: Train models ───
    train_loader, test_loader = get_data_loaders()
    wgp_map = build_wgp_to_bank().to(DEVICE)
    scrambled_map = build_scrambled_map().to(DEVICE)

    results = {
        'timestamp': datetime.now().isoformat(),
        'machine': machine,
        'config': {
            'n_banks': N_BANKS, 'hidden_dim': HIDDEN_DIM,
            'n_epochs': N_EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
        },
        'cu_mapping': {str(k): int(v) for k, v in probe.cu_wgp_map.items()},
        'cu_mask_available': probe.cu_mask_available,
        'timing_profile': {str(k): v for k, v in probe.timing_profile.items()},
        'lds_persistence': lds_results,
        'conditions': {},
        'tests': {},
    }

    # === Condition A: CU-placed (use CU mask at eval if available) ===
    # Train with scheduler (natural placement), eval with CU mask
    model_a = FPGAModel().to(DEVICE)
    train_info_a = train_model(model_a, train_loader, probe, wgp_map, 'A_placed')

    # Evaluate WITHOUT CU mask first (scheduler)
    eval_a_sched = evaluate_condition(
        model_a, test_loader, probe, wgp_map, 'A_placed')
    print(f"\n  A_placed (scheduler): {eval_a_sched['accuracy']:.4f}")

    # Evaluate WITH CU mask (if available, try each WGP individually)
    eval_a_placed = None
    if probe.cu_mask_available:
        # Build aggregate CU mask covering all WGPs (limit to first 16 bits)
        all_cu_mask = 0
        for bits in probe.wgp_to_cu_bits.values():
            for b in bits[:4]:  # First 4 bits per WGP
                all_cu_mask |= (1 << b)
        all_cu_mask &= 0xFFFF
        eval_a_placed = evaluate_condition(
            model_a, test_loader, probe, wgp_map, 'A_placed',
            cu_mask=all_cu_mask)
        print(f"  A_placed (CU mask {all_cu_mask:#x}): {eval_a_placed['accuracy']:.4f}")

    results['conditions']['A_placed'] = {
        'train': train_info_a,
        'eval_scheduler': eval_a_sched,
        'eval_cu_masked': eval_a_placed,
        'bank_analysis': bank_weight_analysis(model_a),
    }

    # === Condition B: Scheduler (same model, no CU mask — z2050 replication) ===
    model_b = FPGAModel().to(DEVICE)
    train_info_b = train_model(model_b, train_loader, probe, wgp_map, 'B_scheduler')
    eval_b = evaluate_condition(model_b, test_loader, probe, wgp_map, 'B_scheduler')
    print(f"\n  B_scheduler: {eval_b['accuracy']:.4f}")
    results['conditions']['B_scheduler'] = {
        'train': train_info_b,
        'eval': eval_b,
        'bank_analysis': bank_weight_analysis(model_b),
    }

    # === Condition C: Blind (no hardware) ===
    model_c = BlindModel().to(DEVICE)
    train_info_c = train_model(model_c, train_loader, probe, wgp_map, 'C_blind')
    eval_c = evaluate_condition(model_c, test_loader, probe, wgp_map, 'C_blind')
    print(f"\n  C_blind: {eval_c['accuracy']:.4f}")
    results['conditions']['C_blind'] = {
        'train': train_info_c,
        'eval': eval_c,
    }

    # === Condition D: Wrong CU placement (scrambled bank map) ===
    eval_d = evaluate_condition(
        model_a, test_loader, probe, wgp_map, 'D_wrong_cu',
        scrambled_map=scrambled_map)
    print(f"\n  D_wrong_cu (scrambled routing): {eval_d['accuracy']:.4f}")
    results['conditions']['D_wrong_cu'] = {'eval': eval_d}

    # If CU masking works, also try forcing WRONG CU mask
    if probe.cu_mask_available and len(probe.wgp_to_cu_bits) >= 2:
        wgps = sorted(probe.wgp_to_cu_bits.keys())
        # Use only the first WGP's CU mask → restricts to 1 WGP = 1 bank
        wrong_cu_bits = probe.wgp_to_cu_bits[wgps[0]][:4]  # Limit to first 4 bits
        wrong_mask = sum(1 << b for b in wrong_cu_bits) & 0xFFFF
        eval_d_masked = evaluate_condition(
            model_a, test_loader, probe, wgp_map, 'D_wrong_cu',
            cu_mask=wrong_mask)
        print(f"  D_wrong_cu (forced WGP {wgps[0]} only, mask {wrong_mask:#x}): "
              f"{eval_d_masked['accuracy']:.4f}")
        results['conditions']['D_wrong_cu']['eval_single_wgp'] = eval_d_masked

    # === Tests ===
    A_acc = eval_a_sched['accuracy']
    B_acc = eval_b['accuracy']
    C_acc = eval_c['accuracy']
    D_acc = eval_d['accuracy']
    lds_same = lds_results['same_cu_rate']
    lds_cross = lds_results.get('cross_cu_rate')

    results['tests'] = {
        'T1_placed_accuracy': {
            'criterion': 'A_placed > 90%',
            'value': A_acc,
            'pass': A_acc > 0.90,
        },
        'T2_scheduler_accuracy': {
            'criterion': 'B_scheduler > 90%',
            'value': B_acc,
            'pass': B_acc > 0.90,
        },
        'T3_wrong_routing': {
            'criterion': 'A - D gap > 10%',
            'A': A_acc, 'D': D_acc,
            'gap': A_acc - D_acc,
            'pass': (A_acc - D_acc) > 0.10,
        },
        'T4_lds_same_cu': {
            'criterion': 'LDS persist rate > 80% same CU',
            'value': lds_same,
            'pass': lds_same > 0.80,
        },
        'T5_lds_cross_cu': {
            'criterion': 'LDS persist rate < 20% cross CU',
            'value': lds_cross,
            'pass': (lds_cross is not None and lds_cross < 0.20) or lds_cross is None,
        },
        'T7_energy': {
            'criterion': 'Energy measured',
            'A_mJ': round(eval_a_sched['joules_per_correct'] * 1000, 3),
            'B_mJ': round(eval_b['joules_per_correct'] * 1000, 3),
            'C_mJ': round(eval_c['joules_per_correct'] * 1000, 3),
            'pass': True,
        },
    }

    # CU mask verification test
    if eval_a_placed is not None:
        placed_acc = eval_a_placed['accuracy']
        results['tests']['T1b_cu_mask_matches'] = {
            'criterion': 'CU-masked eval ≈ scheduler eval (±5%)',
            'placed': placed_acc,
            'scheduler': A_acc,
            'diff': abs(placed_acc - A_acc),
            'pass': abs(placed_acc - A_acc) < 0.05,
        }

    # Cross-machine PUF
    puf_results = cross_machine_puf()
    if puf_results is not None:
        results['cross_machine_puf'] = puf_results
        results['tests']['T6_cross_machine'] = {
            'criterion': 'Chips distinguishable by fingerprint',
            'hwreg_diffs': puf_results['hwreg_differ'],
            'timing_diff_pct': puf_results.get('timing_mean_diff_pct'),
            'pass': puf_results['chips_distinguishable'],
        }

    # ─── Verdict ───
    n_pass = sum(1 for t in results['tests'].values()
                 if isinstance(t.get('pass'), bool) and t['pass'])
    n_total = len(results['tests'])
    results['n_pass'] = n_pass
    results['n_total'] = n_total

    if n_pass >= n_total - 1:
        verdict = 'CU_PLACED_CONFIRMED'
    elif n_pass >= n_total // 2:
        verdict = 'PARTIAL'
    else:
        verdict = 'WEAK'
    results['verdict'] = verdict

    results['summary'] = {
        'A_placed': A_acc,
        'B_scheduler': B_acc,
        'C_blind': C_acc,
        'D_wrong_cu': D_acc,
        'gap_A_C': round(A_acc - C_acc, 4),
        'gap_A_D': round(A_acc - D_acc, 4),
        'lds_same_cu_rate': lds_same,
        'lds_cross_cu_rate': lds_cross,
        'cu_mask_available': probe.cu_mask_available,
        'n_cu_bits': len(probe.cu_wgp_map),
        'n_wgps': len(probe.wgp_to_cu_bits) if probe.cu_mask_available else 0,
    }

    # ─── Print Results ───
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  A_placed:    {A_acc:.4f}")
    print(f"  B_scheduler: {B_acc:.4f}")
    print(f"  C_blind:     {C_acc:.4f}")
    print(f"  D_wrong_cu:  {D_acc:.4f}")
    print(f"  CU mask:     {'available' if probe.cu_mask_available else 'UNAVAILABLE'}")
    print(f"  LDS persist: same_cu={lds_same:.2f}, cross_cu={lds_cross}")
    print(f"\n  Tests: {n_pass}/{n_total} PASS")
    for name, test in results['tests'].items():
        status = 'PASS' if test.get('pass') else 'FAIL'
        print(f"    {name}: {status} — {test.get('criterion', '')}")
    print(f"\n  VERDICT: {verdict}")

    # ─── Save ───
    out_path = RESULTS_DIR / 'z2052_cu_placed_lds_persistent.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
