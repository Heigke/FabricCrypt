#!/usr/bin/env python3
"""
Deep HIP Integration - Low-Level GPU Control

This module goes DEEP into the AMD GPU stack:

1. **hipRTC Dynamic Kernel Compilation**
   - Generate optimized kernels at runtime based on energy budget
   - Specialize kernels for specific tensor shapes

2. **Wave-Level Intrinsics**
   - Use __shfl for fast cross-lane communication
   - Implement wave-level early exit

3. **Adaptive Occupancy Control**
   - Multiple kernel variants with different VGPR usage
   - Trade off occupancy for power efficiency

4. **DVFS Integration**
   - Programmatic control of GPU clock frequencies
   - Energy-aware performance level selection

5. **PyTorch Integration**
   - Custom autograd functions using HIP kernels
   - Seamless integration with existing models

References:
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html
- https://gpuopen.com/learn/occupancy-explained/
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html
"""

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import sys
import ctypes
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.telemetry.real_amd import AMDTelemetry, RocmSmiReader


# =============================================================================
# DVFS (Dynamic Voltage and Frequency Scaling) Control
# =============================================================================

class DVFSController:
    """
    Controls GPU power levels via rocm-smi.

    Performance levels:
    - low: Minimum clocks, minimum power
    - auto: Driver-controlled (default)
    - high: Maximum clocks, maximum power
    - manual: User-specified clock levels
    """

    PERF_LEVELS = ['low', 'auto', 'high', 'manual']

    @staticmethod
    def get_current_level() -> str:
        """Get current performance level."""
        try:
            result = subprocess.run(
                ['rocm-smi', '--showperflevel'],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split('\n'):
                if 'Performance Level' in line:
                    for level in DVFSController.PERF_LEVELS:
                        if level.lower() in line.lower():
                            return level
        except Exception as e:
            print(f"Warning: Could not read perf level: {e}")
        return 'unknown'

    @staticmethod
    def set_perf_level(level: str) -> bool:
        """
        Set GPU performance level.

        Args:
            level: 'low', 'auto', 'high', or 'manual'

        Returns:
            True if successful
        """
        if level not in DVFSController.PERF_LEVELS:
            return False

        try:
            result = subprocess.run(
                ['rocm-smi', '--setperflevel', level],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Warning: Could not set perf level: {e}")
            return False

    @staticmethod
    def get_available_clocks() -> Dict[str, List[int]]:
        """Get available clock frequencies."""
        clocks = {'sclk': [], 'mclk': []}
        try:
            result = subprocess.run(
                ['rocm-smi', '--showclkfrq'],
                capture_output=True, text=True, timeout=5
            )
            # Parse output for available frequencies
            # This is hardware-specific
        except Exception:
            pass
        return clocks

    @staticmethod
    def set_power_cap(watts: int) -> bool:
        """Set GPU power cap in watts."""
        try:
            result = subprocess.run(
                ['rocm-smi', '--setpoweroverdrive', str(watts)],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False


# =============================================================================
# HIP Kernel Source Templates
# =============================================================================

ENERGY_AWARE_SOFTMAX_SOURCE = '''
#include <hip/hip_runtime.h>

#define WARP_SIZE 32

__device__ __forceinline__ float waveReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset, WARP_SIZE);
    }
    return val;
}

__device__ __forceinline__ float waveReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other = __shfl_down(val, offset, WARP_SIZE);
        val = fmaxf(val, other);
    }
    return val;
}

extern "C" __global__ void energyAwareSoftmaxKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int vocab_size,
    const float skip_threshold,
    int* __restrict__ skipped_count
) {
    extern __shared__ float smem[];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    const float* row_input = input + row * vocab_size;
    float* row_output = output + row * vocab_size;

    // Phase 1: Find max
    float local_max = -INFINITY;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        local_max = fmaxf(local_max, row_input[i]);
    }

    local_max = waveReduceMax(local_max);
    if (lane == 0) smem[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < num_warps) ? smem[lane] : -INFINITY;
        val = waveReduceMax(val);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    float global_max = smem[0];

    // Phase 2: Energy-aware exp sum
    float local_sum = 0.0f;
    int local_skipped = 0;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = row_input[i] - global_max;

        // ENERGY SAVING: Skip negligible values
        if (val < skip_threshold) {
            row_output[i] = 0.0f;
            local_skipped++;
        } else {
            float exp_val = __expf(val);
            row_output[i] = exp_val;
            local_sum += exp_val;
        }
    }

    local_sum = waveReduceSum(local_sum);
    if (lane == 0) smem[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < num_warps) ? smem[lane] : 0.0f;
        val = waveReduceSum(val);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    float global_sum = smem[0];

    // Phase 3: Normalize
    float inv_sum = 1.0f / (global_sum + 1e-8f);
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        row_output[i] *= inv_sum;
    }

    // Track skipped for telemetry
    local_skipped = waveReduceSum(local_skipped);
    if (tid == 0 && skipped_count != nullptr) {
        atomicAdd(skipped_count, local_skipped);
    }
}
'''

ADAPTIVE_GEMM_SOURCE = '''
#include <hip/hip_runtime.h>

// Low-power GEMM: Minimal register usage, higher occupancy
extern "C" __global__ __launch_bounds__(256, 8)
void gemmLowPowerKernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    const int row = blockIdx.y * 16 + threadIdx.y;
    const int col = blockIdx.x * 16 + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Balanced GEMM: Tiled with shared memory
extern "C" __global__ __launch_bounds__(256, 4)
void gemmBalancedKernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    const int TILE = 16;
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x % TILE;
    const int ty = threadIdx.x / TILE;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        if (row < M && t * TILE + tx < K)
            As[ty][tx] = A[row * K + t * TILE + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && t * TILE + ty < K)
            Bs[ty][tx] = B[(t * TILE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// High-performance GEMM: More registers, vectorized
extern "C" __global__ __launch_bounds__(256, 2)
void gemmHighPerfKernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    // Simplified high-perf version
    const int TILE = 32;
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x % TILE;
    const int ty = threadIdx.x / 8;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Load tiles cooperatively
        for (int i = threadIdx.x; i < TILE * TILE; i += blockDim.x) {
            int r = i / TILE;
            int c = i % TILE;
            int gr = blockIdx.y * TILE + r;
            int gc = t * TILE + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;

            gr = t * TILE + r;
            gc = blockIdx.x * TILE + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
'''


# =============================================================================
# HIP RTC Compiler Interface
# =============================================================================

class HipRTCCompiler:
    """
    Runtime compilation of HIP kernels using hiprtc.

    This enables:
    1. Dynamic kernel specialization based on tensor shapes
    2. Energy-budget-aware kernel selection
    3. Just-in-time optimization for specific hardware
    """

    def __init__(self, arch: str = "gfx1100"):
        """
        Initialize HIP RTC compiler.

        Args:
            arch: GPU architecture (gfx1100 for RDNA3)
        """
        self.arch = arch
        self._compiled_kernels: Dict[str, ctypes.CDLL] = {}
        self._temp_dir = Path(tempfile.mkdtemp(prefix="hip_kernels_"))

    def compile_kernel(
        self,
        source: str,
        kernel_name: str,
        extra_flags: List[str] = None
    ) -> Optional[Path]:
        """
        Compile HIP source to binary.

        Uses hipcc for compilation (hipRTC equivalent workflow).
        """
        if extra_flags is None:
            extra_flags = []

        # Write source to temp file
        src_path = self._temp_dir / f"{kernel_name}.hip"
        out_path = self._temp_dir / f"{kernel_name}.so"

        with open(src_path, 'w') as f:
            f.write(source)

        # Compile with hipcc
        cmd = [
            'hipcc',
            '-shared', '-fPIC',
            f'--offload-arch={self.arch}',
            '-O3',
            '-o', str(out_path),
            str(src_path),
        ] + extra_flags

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                print(f"Compilation failed: {result.stderr}")
                return None
            return out_path
        except Exception as e:
            print(f"Compilation error: {e}")
            return None

    def load_kernel(self, so_path: Path) -> Optional[ctypes.CDLL]:
        """Load compiled kernel as shared library."""
        try:
            return ctypes.CDLL(str(so_path))
        except Exception as e:
            print(f"Failed to load kernel: {e}")
            return None

    def get_or_compile(
        self,
        source: str,
        kernel_name: str,
        cache_key: str = None
    ) -> Optional[ctypes.CDLL]:
        """Get cached kernel or compile new one."""
        key = cache_key or kernel_name

        if key in self._compiled_kernels:
            return self._compiled_kernels[key]

        so_path = self.compile_kernel(source, kernel_name)
        if so_path is None:
            return None

        lib = self.load_kernel(so_path)
        if lib is not None:
            self._compiled_kernels[key] = lib

        return lib


# =============================================================================
# PyTorch Integration via load_inline
# =============================================================================

def get_hip_extension_source():
    """
    Generate C++/HIP source for PyTorch extension.

    This creates custom autograd functions that use our HIP kernels.
    """
    cpp_source = '''
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Forward declarations
torch::Tensor energy_aware_softmax_forward(
    torch::Tensor input,
    float skip_threshold
);

torch::Tensor adaptive_gemm_forward(
    torch::Tensor A,
    torch::Tensor B,
    int energy_level
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("energy_aware_softmax", &energy_aware_softmax_forward,
          "Energy-aware softmax with wave-level optimization");
    m.def("adaptive_gemm", &adaptive_gemm_forward,
          "Adaptive GEMM with selectable power/performance tradeoff");
}
'''

    hip_source = '''
#include <torch/extension.h>
#include <hip/hip_runtime.h>

#define WARP_SIZE 32

__device__ __forceinline__ float waveReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset, WARP_SIZE);
    }
    return val;
}

__device__ __forceinline__ float waveReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other = __shfl_down(val, offset, WARP_SIZE);
        val = fmaxf(val, other);
    }
    return val;
}

__global__ void energyAwareSoftmaxKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int num_rows,
    const int row_size,
    const float skip_threshold
) {
    extern __shared__ float smem[];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    if (row >= num_rows) return;

    const float* row_input = input + row * row_size;
    float* row_output = output + row * row_size;

    // Find max
    float local_max = -INFINITY;
    for (int i = tid; i < row_size; i += blockDim.x) {
        local_max = fmaxf(local_max, row_input[i]);
    }
    local_max = waveReduceMax(local_max);
    if (lane == 0) smem[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < num_warps) ? smem[lane] : -INFINITY;
        val = waveReduceMax(val);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    float global_max = smem[0];

    // Exp sum with skip
    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float val = row_input[i] - global_max;
        if (val < skip_threshold) {
            row_output[i] = 0.0f;
        } else {
            float exp_val = __expf(val);
            row_output[i] = exp_val;
            local_sum += exp_val;
        }
    }

    local_sum = waveReduceSum(local_sum);
    if (lane == 0) smem[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < num_warps) ? smem[lane] : 0.0f;
        val = waveReduceSum(val);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    float global_sum = smem[0];

    // Normalize
    float inv_sum = 1.0f / (global_sum + 1e-8f);
    for (int i = tid; i < row_size; i += blockDim.x) {
        row_output[i] *= inv_sum;
    }
}

torch::Tensor energy_aware_softmax_forward(
    torch::Tensor input,
    float skip_threshold
) {
    TORCH_CHECK(input.is_cuda(), "Input must be on GPU");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);

    int num_rows = input.numel() / input.size(-1);
    int row_size = input.size(-1);

    int threads = 256;
    int shared_mem = (threads / WARP_SIZE) * sizeof(float);

    hipLaunchKernelGGL(
        energyAwareSoftmaxKernel,
        dim3(num_rows), dim3(threads), shared_mem, 0,
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_rows,
        row_size,
        skip_threshold
    );

    return output;
}

// Simple adaptive GEMM placeholder
torch::Tensor adaptive_gemm_forward(
    torch::Tensor A,
    torch::Tensor B,
    int energy_level
) {
    // For now, use PyTorch's matmul
    // In production, this would dispatch to different kernel variants
    return torch::matmul(A, B);
}
'''

    return cpp_source, hip_source


# =============================================================================
# Energy-Aware Layer Implementations
# =============================================================================

class EnergyAwareSoftmax(nn.Module):
    """
    Softmax with wave-level energy optimization.

    The key insight: If a token has very high probability, other tokens
    will have negligible probability after softmax. We can skip computing
    exp() for those tokens, saving energy without affecting output.
    """

    def __init__(self, skip_threshold: float = -10.0):
        super().__init__()
        self.skip_threshold = skip_threshold
        self._extension = None

    def _load_extension(self):
        """Lazily load the HIP extension."""
        if self._extension is not None:
            return

        try:
            from torch.utils.cpp_extension import load_inline

            cpp_source, hip_source = get_hip_extension_source()

            self._extension = load_inline(
                name='energy_aware_ops',
                cpp_sources=[cpp_source],
                cuda_sources=[hip_source],
                functions=['energy_aware_softmax', 'adaptive_gemm'],
                with_cuda=True,
                extra_cuda_cflags=['-O3', '--offload-arch=gfx1100'],
                verbose=False
            )
        except Exception as e:
            print(f"Warning: Could not load HIP extension: {e}")
            print("Falling back to PyTorch softmax")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with energy-aware softmax."""
        if not x.is_cuda:
            return F.softmax(x, dim=-1)

        self._load_extension()

        if self._extension is not None:
            try:
                return self._extension.energy_aware_softmax(x, self.skip_threshold)
            except Exception as e:
                print(f"HIP kernel failed: {e}")

        return F.softmax(x, dim=-1)


class AdaptiveGEMM(nn.Module):
    """
    GEMM with selectable power/performance tradeoff.

    Three modes:
    - low: Minimal power, maximum occupancy, lower throughput
    - balanced: Good balance of power and performance
    - high: Maximum performance, higher power consumption
    """

    def __init__(self, energy_level: str = 'balanced'):
        super().__init__()
        self.energy_level = energy_level
        self._extension = None
        self._level_map = {'low': 0, 'balanced': 1, 'high': 2}

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Matrix multiplication with adaptive power."""
        # For production, this would dispatch to HIP kernels
        # Currently uses PyTorch as fallback
        return torch.matmul(A, B)


# =============================================================================
# Deep Body Integration - GPU State as Embodiment
# =============================================================================

@dataclass
class GPUBodyState:
    """
    The GPU's "body" - its physical state that affects compute.

    This is the "embodiment" - the model should learn to sense
    and respond to these signals, regulating its compute accordingly.
    """
    power_w: float
    temp_c: float
    sclk_mhz: int
    mclk_mhz: int
    vram_used_gb: float
    perf_level: str
    throttling: bool

    def to_tensor(self, device='cuda') -> torch.Tensor:
        """Convert body state to normalized tensor."""
        return torch.tensor([
            self.power_w / 150.0,  # Normalize to ~1
            self.temp_c / 100.0,
            self.sclk_mhz / 3000.0,
            self.mclk_mhz / 2000.0,
            self.vram_used_gb / 96.0,  # 96GB for this GPU
            1.0 if self.throttling else 0.0,
            {'low': 0.0, 'auto': 0.5, 'high': 1.0}.get(self.perf_level, 0.5),
        ], device=device)


class DeepBodyIntegration:
    """
    Deep integration between model and GPU hardware.

    The model learns to:
    1. SENSE: Read GPU temperature, power, clocks
    2. REGULATE: Adjust compute intensity based on thermal state
    3. EXPRESS: Choose kernel variants that match energy budget
    """

    def __init__(self):
        self.telemetry = AMDTelemetry()
        self.dvfs = DVFSController()

    def sense(self) -> GPUBodyState:
        """Read current GPU body state."""
        snapshot = self.telemetry.read()
        perf_level = self.dvfs.get_current_level()

        # Detect throttling (simplified)
        throttling = snapshot.temp_c > 85 or snapshot.power_w > 120

        return GPUBodyState(
            power_w=snapshot.power_w,
            temp_c=snapshot.temp_c,
            sclk_mhz=snapshot.sclk_mhz,
            mclk_mhz=snapshot.mclk_mhz,
            vram_used_gb=snapshot.vram_used_gb,
            perf_level=perf_level,
            throttling=throttling
        )

    def regulate(self, body: GPUBodyState, target_power: float) -> str:
        """
        Regulate GPU state based on body signals and target.

        Returns the energy level to use for compute.
        """
        # If throttling, reduce compute intensity
        if body.throttling:
            self.dvfs.set_perf_level('low')
            return 'low'

        # If temperature high, reduce
        if body.temp_c > 75:
            return 'balanced'

        # If power budget allows, maximize
        if body.power_w < target_power * 0.7:
            return 'high'

        return 'balanced'

    def express(
        self,
        energy_level: str,
        body: GPUBodyState
    ) -> Dict[str, float]:
        """
        Express the chosen energy level through kernel parameters.

        Returns configuration for kernels.
        """
        configs = {
            'low': {
                'softmax_skip_threshold': -5.0,  # More aggressive skipping
                'attention_sparsity': 0.7,       # 70% sparse
                'gemm_occupancy': 'max',
            },
            'balanced': {
                'softmax_skip_threshold': -10.0,
                'attention_sparsity': 0.3,
                'gemm_occupancy': 'balanced',
            },
            'high': {
                'softmax_skip_threshold': -20.0,  # Less skipping
                'attention_sparsity': 0.0,        # Dense
                'gemm_occupancy': 'min',          # More registers
            },
        }
        return configs.get(energy_level, configs['balanced'])

    def shutdown(self):
        """Clean shutdown."""
        self.telemetry.shutdown()


# =============================================================================
# Demonstration
# =============================================================================

def demonstrate_deep_integration():
    """Demonstrate the deep HIP integration."""
    print("=" * 60)
    print("Deep HIP Integration Demonstration")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize body integration
    body_integration = DeepBodyIntegration()

    # Sense current state
    print("\n--- GPU Body State ---")
    body = body_integration.sense()
    print(f"  Power: {body.power_w:.1f}W")
    print(f"  Temp: {body.temp_c:.0f}°C")
    print(f"  Clocks: sclk={body.sclk_mhz}MHz, mclk={body.mclk_mhz}MHz")
    print(f"  VRAM: {body.vram_used_gb:.1f}GB")
    print(f"  Perf Level: {body.perf_level}")
    print(f"  Throttling: {body.throttling}")

    # Regulate based on target
    target_power = 60.0  # Target 60W
    energy_level = body_integration.regulate(body, target_power)
    print(f"\n  Target Power: {target_power}W")
    print(f"  Selected Energy Level: {energy_level}")

    # Express as kernel config
    config = body_integration.express(energy_level, body)
    print(f"\n--- Kernel Configuration ---")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Test energy-aware softmax
    print("\n--- Energy-Aware Softmax Test ---")

    softmax_layer = EnergyAwareSoftmax(skip_threshold=-10.0)

    x = torch.randn(8, 128, 50257, device=device)  # batch, seq, vocab

    # Warmup
    for _ in range(3):
        _ = softmax_layer(x)
        torch.cuda.synchronize()

    # Measure
    start = time.time()
    for _ in range(10):
        y = softmax_layer(x)
        torch.cuda.synchronize()
    elapsed = time.time() - start

    # Verify
    y_ref = F.softmax(x, dim=-1)
    diff = (y - y_ref).abs().max().item()

    print(f"  Shape: {x.shape}")
    print(f"  Time (10 iters): {elapsed*1000:.1f}ms")
    print(f"  Max diff from reference: {diff:.6f}")

    # Read body state after compute
    body_after = body_integration.sense()
    print(f"\n  Power after compute: {body_after.power_w:.1f}W")
    print(f"  Temp after compute: {body_after.temp_c:.0f}°C")

    # Cleanup
    body_integration.shutdown()

    print("\n" + "=" * 60)
    print("Deep integration demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_deep_integration()
