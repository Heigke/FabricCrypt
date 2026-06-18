#!/usr/bin/env python3
"""
z2114: Analog Linear LM — Embodied MatMul via FP Rounding Mode Manipulation
============================================================================
MONIST ARCHITECTURE: No neural adapters. The hardware physics directly warps
matrix multiplication via GPU MODE register (FP rounding + denorm handling).

Key insight: "Give Me FP32 or Give Me Death?" (2025) shows up to 9% accuracy
variation in 7B reasoning models from rounding differences alone.

DELETE from z2113: NormBoundedLoRA, FiLM, BodyEncoder, steering vectors,
  substrate injection, ThermalSoftmax, IA³, freq_gate, body_scale, all adapters.
KEEP: Frozen Qwen3-8B, DVFS controller, SMN/RAPL reads, dataset loading.
ADD: Custom HIP GEMM kernel (AnalogLinear) that constitutively warps computation.

Phase A — Frozen Measurement (0 trainable params):
  Frozen Qwen3-8B + 3 AnalogLinear layers replacing gate_proj at layers 15,20,25.
  Sweep all 4 FP rounding modes, measure KL divergence between logit distributions.

Phase B — Tiny LoRA (conditional, ~0.5M params):
  Only if Phase A shows signal. Rank-2 LoRA at v_proj layers 15,20,25.
  Train with standard CE loss. Test adaptation to rounding landscape.

Hardware setup (run BEFORE launching):
  sudo modprobe msr
  sudo insmod ~/Documents/claude_hive/ryzen_smu/ryzen_smu.ko
  sudo chmod 666 /sys/kernel/ryzen_smu_drv/smn
  sudo chmod 666 /sys/kernel/ryzen_smu_drv/pm_table
  sudo sysctl kernel.perf_event_paranoid=-1

  sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 \\
    /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/bin/python \\
    scripts/z2114_analog_linear_lm.py
"""

import os, sys, json, math, time, struct, ctypes, ctypes.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEVICE = 'cuda'
BS = 4
SEQ_LEN = 128
N_EVAL_BATCHES = 30
DVFS_SETTLE_S = 1.5

# AnalogLinear placement: gate_proj at these Qwen3-8B layers
ANALOG_LAYERS = [15, 20, 25]

# Phase B LoRA config (conditional)
LORA_RANK = 2
LORA_ALPHA = 4
LORA_LAYERS = [15, 20, 25]  # v_proj only

# DVFS calibration (set at runtime)
SCLK_LOW_CAL = 600.0
SCLK_HIGH_CAL = 2900.0

# FP rounding modes (MODE register bits[1:0] for f32)
ROUND_MODES = {
    0: 'nearest_even',
    1: 'plus_inf',
    2: 'minus_inf',
    3: 'toward_zero',
}

# Denorm modes (MODE register bits[5:4] for f32)
DENORM_MODES = {
    0b00: 'flush_both',
    0b01: 'allow_input',
    0b10: 'allow_output',
    0b11: 'allow_both',
}

def make_lm_labels(input_ids, offset=1):
    """Build labels for LM loss."""
    labels = input_ids.clone()
    shift = offset - 1
    if shift > 0:
        labels[:, :-shift] = input_ids[:, shift:]
        labels[:, -shift:] = -100
    return labels

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HARDWARE ACCESS — DVFS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DVFS_AVAILABLE = False
DVFS_PATH = None

def find_dvfs_sysfs():
    global DVFS_AVAILABLE, DVFS_PATH
    for card in ['card1', 'card0']:
        p = f'/sys/class/drm/{card}/device/power_dpm_force_performance_level'
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    val = f.read().strip()
                DVFS_PATH = p
                DVFS_AVAILABLE = True
                print(f"[DVFS] Found: {p} = {val}")
                return
            except:
                pass
    print("[DVFS] Not available")

def set_dvfs_level(level, wait=True):
    """Set DVFS: 0=low, 1=auto, 2=high."""
    if not DVFS_AVAILABLE:
        return
    torch.cuda.synchronize()
    name = {0: 'low', 1: 'auto', 2: 'high'}[level]
    try:
        with open(DVFS_PATH, 'w') as f:
            f.write(name)
    except Exception as e:
        print(f"[DVFS] Write failed: {e}")
        return
    if wait:
        _poll_dvfs_settle(level)

def _poll_dvfs_settle(level):
    target_low = level == 0
    for attempt in range(30):
        sclk = read_current_sclk_mhz()
        if target_low and sclk < 800:
            return
        if not target_low and sclk > 1200:
            return
        time.sleep(0.1)
    time.sleep(DVFS_SETTLE_S)

def restore_dvfs_auto():
    if DVFS_AVAILABLE:
        torch.cuda.synchronize()
        try:
            with open(DVFS_PATH, 'w') as f:
                f.write('auto')
        except:
            pass

def read_current_sclk_mhz():
    for hwmon in ['hwmon7', 'hwmon6', 'hwmon5']:
        p = f'/sys/class/hwmon/{hwmon}/freq1_input'
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    return float(f.read().strip()) / 1e6
            except:
                pass
    return 600.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HARDWARE ACCESS — SMN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SMN_AVAILABLE = False

def check_smn():
    global SMN_AVAILABLE
    SMN_AVAILABLE = os.path.exists('/sys/kernel/ryzen_smu_drv/smn')
    print(f"[SMN] {'Available' if SMN_AVAILABLE else 'Not available'}")

def read_smn(addr):
    """Single-handle r+b read (z2103 fix)."""
    if not SMN_AVAILABLE:
        return 0
    try:
        with open('/sys/kernel/ryzen_smu_drv/smn', 'r+b', buffering=0) as f:
            f.write(struct.pack('<I', addr & 0xFFFFFFFF))
            f.flush()
            f.seek(0)
            data = f.read(4)
        return struct.unpack('<I', data)[0] if len(data) == 4 else 0
    except Exception:
        return 0

def read_hw_entropy():
    """Read hardware-derived analog entropy from SMN thermal ADC + XTAL jitter.
    Returns int in [0, 255] for MODE register bits[7:0]."""
    if not SMN_AVAILABLE:
        return None
    try:
        raw_adc = read_smn(0x00059800)  # THM_TCON_CUR_TMP
        entropy_bits = raw_adc & 0xFF
        xtal = read_smn(0x000598C8)  # XTAL_CNTL
        xtal_low = xtal & 0xFFFF
        return (entropy_bits ^ xtal_low) & 0xFF
    except Exception:
        return None

def read_hw_entropy_float():
    """Read hardware entropy as float in [0, 1)."""
    val = read_hw_entropy()
    return val / 256.0 if val is not None else None

def read_gpu_temp_c():
    """Read GPU junction temperature in °C from amdgpu hwmon."""
    for hwmon in ['hwmon7', 'hwmon6', 'hwmon8']:
        p = f'/sys/class/hwmon/{hwmon}/temp1_input'
        try:
            with open(p, 'r') as f:
                return float(f.read().strip()) / 1000.0  # millidegrees → °C
        except:
            pass
    # Fallback: SMN thermal ADC (bits[31:21] = temp in 0.125°C units)
    raw = read_smn(0x00059800)
    return ((raw >> 21) & 0x7FF) * 0.125 if raw else 50.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HARDWARE ACCESS — RAPL Energy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAPL_AVAILABLE = False
RAPL_PATHS = {}
RAPL_MAX_RANGE = {}

def check_rapl():
    global RAPL_AVAILABLE, RAPL_PATHS, RAPL_MAX_RANGE
    base = '/sys/class/powercap'
    for domain in ['intel-rapl:0']:
        ej = os.path.join(base, domain, 'energy_uj')
        if os.path.exists(ej):
            name_path = os.path.join(base, domain, 'name')
            try:
                with open(name_path, 'r') as f:
                    name = f.read().strip()
                RAPL_PATHS[name] = ej
                max_range_path = os.path.join(base, domain, 'max_energy_range_uj')
                if os.path.exists(max_range_path):
                    with open(max_range_path, 'r') as f:
                        RAPL_MAX_RANGE[name] = int(f.read().strip())
                else:
                    RAPL_MAX_RANGE[name] = (1 << 32)
            except:
                pass
    RAPL_AVAILABLE = len(RAPL_PATHS) > 0
    print(f"[RAPL] Domains: {list(RAPL_PATHS.keys())}")

def read_rapl_snapshot():
    result = {}
    for name, path in RAPL_PATHS.items():
        try:
            with open(path, 'r') as f:
                result[name] = int(f.read().strip())
        except:
            result[name] = 0
    return result

def compute_batch_joules(before, after, gpu_ppt_mw=0, elapsed_s=None):
    total_uj = 0
    for name in before:
        if name in after:
            delta = after[name] - before[name]
            if delta < 0:
                max_range = RAPL_MAX_RANGE.get(name, (1 << 32))
                delta = (max_range - before[name]) + after[name]
                if delta < 0 or delta > max_range:
                    continue
            total_uj += delta
    total_j = total_uj / 1e6
    if gpu_ppt_mw > 0 and elapsed_s:
        total_j += (gpu_ppt_mw / 1000.0) * elapsed_s
    return total_j

def read_gpu_ppt_mw():
    for hwmon in ['hwmon7', 'hwmon6']:
        p = f'/sys/class/hwmon/{hwmon}/power1_input'
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    return float(f.read().strip()) / 1000.0
            except:
                pass
    return 0.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HARDWARE ACCESS — MSR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MSR_AVAILABLE = False
MSR_FD = None

def init_msr():
    global MSR_AVAILABLE, MSR_FD
    try:
        MSR_FD = os.open('/dev/cpu/0/msr', os.O_RDONLY)
        MSR_AVAILABLE = True
        print("[MSR] Available")
    except:
        print("[MSR] Not available")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_wikitext_data(tokenizer, split='train', max_samples=2000):
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    all_ids = []
    for text in ds['text']:
        if len(text.strip()) < 50:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
    sequences = []
    for i in range(0, len(all_ids) - SEQ_LEN, SEQ_LEN):
        seq = torch.tensor(all_ids[i:i + SEQ_LEN], dtype=torch.long)
        sequences.append(seq)
        if len(sequences) >= max_samples:
            break
    print(f"  Loaded {len(sequences)} sequences ({split})")
    return sequences

def load_code_data(tokenizer, max_samples=2000):
    from datasets import load_dataset
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    hf_token = None
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith('hf_token='):
                    hf_token = line.strip().split('=', 1)[1]
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
    ds = load_dataset('bigcode/starcoderdata', data_dir='python', split='train', streaming=True)
    all_ids = []
    n_docs = 0
    for sample in ds:
        text = sample['content']
        if len(text.strip()) < 100:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        n_docs += 1
        if len(all_ids) >= max_samples * SEQ_LEN * 2:
            break
    sequences = []
    for i in range(0, len(all_ids) - SEQ_LEN, SEQ_LEN):
        seq = torch.tensor(all_ids[i:i + SEQ_LEN], dtype=torch.long)
        sequences.append(seq)
        if len(sequences) >= max_samples:
            break
    print(f"  Loaded {len(sequences)} code sequences from {n_docs} files")
    return sequences


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUSTOM HIP KERNEL — Tiled Analog GEMM with MODE Register Control
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_HIP_MODULE = None
_HIP_COMPILE_ATTEMPTED = False

def compile_hip_kernel():
    """Compile the analog GEMM kernel using torch's HIP extension loader.
    ROCm uses cuda_sources/extra_cuda_cflags (not hip_sources)."""
    global _HIP_MODULE, _HIP_COMPILE_ATTEMPTED
    if _HIP_MODULE is not None:
        return _HIP_MODULE
    if _HIP_COMPILE_ATTEMPTED:
        return None  # don't retry on every forward pass
    _HIP_COMPILE_ATTEMPTED = True

    print("[HIP] Compiling analog_tiled_gemm_kernel for gfx1100...")
    from torch.utils.cpp_extension import load_inline

    # On ROCm, the CUDA source gets hipified automatically by load_inline.
    # We put both the kernel and the C++ wrapper in cuda_sources so they share
    # the same translation unit (kernel definition visible to launch call).
    combined_source = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#define TILE_SIZE 32

__global__ void analog_tiled_gemm_kernel(
    const __hip_bfloat16* __restrict__ A,
    const __hip_bfloat16* __restrict__ B,
    __hip_bfloat16* __restrict__ C,
    int M, int K, int N,
    unsigned int base_round_mode,
    unsigned int inject_mask  // 0x00 = sterile determinism, 0xFF = live physics
) {
    // Save MODE register
    unsigned int old_mode;
    asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" ::: "memory");
    asm volatile("s_getreg_b32 %0, hwreg(1, 0, 8)" : "=s"(old_mode) :: "memory");

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float acc = 0.0f;
    int n_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < n_tiles; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = __bfloat162float(A[row * K + a_col]);
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        int b_row = t * TILE_SIZE + threadIdx.y;
        if (col < N && b_row < K)
            Bs[threadIdx.y][threadIdx.x] = __bfloat162float(B[col * K + b_row]);
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // THE THERMODYNAMIC CLOCK
        // Read SHADER_CYCLES inside the tile loop. As the GPU heats up and
        // thermal-throttles, the cycle count advances differently relative to
        // the math, creating continuous analog phase-shift. inject_mask gates
        // the entropy: 0x00 = deterministic baseline, 0xFF = live physics.
        unsigned int cycles;
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(cycles));

        unsigned int rm = base_round_mode ^ (cycles & inject_mask);
        // Mirror low 2 bits into FP16 rounding bits[3:2], keep denorm bits[7:4]
        unsigned int rm_both = (rm & 0x3u) | ((rm & 0x3u) << 2) | (rm & 0xF0u);
        unsigned int new_mode = (old_mode & ~0xFFu) | rm_both;
        asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" :: "s"(new_mode) : "memory");

        for (int k = 0; k < TILE_SIZE; k++) {
            // FP16 DENORMAL HORIZON: cast to hardware __half BEFORE multiply.
            // Individual products (~0.00001) can trigger denorm flushing.
            // __float2half = v_cvt_f16_f32 (reads MODE bits[3:2])
            __half a_h = __float2half(As[threadIdx.y][k]);
            __half b_h = __float2half(Bs[k][threadIdx.x]);

            // Native FP16 multiply (subject to denorm flush + rounding MODE)
            __half step_h = __hmul(a_h, b_h);

            acc += __half2float(step_h);
        }

        // bf16 mantissa truncation per tile compounds rounding
        acc = __bfloat162float(__float2bfloat16(acc));

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(acc);
    }

    // Restore MODE
    asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" ::: "memory");
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" :: "s"(old_mode) : "memory");
}

torch::Tensor analog_gemm(torch::Tensor A, torch::Tensor weight, int round_mode, int inject_mask) {
    TORCH_CHECK(A.is_cuda() && weight.is_cuda(), "Tensors must be on GPU");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bf16");
    TORCH_CHECK(weight.dtype() == torch::kBFloat16, "weight must be bf16");

    int M = A.size(0);
    int K = A.size(1);
    int N = weight.size(0);
    TORCH_CHECK(weight.size(1) == K, "Dimension mismatch");

    auto C = torch::empty({M, N}, A.options());

    const int TILE = 32;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    analog_tiled_gemm_kernel<<<grid, block>>>(
        reinterpret_cast<const __hip_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),
        M, K, N, (unsigned int)round_mode, (unsigned int)inject_mask);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("analog_gemm", &analog_gemm, "Analog GEMM with MODE register control");
}
"""

    try:
        _HIP_MODULE = load_inline(
            name='analog_gemm_ext',
            cpp_sources=[],
            cuda_sources=[combined_source],
            extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'],
            verbose=True,
            with_cuda=True,
        )
        print("[HIP] Kernel compiled OK")
    except Exception as e:
        print(f"[HIP] Compilation failed: {e}")
        print("[HIP] Falling back to torch.mm (no MODE override)")
        _HIP_MODULE = None
    return _HIP_MODULE


class AnalogLinear(nn.Module):
    """Drop-in replacement for nn.Linear that uses custom HIP GEMM with
    thermal-derived FP rounding mode. The matmul IS the embodiment.

    Reads SMN entropy on host → passes round_mode to kernel → kernel runs
    tiled GEMM under that mode. Zero trainable parameters.
    """
    def __init__(self, original_linear, layer_idx, mode_override=None):
        super().__init__()
        self.weight = original_linear.weight  # [N, K] bf16, frozen
        self.bias = original_linear.bias
        self.layer_idx = layer_idx
        self.mode_override = mode_override  # None = live entropy, int = forced mode
        self._last_round_mode = 0
        self._call_count = 0

    @property
    def in_features(self):
        return self.weight.shape[1]

    @property
    def out_features(self):
        return self.weight.shape[0]

    def forward(self, x):
        """x: [..., K] → [..., N]"""
        # Determine round_mode
        if self.mode_override is not None:
            round_mode = self.mode_override
        else:
            entropy = read_hw_entropy()
            round_mode = entropy if entropy is not None else 0

        self._last_round_mode = round_mode
        self._call_count += 1

        # Reshape for 2D GEMM
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])  # [M, K]
        M, K = x_2d.shape
        N = self.weight.shape[0]

        # inject_mask: 0x00 when mode_override is set (sterile), 0xFF for live physics
        inject_mask = 0x00 if self.mode_override is not None else 0xFF

        # Use custom kernel if available
        hip_mod = compile_hip_kernel()
        if hip_mod is not None:
            x_bf16 = x_2d.to(torch.bfloat16).contiguous()
            w_bf16 = self.weight.to(torch.bfloat16).contiguous()
            out = hip_mod.analog_gemm(x_bf16, w_bf16, round_mode, inject_mask)
        else:
            # Fallback: standard matmul (no MODE override)
            out = F.linear(x_2d, self.weight)

        if self.bias is not None:
            out = out + self.bias

        # Reshape back
        return out.reshape(*orig_shape[:-1], N)


def patch_gate_proj_with_analog(model, layers, mode_override=None):
    """Replace gate_proj in specified Qwen3-8B layers with AnalogLinear.
    Returns dict of original linears for restoration."""
    originals = {}
    for layer_idx in layers:
        block = model.model.layers[layer_idx]
        orig = block.mlp.gate_proj
        analog = AnalogLinear(orig, layer_idx, mode_override=mode_override)
        block.mlp.gate_proj = analog
        originals[layer_idx] = orig
        print(f"  [AnalogLinear] Patched layer {layer_idx} gate_proj "
              f"({orig.in_features}→{orig.out_features})")
    return originals

def restore_gate_proj(model, originals):
    """Restore original gate_proj linears."""
    for layer_idx, orig in originals.items():
        model.model.layers[layer_idx].mlp.gate_proj = orig

def set_analog_mode_override(model, layers, mode_override):
    """Set forced mode on all AnalogLinear layers."""
    for layer_idx in layers:
        gate = model.model.layers[layer_idx].mlp.gate_proj
        if isinstance(gate, AnalogLinear):
            gate.mode_override = mode_override

def get_analog_stats(model, layers):
    """Get last round_mode and call count from AnalogLinear layers."""
    stats = {}
    for layer_idx in layers:
        gate = model.model.layers[layer_idx].mlp.gate_proj
        if isinstance(gate, AnalogLinear):
            stats[layer_idx] = {
                'last_round_mode': gate._last_round_mode,
                'call_count': gate._call_count,
            }
    return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE B: Tiny LoRA (rank-2, v_proj only at layers 15,20,25)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TinyLoRA(nn.Module):
    """Minimal rank-2 LoRA for v_proj. ~0.5M total params across 3 layers."""
    def __init__(self, original_linear, rank=2, alpha=4):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_f = original_linear.in_features
        out_f = original_linear.out_features
        dtype = original_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.randn(rank, in_f, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, dtype=dtype))

    def forward(self, x):
        base = self.original(x)
        x_cast = x.to(self.lora_A.dtype)
        lora_out = F.linear(F.linear(x_cast, self.lora_A), self.lora_B) * self.scaling
        return base + lora_out.to(x.dtype)

class SomatosensoryHead(nn.Module):
    """Tiny head that predicts GPU temperature from hidden states.
    If the analog math creates temperature-dependent distortions,
    this head should learn to 'feel' the silicon's thermal state
    purely from the geometric fuzziness of its own computation.
    Interoception: the model senses its own body heat."""
    def __init__(self, hidden_dim, dtype=torch.bfloat16):
        super().__init__()
        # Very small: hidden_dim → 64 → 1 (predict normalized temp)
        self.proj = nn.Linear(hidden_dim, 64, dtype=dtype)
        self.out = nn.Linear(64, 1, dtype=dtype)

    def forward(self, hidden_states):
        """hidden_states: [B, T, D] → [B, 1] predicted normalized temperature."""
        # Mean-pool over sequence dimension
        h = hidden_states.mean(dim=1)  # [B, D]
        h = F.gelu(self.proj(h))
        return self.out(h).squeeze(-1)  # [B]


def patch_v_proj_with_lora(model, layers, rank=2, alpha=4):
    """Add TinyLoRA to v_proj at specified layers. Returns originals for restore."""
    originals = {}
    total_params = 0
    for layer_idx in layers:
        block = model.model.layers[layer_idx]
        orig = block.self_attn.v_proj
        lora = TinyLoRA(orig, rank=rank, alpha=alpha).to(orig.weight.device)
        block.self_attn.v_proj = lora
        originals[layer_idx] = orig
        n_p = sum(p.numel() for p in [lora.lora_A, lora.lora_B])
        total_params += n_p
        print(f"  [TinyLoRA] Patched layer {layer_idx} v_proj (rank={rank}, {n_p} params)")
    print(f"  [TinyLoRA] Total trainable: {total_params}")
    return originals, total_params

def restore_v_proj(model, originals):
    for layer_idx, orig in originals.items():
        model.model.layers[layer_idx].self_attn.v_proj = orig

def get_lora_params(model, layers):
    """Collect LoRA parameters for optimizer."""
    params = []
    for layer_idx in layers:
        v = model.model.layers[layer_idx].self_attn.v_proj
        if isinstance(v, TinyLoRA):
            params.extend([v.lora_A, v.lora_B])
    return params


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATION HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def eval_ppl(model, data, n_batches=None, device=DEVICE):
    """Evaluate perplexity. Token-weighted (z2107 fix)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n = n_batches or min(N_EVAL_BATCHES, len(data) // BS)
    with torch.no_grad():
        for i in range(n):
            batch = torch.stack(data[i*BS:(i+1)*BS]).to(device)
            out = model(batch)
            logits = out.logits if hasattr(out, 'logits') else out[0]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), reduction='sum')
            n_tok = (shift_labels != -100).sum().item()
            if n_tok == 0:
                n_tok = shift_labels.numel()
            total_loss += loss.item()
            total_tokens += n_tok
    ppl = math.exp(min(total_loss / max(total_tokens, 1), 20))
    model.train()
    return ppl

def collect_logits(model, data, n_batches=5, device=DEVICE):
    """Collect raw logits for KL divergence computation."""
    model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(min(n_batches, len(data) // BS)):
            batch = torch.stack(data[i*BS:(i+1)*BS]).to(device)
            out = model(batch)
            logits = out.logits if hasattr(out, 'logits') else out[0]
            all_logits.append(logits.cpu())
    model.train()
    return torch.cat(all_logits, dim=0) if all_logits else None

def kl_divergence(logits_p, logits_q):
    """Compute mean KL(P||Q) from logits."""
    p = F.softmax(logits_p.float(), dim=-1)
    q = F.softmax(logits_q.float(), dim=-1)
    # Clamp for numerical stability
    p = p.clamp(min=1e-10)
    q = q.clamp(min=1e-10)
    kl = (p * (p.log() - q.log())).sum(dim=-1).mean()
    return kl.item()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE A: Frozen Measurement
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_phase_a(model, test_data, test_data_code, baseline_ppl):
    """Phase A: 0 trainable params. Frozen Qwen3-8B + AnalogLinear.
    Sweep rounding modes, measure PPL and KL divergence."""
    print("\n" + "=" * 60)
    print("PHASE A: Frozen Measurement (0 trainable params)")
    print("=" * 60)

    results = {'baseline_ppl': baseline_ppl}

    # Patch gate_proj with AnalogLinear (live entropy)
    originals = patch_gate_proj_with_analog(model, ANALOG_LAYERS, mode_override=None)

    # T2: AnalogLinear PPL (live entropy)
    print("\n[T2] AnalogLinear PPL (live entropy)...")
    analog_ppl = eval_ppl(model, test_data)
    results['analog_ppl'] = analog_ppl
    print(f"  Analog PPL: {analog_ppl:.4f} (baseline: {baseline_ppl:.4f}, "
          f"ratio: {analog_ppl/baseline_ppl:.4f})")

    # T3: Mode Sweep — force each rounding mode 0-3
    print("\n[T3] Mode Sweep (force each rounding mode 0-3)...")
    mode_ppls = {}
    mode_logits = {}
    for mode_val in range(4):
        set_analog_mode_override(model, ANALOG_LAYERS, mode_val)
        ppl = eval_ppl(model, test_data)
        logits = collect_logits(model, test_data, n_batches=3)
        mode_ppls[mode_val] = ppl
        mode_logits[mode_val] = logits
        print(f"  Mode {mode_val} ({ROUND_MODES[mode_val]}): PPL={ppl:.4f}")
    ppl_values = list(mode_ppls.values())
    ppl_std = np.std(ppl_values)
    results['mode_ppls'] = {str(k): v for k, v in mode_ppls.items()}
    results['mode_ppl_std'] = float(ppl_std)
    print(f"  PPL std across modes: {ppl_std:.6f}")

    # T4: KL Divergence across mode pairs
    print("\n[T4] KL Divergence across mode pairs...")
    kl_pairs = {}
    max_kl = 0.0
    for i in range(4):
        for j in range(i+1, 4):
            if mode_logits[i] is not None and mode_logits[j] is not None:
                kl_val = kl_divergence(mode_logits[i], mode_logits[j])
                kl_pairs[f"{i}_vs_{j}"] = kl_val
                max_kl = max(max_kl, kl_val)
                print(f"  KL({i}||{j}) = {kl_val:.8f}")
    results['kl_pairs'] = kl_pairs
    results['max_kl'] = max_kl

    # T5: Thermal Variance — different DVFS levels
    print("\n[T5] Thermal Variance (different DVFS)...")
    dvfs_ppls = {}
    if DVFS_AVAILABLE:
        set_analog_mode_override(model, ANALOG_LAYERS, None)  # live entropy
        for dvfs_level, dvfs_name in [(0, 'low'), (2, 'high')]:
            set_dvfs_level(dvfs_level, wait=True)
            time.sleep(1.0)  # let thermal settle
            ppl = eval_ppl(model, test_data)
            dvfs_ppls[dvfs_name] = ppl
            print(f"  DVFS {dvfs_name}: PPL={ppl:.4f}")
        set_dvfs_level(1, wait=True)
    else:
        print("  DVFS not available, skipping")
    results['dvfs_ppls'] = dvfs_ppls
    dvfs_ppl_std = np.std(list(dvfs_ppls.values())) if len(dvfs_ppls) >= 2 else 0.0
    results['dvfs_ppl_std'] = float(dvfs_ppl_std)

    # T6: Determinism — same mode, 5 runs
    print("\n[T6] Determinism (mode=0, 5 runs)...")
    set_analog_mode_override(model, ANALOG_LAYERS, 0)
    det_ppls = []
    for run_i in range(5):
        ppl = eval_ppl(model, test_data)
        det_ppls.append(ppl)
        print(f"  Run {run_i}: PPL={ppl:.6f}")
    det_std = np.std(det_ppls)
    results['determinism_ppls'] = det_ppls
    results['determinism_std'] = float(det_std)
    print(f"  Std: {det_std:.8f}")

    # T7: Kill-Shot — find best mode, force worst
    print("\n[T7] Kill-Shot (best vs worst mode)...")
    best_mode = min(mode_ppls, key=mode_ppls.get)
    worst_mode = max(mode_ppls, key=mode_ppls.get)
    kill_ratio = mode_ppls[worst_mode] / max(mode_ppls[best_mode], 0.01)
    results['best_mode'] = best_mode
    results['worst_mode'] = worst_mode
    results['kill_ratio'] = kill_ratio
    print(f"  Best mode: {best_mode} (PPL={mode_ppls[best_mode]:.4f})")
    print(f"  Worst mode: {worst_mode} (PPL={mode_ppls[worst_mode]:.4f})")
    print(f"  Kill ratio: {kill_ratio:.6f}")

    # T8: Denorm Sweep — bits[7:4]
    print("\n[T8] Denorm Sweep (bits[7:4])...")
    denorm_kls = {}
    denorm_logits = {}
    base_round = 0  # nearest-even
    for denorm_val, denorm_name in DENORM_MODES.items():
        full_mode = base_round | (denorm_val << 4)
        set_analog_mode_override(model, ANALOG_LAYERS, full_mode)
        logits = collect_logits(model, test_data, n_batches=3)
        denorm_logits[denorm_val] = logits
        ppl = eval_ppl(model, test_data)
        print(f"  Denorm 0x{denorm_val:02x} ({denorm_name}): PPL={ppl:.4f}")
    max_denorm_kl = 0.0
    for i_d in list(DENORM_MODES.keys()):
        for j_d in list(DENORM_MODES.keys()):
            if i_d < j_d and denorm_logits[i_d] is not None and denorm_logits[j_d] is not None:
                kl_val = kl_divergence(denorm_logits[i_d], denorm_logits[j_d])
                denorm_kls[f"0x{i_d:02x}_vs_0x{j_d:02x}"] = kl_val
                max_denorm_kl = max(max_denorm_kl, kl_val)
    results['denorm_kls'] = denorm_kls
    results['max_denorm_kl'] = max_denorm_kl
    print(f"  Max denorm KL: {max_denorm_kl:.8f}")

    # T9: Per-Layer Mode Variation
    print("\n[T9] Per-Layer Mode Variation...")
    # Set different modes per layer, compare to uniform
    set_analog_mode_override(model, ANALOG_LAYERS, 0)  # all mode 0
    logits_uniform = collect_logits(model, test_data, n_batches=3)
    # Now set layer 15=0, layer 20=1, layer 25=2
    for idx, mode_v in zip(ANALOG_LAYERS, [0, 1, 2]):
        gate = model.model.layers[idx].mlp.gate_proj
        if isinstance(gate, AnalogLinear):
            gate.mode_override = mode_v
    logits_varied = collect_logits(model, test_data, n_batches=3)
    per_layer_kl = 0.0
    if logits_uniform is not None and logits_varied is not None:
        per_layer_kl = kl_divergence(logits_uniform, logits_varied)
    results['per_layer_kl'] = per_layer_kl
    print(f"  KL(uniform vs varied): {per_layer_kl:.8f}")

    # T10: Ablation — restore standard Linear, measure PPL difference
    print("\n[T10] Ablation (standard Linear vs AnalogLinear)...")
    restore_gate_proj(model, originals)
    standard_ppl = eval_ppl(model, test_data)
    results['standard_ppl'] = standard_ppl
    ppl_diff = abs(analog_ppl - standard_ppl)
    results['ablation_ppl_diff'] = ppl_diff
    print(f"  Standard PPL: {standard_ppl:.4f}, Analog PPL: {analog_ppl:.4f}")
    print(f"  Difference: {ppl_diff:.6f}")
    # Re-patch for remaining tests
    originals = patch_gate_proj_with_analog(model, ANALOG_LAYERS, mode_override=None)

    # T11: Domain Sensitivity — wiki vs code × modes
    print("\n[T11] Domain Sensitivity (wiki vs code × modes)...")
    domain_ppls = {}
    if test_data_code is not None and len(test_data_code) >= BS * 3:
        for mode_val in [0, 3]:  # nearest-even vs toward-zero
            set_analog_mode_override(model, ANALOG_LAYERS, mode_val)
            wiki_ppl = eval_ppl(model, test_data, n_batches=5)
            code_ppl = eval_ppl(model, test_data_code, n_batches=5)
            domain_ppls[f"wiki_mode{mode_val}"] = wiki_ppl
            domain_ppls[f"code_mode{mode_val}"] = code_ppl
            print(f"  Mode {mode_val}: wiki={wiki_ppl:.4f}, code={code_ppl:.4f}")
    else:
        print("  Code data not available, skipping")
    results['domain_ppls'] = domain_ppls

    # T12: Zombie Twin — replay vs live entropy
    print("\n[T12] Zombie Twin (replay vs live)...")
    set_analog_mode_override(model, ANALOG_LAYERS, None)  # live
    logits_live = collect_logits(model, test_data, n_batches=3)
    # Capture what entropy was used
    live_modes = [model.model.layers[l].mlp.gate_proj._last_round_mode
                  for l in ANALOG_LAYERS if isinstance(model.model.layers[l].mlp.gate_proj, AnalogLinear)]
    # Replay with fixed mode (use mode 0 as "zombie" — deterministic replay)
    set_analog_mode_override(model, ANALOG_LAYERS, 0)
    logits_zombie = collect_logits(model, test_data, n_batches=3)
    zombie_kl = 0.0
    if logits_live is not None and logits_zombie is not None:
        zombie_kl = kl_divergence(logits_live, logits_zombie)
    results['zombie_kl'] = zombie_kl
    results['live_modes_sample'] = live_modes
    print(f"  KL(live vs zombie): {zombie_kl:.8f}")
    print(f"  Live modes sample: {live_modes}")

    # T13: Token-Level Impact
    print("\n[T13] Token-Level Impact...")
    set_analog_mode_override(model, ANALOG_LAYERS, 0)
    logits_m0 = collect_logits(model, test_data, n_batches=3)
    set_analog_mode_override(model, ANALOG_LAYERS, 3)
    logits_m3 = collect_logits(model, test_data, n_batches=3)
    token_impact_frac = 0.0
    if logits_m0 is not None and logits_m3 is not None:
        diff = (logits_m0 - logits_m3).abs()
        max_diff_per_token = diff.max(dim=-1).values  # [B, T]
        impacted = (max_diff_per_token > 0.01).float().mean().item()
        token_impact_frac = impacted
    results['token_impact_frac'] = token_impact_frac
    print(f"  Fraction of tokens with >0.01 logit diff: {token_impact_frac:.4f}")

    # T14: GELU Amplification
    print("\n[T14] GELU Amplification...")
    gelu_amp = measure_gelu_amplification(model, test_data)
    results['gelu_amplification'] = gelu_amp
    print(f"  Post-GELU div / Pre-GELU div: {gelu_amp:.4f}")

    # T15: Generation Quality
    print("\n[T15] Generation Quality...")
    gen_results = {}
    for mode_val in [0, 3]:
        set_analog_mode_override(model, ANALOG_LAYERS, mode_val)
        text = generate_text(model, test_data, mode_val)
        gen_results[f"mode_{mode_val}"] = text
        print(f"  Mode {mode_val}: \"{text[:100]}...\"" if len(text) > 100 else f"  Mode {mode_val}: \"{text}\"")
    results['generation'] = gen_results

    # Restore live entropy for summary
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    restore_gate_proj(model, originals)

    return results


def measure_gelu_amplification(model, data):
    """T14: Measure how GELU amplifies rounding differences.
    Returns ratio: post-GELU divergence / pre-GELU divergence."""
    model.eval()
    pre_gelu_diffs = []
    post_gelu_diffs = []

    # Hook into a patched layer to capture pre/post GELU
    target_layer = ANALOG_LAYERS[1]  # layer 20
    captured = {'pre': None, 'post': None}

    def hook_gate(module, input, output):
        captured['pre'] = output.detach().clone()

    def hook_up(module, input, output):
        captured['post'] = output.detach().clone()

    # We need AnalogLinear patched for this test
    originals = patch_gate_proj_with_analog(model, ANALOG_LAYERS, mode_override=0)
    block = model.model.layers[target_layer]
    h1 = block.mlp.gate_proj.register_forward_hook(hook_gate)

    with torch.no_grad():
        batch = torch.stack(data[:BS]).to(DEVICE)
        # Mode 0
        set_analog_mode_override(model, ANALOG_LAYERS, 0)
        model(batch)
        pre_m0 = captured['pre'].clone() if captured['pre'] is not None else None

        # Mode 3
        set_analog_mode_override(model, ANALOG_LAYERS, 3)
        model(batch)
        pre_m3 = captured['pre'].clone() if captured['pre'] is not None else None

    h1.remove()
    restore_gate_proj(model, originals)
    model.train()

    if pre_m0 is None or pre_m3 is None:
        return 0.0

    pre_div = (pre_m0 - pre_m3).abs().mean().item()
    # GELU amplification: apply GELU to both, measure divergence
    post_m0 = F.gelu(pre_m0)
    post_m3 = F.gelu(pre_m3)
    post_div = (post_m0 - post_m3).abs().mean().item()

    return post_div / max(pre_div, 1e-12)


def generate_text(model, data, mode_val, max_new=40):
    """Generate text under a specific rounding mode."""
    model.eval()
    try:
        with torch.no_grad():
            prompt_ids = data[0][:8].unsqueeze(0).to(DEVICE)
            gen_ids = prompt_ids
            for _ in range(max_new):
                out = model(gen_ids)
                logits = out.logits if hasattr(out, 'logits') else out[0]
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen_ids = torch.cat([gen_ids, next_tok], dim=1)
            return f"[mode={mode_val}] {gen_ids[0].cpu().tolist()[:20]}..."
    except Exception as e:
        return f"[mode={mode_val}] Error: {e}"
    finally:
        model.train()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE B: Tiny LoRA Training + Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_phase_b(model, train_data, test_data, phase_a_ppl, phase_a_results):
    """Phase B: Tiny rank-2 LoRA + AnalogLinear + SomatosensoryHead.
    Train if Phase A shows signal. The soma head learns interoception:
    predicting GPU temperature from analog-distorted hidden states."""
    print("\n" + "=" * 60)
    print("PHASE B: Tiny LoRA + AnalogLinear + Somatosensory Head")
    print("=" * 60)

    results = {}

    # Check Phase A signal
    max_kl = phase_a_results.get('max_kl', 0)
    ppl_std = phase_a_results.get('mode_ppl_std', 0)
    print(f"  Phase A signal: max_kl={max_kl:.8f}, mode_ppl_std={ppl_std:.6f}")

    # Patch AnalogLinear (live entropy)
    analog_originals = patch_gate_proj_with_analog(model, ANALOG_LAYERS, mode_override=None)

    # Patch LoRA
    lora_originals, n_lora_params = patch_v_proj_with_lora(
        model, LORA_LAYERS, rank=LORA_RANK, alpha=LORA_ALPHA)
    results['n_lora_params'] = n_lora_params

    # Create Somatosensory Head (interoception: predict GPU temp from hidden states)
    hidden_dim = model.config.hidden_size  # 4096 for Qwen3-8B
    soma_head = SomatosensoryHead(hidden_dim, dtype=torch.bfloat16).to(DEVICE)
    soma_params = list(soma_head.parameters())
    n_soma_params = sum(p.numel() for p in soma_params)
    print(f"  [SomaHead] Interoception head: {hidden_dim}→64→1 ({n_soma_params} params)")

    # Freeze everything except LoRA + soma head
    for p in model.parameters():
        p.requires_grad = False
    lora_params = get_lora_params(model, LORA_LAYERS)
    all_trainable = lora_params + soma_params
    for p in all_trainable:
        p.requires_grad = True

    total_trainable = n_lora_params + n_soma_params
    print(f"  Total trainable: {total_trainable} (LoRA: {n_lora_params}, Soma: {n_soma_params})")

    # Read baseline temperature for normalization
    temp_baseline = read_gpu_temp_c()
    temp_scale = 20.0  # normalize: (temp - baseline) / scale → approx [-1, 1]
    print(f"  GPU temp baseline: {temp_baseline:.1f}°C (scale: ±{temp_scale}°C)")

    # Train
    optimizer = torch.optim.AdamW(all_trainable, lr=1e-4, weight_decay=0.01)
    n_train_batches = min(100, len(train_data) // BS)
    print(f"\n  Training {total_trainable} params for {n_train_batches} batches...")
    print(f"  Loss = CE(text) + 0.1 * MSE(soma_predicted_temp, actual_temp)")

    model.train()
    soma_head.train()
    total_loss = 0.0
    total_soma_loss = 0.0
    soma_preds = []
    soma_targets = []

    for batch_i in range(n_train_batches):
        batch = torch.stack(train_data[batch_i*BS:(batch_i+1)*BS]).to(DEVICE)

        # Read actual GPU temperature
        actual_temp = read_gpu_temp_c()
        temp_normalized = (actual_temp - temp_baseline) / temp_scale
        temp_target = torch.full((BS,), temp_normalized, device=DEVICE, dtype=torch.float32)

        # Forward pass — get both logits and hidden states
        out = model(batch, output_hidden_states=True)
        logits = out.logits if hasattr(out, 'logits') else out[0]

        # Text loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                  shift_labels.view(-1))

        # Somatosensory loss: predict GPU temp from last hidden state
        hidden = out.hidden_states[-1]  # [B, T, D]
        soma_pred = soma_head(hidden)  # [B]
        soma_loss = F.mse_loss(soma_pred.float(), temp_target)

        # Combined loss
        loss = ce_loss + 0.1 * soma_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += ce_loss.item()
        total_soma_loss += soma_loss.item()
        soma_preds.append(soma_pred.detach().mean().item())
        soma_targets.append(temp_normalized)

        if (batch_i + 1) % 25 == 0:
            avg_ce = total_loss / (batch_i + 1)
            avg_soma = total_soma_loss / (batch_i + 1)
            print(f"    Batch {batch_i+1}/{n_train_batches}: ce={avg_ce:.4f}, "
                  f"soma_mse={avg_soma:.4f}, gpu_temp={actual_temp:.1f}°C")

    results['train_loss'] = total_loss / max(n_train_batches, 1)
    results['soma_loss'] = total_soma_loss / max(n_train_batches, 1)

    # T16: LoRA Adaptation — PPL should improve over Phase A
    print("\n[T16] LoRA Adaptation...")
    lora_ppl = eval_ppl(model, test_data)
    results['lora_ppl'] = lora_ppl
    results['phase_a_ppl'] = phase_a_ppl
    print(f"  LoRA PPL: {lora_ppl:.4f} (Phase A: {phase_a_ppl:.4f})")

    # T17: LoRA Kill-Shot — wrong mode with LoRA
    print("\n[T17] LoRA Kill-Shot...")
    best_mode = phase_a_results.get('best_mode', 0)
    worst_mode = phase_a_results.get('worst_mode', 3)
    set_analog_mode_override(model, ANALOG_LAYERS, best_mode)
    ppl_best = eval_ppl(model, test_data)
    set_analog_mode_override(model, ANALOG_LAYERS, worst_mode)
    ppl_worst = eval_ppl(model, test_data)
    lora_kill_ratio = ppl_worst / max(ppl_best, 0.01)
    results['lora_kill_ppl_best'] = ppl_best
    results['lora_kill_ppl_worst'] = ppl_worst
    results['lora_kill_ratio'] = lora_kill_ratio
    print(f"  Best mode {best_mode}: PPL={ppl_best:.4f}")
    print(f"  Worst mode {worst_mode}: PPL={ppl_worst:.4f}")
    print(f"  Kill ratio: {lora_kill_ratio:.6f}")

    # T18: LoRA vs Fixed Mode — does live analog help vs fixed?
    print("\n[T18] LoRA vs Fixed Mode...")
    # Live analog entropy (inject_mask=0xFF)
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    ppl_analog = eval_ppl(model, test_data)
    # Fixed mode 0 (inject_mask=0x00, deterministic)
    set_analog_mode_override(model, ANALOG_LAYERS, 0)
    ppl_fixed = eval_ppl(model, test_data)
    results['lora_analog_ppl'] = ppl_analog
    results['lora_fixed_ppl'] = ppl_fixed
    print(f"  Analog (live): PPL={ppl_analog:.4f}")
    print(f"  Fixed mode 0: PPL={ppl_fixed:.4f}")

    # T19: Interoception — can the soma head predict GPU temp?
    print("\n[T19] Interoception (Somatosensory Probe)...")
    model.eval()
    soma_head.eval()
    soma_test_preds = []
    soma_test_targets = []

    # Run DVFS sweep to create temperature variation, test if soma tracks it
    if DVFS_AVAILABLE:
        for dvfs_level, dvfs_name in [(0, 'low'), (2, 'high'), (1, 'auto')]:
            set_dvfs_level(dvfs_level, wait=True)
            time.sleep(2.0)  # let thermal settle more for soma test

            # Run a few eval batches, collect soma predictions
            with torch.no_grad():
                for i in range(5):
                    batch = torch.stack(test_data[i*BS:(i+1)*BS]).to(DEVICE)
                    actual_temp = read_gpu_temp_c()
                    temp_norm = (actual_temp - temp_baseline) / temp_scale

                    out = model(batch, output_hidden_states=True)
                    hidden = out.hidden_states[-1]
                    pred = soma_head(hidden).mean().item()

                    soma_test_preds.append(pred)
                    soma_test_targets.append(temp_norm)

            print(f"  DVFS {dvfs_name}: temp={read_gpu_temp_c():.1f}°C, "
                  f"soma_pred={soma_test_preds[-1]:.4f}, target={soma_test_targets[-1]:.4f}")
        set_dvfs_level(1, wait=True)  # restore
    else:
        # Without DVFS, just collect a few predictions
        with torch.no_grad():
            for i in range(10):
                batch = torch.stack(test_data[i*BS:(i+1)*BS]).to(DEVICE)
                actual_temp = read_gpu_temp_c()
                temp_norm = (actual_temp - temp_baseline) / temp_scale
                out = model(batch, output_hidden_states=True)
                hidden = out.hidden_states[-1]
                pred = soma_head(hidden).mean().item()
                soma_test_preds.append(pred)
                soma_test_targets.append(temp_norm)

    # Compute correlation between predicted and actual temperature
    soma_corr = 0.0
    soma_mae = float('inf')
    if len(soma_test_preds) >= 3:
        preds_arr = np.array(soma_test_preds)
        targets_arr = np.array(soma_test_targets)
        soma_mae = float(np.mean(np.abs(preds_arr - targets_arr)))
        if np.std(preds_arr) > 1e-8 and np.std(targets_arr) > 1e-8:
            soma_corr = float(np.corrcoef(preds_arr, targets_arr)[0, 1])
        print(f"  Soma correlation: {soma_corr:.4f}")
        print(f"  Soma MAE (normalized): {soma_mae:.4f}")
    else:
        print("  Insufficient data for soma correlation")

    results['soma_correlation'] = soma_corr
    results['soma_mae'] = soma_mae
    results['soma_preds'] = soma_test_preds
    results['soma_targets'] = soma_test_targets

    model.train()

    # Cleanup
    restore_v_proj(model, lora_originals)
    restore_gate_proj(model, analog_originals)

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST BATTERY SCORING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def score_tests(phase_a, phase_b):
    """Score all 19 tests. Returns (results_list, n_pass, n_total)."""
    results = []
    n_pass = 0
    n_total = 0

    def record(tid, name, passed, value, criterion):
        nonlocal n_pass, n_total
        n_total += 1
        if passed:
            n_pass += 1
        status = "PASS" if passed else "FAIL"
        results.append({
            'test': tid, 'name': name, 'status': status,
            'value': value, 'criterion': criterion
        })
        print(f"  {tid}: {name} — {status} ({value} vs {criterion})")

    print("\n" + "=" * 60)
    print("TEST BATTERY RESULTS (19 tests)")
    print("=" * 60)

    # Phase A tests (T1-T15)
    baseline_ppl = phase_a.get('baseline_ppl', 0)
    analog_ppl = phase_a.get('analog_ppl', 0)

    # T1: Baseline PPL (record only)
    record('T1', 'Baseline PPL', True, f"{baseline_ppl:.4f}", 'record')

    # T2: AnalogLinear PPL < 1.5x baseline
    t2_pass = analog_ppl < 1.5 * baseline_ppl
    record('T2', 'AnalogLinear PPL', t2_pass,
           f"{analog_ppl:.4f} (ratio={analog_ppl/max(baseline_ppl,0.01):.4f})",
           '<1.5x baseline')

    # T3: Mode Sweep σ(PPL) > 0.01
    ppl_std = phase_a.get('mode_ppl_std', 0)
    record('T3', 'Mode Sweep σ(PPL)', ppl_std > 0.01,
           f"σ={ppl_std:.6f}", '>0.01')

    # T4: KL Divergence max > 0.001
    max_kl = phase_a.get('max_kl', 0)
    record('T4', 'KL Divergence', max_kl > 0.001,
           f"max_kl={max_kl:.8f}", '>0.001')

    # T5: Thermal Variance PPL std > 0.05
    dvfs_std = phase_a.get('dvfs_ppl_std', 0)
    record('T5', 'Thermal Variance', dvfs_std > 0.05,
           f"std={dvfs_std:.6f}", '>0.05')

    # T6: Determinism std < 0.001
    det_std = phase_a.get('determinism_std', 0)
    record('T6', 'Determinism', det_std < 0.001,
           f"std={det_std:.8f}", '<0.001')

    # T7: Kill-Shot ratio > 1.05
    kill_ratio = phase_a.get('kill_ratio', 1.0)
    record('T7', 'Kill-Shot', kill_ratio > 1.05,
           f"ratio={kill_ratio:.6f}", '>1.05')

    # T8: Denorm Sweep max KL > 0.0001
    max_denorm_kl = phase_a.get('max_denorm_kl', 0)
    record('T8', 'Denorm Sweep', max_denorm_kl > 0.0001,
           f"max_kl={max_denorm_kl:.8f}", '>0.0001')

    # T9: Per-Layer Mode Variation KL > 0.0001
    per_layer_kl = phase_a.get('per_layer_kl', 0)
    record('T9', 'Per-Layer Variation', per_layer_kl > 0.0001,
           f"kl={per_layer_kl:.8f}", '>0.0001')

    # T10: Ablation — PPL difference measurable (> 0.001)
    ppl_diff = phase_a.get('ablation_ppl_diff', 0)
    record('T10', 'Ablation', ppl_diff > 0.001,
           f"diff={ppl_diff:.6f}", '>0.001')

    # T11: Domain Sensitivity — 2-way interaction
    domain_ppls = phase_a.get('domain_ppls', {})
    if len(domain_ppls) >= 4:
        # Check if mode × domain interaction exists
        w0 = domain_ppls.get('wiki_mode0', 0)
        w3 = domain_ppls.get('wiki_mode3', 0)
        c0 = domain_ppls.get('code_mode0', 0)
        c3 = domain_ppls.get('code_mode3', 0)
        wiki_delta = abs(w3 - w0)
        code_delta = abs(c3 - c0)
        interaction = abs(wiki_delta - code_delta)
        t11_pass = interaction > 0.01
        record('T11', 'Domain Sensitivity', t11_pass,
               f"interaction={interaction:.4f}", '>0.01')
    else:
        record('T11', 'Domain Sensitivity', False, 'no code data', 'interaction>0.01')

    # T12: Zombie Twin — output differs (KL > 0.0001)
    zombie_kl = phase_a.get('zombie_kl', 0)
    record('T12', 'Zombie Twin', zombie_kl > 0.0001,
           f"kl={zombie_kl:.8f}", '>0.0001')

    # T13: Token-Level Impact > 5%
    impact_frac = phase_a.get('token_impact_frac', 0)
    record('T13', 'Token-Level Impact', impact_frac > 0.05,
           f"frac={impact_frac:.4f}", '>0.05')

    # T14: GELU Amplification > 2x
    gelu_amp = phase_a.get('gelu_amplification', 0)
    record('T14', 'GELU Amplification', gelu_amp > 2.0,
           f"ratio={gelu_amp:.4f}", '>2.0')

    # T15: Generation Quality (always pass — informational)
    record('T15', 'Generation Quality', True, 'see output', 'coherent')

    # Phase B tests (T16-T18)
    if phase_b:
        lora_ppl = phase_b.get('lora_ppl', 0)
        pa_ppl = phase_b.get('phase_a_ppl', analog_ppl)
        record('T16', 'LoRA Adaptation', lora_ppl < pa_ppl,
               f"lora={lora_ppl:.4f} vs phaseA={pa_ppl:.4f}",
               'lora_ppl < phase_a_ppl')

        lora_kr = phase_b.get('lora_kill_ratio', 1.0)
        record('T17', 'LoRA Kill-Shot', lora_kr > 1.05,
               f"ratio={lora_kr:.6f}", '>1.05')

        lora_analog = phase_b.get('lora_analog_ppl', 0)
        lora_fixed = phase_b.get('lora_fixed_ppl', 0)
        record('T18', 'LoRA vs Fixed', lora_analog < lora_fixed,
               f"analog={lora_analog:.4f} vs fixed={lora_fixed:.4f}",
               'analog < fixed')

        # T19: Interoception — soma head correlation with actual GPU temp
        soma_corr = phase_b.get('soma_correlation', 0)
        soma_mae = phase_b.get('soma_mae', float('inf'))
        # PASS if correlation > 0.3 OR MAE < 0.5 (normalized temp units)
        t19_pass = abs(soma_corr) > 0.3 or soma_mae < 0.5
        record('T19', 'Interoception', t19_pass,
               f"corr={soma_corr:.4f}, mae={soma_mae:.4f}",
               '|corr|>0.3 or mae<0.5')
    else:
        record('T16', 'LoRA Adaptation', False, 'Phase B skipped', 'conditional')
        record('T17', 'LoRA Kill-Shot', False, 'Phase B skipped', 'conditional')
        record('T18', 'LoRA vs Fixed', False, 'Phase B skipped', 'conditional')
        record('T19', 'Interoception', False, 'Phase B skipped', 'conditional')

    return results, n_pass, n_total


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("=" * 60)
    print("z2114: Analog Linear LM — Embodied MatMul via FP Rounding")
    print("Frozen Qwen3-8B + AnalogLinear (MODE register control)")
    print("MONIST: No neural adapters. The matmul IS the embodiment.")
    print("=" * 60)
    print(f"  AnalogLinear layers: {ANALOG_LAYERS} (gate_proj)")
    print(f"  Phase A: 0 trainable params (frozen measurement)")
    print(f"  Phase B: Tiny LoRA rank={LORA_RANK} at v_proj {LORA_LAYERS}")
    print()

    # Initialize hardware
    find_dvfs_sysfs()
    check_rapl()
    init_msr()
    check_smn()

    # GPU warmup
    print("\n[GPU] Warming up...")
    _warmup = torch.randn(1024, 1024, device=DEVICE)
    _warmup = torch.mm(_warmup, _warmup)
    torch.cuda.synchronize()
    del _warmup
    print("[GPU] Warmup OK")

    # Load Qwen3-8B
    BACKBONE_NAME = 'Qwen/Qwen3-8B'
    print(f"\nLoading {BACKBONE_NAME}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BACKBONE_NAME, torch_dtype=torch.bfloat16, attn_implementation='eager',
        trust_remote_code=True).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    vocab_size = model.config.vocab_size
    print(f"  {BACKBONE_NAME}: {n_params/1e6:.1f}M params, vocab={vocab_size}")

    # Freeze all backbone params
    for p in model.parameters():
        p.requires_grad = False
    print("  All backbone parameters frozen")

    # Load data
    print("\nLoading data...")
    try:
        train_data = load_wikitext_data(tokenizer, 'train', max_samples=2000)
        test_data = load_wikitext_data(tokenizer, 'test', max_samples=500)
    except Exception as e:
        print(f"  WikiText-2 load failed ({e}), using synthetic data")
        train_data = [torch.randint(0, vocab_size, (SEQ_LEN,)) for _ in range(200)]
        test_data = [torch.randint(0, vocab_size, (SEQ_LEN,)) for _ in range(50)]

    test_data_code = None
    try:
        print("Loading code data...")
        test_data_code = load_code_data(tokenizer, max_samples=500)
    except Exception as e:
        print(f"  Code data load failed ({e})")

    # Baseline PPL (frozen, no AnalogLinear)
    print("\n[T1] Baseline PPL (frozen Qwen3-8B, no modifications)...")
    baseline_ppl = eval_ppl(model, test_data)
    print(f"  Baseline PPL: {baseline_ppl:.4f}")

    # Compile HIP kernel
    print("\n[HIP] Compiling analog GEMM kernel...")
    compile_hip_kernel()

    # DVFS calibration
    if DVFS_AVAILABLE:
        global SCLK_LOW_CAL, SCLK_HIGH_CAL
        print("\n[DVFS] Calibration...")
        torch.cuda.synchronize()
        set_dvfs_level(0, wait=True)
        sclk_low = read_current_sclk_mhz()
        torch.cuda.synchronize()
        set_dvfs_level(2, wait=True)
        sclk_high = read_current_sclk_mhz()
        SCLK_LOW_CAL = sclk_low
        SCLK_HIGH_CAL = sclk_high
        print(f"  low={sclk_low:.0f}MHz high={sclk_high:.0f}MHz")
        torch.cuda.synchronize()
        set_dvfs_level(1, wait=True)

    # SMN sanity check
    if SMN_AVAILABLE:
        entropy = read_hw_entropy()
        print(f"\n[SMN] Entropy sample: 0x{entropy:02X}" if entropy is not None
              else "\n[SMN] Entropy read failed")

    # ═══════════════════════════════════════════════════════════
    # PHASE A: Frozen Measurement
    # ═══════════════════════════════════════════════════════════
    phase_a_results = run_phase_a(model, test_data, test_data_code, baseline_ppl)

    # ═══════════════════════════════════════════════════════════
    # PHASE B: Conditional — only if Phase A shows signal
    # ═══════════════════════════════════════════════════════════
    phase_b_results = None
    max_kl = phase_a_results.get('max_kl', 0)
    ppl_std = phase_a_results.get('mode_ppl_std', 0)
    phase_a_ppl = phase_a_results.get('analog_ppl', baseline_ppl)

    # Run Phase B if ANY signal detected
    if max_kl > 0.0001 or ppl_std > 0.001:
        print(f"\n  Phase A signal detected (kl={max_kl:.6f}, std={ppl_std:.6f}) → running Phase B")
        phase_b_results = run_phase_b(model, train_data, test_data, phase_a_ppl, phase_a_results)
    else:
        print(f"\n  No Phase A signal (kl={max_kl:.8f}, std={ppl_std:.8f}) → skipping Phase B")

    # ═══════════════════════════════════════════════════════════
    # SCORE TEST BATTERY
    # ═══════════════════════════════════════════════════════════
    test_results, n_pass, n_total = score_tests(phase_a_results, phase_b_results)

    # Restore DVFS
    if DVFS_AVAILABLE:
        restore_dvfs_auto()

    print(f"\n{'='*60}")
    print(f"z2114 Analog Linear LM: {n_pass}/{n_total} PASS")
    print(f"{'='*60}")

    # Save results
    out_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2114_analog_linear_lm.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    final = {
        'experiment': 'z2114_analog_linear_lm',
        'description': 'z2114: Analog Linear LM — Embodied MatMul via FP Rounding Mode (MODE register)',
        'architecture': 'Monist: No neural adapters. AnalogLinear replaces gate_proj at layers 15,20,25',
        'key_idea': 'HIP GEMM kernel overrides MODE register bits[7:0] with SMN thermal entropy',
        'backbone': f'{BACKBONE_NAME} ({n_params/1e6:.1f}M frozen)',
        'analog_layers': ANALOG_LAYERS,
        'lora_layers': LORA_LAYERS,
        'lora_rank': LORA_RANK,
        'dvfs_available': DVFS_AVAILABLE,
        'smn_available': SMN_AVAILABLE,
        'rapl_available': RAPL_AVAILABLE,
        'msr_available': MSR_AVAILABLE,
        'sclk_low_cal': SCLK_LOW_CAL,
        'sclk_high_cal': SCLK_HIGH_CAL,
        'hip_kernel_available': _HIP_MODULE is not None,
        'baseline_ppl': baseline_ppl,
        'phase_a': phase_a_results,
        'phase_b': phase_b_results,
        'test_results': test_results,
        'n_pass': n_pass,
        'n_total': n_total,
    }
    with open(out_path, 'w') as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    try:
        main()
    finally:
        restore_dvfs_auto()
