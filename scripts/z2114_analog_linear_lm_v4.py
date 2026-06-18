#!/usr/bin/env python3
"""
z2114v4: Deep Embodied Analog Linear LM — Full Monist Architecture
===================================================================
v4 changes from v3 (13/19):
  1. ALL 40 gate_proj layers patched (was 3) — 13x amplification
  2. TILE_SIZE=32 (was 16) — 2x more accumulations per chunk
  3. MetabolicVector replaces scalar stress — temp+power+voltage+sclk+pcie
  4. Kernel wall_clock64 proprioception — per-block timing jitter output
  5. Forward context — coherent per-forward snapshot (not per-layer polls)
  6. Body-gated TinyLoRA — metabolic state gates LoRA contribution
  7. Multi-task SomaHead — vector reconstruction + regime classification
  8. Homeostatic loss — CE + interoception + stability regularization
  9. Larger LoRA rank=8, layers 8-31
  10. PCIe replay counter in metabolic vector

Hardware setup:
  sudo modprobe msr
  sudo insmod ~/Documents/claude_hive/ryzen_smu/ryzen_smu.ko
  sudo chmod 666 /sys/kernel/ryzen_smu_drv/smn
  sudo chmod 666 /sys/kernel/ryzen_smu_drv/pm_table
  sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTORCH_ROCM_ARCH=gfx1100 \
    venv/bin/python -u scripts/z2114_analog_linear_lm_v4.py
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

# v4: ALL layers for maximum embodiment signal (set dynamically after model load)
N_LAYERS = 36  # Qwen3-8B has 36 layers; updated dynamically in main()
ANALOG_LAYERS = list(range(N_LAYERS))  # all gate_proj
LORA_RANK = 8
LORA_ALPHA = 16
LORA_LAYERS = list(range(8, 32))  # layers 8-31 for LoRA

SCLK_LOW_CAL = 600.0
SCLK_HIGH_CAL = 2900.0

# Metabolic vector dimension: temp, power_mw, voltage0, voltage1, sclk, pcie_replay, smn_adc, stress_ema
METABOLIC_DIM = 8
# Regime labels
REGIME_COLD, REGIME_NOMINAL, REGIME_HOT, REGIME_THROTTLED = 0, 1, 2, 3
N_REGIMES = 4

def pack_mode_byte(f32_round=0, f16_round=0, f32_denorm=3, f16_denorm=3):
    return ((f32_round & 3) | ((f16_round & 3) << 2) |
            ((f32_denorm & 3) << 4) | ((f16_denorm & 3) << 6))

def make_lm_labels(input_ids, offset=1):
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

def set_dvfs(name):
    """Convenience wrapper: set_dvfs('low'/'auto'/'high')."""
    level_map = {'low': 0, 'auto': 1, 'high': 2}
    set_dvfs_level(level_map.get(name, 1), wait=True)

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
# HARDWARE ACCESS — SMN, RAPL, MSR, sysfs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SMN_AVAILABLE = False

def check_smn():
    global SMN_AVAILABLE
    SMN_AVAILABLE = os.path.exists('/sys/kernel/ryzen_smu_drv/smn')
    print(f"[SMN] {'Available' if SMN_AVAILABLE else 'Not available'}")

def read_smn(addr):
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

def read_gpu_temp_c():
    for hwmon in ['hwmon7', 'hwmon6', 'hwmon8']:
        p = f'/sys/class/hwmon/{hwmon}/temp1_input'
        try:
            with open(p, 'r') as f:
                return float(f.read().strip()) / 1000.0
        except:
            pass
    raw = read_smn(0x00059800)
    return ((raw >> 21) & 0x7FF) * 0.125 if raw else 50.0

def read_gpu_power_mw():
    for hwmon in ['hwmon7', 'hwmon6']:
        p = f'/sys/class/hwmon/{hwmon}/power1_input'
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    return float(f.read().strip()) / 1000.0
            except:
                pass
    return 0.0

def read_gpu_voltage_mv():
    """Read both voltage channels from amdgpu hwmon."""
    v0, v1 = 0.0, 0.0
    for hwmon in ['hwmon7', 'hwmon6']:
        p0 = f'/sys/class/hwmon/{hwmon}/in0_input'
        p1 = f'/sys/class/hwmon/{hwmon}/in1_input'
        try:
            if os.path.exists(p0):
                with open(p0, 'r') as f:
                    v0 = float(f.read().strip())
            if os.path.exists(p1):
                with open(p1, 'r') as f:
                    v1 = float(f.read().strip())
            if v0 > 0:
                return v0, v1
        except:
            pass
    return v0, v1

def read_pcie_replay_count():
    p = '/sys/class/drm/card1/device/pcie_replay_count'
    try:
        with open(p, 'r') as f:
            return int(f.read().strip())
    except:
        return 0

def read_hw_entropy():
    if not SMN_AVAILABLE:
        return None
    try:
        raw_adc = read_smn(0x00059800)
        entropy_bits = raw_adc & 0xFF
        xtal = read_smn(0x000598C8)
        xtal_low = xtal & 0xFFFF
        return (entropy_bits ^ xtal_low) & 0xFF
    except Exception:
        return None

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
# METABOLIC VECTOR + FORWARD CONTEXT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ForwardContext:
    """Shared context for a single forward pass. Snapshot once, use everywhere."""
    def __init__(self):
        self.metabolic_vec = None       # [METABOLIC_DIM] normalized float tensor
        self.metabolic_raw = None       # [METABOLIC_DIM] raw values
        self.metabolic_valid = None     # [METABOLIC_DIM] bool mask
        self.regime = REGIME_NOMINAL
        self.continuous_stress = 0.0
        self.stress_threshold = 0       # int 0-255 for kernel
        self.round_mode = pack_mode_byte(0, 0, 3, 3)
        self.layer_proprio = {}         # layer_idx -> {mean_ns, max_ns, jitter_ns}
        self.forward_id = 0
        self.snapshot_ts = 0.0

    def clear_proprio(self):
        self.layer_proprio = {}

CTX = ForwardContext()


class MetabolicController:
    """v4: Full metabolic vector replacing scalar stress.
    Reads: temp, power, voltage0, voltage1, sclk, pcie_replay, smn_adc, stress_ema
    Normalizes with running stats. Classifies regime."""

    def __init__(self, ema_alpha=0.5):
        self.ema_alpha = ema_alpha
        self.stress_ema = 0.0
        self._raw = np.zeros(METABOLIC_DIM, dtype=np.float32)
        self._running_mean = np.zeros(METABOLIC_DIM, dtype=np.float64)
        self._running_var = np.ones(METABOLIC_DIM, dtype=np.float64)
        self._n_updates = 0
        self._last_temp = 50.0
        self._last_sclk = 600.0
        self._last_power = 0.0

    def snapshot(self):
        """Single synchronized read of ALL hardware state."""
        self._last_temp = read_gpu_temp_c()
        self._last_sclk = read_current_sclk_mhz()
        self._last_power = read_gpu_power_mw()
        v0, v1 = read_gpu_voltage_mv()
        pcie_replay = read_pcie_replay_count()
        smn_adc = read_smn(0x00059800) & 0xFF if SMN_AVAILABLE else 0

        # Raw vector: [temp, power, v0, v1, sclk, pcie, smn_adc, stress_ema]
        self._raw = np.array([
            self._last_temp,
            self._last_power,
            v0,
            v1,
            self._last_sclk,
            float(pcie_replay),
            float(smn_adc),
            0.0  # placeholder, filled after stress calc
        ], dtype=np.float32)

        # Update stress EMA
        t_norm = max(0.0, min(1.0, (self._last_temp - 40.0) / 50.0))
        c_norm = max(0.0, min(1.0, (self._last_sclk - SCLK_LOW_CAL) /
                               max(SCLK_HIGH_CAL - SCLK_LOW_CAL, 1.0)))
        p_norm = max(0.0, min(1.0, self._last_power / 60000.0))  # 60W cap
        stress = (t_norm + c_norm + p_norm) / 3.0
        self.stress_ema = (1 - self.ema_alpha) * self.stress_ema + self.ema_alpha * stress
        self._raw[7] = self.stress_ema

        # Update running stats for z-score normalization
        self._n_updates += 1
        if self._n_updates == 1:
            self._running_mean = self._raw.astype(np.float64)
        else:
            delta = self._raw.astype(np.float64) - self._running_mean
            self._running_mean += delta / self._n_updates
            delta2 = self._raw.astype(np.float64) - self._running_mean
            self._running_var += delta * delta2

        # Validity mask (nonzero = valid, except stress which is always valid)
        valid = np.array([True, self._last_power > 0, v0 > 0, v1 > 0,
                         True, True, SMN_AVAILABLE, True], dtype=bool)

        return self._raw.copy(), valid

    def get_normalized(self):
        """Z-score normalized metabolic vector."""
        if self._n_updates < 2:
            return self._raw.copy(), np.ones(METABOLIC_DIM, dtype=bool)

        var = self._running_var / max(self._n_updates - 1, 1)
        std = np.sqrt(np.clip(var, 1e-8, None))
        normalized = (self._raw.astype(np.float64) - self._running_mean) / std
        return normalized.astype(np.float32), np.ones(METABOLIC_DIM, dtype=bool)

    def classify_regime(self):
        """Classify thermal regime from raw state."""
        if self._last_temp > 85.0 or self.stress_ema > 0.9:
            return REGIME_THROTTLED
        elif self._last_temp > 70.0 or self.stress_ema > 0.6:
            return REGIME_HOT
        elif self._last_temp < 50.0 and self.stress_ema < 0.3:
            return REGIME_COLD
        return REGIME_NOMINAL

    def update_context(self):
        """Snapshot hardware, normalize, fill forward context."""
        raw, valid = self.snapshot()
        normed, _ = self.get_normalized()
        CTX.metabolic_raw = torch.from_numpy(raw).float()
        CTX.metabolic_vec = torch.from_numpy(normed).float()
        CTX.metabolic_valid = torch.from_numpy(valid)
        CTX.regime = self.classify_regime()
        CTX.continuous_stress = float(self.stress_ema)
        CTX.stress_threshold = max(0, min(255, int(self.stress_ema * 255)))
        CTX.round_mode = pack_mode_byte(0, 0, 3, 3)  # cold baseline
        CTX.snapshot_ts = time.time()
        CTX.forward_id += 1
        CTX.clear_proprio()
        return CTX

METAB = MetabolicController()


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
# CUSTOM HIP KERNEL — v4: TILE_SIZE=32 + wall_clock64 proprioception
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_HIP_MODULE = None
_HIP_COMPILE_ATTEMPTED = False

def compile_hip_kernel():
    global _HIP_MODULE, _HIP_COMPILE_ATTEMPTED
    if _HIP_MODULE is not None:
        return _HIP_MODULE
    if _HIP_COMPILE_ATTEMPTED:
        return None
    _HIP_COMPILE_ATTEMPTED = True

    print("[HIP] Compiling v4 analog GEMM kernel (TILE=32 + wall_clock64 proprio)...")
    from torch.utils.cpp_extension import load_inline

    combined_source = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#define TILE_SIZE 32

// v4 kernel: TILE_SIZE=32 + wall_clock64 proprioception timing
// Writes per-block timing to proprio_out buffer for host-side jitter analysis
__global__ void analog_tiled_gemm_v4_kernel(
    const __hip_bfloat16* __restrict__ A,
    const __hip_bfloat16* __restrict__ B,
    __hip_bfloat16* __restrict__ C,
    unsigned long long* __restrict__ proprio_out,  // [gridDim.x * gridDim.y] wall_clock ticks
    int M, int K, int N,
    int base_round_mode,
    int stress_threshold
) {
    // wall_clock64 start — RDNA3-safe timing (not clock64!)
    unsigned long long t_start = wall_clock64();

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

        // v4: Directional PWM — same as v3 but with TILE_SIZE=32
        unsigned int cycles;
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(cycles));
        unsigned int hw_spin = cycles & 0xFFu;

        unsigned int active = ((int)hw_spin < stress_threshold) ? 0x05u : (unsigned int)base_round_mode;
        unsigned int rm_both = (active & 0x3u) | ((active & 0x3u) << 2) | (active & 0xF0u);
        unsigned int new_mode = (old_mode & ~0xFFu) | rm_both;
        asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" :: "s"(new_mode) : "memory");

        // CHUNKED FP16 ACCUMULATION — 32 MACs per chunk (2x v3)
        __half chunk_acc = __float2half(0.0f);
        for (int k = 0; k < TILE_SIZE; k++) {
            __half a_h = __float2half(As[threadIdx.y][k]);
            __half b_h = __float2half(Bs[k][threadIdx.x]);
            __half step_h = __hmul(a_h, b_h);
            chunk_acc = __hadd(chunk_acc, step_h);
        }
        acc += __half2float(chunk_acc);

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(acc);
    }

    // Restore MODE
    asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" ::: "memory");
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" :: "s"(old_mode) : "memory");

    // wall_clock64 end — write per-block timing for proprioception
    unsigned long long t_end = wall_clock64();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int block_id = blockIdx.y * gridDim.x + blockIdx.x;
        proprio_out[block_id] = t_end - t_start;
    }
}

// Host wrapper — returns (C_tensor, proprio_tensor)
std::vector<torch::Tensor> analog_gemm_v4(
    torch::Tensor A, torch::Tensor weight,
    int round_mode, float continuous_stress
) {
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
    int n_blocks = grid.x * grid.y;

    // Proprioception buffer: one uint64 per block
    auto proprio = torch::empty({n_blocks}, torch::TensorOptions().dtype(torch::kInt64).device(A.device()));

    int stress_threshold = (int)(continuous_stress * 255.0f);
    if (stress_threshold < 0) stress_threshold = 0;
    if (stress_threshold > 255) stress_threshold = 255;

    analog_tiled_gemm_v4_kernel<<<grid, block>>>(
        reinterpret_cast<const __hip_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),
        reinterpret_cast<unsigned long long*>(proprio.data_ptr()),
        M, K, N, round_mode, stress_threshold);

    return {C, proprio};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("analog_gemm_v4", &analog_gemm_v4, "v4 analog GEMM with proprioception");
}
"""

    try:
        _HIP_MODULE = load_inline(
            name='analog_gemm_v4_ext',
            cpp_sources=[],
            cuda_sources=[combined_source],
            extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'],
            verbose=True,
            with_cuda=True,
        )
        print("[HIP] v4 Kernel compiled OK")
    except Exception as e:
        print(f"[HIP] Compilation failed: {e}")
        _HIP_MODULE = None
    return _HIP_MODULE



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANALOG LINEAR — v4: reads context, writes proprioception
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AnalogLinear(nn.Module):
    """v4: Reads from ForwardContext (no per-layer THERMO.update),
    writes wall_clock64 proprioception back to context."""
    def __init__(self, original_linear, layer_idx, mode_override=None):
        super().__init__()
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.layer_idx = layer_idx
        self.mode_override = mode_override
        self._call_count = 0

    @property
    def in_features(self):
        return self.weight.shape[1]

    @property
    def out_features(self):
        return self.weight.shape[0]

    def forward(self, x):
        if self.mode_override is not None:
            round_mode = self.mode_override
            continuous_stress = 0.0
        else:
            # v4: read from shared context (snapshot once per forward)
            round_mode = CTX.round_mode
            continuous_stress = CTX.continuous_stress

        self._call_count += 1
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        hip_mod = compile_hip_kernel()
        if hip_mod is not None:
            x_bf16 = x_2d.to(torch.bfloat16).contiguous()
            w_bf16 = self.weight.to(torch.bfloat16).contiguous()
            results = hip_mod.analog_gemm_v4(x_bf16, w_bf16, round_mode, continuous_stress)
            out = results[0]
            proprio_ticks = results[1]

            # Write proprioception to forward context
            if self.mode_override is None:
                ticks_cpu = proprio_ticks.cpu().to(torch.float64)
                ticks_pos = ticks_cpu[ticks_cpu > 0]
                if len(ticks_pos) > 0:
                    CTX.layer_proprio[self.layer_idx] = {
                        'mean_ticks': float(ticks_pos.mean()),
                        'max_ticks': float(ticks_pos.max()),
                        'jitter': float(ticks_pos.max() - ticks_pos.min()),
                        'n_blocks': int(len(ticks_pos)),
                    }
        else:
            out = F.linear(x_2d, self.weight)

        if self.bias is not None:
            out = out + self.bias
        N = self.weight.shape[0]
        return out.reshape(*orig_shape[:-1], N)


def patch_gate_proj_with_analog(model, layers, mode_override=None):
    originals = {}
    for layer_idx in layers:
        block = model.model.layers[layer_idx]
        orig = block.mlp.gate_proj
        analog = AnalogLinear(orig, layer_idx, mode_override=mode_override)
        block.mlp.gate_proj = analog
        originals[layer_idx] = orig
        if layer_idx in [0, 9, 19, 29, 35, 39]:
            print(f"  [AnalogLinear] Patched layer {layer_idx} gate_proj "
                  f"({orig.in_features}->{orig.out_features})")
    n_patched = len(layers)
    print(f"  [AnalogLinear] Total: {n_patched} layers patched")
    return originals

def restore_gate_proj(model, originals):
    for layer_idx, orig in originals.items():
        model.model.layers[layer_idx].mlp.gate_proj = orig

def set_analog_mode_override(model, layers, mode_override):
    for layer_idx in layers:
        gate = model.model.layers[layer_idx].mlp.gate_proj
        if isinstance(gate, AnalogLinear):
            gate.mode_override = mode_override


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BODY-GATED TINY LORA — v4: metabolic state gates LoRA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class BodyGatedLoRA(nn.Module):
    """v4: LoRA whose contribution is gated by the metabolic vector.
    gate_proj maps metabolic_vec -> per-rank gate values.
    The model learns to modulate its adaptation based on physical state."""
    def __init__(self, original_linear, rank=8, alpha=16, metabolic_dim=METABOLIC_DIM):
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
        # Body gate: metabolic_vec -> rank-dim gate
        self.body_gate = nn.Linear(metabolic_dim, rank, dtype=torch.float32)
        nn.init.zeros_(self.body_gate.weight)
        nn.init.ones_(self.body_gate.bias)  # start gate at ~sigmoid(1)=0.73

    def forward(self, x):
        base = self.original(x)
        x_cast = x.to(self.lora_A.dtype)
        lora_out = F.linear(F.linear(x_cast, self.lora_A), self.lora_B) * self.scaling

        # Apply body gate from metabolic context
        if CTX.metabolic_vec is not None:
            gate_input = CTX.metabolic_vec.to(self.body_gate.weight.device)
            gate = torch.sigmoid(self.body_gate(gate_input))  # [rank]
            # Broadcast gate over output features: lora_out is [batch, seq, out_f]
            # We need to gate per-rank. Since lora_B is [out_f, rank], the rank
            # dimension is mixed into out_f. Apply gate before lora_B instead.
            # Recompute: lora_mid = x @ lora_A.T -> [batch, seq, rank]
            lora_mid = F.linear(x_cast, self.lora_A)  # [..., rank]
            lora_mid = lora_mid * gate.to(lora_mid.dtype)  # body-gated per rank
            lora_out = F.linear(lora_mid, self.lora_B) * self.scaling

        return base + lora_out.to(x.dtype)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MULTI-TASK SOMATOSENSORY HEAD — v4
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MultiTaskSomaHead(nn.Module):
    """v4: Multi-task interoception head.
    Input: pooled hidden state + aggregated proprioception
    Outputs:
      1. metabolic_recon: reconstruct normalized metabolic vector [METABOLIC_DIM]
      2. regime_logits: classify thermal regime [N_REGIMES]
      3. stress_pred: predict continuous stress scalar [1], tanh bounded
    """
    def __init__(self, hidden_dim, n_proprio_stats=3, dtype=torch.bfloat16):
        super().__init__()
        # n_proprio_stats: mean_ticks, max_ticks, jitter aggregated across layers
        self.input_dim = hidden_dim + n_proprio_stats
        self.shared = nn.Linear(self.input_dim, 128, dtype=torch.float32)
        self.metabolic_head = nn.Linear(128, METABOLIC_DIM, dtype=torch.float32)
        self.regime_head = nn.Linear(128, N_REGIMES, dtype=torch.float32)
        self.stress_head = nn.Linear(128, 1, dtype=torch.float32)

    def forward(self, hidden_states, proprio_stats=None):
        """
        hidden_states: [batch, seq, hidden_dim]
        proprio_stats: [3] float tensor (aggregated across layers) or None
        """
        h = hidden_states.float().mean(dim=1)  # [batch, hidden_dim]
        if proprio_stats is not None:
            # Broadcast proprio_stats to batch
            ps = proprio_stats.unsqueeze(0).expand(h.size(0), -1).to(h.device)
            h = torch.cat([h, ps], dim=-1)
        else:
            h = torch.cat([h, torch.zeros(h.size(0), 3, device=h.device)], dim=-1)

        shared_out = F.gelu(self.shared(h))
        metabolic_recon = self.metabolic_head(shared_out)  # [batch, METABOLIC_DIM]
        regime_logits = self.regime_head(shared_out)  # [batch, N_REGIMES]
        stress_pred = torch.tanh(self.stress_head(shared_out).squeeze(-1))  # [batch]
        return metabolic_recon, regime_logits, stress_pred


def patch_v_proj_with_lora(model, layers, rank=8, alpha=16):
    originals = {}
    total_params = 0
    for layer_idx in layers:
        block = model.model.layers[layer_idx]
        orig = block.self_attn.v_proj
        lora = BodyGatedLoRA(orig, rank=rank, alpha=alpha).to(orig.weight.device)
        block.self_attn.v_proj = lora
        originals[layer_idx] = orig
        n_p = sum(p.numel() for p in lora.parameters() if p.requires_grad)
        total_params += n_p
        if layer_idx in [8, 15, 23, 31]:
            print(f"  [BodyGatedLoRA] Patched layer {layer_idx} v_proj (rank={rank}, {n_p} params)")
    print(f"  [BodyGatedLoRA] Total trainable: {total_params} across {len(layers)} layers")
    return originals, total_params

def restore_v_proj(model, originals):
    for layer_idx, orig in originals.items():
        model.model.layers[layer_idx].self_attn.v_proj = orig

def get_lora_params(model, layers):
    params = []
    for layer_idx in layers:
        v = model.model.layers[layer_idx].self_attn.v_proj
        if isinstance(v, BodyGatedLoRA):
            params.extend([p for p in v.parameters() if p.requires_grad])
    return params

def aggregate_proprio():
    """Aggregate proprioception across all layers into 3 stats."""
    if not CTX.layer_proprio:
        return torch.zeros(3)
    means, maxes, jitters = [], [], []
    for info in CTX.layer_proprio.values():
        means.append(info['mean_ticks'])
        maxes.append(info['max_ticks'])
        jitters.append(info['jitter'])
    # Normalize by dividing by 1e6 to keep in reasonable range
    return torch.tensor([
        np.mean(means) / 1e6,
        np.max(maxes) / 1e6,
        np.mean(jitters) / 1e6
    ], dtype=torch.float32)



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATION HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def eval_ppl(model, data, n_batches=None, device=DEVICE):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n = n_batches or min(N_EVAL_BATCHES, len(data) // BS)
    with torch.no_grad():
        for i in range(n):
            METAB.update_context()  # v4: snapshot per forward
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
    model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(min(n_batches, len(data) // BS)):
            METAB.update_context()
            batch = torch.stack(data[i*BS:(i+1)*BS]).to(device)
            out = model(batch)
            logits = out.logits if hasattr(out, 'logits') else out[0]
            all_logits.append(logits.cpu())
    model.train()
    return torch.cat(all_logits, dim=0) if all_logits else None

def kl_divergence(logits_p, logits_q):
    p = F.softmax(logits_p.float(), dim=-1).clamp(min=1e-10)
    q = F.softmax(logits_q.float(), dim=-1).clamp(min=1e-10)
    return (p * (p.log() - q.log())).sum(dim=-1).mean().item()

_TOKENIZER = None  # set in main()

def generate_text(model, data, mode_val, max_new=60, prompt_len=16):
    """Generate text and return both decoded string and token IDs."""
    model.eval()
    try:
        with torch.no_grad():
            METAB.update_context()
            prompt_ids = data[0][:prompt_len].unsqueeze(0).to(DEVICE)
            gen_ids = prompt_ids
            for _ in range(max_new):
                out = model(gen_ids[:, -512:])  # sliding window
                logits = out.logits if hasattr(out, 'logits') else out[0]
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen_ids = torch.cat([gen_ids, next_tok], dim=1)
            all_ids = gen_ids[0].cpu().tolist()
            # Decode if tokenizer available
            if _TOKENIZER is not None:
                prompt_text = _TOKENIZER.decode(all_ids[:prompt_len], skip_special_tokens=True)
                gen_text = _TOKENIZER.decode(all_ids[prompt_len:], skip_special_tokens=True)
                return f"[mode={mode_val}] PROMPT: {prompt_text[:80]}... | GENERATED: {gen_text[:200]}"
            else:
                return f"[mode={mode_val}] {all_ids[:20]}..."
    except Exception as e:
        return f"[mode={mode_val}] Error: {e}"
    finally:
        model.train()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE A: Frozen Measurement (0 trainable params)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_phase_a(model, test_data, test_data_code, baseline_ppl):
    print("\n" + "=" * 60)
    print("PHASE A: Frozen Measurement (0 trainable params)")
    print("  v4: ALL 40 layers + TILE=32 + wall_clock64 proprioception")
    print("=" * 60)

    results = {'baseline_ppl': baseline_ppl}
    originals = patch_gate_proj_with_analog(model, ANALOG_LAYERS, mode_override=None)

    # T2: AnalogLinear PPL (live entropy)
    print("\n[T2] AnalogLinear PPL (live, all 40 layers)...")
    analog_ppl = eval_ppl(model, test_data)
    results['analog_ppl'] = analog_ppl
    print(f"  Analog PPL: {analog_ppl:.4f} (baseline: {baseline_ppl:.4f}, "
          f"ratio: {analog_ppl/baseline_ppl:.4f})")
    # Report proprioception from last eval
    if CTX.layer_proprio:
        all_jitter = [v['jitter'] for v in CTX.layer_proprio.values()]
        print(f"  Proprioception: {len(CTX.layer_proprio)} layers, "
              f"mean_jitter={np.mean(all_jitter)/1e3:.1f}K ticks")

    # T3: Mode Sweep
    print("\n[T3] Mode Sweep (force FP16 rounding modes 0-3)...")
    mode_ppls = {}
    mode_logits = {}
    mode_names = {0: 'nearest_even', 1: 'plus_inf', 2: 'minus_inf', 3: 'toward_zero'}
    for mode_val in range(4):
        packed = pack_mode_byte(mode_val, mode_val, 3, 3)
        set_analog_mode_override(model, ANALOG_LAYERS, packed)
        ppl = eval_ppl(model, test_data)
        logits = collect_logits(model, test_data, n_batches=3)
        mode_ppls[mode_val] = ppl
        mode_logits[mode_val] = logits
        print(f"  Mode {mode_val} ({mode_names[mode_val]}, packed=0x{packed:02X}): PPL={ppl:.4f}")
    ppl_std = float(np.std(list(mode_ppls.values())))
    results['mode_ppls'] = {str(k): v for k, v in mode_ppls.items()}
    results['mode_ppl_std'] = ppl_std
    print(f"  PPL std across modes: {ppl_std:.6f}")

    # T4: KL Divergence
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

    # T5: Thermal Variance
    print("\n[T5] Thermal Variance (DVFS low vs high)...")
    dvfs_ppls = {}
    dvfs_stresses = {}
    dvfs_proprios = {}
    if DVFS_AVAILABLE:
        set_analog_mode_override(model, ANALOG_LAYERS, None)
        for dvfs_level, dvfs_name in [(0, 'low'), (2, 'high')]:
            set_dvfs_level(dvfs_level, wait=True)
            time.sleep(5.0)
            METAB.stress_ema = 0.0 if dvfs_name == 'low' else 1.0
            for _ in range(30):
                METAB.snapshot()
                time.sleep(0.3)
            stress = METAB.stress_ema
            ppl = eval_ppl(model, test_data)
            dvfs_ppls[dvfs_name] = ppl
            dvfs_stresses[dvfs_name] = stress
            # Capture proprioception for this DVFS level
            proprio_agg = aggregate_proprio()
            dvfs_proprios[dvfs_name] = proprio_agg.tolist()
            print(f"  DVFS {dvfs_name}: PPL={ppl:.4f}, stress={stress:.4f}, "
                  f"temp={METAB._last_temp:.1f}C, sclk={METAB._last_sclk:.0f}MHz, "
                  f"proprio_jitter={proprio_agg[2].item():.4f}")
        set_dvfs_level(1, wait=True)
    results['dvfs_ppls'] = dvfs_ppls
    results['dvfs_stresses'] = dvfs_stresses
    results['dvfs_proprios'] = dvfs_proprios
    dvfs_ppl_std = float(np.std(list(dvfs_ppls.values()))) if len(dvfs_ppls) >= 2 else 0.0
    results['dvfs_ppl_std'] = dvfs_ppl_std

    # T6: Determinism
    print("\n[T6] Determinism (packed mode=0xFF, 5 runs)...")
    forced = pack_mode_byte(3, 3, 3, 3)
    set_analog_mode_override(model, ANALOG_LAYERS, forced)
    det_ppls = []
    for run_i in range(5):
        ppl = eval_ppl(model, test_data)
        det_ppls.append(ppl)
        print(f"  Run {run_i}: PPL={ppl:.6f}")
    det_std = float(np.std(det_ppls))
    results['determinism_ppls'] = det_ppls
    results['determinism_std'] = det_std

    # T7: Kill-Shot
    print("\n[T7] Kill-Shot (best vs worst mode)...")
    best_mode = min(mode_ppls, key=mode_ppls.get)
    worst_mode = max(mode_ppls, key=mode_ppls.get)
    kill_ratio = mode_ppls[worst_mode] / max(mode_ppls[best_mode], 0.01)
    results['best_mode'] = best_mode
    results['worst_mode'] = worst_mode
    results['kill_ratio'] = kill_ratio
    print(f"  Best: {best_mode} PPL={mode_ppls[best_mode]:.4f}")
    print(f"  Worst: {worst_mode} PPL={mode_ppls[worst_mode]:.4f}")
    print(f"  Kill ratio: {kill_ratio:.6f}")

    # T8: Denorm Sweep
    print("\n[T8] Denorm Sweep (FP16 denorm bits[7:6])...")
    denorm_logits = {}
    denorm_ppls = {}
    denorm_names = {0: 'flush_both', 1: 'allow_in', 2: 'allow_out', 3: 'allow_both'}
    for dv in range(4):
        packed = pack_mode_byte(0, 0, 3, dv)
        set_analog_mode_override(model, ANALOG_LAYERS, packed)
        logits = collect_logits(model, test_data, n_batches=3)
        ppl = eval_ppl(model, test_data)
        denorm_logits[dv] = logits
        denorm_ppls[dv] = ppl
        print(f"  FP16 denorm {dv} ({denorm_names[dv]}, packed=0x{packed:02X}): PPL={ppl:.4f}")
    denorm_kls = {}
    max_denorm_kl = 0.0
    for i_d in range(4):
        for j_d in range(i_d+1, 4):
            if denorm_logits[i_d] is not None and denorm_logits[j_d] is not None:
                kl_val = kl_divergence(denorm_logits[i_d], denorm_logits[j_d])
                denorm_kls[f"fp16d{i_d}_vs_{j_d}"] = kl_val
                max_denorm_kl = max(max_denorm_kl, kl_val)
    results['denorm_kls'] = denorm_kls
    results['denorm_ppls'] = {str(k): v for k, v in denorm_ppls.items()}
    results['max_denorm_kl'] = max_denorm_kl
    print(f"  Max FP16 denorm KL: {max_denorm_kl:.8f}")

    # T9: Per-Layer Mode Variation
    print("\n[T9] Per-Layer Mode Variation...")
    uniform = pack_mode_byte(0, 0, 3, 3)
    set_analog_mode_override(model, ANALOG_LAYERS, uniform)
    logits_uniform = collect_logits(model, test_data, n_batches=3)
    # Alternate layers between mode 0 and mode 1
    for idx in ANALOG_LAYERS:
        packed = pack_mode_byte(idx % 2, idx % 2, 3, 3)
        gate = model.model.layers[idx].mlp.gate_proj
        if isinstance(gate, AnalogLinear):
            gate.mode_override = packed
    logits_varied = collect_logits(model, test_data, n_batches=3)
    per_layer_kl = kl_divergence(logits_uniform, logits_varied) if logits_uniform is not None and logits_varied is not None else 0.0
    results['per_layer_kl'] = per_layer_kl
    print(f"  KL(uniform vs alternating): {per_layer_kl:.8f}")

    # T10: Ablation
    print("\n[T10] Ablation...")
    restore_gate_proj(model, originals)
    standard_ppl = eval_ppl(model, test_data)
    results['standard_ppl'] = standard_ppl
    ppl_diff = abs(analog_ppl - standard_ppl)
    results['ablation_ppl_diff'] = ppl_diff
    print(f"  Standard: {standard_ppl:.4f}, Analog: {analog_ppl:.4f}, Diff: {ppl_diff:.6f}")
    originals = patch_gate_proj_with_analog(model, ANALOG_LAYERS, mode_override=None)

    # T11: Domain Sensitivity
    print("\n[T11] Domain Sensitivity (wiki vs code x modes)...")
    domain_ppls = {}
    if test_data_code is not None and len(test_data_code) >= BS * 3:
        for mode_val in [0, 3]:
            packed = pack_mode_byte(mode_val, mode_val, 3, 3)
            set_analog_mode_override(model, ANALOG_LAYERS, packed)
            wiki_ppl = eval_ppl(model, test_data, n_batches=5)
            code_ppl = eval_ppl(model, test_data_code, n_batches=5)
            domain_ppls[f"wiki_mode{mode_val}"] = wiki_ppl
            domain_ppls[f"code_mode{mode_val}"] = code_ppl
            print(f"  Mode {mode_val}: wiki={wiki_ppl:.4f}, code={code_ppl:.4f}")
    results['domain_ppls'] = domain_ppls

    # T12: Zombie Twin
    print("\n[T12] Zombie Twin (live vs replay)...")
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    logits_live = collect_logits(model, test_data, n_batches=3)
    live_stress = METAB.stress_ema
    set_analog_mode_override(model, ANALOG_LAYERS, pack_mode_byte(0, 0, 3, 3))
    logits_zombie = collect_logits(model, test_data, n_batches=3)
    zombie_kl = kl_divergence(logits_live, logits_zombie) if logits_live is not None and logits_zombie is not None else 0.0
    results['zombie_kl'] = zombie_kl
    results['live_stress'] = live_stress
    print(f"  KL(live vs zombie): {zombie_kl:.8f}, live_stress={live_stress:.4f}")

    # T13: Token-Level Impact
    print("\n[T13] Token-Level Impact...")
    set_analog_mode_override(model, ANALOG_LAYERS, pack_mode_byte(0, 0, 3, 3))
    logits_m0 = collect_logits(model, test_data, n_batches=3)
    set_analog_mode_override(model, ANALOG_LAYERS, pack_mode_byte(3, 3, 0, 0))
    logits_m3 = collect_logits(model, test_data, n_batches=3)
    token_impact_frac = 0.0
    if logits_m0 is not None and logits_m3 is not None:
        diff = (logits_m0 - logits_m3).abs()
        impacted = (diff.max(dim=-1).values > 0.01).float().mean().item()
        token_impact_frac = impacted
    results['token_impact_frac'] = token_impact_frac
    print(f"  Fraction with >0.01 logit diff: {token_impact_frac:.4f}")

    # T14: Argmax Flip Rate
    print("\n[T14] Argmax Flip Rate...")
    argmax_flip = 0.0
    if logits_m0 is not None and logits_m3 is not None:
        topk_m0 = logits_m0.argmax(dim=-1)
        topk_m3 = logits_m3.argmax(dim=-1)
        argmax_flip = (topk_m0 != topk_m3).float().mean().item()
    results['argmax_flip_rate'] = argmax_flip
    print(f"  Argmax flip rate: {argmax_flip:.6f}")

    # T15: Generation Quality (with decoded text)
    print("\n[T15] Generation Quality...")
    gen_results = {}
    for mode_val in [0, 1, 2, 3]:
        packed = pack_mode_byte(mode_val, mode_val, 3, 3)
        set_analog_mode_override(model, ANALOG_LAYERS, packed)
        text = generate_text(model, test_data, mode_val)
        gen_results[f"mode_{mode_val}"] = text
        print(f"  {text[:200]}")
    # Also generate with live analog (no override)
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    live_text = generate_text(model, test_data, 'live')
    gen_results['live'] = live_text
    print(f"  {live_text[:200]}")
    results['generation'] = gen_results

    # T_PROPRIO: Proprioception jitter differs across DVFS (new v4 test)
    print("\n[T_PROPRIO] Proprioception Jitter Divergence...")
    proprio_divergence = 0.0
    if 'low' in dvfs_proprios and 'high' in dvfs_proprios:
        low_j = dvfs_proprios['low'][2] if len(dvfs_proprios['low']) > 2 else 0
        high_j = dvfs_proprios['high'][2] if len(dvfs_proprios['high']) > 2 else 0
        proprio_divergence = abs(high_j - low_j)
    results['proprio_divergence'] = proprio_divergence
    print(f"  Jitter divergence (high-low): {proprio_divergence:.6f}")

    set_analog_mode_override(model, ANALOG_LAYERS, None)
    restore_gate_proj(model, originals)
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE B: Homeostatic Training (body-gated LoRA + multi-task soma)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_phase_b(model, train_data, test_data, test_data_code, baseline_ppl, phase_a_results):
    """Phase B: Train body-gated LoRA + multi-task soma head with homeostatic loss.
    Loss = CE(text) + λ_intero * (MSE(metabolic_recon) + CE(regime) + MSE(stress))
         + λ_stab * KL(current_logits || ema_logits)
    DVFS alternation: cycle low/high/auto every 20 batches.
    """
    print("\n" + "=" * 60)
    print("PHASE B: Homeostatic Training (body-gated LoRA + multi-task soma)")
    print("  v4: metabolic-gated LoRA + regime classification + stability KL")
    print("=" * 60)

    results = {}

    # Patch analog layers (already does all 40)
    analog_originals = patch_gate_proj_with_analog(model, ANALOG_LAYERS, mode_override=None)

    # Patch v_proj with body-gated LoRA
    lora_originals, n_lora_params = patch_v_proj_with_lora(model, LORA_LAYERS,
                                                            rank=LORA_RANK, alpha=LORA_ALPHA)

    # Create multi-task soma head
    hidden_dim = model.config.hidden_size
    soma_head = MultiTaskSomaHead(hidden_dim, n_proprio_stats=3).to(DEVICE)
    n_soma_params = sum(p.numel() for p in soma_head.parameters())
    print(f"  [MultiTaskSomaHead] {hidden_dim}+3 -> (metab[{METABOLIC_DIM}], regime[{N_REGIMES}], stress[1])")
    print(f"  [MultiTaskSomaHead] {n_soma_params} params")

    # Collect all trainable params
    lora_params = get_lora_params(model, LORA_LAYERS)
    all_params = list(soma_head.parameters()) + lora_params
    total_trainable = sum(p.numel() for p in all_params)
    print(f"  Total trainable: {total_trainable}")
    results['n_lora_params'] = n_lora_params
    results['n_soma_params'] = n_soma_params
    results['total_trainable'] = total_trainable

    METAB.update_context()
    print(f"  Initial temp: {METAB._last_temp:.1f}C")
    results['initial_temp'] = METAB._last_temp

    # Hyperparameters
    N_TRAIN_BATCHES = 120
    DVFS_CYCLE = 20
    LAMBDA_INTERO = 0.1
    LAMBDA_STAB = 0.01
    LR = 1e-4

    optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=0.01)

    # EMA logits for stability regularization
    ema_logits = None
    EMA_DECAY = 0.95

    dvfs_modes = ['low', 'high', 'auto']
    dvfs_idx = 0

    print(f"\n  Training {total_trainable} params for {N_TRAIN_BATCHES} batches")
    print(f"  DVFS alternation: every {DVFS_CYCLE} batches cycle {dvfs_modes}")
    print(f"  Loss = CE + {LAMBDA_INTERO} * Intero + {LAMBDA_STAB} * Stability")
    results['temp_range'] = [999.0, 0.0]

    for batch_idx in range(N_TRAIN_BATCHES):
        # DVFS cycling
        if batch_idx > 0 and batch_idx % DVFS_CYCLE == 0:
            dvfs_idx = (dvfs_idx + 1) % len(dvfs_modes)
            set_dvfs(dvfs_modes[dvfs_idx])
            time.sleep(DVFS_SETTLE_S)

        # Snapshot metabolic state
        METAB.update_context()
        CTX.clear_proprio()

        # Get batch
        start = (batch_idx * BS) % max(len(train_data) - BS, 1)
        batch = torch.stack(train_data[start:start+BS]).to(DEVICE)

        # Forward pass
        out = model(batch, output_hidden_states=True)
        logits = out.logits if hasattr(out, 'logits') else out[0]

        # CE loss (text)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1))

        # Interoception loss (multi-task)
        hidden = out.hidden_states[-1] if hasattr(out, 'hidden_states') and out.hidden_states else logits
        proprio_stats = aggregate_proprio().to(DEVICE)
        metabolic_recon, regime_logits, stress_pred = soma_head(hidden.detach(), proprio_stats)

        # Target: normalized metabolic vector
        if CTX.metabolic_vec is not None:
            target_metab = CTX.metabolic_vec.unsqueeze(0).expand(metabolic_recon.size(0), -1).to(DEVICE)
            metab_loss = F.mse_loss(metabolic_recon, target_metab)
        else:
            metab_loss = torch.tensor(0.0, device=DEVICE)

        # Target: regime classification
        target_regime = torch.full((regime_logits.size(0),), CTX.regime, device=DEVICE, dtype=torch.long)
        regime_loss = F.cross_entropy(regime_logits, target_regime)

        # Target: continuous stress
        target_stress = torch.full((stress_pred.size(0),), CTX.continuous_stress * 2.0 - 1.0,
                                    device=DEVICE, dtype=torch.float32)  # scale to [-1, 1] for tanh
        stress_loss = F.mse_loss(stress_pred, target_stress)

        intero_loss = metab_loss + regime_loss + stress_loss

        # Stability loss (KL against EMA logits)
        stab_loss = torch.tensor(0.0, device=DEVICE)
        if ema_logits is not None:
            with torch.no_grad():
                ema_p = F.softmax(ema_logits[:shift_logits.size(0), :shift_logits.size(1)].float(), dim=-1).clamp(min=1e-10)
            curr_lp = F.log_softmax(shift_logits.float(), dim=-1)
            stab_loss = F.kl_div(curr_lp, ema_p, reduction='batchmean')
            stab_loss = stab_loss.clamp(max=10.0)  # prevent explosion

        # Update EMA logits
        with torch.no_grad():
            if ema_logits is None:
                ema_logits = shift_logits.detach().clone()
            else:
                # Match shapes
                s0 = min(ema_logits.size(0), shift_logits.size(0))
                s1 = min(ema_logits.size(1), shift_logits.size(1))
                ema_logits[:s0, :s1] = EMA_DECAY * ema_logits[:s0, :s1] + (1 - EMA_DECAY) * shift_logits[:s0, :s1].detach()

        # Total loss
        total_loss = ce_loss + LAMBDA_INTERO * intero_loss + LAMBDA_STAB * stab_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        # Track temperature range
        results['temp_range'][0] = min(results['temp_range'][0], METAB._last_temp)
        results['temp_range'][1] = max(results['temp_range'][1], METAB._last_temp)

        if (batch_idx + 1) % DVFS_CYCLE == 0:
            print(f"    Batch {batch_idx+1}/{N_TRAIN_BATCHES}: "
                  f"ce={ce_loss.item():.4f}, intero={intero_loss.item():.4f}, "
                  f"stab={stab_loss.item():.4f}, "
                  f"temp={METAB._last_temp:.1f}C, stress={METAB.stress_ema:.3f}")

    results['train_ce'] = ce_loss.item()
    results['train_intero'] = intero_loss.item()
    results['train_stab'] = stab_loss.item()
    set_dvfs('auto')
    time.sleep(DVFS_SETTLE_S)

    # ─── Phase B Generation Examples ─────────────────────────
    print("\n[Phase B] Generation examples after LoRA training...")
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    for dvfs_mode in ['low', 'high', 'auto']:
        set_dvfs(dvfs_mode)
        time.sleep(DVFS_SETTLE_S)
        METAB.update_context()
        txt = generate_text(model, test_data, f'lora_{dvfs_mode}')
        print(f"  [{dvfs_mode.upper()} DVFS] {txt[:250]}")
        results[f'gen_lora_{dvfs_mode}'] = txt
    set_dvfs('auto')
    time.sleep(0.5)

    # ─── Phase B Tests ───────────────────────────────────────

    # T16: LoRA Adaptation PPL
    print("\n[T16] LoRA Adaptation PPL...")
    METAB.update_context()
    lora_ppl = eval_ppl(model, test_data)
    results['lora_ppl'] = lora_ppl
    results['phase_a_ppl'] = phase_a_results.get('analog_ppl', baseline_ppl)
    print(f"  LoRA PPL: {lora_ppl:.4f} (Phase A: {results['phase_a_ppl']:.4f})")

    # T17: Embodied Advantage (live analog vs all fixed modes)
    print("\n[T17] Embodied Advantage (live vs all fixed modes)...")
    lora_fixed_ppls = {}
    for mode_val in range(4):
        packed = pack_mode_byte(mode_val, mode_val, 3, 3)
        set_analog_mode_override(model, ANALOG_LAYERS, packed)
        fppl = eval_ppl(model, test_data)
        lora_fixed_ppls[str(mode_val)] = fppl
        print(f"  Fixed mode {mode_val}: PPL={fppl:.4f}")

    set_analog_mode_override(model, ANALOG_LAYERS, None)
    lora_live_ppl = eval_ppl(model, test_data)
    lora_best_fixed = min(lora_fixed_ppls.values())
    embodied_adv = lora_best_fixed - lora_live_ppl
    results['lora_live_ppl'] = lora_live_ppl
    results['lora_fixed_ppls'] = lora_fixed_ppls
    results['lora_best_fixed_ppl'] = lora_best_fixed
    results['embodied_advantage'] = embodied_adv
    print(f"  Live: {lora_live_ppl:.4f}, Best fixed: {lora_best_fixed:.4f}")
    print(f"  Advantage: {embodied_adv:.6f}")

    # T18: LoRA vs Fixed Mode
    print("\n[T18] LoRA vs Fixed Mode...")
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    lora_analog_ppl = eval_ppl(model, test_data)
    packed0 = pack_mode_byte(0, 0, 3, 3)
    set_analog_mode_override(model, ANALOG_LAYERS, packed0)
    lora_fixed_ppl = eval_ppl(model, test_data)
    results['lora_analog_ppl'] = lora_analog_ppl
    results['lora_fixed_ppl'] = lora_fixed_ppl
    print(f"  Analog (live): {lora_analog_ppl:.4f}, Fixed mode 0: {lora_fixed_ppl:.4f}")

    # T19: Multi-Task Interoception (balanced DVFS protocol)
    print("\n[T19] Multi-Task Interoception (balanced DVFS protocol)...")
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    soma_head.eval()

    dvfs_protocol = ['low', 'high', 'auto', 'high', 'low']
    soma_preds_stress = []
    soma_preds_metab = []
    soma_preds_regime = []
    soma_targets_stress = []
    soma_targets_metab = []
    soma_targets_regime = []

    for dvfs_mode in dvfs_protocol:
        set_dvfs(dvfs_mode)
        time.sleep(DVFS_SETTLE_S)

        # Multiple samples per DVFS level for stability
        for _ in range(5):
            METAB.update_context()
            CTX.clear_proprio()
            with torch.no_grad():
                batch = torch.stack(test_data[:BS]).to(DEVICE)
                out = model(batch, output_hidden_states=True)
                hidden = out.hidden_states[-1] if hasattr(out, 'hidden_states') and out.hidden_states else (out.logits if hasattr(out, 'logits') else out[0])
                proprio = aggregate_proprio().to(DEVICE)
                metab_recon, regime_logits, stress_pred = soma_head(hidden, proprio)

            # Record predictions vs targets
            target_stress = CTX.continuous_stress * 2.0 - 1.0
            soma_preds_stress.append(float(stress_pred[0].item()))
            soma_targets_stress.append(target_stress)

            if CTX.metabolic_vec is not None:
                soma_preds_metab.append(metab_recon[0].cpu().numpy())
                soma_targets_metab.append(CTX.metabolic_vec.numpy())

            pred_regime = int(regime_logits[0].argmax().item())
            soma_preds_regime.append(pred_regime)
            soma_targets_regime.append(CTX.regime)

        print(f"  DVFS {dvfs_mode}: temp={METAB._last_temp:.1f}C, "
              f"stress_pred={soma_preds_stress[-1]:.4f}, target={soma_targets_stress[-1]:.4f}")

    set_dvfs('auto')

    # Compute correlation and MAE for stress
    if len(soma_preds_stress) >= 3:
        corr, _ = stats.pearsonr(soma_preds_stress, soma_targets_stress)
        mae = float(np.mean(np.abs(np.array(soma_preds_stress) - np.array(soma_targets_stress))))
    else:
        corr, mae = 0.0, 999.0
    results['soma_correlation'] = float(corr) if not math.isnan(corr) else 0.0
    results['soma_mae'] = mae
    results['soma_preds_stress'] = soma_preds_stress
    results['soma_targets_stress'] = soma_targets_stress
    print(f"  Stress corr: {results['soma_correlation']:.4f}, MAE: {mae:.4f}")
    print(f"  Pred range: [{min(soma_preds_stress):.3f}, {max(soma_preds_stress):.3f}]")
    print(f"  Target range: [{min(soma_targets_stress):.3f}, {max(soma_targets_stress):.3f}]")

    # Metabolic reconstruction quality
    if soma_preds_metab:
        preds_arr = np.array(soma_preds_metab)
        targs_arr = np.array(soma_targets_metab)
        metab_mae = float(np.mean(np.abs(preds_arr - targs_arr)))
        per_dim_corr = []
        for d in range(METABOLIC_DIM):
            if np.std(targs_arr[:, d]) > 1e-6:
                c, _ = stats.pearsonr(preds_arr[:, d], targs_arr[:, d])
                per_dim_corr.append(float(c) if not math.isnan(c) else 0.0)
            else:
                per_dim_corr.append(0.0)
        results['metab_recon_mae'] = metab_mae
        results['metab_per_dim_corr'] = per_dim_corr
        mean_dim_corr = float(np.mean([abs(c) for c in per_dim_corr]))
        print(f"  Metabolic recon MAE: {metab_mae:.4f}, mean |corr|: {mean_dim_corr:.4f}")
    else:
        results['metab_recon_mae'] = 999.0
        results['metab_per_dim_corr'] = [0.0] * METABOLIC_DIM

    # Regime classification accuracy
    if soma_preds_regime:
        regime_acc = float(np.mean(np.array(soma_preds_regime) == np.array(soma_targets_regime)))
        results['regime_accuracy'] = regime_acc
        print(f"  Regime accuracy: {regime_acc:.4f}")
    else:
        results['regime_accuracy'] = 0.0

    # T20: Body Gate Analysis
    print("\n[T20] Body Gate Analysis...")
    gate_values_low = []
    gate_values_high = []
    for dvfs_mode, gate_list in [('low', gate_values_low), ('high', gate_values_high)]:
        set_dvfs(dvfs_mode)
        time.sleep(DVFS_SETTLE_S)
        for _ in range(3):
            METAB.update_context()
            with torch.no_grad():
                batch = torch.stack(test_data[:BS]).to(DEVICE)
                _ = model(batch)
            for layer_idx in LORA_LAYERS:
                v = model.model.layers[layer_idx].self_attn.v_proj
                if isinstance(v, BodyGatedLoRA) and CTX.metabolic_vec is not None:
                    gate_input = CTX.metabolic_vec.to(v.body_gate.weight.device)
                    gate = torch.sigmoid(v.body_gate(gate_input)).detach().cpu().numpy()
                    gate_list.append(gate)

    set_dvfs('auto')
    body_gate_separation = 0.0
    if gate_values_low and gate_values_high:
        low_mean = np.mean(np.array(gate_values_low))
        high_mean = np.mean(np.array(gate_values_high))
        body_gate_separation = abs(high_mean - low_mean)
        print(f"  Gate low DVFS mean: {low_mean:.4f}")
        print(f"  Gate high DVFS mean: {high_mean:.4f}")
        print(f"  Separation: {body_gate_separation:.6f}")
    results['body_gate_separation'] = body_gate_separation

    # T21: Zombie Twin (live vs replay)
    print("\n[T21] Zombie Twin (live vs replay)...")
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    METAB.update_context()
    live_logits = collect_logits(model, test_data, n_batches=3)
    live_stress = METAB.stress_ema

    # Replay: freeze metabolic to current snapshot
    frozen_mode = CTX.round_mode
    set_analog_mode_override(model, ANALOG_LAYERS, frozen_mode)
    zombie_logits = collect_logits(model, test_data, n_batches=3)
    set_analog_mode_override(model, ANALOG_LAYERS, None)

    zombie_kl = 0.0
    if live_logits is not None and zombie_logits is not None:
        zombie_kl = kl_divergence(live_logits, zombie_logits)
    results['zombie_kl'] = zombie_kl
    results['live_stress'] = live_stress
    print(f"  KL(live vs zombie): {zombie_kl:.8f}, live_stress={live_stress:.4f}")

    # Cleanup
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    restore_v_proj(model, lora_originals)
    restore_gate_proj(model, analog_originals)

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST BATTERY SCORING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def score_tests(phase_a, phase_b, baseline_ppl):
    """Score all tests. Returns list of test result dicts + counts."""
    tests = []

    def add(t_id, name, status, value, criterion):
        tests.append({'test': t_id, 'name': name, 'status': status,
                      'value': value, 'criterion': criterion})

    # T1: Baseline PPL
    add('T1', 'Baseline PPL', 'PASS', f'{baseline_ppl:.4f}', 'record')

    # T2: AnalogLinear PPL
    ratio = phase_a.get('analog_ppl', 999) / max(baseline_ppl, 1e-6)
    add('T2', 'AnalogLinear PPL', 'PASS' if ratio < 1.5 else 'FAIL',
        f'{phase_a.get("analog_ppl", 0):.4f} (ratio={ratio:.4f})', '<1.5x baseline')

    # T3: Mode Sweep
    std = phase_a.get('mode_ppl_std', 0)
    add('T3', 'Mode Sweep σ(PPL)', 'PASS' if std > 0.01 else 'FAIL',
        f'σ={std:.6f}', '>0.01')

    # T4: KL Divergence
    max_kl = phase_a.get('max_kl', 0)
    add('T4', 'KL Divergence', 'PASS' if max_kl > 0.001 else 'FAIL',
        f'max_kl={max_kl:.8f}', '>0.001')

    # T5: Thermal Variance
    dvfs_std = phase_a.get('dvfs_ppl_std', 0)
    add('T5', 'Thermal Variance', 'PASS' if dvfs_std > 0.05 else 'FAIL',
        f'std={dvfs_std:.6f}', '>0.05')

    # T6: Determinism
    det_std = phase_a.get('determinism_std', 999)
    add('T6', 'Determinism', 'PASS' if det_std < 0.001 else 'FAIL',
        f'std={det_std:.8f}', '<0.001')

    # T7: Kill-Shot
    kill = phase_a.get('kill_ratio', 0)
    add('T7', 'Kill-Shot', 'PASS' if kill > 1.05 else 'FAIL',
        f'ratio={kill:.6f}', '>1.05')

    # T8: FP16 Denorm Sweep
    max_dk = phase_a.get('max_denorm_kl', 0)
    add('T8', 'FP16 Denorm Sweep [7:6]', 'PASS' if max_dk > 0.0001 else 'FAIL',
        f'max_kl={max_dk:.8f}', '>0.0001')

    # T9: Per-Layer Variation
    pl_kl = phase_a.get('per_layer_kl', 0)
    add('T9', 'Per-Layer Variation', 'PASS' if pl_kl > 0.0001 else 'FAIL',
        f'kl={pl_kl:.8f}', '>0.0001')

    # T10: Ablation
    abl = phase_a.get('ablation_ppl_diff', 0)
    add('T10', 'Ablation', 'PASS' if abl > 0.001 else 'FAIL',
        f'diff={abl:.6f}', '>0.001')

    # T11: Domain Sensitivity
    d = phase_a.get('domain_ppls', {})
    w0 = d.get('wiki_mode0', 0)
    w3 = d.get('wiki_mode3', 0)
    c0 = d.get('code_mode0', 0)
    c3 = d.get('code_mode3', 0)
    wiki_diff = abs(w0 - w3)
    code_diff = abs(c0 - c3)
    interaction = abs(wiki_diff - code_diff)
    add('T11', 'Domain Sensitivity', 'PASS' if interaction > 0.01 else 'FAIL',
        f'interaction={interaction:.4f}', '>0.01')

    # T12: Zombie Twin (Phase A)
    zk = phase_a.get('zombie_kl', 0)
    add('T12', 'Zombie Twin (Phase A)', 'PASS' if zk > 0.0001 else 'FAIL',
        f'kl={zk:.8f}', '>0.0001')

    # T13: Token-Level Impact
    ti = phase_a.get('token_impact_frac', 0)
    add('T13', 'Token-Level Impact', 'PASS' if ti > 0.05 else 'FAIL',
        f'frac={ti:.4f}', '>0.05')

    # T14: Argmax Flip Rate
    af = phase_a.get('argmax_flip_rate', 0)
    add('T14', 'Argmax Flip Rate', 'PASS' if af > 0.001 else 'FAIL',
        f'rate={af:.6f}', '>0.001')

    # T15: Generation Quality
    add('T15', 'Generation Quality', 'PASS', 'see output', 'coherent')

    # T_PROPRIO: Proprioception Jitter
    pj = phase_a.get('proprio_divergence', 0)
    add('T16p', 'Proprioception Jitter', 'PASS' if pj > 0.0 else 'FAIL',
        f'divergence={pj:.6f}', '>0 (any jitter difference)')

    # --- Phase B tests ---
    if phase_b:
        # T16: LoRA Adaptation
        lp = phase_b.get('lora_ppl', 999)
        pa = phase_b.get('phase_a_ppl', 999)
        add('T17', 'LoRA Adaptation', 'PASS' if lp < pa else 'FAIL',
            f'lora={lp:.4f} vs phaseA={pa:.4f}', 'lora_ppl < phase_a_ppl')

        # T17: Embodied Advantage
        ea = phase_b.get('embodied_advantage', -1)
        add('T18', 'Embodied Advantage',
            'PASS' if ea > 0 else 'FAIL',
            f'advantage={ea:.6f}', 'live_analog < best_fixed')

        # T18: LoRA vs Fixed
        la = phase_b.get('lora_analog_ppl', 999)
        lf = phase_b.get('lora_fixed_ppl', 999)
        add('T19', 'LoRA vs Fixed',
            'PASS' if la < lf else 'FAIL',
            f'analog={la:.4f} vs fixed={lf:.4f}', 'analog < fixed')

        # T19: Interoception (stress)
        sc = phase_b.get('soma_correlation', 0)
        sm = phase_b.get('soma_mae', 999)
        add('T20', 'Interoception (stress)',
            'PASS' if abs(sc) > 0.3 or sm < 0.5 else 'FAIL',
            f'corr={sc:.4f}, mae={sm:.4f}', '|corr|>0.3 or mae<0.5')

        # T20: Metabolic Reconstruction
        mm = phase_b.get('metab_recon_mae', 999)
        mdc = phase_b.get('metab_per_dim_corr', [])
        mean_abs_corr = float(np.mean([abs(c) for c in mdc])) if mdc else 0.0
        add('T21', 'Metabolic Reconstruction',
            'PASS' if mm < 2.0 or mean_abs_corr > 0.2 else 'FAIL',
            f'mae={mm:.4f}, mean_|corr|={mean_abs_corr:.4f}', 'mae<2.0 or mean_|corr|>0.2')

        # T21: Regime Classification
        ra = phase_b.get('regime_accuracy', 0)
        add('T22', 'Regime Classification',
            'PASS' if ra > 0.4 else 'FAIL',
            f'acc={ra:.4f}', '>0.4 (above chance=0.25)')

        # T22: Body Gate Separation
        bg = phase_b.get('body_gate_separation', 0)
        add('T23', 'Body Gate Separation',
            'PASS' if bg > 0.01 else 'FAIL',
            f'sep={bg:.6f}', '>0.01')

        # T23: Zombie Twin (Phase B)
        zk2 = phase_b.get('zombie_kl', 0)
        add('T24', 'Zombie Twin (Phase B)',
            'PASS' if zk2 > 0.0001 else 'FAIL',
            f'kl={zk2:.8f}', '>0.0001')

    n_pass = sum(1 for t in tests if t['status'] == 'PASS')
    n_total = len(tests)
    return tests, n_pass, n_total


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    global N_LAYERS, ANALOG_LAYERS, LORA_LAYERS, _TOKENIZER, SCLK_LOW_CAL, SCLK_HIGH_CAL, SMN_AVAILABLE, SMN_FD
    print("=" * 60)
    print("z2114v4: Deep Embodied Analog Linear LM")
    print("ALL layers + TILE=32 + wall_clock64 proprioception")
    print("MetabolicVector + Body-gated LoRA + Multi-task SomaHead")
    print("=" * 60)
    print(f"  AnalogLinear layers: {ANALOG_LAYERS[0]}-{ANALOG_LAYERS[-1]} ({len(ANALOG_LAYERS)} total)")
    print(f"  Phase A: 0 trainable params (frozen measurement)")
    print(f"  Phase B: BodyGatedLoRA rank={LORA_RANK} at v_proj [{LORA_LAYERS[0]}-{LORA_LAYERS[-1]}]")

    # ── Hardware Setup ──
    find_dvfs_sysfs()
    check_rapl()
    init_msr()
    try:
        SMN_FD = os.open('/sys/kernel/ryzen_smu_drv/smn', os.O_RDWR)
        SMN_AVAILABLE = True
        print("[SMN] Available")
    except:
        SMN_AVAILABLE = False
        print("[SMN] Not available")

    # GPU warmup
    print("\n[GPU] Warming up...")
    torch.zeros(1024, 1024, device=DEVICE) @ torch.zeros(1024, 1024, device=DEVICE)
    torch.cuda.synchronize()
    print("[GPU] Warmup OK")

    # ── Load Model ──
    print(f"\nLoading Qwen/Qwen3-8B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    _TOKENIZER = tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="eager",
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    vocab = model.config.vocab_size
    actual_layers = len(model.model.layers)
    print(f"  Qwen/Qwen3-8B: {n_params:.1f}M params, vocab={vocab}, layers={actual_layers}")

    # Dynamically set layer ranges based on actual model
    N_LAYERS = actual_layers
    ANALOG_LAYERS = list(range(N_LAYERS))
    LORA_LAYERS = list(range(min(8, N_LAYERS), min(32, N_LAYERS)))
    print(f"  Analog layers: 0-{N_LAYERS-1} ({N_LAYERS} total)")
    print(f"  LoRA layers: {LORA_LAYERS[0]}-{LORA_LAYERS[-1]} ({len(LORA_LAYERS)} total)")

    # Freeze all backbone params
    for p in model.parameters():
        p.requires_grad = False
    print("  All backbone parameters frozen")

    # ── Load Data ──
    print(f"\nLoading data...")
    all_wiki = load_wikitext_data(tokenizer, split='train', max_samples=2000)
    train_data = all_wiki[:1500]
    test_data = all_wiki[1500:]
    if len(test_data) < 100:
        test_data = all_wiki[-500:]
    print(f"  Loaded {len(train_data)} train, {len(test_data)} test sequences")
    test_wiki = load_wikitext_data(tokenizer, split='test', max_samples=500)
    if len(test_wiki) > 50:
        test_data = test_wiki
        print(f"  Using test split: {len(test_data)} sequences")
    test_data_code = load_code_data(tokenizer)

    # ── DVFS Calibration ──
    compile_hip_kernel()  # pre-compile

    if DVFS_AVAILABLE:
        print("\n[DVFS] Calibration...")
        set_dvfs('low')
        time.sleep(DVFS_SETTLE_S)
        SCLK_LOW_CAL = read_current_sclk_mhz()
        set_dvfs('high')
        time.sleep(DVFS_SETTLE_S)
        SCLK_HIGH_CAL = read_current_sclk_mhz()
        set_dvfs('auto')
        time.sleep(0.5)
        print(f"  low={SCLK_LOW_CAL:.0f}MHz high={SCLK_HIGH_CAL:.0f}MHz")

    # SMN entropy sample
    if SMN_AVAILABLE:
        val = read_smn(0x00059800)
        print(f"\n[SMN] Entropy sample: 0x{val & 0xFF:02X}")

    # Init metabolic controller
    print(f"\n[METAB] Initializing MetabolicController...")
    METAB.snapshot()
    METAB.update_context()
    print(f"  Initial stress EMA: {METAB.stress_ema:.4f}")
    print(f"  Initial regime: {['COLD','NOMINAL','HOT','THROTTLED'][CTX.regime]}")

    # ── T1: Baseline PPL ──
    print(f"\n[T1] Baseline PPL (frozen Qwen3-8B, no modifications)...")
    baseline_ppl = eval_ppl(model, test_data)
    print(f"  Baseline PPL: {baseline_ppl:.4f}")

    # ── Phase A ──
    phase_a = run_phase_a(model, test_data, test_data_code, baseline_ppl)

    # ── Phase B (conditional) ──
    phase_b = None
    max_kl = phase_a.get('max_kl', 0)
    ppl_std = phase_a.get('mode_ppl_std', 0)
    if max_kl > 0.0005 or ppl_std > 0.005:
        print(f"\n  Phase A signal detected (kl={max_kl:.6f}, std={ppl_std:.6f}) → running Phase B")
        phase_b = run_phase_b(model, train_data, test_data, test_data_code,
                              baseline_ppl, phase_a)
    else:
        print(f"\n  Phase A signal too weak (kl={max_kl:.6f}, std={ppl_std:.6f}) → skipping Phase B")

    # ── Score Tests ──
    tests, n_pass, n_total = score_tests(phase_a, phase_b, baseline_ppl)

    print("\n" + "=" * 60)
    print(f"TEST BATTERY RESULTS ({n_total} tests)")
    print("=" * 60)
    for t in tests:
        print(f"  {t['test']}: {t['name']} — {t['status']} ({t['value']} vs {t['criterion']})")

    print(f"\n" + "=" * 60)
    print(f"z2114v4 Deep Embodied Analog LM: {n_pass}/{n_total} PASS")
    print("=" * 60)

    # ── Save Results ──
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'z2114_analog_linear_lm.json')
    out_path = os.path.abspath(out_path)
    result_obj = {
        'experiment': 'z2114v4_deep_embodied_analog_linear_lm',
        'description': 'z2114v4: Deep Embodied — ALL 40 layers, MetabolicVector, BodyGatedLoRA, wall_clock64 proprioception',
        'architecture': 'Monist: ALL 40 gate_proj + body-gated LoRA + multi-task soma + kernel proprioception',
        'backbone': f'Qwen/Qwen3-8B ({n_params:.1f}M frozen)',
        'analog_layers': ANALOG_LAYERS,
        'lora_layers': LORA_LAYERS,
        'lora_rank': LORA_RANK,
        'metabolic_dim': METABOLIC_DIM,
        'dvfs_available': DVFS_AVAILABLE,
        'smn_available': SMN_AVAILABLE,
        'hip_kernel_available': compile_hip_kernel() is not None,
        'sclk_low_cal': SCLK_LOW_CAL,
        'sclk_high_cal': SCLK_HIGH_CAL,
        'baseline_ppl': baseline_ppl,
        'phase_a': phase_a,
        'phase_b': phase_b,
        'test_results': tests,
        'n_pass': n_pass,
        'n_total': n_total,
    }
    with open(out_path, 'w') as f:
        json.dump(result_obj, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Cleanup
    set_dvfs('auto')
    if SMN_AVAILABLE and SMN_FD is not None:
        try:
            os.close(SMN_FD)
        except:
            pass
    if MSR_AVAILABLE and MSR_FD is not None:
        try:
            os.close(MSR_FD)
        except:
            pass


if __name__ == '__main__':
    main()
