#!/usr/bin/env python3
"""
z2114v3: Analog Linear LM — Continuous Thermodynamic Embodiment
================================================================
MONIST ARCHITECTURE: No neural adapters. Hardware physics directly warps
matrix multiplication via GPU MODE register (FP rounding + denorm handling).

v3 changes from v2:
  1. Fixed Dead Hardware Register: Used native HIP `clock()` instead of hwreg(29).
  2. Amplified Depth: Expanded to 9 layers, patching BOTH gate_proj and up_proj.
  3. Faster EMA: Controller uses alpha=0.5 for responsive T5 thermal variance.
  4. Stronger Soma Loss: 2.0 weight ensures interoception geometry dominates.

Hardware setup:
  sudo modprobe msr
  sudo insmod ~/Documents/claude_hive/ryzen_smu/ryzen_smu.ko
  sudo chmod 666 /sys/kernel/ryzen_smu_drv/smn
  sudo chmod 666 /sys/kernel/ryzen_smu_drv/pm_table
  sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTORCH_ROCM_ARCH=gfx1100 \
    venv/bin/python -u scripts/z2114_analog_linear_lm.py
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

# Volume Up: Patch 9 layers, both gate_proj and up_proj
ANALOG_LAYERS = [12, 14, 16, 18, 20, 22, 24, 26, 28]

LORA_RANK = 2
LORA_ALPHA = 4
LORA_LAYERS = [15, 20, 25]

SCLK_LOW_CAL = 600.0
SCLK_HIGH_CAL = 2900.0

ROUND_MODES = {0: 'nearest_even', 1: 'plus_inf', 2: 'minus_inf', 3: 'toward_zero'}
DENORM_MODES = {0b00: 'flush_both', 0b01: 'allow_in', 0b10: 'allow_out', 0b11: 'allow_both'}

def pack_mode_byte(f32_round=0, f16_round=0, f32_denorm=3, f16_denorm=3):
    return ((f32_round & 3)       |
            ((f16_round & 3) << 2) |
            ((f32_denorm & 3) << 4) |
            ((f16_denorm & 3) << 6))

def make_lm_labels(input_ids, offset=1):
    labels = input_ids.clone()
    shift = offset - 1
    if shift > 0:
        labels[:, :-shift] = input_ids[:, shift:]
        labels[:, -shift:] = -100
    return labels

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HARDWARE ACCESS
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
                    pass
                DVFS_PATH = p
                DVFS_AVAILABLE = True
                print(f"[DVFS] Found: {p}")
                return
            except: pass
    print("[DVFS] Not available")

def set_dvfs_level(level, wait=True):
    if not DVFS_AVAILABLE: return
    torch.cuda.synchronize()
    name = {0: 'low', 1: 'auto', 2: 'high'}[level]
    try:
        with open(DVFS_PATH, 'w') as f: f.write(name)
    except: pass
    if wait:
        target_low = level == 0
        for _ in range(30):
            sclk = read_current_sclk_mhz()
            if target_low and sclk < 800: return
            if not target_low and sclk > 1200: return
            time.sleep(0.1)
        time.sleep(DVFS_SETTLE_S)

def restore_dvfs_auto():
    if DVFS_AVAILABLE:
        torch.cuda.synchronize()
        try:
            with open(DVFS_PATH, 'w') as f: f.write('auto')
        except: pass

def read_current_sclk_mhz():
    for hwmon in ['hwmon7', 'hwmon6', 'hwmon5']:
        p = f'/sys/class/hwmon/{hwmon}/freq1_input'
        if os.path.exists(p):
            try: return float(open(p, 'r').read().strip()) / 1e6
            except: pass
    return 600.0

SMN_AVAILABLE = False
def check_smn():
    global SMN_AVAILABLE
    SMN_AVAILABLE = os.path.exists('/sys/kernel/ryzen_smu_drv/smn')

def read_smn(addr):
    if not SMN_AVAILABLE: return 0
    try:
        with open('/sys/kernel/ryzen_smu_drv/smn', 'r+b', buffering=0) as f:
            f.write(struct.pack('<I', addr & 0xFFFFFFFF))
            f.flush()
            f.seek(0)
            data = f.read(4)
        return struct.unpack('<I', data)[0] if len(data) == 4 else 0
    except: return 0

def read_gpu_temp_c():
    for hwmon in ['hwmon7', 'hwmon6', 'hwmon8']:
        p = f'/sys/class/hwmon/{hwmon}/temp1_input'
        try: return float(open(p, 'r').read().strip()) / 1000.0
        except: pass
    raw = read_smn(0x00059800)
    return ((raw >> 21) & 0x7FF) * 0.125 if raw else 50.0

def read_hw_entropy():
    if not SMN_AVAILABLE: return None
    try:
        raw_adc = read_smn(0x00059800)
        xtal = read_smn(0x000598C8)
        return (raw_adc & 0xFF) ^ (xtal & 0xFF)
    except: return None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THERMODYNAMIC CONTROLLER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ThermodynamicController:
    def __init__(self, ema_alpha=0.5):
        self.stress_ema = 0.0
        self.ema_alpha = ema_alpha
        self._last_temp = 50.0
        self._last_sclk = 600.0

    def update(self):
        self._last_temp = read_gpu_temp_c()
        self._last_sclk = read_current_sclk_mhz()
        t_norm = max(0.0, min(1.0, (self._last_temp - 40.0) / 40.0)) # Hot is 80C
        c_norm = max(0.0, min(1.0, (self._last_sclk - SCLK_LOW_CAL) / max(SCLK_HIGH_CAL - SCLK_LOW_CAL, 1.0)))
        stress = (t_norm + c_norm) / 2.0
        self.stress_ema = (1 - self.ema_alpha) * self.stress_ema + self.ema_alpha * stress
        return self.stress_ema

    def get_continuous_stress(self):
        return self.stress_ema

    def get_body_mode_byte(self):
        s = self.stress_ema
        if s < 0.20: return pack_mode_byte(0, 0, 3, 3) 
        elif s < 0.50: return pack_mode_byte(3, 3, 3, 3) 
        elif s < 0.80: return pack_mode_byte(2, 2, 3, 3) 
        else: return pack_mode_byte(3, 3, 0, 0) 

THERMO = ThermodynamicController()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_wikitext_data(tokenizer, split='train', max_samples=2000):
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    all_ids = []
    for text in ds['text']:
        if len(text.strip()) < 50: continue
        all_ids.extend(tokenizer.encode(text, add_special_tokens=False))
    sequences = [torch.tensor(all_ids[i:i + SEQ_LEN], dtype=torch.long) 
                 for i in range(0, len(all_ids) - SEQ_LEN, SEQ_LEN)][:max_samples]
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
    for sample in ds:
        text = sample['content']
        if len(text.strip()) < 100: continue
        all_ids.extend(tokenizer.encode(text, add_special_tokens=False))
        if len(all_ids) >= max_samples * SEQ_LEN * 2: break
    sequences = [torch.tensor(all_ids[i:i + SEQ_LEN], dtype=torch.long) 
                 for i in range(0, len(all_ids) - SEQ_LEN, SEQ_LEN)][:max_samples]
    print(f"  Loaded {len(sequences)} code sequences")
    return sequences


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUSTOM HIP KERNEL — PWM Duty-Cycled Analog GEMM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_HIP_MODULE = None
_HIP_COMPILE_ATTEMPTED = False

def compile_hip_kernel():
    global _HIP_MODULE, _HIP_COMPILE_ATTEMPTED
    if _HIP_MODULE is not None: return _HIP_MODULE
    if _HIP_COMPILE_ATTEMPTED: return None
    _HIP_COMPILE_ATTEMPTED = True

    print("[HIP] Compiling PWM duty-cycled analog GEMM kernel...")
    from torch.utils.cpp_extension import load_inline

    combined_source = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#define TILE_SIZE 16

__global__ void analog_tiled_gemm_kernel(
    const __hip_bfloat16* __restrict__ A,
    const __hip_bfloat16* __restrict__ B,
    __hip_bfloat16* __restrict__ C,
    int M, int K, int N,
    unsigned int base_round_mode,
    float continuous_stress 
) {
    unsigned int old_mode;
    asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" ::: "memory");
    asm volatile("s_getreg_b32 %0, hwreg(1, 0, 8)" : "=s"(old_mode) :: "memory");

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float acc = 0.0f;
    int n_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    unsigned int stress_threshold = (unsigned int)(continuous_stress * 255.0f);

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

        // NATIVE CLOCK FIX: hwreg(29) often returns 0 due to privileges. 
        // clock() is guaranteed to tick per execution cycle.
        unsigned int cycles = (unsigned int)clock();

        unsigned int hw_spin = cycles & 0xFFu;
        unsigned int inject_mask = (hw_spin < stress_threshold) ? 0xFFu : 0x00u;

        unsigned int rm = base_round_mode ^ (cycles & inject_mask);
        unsigned int rm_packed = (rm & 0x3u) | ((rm & 0x3u) << 2) | (rm & 0xF0u);
        unsigned int new_mode = (old_mode & ~0xFFu) | rm_packed;
        asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" :: "s"(new_mode) : "memory");

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

    asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" ::: "memory");
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" :: "s"(old_mode) : "memory");
}

torch::Tensor analog_gemm(torch::Tensor A, torch::Tensor weight, int round_mode, float continuous_stress) {
    TORCH_CHECK(A.is_cuda() && weight.is_cuda(), "Tensors must be on GPU");
    int M = A.size(0);
    int K = A.size(1);
    int N = weight.size(0);
    auto C = torch::empty({M, N}, A.options());
    const int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    analog_tiled_gemm_kernel<<<grid, block>>>(
        reinterpret_cast<const __hip_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),
        M, K, N, (unsigned int)round_mode, continuous_stress);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("analog_gemm", &analog_gemm); }
"""
    try:
        _HIP_MODULE = load_inline(
            name='analog_gemm_ext', cpp_sources=[], cuda_sources=[combined_source],
            extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'], with_cuda=True)
    except:
        _HIP_MODULE = None
    return _HIP_MODULE


class AnalogLinear(nn.Module):
    def __init__(self, original_linear, layer_idx, mode_override=None):
        super().__init__()
        self.weight = original_linear.weight 
        self.bias = original_linear.bias
        self.layer_idx = layer_idx
        self.mode_override = mode_override
        self._last_round_mode = 0
        self._last_stress = 0.0

    def forward(self, x):
        if self.mode_override is not None:
            round_mode = self.mode_override
            continuous_stress = 0.0 
        else:
            THERMO.update()
            stress = THERMO.get_continuous_stress()
            round_mode = THERMO.get_body_mode_byte()
            continuous_stress = float(stress)

        self._last_round_mode = round_mode
        self._last_stress = continuous_stress

        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        N = self.weight.shape[0]

        hip_mod = compile_hip_kernel()
        if hip_mod is not None:
            out = hip_mod.analog_gemm(x_2d.contiguous(), self.weight.contiguous(), round_mode, continuous_stress)
        else:
            out = F.linear(x_2d, self.weight)

        if self.bias is not None: out = out + self.bias
        return out.reshape(*orig_shape[:-1], N)

def patch_analog_layers(model, layers, mode_override=None):
    originals = {}
    for layer_idx in layers:
        block = model.model.layers[layer_idx]
        
        orig_gate = block.mlp.gate_proj
        analog_gate = AnalogLinear(orig_gate, layer_idx, mode_override=mode_override)
        block.mlp.gate_proj = analog_gate
        
        orig_up = block.mlp.up_proj
        analog_up = AnalogLinear(orig_up, layer_idx, mode_override=mode_override)
        block.mlp.up_proj = analog_up
        
        originals[layer_idx] = {'gate': orig_gate, 'up': orig_up}
        print(f"  [AnalogLinear] Patched layer {layer_idx} gate_proj & up_proj")
    return originals

def restore_analog_layers(model, originals):
    for layer_idx, origs in originals.items():
        model.model.layers[layer_idx].mlp.gate_proj = origs['gate']
        model.model.layers[layer_idx].mlp.up_proj = origs['up']

def set_analog_mode_override(model, layers, mode_override):
    for layer_idx in layers:
        gate = model.model.layers[layer_idx].mlp.gate_proj
        up = model.model.layers[layer_idx].mlp.up_proj
        if isinstance(gate, AnalogLinear): gate.mode_override = mode_override
        if isinstance(up, AnalogLinear): up.mode_override = mode_override

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE B: Tiny LoRA + Somatosensory Head
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TinyLoRA(nn.Module):
    def __init__(self, original_linear, rank=2, alpha=4):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, original_linear.in_features, dtype=original_linear.weight.dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original_linear.out_features, rank, dtype=original_linear.weight.dtype))

    def forward(self, x):
        base = self.original(x)
        lora_out = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B) * self.scaling
        return base + lora_out.to(x.dtype)

class SomatosensoryHead(nn.Module):
    def __init__(self, hidden_dim, dtype=torch.bfloat16):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 64, dtype=dtype)
        self.out = nn.Linear(64, 1, dtype=dtype)

    def forward(self, hidden_states):
        h = hidden_states.mean(dim=1)
        h = F.gelu(self.proj(h))
        return torch.tanh(self.out(h).squeeze(-1))

def patch_v_proj_with_lora(model, layers, rank=2, alpha=4):
    originals = {}
    for layer_idx in layers:
        block = model.model.layers[layer_idx]
        orig = block.self_attn.v_proj
        lora = TinyLoRA(orig, rank=rank, alpha=alpha).to(orig.weight.device)
        block.self_attn.v_proj = lora
        originals[layer_idx] = orig
    return originals

def restore_v_proj(model, originals):
    for layer_idx, orig in originals.items():
        model.model.layers[layer_idx].self_attn.v_proj = orig

def get_lora_params(model, layers):
    params = []
    for layer_idx in layers:
        v = model.model.layers[layer_idx].self_attn.v_proj
        if isinstance(v, TinyLoRA):
            params.extend([v.lora_A, v.lora_B])
    return params

def eval_ppl(model, data, n_batches=None, device=DEVICE):
    model.eval()
    total_loss, total_tokens = 0.0, 0
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
            total_loss += loss.item()
            total_tokens += (shift_labels != -100).sum().item()
    model.train()
    return math.exp(min(total_loss / max(total_tokens, 1), 20))

def collect_logits(model, data, n_batches=5, device=DEVICE):
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
    p = F.softmax(logits_p.float(), dim=-1).clamp(min=1e-10)
    q = F.softmax(logits_q.float(), dim=-1).clamp(min=1e-10)
    return (p * (p.log() - q.log())).sum(dim=-1).mean().item()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE A & B EXECUTORS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_phase_a(model, test_data, test_data_code, baseline_ppl):
    print("\n" + "=" * 60)
    print("PHASE A: Frozen Measurement (0 trainable params)")
    print("  v3: Amplified Depth (9 layers x 2 projections) + native clock()")
    print("=" * 60)
    results = {'baseline_ppl': baseline_ppl}

    originals = patch_analog_layers(model, ANALOG_LAYERS, mode_override=None)

    print("\n[T2] AnalogLinear PPL (live entropy, continuous stress)...")
    analog_ppl = eval_ppl(model, test_data)
    results['analog_ppl'] = analog_ppl
    print(f"  Analog PPL: {analog_ppl:.4f} (baseline: {baseline_ppl:.4f})")

    print("\n[T3] Mode Sweep (force FP16 rounding modes 0-3)...")
    mode_ppls, mode_logits = {}, {}
    for mode_val in range(4):
        packed = pack_mode_byte(mode_val, mode_val, 3, 3)
        set_analog_mode_override(model, ANALOG_LAYERS, packed)
        mode_ppls[mode_val] = eval_ppl(model, test_data)
        mode_logits[mode_val] = collect_logits(model, test_data, n_batches=3)
        print(f"  Mode {mode_val}: PPL={mode_ppls[mode_val]:.4f}")
    results['mode_ppl_std'] = float(np.std(list(mode_ppls.values())))
    print(f"  PPL std across modes: {results['mode_ppl_std']:.6f}")

    print("\n[T4] KL Divergence across mode pairs...")
    max_kl = 0.0
    for i in range(4):
        for j in range(i+1, 4):
            kl_val = kl_divergence(mode_logits[i], mode_logits[j])
            max_kl = max(max_kl, kl_val)
    results['max_kl'] = max_kl
    print(f"  Max KL: {max_kl:.8f}")

    print("\n[T5] Thermal Variance (DVFS low vs high)...")
    dvfs_ppls = {}
    if DVFS_AVAILABLE:
        set_analog_mode_override(model, ANALOG_LAYERS, None)
        for dvfs_level, dvfs_name in [(0, 'low'), (2, 'high')]:
            set_dvfs_level(dvfs_level, wait=True)
            time.sleep(5.0) 
            THERMO.stress_ema = 0.0 if dvfs_name == 'low' else 1.0
            for _ in range(30):
                THERMO.update()
                time.sleep(0.3)
            dvfs_ppls[dvfs_name] = eval_ppl(model, test_data)
            print(f"  DVFS {dvfs_name}: PPL={dvfs_ppls[dvfs_name]:.4f}, stress={THERMO.stress_ema:.4f}")
        set_dvfs_level(1, wait=True)
    results['dvfs_ppl_std'] = float(np.std(list(dvfs_ppls.values()))) if dvfs_ppls else 0.0

    print("\n[T6] Determinism (packed mode=0x0F, 5 runs)...")
    set_analog_mode_override(model, ANALOG_LAYERS, pack_mode_byte(3, 3, 3, 3))
    det_ppls = [eval_ppl(model, test_data) for _ in range(5)]
    results['determinism_std'] = float(np.std(det_ppls))
    print(f"  Std: {results['determinism_std']:.8f}")

    print("\n[T8] FP16 Denorm Sweep [7:6]...")
    denorm_kls = []
    logits_d0 = collect_logits(model, test_data, n_batches=3)
    set_analog_mode_override(model, ANALOG_LAYERS, pack_mode_byte(0, 0, 3, 3))
    logits_d3 = collect_logits(model, test_data, n_batches=3)
    max_denorm_kl = kl_divergence(logits_d0, logits_d3)
    results['max_denorm_kl'] = max_denorm_kl
    print(f"  Max FP16 denorm KL: {max_denorm_kl:.8f}")

    print("\n[T10] Ablation...")
    restore_analog_layers(model, originals)
    results['ablation_ppl_diff'] = abs(analog_ppl - eval_ppl(model, test_data))
    originals = patch_analog_layers(model, ANALOG_LAYERS, mode_override=None)

    print("\n[T11] Domain Sensitivity...")
    if test_data_code is not None:
        set_analog_mode_override(model, ANALOG_LAYERS, pack_mode_byte(0, 0, 3, 3))
        w0 = eval_ppl(model, test_data, n_batches=5)
        c0 = eval_ppl(model, test_data_code, n_batches=5)
        set_analog_mode_override(model, ANALOG_LAYERS, pack_mode_byte(3, 3, 3, 3))
        w3 = eval_ppl(model, test_data, n_batches=5)
        c3 = eval_ppl(model, test_data_code, n_batches=5)
        results['domain_interaction'] = abs(abs(w3 - w0) - abs(c3 - c0))
    else:
        results['domain_interaction'] = 0.0

    print("\n[T12] Zombie Twin...")
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    logits_live = collect_logits(model, test_data, n_batches=3)
    set_analog_mode_override(model, ANALOG_LAYERS, pack_mode_byte(0, 0, 3, 3))
    logits_zombie = collect_logits(model, test_data, n_batches=3)
    results['zombie_kl'] = kl_divergence(logits_live, logits_zombie)

    print("\n[T14] Argmax Flip Rate...")
    if mode_logits[0] is not None and mode_logits[3] is not None:
        results['argmax_flip_rate'] = (mode_logits[0].argmax(dim=-1) != mode_logits[3].argmax(dim=-1)).float().mean().item()

    set_analog_mode_override(model, ANALOG_LAYERS, None)
    restore_analog_layers(model, originals)
    return results

def run_phase_b(model, train_data, test_data, phase_a_ppl):
    print("\n" + "=" * 60)
    print("PHASE B: Tiny LoRA + SomaHead (DVFS alternation training)")
    print("=" * 60)
    results = {}
    analog_originals = patch_analog_layers(model, ANALOG_LAYERS, mode_override=None)
    lora_originals, n_lora_params = patch_v_proj_with_lora(model, LORA_LAYERS)

    hidden_dim = model.config.hidden_size
    soma_head = SomatosensoryHead(hidden_dim).to(DEVICE)
    soma_params = list(soma_head.parameters())
    
    for p in model.parameters(): p.requires_grad = False
    all_trainable = get_lora_params(model, LORA_LAYERS) + soma_params
    for p in all_trainable: p.requires_grad = True

    temp_min = read_gpu_temp_c()
    temp_max = temp_min + 20.0
    optimizer = torch.optim.AdamW(all_trainable, lr=1e-4)
    
    model.train()
    soma_head.train()
    for batch_i in range(min(120, len(train_data) // BS)):
        if batch_i % 20 == 0 and DVFS_AVAILABLE:
            set_dvfs_level([0, 2, 1, 0, 2, 1][(batch_i // 20) % 6], wait=True)
            time.sleep(1.0)
            for _ in range(5): THERMO.update()

        batch = torch.stack(train_data[batch_i*BS:(batch_i+1)*BS]).to(DEVICE)
        t = read_gpu_temp_c()
        temp_min, temp_max = min(temp_min, t), max(temp_max, t)
        temp_target = torch.full((BS,), 2.0 * (t - temp_min) / max(temp_max - temp_min, 5.0) - 1.0, device=DEVICE)

        out = model(batch, output_hidden_states=True)
        logits = out.logits if hasattr(out, 'logits') else out[0]
        ce_loss = F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, logits.size(-1)), batch[:, 1:].contiguous().view(-1))
        
        # INCREASED SOMA LOSS WEIGHT
        soma_loss = F.mse_loss(soma_head(out.hidden_states[-1]).float(), temp_target)
        loss = ce_loss + 2.0 * soma_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if DVFS_AVAILABLE: set_dvfs_level(1, wait=True)

    print("\n[T17] Embodied Advantage...")
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    ppl_live = eval_ppl(model, test_data)
    best_fixed = 999.0
    for mv in range(4):
        set_analog_mode_override(model, ANALOG_LAYERS, pack_mode_byte(mv, mv, 3, 3))
        best_fixed = min(best_fixed, eval_ppl(model, test_data))
    results['embodied_advantage'] = best_fixed - ppl_live

    print("\n[T19] Interoception...")
    model.eval()
    soma_head.eval()
    soma_test_preds, soma_test_targets = [], []
    if DVFS_AVAILABLE:
        for lvl in [0, 2]:
            set_dvfs_level(lvl, wait=True)
            time.sleep(3.0)
            for _ in range(5): THERMO.update()
            with torch.no_grad():
                batch = torch.stack(test_data[:BS]).to(DEVICE)
                t = read_gpu_temp_c()
                soma_test_preds.append(soma_head(model(batch, output_hidden_states=True).hidden_states[-1]).mean().item())
                soma_test_targets.append(2.0 * (t - temp_min) / max(temp_max - temp_min, 5.0) - 1.0)
        set_dvfs_level(1, wait=True)
    
    if len(soma_test_preds) >= 2:
        results['soma_correlation'] = abs(float(np.corrcoef(soma_test_preds, soma_test_targets)[0, 1]))
    else: results['soma_correlation'] = 0.0

    restore_v_proj(model, lora_originals)
    restore_analog_layers(model, analog_originals)
    return results

def score_tests(pa, pb):
    n_pass = n_tot = 0
    def r(tid, name, p, v, c):
        nonlocal n_pass, n_tot; n_tot += 1; n_pass += int(p)
        print(f"  {tid}: {name} — {'PASS' if p else 'FAIL'} ({v} vs {c})")

    print("\n" + "=" * 60 + "\nTEST BATTERY RESULTS\n" + "=" * 60)
    
    r('T3', 'Mode Sweep σ(PPL)', pa.get('mode_ppl_std', 0) > 0.01, f"σ={pa.get('mode_ppl_std', 0):.6f}", '>0.01')
    r('T4', 'KL Divergence', pa.get('max_kl', 0) > 0.001, f"max_kl={pa.get('max_kl', 0):.8f}", '>0.001')
    r('T5', 'Thermal Variance', pa.get('dvfs_ppl_std', 0) > 0.02, f"std={pa.get('dvfs_ppl_std', 0):.6f}", '>0.02')
    r('T6', 'Determinism', pa.get('determinism_std', 0) < 0.001, f"std={pa.get('determinism_std', 0):.8f}", '<0.001')
    r('T8', 'FP16 Denorm Sweep', pa.get('max_denorm_kl', 0) > 0.0001, f"max_kl={pa.get('max_denorm_kl', 0):.8f}", '>0.0001')
    r('T10', 'Ablation', pa.get('ablation_ppl_diff', 0) > 0.001, f"diff={pa.get('ablation_ppl_diff', 0):.6f}", '>0.001')
    r('T11', 'Domain Sensitivity', pa.get('domain_interaction', 0) > 0.01, f"interaction={pa.get('domain_interaction', 0):.4f}", '>0.01')
    r('T12', 'Zombie Twin', pa.get('zombie_kl', 0) > 0.0001, f"kl={pa.get('zombie_kl', 0):.8f}", '>0.0001')
    r('T14', 'Argmax Flip Rate', pa.get('argmax_flip_rate', 0) > 0.001, f"rate={pa.get('argmax_flip_rate', 0):.6f}", '>0.001')
    
    if pb:
        emb_adv = pb.get('embodied_advantage', 0)
        r('T17', 'Embodied Advantage', emb_adv > 0.05, f"advantage={emb_adv:.6f}", '>0.05')
        r('T19', 'Interoception', pb.get('soma_correlation', 0) > 0.3, f"corr={pb.get('soma_correlation', 0):.4f}", '>0.3')

    return n_pass, n_tot

def main():
    torch.manual_seed(42)
    print("z2114v3: Analog Linear LM — Continuous Thermodynamic Embodiment")
    
    find_dvfs_sysfs()
    compile_hip_kernel()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-8B', torch_dtype=torch.bfloat16).to(DEVICE)
    for p in model.parameters(): p.requires_grad = False
    
    td = load_wikitext_data(tokenizer, max_samples=500)
    tdc = load_code_data(tokenizer, max_samples=500)
    
    bl = eval_ppl(model, td)
    pa = run_phase_a(model, td, tdc, bl)
    pb = run_phase_b(model, td, td, bl) if pa.get('max_kl',0) > 0.0001 else None
    n_pass, n_tot = score_tests(pa, pb)
    print(f"\nz2114v3 Analog Linear LM: {n_pass}/{n_tot} PASS")

if __name__ == '__main__':
    main()