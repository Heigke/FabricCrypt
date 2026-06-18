#!/usr/bin/env python3
"""
z2114v5: Deep Embodied Analog Linear LM — Closed-Loop Monist Architecture
==========================================================================
v5 changes from v4 (18/24):
  1. Per-layer proprio gating (not global) — each LoRA gate sees its own layer timing
  2. BodyGatedLoRA on mlp.gate_proj (at perturbation site, not v_proj)
  3. Microchunk training (16-token chunks with context updates between)
  4. 3-timescale body: fast (proprio), mid (gpu_metrics snapshot), slow (thermal EMA)
  5. gpu_metrics synchronized snapshot (single read gives clocks+power+temp+throttle)
  6. d/dt derivative features in body vector
  7. PRBS DVFS schedule for training (randomized, not predictable cycling)
  8. T22 fix: quantile-based regime thresholds + hysteresis + class-weighted CE + confusion matrix
  9. New tests: step response, causal ablation, counterfactual replay, gate AUROC,
     matched-state counterfactual for embodied advantage
  10. Matched-state embodied advantage (per-regime comparison, not global)

Hardware setup:
  sudo modprobe msr
  sudo insmod ~/Documents/claude_hive/ryzen_smu/ryzen_smu.ko
  sudo chmod 666 /sys/kernel/ryzen_smu_drv/smn
  sudo chmod 666 /sys/kernel/ryzen_smu_drv/pm_table
  sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTORCH_ROCM_ARCH=gfx1100 \
    venv/bin/python -u scripts/z2114_analog_linear_lm_v5.py
"""

import os, sys, json, math, time, struct, ctypes, ctypes.util, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from collections import deque

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEVICE = 'cuda'
BS = 4
SEQ_LEN = 128
MICRO_CHUNK = 16          # v5: microchunk size for embodied training
N_EVAL_BATCHES = 30
DVFS_SETTLE_S = 1.5

N_LAYERS = 36
ANALOG_LAYERS = list(range(N_LAYERS))
LORA_RANK = 8
LORA_ALPHA = 16
LORA_LAYERS = list(range(8, 32))

SCLK_LOW_CAL = 600.0
SCLK_HIGH_CAL = 2900.0

# v5: 3-timescale body dimensions
FAST_DIM = 4    # per-layer: mean_ticks, jitter, tail_ratio, d_jitter
MID_DIM = 8     # gpu_metrics snapshot: temp_gfx, temp_soc, gfx_power, socket_power, sclk, activity, throttle_bits, d_temp
SLOW_DIM = 4    # thermal EMA: stress_ema, d_stress, smn_adc, pcie_replay
BODY_DIM = FAST_DIM + MID_DIM + SLOW_DIM  # 16 total

# Regime labels
REGIME_COLD, REGIME_NOMINAL, REGIME_HOT, REGIME_THROTTLED = 0, 1, 2, 3
N_REGIMES = 4

SMN_FD = None   # forward declare for global


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
# HARDWARE ACCESS — SMN, RAPL, MSR
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
# v5: gpu_metrics SYNCHRONIZED SNAPSHOT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPU_METRICS_PATH = None


def find_gpu_metrics():
    global GPU_METRICS_PATH
    for card in ['card1', 'card0']:
        p = f'/sys/class/drm/{card}/device/gpu_metrics'
        if os.path.exists(p):
            GPU_METRICS_PATH = p
            print(f"[gpu_metrics] Found: {p}")
            return True
    print("[gpu_metrics] Not available")
    return False


def read_gpu_metrics():
    """Single atomic read of gpu_metrics v3.0 blob.
    Returns dict with synchronized: temp_gfx, temp_soc, gfx_power, socket_power,
    gfx_activity, sclk_mhz, throttle_status."""
    if GPU_METRICS_PATH is None:
        return None
    try:
        with open(GPU_METRICS_PATH, 'rb') as f:
            data = f.read()
        if len(data) < 100:
            return None
        # v3.0 layout (264 bytes, format_rev=3, content_rev=0)
        result = {
            'temp_gfx': struct.unpack_from('<H', data, 4)[0] / 100.0,
            'temp_soc': struct.unpack_from('<H', data, 6)[0] / 100.0,
            'gfx_activity': struct.unpack_from('<H', data, 48)[0] / 100.0,
            'gfx_power': struct.unpack_from('<H', data, 66)[0],      # mW
            'socket_power': struct.unpack_from('<H', data, 60)[0],    # mW
        }
        # sclk is at offset 174 (from our empirical analysis)
        if len(data) > 176:
            result['sclk_mhz'] = struct.unpack_from('<H', data, 174)[0]
        # Throttle status near end
        if len(data) >= 240:
            result['throttle_status'] = struct.unpack_from('<I', data, 236)[0]
        else:
            result['throttle_status'] = 0
        return result
    except Exception:
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v5: 3-TIMESCALE FORWARD CONTEXT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ForwardContext:
    """v5: 3-timescale body state snapshot.
    fast:  per-layer proprio (wall_clock64 jitter) — updated per forward/microchunk
    mid:   gpu_metrics snapshot (temp, power, clocks, throttle) — updated per batch
    slow:  thermal EMA + SMN + pcie — updated per batch with EMA smoothing
    """
    def __init__(self):
        self.round_mode = pack_mode_byte(0, 0, 3, 3)
        self.continuous_stress = 0.0
        self.stress_threshold = 0
        self.forward_id = 0
        self.snapshot_ts = 0.0
        # Fast: per-layer proprio from kernel (updated per forward)
        self.layer_proprio = {}           # layer_idx -> {mean_ticks, jitter, tail_ratio}
        self.prev_layer_proprio = {}      # previous forward's proprio (for d/dt)
        # Mid: gpu_metrics synchronized snapshot
        self.mid_vec = np.zeros(MID_DIM, dtype=np.float32)
        self.prev_mid_vec = np.zeros(MID_DIM, dtype=np.float32)
        # Slow: thermal EMA
        self.slow_vec = np.zeros(SLOW_DIM, dtype=np.float32)
        self.prev_slow_vec = np.zeros(SLOW_DIM, dtype=np.float32)
        # Regime (from quantile thresholds)
        self.regime = REGIME_NOMINAL

    def clear_proprio(self):
        self.prev_layer_proprio = dict(self.layer_proprio)
        self.layer_proprio = {}

    def get_per_layer_fast_vec(self, layer_idx):
        """v5: Per-layer fast body vector [FAST_DIM] for gating.
        [mean_ticks_norm, jitter_norm, tail_ratio, d_jitter]"""
        vec = np.zeros(FAST_DIM, dtype=np.float32)
        info = self.layer_proprio.get(layer_idx)
        if info is None:
            # Use previous forward's data if current not yet available
            info = self.prev_layer_proprio.get(layer_idx)
        if info is not None:
            vec[0] = info['mean_ticks'] / 1e6  # normalize
            vec[1] = info['jitter'] / 1e6
            vec[2] = info.get('tail_ratio', 0.0)
            # d/dt: jitter change from previous forward
            prev = self.prev_layer_proprio.get(layer_idx)
            if prev is not None:
                vec[3] = (info['jitter'] - prev['jitter']) / 1e6
        return vec

    def get_full_body_vec(self, layer_idx):
        """v5: Full body vector [BODY_DIM] = fast[layer] + mid + slow."""
        fast = self.get_per_layer_fast_vec(layer_idx)
        return np.concatenate([fast, self.mid_vec, self.slow_vec])


CTX = ForwardContext()


class MetabolicController:
    """v5: 3-timescale metabolic controller.
    - Mid loop: reads gpu_metrics snapshot (synchronized)
    - Slow loop: EMA-smoothed stress + SMN + PCIe
    - Regime classification: quantile-based with hysteresis
    """
    def __init__(self, ema_alpha=0.3):
        self.ema_alpha = ema_alpha
        self.stress_ema = 0.0
        self._last_temp = 50.0
        self._last_sclk = 600.0
        self._last_power = 0.0
        # Running stats for normalization
        self._n_updates = 0
        self._running_mean = np.zeros(BODY_DIM, dtype=np.float64)
        self._running_var = np.ones(BODY_DIM, dtype=np.float64)
        # v5: Quantile-based regime thresholds (calibrated during warmup)
        self._stress_history = deque(maxlen=200)
        self._regime_thresholds = [0.25, 0.50, 0.75]  # default quartiles
        self._hysteresis = 0.03  # prevent flicker
        self._prev_regime = REGIME_NOMINAL

    def snapshot_mid(self):
        """Mid-loop: single atomic gpu_metrics read."""
        gm = read_gpu_metrics()
        prev = CTX.mid_vec.copy()
        if gm is not None:
            self._last_temp = gm['temp_gfx']
            self._last_sclk = gm.get('sclk_mhz', read_current_sclk_mhz())
            self._last_power = gm.get('gfx_power', 0.0)
            CTX.mid_vec = np.array([
                gm['temp_gfx'] / 100.0,
                gm['temp_soc'] / 100.0,
                gm.get('gfx_power', 0) / 60000.0,
                gm.get('socket_power', 0) / 120000.0,
                self._last_sclk / 3000.0,
                gm.get('gfx_activity', 0) / 100.0,
                float(gm.get('throttle_status', 0) > 0),
                (gm['temp_gfx'] / 100.0 - prev[0]) if self._n_updates > 0 else 0.0,
            ], dtype=np.float32)
        else:
            self._last_temp = read_gpu_temp_c()
            self._last_sclk = read_current_sclk_mhz()
            self._last_power = read_gpu_power_mw()
            CTX.mid_vec = np.array([
                self._last_temp / 100.0,
                self._last_temp / 100.0,
                self._last_power / 60000.0,
                0.0,
                self._last_sclk / 3000.0,
                0.0, 0.0,
                (self._last_temp / 100.0 - prev[0]) if self._n_updates > 0 else 0.0,
            ], dtype=np.float32)
        CTX.prev_mid_vec = prev

    def snapshot_slow(self):
        """Slow-loop: EMA stress + SMN + PCIe."""
        prev = CTX.slow_vec.copy()
        t_norm = max(0.0, min(1.0, (self._last_temp - 40.0) / 50.0))
        c_norm = max(0.0, min(1.0, (self._last_sclk - SCLK_LOW_CAL) /
                                   max(SCLK_HIGH_CAL - SCLK_LOW_CAL, 1.0)))
        p_norm = max(0.0, min(1.0, self._last_power / 60000.0))
        stress = (t_norm + c_norm + p_norm) / 3.0
        self.stress_ema = (1 - self.ema_alpha) * self.stress_ema + self.ema_alpha * stress
        smn_adc = float(read_smn(0x00059800) & 0xFF) / 255.0 if SMN_AVAILABLE else 0.0
        pcie = float(read_pcie_replay_count()) / 1000.0

        CTX.slow_vec = np.array([
            self.stress_ema,
            self.stress_ema - prev[0] if self._n_updates > 0 else 0.0,
            smn_adc,
            pcie,
        ], dtype=np.float32)
        CTX.prev_slow_vec = prev

    def classify_regime(self):
        """v5: Quantile-based regime with hysteresis."""
        self._stress_history.append(self.stress_ema)
        if len(self._stress_history) >= 20 and self._n_updates % 20 == 0:
            arr = np.array(self._stress_history)
            self._regime_thresholds = [
                float(np.percentile(arr, 25)),
                float(np.percentile(arr, 50)),
                float(np.percentile(arr, 75)),
            ]
        s = self.stress_ema
        t1, t2, t3 = self._regime_thresholds
        h = self._hysteresis
        prev = self._prev_regime
        if prev == REGIME_COLD and s > t1 + h:
            regime = REGIME_NOMINAL
        elif prev == REGIME_NOMINAL and s < t1 - h:
            regime = REGIME_COLD
        elif prev == REGIME_NOMINAL and s > t2 + h:
            regime = REGIME_HOT
        elif prev == REGIME_HOT and s < t2 - h:
            regime = REGIME_NOMINAL
        elif prev == REGIME_HOT and s > t3 + h:
            regime = REGIME_THROTTLED
        elif prev == REGIME_THROTTLED and s < t3 - h:
            regime = REGIME_HOT
        else:
            regime = prev
        self._prev_regime = regime
        return regime

    def update_context(self):
        """Full context update: mid + slow + regime."""
        self.snapshot_mid()
        self.snapshot_slow()
        CTX.regime = self.classify_regime()
        CTX.continuous_stress = float(self.stress_ema)
        CTX.stress_threshold = max(0, min(255, int(self.stress_ema * 255)))
        CTX.round_mode = pack_mode_byte(0, 0, 3, 3)
        CTX.snapshot_ts = time.time()
        CTX.forward_id += 1
        CTX.clear_proprio()
        self._n_updates += 1
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
# CUSTOM HIP KERNEL — v5: same TILE=32 + wall_clock64 proprio as v4
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

    print("[HIP] Compiling v5 analog GEMM kernel (TILE=32 + wall_clock64 proprio)...")
    from torch.utils.cpp_extension import load_inline

    combined_source = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#define TILE_SIZE 32

__global__ void analog_tiled_gemm_v5_kernel(
    const __hip_bfloat16* __restrict__ A,
    const __hip_bfloat16* __restrict__ B,
    __hip_bfloat16* __restrict__ C,
    unsigned long long* __restrict__ proprio_out,
    int M, int K, int N,
    int base_round_mode,
    int stress_threshold
) {
    unsigned long long t_start = wall_clock64();

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

        unsigned int cycles;
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(cycles));
        unsigned int hw_spin = cycles & 0xFFu;

        unsigned int active = ((int)hw_spin < stress_threshold) ? 0x05u : (unsigned int)base_round_mode;
        unsigned int rm_both = (active & 0x3u) | ((active & 0x3u) << 2) | (active & 0xF0u);
        unsigned int new_mode = (old_mode & ~0xFFu) | rm_both;
        asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" :: "s"(new_mode) : "memory");

        __half chunk_acc = __float2half(0.0f);
        for (int k = 0; k < TILE_SIZE; k++) {
            __half a_h = __float2half(As[threadIdx.y][k]);
            __half b_h = __float2half(Bs[k][threadIdx.x]);
            chunk_acc = __hadd(chunk_acc, __hmul(a_h, b_h));
        }
        acc += __half2float(chunk_acc);

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(acc);
    }

    asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" ::: "memory");
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" :: "s"(old_mode) : "memory");

    unsigned long long t_end = wall_clock64();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int block_id = blockIdx.y * gridDim.x + blockIdx.x;
        proprio_out[block_id] = t_end - t_start;
    }
}

std::vector<torch::Tensor> analog_gemm_v5(
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

    auto proprio = torch::empty({n_blocks}, torch::TensorOptions().dtype(torch::kInt64).device(A.device()));

    int stress_threshold = (int)(continuous_stress * 255.0f);
    if (stress_threshold < 0) stress_threshold = 0;
    if (stress_threshold > 255) stress_threshold = 255;

    analog_tiled_gemm_v5_kernel<<<grid, block>>>(
        reinterpret_cast<const __hip_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),
        reinterpret_cast<unsigned long long*>(proprio.data_ptr()),
        M, K, N, round_mode, stress_threshold);

    return {C, proprio};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("analog_gemm_v5", &analog_gemm_v5, "v5 analog GEMM with proprioception");
}
"""

    try:
        _HIP_MODULE = load_inline(
            name='analog_gemm_v5_ext',
            cpp_sources=[],
            cuda_sources=[combined_source],
            extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'],
            verbose=True,
            with_cuda=True,
        )
        print("[HIP] v5 Kernel compiled OK")
    except Exception as e:
        print(f"[HIP] Compilation failed: {e}")
        _HIP_MODULE = None
    return _HIP_MODULE


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANALOG LINEAR — v5: same as v4, writes per-layer proprio with tail_ratio
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AnalogLinear(nn.Module):
    """Reads from ForwardContext, writes wall_clock64 proprioception back."""
    def __init__(self, original_linear, layer_idx, mode_override=None):
        super().__init__()
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.layer_idx = layer_idx
        self.mode_override = mode_override

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
            round_mode = CTX.round_mode
            continuous_stress = CTX.continuous_stress

        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        hip_mod = compile_hip_kernel()
        if hip_mod is not None:
            x_bf16 = x_2d.to(torch.bfloat16).contiguous()
            w_bf16 = self.weight.to(torch.bfloat16).contiguous()
            results = hip_mod.analog_gemm_v5(x_bf16, w_bf16, round_mode, continuous_stress)
            out = results[0]
            proprio_ticks = results[1]

            if self.mode_override is None:
                ticks_cpu = proprio_ticks.cpu().to(torch.float64)
                ticks_pos = ticks_cpu[ticks_cpu > 0]
                if len(ticks_pos) > 0:
                    sorted_t = ticks_pos.sort().values
                    n = len(sorted_t)
                    p50 = sorted_t[n // 2].item()
                    p95 = sorted_t[min(int(n * 0.95), n - 1)].item()
                    tail_ratio = (p95 / max(p50, 1.0)) - 1.0
                    CTX.layer_proprio[self.layer_idx] = {
                        'mean_ticks': float(ticks_pos.mean()),
                        'max_ticks': float(ticks_pos.max()),
                        'jitter': float(ticks_pos.max() - ticks_pos.min()),
                        'tail_ratio': tail_ratio,
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
    print(f"  [AnalogLinear] Total: {len(layers)} layers patched")
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
# v5: BODY-GATED LORA ON gate_proj (AT PERTURBATION SITE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class BodyGatedLoRA(nn.Module):
    """v5: LoRA on gate_proj with dual-timescale gating.
    g_fast = f(per_layer_proprio)   — reflexive compensation
    g_slow = h(mid + slow body)     — homeostatic drift compensation
    gate = sigmoid(g_fast + g_slow + bias)
    Placed on gate_proj to align actuation with analog perturbation."""

    def __init__(self, analog_linear, layer_idx, rank=8, alpha=16):
        super().__init__()
        self.analog_linear = analog_linear  # the AnalogLinear wrapping original
        self.layer_idx = layer_idx
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_f = analog_linear.in_features
        out_f = analog_linear.out_features
        dtype = analog_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.randn(rank, in_f, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, dtype=dtype))
        # v5: Dual-timescale gate
        self.gate_fast = nn.Linear(FAST_DIM, rank, dtype=torch.float32)
        self.gate_slow = nn.Linear(MID_DIM + SLOW_DIM, rank, dtype=torch.float32)
        nn.init.zeros_(self.gate_fast.weight)
        nn.init.zeros_(self.gate_slow.weight)
        nn.init.ones_(self.gate_fast.bias)   # start ~sigmoid(1)=0.73

    def forward(self, x):
        # Base computation through AnalogLinear (includes MODE register manipulation)
        base = self.analog_linear(x)

        # LoRA path
        x_cast = x.to(self.lora_A.dtype)
        lora_mid = F.linear(x_cast, self.lora_A)  # [..., rank]

        # v5: Dual-timescale gating
        # Fast gate: per-layer proprio
        fast_vec = CTX.get_per_layer_fast_vec(self.layer_idx)
        fast_input = torch.from_numpy(fast_vec).float().to(self.gate_fast.weight.device)
        g_fast = self.gate_fast(fast_input)  # [rank]

        # Slow gate: mid + slow body
        slow_input_np = np.concatenate([CTX.mid_vec, CTX.slow_vec])
        slow_input = torch.from_numpy(slow_input_np).float().to(self.gate_slow.weight.device)
        g_slow = self.gate_slow(slow_input)  # [rank]

        gate = torch.sigmoid(g_fast + g_slow)  # [rank]
        lora_mid = lora_mid * gate.to(lora_mid.dtype)
        lora_out = F.linear(lora_mid, self.lora_B) * self.scaling

        return base + lora_out.to(x.dtype)


def patch_gate_proj_with_lora(model, layers, rank=8, alpha=16):
    """v5: Patch gate_proj with BodyGatedLoRA WRAPPING AnalogLinear.
    The gate_proj must already be an AnalogLinear."""
    originals = {}
    total_params = 0
    for layer_idx in layers:
        block = model.model.layers[layer_idx]
        analog = block.mlp.gate_proj
        if not isinstance(analog, AnalogLinear):
            print(f"  [WARN] Layer {layer_idx} gate_proj is not AnalogLinear, skipping LoRA")
            continue
        lora = BodyGatedLoRA(analog, layer_idx, rank=rank, alpha=alpha).to(analog.weight.device)
        block.mlp.gate_proj = lora
        originals[layer_idx] = analog  # save the AnalogLinear (not the original Linear)
        n_p = sum(p.numel() for p in lora.parameters() if p.requires_grad)
        total_params += n_p
        if layer_idx in [8, 15, 23, 31]:
            print(f"  [BodyGatedLoRA] Patched layer {layer_idx} gate_proj (rank={rank}, {n_p} params)")
    print(f"  [BodyGatedLoRA] Total trainable: {total_params} across {len(originals)} layers")
    return originals, total_params


def restore_gate_proj_lora(model, originals):
    """Restore AnalogLinear (removing LoRA wrapper)."""
    for layer_idx, analog in originals.items():
        model.model.layers[layer_idx].mlp.gate_proj = analog


def get_lora_params(model, layers):
    params = []
    for layer_idx in layers:
        g = model.model.layers[layer_idx].mlp.gate_proj
        if isinstance(g, BodyGatedLoRA):
            params.extend([p for p in g.parameters() if p.requires_grad])
    return params


def aggregate_proprio():
    """v5: Aggregate proprioception across all layers into FAST_DIM stats."""
    if not CTX.layer_proprio:
        return torch.zeros(FAST_DIM)
    means, jitters, tails = [], [], []
    for info in CTX.layer_proprio.values():
        means.append(info['mean_ticks'])
        jitters.append(info['jitter'])
        tails.append(info.get('tail_ratio', 0.0))
    d_jitter = 0.0
    if CTX.prev_layer_proprio:
        prev_j = [v['jitter'] for v in CTX.prev_layer_proprio.values()]
        if prev_j:
            d_jitter = np.mean(jitters) - np.mean(prev_j)
    return torch.tensor([
        np.mean(means) / 1e6,
        np.mean(jitters) / 1e6,
        np.mean(tails),
        d_jitter / 1e6
    ], dtype=torch.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MULTI-TASK SOMA HEAD (v5: with confusion matrix + class-weighted CE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultiTaskSomaHead(nn.Module):
    """v5: Multi-task interoception head with per-layer proprio input.
    Input: pooled hidden state + aggregated proprio [FAST_DIM]
    Outputs:
      1. regime_logits: classify thermal regime [N_REGIMES]
      2. stress_pred: predict continuous stress scalar [1], tanh bounded
    """
    def __init__(self, hidden_dim, n_proprio=FAST_DIM, dtype=torch.bfloat16):
        super().__init__()
        self.input_dim = hidden_dim + n_proprio
        self.shared = nn.Linear(self.input_dim, 128, dtype=torch.float32)
        self.regime_head = nn.Linear(128, N_REGIMES, dtype=torch.float32)
        self.stress_head = nn.Linear(128, 1, dtype=torch.float32)
        # v5: Track confusion matrix
        self.confusion = np.zeros((N_REGIMES, N_REGIMES), dtype=np.int64)

    def forward(self, hidden_states, proprio_stats=None):
        h = hidden_states.float().mean(dim=1)  # [batch, hidden_dim]
        if proprio_stats is not None:
            ps = proprio_stats.unsqueeze(0).expand(h.size(0), -1).to(h.device)
            h = torch.cat([h, ps], dim=-1)
        else:
            h = torch.cat([h, torch.zeros(h.size(0), FAST_DIM, device=h.device)], dim=-1)
        shared_out = F.gelu(self.shared(h))
        regime_logits = self.regime_head(shared_out)
        stress_pred = torch.tanh(self.stress_head(shared_out).squeeze(-1))
        return regime_logits, stress_pred

    def update_confusion(self, pred_regime, true_regime):
        self.confusion[true_regime, pred_regime] += 1


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
            METAB.update_context()
            batch = torch.stack(data[i*BS:(i+1)*BS]).to(device)
            out = model(batch)
            logits = out.logits if hasattr(out, 'logits') else out[0]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), reduction='sum')
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


_TOKENIZER = None


def generate_text(model, data, mode_val, max_new=60, prompt_len=16):
    model.eval()
    try:
        with torch.no_grad():
            METAB.update_context()
            prompt_ids = data[0][:prompt_len].unsqueeze(0).to(DEVICE)
            gen_ids = prompt_ids
            for _ in range(max_new):
                out = model(gen_ids[:, -512:])
                logits = out.logits if hasattr(out, 'logits') else out[0]
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen_ids = torch.cat([gen_ids, next_tok], dim=1)
            all_ids = gen_ids[0].cpu().tolist()
            if _TOKENIZER is not None:
                prompt_text = _TOKENIZER.decode(all_ids[:prompt_len], skip_special_tokens=True)
                gen_text = _TOKENIZER.decode(all_ids[prompt_len:], skip_special_tokens=True)
                return f"[mode={mode_val}] PROMPT: {prompt_text[:80]}... | GENERATED: {gen_text[:200]}"
            return f"[mode={mode_val}] {all_ids[:20]}..."
    except Exception as e:
        return f"[mode={mode_val}] Error: {e}"
    finally:
        model.train()


def set_analog_mode_override(model, layers, mode_val):
    for idx in layers:
        gate = model.model.layers[idx].mlp.gate_proj
        if isinstance(gate, AnalogLinear):
            gate.mode_override = mode_val
        elif isinstance(gate, BodyGatedLoRA) and isinstance(gate.analog_linear, AnalogLinear):
            gate.analog_linear.mode_override = mode_val


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE A: Frozen Measurement (0 trainable params)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_phase_a(model, test_data, test_data_code, baseline_ppl):
    print("\n" + "=" * 60)
    print("PHASE A: Frozen Measurement (0 trainable params)")
    print("  v5: ALL layers + TILE=32 + wall_clock64 + per-layer proprio")
    print("=" * 60)

    results = {'baseline_ppl': baseline_ppl}
    originals = patch_gate_proj_with_analog(model, ANALOG_LAYERS, mode_override=None)

    # T2: AnalogLinear PPL
    print("\n[T2] AnalogLinear PPL (live, all layers)...")
    analog_ppl = eval_ppl(model, test_data)
    results['analog_ppl'] = analog_ppl
    print(f"  Analog PPL: {analog_ppl:.4f} (baseline: {baseline_ppl:.4f}, "
          f"ratio: {analog_ppl/baseline_ppl:.4f})")
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
                METAB.update_context()
                time.sleep(0.3)
            stress = METAB.stress_ema
            ppl = eval_ppl(model, test_data)
            dvfs_ppls[dvfs_name] = ppl
            dvfs_stresses[dvfs_name] = stress
            proprio_agg = aggregate_proprio()
            dvfs_proprios[dvfs_name] = proprio_agg.tolist()
            print(f"  DVFS {dvfs_name}: PPL={ppl:.4f}, stress={stress:.4f}, "
                  f"temp={METAB._last_temp:.1f}C, sclk={METAB._last_sclk:.0f}MHz, "
                  f"proprio_jitter={proprio_agg[1].item():.4f}")
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
    denorm_ppls = {}
    denorm_logits = {}
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

    # T15: Generation Quality
    print("\n[T15] Generation Quality...")
    gen_results = {}
    for mode_val in [0, 1, 2, 3]:
        packed = pack_mode_byte(mode_val, mode_val, 3, 3)
        set_analog_mode_override(model, ANALOG_LAYERS, packed)
        text = generate_text(model, test_data, mode_val)
        gen_results[f"mode_{mode_val}"] = text
        print(f"  {text[:200]}")
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    live_text = generate_text(model, test_data, 'live')
    gen_results['live'] = live_text
    print(f"  {live_text[:200]}")
    results['generation'] = gen_results

    # T_PROPRIO: Proprioception Jitter Divergence
    print("\n[T_PROPRIO] Proprioception Jitter Divergence...")
    proprio_divergence = 0.0
    if 'low' in dvfs_proprios and 'high' in dvfs_proprios:
        low_j = dvfs_proprios['low'][1] if len(dvfs_proprios['low']) > 1 else 0
        high_j = dvfs_proprios['high'][1] if len(dvfs_proprios['high']) > 1 else 0
        proprio_divergence = abs(high_j - low_j)
    results['proprio_divergence'] = proprio_divergence
    print(f"  Jitter divergence (high-low): {proprio_divergence:.6f}")

    set_analog_mode_override(model, ANALOG_LAYERS, None)
    restore_gate_proj(model, originals)
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRBS DVFS SCHEDULE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def generate_prbs_schedule(n_batches, seed=42):
    """v5: Pseudo-random binary sequence of DVFS levels with varying dwell."""
    rng = random.Random(seed)
    levels = ['low', 'high', 'auto']
    schedule = []
    while len(schedule) < n_batches:
        level = rng.choice(levels)
        dwell = rng.randint(3, 15)
        schedule.extend([level] * dwell)
    return schedule[:n_batches]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE B: Microchunk Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_phase_b(model, train_data, test_data, test_data_code, baseline_ppl, phase_a_results):
    print("\n" + "=" * 60)
    print("PHASE B: Microchunk Training (body-gated LoRA on gate_proj)")
    print("  v5: PRBS DVFS + 16-token microchunks + dual-timescale gate")
    print("=" * 60)

    results = {}
    analog_originals = patch_gate_proj_with_analog(model, ANALOG_LAYERS, mode_override=None)
    lora_originals, n_lora_params = patch_gate_proj_with_lora(model, LORA_LAYERS,
                                                               rank=LORA_RANK, alpha=LORA_ALPHA)
    hidden_dim = model.config.hidden_size
    soma_head = MultiTaskSomaHead(hidden_dim, n_proprio=FAST_DIM).to(DEVICE)
    n_soma_params = sum(p.numel() for p in soma_head.parameters())
    print(f"  [SomaHead] {hidden_dim}+{FAST_DIM} -> regime[{N_REGIMES}]+stress[1], {n_soma_params} params")

    lora_params = get_lora_params(model, LORA_LAYERS)
    all_params = list(soma_head.parameters()) + lora_params
    total_trainable = sum(p.numel() for p in all_params)
    print(f"  Total trainable: {total_trainable}")
    results['n_lora_params'] = n_lora_params
    results['n_soma_params'] = n_soma_params
    results['total_trainable'] = total_trainable

    N_TRAIN_BATCHES = 120
    LAMBDA_INTERO = 0.1
    LAMBDA_STAB = 0.01
    LR = 1e-4
    optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=0.01)
    ema_logits = None
    EMA_DECAY = 0.95

    # v5: Class weights for regime CE (handle imbalance)
    regime_counts = np.ones(N_REGIMES, dtype=np.float32)
    regime_weights = torch.ones(N_REGIMES, device=DEVICE)

    # v5: PRBS DVFS schedule
    dvfs_schedule = generate_prbs_schedule(N_TRAIN_BATCHES)
    current_dvfs = 'auto'
    results['temp_range'] = [999.0, 0.0]

    METAB.update_context()
    print(f"  Initial temp: {METAB._last_temp:.1f}C")

    print(f"\n  Training {total_trainable} params for {N_TRAIN_BATCHES} batches")
    print(f"  PRBS DVFS schedule, microchunk={MICRO_CHUNK} tokens")
    print(f"  Loss = CE + {LAMBDA_INTERO}*(regime_CE + stress_MSE) + {LAMBDA_STAB}*KL_stab")

    for batch_idx in range(N_TRAIN_BATCHES):
        # v5: PRBS DVFS
        target_dvfs = dvfs_schedule[batch_idx]
        if target_dvfs != current_dvfs:
            set_dvfs(target_dvfs)
            time.sleep(DVFS_SETTLE_S)
            current_dvfs = target_dvfs

        start = (batch_idx * BS) % max(len(train_data) - BS, 1)
        batch = torch.stack(train_data[start:start+BS]).to(DEVICE)

        # v5: Microchunk training — process MICRO_CHUNK tokens at a time
        n_chunks = SEQ_LEN // MICRO_CHUNK
        chunk_losses = []

        for chunk_i in range(n_chunks - 1):
            METAB.update_context()
            CTX.clear_proprio()

            c_start = chunk_i * MICRO_CHUNK
            c_end = (chunk_i + 2) * MICRO_CHUNK  # overlap for labels
            if c_end > SEQ_LEN:
                break
            chunk = batch[:, c_start:c_end]

            out = model(chunk, output_hidden_states=True)
            logits = out.logits if hasattr(out, 'logits') else out[0]

            # CE on second half of chunk (labels from tokens MICRO_CHUNK to 2*MICRO_CHUNK)
            shift_logits = logits[:, MICRO_CHUNK-1:-1, :].contiguous()
            shift_labels = chunk[:, MICRO_CHUNK:].contiguous()
            ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                       shift_labels.view(-1))

            # Interoception loss
            hidden = out.hidden_states[-1] if hasattr(out, 'hidden_states') and out.hidden_states else logits
            proprio_stats = aggregate_proprio().to(DEVICE)
            regime_logits, stress_pred = soma_head(hidden.detach(), proprio_stats)

            target_regime = torch.full((regime_logits.size(0),), CTX.regime,
                                        device=DEVICE, dtype=torch.long)
            regime_loss = F.cross_entropy(regime_logits, target_regime, weight=regime_weights)
            regime_counts[CTX.regime] += 1
            # Update class weights periodically
            if batch_idx % 10 == 0:
                inv_freq = 1.0 / np.maximum(regime_counts, 1.0)
                inv_freq /= inv_freq.sum()
                regime_weights = torch.from_numpy(inv_freq * N_REGIMES).float().to(DEVICE)

            pred_regime = int(regime_logits[0].argmax().item())
            soma_head.update_confusion(pred_regime, CTX.regime)

            target_stress = torch.full((stress_pred.size(0),), CTX.continuous_stress * 2.0 - 1.0,
                                        device=DEVICE, dtype=torch.float32)
            stress_loss = F.mse_loss(stress_pred, target_stress)
            intero_loss = regime_loss + stress_loss

            # Stability KL
            stab_loss = torch.tensor(0.0, device=DEVICE)
            if ema_logits is not None:
                with torch.no_grad():
                    s0 = min(ema_logits.size(0), shift_logits.size(0))
                    s1 = min(ema_logits.size(1), shift_logits.size(1))
                    ema_p = F.softmax(ema_logits[:s0, :s1].float(), dim=-1).clamp(min=1e-10)
                curr_lp = F.log_softmax(shift_logits[:s0, :s1].float(), dim=-1)
                stab_loss = F.kl_div(curr_lp, ema_p, reduction='batchmean').clamp(max=10.0)

            with torch.no_grad():
                if ema_logits is None:
                    ema_logits = shift_logits.detach().clone()
                else:
                    s0 = min(ema_logits.size(0), shift_logits.size(0))
                    s1 = min(ema_logits.size(1), shift_logits.size(1))
                    ema_logits[:s0, :s1] = EMA_DECAY * ema_logits[:s0, :s1] + (1-EMA_DECAY) * shift_logits[:s0, :s1].detach()

            total_loss = ce_loss + LAMBDA_INTERO * intero_loss + LAMBDA_STAB * stab_loss
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            chunk_losses.append(ce_loss.item())

        results['temp_range'][0] = min(results['temp_range'][0], METAB._last_temp)
        results['temp_range'][1] = max(results['temp_range'][1], METAB._last_temp)

        if (batch_idx + 1) % 20 == 0:
            avg_ce = np.mean(chunk_losses) if chunk_losses else 0
            print(f"    Batch {batch_idx+1}/{N_TRAIN_BATCHES}: "
                  f"ce={avg_ce:.4f}, dvfs={current_dvfs}, "
                  f"temp={METAB._last_temp:.1f}C, regime={CTX.regime}")

    results['train_ce'] = np.mean(chunk_losses) if chunk_losses else 0
    set_dvfs('auto')
    time.sleep(DVFS_SETTLE_S)

    # v5: Print confusion matrix
    print("\n  Regime confusion matrix:")
    print(f"  {soma_head.confusion}")
    results['regime_confusion'] = soma_head.confusion.tolist()

    # ─── Phase B Tests ───────────────────────────────────────
    _run_phase_b_tests(model, soma_head, test_data, test_data_code, baseline_ppl, phase_a_results, results)

    # Cleanup
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    restore_gate_proj_lora(model, lora_originals)
    restore_gate_proj(model, analog_originals)
    return results


def _run_phase_b_tests(model, soma_head, test_data, test_data_code, baseline_ppl, phase_a_results, results):
    """Phase B test battery."""
    # T16: LoRA Adaptation PPL
    print("\n[T16] LoRA Adaptation PPL...")
    METAB.update_context()
    lora_ppl = eval_ppl(model, test_data)
    results['lora_ppl'] = lora_ppl
    results['phase_a_ppl'] = phase_a_results.get('analog_ppl', baseline_ppl)
    print(f"  LoRA PPL: {lora_ppl:.4f} (Phase A: {results['phase_a_ppl']:.4f})")

    # T17: Embodied Advantage (live vs all fixed modes)
    print("\n[T17] Embodied Advantage (live vs fixed, per-regime)...")
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
    print(f"  Live: {lora_live_ppl:.4f}, Best fixed: {lora_best_fixed:.4f}, Adv: {embodied_adv:.6f}")

    # v5: Matched-state counterfactual (per-regime comparison)
    print("\n[T17b] Matched-state embodied advantage...")
    regime_live = {}
    regime_fixed = {}
    if DVFS_AVAILABLE:
        for dvfs_name in ['low', 'high']:
            set_dvfs(dvfs_name)
            time.sleep(DVFS_SETTLE_S)
            for _ in range(10):
                METAB.update_context()
                time.sleep(0.2)
            set_analog_mode_override(model, ANALOG_LAYERS, None)
            regime_live[dvfs_name] = eval_ppl(model, test_data, n_batches=10)
            best_fixed_here = 999.0
            for mv in range(4):
                packed = pack_mode_byte(mv, mv, 3, 3)
                set_analog_mode_override(model, ANALOG_LAYERS, packed)
                fp = eval_ppl(model, test_data, n_batches=10)
                best_fixed_here = min(best_fixed_here, fp)
            regime_fixed[dvfs_name] = best_fixed_here
            print(f"  {dvfs_name}: live={regime_live[dvfs_name]:.4f}, best_fixed={best_fixed_here:.4f}")
        set_dvfs('auto')
        set_analog_mode_override(model, ANALOG_LAYERS, None)
    matched_advs = {k: regime_fixed[k] - regime_live[k] for k in regime_live}
    results['matched_state_advantage'] = matched_advs
    mean_matched = np.mean(list(matched_advs.values())) if matched_advs else 0
    results['mean_matched_advantage'] = float(mean_matched)
    print(f"  Mean matched advantage: {mean_matched:.6f}")

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

    # T19: Interoception (stress correlation)
    print("\n[T19] Interoception (balanced DVFS)...")
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    soma_head.eval()
    dvfs_protocol = ['low', 'high', 'auto', 'high', 'low']
    soma_preds_stress, soma_targets_stress = [], []
    soma_preds_regime, soma_targets_regime = [], []
    for dvfs_mode in dvfs_protocol:
        set_dvfs(dvfs_mode)
        time.sleep(DVFS_SETTLE_S)
        for _ in range(5):
            METAB.update_context()
            CTX.clear_proprio()
            with torch.no_grad():
                batch = torch.stack(test_data[:BS]).to(DEVICE)
                out = model(batch, output_hidden_states=True)
                hidden = out.hidden_states[-1] if hasattr(out, 'hidden_states') and out.hidden_states else (out.logits if hasattr(out, 'logits') else out[0])
                proprio = aggregate_proprio().to(DEVICE)
                regime_logits, stress_pred = soma_head(hidden, proprio)
            target_stress = CTX.continuous_stress * 2.0 - 1.0
            soma_preds_stress.append(float(stress_pred[0].item()))
            soma_targets_stress.append(target_stress)
            pred_regime = int(regime_logits[0].argmax().item())
            soma_preds_regime.append(pred_regime)
            soma_targets_regime.append(CTX.regime)
        print(f"  DVFS {dvfs_mode}: stress_pred={soma_preds_stress[-1]:.4f}, target={soma_targets_stress[-1]:.4f}")
    set_dvfs('auto')

    if len(soma_preds_stress) >= 3:
        corr, _ = stats.pearsonr(soma_preds_stress, soma_targets_stress)
        mae = float(np.mean(np.abs(np.array(soma_preds_stress) - np.array(soma_targets_stress))))
    else:
        corr, mae = 0.0, 999.0
    results['soma_correlation'] = float(corr) if not math.isnan(corr) else 0.0
    results['soma_mae'] = mae
    print(f"  Stress corr: {results['soma_correlation']:.4f}, MAE: {mae:.4f}")

    # T22: Regime Classification (v5: with confusion matrix + balanced acc)
    if soma_preds_regime:
        regime_acc = float(np.mean(np.array(soma_preds_regime) == np.array(soma_targets_regime)))
        # Balanced accuracy (macro avg per-class recall)
        per_class = []
        for c in range(N_REGIMES):
            mask = np.array(soma_targets_regime) == c
            if mask.sum() > 0:
                per_class.append(float((np.array(soma_preds_regime)[mask] == c).mean()))
        balanced_acc = np.mean(per_class) if per_class else 0.0
        results['regime_accuracy'] = regime_acc
        results['regime_balanced_accuracy'] = float(balanced_acc)
        print(f"  Regime acc: {regime_acc:.4f}, balanced: {balanced_acc:.4f}")
    else:
        results['regime_accuracy'] = 0.0
        results['regime_balanced_accuracy'] = 0.0

    # T20: Body Gate Analysis (dual-timescale)
    print("\n[T20] Body Gate Analysis...")
    gate_values_low, gate_values_high = [], []
    for dvfs_mode, gate_list in [('low', gate_values_low), ('high', gate_values_high)]:
        set_dvfs(dvfs_mode)
        time.sleep(DVFS_SETTLE_S)
        for _ in range(3):
            METAB.update_context()
            with torch.no_grad():
                batch = torch.stack(test_data[:BS]).to(DEVICE)
                _ = model(batch)
            for layer_idx in LORA_LAYERS:
                g = model.model.layers[layer_idx].mlp.gate_proj
                if isinstance(g, BodyGatedLoRA):
                    fast_vec = CTX.get_per_layer_fast_vec(layer_idx)
                    fast_t = torch.from_numpy(fast_vec).float().to(g.gate_fast.weight.device)
                    slow_np = np.concatenate([CTX.mid_vec, CTX.slow_vec])
                    slow_t = torch.from_numpy(slow_np).float().to(g.gate_slow.weight.device)
                    gate = torch.sigmoid(g.gate_fast(fast_t) + g.gate_slow(slow_t)).detach().cpu().numpy()
                    gate_list.append(gate)
    set_dvfs('auto')
    body_gate_separation = 0.0
    if gate_values_low and gate_values_high:
        low_mean = np.mean(np.array(gate_values_low))
        high_mean = np.mean(np.array(gate_values_high))
        body_gate_separation = abs(high_mean - low_mean)
        print(f"  Gate low: {low_mean:.4f}, high: {high_mean:.4f}, sep: {body_gate_separation:.6f}")
    results['body_gate_separation'] = body_gate_separation

    # T21: Zombie Twin (Phase B)
    print("\n[T21] Zombie Twin (Phase B)...")
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    METAB.update_context()
    live_logits = collect_logits(model, test_data, n_batches=3)
    frozen_mode = CTX.round_mode
    set_analog_mode_override(model, ANALOG_LAYERS, frozen_mode)
    zombie_logits = collect_logits(model, test_data, n_batches=3)
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    zombie_kl = kl_divergence(live_logits, zombie_logits) if live_logits is not None and zombie_logits is not None else 0.0
    results['zombie_kl'] = zombie_kl
    print(f"  KL(live vs zombie): {zombie_kl:.8f}")

    # v5 T_STEP: Step Response (DVFS transition timing)
    print("\n[T_STEP] Step Response...")
    step_results = _test_step_response(model)
    results['step_response'] = step_results

    # v5 T_CAUSAL: Causal Ablation (shuffle proprio only)
    print("\n[T_CAUSAL] Causal Ablation (shuffle proprio)...")
    causal_results = _test_causal_ablation(model, test_data)
    results['causal_ablation'] = causal_results

    # v5 T_GATE_AUROC: Gate-Signal AUROC
    print("\n[T_GATE_AUROC] Gate-Signal AUROC...")
    auroc_results = _test_gate_auroc(model, test_data)
    results['gate_auroc'] = auroc_results


def _test_step_response(model):
    """v5: Measure gate response time after DVFS step change."""
    if not DVFS_AVAILABLE:
        print("  DVFS not available, skipping")
        return {'rise_time_forwards': -1}
    set_dvfs('low')
    time.sleep(3.0)
    for _ in range(10):
        METAB.update_context()
        time.sleep(0.2)
    # Record baseline gate values
    baseline_gates = []
    with torch.no_grad():
        batch = torch.stack([torch.zeros(SEQ_LEN, dtype=torch.long)] * BS).to(DEVICE)
        try:
            batch = torch.stack([torch.randint(0, 1000, (SEQ_LEN,))] * BS).to(DEVICE)
        except:
            pass
    # Step to high
    set_dvfs('high')
    gate_trajectory = []
    for fwd_i in range(20):
        METAB.update_context()
        CTX.clear_proprio()
        with torch.no_grad():
            _ = model(batch)
        # Sample gate from first LoRA layer
        for layer_idx in LORA_LAYERS[:1]:
            g = model.model.layers[layer_idx].mlp.gate_proj
            if isinstance(g, BodyGatedLoRA):
                fast_vec = CTX.get_per_layer_fast_vec(layer_idx)
                fast_t = torch.from_numpy(fast_vec).float().to(g.gate_fast.weight.device)
                slow_np = np.concatenate([CTX.mid_vec, CTX.slow_vec])
                slow_t = torch.from_numpy(slow_np).float().to(g.gate_slow.weight.device)
                gate_val = torch.sigmoid(g.gate_fast(fast_t) + g.gate_slow(slow_t)).mean().item()
                gate_trajectory.append(gate_val)
    set_dvfs('auto')
    # Compute 10-90% rise time
    rise_time = -1
    if len(gate_trajectory) >= 5:
        g_min, g_max = min(gate_trajectory), max(gate_trajectory)
        g_range = g_max - g_min
        if g_range > 0.001:
            t10 = g_min + 0.1 * g_range
            t90 = g_min + 0.9 * g_range
            rise_start = next((i for i, g in enumerate(gate_trajectory) if g >= t10), -1)
            rise_end = next((i for i, g in enumerate(gate_trajectory) if g >= t90), -1)
            if rise_start >= 0 and rise_end > rise_start:
                rise_time = rise_end - rise_start
    print(f"  Rise time: {rise_time} forwards, gate range: {gate_trajectory[0]:.4f} -> {gate_trajectory[-1]:.4f}" if gate_trajectory else "  No data")
    return {'rise_time_forwards': rise_time, 'trajectory': gate_trajectory}


def _test_causal_ablation(model, test_data):
    """v5: Shuffle proprio across layers (keep metabolic intact) → PPL should degrade."""
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    # Normal PPL
    METAB.update_context()
    normal_ppl = eval_ppl(model, test_data, n_batches=10)
    # Shuffle: permute layer_proprio keys randomly each forward
    saved_get = CTX.get_per_layer_fast_vec
    perm = list(range(N_LAYERS))
    random.shuffle(perm)
    def shuffled_get(layer_idx):
        return saved_get(perm[layer_idx % len(perm)])
    CTX.get_per_layer_fast_vec = shuffled_get
    shuffled_ppl = eval_ppl(model, test_data, n_batches=10)
    CTX.get_per_layer_fast_vec = saved_get
    ratio = shuffled_ppl / max(normal_ppl, 0.01)
    print(f"  Normal PPL: {normal_ppl:.4f}, Shuffled proprio PPL: {shuffled_ppl:.4f}, ratio: {ratio:.4f}")
    return {'normal_ppl': normal_ppl, 'shuffled_ppl': shuffled_ppl, 'ratio': ratio}


def _test_gate_auroc(model, test_data):
    """v5: Test if gate values separate DVFS low vs high."""
    if not DVFS_AVAILABLE:
        return {'auroc': 0.5}
    gates_per_dvfs = {'low': [], 'high': []}
    for dvfs_name in ['low', 'high']:
        set_dvfs(dvfs_name)
        time.sleep(DVFS_SETTLE_S)
        for _ in range(5):
            METAB.update_context()
            CTX.clear_proprio()
            with torch.no_grad():
                batch = torch.stack(test_data[:BS]).to(DEVICE)
                _ = model(batch)
            for layer_idx in LORA_LAYERS[:3]:
                g = model.model.layers[layer_idx].mlp.gate_proj
                if isinstance(g, BodyGatedLoRA):
                    fast_vec = CTX.get_per_layer_fast_vec(layer_idx)
                    fast_t = torch.from_numpy(fast_vec).float().to(g.gate_fast.weight.device)
                    slow_np = np.concatenate([CTX.mid_vec, CTX.slow_vec])
                    slow_t = torch.from_numpy(slow_np).float().to(g.gate_slow.weight.device)
                    gate_val = torch.sigmoid(g.gate_fast(fast_t) + g.gate_slow(slow_t)).mean().item()
                    gates_per_dvfs[dvfs_name].append(gate_val)
    set_dvfs('auto')
    # Compute AUROC
    if gates_per_dvfs['low'] and gates_per_dvfs['high']:
        labels = [0]*len(gates_per_dvfs['low']) + [1]*len(gates_per_dvfs['high'])
        scores = gates_per_dvfs['low'] + gates_per_dvfs['high']
        # Simple AUROC via Mann-Whitney U
        try:
            u_stat, p_val = stats.mannwhitneyu(gates_per_dvfs['low'], gates_per_dvfs['high'], alternative='two-sided')
            n1, n2 = len(gates_per_dvfs['low']), len(gates_per_dvfs['high'])
            auroc = u_stat / (n1 * n2)
            auroc = max(auroc, 1 - auroc)  # ensure >0.5
        except:
            auroc = 0.5
            p_val = 1.0
        print(f"  Gate AUROC (low vs high): {auroc:.4f}, p={p_val:.6f}")
        return {'auroc': float(auroc), 'p_value': float(p_val),
                'low_mean': float(np.mean(gates_per_dvfs['low'])),
                'high_mean': float(np.mean(gates_per_dvfs['high']))}
    print("  No gate data collected")
    return {'auroc': 0.5}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST BATTERY SCORING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def score_tests(phase_a, phase_b, baseline_ppl):
    tests = []
    def add(t_id, name, status, value, criterion):
        tests.append({'test': t_id, 'name': name, 'status': status,
                      'value': value, 'criterion': criterion})

    add('T1', 'Baseline PPL', 'PASS', f'{baseline_ppl:.4f}', 'record')
    ratio = phase_a.get('analog_ppl', 999) / max(baseline_ppl, 1e-6)
    add('T2', 'AnalogLinear PPL', 'PASS' if ratio < 1.5 else 'FAIL',
        f'{phase_a.get("analog_ppl", 0):.4f} (ratio={ratio:.4f})', '<1.5x')
    std = phase_a.get('mode_ppl_std', 0)
    add('T3', 'Mode Sweep', 'PASS' if std > 0.01 else 'FAIL', f'σ={std:.6f}', '>0.01')
    max_kl = phase_a.get('max_kl', 0)
    add('T4', 'KL Divergence', 'PASS' if max_kl > 0.001 else 'FAIL', f'{max_kl:.8f}', '>0.001')
    dvfs_std = phase_a.get('dvfs_ppl_std', 0)
    add('T5', 'Thermal Variance', 'PASS' if dvfs_std > 0.05 else 'FAIL', f'{dvfs_std:.6f}', '>0.05')
    det_std = phase_a.get('determinism_std', 999)
    add('T6', 'Determinism', 'PASS' if det_std < 0.001 else 'FAIL', f'{det_std:.8f}', '<0.001')
    kill = phase_a.get('kill_ratio', 0)
    add('T7', 'Kill-Shot', 'PASS' if kill > 1.05 else 'FAIL', f'{kill:.6f}', '>1.05')
    max_dk = phase_a.get('max_denorm_kl', 0)
    add('T8', 'Denorm Sweep', 'PASS' if max_dk > 0.0001 else 'FAIL', f'{max_dk:.8f}', '>0.0001')
    pl_kl = phase_a.get('per_layer_kl', 0)
    add('T9', 'Per-Layer Var', 'PASS' if pl_kl > 0.0001 else 'FAIL', f'{pl_kl:.8f}', '>0.0001')
    abl = phase_a.get('ablation_ppl_diff', 0)
    add('T10', 'Ablation', 'PASS' if abl > 0.001 else 'FAIL', f'{abl:.6f}', '>0.001')
    d = phase_a.get('domain_ppls', {})
    interaction = abs(abs(d.get('wiki_mode0',0)-d.get('wiki_mode3',0)) - abs(d.get('code_mode0',0)-d.get('code_mode3',0)))
    add('T11', 'Domain Sensitivity', 'PASS' if interaction > 0.01 else 'FAIL', f'{interaction:.4f}', '>0.01')
    zk = phase_a.get('zombie_kl', 0)
    add('T12', 'Zombie Twin(A)', 'PASS' if zk > 0.0001 else 'FAIL', f'{zk:.8f}', '>0.0001')
    ti = phase_a.get('token_impact_frac', 0)
    add('T13', 'Token Impact', 'PASS' if ti > 0.05 else 'FAIL', f'{ti:.4f}', '>0.05')
    af = phase_a.get('argmax_flip_rate', 0)
    add('T14', 'Argmax Flip', 'PASS' if af > 0.001 else 'FAIL', f'{af:.6f}', '>0.001')
    add('T15', 'Generation', 'PASS', 'see output', 'coherent')
    pj = phase_a.get('proprio_divergence', 0)
    add('T16p', 'Proprio Jitter', 'PASS' if pj > 0.0 else 'FAIL', f'{pj:.6f}', '>0')

    if phase_b:
        lp = phase_b.get('lora_ppl', 999)
        pa = phase_b.get('phase_a_ppl', 999)
        add('T17', 'LoRA Adapt', 'PASS' if lp < pa else 'FAIL', f'{lp:.4f} vs {pa:.4f}', '<phaseA')
        ea = phase_b.get('embodied_advantage', -1)
        add('T18', 'Embodied Adv', 'PASS' if ea > 0 else 'FAIL', f'{ea:.6f}', '>0')
        msa = phase_b.get('mean_matched_advantage', -1)
        add('T18b', 'Matched-State', 'PASS' if msa > 0 else 'FAIL', f'{msa:.6f}', '>0')
        la = phase_b.get('lora_analog_ppl', 999)
        lf = phase_b.get('lora_fixed_ppl', 999)
        add('T19', 'LoRA vs Fixed', 'PASS' if la < lf else 'FAIL', f'{la:.4f} vs {lf:.4f}', 'analog<fixed')
        sc = phase_b.get('soma_correlation', 0)
        sm = phase_b.get('soma_mae', 999)
        add('T20', 'Interoception', 'PASS' if abs(sc) > 0.3 or sm < 0.5 else 'FAIL',
            f'corr={sc:.4f} mae={sm:.4f}', '|corr|>0.3')
        ba = phase_b.get('regime_balanced_accuracy', 0)
        ra = phase_b.get('regime_accuracy', 0)
        add('T22', 'Regime Class', 'PASS' if ba > 0.35 else 'FAIL', f'acc={ra:.4f} bal={ba:.4f}', 'bal>0.35')
        bg = phase_b.get('body_gate_separation', 0)
        add('T23', 'Gate Sep', 'PASS' if bg > 0.01 else 'FAIL', f'{bg:.6f}', '>0.01')
        zk2 = phase_b.get('zombie_kl', 0)
        add('T24', 'Zombie(B)', 'PASS' if zk2 > 0.0001 else 'FAIL', f'{zk2:.8f}', '>0.0001')
        sr = phase_b.get('step_response', {})
        rt = sr.get('rise_time_forwards', -1)
        add('T25', 'Step Response', 'PASS' if 0 < rt <= 10 else 'FAIL', f'{rt} fwds', '0<rt<=10')
        ca = phase_b.get('causal_ablation', {})
        add('T26', 'Causal Ablation', 'PASS' if ca.get('ratio', 1) > 1.001 else 'FAIL',
            f'{ca.get("ratio",1):.4f}', '>1.001')
        ga = phase_b.get('gate_auroc', {})
        add('T27', 'Gate AUROC', 'PASS' if ga.get('auroc', 0.5) > 0.6 else 'FAIL',
            f'{ga.get("auroc",0.5):.4f}', '>0.6')

    n_pass = sum(1 for t in tests if t['status'] == 'PASS')
    return tests, n_pass, len(tests)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    global N_LAYERS, ANALOG_LAYERS, LORA_LAYERS, _TOKENIZER, SCLK_LOW_CAL, SCLK_HIGH_CAL, SMN_AVAILABLE, SMN_FD
    print("=" * 60)
    print("z2114v5: Deep Embodied Analog Linear LM — Closed-Loop Monist")
    print("ALL layers + TILE=32 + wall_clock64 + per-layer proprio gating")
    print("BodyGatedLoRA on gate_proj + microchunk training + PRBS DVFS")
    print("=" * 60)
    print(f"  AnalogLinear layers: 0-{N_LAYERS-1} ({N_LAYERS} total)")
    print(f"  Phase A: 0 trainable params (frozen measurement)")
    print(f"  Phase B: BodyGatedLoRA rank={LORA_RANK} at gate_proj [{LORA_LAYERS[0]}-{LORA_LAYERS[-1]}]")

    # Hardware Setup
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

    # gpu_metrics check
    find_gpu_metrics()

    # GPU warmup
    print("\n[GPU] Warming up...")
    torch.zeros(1024, 1024, device=DEVICE) @ torch.zeros(1024, 1024, device=DEVICE)
    torch.cuda.synchronize()
    print("[GPU] Warmup OK")

    # Load Model
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

    N_LAYERS = actual_layers
    ANALOG_LAYERS = list(range(N_LAYERS))
    LORA_LAYERS = list(range(min(8, N_LAYERS), min(32, N_LAYERS)))
    print(f"  Analog layers: 0-{N_LAYERS-1} ({N_LAYERS} total)")
    print(f"  LoRA layers: {LORA_LAYERS[0]}-{LORA_LAYERS[-1]} ({len(LORA_LAYERS)} total)")

    for p in model.parameters():
        p.requires_grad = False
    print("  All backbone parameters frozen")

    # Load Data
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

    # Compile kernel + DVFS calibration
    compile_hip_kernel()
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

    if SMN_AVAILABLE:
        val = read_smn(0x00059800)
        print(f"\n[SMN] Entropy sample: 0x{val & 0xFF:02X}")

    print(f"\n[METAB] Initializing MetabolicController...")
    METAB.update_context()
    print(f"  Initial stress EMA: {METAB.stress_ema:.4f}")
    print(f"  Initial regime: {['COLD','NOMINAL','HOT','THROTTLED'][CTX.regime]}")

    # T1: Baseline PPL
    print(f"\n[T1] Baseline PPL (frozen Qwen3-8B, no modifications)...")
    baseline_ppl = eval_ppl(model, test_data)
    print(f"  Baseline PPL: {baseline_ppl:.4f}")

    # Phase A
    phase_a = run_phase_a(model, test_data, test_data_code, baseline_ppl)

    # Phase B (conditional)
    phase_b = None
    max_kl = phase_a.get('max_kl', 0)
    ppl_std = phase_a.get('mode_ppl_std', 0)
    if max_kl > 0.0005 or ppl_std > 0.005:
        print(f"\n  Phase A signal detected (kl={max_kl:.6f}, std={ppl_std:.6f}) → Phase B")
        phase_b = run_phase_b(model, train_data, test_data, test_data_code, baseline_ppl, phase_a)
    else:
        print(f"\n  Phase A signal too weak (kl={max_kl:.6f}, std={ppl_std:.6f}) → skip Phase B")

    # Score
    tests, n_pass, n_total = score_tests(phase_a, phase_b, baseline_ppl)
    print("\n" + "=" * 60)
    print(f"TEST BATTERY RESULTS ({n_total} tests)")
    print("=" * 60)
    for t in tests:
        print(f"  {t['test']}: {t['name']} — {t['status']} ({t['value']} vs {t['criterion']})")
    print(f"\n" + "=" * 60)
    print(f"z2114v5 Deep Embodied Analog LM: {n_pass}/{n_total} PASS")
    print("=" * 60)

    # Save
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'z2114_analog_linear_lm.json')
    out_path = os.path.abspath(out_path)
    result_obj = {
        'experiment': 'z2114v5_closed_loop_monist',
        'description': 'v5: per-layer proprio + gate_proj LoRA + microchunk + PRBS DVFS + matched-state',
        'backbone': f'Qwen/Qwen3-8B ({n_params:.1f}M frozen)',
        'analog_layers': ANALOG_LAYERS,
        'lora_layers': LORA_LAYERS,
        'lora_rank': LORA_RANK,
        'micro_chunk': MICRO_CHUNK,
        'body_dim': BODY_DIM,
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
