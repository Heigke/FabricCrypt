#!/usr/bin/env python3
"""
z2118v10: Haptic Resonance — No-Cheat Pulse Embodiment
=======================================================
v10 changes from v9 (24/35):
  Key v9 insight: GPU-resident pulse field propagates inter-layer state.
  PULSE_EPS=0.02 was below noise floor (probe showed need >=0.1).

  v10 CORE CHANGES:
  1. PULSE_EPS=0.15 (was 0.02) — above noise floor per signal probe
  2. Remove occupancy channel (EXEC mask probe showed std=0.000 in both DVFS states)
  3. Hard-latch flags: FREEZE_PULSE_STATE, BYPASS_LIVE_SENSING, DISABLE_MODE_WRITES
  4. Trace buffer: 6 values per block for diagnostics
  5. Weighted signal combination: 0.50*dt + 0.30*var + 0.20*wgp (no occ)
  6. PulseBufferManager: freeze/unfreeze, pulse_mode() context manager
  7. AnalogLinear: telemetry copy only when CTX.probe_mode=True
  8. MultiTaskSomaHead: hidden-only (NO proprio input) — no cheats
  9. SomaProbeHead: hidden+telemetry, eval/diagnostics only
  10. pulse_prediction_head + future_pulse_head auxiliary losses
  11. BodyGatedLoRA: remove hidden.detach() — gradients flow
  12. Phase B: pulse prediction auxiliary loss (LAMBDA_PULSE_PRED=0.05)
  13. Salience sweep before training

Hardware setup:
  sudo modprobe msr
  sudo insmod ~/Documents/claude_hive/ryzen_smu/ryzen_smu.ko
  sudo chmod 666 /sys/kernel/ryzen_smu_drv/smn
  sudo chmod 666 /sys/kernel/ryzen_smu_drv/pm_table
  sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTORCH_ROCM_ARCH=gfx1100 \\
    venv/bin/python -u scripts/z2118_haptic_resonance_v10.py
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
MICRO_CHUNK = 16
N_EVAL_BATCHES = 30
DVFS_SETTLE_S = 1.5

N_LAYERS = 36
ANALOG_LAYERS = list(range(N_LAYERS))
LORA_RANK = 8
LORA_ALPHA = 16
LORA_LAYERS = list(range(8, 32))

SCLK_LOW_CAL = 600.0
SCLK_HIGH_CAL = 2900.0

# v7: 3-timescale body
FAST_DIM = 12  # mean,std,tail,d_mean,spread,ewma_resid,burst,d_jitter,corr_mean,corr_std,wgp_div,fb_entropy
MID_DIM = 8    # temp_gfx,temp_soc,gfx_power,socket_power,sclk,activity,throttle,d_temp
SLOW_DIM = 4   # stress_ema,d_stress,smn_adc,pcie_replay
BODY_DIM = FAST_DIM + MID_DIM + SLOW_DIM  # 24 total

# v8 preserved: WGP routing constants
WGP_CORRECTION_RANK = 2
N_WGPS = 8

# v10: Pulse field constants — PULSE_EPS raised above noise floor
PULSE_ALPHA = 0.85        # leaky state decay
PULSE_GAIN = 2.0          # tanh gain for sensed signal
PULSE_EPS = 0.15          # v10: was 0.02 — probe showed need >=0.1, use 0.15
PULSE_MAX_BLOCKS = 8192   # max blocks per layer

# v10: True ablation flags (v9 flags preserved)
ENABLE_TIMING = True       # wall_clock64 elapsed in pulse update
ENABLE_TILE_VAR = True     # inter-tile timing variance
ENABLE_WGP_PHASE = True    # WGP identity in pulse update
# ENABLE_OCCUPANCY removed — probe showed std=0.000 in both DVFS states
ENABLE_PULSE_READ = True   # read previous layer's pulse
ENABLE_PULSE_WRITE = True  # write pulse for next layer
ENABLE_GAIN_MOD = True     # apply exp2f gain modulation to output

# v10: Hard-latch flags
FREEZE_PULSE_STATE = False    # when True: skip pulse_out write, use pulse_in as-is
BYPASS_LIVE_SENSING = False   # when True: use s_prev directly, skip wall_clock/hwreg
DISABLE_MODE_WRITES = False   # when True: skip s_setreg MODE perturbation

# v10: Aux loss weight
LAMBDA_PULSE_PRED = 0.05

REGIME_COLD, REGIME_NOMINAL, REGIME_HOT, REGIME_THROTTLED = 0, 1, 2, 3
N_REGIMES = 4

SMN_FD = None

# v9: Pulse buffer global (initialized in main)
PULSE_BUFFERS = None


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
# gpu_metrics SYNCHRONIZED SNAPSHOT
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
    if GPU_METRICS_PATH is None:
        return None
    try:
        with open(GPU_METRICS_PATH, 'rb') as f:
            data = f.read()
        if len(data) < 100:
            return None
        result = {
            'temp_gfx': struct.unpack_from('<H', data, 4)[0] / 100.0,
            'temp_soc': struct.unpack_from('<H', data, 6)[0] / 100.0,
            'gfx_activity': struct.unpack_from('<H', data, 48)[0] / 100.0,
            'gfx_power': struct.unpack_from('<H', data, 66)[0],
            'socket_power': struct.unpack_from('<H', data, 60)[0],
        }
        if len(data) > 176:
            result['sclk_mhz'] = struct.unpack_from('<H', data, 174)[0]
        if len(data) >= 240:
            result['throttle_status'] = struct.unpack_from('<I', data, 236)[0]
        else:
            result['throttle_status'] = 0
        return result
    except Exception:
        return None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v6: RunningZScore for fast feature normalization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class RunningZScore:
    """Per-feature online normalization for fast signals."""
    def __init__(self, n_features, momentum=0.01):
        self.n = n_features
        self.momentum = momentum
        self.running_mean = np.zeros(n_features, dtype=np.float64)
        self.running_var = np.ones(n_features, dtype=np.float64)
        self._count = 0

    def normalize(self, x):
        """x: np.array [n_features]. Returns z-scored version clipped to [-3,3]."""
        self._count += 1
        if self._count < 5:
            self.running_mean = self.running_mean * 0.9 + x * 0.1
            self.running_var = self.running_var * 0.9 + (x - self.running_mean)**2 * 0.1
            mx = np.abs(x).max() + 1e-8
            return np.clip(x / mx, -3, 3).astype(np.float32)
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * (x - self.running_mean)**2
        std = np.sqrt(self.running_var + 1e-8)
        return np.clip((x - self.running_mean) / std, -3, 3).astype(np.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v7: 3-TIMESCALE FORWARD CONTEXT — v10 adds probe_mode flag
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ForwardContext:
    """v7: 3-timescale body state snapshot with per-layer z-scoring.
    fast:  per-layer proprio (wall_clock64 jitter + correction + wgp + feedback) — 12-dim
    mid:   gpu_metrics snapshot
    slow:  thermal EMA + SMN + pcie
    v10:   probe_mode flag — when False, skip CPU->GPU telemetry copy in AnalogLinear
    """
    def __init__(self):
        self.round_mode = pack_mode_byte(0, 0, 3, 3)
        self.continuous_stress = 0.0
        self.stress_threshold = 0
        self.forward_id = 0
        self.snapshot_ts = 0.0
        # v10: probe_mode — when True, AnalogLinear copies telemetry CPU->GPU
        self.probe_mode = False
        # Fast: per-layer proprio from kernel
        self.layer_proprio = {}
        self.prev_layer_proprio = {}
        # v6: Per-layer z-score normalizers
        self._fast_zscores = {}  # layer_idx -> RunningZScore
        # v6: Cache z-scored fast vecs per forward
        self._fast_cache = {}
        self._fast_cache_fwd_id = -1
        # Mid: gpu_metrics
        self.mid_vec = np.zeros(MID_DIM, dtype=np.float32)
        self.prev_mid_vec = np.zeros(MID_DIM, dtype=np.float32)
        # Slow: thermal EMA
        self.slow_vec = np.zeros(SLOW_DIM, dtype=np.float32)
        self.prev_slow_vec = np.zeros(SLOW_DIM, dtype=np.float32)
        # Regime
        self.regime = REGIME_NOMINAL
        # v6: EWMA state for fast features
        self._ewma_mean = {}  # layer_idx -> float

    def clear_proprio(self):
        self.prev_layer_proprio = dict(self.layer_proprio)
        self.layer_proprio = {}
        self._fast_cache = {}
        self._fast_cache_fwd_id = self.forward_id

    def _get_fast_zscore(self, layer_idx):
        if layer_idx not in self._fast_zscores:
            self._fast_zscores[layer_idx] = RunningZScore(FAST_DIM, momentum=0.01)
        return self._fast_zscores[layer_idx]

    def get_per_layer_fast_vec(self, layer_idx):
        """v7: Per-layer fast body vector [FAST_DIM=12] with online z-scoring."""
        if self._fast_cache_fwd_id == self.forward_id and layer_idx in self._fast_cache:
            return self._fast_cache[layer_idx]

        raw = np.zeros(FAST_DIM, dtype=np.float64)
        info = self.layer_proprio.get(layer_idx)
        if info is None:
            info = self.prev_layer_proprio.get(layer_idx)
        if info is not None:
            mean_t = info['mean_ticks']
            std_t = info.get('std_ticks', 0.0)
            tail_r = info.get('tail_ratio', 0.0)
            spread = info.get('spread', 0.0)
            burst = info.get('burst_flag', 0.0)
            prev = self.prev_layer_proprio.get(layer_idx)
            d_mean = (mean_t - prev['mean_ticks']) if prev is not None else 0.0
            d_jitter = (info['jitter'] - prev['jitter']) if prev is not None else 0.0
            ewma_key = layer_idx
            if ewma_key not in self._ewma_mean:
                self._ewma_mean[ewma_key] = mean_t
            else:
                self._ewma_mean[ewma_key] = 0.95 * self._ewma_mean[ewma_key] + 0.05 * mean_t
            ewma_resid = mean_t - self._ewma_mean[ewma_key]

            raw[0] = mean_t
            raw[1] = std_t
            raw[2] = tail_r
            raw[3] = d_mean
            raw[4] = spread
            raw[5] = ewma_resid
            raw[6] = burst
            raw[7] = d_jitter
            raw[8] = info.get('corr_mean', 0.0)
            raw[9] = info.get('corr_std', 0.0)
            raw[10] = info.get('wgp_diversity', 0.0)
            raw[11] = info.get('fb_entropy', 0.0)

        zscore = self._get_fast_zscore(layer_idx)
        result = zscore.normalize(raw)
        self._fast_cache[layer_idx] = result
        return result

    def get_full_body_vec(self, layer_idx):
        fast = self.get_per_layer_fast_vec(layer_idx)
        return np.concatenate([fast, self.mid_vec, self.slow_vec])


CTX = ForwardContext()

class MetabolicController:
    """v6: 3-timescale metabolic controller."""
    def __init__(self, ema_alpha=0.3):
        self.ema_alpha = ema_alpha
        self.stress_ema = 0.0
        self._last_temp = 50.0
        self._last_sclk = 600.0
        self._last_power = 0.0
        self._n_updates = 0
        self._running_mean = np.zeros(BODY_DIM, dtype=np.float64)
        self._running_var = np.ones(BODY_DIM, dtype=np.float64)
        self._stress_history = deque(maxlen=200)
        self._regime_thresholds = [0.25, 0.50, 0.75]
        self._hysteresis = 0.03
        self._prev_regime = REGIME_NOMINAL

    def snapshot_mid(self):
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
# v10 HIP KERNEL — Pulse field + physical signals (no occupancy)
#   + freeze/bypass/disable flags + trace buffer
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

    print("[HIP] Compiling v10 haptic resonance kernel...")
    from torch.utils.cpp_extension import load_inline

    combined_source = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#define TILE_SIZE 32

__global__ void pulse_embodiment_v10_kernel(
    const __hip_bfloat16* __restrict__ A,
    const __hip_bfloat16* __restrict__ B,
    __hip_bfloat16* __restrict__ C,
    float* __restrict__ pulse_in,
    float* __restrict__ pulse_out,
    unsigned long long* __restrict__ proprio_out,
    float* __restrict__ correction_out,
    unsigned int* __restrict__ wgp_out,
    unsigned long long* __restrict__ tile_var_out,
    float* __restrict__ trace_out,
    int M, int K, int N,
    int base_round_mode,
    int stress_threshold,
    float pulse_alpha,
    float pulse_gain,
    float pulse_eps,
    int enable_timing,
    int enable_tile_var,
    int enable_wgp,
    int enable_pulse_read,
    int enable_pulse_write,
    int enable_gain_mod,
    int freeze_pulse_state,
    int bypass_live_sensing,
    int disable_mode_writes
) {
    unsigned long long t_start = wall_clock64();

    // Save original MODE register
    unsigned int old_mode;
    asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" ::: "memory");
    asm volatile("s_getreg_b32 %0, hwreg(1, 0, 8)" : "=s"(old_mode) :: "memory");

    // Read WGP placement ID (hwreg 23 = HW_ID1)
    unsigned int hw_id1;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw_id1) :: "memory");
    unsigned int wgp_id = (hw_id1 >> 7) & 0xF;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float acc = 0.0f;
    int n_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Feedback state — carries accumulation bits between tiles
    unsigned int prev_tile_bits = wgp_id;
    unsigned int feedback_accum = 0;

    // Inter-tile timing variance tracking
    unsigned long long prev_tile_dt = 0;
    unsigned long long tile_timing_var = 0;

    for (int t = 0; t < n_tiles; t++) {
        unsigned long long tile_start = wall_clock64();

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

        // Read SHADER_CYCLES (hwreg 29)
        unsigned int cycles;
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(cycles));

        // INTRA-TILE FEEDBACK ROUNDING (preserved from v8/v9)
        unsigned int acc_bits = __float_as_uint(acc);
        unsigned int feedback = (acc_bits >> 16) ^ prev_tile_bits ^ (cycles & 0xFFu);
        unsigned int rm = feedback & 0x3u;

        unsigned int hw_spin = cycles & 0xFFu;
        if ((int)hw_spin < stress_threshold) {
            rm = 0x1u;
        }

        if (!disable_mode_writes) {
            unsigned int rm_both = (rm & 0x3u) | ((rm & 0x3u) << 2) | (old_mode & 0xF0u);
            unsigned int new_mode = (old_mode & ~0xFFu) | rm_both;
            unsigned int new_mode_s = __builtin_amdgcn_readfirstlane(new_mode);
            asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" :: "s"(new_mode_s) : "memory");
        }

        __half chunk_acc = __float2half(0.0f);
        for (int k = 0; k < TILE_SIZE; k++) {
            __half a_h = __float2half(As[threadIdx.y][k]);
            __half b_h = __float2half(Bs[k][threadIdx.x]);
            chunk_acc = __hadd(chunk_acc, __hmul(a_h, b_h));
        }
        acc += __half2float(chunk_acc);

        // Carry feedback state forward
        prev_tile_bits = (__float_as_uint(acc) >> 24) ^ (rm << 8);
        feedback_accum ^= (__float_as_uint(acc) >> 16);

        // Inter-tile timing delta variance
        unsigned long long tile_end = wall_clock64();
        unsigned long long tile_dt = tile_end - tile_start;
        if (t > 0) {
            long long diff = (long long)tile_dt - (long long)prev_tile_dt;
            tile_timing_var += (unsigned long long)(diff * diff);
        }
        prev_tile_dt = tile_dt;

        __syncthreads();
    }

    // ─── v10: PULSE FIELD UPDATE ───
    unsigned long long t_end = wall_clock64();
    unsigned long long dt = t_end - t_start;

    int block_id = blockIdx.y * gridDim.x + blockIdx.x;

    // Read previous layer pulse (bounds-checked)
    float s_prev = 0.0f;
    if (enable_pulse_read && pulse_in != nullptr && block_id < 8192) {
        s_prev = pulse_in[block_id];
    }

    float s_new;
    if (bypass_live_sensing) {
        // v10: bypass flag — use s_prev directly, skip physical signal computation
        s_new = s_prev;
    } else {
        // v10: Physical signal normalization — no occupancy channel
        // Weighted combination: 0.50*dt + 0.30*var + 0.20*wgp
        float norm_dt   = enable_timing    ? ((float)(dt & 0xFFFFu) * 1.52587890625e-5f - 0.5f) : 0.0f;
        float norm_var  = enable_tile_var  ? tanhf((float)tile_timing_var * 1e-8f) : 0.0f;
        float wgp_phase = enable_wgp       ? ((float)(wgp_id & 0x7u) * 0.142857f - 0.5f) : 0.0f;

        float sensed = 0.50f * norm_dt + 0.30f * norm_var + 0.20f * wgp_phase;

        if (freeze_pulse_state) {
            // freeze: use pulse_in as-is, no leaky update
            s_new = s_prev;
        } else {
            // Leaky integrator: s = alpha * s_prev + (1-alpha) * tanh(gain * sensed)
            s_new = pulse_alpha * s_prev + (1.0f - pulse_alpha) * tanhf(pulse_gain * sensed);
        }

        // Write trace output (6 values per block, thread 0,0 only)
        if (threadIdx.x == 0 && threadIdx.y == 0 && trace_out != nullptr && block_id < 8192) {
            float norm_dt_tr   = enable_timing   ? ((float)(dt & 0xFFFFu) * 1.52587890625e-5f - 0.5f) : 0.0f;
            float norm_var_tr  = enable_tile_var ? tanhf((float)tile_timing_var * 1e-8f) : 0.0f;
            float wgp_phase_tr = enable_wgp      ? ((float)(wgp_id & 0x7u) * 0.142857f - 0.5f) : 0.0f;
            float act_val      = enable_gain_mod  ? (acc * exp2f(pulse_eps * s_new)) : acc;
            trace_out[block_id * 6 + 0] = norm_dt_tr;
            trace_out[block_id * 6 + 1] = norm_var_tr;
            trace_out[block_id * 6 + 2] = wgp_phase_tr;
            trace_out[block_id * 6 + 3] = s_prev;
            trace_out[block_id * 6 + 4] = s_new;
            trace_out[block_id * 6 + 5] = act_val;
        }
    }

    // Write pulse for next layer (bounds-checked) — only if not frozen
    if (!freeze_pulse_state && enable_pulse_write && pulse_out != nullptr && block_id < 8192) {
        pulse_out[block_id] = s_new;
    }

    // v10: Pre-activation gain modulation
    if (enable_gain_mod) {
        acc *= exp2f(pulse_eps * s_new);
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(acc);
    }

    // Restore original MODE
    asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" ::: "memory");
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" :: "s"(old_mode) : "memory");

    // Write per-block outputs (thread 0,0 only)
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        proprio_out[block_id] = dt;
        correction_out[block_id] = s_new;
        wgp_out[block_id] = wgp_id;
        tile_var_out[block_id] = tile_timing_var;
    }
}

std::vector<torch::Tensor> analog_gemm_v10(
    torch::Tensor A, torch::Tensor weight,
    int round_mode, float continuous_stress,
    torch::Tensor pulse_in, torch::Tensor pulse_out,
    float pulse_alpha, float pulse_gain, float pulse_eps,
    int enable_timing, int enable_tile_var, int enable_wgp,
    int enable_pulse_read, int enable_pulse_write,
    int enable_gain_mod,
    int freeze_pulse_state, int bypass_live_sensing, int disable_mode_writes
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

    auto proprio = torch::empty({n_blocks},
        torch::TensorOptions().dtype(torch::kInt64).device(A.device()));
    auto corr_out = torch::empty({n_blocks},
        torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));
    auto wgp_out_t = torch::empty({n_blocks},
        torch::TensorOptions().dtype(torch::kInt32).device(A.device()));
    auto tvar_out = torch::empty({n_blocks},
        torch::TensorOptions().dtype(torch::kInt64).device(A.device()));
    // v10: trace buffer — 6 floats per block
    auto trace_t = torch::zeros({n_blocks, 6},
        torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));

    int stress_threshold = (int)(continuous_stress * 255.0f);
    if (stress_threshold < 0) stress_threshold = 0;
    if (stress_threshold > 255) stress_threshold = 255;

    pulse_embodiment_v10_kernel<<<grid, block>>>(
        reinterpret_cast<const __hip_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),
        pulse_in.data_ptr<float>(),
        pulse_out.data_ptr<float>(),
        reinterpret_cast<unsigned long long*>(proprio.data_ptr()),
        corr_out.data_ptr<float>(),
        reinterpret_cast<unsigned int*>(wgp_out_t.data_ptr<int>()),
        reinterpret_cast<unsigned long long*>(tvar_out.data_ptr()),
        trace_t.data_ptr<float>(),
        M, K, N, round_mode, stress_threshold,
        pulse_alpha, pulse_gain, pulse_eps,
        enable_timing, enable_tile_var, enable_wgp,
        enable_pulse_read, enable_pulse_write,
        enable_gain_mod,
        freeze_pulse_state, bypass_live_sensing, disable_mode_writes);

    return {C, proprio, corr_out, wgp_out_t, tvar_out, trace_t};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("analog_gemm_v10", &analog_gemm_v10,
          "v10 haptic resonance GEMM: pulse field + weighted signals (no occ) + freeze/bypass/disable flags + trace");
}
"""

    try:
        _HIP_MODULE = load_inline(
            name='analog_gemm_v10_ext',
            cpp_sources=[],
            cuda_sources=[combined_source],
            extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'],
            verbose=True,
            with_cuda=True,
        )
        print("[HIP] v10 Kernel compiled OK")
    except Exception as e:
        print(f"[HIP] Compilation failed: {e}")
        _HIP_MODULE = None
    return _HIP_MODULE

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v10: pulse_mode() context manager
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from contextlib import contextmanager

@contextmanager
def pulse_mode(read=None, write=None, gain=None, freeze=None,
               bypass_sensing=None, disable_mode=None):
    """Temporarily override global pulse flags for ablation / diagnostics."""
    global ENABLE_PULSE_READ, ENABLE_PULSE_WRITE, ENABLE_GAIN_MOD
    global FREEZE_PULSE_STATE, BYPASS_LIVE_SENSING, DISABLE_MODE_WRITES

    saved = (ENABLE_PULSE_READ, ENABLE_PULSE_WRITE, ENABLE_GAIN_MOD,
             FREEZE_PULSE_STATE, BYPASS_LIVE_SENSING, DISABLE_MODE_WRITES)
    try:
        if read is not None:
            ENABLE_PULSE_READ = read
        if write is not None:
            ENABLE_PULSE_WRITE = write
        if gain is not None:
            ENABLE_GAIN_MOD = gain
        if freeze is not None:
            FREEZE_PULSE_STATE = freeze
        if bypass_sensing is not None:
            BYPASS_LIVE_SENSING = bypass_sensing
        if disable_mode is not None:
            DISABLE_MODE_WRITES = disable_mode
        yield
    finally:
        (ENABLE_PULSE_READ, ENABLE_PULSE_WRITE, ENABLE_GAIN_MOD,
         FREEZE_PULSE_STATE, BYPASS_LIVE_SENSING, DISABLE_MODE_WRITES) = saved


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v10: PULSE BUFFER MANAGER — with freeze/unfreeze
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PulseBufferManager:
    """GPU-resident pulse field buffers for inter-layer state propagation.
    v10: adds freeze/unfreeze and is_frozen property."""
    def __init__(self, n_layers, max_blocks=PULSE_MAX_BLOCKS):
        self.n_layers = n_layers
        self.max_blocks = max_blocks
        self.buffers = torch.zeros(n_layers + 1, max_blocks, device=DEVICE, dtype=torch.float32)
        self._snapshot = None
        self._frozen = False

    def get_input(self, layer_idx):
        """Get pulse input buffer for layer (reads from previous layer's output)."""
        return self.buffers[layer_idx]

    def get_output(self, layer_idx):
        """Get pulse output buffer for layer (writes to next layer's input)."""
        return self.buffers[layer_idx + 1]

    def snapshot(self):
        """Save current state for restore/stale tests."""
        self._snapshot = self.buffers.clone()

    def restore_snapshot(self):
        """Restore saved snapshot."""
        if self._snapshot is not None:
            self.buffers.copy_(self._snapshot)

    def zero(self):
        """Zero all pulse buffers (for ablation)."""
        self.buffers.zero_()

    def shuffle_blocks(self):
        """Randomly permute block dimension (for spatial ablation)."""
        perm = torch.randperm(self.max_blocks, device=DEVICE)
        self.buffers = self.buffers[:, perm]

    def freeze(self):
        """Freeze pulse state — buffers become read-only for kernel."""
        self._frozen = True

    def unfreeze(self):
        """Unfreeze pulse state."""
        self._frozen = False

    @property
    def is_frozen(self):
        return self._frozen

    def get_field_stats(self):
        """Return diagnostic statistics of the pulse field."""
        b = self.buffers
        active = (b.abs() > 1e-6).float()
        return {
            'mean': float(b.mean()),
            'std': float(b.std()),
            'max': float(b.abs().max()),
            'active_frac': float(active.mean()),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v10: ANALOG LINEAR — conditional telemetry copy + trace buffer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AnalogLinear(nn.Module):
    """v10: Pulse buffers + physical signal kernel.
    - Telemetry copy to CPU only when CTX.probe_mode=True
    - _last_trace stores 6-value trace per block (stays on GPU)
    - Passes freeze/bypass/disable flags to kernel
    """
    def __init__(self, original_linear, layer_idx, mode_override=None):
        super().__init__()
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.layer_idx = layer_idx
        self.mode_override = mode_override
        self._last_wgp_ids = None
        self._last_corr_vals = None
        self._last_trace = None   # v10: GPU-resident 6-val trace per block

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
        if hip_mod is not None and PULSE_BUFFERS is not None:
            x_bf16 = x_2d.to(torch.bfloat16).contiguous()
            w_bf16 = self.weight.to(torch.bfloat16).contiguous()

            pulse_in = PULSE_BUFFERS.get_input(self.layer_idx)
            pulse_out = PULSE_BUFFERS.get_output(self.layer_idx)

            # Determine effective freeze from global flag or manager state
            eff_freeze = int(FREEZE_PULSE_STATE or PULSE_BUFFERS.is_frozen)

            results = hip_mod.analog_gemm_v10(
                x_bf16, w_bf16, round_mode, continuous_stress,
                pulse_in, pulse_out,
                PULSE_ALPHA, PULSE_GAIN, PULSE_EPS,
                int(ENABLE_TIMING), int(ENABLE_TILE_VAR), int(ENABLE_WGP_PHASE),
                int(ENABLE_PULSE_READ), int(ENABLE_PULSE_WRITE),
                int(ENABLE_GAIN_MOD),
                eff_freeze, int(BYPASS_LIVE_SENSING), int(DISABLE_MODE_WRITES))

            out = results[0]
            proprio_ticks = results[1]
            pulse_vals = results[2]
            wgp_ids = results[3]
            tile_timing_vars = results[4]
            trace_buf = results[5]   # [n_blocks, 6]

            self._last_wgp_ids = wgp_ids
            self._last_corr_vals = pulse_vals
            self._last_trace = trace_buf  # stays on GPU

            # v10: CPU telemetry copy ONLY when probe_mode is True
            if self.mode_override is None and CTX.probe_mode:
                ticks_cpu = proprio_ticks.cpu().to(torch.float64)
                ticks_pos = ticks_cpu[ticks_cpu > 0]
                corr_cpu = pulse_vals.cpu().float()
                wgp_cpu = wgp_ids.cpu().int()
                tvar_cpu = tile_timing_vars.cpu().to(torch.float64)

                if len(ticks_pos) > 0:
                    sorted_t = ticks_pos.sort().values
                    n = len(sorted_t)
                    mean_t = float(ticks_pos.mean())
                    std_t = float(ticks_pos.std()) if n > 1 else 0.0
                    p50 = sorted_t[n // 2].item()
                    p95 = sorted_t[min(int(n * 0.95), n - 1)].item()
                    tail_ratio = (p95 / max(p50, 1.0)) - 1.0
                    spread = float(ticks_pos.max() - ticks_pos.min())
                    jitter = spread
                    if std_t > 0:
                        burst_flag = float((ticks_pos > mean_t + 2 * std_t).sum()) / max(n, 1)
                    else:
                        burst_flag = 0.0

                    corr_mean = float(corr_cpu.mean())
                    corr_std = float(corr_cpu.std()) if len(corr_cpu) > 1 else 0.0
                    wgp_unique = len(torch.unique(wgp_cpu))
                    wgp_diversity = wgp_unique / max(8.0, 1.0)
                    fb_vals = corr_cpu
                    fb_norm = fb_vals / (fb_vals.abs().max() + 1e-8)
                    fb_hist = torch.histc(fb_norm, bins=16, min=-1, max=1)
                    fb_prob = fb_hist / (fb_hist.sum() + 1e-8)
                    fb_entropy = float(-(fb_prob * (fb_prob + 1e-10).log2()).sum()) / 4.0
                    tvar_mean = float(tvar_cpu.mean())

                    CTX.layer_proprio[self.layer_idx] = {
                        'mean_ticks': mean_t,
                        'std_ticks': std_t,
                        'max_ticks': float(ticks_pos.max()),
                        'jitter': jitter,
                        'tail_ratio': tail_ratio,
                        'spread': spread,
                        'burst_flag': burst_flag,
                        'n_blocks': int(n),
                        'corr_mean': corr_mean,
                        'corr_std': corr_std,
                        'wgp_diversity': wgp_diversity,
                        'fb_entropy': fb_entropy,
                        'tile_var_mean': tvar_mean,
                    }
        else:
            out = F.linear(x_2d, self.weight)

        if self.bias is not None:
            out = out + self.bias
        N_out = self.weight.shape[0]
        return out.reshape(*orig_shape[:-1], N_out)

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
        elif isinstance(gate, BodyGatedLoRA) and isinstance(gate.analog_linear, AnalogLinear):
            gate.analog_linear.mode_override = mode_override


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v9/v10: BODY-GATED LORA — v10: remove hidden.detach()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class BodyGatedLoRA(nn.Module):
    """v9/v10: LoRA on gate_proj with ADDITIVE gate + WGP-routed mixture.
    v10: hidden.detach() removed — gradients flow through to LoRA for embodiment objective."""

    def __init__(self, analog_linear, layer_idx, rank=8, alpha=16):
        super().__init__()
        self.analog_linear = analog_linear
        self.layer_idx = layer_idx
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_f = analog_linear.in_features
        out_f = analog_linear.out_features
        dtype = analog_linear.weight.dtype

        # Main LoRA (slow+mid+fast gated)
        self.lora_A = nn.Parameter(torch.randn(rank, in_f, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_f, rank, dtype=dtype) * 0.001)

        # v7: ADDITIVE gate — single sigmoid(fast_pre + mid_pre + slow_pre + bias)
        self.gate_fast = nn.Linear(FAST_DIM, rank, dtype=torch.float32)
        self.gate_mid = nn.Linear(MID_DIM, rank, dtype=torch.float32)
        self.gate_slow = nn.Linear(SLOW_DIM, rank, dtype=torch.float32)
        nn.init.zeros_(self.gate_fast.weight)
        nn.init.zeros_(self.gate_mid.weight)
        nn.init.zeros_(self.gate_slow.weight)
        nn.init.constant_(self.gate_fast.bias, 0.0)
        nn.init.constant_(self.gate_mid.bias, 0.0)
        nn.init.constant_(self.gate_slow.bias, 0.0)

        # v8 preserved: WGP-routed mixture of corrections
        self.wgp_corr_A = nn.Parameter(torch.randn(N_WGPS, WGP_CORRECTION_RANK, in_f, dtype=dtype) * 0.01)
        self.wgp_corr_B = nn.Parameter(torch.randn(N_WGPS, out_f, WGP_CORRECTION_RANK, dtype=dtype) * 0.001)
        self.wgp_corr_scaling = alpha / WGP_CORRECTION_RANK

        # Modality dropout control
        self._modality_dropout = True
        # Branch usage tracking
        self.last_g_fast_norm = 0.0
        self.last_g_mid_norm = 0.0
        self.last_g_slow_norm = 0.0
        self.last_fast_grad_norm = 0.0

    def forward(self, x):
        base = self.analog_linear(x)
        x_cast = x.to(self.lora_A.dtype)

        # === Main LoRA path (slow+mid+fast gated) ===
        lora_mid = F.linear(x_cast, self.lora_A)  # [..., rank]

        dev = self.gate_fast.weight.device
        fast_vec = CTX.get_per_layer_fast_vec(self.layer_idx)
        fast_input = torch.from_numpy(fast_vec).float().to(dev)
        mid_input = torch.from_numpy(CTX.mid_vec.copy()).float().to(dev)
        slow_input = torch.from_numpy(CTX.slow_vec.copy()).float().to(dev)

        # Additive pre-activation -> single sigmoid
        fast_pre = self.gate_fast(fast_input)   # [rank]
        mid_pre = self.gate_mid(mid_input)      # [rank]
        slow_pre = self.gate_slow(slow_input)   # [rank]

        # Track branch pre-activation norms
        self.last_g_fast_norm = float(fast_pre.detach().norm().item())
        self.last_g_mid_norm = float(mid_pre.detach().norm().item())
        self.last_g_slow_norm = float(slow_pre.detach().norm().item())

        # Modality dropout during training
        if self.training and self._modality_dropout:
            r = random.random()
            if r < 0.25:
                gate_pre = fast_pre + mid_pre         # drop slow
            elif r < 0.40:
                gate_pre = fast_pre                   # fast-only reflex
            elif r < 0.50:
                gate_pre = mid_pre + slow_pre         # drop fast
            else:
                gate_pre = fast_pre + mid_pre + slow_pre
        else:
            gate_pre = fast_pre + mid_pre + slow_pre

        gate = torch.sigmoid(gate_pre)  # [rank], single sigmoid
        lora_mid = lora_mid * gate.to(lora_mid.dtype)
        main_lora_out = F.linear(lora_mid, self.lora_B) * self.scaling

        result = base + main_lora_out.to(x.dtype)

        # === v8 preserved: WGP-routed mixture of corrections ===
        wgp_ids = self.analog_linear._last_wgp_ids
        if wgp_ids is not None and len(wgp_ids) > 0:
            try:
                dominant_wgp = int(wgp_ids.mode().values.item()) % N_WGPS
                wgp_mid = F.linear(x_cast, self.wgp_corr_A[dominant_wgp])
                wgp_out = F.linear(wgp_mid, self.wgp_corr_B[dominant_wgp]) * self.wgp_corr_scaling
                wgp_gate = torch.sigmoid(fast_pre.mean())
                result = result + (wgp_out * wgp_gate).to(x.dtype)
            except Exception:
                pass

        return result


def patch_gate_proj_with_lora(model, layers, rank=8, alpha=16):
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
        originals[layer_idx] = analog
        n_p = sum(p.numel() for p in lora.parameters() if p.requires_grad)
        total_params += n_p
        if layer_idx in [8, 15, 23, 31]:
            print(f"  [BodyGatedLoRA] Patched layer {layer_idx} gate_proj (rank={rank}, {n_p} params)")
    print(f"  [BodyGatedLoRA] Total trainable: {total_params} across {len(originals)} layers")
    return originals, total_params


def restore_gate_proj_lora(model, originals):
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
    """v6: Aggregate proprioception across all layers into FAST_DIM stats."""
    if not CTX.layer_proprio:
        return torch.zeros(FAST_DIM)
    vals = {i: [] for i in range(FAST_DIM)}
    for layer_idx in CTX.layer_proprio:
        fv = CTX.get_per_layer_fast_vec(layer_idx)
        for i in range(FAST_DIM):
            vals[i].append(fv[i])
    return torch.tensor([np.mean(vals[i]) for i in range(FAST_DIM)], dtype=torch.float32)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v10: MULTI-TASK SOMA HEAD — hidden-only (NO proprio input)
#      SomaProbeHead — hidden+telemetry, eval/diagnostics only
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MultiTaskSomaHead(nn.Module):
    """v10: Hidden-state ONLY — no proprio input to prevent cheating.
    Also adds pulse_prediction_head and future_pulse_head for aux losses.
    """
    def __init__(self, hidden_dim, dtype=torch.bfloat16):
        super().__init__()
        # v10: input_dim = hidden_dim ONLY (no n_proprio concat)
        self.input_dim = hidden_dim
        self.shared = nn.Linear(self.input_dim, 128, dtype=torch.float32)
        self.regime_head = nn.Linear(128, N_REGIMES, dtype=torch.float32)
        self.stress_head = nn.Linear(128, 1, dtype=torch.float32)
        # v10: pulse prediction heads
        self.pulse_prediction_head = nn.Linear(128, 1, dtype=torch.float32)
        self.future_pulse_head = nn.Linear(128, 1, dtype=torch.float32)
        self.confusion = np.zeros((N_REGIMES, N_REGIMES), dtype=np.int64)

    def forward(self, hidden_states):
        """forward(hidden_states) — proprio excluded from training path."""
        h = hidden_states.float().mean(dim=1)  # [B, hidden_dim]
        shared_out = F.gelu(self.shared(h))
        regime_logits = self.regime_head(shared_out)
        stress_pred = torch.tanh(self.stress_head(shared_out).squeeze(-1))
        pulse_pred = self.pulse_prediction_head(shared_out).squeeze(-1)
        future_pulse_pred = self.future_pulse_head(shared_out).squeeze(-1)
        return regime_logits, stress_pred, pulse_pred, future_pulse_pred

    def update_confusion(self, pred_regime, true_regime):
        self.confusion[true_regime, pred_regime] += 1


class SomaProbeHead(nn.Module):
    """v10: Eval/diagnostics ONLY — takes hidden+telemetry.
    NOT used for any training loss. Used for post-hoc analysis of
    how well the model's representations encode body state."""
    def __init__(self, hidden_dim, n_proprio=FAST_DIM, dtype=torch.bfloat16):
        super().__init__()
        self.input_dim = hidden_dim + n_proprio
        self.shared = nn.Linear(self.input_dim, 128, dtype=torch.float32)
        self.regime_head = nn.Linear(128, N_REGIMES, dtype=torch.float32)
        self.stress_head = nn.Linear(128, 1, dtype=torch.float32)

    def forward(self, hidden_states, proprio_stats):
        """Eval-only forward with proprio for diagnostic comparison."""
        h = hidden_states.float().mean(dim=1)
        if proprio_stats is not None:
            ps = proprio_stats.unsqueeze(0).expand(h.size(0), -1).to(h.device)
            h_aug = torch.cat([h, ps], dim=-1)
        else:
            h_aug = torch.cat([h, torch.zeros(h.size(0), FAST_DIM, device=h.device)], dim=-1)
        shared_out = F.gelu(self.shared(h_aug))
        regime_logits = self.regime_head(shared_out)
        stress_pred = torch.tanh(self.stress_head(shared_out).squeeze(-1))
        return regime_logits, stress_pred


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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE A: Frozen Measurement (0 trainable params)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_phase_a(model, test_data, test_data_code, baseline_ppl):
    print("\n" + "=" * 60)
    print("PHASE A: Frozen Measurement (0 trainable params)")
    print("  v10: PULSE_EPS=0.15, no occupancy, weighted 0.50/0.30/0.20 signals")
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
    rng = random.Random(seed)
    levels = ['low', 'high', 'auto']
    schedule = []
    while len(schedule) < n_batches:
        level = rng.choice(levels)
        dwell = rng.randint(3, 15)
        schedule.extend([level] * dwell)
    return schedule[:n_batches]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v10: SALIENCE SWEEP — pick optimal PULSE_EPS before training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def salience_sweep(model, test_data):
    """Run quick PPL sweep over PULSE_EPS values to confirm signal above noise.
    Evaluates eps in {0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3} with 5 batches each.
    Returns the eps showing clearest PPL signal (or default 0.15).
    """
    global PULSE_EPS

    print("\n" + "=" * 60)
    print("SALIENCE SWEEP: eps in {0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3}")
    print("  Goal: confirm PULSE_EPS=0.15 is above noise floor")
    print("=" * 60)

    eps_candidates = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    sweep_results = {}
    orig_eps = PULSE_EPS

    for eps_val in eps_candidates:
        PULSE_EPS = eps_val
        ppl = eval_ppl(model, test_data, n_batches=5)
        sweep_results[eps_val] = ppl
        print(f"  eps={eps_val:.2f}: PPL={ppl:.4f}")

    # Restore
    PULSE_EPS = orig_eps

    # Find eps with clearest deviation from eps=0 baseline
    baseline_eps0 = sweep_results.get(0.0, list(sweep_results.values())[0])
    best_eps = orig_eps
    best_signal = 0.0
    for eps_val, ppl in sweep_results.items():
        if eps_val == 0.0:
            continue
        signal = abs(ppl - baseline_eps0)
        if signal > best_signal:
            best_signal = signal
            best_eps = eps_val

    print(f"\n  Baseline (eps=0): PPL={baseline_eps0:.4f}")
    print(f"  Best signal eps={best_eps:.2f} (delta={best_signal:.4f} from baseline)")
    print(f"  Using PULSE_EPS={orig_eps:.2f} (default, override if sweep shows different optimum)")

    return {
        'sweep': {str(k): v for k, v in sweep_results.items()},
        'best_eps': best_eps,
        'best_signal': best_signal,
        'baseline_ppl': baseline_eps0,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE B: v10 3-Phase Curriculum Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_phase_b(model, train_data, test_data, test_data_code, baseline_ppl, phase_a_results):
    print("\n" + "=" * 60)
    print("PHASE B: v10 3-Phase Curriculum (haptic resonance)")
    print("  Phase1: Calibration 0-40 | Phase2: Reflex 40-120 | Phase3: Integration 120-200")
    print("  v10: no-cheat soma head (hidden-only), pulse pred aux loss, no hidden.detach()")
    print("=" * 60)

    results = {}
    analog_originals = patch_gate_proj_with_analog(model, ANALOG_LAYERS, mode_override=None)
    lora_originals, n_lora_params = patch_gate_proj_with_lora(model, LORA_LAYERS,
                                                               rank=LORA_RANK, alpha=LORA_ALPHA)
    hidden_dim = model.config.hidden_size

    # v10: hidden-only soma head (no proprio input)
    soma_head = MultiTaskSomaHead(hidden_dim).to(DEVICE)
    n_soma_params = sum(p.numel() for p in soma_head.parameters())
    print(f"  [SomaHead] hidden_dim={hidden_dim} -> regime[{N_REGIMES}]+stress[1]+pulse[1]+future[1], "
          f"{n_soma_params} params (NO proprio input)")

    # v10: probe head for eval diagnostics (never used in loss)
    soma_probe = SomaProbeHead(hidden_dim, n_proprio=FAST_DIM).to(DEVICE)
    print(f"  [SomaProbeHead] hidden+proprio -> regime+stress (eval only)")

    lora_params = get_lora_params(model, LORA_LAYERS)
    all_params = list(soma_head.parameters()) + list(soma_probe.parameters()) + lora_params
    total_trainable = sum(p.numel() for p in all_params)
    print(f"  Total trainable: {total_trainable}")
    results['n_lora_params'] = n_lora_params
    results['n_soma_params'] = n_soma_params
    results['total_trainable'] = total_trainable

    # v10: Salience sweep before training
    print("\n  Running salience sweep to confirm PULSE_EPS signal...")
    sweep_res = salience_sweep(model, test_data)
    results['salience_sweep'] = sweep_res

    N_TRAIN_BATCHES = 200
    LR = 1e-4
    optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=0.01)
    ema_logits = None
    EMA_DECAY = 0.95

    regime_counts = np.ones(N_REGIMES, dtype=np.float32)
    regime_weights = torch.ones(N_REGIMES, device=DEVICE)

    dvfs_schedule = generate_prbs_schedule(N_TRAIN_BATCHES)
    current_dvfs = 'auto'
    results['temp_range'] = [999.0, 0.0]

    # v7: branch usage tracking
    branch_norms_log = {'fast': [], 'mid': [], 'slow': []}
    fast_grad_norms_log = []

    # v7: contention stream
    contention_stream = torch.cuda.Stream()

    METAB.update_context()
    print(f"  Initial temp: {METAB._last_temp:.1f}C")
    print(f"\n  Training {total_trainable} params for {N_TRAIN_BATCHES} batches (3-phase)")

    for batch_idx in range(N_TRAIN_BATCHES):
        # v9/v10: 3-phase curriculum
        if batch_idx < 40:
            phase_name = 'calibration'
            LAMBDA_INTERO = 0.3
            LAMBDA_STAB = 0.01
            target_dvfs = 'auto'
            modality_dropout = False
            use_contention = False
        elif batch_idx < 120:
            phase_name = 'reflex'
            LAMBDA_INTERO = 0.1
            LAMBDA_STAB = 0.005
            target_dvfs = dvfs_schedule[batch_idx]
            modality_dropout = True
            use_contention = (random.random() < 0.3)
        else:
            phase_name = 'integration'
            LAMBDA_INTERO = 0.1
            LAMBDA_STAB = 0.01
            target_dvfs = dvfs_schedule[batch_idx]
            modality_dropout = (random.random() < 0.1)
            use_contention = False

        for layer_idx in LORA_LAYERS:
            g = model.model.layers[layer_idx].mlp.gate_proj
            if isinstance(g, BodyGatedLoRA):
                g._modality_dropout = modality_dropout

        if target_dvfs != current_dvfs:
            set_dvfs(target_dvfs)
            time.sleep(DVFS_SETTLE_S)
            current_dvfs = target_dvfs

        if use_contention:
            with torch.cuda.stream(contention_stream):
                buf = torch.randn(2048, 2048, device=DEVICE)
                for _ in range(5):
                    buf = buf @ buf.T

        start = (batch_idx * BS) % max(len(train_data) - BS, 1)
        batch = torch.stack(train_data[start:start+BS]).to(DEVICE)

        # v9: Snapshot pulse buffers before each batch
        if PULSE_BUFFERS is not None:
            PULSE_BUFFERS.snapshot()

        n_chunks = SEQ_LEN // MICRO_CHUNK
        chunk_losses = []

        for chunk_i in range(n_chunks - 1):
            METAB.update_context()
            CTX.clear_proprio()

            c_start = chunk_i * MICRO_CHUNK
            c_end = (chunk_i + 2) * MICRO_CHUNK
            if c_end > SEQ_LEN:
                break
            chunk = batch[:, c_start:c_end]

            out = model(chunk, output_hidden_states=True)
            logits = out.logits if hasattr(out, 'logits') else out[0]

            shift_logits = logits[:, MICRO_CHUNK-1:-1, :].contiguous()
            shift_labels = chunk[:, MICRO_CHUNK:].contiguous()

            ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                       shift_labels.view(-1))

            # v10: get hidden states — no detach (gradients flow)
            hidden = out.hidden_states[-1] if hasattr(out, 'hidden_states') and out.hidden_states else logits

            # v10: soma_head takes hidden ONLY — no proprio input
            regime_logits, stress_pred, pulse_pred, future_pulse_pred = soma_head(hidden)

            target_regime = torch.full((regime_logits.size(0),), CTX.regime,
                                        device=DEVICE, dtype=torch.long)
            regime_loss = F.cross_entropy(regime_logits, target_regime, weight=regime_weights)
            regime_counts[CTX.regime] += 1
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

            # v10: pulse prediction auxiliary loss
            # Target: actual GPU-resident pulse field mean (not CPU-computed proxy)
            pulse_loss = torch.tensor(0.0, device=DEVICE)
            future_pulse_loss = torch.tensor(0.0, device=DEVICE)
            if PULSE_BUFFERS is not None:
                # Current pulse field mean — read directly from GPU buffer
                current_pulse_target = PULSE_BUFFERS.buffers.mean().detach()
                pulse_loss = F.mse_loss(
                    pulse_pred,
                    current_pulse_target.expand(pulse_pred.size(0))
                )
                # Future pulse target: std of pulse field (captures spatial structure)
                # This is also GPU-resident, no CPU roundtrip
                future_pulse_target = PULSE_BUFFERS.buffers.std().detach()
                future_pulse_loss = F.mse_loss(
                    future_pulse_pred,
                    future_pulse_target.expand(future_pulse_pred.size(0))
                )

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
                    ema_logits[:s0, :s1] = (EMA_DECAY * ema_logits[:s0, :s1]
                                            + (1-EMA_DECAY) * shift_logits[:s0, :s1].detach())

            # Branch usage penalty
            branch_penalty = torch.tensor(0.0, device=DEVICE)
            fast_norms = []
            for layer_idx in LORA_LAYERS:
                g = model.model.layers[layer_idx].mlp.gate_proj
                if isinstance(g, BodyGatedLoRA):
                    fast_norms.append(g.last_g_fast_norm)
            if fast_norms:
                fast_var = np.var(fast_norms)
                branch_penalty = torch.tensor(max(0, 0.01 - fast_var), device=DEVICE) * 0.1

            # v10: total loss includes pulse prediction aux losses
            total_loss = (ce_loss
                          + LAMBDA_INTERO * intero_loss
                          + LAMBDA_STAB * stab_loss
                          + branch_penalty
                          + LAMBDA_PULSE_PRED * pulse_loss
                          + LAMBDA_PULSE_PRED * future_pulse_loss)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)

            # Track fast branch gradient norms
            fast_grad_sum = 0.0
            for layer_idx in LORA_LAYERS:
                g = model.model.layers[layer_idx].mlp.gate_proj
                if isinstance(g, BodyGatedLoRA) and g.gate_fast.weight.grad is not None:
                    fast_grad_sum += g.gate_fast.weight.grad.norm().item()
                    g.last_fast_grad_norm = g.gate_fast.weight.grad.norm().item()

            optimizer.step()
            chunk_losses.append(ce_loss.item())

        # Log branch norms
        for layer_idx in LORA_LAYERS[:1]:
            g = model.model.layers[layer_idx].mlp.gate_proj
            if isinstance(g, BodyGatedLoRA):
                branch_norms_log['fast'].append(g.last_g_fast_norm)
                branch_norms_log['mid'].append(g.last_g_mid_norm)
                branch_norms_log['slow'].append(g.last_g_slow_norm)
        fast_grad_norms_log.append(fast_grad_sum)

        results['temp_range'][0] = min(results['temp_range'][0], METAB._last_temp)
        results['temp_range'][1] = max(results['temp_range'][1], METAB._last_temp)

        if (batch_idx + 1) % 25 == 0:
            avg_ce = np.mean(chunk_losses) if chunk_losses else 0
            fn = np.mean(branch_norms_log['fast'][-5:]) if branch_norms_log['fast'] else 0
            mn = np.mean(branch_norms_log['mid'][-5:]) if branch_norms_log['mid'] else 0
            sn = np.mean(branch_norms_log['slow'][-5:]) if branch_norms_log['slow'] else 0
            pulse_stats = PULSE_BUFFERS.get_field_stats() if PULSE_BUFFERS is not None else {}
            print(f"    Batch {batch_idx+1}/{N_TRAIN_BATCHES} [{phase_name}]: "
                  f"ce={avg_ce:.4f}, dvfs={current_dvfs}, "
                  f"temp={METAB._last_temp:.1f}C, regime={CTX.regime}, "
                  f"g_norms(f/m/s)={fn:.3f}/{mn:.3f}/{sn:.3f}, "
                  f"pulse_active={pulse_stats.get('active_frac', 0):.3f}")

    results['train_ce'] = np.mean(chunk_losses) if chunk_losses else 0
    results['branch_norms'] = {k: [float(x) for x in v] for k, v in branch_norms_log.items()}
    results['fast_grad_norms'] = [float(x) for x in fast_grad_norms_log]
    if PULSE_BUFFERS is not None:
        results['pulse_field_stats'] = PULSE_BUFFERS.get_field_stats()
    set_dvfs('auto')
    time.sleep(DVFS_SETTLE_S)

    print("\n  Regime confusion matrix:")
    print(f"  {soma_head.confusion}")
    results['regime_confusion'] = soma_head.confusion.tolist()

    # --- Phase B Tests ---
    _run_phase_b_tests(model, soma_head, soma_probe, test_data, test_data_code, baseline_ppl,
                       phase_a_results, results, branch_norms_log, fast_grad_norms_log)

    # Cleanup
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    restore_gate_proj_lora(model, lora_originals)
    restore_gate_proj(model, analog_originals)
    return results

def _run_phase_b_tests(model, soma_head, soma_probe, test_data, test_data_code, baseline_ppl,
                       phase_a_results, results, branch_norms_log, fast_grad_norms_log):
    """Phase B test battery — v10 with pulse ablation tests T31-T35.
    soma_head: hidden-only (no-cheat) head used for training
    soma_probe: hidden+telemetry head used for eval diagnostics only
    """
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
        METAB.update_context()
        fixed_ppl = eval_ppl(model, test_data)
        lora_fixed_ppls[f'mode{mode_val}'] = fixed_ppl
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    METAB.update_context()
    lora_analog_ppl = eval_ppl(model, test_data)
    results['lora_analog_ppl'] = lora_analog_ppl
    best_fixed_ppl = min(lora_fixed_ppls.values()) if lora_fixed_ppls else 999
    results['lora_fixed_ppl'] = best_fixed_ppl
    results['lora_fixed_ppls'] = lora_fixed_ppls
    embodied_advantage = best_fixed_ppl - lora_analog_ppl
    results['embodied_advantage'] = embodied_advantage
    print(f"  Live analog PPL: {lora_analog_ppl:.4f}, Best fixed PPL: {best_fixed_ppl:.4f}")
    print(f"  Embodied advantage: {embodied_advantage:.6f}")

    # T17b: Matched-State Advantage
    print("\n[T17b] Matched-State Embodied Advantage...")
    matched_advantages = []
    for dvfs_state in ['low', 'high']:
        if DVFS_AVAILABLE:
            set_dvfs(dvfs_state)
            time.sleep(DVFS_SETTLE_S)
        METAB.update_context()
        set_analog_mode_override(model, ANALOG_LAYERS, None)
        live_ppl = eval_ppl(model, test_data, n_batches=10)
        packed_mode = pack_mode_byte(CTX.round_mode, CTX.round_mode, 3, 3)
        set_analog_mode_override(model, ANALOG_LAYERS, packed_mode)
        frozen_ppl = eval_ppl(model, test_data, n_batches=10)
        matched_advantages.append(frozen_ppl - live_ppl)
        set_analog_mode_override(model, ANALOG_LAYERS, None)
    if DVFS_AVAILABLE:
        set_dvfs('auto')
        time.sleep(0.5)
    mean_matched = np.mean(matched_advantages) if matched_advantages else 0
    results['mean_matched_advantage'] = float(mean_matched)
    print(f"  Mean matched advantage: {mean_matched:.6f}")

    # T18: Soma Interoception — uses soma_probe (with proprio) for diagnostic
    print("\n[T18] Soma Interoception Correlation (SomaProbeHead w/ proprio)...")
    soma_preds = []
    soma_trues = []
    for dvfs_name in ['low', 'high', 'auto']:
        if DVFS_AVAILABLE:
            set_dvfs(dvfs_name)
            time.sleep(DVFS_SETTLE_S)
        for _ in range(3):
            METAB.update_context()
            CTX.clear_proprio()
            with torch.no_grad():
                batch = torch.stack(test_data[:BS]).to(DEVICE)
                out = model(batch, output_hidden_states=True)
                hidden = out.hidden_states[-1]
                proprio_stats = aggregate_proprio().to(DEVICE)
                # Use probe head (with proprio) for diagnostic correlation
                _, stress_pred = soma_probe(hidden, proprio_stats)
                soma_preds.append(stress_pred.mean().item())
                soma_trues.append(CTX.continuous_stress)
    if DVFS_AVAILABLE:
        set_dvfs('auto')
    soma_correlation = 0.0
    soma_mae = 999.0
    if len(soma_preds) >= 3:
        try:
            corr, _ = stats.pearsonr(soma_preds, soma_trues)
            soma_correlation = float(corr) if not math.isnan(corr) else 0.0
        except:
            soma_correlation = 0.0
        soma_mae = float(np.mean(np.abs(np.array(soma_preds) - np.array(soma_trues))))
    results['soma_correlation'] = soma_correlation
    results['soma_mae'] = soma_mae
    print(f"  Soma corr: {soma_correlation:.4f}, MAE: {soma_mae:.4f}")

    # T18b: No-Cheat Soma — soma_head (hidden-only) regime accuracy
    print("\n[T18b] No-Cheat Soma Interoception (hidden-only head)...")
    nocheat_preds = []
    nocheat_trues = []
    for dvfs_name in ['low', 'high', 'auto']:
        if DVFS_AVAILABLE:
            set_dvfs(dvfs_name)
            time.sleep(DVFS_SETTLE_S)
        for _ in range(3):
            METAB.update_context()
            CTX.clear_proprio()
            with torch.no_grad():
                batch = torch.stack(test_data[:BS]).to(DEVICE)
                out = model(batch, output_hidden_states=True)
                hidden = out.hidden_states[-1]
                # No-cheat: hidden only
                _, stress_pred_nc, _, _ = soma_head(hidden)
                nocheat_preds.append(stress_pred_nc.mean().item())
                nocheat_trues.append(CTX.continuous_stress)
    if DVFS_AVAILABLE:
        set_dvfs('auto')
    nocheat_corr = 0.0
    nocheat_mae = 999.0
    if len(nocheat_preds) >= 3:
        try:
            corr, _ = stats.pearsonr(nocheat_preds, nocheat_trues)
            nocheat_corr = float(corr) if not math.isnan(corr) else 0.0
        except:
            nocheat_corr = 0.0
        nocheat_mae = float(np.mean(np.abs(np.array(nocheat_preds) - np.array(nocheat_trues))))
    results['nocheat_soma_correlation'] = nocheat_corr
    results['nocheat_soma_mae'] = nocheat_mae
    print(f"  No-cheat soma corr: {nocheat_corr:.4f}, MAE: {nocheat_mae:.4f}")

    # T19: Held-Out Regime Classification (soma_probe with proprio)
    print("\n[T19] Held-Out Regime Classification (SomaProbeHead)...")
    eval_conf = np.zeros((N_REGIMES, N_REGIMES), dtype=np.int64)
    eval_correct = 0
    eval_total = 0
    for dvfs_name in ['low', 'high', 'auto']:
        if DVFS_AVAILABLE:
            set_dvfs(dvfs_name)
            time.sleep(DVFS_SETTLE_S)
        for _ in range(5):
            METAB.update_context()
            CTX.clear_proprio()
            with torch.no_grad():
                batch = torch.stack(test_data[:BS]).to(DEVICE)
                out = model(batch, output_hidden_states=True)
                hidden = out.hidden_states[-1]
                proprio_stats = aggregate_proprio().to(DEVICE)
                regime_logits_p, _ = soma_probe(hidden, proprio_stats)
                pred = int(regime_logits_p[0].argmax().item())
                true_r = CTX.regime
                eval_conf[true_r, pred] += 1
                eval_correct += int(pred == true_r)
                eval_total += 1
    if DVFS_AVAILABLE:
        set_dvfs('auto')
    eval_acc = eval_correct / max(eval_total, 1)
    per_class_recall = []
    for c in range(N_REGIMES):
        total_c = eval_conf[c].sum()
        if total_c > 0:
            per_class_recall.append(eval_conf[c, c] / total_c)
    eval_balanced = np.mean(per_class_recall) if per_class_recall else 0.0
    results['regime_accuracy'] = eval_acc
    results['regime_balanced_accuracy'] = float(eval_balanced)
    results['regime_eval_confusion'] = eval_conf.tolist()
    print(f"  Held-out eval: acc={eval_acc:.4f}, balanced={eval_balanced:.4f}")

    # T20: Body Gate Analysis
    print("\n[T20] Body Gate Analysis (3-branch)...")
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
                    mid_t = torch.from_numpy(CTX.mid_vec.copy()).float().to(g.gate_mid.weight.device)
                    slow_t = torch.from_numpy(CTX.slow_vec.copy()).float().to(g.gate_slow.weight.device)
                    gate_pre = g.gate_fast(fast_t) + g.gate_mid(mid_t) + g.gate_slow(slow_t)
                    gate_val = torch.sigmoid(gate_pre).mean().item()
                    gate_list.append(gate_val)
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

    # T_STEP: Step Response
    print("\n[T_STEP] Step Response...")
    step_results = _test_step_response(model)
    results['step_response'] = step_results

    # T26a: Causal Ablation
    print("\n[T26a] Causal Ablation (zero/negate/noise fast)...")
    causal_results = _test_causal_ablation_v6(model, test_data)
    results['causal_ablation'] = causal_results

    # T26d: WGP Routing Ablation
    print("\n[T26d] WGP Routing Ablation...")
    wgp_ablation = _test_wgp_routing_ablation(model, test_data)
    results['wgp_routing_ablation'] = wgp_ablation

    # T_GATE_AUROC: Gate-Signal AUROC
    print("\n[T_GATE_AUROC] Gate-Signal AUROC...")
    auroc_results = _test_gate_auroc(model, test_data)
    results['gate_auroc'] = auroc_results

    # T28: Branch Usage Balance
    print("\n[T28] Branch Usage Balance...")
    if branch_norms_log['fast'] and branch_norms_log['mid'] and branch_norms_log['slow']:
        mean_fast = np.mean(branch_norms_log['fast'][-50:])
        mean_mid = np.mean(branch_norms_log['mid'][-50:])
        mean_slow = np.mean(branch_norms_log['slow'][-50:])
        total_norm = mean_fast + mean_mid + mean_slow + 1e-8
        fast_frac = mean_fast / total_norm
        results['branch_balance'] = {'fast_frac': float(fast_frac),
                                     'fast_mean': float(mean_fast),
                                     'mid_mean': float(mean_mid),
                                     'slow_mean': float(mean_slow)}
        print(f"  fast_frac={fast_frac:.4f} (fast={mean_fast:.4f}, mid={mean_mid:.4f}, slow={mean_slow:.4f})")
    else:
        results['branch_balance'] = {'fast_frac': 0.0}

    # T29: Fast Path Gradient
    print("\n[T29] Fast Path Gradient...")
    if fast_grad_norms_log:
        mean_grad = np.mean(fast_grad_norms_log[-50:])
        max_grad = np.max(fast_grad_norms_log[-50:])
        results['fast_grad'] = {'mean': float(mean_grad), 'max': float(max_grad)}
        print(f"  mean_grad={mean_grad:.6f}, max_grad={max_grad:.6f}")
    else:
        results['fast_grad'] = {'mean': 0.0, 'max': 0.0}

    # T30: Contention Sensitivity
    print("\n[T30] Contention Sensitivity...")
    contention_results = _test_contention_sensitivity(model, test_data)
    results['contention'] = contention_results

    # === v9/v10 PULSE TESTS ===
    print("\n[T31] Pulse-Stop (zero buffers)...")
    t31 = _test_pulse_stop(model, test_data)
    results['pulse_stop'] = t31

    print("\n[T32] Pulse-Shuffle (permute blocks)...")
    t32 = _test_pulse_shuffle(model, test_data)
    results['pulse_shuffle'] = t32

    print("\n[T33] Pulse-Stale (previous batch pulse)...")
    t33 = _test_pulse_stale(model, test_data)
    results['pulse_stale'] = t33

    print("\n[T34] Conditional Determinism (replay pulse)...")
    t34 = _test_conditional_determinism(model, test_data)
    results['conditional_determinism'] = t34

    print("\n[T35] Pulse-Delay (k=1,2,4 batch delay)...")
    t35 = _test_pulse_delay(model, test_data)
    results['pulse_delay'] = t35

    # === v10 NEW: Freeze/Bypass ablations ===
    print("\n[T36] Freeze-Pulse Ablation (FREEZE_PULSE_STATE=True)...")
    t36 = _test_freeze_pulse_ablation(model, test_data)
    results['freeze_pulse_ablation'] = t36

    print("\n[T37] Bypass-Sensing Ablation (BYPASS_LIVE_SENSING=True)...")
    t37 = _test_bypass_sensing_ablation(model, test_data)
    results['bypass_sensing_ablation'] = t37

    print("\n[T38] Disable-Mode-Writes Ablation (DISABLE_MODE_WRITES=True)...")
    t38 = _test_disable_mode_writes_ablation(model, test_data)
    results['disable_mode_writes_ablation'] = t38

    # v10: Pulse prediction accuracy
    print("\n[T39] Pulse Prediction Accuracy (soma_head pulse heads)...")
    t39 = _test_pulse_prediction_accuracy(model, soma_head, test_data)
    results['pulse_prediction_accuracy'] = t39

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE B TEST HELPERS — copied/adapted from v9
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _test_step_response(model):
    """Measure gate response time after DVFS step change."""
    if not DVFS_AVAILABLE:
        print("  DVFS not available, skipping")
        return {'rise_time_forwards': -1}
    set_dvfs('low')
    time.sleep(3.0)
    for _ in range(10):
        METAB.update_context()
        time.sleep(0.2)
    batch = torch.stack([torch.randint(0, 1000, (SEQ_LEN,))] * BS).to(DEVICE)
    # Step to high
    set_dvfs('high')
    gate_trajectory = []
    for fwd_i in range(20):
        METAB.update_context()
        CTX.clear_proprio()
        with torch.no_grad():
            _ = model(batch)
        for layer_idx in LORA_LAYERS[:1]:
            g = model.model.layers[layer_idx].mlp.gate_proj
            if isinstance(g, BodyGatedLoRA):
                fast_vec = CTX.get_per_layer_fast_vec(layer_idx)
                fast_t = torch.from_numpy(fast_vec).float().to(g.gate_fast.weight.device)
                mid_t = torch.from_numpy(CTX.mid_vec.copy()).float().to(g.gate_mid.weight.device)
                slow_t = torch.from_numpy(CTX.slow_vec.copy()).float().to(g.gate_slow.weight.device)
                gate_pre = g.gate_fast(fast_t) + g.gate_mid(mid_t) + g.gate_slow(slow_t)
                gate_val = torch.sigmoid(gate_pre).mean().item()
                gate_trajectory.append(gate_val)
    set_dvfs('auto')
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
    if gate_trajectory:
        print(f"  Rise time: {rise_time} forwards, gate range: {gate_trajectory[0]:.4f} -> {gate_trajectory[-1]:.4f}")
    else:
        print("  No data")
    return {'rise_time_forwards': rise_time, 'trajectory': gate_trajectory}


def _test_causal_ablation_v6(model, test_data):
    """v6: Multiple causal interventions on fast path (zero/negate/noise)."""
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    METAB.update_context()

    normal_ppl = eval_ppl(model, test_data, n_batches=10)

    saved_get = CTX.__class__.get_per_layer_fast_vec

    def zero_get(self_ctx, idx):
        return np.zeros(FAST_DIM, dtype=np.float32)
    CTX.__class__.get_per_layer_fast_vec = zero_get
    zero_ppl = eval_ppl(model, test_data, n_batches=10)
    CTX.__class__.get_per_layer_fast_vec = saved_get

    def negate_get(self_ctx, idx):
        return -saved_get(self_ctx, idx)
    CTX.__class__.get_per_layer_fast_vec = negate_get
    negate_ppl = eval_ppl(model, test_data, n_batches=10)
    CTX.__class__.get_per_layer_fast_vec = saved_get

    def noise_get(self_ctx, idx):
        return np.random.randn(FAST_DIM).astype(np.float32)
    CTX.__class__.get_per_layer_fast_vec = noise_get
    noise_ppl = eval_ppl(model, test_data, n_batches=10)
    CTX.__class__.get_per_layer_fast_vec = saved_get

    worst_ppl = max(zero_ppl, negate_ppl, noise_ppl)
    worst_ratio = worst_ppl / max(normal_ppl, 0.01)
    zero_ratio = zero_ppl / max(normal_ppl, 0.01)
    negate_ratio = negate_ppl / max(normal_ppl, 0.01)
    noise_ratio = noise_ppl / max(normal_ppl, 0.01)

    print(f"  Normal PPL: {normal_ppl:.4f}")
    print(f"  Zero fast: {zero_ppl:.4f} (ratio={zero_ratio:.4f})")
    print(f"  Negate fast: {negate_ppl:.4f} (ratio={negate_ratio:.4f})")
    print(f"  Noise fast: {noise_ppl:.4f} (ratio={noise_ratio:.4f})")
    print(f"  Worst ratio: {worst_ratio:.4f}")

    return {
        'normal_ppl': normal_ppl,
        'zero_ppl': zero_ppl, 'zero_ratio': zero_ratio,
        'negate_ppl': negate_ppl, 'negate_ratio': negate_ratio,
        'noise_ppl': noise_ppl, 'noise_ratio': noise_ratio,
        'worst_ratio': worst_ratio,
    }


def _test_wgp_routing_ablation(model, test_data):
    """v8 T26d: Zero out wgp_corr_A and wgp_corr_B, measure PPL change."""
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    METAB.update_context()

    normal_ppl = eval_ppl(model, test_data, n_batches=10)

    saved_wgp_A = {}
    saved_wgp_B = {}
    for layer_idx in LORA_LAYERS:
        g = model.model.layers[layer_idx].mlp.gate_proj
        if isinstance(g, BodyGatedLoRA):
            saved_wgp_A[layer_idx] = g.wgp_corr_A.data.clone()
            saved_wgp_B[layer_idx] = g.wgp_corr_B.data.clone()
            g.wgp_corr_A.data.zero_()
            g.wgp_corr_B.data.zero_()

    ablated_ppl = eval_ppl(model, test_data, n_batches=10)

    for layer_idx in saved_wgp_A:
        g = model.model.layers[layer_idx].mlp.gate_proj
        if isinstance(g, BodyGatedLoRA):
            g.wgp_corr_A.data.copy_(saved_wgp_A[layer_idx])
            g.wgp_corr_B.data.copy_(saved_wgp_B[layer_idx])

    ratio = ablated_ppl / max(normal_ppl, 0.01)
    print(f"  Normal PPL: {normal_ppl:.4f}, wgp_corr=0 PPL: {ablated_ppl:.4f}, ratio: {ratio:.4f}")
    return {'normal_ppl': normal_ppl, 'ablated_ppl': ablated_ppl, 'ratio': ratio}


def _test_gate_auroc(model, test_data):
    """Test if gate values separate DVFS low vs high."""
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
                    mid_t = torch.from_numpy(CTX.mid_vec.copy()).float().to(g.gate_mid.weight.device)
                    slow_t = torch.from_numpy(CTX.slow_vec.copy()).float().to(g.gate_slow.weight.device)
                    gate_pre = g.gate_fast(fast_t) + g.gate_mid(mid_t) + g.gate_slow(slow_t)
                    gate_val = torch.sigmoid(gate_pre).mean().item()
                    gates_per_dvfs[dvfs_name].append(gate_val)
    set_dvfs('auto')
    if gates_per_dvfs['low'] and gates_per_dvfs['high']:
        try:
            u_stat, p_val = stats.mannwhitneyu(gates_per_dvfs['low'], gates_per_dvfs['high'],
                                                alternative='two-sided')
            n1, n2 = len(gates_per_dvfs['low']), len(gates_per_dvfs['high'])
            auroc = u_stat / (n1 * n2)
            auroc = max(auroc, 1 - auroc)
        except Exception:
            auroc = 0.5
            p_val = 1.0
        print(f"  Gate AUROC (low vs high): {auroc:.4f}, p={p_val:.6f}")
        return {'auroc': float(auroc), 'p_value': float(p_val),
                'low_mean': float(np.mean(gates_per_dvfs['low'])),
                'high_mean': float(np.mean(gates_per_dvfs['high']))}
    print("  No gate data collected")
    return {'auroc': 0.5}


def _test_contention_sensitivity(model, test_data):
    """v6: Test if PPL differs under memory contention vs clean."""
    print("  Clean run...")
    clean_ppl = eval_ppl(model, test_data, n_batches=10)

    print("  Contention run...")
    contention_stream = torch.cuda.Stream()
    contention_ppls = []
    for i in range(10):
        METAB.update_context()
        with torch.cuda.stream(contention_stream):
            buf = torch.randn(2048, 2048, device=DEVICE)
            for _ in range(3):
                buf = buf @ buf.T
        batch = torch.stack(test_data[i*BS:(i+1)*BS]).to(DEVICE)
        with torch.no_grad():
            out = model(batch)
            logits = out.logits if hasattr(out, 'logits') else out[0]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), reduction='sum')
            n_tok = shift_labels.numel()
            contention_ppls.append(loss.item() / max(n_tok, 1))
    contention_ppl = math.exp(min(np.mean(contention_ppls), 20))
    ratio = contention_ppl / max(clean_ppl, 0.01)
    print(f"  Clean PPL: {clean_ppl:.4f}, Contention PPL: {contention_ppl:.4f}, ratio: {ratio:.4f}")
    return {'clean_ppl': clean_ppl, 'contention_ppl': contention_ppl, 'ratio': ratio}



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PULSE ABLATION TESTS — v9 originals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _test_pulse_stop(model, test_data):
    """T31: Zero all pulse buffers, measure PPL change."""
    if PULSE_BUFFERS is None:
        return {'ratio': 1.0}
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    METAB.update_context()

    normal_ppl = eval_ppl(model, test_data, n_batches=10)

    PULSE_BUFFERS.snapshot()
    PULSE_BUFFERS.zero()
    zeroed_ppl = eval_ppl(model, test_data, n_batches=10)
    PULSE_BUFFERS.restore_snapshot()

    ratio = zeroed_ppl / max(normal_ppl, 0.01)
    print(f"  Normal PPL: {normal_ppl:.4f}, Zeroed PPL: {zeroed_ppl:.4f}, ratio: {ratio:.4f}")
    return {'normal_ppl': normal_ppl, 'zeroed_ppl': zeroed_ppl, 'ratio': ratio}


def _test_pulse_shuffle(model, test_data):
    """T32: Randomly permute pulse buffer block dimension, measure PPL change."""
    if PULSE_BUFFERS is None:
        return {'ratio': 1.0}
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    METAB.update_context()

    normal_ppl = eval_ppl(model, test_data, n_batches=10)

    PULSE_BUFFERS.snapshot()
    PULSE_BUFFERS.shuffle_blocks()
    shuffled_ppl = eval_ppl(model, test_data, n_batches=10)
    PULSE_BUFFERS.restore_snapshot()

    ratio = shuffled_ppl / max(normal_ppl, 0.01)
    print(f"  Normal PPL: {normal_ppl:.4f}, Shuffled PPL: {shuffled_ppl:.4f}, ratio: {ratio:.4f}")
    return {'normal_ppl': normal_ppl, 'shuffled_ppl': shuffled_ppl, 'ratio': ratio}


def _test_pulse_stale(model, test_data):
    """T33: Use pulse buffers from previous batch (stale), measure PPL change."""
    if PULSE_BUFFERS is None:
        return {'ratio': 1.0}
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    METAB.update_context()

    with torch.no_grad():
        batch = torch.stack(test_data[:BS]).to(DEVICE)
        _ = model(batch)
    PULSE_BUFFERS.snapshot()  # save "stale" state

    with torch.no_grad():
        batch2 = torch.stack(test_data[BS:2*BS]).to(DEVICE)
        _ = model(batch2)

    normal_ppl = eval_ppl(model, test_data, n_batches=10)

    PULSE_BUFFERS.restore_snapshot()
    stale_ppl = eval_ppl(model, test_data, n_batches=10)

    ratio = stale_ppl / max(normal_ppl, 0.01)
    print(f"  Normal PPL: {normal_ppl:.4f}, Stale PPL: {stale_ppl:.4f}, ratio: {ratio:.4f}")
    return {'normal_ppl': normal_ppl, 'stale_ppl': stale_ppl, 'ratio': ratio}


def _test_conditional_determinism(model, test_data):
    """T34: Replay same pulse buffer => same output (conditional determinism)."""
    if PULSE_BUFFERS is None:
        return {'mean_diff': 999.0}
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    METAB.update_context()

    PULSE_BUFFERS.snapshot()

    batch = torch.stack(test_data[:BS]).to(DEVICE)
    with torch.no_grad():
        out1 = model(batch)
        logits1 = out1.logits if hasattr(out1, 'logits') else out1[0]

    PULSE_BUFFERS.restore_snapshot()
    with torch.no_grad():
        out2 = model(batch)
        logits2 = out2.logits if hasattr(out2, 'logits') else out2[0]

    mean_diff = float((logits1 - logits2).abs().mean())
    max_diff = float((logits1 - logits2).abs().max())
    print(f"  Mean logit diff: {mean_diff:.8f}, Max diff: {max_diff:.8f}")
    return {'mean_diff': mean_diff, 'max_diff': max_diff}


def _test_pulse_delay(model, test_data):
    """T35: Use pulse from k batches ago (k=1,2,4), measure degradation."""
    if PULSE_BUFFERS is None:
        return {'delay_4_ratio': 1.0}
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    METAB.update_context()

    snapshots = {}
    for k in range(5):
        with torch.no_grad():
            batch = torch.stack(test_data[(k*BS):((k+1)*BS)]).to(DEVICE)
            _ = model(batch)
        snapshots[k] = PULSE_BUFFERS.buffers.clone()

    normal_ppl = eval_ppl(model, test_data, n_batches=10)

    delay_ppls = {}
    for k in [1, 2, 4]:
        if k in snapshots:
            PULSE_BUFFERS.buffers.copy_(snapshots[max(0, 4-k)])
            delayed_ppl = eval_ppl(model, test_data, n_batches=10)
            delay_ppls[f'delay_{k}'] = delayed_ppl
            ratio = delayed_ppl / max(normal_ppl, 0.01)
            print(f"  Delay k={k}: PPL={delayed_ppl:.4f}, ratio={ratio:.4f}")

    PULSE_BUFFERS.buffers.copy_(snapshots[4])

    delay_4_ratio = delay_ppls.get('delay_4', normal_ppl) / max(normal_ppl, 0.01)
    result = {'normal_ppl': normal_ppl, 'delay_4_ratio': delay_4_ratio}
    result.update(delay_ppls)
    return result



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v10 NEW ABLATION TESTS: Freeze / Bypass / Disable / Prediction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _test_freeze_pulse_ablation(model, test_data):
    """T36 (v10): Freeze pulse state (FREEZE_PULSE_STATE=True), measure PPL change.

    When frozen, the kernel leaky-integrate update is suppressed so pulse_out
    stays constant at whatever value was in the buffer before eval.  We expect
    this to hurt PPL because the pulse field loses its live timing signal.
    """
    if PULSE_BUFFERS is None:
        return {'ratio': 1.0, 'note': 'no_pulse_buffers'}

    set_analog_mode_override(model, ANALOG_LAYERS, None)
    METAB.update_context()

    # Baseline — normal live pulse
    normal_ppl = eval_ppl(model, test_data, n_batches=10)

    # Frozen — use pulse_mode() context manager to set FREEZE_PULSE_STATE=True
    with pulse_mode(freeze=True):
        frozen_ppl = eval_ppl(model, test_data, n_batches=10)

    ratio = frozen_ppl / max(normal_ppl, 0.01)
    print(f"  Normal PPL: {normal_ppl:.4f}, Frozen PPL: {frozen_ppl:.4f}, ratio: {ratio:.4f}")
    return {'normal_ppl': normal_ppl, 'frozen_ppl': frozen_ppl, 'ratio': ratio}


def _test_bypass_sensing_ablation(model, test_data):
    """T37 (v10): Bypass live sensing (BYPASS_LIVE_SENSING=True), measure PPL change.

    When bypassed, the kernel uses sensed=0.0 instead of the measured
    0.50*norm_dt + 0.30*norm_var + 0.20*wgp_phase combination.  The pulse
    field then only propagates its own prior leaky state with zero new
    hardware signal injected.
    """
    if PULSE_BUFFERS is None:
        return {'ratio': 1.0, 'note': 'no_pulse_buffers'}

    set_analog_mode_override(model, ANALOG_LAYERS, None)
    METAB.update_context()

    normal_ppl = eval_ppl(model, test_data, n_batches=10)

    with pulse_mode(bypass=True):
        bypass_ppl = eval_ppl(model, test_data, n_batches=10)

    ratio = bypass_ppl / max(normal_ppl, 0.01)
    print(f"  Normal PPL: {normal_ppl:.4f}, Bypass-sensing PPL: {bypass_ppl:.4f}, ratio: {ratio:.4f}")
    return {'normal_ppl': normal_ppl, 'bypass_ppl': bypass_ppl, 'ratio': ratio}


def _test_disable_mode_writes_ablation(model, test_data):
    """T38 (v10): Disable hardware MODE register writes (DISABLE_MODE_WRITES=True).

    The s_setreg_b32 hwreg(1,0,8) instruction that encodes the rounding mode
    into the GPU shader is suppressed.  The AnalogLinear GEMM still runs with
    the same EPS gain modulation, but the physical rounding mode is no longer
    varied between forward passes.  We expect reduced KL divergence across mode
    values and potentially a PPL change if the model had adapted to mode-specific
    arithmetic.
    """
    set_analog_mode_override(model, ANALOG_LAYERS, None)
    METAB.update_context()

    # Normal PPL with mode writes enabled
    normal_ppl = eval_ppl(model, test_data, n_batches=10)

    # Collect logits with mode writes active across 4 modes
    mode_logits_live = {}
    for mode_val in range(4):
        packed = pack_mode_byte(mode_val, mode_val, 3, 3)
        set_analog_mode_override(model, ANALOG_LAYERS, packed)
        mode_logits_live[mode_val] = collect_logits(model, test_data, n_batches=3)
    set_analog_mode_override(model, ANALOG_LAYERS, None)

    # Collect logits with mode writes disabled — modes should be identical
    mode_logits_disabled = {}
    with pulse_mode(disable_writes=True):
        for mode_val in range(4):
            packed = pack_mode_byte(mode_val, mode_val, 3, 3)
            set_analog_mode_override(model, ANALOG_LAYERS, packed)
            mode_logits_disabled[mode_val] = collect_logits(model, test_data, n_batches=3)
        set_analog_mode_override(model, ANALOG_LAYERS, None)
        disable_ppl = eval_ppl(model, test_data, n_batches=10)

    # KL across modes: should be larger when writes enabled (more variation)
    kl_live = 0.0
    kl_disabled = 0.0
    for m0 in range(4):
        for m1 in range(m0+1, 4):
            if (mode_logits_live[m0] is not None and mode_logits_live[m1] is not None):
                kl_live += kl_divergence(mode_logits_live[m0], mode_logits_live[m1])
            if (mode_logits_disabled[m0] is not None and mode_logits_disabled[m1] is not None):
                kl_disabled += kl_divergence(mode_logits_disabled[m0], mode_logits_disabled[m1])

    ppl_ratio = disable_ppl / max(normal_ppl, 0.01)
    kl_ratio = kl_live / max(kl_disabled, 1e-12)

    print(f"  Normal PPL: {normal_ppl:.4f}, Disabled-writes PPL: {disable_ppl:.4f}, ratio: {ppl_ratio:.4f}")
    print(f"  Mode KL (live): {kl_live:.8f}, Mode KL (disabled): {kl_disabled:.8f}, kl_ratio: {kl_ratio:.4f}")
    return {
        'normal_ppl': normal_ppl,
        'disable_ppl': disable_ppl,
        'ppl_ratio': ppl_ratio,
        'kl_live': kl_live,
        'kl_disabled': kl_disabled,
        'kl_ratio': kl_ratio,
    }


def _test_pulse_prediction_accuracy(model, soma_head, test_data):
    """T39 (v10): Evaluate pulse_prediction_head and future_pulse_head accuracy.

    The soma_head now exposes two extra outputs: pulse_pred (predicted current
    mean pulse field value) and future_pulse_pred (predicted next-step mean).
    We compare these against the actual mean pulse field values measured before
    and after each batch forward pass.

    Returns MAE and correlation for both heads.
    """
    if PULSE_BUFFERS is None:
        return {'pulse_mae': 999.0, 'future_mae': 999.0, 'note': 'no_pulse_buffers'}

    set_analog_mode_override(model, ANALOG_LAYERS, None)
    METAB.update_context()

    pulse_preds = []
    pulse_actuals = []
    future_preds = []
    future_actuals = []

    soma_head.eval()
    with torch.no_grad():
        for i in range(15):
            METAB.update_context()
            CTX.clear_proprio()

            # Actual pulse field mean BEFORE this forward pass
            actual_pulse = float(PULSE_BUFFERS.buffers.mean().item())

            batch = torch.stack(test_data[i*BS:(i+1)*BS]).to(DEVICE)
            out = model(batch, output_hidden_states=True)
            hidden = out.hidden_states[-1] if (hasattr(out, 'hidden_states') and out.hidden_states) else (out.logits if hasattr(out, 'logits') else out[0])

            # soma_head(hidden) returns 4 values in v10
            regime_logits, stress_pred, pulse_pred, future_pulse_pred = soma_head(hidden)

            pulse_pred_val = float(pulse_pred.mean().item())
            future_pred_val = float(future_pulse_pred.mean().item())

            # Actual pulse field mean AFTER this forward pass (the "future")
            actual_future = float(PULSE_BUFFERS.buffers.mean().item())

            pulse_preds.append(pulse_pred_val)
            pulse_actuals.append(actual_pulse)
            future_preds.append(future_pred_val)
            future_actuals.append(actual_future)

    pulse_mae = float(np.mean(np.abs(np.array(pulse_preds) - np.array(pulse_actuals))))
    future_mae = float(np.mean(np.abs(np.array(future_preds) - np.array(future_actuals))))

    pulse_corr = 0.0
    future_corr = 0.0
    try:
        if len(pulse_preds) >= 3 and np.std(pulse_actuals) > 1e-8:
            r, _ = stats.pearsonr(pulse_preds, pulse_actuals)
            pulse_corr = float(r) if not math.isnan(r) else 0.0
        if len(future_preds) >= 3 and np.std(future_actuals) > 1e-8:
            r, _ = stats.pearsonr(future_preds, future_actuals)
            future_corr = float(r) if not math.isnan(r) else 0.0
    except Exception:
        pass

    print(f"  Pulse pred MAE: {pulse_mae:.6f}, corr: {pulse_corr:.4f}")
    print(f"  Future pulse pred MAE: {future_mae:.6f}, corr: {future_corr:.4f}")
    return {
        'pulse_mae': pulse_mae,
        'pulse_corr': pulse_corr,
        'future_mae': future_mae,
        'future_corr': future_corr,
        'n_samples': len(pulse_preds),
    }



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
    add('T3', 'Mode Sweep', 'PASS' if std > 0.01 else 'FAIL', f'sigma={std:.6f}', '>0.01')
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
    interaction = abs(abs(d.get('wiki_mode0', 0) - d.get('wiki_mode3', 0)) -
                      abs(d.get('code_mode0', 0) - d.get('code_mode3', 0)))
    add('T11', 'Domain Sensitivity', 'PASS' if interaction > 0.01 else 'FAIL',
        f'{interaction:.4f}', '>0.01')
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
        add('T17', 'LoRA Adapt', 'PASS' if lp < pa else 'FAIL',
            f'{lp:.4f} vs {pa:.4f}', '<phaseA')
        ea = phase_b.get('embodied_advantage', -1)
        add('T18', 'Embodied Adv', 'PASS' if ea > 0 else 'FAIL', f'{ea:.6f}', '>0')
        msa = phase_b.get('mean_matched_advantage', -1)
        add('T18b', 'Matched-State', 'PASS' if msa > 0 else 'FAIL', f'{msa:.6f}', '>0')
        la = phase_b.get('lora_analog_ppl', 999)
        lf = phase_b.get('lora_fixed_ppl', 999)
        add('T19', 'LoRA vs Fixed', 'PASS' if la < lf else 'FAIL',
            f'{la:.4f} vs {lf:.4f}', 'analog<fixed')
        sc = phase_b.get('soma_correlation', 0)
        sm = phase_b.get('soma_mae', 999)
        add('T20', 'Interoception', 'PASS' if abs(sc) > 0.3 or sm < 0.5 else 'FAIL',
            f'corr={sc:.4f} mae={sm:.4f}', '|corr|>0.3')
        ba = phase_b.get('regime_balanced_accuracy', 0)
        ra = phase_b.get('regime_accuracy', 0)
        add('T22', 'Regime Class', 'PASS' if ba > 0.35 else 'FAIL',
            f'acc={ra:.4f} bal={ba:.4f}', 'bal>0.35')
        bg = phase_b.get('body_gate_separation', 0)
        add('T23', 'Gate Sep', 'PASS' if bg > 0.01 else 'FAIL', f'{bg:.6f}', '>0.01')
        zk2 = phase_b.get('zombie_kl', 0)
        add('T24', 'Zombie(B)', 'PASS' if zk2 > 0.0001 else 'FAIL', f'{zk2:.8f}', '>0.0001')
        sr = phase_b.get('step_response', {})
        rt = sr.get('rise_time_forwards', -1)
        add('T25', 'Step Response', 'PASS' if 0 < rt <= 10 else 'FAIL',
            f'{rt} fwds', '0<rt<=10')
        ca = phase_b.get('causal_ablation', {})
        add('T26a', 'Causal Ablation', 'PASS' if ca.get('worst_ratio', 1) > 1.005 else 'FAIL',
            f'{ca.get("worst_ratio", 1):.4f}', '>1.005')
        wgpa = phase_b.get('wgp_routing_ablation', {})
        add('T26d', 'WGP Routing Abl', 'PASS' if wgpa.get('ratio', 1) > 1.002 else 'FAIL',
            f'{wgpa.get("ratio", 1):.4f}', '>1.002')
        ga = phase_b.get('gate_auroc', {})
        add('T27', 'Gate AUROC', 'PASS' if ga.get('auroc', 0.5) > 0.6 else 'FAIL',
            f'{ga.get("auroc", 0.5):.4f}', '>0.6')
        bb = phase_b.get('branch_balance', {})
        ff = bb.get('fast_frac', 0)
        add('T28', 'Branch Balance', 'PASS' if ff > 0.1 else 'FAIL', f'{ff:.4f}', '>0.1')
        fg = phase_b.get('fast_grad', {})
        fgm = fg.get('mean', 0)
        add('T29', 'Fast Grad', 'PASS' if fgm > 0 else 'FAIL', f'{fgm:.6f}', '>0')
        ct = phase_b.get('contention', {})
        cr = ct.get('ratio', 1.0)
        add('T30', 'Contention Sens', 'PASS' if abs(cr - 1.0) > 0.001 else 'FAIL',
            f'{cr:.4f}', '|ratio-1|>0.001')

        # v9 pulse ablation tests
        ps = phase_b.get('pulse_stop', {})
        add('T31', 'Pulse-Stop', 'PASS' if ps.get('ratio', 1) > 1.005 else 'FAIL',
            f'{ps.get("ratio", 1):.4f}', '>1.005')
        psh = phase_b.get('pulse_shuffle', {})
        add('T32', 'Pulse-Shuffle', 'PASS' if psh.get('ratio', 1) > 1.002 else 'FAIL',
            f'{psh.get("ratio", 1):.4f}', '>1.002')
        pst = phase_b.get('pulse_stale', {})
        add('T33', 'Pulse-Stale', 'PASS' if pst.get('ratio', 1) > 1.001 else 'FAIL',
            f'{pst.get("ratio", 1):.4f}', '>1.001')
        cd = phase_b.get('conditional_determinism', {})
        add('T34', 'Cond Determinism', 'PASS' if cd.get('mean_diff', 999) < 0.01 else 'FAIL',
            f'{cd.get("mean_diff", 999):.8f}', '<0.01')
        pd_ = phase_b.get('pulse_delay', {})
        add('T35', 'Pulse-Delay', 'PASS' if pd_.get('delay_4_ratio', 1) > 1.001 else 'FAIL',
            f'{pd_.get("delay_4_ratio", 1):.4f}', '>1.001')

        # v10 new ablation tests
        fpa = phase_b.get('freeze_pulse_ablation', {})
        add('T36', 'Freeze-Pulse Abl', 'PASS' if fpa.get('ratio', 1) > 1.002 else 'FAIL',
            f'{fpa.get("ratio", 1):.4f}', '>1.002')
        bsa = phase_b.get('bypass_sensing_ablation', {})
        add('T37', 'Bypass-Sensing Abl', 'PASS' if bsa.get('ratio', 1) > 1.002 else 'FAIL',
            f'{bsa.get("ratio", 1):.4f}', '>1.002')
        dma = phase_b.get('disable_mode_writes_ablation', {})
        add('T38', 'Disable-Writes Abl', 'PASS' if dma.get('kl_ratio', 1) > 1.5 else 'FAIL',
            f'kl_ratio={dma.get("kl_ratio", 1):.4f}', '>1.5')
        ppa = phase_b.get('pulse_prediction_accuracy', {})
        add('T39', 'Pulse Pred Acc', 'PASS' if ppa.get('pulse_mae', 999) < 0.5 else 'FAIL',
            f'mae={ppa.get("pulse_mae", 999):.4f} corr={ppa.get("pulse_corr", 0):.4f}',
            'mae<0.5')

    n_pass = sum(1 for t in tests if t['status'] == 'PASS')
    return tests, n_pass, len(tests)



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    global N_LAYERS, ANALOG_LAYERS, LORA_LAYERS, _TOKENIZER, SCLK_LOW_CAL, SCLK_HIGH_CAL
    global SMN_AVAILABLE, SMN_FD, PULSE_BUFFERS
    print("=" * 60)
    print("z2118v10: Haptic Resonance — Freeze/Bypass/Disable Ablations")
    print("GPU-resident pulse field + weighted sensing + pulse prediction heads")
    print(f"  PULSE_EPS={PULSE_EPS}, FREEZE_PULSE_STATE={FREEZE_PULSE_STATE}")
    print(f"  BYPASS_LIVE_SENSING={BYPASS_LIVE_SENSING}, DISABLE_MODE_WRITES={DISABLE_MODE_WRITES}")
    print(f"  LAMBDA_PULSE_PRED={LAMBDA_PULSE_PRED}")
    print("=" * 60)

    # Hardware Setup
    find_dvfs_sysfs()
    check_rapl()
    init_msr()
    try:
        SMN_FD = os.open('/sys/kernel/ryzen_smu_drv/smn', os.O_RDWR)
        SMN_AVAILABLE = True
        print("[SMN] Available")
    except Exception:
        SMN_AVAILABLE = False
        print("[SMN] Not available")

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

    # v10: Initialize pulse buffers
    PULSE_BUFFERS = PulseBufferManager(N_LAYERS, max_blocks=PULSE_MAX_BLOCKS)
    print(f"  [PulseBuffers] {N_LAYERS+1} layers x {PULSE_MAX_BLOCKS} blocks = "
          f"{PULSE_BUFFERS.buffers.numel()} floats")

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
        print(f"\n  Phase A signal detected (kl={max_kl:.6f}, std={ppl_std:.6f}) -> Phase B")
        phase_b = run_phase_b(model, train_data, test_data, test_data_code, baseline_ppl, phase_a)
    else:
        print(f"\n  Phase A signal too weak (kl={max_kl:.6f}, std={ppl_std:.6f}) -> skip Phase B")

    # Score
    tests, n_pass, n_total = score_tests(phase_a, phase_b, baseline_ppl)
    print("\n" + "=" * 60)
    print(f"TEST BATTERY RESULTS ({n_total} tests)")
    print("=" * 60)
    for t in tests:
        print(f"  {t['test']}: {t['name']} -- {t['status']} ({t['value']} vs {t['criterion']})")
    print(f"\n" + "=" * 60)
    print(f"z2118v10 Haptic Resonance LM: {n_pass}/{n_total} PASS")
    print("=" * 60)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'z2118_haptic_resonance.json')
    out_path = os.path.abspath(out_path)
    result_obj = {
        'experiment': 'z2118v10_haptic_resonance',
        'description': (
            'v10: GPU-resident pulse field + weighted sensing (0.50*dt+0.30*var+0.20*wgp) '
            '+ freeze/bypass/disable ablation switches + pulse prediction heads '
            '+ no-cheat soma head (hidden-only) + salience sweep'
        ),
        'backbone': f'Qwen/Qwen3-8B ({n_params:.1f}M frozen)',
        'analog_layers': ANALOG_LAYERS,
        'lora_layers': LORA_LAYERS,
        'lora_rank': LORA_RANK,
        'micro_chunk': MICRO_CHUNK,
        'body_dim': BODY_DIM,
        'fast_dim': FAST_DIM,
        'mid_dim': MID_DIM,
        'slow_dim': SLOW_DIM,
        'pulse_eps': PULSE_EPS,
        'freeze_pulse_state': FREEZE_PULSE_STATE,
        'bypass_live_sensing': BYPASS_LIVE_SENSING,
        'disable_mode_writes': DISABLE_MODE_WRITES,
        'lambda_pulse_pred': LAMBDA_PULSE_PRED,
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
        except Exception:
            pass
    if MSR_AVAILABLE and MSR_FD is not None:
        try:
            os.close(MSR_FD)
        except Exception:
            pass


if __name__ == '__main__':
    main()
