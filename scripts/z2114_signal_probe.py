#!/usr/bin/env python3
"""
z2114 Analog Signal Probe — Watch deepest HW signals under varying load
========================================================================
Quick diagnostic: how do wall_clock64 jitter, timing tails, rounding error,
SMN ADC, gpu_metrics, and power change as the GPU transitions between
idle → load → cool-down, and across DVFS levels?

Outputs a time-series table every ~0.5s showing:
  - Per-block wall_clock64 tick statistics (mean, std, p95/p50 tail, min-max spread)
  - Rounding error magnitude (AnalogLinear output vs F.linear reference)
  - SMN thermal ADC raw bits, XTAL entropy
  - gpu_metrics: temp, power, sclk, activity, throttle
  - Derived: d/dt of jitter, d/dt of temp

Usage:
  sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python -u scripts/z2114_signal_probe.py
"""

import os, sys, struct, time
import numpy as np
import torch
import torch.nn.functional as F

# ── Hardware access (minimal, from v5) ──────────────────────────

SMN_AVAILABLE = False

def check_smn():
    global SMN_AVAILABLE
    SMN_AVAILABLE = os.path.exists('/sys/kernel/ryzen_smu_drv/smn')

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
    except:
        return 0

GPU_METRICS_PATH = None

def find_gpu_metrics():
    global GPU_METRICS_PATH
    for card in ['card1', 'card0']:
        p = f'/sys/class/drm/{card}/device/gpu_metrics'
        if os.path.exists(p):
            GPU_METRICS_PATH = p
            return True
    return False

def read_gpu_metrics():
    if GPU_METRICS_PATH is None:
        return {}
    try:
        with open(GPU_METRICS_PATH, 'rb') as f:
            data = f.read()
        if len(data) < 100:
            return {}
        r = {
            'temp_gfx': struct.unpack_from('<H', data, 4)[0] / 100.0,
            'temp_soc': struct.unpack_from('<H', data, 6)[0] / 100.0,
            'gfx_activity': struct.unpack_from('<H', data, 48)[0] / 100.0,
            'gfx_power': struct.unpack_from('<H', data, 66)[0],
            'socket_power': struct.unpack_from('<H', data, 60)[0],
        }
        if len(data) > 176:
            r['sclk_mhz'] = struct.unpack_from('<H', data, 174)[0]
        if len(data) >= 240:
            r['throttle'] = struct.unpack_from('<I', data, 236)[0]
        else:
            r['throttle'] = 0
        return r
    except:
        return {}

DVFS_PATH = None

def find_dvfs():
    global DVFS_PATH
    for card in ['card1', 'card0']:
        p = f'/sys/class/drm/{card}/device/power_dpm_force_performance_level'
        if os.path.exists(p):
            DVFS_PATH = p
            return True
    return False

def set_dvfs(name):
    if DVFS_PATH:
        torch.cuda.synchronize()
        try:
            with open(DVFS_PATH, 'w') as f:
                f.write(name)
        except:
            pass
        time.sleep(1.5)

def restore_dvfs():
    if DVFS_PATH:
        try:
            with open(DVFS_PATH, 'w') as f:
                f.write('auto')
        except:
            pass


# ── Compile the HIP kernel (reuse v5 kernel) ─────────────────

HIP_MOD = None

def compile_kernel():
    global HIP_MOD
    from torch.utils.cpp_extension import load_inline

    src = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#define TILE_SIZE 32

__global__ void probe_gemm_kernel(
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

        // Read SHADER_CYCLES for hw_spin entropy
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

std::vector<torch::Tensor> probe_gemm(
    torch::Tensor A, torch::Tensor weight,
    int round_mode, float continuous_stress
) {
    int M = A.size(0), K = A.size(1), N = weight.size(0);
    auto C = torch::empty({M, N}, A.options());

    const int TILE = 32;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    int n_blocks = grid.x * grid.y;

    auto proprio = torch::empty({n_blocks}, torch::TensorOptions().dtype(torch::kInt64).device(A.device()));

    int st = (int)(continuous_stress * 255.0f);
    st = st < 0 ? 0 : (st > 255 ? 255 : st);

    probe_gemm_kernel<<<grid, block>>>(
        reinterpret_cast<const __hip_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),
        reinterpret_cast<unsigned long long*>(proprio.data_ptr()),
        M, K, N, round_mode, st);
    return {C, proprio};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("probe_gemm", &probe_gemm, "Probe GEMM with proprioception");
}
"""
    HIP_MOD = load_inline(
        name='signal_probe_ext',
        cpp_sources=[],
        cuda_sources=[src],
        extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'],
        verbose=True,
        with_cuda=True,
    )
    print("[HIP] Probe kernel compiled OK")
    return HIP_MOD


def pack_mode(f32_round=0, f16_round=0, f32_denorm=3, f16_denorm=3):
    return ((f32_round & 3) | ((f16_round & 3) << 2) |
            ((f32_denorm & 3) << 4) | ((f16_denorm & 3) << 6))


# ── Probe function ──────────────────────────────────────────────

def run_probe(A_bf16, W_bf16, round_mode, stress, ref_out_fp32):
    """Run one analog GEMM and collect all signals."""
    results = HIP_MOD.probe_gemm(A_bf16, W_bf16, round_mode, stress)
    out_bf16 = results[0]
    proprio = results[1]
    torch.cuda.synchronize()

    # Proprio statistics
    ticks = proprio.cpu().to(torch.float64)
    ticks_pos = ticks[ticks > 0]
    n = len(ticks_pos)
    if n > 0:
        sorted_t = ticks_pos.sort().values
        p50 = sorted_t[n // 2].item()
        p05 = sorted_t[max(int(n * 0.05), 0)].item()
        p95 = sorted_t[min(int(n * 0.95), n - 1)].item()
        p99 = sorted_t[min(int(n * 0.99), n - 1)].item()
        tick_stats = {
            'mean': ticks_pos.mean().item(),
            'std': ticks_pos.std().item(),
            'min': ticks_pos.min().item(),
            'max': ticks_pos.max().item(),
            'p05': p05, 'p50': p50, 'p95': p95, 'p99': p99,
            'jitter': ticks_pos.max().item() - ticks_pos.min().item(),
            'tail_ratio': (p95 / max(p50, 1.0)) - 1.0,
            'iqr': (sorted_t[min(int(n * 0.75), n-1)] - sorted_t[max(int(n * 0.25), 0)]).item(),
            'n_blocks': n,
        }
    else:
        tick_stats = {'mean': 0, 'std': 0, 'min': 0, 'max': 0,
                      'p50': 0, 'p95': 0, 'p99': 0, 'p05': 0,
                      'jitter': 0, 'tail_ratio': 0, 'iqr': 0, 'n_blocks': 0}

    # Rounding error vs reference
    out_fp32 = out_bf16.float()
    abs_err = (out_fp32 - ref_out_fp32).abs()
    rel_err = abs_err / (ref_out_fp32.abs() + 1e-8)
    err_stats = {
        'abs_mean': abs_err.mean().item(),
        'abs_max': abs_err.max().item(),
        'abs_std': abs_err.std().item(),
        'rel_mean': rel_err.mean().item(),
        'rel_max': rel_err.max().item(),
        'nonzero_frac': (abs_err > 0).float().mean().item(),
        'large_frac': (rel_err > 0.01).float().mean().item(),  # >1% relative error
    }

    # SMN raw reads
    smn_adc_raw = read_smn(0x00059800)
    smn_xtal_raw = read_smn(0x000598C8)
    smn_stats = {
        'adc_raw': smn_adc_raw,
        'adc_low8': smn_adc_raw & 0xFF,
        'temp_bits': ((smn_adc_raw >> 21) & 0x7FF),
        'xtal_low16': smn_xtal_raw & 0xFFFF,
        'entropy': (smn_adc_raw & 0xFF) ^ (smn_xtal_raw & 0xFF),
    }

    # gpu_metrics
    gm = read_gpu_metrics()

    return tick_stats, err_stats, smn_stats, gm


# ── Main ────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("z2114 ANALOG SIGNAL PROBE — Deep HW signals under varying load")
    print("=" * 80)

    # Hardware init
    check_smn()
    find_gpu_metrics()
    find_dvfs()

    # GPU warmup
    print("\n[GPU] Warming up...")
    x = torch.randn(64, 64, device='cuda')
    for _ in range(10):
        x = x @ x.T
    torch.cuda.synchronize()
    print("[GPU] OK")

    # Compile kernel
    compile_kernel()

    # Test matrix: gate_proj-sized (4096 -> 12288) but smaller for speed
    # Use 128x4096 * 4096x4096 so we get plenty of blocks
    M, K, N = 32, 4096, 4096
    A = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    W = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)  # [N, K] like nn.Linear weight

    # Reference output
    ref = F.linear(A.float(), W.float())

    # ── Scenario schedule ─────────────────────────────────
    # Each scenario: (name, dvfs_level, n_reps, stress, round_mode)
    scenarios = [
        ("idle_cold",     "low",  8,  0.0,  pack_mode(0, 0, 3, 3)),
        ("load_ramp",     "high", 6,  0.0,  pack_mode(0, 0, 3, 3)),
        ("load_hot",      "high", 8,  0.0,  pack_mode(0, 0, 3, 3)),
        ("load_stress50", "high", 6,  0.5,  pack_mode(0, 0, 3, 3)),
        ("load_stress100","high", 6,  1.0,  pack_mode(0, 0, 3, 3)),
        ("mode_nearest",  "high", 4,  0.0,  pack_mode(0, 0, 3, 3)),
        ("mode_plusinf",  "high", 4,  0.0,  pack_mode(1, 1, 3, 3)),
        ("mode_minusinf", "high", 4,  0.0,  pack_mode(2, 2, 3, 3)),
        ("mode_zero",     "high", 4,  0.0,  pack_mode(3, 3, 3, 3)),
        ("denorm_flush",  "high", 4,  0.0,  pack_mode(0, 0, 0, 0)),
        ("denorm_allow",  "high", 4,  0.0,  pack_mode(0, 0, 3, 3)),
        ("cooldown",      "low",  8,  0.0,  pack_mode(0, 0, 3, 3)),
        ("back_to_hot",   "high", 6,  0.0,  pack_mode(0, 0, 3, 3)),
        ("final_cold",    "low",  8,  0.0,  pack_mode(0, 0, 3, 3)),
    ]

    # Header
    print(f"\n{'step':>4} {'scenario':<16} "
          f"{'tick_mean':>10} {'tick_std':>10} {'jitter':>10} {'tail%':>7} {'IQR':>10} "
          f"{'err_abs':>10} {'err_rel':>10} {'err_std':>10} {'lg_frac':>8} "
          f"{'smn_lo8':>8} {'entropy':>8} "
          f"{'temp':>6} {'sclk':>6} {'pwr_gfx':>8} {'activ':>6} {'thr':>5}")
    print("-" * 190)

    prev_jitter = None
    prev_temp = None
    step = 0

    try:
        for scenario_name, dvfs, n_reps, stress, rmode in scenarios:
            set_dvfs(dvfs)

            # Warm a few iterations to settle thermal
            for _ in range(2):
                HIP_MOD.probe_gemm(A, W, rmode, stress)
            torch.cuda.synchronize()
            time.sleep(0.3)

            for rep in range(n_reps):
                tick_s, err_s, smn_s, gm = run_probe(A, W, rmode, stress, ref)

                # d/dt
                d_jitter = ""
                if prev_jitter is not None:
                    dj = tick_s['jitter'] - prev_jitter
                    d_jitter = f" dj={dj:+.0f}"
                prev_jitter = tick_s['jitter']

                d_temp = ""
                t = gm.get('temp_gfx', 0)
                if prev_temp is not None and t > 0:
                    dt = t - prev_temp
                    d_temp = f" dt={dt:+.1f}"
                prev_temp = t

                print(f"{step:>4} {scenario_name:<16} "
                      f"{tick_s['mean']:>10.0f} {tick_s['std']:>10.1f} {tick_s['jitter']:>10.0f} "
                      f"{tick_s['tail_ratio']*100:>6.2f}% {tick_s['iqr']:>10.0f} "
                      f"{err_s['abs_mean']:>10.6f} {err_s['rel_mean']:>10.6f} "
                      f"{err_s['abs_std']:>10.6f} {err_s['large_frac']:>7.4f} "
                      f"{smn_s['adc_low8']:>8d} {smn_s['entropy']:>8d} "
                      f"{gm.get('temp_gfx', 0):>5.1f}C {gm.get('sclk_mhz', 0):>5.0f} "
                      f"{gm.get('gfx_power', 0):>7.0f}mW {gm.get('gfx_activity', 0):>5.1f}% "
                      f"{gm.get('throttle', 0):>5d}"
                      f"{d_jitter}{d_temp}")

                step += 1
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[Interrupted]")
    finally:
        restore_dvfs()

    # ── Quick summary statistics ──────────────────────────
    print("\n" + "=" * 80)
    print("DONE — Restore DVFS to auto")
    print("=" * 80)


if __name__ == '__main__':
    main()
