#!/usr/bin/env python3
"""
z2117 Deep Signal Probe — v9 Pulse Embodiment kernel diagnostics
================================================================
Standalone diagnostic probing pulse field propagation, WGP spatial
structure, per-channel SNR, DVFS step response, gain modulation vs
noise floor, determinism structure, multi-layer chain coherence,
and cross-signal correlations.

Usage:
  sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTORCH_ROCM_ARCH=gfx1100 \
    venv/bin/python -u scripts/z2117_deep_signal_probe.py
"""
import os, sys, struct, time, json, math
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats

# ── Constants ──────────────────────────────────────────────────
M, K, N = 128, 4096, 4096  # 4*128 = 512 blocks
TILE = 32
N_BLOCKS = (M // TILE) * (N // TILE)  # 512
PULSE_BUF_SIZE = 8192
DVFS_SETTLE_S = 1.5
PULSE_ALPHA = 0.85
PULSE_GAIN = 2.0
PULSE_EPS = 0.02

# ── HIP Kernel Source (v9 — copied exactly) ───────────────────
V9_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#define TILE_SIZE 32

__global__ void pulse_embodiment_v9_kernel(
    const __hip_bfloat16* __restrict__ A,
    const __hip_bfloat16* __restrict__ B,
    __hip_bfloat16* __restrict__ C,
    float* __restrict__ pulse_in,
    float* __restrict__ pulse_out,
    unsigned long long* __restrict__ proprio_out,
    float* __restrict__ correction_out,
    unsigned int* __restrict__ wgp_out,
    unsigned int* __restrict__ exec_out,
    unsigned long long* __restrict__ tile_var_out,
    int M, int K, int N,
    int base_round_mode,
    int stress_threshold,
    float pulse_alpha,
    float pulse_gain,
    float pulse_eps,
    int enable_timing,
    int enable_tile_var,
    int enable_wgp,
    int enable_occupancy,
    int enable_pulse_read,
    int enable_pulse_write,
    int enable_gain_mod
) {
    unsigned long long t_start = wall_clock64();
    unsigned int old_mode;
    asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" ::: "memory");
    asm volatile("s_getreg_b32 %0, hwreg(1, 0, 8)" : "=s"(old_mode) :: "memory");
    unsigned int hw_id1;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw_id1) :: "memory");
    unsigned int wgp_id = (hw_id1 >> 7) & 0xF;
    unsigned int exec_lo, exec_hi;
    asm volatile("s_mov_b32 %0, exec_lo" : "=s"(exec_lo));
    asm volatile("s_mov_b32 %0, exec_hi" : "=s"(exec_hi));
    unsigned int active_lanes = __builtin_popcount(exec_lo) + __builtin_popcount(exec_hi);

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float acc = 0.0f;
    int n_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    unsigned int prev_tile_bits = wgp_id;
    unsigned int feedback_accum = 0;
    unsigned long long prev_tile_dt = 0;
    unsigned long long tile_timing_var = 0;

    for (int t = 0; t < n_tiles; t++) {
        unsigned long long tile_start = wall_clock64();
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) As[threadIdx.y][threadIdx.x] = __bfloat162float(A[row * K + a_col]);
        else As[threadIdx.y][threadIdx.x] = 0.0f;
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (col < N && b_row < K) Bs[threadIdx.y][threadIdx.x] = __bfloat162float(B[col * K + b_row]);
        else Bs[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();
        unsigned int cycles;
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(cycles));
        unsigned int acc_bits = __float_as_uint(acc);
        unsigned int feedback = (acc_bits >> 16) ^ prev_tile_bits ^ (cycles & 0xFFu);
        unsigned int rm = feedback & 0x3u;
        unsigned int hw_spin = cycles & 0xFFu;
        if ((int)hw_spin < stress_threshold) rm = 0x1u;
        unsigned int rm_both = (rm & 0x3u) | ((rm & 0x3u) << 2) | (old_mode & 0xF0u);
        unsigned int new_mode = (old_mode & ~0xFFu) | rm_both;
        unsigned int new_mode_s = __builtin_amdgcn_readfirstlane(new_mode);
        asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" :: "s"(new_mode_s) : "memory");
        __half chunk_acc = __float2half(0.0f);
        for (int k = 0; k < TILE_SIZE; k++) {
            __half a_h = __float2half(As[threadIdx.y][k]);
            __half b_h = __float2half(Bs[k][threadIdx.x]);
            chunk_acc = __hadd(chunk_acc, __hmul(a_h, b_h));
        }
        acc += __half2float(chunk_acc);
        prev_tile_bits = (__float_as_uint(acc) >> 24) ^ (rm << 8);
        feedback_accum ^= (__float_as_uint(acc) >> 16);
        unsigned long long tile_end = wall_clock64();
        unsigned long long tile_dt = tile_end - tile_start;
        if (t > 0) {
            long long diff = (long long)tile_dt - (long long)prev_tile_dt;
            tile_timing_var += (unsigned long long)(diff * diff);
        }
        prev_tile_dt = tile_dt;
        __syncthreads();
    }

    unsigned long long t_end = wall_clock64();
    unsigned long long dt = t_end - t_start;
    int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    float norm_dt = enable_timing ? ((float)(dt & 0xFFFFu) * 1.52587890625e-5f - 0.5f) : 0.0f;
    float norm_var = enable_tile_var ? tanhf((float)tile_timing_var * 1e-8f) : 0.0f;
    float wgp_phase = enable_wgp ? ((float)(wgp_id & 0x7u) * 0.142857f - 0.5f) : 0.0f;
    float occ_norm = enable_occupancy ? ((float)active_lanes / 64.0f - 0.5f) : 0.0f;
    float sensed = norm_dt + norm_var + wgp_phase + occ_norm;
    float s_prev = 0.0f;
    if (enable_pulse_read && pulse_in != nullptr && block_id < 8192) s_prev = pulse_in[block_id];
    float s_new = pulse_alpha * s_prev + (1.0f - pulse_alpha) * tanhf(pulse_gain * sensed);
    if (enable_pulse_write && pulse_out != nullptr && block_id < 8192) pulse_out[block_id] = s_new;
    if (enable_gain_mod) acc *= exp2f(pulse_eps * s_new);
    if (row < M && col < N) C[row * N + col] = __float2bfloat16(acc);
    asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" ::: "memory");
    asm volatile("s_setreg_b32 hwreg(1, 0, 8), %0" :: "s"(old_mode) : "memory");
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        proprio_out[block_id] = dt;
        correction_out[block_id] = s_new;
        wgp_out[block_id] = wgp_id;
        exec_out[block_id] = active_lanes;
        tile_var_out[block_id] = tile_timing_var;
    }
}

std::vector<torch::Tensor> analog_gemm_v9(
    torch::Tensor A, torch::Tensor weight,
    int round_mode, float continuous_stress,
    torch::Tensor pulse_in, torch::Tensor pulse_out,
    float pulse_alpha, float pulse_gain, float pulse_eps,
    int enable_timing, int enable_tile_var, int enable_wgp,
    int enable_occupancy, int enable_pulse_read, int enable_pulse_write,
    int enable_gain_mod
) {
    TORCH_CHECK(A.is_cuda() && weight.is_cuda(), "Tensors must be on GPU");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bf16");
    TORCH_CHECK(weight.dtype() == torch::kBFloat16, "weight must be bf16");
    int M = A.size(0), K = A.size(1), N = weight.size(0);
    TORCH_CHECK(weight.size(1) == K, "Dimension mismatch");
    auto C = torch::empty({M, N}, A.options());
    const int TILE = 32;
    dim3 block(TILE, TILE); dim3 grid((N+TILE-1)/TILE, (M+TILE-1)/TILE);
    int n_blocks = grid.x * grid.y;
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(A.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    auto opts_u32 = torch::TensorOptions().dtype(torch::kInt32).device(A.device());
    auto proprio = torch::empty({n_blocks}, opts_i64);
    auto corr = torch::empty({n_blocks}, opts_f32);
    auto wgp = torch::empty({n_blocks}, opts_u32);
    auto exec_cnt = torch::empty({n_blocks}, opts_u32);
    auto tvar = torch::empty({n_blocks}, opts_i64);
    int st = (int)(continuous_stress * 255.0f);
    st = st < 0 ? 0 : (st > 255 ? 255 : st);
    pulse_embodiment_v9_kernel<<<grid, block>>>(
        reinterpret_cast<const __hip_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),
        pulse_in.data_ptr<float>(), pulse_out.data_ptr<float>(),
        reinterpret_cast<unsigned long long*>(proprio.data_ptr()),
        corr.data_ptr<float>(),
        reinterpret_cast<unsigned int*>(wgp.data_ptr()),
        reinterpret_cast<unsigned int*>(exec_cnt.data_ptr()),
        reinterpret_cast<unsigned long long*>(tvar.data_ptr()),
        M, K, N, round_mode, st,
        pulse_alpha, pulse_gain, pulse_eps,
        enable_timing, enable_tile_var, enable_wgp,
        enable_occupancy, enable_pulse_read, enable_pulse_write,
        enable_gain_mod);
    return {C, proprio, corr, wgp, exec_cnt, tvar};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("analog_gemm_v9", &analog_gemm_v9, "v9 Pulse Embodiment GEMM");
}
"""

HIP_MOD = None

def compile_kernel():
    global HIP_MOD
    from torch.utils.cpp_extension import load_inline
    HIP_MOD = load_inline(
        name='deep_probe_v9', cpp_sources=[], cuda_sources=[V9_SRC],
        extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'],
        verbose=True, with_cuda=True)
    print("[HIP] v9 kernel compiled OK")

# ── Hardware access ────────────────────────────────────────────
GPU_METRICS_PATH = None
DVFS_PATH = None

def find_hw():
    global GPU_METRICS_PATH, DVFS_PATH
    for card in ['card1', 'card0']:
        base = f'/sys/class/drm/{card}/device'
        gm = f'{base}/gpu_metrics'
        dv = f'{base}/power_dpm_force_performance_level'
        if os.path.exists(gm) and GPU_METRICS_PATH is None:
            GPU_METRICS_PATH = gm
        if os.path.exists(dv) and DVFS_PATH is None:
            DVFS_PATH = dv
    print(f"[HW] gpu_metrics={GPU_METRICS_PATH is not None}, dvfs={DVFS_PATH is not None}")

def read_gpu_metrics():
    if not GPU_METRICS_PATH: return {}
    try:
        with open(GPU_METRICS_PATH, 'rb') as f: data = f.read()
        if len(data) < 100: return {}
        r = {'temp_gfx': struct.unpack_from('<H', data, 4)[0]/100.0,
             'gfx_power': struct.unpack_from('<H', data, 66)[0],
             'socket_power': struct.unpack_from('<H', data, 60)[0]}
        if len(data) > 176: r['sclk_mhz'] = struct.unpack_from('<H', data, 174)[0]
        return r
    except: return {}

def set_dvfs(name):
    if not DVFS_PATH: return
    torch.cuda.synchronize()
    try:
        with open(DVFS_PATH, 'w') as f: f.write(name)
    except: pass
    time.sleep(DVFS_SETTLE_S)

def restore_dvfs():
    if DVFS_PATH:
        try:
            with open(DVFS_PATH, 'w') as f: f.write('auto')
        except: pass

# ── Kernel wrapper ─────────────────────────────────────────────
def run_v9(A, W, pin, pout, alpha=PULSE_ALPHA, gain=PULSE_GAIN, eps=PULSE_EPS,
           et=1, ev=1, ew=1, eo=1, epr=1, epw=1, egm=1):
    """Run v9 kernel, return dict of numpy arrays."""
    res = HIP_MOD.analog_gemm_v9(A, W, 0, 0.0, pin, pout,
                                  alpha, gain, eps, et, ev, ew, eo, epr, epw, egm)
    torch.cuda.synchronize()
    C, proprio, corr, wgp, exc, tvar = res
    nb = proprio.shape[0]
    return {'C': C, 'ticks': proprio[:nb].cpu().numpy().astype(np.float64),
            'pulse': corr[:nb].cpu().numpy(), 'wgp': wgp[:nb].cpu().numpy().astype(np.int32),
            'exec': exc[:nb].cpu().numpy().astype(np.int32),
            'tvar': tvar[:nb].cpu().numpy().astype(np.float64),
            'pulse_buf': pout[:nb].cpu().numpy()}

def fresh_bufs():
    return (torch.zeros(PULSE_BUF_SIZE, device='cuda', dtype=torch.float32),
            torch.zeros(PULSE_BUF_SIZE, device='cuda', dtype=torch.float32))

# ── P1: Pulse Memory Depth ─────────────────────────────────────
def probe_p1(A, W):
    print("\n" + "="*70)
    print("P1: PULSE MEMORY DEPTH — leaky integrator decay analysis")
    print("="*70)
    results = {}
    for alpha in [0.5, 0.7, 0.85, 0.95, 0.99]:
        n_steps = 60
        pin = torch.ones(PULSE_BUF_SIZE, device='cuda', dtype=torch.float32)
        pout = torch.zeros(PULSE_BUF_SIZE, device='cuda', dtype=torch.float32)
        trace = []
        for i in range(n_steps):
            r = run_v9(A, W, pin, pout, alpha=alpha, eps=0.0, egm=0)
            pb = pout[:N_BLOCKS].cpu().numpy()
            trace.append({'mean': float(np.mean(pb)), 'std': float(np.std(pb)),
                          'max': float(np.max(pb)), 'min': float(np.min(pb))})
            pin.copy_(pout)
            pout.zero_()
        # Find step where mean < 1% of initial
        initial = trace[0]['mean'] if abs(trace[0]['mean']) > 1e-9 else 1.0
        decay_step = n_steps
        for j, t in enumerate(trace):
            if abs(t['mean']) < abs(initial) * 0.01:
                decay_step = j; break
        results[f'alpha={alpha}'] = {'trace': trace, 'decay_to_1pct': decay_step}
        print(f"  alpha={alpha:.2f}: decay to <1% at step {decay_step}/{n_steps}, "
              f"final mean={trace[-1]['mean']:.6f}, final std={trace[-1]['std']:.6f}")
    return results

# ── P2: WGP Spatial Correlation ────────────────────────────────
def probe_p2(A, W):
    print("\n" + "="*70)
    print("P2: WGP SPATIAL CORRELATION — within vs between WGP variance")
    print("="*70)
    n_runs = 50
    all_wgp, all_ticks, all_pulse = [], [], []
    for _ in range(n_runs):
        pin, pout = fresh_bufs()
        r = run_v9(A, W, pin, pout)
        all_wgp.append(r['wgp'][:N_BLOCKS])
        all_ticks.append(r['ticks'][:N_BLOCKS])
        all_pulse.append(r['pulse'][:N_BLOCKS])
    wgp_ids = np.stack(all_wgp)   # [50, 512]
    ticks = np.stack(all_ticks)
    pulses = np.stack(all_pulse)

    # Use first run's WGP assignment (should be stable)
    wgp_map = wgp_ids[0]
    unique_wgps = np.unique(wgp_map)
    print(f"  Unique WGPs: {unique_wgps.tolist()} ({len(unique_wgps)} found)")

    # F-test: WGP as factor for timing (pooled across runs)
    ticks_flat = ticks.flatten()
    wgp_flat = np.tile(wgp_map, n_runs)
    groups = [ticks_flat[wgp_flat == w] for w in unique_wgps if np.sum(wgp_flat == w) > 1]
    if len(groups) >= 2:
        F_tick, p_tick = sp_stats.f_oneway(*groups)
    else:
        F_tick, p_tick = 0.0, 1.0
    print(f"  Timing F-stat={F_tick:.2f}, p={p_tick:.2e} ({'SIGNIFICANT' if p_tick < 0.01 else 'not sig'})")

    # Same for pulse
    pulse_flat = pulses.flatten()
    pgroups = [pulse_flat[wgp_flat == w] for w in unique_wgps if np.sum(wgp_flat == w) > 1]
    if len(pgroups) >= 2:
        F_pulse, p_pulse = sp_stats.f_oneway(*pgroups)
    else:
        F_pulse, p_pulse = 0.0, 1.0
    print(f"  Pulse  F-stat={F_pulse:.2f}, p={p_pulse:.2e} ({'SIGNIFICANT' if p_pulse < 0.01 else 'not sig'})")

    # Within vs between variance
    within_vars, between_means = [], []
    for w in unique_wgps:
        mask = wgp_map == w
        wg_ticks = ticks[:, mask]  # [50, n_in_wgp]
        within_vars.append(np.mean(np.var(wg_ticks, axis=1)))
        between_means.append(np.mean(wg_ticks))
    within_mean = np.mean(within_vars)
    between_var = np.var(between_means)
    print(f"  Within-WGP var (mean): {within_mean:.1f}")
    print(f"  Between-WGP var:       {between_var:.1f}")
    print(f"  Ratio (between/within): {between_var/max(within_mean,1e-9):.4f}")

    return {'F_tick': float(F_tick), 'p_tick': float(p_tick),
            'F_pulse': float(F_pulse), 'p_pulse': float(p_pulse),
            'within_var': float(within_mean), 'between_var': float(between_var),
            'n_wgps': len(unique_wgps), 'wgp_ids': unique_wgps.tolist()}

# ── P3: Per-Channel SNR ───────────────────────────────────────
def probe_p3(A, W):
    print("\n" + "="*70)
    print("P3: PER-CHANNEL SNR — individual channel contribution under DVFS")
    print("="*70)
    channels = [
        ('timing',    {'et':1,'ev':0,'ew':0,'eo':0}),
        ('tile_var',  {'et':0,'ev':1,'ew':0,'eo':0}),
        ('wgp',       {'et':0,'ev':0,'ew':1,'eo':0}),
        ('occupancy', {'et':0,'ev':0,'ew':0,'eo':1}),
        ('none',      {'et':0,'ev':0,'ew':0,'eo':0}),
        ('all',       {'et':1,'ev':1,'ew':1,'eo':1}),
    ]
    results = {}
    for dvfs_name in ['low', 'high']:
        set_dvfs(dvfs_name)
        dvfs_res = {}
        for ch_name, flags in channels:
            pulse_stds, pulse_means, pulse_ranges = [], [], []
            for _ in range(10):
                pin, pout = fresh_bufs()
                r = run_v9(A, W, pin, pout, epr=0, epw=1, egm=0, **flags)
                pb = r['pulse_buf'][:N_BLOCKS]
                pulse_stds.append(float(np.std(pb)))
                pulse_means.append(float(np.mean(pb)))
                pulse_ranges.append(float(np.ptp(pb)))
            dvfs_res[ch_name] = {
                'std': float(np.mean(pulse_stds)),
                'mean': float(np.mean(pulse_means)),
                'range': float(np.mean(pulse_ranges)),
            }
            print(f"  [{dvfs_name:4s}] {ch_name:10s}: std={dvfs_res[ch_name]['std']:.6f}, "
                  f"range={dvfs_res[ch_name]['range']:.6f}, mean={dvfs_res[ch_name]['mean']:.6f}")
        results[dvfs_name] = dvfs_res

    # Compute per-channel signal = |high - low| difference
    print("\n  Per-channel DVFS signal (|high_std - low_std|):")
    for ch_name, _ in channels:
        sig = abs(results['high'][ch_name]['std'] - results['low'][ch_name]['std'])
        noise = results['low']['none']['std'] + 1e-9
        snr = sig / noise
        results.setdefault('snr', {})[ch_name] = {'signal': float(sig), 'snr': float(snr)}
        print(f"    {ch_name:10s}: signal={sig:.6f}, SNR={snr:.2f}")
    restore_dvfs()
    return results

# ── P4: DVFS Step Response ────────────────────────────────────
def probe_p4(A, W):
    print("\n" + "="*70)
    print("P4: DVFS STEP RESPONSE — signal latency after DVFS switch")
    print("="*70)
    set_dvfs('low')
    time.sleep(2.0)
    baseline = []
    for _ in range(10):
        pin, pout = fresh_bufs()
        r = run_v9(A, W, pin, pout)
        gm = read_gpu_metrics()
        baseline.append({
            'tick_mean': float(np.mean(r['ticks'][:N_BLOCKS])),
            'tick_std': float(np.std(r['ticks'][:N_BLOCKS])),
            'tvar_mean': float(np.mean(r['tvar'][:N_BLOCKS])),
            'pulse_mean': float(np.mean(r['pulse'][:N_BLOCKS])),
            'pulse_std': float(np.std(r['pulse'][:N_BLOCKS])),
            **{f'gm_{k}': v for k, v in gm.items()},
        })
    # Compute baseline stats
    bl_means = {k: np.mean([b[k] for b in baseline]) for k in baseline[0] if isinstance(baseline[0][k], float)}
    bl_stds = {k: np.std([b[k] for b in baseline]) + 1e-9 for k in bl_means}

    # Switch to high
    set_dvfs('high')
    response = []
    for i in range(30):
        pin, pout = fresh_bufs()
        r = run_v9(A, W, pin, pout)
        gm = read_gpu_metrics()
        step = {'step': i,
                'tick_mean': float(np.mean(r['ticks'][:N_BLOCKS])),
                'tick_std': float(np.std(r['ticks'][:N_BLOCKS])),
                'tvar_mean': float(np.mean(r['tvar'][:N_BLOCKS])),
                'pulse_mean': float(np.mean(r['pulse'][:N_BLOCKS])),
                'pulse_std': float(np.std(r['pulse'][:N_BLOCKS])),
                **{f'gm_{k}': v for k, v in gm.items()}}
        response.append(step)

    # Find first significant change per signal
    signals = ['tick_mean', 'tick_std', 'tvar_mean', 'pulse_mean', 'pulse_std']
    first_change = {}
    for sig in signals:
        if sig not in bl_means: continue
        for i, s in enumerate(response):
            if sig in s and abs(s[sig] - bl_means[sig]) > 2 * bl_stds[sig]:
                first_change[sig] = i; break
        else:
            first_change[sig] = -1  # never changed
        print(f"  {sig:15s}: baseline={bl_means[sig]:.1f} +/- {bl_stds[sig]:.1f}, "
              f"first change at step {first_change[sig]}")

    restore_dvfs()
    return {'baseline': baseline, 'response': response, 'first_change': first_change}

# ── P5: Gain Modulation vs Noise Floor ────────────────────────
def probe_p5(A, W):
    print("\n" + "="*70)
    print("P5: GAIN MODULATION vs NOISE FLOOR — eps sweep")
    print("="*70)
    ref = F.linear(A.float(), W.float())
    results = {}
    for eps_val in [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]:
        errs = []
        for _ in range(5):
            pin = torch.full((PULSE_BUF_SIZE,), 0.5, device='cuda', dtype=torch.float32)
            pout = torch.zeros(PULSE_BUF_SIZE, device='cuda', dtype=torch.float32)
            r = run_v9(A, W, pin, pout, eps=eps_val, epr=1, epw=1, egm=int(eps_val > 0))
            err = (r['C'].float() - ref).abs()
            errs.append({'abs_mean': float(err.mean()), 'abs_max': float(err.max()),
                         'abs_std': float(err.std())})
        avg = {k: float(np.mean([e[k] for e in errs])) for k in errs[0]}
        results[f'eps={eps_val}'] = avg
        print(f"  eps={eps_val:.3f}: abs_mean={avg['abs_mean']:.6f}, "
              f"abs_max={avg['abs_max']:.4f}, abs_std={avg['abs_std']:.6f}")
    # Compute gain signal vs rounding noise
    rounding = results['eps=0.0']['abs_mean']
    print(f"\n  Rounding noise floor (eps=0): {rounding:.6f}")
    for k, v in results.items():
        if k == 'eps=0.0': continue
        gain_sig = v['abs_mean'] - rounding
        ratio = gain_sig / max(rounding, 1e-9)
        results[k]['gain_over_noise'] = float(ratio)
        print(f"  {k}: gain_signal={gain_sig:.6f}, gain/noise={ratio:.2f}x")
    return results

# ── P6: Determinism Structure ─────────────────────────────────
def probe_p6(A, W):
    print("\n" + "="*70)
    print("P6: DETERMINISM STRUCTURE — same input variance analysis")
    print("="*70)
    n_runs = 20
    outputs, pulses, wgps_all = [], [], []
    for _ in range(n_runs):
        pin, pout = fresh_bufs()
        r = run_v9(A, W, pin, pout)
        outputs.append(r['C'].float().cpu().numpy())
        pulses.append(r['pulse_buf'][:N_BLOCKS])
        wgps_all.append(r['wgp'][:N_BLOCKS])
    outputs = np.stack(outputs)  # [20, M, N]
    pulses = np.stack(pulses)    # [20, 512]

    out_var = np.var(outputs, axis=0)  # [M, N]
    pulse_var = np.var(pulses, axis=0) # [512]
    print(f"  Output variance: mean={out_var.mean():.8f}, max={out_var.max():.6f}")
    print(f"  Pulse variance:  mean={pulse_var.mean():.8f}, max={pulse_var.max():.6f}")

    # WGP-grouped pulse variance
    wgp_map = wgps_all[0]
    unique_wgps = np.unique(wgp_map)
    wgp_pulse_vars = {}
    for w in unique_wgps:
        mask = wgp_map == w
        wgp_pulse_vars[int(w)] = float(np.mean(pulse_var[mask]))
    print(f"  Pulse var by WGP: {wgp_pulse_vars}")

    # Is pulse variance white or structured?
    if len(unique_wgps) >= 2:
        groups = [pulse_var[wgp_map == w] for w in unique_wgps if np.sum(wgp_map == w) > 1]
        if len(groups) >= 2:
            F_val, p_val = sp_stats.f_oneway(*groups)
        else:
            F_val, p_val = 0.0, 1.0
        print(f"  Pulse var WGP F-test: F={F_val:.2f}, p={p_val:.2e} "
              f"({'STRUCTURED' if p_val < 0.05 else 'white noise'})")
    else:
        F_val, p_val = 0.0, 1.0

    # Correlation: output variance vs pulse variance (per block-row)
    # Map blocks to output rows: block row = blockIdx.y, each covers TILE rows
    n_brow = M // TILE
    n_bcol = N // TILE
    block_out_var = np.zeros(N_BLOCKS)
    for by in range(n_brow):
        for bx in range(n_bcol):
            bid = by * n_bcol + bx
            if bid < N_BLOCKS:
                row_s, row_e = by*TILE, min((by+1)*TILE, M)
                col_s, col_e = bx*TILE, min((bx+1)*TILE, N)
                block_out_var[bid] = out_var[row_s:row_e, col_s:col_e].mean()
    corr_r, corr_p = sp_stats.pearsonr(block_out_var[:N_BLOCKS], pulse_var[:N_BLOCKS])
    print(f"  Output-pulse variance correlation: r={corr_r:.4f}, p={corr_p:.2e}")

    return {'out_var_mean': float(out_var.mean()), 'out_var_max': float(out_var.max()),
            'pulse_var_mean': float(pulse_var.mean()), 'pulse_var_max': float(pulse_var.max()),
            'wgp_pulse_vars': wgp_pulse_vars, 'F_val': float(F_val), 'p_val': float(p_val),
            'out_pulse_corr': float(corr_r), 'out_pulse_corr_p': float(corr_p)}

# ── P7: Multi-Layer Chain ─────────────────────────────────────
def probe_p7(A, W):
    print("\n" + "="*70)
    print("P7: MULTI-LAYER CHAIN — pulse field evolution across 8 layers")
    print("="*70)
    n_layers = 8
    # Different weights per layer (fixed random)
    torch.manual_seed(42)
    weights = [torch.randn(N, K, device='cuda', dtype=torch.bfloat16) * 0.02 for _ in range(n_layers)]
    # Use same A for all layers (simulating fixed input to each layer's GEMM)

    results = {'with_gain': {}, 'no_gain': {}}
    for mode_name, egm_val in [('with_gain', 1), ('no_gain', 0)]:
        pin = torch.zeros(PULSE_BUF_SIZE, device='cuda', dtype=torch.float32)
        layer_traces = []
        for layer_i in range(n_layers):
            pout = torch.zeros(PULSE_BUF_SIZE, device='cuda', dtype=torch.float32)
            r = run_v9(A, weights[layer_i], pin, pout, egm=egm_val)
            pb = pout[:N_BLOCKS].cpu().numpy()
            layer_traces.append({
                'mean': float(np.mean(pb)), 'std': float(np.std(pb)),
                'max': float(np.max(pb)), 'min': float(np.min(pb)),
                'range': float(np.ptp(pb)),
            })
            pin = pout.clone()

        # Spatial correlation between consecutive layers
        # Rerun to get pairs
        pin = torch.zeros(PULSE_BUF_SIZE, device='cuda', dtype=torch.float32)
        prev_pulse = None
        layer_corrs = []
        for layer_i in range(n_layers):
            pout = torch.zeros(PULSE_BUF_SIZE, device='cuda', dtype=torch.float32)
            run_v9(A, weights[layer_i], pin, pout, egm=egm_val)
            cur_pulse = pout[:N_BLOCKS].cpu().numpy()
            if prev_pulse is not None and np.std(cur_pulse) > 1e-9 and np.std(prev_pulse) > 1e-9:
                cr, _ = sp_stats.pearsonr(prev_pulse, cur_pulse)
                layer_corrs.append(float(cr))
            else:
                layer_corrs.append(0.0)
            prev_pulse = cur_pulse.copy()
            pin = pout.clone()

        # Check convergence: is the last layer pulse similar to layer before?
        convergence = 'unknown'
        if len(layer_traces) >= 3:
            last_std = layer_traces[-1]['std']
            mid_std = layer_traces[n_layers//2]['std']
            if last_std < 0.001:
                convergence = 'collapsed'
            elif abs(last_std - mid_std) / max(mid_std, 1e-9) < 0.1:
                convergence = 'stable'
            elif last_std > mid_std * 2:
                convergence = 'diverging'
            else:
                convergence = 'structured'

        results[mode_name] = {'traces': layer_traces, 'inter_layer_corr': layer_corrs,
                              'convergence': convergence}
        print(f"  [{mode_name}] convergence={convergence}")
        for i, t in enumerate(layer_traces):
            corr_str = f", corr_prev={layer_corrs[i]:.3f}" if i > 0 else ""
            print(f"    L{i}: mean={t['mean']:.5f}, std={t['std']:.5f}, range={t['range']:.5f}{corr_str}")
    return results

# ── P8: Correlation Matrix ────────────────────────────────────
def probe_p8(A, W):
    print("\n" + "="*70)
    print("P8: MASTER CORRELATION MATRIX — cross-signal relationships")
    print("="*70)
    n_runs = 30
    records = []
    for _ in range(n_runs):
        pin, pout = fresh_bufs()
        r = run_v9(A, W, pin, pout)
        nb = min(N_BLOCKS, len(r['ticks']))
        gm = read_gpu_metrics()
        # Reference error
        ref = F.linear(A.float(), W.float())
        err = (r['C'].float() - ref).abs().mean().item()
        records.append({
            'tick_mean': float(np.mean(r['ticks'][:nb])),
            'tick_std': float(np.std(r['ticks'][:nb])),
            'tvar_mean': float(np.mean(r['tvar'][:nb])),
            'pulse_mean': float(np.mean(r['pulse'][:nb])),
            'pulse_std': float(np.std(r['pulse'][:nb])),
            'wgp_diversity': float(len(np.unique(r['wgp'][:nb]))),
            'exec_mean': float(np.mean(r['exec'][:nb])),
            'output_error': err,
            'gpu_temp': gm.get('temp_gfx', 0.0),
            'gpu_power': gm.get('gfx_power', 0.0),
            'sclk': gm.get('sclk_mhz', 0.0),
        })

    keys = list(records[0].keys())
    n_keys = len(keys)
    data = np.array([[r[k] for k in keys] for r in records])
    corr_matrix = np.corrcoef(data.T)

    # Top correlations
    pairs = []
    for i in range(n_keys):
        for j in range(i+1, n_keys):
            if np.isfinite(corr_matrix[i, j]):
                pairs.append((keys[i], keys[j], corr_matrix[i, j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\n  Top-10 correlations (from {n_runs} runs):")
    for rank, (a, b, r) in enumerate(pairs[:10]):
        tag = "***" if abs(r) > 0.7 else ""
        print(f"    {rank+1:2d}. {a:15s} x {b:15s} = {r:+.4f} {tag}")

    # Flag anti-correlations
    anti = [(a, b, r) for a, b, r in pairs if r < -0.5]
    if anti:
        print(f"\n  Surprising anti-correlations:")
        for a, b, r in anti:
            print(f"    {a} x {b} = {r:+.4f}")

    return {'correlations': {f'{a}_x_{b}': float(r) for a, b, r in pairs},
            'matrix': corr_matrix.tolist(), 'keys': keys}

# ── Main ──────────────────────────────────────────────────────
def main():
    print("="*70)
    print("z2117 DEEP SIGNAL PROBE — v9 Pulse Embodiment Kernel Diagnostics")
    print("="*70)
    print(f"Matrix: M={M}, K={K}, N={N} -> {N_BLOCKS} blocks")

    find_hw()
    print("\n[GPU] Warming up...")
    x = torch.randn(64, 64, device='cuda')
    for _ in range(10): x = x @ x.T
    torch.cuda.synchronize()
    print("[GPU] OK")

    compile_kernel()

    A = torch.randn(M, K, device='cuda', dtype=torch.bfloat16) * 0.02
    W = torch.randn(N, K, device='cuda', dtype=torch.bfloat16) * 0.02

    all_results = {}
    try:
        all_results['P1_pulse_memory'] = probe_p1(A, W)
        all_results['P2_wgp_spatial'] = probe_p2(A, W)
        all_results['P3_channel_snr'] = probe_p3(A, W)
        all_results['P4_dvfs_step'] = probe_p4(A, W)
        all_results['P5_gain_vs_noise'] = probe_p5(A, W)
        all_results['P6_determinism'] = probe_p6(A, W)
        all_results['P7_multilayer'] = probe_p7(A, W)
        all_results['P8_correlations'] = probe_p8(A, W)
    finally:
        restore_dvfs()

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'results', 'z2117_deep_probe.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Make JSON-serializable: convert any remaining numpy types
    def jsonify(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [jsonify(v) for v in obj]
        return obj

    with open(out_path, 'w') as f:
        json.dump(jsonify(all_results), f, indent=2)
    print(f"\n{'='*70}")
    print(f"All results saved to {out_path}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
