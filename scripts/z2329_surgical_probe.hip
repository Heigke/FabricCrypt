/**
 * z2329_surgical_probe.hip — Per-CU, Per-Bank, Per-Set Timing Probes
 * ====================================================================
 * Surgical actuators for transistor-level characterization:
 *   1. wall_clock64() for fixed 100MHz timing (DVFS-invariant)
 *   2. clock64() for core-clock timing (DVFS-sensitive)
 *   3. s_getreg(HW_ID1) for physical CU/WGP/SE/SA identification
 *   4. LDS bank targeting via stride patterns
 *   5. Cache set targeting via address alignment
 *   6. Per-CU process variation detection
 *
 * Build:
 *   hipcc -shared -fPIC -O2 --offload-arch=gfx1100 \
 *     -o scripts/z2329_surgical_probe.so scripts/z2329_surgical_probe.hip
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

// =====================================================================
// HW_ID register decoding (RDNA 3.5)
// =====================================================================
#define HWREG(id, offset, size) ((id) | ((offset) << 6) | (((size)-1) << 11))
#define HW_REG_HW_ID1 23

struct CULocation {
    unsigned wave_id;
    unsigned simd_id;
    unsigned wgp_id;
    unsigned sa_id;
    unsigned se_id;
};

__device__ CULocation get_cu_location() {
    unsigned hw = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID1, 0, 32));
    CULocation loc;
    loc.wave_id = (hw >> 0) & 0xF;
    loc.simd_id = (hw >> 4) & 0x3;
    loc.wgp_id  = (hw >> 8) & 0xF;
    loc.sa_id   = (hw >> 12) & 0x1;
    loc.se_id   = (hw >> 13) & 0x7;
    return loc;
}

// =====================================================================
// Probe 1: Per-CU Compute Timing
// Each thread does identical ALU work, reports wall_clock + clock64 + CU ID
// Timing differences at same voltage = process variation (ΔVth)
// =====================================================================
struct ProbeResult {
    float wall_ns;      // wall_clock64 delta (fixed 100MHz = 10ns resolution)
    float core_cycles;  // clock64 delta (varies with DVFS)
    float cu_id;        // physical WGP ID
    float se_sa;        // SE*2 + SA encoded
};

__global__ void probe_cu_compute(
    float4* __restrict__ out,  // [n] ProbeResult as float4
    int n_iters,
    int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    // Identical work for all threads
    float x = (float)(tid + 1) * 0.001f;

    long long w0 = wall_clock64();
    uint64_t c0 = clock64();

    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        x = __sinf(x) * __cosf(x + 0.1f) + __expf(-x * x);
        x = x * 1.0001f + 0.0001f;
    }

    uint64_t c1 = clock64();
    long long w1 = wall_clock64();

    CULocation loc = get_cu_location();

    float4 r;
    r.x = (float)(w1 - w0) * 10.0f;  // wall_clock ticks → ns (100MHz)
    r.y = (float)(c1 - c0);           // core clock cycles
    r.z = (float)loc.wgp_id;
    r.w = (float)(loc.se_id * 2 + loc.sa_id);
    out[tid] = r;

    // Anti-optimization
    if (__builtin_expect(x == -1e30f, 0)) out[0].x = x;
}

// =====================================================================
// Probe 2: LDS Bank Timing
// Targets a SPECIFIC LDS bank by controlling access stride
// bank = (byte_address / 4) % 32
// Measures timing per bank — timing variation = bank-level Vth variation
// =====================================================================
__global__ void probe_lds_bank(
    float4* __restrict__ out,  // [n] ProbeResult
    int target_bank,           // 0-31: which bank to hammer
    int n_iters,
    int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    // Allocate LDS — 32 banks × 128 entries = 16KB
    __shared__ float lds[4096];

    // Initialize LDS (avoid first-touch penalty)
    for (int i = threadIdx.x; i < 4096; i += blockDim.x)
        lds[i] = (float)i;
    __syncthreads();

    // Target specific bank: address = (target_bank + k*32) for k=0,1,2,...
    // Each float is 4 bytes, so lds[target_bank + k*32] hits bank target_bank
    float sum = 0.0f;

    long long w0 = wall_clock64();
    uint64_t c0 = clock64();

    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        // All threads in wavefront access SAME bank = worst case conflict
        int idx = target_bank + (i % 128) * 32;
        sum += lds[idx];
        lds[idx] = sum * 0.999f;
    }

    uint64_t c1 = clock64();
    long long w1 = wall_clock64();

    CULocation loc = get_cu_location();

    float4 r;
    r.x = (float)(w1 - w0) * 10.0f;
    r.y = (float)(c1 - c0);
    r.z = (float)loc.wgp_id;
    r.w = (float)target_bank;
    out[tid] = r;

    if (__builtin_expect(sum == -1e30f, 0)) out[0].x = sum;
}

// =====================================================================
// Probe 3: Memory Latency (cache set targeting)
// Stride access to hit specific L0V cache sets
// L0V: 32KB, 32-way, 64B lines → 16 sets
// set = (addr >> 6) & 0xF
// =====================================================================
__global__ void probe_cache_set(
    float4* __restrict__ out,  // [n] ProbeResult
    const float* __restrict__ arena,  // Large buffer for cache probing
    int arena_size,
    int target_set,            // 0-15: which L0V set to target
    int n_iters,
    int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    // To hit set S: access addresses where (addr >> 6) & 0xF == S
    // float index: (byte_addr / 4) = (set * 16 + way * 16 * 16) = set*16 + way*256
    // Stride in floats to stay in same set: 256 (= 16 sets * 16 floats/line)
    int base_idx = target_set * 16;  // First float in target set
    int stride = 256;                // 16 sets × 64B/4B = 256 floats per set wrap

    float sum = 0.0f;

    long long w0 = wall_clock64();
    uint64_t c0 = clock64();

    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        // Access 33 different cache lines all mapping to same set
        // (32-way + 1 = guaranteed eviction → measure miss latency)
        int way = i % 33;
        int idx = (base_idx + way * stride + tid) % arena_size;
        sum += arena[idx];
    }

    uint64_t c1 = clock64();
    long long w1 = wall_clock64();

    CULocation loc = get_cu_location();

    float4 r;
    r.x = (float)(w1 - w0) * 10.0f;
    r.y = (float)(c1 - c0);
    r.z = (float)loc.wgp_id;
    r.w = (float)target_set;
    out[tid] = r;

    if (__builtin_expect(sum == -1e30f, 0)) out[0].x = sum;
}

// =====================================================================
// Probe 4: Memory vs Compute Ratio (per-CU)
// Same as timing_probe_lib but with wall_clock + CU identification
// The ratio mem_time/compute_time isolates DVFS from process variation
// =====================================================================
__global__ void probe_mem_compute_ratio(
    float4* __restrict__ out,  // [n] {wall_ns, ratio, wgp_id, se_sa}
    const float* __restrict__ workspace,
    int ws_size,
    int n_iters,
    int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    // Compute phase
    float x = (float)(tid + 1) * 0.001f;
    uint64_t cc0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        x = __sinf(x) * __cosf(x + 0.1f);
        x = x * 1.0001f + 0.0001f;
    }
    uint64_t cc1 = clock64();

    // Memory phase
    float sum = 0.0f;
    uint64_t mc0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        int idx = ((tid * 2654435761u + i * 40503u) >> 4) % ws_size;
        sum += workspace[idx];
    }
    uint64_t mc1 = clock64();

    long long w1 = wall_clock64();

    float compute_cycles = (float)(cc1 - cc0);
    float memory_cycles = (float)(mc1 - mc0);
    float ratio = (compute_cycles > 1.0f) ? memory_cycles / compute_cycles : 0.0f;

    CULocation loc = get_cu_location();

    float4 r;
    r.x = (float)w1 * 10.0f;  // timestamp ns
    r.y = ratio;
    r.z = (float)loc.wgp_id;
    r.w = (float)(loc.se_id * 2 + loc.sa_id);
    out[tid] = r;

    if (__builtin_expect((x + sum) == -1e30f, 0)) out[0].x = x;
}

// =====================================================================
// Probe 5: Register File Pressure (VGPR stress)
// Many live variables force deeper register file rows
// Timing variation across VGPR depth = register file bank stalls
// =====================================================================
__global__ void probe_vgpr_depth(
    float4* __restrict__ out,
    int depth,     // 1-8: how many live float4 registers to keep
    int n_iters,
    int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    // Allocate many live variables based on depth parameter
    float4 v0 = make_float4(tid * 0.001f, tid * 0.002f, tid * 0.003f, tid * 0.004f);
    float4 v1 = v0, v2 = v0, v3 = v0, v4 = v0, v5 = v0, v6 = v0, v7 = v0;

    long long w0 = wall_clock64();
    uint64_t c0 = clock64();

    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        v0.x = __sinf(v0.x) + v1.y;
        v0.y = __cosf(v0.y) + v2.z;
        if (depth >= 2) { v1.x += v0.x * 0.01f; v1.y += v0.y * 0.01f; }
        if (depth >= 3) { v2.x += v1.x * 0.01f; v2.y += v1.y * 0.01f; }
        if (depth >= 4) { v3.x += v2.x * 0.01f; v3.y += v2.y * 0.01f; }
        if (depth >= 5) { v4.x += v3.x * 0.01f; v4.y += v3.y * 0.01f; }
        if (depth >= 6) { v5.x += v4.x * 0.01f; v5.y += v4.y * 0.01f; }
        if (depth >= 7) { v6.x += v5.x * 0.01f; v6.y += v5.y * 0.01f; }
        if (depth >= 8) { v7.x += v6.x * 0.01f; v7.y += v6.y * 0.01f; }
    }

    uint64_t c1 = clock64();
    long long w1 = wall_clock64();

    CULocation loc = get_cu_location();

    float total = v0.x + v1.x + v2.x + v3.x + v4.x + v5.x + v6.x + v7.x;
    float4 r;
    r.x = (float)(w1 - w0) * 10.0f;
    r.y = (float)(c1 - c0);
    r.z = (float)loc.wgp_id;
    r.w = (float)depth;
    out[tid] = r;

    if (__builtin_expect(total == -1e30f, 0)) out[0].x = total;
}

// =====================================================================
// C API
// =====================================================================
extern "C" {

int z2329_probe_cu_compute(void* d_out, int n_iters, int n) {
    hipSetDevice(0);
    int threads = 32;  // Single wavefront for minimal interference
    int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(probe_cu_compute, dim3(blocks), dim3(threads), 0, 0,
        (float4*)d_out, n_iters, n);
    return (int)hipDeviceSynchronize();
}

int z2329_probe_lds_bank(void* d_out, int target_bank, int n_iters, int n) {
    hipSetDevice(0);
    int threads = 32;
    int blocks = (n + threads - 1) / threads;
    size_t lds_bytes = 4096 * sizeof(float);  // 16KB LDS
    hipLaunchKernelGGL(probe_lds_bank, dim3(blocks), dim3(threads), lds_bytes, 0,
        (float4*)d_out, target_bank, n_iters, n);
    return (int)hipDeviceSynchronize();
}

int z2329_probe_cache_set(void* d_out, void* d_arena, int arena_size,
                           int target_set, int n_iters, int n) {
    hipSetDevice(0);
    int threads = 32;
    int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(probe_cache_set, dim3(blocks), dim3(threads), 0, 0,
        (float4*)d_out, (const float*)d_arena, arena_size, target_set, n_iters, n);
    return (int)hipDeviceSynchronize();
}

int z2329_probe_mem_compute_ratio(void* d_out, void* d_workspace, int ws_size,
                                    int n_iters, int n) {
    hipSetDevice(0);
    int threads = 32;
    int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(probe_mem_compute_ratio, dim3(blocks), dim3(threads), 0, 0,
        (float4*)d_out, (const float*)d_workspace, ws_size, n_iters, n);
    return (int)hipDeviceSynchronize();
}

int z2329_probe_vgpr_depth(void* d_out, int depth, int n_iters, int n) {
    hipSetDevice(0);
    int threads = 32;
    int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(probe_vgpr_depth, dim3(blocks), dim3(threads), 0, 0,
        (float4*)d_out, depth, n_iters, n);
    return (int)hipDeviceSynchronize();
}

}  // extern "C"
