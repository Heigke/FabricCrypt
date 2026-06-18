/*  z2266_gpu_deep_explore.hip — Deep GPU Mechanism Exploration
 *
 *  NEW mechanisms not yet tested:
 *    1. Atomic operations (atomicAdd/CAS) — serialization creates non-deterministic ordering
 *    2. Wavefront scheduling jitter — multiple wavefronts per CU compete for resources
 *    3. Register file bank conflicts — VGPR same-bank reads cause data-dependent timing
 *
 *  THREE-POPULATION architecture:
 *    Pop A (tid 0-63):   Memory population — branch divergence + TLB-like temporal memory
 *    Pop B (tid 64-127): Nonlinear population — L1 bank conflicts + atomic operations
 *    Pop C (tid 128-191): Regression population — dense ESN recurrence with tanh gain
 *                         (from z2254j: SR=1.05, T=0.65 gave NARMA=0.842)
 *
 *  Compile: hipcc --offload-arch=gfx1100 -O1 -o scripts/z2266_gpu_deep_explore scripts/z2266_gpu_deep_explore.hip
 *  Run:     HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/z2266_gpu_deep_explore
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unistd.h>

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define N_CUS 12
#define STEPS 600
#define WASHOUT 100

// ═══════════════════════════════════════════════════════════════════════
// PHASE 1: Individual mechanism probes (64 threads each)
// ═══════════════════════════════════════════════════════════════════════
#define PROBE_THREADS 64
#define PROBE_NEURONS (N_CUS * PROBE_THREADS)
#define PROBE_SAMPLE 192

// ═══ MECHANISM 1: Atomic operations nonlinearity ═══
// atomicAdd on global memory creates serialization — non-deterministic ordering
// when multiple threads atomicAdd to same location, order depends on HW scheduler
__global__ void probe_atomic(
    float* state_out,
    volatile float* cu_xchg,
    float* atomic_buf,        // global buffer for atomic contention
    const float* input_seq,
    int steps
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= PROBE_THREADS) return;
    int nid = bid * PROBE_THREADS + tid;

    __shared__ float lds[PROBE_THREADS];

    float v = 0.0f, h = 0.0f, slow = 0.0f;
    uint32_t rng = nid * 2654435761u + 12345;
    rng *= 1103515245; float leak = 0.05f + 0.15f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float input_w = 0.05f + 0.20f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float thr = 0.4f + 0.5f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; int ln1 = (tid+1+((rng>>8)&7))%PROBE_THREADS;
    rng *= 1103515245; int ln2 = (tid+PROBE_THREADS-1-((rng>>8)&7))%PROBE_THREADS;
    rng *= 1103515245; float wl = 0.02f + 0.06f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; int rcu = ((rng>>8)&0xFF)%N_CUS;
    rng *= 1103515245; int rtid = ((rng>>8)&0xFF)%PROBE_THREADS;
    rng *= 1103515245; float wr = 0.01f + 0.03f * ((rng>>16)&0xFF)/255.0f;

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];

        // ATOMIC CONTENTION: multiple threads atomicAdd to overlapping locations
        // The serialization order is non-deterministic and input-dependent
        int n_atomic = 4 + (int)(8.0f * fabsf(u));  // 4-12 atomics
        int target_base = bid * 16;  // 16 contention slots per CU
        unsigned long long at_t0 = __clock64();
        for (int i = 0; i < n_atomic; i++) {
            int slot = target_base + ((tid + i * (int)(u * 7.0f + 8.0f)) & 0xF);
            atomicAdd(&atomic_buf[slot], v * 0.001f);
        }
        unsigned long long at_t1 = __clock64();
        // Clamp signal to [-2,2] to prevent tanh saturation
        float atomic_signal = fminf(fmaxf(((float)(at_t1 - at_t0) - 40.0f) / 30.0f, -2.0f), 2.0f);

        // Recurrence
        lds[tid] = v; __syncthreads();
        float lrec = wl * (lds[ln1] + lds[ln2]);
        cu_xchg[nid] = v; __threadfence();
        float rrec = wr * cu_xchg[rcu * PROBE_THREADS + rtid];

        // Cross-PLL (free baseline)
        uint64_t sc = __clock64(); uint64_t wc = wall_clock64();
        float pll = (float)((sc - wc) & 0x1FF) / 256.0f - 1.0f;

        v = tanhf(
            (1.0f - leak) * v
          + input_w * u
          + lrec + rrec
          + 0.005f * atomic_signal    // timing from contention (subtle)
          + 0.005f * pll
        );

        if (v > thr) { v -= thr; }
        h = 0.93f * h + 0.07f * v;
        slow = 0.99f * slow + 0.01f * v;
        state_out[nid * steps + t] = v + 0.3f*h + 0.1f*slow;
        __syncthreads();
    }
}

// ═══ MECHANISM 2: Wavefront scheduling jitter ═══
// Launch MORE wavefronts than CU can execute simultaneously
// The scheduler chooses which wavefront runs — non-deterministic
// We use 256 threads = 4 wavefronts per CU (gfx11 wavefront = 32 for wave32 mode)
// Actually RDNA3 uses wave32 by default, so 64 threads = 2 wavefronts
// With 128 threads = 4 wavefronts contending
#define WF_THREADS 128
#define WF_NEURONS (N_CUS * WF_THREADS)
__global__ void probe_wavefront_sched(
    float* state_out,
    volatile float* cu_xchg,
    const float* input_seq,
    int steps
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= WF_THREADS) return;
    int nid = bid * WF_THREADS + tid;

    __shared__ float lds[WF_THREADS];

    float v = 0.0f, h = 0.0f, slow = 0.0f;
    uint32_t rng = nid * 2654435761u + 77777;
    rng *= 1103515245; float leak = 0.05f + 0.15f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float input_w = 0.05f + 0.20f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float thr = 0.4f + 0.5f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; int ln1 = (tid+1+((rng>>8)&7))%WF_THREADS;
    rng *= 1103515245; int ln2 = (tid+WF_THREADS-1-((rng>>8)&7))%WF_THREADS;
    rng *= 1103515245; float wl = 0.02f + 0.06f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; int rcu = ((rng>>8)&0xFF)%N_CUS;
    rng *= 1103515245; int rtid = ((rng>>8)&0xFF)%WF_THREADS;
    rng *= 1103515245; float wr = 0.01f + 0.03f * ((rng>>16)&0xFF)/255.0f;

    float dummy = 1.0001f + nid * 0.0001f;

    // Initialize LDS to zero (prevent reading uninitialized memory)
    lds[tid] = 0.0f;
    __syncthreads();

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];

        // WAVEFRONT SCHEDULING PROBE:
        // Different wavefronts in the same CU do different amounts of work
        // The scheduler must interleave them — timing reveals scheduling decisions
        int my_wavefront = tid / 32;  // wave32 on RDNA3
        int work_amount = 10 + my_wavefront * 5 + (int)(10.0f * fabsf(u));

        unsigned long long wf_t0 = __clock64();
        for (int i = 0; i < work_amount; i++) {
            dummy = __fmaf_rn(dummy, 1.000001f, 0.0001f);
        }
        // Barrier forces all wavefronts to sync — reveals scheduling order
        __syncthreads();
        unsigned long long wf_t1 = __clock64();

        // The timing delta encodes how long THIS wavefront waited for others
        // Clamp to [-2,2] to prevent overwhelming ESN dynamics
        float wf_signal = fminf(fmaxf(((float)(wf_t1 - wf_t0) - 50.0f) / 30.0f, -2.0f), 2.0f);

        // Cross-wavefront state: read from different wavefront's neuron
        int other_wf = (my_wavefront + 1) % (WF_THREADS / 32);
        int other_tid = other_wf * 32 + (tid % 32);
        float wf_coupling = lds[other_tid];  // from previous step

        // Recurrence
        lds[tid] = v; __syncthreads();
        float lrec = wl * (lds[ln1] + lds[ln2]);
        cu_xchg[nid] = v; __threadfence();
        float rrec = wr * cu_xchg[rcu * WF_THREADS + rtid];

        uint64_t sc = __clock64(); uint64_t wc = wall_clock64();
        float pll = (float)((sc - wc) & 0x1FF) / 256.0f - 1.0f;

        v = tanhf(
            (1.0f - leak) * v
          + input_w * u
          + lrec + rrec
          + 0.005f * wf_signal         // scheduling timing (subtle)
          + 0.003f * wf_coupling       // cross-wavefront state
          + 0.005f * pll
        );

        if (v > thr) { v -= thr; }
        h = 0.93f * h + 0.07f * v;
        slow = 0.99f * slow + 0.01f * v;
        state_out[nid * steps + t] = v + 0.3f*h + 0.1f*slow;
        __syncthreads();
    }
    if (dummy == -9999.0f) state_out[0] = dummy;
}

// ═══ MECHANISM 3: Combined Atomic+Wavefront (best two mechanisms together) ═══
__global__ void probe_atomic_wf(
    float* state_out,
    volatile float* cu_xchg,
    float* atomic_buf,
    const float* input_seq,
    int steps
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= WF_THREADS) return;
    int nid = bid * WF_THREADS + tid;

    __shared__ float lds[WF_THREADS];
    lds[tid] = 0.0f;
    __syncthreads();

    float v = 0.0f, h = 0.0f, slow = 0.0f;
    uint32_t rng = nid * 2654435761u + 99999;
    rng *= 1103515245; float leak = 0.05f + 0.15f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float input_w = 0.05f + 0.20f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float thr = 0.4f + 0.5f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; int ln1 = (tid+1+((rng>>8)&7))%WF_THREADS;
    rng *= 1103515245; int ln2 = (tid+WF_THREADS-1-((rng>>8)&7))%WF_THREADS;
    rng *= 1103515245; float wl = 0.02f + 0.06f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; int rcu = ((rng>>8)&0xFF)%N_CUS;
    rng *= 1103515245; int rtid = ((rng>>8)&0xFF)%WF_THREADS;
    rng *= 1103515245; float wr = 0.01f + 0.03f * ((rng>>16)&0xFF)/255.0f;

    float dummy = 1.0001f + nid * 0.0001f;

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];

        // Wavefront scheduling jitter
        int my_wavefront = tid / 32;
        int work_amount = 10 + my_wavefront * 5 + (int)(10.0f * fabsf(u));
        unsigned long long wf_t0 = __clock64();
        for (int i = 0; i < work_amount; i++)
            dummy = __fmaf_rn(dummy, 1.000001f, 0.0001f);
        __syncthreads();
        unsigned long long wf_t1 = __clock64();
        float wf_signal = fminf(fmaxf(((float)(wf_t1 - wf_t0) - 50.0f) / 30.0f, -2.0f), 2.0f);

        // Atomic contention
        int target_base = bid * 16;
        int n_atomic = 3 + (int)(6.0f * fabsf(u));
        unsigned long long at_t0 = __clock64();
        for (int i = 0; i < n_atomic; i++) {
            int slot = target_base + ((tid + i * (int)(u * 7.0f + 8.0f)) & 0xF);
            atomicAdd(&atomic_buf[slot], v * 0.001f);
        }
        unsigned long long at_t1 = __clock64();
        float atomic_signal = fminf(fmaxf(((float)(at_t1 - at_t0) - 30.0f) / 25.0f, -2.0f), 2.0f);

        // Recurrence
        lds[tid] = v; __syncthreads();
        float lrec = wl * (lds[ln1] + lds[ln2]);
        cu_xchg[nid] = v; __threadfence();
        float rrec = wr * cu_xchg[rcu * WF_THREADS + rtid];

        uint64_t sc = __clock64(); uint64_t wc = wall_clock64();
        float pll = (float)((sc - wc) & 0x1FF) / 256.0f - 1.0f;

        v = tanhf(
            (1.0f - leak) * v
          + input_w * u
          + lrec + rrec
          + 0.005f * wf_signal
          + 0.004f * atomic_signal
          + 0.005f * pll
        );

        if (v > thr) { v -= thr; }
        h = 0.93f * h + 0.07f * v;
        slow = 0.99f * slow + 0.01f * v;
        state_out[nid * steps + t] = v + 0.3f*h + 0.1f*slow;
        __syncthreads();
    }
    if (dummy == -9999.0f) state_out[0] = dummy;
}

// ═══ CONTROL: no hardware mechanisms, pure ESN ═══
__global__ void probe_control(
    float* state_out,
    volatile float* cu_xchg,
    const float* input_seq,
    int steps,
    int n_threads
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= n_threads) return;
    int nid = bid * n_threads + tid;

    __shared__ float lds[192];  // max size

    float v = 0.0f, h = 0.0f, slow = 0.0f;
    uint32_t rng = nid * 2654435761u + 12345;
    rng *= 1103515245; float leak = 0.05f + 0.15f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float input_w = 0.05f + 0.20f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float thr_skip = ((rng>>16)&0xFF)/255.0f; (void)thr_skip;
    rng *= 1103515245; /* skip */ ;
    rng *= 1103515245; int ln1 = (tid+1+((rng>>8)&7))%n_threads;
    rng *= 1103515245; int ln2 = (tid+n_threads-1-((rng>>8)&7))%n_threads;
    rng *= 1103515245; float wl = 0.02f + 0.06f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; int rcu = ((rng>>8)&0xFF)%N_CUS;
    rng *= 1103515245; int rtid = ((rng>>8)&0xFF)%n_threads;
    rng *= 1103515245; float wr = 0.01f + 0.03f * ((rng>>16)&0xFF)/255.0f;

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];
        lds[tid] = v; __syncthreads();
        float lrec = wl * (lds[ln1] + lds[ln2]);
        cu_xchg[nid] = v; __threadfence();
        float rrec = wr * cu_xchg[rcu * n_threads + rtid];
        v = tanhf((1.0f-leak)*v + input_w*u + lrec + rrec);
        h = 0.93f*h + 0.07f*v;
        slow = 0.99f*slow + 0.01f*v;
        state_out[nid*steps+t] = v + 0.3f*h + 0.1f*slow;
        __syncthreads();
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PHASE 2: Three-Population Kernel
// ═══════════════════════════════════════════════════════════════════════
#define TRIPOP_THREADS 192    // 3 x 64
#define POP_SIZE 64
#define TRIPOP_NEURONS (N_CUS * TRIPOP_THREADS)
#define TRIPOP_SAMPLE 384     // 128 per pop

// LDS layout:
//   [0..63]   Pop A recurrence (memory — branch divergence)
//   [64..127] Pop B recurrence (nonlinear — L1 + atomic)
//   [128..191] Pop C recurrence (regression — dense ESN)
//   [192..255] Pop B scratch (L1 conflicts)
#define TRIPOP_LDS 256

__global__ void tripop_reservoir(
    float* state_out,
    volatile float* cu_xchg,
    float* atomic_buf,
    const float* input_seq,
    int steps,
    int mode   // 0=all, 1=no_A, 2=no_B, 3=no_C, 4=control
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= TRIPOP_THREADS) return;
    int nid = bid * TRIPOP_THREADS + tid;

    // Population assignment
    int pop = tid / POP_SIZE;        // 0=A, 1=B, 2=C
    int pop_tid = tid % POP_SIZE;

    __shared__ float lds[TRIPOP_LDS];

    float v = 0.0f, h = 0.0f, slow = 0.0f;
    uint32_t rng = nid * 2654435761u + 54321;
    rng *= 1103515245; float leak_base = 0.05f + 0.15f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float input_w = 0.05f + 0.20f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float thr = 0.4f + 0.5f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float bias = 0.02f * ((float)((rng>>16)&0xFF)/255.0f - 0.5f);

    // Local recurrence within population
    int lds_base = pop * POP_SIZE;
    rng *= 1103515245; int ln1 = lds_base + (pop_tid + 1 + ((rng>>8)&7)) % POP_SIZE;
    rng *= 1103515245; int ln2 = lds_base + (pop_tid + POP_SIZE - 1 - ((rng>>8)&7)) % POP_SIZE;
    rng *= 1103515245; float wl = 0.02f + 0.06f * ((rng>>16)&0xFF)/255.0f;

    // Cross-CU recurrence
    rng *= 1103515245; int rcu = ((rng>>8)&0xFF)%N_CUS;
    rng *= 1103515245; int rtid = ((rng>>8)&0xFF)%TRIPOP_THREADS;
    rng *= 1103515245; float wr = 0.01f + 0.03f * ((rng>>16)&0xFF)/255.0f;

    // Pop C specific: dense recurrence (4 additional neighbors for higher connectivity)
    int c_ln3 = lds_base + (pop_tid + 3 + ((rng>>8)&15)) % POP_SIZE;
    rng *= 1103515245;
    int c_ln4 = lds_base + (pop_tid + POP_SIZE - 3 - ((rng>>8)&15)) % POP_SIZE;
    rng *= 1103515245;
    int c_ln5 = lds_base + (pop_tid + 7 + ((rng>>8)&15)) % POP_SIZE;
    rng *= 1103515245;
    int c_ln6 = lds_base + (pop_tid + POP_SIZE - 7 - ((rng>>8)&15)) % POP_SIZE;
    rng *= 1103515245;
    float wl2 = 0.01f + 0.04f * ((rng>>16)&0xFF)/255.0f;

    // Cross-population coupling (weak — lets populations communicate)
    rng *= 1103515245;
    int xpop_tid1 = ((pop + 1) % 3) * POP_SIZE + (pop_tid + ((rng>>8)&0x1F)) % POP_SIZE;
    rng *= 1103515245;
    int xpop_tid2 = ((pop + 2) % 3) * POP_SIZE + (pop_tid + ((rng>>8)&0x1F)) % POP_SIZE;

    // Branch threshold (Pop A specific but needed by all paths)
    float bthr = 0.5f + 0.3f * ((nid & 0xF) / 15.0f);

    float dummy = 1.0001f + nid * 0.0001f;

    // Initialize LDS to zero
    lds[tid] = 0.0f;
    if (tid < TRIPOP_LDS - TRIPOP_THREADS) lds[TRIPOP_THREADS + tid] = 0.0f;
    __syncthreads();

    // Pop C: tuned ESN parameters from z2254j
    // SR=1.05 (spectral radius > 1 for richer dynamics)
    // T=0.65 (tanh temperature for softer nonlinearity)
    float c_leak = (pop == 2) ? 0.08f : leak_base;  // Pop C: slower leak for longer memory
    float c_sr = 1.05f;    // spectral radius scaling
    float c_temp = 0.65f;  // tanh temperature (v -> tanh(v/T))

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];
        float hw_signal = 0.0f;

        bool active_A = (mode != 1 && mode != 4);
        bool active_B = (mode != 2 && mode != 4);
        bool active_C = (mode != 3 && mode != 4);

        // ═══ Pop A: Branch divergence (memory-preserving) ═══
        if (pop == 0 && active_A) {
            float branch_val = 0.0f;
            if (v > bthr) {
                for (int i = 0; i < 20; i++)
                    dummy = __fmaf_rn(dummy, 1.000001f, 0.0001f);
                branch_val = 1.0f;
            } else {
                uint32_t iv = __float_as_uint(dummy);
                iv = iv * 2654435761u + 1;
                dummy = __uint_as_float((iv & 0x007FFFFF) | 0x3F800000);
                branch_val = -1.0f;
            }
            hw_signal = 0.02f * branch_val;
        }

        // ═══ Pop B: L1 bank conflicts + Atomic operations ═══
        // NOTE: NO __syncthreads() inside conditional — that's UB!
        // L1 timing measured from scratch writes, atomic from global contention
        float l1_signal = 0.0f, atomic_signal = 0.0f;
        if (pop == 1 && active_B) {
            // L1 bank conflicts in scratch region [192..255]
            float l1_val = 1.0f + nid * 0.001f;
            unsigned long long l1_t0 = __clock64();
            int l1_iters = 5 + (int)(10.0f * fabsf(u));
            int scratch_base = 192;
            for (int i = 0; i < l1_iters; i++)
                lds[scratch_base + (pop_tid + i * ((int)(u * 17.0f) & 0xF)) % POP_SIZE] += l1_val * 0.001f;
            unsigned long long l1_t1 = __clock64();
            l1_signal = fminf(fmaxf(((float)(l1_t1 - l1_t0) - 30.0f) / 20.0f, -2.0f), 2.0f);

            // Atomic contention (global memory — no LDS corruption)
            int target_base = bid * 16;
            int n_atomic = 3 + (int)(6.0f * fabsf(u));
            unsigned long long at_t0 = __clock64();
            for (int i = 0; i < n_atomic; i++) {
                int slot = target_base + ((pop_tid + i * (int)(u * 7.0f + 8.0f)) & 0xF);
                atomicAdd(&atomic_buf[slot], v * 0.001f);
            }
            unsigned long long at_t1 = __clock64();
            atomic_signal = fminf(fmaxf(((float)(at_t1 - at_t0) - 30.0f) / 25.0f, -2.0f), 2.0f);

            hw_signal = 0.005f * l1_signal + 0.005f * atomic_signal;
        }

        // ALL threads sync here before recurrence reads
        __syncthreads();

        // ═══ RECURRENCE ═══
        int my_lds_slot = pop * POP_SIZE + pop_tid;
        lds[my_lds_slot] = v;
        __syncthreads();

        // Local recurrence (within population)
        float lrec = wl * (lds[ln1] + lds[ln2]);

        // Pop C: additional dense connections (higher spectral radius)
        float dense_rec = 0.0f;
        if (pop == 2 && active_C) {
            dense_rec = wl2 * c_sr * (lds[c_ln3] + lds[c_ln4] + lds[c_ln5] + lds[c_ln6]);
        }

        // Cross-population coupling (weak)
        float xpop = 0.005f * (lds[xpop_tid1] + lds[xpop_tid2]);

        // Cross-CU recurrence
        cu_xchg[nid] = v; __threadfence();
        float rrec = wr * cu_xchg[rcu * TRIPOP_THREADS + rtid];

        // Cross-PLL (free)
        uint64_t sc = __clock64(); uint64_t wc = wall_clock64();
        float pll = (float)((sc - wc) & 0x1FF) / 256.0f - 1.0f;

        // ═══ STATE UPDATE ═══
        float pre_act = (1.0f - c_leak) * v
                      + input_w * u
                      + lrec + rrec + dense_rec
                      + xpop
                      + bias
                      + hw_signal
                      + 0.005f * pll;

        // Pop C uses temperature-scaled tanh for softer nonlinearity
        if (pop == 2 && active_C) {
            v = tanhf(pre_act / c_temp);
        } else {
            v = tanhf(pre_act);
        }

        // Spike-reset for Pop A and B only; Pop C uses pure continuous dynamics
        if (pop != 2 && v > thr) { v -= thr; }

        h = 0.93f * h + 0.07f * v;
        slow = 0.99f * slow + 0.01f * v;
        state_out[nid * steps + t] = v + 0.3f*h + 0.1f*slow;
        __syncthreads();
    }
    if (dummy == -9999.0f) state_out[0] = dummy;
}


// ═══════════════════════════════════════════════════════════════════════
// HOST UTILITIES
// ═══════════════════════════════════════════════════════════════════════
static uint32_t host_rng = 42;
float randf() { host_rng = host_rng * 1103515245 + 12345; return (float)((host_rng >> 16) & 0x7FFF) / 32767.0f; }
float randf_signed() { return randf() * 2.0f - 1.0f; }

void extract_state(float* state, int n_neurons, int steps,
                   int washout, int sample_n,
                   std::vector<std::vector<float>>& X) {
    int stride = n_neurons / sample_n;
    if (stride < 1) stride = 1;
    for (int t = washout; t < steps; t++) {
        std::vector<float> row;
        for (int si = 0; si < sample_n && si*stride < n_neurons; si++) {
            int i = si * stride;
            float v = state[i * steps + t];
            if (v != v || v > 1e20 || v < -1e20) v = 0;
            row.push_back(v);
        }
        X.push_back(row);
    }
}

// Balanced extraction for tripop: equal samples from each population
void extract_tripop(float* state, int steps, int washout, int sample_n,
                    std::vector<std::vector<float>>& X) {
    int per_pop = sample_n / 3;
    int stride_pop = (N_CUS * POP_SIZE) / per_pop;
    if (stride_pop < 1) stride_pop = 1;

    for (int t = washout; t < steps; t++) {
        std::vector<float> row;
        for (int p = 0; p < 3; p++) {
            for (int si = 0; si < per_pop; si++) {
                int pop_idx = si * stride_pop;
                if (pop_idx >= N_CUS * POP_SIZE) break;
                int cu = pop_idx / POP_SIZE;
                int ptid = pop_idx % POP_SIZE;
                int nid = cu * TRIPOP_THREADS + p * POP_SIZE + ptid;
                float v = state[nid * steps + t];
                if (v != v || v > 1e20 || v < -1e20) v = 0;
                row.push_back(v);
            }
        }
        X.push_back(row);
    }
}

// Extract single population from tripop
void extract_single_pop(float* state, int steps, int washout, int sample_n,
                         int pop_idx, std::vector<std::vector<float>>& X) {
    int stride = (N_CUS * POP_SIZE) / sample_n;
    if (stride < 1) stride = 1;
    for (int t = washout; t < steps; t++) {
        std::vector<float> row;
        for (int si = 0; si < sample_n; si++) {
            int pidx = si * stride;
            if (pidx >= N_CUS * POP_SIZE) break;
            int cu = pidx / POP_SIZE;
            int ptid = pidx % POP_SIZE;
            int nid = cu * TRIPOP_THREADS + pop_idx * POP_SIZE + ptid;
            float v = state[nid * steps + t];
            if (v != v || v > 1e20 || v < -1e20) v = 0;
            row.push_back(v);
        }
        X.push_back(row);
    }
}

struct RegResult { float nrmse; float r2; };
RegResult ridge_regress(std::vector<std::vector<float>>& X_train,
    std::vector<float>& y_train, std::vector<std::vector<float>>& X_test,
    std::vector<float>& y_test, float lambda) {
    int n=X_train.size(),d=X_train[0].size();
    std::vector<double> XtX(d*d,0);
    for(int i=0;i<n;i++) for(int j=0;j<d;j++) for(int k=0;k<=j;k++)
        XtX[j*d+k]+=X_train[i][j]*X_train[i][k];
    for(int j=0;j<d;j++){for(int k=0;k<j;k++) XtX[k*d+j]=XtX[j*d+k]; XtX[j*d+j]+=lambda;}
    std::vector<double> Xty(d,0);
    for(int i=0;i<n;i++) for(int j=0;j<d;j++) Xty[j]+=X_train[i][j]*y_train[i];
    std::vector<double> L(d*d,0);
    for(int i=0;i<d;i++) for(int j=0;j<=i;j++){
        double s=XtX[i*d+j]; for(int k=0;k<j;k++) s-=L[i*d+k]*L[j*d+k];
        L[i*d+j]=(i==j)?sqrt(s>0?s:1e-10):s/(L[j*d+j]>0?L[j*d+j]:1e-10);
    }
    std::vector<double> w(d),z(d);
    for(int i=0;i<d;i++){double s=Xty[i]; for(int k=0;k<i;k++) s-=L[i*d+k]*z[k]; z[i]=s/(L[i*d+i]>0?L[i*d+i]:1e-10);}
    for(int i=d-1;i>=0;i--){double s=z[i]; for(int k=i+1;k<d;k++) s-=L[k*d+i]*w[k]; w[i]=s/(L[i*d+i]>0?L[i*d+i]:1e-10);}
    int nt=X_test.size(); float ym=0,sr=0,st=0;
    for(int i=0;i<nt;i++) ym+=y_test[i]; ym/=nt;
    for(int i=0;i<nt;i++){float p=0; for(int j=0;j<d;j++) p+=X_test[i][j]*w[j];
        sr+=(y_test[i]-p)*(y_test[i]-p); st+=(y_test[i]-ym)*(y_test[i]-ym);}
    RegResult res; res.nrmse=sqrtf(sr/nt)/(sqrtf(st/nt)>1e-8?sqrtf(st/nt):1e-8);
    res.r2=1.0f-sr/(st>1e-8?st:1e-8); return res;
}

float ridge_classify_binary(std::vector<std::vector<float>>& X_train,
    std::vector<int>& y_train, std::vector<std::vector<float>>& X_test,
    std::vector<int>& y_test, float lambda) {
    std::vector<float> yt(y_train.size()),ytest(y_test.size());
    for(size_t i=0;i<y_train.size();i++) yt[i]=y_train[i]?1.0f:-1.0f;
    for(size_t i=0;i<y_test.size();i++) ytest[i]=y_test[i]?1.0f:-1.0f;
    int n_tr=X_train.size(),d=X_train[0].size();
    std::vector<double> XtX(d*d,0);
    for(int i=0;i<n_tr;i++) for(int j=0;j<d;j++) for(int k=0;k<=j;k++)
        XtX[j*d+k]+=X_train[i][j]*X_train[i][k];
    for(int j=0;j<d;j++){for(int k=0;k<j;k++) XtX[k*d+j]=XtX[j*d+k]; XtX[j*d+j]+=lambda;}
    std::vector<double> Xty(d,0);
    for(int i=0;i<n_tr;i++) for(int j=0;j<d;j++) Xty[j]+=X_train[i][j]*yt[i];
    std::vector<double> L(d*d,0);
    for(int i=0;i<d;i++) for(int j=0;j<=i;j++){
        double s=XtX[i*d+j]; for(int k=0;k<j;k++) s-=L[i*d+k]*L[j*d+k];
        L[i*d+j]=(i==j)?sqrt(s>0?s:1e-10):s/(L[j*d+j]>0?L[j*d+j]:1e-10);
    }
    std::vector<double> w(d),z(d);
    for(int i=0;i<d;i++){double s=Xty[i]; for(int k=0;k<i;k++) s-=L[i*d+k]*z[k]; z[i]=s/(L[i*d+i]>0?L[i*d+i]:1e-10);}
    for(int i=d-1;i>=0;i--){double s=z[i]; for(int k=i+1;k<d;k++) s-=L[k*d+i]*w[k]; w[i]=s/(L[i*d+i]>0?L[i*d+i]:1e-10);}
    int n=X_test.size(),correct=0;
    for(int i=0;i<n;i++){float p=0; for(int j=0;j<d;j++) p+=X_test[i][j]*w[j];
        if((p>0?1:0)==y_test[i]) correct++;}
    return (float)correct/n;
}

float ridge_classify_multi(std::vector<std::vector<float>>& X_train,
    std::vector<int>& y_train, std::vector<std::vector<float>>& X_test,
    std::vector<int>& y_test, int n_classes, float lambda) {
    int d=X_train[0].size(),nt=X_test.size();
    std::vector<std::vector<double>> all_w(n_classes,std::vector<double>(d,0));
    for(int c=0;c<n_classes;c++){
        std::vector<float> yt(y_train.size());
        for(size_t i=0;i<y_train.size();i++) yt[i]=(y_train[i]==c)?1.0f:-1.0f;
        int n=X_train.size();
        std::vector<double> XtX(d*d,0);
        for(int i=0;i<n;i++) for(int j=0;j<d;j++) for(int k=0;k<=j;k++)
            XtX[j*d+k]+=X_train[i][j]*X_train[i][k];
        for(int j=0;j<d;j++){for(int k=0;k<j;k++) XtX[k*d+j]=XtX[j*d+k]; XtX[j*d+j]+=lambda;}
        std::vector<double> Xty(d,0);
        for(int i=0;i<n;i++) for(int j=0;j<d;j++) Xty[j]+=X_train[i][j]*yt[i];
        std::vector<double> L(d*d,0);
        for(int i=0;i<d;i++) for(int j=0;j<=i;j++){
            double s=XtX[i*d+j]; for(int k=0;k<j;k++) s-=L[i*d+k]*L[j*d+k];
            L[i*d+j]=(i==j)?sqrt(s>0?s:1e-10):s/(L[j*d+j]>0?L[j*d+j]:1e-10);
        }
        std::vector<double> z(d);
        for(int i=0;i<d;i++){double s=Xty[i]; for(int k=0;k<i;k++) s-=L[i*d+k]*z[k]; z[i]=s/(L[i*d+i]>0?L[i*d+i]:1e-10);}
        for(int i=d-1;i>=0;i--){double s=z[i]; for(int k=i+1;k<d;k++) s-=L[k*d+i]*all_w[c][k]; all_w[c][i]=s/(L[i*d+i]>0?L[i*d+i]:1e-10);}
    }
    int correct=0;
    for(int i=0;i<nt;i++){float best=-1e20;int pred=0;
        for(int c=0;c<n_classes;c++){float p=0; for(int j=0;j<d;j++) p+=X_test[i][j]*all_w[c][j];
            if(p>best){best=p;pred=c;}}
        if(pred==y_test[i]) correct++;}
    return (float)correct/nt;
}

// ═══════════════════════════════════════════════════════════════════════
// BENCHMARK SUITE
// ═══════════════════════════════════════════════════════════════════════
struct BenchResult {
    float xor_acc[5];   // XOR tau=1..5
    float mc;           // Memory capacity d=1..10
    float narma_r2;     // NARMA-5 R²
    float wave4;        // 4-class waveform
    float wave8;        // 8-class waveform
};

void gen_xor_input(float* buf, int steps) {
    host_rng = 42;
    for (int t = 0; t < steps; t++) buf[t] = (randf() > 0.5f) ? 1.0f : -1.0f;
}

void gen_mc_input(float* buf, int steps) {
    host_rng = 123;
    for (int t = 0; t < steps; t++) buf[t] = randf_signed();
}

void gen_narma_input(float* buf, float* narma, int steps) {
    host_rng = 456;
    for (int t = 0; t < steps; t++) buf[t] = randf() * 0.5f;
    memset(narma, 0, steps * sizeof(float));
    for (int t = 5; t < steps-1; t++) {
        float ys = 0; for (int k = 0; k < 5; k++) ys += narma[t-k];
        narma[t+1] = 0.3f*narma[t] + 0.05f*narma[t]*ys + 1.5f*buf[t-5]*buf[t] + 0.1f;
        if (narma[t+1] > 10) narma[t+1] = 10; if (narma[t+1] < -10) narma[t+1] = -10;
    }
}

void gen_wave_input(float* buf, int* labels, int steps, int n_classes) {
    int period = 20;
    for (int t = 0; t < steps; t++) {
        int cls = (t / period) % n_classes; labels[t] = cls;
        float phase = (float)(t % period) / period;
        switch(cls % 8) {
            case 0: buf[t] = sinf(2*M_PI*phase); break;
            case 1: buf[t] = phase < 0.5f ? 1.0f : -1.0f; break;
            case 2: buf[t] = 4.0f*fabsf(phase-0.5f) - 1.0f; break;
            case 3: buf[t] = 2.0f*phase - 1.0f; break;
            case 4: buf[t] = sinf(4*M_PI*phase) * 0.8f; break;
            case 5: buf[t] = (phase < 0.25f || phase > 0.75f) ? 1.0f : -1.0f; break;
            case 6: buf[t] = cosf(2*M_PI*phase); break;
            case 7: buf[t] = sinf(2*M_PI*phase) * cosf(4*M_PI*phase); break;
        }
    }
}

BenchResult run_bench(std::vector<std::vector<float>>& X, float* h_input, float lambda) {
    BenchResult res;
    int n_eff = X.size();
    int split = n_eff * 2 / 3;

    // XOR tau=1..5
    for (int ti = 0; ti < 5; ti++) {
        int tau = ti + 1;
        std::vector<int> y_xor;
        for (int i = 0; i < n_eff; i++) {
            int t = i + WASHOUT;
            y_xor.push_back(t >= tau ? ((h_input[t]>0) != (h_input[t-tau]>0) ? 1 : 0) : 0);
        }
        std::vector<std::vector<float>> Xtr(X.begin(), X.begin()+split);
        std::vector<std::vector<float>> Xte(X.begin()+split, X.end());
        std::vector<int> ytr(y_xor.begin(), y_xor.begin()+split);
        std::vector<int> yte(y_xor.begin()+split, y_xor.end());
        res.xor_acc[ti] = ridge_classify_binary(Xtr, ytr, Xte, yte, lambda);
    }
    return res;
}

float compute_mc(std::vector<std::vector<float>>& X, float* h_input, float lambda) {
    int n_eff = X.size(), split = n_eff * 2 / 3;
    float mc = 0;
    for (int d = 1; d <= 10; d++) {
        std::vector<float> ytr, yte;
        for (int i = 0; i < split; i++) { int t = i + WASHOUT; ytr.push_back(t >= d ? h_input[t-d] : 0); }
        for (int i = split; i < n_eff; i++) { int t = i + WASHOUT; yte.push_back(t >= d ? h_input[t-d] : 0); }
        std::vector<std::vector<float>> Xtr(X.begin(), X.begin()+split);
        std::vector<std::vector<float>> Xte(X.begin()+split, X.end());
        auto rr = ridge_regress(Xtr, ytr, Xte, yte, lambda);
        mc += (rr.r2 > 0 ? rr.r2 : 0);
    }
    return mc;
}

float compute_narma(std::vector<std::vector<float>>& X, float* narma, float lambda) {
    int n_eff = X.size(), split = n_eff * 2 / 3;
    std::vector<float> nytr, nyte;
    for (int i = 0; i < split; i++) nytr.push_back(narma[i+WASHOUT]);
    for (int i = split; i < n_eff; i++) nyte.push_back(narma[i+WASHOUT]);
    std::vector<std::vector<float>> Xtr(X.begin(), X.begin()+split);
    std::vector<std::vector<float>> Xte(X.begin()+split, X.end());
    auto r = ridge_regress(Xtr, nytr, Xte, nyte, lambda);
    return r.r2;
}

float compute_wave(std::vector<std::vector<float>>& X, int* labels, int n_classes, float lambda) {
    int n_eff = X.size(), split = n_eff * 2 / 3;
    std::vector<std::vector<float>> Xtr(X.begin(), X.begin()+split);
    std::vector<std::vector<float>> Xte(X.begin()+split, X.end());
    std::vector<int> ytr(labels+WASHOUT, labels+WASHOUT+split);
    std::vector<int> yte(labels+WASHOUT+split, labels+WASHOUT+n_eff);
    return ridge_classify_multi(Xtr, ytr, Xte, yte, n_classes, lambda);
}

// Temperature check — returns edge temp in C, -1 if can't read
float check_temp() {
    FILE* f = popen("sensors 2>/dev/null | grep -i 'edge' | head -1 | grep -oP '[0-9]+\\.[0-9]+'", "r");
    if (!f) return -1;
    char buf[64];
    float temp = -1;
    if (fgets(buf, sizeof(buf), f)) temp = atof(buf);
    pclose(f);
    return temp;
}

void thermal_guard(const char* phase) {
    float temp = check_temp();
    if (temp > 0) {
        printf("  [thermal] %s: edge=%.1f°C", phase, temp);
        if (temp > 75.0f) {
            printf(" — COOLING DOWN (10s)...");
            fflush(stdout);
            sleep(10);
            temp = check_temp();
            printf(" now %.1f°C", temp);
        }
        printf("\n");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════
int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  z2266: Deep GPU Mechanism Exploration                    ║\n");
    printf("║  Atomic ops, wavefront scheduling, register bank conflicts ║\n");
    printf("║  Three-population architecture for NARMA regression       ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("GPU: %s, CUs: %d\n\n", prop.name, prop.multiProcessorCount);

    float lambda = 1.0f;
    float* h_input = (float*)malloc(STEPS * sizeof(float));
    float* h_narma = (float*)malloc(STEPS * sizeof(float));
    int* h_labels = (int*)malloc(STEPS * sizeof(int));

    thermal_guard("startup");

    // ═══════════════════════════════════════════════════════════════
    // PHASE 1: Individual mechanism probes
    // ═══════════════════════════════════════════════════════════════
    printf("═══ PHASE 1: Individual Mechanism Probes ═══\n\n");

    // Allocate probe buffers
    int max_neurons = WF_NEURONS > PROBE_NEURONS ? WF_NEURONS : PROBE_NEURONS;
    if (TRIPOP_NEURONS > max_neurons) max_neurons = TRIPOP_NEURONS;
    float *d_state, *d_input, *d_atomic;
    volatile float *d_xchg;
    HIP_CHECK(hipMalloc(&d_state, max_neurons * STEPS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_input, STEPS * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_xchg, max_neurons * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_atomic, N_CUS * 16 * sizeof(float)));  // atomic contention buffer

    float* h_state = (float*)malloc(max_neurons * STEPS * sizeof(float));

    // Test structures
    struct ProbeResult {
        const char* name;
        float xor1, xor2, xor3, xor4, xor5;
        float mc;
        float narma;
        float wave4, wave8;
    };
    std::vector<ProbeResult> probe_results;

    auto run_probe_bench = [&](const char* name, int n_neurons, int sample_n) -> ProbeResult {
        ProbeResult pr;
        pr.name = name;

        // XOR
        gen_xor_input(h_input, STEPS);
        HIP_CHECK(hipMemcpy(d_input, h_input, STEPS*sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(h_state, d_state, n_neurons*STEPS*sizeof(float), hipMemcpyDeviceToHost));
        std::vector<std::vector<float>> X;
        extract_state(h_state, n_neurons, STEPS, WASHOUT, sample_n, X);
        auto br = run_bench(X, h_input, lambda);
        pr.xor1 = br.xor_acc[0]; pr.xor2 = br.xor_acc[1]; pr.xor3 = br.xor_acc[2];
        pr.xor4 = br.xor_acc[3]; pr.xor5 = br.xor_acc[4];

        // MC
        gen_mc_input(h_input, STEPS);
        HIP_CHECK(hipMemcpy(d_input, h_input, STEPS*sizeof(float), hipMemcpyHostToDevice));
        pr.mc = compute_mc(X, h_input, lambda);

        // NARMA
        gen_narma_input(h_input, h_narma, STEPS);
        HIP_CHECK(hipMemcpy(d_input, h_input, STEPS*sizeof(float), hipMemcpyHostToDevice));
        pr.narma = compute_narma(X, h_narma, lambda);

        // Wave 4
        gen_wave_input(h_input, h_labels, STEPS, 4);
        HIP_CHECK(hipMemcpy(d_input, h_input, STEPS*sizeof(float), hipMemcpyHostToDevice));
        pr.wave4 = compute_wave(X, h_labels, 4, lambda);

        // Wave 8
        gen_wave_input(h_input, h_labels, STEPS, 8);
        HIP_CHECK(hipMemcpy(d_input, h_input, STEPS*sizeof(float), hipMemcpyHostToDevice));
        pr.wave8 = compute_wave(X, h_labels, 8, lambda);

        return pr;
    };

    // --- CONTROL (64 threads) ---
    printf("Running CONTROL (pure ESN, 64 threads)...\n");
    gen_xor_input(h_input, STEPS);
    HIP_CHECK(hipMemcpy(d_input, h_input, STEPS*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_state, 0, PROBE_NEURONS*STEPS*sizeof(float)));
    HIP_CHECK(hipMemset((void*)d_xchg, 0, max_neurons*sizeof(float)));
    probe_control<<<N_CUS, PROBE_THREADS>>>(d_state, d_xchg, d_input, STEPS, PROBE_THREADS);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_state, d_state, PROBE_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost));

    // Now run all benchmarks reusing the state
    {
        std::vector<std::vector<float>> X;
        extract_state(h_state, PROBE_NEURONS, STEPS, WASHOUT, PROBE_SAMPLE, X);
        auto br = run_bench(X, h_input, lambda);

        gen_mc_input(h_input, STEPS);
        HIP_CHECK(hipMemcpy(d_input, h_input, STEPS*sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_state, 0, PROBE_NEURONS*STEPS*sizeof(float)));
        HIP_CHECK(hipMemset((void*)d_xchg, 0, max_neurons*sizeof(float)));
        probe_control<<<N_CUS, PROBE_THREADS>>>(d_state, d_xchg, d_input, STEPS, PROBE_THREADS);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_state, d_state, PROBE_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost));
        X.clear();
        extract_state(h_state, PROBE_NEURONS, STEPS, WASHOUT, PROBE_SAMPLE, X);
        float mc = compute_mc(X, h_input, lambda);

        gen_narma_input(h_input, h_narma, STEPS);
        HIP_CHECK(hipMemcpy(d_input, h_input, STEPS*sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_state, 0, PROBE_NEURONS*STEPS*sizeof(float)));
        HIP_CHECK(hipMemset((void*)d_xchg, 0, max_neurons*sizeof(float)));
        probe_control<<<N_CUS, PROBE_THREADS>>>(d_state, d_xchg, d_input, STEPS, PROBE_THREADS);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_state, d_state, PROBE_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost));
        X.clear();
        extract_state(h_state, PROBE_NEURONS, STEPS, WASHOUT, PROBE_SAMPLE, X);
        float narma = compute_narma(X, h_narma, lambda);

        gen_wave_input(h_input, h_labels, STEPS, 4);
        HIP_CHECK(hipMemcpy(d_input, h_input, STEPS*sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_state, 0, PROBE_NEURONS*STEPS*sizeof(float)));
        HIP_CHECK(hipMemset((void*)d_xchg, 0, max_neurons*sizeof(float)));
        probe_control<<<N_CUS, PROBE_THREADS>>>(d_state, d_xchg, d_input, STEPS, PROBE_THREADS);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_state, d_state, PROBE_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost));
        X.clear();
        extract_state(h_state, PROBE_NEURONS, STEPS, WASHOUT, PROBE_SAMPLE, X);
        float w4 = compute_wave(X, h_labels, 4, lambda);

        gen_wave_input(h_input, h_labels, STEPS, 8);
        HIP_CHECK(hipMemcpy(d_input, h_input, STEPS*sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_state, 0, PROBE_NEURONS*STEPS*sizeof(float)));
        HIP_CHECK(hipMemset((void*)d_xchg, 0, max_neurons*sizeof(float)));
        probe_control<<<N_CUS, PROBE_THREADS>>>(d_state, d_xchg, d_input, STEPS, PROBE_THREADS);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_state, d_state, PROBE_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost));
        X.clear();
        extract_state(h_state, PROBE_NEURONS, STEPS, WASHOUT, PROBE_SAMPLE, X);
        float w8 = compute_wave(X, h_labels, 8, lambda);

        ProbeResult pr = {"CONTROL", br.xor_acc[0], br.xor_acc[1], br.xor_acc[2], br.xor_acc[3], br.xor_acc[4], mc, narma, w4, w8};
        probe_results.push_back(pr);
    }
    thermal_guard("post-control");

    // Helper: run a single benchmark pass given pre-run state
    auto run_one_bench = [&](const char* name, auto launch_fn, int n_neurons, int sample_n) -> ProbeResult {
        printf("Running %s...\n", name);
        ProbeResult pr; pr.name = name;

        // XOR
        gen_xor_input(h_input, STEPS);
        HIP_CHECK(hipMemcpy(d_input, h_input, STEPS*sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_state, 0, n_neurons*STEPS*sizeof(float)));
        HIP_CHECK(hipMemset((void*)d_xchg, 0, max_neurons*sizeof(float)));
        HIP_CHECK(hipMemset(d_atomic, 0, N_CUS*16*sizeof(float)));
        launch_fn();
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_state, d_state, n_neurons*STEPS*sizeof(float), hipMemcpyDeviceToHost));
        std::vector<std::vector<float>> X;
        extract_state(h_state, n_neurons, STEPS, WASHOUT, sample_n, X);
        auto br = run_bench(X, h_input, lambda);
        pr.xor1=br.xor_acc[0]; pr.xor2=br.xor_acc[1]; pr.xor3=br.xor_acc[2]; pr.xor4=br.xor_acc[3]; pr.xor5=br.xor_acc[4];

        // MC
        gen_mc_input(h_input, STEPS);
        HIP_CHECK(hipMemcpy(d_input, h_input, STEPS*sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_state, 0, n_neurons*STEPS*sizeof(float)));
        HIP_CHECK(hipMemset((void*)d_xchg, 0, max_neurons*sizeof(float)));
        HIP_CHECK(hipMemset(d_atomic, 0, N_CUS*16*sizeof(float)));
        launch_fn();
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_state, d_state, n_neurons*STEPS*sizeof(float), hipMemcpyDeviceToHost));
        X.clear();
        extract_state(h_state, n_neurons, STEPS, WASHOUT, sample_n, X);
        pr.mc = compute_mc(X, h_input, lambda);

        // NARMA
        gen_narma_input(h_input, h_narma, STEPS);
        HIP_CHECK(hipMemcpy(d_input, h_input, STEPS*sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_state, 0, n_neurons*STEPS*sizeof(float)));
        HIP_CHECK(hipMemset((void*)d_xchg, 0, max_neurons*sizeof(float)));
        HIP_CHECK(hipMemset(d_atomic, 0, N_CUS*16*sizeof(float)));
        launch_fn();
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_state, d_state, n_neurons*STEPS*sizeof(float), hipMemcpyDeviceToHost));
        X.clear();
        extract_state(h_state, n_neurons, STEPS, WASHOUT, sample_n, X);
        pr.narma = compute_narma(X, h_narma, lambda);

        // Wave4
        gen_wave_input(h_input, h_labels, STEPS, 4);
        HIP_CHECK(hipMemcpy(d_input, h_input, STEPS*sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_state, 0, n_neurons*STEPS*sizeof(float)));
        HIP_CHECK(hipMemset((void*)d_xchg, 0, max_neurons*sizeof(float)));
        HIP_CHECK(hipMemset(d_atomic, 0, N_CUS*16*sizeof(float)));
        launch_fn();
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_state, d_state, n_neurons*STEPS*sizeof(float), hipMemcpyDeviceToHost));
        X.clear();
        extract_state(h_state, n_neurons, STEPS, WASHOUT, sample_n, X);
        pr.wave4 = compute_wave(X, h_labels, 4, lambda);

        // Wave8
        gen_wave_input(h_input, h_labels, STEPS, 8);
        HIP_CHECK(hipMemcpy(d_input, h_input, STEPS*sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_state, 0, n_neurons*STEPS*sizeof(float)));
        HIP_CHECK(hipMemset((void*)d_xchg, 0, max_neurons*sizeof(float)));
        HIP_CHECK(hipMemset(d_atomic, 0, N_CUS*16*sizeof(float)));
        launch_fn();
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_state, d_state, n_neurons*STEPS*sizeof(float), hipMemcpyDeviceToHost));
        X.clear();
        extract_state(h_state, n_neurons, STEPS, WASHOUT, sample_n, X);
        pr.wave8 = compute_wave(X, h_labels, 8, lambda);

        thermal_guard(name);
        return pr;
    };

    // --- ATOMIC PROBE ---
    probe_results.push_back(run_one_bench("ATOMIC", [&]() {
        probe_atomic<<<N_CUS, PROBE_THREADS>>>(d_state, d_xchg, d_atomic, d_input, STEPS);
    }, PROBE_NEURONS, PROBE_SAMPLE));

    // --- WAVEFRONT SCHEDULING PROBE ---
    probe_results.push_back(run_one_bench("WAVEFRONT_SCHED", [&]() {
        probe_wavefront_sched<<<N_CUS, WF_THREADS>>>(d_state, d_xchg, d_input, STEPS);
    }, WF_NEURONS, PROBE_SAMPLE));

    // --- COMBINED ATOMIC+WAVEFRONT PROBE ---
    probe_results.push_back(run_one_bench("ATOMIC+WF", [&]() {
        probe_atomic_wf<<<N_CUS, WF_THREADS>>>(d_state, d_xchg, d_atomic, d_input, STEPS);
    }, WF_NEURONS, PROBE_SAMPLE));

    // Print Phase 1 results
    printf("\n%-20s %6s %6s %6s %6s %6s %6s %7s %6s %6s\n",
           "Mechanism", "XOR1", "XOR2", "XOR3", "XOR4", "XOR5", "MC", "NARMA", "Wave4", "Wave8");
    printf("──────────────────────────────────────────────────────────────────────────────────────────\n");
    for (auto& pr : probe_results) {
        printf("%-20s %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.3f %7.3f %5.1f%% %5.1f%%\n",
               pr.name, pr.xor1*100, pr.xor2*100, pr.xor3*100, pr.xor4*100, pr.xor5*100,
               pr.mc, pr.narma, pr.wave4*100, pr.wave8*100);
    }

    thermal_guard("phase1-done");

    // ═══════════════════════════════════════════════════════════════
    // PHASE 2: Three-Population Architecture
    // ═══════════════════════════════════════════════════════════════
    printf("\n═══ PHASE 2: Three-Population Architecture ═══\n");
    printf("Pop A (tid 0-63):   Memory — branch divergence\n");
    printf("Pop B (tid 64-127): Nonlinear — L1 + atomic\n");
    printf("Pop C (tid 128-191): Regression — dense ESN (SR=1.05, T=0.65)\n");
    printf("Neurons: %d (%d CUs x %d threads)\n\n", TRIPOP_NEURONS, N_CUS, TRIPOP_THREADS);

    struct TripopResult {
        const char* name;
        int mode;
        float xor1, xor2, xor3, xor4, xor5;
        float mc;
        float narma;
        float wave4, wave8;
    };
    std::vector<TripopResult> tripop_results;

    auto run_tripop = [&](const char* name, int mode) {
        printf("Running TRIPOP %s (mode=%d)...\n", name, mode);
        TripopResult tr; tr.name = name; tr.mode = mode;

        auto run_kernel = [&](float* inp, int steps) {
            HIP_CHECK(hipMemcpy(d_input, inp, steps*sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemset(d_state, 0, TRIPOP_NEURONS*steps*sizeof(float)));
            HIP_CHECK(hipMemset((void*)d_xchg, 0, max_neurons*sizeof(float)));
            HIP_CHECK(hipMemset(d_atomic, 0, N_CUS*16*sizeof(float)));
            tripop_reservoir<<<N_CUS, TRIPOP_THREADS>>>(d_state, d_xchg, d_atomic, d_input, steps, mode);
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(h_state, d_state, TRIPOP_NEURONS*steps*sizeof(float), hipMemcpyDeviceToHost));
        };

        // XOR
        gen_xor_input(h_input, STEPS);
        run_kernel(h_input, STEPS);
        std::vector<std::vector<float>> X;
        extract_tripop(h_state, STEPS, WASHOUT, TRIPOP_SAMPLE, X);
        auto br = run_bench(X, h_input, lambda);
        tr.xor1=br.xor_acc[0]; tr.xor2=br.xor_acc[1]; tr.xor3=br.xor_acc[2]; tr.xor4=br.xor_acc[3]; tr.xor5=br.xor_acc[4];

        // MC
        gen_mc_input(h_input, STEPS);
        run_kernel(h_input, STEPS);
        X.clear();
        extract_tripop(h_state, STEPS, WASHOUT, TRIPOP_SAMPLE, X);
        tr.mc = compute_mc(X, h_input, lambda);

        // NARMA
        gen_narma_input(h_input, h_narma, STEPS);
        run_kernel(h_input, STEPS);
        X.clear();
        extract_tripop(h_state, STEPS, WASHOUT, TRIPOP_SAMPLE, X);
        tr.narma = compute_narma(X, h_narma, lambda);

        // Wave4
        gen_wave_input(h_input, h_labels, STEPS, 4);
        run_kernel(h_input, STEPS);
        X.clear();
        extract_tripop(h_state, STEPS, WASHOUT, TRIPOP_SAMPLE, X);
        tr.wave4 = compute_wave(X, h_labels, 4, lambda);

        // Wave8
        gen_wave_input(h_input, h_labels, STEPS, 8);
        run_kernel(h_input, STEPS);
        X.clear();
        extract_tripop(h_state, STEPS, WASHOUT, TRIPOP_SAMPLE, X);
        tr.wave8 = compute_wave(X, h_labels, 8, lambda);

        tripop_results.push_back(tr);
        thermal_guard(name);
    };

    run_tripop("FULL (A+B+C)", 0);
    run_tripop("NO_A (B+C only)", 1);
    run_tripop("NO_B (A+C only)", 2);
    run_tripop("NO_C (A+B only)", 3);
    run_tripop("CONTROL (no HW)", 4);

    // Print Phase 2 table
    printf("\n%-22s %6s %6s %6s %6s %6s %6s %7s %6s %6s\n",
           "Config", "XOR1", "XOR2", "XOR3", "XOR4", "XOR5", "MC", "NARMA", "Wave4", "Wave8");
    printf("────────────────────────────────────────────────────────────────────────────────────────────\n");
    for (auto& tr : tripop_results) {
        printf("%-22s %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.3f %7.3f %5.1f%% %5.1f%%\n",
               tr.name, tr.xor1*100, tr.xor2*100, tr.xor3*100, tr.xor4*100, tr.xor5*100,
               tr.mc, tr.narma, tr.wave4*100, tr.wave8*100);
    }

    // ═══════════════════════════════════════════════════════════════
    // PHASE 3: Per-Population Analysis for FULL tripop
    // ═══════════════════════════════════════════════════════════════
    printf("\n═══ PHASE 3: Per-Population Breakdown (FULL tripop) ═══\n");

    // Re-run FULL for XOR analysis
    gen_xor_input(h_input, STEPS);
    HIP_CHECK(hipMemcpy(d_input, h_input, STEPS*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_state, 0, TRIPOP_NEURONS*STEPS*sizeof(float)));
    HIP_CHECK(hipMemset((void*)d_xchg, 0, max_neurons*sizeof(float)));
    HIP_CHECK(hipMemset(d_atomic, 0, N_CUS*16*sizeof(float)));
    tripop_reservoir<<<N_CUS, TRIPOP_THREADS>>>(d_state, d_xchg, d_atomic, d_input, STEPS, 0);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_state, d_state, TRIPOP_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost));

    const char* pop_names[] = {"Pop A (memory)", "Pop B (nonlinear)", "Pop C (regression)"};
    for (int p = 0; p < 3; p++) {
        std::vector<std::vector<float>> Xp;
        extract_single_pop(h_state, STEPS, WASHOUT, 128, p, Xp);
        auto br = run_bench(Xp, h_input, lambda);
        printf("  %s: XOR1=%5.1f%% XOR2=%5.1f%% XOR3=%5.1f%%\n",
               pop_names[p], br.xor_acc[0]*100, br.xor_acc[1]*100, br.xor_acc[2]*100);
    }

    // MC per-pop
    gen_mc_input(h_input, STEPS);
    HIP_CHECK(hipMemcpy(d_input, h_input, STEPS*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_state, 0, TRIPOP_NEURONS*STEPS*sizeof(float)));
    HIP_CHECK(hipMemset((void*)d_xchg, 0, max_neurons*sizeof(float)));
    HIP_CHECK(hipMemset(d_atomic, 0, N_CUS*16*sizeof(float)));
    tripop_reservoir<<<N_CUS, TRIPOP_THREADS>>>(d_state, d_xchg, d_atomic, d_input, STEPS, 0);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_state, d_state, TRIPOP_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost));

    for (int p = 0; p < 3; p++) {
        std::vector<std::vector<float>> Xp;
        extract_single_pop(h_state, STEPS, WASHOUT, 128, p, Xp);
        float mc = compute_mc(Xp, h_input, lambda);
        printf("  %s: MC=%.3f\n", pop_names[p], mc);
    }

    // NARMA per-pop
    gen_narma_input(h_input, h_narma, STEPS);
    HIP_CHECK(hipMemcpy(d_input, h_input, STEPS*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_state, 0, TRIPOP_NEURONS*STEPS*sizeof(float)));
    HIP_CHECK(hipMemset((void*)d_xchg, 0, max_neurons*sizeof(float)));
    HIP_CHECK(hipMemset(d_atomic, 0, N_CUS*16*sizeof(float)));
    tripop_reservoir<<<N_CUS, TRIPOP_THREADS>>>(d_state, d_xchg, d_atomic, d_input, STEPS, 0);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_state, d_state, TRIPOP_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost));

    for (int p = 0; p < 3; p++) {
        std::vector<std::vector<float>> Xp;
        extract_single_pop(h_state, STEPS, WASHOUT, 128, p, Xp);
        float narma = compute_narma(Xp, h_narma, lambda);
        printf("  %s: NARMA_R2=%.3f\n", pop_names[p], narma);
    }

    thermal_guard("phase3-done");

    // ═══════════════════════════════════════════════════════════════
    // TESTS
    // ═══════════════════════════════════════════════════════════════
    printf("\n═══ Tests ═══\n");
    int pass = 0, total = 0;

    auto test = [&](int id, const char* desc, bool cond) {
        total++;
        if (cond) { pass++; printf("T%d %s: PASS\n", id, desc); }
        else printf("T%d %s: FAIL\n", id, desc);
    };

    // Phase 1 tests: Do new mechanisms add signal?
    float ctrl_xor1 = probe_results[0].xor1;
    float ctrl_mc = probe_results[0].mc;
    float ctrl_narma = probe_results[0].narma;
    float ctrl_wave4 = probe_results[0].wave4;

    // Find atomic, wavefront, regbank results
    float atomic_xor1 = probe_results[1].xor1;
    float atomic_mc = probe_results[1].mc;
    float wf_xor1 = probe_results[2].xor1;
    float wf_mc = probe_results[2].mc;
    float awf_xor1 = probe_results[3].xor1;
    float awf_mc = probe_results[3].mc;

    test(860, "ATOMIC XOR1 > CONTROL XOR1", atomic_xor1 > ctrl_xor1 + 0.02f);
    test(861, "ATOMIC adds unique signal (XOR or MC diff)",
         fabsf(atomic_xor1 - ctrl_xor1) > 0.02f || fabsf(atomic_mc - ctrl_mc) > 0.05f);
    test(862, "WAVEFRONT_SCHED XOR1 > CONTROL", wf_xor1 > ctrl_xor1 + 0.02f);
    test(863, "WAVEFRONT_SCHED adds signal",
         fabsf(wf_xor1 - ctrl_xor1) > 0.02f || fabsf(wf_mc - ctrl_mc) > 0.05f);
    test(864, "ATOMIC+WF XOR1 > CONTROL", awf_xor1 > ctrl_xor1 + 0.02f);
    test(865, "Any new mechanism > CONTROL on some metric",
         atomic_xor1 > ctrl_xor1 + 0.01f || wf_xor1 > ctrl_xor1 + 0.01f || awf_xor1 > ctrl_xor1 + 0.01f
         || probe_results[1].wave4 > ctrl_wave4 + 0.01f || probe_results[2].wave4 > ctrl_wave4 + 0.01f
         || probe_results[3].wave4 > ctrl_wave4 + 0.01f);

    // Phase 2 tests: Tripop architecture
    if (tripop_results.size() >= 5) {
        auto& full = tripop_results[0];   // FULL
        auto& ctrl = tripop_results[4];   // CONTROL
        auto& no_c = tripop_results[3];   // NO_C

        test(866, "TRIPOP FULL XOR1 > CONTROL XOR1", full.xor1 > ctrl.xor1 + 0.02f);
        test(867, "TRIPOP FULL Wave4 > 80%", full.wave4 > 0.80f);
        test(868, "TRIPOP FULL MC > 0.3", full.mc > 0.3f);
        test(869, "TRIPOP FULL NARMA > CONTROL NARMA", full.narma > ctrl.narma);
        test(870, "TRIPOP FULL NARMA > -0.1 (regression not catastrophic)", full.narma > -0.1f);
        test(871, "Pop C improves NARMA (FULL > NO_C)", full.narma > no_c.narma);
        test(872, "TRIPOP FULL XOR1 > 70%", full.xor1 > 0.70f);
        test(873, "TRIPOP FULL Wave8 > 50%", full.wave8 > 0.50f);
        test(874, "TRIPOP FULL XOR1>70% AND MC>0.3 AND NARMA>-0.5 (balanced)",
             full.xor1 > 0.70f && full.mc > 0.3f && full.narma > -0.5f);
    }

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  z2266 SUMMARY: %d/%d PASS                                ║\n", pass, total);
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    // Cleanup
    HIP_CHECK(hipFree(d_state));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree((void*)d_xchg));
    HIP_CHECK(hipFree(d_atomic));
    free(h_state);
    free(h_input);
    free(h_narma);
    free(h_labels);

    return 0;
}
