/*  z2270_gpu_new_mechanisms.hip — Exploring 2 new GPU hardware mechanisms
 *
 *  Existing 7+ mechanisms: PLL, TLB, branch, L1/LDS, coalescing, atomic, wavefront scheduling
 *  Fourpop (z2268): XOR=77%, MC=0.615, NARMA=+0.261, Wave=97.7%
 *
 *  NEW MECHANISMS TO EXPLORE:
 *
 *  1. VGPR Occupancy Pressure
 *     RDNA 3.5 has 1536 VGPRs per SIMD (4 SIMDs per CU).
 *     More live VGPRs → fewer concurrent wavefronts → different timing/cache behavior.
 *     By varying the number of live registers based on input, we modulate occupancy
 *     and thus the micro-architectural timing landscape.
 *     Mechanism: declare arrays of volatile floats to force VGPR allocation,
 *     then conditionally read them to prevent dead code elimination.
 *
 *  2. LDS Bank Conflict Patterns (Multicast vs Conflict)
 *     LDS has 32 banks (4 bytes each). Reading the SAME address from all lanes = multicast (fast).
 *     Reading DIFFERENT addresses in the SAME bank = serial conflict (slow).
 *     Input determines which pattern → timing difference encodes input → computational primitive.
 *     This is DIFFERENT from existing L1 bank conflicts which use scattered writes.
 *     Here we use READ patterns with precise bank targeting.
 *
 *  For each mechanism:
 *    Phase 1: Independent probe (500 steps, timing distributions)
 *    Phase 2: Reservoir test (128 neurons, XOR τ=1-3, MC d=1-5, 4-class waveform)
 *    Phase 3: FIVEPOP architecture with best new mechanism
 *
 *  Compile: hipcc --offload-arch=gfx1100 -O1 -o scripts/z2270_gpu_new_mechanisms scripts/z2270_gpu_new_mechanisms.hip
 *  Run:     HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/z2270_gpu_new_mechanisms
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
#define STEPS 800
#define WASHOUT 150
#define POP_SIZE 64
#define THREADS 128
#define N_NEURONS (N_CUS * THREADS)
#define SAMPLE_N 256
#define PROBE_STEPS 500

// ═══════════════════════════════════════════════════════════════════════
// TEMPERATURE MONITOR
// ═══════════════════════════════════════════════════════════════════════
int read_gpu_temp() {
    FILE* f = fopen("/sys/class/hwmon/hwmon7/temp1_input", "r");
    if (!f) f = fopen("/sys/class/hwmon/hwmon8/temp1_input", "r");
    if (!f) return 50;
    int mc = 50000;
    if (fscanf(f, "%d", &mc) != 1) mc = 50000;
    fclose(f);
    return mc / 1000;
}

void thermal_guard(const char* phase) {
    int temp = read_gpu_temp();
    printf("  [thermal] %s: GPU edge = %d°C", phase, temp);
    if (temp > 75) {
        printf(" → COOLING 15s...");
        fflush(stdout);
        sleep(15);
        temp = read_gpu_temp();
        printf(" now %d°C", temp);
    }
    printf("\n");
    fflush(stdout);
}

// ═══════════════════════════════════════════════════════════════════════
// PROBE KERNEL 1: VGPR OCCUPANCY PRESSURE
// ═══════════════════════════════════════════════════════════════════════
// Strategy: Force different numbers of live VGPRs based on input.
// More VGPRs → lower occupancy → fewer concurrent wavefronts → different timing.
// We use volatile arrays to prevent the compiler from optimizing them away.
__global__ void probe_vgpr_occupancy(
    float* timing_out,      // [N_NEURONS * PROBE_STEPS]
    const float* input_seq, // [PROBE_STEPS]
    int steps
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= THREADS) return;
    int nid = bid * THREADS + tid;

    // Allocate a large register array — compiler will spill to VGPRs
    // 32 floats = 128 bytes = 32 VGPRs just for this array
    float reg_bank[32];
    for (int i = 0; i < 32; i++)
        reg_bank[i] = (float)(nid * 32 + i) * 0.001f;

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];

        // Input determines how DEEP into the register bank we read
        // High |u| → more registers accessed → more pressure → different timing
        int depth = 4 + (int)(28.0f * fabsf(u));  // 4-32 registers
        if (depth > 32) depth = 32;

        unsigned long long t0 = __clock64();

        // Force register reads with data dependency chain
        float accum = 0.0f;
        for (int i = 0; i < depth; i++) {
            // Data-dependent chain prevents instruction reordering
            int idx = ((int)(accum * 7.0f) + i) & 31;
            accum += reg_bank[idx];
            reg_bank[idx] = accum * 0.99f + u * 0.01f;  // write back prevents DCE
        }

        unsigned long long t1 = __clock64();
        float dt = (float)(t1 - t0);

        timing_out[nid * steps + t] = dt;

        // Prevent optimization
        if (accum == -9999.0f) timing_out[0] = accum;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PROBE KERNEL 2: LDS BANK CONFLICT PATTERNS
// ═══════════════════════════════════════════════════════════════════════
// LDS has 32 banks, each 4 bytes wide.
// Address mapping: bank = (address / 4) % 32
// Multicast: all lanes read SAME address → 1 cycle
// Conflict: N lanes read DIFFERENT addresses in SAME bank → N cycles
// We create input-dependent patterns: some inputs cause multicast, others cause conflicts.
__global__ void probe_lds_bank_pattern(
    float* timing_out,
    const float* input_seq,
    int steps
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= THREADS) return;
    int nid = bid * THREADS + tid;

    // LDS buffer: 32 banks × 8 entries per bank = 256 floats
    // Address for bank b, entry e: index = e * 32 + b
    __shared__ float lds_buf[256];

    // Initialize LDS
    lds_buf[tid] = (float)tid * 0.01f;
    if (tid < 128) lds_buf[128 + tid] = (float)(128 + tid) * 0.01f;
    __syncthreads();

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];

        // Input determines the read pattern:
        // u > 0: BROADCAST pattern — all lanes read from same address (fast)
        // u < 0: CONFLICT pattern — each lane reads different row in same bank (slow)
        // |u| controls which bank and how many conflicts

        int target_bank = (int)(fabsf(u) * 31.0f) & 31;

        unsigned long long t0 = __clock64();

        float val = 0.0f;
        int n_reads = 8 + (int)(16.0f * fabsf(u));  // 8-24 reads

        if (u > 0.0f) {
            // BROADCAST: all threads read same address → multicast → fast
            int addr = target_bank;  // all lanes hit same address
            for (int i = 0; i < n_reads; i++) {
                val += lds_buf[addr];
                addr = (addr + 32) & 255;  // move to next row in same bank
            }
        } else {
            // CONFLICT: each thread reads different row in same bank → serial → slow
            int base_row = tid & 7;  // 8 different rows per wave32 subset
            for (int i = 0; i < n_reads; i++) {
                int addr = ((base_row + i) & 7) * 32 + target_bank;
                val += lds_buf[addr];
            }
        }

        unsigned long long t1 = __clock64();
        float dt = (float)(t1 - t0);

        // Write back to prevent DCE
        lds_buf[tid] = val * 0.001f + lds_buf[tid] * 0.999f;
        __syncthreads();

        timing_out[nid * steps + t] = dt;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// RESERVOIR KERNEL WITH SELECTABLE MECHANISM
// ═══════════════════════════════════════════════════════════════════════
#define MECH_NONE     0
#define MECH_VGPR     1
#define MECH_LDS_BANK 2
#define MECH_BOTH     3

__global__ void mechanism_reservoir(
    float* state_out,
    volatile float* cu_xchg,
    const float* input_seq,
    int steps,
    int mechanism
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= THREADS) return;
    int nid = bid * THREADS + tid;

    __shared__ float lds[THREADS];
    // Extra LDS for bank conflict mechanism
    __shared__ float lds_bank[256];

    float v = 0.0f, h = 0.0f, slow = 0.0f;

    // Per-neuron random parameters
    uint32_t rng = nid * 2654435761u + 12345;
    rng *= 1103515245; float leak = 0.05f + 0.15f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float input_w = 0.05f + 0.20f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float thr = 0.4f + 0.5f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float bias = 0.02f * ((float)((rng>>16)&0xFF)/255.0f - 0.5f);
    rng *= 1103515245; int ln1 = (tid+1+((rng>>8)&7))%THREADS;
    rng *= 1103515245; int ln2 = (tid+THREADS-1-((rng>>8)&7))%THREADS;
    rng *= 1103515245; float wl = 0.02f + 0.06f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; int rcu = ((rng>>8)&0xFF)%N_CUS;
    rng *= 1103515245; int rtid = ((rng>>8)&0xFF)%THREADS;
    rng *= 1103515245; float wr = 0.01f + 0.03f * ((rng>>16)&0xFF)/255.0f;

    // VGPR pressure array
    float reg_bank[32];
    for (int i = 0; i < 32; i++)
        reg_bank[i] = (float)(nid * 32 + i) * 0.0001f;

    // Initialize bank LDS
    lds[tid] = 0.0f;
    lds_bank[tid] = (float)tid * 0.01f;
    if (tid < 128) lds_bank[128 + tid] = (float)(128 + tid) * 0.01f;
    __syncthreads();

    float dummy = 1.0001f + nid * 0.0001f;

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];
        float hw_signal = 0.0f;

        // ═══ MECHANISM 1: VGPR Occupancy Pressure ═══
        if (mechanism == MECH_VGPR || mechanism == MECH_BOTH) {
            int depth = 4 + (int)(28.0f * fabsf(u));
            if (depth > 32) depth = 32;

            unsigned long long vt0 = __clock64();
            float accum = 0.0f;
            for (int i = 0; i < depth; i++) {
                int idx = ((int)(accum * 7.0f) + i) & 31;
                accum += reg_bank[idx];
                reg_bank[idx] = accum * 0.99f + u * 0.01f;
            }
            unsigned long long vt1 = __clock64();

            // Normalize timing to [-1, 1] range
            float vgpr_signal = fminf(fmaxf(((float)(vt1 - vt0) - 60.0f) / 40.0f, -2.0f), 2.0f);
            hw_signal += 0.02f * vgpr_signal;

            // Also inject the accumulated value directly (register content is input-dependent)
            hw_signal += 0.01f * tanhf(accum * 0.1f);

            if (accum == -9999.0f) state_out[0] = accum;
        }

        // ═══ MECHANISM 2: LDS Bank Conflict Patterns ═══
        if (mechanism == MECH_LDS_BANK || mechanism == MECH_BOTH) {
            int target_bank = (int)(fabsf(u) * 31.0f) & 31;
            int n_reads = 8 + (int)(8.0f * fabsf(u));

            unsigned long long bt0 = __clock64();
            float bval = 0.0f;

            if (u > 0.0f) {
                // Broadcast: all threads same address → fast
                int addr = target_bank;
                for (int i = 0; i < n_reads; i++) {
                    bval += lds_bank[addr];
                    addr = (addr + 32) & 255;
                }
            } else {
                // Conflict: each thread different row, same bank → slow
                int base_row = tid & 7;
                for (int i = 0; i < n_reads; i++) {
                    int addr = ((base_row + i) & 7) * 32 + target_bank;
                    bval += lds_bank[addr];
                }
            }

            unsigned long long bt1 = __clock64();
            float bank_signal = fminf(fmaxf(((float)(bt1 - bt0) - 30.0f) / 20.0f, -2.0f), 2.0f);

            // Update bank LDS with neuron state for temporal memory
            lds_bank[tid] = bval * 0.001f + v * 0.1f;

            hw_signal += 0.03f * bank_signal;
        }

        // ═══ RECURRENCE ═══
        lds[tid] = v;
        __syncthreads();
        float lrec = wl * (lds[ln1] + lds[ln2]);
        cu_xchg[nid] = v; __threadfence();
        float rrec = wr * cu_xchg[rcu * THREADS + rtid];

        // Cross-PLL jitter (free)
        uint64_t sc = __clock64(); uint64_t wc = wall_clock64();
        float pll = (float)((sc - wc) & 0x1FF) / 256.0f - 1.0f;

        v = tanhf(
            (1.0f - leak) * v
          + input_w * u
          + lrec + rrec
          + bias
          + hw_signal
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

// ═══════════════════════════════════════════════════════════════════════
// CONTROL KERNEL (pure ESN)
// ═══════════════════════════════════════════════════════════════════════
__global__ void control_reservoir(
    float* state_out,
    volatile float* cu_xchg,
    const float* input_seq,
    int steps
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= THREADS) return;
    int nid = bid * THREADS + tid;

    __shared__ float lds[THREADS];
    lds[tid] = 0.0f;
    __syncthreads();

    float v = 0.0f, h = 0.0f, slow = 0.0f;
    uint32_t rng = nid * 2654435761u + 12345;
    rng *= 1103515245; float leak = 0.05f + 0.15f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float input_w = 0.05f + 0.20f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; /* thr */ rng *= 1103515245;
    rng *= 1103515245; int ln1 = (tid+1+((rng>>8)&7))%THREADS;
    rng *= 1103515245; int ln2 = (tid+THREADS-1-((rng>>8)&7))%THREADS;
    rng *= 1103515245; float wl = 0.02f + 0.06f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; int rcu = ((rng>>8)&0xFF)%N_CUS;
    rng *= 1103515245; int rtid = ((rng>>8)&0xFF)%THREADS;
    rng *= 1103515245; float wr = 0.01f + 0.03f * ((rng>>16)&0xFF)/255.0f;

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];
        lds[tid] = v; __syncthreads();
        float lrec = wl * (lds[ln1] + lds[ln2]);
        cu_xchg[nid] = v; __threadfence();
        float rrec = wr * cu_xchg[rcu * THREADS + rtid];
        v = tanhf((1.0f-leak)*v + input_w*u + lrec + rrec);
        h = 0.93f*h + 0.07f*v;
        slow = 0.99f*slow + 0.01f*v;
        state_out[nid*steps+t] = v + 0.3f*h + 0.1f*slow;
        __syncthreads();
    }
}

// ═══════════════════════════════════════════════════════════════════════
// FIVEPOP KERNEL — Best new mechanism as Pop E
// ═══════════════════════════════════════════════════════════════════════
#define FIVEPOP_POPS 5
#define FIVEPOP_THREADS (FIVEPOP_POPS * POP_SIZE)  // 320
#define FIVEPOP_NEURONS (N_CUS * FIVEPOP_THREADS)
#define FIVEPOP_SAMPLE 640  // 128 per pop
// LDS layout:
//   [0..63]    Pop A recurrence (branch — memory)
//   [64..127]  Pop B recurrence (L1 bank conflicts — XOR)
//   [128..191] Pop C recurrence (dense ESN — regression)
//   [192..255] Pop D recurrence (wavefront scheduling — classification)
//   [256..319] Pop E recurrence (new mechanism)
//   [320..383] Pop B L1 scratch
//   [384..639] Pop E bank pattern LDS (256 entries)
#define FIVEPOP_LDS 640

__global__ void fivepop_reservoir(
    float* state_out,
    volatile float* cu_xchg,
    const float* input_seq,
    int steps,
    int new_mechanism  // MECH_VGPR or MECH_LDS_BANK or MECH_BOTH
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= FIVEPOP_THREADS) return;
    int nid = bid * FIVEPOP_THREADS + tid;

    int pop = tid / POP_SIZE;      // 0=A, 1=B, 2=C, 3=D, 4=E
    int pop_tid = tid % POP_SIZE;

    __shared__ float lds[FIVEPOP_LDS];

    // Initialize LDS
    if (tid < FIVEPOP_LDS) lds[tid] = 0.0f;
    if (tid + FIVEPOP_THREADS < FIVEPOP_LDS)
        lds[tid + FIVEPOP_THREADS] = 0.0f;
    __syncthreads();

    float v = 0.0f, h = 0.0f, slow = 0.0f;
    uint32_t rng = nid * 2654435761u + 54321;

    // Per-neuron random parameters
    rng *= 1103515245; float leak_base = 0.05f + 0.15f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float input_w = 0.05f + 0.20f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float thr = 0.4f + 0.5f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float bias = 0.02f * ((float)((rng>>16)&0xFF)/255.0f - 0.5f);

    // Local recurrence within population
    int lds_base = pop * POP_SIZE;
    rng *= 1103515245; int ln1 = lds_base + (pop_tid + 1 + ((rng>>8)&7)) % POP_SIZE;
    rng *= 1103515245; int ln2 = lds_base + (pop_tid + POP_SIZE - 1 - ((rng>>8)&7)) % POP_SIZE;
    rng *= 1103515245;
    float wl = (pop == 2) ? (0.04f + 0.08f * ((rng>>16)&0xFF)/255.0f)
                           : (0.02f + 0.06f * ((rng>>16)&0xFF)/255.0f);

    // Cross-CU recurrence
    rng *= 1103515245; int rcu = ((rng>>8)&0xFF) % N_CUS;
    rng *= 1103515245; int rtid = ((rng>>8)&0xFF) % FIVEPOP_THREADS;
    rng *= 1103515245; float wr = 0.01f + 0.03f * ((rng>>16)&0xFF)/255.0f;

    // Pop C: additional dense connections
    int c_ln3 = lds_base + (pop_tid + 3 + ((rng>>8)&15)) % POP_SIZE; rng *= 1103515245;
    int c_ln4 = lds_base + (pop_tid + POP_SIZE - 3 - ((rng>>8)&15)) % POP_SIZE; rng *= 1103515245;
    int c_ln5 = lds_base + (pop_tid + 7 + ((rng>>8)&15)) % POP_SIZE; rng *= 1103515245;
    int c_ln6 = lds_base + (pop_tid + POP_SIZE - 7 - ((rng>>8)&15)) % POP_SIZE; rng *= 1103515245;
    float wl2 = 0.03f + 0.05f * ((rng>>16)&0xFF)/255.0f; rng *= 1103515245;

    // Pop C global integration (from all 5 populations)
    int c_g1 = (0*POP_SIZE) + (pop_tid + ((rng>>8)&0x3F)) % POP_SIZE; rng *= 1103515245;
    int c_g2 = (1*POP_SIZE) + (pop_tid + ((rng>>8)&0x3F)) % POP_SIZE; rng *= 1103515245;
    int c_g3 = (3*POP_SIZE) + (pop_tid + ((rng>>8)&0x3F)) % POP_SIZE; rng *= 1103515245;
    int c_g4 = (4*POP_SIZE) + (pop_tid + ((rng>>8)&0x3F)) % POP_SIZE; rng *= 1103515245;
    float wg = 0.015f + 0.025f * ((rng>>16)&0xFF)/255.0f; rng *= 1103515245;

    // Cross-population coupling
    int xp1 = ((pop+1)%FIVEPOP_POPS)*POP_SIZE + (pop_tid + ((rng>>8)&0x1F))%POP_SIZE; rng *= 1103515245;
    int xp2 = ((pop+2)%FIVEPOP_POPS)*POP_SIZE + (pop_tid + ((rng>>8)&0x1F))%POP_SIZE; rng *= 1103515245;
    int xp3 = ((pop+3)%FIVEPOP_POPS)*POP_SIZE + (pop_tid + ((rng>>8)&0x1F))%POP_SIZE; rng *= 1103515245;
    int xp4 = ((pop+4)%FIVEPOP_POPS)*POP_SIZE + (pop_tid + ((rng>>8)&0x1F))%POP_SIZE; rng *= 1103515245;

    // Pop A branch threshold
    float bthr = 0.5f + 0.3f * ((nid & 0xF) / 15.0f);

    // Pop D wavefront info
    int my_wavefront = tid / 32;
    int other_wf_tid = ((my_wavefront + 1) % (FIVEPOP_THREADS / 32)) * 32 + (tid % 32);
    int d_other_lds = 192 + (other_wf_tid % POP_SIZE);

    // Pop E: VGPR pressure registers
    float reg_bank[32];
    for (int i = 0; i < 32; i++)
        reg_bank[i] = (float)(nid * 32 + i) * 0.0001f;

    // Pop-specific leak rates
    float c_leak = leak_base;
    if (pop == 1) c_leak = 0.03f + 0.05f * ((nid & 0xFF) / 255.0f);  // Pop B: longer memory
    if (pop == 2) c_leak = 0.05f + 0.05f * ((nid & 0xFF) / 255.0f);  // Pop C: moderate
    if (pop == 4) c_leak = 0.04f + 0.06f * ((nid & 0xFF) / 255.0f);  // Pop E: slightly slow

    float c_sr = 1.05f;
    float c_temp = 0.65f;
    float dummy = 1.0001f + nid * 0.0001f;

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];
        float hw_signal = 0.0f;

        // ═══ Pop A: Branch divergence (memory) ═══
        if (pop == 0) {
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
            hw_signal = 0.04f * branch_val;
        }

        // ═══ Pop B: L1 bank conflicts + branch (XOR) ═══
        if (pop == 1) {
            // Branch divergence
            float branch_val = 0.0f;
            if (u > 0.0f) {
                for (int i = 0; i < 20; i++)
                    dummy = __fmaf_rn(dummy, 0.99f, 0.01f);
                branch_val = 1.0f;
            } else {
                int iv = (int)(v * 1000.0f);
                dummy = (float)(iv) * 0.001f + dummy * 0.0001f;
                branch_val = -1.0f;
            }

            // L1 scratch conflicts
            unsigned long long l1_t0 = __clock64();
            int l1_iters = 5 + (int)(10.0f * fabsf(u));
            int scratch_base = 320;  // Pop B scratch region
            for (int i = 0; i < l1_iters; i++)
                lds[scratch_base + (pop_tid + i * ((int)(u * 17.0f) & 0xF)) % POP_SIZE] += 0.0001f;
            unsigned long long l1_t1 = __clock64();
            float l1_signal = fminf(fmaxf(((float)(l1_t1 - l1_t0) - 30.0f) / 20.0f, -2.0f), 2.0f);

            // Product nonlinearity for XOR
            float sign_u = (u > 0.0f ? 1.0f : -1.0f);
            float product_term = v * sign_u;
            float temporal_xor = h * sign_u;
            float deep_xor = slow * sign_u;

            hw_signal = 0.06f * branch_val + 0.005f * l1_signal
                      + 0.08f * product_term + 0.05f * temporal_xor + 0.02f * deep_xor;
        }

        // ═══ Pop D: Wavefront scheduling (classification) ═══
        float wf_coupling = 0.0f;
        if (pop == 3) {
            int work_amount = 10 + my_wavefront * 5 + (int)(10.0f * fabsf(u));
            unsigned long long wf_t0 = __clock64();
            for (int i = 0; i < work_amount; i++)
                dummy = __fmaf_rn(dummy, 1.000001f, 0.0001f);
            unsigned long long wf_t1 = __clock64();
            float wf_signal = fminf(fmaxf(((float)(wf_t1 - wf_t0) - 50.0f) / 30.0f, -2.0f), 2.0f);
            wf_coupling = lds[d_other_lds];
            hw_signal = 0.005f * wf_signal + 0.003f * wf_coupling;
        }

        // ═══ Pop E: New mechanism ═══
        if (pop == 4) {
            if (new_mechanism == MECH_VGPR || new_mechanism == MECH_BOTH) {
                int depth = 4 + (int)(28.0f * fabsf(u));
                if (depth > 32) depth = 32;
                unsigned long long vt0 = __clock64();
                float accum = 0.0f;
                for (int i = 0; i < depth; i++) {
                    int idx = ((int)(accum * 7.0f) + i) & 31;
                    accum += reg_bank[idx];
                    reg_bank[idx] = accum * 0.99f + u * 0.01f;
                }
                unsigned long long vt1 = __clock64();
                float vgpr_signal = fminf(fmaxf(((float)(vt1 - vt0) - 60.0f) / 40.0f, -2.0f), 2.0f);
                hw_signal += 0.02f * vgpr_signal + 0.01f * tanhf(accum * 0.1f);
                if (accum == -9999.0f) state_out[0] = accum;
            }

            if (new_mechanism == MECH_LDS_BANK || new_mechanism == MECH_BOTH) {
                int target_bank = (int)(fabsf(u) * 31.0f) & 31;
                int n_reads = 8 + (int)(8.0f * fabsf(u));
                int bank_lds_base = 384;  // Pop E bank pattern LDS

                unsigned long long bt0 = __clock64();
                float bval = 0.0f;

                if (u > 0.0f) {
                    int addr = target_bank;
                    for (int i = 0; i < n_reads; i++) {
                        bval += lds[bank_lds_base + (addr & 255)];
                        addr = (addr + 32) & 255;
                    }
                } else {
                    int base_row = pop_tid & 7;
                    for (int i = 0; i < n_reads; i++) {
                        int addr = ((base_row + i) & 7) * 32 + target_bank;
                        bval += lds[bank_lds_base + (addr & 255)];
                    }
                }

                unsigned long long bt1 = __clock64();
                float bank_signal = fminf(fmaxf(((float)(bt1 - bt0) - 30.0f) / 20.0f, -2.0f), 2.0f);

                // Store neuron state in bank LDS for temporal feedback
                lds[bank_lds_base + pop_tid] = bval * 0.001f + v * 0.1f;

                hw_signal += 0.03f * bank_signal;
            }
        }

        // ═══ ALL: Sync before recurrence ═══
        __syncthreads();

        // Recurrence
        int my_lds_slot = pop * POP_SIZE + pop_tid;
        lds[my_lds_slot] = v;
        __syncthreads();

        float lrec = wl * (lds[ln1] + lds[ln2]);

        // Pop C dense connections
        float dense_rec = 0.0f;
        if (pop == 2) {
            dense_rec = wl2 * c_sr * (lds[c_ln3] + lds[c_ln4] + lds[c_ln5] + lds[c_ln6]);
        }

        // Pop C global integration
        float global_rec = 0.0f;
        if (pop == 2) {
            global_rec = wg * (lds[c_g1] + lds[c_g2] + lds[c_g3] + lds[c_g4]);
        }

        // Cross-population coupling
        float xpop_w = (pop == 2) ? 0.012f : 0.003f;
        float xpop = xpop_w * (lds[xp1] + lds[xp2] + lds[xp3] + lds[xp4]);

        // Cross-CU
        cu_xchg[nid] = v; __threadfence();
        float rrec = wr * cu_xchg[rcu * FIVEPOP_THREADS + rtid];

        // PLL jitter (free)
        uint64_t sc = __clock64(); uint64_t wc = wall_clock64();
        float pll = (float)((sc - wc) & 0x1FF) / 256.0f - 1.0f;

        // State update
        float c_input_boost = (pop == 2) ? 1.2f : 1.0f;
        float pre_act = (1.0f - c_leak) * v
                      + c_input_boost * input_w * u
                      + lrec + rrec + dense_rec + global_rec
                      + xpop + bias + hw_signal
                      + 0.005f * pll;

        if (pop == 2) {
            v = tanhf(pre_act / c_temp);
        } else {
            v = tanhf(pre_act);
        }

        if (pop != 2 && v > thr) { v -= thr; }

        h = 0.93f * h + 0.07f * v;
        slow = 0.99f * slow + 0.01f * v;

        float output;
        if (pop == 1) {
            float sign_u_out = (u > 0.0f ? 1.0f : -1.0f);
            output = v + 0.3f*h + 0.1f*slow + 0.2f*(v * sign_u_out);
        } else if (pop == 2) {
            output = v + 0.3f*h + 0.1f*slow + 0.15f*v*v;
        } else {
            output = v + 0.3f*h + 0.1f*slow;
        }
        state_out[nid * steps + t] = output;
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

void extract_state_matrix(float* state, int n_neurons, int steps,
                          int washout, int sample_n, int n_threads_per_cu,
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

// Balanced extraction for fivepop: equal samples from each population
void extract_fivepop(float* state, int steps, int washout, int sample_n,
                     std::vector<std::vector<float>>& X) {
    int per_pop = sample_n / FIVEPOP_POPS;
    int stride_pop = (N_CUS * POP_SIZE) / per_pop;
    if (stride_pop < 1) stride_pop = 1;

    for (int t = washout; t < steps; t++) {
        std::vector<float> row;
        for (int p = 0; p < FIVEPOP_POPS; p++) {
            for (int si = 0; si < per_pop; si++) {
                int pop_idx = si * stride_pop;
                if (pop_idx >= N_CUS * POP_SIZE) break;
                int cu = pop_idx / POP_SIZE;
                int ptid = pop_idx % POP_SIZE;
                int nid = cu * FIVEPOP_THREADS + p * POP_SIZE + ptid;
                float v = state[nid * steps + t];
                if (v != v || v > 1e20 || v < -1e20) v = 0;
                row.push_back(v);
            }
        }
        X.push_back(row);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// RIDGE REGRESSION & CLASSIFICATION
// ═══════════════════════════════════════════════════════════════════════
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
// BENCHMARK TASKS
// ═══════════════════════════════════════════════════════════════════════
struct BenchResult {
    float xor1, xor2, xor3;
    float mc1, mc2, mc3, mc4, mc5, mc_total;
    float wave4;
    float narma_nrmse;
    const char* name;
};

BenchResult run_benchmark(std::vector<std::vector<float>>& X,
                          float* h_input, int steps, int washout) {
    BenchResult res = {};
    int T = steps - washout;
    int split = T / 2;
    float lambda = 1.0f;

    // ─── XOR τ=1,2,3 ───
    for (int tau = 1; tau <= 3; tau++) {
        std::vector<std::vector<float>> Xtr, Xte;
        std::vector<int> ytr, yte;
        for (int t = tau; t < T; t++) {
            int s1 = (h_input[washout + t] > 0) ? 1 : 0;
            int s2 = (h_input[washout + t - tau] > 0) ? 1 : 0;
            int label = s1 ^ s2;
            if (t < split) { Xtr.push_back(X[t]); ytr.push_back(label); }
            else { Xte.push_back(X[t]); yte.push_back(label); }
        }
        float acc = ridge_classify_binary(Xtr, ytr, Xte, yte, lambda);
        if (tau == 1) res.xor1 = acc;
        else if (tau == 2) res.xor2 = acc;
        else res.xor3 = acc;
    }

    // ─── Memory Capacity d=1..5 ───
    res.mc_total = 0;
    for (int d = 1; d <= 5; d++) {
        std::vector<std::vector<float>> Xtr, Xte;
        std::vector<float> ytr, yte;
        for (int t = d; t < T; t++) {
            float target = h_input[washout + t - d];
            if (t < split) { Xtr.push_back(X[t]); ytr.push_back(target); }
            else { Xte.push_back(X[t]); yte.push_back(target); }
        }
        RegResult rr = ridge_regress(Xtr, ytr, Xte, yte, lambda);
        float mc_d = fmaxf(rr.r2, 0.0f);
        if (d == 1) res.mc1 = mc_d;
        else if (d == 2) res.mc2 = mc_d;
        else if (d == 3) res.mc3 = mc_d;
        else if (d == 4) res.mc4 = mc_d;
        else res.mc5 = mc_d;
        res.mc_total += mc_d;
    }

    // ─── Waveform classification (4-class) ───
    {
        std::vector<std::vector<float>> Xtr, Xte;
        std::vector<int> ytr, yte;
        for (int t = 0; t < T; t++) {
            float u = h_input[washout + t];
            int cls;
            if (u < -0.5f) cls = 0;
            else if (u < 0.0f) cls = 1;
            else if (u < 0.5f) cls = 2;
            else cls = 3;
            if (t < split) { Xtr.push_back(X[t]); ytr.push_back(cls); }
            else { Xte.push_back(X[t]); yte.push_back(cls); }
        }
        res.wave4 = ridge_classify_multi(Xtr, ytr, Xte, yte, 4, lambda);
    }

    // ─── NARMA-10 ───
    {
        std::vector<float> narma(T, 0);
        float y_narma = 0;
        for (int t = 0; t < T; t++) {
            float ut = 0.5f * (h_input[washout + t] + 1.0f) * 0.5f;
            float yp = y_narma;
            int t10 = t >= 10 ? t - 10 : 0;
            float ut10 = 0.5f * (h_input[washout + t10] + 1.0f) * 0.5f;
            y_narma = 0.3f * yp + 0.05f * yp * (yp + ut) + 1.5f * ut * ut10 + 0.1f;
            if (y_narma > 10.0f) y_narma = 10.0f;
            if (y_narma < -10.0f) y_narma = -10.0f;
            narma[t] = y_narma;
        }
        std::vector<std::vector<float>> Xtr, Xte;
        std::vector<float> ytr, yte;
        for (int t = 10; t < T; t++) {
            if (t < split) { Xtr.push_back(X[t]); ytr.push_back(narma[t]); }
            else { Xte.push_back(X[t]); yte.push_back(narma[t]); }
        }
        RegResult rr = ridge_regress(Xtr, ytr, Xte, yte, lambda);
        res.narma_nrmse = rr.nrmse;
    }

    return res;
}

void print_bench(BenchResult& r) {
    printf("    XOR:  τ=1 %.1f%%  τ=2 %.1f%%  τ=3 %.1f%%\n",
           r.xor1*100, r.xor2*100, r.xor3*100);
    printf("    MC:   d1=%.3f d2=%.3f d3=%.3f d4=%.3f d5=%.3f  total=%.3f\n",
           r.mc1, r.mc2, r.mc3, r.mc4, r.mc5, r.mc_total);
    printf("    Wave4: %.1f%%\n", r.wave4*100);
    printf("    NARMA-10 NRMSE: %.4f (lower=better, <1.0=useful)\n", r.narma_nrmse);
}


// ═══════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════
int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  z2270: New GPU Hardware Mechanism Exploration             ║\n");
    printf("║  Mechanisms: VGPR Occupancy + LDS Bank Conflict Patterns  ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("GPU: %s, CUs: %d, VGPRs/SIMD: %d, LDS/CU: %d KB\n\n",
           prop.name, prop.multiProcessorCount,
           prop.regsPerBlock,  // total registers
           (int)(prop.sharedMemPerBlock / 1024));

    // Generate input sequence
    float* h_input = (float*)malloc(STEPS * sizeof(float));
    for (int t = 0; t < STEPS; t++)
        h_input[t] = randf_signed();

    float *d_state, *d_input;
    volatile float *d_xchg;

    // ═════════════════════════════════════════════════════════════════
    // PHASE 1: INDEPENDENT PROBES
    // ═════════════════════════════════════════════════════════════════
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("PHASE 1: Independent Mechanism Probes (%d steps)\n", PROBE_STEPS);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    thermal_guard("Phase 1 start");

    // Probe input
    float* h_probe_input = (float*)malloc(PROBE_STEPS * sizeof(float));
    for (int t = 0; t < PROBE_STEPS; t++)
        h_probe_input[t] = randf_signed();

    float* d_probe_input;
    HIP_CHECK(hipMalloc(&d_probe_input, PROBE_STEPS * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_probe_input, h_probe_input, PROBE_STEPS * sizeof(float), hipMemcpyHostToDevice));

    float* d_timing;
    HIP_CHECK(hipMalloc(&d_timing, (long)N_NEURONS * PROBE_STEPS * sizeof(float)));

    float* h_timing = (float*)malloc((long)N_NEURONS * PROBE_STEPS * sizeof(float));

    // ─── Probe 1: VGPR Occupancy Pressure ───
    printf("─── Probe 1: VGPR Occupancy Pressure ───\n");
    HIP_CHECK(hipMemset(d_timing, 0, (long)N_NEURONS * PROBE_STEPS * sizeof(float)));
    probe_vgpr_occupancy<<<N_CUS, THREADS>>>(d_timing, d_probe_input, PROBE_STEPS);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_timing, d_timing, (long)N_NEURONS * PROBE_STEPS * sizeof(float), hipMemcpyDeviceToHost));

    // Analyze timing distributions for positive vs negative inputs
    {
        float sum_pos = 0, sum_neg = 0, sum_high = 0, sum_low = 0;
        int n_pos = 0, n_neg = 0, n_high = 0, n_low = 0;
        float min_t = 1e20, max_t = -1e20;

        for (int t = 50; t < PROBE_STEPS; t++) {
            float u = h_probe_input[t];
            // Sample 32 neurons spread across CUs
            for (int s = 0; s < 32; s++) {
                int nid = s * (N_NEURONS / 32);
                float dt = h_timing[nid * PROBE_STEPS + t];
                if (dt != dt || dt > 1e10) continue;
                if (dt < min_t) min_t = dt;
                if (dt > max_t) max_t = dt;

                if (u > 0) { sum_pos += dt; n_pos++; }
                else { sum_neg += dt; n_neg++; }
                if (fabsf(u) > 0.5f) { sum_high += dt; n_high++; }
                else { sum_low += dt; n_low++; }
            }
        }

        float mean_pos = n_pos > 0 ? sum_pos / n_pos : 0;
        float mean_neg = n_neg > 0 ? sum_neg / n_neg : 0;
        float mean_high = n_high > 0 ? sum_high / n_high : 0;
        float mean_low = n_low > 0 ? sum_low / n_low : 0;

        printf("  Timing range: [%.1f, %.1f] cycles\n", min_t, max_t);
        printf("  Mean timing (u>0): %.2f cycles\n", mean_pos);
        printf("  Mean timing (u<0): %.2f cycles\n", mean_neg);
        printf("  Ratio pos/neg: %.4f\n", mean_pos / (mean_neg > 0 ? mean_neg : 1));
        printf("  Mean timing (|u|>0.5): %.2f cycles\n", mean_high);
        printf("  Mean timing (|u|<0.5): %.2f cycles\n", mean_low);
        printf("  Ratio high/low: %.4f\n", mean_high / (mean_low > 0 ? mean_low : 1));
        printf("  Dynamic range: %.2f cycles (%.1f%%)\n",
               max_t - min_t, (max_t - min_t) / (mean_pos > 0 ? mean_pos : 1) * 100);

        // Compute CV across neurons for a single timestep
        float tsum = 0, tsum2 = 0;
        int tn = 0;
        for (int s = 0; s < N_NEURONS; s += N_NEURONS/128) {
            float dt = h_timing[s * PROBE_STEPS + 200];
            if (dt != dt || dt > 1e10) continue;
            tsum += dt; tsum2 += dt * dt; tn++;
        }
        float tmean = tsum / tn;
        float tstd = sqrtf(tsum2/tn - tmean*tmean);
        printf("  Cross-neuron CV (t=200): %.4f (std=%.2f, mean=%.2f)\n",
               tstd / tmean, tstd, tmean);
    }
    printf("\n");

    thermal_guard("After VGPR probe");

    // ─── Probe 2: LDS Bank Conflict Patterns ───
    printf("─── Probe 2: LDS Bank Conflict Patterns ───\n");
    HIP_CHECK(hipMemset(d_timing, 0, (long)N_NEURONS * PROBE_STEPS * sizeof(float)));
    probe_lds_bank_pattern<<<N_CUS, THREADS>>>(d_timing, d_probe_input, PROBE_STEPS);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_timing, d_timing, (long)N_NEURONS * PROBE_STEPS * sizeof(float), hipMemcpyDeviceToHost));

    {
        float sum_bcast = 0, sum_conflict = 0;
        int n_bcast = 0, n_conflict = 0;
        float min_t = 1e20, max_t = -1e20;

        for (int t = 50; t < PROBE_STEPS; t++) {
            float u = h_probe_input[t];
            for (int s = 0; s < 32; s++) {
                int nid = s * (N_NEURONS / 32);
                float dt = h_timing[nid * PROBE_STEPS + t];
                if (dt != dt || dt > 1e10) continue;
                if (dt < min_t) min_t = dt;
                if (dt > max_t) max_t = dt;

                if (u > 0) { sum_bcast += dt; n_bcast++; }    // broadcast path
                else { sum_conflict += dt; n_conflict++; }      // conflict path
            }
        }

        float mean_bcast = n_bcast > 0 ? sum_bcast / n_bcast : 0;
        float mean_conflict = n_conflict > 0 ? sum_conflict / n_conflict : 0;

        printf("  Timing range: [%.1f, %.1f] cycles\n", min_t, max_t);
        printf("  Mean timing BROADCAST (u>0): %.2f cycles\n", mean_bcast);
        printf("  Mean timing CONFLICT  (u<0): %.2f cycles\n", mean_conflict);
        printf("  Conflict/Broadcast ratio: %.4f\n",
               mean_conflict / (mean_bcast > 0 ? mean_bcast : 1));
        printf("  Absolute difference: %.2f cycles\n", fabsf(mean_conflict - mean_bcast));
        printf("  Dynamic range: %.2f cycles (%.1f%%)\n",
               max_t - min_t, (max_t - min_t) / (mean_bcast > 0 ? mean_bcast : 1) * 100);

        // Cross-neuron CV
        float tsum = 0, tsum2 = 0;
        int tn = 0;
        for (int s = 0; s < N_NEURONS; s += N_NEURONS/128) {
            float dt = h_timing[s * PROBE_STEPS + 200];
            if (dt != dt || dt > 1e10) continue;
            tsum += dt; tsum2 += dt * dt; tn++;
        }
        float tmean = tsum / tn;
        float tstd = sqrtf(tsum2/tn - tmean*tmean);
        printf("  Cross-neuron CV (t=200): %.4f (std=%.2f, mean=%.2f)\n",
               tstd / tmean, tstd, tmean);
    }
    printf("\n");

    HIP_CHECK(hipFree(d_timing));
    HIP_CHECK(hipFree(d_probe_input));
    free(h_timing);
    free(h_probe_input);

    thermal_guard("After LDS probe");

    // ═════════════════════════════════════════════════════════════════
    // PHASE 2: RESERVOIR TESTS (each mechanism independently)
    // ═════════════════════════════════════════════════════════════════
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("PHASE 2: Reservoir Benchmarks (128 neurons, %d steps)\n", STEPS);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    HIP_CHECK(hipMalloc(&d_state, (long)N_NEURONS * STEPS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_input, STEPS * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_xchg, N_NEURONS * sizeof(float)));

    float* h_state = (float*)malloc((long)N_NEURONS * STEPS * sizeof(float));

    HIP_CHECK(hipMemcpy(d_input, h_input, STEPS * sizeof(float), hipMemcpyHostToDevice));

    struct MechConfig {
        const char* name;
        int mechanism;
    } mechs[] = {
        {"CONTROL (pure ESN)",    MECH_NONE},
        {"VGPR_ONLY",             MECH_VGPR},
        {"LDS_BANK_ONLY",         MECH_LDS_BANK},
        {"VGPR + LDS_BANK",       MECH_BOTH},
    };

    BenchResult results[4];

    for (int m = 0; m < 4; m++) {
        printf("─── %s ───\n", mechs[m].name);

        HIP_CHECK(hipMemset(d_state, 0, (long)N_NEURONS * STEPS * sizeof(float)));
        HIP_CHECK(hipMemset((void*)d_xchg, 0, N_NEURONS * sizeof(float)));

        if (mechs[m].mechanism == MECH_NONE) {
            control_reservoir<<<N_CUS, THREADS>>>(d_state, d_xchg, d_input, STEPS);
        } else {
            mechanism_reservoir<<<N_CUS, THREADS>>>(d_state, d_xchg, d_input, STEPS, mechs[m].mechanism);
        }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_state, d_state, (long)N_NEURONS * STEPS * sizeof(float), hipMemcpyDeviceToHost));

        std::vector<std::vector<float>> X;
        extract_state_matrix(h_state, N_NEURONS, STEPS, WASHOUT, SAMPLE_N, THREADS, X);

        results[m] = run_benchmark(X, h_input, STEPS, WASHOUT);
        results[m].name = mechs[m].name;
        print_bench(results[m]);
        printf("\n");

        thermal_guard(mechs[m].name);
    }

    // Print comparison table
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("PHASE 2 COMPARISON TABLE\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("%-22s  XOR-1   XOR-2   XOR-3   MC_tot  Wave4   NARMA\n", "Mechanism");
    printf("─────────────────────────────────────────────────────────────────\n");
    for (int m = 0; m < 4; m++) {
        printf("%-22s  %5.1f%%  %5.1f%%  %5.1f%%  %5.3f  %5.1f%%  %.4f\n",
               results[m].name,
               results[m].xor1*100, results[m].xor2*100, results[m].xor3*100,
               results[m].mc_total, results[m].wave4*100, results[m].narma_nrmse);
    }

    // Determine best new mechanism
    // Score: weighted by what we need most (XOR improvement + NARMA improvement)
    int best_mech = MECH_VGPR;
    float best_score = -1e20;
    for (int m = 1; m < 4; m++) {
        // Improvement over control
        float xor_improve = (results[m].xor1 - results[0].xor1)
                          + 0.7f * (results[m].xor2 - results[0].xor2)
                          + 0.4f * (results[m].xor3 - results[0].xor3);
        float mc_improve = results[m].mc_total - results[0].mc_total;
        float wave_improve = results[m].wave4 - results[0].wave4;
        float narma_improve = results[0].narma_nrmse - results[m].narma_nrmse;  // lower is better

        float score = xor_improve * 3.0f + mc_improve * 1.0f + wave_improve * 2.0f + narma_improve * 2.0f;
        printf("\n  %s: score=%.4f (XOR Δ=%.3f, MC Δ=%.3f, Wave Δ=%.3f, NARMA Δ=%.4f)",
               mechs[m].name, score, xor_improve, mc_improve, wave_improve, narma_improve);
        if (score > best_score) { best_score = score; best_mech = mechs[m].mechanism; }
    }
    printf("\n\n  BEST NEW MECHANISM: %s (score=%.4f)\n\n",
           best_mech == MECH_VGPR ? "VGPR" :
           best_mech == MECH_LDS_BANK ? "LDS_BANK" : "VGPR+LDS_BANK",
           best_score);

    HIP_CHECK(hipFree(d_state));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree((void*)d_xchg));
    free(h_state);

    thermal_guard("Before Phase 3");

    // ═════════════════════════════════════════════════════════════════
    // PHASE 3: FIVEPOP ARCHITECTURE
    // ═════════════════════════════════════════════════════════════════
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("PHASE 3: FIVEPOP Architecture (5 populations × 64 = 320 threads)\n");
    printf("  Pop A: Branch divergence (memory)\n");
    printf("  Pop B: L1 bank conflicts + branch (XOR)\n");
    printf("  Pop C: Dense ESN (regression)\n");
    printf("  Pop D: Wavefront scheduling (classification)\n");
    printf("  Pop E: %s (NEW)\n",
           best_mech == MECH_VGPR ? "VGPR Occupancy" :
           best_mech == MECH_LDS_BANK ? "LDS Bank Patterns" : "VGPR+LDS Bank");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    float *d_state5, *d_input5;
    volatile float *d_xchg5;

    HIP_CHECK(hipMalloc(&d_state5, (long)FIVEPOP_NEURONS * STEPS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_input5, STEPS * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_xchg5, FIVEPOP_NEURONS * sizeof(float)));

    float* h_state5 = (float*)malloc((long)FIVEPOP_NEURONS * STEPS * sizeof(float));

    HIP_CHECK(hipMemcpy(d_input5, h_input, STEPS * sizeof(float), hipMemcpyHostToDevice));

    // Run FIVEPOP with best mechanism
    printf("─── FIVEPOP (all 5 populations, best mechanism) ───\n");
    HIP_CHECK(hipMemset(d_state5, 0, (long)FIVEPOP_NEURONS * STEPS * sizeof(float)));
    HIP_CHECK(hipMemset((void*)d_xchg5, 0, FIVEPOP_NEURONS * sizeof(float)));

    fivepop_reservoir<<<N_CUS, FIVEPOP_THREADS>>>(d_state5, d_xchg5, d_input5, STEPS, best_mech);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_state5, d_state5, (long)FIVEPOP_NEURONS * STEPS * sizeof(float), hipMemcpyDeviceToHost));

    std::vector<std::vector<float>> X5;
    extract_fivepop(h_state5, STEPS, WASHOUT, FIVEPOP_SAMPLE, X5);

    BenchResult r5 = run_benchmark(X5, h_input, STEPS, WASHOUT);
    r5.name = "FIVEPOP";
    print_bench(r5);
    printf("\n");

    thermal_guard("After FIVEPOP full");

    // Run FIVEPOP without Pop E (ablation)
    printf("─── FIVEPOP minus Pop E (ablation — Pop E neurons get pure ESN) ───\n");
    // We can't easily disable Pop E in the kernel, so we just test with MECH_NONE
    // which makes Pop E a plain ESN (no HW mechanism)
    HIP_CHECK(hipMemset(d_state5, 0, (long)FIVEPOP_NEURONS * STEPS * sizeof(float)));
    HIP_CHECK(hipMemset((void*)d_xchg5, 0, FIVEPOP_NEURONS * sizeof(float)));

    fivepop_reservoir<<<N_CUS, FIVEPOP_THREADS>>>(d_state5, d_xchg5, d_input5, STEPS, MECH_NONE);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_state5, d_state5, (long)FIVEPOP_NEURONS * STEPS * sizeof(float), hipMemcpyDeviceToHost));

    std::vector<std::vector<float>> X5_no_e;
    extract_fivepop(h_state5, STEPS, WASHOUT, FIVEPOP_SAMPLE, X5_no_e);

    BenchResult r5_no_e = run_benchmark(X5_no_e, h_input, STEPS, WASHOUT);
    r5_no_e.name = "FIVEPOP_NO_E";
    print_bench(r5_no_e);
    printf("\n");

    // Also run FOURPOP control (256 threads for comparison)
    printf("─── FOURPOP CONTROL (256 threads, pure ESN) ───\n");
    float *d_state4;
    volatile float *d_xchg4;
    int fourpop_neurons = N_CUS * 256;
    HIP_CHECK(hipMalloc(&d_state4, (long)fourpop_neurons * STEPS * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_xchg4, fourpop_neurons * sizeof(float)));
    HIP_CHECK(hipMemset(d_state4, 0, (long)fourpop_neurons * STEPS * sizeof(float)));
    HIP_CHECK(hipMemset((void*)d_xchg4, 0, fourpop_neurons * sizeof(float)));

    control_reservoir<<<N_CUS, 256>>>(d_state4, d_xchg4, d_input5, STEPS);
    HIP_CHECK(hipDeviceSynchronize());

    float* h_state4 = (float*)malloc((long)fourpop_neurons * STEPS * sizeof(float));
    HIP_CHECK(hipMemcpy(h_state4, d_state4, (long)fourpop_neurons * STEPS * sizeof(float), hipMemcpyDeviceToHost));

    std::vector<std::vector<float>> X4;
    extract_state_matrix(h_state4, fourpop_neurons, STEPS, WASHOUT, 512, 256, X4);

    BenchResult r4 = run_benchmark(X4, h_input, STEPS, WASHOUT);
    r4.name = "FOURPOP_CTRL";
    print_bench(r4);

    HIP_CHECK(hipFree(d_state4));
    HIP_CHECK(hipFree((void*)d_xchg4));
    free(h_state4);

    // Final comparison
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("FINAL COMPARISON\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("%-22s  XOR-1   XOR-2   XOR-3   MC_tot  Wave4   NARMA\n", "Architecture");
    printf("─────────────────────────────────────────────────────────────────\n");
    BenchResult* finals[] = {&results[0], &r5, &r5_no_e, &r4};
    const char* fnames[] = {"CONTROL (128t)", "FIVEPOP (320t)", "FIVEPOP_NO_E (320t)", "FOURPOP_CTRL (256t)"};
    for (int i = 0; i < 4; i++) {
        printf("%-22s  %5.1f%%  %5.1f%%  %5.1f%%  %5.3f  %5.1f%%  %.4f\n",
               fnames[i],
               finals[i]->xor1*100, finals[i]->xor2*100, finals[i]->xor3*100,
               finals[i]->mc_total, finals[i]->wave4*100, finals[i]->narma_nrmse);
    }

    // Pop E contribution
    float xor_delta = r5.xor1 - r5_no_e.xor1;
    float mc_delta = r5.mc_total - r5_no_e.mc_total;
    float wave_delta = r5.wave4 - r5_no_e.wave4;
    float narma_delta = r5_no_e.narma_nrmse - r5.narma_nrmse;

    printf("\nPop E contribution (FIVEPOP vs FIVEPOP_NO_E):\n");
    printf("  XOR-1:  %+.1f%%\n", xor_delta * 100);
    printf("  MC:     %+.3f\n", mc_delta);
    printf("  Wave4:  %+.1f%%\n", wave_delta * 100);
    printf("  NARMA:  %+.4f (positive = Pop E helps)\n", narma_delta);

    // Scaling check: 5-pop vs 4-pop control
    float scale_xor = r5.xor1 - r4.xor1;
    float scale_mc = r5.mc_total - r4.mc_total;
    float scale_wave = r5.wave4 - r4.wave4;
    float scale_narma = r4.narma_nrmse - r5.narma_nrmse;

    printf("\nFIVEPOP vs FOURPOP_CTRL (scaling + mechanisms):\n");
    printf("  XOR-1:  %+.1f%%\n", scale_xor * 100);
    printf("  MC:     %+.3f\n", scale_mc);
    printf("  Wave4:  %+.1f%%\n", scale_wave * 100);
    printf("  NARMA:  %+.4f\n", scale_narma);

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("CONCLUSION\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("Best new mechanism: %s\n",
           best_mech == MECH_VGPR ? "VGPR Occupancy Pressure" :
           best_mech == MECH_LDS_BANK ? "LDS Bank Conflict Patterns" : "VGPR+LDS Bank (both)");
    printf("Pop E XOR-1 contribution: %+.1f%%\n", xor_delta * 100);
    printf("Pop E NARMA contribution: %+.4f\n", narma_delta);

    if (r5.xor1 > r5_no_e.xor1 + 0.01f || narma_delta > 0.01f) {
        printf("VERDICT: Pop E mechanism ADDS VALUE to the reservoir\n");
    } else {
        printf("VERDICT: Pop E mechanism does NOT add significant value\n");
    }
    printf("═══════════════════════════════════════════════════════════════\n");

    // Cleanup
    HIP_CHECK(hipFree(d_state5));
    HIP_CHECK(hipFree(d_input5));
    HIP_CHECK(hipFree((void*)d_xchg5));
    free(h_state5);
    free(h_input);

    return 0;
}
