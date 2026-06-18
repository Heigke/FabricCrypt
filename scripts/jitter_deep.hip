/*
 * jitter_deep.hip — Deep jitter analysis: what creates the ~9 clock variations?
 * ===============================================================================
 * The basic probe showed timing is deterministic to ~9 clk. This probe investigates:
 *
 * 1. Multi-CU contention: launch on ALL CUs simultaneously — does contention
 *    create jitter between wavefronts?
 * 2. Memory contention: multiple wavefronts hitting same memory → cache thrashing
 * 3. Thermal drift: does timing change as GPU warms up over millions of iterations?
 * 4. DPM transition capture: force clock changes during measurement
 * 5. Cross-wavefront timing correlation: do different CUs jitter independently?
 * 6. Tail analysis: histogram of timing — is the 9-clk noise uniform or structured?
 * 7. Power-dependent jitter: same work at different DPM levels
 *
 * Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_deep scripts/jitter_deep.hip
 * Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_deep [n_iters]
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <unistd.h>

#define WARP_SIZE 32
#define MAX_ITERS 4096
#define MAX_BLOCKS 20   // one per CU

__device__ __forceinline__ unsigned long long gpu_ts() {
    return (unsigned long long)clock64();
}

// ═══ KERNEL 1: Multi-CU contention ═══
// One block per CU, all doing identical ALU work
// Measures if CU-to-CU contention creates jitter
__global__ void multi_cu_alu(unsigned long long* timestamps, float* sink, int n_iters, int work) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float a = 1.0001f + 0.0001f * (tid + bid * WARP_SIZE);
    float b = 0.9999f;

    for (int i = 0; i < n_iters; i++) {
        unsigned long long t0 = gpu_ts();
        for (int w = 0; w < work; w++) {
            a = fmaf(a, b, 0.0001f);
            b = fmaf(b, a, -0.0001f);
        }
        unsigned long long t1 = gpu_ts();
        timestamps[(bid * n_iters + i) * WARP_SIZE + tid] = t1 - t0;
    }
    sink[bid * WARP_SIZE + tid] = a + b;
}

// ═══ KERNEL 2: Memory contention ═══
// All blocks read from SAME global memory region → L2 thrashing
__global__ void multi_cu_mem(unsigned long long* timestamps, float* sink,
                              float* shared_data, int data_n, int n_iters, int work) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    // All blocks start from same index → maximum L2 contention
    int idx = tid * 137 + bid;

    for (int i = 0; i < n_iters; i++) {
        unsigned long long t0 = gpu_ts();
        for (int w = 0; w < work; w++) {
            idx = (idx * 7919 + 104729) & (data_n - 1);
            sum += shared_data[idx];
        }
        unsigned long long t1 = gpu_ts();
        timestamps[(bid * n_iters + i) * WARP_SIZE + tid] = t1 - t0;
    }
    sink[bid * WARP_SIZE + tid] = sum;
}

// ═══ KERNEL 3: Thermal drift — long-running, record timing every N iterations ═══
__global__ void thermal_drift(unsigned long long* timestamps, float* sink,
                               int total_iters, int sample_every, int work) {
    int tid = threadIdx.x;
    float a = 1.0001f + 0.0001f * tid;
    float b = 0.9999f;
    int sample_idx = 0;

    for (int i = 0; i < total_iters; i++) {
        unsigned long long t0 = gpu_ts();
        for (int w = 0; w < work; w++) {
            a = fmaf(a, b, 0.0001f);
            b = fmaf(b, a, -0.0001f);
        }
        unsigned long long t1 = gpu_ts();

        if (i % sample_every == 0) {
            timestamps[sample_idx * WARP_SIZE + tid] = t1 - t0;
            sample_idx++;
        }
    }
    sink[tid] = a + b;
}

// ═══ KERNEL 4: Mixed workload — ALU+MEM interleaved within wavefront ═══
// Some lanes do ALU, some do memory → intra-wavefront timing divergence
__global__ void mixed_intra_wf(unsigned long long* timestamps, float* sink,
                                float* gmem, int gmem_n, int n_iters, int work) {
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    float a = 1.0001f + 0.0001f * tid;
    float b = 0.9999f;
    int idx = tid * 137;

    for (int i = 0; i < n_iters; i++) {
        unsigned long long t0 = gpu_ts();

        if (lane < 16) {
            // ALU-heavy lanes
            for (int w = 0; w < work; w++) {
                a = fmaf(a, b, 0.0001f);
                b = fmaf(b, a, -0.0001f);
            }
        } else {
            // Memory-heavy lanes
            for (int w = 0; w < work; w++) {
                idx = (idx * 7919 + 104729) & (gmem_n - 1);
                a += gmem[idx];
            }
        }

        unsigned long long t1 = gpu_ts();
        timestamps[i * WARP_SIZE + tid] = t1 - t0;
    }
    sink[tid] = a + b;
}

// ═══ KERNEL 5: Atomic contention — global atomics create true hardware contention ═══
__global__ void atomic_contention(unsigned long long* timestamps, float* sink,
                                   int* counter, int n_iters, int work) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float a = 1.0f;

    for (int i = 0; i < n_iters; i++) {
        unsigned long long t0 = gpu_ts();
        for (int w = 0; w < work; w++) {
            atomicAdd(counter, 1);  // ALL threads contend on same counter
            a = a + 1.0f;
        }
        unsigned long long t1 = gpu_ts();
        timestamps[(bid * n_iters + i) * WARP_SIZE + tid] = t1 - t0;
    }
    sink[bid * WARP_SIZE + tid] = a;
}


// ═══ Host helpers ═══

struct Stats {
    double mean, std, cv, min, max;
    int valid;
};

Stats compute_stats(unsigned long long* data, int n, int stride, int offset) {
    Stats s = {0,0,0,1e30,0,0};
    double sum = 0, sum2 = 0;
    for (int i = 0; i < n; i++) {
        unsigned long long d = data[i * stride + offset];
        if (d > 0 && d < 10000000000ULL) {
            double v = (double)d;
            sum += v; sum2 += v*v;
            if (v < s.min) s.min = v;
            if (v > s.max) s.max = v;
            s.valid++;
        }
    }
    if (s.valid < 2) return s;
    s.mean = sum / s.valid;
    s.std = sqrt(sum2 / s.valid - s.mean * s.mean);
    s.cv = (s.mean > 0) ? s.std / s.mean : 0;
    return s;
}

double pearson(unsigned long long* a_data, unsigned long long* b_data,
               int n, int stride_a, int off_a, int stride_b, int off_b) {
    double sa=0,sb=0,sab=0,sa2=0,sb2=0;
    int valid=0;
    for (int i = 0; i < n; i++) {
        unsigned long long va = a_data[i * stride_a + off_a];
        unsigned long long vb = b_data[i * stride_b + off_b];
        if (va > 0 && va < 10000000000ULL && vb > 0 && vb < 10000000000ULL) {
            double a=(double)va, b=(double)vb;
            sa+=a; sb+=b; sab+=a*b; sa2+=a*a; sb2+=b*b;
            valid++;
        }
    }
    if (valid < 10) return 0;
    double nn=valid;
    double denom = sqrt((nn*sa2-sa*sa)*(nn*sb2-sb*sb));
    return denom > 1e-30 ? (nn*sab - sa*sb)/denom : 0;
}


int main(int argc, char** argv) {
    int n_iters = 2000;
    if (argc > 1) n_iters = atoi(argv[1]);
    if (n_iters > MAX_ITERS) n_iters = MAX_ITERS;

    hipSetDevice(0);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    int n_cus = prop.multiProcessorCount;
    fprintf(stderr, "Device: %s, clock: %d MHz, CUs: %d\n",
            prop.name, prop.clockRate/1000, n_cus);
    fprintf(stderr, "Running %d iterations per test\n\n", n_iters);

    int n_blocks = n_cus;  // one block per CU
    if (n_blocks > MAX_BLOCKS) n_blocks = MAX_BLOCKS;

    // Allocations
    size_t ts_size = n_blocks * n_iters * WARP_SIZE * sizeof(unsigned long long);
    unsigned long long *d_ts, *h_ts;
    float *d_sink, *d_gmem;
    int *d_counter;
    int gmem_n = 1 << 20;

    hipMalloc(&d_ts, ts_size);
    hipMalloc(&d_sink, n_blocks * WARP_SIZE * sizeof(float));
    hipMalloc(&d_gmem, gmem_n * sizeof(float));
    hipMalloc(&d_counter, sizeof(int));
    h_ts = (unsigned long long*)malloc(ts_size);

    float* h_gmem = (float*)malloc(gmem_n * sizeof(float));
    for (int i = 0; i < gmem_n; i++) h_gmem[i] = sinf(i * 0.001f);
    hipMemcpy(d_gmem, h_gmem, gmem_n * sizeof(float), hipMemcpyHostToDevice);
    free(h_gmem);

    printf("test,param,block,iteration,lane,delta_clocks\n");

    // ═══════════════════════════════════════════════════════════
    // TEST 1: Multi-CU ALU contention
    // ═══════════════════════════════════════════════════════════
    fprintf(stderr, "═══ TEST 1: Multi-CU ALU Contention (%d blocks) ═══\n", n_blocks);

    // Single block (no contention)
    hipMemset(d_ts, 0, ts_size);
    hipLaunchKernelGGL(multi_cu_alu, 1, WARP_SIZE, 0, 0, d_ts, d_sink, n_iters, 200);
    hipDeviceSynchronize();
    hipMemcpy(h_ts, d_ts, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);
    Stats s1 = compute_stats(h_ts, n_iters, WARP_SIZE, 0);
    fprintf(stderr, "  1 block:   mean=%8.0f, std=%6.1f, CV=%.6f (%d valid)\n",
            s1.mean, s1.std, s1.cv, s1.valid);

    // All CUs (maximum contention)
    hipMemset(d_ts, 0, ts_size);
    hipLaunchKernelGGL(multi_cu_alu, n_blocks, WARP_SIZE, 0, 0, d_ts, d_sink, n_iters, 200);
    hipDeviceSynchronize();
    hipMemcpy(h_ts, d_ts, ts_size, hipMemcpyDeviceToHost);

    fprintf(stderr, "  %d blocks (per-block stats):\n", n_blocks);
    double total_mean = 0, total_std = 0;
    for (int b = 0; b < n_blocks; b++) {
        Stats sb = compute_stats(h_ts + b * n_iters * WARP_SIZE, n_iters, WARP_SIZE, 0);
        fprintf(stderr, "    block %2d: mean=%8.0f, std=%6.1f, CV=%.6f\n",
                b, sb.mean, sb.std, sb.cv);
        total_mean += sb.mean;
        total_std += sb.std;
        for (int i = 0; i < n_iters; i++)
            printf("multi_cu_alu,%d,%d,%d,0,%llu\n", n_blocks, b, i,
                   h_ts[(b * n_iters + i) * WARP_SIZE]);
    }
    fprintf(stderr, "  Avg across blocks: mean=%.0f, std=%.1f\n",
            total_mean/n_blocks, total_std/n_blocks);

    // Cross-block correlation
    fprintf(stderr, "  Cross-block timing correlation:\n");
    for (int b = 1; b < n_blocks && b < 5; b++) {
        double r = pearson(h_ts, h_ts, n_iters,
                           WARP_SIZE, 0,  // block 0 lane 0
                           WARP_SIZE, b * n_iters * WARP_SIZE);  // block b lane 0
        fprintf(stderr, "    block 0 ↔ block %d: r=%.4f\n", b, r);
    }

    // ═══════════════════════════════════════════════════════════
    // TEST 2: Memory contention — all blocks hit same data
    // ═══════════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ TEST 2: Multi-CU Memory Contention ═══\n");

    hipMemset(d_ts, 0, ts_size);
    hipLaunchKernelGGL(multi_cu_mem, n_blocks, WARP_SIZE, 0, 0,
                       d_ts, d_sink, d_gmem, gmem_n, n_iters, 50);
    hipDeviceSynchronize();
    hipMemcpy(h_ts, d_ts, ts_size, hipMemcpyDeviceToHost);

    for (int b = 0; b < n_blocks; b++) {
        Stats sb = compute_stats(h_ts + b * n_iters * WARP_SIZE, n_iters, WARP_SIZE, 0);
        fprintf(stderr, "  block %2d: mean=%8.0f, std=%6.1f, CV=%.6f\n",
                b, sb.mean, sb.std, sb.cv);
        for (int i = 0; i < n_iters; i++)
            printf("multi_cu_mem,%d,%d,%d,0,%llu\n", n_blocks, b, i,
                   h_ts[(b * n_iters + i) * WARP_SIZE]);
    }

    // ═══════════════════════════════════════════════════════════
    // TEST 3: Thermal drift — long run, sample periodically
    // ═══════════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ TEST 3: Thermal Drift (10M iterations) ═══\n");

    int total_thermal = 10000000;
    int sample_every = 1000;
    int n_samples = total_thermal / sample_every;
    if (n_samples > MAX_ITERS) { sample_every = total_thermal / MAX_ITERS; n_samples = MAX_ITERS; }

    hipMemset(d_ts, 0, n_samples * WARP_SIZE * sizeof(unsigned long long));
    hipLaunchKernelGGL(thermal_drift, 1, WARP_SIZE, 0, 0,
                       d_ts, d_sink, total_thermal, sample_every, 200);
    hipDeviceSynchronize();
    hipMemcpy(h_ts, d_ts, n_samples * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

    // Check for drift: compare first 10% vs last 10%
    int seg = n_samples / 10;
    Stats s_early = compute_stats(h_ts, seg, WARP_SIZE, 0);
    Stats s_late = compute_stats(h_ts + (n_samples - seg) * WARP_SIZE, seg, WARP_SIZE, 0);
    fprintf(stderr, "  Early (first 10%%): mean=%.1f, std=%.1f\n", s_early.mean, s_early.std);
    fprintf(stderr, "  Late  (last  10%%): mean=%.1f, std=%.1f\n", s_late.mean, s_late.std);
    fprintf(stderr, "  Drift: %.2f clk (%.4f%%)\n",
            s_late.mean - s_early.mean,
            100.0 * (s_late.mean - s_early.mean) / (s_early.mean + 1e-30));

    for (int i = 0; i < n_samples; i++)
        printf("thermal,%d,0,%d,0,%llu\n", sample_every, i, h_ts[i * WARP_SIZE]);

    // ═══════════════════════════════════════════════════════════
    // TEST 4: Mixed intra-wavefront (ALU lanes + MEM lanes)
    // ═══════════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ TEST 4: Mixed Intra-Wavefront (ALU vs MEM lanes) ═══\n");

    hipMemset(d_ts, 0, n_iters * WARP_SIZE * sizeof(unsigned long long));
    hipLaunchKernelGGL(mixed_intra_wf, 1, WARP_SIZE, 0, 0,
                       d_ts, d_sink, d_gmem, gmem_n, n_iters, 100);
    hipDeviceSynchronize();
    hipMemcpy(h_ts, d_ts, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

    // Compare ALU lanes (0-15) vs MEM lanes (16-31)
    Stats s_alu = compute_stats(h_ts, n_iters, WARP_SIZE, 0);   // lane 0 (ALU)
    Stats s_mem = compute_stats(h_ts, n_iters, WARP_SIZE, 16);  // lane 16 (MEM)
    fprintf(stderr, "  ALU lane 0:  mean=%.0f, std=%.1f, CV=%.6f\n", s_alu.mean, s_alu.std, s_alu.cv);
    fprintf(stderr, "  MEM lane 16: mean=%.0f, std=%.1f, CV=%.6f\n", s_mem.mean, s_mem.std, s_mem.cv);

    // Cross-type correlation
    double r_cross = pearson(h_ts, h_ts, n_iters, WARP_SIZE, 0, WARP_SIZE, 16);
    fprintf(stderr, "  ALU↔MEM correlation: r=%.4f\n", r_cross);

    // Output all lanes
    for (int i = 0; i < n_iters; i++)
        for (int lane = 0; lane < WARP_SIZE; lane++)
            printf("mixed,%d,0,%d,%d,%llu\n", 100, i, lane, h_ts[i * WARP_SIZE + lane]);

    // ═══════════════════════════════════════════════════════════
    // TEST 5: Atomic contention — true hardware serialization
    // ═══════════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ TEST 5: Atomic Contention (%d blocks) ═══\n", n_blocks);

    hipMemset(d_counter, 0, sizeof(int));
    hipMemset(d_ts, 0, ts_size);
    hipLaunchKernelGGL(atomic_contention, n_blocks, WARP_SIZE, 0, 0,
                       d_ts, d_sink, d_counter, n_iters, 10);
    hipDeviceSynchronize();
    hipMemcpy(h_ts, d_ts, ts_size, hipMemcpyDeviceToHost);

    for (int b = 0; b < n_blocks && b < 5; b++) {
        Stats sb = compute_stats(h_ts + b * n_iters * WARP_SIZE, n_iters, WARP_SIZE, 0);
        fprintf(stderr, "  block %2d: mean=%8.0f, std=%6.1f, CV=%.6f\n",
                b, sb.mean, sb.std, sb.cv);
    }

    // Cross-block correlation for atomics
    if (n_blocks >= 2) {
        double r_atom = pearson(h_ts, h_ts, n_iters,
                                WARP_SIZE, 0,
                                WARP_SIZE, n_iters * WARP_SIZE);
        fprintf(stderr, "  block 0 ↔ block 1 correlation: r=%.4f\n", r_atom);
    }

    for (int b = 0; b < n_blocks; b++)
        for (int i = 0; i < n_iters; i++)
            printf("atomic,%d,%d,%d,0,%llu\n", n_blocks, b, i,
                   h_ts[(b * n_iters + i) * WARP_SIZE]);

    // ═══════════════════════════════════════════════════════════
    // TEST 6: Tail distribution — histogram of timing values
    // ═══════════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ TEST 6: Timing Distribution (ALU work=200) ═══\n");

    hipMemset(d_ts, 0, n_iters * WARP_SIZE * sizeof(unsigned long long));
    hipLaunchKernelGGL(multi_cu_alu, 1, WARP_SIZE, 0, 0, d_ts, d_sink, n_iters, 200);
    hipDeviceSynchronize();
    hipMemcpy(h_ts, d_ts, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

    // Build histogram for lane 0
    int hist[64] = {0};
    unsigned long long base_val = 0;
    int valid_count = 0;
    for (int i = 0; i < n_iters; i++) {
        unsigned long long d = h_ts[i * WARP_SIZE];
        if (d > 0 && d < 10000000000ULL) {
            if (valid_count == 0) base_val = d;
            int bin = (int)(d - base_val + 32);  // center at 32
            if (bin >= 0 && bin < 64) hist[bin]++;
            valid_count++;
        }
    }

    fprintf(stderr, "  Base value: %llu\n", base_val);
    fprintf(stderr, "  Distribution (offset from base):\n");
    for (int b = 0; b < 64; b++) {
        if (hist[b] > 0) {
            fprintf(stderr, "    %+4d: %d", b - 32, hist[b]);
            // ASCII bar
            for (int j = 0; j < hist[b] && j < 50; j++) fprintf(stderr, "█");
            fprintf(stderr, "\n");
        }
    }

    // Cleanup
    hipFree(d_ts);
    hipFree(d_sink);
    hipFree(d_gmem);
    hipFree(d_counter);
    free(h_ts);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
