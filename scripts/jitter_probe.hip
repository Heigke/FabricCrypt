/*
 * jitter_probe.hip — Targeted GPU clock jitter measurement v2
 * ==============================================================
 * Uses __builtin_amdgcn_s_memtime() for cycle-accurate timing on RDNA 3/4.
 * Writes to global memory to prevent compiler optimization of workloads.
 *
 * Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_probe scripts/jitter_probe.hip
 * Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_probe [n_iters]
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define WARP_SIZE 32
#define MAX_ITERS 2000

// clock64() confirmed working on gfx1151 with HSA_OVERRIDE after clean boot
__device__ __forceinline__ unsigned long long gpu_timestamp() {
    return (unsigned long long)clock64();
}

// Anti-optimization: results written to global memory
__global__ void probe_pattern_idle(unsigned long long* timestamps, float* sink, int n_iters) {
    int tid = threadIdx.x;
    float x = 1.0f + 0.001f * tid;

    for (int i = 0; i < n_iters; i++) {
        unsigned long long t0 = gpu_timestamp();

        // Minimal work — just enough to not be optimized out
        x = x + 0.0f;
        asm volatile("");  // prevent full optimization

        unsigned long long t1 = gpu_timestamp();
        timestamps[i * WARP_SIZE + tid] = t1 - t0;
    }
    sink[tid] = x;
}

__global__ void probe_pattern_alu(unsigned long long* timestamps, float* sink, int n_iters, int work) {
    int tid = threadIdx.x;
    float a = 1.0001f + 0.0001f * tid;
    float b = 0.9999f;

    for (int i = 0; i < n_iters; i++) {
        unsigned long long t0 = gpu_timestamp();

        for (int w = 0; w < work; w++) {
            a = fmaf(a, b, 0.0001f);
            b = fmaf(b, a, -0.0001f);
        }

        unsigned long long t1 = gpu_timestamp();
        timestamps[i * WARP_SIZE + tid] = t1 - t0;
    }
    sink[tid] = a + b;
}

__global__ void probe_pattern_memory(unsigned long long* timestamps, float* sink,
                                      float* gmem, int gmem_n, int n_iters, int work) {
    int tid = threadIdx.x;
    float sum = 0.0f;
    int idx = tid * 137;

    for (int i = 0; i < n_iters; i++) {
        unsigned long long t0 = gpu_timestamp();

        for (int w = 0; w < work; w++) {
            idx = (idx * 7919 + 104729) & (gmem_n - 1);
            sum += gmem[idx];
        }

        unsigned long long t1 = gpu_timestamp();
        timestamps[i * WARP_SIZE + tid] = t1 - t0;
    }
    sink[tid] = sum;
}

__global__ void probe_pattern_divergent(unsigned long long* timestamps, float* sink, int n_iters, int work) {
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    float a = 1.0f;

    for (int i = 0; i < n_iters; i++) {
        unsigned long long t0 = gpu_timestamp();

        for (int w = 0; w < work; w++) {
            // 4-way divergence
            if (lane < 8) {
                a = fmaf(a, 1.0001f, 0.0001f);
            } else if (lane < 16) {
                a = a * 1.0001f + 0.0001f;
                a = a * 0.9999f;
            } else if (lane < 24) {
                a = sqrtf(a * a + 0.001f);
            } else {
                a = sinf(a) * 0.9999f + 0.5f;
            }
        }

        unsigned long long t1 = gpu_timestamp();
        timestamps[i * WARP_SIZE + tid] = t1 - t0;
    }
    sink[tid] = a;
}

// Shared memory bank conflict pattern
__global__ void probe_pattern_lds(unsigned long long* timestamps, float* sink, int n_iters, int work) {
    __shared__ float lds[1024];
    int tid = threadIdx.x;

    // Init
    for (int j = tid; j < 1024; j += WARP_SIZE)
        lds[j] = (float)j * 0.001f;
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < n_iters; i++) {
        unsigned long long t0 = gpu_timestamp();

        for (int w = 0; w < work; w++) {
            // stride-32 access = bank conflicts on all lanes
            int addr = (w * 32 + tid) & 1023;
            sum += lds[addr];
            lds[(addr + 1) & 1023] = sum * 0.001f;
        }

        unsigned long long t1 = gpu_timestamp();
        timestamps[i * WARP_SIZE + tid] = t1 - t0;
    }
    sink[tid] = sum;
}

// Cross-lane shuffle pattern — information propagation between lanes
__global__ void probe_pattern_shuffle(unsigned long long* timestamps, float* sink, int n_iters, int work) {
    int tid = threadIdx.x;
    float a = 1.0f + 0.001f * tid;

    for (int i = 0; i < n_iters; i++) {
        unsigned long long t0 = gpu_timestamp();

        for (int w = 0; w < work; w++) {
            float neighbor = __shfl_xor(a, 1);  // exchange with adjacent lane
            a = fmaf(a, 0.5f, neighbor * 0.5f);
            neighbor = __shfl_xor(a, 2);  // exchange with lane+2
            a = fmaf(a, 0.5f, neighbor * 0.5f);
        }

        unsigned long long t1 = gpu_timestamp();
        timestamps[i * WARP_SIZE + tid] = t1 - t0;
    }
    sink[tid] = a;
}

// Varying intensity — sweep work amount within one kernel
__global__ void probe_work_sweep(unsigned long long* timestamps, float* sink,
                                  int n_iters, int* work_schedule) {
    int tid = threadIdx.x;
    float a = 1.0001f + 0.0001f * tid;
    float b = 0.9999f;

    for (int i = 0; i < n_iters; i++) {
        int work = work_schedule[i];

        unsigned long long t0 = gpu_timestamp();

        for (int w = 0; w < work; w++) {
            a = fmaf(a, b, 0.0001f);
            b = fmaf(b, a, -0.0001f);
        }

        unsigned long long t1 = gpu_timestamp();
        timestamps[i * WARP_SIZE + tid] = t1 - t0;
    }
    sink[tid] = a + b;
}


void print_stats(const char* name, unsigned long long* data, int n_iters, int lane) {
    double sum = 0, sum2 = 0;
    unsigned long long mn = (unsigned long long)-1, mx = 0;
    int valid = 0;

    for (int i = 0; i < n_iters; i++) {
        unsigned long long d = data[i * WARP_SIZE + lane];
        if (d > 0 && d < 1000000000ULL) {  // sanity: < 1 billion cycles
            double v = (double)d;
            sum += v;
            sum2 += v * v;
            if (d < mn) mn = d;
            if (d > mx) mx = d;
            valid++;
        }
    }

    if (valid < 10) {
        fprintf(stderr, "  %-18s: %d/%d valid (too few)\n", name, valid, n_iters);
        return;
    }

    double mean = sum / valid;
    double std = sqrt(sum2 / valid - mean * mean);
    double cv = (mean > 0) ? std / mean : 0;

    fprintf(stderr, "  %-18s: mean=%8.0f clk, std=%6.0f, CV=%.6f, range=[%llu,%llu] (%d valid)\n",
            name, mean, std, cv, mn, mx, valid);
}

double compute_correlation(unsigned long long* data, int n, int lane, double* expected) {
    double sx = 0, sy = 0, sxy = 0, sx2 = 0, sy2 = 0;
    int valid = 0;
    for (int i = 0; i < n; i++) {
        unsigned long long d = data[i * WARP_SIZE + lane];
        if (d > 0 && d < 1000000000ULL) {
            double x = expected[i];
            double y = (double)d;
            sx += x; sy += y; sxy += x*y; sx2 += x*x; sy2 += y*y;
            valid++;
        }
    }
    if (valid < 10) return 0.0;
    double n_ = valid;
    double denom = sqrt((n_*sx2 - sx*sx) * (n_*sy2 - sy*sy));
    if (denom < 1e-30) return 0.0;
    return (n_*sxy - sx*sy) / denom;
}

double lane_correlation(unsigned long long* data, int n, int lane_a, int lane_b) {
    double sa = 0, sb = 0, sab = 0, sa2 = 0, sb2 = 0;
    int valid = 0;
    for (int i = 0; i < n; i++) {
        unsigned long long da = data[i * WARP_SIZE + lane_a];
        unsigned long long db = data[i * WARP_SIZE + lane_b];
        if (da > 0 && da < 1000000000ULL && db > 0 && db < 1000000000ULL) {
            double a = (double)da, b = (double)db;
            sa += a; sb += b; sab += a*b; sa2 += a*a; sb2 += b*b;
            valid++;
        }
    }
    if (valid < 10) return 0.0;
    double n_ = valid;
    double denom = sqrt((n_*sa2 - sa*sa) * (n_*sb2 - sb*sb));
    if (denom < 1e-30) return 0.0;
    return (n_*sab - sa*sb) / denom;
}


int main(int argc, char** argv) {
    int n_iters = 1000;
    if (argc > 1) n_iters = atoi(argv[1]);
    if (n_iters < 50) n_iters = 50;
    if (n_iters > MAX_ITERS) n_iters = MAX_ITERS;

    hipSetDevice(0);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    fprintf(stderr, "Device: %s, clock: %d MHz, CUs: %d\n",
            prop.name, prop.clockRate / 1000, prop.multiProcessorCount);
    fprintf(stderr, "Running %d iterations per test\n\n", n_iters);

    // Allocations
    unsigned long long* d_timestamps;
    float* d_sink;
    float* d_gmem;
    int gmem_n = 1 << 20;  // 1M floats, power-of-2 for masking

    hipMalloc(&d_timestamps, n_iters * WARP_SIZE * sizeof(unsigned long long));
    hipMalloc(&d_sink, WARP_SIZE * sizeof(float));
    hipMalloc(&d_gmem, gmem_n * sizeof(float));

    // Fill global memory
    float* h_gmem = (float*)malloc(gmem_n * sizeof(float));
    for (int i = 0; i < gmem_n; i++) h_gmem[i] = sinf(i * 0.001f);
    hipMemcpy(d_gmem, h_gmem, gmem_n * sizeof(float), hipMemcpyHostToDevice);
    free(h_gmem);

    unsigned long long* h_timestamps = (unsigned long long*)malloc(n_iters * WARP_SIZE * sizeof(unsigned long long));

    // CSV header
    printf("test,pattern,work,iteration,lane,delta_clocks\n");

    // ══════════════════════════════════════════════════════════════
    // TEST 1: Pattern fingerprinting — does compute TYPE affect jitter?
    // ══════════════════════════════════════════════════════════════
    fprintf(stderr, "═══ TEST 1: Pattern Fingerprinting ═══\n");

    const char* pattern_names[] = {"idle", "alu", "memory", "divergent", "lds_conflict", "shuffle"};
    int work_amount = 200;

    // Idle
    hipLaunchKernelGGL(probe_pattern_idle, 1, WARP_SIZE, 0, 0,
                       d_timestamps, d_sink, n_iters);
    hipDeviceSynchronize();
    hipMemcpy(h_timestamps, d_timestamps, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);
    print_stats("idle", h_timestamps, n_iters, 0);
    for (int i = 0; i < n_iters; i++)
        printf("pattern,idle,%d,%d,0,%llu\n", 1, i, h_timestamps[i * WARP_SIZE]);

    // ALU
    hipLaunchKernelGGL(probe_pattern_alu, 1, WARP_SIZE, 0, 0,
                       d_timestamps, d_sink, n_iters, work_amount);
    hipDeviceSynchronize();
    hipMemcpy(h_timestamps, d_timestamps, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);
    print_stats("alu", h_timestamps, n_iters, 0);
    for (int i = 0; i < n_iters; i++)
        printf("pattern,alu,%d,%d,0,%llu\n", work_amount, i, h_timestamps[i * WARP_SIZE]);

    // Memory
    hipLaunchKernelGGL(probe_pattern_memory, 1, WARP_SIZE, 0, 0,
                       d_timestamps, d_sink, d_gmem, gmem_n, n_iters, work_amount);
    hipDeviceSynchronize();
    hipMemcpy(h_timestamps, d_timestamps, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);
    print_stats("memory", h_timestamps, n_iters, 0);
    for (int i = 0; i < n_iters; i++)
        printf("pattern,memory,%d,%d,0,%llu\n", work_amount, i, h_timestamps[i * WARP_SIZE]);

    // Divergent
    hipLaunchKernelGGL(probe_pattern_divergent, 1, WARP_SIZE, 0, 0,
                       d_timestamps, d_sink, n_iters, work_amount);
    hipDeviceSynchronize();
    hipMemcpy(h_timestamps, d_timestamps, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);
    print_stats("divergent", h_timestamps, n_iters, 0);
    for (int i = 0; i < n_iters; i++)
        printf("pattern,divergent,%d,%d,0,%llu\n", work_amount, i, h_timestamps[i * WARP_SIZE]);

    // LDS conflict
    hipLaunchKernelGGL(probe_pattern_lds, 1, WARP_SIZE, 0, 0,
                       d_timestamps, d_sink, n_iters, work_amount);
    hipDeviceSynchronize();
    hipMemcpy(h_timestamps, d_timestamps, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);
    print_stats("lds_conflict", h_timestamps, n_iters, 0);
    for (int i = 0; i < n_iters; i++)
        printf("pattern,lds_conflict,%d,%d,0,%llu\n", work_amount, i, h_timestamps[i * WARP_SIZE]);

    // Shuffle
    hipLaunchKernelGGL(probe_pattern_shuffle, 1, WARP_SIZE, 0, 0,
                       d_timestamps, d_sink, n_iters, work_amount);
    hipDeviceSynchronize();
    hipMemcpy(h_timestamps, d_timestamps, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);
    print_stats("shuffle", h_timestamps, n_iters, 0);
    for (int i = 0; i < n_iters; i++)
        printf("pattern,shuffle,%d,%d,0,%llu\n", work_amount, i, h_timestamps[i * WARP_SIZE]);


    // ══════════════════════════════════════════════════════════════
    // TEST 2: Work scaling — does jitter scale with work amount?
    // ══════════════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ TEST 2: Work Scaling (ALU) ═══\n");

    int work_levels[] = {1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000};
    int n_levels = 10;

    for (int wl = 0; wl < n_levels; wl++) {
        hipLaunchKernelGGL(probe_pattern_alu, 1, WARP_SIZE, 0, 0,
                           d_timestamps, d_sink, n_iters, work_levels[wl]);
        hipDeviceSynchronize();
        hipMemcpy(h_timestamps, d_timestamps, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

        char label[64];
        snprintf(label, 64, "alu_work=%d", work_levels[wl]);
        print_stats(label, h_timestamps, n_iters, 0);

        for (int i = 0; i < n_iters; i++)
            printf("scaling,alu,%d,%d,0,%llu\n", work_levels[wl], i, h_timestamps[i * WARP_SIZE]);
    }


    // ══════════════════════════════════════════════════════════════
    // TEST 3: Cross-lane correlation — does jitter propagate between lanes?
    // ══════════════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ TEST 3: Cross-Lane Jitter Correlation ═══\n");

    // Re-run ALU with enough work to see jitter
    hipLaunchKernelGGL(probe_pattern_alu, 1, WARP_SIZE, 0, 0,
                       d_timestamps, d_sink, n_iters, 500);
    hipDeviceSynchronize();
    hipMemcpy(h_timestamps, d_timestamps, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

    fprintf(stderr, "  ALU (work=500), lane 0 vs:\n");
    for (int lane = 1; lane < WARP_SIZE; lane++) {
        double r = lane_correlation(h_timestamps, n_iters, 0, lane);
        fprintf(stderr, "    lane %2d: r=%.4f\n", lane, r);
        printf("crosslane,alu,500,0,%d,%.6f\n", lane, r);
    }

    // Also for shuffle (should have highest cross-lane coupling)
    hipLaunchKernelGGL(probe_pattern_shuffle, 1, WARP_SIZE, 0, 0,
                       d_timestamps, d_sink, n_iters, 500);
    hipDeviceSynchronize();
    hipMemcpy(h_timestamps, d_timestamps, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

    fprintf(stderr, "  Shuffle (work=500), lane 0 vs:\n");
    for (int lane = 1; lane < WARP_SIZE; lane++) {
        double r = lane_correlation(h_timestamps, n_iters, 0, lane);
        fprintf(stderr, "    lane %2d: r=%.4f\n", lane, r);
        printf("crosslane,shuffle,500,0,%d,%.6f\n", lane, r);
    }

    // Output all lanes for shuffle for detailed analysis
    for (int i = 0; i < n_iters; i++) {
        for (int lane = 0; lane < WARP_SIZE; lane++) {
            printf("alllanes,shuffle,500,%d,%d,%llu\n", i, lane,
                   h_timestamps[i * WARP_SIZE + lane]);
        }
    }


    // ══════════════════════════════════════════════════════════════
    // TEST 4: Frequency response — can jitter track oscillating work?
    // ══════════════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ TEST 4: Frequency Response ═══\n");

    int* d_schedule;
    hipMalloc(&d_schedule, n_iters * sizeof(int));
    int* h_schedule = (int*)malloc(n_iters * sizeof(int));
    double* expected = (double*)malloc(n_iters * sizeof(double));

    int periods[] = {5, 10, 20, 50, 100, 200, 500};
    int n_periods = 7;

    for (int p = 0; p < n_periods; p++) {
        // Create oscillating work schedule
        for (int i = 0; i < n_iters; i++) {
            double phase = 6.2831853 * (double)i / (double)periods[p];
            h_schedule[i] = 50 + (int)(450.0 * (0.5 + 0.5 * sin(phase)));  // 50-500
            expected[i] = (double)h_schedule[i];
        }
        hipMemcpy(d_schedule, h_schedule, n_iters * sizeof(int), hipMemcpyHostToDevice);

        hipLaunchKernelGGL(probe_work_sweep, 1, WARP_SIZE, 0, 0,
                           d_timestamps, d_sink, n_iters, d_schedule);
        hipDeviceSynchronize();
        hipMemcpy(h_timestamps, d_timestamps, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

        double r = compute_correlation(h_timestamps, n_iters, 0, expected);
        fprintf(stderr, "  period=%3d: work↔timing r=%.6f\n", periods[p], r);

        for (int i = 0; i < n_iters; i++)
            printf("freq,sweep,%d,%d,0,%llu\n", periods[p], i, h_timestamps[i * WARP_SIZE]);
    }


    // ══════════════════════════════════════════════════════════════
    // TEST 5: Jitter classification — can we identify pattern from timing?
    // ══════════════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ TEST 5: Pattern Classification (from timing alone) ═══\n");

    // Collect 200 samples from each of 4 patterns, interleaved
    int class_iters = 200;
    unsigned long long* class_data = (unsigned long long*)malloc(4 * class_iters * sizeof(unsigned long long));

    // Idle
    hipLaunchKernelGGL(probe_pattern_idle, 1, WARP_SIZE, 0, 0,
                       d_timestamps, d_sink, class_iters);
    hipDeviceSynchronize();
    hipMemcpy(h_timestamps, d_timestamps, class_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);
    for (int i = 0; i < class_iters; i++) class_data[0 * class_iters + i] = h_timestamps[i * WARP_SIZE];

    // ALU
    hipLaunchKernelGGL(probe_pattern_alu, 1, WARP_SIZE, 0, 0,
                       d_timestamps, d_sink, class_iters, 200);
    hipDeviceSynchronize();
    hipMemcpy(h_timestamps, d_timestamps, class_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);
    for (int i = 0; i < class_iters; i++) class_data[1 * class_iters + i] = h_timestamps[i * WARP_SIZE];

    // Memory
    hipLaunchKernelGGL(probe_pattern_memory, 1, WARP_SIZE, 0, 0,
                       d_timestamps, d_sink, d_gmem, gmem_n, class_iters, 200);
    hipDeviceSynchronize();
    hipMemcpy(h_timestamps, d_timestamps, class_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);
    for (int i = 0; i < class_iters; i++) class_data[2 * class_iters + i] = h_timestamps[i * WARP_SIZE];

    // Divergent
    hipLaunchKernelGGL(probe_pattern_divergent, 1, WARP_SIZE, 0, 0,
                       d_timestamps, d_sink, class_iters, 200);
    hipDeviceSynchronize();
    hipMemcpy(h_timestamps, d_timestamps, class_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);
    for (int i = 0; i < class_iters; i++) class_data[3 * class_iters + i] = h_timestamps[i * WARP_SIZE];

    // Print for external classification
    fprintf(stderr, "  Timing means per pattern:\n");
    for (int p = 0; p < 4; p++) {
        const char* names[] = {"idle", "alu", "memory", "divergent"};
        double sum = 0;
        int valid = 0;
        for (int i = 0; i < class_iters; i++) {
            if (class_data[p * class_iters + i] > 0 && class_data[p * class_iters + i] < 1000000000ULL) {
                sum += class_data[p * class_iters + i];
                valid++;
            }
        }
        fprintf(stderr, "    %-12s: mean=%.0f clk (%d valid)\n", names[p], valid > 0 ? sum/valid : 0, valid);

        for (int i = 0; i < class_iters; i++) {
            printf("classify,%s,200,%d,0,%llu\n", names[p], i, class_data[p * class_iters + i]);
        }
    }

    free(class_data);

    // Cleanup
    hipFree(d_timestamps);
    hipFree(d_sink);
    hipFree(d_gmem);
    hipFree(d_schedule);
    free(h_timestamps);
    free(h_schedule);
    free(expected);

    fprintf(stderr, "\nDone. CSV data written to stdout.\n");
    return 0;
}
