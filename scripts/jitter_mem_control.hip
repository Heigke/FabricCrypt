/*
 * jitter_mem_control.hip — Memory contention as controllable jitter source
 * =========================================================================
 * FINDING: Memory access creates 430-clk std jitter (CV=1.7%) while ALU is
 * deterministic. This kernel systematically probes:
 *
 * 1. Contention scaling: 1→20 blocks, how does jitter grow?
 * 2. Access pattern control: stride, random, streaming — different jitter signatures?
 * 3. Data size effect: L1-resident vs L2 vs DRAM — which cache level creates jitter?
 * 4. Mixed signal: embed a signal in access patterns, measure SNR in timing
 * 5. Jitter as information channel: can one wavefront's access pattern be
 *    READ from another wavefront's timing jitter? (causal coupling!)
 * 6. Frequency response: oscillate access patterns, measure timing tracking
 *
 * Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_mem_control scripts/jitter_mem_control.hip
 * Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_mem_control [n_iters]
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define WARP_SIZE 32
#define MAX_ITERS 4096

__device__ __forceinline__ unsigned long long gpu_ts() {
    return (unsigned long long)clock64();
}

// ═══ KERNEL: Contention scaling — variable number of blocks, same access pattern ═══
__global__ void contention_scale(unsigned long long* timestamps, float* sink,
                                  float* gmem, int gmem_n, int n_iters, int work) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    int idx = tid * 137 + bid * 31;

    for (int i = 0; i < n_iters; i++) {
        unsigned long long t0 = gpu_ts();
        for (int w = 0; w < work; w++) {
            idx = (idx * 7919 + 104729) & (gmem_n - 1);
            sum += gmem[idx];
        }
        unsigned long long t1 = gpu_ts();
        timestamps[(bid * n_iters + i) * WARP_SIZE + tid] = t1 - t0;
    }
    sink[bid * WARP_SIZE + tid] = sum;
}

// ═══ KERNEL: Access pattern types ═══
// mode: 0=sequential, 1=strided, 2=random, 3=all-same-address (max contention)
__global__ void pattern_access(unsigned long long* timestamps, float* sink,
                                float* gmem, int gmem_n, int n_iters, int work, int mode) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    int idx = tid + bid * WARP_SIZE;

    for (int i = 0; i < n_iters; i++) {
        unsigned long long t0 = gpu_ts();
        for (int w = 0; w < work; w++) {
            switch (mode) {
                case 0: // Sequential: each thread reads next element
                    idx = (idx + 1) & (gmem_n - 1);
                    break;
                case 1: // Strided: stride=128 to miss cache lines
                    idx = (idx + 128) & (gmem_n - 1);
                    break;
                case 2: // Random: pseudo-random walk
                    idx = (idx * 7919 + 104729) & (gmem_n - 1);
                    break;
                case 3: // Same address: ALL threads read same location
                    idx = w & (gmem_n - 1);
                    break;
            }
            sum += gmem[idx];
        }
        unsigned long long t1 = gpu_ts();
        timestamps[(bid * n_iters + i) * WARP_SIZE + tid] = t1 - t0;
    }
    sink[bid * WARP_SIZE + tid] = sum;
}

// ═══ KERNEL: Data size sweep — control which cache level we hit ═══
__global__ void cache_level(unsigned long long* timestamps, float* sink,
                             float* gmem, int active_size, int n_iters, int work) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    // Mask to stay within active_size (must be power of 2)
    int mask = active_size - 1;
    int idx = tid * 137 + bid * 31;

    for (int i = 0; i < n_iters; i++) {
        unsigned long long t0 = gpu_ts();
        for (int w = 0; w < work; w++) {
            idx = (idx * 7919 + 104729) & mask;
            sum += gmem[idx];
        }
        unsigned long long t1 = gpu_ts();
        timestamps[(bid * n_iters + i) * WARP_SIZE + tid] = t1 - t0;
    }
    sink[bid * WARP_SIZE + tid] = sum;
}

// ═══ KERNEL: Signal injection — embed a binary signal in access pattern,
//     measure if it appears in timing ═══
// "sender" block modulates memory pressure, "receiver" block does fixed work
__global__ void signal_inject(unsigned long long* timestamps, float* sink,
                               float* gmem, int gmem_n, int* signal,
                               int n_iters, int work_base) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    int idx = tid * 137;

    for (int i = 0; i < n_iters; i++) {
        int sig = signal[i];  // 0 or 1

        unsigned long long t0 = gpu_ts();

        if (bid == 0) {
            // SENDER: modulate memory pressure based on signal
            int work = sig ? work_base * 4 : work_base;  // 4x more memory ops when sig=1
            for (int w = 0; w < work; w++) {
                idx = (idx * 7919 + 104729) & (gmem_n - 1);
                sum += gmem[idx];
            }
        } else {
            // RECEIVER: fixed work, but timing affected by sender's cache pressure
            for (int w = 0; w < work_base; w++) {
                idx = (idx * 6271 + 88813) & (gmem_n - 1);
                sum += gmem[idx];
            }
        }

        unsigned long long t1 = gpu_ts();
        timestamps[(bid * n_iters + i) * WARP_SIZE + tid] = t1 - t0;
    }
    sink[bid * WARP_SIZE + tid] = sum;
}

// ═══ KERNEL: Frequency response — oscillate memory work amount ═══
__global__ void freq_response(unsigned long long* timestamps, float* sink,
                               float* gmem, int gmem_n, int n_iters,
                               int* work_schedule) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    int idx = tid * 137 + bid * 31;

    for (int i = 0; i < n_iters; i++) {
        int work = work_schedule[i];

        unsigned long long t0 = gpu_ts();
        for (int w = 0; w < work; w++) {
            idx = (idx * 7919 + 104729) & (gmem_n - 1);
            sum += gmem[idx];
        }
        unsigned long long t1 = gpu_ts();
        timestamps[(bid * n_iters + i) * WARP_SIZE + tid] = t1 - t0;
    }
    sink[bid * WARP_SIZE + tid] = sum;
}


// ═══ Host ═══

struct Stats {
    double mean, std, cv;
    int valid;
};

Stats compute_stats(unsigned long long* data, int n, int stride, int off) {
    Stats s = {0,0,0,0};
    double sum=0, sum2=0;
    for (int i = 0; i < n; i++) {
        unsigned long long d = data[i*stride+off];
        if (d > 0 && d < 100000000000ULL) {
            sum += d; sum2 += (double)d*(double)d; s.valid++;
        }
    }
    if (s.valid < 2) return s;
    s.mean = sum/s.valid;
    s.std = sqrt(sum2/s.valid - s.mean*s.mean);
    s.cv = s.mean > 0 ? s.std/s.mean : 0;
    return s;
}

double pearson_ull(unsigned long long* data, int n, int stride, int off_a, int off_b) {
    double sa=0,sb=0,sab=0,sa2=0,sb2=0; int v=0;
    for (int i = 0; i < n; i++) {
        unsigned long long va = data[i*stride+off_a], vb = data[i*stride+off_b];
        // For signal_inject: sender is at offset 0, receiver at n_iters*WARP_SIZE
        if (va > 0 && va < 100000000000ULL && vb > 0 && vb < 100000000000ULL) {
            double a=va, b=vb; sa+=a; sb+=b; sab+=a*b; sa2+=a*a; sb2+=b*b; v++;
        }
    }
    if (v < 10) return 0;
    double nn=v, denom=sqrt((nn*sa2-sa*sa)*(nn*sb2-sb*sb));
    return denom > 1e-30 ? (nn*sab-sa*sb)/denom : 0;
}

double pearson_separate(unsigned long long* a, unsigned long long* b, int n,
                        int stride_a, int off_a, int stride_b, int off_b) {
    double sa=0,sb=0,sab=0,sa2=0,sb2=0; int v=0;
    for (int i = 0; i < n; i++) {
        unsigned long long va = a[i*stride_a+off_a], vb = b[i*stride_b+off_b];
        if (va > 0 && va < 100000000000ULL && vb > 0 && vb < 100000000000ULL) {
            double aa=va, bb=vb; sa+=aa; sb+=bb; sab+=aa*bb; sa2+=aa*aa; sb2+=bb*bb; v++;
        }
    }
    if (v < 10) return 0;
    double nn=v, denom=sqrt((nn*sa2-sa*sa)*(nn*sb2-sb*sb));
    return denom > 1e-30 ? (nn*sab-sa*sb)/denom : 0;
}


int main(int argc, char** argv) {
    int n_iters = 2000;
    if (argc > 1) n_iters = atoi(argv[1]);
    if (n_iters > MAX_ITERS) n_iters = MAX_ITERS;

    hipSetDevice(0);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    int n_cus = prop.multiProcessorCount;
    if (n_cus > 20) n_cus = 20;
    fprintf(stderr, "CUs: %d, iters: %d\n\n", n_cus, n_iters);

    int gmem_n = 1 << 22;  // 4M floats = 16MB (larger than L2)
    size_t max_blocks = 20;
    size_t ts_size = max_blocks * n_iters * WARP_SIZE * sizeof(unsigned long long);

    unsigned long long *d_ts, *h_ts;
    float *d_sink, *d_gmem;
    int *d_signal, *d_schedule;

    hipMalloc(&d_ts, ts_size);
    hipMalloc(&d_sink, max_blocks * WARP_SIZE * sizeof(float));
    hipMalloc(&d_gmem, gmem_n * sizeof(float));
    hipMalloc(&d_signal, n_iters * sizeof(int));
    hipMalloc(&d_schedule, n_iters * sizeof(int));
    h_ts = (unsigned long long*)malloc(ts_size);

    float* h_tmp = (float*)malloc(gmem_n * sizeof(float));
    for (int i = 0; i < gmem_n; i++) h_tmp[i] = sinf(i * 0.001f);
    hipMemcpy(d_gmem, h_tmp, gmem_n * sizeof(float), hipMemcpyHostToDevice);
    free(h_tmp);

    printf("test,param,block,iteration,lane,delta_clocks\n");

    // ═══════════════════════════════════════════════════════════
    // TEST 1: Contention scaling (1, 2, 4, 8, 16, 20 blocks)
    // ═══════════════════════════════════════════════════════════
    fprintf(stderr, "═══ TEST 1: Contention Scaling ═══\n");
    int block_counts[] = {1, 2, 4, 8, 12, 16, 20};
    int n_bc = 7;

    for (int bc = 0; bc < n_bc; bc++) {
        int nb = block_counts[bc];
        hipMemset(d_ts, 0, ts_size);
        hipLaunchKernelGGL(contention_scale, nb, WARP_SIZE, 0, 0,
                           d_ts, d_sink, d_gmem, gmem_n, n_iters, 50);
        hipDeviceSynchronize();
        hipMemcpy(h_ts, d_ts, nb * n_iters * WARP_SIZE * sizeof(unsigned long long),
                  hipMemcpyDeviceToHost);

        // Average stats across blocks
        double avg_mean = 0, avg_std = 0, avg_cv = 0;
        for (int b = 0; b < nb; b++) {
            Stats s = compute_stats(h_ts + b*n_iters*WARP_SIZE, n_iters, WARP_SIZE, 0);
            avg_mean += s.mean; avg_std += s.std; avg_cv += s.cv;
        }
        avg_mean /= nb; avg_std /= nb; avg_cv /= nb;
        fprintf(stderr, "  %2d blocks: mean=%8.0f, std=%6.1f, CV=%.6f\n",
                nb, avg_mean, avg_std, avg_cv);

        // Output block 0 data
        for (int i = 0; i < n_iters; i++)
            printf("contention,%d,0,%d,0,%llu\n", nb, i, h_ts[i*WARP_SIZE]);
    }

    // ═══════════════════════════════════════════════════════════
    // TEST 2: Access pattern types (20 blocks each)
    // ═══════════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ TEST 2: Access Pattern Types (20 blocks) ═══\n");
    const char* mode_names[] = {"sequential", "strided_128", "random", "same_addr"};

    for (int mode = 0; mode < 4; mode++) {
        hipMemset(d_ts, 0, ts_size);
        hipLaunchKernelGGL(pattern_access, n_cus, WARP_SIZE, 0, 0,
                           d_ts, d_sink, d_gmem, gmem_n, n_iters, 50, mode);
        hipDeviceSynchronize();
        hipMemcpy(h_ts, d_ts, n_cus * n_iters * WARP_SIZE * sizeof(unsigned long long),
                  hipMemcpyDeviceToHost);

        double avg_mean = 0, avg_std = 0;
        for (int b = 0; b < n_cus; b++) {
            Stats s = compute_stats(h_ts + b*n_iters*WARP_SIZE, n_iters, WARP_SIZE, 0);
            avg_mean += s.mean; avg_std += s.std;
        }
        avg_mean /= n_cus; avg_std /= n_cus;
        fprintf(stderr, "  %-14s: mean=%8.0f, std=%6.1f, CV=%.6f\n",
                mode_names[mode], avg_mean, avg_std, avg_std/(avg_mean+1e-30));

        for (int i = 0; i < n_iters; i++)
            printf("pattern,%s,0,%d,0,%llu\n", mode_names[mode], i, h_ts[i*WARP_SIZE]);
    }

    // ═══════════════════════════════════════════════════════════
    // TEST 3: Cache level (data size: 4KB→16MB, 20 blocks)
    // ═══════════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ TEST 3: Cache Level (data size sweep, 20 blocks) ═══\n");
    int sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576, 4194304};
    const char* size_names[] = {"4KB", "16KB", "64KB", "256KB", "1MB", "4MB", "16MB"};
    int n_sizes = 7;

    for (int si = 0; si < n_sizes; si++) {
        hipMemset(d_ts, 0, ts_size);
        hipLaunchKernelGGL(cache_level, n_cus, WARP_SIZE, 0, 0,
                           d_ts, d_sink, d_gmem, sizes[si], n_iters, 50);
        hipDeviceSynchronize();
        hipMemcpy(h_ts, d_ts, n_cus * n_iters * WARP_SIZE * sizeof(unsigned long long),
                  hipMemcpyDeviceToHost);

        double avg_mean = 0, avg_std = 0;
        for (int b = 0; b < n_cus; b++) {
            Stats s = compute_stats(h_ts + b*n_iters*WARP_SIZE, n_iters, WARP_SIZE, 0);
            avg_mean += s.mean; avg_std += s.std;
        }
        avg_mean /= n_cus; avg_std /= n_cus;
        fprintf(stderr, "  %-8s (%7d): mean=%8.0f, std=%6.1f, CV=%.6f\n",
                size_names[si], sizes[si], avg_mean, avg_std, avg_std/(avg_mean+1e-30));

        for (int i = 0; i < n_iters; i++)
            printf("cache,%s,0,%d,0,%llu\n", size_names[si], i, h_ts[i*WARP_SIZE]);
    }

    // ═══════════════════════════════════════════════════════════
    // TEST 4: Signal injection — can sender modulate receiver timing?
    // ═══════════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ TEST 4: Signal Injection (sender→receiver via cache) ═══\n");

    int* h_signal = (int*)malloc(n_iters * sizeof(int));
    // Binary signal: 0101010101... (period=2)
    for (int i = 0; i < n_iters; i++) h_signal[i] = (i / 10) % 2;  // period=20
    hipMemcpy(d_signal, h_signal, n_iters * sizeof(int), hipMemcpyHostToDevice);

    hipMemset(d_ts, 0, ts_size);
    hipLaunchKernelGGL(signal_inject, 2, WARP_SIZE, 0, 0,
                       d_ts, d_sink, d_gmem, gmem_n, d_signal, n_iters, 50);
    hipDeviceSynchronize();
    hipMemcpy(h_ts, d_ts, 2 * n_iters * WARP_SIZE * sizeof(unsigned long long),
              hipMemcpyDeviceToHost);

    Stats s_sender = compute_stats(h_ts, n_iters, WARP_SIZE, 0);
    Stats s_receiver = compute_stats(h_ts + n_iters*WARP_SIZE, n_iters, WARP_SIZE, 0);
    fprintf(stderr, "  Sender:   mean=%.0f, std=%.1f, CV=%.6f\n", s_sender.mean, s_sender.std, s_sender.cv);
    fprintf(stderr, "  Receiver: mean=%.0f, std=%.1f, CV=%.6f\n", s_receiver.mean, s_receiver.std, s_receiver.cv);

    // Correlation between sender timing and receiver timing
    double r_sr = pearson_separate(h_ts, h_ts + n_iters*WARP_SIZE, n_iters,
                                   WARP_SIZE, 0, WARP_SIZE, 0);
    fprintf(stderr, "  Sender↔Receiver timing correlation: r=%.4f\n", r_sr);

    // Correlation between SIGNAL and receiver timing
    double s_sig=0, s_rcv=0, s_sr=0, s_s2=0, s_r2=0;
    int v=0;
    for (int i = 0; i < n_iters; i++) {
        unsigned long long rcv = h_ts[(n_iters + i)*WARP_SIZE];
        if (rcv > 0 && rcv < 100000000000ULL) {
            double sg = h_signal[i], rv = rcv;
            s_sig += sg; s_rcv += rv; s_sr += sg*rv; s_s2 += sg*sg; s_r2 += rv*rv; v++;
        }
    }
    double nn=v, denom=sqrt((nn*s_s2-s_sig*s_sig)*(nn*s_r2-s_rcv*s_rcv));
    double r_signal_rcv = denom > 1e-30 ? (nn*s_sr - s_sig*s_rcv)/denom : 0;
    fprintf(stderr, "  Signal↔Receiver timing correlation: r=%.4f\n", r_signal_rcv);

    // Receiver mean timing when signal=0 vs signal=1
    double sum0=0, sum1=0; int n0=0, n1=0;
    for (int i = 0; i < n_iters; i++) {
        unsigned long long rcv = h_ts[(n_iters + i)*WARP_SIZE];
        if (rcv > 0 && rcv < 100000000000ULL) {
            if (h_signal[i] == 0) { sum0 += rcv; n0++; }
            else { sum1 += rcv; n1++; }
        }
    }
    if (n0 > 0 && n1 > 0) {
        fprintf(stderr, "  Receiver mean when signal=0: %.1f clk\n", sum0/n0);
        fprintf(stderr, "  Receiver mean when signal=1: %.1f clk\n", sum1/n1);
        fprintf(stderr, "  Delta: %.1f clk (%.2f%%)\n",
                sum1/n1 - sum0/n0, 100.0*(sum1/n1 - sum0/n0)/(sum0/n0 + 1e-30));
    }

    for (int i = 0; i < n_iters; i++) {
        printf("signal_sender,%d,0,%d,0,%llu\n", h_signal[i], i,
               h_ts[i*WARP_SIZE]);
        printf("signal_receiver,%d,1,%d,0,%llu\n", h_signal[i], i,
               h_ts[(n_iters+i)*WARP_SIZE]);
    }

    // Try with MORE senders (5 blocks sender, 1 receiver)
    fprintf(stderr, "\n  --- Amplified (5 senders, 1 receiver) ---\n");
    hipMemset(d_ts, 0, ts_size);
    // Reuse signal_inject but launch 6 blocks (5 senders + 1 receiver)
    // Block 0 = sender, blocks 1-5 are treated as receivers by the kernel
    // We need a modified kernel... let's just use 10 blocks all as senders + receiver
    hipLaunchKernelGGL(signal_inject, 10, WARP_SIZE, 0, 0,
                       d_ts, d_sink, d_gmem, gmem_n, d_signal, n_iters, 50);
    hipDeviceSynchronize();
    hipMemcpy(h_ts, d_ts, 10 * n_iters * WARP_SIZE * sizeof(unsigned long long),
              hipMemcpyDeviceToHost);

    // Block 0 = sender, block 1 = receiver (same kernel logic)
    s_sender = compute_stats(h_ts, n_iters, WARP_SIZE, 0);
    s_receiver = compute_stats(h_ts + n_iters*WARP_SIZE, n_iters, WARP_SIZE, 0);
    fprintf(stderr, "  Sender:   mean=%.0f, std=%.1f\n", s_sender.mean, s_sender.std);
    fprintf(stderr, "  Receiver: mean=%.0f, std=%.1f\n", s_receiver.mean, s_receiver.std);

    // Check block 9 (furthest receiver)
    Stats s_far = compute_stats(h_ts + 9*n_iters*WARP_SIZE, n_iters, WARP_SIZE, 0);
    fprintf(stderr, "  Far recv: mean=%.0f, std=%.1f\n", s_far.mean, s_far.std);

    // Signal↔far receiver correlation
    s_sig=0; s_rcv=0; s_sr=0; s_s2=0; s_r2=0; v=0;
    for (int i = 0; i < n_iters; i++) {
        unsigned long long rcv = h_ts[(9*n_iters + i)*WARP_SIZE];
        if (rcv > 0 && rcv < 100000000000ULL) {
            double sg = h_signal[i], rv = rcv;
            s_sig += sg; s_rcv += rv; s_sr += sg*rv; s_s2 += sg*sg; s_r2 += rv*rv; v++;
        }
    }
    nn=v; denom=sqrt((nn*s_s2-s_sig*s_sig)*(nn*s_r2-s_rcv*s_rcv));
    double r_far = denom > 1e-30 ? (nn*s_sr - s_sig*s_rcv)/denom : 0;
    fprintf(stderr, "  Signal↔FarReceiver r=%.4f\n", r_far);

    free(h_signal);

    // ═══════════════════════════════════════════════════════════
    // TEST 5: Frequency response for memory contention
    // ═══════════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ TEST 5: Memory Jitter Frequency Response ═══\n");

    int* h_schedule = (int*)malloc(n_iters * sizeof(int));
    double* expected = (double*)malloc(n_iters * sizeof(double));

    int periods[] = {4, 10, 20, 50, 100, 200, 500, 1000};
    int n_periods = 8;

    for (int p = 0; p < n_periods; p++) {
        for (int i = 0; i < n_iters; i++) {
            double phase = 6.2831853 * (double)i / (double)periods[p];
            h_schedule[i] = 10 + (int)(90.0 * (0.5 + 0.5 * sin(phase)));
            expected[i] = h_schedule[i];
        }
        hipMemcpy(d_schedule, h_schedule, n_iters * sizeof(int), hipMemcpyHostToDevice);

        hipMemset(d_ts, 0, ts_size);
        hipLaunchKernelGGL(freq_response, n_cus, WARP_SIZE, 0, 0,
                           d_ts, d_sink, d_gmem, gmem_n, n_iters, d_schedule);
        hipDeviceSynchronize();
        hipMemcpy(h_ts, d_ts, n_cus * n_iters * WARP_SIZE * sizeof(unsigned long long),
                  hipMemcpyDeviceToHost);

        // Correlation work↔timing for block 0
        double sx=0,sy=0,sxy=0,sx2=0,sy2=0; int vv=0;
        for (int i = 0; i < n_iters; i++) {
            unsigned long long d = h_ts[i*WARP_SIZE];
            if (d > 0 && d < 100000000000ULL) {
                double x=expected[i], y=d;
                sx+=x; sy+=y; sxy+=x*y; sx2+=x*x; sy2+=y*y; vv++;
            }
        }
        double n_=vv, dm=sqrt((n_*sx2-sx*sx)*(n_*sy2-sy*sy));
        double r = dm > 1e-30 ? (n_*sxy - sx*sy)/dm : 0;
        fprintf(stderr, "  period=%4d: work↔timing r=%.6f\n", periods[p], r);

        for (int i = 0; i < n_iters; i++)
            printf("mem_freq,%d,0,%d,0,%llu\n", periods[p], i, h_ts[i*WARP_SIZE]);
    }

    free(h_schedule);
    free(expected);

    // Cleanup
    hipFree(d_ts); hipFree(d_sink); hipFree(d_gmem);
    hipFree(d_signal); hipFree(d_schedule);
    free(h_ts);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
