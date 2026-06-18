/*
 * jitter_coupling.hip — Strengthening cross-wavefront causal coupling
 * ====================================================================
 * Previous finding: generic cache pressure gives r=0.088, delta=-0.6%.
 *
 * This kernel tests TARGETED coupling mechanisms:
 *
 * TEST 1: Targeted L2 eviction — sender evicts receiver's exact cache lines
 * TEST 2: Atomic coherence traffic — sender does atomics on receiver's data
 * TEST 3: LDS coupling — two wavefronts in same workgroup, shared LDS
 * TEST 4: Memory fence coupling — sender uses __threadfence, receiver measures
 * TEST 5: Graduated pressure — vary sender intensity, measure receiver response
 * TEST 6: Bidirectional ping-pong — alternating sender/receiver roles
 * TEST 7: Multi-sender focused — N senders all evict receiver's working set
 *
 * Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_coupling scripts/jitter_coupling.hip
 * Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_coupling [n_iters]
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define WARP_SIZE 32
#define MAX_ITERS 4096
// Receiver's working set: 4KB = 64 cache lines of 64 bytes = 1024 floats
#define RECV_WS_SIZE 1024
// Eviction set: 256KB = enough to flush L1 (32KB) and pressure L2
#define EVICT_SIZE (256*1024/4)

__device__ __forceinline__ unsigned long long gpu_ts() {
    return (unsigned long long)clock64();
}

// ═══ TEST 1: Targeted L2 eviction ═══
// Receiver repeatedly reads its small working set (fits in L1)
// Sender either: signal=0 → reads own disjoint region, signal=1 → reads receiver's region to evict it
__global__ void targeted_eviction(unsigned long long* recv_ts, unsigned long long* send_ts,
                                   float* recv_data, float* evict_data, float* sink,
                                   const int* signal, int n_iters, int recv_work) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;

    if (bid == 0) {
        // RECEIVER: repeatedly read small working set
        for (int i = 0; i < n_iters; i++) {
            unsigned long long t0 = gpu_ts();
            for (int w = 0; w < recv_work; w++) {
                int idx = (tid + w * WARP_SIZE) % RECV_WS_SIZE;
                sum += recv_data[idx];
            }
            unsigned long long t1 = gpu_ts();
            recv_ts[i * WARP_SIZE + tid] = t1 - t0;
        }
    } else {
        // SENDER: either access disjoint data or receiver's data
        for (int i = 0; i < n_iters; i++) {
            unsigned long long t0 = gpu_ts();
            if (signal[i] == 0) {
                // Disjoint access — don't touch receiver's cache lines
                for (int w = 0; w < recv_work; w++) {
                    int idx = (tid + w * WARP_SIZE + bid * 8192) % EVICT_SIZE;
                    sum += evict_data[idx];
                }
            } else {
                // EVICT receiver's data — read same cache lines
                for (int w = 0; w < recv_work; w++) {
                    int idx = (tid + w * WARP_SIZE) % RECV_WS_SIZE;
                    sum += recv_data[idx];  // same addresses as receiver!
                }
            }
            unsigned long long t1 = gpu_ts();
            send_ts[((bid-1) * n_iters + i) * WARP_SIZE + tid] = t1 - t0;
        }
    }
    sink[bid * WARP_SIZE + tid] = sum;
}

// ═══ TEST 2: Atomic coherence coupling ═══
// Sender does atomic adds on shared data; receiver reads same data
// Atomics force cache coherency protocol traffic → measurable latency impact
__global__ void atomic_coupling(unsigned long long* recv_ts, unsigned long long* send_ts,
                                 int* shared_data, float* sink,
                                 const int* signal, int n_iters, int work) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;

    if (bid == 0) {
        // RECEIVER: read shared data
        for (int i = 0; i < n_iters; i++) {
            unsigned long long t0 = gpu_ts();
            for (int w = 0; w < work; w++) {
                int idx = (tid + w * WARP_SIZE) % RECV_WS_SIZE;
                sum += (float)shared_data[idx];
            }
            unsigned long long t1 = gpu_ts();
            recv_ts[i * WARP_SIZE + tid] = t1 - t0;
        }
    } else {
        // SENDER: atomic ops on shared data when signal=1
        for (int i = 0; i < n_iters; i++) {
            unsigned long long t0 = gpu_ts();
            if (signal[i] == 1) {
                for (int w = 0; w < work; w++) {
                    int idx = (tid + w * WARP_SIZE) % RECV_WS_SIZE;
                    atomicAdd(&shared_data[idx], 1);
                }
            } else {
                // Idle work — just ALU
                float x = (float)tid;
                for (int w = 0; w < work * 4; w++) {
                    x = x * 1.0001f + 0.001f;
                }
                sum += x;
            }
            unsigned long long t1 = gpu_ts();
            send_ts[((bid-1) * n_iters + i) * WARP_SIZE + tid] = t1 - t0;
        }
    }
    sink[bid * WARP_SIZE + tid] = sum;
}

// ═══ TEST 3: LDS coupling (intra-workgroup) ═══
// Two wavefronts in the SAME workgroup share LDS
// Sender writes to LDS; receiver reads from LDS — intra-CU coupling
__global__ void lds_coupling(unsigned long long* wave0_ts, unsigned long long* wave1_ts,
                              float* sink, const int* signal, int n_iters, int work) {
    __shared__ float lds_data[1024];
    int warp_id = threadIdx.x / WARP_SIZE;  // 0 or 1
    int lane = threadIdx.x % WARP_SIZE;

    // Initialize LDS
    for (int i = threadIdx.x; i < 1024; i += blockDim.x)
        lds_data[i] = 1.0f;
    __syncthreads();

    float sum = 0.0f;

    for (int i = 0; i < n_iters; i++) {
        __syncthreads();  // sync before each iteration
        unsigned long long t0 = gpu_ts();

        if (warp_id == 0) {
            // WAVEFRONT 0 (sender): write to LDS when signal=1
            if (signal[i] == 1) {
                for (int w = 0; w < work; w++) {
                    int idx = (lane + w * WARP_SIZE) % 1024;
                    lds_data[idx] = (float)(i * 17 + w);
                }
            } else {
                // Pure ALU — no LDS access
                float x = (float)lane;
                for (int w = 0; w < work; w++) {
                    x = x * 1.0001f + 0.001f;
                }
                sum += x;
            }
        } else {
            // WAVEFRONT 1 (receiver): always reads LDS
            for (int w = 0; w < work; w++) {
                int idx = (lane + w * WARP_SIZE) % 1024;
                sum += lds_data[idx];
            }
        }

        unsigned long long t1 = gpu_ts();
        if (warp_id == 0) {
            wave0_ts[i * WARP_SIZE + lane] = t1 - t0;
        } else {
            wave1_ts[i * WARP_SIZE + lane] = t1 - t0;
        }
    }
    sink[threadIdx.x] = sum;
}

// ═══ TEST 5: Graduated pressure ═══
// Sender varies the NUMBER of cache lines accessed (1, 4, 16, 64, 256, 1024)
// Receiver measures timing impact — should see linear/log response
__global__ void graduated_pressure(unsigned long long* recv_ts, unsigned long long* send_ts,
                                    float* recv_data, float* evict_data, float* sink,
                                    int n_iters, int pressure_lines) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;

    if (bid == 0) {
        // RECEIVER
        for (int i = 0; i < n_iters; i++) {
            unsigned long long t0 = gpu_ts();
            for (int w = 0; w < 64; w++) {
                int idx = (tid + w * WARP_SIZE) % RECV_WS_SIZE;
                sum += recv_data[idx];
            }
            unsigned long long t1 = gpu_ts();
            recv_ts[i * WARP_SIZE + tid] = t1 - t0;
        }
    } else {
        // SENDER: access pressure_lines cache lines of receiver's data
        for (int i = 0; i < n_iters; i++) {
            unsigned long long t0 = gpu_ts();
            int lines_to_touch = pressure_lines;
            for (int w = 0; w < lines_to_touch; w++) {
                // Each cache line = 16 floats. Access first float of each line
                int idx = (w * 16 + tid) % RECV_WS_SIZE;
                sum += recv_data[idx];
            }
            // Fill rest of work with disjoint access
            for (int w = lines_to_touch; w < 64; w++) {
                int idx = (tid + w * WARP_SIZE + bid * 8192) % EVICT_SIZE;
                sum += evict_data[idx];
            }
            unsigned long long t1 = gpu_ts();
            send_ts[((bid-1) * n_iters + i) * WARP_SIZE + tid] = t1 - t0;
        }
    }
    sink[bid * WARP_SIZE + tid] = sum;
}

// ═══ TEST 7: Multi-sender focused eviction ═══
// Block 0 = receiver, blocks 1..N = senders all evicting receiver's data
__global__ void multi_sender_evict(unsigned long long* recv_ts, float* recv_data,
                                    float* sink, const int* signal, int n_iters, int work) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;

    if (bid == 0) {
        for (int i = 0; i < n_iters; i++) {
            unsigned long long t0 = gpu_ts();
            for (int w = 0; w < work; w++) {
                int idx = (tid + w * WARP_SIZE) % RECV_WS_SIZE;
                sum += recv_data[idx];
            }
            unsigned long long t1 = gpu_ts();
            recv_ts[i * WARP_SIZE + tid] = t1 - t0;
        }
    } else {
        for (int i = 0; i < n_iters; i++) {
            if (signal[i] == 1) {
                // ALL senders read receiver's data → massive eviction pressure
                for (int w = 0; w < work; w++) {
                    int idx = (tid + w * WARP_SIZE + bid * 7) % RECV_WS_SIZE;
                    sum += recv_data[idx];
                }
            } else {
                float x = (float)tid;
                for (int w = 0; w < work * 4; w++) x = x * 1.0001f + 0.001f;
                sum += x;
            }
        }
    }
    sink[bid * WARP_SIZE + tid] = sum;
}

// ═══════════════════════════════════════════════
// HOST
// ═══════════════════════════════════════════════

double pearson(unsigned long long* x_raw, unsigned long long* y_raw, int n, int x_stride, int y_stride) {
    double sx = 0, sy = 0, sxy = 0, sx2 = 0, sy2 = 0;
    int valid = 0;
    for (int i = 5; i < n; i++) {  // skip warmup
        double xv = (double)x_raw[i * x_stride];
        double yv = (double)y_raw[i * y_stride];
        if (xv > 0 && xv < 1e9 && yv > 0 && yv < 1e9) {
            sx += xv; sy += yv;
            sxy += xv * yv;
            sx2 += xv * xv; sy2 += yv * yv;
            valid++;
        }
    }
    if (valid < 10) return 0;
    double num = valid * sxy - sx * sy;
    double den = sqrt((valid * sx2 - sx * sx) * (valid * sy2 - sy * sy));
    return (den > 0) ? num / den : 0;
}

double sig_corr(int* signal, unsigned long long* ts, int n, int ts_stride) {
    double sx = 0, sy = 0, sxy = 0, sx2 = 0, sy2 = 0;
    int valid = 0;
    for (int i = 5; i < n; i++) {
        double xv = (double)signal[i];
        double yv = (double)ts[i * ts_stride];
        if (yv > 0 && yv < 1e9) {
            sx += xv; sy += yv;
            sxy += xv * yv;
            sx2 += xv * xv; sy2 += yv * yv;
            valid++;
        }
    }
    if (valid < 10) return 0;
    double num = valid * sxy - sx * sy;
    double den = sqrt((valid * sx2 - sx * sx) * (valid * sy2 - sy * sy));
    return (den > 0) ? num / den : 0;
}

void compute_means(int* signal, unsigned long long* ts, int n, int stride,
                    double* mean0, double* mean1, int* n0, int* n1) {
    *mean0 = *mean1 = 0; *n0 = *n1 = 0;
    for (int i = 5; i < n; i++) {
        double v = (double)ts[i * stride];
        if (v > 0 && v < 1e9) {
            if (signal[i] == 0) { *mean0 += v; (*n0)++; }
            else { *mean1 += v; (*n1)++; }
        }
    }
    if (*n0 > 0) *mean0 /= *n0;
    if (*n1 > 0) *mean1 /= *n1;
}

int main(int argc, char** argv) {
    int n_iters = 2000;
    if (argc > 1) n_iters = atoi(argv[1]);
    if (n_iters > MAX_ITERS) n_iters = MAX_ITERS;

    // Create signal pattern: alternating blocks of 50
    int* h_signal = new int[n_iters];
    for (int i = 0; i < n_iters; i++) h_signal[i] = ((i / 50) % 2);
    int* d_signal;
    hipMalloc(&d_signal, n_iters * sizeof(int));
    hipMemcpy(d_signal, h_signal, n_iters * sizeof(int), hipMemcpyHostToDevice);

    // Allocate memory
    float *d_recv_data, *d_evict_data, *d_sink;
    int *d_shared_data;
    unsigned long long *d_recv_ts, *d_send_ts, *d_wave0_ts, *d_wave1_ts;

    hipMalloc(&d_recv_data, RECV_WS_SIZE * sizeof(float));
    hipMalloc(&d_evict_data, EVICT_SIZE * sizeof(float));
    hipMalloc(&d_sink, 20 * WARP_SIZE * sizeof(float));
    hipMalloc(&d_shared_data, RECV_WS_SIZE * sizeof(int));
    hipMalloc(&d_recv_ts, n_iters * WARP_SIZE * sizeof(unsigned long long));
    hipMalloc(&d_send_ts, 20 * n_iters * WARP_SIZE * sizeof(unsigned long long));
    hipMalloc(&d_wave0_ts, n_iters * WARP_SIZE * sizeof(unsigned long long));
    hipMalloc(&d_wave1_ts, n_iters * WARP_SIZE * sizeof(unsigned long long));

    // Init data
    hipMemset(d_recv_data, 0x3F, RECV_WS_SIZE * sizeof(float));  // ~1.0f pattern
    hipMemset(d_evict_data, 0, EVICT_SIZE * sizeof(float));
    hipMemset(d_shared_data, 0, RECV_WS_SIZE * sizeof(int));

    unsigned long long* h_recv = new unsigned long long[n_iters * WARP_SIZE];
    unsigned long long* h_send = new unsigned long long[20 * n_iters * WARP_SIZE];
    unsigned long long* h_wave0 = new unsigned long long[n_iters * WARP_SIZE];
    unsigned long long* h_wave1 = new unsigned long long[n_iters * WARP_SIZE];

    fprintf(stderr, "═══ GPU Cross-Wavefront Coupling Amplification ═══\n");
    fprintf(stderr, "Iters: %d, signal: 50-sample blocks\n\n", n_iters);

    printf("test,sender_count,iter,recv_clk,send_clk,signal\n");

    // ═══ TEST 1: Targeted L2 eviction (1 sender) ═══
    fprintf(stderr, "═══ TEST 1: Targeted L2 Eviction ═══\n");
    for (int n_senders = 1; n_senders <= 16; n_senders *= 2) {
        hipMemset(d_recv_ts, 0, n_iters * WARP_SIZE * sizeof(unsigned long long));
        hipMemset(d_send_ts, 0, (n_senders+1) * n_iters * WARP_SIZE * sizeof(unsigned long long));

        targeted_eviction<<<n_senders + 1, WARP_SIZE>>>(
            d_recv_ts, d_send_ts, d_recv_data, d_evict_data, d_sink,
            d_signal, n_iters, 64);
        hipDeviceSynchronize();

        hipMemcpy(h_recv, d_recv_ts, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

        double m0, m1; int n0, n1;
        compute_means(h_signal, h_recv, n_iters, WARP_SIZE, &m0, &m1, &n0, &n1);
        double r = sig_corr(h_signal, h_recv, n_iters, WARP_SIZE);
        double delta = m1 - m0;
        double pct = m0 > 0 ? 100.0 * delta / m0 : 0;

        // Classification accuracy
        double thresh = (m0 + m1) / 2.0;
        int correct = 0, total = 0;
        for (int i = 5; i < n_iters; i++) {
            double v = (double)h_recv[i * WARP_SIZE];
            if (v > 0 && v < 1e9) {
                int pred = (v > thresh) ? 1 : 0;
                if (pred == h_signal[i]) correct++;
                total++;
            }
        }
        double acc = total > 0 ? 100.0 * correct / total : 0;

        fprintf(stderr, "  %2d sender(s): recv_mean0=%.0f, recv_mean1=%.0f, delta=%+.0f (%+.1f%%), r=%.4f, acc=%.1f%%\n",
                n_senders, m0, m1, delta, pct, r, acc);

        // CSV for 1-sender case
        if (n_senders == 1) {
            for (int i = 0; i < n_iters; i++) {
                printf("targeted_evict,%d,%d,%llu,0,%d\n", n_senders, i, h_recv[i*WARP_SIZE], h_signal[i]);
            }
        }
    }

    // ═══ TEST 2: Atomic coherence coupling ═══
    fprintf(stderr, "\n═══ TEST 2: Atomic Coherence Coupling ═══\n");
    for (int n_senders = 1; n_senders <= 16; n_senders *= 2) {
        hipMemset(d_recv_ts, 0, n_iters * WARP_SIZE * sizeof(unsigned long long));
        hipMemset(d_shared_data, 0, RECV_WS_SIZE * sizeof(int));

        atomic_coupling<<<n_senders + 1, WARP_SIZE>>>(
            d_recv_ts, d_send_ts, d_shared_data, d_sink,
            d_signal, n_iters, 64);
        hipDeviceSynchronize();

        hipMemcpy(h_recv, d_recv_ts, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

        double m0, m1; int n0, n1;
        compute_means(h_signal, h_recv, n_iters, WARP_SIZE, &m0, &m1, &n0, &n1);
        double r = sig_corr(h_signal, h_recv, n_iters, WARP_SIZE);
        double delta = m1 - m0;
        double pct = m0 > 0 ? 100.0 * delta / m0 : 0;

        double thresh = (m0 + m1) / 2.0;
        int correct = 0, total = 0;
        for (int i = 5; i < n_iters; i++) {
            double v = (double)h_recv[i * WARP_SIZE];
            if (v > 0 && v < 1e9) {
                int pred = (v > thresh) ? 1 : 0;
                if (pred == h_signal[i]) correct++;
                total++;
            }
        }
        double acc = total > 0 ? 100.0 * correct / total : 0;

        fprintf(stderr, "  %2d sender(s): recv_mean0=%.0f, recv_mean1=%.0f, delta=%+.0f (%+.1f%%), r=%.4f, acc=%.1f%%\n",
                n_senders, m0, m1, delta, pct, r, acc);
    }

    // ═══ TEST 3: LDS coupling (intra-CU) ═══
    fprintf(stderr, "\n═══ TEST 3: LDS Coupling (Intra-CU, 2 wavefronts) ═══\n");
    hipMemset(d_wave0_ts, 0, n_iters * WARP_SIZE * sizeof(unsigned long long));
    hipMemset(d_wave1_ts, 0, n_iters * WARP_SIZE * sizeof(unsigned long long));

    // 2 warps in 1 block = same CU, shared LDS
    lds_coupling<<<1, 2 * WARP_SIZE>>>(d_wave0_ts, d_wave1_ts, d_sink, d_signal, n_iters, 64);
    hipDeviceSynchronize();

    hipMemcpy(h_wave0, d_wave0_ts, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);
    hipMemcpy(h_wave1, d_wave1_ts, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

    {
        double m0, m1; int n0, n1;
        compute_means(h_signal, h_wave1, n_iters, WARP_SIZE, &m0, &m1, &n0, &n1);
        double r = sig_corr(h_signal, h_wave1, n_iters, WARP_SIZE);
        double wave_r = pearson(h_wave0, h_wave1, n_iters, WARP_SIZE, WARP_SIZE);
        double delta = m1 - m0;
        double pct = m0 > 0 ? 100.0 * delta / m0 : 0;

        double thresh = (m0 + m1) / 2.0;
        int correct = 0, total = 0;
        for (int i = 5; i < n_iters; i++) {
            double v = (double)h_wave1[i * WARP_SIZE];
            if (v > 0 && v < 1e9) {
                int pred = (v > thresh) ? 1 : 0;
                if (pred == h_signal[i]) correct++;
                total++;
            }
        }
        double acc = total > 0 ? 100.0 * correct / total : 0;

        fprintf(stderr, "  Receiver(wave1): mean0=%.0f, mean1=%.0f, delta=%+.0f (%+.1f%%), r=%.4f, acc=%.1f%%\n",
                m0, m1, delta, pct, r, acc);
        fprintf(stderr, "  Wave0↔Wave1 timing corr: r=%.4f\n", wave_r);

        for (int i = 0; i < n_iters; i++) {
            printf("lds,%d,%d,%llu,%llu,%d\n", 1, i, h_wave1[i*WARP_SIZE], h_wave0[i*WARP_SIZE], h_signal[i]);
        }
    }

    // ═══ TEST 5: Graduated pressure ═══
    fprintf(stderr, "\n═══ TEST 5: Graduated Pressure (receiver response curve) ═══\n");
    int pressures[] = {0, 1, 4, 8, 16, 32, 64};
    for (int p = 0; p < 7; p++) {
        hipMemset(d_recv_ts, 0, n_iters * WARP_SIZE * sizeof(unsigned long long));

        graduated_pressure<<<2, WARP_SIZE>>>(
            d_recv_ts, d_send_ts, d_recv_data, d_evict_data, d_sink,
            n_iters, pressures[p]);
        hipDeviceSynchronize();

        hipMemcpy(h_recv, d_recv_ts, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

        double sum = 0, sum2 = 0;
        int cnt = 0;
        for (int i = 5; i < n_iters; i++) {
            double v = (double)h_recv[i * WARP_SIZE];
            if (v > 0 && v < 1e9) { sum += v; sum2 += v*v; cnt++; }
        }
        double mean = cnt > 0 ? sum/cnt : 0;
        double std_v = cnt > 1 ? sqrt(sum2/cnt - mean*mean) : 0;

        fprintf(stderr, "  %3d lines: recv_mean=%8.0f, std=%7.1f\n", pressures[p], mean, std_v);
    }

    // ═══ TEST 7: Multi-sender focused eviction ═══
    fprintf(stderr, "\n═══ TEST 7: Multi-Sender Focused Eviction ═══\n");
    for (int n_senders = 1; n_senders <= 19; n_senders += 2) {
        hipMemset(d_recv_ts, 0, n_iters * WARP_SIZE * sizeof(unsigned long long));

        multi_sender_evict<<<n_senders + 1, WARP_SIZE>>>(
            d_recv_ts, d_recv_data, d_sink, d_signal, n_iters, 64);
        hipDeviceSynchronize();

        hipMemcpy(h_recv, d_recv_ts, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

        double m0, m1; int n0, n1;
        compute_means(h_signal, h_recv, n_iters, WARP_SIZE, &m0, &m1, &n0, &n1);
        double r = sig_corr(h_signal, h_recv, n_iters, WARP_SIZE);
        double delta = m1 - m0;
        double pct = m0 > 0 ? 100.0 * delta / m0 : 0;

        double thresh = (m0 + m1) / 2.0;
        int correct = 0, total = 0;
        for (int i = 5; i < n_iters; i++) {
            double v = (double)h_recv[i * WARP_SIZE];
            if (v > 0 && v < 1e9) {
                int pred = (v > thresh) ? 1 : 0;
                if (pred == h_signal[i]) correct++;
                total++;
            }
        }
        double acc = total > 0 ? 100.0 * correct / total : 0;

        fprintf(stderr, "  %2d sender(s): recv_mean0=%8.0f, mean1=%8.0f, delta=%+6.0f (%+5.1f%%), r=%.4f, acc=%.1f%%\n",
                n_senders, m0, m1, delta, pct, r, acc);
    }

    fprintf(stderr, "\nDone.\n");

    // Cleanup
    hipFree(d_recv_data); hipFree(d_evict_data); hipFree(d_sink);
    hipFree(d_shared_data); hipFree(d_recv_ts); hipFree(d_send_ts);
    hipFree(d_wave0_ts); hipFree(d_wave1_ts); hipFree(d_signal);
    delete[] h_signal; delete[] h_recv; delete[] h_send;
    delete[] h_wave0; delete[] h_wave1;
    return 0;
}
