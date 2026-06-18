/*
 * jitter_coupling2.hip — Amplifying cross-wavefront coupling
 * ============================================================
 * v1 finding: L1 cache isolates CUs completely. Cross-CU coupling = 0.
 * BUT: intra-CU LDS coupling r=0.97. And original L2-dependent test gave r=0.088.
 *
 * Strategy: Force BOTH sender and receiver through L2 (large data), then
 * modulate L2 bandwidth. Also amplify intra-CU LDS channel.
 *
 * TEST 1: L2-forced coupling — both sides use 4MB data (L2-dependent)
 * TEST 2: L2 bandwidth starvation — N senders saturate L2, receiver measures
 * TEST 3: LDS amplified — larger LDS working set, more bank conflicts
 * TEST 4: LDS with variable conflict intensity (0, 2-way, 4-way, 8-way, 16-way)
 * TEST 5: Combined L2+LDS — intra-CU LDS coupling WHILE L2-pressured
 * TEST 6: Temporal accumulation — average over K samples, improve SNR
 *
 * Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_coupling2 scripts/jitter_coupling2.hip
 * Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_coupling2 [n_iters]
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define WARP_SIZE 32
#define MAX_ITERS 4096
#define L2_SIZE (4*1024*1024/4)  // 4MB in floats

__device__ __forceinline__ unsigned long long gpu_ts() {
    return (unsigned long long)clock64();
}

// ═══ TEST 1: L2-forced coupling ═══
// BOTH receiver and senders access SAME 4MB region randomly
// When signal=1: senders do heavy random access (L2 contention)
// When signal=0: senders do light sequential access or ALU
__global__ void l2_forced_coupling(unsigned long long* recv_ts, float* shared_l2,
                                    float* sink, const int* signal,
                                    int n_iters, int work, int l2_n) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    // Different RNG seed per block
    unsigned int rng = tid * 2654435761u + bid * 1013904223u + 42u;

    if (bid == 0) {
        // RECEIVER: random access to shared L2 region
        for (int i = 0; i < n_iters; i++) {
            unsigned long long t0 = gpu_ts();
            for (int w = 0; w < work; w++) {
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                int idx = rng % l2_n;
                sum += shared_l2[idx];
            }
            unsigned long long t1 = gpu_ts();
            recv_ts[i * WARP_SIZE + tid] = t1 - t0;
        }
    } else {
        // SENDER(s)
        for (int i = 0; i < n_iters; i++) {
            if (signal[i] == 1) {
                // Heavy random L2 access — create contention
                for (int w = 0; w < work * 2; w++) {
                    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                    int idx = rng % l2_n;
                    sum += shared_l2[idx];
                }
            } else {
                // Light ALU — no L2 pressure
                float x = (float)tid;
                for (int w = 0; w < work * 8; w++) {
                    x = x * 1.0001f + 0.001f;
                }
                sum += x;
            }
        }
    }
    sink[bid * WARP_SIZE + tid] = sum;
}

// ═══ TEST 2: L2 bandwidth saturation ═══
// All senders do strided access to fill L2 lines, receiver random-reads
__global__ void l2_saturation(unsigned long long* recv_ts, float* shared_l2,
                               float* sink, const int* signal,
                               int n_iters, int work, int l2_n) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    unsigned int rng = tid * 2654435761u + bid * 1013904223u + 7u;

    if (bid == 0) {
        // RECEIVER: random reads
        for (int i = 0; i < n_iters; i++) {
            unsigned long long t0 = gpu_ts();
            for (int w = 0; w < work; w++) {
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                sum += shared_l2[rng % l2_n];
            }
            unsigned long long t1 = gpu_ts();
            recv_ts[i * WARP_SIZE + tid] = t1 - t0;
        }
    } else {
        // SENDER: strided writes that evict L2 cache lines
        for (int i = 0; i < n_iters; i++) {
            if (signal[i] == 1) {
                for (int w = 0; w < work; w++) {
                    // Strided by 128 = different L2 set each access
                    int idx = ((bid * WARP_SIZE + tid) * 128 + w * 128 * gridDim.x) % l2_n;
                    shared_l2[idx] = (float)(i + w);  // WRITE — forces L2 line ownership
                }
            } else {
                float x = (float)tid;
                for (int w = 0; w < work * 8; w++) x = x * 1.0001f + 0.001f;
                sum += x;
            }
        }
    }
    sink[bid * WARP_SIZE + tid] = sum;
}

// ═══ TEST 3: LDS amplified — multiple wavefronts, max bank conflicts ═══
__global__ void lds_amplified(unsigned long long* recv_ts, float* sink,
                               const int* signal, int n_iters, int work, int n_waves) {
    __shared__ float lds[4096];  // 16KB LDS
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int is_receiver = (warp_id == 0);

    // Init LDS
    for (int i = threadIdx.x; i < 4096; i += blockDim.x)
        lds[i] = 1.0f;
    __syncthreads();

    float sum = 0.0f;

    for (int i = 0; i < n_iters; i++) {
        // No sync between iterations — let wavefronts run freely
        unsigned long long t0 = gpu_ts();

        if (is_receiver) {
            // Receiver: read LDS with specific bank pattern
            for (int w = 0; w < work; w++) {
                // Access pattern: stride=1 → bank 0,1,2,...,31
                int idx = (lane + w * 32) % 4096;
                sum += lds[idx];
            }
        } else {
            if (signal[i] == 1) {
                // Sender wavefronts: WRITE to SAME banks as receiver → conflicts!
                for (int w = 0; w < work; w++) {
                    // Same stride pattern as receiver → guaranteed bank conflicts
                    int idx = (lane + w * 32 + warp_id) % 4096;
                    lds[idx] = (float)(i + w + warp_id);
                }
            } else {
                // ALU only — no LDS
                float x = (float)lane;
                for (int w = 0; w < work; w++) x = x * 1.0001f + 0.001f;
                sum += x;
            }
        }

        unsigned long long t1 = gpu_ts();
        if (is_receiver) {
            recv_ts[i * WARP_SIZE + lane] = t1 - t0;
        }
    }
    sink[threadIdx.x] = sum;
}

// ═══ TEST 4: LDS bank conflict intensity sweep ═══
// Vary conflict level: 0-way (no conflict), 2-way, 4-way, 8-way, 16-way, 32-way
__global__ void lds_conflict_sweep(unsigned long long* recv_ts, float* sink,
                                    int n_iters, int work, int conflict_level) {
    __shared__ float lds[4096];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    for (int i = threadIdx.x; i < 4096; i += blockDim.x) lds[i] = 1.0f;
    __syncthreads();

    float sum = 0.0f;

    for (int i = 0; i < n_iters; i++) {
        unsigned long long t0 = gpu_ts();

        if (warp_id == 0) {
            // Receiver: sequential LDS read
            for (int w = 0; w < work; w++) {
                int idx = (lane + w * 32) % 4096;
                sum += lds[idx];
            }
        } else {
            // Sender: write with controlled conflict level
            // conflict_level = stride between accesses
            // stride=32 → 0 conflicts (different banks)
            // stride=1 → all 32 lanes hit same bank → 32-way conflict
            for (int w = 0; w < work; w++) {
                int base = w * 32 + warp_id;
                int idx;
                if (conflict_level == 0) {
                    idx = (lane * 32 + base) % 4096;  // no conflict
                } else {
                    // conflict_level lanes map to same bank
                    idx = ((lane / conflict_level) * 32 + (lane % conflict_level) + base) % 4096;
                }
                lds[idx] = (float)(i + w);
            }
        }

        unsigned long long t1 = gpu_ts();
        if (warp_id == 0) {
            recv_ts[i * WARP_SIZE + lane] = t1 - t0;
        }
    }
    sink[threadIdx.x] = sum;
}

// ═══════════════════════════════════════════════
// HOST
// ═══════════════════════════════════════════════

double sig_corr_f(int* signal, unsigned long long* ts, int n, int stride) {
    double sx = 0, sy = 0, sxy = 0, sx2 = 0, sy2 = 0;
    int v = 0;
    for (int i = 5; i < n; i++) {
        double xv = (double)signal[i];
        double yv = (double)ts[i * stride];
        if (yv > 0 && yv < 1e9) {
            sx += xv; sy += yv; sxy += xv * yv;
            sx2 += xv * xv; sy2 += yv * yv; v++;
        }
    }
    if (v < 10) return 0;
    double num = v * sxy - sx * sy;
    double den = sqrt((v * sx2 - sx * sx) * (v * sy2 - sy * sy));
    return (den > 0) ? num / den : 0;
}

void analyze(const char* label, int* signal, unsigned long long* ts, int n, int stride) {
    double s0 = 0, s1 = 0; int n0 = 0, n1 = 0;
    double all_sum = 0, all_sum2 = 0; int all_n = 0;
    for (int i = 5; i < n; i++) {
        double v = (double)ts[i * stride];
        if (v > 0 && v < 1e9) {
            if (signal[i] == 0) { s0 += v; n0++; }
            else { s1 += v; n1++; }
            all_sum += v; all_sum2 += v*v; all_n++;
        }
    }
    double m0 = n0 > 0 ? s0/n0 : 0;
    double m1 = n1 > 0 ? s1/n1 : 0;
    double mean = all_n > 0 ? all_sum/all_n : 0;
    double std_v = all_n > 1 ? sqrt(all_sum2/all_n - mean*mean) : 0;
    double r = sig_corr_f(signal, ts, n, stride);
    double delta = m1 - m0;
    double pct = m0 > 0 ? 100.0 * delta / m0 : 0;

    // Classification accuracy
    double thresh = (m0 + m1) / 2.0;
    int correct = 0, total = 0;
    for (int i = 5; i < n; i++) {
        double v = (double)ts[i * stride];
        if (v > 0 && v < 1e9) {
            int pred = (v > thresh) ? 1 : 0;
            if (pred == signal[i]) correct++;
            total++;
        }
    }
    double acc = total > 0 ? 100.0 * correct / total : 0;

    // Temporal accumulation: average over windows of K, then classify
    int K = 10;
    int win_correct = 0, win_total = 0;
    for (int start = 5; start + K <= n; start += K) {
        double avg = 0; int cnt = 0;
        int sig_mode = signal[start];
        bool uniform = true;
        for (int j = start; j < start + K; j++) {
            double v = (double)ts[j * stride];
            if (v > 0 && v < 1e9) { avg += v; cnt++; }
            if (signal[j] != sig_mode) uniform = false;
        }
        if (cnt > 0 && uniform) {
            avg /= cnt;
            int pred = (avg > thresh) ? 1 : 0;
            if (pred == sig_mode) win_correct++;
            win_total++;
        }
    }
    double win_acc = win_total > 0 ? 100.0 * win_correct / win_total : 0;

    fprintf(stderr, "  %s: mean=%.0f, std=%.1f, m0=%.0f, m1=%.0f, delta=%+.0f (%+.1f%%), r=%.4f, acc=%.1f%%, win10_acc=%.1f%%\n",
            label, mean, std_v, m0, m1, delta, pct, r, acc, win_acc);
}

int main(int argc, char** argv) {
    int n_iters = 2000;
    if (argc > 1) n_iters = atoi(argv[1]);
    if (n_iters > MAX_ITERS) n_iters = MAX_ITERS;

    int* h_signal = new int[n_iters];
    for (int i = 0; i < n_iters; i++) h_signal[i] = ((i / 50) % 2);
    int* d_signal;
    hipMalloc(&d_signal, n_iters * sizeof(int));
    hipMemcpy(d_signal, h_signal, n_iters * sizeof(int), hipMemcpyHostToDevice);

    float *d_shared_l2, *d_sink;
    unsigned long long *d_recv_ts;
    hipMalloc(&d_shared_l2, L2_SIZE * sizeof(float));
    hipMalloc(&d_sink, 20 * WARP_SIZE * sizeof(float));
    hipMalloc(&d_recv_ts, n_iters * WARP_SIZE * sizeof(unsigned long long));
    hipMemset(d_shared_l2, 0, L2_SIZE * sizeof(float));

    unsigned long long* h_recv = new unsigned long long[n_iters * WARP_SIZE];

    fprintf(stderr, "═══ Cross-Wavefront Coupling Amplification v2 ═══\n");
    fprintf(stderr, "Iters: %d, L2_SIZE: %d floats (4MB)\n\n", n_iters, L2_SIZE);

    printf("test,param,iter,recv_clk,signal\n");

    // ═══ TEST 1: L2-forced coupling ═══
    fprintf(stderr, "═══ TEST 1: L2-Forced Coupling (both sides random 4MB) ═══\n");
    int sender_counts[] = {1, 2, 4, 8, 12, 16, 19};
    for (int sc = 0; sc < 7; sc++) {
        int ns = sender_counts[sc];
        hipMemset(d_recv_ts, 0, n_iters * WARP_SIZE * sizeof(unsigned long long));
        l2_forced_coupling<<<ns + 1, WARP_SIZE>>>(
            d_recv_ts, d_shared_l2, d_sink, d_signal, n_iters, 128, L2_SIZE);
        hipDeviceSynchronize();
        hipMemcpy(h_recv, d_recv_ts, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

        char label[64];
        snprintf(label, 64, "%2d senders", ns);
        analyze(label, h_signal, h_recv, n_iters, WARP_SIZE);

        if (ns == 19) {
            for (int i = 0; i < n_iters; i++)
                printf("l2_forced,%d,%d,%llu,%d\n", ns, i, h_recv[i*WARP_SIZE], h_signal[i]);
        }
    }

    // ═══ TEST 2: L2 saturation (senders WRITE) ═══
    fprintf(stderr, "\n═══ TEST 2: L2 Saturation (senders WRITE to shared 4MB) ═══\n");
    for (int sc = 0; sc < 7; sc++) {
        int ns = sender_counts[sc];
        hipMemset(d_recv_ts, 0, n_iters * WARP_SIZE * sizeof(unsigned long long));
        l2_saturation<<<ns + 1, WARP_SIZE>>>(
            d_recv_ts, d_shared_l2, d_sink, d_signal, n_iters, 128, L2_SIZE);
        hipDeviceSynchronize();
        hipMemcpy(h_recv, d_recv_ts, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

        char label[64];
        snprintf(label, 64, "%2d senders(W)", ns);
        analyze(label, h_signal, h_recv, n_iters, WARP_SIZE);
    }

    // ═══ TEST 3: LDS amplified (multiple sender wavefronts) ═══
    fprintf(stderr, "\n═══ TEST 3: LDS Amplified (multiple sender wavefronts per block) ═══\n");
    int wave_counts[] = {2, 4, 8, 16};
    for (int wc = 0; wc < 4; wc++) {
        int nw = wave_counts[wc];
        hipMemset(d_recv_ts, 0, n_iters * WARP_SIZE * sizeof(unsigned long long));
        lds_amplified<<<1, nw * WARP_SIZE>>>(
            d_recv_ts, d_sink, d_signal, n_iters, 64, nw);
        hipDeviceSynchronize();
        hipMemcpy(h_recv, d_recv_ts, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

        char label[64];
        snprintf(label, 64, "%2d waves", nw);
        analyze(label, h_signal, h_recv, n_iters, WARP_SIZE);

        if (nw == 16) {
            for (int i = 0; i < n_iters; i++)
                printf("lds_amp,%d,%d,%llu,%d\n", nw, i, h_recv[i*WARP_SIZE], h_signal[i]);
        }
    }

    // ═══ TEST 4: LDS bank conflict intensity ═══
    fprintf(stderr, "\n═══ TEST 4: LDS Bank Conflict Intensity Sweep ═══\n");
    int conflicts[] = {0, 2, 4, 8, 16, 32};
    for (int c = 0; c < 6; c++) {
        hipMemset(d_recv_ts, 0, n_iters * WARP_SIZE * sizeof(unsigned long long));
        lds_conflict_sweep<<<1, 2 * WARP_SIZE>>>(
            d_recv_ts, d_sink, n_iters, 64, conflicts[c]);
        hipDeviceSynchronize();
        hipMemcpy(h_recv, d_recv_ts, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

        // No signal modulation — just measure mean timing at each conflict level
        double sum = 0, sum2 = 0; int cnt = 0;
        for (int i = 5; i < n_iters; i++) {
            double v = (double)h_recv[i * WARP_SIZE];
            if (v > 0 && v < 1e9) { sum += v; sum2 += v*v; cnt++; }
        }
        double mean = cnt > 0 ? sum/cnt : 0;
        double std_v = cnt > 1 ? sqrt(sum2/cnt - mean*mean) : 0;
        fprintf(stderr, "  conflict=%2d-way: mean=%8.0f, std=%7.1f, CV=%.6f\n",
                conflicts[c], mean, std_v, mean > 0 ? std_v/mean : 0);
    }

    // ═══ TEST 5: Combined L2+LDS ═══
    fprintf(stderr, "\n═══ TEST 5: Combined L2 + LDS (intra-CU LDS with L2 background) ═══\n");
    // Run LDS coupling while also running L2-heavy blocks on other CUs
    // Use 18 L2-heavy blocks + 1 LDS block (2 wavefronts)
    // This is done by launching L2 blocks in background, then LDS block

    // First: LDS alone (baseline)
    hipMemset(d_recv_ts, 0, n_iters * WARP_SIZE * sizeof(unsigned long long));
    lds_amplified<<<1, 4 * WARP_SIZE>>>(d_recv_ts, d_sink, d_signal, n_iters, 64, 4);
    hipDeviceSynchronize();
    hipMemcpy(h_recv, d_recv_ts, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);
    analyze("LDS alone(4w)", h_signal, h_recv, n_iters, WARP_SIZE);

    fprintf(stderr, "\nDone.\n");

    hipFree(d_shared_l2); hipFree(d_sink); hipFree(d_recv_ts); hipFree(d_signal);
    delete[] h_signal; delete[] h_recv;
    return 0;
}
