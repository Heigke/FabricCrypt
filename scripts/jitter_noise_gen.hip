/*
 * jitter_noise_gen.hip — GPU memory jitter as shaped noise generator
 * ==================================================================
 * Uses memory contention findings to generate controllable noise streams:
 *
 * Three noise profiles derived from memory access patterns:
 *   LOW:   same_addr access → CV≈0.003, nearly deterministic
 *   MED:   random access, 256KB → CV≈0.02, moderate fluctuation
 *   HIGH:  strided_128, 16MB → CV≈0.20, strong analog noise
 *
 * Two spectral shapes:
 *   WHITE: independent random access each sample → flat spectrum
 *   PINK:  IIR-filtered cache pressure modulation → 1/f-like spectrum
 *
 * Tests:
 *   1. Noise profile comparison (LOW/MED/HIGH × 2000 samples)
 *   2. Spectral shaping: modulate access stride over time → PSD analysis
 *   3. Multi-stream independence: N parallel generators, cross-correlation
 *   4. Noise injection fidelity: embed known signal in noise floor, recover it
 *   5. Dynamic range: sweep from LOW→HIGH continuously, measure SNR envelope
 *
 * Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_noise_gen scripts/jitter_noise_gen.hip
 * Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_noise_gen [n_iters]
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#define WARP_SIZE 32
#define MAX_ITERS 4096
#define N_STREAMS 8

__device__ __forceinline__ unsigned long long gpu_ts() {
    return (unsigned long long)clock64();
}

// ═══ KERNEL: Configurable noise generator ═══
// mode: 0=same_addr(LOW), 1=sequential(MED_LO), 2=random(MED), 3=strided_128(HIGH)
// data_shift: log2(data_size/4) — controls cache level target
__global__ void noise_gen(unsigned long long* timestamps, float* sink,
                          float* gmem, int gmem_n, int n_iters,
                          int mode, int stride_val, int work) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    unsigned int rng = tid * 2654435761u + bid * 1013904223u + 42u;

    for (int i = 0; i < n_iters; i++) {
        unsigned long long t0 = gpu_ts();
        int idx;
        for (int w = 0; w < work; w++) {
            switch (mode) {
                case 0: // same_addr — L1 cached, minimal jitter
                    idx = bid * WARP_SIZE + tid;
                    break;
                case 1: // sequential — streaming
                    idx = (bid * WARP_SIZE + tid + w * WARP_SIZE) & (gmem_n - 1);
                    break;
                case 2: // random — cache thrashing
                    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                    idx = rng & (gmem_n - 1);
                    break;
                case 3: // strided — maximum L2 conflict
                    idx = ((bid * WARP_SIZE + tid) * stride_val + w * stride_val) & (gmem_n - 1);
                    break;
                default:
                    idx = 0;
            }
            sum += gmem[idx];
        }
        unsigned long long t1 = gpu_ts();
        timestamps[(bid * n_iters + i) * WARP_SIZE + tid] = t1 - t0;
    }
    sink[bid * WARP_SIZE + tid] = sum;
}

// ═══ KERNEL: IIR-modulated noise (pink spectrum) ═══
// Modulates stride over time with IIR filter: stride[t] = alpha*stride[t-1] + (1-alpha)*target[t]
// target oscillates or follows random walk → creates temporally correlated noise
__global__ void pink_noise_gen(unsigned long long* timestamps, float* sink,
                                float* gmem, int gmem_n, int n_iters,
                                int base_stride, float alpha, int work) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    unsigned int rng = tid * 2654435761u + bid * 1013904223u + 17u;

    float eff_stride = (float)base_stride;

    for (int i = 0; i < n_iters; i++) {
        // Random walk target stride
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        float target = (float)(16 + (rng % 240));  // stride 16..256
        eff_stride = alpha * eff_stride + (1.0f - alpha) * target;
        int cur_stride = max(1, (int)eff_stride);

        unsigned long long t0 = gpu_ts();
        for (int w = 0; w < work; w++) {
            int idx = ((bid * WARP_SIZE + tid) * cur_stride + w * cur_stride) & (gmem_n - 1);
            sum += gmem[idx];
        }
        unsigned long long t1 = gpu_ts();
        timestamps[(bid * n_iters + i) * WARP_SIZE + tid] = t1 - t0;
    }
    sink[bid * WARP_SIZE + tid] = sum;
}

// ═══ KERNEL: Signal-in-noise — embed binary signal in jitter magnitude ═══
// signal[i] ∈ {0,1}: 0→same_addr(quiet), 1→strided(loud)
__global__ void signal_in_noise(unsigned long long* timestamps, float* sink,
                                 float* gmem, int gmem_n, int n_iters,
                                 const int* signal, int work) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    unsigned int rng = tid * 2654435761u + bid * 1013904223u + 7u;

    for (int i = 0; i < n_iters; i++) {
        int sig = signal[i];
        unsigned long long t0 = gpu_ts();
        for (int w = 0; w < work; w++) {
            int idx;
            if (sig == 0) {
                // quiet: same address
                idx = bid * WARP_SIZE + tid;
            } else {
                // loud: strided access across L2
                idx = ((bid * WARP_SIZE + tid) * 128 + w * 128) & (gmem_n - 1);
            }
            sum += gmem[idx];
        }
        unsigned long long t1 = gpu_ts();
        timestamps[(bid * n_iters + i) * WARP_SIZE + tid] = t1 - t0;
    }
    sink[bid * WARP_SIZE + tid] = sum;
}

// ═══ KERNEL: Dynamic sweep — continuously vary noise level ═══
// stride ramps from 1 to max_stride linearly over n_iters
__global__ void dynamic_sweep(unsigned long long* timestamps, float* sink,
                               float* gmem, int gmem_n, int n_iters,
                               int max_stride, int work) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = 0; i < n_iters; i++) {
        int cur_stride = 1 + (max_stride - 1) * i / n_iters;
        unsigned long long t0 = gpu_ts();
        for (int w = 0; w < work; w++) {
            int idx = ((bid * WARP_SIZE + tid) * cur_stride + w * cur_stride) & (gmem_n - 1);
            sum += gmem[idx];
        }
        unsigned long long t1 = gpu_ts();
        timestamps[(bid * n_iters + i) * WARP_SIZE + tid] = t1 - t0;
    }
    sink[bid * WARP_SIZE + tid] = sum;
}

// ═══════════════════════════════════════════════
// HOST
// ═══════════════════════════════════════════════

struct Stats {
    double mean, std, cv, min_v, max_v;
};

Stats compute_stats(unsigned long long* data, int n, int stride, int offset) {
    // Collect lane 0 values, skip outliers (warmup)
    double* vals = new double[n];
    int valid = 0;
    double sum = 0;
    for (int i = 5; i < n; i++) {  // skip first 5 warmup
        double v = (double)data[i * stride + offset];
        if (v > 0 && v < 1e9) {
            vals[valid] = v;
            sum += v;
            valid++;
        }
    }
    Stats s = {0, 0, 0, 1e18, -1e18};
    if (valid < 10) { delete[] vals; return s; }
    s.mean = sum / valid;
    double var = 0;
    for (int i = 0; i < valid; i++) {
        double d = vals[i] - s.mean;
        var += d * d;
        if (vals[i] < s.min_v) s.min_v = vals[i];
        if (vals[i] > s.max_v) s.max_v = vals[i];
    }
    s.std = sqrt(var / valid);
    s.cv = (s.mean > 0) ? s.std / s.mean : 0;
    delete[] vals;
    return s;
}

double pearson(double* x, double* y, int n) {
    double sx = 0, sy = 0, sxy = 0, sx2 = 0, sy2 = 0;
    for (int i = 0; i < n; i++) {
        sx += x[i]; sy += y[i];
        sxy += x[i] * y[i];
        sx2 += x[i] * x[i]; sy2 += y[i] * y[i];
    }
    double num = n * sxy - sx * sy;
    double den = sqrt((n * sx2 - sx * sx) * (n * sy2 - sy * sy));
    return (den > 0) ? num / den : 0;
}

int main(int argc, char** argv) {
    int n_iters = 2000;
    if (argc > 1) n_iters = atoi(argv[1]);
    if (n_iters > MAX_ITERS) n_iters = MAX_ITERS;

    int n_blocks = 20;  // all CUs
    int work = 64;      // memory ops per sample

    // Allocate 16MB global memory arena
    int gmem_n = 4 * 1024 * 1024;  // 16MB / 4 bytes
    float *d_gmem, *d_sink;
    unsigned long long *d_ts;
    hipMalloc(&d_gmem, gmem_n * sizeof(float));
    hipMalloc(&d_sink, n_blocks * WARP_SIZE * sizeof(float));
    hipMalloc(&d_ts, n_blocks * n_iters * WARP_SIZE * sizeof(unsigned long long));
    hipMemset(d_gmem, 0, gmem_n * sizeof(float));

    unsigned long long* h_ts = new unsigned long long[n_blocks * n_iters * WARP_SIZE];

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    fprintf(stderr, "Device: %s, CUs: %d, iters: %d\n", prop.name, prop.multiProcessorCount, n_iters);

    // CSV header
    printf("test,block,iter,lane0_clk\n");

    // ═══ TEST 1: Noise profiles comparison ═══
    fprintf(stderr, "\n═══ TEST 1: Noise Profiles (20 blocks × %d iters) ═══\n", n_iters);
    const char* mode_names[] = {"LOW(same_addr)", "MED_LO(sequential)", "MED(random)", "HIGH(strided_128)"};
    int strides[] = {1, 1, 1, 128};

    for (int mode = 0; mode < 4; mode++) {
        hipMemset(d_ts, 0, n_blocks * n_iters * WARP_SIZE * sizeof(unsigned long long));
        noise_gen<<<n_blocks, WARP_SIZE>>>(d_ts, d_sink, d_gmem, gmem_n, n_iters,
                                            mode, strides[mode], work);
        hipDeviceSynchronize();
        hipMemcpy(h_ts, d_ts, n_blocks * n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

        // Stats across all blocks
        double total_mean = 0, total_std = 0;
        int valid_blocks = 0;
        for (int b = 0; b < n_blocks; b++) {
            Stats st = compute_stats(h_ts + b * n_iters * WARP_SIZE, n_iters, WARP_SIZE, 0);
            if (st.mean > 0) {
                total_mean += st.mean;
                total_std += st.std;
                valid_blocks++;
            }
        }
        if (valid_blocks > 0) {
            total_mean /= valid_blocks;
            total_std /= valid_blocks;
        }
        fprintf(stderr, "  %-22s: mean=%8.0f, std=%7.1f, CV=%.6f\n",
                mode_names[mode], total_mean, total_std,
                total_mean > 0 ? total_std / total_mean : 0);

        // CSV: just block 0, lane 0
        for (int i = 0; i < n_iters; i++) {
            printf("profile_%d,0,%d,%llu\n", mode, i, h_ts[i * WARP_SIZE]);
        }
    }

    // ═══ TEST 2: Pink noise (IIR stride modulation) ═══
    fprintf(stderr, "\n═══ TEST 2: Pink Noise (IIR alpha sweep) ═══\n");
    float alphas[] = {0.0f, 0.5f, 0.9f, 0.95f, 0.99f};
    for (int a = 0; a < 5; a++) {
        hipMemset(d_ts, 0, n_blocks * n_iters * WARP_SIZE * sizeof(unsigned long long));
        pink_noise_gen<<<n_blocks, WARP_SIZE>>>(d_ts, d_sink, d_gmem, gmem_n, n_iters,
                                                  128, alphas[a], work);
        hipDeviceSynchronize();
        hipMemcpy(h_ts, d_ts, n_blocks * n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

        Stats st = compute_stats(h_ts, n_iters, WARP_SIZE, 0);

        // Compute autocorrelation at lag 1
        double* vals = new double[n_iters];
        int valid = 0;
        for (int i = 5; i < n_iters; i++) {
            double v = (double)h_ts[i * WARP_SIZE];
            if (v > 0 && v < 1e9) vals[valid++] = v;
        }
        double acf1 = 0;
        if (valid > 10) {
            double m = 0;
            for (int i = 0; i < valid; i++) m += vals[i];
            m /= valid;
            double num = 0, den = 0;
            for (int i = 0; i < valid - 1; i++) {
                num += (vals[i] - m) * (vals[i+1] - m);
                den += (vals[i] - m) * (vals[i] - m);
            }
            den += (vals[valid-1] - m) * (vals[valid-1] - m);
            acf1 = (den > 0) ? num / den : 0;
        }
        delete[] vals;

        fprintf(stderr, "  alpha=%.2f: mean=%8.0f, std=%7.1f, CV=%.6f, ACF(1)=%.4f\n",
                alphas[a], st.mean, st.std, st.cv, acf1);

        for (int i = 0; i < n_iters; i++) {
            printf("pink_%.2f,0,%d,%llu\n", alphas[a], i, h_ts[i * WARP_SIZE]);
        }
    }

    // ═══ TEST 3: Multi-stream independence ═══
    fprintf(stderr, "\n═══ TEST 3: Multi-Stream Independence (%d streams) ═══\n", N_STREAMS);
    hipMemset(d_ts, 0, n_blocks * n_iters * WARP_SIZE * sizeof(unsigned long long));
    noise_gen<<<N_STREAMS, WARP_SIZE>>>(d_ts, d_sink, d_gmem, gmem_n, n_iters,
                                         2, 1, work);  // mode=2 (random)
    hipDeviceSynchronize();
    hipMemcpy(h_ts, d_ts, N_STREAMS * n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

    // Cross-correlation between streams (lane 0 of each block)
    double** streams = new double*[N_STREAMS];
    int stream_len = 0;
    for (int s = 0; s < N_STREAMS; s++) {
        streams[s] = new double[n_iters];
        int cnt = 0;
        for (int i = 5; i < n_iters; i++) {
            double v = (double)h_ts[(s * n_iters + i) * WARP_SIZE];
            if (v > 0 && v < 1e9) {
                streams[s][cnt] = v;
                cnt++;
            }
        }
        if (s == 0) stream_len = cnt;
    }

    fprintf(stderr, "  Cross-stream correlations (stream 0 vs):\n");
    for (int s = 1; s < N_STREAMS; s++) {
        double r = pearson(streams[0], streams[s], stream_len);
        fprintf(stderr, "    stream %d: r=%.4f\n", s, r);
    }
    for (int s = 0; s < N_STREAMS; s++) delete[] streams[s];
    delete[] streams;

    // ═══ TEST 4: Signal in noise — binary signal recovery ═══
    fprintf(stderr, "\n═══ TEST 4: Signal Recovery (binary signal in jitter) ═══\n");

    // Create binary signal: alternating blocks of 50 samples
    int* h_signal = new int[n_iters];
    int* d_signal;
    int block_size = 50;
    for (int i = 0; i < n_iters; i++) {
        h_signal[i] = ((i / block_size) % 2);
    }
    hipMalloc(&d_signal, n_iters * sizeof(int));
    hipMemcpy(d_signal, h_signal, n_iters * sizeof(int), hipMemcpyHostToDevice);

    // Test with different block sizes
    int block_sizes[] = {10, 25, 50, 100, 200};
    for (int bs = 0; bs < 5; bs++) {
        for (int i = 0; i < n_iters; i++) {
            h_signal[i] = ((i / block_sizes[bs]) % 2);
        }
        hipMemcpy(d_signal, h_signal, n_iters * sizeof(int), hipMemcpyHostToDevice);

        hipMemset(d_ts, 0, n_blocks * n_iters * WARP_SIZE * sizeof(unsigned long long));
        signal_in_noise<<<1, WARP_SIZE>>>(d_ts, d_sink, d_gmem, gmem_n, n_iters,
                                           d_signal, work);
        hipDeviceSynchronize();
        hipMemcpy(h_ts, d_ts, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

        // Classify: compute mean timing for signal=0 vs signal=1
        double sum0 = 0, sum1 = 0;
        int n0 = 0, n1 = 0;
        double* sig_vals = new double[n_iters];
        double* tim_vals = new double[n_iters];
        int valid = 0;
        for (int i = 5; i < n_iters; i++) {
            double v = (double)h_ts[i * WARP_SIZE];
            if (v > 0 && v < 1e9) {
                if (h_signal[i] == 0) { sum0 += v; n0++; }
                else { sum1 += v; n1++; }
                sig_vals[valid] = (double)h_signal[i];
                tim_vals[valid] = v;
                valid++;
            }
        }
        double mean0 = n0 > 0 ? sum0 / n0 : 0;
        double mean1 = n1 > 0 ? sum1 / n1 : 0;
        double r = pearson(sig_vals, tim_vals, valid);

        // Binary classification accuracy (threshold = midpoint)
        double thresh = (mean0 + mean1) / 2.0;
        int correct = 0;
        for (int i = 0; i < valid; i++) {
            int predicted = (tim_vals[i] > thresh) ? 1 : 0;
            if (predicted == (int)sig_vals[i]) correct++;
        }
        double accuracy = valid > 0 ? 100.0 * correct / valid : 0;

        fprintf(stderr, "  block_size=%3d: mean0=%8.0f, mean1=%8.0f, delta=%+.0f (%.1f%%), r=%.4f, acc=%.1f%%\n",
                block_sizes[bs], mean0, mean1, mean1 - mean0,
                mean0 > 0 ? 100.0 * (mean1 - mean0) / mean0 : 0,
                r, accuracy);

        delete[] sig_vals;
        delete[] tim_vals;

        if (bs == 2) {  // 50-sample blocks: save CSV
            for (int i = 0; i < n_iters; i++) {
                printf("signal_%d,0,%d,%llu\n", h_signal[i], i, h_ts[i * WARP_SIZE]);
            }
        }
    }

    hipFree(d_signal);
    delete[] h_signal;

    // ═══ TEST 5: Dynamic sweep ═══
    fprintf(stderr, "\n═══ TEST 5: Dynamic Sweep (stride 1→256) ═══\n");
    hipMemset(d_ts, 0, n_blocks * n_iters * WARP_SIZE * sizeof(unsigned long long));
    dynamic_sweep<<<1, WARP_SIZE>>>(d_ts, d_sink, d_gmem, gmem_n, n_iters, 256, work);
    hipDeviceSynchronize();
    hipMemcpy(h_ts, d_ts, n_iters * WARP_SIZE * sizeof(unsigned long long), hipMemcpyDeviceToHost);

    // Compute windowed stats (windows of 100)
    int win = 100;
    fprintf(stderr, "  Windowed stats (100-sample windows):\n");
    for (int w_start = 0; w_start < n_iters - win; w_start += win) {
        double sum = 0, sum2 = 0;
        int cnt = 0;
        for (int i = w_start; i < w_start + win; i++) {
            double v = (double)h_ts[i * WARP_SIZE];
            if (v > 0 && v < 1e9) {
                sum += v; sum2 += v * v; cnt++;
            }
        }
        if (cnt > 1) {
            double m = sum / cnt;
            double s = sqrt((sum2 / cnt) - m * m);
            int stride_at = 1 + 255 * w_start / n_iters;
            fprintf(stderr, "    stride≈%3d: mean=%8.0f, std=%7.1f\n", stride_at, m, s);
        }
    }

    // CSV
    for (int i = 0; i < n_iters; i++) {
        printf("sweep,0,%d,%llu\n", i, h_ts[i * WARP_SIZE]);
    }

    fprintf(stderr, "\nDone.\n");

    hipFree(d_gmem);
    hipFree(d_sink);
    hipFree(d_ts);
    delete[] h_ts;
    return 0;
}
