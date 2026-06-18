/*  encode_probe_channels.hip  —  z2259: Can we ENCODE information into high-ACF channels?
 *
 *  Key insight: ACF=0.98 means step N's hardware state influences step N+1.
 *  If we make step N's workload STRONGLY input-dependent, step N+1's probe
 *  should carry a trace of step N's input.
 *
 *  Protocol for each channel:
 *    1. ENCODE kernel: input-dependent workload (heavy vs light)
 *    2. PROBE kernel:  FIXED workload, measure timing → does it depend on previous input?
 *
 *  Channels tested (by ACF):
 *    A. entry_clock  (0.983) — PLL thermal drift via compute load
 *    B. FMA pipeline  (0.943) — voltage droop via FMA count
 *    C. SFU pipeline  (0.939) — voltage droop via SFU count
 *    D. LDS bank     (0.783) — bank conflict state via access pattern
 *    E. dispatch_Δ   (0.623) — dispatch overhead via kernel size
 *    F. L2 cache     (0.280) — cache line state via access footprint
 *    G. Global mem    (new)  — DRAM page/row buffer via access pattern
 *
 *  Compile: hipcc --offload-arch=gfx1100 -O1 -o scripts/encode_probe_channels scripts/encode_probe_channels.hip
 *  Run:     HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/encode_probe_channels
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ─── ENCODE kernels: workload depends on input_class ───

// Channel A: Compute load → PLL thermal drift
__global__ void k_encode_compute(int input_class, int heavy_iters, int light_iters,
                                  unsigned long long* out_time) {
    int iters = (input_class == 1) ? heavy_iters : light_iters;
    float x = 1.0001f + (float)threadIdx.x * 0.00001f;
    unsigned long long t0 = wall_clock64();
    for (int i = 0; i < iters; i++) {
        x = x * 1.0001f + 0.0001f;
    }
    unsigned long long t1 = wall_clock64();
    if (threadIdx.x == 0) {
        out_time[0] = t1 - t0;
        // prevent optimization
        if (x == -999.0f) out_time[1] = (unsigned long long)x;
    }
}

// Channel B: FMA-specific → voltage droop on FMA unit
__global__ void k_encode_fma(int input_class, int heavy_iters, int light_iters,
                              unsigned long long* out_time) {
    int iters = (input_class == 1) ? heavy_iters : light_iters;
    float a = 1.0001f + (float)threadIdx.x * 0.00001f;
    float b = 0.9999f;
    float c = 0.0001f;
    unsigned long long t0 = wall_clock64();
    for (int i = 0; i < iters; i++) {
        a = __fmaf_rn(a, b, c);  // FMA: a = a*b + c
    }
    unsigned long long t1 = wall_clock64();
    if (threadIdx.x == 0) {
        out_time[0] = t1 - t0;
        if (a == -999.0f) out_time[1] = (unsigned long long)a;
    }
}

// Channel C: SFU-specific → voltage droop on SFU unit
__global__ void k_encode_sfu(int input_class, int heavy_iters, int light_iters,
                              unsigned long long* out_time) {
    int iters = (input_class == 1) ? heavy_iters : light_iters;
    float x = 1.0f + (float)threadIdx.x * 0.001f;
    unsigned long long t0 = wall_clock64();
    for (int i = 0; i < iters; i++) {
        x = __sinf(x) + 1.5f;  // SFU transcendental
    }
    unsigned long long t1 = wall_clock64();
    if (threadIdx.x == 0) {
        out_time[0] = t1 - t0;
        if (x == -999.0f) out_time[1] = (unsigned long long)x;
    }
}

// Channel D: LDS — different access patterns based on input
__global__ void k_encode_lds(int input_class, int iters,
                              unsigned long long* out_time) {
    __shared__ float smem[1024];
    int tid = threadIdx.x;
    smem[tid] = (float)tid;
    __syncthreads();

    float acc = 0.0f;
    unsigned long long t0 = wall_clock64();
    if (input_class == 1) {
        // Heavy: bank-conflict pattern (stride 17 → conflicts on 32-bank LDS)
        for (int i = 0; i < iters; i++) {
            int idx = (tid * 17 + i) & 1023;
            acc += smem[idx];
            smem[idx] = acc * 0.001f;
        }
    } else {
        // Light: conflict-free sequential
        for (int i = 0; i < iters; i++) {
            int idx = (tid + i) & 1023;
            acc += smem[idx];
            smem[idx] = acc * 0.001f;
        }
    }
    unsigned long long t1 = wall_clock64();
    if (threadIdx.x == 0) {
        out_time[0] = t1 - t0;
        if (acc == -999.0f) out_time[1] = (unsigned long long)acc;
    }
}

// Channel F: L2 cache — different access footprint based on input
__global__ void k_encode_l2(int input_class, float* buf_small, float* buf_large,
                             int small_n, int large_n, int iters,
                             unsigned long long* out_time) {
    float acc = 0.0f;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long t0 = wall_clock64();
    if (input_class == 1) {
        // Heavy: thrash L2 with large buffer (4MB)
        for (int i = 0; i < iters; i++) {
            int idx = (tid * 127 + i * 65537) % large_n;
            acc += buf_large[idx];
        }
    } else {
        // Light: stay in L2 with small buffer (16KB)
        for (int i = 0; i < iters; i++) {
            int idx = (tid * 3 + i) % small_n;
            acc += buf_small[idx];
        }
    }
    unsigned long long t1 = wall_clock64();
    if (threadIdx.x == 0) {
        out_time[blockIdx.x] = t1 - t0;
        if (acc == -999.0f) buf_small[0] = acc;
    }
}

// Channel G: Global memory — different DRAM row patterns
__global__ void k_encode_global(int input_class, float* buf, int buf_n, int iters,
                                 unsigned long long* out_time) {
    float acc = 0.0f;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long t0 = wall_clock64();
    if (input_class == 1) {
        // Heavy: random stride across DRAM rows (stride=2048 floats = 8KB ≈ row size)
        for (int i = 0; i < iters; i++) {
            int idx = (tid * 2048 + i * 131071) % buf_n;
            acc += buf[idx];
        }
    } else {
        // Light: sequential (same DRAM row)
        for (int i = 0; i < iters; i++) {
            int idx = (tid * 4 + i) % buf_n;
            acc += buf[idx];
        }
    }
    unsigned long long t1 = wall_clock64();
    if (threadIdx.x == 0) {
        out_time[blockIdx.x] = t1 - t0;
        if (acc == -999.0f) buf[0] = acc;
    }
}

// ─── PROBE kernel: FIXED workload, measures all channels ───
__global__ void k_probe_fixed(float* lds_buf_dummy, float* l2_buf, int l2_n,
                               unsigned long long* out_probes) {
    // out_probes layout per block: [entry, fma, sfu, lds, l2, exit] = 6 values
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int base = blockIdx.x * 6;

    // Entry clock
    unsigned long long entry = wall_clock64();

    // FMA probe: exactly 200 iterations
    float a = 1.0001f + (float)tid * 0.00001f;
    float b = 0.9999f, c = 0.0001f;
    unsigned long long t0 = wall_clock64();
    for (int i = 0; i < 200; i++) a = __fmaf_rn(a, b, c);
    unsigned long long fma_time = wall_clock64() - t0;

    // SFU probe: exactly 100 iterations
    float x = 1.0f + (float)tid * 0.001f;
    t0 = wall_clock64();
    for (int i = 0; i < 100; i++) x = __sinf(x) + 1.5f;
    unsigned long long sfu_time = wall_clock64() - t0;

    // LDS probe: exactly 200 sequential reads
    smem[tid & 255] = (float)tid;
    __syncthreads();
    float lds_acc = 0.0f;
    t0 = wall_clock64();
    for (int i = 0; i < 200; i++) {
        lds_acc += smem[(tid + i) & 255];
    }
    unsigned long long lds_time = wall_clock64() - t0;

    // L2 probe: exactly 100 sequential reads from small region
    float l2_acc = 0.0f;
    t0 = wall_clock64();
    for (int i = 0; i < 100; i++) {
        int idx = (tid * 4 + i) % l2_n;
        l2_acc += l2_buf[idx];
    }
    unsigned long long l2_time = wall_clock64() - t0;

    // Exit clock
    unsigned long long exit_clk = wall_clock64();

    if (tid == 0) {
        out_probes[base + 0] = entry;
        out_probes[base + 1] = fma_time;
        out_probes[base + 2] = sfu_time;
        out_probes[base + 3] = lds_time;
        out_probes[base + 4] = l2_time;
        out_probes[base + 5] = exit_clk;
    }

    // Prevent optimization
    if (a == -999.0f && x == -999.0f) l2_buf[0] = lds_acc + l2_acc;
}


// ─── Utility ───

double mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

double stddev(const std::vector<double>& v) {
    double m = mean(v);
    double s = 0;
    for (auto x : v) s += (x - m) * (x - m);
    return sqrt(s / v.size());
}

double correlation(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.size() < 3) return 0;
    double ma = mean(a), mb = mean(b);
    double sa = stddev(a), sb = stddev(b);
    if (sa < 1e-12 || sb < 1e-12) return 0;
    double r = 0;
    for (size_t i = 0; i < a.size(); i++)
        r += (a[i] - ma) * (b[i] - mb);
    return r / (a.size() * sa * sb);
}

// Classification accuracy: can probe_after[i] predict encode_class[i-1]?
double classify_accuracy(const std::vector<int>& classes, const std::vector<double>& probes, int delay=1) {
    if (probes.size() < (size_t)(delay + 10)) return 0.5;

    // Split probe values by previous class
    std::vector<double> when_heavy, when_light;
    for (size_t i = delay; i < classes.size() && i < probes.size(); i++) {
        if (classes[i - delay] == 1) when_heavy.push_back(probes[i]);
        else when_light.push_back(probes[i]);
    }

    if (when_heavy.empty() || when_light.empty()) return 0.5;

    double mean_h = mean(when_heavy), mean_l = mean(when_light);
    double threshold = (mean_h + mean_l) / 2.0;

    // Which direction? Heavy might make probe higher or lower
    int correct_high = 0, total = 0;
    int correct_low = 0;
    for (size_t i = delay; i < classes.size() && i < probes.size(); i++) {
        int prev_class = classes[i - delay];
        double val = probes[i];
        if (prev_class == 1 && val > threshold) correct_high++;
        if (prev_class == 0 && val <= threshold) correct_high++;
        if (prev_class == 1 && val <= threshold) correct_low++;
        if (prev_class == 0 && val > threshold) correct_low++;
        total++;
    }

    return std::max((double)correct_high / total, (double)correct_low / total);
}

// Effect size: Cohen's d between probe-after-heavy and probe-after-light
double effect_size(const std::vector<int>& classes, const std::vector<double>& probes, int delay=1) {
    std::vector<double> when_heavy, when_light;
    for (size_t i = delay; i < classes.size() && i < probes.size(); i++) {
        if (classes[i - delay] == 1) when_heavy.push_back(probes[i]);
        else when_light.push_back(probes[i]);
    }
    if (when_heavy.size() < 5 || when_light.size() < 5) return 0;
    double mh = mean(when_heavy), ml = mean(when_light);
    double sh = stddev(when_heavy), sl = stddev(when_light);
    double pooled = sqrt((sh*sh + sl*sl) / 2.0);
    if (pooled < 1e-12) return 0;
    return fabs(mh - ml) / pooled;
}

// Memory capacity: R² of linear regression probe[t] ~ class[t-d]
double memory_r2(const std::vector<int>& classes, const std::vector<double>& probes, int delay) {
    if (probes.size() < (size_t)(delay + 20)) return 0;
    // Simple linear regression
    double sx = 0, sy = 0, sxy = 0, sx2 = 0, sy2 = 0;
    int n = 0;
    for (size_t i = delay; i < classes.size() && i < probes.size(); i++) {
        double x = (double)classes[i - delay];
        double y = probes[i];
        sx += x; sy += y; sxy += x*y; sx2 += x*x; sy2 += y*y;
        n++;
    }
    if (n < 10) return 0;
    double denom = (n*sx2 - sx*sx) * (n*sy2 - sy*sy);
    if (denom <= 0) return 0;
    double r = (n*sxy - sx*sy) / sqrt(denom);
    return r * r;
}


int main() {
    printf("=== z2259: ENCODE-PROBE Channel Test ===\n");
    printf("Can we encode information into high-ACF hardware channels?\n\n");

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("GPU: %s, CUs: %d, Clock: %d MHz\n\n", prop.name, prop.multiProcessorCount, prop.clockRate/1000);

    const int N_STEPS = 500;   // encode-probe pairs
    const int N_REPS = 5;      // repetitions for statistics

    // Allocate device buffers
    unsigned long long *d_encode_time, *d_probe_out;
    float *d_l2_small, *d_l2_large, *d_global_buf;

    HIP_CHECK(hipMalloc(&d_encode_time, 64 * sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(&d_probe_out, 64 * 6 * sizeof(unsigned long long)));

    int small_n = 4096;    // 16KB in L2
    int large_n = 1048576; // 4MB thrashes L2
    int global_n = 4194304; // 16MB DRAM
    HIP_CHECK(hipMalloc(&d_l2_small, small_n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_l2_large, large_n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_global_buf, global_n * sizeof(float)));

    // Init buffers
    HIP_CHECK(hipMemset(d_l2_small, 0, small_n * sizeof(float)));
    HIP_CHECK(hipMemset(d_l2_large, 0, large_n * sizeof(float)));
    HIP_CHECK(hipMemset(d_global_buf, 0, global_n * sizeof(float)));

    // Host probe buffer
    unsigned long long h_probe[64 * 6];

    // Random input sequence (balanced 50/50)
    srand(42);
    std::vector<int> input_seq(N_STEPS);
    for (int i = 0; i < N_STEPS; i++) input_seq[i] = (i % 2 == 0) ? 1 : 0; // alternating
    // Shuffle for randomness
    for (int i = N_STEPS - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        std::swap(input_seq[i], input_seq[j]);
    }

    const char* channel_names[] = {
        "A_entry_clock", "B_fma_pipe", "C_sfu_pipe",
        "D_lds_bank", "E_dispatch_delta", "F_l2_cache", "G_global_mem"
    };

    // Parameters: heavy/light iteration counts per channel
    // Extreme contrast to maximize encoding
    int heavy_compute = 5000, light_compute = 100;
    int heavy_fma = 5000, light_fma = 100;
    int heavy_sfu = 2000, light_sfu = 50;
    int lds_iters = 500;
    int l2_iters = 200;
    int global_iters = 200;

    printf("Encoding contrast: heavy/light ratios:\n");
    printf("  Compute: %d/%d (%.0fx)\n", heavy_compute, light_compute, (float)heavy_compute/light_compute);
    printf("  FMA:     %d/%d (%.0fx)\n", heavy_fma, light_fma, (float)heavy_fma/light_fma);
    printf("  SFU:     %d/%d (%.0fx)\n", heavy_sfu, light_sfu, (float)heavy_sfu/light_sfu);
    printf("  LDS:     bank-conflict vs sequential (%d iters)\n", lds_iters);
    printf("  L2:      4MB-thrash vs 16KB-cached (%d iters)\n", l2_iters);
    printf("  Global:  random-stride vs sequential (%d iters)\n", global_iters);
    printf("\nSteps: %d, Reps: %d\n\n", N_STEPS, N_REPS);

    // ═══════════════════════════════════════════════════════
    // TEST 1: Per-channel encode-probe
    // For each channel, alternate ENCODE(input-dependent) → PROBE(fixed)
    // Then check if probe readings correlate with previous input
    // ═══════════════════════════════════════════════════════

    printf("═══ TEST 1: Per-Channel Encode → Probe ═══\n\n");

    // We test 7 encoding channels × 6 probe channels = 42 combinations
    // probe channels: entry, fma, sfu, lds, l2, exit

    const char* probe_names[] = {"entry_clk", "fma_time", "sfu_time", "lds_time", "l2_time", "exit_clk"};

    for (int ch = 0; ch < 7; ch++) {
        printf("--- Channel %s ---\n", channel_names[ch]);

        // Collect probe readings across all steps
        std::vector<std::vector<double>> probe_data(6); // 6 probe types

        for (int step = 0; step < N_STEPS; step++) {
            int inp = input_seq[step];

            // ENCODE phase: input-dependent kernel
            switch (ch) {
                case 0: // Compute load
                    k_encode_compute<<<1, 256>>>(inp, heavy_compute, light_compute, d_encode_time);
                    break;
                case 1: // FMA
                    k_encode_fma<<<1, 256>>>(inp, heavy_fma, light_fma, d_encode_time);
                    break;
                case 2: // SFU
                    k_encode_sfu<<<1, 256>>>(inp, heavy_sfu, light_sfu, d_encode_time);
                    break;
                case 3: // LDS
                    k_encode_lds<<<1, 256>>>(inp, lds_iters, d_encode_time);
                    break;
                case 4: // dispatch_delta — encode by varying GRID SIZE
                    if (inp == 1) {
                        k_encode_compute<<<16, 256>>>(inp, 500, 500, d_encode_time); // 16 blocks = heavy dispatch
                    } else {
                        k_encode_compute<<<1, 64>>>(inp, 50, 50, d_encode_time); // 1 small block
                    }
                    break;
                case 5: // L2
                    k_encode_l2<<<4, 256>>>(inp, d_l2_small, d_l2_large, small_n, large_n, l2_iters, d_encode_time);
                    break;
                case 6: // Global
                    k_encode_global<<<4, 256>>>(inp, d_global_buf, global_n, global_iters, d_encode_time);
                    break;
            }
            HIP_CHECK(hipDeviceSynchronize());

            // PROBE phase: FIXED kernel — measures all 6 channels
            k_probe_fixed<<<1, 256>>>(nullptr, d_l2_small, small_n, d_probe_out);
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(h_probe, d_probe_out, 6 * sizeof(unsigned long long), hipMemcpyDeviceToHost));
            for (int p = 0; p < 6; p++) {
                probe_data[p].push_back((double)h_probe[p]);
            }
        }

        // Analyze: for each probe channel, can it predict previous input?
        for (int p = 0; p < 6; p++) {
            double acc = classify_accuracy(input_seq, probe_data[p], 1);
            double d = effect_size(input_seq, probe_data[p], 1);
            double r2 = memory_r2(input_seq, probe_data[p], 1);

            // Also check delay=0 (same-step) as sanity
            double acc0 = classify_accuracy(input_seq, probe_data[p], 0);

            const char* verdict = (acc > 0.55 && d > 0.1) ? "SIGNAL" :
                                  (acc > 0.52) ? "weak" : "none";

            printf("  probe=%s: acc=%.1f%% (d=0: %.1f%%), Cohen_d=%.3f, R²=%.4f  [%s]\n",
                   probe_names[p], acc*100, acc0*100, d, r2, verdict);
        }
        printf("\n");
    }

    // ═══════════════════════════════════════════════════════
    // TEST 2: Multi-step memory — how far back can probe read?
    // Use the best channel (likely FMA encode → FMA probe)
    // ═══════════════════════════════════════════════════════

    printf("═══ TEST 2: Memory Depth (FMA encode, all probes) ═══\n\n");
    {
        std::vector<std::vector<double>> probe_data(6);

        for (int step = 0; step < N_STEPS; step++) {
            int inp = input_seq[step];

            // Always FMA encode with extreme contrast
            k_encode_fma<<<1, 256>>>(inp, 10000, 100, d_encode_time);
            HIP_CHECK(hipDeviceSynchronize());

            k_probe_fixed<<<1, 256>>>(nullptr, d_l2_small, small_n, d_probe_out);
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(h_probe, d_probe_out, 6 * sizeof(unsigned long long), hipMemcpyDeviceToHost));
            for (int p = 0; p < 6; p++)
                probe_data[p].push_back((double)h_probe[p]);
        }

        printf("Delay:  ");
        for (int d = 0; d <= 10; d++) printf("  d=%d   ", d);
        printf("\n");

        for (int p = 0; p < 6; p++) {
            printf("%-10s", probe_names[p]);
            for (int d = 0; d <= 10; d++) {
                double r2 = memory_r2(input_seq, probe_data[p], d);
                printf(" %.4f ", r2);
            }
            printf("\n");
        }
        printf("\n");

        // Also show accuracy at each delay for best probes
        printf("Accuracy at each delay:\n");
        printf("Delay:  ");
        for (int d = 0; d <= 5; d++) printf("  d=%d    ", d);
        printf("\n");
        for (int p = 0; p < 6; p++) {
            printf("%-10s", probe_names[p]);
            for (int d = 0; d <= 5; d++) {
                double acc = classify_accuracy(input_seq, probe_data[p], d);
                printf(" %5.1f%%  ", acc*100);
            }
            printf("\n");
        }
    }

    // ═══════════════════════════════════════════════════════
    // TEST 3: Extreme encoding — ALL channels at once
    // Encode with EVERY channel simultaneously, then probe
    // ═══════════════════════════════════════════════════════

    printf("\n═══ TEST 3: Simultaneous All-Channel Encode ═══\n\n");
    {
        std::vector<std::vector<double>> probe_data(6);

        for (int step = 0; step < N_STEPS; step++) {
            int inp = input_seq[step];

            // Fire ALL encoding channels simultaneously
            if (inp == 1) {
                // HEAVY: massive compute + L2 thrash + LDS conflicts
                k_encode_fma<<<8, 256>>>(1, 5000, 5000, d_encode_time);
                HIP_CHECK(hipDeviceSynchronize());
                k_encode_sfu<<<4, 256>>>(1, 2000, 2000, d_encode_time);
                HIP_CHECK(hipDeviceSynchronize());
                k_encode_l2<<<4, 256>>>(1, d_l2_small, d_l2_large, small_n, large_n, 200, d_encode_time);
                HIP_CHECK(hipDeviceSynchronize());
                k_encode_global<<<4, 256>>>(1, d_global_buf, global_n, 200, d_encode_time);
                HIP_CHECK(hipDeviceSynchronize());
            } else {
                // LIGHT: minimal everything
                k_encode_fma<<<1, 64>>>(0, 100, 100, d_encode_time);
                HIP_CHECK(hipDeviceSynchronize());
            }

            // PROBE
            k_probe_fixed<<<1, 256>>>(nullptr, d_l2_small, small_n, d_probe_out);
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(h_probe, d_probe_out, 6 * sizeof(unsigned long long), hipMemcpyDeviceToHost));
            for (int p = 0; p < 6; p++)
                probe_data[p].push_back((double)h_probe[p]);
        }

        printf("All-channel encode → probe accuracy (delay=1):\n");
        for (int p = 0; p < 6; p++) {
            double acc = classify_accuracy(input_seq, probe_data[p], 1);
            double d = effect_size(input_seq, probe_data[p], 1);
            double r2 = memory_r2(input_seq, probe_data[p], 1);
            printf("  %s: acc=%.1f%%, d=%.3f, R²=%.4f\n", probe_names[p], acc*100, d, r2);
        }

        // Combined feature: all 6 probes → simple threshold vote
        int correct = 0, total = 0;
        for (int i = 1; i < N_STEPS; i++) {
            int votes = 0;
            for (int p = 0; p < 6; p++) {
                // Vote based on whether probe is above/below running median
                double running_sum = 0;
                int cnt = 0;
                for (int j = std::max(0, i-50); j < i; j++) {
                    running_sum += probe_data[p][j];
                    cnt++;
                }
                double med = running_sum / std::max(cnt, 1);
                if (probe_data[p][i] > med) votes++;
            }
            int pred = (votes >= 3) ? 1 : 0;
            // Try both polarities
            if (pred == input_seq[i-1]) correct++;
            total++;
        }
        double vote_acc = std::max((double)correct / total, 1.0 - (double)correct / total);
        printf("  Combined 6-channel vote: %.1f%%\n", vote_acc * 100);
    }

    // ═══════════════════════════════════════════════════════
    // TEST 4: Direct encoding — can encode kernel's OWN timing
    //         carry information to the NEXT encode kernel?
    //         (No separate probe — the encode IS the read)
    // ═══════════════════════════════════════════════════════

    printf("\n═══ TEST 4: Self-Read (encode timing carries previous info) ═══\n\n");
    {
        unsigned long long h_time[2];
        std::vector<double> fma_times, sfu_times, compute_times;

        for (int step = 0; step < N_STEPS; step++) {
            int inp = input_seq[step];

            // FMA encode — but also READS its own timing
            k_encode_fma<<<1, 256>>>(inp, heavy_fma, light_fma, d_encode_time);
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(h_time, d_encode_time, sizeof(unsigned long long), hipMemcpyDeviceToHost));
            fma_times.push_back((double)h_time[0]);

            // SFU encode
            k_encode_sfu<<<1, 256>>>(inp, heavy_sfu, light_sfu, d_encode_time);
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(h_time, d_encode_time, sizeof(unsigned long long), hipMemcpyDeviceToHost));
            sfu_times.push_back((double)h_time[0]);

            // General compute
            k_encode_compute<<<1, 256>>>(inp, heavy_compute, light_compute, d_encode_time);
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(h_time, d_encode_time, sizeof(unsigned long long), hipMemcpyDeviceToHost));
            compute_times.push_back((double)h_time[0]);
        }

        // The encode kernel timing is dominated by iteration count (same-step).
        // But is there a RESIDUAL from the previous step visible in the timing NOISE?
        // Normalize: divide by expected time based on input class
        printf("Self-read: Residual analysis (normalize out iteration count)\n\n");

        auto analyze_residual = [&](const char* name, std::vector<double>& times,
                                     int heavy_n, int light_n) {
            // Compute per-class mean
            double sum_h = 0, sum_l = 0; int n_h = 0, n_l = 0;
            for (size_t i = 0; i < times.size(); i++) {
                if (input_seq[i] == 1) { sum_h += times[i]; n_h++; }
                else { sum_l += times[i]; n_l++; }
            }
            double mean_h = sum_h / n_h, mean_l = sum_l / n_l;

            // Residual = actual - expected
            std::vector<double> residuals(times.size());
            for (size_t i = 0; i < times.size(); i++) {
                double expected = (input_seq[i] == 1) ? mean_h : mean_l;
                residuals[i] = times[i] - expected;
            }

            // Can residual[t] predict class[t-1]?
            double acc = classify_accuracy(input_seq, residuals, 1);
            double d = effect_size(input_seq, residuals, 1);
            double r2 = memory_r2(input_seq, residuals, 1);

            // ACF of residuals
            double acf1 = correlation(
                std::vector<double>(residuals.begin(), residuals.end()-1),
                std::vector<double>(residuals.begin()+1, residuals.end()));

            printf("  %s: mean_H=%.0f mean_L=%.0f ratio=%.1fx\n",
                   name, mean_h, mean_l, mean_h/mean_l);
            printf("    residual: acc=%.1f%%, d=%.3f, R²=%.4f, ACF(1)=%.3f\n",
                   acc*100, d, r2, acf1);

            // Split residuals by PREVIOUS class
            std::vector<double> res_after_heavy, res_after_light;
            for (size_t i = 1; i < residuals.size(); i++) {
                if (input_seq[i-1] == 1) res_after_heavy.push_back(residuals[i]);
                else res_after_light.push_back(residuals[i]);
            }
            double m_ah = mean(res_after_heavy), m_al = mean(res_after_light);
            double s_ah = stddev(res_after_heavy), s_al = stddev(res_after_light);
            printf("    after_heavy: mean=%.1f±%.1f, after_light: mean=%.1f±%.1f, delta=%.1f\n\n",
                   m_ah, s_ah, m_al, s_al, m_ah - m_al);
        };

        analyze_residual("FMA", fma_times, heavy_fma, light_fma);
        analyze_residual("SFU", sfu_times, heavy_sfu, light_sfu);
        analyze_residual("Compute", compute_times, heavy_compute, light_compute);
    }

    // ═══════════════════════════════════════════════════════
    // TEST 5: DVFS exploitation — can extreme workload shift
    //         clock frequency and persist to next kernel?
    // ═══════════════════════════════════════════════════════

    printf("═══ TEST 5: DVFS Clock Shift Encoding ═══\n\n");
    {
        // Idea: heavy kernel (many blocks, long runtime) → GPU heats up → DVFS drops clock
        // Light kernel → GPU cools → DVFS raises clock
        // Probe should see different entry_clock rate
        // Need LONG encode kernels to actually trigger DVFS (~10ms+)

        std::vector<std::vector<double>> probe_data(6);

        printf("Using extreme-duration encode kernels (50K iters) to trigger DVFS...\n");

        for (int step = 0; step < 200; step++) { // fewer steps, heavier kernels
            int inp = input_seq[step];

            if (inp == 1) {
                // MASSIVE: 16 blocks × 256 threads × 50K FMA
                k_encode_fma<<<16, 256>>>(1, 50000, 50000, d_encode_time);
            } else {
                // TINY: 1 block × 64 threads × 100 FMA
                k_encode_fma<<<1, 64>>>(0, 100, 100, d_encode_time);
            }
            HIP_CHECK(hipDeviceSynchronize());

            // Immediate probe
            k_probe_fixed<<<1, 256>>>(nullptr, d_l2_small, small_n, d_probe_out);
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(h_probe, d_probe_out, 6 * sizeof(unsigned long long), hipMemcpyDeviceToHost));
            for (int p = 0; p < 6; p++)
                probe_data[p].push_back((double)h_probe[p]);
        }

        std::vector<int> sub_seq(input_seq.begin(), input_seq.begin() + 200);
        printf("DVFS encode → probe (delay=1):\n");
        for (int p = 0; p < 6; p++) {
            double acc = classify_accuracy(sub_seq, probe_data[p], 1);
            double d = effect_size(sub_seq, probe_data[p], 1);
            printf("  %s: acc=%.1f%%, d=%.3f\n", probe_names[p], acc*100, d);
        }
    }

    // ═══════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════

    printf("\n═══ SUMMARY ═══\n");
    printf("If any channel shows acc > 55%% and Cohen's d > 0.1 at delay=1,\n");
    printf("then that channel can carry input-dependent information across kernel launches.\n");
    printf("This would enable TRUE temporal computing on GPU silicon alone.\n\n");

    // Cleanup
    HIP_CHECK(hipFree(d_encode_time));
    HIP_CHECK(hipFree(d_probe_out));
    HIP_CHECK(hipFree(d_l2_small));
    HIP_CHECK(hipFree(d_l2_large));
    HIP_CHECK(hipFree(d_global_buf));

    printf("Done.\n");
    return 0;
}
