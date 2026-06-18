/*  deep_analog_encode.hip — z2260: Deep Analog Encoding
 *
 *  Three untested mechanisms that might carry input-dependent information:
 *
 *  TEST 1: DRAM Row Buffer Persistence
 *    - Open specific rows with heavy access → next kernel sees row hit/miss latency
 *    - Signal: 20-100× clock cycles (HUGE vs L1's 1-cycle noise)
 *
 *  TEST 2: Single-Kernel Encode-Probe (NO sync gap)
 *    - Encode phase: input-dependent FMA/SFU burst
 *    - Probe phase: IMMEDIATELY after, in SAME kernel, measure timing
 *    - Captures sub-µs voltage droop that sync washes out
 *
 *  TEST 3: Sustained Load Voltage Droop (persistent kernel)
 *    - Long-running kernel with alternating heavy/light phases
 *    - Probes voltage regulator settling dynamics (1-100 µs)
 *    - No kernel boundaries = no sync = pure analog coupling
 *
 *  TEST 4: TLB Persistence
 *    - Encode: access many vs few pages → TLB state differs
 *    - Probe: measure TLB hit/miss timing
 *
 *  TEST 5: Leakage Current Temperature Coupling
 *    - Heavy sustained load → die heats → leakage increases → Vdd droops
 *    - Measure: timing of IDENTICAL instructions changes with thermal history
 *    - Key: accumulate many short bursts vs measure after long burn
 *
 *  TEST 6: Intra-Kernel Voltage Ripple
 *    - Burst of 1000 FMA → immediate timing probe → repeat
 *    - Catches PDN LC ringing at ~10 µs period
 *
 *  Compile: hipcc --offload-arch=gfx1100 -O1 -o scripts/deep_analog_encode scripts/deep_analog_encode.hip
 *  Run:     HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/deep_analog_encode
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
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

// ─── Utility ───
double vec_mean(const std::vector<double>& v) {
    if (v.empty()) return 0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}
double vec_std(const std::vector<double>& v) {
    double m = vec_mean(v);
    double s = 0;
    for (auto x : v) s += (x-m)*(x-m);
    return sqrt(s / std::max((size_t)1, v.size()));
}
double vec_corr(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.size() < 5) return 0;
    double ma = vec_mean(a), mb = vec_mean(b);
    double sa = vec_std(a), sb = vec_std(b);
    if (sa < 1e-12 || sb < 1e-12) return 0;
    double r = 0;
    for (size_t i = 0; i < a.size(); i++) r += (a[i]-ma)*(b[i]-mb);
    return r / (a.size() * sa * sb);
}
double classify_acc(const std::vector<int>& cls, const std::vector<double>& vals, int delay) {
    std::vector<double> wh, wl;
    for (size_t i = delay; i < cls.size() && i < vals.size(); i++) {
        if (cls[i-delay] == 1) wh.push_back(vals[i]); else wl.push_back(vals[i]);
    }
    if (wh.empty() || wl.empty()) return 0.5;
    double thr = (vec_mean(wh) + vec_mean(wl)) / 2.0;
    int c1=0, c2=0, n=0;
    for (size_t i = delay; i < cls.size() && i < vals.size(); i++) {
        int pc = cls[i-delay]; double v = vals[i];
        if ((pc==1 && v>thr) || (pc==0 && v<=thr)) c1++;
        if ((pc==1 && v<=thr) || (pc==0 && v>thr)) c2++;
        n++;
    }
    return std::max((double)c1/n, (double)c2/n);
}
double cohen_d(const std::vector<int>& cls, const std::vector<double>& vals, int delay) {
    std::vector<double> wh, wl;
    for (size_t i = delay; i < cls.size() && i < vals.size(); i++) {
        if (cls[i-delay] == 1) wh.push_back(vals[i]); else wl.push_back(vals[i]);
    }
    if (wh.size()<5 || wl.size()<5) return 0;
    double sp = sqrt((vec_std(wh)*vec_std(wh) + vec_std(wl)*vec_std(wl))/2.0);
    if (sp<1e-12) return 0;
    return fabs(vec_mean(wh) - vec_mean(wl)) / sp;
}

// ═══════════════════════════════════════════════════════════════
// TEST 1: DRAM ROW BUFFER
// Idea: access rows in buf_A (input=1) vs buf_B (input=0)
// Then probe: time access to buf_A rows → should be faster if A was last accessed
// ═══════════════════════════════════════════════════════════════

__global__ void k_dram_encode(int input_class, float* buf_a, float* buf_b,
                               int n_elements, int n_accesses) {
    float acc = 0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Access pattern: stride 2048 floats = 8KB ≈ DRAM row size
    float* buf = (input_class == 1) ? buf_a : buf_b;
    for (int i = 0; i < n_accesses; i++) {
        int idx = (tid * 2048 + i * 2048) % n_elements;
        acc += buf[idx];
    }
    if (acc == -999.0f) buf[0] = acc; // prevent opt
}

__global__ void k_dram_probe(float* buf_a, float* buf_b, int n_elements,
                              unsigned long long* out_times) {
    // Probe BOTH buffers, measure latency difference
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float acc = 0;

    // Probe buf_a (should be fast if input was 1)
    unsigned long long t0 = wall_clock64();
    for (int i = 0; i < 64; i++) {
        int idx = (tid * 2048 + i * 2048) % n_elements;
        acc += buf_a[idx];
    }
    unsigned long long ta = wall_clock64() - t0;

    // Probe buf_b (should be fast if input was 0)
    t0 = wall_clock64();
    for (int i = 0; i < 64; i++) {
        int idx = (tid * 2048 + i * 2048) % n_elements;
        acc += buf_b[idx];
    }
    unsigned long long tb = wall_clock64() - t0;

    if (tid == 0) {
        out_times[0] = ta;
        out_times[1] = tb;
    }
    if (acc == -999.0f) buf_a[0] = acc;
}

// ═══════════════════════════════════════════════════════════════
// TEST 2: SINGLE-KERNEL ENCODE-PROBE (no sync gap)
// Encode and probe happen in SAME kernel launch
// ═══════════════════════════════════════════════════════════════

// Each step: encode(input) → probe → store results
// N_STEPS steps per kernel launch, all in one kernel
__global__ void k_intra_kernel_encode_probe(
    const int* inputs,     // input sequence [N_STEPS]
    unsigned long long* probe_results, // [N_STEPS * 4] = entry, fma, sfu, total
    int n_steps,
    int heavy_iters,
    int light_iters)
{
    int tid = threadIdx.x;

    for (int step = 0; step < n_steps; step++) {
        int inp = inputs[step];
        int iters = (inp == 1) ? heavy_iters : light_iters;

        // ─── ENCODE: input-dependent FMA burst ───
        float a = 1.0001f + (float)tid * 0.00001f;
        float b = 0.9999f, c = 0.0001f;
        for (int i = 0; i < iters; i++) {
            a = __fmaf_rn(a, b, c);
        }

        // ─── TINY GAP: __syncthreads to separate encode from probe ───
        __syncthreads();

        // ─── PROBE: fixed workload, measure timing ───
        unsigned long long entry = wall_clock64();

        // FMA probe: exactly 100 iters
        float pa = 1.0001f + (float)tid * 0.00001f;
        unsigned long long t0 = wall_clock64();
        for (int i = 0; i < 100; i++) pa = __fmaf_rn(pa, b, c);
        unsigned long long fma_t = wall_clock64() - t0;

        // SFU probe: exactly 50 iters
        float sx = 1.0f + (float)tid * 0.001f;
        t0 = wall_clock64();
        for (int i = 0; i < 50; i++) sx = __sinf(sx) + 1.5f;
        unsigned long long sfu_t = wall_clock64() - t0;

        unsigned long long total = wall_clock64() - entry;

        if (tid == 0) {
            int base = step * 4;
            probe_results[base + 0] = entry;
            probe_results[base + 1] = fma_t;
            probe_results[base + 2] = sfu_t;
            probe_results[base + 3] = total;
        }

        __syncthreads();

        // Prevent optimization
        if (a == -999.0f && pa == -999.0f && sx == -999.0f) {
            probe_results[0] = (unsigned long long)(a + pa + sx);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// TEST 3: SUSTAINED LOAD — Persistent kernel with alternating phases
// No kernel boundaries at all. Pure intra-kernel voltage dynamics.
// ═══════════════════════════════════════════════════════════════

__global__ void k_sustained_droop(
    const int* inputs,
    unsigned long long* probe_results, // [N_STEPS * 6]
    int n_steps,
    int encode_iters,    // heavy encode iterations
    int probe_iters,     // fixed probe iterations
    int cooldown_iters)  // light work between steps
{
    int tid = threadIdx.x;
    float b = 0.9999f, c = 0.0001f;

    for (int step = 0; step < n_steps; step++) {
        int inp = inputs[step];

        // ─── ENCODE: Variable-intensity burst ───
        float a = 1.0001f + (float)tid * 0.00001f;
        int enc_n = (inp == 1) ? encode_iters : (encode_iters / 50);
        for (int i = 0; i < enc_n; i++) a = __fmaf_rn(a, b, c);

        // ─── IMMEDIATE PROBE (no sync!) ───
        // Probe at 3 different delays to catch LC ringing
        unsigned long long t0, t1;

        // Probe 0: immediate
        float p0 = 1.0001f;
        t0 = wall_clock64();
        for (int i = 0; i < probe_iters; i++) p0 = __fmaf_rn(p0, b, c);
        t1 = wall_clock64();
        unsigned long long imm = t1 - t0;

        // Probe 1: after ~100 light iters (~50ns gap)
        float gap1 = 1.0f;
        for (int i = 0; i < 100; i++) gap1 += 0.001f;
        float p1 = 1.0001f;
        t0 = wall_clock64();
        for (int i = 0; i < probe_iters; i++) p1 = __fmaf_rn(p1, b, c);
        t1 = wall_clock64();
        unsigned long long short_gap = t1 - t0;

        // Probe 2: after ~1000 light iters (~500ns gap)
        float gap2 = 1.0f;
        for (int i = 0; i < 1000; i++) gap2 += 0.001f;
        float p2 = 1.0001f;
        t0 = wall_clock64();
        for (int i = 0; i < probe_iters; i++) p2 = __fmaf_rn(p2, b, c);
        t1 = wall_clock64();
        unsigned long long med_gap = t1 - t0;

        // Probe 3: after ~10000 light iters (~5µs gap)
        float gap3 = 1.0f;
        for (int i = 0; i < 10000; i++) gap3 += 0.001f;
        float p3 = 1.0001f;
        t0 = wall_clock64();
        for (int i = 0; i < probe_iters; i++) p3 = __fmaf_rn(p3, b, c);
        t1 = wall_clock64();
        unsigned long long long_gap = t1 - t0;

        // Probe 4: SFU at immediate
        float s0 = 1.0f + (float)tid * 0.001f;
        t0 = wall_clock64();
        for (int i = 0; i < 50; i++) s0 = __sinf(s0) + 1.5f;
        t1 = wall_clock64();
        unsigned long long sfu_imm = t1 - t0;

        // Probe 5: wall_clock delta (PLL phase)
        t0 = wall_clock64();
        // tiny gap
        for (volatile int i = 0; i < 10; i++) {}
        t1 = wall_clock64();
        unsigned long long pll = t1 - t0;

        if (tid == 0) {
            int base = step * 6;
            probe_results[base + 0] = imm;
            probe_results[base + 1] = short_gap;
            probe_results[base + 2] = med_gap;
            probe_results[base + 3] = long_gap;
            probe_results[base + 4] = sfu_imm;
            probe_results[base + 5] = pll;
        }

        // Cooldown between steps
        float cool = 1.0f;
        for (int i = 0; i < cooldown_iters; i++) cool += 0.001f;

        if (a == -999.0f && p0+p1+p2+p3+s0+gap1+gap2+gap3+cool == -1.0f)
            probe_results[0] = 0;
    }
}

// ═══════════════════════════════════════════════════════════════
// TEST 4: TLB PERSISTENCE
// ═══════════════════════════════════════════════════════════════

__global__ void k_tlb_encode(int input_class, char* buf, long buf_size, int n_pages) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float acc = 0;
    if (input_class == 1) {
        // Touch MANY pages (64KB stride = 1 page on most configs)
        for (int i = 0; i < n_pages; i++) {
            long offset = ((long)tid * 65536 + (long)i * 65536) % buf_size;
            acc += (float)buf[offset];
        }
    } else {
        // Touch FEW pages (stay in 1 page)
        for (int i = 0; i < n_pages; i++) {
            long offset = (tid * 4 + i * 4) % 4096;
            acc += (float)buf[offset];
        }
    }
    if (acc == -999.0f) buf[0] = (char)acc;
}

__global__ void k_tlb_probe(char* buf, long buf_size, int n_pages,
                             unsigned long long* out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float acc = 0;

    // Probe: access SPECIFIC set of pages and time it
    unsigned long long t0 = wall_clock64();
    for (int i = 0; i < n_pages; i++) {
        long offset = ((long)tid * 65536 + (long)i * 65536) % buf_size;
        acc += (float)buf[offset];
    }
    unsigned long long t1 = wall_clock64();

    if (tid == 0) out[0] = t1 - t0;
    if (acc == -999.0f) buf[0] = (char)acc;
}

// ═══════════════════════════════════════════════════════════════
// TEST 5: THERMAL LEAKAGE COUPLING
// Heavy sustained load → die heats → leakage current increases →
// → less voltage headroom → timing slows down
// Use MANY sequential kernels to build thermal mass, then probe
// ═══════════════════════════════════════════════════════════════

__global__ void k_thermal_burn(int n_iters) {
    float a = 1.0001f + (float)threadIdx.x * 0.00001f;
    float b = 0.9999f, c = 0.0001f;
    for (int i = 0; i < n_iters; i++) a = __fmaf_rn(a, b, c);
    if (a == -999.0f) printf("x");
}

__global__ void k_thermal_probe(unsigned long long* out) {
    float a = 1.0001f + (float)threadIdx.x * 0.00001f;
    float b = 0.9999f, c = 0.0001f;
    unsigned long long t0 = wall_clock64();
    for (int i = 0; i < 200; i++) a = __fmaf_rn(a, b, c);
    unsigned long long t1 = wall_clock64();
    if (threadIdx.x == 0) out[0] = t1 - t0;
    if (a == -999.0f) out[1] = (unsigned long long)a;
}

// ═══════════════════════════════════════════════════════════════
// TEST 6: VOLTAGE RIPPLE PROBING (intra-kernel LC resonance)
// Burst → probe → burst → probe → ... in rapid succession
// Catches PDN ringing at different phases
// ═══════════════════════════════════════════════════════════════

__global__ void k_ripple_probe(
    const int* inputs,
    unsigned long long* results, // [N_STEPS * N_DELAYS]
    int n_steps, int n_delays,
    int burst_iters)
{
    int tid = threadIdx.x;
    float b = 0.9999f, c = 0.0001f;

    for (int step = 0; step < n_steps; step++) {
        int inp = inputs[step];
        int enc_n = (inp == 1) ? burst_iters : (burst_iters / 20);

        // BURST
        float a = 1.0001f + (float)tid * 0.00001f;
        for (int i = 0; i < enc_n; i++) a = __fmaf_rn(a, b, c);

        // Probe at N_DELAYS different gaps
        for (int d = 0; d < n_delays; d++) {
            // Gap: 2^d iterations of light work (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
            int gap_n = 1 << d;
            float g = 1.0f;
            for (int i = 0; i < gap_n; i++) g += 0.001f;

            // Timing probe
            float p = 1.0001f;
            unsigned long long t0 = wall_clock64();
            for (int i = 0; i < 100; i++) p = __fmaf_rn(p, b, c);
            unsigned long long dt = wall_clock64() - t0;

            if (tid == 0) results[step * n_delays + d] = dt;

            if (a+g+p == -999.0f) results[0] = 0;
        }
    }
}


int main() {
    printf("=== z2260: Deep Analog Encoding ===\n");
    printf("Testing DRAM row buffers, intra-kernel droop, TLB, thermal leakage, voltage ripple\n\n");

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("GPU: %s, CUs: %d, Clock: %d MHz\n\n", prop.name, prop.multiProcessorCount, prop.clockRate/1000);

    const int N_STEPS = 500;

    // Generate random balanced input
    srand(42);
    std::vector<int> input_seq(N_STEPS);
    for (int i = 0; i < N_STEPS; i++) input_seq[i] = i % 2;
    for (int i = N_STEPS-1; i > 0; i--) { int j = rand()%(i+1); std::swap(input_seq[i], input_seq[j]); }

    int* d_inputs;
    HIP_CHECK(hipMalloc(&d_inputs, N_STEPS * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_inputs, input_seq.data(), N_STEPS * sizeof(int), hipMemcpyHostToDevice));

    unsigned long long* d_out;
    HIP_CHECK(hipMalloc(&d_out, N_STEPS * 16 * sizeof(unsigned long long)));

    // ═══════════════════════════════════════════════
    // TEST 1: DRAM ROW BUFFER
    // ═══════════════════════════════════════════════
    printf("═══ TEST 1: DRAM Row Buffer Persistence ═══\n\n");
    {
        int n_elem = 4 * 1024 * 1024; // 16MB per buffer
        float *d_buf_a, *d_buf_b;
        HIP_CHECK(hipMalloc(&d_buf_a, n_elem * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_buf_b, n_elem * sizeof(float)));
        HIP_CHECK(hipMemset(d_buf_a, 0, n_elem * sizeof(float)));
        HIP_CHECK(hipMemset(d_buf_b, 0, n_elem * sizeof(float)));

        unsigned long long h_times[4];
        std::vector<double> diff_vals; // ta - tb: positive = buf_a slower = wasn't cached

        for (int step = 0; step < N_STEPS; step++) {
            int inp = input_seq[step];

            k_dram_encode<<<4, 256>>>(inp, d_buf_a, d_buf_b, n_elem, 256);
            HIP_CHECK(hipDeviceSynchronize());

            k_dram_probe<<<1, 256>>>(d_buf_a, d_buf_b, n_elem, d_out);
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(h_times, d_out, 2*sizeof(unsigned long long), hipMemcpyDeviceToHost));
            diff_vals.push_back((double)h_times[0] - (double)h_times[1]); // ta - tb
        }

        double acc1 = classify_acc(input_seq, diff_vals, 1);
        double d1 = cohen_d(input_seq, diff_vals, 1);
        double acc0 = classify_acc(input_seq, diff_vals, 0);
        double d0 = cohen_d(input_seq, diff_vals, 0);

        // Split by class
        std::vector<double> after_h, after_l;
        for (size_t i = 1; i < diff_vals.size(); i++) {
            if (input_seq[i-1]==1) after_h.push_back(diff_vals[i]);
            else after_l.push_back(diff_vals[i]);
        }

        printf("DRAM row buffer (ta-tb):\n");
        printf("  Same-step:  acc=%.1f%%, d=%.3f\n", acc0*100, d0);
        printf("  Delay=1:    acc=%.1f%%, d=%.3f\n", acc1*100, d1);
        printf("  After heavy: mean=%.1f±%.1f\n", vec_mean(after_h), vec_std(after_h));
        printf("  After light: mean=%.1f±%.1f\n", vec_mean(after_l), vec_std(after_l));
        printf("  Delta: %.1f clk\n\n", vec_mean(after_h) - vec_mean(after_l));

        HIP_CHECK(hipFree(d_buf_a));
        HIP_CHECK(hipFree(d_buf_b));
    }

    // ═══════════════════════════════════════════════
    // TEST 2: INTRA-KERNEL ENCODE-PROBE
    // ═══════════════════════════════════════════════
    printf("═══ TEST 2: Intra-Kernel Encode-Probe (NO sync gap) ═══\n\n");
    {
        unsigned long long* d_probes;
        HIP_CHECK(hipMalloc(&d_probes, N_STEPS * 4 * sizeof(unsigned long long)));

        // Try different contrast ratios
        int contrasts[][2] = {{5000,100}, {2000,100}, {10000,50}, {1000,10}};
        for (int ci = 0; ci < 4; ci++) {
            int heavy = contrasts[ci][0], light = contrasts[ci][1];

            k_intra_kernel_encode_probe<<<1, 256>>>(
                d_inputs, d_probes, N_STEPS, heavy, light);
            HIP_CHECK(hipDeviceSynchronize());

            std::vector<unsigned long long> h_probes(N_STEPS * 4);
            HIP_CHECK(hipMemcpy(h_probes.data(), d_probes, N_STEPS*4*sizeof(unsigned long long), hipMemcpyDeviceToHost));

            // Extract probe channels
            std::vector<double> entry(N_STEPS), fma(N_STEPS), sfu(N_STEPS), total(N_STEPS);
            for (int i = 0; i < N_STEPS; i++) {
                entry[i] = (double)h_probes[i*4+0];
                fma[i] = (double)h_probes[i*4+1];
                sfu[i] = (double)h_probes[i*4+2];
                total[i] = (double)h_probes[i*4+3];
            }

            printf("  Contrast %d/%d (%.0fx):\n", heavy, light, (float)heavy/light);

            const char* pn[] = {"entry_clk", "fma_time", "sfu_time", "total"};
            std::vector<double>* pvs[] = {&entry, &fma, &sfu, &total};
            for (int p = 0; p < 4; p++) {
                double a0 = classify_acc(input_seq, *pvs[p], 0);
                double a1 = classify_acc(input_seq, *pvs[p], 1);
                double d0 = cohen_d(input_seq, *pvs[p], 0);
                double d1 = cohen_d(input_seq, *pvs[p], 1);

                // ACF of probe values
                std::vector<double>& v = *pvs[p];
                double acf = vec_corr(
                    std::vector<double>(v.begin(), v.end()-1),
                    std::vector<double>(v.begin()+1, v.end()));

                const char* tag = (a1 > 0.55 && d1 > 0.1) ? " ★SIGNAL★" :
                                  (a1 > 0.52) ? " ~weak~" : "";
                printf("    %s: d=0 acc=%.1f%% d=%.3f | d=1 acc=%.1f%% d=%.3f ACF=%.3f%s\n",
                       pn[p], a0*100, d0, a1*100, d1, acf, tag);
            }
            printf("\n");
        }
        HIP_CHECK(hipFree(d_probes));
    }

    // ═══════════════════════════════════════════════
    // TEST 3: SUSTAINED VOLTAGE DROOP
    // ═══════════════════════════════════════════════
    printf("═══ TEST 3: Sustained Voltage Droop (persistent kernel) ═══\n\n");
    {
        unsigned long long* d_probes;
        HIP_CHECK(hipMalloc(&d_probes, N_STEPS * 6 * sizeof(unsigned long long)));

        // Try different encode intensities and probe delays
        int configs[][3] = {
            {5000, 200, 100},    // moderate encode, standard probe, light cooldown
            {20000, 200, 100},   // heavy encode
            {5000, 200, 5000},   // heavy cooldown (more gap)
            {50000, 200, 100},   // extreme encode
        };
        const char* cfg_names[] = {"moderate", "heavy", "long_cool", "extreme"};

        for (int ci = 0; ci < 4; ci++) {
            k_sustained_droop<<<1, 256>>>(
                d_inputs, d_probes, std::min(N_STEPS, 300),
                configs[ci][0], configs[ci][1], configs[ci][2]);
            HIP_CHECK(hipDeviceSynchronize());

            int ns = std::min(N_STEPS, 300);
            std::vector<unsigned long long> h_p(ns * 6);
            HIP_CHECK(hipMemcpy(h_p.data(), d_probes, ns*6*sizeof(unsigned long long), hipMemcpyDeviceToHost));

            const char* dn[] = {"immediate", "50ns_gap", "500ns_gap", "5us_gap", "sfu_imm", "pll_phase"};
            printf("  Config: %s (encode=%d, probe=%d, cool=%d)\n",
                   cfg_names[ci], configs[ci][0], configs[ci][1], configs[ci][2]);

            for (int d = 0; d < 6; d++) {
                std::vector<double> vals(ns);
                for (int i = 0; i < ns; i++) vals[i] = (double)h_p[i*6+d];

                double a0 = classify_acc(input_seq, vals, 0);
                double a1 = classify_acc(input_seq, vals, 1);
                double cd0 = cohen_d(input_seq, vals, 0);
                double cd1 = cohen_d(input_seq, vals, 1);

                const char* tag = (a0 > 0.55 && cd0 > 0.2) ? " ★SAME-STEP★" :
                                  (a1 > 0.55 && cd1 > 0.1) ? " ★CROSS-STEP★" : "";
                printf("    %s: d=0 %.1f%% (d=%.3f) | d=1 %.1f%% (d=%.3f)%s\n",
                       dn[d], a0*100, cd0, a1*100, cd1, tag);
            }
            printf("\n");
        }
        HIP_CHECK(hipFree(d_probes));
    }

    // ═══════════════════════════════════════════════
    // TEST 4: TLB PERSISTENCE
    // ═══════════════════════════════════════════════
    printf("═══ TEST 4: TLB Persistence ═══\n\n");
    {
        long buf_size = 256L * 1024 * 1024; // 256MB to span many TLB entries
        char* d_tlb_buf;
        HIP_CHECK(hipMalloc(&d_tlb_buf, buf_size));
        HIP_CHECK(hipMemset(d_tlb_buf, 0, buf_size));

        std::vector<double> probe_times;

        for (int step = 0; step < N_STEPS; step++) {
            int inp = input_seq[step];

            k_tlb_encode<<<4, 256>>>(inp, d_tlb_buf, buf_size, 512);
            HIP_CHECK(hipDeviceSynchronize());

            k_tlb_probe<<<1, 256>>>(d_tlb_buf, buf_size, 256, d_out);
            HIP_CHECK(hipDeviceSynchronize());

            unsigned long long h_t;
            HIP_CHECK(hipMemcpy(&h_t, d_out, sizeof(unsigned long long), hipMemcpyDeviceToHost));
            probe_times.push_back((double)h_t);
        }

        double acc0 = classify_acc(input_seq, probe_times, 0);
        double acc1 = classify_acc(input_seq, probe_times, 1);
        double d0 = cohen_d(input_seq, probe_times, 0);
        double d1 = cohen_d(input_seq, probe_times, 1);

        std::vector<double> after_h, after_l;
        for (size_t i = 1; i < probe_times.size(); i++) {
            if (input_seq[i-1]==1) after_h.push_back(probe_times[i]);
            else after_l.push_back(probe_times[i]);
        }

        printf("TLB probe timing:\n");
        printf("  Same-step: acc=%.1f%%, d=%.3f\n", acc0*100, d0);
        printf("  Delay=1:   acc=%.1f%%, d=%.3f\n", acc1*100, d1);
        printf("  After many-page encode: mean=%.1f±%.1f\n", vec_mean(after_h), vec_std(after_h));
        printf("  After few-page encode:  mean=%.1f±%.1f\n", vec_mean(after_l), vec_std(after_l));
        printf("  Delta: %.1f clk\n\n", vec_mean(after_h) - vec_mean(after_l));

        HIP_CHECK(hipFree(d_tlb_buf));
    }

    // ═══════════════════════════════════════════════
    // TEST 5: THERMAL LEAKAGE ACCUMULATION
    // ═══════════════════════════════════════════════
    printf("═══ TEST 5: Thermal Leakage Coupling ═══\n\n");
    {
        // Burn → probe → burn → probe, varying burn intensity
        std::vector<double> probe_cold, probe_hot;

        printf("Phase 1: Cold probes (no burn)...\n");
        for (int i = 0; i < 50; i++) {
            k_thermal_probe<<<1, 256>>>(d_out);
            HIP_CHECK(hipDeviceSynchronize());
            unsigned long long h_t;
            HIP_CHECK(hipMemcpy(&h_t, d_out, sizeof(unsigned long long), hipMemcpyDeviceToHost));
            probe_cold.push_back((double)h_t);
        }

        printf("Phase 2: Heavy burn (20 × 16-block × 100K FMA each)...\n");
        for (int i = 0; i < 20; i++) {
            k_thermal_burn<<<16, 256>>>(100000);
            HIP_CHECK(hipDeviceSynchronize());
        }

        printf("Phase 3: Hot probes (after burn)...\n");
        for (int i = 0; i < 50; i++) {
            k_thermal_probe<<<1, 256>>>(d_out);
            HIP_CHECK(hipDeviceSynchronize());
            unsigned long long h_t;
            HIP_CHECK(hipMemcpy(&h_t, d_out, sizeof(unsigned long long), hipMemcpyDeviceToHost));
            probe_hot.push_back((double)h_t);
        }

        printf("  Cold FMA timing: mean=%.1f±%.1f\n", vec_mean(probe_cold), vec_std(probe_cold));
        printf("  Hot FMA timing:  mean=%.1f±%.1f\n", vec_mean(probe_hot), vec_std(probe_hot));
        double delta = vec_mean(probe_hot) - vec_mean(probe_cold);
        double pct = delta / vec_mean(probe_cold) * 100.0;
        printf("  Delta: %.1f clk (%.2f%%)\n", delta, pct);
        printf("  %s\n\n", fabs(pct) > 0.5 ? "★ THERMAL LEAKAGE VISIBLE ★" :
                           fabs(pct) > 0.1 ? "~ weak thermal signal ~" :
                           "no thermal effect");

        // Decay curve: how fast does it recover?
        printf("  Decay curve (probes after burn):\n");
        for (int i = 0; i < std::min(20, (int)probe_hot.size()); i++) {
            printf("    t+%d: %.1f clk (%.2f%% vs cold)\n", i,
                   probe_hot[i], (probe_hot[i] - vec_mean(probe_cold))/vec_mean(probe_cold)*100);
        }
    }

    // ═══════════════════════════════════════════════
    // TEST 6: VOLTAGE RIPPLE (LC RINGING)
    // ═══════════════════════════════════════════════
    printf("\n═══ TEST 6: Voltage Ripple / LC Ringing ═══\n\n");
    {
        const int N_DELAYS = 10; // 2^0 to 2^9 gap iterations
        unsigned long long* d_ripple;
        HIP_CHECK(hipMalloc(&d_ripple, N_STEPS * N_DELAYS * sizeof(unsigned long long)));

        k_ripple_probe<<<1, 256>>>(d_inputs, d_ripple, N_STEPS, N_DELAYS, 2000);
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<unsigned long long> h_rip(N_STEPS * N_DELAYS);
        HIP_CHECK(hipMemcpy(h_rip.data(), d_ripple, N_STEPS*N_DELAYS*sizeof(unsigned long long), hipMemcpyDeviceToHost));

        printf("Ripple probe at different delays after burst:\n");
        printf("  %-12s %8s %8s %8s %8s\n", "Gap (iters)", "acc_d=0", "d_d=0", "acc_d=1", "d_d=1");

        for (int d = 0; d < N_DELAYS; d++) {
            std::vector<double> vals(N_STEPS);
            for (int i = 0; i < N_STEPS; i++) vals[i] = (double)h_rip[i*N_DELAYS + d];

            double a0 = classify_acc(input_seq, vals, 0);
            double a1 = classify_acc(input_seq, vals, 1);
            double cd0 = cohen_d(input_seq, vals, 0);
            double cd1 = cohen_d(input_seq, vals, 1);

            const char* tag = (a0 > 0.55 && cd0 > 0.2) ? " ★" : "";
            printf("  2^%-3d = %-5d %6.1f%% %8.3f %6.1f%% %8.3f%s\n",
                   d, 1<<d, a0*100, cd0, a1*100, cd1, tag);
        }
        HIP_CHECK(hipFree(d_ripple));
    }

    printf("\n═══ DONE ═══\n");

    HIP_CHECK(hipFree(d_inputs));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
