// jitter_substrate.hip — Ultra-low-level analog substrate noise probes
// Goes BELOW memory contention: transistor timing, SRAM charge, pipeline jitter,
// denormal arithmetic, register file noise, voltage ripple via timing.
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_substrate scripts/jitter_substrate.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_substrate

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <vector>

#define ITERS 4000
#define LANES 64

// ============================================================================
// TEST 1: Bare clock64() floor noise
// How much jitter exists in JUST reading the clock twice in succession?
// This measures the pipeline's own timing granularity.
// ============================================================================
__global__ void bare_clock_noise(uint64_t* out, int iters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= LANES) return;

    for (int i = 0; i < iters; i++) {
        uint64_t t0 = __clock64();
        // Nothing between — pure clock readback latency
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
    }
}

// ============================================================================
// TEST 2: Single ALU instruction jitter
// Measure timing of ONE arithmetic op. The variance reveals pipeline noise.
// ============================================================================
__global__ void single_alu_jitter(uint64_t* out, int iters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= LANES) return;

    float x = 1.0f + tid * 0.001f;  // prevent optimization
    for (int i = 0; i < iters; i++) {
        uint64_t t0 = __clock64();
        x = x * 1.000001f;  // single FMUL
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
        // Use x to prevent dead code elimination
        if (x == -999.0f) out[0] = 0;
    }
}

// ============================================================================
// TEST 3: Denormal arithmetic — operations near the float precision boundary
// Denormals cause microcode traps on some GPUs → huge timing variation
// ============================================================================
__global__ void denormal_jitter(uint64_t* out, float* scratch, int iters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= LANES) return;

    // Start near denormal boundary
    float x = 1.0e-38f * (1.0f + tid * 0.01f);

    for (int i = 0; i < iters; i++) {
        uint64_t t0 = __clock64();
        x = x * 0.5f;  // push toward denormal
        x = x + 1.0e-45f;  // add smallest denormal
        x = x * 2.0f;  // pull back
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;

        // Reset if too small (avoid flush to zero dominating)
        if (x < 1.0e-44f) x = 1.0e-38f * (1.0f + tid * 0.01f);
        scratch[tid] = x;
    }
}

// ============================================================================
// TEST 4: Normal vs denormal comparison
// Same operations but in normal float range — baseline for TEST 3
// ============================================================================
__global__ void normal_arith_jitter(uint64_t* out, float* scratch, int iters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= LANES) return;

    float x = 1.0f + tid * 0.001f;

    for (int i = 0; i < iters; i++) {
        uint64_t t0 = __clock64();
        x = x * 0.5f;
        x = x + 1.0e-7f;
        x = x * 2.0f;
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;

        if (x < 0.5f) x = 1.0f + tid * 0.001f;
        scratch[tid] = x;
    }
}

// ============================================================================
// TEST 5: Register pressure sweep
// More live registers → more spilling → timing variation from register file access
// ============================================================================
__global__ void reg_pressure_low(uint64_t* out, int iters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= LANES) return;

    float a = 1.0f + tid * 0.001f;

    for (int i = 0; i < iters; i++) {
        uint64_t t0 = __clock64();
        a = a * 1.000001f;
        a = a + 0.000001f;
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
        if (a > 100.0f) a = 1.0f;
    }
}

__global__ void reg_pressure_high(uint64_t* out, int iters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= LANES) return;

    // 32 live float registers
    float a=1.0f+tid*0.001f, b=2.0f, c=3.0f, d=4.0f;
    float e=5.0f, f=6.0f, g=7.0f, h=8.0f;
    float j=9.0f, k=10.0f, l=11.0f, m=12.0f;
    float n=13.0f, o=14.0f, p=15.0f, q=16.0f;
    float r=17.0f, s=18.0f, t=19.0f, u=20.0f;
    float v=21.0f, w=22.0f, x=23.0f, y=24.0f;
    float z=25.0f, aa=26.0f, bb=27.0f, cc=28.0f;
    float dd=29.0f, ee=30.0f, ff=31.0f, gg=32.0f;

    for (int i = 0; i < iters; i++) {
        uint64_t t0 = __clock64();
        a=a*b+c; b=b*c+d; c=c*d+e; d=d*e+f;
        e=e*f+g; f=f*g+h; g=g*h+j; h=h*j+k;
        j=j*k+l; k=k*l+m; l=l*m+n; m=m*n+o;
        n=n*o+p; o=o*p+q; p=p*q+r; q=q*r+s;
        r=r*s+t; s=s*t+u; t=t*u+v; u=u*v+w;
        v=v*w+x; w=w*x+y; x=x*y+z; y=y*z+aa;
        z=z*aa+bb; aa=aa*bb+cc; bb=bb*cc+dd; cc=cc*dd+ee;
        dd=dd*ee+ff; ee=ee*ff+gg; ff=ff*gg+a; gg=gg*a+b;
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
        // Prevent overflow
        if (a!=a || fabsf(a)>1e30f) { a=1.0f+tid*0.001f; b=2.0f; c=3.0f; d=4.0f; }
    }
}

// ============================================================================
// TEST 6: Same-address repeated read timing
// Read same L1-cached word repeatedly. Timing variation = SRAM charge noise +
// power rail ripple coupling into cache sense amplifiers.
// ============================================================================
__global__ void sram_read_noise(uint64_t* out, uint32_t* data, int iters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= LANES) return;

    // Prime cache line
    uint32_t val = data[tid];

    for (int i = 0; i < iters; i++) {
        uint64_t t0 = __clock64();
        val = data[tid];  // Same address, should be L1 hit every time
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
        // Use val to prevent dead code elimination
        if (val == 0xDEADDEAD) out[0] = 0;
    }
}

// ============================================================================
// TEST 7: LDS read noise — intra-CU shared memory timing
// Even more local than L1 — this is on-die SRAM within the CU itself.
// ============================================================================
__global__ void lds_read_noise(uint64_t* out, int iters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lid = threadIdx.x;
    if (tid >= LANES) return;

    __shared__ uint32_t lds_data[64];
    lds_data[lid] = lid + 42;
    __syncthreads();

    for (int i = 0; i < iters; i++) {
        uint64_t t0 = __clock64();
        uint32_t val = lds_data[lid];  // Same LDS address, no conflict
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
        if (val == 0xDEADDEAD) out[0] = 0;
    }
}

// ============================================================================
// TEST 8: Integer vs float pipeline jitter comparison
// Different execution units may have different noise characteristics.
// ============================================================================
__global__ void int_pipeline_jitter(uint64_t* out, int iters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= LANES) return;

    uint32_t x = tid + 1;
    for (int i = 0; i < iters; i++) {
        uint64_t t0 = __clock64();
        x = x * 31 + 17;  // Integer MUL + ADD
        x = x ^ (x >> 7);
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
        if (x == 0) x = 1;
    }
}

__global__ void float_pipeline_jitter(uint64_t* out, int iters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= LANES) return;

    float x = 1.0f + tid * 0.001f;
    for (int i = 0; i < iters; i++) {
        uint64_t t0 = __clock64();
        x = x * 1.000001f + 0.000001f;  // Float MUL + ADD
        x = sqrtf(x);  // SQRT — complex FP op
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
        if (x < 0.5f || x != x) x = 1.0f + tid * 0.001f;
    }
}

// ============================================================================
// TEST 9: Transcendental function jitter
// sin/cos/exp use microcode sequences — data-dependent timing!
// ============================================================================
__global__ void transcendental_jitter(uint64_t* out, int iters, int func_id) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= LANES) return;

    float x = 0.5f + tid * 0.01f;
    for (int i = 0; i < iters; i++) {
        uint64_t t0 = __clock64();
        if (func_id == 0) x = sinf(x);
        else if (func_id == 1) x = expf(x * 0.01f);
        else if (func_id == 2) x = logf(fabsf(x) + 1.0f);
        else if (func_id == 3) x = __frcp_rn(x + 1.0f);  // reciprocal
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
        if (x != x || fabsf(x) > 1e6f) x = 0.5f + tid * 0.01f;
    }
}

// ============================================================================
// TEST 10: Wavefront divergence timing
// Divergent branches cause SIMD masking — some lanes execute, others wait.
// The timing pattern reveals the branch prediction / masking noise.
// ============================================================================
__global__ void divergence_jitter(uint64_t* out, int iters, int div_ratio) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lid = threadIdx.x;
    if (tid >= LANES) return;

    float x = 1.0f + tid * 0.001f;
    for (int i = 0; i < iters; i++) {
        uint64_t t0 = __clock64();
        if (lid % div_ratio == 0) {
            x = x * 1.5f + 0.1f;
            x = x * 0.9f;
        } else {
            x = x + 0.001f;
        }
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
        if (x > 1e6f || x != x) x = 1.0f + tid * 0.001f;
    }
}

// ============================================================================
// TEST 11: Power supply ripple correlation
// Run a known-intensity workload, measure timing of a separate lightweight
// operation. If power rail noise couples into timing, the heavy workload
// will modulate the light measurement.
// ============================================================================
__global__ void power_ripple_probe(uint64_t* out_heavy, uint64_t* out_light,
                                    float* scratch, int iters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lid = threadIdx.x;
    if (tid >= LANES) return;

    float x = 1.0f + tid * 0.001f;

    for (int i = 0; i < iters; i++) {
        // Phase 1: Heavy computation (creates voltage droop)
        uint64_t t0 = __clock64();
        for (int j = 0; j < 100; j++) {
            x = fmaf(x, 1.000001f, 0.000001f);
        }
        uint64_t t1 = __clock64();
        out_heavy[tid * iters + i] = t1 - t0;

        // Phase 2: Lightweight read right after (measures recovery)
        uint64_t t2 = __clock64();
        float val = scratch[tid];  // Single L1 read
        uint64_t t3 = __clock64();
        out_light[tid * iters + i] = t3 - t2;

        scratch[tid] = x + val;
    }
}

// ============================================================================
// TEST 12: Bit-level float noise
// Multiply same value, check if LSB of result varies.
// This probes whether the FPU produces bit-exact results or has analog noise.
// ============================================================================
__global__ void float_bit_noise(uint32_t* bit_results, int iters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= LANES) return;

    float base = 1.0f + tid * 0.001f;
    uint32_t last_bits = 0;
    uint32_t flip_count = 0;

    for (int i = 0; i < iters; i++) {
        // Same computation every iteration
        float result = base * 3.14159265f;
        result = sqrtf(result);
        result = result * result;  // Should recover base*pi... almost

        uint32_t bits;
        memcpy(&bits, &result, 4);

        if (i > 0 && bits != last_bits) {
            flip_count++;
        }
        last_bits = bits;
        bit_results[tid * iters + i] = bits;
    }
    // Store flip count in first element
    if (tid == 0) bit_results[LANES * iters] = flip_count;
}

// ============================================================================
// Helpers
// ============================================================================
struct Stats {
    double mean, std, min_v, max_v, cv;
    uint64_t mode;
};

Stats compute_stats(uint64_t* data, int n) {
    Stats s;
    // Sort for mode finding
    std::vector<uint64_t> sorted(data, data + n);
    std::sort(sorted.begin(), sorted.end());

    double sum = 0, sum2 = 0;
    s.min_v = sorted[0];
    s.max_v = sorted[n-1];

    for (int i = 0; i < n; i++) {
        sum += sorted[i];
        sum2 += (double)sorted[i] * sorted[i];
    }
    s.mean = sum / n;
    s.std = sqrt(sum2/n - s.mean*s.mean);
    s.cv = s.mean > 0 ? s.std / s.mean : 0;

    // Mode (most common value)
    int best_count = 0;
    s.mode = sorted[0];
    int count = 1;
    for (int i = 1; i < n; i++) {
        if (sorted[i] == sorted[i-1]) {
            count++;
        } else {
            if (count > best_count) {
                best_count = count;
                s.mode = sorted[i-1];
            }
            count = 1;
        }
    }
    if (count > best_count) s.mode = sorted[n-1];

    return s;
}

// Count unique values
int count_unique(uint64_t* data, int n) {
    std::vector<uint64_t> sorted(data, data + n);
    std::sort(sorted.begin(), sorted.end());
    return std::unique(sorted.begin(), sorted.end()) - sorted.begin();
}

// Histogram of values
void print_histogram(uint64_t* data, int n, const char* label) {
    std::vector<uint64_t> sorted(data, data + n);
    std::sort(sorted.begin(), sorted.end());

    fprintf(stderr, "  %s histogram:\n", label);
    uint64_t cur = sorted[0];
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        if (sorted[i] == cur) {
            cnt++;
        } else {
            if (cnt > n/200) // only show >0.5%
                fprintf(stderr, "    %5lu: %6d (%5.1f%%)\n", cur, cnt, 100.0*cnt/n);
            cur = sorted[i];
            cnt = 1;
        }
    }
    if (cnt > n/200)
        fprintf(stderr, "    %5lu: %6d (%5.1f%%)\n", cur, cnt, 100.0*cnt/n);
}

int main() {
    fprintf(stderr, "═══ GPU Substrate-Level Analog Noise Probes ═══\n");
    fprintf(stderr, "Iters: %d, Lanes: %d\n\n", ITERS, LANES);

    int N = LANES * ITERS;

    uint64_t *d_out, *h_out;
    uint64_t *d_out2, *h_out2;
    float *d_scratch;
    uint32_t *d_bits, *h_bits;
    uint32_t *d_data;

    hipMalloc(&d_out, N * sizeof(uint64_t));
    hipMalloc(&d_out2, N * sizeof(uint64_t));
    hipMalloc(&d_scratch, LANES * sizeof(float));
    hipMalloc(&d_bits, (N + 1) * sizeof(uint32_t));
    hipMalloc(&d_data, LANES * sizeof(uint32_t));

    h_out = (uint64_t*)malloc(N * sizeof(uint64_t));
    h_out2 = (uint64_t*)malloc(N * sizeof(uint64_t));
    h_bits = (uint32_t*)malloc((N + 1) * sizeof(uint32_t));

    // Init data
    uint32_t init_data[LANES];
    for (int i = 0; i < LANES; i++) init_data[i] = i + 42;
    hipMemcpy(d_data, init_data, LANES * sizeof(uint32_t), hipMemcpyHostToDevice);

    float zeros[LANES] = {};
    hipMemcpy(d_scratch, zeros, LANES * sizeof(float), hipMemcpyHostToDevice);

    // Print CSV header
    printf("test,lane,iter,value\n");

    auto run_and_report = [&](const char* name, int test_id) {
        hipMemcpy(h_out, d_out, N * sizeof(uint64_t), hipMemcpyDeviceToHost);

        // Aggregate across all lanes (lane 0 for stats)
        Stats s = compute_stats(h_out, ITERS);  // Lane 0
        int unique = count_unique(h_out, ITERS);

        // Cross-lane: check if all lanes give same value
        int lane_agree = 0;
        for (int i = 0; i < ITERS; i++) {
            bool all_same = true;
            for (int l = 1; l < LANES; l++) {
                if (h_out[l * ITERS + i] != h_out[0 * ITERS + i]) {
                    all_same = false; break;
                }
            }
            if (all_same) lane_agree++;
        }

        fprintf(stderr, "  %s: mean=%.1f, std=%.2f, CV=%.6f, mode=%lu, unique=%d, range=[%lu,%lu], lane_agree=%.1f%%\n",
                name, s.mean, s.std, s.cv, s.mode, unique,
                (uint64_t)s.min_v, (uint64_t)s.max_v,
                100.0 * lane_agree / ITERS);

        // Print histogram for lane 0
        print_histogram(h_out, ITERS, name);

        // CSV output (lane 0 only, for brevity)
        for (int i = 0; i < ITERS; i++) {
            printf("%d,0,%d,%lu\n", test_id, i, h_out[i]);
        }
    };

    // ── TEST 1: Bare clock noise ──
    fprintf(stderr, "═══ TEST 1: Bare clock64() readback noise ═══\n");
    bare_clock_noise<<<1, LANES>>>(d_out, ITERS);
    hipDeviceSynchronize();
    run_and_report("bare_clock", 1);

    // ── TEST 2: Single ALU instruction ──
    fprintf(stderr, "\n═══ TEST 2: Single ALU instruction (FMUL) jitter ═══\n");
    single_alu_jitter<<<1, LANES>>>(d_out, ITERS);
    hipDeviceSynchronize();
    run_and_report("single_fmul", 2);

    // ── TEST 3: Denormal arithmetic ──
    fprintf(stderr, "\n═══ TEST 3: Denormal arithmetic jitter ═══\n");
    denormal_jitter<<<1, LANES>>>(d_out, d_scratch, ITERS);
    hipDeviceSynchronize();
    run_and_report("denormal", 3);

    // ── TEST 4: Normal arithmetic (baseline) ──
    fprintf(stderr, "\n═══ TEST 4: Normal arithmetic (baseline for T3) ═══\n");
    normal_arith_jitter<<<1, LANES>>>(d_out, d_scratch, ITERS);
    hipDeviceSynchronize();
    run_and_report("normal_arith", 4);

    // ── TEST 5: Register pressure ──
    fprintf(stderr, "\n═══ TEST 5: Register pressure (low vs high) ═══\n");
    reg_pressure_low<<<1, LANES>>>(d_out, ITERS);
    hipDeviceSynchronize();
    run_and_report("reg_low", 5);

    reg_pressure_high<<<1, LANES>>>(d_out, ITERS);
    hipDeviceSynchronize();
    run_and_report("reg_high", 5);

    // ── TEST 6: L1 SRAM repeated read ──
    fprintf(stderr, "\n═══ TEST 6: L1 SRAM repeated read noise ═══\n");
    sram_read_noise<<<1, LANES>>>(d_out, d_data, ITERS);
    hipDeviceSynchronize();
    run_and_report("sram_l1", 6);

    // ── TEST 7: LDS read noise ──
    fprintf(stderr, "\n═══ TEST 7: LDS (on-CU SRAM) read noise ═══\n");
    lds_read_noise<<<1, LANES>>>(d_out, ITERS);
    hipDeviceSynchronize();
    run_and_report("lds_read", 7);

    // ── TEST 8: INT vs FLOAT pipeline ──
    fprintf(stderr, "\n═══ TEST 8: Integer vs Float pipeline noise ═══\n");
    int_pipeline_jitter<<<1, LANES>>>(d_out, ITERS);
    hipDeviceSynchronize();
    run_and_report("int_pipe", 8);

    float_pipeline_jitter<<<1, LANES>>>(d_out, ITERS);
    hipDeviceSynchronize();
    run_and_report("float_pipe", 8);

    // ── TEST 9: Transcendental functions ──
    fprintf(stderr, "\n═══ TEST 9: Transcendental function jitter ═══\n");
    const char* func_names[] = {"sinf", "expf", "logf", "rcpf"};
    for (int f = 0; f < 4; f++) {
        transcendental_jitter<<<1, LANES>>>(d_out, ITERS, f);
        hipDeviceSynchronize();
        run_and_report(func_names[f], 9);
    }

    // ── TEST 10: Wavefront divergence ──
    fprintf(stderr, "\n═══ TEST 10: Wavefront divergence (branch masking noise) ═══\n");
    int divs[] = {2, 4, 8, 16, 32};
    for (int d = 0; d < 5; d++) {
        char buf[32];
        snprintf(buf, sizeof(buf), "div_%d", divs[d]);
        divergence_jitter<<<1, LANES>>>(d_out, ITERS, divs[d]);
        hipDeviceSynchronize();
        run_and_report(buf, 10);
    }

    // ── TEST 11: Power ripple correlation ──
    fprintf(stderr, "\n═══ TEST 11: Power supply ripple (heavy→light timing coupling) ═══\n");
    power_ripple_probe<<<1, LANES>>>(d_out, d_out2, d_scratch, ITERS);
    hipDeviceSynchronize();

    hipMemcpy(h_out, d_out, N * sizeof(uint64_t), hipMemcpyDeviceToHost);
    hipMemcpy(h_out2, d_out2, N * sizeof(uint64_t), hipMemcpyDeviceToHost);

    Stats heavy = compute_stats(h_out, ITERS);
    Stats light = compute_stats(h_out2, ITERS);

    // Correlation between heavy and light timing
    double sum_hl = 0, sum_h = 0, sum_l = 0, sum_hh = 0, sum_ll = 0;
    for (int i = 0; i < ITERS; i++) {
        double hv = h_out[i], lv = h_out2[i];
        sum_hl += hv * lv; sum_h += hv; sum_l += lv;
        sum_hh += hv * hv; sum_ll += lv * lv;
    }
    double r_hl = (ITERS * sum_hl - sum_h * sum_l) /
                  (sqrt(ITERS * sum_hh - sum_h * sum_h) * sqrt(ITERS * sum_ll - sum_l * sum_l) + 1e-15);

    fprintf(stderr, "  heavy_phase: mean=%.1f, std=%.2f, CV=%.6f\n", heavy.mean, heavy.std, heavy.cv);
    fprintf(stderr, "  light_phase: mean=%.1f, std=%.2f, CV=%.6f\n", light.mean, light.std, light.cv);
    fprintf(stderr, "  heavy↔light correlation: r=%.4f\n", r_hl);
    print_histogram(h_out2, ITERS, "light_phase");

    // ── TEST 12: Float bit-level noise ──
    fprintf(stderr, "\n═══ TEST 12: Float bit-level determinism ═══\n");
    float_bit_noise<<<1, LANES>>>(d_bits, ITERS);
    hipDeviceSynchronize();
    hipMemcpy(h_bits, d_bits, (N + 1) * sizeof(uint32_t), hipMemcpyDeviceToHost);

    // Check how many unique bit patterns lane 0 produces
    std::vector<uint32_t> lane0_bits(ITERS);
    for (int i = 0; i < ITERS; i++) lane0_bits[i] = h_bits[i];
    std::sort(lane0_bits.begin(), lane0_bits.end());
    int unique_bits = std::unique(lane0_bits.begin(), lane0_bits.end()) - lane0_bits.begin();

    // Check cross-lane bit variation
    int cross_lane_diffs = 0;
    for (int i = 0; i < ITERS; i++) {
        for (int l = 1; l < LANES; l++) {
            if (h_bits[l * ITERS + i] != h_bits[0 * ITERS + i]) {
                cross_lane_diffs++; break;
            }
        }
    }

    fprintf(stderr, "  Lane 0: %d unique bit patterns across %d iterations\n", unique_bits, ITERS);
    fprintf(stderr, "  Cross-lane: %d/%d iterations with different bits (%.1f%%)\n",
            cross_lane_diffs, ITERS, 100.0 * cross_lane_diffs / ITERS);
    fprintf(stderr, "  Bit flips (lane 0): %u\n", h_bits[N]);

    // Show first few bit patterns
    fprintf(stderr, "  First 10 bit patterns (lane 0): ");
    for (int i = 0; i < 10 && i < ITERS; i++) {
        fprintf(stderr, "0x%08x ", h_bits[i]);
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "\nDone.\n");

    hipFree(d_out); hipFree(d_out2); hipFree(d_scratch); hipFree(d_bits); hipFree(d_data);
    free(h_out); free(h_out2); free(h_bits);
    return 0;
}
