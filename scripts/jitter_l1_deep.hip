// jitter_l1_deep.hip — Deep characterization of L1 cache SRAM noise
// The ONLY analog noise source found in gfx1151: 0.8% of L1 reads take
// 17 cycles instead of 16. This probe determines if the noise is:
//   - Data-dependent (different bit patterns in cache)
//   - Address-dependent (different cache sets/ways)
//   - Temperature-dependent (hot die vs cold die)
//   - Temporally structured (autocorrelation, 1/f, bursty)
//   - Amplifiable (multiple sequential reads)
//   - Voltage-sensitive (different DPM states)
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_l1_deep scripts/jitter_l1_deep.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_l1_deep

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <vector>

#define ITERS 20000
#define LANES 64

// ============================================================================
// TEST 1: Data-dependent noise — do different bit patterns in L1 cause
// different noise rates? Read same address but with different stored values.
// All-zeros vs all-ones vs alternating vs random patterns.
// ============================================================================
__global__ void l1_data_dependent(uint64_t* out, uint32_t* data, int iters, int pattern_id) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= LANES) return;

    // Prime the cache line
    uint32_t val = data[tid];

    for (int i = 0; i < iters; i++) {
        uint64_t t0 = __clock64();
        val = data[tid];
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
        if (val == 0xDEADDEAD) out[0] = 0;
    }
}

// ============================================================================
// TEST 2: Address-dependent noise — do different cache sets have different
// noise characteristics? Read from addresses that map to different L1 sets.
// gfx1100 L1 = 32KB, 4-way, 128B lines → 64 sets
// Set index = (addr >> 7) & 0x3F
// ============================================================================
__global__ void l1_address_sweep(uint64_t* out, uint32_t* data, int iters, int set_offset) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= LANES) return;

    // Each thread reads from a different cache set
    // set_offset shifts which set we target
    int addr_idx = ((set_offset + tid) * 32);  // 32 uint32_t = 128 bytes = 1 cache line apart

    // Prime
    uint32_t val = data[addr_idx];

    for (int i = 0; i < iters; i++) {
        uint64_t t0 = __clock64();
        val = data[addr_idx];
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
        if (val == 0xDEADDEAD) out[0] = 0;
    }
}

// ============================================================================
// TEST 3: Temporal autocorrelation — are the +1 cycle events clustered
// or uniformly random? Record raw timing and compute autocorrelation on host.
// Uses 20000 iterations for statistical power.
// ============================================================================
__global__ void l1_temporal_raw(uint64_t* out, uint32_t* data, int iters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= LANES) return;

    uint32_t val = data[tid];

    for (int i = 0; i < iters; i++) {
        uint64_t t0 = __clock64();
        val = data[tid];
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
        if (val == 0xDEADDEAD) out[0] = 0;
    }
}

// ============================================================================
// TEST 4: Sequential amplification — do N consecutive L1 reads accumulate
// more noise than 1 read? If noise is independent per-read, std should grow
// as sqrt(N). If correlated, it grows faster or slower.
// ============================================================================
__global__ void l1_sequential_reads(uint64_t* out, uint32_t* data, int iters, int n_reads) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= LANES) return;

    uint32_t val = data[tid];

    for (int i = 0; i < iters; i++) {
        uint64_t t0 = __clock64();
        // N sequential reads from same cached address
        for (int r = 0; r < n_reads; r++) {
            val = data[tid];
        }
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
        if (val == 0xDEADDEAD) out[0] = 0;
    }
}

// ============================================================================
// TEST 5: Cross-address sequential — read from N DIFFERENT L1-cached addresses
// in sequence. Tests if noise is per-access or per-cache-line.
// ============================================================================
__global__ void l1_multi_address(uint64_t* out, uint32_t* data, int iters, int n_addrs) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= LANES) return;

    // Prime all addresses into cache
    uint32_t val = 0;
    for (int a = 0; a < n_addrs; a++) {
        val += data[tid + a * 64];  // stride by warp size to avoid conflicts
    }

    for (int i = 0; i < iters; i++) {
        uint64_t t0 = __clock64();
        for (int a = 0; a < n_addrs; a++) {
            val = data[tid + a * 64];
        }
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
        if (val == 0xDEADDEAD) out[0] = 0;
    }
}

// ============================================================================
// TEST 6: Thermal loading — one wavefront heats the CU with heavy FMA while
// another wavefront reads L1. Does thermal loading increase noise rate?
// ============================================================================
__global__ void l1_thermal_load(uint64_t* out, uint32_t* data, int iters, int heavy_waves) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int wid = tid / 64;  // wavefront ID

    if (wid == 0) {
        // Wavefront 0: L1 read probe (same as baseline)
        int lane = tid % 64;
        if (lane >= LANES) return;

        uint32_t val = data[lane];
        for (int i = 0; i < iters; i++) {
            uint64_t t0 = __clock64();
            val = data[lane];
            uint64_t t1 = __clock64();
            out[lane * iters + i] = t1 - t0;
            if (val == 0xDEADDEAD) out[0] = 0;
        }
    } else if (wid <= heavy_waves) {
        // Heavy wavefronts: burn power with FMA chains
        float a = 1.0f + tid * 0.001f;
        float b = 2.0f, c = 3.0f, d = 4.0f;
        for (int i = 0; i < iters * 100; i++) {
            a = __fmaf_rn(a, b, c);
            b = __fmaf_rn(b, c, d);
            c = __fmaf_rn(c, d, a);
            d = __fmaf_rn(d, a, b);
        }
        // Prevent DCE
        if (a == -999.0f) out[0] = __float_as_uint(a);
    }
}

// ============================================================================
// TEST 7: L1↔L2 boundary noise — intentionally cause L1 misses by reading
// a working set larger than L1 (32KB). Miss latency variance = L2 path noise.
// ============================================================================
__global__ void l1_l2_boundary(uint64_t* out, uint32_t* data, int iters, int stride_lines) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= LANES) return;

    // stride_lines cache lines apart → controls working set size
    // 1 line = 128 bytes = 32 uint32_t
    // > 256 lines (32KB) = guaranteed L1 miss

    for (int i = 0; i < iters; i++) {
        int addr = tid + (i % stride_lines) * 32;
        uint64_t t0 = __clock64();
        uint32_t val = data[addr];
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
        if (val == 0xDEADDEAD) out[0] = 0;
    }
}

// ============================================================================
// TEST 8: Bit-position sensitivity — store values with only 1 bit set,
// then read. Tests if specific SRAM bit cells are noisier than others.
// ============================================================================
__global__ void l1_bit_sensitivity(uint64_t* out, uint32_t* data, int iters, int bit_pos) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= LANES) return;

    // data[tid] has been pre-loaded with (1 << bit_pos) by host
    uint32_t val = data[tid];

    for (int i = 0; i < iters; i++) {
        uint64_t t0 = __clock64();
        val = data[tid];
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
        if (val == 0xDEADDEAD) out[0] = 0;
    }
}

// ============================================================================
// TEST 9: Rapid toggle — write then immediately read same address.
// The write→read turnaround might expose different noise than read-only.
// ============================================================================
__global__ void l1_write_read_noise(uint64_t* out, uint32_t* data, int iters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= LANES) return;

    for (int i = 0; i < iters; i++) {
        data[tid] = i + tid;  // Write (modifies cache line)
        __threadfence_block();
        uint64_t t0 = __clock64();
        uint32_t val = data[tid];  // Immediate read
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
        if (val == 0xDEADDEAD) out[0] = 0;
    }
}

// ============================================================================
// Host analysis functions
// ============================================================================

struct Stats {
    double mean, std, cv;
    int unique;
    uint64_t min_val, max_val;
    int slow_count;  // count of values > min_val
    double slow_rate; // fraction of slow reads
};

Stats compute_stats(const std::vector<uint64_t>& data, int offset, int count) {
    Stats s = {};
    if (count == 0) return s;

    double sum = 0;
    s.min_val = UINT64_MAX;
    s.max_val = 0;
    std::vector<uint64_t> vals(count);

    for (int i = 0; i < count; i++) {
        uint64_t v = data[offset + i];
        vals[i] = v;
        sum += v;
        if (v < s.min_val) s.min_val = v;
        if (v > s.max_val) s.max_val = v;
    }

    s.mean = sum / count;

    double var = 0;
    for (int i = 0; i < count; i++) {
        double d = vals[i] - s.mean;
        var += d * d;
    }
    s.std = sqrt(var / count);
    s.cv = (s.mean > 0) ? s.std / s.mean : 0;

    // Count unique values
    std::sort(vals.begin(), vals.end());
    s.unique = std::unique(vals.begin(), vals.end()) - vals.begin();

    // Count slow reads (> minimum)
    s.slow_count = 0;
    for (int i = 0; i < count; i++) {
        if (vals[i] > s.min_val) s.slow_count++;
    }
    s.slow_rate = (double)s.slow_count / count;

    return s;
}

// Compute autocorrelation at given lag
double autocorrelation(const std::vector<uint64_t>& data, int offset, int count, int lag) {
    if (lag >= count) return 0;

    double mean = 0;
    for (int i = 0; i < count; i++) mean += data[offset + i];
    mean /= count;

    double num = 0, den = 0;
    for (int i = 0; i < count - lag; i++) {
        double a = data[offset + i] - mean;
        double b = data[offset + i + lag] - mean;
        num += a * b;
    }
    for (int i = 0; i < count; i++) {
        double a = data[offset + i] - mean;
        den += a * a;
    }

    return (den > 0) ? num / den : 0;
}

// Compute runs test — are slow events clustered or random?
// Returns z-score: |z| > 1.96 means non-random at p<0.05
double runs_test(const std::vector<uint64_t>& data, int offset, int count) {
    if (count < 20) return 0;

    uint64_t min_val = UINT64_MAX;
    for (int i = 0; i < count; i++)
        if (data[offset + i] < min_val) min_val = data[offset + i];

    int n1 = 0, n2 = 0;  // counts above and at minimum
    int runs = 1;
    bool prev_slow = (data[offset] > min_val);

    for (int i = 0; i < count; i++) {
        bool slow = (data[offset + i] > min_val);
        if (slow) n1++; else n2++;
        if (i > 0 && slow != prev_slow) runs++;
        prev_slow = slow;
    }

    if (n1 == 0 || n2 == 0) return 0;

    double N = n1 + n2;
    double expected = 1.0 + 2.0 * n1 * n2 / N;
    double var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - N) / (N * N * (N - 1));
    if (var <= 0) return 0;

    return (runs - expected) / sqrt(var);
}

// Power spectral density slope estimate (log-log regression on periodogram)
double psd_slope(const std::vector<uint64_t>& data, int offset, int count) {
    if (count < 64) return 0;

    // Simple periodogram: compute magnitude of DFT at low frequencies
    int n_freqs = count / 4;  // use quarter of frequencies
    if (n_freqs > 512) n_freqs = 512;

    double mean = 0;
    for (int i = 0; i < count; i++) mean += data[offset + i];
    mean /= count;

    std::vector<double> power(n_freqs);
    for (int f = 1; f <= n_freqs; f++) {
        double re = 0, im = 0;
        for (int t = 0; t < count; t++) {
            double phase = 2.0 * M_PI * f * t / count;
            double val = data[offset + t] - mean;
            re += val * cos(phase);
            im += val * sin(phase);
        }
        power[f-1] = (re*re + im*im) / count;
    }

    // Log-log linear regression: log(P) vs log(f)
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    int n = 0;
    for (int f = 0; f < n_freqs; f++) {
        if (power[f] > 0) {
            double lf = log(f + 1);
            double lp = log(power[f]);
            sum_x += lf;
            sum_y += lp;
            sum_xy += lf * lp;
            sum_xx += lf * lf;
            n++;
        }
    }
    if (n < 2) return 0;
    return (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
}

int main() {
    fprintf(stderr, "═══ L1 Cache SRAM Deep Noise Characterization ═══\n");
    fprintf(stderr, "Iters: %d per test, %d lanes\n\n", ITERS, LANES);

    // Allocate GPU memory
    uint64_t *d_out;
    uint32_t *d_data;
    size_t out_size = LANES * ITERS * sizeof(uint64_t);
    // Large data buffer for address sweep (1MB)
    size_t data_size = 1024 * 1024 * sizeof(uint32_t);

    hipMalloc(&d_out, out_size);
    hipMalloc(&d_data, data_size);

    std::vector<uint64_t> h_out(LANES * ITERS);
    std::vector<uint32_t> h_data(1024 * 1024);

    // CSV output
    printf("test,param,lane,mean,std,cv,unique,min,max,slow_rate,slow_count\n");

    // ====================================================================
    // TEST 1: Data pattern dependency
    // ====================================================================
    fprintf(stderr, "═══ TEST 1: Data Pattern Dependency ═══\n");
    const char* pattern_names[] = {"zeros", "ones", "alt_0x55", "alt_0xAA", "random", "walking_1"};
    for (int p = 0; p < 6; p++) {
        // Fill data with pattern
        for (int i = 0; i < 64; i++) {
            switch (p) {
                case 0: h_data[i] = 0x00000000; break;
                case 1: h_data[i] = 0xFFFFFFFF; break;
                case 2: h_data[i] = 0x55555555; break;
                case 3: h_data[i] = 0xAAAAAAAA; break;
                case 4: h_data[i] = rand(); break;
                case 5: h_data[i] = 1u << (i % 32); break;
            }
        }
        hipMemcpy(d_data, h_data.data(), 256, hipMemcpyHostToDevice);
        hipMemset(d_out, 0, out_size);

        l1_data_dependent<<<1, LANES>>>(d_out, d_data, ITERS, p);
        hipDeviceSynchronize();
        hipMemcpy(h_out.data(), d_out, out_size, hipMemcpyDeviceToHost);

        // Aggregate across all lanes
        int total_slow = 0, total = 0;
        double total_mean = 0;
        for (int lane = 0; lane < LANES; lane++) {
            Stats s = compute_stats(h_out, lane * ITERS, ITERS);
            total_slow += s.slow_count;
            total += ITERS;
            total_mean += s.mean;
        }
        double rate = (double)total_slow / total;
        total_mean /= LANES;
        fprintf(stderr, "  pattern=%-10s: mean=%.1f, slow_rate=%.4f%% (%d/%d)\n",
                pattern_names[p], total_mean, rate * 100, total_slow, total);
        printf("data_pattern,%s,all,%.3f,,,,%d,%d,%.6f,%d\n",
               pattern_names[p], total_mean, 0, 0, rate, total_slow);
    }
    fprintf(stderr, "\n");

    // ====================================================================
    // TEST 2: Address/cache set dependency
    // ====================================================================
    fprintf(stderr, "═══ TEST 2: Cache Set Dependency ═══\n");
    // Fill all data
    for (size_t i = 0; i < 1024*1024; i++) h_data[i] = i + 42;
    hipMemcpy(d_data, h_data.data(), data_size, hipMemcpyHostToDevice);

    for (int set_off = 0; set_off < 64; set_off += 8) {
        hipMemset(d_out, 0, out_size);
        l1_address_sweep<<<1, LANES>>>(d_out, d_data, ITERS, set_off);
        hipDeviceSynchronize();
        hipMemcpy(h_out.data(), d_out, out_size, hipMemcpyDeviceToHost);

        int total_slow = 0;
        double total_mean = 0;
        for (int lane = 0; lane < LANES; lane++) {
            Stats s = compute_stats(h_out, lane * ITERS, ITERS);
            total_slow += s.slow_count;
            total_mean += s.mean;
        }
        double rate = (double)total_slow / (LANES * ITERS);
        total_mean /= LANES;
        fprintf(stderr, "  set_offset=%2d: mean=%.1f, slow_rate=%.4f%%\n",
                set_off, total_mean, rate * 100);
        printf("cache_set,%d,all,%.3f,,,,,%.6f,%d\n",
               set_off, total_mean, rate, total_slow);
    }
    fprintf(stderr, "\n");

    // ====================================================================
    // TEST 3: Temporal structure (autocorrelation, runs, PSD)
    // ====================================================================
    fprintf(stderr, "═══ TEST 3: Temporal Structure ═══\n");
    // Use simple L1 read data — fill fresh
    for (int i = 0; i < 64; i++) h_data[i] = i + 100;
    hipMemcpy(d_data, h_data.data(), 256, hipMemcpyHostToDevice);
    hipMemset(d_out, 0, out_size);

    l1_temporal_raw<<<1, LANES>>>(d_out, d_data, ITERS);
    hipDeviceSynchronize();
    hipMemcpy(h_out.data(), d_out, out_size, hipMemcpyDeviceToHost);

    // Pick lane 0 for detailed analysis
    Stats s0 = compute_stats(h_out, 0, ITERS);
    fprintf(stderr, "  Lane 0: mean=%.1f, std=%.3f, CV=%.6f, unique=%d, range=[%lu,%lu]\n",
            s0.mean, s0.std, s0.cv, s0.unique, s0.min_val, s0.max_val);
    fprintf(stderr, "  Slow rate: %.3f%% (%d/%d)\n", s0.slow_rate * 100, s0.slow_count, ITERS);

    // Autocorrelation at lags 1..20
    fprintf(stderr, "  Autocorrelation: ");
    for (int lag = 1; lag <= 20; lag++) {
        double acf = autocorrelation(h_out, 0, ITERS, lag);
        fprintf(stderr, "r(%d)=%.4f ", lag, acf);
        if (lag % 5 == 0) fprintf(stderr, "\n                   ");
        printf("temporal_acf,%d,0,%.6f,,,,,,\n", lag, acf);
    }
    fprintf(stderr, "\n");

    // Runs test
    double z = runs_test(h_out, 0, ITERS);
    fprintf(stderr, "  Runs test z=%.3f (|z|>1.96 = non-random clustering)\n", z);
    printf("temporal_runs,z,0,%.6f,,,,,,\n", z);

    // PSD slope
    double slope = psd_slope(h_out, 0, ITERS);
    fprintf(stderr, "  PSD slope=%.3f (0=white, -1=1/f, -2=brown)\n", slope);
    printf("temporal_psd,slope,0,%.6f,,,,,,\n", slope);

    // Cross-lane correlation — are slow events simultaneous across lanes?
    fprintf(stderr, "\n  Cross-lane simultaneity:\n");
    for (int l2 = 1; l2 <= 5; l2++) {
        int simultaneous = 0;
        int lane0_slow = 0;
        for (int i = 0; i < ITERS; i++) {
            bool s0_slow = (h_out[0 * ITERS + i] > s0.min_val);
            bool s1_slow = (h_out[l2 * ITERS + i] > s0.min_val);
            if (s0_slow) {
                lane0_slow++;
                if (s1_slow) simultaneous++;
            }
        }
        double cond_prob = (lane0_slow > 0) ? (double)simultaneous / lane0_slow : 0;
        fprintf(stderr, "    L0&L%d: %d simultaneous / %d L0-slow = %.1f%% (baseline=%.1f%%)\n",
                l2, simultaneous, lane0_slow, cond_prob * 100, s0.slow_rate * 100);
        printf("cross_lane,%d,0,%.6f,,%d,%d,,,\n", l2, cond_prob, simultaneous, lane0_slow);
    }
    fprintf(stderr, "\n");

    // ====================================================================
    // TEST 4: Sequential amplification
    // ====================================================================
    fprintf(stderr, "═══ TEST 4: Sequential Read Amplification ═══\n");
    int read_counts[] = {1, 2, 4, 8, 16, 32, 64};
    for (int rc = 0; rc < 7; rc++) {
        int n = read_counts[rc];
        hipMemset(d_out, 0, out_size);
        l1_sequential_reads<<<1, LANES>>>(d_out, d_data, ITERS, n);
        hipDeviceSynchronize();
        hipMemcpy(h_out.data(), d_out, out_size, hipMemcpyDeviceToHost);

        Stats s = compute_stats(h_out, 0, ITERS);
        fprintf(stderr, "  %2d reads: mean=%.1f, std=%.3f, CV=%.6f, unique=%d, range=[%lu,%lu]\n",
                n, s.mean, s.std, s.cv, s.unique, s.min_val, s.max_val);
        printf("seq_reads,%d,0,%.3f,%.3f,%.6f,%d,%lu,%lu,%.6f,%d\n",
               n, s.mean, s.std, s.cv, s.unique, s.min_val, s.max_val, s.slow_rate, s.slow_count);
    }
    fprintf(stderr, "\n");

    // ====================================================================
    // TEST 5: Multi-address reads (different cache lines)
    // ====================================================================
    fprintf(stderr, "═══ TEST 5: Multi-Address Read (different cache lines) ═══\n");
    int addr_counts[] = {1, 2, 4, 8, 16};
    for (int ac = 0; ac < 5; ac++) {
        int n = addr_counts[ac];
        hipMemset(d_out, 0, out_size);
        l1_multi_address<<<1, LANES>>>(d_out, d_data, ITERS, n);
        hipDeviceSynchronize();
        hipMemcpy(h_out.data(), d_out, out_size, hipMemcpyDeviceToHost);

        Stats s = compute_stats(h_out, 0, ITERS);
        fprintf(stderr, "  %2d addrs: mean=%.1f, std=%.3f, CV=%.6f, unique=%d, range=[%lu,%lu]\n",
                n, s.mean, s.std, s.cv, s.unique, s.min_val, s.max_val);
        printf("multi_addr,%d,0,%.3f,%.3f,%.6f,%d,%lu,%lu,%.6f,%d\n",
               n, s.mean, s.std, s.cv, s.unique, s.min_val, s.max_val, s.slow_rate, s.slow_count);
    }
    fprintf(stderr, "\n");

    // ====================================================================
    // TEST 6: Thermal loading effect on noise rate
    // ====================================================================
    fprintf(stderr, "═══ TEST 6: Thermal Loading Effect ═══\n");
    int thermal_waves[] = {0, 1, 4, 8, 15};
    for (int tw = 0; tw < 5; tw++) {
        int n = thermal_waves[tw];
        hipMemset(d_out, 0, out_size);
        // Need more threads: wave0 = probe, waves 1..n = heavy
        int total_threads = (n + 1) * 64;
        l1_thermal_load<<<1, total_threads>>>(d_out, d_data, ITERS, n);
        hipDeviceSynchronize();
        hipMemcpy(h_out.data(), d_out, out_size, hipMemcpyDeviceToHost);

        // Only analyze wavefront 0 (lanes 0-63)
        Stats s = compute_stats(h_out, 0, ITERS);
        int total_slow = 0;
        for (int lane = 0; lane < LANES; lane++) {
            Stats sl = compute_stats(h_out, lane * ITERS, ITERS);
            total_slow += sl.slow_count;
        }
        double rate = (double)total_slow / (LANES * ITERS);
        fprintf(stderr, "  %2d heavy waves: mean=%.1f, slow_rate=%.4f%%, L0_unique=%d\n",
                n, s.mean, rate * 100, s.unique);
        printf("thermal,%d,all,%.3f,%.3f,,,,,%.6f,%d\n",
               n, s.mean, s.std, rate, total_slow);
    }
    fprintf(stderr, "\n");

    // ====================================================================
    // TEST 7: L1↔L2 boundary (cache miss timing)
    // ====================================================================
    fprintf(stderr, "═══ TEST 7: L1↔L2 Boundary (cache misses) ═══\n");
    int stride_vals[] = {1, 4, 16, 64, 256, 512, 1024};
    const char* stride_labels[] = {"128B", "512B", "2KB", "8KB", "32KB(=L1)", "64KB(>L1)", "128KB(>>L1)"};
    for (int sv = 0; sv < 7; sv++) {
        int stride = stride_vals[sv];
        hipMemset(d_out, 0, out_size);
        l1_l2_boundary<<<1, LANES>>>(d_out, d_data, ITERS, stride);
        hipDeviceSynchronize();
        hipMemcpy(h_out.data(), d_out, out_size, hipMemcpyDeviceToHost);

        Stats s = compute_stats(h_out, 0, ITERS);
        fprintf(stderr, "  stride=%4d (%s): mean=%.1f, std=%.2f, CV=%.6f, unique=%d, range=[%lu,%lu]\n",
                stride, stride_labels[sv], s.mean, s.std, s.cv, s.unique, s.min_val, s.max_val);
        printf("l1_l2,%d,0,%.3f,%.3f,%.6f,%d,%lu,%lu,%.6f,%d\n",
               stride, s.mean, s.std, s.cv, s.unique, s.min_val, s.max_val, s.slow_rate, s.slow_count);
    }
    fprintf(stderr, "\n");

    // ====================================================================
    // TEST 8: Bit position sensitivity
    // ====================================================================
    fprintf(stderr, "═══ TEST 8: SRAM Bit Position Sensitivity ═══\n");
    for (int bit = 0; bit < 32; bit += 4) {
        for (int i = 0; i < 64; i++) h_data[i] = 1u << bit;
        hipMemcpy(d_data, h_data.data(), 256, hipMemcpyHostToDevice);
        hipMemset(d_out, 0, out_size);

        l1_bit_sensitivity<<<1, LANES>>>(d_out, d_data, ITERS, bit);
        hipDeviceSynchronize();
        hipMemcpy(h_out.data(), d_out, out_size, hipMemcpyDeviceToHost);

        int total_slow = 0;
        for (int lane = 0; lane < LANES; lane++) {
            Stats sl = compute_stats(h_out, lane * ITERS, ITERS);
            total_slow += sl.slow_count;
        }
        double rate = (double)total_slow / (LANES * ITERS);
        Stats s = compute_stats(h_out, 0, ITERS);
        fprintf(stderr, "  bit[%2d]: mean=%.1f, slow_rate=%.4f%%\n", bit, s.mean, rate * 100);
        printf("bit_pos,%d,all,%.3f,,,,,%.6f,%d\n", bit, s.mean, rate, total_slow);
    }
    fprintf(stderr, "\n");

    // ====================================================================
    // TEST 9: Write→Read turnaround noise
    // ====================================================================
    fprintf(stderr, "═══ TEST 9: Write→Read Turnaround ═══\n");
    for (int i = 0; i < 64; i++) h_data[i] = i;
    hipMemcpy(d_data, h_data.data(), 256, hipMemcpyHostToDevice);
    hipMemset(d_out, 0, out_size);

    l1_write_read_noise<<<1, LANES>>>(d_out, d_data, ITERS);
    hipDeviceSynchronize();
    hipMemcpy(h_out.data(), d_out, out_size, hipMemcpyDeviceToHost);

    int total_slow_wr = 0;
    double wr_mean = 0;
    for (int lane = 0; lane < LANES; lane++) {
        Stats sl = compute_stats(h_out, lane * ITERS, ITERS);
        total_slow_wr += sl.slow_count;
        wr_mean += sl.mean;
    }
    wr_mean /= LANES;
    double wr_rate = (double)total_slow_wr / (LANES * ITERS);
    Stats swr = compute_stats(h_out, 0, ITERS);
    fprintf(stderr, "  Write→Read: mean=%.1f, std=%.3f, CV=%.6f, slow_rate=%.4f%%, unique=%d\n",
            wr_mean, swr.std, swr.cv, wr_rate * 100, swr.unique);
    printf("write_read,0,all,%.3f,%.3f,%.6f,%d,,,%.6f,%d\n",
           wr_mean, swr.std, swr.cv, swr.unique, wr_rate, total_slow_wr);
    fprintf(stderr, "\n");

    // ====================================================================
    // SUMMARY
    // ====================================================================
    fprintf(stderr, "═══ SUMMARY ═══\n");
    fprintf(stderr, "L1 SRAM read is the ONLY noise source in gfx1151.\n");
    fprintf(stderr, "This probe tests whether that noise is:\n");
    fprintf(stderr, "  1. Data-dependent (bit pattern in cache)\n");
    fprintf(stderr, "  2. Address-dependent (cache set)\n");
    fprintf(stderr, "  3. Temporally structured (autocorrelation, 1/f, clustering)\n");
    fprintf(stderr, "  4. Amplifiable (sequential reads)\n");
    fprintf(stderr, "  5. Temperature-sensitive (thermal loading)\n");
    fprintf(stderr, "  6. Related to L1↔L2 boundary effects\n");
    fprintf(stderr, "  7. Bit-cell specific (bit position)\n");
    fprintf(stderr, "  8. Write→Read turnaround dependent\n");

    hipFree(d_out);
    hipFree(d_data);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
