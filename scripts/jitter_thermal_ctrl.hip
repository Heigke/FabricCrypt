// jitter_thermal_ctrl.hip — Thermal-controlled L1 jitter as analog noise source
// KEY FINDING: L1 read noise is CU-wide clock/power jitter, thermally controllable
// from 0.1% to 82% slow rate. This probe characterizes:
//   - Fine-grained thermal modulation curve (0..15 heavy waves)
//   - Multi-CU independence (are different CUs' noise uncorrelated?)
//   - Dynamic thermal switching (can we toggle noise rate in real time?)
//   - Noise quality at different thermal levels (entropy, autocorrelation)
//   - Controllability: can we set a TARGET noise rate?
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_thermal_ctrl scripts/jitter_thermal_ctrl.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_thermal_ctrl

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <vector>

#define ITERS 10000
#define LANES 64

// ============================================================================
// KERNEL 1: Fine-grained thermal modulation
// Wavefront 0 = probe, wavefronts 1..N = thermal load
// Records per-iteration binary noise (16 or 17 cycles)
// ============================================================================
__global__ void thermal_modulation(uint64_t* out, uint32_t* data, int iters, int heavy_waves) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int wid = tid / 64;

    if (wid == 0) {
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
        float a = 1.0f + tid * 0.001f;
        float b = 2.0f, c = 3.0f, d = 4.0f;
        for (int i = 0; i < iters * 50; i++) {
            a = __fmaf_rn(a, b, c);
            b = __fmaf_rn(b, c, d);
            c = __fmaf_rn(c, d, a);
            d = __fmaf_rn(d, a, b);
        }
        if (a == -999.0f) out[0] = __float_as_uint(a);
    }
}

// ============================================================================
// KERNEL 2: Multi-CU probe — launch on MULTIPLE CUs simultaneously
// Each CU block has its own probe wavefront + thermal wavefronts
// Tests inter-CU noise independence
// ============================================================================
__global__ void multi_cu_probe(uint64_t* out, uint32_t* data, int iters, int heavy_per_cu) {
    int block = blockIdx.x;  // CU index
    int local_tid = threadIdx.x;
    int wid = local_tid / 64;
    int lane = local_tid % 64;

    if (wid == 0 && lane < LANES) {
        // Probe wavefront — use block-specific data region
        int data_idx = block * 64 + lane;
        uint32_t val = data[data_idx];
        for (int i = 0; i < iters; i++) {
            uint64_t t0 = __clock64();
            val = data[data_idx];
            uint64_t t1 = __clock64();
            out[(block * LANES + lane) * iters + i] = t1 - t0;
            if (val == 0xDEADDEAD) out[0] = 0;
        }
    } else if (wid > 0 && wid <= heavy_per_cu) {
        // Thermal load wavefronts
        float a = 1.0f + local_tid * 0.001f;
        float b = 2.0f, c = 3.0f, d = 4.0f;
        for (int i = 0; i < iters * 50; i++) {
            a = __fmaf_rn(a, b, c);
            b = __fmaf_rn(b, c, d);
            c = __fmaf_rn(c, d, a);
            d = __fmaf_rn(d, a, b);
        }
        if (a == -999.0f) out[0] = __float_as_uint(a);
    }
}

// ============================================================================
// KERNEL 3: Dynamic switching — alternate between hot and cold phases
// within a single kernel launch. Tests switching speed.
// ============================================================================
__global__ void dynamic_switch(uint64_t* out, uint32_t* data, int iters, int phase_len) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int wid = tid / 64;

    if (wid == 0) {
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
    } else {
        // Alternating hot/cold phases
        float a = 1.0f + tid * 0.001f;
        float b = 2.0f, c = 3.0f, d = 4.0f;
        int total_cycles = iters * 50;
        for (int i = 0; i < total_cycles; i++) {
            // Hot phase: FMA chain
            int phase = (i / phase_len) % 2;
            if (phase == 0) {
                a = __fmaf_rn(a, b, c);
                b = __fmaf_rn(b, c, d);
                c = __fmaf_rn(c, d, a);
                d = __fmaf_rn(d, a, b);
            }
            // Cold phase: just loop (minimal work)
        }
        if (a == -999.0f) out[0] = __float_as_uint(a);
    }
}

// ============================================================================
// KERNEL 4: Noise harvest — convert L1 jitter into random bits
// Each read produces 0 (fast) or 1 (slow). Pack 32 bits per uint32_t.
// This tests if the noise is usable as a random bit source.
// ============================================================================
__global__ void noise_harvest(uint32_t* bits_out, uint32_t* data, int n_words) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int wid = tid / 64;

    if (wid == 0) {
        int lane = tid % 64;
        if (lane >= LANES) return;
        uint32_t val = data[lane];
        for (int w = 0; w < n_words; w++) {
            uint32_t word = 0;
            for (int b = 0; b < 32; b++) {
                uint64_t t0 = __clock64();
                val = data[lane];
                uint64_t t1 = __clock64();
                if ((t1 - t0) > 16) word |= (1u << b);
                if (val == 0xDEADDEAD) bits_out[0] = 0;
            }
            bits_out[lane * n_words + w] = word;
        }
    } else {
        // Thermal loading to increase bit rate
        float a = 1.0f + tid * 0.001f;
        float b = 2.0f, c = 3.0f, d = 4.0f;
        for (int i = 0; i < n_words * 32 * 50; i++) {
            a = __fmaf_rn(a, b, c);
            b = __fmaf_rn(b, c, d);
            c = __fmaf_rn(c, d, a);
            d = __fmaf_rn(d, a, b);
        }
        if (a == -999.0f) bits_out[0] = __float_as_uint(a);
    }
}

// ============================================================================
// Host helpers
// ============================================================================

struct NoiseStats {
    double slow_rate;
    double entropy;  // Shannon entropy of binary (fast/slow) stream
    double acf1;     // Autocorrelation at lag 1
    int total_slow;
    int total;
};

NoiseStats analyze_noise(const std::vector<uint64_t>& data, int offset, int count) {
    NoiseStats ns = {};
    ns.total = count;

    uint64_t min_val = UINT64_MAX;
    for (int i = 0; i < count; i++)
        if (data[offset + i] < min_val) min_val = data[offset + i];

    // Binary classification: fast (=min) or slow (>min)
    std::vector<int> binary(count);
    for (int i = 0; i < count; i++) {
        binary[i] = (data[offset + i] > min_val) ? 1 : 0;
        if (binary[i]) ns.total_slow++;
    }
    ns.slow_rate = (double)ns.total_slow / count;

    // Shannon entropy
    double p1 = ns.slow_rate;
    double p0 = 1.0 - p1;
    if (p0 > 0 && p1 > 0)
        ns.entropy = -(p0 * log2(p0) + p1 * log2(p1));
    else
        ns.entropy = 0;

    // Autocorrelation at lag 1
    double mean = ns.slow_rate;
    double num = 0, den = 0;
    for (int i = 0; i < count - 1; i++) {
        num += (binary[i] - mean) * (binary[i+1] - mean);
    }
    for (int i = 0; i < count; i++) {
        den += (binary[i] - mean) * (binary[i] - mean);
    }
    ns.acf1 = (den > 0) ? num / den : 0;

    return ns;
}

// Popcount for uint32_t
int popcount32(uint32_t x) {
    int c = 0;
    while (x) { c += x & 1; x >>= 1; }
    return c;
}

int main() {
    fprintf(stderr, "═══ Thermal-Controlled L1 Jitter as Analog Noise Source ═══\n\n");

    uint64_t *d_out;
    uint32_t *d_data, *d_bits;
    size_t out_size = 16 * LANES * ITERS * sizeof(uint64_t);  // room for multi-CU
    size_t data_size = 16 * 64 * sizeof(uint32_t);

    hipMalloc(&d_out, out_size);
    hipMalloc(&d_data, data_size);
    hipMalloc(&d_bits, LANES * 1024 * sizeof(uint32_t));

    std::vector<uint64_t> h_out(16 * LANES * ITERS);
    std::vector<uint32_t> h_data(16 * 64);
    for (size_t i = 0; i < h_data.size(); i++) h_data[i] = i + 42;
    hipMemcpy(d_data, h_data.data(), data_size, hipMemcpyHostToDevice);

    printf("test,param,value,slow_rate,entropy,acf1\n");

    // ====================================================================
    // TEST 1: Fine-grained thermal modulation curve (0..15 waves)
    // ====================================================================
    fprintf(stderr, "═══ TEST 1: Thermal Modulation Curve ═══\n");
    for (int hw = 0; hw <= 15; hw++) {
        hipMemset(d_out, 0, LANES * ITERS * sizeof(uint64_t));
        int threads = (hw + 1) * 64;
        thermal_modulation<<<1, threads>>>(d_out, d_data, ITERS, hw);
        hipDeviceSynchronize();
        hipMemcpy(h_out.data(), d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);

        // Aggregate all lanes
        int total_slow = 0;
        double total_entropy = 0;
        double total_acf1 = 0;
        for (int lane = 0; lane < LANES; lane++) {
            NoiseStats ns = analyze_noise(h_out, lane * ITERS, ITERS);
            total_slow += ns.total_slow;
            total_entropy += ns.entropy;
            total_acf1 += ns.acf1;
        }
        double rate = (double)total_slow / (LANES * ITERS);
        double avg_entropy = total_entropy / LANES;
        double avg_acf1 = total_acf1 / LANES;
        fprintf(stderr, "  %2d waves: slow=%.3f%%, entropy=%.4f bits, acf1=%.4f\n",
                hw, rate * 100, avg_entropy, avg_acf1);
        printf("thermal_curve,%d,%d,%.6f,%.6f,%.6f\n", hw, total_slow, rate, avg_entropy, avg_acf1);
    }
    fprintf(stderr, "\n");

    // ====================================================================
    // TEST 2: Multi-CU independence
    // ====================================================================
    fprintf(stderr, "═══ TEST 2: Multi-CU Independence ═══\n");
    int n_cus = 8;
    int waves_per_cu = 8;  // moderate thermal load
    int threads_per_cu = (waves_per_cu + 1) * 64;

    hipMemset(d_out, 0, n_cus * LANES * ITERS * sizeof(uint64_t));
    multi_cu_probe<<<n_cus, threads_per_cu>>>(d_out, d_data, ITERS, waves_per_cu);
    hipDeviceSynchronize();
    hipMemcpy(h_out.data(), d_out, n_cus * LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);

    // Per-CU noise rates
    std::vector<double> cu_rates(n_cus);
    for (int cu = 0; cu < n_cus; cu++) {
        int cu_slow = 0;
        for (int lane = 0; lane < LANES; lane++) {
            NoiseStats ns = analyze_noise(h_out, (cu * LANES + lane) * ITERS, ITERS);
            cu_slow += ns.total_slow;
        }
        cu_rates[cu] = (double)cu_slow / (LANES * ITERS);
        fprintf(stderr, "  CU %d: slow=%.3f%%\n", cu, cu_rates[cu] * 100);
        printf("multi_cu,%d,%d,%.6f,,\n", cu, cu_slow, cu_rates[cu]);
    }

    // Cross-CU correlation: are CU0 and CUn slow at the same iterations?
    fprintf(stderr, "  Cross-CU correlation (lane 0 of each CU):\n");
    for (int cu = 1; cu < n_cus; cu++) {
        int both_slow = 0, cu0_slow = 0;
        uint64_t cu0_min = UINT64_MAX, cun_min = UINT64_MAX;
        for (int i = 0; i < ITERS; i++) {
            if (h_out[0 * ITERS + i] < cu0_min) cu0_min = h_out[0 * ITERS + i];
            if (h_out[(cu * LANES) * ITERS + i] < cun_min) cun_min = h_out[(cu * LANES) * ITERS + i];
        }
        for (int i = 0; i < ITERS; i++) {
            bool s0 = h_out[0 * ITERS + i] > cu0_min;
            bool sn = h_out[(cu * LANES) * ITERS + i] > cun_min;
            if (s0) { cu0_slow++; if (sn) both_slow++; }
        }
        double cond = (cu0_slow > 0) ? (double)both_slow / cu0_slow : 0;
        double base = cu_rates[cu];
        fprintf(stderr, "    CU0↔CU%d: P(CU%d slow | CU0 slow)=%.1f%% (baseline=%.1f%%)\n",
                cu, cu, cond * 100, base * 100);
        printf("cu_corr,%d,%.6f,%.6f,,\n", cu, cond, base);
    }
    fprintf(stderr, "\n");

    // ====================================================================
    // TEST 3: Dynamic switching — does noise respond to thermal changes?
    // ====================================================================
    fprintf(stderr, "═══ TEST 3: Dynamic Thermal Switching ═══\n");
    int phase_lengths[] = {100, 500, 1000, 5000, 10000};
    for (int pl = 0; pl < 5; pl++) {
        int plen = phase_lengths[pl];
        hipMemset(d_out, 0, LANES * ITERS * sizeof(uint64_t));
        // 8 heavy waves + 1 probe
        dynamic_switch<<<1, 9 * 64>>>(d_out, d_data, ITERS, plen);
        hipDeviceSynchronize();
        hipMemcpy(h_out.data(), d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);

        // Split into first half and second half — check if switching creates two modes
        NoiseStats first_half = analyze_noise(h_out, 0, ITERS / 2);
        NoiseStats second_half = analyze_noise(h_out, ITERS / 2, ITERS / 2);
        NoiseStats full = analyze_noise(h_out, 0, ITERS);

        fprintf(stderr, "  phase=%5d: full=%.3f%%, first_half=%.3f%%, second_half=%.3f%%, entropy=%.4f\n",
                plen, full.slow_rate * 100, first_half.slow_rate * 100, second_half.slow_rate * 100, full.entropy);
        printf("dynamic,%d,%.6f,%.6f,%.6f,%.6f\n", plen, full.slow_rate, first_half.slow_rate, second_half.slow_rate, full.entropy);
    }
    fprintf(stderr, "\n");

    // ====================================================================
    // TEST 4: Noise quality at thermal sweet spots
    // Pick 3 thermal levels targeting ~10%, ~50%, ~80% slow rate
    // Measure: entropy, autocorrelation lag 1-10, runs test, unique values
    // ====================================================================
    fprintf(stderr, "═══ TEST 4: Noise Quality at Thermal Sweet Spots ═══\n");
    int sweet_spots[] = {2, 4, 8, 12, 15};
    for (int ss = 0; ss < 5; ss++) {
        int hw = sweet_spots[ss];
        hipMemset(d_out, 0, LANES * ITERS * sizeof(uint64_t));
        thermal_modulation<<<1, (hw + 1) * 64>>>(d_out, d_data, ITERS, hw);
        hipDeviceSynchronize();
        hipMemcpy(h_out.data(), d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);

        NoiseStats ns = analyze_noise(h_out, 0, ITERS);

        // Detailed autocorrelation
        fprintf(stderr, "  %2d waves (rate=%.1f%%): entropy=%.4f, acf1=%.4f\n",
                hw, ns.slow_rate * 100, ns.entropy, ns.acf1);

        // Count unique timing values
        uint64_t min_v = UINT64_MAX, max_v = 0;
        std::vector<uint64_t> vals(ITERS);
        for (int i = 0; i < ITERS; i++) {
            vals[i] = h_out[i];
            if (vals[i] < min_v) min_v = vals[i];
            if (vals[i] > max_v) max_v = vals[i];
        }
        std::sort(vals.begin(), vals.end());
        int unique = std::unique(vals.begin(), vals.end()) - vals.begin();
        fprintf(stderr, "    unique=%d, range=[%lu,%lu]\n", unique, min_v, max_v);

        // ACF at lags 1..10
        fprintf(stderr, "    ACF: ");
        for (int lag = 1; lag <= 10; lag++) {
            double mean = 0;
            for (int i = 0; i < ITERS; i++) mean += h_out[i];
            mean /= ITERS;
            double num = 0, den = 0;
            for (int i = 0; i < ITERS - lag; i++) {
                num += (h_out[i] - mean) * (h_out[i + lag] - mean);
            }
            for (int i = 0; i < ITERS; i++) {
                den += (h_out[i] - mean) * (h_out[i] - mean);
            }
            double acf = (den > 0) ? num / den : 0;
            fprintf(stderr, "r(%d)=%.3f ", lag, acf);
        }
        fprintf(stderr, "\n");

        printf("quality,%d,%.6f,%.6f,%.6f,%.6f\n", hw, ns.slow_rate, ns.entropy, ns.acf1, (double)unique);
    }
    fprintf(stderr, "\n");

    // ====================================================================
    // TEST 5: Noise harvest — convert jitter to random bits
    // ====================================================================
    fprintf(stderr, "═══ TEST 5: Random Bit Harvest ═══\n");
    int n_words = 512;  // 512 × 32 = 16384 bits per lane
    hipMemset(d_bits, 0, LANES * n_words * sizeof(uint32_t));
    // Use 8 heavy waves for ~60-70% bit rate
    noise_harvest<<<1, 9 * 64>>>(d_bits, d_data, n_words);
    hipDeviceSynchronize();

    std::vector<uint32_t> h_bits(LANES * n_words);
    hipMemcpy(h_bits.data(), d_bits, LANES * n_words * sizeof(uint32_t), hipMemcpyDeviceToHost);

    // Per-lane bit statistics
    fprintf(stderr, "  Per-lane bit density (ideal=50%%):\n");
    double total_density = 0;
    for (int lane = 0; lane < 8; lane++) {  // show first 8 lanes
        int ones = 0;
        for (int w = 0; w < n_words; w++) {
            ones += popcount32(h_bits[lane * n_words + w]);
        }
        double density = (double)ones / (n_words * 32);
        total_density += density;
        fprintf(stderr, "    Lane %d: %.2f%% ones\n", lane, density * 100);
        printf("harvest,%d,%.6f,,,\n", lane, density);
    }

    // Cross-lane independence: XOR lanes 0 and 1, check if result is more random
    int xor_ones = 0;
    for (int w = 0; w < n_words; w++) {
        uint32_t x = h_bits[0 * n_words + w] ^ h_bits[1 * n_words + w];
        xor_ones += popcount32(x);
    }
    double xor_density = (double)xor_ones / (n_words * 32);
    fprintf(stderr, "  XOR(L0,L1) density: %.2f%% (expect ~50%% if independent)\n", xor_density * 100);
    printf("harvest_xor,01,%.6f,,,\n", xor_density);

    // Monobit test on lane 0
    int lane0_ones = 0;
    for (int w = 0; w < n_words; w++) lane0_ones += popcount32(h_bits[w]);
    int n_bits = n_words * 32;
    double monobit_z = fabs((double)(2 * lane0_ones - n_bits)) / sqrt(n_bits);
    fprintf(stderr, "  Monobit test (lane 0): z=%.3f (PASS if < 1.96)\n", monobit_z);
    printf("monobit,0,%.6f,,,\n", monobit_z);

    // Serial test: count 00, 01, 10, 11 pairs
    int pairs[4] = {0, 0, 0, 0};
    for (int w = 0; w < n_words; w++) {
        uint32_t word = h_bits[w];
        for (int b = 0; b < 31; b++) {
            int pair = ((word >> b) & 3);
            pairs[pair]++;
        }
    }
    int total_pairs = n_words * 31;
    fprintf(stderr, "  Serial test: 00=%.2f%% 01=%.2f%% 10=%.2f%% 11=%.2f%% (ideal=25%%)\n",
            100.0 * pairs[0] / total_pairs, 100.0 * pairs[1] / total_pairs,
            100.0 * pairs[2] / total_pairs, 100.0 * pairs[3] / total_pairs);
    printf("serial,00,%.6f,,,\n", (double)pairs[0] / total_pairs);
    printf("serial,01,%.6f,,,\n", (double)pairs[1] / total_pairs);
    printf("serial,10,%.6f,,,\n", (double)pairs[2] / total_pairs);
    printf("serial,11,%.6f,,,\n", (double)pairs[3] / total_pairs);

    fprintf(stderr, "\n");

    // ====================================================================
    // TEST 6: Throughput — how many noise bits/second can we harvest?
    // ====================================================================
    fprintf(stderr, "═══ TEST 6: Noise Throughput ═══\n");
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    int harvest_words = 1024;
    hipMemset(d_bits, 0, LANES * harvest_words * sizeof(uint32_t));

    hipEventRecord(start);
    noise_harvest<<<1, 9 * 64>>>(d_bits, d_data, harvest_words);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0;
    hipEventElapsedTime(&ms, start, stop);
    int total_bits = LANES * harvest_words * 32;
    double bits_per_sec = total_bits / (ms / 1000.0);
    fprintf(stderr, "  %d bits in %.2f ms = %.1f Kbits/sec (%.1f bits/sec/lane)\n",
            total_bits, ms, bits_per_sec / 1000.0, bits_per_sec / LANES);
    printf("throughput,total,%d,%.3f,,\n", total_bits, bits_per_sec);

    hipEventDestroy(start);
    hipEventDestroy(stop);

    hipFree(d_out);
    hipFree(d_data);
    hipFree(d_bits);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
