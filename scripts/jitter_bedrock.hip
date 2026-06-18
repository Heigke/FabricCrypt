// jitter_bedrock.hip — Probe the ABSOLUTE LOWEST noise sources in gfx1151
// Below L1 cache jitter: PLL phase noise, DRAM refresh, wavefront scheduler,
// fabric latency, memory controller bank conflicts
//
// These are BELOW the instruction-level tests we already did.
// We use differential timing to detect sub-cycle jitter.
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_bedrock scripts/jitter_bedrock.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_bedrock

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <map>

#define ITERS 40000
#define LANES 64

// ============================================================================
// TEST 1: PLL PHASE NOISE
// If clock64() has sub-cycle jitter from PLL phase noise, back-to-back
// clock reads should occasionally show +/-1 variation. We measure the
// distribution of (clock64_B - clock64_A) for pure overhead with NOTHING
// in between — just two consecutive clock reads.
// If PLL is perfect: always same delta. If phase noise: occasional +/-1.
// We also measure DIFFERENTIAL: (delta_i - delta_{i-1}) to amplify drift.
// ============================================================================
__global__ void pll_phase_noise(uint64_t* out, int iters) {
    int tid = threadIdx.x;
    if (tid >= LANES) return;

    for (int i = 0; i < iters; i++) {
        uint64_t t0 = __clock64();
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
    }
}

// Same but with wall_clock64 (different clock domain — system clock vs shader clock)
__global__ void pll_wall_clock_noise(uint64_t* out, int iters) {
    int tid = threadIdx.x;
    if (tid >= LANES) return;

    for (int i = 0; i < iters; i++) {
        uint64_t t0 = wall_clock64();
        uint64_t t1 = wall_clock64();
        out[tid * iters + i] = t1 - t0;
    }
}

// Cross-domain: measure skew between shader clock and wall clock
// If PLLs drift independently, delta(shader - wall) should wander
__global__ void pll_cross_domain(uint64_t* out_shader, uint64_t* out_wall, int iters) {
    int tid = threadIdx.x;
    if (tid >= LANES) return;

    for (int i = 0; i < iters; i++) {
        uint64_t s0 = __clock64();
        uint64_t w0 = wall_clock64();
        out_shader[tid * iters + i] = s0;
        out_wall[tid * iters + i] = w0;
    }
}

// ============================================================================
// TEST 2: DRAM REFRESH INTERFERENCE
// DRAM refreshes every ~64ms (tREFI). During refresh, banks are unavailable.
// We do continuous DRAM reads (large stride to bypass all caches) and look
// for periodic spikes at ~64ms intervals.
// Using 16MB stride to ensure DRAM, not L2.
// ============================================================================
__global__ void dram_refresh_probe(uint64_t* out, uint32_t* data, int iters, int stride) {
    int tid = threadIdx.x;
    if (tid >= LANES) return;

    uint32_t sum = 0;
    int mask = (stride * 256) - 1;  // wrap around within allocation

    for (int i = 0; i < iters; i++) {
        int addr = ((tid + i * stride) * 128) & mask;  // stride through DRAM
        uint64_t t0 = __clock64();
        sum += data[addr];
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
    }
    if (sum == 0xDEADBEEF) out[0] = sum;
}

// ============================================================================
// TEST 3: WAVEFRONT SCHEDULER NON-DETERMINISM
// Launch identical kernels back-to-back and measure total execution time.
// If the wavefront scheduler is non-deterministic, identical workloads
// will have varying completion times.
// We use a trivial ALU-only kernel (no memory) so any variation comes
// purely from scheduling/dispatch.
// ============================================================================
__global__ void scheduler_probe(uint64_t* out, int work_amount) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t t0 = __clock64();

    // Pure ALU work — deterministic compute
    uint32_t x = tid + 1;
    for (int i = 0; i < work_amount; i++) {
        x = x * 1103515245 + 12345;
    }

    uint64_t t1 = __clock64();

    // Each thread records its own start-to-end
    if (tid < LANES * 4) {
        out[tid] = t1 - t0;
    }

    // Prevent optimization
    if (x == 0xDEADBEEF) out[0] = x;
}

// ============================================================================
// TEST 4: CPU↔GPU FABRIC LATENCY JITTER
// On APU, CPU and GPU share memory through the data fabric (Infinity Fabric).
// We measure the jitter of GPU reading CPU-pinned (system) memory vs
// GPU-local (device) memory. The fabric hop adds latency AND jitter.
// ============================================================================
__global__ void fabric_latency_probe(uint64_t* out, uint32_t* system_mem,
                                      uint32_t* device_mem, int iters, int use_system) {
    int tid = threadIdx.x;
    if (tid >= LANES) return;

    uint32_t* ptr = use_system ? system_mem : device_mem;
    uint32_t sum = 0;

    for (int i = 0; i < iters; i++) {
        // Read from cache line that maps to tid
        uint64_t t0 = __clock64();
        sum += ptr[tid * 32];  // 128-byte stride = different cache lines
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
    }
    if (sum == 0xDEADBEEF) out[0] = sum;
}

// ============================================================================
// TEST 5: MEMORY CONTROLLER BANK CONFLICT JITTER
// DRAM has multiple banks (typically 16-32). Accessing the same bank from
// different threads causes conflict-induced jitter. We compare:
//   (a) All threads hit SAME bank (maximum conflict)
//   (b) Threads spread across ALL banks (minimum conflict)
// The timing distribution difference reveals MC scheduling noise.
// Bank bits are typically address[9:7] or similar on RDNA.
// ============================================================================
__global__ void mc_bank_conflict(uint64_t* out, uint32_t* data, int iters,
                                  int same_bank) {
    int tid = threadIdx.x;
    if (tid >= LANES) return;

    uint32_t sum = 0;

    for (int i = 0; i < iters; i++) {
        int addr;
        if (same_bank) {
            // All threads access same bank: stride by full bank size (8KB typical)
            // addr[13:7] determines bank → keep those bits same
            addr = (i * 8192 + tid * 4) / 4;  // same bank, different row
        } else {
            // Spread across banks: each thread different bank
            addr = (i * 128 + tid * 8192) / 4;  // different banks
        }
        addr = addr & ((16 * 1024 * 1024 / 4) - 1);  // wrap to 16MB

        uint64_t t0 = __clock64();
        sum += data[addr];
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
    }
    if (sum == 0xDEADBEEF) out[0] = sum;
}

// ============================================================================
// TEST 6: INSTRUCTION PIPELINE BUBBLE NOISE
// Even in a "deterministic" ALU, pipeline bubbles from resource conflicts
// (VGPR bank conflicts, SGPR read-after-write hazards, execution port
// contention) can create micro-jitter. We test sequences designed to
// stress specific pipeline hazards:
//   (a) VGPR bank conflict: consecutive reads from same VGPR bank
//   (b) Dependent chain: long RAW dependency chain
//   (c) Mixed pipe: interleave VALU + SALU + memory
// ============================================================================
__global__ void pipeline_bubble_probe(uint64_t* out, int iters, int pattern) {
    int tid = threadIdx.x;
    if (tid >= LANES) return;

    for (int i = 0; i < iters; i++) {
        uint64_t t0 = __clock64();

        if (pattern == 0) {
            // Pattern A: VGPR bank conflict — read v0,v4,v8,v12 (same bank on 4-bank VGPR)
            asm volatile(
                "v_add_f32 v0, v0, v4\n"
                "v_add_f32 v0, v0, v8\n"
                "v_add_f32 v0, v0, v12\n"
                "v_add_f32 v0, v0, v16\n"
                "v_add_f32 v0, v0, v20\n"
                "v_add_f32 v0, v0, v24\n"
                "v_add_f32 v0, v0, v28\n"
                "v_add_f32 v0, v0, v4\n"
                : : : "v0"
            );
        } else if (pattern == 1) {
            // Pattern B: Long RAW chain — each op depends on previous
            asm volatile(
                "v_mul_f32 v0, v0, 0x3f800001\n"  // v0 = v0 * 1.0+eps
                "v_add_f32 v0, v0, 0x33800000\n"  // v0 += tiny
                "v_mul_f32 v0, v0, 0x3f800001\n"
                "v_add_f32 v0, v0, 0x33800000\n"
                "v_mul_f32 v0, v0, 0x3f800001\n"
                "v_add_f32 v0, v0, 0x33800000\n"
                "v_mul_f32 v0, v0, 0x3f800001\n"
                "v_add_f32 v0, v0, 0x33800000\n"
                : : : "v0"
            );
        } else {
            // Pattern C: Mixed VALU + SALU — cross-pipe interaction
            asm volatile(
                "v_add_f32 v0, v0, v1\n"
                "s_add_u32 s0, s0, 1\n"
                "v_mul_f32 v0, v0, v2\n"
                "s_mul_i32 s0, s0, 3\n"
                "v_add_f32 v0, v0, v3\n"
                "s_add_u32 s0, s0, 1\n"
                "v_mul_f32 v0, v0, v4\n"
                "s_mul_i32 s0, s0, 3\n"
                : : : "v0", "s0"
            );
        }

        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
    }
}

// ============================================================================
// TEST 7: INTER-WAVEFRONT TIMING INTERFERENCE
// Launch many wavefronts on same CU and measure if they create timing
// interference on each other through shared resources (LDS, L1, VGPR file).
// Unlike cross-block tests, these are wavefronts within the SAME workgroup.
// ============================================================================
__global__ void intra_wg_interference(uint64_t* out, int iters, int active_waves) {
    int wave_id = threadIdx.x / 64;
    int lane = threadIdx.x % 64;

    if (wave_id >= active_waves) return;
    if (lane >= 32) return;  // only first 32 lanes measure

    // Wave 0 does L1 reads (measurement), other waves do ALU (interference)
    if (wave_id == 0) {
        // Measurement wave: L1 cache reads
        __shared__ uint32_t shared_data[256];
        shared_data[lane] = lane;
        __syncthreads();

        for (int i = 0; i < iters; i++) {
            uint64_t t0 = __clock64();
            uint32_t v = shared_data[lane];
            uint64_t t1 = __clock64();
            out[lane * iters + i] = t1 - t0;
            if (v == 0xDEAD) out[0] = 0;
        }
    } else {
        // Interference waves: pure ALU
        uint32_t x = threadIdx.x;
        for (int i = 0; i < iters; i++) {
            x = x * 1103515245 + 12345;
            x = x ^ (x >> 16);
            x = x * 2654435761u;
        }
        if (x == 0xDEAD) out[0] = x;
    }
}

// ============================================================================
// TEST 8: CLOCK COUNTER GRANULARITY & MONOTONICITY
// Probe the absolute resolution of clock64() and wall_clock64().
// Read clock in tight loop, record all deltas. Check for:
//   - Minimum non-zero delta (clock granularity)
//   - Non-monotonic readings (clock glitches)
//   - Delta distribution (uniform? bimodal? discrete?)
// ============================================================================
__global__ void clock_granularity(uint64_t* out_delta, uint64_t* out_raw, int iters) {
    int tid = threadIdx.x;
    if (tid >= 1) return;  // single thread to avoid contention

    uint64_t prev = __clock64();
    for (int i = 0; i < iters; i++) {
        uint64_t now = __clock64();
        out_delta[i] = now - prev;
        out_raw[i] = now;
        prev = now;
    }
}

// Same for wall_clock64
__global__ void wall_clock_granularity(uint64_t* out_delta, uint64_t* out_raw, int iters) {
    int tid = threadIdx.x;
    if (tid >= 1) return;

    uint64_t prev = wall_clock64();
    for (int i = 0; i < iters; i++) {
        uint64_t now = wall_clock64();
        out_delta[i] = now - prev;
        out_raw[i] = now;
        prev = now;
    }
}

// ============================================================================
// HOST ANALYSIS
// ============================================================================

struct Stats {
    double mean, std, min_val, max_val;
    int unique;
    std::vector<int> histogram;  // count per value
};

Stats compute_stats(uint64_t* data, int n) {
    Stats s;
    // Filter out overflow values
    std::vector<double> valid;
    for (int i = 0; i < n; i++) {
        double v = (double)data[i];
        if (v < 1e12 && v > 0) valid.push_back(v);
    }
    if (valid.empty()) {
        s.mean = 0; s.std = 0; s.min_val = 0; s.max_val = 0; s.unique = 0;
        return s;
    }

    double sum = 0;
    s.min_val = valid[0]; s.max_val = valid[0];
    for (double v : valid) {
        sum += v;
        if (v < s.min_val) s.min_val = v;
        if (v > s.max_val) s.max_val = v;
    }
    s.mean = sum / valid.size();

    double var = 0;
    for (double v : valid) var += (v - s.mean) * (v - s.mean);
    s.std = sqrt(var / valid.size());

    // Count unique values
    std::vector<double> sorted = valid;
    std::sort(sorted.begin(), sorted.end());
    s.unique = std::unique(sorted.begin(), sorted.end()) - sorted.begin();

    return s;
}

// Compute autocorrelation at lag k
double autocorr(uint64_t* data, int n, int lag) {
    std::vector<double> valid;
    for (int i = 0; i < n; i++) {
        double v = (double)data[i];
        if (v < 1e12 && v > 0) valid.push_back(v);
    }
    if ((int)valid.size() < lag + 10) return 0;

    double mean = 0;
    for (double v : valid) mean += v;
    mean /= valid.size();

    double num = 0, den = 0;
    for (int i = 0; i < (int)valid.size() - lag; i++) {
        num += (valid[i] - mean) * (valid[i+lag] - mean);
    }
    for (int i = 0; i < (int)valid.size(); i++) {
        den += (valid[i] - mean) * (valid[i] - mean);
    }
    return den > 0 ? num / den : 0;
}

// Value distribution histogram
void print_distribution(uint64_t* data, int n, const char* label) {
    int valid_n = 0;
    uint64_t lo = UINT64_MAX, hi = 0;
    for (int i = 0; i < n; i++) {
        uint64_t v = data[i];
        if (v < 1000000 && v > 0) {
            valid_n++;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
    }
    if (valid_n == 0) { fprintf(stderr, "  %s: no valid data\n", label); return; }

    // Count each value
    std::vector<int> hist(hi - lo + 1, 0);
    for (int i = 0; i < n; i++) {
        uint64_t v = data[i];
        if (v >= lo && v <= hi) hist[v - lo]++;
    }

    fprintf(stderr, "  %s distribution [%lu..%lu] (%d samples):\n", label, lo, hi, valid_n);
    for (uint64_t v = lo; v <= hi && v < lo + 20; v++) {
        if (hist[v - lo] > 0) {
            double pct = 100.0 * hist[v - lo] / valid_n;
            fprintf(stderr, "    %3lu: %6d (%5.2f%%)", v, hist[v - lo], pct);
            // Bar chart
            int bar = (int)(pct / 2);
            fprintf(stderr, " ");
            for (int b = 0; b < bar && b < 40; b++) fprintf(stderr, "█");
            fprintf(stderr, "\n");
        }
    }
}

#include <map>

int main() {
    fprintf(stderr, "═══ BEDROCK: Lowest-Level GPU Noise Sources ═══\n\n");

    // ===== TEST 1: PLL PHASE NOISE =====
    fprintf(stderr, "═══ TEST 1: PLL Phase Noise (clock64 back-to-back) ═══\n");
    {
        uint64_t *d_out;
        hipMalloc(&d_out, LANES * ITERS * sizeof(uint64_t));
        hipMemset(d_out, 0, LANES * ITERS * sizeof(uint64_t));

        pll_phase_noise<<<1, LANES>>>(d_out, ITERS);
        hipDeviceSynchronize();

        uint64_t* h_out = new uint64_t[LANES * ITERS];
        hipMemcpy(h_out, d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);

        // Analyze lane 0
        Stats s = compute_stats(h_out, ITERS);
        fprintf(stderr, "  Shader clock: mean=%.2f std=%.4f min=%.0f max=%.0f unique=%d\n",
                s.mean, s.std, s.min_val, s.max_val, s.unique);
        print_distribution(h_out, ITERS, "shader_clock");

        // Cross-lane comparison: do all lanes see same delta?
        int same_count = 0;
        for (int i = 0; i < ITERS; i++) {
            bool all_same = true;
            uint64_t ref = h_out[i];  // lane 0
            for (int l = 1; l < LANES; l++) {
                if (h_out[l * ITERS + i] != ref) { all_same = false; break; }
            }
            if (all_same) same_count++;
        }
        fprintf(stderr, "  Cross-lane unanimity: %.1f%%\n", 100.0 * same_count / ITERS);

        // ACF
        fprintf(stderr, "  ACF:");
        for (int lag = 1; lag <= 10; lag++)
            fprintf(stderr, " r(%d)=%.4f", lag, autocorr(h_out, ITERS, lag));
        fprintf(stderr, "\n");

        // Differential analysis: delta[i] - delta[i-1]
        std::vector<int64_t> diffs;
        for (int i = 1; i < ITERS; i++) {
            if (h_out[i] < 1e12 && h_out[i-1] < 1e12) {
                diffs.push_back((int64_t)h_out[i] - (int64_t)h_out[i-1]);
            }
        }
        int nonzero_diffs = 0;
        for (auto d : diffs) if (d != 0) nonzero_diffs++;
        fprintf(stderr, "  Differential: %d/%d non-zero (%.2f%%)\n",
                nonzero_diffs, (int)diffs.size(), 100.0 * nonzero_diffs / diffs.size());

        delete[] h_out;
        hipFree(d_out);
    }

    // Now wall_clock64
    fprintf(stderr, "\n═══ TEST 1b: PLL Phase Noise (wall_clock64 back-to-back) ═══\n");
    {
        uint64_t *d_out;
        hipMalloc(&d_out, LANES * ITERS * sizeof(uint64_t));
        hipMemset(d_out, 0, LANES * ITERS * sizeof(uint64_t));

        pll_wall_clock_noise<<<1, LANES>>>(d_out, ITERS);
        hipDeviceSynchronize();

        uint64_t* h_out = new uint64_t[LANES * ITERS];
        hipMemcpy(h_out, d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);

        Stats s = compute_stats(h_out, ITERS);
        fprintf(stderr, "  Wall clock: mean=%.2f std=%.4f min=%.0f max=%.0f unique=%d\n",
                s.mean, s.std, s.min_val, s.max_val, s.unique);
        print_distribution(h_out, ITERS, "wall_clock");

        fprintf(stderr, "  ACF:");
        for (int lag = 1; lag <= 10; lag++)
            fprintf(stderr, " r(%d)=%.4f", lag, autocorr(h_out, ITERS, lag));
        fprintf(stderr, "\n");

        delete[] h_out;
        hipFree(d_out);
    }

    // Cross-domain
    fprintf(stderr, "\n═══ TEST 1c: PLL Cross-Domain Drift (shader vs wall clock) ═══\n");
    {
        uint64_t *d_shader, *d_wall;
        hipMalloc(&d_shader, LANES * ITERS * sizeof(uint64_t));
        hipMalloc(&d_wall, LANES * ITERS * sizeof(uint64_t));

        pll_cross_domain<<<1, LANES>>>(d_shader, d_wall, ITERS);
        hipDeviceSynchronize();

        uint64_t* h_s = new uint64_t[LANES * ITERS];
        uint64_t* h_w = new uint64_t[LANES * ITERS];
        hipMemcpy(h_s, d_shader, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);
        hipMemcpy(h_w, d_wall, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);

        // Compute drift: (shader[i] - wall[i]) - (shader[0] - wall[0])
        int64_t base_diff = (int64_t)h_s[0] - (int64_t)h_w[0];
        double max_drift = 0, sum_drift = 0;
        int drift_changes = 0;
        int64_t prev_drift = 0;
        for (int i = 0; i < ITERS; i++) {
            int64_t drift = ((int64_t)h_s[i] - (int64_t)h_w[i]) - base_diff;
            double d = fabs((double)drift);
            if (d > max_drift) max_drift = d;
            sum_drift += d;
            if (i > 0 && drift != prev_drift) drift_changes++;
            prev_drift = drift;
        }
        fprintf(stderr, "  Max drift: %.0f cycles over %d samples\n", max_drift, ITERS);
        fprintf(stderr, "  Mean |drift|: %.2f cycles\n", sum_drift / ITERS);
        fprintf(stderr, "  Drift changes: %d/%d (%.1f%%)\n",
                drift_changes, ITERS, 100.0 * drift_changes / ITERS);

        delete[] h_s;
        delete[] h_w;
        hipFree(d_shader);
        hipFree(d_wall);
    }

    // ===== TEST 2: DRAM REFRESH =====
    fprintf(stderr, "\n═══ TEST 2: DRAM Refresh Interference ═══\n");
    {
        // Allocate 64MB to ensure DRAM access
        size_t alloc_size = 64 * 1024 * 1024;
        uint32_t *d_data, *d_out_raw;
        uint64_t *d_out;
        hipMalloc(&d_data, alloc_size);
        hipMalloc(&d_out, LANES * ITERS * sizeof(uint64_t));
        hipMemset(d_data, 0xAB, alloc_size);

        // Large stride to bypass L2 (16MB stride)
        dram_refresh_probe<<<1, LANES>>>(d_out, d_data, ITERS, 32768);
        hipDeviceSynchronize();

        uint64_t* h_out = new uint64_t[LANES * ITERS];
        hipMemcpy(h_out, d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);

        Stats s = compute_stats(h_out, ITERS);
        fprintf(stderr, "  DRAM reads: mean=%.1f std=%.2f min=%.0f max=%.0f unique=%d CV=%.4f\n",
                s.mean, s.std, s.min_val, s.max_val, s.unique, s.std/s.mean);
        print_distribution(h_out, ITERS, "dram_timing");

        // Look for periodic spikes (DRAM refresh at ~64ms)
        // At ~2GHz, 64ms = ~128M cycles. With ITERS=40000 reads, each ~200 cycles,
        // that's ~8us per iter, so refresh every ~8000 iterations.
        // Look for spikes > 2*mean
        double threshold = s.mean * 1.5;
        int spike_count = 0;
        std::vector<int> spike_positions;
        for (int i = 0; i < ITERS; i++) {
            if ((double)h_out[i] > threshold && h_out[i] < 1e12) {
                spike_count++;
                spike_positions.push_back(i);
            }
        }
        fprintf(stderr, "  Spikes (>1.5×mean): %d/%d (%.2f%%)\n",
                spike_count, ITERS, 100.0 * spike_count / ITERS);

        // Check periodicity of spikes
        if (spike_positions.size() >= 3) {
            fprintf(stderr, "  Spike intervals (first 10): ");
            for (int i = 1; i < (int)spike_positions.size() && i <= 10; i++) {
                fprintf(stderr, "%d ", spike_positions[i] - spike_positions[i-1]);
            }
            fprintf(stderr, "\n");
        }

        fprintf(stderr, "  ACF:");
        for (int lag = 1; lag <= 10; lag++)
            fprintf(stderr, " r(%d)=%.4f", lag, autocorr(h_out, ITERS, lag));
        fprintf(stderr, "\n");

        delete[] h_out;
        hipFree(d_data);
        hipFree(d_out);
    }

    // ===== TEST 3: WAVEFRONT SCHEDULER =====
    fprintf(stderr, "\n═══ TEST 3: Wavefront Scheduler Non-Determinism ═══\n");
    {
        const int LAUNCHES = 200;
        const int BLOCKS = 4;
        const int THREADS = 64;
        uint64_t *d_out;
        hipMalloc(&d_out, BLOCKS * THREADS * sizeof(uint64_t));

        std::vector<double> launch_means(LAUNCHES);
        std::vector<double> launch_stds(LAUNCHES);

        for (int l = 0; l < LAUNCHES; l++) {
            hipMemset(d_out, 0, BLOCKS * THREADS * sizeof(uint64_t));
            scheduler_probe<<<BLOCKS, THREADS>>>(d_out, 10000);
            hipDeviceSynchronize();

            uint64_t* h_out = new uint64_t[BLOCKS * THREADS];
            hipMemcpy(h_out, d_out, BLOCKS * THREADS * sizeof(uint64_t), hipMemcpyDeviceToHost);

            Stats s = compute_stats(h_out, BLOCKS * THREADS);
            launch_means[l] = s.mean;
            launch_stds[l] = s.std;
            delete[] h_out;
        }

        // Stats of means across launches
        double sum = 0, min_m = 1e18, max_m = 0;
        for (double m : launch_means) {
            sum += m;
            if (m < min_m) min_m = m;
            if (m > max_m) max_m = m;
        }
        double mean_of_means = sum / LAUNCHES;
        double var = 0;
        for (double m : launch_means) var += (m - mean_of_means) * (m - mean_of_means);
        double std_of_means = sqrt(var / LAUNCHES);

        fprintf(stderr, "  %d identical launches: mean=%.1f std_across=%.2f CV=%.6f\n",
                LAUNCHES, mean_of_means, std_of_means, std_of_means / mean_of_means);
        fprintf(stderr, "  Range: [%.1f, %.1f] (%.1f cycles, %.4f%%)\n",
                min_m, max_m, max_m - min_m, 100.0 * (max_m - min_m) / mean_of_means);

        // Intra-launch std (are threads within one launch uniform?)
        double mean_intra_std = 0;
        for (double s : launch_stds) mean_intra_std += s;
        mean_intra_std /= LAUNCHES;
        fprintf(stderr, "  Mean intra-launch std: %.2f cycles\n", mean_intra_std);

        hipFree(d_out);
    }

    // ===== TEST 4: CPU↔GPU FABRIC LATENCY =====
    fprintf(stderr, "\n═══ TEST 4: CPU↔GPU Fabric Latency Jitter ═══\n");
    {
        uint32_t *d_system, *d_device;
        uint64_t *d_out;

        // System (CPU-pinned) memory
        hipHostMalloc(&d_system, LANES * 32 * sizeof(uint32_t), hipHostMallocDefault);
        memset(d_system, 0xAB, LANES * 32 * sizeof(uint32_t));

        // Device (GPU-local) memory
        hipMalloc(&d_device, LANES * 32 * sizeof(uint32_t));
        hipMemset(d_device, 0xAB, LANES * 32 * sizeof(uint32_t));

        hipMalloc(&d_out, LANES * ITERS * sizeof(uint64_t));

        // Test device memory
        hipMemset(d_out, 0, LANES * ITERS * sizeof(uint64_t));
        fabric_latency_probe<<<1, LANES>>>(d_out, d_system, d_device, ITERS, 0);
        hipDeviceSynchronize();

        uint64_t* h_dev = new uint64_t[LANES * ITERS];
        hipMemcpy(h_dev, d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);

        Stats s_dev = compute_stats(h_dev, ITERS);
        fprintf(stderr, "  Device mem: mean=%.1f std=%.2f CV=%.6f unique=%d\n",
                s_dev.mean, s_dev.std, s_dev.std/s_dev.mean, s_dev.unique);

        // Test system memory (fabric hop)
        hipMemset(d_out, 0, LANES * ITERS * sizeof(uint64_t));
        fabric_latency_probe<<<1, LANES>>>(d_out, d_system, d_device, ITERS, 1);
        hipDeviceSynchronize();

        uint64_t* h_sys = new uint64_t[LANES * ITERS];
        hipMemcpy(h_sys, d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);

        Stats s_sys = compute_stats(h_sys, ITERS);
        fprintf(stderr, "  System mem: mean=%.1f std=%.2f CV=%.6f unique=%d\n",
                s_sys.mean, s_sys.std, s_sys.std/s_sys.mean, s_sys.unique);
        fprintf(stderr, "  Fabric overhead: +%.1f cycles (+%.1f%%)\n",
                s_sys.mean - s_dev.mean, 100.0 * (s_sys.mean - s_dev.mean) / s_dev.mean);
        fprintf(stderr, "  Jitter amplification: %.2fx\n",
                s_sys.std > 0 && s_dev.std > 0 ? s_sys.std / s_dev.std : 0);

        print_distribution(h_dev, ITERS, "device_mem");
        print_distribution(h_sys, ITERS, "system_mem");

        delete[] h_dev;
        delete[] h_sys;
        hipHostFree(d_system);
        hipFree(d_device);
        hipFree(d_out);
    }

    // ===== TEST 5: MEMORY CONTROLLER BANK CONFLICTS =====
    fprintf(stderr, "\n═══ TEST 5: Memory Controller Bank Conflict Jitter ═══\n");
    {
        size_t alloc_size = 16 * 1024 * 1024;
        uint32_t *d_data;
        uint64_t *d_out;
        hipMalloc(&d_data, alloc_size);
        hipMalloc(&d_out, LANES * ITERS * sizeof(uint64_t));
        hipMemset(d_data, 0xCD, alloc_size);

        // Same bank
        hipMemset(d_out, 0, LANES * ITERS * sizeof(uint64_t));
        mc_bank_conflict<<<1, LANES>>>(d_out, d_data, ITERS, 1);
        hipDeviceSynchronize();

        uint64_t* h_same = new uint64_t[LANES * ITERS];
        hipMemcpy(h_same, d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);
        Stats s_same = compute_stats(h_same, ITERS);

        // Different banks
        hipMemset(d_out, 0, LANES * ITERS * sizeof(uint64_t));
        mc_bank_conflict<<<1, LANES>>>(d_out, d_data, ITERS, 0);
        hipDeviceSynchronize();

        uint64_t* h_diff = new uint64_t[LANES * ITERS];
        hipMemcpy(h_diff, d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);
        Stats s_diff = compute_stats(h_diff, ITERS);

        fprintf(stderr, "  Same bank:  mean=%.1f std=%.2f CV=%.6f unique=%d\n",
                s_same.mean, s_same.std, s_same.std/s_same.mean, s_same.unique);
        fprintf(stderr, "  Diff banks: mean=%.1f std=%.2f CV=%.6f unique=%d\n",
                s_diff.mean, s_diff.std, s_diff.std/s_diff.mean, s_diff.unique);
        fprintf(stderr, "  Bank conflict penalty: +%.1f cycles (+%.1f%%)\n",
                s_same.mean - s_diff.mean, 100.0 * (s_same.mean - s_diff.mean) / s_diff.mean);

        print_distribution(h_same, ITERS, "same_bank");
        print_distribution(h_diff, ITERS, "diff_banks");

        delete[] h_same;
        delete[] h_diff;
        hipFree(d_data);
        hipFree(d_out);
    }

    // ===== TEST 6: PIPELINE BUBBLE NOISE =====
    fprintf(stderr, "\n═══ TEST 6: Instruction Pipeline Bubble Noise ═══\n");
    {
        uint64_t *d_out;
        hipMalloc(&d_out, LANES * ITERS * sizeof(uint64_t));

        const char* names[] = {"VGPR_bank_conflict", "RAW_dependency", "mixed_VALU_SALU"};

        for (int p = 0; p < 3; p++) {
            hipMemset(d_out, 0, LANES * ITERS * sizeof(uint64_t));
            pipeline_bubble_probe<<<1, LANES>>>(d_out, ITERS, p);
            hipDeviceSynchronize();

            uint64_t* h_out = new uint64_t[LANES * ITERS];
            hipMemcpy(h_out, d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);

            Stats s = compute_stats(h_out, ITERS);
            fprintf(stderr, "  %s: mean=%.2f std=%.4f CV=%.6f unique=%d\n",
                    names[p], s.mean, s.std, s.std/(s.mean > 0 ? s.mean : 1), s.unique);
            print_distribution(h_out, ITERS, names[p]);

            delete[] h_out;
        }

        hipFree(d_out);
    }

    // ===== TEST 7: INTRA-WORKGROUP INTERFERENCE =====
    fprintf(stderr, "\n═══ TEST 7: Intra-Workgroup Wavefront Interference ═══\n");
    {
        uint64_t *d_out;
        // Max 8 wavefronts per workgroup (512 threads)
        hipMalloc(&d_out, 32 * ITERS * sizeof(uint64_t));

        int wave_counts[] = {1, 2, 4, 8};
        for (int w : wave_counts) {
            hipMemset(d_out, 0, 32 * ITERS * sizeof(uint64_t));
            intra_wg_interference<<<1, w * 64>>>(d_out, ITERS, w);
            hipDeviceSynchronize();

            uint64_t* h_out = new uint64_t[32 * ITERS];
            hipMemcpy(h_out, d_out, 32 * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);

            Stats s = compute_stats(h_out, ITERS);
            // Count slow reads (> min + 1)
            int slow = 0, total = 0;
            for (int i = 0; i < ITERS; i++) {
                if (h_out[i] > 0 && h_out[i] < 1e12) {
                    total++;
                    if (h_out[i] > s.min_val + 0.5) slow++;
                }
            }
            fprintf(stderr, "  %d waves: mean=%.2f std=%.4f slow=%.2f%% unique=%d\n",
                    w, s.mean, s.std, 100.0 * slow / (total > 0 ? total : 1), s.unique);
        }

        hipFree(d_out);
    }

    // ===== TEST 8: CLOCK GRANULARITY =====
    fprintf(stderr, "\n═══ TEST 8: Clock Counter Granularity ═══\n");
    {
        uint64_t *d_delta, *d_raw;
        hipMalloc(&d_delta, ITERS * sizeof(uint64_t));
        hipMalloc(&d_raw, ITERS * sizeof(uint64_t));

        // Shader clock
        clock_granularity<<<1, 1>>>(d_delta, d_raw, ITERS);
        hipDeviceSynchronize();

        uint64_t* h_delta = new uint64_t[ITERS];
        uint64_t* h_raw = new uint64_t[ITERS];
        hipMemcpy(h_delta, d_delta, ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);
        hipMemcpy(h_raw, d_raw, ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);

        Stats s = compute_stats(h_delta, ITERS);
        fprintf(stderr, "  Shader clock granularity: mean=%.2f std=%.4f min=%.0f unique=%d\n",
                s.mean, s.std, s.min_val, s.unique);
        print_distribution(h_delta, ITERS, "shader_delta");

        // Check monotonicity
        int non_mono = 0;
        for (int i = 1; i < ITERS; i++) {
            if (h_raw[i] <= h_raw[i-1]) non_mono++;
        }
        fprintf(stderr, "  Non-monotonic: %d/%d (%.4f%%)\n",
                non_mono, ITERS, 100.0 * non_mono / ITERS);

        // Wall clock
        wall_clock_granularity<<<1, 1>>>(d_delta, d_raw, ITERS);
        hipDeviceSynchronize();

        hipMemcpy(h_delta, d_delta, ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);
        hipMemcpy(h_raw, d_raw, ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);

        Stats s2 = compute_stats(h_delta, ITERS);
        fprintf(stderr, "  Wall clock granularity: mean=%.2f std=%.4f min=%.0f unique=%d\n",
                s2.mean, s2.std, s2.min_val, s2.unique);
        print_distribution(h_delta, ITERS, "wall_delta");

        non_mono = 0;
        for (int i = 1; i < ITERS; i++) {
            if (h_raw[i] <= h_raw[i-1]) non_mono++;
        }
        fprintf(stderr, "  Wall non-monotonic: %d/%d (%.4f%%)\n",
                non_mono, ITERS, 100.0 * non_mono / ITERS);

        delete[] h_delta;
        delete[] h_raw;
        hipFree(d_delta);
        hipFree(d_raw);
    }

    fprintf(stderr, "\nDone.\n");
    return 0;
}
