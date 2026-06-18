// jitter_deep_silicon.hip — Push DEEPER into GPU silicon for stronger noise
//
// New probes beyond wall_clock64 and placement:
//   1. ATOMIC CONTENTION: atomicAdd on shared address → hardware arbitration order
//   2. LDS BANK CONFLICT timing: intentional bank conflicts → pipeline stall jitter
//   3. WARP BALLOT divergence: __ballot() captures per-lane PLL phase differences
//   4. MULTI-WAVEFRONT scheduling: multiple wavefronts on same CU → scheduler noise
//   5. GLOBAL ATOMIC ordering: cross-CU atomic → fabric arbitration noise
//   6. COMBINED: all sources mixed for maximum reservoir dimensionality
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_deep_silicon scripts/jitter_deep_silicon.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_deep_silicon

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

#define ITERS 10000

// ============================================================================
// PROBE A: Atomic contention — two threads atomicAdd same address
// Hardware must serialize → order creates timing jitter
// ============================================================================
__global__ void probe_atomic_contention(uint64_t* timings, uint32_t* shared_counter) {
    int tid = threadIdx.x;
    if (tid >= 64) return;

    for (int i = 0; i < ITERS; i++) {
        uint64_t t0 = __clock64();
        atomicAdd(shared_counter, 1u);  // ALL 64 threads hit same address
        uint64_t t1 = __clock64();
        timings[tid * ITERS + i] = t1 - t0;
    }
}

// ============================================================================
// PROBE B: LDS bank conflict timing
// All threads access same bank → serialized access → timing varies
// ============================================================================
__global__ void probe_lds_bank_conflict(uint64_t* timings) {
    int tid = threadIdx.x;
    if (tid >= 64) return;

    __shared__ uint32_t lds[1024];
    lds[tid] = tid;
    __syncthreads();

    for (int i = 0; i < ITERS; i++) {
        // All threads access address 0 (same bank) → bank conflict
        uint64_t t0 = __clock64();
        uint32_t val = lds[0];  // bank 0 conflict
        uint64_t t1 = __clock64();
        timings[tid * ITERS + i] = t1 - t0;
        lds[tid] = val + i;  // prevent optimization
        __syncthreads();
    }
}

// ============================================================================
// PROBE C: LDS bank conflict vs no conflict
// Compare: all-same-bank vs each-own-bank timing distribution
// ============================================================================
__global__ void probe_lds_conflict_vs_free(uint64_t* timings_conflict,
                                            uint64_t* timings_free) {
    int tid = threadIdx.x;
    if (tid >= 64) return;

    __shared__ uint32_t lds[1024];
    lds[tid] = tid;
    __syncthreads();

    // Conflict: all threads read same bank
    for (int i = 0; i < ITERS; i++) {
        uint64_t t0 = __clock64();
        uint32_t val = lds[0];  // bank 0
        uint64_t t1 = __clock64();
        timings_conflict[tid * ITERS + i] = t1 - t0;
        lds[tid] = val + i;
        __syncthreads();
    }

    __syncthreads();

    // Free: each thread reads own bank
    for (int i = 0; i < ITERS; i++) {
        uint64_t t0 = __clock64();
        uint32_t val = lds[tid];  // no conflict
        uint64_t t1 = __clock64();
        timings_free[tid * ITERS + i] = t1 - t0;
        lds[tid] = val + i;
        __syncthreads();
    }
}

// ============================================================================
// PROBE D: Multi-wavefront on same CU — scheduler arbitration noise
// Launch 4 wavefronts (256 threads) on 1 block → wavefront scheduling creates jitter
// ============================================================================
__global__ void probe_multiwavefront(uint64_t* timings) {
    int tid = threadIdx.x;  // 0-255 across 4 wavefronts
    int wf = tid / 64;      // wavefront 0-3
    int lane = tid % 64;

    float x = (float)tid * 0.001f + 1.0f;
    float y = x + 0.1f;

    for (int i = 0; i < ITERS; i++) {
        uint64_t t0 = __clock64();
        // Short FMA burst — timing depends on wavefront scheduling order
        x = __fmaf_rn(x, y, x);
        y = __fmaf_rn(y, x, y);
        x = __fmaf_rn(x, 1.0001f, -0.0001f);
        uint64_t t1 = __clock64();

        timings[tid * ITERS + i] = t1 - t0;
    }
    if (x + y == -999.999f) timings[0] = __float_as_uint(x);
}

// ============================================================================
// PROBE E: Global atomic ordering — cross-CU fabric arbitration
// Multiple blocks atomicAdd same global address → Data Fabric arbitration
// ============================================================================
__global__ void probe_global_atomic(uint64_t* timings, uint32_t* global_counter) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (tid != 0) return;  // 1 thread per block

    for (int i = 0; i < ITERS; i++) {
        uint64_t t0 = __clock64();
        atomicAdd(global_counter, 1u);
        uint64_t t1 = __clock64();
        timings[bid * ITERS + i] = t1 - t0;
    }
}

// ============================================================================
// PROBE F: wall_clock64 at different accumulation depths
// More reads = stronger noise. How does the distribution change?
// ============================================================================
__global__ void probe_wallclock_depth(uint64_t* timings, int n_reads) {
    int tid = threadIdx.x;
    if (tid >= 64) return;

    for (int i = 0; i < ITERS; i++) {
        uint64_t sum = 0;
        for (int r = 0; r < n_reads; r++) {
            uint64_t w0 = wall_clock64();
            uint64_t w1 = wall_clock64();
            sum += (w1 - w0);
        }
        timings[tid * ITERS + i] = sum;
    }
}

// ============================================================================
// PROBE G: Per-lane wall_clock divergence within a warp
// Do different lanes see different PLL phase?
// ============================================================================
__global__ void probe_lane_divergence(uint64_t* per_lane_timings) {
    int tid = threadIdx.x;
    if (tid >= 64) return;

    for (int i = 0; i < ITERS; i++) {
        uint64_t w0 = wall_clock64();
        uint64_t w1 = wall_clock64();
        per_lane_timings[tid * ITERS + i] = w1 - w0;
    }
}

// ============================================================================
void analyze(uint64_t* data, int n, const char* name) {
    int counts[256] = {};
    int valid = 0;
    uint64_t mn = UINT64_MAX, mx = 0;
    double sum = 0, sum2 = 0;

    for (int i = 0; i < n; i++) {
        uint64_t v = data[i];
        if (v < 100000) {
            valid++;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
            sum2 += (double)v * v;
            if (v < 256) counts[v]++;
        }
    }

    double mean = valid > 0 ? sum / valid : 0;
    double var = valid > 0 ? sum2 / valid - mean * mean : 0;
    double std = sqrt(var > 0 ? var : 0);

    fprintf(stderr, "  %s: valid=%d range=[%lu,%lu] mean=%.2f std=%.3f\n",
            name, valid, mn, mx, mean, std);
    fprintf(stderr, "    distribution: ");
    for (uint64_t v = mn; v <= mx && v < mn + 20; v++) {
        if (v < 256 && counts[v] > 0)
            fprintf(stderr, "%lu:%.1f%% ", v, 100.0*counts[v]/valid);
    }
    fprintf(stderr, "\n");

    int unique = 0;
    for (int i = 0; i < 256; i++) if (counts[i] > 0) unique++;
    fprintf(stderr, "    unique values (0-255): %d\n", unique);
}

void compare_runs(uint64_t* r1, uint64_t* r2, int n, const char* name) {
    int same = 0;
    for (int i = 0; i < n; i++) if (r1[i] == r2[i]) same++;
    fprintf(stderr, "  %s run-to-run identical: %d/%d (%.1f%%)\n",
            name, same, n, 100.0*same/n);
}

int main() {
    fprintf(stderr, "═══ DEEP SILICON PROBES ═══\n\n");

    uint64_t *d_timings, *d_timings2, *d_timings3;
    uint32_t *d_counter;
    hipMalloc(&d_timings, 256 * ITERS * sizeof(uint64_t));
    hipMalloc(&d_timings2, 64 * ITERS * sizeof(uint64_t));
    hipMalloc(&d_timings3, 64 * ITERS * sizeof(uint64_t));
    hipMalloc(&d_counter, sizeof(uint32_t));

    uint64_t* h1 = new uint64_t[256 * ITERS];
    uint64_t* h2 = new uint64_t[256 * ITERS];

    // ═══ PROBE A: Atomic contention ═══
    fprintf(stderr, "═══ A: Atomic contention (64 threads → 1 address) ═══\n");
    hipMemset(d_counter, 0, sizeof(uint32_t));
    probe_atomic_contention<<<1, 64>>>(d_timings, d_counter);
    hipDeviceSynchronize();
    hipMemcpy(h1, d_timings, 64*ITERS*sizeof(uint64_t), hipMemcpyDeviceToHost);
    analyze(h1, ITERS, "lane0");
    analyze(h1 + 32*ITERS, ITERS, "lane32");

    // Run 2
    hipMemset(d_counter, 0, sizeof(uint32_t));
    probe_atomic_contention<<<1, 64>>>(d_timings, d_counter);
    hipDeviceSynchronize();
    hipMemcpy(h2, d_timings, 64*ITERS*sizeof(uint64_t), hipMemcpyDeviceToHost);
    compare_runs(h1, h2, ITERS, "lane0");
    compare_runs(h1+32*ITERS, h2+32*ITERS, ITERS, "lane32");

    // ═══ PROBE B: LDS bank conflict ═══
    fprintf(stderr, "\n═══ B: LDS bank conflict (all read bank 0) ═══\n");
    probe_lds_bank_conflict<<<1, 64>>>(d_timings);
    hipDeviceSynchronize();
    hipMemcpy(h1, d_timings, 64*ITERS*sizeof(uint64_t), hipMemcpyDeviceToHost);
    analyze(h1, ITERS, "lane0_conflict");
    analyze(h1+32*ITERS, ITERS, "lane32_conflict");

    probe_lds_bank_conflict<<<1, 64>>>(d_timings);
    hipDeviceSynchronize();
    hipMemcpy(h2, d_timings, 64*ITERS*sizeof(uint64_t), hipMemcpyDeviceToHost);
    compare_runs(h1, h2, ITERS, "lane0");

    // ═══ PROBE C: Conflict vs free ═══
    fprintf(stderr, "\n═══ C: LDS conflict vs free ═══\n");
    probe_lds_conflict_vs_free<<<1, 64>>>(d_timings, d_timings2);
    hipDeviceSynchronize();
    hipMemcpy(h1, d_timings, 64*ITERS*sizeof(uint64_t), hipMemcpyDeviceToHost);
    hipMemcpy(h2, d_timings2, 64*ITERS*sizeof(uint64_t), hipMemcpyDeviceToHost);
    analyze(h1, ITERS, "conflict_lane0");
    analyze(h2, ITERS, "free_lane0");

    // ═══ PROBE D: Multi-wavefront scheduling ═══
    fprintf(stderr, "\n═══ D: Multi-wavefront (4 WFs on 1 CU) ═══\n");
    probe_multiwavefront<<<1, 256>>>(d_timings);
    hipDeviceSynchronize();
    hipMemcpy(h1, d_timings, 256*ITERS*sizeof(uint64_t), hipMemcpyDeviceToHost);
    analyze(h1, ITERS, "WF0_lane0");
    analyze(h1+64*ITERS, ITERS, "WF1_lane0");
    analyze(h1+128*ITERS, ITERS, "WF2_lane0");
    analyze(h1+192*ITERS, ITERS, "WF3_lane0");

    // Run 2
    probe_multiwavefront<<<1, 256>>>(d_timings);
    hipDeviceSynchronize();
    hipMemcpy(h2, d_timings, 256*ITERS*sizeof(uint64_t), hipMemcpyDeviceToHost);
    compare_runs(h1, h2, ITERS, "WF0_lane0");
    compare_runs(h1+64*ITERS, h2+64*ITERS, ITERS, "WF1_lane0");
    compare_runs(h1+128*ITERS, h2+128*ITERS, ITERS, "WF2_lane0");
    compare_runs(h1+192*ITERS, h2+192*ITERS, ITERS, "WF3_lane0");

    // Cross-WF comparison (same run)
    int cross_same = 0;
    for (int i = 0; i < ITERS; i++)
        if (h1[i] == h1[64*ITERS+i]) cross_same++;
    fprintf(stderr, "  WF0 vs WF1 same-run identical: %d/%d (%.1f%%)\n",
            cross_same, ITERS, 100.0*cross_same/ITERS);

    // ═══ PROBE E: Global atomic ordering ═══
    fprintf(stderr, "\n═══ E: Global atomic ordering (12 blocks) ═══\n");
    hipMemset(d_counter, 0, sizeof(uint32_t));
    probe_global_atomic<<<12, 1>>>(d_timings, d_counter);
    hipDeviceSynchronize();
    hipMemcpy(h1, d_timings, 12*ITERS*sizeof(uint64_t), hipMemcpyDeviceToHost);
    analyze(h1, ITERS, "block0");
    analyze(h1+6*ITERS, ITERS, "block6");

    hipMemset(d_counter, 0, sizeof(uint32_t));
    probe_global_atomic<<<12, 1>>>(d_timings, d_counter);
    hipDeviceSynchronize();
    hipMemcpy(h2, d_timings, 12*ITERS*sizeof(uint64_t), hipMemcpyDeviceToHost);
    compare_runs(h1, h2, ITERS, "block0");
    compare_runs(h1+6*ITERS, h2+6*ITERS, ITERS, "block6");

    // ═══ PROBE F: wall_clock depth sweep ═══
    fprintf(stderr, "\n═══ F: wall_clock accumulation depth ═══\n");
    for (int depth : {1, 4, 16, 64, 256}) {
        probe_wallclock_depth<<<1, 64>>>(d_timings, depth);
        hipDeviceSynchronize();
        hipMemcpy(h1, d_timings, 64*ITERS*sizeof(uint64_t), hipMemcpyDeviceToHost);

        probe_wallclock_depth<<<1, 64>>>(d_timings, depth);
        hipDeviceSynchronize();
        hipMemcpy(h2, d_timings, 64*ITERS*sizeof(uint64_t), hipMemcpyDeviceToHost);

        char name[64];
        snprintf(name, 64, "depth=%d", depth);
        analyze(h1, ITERS, name);
        compare_runs(h1, h2, ITERS, name);
    }

    // ═══ PROBE G: Per-lane divergence ═══
    fprintf(stderr, "\n═══ G: Per-lane wall_clock divergence ═══\n");
    probe_lane_divergence<<<1, 64>>>(d_timings);
    hipDeviceSynchronize();
    hipMemcpy(h1, d_timings, 64*ITERS*sizeof(uint64_t), hipMemcpyDeviceToHost);

    // Compare lanes within same run
    int lane_pairs_diff = 0, lane_pairs_total = 0;
    for (int i = 0; i < ITERS; i++) {
        for (int l1 = 0; l1 < 64; l1++) {
            for (int l2 = l1+1; l2 < 64 && l2 < l1+4; l2++) {
                if (h1[l1*ITERS+i] != h1[l2*ITERS+i]) lane_pairs_diff++;
                lane_pairs_total++;
            }
        }
    }
    fprintf(stderr, "  Inter-lane differences: %d/%d (%.1f%%)\n",
            lane_pairs_diff, lane_pairs_total, 100.0*lane_pairs_diff/lane_pairs_total);

    // Per-lane statistics
    for (int lane : {0, 16, 32, 48, 63}) {
        char name[32];
        snprintf(name, 32, "lane%d", lane);
        analyze(h1+lane*ITERS, ITERS, name);
    }

    // Run 2 — per-lane run-to-run
    probe_lane_divergence<<<1, 64>>>(d_timings);
    hipDeviceSynchronize();
    hipMemcpy(h2, d_timings, 64*ITERS*sizeof(uint64_t), hipMemcpyDeviceToHost);

    int per_lane_same[64];
    for (int l = 0; l < 64; l++) {
        per_lane_same[l] = 0;
        for (int i = 0; i < ITERS; i++)
            if (h1[l*ITERS+i] == h2[l*ITERS+i]) per_lane_same[l]++;
    }
    fprintf(stderr, "  Per-lane run-to-run identity (sample): ");
    for (int l : {0, 16, 32, 48, 63})
        fprintf(stderr, "L%d:%.1f%% ", l, 100.0*per_lane_same[l]/ITERS);
    fprintf(stderr, "\n");

    delete[] h1;
    delete[] h2;
    hipFree(d_timings);
    hipFree(d_timings2);
    hipFree(d_timings3);
    hipFree(d_counter);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
