// jitter_noise_verify.hip — Verify noise exists during concurrent execution
// Diagnose WHY the reservoir sees r=1.000 (zero noise)
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_noise_verify scripts/jitter_noise_verify.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_noise_verify

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define ITERS 20000
#define LANES 64

// TEST A: Single block — L1 read timing raw values
__global__ void diag_l1_raw(uint64_t* out, uint32_t* data, int iters) {
    int tid = threadIdx.x;
    if (tid >= LANES) return;

    for (int i = 0; i < iters; i++) {
        uint32_t addr = (tid * 32 + (i & 0x3F)) & (32768/4 - 1);
        uint64_t t0 = __clock64();
        uint32_t val = data[addr];
        uint64_t t1 = __clock64();
        out[tid * iters + i] = t1 - t0;
        if (val == 0xDEADDEAD) out[0] = 0;
    }
}

// TEST B: Multi-block with concurrent thermal — reservoir block reads L1 timing
__global__ void diag_concurrent(uint64_t* out, uint32_t* data, int iters) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    if (bid == 0) {
        // Measurement block: L1 reads
        if (tid >= LANES) return;
        for (int i = 0; i < iters; i++) {
            uint32_t addr = (tid * 32 + (i & 0x3F)) & (32768/4 - 1);
            uint64_t t0 = __clock64();
            uint32_t val = data[addr];
            uint64_t t1 = __clock64();
            out[tid * iters + i] = t1 - t0;
            if (val == 0xDEADDEAD) out[0] = 0;
        }
    } else {
        // Thermal load blocks: heavy FMA
        float x = (float)(tid + bid*64) * 0.001f;
        float y = x + 0.1f;
        for (int i = 0; i < iters * 100; i++) {
            x = __fmaf_rn(x, y, x);
            y = __fmaf_rn(y, x, y);
            x = __fmaf_rn(x, 1.0001f, -0.0001f);
            y = __fmaf_rn(y, 0.9999f, 0.0001f);
        }
        if (x + y == -999.999f) out[0] = __float_as_uint(x);
    }
}

// TEST C: wall_clock64 raw values — single block vs concurrent
__global__ void diag_wallclock(uint64_t* out, int iters) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (bid == 0) {
        if (tid >= LANES) return;
        for (int i = 0; i < iters; i++) {
            uint64_t w0 = wall_clock64();
            uint64_t w1 = wall_clock64();
            out[tid * iters + i] = w1 - w0;
        }
    } else {
        float x = (float)(tid + bid*64) * 0.001f;
        float y = x + 0.1f;
        for (int i = 0; i < iters * 100; i++) {
            x = __fmaf_rn(x, y, x); y = __fmaf_rn(y, x, y);
        }
        if (x + y == -999.999f) out[0] = __float_as_uint(x);
    }
}

// TEST D: Memory contention — does stride ACTUALLY affect timing?
__global__ void diag_contention(uint64_t* out, uint32_t* data, int iters, int stride) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (bid == 0) {
        if (tid >= LANES) return;
        uint32_t sum = 0;
        for (int i = 0; i < iters; i++) {
            uint32_t addr = (tid * stride * 64 + i * 997) & ((16*1024*1024/4)-1);
            uint64_t t0 = __clock64();
            sum += data[addr];
            uint64_t t1 = __clock64();
            out[tid * iters + i] = t1 - t0;
        }
        if (sum == 0xDEADDEAD) out[0] = sum;
    } else {
        // Other blocks also thrash memory
        uint32_t sum = 0;
        for (int i = 0; i < iters * 20; i++) {
            uint32_t addr = ((tid + bid*64) * 128 + i * 8191) & ((16*1024*1024/4)-1);
            sum += data[addr];
        }
        if (sum == 0xDEADDEAD) data[0] = sum;
    }
}

// TEST E: Cross-PLL drift raw
__global__ void diag_pll_drift(uint64_t* out_diff, int iters) {
    int tid = threadIdx.x;
    if (tid >= 1) return;  // single thread

    for (int i = 0; i < iters; i++) {
        uint64_t s = __clock64();
        uint64_t w = wall_clock64();
        out_diff[i] = s - w;
    }
}

void analyze(uint64_t* data, int n, const char* name) {
    // Count values
    int counts[64] = {};
    int valid = 0;
    uint64_t mn = UINT64_MAX, mx = 0;

    for (int i = 0; i < n; i++) {
        uint64_t v = data[i];
        if (v < 10000) {
            valid++;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            if (v < 64) counts[v]++;
        }
    }

    fprintf(stderr, "  %s: valid=%d range=[%lu,%lu]\n", name, valid, mn, mx);
    fprintf(stderr, "    distribution: ");
    for (uint64_t v = mn; v <= mx && v < mn + 15; v++) {
        if (counts[v] > 0) fprintf(stderr, "%lu:%.1f%% ", v, 100.0*counts[v]/valid);
    }
    fprintf(stderr, "\n");

    // Check if all same
    int unique = 0;
    for (int i = 0; i < 64; i++) if (counts[i] > 0) unique++;
    fprintf(stderr, "    unique values (0-63): %d\n", unique);

    // Run-to-run comparison
    int diff_from_first = 0;
    for (int i = 1; i < n && i < 20000; i++) {
        if (data[i] != data[0]) diff_from_first++;
    }
    fprintf(stderr, "    values ≠ first: %d/%d (%.2f%%)\n",
            diff_from_first, n<20000?n:20000, 100.0*diff_from_first/(n<20000?n:20000));
}

int main() {
    fprintf(stderr, "═══ NOISE DIAGNOSTICS ═══\n\n");

    size_t alloc = 16 * 1024 * 1024;
    uint32_t *d_data;
    uint64_t *d_out;
    hipMalloc(&d_data, alloc);
    hipMalloc(&d_out, LANES * ITERS * sizeof(uint64_t));
    hipMemset(d_data, 0xAB, alloc);

    uint64_t* h_out = new uint64_t[LANES * ITERS];

    // ═══ TEST A: L1 solo (no load) ═══
    fprintf(stderr, "═══ A: L1 timing — SOLO (no concurrent load) ═══\n");
    hipMemset(d_out, 0, LANES * ITERS * sizeof(uint64_t));
    diag_l1_raw<<<1, LANES>>>(d_out, d_data, ITERS);
    hipDeviceSynchronize();
    hipMemcpy(h_out, d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);
    analyze(h_out, ITERS, "L1_solo_lane0");

    // Run again — compare
    hipMemset(d_out, 0, LANES * ITERS * sizeof(uint64_t));
    diag_l1_raw<<<1, LANES>>>(d_out, d_data, ITERS);
    hipDeviceSynchronize();
    uint64_t* h_out2 = new uint64_t[LANES * ITERS];
    hipMemcpy(h_out2, d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);

    int same = 0;
    for (int i = 0; i < ITERS; i++) if (h_out[i] == h_out2[i]) same++;
    fprintf(stderr, "  Run-to-run identical: %d/%d (%.1f%%)\n\n", same, ITERS, 100.0*same/ITERS);

    // ═══ TEST B: L1 with concurrent thermal (7 blocks heat) ═══
    fprintf(stderr, "═══ B: L1 timing — WITH 7 thermal blocks ═══\n");
    hipMemset(d_out, 0, LANES * ITERS * sizeof(uint64_t));
    diag_concurrent<<<8, LANES>>>(d_out, d_data, ITERS);
    hipDeviceSynchronize();
    hipMemcpy(h_out, d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);
    analyze(h_out, ITERS, "L1_hot_lane0");

    // Second run
    hipMemset(d_out, 0, LANES * ITERS * sizeof(uint64_t));
    diag_concurrent<<<8, LANES>>>(d_out, d_data, ITERS);
    hipDeviceSynchronize();
    hipMemcpy(h_out2, d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);
    same = 0;
    for (int i = 0; i < ITERS; i++) if (h_out[i] == h_out2[i]) same++;
    fprintf(stderr, "  Run-to-run identical: %d/%d (%.1f%%)\n\n", same, ITERS, 100.0*same/ITERS);

    // ═══ TEST C: wall_clock64 solo vs hot ═══
    fprintf(stderr, "═══ C: wall_clock64 — SOLO ═══\n");
    hipMemset(d_out, 0, LANES * ITERS * sizeof(uint64_t));
    diag_wallclock<<<1, LANES>>>(d_out, ITERS);
    hipDeviceSynchronize();
    hipMemcpy(h_out, d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);
    analyze(h_out, ITERS, "wall_solo");

    fprintf(stderr, "\n═══ C2: wall_clock64 — WITH 7 thermal blocks ═══\n");
    hipMemset(d_out, 0, LANES * ITERS * sizeof(uint64_t));
    diag_wallclock<<<8, LANES>>>(d_out, ITERS);
    hipDeviceSynchronize();
    hipMemcpy(h_out, d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);
    analyze(h_out, ITERS, "wall_hot");

    // Second run
    hipMemset(d_out, 0, LANES * ITERS * sizeof(uint64_t));
    diag_wallclock<<<8, LANES>>>(d_out, ITERS);
    hipDeviceSynchronize();
    hipMemcpy(h_out2, d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);
    same = 0;
    for (int i = 0; i < ITERS; i++) if (h_out[i] == h_out2[i]) same++;
    fprintf(stderr, "  Run-to-run identical: %d/%d (%.1f%%)\n", same, ITERS, 100.0*same/ITERS);

    // ═══ TEST D: Contention — stride 4 vs 128 ═══
    fprintf(stderr, "\n═══ D: Memory contention — stride 4 (cached) ═══\n");
    hipMemset(d_out, 0, LANES * ITERS * sizeof(uint64_t));
    diag_contention<<<8, LANES>>>(d_out, d_data, ITERS, 4);
    hipDeviceSynchronize();
    hipMemcpy(h_out, d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);
    analyze(h_out, ITERS, "stride_4");

    fprintf(stderr, "\n═══ D2: Memory contention — stride 128 (thrash) ═══\n");
    hipMemset(d_out, 0, LANES * ITERS * sizeof(uint64_t));
    diag_contention<<<8, LANES>>>(d_out, d_data, ITERS, 128);
    hipDeviceSynchronize();
    hipMemcpy(h_out, d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);
    analyze(h_out, ITERS, "stride_128");

    // Second run stride 128
    hipMemset(d_out, 0, LANES * ITERS * sizeof(uint64_t));
    diag_contention<<<8, LANES>>>(d_out, d_data, ITERS, 128);
    hipDeviceSynchronize();
    hipMemcpy(h_out2, d_out, LANES * ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);
    same = 0;
    for (int i = 0; i < ITERS; i++) if (h_out[i] == h_out2[i]) same++;
    fprintf(stderr, "  Run-to-run identical: %d/%d (%.1f%%)\n", same, ITERS, 100.0*same/ITERS);

    // ═══ TEST E: Cross-PLL drift ═══
    fprintf(stderr, "\n═══ E: Cross-PLL drift (shader - wall) ═══\n");
    hipMemset(d_out, 0, ITERS * sizeof(uint64_t));
    diag_pll_drift<<<1, 1>>>(d_out, ITERS);
    hipDeviceSynchronize();
    hipMemcpy(h_out, d_out, ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);

    // Show first 30 values
    fprintf(stderr, "  First 30 drift values: ");
    for (int i = 0; i < 30; i++) fprintf(stderr, "%lu ", h_out[i]);
    fprintf(stderr, "\n");

    // Drift statistics
    int drift_changes = 0;
    for (int i = 1; i < ITERS; i++)
        if (h_out[i] != h_out[i-1]) drift_changes++;
    fprintf(stderr, "  Drift changes: %d/%d (%.1f%%)\n", drift_changes, ITERS, 100.0*drift_changes/ITERS);

    // Second run
    hipMemset(d_out, 0, ITERS * sizeof(uint64_t));
    diag_pll_drift<<<1, 1>>>(d_out, ITERS);
    hipDeviceSynchronize();
    hipMemcpy(h_out2, d_out, ITERS * sizeof(uint64_t), hipMemcpyDeviceToHost);
    same = 0;
    for (int i = 0; i < ITERS; i++) if (h_out[i] == h_out2[i]) same++;
    fprintf(stderr, "  Run-to-run identical: %d/%d (%.1f%%)\n", same, ITERS, 100.0*same/ITERS);

    delete[] h_out;
    delete[] h_out2;
    hipFree(d_data);
    hipFree(d_out);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
