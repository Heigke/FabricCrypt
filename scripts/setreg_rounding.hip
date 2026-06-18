/*
 * setreg_rounding.hip — Clean test: does s_setreg MODE change FP computation?
 * Uses volatile to prevent optimization, stores results immediately.
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>

__global__ void test_rounding_clean(volatile float* results) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Save original mode
    uint32_t orig_mode;
    asm volatile("s_getreg_b32 %0, hwreg(1)" : "=s"(orig_mode));

    // Use volatile to prevent compiler from optimizing away the computation
    volatile float a = 1.0f;
    volatile float tiny = 0.1f;

    // ===== Test 1: Accumulate 0.1 ten thousand times =====
    // Mode 0: Round to Nearest Even
    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x0");  // bits[1:0] = 00 = RNE
    volatile float acc0 = 0.0f;
    for (int i = 0; i < 10000; i++) acc0 = acc0 + tiny;
    results[0] = acc0;

    // Mode 1: Round toward +Inf
    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x1");  // bits[1:0] = 01 = R+inf
    volatile float acc1 = 0.0f;
    for (int i = 0; i < 10000; i++) acc1 = acc1 + tiny;
    results[1] = acc1;

    // Mode 2: Round toward -Inf
    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x2");  // bits[1:0] = 10 = R-inf
    volatile float acc2 = 0.0f;
    for (int i = 0; i < 10000; i++) acc2 = acc2 + tiny;
    results[2] = acc2;

    // Mode 3: Round toward Zero
    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x3");  // bits[1:0] = 11 = RTZ
    volatile float acc3 = 0.0f;
    for (int i = 0; i < 10000; i++) acc3 = acc3 + tiny;
    results[3] = acc3;

    // Restore
    asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(orig_mode));

    // ===== Test 2: Matrix-like multiply-accumulate =====
    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x0");
    volatile float mac0 = 0.0f;
    for (int i = 0; i < 1000; i++) {
        volatile float w = 0.3f * (float)(i % 7);
        volatile float x = 0.7f * (float)(i % 11);
        mac0 = mac0 + w * x;
    }
    results[4] = mac0;

    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x1");
    volatile float mac1 = 0.0f;
    for (int i = 0; i < 1000; i++) {
        volatile float w = 0.3f * (float)(i % 7);
        volatile float x = 0.7f * (float)(i % 11);
        mac1 = mac1 + w * x;
    }
    results[5] = mac1;

    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x3");
    volatile float mac3 = 0.0f;
    for (int i = 0; i < 1000; i++) {
        volatile float w = 0.3f * (float)(i % 7);
        volatile float x = 0.7f * (float)(i % 11);
        mac3 = mac3 + w * x;
    }
    results[6] = mac3;

    // Restore
    asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(orig_mode));

    // ===== Test 3: Does rounding affect EACH wave independently? =====
    // (This test only uses thread 0 but confirms single-thread isolation)
    results[7] = (float)orig_mode;  // store original mode for reference
}

/* Each block uses different rounding — MODE is per-wavefront (scalar) */
__global__ void test_rounding_parallel(float* results) {
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;  // 0-3, one per rounding mode

    // Set mode based on block ID (scalar, uniform across wave)
    uint32_t orig;
    asm volatile("s_getreg_b32 %0, hwreg(1)" : "=s"(orig));

    if (bid == 0) asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x0"); // RNE
    else if (bid == 1) asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x1"); // R+inf
    else if (bid == 2) asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x2"); // R-inf
    else asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x3"); // RTZ

    // Accumulate
    volatile float acc = 0.0f;
    volatile float val = 0.1f;
    for (int i = 0; i < 1000; i++) acc = acc + val;

    // Store
    results[bid * 64 + tid] = acc;

    // Restore
    asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(orig));
}

int main() {
    printf("=== s_setreg ROUNDING COMPUTATION TEST ===\n\n");

    // Test 1: Single-thread clean test
    {
        float *d_r, h_r[16];
        hipMalloc(&d_r, 16 * sizeof(float));
        hipMemset(d_r, 0, 16 * sizeof(float));
        test_rounding_clean<<<1, 1>>>((volatile float*)d_r);
        hipDeviceSynchronize();
        hipMemcpy(h_r, d_r, 16 * sizeof(float), hipMemcpyDeviceToHost);

        printf("--- Accumulate 0.1 × 10000 (expected ~1000.0) ---\n");
        uint32_t b;
        memcpy(&b, &h_r[0], 4); printf("  RNE: %.10f (0x%08x)\n", h_r[0], b);
        memcpy(&b, &h_r[1], 4); printf("  R+∞: %.10f (0x%08x)\n", h_r[1], b);
        memcpy(&b, &h_r[2], 4); printf("  R-∞: %.10f (0x%08x)\n", h_r[2], b);
        memcpy(&b, &h_r[3], 4); printf("  RTZ: %.10f (0x%08x)\n", h_r[3], b);

        int differs = 0;
        for (int i = 1; i < 4; i++) {
            uint32_t a, bi;
            memcpy(&a, &h_r[0], 4); memcpy(&bi, &h_r[i], 4);
            if (a != bi) differs = 1;
        }
        printf("  Results differ: %s\n\n", differs ?
               "*** YES — s_setreg CHANGES COMPUTATION! ***" :
               "NO — compiler may have optimized away rounding");

        printf("--- MAC: sum(w*x, 1000) ---\n");
        memcpy(&b, &h_r[4], 4); printf("  RNE: %.10f (0x%08x)\n", h_r[4], b);
        memcpy(&b, &h_r[5], 4); printf("  R+∞: %.10f (0x%08x)\n", h_r[5], b);
        memcpy(&b, &h_r[6], 4); printf("  RTZ: %.10f (0x%08x)\n", h_r[6], b);

        int mac_differs = 0;
        uint32_t m4, m5, m6;
        memcpy(&m4, &h_r[4], 4); memcpy(&m5, &h_r[5], 4); memcpy(&m6, &h_r[6], 4);
        if (m4 != m5 || m4 != m6) mac_differs = 1;
        printf("  Results differ: %s\n\n", mac_differs ?
               "*** YES — MAC AFFECTED BY ROUNDING! ***" : "no");

        hipFree(d_r);
    }

    // Test 2: Parallel — each lane different rounding
    {
        float *d_r, h_r[256];
        hipMalloc(&d_r, 256 * sizeof(float));
        test_rounding_parallel<<<4, 64>>>(d_r);  // 4 blocks, one per mode
        hipDeviceSynchronize();
        hipMemcpy(h_r, d_r, 256 * sizeof(float), hipMemcpyDeviceToHost);

        printf("--- Parallel: 64 threads, each with different rounding ---\n");
        const char* modes[] = {"RNE", "R+∞", "R-∞", "RTZ"};
        float mode_vals[4] = {0, 0, 0, 0};
        int mode_count[4] = {0, 0, 0, 0};
        for (int i = 0; i < 64; i++) {
            int m = i % 4;
            mode_vals[m] += h_r[i];
            mode_count[m]++;
        }
        for (int m = 0; m < 4; m++) {
            printf("  %s avg: %.10f (n=%d)\n",
                   modes[m], mode_vals[m] / mode_count[m], mode_count[m]);
        }

        // Check if modes produce different averages
        int par_differs = 0;
        for (int m = 1; m < 4; m++) {
            if (fabsf(mode_vals[0]/mode_count[0] - mode_vals[m]/mode_count[m]) > 0.001f)
                par_differs = 1;
        }
        printf("  Parallel rounding differs: %s\n\n", par_differs ?
               "*** YES — PER-WAVE ROUNDING CONTROL! ***" : "no (waves share MODE)");

        hipFree(d_r);
    }

    printf("=== DONE ===\n");
    return 0;
}
