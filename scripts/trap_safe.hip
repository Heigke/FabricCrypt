/*
 * trap_safe.hip — Safe trap investigation
 * Only tests operations that DON'T cause GPU faults.
 * Separated into individual kernels run sequentially.
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>

/* Test A: TRAPSTS bits 0-1 are writable. Can we use them as state? */
__global__ void test_trapsts_as_state(uint32_t* results) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int idx = 0;

    // TRAPSTS bits 0-1 are writable without fault
    // Can we use them as 2-bit state register?

    uint32_t orig;
    asm volatile("s_getreg_b32 %0, hwreg(3)" : "=s"(orig));
    results[idx++] = orig;  // original

    // Write 4 different states (00, 01, 10, 11) and verify
    for (uint32_t state = 0; state < 4; state++) {
        uint32_t val = (orig & ~0x3u) | state;
        asm volatile("s_setreg_b32 hwreg(3), %0" : : "s"(val));
        uint32_t readback;
        asm volatile("s_getreg_b32 %0, hwreg(3)" : "=s"(readback));
        results[idx++] = readback & 0x3;  // just bits 0-1
    }

    // Restore
    asm volatile("s_setreg_b32 hwreg(3), %0" : : "s"(orig));

    // Test: does TRAPSTS state affect computation?
    // Set state 0, do FP op, read state
    asm volatile("s_setreg_b32 hwreg(3), %0" : : "s"(orig & ~0x3u));  // clear bits 0-1

    // Do a computation that MIGHT set exception flags
    volatile float a = 1.0f;
    volatile float b = 0.0f;
    volatile float big = 1e38f;

    // Normal op — should NOT set any exception bits
    volatile float r1 = a + a;
    uint32_t after_normal;
    asm volatile("s_getreg_b32 %0, hwreg(3)" : "=s"(after_normal));
    results[idx++] = after_normal & 0x3;  // bits 0-1 after normal op

    // Near-overflow — might set inexact
    volatile float r2 = big * 1.5f;
    uint32_t after_overflow;
    asm volatile("s_getreg_b32 %0, hwreg(3)" : "=s"(after_overflow));
    results[idx++] = after_overflow & 0x3;

    // Prevent optimization
    results[idx++] = __float_as_uint(r1 + r2);

    // Restore
    asm volatile("s_setreg_b32 hwreg(3), %0" : : "s"(orig));
    results[50] = idx;
}

/* Test B: MODE EXCP_EN bits — which exceptions are enabled? */
__global__ void test_mode_excp(uint32_t* results) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t mode;
    asm volatile("s_getreg_b32 %0, hwreg(1)" : "=s"(mode));
    results[0] = mode;

    // MODE register layout on GFX11:
    // [3:0]   FP_ROUND (32-bit)
    // [7:4]   FP_DENORM (32-bit)
    // [8]     DX10_CLAMP
    // [9]     IEEE
    // [11:10] FP_ROUND (16-bit)
    // [15:12] FP_DENORM (16-bit)
    // [22:16] EXCP_EN — which FP exceptions trigger traps
    //   bit 16: INVALID
    //   bit 17: INPUT_DENORMAL
    //   bit 18: DIV0
    //   bit 19: OVERFLOW
    //   bit 20: UNDERFLOW
    //   bit 21: INEXACT
    //   bit 22: INT_DIV0
    // [23]    GPR_IDX_EN (GFX9+)

    results[1] = (mode >> 16) & 0x7F;  // EXCP_EN bits

    // Try to ENABLE an exception trap (set EXCP_EN bits)
    // This is safe as long as the exception doesn't actually occur
    uint32_t orig_mode = mode;

    // Enable OVERFLOW exception (bit 19)
    uint32_t new_mode = mode | (1u << 19);
    asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(new_mode));
    uint32_t readback;
    asm volatile("s_getreg_b32 %0, hwreg(1)" : "=s"(readback));
    results[2] = (readback >> 16) & 0x7F;  // EXCP_EN after write

    int overflow_enabled = (readback >> 19) & 1;
    results[3] = overflow_enabled;

    // Restore before doing any FP ops that might trap
    asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(orig_mode));

    // Now test: can we enable INVALID and then trigger it?
    // FIRST verify the enable bit sticks
    for (int bit = 16; bit <= 22; bit++) {
        uint32_t test = orig_mode | (1u << bit);
        asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(test));
        asm volatile("s_getreg_b32 %0, hwreg(1)" : "=s"(readback));
        results[4 + (bit - 16)] = (readback >> bit) & 1;  // 1 = bit accepted
        asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(orig_mode));  // restore
    }

    results[20] = 0xBEEF;  // alive
}

/* Test C: s_sendmsg alone (isolated) — does it signal without faulting? */
__global__ void test_sendmsg_isolated(uint32_t* results) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t t0, t1;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t0));

    // s_sendmsg MSG_INTERRUPT (1) — signal host
    asm volatile("s_sendmsg 0x001");

    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t1));
    results[0] = t1 - t0;  // cycles for sendmsg
    results[1] = 0xCAFE;   // alive
}

/* Test D: Wavefront-level control — s_sleep, s_nop, s_barrier timing */
__global__ void test_wave_control(uint32_t* results) {
    uint32_t tid = threadIdx.x;
    if (tid >= 64 || blockIdx.x != 0) return;

    uint32_t t0, t1;

    // s_nop timing
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t0));
    asm volatile("s_nop 0");
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t1));
    results[tid * 4 + 0] = t1 - t0;

    // s_sleep timing (different sleep values = different delays)
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t0));
    asm volatile("s_sleep 1");
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t1));
    results[tid * 4 + 1] = t1 - t0;

    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t0));
    asm volatile("s_sleep 4");
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t1));
    results[tid * 4 + 2] = t1 - t0;

    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t0));
    asm volatile("s_sleep 15");
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t1));
    results[tid * 4 + 3] = t1 - t0;
}

int main() {
    printf("=== SAFE TRAP & WAVE CONTROL PROBE ===\n\n");

    /* Test A */
    printf("--- Test A: TRAPSTS as 2-bit state register ---\n");
    {
        uint32_t *d_r, h[64];
        hipMalloc(&d_r, 64 * sizeof(uint32_t));
        hipMemset(d_r, 0, 64 * sizeof(uint32_t));
        test_trapsts_as_state<<<1, 1>>>(d_r);
        hipError_t e = hipDeviceSynchronize();
        hipMemcpy(h, d_r, 64 * sizeof(uint32_t), hipMemcpyDeviceToHost);
        if (e != hipSuccess) {
            printf("  FAULTED: %s\n", hipGetErrorString(e));
        } else {
            printf("  Original: 0x%x\n", h[0]);
            printf("  State 00: readback=%u, 01: %u, 10: %u, 11: %u\n",
                   h[1], h[2], h[3], h[4]);
            int all_work = (h[1]==0 && h[2]==1 && h[3]==2 && h[4]==3);
            printf("  2-bit state register: %s\n",
                   all_work ? "*** WORKS! ***" : "partial");
            printf("  After normal op: bits=%u\n", h[5]);
            printf("  After near-overflow: bits=%u\n", h[6]);
        }
        printf("\n");
        hipFree(d_r);
    }

    /* Test B */
    printf("--- Test B: MODE EXCP_EN writability ---\n");
    {
        uint32_t *d_r, h[32];
        hipMalloc(&d_r, 32 * sizeof(uint32_t));
        hipMemset(d_r, 0, 32 * sizeof(uint32_t));
        test_mode_excp<<<1, 1>>>(d_r);
        hipError_t e = hipDeviceSynchronize();
        hipMemcpy(h, d_r, 32 * sizeof(uint32_t), hipMemcpyDeviceToHost);
        if (e != hipSuccess) {
            printf("  FAULTED: %s\n", hipGetErrorString(e));
        } else {
            printf("  MODE: 0x%08x, EXCP_EN: 0x%02x\n", h[0], h[1]);
            printf("  After enabling OVERFLOW: EXCP_EN=0x%02x (bit19=%d)\n", h[2], h[3]);
            const char* excp_names[] = {"INVALID", "INPUT_DENORM", "DIV0",
                                         "OVERFLOW", "UNDERFLOW", "INEXACT", "INT_DIV0"};
            printf("  Per-bit EXCP_EN writability:\n");
            for (int i = 0; i < 7; i++) {
                printf("    bit %d (%s): %s\n", 16+i, excp_names[i],
                       h[4+i] ? "WRITABLE" : "read-only");
            }
            printf("  Alive: %s\n", h[20] == 0xBEEF ? "yes" : "DEAD");
        }
        printf("\n");
        hipFree(d_r);
    }

    /* Test C */
    printf("--- Test C: s_sendmsg (isolated) ---\n");
    {
        uint32_t *d_r, h[4];
        hipMalloc(&d_r, 4 * sizeof(uint32_t));
        hipMemset(d_r, 0, 4 * sizeof(uint32_t));
        test_sendmsg_isolated<<<1, 1>>>(d_r);
        hipError_t e = hipDeviceSynchronize();
        hipMemcpy(h, d_r, 4 * sizeof(uint32_t), hipMemcpyDeviceToHost);
        if (e != hipSuccess) {
            printf("  s_sendmsg FAULTED: %s\n", hipGetErrorString(e));
        } else {
            printf("  s_sendmsg cycles: %u\n", h[0]);
            printf("  Alive: %s\n", h[1] == 0xCAFE ? "yes" : "no");
        }
        printf("\n");
        hipFree(d_r);
    }

    /* Test D */
    printf("--- Test D: Wave control timing (s_nop, s_sleep) ---\n");
    {
        uint32_t *d_r, h[256];
        hipMalloc(&d_r, 256 * sizeof(uint32_t));
        hipMemset(d_r, 0, 256 * sizeof(uint32_t));
        test_wave_control<<<1, 64>>>(d_r);
        hipDeviceSynchronize();
        hipMemcpy(h, d_r, 256 * sizeof(uint32_t), hipMemcpyDeviceToHost);

        printf("  Thread 0 timings:\n");
        printf("    s_nop 0:    %u cycles\n", h[0]);
        printf("    s_sleep 1:  %u cycles\n", h[1]);
        printf("    s_sleep 4:  %u cycles\n", h[2]);
        printf("    s_sleep 15: %u cycles\n", h[3]);

        // Analyze across threads
        uint32_t nop_min=UINT32_MAX, nop_max=0;
        uint32_t s1_min=UINT32_MAX, s1_max=0;
        uint32_t s4_min=UINT32_MAX, s4_max=0;
        uint32_t s15_min=UINT32_MAX, s15_max=0;
        for (int i = 0; i < 64; i++) {
            uint32_t n = h[i*4], s1 = h[i*4+1], s4 = h[i*4+2], s15 = h[i*4+3];
            if (n < nop_min) nop_min = n; if (n > nop_max) nop_max = n;
            if (s1 < s1_min) s1_min = s1; if (s1 > s1_max) s1_max = s1;
            if (s4 < s4_min) s4_min = s4; if (s4 > s4_max) s4_max = s4;
            if (s15 < s15_min) s15_min = s15; if (s15 > s15_max) s15_max = s15;
        }
        printf("  Across 64 threads:\n");
        printf("    s_nop:     [%u, %u]\n", nop_min, nop_max);
        printf("    s_sleep 1: [%u, %u]\n", s1_min, s1_max);
        printf("    s_sleep 4: [%u, %u]\n", s4_min, s4_max);
        printf("    s_sleep 15:[%u, %u]\n", s15_min, s15_max);

        int sleep_scales = (s4_min > s1_max) && (s15_min > s4_max);
        printf("  Sleep scales linearly: %s\n",
               sleep_scales ? "YES — controllable delay!" : "no");
    }

    printf("\n=== PROBE COMPLETE ===\n");
    return 0;
}
