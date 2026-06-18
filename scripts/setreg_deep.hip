/*
 * setreg_deep.hip — Deep test of s_setreg capabilities
 * 1. Verify MODE writable with clean read-write-read per thread
 * 2. Test individual MODE bits (FP_ROUND, FP_DENORM, etc.)
 * 3. Test computation with values near rounding boundaries
 * 4. Test FLAT_SCR_LO/HI writability (memory address control!)
 * 5. Test TRAPSTS (trap handler redirection?)
 * 6. Systematic single-thread hwreg write scan with proper isolation
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ============================================================
 * TEST A: Systematic hwreg writability (single thread, no race)
 * Each hwreg tested in complete isolation
 * ============================================================ */
__global__ void test_hwreg_systematic(uint32_t* results) {
    // Only one thread, no scheduling interference
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int idx = 0;

    // For each hwreg: read orig, write pattern, read back, restore, read restored
    // Test pattern carefully chosen — use low bits only to avoid traps

    // hwreg(1) = MODE [FP_ROUND[3:0], FP_DENORM[7:4], DX10_CLAMP[8], IEEE[9], ...]
    {
        uint32_t orig, wb, restored;
        asm volatile("s_getreg_b32 %0, hwreg(1)" : "=s"(orig));

        // Toggle bit 0 (FP32 round mode LSB)
        uint32_t test = orig ^ 0x1;
        asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(test));
        asm volatile("s_getreg_b32 %0, hwreg(1)" : "=s"(wb));

        asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(orig));
        asm volatile("s_getreg_b32 %0, hwreg(1)" : "=s"(restored));

        results[idx++] = 1;      // hwreg id
        results[idx++] = orig;
        results[idx++] = test;   // what we wrote
        results[idx++] = wb;     // what we read back
        results[idx++] = restored;
        results[idx++] = (wb == test) ? 1 : 0;  // writable?
    }

    // hwreg(2) = STATUS
    {
        uint32_t orig, wb, restored;
        asm volatile("s_getreg_b32 %0, hwreg(2)" : "=s"(orig));
        uint32_t test = orig ^ 0x1;
        asm volatile("s_setreg_b32 hwreg(2), %0" : : "s"(test));
        asm volatile("s_getreg_b32 %0, hwreg(2)" : "=s"(wb));
        asm volatile("s_setreg_b32 hwreg(2), %0" : : "s"(orig));
        asm volatile("s_getreg_b32 %0, hwreg(2)" : "=s"(restored));
        results[idx++] = 2;
        results[idx++] = orig;
        results[idx++] = test;
        results[idx++] = wb;
        results[idx++] = restored;
        results[idx++] = (wb == test) ? 1 : 0;
    }

    // hwreg(3) = TRAPSTS
    {
        uint32_t orig, wb, restored;
        asm volatile("s_getreg_b32 %0, hwreg(3)" : "=s"(orig));
        uint32_t test = orig ^ 0x100;  // toggle a safe bit
        asm volatile("s_setreg_b32 hwreg(3), %0" : : "s"(test));
        asm volatile("s_getreg_b32 %0, hwreg(3)" : "=s"(wb));
        asm volatile("s_setreg_b32 hwreg(3), %0" : : "s"(orig));
        asm volatile("s_getreg_b32 %0, hwreg(3)" : "=s"(restored));
        results[idx++] = 3;
        results[idx++] = orig;
        results[idx++] = test;
        results[idx++] = wb;
        results[idx++] = restored;
        results[idx++] = (wb == test) ? 1 : 0;
    }

    // hwreg(14) = FLAT_SCR_LO — this controls flat scratch ADDRESS
    {
        uint32_t orig, wb, restored;
        asm volatile("s_getreg_b32 %0, hwreg(14)" : "=s"(orig));
        uint32_t test = orig ^ 0x1000;  // toggle bit 12
        asm volatile("s_setreg_b32 hwreg(14), %0" : : "s"(test));
        asm volatile("s_getreg_b32 %0, hwreg(14)" : "=s"(wb));
        asm volatile("s_setreg_b32 hwreg(14), %0" : : "s"(orig));
        asm volatile("s_getreg_b32 %0, hwreg(14)" : "=s"(restored));
        results[idx++] = 14;
        results[idx++] = orig;
        results[idx++] = test;
        results[idx++] = wb;
        results[idx++] = restored;
        results[idx++] = (wb == test) ? 1 : 0;
    }

    // hwreg(15) = FLAT_SCR_HI
    {
        uint32_t orig, wb, restored;
        asm volatile("s_getreg_b32 %0, hwreg(15)" : "=s"(orig));
        uint32_t test = orig ^ 0x1;
        asm volatile("s_setreg_b32 hwreg(15), %0" : : "s"(test));
        asm volatile("s_getreg_b32 %0, hwreg(15)" : "=s"(wb));
        asm volatile("s_setreg_b32 hwreg(15), %0" : : "s"(orig));
        asm volatile("s_getreg_b32 %0, hwreg(15)" : "=s"(restored));
        results[idx++] = 15;
        results[idx++] = orig;
        results[idx++] = test;
        results[idx++] = wb;
        results[idx++] = restored;
        results[idx++] = (wb == test) ? 1 : 0;
    }

    // hwreg(5) = GPR_ALLOC
    {
        uint32_t orig, wb;
        asm volatile("s_getreg_b32 %0, hwreg(5)" : "=s"(orig));
        uint32_t test = orig ^ 0x1;
        asm volatile("s_setreg_b32 hwreg(5), %0" : : "s"(test));
        asm volatile("s_getreg_b32 %0, hwreg(5)" : "=s"(wb));
        asm volatile("s_setreg_b32 hwreg(5), %0" : : "s"(orig));
        results[idx++] = 5;
        results[idx++] = orig;
        results[idx++] = test;
        results[idx++] = wb;
        results[idx++] = 0;
        results[idx++] = (wb == test) ? 1 : 0;
    }

    // hwreg(6) = LDS_ALLOC
    {
        uint32_t orig, wb;
        asm volatile("s_getreg_b32 %0, hwreg(6)" : "=s"(orig));
        uint32_t test = orig ^ 0x1;
        asm volatile("s_setreg_b32 hwreg(6), %0" : : "s"(test));
        asm volatile("s_getreg_b32 %0, hwreg(6)" : "=s"(wb));
        asm volatile("s_setreg_b32 hwreg(6), %0" : : "s"(orig));
        results[idx++] = 6;
        results[idx++] = orig;
        results[idx++] = test;
        results[idx++] = wb;
        results[idx++] = 0;
        results[idx++] = (wb == test) ? 1 : 0;
    }

    // hwreg(13) = IB_STS (instruction buffer status)
    {
        uint32_t orig, wb;
        asm volatile("s_getreg_b32 %0, hwreg(13)" : "=s"(orig));
        uint32_t test = orig ^ 0x1;
        asm volatile("s_setreg_b32 hwreg(13), %0" : : "s"(test));
        asm volatile("s_getreg_b32 %0, hwreg(13)" : "=s"(wb));
        asm volatile("s_setreg_b32 hwreg(13), %0" : : "s"(orig));
        results[idx++] = 13;
        results[idx++] = orig;
        results[idx++] = test;
        results[idx++] = wb;
        results[idx++] = 0;
        results[idx++] = (wb == test) ? 1 : 0;
    }

    results[200] = idx;  // total entries written
}

/* ============================================================
 * TEST B: MODE register deep exploration
 * Test each mode bit individually
 * ============================================================ */
__global__ void test_mode_bits(uint32_t* results) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t orig;
    asm volatile("s_getreg_b32 %0, hwreg(1)" : "=s"(orig));
    results[0] = orig;  // original MODE

    // Try setting each bit individually
    for (int bit = 0; bit < 24; bit++) {
        uint32_t test = orig | (1u << bit);
        asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(test));
        uint32_t readback;
        asm volatile("s_getreg_b32 %0, hwreg(1)" : "=s"(readback));
        results[1 + bit] = readback;
        // Restore
        asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(orig));
    }
    results[25] = 0xDEAD;  // sentinel
}

/* ============================================================
 * TEST C: Rounding boundary computation
 * Choose values where rounding mode MATTERS
 * ============================================================ */
__global__ void test_rounding_boundary(uint32_t* results) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Save original mode
    uint32_t orig_mode;
    asm volatile("s_getreg_b32 %0, hwreg(1)" : "=s"(orig_mode));

    int idx = 0;

    // Value exactly at midpoint: 1.5 * 2^-24 + 1.0
    // In FP32: 1.0 = 0x3F800000. Adding 1.5 ULP should give different results
    // for RNE vs RTZ vs RPI vs RNI

    // Test 1: a + b where result is exactly between two representable values
    // 0x3F800001 = 1.0000001192... (1 + 1 ULP)
    // 0x3F800000 + 0x33C00000 = 1.0 + 0.75*2^-23 = exactly halfway
    float a = 1.0f;
    // b = 0.75 * 2^-23 = exactly half ULP of 1.0
    uint32_t b_bits = 0x33C00000;
    float b;
    memcpy(&b, &b_bits, 4);

    // Mode 0: Round to Nearest Even (default)
    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 4), 0x0");
    float r0 = a + b;
    uint32_t r0b; memcpy(&r0b, &r0, 4);

    // Mode 1: Round to +Infinity
    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 4), 0x1");
    float r1 = a + b;
    uint32_t r1b; memcpy(&r1b, &r1, 4);

    // Mode 2: Round to -Infinity
    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 4), 0x2");
    float r2 = a + b;
    uint32_t r2b; memcpy(&r2b, &r2, 4);

    // Mode 3: Round toward Zero
    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 4), 0x3");
    float r3 = a + b;
    uint32_t r3b; memcpy(&r3b, &r3, 4);

    // Restore
    asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(orig_mode));

    results[idx++] = r0b;  // RNE
    results[idx++] = r1b;  // RPI
    results[idx++] = r2b;  // RNI
    results[idx++] = r3b;  // RTZ

    // Test 2: multiplication near boundary
    float c = 1.00000011920928955f;  // 1 + 1 ULP
    float d = 1.00000011920928955f;

    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 4), 0x0");
    float m0 = c * d;
    uint32_t m0b; memcpy(&m0b, &m0, 4);

    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 4), 0x1");
    float m1 = c * d;
    uint32_t m1b; memcpy(&m1b, &m1, 4);

    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 4), 0x3");
    float m3 = c * d;
    uint32_t m3b; memcpy(&m3b, &m3, 4);

    asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(orig_mode));

    results[idx++] = m0b;
    results[idx++] = m1b;
    results[idx++] = m3b;

    // Test 3: large number of accumulations (rounding error compounds)
    for (int mode = 0; mode < 4; mode++) {
        uint32_t mode_val = mode;
        asm volatile("s_setreg_b32 hwreg(1, 0, 4), %0" : : "s"(mode_val));

        float acc = 0.0f;
        for (int i = 0; i < 1000; i++) {
            acc += 0.1f;  // 0.1 is not exactly representable
        }
        uint32_t accb; memcpy(&accb, &acc, 4);
        results[idx++] = accb;
    }

    asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(orig_mode));
    results[50] = idx;
}

int main() {
    printf("=== DEEP s_setreg PROBE ===\n\n");

    /* ---- Test A: Systematic writability ---- */
    printf("--- TEST A: Systematic hwreg writability (single thread, no race) ---\n");
    {
        uint32_t *d_r, h_r[256];
        hipMalloc(&d_r, 256 * sizeof(uint32_t));
        hipMemset(d_r, 0, 256 * sizeof(uint32_t));
        test_hwreg_systematic<<<1, 1>>>(d_r);
        hipDeviceSynchronize();
        hipMemcpy(h_r, d_r, 256 * sizeof(uint32_t), hipMemcpyDeviceToHost);

        const char* names[] = {"MODE(1)", "STATUS(2)", "TRAPSTS(3)",
                                "FLAT_SCR_LO(14)", "FLAT_SCR_HI(15)",
                                "GPR_ALLOC(5)", "LDS_ALLOC(6)", "IB_STS(13)"};
        for (int i = 0; i < 8; i++) {
            int base = i * 6;
            printf("  %-18s orig=0x%08x wrote=0x%08x read=0x%08x %s\n",
                   names[i], h_r[base+1], h_r[base+2], h_r[base+3],
                   h_r[base+5] ? "*** WRITABLE ***" : "read-only");
        }
        printf("\n");
        hipFree(d_r);
    }

    /* ---- Test B: MODE bit map ---- */
    printf("--- TEST B: MODE register bit map ---\n");
    {
        uint32_t *d_r, h_r[32];
        hipMalloc(&d_r, 32 * sizeof(uint32_t));
        hipMemset(d_r, 0, 32 * sizeof(uint32_t));
        test_mode_bits<<<1, 1>>>(d_r);
        hipDeviceSynchronize();
        hipMemcpy(h_r, d_r, 32 * sizeof(uint32_t), hipMemcpyDeviceToHost);

        printf("  Original MODE: 0x%08x\n", h_r[0]);
        printf("  Bit map (which bits accept writes):\n  ");
        for (int bit = 0; bit < 24; bit++) {
            uint32_t expected = h_r[0] | (1u << bit);
            int accepted = (h_r[1 + bit] == expected);
            printf("%c", accepted ? '1' : '0');
            if (bit % 4 == 3) printf(" ");
        }
        printf("\n");
        printf("  Bit meanings: [3:0]=FP32_ROUND [7:4]=FP32_DENORM [8]=DX10_CLAMP [9]=IEEE\n");
        printf("                [11:10]=FP16_ROUND [15:12]=FP16_DENORM [18:16]=FP_ROUND_16_64\n");
        printf("                [22:20]=FP_DENORM_16_64 [23]=EXCP_EN\n");
        for (int bit = 0; bit < 24; bit++) {
            uint32_t expected = h_r[0] | (1u << bit);
            printf("    bit %2d: wrote 0x%08x, read 0x%08x %s\n",
                   bit, expected, h_r[1 + bit],
                   (h_r[1 + bit] == expected) ? "ACCEPTED" : "REJECTED");
        }
        printf("\n");
        hipFree(d_r);
    }

    /* ---- Test C: Rounding boundary computation ---- */
    printf("--- TEST C: Rounding affects computation? ---\n");
    {
        uint32_t *d_r, h_r[64];
        hipMalloc(&d_r, 64 * sizeof(uint32_t));
        hipMemset(d_r, 0, 64 * sizeof(uint32_t));
        test_rounding_boundary<<<1, 1>>>(d_r);
        hipDeviceSynchronize();
        hipMemcpy(h_r, d_r, 64 * sizeof(uint32_t), hipMemcpyDeviceToHost);

        printf("  Addition: 1.0 + 0.75*ULP\n");
        float v; uint32_t b;
        b = h_r[0]; memcpy(&v, &b, 4); printf("    RNE: 0x%08x = %.10e\n", b, v);
        b = h_r[1]; memcpy(&v, &b, 4); printf("    RPI: 0x%08x = %.10e\n", b, v);
        b = h_r[2]; memcpy(&v, &b, 4); printf("    RNI: 0x%08x = %.10e\n", b, v);
        b = h_r[3]; memcpy(&v, &b, 4); printf("    RTZ: 0x%08x = %.10e\n", b, v);

        int add_differs = (h_r[0] != h_r[1]) || (h_r[0] != h_r[2]) || (h_r[0] != h_r[3]);
        printf("    Different results: %s\n\n", add_differs ? "*** YES ***" : "no");

        printf("  Multiplication: (1+ULP) * (1+ULP)\n");
        b = h_r[4]; memcpy(&v, &b, 4); printf("    RNE: 0x%08x = %.10e\n", b, v);
        b = h_r[5]; memcpy(&v, &b, 4); printf("    RPI: 0x%08x = %.10e\n", b, v);
        b = h_r[6]; memcpy(&v, &b, 4); printf("    RTZ: 0x%08x = %.10e\n", b, v);

        int mul_differs = (h_r[4] != h_r[5]) || (h_r[4] != h_r[6]);
        printf("    Different results: %s\n\n", mul_differs ? "*** YES ***" : "no");

        printf("  Accumulation: sum(0.1, 1000 times)\n");
        const char* rnames[] = {"RNE", "RPI", "RNI", "RTZ"};
        for (int i = 0; i < 4; i++) {
            b = h_r[7 + i]; memcpy(&v, &b, 4);
            printf("    %s: 0x%08x = %.10e\n", rnames[i], b, v);
        }
        int acc_differs = (h_r[7] != h_r[8]) || (h_r[7] != h_r[9]) || (h_r[7] != h_r[10]);
        printf("    Different results: %s\n", acc_differs ? "*** YES — ROUNDING IS CONSTITUTIVE! ***" : "no");
        printf("\n");

        hipFree(d_r);
    }

    printf("=== PROBE COMPLETE ===\n");
    return 0;
}
