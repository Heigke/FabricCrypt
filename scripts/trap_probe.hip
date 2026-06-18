/*
 * trap_probe.hip — Investigate GPU trap mechanism as neuromorphic primitive
 *
 * Tests:
 *  1. Read TBA (Trap Base Address) and TMA (Trap Memory Address)
 *  2. Map which TRAPSTS bits are writable without causing fault
 *  3. Test if we can install a custom trap handler
 *  4. Test trap-based "spike" mechanism
 *  5. Probe all trap-related hwregs systematically
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>

/* ============================================================
 * TEST 1: Read all trap-related registers
 * hwreg(3)  = TRAPSTS — trap status (we know bit 8 writable)
 * hwreg(4)  = HW_ID — hardware thread ID
 * hwreg(7)  = GPR_ALLOC
 * hwreg(8)  = PC_LO — program counter low
 * hwreg(9)  = PC_HI — program counter high
 * hwreg(10) = INST_DW0 — current instruction word 0
 * hwreg(11) = INST_DW1 — current instruction word 1
 * hwreg(12) = IB_DBG0
 * hwreg(23) = HW_ID1 — WGP placement
 * hwreg(24) = HW_ID2 — VMID, pipe, queue
 * hwreg(25) = POPS_PACKER — overlap packer state
 * ============================================================ */
__global__ void test_read_trap_regs(uint32_t* results) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int idx = 0;
    uint32_t val;

    // Read all potentially trap-related hwregs
    #define READ_HWREG(N) \
        asm volatile("s_getreg_b32 %0, hwreg(" #N ")" : "=s"(val)); \
        results[idx++] = N; \
        results[idx++] = val;

    READ_HWREG(0)   // HW_ID (GFX10+)
    READ_HWREG(1)   // MODE
    READ_HWREG(2)   // STATUS
    READ_HWREG(3)   // TRAPSTS
    READ_HWREG(4)   // HW_ID (legacy)
    READ_HWREG(5)   // GPR_ALLOC
    READ_HWREG(6)   // LDS_ALLOC
    READ_HWREG(7)   // IB_STS
    READ_HWREG(8)   // PC_LO
    READ_HWREG(9)   // PC_HI
    READ_HWREG(10)  // INST_DW0
    READ_HWREG(11)  // INST_DW1
    READ_HWREG(12)  // IB_DBG0
    READ_HWREG(13)  // IB_STS2 / FLUSH_IB
    READ_HWREG(14)  // FLAT_SCR_LO
    READ_HWREG(15)  // FLAT_SCR_HI
    READ_HWREG(16)  // XNACK_MASK
    READ_HWREG(17)  // POPS_PACKER
    READ_HWREG(18)  // reg18
    READ_HWREG(19)  // SHADER_CYCLES (perf)
    READ_HWREG(20)  // reg20
    READ_HWREG(21)  // reg21
    READ_HWREG(22)  // reg22
    READ_HWREG(23)  // HW_ID1
    READ_HWREG(24)  // HW_ID2
    READ_HWREG(25)  // POPS_PACKER
    READ_HWREG(26)  // reg26
    READ_HWREG(27)  // PERF_SNAPSHOT
    READ_HWREG(28)  // reg28
    READ_HWREG(29)  // SHADER_CYCLES
    READ_HWREG(30)  // reg30
    READ_HWREG(31)  // reg31

    results[200] = idx;
}

/* ============================================================
 * TEST 2: Probe TRAPSTS bit-by-bit — which bits writable?
 * We already know bit 8 causes a fault. Test EACH bit individually
 * with immediate restore, using the safest possible pattern.
 * ============================================================ */
__global__ void test_trapsts_bits(uint32_t* results) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t orig;
    asm volatile("s_getreg_b32 %0, hwreg(3)" : "=s"(orig));
    results[0] = orig;

    // Test each bit: set it, read back, immediately clear
    for (int bit = 0; bit < 32; bit++) {
        uint32_t test_val = orig | (1u << bit);
        uint32_t readback;

        asm volatile("s_setreg_b32 hwreg(3), %0" : : "s"(test_val));
        asm volatile("s_getreg_b32 %0, hwreg(3)" : "=s"(readback));
        // IMMEDIATELY restore to prevent trap dispatch
        asm volatile("s_setreg_b32 hwreg(3), %0" : : "s"(orig));

        // Store: 1 = bit accepted, 0 = bit rejected
        results[1 + bit] = (readback & (1u << bit)) ? 1 : 0;
    }

    // Verify we're still alive
    results[33] = 0xA11CE;
    // Store final TRAPSTS
    uint32_t final_val;
    asm volatile("s_getreg_b32 %0, hwreg(3)" : "=s"(final_val));
    results[34] = final_val;
}

/* ============================================================
 * TEST 3: Read TBA/TMA — trap handler address
 * On GFX9+, TBA and TMA are accessible via s_getreg or via
 * specific SQ registers. They may also be in SGPR space.
 * On RDNA, the trap handler is set by the driver/MES.
 * ============================================================ */
__global__ void test_trap_handler_addr(uint32_t* results) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // On GFX11/RDNA3, TBA might be in ttmp registers (trap temporary)
    // ttmp0-ttmp15 are accessible via s_getreg on some architectures
    // Try reading them

    uint32_t val;
    int idx = 0;

    // STATUS register contains TRAP bit (bit 6) and TRAP_EN (bit 7)
    asm volatile("s_getreg_b32 %0, hwreg(2)" : "=s"(val));
    results[idx++] = 2;  // STATUS
    results[idx++] = val;

    // Check if TRAP_EN is set in STATUS
    results[idx++] = (val >> 6) & 0x3;  // bits 6-7: TRAP, TRAP_EN

    // MODE register — check EXCP_EN bits
    asm volatile("s_getreg_b32 %0, hwreg(1)" : "=s"(val));
    results[idx++] = 1;  // MODE
    results[idx++] = val;
    results[idx++] = (val >> 12) & 0x1FF;  // EXCP_EN bits [20:12]

    // On RDNA3, s_sendmsg might be available for trap signaling
    // But we need to be careful not to trigger actual traps

    // Read PC to understand where we are
    uint32_t pc_lo, pc_hi;
    asm volatile("s_getreg_b32 %0, hwreg(8)" : "=s"(pc_lo));
    asm volatile("s_getreg_b32 %0, hwreg(9)" : "=s"(pc_hi));
    results[idx++] = pc_lo;
    results[idx++] = pc_hi;

    // HW_ID2 gives VMID — important for trap handler context
    asm volatile("s_getreg_b32 %0, hwreg(24)" : "=s"(val));
    results[idx++] = 24;
    results[idx++] = val;
    results[idx++] = val & 0xF;  // VMID (bits 3:0)

    results[50] = idx;
}

/* ============================================================
 * TEST 4: s_trap instruction — controlled trap trigger
 * s_trap N sends trap with code N. If trap handler exists,
 * execution jumps there. If not, GPU faults.
 * We test with s_trap 0 which is the mildest form.
 * NOTE: This WILL fault if no trap handler is installed.
 * Only run if we determine trap handler exists.
 * ============================================================ */
__global__ void test_s_sendmsg(uint32_t* results) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // s_sendmsg is safer than s_trap — it sends a message to
    // the host/scheduler without necessarily faulting
    // Message types:
    //   MSG_INTERRUPT = 1 — send interrupt to host
    //   MSG_GS = 2 — geometry shader related
    //   MSG_GS_DONE = 3
    //   MSG_SAVEWAVE = 4 — save wavefront state
    //   MSG_HALT_COND = 8 — conditional halt
    //   MSG_ORDERED_PS_DONE = 13

    // Read pre-state
    uint32_t status_before, trapsts_before;
    asm volatile("s_getreg_b32 %0, hwreg(2)" : "=s"(status_before));
    asm volatile("s_getreg_b32 %0, hwreg(3)" : "=s"(trapsts_before));

    results[0] = status_before;
    results[1] = trapsts_before;

    // Try s_sendmsg with MSG_INTERRUPT (type 1)
    // This should signal the host without crashing
    // Format: s_sendmsg sendmsg(MSG_INTERRUPT)
    // The immediate is: (MSG_ID) | (OP << 4) | (STREAM << 8)
    // MSG_INTERRUPT = 1, OP=0, STREAM=0 → imm = 0x001
    asm volatile("s_sendmsg 0x001");

    // Read post-state
    uint32_t status_after, trapsts_after;
    asm volatile("s_getreg_b32 %0, hwreg(2)" : "=s"(status_after));
    asm volatile("s_getreg_b32 %0, hwreg(3)" : "=s"(trapsts_after));

    results[2] = status_after;
    results[3] = trapsts_after;

    // Check what changed
    results[4] = status_before ^ status_after;   // changed bits
    results[5] = trapsts_before ^ trapsts_after;  // changed bits
    results[6] = 0xDEAD;  // alive marker
}

/* ============================================================
 * TEST 5: s_sethalt — halt wavefront execution
 * This pauses the wavefront and signals the host.
 * On RDNA, this is used by the debugger (rocgdb).
 * Could be a "spike" mechanism — wavefront fires and halts.
 * ============================================================ */
__global__ void test_sethalt(uint32_t* results, volatile int* flag) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Record pre-halt state
    uint32_t cycles_before;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(cycles_before));

    if (tid < 64) {
        results[tid * 3] = cycles_before;

        // Small computation
        float x = sinf((float)tid * 0.1f);
        results[tid * 3 + 1] = __float_as_uint(x);

        // Post computation cycles
        uint32_t cycles_after;
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(cycles_after));
        results[tid * 3 + 2] = cycles_after - cycles_before;
    }

    // Signal completion
    if (tid == 0) *flag = 1;
}

int main() {
    printf("=== TRAP MECHANISM PROBE ===\n\n");

    /* ---- Test 1: Read all hwregs ---- */
    printf("--- TEST 1: All hwreg values ---\n");
    {
        uint32_t *d_r, h_r[256];
        hipMalloc(&d_r, 256 * sizeof(uint32_t));
        hipMemset(d_r, 0, 256 * sizeof(uint32_t));
        test_read_trap_regs<<<1, 1>>>(d_r);
        hipDeviceSynchronize();
        hipMemcpy(h_r, d_r, 256 * sizeof(uint32_t), hipMemcpyDeviceToHost);

        int n = h_r[200] / 2;
        const char* names[] = {
            "HW_ID0", "MODE", "STATUS", "TRAPSTS",
            "HW_ID_legacy", "GPR_ALLOC", "LDS_ALLOC", "IB_STS",
            "PC_LO", "PC_HI", "INST_DW0", "INST_DW1",
            "IB_DBG0", "IB_STS2", "FLAT_SCR_LO", "FLAT_SCR_HI",
            "XNACK_MASK", "POPS_PACKER", "reg18", "SHADER_CYCLES_0",
            "reg20", "reg21", "reg22", "HW_ID1",
            "HW_ID2", "POPS_PACKER2", "reg26", "PERF_SNAPSHOT",
            "reg28", "SHADER_CYCLES", "reg30", "reg31"
        };

        for (int i = 0; i < n && i < 32; i++) {
            uint32_t id = h_r[i * 2];
            uint32_t val = h_r[i * 2 + 1];
            const char* name = (id < 32) ? names[id] : "?";
            if (val != 0)
                printf("  hwreg(%2u) %-16s = 0x%08x\n", id, name, val);
        }
        printf("\n");
        hipFree(d_r);
    }

    /* ---- Test 2: TRAPSTS bit map ---- */
    printf("--- TEST 2: TRAPSTS bit-by-bit writability ---\n");
    {
        uint32_t *d_r, h_r[64];
        hipMalloc(&d_r, 64 * sizeof(uint32_t));
        hipMemset(d_r, 0, 64 * sizeof(uint32_t));
        test_trapsts_bits<<<1, 1>>>(d_r);
        hipError_t err = hipDeviceSynchronize();
        hipMemcpy(h_r, d_r, 64 * sizeof(uint32_t), hipMemcpyDeviceToHost);

        if (err != hipSuccess) {
            printf("  GPU FAULTED during TRAPSTS probe: %s\n", hipGetErrorString(err));
            printf("  Original TRAPSTS: 0x%08x\n", h_r[0]);
            // Show which bits we got before fault
            printf("  Bits tested before fault: ");
            for (int b = 0; b < 32; b++) {
                if (h_r[1 + b] == 1) printf("%d ", b);
            }
            printf("\n");
        } else {
            printf("  Original TRAPSTS: 0x%08x\n", h_r[0]);
            printf("  Alive marker: 0x%08x\n", h_r[33]);
            printf("  Final TRAPSTS: 0x%08x\n", h_r[34]);
            printf("  Writable bits: ");
            int writable_count = 0;
            for (int b = 0; b < 32; b++) {
                if (h_r[1 + b] == 1) {
                    printf("%d ", b);
                    writable_count++;
                }
            }
            printf("\n  Total writable: %d/32\n", writable_count);

            printf("  Bit map: ");
            for (int b = 0; b < 32; b++) {
                printf("%c", h_r[1 + b] ? '1' : '0');
                if (b % 4 == 3) printf(" ");
            }
            printf("\n");
        }
        printf("\n");
        hipFree(d_r);
    }

    /* ---- Test 3: Trap handler address ---- */
    printf("--- TEST 3: Trap handler context ---\n");
    {
        uint32_t *d_r, h_r[64];
        hipMalloc(&d_r, 64 * sizeof(uint32_t));
        hipMemset(d_r, 0, 64 * sizeof(uint32_t));
        test_trap_handler_addr<<<1, 1>>>(d_r);
        hipDeviceSynchronize();
        hipMemcpy(h_r, d_r, 64 * sizeof(uint32_t), hipMemcpyDeviceToHost);

        printf("  STATUS: 0x%08x (TRAP=%d, TRAP_EN=%d)\n",
               h_r[1], (h_r[2] >> 0) & 1, (h_r[2] >> 1) & 1);
        printf("  MODE: 0x%08x (EXCP_EN=0x%03x)\n", h_r[4], h_r[5]);
        printf("  PC: 0x%08x_%08x\n", h_r[7], h_r[6]);
        printf("  HW_ID2: 0x%08x (VMID=%d)\n", h_r[9], h_r[10]);
        printf("\n");

        hipFree(d_r);
    }

    /* ---- Test 4: s_sendmsg ---- */
    printf("--- TEST 4: s_sendmsg (host interrupt) ---\n");
    {
        uint32_t *d_r, h_r[16];
        hipMalloc(&d_r, 16 * sizeof(uint32_t));
        hipMemset(d_r, 0, 16 * sizeof(uint32_t));
        test_s_sendmsg<<<1, 1>>>(d_r);
        hipError_t err = hipDeviceSynchronize();
        hipMemcpy(h_r, d_r, 16 * sizeof(uint32_t), hipMemcpyDeviceToHost);

        if (err != hipSuccess) {
            printf("  s_sendmsg FAULTED: %s\n", hipGetErrorString(err));
        } else {
            printf("  STATUS before: 0x%08x, after: 0x%08x (changed: 0x%08x)\n",
                   h_r[0], h_r[2], h_r[4]);
            printf("  TRAPSTS before: 0x%08x, after: 0x%08x (changed: 0x%08x)\n",
                   h_r[1], h_r[3], h_r[5]);
            printf("  Alive: %s\n", h_r[6] == 0xDEAD ? "YES" : "no");

            if (h_r[4] == 0 && h_r[5] == 0)
                printf("  s_sendmsg had NO observable effect on STATUS/TRAPSTS\n");
            else
                printf("  s_sendmsg CHANGED state!\n");
        }
        printf("\n");

        hipFree(d_r);
    }

    /* ---- Test 5: Baseline timing without traps ---- */
    printf("--- TEST 5: Wavefront timing baseline ---\n");
    {
        uint32_t *d_r, h_r[256];
        volatile int *d_flag;
        hipMalloc(&d_r, 256 * sizeof(uint32_t));
        hipMalloc((void**)&d_flag, sizeof(int));
        hipMemset(d_r, 0, 256 * sizeof(uint32_t));
        hipMemset((void*)d_flag, 0, sizeof(int));

        test_sethalt<<<1, 64>>>(d_r, d_flag);
        hipDeviceSynchronize();
        hipMemcpy(h_r, d_r, 256 * sizeof(uint32_t), hipMemcpyDeviceToHost);

        printf("  Per-thread cycle counts (first 8 threads):\n");
        for (int i = 0; i < 8; i++) {
            printf("    thread %d: cycles=%u sin=%.4f\n",
                   i, h_r[i * 3 + 2],
                   *((float*)&h_r[i * 3 + 1]));
        }

        // Analyze jitter
        uint32_t min_c = UINT32_MAX, max_c = 0;
        for (int i = 0; i < 64; i++) {
            uint32_t c = h_r[i * 3 + 2];
            if (c < min_c) min_c = c;
            if (c > max_c) max_c = c;
        }
        printf("  Cycle range: [%u, %u] (jitter = %u)\n", min_c, max_c, max_c - min_c);

        hipFree(d_r);
        hipFree((void*)d_flag);
    }

    printf("\n=== TRAP PROBE COMPLETE ===\n");
    return 0;
}
