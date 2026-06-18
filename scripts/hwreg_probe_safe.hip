// hwreg_probe_safe.hip — Probe hwregs ONE AT A TIME to find all dynamic registers
// Reads each hwreg 100 times from the same kernel to detect variability
// Safe: only one hwreg per kernel launch, with error checking
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/hwreg_probe_safe scripts/hwreg_probe_safe.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/hwreg_probe_safe

#include <hip/hip_runtime.h>
#include <cstdio>

#define HIP_CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        printf("HIP ERROR: %s\n", hipGetErrorString(e)); \
        return; \
    } \
} while(0)

#define N_READS 100

// Each kernel reads ONE specific hwreg N_READS times + does work between reads
#define MAKE_HWREG_KERNEL(NUM) \
__global__ void k_hwreg_##NUM(uint32_t* out) { \
    if (threadIdx.x != 0) return; \
    for (int i = 0; i < N_READS; i++) { \
        uint32_t v; \
        asm volatile("s_getreg_b32 %0, hwreg(" #NUM ")" : "=s"(v)); \
        out[i] = v; \
        /* Do work between reads */ \
        float d = 1.0f; \
        for (int j = 0; j < 100 * (i+1); j++) d *= 1.00001f; \
        if (d < 0) out[0] = 0; \
    } \
}

MAKE_HWREG_KERNEL(0)
MAKE_HWREG_KERNEL(1)
MAKE_HWREG_KERNEL(2)
MAKE_HWREG_KERNEL(3)
MAKE_HWREG_KERNEL(4)
MAKE_HWREG_KERNEL(5)
MAKE_HWREG_KERNEL(6)
MAKE_HWREG_KERNEL(7)
MAKE_HWREG_KERNEL(8)
MAKE_HWREG_KERNEL(9)
MAKE_HWREG_KERNEL(10)
MAKE_HWREG_KERNEL(11)
MAKE_HWREG_KERNEL(12)
MAKE_HWREG_KERNEL(13)
MAKE_HWREG_KERNEL(14)
MAKE_HWREG_KERNEL(15)
MAKE_HWREG_KERNEL(16)
MAKE_HWREG_KERNEL(17)
MAKE_HWREG_KERNEL(18)
MAKE_HWREG_KERNEL(19)
MAKE_HWREG_KERNEL(20)
MAKE_HWREG_KERNEL(21)
MAKE_HWREG_KERNEL(22)
MAKE_HWREG_KERNEL(23)
MAKE_HWREG_KERNEL(24)
MAKE_HWREG_KERNEL(25)
MAKE_HWREG_KERNEL(26)
MAKE_HWREG_KERNEL(27)
MAKE_HWREG_KERNEL(28)
MAKE_HWREG_KERNEL(29)
MAKE_HWREG_KERNEL(30)
MAKE_HWREG_KERNEL(31)

// Atomic kernel for contention test
__global__ void k_atomic_order(unsigned int* ctr, unsigned int* res) {
    unsigned int old = atomicAdd(ctr, 1u);
    res[threadIdx.x + blockIdx.x * blockDim.x] = old;
}

typedef void (*hwreg_func)(uint32_t*);

void probe_one_hwreg(int id, hwreg_func func) {
    uint32_t *d_out;
    uint32_t h_out[N_READS];

    hipError_t e;
    e = hipMalloc(&d_out, N_READS * sizeof(uint32_t));
    if (e != hipSuccess) { printf("  hwreg(%2d): MALLOC FAILED\n", id); return; }

    hipMemset(d_out, 0xAA, N_READS * sizeof(uint32_t));

    func<<<1, 64>>>(d_out);
    e = hipDeviceSynchronize();
    if (e != hipSuccess) {
        printf("  hwreg(%2d): KERNEL FAILED (%s)\n", id, hipGetErrorString(e));
        hipFree(d_out);
        return;
    }

    hipMemcpy(h_out, d_out, N_READS * sizeof(uint32_t), hipMemcpyDeviceToHost);

    // Analysis
    int n_unique = 0;
    uint32_t min_val = h_out[0], max_val = h_out[0];
    bool monotonic = true;
    for (int i = 0; i < N_READS; i++) {
        if (h_out[i] < min_val) min_val = h_out[i];
        if (h_out[i] > max_val) max_val = h_out[i];
        if (i > 0 && h_out[i] < h_out[i-1]) monotonic = false;
        bool seen = false;
        for (int j = 0; j < i; j++) if (h_out[j] == h_out[i]) { seen = true; break; }
        if (!seen) n_unique++;
    }

    // Show interesting ones
    const char* tags = "";
    if (n_unique == 1 && h_out[0] == 0xAAAAAAAA) tags = "NOT_WRITTEN";
    else if (n_unique == 1) tags = "CONSTANT";
    else if (monotonic && n_unique > 10) tags = "COUNTER ★★★";
    else if (n_unique > 50) tags = "HIGH_ENTROPY ★★★";
    else if (n_unique > 10) tags = "DYNAMIC ★★";
    else if (n_unique > 1) tags = "SOME_CHANGE ★";

    printf("  hwreg(%2d): unique=%3d/%d  range=[0x%08X, 0x%08X]  mono=%d  %s",
           id, n_unique, N_READS, min_val, max_val, monotonic, tags);

    // Show first few values for interesting ones
    if (n_unique > 1) {
        printf("\n            first 10: ");
        for (int i = 0; i < 10 && i < N_READS; i++) printf("0x%X ", h_out[i]);
    }
    printf("\n");

    // Second launch — does the starting value change?
    hipMemset(d_out, 0xBB, N_READS * sizeof(uint32_t));
    func<<<1, 64>>>(d_out);
    e = hipDeviceSynchronize();
    if (e == hipSuccess) {
        uint32_t h_out2[N_READS];
        hipMemcpy(h_out2, d_out, N_READS * sizeof(uint32_t), hipMemcpyDeviceToHost);
        if (h_out2[0] != h_out[0] || h_out2[N_READS-1] != h_out[N_READS-1]) {
            printf("            CROSS-LAUNCH: first 0x%X→0x%X  last 0x%X→0x%X\n",
                   h_out[0], h_out2[0], h_out[N_READS-1], h_out2[N_READS-1]);
        }
    }

    hipFree(d_out);
}

int main() {
    printf("=== SAFE HWREG PROBE — ONE REGISTER AT A TIME ===\n");
    printf("Each register read 100 times within one kernel + 100 more in second launch\n\n");

    hwreg_func funcs[] = {
        k_hwreg_0, k_hwreg_1, k_hwreg_2, k_hwreg_3,
        k_hwreg_4, k_hwreg_5, k_hwreg_6, k_hwreg_7,
        k_hwreg_8, k_hwreg_9, k_hwreg_10, k_hwreg_11,
        k_hwreg_12, k_hwreg_13, k_hwreg_14, k_hwreg_15,
        k_hwreg_16, k_hwreg_17, k_hwreg_18, k_hwreg_19,
        k_hwreg_20, k_hwreg_21, k_hwreg_22, k_hwreg_23,
        k_hwreg_24, k_hwreg_25, k_hwreg_26, k_hwreg_27,
        k_hwreg_28, k_hwreg_29, k_hwreg_30, k_hwreg_31,
    };

    const char* names[] = {
        "???", "MODE", "STATUS", "TRAPSTS",
        "HW_ID1", "HW_ID2", "GPR_ALLOC", "LDS_ALLOC",
        "IB_STS", "???", "???", "???",
        "???", "???", "???", "SH_MEM_BASES",
        "FLAT_SCR_LO", "FLAT_SCR_HI", "???", "???",
        "???", "???", "???", "HW_ID1_v2",
        "HW_ID2_v2", "???", "???", "PERF_SNAP?",
        "???", "SHADER_CYCLES", "???", "???"
    };

    for (int i = 0; i < 32; i++) {
        printf("  [%-15s] ", names[i]);
        fflush(stdout);
        probe_one_hwreg(i, funcs[i]);
        fflush(stdout);
    }

    // Now test deeper: hwreg with different bit fields
    printf("\n=== PERF_SNAPSHOT (hwreg 27) DEEP ANALYSIS ===\n");
    {
        uint32_t *d_out;
        hipMalloc(&d_out, N_READS * sizeof(uint32_t));
        for (int run = 0; run < 5; run++) {
            hipMemset(d_out, 0, N_READS * sizeof(uint32_t));
            k_hwreg_27<<<1, 64>>>(d_out);
            hipDeviceSynchronize();
            uint32_t h[N_READS];
            hipMemcpy(h, d_out, N_READS * sizeof(uint32_t), hipMemcpyDeviceToHost);
            printf("  run %d: ", run);
            for (int i = 0; i < 20; i++) printf("%u ", h[i]);
            printf("...\n");
        }
        hipFree(d_out);
    }

    printf("\n=== SHADER_CYCLES (hwreg 29) DEEP ANALYSIS ===\n");
    {
        uint32_t *d_out;
        hipMalloc(&d_out, N_READS * sizeof(uint32_t));
        for (int run = 0; run < 5; run++) {
            hipMemset(d_out, 0, N_READS * sizeof(uint32_t));
            k_hwreg_29<<<1, 64>>>(d_out);
            hipDeviceSynchronize();
            uint32_t h[N_READS];
            hipMemcpy(h, d_out, N_READS * sizeof(uint32_t), hipMemcpyDeviceToHost);
            printf("  run %d first=%u last=%u  deltas: ", run, h[0], h[N_READS-1]);
            for (int i = 1; i < 15; i++) printf("%d ", (int)(h[i] - h[i-1]));
            printf("...\n");
        }
        hipFree(d_out);
    }

    printf("\n=== ATOMIC ORDER ENTROPY (deeper) ===\n");
    printf("Testing if atomic completion order varies across 10 launches...\n");
    {
        unsigned int *d_counter, *d_results;
        hipMalloc(&d_counter, sizeof(unsigned int));
        hipMalloc(&d_results, 256 * sizeof(unsigned int));

        uint32_t prev_first16[16] = {};
        int total_order_changes = 0;

        for (int run = 0; run < 10; run++) {
            hipMemset(d_counter, 0, sizeof(unsigned int));
            hipMemset(d_results, 0xFF, 256 * sizeof(unsigned int));

            k_atomic_order<<<4, 64>>>(d_counter, d_results);
            hipDeviceSynchronize();

            unsigned int h[256];
            hipMemcpy(h, d_results, 256 * sizeof(unsigned int), hipMemcpyDeviceToHost);

            int changed = 0;
            if (run > 0) {
                for (int i = 0; i < 16; i++) if (h[i] != prev_first16[i]) changed++;
                total_order_changes += changed;
            }
            printf("  run %d: first16=[", run);
            for (int i = 0; i < 16; i++) { printf("%u", h[i]); if (i<15) printf(","); }
            printf("]");
            if (run > 0) printf("  changed=%d/16", changed);
            printf("\n");

            for (int i = 0; i < 16; i++) prev_first16[i] = h[i];
        }
        printf("  Total order changes across 9 comparisons: %d\n", total_order_changes);
        hipFree(d_counter); hipFree(d_results);
    }

    printf("\nDone.\n");
    return 0;
}
