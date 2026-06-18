// noise_minimal_diag.hip — Absolute minimal diagnostic for clock/hwreg sources
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/noise_minimal_diag scripts/noise_minimal_diag.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/noise_minimal_diag

#include <hip/hip_runtime.h>
#include <cstdio>

#define HIP_CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        printf("HIP ERROR: %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

// Kernel 1: Just write known pattern to verify GPU execution
__global__ void k_canary(uint32_t* out) {
    out[threadIdx.x] = 0xCAFEBEEF + threadIdx.x;
}

// Kernel 2: clock64() test — one wavefront, write sequentially
__global__ void k_clock64(uint64_t* out) {
    if (threadIdx.x != 0) return; // only lane 0
    for (int i = 0; i < 10; i++) {
        out[i] = clock64();
    }
}

// Kernel 3: wall_clock64() test
__global__ void k_wallclock(uint64_t* out) {
    if (threadIdx.x != 0) return;
    for (int i = 0; i < 10; i++) {
        out[i] = wall_clock64();
    }
}

// Kernel 4: wall_clock64 DELTAS (consecutive reads)
__global__ void k_wallclock_deltas(uint64_t* out) {
    if (threadIdx.x != 0) return;
    uint64_t prev = wall_clock64();
    for (int i = 0; i < 20; i++) {
        uint64_t now = wall_clock64();
        out[i] = now - prev;
        prev = now;
    }
}

// Kernel 5: s_getreg for specific known-working registers
// Use VGPR output directly, avoid SGPR complications
__global__ void k_hwregs(uint32_t* out) {
    if (threadIdx.x != 0) return;

    // hwreg(2) = STATUS — known to return non-zero
    uint32_t status;
    asm volatile("s_getreg_b32 %0, hwreg(2)" : "=s"(status));
    out[0] = status;

    // hwreg(4) = HW_ID1 on GFX11 — wave placement
    uint32_t hw_id1;
    asm volatile("s_getreg_b32 %0, hwreg(4)" : "=s"(hw_id1));
    out[1] = hw_id1;

    // hwreg(5) = HW_ID2
    uint32_t hw_id2;
    asm volatile("s_getreg_b32 %0, hwreg(5)" : "=s"(hw_id2));
    out[2] = hw_id2;

    // hwreg(29) = SHADER_CYCLES
    uint32_t cycles1;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(cycles1));
    out[3] = cycles1;

    // Do some work
    float d = 1.0f;
    for (int i = 0; i < 1000; i++) d *= 1.00001f;
    if (d < 0) out[0] = 0; // prevent opt

    // Read SHADER_CYCLES again — should be different
    uint32_t cycles2;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(cycles2));
    out[4] = cycles2;
    out[5] = cycles2 - cycles1; // delta

    // hwreg(27) = PERF_SNAPSHOT (what we think)
    uint32_t perf;
    asm volatile("s_getreg_b32 %0, hwreg(27)" : "=s"(perf));
    out[6] = perf;

    // Read again
    asm volatile("s_getreg_b32 %0, hwreg(27)" : "=s"(perf));
    out[7] = perf;
}

// Kernel 6: Atomic contention — verify atomics work
__global__ void k_atomic(unsigned int* counter, unsigned int* results) {
    unsigned int old = atomicAdd(counter, 1u);
    results[threadIdx.x + blockIdx.x * blockDim.x] = old;
}

// Kernel 7: LDS race — intentional race condition
__global__ void k_lds_race(uint32_t* out) {
    __shared__ volatile uint32_t target;
    // NO syncthreads — all 64 threads race to write
    target = threadIdx.x;
    // Tiny delay varies by thread
    float d = 1.0f;
    for (int i = 0; i < (int)threadIdx.x; i++) d *= 1.001f;
    if (d < 0) target = 999;
    // Only thread 0 reads result
    __syncthreads();
    if (threadIdx.x == 0) {
        out[blockIdx.x] = target;
    }
}

// Kernel 8: Memory timing — measure global load latency variance
__global__ void k_mem_timing(uint64_t* out, float* data, int data_size) {
    if (threadIdx.x != 0) return;
    uint32_t seed = 12345;
    for (int i = 0; i < 20; i++) {
        seed = seed * 1103515245 + 12345;
        int idx = (seed >> 8) % data_size;
        uint64_t t0 = clock64();
        float v = data[idx];
        uint64_t t1 = clock64();
        out[i] = t1 - t0;
        if (v < -1e30f) out[0] = 0; // prevent opt
    }
}

int main() {
    printf("=== MINIMAL NOISE SOURCE DIAGNOSTIC ===\n\n");

    // Check GPU
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("GPU: %s, arch: %s\n\n", prop.name, prop.gcnArchName);

    // ---- Test 0: Canary ----
    printf("--- TEST 0: GPU canary (verify execution) ---\n");
    {
        uint32_t *d_out, h_out[64];
        HIP_CHECK(hipMalloc(&d_out, 64 * sizeof(uint32_t)));
        HIP_CHECK(hipMemset(d_out, 0, 64 * sizeof(uint32_t)));
        k_canary<<<1, 64>>>(d_out);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_out, d_out, 64 * sizeof(uint32_t), hipMemcpyDeviceToHost));
        bool ok = true;
        for (int i = 0; i < 8; i++) {
            printf("  [%d] = 0x%08X (expect 0x%08X) %s\n", i, h_out[i], 0xCAFEBEEF + i,
                   h_out[i] == (uint32_t)(0xCAFEBEEF + i) ? "OK" : "FAIL!");
            if (h_out[i] != (uint32_t)(0xCAFEBEEF + i)) ok = false;
        }
        printf("  GPU execution: %s\n\n", ok ? "WORKING" : "BROKEN!");
        hipFree(d_out);
        if (!ok) { printf("GPU not executing kernels! Aborting.\n"); return 1; }
    }

    // ---- Test 1: clock64 ----
    printf("--- TEST 1: clock64() sequential reads ---\n");
    {
        uint64_t *d_out, h_out[10];
        HIP_CHECK(hipMalloc(&d_out, 10 * sizeof(uint64_t)));
        HIP_CHECK(hipMemset(d_out, 0, 10 * sizeof(uint64_t)));
        k_clock64<<<1, 64>>>(d_out);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_out, d_out, 10 * sizeof(uint64_t), hipMemcpyDeviceToHost));
        for (int i = 0; i < 10; i++) {
            printf("  [%d] = %lu", i, h_out[i]);
            if (i > 0) printf("  delta=%ld", (long)(h_out[i] - h_out[i-1]));
            printf("\n");
        }
        hipFree(d_out);
    }

    // ---- Test 2: wall_clock64 ----
    printf("\n--- TEST 2: wall_clock64() sequential reads ---\n");
    {
        uint64_t *d_out, h_out[10];
        HIP_CHECK(hipMalloc(&d_out, 10 * sizeof(uint64_t)));
        HIP_CHECK(hipMemset(d_out, 0, 10 * sizeof(uint64_t)));
        k_wallclock<<<1, 64>>>(d_out);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_out, d_out, 10 * sizeof(uint64_t), hipMemcpyDeviceToHost));
        for (int i = 0; i < 10; i++) {
            printf("  [%d] = %lu", i, h_out[i]);
            if (i > 0) printf("  delta=%ld", (long)(h_out[i] - h_out[i-1]));
            printf("\n");
        }
        hipFree(d_out);
    }

    // ---- Test 3: wall_clock64 deltas ----
    printf("\n--- TEST 3: wall_clock64() consecutive deltas ---\n");
    {
        uint64_t *d_out, h_out[20];
        HIP_CHECK(hipMalloc(&d_out, 20 * sizeof(uint64_t)));
        HIP_CHECK(hipMemset(d_out, 0, 20 * sizeof(uint64_t)));
        k_wallclock_deltas<<<1, 64>>>(d_out);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_out, d_out, 20 * sizeof(uint64_t), hipMemcpyDeviceToHost));
        printf("  Deltas: ");
        for (int i = 0; i < 20; i++) printf("%lu ", h_out[i]);
        printf("\n");
        // Count unique values
        int unique = 0;
        for (int i = 0; i < 20; i++) {
            bool seen = false;
            for (int j = 0; j < i; j++) if (h_out[j] == h_out[i]) { seen = true; break; }
            if (!seen) unique++;
        }
        printf("  Unique delta values: %d/20\n", unique);
        hipFree(d_out);
    }

    // ---- Test 4: hwregs ----
    printf("\n--- TEST 4: Hardware registers (s_getreg_b32) ---\n");
    {
        uint32_t *d_out, h_out[8];
        HIP_CHECK(hipMalloc(&d_out, 8 * sizeof(uint32_t)));
        HIP_CHECK(hipMemset(d_out, 0xAA, 8 * sizeof(uint32_t)));
        k_hwregs<<<1, 64>>>(d_out);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_out, d_out, 8 * sizeof(uint32_t), hipMemcpyDeviceToHost));
        printf("  hwreg(2)  STATUS       = 0x%08X\n", h_out[0]);
        printf("  hwreg(4)  HW_ID1       = 0x%08X\n", h_out[1]);
        printf("  hwreg(5)  HW_ID2       = 0x%08X\n", h_out[2]);
        printf("  hwreg(29) CYCLES_1     = 0x%08X (%u)\n", h_out[3], h_out[3]);
        printf("  hwreg(29) CYCLES_2     = 0x%08X (%u)\n", h_out[4], h_out[4]);
        printf("  hwreg(29) DELTA        = %u cycles\n", h_out[5]);
        printf("  hwreg(27) PERF_SNAP_1  = 0x%08X\n", h_out[6]);
        printf("  hwreg(27) PERF_SNAP_2  = 0x%08X\n", h_out[7]);

        // Run AGAIN for variability
        printf("\n  --- Second run ---\n");
        HIP_CHECK(hipMemset(d_out, 0xBB, 8 * sizeof(uint32_t)));
        k_hwregs<<<1, 64>>>(d_out);
        HIP_CHECK(hipDeviceSynchronize());
        uint32_t h_out2[8];
        HIP_CHECK(hipMemcpy(h_out2, d_out, 8 * sizeof(uint32_t), hipMemcpyDeviceToHost));
        printf("  hwreg(2)  STATUS       = 0x%08X %s\n", h_out2[0], h_out2[0]!=h_out[0]?"CHANGED":"same");
        printf("  hwreg(4)  HW_ID1       = 0x%08X %s\n", h_out2[1], h_out2[1]!=h_out[1]?"CHANGED":"same");
        printf("  hwreg(29) CYCLES_1     = %u %s\n", h_out2[3], h_out2[3]!=h_out[3]?"CHANGED":"same");
        printf("  hwreg(29) DELTA        = %u\n", h_out2[5]);
        printf("  hwreg(27) PERF_SNAP_1  = 0x%08X %s\n", h_out2[6], h_out2[6]!=h_out[6]?"CHANGED":"same");
        hipFree(d_out);
    }

    // ---- Test 5: Atomics ----
    printf("\n--- TEST 5: Atomic contention (2 blocks × 64 threads) ---\n");
    {
        unsigned int *d_counter, *d_results;
        unsigned int h_results[128];
        HIP_CHECK(hipMalloc(&d_counter, sizeof(unsigned int)));
        HIP_CHECK(hipMalloc(&d_results, 128 * sizeof(unsigned int)));
        HIP_CHECK(hipMemset(d_counter, 0, sizeof(unsigned int)));
        HIP_CHECK(hipMemset(d_results, 0xFF, 128 * sizeof(unsigned int)));
        k_atomic<<<2, 64>>>(d_counter, d_results);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_results, d_results, 128 * sizeof(unsigned int), hipMemcpyDeviceToHost));
        printf("  First 16: ");
        for (int i = 0; i < 16; i++) printf("%u ", h_results[i]);
        printf("\n");
        int unique = 0;
        for (int i = 0; i < 128; i++) {
            bool dup = false;
            for (int j = 0; j < i; j++) if (h_results[j] == h_results[i]) { dup = true; break; }
            if (!dup) unique++;
        }
        printf("  Unique: %d/128 (should be 128)\n", unique);

        // Run again — check ORDER changes
        HIP_CHECK(hipMemset(d_counter, 0, sizeof(unsigned int)));
        HIP_CHECK(hipMemset(d_results, 0xFF, 128 * sizeof(unsigned int)));
        k_atomic<<<2, 64>>>(d_counter, d_results);
        HIP_CHECK(hipDeviceSynchronize());
        unsigned int h_results2[128];
        HIP_CHECK(hipMemcpy(h_results2, d_results, 128 * sizeof(unsigned int), hipMemcpyDeviceToHost));
        int order_changed = 0;
        for (int i = 0; i < 128; i++) if (h_results[i] != h_results2[i]) order_changed++;
        printf("  Atomic order changed: %d/128 (>0 = non-deterministic!)\n", order_changed);
        hipFree(d_counter); hipFree(d_results);
    }

    // ---- Test 6: LDS race ----
    printf("\n--- TEST 6: LDS race condition (16 blocks) ---\n");
    {
        uint32_t *d_out, h_out[16];
        HIP_CHECK(hipMalloc(&d_out, 16 * sizeof(uint32_t)));
        HIP_CHECK(hipMemset(d_out, 0xFF, 16 * sizeof(uint32_t)));
        k_lds_race<<<16, 64>>>(d_out);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_out, d_out, 16 * sizeof(uint32_t), hipMemcpyDeviceToHost));
        printf("  Winners: ");
        for (int i = 0; i < 16; i++) printf("%u ", h_out[i]);
        printf("\n");

        // Run again
        HIP_CHECK(hipMemset(d_out, 0xFF, 16 * sizeof(uint32_t)));
        k_lds_race<<<16, 64>>>(d_out);
        HIP_CHECK(hipDeviceSynchronize());
        uint32_t h_out2[16];
        HIP_CHECK(hipMemcpy(h_out2, d_out, 16 * sizeof(uint32_t), hipMemcpyDeviceToHost));
        int changed = 0;
        for (int i = 0; i < 16; i++) if (h_out[i] != h_out2[i]) changed++;
        printf("  Winners changed: %d/16\n", changed);
        hipFree(d_out);
    }

    // ---- Test 7: Memory timing ----
    printf("\n--- TEST 7: Global memory load latency variance ---\n");
    {
        // Allocate 16MB buffer to force L2/DRAM access
        size_t data_sz = 16 * 1024 * 1024;
        float *d_data;
        uint64_t *d_out, h_out[20];
        HIP_CHECK(hipMalloc(&d_data, data_sz));
        HIP_CHECK(hipMalloc(&d_out, 20 * sizeof(uint64_t)));
        HIP_CHECK(hipMemset(d_data, 0, data_sz));
        HIP_CHECK(hipMemset(d_out, 0, 20 * sizeof(uint64_t)));
        k_mem_timing<<<1, 64>>>(d_out, d_data, data_sz / sizeof(float));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_out, d_out, 20 * sizeof(uint64_t), hipMemcpyDeviceToHost));
        printf("  Load latencies (cycles): ");
        for (int i = 0; i < 20; i++) printf("%lu ", h_out[i]);
        printf("\n");
        int unique = 0;
        for (int i = 0; i < 20; i++) {
            bool seen = false;
            for (int j = 0; j < i; j++) if (h_out[j] == h_out[i]) { seen = true; break; }
            if (!seen) unique++;
        }
        printf("  Unique latencies: %d/20\n", unique);
        hipFree(d_data); hipFree(d_out);
    }

    printf("\nDone.\n");
    return 0;
}
