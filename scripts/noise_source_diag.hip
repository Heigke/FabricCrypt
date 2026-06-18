// noise_source_diag.hip — Minimal diagnostic: what raw values do we get from each noise source?
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/noise_source_diag scripts/noise_source_diag.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/noise_source_diag

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define N_SAMPLES 16

// Test 1: wall_clock64 raw values
__global__ void test_wall_clock(uint64_t* out) {
    int tid = threadIdx.x;
    if (tid >= N_SAMPLES) return;
    uint64_t w0 = wall_clock64();
    // Small delay
    float dummy = 1.0f;
    for (int i = 0; i < 100; i++) dummy = dummy * 1.00001f + 0.00001f;
    uint64_t w1 = wall_clock64();
    out[tid * 3 + 0] = w0;
    out[tid * 3 + 1] = w1;
    out[tid * 3 + 2] = w1 - w0;
    if (dummy < 0) out[0] = 0; // prevent opt
}

// Test 2: hwreg(27) PERF_SNAPSHOT
__global__ void test_perf_snapshot(uint32_t* out) {
    int tid = threadIdx.x;
    if (tid >= N_SAMPLES) return;
    uint32_t v;
    asm volatile("s_getreg_b32 %0, hwreg(27)" : "=s"(v));
    out[tid] = v;
}

// Test 3: hwreg(29) SHADER_CYCLES
__global__ void test_shader_cycles(uint32_t* out) {
    int tid = threadIdx.x;
    if (tid >= N_SAMPLES) return;
    uint32_t v;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(v));
    out[tid] = v;
}

// Test 4: Probe ALL hwregs 0-31
__global__ void test_all_hwregs(uint32_t* out) {
    int tid = threadIdx.x;
    if (tid >= 32) return;
    uint32_t v = 0xDEADBEEF;
    switch(tid) {
        case 0:  asm volatile("s_getreg_b32 %0, hwreg(0)"  : "=s"(v)); break;
        case 1:  asm volatile("s_getreg_b32 %0, hwreg(1)"  : "=s"(v)); break;
        case 2:  asm volatile("s_getreg_b32 %0, hwreg(2)"  : "=s"(v)); break;
        case 3:  asm volatile("s_getreg_b32 %0, hwreg(3)"  : "=s"(v)); break;
        case 4:  asm volatile("s_getreg_b32 %0, hwreg(4)"  : "=s"(v)); break;
        case 5:  asm volatile("s_getreg_b32 %0, hwreg(5)"  : "=s"(v)); break;
        case 6:  asm volatile("s_getreg_b32 %0, hwreg(6)"  : "=s"(v)); break;
        case 7:  asm volatile("s_getreg_b32 %0, hwreg(7)"  : "=s"(v)); break;
        case 8:  asm volatile("s_getreg_b32 %0, hwreg(8)"  : "=s"(v)); break;
        case 9:  asm volatile("s_getreg_b32 %0, hwreg(9)"  : "=s"(v)); break;
        case 10: asm volatile("s_getreg_b32 %0, hwreg(10)" : "=s"(v)); break;
        case 11: asm volatile("s_getreg_b32 %0, hwreg(11)" : "=s"(v)); break;
        case 12: asm volatile("s_getreg_b32 %0, hwreg(12)" : "=s"(v)); break;
        case 13: asm volatile("s_getreg_b32 %0, hwreg(13)" : "=s"(v)); break;
        case 14: asm volatile("s_getreg_b32 %0, hwreg(14)" : "=s"(v)); break;
        case 15: asm volatile("s_getreg_b32 %0, hwreg(15)" : "=s"(v)); break;
        case 16: asm volatile("s_getreg_b32 %0, hwreg(16)" : "=s"(v)); break;
        case 17: asm volatile("s_getreg_b32 %0, hwreg(17)" : "=s"(v)); break;
        case 18: asm volatile("s_getreg_b32 %0, hwreg(18)" : "=s"(v)); break;
        case 19: asm volatile("s_getreg_b32 %0, hwreg(19)" : "=s"(v)); break;
        case 20: asm volatile("s_getreg_b32 %0, hwreg(20)" : "=s"(v)); break;
        case 21: asm volatile("s_getreg_b32 %0, hwreg(21)" : "=s"(v)); break;
        case 22: asm volatile("s_getreg_b32 %0, hwreg(22)" : "=s"(v)); break;
        case 23: asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(v)); break;
        case 24: asm volatile("s_getreg_b32 %0, hwreg(24)" : "=s"(v)); break;
        case 25: asm volatile("s_getreg_b32 %0, hwreg(25)" : "=s"(v)); break;
        case 26: asm volatile("s_getreg_b32 %0, hwreg(26)" : "=s"(v)); break;
        case 27: asm volatile("s_getreg_b32 %0, hwreg(27)" : "=s"(v)); break;
        case 28: asm volatile("s_getreg_b32 %0, hwreg(28)" : "=s"(v)); break;
        case 29: asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(v)); break;
        case 30: asm volatile("s_getreg_b32 %0, hwreg(30)" : "=s"(v)); break;
        case 31: asm volatile("s_getreg_b32 %0, hwreg(31)" : "=s"(v)); break;
    }
    out[tid] = v;
}

// Test 5: Atomic contention — do atomics actually work?
__global__ void test_atomics(unsigned int* counter, unsigned int* results) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int old = atomicAdd(counter, 1u);
    results[tid] = old;
}

// Test 6: LDS race condition — unsynchronized writes
__global__ void test_lds_race(uint32_t* out) {
    __shared__ uint32_t race_target;
    int tid = threadIdx.x;
    // NO syncthreads — intentional race
    race_target = tid; // all 64 threads write simultaneously
    // Small delay
    float d = 1.0f;
    for (int i = 0; i < 10; i++) d *= 1.001f;
    if (d < 0) race_target = 999;
    // Only thread 0 reads the result
    if (tid == 0) {
        out[blockIdx.x] = race_target; // who won?
    }
}

// Test 7: Uninitialized LDS — SRAM PUF attempt
__global__ void test_uninitialized_lds(uint32_t* out) {
    // Read shared memory WITHOUT writing first
    extern __shared__ uint32_t lds_raw[];
    int tid = threadIdx.x;
    // Read before any write — SRAM cell power-on state
    out[blockIdx.x * 64 + tid] = lds_raw[tid];
}

// Test 8: clock64() vs wall_clock64()
__global__ void test_clock_variants(uint64_t* out) {
    int tid = threadIdx.x;
    if (tid >= 4) return;

    uint64_t c0 = clock64();
    uint64_t w0 = wall_clock64();
    // delay
    float d = 1.0f;
    for (int i = 0; i < 1000; i++) d = d * 1.00001f + 0.00001f;
    uint64_t c1 = clock64();
    uint64_t w1 = wall_clock64();
    if (d < 0) out[0] = 0;

    out[tid * 4 + 0] = c0;
    out[tid * 4 + 1] = c1;
    out[tid * 4 + 2] = w0;
    out[tid * 4 + 3] = w1;
}

int main() {
    printf("=== NOISE SOURCE DIAGNOSTIC — RAW VALUES ===\n\n");

    // ---- Test 1: wall_clock64 ----
    printf("--- TEST 1: wall_clock64() raw values (16 threads) ---\n");
    {
        uint64_t* d_out;
        hipMalloc(&d_out, N_SAMPLES * 3 * sizeof(uint64_t));
        test_wall_clock<<<1, N_SAMPLES>>>(d_out);
        hipDeviceSynchronize();
        uint64_t h_out[N_SAMPLES * 3];
        hipMemcpy(h_out, d_out, N_SAMPLES * 3 * sizeof(uint64_t), hipMemcpyDeviceToHost);
        for (int i = 0; i < N_SAMPLES; i++) {
            printf("  tid=%2d: w0=%lu  w1=%lu  delta=%lu\n", i, h_out[i*3], h_out[i*3+1], h_out[i*3+2]);
        }
        hipFree(d_out);
    }

    // ---- Test 2: PERF_SNAPSHOT ----
    printf("\n--- TEST 2: hwreg(27) PERF_SNAPSHOT (16 threads) ---\n");
    {
        uint32_t* d_out;
        hipMalloc(&d_out, N_SAMPLES * sizeof(uint32_t));
        test_perf_snapshot<<<1, N_SAMPLES>>>(d_out);
        hipDeviceSynchronize();
        uint32_t h_out[N_SAMPLES];
        hipMemcpy(h_out, d_out, N_SAMPLES * sizeof(uint32_t), hipMemcpyDeviceToHost);
        for (int i = 0; i < N_SAMPLES; i++) {
            printf("  tid=%2d: 0x%08X (%u)\n", i, h_out[i], h_out[i]);
        }
        hipFree(d_out);
    }

    // ---- Test 3: SHADER_CYCLES ----
    printf("\n--- TEST 3: hwreg(29) SHADER_CYCLES (16 threads) ---\n");
    {
        uint32_t* d_out;
        hipMalloc(&d_out, N_SAMPLES * sizeof(uint32_t));
        test_shader_cycles<<<1, N_SAMPLES>>>(d_out);
        hipDeviceSynchronize();
        uint32_t h_out[N_SAMPLES];
        hipMemcpy(h_out, d_out, N_SAMPLES * sizeof(uint32_t), hipMemcpyDeviceToHost);
        for (int i = 0; i < N_SAMPLES; i++) {
            printf("  tid=%2d: 0x%08X (%u)\n", i, h_out[i], h_out[i]);
        }
        hipFree(d_out);
    }

    // ---- Test 4: ALL hwregs 0-31 ----
    printf("\n--- TEST 4: ALL hwregs 0-31 ---\n");
    {
        uint32_t* d_out;
        hipMalloc(&d_out, 32 * sizeof(uint32_t));
        hipMemset(d_out, 0xFF, 32 * sizeof(uint32_t));
        test_all_hwregs<<<1, 32>>>(d_out);
        hipDeviceSynchronize();
        uint32_t h_out[32];
        hipMemcpy(h_out, d_out, 32 * sizeof(uint32_t), hipMemcpyDeviceToHost);
        for (int i = 0; i < 32; i++) {
            const char* name = "???";
            switch(i) {
                case 1: name = "MODE"; break;
                case 2: name = "STATUS"; break;
                case 3: name = "TRAPSTS"; break;
                case 4: name = "HW_ID1"; break;
                case 5: name = "HW_ID2"; break;
                case 6: name = "GPR_ALLOC"; break;
                case 7: name = "LDS_ALLOC"; break;
                case 8: name = "IB_STS"; break;
                case 15: name = "SH_MEM_BASES"; break;
                case 16: name = "FLAT_SCR_LO"; break;
                case 17: name = "FLAT_SCR_HI"; break;
                case 23: name = "HW_ID1_v2"; break;
                case 24: name = "HW_ID2_v2"; break;
                case 29: name = "SHADER_CYCLES"; break;
            }
            if (h_out[i] != 0 && h_out[i] != 0xFFFFFFFF) {
                printf("  hwreg(%2d) %-15s = 0x%08X (%u)  <-- NON-ZERO!\n", i, name, h_out[i], h_out[i]);
            } else {
                printf("  hwreg(%2d) %-15s = 0x%08X\n", i, name, h_out[i]);
            }
        }
        // Run AGAIN to see if values change between invocations
        printf("\n  --- Second invocation (check for variability) ---\n");
        hipMemset(d_out, 0xFF, 32 * sizeof(uint32_t));
        test_all_hwregs<<<1, 32>>>(d_out);
        hipDeviceSynchronize();
        uint32_t h_out2[32];
        hipMemcpy(h_out2, d_out, 32 * sizeof(uint32_t), hipMemcpyDeviceToHost);
        for (int i = 0; i < 32; i++) {
            if (h_out2[i] != h_out[i]) {
                printf("  hwreg(%2d) CHANGED: 0x%08X -> 0x%08X  *** DYNAMIC! ***\n", i, h_out[i], h_out2[i]);
            }
        }
        // Third run from different block
        printf("  --- Third invocation (blockIdx=1) ---\n");
        hipMemset(d_out, 0xFF, 32 * sizeof(uint32_t));
        test_all_hwregs<<<2, 32>>>(d_out);
        hipDeviceSynchronize();
        uint32_t h_out3[32];
        hipMemcpy(h_out3, d_out, 32 * sizeof(uint32_t), hipMemcpyDeviceToHost);
        for (int i = 0; i < 32; i++) {
            if (h_out3[i] != h_out[i]) {
                printf("  hwreg(%2d) BLOCK-DEPENDENT: block0=0x%08X block1=0x%08X\n", i, h_out[i], h_out3[i]);
            }
        }
        hipFree(d_out);
    }

    // ---- Test 5: Atomics ----
    printf("\n--- TEST 5: Atomic contention (4 blocks × 64 threads) ---\n");
    {
        unsigned int *d_counter, *d_results;
        hipMalloc(&d_counter, sizeof(unsigned int));
        hipMalloc(&d_results, 256 * sizeof(unsigned int));
        hipMemset(d_counter, 0, sizeof(unsigned int));
        test_atomics<<<4, 64>>>(d_counter, d_results);
        hipDeviceSynchronize();
        unsigned int h_results[256];
        hipMemcpy(h_results, d_results, 256 * sizeof(unsigned int), hipMemcpyDeviceToHost);
        // Show first 16 results to verify atomics work
        printf("  First 16 atomic return values: ");
        for (int i = 0; i < 16; i++) printf("%u ", h_results[i]);
        printf("\n");
        // Check uniqueness
        int unique = 0;
        for (int i = 0; i < 256; i++) {
            bool dup = false;
            for (int j = 0; j < i; j++) if (h_results[j] == h_results[i]) { dup = true; break; }
            if (!dup) unique++;
        }
        printf("  Unique values: %d/256 (should be 256 if atomics work)\n", unique);
        hipFree(d_counter);
        hipFree(d_results);
    }

    // ---- Test 6: LDS race ----
    printf("\n--- TEST 6: LDS race condition (32 blocks, who wins?) ---\n");
    {
        uint32_t* d_out;
        hipMalloc(&d_out, 32 * sizeof(uint32_t));
        test_lds_race<<<32, 64>>>(d_out);
        hipDeviceSynchronize();
        uint32_t h_out[32];
        hipMemcpy(h_out, d_out, 32 * sizeof(uint32_t), hipMemcpyDeviceToHost);
        printf("  Winners: ");
        for (int i = 0; i < 32; i++) printf("%u ", h_out[i]);
        printf("\n");
        int n_unique = 0;
        for (int i = 0; i < 32; i++) {
            bool seen = false;
            for (int j = 0; j < i; j++) if (h_out[j] == h_out[i]) { seen = true; break; }
            if (!seen) n_unique++;
        }
        printf("  Unique winners: %d/32\n", n_unique);
        // Run again to see if results change
        test_lds_race<<<32, 64>>>(d_out);
        hipDeviceSynchronize();
        uint32_t h_out2[32];
        hipMemcpy(h_out2, d_out, 32 * sizeof(uint32_t), hipMemcpyDeviceToHost);
        int changed = 0;
        for (int i = 0; i < 32; i++) if (h_out[i] != h_out2[i]) changed++;
        printf("  Changed between runs: %d/32\n", changed);
        hipFree(d_out);
    }

    // ---- Test 7: Uninitialized LDS ----
    printf("\n--- TEST 7: Uninitialized LDS (SRAM PUF, 8 blocks × 64) ---\n");
    {
        uint32_t* d_out;
        hipMalloc(&d_out, 8 * 64 * sizeof(uint32_t));
        hipMemset(d_out, 0xAA, 8 * 64 * sizeof(uint32_t)); // sentinel
        test_uninitialized_lds<<<8, 64, 64*sizeof(uint32_t)>>>(d_out);
        hipDeviceSynchronize();
        uint32_t h_out[8 * 64];
        hipMemcpy(h_out, d_out, 8 * 64 * sizeof(uint32_t), hipMemcpyDeviceToHost);
        // Show first 8 values from each block
        for (int b = 0; b < 8; b++) {
            printf("  block %d: ", b);
            for (int t = 0; t < 8; t++) printf("0x%08X ", h_out[b*64+t]);
            printf("...\n");
        }
        // Count non-zero
        int nonzero = 0, unique_vals = 0;
        for (int i = 0; i < 8*64; i++) {
            if (h_out[i] != 0 && h_out[i] != 0xAAAAAAAA) nonzero++;
        }
        printf("  Non-zero/non-sentinel: %d/512\n", nonzero);
        // Run again — does it change?
        test_uninitialized_lds<<<8, 64, 64*sizeof(uint32_t)>>>(d_out);
        hipDeviceSynchronize();
        uint32_t h_out2[8 * 64];
        hipMemcpy(h_out2, d_out, 8 * 64 * sizeof(uint32_t), hipMemcpyDeviceToHost);
        int changed = 0;
        for (int i = 0; i < 8*64; i++) if (h_out[i] != h_out2[i]) changed++;
        printf("  Changed between runs: %d/512\n", changed);
        hipFree(d_out);
    }

    // ---- Test 8: clock64 vs wall_clock64 ----
    printf("\n--- TEST 8: clock64() vs wall_clock64() ---\n");
    {
        uint64_t* d_out;
        hipMalloc(&d_out, 4 * 4 * sizeof(uint64_t));
        test_clock_variants<<<1, 4>>>(d_out);
        hipDeviceSynchronize();
        uint64_t h_out[16];
        hipMemcpy(h_out, d_out, 16 * sizeof(uint64_t), hipMemcpyDeviceToHost);
        for (int i = 0; i < 4; i++) {
            printf("  tid=%d: clock64=[%lu,%lu] delta=%lu  wall=[%lu,%lu] delta=%lu\n",
                   i, h_out[i*4], h_out[i*4+1], h_out[i*4+1]-h_out[i*4],
                   h_out[i*4+2], h_out[i*4+3], h_out[i*4+3]-h_out[i*4+2]);
        }
        // Run twice more to check variability
        printf("  --- Run 2 ---\n");
        test_clock_variants<<<1, 4>>>(d_out);
        hipDeviceSynchronize();
        hipMemcpy(h_out, d_out, 16 * sizeof(uint64_t), hipMemcpyDeviceToHost);
        for (int i = 0; i < 2; i++) {
            printf("  tid=%d: clock_delta=%lu  wall_delta=%lu\n",
                   i, h_out[i*4+1]-h_out[i*4], h_out[i*4+3]-h_out[i*4+2]);
        }
        hipFree(d_out);
    }

    printf("\nDone.\n");
    return 0;
}
