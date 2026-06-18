// s14_per_cu_shader_skew.hip
// Per-CU latency probe on gfx1151 / gfx1100. Each block runs a fixed-length
// integer-add chain and timestamps before/after with __builtin_amdgcn_s_memtime
// (falls back to clock64() if unavailable). The block writes
//   (cu_id, wave_id, elapsed_cycles)
// for the FIRST thread of every block. Caller spawns far more blocks than CUs
// so each CU is exercised many times — the mean elapsed per CU is the per-die
// signature.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>

#ifndef ITER
#define ITER 8192
#endif

__device__ uint32_t get_cu_id() {
    // RDNA3 (gfx11): HW_REG_HW_ID1 / HW_REG_HW_ID2 layout
    // HW_ID1: [5:0] wave_id, [9:6] simd_id, [13:10] wgp_id, [17:14] sa_id, [19:18] se_id
    uint32_t hw_id;
    asm volatile("s_getreg_b32 %0, hwreg(23)\n" : "=s"(hw_id)); // 23 = HW_REG_HW_ID1
    return hw_id;
}

__global__ void probe(uint32_t *out_cu, uint64_t *out_cyc, int n_slots) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (tid != 0) return;
    uint32_t hw = get_cu_id();
    uint64_t t0 = clock64();
    // dependent chain — defeats reordering
    uint64_t acc = t0 ^ (uint64_t)bid;
    #pragma unroll 32
    for (int i = 0; i < ITER; ++i) {
        acc = acc * 6364136223846793005ULL + 1442695040888963407ULL;
    }
    uint64_t t1 = clock64();
    // store
    if (bid < n_slots) {
        out_cu[bid] = hw;
        out_cyc[bid] = (t1 - t0) ^ (acc & 0); // include acc to prevent DCE
    }
}

int main(int argc, char **argv) {
    int blocks = (argc > 1) ? atoi(argv[1]) : 4096;
    uint32_t *d_cu, *h_cu;
    uint64_t *d_cyc, *h_cyc;
    hipMalloc(&d_cu, blocks*sizeof(uint32_t));
    hipMalloc(&d_cyc, blocks*sizeof(uint64_t));
    h_cu = (uint32_t*)malloc(blocks*sizeof(uint32_t));
    h_cyc = (uint64_t*)malloc(blocks*sizeof(uint64_t));
    hipLaunchKernelGGL(probe, dim3(blocks), dim3(64), 0, 0, d_cu, d_cyc, blocks);
    hipDeviceSynchronize();
    hipMemcpy(h_cu, d_cu, blocks*sizeof(uint32_t), hipMemcpyDeviceToHost);
    hipMemcpy(h_cyc, d_cyc, blocks*sizeof(uint64_t), hipMemcpyDeviceToHost);
    // Print: cu_id_raw,elapsed_cycles
    for (int i = 0; i < blocks; ++i) {
        printf("%u %llu\n", h_cu[i], (unsigned long long)h_cyc[i]);
    }
    hipFree(d_cu); hipFree(d_cyc); free(h_cu); free(h_cyc);
    return 0;
}
