/*
 * Minimal test: which clock intrinsics work on gfx1151 with HSA_OVERRIDE?
 */
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void test_clocks(unsigned long long* out, int n) {
    int tid = threadIdx.x;
    if (tid != 0) return;

    // Method A: clock()
    long long c0 = clock();
    float x = 1.0f;
    for (int i = 0; i < 10000; i++) x = x * 1.0001f + 0.0001f;
    long long c1 = clock();
    out[0] = (unsigned long long)(c1 - c0);
    out[1] = (unsigned long long)c0;
    out[2] = (unsigned long long)c1;

    // Method B: clock64()
    long long c64_0 = clock64();
    x = 1.0f;
    for (int i = 0; i < 10000; i++) x = x * 1.0001f + 0.0001f;
    long long c64_1 = clock64();
    out[3] = (unsigned long long)(c64_1 - c64_0);
    out[4] = (unsigned long long)c64_0;
    out[5] = (unsigned long long)c64_1;

    // Method C: wall_clock64()
    long long w0 = wall_clock64();
    x = 1.0f;
    for (int i = 0; i < 10000; i++) x = x * 1.0001f + 0.0001f;
    long long w1 = wall_clock64();
    out[6] = (unsigned long long)(w1 - w0);
    out[7] = (unsigned long long)w0;
    out[8] = (unsigned long long)w1;

    // Method D: s_getreg HW_REG_SHADER_CYCLES
    unsigned int sc0, sc1;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES)" : "=s"(sc0));
    x = 1.0f;
    for (int i = 0; i < 10000; i++) x = x * 1.0001f + 0.0001f;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES)" : "=s"(sc1));
    out[9] = (unsigned long long)(sc1 - sc0);
    out[10] = (unsigned long long)sc0;
    out[11] = (unsigned long long)sc1;

    // Method E: inline asm s_clock (s_getreg with different encoding)
    unsigned int hw0, hw1;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(hw0));
    x = 1.0f;
    for (int i = 0; i < 10000; i++) x = x * 1.0001f + 0.0001f;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(hw1));
    out[12] = (unsigned long long)(hw1 - hw0);
    out[13] = (unsigned long long)hw0;
    out[14] = (unsigned long long)hw1;

    // Prevent x from being optimized away
    out[15] = *(unsigned long long*)&x;
}

int main() {
    unsigned long long* d_out;
    unsigned long long h_out[16] = {0};

    hipMalloc(&d_out, 16 * sizeof(unsigned long long));
    hipMemset(d_out, 0, 16 * sizeof(unsigned long long));

    hipLaunchKernelGGL(test_clocks, 1, 32, 0, 0, d_out, 16);
    hipError_t err = hipDeviceSynchronize();

    if (err != hipSuccess) {
        fprintf(stderr, "Kernel error: %s\n", hipGetErrorString(err));
        return 1;
    }

    hipMemcpy(h_out, d_out, 16 * sizeof(unsigned long long), hipMemcpyDeviceToHost);

    printf("Method A  clock():     delta=%llu  raw=[%llu, %llu]\n", h_out[0], h_out[1], h_out[2]);
    printf("Method B  clock64():   delta=%llu  raw=[%llu, %llu]\n", h_out[3], h_out[4], h_out[5]);
    printf("Method C  wall_clock64(): delta=%llu  raw=[%llu, %llu]\n", h_out[6], h_out[7], h_out[8]);
    printf("Method D  s_getreg(SHADER_CYCLES): delta=%llu  raw=[%llu, %llu]\n", h_out[9], h_out[10], h_out[11]);
    printf("Method E  s_getreg(29): delta=%llu  raw=[%llu, %llu]\n", h_out[12], h_out[13], h_out[14]);
    printf("Sink:     %llu\n", h_out[15]);

    hipFree(d_out);
    return 0;
}
