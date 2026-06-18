/* test_ssleep.hip — ISOLATED test of s_sleep instruction timing */
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void k(uint32_t* r) {
    uint32_t tid = threadIdx.x;
    if (tid >= 64 || blockIdx.x != 0) return;
    uint32_t t0, t1;

    // s_nop
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t0));
    asm volatile("s_nop 0");
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t1));
    r[tid*5+0] = t1 - t0;

    // s_sleep 1
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t0));
    asm volatile("s_sleep 1");
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t1));
    r[tid*5+1] = t1 - t0;

    // s_sleep 4
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t0));
    asm volatile("s_sleep 4");
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t1));
    r[tid*5+2] = t1 - t0;

    // s_sleep 15
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t0));
    asm volatile("s_sleep 15");
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t1));
    r[tid*5+3] = t1 - t0;

    // s_sleep 63 (max)
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t0));
    asm volatile("s_sleep 63");
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t1));
    r[tid*5+4] = t1 - t0;
}
int main() {
    uint32_t *d, h[320];
    hipMalloc(&d, 320*sizeof(uint32_t));
    hipMemset(d, 0, 320*sizeof(uint32_t));
    k<<<1,64>>>(d);
    hipError_t e = hipDeviceSynchronize();
    hipMemcpy(h, d, 320*sizeof(uint32_t), hipMemcpyDeviceToHost);
    if (e != hipSuccess) {
        printf("FAULTED: %s\n", hipGetErrorString(e));
    } else {
        printf("Thread 0:\n");
        printf("  s_nop 0:    %u cycles\n", h[0]);
        printf("  s_sleep 1:  %u cycles\n", h[1]);
        printf("  s_sleep 4:  %u cycles\n", h[2]);
        printf("  s_sleep 15: %u cycles\n", h[3]);
        printf("  s_sleep 63: %u cycles\n", h[4]);

        // Stats across 64 threads
        const char* names[] = {"s_nop", "s_sleep1", "s_sleep4", "s_sleep15", "s_sleep63"};
        for (int s = 0; s < 5; s++) {
            uint32_t mn=0xFFFFFFFF, mx=0, sum=0;
            for (int i = 0; i < 64; i++) {
                uint32_t v = h[i*5+s];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
                sum += v;
            }
            printf("  %-10s: min=%u max=%u avg=%.1f\n", names[s], mn, mx, sum/64.0);
        }

        int scales = (h[2] > h[1]) && (h[3] > h[2]) && (h[4] > h[3]);
        printf("\nSleep scales with argument: %s\n", scales ? "YES" : "no");
    }
    hipFree(d);
}
