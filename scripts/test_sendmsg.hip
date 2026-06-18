/* test_sendmsg.hip — ISOLATED test of s_sendmsg instruction */
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void k(uint32_t* r) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t t0, t1, s_before, s_after, ts_before, ts_after;
    asm volatile("s_getreg_b32 %0, hwreg(2)" : "=s"(s_before));
    asm volatile("s_getreg_b32 %0, hwreg(3)" : "=s"(ts_before));
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t0));
    asm volatile("s_sendmsg 0x001");  // MSG_INTERRUPT
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t1));
    asm volatile("s_getreg_b32 %0, hwreg(2)" : "=s"(s_after));
    asm volatile("s_getreg_b32 %0, hwreg(3)" : "=s"(ts_after));
    r[0] = s_before; r[1] = s_after;
    r[2] = ts_before; r[3] = ts_after;
    r[4] = t1 - t0;
    r[5] = 0xCAFE;
}
int main() {
    uint32_t *d, h[8];
    hipMalloc(&d, 8*sizeof(uint32_t));
    hipMemset(d, 0, 8*sizeof(uint32_t));
    k<<<1,1>>>(d);
    hipError_t e = hipDeviceSynchronize();
    hipMemcpy(h, d, 8*sizeof(uint32_t), hipMemcpyDeviceToHost);
    if (e != hipSuccess) {
        printf("FAULTED: %s\n", hipGetErrorString(e));
    } else {
        printf("STATUS:  before=0x%08x after=0x%08x delta=0x%08x\n", h[0], h[1], h[0]^h[1]);
        printf("TRAPSTS: before=0x%08x after=0x%08x delta=0x%08x\n", h[2], h[3], h[2]^h[3]);
        printf("s_sendmsg cycles: %u\n", h[4]);
        printf("Alive: %s\n", h[5]==0xCAFE ? "YES" : "NO");
    }
    hipFree(d);
}
