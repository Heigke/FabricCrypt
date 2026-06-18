// H7 in-kernel ANALOG self-sensing on RDNA (gfx1151). A wavefront measures its OWN effective shader
// clock from inside the silicon: it runs FMA work for a fixed wall-clock window (s_memrealtime, a
// constant-rate on-die counter) and counts how many iterations it completed. iters/window ∝ realized
// shader clock × IPC -> directly senses voltage-droop / clock-stretch. Under combined load the GPU
// rail droops (power cap binds) -> the self-sensed rate is a SUB-ADDITIVE (nonlinear) function of load.
//
// Build: hipcc -O3 gpu_selfsense.hip -o gpu_selfsense
// Run:   ./gpu_selfsense <n_workgroups> <window_ticks>   -> prints: total_iters realized_rate
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

__global__ void selfsense(unsigned long long *out, unsigned long long window) {
    unsigned long long t0 = wall_clock64();
    double x = 1.0001 + (threadIdx.x & 31) * 1e-4;
    unsigned long long iters = 0;
    // loop until `window` real-time ticks elapse; inner block keeps the FP pipes busy (throughput)
    while (wall_clock64() - t0 < window) {
        #pragma unroll
        for (int i = 0; i < 256; i++) { x = x * 1.0000001 + 0.1; }
        iters++;
    }
    if (x == 1234.5678) out[1] = (unsigned long long)x;   // keep x live
    atomicAdd(out, iters);
}

int main(int argc, char **argv) {
    int wg = argc > 1 ? atoi(argv[1]) : 64;
    unsigned long long window = argc > 2 ? strtoull(argv[2], 0, 10) : 2000000ULL; // ~ticks
    unsigned long long *d_out, h_out[2] = {0, 0};
    hipMalloc(&d_out, 2 * sizeof(unsigned long long));
    hipMemcpy(d_out, h_out, 2 * sizeof(unsigned long long), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(selfsense, dim3(wg), dim3(256), 0, 0, d_out, window);
    hipDeviceSynchronize();
    hipMemcpy(h_out, d_out, 2 * sizeof(unsigned long long), hipMemcpyDeviceToHost);
    // realized_rate = total iterations across all threads, normalized by window -> ∝ effective clock
    double rate = (double)h_out[0] / (double)window;
    printf("%llu %.6f\n", h_out[0], rate);
    hipFree(d_out);
    return 0;
}
