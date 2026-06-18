/*  B_lorenz.hip — Per-CU Lorenz trajectory fingerprint for gfx1151
 *
 *  Angle B of IDENTITY_BENCHMARK_2026-05-30 novel_v2.
 *
 *  Each wave runs a Lorenz attractor (σ=10, ρ=28, β=8/3) RK4 integrator
 *  for N_STEPS steps with fixed initial conditions. The per-wave (x,y,z)
 *  tail (last TAIL_LEN states) is written out. CU placement is read from
 *  HW_ID so the host can group results by physical CU.
 *
 *  Output layout per wave w in [0, N_WAVES):
 *    tail[w*TAIL_LEN*3 + 3*i + {0,1,2}] = x,y,z at step (N_STEPS-TAIL_LEN+i)
 *    hwids[w]   = HW_ID raw (#23)
 *    cycles[w]  = SHADER_CYCLES delta
 *
 *  Build:
 *    hipcc --offload-arch=gfx1151 -O1 -o B_lorenz B_lorenz.hip
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define N_WAVES   1024
#define WAVE_SIZE 32
#define N_STEPS   10000
#define TAIL_LEN  256

__device__ inline void lorenz_rhs(float x, float y, float z,
                                  float& dx, float& dy, float& dz)
{
    const float sigma = 10.0f, rho = 28.0f, beta = 8.0f/3.0f;
    dx = sigma * (y - x);
    dy = x * (rho - z) - y;
    dz = x * y - beta * z;
}

__global__ void lorenz_kernel(float* tail,
                              uint32_t* hwids,
                              uint32_t* cycles)
{
    int lane = threadIdx.x & (WAVE_SIZE - 1);
    int wave_in_block = threadIdx.x / WAVE_SIZE;
    int waves_per_block = blockDim.x / WAVE_SIZE;
    int wave_id = blockIdx.x * waves_per_block + wave_in_block;
    if (wave_id >= N_WAVES) return;

    // Each lane gets a slightly different but deterministic IC so the
    // intra-wave divergence is bounded and FP ordering matters.
    float x = 1.0f + 0.001f * (float)lane;
    float y = 1.0f - 0.0007f * (float)lane;
    float z = 1.0f + 0.0003f * (float)((lane * 17) & 31);
    const float dt = 0.01f;

    // Only lane 0 reads HW_ID / cycles
    if (lane == 0) {
        hwids[wave_id]  = __builtin_amdgcn_s_getreg((23) | (0 << 6) | ((32-1) << 11));
        cycles[wave_id] = __builtin_amdgcn_s_getreg((29) | (0 << 6) | ((32-1) << 11));
    }

    // RK4 integration
    for (int step = 0; step < N_STEPS; step++) {
        float k1x,k1y,k1z, k2x,k2y,k2z, k3x,k3y,k3z, k4x,k4y,k4z;
        lorenz_rhs(x, y, z, k1x,k1y,k1z);
        lorenz_rhs(x + 0.5f*dt*k1x, y + 0.5f*dt*k1y, z + 0.5f*dt*k1z, k2x,k2y,k2z);
        lorenz_rhs(x + 0.5f*dt*k2x, y + 0.5f*dt*k2y, z + 0.5f*dt*k2z, k3x,k3y,k3z);
        lorenz_rhs(x + dt*k3x,      y + dt*k3y,      z + dt*k3z,      k4x,k4y,k4z);
        x += (dt/6.0f)*(k1x + 2.0f*k2x + 2.0f*k3x + k4x);
        y += (dt/6.0f)*(k1y + 2.0f*k2y + 2.0f*k3y + k4y);
        z += (dt/6.0f)*(k1z + 2.0f*k2z + 2.0f*k3z + k4z);

        // Write tail (only lane 0 writes the wave's representative state)
        int tail_step = step - (N_STEPS - TAIL_LEN);
        if (tail_step >= 0 && lane == 0) {
            int idx = wave_id * TAIL_LEN * 3 + tail_step * 3;
            tail[idx + 0] = x;
            tail[idx + 1] = y;
            tail[idx + 2] = z;
        }
    }

    if (lane == 0) {
        uint32_t cyc1 = __builtin_amdgcn_s_getreg((29) | (0 << 6) | ((32-1) << 11));
        cycles[wave_id] = cyc1 - cycles[wave_id];
    }
}

int main(int argc, char** argv) {
    int batch_id = (argc > 1) ? atoi(argv[1]) : 0;

    size_t tail_bytes = (size_t)N_WAVES * TAIL_LEN * 3 * sizeof(float);
    size_t id_bytes   = (size_t)N_WAVES * sizeof(uint32_t);

    float*    d_tail = nullptr;
    uint32_t* d_hwids = nullptr;
    uint32_t* d_cycles = nullptr;
    HIP_CHECK(hipMalloc(&d_tail,   tail_bytes));
    HIP_CHECK(hipMalloc(&d_hwids,  id_bytes));
    HIP_CHECK(hipMalloc(&d_cycles, id_bytes));
    HIP_CHECK(hipMemset(d_tail,   0, tail_bytes));
    HIP_CHECK(hipMemset(d_hwids,  0, id_bytes));
    HIP_CHECK(hipMemset(d_cycles, 0, id_bytes));

    int threads_per_block = 256; // 8 waves per block
    int blocks = (N_WAVES * WAVE_SIZE + threads_per_block - 1) / threads_per_block;

    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0);
    hipLaunchKernelGGL(lorenz_kernel,
                       dim3(blocks), dim3(threads_per_block), 0, 0,
                       d_tail, d_hwids, d_cycles);
    hipEventRecord(e1);
    HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0.0f;
    hipEventElapsedTime(&ms, e0, e1);

    float* h_tail = (float*)malloc(tail_bytes);
    uint32_t* h_hwids = (uint32_t*)malloc(id_bytes);
    uint32_t* h_cycles = (uint32_t*)malloc(id_bytes);
    HIP_CHECK(hipMemcpy(h_tail,   d_tail,   tail_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_hwids,  d_hwids,  id_bytes,   hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_cycles, d_cycles, id_bytes,   hipMemcpyDeviceToHost));

    // Write binary blob to stdout: header(4 uint32) + tail + hwids + cycles
    uint32_t header[4] = { (uint32_t)N_WAVES, (uint32_t)TAIL_LEN, (uint32_t)N_STEPS, (uint32_t)batch_id };
    fwrite(header,   sizeof(uint32_t), 4,            stdout);
    fwrite(h_tail,   sizeof(float),    N_WAVES*TAIL_LEN*3, stdout);
    fwrite(h_hwids,  sizeof(uint32_t), N_WAVES,      stdout);
    fwrite(h_cycles, sizeof(uint32_t), N_WAVES,      stdout);
    fflush(stdout);

    fprintf(stderr, "[B_lorenz] batch=%d kernel_ms=%.2f\n", batch_id, ms);

    free(h_tail); free(h_hwids); free(h_cycles);
    hipFree(d_tail); hipFree(d_hwids); hipFree(d_cycles);
    return 0;
}
