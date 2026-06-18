// z2151_perf_snapshot_stats.cpp — PERF_SNAPSHOT + SHADER_CYCLES statistical sampler
// V2: Adds inter-iteration workload to exercise different HW paths + temporal spacing
//
// Target: AMD gfx1151 (RDNA 3.5) via HIP
// Requires: HSA_OVERRIDE_GFX_VERSION=11.0.0
//
// Output: CSV to stdout with columns:
//   iteration,wave_id,perf_snapshot,shader_cycles,hw_id1,status,work_result
//
// Usage: ./z2151_perf_snapshot_stats [num_iterations] [waves_per_iter] [work_iters]

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>

#define HIP_CHECK(call)                                                        \
    do {                                                                        \
        hipError_t err = (call);                                                \
        if (err != hipSuccess) {                                                \
            fprintf(stderr, "HIP error at %s:%d — %s (%d)\n",                 \
                    __FILE__, __LINE__, hipGetErrorString(err), err);           \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

struct PerfSample {
    uint32_t perf_snapshot;  // hwreg(27)
    uint32_t shader_cycles;  // hwreg(29)
    uint32_t hw_id1;         // hwreg(23)
    uint32_t status;         // hwreg(2)
    uint32_t work_result;    // result from workload (prevents optimizer elimination)
};

// Kernel: do variable-length work THEN sample hwregs
// work_type cycles through different ALU patterns each iteration
__global__ void perf_sample_kernel(PerfSample* out, int max_waves,
                                    int work_iters, int work_type) {
    int wave_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane = threadIdx.x % warpSize;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // ALL lanes do work (exercises full SIMD), but only lane 0 reads hwregs
    uint32_t acc = tid ^ 0xDEADBEEF;

    // Variable workload to exercise different HW units
    for (int i = 0; i < work_iters; i++) {
        switch (work_type % 4) {
            case 0: // Integer ALU chain
                acc = acc * 2654435761u + i;
                acc ^= (acc >> 16);
                break;
            case 1: // Bit manipulation
                acc = __brev(acc) ^ (acc << 3);
                acc += __clz(acc | 1);
                break;
            case 2: // Mixed ops
                acc = (acc >> (i & 15)) | (acc << (16 - (i & 15)));
                acc ^= lane * 37;
                break;
            case 3: // Multiply-heavy
                acc *= (acc | 0xFF) + i;
                acc ^= acc >> 8;
                break;
        }
    }

    if (lane != 0) return;
    if (wave_id >= max_waves) return;

    PerfSample s;
    uint32_t val;

    asm volatile("s_getreg_b32 %0, hwreg(27)" : "=s"(val)); s.perf_snapshot = val;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(val)); s.shader_cycles = val;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(val)); s.hw_id1 = val;
    asm volatile("s_getreg_b32 %0, hwreg(2)"  : "=s"(val)); s.status = val;
    s.work_result = acc;

    out[wave_id] = s;
}

int main(int argc, char** argv) {
    int num_iters = 500;
    int waves_per_iter = 16;
    int work_iters = 1000;  // ALU iterations per wavefront per launch

    if (argc > 1) num_iters = atoi(argv[1]);
    if (argc > 2) waves_per_iter = atoi(argv[2]);
    if (argc > 3) work_iters = atoi(argv[3]);

    int warp_size = 32;
    int threads_per_block = 64;
    int num_blocks = (waves_per_iter * warp_size + threads_per_block - 1) / threads_per_block;
    int actual_waves = num_blocks * (threads_per_block / warp_size);

    fprintf(stderr, "z2151v2: PERF_SNAPSHOT Statistical Sampler (with workload)\n");
    fprintf(stderr, "  Iterations: %d, Wavefronts/iter: %d, Work/iter: %d ALU ops\n",
            num_iters, actual_waves, work_iters);
    fprintf(stderr, "  Total samples: %d\n", num_iters * actual_waves);

    size_t buf_size = actual_waves * sizeof(PerfSample);
    PerfSample* d_samples = nullptr;
    PerfSample* h_samples = (PerfSample*)malloc(buf_size);
    HIP_CHECK(hipMalloc(&d_samples, buf_size));

    printf("iteration,wave_id,perf_snapshot,shader_cycles,hw_id1,status,work_result\n");
    fflush(stdout);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < num_iters; iter++) {
        HIP_CHECK(hipMemset(d_samples, 0, buf_size));

        // Vary work_type each iteration to exercise different HW paths
        int work_type = iter % 4;

        hipLaunchKernelGGL(perf_sample_kernel, dim3(num_blocks), dim3(threads_per_block),
                           0, 0, d_samples, actual_waves, work_iters, work_type);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipMemcpy(h_samples, d_samples, buf_size, hipMemcpyDeviceToHost));

        for (int w = 0; w < actual_waves; w++) {
            PerfSample& s = h_samples[w];
            printf("%d,%d,%u,%u,0x%08X,0x%08X,%u\n",
                   iter, w, s.perf_snapshot, s.shader_cycles, s.hw_id1, s.status, s.work_result);
        }

        // Flush every 50 iterations for crash safety
        if ((iter + 1) % 50 == 0) {
            fflush(stdout);
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            fprintf(stderr, "  [checkpoint] iter %d/%d (%.1f%%) — %.1fs elapsed\n",
                    iter + 1, num_iters, 100.0 * (iter + 1) / num_iters, elapsed);
        }

        // Small host-side delay every 10 iterations for temporal spacing
        if (iter % 10 == 9) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    fflush(stdout);
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    fprintf(stderr, "  Done: %.3f seconds (%.1f iter/sec, %.1f samples/sec)\n",
            elapsed, num_iters / elapsed, (num_iters * actual_waves) / elapsed);

    free(h_samples);
    HIP_CHECK(hipFree(d_samples));
    return 0;
}
