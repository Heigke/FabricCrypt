// z2153_deep_probe_bridge.cpp — Enhanced PERF_SNAPSHOT probe with ALL safe hwregs
// Adds hwreg(28) IB_STS2, hwreg(15) MEM_BASES, hwreg(5) GPR_ALLOC, hwreg(7) IB_STS
// + Outputs stochastic jitter values for FPGA ISI injection (z2153 bridge)
//
// Target: AMD gfx1151 (RDNA 3.5) via HIP
// Requires: HSA_OVERRIDE_GFX_VERSION=11.0.0
//
// Output: CSV to stdout with ALL hwreg columns + computed jitter byte

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

struct DeepSample {
    uint32_t perf_snapshot;  // hwreg(27) — 24-bit performance counter snapshot
    uint32_t shader_cycles;  // hwreg(29) — shader execution cycles
    uint32_t hw_id1;         // hwreg(23) — SE/SA/WGP/SIMD/WAVE topology
    uint32_t status;         // hwreg(2)  — SCC/EXECZ/VCCZ wave status
    uint32_t ib_sts2;        // hwreg(28) — NEW: extended instruction buffer status (GFX11)
    uint32_t mem_bases;      // hwreg(15) — NEW: memory base addresses
    uint32_t gpr_alloc;      // hwreg(5)  — VGPR/SGPR allocation state
    uint32_t ib_sts;         // hwreg(7)  — instruction buffer status
    uint32_t mode;           // hwreg(1)  — FP rounding/denorm mode
    uint32_t work_result;    // prevents optimizer elimination
};

// Kernel: variable workload + read ALL safe hwregs
__global__ void deep_sample_kernel(DeepSample* out, int max_waves,
                                    int work_iters, int work_type) {
    int wave_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane = threadIdx.x % warpSize;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // ALL lanes do work
    uint32_t acc = tid ^ 0xDEADBEEF;
    for (int i = 0; i < work_iters; i++) {
        switch (work_type % 4) {
            case 0: acc = acc * 2654435761u + i; acc ^= (acc >> 16); break;
            case 1: acc = __brev(acc) ^ (acc << 3); acc += __clz(acc | 1); break;
            case 2: acc = (acc >> (i & 15)) | (acc << (16 - (i & 15))); acc ^= lane * 37; break;
            case 3: acc *= (acc | 0xFF) + i; acc ^= acc >> 8; break;
        }
    }

    if (lane != 0) return;
    if (wave_id >= max_waves) return;

    DeepSample s;
    uint32_t val;

    // Read ALL safe hwregs in sequence
    asm volatile("s_getreg_b32 %0, hwreg(27)" : "=s"(val)); s.perf_snapshot = val;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(val)); s.shader_cycles = val;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(val)); s.hw_id1 = val;
    asm volatile("s_getreg_b32 %0, hwreg(2)"  : "=s"(val)); s.status = val;
    asm volatile("s_getreg_b32 %0, hwreg(28)" : "=s"(val)); s.ib_sts2 = val;
    asm volatile("s_getreg_b32 %0, hwreg(15)" : "=s"(val)); s.mem_bases = val;
    asm volatile("s_getreg_b32 %0, hwreg(5)"  : "=s"(val)); s.gpr_alloc = val;
    asm volatile("s_getreg_b32 %0, hwreg(7)"  : "=s"(val)); s.ib_sts = val;
    asm volatile("s_getreg_b32 %0, hwreg(1)"  : "=s"(val)); s.mode = val;
    s.work_result = acc;

    out[wave_id] = s;
}

int main(int argc, char** argv) {
    int num_iters = 500;
    int waves_per_iter = 16;
    int work_iters = 50000;  // substantial ALU work

    if (argc > 1) num_iters = atoi(argv[1]);
    if (argc > 2) waves_per_iter = atoi(argv[2]);
    if (argc > 3) work_iters = atoi(argv[3]);

    int warp_size = 32;
    int threads_per_block = 64;
    int num_blocks = (waves_per_iter * warp_size + threads_per_block - 1) / threads_per_block;
    int actual_waves = num_blocks * (threads_per_block / warp_size);

    fprintf(stderr, "z2153: Deep Probe Bridge — ALL safe hwregs + stochastic jitter\n");
    fprintf(stderr, "  Iterations: %d, Wavefronts/iter: %d, Work/iter: %d ALU ops\n",
            num_iters, actual_waves, work_iters);
    fprintf(stderr, "  Total samples: %d\n", num_iters * actual_waves);
    fprintf(stderr, "  Registers: PERF_SNAPSHOT(27), SHADER_CYCLES(29), HW_ID1(23),\n");
    fprintf(stderr, "             STATUS(2), IB_STS2(28), MEM_BASES(15), GPR_ALLOC(5),\n");
    fprintf(stderr, "             IB_STS(7), MODE(1)\n");

    size_t buf_size = actual_waves * sizeof(DeepSample);
    DeepSample* d_samples = nullptr;
    DeepSample* h_samples = (DeepSample*)malloc(buf_size);
    HIP_CHECK(hipMalloc(&d_samples, buf_size));

    printf("iteration,wave_id,perf_snapshot,shader_cycles,hw_id1,status,"
           "ib_sts2,mem_bases,gpr_alloc,ib_sts,mode,work_result,jitter_byte\n");
    fflush(stdout);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < num_iters; iter++) {
        HIP_CHECK(hipMemset(d_samples, 0, buf_size));

        int work_type = iter % 4;
        hipLaunchKernelGGL(deep_sample_kernel, dim3(num_blocks), dim3(threads_per_block),
                           0, 0, d_samples, actual_waves, work_iters, work_type);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_samples, d_samples, buf_size, hipMemcpyDeviceToHost));

        for (int w = 0; w < actual_waves; w++) {
            DeepSample& s = h_samples[w];
            // Compute jitter byte: XOR of low bytes from all dynamic regs
            // This fuses multiple entropy sources into one stochastic value
            uint8_t jitter = (uint8_t)(
                (s.perf_snapshot & 0xFF) ^
                (s.shader_cycles & 0xFF) ^
                (s.ib_sts2 & 0xFF) ^
                (s.status & 0xFF)
            );
            printf("%d,%d,%u,%u,0x%08X,0x%08X,0x%08X,0x%08X,0x%08X,0x%08X,0x%08X,%u,%u\n",
                   iter, w, s.perf_snapshot, s.shader_cycles, s.hw_id1, s.status,
                   s.ib_sts2, s.mem_bases, s.gpr_alloc, s.ib_sts, s.mode,
                   s.work_result, (unsigned)jitter);
        }

        // Crash-safe flush every 50 iterations
        if ((iter + 1) % 50 == 0) {
            fflush(stdout);
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            fprintf(stderr, "  [checkpoint] iter %d/%d (%.1f%%) — %.1fs elapsed\n",
                    iter + 1, num_iters, 100.0 * (iter + 1) / num_iters, elapsed);
        }

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
