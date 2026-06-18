/*  locked_apart.hip — LockedApart-style atomic-contention per-CU probe (gfx1151).
 *
 *  H2 (research_plan/H2_PREREG_2026-06-09.md): test whether bursty atomic_add
 *  contention on a single LDS counter reveals stable per-CU latency moments
 *  that identify the host chassis.
 *
 *  Protocol (frozen):
 *    - N_BLOCKS workgroups of N_THREADS threads each, racing atomicAdd on
 *      a single 32-bit LDS counter for N_ROUNDS rounds.
 *    - Each block records cycles_start / cycles_end via hwreg(29) and reports
 *      its (CU_id, SE_id, latency_cycles) tuple to global memory.
 *    - 1000 traces per launch. Each trace = vector of per-CU {mean,std,p95,p99}
 *      computed by the host after readback.
 *
 *  Reuses the hwreg/HW_ID decode established in puf_kernel.hip. Does NOT
 *  touch SMU mailbox or amdgpu_regs_didt — both are reboot-traps (CLAUDE.md).
 *
 *  Build (gfx1151):
 *    hipcc --offload-arch=gfx1151 -O1 -o locked_apart locked_apart.hip
 *  Run:
 *    ./locked_apart 1000 results/IDENTITY_H2_2026-06-09/ikaros_amb30.bin
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <string>
#include <algorithm>

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define N_BLOCKS   64
#define N_THREADS  256
#define N_ROUNDS   1024
#define WAVE_SIZE  32

struct CUSample {
    uint32_t hw_id;
    uint32_t cyc_start;
    uint32_t cyc_end;
    uint32_t latency;
};

__global__ void locked_apart_kernel(CUSample* out,
                                    unsigned int* lds_counter_global,
                                    int rounds)
{
    __shared__ unsigned int lds_counter;
    if (threadIdx.x == 0) lds_counter = 0;
    __syncthreads();

    uint32_t hw_id  = __builtin_amdgcn_s_getreg((23) | (0 << 6) | ((32-1) << 11));
    uint32_t cyc0   = __builtin_amdgcn_s_getreg((29) | (0 << 6) | ((32-1) << 11));

    for (int r = 0; r < rounds; ++r) {
        atomicAdd(&lds_counter, 1u);
    }

    uint32_t cyc1   = __builtin_amdgcn_s_getreg((29) | (0 << 6) | ((32-1) << 11));

    if (threadIdx.x == 0) {
        CUSample s;
        s.hw_id     = hw_id;
        s.cyc_start = cyc0;
        s.cyc_end   = cyc1;
        s.latency   = (cyc1 - cyc0);
        out[blockIdx.x] = s;
        // Also push lds_counter to a global scratch so the compiler can't
        // optimize the atomicAdd path away.
        atomicAdd(lds_counter_global, lds_counter);
    }
}

int main(int argc, char** argv)
{
    int n_traces = (argc >= 2) ? std::atoi(argv[1]) : 1000;
    std::string out_path = (argc >= 3) ? argv[2] : "locked_apart_out.bin";

    CUSample* d_out = nullptr;
    unsigned int* d_scratch = nullptr;
    HIP_CHECK(hipMalloc(&d_out, sizeof(CUSample) * N_BLOCKS));
    HIP_CHECK(hipMalloc(&d_scratch, sizeof(unsigned int)));
    HIP_CHECK(hipMemset(d_scratch, 0, sizeof(unsigned int)));

    std::vector<CUSample> h_traces;
    h_traces.reserve((size_t)n_traces * N_BLOCKS);

    // Warm up
    for (int w = 0; w < 5; ++w) {
        locked_apart_kernel<<<N_BLOCKS, N_THREADS>>>(d_out, d_scratch, N_ROUNDS);
        HIP_CHECK(hipDeviceSynchronize());
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < n_traces; ++t) {
        locked_apart_kernel<<<N_BLOCKS, N_THREADS>>>(d_out, d_scratch, N_ROUNDS);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<CUSample> h_one(N_BLOCKS);
        HIP_CHECK(hipMemcpy(h_one.data(), d_out, sizeof(CUSample) * N_BLOCKS,
                            hipMemcpyDeviceToHost));
        h_traces.insert(h_traces.end(), h_one.begin(), h_one.end());
        // Brief inter-trace cooldown to avoid runaway thermal accumulation;
        // CLAUDE.md notes APU hits 100C from FPGA I/O + numpy alone, so be
        // conservative here even though this is GPU-only.
        if ((t % 50) == 49) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    fprintf(stderr, "captured %d traces × %d blocks in %.2fs (%.1f traces/s)\n",
            n_traces, N_BLOCKS, elapsed, n_traces / elapsed);

    FILE* fp = fopen(out_path.c_str(), "wb");
    if (!fp) { fprintf(stderr, "fopen failed: %s\n", out_path.c_str()); return 1; }
    uint32_t hdr[4] = { 0x4C41504Au /* "LAPJ" */, (uint32_t)n_traces, (uint32_t)N_BLOCKS, (uint32_t)N_ROUNDS };
    fwrite(hdr, sizeof(uint32_t), 4, fp);
    fwrite(h_traces.data(), sizeof(CUSample), h_traces.size(), fp);
    fclose(fp);
    fprintf(stderr, "wrote %zu samples to %s\n", h_traces.size(), out_path.c_str());

    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_scratch));
    return 0;
}
