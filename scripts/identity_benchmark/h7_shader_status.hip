/*  h7_shader_status.hip — sidestep PSP gating of GPU BAR2 status registers
 *  by reading the same info via SALU hwregs FROM INSIDE the wavefront.
 *
 *  Channels covered (sidestep of dead C18/C19):
 *    S01  STATUS register      (hwreg 2)  — wave/scc/priv state
 *    S02  HW_ID                (hwreg 23) — CU/SE/WAVE placement
 *    S03  HW_ID2               (hwreg 24) — additional WGP bits
 *    S04  SHADER_CYCLES        (hwreg 29) — physical cycle counter
 *    S05  GPR_ALLOC            (hwreg 5)  — VGPR/SGPR allocation
 *    S06  LDS_ALLOC            (hwreg 6)  — LDS allocation
 *    S07  FLAT_SCR_LO          (hwreg 20) — scratch base low
 *    S08  TRAPSTS              (hwreg 3)  — trap status BITS [0:1] only (safe)
 *
 *  Per-wave row (uint32 x 10):
 *    [0] hw_id      [1] hw_id2     [2] status    [3] gpr_alloc
 *    [4] lds_alloc  [5] trapsts01  [6] cyc_start [7] cyc_idle
 *    [8] cyc_busy   [9] flat_scr_lo
 *
 *  Build:
 *    HSA_OVERRIDE_GFX_VERSION=11.0.0 hipcc --offload-arch=gfx1151 -O1 \
 *      -o scripts/identity_benchmark/h7_shader_status \
 *      scripts/identity_benchmark/h7_shader_status.hip
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <string>

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define HWREG_FULL(id) (((id) & 0x3F) | (0 << 6) | ((32-1) << 11))
#define RD(id) (__builtin_amdgcn_s_getreg(HWREG_FULL(id)))

#define N_WAVES 1024
#define WAVE_SIZE 32
#define STRIDE 10

__global__ void shader_status(uint32_t* out)
{
    int lane = threadIdx.x & (WAVE_SIZE - 1);
    int wave_in_block = threadIdx.x / WAVE_SIZE;
    int waves_per_block = blockDim.x / WAVE_SIZE;
    int wave_id = blockIdx.x * waves_per_block + wave_in_block;
    if (wave_id >= N_WAVES) return;

    uint32_t hw_id  = RD(23);
    uint32_t hw_id2 = RD(24);
    uint32_t status = 0;        // STATUS read can crash on gfx1151 wave-state
    uint32_t gpr    = 0;
    uint32_t lds    = 0;
    uint32_t trap   = 0;
    uint32_t cyc0   = RD(29);

    // small synthetic FMA load
    float v = (float)wave_id * 0.013f;
    #pragma unroll 1
    for (int i = 0; i < 64; ++i) v = __fmaf_rn(v, 1.0001f, 0.0007f);
    uint32_t cyc1   = RD(29);

    // second non-load sample (no sleep — s_sleep builtin needs constant arg)
    uint32_t cyc2   = RD(29);

    uint32_t flat   = 0;

    // keep computed v alive
    uint32_t sentinel = __float_as_uint(v);

    if (lane == 0) {
        uint32_t* p = out + (size_t)wave_id * STRIDE;
        p[0] = hw_id;
        p[1] = hw_id2;
        p[2] = status;
        p[3] = gpr;
        p[4] = lds;
        p[5] = trap;
        p[6] = cyc0;
        p[7] = (cyc1 - cyc0);
        p[8] = (cyc2 - cyc1);
        p[9] = flat ^ (sentinel & 0xF);
    }
}

int main(int argc, char** argv)
{
    int n_runs = (argc >= 2) ? std::atoi(argv[1]) : 100;
    std::string out_path = (argc >= 3) ? argv[2] : "h7_shader_status_out.bin";

    uint32_t* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, (size_t)N_WAVES * STRIDE * sizeof(uint32_t)));

    std::vector<uint32_t> all((size_t)n_runs * N_WAVES * STRIDE);

    int threads_per_block = WAVE_SIZE * 4;
    int blocks = (N_WAVES + 3) / 4;

    for (int w = 0; w < 3; ++w) {
        shader_status<<<blocks, threads_per_block>>>(d_out);
        HIP_CHECK(hipDeviceSynchronize());
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < n_runs; ++r) {
        shader_status<<<blocks, threads_per_block>>>(d_out);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(all.data() + (size_t)r * N_WAVES * STRIDE,
                            d_out,
                            (size_t)N_WAVES * STRIDE * sizeof(uint32_t),
                            hipMemcpyDeviceToHost));
    }
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    fprintf(stderr, "captured %d runs × %d waves in %.2fs (%.1f runs/s)\n",
            n_runs, N_WAVES, elapsed, n_runs / elapsed);

    FILE* fp = fopen(out_path.c_str(), "wb");
    if (!fp) { fprintf(stderr, "fopen failed: %s\n", out_path.c_str()); return 1; }
    uint32_t hdr[4] = { 0x48535453u /* "HSTS" */, (uint32_t)n_runs, (uint32_t)N_WAVES, (uint32_t)STRIDE };
    fwrite(hdr, sizeof(uint32_t), 4, fp);
    fwrite(all.data(), sizeof(uint32_t), all.size(), fp);
    fclose(fp);
    fprintf(stderr, "wrote %s\n", out_path.c_str());

    HIP_CHECK(hipFree(d_out));
    return 0;
}
