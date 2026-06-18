/*  h7_shader_probe.hip — per-wave deep ISA-level identity sip on gfx1151.
 *
 *  Implements channels C12-C15 of H7 (research_plan/H7_PREREG_2026-06-09.md):
 *    C12  hwreg(29) SHADER_CYCLES skew per wave
 *    C13  hwreg(23) HW_ID  -> CU placement
 *    C14  s_setreg MODE: 4 FP rounding modes -> 4 distinct dot-product bit-patterns
 *    C15  sinf cycle jitter (transcendental data-dep timing 0-62 cycles)
 *
 *  Per-wave record layout (uint32):
 *    samples[w*12 + 0] = HW_ID                      (hwreg 23)
 *    samples[w*12 + 1] = HW_ID2                     (hwreg 24)
 *    samples[w*12 + 2] = MODE original              (hwreg 1)
 *    samples[w*12 + 3] = cyc start                  (hwreg 29)
 *    samples[w*12 + 4] = cyc after RNE FMA chain    (delta vs prev)
 *    samples[w*12 + 5] = cyc after R+inf chain
 *    samples[w*12 + 6] = cyc after R-inf chain
 *    samples[w*12 + 7] = cyc after RTZ chain
 *    samples[w*12 + 8] = dot result RNE  (float bits)
 *    samples[w*12 + 9] = dot result R+inf
 *    samples[w*12 +10] = dot result RTZ
 *    samples[w*12 +11] = sinf jitter avg (uint cycle, lane 0 reports)
 *
 *  Build:
 *    HSA_OVERRIDE_GFX_VERSION=11.0.0 hipcc --offload-arch=gfx1151 -O1 \
 *      -o h7_shader_probe scripts/identity_benchmark/h7_shader_probe.hip
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define N_WAVES 1024
#define WAVE_SIZE 32
#define DOT_LEN 128
#define SAMPLE_STRIDE 12

#define HWREG_FULL(id) (((id) & 0x3F) | (0 << 6) | ((32-1) << 11))
#define RD_HWREG(id) (__builtin_amdgcn_s_getreg(HWREG_FULL(id)))
#define WR_MODE(val) (__builtin_amdgcn_s_setreg(HWREG_FULL(1), (val)))

__global__ void h7_kernel(uint32_t* samples,
                          const float* fixed_input,
                          const float* fixed_weights)
{
    int lane = threadIdx.x & (WAVE_SIZE - 1);
    int wave_in_block = threadIdx.x / WAVE_SIZE;
    int waves_per_block = blockDim.x / WAVE_SIZE;
    int wave_id = blockIdx.x * waves_per_block + wave_in_block;
    if (wave_id >= N_WAVES) return;

    uint32_t hw_id  = RD_HWREG(23);
    uint32_t hw_id2 = RD_HWREG(24);
    uint32_t mode0  = RD_HWREG(1);
    uint32_t cyc0   = RD_HWREG(29);

    // ---- 4 FP rounding modes, identical input -- get 4 bit patterns ----
    float acc = 0.0f;

    // Mode 0: RNE (default, mode bits 1:0 = 0)
    WR_MODE((mode0 & ~0x3) | 0x0);
    #pragma unroll 1
    for (int i = 0; i < DOT_LEN; ++i) {
        float a = fixed_input [(lane * DOT_LEN + i) & 1023];
        float b = fixed_weights[(lane * DOT_LEN + i) & 1023];
        acc = __fmaf_rn(a, b, acc);
    }
    uint32_t cyc_rne = RD_HWREG(29);
    float acc_rne = acc;

    // Mode 1: round toward +inf
    WR_MODE((mode0 & ~0x3) | 0x1);
    acc = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < DOT_LEN; ++i) {
        float a = fixed_input [(lane * DOT_LEN + i) & 1023];
        float b = fixed_weights[(lane * DOT_LEN + i) & 1023];
        acc = __fmaf_rn(a, b, acc);
    }
    uint32_t cyc_pinf = RD_HWREG(29);
    float acc_pinf = acc;

    // Mode 2: round toward -inf
    WR_MODE((mode0 & ~0x3) | 0x2);
    acc = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < DOT_LEN; ++i) {
        float a = fixed_input [(lane * DOT_LEN + i) & 1023];
        float b = fixed_weights[(lane * DOT_LEN + i) & 1023];
        acc = __fmaf_rn(a, b, acc);
    }
    uint32_t cyc_minf = RD_HWREG(29);

    // Mode 3: RTZ
    WR_MODE((mode0 & ~0x3) | 0x3);
    acc = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < DOT_LEN; ++i) {
        float a = fixed_input [(lane * DOT_LEN + i) & 1023];
        float b = fixed_weights[(lane * DOT_LEN + i) & 1023];
        acc = __fmaf_rn(a, b, acc);
    }
    uint32_t cyc_rtz = RD_HWREG(29);
    float acc_rtz = acc;

    // restore MODE
    WR_MODE(mode0);

    // ---- sinf jitter — 8 transcendental calls, accumulate timing delta ----
    uint32_t cyc_s0 = RD_HWREG(29);
    float s = (float)(wave_id & 0xFF) * 0.073f + 0.011f;
    #pragma unroll 1
    for (int i = 0; i < 8; ++i) {
        s = sinf(s + (float)i * 0.13f);
    }
    uint32_t cyc_s1 = RD_HWREG(29);
    uint32_t sinf_jitter = (cyc_s1 - cyc_s0);
    // mix s into the sentinel slot via reinterpret to defeat DCE
    uint32_t sentinel = __float_as_uint(s);

    if (lane == 0) {
        uint32_t* p = samples + (size_t)wave_id * SAMPLE_STRIDE;
        p[0]  = hw_id;
        p[1]  = hw_id2;
        p[2]  = mode0;
        p[3]  = cyc0;
        p[4]  = cyc_rne;
        p[5]  = cyc_pinf;
        p[6]  = cyc_minf;
        p[7]  = cyc_rtz;
        p[8]  = __float_as_uint(acc_rne);
        p[9]  = __float_as_uint(acc_pinf);
        p[10] = __float_as_uint(acc_rtz);
        p[11] = sinf_jitter ^ (sentinel & 0xF);  // keep s alive
    }
}

int main(int argc, char** argv)
{
    int n_runs = (argc >= 2) ? std::atoi(argv[1]) : 100;
    std::string out_path = (argc >= 3) ? argv[2] : "h7_shader_out.bin";

    // Deterministic source — same on every host. Variation MUST come from silicon.
    std::vector<float> h_in (1024), h_w(1024);
    for (int i = 0; i < 1024; ++i) {
        h_in[i] = ((i * 1103515245u + 12345u) >> 16) * (1.0f / 32768.0f);
        h_w [i] = ((i * 2654435761u + 5u) >> 16)    * (1.0f / 32768.0f);
    }

    float *d_in = nullptr, *d_w = nullptr;
    uint32_t *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, 1024 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_w,  1024 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, (size_t)N_WAVES * SAMPLE_STRIDE * sizeof(uint32_t)));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), 1024 * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w,  h_w.data(),  1024 * sizeof(float), hipMemcpyHostToDevice));

    std::vector<uint32_t> all((size_t)n_runs * N_WAVES * SAMPLE_STRIDE);

    int threads_per_block = WAVE_SIZE * 4;
    int blocks = (N_WAVES + 3) / 4;

    // warmup
    for (int w = 0; w < 3; ++w) {
        h7_kernel<<<blocks, threads_per_block>>>(d_out, d_in, d_w);
        HIP_CHECK(hipDeviceSynchronize());
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < n_runs; ++r) {
        h7_kernel<<<blocks, threads_per_block>>>(d_out, d_in, d_w);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(all.data() + (size_t)r * N_WAVES * SAMPLE_STRIDE,
                            d_out,
                            (size_t)N_WAVES * SAMPLE_STRIDE * sizeof(uint32_t),
                            hipMemcpyDeviceToHost));
    }
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    fprintf(stderr, "captured %d runs × %d waves in %.2fs (%.1f runs/s)\n",
            n_runs, N_WAVES, elapsed, n_runs / elapsed);

    FILE* fp = fopen(out_path.c_str(), "wb");
    if (!fp) { fprintf(stderr, "fopen failed: %s\n", out_path.c_str()); return 1; }
    uint32_t hdr[4] = { 0x4837535Au /* "H7SZ" */, (uint32_t)n_runs, (uint32_t)N_WAVES, (uint32_t)SAMPLE_STRIDE };
    fwrite(hdr, sizeof(uint32_t), 4, fp);
    fwrite(all.data(), sizeof(uint32_t), all.size(), fp);
    fclose(fp);
    fprintf(stderr, "wrote %zu uint32 to %s\n", all.size(), out_path.c_str());

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_w));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
