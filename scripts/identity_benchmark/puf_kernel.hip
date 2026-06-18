/*  puf_kernel.hip — Hardware Identity PUF challenge kernel for gfx1151
 *
 *  Phase 1 of IDENTITY_BENCHMARK_2026-05-30.
 *
 *  Deterministic source-code: fixed inputs, fixed weights, fixed launch.
 *  Per-wave samples: HW_ID, SHADER_CYCLES (#29), PERF_SNAPSHOT (#17),
 *  dot product result, atomic ordering signature.
 *
 *  Output layout (per wave w in [0, n_waves)):
 *    samples[w*8 + 0]  : dot result (float bits, reinterpret as uint32_t)
 *    samples[w*8 + 1]  : HW_ID (#12)  (raw 32-bit)
 *    samples[w*8 + 2]  : SALU cycles delta (hwreg(29) end - start)
 *    samples[w*8 + 3]  : PERF_SNAPSHOT (#17) raw
 *    samples[w*8 + 4]  : atomic ordering "old" returned to lane 0
 *    samples[w*8 + 5]  : atomic ordering xor of all 8 contributors
 *    samples[w*8 + 6]  : wave_id field of HW_ID (bits inferred CU/SE)
 *    samples[w*8 + 7]  : SHADER_CYCLES start raw
 *
 *  Build (gfx1151):
 *    hipcc --offload-arch=gfx1151 -O1 -o puf_kernel puf_kernel.hip
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define N_WAVES 1024
#define WAVE_SIZE 32          // gfx1151 wavefront
#define DOT_LEN 64
#define ATOMIC_THREADS 8

__global__ void puf_kernel(uint32_t* samples,
                           const float* fixed_input,
                           const float* fixed_weights,
                           unsigned int* atomic_scratch)
{
    // Lane / wave indices
    int lane = threadIdx.x & (WAVE_SIZE - 1);
    int wave_in_block = threadIdx.x / WAVE_SIZE;
    int waves_per_block = blockDim.x / WAVE_SIZE;
    int wave_id = blockIdx.x * waves_per_block + wave_in_block;
    if (wave_id >= N_WAVES) return;

    // --- read HW_ID1 (hwreg 23) on gfx11 = WGP placement ---
    // Use intrinsic with explicit (id, offset, size) encoding — bare hwreg(N)
    // in inline asm returned 0 on some driver versions.
    uint32_t hw_id  = __builtin_amdgcn_s_getreg((23) | (0 << 6) | ((32-1) << 11));
    uint32_t hw_id2 = __builtin_amdgcn_s_getreg((24) | (0 << 6) | ((32-1) << 11));

    // --- read SHADER_CYCLES start (hwreg 29) ---
    uint32_t cyc0 = __builtin_amdgcn_s_getreg((29) | (0 << 6) | ((32-1) << 11));

    // --- fixed dot product (deterministic source) ---
    // Each lane multiplies its slice of fixed_input * fixed_weights then
    // we do a wavefront reduction. Source is deterministic; result variation
    // is FP ordering / silicon FMA path differences.
    float acc = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < DOT_LEN; i++) {
        float a = fixed_input[(lane * DOT_LEN + i) & 1023];
        float b = fixed_weights[(lane * DOT_LEN + i) & 1023];
        acc = __fmaf_rn(a, b, acc);
    }
    // chain a few tanh + branch so the SALU path is non-trivial
    float chained = tanhf(acc * 0.5f);
    if (chained > 0.0f) chained = __fmaf_rn(chained, chained, 0.25f);
    else                chained = __fmaf_rn(chained, -chained, 0.25f);

    // wave-shuffle reduce (DPP / readlane). Use simple loop with __shfl_xor.
    float red = chained;
    for (int off = 16; off > 0; off >>= 1) {
        red += __shfl_xor(red, off);
    }
    // Now lane 0 of the wave has the sum.

    // --- atomic ordering signature ---
    // 8 lanes (0..7) atomicAdd to the same scratch slot; the "old" value
    // each lane sees is its arrival index. We record lane 0's old and xor
    // the contributions to capture full ordering as a tiny signature.
    unsigned int my_old = 0xFFFFFFFFu;
    if (lane < ATOMIC_THREADS) {
        unsigned int tag = (unsigned int)(0xA5 + lane * 0x11);
        my_old = atomicAdd(&atomic_scratch[wave_id], tag);
    }
    // gather "old"s via shfl from lanes 0..7 into lane 0
    unsigned int old0 = my_old;
    unsigned int xor_all = my_old;
    for (int src = 1; src < ATOMIC_THREADS; src++) {
        unsigned int v = __shfl(my_old, src);
        if (src == 0) old0 = v;
        xor_all ^= v;
    }

    // --- read SHADER_CYCLES end ---
    uint32_t cyc1 = __builtin_amdgcn_s_getreg((29) | (0 << 6) | ((32-1) << 11));

    // --- read PERF_SNAPSHOT (hwreg 27) ---
    uint32_t perf = __builtin_amdgcn_s_getreg((27) | (0 << 6) | ((32-1) << 11));

    if (lane == 0) {
        uint32_t dot_bits;
        memcpy(&dot_bits, &red, sizeof(uint32_t));
        // HW_ID1 gfx11 (per z2338 validation):
        // wave_id [5:0], simd_id [9:8], wgp_id [15:10], sa_id [16], se_id [20:18]
        uint32_t se_field  = (hw_id >> 18) & 0x7;
        uint32_t sa_field  = (hw_id >> 16) & 0x1;
        uint32_t wgp_field = (hw_id >> 10) & 0x3F;
        uint32_t simd_field= (hw_id >>  8) & 0x3;
        // CU-global packed: se(3) | sa(1) | wgp(6) | simd(2) = 12 bits
        uint32_t cu_global = (se_field << 9) | (sa_field << 8) | (wgp_field << 2) | simd_field;

        uint32_t cyc_delta = cyc1 - cyc0;

        uint32_t base = wave_id * 8u;
        samples[base + 0] = dot_bits;
        samples[base + 1] = hw_id;
        samples[base + 2] = cyc_delta;
        samples[base + 3] = perf;
        samples[base + 4] = old0;
        samples[base + 5] = xor_all;
        samples[base + 6] = cu_global;
        samples[base + 7] = cyc0;
    }
}


int main(int argc, char** argv) {
    int n_reps = (argc > 1) ? atoi(argv[1]) : 1;
    const char* out_path = (argc > 2) ? argv[2] : "puf_out.bin";

    // Deterministic fixed inputs (NO RNG). Generated by a fixed integer recipe.
    float h_input[1024];
    float h_weights[1024];
    for (int i = 0; i < 1024; i++) {
        // Fixed transcendental-free pattern; reproducible across machines.
        float fi = (float)((i * 1103515245u + 12345u) & 0xFFFF) / 65535.0f - 0.5f;
        float fw = (float)(((i * 22695477u) + 1u) & 0xFFFF) / 65535.0f - 0.5f;
        h_input[i] = fi;
        h_weights[i] = fw;
    }

    float *d_input, *d_weights;
    HIP_CHECK(hipMalloc(&d_input, sizeof(h_input)));
    HIP_CHECK(hipMalloc(&d_weights, sizeof(h_weights)));
    HIP_CHECK(hipMemcpy(d_input, h_input, sizeof(h_input), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weights, h_weights, sizeof(h_weights), hipMemcpyHostToDevice));

    uint32_t *d_samples;
    size_t bytes = N_WAVES * 8 * sizeof(uint32_t);
    HIP_CHECK(hipMalloc(&d_samples, bytes));
    unsigned int *d_atomic;
    HIP_CHECK(hipMalloc(&d_atomic, sizeof(unsigned int) * N_WAVES));

    // Output file: header + (n_reps * N_WAVES * 8 * uint32_t)
    FILE* f = fopen(out_path, "wb");
    if (!f) { perror("fopen"); return 1; }
    uint32_t hdr[4] = { (uint32_t)n_reps, (uint32_t)N_WAVES, 8u, 0x50554631u /* 'PUF1' */ };
    fwrite(hdr, sizeof(uint32_t), 4, f);

    // Launch config: 4 waves per block * 256 blocks = 1024 waves total.
    int waves_per_block = 4;
    int threads_per_block = waves_per_block * WAVE_SIZE;
    int blocks = N_WAVES / waves_per_block;

    for (int r = 0; r < n_reps; r++) {
        HIP_CHECK(hipMemset(d_samples, 0, bytes));
        HIP_CHECK(hipMemset(d_atomic, 0, sizeof(unsigned int) * N_WAVES));

        hipLaunchKernelGGL(puf_kernel,
                           dim3(blocks), dim3(threads_per_block), 0, 0,
                           d_samples, d_input, d_weights, d_atomic);
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<uint32_t> h_samples(N_WAVES * 8);
        HIP_CHECK(hipMemcpy(h_samples.data(), d_samples, bytes, hipMemcpyDeviceToHost));
        fwrite(h_samples.data(), sizeof(uint32_t), h_samples.size(), f);
    }

    fclose(f);
    HIP_CHECK(hipFree(d_samples));
    HIP_CHECK(hipFree(d_atomic));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_weights));
    printf("WROTE %s n_reps=%d n_waves=%d\n", out_path, n_reps, N_WAVES);
    return 0;
}
