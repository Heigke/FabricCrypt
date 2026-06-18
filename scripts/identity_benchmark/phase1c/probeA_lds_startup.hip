/* probeA_lds_startup.hip — LDS startup PUF + FMA-LSB silicon PUF.
 *
 * RDNA3.5 / ROCm zeros LDS on workgroup launch (verified empirically: every
 * LDS read returns 0 even without a prior store). This probe therefore runs
 * BOTH measurements and writes BOTH bit-streams; the harness can use whichever
 * has signal:
 *
 *   (1) LDS_RAW (256 words): direct read of LDS before any store. On ROCm this
 *       will be all-zero -> a NULL on this stack. Recorded for completeness.
 *   (2) FMA_LSB (256 words): deterministic chained FMA, then we record the low
 *       16 bits of the FP32 mantissa per lane across many iterations. Per-CU
 *       FMA rounding takes silicon-bound paths (Becker 2014 active-side-channel
 *       literature; per-lane FMA LSB variation is a documented PUF surface on
 *       GPUs).
 *
 * Output: per-rep payload = LDS_WORDS uint32 (LDS) + LDS_WORDS uint32 (FMA-LSB)
 *         = 512 uint32 / rep.
 *
 * Build:
 *   hipcc --offload-arch=gfx1151 -O1 -o probeA_lds_startup probeA_lds_startup.hip
 *
 * Usage: probeA_lds_startup <reps> <out.bin>
 *   Header: magic 0x4C445331, reps, 512 (words_per_rep), 0
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <thread>

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define LDS_WORDS    256
#define BLOCK_THREADS 64
#define FMA_LANES    256
#define FMA_ITERS    128

__global__ void probeA_kernel(uint32_t* out_lds, uint32_t* out_fma,
                              const float* fixed_seed)
{
    __shared__ uint32_t lds[LDS_WORDS];
    int tid = threadIdx.x;

    // (1) Raw LDS read (no prior store).
    #pragma unroll
    for (int i = 0; i < (LDS_WORDS / BLOCK_THREADS); i++) {
        int idx = tid + i * BLOCK_THREADS;
        out_lds[idx] = lds[idx];
    }

    // (2) FMA LSB capture. Each of FMA_LANES virtual lanes does FMA_ITERS
    //     chained FMAs from the fixed seed; we record the low 16 bits of the
    //     final mantissa.
    #pragma unroll 1
    for (int lane = tid; lane < FMA_LANES; lane += BLOCK_THREADS) {
        float acc = fixed_seed[lane & 1023];
        float k1  = 1.0f + 1.1920929e-7f * (float)((lane * 13) & 31);   // ~ULP-scale
        float k2  = 1.0f - 1.1920929e-7f * (float)((lane * 17) & 31);
        #pragma unroll 4
        for (int i = 0; i < FMA_ITERS; i++) {
            acc = __fmaf_rn(acc, k1, 1.0e-9f);
            acc = __fmaf_rn(acc, k2, -1.0e-9f);
            acc = tanhf(acc * 0.5f) + 0.5f;
        }
        uint32_t bits;
        __builtin_memcpy(&bits, &acc, 4);
        out_fma[lane] = bits & 0xFFFFu;  // low 16 bits = mantissa LSBs
    }
}

int main(int argc, char** argv)
{
    int reps = (argc > 1) ? atoi(argv[1]) : 10000;
    const char* out_path = (argc > 2) ? argv[2] : "probeA.bin";

    HIP_CHECK(hipSetDevice(0));

    // Fixed deterministic seed buffer (1024 floats).
    float seed[1024];
    for (int i = 0; i < 1024; i++) seed[i] = 0.123456789f + 1.0e-3f * (float)i;
    float* dev_seed = nullptr;
    HIP_CHECK(hipMalloc(&dev_seed, 1024 * sizeof(float)));
    HIP_CHECK(hipMemcpy(dev_seed, seed, 1024 * sizeof(float), hipMemcpyHostToDevice));

    size_t words_per_rep = (size_t)LDS_WORDS + (size_t)FMA_LANES;  // 512
    size_t payload_words = (size_t)reps * words_per_rep;
    uint32_t* host = (uint32_t*)malloc((4 + payload_words) * sizeof(uint32_t));
    host[0] = 0x4C445331u;
    host[1] = (uint32_t)reps;
    host[2] = (uint32_t)words_per_rep;
    host[3] = 0;

    uint32_t* dev_lds = nullptr;
    uint32_t* dev_fma = nullptr;
    HIP_CHECK(hipMalloc(&dev_lds, LDS_WORDS * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&dev_fma, FMA_LANES * sizeof(uint32_t)));

    for (int r = 0; r < reps; r++) {
        HIP_CHECK(hipMemset(dev_lds, 0xFF, LDS_WORDS * sizeof(uint32_t)));
        HIP_CHECK(hipMemset(dev_fma, 0xFF, FMA_LANES * sizeof(uint32_t)));
        hipLaunchKernelGGL(probeA_kernel,
                           dim3(1), dim3(BLOCK_THREADS), 0, 0,
                           dev_lds, dev_fma, dev_seed);
        HIP_CHECK(hipDeviceSynchronize());
        size_t off = 4 + (size_t)r * words_per_rep;
        HIP_CHECK(hipMemcpy(host + off,             dev_lds, LDS_WORDS * sizeof(uint32_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(host + off + LDS_WORDS, dev_fma, FMA_LANES * sizeof(uint32_t), hipMemcpyDeviceToHost));
        if ((r % 500) == 499) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    FILE* f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "open %s failed\n", out_path); return 1; }
    fwrite(host, sizeof(uint32_t), 4 + payload_words, f);
    fclose(f);

    HIP_CHECK(hipFree(dev_lds));
    HIP_CHECK(hipFree(dev_fma));
    HIP_CHECK(hipFree(dev_seed));
    free(host);
    printf("[probeA] wrote %s reps=%d words/rep=%zu (LDS=%d + FMA=%d)\n",
           out_path, reps, words_per_rep, LDS_WORDS, FMA_LANES);
    return 0;
}
