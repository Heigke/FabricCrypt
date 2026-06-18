/*  F5 — flag isolation. Same kernel as divergent_matmul.hip but the
 *  build-time flag matrix is controlled by the compile command.
 *  We use macros so the SAME source compiles N variants by toggling:
 *    -DF5_DISABLE_ATOMICS  : replace atomicAdd with deterministic per-row reduction
 *    -DF5_DISABLE_FMA      : force unfused multiply-add only
 *    -DF5_STRICT_DENORM    : use compile-time -fno-fast-math, daz_opt_on
 *    -DF5_NO_FAST_MATH     : (handled at command-line only)
 *
 *  Build matrix is driven by F5_run.sh.
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <unistd.h>

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define TBSIZE 256
#define KCHUNK 32

__global__ void f5_kernel(const float* __restrict__ W,
                          const float* __restrict__ x,
                          float*       __restrict__ y,
                          int M, int K)
{
    int tid    = threadIdx.x;
    int blocks_k = (K + TBSIZE * KCHUNK - 1) / (TBSIZE * KCHUNK);
    int kblock = blockIdx.x % blocks_k;
    int m      = blockIdx.x / blocks_k;
    if (m >= M) return;

    int k_base = kblock * TBSIZE * KCHUNK + tid * KCHUNK;

    float acc_fma   = 0.0f;
    float acc_split = 0.0f;
    #pragma unroll
    for (int i = 0; i < KCHUNK; i++) {
        int k = k_base + i;
        if (k < K) {
            float a = W[m * K + k];
            float b = x[k];
            if ((i & 1) == 0) {
#ifdef F5_DISABLE_FMA
                float prod = a * b;
                acc_fma = acc_fma + prod;
#else
                acc_fma = __fmaf_rn(a, b, acc_fma);
#endif
            } else {
                float prod = a * b;
                acc_split = acc_split + prod;
            }
        }
    }
    float partial = acc_fma + acc_split;

    for (int off = 16; off > 0; off >>= 1) {
        partial += __shfl_xor(partial, off);
    }
    if ((tid & 31) == 0) {
#ifdef F5_DISABLE_ATOMICS
        // Deterministic single-writer: only block 0 writes y[m], others skip.
        // (Loses some sum precision but eliminates atomic ordering.)
        // Instead use a stable per-block layout: only the first kblock writes
        // a single sequential sum that covers the entire K.
        if (kblock == 0) {
            // Recompute serial sum for stable order
            float s = 0.f;
            for (int k = 0; k < K; k++) {
                s = __fmaf_rn(W[m*K + k], x[k], s);
            }
            y[m] = s;
        }
#else
        atomicAdd(&y[m], partial);
#endif
    }
}

int main(int argc, char** argv)
{
    int M       = (argc > 1) ? atoi(argv[1]) : 64;
    int K       = (argc > 2) ? atoi(argv[2]) : 4096;
    int N_REPS  = (argc > 3) ? atoi(argv[3]) : 32;
    const char* out_path = (argc > 4) ? argv[4] : "f5_out.bin";

    float* h_W = (float*)malloc(sizeof(float) * M * K);
    float* h_x = (float*)malloc(sizeof(float) * K);
    for (int i = 0; i < M * K; i++) {
        uint32_t s = (uint32_t)(i * 1103515245u + 12345u);
        float v = (float)((s >> 8) & 0xFFFF) / 65535.0f - 0.5f;
        if (((s >> 24) & 0xFF) < 3) v *= 1e-38f;
        h_W[i] = v;
    }
    for (int i = 0; i < K; i++) {
        uint32_t s = (uint32_t)((i + 7) * 22695477u + 1u);
        h_x[i] = (float)((s >> 8) & 0xFFFF) / 65535.0f - 0.5f;
    }

    float *d_W, *d_x, *d_y;
    HIP_CHECK(hipMalloc(&d_W, sizeof(float) * M * K));
    HIP_CHECK(hipMalloc(&d_x, sizeof(float) * K));
    HIP_CHECK(hipMalloc(&d_y, sizeof(float) * M));
    HIP_CHECK(hipMemcpy(d_W, h_W, sizeof(float) * M * K, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x, sizeof(float) * K, hipMemcpyHostToDevice));

    int blocks_k = (K + TBSIZE * KCHUNK - 1) / (TBSIZE * KCHUNK);
    int grid = M * blocks_k;

    FILE* f = fopen(out_path, "wb");
    if (!f) { perror(out_path); return 2; }
    char hostname[64] = {0};
    gethostname(hostname, 63);
    int32_t hl = (int32_t)strlen(hostname);
    int32_t M32 = M, R32 = N_REPS;
    fwrite(&M32, sizeof(int32_t), 1, f);
    fwrite(&R32, sizeof(int32_t), 1, f);
    fwrite(&hl,  sizeof(int32_t), 1, f);
    fwrite(hostname, 1, hl, f);

    float* h_y = (float*)malloc(sizeof(float) * M);
    for (int r = 0; r < N_REPS; r++) {
        HIP_CHECK(hipMemset(d_y, 0, sizeof(float) * M));
        hipLaunchKernelGGL(f5_kernel,
                           dim3(grid), dim3(TBSIZE), 0, 0,
                           d_W, d_x, d_y, M, K);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_y, d_y, sizeof(float) * M, hipMemcpyDeviceToHost));
        fwrite(h_y, sizeof(float), M, f);
    }
    fclose(f);
    fprintf(stderr, "[F5] host=%s out=%s y[0]=%.8e\n", hostname, out_path, h_y[0]);

    free(h_W); free(h_x); free(h_y);
    hipFree(d_W); hipFree(d_x); hipFree(d_y);
    return 0;
}
