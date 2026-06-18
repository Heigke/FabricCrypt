/*  divergent_matmul_DET.hip — F1 deterministic control
 *
 *  Same M,K,inputs as divergent_matmul, but ALL non-determinism removed:
 *   - One thread per output row m, serial inner reduction over K
 *   - No atomicAdd, no warp shuffle, no cross-wave reduction
 *   - Build flags: -ffp-contract=off -fdenormal-fp-math=ieee  -O2 (no -ffast-math)
 *   - Same FMA-mix pattern (alt fma vs unfused mul+add) so the *math*
 *     is identical to divergent kernel — just executed serially per row.
 *
 *  If math itself is hardware-invariant, ikaros and daedalus must produce
 *  bit-identical outputs. Cross-chip diff > 0 => math diverges => kill track.
 *
 *  Build:
 *    hipcc --offload-arch=gfx1151 -O2 \
 *          -ffp-contract=off -fdenormal-fp-math=ieee \
 *          -o divergent_matmul_DET divergent_matmul_DET.hip
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

// One block, one thread per row m. Serial inner reduction. No atomics, no shuffles.
__global__ void det_matmul_kernel(const float* __restrict__ W,
                                  const float* __restrict__ x,
                                  float*       __restrict__ y,
                                  int M, int K)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    float acc_fma   = 0.0f;
    float acc_split = 0.0f;
    for (int k = 0; k < K; k++) {
        float a = W[m * K + k];
        float b = x[k];
        if ((k & 1) == 0) {
            acc_fma = __fmaf_rn(a, b, acc_fma);
        } else {
            float prod = a * b;
            acc_split = acc_split + prod;
        }
    }
    y[m] = acc_fma + acc_split;
}

int main(int argc, char** argv)
{
    int M       = (argc > 1) ? atoi(argv[1]) : 64;
    int K       = (argc > 2) ? atoi(argv[2]) : 4096;
    int N_REPS  = (argc > 3) ? atoi(argv[3]) : 32;
    const char* out_path = (argc > 4) ? argv[4] : "divergent_out_DET.bin";

    // IDENTICAL deterministic input generation as divergent_matmul.hip
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

    int TB = 256;
    int grid = (M + TB - 1) / TB;

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
        hipLaunchKernelGGL(det_matmul_kernel,
                           dim3(grid), dim3(TB), 0, 0,
                           d_W, d_x, d_y, M, K);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_y, d_y, sizeof(float) * M, hipMemcpyDeviceToHost));
        fwrite(h_y, sizeof(float), M, f);
    }
    fclose(f);
    fprintf(stderr, "host=%s M=%d K=%d REPS=%d  y[0..4]= %.8e %.8e %.8e %.8e %.8e\n",
            hostname, M, K, N_REPS, h_y[0], h_y[1], h_y[2], h_y[3], h_y[4]);

    free(h_W); free(h_x); free(h_y);
    hipFree(d_W); hipFree(d_x); hipFree(d_y);
    return 0;
}
