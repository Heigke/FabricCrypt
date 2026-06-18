/*  divergent_matmul.hip — Substrate-as-operator C1 smoke-test
 *
 *  Goal: produce bit-divergent output for identical (W, x) on
 *  ikaros vs daedalus by exploiting IEEE-754 implementation freedoms.
 *
 *  Mechanisms stacked:
 *   - Atomic-add reduction (order is wavefront-arrival-dependent,
 *     which depends on per-die contention, scheduler, voltage).
 *   - Subnormal-range inputs (FTZ behaviour can differ per die under
 *     different DPM/Vmin states).
 *   - FMA fusion explicitly via __fmaf_rn vs unfused multiply+add to
 *     create different rounding paths inside the same kernel.
 *   - Reduction tree depth varies with block size; we use a
 *     contention-dependent partial reduction.
 *
 *  Output: y[M] = sum_k W[m,k] * x[k] but accumulated via global
 *  atomicAdd from every threadblock — order is undefined.
 *
 *  Build (gfx1151):
 *    hipcc --offload-arch=gfx1151 -O3 -ffast-math \
 *          -o scripts/identity_benchmark/operator/divergent_matmul \
 *          scripts/identity_benchmark/operator/divergent_matmul.hip
 *
 *  Usage:
 *    divergent_matmul <M> <K> <N_REPS> <out.bin>
 *      writes header(M, REPS, hostname_len, hostname) then
 *      REPS * M floats (y vectors).
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
#define KCHUNK 32     // each thread handles KCHUNK inner elements

// W is M x K row-major. x is K. y is M (must be zeroed before launch).
// Each block (b) handles one inner-tile of K for ALL M rows.
// All blocks atomicAdd into the same y[m] — non-deterministic order.
__global__ void divergent_matmul_kernel(const float* __restrict__ W,
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

    // Local accumulation — half via FMA, half via split mul+add,
    // to ensure two rounding paths appear in the same partial sum.
    float acc_fma   = 0.0f;
    float acc_split = 0.0f;
    #pragma unroll
    for (int i = 0; i < KCHUNK; i++) {
        int k = k_base + i;
        if (k < K) {
            float a = W[m * K + k];
            float b = x[k];
            if ((i & 1) == 0) {
                acc_fma = __fmaf_rn(a, b, acc_fma);   // fused
            } else {
                float prod = a * b;                    // unfused
                acc_split = acc_split + prod;
            }
        }
    }
    float partial = acc_fma + acc_split;

    // Warp-level reduction first (deterministic per wave)
    for (int off = 16; off > 0; off >>= 1) {
        partial += __shfl_xor(partial, off);
    }
    // Then ONE atomicAdd per wave to the global y[m].
    // Order across waves is implementation-defined → per-die divergence.
    if ((tid & 31) == 0) {
        atomicAdd(&y[m], partial);
    }
}

int main(int argc, char** argv)
{
    int M       = (argc > 1) ? atoi(argv[1]) : 64;
    int K       = (argc > 2) ? atoi(argv[2]) : 4096;
    int N_REPS  = (argc > 3) ? atoi(argv[3]) : 32;
    const char* out_path = (argc > 4) ? argv[4] : "divergent_out.bin";

    // Deterministic fixed inputs (NO RNG, identical across hosts)
    float* h_W = (float*)malloc(sizeof(float) * M * K);
    float* h_x = (float*)malloc(sizeof(float) * K);
    for (int i = 0; i < M * K; i++) {
        uint32_t s = (uint32_t)(i * 1103515245u + 12345u);
        float v = (float)((s >> 8) & 0xFFFF) / 65535.0f - 0.5f;
        // Inject some subnormal-range values (~1% of entries)
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

    // Write header + REPS results
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
        hipLaunchKernelGGL(divergent_matmul_kernel,
                           dim3(grid), dim3(TBSIZE), 0, 0,
                           d_W, d_x, d_y, M, K);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_y, d_y, sizeof(float) * M, hipMemcpyDeviceToHost));
        fwrite(h_y, sizeof(float), M, f);
    }
    fclose(f);

    // Print first vector to stderr for quick eyeballing
    fprintf(stderr, "host=%s M=%d K=%d REPS=%d  y[0..4]= %.8e %.8e %.8e %.8e %.8e\n",
            hostname, M, K, N_REPS, h_y[0], h_y[1], h_y[2], h_y[3], h_y[4]);

    free(h_W); free(h_x); free(h_y);
    hipFree(d_W); hipFree(d_x); hipFree(d_y);
    return 0;
}
