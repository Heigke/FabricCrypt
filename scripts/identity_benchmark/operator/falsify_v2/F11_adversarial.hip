/*  F11 — adversarial input.  Same kernel as divergent_matmul.hip,
 *  but the W/x distributions are crafted to maximise rounding instability:
 *   - W rows contain near-cancellation pairs (a, -a + epsilon)
 *   - x has subnormal values mixed with O(1) values to span 38 decades
 *   - alternating sign + heavy tail to maximise reduction-order sensitivity
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <unistd.h>

#define HIP_CHECK(call) do { hipError_t err = (call); \
    if (err != hipSuccess) { fprintf(stderr,"HIP %s\n",hipGetErrorString(err)); exit(1);} } while(0)

#define TBSIZE 256
#define KCHUNK 32

__global__ void f11_kernel(const float* __restrict__ W,
                           const float* __restrict__ x,
                           float*       __restrict__ y,
                           int M, int K)
{
    int tid = threadIdx.x;
    int blocks_k = (K + TBSIZE * KCHUNK - 1) / (TBSIZE * KCHUNK);
    int kblock = blockIdx.x % blocks_k;
    int m      = blockIdx.x / blocks_k;
    if (m >= M) return;
    int k_base = kblock * TBSIZE * KCHUNK + tid * KCHUNK;
    float acc_fma = 0.0f, acc_split = 0.0f;
    #pragma unroll
    for (int i = 0; i < KCHUNK; i++) {
        int k = k_base + i;
        if (k < K) {
            float a = W[m * K + k];
            float b = x[k];
            if ((i & 1) == 0) acc_fma = __fmaf_rn(a, b, acc_fma);
            else { float p = a * b; acc_split = acc_split + p; }
        }
    }
    float partial = acc_fma + acc_split;
    for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor(partial, off);
    if ((tid & 31) == 0) atomicAdd(&y[m], partial);
}

int main(int argc, char** argv)
{
    int M = (argc > 1) ? atoi(argv[1]) : 64;
    int K = (argc > 2) ? atoi(argv[2]) : 4096;
    int REPS = (argc > 3) ? atoi(argv[3]) : 32;
    const char* out_path = (argc > 4) ? argv[4] : "f11_out.bin";

    float* h_W = (float*)malloc(sizeof(float)*M*K);
    float* h_x = (float*)malloc(sizeof(float)*K);

    // Adversarial W: pairs of (a, -a+eps) for near-cancellation.
    // half the entries are near-cancelling pairs, mixed with subnormals.
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            int idx = m*K + k;
            uint32_t s = (uint32_t)((uint32_t)idx * 2654435761u + 1234567u);
            float r = (float)((s >> 8) & 0xFFFF) / 65535.0f - 0.5f;
            if ((k & 1) == 0) {
                h_W[idx] = r * 1.0f;          // O(1)
            } else {
                float eps = (float)((s >> 16) & 0xFF) / 255.0f * 1e-6f;
                h_W[idx] = -h_W[idx-1] + eps;  // near-cancellation
            }
            if (((s >> 24) & 0x1F) == 0) h_W[idx] = h_W[idx] * 1e-38f; // subnormal
        }
    }
    // Adversarial x: alternating O(1) and subnormal
    for (int k = 0; k < K; k++) {
        uint32_t s = (uint32_t)((k+11) * 2246822519u + 7u);
        float r = (float)((s >> 8) & 0xFFFF) / 65535.0f - 0.5f;
        if ((k & 3) == 0)      h_x[k] = r * 1.0f;
        else if ((k & 3) == 1) h_x[k] = r * 1e-38f;  // subnormal
        else if ((k & 3) == 2) h_x[k] = r * 1e3f;    // large but finite
        else                   h_x[k] = -r * 1e-20f; // tiny
    }

    float *d_W, *d_x, *d_y;
    HIP_CHECK(hipMalloc(&d_W, sizeof(float)*M*K));
    HIP_CHECK(hipMalloc(&d_x, sizeof(float)*K));
    HIP_CHECK(hipMalloc(&d_y, sizeof(float)*M));
    HIP_CHECK(hipMemcpy(d_W, h_W, sizeof(float)*M*K, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x, sizeof(float)*K, hipMemcpyHostToDevice));

    int blocks_k = (K + TBSIZE*KCHUNK - 1) / (TBSIZE*KCHUNK);
    int grid = M * blocks_k;
    FILE* f = fopen(out_path, "wb"); if (!f) return 2;
    char hn[64] = {0}; gethostname(hn, 63);
    int32_t hl = strlen(hn), M32 = M, R32 = REPS;
    fwrite(&M32,4,1,f); fwrite(&R32,4,1,f); fwrite(&hl,4,1,f); fwrite(hn,1,hl,f);

    float* h_y = (float*)malloc(sizeof(float)*M);
    for (int r = 0; r < REPS; r++) {
        HIP_CHECK(hipMemset(d_y, 0, sizeof(float)*M));
        hipLaunchKernelGGL(f11_kernel, dim3(grid), dim3(TBSIZE), 0, 0, d_W, d_x, d_y, M, K);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_y, d_y, sizeof(float)*M, hipMemcpyDeviceToHost));
        fwrite(h_y, sizeof(float), M, f);
    }
    fclose(f);
    fprintf(stderr,"[F11] host=%s y[0..2]=%.6e %.6e %.6e\n", hn, h_y[0], h_y[1], h_y[2]);
    free(h_W); free(h_x); free(h_y);
    hipFree(d_W); hipFree(d_x); hipFree(d_y);
    return 0;
}
