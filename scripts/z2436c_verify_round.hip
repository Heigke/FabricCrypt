/**
 * z2436c: Verify s_setreg round_diff as OOD detector
 * Print actual values so we can see the distribution
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Compute round_diff = sum(activations, R+inf) - sum(activations, RNE)
__global__ void compute_round_diff(
    const float* __restrict__ X, const float* __restrict__ W, const float* __restrict__ b,
    float* __restrict__ round_diffs,  // [N] one per sample
    int N, int in_dim, int out_dim
) {
    int sample = blockIdx.x;
    if (sample >= N) return;
    int tid = threadIdx.x;

    // First: compute activations (layer 1)
    __shared__ float s_act[128];
    if (tid < out_dim) {
        float sum = b[tid];
        for (int k = 0; k < in_dim; k++) sum += W[tid*in_dim+k] * X[sample*in_dim+k];
        s_act[tid] = (sum > 0) ? sum : 0;
    }
    __syncthreads();

    // Sum in RNE mode (default)
    float sum_rne = 0;
    for (int j = tid; j < out_dim; j += blockDim.x) sum_rne += s_act[j];

    // Sum in R+inf mode
    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x1" ::: "memory");
    float sum_rinf = 0;
    for (int j = tid; j < out_dim; j += blockDim.x) sum_rinf += s_act[j];
    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x0" ::: "memory");

    if (tid == 0) {
        round_diffs[sample] = sum_rinf - sum_rne;
    }
}

float* load_bin(const char* p, int n) {
    FILE* f=fopen(p,"rb"); float* d=(float*)malloc(n*4); fread(d,4,n,f); fclose(f); return d;
}

int main() {
    const char* B = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy";
    char p[512];

    sprintf(p,"%s/data/MNIST/raw/t10k-images-idx3-ubyte",B);
    FILE* fi=fopen(p,"rb"); unsigned char hd[16]; fread(hd,1,16,fi);
    int NM=10000;
    unsigned char* raw=(unsigned char*)malloc(NM*784);
    fread(raw,1,NM*784,fi); fclose(fi);
    float* h_mnist=(float*)malloc(NM*784*4);
    for(int i=0;i<NM*784;i++) h_mnist[i]=raw[i]/255.0f;
    free(raw);

    sprintf(p,"%s/models/mnist_mlp/w1.bin",B); float* hw1=load_bin(p,128*784);
    sprintf(p,"%s/models/mnist_mlp/b1.bin",B); float* hb1=load_bin(p,128);

    int N_ID=500, N_OOD=500;

    // Generate inputs
    float* h_uniform=(float*)malloc(N_OOD*784*4);
    float* h_const1=(float*)malloc(N_OOD*784*4);
    float* h_invert=(float*)malloc(N_OOD*784*4);
    srand(42);
    for(int i=0;i<N_OOD*784;i++) {
        h_uniform[i]=(float)rand()/RAND_MAX;
        h_const1[i]=1.0f;
        h_invert[i]=1.0f-h_mnist[i];
    }

    float *di, *dw1, *db1, *drd;
    hipMalloc(&di,N_ID*784*4);
    hipMalloc(&dw1,128*784*4); hipMalloc(&db1,128*4);
    hipMalloc(&drd,N_ID*4);
    hipMemcpy(dw1,hw1,128*784*4,hipMemcpyHostToDevice);
    hipMemcpy(db1,hb1,128*4,hipMemcpyHostToDevice);

    float* h_rd=(float*)malloc(N_ID*4);

    auto run = [&](float* h_input, int N, const char* name) {
        hipMemcpy(di, h_input, N*784*4, hipMemcpyHostToDevice);
        compute_round_diff<<<N, 128>>>(di, dw1, db1, drd, N, 784, 128);
        hipDeviceSynchronize();
        hipMemcpy(h_rd, drd, N*4, hipMemcpyDeviceToHost);

        double sum=0, sq=0, mn=1e30, mx=-1e30;
        int n_zero=0;
        for(int i=0;i<N;i++) {
            sum+=h_rd[i]; sq+=h_rd[i]*h_rd[i];
            if(h_rd[i]<mn) mn=h_rd[i];
            if(h_rd[i]>mx) mx=h_rd[i];
            if(h_rd[i]==0.0f) n_zero++;
        }
        double mean=sum/N, std=sqrt(sq/N-mean*mean);
        printf("  %-12s: mean=%12.8f std=%12.8f min=%12.8f max=%12.8f zeros=%d/%d\n",
               name, mean, std, mn, mx, n_zero, N);
    };

    printf("s_setreg round_diff (sum_R+inf - sum_RNE) on layer 1 activations:\n\n");
    run(h_mnist, N_ID, "MNIST (ID)");
    run(h_uniform, N_OOD, "Uniform");
    run(h_const1, N_OOD, "Constant 1");
    run(h_invert, N_OOD, "Inverted");

    free(h_mnist);free(h_uniform);free(h_const1);free(h_invert);
    free(hw1);free(hb1);free(h_rd);
    hipFree(di);hipFree(dw1);hipFree(db1);hipFree(drd);
    return 0;
}
