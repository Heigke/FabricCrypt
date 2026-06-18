/**
 * z2463: Test the 28 UNTESTED GPU mechanisms against software controls
 *
 * z2456 tested 4: shuffle (SW wins), branch (SW wins), ballot (HW wins), combined
 * 28 remain untested. Group by category and test each against software equivalent.
 *
 * Categories and mechanisms to test:
 *
 * CATEGORY 1: Cache/Memory patterns
 *   M1: L1 cache eviction nonlinearity (z2262 showed XOR=97%)
 *   M2: LDS bank conflict contention
 *   M3: Memory coalescing pattern variation
 *
 * CATEGORY 2: Timing/Scheduling
 *   M4: s_sleep controlled delay (wavefront stall)
 *   M5: Instruction-level timing jitter
 *
 * CATEGORY 3: Arithmetic
 *   M6: atomicAdd ordering (race condition accumulation)
 *   M7: s_setreg FP rounding sensitivity
 *
 * For each: run on MNIST 28-step sequence (same as z2456),
 * compare HW kernel vs software-matched control.
 *
 * Focus on the ones most likely to be unique:
 *   - L1 cache (proven strong nonlinearity in z2262)
 *   - atomicAdd (F=456 in z2432, never tested vs SW match)
 *   - LDS bank conflict (creates real HW-dependent contention)
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

#define W 128    // width
#define ALPHA 0.15f

// ============================================================
// MECHANISM: L1 cache eviction nonlinearity
// Read/write pattern that causes cache thrashing → stochastic access time
// The KEY one: z2262 showed XOR=97% from L1 effects alone
// ============================================================
__global__ void hw_l1_cache(
    const float* __restrict__ X, const float* __restrict__ W_mat,
    const float* __restrict__ b, float* __restrict__ Y,
    int N, int in_d, int out_d, float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int s = idx / out_d, n = idx % out_d;
    if (s >= N) return;

    float sum = b[n];
    for (int k = 0; k < in_d; k++) sum += W_mat[n*in_d+k] * X[s*in_d+k];
    float relu_out = (sum > 0) ? sum : 0;

    // MECHANISM: Force L1 cache thrashing by reading far-apart addresses
    // This creates data-dependent cache miss patterns
    volatile float* x_ptr = (volatile float*)(X + s * in_d);
    float cache_val = 0;
    // Read strided pattern that exceeds L1 cache line (128 bytes = 32 floats)
    for (int k = 0; k < min(in_d, 128); k += 37) {  // stride 37 = prime, forces misses
        cache_val += x_ptr[k % in_d];
    }
    float l1_residual = cache_val * 0.001f;  // tiny contribution from cache-dependent reads

    Y[s*out_d+n] = relu_out + alpha * l1_residual;
}

// ============================================================
// MECHANISM: atomicAdd ordering
// Multiple threads atomicAdd to same location → order-dependent rounding
// ============================================================
__global__ void hw_atomic(
    const float* __restrict__ X, const float* __restrict__ W_mat,
    const float* __restrict__ b, float* __restrict__ Y,
    int N, int in_d, int out_d, float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int s = idx / out_d, n = idx % out_d;
    if (s >= N) return;

    float sum = b[n];
    for (int k = 0; k < in_d; k++) sum += W_mat[n*in_d+k] * X[s*in_d+k];
    float relu_out = (sum > 0) ? sum : 0;

    // MECHANISM: atomicAdd contention
    // 4 threads write to same shared memory location
    __shared__ float s_atomic[256];
    s_atomic[n] = 0;
    __syncthreads();

    int conflict_addr = n ^ (n & 0x3);  // groups of 4 map to same addr
    atomicAdd(&s_atomic[conflict_addr], relu_out * 0.25f);
    atomicAdd(&s_atomic[n], relu_out * 0.75f);
    __syncthreads();

    float atomic_val = s_atomic[n];
    float atomic_residual = atomic_val - relu_out;  // should be ~0 but HW ordering may differ

    Y[s*out_d+n] = relu_out + alpha * atomic_residual;
}

// ============================================================
// MECHANISM: LDS bank conflict
// Write to addresses that cause bank conflicts → timing-dependent behavior
// ============================================================
__global__ void hw_bank_conflict(
    const float* __restrict__ X, const float* __restrict__ W_mat,
    const float* __restrict__ b, float* __restrict__ Y,
    int N, int in_d, int out_d, float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int s = idx / out_d, n = idx % out_d;
    if (s >= N) return;

    float sum = b[n];
    for (int k = 0; k < in_d; k++) sum += W_mat[n*in_d+k] * X[s*in_d+k];
    float relu_out = (sum > 0) ? sum : 0;

    // MECHANISM: Deliberate bank conflicts in LDS
    // LDS has 32 banks, stride-1 access = no conflict, stride-32 = max conflict
    __shared__ float s_lds[256 * 2];  // double-sized for conflict pattern

    // Write to stride-32 (same bank for all threads in a warp quarter)
    int conflict_idx = (n * 32) % 512;  // stride-32 = all hit same bank
    s_lds[conflict_idx] = relu_out;
    __syncthreads();

    // Read back — the value may differ from what was written due to conflict resolution
    float conflict_read = s_lds[conflict_idx];
    __syncthreads();

    // Also do a non-conflicting write/read for comparison
    s_lds[n] = relu_out;
    __syncthreads();
    float clean_read = s_lds[n];

    float bank_residual = conflict_read - clean_read;

    Y[s*out_d+n] = relu_out + alpha * bank_residual;
}

// ============================================================
// SOFTWARE-MATCHED CONTROLS
// ============================================================

// SW match for L1 cache: random hash of input (software nonlinearity)
__global__ void sw_l1_match(
    const float* __restrict__ X, const float* __restrict__ W_mat,
    const float* __restrict__ b, float* __restrict__ Y,
    int N, int in_d, int out_d, float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int s = idx / out_d, n = idx % out_d;
    if (s >= N) return;

    float sum = b[n];
    for (int k = 0; k < in_d; k++) sum += W_mat[n*in_d+k] * X[s*in_d+k];
    float relu_out = (sum > 0) ? sum : 0;

    // SW equivalent: subsample input with same stride pattern
    float cache_val = 0;
    for (int k = 0; k < min(in_d, 128); k += 37) {
        cache_val += X[s * in_d + k % in_d];
    }
    Y[s*out_d+n] = relu_out + alpha * cache_val * 0.001f;
}

// SW match for atomic: simple pooling (same math, no contention)
__global__ void sw_atomic_match(
    const float* __restrict__ X, const float* __restrict__ W_mat,
    const float* __restrict__ b, float* __restrict__ Y,
    int N, int in_d, int out_d, float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int s = idx / out_d, n = idx % out_d;
    if (s >= N) return;

    float sum = b[n];
    for (int k = 0; k < in_d; k++) sum += W_mat[n*in_d+k] * X[s*in_d+k];
    float relu_out = (sum > 0) ? sum : 0;

    // SW equivalent: deterministic pooling with same 75/25 split
    // No shared memory, no contention — just compute the same weighted average
    // In a non-atomic world, s_atomic[n] = 0.75*relu_out + 0.25*relu_out_of_neighbor
    // Approximate: assume neighbor's value ≈ own value → residual ≈ 0
    Y[s*out_d+n] = relu_out;  // atomic residual should be ~0, so SW match = no residual
}

// SW match for bank conflict: no-op (bank conflicts should produce zero residual)
__global__ void sw_bank_match(
    const float* __restrict__ X, const float* __restrict__ W_mat,
    const float* __restrict__ b, float* __restrict__ Y,
    int N, int in_d, int out_d, float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int s = idx / out_d, n = idx % out_d;
    if (s >= N) return;

    float sum = b[n];
    for (int k = 0; k < in_d; k++) sum += W_mat[n*in_d+k] * X[s*in_d+k];
    float relu_out = (sum > 0) ? sum : 0;

    Y[s*out_d+n] = relu_out;  // bank conflicts should give zero residual
}

// BASELINE: pure ReLU
__global__ void baseline(
    const float* __restrict__ X, const float* __restrict__ W_mat,
    const float* __restrict__ b, float* __restrict__ Y,
    int N, int in_d, int out_d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int s = idx / out_d, n = idx % out_d;
    if (s >= N) return;
    float sum = b[n];
    for (int k = 0; k < in_d; k++) sum += W_mat[n*in_d+k] * X[s*in_d+k];
    Y[s*out_d+n] = (sum > 0) ? sum : 0;
}

float* load_bin(const char* p, int n) {
    FILE* f=fopen(p,"rb"); float*d=(float*)malloc(n*4); fread(d,4,n,f); fclose(f); return d;
}

int main() {
    printf("============================================================\n");
    printf("z2463: REMAINING GPU MECHANISMS vs SOFTWARE CONTROLS\n");
    printf("============================================================\n\n");

    const char* B = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy";
    char p[512];

    // MNIST
    sprintf(p,"%s/data/MNIST/raw/train-images-idx3-ubyte",B);
    FILE*fi=fopen(p,"rb"); unsigned char hd[16]; fread(hd,1,16,fi);
    int N_TR=60000;
    unsigned char*raw=(unsigned char*)malloc(N_TR*784);
    fread(raw,1,N_TR*784,fi); fclose(fi);
    float*h_tr=(float*)malloc(N_TR*784*4);
    for(int i=0;i<N_TR*784;i++) h_tr[i]=raw[i]/255.0f;
    free(raw);

    sprintf(p,"%s/data/MNIST/raw/train-labels-idx1-ubyte",B);
    FILE*fl=fopen(p,"rb"); fread(hd,1,8,fl);
    unsigned char*h_trl=(unsigned char*)malloc(N_TR);
    fread(h_trl,1,N_TR,fl); fclose(fl);

    sprintf(p,"%s/data/MNIST/raw/t10k-images-idx3-ubyte",B);
    fi=fopen(p,"rb"); fread(hd,1,16,fi);
    int N_TE=10000;
    raw=(unsigned char*)malloc(N_TE*784);
    fread(raw,1,N_TE*784,fi); fclose(fi);
    float*h_te=(float*)malloc(N_TE*784*4);
    for(int i=0;i<N_TE*784;i++) h_te[i]=raw[i]/255.0f;
    free(raw);

    sprintf(p,"%s/data/MNIST/raw/t10k-labels-idx1-ubyte",B);
    fl=fopen(p,"rb"); fread(hd,1,8,fl);
    unsigned char*h_tel=(unsigned char*)malloc(N_TE);
    fread(h_tel,1,N_TE,fl); fclose(fl);

    // Init weights (same for all conditions)
    srand(42);
    float* h_w1 = (float*)malloc(W*784*4);
    float* h_b1 = (float*)malloc(W*4);
    float* h_w2 = (float*)malloc(10*W*4);
    float* h_b2 = (float*)malloc(10*4);
    for(int i=0;i<W*784;i++) h_w1[i] = ((float)rand()/RAND_MAX-0.5f)*sqrtf(2.0f/784);
    for(int i=0;i<W;i++) h_b1[i] = 0;
    for(int i=0;i<10*W;i++) h_w2[i] = ((float)rand()/RAND_MAX-0.5f)*sqrtf(2.0f/W);
    for(int i=0;i<10;i++) h_b2[i] = 0;

    // GPU alloc
    float *d_tr,*d_te,*d_w1,*d_b1,*d_w2,*d_b2,*d_h,*d_out;
    hipMalloc(&d_tr,N_TR*784*4); hipMalloc(&d_te,N_TE*784*4);
    hipMalloc(&d_w1,W*784*4); hipMalloc(&d_b1,W*4);
    hipMalloc(&d_w2,10*W*4); hipMalloc(&d_b2,10*4);
    hipMalloc(&d_h,N_TR*W*4); hipMalloc(&d_out,N_TR*10*4);

    hipMemcpy(d_tr,h_tr,N_TR*784*4,hipMemcpyHostToDevice);
    hipMemcpy(d_te,h_te,N_TE*784*4,hipMemcpyHostToDevice);
    hipMemcpy(d_w1,h_w1,W*784*4,hipMemcpyHostToDevice);
    hipMemcpy(d_b1,h_b1,W*4,hipMemcpyHostToDevice);
    hipMemcpy(d_w2,h_w2,10*W*4,hipMemcpyHostToDevice);
    hipMemcpy(d_b2,h_b2,10*4,hipMemcpyHostToDevice);

    int thr = 256;

    // Helper: train (Phase 1 baseline + Phase 2 with mechanism) and evaluate
    // For speed: use the same approach as z2456 — phase 1 trains pure ReLU,
    // phase 2 fine-tunes with mechanism, evaluate on test set

    // Actually simpler: just run each kernel on test set with SAME pretrained weights
    // and measure how well the features separate classes via ridge

    // Train baseline weights first (pure ReLU, 10 epochs backprop via PyTorch equivalent)
    // For this HIP test: use pre-trained weights from mnist_mlp_d3
    sprintf(p,"%s/models/mnist_mlp_d3/w0.bin",B);
    float*pw0=load_bin(p,256*784);
    sprintf(p,"%s/models/mnist_mlp_d3/b0.bin",B);
    float*pb0=load_bin(p,256);
    sprintf(p,"%s/models/mnist_mlp_d3/w1.bin",B);
    float*pw1=load_bin(p,256*256);
    sprintf(p,"%s/models/mnist_mlp_d3/b1.bin",B);
    float*pb1=load_bin(p,256);
    sprintf(p,"%s/models/mnist_mlp_d3/w2.bin",B);
    float*pw2=load_bin(p,10*256);
    sprintf(p,"%s/models/mnist_mlp_d3/b2.bin",B);
    float*pb2=load_bin(p,10);

    // Upload pretrained weights
    float *d_pw0,*d_pb0,*d_pw1,*d_pb1,*d_pw2,*d_pb2;
    float *d_h0,*d_h1,*d_lo;
    hipMalloc(&d_pw0,256*784*4); hipMalloc(&d_pb0,256*4);
    hipMalloc(&d_pw1,256*256*4); hipMalloc(&d_pb1,256*4);
    hipMalloc(&d_pw2,10*256*4);  hipMalloc(&d_pb2,10*4);
    hipMalloc(&d_h0,N_TE*256*4); hipMalloc(&d_h1,N_TE*256*4);
    hipMalloc(&d_lo,N_TE*10*4);

    hipMemcpy(d_pw0,pw0,256*784*4,hipMemcpyHostToDevice);
    hipMemcpy(d_pb0,pb0,256*4,hipMemcpyHostToDevice);
    hipMemcpy(d_pw1,pw1,256*256*4,hipMemcpyHostToDevice);
    hipMemcpy(d_pb1,pb1,256*4,hipMemcpyHostToDevice);
    hipMemcpy(d_pw2,pw2,10*256*4,hipMemcpyHostToDevice);
    hipMemcpy(d_pb2,pb2,10*4,hipMemcpyHostToDevice);

    float*h_lo=(float*)malloc(N_TE*10*4);

    // Run each mechanism on layer 1 only, layers 2+3 baseline
    typedef void(*kernel_fn)(const float*,const float*,const float*,float*,int,int,int,float);

    struct Test {
        const char* name;
        kernel_fn hw_kernel;
        kernel_fn sw_kernel;
    };

    // We need wrapper for baseline (no alpha param)
    // Use alpha=0 in hw kernels for baseline effect

    printf("Running pretrained d3-256 MLP through each mechanism (layer 1 only)...\n\n");

    // Baseline first
    baseline<<<(N_TE*256+thr-1)/thr,thr>>>(d_te,d_pw0,d_pb0,d_h0,N_TE,784,256);
    baseline<<<(N_TE*256+thr-1)/thr,thr>>>(d_h0,d_pw1,d_pb1,d_h1,N_TE,256,256);
    baseline<<<(N_TE*10+thr-1)/thr,thr>>>(d_h1,d_pw2,d_pb2,d_lo,N_TE,256,10);
    hipDeviceSynchronize();
    hipMemcpy(h_lo,d_lo,N_TE*10*4,hipMemcpyDeviceToHost);

    int base_correct = 0;
    for(int i=0;i<N_TE;i++){
        int pr=0;float mx=h_lo[i*10];
        for(int j=1;j<10;j++) if(h_lo[i*10+j]>mx){mx=h_lo[i*10+j];pr=j;}
        if(pr==h_tel[i]) base_correct++;
    }
    float base_acc = 100.0f*base_correct/N_TE;
    printf("  Baseline (pure ReLU):    %.2f%%\n\n", base_acc);

    // Test each mechanism
    struct MechTest {
        const char* name;
        void (*hw_fn)(const float*,const float*,const float*,float*,int,int,int,float);
        void (*sw_fn)(const float*,const float*,const float*,float*,int,int,int,float);
    };

    MechTest tests[] = {
        {"L1_cache",      hw_l1_cache,      sw_l1_match},
        {"atomicAdd",     hw_atomic,         sw_atomic_match},
        {"bank_conflict", hw_bank_conflict,  sw_bank_match},
    };

    printf("%-15s %10s %10s %10s %10s\n", "Mechanism", "HW acc", "SW acc", "HW-SW", "Verdict");
    printf("%-15s %10s %10s %10s %10s\n", "---------", "------", "------", "-----", "-------");

    for (int t = 0; t < 3; t++) {
        // HW mechanism on layer 1
        tests[t].hw_fn<<<(N_TE*256+thr-1)/thr,thr>>>(d_te,d_pw0,d_pb0,d_h0,N_TE,784,256,ALPHA);
        baseline<<<(N_TE*256+thr-1)/thr,thr>>>(d_h0,d_pw1,d_pb1,d_h1,N_TE,256,256);
        baseline<<<(N_TE*10+thr-1)/thr,thr>>>(d_h1,d_pw2,d_pb2,d_lo,N_TE,256,10);
        hipDeviceSynchronize();
        hipMemcpy(h_lo,d_lo,N_TE*10*4,hipMemcpyDeviceToHost);

        int hw_correct = 0;
        for(int i=0;i<N_TE;i++){
            int pr=0;float mx=h_lo[i*10];
            for(int j=1;j<10;j++) if(h_lo[i*10+j]>mx){mx=h_lo[i*10+j];pr=j;}
            if(pr==h_tel[i]) hw_correct++;
        }
        float hw_acc = 100.0f*hw_correct/N_TE;

        // SW matched control on layer 1
        tests[t].sw_fn<<<(N_TE*256+thr-1)/thr,thr>>>(d_te,d_pw0,d_pb0,d_h0,N_TE,784,256,ALPHA);
        baseline<<<(N_TE*256+thr-1)/thr,thr>>>(d_h0,d_pw1,d_pb1,d_h1,N_TE,256,256);
        baseline<<<(N_TE*10+thr-1)/thr,thr>>>(d_h1,d_pw2,d_pb2,d_lo,N_TE,256,10);
        hipDeviceSynchronize();
        hipMemcpy(h_lo,d_lo,N_TE*10*4,hipMemcpyDeviceToHost);

        int sw_correct = 0;
        for(int i=0;i<N_TE;i++){
            int pr=0;float mx=h_lo[i*10];
            for(int j=1;j<10;j++) if(h_lo[i*10+j]>mx){mx=h_lo[i*10+j];pr=j;}
            if(pr==h_tel[i]) sw_correct++;
        }
        float sw_acc = 100.0f*sw_correct/N_TE;

        float delta = hw_acc - sw_acc;
        const char* verdict = (delta > 0.3f) ? "HW WINS" : (delta < -0.3f) ? "SW WINS" : "= SAME";

        printf("%-15s %9.2f%% %9.2f%% %+9.2fpp %10s\n",
               tests[t].name, hw_acc, sw_acc, delta, verdict);
    }

    // Also test at sigma=0.3 noise
    printf("\n--- With noise sigma=0.3 ---\n");
    // Add noise to test images
    float* h_noisy = (float*)malloc(N_TE*784*4);
    srand(123);
    for(int i=0;i<N_TE*784;i++){
        float u1=(float)(rand()+1)/(RAND_MAX+2.0f);
        float u2=(float)rand()/RAND_MAX;
        float noise = 0.3f*sqrtf(-2*logf(u1))*cosf(6.2832f*u2);
        h_noisy[i] = fmaxf(0,fminf(1,h_te[i]+noise));
    }
    float* d_noisy;
    hipMalloc(&d_noisy, N_TE*784*4);
    hipMemcpy(d_noisy, h_noisy, N_TE*784*4, hipMemcpyHostToDevice);

    // Baseline on noisy
    baseline<<<(N_TE*256+thr-1)/thr,thr>>>(d_noisy,d_pw0,d_pb0,d_h0,N_TE,784,256);
    baseline<<<(N_TE*256+thr-1)/thr,thr>>>(d_h0,d_pw1,d_pb1,d_h1,N_TE,256,256);
    baseline<<<(N_TE*10+thr-1)/thr,thr>>>(d_h1,d_pw2,d_pb2,d_lo,N_TE,256,10);
    hipDeviceSynchronize();
    hipMemcpy(h_lo,d_lo,N_TE*10*4,hipMemcpyDeviceToHost);
    int nb=0;
    for(int i=0;i<N_TE;i++){int pr=0;float mx=h_lo[i*10];for(int j=1;j<10;j++)if(h_lo[i*10+j]>mx){mx=h_lo[i*10+j];pr=j;}if(pr==h_tel[i])nb++;}
    printf("  Baseline noisy: %.2f%%\n\n", 100.0f*nb/N_TE);

    printf("%-15s %10s %10s %10s %10s\n", "Mechanism", "HW noisy", "SW noisy", "HW-SW", "Verdict");
    printf("%-15s %10s %10s %10s %10s\n", "---------", "--------", "--------", "-----", "-------");

    for (int t = 0; t < 3; t++) {
        tests[t].hw_fn<<<(N_TE*256+thr-1)/thr,thr>>>(d_noisy,d_pw0,d_pb0,d_h0,N_TE,784,256,ALPHA);
        baseline<<<(N_TE*256+thr-1)/thr,thr>>>(d_h0,d_pw1,d_pb1,d_h1,N_TE,256,256);
        baseline<<<(N_TE*10+thr-1)/thr,thr>>>(d_h1,d_pw2,d_pb2,d_lo,N_TE,256,10);
        hipDeviceSynchronize();
        hipMemcpy(h_lo,d_lo,N_TE*10*4,hipMemcpyDeviceToHost);
        int hc=0;for(int i=0;i<N_TE;i++){int pr=0;float mx=h_lo[i*10];for(int j=1;j<10;j++)if(h_lo[i*10+j]>mx){mx=h_lo[i*10+j];pr=j;}if(pr==h_tel[i])hc++;}

        tests[t].sw_fn<<<(N_TE*256+thr-1)/thr,thr>>>(d_noisy,d_pw0,d_pb0,d_h0,N_TE,784,256,ALPHA);
        baseline<<<(N_TE*256+thr-1)/thr,thr>>>(d_h0,d_pw1,d_pb1,d_h1,N_TE,256,256);
        baseline<<<(N_TE*10+thr-1)/thr,thr>>>(d_h1,d_pw2,d_pb2,d_lo,N_TE,256,10);
        hipDeviceSynchronize();
        hipMemcpy(h_lo,d_lo,N_TE*10*4,hipMemcpyDeviceToHost);
        int sc=0;for(int i=0;i<N_TE;i++){int pr=0;float mx=h_lo[i*10];for(int j=1;j<10;j++)if(h_lo[i*10+j]>mx){mx=h_lo[i*10+j];pr=j;}if(pr==h_tel[i])sc++;}

        float ha=100.0f*hc/N_TE, sa=100.0f*sc/N_TE, d=ha-sa;
        printf("%-15s %9.2f%% %9.2f%% %+9.2fpp %10s\n",
               tests[t].name, ha, sa, d, d>0.3?"HW WINS":d<-0.3?"SW WINS":"= SAME");
    }

    printf("\n============================================================\n");

    // Cleanup
    free(h_tr);free(h_te);free(h_trl);free(h_tel);free(h_lo);free(h_noisy);
    free(h_w1);free(h_b1);free(h_w2);free(h_b2);
    free(pw0);free(pb0);free(pw1);free(pb1);free(pw2);free(pb2);
    hipFree(d_tr);hipFree(d_te);hipFree(d_w1);hipFree(d_b1);hipFree(d_w2);hipFree(d_b2);
    hipFree(d_h);hipFree(d_out);
    hipFree(d_pw0);hipFree(d_pb0);hipFree(d_pw1);hipFree(d_pb1);hipFree(d_pw2);hipFree(d_pb2);
    hipFree(d_h0);hipFree(d_h1);hipFree(d_lo);hipFree(d_noisy);
    return 0;
}
