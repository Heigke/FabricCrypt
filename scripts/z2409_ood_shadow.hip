/*
 * z2409_ood_shadow.hip — Out-of-Distribution Detection via Shadow Signals
 *
 * Hypothesis: GPU matmul side effects (timing, sign patterns, population
 * activity) differ for in-distribution vs OOD inputs, even when softmax
 * gives high confidence on both.
 *
 * Test: MNIST-trained model run on:
 *   1. MNIST test (in-distribution)
 *   2. Random noise (uniform [0,1])
 *   3. Constant (all zeros)
 *   4. Constant (all ones)
 *   5. Shuffled MNIST (pixel order randomized per image)
 *   6. Inverted MNIST (1 - pixel)
 *
 * If shadow signals differ between ID and OOD while softmax doesn't,
 * we have a zero-cost OOD detector.
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define INPUT_DIM 784
#define HIDDEN1 128
#define HIDDEN2 64
#define OUTPUT_DIM 10

struct Shadow {
    float sign_ratio;
    float timing_cv;
    float neighbor_diff;
    float pop_minority;
    float spike_count;
    float perf_entropy;
};

// Same nosync shadow matmul
__global__ void matmul_sh(
    const float* __restrict__ A, const float* __restrict__ W,
    const float* __restrict__ bias, float* __restrict__ C,
    Shadow* __restrict__ shadow,
    int batch, int in_dim, int out_dim, int relu)
{
    int b = blockIdx.x, o = threadIdx.x, lane = o % 32;
    if (b >= batch || o >= out_dim) return;

    float membrane = 0.0f;
    uint32_t spike_count = 0, sign_agree = 0, sign_total = 0;
    float timing_sum = 0.0f, timing_sumsq = 0.0f;
    float neighbor_diff_sum = 0.0f, pop_minority_sum = 0.0f, perf_sum = 0.0f;
    uint32_t n_blocks = 0;

    uint32_t t_prev;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t_prev));

    float acc = bias[o];
    for (int k = 0; k < in_dim; k += 4) {
        float p0 = W[o*in_dim+k]*A[b*in_dim+k];
        float p1 = (k+1<in_dim)?W[o*in_dim+k+1]*A[b*in_dim+k+1]:0;
        float p2 = (k+2<in_dim)?W[o*in_dim+k+2]*A[b*in_dim+k+2]:0;
        float p3 = (k+3<in_dim)?W[o*in_dim+k+3]*A[b*in_dim+k+3]:0;
        acc += p0+p1+p2+p3;

        int np = (p0>0)+(p1>0)+(p2>0)+(p3>0);
        sign_agree += (np==0||np==4)?1:0; sign_total++;

        float bs = p0+p1+p2+p3;
        uint64_t pm = __ballot(bs>0); uint32_t pp = __popcll(pm);
        uint32_t mn = (pp<16)?pp:(32-pp);
        pop_minority_sum += (float)mn;

        float left = __shfl(acc,(lane+31)%32);
        float right = __shfl(acc,(lane+1)%32);
        neighbor_diff_sum += fabsf(acc-left)+fabsf(acc-right);

        uint32_t t_now;
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t_now));
        float dt = (float)(t_now-t_prev); t_prev = t_now;
        timing_sum += dt; timing_sumsq += dt*dt;

        uint32_t perf;
        asm volatile("s_getreg_b32 %0, hwreg(27)" : "=s"(perf));
        perf_sum += (float)(perf&0xFF);
        n_blocks++;

        float sig = (float)(4-np)*0.2f+(dt>25?0.4f:0)+(float)mn*0.05f;
        membrane += sig;
        if (membrane>6.0f) { spike_count++; membrane=0; asm volatile("s_sleep 1"); }
    }

    if (relu) acc = fmaxf(0,acc);
    C[b*out_dim+o] = acc;

    int idx = b*out_dim+o;
    shadow[idx].sign_ratio = (sign_total>0)?(float)sign_agree/sign_total:0;
    float tm = (n_blocks>0)?timing_sum/n_blocks:0;
    float tv = (n_blocks>1)?fmaxf(0,timing_sumsq/n_blocks-tm*tm):0;
    shadow[idx].timing_cv = (tm>0)?sqrtf(tv)/tm:0;
    shadow[idx].neighbor_diff = (n_blocks>0)?neighbor_diff_sum/n_blocks:0;
    shadow[idx].pop_minority = (n_blocks>0)?pop_minority_sum/n_blocks:0;
    shadow[idx].spike_count = (float)spike_count;
    shadow[idx].perf_entropy = (n_blocks>0)?perf_sum/n_blocks:0;
}

__global__ void argmax_k(const float* L, int* P, int n, int c) {
    int b = blockIdx.x*blockDim.x+threadIdx.x;
    if(b>=n) return;
    float best=-1e30; int bc=0;
    for(int i=0;i<c;i++){float v=L[b*c+i];if(v>best){best=v;bc=i;}}
    P[b]=bc;
}

// Softmax confidence (on host)
void compute_softmax_conf(float* logits, float* conf, int n, int nc) {
    for (int i = 0; i < n; i++) {
        float mx = -1e30;
        for (int c = 0; c < nc; c++) if (logits[i*nc+c] > mx) mx = logits[i*nc+c];
        float sum = 0;
        for (int c = 0; c < nc; c++) sum += expf(logits[i*nc+c] - mx);
        float max_prob = 0;
        for (int c = 0; c < nc; c++) {
            float p = expf(logits[i*nc+c] - mx) / sum;
            if (p > max_prob) max_prob = p;
        }
        conf[i] = max_prob;
    }
}

float* load_bin(const char* p, int n) {
    FILE* f=fopen(p,"rb"); if(!f) return NULL;
    float* d=(float*)malloc(n*4); fread(d,4,n,f); fclose(f); return d;
}

int main() {
    printf("============================================================\n");
    printf("z2409: OOD DETECTION VIA SHADOW SIGNALS\n");
    printf("============================================================\n\n");

    const char* base="/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy";
    char p[512];

    // Load MNIST test
    sprintf(p,"%s/data/MNIST/raw/t10k-images-idx3-ubyte",base);
    FILE* f=fopen(p,"rb"); uint32_t hdr[4]; fread(hdr,4,4,f);
    int N=__builtin_bswap32(hdr[1]); N = 2000; // use 2000 per condition
    float* mnist_img=(float*)malloc(N*784*4);
    uint8_t* raw=(uint8_t*)malloc(10000*784); fread(raw,1,10000*784,f);
    for(int i=0;i<N*784;i++) mnist_img[i]=raw[i]/255.0f;
    fclose(f);

    // Generate OOD datasets
    float* noise_img = (float*)malloc(N*784*4);
    float* zeros_img = (float*)calloc(N*784, 4);
    float* ones_img = (float*)malloc(N*784*4);
    float* shuffled_img = (float*)malloc(N*784*4);
    float* inverted_img = (float*)malloc(N*784*4);

    srand(42);
    for (int i = 0; i < N*784; i++) {
        noise_img[i] = (float)rand() / RAND_MAX;
        ones_img[i] = 1.0f;
        inverted_img[i] = 1.0f - mnist_img[i];
    }
    // Shuffled: randomize pixel ORDER per image
    memcpy(shuffled_img, mnist_img, N*784*4);
    for (int i = 0; i < N; i++) {
        for (int j = 783; j > 0; j--) {
            int k = rand() % (j+1);
            float tmp = shuffled_img[i*784+j];
            shuffled_img[i*784+j] = shuffled_img[i*784+k];
            shuffled_img[i*784+k] = tmp;
        }
    }

    // Load weights
    const char* md="/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/models/mnist_mlp";
    sprintf(p,"%s/w1.bin",md); float*w1=load_bin(p,128*784);
    sprintf(p,"%s/b1.bin",md); float*b1=load_bin(p,128);
    sprintf(p,"%s/w2.bin",md); float*w2=load_bin(p,64*128);
    sprintf(p,"%s/b2.bin",md); float*b2=load_bin(p,64);
    sprintf(p,"%s/w3.bin",md); float*w3=load_bin(p,10*64);
    sprintf(p,"%s/b3.bin",md); float*b3=load_bin(p,10);

    // GPU alloc
    float *d_in, *d_h1, *d_h2, *d_out;
    float *d_w1,*d_b1,*d_w2,*d_b2,*d_w3,*d_b3;
    int *d_pred; Shadow *d_sh3;

    hipMalloc(&d_in,N*784*4); hipMalloc(&d_h1,N*128*4);
    hipMalloc(&d_h2,N*64*4); hipMalloc(&d_out,N*10*4);
    hipMalloc(&d_pred,N*4);
    hipMalloc(&d_w1,128*784*4); hipMalloc(&d_b1,128*4);
    hipMalloc(&d_w2,64*128*4); hipMalloc(&d_b2,64*4);
    hipMalloc(&d_w3,10*64*4); hipMalloc(&d_b3,10*4);
    Shadow *d_sh1, *d_sh2;
    hipMalloc(&d_sh1,N*128*sizeof(Shadow));
    hipMalloc(&d_sh2,N*64*sizeof(Shadow));
    hipMalloc(&d_sh3,N*10*sizeof(Shadow));

    hipMemcpy(d_w1,w1,128*784*4,hipMemcpyHostToDevice);
    hipMemcpy(d_b1,b1,128*4,hipMemcpyHostToDevice);
    hipMemcpy(d_w2,w2,64*128*4,hipMemcpyHostToDevice);
    hipMemcpy(d_b2,b2,64*4,hipMemcpyHostToDevice);
    hipMemcpy(d_w3,w3,10*64*4,hipMemcpyHostToDevice);
    hipMemcpy(d_b3,b3,10*4,hipMemcpyHostToDevice);

    float h_logits[N*10];
    Shadow h_sh3[N*10];
    int h_pred[N];
    float h_conf[N];

    // ================================================================
    // Run each condition
    // ================================================================
    struct { const char* name; float* data; int is_id; } conditions[] = {
        {"MNIST (in-dist)", mnist_img, 1},
        {"Random noise", noise_img, 0},
        {"All zeros", zeros_img, 0},
        {"All ones", ones_img, 0},
        {"Shuffled MNIST", shuffled_img, 0},
        {"Inverted MNIST", inverted_img, 0},
    };
    int n_conds = 6;

    printf("%-20s %8s %8s %8s %8s %8s %8s %8s\n",
           "Condition", "MaxConf", "SignR", "TimCV", "NeighD", "PopMin", "Spikes", "PerfE");
    printf("%-20s %8s %8s %8s %8s %8s %8s %8s\n",
           "----------", "------", "------", "------", "------", "------", "------", "------");

    for (int ci = 0; ci < n_conds; ci++) {
        hipMemcpy(d_in, conditions[ci].data, N*784*4, hipMemcpyHostToDevice);

        matmul_sh<<<N,128>>>(d_in,d_w1,d_b1,d_h1,d_sh1,N,784,128,1);
        matmul_sh<<<N,64>>>(d_h1,d_w2,d_b2,d_h2,d_sh2,N,128,64,1);
        matmul_sh<<<N,10>>>(d_h2,d_w3,d_b3,d_out,d_sh3,N,64,10,0);
        argmax_k<<<(N+255)/256,256>>>(d_out,d_pred,N,10);
        hipDeviceSynchronize();

        hipMemcpy(h_logits, d_out, N*10*4, hipMemcpyDeviceToHost);
        hipMemcpy(h_sh3, d_sh3, N*10*sizeof(Shadow), hipMemcpyDeviceToHost);
        hipMemcpy(h_pred, d_pred, N*4, hipMemcpyDeviceToHost);

        // Softmax confidence
        compute_softmax_conf(h_logits, h_conf, N, 10);

        // Average shadow signals
        float avg_conf=0, avg_sr=0, avg_tc=0, avg_nd=0, avg_pm=0, avg_sp=0, avg_pe=0;
        for (int i = 0; i < N; i++) {
            avg_conf += h_conf[i];
            for (int c = 0; c < 10; c++) {
                int x = i*10+c;
                avg_sr += h_sh3[x].sign_ratio;
                avg_tc += h_sh3[x].timing_cv;
                avg_nd += h_sh3[x].neighbor_diff;
                avg_pm += h_sh3[x].pop_minority;
                avg_sp += h_sh3[x].spike_count;
                avg_pe += h_sh3[x].perf_entropy;
            }
        }
        int total = N * 10;
        avg_conf /= N;
        avg_sr /= total; avg_tc /= total; avg_nd /= total;
        avg_pm /= total; avg_sp /= total; avg_pe /= total;

        printf("%-20s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f  %s\n",
               conditions[ci].name, avg_conf, avg_sr, avg_tc, avg_nd, avg_pm, avg_sp, avg_pe,
               conditions[ci].is_id ? "ID" : "OOD");
    }

    // Cleanup
    hipFree(d_in); hipFree(d_h1); hipFree(d_h2); hipFree(d_out); hipFree(d_pred);
    hipFree(d_w1); hipFree(d_b1); hipFree(d_w2); hipFree(d_b2); hipFree(d_w3); hipFree(d_b3);
    hipFree(d_sh1); hipFree(d_sh2); hipFree(d_sh3);
    free(mnist_img); free(noise_img); free(zeros_img); free(ones_img);
    free(shuffled_img); free(inverted_img);
    free(w1); free(b1); free(w2); free(b2); free(w3); free(b3);

    printf("\n=== z2409 COMPLETE ===\n");
    return 0;
}
