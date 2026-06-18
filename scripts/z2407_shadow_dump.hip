/*
 * z2407_shadow_dump.hip — Dump per-sample shadow signals + logits from GPU
 * Then analyze complementarity: does shadow + softmax beat softmax alone?
 *
 * Outputs raw binary file with per-sample:
 *   - 10 logits (float)
 *   - 10 shadow structs (sign_ratio, timing_cv, neighbor_diff, pop_minority,
 *     spike_count, perf_entropy per output neuron)
 *   - label (int)
 *   - prediction (int)
 *   - correct (int)
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

struct ShadowOut {
    float sign_ratio;
    float timing_cv;
    float neighbor_diff;
    float pop_minority;
    float spike_count;
    float perf_entropy;
};

// Same nosync shadow kernel as z2405
__global__ void matmul_shadow(
    const float* __restrict__ A, const float* __restrict__ W,
    const float* __restrict__ bias, float* __restrict__ C,
    ShadowOut* __restrict__ shadow,
    int batch, int in_dim, int out_dim, int relu, int fb)
{
    int b = blockIdx.x, o = threadIdx.x, lane = o % 32;
    if (b >= batch || o >= out_dim) return;

    uint32_t orig_mode;
    if (fb) asm volatile("s_getreg_b32 %0, hwreg(1)" : "=s"(orig_mode));

    float membrane = 0.0f;
    uint32_t spike_count = 0, sign_agree = 0, sign_total = 0;
    float timing_sum = 0.0f, timing_sumsq = 0.0f;
    float neighbor_diff_sum = 0.0f, pop_minority_sum = 0.0f, perf_sum = 0.0f;
    uint32_t n_blocks = 0;

    uint32_t t_prev;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t_prev));

    float acc = bias[o];
    for (int k = 0; k < in_dim; k += 4) {
        float p0 = W[o*in_dim+k] * A[b*in_dim+k];
        float p1 = (k+1<in_dim) ? W[o*in_dim+k+1]*A[b*in_dim+k+1] : 0.0f;
        float p2 = (k+2<in_dim) ? W[o*in_dim+k+2]*A[b*in_dim+k+2] : 0.0f;
        float p3 = (k+3<in_dim) ? W[o*in_dim+k+3]*A[b*in_dim+k+3] : 0.0f;
        acc += p0 + p1 + p2 + p3;

        int n_pos = (p0>0)+(p1>0)+(p2>0)+(p3>0);
        sign_agree += (n_pos==0||n_pos==4)?1:0;
        sign_total++;

        float bs = p0+p1+p2+p3;
        uint64_t pm = __ballot(bs>0.0f);
        uint32_t pp = __popcll(pm);
        uint32_t mn = (pp<16)?pp:(32-pp);
        pop_minority_sum += (float)mn;

        float left = __shfl(acc, (lane+31)%32);
        float right = __shfl(acc, (lane+1)%32);
        neighbor_diff_sum += fabsf(acc-left) + fabsf(acc-right);

        uint32_t t_now;
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t_now));
        float dt = (float)(t_now - t_prev); t_prev = t_now;
        timing_sum += dt; timing_sumsq += dt*dt;

        uint32_t perf;
        asm volatile("s_getreg_b32 %0, hwreg(27)" : "=s"(perf));
        perf_sum += (float)(perf & 0xFF);
        n_blocks++;

        float sig = (float)(4-n_pos)*0.2f + (dt>25.0f?0.4f:0.0f) + (float)mn*0.05f;
        membrane += sig;
        if (membrane > 6.0f) { spike_count++; membrane = 0.0f; asm volatile("s_sleep 1"); }

        if (fb && (n_blocks&0xF)==0) {
            float ar = (float)sign_agree/(float)sign_total;
            if (ar < 0.3f) asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x3");
            else asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x0");
        }
    }
    if (fb) asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(orig_mode));
    if (relu) acc = fmaxf(0.0f, acc);
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
    if (b>=n) return;
    float best=-1e30f; int bc=0;
    for (int i=0;i<c;i++) { float v=L[b*c+i]; if(v>best){best=v;bc=i;} }
    P[b]=bc;
}

float* load_bin(const char* p, int n) {
    FILE* f=fopen(p,"rb"); if(!f) return NULL;
    float* d=(float*)malloc(n*4); fread(d,4,n,f); fclose(f); return d;
}

int main() {
    printf("z2407: Shadow signal dump for ROC analysis\n\n");
    const char* base="/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy";
    char p[512];

    // Load MNIST
    sprintf(p,"%s/data/MNIST/raw/t10k-images-idx3-ubyte",base);
    FILE* f=fopen(p,"rb"); uint32_t hdr[4]; fread(hdr,4,4,f);
    int N=__builtin_bswap32(hdr[1]);
    float* h_img=(float*)malloc(N*784*4);
    uint8_t* raw=(uint8_t*)malloc(N*784); fread(raw,1,N*784,f);
    for(int i=0;i<N*784;i++) h_img[i]=raw[i]/255.0f;
    free(raw); fclose(f);

    sprintf(p,"%s/data/MNIST/raw/t10k-labels-idx1-ubyte",base);
    f=fopen(p,"rb"); fread(hdr,4,2,f);
    int* h_lbl=(int*)malloc(N*4);
    raw=(uint8_t*)malloc(N); fread(raw,1,N,f);
    for(int i=0;i<N;i++) h_lbl[i]=raw[i]; free(raw); fclose(f);

    const char* md="/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/models/mnist_mlp";
    sprintf(p,"%s/w1.bin",md); float*w1=load_bin(p,128*784);
    sprintf(p,"%s/b1.bin",md); float*b1=load_bin(p,128);
    sprintf(p,"%s/w2.bin",md); float*w2=load_bin(p,64*128);
    sprintf(p,"%s/b2.bin",md); float*b2=load_bin(p,64);
    sprintf(p,"%s/w3.bin",md); float*w3=load_bin(p,10*64);
    sprintf(p,"%s/b3.bin",md); float*b3=load_bin(p,10);
    printf("Loaded %d images + trained weights\n",N);

    int BS=1000;
    float*d_in,*d_h1,*d_h2,*d_out;
    float*d_w1,*d_b1,*d_w2,*d_b2,*d_w3,*d_b3;
    int*d_pred; ShadowOut*d_sh1,*d_sh2,*d_sh3;

    hipMalloc(&d_in,BS*784*4); hipMalloc(&d_h1,BS*128*4);
    hipMalloc(&d_h2,BS*64*4); hipMalloc(&d_out,BS*10*4);
    hipMalloc(&d_pred,BS*4);
    hipMalloc(&d_w1,128*784*4); hipMalloc(&d_b1,128*4);
    hipMalloc(&d_w2,64*128*4); hipMalloc(&d_b2,64*4);
    hipMalloc(&d_w3,10*64*4); hipMalloc(&d_b3,10*4);
    hipMalloc(&d_sh1,BS*128*sizeof(ShadowOut));
    hipMalloc(&d_sh2,BS*64*sizeof(ShadowOut));
    hipMalloc(&d_sh3,BS*10*sizeof(ShadowOut));

    hipMemcpy(d_w1,w1,128*784*4,hipMemcpyHostToDevice);
    hipMemcpy(d_b1,b1,128*4,hipMemcpyHostToDevice);
    hipMemcpy(d_w2,w2,64*128*4,hipMemcpyHostToDevice);
    hipMemcpy(d_b2,b2,64*4,hipMemcpyHostToDevice);
    hipMemcpy(d_w3,w3,10*64*4,hipMemcpyHostToDevice);
    hipMemcpy(d_b3,b3,10*4,hipMemcpyHostToDevice);

    // Output arrays
    float* all_logits = (float*)malloc(N*10*sizeof(float));
    ShadowOut* all_shadow = (ShadowOut*)malloc(N*10*sizeof(ShadowOut));
    int* all_preds = (int*)malloc(N*sizeof(int));
    int h_pred[BS];
    float h_logits[BS*10];
    ShadowOut h_sh3[BS*10];

    int correct = 0;
    for (int batch = 0; batch < N/BS; batch++) {
        hipMemcpy(d_in, h_img+batch*BS*784, BS*784*4, hipMemcpyHostToDevice);
        matmul_shadow<<<BS,128>>>(d_in,d_w1,d_b1,d_h1,d_sh1,BS,784,128,1,0);
        matmul_shadow<<<BS,64>>>(d_h1,d_w2,d_b2,d_h2,d_sh2,BS,128,64,1,0);
        matmul_shadow<<<BS,10>>>(d_h2,d_w3,d_b3,d_out,d_sh3,BS,64,10,0,0);
        argmax_k<<<(BS+255)/256,256>>>(d_out,d_pred,BS,10);
        hipDeviceSynchronize();

        hipMemcpy(h_logits, d_out, BS*10*4, hipMemcpyDeviceToHost);
        hipMemcpy(h_sh3, d_sh3, BS*10*sizeof(ShadowOut), hipMemcpyDeviceToHost);
        hipMemcpy(h_pred, d_pred, BS*4, hipMemcpyDeviceToHost);

        memcpy(all_logits+batch*BS*10, h_logits, BS*10*4);
        memcpy(all_shadow+batch*BS*10, h_sh3, BS*10*sizeof(ShadowOut));
        memcpy(all_preds+batch*BS, h_pred, BS*4);

        for(int i=0;i<BS;i++) if(h_pred[i]==h_lbl[batch*BS+i]) correct++;
    }
    printf("Accuracy: %.2f%%\n", 100.0*correct/N);

    // Write binary dump
    sprintf(p,"%s/results/z2407_shadow_dump.bin",base);
    f = fopen(p,"wb");
    // Header: N, n_classes
    int header[2] = {N, 10};
    fwrite(header, 4, 2, f);
    for (int i = 0; i < N; i++) {
        // logits [10 floats]
        fwrite(all_logits+i*10, 4, 10, f);
        // shadow signals [10 structs × 6 floats = 60 floats]
        fwrite(all_shadow+i*10, sizeof(ShadowOut), 10, f);
        // label, pred, correct [3 ints]
        int meta[3] = {h_lbl[i], all_preds[i], (all_preds[i]==h_lbl[i])?1:0};
        fwrite(meta, 4, 3, f);
    }
    fclose(f);
    printf("Dumped to %s (%d samples)\n", p, N);

    // Quick per-signal stats
    printf("\nPer-signal stats (output layer, all samples):\n");
    const char* snames[] = {"sign_ratio","timing_cv","neighbor_diff","pop_minority","spike_count","perf_entropy"};
    for (int s = 0; s < 6; s++) {
        double corr_sum=0, incorr_sum=0;
        int nc=0, ni=0;
        for (int i = 0; i < N; i++) {
            float val = 0;
            for (int c = 0; c < 10; c++) {
                float* ptr = (float*)&all_shadow[i*10+c];
                val += ptr[s];
            }
            val /= 10;
            if (all_preds[i] == h_lbl[i]) { corr_sum += val; nc++; }
            else { incorr_sum += val; ni++; }
        }
        printf("  %-16s corr=%.4f incorr=%.4f ratio=%.3f\n",
               snames[s], corr_sum/nc, incorr_sum/ni,
               (corr_sum/nc)/(incorr_sum/ni+1e-30));
    }

    hipFree(d_in);hipFree(d_h1);hipFree(d_h2);hipFree(d_out);hipFree(d_pred);
    hipFree(d_w1);hipFree(d_b1);hipFree(d_w2);hipFree(d_b2);hipFree(d_w3);hipFree(d_b3);
    hipFree(d_sh1);hipFree(d_sh2);hipFree(d_sh3);
    free(h_img);free(h_lbl);free(w1);free(b1);free(w2);free(b2);free(w3);free(b3);
    free(all_logits);free(all_shadow);free(all_preds);

    printf("\nDone. Run z2407_analyze.py for ROC + complementarity analysis.\n");
    return 0;
}
