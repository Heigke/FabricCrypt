// jitter_gpu_esn_v2.hip — Deep push: quadratic readout, longer sequences, noise sweep
// Can silicon noise expand the computational capacity beyond deterministic ESN?
//
// Improvements over v1:
//   1. Quadratic readout (x, x², x_i*x_j cross-terms)
//   2. Longer sequences (1600 steps, 200 washout = 1400 training)
//   3. State-dependent noise (noise amplitude modulated by neuron activity)
//   4. Detailed per-delay MC and XOR analysis
//   5. Multiple noise mechanisms: additive, multiplicative, stochastic threshold
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_gpu_esn_v2 scripts/jitter_gpu_esn_v2.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_gpu_esn_v2

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

#define N_CUS 12
#define THREADS 64
#define N_NEURONS (N_CUS * THREADS)
#define STEPS 1600
#define WASHOUT 200
#define SAMPLE_N 96  // fewer neurons, but with quad features → 96*3=288 dims

// Noise injection modes
#define NOISE_NONE 0
#define NOISE_ADDITIVE 1      // standard: noise added to pre-activation
#define NOISE_MULTIPLICATIVE 2 // noise modulates recurrent weights
#define NOISE_THRESHOLD 3     // stochastic activation threshold

__global__ void esn_v2(
    float* state_out,          // [N_NEURONS × STEPS]
    volatile float* cu_xchg,
    const float* input_seq,
    int steps,
    int noise_mode,
    float spectral_radius,
    float noise_gain,
    float input_scale
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= THREADS) return;
    int nid = bid * THREADS + tid;

    __shared__ float lds[THREADS];

    // Dense weight row (hash-based)
    float w_row[THREADS];
    float scale = spectral_radius / 8.0f;
    for (int j = 0; j < THREADS; j++) {
        uint32_t h = (uint32_t)(bid*7919 + tid*6271 + j*3571 + 42);
        h ^= h>>16; h *= 0x45d9f3b; h ^= h>>16; h *= 0x45d9f3b; h ^= h>>16;
        w_row[j] = scale * ((float)(h & 0xFFFF)/32768.0f - 1.0f);
    }

    // Input weight
    uint32_t irng = nid * 2654435761u + 99991;
    irng ^= irng>>16; irng *= 0x45d9f3b; irng ^= irng>>16;
    float w_in = input_scale * ((float)(irng & 0xFFFF)/32768.0f - 1.0f);

    // Bias
    irng *= 0x45d9f3b; irng ^= irng>>16;
    float bias = 0.005f * ((float)(irng & 0xFFFF)/32768.0f - 1.0f);

    // Cross-CU sparse
    irng *= 1103515245; int rcu = ((irng>>8)&0xFF)%N_CUS;
    irng *= 1103515245; int rtid = ((irng>>8)&0xFF)%THREADS;
    irng *= 1103515245; float w_cross = 0.03f * ((float)(irng & 0xFFFF)/32768.0f - 1.0f);

    float v = 0.0f;
    float dummy = 1.0001f + nid * 0.0001f;

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];

        // Dense recurrence
        lds[tid] = v;
        __syncthreads();

        float rec = 0.0f;
        for (int j = 0; j < THREADS; j++)
            rec += w_row[j] * lds[j];

        // Cross-CU
        cu_xchg[nid] = v;
        __threadfence();
        float cross = w_cross * cu_xchg[rcu*THREADS+rtid];

        // PLL noise
        float noise = 0.0f;
        if (noise_mode != NOISE_NONE) {
            int depth = 16 + (int)(48.0f * fabsf(v));
            if (depth > 64) depth = 64;
            float ns = 0.0f;
            for (int r = 0; r < depth; r++) {
                uint64_t w0 = wall_clock64();
                uint64_t w1 = wall_clock64();
                ns += (float)(w1 - w0);
            }
            noise = (ns - depth*2.2f) / (sqrtf((float)depth) * 2.0f);

            // Instruction-mix PLL modulation
            if (v > 0.0f) {
                for (int i = 0; i < 4; i++)
                    dummy = __fmaf_rn(dummy, 1.000001f, 0.0001f);
            } else {
                uint32_t iv = __float_as_uint(dummy);
                for (int i = 0; i < 4; i++) iv = iv*2654435761u+1;
                dummy = __uint_as_float((iv & 0x007FFFFF) | 0x3F800000);
            }
        }

        // State update (depends on noise mode)
        float pre = rec + cross + w_in * u + bias;

        if (noise_mode == NOISE_ADDITIVE) {
            v = tanhf(pre + noise_gain * noise);
        } else if (noise_mode == NOISE_MULTIPLICATIVE) {
            // Noise modulates the recurrent contribution
            v = tanhf(pre * (1.0f + noise_gain * noise));
        } else if (noise_mode == NOISE_THRESHOLD) {
            // Stochastic activation: threshold shifts with noise
            float thresh = noise_gain * noise;
            float act = tanhf(pre);
            v = (fabsf(act) > 0.5f + thresh*0.3f) ? act : act * 0.5f;
        } else {
            v = tanhf(pre);
        }

        state_out[nid * steps + t] = v;
        __syncthreads();
    }
    if (dummy == -9999.0f) state_out[0] = dummy;
}

// ============================================================================
// HOST
// ============================================================================
static uint32_t host_rng = 42;
float randf() { host_rng=host_rng*1103515245+12345; return (float)((host_rng>>16)&0x7FFF)/32767.0f; }
float randf_signed() { return randf()*2.0f-1.0f; }

// Extract features with quadratic terms
void extract_quad_features(
    float* state, int n, int steps, int t,
    std::vector<float>& feat
) {
    int stride = n / SAMPLE_N;
    if (stride < 1) stride = 1;
    feat.clear();

    // Linear features
    for (int si = 0; si < SAMPLE_N && si*stride < n; si++) {
        float v = state[si*stride*steps + t];
        if (v!=v || v>1e20 || v<-1e20) v=0;
        feat.push_back(v);          // x
        feat.push_back(v * v);      // x²
    }
    // Cross-products (subset: every 8th pair for tractability)
    for (int si = 0; si < SAMPLE_N-8 && si*stride < n; si += 8) {
        int i1 = si*stride, i2 = (si+8)*stride;
        if (i2 >= n) break;
        float v1 = state[i1*steps+t], v2 = state[i2*steps+t];
        if (v1!=v1||v1>1e20)v1=0; if(v2!=v2||v2>1e20)v2=0;
        feat.push_back(v1 * v2);    // x_i * x_j
    }
}

float ridge_regress(
    std::vector<std::vector<float>>& X, std::vector<float>& y, int n_train
) {
    int n = X.size();
    if (n < n_train+10) return -1.0f;
    int d = X[0].size();
    float reg = 1.0f;

    std::vector<float> XtX(d*d,0), Xty(d,0);
    for (int i = 0; i < n_train; i++) {
        for (int a = 0; a < d; a++) {
            Xty[a] += X[i][a]*y[i];
            for (int b = a; b < d; b++)
                XtX[a*d+b] += X[i][a]*X[i][b];
        }
    }
    for (int a = 0; a < d; a++) {
        for (int b=a+1;b<d;b++) XtX[b*d+a]=XtX[a*d+b];
        XtX[a*d+a]+=reg;
    }
    std::vector<float> L(d*d,0);
    for (int i=0;i<d;i++){
        for(int j=0;j<=i;j++){
            float s=XtX[i*d+j];
            for(int k=0;k<j;k++)s-=L[i*d+k]*L[j*d+k];
            if(i==j){if(s<=0)s=1e-6f;L[i*d+j]=sqrtf(s);}
            else L[i*d+j]=s/L[j*d+j];
        }
    }
    std::vector<float> z(d),w(d);
    for(int i=0;i<d;i++){float s=Xty[i];for(int k=0;k<i;k++)s-=L[i*d+k]*z[k];z[i]=s/L[i*d+i];}
    for(int i=d-1;i>=0;i--){float s=z[i];for(int k=i+1;k<d;k++)s-=L[k*d+i]*w[k];w[i]=s/L[i*d+i];}

    float ss_res=0,ss_tot=0,ym=0;
    int nt=n-n_train;
    for(int i=n_train;i<n;i++)ym+=y[i];
    ym/=nt;
    for(int i=n_train;i<n;i++){
        float p=0;for(int j=0;j<d;j++)p+=w[j]*X[i][j];
        float e=y[i]-p; ss_res+=e*e; ss_tot+=(y[i]-ym)*(y[i]-ym);
    }
    return ss_tot<1e-10f?0.0f:1.0f-ss_res/ss_tot;
}

float ridge_classify(
    std::vector<std::vector<float>>& X, std::vector<int>& y, int n_train
) {
    int n=X.size(),d=X[0].size();
    std::vector<float> yf(n);
    for(int i=0;i<n;i++)yf[i]=(float)y[i];
    float reg=1.0f;
    std::vector<float> XtX(d*d,0),Xty(d,0);
    for(int i=0;i<n_train;i++){
        for(int a=0;a<d;a++){
            Xty[a]+=X[i][a]*yf[i];
            for(int b=a;b<d;b++)XtX[a*d+b]+=X[i][a]*X[i][b];
        }
    }
    for(int a=0;a<d;a++){for(int b=a+1;b<d;b++)XtX[b*d+a]=XtX[a*d+b];XtX[a*d+a]+=reg;}
    std::vector<float> L(d*d,0);
    for(int i=0;i<d;i++){for(int j=0;j<=i;j++){float s=XtX[i*d+j];for(int k=0;k<j;k++)s-=L[i*d+k]*L[j*d+k];if(i==j){if(s<=0)s=1e-6f;L[i*d+j]=sqrtf(s);}else L[i*d+j]=s/L[j*d+j];}}
    std::vector<float> z(d),w(d);
    for(int i=0;i<d;i++){float s=Xty[i];for(int k=0;k<i;k++)s-=L[i*d+k]*z[k];z[i]=s/L[i*d+i];}
    for(int i=d-1;i>=0;i--){float s=z[i];for(int k=i+1;k<d;k++)s-=L[k*d+i]*w[k];w[i]=s/L[i*d+i];}
    int correct=0,total=0;
    for(int i=n_train;i<n;i++){float p=0;for(int j=0;j<d;j++)p+=w[j]*X[i][j];if((p>0.5f?1:0)==y[i])correct++;total++;}
    return (float)correct/total;
}

void run_and_collect(
    float* d_state, volatile float* d_xchg, float* d_input,
    float* h_state, const float* h_input, int steps,
    int noise_mode, float sr, float ng, float is,
    std::vector<std::vector<float>>& feats, bool quad
) {
    hipMemcpy(d_input, h_input, steps*sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_state, 0, N_NEURONS*steps*sizeof(float));
    hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
    esn_v2<<<N_CUS,THREADS>>>(d_state,d_xchg,d_input,steps,noise_mode,sr,ng,is);
    hipDeviceSynchronize();
    hipMemcpy(h_state, d_state, N_NEURONS*steps*sizeof(float), hipMemcpyDeviceToHost);

    feats.clear();
    for (int t = WASHOUT; t < steps; t++) {
        std::vector<float> f;
        if (quad) {
            extract_quad_features(h_state, N_NEURONS, steps, t, f);
        } else {
            int stride = N_NEURONS / SAMPLE_N;
            for (int si = 0; si < SAMPLE_N && si*stride < N_NEURONS; si++)
                f.push_back(h_state[si*stride*steps+t]);
        }
        feats.push_back(f);
    }
}

int main() {
    printf("═══ GPU SILICON ESN v2 — Quadratic Readout + Noise Modes ═══\n");
    printf("  %d neurons, %d steps (washout=%d), quad features\n\n", N_NEURONS, STEPS, WASHOUT);

    float *d_state, *d_input;
    volatile float *d_xchg;
    hipMalloc(&d_state, N_NEURONS*STEPS*sizeof(float));
    hipMalloc((void**)&d_xchg, N_NEURONS*sizeof(float));
    hipMalloc(&d_input, STEPS*sizeof(float));
    float* h_state = new float[N_NEURONS*STEPS];

    host_rng = 31415;
    float inp_bin[STEPS], inp_cont[STEPS];
    for(int t=0;t<STEPS;t++){
        inp_bin[t] = randf()>0.5f?1.0f:0.0f;
        inp_cont[t] = randf_signed()*0.5f;
    }

    // NARMA-5
    float narma[STEPS]={};
    for(int t=5;t<STEPS;t++){
        narma[t]=0.3f*narma[t-1]+0.05f*narma[t-1]*(narma[t-1]+narma[t-2]+narma[t-3]+narma[t-4]+narma[t-5])
                +1.5f*inp_cont[t-1]*inp_cont[t-5]+0.1f;
        if(narma[t]>10)narma[t]=10;if(narma[t]<-10)narma[t]=-10;
    }

    // Targets
    std::vector<int> xor_t[8]; // XOR at delays 1-8
    for(int d=1;d<=8;d++){
        xor_t[d-1].resize(STEPS,0);
        for(int t=d;t<STEPS;t++) xor_t[d-1][t]=(int)inp_bin[t]^(int)inp_bin[t-d];
    }
    std::vector<int> par3(STEPS,0);
    for(int t=3;t<STEPS;t++) par3[t]=((int)inp_bin[t]^(int)inp_bin[t-1]^(int)inp_bin[t-2]^(int)inp_bin[t-3])&1;

    float SR = 0.99f, IS = 1.0f;

    // ═══ TEST 1: Linear vs Quadratic readout ═══
    printf("═══ TEST 1: LINEAR vs QUADRATIC READOUT (SR=%.2f, IS=%.1f) ═══\n", SR, IS);
    printf("%-6s %-6s | XOR-1  XOR-2  XOR-3  XOR-5  XOR-8 | NARMA   MC     Par3\n", "Read", "Noise");
    printf("──────────────────────────────────────────────────────────────────────\n");

    const char* readout_names[] = {"Lin", "Quad"};
    const char* noise_names[] = {"none", "add", "mult", "thresh"};

    for (int quad = 0; quad <= 1; quad++) {
        for (int nm = 0; nm <= 3; nm++) {
            float ng = (nm == 0) ? 0.0f : 0.003f;

            // Continuous input (NARMA + MC)
            std::vector<std::vector<float>> fc;
            run_and_collect(d_state,d_xchg,d_input,h_state,inp_cont,STEPS,nm,SR,ng,IS,fc,quad);
            int ns = fc.size(), nt = ns*2/3;

            std::vector<float> yn;
            for(int t=WASHOUT;t<STEPS;t++) yn.push_back(narma[t]);
            float r2_narma = ridge_regress(fc, yn, nt);

            float mc = 0;
            for(int d=1;d<=10;d++){
                std::vector<float> ym;
                for(int t=WASHOUT;t<STEPS;t++) ym.push_back(t-d>=0?inp_cont[t-d]:0.0f);
                float r2=ridge_regress(fc,ym,nt);
                if(r2>0)mc+=r2;
            }

            // Binary input (XOR + parity)
            std::vector<std::vector<float>> fb;
            run_and_collect(d_state,d_xchg,d_input,h_state,inp_bin,STEPS,nm,SR,ng,IS,fb,quad);
            ns=fb.size(); nt=ns*2/3;

            float xor_accs[8];
            for(int d=0;d<8;d++){
                std::vector<int> yt;
                for(int t=WASHOUT;t<STEPS;t++) yt.push_back(xor_t[d][t]);
                xor_accs[d] = ridge_classify(fb, yt, nt);
            }
            std::vector<int> yp;
            for(int t=WASHOUT;t<STEPS;t++) yp.push_back(par3[t]);
            float acc_par = ridge_classify(fb, yp, nt);

            printf("%-6s %-6s | %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.1f%% | %+.3f  %5.2f  %5.1f%%",
                   readout_names[quad], noise_names[nm],
                   xor_accs[0]*100, xor_accs[1]*100, xor_accs[2]*100,
                   xor_accs[4]*100, xor_accs[7]*100,
                   r2_narma, mc, acc_par*100);

            if (xor_accs[4] > 0.55 || r2_narma > 0.1) printf(" ★★★");
            else if (xor_accs[2] > 0.90 || mc > 5.0) printf(" ★");
            printf("\n");
        }
    }

    // ═══ TEST 2: Noise gain sweep (best readout) ═══
    printf("\n═══ TEST 2: NOISE GAIN SWEEP (Quad readout, additive) ═══\n");
    printf("NG       | XOR-2  XOR-3  XOR-5 | NARMA    MC    | Δ vs NG=0\n");
    printf("──────────────────────────────────────────────────────────────\n");

    float ref_xor2=0, ref_narma=0, ref_mc=0;
    float ng_sweep[] = {0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20};

    for (int gi = 0; gi < 9; gi++) {
        float ng = ng_sweep[gi];
        int nm = (ng > 0) ? NOISE_ADDITIVE : NOISE_NONE;

        std::vector<std::vector<float>> fc, fb;
        run_and_collect(d_state,d_xchg,d_input,h_state,inp_cont,STEPS,nm,SR,ng,IS,fc,true);
        run_and_collect(d_state,d_xchg,d_input,h_state,inp_bin,STEPS,nm,SR,ng,IS,fb,true);

        int ns=fc.size(), nt=ns*2/3;
        std::vector<float> yn;
        for(int t=WASHOUT;t<STEPS;t++) yn.push_back(narma[t]);
        float r2n = ridge_regress(fc, yn, nt);

        float mc=0;
        for(int d=1;d<=10;d++){
            std::vector<float> ym;
            for(int t=WASHOUT;t<STEPS;t++) ym.push_back(t-d>=0?inp_cont[t-d]:0.0f);
            float r2=ridge_regress(fc,ym,nt);
            if(r2>0)mc+=r2;
        }

        ns=fb.size(); nt=ns*2/3;
        std::vector<int> y2,y3,y5;
        for(int t=WASHOUT;t<STEPS;t++){y2.push_back(xor_t[1][t]);y3.push_back(xor_t[2][t]);y5.push_back(xor_t[4][t]);}
        float x2=ridge_classify(fb,y2,nt), x3=ridge_classify(fb,y3,nt), x5=ridge_classify(fb,y5,nt);

        if (gi == 0) { ref_xor2=x2; ref_narma=r2n; ref_mc=mc; }

        printf("%.3f    | %5.1f%% %5.1f%% %5.1f%% | %+.3f   %5.2f  | Δxor=%+.1f Δnrm=%+.3f Δmc=%+.2f\n",
               ng, x2*100, x3*100, x5*100, r2n, mc,
               (x2-ref_xor2)*100, r2n-ref_narma, mc-ref_mc);
    }

    // ═══ TEST 3: Per-delay Memory Capacity breakdown ═══
    printf("\n═══ TEST 3: MEMORY CAPACITY (per delay, 3 trials avg) ═══\n");
    printf("Delay | NOISE R²    CTRL R²    Δ\n");
    printf("──────────────────────────────────\n");

    float mc_total_n=0, mc_total_c=0;
    for (int d = 1; d <= 15; d++) {
        float avg_n=0, avg_c=0;
        for (int trial = 0; trial < 3; trial++) {
            std::vector<std::vector<float>> fn, fc;
            run_and_collect(d_state,d_xchg,d_input,h_state,inp_cont,STEPS,NOISE_ADDITIVE,SR,0.003f,IS,fn,true);
            run_and_collect(d_state,d_xchg,d_input,h_state,inp_cont,STEPS,NOISE_NONE,SR,0,IS,fc,true);

            int ns=fn.size(), nt=ns*2/3;
            std::vector<float> ym;
            for(int t=WASHOUT;t<STEPS;t++) ym.push_back(t-d>=0?inp_cont[t-d]:0.0f);
            float rn = ridge_regress(fn, ym, nt);
            float rc = ridge_regress(fc, ym, nt);
            avg_n += rn; avg_c += rc;
        }
        avg_n /= 3; avg_c /= 3;
        if (avg_n > 0) mc_total_n += avg_n;
        if (avg_c > 0) mc_total_c += avg_c;

        char bar_n[21]={}, bar_c[21]={};
        int bn = (int)(avg_n*20); if(bn<0)bn=0; if(bn>20)bn=20;
        int bc = (int)(avg_c*20); if(bc<0)bc=0; if(bc>20)bc=20;
        memset(bar_n,'#',bn); memset(bar_c,'#',bc);

        printf("  %2d  | %+.3f %-20s  %+.3f %-20s  %+.3f%s\n",
               d, avg_n, bar_n, avg_c, bar_c, avg_n-avg_c,
               (avg_n > avg_c + 0.01) ? " *" : "");
    }
    printf("Total | MC_N=%.2f  MC_C=%.2f  Δ=%+.2f\n", mc_total_n, mc_total_c, mc_total_n-mc_total_c);

    printf("\nDone.\n");
    hipFree(d_state); hipFree((void*)d_xchg); hipFree(d_input);
    delete[] h_state;
    return 0;
}
