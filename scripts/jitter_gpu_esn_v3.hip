// jitter_gpu_esn_v3.hip — Mixed reservoir: deterministic memory + stochastic nonlinearity
// Uses GPU silicon PLL noise for stochastic thresholding in HALF the neurons
//
// Architecture:
//   CU 0-5  (384 neurons): Deterministic (no noise) → high MC, temporal memory
//   CU 6-11 (384 neurons): Stochastic threshold (PLL noise) → strong nonlinearity
//   Cross-CU coupling mixes memory and nonlinearity
//
// Also tests: 2-layer hierarchical (memory → nonlinear mixer)
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_gpu_esn_v3 scripts/jitter_gpu_esn_v3.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_gpu_esn_v3

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
#define SAMPLE_N 96

// ============================================================================
// MIXED RESERVOIR: first half deterministic, second half stochastic
// ============================================================================
__global__ void esn_mixed(
    float* state_out,
    volatile float* cu_xchg,
    const float* input_seq,
    int steps,
    float spectral_radius,
    float input_scale,
    int stochastic_start_cu  // CUs >= this use stochastic threshold
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= THREADS) return;
    int nid = bid * THREADS + tid;
    int is_stochastic = (bid >= stochastic_start_cu) ? 1 : 0;

    __shared__ float lds[THREADS];

    // Dense weight row
    float w_row[THREADS];
    float scale = spectral_radius / 8.0f;
    for (int j = 0; j < THREADS; j++) {
        uint32_t h = (uint32_t)(bid*7919 + tid*6271 + j*3571 + 42);
        h ^= h>>16; h *= 0x45d9f3b; h ^= h>>16; h *= 0x45d9f3b; h ^= h>>16;
        w_row[j] = scale * ((float)(h & 0xFFFF)/32768.0f - 1.0f);
    }

    uint32_t irng = nid * 2654435761u + 99991;
    irng ^= irng>>16; irng *= 0x45d9f3b; irng ^= irng>>16;
    float w_in = input_scale * ((float)(irng & 0xFFFF)/32768.0f - 1.0f);
    irng *= 0x45d9f3b; irng ^= irng>>16;
    float bias = 0.005f * ((float)(irng & 0xFFFF)/32768.0f - 1.0f);

    // Cross-CU: deterministic neurons connect to stochastic and vice versa
    irng *= 1103515245;
    int target_cu;
    if (is_stochastic) {
        target_cu = ((irng>>8)&0xFF) % stochastic_start_cu;  // → deterministic
    } else {
        target_cu = stochastic_start_cu + ((irng>>8)&0xFF) % (N_CUS - stochastic_start_cu); // → stochastic
    }
    irng *= 1103515245; int rtid = ((irng>>8)&0xFF) % THREADS;
    irng *= 1103515245; float w_cross = 0.05f * ((float)(irng & 0xFFFF)/32768.0f - 1.0f);

    float v = 0.0f;
    float dummy = 1.0001f + nid * 0.0001f;

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];

        lds[tid] = v;
        __syncthreads();

        float rec = 0.0f;
        for (int j = 0; j < THREADS; j++)
            rec += w_row[j] * lds[j];

        cu_xchg[nid] = v;
        __threadfence();
        float cross = w_cross * cu_xchg[target_cu * THREADS + rtid];

        float pre = rec + cross + w_in * u + bias;

        if (is_stochastic) {
            // Stochastic threshold using PLL noise
            int depth = 16 + (int)(48.0f * fabsf(v));
            if (depth > 64) depth = 64;
            float ns = 0.0f;
            for (int r = 0; r < depth; r++) {
                uint64_t w0 = wall_clock64();
                uint64_t w1 = wall_clock64();
                ns += (float)(w1 - w0);
            }
            float noise = (ns - depth*2.2f) / (sqrtf((float)depth) * 2.0f);

            // Instruction-mix modulation
            if (v > 0.0f) {
                for (int i = 0; i < 4; i++)
                    dummy = __fmaf_rn(dummy, 1.000001f, 0.0001f);
            } else {
                uint32_t iv = __float_as_uint(dummy);
                for (int i = 0; i < 4; i++) iv = iv*2654435761u+1;
                dummy = __uint_as_float((iv & 0x007FFFFF) | 0x3F800000);
            }

            // Stochastic activation
            float act = tanhf(pre);
            float thresh = 0.5f + 0.15f * noise;
            v = (fabsf(act) > thresh) ? act : act * 0.3f;
        } else {
            // Deterministic tanh
            v = tanhf(pre);
        }

        state_out[nid * steps + t] = v;
        __syncthreads();
    }
    if (dummy == -9999.0f) state_out[0] = dummy;
}

// Pure deterministic (for comparison)
__global__ void esn_pure(
    float* state_out, volatile float* cu_xchg, const float* input_seq,
    int steps, float spectral_radius, float input_scale
) {
    int bid = blockIdx.x, tid = threadIdx.x;
    if (bid >= N_CUS || tid >= THREADS) return;
    int nid = bid*THREADS+tid;
    __shared__ float lds[THREADS];

    float w_row[THREADS];
    float scale = spectral_radius / 8.0f;
    for (int j = 0; j < THREADS; j++) {
        uint32_t h = (uint32_t)(bid*7919+tid*6271+j*3571+42);
        h^=h>>16;h*=0x45d9f3b;h^=h>>16;h*=0x45d9f3b;h^=h>>16;
        w_row[j] = scale*((float)(h&0xFFFF)/32768.0f-1.0f);
    }
    uint32_t irng=nid*2654435761u+99991;
    irng^=irng>>16;irng*=0x45d9f3b;irng^=irng>>16;
    float w_in=input_scale*((float)(irng&0xFFFF)/32768.0f-1.0f);
    irng*=0x45d9f3b;irng^=irng>>16;
    float bias=0.005f*((float)(irng&0xFFFF)/32768.0f-1.0f);
    irng*=1103515245;int rcu=((irng>>8)&0xFF)%N_CUS;
    irng*=1103515245;int rtid=((irng>>8)&0xFF)%THREADS;
    irng*=1103515245;float w_cross=0.05f*((float)(irng&0xFFFF)/32768.0f-1.0f);

    float v=0.0f;
    for(int t=0;t<steps;t++){
        float u=input_seq[t];
        lds[tid]=v;__syncthreads();
        float rec=0;for(int j=0;j<THREADS;j++)rec+=w_row[j]*lds[j];
        cu_xchg[nid]=v;__threadfence();
        float cross=w_cross*cu_xchg[rcu*THREADS+rtid];
        v=tanhf(rec+cross+w_in*u+bias);
        state_out[nid*steps+t]=v;
        __syncthreads();
    }
}

// ============================================================================
// HOST
// ============================================================================
static uint32_t host_rng = 42;
float randf(){host_rng=host_rng*1103515245+12345;return(float)((host_rng>>16)&0x7FFF)/32767.0f;}
float randf_signed(){return randf()*2.0f-1.0f;}

void extract_quad(float* st, int n, int steps, int t, std::vector<float>& f) {
    int stride=n/SAMPLE_N; if(stride<1)stride=1;
    f.clear();
    for(int si=0;si<SAMPLE_N&&si*stride<n;si++){
        float v=st[si*stride*steps+t];
        if(v!=v||v>1e20||v<-1e20)v=0;
        f.push_back(v);
        f.push_back(v*v);
    }
    for(int si=0;si<SAMPLE_N-8&&si*stride<n;si+=8){
        int i1=si*stride,i2=(si+8)*stride;
        if(i2>=n)break;
        float v1=st[i1*steps+t],v2=st[i2*steps+t];
        if(v1!=v1||v1>1e20)v1=0;if(v2!=v2||v2>1e20)v2=0;
        f.push_back(v1*v2);
    }
}

float ridge_regress(std::vector<std::vector<float>>&X,std::vector<float>&y,int nt){
    int n=X.size();if(n<nt+10)return-1;int d=X[0].size();float reg=1.0f;
    std::vector<float>XtX(d*d,0),Xty(d,0);
    for(int i=0;i<nt;i++)for(int a=0;a<d;a++){Xty[a]+=X[i][a]*y[i];for(int b=a;b<d;b++)XtX[a*d+b]+=X[i][a]*X[i][b];}
    for(int a=0;a<d;a++){for(int b=a+1;b<d;b++)XtX[b*d+a]=XtX[a*d+b];XtX[a*d+a]+=reg;}
    std::vector<float>L(d*d,0);
    for(int i=0;i<d;i++)for(int j=0;j<=i;j++){float s=XtX[i*d+j];for(int k=0;k<j;k++)s-=L[i*d+k]*L[j*d+k];if(i==j){if(s<=0)s=1e-6f;L[i*d+j]=sqrtf(s);}else L[i*d+j]=s/L[j*d+j];}
    std::vector<float>z(d),w(d);
    for(int i=0;i<d;i++){float s=Xty[i];for(int k=0;k<i;k++)s-=L[i*d+k]*z[k];z[i]=s/L[i*d+i];}
    for(int i=d-1;i>=0;i--){float s=z[i];for(int k=i+1;k<d;k++)s-=L[k*d+i]*w[k];w[i]=s/L[i*d+i];}
    float sr=0,st2=0,ym=0;int ntt=n-nt;
    for(int i=nt;i<n;i++)ym+=y[i];ym/=ntt;
    for(int i=nt;i<n;i++){float p=0;for(int j=0;j<d;j++)p+=w[j]*X[i][j];float e=y[i]-p;sr+=e*e;st2+=(y[i]-ym)*(y[i]-ym);}
    return st2<1e-10f?0:1-sr/st2;
}

float ridge_classify(std::vector<std::vector<float>>&X,std::vector<int>&y,int nt){
    int n=X.size(),d=X[0].size();
    std::vector<float>yf(n);for(int i=0;i<n;i++)yf[i]=(float)y[i];
    float reg=1.0f;
    std::vector<float>XtX(d*d,0),Xty(d,0);
    for(int i=0;i<nt;i++)for(int a=0;a<d;a++){Xty[a]+=X[i][a]*yf[i];for(int b=a;b<d;b++)XtX[a*d+b]+=X[i][a]*X[i][b];}
    for(int a=0;a<d;a++){for(int b=a+1;b<d;b++)XtX[b*d+a]=XtX[a*d+b];XtX[a*d+a]+=reg;}
    std::vector<float>L(d*d,0);
    for(int i=0;i<d;i++)for(int j=0;j<=i;j++){float s=XtX[i*d+j];for(int k=0;k<j;k++)s-=L[i*d+k]*L[j*d+k];if(i==j){if(s<=0)s=1e-6f;L[i*d+j]=sqrtf(s);}else L[i*d+j]=s/L[j*d+j];}
    std::vector<float>z(d),w(d);
    for(int i=0;i<d;i++){float s=Xty[i];for(int k=0;k<i;k++)s-=L[i*d+k]*z[k];z[i]=s/L[i*d+i];}
    for(int i=d-1;i>=0;i--){float s=z[i];for(int k=i+1;k<d;k++)s-=L[k*d+i]*w[k];w[i]=s/L[i*d+i];}
    int c=0,tot=0;
    for(int i=nt;i<n;i++){float p=0;for(int j=0;j<d;j++)p+=w[j]*X[i][j];if((p>0.5f?1:0)==y[i])c++;tot++;}
    return(float)c/tot;
}

struct Results {
    float xor1, xor2, xor3, xor5, xor8;
    float narma, mc;
    float par3;
};

Results eval_reservoir(
    float* d_state, volatile float* d_xchg, float* d_input,
    float* h_state, float* inp_bin, float* inp_cont, float* narma,
    std::vector<int>* xor_t, std::vector<int>& par3_t,
    bool is_mixed, int stochastic_start_cu, float sr, float is_val
) {
    Results r = {};

    // Continuous tasks
    hipMemcpy(d_input, inp_cont, STEPS*sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_state, 0, N_NEURONS*STEPS*sizeof(float));
    hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
    if (is_mixed)
        esn_mixed<<<N_CUS,THREADS>>>(d_state,d_xchg,d_input,STEPS,sr,is_val,stochastic_start_cu);
    else
        esn_pure<<<N_CUS,THREADS>>>(d_state,d_xchg,d_input,STEPS,sr,is_val);
    hipDeviceSynchronize();
    hipMemcpy(h_state, d_state, N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);

    std::vector<std::vector<float>> fc;
    for(int t=WASHOUT;t<STEPS;t++){std::vector<float>f;extract_quad(h_state,N_NEURONS,STEPS,t,f);fc.push_back(f);}
    int ns=fc.size(),nt=ns*2/3;

    std::vector<float> yn;
    for(int t=WASHOUT;t<STEPS;t++) yn.push_back(narma[t]);
    r.narma = ridge_regress(fc, yn, nt);

    r.mc = 0;
    for(int d=1;d<=10;d++){
        std::vector<float>ym;
        for(int t=WASHOUT;t<STEPS;t++) ym.push_back(t-d>=0?inp_cont[t-d]:0.0f);
        float r2=ridge_regress(fc,ym,nt);
        if(r2>0)r.mc+=r2;
    }

    // Binary tasks
    hipMemcpy(d_input, inp_bin, STEPS*sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_state, 0, N_NEURONS*STEPS*sizeof(float));
    hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
    if (is_mixed)
        esn_mixed<<<N_CUS,THREADS>>>(d_state,d_xchg,d_input,STEPS,sr,is_val,stochastic_start_cu);
    else
        esn_pure<<<N_CUS,THREADS>>>(d_state,d_xchg,d_input,STEPS,sr,is_val);
    hipDeviceSynchronize();
    hipMemcpy(h_state, d_state, N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);

    std::vector<std::vector<float>> fb;
    for(int t=WASHOUT;t<STEPS;t++){std::vector<float>f;extract_quad(h_state,N_NEURONS,STEPS,t,f);fb.push_back(f);}
    ns=fb.size();nt=ns*2/3;

    int delays[]={1,2,3,5,8};
    float* xor_accs[] = {&r.xor1,&r.xor2,&r.xor3,&r.xor5,&r.xor8};
    for(int di=0;di<5;di++){
        int d=delays[di];
        std::vector<int>yt;
        for(int t=WASHOUT;t<STEPS;t++) yt.push_back(xor_t[d-1][t]);
        *xor_accs[di] = ridge_classify(fb,yt,nt);
    }
    std::vector<int>yp;
    for(int t=WASHOUT;t<STEPS;t++) yp.push_back(par3_t[t]);
    r.par3 = ridge_classify(fb,yp,nt);

    return r;
}

int main() {
    printf("=== GPU SILICON ESN v3: MIXED RESERVOIR ===\n");
    printf("  Architecture: deterministic CUs (memory) + stochastic CUs (nonlinearity)\n\n");

    float *d_state, *d_input;
    volatile float *d_xchg;
    hipMalloc(&d_state, N_NEURONS*STEPS*sizeof(float));
    hipMalloc((void**)&d_xchg, N_NEURONS*sizeof(float));
    hipMalloc(&d_input, STEPS*sizeof(float));
    float* h_state = new float[N_NEURONS*STEPS];

    host_rng = 31415;
    float inp_bin[STEPS], inp_cont[STEPS];
    for(int t=0;t<STEPS;t++){inp_bin[t]=randf()>0.5f?1.0f:0.0f;inp_cont[t]=randf_signed()*0.5f;}

    float narma[STEPS]={};
    for(int t=5;t<STEPS;t++){
        narma[t]=0.3f*narma[t-1]+0.05f*narma[t-1]*(narma[t-1]+narma[t-2]+narma[t-3]+narma[t-4]+narma[t-5])
                +1.5f*inp_cont[t-1]*inp_cont[t-5]+0.1f;
        if(narma[t]>10)narma[t]=10;if(narma[t]<-10)narma[t]=-10;
    }

    std::vector<int> xor_t[8], par3(STEPS,0);
    for(int d=1;d<=8;d++){xor_t[d-1].resize(STEPS,0);for(int t=d;t<STEPS;t++)xor_t[d-1][t]=(int)inp_bin[t]^(int)inp_bin[t-d];}
    for(int t=3;t<STEPS;t++) par3[t]=((int)inp_bin[t]^(int)inp_bin[t-1]^(int)inp_bin[t-2]^(int)inp_bin[t-3])&1;

    float SR = 0.99f, IS = 1.0f;

    // ═══ EXPERIMENT 1: Sweep stochastic fraction ═══
    printf("=== EXP 1: Stochastic Fraction (how many CUs use noise?) ===\n");
    printf("Stoch CUs | XOR-1  XOR-2  XOR-3  XOR-5  XOR-8 | NARMA   MC    Par3\n");
    printf("----------+-----------------------------------------+------------------\n");

    // stochastic_start_cu: 12=none, 6=half, 3=three-quarters, 0=all
    int starts[] = {12, 10, 8, 6, 4, 2, 0};
    const char* labels[] = {"0/12", "2/12", "4/12", "6/12", "8/12", "10/12", "12/12"};

    for (int si = 0; si < 7; si++) {
        bool is_mixed = (starts[si] < 12 && starts[si] > 0);
        bool is_all_stoch = (starts[si] == 0);

        Results r;
        if (starts[si] == 12) {
            // Pure deterministic
            r = eval_reservoir(d_state,d_xchg,d_input,h_state,inp_bin,inp_cont,narma,
                             xor_t,par3,false,12,SR,IS);
        } else {
            r = eval_reservoir(d_state,d_xchg,d_input,h_state,inp_bin,inp_cont,narma,
                             xor_t,par3,true,starts[si],SR,IS);
        }

        printf("%-9s | %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.1f%% | %+.3f  %5.2f  %5.1f%%",
               labels[si],
               r.xor1*100,r.xor2*100,r.xor3*100,r.xor5*100,r.xor8*100,
               r.narma,r.mc,r.par3*100);

        if (r.xor5 > 0.90 && r.par3 > 0.60) printf("  ***");
        else if (r.xor5 > 0.70 || r.par3 > 0.70) printf("  **");
        else if (r.xor3 > 0.90) printf("  *");
        printf("\n");
    }

    // ═══ EXPERIMENT 2: Multiple trials for best configs (5 trials) ═══
    printf("\n=== EXP 2: Best configs x5 trials ===\n");

    struct Config {
        const char* name;
        int start_cu;
        bool is_mixed;
    };
    Config cfgs[] = {
        {"PURE_DET", 12, false},
        {"MIXED_6/12", 6, true},
        {"MIXED_4/12", 8, true},
        {"ALL_STOCH", 0, true},
    };

    for (int ci = 0; ci < 4; ci++) {
        float sum[8] = {};
        for (int trial = 0; trial < 5; trial++) {
            Results r;
            if (cfgs[ci].is_mixed || cfgs[ci].start_cu == 0)
                r = eval_reservoir(d_state,d_xchg,d_input,h_state,inp_bin,inp_cont,narma,
                                 xor_t,par3,true,cfgs[ci].start_cu,SR,IS);
            else
                r = eval_reservoir(d_state,d_xchg,d_input,h_state,inp_bin,inp_cont,narma,
                                 xor_t,par3,false,12,SR,IS);
            sum[0]+=r.xor1;sum[1]+=r.xor2;sum[2]+=r.xor3;sum[3]+=r.xor5;
            sum[4]+=r.xor8;sum[5]+=r.narma;sum[6]+=r.mc;sum[7]+=r.par3;
        }
        for(int i=0;i<8;i++)sum[i]/=5;
        printf("%-12s | XOR: %5.1f %5.1f %5.1f %5.1f %5.1f | NARMA=%+.3f MC=%.2f Par3=%5.1f%%\n",
               cfgs[ci].name,
               sum[0]*100,sum[1]*100,sum[2]*100,sum[3]*100,sum[4]*100,
               sum[5],sum[6],sum[7]*100);
    }

    // ═══ EXPERIMENT 3: Is noise reproducible? (same input, different noise → different output?) ═══
    printf("\n=== EXP 3: Noise-induced state diversity (5 runs, same input) ===\n");

    float diversity_det = 0, diversity_mix = 0;
    float states_det[5][SAMPLE_N], states_mix[5][SAMPLE_N];

    for (int trial = 0; trial < 5; trial++) {
        // Deterministic
        hipMemcpy(d_input, inp_bin, STEPS*sizeof(float), hipMemcpyHostToDevice);
        hipMemset(d_state, 0, N_NEURONS*STEPS*sizeof(float));
        hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
        esn_pure<<<N_CUS,THREADS>>>(d_state,d_xchg,d_input,STEPS,SR,IS);
        hipDeviceSynchronize();
        hipMemcpy(h_state, d_state, N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);
        int stride = N_NEURONS/SAMPLE_N;
        for(int si=0;si<SAMPLE_N;si++) states_det[trial][si] = h_state[si*stride*STEPS+(STEPS-1)];

        // Mixed
        hipMemcpy(d_input, inp_bin, STEPS*sizeof(float), hipMemcpyHostToDevice);
        hipMemset(d_state, 0, N_NEURONS*STEPS*sizeof(float));
        hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
        esn_mixed<<<N_CUS,THREADS>>>(d_state,d_xchg,d_input,STEPS,SR,IS,6);
        hipDeviceSynchronize();
        hipMemcpy(h_state, d_state, N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);
        for(int si=0;si<SAMPLE_N;si++) states_mix[trial][si] = h_state[si*stride*STEPS+(STEPS-1)];
    }

    // Compute pairwise state differences
    float det_diff=0, mix_diff=0;
    int pairs = 0;
    for(int a=0;a<5;a++)for(int b=a+1;b<5;b++){
        float dd=0,md=0;
        for(int i=0;i<SAMPLE_N;i++){
            dd += fabsf(states_det[a][i]-states_det[b][i]);
            md += fabsf(states_mix[a][i]-states_mix[b][i]);
        }
        det_diff += dd/SAMPLE_N;
        mix_diff += md/SAMPLE_N;
        pairs++;
    }
    det_diff /= pairs;
    mix_diff /= pairs;

    printf("  Deterministic: mean pairwise state diff = %.6f\n", det_diff);
    printf("  Mixed (6/12):  mean pairwise state diff = %.6f\n", mix_diff);
    printf("  Diversity ratio: %.1fx\n", mix_diff / (det_diff + 1e-10f));
    printf("  -> %s\n", mix_diff > det_diff * 10 ?
           "NOISE CREATES STATE DIVERSITY (stochastic neurons explore state space)" :
           mix_diff > det_diff * 2 ?
           "MODERATE diversity from noise" :
           "MINIMAL diversity (noise effect washed out by deterministic dynamics)");

    printf("\nDone.\n");
    hipFree(d_state); hipFree((void*)d_xchg); hipFree(d_input);
    delete[] h_state;
    return 0;
}
