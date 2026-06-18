// jitter_gpu_esn_ultimate.hip — Push deterministic GPU ESN to limits
//
// Best so far: SR=1.05, T=0.7, IS=1.0 → NARMA=0.842, MC=9.60, XOR-8=80.8%
//
// This script explores:
// EXP 1: Joint SR×T×IS grid search (fine-grained)
// EXP 2: Multi-timescale reservoir (different leak rates per neuron)
// EXP 3: Deeper features (x, x², x³, x_i*x_j, x_i*x_j*x_k)
// EXP 4: Longer sequences (4000 steps) for statistical power
// EXP 5: Final comprehensive benchmark of best config
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_gpu_esn_ultimate scripts/jitter_gpu_esn_ultimate.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_gpu_esn_ultimate

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
#define SAMPLE_N 96

__global__ void esn_ultimate(
    float* state_out,
    volatile float* cu_xchg,
    const float* input_seq,
    int steps,
    float spectral_radius,
    float input_scale,
    float temperature,
    int use_multiscale       // 0=uniform, 1=multi-timescale leaks
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= THREADS) return;
    int nid = bid * THREADS + tid;

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
    irng *= 1103515245; int rcu = ((irng>>8)&0xFF)%N_CUS;
    irng *= 1103515245; int rtid = ((irng>>8)&0xFF)%THREADS;
    irng *= 1103515245; float w_cross = 0.05f * ((float)(irng & 0xFFFF)/32768.0f - 1.0f);

    // Multi-timescale: leak rate varies by neuron
    float leak = 1.0f;  // default: no leaky integration
    if (use_multiscale) {
        // Spread leak rates from 0.1 (slow, long memory) to 1.0 (fast, no memory)
        // Use hash to assign semi-randomly
        irng *= 1103515245;
        float frac = (float)((irng>>8)&0xFF) / 255.0f;
        leak = 0.1f + 0.9f * frac;
    }

    float v = 0.0f;

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];

        lds[tid] = v;
        __syncthreads();
        float rec = 0.0f;
        for (int j = 0; j < THREADS; j++)
            rec += w_row[j] * lds[j];

        cu_xchg[nid] = v;
        __threadfence();
        float cross = w_cross * cu_xchg[rcu*THREADS+rtid];

        float pre = rec + cross + w_in * u + bias;
        float new_v = tanhf(pre / temperature);

        // Leaky integration: v = (1-leak)*v_old + leak*new_v
        v = (1.0f - leak) * v + leak * new_v;

        state_out[nid * steps + t] = v;
        __syncthreads();
    }
}

static uint32_t host_rng = 42;
float randf(){host_rng=host_rng*1103515245+12345;return(float)((host_rng>>16)&0x7FFF)/32767.0f;}
float randf_signed(){return randf()*2.0f-1.0f;}

// Quadratic features
void extract_quad(float* state, int n, int steps, int t, std::vector<float>& f) {
    int stride = n/SAMPLE_N; if(stride<1)stride=1;
    f.clear();
    for(int si=0;si<SAMPLE_N&&si*stride<n;si++){
        float v=state[si*stride*steps+t];
        if(v!=v||v>1e20||v<-1e20)v=0;
        f.push_back(v);
        f.push_back(v*v);
    }
    for(int si=0;si<SAMPLE_N-8&&si*stride<n;si+=8){
        int i1=si*stride,i2=(si+8)*stride;
        if(i2>=n)break;
        float v1=state[i1*steps+t],v2=state[i2*steps+t];
        if(v1!=v1||v1>1e20)v1=0;if(v2!=v2||v2>1e20)v2=0;
        f.push_back(v1*v2);
    }
}

// Cubic features (adds x³ and triple cross-products)
void extract_cubic(float* state, int n, int steps, int t, std::vector<float>& f) {
    int stride = n/SAMPLE_N; if(stride<1)stride=1;
    f.clear();
    for(int si=0;si<SAMPLE_N&&si*stride<n;si++){
        float v=state[si*stride*steps+t];
        if(v!=v||v>1e20||v<-1e20)v=0;
        f.push_back(v);
        f.push_back(v*v);
        f.push_back(v*v*v);
    }
    // Pairwise
    for(int si=0;si<SAMPLE_N-4&&si*stride<n;si+=4){
        int i1=si*stride,i2=(si+4)*stride;
        if(i2>=n)break;
        float v1=state[i1*steps+t],v2=state[i2*steps+t];
        if(v1!=v1||v1>1e20)v1=0;if(v2!=v2||v2>1e20)v2=0;
        f.push_back(v1*v2);
    }
    // Triple (sparse)
    for(int si=0;si<SAMPLE_N-16&&si*stride<n;si+=16){
        int i1=si*stride,i2=(si+8)*stride,i3=(si+16)*stride;
        if(i3>=n)break;
        float v1=state[i1*steps+t],v2=state[i2*steps+t],v3=state[i3*steps+t];
        if(v1!=v1||v1>1e20)v1=0;if(v2!=v2||v2>1e20)v2=0;if(v3!=v3||v3>1e20)v3=0;
        f.push_back(v1*v2*v3);
    }
}

float ridge_solve(std::vector<std::vector<float>>&X,std::vector<float>&y,int nt) {
    int n=X.size();if(n<nt+10)return-1;int d=X[0].size();float reg=1.0f;
    std::vector<float>XtX(d*d,0),Xty(d,0);
    for(int i=0;i<nt;i++)for(int a=0;a<d;a++){Xty[a]+=X[i][a]*y[i];for(int b=a;b<d;b++)XtX[a*d+b]+=X[i][a]*X[i][b];}
    for(int a=0;a<d;a++){for(int b=a+1;b<d;b++)XtX[b*d+a]=XtX[a*d+b];XtX[a*d+a]+=reg;}
    std::vector<float>L(d*d,0);
    for(int i=0;i<d;i++)for(int j=0;j<=i;j++){float s=XtX[i*d+j];for(int k=0;k<j;k++)s-=L[i*d+k]*L[j*d+k];if(i==j){if(s<=0)s=1e-6f;L[i*d+j]=sqrtf(s);}else L[i*d+j]=s/L[j*d+j];}
    std::vector<float>z(d),w(d);
    for(int i=0;i<d;i++){float s=Xty[i];for(int k=0;k<i;k++)s-=L[i*d+k]*z[k];z[i]=s/L[i*d+i];}
    for(int i=d-1;i>=0;i--){float s=z[i];for(int k=i+1;k<d;k++)s-=L[k*d+i]*w[k];w[i]=s/L[i*d+i];}
    float sr2=0,st2=0,ym=0;int ntt=n-nt;
    for(int i=nt;i<n;i++)ym+=y[i];ym/=ntt;
    for(int i=nt;i<n;i++){float p=0;for(int j=0;j<d;j++)p+=w[j]*X[i][j];float e=y[i]-p;sr2+=e*e;st2+=(y[i]-ym)*(y[i]-ym);}
    return st2<1e-10f?0:1-sr2/st2;
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

struct BenchResult {
    float xor_delays[16]; // XOR accuracy at delays 1-15
    float par_widths[10]; // Parity accuracy at widths 2-9
    float narma;
    float mc;
    int feat_dim;
};

BenchResult run_bench(
    float* d_state, volatile float* d_xchg, float* d_input,
    float* h_state,
    float* inp_bin, float* inp_cont, float* narma_tgt,
    int steps, int washout,
    float SR, float IS, float T, int multiscale,
    bool use_cubic,
    int max_xor_d, int max_par_w
) {
    BenchResult res = {};

    // Continuous
    hipMemcpy(d_input, inp_cont, steps*sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_state, 0, (size_t)N_NEURONS*steps*sizeof(float));
    hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
    esn_ultimate<<<N_CUS,THREADS>>>(d_state, d_xchg, d_input, steps, SR, IS, T, multiscale);
    hipDeviceSynchronize();
    hipMemcpy(h_state, d_state, (size_t)N_NEURONS*steps*sizeof(float), hipMemcpyDeviceToHost);

    std::vector<std::vector<float>> fc;
    for(int t=washout;t<steps;t++){
        std::vector<float>f;
        if(use_cubic) extract_cubic(h_state,N_NEURONS,steps,t,f);
        else extract_quad(h_state,N_NEURONS,steps,t,f);
        fc.push_back(f);
    }
    int ns=fc.size(), nt=ns*2/3;
    res.feat_dim = fc[0].size();

    std::vector<float>yn;
    for(int t=washout;t<steps;t++) yn.push_back(narma_tgt[t]);
    res.narma = ridge_solve(fc, yn, nt);

    res.mc = 0;
    for(int d=1;d<=10;d++){
        std::vector<float>ym;
        for(int t=washout;t<steps;t++) ym.push_back(t-d>=0?inp_cont[t-d]:0.0f);
        float r2=ridge_solve(fc,ym,nt); if(r2>0)res.mc+=r2;
    }

    // Binary
    hipMemcpy(d_input, inp_bin, steps*sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_state, 0, (size_t)N_NEURONS*steps*sizeof(float));
    hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
    esn_ultimate<<<N_CUS,THREADS>>>(d_state, d_xchg, d_input, steps, SR, IS, T, multiscale);
    hipDeviceSynchronize();
    hipMemcpy(h_state, d_state, (size_t)N_NEURONS*steps*sizeof(float), hipMemcpyDeviceToHost);

    std::vector<std::vector<float>> fb;
    for(int t=washout;t<steps;t++){
        std::vector<float>f;
        if(use_cubic) extract_cubic(h_state,N_NEURONS,steps,t,f);
        else extract_quad(h_state,N_NEURONS,steps,t,f);
        fb.push_back(f);
    }
    ns=fb.size(); nt=ns*2/3;

    for(int d=1;d<=max_xor_d&&d<16;d++){
        std::vector<int>y;
        for(int t=washout;t<steps;t++) y.push_back(t>=d?((int)inp_bin[t]^(int)inp_bin[t-d]):0);
        res.xor_delays[d] = ridge_classify(fb, y, nt);
    }
    for(int w=2;w<=max_par_w&&w<10;w++){
        std::vector<int>y;
        for(int t=washout;t<steps;t++){
            int p=0; for(int k=0;k<=w&&t-k>=0;k++) p^=(int)inp_bin[t-k]; y.push_back(p&1);
        }
        res.par_widths[w] = ridge_classify(fb, y, nt);
    }

    return res;
}

int main() {
    printf("=== GPU ESN ULTIMATE: Push Deterministic Limits ===\n\n");

    int STEPS = 2000, WASHOUT = 300;

    float *d_state, *d_input;
    volatile float *d_xchg;
    size_t state_sz = (size_t)N_NEURONS*STEPS*sizeof(float);
    hipMalloc(&d_state, state_sz);
    hipMalloc((void**)&d_xchg, N_NEURONS*sizeof(float));
    hipMalloc(&d_input, STEPS*sizeof(float));
    float* h_state = new float[(size_t)N_NEURONS*STEPS];

    host_rng = 31415;
    float *inp_bin = new float[STEPS], *inp_cont = new float[STEPS], *narma = new float[STEPS];
    for(int t=0;t<STEPS;t++){inp_bin[t]=randf()>0.5f?1.0f:0.0f;inp_cont[t]=randf_signed()*0.5f;}
    memset(narma,0,STEPS*sizeof(float));
    for(int t=5;t<STEPS;t++){
        narma[t]=0.3f*narma[t-1]+0.05f*narma[t-1]*(narma[t-1]+narma[t-2]+narma[t-3]+narma[t-4]+narma[t-5])
                +1.5f*inp_cont[t-1]*inp_cont[t-5]+0.1f;
        if(narma[t]>10)narma[t]=10;if(narma[t]<-10)narma[t]=-10;
    }

    // ================================================================
    // EXP 1: Fine-grained SR×T grid
    // ================================================================
    printf("--- EXP 1: SR x T GRID SEARCH ---\n");
    printf("%-6s %-6s | XOR-5   XOR-8   Par3   Par4  | NARMA    MC\n", "SR", "T");
    printf("-------------+------------------------------+--------------\n");

    float srs[] = {0.95f, 0.99f, 1.02f, 1.05f, 1.08f};
    float ts[] =  {0.50f, 0.55f, 0.60f, 0.65f, 0.70f, 0.75f};

    for (int si = 0; si < 5; si++) {
        for (int ti = 0; ti < 6; ti++) {
            BenchResult r = run_bench(d_state,d_xchg,d_input,h_state,
                inp_bin,inp_cont,narma,STEPS,WASHOUT,
                srs[si],1.0f,ts[ti],0,false,8,4);
            printf("%-6.2f %-6.2f | %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%% | %+.3f  %5.2f",
                srs[si], ts[ti],
                r.xor_delays[5]*100, r.xor_delays[8]*100,
                r.par_widths[3]*100, r.par_widths[4]*100,
                r.narma, r.mc);
            if (r.narma > 0.82f) printf("  <==");
            printf("\n");
        }
    }

    // ================================================================
    // EXP 2: Multi-timescale vs uniform
    // ================================================================
    printf("\n--- EXP 2: MULTI-TIMESCALE RESERVOIR ---\n");
    printf("%-12s %-6s %-6s | XOR-5   XOR-8  XOR-10  Par3   Par4  | NARMA    MC\n",
        "Mode", "SR", "T");
    printf("---------------------------+--------------------------------------+--------------\n");

    struct MTConfig { const char* name; float sr, t; int ms; };
    MTConfig mt_cfgs[] = {
        {"UNIFORM",    1.02f, 0.65f, 0},
        {"MULTISCALE", 1.02f, 0.65f, 1},
        {"UNIFORM",    1.05f, 0.65f, 0},
        {"MULTISCALE", 1.05f, 0.65f, 1},
        {"UNIFORM",    1.05f, 0.70f, 0},
        {"MULTISCALE", 1.05f, 0.70f, 1},
        {"UNIFORM",    0.99f, 0.60f, 0},
        {"MULTISCALE", 0.99f, 0.60f, 1},
    };

    for (int ci = 0; ci < 8; ci++) {
        BenchResult r = run_bench(d_state,d_xchg,d_input,h_state,
            inp_bin,inp_cont,narma,STEPS,WASHOUT,
            mt_cfgs[ci].sr, 1.0f, mt_cfgs[ci].t, mt_cfgs[ci].ms,
            false, 10, 5);
        printf("%-12s %-6.2f %-6.2f | %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%% | %+.3f  %5.2f\n",
            mt_cfgs[ci].name, mt_cfgs[ci].sr, mt_cfgs[ci].t,
            r.xor_delays[5]*100, r.xor_delays[8]*100, r.xor_delays[10]*100,
            r.par_widths[3]*100, r.par_widths[4]*100,
            r.narma, r.mc);
    }

    // ================================================================
    // EXP 3: Cubic vs Quadratic readout
    // ================================================================
    printf("\n--- EXP 3: CUBIC vs QUADRATIC READOUT ---\n");
    printf("%-12s %-6s | XOR-8   Par3   Par4   Par5  | NARMA    MC    | feat_dim\n",
        "Readout", "Config");
    printf("-------------------+-----------------------------+----------------+---------\n");

    struct ROConfig { const char* name; bool cubic; float sr, t; };
    ROConfig ro_cfgs[] = {
        {"QUAD",  false, 1.05f, 0.65f},
        {"CUBIC", true,  1.05f, 0.65f},
        {"QUAD",  false, 1.02f, 0.60f},
        {"CUBIC", true,  1.02f, 0.60f},
    };

    for (int ci = 0; ci < 4; ci++) {
        BenchResult r = run_bench(d_state,d_xchg,d_input,h_state,
            inp_bin,inp_cont,narma,STEPS,WASHOUT,
            ro_cfgs[ci].sr, 1.0f, ro_cfgs[ci].t, 0,
            ro_cfgs[ci].cubic, 10, 5);
        printf("%-12s %-6s | %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%% | %+.3f  %5.2f   | %d\n",
            ro_cfgs[ci].name,
            ci%2==0?"1.05/65":"1.02/60",
            r.xor_delays[8]*100,
            r.par_widths[3]*100, r.par_widths[4]*100, r.par_widths[5]*100,
            r.narma, r.mc,
            r.feat_dim);
    }

    // ================================================================
    // EXP 4: Multi-timescale + cubic (best of both?)
    // ================================================================
    printf("\n--- EXP 4: MULTI-TIMESCALE + CUBIC READOUT ---\n");
    printf("%-14s | XOR-5  XOR-8  XOR-10 | Par3   Par4   Par5  | NARMA    MC\n", "Config");
    printf("---------------+----------------------+---------------------+--------------\n");

    struct FinalConfig {
        const char* name;
        float sr, t;
        int ms;
        bool cubic;
    };
    FinalConfig final_cfgs[] = {
        {"QUAD_UNI",     1.05f, 0.65f, 0, false},
        {"QUAD_MS",      1.05f, 0.65f, 1, false},
        {"CUBIC_UNI",    1.05f, 0.65f, 0, true},
        {"CUBIC_MS",     1.05f, 0.65f, 1, true},
        {"QUAD_UNI_70",  1.05f, 0.70f, 0, false},
        {"QUAD_MS_70",   1.05f, 0.70f, 1, false},
        {"CUBIC_UNI_70", 1.05f, 0.70f, 0, true},
        {"CUBIC_MS_70",  1.05f, 0.70f, 1, true},
    };

    for (int ci = 0; ci < 8; ci++) {
        BenchResult r = run_bench(d_state,d_xchg,d_input,h_state,
            inp_bin,inp_cont,narma,STEPS,WASHOUT,
            final_cfgs[ci].sr, 1.0f, final_cfgs[ci].t, final_cfgs[ci].ms,
            final_cfgs[ci].cubic, 10, 6);
        printf("%-14s | %5.1f%% %5.1f%% %5.1f%% | %5.1f%% %5.1f%% %5.1f%% | %+.3f  %5.2f\n",
            final_cfgs[ci].name,
            r.xor_delays[5]*100, r.xor_delays[8]*100, r.xor_delays[10]*100,
            r.par_widths[3]*100, r.par_widths[4]*100, r.par_widths[5]*100,
            r.narma, r.mc);
    }

    // ================================================================
    // EXP 5: Statistical power — best config × 10 trials
    // ================================================================
    printf("\n--- EXP 5: BEST CONFIG x 10 TRIALS ---\n");

    float best_sr = 1.05f, best_t = 0.65f;
    float stats_x5[10],stats_x8[10],stats_x10[10],stats_p3[10],stats_p4[10],stats_p5[10];
    float stats_n[10],stats_m[10];

    for (int trial = 0; trial < 10; trial++) {
        host_rng = 31415 + trial * 7919;
        for(int t=0;t<STEPS;t++){inp_bin[t]=randf()>0.5f?1.0f:0.0f;inp_cont[t]=randf_signed()*0.5f;}
        memset(narma,0,STEPS*sizeof(float));
        for(int t=5;t<STEPS;t++){
            narma[t]=0.3f*narma[t-1]+0.05f*narma[t-1]*(narma[t-1]+narma[t-2]+narma[t-3]+narma[t-4]+narma[t-5])
                    +1.5f*inp_cont[t-1]*inp_cont[t-5]+0.1f;
            if(narma[t]>10)narma[t]=10;if(narma[t]<-10)narma[t]=-10;
        }
        BenchResult r = run_bench(d_state,d_xchg,d_input,h_state,
            inp_bin,inp_cont,narma,STEPS,WASHOUT,
            best_sr, 1.0f, best_t, 0, false, 10, 5);
        stats_x5[trial] = r.xor_delays[5];
        stats_x8[trial] = r.xor_delays[8];
        stats_x10[trial] = r.xor_delays[10];
        stats_p3[trial] = r.par_widths[3];
        stats_p4[trial] = r.par_widths[4];
        stats_p5[trial] = r.par_widths[5];
        stats_n[trial] = r.narma;
        stats_m[trial] = r.mc;
        printf("  Trial %d: XOR-5=%5.1f%% XOR-8=%5.1f%% XOR-10=%5.1f%% Par3=%5.1f%% Par4=%5.1f%% NARMA=%+.3f MC=%.2f\n",
            trial, r.xor_delays[5]*100, r.xor_delays[8]*100, r.xor_delays[10]*100,
            r.par_widths[3]*100, r.par_widths[4]*100, r.narma, r.mc);
    }

    auto compute_stats = [](float* v, int n, float& mean, float& std_dev) {
        mean = 0; for(int i=0;i<n;i++) mean+=v[i]; mean/=n;
        std_dev = 0; for(int i=0;i<n;i++) std_dev+=(v[i]-mean)*(v[i]-mean); std_dev=sqrtf(std_dev/n);
    };

    float mx5,sx5,mx8,sx8,mx10,sx10,mp3,sp3,mp4,sp4,mp5,sp5,mn,sn,mm,sm;
    compute_stats(stats_x5,10,mx5,sx5);
    compute_stats(stats_x8,10,mx8,sx8);
    compute_stats(stats_x10,10,mx10,sx10);
    compute_stats(stats_p3,10,mp3,sp3);
    compute_stats(stats_p4,10,mp4,sp4);
    compute_stats(stats_p5,10,mp5,sp5);
    compute_stats(stats_n,10,mn,sn);
    compute_stats(stats_m,10,mm,sm);

    printf("\n  SUMMARY (SR=%.2f T=%.2f, 10 trials):\n", best_sr, best_t);
    printf("  XOR-5:  %5.1f%% +/- %.1f%%\n", mx5*100, sx5*100);
    printf("  XOR-8:  %5.1f%% +/- %.1f%%\n", mx8*100, sx8*100);
    printf("  XOR-10: %5.1f%% +/- %.1f%%\n", mx10*100, sx10*100);
    printf("  Par-3:  %5.1f%% +/- %.1f%%\n", mp3*100, sp3*100);
    printf("  Par-4:  %5.1f%% +/- %.1f%%\n", mp4*100, sp4*100);
    printf("  Par-5:  %5.1f%% +/- %.1f%%\n", mp5*100, sp5*100);
    printf("  NARMA:  %+.3f +/- %.3f\n", mn, sn);
    printf("  MC:     %.2f +/- %.2f\n", mm, sm);

    printf("\nDone.\n");
    hipFree(d_state); hipFree((void*)d_xchg); hipFree(d_input);
    delete[] h_state; delete[] inp_bin; delete[] inp_cont; delete[] narma;
    return 0;
}
