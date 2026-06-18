// jitter_gpu_lif_noise.hip — True LIF neurons with per-neuron decorrelated noise
//
// KEY INSIGHT: Previous experiments used tanh-ESN (continuous), not LIF (spiking).
// AND: PLL noise was shared across all neurons (correlation=1.0).
//
// This script tests:
// 1. ACTUAL LIF dynamics (integrate, threshold, spike, reset, refractory)
// 2. Per-neuron DECORRELATED noise via hash(wall_clock64() + neuron_id)
//    → Same PLL entropy source but hash creates independent streams
// 3. Noise injection into membrane potential (thermal noise) AND
//    threshold (stochastic spiking) AND synaptic weights (unreliable synapses)
//
// EXP 1: LIF vs tanh-ESN comparison (matched parameters)
// EXP 2: Decorrelated noise vs shared noise vs no noise (LIF)
// EXP 3: Where to inject noise: membrane vs threshold vs synaptic
// EXP 4: Noise amplitude sweep for best decorrelated config
// EXP 5: Hard benchmarks (XOR, parity, NARMA, MC) with best LIF config
// EXP 6: Statistical reliability (10 trials)
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_gpu_lif_noise scripts/jitter_gpu_lif_noise.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_gpu_lif_noise

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
#define STEPS 2000
#define WASHOUT 300
#define SAMPLE_N 96

// Noise modes
#define NOISE_NONE 0
#define NOISE_SHARED 1      // raw wall_clock64, same for all lanes
#define NOISE_DECORR 2      // hash(wall_clock64 + nid), independent per neuron
#define NOISE_PRNG 3        // pure PRNG (no physics), control for decorrelation

// Injection sites
#define INJECT_MEMBRANE 1   // noise added to membrane potential
#define INJECT_THRESHOLD 2  // noise modulates firing threshold
#define INJECT_SYNAPTIC 4   // noise modulates synaptic weights
#define INJECT_ALL 7        // all three

// Dynamics modes
#define DYN_TANH 0          // continuous tanh ESN (baseline)
#define DYN_LIF 1           // leaky integrate-and-fire with spikes

__global__ void esn_lif_probe(
    float* state_out,          // [N_NEURONS × STEPS] — membrane voltage or tanh state
    float* spike_out,          // [N_NEURONS × STEPS] — spike train (LIF) or state copy (tanh)
    volatile float* cu_xchg,
    const float* input_seq,
    int steps,
    int dynamics,              // DYN_TANH or DYN_LIF
    int noise_mode,            // NOISE_NONE/SHARED/DECORR/PRNG
    int inject_site,           // INJECT_MEMBRANE/THRESHOLD/SYNAPTIC/ALL
    float noise_amp,           // noise amplitude
    float spectral_radius,
    float input_scale,
    float temperature,         // tanh gain (for DYN_TANH)
    float lif_leak,            // membrane leak rate (for DYN_LIF)
    float lif_threshold,       // firing threshold (for DYN_LIF)
    float lif_reset            // reset potential (for DYN_LIF)
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
    float bias = 0.01f * ((float)(irng & 0xFFFF)/32768.0f - 1.0f);
    irng *= 1103515245; int rcu = ((irng>>8)&0xFF)%N_CUS;
    irng *= 1103515245; int rtid = ((irng>>8)&0xFF)%THREADS;
    irng *= 1103515245; float w_cross = 0.05f * ((float)(irng & 0xFFFF)/32768.0f - 1.0f);

    // Per-neuron heterogeneous threshold (biological: neurons have different sensitivities)
    irng *= 1103515245;
    float neuron_thresh = lif_threshold + 0.1f * ((float)(irng & 0xFFFF)/32768.0f - 1.0f);

    float v = 0.0f;         // membrane potential (LIF) or state (tanh)
    float spike = 0.0f;     // spike output
    int refractory = 0;     // refractory counter
    uint32_t prng_state = nid * 2654435761u + 12345;  // per-neuron PRNG

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];

        // ========== NOISE GENERATION ==========
        float noise = 0.0f;
        if (noise_mode != NOISE_NONE) {
            if (noise_mode == NOISE_SHARED) {
                // Raw PLL — same for all lanes in wavefront
                uint64_t w0 = wall_clock64();
                uint64_t w1 = wall_clock64();
                noise = (float)(w1 - w0) - 2.2f;
            }
            else if (noise_mode == NOISE_DECORR) {
                // DECORRELATED: hash(PLL + neuron_id)
                // Sample PLL multiple times for richer entropy
                uint64_t pll_bits = 0;
                for (int r = 0; r < 8; r++) {
                    uint64_t w0 = wall_clock64();
                    uint64_t w1 = wall_clock64();
                    pll_bits = (pll_bits << 4) | ((w1 - w0) & 0xF);
                }
                // Hash with neuron ID to decorrelate
                uint32_t h = (uint32_t)(pll_bits ^ (uint64_t)(nid * 2654435761u));
                h ^= h >> 16; h *= 0x85ebca6b;
                h ^= h >> 13; h *= 0xc2b2ae35;
                h ^= h >> 16;
                // Also fold in timestep for temporal decorrelation
                h ^= (uint32_t)(t * 1103515245);
                h *= 0x45d9f3b; h ^= h >> 16;
                noise = ((float)(h & 0xFFFF) / 32768.0f - 1.0f);
            }
            else if (noise_mode == NOISE_PRNG) {
                // Pure PRNG (no PLL physics) — control for independence
                prng_state = prng_state * 1103515245u + 12345u;
                noise = ((float)((prng_state >> 16) & 0xFFFF) / 32768.0f - 1.0f);
            }
        }

        // ========== RECURRENCE ==========
        // For LIF: share spikes (not membrane potential)
        lds[tid] = (dynamics == DYN_LIF) ? spike : v;
        __syncthreads();

        float rec = 0.0f;
        float syn_noise_factor = 1.0f;
        if ((inject_site & INJECT_SYNAPTIC) && noise_mode != NOISE_NONE) {
            // Synaptic noise: each weight gets noise modulation
            // Use per-neuron hash to make it independent
            uint32_t syn_seed = (uint32_t)(nid * 7919 + t * 3571);
            for (int j = 0; j < THREADS; j++) {
                uint32_t sh = syn_seed + j * 6271;
                sh ^= sh >> 16; sh *= 0x45d9f3b;
                float syn_n = 1.0f + noise_amp * 0.5f * ((float)(sh & 0xFFFF)/32768.0f - 1.0f);
                rec += syn_n * w_row[j] * lds[j];
            }
        } else {
            for (int j = 0; j < THREADS; j++)
                rec += w_row[j] * lds[j];
        }

        cu_xchg[nid] = (dynamics == DYN_LIF) ? spike : v;
        __threadfence();
        float cross = w_cross * cu_xchg[rcu*THREADS+rtid];

        float current = rec + cross + w_in * u + bias;

        // ========== DYNAMICS ==========
        if (dynamics == DYN_TANH) {
            // Standard tanh ESN with gain
            float pre = current;
            if ((inject_site & INJECT_MEMBRANE) && noise_mode != NOISE_NONE)
                pre += noise_amp * noise;
            v = tanhf(pre / temperature);
            spike = v;  // no separate spike for tanh
        }
        else {  // DYN_LIF
            spike = 0.0f;

            if (refractory > 0) {
                refractory--;
                v *= (1.0f - lif_leak);  // still leaks during refractory
            } else {
                // Leaky integration
                v = v * (1.0f - lif_leak) + current * lif_leak;

                // Membrane noise
                if ((inject_site & INJECT_MEMBRANE) && noise_mode != NOISE_NONE)
                    v += noise_amp * noise;

                // Threshold (possibly noisy)
                float thresh = neuron_thresh;
                if ((inject_site & INJECT_THRESHOLD) && noise_mode != NOISE_NONE)
                    thresh += noise_amp * 0.3f * noise;

                // Spike!
                if (v >= thresh) {
                    spike = 1.0f;
                    v = lif_reset;
                    refractory = 2;  // 2-step refractory
                }
            }
        }

        state_out[nid * steps + t] = v;
        spike_out[nid * steps + t] = spike;
        __syncthreads();
    }
}

// ============================================================================
// HOST
// ============================================================================
static uint32_t host_rng = 42;
float randf(){host_rng=host_rng*1103515245+12345;return(float)((host_rng>>16)&0x7FFF)/32767.0f;}
float randf_signed(){return randf()*2.0f-1.0f;}

void extract_features(float* state, float* spikes, int n, int steps, int t,
                      std::vector<float>& f, bool use_spikes) {
    int stride = n/SAMPLE_N; if(stride<1)stride=1;
    f.clear();
    for(int si=0;si<SAMPLE_N&&si*stride<n;si++){
        float v = state[si*stride*steps+t];
        if(v!=v||v>1e20||v<-1e20)v=0;
        f.push_back(v);
        f.push_back(v*v);
        if (use_spikes) {
            // Spike features: current spike, recent spike count (sliding window)
            float s = spikes[si*stride*steps+t];
            f.push_back(s);
            // Spike count in last 5 steps
            float sc = 0;
            for (int dt = 0; dt < 5 && t-dt >= 0; dt++)
                sc += spikes[si*stride*steps+(t-dt)];
            f.push_back(sc / 5.0f);
        }
    }
    // Cross-products
    for(int si=0;si<SAMPLE_N-8&&si*stride<n;si+=8){
        int i1=si*stride,i2=(si+8)*stride;
        if(i2>=n)break;
        float v1=state[i1*steps+t],v2=state[i2*steps+t];
        if(v1!=v1||v1>1e20)v1=0;if(v2!=v2||v2>1e20)v2=0;
        f.push_back(v1*v2);
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

struct Metrics {
    float xor5, xor8, par3, par4, narma, mc;
    float spike_rate;  // mean spikes per neuron per step
    float spike_cv;    // CV of spike rates across neurons (heterogeneity)
};

Metrics run_test(
    float* d_state, float* d_spike, volatile float* d_xchg, float* d_input,
    float* h_state, float* h_spike,
    float* inp_bin, float* inp_cont, float* narma_tgt,
    int dynamics, int noise_mode, int inject_site, float noise_amp,
    float SR, float IS, float T, float leak, float thresh, float reset
) {
    Metrics m = {};

    // Continuous
    hipMemcpy(d_input, inp_cont, STEPS*sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_state, 0, (size_t)N_NEURONS*STEPS*sizeof(float));
    hipMemset(d_spike, 0, (size_t)N_NEURONS*STEPS*sizeof(float));
    hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
    esn_lif_probe<<<N_CUS,THREADS>>>(d_state, d_spike, d_xchg, d_input, STEPS,
        dynamics, noise_mode, inject_site, noise_amp, SR, IS, T, leak, thresh, reset);
    hipDeviceSynchronize();
    hipMemcpy(h_state, d_state, (size_t)N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_spike, d_spike, (size_t)N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);

    bool is_lif = (dynamics == DYN_LIF);
    std::vector<std::vector<float>> fc;
    for(int t=WASHOUT;t<STEPS;t++){
        std::vector<float>f;
        extract_features(h_state, h_spike, N_NEURONS, STEPS, t, f, is_lif);
        fc.push_back(f);
    }
    int ns=fc.size(), nt=ns*2/3;

    // Spike statistics
    if (is_lif) {
        std::vector<float> rates;
        for(int n=0;n<N_NEURONS;n++){
            float sc=0;
            for(int t=WASHOUT;t<STEPS;t++) sc+=h_spike[n*STEPS+t];
            rates.push_back(sc/(STEPS-WASHOUT));
        }
        float mean_r=0; for(auto r:rates) mean_r+=r; mean_r/=rates.size();
        float var_r=0; for(auto r:rates) var_r+=(r-mean_r)*(r-mean_r); var_r/=rates.size();
        m.spike_rate = mean_r;
        m.spike_cv = (mean_r > 0.001f) ? sqrtf(var_r)/mean_r : 0;
    }

    std::vector<float>yn;
    for(int t=WASHOUT;t<STEPS;t++) yn.push_back(narma_tgt[t]);
    m.narma = ridge_solve(fc, yn, nt);

    m.mc = 0;
    for(int d=1;d<=10;d++){
        std::vector<float>ym;
        for(int t=WASHOUT;t<STEPS;t++) ym.push_back(t-d>=0?inp_cont[t-d]:0.0f);
        float r2=ridge_solve(fc,ym,nt); if(r2>0)m.mc+=r2;
    }

    // Binary
    hipMemcpy(d_input, inp_bin, STEPS*sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_state, 0, (size_t)N_NEURONS*STEPS*sizeof(float));
    hipMemset(d_spike, 0, (size_t)N_NEURONS*STEPS*sizeof(float));
    hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
    esn_lif_probe<<<N_CUS,THREADS>>>(d_state, d_spike, d_xchg, d_input, STEPS,
        dynamics, noise_mode, inject_site, noise_amp, SR, IS, T, leak, thresh, reset);
    hipDeviceSynchronize();
    hipMemcpy(h_state, d_state, (size_t)N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_spike, d_spike, (size_t)N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);

    std::vector<std::vector<float>> fb;
    for(int t=WASHOUT;t<STEPS;t++){
        std::vector<float>f;
        extract_features(h_state, h_spike, N_NEURONS, STEPS, t, f, is_lif);
        fb.push_back(f);
    }
    ns=fb.size(); nt=ns*2/3;

    std::vector<int>y5,y8,yp3,yp4;
    for(int t=WASHOUT;t<STEPS;t++){
        y5.push_back(t>=5?((int)inp_bin[t]^(int)inp_bin[t-5]):0);
        y8.push_back(t>=8?((int)inp_bin[t]^(int)inp_bin[t-8]):0);
        int p3=0; for(int k=0;k<=3&&t-k>=0;k++) p3^=(int)inp_bin[t-k]; yp3.push_back(p3&1);
        int p4=0; for(int k=0;k<=4&&t-k>=0;k++) p4^=(int)inp_bin[t-k]; yp4.push_back(p4&1);
    }
    m.xor5 = ridge_classify(fb,y5,nt);
    m.xor8 = ridge_classify(fb,y8,nt);
    m.par3 = ridge_classify(fb,yp3,nt);
    m.par4 = ridge_classify(fb,yp4,nt);

    return m;
}

int main() {
    printf("=== GPU LIF WITH DECORRELATED NOISE ===\n\n");

    float *d_state, *d_spike, *d_input;
    volatile float *d_xchg;
    size_t sz = (size_t)N_NEURONS*STEPS*sizeof(float);
    hipMalloc(&d_state, sz);
    hipMalloc(&d_spike, sz);
    hipMalloc((void**)&d_xchg, N_NEURONS*sizeof(float));
    hipMalloc(&d_input, STEPS*sizeof(float));
    float* h_state = new float[N_NEURONS*STEPS];
    float* h_spike = new float[N_NEURONS*STEPS];

    host_rng = 31415;
    float inp_bin[STEPS], inp_cont[STEPS], narma[STEPS];
    for(int t=0;t<STEPS;t++){inp_bin[t]=randf()>0.5f?1.0f:0.0f;inp_cont[t]=randf_signed()*0.5f;}
    memset(narma,0,sizeof(narma));
    for(int t=5;t<STEPS;t++){
        narma[t]=0.3f*narma[t-1]+0.05f*narma[t-1]*(narma[t-1]+narma[t-2]+narma[t-3]+narma[t-4]+narma[t-5])
                +1.5f*inp_cont[t-1]*inp_cont[t-5]+0.1f;
        if(narma[t]>10)narma[t]=10;if(narma[t]<-10)narma[t]=-10;
    }

    // ================================================================
    // EXP 1: LIF vs tanh-ESN baseline comparison
    // ================================================================
    printf("--- EXP 1: LIF vs TANH-ESN (no noise) ---\n");
    printf("%-14s | XOR-5   XOR-8   Par3   Par4  | NARMA    MC    | SpikeRate  CV\n", "Mode");
    printf("---------------+------------------------------+---------------+-------------\n");

    // Tanh baseline (best known: SR=1.05, T=0.65)
    Metrics tanh_base = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
        inp_bin,inp_cont,narma,
        DYN_TANH, NOISE_NONE, 0, 0.0f,
        1.05f, 1.0f, 0.65f, 0.0f, 0.0f, 0.0f);
    printf("TANH(T=0.65)   | %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%% | %+.3f  %5.2f  | n/a\n",
        tanh_base.xor5*100, tanh_base.xor8*100, tanh_base.par3*100, tanh_base.par4*100,
        tanh_base.narma, tanh_base.mc);

    // LIF with various parameters
    struct LIFConfig {
        const char* name;
        float sr, is, leak, thresh, reset;
    };
    LIFConfig lif_cfgs[] = {
        {"LIF_fast",     0.10f, 0.5f, 0.20f, 1.0f, 0.0f},
        {"LIF_med",      0.10f, 0.5f, 0.10f, 1.0f, 0.0f},
        {"LIF_slow",     0.10f, 0.5f, 0.05f, 1.0f, 0.0f},
        {"LIF_low_th",   0.10f, 0.5f, 0.10f, 0.5f, 0.0f},
        {"LIF_hi_th",    0.10f, 0.5f, 0.10f, 2.0f, 0.0f},
        {"LIF_neg_rst",  0.10f, 0.5f, 0.10f, 1.0f, -0.5f},
        {"LIF_dense",    0.30f, 1.0f, 0.10f, 1.0f, 0.0f},
        {"LIF_dense2",   0.50f, 1.0f, 0.10f, 1.0f, 0.0f},
        {"LIF_dense3",   0.20f, 1.0f, 0.05f, 0.5f, -0.2f},
    };

    for (int ci = 0; ci < 9; ci++) {
        Metrics m = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
            inp_bin,inp_cont,narma,
            DYN_LIF, NOISE_NONE, 0, 0.0f,
            lif_cfgs[ci].sr, lif_cfgs[ci].is, 1.0f,
            lif_cfgs[ci].leak, lif_cfgs[ci].thresh, lif_cfgs[ci].reset);
        printf("%-14s | %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%% | %+.3f  %5.2f  | %.4f  %.2f\n",
            lif_cfgs[ci].name,
            m.xor5*100, m.xor8*100, m.par3*100, m.par4*100,
            m.narma, m.mc, m.spike_rate, m.spike_cv);
    }

    // ================================================================
    // EXP 2: Decorrelated vs Shared vs PRNG noise (LIF)
    // ================================================================
    printf("\n--- EXP 2: NOISE TYPE COMPARISON (LIF, membrane injection) ---\n");
    printf("%-14s | XOR-5   XOR-8   Par3   Par4  | NARMA    MC    | SpikeRate  CV\n", "NoiseMode");
    printf("---------------+------------------------------+---------------+-------------\n");

    // Use LIF_dense3 as base (or whichever works best from EXP 1)
    float best_sr=0.20f, best_is=1.0f, best_leak=0.05f, best_thresh=0.5f, best_reset=-0.2f;

    const char* noise_names[] = {"NONE", "SHARED", "DECORR", "PRNG"};
    float noise_amps[] = {0.01f, 0.03f, 0.05f, 0.10f, 0.20f};

    for (int nm = 0; nm <= 3; nm++) {
        for (int ai = 0; ai < 5; ai++) {
            float amp = (nm == 0) ? 0.0f : noise_amps[ai];
            if (nm == 0 && ai > 0) continue;  // only one no-noise run

            Metrics m = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
                inp_bin,inp_cont,narma,
                DYN_LIF, nm, INJECT_MEMBRANE, amp,
                best_sr, best_is, 1.0f, best_leak, best_thresh, best_reset);
            printf("%-6s amp=%-5.2f | %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%% | %+.3f  %5.2f  | %.4f  %.2f",
                noise_names[nm], amp,
                m.xor5*100, m.xor8*100, m.par3*100, m.par4*100,
                m.narma, m.mc, m.spike_rate, m.spike_cv);
            if (nm == NOISE_DECORR && m.xor8 > tanh_base.xor8)
                printf("  !!!");
            printf("\n");
        }
    }

    // ================================================================
    // EXP 3: Injection site comparison
    // ================================================================
    printf("\n--- EXP 3: WHERE TO INJECT NOISE (LIF, decorrelated) ---\n");
    printf("%-18s | XOR-5   XOR-8   Par3  | NARMA    MC    | SpikeRate  CV\n", "InjectSite");
    printf("-------------------+----------------------+---------------+-------------\n");

    struct InjectConfig {
        const char* name;
        int site;
    };
    InjectConfig inj_cfgs[] = {
        {"NONE",           0},
        {"MEMBRANE",       INJECT_MEMBRANE},
        {"THRESHOLD",      INJECT_THRESHOLD},
        {"SYNAPTIC",       INJECT_SYNAPTIC},
        {"MEMB+THRESH",    INJECT_MEMBRANE | INJECT_THRESHOLD},
        {"ALL",            INJECT_ALL},
    };

    for (int ci = 0; ci < 6; ci++) {
        int nm = (ci == 0) ? NOISE_NONE : NOISE_DECORR;
        Metrics m = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
            inp_bin,inp_cont,narma,
            DYN_LIF, nm, inj_cfgs[ci].site, 0.05f,
            best_sr, best_is, 1.0f, best_leak, best_thresh, best_reset);
        printf("%-18s | %5.1f%%  %5.1f%%  %5.1f%% | %+.3f  %5.2f  | %.4f  %.2f\n",
            inj_cfgs[ci].name,
            m.xor5*100, m.xor8*100, m.par3*100,
            m.narma, m.mc, m.spike_rate, m.spike_cv);
    }

    // ================================================================
    // EXP 4: LIF parameter sweep WITH best noise config
    // ================================================================
    printf("\n--- EXP 4: LIF PARAMETER SWEEP (decorrelated noise) ---\n");
    printf("%-6s %-6s %-6s %-6s | XOR-5   XOR-8   Par3  | NARMA    MC    | Rate   CV\n",
        "SR", "Leak", "Thres", "Amp");
    printf("---------------------------+----------------------+---------------+-----------\n");

    float sweep_srs[] = {0.10f, 0.20f, 0.30f, 0.50f};
    float sweep_leaks[] = {0.03f, 0.05f, 0.10f};
    float sweep_thresh[] = {0.3f, 0.5f, 1.0f};
    float sweep_amps[] = {0.02f, 0.05f, 0.10f};

    float best_total = -999;
    float bsr, blk, bth, bamp;

    for (int si = 0; si < 4; si++) {
        for (int li = 0; li < 3; li++) {
            for (int ti = 0; ti < 3; ti++) {
                for (int ai = 0; ai < 3; ai++) {
                    Metrics m = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
                        inp_bin,inp_cont,narma,
                        DYN_LIF, NOISE_DECORR, INJECT_MEMBRANE|INJECT_THRESHOLD,
                        sweep_amps[ai],
                        sweep_srs[si], 1.0f, 1.0f,
                        sweep_leaks[li], sweep_thresh[ti], -0.2f);

                    float score = m.xor8 + m.par3 + m.narma*0.5f + m.mc*0.05f;
                    if (score > best_total) {
                        best_total = score;
                        bsr = sweep_srs[si]; blk = sweep_leaks[li];
                        bth = sweep_thresh[ti]; bamp = sweep_amps[ai];
                    }

                    if (m.xor8 > 0.60f || m.narma > 0.3f || m.par3 > 0.70f) {
                        printf("%-6.2f %-6.2f %-6.2f %-6.2f | %5.1f%%  %5.1f%%  %5.1f%% | %+.3f  %5.2f  | %.3f  %.2f",
                            sweep_srs[si], sweep_leaks[li], sweep_thresh[ti], sweep_amps[ai],
                            m.xor5*100, m.xor8*100, m.par3*100,
                            m.narma, m.mc, m.spike_rate, m.spike_cv);
                        if (score > best_total - 0.01f) printf("  <== BEST");
                        printf("\n");
                    }
                }
            }
        }
    }
    printf("BEST: SR=%.2f leak=%.2f thresh=%.2f amp=%.2f\n", bsr, blk, bth, bamp);

    // ================================================================
    // EXP 5: Best LIF vs best tanh — final showdown
    // ================================================================
    printf("\n--- EXP 5: FINAL SHOWDOWN (5 trials each) ---\n");

    float tx5[5],tx8[5],tp3[5],tp4[5],tn[5],tm[5];
    float lx5[5],lx8[5],lp3[5],lp4[5],ln[5],lm[5];

    for (int trial = 0; trial < 5; trial++) {
        host_rng = 31415 + trial * 7919;
        for(int t=0;t<STEPS;t++){inp_bin[t]=randf()>0.5f?1.0f:0.0f;inp_cont[t]=randf_signed()*0.5f;}
        memset(narma,0,sizeof(narma));
        for(int t=5;t<STEPS;t++){
            narma[t]=0.3f*narma[t-1]+0.05f*narma[t-1]*(narma[t-1]+narma[t-2]+narma[t-3]+narma[t-4]+narma[t-5])
                    +1.5f*inp_cont[t-1]*inp_cont[t-5]+0.1f;
            if(narma[t]>10)narma[t]=10;if(narma[t]<-10)narma[t]=-10;
        }

        // Best tanh
        Metrics mt = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
            inp_bin,inp_cont,narma,
            DYN_TANH, NOISE_NONE, 0, 0.0f,
            1.05f, 1.0f, 0.65f, 0.0f, 0.0f, 0.0f);
        tx5[trial]=mt.xor5; tx8[trial]=mt.xor8; tp3[trial]=mt.par3;
        tp4[trial]=mt.par4; tn[trial]=mt.narma; tm[trial]=mt.mc;

        // Best LIF with decorrelated noise
        Metrics ml = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
            inp_bin,inp_cont,narma,
            DYN_LIF, NOISE_DECORR, INJECT_MEMBRANE|INJECT_THRESHOLD, bamp,
            bsr, 1.0f, 1.0f, blk, bth, -0.2f);
        lx5[trial]=ml.xor5; lx8[trial]=ml.xor8; lp3[trial]=ml.par3;
        lp4[trial]=ml.par4; ln[trial]=ml.narma; lm[trial]=ml.mc;
    }

    auto avg = [](float* v, int n) { float s=0; for(int i=0;i<n;i++) s+=v[i]; return s/n; };

    printf("%-14s | XOR-5   XOR-8   Par3   Par4  | NARMA    MC\n", "Mode");
    printf("---------------+------------------------------+--------------\n");
    printf("TANH_DET       | %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%% | %+.3f  %5.2f\n",
        avg(tx5,5)*100, avg(tx8,5)*100, avg(tp3,5)*100, avg(tp4,5)*100, avg(tn,5), avg(tm,5));
    printf("LIF_DECORR     | %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%% | %+.3f  %5.2f\n",
        avg(lx5,5)*100, avg(lx8,5)*100, avg(lp3,5)*100, avg(lp4,5)*100, avg(ln,5), avg(lm,5));

    printf("\nDelta (LIF-TANH): XOR-8=%+.1fpp  Par3=%+.1fpp  NARMA=%+.3f  MC=%+.2f\n",
        (avg(lx8,5)-avg(tx8,5))*100,
        (avg(lp3,5)-avg(tp3,5))*100,
        avg(ln,5)-avg(tn,5),
        avg(lm,5)-avg(tm,5));

    printf("\nDone.\n");
    hipFree(d_state); hipFree(d_spike); hipFree((void*)d_xchg); hipFree(d_input);
    delete[] h_state; delete[] h_spike;
    return 0;
}
