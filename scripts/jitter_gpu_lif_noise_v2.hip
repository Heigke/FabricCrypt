// jitter_gpu_lif_noise_v2.hip — LIF with proper subthreshold biasing
//
// v1 FAILED because: spike_rate = 0.000 for ALL configs
// Root cause: bias ≈ 0, input scaled by leak → membrane never reaches threshold
//
// v2 FIX: Neurons biased to 80-95% of threshold (subthreshold regime).
// This is where stochastic resonance WORKS — noise pushes some neurons over threshold.
// Standard LIF: dv/dt = -v/τ + I. Discrete: v = v*(1-α) + I, NOT v*(1-α) + I*α
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_gpu_lif_noise_v2 scripts/jitter_gpu_lif_noise_v2.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_gpu_lif_noise_v2

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

#define NOISE_NONE 0
#define NOISE_SHARED 1
#define NOISE_DECORR 2
#define NOISE_PRNG 3

#define INJECT_MEMBRANE 1
#define INJECT_THRESHOLD 2
#define INJECT_SYNAPTIC 4
#define INJECT_ALL 7

#define DYN_TANH 0
#define DYN_LIF 1

__global__ void esn_lif_v2(
    float* state_out,
    float* spike_out,
    volatile float* cu_xchg,
    const float* input_seq,
    int steps,
    int dynamics,
    int noise_mode,
    int inject_site,
    float noise_amp,
    float spectral_radius,
    float input_scale,
    float temperature,       // tanh gain (DYN_TANH only)
    float lif_alpha,         // membrane decay: v *= (1-alpha). alpha = dt/tau
    float lif_threshold,     // firing threshold
    float lif_reset,         // reset potential after spike
    float lif_bias_frac      // bias as fraction of threshold (0.8 = 80%)
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= THREADS) return;
    int nid = bid * THREADS + tid;

    __shared__ float lds[THREADS];

    // Dense weight row — for LIF, recurrence operates on spikes (binary)
    // so weights should be STRONGER than tanh-ESN
    float w_row[THREADS];
    float scale = spectral_radius / sqrtf((float)THREADS);  // proper scaling
    for (int j = 0; j < THREADS; j++) {
        uint32_t h = (uint32_t)(bid*7919 + tid*6271 + j*3571 + 42);
        h ^= h>>16; h *= 0x45d9f3b; h ^= h>>16; h *= 0x45d9f3b; h ^= h>>16;
        w_row[j] = scale * ((float)(h & 0xFFFF)/32768.0f - 1.0f);
    }

    uint32_t irng = nid * 2654435761u + 99991;
    irng ^= irng>>16; irng *= 0x45d9f3b; irng ^= irng>>16;
    float w_in = input_scale * ((float)(irng & 0xFFFF)/32768.0f - 1.0f);
    irng *= 0x45d9f3b; irng ^= irng>>16;

    // CRITICAL FIX: Bias neurons near threshold (subthreshold regime)
    // Spread biases from (bias_frac-0.1)*thresh to (bias_frac+0.1)*thresh
    float bias_rand = (float)(irng & 0xFFFF)/65536.0f;  // [0, 1)
    float bias = lif_threshold * (lif_bias_frac - 0.1f + 0.2f * bias_rand);

    irng *= 1103515245; int rcu = ((irng>>8)&0xFF)%N_CUS;
    irng *= 1103515245; int rtid = ((irng>>8)&0xFF)%THREADS;
    irng *= 1103515245; float w_cross = 0.05f * ((float)(irng & 0xFFFF)/32768.0f - 1.0f);

    // Per-neuron heterogeneous threshold (±10% spread)
    irng *= 1103515245;
    float neuron_thresh = lif_threshold * (0.9f + 0.2f * ((float)(irng & 0xFFFF)/65536.0f));

    float v = 0.0f;
    float spike = 0.0f;
    int refractory = 0;
    uint32_t prng_state = nid * 2654435761u + 12345;

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];

        // ========== NOISE GENERATION ==========
        float noise = 0.0f;
        if (noise_mode != NOISE_NONE) {
            if (noise_mode == NOISE_SHARED) {
                uint64_t w0 = wall_clock64();
                uint64_t w1 = wall_clock64();
                noise = (float)(w1 - w0) - 2.2f;  // centered near 0
            }
            else if (noise_mode == NOISE_DECORR) {
                uint64_t pll_bits = 0;
                for (int r = 0; r < 8; r++) {
                    uint64_t w0 = wall_clock64();
                    uint64_t w1 = wall_clock64();
                    pll_bits = (pll_bits << 4) | ((w1 - w0) & 0xF);
                }
                uint32_t h = (uint32_t)(pll_bits ^ (uint64_t)(nid * 2654435761u));
                h ^= h >> 16; h *= 0x85ebca6b;
                h ^= h >> 13; h *= 0xc2b2ae35;
                h ^= h >> 16;
                h ^= (uint32_t)(t * 1103515245);
                h *= 0x45d9f3b; h ^= h >> 16;
                noise = ((float)(h & 0xFFFF) / 32768.0f - 1.0f);
            }
            else if (noise_mode == NOISE_PRNG) {
                prng_state = prng_state * 1103515245u + 12345u;
                noise = ((float)((prng_state >> 16) & 0xFFFF) / 32768.0f - 1.0f);
            }
        }

        // ========== RECURRENCE ==========
        lds[tid] = (dynamics == DYN_LIF) ? spike : v;
        __syncthreads();

        float rec = 0.0f;
        if ((inject_site & INJECT_SYNAPTIC) && noise_mode != NOISE_NONE) {
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

        // Total synaptic current
        float I_syn = rec + cross + w_in * u;

        // ========== DYNAMICS ==========
        if (dynamics == DYN_TANH) {
            float pre = I_syn + bias;
            if ((inject_site & INJECT_MEMBRANE) && noise_mode != NOISE_NONE)
                pre += noise_amp * noise;
            v = tanhf(pre / temperature);
            spike = v;
        }
        else {  // DYN_LIF
            spike = 0.0f;

            if (refractory > 0) {
                refractory--;
                v *= (1.0f - lif_alpha);  // leak during refractory
            } else {
                // Standard LIF: v decays + current adds directly
                // v(t+1) = v(t)*(1-α) + I_syn + bias
                // bias keeps neuron near threshold
                v = v * (1.0f - lif_alpha) + I_syn + bias * lif_alpha;

                // Membrane noise — adds directly to potential
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
                    refractory = 2;
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
            float s = spikes[si*stride*steps+t];
            f.push_back(s);
            // Spike count in last 10 steps (wider window)
            float sc = 0;
            for (int dt = 0; dt < 10 && t-dt >= 0; dt++)
                sc += spikes[si*stride*steps+(t-dt)];
            f.push_back(sc / 10.0f);
            // Spike count in last 3 steps (fast dynamics)
            float sc3 = 0;
            for (int dt = 0; dt < 3 && t-dt >= 0; dt++)
                sc3 += spikes[si*stride*steps+(t-dt)];
            f.push_back(sc3 / 3.0f);
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
    float spike_rate, spike_cv;
};

Metrics run_test(
    float* d_state, float* d_spike, volatile float* d_xchg, float* d_input,
    float* h_state, float* h_spike,
    float* inp_bin, float* inp_cont, float* narma_tgt,
    int dynamics, int noise_mode, int inject_site, float noise_amp,
    float SR, float IS, float T, float alpha, float thresh, float reset, float bias_frac
) {
    Metrics m = {};

    // Continuous input
    hipMemcpy(d_input, inp_cont, STEPS*sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_state, 0, (size_t)N_NEURONS*STEPS*sizeof(float));
    hipMemset(d_spike, 0, (size_t)N_NEURONS*STEPS*sizeof(float));
    hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
    esn_lif_v2<<<N_CUS,THREADS>>>(d_state, d_spike, d_xchg, d_input, STEPS,
        dynamics, noise_mode, inject_site, noise_amp, SR, IS, T, alpha, thresh, reset, bias_frac);
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

    // Spike stats
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

    // Binary input
    hipMemcpy(d_input, inp_bin, STEPS*sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_state, 0, (size_t)N_NEURONS*STEPS*sizeof(float));
    hipMemset(d_spike, 0, (size_t)N_NEURONS*STEPS*sizeof(float));
    hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
    esn_lif_v2<<<N_CUS,THREADS>>>(d_state, d_spike, d_xchg, d_input, STEPS,
        dynamics, noise_mode, inject_site, noise_amp, SR, IS, T, alpha, thresh, reset, bias_frac);
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
    printf("=== GPU LIF v2: SUBTHRESHOLD REGIME + DECORRELATED NOISE ===\n\n");

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
    // EXP 1: Subthreshold sweep — find the firing sweet spot
    // ================================================================
    printf("--- EXP 1: SUBTHRESHOLD REGIME SWEEP (no noise) ---\n");
    printf("Bias%%   Thresh  Alpha   SR     IS    | Rate   CV   | XOR-5  XOR-8  Par3  | NARMA   MC\n");
    printf("--------------------------------------+-------------+----------------------+-----------\n");

    float best1_score = -999;
    float b1_bf, b1_th, b1_al, b1_sr, b1_is;

    float bf_vals[] = {0.70f, 0.80f, 0.85f, 0.90f, 0.95f, 1.00f};
    float th_vals[] = {0.5f, 1.0f, 2.0f};
    float al_vals[] = {0.02f, 0.05f, 0.10f, 0.20f};
    float sr_vals[] = {0.30f, 0.50f, 0.80f, 1.00f};
    float is_vals[] = {0.3f, 0.5f, 1.0f};

    // Targeted sweep: bias_frac × threshold × alpha × SR × IS
    // Focus on configs that give 5-30% spike rate
    int n_good = 0;
    for (int bi = 0; bi < 6; bi++) {
        for (int ti = 0; ti < 3; ti++) {
            for (int ai = 0; ai < 4; ai++) {
                for (int si = 0; si < 4; si++) {
                    for (int ii = 0; ii < 3; ii++) {
                        Metrics m = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
                            inp_bin,inp_cont,narma,
                            DYN_LIF, NOISE_NONE, 0, 0.0f,
                            sr_vals[si], is_vals[ii], 1.0f,
                            al_vals[ai], th_vals[ti], 0.0f, bf_vals[bi]);

                        // Score: penalize too-low or too-high spike rates
                        float rate_penalty = 0;
                        if (m.spike_rate < 0.02f) rate_penalty = -0.5f;
                        if (m.spike_rate > 0.50f) rate_penalty = -0.3f;
                        float score = m.xor8 + m.par3 + m.narma*0.5f + m.mc*0.05f + rate_penalty;

                        if (score > best1_score) {
                            best1_score = score;
                            b1_bf=bf_vals[bi]; b1_th=th_vals[ti]; b1_al=al_vals[ai];
                            b1_sr=sr_vals[si]; b1_is=is_vals[ii];
                        }

                        // Print configs with reasonable spike rate
                        if (m.spike_rate > 0.02f && m.spike_rate < 0.50f) {
                            n_good++;
                            printf("%.2f   %.1f     %.2f    %.2f   %.1f   | %.3f  %.2f | %5.1f%%  %5.1f%%  %5.1f%% | %+.3f %5.2f",
                                bf_vals[bi], th_vals[ti], al_vals[ai], sr_vals[si], is_vals[ii],
                                m.spike_rate, m.spike_cv,
                                m.xor5*100, m.xor8*100, m.par3*100,
                                m.narma, m.mc);
                            if (m.xor8 > 0.55f || m.narma > 0.0f) printf(" <<<");
                            printf("\n");
                        }
                    }
                }
            }
        }
    }
    printf("Configs with 2-50%% spike rate: %d / %d\n", n_good, 6*3*4*4*3);
    printf("BEST: bias_frac=%.2f thresh=%.1f alpha=%.2f SR=%.2f IS=%.1f\n",
        b1_bf, b1_th, b1_al, b1_sr, b1_is);

    // ================================================================
    // EXP 2: NOISE ON vs OFF at best LIF config
    // ================================================================
    printf("\n--- EXP 2: NOISE COMPARISON at best LIF config ---\n");
    printf("%-14s | XOR-5  XOR-8  Par3   Par4  | NARMA   MC   | Rate  CV\n", "NoiseMode");
    printf("---------------+-----------------------------+--------------+---------\n");

    // Run no-noise baseline
    Metrics m_none = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
        inp_bin,inp_cont,narma,
        DYN_LIF, NOISE_NONE, 0, 0.0f,
        b1_sr, b1_is, 1.0f, b1_al, b1_th, 0.0f, b1_bf);
    printf("LIF_NONE       | %5.1f%% %5.1f%% %5.1f%% %5.1f%% | %+.3f %5.2f | %.3f %.2f\n",
        m_none.xor5*100, m_none.xor8*100, m_none.par3*100, m_none.par4*100,
        m_none.narma, m_none.mc, m_none.spike_rate, m_none.spike_cv);

    // Tanh baseline for comparison
    Metrics m_tanh = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
        inp_bin,inp_cont,narma,
        DYN_TANH, NOISE_NONE, 0, 0.0f,
        1.05f, 1.0f, 0.65f, 0.0f, 0.0f, 0.0f, 0.0f);
    printf("TANH_DET       | %5.1f%% %5.1f%% %5.1f%% %5.1f%% | %+.3f %5.2f | n/a\n",
        m_tanh.xor5*100, m_tanh.xor8*100, m_tanh.par3*100, m_tanh.par4*100,
        m_tanh.narma, m_tanh.mc);

    // Test noise types and amplitudes
    int noise_modes[] = {NOISE_SHARED, NOISE_DECORR, NOISE_PRNG};
    const char* nm_names[] = {"SHARED", "DECORR", "PRNG"};
    float test_amps[] = {0.01f, 0.03f, 0.05f, 0.10f, 0.20f, 0.50f};

    float best_noise_score = m_none.xor8 + m_none.par3 + m_none.narma*0.5f;
    int best_nm = NOISE_NONE;
    float best_amp = 0;
    int best_inj = 0;

    for (int nmi = 0; nmi < 3; nmi++) {
        for (int ai = 0; ai < 6; ai++) {
            // Test membrane injection
            Metrics m = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
                inp_bin,inp_cont,narma,
                DYN_LIF, noise_modes[nmi], INJECT_MEMBRANE, test_amps[ai],
                b1_sr, b1_is, 1.0f, b1_al, b1_th, 0.0f, b1_bf);
            printf("%-6s M a=%.2f | %5.1f%% %5.1f%% %5.1f%% %5.1f%% | %+.3f %5.2f | %.3f %.2f",
                nm_names[nmi], test_amps[ai],
                m.xor5*100, m.xor8*100, m.par3*100, m.par4*100,
                m.narma, m.mc, m.spike_rate, m.spike_cv);

            float sc = m.xor8 + m.par3 + m.narma*0.5f;
            if (sc > best_noise_score) {
                best_noise_score = sc;
                best_nm = noise_modes[nmi]; best_amp = test_amps[ai]; best_inj = INJECT_MEMBRANE;
                printf(" <<<BEST");
            }
            printf("\n");

            // Also test threshold injection (key for stochastic resonance)
            m = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
                inp_bin,inp_cont,narma,
                DYN_LIF, noise_modes[nmi], INJECT_THRESHOLD, test_amps[ai],
                b1_sr, b1_is, 1.0f, b1_al, b1_th, 0.0f, b1_bf);
            printf("%-6s T a=%.2f | %5.1f%% %5.1f%% %5.1f%% %5.1f%% | %+.3f %5.2f | %.3f %.2f",
                nm_names[nmi], test_amps[ai],
                m.xor5*100, m.xor8*100, m.par3*100, m.par4*100,
                m.narma, m.mc, m.spike_rate, m.spike_cv);

            sc = m.xor8 + m.par3 + m.narma*0.5f;
            if (sc > best_noise_score) {
                best_noise_score = sc;
                best_nm = noise_modes[nmi]; best_amp = test_amps[ai]; best_inj = INJECT_THRESHOLD;
                printf(" <<<BEST");
            }
            printf("\n");
        }
    }

    printf("Best noise: mode=%d amp=%.2f inject=%d\n", best_nm, best_amp, best_inj);

    // ================================================================
    // EXP 3: Stochastic resonance scan — does optimal noise exist?
    // ================================================================
    printf("\n--- EXP 3: STOCHASTIC RESONANCE SCAN (DECORR, membrane+threshold) ---\n");
    printf("Amplitude  | XOR-5  XOR-8  Par3  | NARMA   MC   | Rate  CV  | Delta vs NONE\n");
    printf("-----------+----------------------+--------------+-----------+--------------\n");

    float sr_amps[] = {0.001f, 0.003f, 0.005f, 0.01f, 0.02f, 0.03f, 0.05f, 0.07f,
                       0.10f, 0.15f, 0.20f, 0.30f, 0.50f, 0.70f, 1.00f};
    float none_score = m_none.xor8;

    for (int ai = 0; ai < 15; ai++) {
        Metrics m = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
            inp_bin,inp_cont,narma,
            DYN_LIF, NOISE_DECORR, INJECT_MEMBRANE|INJECT_THRESHOLD, sr_amps[ai],
            b1_sr, b1_is, 1.0f, b1_al, b1_th, 0.0f, b1_bf);
        float delta = (m.xor8 - m_none.xor8) * 100;
        printf("%-10.3f | %5.1f%% %5.1f%% %5.1f%% | %+.3f %5.2f | %.3f %.2f | %+.1fpp XOR-8\n",
            sr_amps[ai],
            m.xor5*100, m.xor8*100, m.par3*100,
            m.narma, m.mc, m.spike_rate, m.spike_cv, delta);
    }

    // ================================================================
    // EXP 4: Final showdown — 5 trials, best LIF+noise vs tanh-det
    // ================================================================
    printf("\n--- EXP 4: FINAL SHOWDOWN (5 trials) ---\n");

    float tx5[5],tx8[5],tp3[5],tp4[5],tn[5],tm[5];
    float lnx5[5],lnx8[5],lnp3[5],lnp4[5],lnn[5],lnm[5];
    float ldx5[5],ldx8[5],ldp3[5],ldp4[5],ldn[5],ldm[5];

    for (int trial = 0; trial < 5; trial++) {
        host_rng = 31415 + trial * 7919;
        for(int t=0;t<STEPS;t++){inp_bin[t]=randf()>0.5f?1.0f:0.0f;inp_cont[t]=randf_signed()*0.5f;}
        memset(narma,0,sizeof(narma));
        for(int t=5;t<STEPS;t++){
            narma[t]=0.3f*narma[t-1]+0.05f*narma[t-1]*(narma[t-1]+narma[t-2]+narma[t-3]+narma[t-4]+narma[t-5])
                    +1.5f*inp_cont[t-1]*inp_cont[t-5]+0.1f;
            if(narma[t]>10)narma[t]=10;if(narma[t]<-10)narma[t]=-10;
        }

        // Tanh deterministic
        Metrics mt = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
            inp_bin,inp_cont,narma,
            DYN_TANH, NOISE_NONE, 0, 0.0f,
            1.05f, 1.0f, 0.65f, 0.0f, 0.0f, 0.0f, 0.0f);
        tx5[trial]=mt.xor5; tx8[trial]=mt.xor8; tp3[trial]=mt.par3;
        tp4[trial]=mt.par4; tn[trial]=mt.narma; tm[trial]=mt.mc;

        // LIF no noise
        Metrics mln = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
            inp_bin,inp_cont,narma,
            DYN_LIF, NOISE_NONE, 0, 0.0f,
            b1_sr, b1_is, 1.0f, b1_al, b1_th, 0.0f, b1_bf);
        lnx5[trial]=mln.xor5; lnx8[trial]=mln.xor8; lnp3[trial]=mln.par3;
        lnp4[trial]=mln.par4; lnn[trial]=mln.narma; lnm[trial]=mln.mc;

        // LIF + best noise
        int use_nm = (best_nm != NOISE_NONE) ? best_nm : NOISE_DECORR;
        float use_amp = (best_amp > 0) ? best_amp : 0.05f;
        int use_inj = (best_inj > 0) ? best_inj : INJECT_MEMBRANE|INJECT_THRESHOLD;
        Metrics mld = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
            inp_bin,inp_cont,narma,
            DYN_LIF, use_nm, use_inj, use_amp,
            b1_sr, b1_is, 1.0f, b1_al, b1_th, 0.0f, b1_bf);
        ldx5[trial]=mld.xor5; ldx8[trial]=mld.xor8; ldp3[trial]=mld.par3;
        ldp4[trial]=mld.par4; ldn[trial]=mld.narma; ldm[trial]=mld.mc;
    }

    auto avg = [](float* v, int n) { float s=0; for(int i=0;i<n;i++) s+=v[i]; return s/n; };
    auto stddev = [&avg](float* v, int n) { float m=avg(v,n); float s=0; for(int i=0;i<n;i++) s+=(v[i]-m)*(v[i]-m); return sqrtf(s/n); };

    printf("%-14s | XOR-5       XOR-8       Par3       Par4      | NARMA          MC\n", "Mode");
    printf("---------------+--------------------------------------------------+-------------------\n");
    printf("TANH_DET       | %5.1f±%.1f%%  %5.1f±%.1f%%  %5.1f±%.1f%%  %5.1f±%.1f%% | %+.3f±%.3f  %5.2f±%.2f\n",
        avg(tx5,5)*100, stddev(tx5,5)*100, avg(tx8,5)*100, stddev(tx8,5)*100,
        avg(tp3,5)*100, stddev(tp3,5)*100, avg(tp4,5)*100, stddev(tp4,5)*100,
        avg(tn,5), stddev(tn,5), avg(tm,5), stddev(tm,5));
    printf("LIF_NONE       | %5.1f±%.1f%%  %5.1f±%.1f%%  %5.1f±%.1f%%  %5.1f±%.1f%% | %+.3f±%.3f  %5.2f±%.2f\n",
        avg(lnx5,5)*100, stddev(lnx5,5)*100, avg(lnx8,5)*100, stddev(lnx8,5)*100,
        avg(lnp3,5)*100, stddev(lnp3,5)*100, avg(lnp4,5)*100, stddev(lnp4,5)*100,
        avg(lnn,5), stddev(lnn,5), avg(lnm,5), stddev(lnm,5));
    printf("LIF_NOISE      | %5.1f±%.1f%%  %5.1f±%.1f%%  %5.1f±%.1f%%  %5.1f±%.1f%% | %+.3f±%.3f  %5.2f±%.2f\n",
        avg(ldx5,5)*100, stddev(ldx5,5)*100, avg(ldx8,5)*100, stddev(ldx8,5)*100,
        avg(ldp3,5)*100, stddev(ldp3,5)*100, avg(ldp4,5)*100, stddev(ldp4,5)*100,
        avg(ldn,5), stddev(ldn,5), avg(ldm,5), stddev(ldm,5));

    printf("\nDelta NOISE-NONE: XOR-8=%+.1fpp  Par3=%+.1fpp  NARMA=%+.3f  MC=%+.2f\n",
        (avg(ldx8,5)-avg(lnx8,5))*100,
        (avg(ldp3,5)-avg(lnp3,5))*100,
        avg(ldn,5)-avg(lnn,5),
        avg(ldm,5)-avg(lnm,5));

    printf("\nDone.\n");
    hipFree(d_state); hipFree(d_spike); hipFree((void*)d_xchg); hipFree(d_input);
    delete[] h_state; delete[] h_spike;
    return 0;
}
