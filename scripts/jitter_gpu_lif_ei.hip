// jitter_gpu_lif_ei.hip — LIF with E/I balance + synaptic delays + noise
//
// WHY PREVIOUS ATTEMPTS FAILED:
// - Binary spikes carry less info than continuous tanh
// - Random-sign weights don't create the DYNAMICS that make noise useful
//
// KEY INSIGHT FROM NEUROSCIENCE:
// Noise is useful in balanced E/I networks because:
// 1. E/I balance creates bistability → noise drives transitions
// 2. Inhibition creates OSCILLATIONS → noise modulates phase
// 3. Delays create temporal diversity → noise decorrelates trajectories
//
// THIS SCRIPT tests:
// EXP 1: E/I balanced LIF (80% excitatory, 20% inhibitory) vs random signs
// EXP 2: Synaptic delays (0-4 step delays per connection)
// EXP 3: Rate coding + population coding (not just binary spikes)
// EXP 4: Combined E/I + delays + noise — the full biological recipe
// EXP 5: Does noise help the BIOLOGICAL architecture? (the real test)
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_gpu_lif_ei scripts/jitter_gpu_lif_ei.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_gpu_lif_ei

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
#define MAX_DELAY 5
#define HIST_LEN (MAX_DELAY + 1)

// Architecture modes
#define ARCH_RANDOM 0       // random sign weights (standard ESN)
#define ARCH_EI 1           // 80/20 excitatory/inhibitory balance
#define ARCH_EI_DELAY 2     // E/I + per-synapse delays
#define ARCH_EI_DALE 3      // E/I + Dale's law (neuron is ALL exc or ALL inh)

#define NOISE_NONE 0
#define NOISE_DECORR 1

#define DYN_TANH 0
#define DYN_LIF 1
#define DYN_LIF_RATE 2      // LIF with rate-coded output (sliding window)

__global__ void esn_ei_probe(
    float* state_out,       // [N_NEURONS × STEPS]
    float* spike_out,       // [N_NEURONS × STEPS]
    volatile float* cu_xchg,
    const float* input_seq,
    int steps,
    int dynamics,
    int arch_mode,
    int noise_on,           // 0 or 1
    float noise_amp,
    float spectral_radius,
    float input_scale,
    float temperature,
    float lif_alpha,
    float lif_threshold,
    float lif_reset,
    float lif_bias_frac,
    float ei_ratio,         // fraction excitatory (0.8 = 80%)
    float inh_gain          // inhibition strength multiplier (>1 = strong inhibition)
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= THREADS) return;
    int nid = bid * THREADS + tid;

    __shared__ float lds[THREADS];
    __shared__ float spike_hist[THREADS * HIST_LEN];  // circular buffer

    // Determine if THIS neuron is excitatory or inhibitory (Dale's law)
    uint32_t h0 = nid * 2654435761u;
    h0 ^= h0>>16; h0 *= 0x45d9f3b;
    bool is_excitatory = ((float)(h0 & 0xFFFF) / 65536.0f) < ei_ratio;
    float dale_sign = is_excitatory ? 1.0f : -inh_gain;

    // Dense weight row
    float w_row[THREADS];
    int delays[THREADS];  // per-synapse delay
    float scale = spectral_radius / sqrtf((float)THREADS);

    for (int j = 0; j < THREADS; j++) {
        uint32_t h = (uint32_t)(bid*7919 + tid*6271 + j*3571 + 42);
        h ^= h>>16; h *= 0x45d9f3b; h ^= h>>16; h *= 0x45d9f3b; h ^= h>>16;
        float w = scale * ((float)(h & 0xFFFF)/32768.0f - 1.0f);

        if (arch_mode == ARCH_RANDOM) {
            w_row[j] = w;
        }
        else if (arch_mode == ARCH_EI) {
            // Source neuron j: determine if excitatory
            uint32_t hj = j * 2654435761u + bid * 7919;
            hj ^= hj>>16; hj *= 0x45d9f3b;
            bool j_exc = ((float)(hj & 0xFFFF) / 65536.0f) < ei_ratio;
            // E→* positive, I→* negative (Dale's law at source)
            w_row[j] = fabsf(w) * (j_exc ? 1.0f : -inh_gain);
        }
        else {  // ARCH_EI_DELAY or ARCH_EI_DALE
            uint32_t hj = j * 2654435761u + bid * 7919;
            hj ^= hj>>16; hj *= 0x45d9f3b;
            bool j_exc = ((float)(hj & 0xFFFF) / 65536.0f) < ei_ratio;
            w_row[j] = fabsf(w) * (j_exc ? 1.0f : -inh_gain);

            // Per-synapse delay (1-4 steps)
            uint32_t hd = h * 1103515245u + 12345u;
            delays[j] = (arch_mode == ARCH_EI_DELAY) ? (1 + (hd >> 16) % MAX_DELAY) : 0;
        }
    }

    uint32_t irng = nid * 2654435761u + 99991;
    irng ^= irng>>16; irng *= 0x45d9f3b; irng ^= irng>>16;
    float w_in = input_scale * ((float)(irng & 0xFFFF)/32768.0f - 1.0f);

    // Bias near threshold
    irng *= 0x45d9f3b; irng ^= irng>>16;
    float bias_rand = (float)(irng & 0xFFFF)/65536.0f;
    float bias = lif_threshold * (lif_bias_frac - 0.1f + 0.2f * bias_rand);

    irng *= 1103515245; int rcu = ((irng>>8)&0xFF)%N_CUS;
    irng *= 1103515245; int rtid = ((irng>>8)&0xFF)%THREADS;
    irng *= 1103515245; float w_cross = 0.03f * ((float)(irng & 0xFFFF)/32768.0f - 1.0f);

    // Per-neuron threshold heterogeneity (±15%)
    irng *= 1103515245;
    float neuron_thresh = lif_threshold * (0.85f + 0.30f * ((float)(irng & 0xFFFF)/65536.0f));

    float v = 0.0f;
    float spike = 0.0f;
    int refractory = 0;
    uint32_t prng_state = nid * 2654435761u + 12345;

    // Rate coding window
    float rate_window[10] = {0};
    int rate_idx = 0;

    // Initialize spike history
    for (int d = 0; d < HIST_LEN; d++)
        spike_hist[tid * HIST_LEN + d] = 0.0f;

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];

        // ========== NOISE ==========
        float noise = 0.0f;
        if (noise_on) {
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

        // ========== RECURRENCE ==========
        // Store current spike in history
        int hist_slot = t % HIST_LEN;
        spike_hist[tid * HIST_LEN + hist_slot] = spike;
        __syncthreads();

        // For tanh: share state directly
        if (dynamics == DYN_TANH) {
            lds[tid] = v;
            __syncthreads();
        }

        float rec = 0.0f;
        if (dynamics == DYN_TANH) {
            for (int j = 0; j < THREADS; j++)
                rec += w_row[j] * lds[j];
        } else {
            // LIF: read from spike history with per-synapse delay
            for (int j = 0; j < THREADS; j++) {
                int d = (arch_mode >= ARCH_EI_DELAY) ? delays[j] : 0;
                int read_slot = ((t - d) % HIST_LEN + HIST_LEN) % HIST_LEN;
                float s_delayed = spike_hist[j * HIST_LEN + read_slot];
                rec += w_row[j] * s_delayed;
            }
        }

        cu_xchg[nid] = (dynamics == DYN_TANH) ? v : spike;
        __threadfence();
        float cross = w_cross * cu_xchg[rcu*THREADS+rtid];

        float I_syn = rec + cross + w_in * u;

        // ========== DYNAMICS ==========
        if (dynamics == DYN_TANH) {
            float pre = I_syn + bias;
            if (noise_on) pre += noise_amp * noise;
            v = tanhf(pre / temperature);
            spike = v;
        }
        else {  // DYN_LIF or DYN_LIF_RATE
            spike = 0.0f;

            if (refractory > 0) {
                refractory--;
                v *= (1.0f - lif_alpha);
            } else {
                v = v * (1.0f - lif_alpha) + I_syn + bias * lif_alpha;

                if (noise_on)
                    v += noise_amp * noise;

                float thresh = neuron_thresh;
                if (noise_on)
                    thresh += noise_amp * 0.2f * noise;  // threshold jitter

                if (v >= thresh) {
                    spike = 1.0f;
                    v = lif_reset;
                    refractory = 2;
                }
            }

            // Rate coding: sliding window spike count
            rate_window[rate_idx] = spike;
            rate_idx = (rate_idx + 1) % 10;
        }

        // Output: for LIF_RATE, use rate-coded values
        float out_val = v;
        if (dynamics == DYN_LIF_RATE) {
            float rate = 0;
            for (int k = 0; k < 10; k++) rate += rate_window[k];
            out_val = rate / 10.0f;  // firing rate in [0, 1]
        }

        state_out[nid * steps + t] = out_val;
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
            float sc = 0;
            for (int dt = 0; dt < 10 && t-dt >= 0; dt++)
                sc += spikes[si*stride*steps+(t-dt)];
            f.push_back(sc / 10.0f);
        }
    }
    // Cross-products (population coding)
    for(int si=0;si<SAMPLE_N-4&&si*stride<n;si+=4){
        int i1=si*stride,i2=(si+4)*stride;
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
    int dynamics, int arch_mode, int noise_on, float noise_amp,
    float SR, float IS, float T, float alpha, float thresh, float reset,
    float bias_frac, float ei_ratio, float inh_gain
) {
    Metrics m = {};

    hipMemcpy(d_input, inp_cont, STEPS*sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_state, 0, (size_t)N_NEURONS*STEPS*sizeof(float));
    hipMemset(d_spike, 0, (size_t)N_NEURONS*STEPS*sizeof(float));
    hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
    esn_ei_probe<<<N_CUS,THREADS>>>(d_state, d_spike, d_xchg, d_input, STEPS,
        dynamics, arch_mode, noise_on, noise_amp, SR, IS, T, alpha, thresh, reset,
        bias_frac, ei_ratio, inh_gain);
    hipDeviceSynchronize();
    hipMemcpy(h_state, d_state, (size_t)N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_spike, d_spike, (size_t)N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);

    bool use_spikes = (dynamics != DYN_TANH);
    std::vector<std::vector<float>> fc;
    for(int t=WASHOUT;t<STEPS;t++){
        std::vector<float>f;
        extract_features(h_state, h_spike, N_NEURONS, STEPS, t, f, use_spikes);
        fc.push_back(f);
    }
    int ns=fc.size(), nt=ns*2/3;

    if (dynamics != DYN_TANH) {
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
    esn_ei_probe<<<N_CUS,THREADS>>>(d_state, d_spike, d_xchg, d_input, STEPS,
        dynamics, arch_mode, noise_on, noise_amp, SR, IS, T, alpha, thresh, reset,
        bias_frac, ei_ratio, inh_gain);
    hipDeviceSynchronize();
    hipMemcpy(h_state, d_state, (size_t)N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_spike, d_spike, (size_t)N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);

    std::vector<std::vector<float>> fb;
    for(int t=WASHOUT;t<STEPS;t++){
        std::vector<float>f;
        extract_features(h_state, h_spike, N_NEURONS, STEPS, t, f, use_spikes);
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

void print_metrics(const char* name, Metrics& m, bool show_spikes=true) {
    printf("%-22s | %5.1f%% %5.1f%% %5.1f%% %5.1f%% | %+.3f %5.2f",
        name, m.xor5*100, m.xor8*100, m.par3*100, m.par4*100, m.narma, m.mc);
    if (show_spikes && m.spike_rate > 0)
        printf(" | %.3f %.2f", m.spike_rate, m.spike_cv);
    printf("\n");
}

int main() {
    printf("=== GPU LIF: E/I BALANCE + DELAYS + NOISE ===\n\n");

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

    // Best LIF params from v2
    float SR=1.00f, IS=1.0f, ALPHA=0.20f, THRESH=0.5f, RESET=0.0f, BIAS=0.95f;

    // ================================================================
    // EXP 1: Architecture comparison (no noise)
    // ================================================================
    printf("--- EXP 1: ARCHITECTURE COMPARISON (no noise) ---\n");
    printf("%-22s | XOR-5  XOR-8  Par3   Par4  | NARMA   MC   | Rate  CV\n", "Architecture");
    printf("-----------------------+-----------------------------+--------------+---------\n");

    // Tanh reference
    Metrics m_tanh = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
        inp_bin,inp_cont,narma,
        DYN_TANH, ARCH_RANDOM, 0, 0.0f,
        1.05f, 1.0f, 0.65f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 1.0f);
    print_metrics("TANH_DET (reference)", m_tanh, false);

    // LIF random weights
    Metrics m_rand = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
        inp_bin,inp_cont,narma,
        DYN_LIF, ARCH_RANDOM, 0, 0.0f,
        SR, IS, 1.0f, ALPHA, THRESH, RESET, BIAS, 0.5f, 1.0f);
    print_metrics("LIF_RANDOM", m_rand);

    // LIF E/I balanced — sweep inhibition gain
    float inh_gains[] = {1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 5.0f};
    float ei_ratios[] = {0.70f, 0.80f, 0.90f};

    float best_ei_score = -999;
    float best_ig, best_ei;
    for (int ei = 0; ei < 3; ei++) {
        for (int ig = 0; ig < 6; ig++) {
            Metrics m = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
                inp_bin,inp_cont,narma,
                DYN_LIF, ARCH_EI, 0, 0.0f,
                SR, IS, 1.0f, ALPHA, THRESH, RESET, BIAS, ei_ratios[ei], inh_gains[ig]);
            char name[64];
            snprintf(name, 64, "EI(%.0f/%.0f,ig=%.1f)", ei_ratios[ei]*100, (1-ei_ratios[ei])*100, inh_gains[ig]);
            print_metrics(name, m);

            float sc = m.xor8 + m.par3 + m.narma*0.5f + m.mc*0.05f;
            if (m.spike_rate < 0.02f) sc -= 0.5f;
            if (sc > best_ei_score) {
                best_ei_score = sc;
                best_ig = inh_gains[ig]; best_ei = ei_ratios[ei];
            }
        }
    }
    printf("Best E/I: ratio=%.2f inh_gain=%.1f\n\n", best_ei, best_ig);

    // ================================================================
    // EXP 2: Add synaptic delays to best E/I
    // ================================================================
    printf("--- EXP 2: SYNAPTIC DELAYS ---\n");
    printf("%-22s | XOR-5  XOR-8  Par3   Par4  | NARMA   MC   | Rate  CV\n", "Architecture");
    printf("-----------------------+-----------------------------+--------------+---------\n");

    Metrics m_ei = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
        inp_bin,inp_cont,narma,
        DYN_LIF, ARCH_EI, 0, 0.0f,
        SR, IS, 1.0f, ALPHA, THRESH, RESET, BIAS, best_ei, best_ig);
    print_metrics("EI_NO_DELAY", m_ei);

    Metrics m_eid = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
        inp_bin,inp_cont,narma,
        DYN_LIF, ARCH_EI_DELAY, 0, 0.0f,
        SR, IS, 1.0f, ALPHA, THRESH, RESET, BIAS, best_ei, best_ig);
    print_metrics("EI_WITH_DELAY", m_eid);

    // Rate-coded LIF
    Metrics m_rate = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
        inp_bin,inp_cont,narma,
        DYN_LIF_RATE, ARCH_EI, 0, 0.0f,
        SR, IS, 1.0f, ALPHA, THRESH, RESET, BIAS, best_ei, best_ig);
    print_metrics("EI_RATE_CODED", m_rate);

    Metrics m_rated = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
        inp_bin,inp_cont,narma,
        DYN_LIF_RATE, ARCH_EI_DELAY, 0, 0.0f,
        SR, IS, 1.0f, ALPHA, THRESH, RESET, BIAS, best_ei, best_ig);
    print_metrics("EI_RATE+DELAY", m_rated);

    // ================================================================
    // EXP 3: THE KEY TEST — Does noise help E/I balanced networks?
    // ================================================================
    printf("\n--- EXP 3: NOISE + E/I ARCHITECTURE (the real test) ---\n");
    printf("%-22s | XOR-5  XOR-8  Par3   Par4  | NARMA   MC   | Rate  CV\n", "Config");
    printf("-----------------------+-----------------------------+--------------+---------\n");

    // Determine best architecture from EXP 1-2
    int best_arch = ARCH_EI;
    int best_dyn = DYN_LIF;
    float best_arch_score = m_ei.xor8 + m_ei.par3;
    if (m_eid.xor8 + m_eid.par3 > best_arch_score) { best_arch = ARCH_EI_DELAY; best_arch_score = m_eid.xor8 + m_eid.par3; }
    if (m_rate.xor8 + m_rate.par3 > best_arch_score) { best_arch = ARCH_EI; best_dyn = DYN_LIF_RATE; best_arch_score = m_rate.xor8 + m_rate.par3; }
    if (m_rated.xor8 + m_rated.par3 > best_arch_score) { best_arch = ARCH_EI_DELAY; best_dyn = DYN_LIF_RATE; }

    // No noise baseline
    Metrics m_base = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
        inp_bin,inp_cont,narma,
        best_dyn, best_arch, 0, 0.0f,
        SR, IS, 1.0f, ALPHA, THRESH, RESET, BIAS, best_ei, best_ig);
    print_metrics("BEST_ARCH (no noise)", m_base);

    // Noise amplitude sweep
    float amps[] = {0.005f, 0.01f, 0.02f, 0.03f, 0.05f, 0.07f, 0.10f, 0.15f, 0.20f, 0.30f, 0.50f};
    float best_noise_delta = -999;
    float best_noise_amp = 0;

    for (int ai = 0; ai < 11; ai++) {
        Metrics m = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
            inp_bin,inp_cont,narma,
            best_dyn, best_arch, 1, amps[ai],
            SR, IS, 1.0f, ALPHA, THRESH, RESET, BIAS, best_ei, best_ig);
        char name[64];
        snprintf(name, 64, "NOISE amp=%.3f", amps[ai]);
        print_metrics(name, m);

        float delta = (m.xor8 - m_base.xor8) + (m.par3 - m_base.par3);
        if (delta > best_noise_delta) {
            best_noise_delta = delta;
            best_noise_amp = amps[ai];
        }
    }

    printf("\nBest noise amp: %.3f (delta vs none: %+.1fpp combined)\n", best_noise_amp, best_noise_delta*100);

    // ================================================================
    // EXP 4: Statistical validation (5 trials)
    // ================================================================
    printf("\n--- EXP 4: STATISTICAL VALIDATION (5 trials) ---\n");

    float tx8[5],tp3[5],tn5[5],tm5[5];
    float bx8[5],bp3[5],bn5[5],bm5[5];
    float nx8[5],np3[5],nn5[5],nm5[5];

    for (int trial = 0; trial < 5; trial++) {
        host_rng = 31415 + trial * 7919;
        for(int t=0;t<STEPS;t++){inp_bin[t]=randf()>0.5f?1.0f:0.0f;inp_cont[t]=randf_signed()*0.5f;}
        memset(narma,0,sizeof(narma));
        for(int t=5;t<STEPS;t++){
            narma[t]=0.3f*narma[t-1]+0.05f*narma[t-1]*(narma[t-1]+narma[t-2]+narma[t-3]+narma[t-4]+narma[t-5])
                    +1.5f*inp_cont[t-1]*inp_cont[t-5]+0.1f;
            if(narma[t]>10)narma[t]=10;if(narma[t]<-10)narma[t]=-10;
        }

        // Tanh
        Metrics mt = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
            inp_bin,inp_cont,narma,
            DYN_TANH, ARCH_RANDOM, 0, 0.0f,
            1.05f, 1.0f, 0.65f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 1.0f);
        tx8[trial]=mt.xor8; tp3[trial]=mt.par3; tn5[trial]=mt.narma; tm5[trial]=mt.mc;

        // Best E/I LIF no noise
        Metrics mb = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
            inp_bin,inp_cont,narma,
            best_dyn, best_arch, 0, 0.0f,
            SR, IS, 1.0f, ALPHA, THRESH, RESET, BIAS, best_ei, best_ig);
        bx8[trial]=mb.xor8; bp3[trial]=mb.par3; bn5[trial]=mb.narma; bm5[trial]=mb.mc;

        // Best E/I LIF + noise
        Metrics mn = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
            inp_bin,inp_cont,narma,
            best_dyn, best_arch, 1, best_noise_amp,
            SR, IS, 1.0f, ALPHA, THRESH, RESET, BIAS, best_ei, best_ig);
        nx8[trial]=mn.xor8; np3[trial]=mn.par3; nn5[trial]=mn.narma; nm5[trial]=mn.mc;
    }

    auto avg = [](float* v, int n) { float s=0; for(int i=0;i<n;i++) s+=v[i]; return s/n; };
    auto sd = [&avg](float* v, int n) { float m=avg(v,n); float s=0; for(int i=0;i<n;i++) s+=(v[i]-m)*(v[i]-m); return sqrtf(s/n); };

    printf("%-22s | XOR-8          Par3          | NARMA            MC\n", "Mode");
    printf("-----------------------+-------------------------------+---------------------\n");
    printf("TANH_DET               | %5.1f±%.1f%%     %5.1f±%.1f%%     | %+.3f±%.3f   %5.2f±%.2f\n",
        avg(tx8,5)*100, sd(tx8,5)*100, avg(tp3,5)*100, sd(tp3,5)*100,
        avg(tn5,5), sd(tn5,5), avg(tm5,5), sd(tm5,5));
    printf("EI_LIF (no noise)      | %5.1f±%.1f%%     %5.1f±%.1f%%     | %+.3f±%.3f   %5.2f±%.2f\n",
        avg(bx8,5)*100, sd(bx8,5)*100, avg(bp3,5)*100, sd(bp3,5)*100,
        avg(bn5,5), sd(bn5,5), avg(bm5,5), sd(bm5,5));
    printf("EI_LIF + NOISE         | %5.1f±%.1f%%     %5.1f±%.1f%%     | %+.3f±%.3f   %5.2f±%.2f\n",
        avg(nx8,5)*100, sd(nx8,5)*100, avg(np3,5)*100, sd(np3,5)*100,
        avg(nn5,5), sd(nn5,5), avg(nm5,5), sd(nm5,5));

    printf("\nDelta NOISE-NONE: XOR-8=%+.1fpp  Par3=%+.1fpp  NARMA=%+.3f  MC=%+.2f\n",
        (avg(nx8,5)-avg(bx8,5))*100,
        (avg(np3,5)-avg(bp3,5))*100,
        avg(nn5,5)-avg(bn5,5),
        avg(nm5,5)-avg(bm5,5));

    // ================================================================
    // EXP 5: Does E/I help TANH too? (control for architecture)
    // ================================================================
    printf("\n--- EXP 5: E/I BALANCE IN TANH-ESN (architecture control) ---\n");
    printf("%-22s | XOR-5  XOR-8  Par3   Par4  | NARMA   MC\n", "Config");
    printf("-----------------------+-----------------------------+-----------\n");

    // Tanh with E/I (same weights structure but continuous dynamics)
    for (int ig = 0; ig < 6; ig++) {
        Metrics m = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
            inp_bin,inp_cont,narma,
            DYN_TANH, ARCH_EI, 0, 0.0f,
            1.05f, 1.0f, 0.65f, 0.0f, 0.0f, 0.0f, 0.0f, 0.80f, inh_gains[ig]);
        char name[64];
        snprintf(name, 64, "TANH_EI(ig=%.1f)", inh_gains[ig]);
        print_metrics(name, m, false);
    }

    // Tanh with noise
    Metrics m_tanh_noise = run_test(d_state,d_spike,d_xchg,d_input,h_state,h_spike,
        inp_bin,inp_cont,narma,
        DYN_TANH, ARCH_EI, 1, 0.05f,
        1.05f, 1.0f, 0.65f, 0.0f, 0.0f, 0.0f, 0.0f, 0.80f, best_ig);
    print_metrics("TANH_EI+NOISE(0.05)", m_tanh_noise, false);

    printf("\nDone.\n");
    hipFree(d_state); hipFree(d_spike); hipFree((void*)d_xchg); hipFree(d_input);
    delete[] h_state; delete[] h_spike;
    return 0;
}
