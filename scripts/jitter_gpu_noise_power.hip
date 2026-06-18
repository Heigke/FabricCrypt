// jitter_gpu_noise_power.hip — Deep probe: 6 ways noise COULD help
//
// Previous finding: additive/multiplicative/threshold noise ALL hurt.
// But maybe we used noise wrong. Probe 6 alternative mechanisms:
//
// PROBE 1: NOISE AS FEATURE — Don't inject noise into dynamics.
//          Instead, READ noise statistics and append to readout features.
//          PLL noise carries information about GPU thermal/power state.
//
// PROBE 2: STOCHASTIC RESONANCE — Sub-threshold signals boosted by
//          optimal noise level. Sweep signal amplitude near threshold.
//
// PROBE 3: NOISE-GATED CONNECTIONS — Noise gates recurrent weights on/off.
//          Creates dynamic topology that changes per timestep.
//          Like random dropout but driven by REAL physics, not PRNG.
//
// PROBE 4: NOISE AS INPUT CHANNEL — Feed raw PLL jitter as ADDITIONAL
//          input alongside the task input. The jitter sequence itself
//          has temporal structure (autocorrelation from PLL dynamics).
//
// PROBE 5: BOLTZMANN SAMPLING — Use noise for probabilistic state transitions.
//          Energy-based dynamics where noise enables exploration.
//
// PROBE 6: NOISE ACCUMULATOR — Integrate noise over time as a separate
//          slow variable. The integral of PLL jitter = random walk with
//          drift. This creates a non-stationary feature unavailable
//          to deterministic systems.
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_gpu_noise_power scripts/jitter_gpu_noise_power.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_gpu_noise_power

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

// Noise mode flags
#define MODE_DETERMINISTIC 0
#define MODE_NOISE_FEATURE 1    // noise stats as extra readout features
#define MODE_STOCH_RESONANCE 2  // optimal noise for sub-threshold signals
#define MODE_NOISE_GATE 3       // noise gates connections
#define MODE_NOISE_INPUT 4      // raw noise as extra input channel
#define MODE_BOLTZMANN 5        // energy-based sampling
#define MODE_NOISE_ACCUM 6      // integrated noise random walk

__global__ void esn_noise_probe(
    float* state_out,          // [N_NEURONS × STEPS]
    float* noise_out,          // [N_NEURONS × STEPS] — noise statistics
    volatile float* cu_xchg,
    const float* input_seq,
    int steps,
    int mode,
    float spectral_radius,
    float input_scale,
    float noise_param          // mode-specific parameter
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

    float v = 0.0f;
    float noise_integral = 0.0f;   // for MODE_NOISE_ACCUM
    float noise_ema = 0.0f;        // exponential moving average of noise
    float dummy = 1.0001f + nid * 0.0001f;

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];

        // Always sample PLL noise (for all modes except pure deterministic)
        float raw_noise = 0.0f;
        if (mode != MODE_DETERMINISTIC) {
            int depth = 32;
            float ns = 0.0f;
            for (int r = 0; r < depth; r++) {
                uint64_t w0 = wall_clock64();
                uint64_t w1 = wall_clock64();
                ns += (float)(w1 - w0);
            }
            raw_noise = (ns - depth * 2.2f) / (sqrtf((float)depth) * 2.0f);

            // Instruction-mix modulation (state-dependent)
            if (v > 0.0f) {
                for (int i = 0; i < 4; i++)
                    dummy = __fmaf_rn(dummy, 1.000001f, 0.0001f);
            } else {
                uint32_t iv = __float_as_uint(dummy);
                for (int i = 0; i < 4; i++) iv = iv*2654435761u+1;
                dummy = __uint_as_float((iv & 0x007FFFFF) | 0x3F800000);
            }

            // Update noise statistics
            noise_ema = 0.9f * noise_ema + 0.1f * raw_noise;
            noise_integral += raw_noise * 0.01f;
        }

        // Dense recurrence
        lds[tid] = v;
        __syncthreads();
        float rec = 0.0f;
        for (int j = 0; j < THREADS; j++)
            rec += w_row[j] * lds[j];

        cu_xchg[nid] = v;
        __threadfence();
        float cross = w_cross * cu_xchg[rcu*THREADS+rtid];

        float pre = rec + cross + w_in * u + bias;

        // Mode-specific noise utilization
        float noise_feat = 0.0f;  // stored as extra feature

        switch (mode) {
            case MODE_DETERMINISTIC:
                v = tanhf(pre);
                noise_feat = 0.0f;
                break;

            case MODE_NOISE_FEATURE:
                // Noise does NOT affect dynamics — pure deterministic ESN
                // But noise statistics are recorded as EXTRA readout features
                v = tanhf(pre);
                noise_feat = noise_ema;  // running noise average as feature
                break;

            case MODE_STOCH_RESONANCE: {
                // Stochastic resonance: noise helps detect sub-threshold signals
                // Scale input to be near threshold
                float weak_pre = pre * noise_param;  // noise_param < 1 weakens signal
                v = tanhf(weak_pre + 0.02f * raw_noise);
                noise_feat = raw_noise;
                break;
            }

            case MODE_NOISE_GATE: {
                // Noise gates recurrent connections
                // Each weight is multiplied by (noise > threshold ? 1 : 0)
                // This creates dynamic random topology
                lds[tid] = v;
                __syncthreads();
                float gated_rec = 0.0f;
                // Use noise to randomly mask ~50% of connections
                // Different mask each timestep (from PLL physics)
                uint64_t gate_seed = wall_clock64();
                for (int j = 0; j < THREADS; j++) {
                    // Hash gate_seed + j to get per-connection gate
                    uint32_t gh = (uint32_t)(gate_seed + j * 7 + tid * 13);
                    gh ^= gh>>16; gh *= 0x45d9f3b;
                    float gate = (gh & 1) ? noise_param * 2.0f : 0.0f;
                    gated_rec += gate * w_row[j] * lds[j];
                }
                v = tanhf(gated_rec + cross + w_in * u + bias);
                noise_feat = (float)(gate_seed & 0xFF) / 128.0f - 1.0f;
                break;
            }

            case MODE_NOISE_INPUT:
                // Raw noise as additional input channel
                // The PLL jitter sequence has temporal autocorrelation
                v = tanhf(pre + noise_param * raw_noise);
                noise_feat = raw_noise;  // also available as feature
                break;

            case MODE_BOLTZMANN: {
                // Energy-based: P(v=1) = sigmoid(pre/temperature)
                // Noise creates temperature fluctuations
                float temp = 0.5f + 0.3f * fabsf(raw_noise);
                float prob = 1.0f / (1.0f + expf(-pre / temp));
                // Soft Boltzmann (continuous, not binary)
                v = tanhf(pre / temp);
                noise_feat = temp;
                break;
            }

            case MODE_NOISE_ACCUM:
                // Noise integral as slow variable (random walk)
                // This creates a non-stationary feature unique to stochastic systems
                v = tanhf(pre);
                // Clamp integral to prevent divergence
                if (noise_integral > 2.0f) noise_integral = 2.0f;
                if (noise_integral < -2.0f) noise_integral = -2.0f;
                noise_feat = noise_integral;
                break;
        }

        state_out[nid * steps + t] = v;
        noise_out[nid * steps + t] = noise_feat;
        __syncthreads();
    }
    if (dummy == -9999.0f) state_out[0] = dummy;
}

// ============================================================================
// HOST
// ============================================================================
static uint32_t host_rng = 42;
float randf(){host_rng=host_rng*1103515245+12345;return(float)((host_rng>>16)&0x7FFF)/32767.0f;}
float randf_signed(){return randf()*2.0f-1.0f;}

// Quadratic features from state + optional noise features
void extract_features(
    float* state, float* noise, int n, int steps, int t,
    std::vector<float>& f, bool include_noise
) {
    int stride = n/SAMPLE_N; if(stride<1)stride=1;
    f.clear();
    for(int si=0;si<SAMPLE_N&&si*stride<n;si++){
        float v=state[si*stride*steps+t];
        if(v!=v||v>1e20||v<-1e20)v=0;
        f.push_back(v);
        f.push_back(v*v);
    }
    // Cross-products
    for(int si=0;si<SAMPLE_N-8&&si*stride<n;si+=8){
        int i1=si*stride,i2=(si+8)*stride;
        if(i2>=n)break;
        float v1=state[i1*steps+t],v2=state[i2*steps+t];
        if(v1!=v1||v1>1e20)v1=0;if(v2!=v2||v2>1e20)v2=0;
        f.push_back(v1*v2);
    }
    // Noise features (if enabled)
    if (include_noise) {
        for(int si=0;si<SAMPLE_N&&si*stride<n;si++){
            float nv=noise[si*stride*steps+t];
            if(nv!=nv||nv>1e20||nv<-1e20)nv=0;
            f.push_back(nv);
        }
    }
}

float ridge_solve(
    std::vector<std::vector<float>>&X,std::vector<float>&y,int nt,
    std::vector<float>* w_out = nullptr
) {
    int n=X.size();if(n<nt+10)return-1;int d=X[0].size();float reg=1.0f;
    std::vector<float>XtX(d*d,0),Xty(d,0);
    for(int i=0;i<nt;i++)for(int a=0;a<d;a++){Xty[a]+=X[i][a]*y[i];for(int b=a;b<d;b++)XtX[a*d+b]+=X[i][a]*X[i][b];}
    for(int a=0;a<d;a++){for(int b=a+1;b<d;b++)XtX[b*d+a]=XtX[a*d+b];XtX[a*d+a]+=reg;}
    std::vector<float>L(d*d,0);
    for(int i=0;i<d;i++)for(int j=0;j<=i;j++){float s=XtX[i*d+j];for(int k=0;k<j;k++)s-=L[i*d+k]*L[j*d+k];if(i==j){if(s<=0)s=1e-6f;L[i*d+j]=sqrtf(s);}else L[i*d+j]=s/L[j*d+j];}
    std::vector<float>z(d),w(d);
    for(int i=0;i<d;i++){float s=Xty[i];for(int k=0;k<i;k++)s-=L[i*d+k]*z[k];z[i]=s/L[i*d+i];}
    for(int i=d-1;i>=0;i--){float s=z[i];for(int k=i+1;k<d;k++)s-=L[k*d+i]*w[k];w[i]=s/L[i*d+i];}
    if(w_out)*w_out=w;
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

int main() {
    printf("=== DEEP NOISE POWER PROBE: 6 Alternative Mechanisms ===\n\n");

    float *d_state, *d_noise, *d_input;
    volatile float *d_xchg;
    size_t state_sz = N_NEURONS*STEPS*sizeof(float);
    hipMalloc(&d_state, state_sz);
    hipMalloc(&d_noise, state_sz);
    hipMalloc((void**)&d_xchg, N_NEURONS*sizeof(float));
    hipMalloc(&d_input, STEPS*sizeof(float));
    float* h_state = new float[N_NEURONS*STEPS];
    float* h_noise = new float[N_NEURONS*STEPS];

    host_rng = 31415;
    float inp_bin[STEPS], inp_cont[STEPS];
    for(int t=0;t<STEPS;t++){inp_bin[t]=randf()>0.5f?1.0f:0.0f;inp_cont[t]=randf_signed()*0.5f;}

    float narma[STEPS]={};
    for(int t=5;t<STEPS;t++){
        narma[t]=0.3f*narma[t-1]+0.05f*narma[t-1]*(narma[t-1]+narma[t-2]+narma[t-3]+narma[t-4]+narma[t-5])
                +1.5f*inp_cont[t-1]*inp_cont[t-5]+0.1f;
        if(narma[t]>10)narma[t]=10;if(narma[t]<-10)narma[t]=-10;
    }

    std::vector<int> xor5(STEPS,0), xor8(STEPS,0), par3(STEPS,0);
    for(int t=5;t<STEPS;t++) xor5[t]=(int)inp_bin[t]^(int)inp_bin[t-5];
    for(int t=8;t<STEPS;t++) xor8[t]=(int)inp_bin[t]^(int)inp_bin[t-8];
    for(int t=3;t<STEPS;t++) par3[t]=((int)inp_bin[t]^(int)inp_bin[t-1]^(int)inp_bin[t-2]^(int)inp_bin[t-3])&1;

    float SR = 0.99f, IS = 1.0f;

    const char* mode_names[] = {
        "DETERMINISTIC",
        "NOISE_FEATURE",
        "STOCH_RESON",
        "NOISE_GATE",
        "NOISE_INPUT",
        "BOLTZMANN",
        "NOISE_ACCUM"
    };

    // For each mode, test with and without noise features in readout
    printf("%-16s | XOR-5   XOR-8   Par3   | NARMA    MC    | vs DET\n", "Mode");
    printf("------------------+------------------------+-----------------+--------\n");

    float det_xor5=0, det_xor8=0, det_par3=0, det_narma=0, det_mc=0;

    for (int mode = 0; mode <= 6; mode++) {
        float np = 1.0f; // noise_param default
        if (mode == MODE_STOCH_RESONANCE) np = 0.3f;  // weaken signal
        if (mode == MODE_NOISE_GATE) np = 1.0f;
        if (mode == MODE_NOISE_INPUT) np = 0.01f;  // subtle noise input

        bool include_noise = (mode == MODE_NOISE_FEATURE ||
                             mode == MODE_NOISE_ACCUM ||
                             mode == MODE_NOISE_INPUT);

        // Average over 3 trials
        float avg_xor5=0,avg_xor8=0,avg_par3=0,avg_narma=0,avg_mc=0;
        int trials = 3;

        for (int trial = 0; trial < trials; trial++) {
            // Continuous input
            hipMemcpy(d_input,inp_cont,STEPS*sizeof(float),hipMemcpyHostToDevice);
            hipMemset(d_state,0,state_sz); hipMemset(d_noise,0,state_sz);
            hipMemset((void*)d_xchg,0,N_NEURONS*sizeof(float));
            esn_noise_probe<<<N_CUS,THREADS>>>(d_state,d_noise,d_xchg,d_input,STEPS,mode,SR,IS,np);
            hipDeviceSynchronize();
            hipMemcpy(h_state,d_state,state_sz,hipMemcpyDeviceToHost);
            hipMemcpy(h_noise,d_noise,state_sz,hipMemcpyDeviceToHost);

            std::vector<std::vector<float>> fc;
            for(int t=WASHOUT;t<STEPS;t++){
                std::vector<float>f;
                extract_features(h_state,h_noise,N_NEURONS,STEPS,t,f,include_noise);
                fc.push_back(f);
            }
            int ns=fc.size(),nt=ns*2/3;

            std::vector<float>yn;
            for(int t=WASHOUT;t<STEPS;t++)yn.push_back(narma[t]);
            avg_narma += ridge_solve(fc,yn,nt);

            float mc=0;
            for(int d=1;d<=10;d++){
                std::vector<float>ym;
                for(int t=WASHOUT;t<STEPS;t++)ym.push_back(t-d>=0?inp_cont[t-d]:0.0f);
                float r2=ridge_solve(fc,ym,nt);
                if(r2>0)mc+=r2;
            }
            avg_mc += mc;

            // Binary input
            hipMemcpy(d_input,inp_bin,STEPS*sizeof(float),hipMemcpyHostToDevice);
            hipMemset(d_state,0,state_sz); hipMemset(d_noise,0,state_sz);
            hipMemset((void*)d_xchg,0,N_NEURONS*sizeof(float));
            esn_noise_probe<<<N_CUS,THREADS>>>(d_state,d_noise,d_xchg,d_input,STEPS,mode,SR,IS,np);
            hipDeviceSynchronize();
            hipMemcpy(h_state,d_state,state_sz,hipMemcpyDeviceToHost);
            hipMemcpy(h_noise,d_noise,state_sz,hipMemcpyDeviceToHost);

            std::vector<std::vector<float>> fb;
            for(int t=WASHOUT;t<STEPS;t++){
                std::vector<float>f;
                extract_features(h_state,h_noise,N_NEURONS,STEPS,t,f,include_noise);
                fb.push_back(f);
            }
            ns=fb.size();nt=ns*2/3;

            std::vector<int>y5,y8,yp;
            for(int t=WASHOUT;t<STEPS;t++){y5.push_back(xor5[t]);y8.push_back(xor8[t]);yp.push_back(par3[t]);}
            avg_xor5+=ridge_classify(fb,y5,nt);
            avg_xor8+=ridge_classify(fb,y8,nt);
            avg_par3+=ridge_classify(fb,yp,nt);
        }

        avg_xor5/=trials; avg_xor8/=trials; avg_par3/=trials;
        avg_narma/=trials; avg_mc/=trials;

        if (mode == 0) {
            det_xor5=avg_xor5; det_xor8=avg_xor8; det_par3=avg_par3;
            det_narma=avg_narma; det_mc=avg_mc;
        }

        float dx5=(avg_xor5-det_xor5)*100;
        float dx8=(avg_xor8-det_xor8)*100;
        float dp=(avg_par3-det_par3)*100;
        float dn=avg_narma-det_narma;
        float dm=avg_mc-det_mc;

        printf("%-16s | %5.1f%%  %5.1f%%  %5.1f%% | %+.3f  %5.2f  |",
               mode_names[mode],
               avg_xor5*100, avg_xor8*100, avg_par3*100,
               avg_narma, avg_mc);

        // Color delta
        int wins = 0;
        if (dx5 > 1.0) wins++;
        if (dx8 > 1.0) wins++;
        if (dp > 1.0) wins++;
        if (dn > 0.01) wins++;
        if (dm > 0.1) wins++;

        if (mode > 0) {
            printf(" x5%+.1f x8%+.1f p%+.1f n%+.3f m%+.2f",
                   dx5,dx8,dp,dn,dm);
            if (wins >= 3) printf(" !!!!");
            else if (wins >= 2) printf(" **");
            else if (wins >= 1) printf(" *");
        }
        printf("\n");
    }

    // ═══ DEEP DIVE: Best noise mode with param sweep ═══
    printf("\n=== PARAM SWEEP for most promising modes ===\n");

    // Noise-as-feature: sweep nothing (it's parameter-free)
    // Noise-input: sweep noise_param (injection strength)
    // Noise-accumulator: sweep accumulation rate

    printf("\n--- NOISE_INPUT sweep (noise injection strength) ---\n");
    printf("Param   | XOR-5  XOR-8  Par3  | NARMA   MC\n");
    float ni_params[] = {0.001, 0.003, 0.01, 0.03, 0.1, 0.3};
    for (int pi = 0; pi < 6; pi++) {
        float np = ni_params[pi];
        float ax5=0,ax8=0,ap=0,an=0,am=0;
        for(int trial=0;trial<3;trial++){
            hipMemcpy(d_input,inp_cont,STEPS*sizeof(float),hipMemcpyHostToDevice);
            hipMemset(d_state,0,state_sz);hipMemset(d_noise,0,state_sz);
            hipMemset((void*)d_xchg,0,N_NEURONS*sizeof(float));
            esn_noise_probe<<<N_CUS,THREADS>>>(d_state,d_noise,d_xchg,d_input,STEPS,MODE_NOISE_INPUT,SR,IS,np);
            hipDeviceSynchronize();
            hipMemcpy(h_state,d_state,state_sz,hipMemcpyDeviceToHost);
            hipMemcpy(h_noise,d_noise,state_sz,hipMemcpyDeviceToHost);
            std::vector<std::vector<float>>fc;
            for(int t=WASHOUT;t<STEPS;t++){std::vector<float>f;extract_features(h_state,h_noise,N_NEURONS,STEPS,t,f,true);fc.push_back(f);}
            int ns=fc.size(),nt=ns*2/3;
            std::vector<float>yn;for(int t=WASHOUT;t<STEPS;t++)yn.push_back(narma[t]);
            an+=ridge_solve(fc,yn,nt);
            float mc=0;for(int d=1;d<=10;d++){std::vector<float>ym;for(int t=WASHOUT;t<STEPS;t++)ym.push_back(t-d>=0?inp_cont[t-d]:0.0f);float r2=ridge_solve(fc,ym,nt);if(r2>0)mc+=r2;}
            am+=mc;

            hipMemcpy(d_input,inp_bin,STEPS*sizeof(float),hipMemcpyHostToDevice);
            hipMemset(d_state,0,state_sz);hipMemset(d_noise,0,state_sz);
            hipMemset((void*)d_xchg,0,N_NEURONS*sizeof(float));
            esn_noise_probe<<<N_CUS,THREADS>>>(d_state,d_noise,d_xchg,d_input,STEPS,MODE_NOISE_INPUT,SR,IS,np);
            hipDeviceSynchronize();
            hipMemcpy(h_state,d_state,state_sz,hipMemcpyDeviceToHost);
            hipMemcpy(h_noise,d_noise,state_sz,hipMemcpyDeviceToHost);
            std::vector<std::vector<float>>fb;
            for(int t=WASHOUT;t<STEPS;t++){std::vector<float>f;extract_features(h_state,h_noise,N_NEURONS,STEPS,t,f,true);fb.push_back(f);}
            ns=fb.size();nt=ns*2/3;
            std::vector<int>y5,y8,yp;
            for(int t=WASHOUT;t<STEPS;t++){y5.push_back(xor5[t]);y8.push_back(xor8[t]);yp.push_back(par3[t]);}
            ax5+=ridge_classify(fb,y5,nt);ax8+=ridge_classify(fb,y8,nt);ap+=ridge_classify(fb,yp,nt);
        }
        ax5/=3;ax8/=3;ap/=3;an/=3;am/=3;
        printf("%.3f   | %5.1f%% %5.1f%% %5.1f%% | %+.3f  %5.2f  | x5%+.1f x8%+.1f\n",
               np,ax5*100,ax8*100,ap*100,an,am,
               (ax5-det_xor5)*100,(ax8-det_xor8)*100);
    }

    printf("\nDone.\n");
    hipFree(d_state);hipFree(d_noise);hipFree((void*)d_xchg);hipFree(d_input);
    delete[] h_state; delete[] h_noise;
    return 0;
}
