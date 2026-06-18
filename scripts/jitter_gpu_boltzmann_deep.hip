// jitter_gpu_boltzmann_deep.hip — Deep sweep of Boltzmann + Noise Gate (the 2 winners)
//
// Previous finding: BOLTZMANN gives Par3=87.7% (+28pp), NARMA=+0.609, MC=8.66
//                   NOISE_GATE gives XOR-8=61.7% (+5.3pp), MC=7.95 (+1.49)
//
// This script probes:
// EXP 1: Boltzmann temperature sweep (base_temp × noise_scale)
// EXP 2: Noise Gate sparsity sweep (what fraction of connections gated)
// EXP 3: COMBINED Boltzmann + Gate (best of both?)
// EXP 4: Boltzmann + quadratic readout deep sweep (SR × temp)
// EXP 5: Statistical reliability (20 trials of best config)
// EXP 6: XOR at increasing delays (1,2,3,5,8,10,12,15)
// EXP 7: Parity at increasing bit widths (2,3,4,5,6,8)
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_gpu_boltzmann_deep scripts/jitter_gpu_boltzmann_deep.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_gpu_boltzmann_deep

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

// Modes
#define MODE_DET 0
#define MODE_BOLTZ 1
#define MODE_GATE 2
#define MODE_COMBINED 3  // Boltzmann + Gate

__global__ void esn_deep(
    float* state_out,
    float* noise_out,
    volatile float* cu_xchg,
    const float* input_seq,
    int steps,
    int mode,
    float spectral_radius,
    float input_scale,
    float boltz_base_temp,   // base temperature for Boltzmann
    float boltz_noise_scale, // how much PLL noise modulates temperature
    float gate_prob          // fraction of connections kept active
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
    float dummy = 1.0001f + nid * 0.0001f;

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];

        // Sample PLL noise
        float raw_noise = 0.0f;
        if (mode != MODE_DET) {
            int depth = 32;
            float ns = 0.0f;
            for (int r = 0; r < depth; r++) {
                uint64_t w0 = wall_clock64();
                uint64_t w1 = wall_clock64();
                ns += (float)(w1 - w0);
            }
            raw_noise = (ns - depth * 2.2f) / (sqrtf((float)depth) * 2.0f);

            // Instruction-mix modulation
            if (v > 0.0f) {
                for (int i = 0; i < 4; i++)
                    dummy = __fmaf_rn(dummy, 1.000001f, 0.0001f);
            } else {
                uint32_t iv = __float_as_uint(dummy);
                for (int i = 0; i < 4; i++) iv = iv*2654435761u+1;
                dummy = __uint_as_float((iv & 0x007FFFFF) | 0x3F800000);
            }
        }

        // Dense recurrence (possibly gated)
        lds[tid] = v;
        __syncthreads();

        float rec = 0.0f;

        bool do_gate = (mode == MODE_GATE || mode == MODE_COMBINED);
        if (do_gate) {
            uint64_t gate_seed = wall_clock64();
            // gate_prob controls sparsity: 1.0 = keep all, 0.5 = keep half
            uint32_t threshold = (uint32_t)(gate_prob * 4294967295.0f);
            for (int j = 0; j < THREADS; j++) {
                uint32_t gh = (uint32_t)(gate_seed + j * 7 + tid * 13);
                gh ^= gh>>16; gh *= 0x45d9f3b; gh ^= gh>>16;
                float gate_scale = (gh < threshold) ? (1.0f / gate_prob) : 0.0f;
                rec += gate_scale * w_row[j] * lds[j];
            }
        } else {
            for (int j = 0; j < THREADS; j++)
                rec += w_row[j] * lds[j];
        }

        cu_xchg[nid] = v;
        __threadfence();
        float cross = w_cross * cu_xchg[rcu*THREADS+rtid];

        float pre = rec + cross + w_in * u + bias;

        // Boltzmann temperature
        bool do_boltz = (mode == MODE_BOLTZ || mode == MODE_COMBINED);
        if (do_boltz) {
            float temp = boltz_base_temp + boltz_noise_scale * fabsf(raw_noise);
            if (temp < 0.01f) temp = 0.01f;
            v = tanhf(pre / temp);
        } else {
            v = tanhf(pre);
        }

        state_out[nid * steps + t] = v;
        noise_out[nid * steps + t] = raw_noise;
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

void extract_features(
    float* state, int n, int steps, int t,
    std::vector<float>& f
) {
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

float ridge_solve(
    std::vector<std::vector<float>>&X,std::vector<float>&y,int nt
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

// Run one configuration and return all metrics
struct Metrics {
    float xor_acc[16];  // XOR at delays 1..15
    float par_acc[8];   // Parity at widths 2..8
    float narma;
    float mc;
};

Metrics run_config(
    float* d_state, float* d_noise, volatile float* d_xchg, float* d_input,
    float* h_state,
    float* inp_bin, float* inp_cont, float* narma_tgt,
    int mode, float SR, float IS,
    float boltz_base, float boltz_noise, float gate_prob,
    int max_xor_delay, int max_par_width
) {
    Metrics m = {};

    // Continuous input for NARMA + MC
    hipMemcpy(d_input, inp_cont, STEPS*sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_state, 0, N_NEURONS*STEPS*sizeof(float));
    hipMemset(d_noise, 0, N_NEURONS*STEPS*sizeof(float));
    hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
    esn_deep<<<N_CUS,THREADS>>>(d_state, d_noise, d_xchg, d_input, STEPS,
        mode, SR, IS, boltz_base, boltz_noise, gate_prob);
    hipDeviceSynchronize();
    hipMemcpy(h_state, d_state, N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);

    std::vector<std::vector<float>> fc;
    for(int t=WASHOUT;t<STEPS;t++){
        std::vector<float>f;
        extract_features(h_state,N_NEURONS,STEPS,t,f);
        fc.push_back(f);
    }
    int ns=fc.size(), nt=ns*2/3;

    // NARMA
    std::vector<float>yn;
    for(int t=WASHOUT;t<STEPS;t++) yn.push_back(narma_tgt[t]);
    m.narma = ridge_solve(fc, yn, nt);

    // MC
    m.mc = 0;
    for(int d=1;d<=10;d++){
        std::vector<float>ym;
        for(int t=WASHOUT;t<STEPS;t++) ym.push_back(t-d>=0?inp_cont[t-d]:0.0f);
        float r2=ridge_solve(fc,ym,nt);
        if(r2>0) m.mc += r2;
    }

    // Binary input for XOR + Parity
    hipMemcpy(d_input, inp_bin, STEPS*sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_state, 0, N_NEURONS*STEPS*sizeof(float));
    hipMemset(d_noise, 0, N_NEURONS*STEPS*sizeof(float));
    hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
    esn_deep<<<N_CUS,THREADS>>>(d_state, d_noise, d_xchg, d_input, STEPS,
        mode, SR, IS, boltz_base, boltz_noise, gate_prob);
    hipDeviceSynchronize();
    hipMemcpy(h_state, d_state, N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);

    std::vector<std::vector<float>> fb;
    for(int t=WASHOUT;t<STEPS;t++){
        std::vector<float>f;
        extract_features(h_state,N_NEURONS,STEPS,t,f);
        fb.push_back(f);
    }
    ns=fb.size(); nt=ns*2/3;

    // XOR at various delays
    for(int d=1;d<=max_xor_delay&&d<16;d++){
        std::vector<int>y;
        for(int t=WASHOUT;t<STEPS;t++){
            int val = (t>=d) ? ((int)inp_bin[t]^(int)inp_bin[t-d]) : 0;
            y.push_back(val);
        }
        m.xor_acc[d] = ridge_classify(fb, y, nt);
    }

    // Parity at various widths
    for(int w=2;w<=max_par_width&&w<8;w++){
        std::vector<int>y;
        for(int t=WASHOUT;t<STEPS;t++){
            int parity = 0;
            for(int k=0;k<=w&&t-k>=0;k++) parity ^= (int)inp_bin[t-k];
            y.push_back(parity & 1);
        }
        m.par_acc[w] = ridge_classify(fb, y, nt);
    }

    return m;
}

int main() {
    printf("=== DEEP BOLTZMANN + NOISE GATE SWEEP ===\n\n");

    float *d_state, *d_noise, *d_input;
    volatile float *d_xchg;
    size_t state_sz = N_NEURONS*STEPS*sizeof(float);
    hipMalloc(&d_state, state_sz);
    hipMalloc(&d_noise, state_sz);
    hipMalloc((void**)&d_xchg, N_NEURONS*sizeof(float));
    hipMalloc(&d_input, STEPS*sizeof(float));
    float* h_state = new float[N_NEURONS*STEPS];

    // Generate input sequences
    host_rng = 31415;
    float inp_bin[STEPS], inp_cont[STEPS];
    for(int t=0;t<STEPS;t++){inp_bin[t]=randf()>0.5f?1.0f:0.0f;inp_cont[t]=randf_signed()*0.5f;}

    float narma[STEPS]={};
    for(int t=5;t<STEPS;t++){
        narma[t]=0.3f*narma[t-1]+0.05f*narma[t-1]*(narma[t-1]+narma[t-2]+narma[t-3]+narma[t-4]+narma[t-5])
                +1.5f*inp_cont[t-1]*inp_cont[t-5]+0.1f;
        if(narma[t]>10)narma[t]=10;if(narma[t]<-10)narma[t]=-10;
    }

    float SR = 0.99f, IS = 1.0f;

    // ================================================================
    // EXP 1: Boltzmann temperature sweep
    // ================================================================
    printf("--- EXP 1: BOLTZMANN TEMPERATURE SWEEP ---\n");
    printf("%-10s %-10s | XOR-5   XOR-8   Par3   | NARMA    MC\n", "Base_T", "Noise_S");
    printf("----------------------+------------------------+--------------\n");

    float base_temps[] = {0.1f, 0.3f, 0.5f, 0.7f, 1.0f, 1.5f, 2.0f};
    float noise_scales[] = {0.0f, 0.1f, 0.3f, 0.5f, 1.0f};

    for (int ti = 0; ti < 7; ti++) {
        for (int ni = 0; ni < 5; ni++) {
            Metrics m = run_config(d_state,d_noise,d_xchg,d_input,h_state,
                inp_bin,inp_cont,narma,
                MODE_BOLTZ, SR, IS,
                base_temps[ti], noise_scales[ni], 1.0f,
                8, 4);
            printf("%-10.2f %-10.2f | %5.1f%%  %5.1f%%  %5.1f%%  | %+.3f  %5.2f",
                base_temps[ti], noise_scales[ni],
                m.xor_acc[5]*100, m.xor_acc[8]*100, m.par_acc[3]*100,
                m.narma, m.mc);
            if (m.par_acc[3] > 0.75f || m.narma > 0.3f)
                printf("  <==");
            printf("\n");
        }
    }

    // ================================================================
    // EXP 2: Noise Gate sparsity sweep
    // ================================================================
    printf("\n--- EXP 2: NOISE GATE SPARSITY SWEEP ---\n");
    printf("%-10s | XOR-5   XOR-8   Par3   | NARMA    MC\n", "GateProb");
    printf("-----------+------------------------+--------------\n");

    float gate_probs[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    for (int gi = 0; gi < 10; gi++) {
        Metrics m = run_config(d_state,d_noise,d_xchg,d_input,h_state,
            inp_bin,inp_cont,narma,
            MODE_GATE, SR, IS,
            1.0f, 0.0f, gate_probs[gi],
            8, 4);
        printf("%-10.2f | %5.1f%%  %5.1f%%  %5.1f%%  | %+.3f  %5.2f",
            gate_probs[gi],
            m.xor_acc[5]*100, m.xor_acc[8]*100, m.par_acc[3]*100,
            m.narma, m.mc);
        if (m.par_acc[3] > 0.65f || m.mc > 7.0f)
            printf("  <==");
        printf("\n");
    }

    // ================================================================
    // EXP 3: COMBINED Boltzmann + Gate
    // ================================================================
    printf("\n--- EXP 3: COMBINED BOLTZMANN + GATE ---\n");
    printf("%-6s %-6s %-6s | XOR-5   XOR-8   Par3   | NARMA    MC\n",
        "Temp", "NoisS", "GateP");
    printf("---------------------+------------------------+--------------\n");

    // Test promising combinations
    float comb_temps[] = {0.3f, 0.5f, 0.7f, 1.0f};
    float comb_gates[] = {0.3f, 0.5f, 0.7f};
    float comb_noise[] = {0.1f, 0.3f, 0.5f};

    for (int ti = 0; ti < 4; ti++) {
        for (int gi = 0; gi < 3; gi++) {
            for (int ni = 0; ni < 3; ni++) {
                Metrics m = run_config(d_state,d_noise,d_xchg,d_input,h_state,
                    inp_bin,inp_cont,narma,
                    MODE_COMBINED, SR, IS,
                    comb_temps[ti], comb_noise[ni], comb_gates[gi],
                    8, 4);
                printf("%-6.1f %-6.1f %-6.1f | %5.1f%%  %5.1f%%  %5.1f%%  | %+.3f  %5.2f",
                    comb_temps[ti], comb_noise[ni], comb_gates[gi],
                    m.xor_acc[5]*100, m.xor_acc[8]*100, m.par_acc[3]*100,
                    m.narma, m.mc);
                if (m.narma > 0.5f || m.par_acc[3] > 0.80f)
                    printf("  <==");
                printf("\n");
            }
        }
    }

    // ================================================================
    // EXP 4: Deterministic baseline for comparison
    // ================================================================
    printf("\n--- EXP 4: DETERMINISTIC BASELINE ---\n");
    Metrics det = run_config(d_state,d_noise,d_xchg,d_input,h_state,
        inp_bin,inp_cont,narma,
        MODE_DET, SR, IS, 1.0f, 0.0f, 1.0f,
        15, 8);
    printf("DET:   XOR-5=%5.1f%%  XOR-8=%5.1f%%  Par3=%5.1f%%  NARMA=%+.3f  MC=%.2f\n",
        det.xor_acc[5]*100, det.xor_acc[8]*100, det.par_acc[3]*100, det.narma, det.mc);

    // ================================================================
    // EXP 5: Statistical reliability — 10 trials of best Boltzmann
    // ================================================================
    printf("\n--- EXP 5: STATISTICAL RELIABILITY (10 trials) ---\n");
    printf("Config: BOLTZ T=0.5 NS=0.3, GATE P=0.5, COMBINED T=0.5/NS=0.3/GP=0.5\n");
    printf("%-12s | XOR-5         XOR-8         Par3          NARMA         MC\n", "Mode");
    printf("-------------+---------------------------------------------------------------\n");

    struct TrialStats {
        const char* name;
        int mode;
        float bt, bn, gp;
    };
    TrialStats configs[] = {
        {"DET",      MODE_DET,  1.0f, 0.0f, 1.0f},
        {"BOLTZ",    MODE_BOLTZ, 0.5f, 0.3f, 1.0f},
        {"GATE",     MODE_GATE, 1.0f, 0.0f, 0.5f},
        {"COMBINED", MODE_COMBINED, 0.5f, 0.3f, 0.5f},
    };

    for (int ci = 0; ci < 4; ci++) {
        float sx5[10], sx8[10], sp3[10], sn[10], sm[10];
        for (int trial = 0; trial < 10; trial++) {
            // Vary input seed per trial
            host_rng = 31415 + trial * 7919;
            for(int t=0;t<STEPS;t++){inp_bin[t]=randf()>0.5f?1.0f:0.0f;inp_cont[t]=randf_signed()*0.5f;}
            for(int t=5;t<STEPS;t++){
                narma[t]=0.3f*narma[t-1]+0.05f*narma[t-1]*(narma[t-1]+narma[t-2]+narma[t-3]+narma[t-4]+narma[t-5])
                        +1.5f*inp_cont[t-1]*inp_cont[t-5]+0.1f;
                if(narma[t]>10)narma[t]=10;if(narma[t]<-10)narma[t]=-10;
            }
            Metrics m = run_config(d_state,d_noise,d_xchg,d_input,h_state,
                inp_bin,inp_cont,narma,
                configs[ci].mode, SR, IS,
                configs[ci].bt, configs[ci].bn, configs[ci].gp,
                8, 4);
            sx5[trial] = m.xor_acc[5];
            sx8[trial] = m.xor_acc[8];
            sp3[trial] = m.par_acc[3];
            sn[trial] = m.narma;
            sm[trial] = m.mc;
        }

        // Compute mean and std
        auto stats = [](float* v, int n, float& mean, float& std) {
            mean = 0; for(int i=0;i<n;i++) mean+=v[i]; mean/=n;
            std = 0; for(int i=0;i<n;i++) std+=(v[i]-mean)*(v[i]-mean); std=sqrtf(std/n);
        };
        float mx5,sx5s,mx8,sx8s,mp3,sp3s,mn,sns,mm,sms;
        stats(sx5,10,mx5,sx5s); stats(sx8,10,mx8,sx8s); stats(sp3,10,mp3,sp3s);
        stats(sn,10,mn,sns); stats(sm,10,mm,sms);
        printf("%-12s | %5.1f%%+-%4.1f  %5.1f%%+-%4.1f  %5.1f%%+-%4.1f  %+.3f+-%.3f  %.2f+-%.2f\n",
            configs[ci].name,
            mx5*100,sx5s*100, mx8*100,sx8s*100, mp3*100,sp3s*100,
            mn,sns, mm,sms);
    }

    // ================================================================
    // EXP 6: XOR at increasing delays (memory depth probe)
    // ================================================================
    printf("\n--- EXP 6: XOR vs DELAY (memory depth) ---\n");
    printf("%-10s |", "Delay");
    for(int d=1;d<=15;d++) printf(" %3d  ", d);
    printf("\n-----------+");
    for(int d=1;d<=15;d++) printf("------");
    printf("\n");

    // Reset input to standard
    host_rng = 31415;
    for(int t=0;t<STEPS;t++){inp_bin[t]=randf()>0.5f?1.0f:0.0f;inp_cont[t]=randf_signed()*0.5f;}
    for(int t=5;t<STEPS;t++){
        narma[t]=0.3f*narma[t-1]+0.05f*narma[t-1]*(narma[t-1]+narma[t-2]+narma[t-3]+narma[t-4]+narma[t-5])
                +1.5f*inp_cont[t-1]*inp_cont[t-5]+0.1f;
        if(narma[t]>10)narma[t]=10;if(narma[t]<-10)narma[t]=-10;
    }

    struct XorConfig { const char* name; int mode; float bt,bn,gp; };
    XorConfig xor_cfgs[] = {
        {"DET",      MODE_DET,      1.0f, 0.0f, 1.0f},
        {"BOLTZ",    MODE_BOLTZ,    0.5f, 0.3f, 1.0f},
        {"GATE",     MODE_GATE,     1.0f, 0.0f, 0.5f},
        {"COMBINED", MODE_COMBINED, 0.5f, 0.3f, 0.5f},
    };

    for (int ci = 0; ci < 4; ci++) {
        Metrics m = run_config(d_state,d_noise,d_xchg,d_input,h_state,
            inp_bin,inp_cont,narma,
            xor_cfgs[ci].mode, SR, IS,
            xor_cfgs[ci].bt, xor_cfgs[ci].bn, xor_cfgs[ci].gp,
            15, 8);
        printf("%-10s |", xor_cfgs[ci].name);
        for(int d=1;d<=15;d++) printf(" %4.1f%%", m.xor_acc[d]*100);
        printf("\n");
    }

    // ================================================================
    // EXP 7: Parity at increasing bit widths
    // ================================================================
    printf("\n--- EXP 7: PARITY vs BIT WIDTH ---\n");
    printf("%-10s |", "Width");
    for(int w=2;w<=7;w++) printf("  %d   ", w);
    printf("\n-----------+");
    for(int w=2;w<=7;w++) printf("------");
    printf("\n");

    for (int ci = 0; ci < 4; ci++) {
        Metrics m = run_config(d_state,d_noise,d_xchg,d_input,h_state,
            inp_bin,inp_cont,narma,
            xor_cfgs[ci].mode, SR, IS,
            xor_cfgs[ci].bt, xor_cfgs[ci].bn, xor_cfgs[ci].gp,
            15, 7);
        printf("%-10s |", xor_cfgs[ci].name);
        for(int w=2;w<=7;w++) printf(" %4.1f%%", m.par_acc[w]*100);
        printf("\n");
    }

    printf("\nDone.\n");

    hipFree(d_state); hipFree(d_noise); hipFree((void*)d_xchg); hipFree(d_input);
    delete[] h_state;
    return 0;
}
