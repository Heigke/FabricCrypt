// jitter_gpu_gain_vs_noise.hip — Disentangle gain change from noise
//
// Critical question: Is Boltzmann's improvement from:
//   (A) Steeper activation tanh(pre/T) with T<1  (deterministic gain change)
//   (B) Stochastic temperature fluctuations from PLL noise
//
// Test: Compare deterministic gain sweep vs noise-modulated temperature
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_gpu_gain_vs_noise scripts/jitter_gpu_gain_vs_noise.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_gpu_gain_vs_noise

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

// Mode: 0=deterministic gain, 1=noisy temperature (same mean gain)
__global__ void esn_gain_test(
    float* state_out,
    volatile float* cu_xchg,
    const float* input_seq,
    int steps,
    int use_noise,         // 0=pure deterministic, 1=noisy temp
    float base_temp,       // base temperature
    float noise_amplitude, // how much noise modulates temp
    float spectral_radius,
    float input_scale
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= THREADS) return;
    int nid = bid * THREADS + tid;

    __shared__ float lds[THREADS];

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

        float raw_noise = 0.0f;
        if (use_noise) {
            int depth = 32;
            float ns = 0.0f;
            for (int r = 0; r < depth; r++) {
                uint64_t w0 = wall_clock64();
                uint64_t w1 = wall_clock64();
                ns += (float)(w1 - w0);
            }
            raw_noise = (ns - depth * 2.2f) / (sqrtf((float)depth) * 2.0f);
            if (v > 0.0f) {
                for (int i = 0; i < 4; i++)
                    dummy = __fmaf_rn(dummy, 1.000001f, 0.0001f);
            } else {
                uint32_t iv = __float_as_uint(dummy);
                for (int i = 0; i < 4; i++) iv = iv*2654435761u+1;
                dummy = __uint_as_float((iv & 0x007FFFFF) | 0x3F800000);
            }
        }

        lds[tid] = v;
        __syncthreads();
        float rec = 0.0f;
        for (int j = 0; j < THREADS; j++)
            rec += w_row[j] * lds[j];

        cu_xchg[nid] = v;
        __threadfence();
        float cross = w_cross * cu_xchg[rcu*THREADS+rtid];

        float pre = rec + cross + w_in * u + bias;

        float temp = base_temp;
        if (use_noise) {
            temp = base_temp + noise_amplitude * fabsf(raw_noise);
            if (temp < 0.01f) temp = 0.01f;
        }

        v = tanhf(pre / temp);

        state_out[nid * steps + t] = v;
        __syncthreads();
    }
    if (dummy == -9999.0f) state_out[0] = dummy;
}

static uint32_t host_rng = 42;
float randf(){host_rng=host_rng*1103515245+12345;return(float)((host_rng>>16)&0x7FFF)/32767.0f;}
float randf_signed(){return randf()*2.0f-1.0f;}

void extract_features(float* state, int n, int steps, int t, std::vector<float>& f) {
    int stride = n/SAMPLE_N; if(stride<1)stride=1;
    f.clear();
    for(int si=0;si<SAMPLE_N&&si*stride<n;si++){
        float v=state[si*stride*steps+t];
        if(v!=v||v>1e20||v<-1e20)v=0;
        f.push_back(v); f.push_back(v*v);
    }
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

int main() {
    printf("=== GAIN vs NOISE: Disentangling the Boltzmann Effect ===\n\n");

    float *d_state, *d_input;
    volatile float *d_xchg;
    size_t state_sz = N_NEURONS*STEPS*sizeof(float);
    hipMalloc(&d_state, state_sz);
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

    float SR = 0.99f, IS = 1.0f;

    // ================================================================
    // EXP 1: Pure deterministic gain sweep (no noise at all)
    // ================================================================
    printf("--- EXP 1: DETERMINISTIC GAIN SWEEP (tanh(pre/T), T=temperature) ---\n");
    printf("%-8s | XOR-5   XOR-8   Par3   Par4   | NARMA    MC\n", "Temp");
    printf("---------+-------------------------------+--------------\n");

    float temps[] = {0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.2f, 1.5f};

    for (int ti = 0; ti < 10; ti++) {
        float T = temps[ti];

        // Continuous run
        hipMemcpy(d_input, inp_cont, STEPS*sizeof(float), hipMemcpyHostToDevice);
        hipMemset(d_state, 0, state_sz);
        hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
        esn_gain_test<<<N_CUS,THREADS>>>(d_state, d_xchg, d_input, STEPS,
            0, T, 0.0f, SR, IS);
        hipDeviceSynchronize();
        hipMemcpy(h_state, d_state, state_sz, hipMemcpyDeviceToHost);

        std::vector<std::vector<float>> fc;
        for(int t=WASHOUT;t<STEPS;t++){
            std::vector<float>f;
            extract_features(h_state,N_NEURONS,STEPS,t,f);
            fc.push_back(f);
        }
        int ns=fc.size(), nt=ns*2/3;

        std::vector<float>yn;
        for(int t=WASHOUT;t<STEPS;t++) yn.push_back(narma[t]);
        float r2_narma = ridge_solve(fc, yn, nt);

        float mc = 0;
        for(int d=1;d<=10;d++){
            std::vector<float>ym;
            for(int t=WASHOUT;t<STEPS;t++) ym.push_back(t-d>=0?inp_cont[t-d]:0.0f);
            float r2=ridge_solve(fc,ym,nt);
            if(r2>0) mc+=r2;
        }

        // Binary run
        hipMemcpy(d_input, inp_bin, STEPS*sizeof(float), hipMemcpyHostToDevice);
        hipMemset(d_state, 0, state_sz);
        hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
        esn_gain_test<<<N_CUS,THREADS>>>(d_state, d_xchg, d_input, STEPS,
            0, T, 0.0f, SR, IS);
        hipDeviceSynchronize();
        hipMemcpy(h_state, d_state, state_sz, hipMemcpyDeviceToHost);

        std::vector<std::vector<float>> fb;
        for(int t=WASHOUT;t<STEPS;t++){
            std::vector<float>f;
            extract_features(h_state,N_NEURONS,STEPS,t,f);
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

        printf("%-8.2f | %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%  | %+.3f  %5.2f",
            T,
            ridge_classify(fb,y5,nt)*100,
            ridge_classify(fb,y8,nt)*100,
            ridge_classify(fb,yp3,nt)*100,
            ridge_classify(fb,yp4,nt)*100,
            r2_narma, mc);
        if (r2_narma > 0.5f || ridge_classify(fb,yp3,nt) > 0.85f)
            printf("  <==");
        printf("\n");
    }

    // ================================================================
    // EXP 2: At the BEST deterministic temperature, does noise help?
    // ================================================================
    printf("\n--- EXP 2: NOISE ON TOP OF BEST DETERMINISTIC GAIN ---\n");
    printf("%-8s %-8s %-8s | XOR-5   XOR-8   Par3   | NARMA    MC     | delta_vs_det\n",
        "Temp", "Noise", "NoiseA");
    printf("---------------------------+------------------------+-----------------+-----------\n");

    // We'll test at T=0.5, 0.6, 0.7 with noise amplitudes
    float test_temps[] = {0.5f, 0.6f, 0.7f};
    float test_noises[] = {0.0f, 0.05f, 0.10f, 0.20f, 0.30f, 0.50f};

    for (int ti = 0; ti < 3; ti++) {
        float T = test_temps[ti];
        for (int ni = 0; ni < 6; ni++) {
            float NA = test_noises[ni];
            int use_noise = (NA > 0.001f) ? 1 : 0;

            // Average over 5 trials for reliability
            float avg_x5=0,avg_x8=0,avg_p3=0,avg_n=0,avg_m=0;
            int trials = 5;

            for (int trial = 0; trial < trials; trial++) {
                host_rng = 31415 + trial * 7919;
                for(int t=0;t<STEPS;t++){inp_bin[t]=randf()>0.5f?1.0f:0.0f;inp_cont[t]=randf_signed()*0.5f;}
                for(int t=5;t<STEPS;t++){
                    narma[t]=0.3f*narma[t-1]+0.05f*narma[t-1]*(narma[t-1]+narma[t-2]+narma[t-3]+narma[t-4]+narma[t-5])
                            +1.5f*inp_cont[t-1]*inp_cont[t-5]+0.1f;
                    if(narma[t]>10)narma[t]=10;if(narma[t]<-10)narma[t]=-10;
                }

                // Continuous
                hipMemcpy(d_input, inp_cont, STEPS*sizeof(float), hipMemcpyHostToDevice);
                hipMemset(d_state, 0, state_sz);
                hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
                esn_gain_test<<<N_CUS,THREADS>>>(d_state, d_xchg, d_input, STEPS,
                    use_noise, T, NA, SR, IS);
                hipDeviceSynchronize();
                hipMemcpy(h_state, d_state, state_sz, hipMemcpyDeviceToHost);

                std::vector<std::vector<float>> fc;
                for(int t=WASHOUT;t<STEPS;t++){
                    std::vector<float>f;
                    extract_features(h_state,N_NEURONS,STEPS,t,f);
                    fc.push_back(f);
                }
                int ns=fc.size(), nt=ns*2/3;

                std::vector<float>yn;
                for(int t=WASHOUT;t<STEPS;t++) yn.push_back(narma[t]);
                avg_n += ridge_solve(fc, yn, nt);

                float mc=0;
                for(int d=1;d<=10;d++){
                    std::vector<float>ym;
                    for(int t=WASHOUT;t<STEPS;t++) ym.push_back(t-d>=0?inp_cont[t-d]:0.0f);
                    float r2=ridge_solve(fc,ym,nt); if(r2>0)mc+=r2;
                }
                avg_m += mc;

                // Binary
                hipMemcpy(d_input, inp_bin, STEPS*sizeof(float), hipMemcpyHostToDevice);
                hipMemset(d_state, 0, state_sz);
                hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
                esn_gain_test<<<N_CUS,THREADS>>>(d_state, d_xchg, d_input, STEPS,
                    use_noise, T, NA, SR, IS);
                hipDeviceSynchronize();
                hipMemcpy(h_state, d_state, state_sz, hipMemcpyDeviceToHost);

                std::vector<std::vector<float>> fb;
                for(int t=WASHOUT;t<STEPS;t++){
                    std::vector<float>f;
                    extract_features(h_state,N_NEURONS,STEPS,t,f);
                    fb.push_back(f);
                }
                ns=fb.size(); nt=ns*2/3;

                std::vector<int>y5,y8,yp3;
                for(int t=WASHOUT;t<STEPS;t++){
                    y5.push_back(t>=5?((int)inp_bin[t]^(int)inp_bin[t-5]):0);
                    y8.push_back(t>=8?((int)inp_bin[t]^(int)inp_bin[t-8]):0);
                    int p3=0; for(int k=0;k<=3&&t-k>=0;k++) p3^=(int)inp_bin[t-k]; yp3.push_back(p3&1);
                }
                avg_x5 += ridge_classify(fb,y5,nt);
                avg_x8 += ridge_classify(fb,y8,nt);
                avg_p3 += ridge_classify(fb,yp3,nt);
            }

            avg_x5/=trials; avg_x8/=trials; avg_p3/=trials; avg_n/=trials; avg_m/=trials;

            printf("%-8.2f %-8s %-8.2f | %5.1f%%  %5.1f%%  %5.1f%%  | %+.3f  %5.2f",
                T, use_noise?"NOISE":"DET", NA,
                avg_x5*100, avg_x8*100, avg_p3*100, avg_n, avg_m);

            // Mark promising
            if (avg_p3 > 0.85f || avg_n > 0.6f)
                printf("    <==");
            printf("\n");
        }
        printf("\n");
    }

    // ================================================================
    // EXP 3: SR sweep at optimal temperature
    // ================================================================
    printf("--- EXP 3: SPECTRAL RADIUS SWEEP at T=0.7 (det) ---\n");
    printf("%-8s | XOR-5   XOR-8   Par3   | NARMA    MC\n", "SR");
    printf("---------+------------------------+--------------\n");

    host_rng = 31415;
    for(int t=0;t<STEPS;t++){inp_bin[t]=randf()>0.5f?1.0f:0.0f;inp_cont[t]=randf_signed()*0.5f;}
    for(int t=5;t<STEPS;t++){
        narma[t]=0.3f*narma[t-1]+0.05f*narma[t-1]*(narma[t-1]+narma[t-2]+narma[t-3]+narma[t-4]+narma[t-5])
                +1.5f*inp_cont[t-1]*inp_cont[t-5]+0.1f;
        if(narma[t]>10)narma[t]=10;if(narma[t]<-10)narma[t]=-10;
    }

    float srs[] = {0.80f, 0.85f, 0.90f, 0.95f, 0.97f, 0.99f, 1.00f, 1.02f, 1.05f};
    for (int si = 0; si < 9; si++) {
        float sr = srs[si];

        hipMemcpy(d_input, inp_cont, STEPS*sizeof(float), hipMemcpyHostToDevice);
        hipMemset(d_state, 0, state_sz);
        hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
        esn_gain_test<<<N_CUS,THREADS>>>(d_state, d_xchg, d_input, STEPS,
            0, 0.7f, 0.0f, sr, IS);
        hipDeviceSynchronize();
        hipMemcpy(h_state, d_state, state_sz, hipMemcpyDeviceToHost);

        std::vector<std::vector<float>> fc;
        for(int t=WASHOUT;t<STEPS;t++){
            std::vector<float>f;
            extract_features(h_state,N_NEURONS,STEPS,t,f);
            fc.push_back(f);
        }
        int ns=fc.size(), nt=ns*2/3;

        std::vector<float>yn;
        for(int t=WASHOUT;t<STEPS;t++) yn.push_back(narma[t]);
        float r2_narma = ridge_solve(fc, yn, nt);

        float mc=0;
        for(int d=1;d<=10;d++){
            std::vector<float>ym;
            for(int t=WASHOUT;t<STEPS;t++) ym.push_back(t-d>=0?inp_cont[t-d]:0.0f);
            float r2=ridge_solve(fc,ym,nt); if(r2>0)mc+=r2;
        }

        hipMemcpy(d_input, inp_bin, STEPS*sizeof(float), hipMemcpyHostToDevice);
        hipMemset(d_state, 0, state_sz);
        hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
        esn_gain_test<<<N_CUS,THREADS>>>(d_state, d_xchg, d_input, STEPS,
            0, 0.7f, 0.0f, sr, IS);
        hipDeviceSynchronize();
        hipMemcpy(h_state, d_state, state_sz, hipMemcpyDeviceToHost);

        std::vector<std::vector<float>> fb;
        for(int t=WASHOUT;t<STEPS;t++){
            std::vector<float>f;
            extract_features(h_state,N_NEURONS,STEPS,t,f);
            fb.push_back(f);
        }
        ns=fb.size(); nt=ns*2/3;

        std::vector<int>y5,y8,yp3;
        for(int t=WASHOUT;t<STEPS;t++){
            y5.push_back(t>=5?((int)inp_bin[t]^(int)inp_bin[t-5]):0);
            y8.push_back(t>=8?((int)inp_bin[t]^(int)inp_bin[t-8]):0);
            int p3=0; for(int k=0;k<=3&&t-k>=0;k++) p3^=(int)inp_bin[t-k]; yp3.push_back(p3&1);
        }

        printf("%-8.2f | %5.1f%%  %5.1f%%  %5.1f%%  | %+.3f  %5.2f\n",
            sr,
            ridge_classify(fb,y5,nt)*100,
            ridge_classify(fb,y8,nt)*100,
            ridge_classify(fb,yp3,nt)*100,
            r2_narma, mc);
    }

    printf("\nDone.\n");
    hipFree(d_state); hipFree((void*)d_xchg); hipFree(d_input);
    delete[] h_state;
    return 0;
}
