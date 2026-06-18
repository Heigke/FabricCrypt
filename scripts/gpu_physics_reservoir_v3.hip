/*  gpu_physics_reservoir_v3.hip — z2262v3: Ablation + Fixed Benchmarks
 *
 *  v2 showed XOR τ=1 = 97.0% (FULL) vs 50.3% (CONTROL) — BREAKTHROUGH.
 *  But classification was broken (sequential classes) and MC=0 (noise too high?).
 *
 *  v3 fixes:
 *    1. Classification: repeating waveform pattern, not sequential blocks
 *    2. Ablation: test which mechanism drives XOR (branch? TLB? noise?)
 *    3. Lower noise for MC test
 *
 *  Compile: hipcc --offload-arch=gfx1100 -O1 -o scripts/gpu_physics_reservoir_v3 scripts/gpu_physics_reservoir_v3.hip
 *  Run:     HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/gpu_physics_reservoir_v3
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define N_CUS 12
#define THREADS 64
#define N_NEURONS (N_CUS * THREADS)
#define STEPS 600
#define WASHOUT 100
#define SAMPLE_N 192
#define TLB_BUF_SIZE (512L * 1024 * 1024)

// Flags to enable/disable mechanisms
#define FLAG_PLL     0x01
#define FLAG_TLB     0x02
#define FLAG_BRANCH  0x04
#define FLAG_L1      0x08
#define FLAG_COAL    0x10
#define FLAG_ALL     0x1F

// ═══════════════════════════════════════════════════════════════════════
// CONFIGURABLE RESERVOIR — enable/disable individual mechanisms
// ═══════════════════════════════════════════════════════════════════════
__global__ void physics_reservoir(
    float* state_out,
    volatile float* cu_xchg,
    const float* input_seq,
    char* tlb_buf,
    long tlb_size,
    int steps,
    int flags  // bitmask of mechanisms to enable
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= THREADS) return;
    int nid = bid * THREADS + tid;

    __shared__ float lds[THREADS];
    __shared__ int shared_class[1];

    float v = 0.0f, h = 0.0f, slow = 0.0f;

    uint32_t rng = nid * 2654435761u + 12345;
    rng *= 1103515245; float leak = 0.05f + 0.15f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float ng = 0.10f + 0.30f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float thr = 0.4f + 0.5f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float input_w = 0.05f + 0.20f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; int ln1 = (tid+1+((rng>>8)&7))%THREADS;
    rng *= 1103515245; int ln2 = (tid+THREADS-1-((rng>>8)&7))%THREADS;
    rng *= 1103515245; float wl = 0.02f + 0.06f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; int rcu = ((rng>>8)&0xFF)%N_CUS;
    rng *= 1103515245; int rtid = ((rng>>8)&0xFF)%THREADS;
    rng *= 1103515245; float wr = 0.01f + 0.03f * ((rng>>16)&0xFF)/255.0f;

    float dummy = 1.0001f + nid * 0.0001f;
    long region_off = tlb_size / 2;

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];

        // ═══ PLL NOISE ═══
        float noise = 0.0f;
        if (flags & FLAG_PLL) {
            int depth = 8 + (int)(24.0f * fabsf(u));
            float noise_sum = 0.0f;
            for (int r = 0; r < depth; r++) {
                uint64_t w0 = wall_clock64(); uint64_t w1 = wall_clock64();
                noise_sum += (float)(w1 - w0);
            }
            noise = (noise_sum - depth * 2.2f) / (sqrtf((float)depth) * 2.0f);
        }

        // ═══ TLB ═══
        float tlb_signal = 0.0f;
        if (flags & FLAG_TLB) {
            int tlb_class = (u > 0.0f) ? 1 : 0;
            if (tid == 0) shared_class[0] = tlb_class;
            __syncthreads();
            int n_touch = 8 + (int)(8.0f * fabsf(u));
            long base = (shared_class[0] == 1) ? region_off : 0;
            float tlb_acc = 0;
            for (int i = 0; i < n_touch; i++) {
                long offset = base + ((long)tid * 4096 + (long)i * 65536) % (tlb_size / 2);
                tlb_acc += (float)tlb_buf[offset];
            }
            unsigned long long t0 = wall_clock64();
            for (int i = 0; i < 8; i++) {
                long offset = region_off + ((long)tid * 4096 + (long)i * 65536) % (tlb_size / 2);
                tlb_acc += (float)tlb_buf[offset];
            }
            unsigned long long t1 = wall_clock64();
            tlb_signal = ((float)(t1 - t0) - 300.0f) / 300.0f;
            if (tlb_acc == -999.0f) state_out[0] = tlb_acc;
        }

        // ═══ BRANCH DIVERGENCE ═══
        float branch_val = 0.0f;
        if (flags & FLAG_BRANCH) {
            float bthr = 0.5f + 0.3f * ((nid & 0xF) / 15.0f);
            if (v > bthr) {
                for (int i = 0; i < 20; i++)
                    dummy = __fmaf_rn(dummy, 1.000001f, 0.0001f);
                branch_val = 1.0f;
            } else {
                uint32_t iv = __float_as_uint(dummy);
                iv = iv * 2654435761u + 1;
                dummy = __uint_as_float((iv & 0x007FFFFF) | 0x3F800000);
                branch_val = -1.0f;
            }
        }

        // ═══ L1 CACHE ═══
        float l1_signal = 0.0f;
        if (flags & FLAG_L1) {
            float l1_val = 1.0f + nid * 0.001f;
            unsigned long long l1_t0 = __clock64();
            int l1_iters = 5 + (int)(10.0f * fabsf(u));
            for (int i = 0; i < l1_iters; i++)
                lds[(tid + i * ((int)(u * 17.0f) & 0xF)) % THREADS] += l1_val * 0.001f;
            __syncthreads();
            unsigned long long l1_t1 = __clock64();
            l1_signal = ((float)(l1_t1 - l1_t0) - 30.0f) / 20.0f;
        }

        // ═══ COALESCING ═══
        float coal_signal = 0.0f;
        if (flags & FLAG_COAL) {
            int scatter = (int)(fabsf(u) * 32.0f);
            unsigned long long ct0 = wall_clock64();
            float cv = 0;
            for (int i = 0; i < 8; i++) {
                int idx = (nid + i * (1 + scatter)) % N_NEURONS;
                cv += cu_xchg[idx];
            }
            unsigned long long ct1 = wall_clock64();
            coal_signal = ((float)(ct1 - ct0) - 50.0f) / 50.0f;
            if (cv == -999.0f) state_out[0] = cv;
        }

        // Cross-PLL
        uint64_t sc = __clock64(); uint64_t wc = wall_clock64();
        float pll = (float)((sc - wc) & 0x1FF) / 256.0f - 1.0f;

        // Recurrence
        lds[tid] = v; __syncthreads();
        float lrec = wl * (lds[ln1] + lds[ln2]);
        cu_xchg[nid] = v; __threadfence();
        float rrec = wr * cu_xchg[rcu * THREADS + rtid];

        // State update
        v = tanhf(
            (1.0f - leak) * v
          + input_w * u
          + lrec + rrec
          + ng * noise * 0.02f
          + 0.01f * tlb_signal
          + 0.02f * branch_val
          + 0.005f * l1_signal
          + 0.005f * coal_signal
          + 0.005f * pll
        );

        if (v > thr) { v -= thr; }
        h = 0.93f * h + 0.07f * v;
        slow = 0.99f * slow + 0.01f * v;
        state_out[nid * steps + t] = v + 0.3f*h + 0.1f*slow;
        __syncthreads();
    }
    if (dummy == -9999.0f) state_out[0] = dummy;
}

// ═══════════════════════════════════════════════════════════════════════
// HOST
// ═══════════════════════════════════════════════════════════════════════
static uint32_t host_rng = 42;
float randf() { host_rng = host_rng * 1103515245 + 12345; return (float)((host_rng >> 16) & 0x7FFF) / 32767.0f; }
float randf_signed() { return randf() * 2.0f - 1.0f; }

void extract_state_matrix(float* state, int n_neurons, int steps,
                          int washout, int sample_n,
                          std::vector<std::vector<float>>& X) {
    int stride = n_neurons / sample_n;
    if (stride < 1) stride = 1;
    for (int t = washout; t < steps; t++) {
        std::vector<float> row;
        for (int si = 0; si < sample_n && si*stride < n_neurons; si++) {
            int i = si * stride;
            float v = state[i * steps + t];
            if (v != v || v > 1e20 || v < -1e20) v = 0;
            row.push_back(v);
        }
        X.push_back(row);
    }
}

struct RegResult { float nrmse; float r2; };
RegResult ridge_regress(std::vector<std::vector<float>>& X_train,
                        std::vector<float>& y_train,
                        std::vector<std::vector<float>>& X_test,
                        std::vector<float>& y_test, float lambda) {
    int n = X_train.size(), d = X_train[0].size();
    std::vector<double> XtX(d*d, 0);
    for (int i=0;i<n;i++) for (int j=0;j<d;j++) for (int k=0;k<=j;k++)
        XtX[j*d+k]+=X_train[i][j]*X_train[i][k];
    for (int j=0;j<d;j++){for(int k=0;k<j;k++) XtX[k*d+j]=XtX[j*d+k]; XtX[j*d+j]+=lambda;}
    std::vector<double> Xty(d,0);
    for(int i=0;i<n;i++) for(int j=0;j<d;j++) Xty[j]+=X_train[i][j]*y_train[i];
    std::vector<double> L(d*d,0);
    for(int i=0;i<d;i++) for(int j=0;j<=i;j++){
        double s=XtX[i*d+j]; for(int k=0;k<j;k++) s-=L[i*d+k]*L[j*d+k];
        L[i*d+j]=(i==j)?sqrt(s>0?s:1e-10):s/(L[j*d+j]>0?L[j*d+j]:1e-10);
    }
    std::vector<double> w(d),z(d);
    for(int i=0;i<d;i++){double s=Xty[i]; for(int k=0;k<i;k++) s-=L[i*d+k]*z[k]; z[i]=s/(L[i*d+i]>0?L[i*d+i]:1e-10);}
    for(int i=d-1;i>=0;i--){double s=z[i]; for(int k=i+1;k<d;k++) s-=L[k*d+i]*w[k]; w[i]=s/(L[i*d+i]>0?L[i*d+i]:1e-10);}
    int nt=X_test.size(); float ym=0,sr=0,st=0;
    for(int i=0;i<nt;i++) ym+=y_test[i]; ym/=nt;
    for(int i=0;i<nt;i++){float p=0; for(int j=0;j<d;j++) p+=X_test[i][j]*w[j];
        sr+=(y_test[i]-p)*(y_test[i]-p); st+=(y_test[i]-ym)*(y_test[i]-ym);}
    RegResult res; res.nrmse=sqrtf(sr/nt)/(sqrtf(st/nt)>1e-8?sqrtf(st/nt):1e-8);
    res.r2=1.0f-sr/(st>1e-8?st:1e-8); return res;
}

float ridge_classify_binary(std::vector<std::vector<float>>& X_train,
    std::vector<int>& y_train, std::vector<std::vector<float>>& X_test,
    std::vector<int>& y_test, float lambda) {
    std::vector<float> yt(y_train.size()),ytest(y_test.size());
    for(size_t i=0;i<y_train.size();i++) yt[i]=y_train[i]?1.0f:-1.0f;
    for(size_t i=0;i<y_test.size();i++) ytest[i]=y_test[i]?1.0f:-1.0f;
    int n_tr=X_train.size(),d=X_train[0].size();
    std::vector<double> XtX(d*d,0);
    for(int i=0;i<n_tr;i++) for(int j=0;j<d;j++) for(int k=0;k<=j;k++)
        XtX[j*d+k]+=X_train[i][j]*X_train[i][k];
    for(int j=0;j<d;j++){for(int k=0;k<j;k++) XtX[k*d+j]=XtX[j*d+k]; XtX[j*d+j]+=lambda;}
    std::vector<double> Xty(d,0);
    for(int i=0;i<n_tr;i++) for(int j=0;j<d;j++) Xty[j]+=X_train[i][j]*yt[i];
    std::vector<double> L(d*d,0);
    for(int i=0;i<d;i++) for(int j=0;j<=i;j++){
        double s=XtX[i*d+j]; for(int k=0;k<j;k++) s-=L[i*d+k]*L[j*d+k];
        L[i*d+j]=(i==j)?sqrt(s>0?s:1e-10):s/(L[j*d+j]>0?L[j*d+j]:1e-10);
    }
    std::vector<double> w(d),z(d);
    for(int i=0;i<d;i++){double s=Xty[i]; for(int k=0;k<i;k++) s-=L[i*d+k]*z[k]; z[i]=s/(L[i*d+i]>0?L[i*d+i]:1e-10);}
    for(int i=d-1;i>=0;i--){double s=z[i]; for(int k=i+1;k<d;k++) s-=L[k*d+i]*w[k]; w[i]=s/(L[i*d+i]>0?L[i*d+i]:1e-10);}
    int n=X_test.size(),correct=0;
    for(int i=0;i<n;i++){float p=0; for(int j=0;j<d;j++) p+=X_test[i][j]*w[j];
        if((p>0?1:0)==y_test[i]) correct++;}
    return (float)correct/n;
}

float ridge_classify_multi(std::vector<std::vector<float>>& X_train,
    std::vector<int>& y_train, std::vector<std::vector<float>>& X_test,
    std::vector<int>& y_test, int n_classes, float lambda) {
    int d=X_train[0].size(),nt=X_test.size();
    std::vector<std::vector<double>> all_w(n_classes,std::vector<double>(d,0));
    for(int c=0;c<n_classes;c++){
        std::vector<float> yt(y_train.size());
        for(size_t i=0;i<y_train.size();i++) yt[i]=(y_train[i]==c)?1.0f:-1.0f;
        int n=X_train.size();
        std::vector<double> XtX(d*d,0);
        for(int i=0;i<n;i++) for(int j=0;j<d;j++) for(int k=0;k<=j;k++)
            XtX[j*d+k]+=X_train[i][j]*X_train[i][k];
        for(int j=0;j<d;j++){for(int k=0;k<j;k++) XtX[k*d+j]=XtX[j*d+k]; XtX[j*d+j]+=lambda;}
        std::vector<double> Xty(d,0);
        for(int i=0;i<n;i++) for(int j=0;j<d;j++) Xty[j]+=X_train[i][j]*yt[i];
        std::vector<double> L(d*d,0);
        for(int i=0;i<d;i++) for(int j=0;j<=i;j++){
            double s=XtX[i*d+j]; for(int k=0;k<j;k++) s-=L[i*d+k]*L[j*d+k];
            L[i*d+j]=(i==j)?sqrt(s>0?s:1e-10):s/(L[j*d+j]>0?L[j*d+j]:1e-10);
        }
        std::vector<double> z(d);
        for(int i=0;i<d;i++){double s=Xty[i]; for(int k=0;k<i;k++) s-=L[i*d+k]*z[k]; z[i]=s/(L[i*d+i]>0?L[i*d+i]:1e-10);}
        for(int i=d-1;i>=0;i--){double s=z[i]; for(int k=i+1;k<d;k++) s-=L[k*d+i]*all_w[c][k]; all_w[c][i]=s/(L[i*d+i]>0?L[i*d+i]:1e-10);}
    }
    int correct=0;
    for(int i=0;i<nt;i++){float best=-1e20;int pred=0;
        for(int c=0;c<n_classes;c++){float p=0; for(int j=0;j<d;j++) p+=X_test[i][j]*all_w[c][j];
            if(p>best){best=p;pred=c;}}
        if(pred==y_test[i]) correct++;}
    return (float)correct/nt;
}

void run_reservoir(int flags, float* d_state, volatile float* d_xchg,
                   float* d_input, char* d_tlb, float* h_input, float* h_state,
                   int steps, std::vector<std::vector<float>>& X) {
    HIP_CHECK(hipMemcpy(d_input, h_input, steps*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_state, 0, N_NEURONS*steps*sizeof(float)));
    HIP_CHECK(hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float)));
    physics_reservoir<<<N_CUS, THREADS>>>(d_state, d_xchg, d_input, d_tlb, TLB_BUF_SIZE, steps, flags);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_state, d_state, N_NEURONS*steps*sizeof(float), hipMemcpyDeviceToHost));
    extract_state_matrix(h_state, N_NEURONS, steps, WASHOUT, SAMPLE_N, X);
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  z2262v3: GPU-Physics Reservoir — Ablation + Fixed Tests   ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("GPU: %s, CUs: %d\n", prop.name, prop.multiProcessorCount);
    printf("Neurons: %d, Steps: %d, Washout: %d, Features: %d\n\n", N_NEURONS, STEPS, WASHOUT, SAMPLE_N);

    float *d_state, *d_input; volatile float *d_xchg; char *d_tlb;
    HIP_CHECK(hipMalloc(&d_state, N_NEURONS*STEPS*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_input, STEPS*sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_xchg, N_NEURONS*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_tlb, TLB_BUF_SIZE));
    HIP_CHECK(hipMemset(d_tlb, 0, TLB_BUF_SIZE));

    float* h_state = (float*)malloc(N_NEURONS*STEPS*sizeof(float));
    float* h_input = (float*)malloc(STEPS*sizeof(float));

    float lambda = 1.0f;
    int pass = 0, fail = 0;

    // ═══════════════════════════════════════════════════════════════
    // EXP 1: XOR ABLATION — Which mechanism provides nonlinearity?
    // ═══════════════════════════════════════════════════════════════
    printf("═══ EXP 1: XOR τ=1 Ablation ═══\n");
    printf("Which hardware mechanism enables XOR? (chance=50%%)\n\n");
    {
        for (int t = 0; t < STEPS; t++) h_input[t] = (randf() > 0.5f) ? 1.0f : -1.0f;

        struct { const char* name; int flags; } ablations[] = {
            {"ALL",          FLAG_ALL},
            {"PLL_ONLY",     FLAG_PLL},
            {"TLB_ONLY",     FLAG_TLB},
            {"BRANCH_ONLY",  FLAG_BRANCH},
            {"L1_ONLY",      FLAG_L1},
            {"COAL_ONLY",    FLAG_COAL},
            {"ALL-no_BRANCH",FLAG_PLL|FLAG_TLB|FLAG_L1|FLAG_COAL},
            {"ALL-no_TLB",   FLAG_PLL|FLAG_BRANCH|FLAG_L1|FLAG_COAL},
            {"ALL-no_PLL",   FLAG_TLB|FLAG_BRANCH|FLAG_L1|FLAG_COAL},
            {"NONE(control)",0},
        };

        printf("%-18s %8s %8s %8s %8s\n", "Condition", "XOR_1", "XOR_2", "XOR_3", "XOR_5");

        for (int ai = 0; ai < 10; ai++) {
            host_rng = 42; // reset for same input sequence
            for (int t = 0; t < STEPS; t++) h_input[t] = (randf() > 0.5f) ? 1.0f : -1.0f;

            std::vector<std::vector<float>> X;
            run_reservoir(ablations[ai].flags, d_state, d_xchg, d_input, d_tlb,
                         h_input, h_state, STEPS, X);

            printf("%-18s", ablations[ai].name);
            int taus[] = {1, 2, 3, 5};
            for (int ti = 0; ti < 4; ti++) {
                int tau = taus[ti];
                int n_eff = X.size();
                std::vector<int> y_xor;
                for (int i = 0; i < n_eff; i++) {
                    int t = i + WASHOUT;
                    int target = 0;
                    if (t >= tau) {
                        float a = h_input[t], b = h_input[t - tau];
                        target = (a > 0) != (b > 0) ? 1 : 0;
                    }
                    y_xor.push_back(target);
                }
                int split = n_eff * 2 / 3;
                std::vector<std::vector<float>> X_tr(X.begin(), X.begin()+split);
                std::vector<std::vector<float>> X_te(X.begin()+split, X.end());
                std::vector<int> y_tr(y_xor.begin(), y_xor.begin()+split);
                std::vector<int> y_te(y_xor.begin()+split, y_xor.end());
                float acc = ridge_classify_binary(X_tr, y_tr, X_te, y_te, lambda);
                printf(" %6.1f%%", acc*100);
            }
            printf("\n");
        }
        printf("\n");
    }

    // ═══════════════════════════════════════════════════════════════
    // EXP 2: Waveform classification (FIXED — repeating pattern)
    // ═══════════════════════════════════════════════════════════════
    printf("═══ EXP 2: 4-Class Waveform (repeating pattern) ═══\n\n");
    {
        // Each 20-step window is one of 4 waveforms, cycling repeatedly
        int period = 20;
        std::vector<int> labels(STEPS);
        for (int t = 0; t < STEPS; t++) {
            int cls = (t / period) % 4;
            labels[t] = cls;
            float phase = (float)(t % period) / period;
            switch(cls) {
                case 0: h_input[t] = sinf(2*M_PI*phase); break;
                case 1: h_input[t] = phase < 0.5f ? 1.0f : -1.0f; break;
                case 2: h_input[t] = 4.0f*fabsf(phase-0.5f)-1.0f; break;
                case 3: h_input[t] = 2.0f*phase - 1.0f; break;
            }
        }

        struct { const char* name; int flags; } conds[] = {
            {"ALL", FLAG_ALL}, {"PLL_ONLY", FLAG_PLL},
            {"BRANCH_ONLY", FLAG_BRANCH}, {"NONE", 0}
        };

        printf("%-18s %8s\n", "Condition", "Acc");
        float accs[4];
        for (int ci = 0; ci < 4; ci++) {
            std::vector<std::vector<float>> X;
            run_reservoir(conds[ci].flags, d_state, d_xchg, d_input, d_tlb,
                         h_input, h_state, STEPS, X);
            int n_eff = X.size();
            int split = n_eff * 2 / 3;
            std::vector<std::vector<float>> X_tr(X.begin(), X.begin()+split);
            std::vector<std::vector<float>> X_te(X.begin()+split, X.end());
            std::vector<int> y_tr(labels.begin()+WASHOUT, labels.begin()+WASHOUT+split);
            std::vector<int> y_te(labels.begin()+WASHOUT+split, labels.begin()+WASHOUT+n_eff);
            accs[ci] = ridge_classify_multi(X_tr, y_tr, X_te, y_te, 4, lambda);
            printf("%-18s %7.1f%%\n", conds[ci].name, accs[ci]*100);
        }
        bool t825 = accs[0] > 0.50f;
        bool t826 = accs[0] > accs[3];
        printf("\nT825 ALL > 50%%:     %s (%.1f%%)\n", t825?"PASS":"FAIL", accs[0]*100);
        printf("T826 ALL > CONTROL: %s (%.1f%% vs %.1f%%)\n", t826?"PASS":"FAIL", accs[0]*100, accs[3]*100);
        if (t825) pass++; else fail++;
        if (t826) pass++; else fail++;
    }
    printf("\n");

    // ═══════════════════════════════════════════════════════════════
    // EXP 3: Memory capacity — ALL vs NONE
    // ═══════════════════════════════════════════════════════════════
    printf("═══ EXP 3: Memory Capacity ═══\n\n");
    {
        host_rng = 123;
        for (int t = 0; t < STEPS; t++) h_input[t] = randf_signed();

        struct { const char* name; int flags; } conds[] = {
            {"ALL", FLAG_ALL}, {"BRANCH_ONLY", FLAG_BRANCH}, {"NONE", 0}
        };

        printf("%-18s ", "Condition");
        for (int d = 1; d <= 8; d++) printf("  d=%d", d);
        printf("   MC\n");

        float mc_vals[3] = {};
        for (int ci = 0; ci < 3; ci++) {
            std::vector<std::vector<float>> X;
            run_reservoir(conds[ci].flags, d_state, d_xchg, d_input, d_tlb,
                         h_input, h_state, STEPS, X);
            printf("%-18s", conds[ci].name);
            int n_eff = X.size(), split = n_eff*2/3;
            for (int d = 1; d <= 8; d++) {
                std::vector<float> y_tr, y_te;
                for(int i=0;i<split;i++){int t=i+WASHOUT; y_tr.push_back(t>=d?h_input[t-d]:0);}
                for(int i=split;i<n_eff;i++){int t=i+WASHOUT; y_te.push_back(t>=d?h_input[t-d]:0);}
                std::vector<std::vector<float>> X_tr(X.begin(),X.begin()+split);
                std::vector<std::vector<float>> X_te(X.begin()+split,X.end());
                auto res = ridge_regress(X_tr,y_tr,X_te,y_te,lambda);
                float r2 = res.r2 > 0 ? res.r2 : 0;
                mc_vals[ci] += r2;
                printf(" %.3f", r2);
            }
            printf("   %.3f\n", mc_vals[ci]);
        }
        bool t827 = mc_vals[0] > 0.5f;
        bool t828 = mc_vals[0] > mc_vals[2] * 0.5f;  // at least half of control
        printf("\nT827 MC(ALL) > 0.5:           %s (%.3f)\n", t827?"PASS":"FAIL", mc_vals[0]);
        printf("T828 MC(ALL) > 0.5*MC(CTRL):  %s (%.3f vs %.3f)\n", t828?"PASS":"FAIL", mc_vals[0], mc_vals[2]*0.5f);
        if (t827) pass++; else fail++;
        if (t828) pass++; else fail++;
    }
    printf("\n");

    // ═══════════════════════════════════════════════════════════════
    // EXP 4: NARMA-5
    // ═══════════════════════════════════════════════════════════════
    printf("═══ EXP 4: NARMA-5 ═══\n\n");
    {
        host_rng = 456;
        for (int t = 0; t < STEPS; t++) h_input[t] = randf() * 0.5f;
        std::vector<float> narma(STEPS, 0);
        for (int t = 5; t < STEPS-1; t++) {
            float ysum = 0;
            for (int k = 0; k < 5; k++) ysum += narma[t-k];
            narma[t+1] = 0.3f*narma[t] + 0.05f*narma[t]*ysum
                        + 1.5f*h_input[t-5]*h_input[t] + 0.1f;
            if (narma[t+1]>10) narma[t+1]=10;
            if (narma[t+1]<-10) narma[t+1]=-10;
        }

        struct { const char* name; int flags; } conds[] = {
            {"ALL", FLAG_ALL}, {"BRANCH_ONLY", FLAG_BRANCH}, {"NONE", 0}
        };

        printf("%-18s %8s %8s\n", "Condition", "NRMSE", "R²");
        float r2_vals[3];
        for (int ci = 0; ci < 3; ci++) {
            std::vector<std::vector<float>> X;
            run_reservoir(conds[ci].flags, d_state, d_xchg, d_input, d_tlb,
                         h_input, h_state, STEPS, X);
            int n_eff=X.size(), split=n_eff*2/3;
            std::vector<float> y_tr, y_te;
            for(int i=0;i<split;i++) y_tr.push_back(narma[i+WASHOUT]);
            for(int i=split;i<n_eff;i++) y_te.push_back(narma[i+WASHOUT]);
            std::vector<std::vector<float>> X_tr(X.begin(),X.begin()+split);
            std::vector<std::vector<float>> X_te(X.begin()+split,X.end());
            auto res = ridge_regress(X_tr,y_tr,X_te,y_te,lambda);
            r2_vals[ci] = res.r2;
            printf("%-18s %8.3f %8.3f\n", conds[ci].name, res.nrmse, res.r2);
        }
        bool t829 = r2_vals[0] > r2_vals[2];
        printf("\nT829 R²(ALL) > R²(CTRL): %s (%.3f vs %.3f)\n", t829?"PASS":"FAIL", r2_vals[0], r2_vals[2]);
        if (t829) pass++; else fail++;
    }
    printf("\n");

    // ═══════════════════════════════════════════════════════════════
    // EXP 5: Reproducibility — run XOR 3 times, check consistency
    // ═══════════════════════════════════════════════════════════════
    printf("═══ EXP 5: XOR τ=1 Reproducibility (3 runs) ═══\n\n");
    {
        float xor_runs[3];
        for (int run = 0; run < 3; run++) {
            host_rng = 42 + run * 100;
            for (int t = 0; t < STEPS; t++) h_input[t] = (randf() > 0.5f) ? 1.0f : -1.0f;
            std::vector<std::vector<float>> X;
            run_reservoir(FLAG_ALL, d_state, d_xchg, d_input, d_tlb,
                         h_input, h_state, STEPS, X);
            int n_eff = X.size();
            std::vector<int> y_xor;
            for (int i = 0; i < n_eff; i++) {
                int t = i + WASHOUT;
                int target = (t>=1) ? ((h_input[t]>0)!=(h_input[t-1]>0)?1:0) : 0;
                y_xor.push_back(target);
            }
            int split=n_eff*2/3;
            std::vector<std::vector<float>> X_tr(X.begin(),X.begin()+split);
            std::vector<std::vector<float>> X_te(X.begin()+split,X.end());
            std::vector<int> y_tr(y_xor.begin(),y_xor.begin()+split);
            std::vector<int> y_te(y_xor.begin()+split,y_xor.end());
            xor_runs[run] = ridge_classify_binary(X_tr,y_tr,X_te,y_te,lambda);
            printf("  Run %d: %.1f%%\n", run+1, xor_runs[run]*100);
        }
        float xor_mean = (xor_runs[0]+xor_runs[1]+xor_runs[2])/3.0f;
        float xor_min = std::min({xor_runs[0],xor_runs[1],xor_runs[2]});
        printf("  Mean: %.1f%%, Min: %.1f%%\n", xor_mean*100, xor_min*100);
        bool t830 = xor_min > 0.60f;
        printf("\nT830 XOR min across 3 runs > 60%%: %s (%.1f%%)\n", t830?"PASS":"FAIL", xor_min*100);
        if (t830) pass++; else fail++;
    }
    printf("\n");

    // ═══════════════════════════════════════════════════════════════
    // EXP 6: Effective dimension
    // ═══════════════════════════════════════════════════════════════
    printf("═══ EXP 6: Effective Dimension ═══\n\n");
    {
        host_rng = 789;
        for (int t = 0; t < STEPS; t++) h_input[t] = randf_signed();

        struct { const char* name; int flags; } conds[] = {
            {"ALL", FLAG_ALL}, {"BRANCH_ONLY", FLAG_BRANCH}, {"NONE", 0}
        };
        float eff_dim[3];
        for (int ci = 0; ci < 3; ci++) {
            std::vector<std::vector<float>> X;
            run_reservoir(conds[ci].flags, d_state, d_xchg, d_input, d_tlb,
                         h_input, h_state, STEPS, X);
            int n_eff=X.size(), d=X[0].size();
            int n_pairs=200; float total_corr=0; int count=0;
            for(int p=0;p<n_pairs;p++){
                int j1=(p*7+3)%d, j2=(p*13+7)%d; if(j1==j2) continue;
                float m1=0,m2_=0,s1=0,s2=0,c=0;
                for(int i=0;i<n_eff;i++){m1+=X[i][j1]; m2_+=X[i][j2];}
                m1/=n_eff; m2_/=n_eff;
                for(int i=0;i<n_eff;i++){float a=X[i][j1]-m1,b=X[i][j2]-m2_;
                    s1+=a*a; s2+=b*b; c+=a*b;}
                s1=sqrtf(s1); s2=sqrtf(s2);
                if(s1>1e-8&&s2>1e-8){total_corr+=fabsf(c/(s1*s2)); count++;}
            }
            float avg_corr=count>0?total_corr/count:1.0f;
            eff_dim[ci]=d/(1.0f+(d-1)*avg_corr*avg_corr);
            printf("  %-18s avg_corr=%.4f eff_dim=%.1f\n", conds[ci].name, avg_corr, eff_dim[ci]);
        }
        bool t831 = eff_dim[0] > eff_dim[2];
        printf("\nT831 EffDim(ALL) > EffDim(CTRL): %s (%.1f vs %.1f)\n",
               t831?"PASS":"FAIL", eff_dim[0], eff_dim[2]);
        if (t831) pass++; else fail++;
    }
    printf("\n");

    // ═══════════════════════════════════════════════════════════════
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  z2262v3 SUMMARY: %d/%d PASS                                ║\n", pass, pass+fail);
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    printf("Key: T825-T826 classification, T827-T828 memory,\n");
    printf("     T829 NARMA, T830 XOR reproducibility, T831 eff_dim\n");

    free(h_state); free(h_input);
    HIP_CHECK(hipFree(d_state)); HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree((void*)d_xchg)); HIP_CHECK(hipFree(d_tlb));
    return 0;
}
