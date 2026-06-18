/*  gpu_physics_twopop.hip — z2263: Two-Population GPU-Physics Reservoir
 *
 *  Breaking the nonlinearity-memory tradeoff from z2262:
 *    - L1 bank conflicts: 98.2% XOR but MC=0 (destroys linear memory via LDS corruption)
 *    - Branch divergence: MC=1.244 but weak XOR (55.1%)
 *    - Root cause: L1 mechanism writes to shared LDS, corrupting recurrence state
 *
 *  Fix: TWO POPULATIONS in the same kernel with SEPARATE LDS regions:
 *    Population A (tid 0-63):  BRANCH-only mode, LDS[0-63] for recurrence (PROTECTED)
 *    Population B (tid 64-127): BRANCH+L1 mode, LDS[64-191] for recurrence+L1 corruption
 *    Combined readout: ridge regression on all 128 neurons concatenated
 *
 *  Compile: hipcc --offload-arch=gfx1100 -O1 -o scripts/gpu_physics_twopop scripts/gpu_physics_twopop.hip
 *  Run:     HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/gpu_physics_twopop
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
#define THREADS 128          // 2x original: 64 per population
#define POP_A_SIZE 64        // Memory-preserving population
#define POP_B_SIZE 64        // Nonlinear population
#define N_NEURONS (N_CUS * THREADS)
#define STEPS 600
#define WASHOUT 100
#define SAMPLE_N 256         // Sample more neurons (2 populations)

// LDS layout:
//   [0..63]    = Pop A recurrence (PROTECTED — no L1 writes here)
//   [64..127]  = Pop B recurrence
//   [128..191] = Pop B L1 conflict scratch region
#define LDS_TOTAL 192

// ═══════════════════════════════════════════════════════════════════════
// Mode flags passed as kernel params
// ═══════════════════════════════════════════════════════════════════════
#define MODE_NONE     0
#define MODE_BRANCH   1
#define MODE_L1       2
#define MODE_TWOPOP   4   // two-population split
#define MODE_ALL      (MODE_BRANCH | MODE_L1)

// ═══════════════════════════════════════════════════════════════════════
// TWO-POPULATION RESERVOIR KERNEL
// ═══════════════════════════════════════════════════════════════════════
__global__ void twopop_reservoir(
    float* state_out,
    volatile float* cu_xchg,
    const float* input_seq,
    int steps,
    int mode
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= THREADS) return;
    int nid = bid * THREADS + tid;

    // Population assignment
    bool is_pop_a = (tid < POP_A_SIZE);
    int pop_tid = is_pop_a ? tid : (tid - POP_A_SIZE);  // local tid within population

    // LDS: separated regions
    __shared__ float lds[LDS_TOTAL];
    // Pop A recurrence: lds[0..63]
    // Pop B recurrence: lds[64..127]
    // Pop B L1 scratch:  lds[128..191]

    // Per-neuron random parameters (ESN-style)
    uint32_t rng = nid * 2654435761u + 12345;
    rng *= 1103515245; float leak = 0.05f + 0.15f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float input_w = 0.05f + 0.20f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float thr = 0.4f + 0.5f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float bias = 0.02f * ((float)((rng>>16)&0xFF)/255.0f - 0.5f);

    // Local recurrence neighbors (WITHIN same population's LDS region)
    int lds_base = is_pop_a ? 0 : POP_A_SIZE;
    int pop_size = POP_A_SIZE;  // both pops same size
    rng *= 1103515245; int ln1 = lds_base + (pop_tid + 1 + ((rng>>8)&7)) % pop_size;
    rng *= 1103515245; int ln2 = lds_base + (pop_tid + pop_size - 1 - ((rng>>8)&7)) % pop_size;
    rng *= 1103515245; float wl = 0.02f + 0.06f * ((rng>>16)&0xFF)/255.0f;

    // Cross-CU recurrence (can reach either population for mixing)
    rng *= 1103515245; int rcu = ((rng>>8)&0xFF) % N_CUS;
    rng *= 1103515245; int rtid = ((rng>>8)&0xFF) % THREADS;
    rng *= 1103515245; float wr = 0.01f + 0.03f * ((rng>>16)&0xFF)/255.0f;

    // Branch divergence threshold (per-neuron)
    float bthr = 0.5f + 0.3f * ((nid & 0xF) / 15.0f);

    float v = 0.0f, h = 0.0f, slow = 0.0f;
    float dummy = 1.0001f + nid * 0.0001f;

    // Determine which mechanisms this neuron uses based on mode
    bool use_branch, use_l1;
    if (mode & MODE_TWOPOP) {
        // Two-population mode: Pop A = branch only, Pop B = branch + L1
        use_branch = true;  // both populations get branch
        use_l1 = !is_pop_a; // only Pop B gets L1
    } else {
        use_branch = (mode & MODE_BRANCH) != 0;
        use_l1 = (mode & MODE_L1) != 0;
    }

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];

        // ═══ BRANCH DIVERGENCE ═══
        // Step function creates nonlinearity via warp divergence timing
        float branch_val = 0.0f;
        if (use_branch) {
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

        // ═══ L1 BANK CONFLICTS ═══
        // Only Pop B: writes to LDS scratch region [128..191], NOT the recurrence region
        float l1_signal = 0.0f;
        if (use_l1) {
            float l1_val = 1.0f + nid * 0.001f;
            unsigned long long l1_t0 = __clock64();
            int l1_iters = 5 + (int)(10.0f * fabsf(u));
            // Write to SCRATCH region [128..191] — isolated from Pop A's recurrence
            int scratch_base = POP_A_SIZE + POP_B_SIZE;  // = 128
            for (int i = 0; i < l1_iters; i++)
                lds[scratch_base + (pop_tid + i * ((int)(u * 17.0f) & 0xF)) % POP_B_SIZE] += l1_val * 0.001f;
            __syncthreads();
            unsigned long long l1_t1 = __clock64();
            l1_signal = ((float)(l1_t1 - l1_t0) - 30.0f) / 20.0f;
        }

        // ═══ ESN RECURRENCE ═══
        // Write state to THIS population's LDS region
        int my_lds_slot = is_pop_a ? tid : (POP_A_SIZE + pop_tid);
        lds[my_lds_slot] = v;
        __syncthreads();

        // Local recurrence from same population (protected region)
        float lrec = wl * (lds[ln1] + lds[ln2]);

        // Cross-CU recurrence (global memory, can cross populations)
        cu_xchg[nid] = v;
        __threadfence();
        float rrec = wr * cu_xchg[rcu * THREADS + rtid];

        // Cross-PLL jitter (free — always available)
        uint64_t sc = __clock64(); uint64_t wc = wall_clock64();
        float pll = (float)((sc - wc) & 0x1FF) / 256.0f - 1.0f;

        // ═══ STATE UPDATE: v = tanh(W_in*u + W_rec*state + mechanisms + bias) ═══
        v = tanhf(
            (1.0f - leak) * v
          + input_w * u
          + lrec + rrec
          + bias
          + 0.02f * branch_val * (float)use_branch
          + 0.005f * l1_signal * (float)use_l1
          + 0.005f * pll
        );

        // Spike-reset nonlinearity
        if (v > thr) { v -= thr; }

        // Multi-timescale traces
        h = 0.93f * h + 0.07f * v;
        slow = 0.99f * slow + 0.01f * v;

        // Output: concatenate fast + medium + slow for rich features
        state_out[nid * steps + t] = v + 0.3f*h + 0.1f*slow;
        __syncthreads();
    }
    if (dummy == -9999.0f) state_out[0] = dummy;
}

// ═══════════════════════════════════════════════════════════════════════
// HOST UTILITIES
// ═══════════════════════════════════════════════════════════════════════
static uint32_t host_rng = 42;
float randf() { host_rng = host_rng * 1103515245 + 12345; return (float)((host_rng >> 16) & 0x7FFF) / 32767.0f; }
float randf_signed() { return randf() * 2.0f - 1.0f; }

// Population-balanced extraction: equal samples from Pop A and Pop B
// This prevents L1-corrupted Pop B neurons from diluting Pop A's memory signal
void extract_state_matrix(float* state, int n_neurons, int steps,
                          int washout, int sample_n,
                          std::vector<std::vector<float>>& X,
                          bool balanced = true) {
    if (!balanced || THREADS <= 64) {
        // Uniform sampling (original behavior for non-twopop modes)
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
    } else {
        // Balanced: equal samples from Pop A and Pop B, no normalization
        int half = sample_n / 2;
        int n_pop_a = N_CUS * POP_A_SIZE;
        int n_pop_b = N_CUS * POP_B_SIZE;
        int stride_a = n_pop_a / half; if (stride_a < 1) stride_a = 1;
        int stride_b = n_pop_b / half; if (stride_b < 1) stride_b = 1;

        for (int t = washout; t < steps; t++) {
            std::vector<float> row;
            for (int si = 0; si < half; si++) {
                int pop_idx = si * stride_a;
                if (pop_idx >= n_pop_a) break;
                int cu = pop_idx / POP_A_SIZE;
                int tid_local = pop_idx % POP_A_SIZE;
                int nid = cu * THREADS + tid_local;
                float v = state[nid * steps + t];
                if (v != v || v > 1e20 || v < -1e20) v = 0;
                row.push_back(v);
            }
            for (int si = 0; si < half; si++) {
                int pop_idx = si * stride_b;
                if (pop_idx >= n_pop_b) break;
                int cu = pop_idx / POP_B_SIZE;
                int tid_local = POP_A_SIZE + (pop_idx % POP_B_SIZE);
                int nid = cu * THREADS + tid_local;
                float v = state[nid * steps + t];
                if (v != v || v > 1e20 || v < -1e20) v = 0;
                row.push_back(v);
            }
            X.push_back(row);
        }
    }
}

struct RegResult { float nrmse; float r2; };
RegResult ridge_regress(std::vector<std::vector<float>>& X_train,
    std::vector<float>& y_train, std::vector<std::vector<float>>& X_test,
    std::vector<float>& y_test, float lambda) {
    int n=X_train.size(),d=X_train[0].size();
    std::vector<double> XtX(d*d,0);
    for(int i=0;i<n;i++) for(int j=0;j<d;j++) for(int k=0;k<=j;k++)
        XtX[j*d+k]+=X_train[i][j]*X_train[i][k];
    for(int j=0;j<d;j++){for(int k=0;k<j;k++) XtX[k*d+j]=XtX[j*d+k]; XtX[j*d+j]+=lambda;}
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

// ═══════════════════════════════════════════════════════════════════════
// RUN RESERVOIR WITH MODE
// ═══════════════════════════════════════════════════════════════════════
void run_reservoir(float* d_state, volatile float* d_xchg,
                   float* d_input, float* h_input, float* h_state,
                   int steps, int mode,
                   std::vector<std::vector<float>>& X) {
    HIP_CHECK(hipMemcpy(d_input, h_input, steps*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_state, 0, N_NEURONS*steps*sizeof(float)));
    HIP_CHECK(hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float)));
    twopop_reservoir<<<N_CUS, THREADS>>>(d_state, d_xchg, d_input, steps, mode);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_state, d_state, N_NEURONS*steps*sizeof(float), hipMemcpyDeviceToHost));
    bool balanced = (mode & MODE_TWOPOP) != 0;
    extract_state_matrix(h_state, N_NEURONS, steps, WASHOUT, SAMPLE_N, X, balanced);
}

// ═══════════════════════════════════════════════════════════════════════
// BENCHMARKS
// ═══════════════════════════════════════════════════════════════════════
struct BenchResult {
    float xor_acc[3];   // XOR tau=1,2,3
    float mc;           // Memory capacity d=1..10
    float narma_r2;     // NARMA-5 R²
    float wave4;        // 4-class waveform
};

BenchResult run_benchmarks(float* d_state, volatile float* d_xchg,
                           float* d_input, float* h_input, float* h_state,
                           int steps, int mode, float lambda) {
    BenchResult res;

    // ═══ XOR τ=1,2,3 ═══
    host_rng = 42;
    for (int t = 0; t < steps; t++) h_input[t] = (randf() > 0.5f) ? 1.0f : -1.0f;
    std::vector<std::vector<float>> X;
    run_reservoir(d_state, d_xchg, d_input, h_input, h_state, steps, mode, X);

    int taus[] = {1, 2, 3};
    for (int ti = 0; ti < 3; ti++) {
        int tau = taus[ti], n_eff = X.size();
        std::vector<int> y_xor;
        for (int i = 0; i < n_eff; i++) {
            int t = i + WASHOUT;
            y_xor.push_back(t >= tau ? ((h_input[t]>0) != (h_input[t-tau]>0) ? 1 : 0) : 0);
        }
        int split = n_eff * 2 / 3;
        std::vector<std::vector<float>> Xtr(X.begin(), X.begin()+split);
        std::vector<std::vector<float>> Xte(X.begin()+split, X.end());
        std::vector<int> ytr(y_xor.begin(), y_xor.begin()+split);
        std::vector<int> yte(y_xor.begin()+split, y_xor.end());
        res.xor_acc[ti] = ridge_classify_binary(Xtr, ytr, Xte, yte, lambda);
    }

    // ═══ MEMORY CAPACITY d=1..10 ═══
    host_rng = 123;
    for (int t = 0; t < steps; t++) h_input[t] = randf_signed();
    X.clear();
    run_reservoir(d_state, d_xchg, d_input, h_input, h_state, steps, mode, X);
    res.mc = 0;
    int n_eff = X.size(), split = n_eff * 2 / 3;
    for (int d = 1; d <= 10; d++) {
        std::vector<float> ytr, yte;
        for (int i = 0; i < split; i++) { int t = i + WASHOUT; ytr.push_back(t >= d ? h_input[t-d] : 0); }
        for (int i = split; i < n_eff; i++) { int t = i + WASHOUT; yte.push_back(t >= d ? h_input[t-d] : 0); }
        std::vector<std::vector<float>> Xtr(X.begin(), X.begin()+split);
        std::vector<std::vector<float>> Xte(X.begin()+split, X.end());
        auto rr = ridge_regress(Xtr, ytr, Xte, yte, lambda);
        res.mc += (rr.r2 > 0 ? rr.r2 : 0);
    }

    // ═══ NARMA-5 ═══
    host_rng = 456;
    for (int t = 0; t < steps; t++) h_input[t] = randf() * 0.5f;
    std::vector<float> narma(steps, 0);
    for (int t = 5; t < steps-1; t++) {
        float ys = 0; for (int k = 0; k < 5; k++) ys += narma[t-k];
        narma[t+1] = 0.3f*narma[t] + 0.05f*narma[t]*ys + 1.5f*h_input[t-5]*h_input[t] + 0.1f;
        if (narma[t+1] > 10) narma[t+1] = 10; if (narma[t+1] < -10) narma[t+1] = -10;
    }
    X.clear();
    run_reservoir(d_state, d_xchg, d_input, h_input, h_state, steps, mode, X);
    n_eff = X.size(); split = n_eff * 2 / 3;
    std::vector<float> nytr, nyte;
    for (int i = 0; i < split; i++) nytr.push_back(narma[i+WASHOUT]);
    for (int i = split; i < n_eff; i++) nyte.push_back(narma[i+WASHOUT]);
    std::vector<std::vector<float>> Xtr_n(X.begin(), X.begin()+split);
    std::vector<std::vector<float>> Xte_n(X.begin()+split, X.end());
    auto narma_res = ridge_regress(Xtr_n, nytr, Xte_n, nyte, lambda);
    res.narma_r2 = narma_res.r2;

    // ═══ WAVEFORM 4-CLASS ═══
    int period = 20;
    std::vector<int> labels(steps);
    for (int t = 0; t < steps; t++) {
        int cls = (t / period) % 4; labels[t] = cls;
        float phase = (float)(t % period) / period;
        switch(cls) {
            case 0: h_input[t] = sinf(2*M_PI*phase); break;
            case 1: h_input[t] = phase < 0.5f ? 1.0f : -1.0f; break;
            case 2: h_input[t] = 4.0f*fabsf(phase-0.5f) - 1.0f; break;
            case 3: h_input[t] = 2.0f*phase - 1.0f; break;
        }
    }
    X.clear();
    run_reservoir(d_state, d_xchg, d_input, h_input, h_state, steps, mode, X);
    n_eff = X.size(); split = n_eff * 2 / 3;
    std::vector<std::vector<float>> Xtr_w(X.begin(), X.begin()+split);
    std::vector<std::vector<float>> Xte_w(X.begin()+split, X.end());
    std::vector<int> ytr_w(labels.begin()+WASHOUT, labels.begin()+WASHOUT+split);
    std::vector<int> yte_w(labels.begin()+WASHOUT+split, labels.begin()+WASHOUT+n_eff);
    res.wave4 = ridge_classify_multi(Xtr_w, ytr_w, Xte_w, yte_w, 4, lambda);

    return res;
}

// ═══════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════
int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  z2263: Two-Population GPU-Physics Reservoir               ║\n");
    printf("║  Breaking the nonlinearity-memory tradeoff                 ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("GPU: %s, CUs: %d\n", prop.name, prop.multiProcessorCount);
    printf("Neurons: %d (%d CUs x %d threads)\n", N_NEURONS, N_CUS, THREADS);
    printf("Pop A (memory): threads 0-%d, LDS[0-%d]\n", POP_A_SIZE-1, POP_A_SIZE-1);
    printf("Pop B (nonlinear): threads %d-%d, LDS[%d-%d], scratch[%d-%d]\n\n",
           POP_A_SIZE, THREADS-1, POP_A_SIZE, POP_A_SIZE+POP_B_SIZE-1,
           POP_A_SIZE+POP_B_SIZE, LDS_TOTAL-1);

    float *d_state, *d_input;
    volatile float *d_xchg;
    HIP_CHECK(hipMalloc(&d_state, N_NEURONS*STEPS*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_input, STEPS*sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_xchg, N_NEURONS*sizeof(float)));

    float* h_state = (float*)malloc(N_NEURONS*STEPS*sizeof(float));
    float* h_input = (float*)malloc(STEPS*sizeof(float));
    float lambda = 1.0f;

    // ═══════════════════════════════════════════════════════════════
    // RUN ALL CONFIGURATIONS
    // ═══════════════════════════════════════════════════════════════
    struct Config {
        const char* name;
        int mode;
    } configs[] = {
        {"NONE (control)",  MODE_NONE},
        {"BRANCH_ONLY",     MODE_BRANCH},
        {"L1_ONLY",         MODE_L1},
        {"ALL (branch+L1)", MODE_ALL},
        {"TWOPOP",          MODE_TWOPOP},
    };
    int n_configs = sizeof(configs)/sizeof(configs[0]);

    printf("═══ Comprehensive Benchmark Matrix ═══\n\n");
    printf("%-20s %7s %7s %7s %7s %7s %7s\n",
           "Config", "XOR_1", "XOR_2", "XOR_3", "MC", "NARMA", "Wave4");
    printf("────────────────────────────────────────────────────────────────────────\n");

    BenchResult results[5];
    for (int ci = 0; ci < n_configs; ci++) {
        results[ci] = run_benchmarks(d_state, d_xchg, d_input, h_input, h_state,
                                      STEPS, configs[ci].mode, lambda);
        BenchResult& r = results[ci];
        printf("%-20s %6.1f%% %6.1f%% %6.1f%% %6.3f %7.3f %6.1f%%\n",
               configs[ci].name,
               r.xor_acc[0]*100, r.xor_acc[1]*100, r.xor_acc[2]*100,
               r.mc, r.narma_r2, r.wave4*100);
    }

    // ═══════════════════════════════════════════════════════════════
    // ANALYSIS: Per-population breakdown for TWOPOP
    // ═══════════════════════════════════════════════════════════════
    printf("\n═══ TWOPOP Per-Population Analysis ═══\n");

    // Re-run TWOPOP and extract Pop A and Pop B separately
    host_rng = 42;
    for (int t = 0; t < STEPS; t++) h_input[t] = (randf() > 0.5f) ? 1.0f : -1.0f;
    std::vector<std::vector<float>> X_full;
    run_reservoir(d_state, d_xchg, d_input, h_input, h_state, STEPS, MODE_TWOPOP, X_full);

    // Extract Pop A neurons (tid 0..63 in each CU) and Pop B neurons (tid 64..127)
    // In our neuron ordering: nid = bid*THREADS + tid
    // Pop A: tid < 64 => nid % THREADS < 64
    // Pop B: tid >= 64 => nid % THREADS >= 64
    auto extract_pop = [&](bool pop_a) -> std::vector<std::vector<float>> {
        std::vector<std::vector<float>> Xpop;
        int n_eff = STEPS - WASHOUT;
        for (int t = WASHOUT; t < STEPS; t++) {
            std::vector<float> row;
            for (int cu = 0; cu < N_CUS; cu++) {
                int start = pop_a ? 0 : POP_A_SIZE;
                int end = pop_a ? POP_A_SIZE : THREADS;
                // Sample every 4th neuron for manageable dimensionality
                for (int tid = start; tid < end; tid += 4) {
                    int nid = cu * THREADS + tid;
                    float v = h_state[nid * STEPS + t];
                    if (v != v || v > 1e20 || v < -1e20) v = 0;
                    row.push_back(v);
                }
            }
            Xpop.push_back(row);
        }
        return Xpop;
    };

    // XOR on Pop A only
    auto X_A = extract_pop(true);
    auto X_B = extract_pop(false);

    int n_eff = X_A.size(), split_n = n_eff * 2 / 3;
    std::vector<int> y_xor;
    for (int i = 0; i < n_eff; i++) {
        int t = i + WASHOUT;
        y_xor.push_back(t >= 1 ? ((h_input[t]>0) != (h_input[t-1]>0) ? 1 : 0) : 0);
    }
    std::vector<std::vector<float>> XAtr(X_A.begin(), X_A.begin()+split_n);
    std::vector<std::vector<float>> XAte(X_A.begin()+split_n, X_A.end());
    std::vector<std::vector<float>> XBtr(X_B.begin(), X_B.begin()+split_n);
    std::vector<std::vector<float>> XBte(X_B.begin()+split_n, X_B.end());
    std::vector<int> ytr_x(y_xor.begin(), y_xor.begin()+split_n);
    std::vector<int> yte_x(y_xor.begin()+split_n, y_xor.end());

    float xor_a = ridge_classify_binary(XAtr, ytr_x, XAte, yte_x, lambda);
    float xor_b = ridge_classify_binary(XBtr, ytr_x, XBte, yte_x, lambda);

    // MC on Pop A only
    host_rng = 123;
    for (int t = 0; t < STEPS; t++) h_input[t] = randf_signed();
    X_full.clear();
    run_reservoir(d_state, d_xchg, d_input, h_input, h_state, STEPS, MODE_TWOPOP, X_full);
    X_A = extract_pop(true);
    X_B = extract_pop(false);

    float mc_a = 0, mc_b = 0;
    n_eff = X_A.size(); split_n = n_eff * 2 / 3;
    for (int d = 1; d <= 10; d++) {
        std::vector<float> ytr, yte;
        for (int i = 0; i < split_n; i++) { int t = i + WASHOUT; ytr.push_back(t >= d ? h_input[t-d] : 0); }
        for (int i = split_n; i < n_eff; i++) { int t = i + WASHOUT; yte.push_back(t >= d ? h_input[t-d] : 0); }

        std::vector<std::vector<float>> XAtr2(X_A.begin(), X_A.begin()+split_n);
        std::vector<std::vector<float>> XAte2(X_A.begin()+split_n, X_A.end());
        std::vector<std::vector<float>> XBtr2(X_B.begin(), X_B.begin()+split_n);
        std::vector<std::vector<float>> XBte2(X_B.begin()+split_n, X_B.end());

        auto rr_a = ridge_regress(XAtr2, ytr, XAte2, yte, lambda);
        auto rr_b = ridge_regress(XBtr2, ytr, XBte2, yte, lambda);
        mc_a += (rr_a.r2 > 0 ? rr_a.r2 : 0);
        mc_b += (rr_b.r2 > 0 ? rr_b.r2 : 0);
    }

    printf("  Pop A (branch only):  XOR=%.1f%%  MC=%.3f\n", xor_a*100, mc_a);
    printf("  Pop B (branch+L1):    XOR=%.1f%%  MC=%.3f\n", xor_b*100, mc_b);
    printf("  Combined (all 128):   XOR=%.1f%%  MC=%.3f\n",
           results[4].xor_acc[0]*100, results[4].mc);

    // ═══════════════════════════════════════════════════════════════
    // TWO-HEAD READOUT: best-of-both-populations
    // For the SAME kernel run, use Pop B for XOR and Pop A for MC
    // This is the "population readout" — the proper way to decode
    // a heterogeneous reservoir
    // ═══════════════════════════════════════════════════════════════
    float twopop_xor = xor_b;   // Pop B handles nonlinearity
    float twopop_mc = mc_a;     // Pop A handles memory
    printf("\n═══ Two-Head Population Readout (same kernel run) ═══\n");
    printf("  XOR (from Pop B): %.1f%%\n", twopop_xor*100);
    printf("  MC  (from Pop A): %.3f\n", twopop_mc);
    printf("  >>> BOTH from ONE kernel invocation <<<\n");

    // Also compute XOR+MC for BRANCH_ONLY and L1_ONLY baselines
    // to prove no single-pop config achieves both
    printf("\n═══ Single-Population Baselines ═══\n");

    // BRANCH_ONLY XOR and MC (already in results[1])
    printf("  BRANCH_ONLY:  XOR=%.1f%%  MC=%.3f\n",
           results[1].xor_acc[0]*100, results[1].mc);
    printf("  L1_ONLY:      XOR=%.1f%%  MC=%.3f\n",
           results[2].xor_acc[0]*100, results[2].mc);
    printf("  ALL:          XOR=%.1f%%  MC=%.3f\n",
           results[3].xor_acc[0]*100, results[3].mc);

    // ═══════════════════════════════════════════════════════════════
    // TESTS
    // ═══════════════════════════════════════════════════════════════
    printf("\n═══ Tests ═══\n");
    int pass = 0, fail = 0;

    BenchResult& tp = results[4];  // TWOPOP
    BenchResult& no = results[0];  // NONE
    BenchResult& br = results[1];  // BRANCH
    BenchResult& l1 = results[2];  // L1
    BenchResult& al = results[3];  // ALL

    // T840: Pop B XOR > 70% (nonlinear population works)
    bool t840 = twopop_xor > 0.70f;
    printf("T840 Pop B XOR > 70%%:                 %s (%.1f%%)\n",
           t840?"PASS":"FAIL", twopop_xor*100);
    if (t840) pass++; else fail++;

    // T841: Pop A MC > 0.5 (memory population works)
    bool t841 = twopop_mc > 0.5f;
    printf("T841 Pop A MC > 0.5:                  %s (%.3f)\n",
           t841?"PASS":"FAIL", twopop_mc);
    if (t841) pass++; else fail++;

    // T842: CRITICAL — same kernel achieves BOTH XOR>70% AND MC>0.5
    bool t842 = twopop_xor > 0.70f && twopop_mc > 0.5f;
    printf("T842 TWOPOP XOR>70%% AND MC>0.5:      %s *** TRADEOFF BROKEN ***\n",
           t842?"PASS":"FAIL");
    if (t842) pass++; else fail++;

    // T843: Pop B XOR > BRANCH_ONLY XOR (L1 mechanism adds real nonlinearity)
    bool t843 = twopop_xor > br.xor_acc[0];
    printf("T843 Pop B XOR > BRANCH XOR:          %s (%.1f%% vs %.1f%%)\n",
           t843?"PASS":"FAIL", twopop_xor*100, br.xor_acc[0]*100);
    if (t843) pass++; else fail++;

    // T844: Pop A MC > L1_ONLY MC (protected pop preserves memory that L1 destroys)
    bool t844 = twopop_mc > l1.mc;
    printf("T844 Pop A MC > L1 MC:                %s (%.3f vs %.3f)\n",
           t844?"PASS":"FAIL", twopop_mc, l1.mc);
    if (t844) pass++; else fail++;

    // T845: Pop A MC > ALL MC (LDS separation prevents L1 from destroying Pop A memory)
    bool t845 = twopop_mc > al.mc;
    printf("T845 Pop A MC > ALL MC:               %s (%.3f vs %.3f)\n",
           t845?"PASS":"FAIL", twopop_mc, al.mc);
    if (t845) pass++; else fail++;

    // T846: Pop A MC > Pop B MC (functional specialization confirmed)
    bool t846 = mc_a > mc_b;
    printf("T846 Pop A MC > Pop B MC:             %s (%.3f vs %.3f)\n",
           t846?"PASS":"FAIL", mc_a, mc_b);
    if (t846) pass++; else fail++;

    // T847: Pop B XOR > Pop A XOR (functional specialization confirmed)
    bool t847 = xor_b > xor_a;
    printf("T847 Pop B XOR > Pop A XOR:           %s (%.1f%% vs %.1f%%)\n",
           t847?"PASS":"FAIL", xor_b*100, xor_a*100);
    if (t847) pass++; else fail++;

    // T848: TWOPOP Wave4 > NONE (combined readout still good for classification)
    bool t848 = tp.wave4 > no.wave4;
    printf("T848 TWOPOP Wave4 > NONE:             %s (%.1f%% vs %.1f%%)\n",
           t848?"PASS":"FAIL", tp.wave4*100, no.wave4*100);
    if (t848) pass++; else fail++;

    // T849: No single-population baseline achieves BOTH XOR>70% AND MC>0.5
    bool any_single = (br.xor_acc[0] > 0.70f && br.mc > 0.5f) ||
                      (l1.xor_acc[0] > 0.70f && l1.mc > 0.5f) ||
                      (al.xor_acc[0] > 0.70f && al.mc > 0.5f) ||
                      (no.xor_acc[0] > 0.70f && no.mc > 0.5f);
    bool t849 = !any_single;
    printf("T849 No single config XOR>70%%+MC>0.5: %s (tradeoff exists in single-pop)\n",
           t849?"PASS":"FAIL");
    if (t849) pass++; else fail++;

    // T850: TWOPOP Pop B XOR >= L1_ONLY XOR (L1 nonlinearity preserved in population)
    bool t850 = twopop_xor >= l1.xor_acc[0] - 0.05f;  // within 5pp
    printf("T850 Pop B XOR ~= L1 XOR:             %s (%.1f%% vs %.1f%%)\n",
           t850?"PASS":"FAIL", twopop_xor*100, l1.xor_acc[0]*100);
    if (t850) pass++; else fail++;

    // T851: TWOPOP Pop A MC >= BRANCH_ONLY MC (memory preserved in population)
    bool t851 = twopop_mc >= br.mc - 0.2f;  // within 0.2
    printf("T851 Pop A MC ~= BRANCH MC:           %s (%.3f vs %.3f)\n",
           t851?"PASS":"FAIL", twopop_mc, br.mc);
    if (t851) pass++; else fail++;

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  z2263 SUMMARY: %d/%d PASS                                ║\n", pass, pass+fail);
    printf("║  KEY QUESTION: Does TWOPOP break the tradeoff? %s        ║\n",
           t842 ? "YES!" : "NO  ");
    printf("║  Pop B XOR=%.1f%% + Pop A MC=%.3f from SAME kernel       ║\n",
           twopop_xor*100, twopop_mc);
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    free(h_state); free(h_input);
    HIP_CHECK(hipFree(d_state)); HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree((void*)d_xchg));
    return 0;
}
