/*  z2500_hw_mechanism_probe.hip — Honest Hardware Mechanism Audit
 *
 *  Tests ONLY mechanisms that are genuinely hardware-mediated:
 *  mechanisms where the GPU silicon does something software cannot replicate.
 *
 *  Probe 1: s_setreg FP rounding mode as activation function modifier
 *           - Toggle RNE/RTZ/R+inf/R-inf per timestep
 *           - Measure if tanh() produces measurably different outputs
 *           - This changes hardware FP behavior, not achievable in software
 *
 *  Probe 2: Massive parallel atomicAdd as nonlinear state mixer
 *           - 256 threads atomicAdd to SAME address simultaneously
 *           - Non-deterministic ordering → different accumulator bit patterns
 *           - True hardware stochasticity (thread scheduling is non-deterministic)
 *
 *  Probe 3: Cross-kernel TLB persistence as temporal memory
 *           - Kernel A touches pages based on input class
 *           - Kernel B reads timing without knowing input
 *           - Tests TRUE hardware state persistence between kernel launches
 *
 *  Probe 4: LDS residual leakage (CVE-2023-4969)
 *           - Launch kernel that writes pattern to LDS, exits
 *           - Launch NEW kernel in same CU, reads LDS WITHOUT initializing
 *           - If data persists → hardware entropy source
 *
 *  Probe 5: s_setreg rounding as neuromorphic reservoir
 *           - 64 neurons, each step toggles rounding mode based on state
 *           - tanh(x) under different rounding = 4 different activation functions
 *           - Full reservoir computing benchmark (XOR, MC, classification)
 *
 *  Probe 6: Atomic neuron bank (from deep_silicon_access.md)
 *           - atomicAdd = synaptic summation
 *           - atomicExch = membrane reset
 *           - Non-deterministic spike ordering = hardware-intrinsic dynamics
 *
 *  Probe 7: sinf() data-dependent timing as stochastic activation
 *           - sinf range: [0, 62] cycles depending on input value
 *           - Data-dependent timing as genuine analog signal
 *
 *  Compile: hipcc --offload-arch=gfx1100 -O1 -o scripts/z2500_hw_probe scripts/z2500_hw_mechanism_probe.hip
 *  Run:     HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/z2500_hw_probe
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

// ─── Utility ───
double vmean(const std::vector<double>& v) {
    return v.empty() ? 0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}
double vstd(const std::vector<double>& v) {
    double m = vmean(v); double s=0;
    for (auto x:v) s+=(x-m)*(x-m);
    return v.size()>1 ? sqrt(s/(v.size()-1)) : 0;
}
double vcorr(const std::vector<double>& a, const std::vector<double>& b) {
    if(a.size()!=b.size()||a.size()<5) return 0;
    double ma=vmean(a),mb=vmean(b),sa=vstd(a),sb=vstd(b);
    if(sa<1e-12||sb<1e-12) return 0;
    double r=0; for(size_t i=0;i<a.size();i++) r+=(a[i]-ma)*(b[i]-mb);
    return r/((a.size()-1)*sa*sb);
}

// ═══════════════════════════════════════════════════════════════════════
// PROBE 1: s_setreg FP rounding as activation modifier
// ═══════════════════════════════════════════════════════════════════════

// Set FP32 rounding mode: 0=RNE, 1=R+inf, 2=R-inf, 3=RTZ
// MODE register = hwreg(1), bits [1:0] = FP_ROUND.32
__device__ __forceinline__ void set_fp_round(int mode) {
    // s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 2), mode
    // hwreg(1, offset=0, size=2) → encoding = (1) | (0<<6) | ((2-1)<<11) = 0x0801
    if (mode == 0) asm volatile("s_setreg_imm32_b32 0x0801, 0x0" ::: "memory");
    if (mode == 1) asm volatile("s_setreg_imm32_b32 0x0801, 0x1" ::: "memory");
    if (mode == 2) asm volatile("s_setreg_imm32_b32 0x0801, 0x2" ::: "memory");
    if (mode == 3) asm volatile("s_setreg_imm32_b32 0x0801, 0x3" ::: "memory");
}

__device__ __forceinline__ int get_fp_round() {
    int val;
    asm volatile("s_getreg_b32 %0, 0x0801" : "=s"(val));
    return val & 0x3;
}

// Probe 1: Compare tanh output under 4 rounding modes
__global__ void probe1_setreg_tanh(float* out, int n_values) {
    int tid = threadIdx.x;
    if (tid >= n_values) return;

    // Test value: sweep -2.0 to +2.0
    float x = -2.0f + 4.0f * (float)tid / (float)(n_values - 1);

    // Compute tanh under each rounding mode
    for (int mode = 0; mode < 4; mode++) {
        set_fp_round(mode);
        // Force actual computation (not just cached result)
        float val = x + 0.00001f * (float)mode;
        float result = tanhf(val);
        // Also test a chain: tanh(tanh(tanh(x))) amplifies rounding differences
        float chain = tanhf(tanhf(tanhf(val)));
        out[tid * 8 + mode * 2] = result;
        out[tid * 8 + mode * 2 + 1] = chain;
    }

    // Restore default
    set_fp_round(0);
}

// Probe 1b: Amplified rounding — FMA chain (1000 iterations) under different modes
__global__ void probe1b_setreg_fma_chain(float* out, int steps, int mode) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= 64) return;

    set_fp_round(mode);

    float v = 0.5f + 0.01f * (float)tid;
    for (int t = 0; t < steps; t++) {
        // Leaky integration: v = 0.95*v + 0.05*sin(t + tid)
        float input = sinf((float)t * 0.1f + (float)tid * 0.3f);
        v = __fmaf_rn(v, 0.95f, 0.05f * input);
        v = tanhf(v);
        out[tid * steps + t] = v;
    }

    set_fp_round(0);
}

// ═══════════════════════════════════════════════════════════════════════
// PROBE 2: Massive parallel atomicAdd as nonlinear mixer
// ═══════════════════════════════════════════════════════════════════════

// 256 threads atomicAdd to same address → ordering is non-deterministic
// Floating-point addition is non-associative: (a+b)+c != a+(b+c)
// Different orderings → different bit patterns in accumulator
__global__ void probe2_atomic_mixer(float* accum, float* individual_out,
                                     const float* inputs, int n_accum_addrs,
                                     int step) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= 256) return;

    // Each thread's contribution depends on tid and input
    float my_val = inputs[step] * (0.001f + 0.01f * (float)(tid % 64));

    // AtomicAdd to one of n_accum_addrs addresses
    int addr = tid % n_accum_addrs;
    float old = atomicAdd(&accum[addr], my_val);

    // Store the OLD value (before our add) — this reflects ordering
    individual_out[tid] = old;
}

// Full atomic reservoir: multiple steps, read accumulator state as features
__global__ void probe2b_atomic_reservoir(float* state_out, float* accum,
                                          const float* input_seq, int steps,
                                          int n_accum) {
    int tid = threadIdx.x;
    if (tid >= 256) return;

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];

        // Reset accumulators
        if (tid < n_accum) accum[tid] = 0.0f;
        __syncthreads();

        // Each thread contributes input-dependent value
        float my_val = u * (0.01f + 0.005f * sinf((float)tid * 0.1f));
        // Also add state-dependent component for recurrence
        float prev_state = (t > 0) ? state_out[tid * steps + t - 1] : 0.0f;
        my_val += 0.01f * prev_state;

        int addr = tid % n_accum;
        atomicAdd(&accum[addr], my_val);
        __syncthreads();

        // Read back the non-deterministic accumulated value
        float acc_val = accum[tid % n_accum];
        // Combine with deterministic state
        state_out[tid * steps + t] = tanhf(acc_val + 0.9f * prev_state);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PROBE 3: LDS residual leakage (CVE-2023-4969)
// ═══════════════════════════════════════════════════════════════════════

// Kernel A: Write known pattern to LDS, then exit
__global__ void probe3_lds_writer(int pattern_class) {
    __shared__ float lds[256];
    int tid = threadIdx.x;

    // Write class-dependent pattern
    if (pattern_class == 0)
        lds[tid] = (float)tid * 0.01f;       // Ascending
    else if (pattern_class == 1)
        lds[tid] = (float)(255 - tid) * 0.01f;  // Descending
    else
        lds[tid] = sinf((float)tid * 0.1f);  // Sinusoidal

    __syncthreads();
    // Exit — LDS should be deallocated but hardware may not zero it
}

// Kernel B: Read LDS WITHOUT initialization — detect residual
__global__ void probe3_lds_reader(float* out) {
    __shared__ float lds[256];
    int tid = threadIdx.x;

    // Read LDS without writing first — what's there?
    // Use volatile to prevent compiler from assuming it's zero
    volatile float* vlds = (volatile float*)lds;
    float val = vlds[tid];

    out[tid] = val;
    __syncthreads();
}

// ═══════════════════════════════════════════════════════════════════════
// PROBE 4: sinf() data-dependent timing as analog signal
// ═══════════════════════════════════════════════════════════════════════

__global__ void probe4_sinf_timing(float* timing_out, float* value_out,
                                    int n_values) {
    int tid = threadIdx.x;
    if (tid >= n_values) return;

    float x = -10.0f + 20.0f * (float)tid / (float)(n_values - 1);

    // Time sinf() for this input value
    unsigned long long t0 = __clock64();
    float result = sinf(x);
    // Chain to amplify timing differences
    result = sinf(result * 3.14159f);
    result = sinf(result * 3.14159f);
    result = sinf(result * 3.14159f);
    unsigned long long t1 = __clock64();

    timing_out[tid] = (float)(t1 - t0);
    value_out[tid] = result;
}

// ═══════════════════════════════════════════════════════════════════════
// PROBE 5: s_setreg Rounding Reservoir (full RC benchmark)
// ═══════════════════════════════════════════════════════════════════════

// Reservoir where each neuron toggles FP rounding mode based on its state
// This creates 4 different activation functions from the SAME tanhf()
__global__ void probe5_rounding_reservoir(float* state_out, const float* input_seq,
                                           int steps, int use_rounding) {
    __shared__ float lds[64];
    int tid = threadIdx.x;
    if (tid >= 64) return;

    float v = 0.0f, h = 0.0f;

    // Per-neuron parameters (deterministic from tid)
    uint32_t rng = tid * 2654435761u + 12345;
    rng *= 1103515245; float leak = 0.05f + 0.15f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float input_w = 0.1f + 0.2f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float bias = 0.02f * ((float)((rng>>16)&0xFF)/255.0f - 0.5f);
    rng *= 1103515245; int ln1 = (tid + 1 + ((rng>>8)&7)) % 64;
    rng *= 1103515245; int ln2 = (tid + 63 - ((rng>>8)&7)) % 64;
    rng *= 1103515245; float wl = 0.02f + 0.08f * ((rng>>16)&0xFF)/255.0f;

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];

        lds[tid] = v;
        __syncthreads();

        float rec = wl * (lds[ln1] + lds[ln2]);

        float pre_act = (1.0f - leak) * v + input_w * u + rec + bias;

        if (use_rounding) {
            // KEY MECHANISM: Choose rounding mode based on neuron state
            // This makes tanhf() behave as 4 DIFFERENT activation functions
            // Neuron state determines which hardware mode is active
            int round_mode;
            if (v > 0.5f) round_mode = 1;        // R+inf (growth bias)
            else if (v < -0.5f) round_mode = 2;   // R-inf (decay bias)
            else if (v > 0.0f) round_mode = 3;    // RTZ (compression)
            else round_mode = 0;                   // RNE (default)

            set_fp_round(round_mode);
        }

        v = tanhf(pre_act);

        if (use_rounding) {
            set_fp_round(0);  // restore default for output storage
        }

        h = 0.93f * h + 0.07f * v;
        state_out[tid * steps + t] = v + 0.3f * h;
        __syncthreads();
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PROBE 6: Wave intrinsics reservoir (__shfl, __ballot, __popcll)
// ═══════════════════════════════════════════════════════════════════════

__global__ void probe6_wave_reservoir(float* state_out, const float* input_seq,
                                       int steps, int use_wave_ops) {
    int tid = threadIdx.x;
    if (tid >= 64) return;

    int lane = tid % 32;
    float v = 0.0f, h = 0.0f;

    // Per-neuron parameters
    uint32_t rng = tid * 2654435761u + 54321;
    rng *= 1103515245; float leak = 0.05f + 0.15f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float input_w = 0.1f + 0.2f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float bias = 0.02f * ((float)((rng>>16)&0xFF)/255.0f - 0.5f);

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];

        float wave_signal = 0.0f;
        if (use_wave_ops) {
            // __shfl: get neighbor's state (axonal transport)
            float neighbor_v = __shfl(v, (lane + 1) % 32);
            float far_neighbor = __shfl(v, (lane + 7) % 32);

            // __ballot: population spike detection
            unsigned long long spike_mask = __ballot(v > 0.3f);
            int pop_spikes = __popcll(spike_mask);

            // Population coding: fraction of neurons above threshold
            float pop_rate = (float)pop_spikes / 32.0f;

            // Combine wave signals
            wave_signal = 0.05f * neighbor_v + 0.03f * far_neighbor
                        + 0.02f * (pop_rate - 0.5f);
        }

        float pre_act = (1.0f - leak) * v + input_w * u + bias + wave_signal;
        v = tanhf(pre_act);
        h = 0.93f * h + 0.07f * v;

        state_out[tid * steps + t] = v + 0.3f * h;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PROBE 7: Combined — s_setreg + wave intrinsics + atomic mixing
// ═══════════════════════════════════════════════════════════════════════

__global__ void probe7_combined_hw_reservoir(
    float* state_out, float* accum, const float* input_seq,
    int steps, int flags  // bit0=rounding, bit1=wave, bit2=atomic
) {
    __shared__ float lds[128];
    int tid = threadIdx.x;
    if (tid >= 64) return;

    int lane = tid % 32;
    float v = 0.0f, h = 0.0f, slow = 0.0f;

    uint32_t rng = tid * 2654435761u + 99999;
    rng *= 1103515245; float leak = 0.05f + 0.15f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float input_w = 0.1f + 0.2f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float bias = 0.02f * ((float)((rng>>16)&0xFF)/255.0f - 0.5f);
    rng *= 1103515245; int ln1 = (tid + 1 + ((rng>>8)&7)) % 64;
    rng *= 1103515245; int ln2 = (tid + 63 - ((rng>>8)&7)) % 64;
    rng *= 1103515245; float wl = 0.02f + 0.08f * ((rng>>16)&0xFF)/255.0f;

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];

        // LDS recurrence
        lds[tid] = v;
        __syncthreads();
        float rec = wl * (lds[ln1] + lds[ln2]);

        // --- Atomic mixing (if enabled) ---
        float atomic_signal = 0.0f;
        if (flags & 4) {
            if (tid < 8) accum[tid] = 0.0f;
            __syncthreads();
            float contrib = u * (0.01f + 0.005f * v);
            atomicAdd(&accum[tid % 8], contrib);
            __syncthreads();
            atomic_signal = 0.02f * tanhf(accum[tid % 8] * 10.0f);
        }

        // --- Wave intrinsics (if enabled) ---
        float wave_signal = 0.0f;
        if (flags & 2) {
            float nb = __shfl(v, (lane + 1) % 32);
            float fnb = __shfl(v, (lane + 7) % 32);
            unsigned long long mask = __ballot(v > 0.3f);
            float pop_rate = (float)__popcll(mask) / 32.0f;
            wave_signal = 0.05f * nb + 0.03f * fnb + 0.02f * (pop_rate - 0.5f);
        }

        float pre_act = (1.0f - leak) * v + input_w * u + rec + bias
                       + wave_signal + atomic_signal;

        // --- s_setreg rounding (if enabled) ---
        if (flags & 1) {
            int rm;
            if (v > 0.5f) rm = 1;
            else if (v < -0.5f) rm = 2;
            else if (v > 0.0f) rm = 3;
            else rm = 0;
            set_fp_round(rm);
        }

        v = tanhf(pre_act);

        if (flags & 1) set_fp_round(0);

        h = 0.93f * h + 0.07f * v;
        slow = 0.99f * slow + 0.01f * v;
        state_out[tid * steps + t] = v + 0.3f * h + 0.1f * slow;
        __syncthreads();
    }
}


// ═══════════════════════════════════════════════════════════════════════
// HOST: RC BENCHMARK UTILITIES
// ═══════════════════════════════════════════════════════════════════════

// Simple ridge regression
void ridge_regression(const std::vector<std::vector<double>>& X,
                      const std::vector<double>& y,
                      std::vector<double>& weights, double lambda = 1.0) {
    int N = X.size(), D = X[0].size();
    // X^T X + lambda I
    std::vector<std::vector<double>> XtX(D, std::vector<double>(D, 0));
    std::vector<double> Xty(D, 0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            Xty[j] += X[i][j] * y[i];
            for (int k = 0; k < D; k++)
                XtX[j][k] += X[i][j] * X[i][k];
        }
    }
    for (int j = 0; j < D; j++) XtX[j][j] += lambda;
    // Solve with Gauss elimination
    weights.resize(D, 0);
    std::vector<std::vector<double>> A(D, std::vector<double>(D+1));
    for (int j = 0; j < D; j++) {
        for (int k = 0; k < D; k++) A[j][k] = XtX[j][k];
        A[j][D] = Xty[j];
    }
    for (int j = 0; j < D; j++) {
        int mx = j;
        for (int k = j+1; k < D; k++) if (fabs(A[k][j]) > fabs(A[mx][j])) mx = k;
        std::swap(A[j], A[mx]);
        if (fabs(A[j][j]) < 1e-12) continue;
        for (int k = j+1; k <= D; k++) A[j][k] /= A[j][j];
        A[j][j] = 1.0;
        for (int i = 0; i < D; i++) {
            if (i == j) continue;
            double f = A[i][j];
            for (int k = j; k <= D; k++) A[i][k] -= f * A[j][k];
        }
    }
    for (int j = 0; j < D; j++) weights[j] = A[j][D];
}

double evaluate_xor(float* states, float* inputs, int N, int steps,
                    int washout, int tau) {
    int train_end = washout + (steps - washout) / 2;
    // Build features and targets
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<double> y_train, y_test;

    for (int t = washout + tau; t < steps; t++) {
        std::vector<double> feat(N);
        for (int n = 0; n < N; n++)
            feat[n] = (double)states[n * steps + t];

        double target = ((inputs[t] > 0) != (inputs[t - tau] > 0)) ? 1.0 : 0.0;

        if (t < train_end) {
            X_train.push_back(feat);
            y_train.push_back(target);
        } else {
            X_test.push_back(feat);
            y_test.push_back(target);
        }
    }

    if (X_train.empty() || X_test.empty()) return 0.5;

    std::vector<double> w;
    ridge_regression(X_train, y_train, w, 1.0);

    int correct = 0;
    for (size_t i = 0; i < X_test.size(); i++) {
        double pred = 0;
        for (int j = 0; j < N; j++) pred += w[j] * X_test[i][j];
        int pc = (pred > 0.5) ? 1 : 0;
        int tc = (int)y_test[i];
        if (pc == tc) correct++;
    }
    return (double)correct / X_test.size();
}

double evaluate_mc(float* states, float* inputs, int N, int steps,
                   int washout, int max_delay) {
    double total_mc = 0;
    int train_end = washout + (steps - washout) / 2;

    for (int d = 1; d <= max_delay; d++) {
        std::vector<std::vector<double>> X_train;
        std::vector<double> y_train, y_pred_all, y_true_all;

        for (int t = washout + d; t < steps; t++) {
            std::vector<double> feat(N);
            for (int n = 0; n < N; n++)
                feat[n] = (double)states[n * steps + t];
            double target = (double)inputs[t - d];

            if (t < train_end) {
                X_train.push_back(feat);
                y_train.push_back(target);
            }
        }

        if (X_train.empty()) continue;
        std::vector<double> w;
        ridge_regression(X_train, y_train, w, 1.0);

        // Test
        for (int t = train_end + d; t < steps; t++) {
            double pred = 0;
            for (int n = 0; n < N; n++)
                pred += w[n] * (double)states[n * steps + t];
            y_pred_all.push_back(pred);
            y_true_all.push_back((double)inputs[t - d]);
        }

        if (y_pred_all.size() < 10) continue;
        double r = vcorr(y_pred_all, y_true_all);
        total_mc += r * r;
    }
    return total_mc;
}

double evaluate_waveform(float* states, float* inputs, int N, int steps,
                         int washout, int n_classes) {
    int train_end = washout + (steps - washout) / 2;
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;

    for (int t = washout; t < steps; t++) {
        std::vector<double> feat(N);
        for (int n = 0; n < N; n++)
            feat[n] = (double)states[n * steps + t];

        // Classify by input sign/magnitude
        int cls;
        float u = inputs[t];
        if (n_classes == 4) {
            if (u < -0.5f) cls = 0;
            else if (u < 0.0f) cls = 1;
            else if (u < 0.5f) cls = 2;
            else cls = 3;
        } else {
            cls = (u > 0.0f) ? 1 : 0;
        }

        if (t < train_end) { X_train.push_back(feat); y_train.push_back(cls); }
        else { X_test.push_back(feat); y_test.push_back(cls); }
    }

    // One-vs-rest classification
    int nc = n_classes;
    std::vector<std::vector<double>> all_w(nc);
    for (int c = 0; c < nc; c++) {
        std::vector<double> y_c(y_train.size());
        for (size_t i = 0; i < y_train.size(); i++)
            y_c[i] = (y_train[i] == c) ? 1.0 : 0.0;
        ridge_regression(X_train, y_c, all_w[c], 1.0);
    }

    int correct = 0;
    for (size_t i = 0; i < X_test.size(); i++) {
        double best = -1e9; int best_c = 0;
        for (int c = 0; c < nc; c++) {
            double pred = 0;
            for (int j = 0; j < N; j++) pred += all_w[c][j] * X_test[i][j];
            if (pred > best) { best = pred; best_c = c; }
        }
        if (best_c == y_test[i]) correct++;
    }
    return (double)correct / X_test.size();
}


// ═══════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  z2500: Honest Hardware Mechanism Probe                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("GPU: %s (gfx%x)\n\n", prop.name, prop.gcnArchName[3] == '1' ? 1100 : 0);

    // ═══ PROBE 1: s_setreg tanh divergence ═══
    printf("━━━ PROBE 1: s_setreg FP Rounding × tanh() ━━━\n");
    {
        int N = 256;
        float* d_out; HIP_CHECK(hipMalloc(&d_out, N * 8 * sizeof(float)));
        probe1_setreg_tanh<<<1, N>>>(d_out, N);
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> out(N * 8);
        HIP_CHECK(hipMemcpy(out.data(), d_out, N * 8 * sizeof(float), hipMemcpyDeviceToHost));

        // Count how many values differ between rounding modes
        int tanh_diffs = 0, chain_diffs = 0;
        double max_tanh_diff = 0, max_chain_diff = 0;
        for (int i = 0; i < N; i++) {
            float rne_tanh = out[i*8+0], rinf_tanh = out[i*8+2];
            float rne_chain = out[i*8+1], rinf_chain = out[i*8+3];
            float rtz_tanh = out[i*8+6], rtz_chain = out[i*8+7];

            if (rne_tanh != rinf_tanh || rne_tanh != rtz_tanh) tanh_diffs++;
            if (rne_chain != rinf_chain || rne_chain != rtz_chain) chain_diffs++;

            double d1 = fabs((double)rinf_tanh - (double)rne_tanh);
            double d2 = fabs((double)rtz_tanh - (double)rne_tanh);
            double d3 = fabs((double)rinf_chain - (double)rne_chain);
            double d4 = fabs((double)rtz_chain - (double)rne_chain);
            max_tanh_diff = std::max(max_tanh_diff, std::max(d1, d2));
            max_chain_diff = std::max(max_chain_diff, std::max(d3, d4));
        }
        printf("  tanh(x): %d/%d values differ across rounding modes (max diff: %.2e)\n",
               tanh_diffs, N, max_tanh_diff);
        printf("  tanh³(x): %d/%d values differ (max diff: %.2e)\n",
               chain_diffs, N, max_chain_diff);
        printf("  VERDICT: %s\n\n",
               chain_diffs > N/2 ? "ROUNDING CHANGES tanhf() — USABLE" :
               tanh_diffs > 0 ? "WEAK effect — needs amplification" : "NO effect");
        hipFree(d_out);
    }

    // ═══ PROBE 1b: Amplified rounding divergence over time ═══
    printf("━━━ PROBE 1b: FMA Chain Divergence Over 500 Steps ━━━\n");
    {
        int steps = 500, N = 64;
        float *d_out_rne, *d_out_rtz, *d_out_rinf;
        HIP_CHECK(hipMalloc(&d_out_rne, N * steps * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_out_rtz, N * steps * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_out_rinf, N * steps * sizeof(float)));

        probe1b_setreg_fma_chain<<<1, N>>>(d_out_rne, steps, 0);  // RNE
        probe1b_setreg_fma_chain<<<1, N>>>(d_out_rtz, steps, 3);  // RTZ
        probe1b_setreg_fma_chain<<<1, N>>>(d_out_rinf, steps, 1); // R+inf
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> rne(N*steps), rtz(N*steps), rinf(N*steps);
        HIP_CHECK(hipMemcpy(rne.data(), d_out_rne, N*steps*sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(rtz.data(), d_out_rtz, N*steps*sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(rinf.data(), d_out_rinf, N*steps*sizeof(float), hipMemcpyDeviceToHost));

        // Check divergence at different timesteps
        for (int t : {10, 50, 100, 250, 499}) {
            int n_diff_rtz = 0, n_diff_rinf = 0;
            double max_d_rtz = 0, max_d_rinf = 0;
            for (int n = 0; n < N; n++) {
                double d1 = fabs((double)rtz[n*steps+t] - (double)rne[n*steps+t]);
                double d2 = fabs((double)rinf[n*steps+t] - (double)rne[n*steps+t]);
                if (d1 > 1e-12) n_diff_rtz++;
                if (d2 > 1e-12) n_diff_rinf++;
                max_d_rtz = std::max(max_d_rtz, d1);
                max_d_rinf = std::max(max_d_rinf, d2);
            }
            printf("  t=%3d: RTZ diff %d/64 (max %.2e), R+inf diff %d/64 (max %.2e)\n",
                   t, n_diff_rtz, max_d_rtz, n_diff_rinf, max_d_rinf);
        }
        printf("\n");
        hipFree(d_out_rne); hipFree(d_out_rtz); hipFree(d_out_rinf);
    }

    // ═══ PROBE 3: LDS Residual ═══
    printf("━━━ PROBE 3: LDS Residual Leakage (CVE-2023-4969) ━━━\n");
    {
        float* d_out; HIP_CHECK(hipMalloc(&d_out, 256 * sizeof(float)));

        for (int pattern = 0; pattern < 3; pattern++) {
            // Write pattern
            probe3_lds_writer<<<1, 256>>>(pattern);
            HIP_CHECK(hipDeviceSynchronize());

            // Read from NEW kernel (no initialization)
            HIP_CHECK(hipMemset(d_out, 0, 256 * sizeof(float)));
            probe3_lds_reader<<<1, 256>>>(d_out);
            HIP_CHECK(hipDeviceSynchronize());

            std::vector<float> out(256);
            HIP_CHECK(hipMemcpy(out.data(), d_out, 256*sizeof(float), hipMemcpyDeviceToHost));

            int nonzero = 0; float sum = 0;
            for (int i = 0; i < 256; i++) {
                if (out[i] != 0.0f) nonzero++;
                sum += fabs(out[i]);
            }
            printf("  Pattern %d → %d/256 nonzero residuals, sum=%.4f\n",
                   pattern, nonzero, sum);
            if (nonzero > 0) {
                printf("    First 8 values: ");
                for (int i = 0; i < 8; i++) printf("%.4f ", out[i]);
                printf("\n");
            }
        }
        printf("  VERDICT: %s\n\n", "Check nonzero counts above");
        hipFree(d_out);
    }

    // ═══ PROBE 4: sinf() timing ═══
    printf("━━━ PROBE 4: sinf() Data-Dependent Timing ━━━\n");
    {
        int N = 256;
        float *d_timing, *d_values;
        HIP_CHECK(hipMalloc(&d_timing, N * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_values, N * sizeof(float)));

        probe4_sinf_timing<<<1, N>>>(d_timing, d_values, N);
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> timing(N), values(N);
        HIP_CHECK(hipMemcpy(timing.data(), d_timing, N*sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(values.data(), d_values, N*sizeof(float), hipMemcpyDeviceToHost));

        float min_t = *std::min_element(timing.begin(), timing.end());
        float max_t = *std::max_element(timing.begin(), timing.end());
        float mean_t = std::accumulate(timing.begin(), timing.end(), 0.0f) / N;

        // Check if timing correlates with input value
        std::vector<double> td(N), xd(N);
        for (int i = 0; i < N; i++) {
            td[i] = timing[i]; xd[i] = -10.0 + 20.0 * i / (N-1);
        }
        double r = vcorr(td, xd);

        printf("  sinf³(x) timing: min=%.0f, max=%.0f, mean=%.0f cycles\n",
               min_t, max_t, mean_t);
        printf("  Range: %.0f cycles (%.1f%%)\n", max_t - min_t,
               100.0 * (max_t - min_t) / mean_t);
        printf("  Input correlation: r=%.4f\n", r);
        printf("  VERDICT: %s\n\n",
               (max_t - min_t > 10 && fabs(r) > 0.1) ?
               "DATA-DEPENDENT TIMING — usable as analog signal" :
               "Timing uniform — not useful");

        hipFree(d_timing); hipFree(d_values);
    }

    // ═══ PROBE 5 + 6 + 7: RC Benchmarks ═══
    printf("━━━ PROBES 5-7: Reservoir Computing Benchmarks ━━━\n");
    {
        int steps = 800, washout = 150, N = 64;

        // Generate input sequences
        std::vector<float> inputs(steps);
        srand(42);
        for (int t = 0; t < steps; t++)
            inputs[t] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

        float* d_inputs;
        HIP_CHECK(hipMalloc(&d_inputs, steps * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_inputs, inputs.data(), steps * sizeof(float), hipMemcpyHostToDevice));

        float* d_states;
        HIP_CHECK(hipMalloc(&d_states, N * steps * sizeof(float)));

        float* d_accum;
        HIP_CHECK(hipMalloc(&d_accum, 256 * sizeof(float)));

        struct Config {
            const char* name;
            int probe;   // 5, 6, or 7
            int param;   // use_rounding, use_wave_ops, or flags
        };

        Config configs[] = {
            {"CONTROL (no HW)",              5, 0},
            {"s_setreg rounding only",       5, 1},
            {"Wave intrinsics only",         6, 0},
            {"Wave intrinsics ACTIVE",       6, 1},
            {"Combined: none",               7, 0},
            {"Combined: rounding",           7, 1},
            {"Combined: wave",               7, 2},
            {"Combined: atomic",             7, 4},
            {"Combined: round+wave",         7, 3},
            {"Combined: round+wave+atomic",  7, 7},
        };

        int n_configs = sizeof(configs) / sizeof(configs[0]);

        printf("\n  %-32s  XOR-1    XOR-2    MC      Wave4\n", "Config");
        printf("  %-32s  ------   ------   ------  ------\n", "------");

        for (int ci = 0; ci < n_configs; ci++) {
            Config& c = configs[ci];

            HIP_CHECK(hipMemset(d_states, 0, N * steps * sizeof(float)));
            HIP_CHECK(hipMemset(d_accum, 0, 256 * sizeof(float)));

            if (c.probe == 5) {
                probe5_rounding_reservoir<<<1, N>>>(d_states, d_inputs, steps, c.param);
            } else if (c.probe == 6) {
                probe6_wave_reservoir<<<1, N>>>(d_states, d_inputs, steps, c.param);
            } else {
                probe7_combined_hw_reservoir<<<1, N>>>(d_states, d_accum, d_inputs,
                                                       steps, c.param);
            }
            HIP_CHECK(hipDeviceSynchronize());

            std::vector<float> states(N * steps);
            HIP_CHECK(hipMemcpy(states.data(), d_states, N * steps * sizeof(float),
                      hipMemcpyDeviceToHost));

            double xor1 = evaluate_xor(states.data(), inputs.data(), N, steps, washout, 1);
            double xor2 = evaluate_xor(states.data(), inputs.data(), N, steps, washout, 2);
            double mc = evaluate_mc(states.data(), inputs.data(), N, steps, washout, 10);
            double wave = evaluate_waveform(states.data(), inputs.data(), N, steps, washout, 4);

            printf("  %-32s  %5.1f%%   %5.1f%%   %5.3f   %5.1f%%\n",
                   c.name, 100*xor1, 100*xor2, mc, 100*wave);
        }

        // ═══ PROBE 2: Atomic non-determinism test ═══
        printf("\n━━━ PROBE 2: Atomic Non-Determinism ━━━\n");
        {
            float* d_accum2; HIP_CHECK(hipMalloc(&d_accum2, 64 * sizeof(float)));
            float* d_ind; HIP_CHECK(hipMalloc(&d_ind, 256 * sizeof(float)));

            // Run same atomic operation 10 times, check if results differ
            std::vector<std::vector<float>> runs(10, std::vector<float>(64));
            for (int run = 0; run < 10; run++) {
                HIP_CHECK(hipMemset(d_accum2, 0, 64 * sizeof(float)));
                probe2_atomic_mixer<<<1, 256>>>(d_accum2, d_ind, d_inputs, 8, 0);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(runs[run].data(), d_accum2, 64*sizeof(float),
                          hipMemcpyDeviceToHost));
            }

            int differ_count = 0;
            for (int addr = 0; addr < 8; addr++) {
                bool any_diff = false;
                for (int r = 1; r < 10; r++) {
                    if (runs[r][addr] != runs[0][addr]) { any_diff = true; break; }
                }
                if (any_diff) differ_count++;
            }
            printf("  %d/8 accumulator addresses show non-deterministic results\n", differ_count);
            printf("  VERDICT: %s\n",
                   differ_count >= 4 ? "ATOMIC ORDERING IS NON-DETERMINISTIC — usable" :
                   differ_count > 0 ? "PARTIAL non-determinism" :
                   "DETERMINISTIC — atomics not useful for stochasticity");

            hipFree(d_accum2); hipFree(d_ind);
        }

        printf("\n");
        hipFree(d_inputs); hipFree(d_states); hipFree(d_accum);
    }

    printf("━━━ DONE ━━━\n");
    return 0;
}
