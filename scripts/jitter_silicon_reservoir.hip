// jitter_silicon_reservoir.hip — GPU-INTERNAL reservoir computing
// Uses the LOWEST noise sources directly on silicon at GHz speed:
//   1. wall_clock64 PLL jitter (trimodal: 4/8/12 cycles)
//   2. L1 cache thermal noise (controllable 0.1%→82%)
//   3. Cross-PLL drift (shader vs wall clock divergence)
//   4. Memory contention noise (per-wavefront, shaped)
//
// The reservoir runs ENTIRELY inside the GPU kernel.
// No host roundtrips. No firmware. No hwmon. Pure silicon.
//
// Architecture:
//   - N "neurons" = wavefront lanes, each reading noise sources
//   - Recurrent connections via LDS (shared memory)
//   - Input injected by modulating memory access pattern
//   - State = accumulated noise-influenced timing
//   - Readout = linear classifier on final state vector
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_silicon_reservoir scripts/jitter_silicon_reservoir.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_silicon_reservoir

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <numeric>

#define N_NEURONS 64       // reservoir neurons = lanes in wavefront
#define STEPS 2000         // reservoir timesteps per trial
#define N_CLASSES 4        // classification targets
#define N_TRIALS 80        // trials per class for training
#define N_TEST 40          // trials per class for testing
#define NOISE_SOURCES 4    // number of noise channels per neuron

// ============================================================================
// RESERVOIR KERNEL
// Each lane is a "neuron". At each timestep:
//   1. Read noise from multiple sources (wall_clock, L1, cross-PLL, contention)
//   2. Mix with recurrent state from neighbors via LDS
//   3. Apply nonlinearity (tanh-like via fixed-point)
//   4. Inject input signal by modulating stride/pattern
//
// Output: state[N_NEURONS * STEPS] — full trajectory for readout
// ============================================================================
__global__ void silicon_reservoir(
    float* state_out,           // [N_NEURONS * STEPS] reservoir states
    uint32_t* scratch,          // scratch memory for contention noise (16MB)
    int input_class,            // which input pattern to inject
    int input_phase,            // phase within the pattern
    int heavy_waves_hint,       // thermal load level (0-8)
    int trial_seed              // seed for reproducibility
) {
    int nid = threadIdx.x;      // neuron ID = lane
    if (nid >= N_NEURONS) return;

    // Shared memory for recurrent coupling
    __shared__ float lds_state[N_NEURONS];
    __shared__ uint32_t lds_noise[N_NEURONS];

    // Initialize state
    float v = 0.0f;             // membrane potential analog
    float h = 0.0f;             // hidden state (slow)
    uint32_t rng = nid * 1103515245 + trial_seed + 12345;

    // Input patterns: different classes modulate different parameters
    // Class 0: constant stride (boring)
    // Class 1: oscillating stride (periodic)
    // Class 2: ramping stride (monotonic)
    // Class 3: burst stride (bursty)
    int base_stride = 1;

    for (int t = 0; t < STEPS; t++) {
        // ─── INPUT INJECTION via stride modulation ───
        int stride;
        switch (input_class) {
            case 0:  // Constant
                stride = 32;
                break;
            case 1:  // Oscillating (period depends on phase)
                stride = 16 + 48 * ((t / (8 + input_phase * 4)) & 1);
                break;
            case 2:  // Ramp
                stride = 4 + (t * (1 + input_phase)) / 32;
                if (stride > 128) stride = 128;
                break;
            case 3:  // Burst
                stride = ((t % (20 + input_phase * 5)) < 5) ? 128 : 4;
                break;
            default:
                stride = 32;
        }

        // ─── NOISE SOURCE 1: wall_clock64 PLL jitter ───
        uint64_t w0 = wall_clock64();
        uint64_t w1 = wall_clock64();
        float pll_noise = (float)(w1 - w0);  // typically 4, sometimes 8 or 12

        // ─── NOISE SOURCE 2: L1 cache read timing ───
        uint32_t addr = (nid * 32 + t * stride) & ((16 * 1024 * 1024 / 4) - 1);
        uint64_t c0 = __clock64();
        uint32_t val = scratch[addr];
        uint64_t c1 = __clock64();
        float l1_noise = (float)(c1 - c0);  // typically 16-17, varies with contention

        // ─── NOISE SOURCE 3: Cross-PLL drift ───
        uint64_t s_now = __clock64();
        uint64_t w_now = wall_clock64();
        // Low bits of drift contain phase noise
        float drift_noise = (float)((s_now - w_now) & 0xFF);

        // ─── NOISE SOURCE 4: Memory contention (stride-dependent) ───
        uint32_t addr2 = (nid * stride * 128 + t * 7919) & ((16 * 1024 * 1024 / 4) - 1);
        uint64_t m0 = __clock64();
        val += scratch[addr2];
        uint64_t m1 = __clock64();
        float mem_noise = (float)(m1 - m0);

        // Prevent dead code elimination
        if (val == 0xDEADDEAD) state_out[0] = 0;

        // ─── RECURRENT COUPLING via LDS ───
        lds_state[nid] = v;
        lds_noise[nid] = (uint32_t)(l1_noise * 1000);
        __syncthreads();

        // Each neuron reads from 3 neighbors (ring topology)
        int left = (nid + N_NEURONS - 1) % N_NEURONS;
        int right = (nid + 1) % N_NEURONS;
        int across = (nid + N_NEURONS/2) % N_NEURONS;

        float neighbor_input = 0.15f * lds_state[left]
                             + 0.15f * lds_state[right]
                             + 0.10f * lds_state[across];

        // ─── STATE UPDATE (leaky integrator with noise) ───
        // Normalize noise sources to ~[-1, 1] range
        float n1 = (pll_noise - 4.0f) / 4.0f;       // PLL: [-0, 2] typically
        float n2 = (l1_noise - 16.0f) / 4.0f;        // L1: [-0, +large]
        float n3 = (drift_noise - 128.0f) / 64.0f;   // drift: [-2, 2]
        float n4 = (mem_noise - 40.0f) / 20.0f;       // mem: varies with stride

        // Weighted noise mix (different neurons weight differently)
        float noise_weights[4];
        rng = rng * 1103515245 + 12345;
        noise_weights[0] = 0.2f + 0.3f * ((rng >> 16) & 0xFF) / 255.0f;
        rng = rng * 1103515245 + 12345;
        noise_weights[1] = 0.2f + 0.3f * ((rng >> 16) & 0xFF) / 255.0f;
        rng = rng * 1103515245 + 12345;
        noise_weights[2] = 0.1f + 0.2f * ((rng >> 16) & 0xFF) / 255.0f;
        rng = rng * 1103515245 + 12345;
        noise_weights[3] = 0.2f + 0.3f * ((rng >> 16) & 0xFF) / 255.0f;

        float noise_total = noise_weights[0] * n1 + noise_weights[1] * n2
                          + noise_weights[2] * n3 + noise_weights[3] * n4;

        // Leak + input + noise + recurrent
        float leak = 0.05f;
        v = (1.0f - leak) * v + 0.3f * noise_total + neighbor_input;

        // Soft clamp (tanh approximation)
        if (v > 3.0f) v = 3.0f;
        if (v < -3.0f) v = -3.0f;
        float v_tanh = v * (27.0f + v * v) / (27.0f + 9.0f * v * v);  // fast tanh approx

        // Slow hidden state (temporal memory)
        h = 0.98f * h + 0.02f * v_tanh;

        // Store state for readout (mix fast + slow)
        state_out[nid * STEPS + t] = 0.7f * v_tanh + 0.3f * h;

        __syncthreads();
    }
}

// ============================================================================
// SIMPLER RESERVOIR: Just raw noise, no dynamics
// Baseline comparison — does noise alone carry information?
// ============================================================================
__global__ void raw_noise_reservoir(
    float* state_out,
    uint32_t* scratch,
    int input_class,
    int input_phase,
    int trial_seed
) {
    int nid = threadIdx.x;
    if (nid >= N_NEURONS) return;

    for (int t = 0; t < STEPS; t++) {
        // Same stride modulation as above
        int stride;
        switch (input_class) {
            case 0: stride = 32; break;
            case 1: stride = 16 + 48 * ((t / (8 + input_phase * 4)) & 1); break;
            case 2: stride = 4 + (t * (1 + input_phase)) / 32; if (stride > 128) stride = 128; break;
            case 3: stride = ((t % (20 + input_phase * 5)) < 5) ? 128 : 4; break;
            default: stride = 32;
        }

        // Just read all noise and output directly — no recurrence, no nonlinearity
        uint64_t w0 = wall_clock64();
        uint64_t w1 = wall_clock64();
        float pll = (float)(w1 - w0);

        uint32_t addr = (nid * stride * 128 + t * 7919) & ((16 * 1024 * 1024 / 4) - 1);
        uint64_t c0 = __clock64();
        uint32_t val = scratch[addr];
        uint64_t c1 = __clock64();
        float l1 = (float)(c1 - c0);

        if (val == 0xDEADDEAD) state_out[0] = 0;

        state_out[nid * STEPS + t] = pll + l1;
    }
}

// ============================================================================
// MULTI-CU RESERVOIR: Multiple blocks = multiple CUs, richer dynamics
// ============================================================================
__global__ void multi_cu_reservoir(
    float* state_out,           // [N_BLOCKS * N_NEURONS * STEPS]
    uint32_t* scratch,
    int input_class,
    int input_phase,
    int trial_seed
) {
    int cu_id = blockIdx.x;
    int nid = threadIdx.x;
    if (nid >= N_NEURONS) return;

    __shared__ float lds_state[N_NEURONS];

    float v = 0.0f;
    float h = 0.0f;
    uint32_t rng = nid * 1103515245 + cu_id * 999983 + trial_seed;

    for (int t = 0; t < STEPS; t++) {
        int stride;
        switch (input_class) {
            case 0: stride = 32; break;
            case 1: stride = 16 + 48 * ((t / (8 + input_phase * 4)) & 1); break;
            case 2: stride = 4 + (t * (1 + input_phase)) / 32; if (stride > 128) stride = 128; break;
            case 3: stride = ((t % (20 + input_phase * 5)) < 5) ? 128 : 4; break;
            default: stride = 32;
        }

        // Noise sources — EACH CU gets slightly different noise due to
        // different cache pressure and scheduling
        uint64_t w0 = wall_clock64();
        uint64_t w1 = wall_clock64();
        float n_pll = (float)(w1 - w0) - 4.0f;

        // Different CUs access different memory regions for decorrelation
        uint32_t addr = ((cu_id * N_NEURONS + nid) * stride * 64 + t * 7919)
                        & ((16 * 1024 * 1024 / 4) - 1);
        uint64_t c0 = __clock64();
        uint32_t val = scratch[addr];
        uint64_t c1 = __clock64();
        float n_l1 = (float)(c1 - c0) - 16.0f;

        uint64_t s = __clock64();
        uint64_t w = wall_clock64();
        float n_drift = (float)((s - w) & 0xFF) / 64.0f - 2.0f;

        if (val == 0xDEADDEAD) state_out[0] = 0;

        // Recurrent
        lds_state[nid] = v;
        __syncthreads();

        int left = (nid + N_NEURONS - 1) % N_NEURONS;
        int right = (nid + 1) % N_NEURONS;
        float nb = 0.2f * lds_state[left] + 0.2f * lds_state[right];

        // Heterogeneous weights per CU
        rng = rng * 1103515245 + 12345;
        float w1f = 0.1f + 0.4f * ((rng >> 16) & 0xFF) / 255.0f;
        rng = rng * 1103515245 + 12345;
        float w2f = 0.1f + 0.4f * ((rng >> 16) & 0xFF) / 255.0f;
        rng = rng * 1103515245 + 12345;
        float w3f = 0.05f + 0.15f * ((rng >> 16) & 0xFF) / 255.0f;

        float noise = w1f * n_pll / 4.0f + w2f * n_l1 / 4.0f + w3f * n_drift;

        v = 0.92f * v + 0.3f * noise + nb;
        if (v > 3.0f) v = 3.0f;
        if (v < -3.0f) v = -3.0f;
        float v_tanh = v * (27.0f + v * v) / (27.0f + 9.0f * v * v);

        h = 0.97f * h + 0.03f * v_tanh;

        int global_nid = cu_id * N_NEURONS + nid;
        state_out[global_nid * STEPS + t] = 0.6f * v_tanh + 0.4f * h;

        __syncthreads();
    }
}

// ============================================================================
// HOST: Ridge regression readout + experiment loop
// ============================================================================

// Simple ridge regression: W = (X^T X + lambda I)^{-1} X^T Y
// X: [n_samples, n_features], Y: [n_samples, n_classes] (one-hot)
// Returns accuracy on test set
struct Dataset {
    std::vector<std::vector<float>> X;  // features
    std::vector<int> y;                  // labels
};

float ridge_classify(Dataset& train, Dataset& test, float lambda = 1.0f) {
    int n = train.X.size();
    int d = train.X[0].size();
    int c = N_CLASSES;

    // X^T X + lambda I
    std::vector<double> XtX(d * d, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            for (int k = 0; k <= j; k++) {
                XtX[j * d + k] += train.X[i][j] * train.X[i][k];
            }
        }
    }
    for (int j = 0; j < d; j++) {
        for (int k = 0; k < j; k++) XtX[k * d + j] = XtX[j * d + k];
        XtX[j * d + j] += lambda;
    }

    // X^T Y (one-hot)
    std::vector<double> XtY(d * c, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            XtY[j * c + train.y[i]] += train.X[i][j];
        }
    }

    // Solve via Cholesky (simple, d is small)
    // L L^T = XtX
    std::vector<double> L(d * d, 0);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = XtX[i * d + j];
            for (int k = 0; k < j; k++) sum -= L[i * d + k] * L[j * d + k];
            if (i == j) {
                L[i * d + j] = sqrt(sum > 0 ? sum : 1e-10);
            } else {
                L[i * d + j] = sum / (L[j * d + j] > 0 ? L[j * d + j] : 1e-10);
            }
        }
    }

    // Solve L Z = XtY, then L^T W = Z
    std::vector<double> W(d * c, 0);
    for (int cls = 0; cls < c; cls++) {
        // Forward: L z = XtY[:,cls]
        std::vector<double> z(d);
        for (int i = 0; i < d; i++) {
            double sum = XtY[i * c + cls];
            for (int k = 0; k < i; k++) sum -= L[i * d + k] * z[k];
            z[i] = sum / (L[i * d + i] > 0 ? L[i * d + i] : 1e-10);
        }
        // Backward: L^T w = z
        for (int i = d - 1; i >= 0; i--) {
            double sum = z[i];
            for (int k = i + 1; k < d; k++) sum -= L[k * d + i] * W[k * c + cls];
            W[i * c + cls] = sum / (L[i * d + i] > 0 ? L[i * d + i] : 1e-10);
        }
    }

    // Predict on test
    int correct = 0;
    for (int i = 0; i < (int)test.X.size(); i++) {
        float best = -1e30;
        int pred = 0;
        for (int cls = 0; cls < c; cls++) {
            float score = 0;
            for (int j = 0; j < d; j++) score += test.X[i][j] * W[j * c + cls];
            if (score > best) { best = score; pred = cls; }
        }
        if (pred == test.y[i]) correct++;
    }
    return (float)correct / test.X.size();
}

// Extract features from reservoir state trajectory
std::vector<float> extract_features(float* state, int n_neurons, int steps) {
    // For each neuron: mean, std, min, max of last 500 steps
    // + mean of first/middle/last third (temporal structure)
    int feat_per_neuron = 7;
    std::vector<float> feat(n_neurons * feat_per_neuron);

    int start = steps > 500 ? steps - 500 : 0;
    int third = (steps - start) / 3;

    for (int n = 0; n < n_neurons; n++) {
        float sum = 0, sum2 = 0;
        float mn = 1e30, mx = -1e30;
        float t1 = 0, t2 = 0, t3 = 0;
        int cnt = 0;

        for (int t = start; t < steps; t++) {
            float v = state[n * steps + t];
            sum += v;
            sum2 += v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            cnt++;

            int rel = t - start;
            if (rel < third) t1 += v;
            else if (rel < 2 * third) t2 += v;
            else t3 += v;
        }

        float mean = sum / cnt;
        float std = sqrtf(sum2 / cnt - mean * mean);

        feat[n * feat_per_neuron + 0] = mean;
        feat[n * feat_per_neuron + 1] = std;
        feat[n * feat_per_neuron + 2] = mn;
        feat[n * feat_per_neuron + 3] = mx;
        feat[n * feat_per_neuron + 4] = t1 / third;
        feat[n * feat_per_neuron + 5] = t2 / third;
        feat[n * feat_per_neuron + 6] = t3 / (steps - start - 2 * third);
    }
    return feat;
}

int main() {
    fprintf(stderr, "═══ z2254: Silicon-Level GPU Reservoir Computing ═══\n");
    fprintf(stderr, "  %d neurons, %d steps, %d classes\n", N_NEURONS, STEPS, N_CLASSES);
    fprintf(stderr, "  Noise: wall_clock PLL + L1 cache + cross-PLL drift + memory contention\n");
    fprintf(stderr, "  All computation ON GPU — no host roundtrips\n\n");

    // Allocate
    size_t scratch_size = 16 * 1024 * 1024;  // 16MB for memory noise
    uint32_t *d_scratch;
    hipMalloc(&d_scratch, scratch_size);
    hipMemset(d_scratch, 0xAB, scratch_size);

    float *d_state;
    hipMalloc(&d_state, N_NEURONS * STEPS * sizeof(float));

    float* h_state = new float[N_NEURONS * STEPS];

    int total_trials = N_TRIALS + N_TEST;
    int feat_dim = N_NEURONS * 7;

    // ═══ EXPERIMENT 1: Single-CU Reservoir Classification ═══
    fprintf(stderr, "═══ EXP 1: Single-CU Reservoir (64 neurons, 4 classes) ═══\n");
    {
        Dataset train, test;

        for (int cls = 0; cls < N_CLASSES; cls++) {
            for (int trial = 0; trial < total_trials; trial++) {
                hipMemset(d_state, 0, N_NEURONS * STEPS * sizeof(float));

                int phase = trial % 4;
                silicon_reservoir<<<1, N_NEURONS>>>(
                    d_state, d_scratch, cls, phase, 0, trial * 1000 + cls);
                hipDeviceSynchronize();

                hipMemcpy(h_state, d_state, N_NEURONS * STEPS * sizeof(float),
                          hipMemcpyDeviceToHost);

                auto feat = extract_features(h_state, N_NEURONS, STEPS);

                if (trial < N_TRIALS) {
                    train.X.push_back(feat);
                    train.y.push_back(cls);
                } else {
                    test.X.push_back(feat);
                    test.y.push_back(cls);
                }

                if (trial % 20 == 0)
                    fprintf(stderr, "  cls=%d trial=%d/%d\r", cls, trial, total_trials);
            }
        }
        fprintf(stderr, "  Train: %d, Test: %d, Features: %d    \n",
                (int)train.X.size(), (int)test.X.size(), feat_dim);

        float acc = ridge_classify(train, test);
        fprintf(stderr, "  ★ Single-CU accuracy: %.1f%% (chance=%.1f%%)\n",
                100 * acc, 100.0f / N_CLASSES);
        fprintf(stderr, "  PASS: %s\n\n", acc > 1.0f / N_CLASSES + 0.05f ? "YES" : "NO");
    }

    // ═══ EXPERIMENT 2: Raw Noise Baseline (no recurrence) ═══
    fprintf(stderr, "═══ EXP 2: Raw Noise Baseline (no dynamics) ═══\n");
    {
        Dataset train, test;

        for (int cls = 0; cls < N_CLASSES; cls++) {
            for (int trial = 0; trial < total_trials; trial++) {
                hipMemset(d_state, 0, N_NEURONS * STEPS * sizeof(float));

                raw_noise_reservoir<<<1, N_NEURONS>>>(
                    d_state, d_scratch, cls, trial % 4, trial * 1000 + cls);
                hipDeviceSynchronize();

                hipMemcpy(h_state, d_state, N_NEURONS * STEPS * sizeof(float),
                          hipMemcpyDeviceToHost);

                auto feat = extract_features(h_state, N_NEURONS, STEPS);

                if (trial < N_TRIALS) {
                    train.X.push_back(feat);
                    train.y.push_back(cls);
                } else {
                    test.X.push_back(feat);
                    test.y.push_back(cls);
                }
            }
        }

        float acc = ridge_classify(train, test);
        fprintf(stderr, "  ★ Raw noise accuracy: %.1f%% (chance=%.1f%%)\n",
                100 * acc, 100.0f / N_CLASSES);
        fprintf(stderr, "  PASS: %s\n\n", acc > 1.0f / N_CLASSES + 0.05f ? "YES" : "NO");
    }

    // ═══ EXPERIMENT 3: Multi-CU Reservoir (8 CUs = 512 neurons) ═══
    fprintf(stderr, "═══ EXP 3: Multi-CU Reservoir (8 CUs × 64 = 512 neurons) ═══\n");
    {
        int n_cus = 8;
        int total_neurons = n_cus * N_NEURONS;
        int multi_feat_dim = total_neurons * 7;

        float *d_multi_state;
        hipMalloc(&d_multi_state, total_neurons * STEPS * sizeof(float));
        float* h_multi = new float[total_neurons * STEPS];

        Dataset train, test;

        for (int cls = 0; cls < N_CLASSES; cls++) {
            for (int trial = 0; trial < total_trials; trial++) {
                hipMemset(d_multi_state, 0, total_neurons * STEPS * sizeof(float));

                multi_cu_reservoir<<<n_cus, N_NEURONS>>>(
                    d_multi_state, d_scratch, cls, trial % 4, trial * 1000 + cls);
                hipDeviceSynchronize();

                hipMemcpy(h_multi, d_multi_state, total_neurons * STEPS * sizeof(float),
                          hipMemcpyDeviceToHost);

                auto feat = extract_features(h_multi, total_neurons, STEPS);

                if (trial < N_TRIALS) {
                    train.X.push_back(feat);
                    train.y.push_back(cls);
                } else {
                    test.X.push_back(feat);
                    test.y.push_back(cls);
                }

                if (trial % 20 == 0)
                    fprintf(stderr, "  cls=%d trial=%d/%d\r", cls, trial, total_trials);
            }
        }
        fprintf(stderr, "  Train: %d, Test: %d, Features: %d     \n",
                (int)train.X.size(), (int)test.X.size(), multi_feat_dim);

        float acc = ridge_classify(train, test);
        fprintf(stderr, "  ★ Multi-CU accuracy: %.1f%% (chance=%.1f%%)\n",
                100 * acc, 100.0f / N_CLASSES);
        fprintf(stderr, "  PASS: %s\n\n", acc > 1.0f / N_CLASSES + 0.05f ? "YES" : "NO");

        delete[] h_multi;
        hipFree(d_multi_state);
    }

    // ═══ EXPERIMENT 4: Memory Capacity (can reservoir remember past inputs?) ═══
    fprintf(stderr, "═══ EXP 4: Memory Capacity ═══\n");
    {
        // Generate random input sequence, try to reconstruct past inputs from state
        int seq_len = 200;
        int n_reps = 5;
        float total_mc = 0;

        for (int delay = 0; delay <= 8; delay++) {
            float delay_acc_sum = 0;

            for (int rep = 0; rep < n_reps; rep++) {
                // Random sequence of classes
                std::vector<int> sequence(seq_len);
                uint32_t rng = rep * 99991 + delay * 7;
                for (int i = 0; i < seq_len; i++) {
                    rng = rng * 1103515245 + 12345;
                    sequence[i] = (rng >> 16) % N_CLASSES;
                }

                Dataset train, test;
                int split = seq_len * 2 / 3;

                for (int i = delay; i < seq_len; i++) {
                    hipMemset(d_state, 0, N_NEURONS * STEPS * sizeof(float));
                    silicon_reservoir<<<1, N_NEURONS>>>(
                        d_state, d_scratch, sequence[i], i % 4, 0, i * 1000 + rep);
                    hipDeviceSynchronize();

                    hipMemcpy(h_state, d_state, N_NEURONS * STEPS * sizeof(float),
                              hipMemcpyDeviceToHost);
                    auto feat = extract_features(h_state, N_NEURONS, STEPS);

                    int target = sequence[i - delay];
                    if (i < split) {
                        train.X.push_back(feat);
                        train.y.push_back(target);
                    } else {
                        test.X.push_back(feat);
                        test.y.push_back(target);
                    }
                }

                if (!test.X.empty() && !train.X.empty()) {
                    float acc = ridge_classify(train, test);
                    delay_acc_sum += acc;
                }
            }

            float mean_acc = delay_acc_sum / n_reps;
            float mc_d = (mean_acc - 1.0f / N_CLASSES) / (1.0f - 1.0f / N_CLASSES);
            if (mc_d < 0) mc_d = 0;
            total_mc += mc_d;

            fprintf(stderr, "  delay=%d: acc=%.1f%% MC_d=%.3f %s\n",
                    delay, 100 * mean_acc, mc_d,
                    mean_acc > 1.0f / N_CLASSES + 0.05f ? "█" : "░");
        }
        fprintf(stderr, "  Total MC: %.3f\n\n", total_mc);
    }

    // ═══ EXPERIMENT 5: Temporal XOR ═══
    fprintf(stderr, "═══ EXP 5: Temporal XOR ═══\n");
    {
        int seq_len = 200;
        int n_reps = 5;

        for (int tau = 1; tau <= 3; tau++) {
            float acc_sum = 0;

            for (int rep = 0; rep < n_reps; rep++) {
                std::vector<int> sequence(seq_len);
                uint32_t rng = rep * 77773 + tau * 13;
                for (int i = 0; i < seq_len; i++) {
                    rng = rng * 1103515245 + 12345;
                    sequence[i] = (rng >> 16) % 2;  // binary
                }

                Dataset train, test;
                int split = seq_len * 2 / 3;

                for (int i = tau; i < seq_len; i++) {
                    hipMemset(d_state, 0, N_NEURONS * STEPS * sizeof(float));
                    // Map binary to classes 0/1 with different phases for 0/1
                    silicon_reservoir<<<1, N_NEURONS>>>(
                        d_state, d_scratch, sequence[i] * 2, i % 4, 0, i * 1000 + rep);
                    hipDeviceSynchronize();

                    hipMemcpy(h_state, d_state, N_NEURONS * STEPS * sizeof(float),
                              hipMemcpyDeviceToHost);
                    auto feat = extract_features(h_state, N_NEURONS, STEPS);

                    int target = sequence[i] ^ sequence[i - tau];
                    if (i < split) {
                        train.X.push_back(feat);
                        train.y.push_back(target);
                    } else {
                        test.X.push_back(feat);
                        test.y.push_back(target);
                    }
                }

                // Binary classification via ridge (2 classes)
                if (!test.X.empty() && !train.X.empty()) {
                    // Simple: predict class 0 or 1
                    int correct = 0;
                    // Use ridge with N_CLASSES=2
                    int n = train.X.size();
                    int d = train.X[0].size();
                    float lambda = 1.0f;

                    // X^T X + lambda I
                    std::vector<double> XtX(d * d, 0);
                    for (int i2 = 0; i2 < n; i2++)
                        for (int j = 0; j < d; j++)
                            for (int k = 0; k <= j; k++)
                                XtX[j * d + k] += train.X[i2][j] * train.X[i2][k];
                    for (int j = 0; j < d; j++) {
                        for (int k = 0; k < j; k++) XtX[k * d + j] = XtX[j * d + k];
                        XtX[j * d + j] += lambda;
                    }

                    // X^T y (scalar target)
                    std::vector<double> Xty(d, 0);
                    for (int i2 = 0; i2 < n; i2++)
                        for (int j = 0; j < d; j++)
                            Xty[j] += train.X[i2][j] * train.y[i2];

                    // Cholesky solve
                    std::vector<double> L(d * d, 0);
                    for (int i2 = 0; i2 < d; i2++) {
                        for (int j = 0; j <= i2; j++) {
                            double sum = XtX[i2 * d + j];
                            for (int k = 0; k < j; k++) sum -= L[i2 * d + k] * L[j * d + k];
                            if (i2 == j) L[i2 * d + j] = sqrt(sum > 0 ? sum : 1e-10);
                            else L[i2 * d + j] = sum / (L[j * d + j] > 0 ? L[j * d + j] : 1e-10);
                        }
                    }
                    std::vector<double> w(d);
                    std::vector<double> z(d);
                    for (int i2 = 0; i2 < d; i2++) {
                        double sum = Xty[i2];
                        for (int k = 0; k < i2; k++) sum -= L[i2 * d + k] * z[k];
                        z[i2] = sum / (L[i2 * d + i2] > 0 ? L[i2 * d + i2] : 1e-10);
                    }
                    for (int i2 = d - 1; i2 >= 0; i2--) {
                        double sum = z[i2];
                        for (int k = i2 + 1; k < d; k++) sum -= L[k * d + i2] * w[k];
                        w[i2] = sum / (L[i2 * d + i2] > 0 ? L[i2 * d + i2] : 1e-10);
                    }

                    for (int i2 = 0; i2 < (int)test.X.size(); i2++) {
                        double score = 0;
                        for (int j = 0; j < d; j++) score += test.X[i2][j] * w[j];
                        int pred = score > 0.5 ? 1 : 0;
                        if (pred == test.y[i2]) correct++;
                    }
                    acc_sum += (float)correct / test.X.size();
                }
            }

            float mean_acc = acc_sum / n_reps;
            fprintf(stderr, "  XOR(t, t-%d): %.1f%% (chance=50%%) %s\n",
                    tau, 100 * mean_acc,
                    mean_acc > 0.55f ? "█ PASS" : "░ FAIL");
        }
    }

    // ═══ EXPERIMENT 6: Reproducibility (Echo State) ═══
    fprintf(stderr, "\n═══ EXP 6: Reproducibility ═══\n");
    {
        // Same input, same seed → how similar are outputs?
        for (int cls = 0; cls < N_CLASSES; cls++) {
            float corr_sum = 0;
            int pairs = 0;
            std::vector<std::vector<float>> feats;

            for (int rep = 0; rep < 5; rep++) {
                hipMemset(d_state, 0, N_NEURONS * STEPS * sizeof(float));
                silicon_reservoir<<<1, N_NEURONS>>>(
                    d_state, d_scratch, cls, 0, 0, 42);  // same seed!
                hipDeviceSynchronize();

                hipMemcpy(h_state, d_state, N_NEURONS * STEPS * sizeof(float),
                          hipMemcpyDeviceToHost);
                feats.push_back(extract_features(h_state, N_NEURONS, STEPS));
            }

            // Pairwise correlation
            for (int a = 0; a < (int)feats.size(); a++) {
                for (int b = a + 1; b < (int)feats.size(); b++) {
                    double mx = 0, my = 0;
                    int d = feats[a].size();
                    for (int i = 0; i < d; i++) { mx += feats[a][i]; my += feats[b][i]; }
                    mx /= d; my /= d;

                    double num = 0, dx = 0, dy = 0;
                    for (int i = 0; i < d; i++) {
                        num += (feats[a][i] - mx) * (feats[b][i] - my);
                        dx += (feats[a][i] - mx) * (feats[a][i] - mx);
                        dy += (feats[b][i] - my) * (feats[b][i] - my);
                    }
                    float r = (dx > 0 && dy > 0) ? num / sqrt(dx * dy) : 0;
                    corr_sum += r;
                    pairs++;
                }
            }

            fprintf(stderr, "  Class %d: reproducibility r=%.3f\n",
                    cls, corr_sum / pairs);
        }
    }

    fprintf(stderr, "\nDone.\n");

    delete[] h_state;
    hipFree(d_scratch);
    hipFree(d_state);
    return 0;
}
