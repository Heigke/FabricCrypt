/**
 * cross_kernel_memory.hip — Exploit REAL cross-kernel temporal dependencies
 *
 * KEY INSIGHT: GPU hardware state persists between kernel launches:
 *   1. L2 cache contents (no auto-flush between dispatches)
 *   2. DVFS frequency state (clock stretching from voltage droop)
 *   3. Voltage regulator transients (30-50ms settling time)
 *   4. PLL thermal drift (1/f character, ACF=0.989 measured in z2160)
 *   5. Dispatch queue state (ACF=0.964 measured in z2225)
 *
 * Instead of one big kernel with internal "recurrence", we launch MANY
 * small kernels and let the HARDWARE carry temporal memory between them.
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", e, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ── Architecture constants ──
static constexpr int N_NEURONS  = 128;    // threads per kernel
static constexpr int N_STEPS    = 200;    // kernel launches per trial
static constexpr int N_PROBES   = 8;      // timing probes per neuron per kernel
static constexpr int WARP_SIZE  = 64;
static constexpr int N_TRIALS   = 60;     // samples per class
static constexpr int N_CLASSES  = 4;      // waveform classes

// ── Probe channels ──
// Each kernel launch measures 8 temporal probes per neuron:
//   0: first_clock     — wall_clock64 at kernel entry (PLL state)
//   1: dispatch_delta  — entry_clock - previous_kernel_exit_clock
//   2: l2_probe_hit    — timing of read from L2-warm address
//   3: l2_probe_cold   — timing of read from L2-cold address
//   4: fma_timing      — FMA pipeline timing (voltage-dependent)
//   5: sfu_timing      — SFU pipeline timing (power-dependent)
//   6: lds_timing      — LDS bank conflict timing (occupancy-dependent)
//   7: exit_clock      — wall_clock64 at kernel exit

// ── Per-neuron persistent state (survives across kernel launches via global memory) ──
struct NeuronState {
    float activation;      // previous kernel's output
    float running_mean;    // EMA of activation
    float running_var;     // EMA of variance
    long long last_exit;   // exit clock of previous kernel
};

// ── Small kernel: one step of the reservoir ──
// Called N_STEPS times per trial. Hardware carries temporal memory between calls.
__global__ void k_cross_kernel_step(
    const float* __restrict__ signal,   // [N_NEURONS] input for this step
    float* __restrict__ probes,         // [N_NEURONS * N_PROBES] output probes
    NeuronState* __restrict__ state,    // [N_NEURONS] persistent state
    float* __restrict__ l2_warm,        // [N_NEURONS * 256] L2 cache probe buffer
    volatile int* __restrict__ atomics, // [4] atomic counters for contention
    int step,
    float recurrence_strength
) {
    int nid = threadIdx.x;  // neuron ID
    if (nid >= N_NEURONS) return;

    // ── Probe 0: Entry clock (captures PLL state from previous kernel's thermal load) ──
    long long entry_clock = wall_clock64();

    // ── Probe 1: Dispatch delta (time since previous kernel exited) ──
    long long prev_exit = state[nid].last_exit;
    float dispatch_delta = (prev_exit > 0) ? (float)(entry_clock - prev_exit) : 0.0f;

    // ── Probe 2: L2 warm read (address touched by previous kernel → should be cached) ──
    long long t0 = wall_clock64();
    float warm_val = 0.0f;
    int warm_base = nid * 256;
    for (int i = 0; i < 64; i++) {
        warm_val += l2_warm[warm_base + i];  // same addresses as previous step
    }
    long long t1 = wall_clock64();
    float l2_hit_time = (float)(t1 - t0);

    // ── Probe 3: L2 cold read (address NOT touched recently → cache miss expected) ──
    // Use a different offset each step to ensure cold access
    long long t2 = wall_clock64();
    float cold_val = 0.0f;
    int cold_base = ((nid + 64 + step * 7) % N_NEURONS) * 256 + 128;
    for (int i = 0; i < 64; i++) {
        cold_val += l2_warm[cold_base + (i * 3) % 128];  // strided, cold
    }
    long long t3 = wall_clock64();
    float l2_cold_time = (float)(t3 - t2);

    // ── Build workload from input + recurrence (hardware-modulated) ──
    float inp = signal[nid];
    float prev_act = state[nid].activation;
    float work_modifier = inp + recurrence_strength * prev_act;

    // Clamp to [0.5, 2.0] to keep workload bounded
    work_modifier = fmaxf(0.5f, fminf(2.0f, fabsf(work_modifier) + 0.5f));
    int work_iters = (int)(40.0f * work_modifier);

    // ── Probe 4: FMA timing (voltage/frequency dependent) ──
    long long t4 = wall_clock64();
    float fma_acc = 1.0001f;
    for (int i = 0; i < work_iters; i++) {
        fma_acc = fma_acc * 1.0001f + 0.0001f;
    }
    long long t5 = wall_clock64();
    float fma_time = (float)(t5 - t4);

    // ── Probe 5: SFU timing (different power draw → different droop) ──
    long long t6 = wall_clock64();
    float sfu_acc = 0.001f + (float)nid * 0.001f;
    int sfu_iters = work_iters / 2 + 10;
    for (int i = 0; i < sfu_iters; i++) {
        sfu_acc = __sinf(sfu_acc + 0.01f * (float)i);
    }
    long long t7 = wall_clock64();
    float sfu_time = (float)(t7 - t6);

    // ── Probe 6: LDS timing (occupancy + bank conflicts) ──
    __shared__ float lds_buf[256];
    long long t8 = wall_clock64();
    lds_buf[nid] = fma_acc + sfu_acc;
    __syncthreads();
    float lds_acc = 0.0f;
    // Circular neighbor reads (different bank access patterns)
    for (int k = 0; k < 8; k++) {
        int neighbor = (nid + k * 17) % N_NEURONS;  // coprime stride
        lds_acc += lds_buf[neighbor];
    }
    __syncthreads();
    long long t9 = wall_clock64();
    float lds_time = (float)(t9 - t8);

    // ── Probe 7: Exit clock ──
    // But first: write to L2 warm buffer so NEXT kernel sees cached data
    for (int i = 0; i < 64; i++) {
        l2_warm[warm_base + i] = fma_acc * 0.001f + (float)i;
    }

    // Atomic contention (cross-neuron synchronization artifact)
    atomicAdd((int*)&atomics[nid % 4], 1);

    long long exit_clock = wall_clock64();

    // ── Store probes ──
    int base = nid * N_PROBES;
    probes[base + 0] = (float)(entry_clock & 0xFFFFF);  // low 20 bits (avoid overflow)
    probes[base + 1] = dispatch_delta;
    probes[base + 2] = l2_hit_time;
    probes[base + 3] = l2_cold_time;
    probes[base + 4] = fma_time;
    probes[base + 5] = sfu_time;
    probes[base + 6] = lds_time;
    probes[base + 7] = (float)(exit_clock & 0xFFFFF);

    // ── Update persistent state ──
    float new_act = (fma_time + sfu_time + l2_hit_time) / 1000.0f;  // normalize
    state[nid].activation = new_act;
    float alpha = 0.1f;
    state[nid].running_mean = (1.0f - alpha) * state[nid].running_mean + alpha * new_act;
    float diff = new_act - state[nid].running_mean;
    state[nid].running_var = (1.0f - alpha) * state[nid].running_var + alpha * diff * diff;
    state[nid].last_exit = exit_clock;
}

// ── Software LIF baseline (same per-kernel-launch architecture) ──
__global__ void k_lif_step(
    const float* __restrict__ signal,
    float* __restrict__ probes,
    NeuronState* __restrict__ state,
    int step,
    unsigned int seed_base
) {
    int nid = threadIdx.x;
    if (nid >= N_NEURONS) return;

    unsigned int rng = seed_base + nid * 1337 + step * 7919;

    // xorshift32 noise
    auto xorshift = [](unsigned int& s) -> float {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return (float)(s & 0xFFFF) / 65536.0f - 0.5f;
    };

    float inp = signal[nid];
    float prev_act = state[nid].activation;

    // LIF membrane with PRNG noise
    float leak = 0.9f;
    float vmem = leak * prev_act + inp * 0.3f + xorshift(rng) * 0.1f;
    float threshold = 1.0f;
    float spike = (vmem > threshold) ? 1.0f : 0.0f;
    if (spike > 0.5f) vmem = 0.0f;

    // Generate 8 "probes" from software state
    probes[nid * N_PROBES + 0] = vmem;
    probes[nid * N_PROBES + 1] = spike;
    probes[nid * N_PROBES + 2] = xorshift(rng);
    probes[nid * N_PROBES + 3] = xorshift(rng);
    probes[nid * N_PROBES + 4] = fabsf(vmem - prev_act);  // delta
    probes[nid * N_PROBES + 5] = state[nid].running_mean;
    probes[nid * N_PROBES + 6] = state[nid].running_var;
    probes[nid * N_PROBES + 7] = (float)step * 0.01f;

    state[nid].activation = vmem;
    float alpha = 0.1f;
    state[nid].running_mean = (1.0f - alpha) * state[nid].running_mean + alpha * vmem;
    float diff = vmem - state[nid].running_mean;
    state[nid].running_var = (1.0f - alpha) * state[nid].running_var + alpha * diff * diff;
}

// ── Tanh ESN baseline ──
__global__ void k_esn_step(
    const float* __restrict__ signal,
    float* __restrict__ probes,
    NeuronState* __restrict__ state,
    int step,
    unsigned int seed_base
) {
    int nid = threadIdx.x;
    if (nid >= N_NEURONS) return;

    unsigned int rng = seed_base + nid * 2137 + step * 6581;
    auto xorshift = [](unsigned int& s) -> float {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return (float)(s & 0xFFFF) / 65536.0f - 0.5f;
    };

    float inp = signal[nid];
    float prev = state[nid].activation;

    // ESN: tanh(Win*input + Wrec*prev + noise)
    float Win = 0.3f;
    float Wrec = 0.9f;
    float noise_scale = 0.05f;
    float x = tanhf(Win * inp + Wrec * prev + noise_scale * xorshift(rng));

    probes[nid * N_PROBES + 0] = x;
    probes[nid * N_PROBES + 1] = (x > 0.0f) ? 1.0f : 0.0f;
    probes[nid * N_PROBES + 2] = x - prev;
    probes[nid * N_PROBES + 3] = xorshift(rng);
    probes[nid * N_PROBES + 4] = fabsf(x - prev);
    probes[nid * N_PROBES + 5] = state[nid].running_mean;
    probes[nid * N_PROBES + 6] = state[nid].running_var;
    probes[nid * N_PROBES + 7] = x * x;

    state[nid].activation = x;
    float alpha = 0.1f;
    state[nid].running_mean = (1.0f - alpha) * state[nid].running_mean + alpha * x;
    float diff = x - state[nid].running_mean;
    state[nid].running_var = (1.0f - alpha) * state[nid].running_var + alpha * diff * diff;
}


// ══════════════════════════════════════════════════════════════
//  HOST: Signal generation, feature extraction, classification
// ══════════════════════════════════════════════════════════════

void generate_signal(float* sig, int n_neurons, int step, int n_steps,
                     int waveform_class, float amplitude = 1.0f) {
    float t = (float)step / (float)n_steps;
    for (int i = 0; i < n_neurons; i++) {
        float phase = (float)i / (float)n_neurons * 2.0f * M_PI;
        float val = 0.0f;
        switch (waveform_class) {
            case 0: val = amplitude * sinf(2.0f * M_PI * t * 3.0f + phase); break;
            case 1: val = amplitude * (fmodf(t * 5.0f + phase/(2.0f*M_PI), 1.0f) * 2.0f - 1.0f); break;
            case 2: val = amplitude * ((sinf(2.0f * M_PI * t * 4.0f + phase) > 0) ? 1.0f : -1.0f); break;
            case 3: val = amplitude * sinf(2.0f * M_PI * t * 2.0f + phase) *
                          cosf(2.0f * M_PI * t * 7.0f + phase * 0.5f); break;
        }
        sig[i] = val;
    }
}

struct Features {
    std::vector<float> data;
    int dim;
};

// Extract rich features from probe time series
// probes_all: [N_STEPS][N_NEURONS * N_PROBES]
Features extract_features(const std::vector<std::vector<float>>& probes_all,
                          int n_steps, int n_neurons, int n_probes) {
    // For each neuron × probe: mean, std, min, max, autocorrelation(1),
    // derivative_mean, derivative_std, range, q25, q75
    int feat_per_channel = 10;
    int dim = n_neurons * n_probes * feat_per_channel;
    Features f;
    f.dim = dim;
    f.data.resize(dim, 0.0f);

    for (int n = 0; n < n_neurons; n++) {
        for (int p = 0; p < n_probes; p++) {
            // Collect time series for this neuron-probe
            std::vector<float> ts(n_steps);
            for (int t = 0; t < n_steps; t++) {
                ts[t] = probes_all[t][n * n_probes + p];
            }

            // Basic stats
            float sum = 0, sum2 = 0;
            float mn = ts[0], mx = ts[0];
            for (int t = 0; t < n_steps; t++) {
                sum += ts[t];
                sum2 += ts[t] * ts[t];
                mn = std::min(mn, ts[t]);
                mx = std::max(mx, ts[t]);
            }
            float mean = sum / n_steps;
            float var = sum2 / n_steps - mean * mean;
            float std_dev = sqrtf(fmaxf(var, 1e-12f));

            // Autocorrelation lag-1
            float acf = 0.0f, acf_denom = 0.0f;
            for (int t = 0; t < n_steps - 1; t++) {
                acf += (ts[t] - mean) * (ts[t+1] - mean);
                acf_denom += (ts[t] - mean) * (ts[t] - mean);
            }
            acf = (acf_denom > 1e-12f) ? acf / acf_denom : 0.0f;

            // Derivative stats
            float d_sum = 0, d_sum2 = 0;
            for (int t = 1; t < n_steps; t++) {
                float d = ts[t] - ts[t-1];
                d_sum += d;
                d_sum2 += d * d;
            }
            float d_mean = d_sum / (n_steps - 1);
            float d_var = d_sum2 / (n_steps - 1) - d_mean * d_mean;
            float d_std = sqrtf(fmaxf(d_var, 1e-12f));

            // Quartiles (approximate via sorted)
            std::vector<float> sorted_ts = ts;
            std::sort(sorted_ts.begin(), sorted_ts.end());
            float q25 = sorted_ts[n_steps / 4];
            float q75 = sorted_ts[3 * n_steps / 4];

            // Store features
            int base = (n * n_probes + p) * feat_per_channel;
            f.data[base + 0] = mean;
            f.data[base + 1] = std_dev;
            f.data[base + 2] = mn;
            f.data[base + 3] = mx;
            f.data[base + 4] = acf;
            f.data[base + 5] = d_mean;
            f.data[base + 6] = d_std;
            f.data[base + 7] = mx - mn;
            f.data[base + 8] = q25;
            f.data[base + 9] = q75;
        }
    }
    return f;
}

// Ridge regression classifier
float classify(const std::vector<Features>& all_features,
               const std::vector<int>& labels, int n_classes) {
    int n = all_features.size();
    int d = all_features[0].dim;

    // 3-fold cross-validation
    float total_correct = 0;
    int total_test = 0;

    for (int fold = 0; fold < 3; fold++) {
        std::vector<int> train_idx, test_idx;
        for (int i = 0; i < n; i++) {
            if (i % 3 == fold) test_idx.push_back(i);
            else train_idx.push_back(i);
        }

        int n_train = train_idx.size();
        int n_test = test_idx.size();

        // Build X_train, Y_train (one-hot)
        std::vector<float> X(n_train * d);
        std::vector<float> Y(n_train * n_classes, 0.0f);
        for (int i = 0; i < n_train; i++) {
            for (int j = 0; j < d; j++) {
                X[i * d + j] = all_features[train_idx[i]].data[j];
            }
            Y[i * n_classes + labels[train_idx[i]]] = 1.0f;
        }

        // Standardize
        std::vector<float> col_mean(d, 0.0f), col_std(d, 1.0f);
        for (int j = 0; j < d; j++) {
            float s = 0, s2 = 0;
            for (int i = 0; i < n_train; i++) {
                s += X[i * d + j];
                s2 += X[i * d + j] * X[i * d + j];
            }
            col_mean[j] = s / n_train;
            float v = s2 / n_train - col_mean[j] * col_mean[j];
            col_std[j] = (v > 1e-10f) ? sqrtf(v) : 1.0f;
        }
        for (int i = 0; i < n_train; i++)
            for (int j = 0; j < d; j++)
                X[i * d + j] = (X[i * d + j] - col_mean[j]) / col_std[j];

        // XtX + lambda*I
        float lambda = 1.0f;
        std::vector<float> XtX(d * d, 0.0f);
        std::vector<float> XtY(d * n_classes, 0.0f);

        for (int j1 = 0; j1 < d; j1++) {
            for (int j2 = j1; j2 < d; j2++) {
                float dot = 0;
                for (int i = 0; i < n_train; i++)
                    dot += X[i * d + j1] * X[i * d + j2];
                XtX[j1 * d + j2] = dot;
                XtX[j2 * d + j1] = dot;
            }
            XtX[j1 * d + j1] += lambda;
        }
        for (int j = 0; j < d; j++)
            for (int c = 0; c < n_classes; c++) {
                float dot = 0;
                for (int i = 0; i < n_train; i++)
                    dot += X[i * d + j] * Y[i * n_classes + c];
                XtY[j * n_classes + c] = dot;
            }

        // Solve via Cholesky
        // L * L^T = XtX
        std::vector<float> L(d * d, 0.0f);
        for (int i = 0; i < d; i++) {
            for (int j = 0; j <= i; j++) {
                float s = XtX[i * d + j];
                for (int k = 0; k < j; k++)
                    s -= L[i * d + k] * L[j * d + k];
                if (i == j) {
                    if (s <= 0) s = 1e-6f;
                    L[i * d + j] = sqrtf(s);
                } else {
                    L[i * d + j] = s / L[j * d + j];
                }
            }
        }

        // Solve L*Z = XtY, then L^T*W = Z
        std::vector<float> W(d * n_classes, 0.0f);
        for (int c = 0; c < n_classes; c++) {
            // Forward: L*z = XtY[:,c]
            std::vector<float> z(d);
            for (int i = 0; i < d; i++) {
                float s = XtY[i * n_classes + c];
                for (int k = 0; k < i; k++)
                    s -= L[i * d + k] * z[k];
                z[i] = s / L[i * d + i];
            }
            // Backward: L^T*w = z
            for (int i = d - 1; i >= 0; i--) {
                float s = z[i];
                for (int k = i + 1; k < d; k++)
                    s -= L[k * d + i] * W[k * n_classes + c];
                W[i * n_classes + c] = s / L[i * d + i];
            }
        }

        // Predict test
        for (int ti = 0; ti < n_test; ti++) {
            int idx = test_idx[ti];
            std::vector<float> scores(n_classes, 0.0f);
            for (int j = 0; j < d; j++) {
                float xval = (all_features[idx].data[j] - col_mean[j]) / col_std[j];
                for (int c = 0; c < n_classes; c++)
                    scores[c] += xval * W[j * n_classes + c];
            }
            int pred = std::max_element(scores.begin(), scores.end()) - scores.begin();
            if (pred == labels[idx]) total_correct++;
            total_test++;
        }
    }

    return total_correct / total_test;
}

// ── XOR task with temporal delay ──
float xor_task(const std::vector<std::vector<float>>& probes_all,
               const std::vector<float>& input_sequence,
               int n_steps, int n_neurons, int n_probes, int delay) {
    // Target: XOR of input sign at t and t-delay
    int start = delay + 10;  // skip warmup
    if (start >= n_steps - 10) return 0.5f;

    int n_samples = n_steps - start;
    int feat_dim = n_neurons * n_probes;

    std::vector<Features> feats(n_samples);
    std::vector<int> labels(n_samples);

    for (int t = 0; t < n_samples; t++) {
        int step = start + t;
        feats[t].dim = feat_dim;
        feats[t].data.resize(feat_dim);
        for (int j = 0; j < feat_dim; j++) {
            feats[t].data[j] = probes_all[step][j];
        }

        int bit_now = (input_sequence[step] > 0.0f) ? 1 : 0;
        int bit_past = (input_sequence[step - delay] > 0.0f) ? 1 : 0;
        labels[t] = bit_now ^ bit_past;
    }

    return classify(feats, labels, 2);
}

// ── Autocorrelation measurement ──
struct ACFResult {
    float acf1;
    float acf2;
    float acf5;
    float mean_acf;
};

ACFResult measure_acf(const std::vector<std::vector<float>>& probes_all,
                      int n_steps, int n_neurons, int n_probes) {
    // Compute ACF at lags 1, 2, 5 averaged over all neuron-probe channels
    ACFResult result = {0, 0, 0, 0};
    int n_channels = n_neurons * n_probes;
    int count = 0;

    for (int ch = 0; ch < n_channels; ch++) {
        std::vector<float> ts(n_steps);
        float mean = 0;
        for (int t = 0; t < n_steps; t++) {
            ts[t] = probes_all[t][ch];
            mean += ts[t];
        }
        mean /= n_steps;

        float var = 0;
        for (int t = 0; t < n_steps; t++)
            var += (ts[t] - mean) * (ts[t] - mean);
        if (var < 1e-12f) continue;

        auto compute_acf = [&](int lag) -> float {
            float cov = 0;
            for (int t = 0; t < n_steps - lag; t++)
                cov += (ts[t] - mean) * (ts[t + lag] - mean);
            return cov / var;
        };

        result.acf1 += compute_acf(1);
        result.acf2 += compute_acf(2);
        result.acf5 += compute_acf(5);
        count++;
    }

    if (count > 0) {
        result.acf1 /= count;
        result.acf2 /= count;
        result.acf5 /= count;
        result.mean_acf = (result.acf1 + result.acf2 + result.acf5) / 3.0f;
    }
    return result;
}


// ══════════════════════════════════════════════════════════════
//  MAIN: Run all experiments
// ══════════════════════════════════════════════════════════════

int main() {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  CROSS-KERNEL TEMPORAL MEMORY — Exploiting Hardware Persistence\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Neurons: %d, Steps(kernel launches): %d, Probes: %d/neuron\n",
           N_NEURONS, N_STEPS, N_PROBES);
    printf("  Key: each 'step' is a SEPARATE kernel launch.\n");
    printf("  Hardware carries temporal memory via L2 cache, DVFS, voltage droop.\n\n");

    // Allocate GPU memory
    float *d_signal, *d_probes, *d_l2_warm;
    NeuronState *d_state;
    int *d_atomics;

    HIP_CHECK(hipMalloc(&d_signal, N_NEURONS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_probes, N_NEURONS * N_PROBES * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_state, N_NEURONS * sizeof(NeuronState)));
    HIP_CHECK(hipMalloc(&d_l2_warm, N_NEURONS * 256 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_atomics, 4 * sizeof(int)));

    std::vector<float> h_signal(N_NEURONS);
    std::vector<float> h_probes(N_NEURONS * N_PROBES);

    // ═══ TEST 1: AUTOCORRELATION ACROSS KERNEL LAUNCHES ═══
    printf("═══ TEST 1: AUTOCORRELATION ACROSS KERNEL LAUNCHES ═══\n");
    printf("  Measuring ACF of 8 probe channels across %d separate kernel launches\n", N_STEPS);
    {
        // Reset state
        HIP_CHECK(hipMemset(d_state, 0, N_NEURONS * sizeof(NeuronState)));
        HIP_CHECK(hipMemset(d_l2_warm, 0, N_NEURONS * 256 * sizeof(float)));
        HIP_CHECK(hipMemset(d_atomics, 0, 4 * sizeof(int)));

        // Run with constant input (isolate hardware temporal effects)
        std::vector<std::vector<float>> probes_neuro(N_STEPS);
        for (int t = 0; t < N_STEPS; t++) {
            // Constant input — any temporal variation is from HARDWARE
            for (int i = 0; i < N_NEURONS; i++) h_signal[i] = 0.5f;
            HIP_CHECK(hipMemcpy(d_signal, h_signal.data(), N_NEURONS * sizeof(float),
                                hipMemcpyHostToDevice));

            k_cross_kernel_step<<<1, N_NEURONS>>>(
                d_signal, d_probes, d_state, d_l2_warm, d_atomics, t, 0.0f);
            HIP_CHECK(hipDeviceSynchronize());

            probes_neuro[t].resize(N_NEURONS * N_PROBES);
            HIP_CHECK(hipMemcpy(probes_neuro[t].data(), d_probes,
                                N_NEURONS * N_PROBES * sizeof(float),
                                hipMemcpyDeviceToHost));
        }

        ACFResult acf = measure_acf(probes_neuro, N_STEPS, N_NEURONS, N_PROBES);
        printf("  NEUROMORPHIC (hardware):  ACF(1)=%.3f  ACF(2)=%.3f  ACF(5)=%.3f\n",
               acf.acf1, acf.acf2, acf.acf5);

        // Software baseline — same test with LIF
        HIP_CHECK(hipMemset(d_state, 0, N_NEURONS * sizeof(NeuronState)));
        std::vector<std::vector<float>> probes_lif(N_STEPS);
        for (int t = 0; t < N_STEPS; t++) {
            for (int i = 0; i < N_NEURONS; i++) h_signal[i] = 0.5f;
            HIP_CHECK(hipMemcpy(d_signal, h_signal.data(), N_NEURONS * sizeof(float),
                                hipMemcpyHostToDevice));

            k_lif_step<<<1, N_NEURONS>>>(d_signal, d_probes, d_state, t, 42);
            HIP_CHECK(hipDeviceSynchronize());

            probes_lif[t].resize(N_NEURONS * N_PROBES);
            HIP_CHECK(hipMemcpy(probes_lif[t].data(), d_probes,
                                N_NEURONS * N_PROBES * sizeof(float),
                                hipMemcpyDeviceToHost));
        }

        ACFResult acf_lif = measure_acf(probes_lif, N_STEPS, N_NEURONS, N_PROBES);
        printf("  LIF (software):           ACF(1)=%.3f  ACF(2)=%.3f  ACF(5)=%.3f\n",
               acf_lif.acf1, acf_lif.acf2, acf_lif.acf5);

        // Per-probe breakdown for neuromorphic
        const char* probe_names[] = {"entry_clk", "dispatch_Δ", "L2_hit", "L2_cold",
                                     "FMA_time", "SFU_time", "LDS_time", "exit_clk"};
        printf("\n  Per-probe ACF(1) [averaged over %d neurons]:\n", N_NEURONS);
        for (int p = 0; p < N_PROBES; p++) {
            float probe_acf = 0;
            int cnt = 0;
            for (int n = 0; n < N_NEURONS; n++) {
                std::vector<float> ts(N_STEPS);
                float mean = 0;
                for (int t = 0; t < N_STEPS; t++) {
                    ts[t] = probes_neuro[t][n * N_PROBES + p];
                    mean += ts[t];
                }
                mean /= N_STEPS;
                float var = 0, cov = 0;
                for (int t = 0; t < N_STEPS; t++) var += (ts[t]-mean)*(ts[t]-mean);
                if (var < 1e-12f) continue;
                for (int t = 0; t < N_STEPS-1; t++)
                    cov += (ts[t]-mean)*(ts[t+1]-mean);
                probe_acf += cov / var;
                cnt++;
            }
            if (cnt > 0) probe_acf /= cnt;
            printf("    %-12s  ACF(1)=%.4f\n", probe_names[p], probe_acf);
        }
    }
    fflush(stdout);

    // ═══ TEST 2: WAVEFORM CLASSIFICATION VIA CROSS-KERNEL DYNAMICS ═══
    printf("\n═══ TEST 2: WAVEFORM CLASSIFICATION (cross-kernel dynamics) ═══\n");
    {
        auto run_classification = [&](const char* name, int method) -> float {
            std::vector<Features> all_feats;
            std::vector<int> all_labels;

            for (int cls = 0; cls < N_CLASSES; cls++) {
                int n_per_class = N_TRIALS / N_CLASSES;
                for (int trial = 0; trial < n_per_class; trial++) {
                    // Reset state
                    HIP_CHECK(hipMemset(d_state, 0, N_NEURONS * sizeof(NeuronState)));
                    HIP_CHECK(hipMemset(d_l2_warm, 0, N_NEURONS * 256 * sizeof(float)));
                    HIP_CHECK(hipMemset(d_atomics, 0, 4 * sizeof(int)));

                    // Run N_STEPS separate kernel launches
                    std::vector<std::vector<float>> probes_ts(N_STEPS);
                    for (int t = 0; t < N_STEPS; t++) {
                        float amp = 0.5f + 0.5f * (float)trial / n_per_class;
                        generate_signal(h_signal.data(), N_NEURONS, t, N_STEPS, cls, amp);
                        HIP_CHECK(hipMemcpy(d_signal, h_signal.data(),
                                            N_NEURONS * sizeof(float),
                                            hipMemcpyHostToDevice));

                        if (method == 0) {
                            k_cross_kernel_step<<<1, N_NEURONS>>>(
                                d_signal, d_probes, d_state, d_l2_warm, d_atomics,
                                t, 0.2f);
                        } else if (method == 1) {
                            k_lif_step<<<1, N_NEURONS>>>(
                                d_signal, d_probes, d_state, t,
                                42 + trial * 1000 + cls * 100000);
                        } else {
                            k_esn_step<<<1, N_NEURONS>>>(
                                d_signal, d_probes, d_state, t,
                                42 + trial * 1000 + cls * 100000);
                        }
                        HIP_CHECK(hipDeviceSynchronize());

                        probes_ts[t].resize(N_NEURONS * N_PROBES);
                        HIP_CHECK(hipMemcpy(probes_ts[t].data(), d_probes,
                                            N_NEURONS * N_PROBES * sizeof(float),
                                            hipMemcpyDeviceToHost));
                    }

                    Features f = extract_features(probes_ts, N_STEPS, N_NEURONS, N_PROBES);
                    all_feats.push_back(f);
                    all_labels.push_back(cls);
                }
            }

            float acc = classify(all_feats, all_labels, N_CLASSES);
            printf("  %-22s  4-class=%.1f%%\n", name, acc * 100.0f);
            return acc;
        };

        float acc_n = run_classification("CROSS_KERNEL_NEURO", 0);
        float acc_l = run_classification("LIF_BASELINE", 1);
        float acc_e = run_classification("ESN_BASELINE", 2);
        printf("  Δ(Neuro-LIF)=%+.1fpp  Δ(Neuro-ESN)=%+.1fpp\n",
               (acc_n - acc_l) * 100.0f, (acc_n - acc_e) * 100.0f);
    }
    fflush(stdout);

    // ═══ TEST 3: XOR TEMPORAL MEMORY (the real test) ═══
    printf("\n═══ TEST 3: XOR TEMPORAL MEMORY (cross-kernel) ═══\n");
    printf("  Each step = separate kernel launch. Memory from HW persistence.\n");
    {
        int delays[] = {1, 2, 3, 5, 10, 15, 20};
        int n_delays = 7;
        int xor_steps = 150;

        printf("  delay  NEURO   LIF     ESN\n");

        for (int di = 0; di < n_delays; di++) {
            int delay = delays[di];

            // Generate random binary input sequence
            std::vector<float> input_seq(xor_steps);
            unsigned int rng_seed = 12345 + delay;
            for (int t = 0; t < xor_steps; t++) {
                rng_seed ^= rng_seed << 13;
                rng_seed ^= rng_seed >> 17;
                rng_seed ^= rng_seed << 5;
                input_seq[t] = ((rng_seed & 1) == 0) ? 1.0f : -1.0f;
            }

            // Run each method
            float results[3] = {0, 0, 0};
            for (int method = 0; method < 3; method++) {
                HIP_CHECK(hipMemset(d_state, 0, N_NEURONS * sizeof(NeuronState)));
                HIP_CHECK(hipMemset(d_l2_warm, 0, N_NEURONS * 256 * sizeof(float)));
                HIP_CHECK(hipMemset(d_atomics, 0, 4 * sizeof(int)));

                std::vector<std::vector<float>> probes_ts(xor_steps);
                for (int t = 0; t < xor_steps; t++) {
                    // Broadcast input to all neurons with slight phase shift
                    for (int i = 0; i < N_NEURONS; i++)
                        h_signal[i] = input_seq[t] * (1.0f + 0.01f * (float)i / N_NEURONS);
                    HIP_CHECK(hipMemcpy(d_signal, h_signal.data(),
                                        N_NEURONS * sizeof(float),
                                        hipMemcpyHostToDevice));

                    if (method == 0) {
                        k_cross_kernel_step<<<1, N_NEURONS>>>(
                            d_signal, d_probes, d_state, d_l2_warm, d_atomics,
                            t, 0.2f);
                    } else if (method == 1) {
                        k_lif_step<<<1, N_NEURONS>>>(d_signal, d_probes, d_state,
                                                      t, 77777 + delay * 100);
                    } else {
                        k_esn_step<<<1, N_NEURONS>>>(d_signal, d_probes, d_state,
                                                      t, 77777 + delay * 100);
                    }
                    HIP_CHECK(hipDeviceSynchronize());

                    probes_ts[t].resize(N_NEURONS * N_PROBES);
                    HIP_CHECK(hipMemcpy(probes_ts[t].data(), d_probes,
                                        N_NEURONS * N_PROBES * sizeof(float),
                                        hipMemcpyDeviceToHost));
                }

                results[method] = xor_task(probes_ts, input_seq, xor_steps,
                                           N_NEURONS, N_PROBES, delay);
            }

            printf("  %5d  %.3f   %.3f   %.3f\n", delay, results[0], results[1], results[2]);
        }
    }
    fflush(stdout);

    // ═══ TEST 4: L2 CACHE PERSISTENCE PROBE ═══
    printf("\n═══ TEST 4: L2 CACHE TEMPORAL MEMORY ═══\n");
    printf("  Does L2 cache state from kernel N affect kernel N+1 timing?\n");
    {
        // Run pairs of kernels: (heavy write, then probe read) vs (no write, then probe read)
        // If L2 persists, the probe-read after heavy-write should be faster (warm cache)
        int n_pairs = 50;
        std::vector<float> warm_times(n_pairs), cold_times(n_pairs);

        for (int pair = 0; pair < n_pairs; pair++) {
            // === WARM condition: kernel that writes L2, then probe kernel ===
            HIP_CHECK(hipMemset(d_state, 0, N_NEURONS * sizeof(NeuronState)));
            // First kernel: writes to l2_warm buffer heavily
            for (int i = 0; i < N_NEURONS; i++) h_signal[i] = 1.0f;
            HIP_CHECK(hipMemcpy(d_signal, h_signal.data(), N_NEURONS * sizeof(float),
                                hipMemcpyHostToDevice));
            k_cross_kernel_step<<<1, N_NEURONS>>>(
                d_signal, d_probes, d_state, d_l2_warm, d_atomics, 0, 0.0f);
            HIP_CHECK(hipDeviceSynchronize());

            // Second kernel: reads same addresses (should be L2-warm)
            k_cross_kernel_step<<<1, N_NEURONS>>>(
                d_signal, d_probes, d_state, d_l2_warm, d_atomics, 1, 0.0f);
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(h_probes.data(), d_probes,
                                N_NEURONS * N_PROBES * sizeof(float),
                                hipMemcpyDeviceToHost));
            // Probe 2 = L2 hit time
            float avg_hit = 0;
            for (int n = 0; n < N_NEURONS; n++)
                avg_hit += h_probes[n * N_PROBES + 2];
            warm_times[pair] = avg_hit / N_NEURONS;

            // === COLD condition: flush by touching different memory, then probe ===
            // Allocate and touch a large buffer to evict L2
            float *d_flush;
            HIP_CHECK(hipMalloc(&d_flush, 4 * 1024 * 1024));  // 4MB > L2 on most GPUs
            HIP_CHECK(hipMemset(d_flush, 0x42, 4 * 1024 * 1024));
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipFree(d_flush));

            // Reset state
            HIP_CHECK(hipMemset(d_state, 0, N_NEURONS * sizeof(NeuronState)));
            // Probe kernel reads (should be L2-cold after flush)
            k_cross_kernel_step<<<1, N_NEURONS>>>(
                d_signal, d_probes, d_state, d_l2_warm, d_atomics, 0, 0.0f);
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(h_probes.data(), d_probes,
                                N_NEURONS * N_PROBES * sizeof(float),
                                hipMemcpyDeviceToHost));
            float avg_cold = 0;
            for (int n = 0; n < N_NEURONS; n++)
                avg_cold += h_probes[n * N_PROBES + 2];
            cold_times[pair] = avg_cold / N_NEURONS;
        }

        // Statistics
        float warm_mean = 0, cold_mean = 0;
        for (int i = 0; i < n_pairs; i++) {
            warm_mean += warm_times[i];
            cold_mean += cold_times[i];
        }
        warm_mean /= n_pairs;
        cold_mean /= n_pairs;

        float warm_std = 0, cold_std = 0;
        for (int i = 0; i < n_pairs; i++) {
            warm_std += (warm_times[i] - warm_mean) * (warm_times[i] - warm_mean);
            cold_std += (cold_times[i] - cold_mean) * (cold_times[i] - cold_mean);
        }
        warm_std = sqrtf(warm_std / n_pairs);
        cold_std = sqrtf(cold_std / n_pairs);

        printf("  L2-warm (after write kernel): %.1f ± %.1f cycles\n", warm_mean, warm_std);
        printf("  L2-cold (after flush):        %.1f ± %.1f cycles\n", cold_mean, cold_std);
        printf("  Ratio cold/warm: %.2fx\n", cold_mean / fmaxf(warm_mean, 1.0f));
        printf("  → L2 cache %s temporal memory between kernel launches\n",
               (cold_mean > warm_mean * 1.05f) ? "CREATES" : "does NOT create measurable");
    }
    fflush(stdout);

    // ═══ TEST 5: DVFS THERMAL DROOP PERSISTENCE ═══
    printf("\n═══ TEST 5: DVFS/THERMAL PERSISTENCE ═══\n");
    printf("  Does heavy workload in kernel N slow down kernel N+1?\n");
    {
        int n_pairs = 30;
        std::vector<float> after_heavy(n_pairs), after_light(n_pairs);

        for (int pair = 0; pair < n_pairs; pair++) {
            // === HEAVY preceding kernel ===
            HIP_CHECK(hipMemset(d_state, 0, N_NEURONS * sizeof(NeuronState)));
            // Launch 5 heavy kernels in succession (warm up DVFS)
            for (int i = 0; i < N_NEURONS; i++) h_signal[i] = 2.0f;  // max workload
            HIP_CHECK(hipMemcpy(d_signal, h_signal.data(), N_NEURONS * sizeof(float),
                                hipMemcpyHostToDevice));
            for (int warmup = 0; warmup < 5; warmup++) {
                k_cross_kernel_step<<<1, N_NEURONS>>>(
                    d_signal, d_probes, d_state, d_l2_warm, d_atomics, warmup, 0.5f);
                HIP_CHECK(hipDeviceSynchronize());
            }

            // Probe kernel immediately after heavy load
            for (int i = 0; i < N_NEURONS; i++) h_signal[i] = 0.5f;  // moderate
            HIP_CHECK(hipMemcpy(d_signal, h_signal.data(), N_NEURONS * sizeof(float),
                                hipMemcpyHostToDevice));
            k_cross_kernel_step<<<1, N_NEURONS>>>(
                d_signal, d_probes, d_state, d_l2_warm, d_atomics, 10, 0.0f);
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(h_probes.data(), d_probes,
                                N_NEURONS * N_PROBES * sizeof(float),
                                hipMemcpyDeviceToHost));
            float sum_fma = 0;
            for (int n = 0; n < N_NEURONS; n++)
                sum_fma += h_probes[n * N_PROBES + 4];  // FMA timing
            after_heavy[pair] = sum_fma / N_NEURONS;

            // === LIGHT preceding kernel ===
            HIP_CHECK(hipMemset(d_state, 0, N_NEURONS * sizeof(NeuronState)));
            // Light warmup: minimal work
            for (int i = 0; i < N_NEURONS; i++) h_signal[i] = 0.1f;
            HIP_CHECK(hipMemcpy(d_signal, h_signal.data(), N_NEURONS * sizeof(float),
                                hipMemcpyHostToDevice));
            for (int warmup = 0; warmup < 5; warmup++) {
                k_cross_kernel_step<<<1, N_NEURONS>>>(
                    d_signal, d_probes, d_state, d_l2_warm, d_atomics, warmup, 0.0f);
                HIP_CHECK(hipDeviceSynchronize());
            }

            // Same probe kernel after light load
            for (int i = 0; i < N_NEURONS; i++) h_signal[i] = 0.5f;
            HIP_CHECK(hipMemcpy(d_signal, h_signal.data(), N_NEURONS * sizeof(float),
                                hipMemcpyHostToDevice));
            k_cross_kernel_step<<<1, N_NEURONS>>>(
                d_signal, d_probes, d_state, d_l2_warm, d_atomics, 10, 0.0f);
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(h_probes.data(), d_probes,
                                N_NEURONS * N_PROBES * sizeof(float),
                                hipMemcpyDeviceToHost));
            float sum_fma2 = 0;
            for (int n = 0; n < N_NEURONS; n++)
                sum_fma2 += h_probes[n * N_PROBES + 4];
            after_light[pair] = sum_fma2 / N_NEURONS;
        }

        float heavy_mean = 0, light_mean = 0;
        for (int i = 0; i < n_pairs; i++) {
            heavy_mean += after_heavy[i];
            light_mean += after_light[i];
        }
        heavy_mean /= n_pairs;
        light_mean /= n_pairs;

        printf("  FMA timing after HEAVY load: %.1f cycles\n", heavy_mean);
        printf("  FMA timing after LIGHT load: %.1f cycles\n", light_mean);
        printf("  Ratio: %.3f\n", heavy_mean / fmaxf(light_mean, 1.0f));
        printf("  → Previous workload %s next kernel timing\n",
               (fabsf(heavy_mean - light_mean) > fmaxf(heavy_mean, light_mean) * 0.02f) ?
               "MEASURABLY AFFECTS" : "does not measurably affect");
    }
    fflush(stdout);

    // ═══ TEST 6: DISPATCH JITTER AUTOCORRELATION (DIRECT MEASUREMENT) ═══
    printf("\n═══ TEST 6: DISPATCH JITTER TEMPORAL STRUCTURE ═══\n");
    printf("  Measuring host-side kernel launch timing autocorrelation\n");
    {
        int n_launches = 500;
        std::vector<double> launch_times(n_launches);

        HIP_CHECK(hipMemset(d_state, 0, N_NEURONS * sizeof(NeuronState)));
        for (int i = 0; i < N_NEURONS; i++) h_signal[i] = 0.5f;
        HIP_CHECK(hipMemcpy(d_signal, h_signal.data(), N_NEURONS * sizeof(float),
                            hipMemcpyHostToDevice));

        for (int t = 0; t < n_launches; t++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            k_cross_kernel_step<<<1, N_NEURONS>>>(
                d_signal, d_probes, d_state, d_l2_warm, d_atomics, t, 0.2f);
            HIP_CHECK(hipDeviceSynchronize());
            auto t1 = std::chrono::high_resolution_clock::now();
            launch_times[t] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }

        // Compute ACF at lags 1..10
        double mean_lt = 0;
        for (int t = 0; t < n_launches; t++) mean_lt += launch_times[t];
        mean_lt /= n_launches;

        double var_lt = 0;
        for (int t = 0; t < n_launches; t++)
            var_lt += (launch_times[t] - mean_lt) * (launch_times[t] - mean_lt);

        printf("  Launch time: mean=%.1f µs, std=%.1f µs\n",
               mean_lt, sqrt(var_lt / n_launches));

        printf("  Dispatch ACF: ");
        for (int lag = 1; lag <= 10; lag++) {
            double cov = 0;
            for (int t = 0; t < n_launches - lag; t++)
                cov += (launch_times[t] - mean_lt) * (launch_times[t + lag] - mean_lt);
            double acf = (var_lt > 0) ? cov / var_lt : 0;
            printf("lag%d=%.3f ", lag, acf);
        }
        printf("\n");
    }
    fflush(stdout);

    // ═══ TEST 7: COMBINED PROBE-CHANNEL ABLATION FOR XOR ═══
    printf("\n═══ TEST 7: PROBE CHANNEL ABLATION (XOR τ=2) ═══\n");
    printf("  Which hardware signals carry temporal memory?\n");
    {
        const char* probe_names[] = {"entry_clk", "dispatch_Δ", "L2_hit", "L2_cold",
                                     "FMA_time", "SFU_time", "LDS_time", "exit_clk"};

        // First run full XOR with all probes
        int xor_steps = 150;
        int delay = 2;
        unsigned int rng_seed = 99999;
        std::vector<float> input_seq(xor_steps);
        for (int t = 0; t < xor_steps; t++) {
            rng_seed ^= rng_seed << 13; rng_seed ^= rng_seed >> 17; rng_seed ^= rng_seed << 5;
            input_seq[t] = ((rng_seed & 1) == 0) ? 1.0f : -1.0f;
        }

        // Collect probes
        HIP_CHECK(hipMemset(d_state, 0, N_NEURONS * sizeof(NeuronState)));
        HIP_CHECK(hipMemset(d_l2_warm, 0, N_NEURONS * 256 * sizeof(float)));
        HIP_CHECK(hipMemset(d_atomics, 0, 4 * sizeof(int)));

        std::vector<std::vector<float>> probes_ts(xor_steps);
        for (int t = 0; t < xor_steps; t++) {
            for (int i = 0; i < N_NEURONS; i++)
                h_signal[i] = input_seq[t] * (1.0f + 0.01f * (float)i / N_NEURONS);
            HIP_CHECK(hipMemcpy(d_signal, h_signal.data(), N_NEURONS * sizeof(float),
                                hipMemcpyHostToDevice));
            k_cross_kernel_step<<<1, N_NEURONS>>>(
                d_signal, d_probes, d_state, d_l2_warm, d_atomics, t, 0.2f);
            HIP_CHECK(hipDeviceSynchronize());
            probes_ts[t].resize(N_NEURONS * N_PROBES);
            HIP_CHECK(hipMemcpy(probes_ts[t].data(), d_probes,
                                N_NEURONS * N_PROBES * sizeof(float),
                                hipMemcpyDeviceToHost));
        }

        // Full XOR
        float full_xor = xor_task(probes_ts, input_seq, xor_steps, N_NEURONS, N_PROBES, delay);
        printf("  ALL probes:    XOR_τ2=%.3f\n", full_xor);

        // Per-probe XOR (each probe alone)
        for (int p = 0; p < N_PROBES; p++) {
            // Extract only this probe
            std::vector<std::vector<float>> single_probes(xor_steps);
            for (int t = 0; t < xor_steps; t++) {
                single_probes[t].resize(N_NEURONS);
                for (int n = 0; n < N_NEURONS; n++)
                    single_probes[t][n] = probes_ts[t][n * N_PROBES + p];
            }
            float xor_acc = xor_task(single_probes, input_seq, xor_steps, N_NEURONS, 1, delay);
            printf("  %-12s:  XOR_τ2=%.3f  %s\n", probe_names[p], xor_acc,
                   (xor_acc > 0.55f) ? "← TEMPORAL MEMORY" :
                   (xor_acc > 0.52f) ? "← marginal" : "");
        }
    }
    fflush(stdout);

    // Cleanup
    HIP_CHECK(hipFree(d_signal));
    HIP_CHECK(hipFree(d_probes));
    HIP_CHECK(hipFree(d_state));
    HIP_CHECK(hipFree(d_l2_warm));
    HIP_CHECK(hipFree(d_atomics));

    printf("\n═══ DONE ═══\n");
    return 0;
}
