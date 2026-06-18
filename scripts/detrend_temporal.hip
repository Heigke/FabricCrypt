/**
 * detrend_temporal.hip — Extract INPUT-DEPENDENT temporal memory from GPU silicon
 *
 * KEY INSIGHT from cross_kernel_memory results:
 *   - entry_clock ACF(1)=0.983, XOR τ=2 = 0.558 (marginal signal)
 *   - The ACF is dominated by monotonic clock trend, not input modulation
 *   - FMA timing ACF=0.836 but XOR=0.442 (no input-dependent memory)
 *
 * APPROACH:
 *   1. Detrend: subtract running mean, use DERIVATIVES of timing signals
 *   2. Difference features: timing(t) - timing(t-1) removes slow trends
 *   3. Cross-neuron differential: neuron_i(t) - neuron_j(t) cancels global drift
 *   4. Residual after linear prediction: what the previous step CAN'T predict
 *   5. Modulated workload: make kernels HEAVIER to amplify voltage droop effects
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", e, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static constexpr int N_NEURONS = 128;
static constexpr int N_STEPS   = 200;
static constexpr int N_PROBES  = 4;    // focused: FMA, SFU, L2, dispatch_delta

struct NeuronState {
    float activation;
    long long last_exit;
};

// ── HEAVIER kernel to amplify DVFS/droop effects ──
// Each neuron does 10× more work than cross_kernel_memory to actually
// push voltage rails and create measurable power transients
__global__ void k_heavy_step(
    const float* __restrict__ signal,
    float* __restrict__ probes,
    NeuronState* __restrict__ state,
    float* __restrict__ l2_buf,
    int step,
    float recurrence
) {
    int nid = threadIdx.x;
    if (nid >= N_NEURONS) return;

    long long entry = wall_clock64();
    long long prev_exit = state[nid].last_exit;
    float dispatch_delta = (prev_exit > 0) ? (float)(entry - prev_exit) : 0.0f;

    float inp = signal[nid];
    float prev = state[nid].activation;
    float modifier = fabsf(inp + recurrence * prev) + 0.5f;
    modifier = fminf(modifier, 3.0f);

    // HEAVY FMA: 500 iterations × modifier (enough to droop voltage)
    int fma_iters = (int)(500.0f * modifier);
    long long t0 = wall_clock64();
    float acc = 1.0001f;
    for (int i = 0; i < fma_iters; i++) {
        acc = __fmaf_rn(acc, 1.00001f, 0.00001f);
        acc = __fmaf_rn(acc, 0.99999f, 0.00001f);
    }
    long long t1 = wall_clock64();

    // HEAVY SFU: 200 iterations × modifier
    int sfu_iters = (int)(200.0f * modifier);
    long long t2 = wall_clock64();
    float sfu = 0.01f + (float)nid * 0.0001f;
    for (int i = 0; i < sfu_iters; i++) {
        sfu = __sinf(sfu + 0.001f);
        sfu = __cosf(sfu + 0.001f);
    }
    long long t3 = wall_clock64();

    // L2 access pattern modulated by input (different addresses per input sign)
    long long t4 = wall_clock64();
    float l2_acc = 0.0f;
    int base = nid * 256;
    int stride = (inp > 0.0f) ? 1 : 4;  // input modulates access pattern
    for (int i = 0; i < 128; i++) {
        l2_acc += l2_buf[base + (i * stride) % 256];
    }
    long long t5 = wall_clock64();

    // Write back for next kernel's L2 state
    for (int i = 0; i < 64; i++) {
        l2_buf[base + i] = acc * 0.001f + sfu * 0.001f;
    }

    long long exit_clk = wall_clock64();

    // Store detrend-friendly probes
    probes[nid * N_PROBES + 0] = (float)(t1 - t0);   // FMA cycles
    probes[nid * N_PROBES + 1] = (float)(t3 - t2);   // SFU cycles
    probes[nid * N_PROBES + 2] = (float)(t5 - t4);   // L2 cycles
    probes[nid * N_PROBES + 3] = dispatch_delta;       // inter-kernel gap

    state[nid].activation = (float)(t1 - t0) / 10000.0f;
    state[nid].last_exit = exit_clk;
}

// Software LIF baseline with explicit temporal memory
__global__ void k_lif_heavy(
    const float* __restrict__ signal,
    float* __restrict__ probes,
    NeuronState* __restrict__ state,
    int step, unsigned int seed
) {
    int nid = threadIdx.x;
    if (nid >= N_NEURONS) return;

    unsigned int rng = seed + nid * 7919 + step * 1337;
    auto xorshift = [](unsigned int& s) -> float {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return (float)(s & 0xFFFF) / 65536.0f - 0.5f;
    };

    float inp = signal[nid];
    float prev = state[nid].activation;
    float vmem = 0.95f * prev + 0.3f * inp + 0.05f * xorshift(rng);
    float spike = (vmem > 1.0f) ? 1.0f : 0.0f;
    if (spike > 0.5f) vmem = 0.0f;

    probes[nid * N_PROBES + 0] = vmem;
    probes[nid * N_PROBES + 1] = spike;
    probes[nid * N_PROBES + 2] = vmem - prev;
    probes[nid * N_PROBES + 3] = xorshift(rng);

    state[nid].activation = vmem;
}

// ══════════════════════════════════════════════════════════════
//  HOST: Detrending + classification
// ══════════════════════════════════════════════════════════════

// Detrend a time series: subtract running mean with window W
void detrend(std::vector<float>& ts, int window = 10) {
    int n = ts.size();
    std::vector<float> trend(n, 0.0f);
    for (int t = 0; t < n; t++) {
        float sum = 0;
        int cnt = 0;
        for (int w = std::max(0, t - window); w <= std::min(n-1, t + window); w++) {
            sum += ts[w];
            cnt++;
        }
        trend[t] = sum / cnt;
    }
    for (int t = 0; t < n; t++) ts[t] -= trend[t];
}

// Extract detrended features from probe time series
struct Features {
    std::vector<float> data;
    int dim;
};

Features extract_detrended_features(
    const std::vector<std::vector<float>>& probes_all,
    int n_steps, int n_neurons, int n_probes,
    bool use_raw, bool use_diff, bool use_crossneuron, bool use_detrend
) {
    // Feature dimensions per approach:
    // raw: n_neurons * n_probes (mean of time series)
    // diff: n_neurons * n_probes (mean of first difference)
    // crossneuron: (n_neurons/2) * n_probes (pairwise differences)
    // detrend: n_neurons * n_probes * 3 (mean, std, acf1 of detrended)

    std::vector<float> feats;

    for (int n = 0; n < n_neurons; n++) {
        for (int p = 0; p < n_probes; p++) {
            std::vector<float> ts(n_steps);
            for (int t = 0; t < n_steps; t++)
                ts[t] = probes_all[t][n * n_probes + p];

            if (use_raw) {
                float mean = 0;
                for (float v : ts) mean += v;
                mean /= n_steps;
                float std_dev = 0;
                for (float v : ts) std_dev += (v - mean) * (v - mean);
                std_dev = sqrtf(std_dev / n_steps + 1e-12f);
                feats.push_back(mean);
                feats.push_back(std_dev);
            }

            if (use_diff) {
                // First differences: timing(t) - timing(t-1)
                float d_mean = 0, d_std = 0;
                float d_acf = 0, d_var = 0;
                std::vector<float> diffs(n_steps - 1);
                for (int t = 1; t < n_steps; t++) {
                    diffs[t-1] = ts[t] - ts[t-1];
                    d_mean += diffs[t-1];
                }
                d_mean /= (n_steps - 1);
                for (int t = 0; t < n_steps - 1; t++) {
                    d_var += (diffs[t] - d_mean) * (diffs[t] - d_mean);
                }
                d_std = sqrtf(d_var / (n_steps - 1) + 1e-12f);

                // ACF of differences
                if (d_var > 1e-12f) {
                    for (int t = 0; t < n_steps - 2; t++)
                        d_acf += (diffs[t] - d_mean) * (diffs[t+1] - d_mean);
                    d_acf /= d_var;
                }
                feats.push_back(d_mean);
                feats.push_back(d_std);
                feats.push_back(d_acf);
            }

            if (use_detrend) {
                std::vector<float> dt = ts;
                detrend(dt, 15);
                float mean = 0, var = 0;
                for (float v : dt) mean += v;
                mean /= n_steps;
                for (float v : dt) var += (v - mean) * (v - mean);
                float std_dev = sqrtf(var / n_steps + 1e-12f);

                // ACF of detrended
                float acf = 0;
                if (var > 1e-12f) {
                    for (int t = 0; t < n_steps - 1; t++)
                        acf += (dt[t] - mean) * (dt[t+1] - mean);
                    acf /= var;
                }
                feats.push_back(mean);
                feats.push_back(std_dev);
                feats.push_back(acf);
            }
        }

        if (use_crossneuron && n < n_neurons / 2) {
            // Cross-neuron differential: neuron_n - neuron_(n + 64)
            int n2 = n + n_neurons / 2;
            for (int p = 0; p < n_probes; p++) {
                float diff_mean = 0;
                for (int t = 0; t < n_steps; t++) {
                    float d = probes_all[t][n * n_probes + p] -
                              probes_all[t][n2 * n_probes + p];
                    diff_mean += d;
                }
                feats.push_back(diff_mean / n_steps);
            }
        }
    }

    Features f;
    f.dim = feats.size();
    f.data = feats;
    return f;
}

// Point-wise features for XOR (per-step classification)
Features extract_pointwise(
    const std::vector<std::vector<float>>& probes_all,
    int step, int n_neurons, int n_probes,
    const std::vector<std::vector<float>>& history, int hist_depth
) {
    std::vector<float> feats;

    // Current step probes
    for (int j = 0; j < n_neurons * n_probes; j++)
        feats.push_back(probes_all[step][j]);

    // Differences from previous steps (DETRENDED temporal features)
    for (int h = 1; h <= hist_depth && step >= h; h++) {
        for (int j = 0; j < n_neurons * n_probes; j++) {
            feats.push_back(probes_all[step][j] - probes_all[step - h][j]);
        }
    }

    // Cross-neuron differentials at current step
    for (int n = 0; n < n_neurons / 2; n++) {
        for (int p = 0; p < n_probes; p++) {
            feats.push_back(probes_all[step][n * n_probes + p] -
                           probes_all[step][(n + n_neurons/2) * n_probes + p]);
        }
    }

    Features f;
    f.dim = feats.size();
    f.data = feats;
    return f;
}

// Ridge classifier (simplified)
float ridge_classify(const std::vector<Features>& X, const std::vector<int>& Y,
                     int n_classes, float lambda = 1.0f) {
    int n = X.size();
    if (n < 6) return 0.5f;
    int d = X[0].dim;
    if (d > 8000) d = 8000;  // cap to avoid memory issues

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

        // Standardize
        std::vector<float> mu(d, 0), sigma(d, 1);
        for (int j = 0; j < d; j++) {
            float s = 0, s2 = 0;
            for (int i = 0; i < n_train; i++) {
                float v = (j < (int)X[train_idx[i]].data.size()) ? X[train_idx[i]].data[j] : 0;
                s += v; s2 += v * v;
            }
            mu[j] = s / n_train;
            float var = s2 / n_train - mu[j] * mu[j];
            sigma[j] = (var > 1e-10f) ? sqrtf(var) : 1.0f;
        }

        // Build standardized XtX and XtY
        std::vector<float> XtX(d * d, 0), XtY(d * n_classes, 0);
        std::vector<float> x_row(d);
        for (int i = 0; i < n_train; i++) {
            for (int j = 0; j < d; j++) {
                float v = (j < (int)X[train_idx[i]].data.size()) ? X[train_idx[i]].data[j] : 0;
                x_row[j] = (v - mu[j]) / sigma[j];
            }
            for (int j1 = 0; j1 < d; j1++) {
                for (int j2 = j1; j2 < d; j2++) {
                    XtX[j1 * d + j2] += x_row[j1] * x_row[j2];
                }
                XtY[j1 * n_classes + Y[train_idx[i]]] += x_row[j1];
            }
        }
        for (int j1 = 0; j1 < d; j1++) {
            XtX[j1 * d + j1] += lambda;
            for (int j2 = j1 + 1; j2 < d; j2++)
                XtX[j2 * d + j1] = XtX[j1 * d + j2];
        }

        // Cholesky solve
        std::vector<float> L(d * d, 0);
        for (int i = 0; i < d; i++) {
            for (int j = 0; j <= i; j++) {
                float s = XtX[i * d + j];
                for (int k = 0; k < j; k++) s -= L[i*d+k] * L[j*d+k];
                L[i*d+j] = (i == j) ? sqrtf(fmaxf(s, 1e-6f)) : s / L[j*d+j];
            }
        }

        std::vector<float> W(d * n_classes, 0);
        for (int c = 0; c < n_classes; c++) {
            std::vector<float> z(d);
            for (int i = 0; i < d; i++) {
                float s = XtY[i * n_classes + c];
                for (int k = 0; k < i; k++) s -= L[i*d+k] * z[k];
                z[i] = s / L[i*d+i];
            }
            for (int i = d-1; i >= 0; i--) {
                float s = z[i];
                for (int k = i+1; k < d; k++) s -= L[k*d+i] * W[k*n_classes+c];
                W[i*n_classes+c] = s / L[i*d+i];
            }
        }

        // Test
        for (int ti = 0; ti < n_test; ti++) {
            std::vector<float> scores(n_classes, 0);
            for (int j = 0; j < d; j++) {
                float v = (j < (int)X[test_idx[ti]].data.size()) ? X[test_idx[ti]].data[j] : 0;
                float xv = (v - mu[j]) / sigma[j];
                for (int c = 0; c < n_classes; c++)
                    scores[c] += xv * W[j * n_classes + c];
            }
            int pred = std::max_element(scores.begin(), scores.end()) - scores.begin();
            if (pred == Y[test_idx[ti]]) total_correct++;
            total_test++;
        }
    }
    return total_correct / total_test;
}

void generate_signal(float* sig, int n, int step, int total, int cls, float amp = 1.0f) {
    float t = (float)step / (float)total;
    for (int i = 0; i < n; i++) {
        float ph = (float)i / n * 2.0f * M_PI;
        switch (cls) {
            case 0: sig[i] = amp * sinf(2*M_PI*t*3+ph); break;
            case 1: sig[i] = amp * (fmodf(t*5+ph/(2*M_PI),1)*2-1); break;
            case 2: sig[i] = amp * ((sinf(2*M_PI*t*4+ph)>0)?1:-1); break;
            case 3: sig[i] = amp * sinf(2*M_PI*t*2+ph)*cosf(2*M_PI*t*7+ph*0.5f); break;
        }
    }
}

int main() {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  DETRENDED TEMPORAL MEMORY — Finding Input-Dependent Signal\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Strategy: heavier kernels + detrend + diff features + cross-neuron\n\n");

    float *d_sig, *d_probes, *d_l2;
    NeuronState *d_state;
    HIP_CHECK(hipMalloc(&d_sig, N_NEURONS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_probes, N_NEURONS * N_PROBES * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_state, N_NEURONS * sizeof(NeuronState)));
    HIP_CHECK(hipMalloc(&d_l2, N_NEURONS * 256 * sizeof(float)));

    std::vector<float> h_sig(N_NEURONS), h_probes(N_NEURONS * N_PROBES);

    // Helper: run N_STEPS kernel launches, collect probes
    auto run_sequence = [&](int method, int cls, int trial, float recurrence,
                            std::vector<std::vector<float>>& out) {
        HIP_CHECK(hipMemset(d_state, 0, N_NEURONS * sizeof(NeuronState)));
        HIP_CHECK(hipMemset(d_l2, 0, N_NEURONS * 256 * sizeof(float)));
        out.resize(N_STEPS);
        for (int t = 0; t < N_STEPS; t++) {
            float amp = 0.5f + 0.3f * (float)trial / 15.0f;
            generate_signal(h_sig.data(), N_NEURONS, t, N_STEPS, cls, amp);
            HIP_CHECK(hipMemcpy(d_sig, h_sig.data(), N_NEURONS * sizeof(float),
                                hipMemcpyHostToDevice));
            if (method == 0) {
                k_heavy_step<<<1, N_NEURONS>>>(d_sig, d_probes, d_state, d_l2, t, recurrence);
            } else {
                k_lif_heavy<<<1, N_NEURONS>>>(d_sig, d_probes, d_state, t,
                                               42 + trial * 1000 + cls * 100000);
            }
            HIP_CHECK(hipDeviceSynchronize());
            out[t].resize(N_NEURONS * N_PROBES);
            HIP_CHECK(hipMemcpy(out[t].data(), d_probes,
                                N_NEURONS * N_PROBES * sizeof(float),
                                hipMemcpyDeviceToHost));
        }
    };

    // ═══ TEST 1: ACF of DETRENDED signals ═══
    printf("═══ TEST 1: ACF of RAW vs DETRENDED probe signals ═══\n");
    {
        std::vector<std::vector<float>> probes;
        run_sequence(0, 0, 0, 0.2f, probes);

        const char* names[] = {"FMA_time", "SFU_time", "L2_time", "dispatch_Δ"};
        for (int p = 0; p < N_PROBES; p++) {
            // Raw ACF
            float raw_acf = 0, raw_var = 0, raw_mean = 0;
            std::vector<float> ts(N_STEPS);
            for (int t = 0; t < N_STEPS; t++) {
                ts[t] = probes[t][p]; // neuron 0
                raw_mean += ts[t];
            }
            raw_mean /= N_STEPS;
            for (int t = 0; t < N_STEPS; t++) raw_var += (ts[t]-raw_mean)*(ts[t]-raw_mean);
            for (int t = 0; t < N_STEPS-1; t++) raw_acf += (ts[t]-raw_mean)*(ts[t+1]-raw_mean);
            raw_acf = (raw_var > 0) ? raw_acf / raw_var : 0;

            // Detrended ACF
            std::vector<float> dt = ts;
            detrend(dt, 15);
            float dt_acf = 0, dt_var = 0, dt_mean = 0;
            for (float v : dt) dt_mean += v;
            dt_mean /= N_STEPS;
            for (float v : dt) dt_var += (v - dt_mean)*(v - dt_mean);
            for (int t = 0; t < N_STEPS-1; t++) dt_acf += (dt[t]-dt_mean)*(dt[t+1]-dt_mean);
            dt_acf = (dt_var > 0) ? dt_acf / dt_var : 0;

            // First-difference ACF
            float fd_acf = 0, fd_var = 0, fd_mean = 0;
            std::vector<float> diffs(N_STEPS - 1);
            for (int t = 0; t < N_STEPS-1; t++) {
                diffs[t] = ts[t+1] - ts[t];
                fd_mean += diffs[t];
            }
            fd_mean /= (N_STEPS-1);
            for (float v : diffs) fd_var += (v - fd_mean)*(v - fd_mean);
            for (int t = 0; t < N_STEPS-2; t++) fd_acf += (diffs[t]-fd_mean)*(diffs[t+1]-fd_mean);
            fd_acf = (fd_var > 0) ? fd_acf / fd_var : 0;

            printf("  %-12s  raw_ACF=%.3f  detrend_ACF=%.3f  diff_ACF=%.3f\n",
                   names[p], raw_acf, dt_acf, fd_acf);
        }
    }
    fflush(stdout);

    // ═══ TEST 2: XOR with detrended features (τ=1,2,3,5) ═══
    printf("\n═══ TEST 2: XOR WITH DETRENDED FEATURES ═══\n");
    printf("  Using first-differences + history depth as temporal memory\n");
    {
        int delays[] = {1, 2, 3, 5};
        int hist_depths[] = {0, 1, 2, 3};

        for (int di = 0; di < 4; di++) {
            int delay = delays[di];
            printf("  XOR τ=%d: ", delay);

            // Generate input sequence
            int xor_steps = N_STEPS;
            std::vector<float> input_seq(xor_steps);
            unsigned int rng = 54321 + delay;
            for (int t = 0; t < xor_steps; t++) {
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                input_seq[t] = ((rng & 1) == 0) ? 1.0f : -1.0f;
            }

            for (int hi = 0; hi < 4; hi++) {
                int hist = hist_depths[hi];

                // Run neuromorphic
                std::vector<std::vector<float>> probes_n;
                {
                    HIP_CHECK(hipMemset(d_state, 0, N_NEURONS * sizeof(NeuronState)));
                    HIP_CHECK(hipMemset(d_l2, 0, N_NEURONS * 256 * sizeof(float)));
                    probes_n.resize(xor_steps);
                    for (int t = 0; t < xor_steps; t++) {
                        for (int i = 0; i < N_NEURONS; i++)
                            h_sig[i] = input_seq[t] * (1.0f + 0.01f * i / (float)N_NEURONS);
                        HIP_CHECK(hipMemcpy(d_sig, h_sig.data(), N_NEURONS * sizeof(float),
                                            hipMemcpyHostToDevice));
                        k_heavy_step<<<1, N_NEURONS>>>(d_sig, d_probes, d_state, d_l2, t, 0.3f);
                        HIP_CHECK(hipDeviceSynchronize());
                        probes_n[t].resize(N_NEURONS * N_PROBES);
                        HIP_CHECK(hipMemcpy(probes_n[t].data(), d_probes,
                                            N_NEURONS * N_PROBES * sizeof(float),
                                            hipMemcpyDeviceToHost));
                    }
                }

                // Build features with history
                int start = std::max(delay + 5, hist + 5);
                int n_samples = xor_steps - start;
                std::vector<Features> feats(n_samples);
                std::vector<int> labels(n_samples);

                for (int t = 0; t < n_samples; t++) {
                    int step = start + t;
                    feats[t] = extract_pointwise(probes_n, step, N_NEURONS, N_PROBES,
                                                 probes_n, hist);
                    int b1 = (input_seq[step] > 0) ? 1 : 0;
                    int b2 = (input_seq[step - delay] > 0) ? 1 : 0;
                    labels[t] = b1 ^ b2;
                }

                float acc = ridge_classify(feats, labels, 2);
                printf("h%d=%.3f ", hist, acc);
            }
            printf("\n");
        }

        // Same for LIF baseline with best history
        printf("  --- LIF baseline (hist=3) ---\n");
        for (int di = 0; di < 4; di++) {
            int delay = delays[di];
            int xor_steps = N_STEPS;
            std::vector<float> input_seq(xor_steps);
            unsigned int rng = 54321 + delay;
            for (int t = 0; t < xor_steps; t++) {
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                input_seq[t] = ((rng & 1) == 0) ? 1.0f : -1.0f;
            }

            std::vector<std::vector<float>> probes_l;
            {
                HIP_CHECK(hipMemset(d_state, 0, N_NEURONS * sizeof(NeuronState)));
                probes_l.resize(xor_steps);
                for (int t = 0; t < xor_steps; t++) {
                    for (int i = 0; i < N_NEURONS; i++)
                        h_sig[i] = input_seq[t] * (1.0f + 0.01f * i / (float)N_NEURONS);
                    HIP_CHECK(hipMemcpy(d_sig, h_sig.data(), N_NEURONS * sizeof(float),
                                        hipMemcpyHostToDevice));
                    k_lif_heavy<<<1, N_NEURONS>>>(d_sig, d_probes, d_state, t, 42 + delay);
                    HIP_CHECK(hipDeviceSynchronize());
                    probes_l[t].resize(N_NEURONS * N_PROBES);
                    HIP_CHECK(hipMemcpy(probes_l[t].data(), d_probes,
                                        N_NEURONS * N_PROBES * sizeof(float),
                                        hipMemcpyDeviceToHost));
                }
            }

            int start = delay + 8;
            int n_samples = xor_steps - start;
            std::vector<Features> feats(n_samples);
            std::vector<int> labels(n_samples);
            for (int t = 0; t < n_samples; t++) {
                int step = start + t;
                feats[t] = extract_pointwise(probes_l, step, N_NEURONS, N_PROBES,
                                             probes_l, 3);
                int b1 = (input_seq[step] > 0) ? 1 : 0;
                int b2 = (input_seq[step - delay] > 0) ? 1 : 0;
                labels[t] = b1 ^ b2;
            }
            float acc = ridge_classify(feats, labels, 2);
            printf("  XOR τ=%d LIF(h3): %.3f\n", delay, acc);
        }
    }
    fflush(stdout);

    // ═══ TEST 3: CLASSIFICATION with different feature strategies ═══
    printf("\n═══ TEST 3: CLASSIFICATION — Feature Strategy Comparison ═══\n");
    {
        struct Strategy {
            const char* name;
            bool raw, diff, cross, detrend;
        };
        Strategy strategies[] = {
            {"raw_only",      true,  false, false, false},
            {"diff_only",     false, true,  false, false},
            {"detrend_only",  false, false, false, true},
            {"cross_only",    false, false, true,  false},
            {"all_combined",  true,  true,  true,  true},
            {"diff+detrend",  false, true,  false, true},
        };

        for (auto& strat : strategies) {
            int n_trials_per_class = 15;
            std::vector<Features> all_feats;
            std::vector<int> all_labels;

            for (int cls = 0; cls < 4; cls++) {
                for (int trial = 0; trial < n_trials_per_class; trial++) {
                    std::vector<std::vector<float>> probes;
                    run_sequence(0, cls, trial, 0.3f, probes);
                    Features f = extract_detrended_features(probes, N_STEPS, N_NEURONS, N_PROBES,
                                                            strat.raw, strat.diff,
                                                            strat.cross, strat.detrend);
                    all_feats.push_back(f);
                    all_labels.push_back(cls);
                }
            }

            float acc = ridge_classify(all_feats, all_labels, 4);
            printf("  %-16s  4-class=%.1f%%  (dim=%d)\n", strat.name, acc * 100.0f,
                   all_feats[0].dim);
        }

        // LIF baseline with best strategy
        {
            int n_trials_per_class = 15;
            std::vector<Features> all_feats;
            std::vector<int> all_labels;
            for (int cls = 0; cls < 4; cls++) {
                for (int trial = 0; trial < n_trials_per_class; trial++) {
                    std::vector<std::vector<float>> probes;
                    run_sequence(1, cls, trial, 0.0f, probes);
                    Features f = extract_detrended_features(probes, N_STEPS, N_NEURONS, N_PROBES,
                                                            true, true, true, true);
                    all_feats.push_back(f);
                    all_labels.push_back(cls);
                }
            }
            float acc = ridge_classify(all_feats, all_labels, 4);
            printf("  LIF_all_comb    4-class=%.1f%%  (dim=%d)\n", acc * 100.0f,
                   all_feats[0].dim);
        }
    }
    fflush(stdout);

    // ═══ TEST 4: MEMORY CAPACITY via detrended features ═══
    printf("\n═══ TEST 4: MEMORY CAPACITY (detrended probes) ═══\n");
    {
        int mc_steps = N_STEPS;
        // Random input
        std::vector<float> input_seq(mc_steps);
        unsigned int rng = 31415;
        for (int t = 0; t < mc_steps; t++) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            input_seq[t] = (float)(rng & 0xFFFF) / 65536.0f * 2.0f - 1.0f;
        }

        // Run neuromorphic
        std::vector<std::vector<float>> probes_n;
        {
            HIP_CHECK(hipMemset(d_state, 0, N_NEURONS * sizeof(NeuronState)));
            HIP_CHECK(hipMemset(d_l2, 0, N_NEURONS * 256 * sizeof(float)));
            probes_n.resize(mc_steps);
            for (int t = 0; t < mc_steps; t++) {
                for (int i = 0; i < N_NEURONS; i++)
                    h_sig[i] = input_seq[t] * (0.8f + 0.4f * (float)i / N_NEURONS);
                HIP_CHECK(hipMemcpy(d_sig, h_sig.data(), N_NEURONS * sizeof(float),
                                    hipMemcpyHostToDevice));
                k_heavy_step<<<1, N_NEURONS>>>(d_sig, d_probes, d_state, d_l2, t, 0.3f);
                HIP_CHECK(hipDeviceSynchronize());
                probes_n[t].resize(N_NEURONS * N_PROBES);
                HIP_CHECK(hipMemcpy(probes_n[t].data(), d_probes,
                                    N_NEURONS * N_PROBES * sizeof(float),
                                    hipMemcpyDeviceToHost));
            }
        }

        // For each delay d=1..15, compute R² of predicting input(t-d) from probes(t)
        printf("  delay  R²(neuro)  R²(LIF)\n");
        float mc_n = 0, mc_l = 0;

        // Run LIF too
        std::vector<std::vector<float>> probes_l;
        {
            HIP_CHECK(hipMemset(d_state, 0, N_NEURONS * sizeof(NeuronState)));
            probes_l.resize(mc_steps);
            for (int t = 0; t < mc_steps; t++) {
                for (int i = 0; i < N_NEURONS; i++)
                    h_sig[i] = input_seq[t] * (0.8f + 0.4f * (float)i / N_NEURONS);
                HIP_CHECK(hipMemcpy(d_sig, h_sig.data(), N_NEURONS * sizeof(float),
                                    hipMemcpyHostToDevice));
                k_lif_heavy<<<1, N_NEURONS>>>(d_sig, d_probes, d_state, t, 42);
                HIP_CHECK(hipDeviceSynchronize());
                probes_l[t].resize(N_NEURONS * N_PROBES);
                HIP_CHECK(hipMemcpy(probes_l[t].data(), d_probes,
                                    N_NEURONS * N_PROBES * sizeof(float),
                                    hipMemcpyDeviceToHost));
            }
        }

        for (int delay = 1; delay <= 15; delay++) {
            int start = delay + 5;
            int n_samples = mc_steps - start;
            int feat_dim = N_NEURONS * N_PROBES;

            // Simple linear regression: R² = corr² between predicted and actual
            // Using ridge regression to predict input(t-delay) from probes(t)
            auto compute_mc = [&](const std::vector<std::vector<float>>& probes) -> float {
                // Build X and y
                std::vector<float> X(n_samples * feat_dim), y(n_samples);
                for (int i = 0; i < n_samples; i++) {
                    int t = start + i;
                    for (int j = 0; j < feat_dim; j++)
                        X[i * feat_dim + j] = probes[t][j];
                    y[i] = input_seq[t - delay];
                }

                // Standardize X
                std::vector<float> mu(feat_dim), sig(feat_dim);
                for (int j = 0; j < feat_dim; j++) {
                    float s = 0, s2 = 0;
                    for (int i = 0; i < n_samples; i++) {
                        s += X[i*feat_dim+j]; s2 += X[i*feat_dim+j]*X[i*feat_dim+j];
                    }
                    mu[j] = s / n_samples;
                    float v = s2/n_samples - mu[j]*mu[j];
                    sig[j] = (v > 1e-10f) ? sqrtf(v) : 1.0f;
                    for (int i = 0; i < n_samples; i++)
                        X[i*feat_dim+j] = (X[i*feat_dim+j] - mu[j]) / sig[j];
                }

                // Ridge: w = (XtX + λI)^-1 Xty
                // Use leave-one-out via shortcut: predicted = y_hat from LOO
                // Simpler: just compute training R² (biased but indicative)
                float lambda = 10.0f;
                std::vector<float> XtX(feat_dim * feat_dim, 0);
                std::vector<float> Xty(feat_dim, 0);
                for (int j1 = 0; j1 < feat_dim; j1++) {
                    for (int j2 = j1; j2 < feat_dim; j2++) {
                        float dot = 0;
                        for (int i = 0; i < n_samples; i++)
                            dot += X[i*feat_dim+j1] * X[i*feat_dim+j2];
                        XtX[j1*feat_dim+j2] = dot;
                        XtX[j2*feat_dim+j1] = dot;
                    }
                    XtX[j1*feat_dim+j1] += lambda;
                    float dot = 0;
                    for (int i = 0; i < n_samples; i++)
                        dot += X[i*feat_dim+j1] * y[i];
                    Xty[j1] = dot;
                }

                // Cholesky
                std::vector<float> L(feat_dim * feat_dim, 0);
                for (int i = 0; i < feat_dim; i++) {
                    for (int j = 0; j <= i; j++) {
                        float s = XtX[i*feat_dim+j];
                        for (int k = 0; k < j; k++) s -= L[i*feat_dim+k]*L[j*feat_dim+k];
                        L[i*feat_dim+j] = (i==j) ? sqrtf(fmaxf(s,1e-6f)) : s/L[j*feat_dim+j];
                    }
                }
                std::vector<float> w(feat_dim);
                std::vector<float> z(feat_dim);
                for (int i = 0; i < feat_dim; i++) {
                    float s = Xty[i];
                    for (int k = 0; k < i; k++) s -= L[i*feat_dim+k]*z[k];
                    z[i] = s / L[i*feat_dim+i];
                }
                for (int i = feat_dim-1; i >= 0; i--) {
                    float s = z[i];
                    for (int k = i+1; k < feat_dim; k++) s -= L[k*feat_dim+i]*w[k];
                    w[i] = s / L[i*feat_dim+i];
                }

                // Compute R²
                float ss_res = 0, ss_tot = 0;
                float y_mean = 0;
                for (float v : y) y_mean += v;
                y_mean /= n_samples;
                for (int i = 0; i < n_samples; i++) {
                    float pred = 0;
                    for (int j = 0; j < feat_dim; j++)
                        pred += X[i*feat_dim+j] * w[j];
                    ss_res += (y[i] - pred) * (y[i] - pred);
                    ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
                }
                return 1.0f - ss_res / fmaxf(ss_tot, 1e-12f);
            };

            float r2_n = compute_mc(probes_n);
            float r2_l = compute_mc(probes_l);
            mc_n += fmaxf(r2_n, 0.0f);
            mc_l += fmaxf(r2_l, 0.0f);

            printf("  %5d  %+.4f    %+.4f\n", delay, r2_n, r2_l);
        }
        printf("  TOTAL MC:  neuro=%.3f  LIF=%.3f\n", mc_n, mc_l);
    }
    fflush(stdout);

    HIP_CHECK(hipFree(d_sig));
    HIP_CHECK(hipFree(d_probes));
    HIP_CHECK(hipFree(d_state));
    HIP_CHECK(hipFree(d_l2));

    printf("\n═══ DONE ═══\n");
    return 0;
}
