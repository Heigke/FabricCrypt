// jitter_gpu_rc_sweep.hip — Hyperparameter sweep for GPU-only reservoir computing
// Find the sweet spot where PLL noise HELPS rather than HURTS temporal tasks
//
// Sweep: noise_gain × recurrent_weight × leak_range × input_weight
// Tasks: XOR-τ2, NARMA-5, Memory Capacity
// For each config: NOISE vs CONTROL (same recurrence, no noise)
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_gpu_rc_sweep scripts/jitter_gpu_rc_sweep.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_gpu_rc_sweep

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
#define N_NEURONS (N_CUS * THREADS)  // 768
#define STEPS 600
#define WASHOUT 100
#define SAMPLE_N 192
#define FEAT_PER_N 1  // just raw state for speed

// Configurable via constant memory
struct RCConfig {
    float noise_gain;    // how much PLL noise to inject
    float rec_w;         // local recurrence weight scale
    float cross_w;       // cross-CU recurrence weight scale
    float leak_lo;       // leak range [lo, hi]
    float leak_hi;
    float input_w_lo;    // input weight range
    float input_w_hi;
    int   pll_depth;     // wall_clock64 accumulation depth
    float tanh_scale;    // pre-tanh scaling
};

__constant__ RCConfig d_cfg;

// ============================================================================
// RESERVOIR KERNEL — configurable noise + recurrence
// ============================================================================
__global__ void reservoir_kernel(
    float* state_out,          // [N_NEURONS × STEPS]
    volatile float* cu_xchg,   // [N_NEURONS]
    const float* input_seq,    // [STEPS]
    int steps, int use_noise
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= THREADS) return;
    int nid = bid * THREADS + tid;

    __shared__ float lds[THREADS];

    // Heterogeneous params from position + config
    uint32_t rng = nid * 2654435761u + 12345;
    rng *= 1103515245;
    float leak = d_cfg.leak_lo + (d_cfg.leak_hi - d_cfg.leak_lo) * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245;
    float input_w = d_cfg.input_w_lo + (d_cfg.input_w_hi - d_cfg.input_w_lo) * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245;
    int ln1 = (tid + 1 + ((rng>>8)&7)) % THREADS;
    rng *= 1103515245;
    int ln2 = (tid + THREADS - 1 - ((rng>>8)&7)) % THREADS;
    rng *= 1103515245;
    float wl = d_cfg.rec_w * (0.3f + 0.7f * ((rng>>16)&0xFF)/255.0f);
    rng *= 1103515245;
    int rcu = ((rng>>8)&0xFF) % N_CUS;
    rng *= 1103515245;
    int rtid = ((rng>>8)&0xFF) % THREADS;
    rng *= 1103515245;
    float wr = d_cfg.cross_w * (0.3f + 0.7f * ((rng>>16)&0xFF)/255.0f);

    float v = 0.0f;
    float dummy = 1.0001f + nid * 0.0001f;

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];

        // ─── PLL noise (only if use_noise) ───
        float noise = 0.0f;
        if (use_noise) {
            float noise_sum = 0.0f;
            int depth = d_cfg.pll_depth;
            for (int r = 0; r < depth; r++) {
                uint64_t w0 = wall_clock64();
                uint64_t w1 = wall_clock64();
                noise_sum += (float)(w1 - w0);
            }
            float expected = depth * 2.2f;
            noise = (noise_sum - expected) / (sqrtf((float)depth) * 2.0f);

            // Instruction-mix modulation (subtle)
            if (u > 0.0f) {
                for (int i = 0; i < 8; i++)
                    dummy = __fmaf_rn(dummy, 1.000001f, 0.0001f);
            } else {
                uint32_t iv = __float_as_uint(dummy);
                for (int i = 0; i < 8; i++)
                    iv = iv * 2654435761u + 1;
                dummy = __uint_as_float((iv & 0x007FFFFF) | 0x3F800000);
            }
        }

        // ─── Recurrence (LDS intra-CU) ───
        lds[tid] = v;
        __syncthreads();
        float lrec = wl * (lds[ln1] + lds[ln2]);

        // ─── Recurrence (global cross-CU) ───
        cu_xchg[nid] = v;
        __threadfence();
        float rrec = wr * cu_xchg[rcu * THREADS + rtid];

        // ─── State update ───
        float pre = (1.0f - leak) * v
                   + input_w * u
                   + d_cfg.noise_gain * noise
                   + lrec + rrec;

        v = tanhf(d_cfg.tanh_scale * pre);
        state_out[nid * steps + t] = v;
        __syncthreads();
    }
    if (dummy == -9999.0f) state_out[0] = dummy;
}

// ============================================================================
// HOST
// ============================================================================
static uint32_t host_rng = 42;
float randf() {
    host_rng = host_rng * 1103515245 + 12345;
    return (float)((host_rng >> 16) & 0x7FFF) / 32767.0f;
}
float randf_signed() { return randf() * 2.0f - 1.0f; }

std::vector<float> extract_features(float* state, int n, int steps) {
    int stride = n / SAMPLE_N;
    if (stride < 1) stride = 1;
    std::vector<float> f;
    for (int si = 0; si < SAMPLE_N && si*stride < n; si++) {
        int i = si * stride;
        // Use last state value as feature (simplest, fastest)
        float sum = 0.0f;
        int cnt = 0;
        for (int t = steps - 10; t < steps; t++) {
            float v = state[i*steps+t];
            if (v!=v || v>1e20 || v<-1e20) v=0;
            sum += v;
            cnt++;
        }
        f.push_back(sum / cnt);
    }
    return f;
}

// Ridge regression: returns R² on test set
float ridge_regress(
    std::vector<std::vector<float>>& X_all,
    std::vector<float>& y_all,
    int n_train
) {
    int n = X_all.size();
    if (n < n_train + 10) return -1.0f;
    int d = X_all[0].size();
    if (d == 0) return -1.0f;

    float reg = 1.0f;

    // XᵀX + λI
    std::vector<float> XtX(d*d, 0.0f);
    std::vector<float> Xty(d, 0.0f);
    for (int i = 0; i < n_train; i++) {
        for (int a = 0; a < d; a++) {
            Xty[a] += X_all[i][a] * y_all[i];
            for (int b = a; b < d; b++) {
                XtX[a*d+b] += X_all[i][a] * X_all[i][b];
            }
        }
    }
    for (int a = 0; a < d; a++) {
        for (int b = a+1; b < d; b++) XtX[b*d+a] = XtX[a*d+b];
        XtX[a*d+a] += reg;
    }

    // Cholesky
    std::vector<float> L(d*d, 0.0f);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j <= i; j++) {
            float s = XtX[i*d+j];
            for (int k = 0; k < j; k++) s -= L[i*d+k]*L[j*d+k];
            if (i == j) {
                if (s <= 0) s = 1e-6f;
                L[i*d+j] = sqrtf(s);
            } else {
                L[i*d+j] = s / L[j*d+j];
            }
        }
    }

    // Solve L*Lᵀ*w = Xᵀy
    std::vector<float> z(d), w(d);
    for (int i = 0; i < d; i++) {
        float s = Xty[i];
        for (int k = 0; k < i; k++) s -= L[i*d+k]*z[k];
        z[i] = s / L[i*d+i];
    }
    for (int i = d-1; i >= 0; i--) {
        float s = z[i];
        for (int k = i+1; k < d; k++) s -= L[k*d+i]*w[k];
        w[i] = s / L[i*d+i];
    }

    // Test R²
    float ss_res = 0, ss_tot = 0, y_mean = 0;
    int n_test = n - n_train;
    for (int i = n_train; i < n; i++) y_mean += y_all[i];
    y_mean /= n_test;
    for (int i = n_train; i < n; i++) {
        float pred = 0;
        for (int j = 0; j < d; j++) pred += w[j] * X_all[i][j];
        float err = y_all[i] - pred;
        ss_res += err * err;
        ss_tot += (y_all[i] - y_mean) * (y_all[i] - y_mean);
    }
    if (ss_tot < 1e-10f) return 0.0f;
    return 1.0f - ss_res / ss_tot;
}

float ridge_classify_binary(
    std::vector<std::vector<float>>& X_all,
    std::vector<int>& y_all,
    int n_train
) {
    int n = X_all.size();
    std::vector<float> yf(n);
    for (int i = 0; i < n; i++) yf[i] = (float)y_all[i];
    int d = X_all[0].size();

    float reg = 1.0f;
    std::vector<float> XtX(d*d, 0.0f), Xty(d, 0.0f);
    for (int i = 0; i < n_train; i++) {
        for (int a = 0; a < d; a++) {
            Xty[a] += X_all[i][a] * yf[i];
            for (int b = a; b < d; b++)
                XtX[a*d+b] += X_all[i][a] * X_all[i][b];
        }
    }
    for (int a = 0; a < d; a++) {
        for (int b = a+1; b < d; b++) XtX[b*d+a] = XtX[a*d+b];
        XtX[a*d+a] += reg;
    }
    std::vector<float> L(d*d, 0.0f);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j <= i; j++) {
            float s = XtX[i*d+j];
            for (int k = 0; k < j; k++) s -= L[i*d+k]*L[j*d+k];
            if (i == j) { if (s<=0) s=1e-6f; L[i*d+j]=sqrtf(s); }
            else L[i*d+j] = s/L[j*d+j];
        }
    }
    std::vector<float> z(d), w(d);
    for (int i = 0; i < d; i++) {
        float s = Xty[i];
        for (int k = 0; k < i; k++) s -= L[i*d+k]*z[k];
        z[i] = s/L[i*d+i];
    }
    for (int i = d-1; i >= 0; i--) {
        float s = z[i];
        for (int k = i+1; k < d; k++) s -= L[k*d+i]*w[k];
        w[i] = s/L[i*d+i];
    }

    int correct = 0, total = 0;
    for (int i = n_train; i < n; i++) {
        float pred = 0;
        for (int j = 0; j < d; j++) pred += w[j] * X_all[i][j];
        int pc = pred > 0.5f ? 1 : 0;
        if (pc == y_all[i]) correct++;
        total++;
    }
    return (float)correct / total;
}

// Run one reservoir trial: launch kernel, collect features per timestep
void run_reservoir(
    float* d_state, volatile float* d_xchg, float* d_input,
    float* h_state, const float* h_input, int steps, int use_noise,
    std::vector<std::vector<float>>& features_out
) {
    hipMemcpy((void*)d_input, h_input, steps * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_state, 0, N_NEURONS * steps * sizeof(float));
    hipMemset((void*)d_xchg, 0, N_NEURONS * sizeof(float));

    reservoir_kernel<<<N_CUS, THREADS>>>(d_state, d_xchg, d_input, steps, use_noise);
    hipDeviceSynchronize();

    hipMemcpy(h_state, d_state, N_NEURONS * steps * sizeof(float), hipMemcpyDeviceToHost);

    features_out.clear();
    int stride = N_NEURONS / SAMPLE_N;
    if (stride < 1) stride = 1;

    for (int t = WASHOUT; t < steps; t++) {
        std::vector<float> f;
        for (int si = 0; si < SAMPLE_N && si*stride < N_NEURONS; si++) {
            int i = si * stride;
            f.push_back(h_state[i * steps + t]);
        }
        features_out.push_back(f);
    }
}

int main() {
    printf("═══ GPU-ONLY RESERVOIR: HYPERPARAMETER SWEEP ═══\n");
    printf("  768 neurons, %d steps, washout=%d, %d features\n", STEPS, WASHOUT, SAMPLE_N);
    printf("  Tasks: XOR-τ2, NARMA-5, Memory Capacity\n\n");

    // Allocate GPU memory
    float *d_state, *d_input;
    volatile float *d_xchg;
    hipMalloc(&d_state, N_NEURONS * STEPS * sizeof(float));
    hipMalloc((void**)&d_xchg, N_NEURONS * sizeof(float));
    hipMalloc(&d_input, STEPS * sizeof(float));

    float* h_state = new float[N_NEURONS * STEPS];
    float* h_input = new float[STEPS];

    // ═══ SWEEP CONFIGURATIONS ═══
    struct SweepConfig {
        const char* name;
        float noise_gain, rec_w, cross_w, leak_lo, leak_hi;
        float input_w_lo, input_w_hi;
        int pll_depth;
        float tanh_scale;
    };

    SweepConfig configs[] = {
        // Baseline (from failed benchmark)
        {"baseline",      0.10, 0.04, 0.02, 0.05, 0.20, 0.05, 0.25, 64, 1.0},
        // Low noise (noise as subtle perturbation)
        {"low_noise",     0.01, 0.04, 0.02, 0.05, 0.20, 0.10, 0.30, 32, 1.0},
        {"tiny_noise",    0.003,0.04, 0.02, 0.05, 0.20, 0.10, 0.30, 32, 1.0},
        // Strong recurrence
        {"strong_rec",    0.01, 0.10, 0.05, 0.05, 0.20, 0.10, 0.30, 32, 1.0},
        {"very_strong",   0.01, 0.15, 0.08, 0.03, 0.15, 0.15, 0.35, 32, 1.0},
        // Slow leak (more memory)
        {"slow_leak",     0.01, 0.08, 0.04, 0.01, 0.08, 0.10, 0.30, 32, 1.0},
        {"very_slow",     0.005,0.08, 0.04, 0.005,0.03, 0.10, 0.30, 32, 1.0},
        // ESN-style (spectral radius near 1)
        {"esn_edge",      0.005,0.12, 0.06, 0.01, 0.05, 0.15, 0.35, 32, 0.8},
        // Deep PLL (lots of noise, minimal recurrence)
        {"deep_pll",      0.05, 0.02, 0.01, 0.05, 0.20, 0.05, 0.15, 256, 1.0},
        // Multiplicative noise (tanh_scale modulated)
        {"mult_noise",    0.02, 0.08, 0.04, 0.02, 0.10, 0.10, 0.30, 64, 1.5},
        // Wide range leak (multi-timescale)
        {"multiscale",    0.01, 0.08, 0.04, 0.003,0.30, 0.10, 0.30, 32, 1.0},
        // Strong input, weak noise
        {"input_dom",     0.005,0.06, 0.03, 0.02, 0.10, 0.20, 0.50, 32, 1.0},
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    // Generate input sequences (same for all configs)
    host_rng = 12345;
    float input_binary[STEPS];
    for (int t = 0; t < STEPS; t++) input_binary[t] = randf() > 0.5f ? 1.0f : 0.0f;

    float input_cont[STEPS];
    host_rng = 54321;
    for (int t = 0; t < STEPS; t++) input_cont[t] = randf_signed() * 0.5f;

    // NARMA-5 target
    float narma_y[STEPS];
    memset(narma_y, 0, sizeof(narma_y));
    for (int t = 5; t < STEPS; t++) {
        narma_y[t] = 0.3f * narma_y[t-1]
                   + 0.05f * narma_y[t-1] * (narma_y[t-1] + narma_y[t-2] + narma_y[t-3] + narma_y[t-4] + narma_y[t-5])
                   + 1.5f * input_cont[t-1] * input_cont[t-5]
                   + 0.1f;
        if (narma_y[t] > 10.0f) narma_y[t] = 10.0f;
        if (narma_y[t] < -10.0f) narma_y[t] = -10.0f;
    }

    // XOR-τ2 targets
    std::vector<int> xor_target(STEPS, 0);
    for (int t = 2; t < STEPS; t++)
        xor_target[t] = ((int)input_binary[t] ^ (int)input_binary[t-2]);

    printf("%-14s | XOR-2 NOISE  CTRL  Δ    | NARMA  NOISE    CTRL    Δ       | MC    NOISE  CTRL  Δ\n", "Config");
    printf("──────────────────────────────────────────────────────────────────────────────────────────────\n");

    float best_xor = 0, best_narma = -999, best_mc = 0;
    const char* best_xor_cfg = "";
    const char* best_narma_cfg = "";
    const char* best_mc_cfg = "";

    for (int ci = 0; ci < n_configs; ci++) {
        SweepConfig& c = configs[ci];

        // Upload config
        RCConfig cfg;
        cfg.noise_gain = c.noise_gain;
        cfg.rec_w = c.rec_w;
        cfg.cross_w = c.cross_w;
        cfg.leak_lo = c.leak_lo;
        cfg.leak_hi = c.leak_hi;
        cfg.input_w_lo = c.input_w_lo;
        cfg.input_w_hi = c.input_w_hi;
        cfg.pll_depth = c.pll_depth;
        cfg.tanh_scale = c.tanh_scale;
        hipMemcpyToSymbol(HIP_SYMBOL(d_cfg), &cfg, sizeof(RCConfig));

        // ─── XOR-τ2 (3 trials, average) ───
        float xor_noise_acc = 0, xor_ctrl_acc = 0;
        int xor_trials = 3;
        for (int trial = 0; trial < xor_trials; trial++) {
            std::vector<std::vector<float>> feats_n, feats_c;
            run_reservoir(d_state, d_xchg, d_input, h_state, input_binary, STEPS, 1, feats_n);
            run_reservoir(d_state, d_xchg, d_input, h_state, input_binary, STEPS, 0, feats_c);

            int n_samples = feats_n.size();
            int n_train = n_samples * 2 / 3;

            // Build targets (aligned to washout)
            std::vector<int> y_xor;
            for (int t = WASHOUT; t < STEPS; t++) y_xor.push_back(xor_target[t]);

            xor_noise_acc += ridge_classify_binary(feats_n, y_xor, n_train);
            xor_ctrl_acc += ridge_classify_binary(feats_c, y_xor, n_train);
        }
        float xor_n = xor_noise_acc / xor_trials;
        float xor_c = xor_ctrl_acc / xor_trials;

        // ─── NARMA-5 (3 trials) ───
        float narma_noise_acc = 0, narma_ctrl_acc = 0;
        for (int trial = 0; trial < 3; trial++) {
            std::vector<std::vector<float>> feats_n, feats_c;
            run_reservoir(d_state, d_xchg, d_input, h_state, input_cont, STEPS, 1, feats_n);
            run_reservoir(d_state, d_xchg, d_input, h_state, input_cont, STEPS, 0, feats_c);

            int n_samples = feats_n.size();
            int n_train = n_samples * 2 / 3;

            std::vector<float> y_narma;
            for (int t = WASHOUT; t < STEPS; t++) y_narma.push_back(narma_y[t]);

            narma_noise_acc += ridge_regress(feats_n, y_narma, n_train);
            narma_ctrl_acc += ridge_regress(feats_c, y_narma, n_train);
        }
        float nrm_n = narma_noise_acc / 3;
        float nrm_c = narma_ctrl_acc / 3;

        // ─── Memory Capacity (delays 1-10) ───
        float mc_noise = 0, mc_ctrl = 0;
        {
            std::vector<std::vector<float>> feats_n, feats_c;
            run_reservoir(d_state, d_xchg, d_input, h_state, input_cont, STEPS, 1, feats_n);
            run_reservoir(d_state, d_xchg, d_input, h_state, input_cont, STEPS, 0, feats_c);

            int n_samples = feats_n.size();
            int n_train = n_samples * 2 / 3;

            for (int d = 1; d <= 10; d++) {
                std::vector<float> y_mc;
                for (int t = WASHOUT; t < STEPS; t++) {
                    int src = t - d;
                    y_mc.push_back(src >= 0 ? input_cont[src] : 0.0f);
                }
                float r2_n = ridge_regress(feats_n, y_mc, n_train);
                float r2_c = ridge_regress(feats_c, y_mc, n_train);
                if (r2_n > 0) mc_noise += r2_n;
                if (r2_c > 0) mc_ctrl += r2_c;
            }
        }

        float xor_delta = (xor_n - xor_c) * 100;
        float narma_delta = nrm_n - nrm_c;
        float mc_delta = mc_noise - mc_ctrl;

        printf("%-14s | %5.1f%% %5.1f%% %+5.1f | R²=%+6.3f R²=%+6.3f %+6.3f  | %5.2f %5.2f %+5.2f",
               c.name,
               xor_n*100, xor_c*100, xor_delta,
               nrm_n, nrm_c, narma_delta,
               mc_noise, mc_ctrl, mc_delta);

        // Track bests
        int wins = 0;
        if (xor_n > xor_c + 0.005) wins++;
        if (nrm_n > nrm_c + 0.005) wins++;
        if (mc_noise > mc_ctrl + 0.05) wins++;

        if (wins >= 2) printf("  ★★");
        else if (wins >= 1) printf("  ★");
        printf("\n");

        if (xor_n > best_xor) { best_xor = xor_n; best_xor_cfg = c.name; }
        if (nrm_n > best_narma) { best_narma = nrm_n; best_narma_cfg = c.name; }
        if (mc_noise > best_mc) { best_mc = mc_noise; best_mc_cfg = c.name; }
    }

    printf("\n═══ BEST CONFIGS ═══\n");
    printf("  XOR-2:   %5.1f%% (%s)\n", best_xor*100, best_xor_cfg);
    printf("  NARMA-5: R²=%+.3f (%s)\n", best_narma, best_narma_cfg);
    printf("  MC:      %.2f (%s)\n", best_mc, best_mc_cfg);

    // ═══ PHASE 2: Best config → full benchmark ═══
    printf("\n═══ PHASE 2: FULL BENCHMARK WITH BEST CONFIG ═══\n");

    // Find config with most wins
    // Re-run with best overall config for all tasks

    // Clean up
    hipFree(d_state);
    hipFree((void*)d_xchg);
    hipFree(d_input);
    delete[] h_state;
    delete[] h_input;

    printf("\nDone.\n");
    return 0;
}
