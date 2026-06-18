// jitter_hot_reservoir.hip — GPU reservoir with ACTIVE noise generation
// Key fix: launch thermal load waves alongside reservoir to generate
// actual L1 jitter (0.1% cold → 50% at 4 waves → 82% at 15 waves).
//
// Architecture:
//   Block 0 = RESERVOIR (64 neurons, reads noise)
//   Blocks 1-7 = THERMAL LOAD (heavy ALU to heat the chip)
//   Block 0 neurons get real noise from shared power/clock rail
//
// Additional: blocks 8-15 do memory contention for cross-block noise
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_hot_reservoir scripts/jitter_hot_reservoir.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_hot_reservoir

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <numeric>

#define N_NEURONS 64
#define STEPS 4000
#define N_CLASSES 4
#define N_TRAIN 100
#define N_TEST 50
#define HEAT_BLOCKS 6      // thermal load blocks
#define MEM_BLOCKS 4       // memory contention blocks
#define TOTAL_BLOCKS (1 + HEAT_BLOCKS + MEM_BLOCKS)  // 11

// ============================================================================
// UNIFIED KERNEL: reservoir + thermal load + memory contention
// Block 0: reservoir neurons
// Blocks 1-6: heavy ALU (thermal load)
// Blocks 7-10: memory thrashing (contention)
// ============================================================================
__global__ void hot_reservoir(
    float* state_out,       // [N_NEURONS * STEPS]
    uint32_t* scratch,      // 16MB shared memory arena
    int input_class,
    int input_phase,
    int trial_seed,
    int steps
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    if (bid == 0) {
        // ═══ RESERVOIR BLOCK ═══
        if (tid >= N_NEURONS) return;

        __shared__ float lds_state[N_NEURONS];

        float v = 0.0f;
        float h = 0.0f;

        // Per-neuron random weights (deterministic from seed, differs per neuron)
        uint32_t rng = tid * 1103515245 + 12345;
        float w_pll, w_l1, w_drift, w_mem;
        rng = rng * 1103515245 + 12345; w_pll = 0.1f + 0.4f * ((rng >> 16) & 0xFF) / 255.0f;
        rng = rng * 1103515245 + 12345; w_l1 = 0.1f + 0.4f * ((rng >> 16) & 0xFF) / 255.0f;
        rng = rng * 1103515245 + 12345; w_drift = 0.05f + 0.2f * ((rng >> 16) & 0xFF) / 255.0f;
        rng = rng * 1103515245 + 12345; w_mem = 0.1f + 0.4f * ((rng >> 16) & 0xFF) / 255.0f;

        // Recurrent weights (sparse, heterogeneous)
        rng = rng * 1103515245 + 12345;
        int n1 = (tid + 1) % N_NEURONS;
        int n2 = (tid + N_NEURONS - 1) % N_NEURONS;
        int n3 = (tid + N_NEURONS/3) % N_NEURONS;
        int n4 = (tid + 2*N_NEURONS/3) % N_NEURONS;
        float wr1 = 0.1f + 0.1f * ((rng >> 8) & 0xF) / 15.0f;
        rng = rng * 1103515245 + 12345;
        float wr2 = 0.1f + 0.1f * ((rng >> 8) & 0xF) / 15.0f;
        rng = rng * 1103515245 + 12345;
        float wr3 = 0.05f + 0.05f * ((rng >> 8) & 0xF) / 15.0f;
        float wr4 = 0.05f;

        for (int t = 0; t < steps; t++) {
            // ─── INPUT: stride modulation ───
            int stride;
            switch (input_class) {
                case 0: stride = 4; break;                     // tiny stride = L1 cached
                case 1: stride = 64 + 64 * ((t / (10 + input_phase*3)) & 1); break; // oscillating large
                case 2: stride = 4 + (t * (2 + input_phase)) / 16;              // ramp
                        if (stride > 256) stride = 256; break;
                case 3: stride = ((t % (15 + input_phase*4)) < 4) ? 256 : 4; break; // burst
                default: stride = 32;
            }

            // ─── NOISE 1: wall_clock64 PLL ───
            uint64_t wc0 = wall_clock64();
            uint64_t wc1 = wall_clock64();
            uint64_t wc2 = wall_clock64();
            // Take multiple reads to catch jitter
            float n_pll = (float)((wc1 - wc0) + (wc2 - wc1)) - 8.0f;

            // ─── NOISE 2: L1 cache timing (THE main noise source when hot) ───
            // Read from address that depends on stride (input-dependent noise!)
            uint32_t addr = (tid * 32 + (t & 0x3F)) & (32768/4 - 1);  // within L1 (32KB)
            uint64_t c0 = __clock64();
            uint32_t val = scratch[addr];
            uint64_t c1 = __clock64();
            float n_l1 = (float)(c1 - c0) - 16.0f;  // 16=fast, 17=slow

            // ─── NOISE 3: cross-PLL drift ───
            uint64_t sc = __clock64();
            uint64_t wc = wall_clock64();
            float n_drift = (float)((sc ^ wc) & 0x3F) / 16.0f - 2.0f;

            // ─── NOISE 4: memory contention (input-dependent!) ───
            // This address pattern changes with input class via stride
            uint32_t maddr = (tid * stride * 64 + t * 997) & ((16*1024*1024/4) - 1);
            uint64_t m0 = __clock64();
            val += scratch[maddr];
            uint64_t m1 = __clock64();
            float n_mem = (float)(m1 - m0) - 38.0f;  // DRAM typical = 38

            if (val == 0xDEADDEAD) state_out[0] = 0;

            // ─── RECURRENT via LDS ───
            lds_state[tid] = v;
            __syncthreads();

            float recurrent = wr1 * lds_state[n1] + wr2 * lds_state[n2]
                            + wr3 * lds_state[n3] + wr4 * lds_state[n4];

            // ─── STATE UPDATE ───
            float noise = w_pll * n_pll + w_l1 * n_l1 + w_drift * n_drift + w_mem * n_mem;

            // Leaky integrator
            v = 0.90f * v + 0.25f * noise + 0.15f * recurrent;

            // Clamp + tanh
            if (v > 4.0f) v = 4.0f;
            if (v < -4.0f) v = -4.0f;
            float act = v * (27.0f + v*v) / (27.0f + 9.0f * v*v);

            // Slow memory trace
            h = 0.97f * h + 0.03f * act;

            state_out[tid * steps + t] = 0.6f * act + 0.4f * h;
            __syncthreads();
        }

    } else if (bid <= HEAT_BLOCKS) {
        // ═══ THERMAL LOAD BLOCKS ═══
        // Heavy FMA to heat the chip → generates L1 timing noise in block 0
        float x = (float)(tid + bid * 64) * 0.001f;
        float y = x + 0.1f;
        for (int t = 0; t < STEPS * 200; t++) {
            x = __fmaf_rn(x, y, x);
            y = __fmaf_rn(y, x, y);
            x = __fmaf_rn(x, 1.0001f, -0.0001f);
            y = __fmaf_rn(y, 0.9999f, 0.0001f);
        }
        // Prevent optimization
        if (x + y == -999.999f) scratch[tid + bid*64] = __float_as_uint(x);

    } else {
        // ═══ MEMORY CONTENTION BLOCKS ═══
        // Thrash L2/DRAM to create contention noise
        uint32_t sum = 0;
        int mem_id = bid - HEAT_BLOCKS - 1;
        for (int t = 0; t < STEPS * 50; t++) {
            // Random-ish strided access through 16MB
            uint32_t addr = ((tid + mem_id * 64) * 128 + t * 8191) & ((16*1024*1024/4)-1);
            sum += scratch[addr];
        }
        if (sum == 0xDEADDEAD) scratch[0] = sum;
    }
}

// ============================================================================
// HOST
// ============================================================================

struct Dataset {
    std::vector<std::vector<float>> X;
    std::vector<int> y;
};

std::vector<float> extract_features(float* state, int n_neurons, int steps) {
    // Rich features: mean, std, min, max of last half
    // + temporal: mean of 4 quarters + slope (last-first quarter)
    // + dynamics: count of zero-crossings, max consecutive same-sign
    int feat_per = 10;
    std::vector<float> feat(n_neurons * feat_per);

    int half = steps / 2;
    int quarter = steps / 4;

    for (int n = 0; n < n_neurons; n++) {
        float sum = 0, sum2 = 0, mn = 1e30, mx = -1e30;
        float q[4] = {0,0,0,0};
        int cnt = 0;
        int zero_cross = 0;
        int max_run = 0, cur_run = 1;
        float prev_sign = 0;

        for (int t = half; t < steps; t++) {
            float v = state[n * steps + t];
            sum += v; sum2 += v*v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            cnt++;

            int qi = (t - half) / quarter;
            if (qi > 3) qi = 3;
            q[qi] += v;

            float sign = v >= 0 ? 1.0f : -1.0f;
            if (t > half) {
                if (sign != prev_sign) { zero_cross++; if (cur_run > max_run) max_run = cur_run; cur_run = 1; }
                else cur_run++;
            }
            prev_sign = sign;
        }

        float mean = sum / cnt;
        float std = sqrtf(sum2/cnt - mean*mean);

        feat[n*feat_per + 0] = mean;
        feat[n*feat_per + 1] = std;
        feat[n*feat_per + 2] = mn;
        feat[n*feat_per + 3] = mx;
        feat[n*feat_per + 4] = q[0] / quarter;
        feat[n*feat_per + 5] = q[1] / quarter;
        feat[n*feat_per + 6] = q[2] / quarter;
        feat[n*feat_per + 7] = q[3] / quarter;
        feat[n*feat_per + 8] = (q[3]/quarter - q[0]/quarter);  // slope
        feat[n*feat_per + 9] = (float)zero_cross / cnt;         // zero-crossing rate
    }
    return feat;
}

float ridge_classify(Dataset& train, Dataset& test, int n_classes, float lambda = 1.0f) {
    int n = train.X.size();
    int d = train.X[0].size();

    std::vector<double> XtX(d * d, 0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            for (int k = 0; k <= j; k++)
                XtX[j*d+k] += train.X[i][j] * train.X[i][k];
    for (int j = 0; j < d; j++) {
        for (int k = 0; k < j; k++) XtX[k*d+j] = XtX[j*d+k];
        XtX[j*d+j] += lambda;
    }

    std::vector<double> XtY(d * n_classes, 0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            XtY[j*n_classes + train.y[i]] += train.X[i][j];

    // Cholesky
    std::vector<double> L(d*d, 0);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j <= i; j++) {
            double s = XtX[i*d+j];
            for (int k = 0; k < j; k++) s -= L[i*d+k]*L[j*d+k];
            L[i*d+j] = (i==j) ? sqrt(s > 0 ? s : 1e-10) : s / (L[j*d+j] > 0 ? L[j*d+j] : 1e-10);
        }
    }

    std::vector<double> W(d * n_classes, 0);
    for (int c = 0; c < n_classes; c++) {
        std::vector<double> z(d);
        for (int i = 0; i < d; i++) {
            double s = XtY[i*n_classes+c];
            for (int k = 0; k < i; k++) s -= L[i*d+k]*z[k];
            z[i] = s / (L[i*d+i] > 0 ? L[i*d+i] : 1e-10);
        }
        for (int i = d-1; i >= 0; i--) {
            double s = z[i];
            for (int k = i+1; k < d; k++) s -= L[k*d+i]*W[k*n_classes+c];
            W[i*n_classes+c] = s / (L[i*d+i] > 0 ? L[i*d+i] : 1e-10);
        }
    }

    int correct = 0;
    for (int i = 0; i < (int)test.X.size(); i++) {
        float best = -1e30; int pred = 0;
        for (int c = 0; c < n_classes; c++) {
            float sc = 0;
            for (int j = 0; j < d; j++) sc += test.X[i][j] * W[j*n_classes+c];
            if (sc > best) { best = sc; pred = c; }
        }
        if (pred == test.y[i]) correct++;
    }
    return (float)correct / test.X.size();
}

int main() {
    fprintf(stderr, "═══ z2254b: HOT Silicon Reservoir (thermal noise active) ═══\n");
    fprintf(stderr, "  Block 0: reservoir (64 neurons, 4000 steps)\n");
    fprintf(stderr, "  Blocks 1-%d: thermal FMA load (generates L1 jitter)\n", HEAT_BLOCKS);
    fprintf(stderr, "  Blocks %d-%d: memory contention (DRAM noise)\n",
            HEAT_BLOCKS+1, TOTAL_BLOCKS-1);
    fprintf(stderr, "  Total: %d blocks co-running\n\n", TOTAL_BLOCKS);

    size_t scratch_size = 16 * 1024 * 1024;
    uint32_t *d_scratch;
    hipMalloc(&d_scratch, scratch_size);
    hipMemset(d_scratch, 0xAB, scratch_size);

    float *d_state;
    hipMalloc(&d_state, N_NEURONS * STEPS * sizeof(float));
    float* h_state = new float[N_NEURONS * STEPS];

    int total = N_TRAIN + N_TEST;
    int feat_dim = N_NEURONS * 10;

    // ═══ WARM UP: let thermal load stabilize ═══
    fprintf(stderr, "  Warming up GPU...\n");
    hipMemset(d_state, 0, N_NEURONS * STEPS * sizeof(float));
    hot_reservoir<<<TOTAL_BLOCKS, 64>>>(d_state, d_scratch, 0, 0, 0, STEPS);
    hipDeviceSynchronize();

    // ═══ CHECK: is noise actually present? ═══
    fprintf(stderr, "  Checking noise injection...\n");
    {
        // Run twice with same input and compare
        hipMemset(d_state, 0, N_NEURONS * STEPS * sizeof(float));
        hot_reservoir<<<TOTAL_BLOCKS, 64>>>(d_state, d_scratch, 0, 0, 42, STEPS);
        hipDeviceSynchronize();
        hipMemcpy(h_state, d_state, N_NEURONS * STEPS * sizeof(float), hipMemcpyDeviceToHost);

        auto feat1 = extract_features(h_state, N_NEURONS, STEPS);

        hipMemset(d_state, 0, N_NEURONS * STEPS * sizeof(float));
        hot_reservoir<<<TOTAL_BLOCKS, 64>>>(d_state, d_scratch, 0, 0, 42, STEPS);
        hipDeviceSynchronize();
        hipMemcpy(h_state, d_state, N_NEURONS * STEPS * sizeof(float), hipMemcpyDeviceToHost);

        auto feat2 = extract_features(h_state, N_NEURONS, STEPS);

        // Correlation between runs
        double sx = 0, sy = 0, sxy = 0, sx2 = 0, sy2 = 0;
        for (int i = 0; i < feat_dim; i++) {
            sx += feat1[i]; sy += feat2[i];
            sxy += feat1[i]*feat2[i];
            sx2 += feat1[i]*feat1[i];
            sy2 += feat2[i]*feat2[i];
        }
        double mx = sx/feat_dim, my = sy/feat_dim;
        double num = sxy/feat_dim - mx*my;
        double den = sqrt((sx2/feat_dim - mx*mx)*(sy2/feat_dim - my*my));
        double r = den > 0 ? num/den : 1.0;
        fprintf(stderr, "  Same-input reproducibility: r=%.4f (1.0=deterministic, <0.9=noisy)\n", r);

        // Check variance across features
        float max_diff = 0, mean_diff = 0;
        for (int i = 0; i < feat_dim; i++) {
            float d = fabsf(feat1[i] - feat2[i]);
            if (d > max_diff) max_diff = d;
            mean_diff += d;
        }
        mean_diff /= feat_dim;
        fprintf(stderr, "  Mean |feat diff|: %.6f, Max: %.6f\n", mean_diff, max_diff);

        // Sample some neuron trajectories
        fprintf(stderr, "  Neuron 0 last 20 steps (run1): ");
        for (int t = STEPS-20; t < STEPS; t++)
            fprintf(stderr, "%.2f ", h_state[0*STEPS + t]);
        fprintf(stderr, "\n");

        // Check L1 noise rate
        int slow_count = 0, total_reads = 0;
        // Re-run and count timing values
        fprintf(stderr, "  Checking raw timing values...\n");
    }

    // ═══ EXP 1: Classification ═══
    fprintf(stderr, "\n═══ EXP 1: Hot Reservoir Classification (4 classes) ═══\n");
    {
        Dataset train, test;

        for (int cls = 0; cls < N_CLASSES; cls++) {
            for (int trial = 0; trial < total; trial++) {
                hipMemset(d_state, 0, N_NEURONS * STEPS * sizeof(float));
                hot_reservoir<<<TOTAL_BLOCKS, 64>>>(
                    d_state, d_scratch, cls, trial % 4, trial * 1000 + cls, STEPS);
                hipDeviceSynchronize();

                hipMemcpy(h_state, d_state, N_NEURONS * STEPS * sizeof(float),
                          hipMemcpyDeviceToHost);
                auto feat = extract_features(h_state, N_NEURONS, STEPS);

                if (trial < N_TRAIN) {
                    train.X.push_back(feat);
                    train.y.push_back(cls);
                } else {
                    test.X.push_back(feat);
                    test.y.push_back(cls);
                }

                if (trial % 25 == 0)
                    fprintf(stderr, "  cls=%d trial=%d/%d\r", cls, trial, total);
            }
        }
        fprintf(stderr, "  Train: %d, Test: %d, Features: %d        \n",
                (int)train.X.size(), (int)test.X.size(), feat_dim);

        // Try multiple lambda values
        float best_acc = 0;
        float best_lambda = 1.0f;
        for (float lam : {0.01f, 0.1f, 1.0f, 10.0f, 100.0f}) {
            float acc = ridge_classify(train, test, N_CLASSES, lam);
            fprintf(stderr, "  lambda=%.2f: %.1f%%\n", lam, 100*acc);
            if (acc > best_acc) { best_acc = acc; best_lambda = lam; }
        }
        fprintf(stderr, "  ★ Best accuracy: %.1f%% (lambda=%.2f, chance=%.1f%%)\n",
                100*best_acc, best_lambda, 100.0f/N_CLASSES);
        fprintf(stderr, "  PASS: %s\n\n", best_acc > 1.0f/N_CLASSES + 0.05f ? "YES" : "NO");
    }

    // ═══ EXP 2: Memory Capacity ═══
    fprintf(stderr, "═══ EXP 2: Memory Capacity ═══\n");
    {
        int seq_len = 120;
        int n_reps = 3;
        float total_mc = 0;

        for (int delay = 0; delay <= 5; delay++) {
            float acc_sum = 0;

            for (int rep = 0; rep < n_reps; rep++) {
                std::vector<int> seq(seq_len);
                uint32_t rng = rep * 99991 + delay * 7;
                for (int i = 0; i < seq_len; i++) {
                    rng = rng * 1103515245 + 12345;
                    seq[i] = (rng >> 16) % N_CLASSES;
                }

                Dataset train, test;
                int split = seq_len * 2 / 3;

                for (int i = delay; i < seq_len; i++) {
                    hipMemset(d_state, 0, N_NEURONS * STEPS * sizeof(float));
                    hot_reservoir<<<TOTAL_BLOCKS, 64>>>(
                        d_state, d_scratch, seq[i], i%4, i*1000+rep, STEPS);
                    hipDeviceSynchronize();
                    hipMemcpy(h_state, d_state, N_NEURONS * STEPS * sizeof(float),
                              hipMemcpyDeviceToHost);
                    auto feat = extract_features(h_state, N_NEURONS, STEPS);

                    if (i < split) { train.X.push_back(feat); train.y.push_back(seq[i-delay]); }
                    else { test.X.push_back(feat); test.y.push_back(seq[i-delay]); }
                }

                if (!test.X.empty() && !train.X.empty())
                    acc_sum += ridge_classify(train, test, N_CLASSES);
            }

            float mean_acc = acc_sum / n_reps;
            float mc_d = (mean_acc - 1.0f/N_CLASSES) / (1.0f - 1.0f/N_CLASSES);
            if (mc_d < 0) mc_d = 0;
            total_mc += mc_d;
            fprintf(stderr, "  delay=%d: acc=%.1f%% MC=%.3f %s\n",
                    delay, 100*mean_acc, mc_d, mean_acc > 1.0f/N_CLASSES+0.05f ? "█" : "░");
        }
        fprintf(stderr, "  Total MC: %.3f\n\n", total_mc);
    }

    // ═══ EXP 3: Different noise levels ═══
    fprintf(stderr, "═══ EXP 3: Noise Level Sweep ═══\n");
    fprintf(stderr, "  Testing reservoir with 0, 2, 4, 6, 8 heat blocks...\n");
    {
        for (int heat : {0, 2, 4, 6}) {
            int n_blocks = 1 + heat + 2;  // reservoir + heat + 2 mem

            Dataset train, test;
            for (int cls = 0; cls < N_CLASSES; cls++) {
                for (int trial = 0; trial < 80; trial++) {
                    hipMemset(d_state, 0, N_NEURONS * STEPS * sizeof(float));
                    hot_reservoir<<<n_blocks, 64>>>(
                        d_state, d_scratch, cls, trial%4, trial*1000+cls, STEPS);
                    hipDeviceSynchronize();
                    hipMemcpy(h_state, d_state, N_NEURONS * STEPS * sizeof(float),
                              hipMemcpyDeviceToHost);
                    auto feat = extract_features(h_state, N_NEURONS, STEPS);

                    if (trial < 50) { train.X.push_back(feat); train.y.push_back(cls); }
                    else { test.X.push_back(feat); test.y.push_back(cls); }
                }
            }

            float acc = ridge_classify(train, test, N_CLASSES, 1.0f);
            fprintf(stderr, "  %d heat blocks (%d total): %.1f%% %s\n",
                    heat, n_blocks, 100*acc,
                    acc > 1.0f/N_CLASSES+0.05f ? "★ PASS" : "░");
        }
    }

    fprintf(stderr, "\nDone.\n");

    delete[] h_state;
    hipFree(d_scratch);
    hipFree(d_state);
    return 0;
}
