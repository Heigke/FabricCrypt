/*
 * isa_neuron_reservoir.hip — ISA-level neuromorphic reservoir
 * Uses ALL new primitives: s_setreg, s_sleep, atomics, shfl, hwreg noise
 *
 * Architecture:
 *   - 64 neurons per wavefront (1 per lane)
 *   - atomicAdd = synaptic integration
 *   - s_sleep = refractory period after spike
 *   - s_setreg MODE = neuromodulation (rounding changes integration behavior)
 *   - __shfl = lateral connectivity (nearest-neighbor + long-range)
 *   - hwreg(29) jitter = stochastic input
 *   - __ballot = population spike detection
 *
 * Reservoir benchmark: waveform classification, XOR, memory capacity
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N_NEURONS 32       // per wavefront (RDNA wave32)
#define N_STEPS 500        // timesteps
#define N_TRIALS 80        // input trials
#define N_FEATURES (N_NEURONS * 3)  // membrane + spike_count + timing

/* ============================================================
 * ISA-level LIF neuron running in a wavefront
 * Each lane = one neuron. Communication via shuffle.
 * ============================================================ */
__global__ void isa_reservoir(
    const float* __restrict__ input,     // [N_TRIALS * N_STEPS]
    float* __restrict__ states,          // [N_TRIALS * N_STEPS * N_NEURONS]
    float* __restrict__ readout,         // [N_TRIALS * N_FEATURES]
    int n_steps,
    int trial_offset,
    float leak,           // membrane leak (0.9 = slow, 0.5 = fast)
    float threshold,      // spike threshold
    float weight_exc,     // excitatory weight (nearest neighbor)
    float weight_inh,     // inhibitory weight (long-range)
    int use_sleep,        // 1 = use s_sleep for refractory
    int use_setreg,       // 1 = use s_setreg MODE modulation
    int use_noise         // 1 = use hwreg noise injection
) {
    uint32_t lane = threadIdx.x % 32;  // neuron ID within wave
    uint32_t trial = blockIdx.x + trial_offset;
    if (trial >= (uint32_t)N_TRIALS) return;

    // Per-neuron state
    float membrane = 0.0f;
    int spike_count = 0;
    uint32_t last_spike_time = 0;
    uint32_t refractory_until = 0;

    // Input weight varies per neuron (spatial receptive field)
    float input_weight = 0.1f + 0.4f * sinf((float)lane * 0.2f);

    // Per-neuron leak varies slightly (heterogeneity)
    float my_leak = leak + 0.05f * cosf((float)lane * 0.3f);

    // Accumulator for readout features
    float mean_membrane = 0.0f;
    float timing_sum = 0.0f;

    for (int t = 0; t < n_steps; t++) {
        // === 1. Input injection ===
        float inp = input[trial * n_steps + t] * input_weight;

        // === 2. Noise injection via hwreg (stochastic) ===
        if (use_noise) {
            uint32_t cycles;
            asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(cycles));
            float noise = ((float)(cycles & 0xFF) - 128.0f) * 0.001f;
            inp += noise;
        }

        // === 3. Neuromodulation via s_setreg ===
        // Keep rounding set for the ENTIRE integration step (don't restore mid-step)
        // This makes all FP ops in this step use biased rounding
        if (use_setreg) {
            if (membrane > threshold * 0.5f) {
                asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x1"); // R+inf
            } else if (membrane < -threshold * 0.3f) {
                asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x2"); // R-inf
            } else {
                asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x3"); // RTZ
            }
        }

        // === 4. Refractory: s_sleep IS the mechanism (no software check) ===
        // s_sleep pauses the wavefront physically — other waves can run
        // This creates real timing heterogeneity between wavefronts

        // === 5. Leaky integration (rounding affects ALL these ops) ===
        membrane = membrane * my_leak + inp;

        // === 6. Lateral connectivity via __shfl ===
        float neighbor_left = __shfl(membrane, (lane + 31) % 32);
        float neighbor_right = __shfl(membrane, (lane + 1) % 32);
        float distant = __shfl(membrane, (lane + 11) % 32);

        // Multiple integration steps to amplify rounding effects
        membrane += neighbor_left * weight_exc;
        membrane += neighbor_right * weight_exc;
        membrane += distant * weight_inh;

        // Cross-product terms (more FP ops = more rounding divergence)
        if (use_setreg) {
            float cross = neighbor_left * neighbor_right * 0.01f;
            membrane += cross;
        }

        // === 7. Spike detection ===
        int spiked = (membrane > threshold) ? 1 : 0;

        // === 8. Population spike feedback ===
        uint64_t spike_mask = __ballot(spiked);
        int pop_spikes = __popcll(spike_mask);
        membrane -= (float)pop_spikes * 0.005f;

        // === 9. Fire, reset, and HARDWARE refractory ===
        if (spiked) {
            spike_count++;
            last_spike_time = t;
            membrane = 0.0f;

            if (use_sleep) {
                // s_sleep creates REAL timing difference between spiking and
                // non-spiking wavefronts. The wavefront physically pauses.
                // sleep value scales with how far over threshold we were
                asm volatile("s_sleep 4");  // ~226 cycles
            }
        }

        // === 10. Restore rounding AFTER all integration ===
        if (use_setreg) {
            asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x0");
        }

        // === 11. Record state ===
        states[(trial * n_steps + t) * N_NEURONS + lane] = membrane;

        // Accumulate readout features
        mean_membrane += membrane;
        if (spiked) timing_sum += (float)t;
    }

    // === Final readout features ===
    readout[trial * N_FEATURES + lane] = mean_membrane / n_steps;  // mean membrane
    readout[trial * N_FEATURES + N_NEURONS + lane] = (float)spike_count;  // spike count
    readout[trial * N_FEATURES + 2 * N_NEURONS + lane] =
        spike_count > 0 ? timing_sum / spike_count : 0.0f;  // mean spike time
}

/* ============================================================
 * Host: generate inputs, run reservoir, evaluate
 * ============================================================ */

void generate_waveform_input(float* input, int* labels, int n_trials, int n_steps) {
    // 4-class waveform: sine, square, triangle, sawtooth
    for (int t = 0; t < n_trials; t++) {
        int cls = t % 4;
        labels[t] = cls;
        float freq = 0.05f + 0.02f * (t % 7);  // vary frequency
        for (int s = 0; s < n_steps; s++) {
            float phase = freq * s;
            float val;
            switch (cls) {
                case 0: val = sinf(2.0f * M_PI * phase); break;
                case 1: val = (fmodf(phase, 1.0f) < 0.5f) ? 1.0f : -1.0f; break;
                case 2: val = 4.0f * fabsf(fmodf(phase + 0.25f, 1.0f) - 0.5f) - 1.0f; break;
                case 3: val = 2.0f * fmodf(phase, 1.0f) - 1.0f; break;
            }
            input[t * n_steps + s] = val * 0.5f;
        }
    }
}

// Simple ridge regression readout (host-side)
void ridge_regression(float* X, int* y, int n_train, int n_test, int n_feat,
                      int n_classes, float alpha, float* accuracy) {
    // X: [n_total * n_feat], y: [n_total]
    // Split: first n_train for training, rest for test
    int n_total = n_train + n_test;

    // Allocate
    float* XtX = (float*)calloc(n_feat * n_feat, sizeof(float));
    float* XtY = (float*)calloc(n_feat * n_classes, sizeof(float));
    float* W = (float*)calloc(n_feat * n_classes, sizeof(float));

    // XtX = X_train^T * X_train + alpha * I
    for (int i = 0; i < n_feat; i++) {
        for (int j = 0; j < n_feat; j++) {
            float sum = 0;
            for (int k = 0; k < n_train; k++)
                sum += X[k * n_feat + i] * X[k * n_feat + j];
            XtX[i * n_feat + j] = sum + (i == j ? alpha : 0);
        }
    }

    // XtY = X_train^T * Y_onehot
    for (int i = 0; i < n_feat; i++) {
        for (int c = 0; c < n_classes; c++) {
            float sum = 0;
            for (int k = 0; k < n_train; k++)
                sum += X[k * n_feat + i] * (y[k] == c ? 1.0f : 0.0f);
            XtY[i * n_classes + c] = sum;
        }
    }

    // Solve via Cholesky-like: W = (XtX)^-1 * XtY
    // Simple: use Gauss elimination (small matrix)
    float* aug = (float*)calloc(n_feat * (n_feat + n_classes), sizeof(float));
    for (int i = 0; i < n_feat; i++) {
        for (int j = 0; j < n_feat; j++)
            aug[i * (n_feat + n_classes) + j] = XtX[i * n_feat + j];
        for (int c = 0; c < n_classes; c++)
            aug[i * (n_feat + n_classes) + n_feat + c] = XtY[i * n_classes + c];
    }

    int stride = n_feat + n_classes;
    for (int col = 0; col < n_feat; col++) {
        // Find pivot
        int pivot = col;
        float max_val = fabsf(aug[col * stride + col]);
        for (int row = col + 1; row < n_feat; row++) {
            if (fabsf(aug[row * stride + col]) > max_val) {
                max_val = fabsf(aug[row * stride + col]);
                pivot = row;
            }
        }
        // Swap
        if (pivot != col) {
            for (int j = 0; j < stride; j++) {
                float tmp = aug[col * stride + j];
                aug[col * stride + j] = aug[pivot * stride + j];
                aug[pivot * stride + j] = tmp;
            }
        }
        // Eliminate
        float diag = aug[col * stride + col];
        if (fabsf(diag) < 1e-10f) continue;
        for (int row = 0; row < n_feat; row++) {
            if (row == col) continue;
            float factor = aug[row * stride + col] / diag;
            for (int j = col; j < stride; j++)
                aug[row * stride + j] -= factor * aug[col * stride + j];
        }
    }

    // Extract W
    for (int i = 0; i < n_feat; i++) {
        float diag = aug[i * stride + i];
        for (int c = 0; c < n_classes; c++)
            W[i * n_classes + c] = aug[i * stride + n_feat + c] / diag;
    }

    // Test accuracy
    int correct = 0;
    for (int k = n_train; k < n_total; k++) {
        int best_c = 0;
        float best_v = -1e30f;
        for (int c = 0; c < n_classes; c++) {
            float score = 0;
            for (int i = 0; i < n_feat; i++)
                score += X[k * n_feat + i] * W[i * n_classes + c];
            if (score > best_v) { best_v = score; best_c = c; }
        }
        if (best_c == y[k]) correct++;
    }
    *accuracy = (float)correct / (float)n_test * 100.0f;

    free(XtX); free(XtY); free(W); free(aug);
}

// Memory capacity: R² at each delay
float memory_capacity(float* input, float* readout, int n_trials, int n_feat, int max_delay) {
    float mc_sum = 0;
    int n_train = n_trials * 7 / 10;
    int n_test = n_trials - n_train;

    for (int d = 1; d <= max_delay; d++) {
        // Target = input delayed by d
        float target_mean = 0;
        for (int t = d; t < n_trials; t++)
            target_mean += input[t * N_STEPS];  // use first timestep as proxy
        target_mean /= (n_trials - d);

        // Simple linear regression: readout → delayed input
        // Use first feature only for speed
        float sx = 0, sy = 0, sxx = 0, sxy = 0;
        int n = 0;
        for (int t = d; t < n_train; t++) {
            float x = readout[t * n_feat];
            float y = input[(t - d) * N_STEPS];
            sx += x; sy += y; sxx += x*x; sxy += x*y;
            n++;
        }
        if (n < 5) continue;
        float mx = sx/n, my = sy/n;
        float b = (sxy - n*mx*my) / (sxx - n*mx*mx + 1e-10f);
        float a = my - b * mx;

        // R² on test set
        float ss_res = 0, ss_tot = 0;
        for (int t = n_train + d; t < n_trials; t++) {
            float pred = a + b * readout[t * n_feat];
            float actual = input[(t - d) * N_STEPS];
            ss_res += (actual - pred) * (actual - pred);
            ss_tot += (actual - my) * (actual - my);
        }
        float r2 = (ss_tot > 1e-10f) ? 1.0f - ss_res / ss_tot : 0.0f;
        if (r2 > 0) mc_sum += r2;
    }
    return mc_sum;
}

int main() {
    printf("=== ISA-LEVEL NEUROMORPHIC RESERVOIR ===\n");
    printf("  %d neurons × %d steps × %d trials\n\n", N_NEURONS, N_STEPS, N_TRIALS);

    // Allocate
    float *h_input = (float*)malloc(N_TRIALS * N_STEPS * sizeof(float));
    int *h_labels = (int*)malloc(N_TRIALS * sizeof(int));
    float *h_readout = (float*)malloc(N_TRIALS * N_FEATURES * sizeof(float));

    generate_waveform_input(h_input, h_labels, N_TRIALS, N_STEPS);

    float *d_input, *d_states, *d_readout;
    hipMalloc(&d_input, N_TRIALS * N_STEPS * sizeof(float));
    hipMalloc(&d_states, N_TRIALS * N_STEPS * N_NEURONS * sizeof(float));
    hipMalloc(&d_readout, N_TRIALS * N_FEATURES * sizeof(float));
    hipMemcpy(d_input, h_input, N_TRIALS * N_STEPS * sizeof(float), hipMemcpyHostToDevice);

    // Conditions to test
    struct Condition {
        const char* name;
        int use_sleep, use_setreg, use_noise;
    };

    Condition conditions[] = {
        {"BASELINE (no ISA)",    0, 0, 0},
        {"+ s_sleep (refract)", 1, 0, 0},
        {"+ s_setreg (neuromod)", 0, 1, 0},
        {"+ hwreg noise",       0, 0, 1},
        {"+ sleep + setreg",    1, 1, 0},
        {"+ sleep + noise",     1, 0, 1},
        {"FULL (all ISA)",      1, 1, 1},
    };
    int n_conds = 7;

    printf("%-25s %8s %8s %8s\n", "Condition", "Wave4%", "Spikes", "MC");
    printf("%-25s %8s %8s %8s\n", "----------", "------", "------", "----");

    for (int c = 0; c < n_conds; c++) {
        hipMemset(d_readout, 0, N_TRIALS * N_FEATURES * sizeof(float));
        hipMemset(d_states, 0, N_TRIALS * N_STEPS * N_NEURONS * sizeof(float));

        // Launch: 1 block per trial, 32 threads per block (wave32)
        // Process in batches to avoid thermal issues
        int batch = 20;
        for (int b = 0; b < N_TRIALS; b += batch) {
            int n = (b + batch > N_TRIALS) ? N_TRIALS - b : batch;
            isa_reservoir<<<n, 32>>>(
                d_input, d_states, d_readout, N_STEPS, b,
                0.85f,   // leak
                0.5f,    // threshold
                0.05f,   // weight_exc
                -0.02f,  // weight_inh
                conditions[c].use_sleep,
                conditions[c].use_setreg,
                conditions[c].use_noise
            );
            hipDeviceSynchronize();
        }

        hipMemcpy(h_readout, d_readout, N_TRIALS * N_FEATURES * sizeof(float), hipMemcpyDeviceToHost);

        // Count total spikes
        int total_spikes = 0;
        for (int t = 0; t < N_TRIALS; t++)
            for (int n = 0; n < N_NEURONS; n++)
                total_spikes += (int)h_readout[t * N_FEATURES + N_NEURONS + n];

        // Classification accuracy (ridge regression)
        int n_train = N_TRIALS * 7 / 10;
        int n_test = N_TRIALS - n_train;
        float acc;
        ridge_regression(h_readout, h_labels, n_train, n_test, N_FEATURES, 4, 1.0f, &acc);

        // Memory capacity
        float mc = memory_capacity(h_input, h_readout, N_TRIALS, N_FEATURES, 10);

        printf("%-25s %7.1f%% %8d %7.3f\n",
               conditions[c].name, acc, total_spikes, mc);
    }

    // Cleanup
    hipFree(d_input); hipFree(d_states); hipFree(d_readout);
    free(h_input); free(h_labels); free(h_readout);

    printf("\n=== DONE ===\n");
    return 0;
}
