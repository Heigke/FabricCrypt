/**
 * z2432: GPU Mechanism Reservoir + FPGA Temporal Integration
 *
 * PROVEN: GPU mechanisms (L1, branch, atomics, shfl, ballot) work as reservoir (z2262-z2268)
 * PROVEN: FPGA temporal products work for XOR/MC (z2296)
 * UNPROVEN: Combining both in one pipeline
 *
 * Architecture:
 *   1. GPU kernel processes input using multiple ISA mechanisms
 *   2. Each mechanism population produces a scalar output
 *   3. These outputs are the "GPU neuromorphic features"
 *   4. Features are written to host memory for FPGA injection
 *
 * The key: mechanisms DO computation that traditional matmul cannot:
 *   - L1 cache: stochastic nonlinearity from eviction patterns
 *   - __shfl: inter-lane communication (spatial gradient)
 *   - __ballot: population vote (binary feature extraction)
 *   - atomicAdd: accumulation with contention (temporal integration)
 *   - s_setreg: rounding perturbation (analog-like noise)
 *   - Branch: prediction state (temporal memory)
 *
 * Output: array of mechanism features per input sample
 * Host script then feeds these to FPGA and does combined readout.
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define N_MECHANISMS 8
#define WARP_SIZE 64

// Shared memory for atomic accumulation
__shared__ float s_accum[8];

__global__ void mechanism_reservoir(
    const float* __restrict__ input,   // [N, seq_len] — input time series
    float* __restrict__ features,       // [N, N_MECHANISMS] — output features
    int N, int seq_len
) {
    int sample = blockIdx.x;
    if (sample >= N) return;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;

    if (tid < 8) s_accum[tid] = 0.0f;
    __syncthreads();

    const float* sig = input + sample * seq_len;

    // Each thread processes a portion of the time series
    float local_sum = 0.0f;
    float local_shfl_diff = 0.0f;
    float local_ballot_count = 0.0f;
    float branch_state = 0.0f;
    float timing_var = 0.0f;

    for (int t = tid; t < seq_len; t += blockDim.x) {
        float val = sig[t];

        // Mechanism 1: Direct accumulation (baseline)
        local_sum += val;

        // Mechanism 2: Shuffle neighbor difference (spatial gradient)
        float neighbor = __shfl_xor(val, 1);
        local_shfl_diff += fabsf(val - neighbor);

        // Mechanism 3: Ballot population coding
        unsigned long long ballot = __ballot(val > 0.5f);
        local_ballot_count += (float)__popcll(ballot) / WARP_SIZE;

        // Mechanism 4: Branch prediction (temporal state)
        if (val > branch_state) {
            branch_state = branch_state * 0.9f + val * 0.1f;  // slow adapt
        } else {
            branch_state = branch_state * 0.95f + val * 0.05f;  // slower decay
        }

        // Mechanism 5: Rounding perturbation — alternate modes
        if (t % 4 == 0) {
            asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x1" ::: "memory"); // R+inf
        } else if (t % 4 == 2) {
            asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x2" ::: "memory"); // R-inf
        }
        float perturbed_sum = local_sum + val * 0.001f;  // tiny perturbation amplified by rounding
        timing_var += fabsf(perturbed_sum - local_sum - val * 0.001f);  // rounding residual
        if (t % 4 == 0 || t % 4 == 2) {
            asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x0" ::: "memory"); // back to RNE
        }

        // Mechanism 6: Atomic contention (synaptic-like accumulation)
        if (val > 0.6f && tid < 64) {
            atomicAdd(&s_accum[tid % 8], val * 0.01f);
        }
    }

    __syncthreads();

    // Mechanism 7: Cross-lane variance (population diversity)
    float lane_sum = local_sum;
    float cross_var = __shfl_xor(lane_sum, 1) - lane_sum;
    float cross_var2 = __shfl_xor(lane_sum, 2) - lane_sum;

    // Mechanism 8: Temporal derivative of ballot (rate of change)
    // Use last ballot result vs accumulated
    float ballot_derivative = local_ballot_count - (float)seq_len / blockDim.x * 0.5f;

    // Write features (thread 0 per block writes)
    if (tid == 0) {
        features[sample * N_MECHANISMS + 0] = local_sum;                     // accumulation
        features[sample * N_MECHANISMS + 1] = local_shfl_diff;               // spatial gradient
        features[sample * N_MECHANISMS + 2] = local_ballot_count;            // population vote
        features[sample * N_MECHANISMS + 3] = branch_state;                  // temporal memory
        features[sample * N_MECHANISMS + 4] = timing_var;                    // rounding residual
        features[sample * N_MECHANISMS + 5] = s_accum[0] + s_accum[1];      // atomic contention
        features[sample * N_MECHANISMS + 6] = cross_var + cross_var2;        // population diversity
        features[sample * N_MECHANISMS + 7] = ballot_derivative;             // rate of change
    }
}

int main() {
    printf("============================================================\n");
    printf("z2432: GPU MECHANISM RESERVOIR — feature extraction\n");
    printf("============================================================\n\n");

    // Generate waveform data
    int N_PER_CLASS = 300;
    int N_CLASSES = 4;
    int N = N_PER_CLASS * N_CLASSES;
    int SEQ_LEN = 100;

    float* h_input = (float*)malloc(N * SEQ_LEN * sizeof(float));
    int* h_labels = (int*)malloc(N * sizeof(int));

    srand(42);
    for (int c = 0; c < N_CLASSES; c++) {
        for (int i = 0; i < N_PER_CLASS; i++) {
            int idx = c * N_PER_CLASS + i;
            float freq = 2.0f + (rand() % 60) / 10.0f;  // 2-8 Hz
            float phase = (rand() % 628) / 100.0f;        // 0-2π
            float noise_std = 0.15f;

            for (int t = 0; t < SEQ_LEN; t++) {
                float tt = (float)t / SEQ_LEN;
                float clean;
                switch(c) {
                    case 0: clean = sinf(2*M_PI*freq*tt + phase); break;
                    case 1: clean = (sinf(2*M_PI*freq*tt + phase) > 0) ? 1.0f : -1.0f; break;
                    case 2: clean = 2.0f * fmodf(freq*tt + phase/(2*M_PI), 1.0f) - 1.0f; break;
                    case 3: clean = 2.0f * fabsf(2.0f * fmodf(freq*tt + phase/(2*M_PI), 1.0f) - 1.0f) - 1.0f; break;
                }
                float noise = noise_std * ((float)rand()/RAND_MAX * 2.0f - 1.0f);
                h_input[idx * SEQ_LEN + t] = (clean + noise) * 0.4f + 0.5f;  // normalize to [0,1]
            }
            h_labels[idx] = c;
        }
    }

    printf("Generated %d waveforms, %d timesteps, 4 classes\n", N, SEQ_LEN);

    // GPU allocation
    float *d_input, *d_features;
    hipMalloc(&d_input, N * SEQ_LEN * sizeof(float));
    hipMalloc(&d_features, N * N_MECHANISMS * sizeof(float));
    hipMemcpy(d_input, h_input, N * SEQ_LEN * sizeof(float), hipMemcpyHostToDevice);

    // Run mechanism reservoir
    printf("Running mechanism reservoir...\n");
    mechanism_reservoir<<<N, 128>>>(d_input, d_features, N, SEQ_LEN);
    hipDeviceSynchronize();

    float* h_features = (float*)malloc(N * N_MECHANISMS * sizeof(float));
    hipMemcpy(h_features, d_features, N * N_MECHANISMS * sizeof(float), hipMemcpyDeviceToHost);

    // Print feature statistics
    printf("\nMechanism features (per-class means):\n");
    const char* mech_names[] = {
        "Accumulation", "Shfl gradient", "Ballot vote",
        "Branch state", "Round residual", "Atomic contention",
        "Pop diversity", "Ballot deriv"
    };

    for (int m = 0; m < N_MECHANISMS; m++) {
        printf("  %16s:", mech_names[m]);
        for (int c = 0; c < N_CLASSES; c++) {
            float sum = 0;
            for (int i = 0; i < N_PER_CLASS; i++) {
                sum += h_features[(c * N_PER_CLASS + i) * N_MECHANISMS + m];
            }
            printf(" C%d=%.3f", c, sum / N_PER_CLASS);
        }
        printf("\n");
    }

    // Write features + labels to binary files for Python readout
    const char* base = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy";
    char path[512];

    sprintf(path, "%s/results/z2432_mechanism_features.bin", base);
    FILE* f = fopen(path, "wb");
    int n_mech = N_MECHANISMS, sl = SEQ_LEN;
    fwrite(&N, sizeof(int), 1, f);
    fwrite(&n_mech, sizeof(int), 1, f);
    fwrite(&sl, sizeof(int), 1, f);
    fwrite(h_features, sizeof(float), N * N_MECHANISMS, f);
    fwrite(h_labels, sizeof(int), N, f);
    fwrite(h_input, sizeof(float), N * SEQ_LEN, f);
    fclose(f);
    printf("\nSaved features to %s\n", path);

    // Simple ridge classification on GPU features alone
    // (Full analysis done in Python with FPGA integration)
    printf("\nQuick check: per-mechanism class separation (F-ratio):\n");
    for (int m = 0; m < N_MECHANISMS; m++) {
        // Compute F-ratio (between-class variance / within-class variance)
        float grand_mean = 0;
        float class_means[4] = {0};
        for (int c = 0; c < N_CLASSES; c++) {
            for (int i = 0; i < N_PER_CLASS; i++) {
                float v = h_features[(c*N_PER_CLASS+i)*N_MECHANISMS+m];
                class_means[c] += v;
                grand_mean += v;
            }
            class_means[c] /= N_PER_CLASS;
        }
        grand_mean /= N;

        float between = 0, within = 0;
        for (int c = 0; c < N_CLASSES; c++) {
            between += N_PER_CLASS * (class_means[c] - grand_mean) * (class_means[c] - grand_mean);
            for (int i = 0; i < N_PER_CLASS; i++) {
                float v = h_features[(c*N_PER_CLASS+i)*N_MECHANISMS+m];
                within += (v - class_means[c]) * (v - class_means[c]);
            }
        }
        float f_ratio = (between / 3.0f) / (within / (N - 4) + 1e-10f);
        printf("  %16s: F=%.2f %s\n", mech_names[m], f_ratio,
               f_ratio > 10 ? "***" : f_ratio > 3 ? "**" : f_ratio > 1 ? "*" : "");
    }

    hipFree(d_input); hipFree(d_features);
    free(h_input); free(h_labels); free(h_features);
    return 0;
}
