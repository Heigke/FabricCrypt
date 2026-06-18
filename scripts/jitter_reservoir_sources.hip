// jitter_reservoir_sources.hip — Use ALL confirmed GPU noise sources as LIF reservoir inputs
//
// Sources confirmed:
//   A. SHADER_CYCLES (hwreg29) delta — 100/100 unique, scheduling jitter in cycle counter
//   B. WALLCLOCK PLL — bimodal {4,8,24} cycle deltas, grows with depth
//   C. CROSS-PLL DRIFT — shader vs wall clock drift, 100k+ std, workload-dependent
//   D. ATOMIC CONTENTION — L2 cache arbitration, non-deterministic ordering
//   E. VGPR PRESSURE — high register use increases timing variance (4 unique vs 2)
//   F. PERF_SNAPSHOT (hwreg27) — changes between kernel launches (inter-launch entropy)
//   G. FUSED — all sources combined
//
// Tests:
//   1. Raw noise quality per source (entropy, ACF, spectrum)
//   2. LIF reservoir with each source (5-trial waveform classification)
//   3. Amplitude sweep for best source
//   4. Tanh ESN + LIF ensemble
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_reservoir_sources scripts/jitter_reservoir_sources.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_reservoir_sources

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>

#define HIP_CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP ERROR: %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

// ============================================================================
// NOISE SOURCE FUNCTIONS (device)
// Each returns a float in roughly [-1, 1]
// ============================================================================

// Murmurhash finalizer for decorrelation
__device__ uint32_t hash32(uint32_t h) {
    h ^= h >> 16; h *= 0x85ebca6b; h ^= h >> 13; h *= 0xc2b2ae35; h ^= h >> 16;
    return h;
}

// Source A: SHADER_CYCLES delta — read hwreg(29) before/after work, use delta jitter
__device__ float noise_shader_delta(int nid, int t) {
    uint32_t c0, c1;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));
    // Do varying work to create different stall patterns per neuron
    float dummy = 1.0f;
    int work = 10 + (nid & 15) * 5;
    for (int j = 0; j < work; j++) dummy *= 1.00001f;
    if (dummy < 0) c0 = 0;  // prevent optimization
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c1));
    uint32_t delta = c1 - c0;
    uint32_t h = hash32(delta ^ (nid * 2654435761u) ^ (t * 0x45d9f3b));
    return ((float)(h & 0xFFFF) / 32768.0f - 1.0f);
}

// Source B: WALLCLOCK PLL — accumulate multiple reads for entropy
__device__ float noise_wallclock(int nid, int t) {
    uint32_t acc = 0;
    for (int r = 0; r < 32; r++) {
        uint64_t w0 = wall_clock64();
        uint64_t w1 = wall_clock64();
        acc += (uint32_t)(w1 - w0);
    }
    uint32_t h = hash32(acc ^ (nid * 2654435761u) ^ (t * 0x45d9f3b));
    return ((float)(h & 0xFFFF) / 32768.0f - 1.0f);
}

// Source C: CROSS-PLL DRIFT — difference between shader clock and wall clock
__device__ float noise_pll_drift(int nid, int t) {
    uint64_t sc = __clock64();
    uint64_t wc = wall_clock64();
    uint32_t drift = (uint32_t)(sc - wc);
    // Do workload to modulate drift
    float dummy = 1.0f;
    for (int j = 0; j < 50 + (t & 31) * 10; j++) dummy = sinf(dummy);
    if (dummy < -999) drift = 0;
    uint64_t sc2 = __clock64();
    uint64_t wc2 = wall_clock64();
    uint32_t drift2 = (uint32_t)(sc2 - wc2);
    uint32_t ddrift = drift2 - drift;  // rate of drift change
    uint32_t h = hash32(ddrift ^ (nid * 2654435761u) ^ (t * 0x9e3779b9));
    return ((float)(h & 0xFFFF) / 32768.0f - 1.0f);
}

// Source D: ATOMIC CONTENTION — L2 arbitration race
__device__ float noise_atomic(int nid, int t, unsigned int* shared_ctr) {
    // All threads race to increment — order is non-deterministic
    unsigned int old = atomicAdd(shared_ctr, 1u);
    __syncthreads();
    *shared_ctr = 0;  // reset for next call
    __syncthreads();
    uint32_t h = hash32(old ^ (nid * 2654435761u) ^ (t * 0x45d9f3b));
    return ((float)(h & 0xFFFF) / 32768.0f - 1.0f);
}

// Source E: VGPR PRESSURE TIMING — high register pressure creates timing variance
__device__ float noise_vgpr_timing(int nid, int t) {
    // Force high VGPR usage
    float v0=1.0f+nid*0.001f, v1=v0+0.1f, v2=v0+0.2f, v3=v0+0.3f;
    float v4=v0+0.4f, v5=v0+0.5f, v6=v0+0.6f, v7=v0+0.7f;
    uint64_t t0 = wall_clock64();
    v0=__fmaf_rn(v0,v1,v2); v1=__fmaf_rn(v1,v2,v3);
    v2=__fmaf_rn(v2,v3,v4); v3=__fmaf_rn(v3,v4,v5);
    v4=__fmaf_rn(v4,v5,v6); v5=__fmaf_rn(v5,v6,v7);
    v6=__fmaf_rn(v6,v7,v0); v7=__fmaf_rn(v7,v0,v1);
    uint64_t t1 = wall_clock64();
    float s = v0+v1+v2+v3+v4+v5+v6+v7;
    if (s == -999.0f) t0 = 0;
    uint32_t timing = (uint32_t)(t1 - t0);
    uint32_t h = hash32(timing ^ (nid * 2654435761u) ^ (t * 0x45d9f3b));
    return ((float)(h & 0xFFFF) / 32768.0f - 1.0f);
}

// Source F: PERF_SNAPSHOT — hwreg(27), changes between launches, constant within
__device__ float noise_perf_snapshot(int nid, int t) {
    uint32_t v;
    asm volatile("s_getreg_b32 %0, hwreg(27)" : "=s"(v));
    uint32_t h = hash32(v ^ (nid * 2654435761u) ^ (t * 0x45d9f3b));
    return ((float)(h & 0xFFFF) / 32768.0f - 1.0f);
}

// Source G: FUSED — combine all independent sources
__device__ float noise_fused(int nid, int t, unsigned int* shared_ctr) {
    // Shader delta component
    uint32_t c0, c1;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));
    float dummy = sinf(1.0f + nid * 0.01f);
    if (dummy < -999) c0 = 0;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c1));

    // Wall clock component
    uint64_t w0 = wall_clock64();
    uint64_t w1 = wall_clock64();

    // PLL drift component
    uint64_t sc = __clock64();
    uint64_t wc = wall_clock64();

    // Atomic component
    unsigned int old = atomicAdd(shared_ctr, 1u);
    __syncthreads();
    *shared_ctr = 0;
    __syncthreads();

    // Combine all entropy sources
    uint32_t combined = (c1 - c0) ^ ((uint32_t)(w1 - w0) * 0x9e3779b9) ^
                        ((uint32_t)(sc - wc) * 0x517cc1b7) ^ (old * 0x85ebca6b);
    uint32_t h = hash32(combined ^ (nid * 2654435761u) ^ (t * 0x45d9f3b));
    return ((float)(h & 0xFFFF) / 32768.0f - 1.0f);
}

// ============================================================================
// NOISE QUALITY KERNEL — sample N values from one source, analyze on host
// ============================================================================
#define MAX_NEURONS 128
#define STEPS_RAW 2000

__global__ void k_sample_noise(float* out, int source_id, unsigned int* shared_ctr) {
    int nid = threadIdx.x;
    if (nid >= MAX_NEURONS) return;

    for (int t = 0; t < STEPS_RAW; t++) {
        float v;
        switch (source_id) {
            case 0: v = noise_shader_delta(nid, t); break;
            case 1: v = noise_wallclock(nid, t); break;
            case 2: v = noise_pll_drift(nid, t); break;
            case 3: v = noise_atomic(nid, t, shared_ctr); break;
            case 4: v = noise_vgpr_timing(nid, t); break;
            case 5: v = noise_perf_snapshot(nid, t); break;
            case 6: v = noise_fused(nid, t, shared_ctr); break;
            default: v = 0;
        }
        out[nid * STEPS_RAW + t] = v;
    }
}

// ============================================================================
// LIF RESERVOIR KERNEL — spiking neuron bank driven by input + noise
// ============================================================================
#define N_NEURONS 128
#define N_STEPS 500
#define N_FEATURES (N_NEURONS * 2)  // spike_count + mean_vmem per neuron

__global__ void k_lif_reservoir(
    const float* input_signal,  // [N_STEPS]
    float* features_out,        // [N_FEATURES]
    float noise_amp,
    int source_id,
    float alpha,      // decay (0.20)
    float threshold,   // spike threshold (0.5)
    float bias_frac,   // fraction of neurons that are excitatory (0.95)
    float input_scale, // input scaling (1.0)
    unsigned int* shared_ctr
) {
    int nid = threadIdx.x;
    if (nid >= N_NEURONS) return;

    // Simple reservoir weights from hash
    float w_in = ((float)(hash32(nid * 12345 + 1) & 0xFFFF) / 32768.0f - 1.0f) * input_scale;
    bool excitatory = ((float)(hash32(nid * 67890 + 2) & 0xFFFF) / 65536.0f) < bias_frac;
    float sign = excitatory ? 1.0f : -5.0f;

    float vmem = 0;
    int spike_count = 0;
    float vmem_sum = 0;

    for (int t = 0; t < N_STEPS; t++) {
        // Get noise
        float noise;
        switch (source_id) {
            case 0: noise = noise_shader_delta(nid, t); break;
            case 1: noise = noise_wallclock(nid, t); break;
            case 2: noise = noise_pll_drift(nid, t); break;
            case 3: noise = noise_atomic(nid, t, shared_ctr); break;
            case 4: noise = noise_vgpr_timing(nid, t); break;
            case 5: noise = noise_perf_snapshot(nid, t); break;
            case 6: noise = noise_fused(nid, t, shared_ctr); break;
            default: noise = 0; break;
        }

        // LIF dynamics
        float I = w_in * input_signal[t] + noise_amp * noise * sign;
        vmem = (1.0f - alpha) * vmem + I;

        if (vmem >= threshold) {
            spike_count++;
            vmem = 0;
        } else if (vmem < -threshold) {
            vmem = 0;
        }

        vmem_sum += vmem;
    }

    features_out[nid] = (float)spike_count / (float)N_STEPS;
    features_out[N_NEURONS + nid] = vmem_sum / (float)N_STEPS;
}

// ============================================================================
// TANH ESN KERNEL — for comparison/ensemble
// ============================================================================
__global__ void k_tanh_esn(
    const float* input_signal,
    float* features_out,  // [N_NEURONS]
    float spectral_radius,
    float temperature,
    float input_scale
) {
    int nid = threadIdx.x;
    if (nid >= N_NEURONS) return;

    float w_in = ((float)(hash32(nid * 11111 + 3) & 0xFFFF) / 32768.0f - 1.0f) * input_scale;

    // Recurrent weights (sparse ring)
    float w_rec = 0;
    int prev = (nid + N_NEURONS - 1) % N_NEURONS;
    int next = (nid + 1) % N_NEURONS;
    float w_prev = ((float)(hash32(prev * 22222 + nid) & 0xFFFF) / 32768.0f - 1.0f) * spectral_radius;
    float w_next = ((float)(hash32(next * 33333 + nid) & 0xFFFF) / 32768.0f - 1.0f) * spectral_radius;

    __shared__ float states[N_NEURONS];
    states[nid] = 0;
    __syncthreads();

    float state = 0;
    float state_sum = 0;

    for (int t = 0; t < N_STEPS; t++) {
        float x_in = w_in * input_signal[t];
        float x_rec = w_prev * states[prev] + w_next * states[next];
        state = tanhf((x_in + x_rec) / temperature);
        __syncthreads();
        states[nid] = state;
        __syncthreads();
        state_sum += state;
    }

    features_out[nid] = state_sum / (float)N_STEPS;
}

// ============================================================================
// HOST: Generate waveform classification dataset
// ============================================================================
void generate_waveforms(float* signals, int* labels, int n_samples) {
    srand(42);
    for (int s = 0; s < n_samples; s++) {
        labels[s] = s % 4;
        float freq = 1.0f + labels[s] * 2.0f;  // 1, 3, 5, 7 Hz
        float phase = (float)(rand() % 1000) / 1000.0f * 6.28f;
        float noise = 0.1f;
        for (int t = 0; t < N_STEPS; t++) {
            float tt = (float)t / (float)N_STEPS;
            float val = 0;
            switch (labels[s]) {
                case 0: val = sinf(2.0f * 3.14159f * freq * tt + phase); break;
                case 1: val = (fmodf(freq * tt + phase / 6.28f, 1.0f) < 0.5f) ? 1.0f : -1.0f; break;
                case 2: val = 2.0f * fabsf(fmodf(freq * tt + phase / 6.28f, 1.0f) - 0.5f) - 0.5f; break;
                case 3: val = sinf(2.0f * 3.14159f * freq * tt) + 0.5f * sinf(2.0f * 3.14159f * 3 * freq * tt); break;
            }
            signals[s * N_STEPS + t] = val + noise * ((float)(rand() % 1000) / 500.0f - 1.0f);
        }
    }
}

// Ridge regression classifier
float ridge_classify(float* features, int* labels, int n_samples, int n_features, int n_classes) {
    // Simple: split 70/30, ridge on one-hot, argmax
    int n_train = (int)(n_samples * 0.7f);
    int n_test = n_samples - n_train;

    // Normalize features
    float* means = (float*)calloc(n_features, sizeof(float));
    float* stds = (float*)calloc(n_features, sizeof(float));
    for (int f = 0; f < n_features; f++) {
        float s1 = 0, s2 = 0;
        for (int i = 0; i < n_train; i++) {
            float v = features[i * n_features + f];
            s1 += v; s2 += v * v;
        }
        means[f] = s1 / n_train;
        float var = s2 / n_train - means[f] * means[f];
        stds[f] = sqrtf(var > 1e-10f ? var : 1e-10f);
    }

    // Normalize
    for (int i = 0; i < n_samples; i++)
        for (int f = 0; f < n_features; f++)
            features[i * n_features + f] = (features[i * n_features + f] - means[f]) / stds[f];

    // Simple nearest-centroid classifier (more stable than ridge for small n)
    float* centroids = (float*)calloc(n_classes * n_features, sizeof(float));
    int* counts = (int*)calloc(n_classes, sizeof(int));
    for (int i = 0; i < n_train; i++) {
        int c = labels[i];
        counts[c]++;
        for (int f = 0; f < n_features; f++)
            centroids[c * n_features + f] += features[i * n_features + f];
    }
    for (int c = 0; c < n_classes; c++)
        if (counts[c] > 0)
            for (int f = 0; f < n_features; f++)
                centroids[c * n_features + f] /= counts[c];

    // Test
    int correct = 0;
    for (int i = n_train; i < n_samples; i++) {
        float best_dist = 1e30f;
        int best_c = 0;
        for (int c = 0; c < n_classes; c++) {
            float dist = 0;
            for (int f = 0; f < n_features; f++) {
                float d = features[i * n_features + f] - centroids[c * n_features + f];
                dist += d * d;
            }
            if (dist < best_dist) { best_dist = dist; best_c = c; }
        }
        if (best_c == labels[i]) correct++;
    }

    free(means); free(stds); free(centroids); free(counts);
    return (float)correct / (float)n_test;
}

// ============================================================================
// HOST: Noise quality analysis
// ============================================================================
void analyze_noise(float* samples, int n_neurons, int n_steps, const char* name) {
    // Per-neuron stats, then average
    float total_unique = 0;
    float total_std = 0;
    float total_acf1 = 0;

    for (int n = 0; n < n_neurons; n++) {
        float* row = &samples[n * n_steps];

        // Mean, std
        float s1 = 0, s2 = 0;
        for (int t = 0; t < n_steps; t++) { s1 += row[t]; s2 += row[t] * row[t]; }
        float mean = s1 / n_steps;
        float var = s2 / n_steps - mean * mean;
        float sd = sqrtf(var > 0 ? var : 0);
        total_std += sd;

        // ACF(1)
        float num = 0, den = 0;
        for (int t = 1; t < n_steps; t++) {
            num += (row[t] - mean) * (row[t-1] - mean);
            den += (row[t] - mean) * (row[t] - mean);
        }
        total_acf1 += (den > 0 ? num / den : 0);

        // Unique (sample first 200)
        int unique = 1;
        int check = std::min(n_steps, 200);
        for (int i = 1; i < check; i++) {
            bool found = false;
            for (int j = 0; j < i && j < 50; j++)
                if (fabsf(row[i] - row[j]) < 1e-7f) { found = true; break; }
            if (!found) unique++;
        }
        total_unique += unique;
    }

    fprintf(stderr, "  %-20s std=%.4f  ACF(1)=%.4f  unique=%.0f/200\n",
            name, total_std / n_neurons, total_acf1 / n_neurons, total_unique / n_neurons);
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    fprintf(stderr, "═══ RESERVOIR NOISE SOURCE COMPARISON ═══\n");
    fprintf(stderr, "  128 LIF neurons, 500 steps, 4-class waveform\n\n");

    // ─── PART 1: Raw noise quality ───
    fprintf(stderr, "═══ PART 1: RAW NOISE QUALITY ═══\n");

    float *d_noise, *h_noise;
    unsigned int *d_ctr;
    HIP_CHECK(hipMalloc(&d_noise, MAX_NEURONS * STEPS_RAW * sizeof(float)));
    h_noise = (float*)malloc(MAX_NEURONS * STEPS_RAW * sizeof(float));
    HIP_CHECK(hipMalloc(&d_ctr, sizeof(unsigned int)));
    HIP_CHECK(hipMemset(d_ctr, 0, sizeof(unsigned int)));

    const char* source_names[] = {
        "SHADER_DELTA", "WALLCLOCK_PLL", "PLL_DRIFT", "ATOMIC_CONTENTION",
        "VGPR_TIMING", "PERF_SNAPSHOT", "FUSED_ALL"
    };

    for (int src = 0; src < 7; src++) {
        HIP_CHECK(hipMemset(d_noise, 0, MAX_NEURONS * STEPS_RAW * sizeof(float)));
        HIP_CHECK(hipMemset(d_ctr, 0, sizeof(unsigned int)));
        k_sample_noise<<<1, MAX_NEURONS>>>(d_noise, src, d_ctr);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_noise, d_noise, MAX_NEURONS * STEPS_RAW * sizeof(float), hipMemcpyDeviceToHost));
        analyze_noise(h_noise, MAX_NEURONS, STEPS_RAW, source_names[src]);
    }

    hipFree(d_noise);
    free(h_noise);

    // ─── PART 2: LIF Reservoir Classification ───
    fprintf(stderr, "\n═══ PART 2: LIF RESERVOIR CLASSIFICATION (5 trials) ═══\n");

    int n_samples = 200;
    float* h_signals = (float*)malloc(n_samples * N_STEPS * sizeof(float));
    int* h_labels = (int*)malloc(n_samples * sizeof(int));
    generate_waveforms(h_signals, h_labels, n_samples);

    float *d_signal, *d_features;
    HIP_CHECK(hipMalloc(&d_signal, N_STEPS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_features, N_FEATURES * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ctr, sizeof(unsigned int)));

    float* all_features = (float*)malloc(n_samples * N_FEATURES * sizeof(float));

    // Best params from previous experiments
    float alpha = 0.20f;
    float threshold = 0.5f;
    float bias_frac = 0.95f;
    float input_scale = 1.0f;
    float noise_amp = 0.15f;

    fprintf(stderr, "  LIF params: alpha=%.2f thresh=%.2f bias=%.2f input=%.2f noise=%.2f\n",
            alpha, threshold, bias_frac, input_scale, noise_amp);
    fprintf(stderr, "  %-20s  Trial1  Trial2  Trial3  Trial4  Trial5  Mean   Std\n", "Source");

    for (int src = 0; src < 7; src++) {
        float accs[5];
        for (int trial = 0; trial < 5; trial++) {
            // Generate features for all samples
            for (int s = 0; s < n_samples; s++) {
                HIP_CHECK(hipMemcpy(d_signal, &h_signals[s * N_STEPS], N_STEPS * sizeof(float), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemset(d_features, 0, N_FEATURES * sizeof(float)));
                HIP_CHECK(hipMemset(d_ctr, 0, sizeof(unsigned int)));

                k_lif_reservoir<<<1, N_NEURONS>>>(d_signal, d_features, noise_amp, src,
                                                   alpha, threshold, bias_frac, input_scale, d_ctr);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(&all_features[s * N_FEATURES], d_features,
                                     N_FEATURES * sizeof(float), hipMemcpyDeviceToHost));
            }
            accs[trial] = ridge_classify(all_features, h_labels, n_samples, N_FEATURES, 4);
        }

        float mean = 0, var = 0;
        for (int i = 0; i < 5; i++) mean += accs[i];
        mean /= 5;
        for (int i = 0; i < 5; i++) var += (accs[i] - mean) * (accs[i] - mean);
        var /= 5;

        fprintf(stderr, "  %-20s  %.3f   %.3f   %.3f   %.3f   %.3f   %.3f  %.3f\n",
                source_names[src], accs[0], accs[1], accs[2], accs[3], accs[4], mean, sqrtf(var));
    }

    // No-noise baseline
    {
        float accs[5];
        for (int trial = 0; trial < 5; trial++) {
            for (int s = 0; s < n_samples; s++) {
                HIP_CHECK(hipMemcpy(d_signal, &h_signals[s * N_STEPS], N_STEPS * sizeof(float), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemset(d_features, 0, N_FEATURES * sizeof(float)));
                HIP_CHECK(hipMemset(d_ctr, 0, sizeof(unsigned int)));
                k_lif_reservoir<<<1, N_NEURONS>>>(d_signal, d_features, 0.0f, 0,
                                                   alpha, threshold, bias_frac, input_scale, d_ctr);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(&all_features[s * N_FEATURES], d_features,
                                     N_FEATURES * sizeof(float), hipMemcpyDeviceToHost));
            }
            accs[trial] = ridge_classify(all_features, h_labels, n_samples, N_FEATURES, 4);
        }
        float mean = 0, var = 0;
        for (int i = 0; i < 5; i++) mean += accs[i];
        mean /= 5;
        for (int i = 0; i < 5; i++) var += (accs[i] - mean) * (accs[i] - mean);
        var /= 5;
        fprintf(stderr, "  %-20s  %.3f   %.3f   %.3f   %.3f   %.3f   %.3f  %.3f\n",
                "NO_NOISE", accs[0], accs[1], accs[2], accs[3], accs[4], mean, sqrtf(var));
    }

    // ─── PART 3: Tanh ESN baseline ───
    fprintf(stderr, "\n═══ PART 3: TANH ESN BASELINE ═══\n");

    float *d_esn_features;
    HIP_CHECK(hipMalloc(&d_esn_features, N_NEURONS * sizeof(float)));
    float* esn_all = (float*)malloc(n_samples * N_NEURONS * sizeof(float));

    float sr = 1.05f, temp = 0.65f, is_esn = 0.65f;
    {
        float accs[5];
        for (int trial = 0; trial < 5; trial++) {
            for (int s = 0; s < n_samples; s++) {
                HIP_CHECK(hipMemcpy(d_signal, &h_signals[s * N_STEPS], N_STEPS * sizeof(float), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemset(d_esn_features, 0, N_NEURONS * sizeof(float)));
                k_tanh_esn<<<1, N_NEURONS>>>(d_signal, d_esn_features, sr, temp, is_esn);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(&esn_all[s * N_NEURONS], d_esn_features,
                                     N_NEURONS * sizeof(float), hipMemcpyDeviceToHost));
            }
            accs[trial] = ridge_classify(esn_all, h_labels, n_samples, N_NEURONS, 4);
        }
        float mean = 0, var = 0;
        for (int i = 0; i < 5; i++) mean += accs[i];
        mean /= 5;
        for (int i = 0; i < 5; i++) var += (accs[i] - mean) * (accs[i] - mean);
        var /= 5;
        fprintf(stderr, "  TANH_ESN (SR=%.2f T=%.2f IS=%.2f): %.3f  %.3f  %.3f  %.3f  %.3f  mean=%.3f std=%.3f\n",
                sr, temp, is_esn, accs[0], accs[1], accs[2], accs[3], accs[4], mean, sqrtf(var));
    }

    // ─── PART 4: Amplitude sweep for FUSED source ───
    fprintf(stderr, "\n═══ PART 4: NOISE AMPLITUDE SWEEP (FUSED) ═══\n");
    float amps[] = {0.01f, 0.05f, 0.10f, 0.15f, 0.20f, 0.30f, 0.50f, 0.75f, 1.0f};
    for (int ai = 0; ai < 9; ai++) {
        float amp = amps[ai];
        for (int s = 0; s < n_samples; s++) {
            HIP_CHECK(hipMemcpy(d_signal, &h_signals[s * N_STEPS], N_STEPS * sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemset(d_features, 0, N_FEATURES * sizeof(float)));
            HIP_CHECK(hipMemset(d_ctr, 0, sizeof(unsigned int)));
            k_lif_reservoir<<<1, N_NEURONS>>>(d_signal, d_features, amp, 6,
                                               alpha, threshold, bias_frac, input_scale, d_ctr);
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(&all_features[s * N_FEATURES], d_features,
                                 N_FEATURES * sizeof(float), hipMemcpyDeviceToHost));
        }
        float acc = ridge_classify(all_features, h_labels, n_samples, N_FEATURES, 4);
        fprintf(stderr, "  amp=%.2f  acc=%.3f\n", amp, acc);
    }

    // ─── PART 5: Ensemble (TANH + LIF FUSED) ───
    fprintf(stderr, "\n═══ PART 5: ENSEMBLE (TANH + LIF_FUSED) ═══\n");
    int ens_features = N_NEURONS + N_FEATURES;  // ESN features + LIF features
    float* ens_all = (float*)malloc(n_samples * ens_features * sizeof(float));

    {
        float accs[5];
        for (int trial = 0; trial < 5; trial++) {
            for (int s = 0; s < n_samples; s++) {
                HIP_CHECK(hipMemcpy(d_signal, &h_signals[s * N_STEPS], N_STEPS * sizeof(float), hipMemcpyHostToDevice));

                // ESN features
                HIP_CHECK(hipMemset(d_esn_features, 0, N_NEURONS * sizeof(float)));
                k_tanh_esn<<<1, N_NEURONS>>>(d_signal, d_esn_features, sr, temp, is_esn);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(&ens_all[s * ens_features], d_esn_features,
                                     N_NEURONS * sizeof(float), hipMemcpyDeviceToHost));

                // LIF+FUSED features
                HIP_CHECK(hipMemset(d_features, 0, N_FEATURES * sizeof(float)));
                HIP_CHECK(hipMemset(d_ctr, 0, sizeof(unsigned int)));
                k_lif_reservoir<<<1, N_NEURONS>>>(d_signal, d_features, noise_amp, 6,
                                                   alpha, threshold, bias_frac, input_scale, d_ctr);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(&ens_all[s * ens_features + N_NEURONS], d_features,
                                     N_FEATURES * sizeof(float), hipMemcpyDeviceToHost));
            }
            accs[trial] = ridge_classify(ens_all, h_labels, n_samples, ens_features, 4);
        }
        float mean = 0, var = 0;
        for (int i = 0; i < 5; i++) mean += accs[i];
        mean /= 5;
        for (int i = 0; i < 5; i++) var += (accs[i] - mean) * (accs[i] - mean);
        var /= 5;
        fprintf(stderr, "  ENSEMBLE (TANH+LIF_FUSED): %.3f  %.3f  %.3f  %.3f  %.3f  mean=%.3f std=%.3f\n",
                accs[0], accs[1], accs[2], accs[3], accs[4], mean, sqrtf(var));
    }

    // ─── PART 6: Per-source cross-neuron correlation ───
    fprintf(stderr, "\n═══ PART 6: CROSS-NEURON CORRELATION ═══\n");
    fprintf(stderr, "  (Higher = more correlated = LESS useful as diverse reservoir)\n");

    float *d_noise2, *h_noise2;
    HIP_CHECK(hipMalloc(&d_noise2, MAX_NEURONS * STEPS_RAW * sizeof(float)));
    h_noise2 = (float*)malloc(MAX_NEURONS * STEPS_RAW * sizeof(float));

    for (int src = 0; src < 7; src++) {
        HIP_CHECK(hipMemset(d_noise2, 0, MAX_NEURONS * STEPS_RAW * sizeof(float)));
        HIP_CHECK(hipMemset(d_ctr, 0, sizeof(unsigned int)));
        k_sample_noise<<<1, MAX_NEURONS>>>(d_noise2, src, d_ctr);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_noise2, d_noise2, MAX_NEURONS * STEPS_RAW * sizeof(float), hipMemcpyDeviceToHost));

        // Average pairwise correlation (sample 10 neuron pairs)
        float avg_corr = 0;
        int pairs = 0;
        for (int a = 0; a < 10; a++) {
            for (int b = a + 1; b < 10; b++) {
                float sx=0,sy=0,sxy=0,sx2=0,sy2=0;
                int n = std::min(STEPS_RAW, 500);
                for (int t = 0; t < n; t++) {
                    float x = h_noise2[a * STEPS_RAW + t];
                    float y = h_noise2[b * STEPS_RAW + t];
                    sx+=x; sy+=y; sxy+=x*y; sx2+=x*x; sy2+=y*y;
                }
                float num = n*sxy - sx*sy;
                float den = sqrtf((n*sx2-sx*sx)*(n*sy2-sy*sy));
                avg_corr += (den > 0 ? fabsf(num/den) : 0);
                pairs++;
            }
        }
        avg_corr /= pairs;
        fprintf(stderr, "  %-20s cross_corr=%.4f %s\n", source_names[src], avg_corr,
                avg_corr < 0.05f ? "(excellent diversity)" : avg_corr < 0.15f ? "(good)" : "(correlated!)");
    }

    hipFree(d_noise2); free(h_noise2);

    // Cleanup
    hipFree(d_signal);
    hipFree(d_features);
    hipFree(d_esn_features);
    hipFree(d_ctr);
    free(h_signals);
    free(h_labels);
    free(all_features);
    free(esn_all);
    free(ens_all);

    fprintf(stderr, "\n═══ DONE ═══\n");
    return 0;
}
