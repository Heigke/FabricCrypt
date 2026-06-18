// signal_not_noise.hip — Test if GPU analog physics carries SIGNAL, not just noise
//
// Critical tests:
//   TEST 1: RAW vs HASHED — does raw noise perform differently than hashed?
//           If identical → noise is just RNG, no signal content
//           If raw wins → hardware physics encodes something useful
//
//   TEST 2: SIGNAL MODULATION — can we encode an input INTO the noise?
//           Feed different workloads (sin, square, triangle) and measure
//           if the RAW timing output correlates with the input
//
//   TEST 3: RAND REPLACEMENT — replace hardware noise with rand()
//           If rand() matches → we're just doing stochastic resonance with RNG
//           If hardware wins → the analog substrate actually computes
//
//   TEST 4: CROSS-PLL SIGNAL — the biggest new finding
//           Feed a modulated workload, measure drift rate response
//           Is the drift response LINEAR to workload intensity?
//
//   TEST 5: INFORMATION THROUGHPUT — bits of mutual information between
//           input signal and raw hardware output, without any hashing
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/signal_not_noise scripts/signal_not_noise.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/signal_not_noise

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
// Murmurhash (for comparison — this is what DESTROYS signal)
// ============================================================================
__device__ uint32_t hash32(uint32_t h) {
    h ^= h >> 16; h *= 0x85ebca6b; h ^= h >> 13; h *= 0xc2b2ae35; h ^= h >> 16;
    return h;
}

// ============================================================================
// TEST 1 KERNELS: Raw vs Hashed shader delta
// ============================================================================

// RAW: return actual cycle delta without hashing
__global__ void k_raw_shader_delta(float* out, int n_steps) {
    int nid = threadIdx.x;
    if (nid >= 128) return;

    for (int t = 0; t < n_steps; t++) {
        uint32_t c0, c1;
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));
        // Varying work per neuron
        float dummy = 1.0f;
        int work = 10 + (nid & 15) * 5;
        for (int j = 0; j < work; j++) dummy *= 1.00001f;
        if (dummy < 0) c0 = 0;
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c1));
        uint32_t delta = c1 - c0;
        // RAW: just normalize to roughly [-1,1] without destroying structure
        out[nid * n_steps + t] = ((float)delta - 50.0f) / 50.0f;
    }
}

// HASHED: same measurement but hash-decorrelated
__global__ void k_hashed_shader_delta(float* out, int n_steps) {
    int nid = threadIdx.x;
    if (nid >= 128) return;

    for (int t = 0; t < n_steps; t++) {
        uint32_t c0, c1;
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));
        float dummy = 1.0f;
        int work = 10 + (nid & 15) * 5;
        for (int j = 0; j < work; j++) dummy *= 1.00001f;
        if (dummy < 0) c0 = 0;
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c1));
        uint32_t delta = c1 - c0;
        uint32_t h = hash32(delta ^ (nid * 2654435761u) ^ (t * 0x45d9f3b));
        out[nid * n_steps + t] = ((float)(h & 0xFFFF) / 32768.0f - 1.0f);
    }
}

// ============================================================================
// TEST 2 KERNELS: Signal modulation — does workload intensity modulate timing?
// ============================================================================

// Feed a workload whose intensity varies with input_signal[t]
// Measure RAW timing — does timing track the input?
__global__ void k_modulated_timing(float* raw_timings, const float* input_signal, int n_steps) {
    int nid = threadIdx.x;
    if (nid >= 128) return;

    for (int t = 0; t < n_steps; t++) {
        float intensity = fabsf(input_signal[t]);  // [0, 1]
        int work = (int)(10.0f + intensity * 200.0f);  // 10-210 FMA ops

        uint64_t t0 = wall_clock64();
        float dummy = 1.0f + nid * 0.001f;
        for (int j = 0; j < work; j++)
            dummy = __fmaf_rn(dummy, 1.0001f, 0.001f);
        uint64_t t1 = wall_clock64();
        if (dummy < 0) t0 = 0;

        raw_timings[nid * n_steps + t] = (float)(t1 - t0);
    }
}

// Same but with cross-PLL drift measurement
__global__ void k_modulated_drift(float* raw_drift, const float* input_signal, int n_steps) {
    int nid = threadIdx.x;
    if (nid >= 128) return;

    for (int t = 0; t < n_steps; t++) {
        float intensity = fabsf(input_signal[t]);

        uint64_t sc0 = __clock64();
        uint64_t wc0 = wall_clock64();

        // Workload proportional to input
        float dummy = 1.0f;
        int work_fma = (int)(intensity * 100.0f);
        int work_sfu = (int)((1.0f - intensity) * 50.0f);  // inverse: SFU when FMA is low
        for (int j = 0; j < work_fma; j++) dummy = __fmaf_rn(dummy, 1.0001f, 0.001f);
        for (int j = 0; j < work_sfu; j++) dummy = sinf(dummy);
        if (dummy < -999) sc0 = 0;

        uint64_t sc1 = __clock64();
        uint64_t wc1 = wall_clock64();

        // Drift delta = how much shader-vs-wall diverged during this step
        int64_t drift0 = (int64_t)(sc0 - wc0);
        int64_t drift1 = (int64_t)(sc1 - wc1);
        raw_drift[nid * n_steps + t] = (float)(drift1 - drift0);
    }
}

// Raw shader delta with modulated workload
__global__ void k_modulated_shader_delta(float* raw_deltas, const float* input_signal, int n_steps) {
    int nid = threadIdx.x;
    if (nid >= 128) return;

    for (int t = 0; t < n_steps; t++) {
        float intensity = fabsf(input_signal[t]);

        uint32_t c0, c1;
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c0));

        float dummy = 1.0f;
        int work = (int)(5.0f + intensity * 100.0f);
        for (int j = 0; j < work; j++) dummy = __fmaf_rn(dummy, 1.0001f, 0.001f);
        if (dummy < 0) c0 = 0;

        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(c1));
        raw_deltas[nid * n_steps + t] = (float)(c1 - c0);
    }
}

// ============================================================================
// TEST 3 KERNEL: Software PRNG (xorshift) — replacement test
// ============================================================================
__global__ void k_prng_noise(float* out, int n_steps, uint32_t seed_offset) {
    int nid = threadIdx.x;
    if (nid >= 128) return;

    uint32_t state = (nid + 1) * 2654435761u + seed_offset;
    for (int t = 0; t < n_steps; t++) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        out[nid * n_steps + t] = ((float)(state & 0xFFFF) / 32768.0f - 1.0f);
    }
}

// ============================================================================
// LIF reservoir (identical to previous, but takes pre-computed noise)
// ============================================================================
#define N_NEURONS 128
#define N_STEPS 500
#define N_FEATURES (N_NEURONS * 2)

__global__ void k_lif_with_precomputed_noise(
    const float* input_signal,
    const float* noise_array,  // [N_NEURONS * N_STEPS]
    float* features_out,
    float noise_amp,
    float alpha, float threshold, float bias_frac, float input_scale
) {
    int nid = threadIdx.x;
    if (nid >= N_NEURONS) return;

    float w_in = ((float)(hash32(nid * 12345 + 1) & 0xFFFF) / 32768.0f - 1.0f) * input_scale;
    bool excitatory = ((float)(hash32(nid * 67890 + 2) & 0xFFFF) / 65536.0f) < bias_frac;
    float sign = excitatory ? 1.0f : -5.0f;

    float vmem = 0;
    int spike_count = 0;
    float vmem_sum = 0;

    for (int t = 0; t < N_STEPS; t++) {
        float noise = noise_array[nid * N_STEPS + t];
        float I = w_in * input_signal[t] + noise_amp * noise * sign;
        vmem = (1.0f - alpha) * vmem + I;

        if (vmem >= threshold) { spike_count++; vmem = 0; }
        else if (vmem < -threshold) { vmem = 0; }

        vmem_sum += vmem;
    }

    features_out[nid] = (float)spike_count / (float)N_STEPS;
    features_out[N_NEURONS + nid] = vmem_sum / (float)N_STEPS;
}

// ============================================================================
// HOST UTILITIES
// ============================================================================
void generate_waveforms(float* signals, int* labels, int n_samples) {
    srand(42);
    for (int s = 0; s < n_samples; s++) {
        labels[s] = s % 4;
        float freq = 1.0f + labels[s] * 2.0f;
        float phase = (float)(rand() % 1000) / 1000.0f * 6.28f;
        for (int t = 0; t < N_STEPS; t++) {
            float tt = (float)t / (float)N_STEPS;
            float val = 0;
            switch (labels[s]) {
                case 0: val = sinf(2*3.14159f*freq*tt+phase); break;
                case 1: val = (fmodf(freq*tt+phase/6.28f,1.0f)<0.5f)?1.0f:-1.0f; break;
                case 2: val = 2*fabsf(fmodf(freq*tt+phase/6.28f,1.0f)-0.5f)-0.5f; break;
                case 3: val = sinf(2*3.14159f*freq*tt)+0.5f*sinf(6*3.14159f*freq*tt); break;
            }
            signals[s*N_STEPS+t] = val + 0.1f*((float)(rand()%1000)/500.0f-1.0f);
        }
    }
}

float nearest_centroid_classify(float* features, int* labels, int n_samples, int n_features, int n_classes) {
    int n_train = (int)(n_samples * 0.7f);
    int n_test = n_samples - n_train;

    float* means = (float*)calloc(n_features, sizeof(float));
    float* stds = (float*)calloc(n_features, sizeof(float));
    for (int f = 0; f < n_features; f++) {
        float s1=0, s2=0;
        for (int i = 0; i < n_train; i++) { float v=features[i*n_features+f]; s1+=v; s2+=v*v; }
        means[f] = s1/n_train;
        float var = s2/n_train - means[f]*means[f];
        stds[f] = sqrtf(var > 1e-10f ? var : 1e-10f);
    }
    for (int i = 0; i < n_samples; i++)
        for (int f = 0; f < n_features; f++)
            features[i*n_features+f] = (features[i*n_features+f]-means[f])/stds[f];

    float* centroids = (float*)calloc(n_classes*n_features, sizeof(float));
    int* counts = (int*)calloc(n_classes, sizeof(int));
    for (int i = 0; i < n_train; i++) {
        counts[labels[i]]++;
        for (int f = 0; f < n_features; f++)
            centroids[labels[i]*n_features+f] += features[i*n_features+f];
    }
    for (int c = 0; c < n_classes; c++)
        if (counts[c]>0)
            for (int f = 0; f < n_features; f++)
                centroids[c*n_features+f] /= counts[c];

    int correct = 0;
    for (int i = n_train; i < n_samples; i++) {
        float best=1e30f; int best_c=0;
        for (int c = 0; c < n_classes; c++) {
            float dist=0;
            for (int f = 0; f < n_features; f++) {
                float d = features[i*n_features+f]-centroids[c*n_features+f];
                dist += d*d;
            }
            if (dist<best) { best=dist; best_c=c; }
        }
        if (best_c==labels[i]) correct++;
    }
    free(means); free(stds); free(centroids); free(counts);
    return (float)correct/(float)n_test;
}

// Pearson correlation between two arrays
float pearson(float* x, float* y, int n) {
    float sx=0,sy=0,sxy=0,sx2=0,sy2=0;
    for (int i=0;i<n;i++) { sx+=x[i]; sy+=y[i]; sxy+=x[i]*y[i]; sx2+=x[i]*x[i]; sy2+=y[i]*y[i]; }
    float num = n*sxy-sx*sy;
    float den = sqrtf((n*sx2-sx*sx)*(n*sy2-sy*sy));
    return den>0 ? num/den : 0;
}

// Mutual information estimate (binned)
float mutual_info(float* x, float* y, int n, int bins) {
    // Joint histogram
    int* hist = (int*)calloc(bins*bins, sizeof(int));
    float xmin=x[0],xmax=x[0],ymin=y[0],ymax=y[0];
    for (int i=0;i<n;i++) {
        if(x[i]<xmin) xmin=x[i]; if(x[i]>xmax) xmax=x[i];
        if(y[i]<ymin) ymin=y[i]; if(y[i]>ymax) ymax=y[i];
    }
    float xrange = xmax-xmin+1e-10f;
    float yrange = ymax-ymin+1e-10f;
    for (int i=0;i<n;i++) {
        int bx = std::min((int)((x[i]-xmin)/xrange*bins), bins-1);
        int by = std::min((int)((y[i]-ymin)/yrange*bins), bins-1);
        hist[bx*bins+by]++;
    }
    // MI = sum p(x,y) log(p(x,y)/(p(x)p(y)))
    float mi = 0;
    for (int i=0;i<bins;i++) {
        int rx=0; for(int j=0;j<bins;j++) rx+=hist[i*bins+j];
        for (int j=0;j<bins;j++) {
            int cy=0; for(int k=0;k<bins;k++) cy+=hist[k*bins+j];
            if (hist[i*bins+j]>0 && rx>0 && cy>0) {
                float pxy = (float)hist[i*bins+j]/n;
                float px = (float)rx/n;
                float py = (float)cy/n;
                mi += pxy * logf(pxy/(px*py));
            }
        }
    }
    free(hist);
    return mi;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    fprintf(stderr, "═══════════════════════════════════════════════════════\n");
    fprintf(stderr, "  SIGNAL-NOT-NOISE: Is GPU physics computing or just RNG?\n");
    fprintf(stderr, "═══════════════════════════════════════════════════════\n\n");

    int n_samples = 200;
    float* h_signals = (float*)malloc(n_samples * N_STEPS * sizeof(float));
    int* h_labels = (int*)malloc(n_samples * sizeof(int));
    generate_waveforms(h_signals, h_labels, n_samples);

    float *d_signal, *d_features, *d_noise;
    HIP_CHECK(hipMalloc(&d_signal, N_STEPS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_features, N_FEATURES * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_noise, N_NEURONS * N_STEPS * sizeof(float)));

    float* all_features = (float*)malloc(n_samples * N_FEATURES * sizeof(float));

    float alpha=0.20f, threshold=0.5f, bias_frac=0.95f, input_scale=1.0f, noise_amp=0.15f;

    // ═════════════════════════════════════════════════════
    // TEST 1: RAW vs HASHED — does raw have signal structure?
    // ═════════════════════════════════════════════════════
    fprintf(stderr, "═══ TEST 1: RAW vs HASHED SHADER DELTA ═══\n");
    fprintf(stderr, "  If raw ≈ hashed → noise is just RNG, hashing doesn't matter\n");
    fprintf(stderr, "  If raw ≠ hashed → raw signal has structure that hashing destroys\n\n");

    // Test RAW noise in LIF
    {
        float accs[5];
        for (int trial = 0; trial < 5; trial++) {
            for (int s = 0; s < n_samples; s++) {
                HIP_CHECK(hipMemcpy(d_signal, &h_signals[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemset(d_noise, 0, N_NEURONS*N_STEPS*sizeof(float)));
                k_raw_shader_delta<<<1, 128>>>(d_noise, N_STEPS);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemset(d_features, 0, N_FEATURES*sizeof(float)));
                k_lif_with_precomputed_noise<<<1, N_NEURONS>>>(d_signal, d_noise, d_features,
                    noise_amp, alpha, threshold, bias_frac, input_scale);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(&all_features[s*N_FEATURES], d_features, N_FEATURES*sizeof(float), hipMemcpyDeviceToHost));
            }
            accs[trial] = nearest_centroid_classify(all_features, h_labels, n_samples, N_FEATURES, 4);
        }
        float mean=0; for(int i=0;i<5;i++) mean+=accs[i]; mean/=5;
        float var=0; for(int i=0;i<5;i++) var+=(accs[i]-mean)*(accs[i]-mean); var/=5;
        fprintf(stderr, "  RAW shader delta:    %.3f %.3f %.3f %.3f %.3f  mean=%.3f std=%.3f\n",
                accs[0],accs[1],accs[2],accs[3],accs[4], mean, sqrtf(var));
    }

    // Test HASHED noise in LIF
    {
        float accs[5];
        for (int trial = 0; trial < 5; trial++) {
            for (int s = 0; s < n_samples; s++) {
                HIP_CHECK(hipMemcpy(d_signal, &h_signals[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemset(d_noise, 0, N_NEURONS*N_STEPS*sizeof(float)));
                k_hashed_shader_delta<<<1, 128>>>(d_noise, N_STEPS);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemset(d_features, 0, N_FEATURES*sizeof(float)));
                k_lif_with_precomputed_noise<<<1, N_NEURONS>>>(d_signal, d_noise, d_features,
                    noise_amp, alpha, threshold, bias_frac, input_scale);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(&all_features[s*N_FEATURES], d_features, N_FEATURES*sizeof(float), hipMemcpyDeviceToHost));
            }
            accs[trial] = nearest_centroid_classify(all_features, h_labels, n_samples, N_FEATURES, 4);
        }
        float mean=0; for(int i=0;i<5;i++) mean+=accs[i]; mean/=5;
        float var=0; for(int i=0;i<5;i++) var+=(accs[i]-mean)*(accs[i]-mean); var/=5;
        fprintf(stderr, "  HASHED shader delta: %.3f %.3f %.3f %.3f %.3f  mean=%.3f std=%.3f\n",
                accs[0],accs[1],accs[2],accs[3],accs[4], mean, sqrtf(var));
    }

    // Test SOFTWARE PRNG in LIF
    {
        float accs[5];
        for (int trial = 0; trial < 5; trial++) {
            for (int s = 0; s < n_samples; s++) {
                HIP_CHECK(hipMemcpy(d_signal, &h_signals[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemset(d_noise, 0, N_NEURONS*N_STEPS*sizeof(float)));
                k_prng_noise<<<1, 128>>>(d_noise, N_STEPS, trial * 1000 + s);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemset(d_features, 0, N_FEATURES*sizeof(float)));
                k_lif_with_precomputed_noise<<<1, N_NEURONS>>>(d_signal, d_noise, d_features,
                    noise_amp, alpha, threshold, bias_frac, input_scale);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(&all_features[s*N_FEATURES], d_features, N_FEATURES*sizeof(float), hipMemcpyDeviceToHost));
            }
            accs[trial] = nearest_centroid_classify(all_features, h_labels, n_samples, N_FEATURES, 4);
        }
        float mean=0; for(int i=0;i<5;i++) mean+=accs[i]; mean/=5;
        float var=0; for(int i=0;i<5;i++) var+=(accs[i]-mean)*(accs[i]-mean); var/=5;
        fprintf(stderr, "  SOFTWARE PRNG:       %.3f %.3f %.3f %.3f %.3f  mean=%.3f std=%.3f\n",
                accs[0],accs[1],accs[2],accs[3],accs[4], mean, sqrtf(var));
    }

    // No noise
    {
        float accs[5];
        for (int trial = 0; trial < 5; trial++) {
            // Zero noise
            HIP_CHECK(hipMemset(d_noise, 0, N_NEURONS*N_STEPS*sizeof(float)));
            for (int s = 0; s < n_samples; s++) {
                HIP_CHECK(hipMemcpy(d_signal, &h_signals[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemset(d_features, 0, N_FEATURES*sizeof(float)));
                k_lif_with_precomputed_noise<<<1, N_NEURONS>>>(d_signal, d_noise, d_features,
                    0.0f, alpha, threshold, bias_frac, input_scale);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(&all_features[s*N_FEATURES], d_features, N_FEATURES*sizeof(float), hipMemcpyDeviceToHost));
            }
            accs[trial] = nearest_centroid_classify(all_features, h_labels, n_samples, N_FEATURES, 4);
        }
        float mean=0; for(int i=0;i<5;i++) mean+=accs[i]; mean/=5;
        fprintf(stderr, "  NO NOISE:            %.3f %.3f %.3f %.3f %.3f  mean=%.3f\n",
                accs[0],accs[1],accs[2],accs[3],accs[4], mean);
    }

    // ═════════════════════════════════════════════════════
    // TEST 2: SIGNAL MODULATION — does workload encode input?
    // ═════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ TEST 2: INPUT→OUTPUT SIGNAL MODULATION ═══\n");
    fprintf(stderr, "  Feed input signal as workload intensity, measure if output tracks input\n\n");

    float *d_raw_out, *h_raw_out;
    int mod_steps = 500;
    HIP_CHECK(hipMalloc(&d_raw_out, 128 * mod_steps * sizeof(float)));
    h_raw_out = (float*)malloc(128 * mod_steps * sizeof(float));

    // Generate a test signal
    float* h_test_signal = (float*)malloc(mod_steps * sizeof(float));
    for (int t = 0; t < mod_steps; t++)
        h_test_signal[t] = sinf(2.0f * 3.14159f * 3.0f * (float)t / mod_steps);

    HIP_CHECK(hipMemcpy(d_signal, h_test_signal, mod_steps * sizeof(float), hipMemcpyHostToDevice));

    // Measure correlation: input → raw timing
    HIP_CHECK(hipMemset(d_raw_out, 0, 128*mod_steps*sizeof(float)));
    k_modulated_timing<<<1, 128>>>(d_raw_out, d_signal, mod_steps);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_raw_out, d_raw_out, 128*mod_steps*sizeof(float), hipMemcpyDeviceToHost));

    float avg_corr_timing = 0;
    float abs_input[500];
    for (int t = 0; t < mod_steps; t++) abs_input[t] = fabsf(h_test_signal[t]);
    for (int n = 0; n < 128; n++) {
        float r = pearson(abs_input, &h_raw_out[n*mod_steps], mod_steps);
        avg_corr_timing += fabsf(r);
    }
    avg_corr_timing /= 128;
    fprintf(stderr, "  TIMING vs |input|:  avg |r| = %.4f  %s\n", avg_corr_timing,
            avg_corr_timing > 0.3f ? "★★★ STRONG SIGNAL" :
            avg_corr_timing > 0.1f ? "★★ MODERATE SIGNAL" :
            avg_corr_timing > 0.05f ? "★ WEAK SIGNAL" : "no signal");

    // Measure: input → raw PLL drift delta
    HIP_CHECK(hipMemset(d_raw_out, 0, 128*mod_steps*sizeof(float)));
    k_modulated_drift<<<1, 128>>>(d_raw_out, d_signal, mod_steps);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_raw_out, d_raw_out, 128*mod_steps*sizeof(float), hipMemcpyDeviceToHost));

    float avg_corr_drift = 0;
    for (int n = 0; n < 128; n++) {
        float r = pearson(abs_input, &h_raw_out[n*mod_steps], mod_steps);
        avg_corr_drift += fabsf(r);
    }
    avg_corr_drift /= 128;
    fprintf(stderr, "  PLL DRIFT vs |input|: avg |r| = %.4f  %s\n", avg_corr_drift,
            avg_corr_drift > 0.3f ? "★★★ STRONG SIGNAL" :
            avg_corr_drift > 0.1f ? "★★ MODERATE SIGNAL" :
            avg_corr_drift > 0.05f ? "★ WEAK SIGNAL" : "no signal");

    // Measure: input → raw shader delta
    HIP_CHECK(hipMemset(d_raw_out, 0, 128*mod_steps*sizeof(float)));
    k_modulated_shader_delta<<<1, 128>>>(d_raw_out, d_signal, mod_steps);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_raw_out, d_raw_out, 128*mod_steps*sizeof(float), hipMemcpyDeviceToHost));

    float avg_corr_shader = 0;
    for (int n = 0; n < 128; n++) {
        float r = pearson(abs_input, &h_raw_out[n*mod_steps], mod_steps);
        avg_corr_shader += fabsf(r);
    }
    avg_corr_shader /= 128;
    fprintf(stderr, "  SHADER Δ vs |input|:  avg |r| = %.4f  %s\n", avg_corr_shader,
            avg_corr_shader > 0.3f ? "★★★ STRONG SIGNAL" :
            avg_corr_shader > 0.1f ? "★★ MODERATE SIGNAL" :
            avg_corr_shader > 0.05f ? "★ WEAK SIGNAL" : "no signal");

    // ═════════════════════════════════════════════════════
    // TEST 3: MUTUAL INFORMATION — bits of info in raw output
    // ═════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ TEST 3: MUTUAL INFORMATION (input → raw output) ═══\n");
    fprintf(stderr, "  MI > 0 means the hardware output carries information about the input\n\n");

    float mi_timing = mutual_info(abs_input, &h_raw_out[0], mod_steps, 16);
    // Recompute for timing (need to re-run)
    HIP_CHECK(hipMemset(d_raw_out, 0, 128*mod_steps*sizeof(float)));
    k_modulated_timing<<<1, 128>>>(d_raw_out, d_signal, mod_steps);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_raw_out, d_raw_out, 128*mod_steps*sizeof(float), hipMemcpyDeviceToHost));
    mi_timing = mutual_info(abs_input, &h_raw_out[0], mod_steps, 16);

    HIP_CHECK(hipMemset(d_raw_out, 0, 128*mod_steps*sizeof(float)));
    k_modulated_drift<<<1, 128>>>(d_raw_out, d_signal, mod_steps);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_raw_out, d_raw_out, 128*mod_steps*sizeof(float), hipMemcpyDeviceToHost));
    float mi_drift = mutual_info(abs_input, &h_raw_out[0], mod_steps, 16);

    HIP_CHECK(hipMemset(d_raw_out, 0, 128*mod_steps*sizeof(float)));
    k_modulated_shader_delta<<<1, 128>>>(d_raw_out, d_signal, mod_steps);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_raw_out, d_raw_out, 128*mod_steps*sizeof(float), hipMemcpyDeviceToHost));
    float mi_shader = mutual_info(abs_input, &h_raw_out[0], mod_steps, 16);

    // Control: MI between input and PRNG output (should be ~0)
    HIP_CHECK(hipMemset(d_raw_out, 0, 128*mod_steps*sizeof(float)));
    k_prng_noise<<<1, 128>>>(d_raw_out, mod_steps, 9999);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_raw_out, d_raw_out, 128*mod_steps*sizeof(float), hipMemcpyDeviceToHost));
    float mi_prng = mutual_info(abs_input, &h_raw_out[0], mod_steps, 16);

    fprintf(stderr, "  MI(input, TIMING):       %.4f bits\n", mi_timing);
    fprintf(stderr, "  MI(input, PLL_DRIFT):    %.4f bits\n", mi_drift);
    fprintf(stderr, "  MI(input, SHADER_DELTA): %.4f bits\n", mi_shader);
    fprintf(stderr, "  MI(input, PRNG):         %.4f bits  (control — should be ~0)\n", mi_prng);

    // ═════════════════════════════════════════════════════
    // TEST 4: CROSS-RUN REPRODUCIBILITY — does same input give same output?
    // ═════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ TEST 4: CROSS-RUN REPRODUCIBILITY ═══\n");
    fprintf(stderr, "  Same input twice → do raw outputs correlate?\n");
    fprintf(stderr, "  If yes → hardware is COMPUTING (deterministic function of input)\n");
    fprintf(stderr, "  If no → hardware is just RNG (irreproducible)\n\n");

    float *d_raw_out2, *h_raw_out2;
    HIP_CHECK(hipMalloc(&d_raw_out2, 128*mod_steps*sizeof(float)));
    h_raw_out2 = (float*)malloc(128*mod_steps*sizeof(float));

    // Run modulated timing twice with same input
    HIP_CHECK(hipMemset(d_raw_out, 0, 128*mod_steps*sizeof(float)));
    k_modulated_timing<<<1, 128>>>(d_raw_out, d_signal, mod_steps);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_raw_out, d_raw_out, 128*mod_steps*sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipMemset(d_raw_out2, 0, 128*mod_steps*sizeof(float)));
    k_modulated_timing<<<1, 128>>>(d_raw_out2, d_signal, mod_steps);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_raw_out2, d_raw_out2, 128*mod_steps*sizeof(float), hipMemcpyDeviceToHost));

    float avg_repro = 0;
    for (int n = 0; n < 128; n++) {
        float r = pearson(&h_raw_out[n*mod_steps], &h_raw_out2[n*mod_steps], mod_steps);
        avg_repro += r;
    }
    avg_repro /= 128;
    fprintf(stderr, "  TIMING run1 vs run2: avg r = %.4f  %s\n", avg_repro,
            avg_repro > 0.9f ? "★★★ HIGHLY REPRODUCIBLE (deterministic)" :
            avg_repro > 0.5f ? "★★ PARTIALLY REPRODUCIBLE (signal+noise)" :
            avg_repro > 0.1f ? "★ WEAKLY REPRODUCIBLE" : "IRREPRODUCIBLE (pure RNG)");

    // Same for shader delta
    HIP_CHECK(hipMemset(d_raw_out, 0, 128*mod_steps*sizeof(float)));
    k_modulated_shader_delta<<<1, 128>>>(d_raw_out, d_signal, mod_steps);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_raw_out, d_raw_out, 128*mod_steps*sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipMemset(d_raw_out2, 0, 128*mod_steps*sizeof(float)));
    k_modulated_shader_delta<<<1, 128>>>(d_raw_out2, d_signal, mod_steps);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_raw_out2, d_raw_out2, 128*mod_steps*sizeof(float), hipMemcpyDeviceToHost));

    float avg_repro_shader = 0;
    for (int n = 0; n < 128; n++) {
        float r = pearson(&h_raw_out[n*mod_steps], &h_raw_out2[n*mod_steps], mod_steps);
        avg_repro_shader += r;
    }
    avg_repro_shader /= 128;
    fprintf(stderr, "  SHADER run1 vs run2: avg r = %.4f  %s\n", avg_repro_shader,
            avg_repro_shader > 0.9f ? "★★★ HIGHLY REPRODUCIBLE (deterministic)" :
            avg_repro_shader > 0.5f ? "★★ PARTIALLY REPRODUCIBLE (signal+noise)" :
            avg_repro_shader > 0.1f ? "★ WEAKLY REPRODUCIBLE" : "IRREPRODUCIBLE (pure RNG)");

    // Same for PLL drift
    HIP_CHECK(hipMemset(d_raw_out, 0, 128*mod_steps*sizeof(float)));
    k_modulated_drift<<<1, 128>>>(d_raw_out, d_signal, mod_steps);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_raw_out, d_raw_out, 128*mod_steps*sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipMemset(d_raw_out2, 0, 128*mod_steps*sizeof(float)));
    k_modulated_drift<<<1, 128>>>(d_raw_out2, d_signal, mod_steps);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_raw_out2, d_raw_out2, 128*mod_steps*sizeof(float), hipMemcpyDeviceToHost));

    float avg_repro_drift = 0;
    for (int n = 0; n < 128; n++) {
        float r = pearson(&h_raw_out[n*mod_steps], &h_raw_out2[n*mod_steps], mod_steps);
        avg_repro_drift += r;
    }
    avg_repro_drift /= 128;
    fprintf(stderr, "  DRIFT run1 vs run2:  avg r = %.4f  %s\n", avg_repro_drift,
            avg_repro_drift > 0.9f ? "★★★ HIGHLY REPRODUCIBLE (deterministic)" :
            avg_repro_drift > 0.5f ? "★★ PARTIALLY REPRODUCIBLE (signal+noise)" :
            avg_repro_drift > 0.1f ? "★ WEAKLY REPRODUCIBLE" : "IRREPRODUCIBLE (pure RNG)");

    // ═════════════════════════════════════════════════════
    // TEST 5: LIF with MODULATED noise (signal-carrying) vs INDEPENDENT noise
    // ═════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ TEST 5: MODULATED vs INDEPENDENT NOISE IN LIF ═══\n");
    fprintf(stderr, "  If modulated wins → noise carrying signal helps computation\n");
    fprintf(stderr, "  If identical → only stochastic resonance matters, not signal content\n\n");

    // Modulated: noise is correlated with input (via workload modulation)
    {
        float accs[5];
        for (int trial = 0; trial < 5; trial++) {
            for (int s = 0; s < n_samples; s++) {
                HIP_CHECK(hipMemcpy(d_signal, &h_signals[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
                // Generate MODULATED noise (timing depends on input)
                HIP_CHECK(hipMemset(d_noise, 0, N_NEURONS*N_STEPS*sizeof(float)));
                k_modulated_timing<<<1, 128>>>(d_noise, d_signal, N_STEPS);
                HIP_CHECK(hipDeviceSynchronize());
                // Normalize noise to [-1,1] range
                // (modulated_timing outputs raw cycles ~4-60, need to normalize)
                float h_noise_tmp[N_NEURONS * N_STEPS];
                HIP_CHECK(hipMemcpy(h_noise_tmp, d_noise, N_NEURONS*N_STEPS*sizeof(float), hipMemcpyDeviceToHost));
                float nmin=1e30f, nmax=-1e30f;
                for(int i=0;i<N_NEURONS*N_STEPS;i++) { if(h_noise_tmp[i]<nmin) nmin=h_noise_tmp[i]; if(h_noise_tmp[i]>nmax) nmax=h_noise_tmp[i]; }
                float nrange = nmax-nmin+1e-10f;
                for(int i=0;i<N_NEURONS*N_STEPS;i++) h_noise_tmp[i] = (h_noise_tmp[i]-nmin)/nrange*2.0f-1.0f;
                HIP_CHECK(hipMemcpy(d_noise, h_noise_tmp, N_NEURONS*N_STEPS*sizeof(float), hipMemcpyHostToDevice));

                HIP_CHECK(hipMemset(d_features, 0, N_FEATURES*sizeof(float)));
                k_lif_with_precomputed_noise<<<1, N_NEURONS>>>(d_signal, d_noise, d_features,
                    noise_amp, alpha, threshold, bias_frac, input_scale);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(&all_features[s*N_FEATURES], d_features, N_FEATURES*sizeof(float), hipMemcpyDeviceToHost));
            }
            accs[trial] = nearest_centroid_classify(all_features, h_labels, n_samples, N_FEATURES, 4);
        }
        float mean=0; for(int i=0;i<5;i++) mean+=accs[i]; mean/=5;
        float var=0; for(int i=0;i<5;i++) var+=(accs[i]-mean)*(accs[i]-mean); var/=5;
        fprintf(stderr, "  MODULATED noise:   %.3f %.3f %.3f %.3f %.3f  mean=%.3f std=%.3f\n",
                accs[0],accs[1],accs[2],accs[3],accs[4], mean, sqrtf(var));
    }

    // Independent: noise generated independently of input
    {
        float accs[5];
        for (int trial = 0; trial < 5; trial++) {
            for (int s = 0; s < n_samples; s++) {
                HIP_CHECK(hipMemcpy(d_signal, &h_signals[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
                // Generate INDEPENDENT noise (PRNG, no correlation with input)
                HIP_CHECK(hipMemset(d_noise, 0, N_NEURONS*N_STEPS*sizeof(float)));
                k_prng_noise<<<1, 128>>>(d_noise, N_STEPS, trial*10000+s);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemset(d_features, 0, N_FEATURES*sizeof(float)));
                k_lif_with_precomputed_noise<<<1, N_NEURONS>>>(d_signal, d_noise, d_features,
                    noise_amp, alpha, threshold, bias_frac, input_scale);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(&all_features[s*N_FEATURES], d_features, N_FEATURES*sizeof(float), hipMemcpyDeviceToHost));
            }
            accs[trial] = nearest_centroid_classify(all_features, h_labels, n_samples, N_FEATURES, 4);
        }
        float mean=0; for(int i=0;i<5;i++) mean+=accs[i]; mean/=5;
        float var=0; for(int i=0;i<5;i++) var+=(accs[i]-mean)*(accs[i]-mean); var/=5;
        fprintf(stderr, "  INDEPENDENT (PRNG): %.3f %.3f %.3f %.3f %.3f  mean=%.3f std=%.3f\n",
                accs[0],accs[1],accs[2],accs[3],accs[4], mean, sqrtf(var));
    }

    // ═════════════════════════════════════════════════════
    // SUMMARY
    // ═════════════════════════════════════════════════════
    fprintf(stderr, "\n═══════════════════════════════════════════════════════\n");
    fprintf(stderr, "  VERDICT\n");
    fprintf(stderr, "═══════════════════════════════════════════════════════\n");
    fprintf(stderr, "  Input→Output correlation (timing):  r=%.4f\n", avg_corr_timing);
    fprintf(stderr, "  Input→Output correlation (drift):   r=%.4f\n", avg_corr_drift);
    fprintf(stderr, "  Input→Output correlation (shader):  r=%.4f\n", avg_corr_shader);
    fprintf(stderr, "  Cross-run reproducibility (timing):  r=%.4f\n", avg_repro);
    fprintf(stderr, "  Cross-run reproducibility (shader):  r=%.4f\n", avg_repro_shader);
    fprintf(stderr, "  Cross-run reproducibility (drift):   r=%.4f\n", avg_repro_drift);
    fprintf(stderr, "  MI(input, timing):  %.4f bits\n", mi_timing);
    fprintf(stderr, "  MI(input, drift):   %.4f bits\n", mi_drift);
    fprintf(stderr, "  MI(input, shader):  %.4f bits\n", mi_shader);
    fprintf(stderr, "  MI(input, PRNG):    %.4f bits (control)\n", mi_prng);
    fprintf(stderr, "\n");

    if (avg_corr_timing > 0.5f && avg_repro > 0.5f)
        fprintf(stderr, "  → GPU physics IS computing: input modulates output reproducibly\n");
    else if (avg_corr_timing > 0.1f)
        fprintf(stderr, "  → GPU physics carries WEAK signal: some input→output coupling\n");
    else
        fprintf(stderr, "  → GPU physics is RNG: no input→output coupling (stochastic resonance only)\n");

    // Cleanup
    hipFree(d_signal); hipFree(d_features); hipFree(d_noise);
    hipFree(d_raw_out); hipFree(d_raw_out2);
    free(h_signals); free(h_labels); free(all_features);
    free(h_raw_out); free(h_raw_out2); free(h_test_signal);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
