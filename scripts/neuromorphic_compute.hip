// neuromorphic_compute.hip — True neuromorphic computation using GPU analog physics
//
// KEY INSIGHT FROM signal_not_noise.hip:
//   - TIMING channel: r=0.9995 input→output, 2.03 bits MI → DETERMINISTIC analog transducer
//   - PLL DRIFT: r=0.50 reproducibility → PARTIALLY STOCHASTIC (like a real neuron!)
//   - SHADER DELTA: r=0.00 → pure RNG
//
// STRATEGY: Use the GPU's analog physics AS the computation, not just as noise:
//
//   A. ANALOG TIMING NEURONS: Each "neuron" is a workload whose execution time
//      IS the membrane potential. Input modulates workload → timing = f(input).
//      The GPU's pipeline, cache, and clock mesh become the "dendrite".
//
//   B. PLL DRIFT RESERVOIR: The partially-stochastic drift between shader and
//      wall clock PLLs creates a natural reservoir with fading memory.
//      Drift integrates over time → temporal memory without explicit state.
//
//   C. INSTRUCTION-MIX ENCODING: Different instruction types (FMA/SFU/INT)
//      activate different silicon pathways with different analog responses.
//      Input→instruction-mix = input→analog-pathway-selection = physical computation.
//
//   D. CROSS-WAVEFRONT COUPLING: Multiple wavefronts on same CU share physical
//      resources (scheduler, cache, power rail). One wavefront's activity
//      physically modulates another's timing → analog synapses.
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/neuromorphic_compute scripts/neuromorphic_compute.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/neuromorphic_compute

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

#define N_NEURONS 128
#define N_STEPS 500

__device__ uint32_t hash32(uint32_t h) {
    h ^= h >> 16; h *= 0x85ebca6b; h ^= h >> 13; h *= 0xc2b2ae35; h ^= h >> 16;
    return h;
}

// ============================================================================
// APPROACH A: ANALOG TIMING NEURON
// Each neuron's "activation" IS its execution time.
// The GPU pipeline physically computes f(input) → timing.
// Different neurons use different instruction mixes → different transfer functions.
// ============================================================================
__global__ void k_analog_timing_neuron(
    const float* input_signal,  // [N_STEPS]
    float* activations,         // [N_NEURONS * N_STEPS] — raw timing per neuron per step
    int n_steps
) {
    int nid = threadIdx.x;
    if (nid >= N_NEURONS) return;

    // Each neuron has a different instruction mix → different analog transfer function
    int neuron_type = nid % 4;  // 0=FMA, 1=SFU, 2=INT, 3=mixed
    float w_in = ((float)(hash32(nid * 12345 + 1) & 0xFFFF) / 32768.0f - 1.0f);

    for (int t = 0; t < n_steps; t++) {
        float input = input_signal[t] * w_in;
        float intensity = 0.5f + 0.5f * input;  // map to [0, 1]
        intensity = fmaxf(0.0f, fminf(1.0f, intensity));

        int work = (int)(5.0f + intensity * 150.0f);
        float dummy = 1.0f + nid * 0.001f;

        uint64_t t0 = wall_clock64();

        switch (neuron_type) {
            case 0:  // FMA-heavy pathway
                for (int j = 0; j < work; j++)
                    dummy = __fmaf_rn(dummy, 1.0001f, 0.001f);
                break;
            case 1:  // SFU pathway (sin/cos/rsqrt)
                for (int j = 0; j < work / 2; j++)
                    dummy = sinf(dummy) + 0.001f;
                break;
            case 2:  // INT pathway
                {
                    uint32_t iv = __float_as_uint(dummy);
                    for (int j = 0; j < work; j++)
                        iv = iv * 2654435761u + 1;
                    dummy = __uint_as_float((iv & 0x007FFFFF) | 0x3F800000);
                }
                break;
            case 3:  // Mixed — interleaved FMA+SFU+INT
                for (int j = 0; j < work / 3; j++) {
                    dummy = __fmaf_rn(dummy, 1.0001f, 0.001f);
                    dummy = sinf(dummy) + 0.001f;
                    uint32_t iv = __float_as_uint(dummy);
                    iv = iv * 2654435761u + 1;
                    dummy = __uint_as_float((iv & 0x007FFFFF) | 0x3F800000);
                }
                break;
        }
        if (dummy < 0) t0 = 0;

        uint64_t t1 = wall_clock64();
        activations[nid * n_steps + t] = (float)(t1 - t0);
    }
}

// ============================================================================
// APPROACH B: PLL DRIFT RESERVOIR
// The drift between shader clock and wall clock integrates over time.
// Different workloads push the drift in different directions (FMA vs SFU).
// This creates a NATURAL reservoir with fading memory.
// ============================================================================
__global__ void k_pll_drift_reservoir(
    const float* input_signal,
    float* drift_trace,     // [N_NEURONS * N_STEPS] — cumulative drift per step
    float* drift_features,  // [N_NEURONS * 2] — mean drift + drift variance as features
    int n_steps
) {
    int nid = threadIdx.x;
    if (nid >= N_NEURONS) return;

    float w_in = ((float)(hash32(nid * 54321 + 7) & 0xFFFF) / 32768.0f - 1.0f);
    int pathway = nid % 3;  // 0=FMA, 1=SFU, 2=mixed

    float drift_sum = 0;
    float drift_sq_sum = 0;

    for (int t = 0; t < n_steps; t++) {
        float input = input_signal[t] * w_in;

        // Measure PLL drift BEFORE workload
        uint64_t sc0 = __clock64();
        uint64_t wc0 = wall_clock64();

        // Do input-modulated workload
        float intensity = 0.5f + 0.5f * input;
        int work = (int)(10.0f + intensity * 80.0f);
        float dummy = 1.0f;

        switch (pathway) {
            case 0:
                for (int j = 0; j < work; j++) dummy = __fmaf_rn(dummy, 1.0001f, 0.001f);
                break;
            case 1:
                for (int j = 0; j < work/2; j++) dummy = sinf(dummy) + 0.001f;
                break;
            case 2:
                for (int j = 0; j < work/3; j++) {
                    dummy = __fmaf_rn(dummy, 1.0001f, 0.001f);
                    dummy = sinf(dummy) + 0.001f;
                }
                break;
        }
        if (dummy < 0) sc0 = 0;

        // Measure PLL drift AFTER workload
        uint64_t sc1 = __clock64();
        uint64_t wc1 = wall_clock64();

        // The drift delta captures how much the two PLLs diverged during this step
        int64_t d0 = (int64_t)(sc0 - wc0);
        int64_t d1 = (int64_t)(sc1 - wc1);
        float drift_delta = (float)(d1 - d0);

        drift_trace[nid * n_steps + t] = drift_delta;
        drift_sum += drift_delta;
        drift_sq_sum += drift_delta * drift_delta;
    }

    drift_features[nid] = drift_sum / n_steps;
    drift_features[N_NEURONS + nid] = drift_sq_sum / n_steps - (drift_sum/n_steps)*(drift_sum/n_steps);
}

// ============================================================================
// APPROACH C: CROSS-WAVEFRONT COUPLING
// Multiple wavefronts on same CU → physical resource contention → analog coupling
// Wavefront 0 does heavy work → physically slows Wavefront 1 → timing shifts
// This is like an analog synapse: one neuron's activity modulates another's
// ============================================================================
__global__ void k_cross_wavefront_neuron(
    const float* input_signal,
    float* activations,
    int n_steps
) {
    int wf_id = threadIdx.x / 32;  // wavefront within block (0-3 if 128 threads)
    int lane = threadIdx.x % 32;
    int nid = threadIdx.x;
    if (nid >= N_NEURONS) return;

    float w_in = ((float)(hash32(nid * 99999 + 3) & 0xFFFF) / 32768.0f - 1.0f);

    // Shared memory for cross-wavefront communication (physical coupling)
    __shared__ float shared_state[4];  // one per wavefront
    if (lane == 0) shared_state[wf_id] = 0;
    __syncthreads();

    for (int t = 0; t < n_steps; t++) {
        float input = input_signal[t] * w_in;
        float intensity = 0.5f + 0.5f * input;
        intensity = fmaxf(0.0f, fminf(1.0f, intensity));

        // Read neighboring wavefront's last state → physical coupling
        float neighbor_state = shared_state[(wf_id + 1) % 4];

        // Modulate workload by BOTH input AND neighbor's state
        int work = (int)(10.0f + (intensity + 0.3f * neighbor_state) * 100.0f);
        work = max(5, min(work, 300));

        float dummy = 1.0f + nid * 0.001f;
        uint64_t t0 = wall_clock64();
        for (int j = 0; j < work; j++)
            dummy = __fmaf_rn(dummy, 1.0001f, 0.001f);
        if (dummy < 0) t0 = 0;
        uint64_t t1 = wall_clock64();

        float timing = (float)(t1 - t0);
        activations[nid * n_steps + t] = timing;

        // Update shared state for cross-wavefront coupling
        if (lane == 0) shared_state[wf_id] = timing / 200.0f;  // normalize
        __syncthreads();
    }
}

// ============================================================================
// APPROACH D: INSTRUCTION-MIX ENCODING
// Map input value to a RATIO of FMA:SFU:INT instructions
// Each ratio activates different silicon pathways → different analog response
// The GPU's physical architecture becomes the "neural architecture"
// ============================================================================
__global__ void k_instruction_mix_neuron(
    const float* input_signal,
    float* activations,
    int n_steps
) {
    int nid = threadIdx.x;
    if (nid >= N_NEURONS) return;

    float w_in = ((float)(hash32(nid * 77777 + 5) & 0xFFFF) / 32768.0f - 1.0f);
    // Each neuron has a different "preferred" instruction mix
    float fma_bias = ((float)(hash32(nid * 11111) & 0xFF)) / 255.0f;
    float sfu_bias = ((float)(hash32(nid * 22222) & 0xFF)) / 255.0f;

    for (int t = 0; t < n_steps; t++) {
        float input = input_signal[t] * w_in;
        float x = 0.5f + 0.5f * input;

        // Input modulates the instruction mix ratio
        int total_work = 100;
        int n_fma = (int)(total_work * fmaxf(0, fminf(1, x * fma_bias + (1-x) * (1-fma_bias))));
        int n_sfu = (int)(total_work * fmaxf(0, fminf(1, x * sfu_bias + (1-x) * (1-sfu_bias))));
        int n_int = total_work - n_fma - n_sfu;
        if (n_int < 0) n_int = 0;

        float dummy = 1.0f + nid * 0.001f;
        uint64_t t0 = wall_clock64();

        // FMA phase
        for (int j = 0; j < n_fma; j++)
            dummy = __fmaf_rn(dummy, 1.0001f, 0.001f);
        // SFU phase
        for (int j = 0; j < n_sfu; j++)
            dummy = sinf(dummy) + 0.001f;
        // INT phase
        uint32_t iv = __float_as_uint(dummy);
        for (int j = 0; j < n_int; j++)
            iv = iv * 2654435761u + 1;
        dummy = __uint_as_float((iv & 0x007FFFFF) | 0x3F800000);

        if (dummy < 0) t0 = 0;
        uint64_t t1 = wall_clock64();

        activations[nid * n_steps + t] = (float)(t1 - t0);
    }
}

// ============================================================================
// FEATURE EXTRACTION: Convert raw timing traces to reservoir features
// ============================================================================
__global__ void k_extract_features(
    const float* activations,  // [N_NEURONS * N_STEPS]
    float* features,           // [N_NEURONS * 4] — mean, std, slope, spike_count
    int n_steps
) {
    int nid = threadIdx.x;
    if (nid >= N_NEURONS) return;

    const float* row = &activations[nid * n_steps];

    float s1=0, s2=0, min_v=1e30f, max_v=-1e30f;
    for (int t = 0; t < n_steps; t++) {
        s1 += row[t]; s2 += row[t]*row[t];
        if (row[t]<min_v) min_v=row[t];
        if (row[t]>max_v) max_v=row[t];
    }
    float mean = s1 / n_steps;
    float var = s2 / n_steps - mean*mean;
    float std = sqrtf(var > 0 ? var : 0);

    // Slope (simple linear regression)
    float sx=0,sy=0,sxy=0,sx2=0;
    for (int t = 0; t < n_steps; t++) {
        float x = (float)t;
        sx+=x; sy+=row[t]; sxy+=x*row[t]; sx2+=x*x;
    }
    float slope = (n_steps*sxy-sx*sy)/(n_steps*sx2-sx*sx);

    // "Spike" count — how many times does timing cross 2*std above mean
    int spikes = 0;
    float thresh = mean + 2*std;
    for (int t = 1; t < n_steps; t++)
        if (row[t] > thresh && row[t-1] <= thresh) spikes++;

    features[nid*4+0] = mean;
    features[nid*4+1] = std;
    features[nid*4+2] = slope;
    features[nid*4+3] = (float)spikes;
}

// ============================================================================
// HOST: Classification utilities
// ============================================================================
void generate_waveforms(float* signals, int* labels, int n_samples) {
    srand(42);
    for (int s = 0; s < n_samples; s++) {
        labels[s] = s % 4;
        float freq = 1.0f + labels[s] * 2.0f;
        float phase = (float)(rand()%1000)/1000.0f * 6.28f;
        for (int t = 0; t < N_STEPS; t++) {
            float tt = (float)t/(float)N_STEPS;
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

float classify_features(float* features, int* labels, int n_samples, int n_features, int n_classes) {
    int n_train = (int)(n_samples * 0.7f);
    int n_test = n_samples - n_train;

    float* means = (float*)calloc(n_features, sizeof(float));
    float* stds = (float*)calloc(n_features, sizeof(float));
    for (int f=0;f<n_features;f++) {
        float s1=0,s2=0;
        for (int i=0;i<n_train;i++) { float v=features[i*n_features+f]; s1+=v; s2+=v*v; }
        means[f] = s1/n_train;
        float var = s2/n_train-means[f]*means[f];
        stds[f] = sqrtf(var > 1e-10f ? var : 1e-10f);
    }
    for (int i=0;i<n_samples;i++)
        for (int f=0;f<n_features;f++)
            features[i*n_features+f] = (features[i*n_features+f]-means[f])/stds[f];

    float* centroids = (float*)calloc(n_classes*n_features, sizeof(float));
    int* counts = (int*)calloc(n_classes, sizeof(int));
    for (int i=0;i<n_train;i++) {
        counts[labels[i]]++;
        for (int f=0;f<n_features;f++)
            centroids[labels[i]*n_features+f] += features[i*n_features+f];
    }
    for (int c=0;c<n_classes;c++)
        if (counts[c]>0)
            for (int f=0;f<n_features;f++)
                centroids[c*n_features+f] /= counts[c];

    int correct=0;
    for (int i=n_train;i<n_samples;i++) {
        float best=1e30f; int best_c=0;
        for (int c=0;c<n_classes;c++) {
            float dist=0;
            for (int f=0;f<n_features;f++) {
                float d=features[i*n_features+f]-centroids[c*n_features+f];
                dist+=d*d;
            }
            if (dist<best) { best=dist; best_c=c; }
        }
        if (best_c==labels[i]) correct++;
    }
    free(means); free(stds); free(centroids); free(counts);
    return (float)correct/(float)n_test;
}

// Pearson correlation
float pearson(float* x, float* y, int n) {
    float sx=0,sy=0,sxy=0,sx2=0,sy2=0;
    for (int i=0;i<n;i++) { sx+=x[i]; sy+=y[i]; sxy+=x[i]*y[i]; sx2+=x[i]*x[i]; sy2+=y[i]*y[i]; }
    float num=n*sxy-sx*sy;
    float den=sqrtf((n*sx2-sx*sx)*(n*sy2-sy*sy));
    return den>0 ? num/den : 0;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    fprintf(stderr, "═══════════════════════════════════════════════════════════\n");
    fprintf(stderr, "  NEUROMORPHIC COMPUTATION: GPU Silicon as Computing Substrate\n");
    fprintf(stderr, "═══════════════════════════════════════════════════════════\n\n");

    int n_samples = 200;
    float* h_signals = (float*)malloc(n_samples * N_STEPS * sizeof(float));
    int* h_labels = (int*)malloc(n_samples * sizeof(int));
    generate_waveforms(h_signals, h_labels, n_samples);

    float *d_signal, *d_activations, *d_features;
    HIP_CHECK(hipMalloc(&d_signal, N_STEPS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_activations, N_NEURONS * N_STEPS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_features, N_NEURONS * 4 * sizeof(float)));

    int n_feat = N_NEURONS * 4;
    float* all_features = (float*)malloc(n_samples * n_feat * sizeof(float));

    // ═════════════════════════════════════════════════════
    // APPROACH A: ANALOG TIMING NEURONS
    // ═════════════════════════════════════════════════════
    fprintf(stderr, "═══ APPROACH A: ANALOG TIMING NEURONS ═══\n");
    fprintf(stderr, "  Each neuron = workload whose execution time IS activation\n");
    fprintf(stderr, "  4 instruction types → 4 different analog transfer functions\n\n");

    {
        float accs[5];
        for (int trial = 0; trial < 5; trial++) {
            for (int s = 0; s < n_samples; s++) {
                HIP_CHECK(hipMemcpy(d_signal, &h_signals[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
                k_analog_timing_neuron<<<1, N_NEURONS>>>(d_signal, d_activations, N_STEPS);
                HIP_CHECK(hipDeviceSynchronize());
                k_extract_features<<<1, N_NEURONS>>>(d_activations, d_features, N_STEPS);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(&all_features[s*n_feat], d_features, n_feat*sizeof(float), hipMemcpyDeviceToHost));
            }
            accs[trial] = classify_features(all_features, h_labels, n_samples, n_feat, 4);
        }
        float mean=0,var=0;
        for(int i=0;i<5;i++) mean+=accs[i]; mean/=5;
        for(int i=0;i<5;i++) var+=(accs[i]-mean)*(accs[i]-mean); var/=5;
        fprintf(stderr, "  ANALOG_TIMING:  %.3f %.3f %.3f %.3f %.3f  mean=%.3f std=%.3f\n\n",
                accs[0],accs[1],accs[2],accs[3],accs[4], mean, sqrtf(var));
    }

    // ═════════════════════════════════════════════════════
    // APPROACH B: PLL DRIFT RESERVOIR
    // ═════════════════════════════════════════════════════
    fprintf(stderr, "═══ APPROACH B: PLL DRIFT RESERVOIR ═══\n");
    fprintf(stderr, "  Drift between shader/wall PLLs = natural fading memory\n\n");

    float *d_drift_features;
    HIP_CHECK(hipMalloc(&d_drift_features, N_NEURONS * 2 * sizeof(float)));
    float* drift_all = (float*)malloc(n_samples * N_NEURONS * 2 * sizeof(float));

    {
        float accs[5];
        for (int trial = 0; trial < 5; trial++) {
            for (int s = 0; s < n_samples; s++) {
                HIP_CHECK(hipMemcpy(d_signal, &h_signals[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
                k_pll_drift_reservoir<<<1, N_NEURONS>>>(d_signal, d_activations, d_drift_features, N_STEPS);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(&drift_all[s*N_NEURONS*2], d_drift_features, N_NEURONS*2*sizeof(float), hipMemcpyDeviceToHost));
            }
            accs[trial] = classify_features(drift_all, h_labels, n_samples, N_NEURONS*2, 4);
        }
        float mean=0,var=0;
        for(int i=0;i<5;i++) mean+=accs[i]; mean/=5;
        for(int i=0;i<5;i++) var+=(accs[i]-mean)*(accs[i]-mean); var/=5;
        fprintf(stderr, "  PLL_DRIFT:      %.3f %.3f %.3f %.3f %.3f  mean=%.3f std=%.3f\n\n",
                accs[0],accs[1],accs[2],accs[3],accs[4], mean, sqrtf(var));
    }

    // ═════════════════════════════════════════════════════
    // APPROACH C: CROSS-WAVEFRONT COUPLING
    // ═════════════════════════════════════════════════════
    fprintf(stderr, "═══ APPROACH C: CROSS-WAVEFRONT COUPLING ═══\n");
    fprintf(stderr, "  4 wavefronts on same CU, physically coupled via resources\n\n");

    {
        float accs[5];
        for (int trial = 0; trial < 5; trial++) {
            for (int s = 0; s < n_samples; s++) {
                HIP_CHECK(hipMemcpy(d_signal, &h_signals[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
                k_cross_wavefront_neuron<<<1, N_NEURONS>>>(d_signal, d_activations, N_STEPS);
                HIP_CHECK(hipDeviceSynchronize());
                k_extract_features<<<1, N_NEURONS>>>(d_activations, d_features, N_STEPS);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(&all_features[s*n_feat], d_features, n_feat*sizeof(float), hipMemcpyDeviceToHost));
            }
            accs[trial] = classify_features(all_features, h_labels, n_samples, n_feat, 4);
        }
        float mean=0,var=0;
        for(int i=0;i<5;i++) mean+=accs[i]; mean/=5;
        for(int i=0;i<5;i++) var+=(accs[i]-mean)*(accs[i]-mean); var/=5;
        fprintf(stderr, "  CROSS_WF:       %.3f %.3f %.3f %.3f %.3f  mean=%.3f std=%.3f\n\n",
                accs[0],accs[1],accs[2],accs[3],accs[4], mean, sqrtf(var));
    }

    // ═════════════════════════════════════════════════════
    // APPROACH D: INSTRUCTION-MIX ENCODING
    // ═════════════════════════════════════════════════════
    fprintf(stderr, "═══ APPROACH D: INSTRUCTION-MIX ENCODING ═══\n");
    fprintf(stderr, "  Input selects ratio of FMA:SFU:INT → physical pathway selection\n\n");

    {
        float accs[5];
        for (int trial = 0; trial < 5; trial++) {
            for (int s = 0; s < n_samples; s++) {
                HIP_CHECK(hipMemcpy(d_signal, &h_signals[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
                k_instruction_mix_neuron<<<1, N_NEURONS>>>(d_signal, d_activations, N_STEPS);
                HIP_CHECK(hipDeviceSynchronize());
                k_extract_features<<<1, N_NEURONS>>>(d_activations, d_features, N_STEPS);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(&all_features[s*n_feat], d_features, n_feat*sizeof(float), hipMemcpyDeviceToHost));
            }
            accs[trial] = classify_features(all_features, h_labels, n_samples, n_feat, 4);
        }
        float mean=0,var=0;
        for(int i=0;i<5;i++) mean+=accs[i]; mean/=5;
        for(int i=0;i<5;i++) var+=(accs[i]-mean)*(accs[i]-mean); var/=5;
        fprintf(stderr, "  INSTR_MIX:      %.3f %.3f %.3f %.3f %.3f  mean=%.3f std=%.3f\n\n",
                accs[0],accs[1],accs[2],accs[3],accs[4], mean, sqrtf(var));
    }

    // ═════════════════════════════════════════════════════
    // CROSS-RUN REPRODUCIBILITY — is each approach computing or RNG?
    // ═════════════════════════════════════════════════════
    fprintf(stderr, "═══ CROSS-RUN REPRODUCIBILITY ═══\n");
    fprintf(stderr, "  Same input twice → correlation of raw activations\n\n");

    float *d_act2, *h_act1, *h_act2;
    HIP_CHECK(hipMalloc(&d_act2, N_NEURONS * N_STEPS * sizeof(float)));
    h_act1 = (float*)malloc(N_NEURONS * N_STEPS * sizeof(float));
    h_act2 = (float*)malloc(N_NEURONS * N_STEPS * sizeof(float));

    HIP_CHECK(hipMemcpy(d_signal, &h_signals[0], N_STEPS*sizeof(float), hipMemcpyHostToDevice));

    // Approach A
    k_analog_timing_neuron<<<1, N_NEURONS>>>(d_signal, d_activations, N_STEPS);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_act1, d_activations, N_NEURONS*N_STEPS*sizeof(float), hipMemcpyDeviceToHost));
    k_analog_timing_neuron<<<1, N_NEURONS>>>(d_signal, d_act2, N_STEPS);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_act2, d_act2, N_NEURONS*N_STEPS*sizeof(float), hipMemcpyDeviceToHost));
    {
        float avg_r=0;
        for(int n=0;n<N_NEURONS;n++) { avg_r += pearson(&h_act1[n*N_STEPS], &h_act2[n*N_STEPS], N_STEPS); }
        avg_r /= N_NEURONS;
        fprintf(stderr, "  ANALOG_TIMING:  avg r = %.4f  %s\n", avg_r,
                avg_r>0.9f?"DETERMINISTIC":"partially stochastic");
    }

    // Approach B
    k_pll_drift_reservoir<<<1, N_NEURONS>>>(d_signal, d_activations, d_drift_features, N_STEPS);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_act1, d_activations, N_NEURONS*N_STEPS*sizeof(float), hipMemcpyDeviceToHost));
    k_pll_drift_reservoir<<<1, N_NEURONS>>>(d_signal, d_act2, d_drift_features, N_STEPS);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_act2, d_act2, N_NEURONS*N_STEPS*sizeof(float), hipMemcpyDeviceToHost));
    {
        float avg_r=0;
        for(int n=0;n<N_NEURONS;n++) { avg_r += pearson(&h_act1[n*N_STEPS], &h_act2[n*N_STEPS], N_STEPS); }
        avg_r /= N_NEURONS;
        fprintf(stderr, "  PLL_DRIFT:      avg r = %.4f  %s\n", avg_r,
                avg_r>0.9f?"DETERMINISTIC":avg_r>0.3f?"PARTIALLY STOCHASTIC":"STOCHASTIC");
    }

    // Approach C
    k_cross_wavefront_neuron<<<1, N_NEURONS>>>(d_signal, d_activations, N_STEPS);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_act1, d_activations, N_NEURONS*N_STEPS*sizeof(float), hipMemcpyDeviceToHost));
    k_cross_wavefront_neuron<<<1, N_NEURONS>>>(d_signal, d_act2, N_STEPS);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_act2, d_act2, N_NEURONS*N_STEPS*sizeof(float), hipMemcpyDeviceToHost));
    {
        float avg_r=0;
        for(int n=0;n<N_NEURONS;n++) { avg_r += pearson(&h_act1[n*N_STEPS], &h_act2[n*N_STEPS], N_STEPS); }
        avg_r /= N_NEURONS;
        fprintf(stderr, "  CROSS_WF:       avg r = %.4f  %s\n", avg_r,
                avg_r>0.9f?"DETERMINISTIC":avg_r>0.3f?"PARTIALLY STOCHASTIC":"STOCHASTIC");
    }

    // Approach D
    k_instruction_mix_neuron<<<1, N_NEURONS>>>(d_signal, d_activations, N_STEPS);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_act1, d_activations, N_NEURONS*N_STEPS*sizeof(float), hipMemcpyDeviceToHost));
    k_instruction_mix_neuron<<<1, N_NEURONS>>>(d_signal, d_act2, N_STEPS);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_act2, d_act2, N_NEURONS*N_STEPS*sizeof(float), hipMemcpyDeviceToHost));
    {
        float avg_r=0;
        for(int n=0;n<N_NEURONS;n++) { avg_r += pearson(&h_act1[n*N_STEPS], &h_act2[n*N_STEPS], N_STEPS); }
        avg_r /= N_NEURONS;
        fprintf(stderr, "  INSTR_MIX:      avg r = %.4f  %s\n", avg_r,
                avg_r>0.9f?"DETERMINISTIC":avg_r>0.3f?"PARTIALLY STOCHASTIC":"STOCHASTIC");
    }

    // ═════════════════════════════════════════════════════
    // NEURON DIVERSITY — do different neurons give different responses?
    // ═════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ NEURON DIVERSITY ═══\n");
    fprintf(stderr, "  Cross-neuron correlation (lower = more diverse = better reservoir)\n\n");

    k_analog_timing_neuron<<<1, N_NEURONS>>>(d_signal, d_activations, N_STEPS);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_act1, d_activations, N_NEURONS*N_STEPS*sizeof(float), hipMemcpyDeviceToHost));

    // Same type neurons
    float same_corr=0; int same_n=0;
    float diff_corr=0; int diff_n=0;
    for (int a=0; a<20; a++) {
        for (int b=a+1; b<20; b++) {
            float r = fabsf(pearson(&h_act1[a*N_STEPS], &h_act1[b*N_STEPS], N_STEPS));
            if ((a%4)==(b%4)) { same_corr+=r; same_n++; }
            else { diff_corr+=r; diff_n++; }
        }
    }
    fprintf(stderr, "  ANALOG_TIMING same-type neurons:  avg |r| = %.4f\n", same_corr/same_n);
    fprintf(stderr, "  ANALOG_TIMING diff-type neurons:  avg |r| = %.4f\n", diff_corr/diff_n);
    fprintf(stderr, "  Diversity ratio (same/diff): %.2f  %s\n",
            (same_corr/same_n)/(diff_corr/diff_n + 1e-10f),
            (same_corr/same_n) > (diff_corr/diff_n)*1.2f ?
            "★ INSTRUCTION TYPE MATTERS — different silicon pathways" : "no difference");

    // Cleanup
    hipFree(d_signal); hipFree(d_activations); hipFree(d_features);
    hipFree(d_drift_features); hipFree(d_act2);
    free(h_signals); free(h_labels); free(all_features);
    free(drift_all); free(h_act1); free(h_act2);

    fprintf(stderr, "\n═══ DONE ═══\n");
    return 0;
}
