// neuromorphic_recurrent.hip — Recurrent silicon physics reservoir
// Fixes 3 weaknesses of neuromorphic_full:
//   1. NO RECURRENCE → add physical recurrent feedback (prev timing → next workload)
//   2. WEAK FEATURES → richer temporal features (derivatives, ISI, phase)
//   3. WEAK COUPLING → shared memory state + atomic coupling between ALL neurons
//
// All computation done by raw GPU silicon physics. No PRNG, no hashing.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>

#define HIP_CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP ERROR: %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define N_NEURONS 128
#define N_STEPS 300
#define N_FEAT 12  // richer features per neuron

__device__ uint32_t hash32(uint32_t h) {
    h ^= h >> 16; h *= 0x85ebca6b; h ^= h >> 13; h *= 0xc2b2ae35; h ^= h >> 16;
    return h;
}

// ============================================================================
// RECURRENT NEUROMORPHIC KERNEL — silicon physics with feedback
// ============================================================================
// Key innovation: previous timestep's activation feeds back as workload
// modifier. This creates PHYSICAL recurrent dynamics — the GPU's timing
// at step t depends on its timing at step t-1, creating temporal memory
// through silicon, not software.
__global__ void k_neuromorphic_recurrent(
    const float* input_signal,   // [N_STEPS]
    float* activations,          // [N_NEURONS * N_STEPS]
    int n_neurons, int n_steps,
    float recurrence_strength,   // how much prev output affects next workload
    float coupling_strength,     // how much neighbors affect workload
    unsigned int* atomic_pool    // [16] shared coupling pool
) {
    int nid = threadIdx.x;
    if (nid >= n_neurons) return;

    int mode = nid % 5;
    // Deterministic heterogeneous input weight (no hash — use integer math)
    float w_in = ((float)((nid * 7919 + 1) % 1000) / 500.0f - 1.0f);

    // Cross-neuron coupling via shared memory
    __shared__ float neuron_state[N_NEURONS];  // previous timestep activations
    __shared__ float wf_pool[4];               // wavefront-level pooled state

    int wf_id = nid / 32;
    int lane = nid % 32;

    neuron_state[nid] = 0;
    if (lane == 0 && wf_id < 4) wf_pool[wf_id] = 0;
    __syncthreads();

    // Recurrent state — lives across timesteps
    float prev_activation = 0;
    float running_drift = 0;  // accumulated PLL drift for fading memory

    for (int t = 0; t < n_steps; t++) {
        float input = input_signal[t] * w_in;

        // === RECURRENT FEEDBACK ===
        // Previous activation modulates workload intensity
        // This is PHYSICAL recurrence: timing(t) depends on timing(t-1)
        float recurrent = prev_activation * recurrence_strength;

        // === CROSS-NEURON COUPLING ===
        // Read neighbor states (circular coupling)
        int left = (nid - 1 + n_neurons) % n_neurons;
        int right = (nid + 1) % n_neurons;
        int across = (nid + n_neurons/2) % n_neurons;
        float neighbor_signal = (neuron_state[left] + neuron_state[right]
                                 + 0.5f * neuron_state[across]) / 3.0f;
        float coupled = neighbor_signal * coupling_strength;

        // Wavefront-level pooled signal (coarse coupling)
        float wf_signal = wf_pool[wf_id] * 0.1f;

        // Total drive combines: input + recurrence + coupling
        float drive = 0.5f + 0.3f * input + recurrent + coupled + wf_signal;
        drive = fmaxf(0.02f, fminf(0.98f, drive));

        float activation = 0;

        switch (mode) {
        case 0: { // FMA timing with recurrent depth
            int work = (int)(8.0f + drive * 60.0f);
            float dummy = 1.0f + nid * 0.001f;
            uint64_t t0 = wall_clock64();
            for (int j = 0; j < work; j++)
                dummy = __fmaf_rn(dummy, 1.0001f, 0.001f);
            uint64_t t1 = wall_clock64();
            if (dummy < 0) t0 = 0;
            activation = (float)(t1 - t0);
            break;
        }
        case 1: { // SFU timing (trig units — different silicon path)
            int work = (int)(4.0f + drive * 25.0f);
            float dummy = 1.0f + nid * 0.001f;
            uint64_t t0 = wall_clock64();
            for (int j = 0; j < work; j++)
                dummy = sinf(dummy) + 0.001f;
            uint64_t t1 = wall_clock64();
            if (dummy < 0) t0 = 0;
            activation = (float)(t1 - t0);
            break;
        }
        case 2: { // PLL drift accumulator (fading memory neuron)
            uint64_t sc0 = __clock64();
            uint64_t wc0 = wall_clock64();
            float dummy = 1.0f;
            int work = (int)(8.0f + drive * 35.0f);
            for (int j = 0; j < work; j++)
                dummy = __fmaf_rn(dummy, 1.0001f, 0.001f);
            if (dummy < 0) sc0 = 0;
            uint64_t sc1 = __clock64();
            uint64_t wc1 = wall_clock64();
            float drift_delta = (float)((int64_t)(sc1 - wc1) - (int64_t)(sc0 - wc0));
            // Leaky integration of drift — this IS the fading memory
            running_drift = 0.85f * running_drift + 0.15f * drift_delta;
            activation = running_drift;
            break;
        }
        case 3: { // Cross-wavefront timing (resource contention synapses)
            float wf_neighbor = wf_pool[(wf_id + 1) % 4];
            int work = (int)(8.0f + (drive + 0.3f * wf_neighbor) * 50.0f);
            work = max(5, min(work, 120));
            float dummy = 1.0f + nid * 0.001f;
            uint64_t t0 = wall_clock64();
            for (int j = 0; j < work; j++)
                dummy = __fmaf_rn(dummy, 1.0001f, 0.001f);
            uint64_t t1 = wall_clock64();
            if (dummy < 0) t0 = 0;
            activation = (float)(t1 - t0);
            break;
        }
        case 4: { // Atomic contention pool (global coupling)
            // Write to atomic pool — creates non-deterministic ordering
            unsigned int pool_idx = (nid + t) % 16;
            unsigned int old = atomicAdd(&atomic_pool[pool_idx], 1u);
            // Time the contention
            uint64_t t0 = wall_clock64();
            for (int j = 0; j < 4; j++)
                atomicAdd(&atomic_pool[(pool_idx + j) % 16], 1u);
            uint64_t t1 = wall_clock64();
            activation = (float)(t1 - t0) + (float)(old & 0x3FF) * 0.001f;
            break;
        }
        }

        // Normalize activation to ~[0,1] range for stable recurrence
        // Use running statistics to adapt
        float norm_act = activation / fmaxf(1.0f, fabsf(activation) + 100.0f);

        // Store for readout
        activations[nid * n_steps + t] = activation;

        // Update shared state for coupling (BEFORE syncthreads)
        prev_activation = norm_act;

        __syncthreads();
        neuron_state[nid] = norm_act;

        // Update wavefront pool (lane 0 computes mean of its wavefront)
        if (lane == 0 && wf_id < 4) {
            float sum = 0;
            for (int i = wf_id * 32; i < min((wf_id + 1) * 32, n_neurons); i++)
                sum += neuron_state[i];
            wf_pool[wf_id] = sum / 32.0f;
        }
        __syncthreads();
    }
}

// ============================================================================
// RICHER FEATURE EXTRACTION — temporal dynamics, not just statistics
// ============================================================================
__global__ void k_features_rich(const float* act, float* feat,
                                 int n_neurons, int n_steps) {
    int nid = threadIdx.x;
    if (nid >= n_neurons) return;

    const float* row = &act[nid * n_steps];

    // Basic statistics
    float s1 = 0, s2 = 0, mn = 1e30f, mx = -1e30f;
    for (int t = 0; t < n_steps; t++) {
        s1 += row[t]; s2 += row[t] * row[t];
        if (row[t] < mn) mn = row[t];
        if (row[t] > mx) mx = row[t];
    }
    float mean = s1 / n_steps;
    float var = s2 / n_steps - mean * mean;
    float std = sqrtf(var > 0 ? var : 0);

    // Temporal: quarter means
    int q = n_steps / 4;
    float q1 = 0, q2 = 0, q3 = 0, q4 = 0;
    for (int t = 0; t < q; t++) q1 += row[t];
    for (int t = q; t < 2*q; t++) q2 += row[t];
    for (int t = 2*q; t < 3*q; t++) q3 += row[t];
    for (int t = 3*q; t < n_steps; t++) q4 += row[t];
    q1 /= q; q2 /= q; q3 /= q; q4 /= (n_steps - 3*q);

    // Derivative features: how fast activations change
    float deriv_sum = 0, deriv_abs = 0;
    int zero_crossings = 0;
    for (int t = 1; t < n_steps; t++) {
        float d = row[t] - row[t-1];
        deriv_sum += d;
        deriv_abs += fabsf(d);
        if ((row[t] - mean) * (row[t-1] - mean) < 0) zero_crossings++;
    }

    // Autocorrelation at lag 1 (temporal memory indicator)
    float ac = 0;
    for (int t = 1; t < n_steps; t++)
        ac += (row[t] - mean) * (row[t-1] - mean);
    ac /= (var > 1e-10f ? var * (n_steps - 1) : 1.0f);

    int base = nid * N_FEAT;
    feat[base + 0] = mean;
    feat[base + 1] = std;
    feat[base + 2] = mx - mn;
    feat[base + 3] = q1;
    feat[base + 4] = q2;
    feat[base + 5] = q3;
    feat[base + 6] = q4;
    feat[base + 7] = deriv_sum / n_steps;     // trend
    feat[base + 8] = deriv_abs / n_steps;     // volatility
    feat[base + 9] = (float)zero_crossings;   // oscillation frequency
    feat[base + 10] = ac;                     // temporal memory
    feat[base + 11] = q4 - q1;               // temporal slope
}

// ============================================================================
// SOFTWARE BASELINES
// ============================================================================
__global__ void k_lif_recurrent(
    const float* input_signal, float* activations,
    int n_neurons, int n_steps, float alpha, float threshold,
    float noise_amp, float recurrence, uint32_t seed
) {
    int nid = threadIdx.x;
    if (nid >= n_neurons) return;

    float w_in = ((float)((nid * 7919 + 1) % 1000) / 500.0f - 1.0f);
    __shared__ float state[N_NEURONS];
    state[nid] = 0;
    __syncthreads();

    uint32_t rng = (nid + 1) * 2654435761u + seed;
    float vmem = 0;

    for (int t = 0; t < n_steps; t++) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        float noise = ((float)(rng & 0xFFFF) / 32768.0f - 1.0f) * noise_amp;

        int left = (nid - 1 + n_neurons) % n_neurons;
        int right = (nid + 1) % n_neurons;
        float neighbor = (state[left] + state[right]) * 0.5f * recurrence;

        float I = w_in * input_signal[t] + noise + neighbor;
        vmem = (1.0f - alpha) * vmem + I;
        if (vmem >= threshold) vmem = 0;
        else if (vmem < -threshold) vmem = 0;

        activations[nid * n_steps + t] = vmem;
        __syncthreads();
        state[nid] = vmem;
        __syncthreads();
    }
}

__global__ void k_esn_recurrent(
    const float* input_signal, float* activations,
    int n_neurons, int n_steps, float sr, float temp, float is_
) {
    int nid = threadIdx.x;
    if (nid >= n_neurons) return;

    float w_in = ((float)((nid * 7919 + 1) % 1000) / 500.0f - 1.0f) * is_;
    __shared__ float state[N_NEURONS];
    state[nid] = 0;
    __syncthreads();

    int prev = (nid - 1 + n_neurons) % n_neurons;
    int next = (nid + 1) % n_neurons;
    int across = (nid + n_neurons/2) % n_neurons;
    float wp = ((float)((nid * 3571 + 7) % 1000) / 500.0f - 1.0f) * sr * 0.5f;
    float wn = ((float)((nid * 8923 + 3) % 1000) / 500.0f - 1.0f) * sr * 0.5f;
    float wa = ((float)((nid * 4517 + 9) % 1000) / 500.0f - 1.0f) * sr * 0.3f;

    for (int t = 0; t < n_steps; t++) {
        float x = w_in * input_signal[t]
                + wp * state[prev] + wn * state[next] + wa * state[across];
        float s = tanhf(x / temp);
        __syncthreads();
        state[nid] = s;
        __syncthreads();
        activations[nid * n_steps + t] = s;
    }
}

// ============================================================================
// HOST: classification and main
// ============================================================================
void gen_waveforms(float* sig, int* lab, int ns, int nc) {
    srand(42);
    for (int s = 0; s < ns; s++) {
        lab[s] = s % nc;
        float freq = 1.0f + lab[s] * 1.5f;
        float phase = (float)(rand() % 1000) / 1000.0f * 6.28f;
        for (int t = 0; t < N_STEPS; t++) {
            float tt = (float)t / N_STEPS;
            float val = 0;
            switch (lab[s] % 8) {
                case 0: val = sinf(2*3.14159f*freq*tt+phase); break;
                case 1: val = (fmodf(freq*tt+phase/6.28f,1)<0.5f)?1:-1; break;
                case 2: val = 2*fabsf(fmodf(freq*tt+phase/6.28f,1)-0.5f)-0.5f; break;
                case 3: val = sinf(2*3.14159f*freq*tt)+0.5f*sinf(6*3.14159f*freq*tt); break;
                case 4: val = sinf(2*3.14159f*freq*tt)*expf(-2*tt); break;
                case 5: val = (fmodf(freq*tt,1)<0.3f)?sinf(20*3.14159f*tt):0; break;
                case 6: val = tanhf(3*sinf(2*3.14159f*freq*tt)); break;
                case 7: val = sinf(2*3.14159f*freq*tt+sinf(4*3.14159f*tt)); break;
            }
            sig[s*N_STEPS+t] = val + 0.1f*((float)(rand()%1000)/500.0f-1.0f);
        }
    }
}

void gen_xor(float* sig, int* lab, int ns, int delay) {
    srand(123);
    for (int s = 0; s < ns; s++) {
        for (int t = 0; t < N_STEPS; t++)
            sig[s*N_STEPS+t] = (rand()%2) ? 1.0f : -1.0f;
        int pos = N_STEPS / 2;
        int a = sig[s*N_STEPS+pos] > 0 ? 1 : 0;
        int b = (pos - delay >= 0) ? (sig[s*N_STEPS+pos-delay] > 0 ? 1 : 0) : 0;
        lab[s] = a ^ b;
    }
}

// NARMA-like nonlinear memory task
void gen_narma(float* sig, int* lab, int ns, int order) {
    srand(456);
    for (int s = 0; s < ns; s++) {
        // Random input
        for (int t = 0; t < N_STEPS; t++)
            sig[s*N_STEPS+t] = ((float)(rand()%1000)/1000.0f) * 0.5f;

        // NARMA target: y(t) = 0.3*y(t-1) + 0.05*y(t-1)*sum(y(t-1..t-order)) + 1.5*u(t-order)*u(t-1) + 0.1
        float y[N_STEPS];
        y[0] = 0;
        for (int t = 1; t < N_STEPS; t++) {
            float sum_y = 0;
            for (int k = 1; k <= order && t-k >= 0; k++) sum_y += y[t-k];
            float ut_ord = (t-order >= 0) ? sig[s*N_STEPS+t-order] : 0;
            y[t] = 0.3f*y[t-1] + 0.05f*y[t-1]*sum_y + 1.5f*ut_ord*sig[s*N_STEPS+t-1] + 0.1f;
            y[t] = fminf(1.0f, fmaxf(-1.0f, y[t]));
        }
        // Classify based on final y value (4 bins)
        float final_y = y[N_STEPS-1];
        lab[s] = (int)((final_y + 1.0f) * 2.0f);
        if (lab[s] < 0) lab[s] = 0;
        if (lab[s] > 3) lab[s] = 3;
    }
}

float classify(float* feat, int* lab, int ns, int nf, int nc) {
    int nt = (int)(ns * 0.7f), ntest = ns - nt;
    if (ntest == 0) return 0;
    float* mu = (float*)calloc(nf, sizeof(float));
    float* sd = (float*)calloc(nf, sizeof(float));
    for (int f = 0; f < nf; f++) {
        float s1 = 0, s2 = 0;
        for (int i = 0; i < nt; i++) { float v = feat[i*nf+f]; s1 += v; s2 += v*v; }
        mu[f] = s1/nt;
        float v = s2/nt - mu[f]*mu[f];
        sd[f] = sqrtf(v > 1e-10f ? v : 1e-10f);
    }
    for (int i = 0; i < ns; i++)
        for (int f = 0; f < nf; f++)
            feat[i*nf+f] = (feat[i*nf+f] - mu[f]) / sd[f];

    float* cen = (float*)calloc(nc*nf, sizeof(float));
    int* cnt = (int*)calloc(nc, sizeof(int));
    for (int i = 0; i < nt; i++) {
        cnt[lab[i]]++;
        for (int f = 0; f < nf; f++) cen[lab[i]*nf+f] += feat[i*nf+f];
    }
    for (int c = 0; c < nc; c++)
        if (cnt[c] > 0) for (int f = 0; f < nf; f++) cen[c*nf+f] /= cnt[c];

    int ok = 0;
    for (int i = nt; i < ns; i++) {
        float b = 1e30f; int bc = 0;
        for (int c = 0; c < nc; c++) {
            float d = 0;
            for (int f = 0; f < nf; f++) {
                float dd = feat[i*nf+f] - cen[c*nf+f];
                d += dd * dd;
            }
            if (d < b) { b = d; bc = c; }
        }
        if (bc == lab[i]) ok++;
    }
    free(mu); free(sd); free(cen); free(cnt);
    return (float)ok / ntest;
}

int main() {
    fprintf(stderr, "═══════════════════════════════════════════════════════════════\n");
    fprintf(stderr, "  RECURRENT SILICON PHYSICS RESERVOIR — Solving the Weaknesses\n");
    fprintf(stderr, "═══════════════════════════════════════════════════════════════\n");
    fprintf(stderr, "  Neurons: %d, Steps: %d, Features: %d/neuron\n", N_NEURONS, N_STEPS, N_FEAT);
    fprintf(stderr, "  Fixes: recurrence, rich features, cross-neuron coupling\n\n");

    int nn = N_NEURONS;
    int nf = nn * N_FEAT;

    // Allocate GPU
    float *d_sig, *d_act, *d_feat;
    unsigned int *d_atomic;
    HIP_CHECK(hipMalloc(&d_sig, N_STEPS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_act, N_NEURONS * N_STEPS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_feat, N_NEURONS * N_FEAT * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_atomic, 16 * sizeof(unsigned int)));

    // ─────────────────────────────────────────────────────────────
    // TEST 1: Recurrence sweep — what strength gives best temporal memory?
    // ─────────────────────────────────────────────────────────────
    fprintf(stderr, "═══ TEST 1: RECURRENCE STRENGTH SWEEP (XOR τ=2) ═══\n");
    {
        int ns = 200, nc = 2;
        float* h_sig = (float*)malloc(ns * N_STEPS * sizeof(float));
        int* h_lab = (int*)malloc(ns * sizeof(int));
        float* all_f = (float*)malloc(ns * nf * sizeof(float));

        gen_xor(h_sig, h_lab, ns, 2);

        float strengths[] = {0.0f, 0.05f, 0.1f, 0.2f, 0.3f, 0.5f};
        for (int si = 0; si < 6; si++) {
            float r = strengths[si];
            float acc_sum = 0;
            for (int trial = 0; trial < 3; trial++) {
                for (int s = 0; s < ns; s++) {
                    HIP_CHECK(hipMemcpy(d_sig, &h_sig[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
                    HIP_CHECK(hipMemset(d_atomic, 0, 16*sizeof(unsigned int)));
                    k_neuromorphic_recurrent<<<1,nn>>>(d_sig, d_act, nn, N_STEPS, r, 0.1f, d_atomic);
                    HIP_CHECK(hipDeviceSynchronize());
                    k_features_rich<<<1,nn>>>(d_act, d_feat, nn, N_STEPS);
                    HIP_CHECK(hipDeviceSynchronize());
                    HIP_CHECK(hipMemcpy(&all_f[s*nf], d_feat, nf*sizeof(float), hipMemcpyDeviceToHost));
                }
                acc_sum += classify(all_f, h_lab, ns, nf, nc);
            }
            fprintf(stderr, "  recurrence=%.2f  XOR_τ2=%.3f\n", r, acc_sum / 3.0f);
        }
        free(h_sig); free(h_lab); free(all_f);
    }

    // ─────────────────────────────────────────────────────────────
    // TEST 2: Coupling strength sweep
    // ─────────────────────────────────────────────────────────────
    fprintf(stderr, "\n═══ TEST 2: COUPLING STRENGTH SWEEP (4-class waveform) ═══\n");
    {
        int ns = 200, nc = 4;
        float* h_sig = (float*)malloc(ns * N_STEPS * sizeof(float));
        int* h_lab = (int*)malloc(ns * sizeof(int));
        float* all_f = (float*)malloc(ns * nf * sizeof(float));

        gen_waveforms(h_sig, h_lab, ns, nc);

        float couplings[] = {0.0f, 0.05f, 0.1f, 0.2f, 0.3f, 0.5f};
        for (int ci = 0; ci < 6; ci++) {
            float c = couplings[ci];
            float acc_sum = 0;
            for (int trial = 0; trial < 3; trial++) {
                for (int s = 0; s < ns; s++) {
                    HIP_CHECK(hipMemcpy(d_sig, &h_sig[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
                    HIP_CHECK(hipMemset(d_atomic, 0, 16*sizeof(unsigned int)));
                    k_neuromorphic_recurrent<<<1,nn>>>(d_sig, d_act, nn, N_STEPS, 0.1f, c, d_atomic);
                    HIP_CHECK(hipDeviceSynchronize());
                    k_features_rich<<<1,nn>>>(d_act, d_feat, nn, N_STEPS);
                    HIP_CHECK(hipDeviceSynchronize());
                    HIP_CHECK(hipMemcpy(&all_f[s*nf], d_feat, nf*sizeof(float), hipMemcpyDeviceToHost));
                }
                acc_sum += classify(all_f, h_lab, ns, nf, nc);
            }
            fprintf(stderr, "  coupling=%.2f  4class=%.3f\n", c, acc_sum / 3.0f);
        }
        free(h_sig); free(h_lab); free(all_f);
    }

    // ─────────────────────────────────────────────────────────────
    // TEST 3: Full comparison — recurrent neuromorphic vs baselines
    // ─────────────────────────────────────────────────────────────
    fprintf(stderr, "\n═══ TEST 3: FULL COMPARISON (best params) ═══\n");
    {
        // Use moderate recurrence and coupling (will refine based on sweep)
        float best_rec = 0.1f, best_coup = 0.1f;

        struct Task {
            const char* name;
            int nc;
            void (*gen)(float*, int*, int, int);
            int gen_param;
        };
        Task tasks[] = {
            {"4-class waveform", 4, nullptr, 0},
            {"8-class waveform", 8, nullptr, 0},
            {"XOR τ=1", 2, gen_xor, 1},
            {"XOR τ=2", 2, gen_xor, 2},
            {"XOR τ=3", 2, gen_xor, 3},
            {"XOR τ=5", 2, gen_xor, 5},
            {"XOR τ=10", 2, gen_xor, 10},
            {"NARMA-3", 4, gen_narma, 3},
            {"NARMA-5", 4, gen_narma, 5},
        };
        int n_tasks = 9;

        for (int ti = 0; ti < n_tasks; ti++) {
            int ns = 200, nc = tasks[ti].nc;
            float* h_sig = (float*)malloc(ns * N_STEPS * sizeof(float));
            int* h_lab = (int*)malloc(ns * sizeof(int));
            float* all_f = (float*)malloc(ns * nf * sizeof(float));

            if (tasks[ti].gen == nullptr)
                gen_waveforms(h_sig, h_lab, ns, nc);
            else
                tasks[ti].gen(h_sig, h_lab, ns, tasks[ti].gen_param);

            // Run each method 3 times
            float acc_neuro = 0, acc_lif = 0, acc_esn = 0;

            for (int trial = 0; trial < 3; trial++) {
                // Neuromorphic recurrent
                for (int s = 0; s < ns; s++) {
                    HIP_CHECK(hipMemcpy(d_sig, &h_sig[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
                    HIP_CHECK(hipMemset(d_atomic, 0, 16*sizeof(unsigned int)));
                    k_neuromorphic_recurrent<<<1,nn>>>(d_sig, d_act, nn, N_STEPS, best_rec, best_coup, d_atomic);
                    HIP_CHECK(hipDeviceSynchronize());
                    k_features_rich<<<1,nn>>>(d_act, d_feat, nn, N_STEPS);
                    HIP_CHECK(hipDeviceSynchronize());
                    HIP_CHECK(hipMemcpy(&all_f[s*nf], d_feat, nf*sizeof(float), hipMemcpyDeviceToHost));
                }
                acc_neuro += classify(all_f, h_lab, ns, nf, nc);

                // LIF recurrent (software baseline with same topology)
                for (int s = 0; s < ns; s++) {
                    HIP_CHECK(hipMemcpy(d_sig, &h_sig[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
                    k_lif_recurrent<<<1,nn>>>(d_sig, d_act, nn, N_STEPS, 0.15f, 0.5f, 0.15f, best_rec, trial*10000+s);
                    HIP_CHECK(hipDeviceSynchronize());
                    k_features_rich<<<1,nn>>>(d_act, d_feat, nn, N_STEPS);
                    HIP_CHECK(hipDeviceSynchronize());
                    HIP_CHECK(hipMemcpy(&all_f[s*nf], d_feat, nf*sizeof(float), hipMemcpyDeviceToHost));
                }
                acc_lif += classify(all_f, h_lab, ns, nf, nc);

                // ESN recurrent
                for (int s = 0; s < ns; s++) {
                    HIP_CHECK(hipMemcpy(d_sig, &h_sig[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
                    k_esn_recurrent<<<1,nn>>>(d_sig, d_act, nn, N_STEPS, 0.95f, 0.8f, 0.65f);
                    HIP_CHECK(hipDeviceSynchronize());
                    k_features_rich<<<1,nn>>>(d_act, d_feat, nn, N_STEPS);
                    HIP_CHECK(hipDeviceSynchronize());
                    HIP_CHECK(hipMemcpy(&all_f[s*nf], d_feat, nf*sizeof(float), hipMemcpyDeviceToHost));
                }
                acc_esn += classify(all_f, h_lab, ns, nf, nc);
            }

            fprintf(stderr, "  %-20s  NEURO=%.3f  LIF=%.3f  ESN=%.3f  Δ(N-L)=%+.1fpp  Δ(N-E)=%+.1fpp\n",
                    tasks[ti].name,
                    acc_neuro/3, acc_lif/3, acc_esn/3,
                    (acc_neuro - acc_lif)/3*100,
                    (acc_neuro - acc_esn)/3*100);

            free(h_sig); free(h_lab); free(all_f);
        }
    }

    // ─────────────────────────────────────────────────────────────
    // TEST 4: Temporal memory capacity (delay 1..15)
    // ─────────────────────────────────────────────────────────────
    fprintf(stderr, "\n═══ TEST 4: MEMORY CAPACITY (XOR delay 1..15) ═══\n");
    fprintf(stderr, "  delay  NEURO   LIF     ESN\n");
    {
        for (int delay = 1; delay <= 15; delay++) {
            int ns = 200, nc = 2;
            float* h_sig = (float*)malloc(ns * N_STEPS * sizeof(float));
            int* h_lab = (int*)malloc(ns * sizeof(int));
            float* all_f = (float*)malloc(ns * nf * sizeof(float));

            gen_xor(h_sig, h_lab, ns, delay);

            float acc_n = 0, acc_l = 0, acc_e = 0;
            for (int trial = 0; trial < 3; trial++) {
                for (int s = 0; s < ns; s++) {
                    HIP_CHECK(hipMemcpy(d_sig, &h_sig[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
                    HIP_CHECK(hipMemset(d_atomic, 0, 16*sizeof(unsigned int)));
                    k_neuromorphic_recurrent<<<1,nn>>>(d_sig, d_act, nn, N_STEPS, 0.1f, 0.1f, d_atomic);
                    HIP_CHECK(hipDeviceSynchronize());
                    k_features_rich<<<1,nn>>>(d_act, d_feat, nn, N_STEPS);
                    HIP_CHECK(hipDeviceSynchronize());
                    HIP_CHECK(hipMemcpy(&all_f[s*nf], d_feat, nf*sizeof(float), hipMemcpyDeviceToHost));
                }
                acc_n += classify(all_f, h_lab, ns, nf, nc);

                for (int s = 0; s < ns; s++) {
                    HIP_CHECK(hipMemcpy(d_sig, &h_sig[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
                    k_lif_recurrent<<<1,nn>>>(d_sig, d_act, nn, N_STEPS, 0.15f, 0.5f, 0.15f, 0.1f, trial*10000+s);
                    HIP_CHECK(hipDeviceSynchronize());
                    k_features_rich<<<1,nn>>>(d_act, d_feat, nn, N_STEPS);
                    HIP_CHECK(hipDeviceSynchronize());
                    HIP_CHECK(hipMemcpy(&all_f[s*nf], d_feat, nf*sizeof(float), hipMemcpyDeviceToHost));
                }
                acc_l += classify(all_f, h_lab, ns, nf, nc);

                for (int s = 0; s < ns; s++) {
                    HIP_CHECK(hipMemcpy(d_sig, &h_sig[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
                    k_esn_recurrent<<<1,nn>>>(d_sig, d_act, nn, N_STEPS, 0.95f, 0.8f, 0.65f);
                    HIP_CHECK(hipDeviceSynchronize());
                    k_features_rich<<<1,nn>>>(d_act, d_feat, nn, N_STEPS);
                    HIP_CHECK(hipDeviceSynchronize());
                    HIP_CHECK(hipMemcpy(&all_f[s*nf], d_feat, nf*sizeof(float), hipMemcpyDeviceToHost));
                }
                acc_e += classify(all_f, h_lab, ns, nf, nc);
            }

            fprintf(stderr, "  %5d  %.3f   %.3f   %.3f\n", delay, acc_n/3, acc_l/3, acc_e/3);
            free(h_sig); free(h_lab); free(all_f);
        }
    }

    // ─────────────────────────────────────────────────────────────
    // TEST 5: Reproducibility — is recurrent physics still deterministic?
    // ─────────────────────────────────────────────────────────────
    fprintf(stderr, "\n═══ TEST 5: REPRODUCIBILITY (same input, 5 runs) ═══\n");
    {
        int ns = 100, nc = 4;
        float* h_sig = (float*)malloc(ns * N_STEPS * sizeof(float));
        int* h_lab = (int*)malloc(ns * sizeof(int));
        float* all_f = (float*)malloc(ns * nf * sizeof(float));

        gen_waveforms(h_sig, h_lab, ns, nc);

        fprintf(stderr, "  NEUROMORPHIC_RECURRENT: ");
        for (int run = 0; run < 5; run++) {
            for (int s = 0; s < ns; s++) {
                HIP_CHECK(hipMemcpy(d_sig, &h_sig[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemset(d_atomic, 0, 16*sizeof(unsigned int)));
                k_neuromorphic_recurrent<<<1,nn>>>(d_sig, d_act, nn, N_STEPS, 0.1f, 0.1f, d_atomic);
                HIP_CHECK(hipDeviceSynchronize());
                k_features_rich<<<1,nn>>>(d_act, d_feat, nn, N_STEPS);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(&all_f[s*nf], d_feat, nf*sizeof(float), hipMemcpyDeviceToHost));
            }
            float acc = classify(all_f, h_lab, ns, nf, nc);
            fprintf(stderr, "%.3f ", acc);
        }
        fprintf(stderr, "\n");

        fprintf(stderr, "  LIF_RECURRENT:          ");
        for (int run = 0; run < 5; run++) {
            for (int s = 0; s < ns; s++) {
                HIP_CHECK(hipMemcpy(d_sig, &h_sig[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
                k_lif_recurrent<<<1,nn>>>(d_sig, d_act, nn, N_STEPS, 0.15f, 0.5f, 0.15f, 0.1f, run*10000+s);
                HIP_CHECK(hipDeviceSynchronize());
                k_features_rich<<<1,nn>>>(d_act, d_feat, nn, N_STEPS);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(&all_f[s*nf], d_feat, nf*sizeof(float), hipMemcpyDeviceToHost));
            }
            float acc = classify(all_f, h_lab, ns, nf, nc);
            fprintf(stderr, "%.3f ", acc);
        }
        fprintf(stderr, "\n");

        free(h_sig); free(h_lab); free(all_f);
    }

    // ─────────────────────────────────────────────────────────────
    // TEST 6: Feature ablation — which features matter?
    // ─────────────────────────────────────────────────────────────
    fprintf(stderr, "\n═══ TEST 6: FEATURE ABLATION (8-class, recurrent neuro) ═══\n");
    {
        int ns = 200, nc = 8;
        float* h_sig = (float*)malloc(ns * N_STEPS * sizeof(float));
        int* h_lab = (int*)malloc(ns * sizeof(int));

        gen_waveforms(h_sig, h_lab, ns, nc);

        // Full features
        float* all_f = (float*)malloc(ns * nf * sizeof(float));
        for (int s = 0; s < ns; s++) {
            HIP_CHECK(hipMemcpy(d_sig, &h_sig[s*N_STEPS], N_STEPS*sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemset(d_atomic, 0, 16*sizeof(unsigned int)));
            k_neuromorphic_recurrent<<<1,nn>>>(d_sig, d_act, nn, N_STEPS, 0.1f, 0.1f, d_atomic);
            HIP_CHECK(hipDeviceSynchronize());
            k_features_rich<<<1,nn>>>(d_act, d_feat, nn, N_STEPS);
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(&all_f[s*nf], d_feat, nf*sizeof(float), hipMemcpyDeviceToHost));
        }

        // Full 12 features
        float* copy = (float*)malloc(ns * nf * sizeof(float));
        memcpy(copy, all_f, ns * nf * sizeof(float));
        float full_acc = classify(copy, h_lab, ns, nf, nc);
        fprintf(stderr, "  All 12 features:        %.3f\n", full_acc);

        // Only basic (mean, std, range) — 3 features per neuron
        int nf3 = nn * 3;
        float* f3 = (float*)malloc(ns * nf3 * sizeof(float));
        for (int i = 0; i < ns; i++)
            for (int n = 0; n < nn; n++) {
                f3[i*nf3 + n*3 + 0] = all_f[i*nf + n*N_FEAT + 0];
                f3[i*nf3 + n*3 + 1] = all_f[i*nf + n*N_FEAT + 1];
                f3[i*nf3 + n*3 + 2] = all_f[i*nf + n*N_FEAT + 2];
            }
        float acc3 = classify(f3, h_lab, ns, nf3, nc);
        fprintf(stderr, "  Basic only (3/neuron):  %.3f  (Δ=%+.1fpp)\n", acc3, (acc3-full_acc)*100);
        free(f3);

        // Only temporal (q1-q4, deriv, zc, ac, slope) — 8 features per neuron
        int nf8 = nn * 8;
        float* f8 = (float*)malloc(ns * nf8 * sizeof(float));
        for (int i = 0; i < ns; i++)
            for (int n = 0; n < nn; n++) {
                f8[i*nf8 + n*8 + 0] = all_f[i*nf + n*N_FEAT + 3];  // q1
                f8[i*nf8 + n*8 + 1] = all_f[i*nf + n*N_FEAT + 4];  // q2
                f8[i*nf8 + n*8 + 2] = all_f[i*nf + n*N_FEAT + 5];  // q3
                f8[i*nf8 + n*8 + 3] = all_f[i*nf + n*N_FEAT + 6];  // q4
                f8[i*nf8 + n*8 + 4] = all_f[i*nf + n*N_FEAT + 7];  // deriv
                f8[i*nf8 + n*8 + 5] = all_f[i*nf + n*N_FEAT + 8];  // volatility
                f8[i*nf8 + n*8 + 6] = all_f[i*nf + n*N_FEAT + 9];  // zero crossings
                f8[i*nf8 + n*8 + 7] = all_f[i*nf + n*N_FEAT + 10]; // autocorrelation
            }
        float acc8 = classify(f8, h_lab, ns, nf8, nc);
        fprintf(stderr, "  Temporal only (8/n):    %.3f  (Δ=%+.1fpp)\n", acc8, (acc8-full_acc)*100);
        free(f8);

        free(all_f); free(copy);
        free(h_sig); free(h_lab);
    }

    fprintf(stderr, "\n═══ DONE ═══\n");

    hipFree(d_sig); hipFree(d_act); hipFree(d_feat); hipFree(d_atomic);
    return 0;
}
