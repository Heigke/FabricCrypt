// jitter_deep_signal_probe.hip — Deep analog signal discovery in GPU silicon
//
// Tests 6 NEW noise sources beyond wall_clock64() PLL jitter:
//   1. PERF_SNAPSHOT (hwreg 27) — silicon performance counter, 66kHz entropy
//   2. SHADER_CYCLES (hwreg 29) — per-wave cycle counter, nanosecond jitter
//   3. Global atomic contention — L2 cache arbitration creates real HW noise
//   4. LDS bank conflict timing — shared memory bank conflicts as noise
//   5. Float denormal timing — subnormal operations have variable latency
//   6. GDS-like cross-CU atomic — chip-wide contention signal
//   7. Combined multi-source — fuse all sources for maximum entropy
//
// Each source is tested as input to a LIF spiking reservoir, measuring:
//   - Entropy (bits/sample)
//   - Autocorrelation (temporal structure)
//   - Per-neuron uniqueness (decorrelation)
//   - Reservoir computing performance (XOR, parity, NARMA)
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_deep_signal_probe scripts/jitter_deep_signal_probe.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_deep_signal_probe

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
#define N_NEURONS (N_CUS * THREADS)
#define STEPS 2000
#define WASHOUT 300
#define SAMPLE_N 96
#define NOISE_SOURCES 8

// ============================================================================
// NOISE SOURCE KERNELS — each extracts analog signal differently
// ============================================================================

// Source 0: PLL jitter only (baseline — our existing approach)
__device__ float noise_pll_jitter(int nid, int t) {
    uint64_t pll_bits = 0;
    for (int r = 0; r < 8; r++) {
        uint64_t w0 = wall_clock64();
        uint64_t w1 = wall_clock64();
        pll_bits = (pll_bits << 4) | ((w1 - w0) & 0xF);
    }
    uint32_t h = (uint32_t)(pll_bits ^ (uint64_t)(nid * 2654435761u));
    h ^= h >> 16; h *= 0x85ebca6b;
    h ^= h >> 13; h *= 0xc2b2ae35;
    h ^= h >> 16;
    return ((float)(h & 0xFFFF) / 32768.0f - 1.0f);
}

// Source 1: PERF_SNAPSHOT (hwreg 27) — silicon performance counter
// Proven: 66kHz, 16+ bits entropy, silicon physics TRNG (z2151)
__device__ float noise_perf_snapshot(int nid, int t) {
    uint32_t perf;
    asm volatile("s_getreg_b32 %0, hwreg(27)" : "=s"(perf));
    // Decorrelate per-neuron: hash with neuron ID
    uint32_t h = perf ^ (nid * 2654435761u) ^ (t * 1103515245u);
    h ^= h >> 16; h *= 0x85ebca6b;
    h ^= h >> 13; h *= 0xc2b2ae35;
    h ^= h >> 16;
    return ((float)(h & 0xFFFF) / 32768.0f - 1.0f);
}

// Source 2: SHADER_CYCLES (hwreg 29) — per-wave cycle counter
// The low bits accumulate differently per wave due to scheduling jitter
__device__ float noise_shader_cycles(int nid, int t) {
    uint32_t cycles;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(cycles));
    // Low bits have most jitter
    uint32_t h = (cycles & 0xFFFF) ^ (nid * 2654435761u) ^ (t * 0x45d9f3b);
    h ^= h >> 16; h *= 0x85ebca6b;
    h ^= h >> 13; h *= 0xc2b2ae35;
    h ^= h >> 16;
    return ((float)(h & 0xFFFF) / 32768.0f - 1.0f);
}

// Source 3: Global atomic contention — L2 cache arbitration noise
// Multiple CUs atomically incrementing same address creates real HW contention
// The ORDER of completion varies due to arbitration — this is genuine analog noise
__device__ float noise_atomic_contention(int nid, int t, volatile unsigned int* contention_buf) {
    // Each neuron atomically increments a shared counter
    // The returned OLD value depends on hardware arbitration order
    unsigned int old = atomicAdd((unsigned int*)&contention_buf[t % 32], 1u);
    // old varies from 0 to N_NEURONS depending on who got there first
    // This is genuine hardware arbitration noise
    uint32_t h = old ^ (nid * 2654435761u) ^ (t * 0x85ebca6b);
    h ^= h >> 16; h *= 0x45d9f3b;
    h ^= h >> 13; h *= 0xc2b2ae35;
    h ^= h >> 16;
    return ((float)(h & 0xFFFF) / 32768.0f - 1.0f);
}

// Source 4: LDS bank conflict timing noise
// Reading from addresses that conflict on LDS banks creates variable latency
// The timing delta encodes real analog pipeline state
__device__ float noise_lds_bank_conflict(int nid, int tid, int t) {
    __shared__ float lds_noise[THREADS * 4]; // 4 banks worth
    // Write to own slot
    lds_noise[tid] = (float)t;
    __syncthreads();

    // Read from conflicting bank (stride 17 = prime, guarantees bank conflicts on 32-bank LDS)
    uint64_t t0 = wall_clock64();
    float dummy = 0;
    for (int r = 0; r < 4; r++) {
        int conflict_idx = (tid * 17 + r * 37 + t) % (THREADS * 4);
        dummy += lds_noise[conflict_idx];
    }
    uint64_t t1 = wall_clock64();
    // Prevent optimization
    if (dummy == -999999.0f) lds_noise[0] = dummy;
    __syncthreads();

    // The timing delta has analog information about bank conflict resolution
    uint32_t delta = (uint32_t)(t1 - t0);
    uint32_t h = delta ^ (nid * 2654435761u) ^ (t * 1103515245u);
    h ^= h >> 16; h *= 0x85ebca6b;
    h ^= h >> 13; h *= 0xc2b2ae35;
    h ^= h >> 16;
    return ((float)(h & 0xFFFF) / 32768.0f - 1.0f);
}

// Source 5: Float denormal processing — subnormals take variable cycles
// On some GPUs, denormal floats flush-to-zero instantly; on others they're slow.
// The timing difference between normal and denormal ops is data-dependent.
__device__ float noise_denormal_timing(int nid, int t) {
    // Create a value near denormal boundary
    float tiny = 1.0e-38f;
    uint32_t seed = nid * 2654435761u + t * 0x45d9f3b;
    seed ^= seed >> 16; seed *= 0x85ebca6b;
    // Scale to straddle denormal boundary
    float scale = (float)(seed & 0xFF) / 256.0f;
    float val = tiny * scale; // some will be denormal, some not

    uint64_t t0 = wall_clock64();
    // Operations that behave differently on denormals
    float result = 0;
    for (int r = 0; r < 8; r++) {
        val = val * 0.5f + 1.0e-39f; // push toward denormal
        result += val * val; // multiply denormals
        val = result * 1.0e-30f;
    }
    uint64_t t1 = wall_clock64();

    // Prevent optimization
    uint32_t delta = (uint32_t)(t1 - t0);
    // Also mix in the actual denormal result bits
    uint32_t rbits;
    memcpy(&rbits, &result, sizeof(rbits));
    uint32_t h = delta ^ rbits ^ (nid * 2654435761u);
    h ^= h >> 16; h *= 0x85ebca6b;
    h ^= h >> 13; h *= 0xc2b2ae35;
    h ^= h >> 16;
    return ((float)(h & 0xFFFF) / 32768.0f - 1.0f);
}

// Source 6: Cross-CU global memory contention — chip-wide signal
// Different from source 3: uses MULTIPLE contention points with CU-dependent access patterns
__device__ float noise_cross_cu_contention(int nid, int bid, int tid, int t,
                                           volatile unsigned int* contention_buf) {
    // Each CU targets a different slot, but overlaps with neighbors
    // This creates a spatial pattern of contention across the chip
    int slot = (bid * 3 + tid / 16 + t) % 128; // 128 contention points
    unsigned int old = atomicAdd((unsigned int*)&contention_buf[256 + slot], 1u);

    // Also read neighbor CU's slot for cross-CU correlation
    int neighbor_slot = ((bid + 1) % N_CUS * 3 + tid / 16 + t) % 128;
    unsigned int neighbor = contention_buf[256 + neighbor_slot];

    uint32_t h = old ^ (neighbor << 16) ^ (nid * 2654435761u);
    h ^= h >> 16; h *= 0x85ebca6b;
    h ^= h >> 13; h *= 0xc2b2ae35;
    h ^= h >> 16;
    return ((float)(h & 0xFFFF) / 32768.0f - 1.0f);
}

// Source 7: COMBINED — fuse ALL sources for maximum entropy & temporal structure
__device__ float noise_combined(int nid, int bid, int tid, int t,
                                volatile unsigned int* contention_buf) {
    // Get raw bits from each source
    uint32_t perf;
    asm volatile("s_getreg_b32 %0, hwreg(27)" : "=s"(perf));
    uint32_t cycles;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(cycles));
    uint64_t wc = wall_clock64();

    unsigned int atomic_val = atomicAdd((unsigned int*)&contention_buf[t % 32], 1u);

    // Combine all entropy sources via mixing
    uint32_t h = perf;
    h ^= cycles * 0x85ebca6b;
    h ^= (uint32_t)wc * 0xc2b2ae35;
    h ^= atomic_val * 0x45d9f3b;
    h ^= (nid * 2654435761u);
    // Final avalanche mix
    h ^= h >> 16; h *= 0x85ebca6b;
    h ^= h >> 13; h *= 0xc2b2ae35;
    h ^= h >> 16;
    return ((float)(h & 0xFFFF) / 32768.0f - 1.0f);
}

// ============================================================================
// RAW NOISE SAMPLING KERNEL — captures raw noise values for analysis
// ============================================================================
__global__ void sample_noise_raw(
    float* noise_out,     // [N_NEURONS × STEPS]
    volatile unsigned int* contention_buf,
    int source_id, int steps
) {
    int bid = blockIdx.x, tid = threadIdx.x;
    if (bid >= N_CUS || tid >= THREADS) return;
    int nid = bid * THREADS + tid;

    for (int t = 0; t < steps; t++) {
        float n = 0;
        switch (source_id) {
            case 0: n = noise_pll_jitter(nid, t); break;
            case 1: n = noise_perf_snapshot(nid, t); break;
            case 2: n = noise_shader_cycles(nid, t); break;
            case 3: n = noise_atomic_contention(nid, t, contention_buf); break;
            case 4: n = noise_lds_bank_conflict(nid, tid, t); break;
            case 5: n = noise_denormal_timing(nid, t); break;
            case 6: n = noise_cross_cu_contention(nid, bid, tid, t, contention_buf); break;
            case 7: n = noise_combined(nid, bid, tid, t, contention_buf); break;
        }
        noise_out[nid * steps + t] = n;
        if (source_id == 4) __syncthreads(); // needed for LDS source
    }
}

// ============================================================================
// LIF RESERVOIR KERNEL — uses specified noise source
// ============================================================================
__global__ void lif_reservoir(
    float* state_out, float* spike_out,
    volatile float* cu_xchg, const float* input_seq,
    volatile unsigned int* contention_buf,
    int steps, int source_id, float noise_amp,
    float spectral_radius, float input_scale,
    float lif_alpha, float lif_threshold, float lif_reset, float bias_frac,
    float ei_ratio, float inh_gain
) {
    int bid = blockIdx.x, tid = threadIdx.x;
    if (bid >= N_CUS || tid >= THREADS) return;
    int nid = bid * THREADS + tid;
    __shared__ float lds[THREADS];

    // E/I weight matrix (Dale's law)
    float w_row[THREADS];
    float scale = spectral_radius / sqrtf((float)THREADS);
    for (int j = 0; j < THREADS; j++) {
        uint32_t h = (uint32_t)(bid*8713 + tid*5501 + j*4217 + 137);
        h ^= h>>16; h *= 0x45d9f3b; h ^= h>>16; h *= 0x45d9f3b; h ^= h>>16;
        float w = scale * ((float)(h & 0xFFFF)/32768.0f - 1.0f);
        uint32_t hj = j * 2654435761u + bid * 7919;
        hj ^= hj>>16; hj *= 0x45d9f3b;
        bool j_exc = ((float)(hj & 0xFFFF) / 65536.0f) < ei_ratio;
        w_row[j] = fabsf(w) * (j_exc ? 1.0f : -inh_gain);
    }

    uint32_t irng = nid * 2654435761u + 77773;
    irng ^= irng>>16; irng *= 0x45d9f3b; irng ^= irng>>16;
    float w_in = input_scale * ((float)(irng & 0xFFFF)/32768.0f - 1.0f);
    irng *= 0x45d9f3b; irng ^= irng>>16;
    float bias = lif_threshold * (bias_frac - 0.1f + 0.2f * ((float)(irng & 0xFFFF)/65536.0f));
    irng *= 1103515245; int rcu = ((irng>>8)&0xFF)%N_CUS;
    irng *= 1103515245; int rtid = ((irng>>8)&0xFF)%THREADS;
    irng *= 1103515245; float w_cross = 0.03f * ((float)(irng & 0xFFFF)/32768.0f - 1.0f);
    irng *= 1103515245;
    float neuron_thresh = lif_threshold * (0.85f + 0.30f * ((float)(irng & 0xFFFF)/65536.0f));

    float v = 0.0f, spike = 0.0f;
    int refractory = 0;

    for (int t = 0; t < steps; t++) {
        // Get noise from selected source
        float noise = 0.0f;
        if (source_id >= 0) {
            switch (source_id) {
                case 0: noise = noise_pll_jitter(nid, t); break;
                case 1: noise = noise_perf_snapshot(nid, t); break;
                case 2: noise = noise_shader_cycles(nid, t); break;
                case 3: noise = noise_atomic_contention(nid, t, contention_buf); break;
                case 4: noise = noise_lds_bank_conflict(nid, tid, t); break;
                case 5: noise = noise_denormal_timing(nid, t); break;
                case 6: noise = noise_cross_cu_contention(nid, bid, tid, t, contention_buf); break;
                case 7: noise = noise_combined(nid, bid, tid, t, contention_buf); break;
            }
        }

        lds[tid] = spike;
        __syncthreads();
        float rec = 0;
        for (int j = 0; j < THREADS; j++) rec += w_row[j] * lds[j];
        cu_xchg[nid] = spike;
        __threadfence();
        float cross = w_cross * cu_xchg[rcu*THREADS+rtid];
        float I_syn = rec + cross + w_in * input_seq[t];

        spike = 0.0f;
        if (refractory > 0) { refractory--; v *= (1.0f - lif_alpha); }
        else {
            v = v * (1.0f - lif_alpha) + I_syn + bias * lif_alpha;
            if (source_id >= 0) v += noise_amp * noise;
            float thresh = neuron_thresh;
            if (source_id >= 0) thresh += noise_amp * 0.2f * noise;
            if (v >= thresh) { spike = 1.0f; v = lif_reset; refractory = 2; }
        }
        state_out[nid * steps + t] = v;
        spike_out[nid * steps + t] = spike;
        __syncthreads();
    }
}

// ============================================================================
// HOST UTILITIES
// ============================================================================
static uint32_t host_rng = 42;
float randf(){host_rng=host_rng*1103515245+12345;return(float)((host_rng>>16)&0x7FFF)/32767.0f;}

void extract_lif_features(float* state, float* spikes, int n, int steps, int t, std::vector<float>& f) {
    int stride = n/SAMPLE_N; if(stride<1)stride=1;
    f.clear();
    for(int si=0;si<SAMPLE_N&&si*stride<n;si++){
        float v = state[si*stride*steps+t];
        if(v!=v||v>1e20||v<-1e20)v=0;
        f.push_back(v);
        // Recent spike count (temporal feature)
        float sc=0;
        for(int dt=0;dt<10&&t-dt>=0;dt++) sc += spikes[si*stride*steps+(t-dt)];
        f.push_back(sc/10.0f);
    }
    // Cross-neuron products
    for(int si=0;si<SAMPLE_N-8&&si*stride<n;si+=8){
        int i1=si*stride,i2=(si+8)*stride;
        if(i2>=n)break;
        float v1=state[i1*steps+t],v2=state[i2*steps+t];
        if(v1!=v1||v1>1e20)v1=0;if(v2!=v2||v2>1e20)v2=0;
        f.push_back(v1*v2);
    }
}

float ridge_acc(std::vector<std::vector<float>>& X, std::vector<int>& Y, int n_classes) {
    if(X.empty()||Y.empty()) return 0;
    int n=X.size(), d=X[0].size();
    int tr = n*3/4, te = n-tr;
    if(tr<10||te<5||d<2) return 0;
    // Ridge regression one-vs-all
    float lambda=1.0f;
    std::vector<std::vector<float>> W(n_classes, std::vector<float>(d,0));
    // Simple closed-form: W = (X^T X + λI)^(-1) X^T Y — use gradient descent for speed
    for(int c=0;c<n_classes;c++){
        std::vector<float> w(d,0);
        for(int iter=0;iter<200;iter++){
            float lr = 0.001f/(1.0f+iter*0.01f);
            for(int i=0;i<tr;i++){
                float pred=0;
                for(int j=0;j<d;j++) pred+=w[j]*X[i][j];
                float target = (Y[i]==c)?1.0f:0.0f;
                float err = pred-target;
                for(int j=0;j<d;j++) w[j] -= lr*(err*X[i][j] + lambda*w[j]/tr);
            }
        }
        W[c]=w;
    }
    int correct=0;
    for(int i=tr;i<n;i++){
        float best=-1e30; int bc=0;
        for(int c=0;c<n_classes;c++){
            float s=0;
            for(int j=0;j<d;j++) s+=W[c][j]*X[i][j];
            if(s>best){best=s;bc=c;}
        }
        if(bc==Y[i])correct++;
    }
    return (float)correct/te;
}

float ridge_narma(std::vector<std::vector<float>>& X, std::vector<float>& Y) {
    if(X.empty()||Y.empty()) return 0;
    int n=X.size(), d=X[0].size();
    int tr = n*3/4, te = n-tr;
    if(tr<10||te<5||d<2) return 0;
    float lambda=1.0f;
    std::vector<float> w(d,0);
    for(int iter=0;iter<300;iter++){
        float lr = 0.001f/(1.0f+iter*0.01f);
        for(int i=0;i<tr;i++){
            float pred=0;
            for(int j=0;j<d;j++) pred+=w[j]*X[i][j];
            float err = pred-Y[i];
            for(int j=0;j<d;j++) w[j] -= lr*(err*X[i][j] + lambda*w[j]/tr);
        }
    }
    float ss_res=0,ss_tot=0,ymean=0;
    for(int i=tr;i<n;i++) ymean+=Y[i];
    ymean/=te;
    for(int i=tr;i<n;i++){
        float pred=0;
        for(int j=0;j<d;j++) pred+=w[j]*X[i][j];
        ss_res+=(pred-Y[i])*(pred-Y[i]);
        ss_tot+=(Y[i]-ymean)*(Y[i]-ymean);
    }
    return (ss_tot>1e-12)?1.0f-ss_res/ss_tot:-1.0f;
}

int main() {
    printf("=== DEEP ANALOG SIGNAL PROBE — 8 NOISE SOURCES ===\n\n");

    const char* source_names[NOISE_SOURCES] = {
        "PLL_JITTER (wall_clock64)",
        "PERF_SNAPSHOT (hwreg 27)",
        "SHADER_CYCLES (hwreg 29)",
        "ATOMIC_CONTENTION (L2 arb)",
        "LDS_BANK_CONFLICT (timing)",
        "DENORMAL_TIMING (float ops)",
        "CROSS_CU_CONTENTION (chip-wide)",
        "COMBINED (all fused)"
    };

    // Allocate GPU memory
    size_t sz = N_NEURONS * STEPS * sizeof(float);
    float *d_noise, *d_state, *d_spike, *d_input;
    volatile float *d_xchg;
    volatile unsigned int *d_contention;
    hipMalloc(&d_noise, sz);
    hipMalloc(&d_state, sz);
    hipMalloc(&d_spike, sz);
    hipMalloc((void**)&d_xchg, N_NEURONS * sizeof(float));
    hipMalloc(&d_input, STEPS * sizeof(float));
    hipMalloc((void**)&d_contention, 512 * sizeof(unsigned int)); // contention buffer

    float* h_noise = (float*)malloc(sz);
    float* h_state = (float*)malloc(sz);
    float* h_spike = (float*)malloc(sz);

    // ========================================================================
    // PART 1: RAW NOISE ANALYSIS — entropy, autocorrelation, per-neuron uniqueness
    // ========================================================================
    printf("--- PART 1: RAW NOISE CHARACTERIZATION ---\n\n");
    printf("%-35s | Entropy  | ACF(1)   | ACF(10)  | Unique/N | StdDev   | PSD slope\n", "Source");
    printf("------------------------------------+----------+----------+----------+----------+----------+----------\n");

    for (int src = 0; src < NOISE_SOURCES; src++) {
        hipMemset((void*)d_contention, 0, 512 * sizeof(unsigned int));
        hipDeviceSynchronize();

        sample_noise_raw<<<N_CUS, THREADS>>>(d_noise, d_contention, src, STEPS);
        hipDeviceSynchronize();
        hipMemcpy(h_noise, d_noise, sz, hipMemcpyDeviceToHost);

        // Analyze noise from neuron 0 for basic stats
        std::vector<float> vals(STEPS);
        for (int t = 0; t < STEPS; t++) vals[t] = h_noise[t]; // neuron 0

        // Entropy: bin into 256 bins and compute Shannon entropy
        int bins[256] = {};
        for (int t = 0; t < STEPS; t++) {
            int b = (int)((vals[t] + 1.0f) * 127.5f);
            b = std::max(0, std::min(255, b));
            bins[b]++;
        }
        float entropy = 0;
        for (int b = 0; b < 256; b++) {
            if (bins[b] > 0) {
                float p = (float)bins[b] / STEPS;
                entropy -= p * log2f(p);
            }
        }

        // Autocorrelation at lag 1 and 10
        float mean = 0, var = 0;
        for (int t = 0; t < STEPS; t++) mean += vals[t];
        mean /= STEPS;
        for (int t = 0; t < STEPS; t++) var += (vals[t]-mean)*(vals[t]-mean);
        var /= STEPS;

        float acf1 = 0, acf10 = 0;
        if (var > 1e-12) {
            for (int t = 0; t < STEPS-1; t++) acf1 += (vals[t]-mean)*(vals[t+1]-mean);
            acf1 /= (STEPS-1) * var;
            for (int t = 0; t < STEPS-10; t++) acf10 += (vals[t]-mean)*(vals[t+10]-mean);
            acf10 /= (STEPS-10) * var;
        }

        // Per-neuron uniqueness: how many unique sequences across neurons?
        // Compare first 100 values of 8 randomly chosen neurons
        int n_unique = 0;
        int check_neurons[] = {0, 1, 64, 65, 128, 256, 384, 512};
        for (int ni = 0; ni < 8; ni++) {
            int nid = check_neurons[ni];
            if (nid >= N_NEURONS) continue;
            bool is_unique = true;
            for (int nj = 0; nj < ni; nj++) {
                int nid2 = check_neurons[nj];
                float diff = 0;
                for (int t = 0; t < 100; t++) {
                    float d = h_noise[nid * STEPS + t] - h_noise[nid2 * STEPS + t];
                    diff += d * d;
                }
                if (diff < 0.01f) { is_unique = false; break; }
            }
            if (is_unique) n_unique++;
        }

        // StdDev
        float stddev = sqrtf(var);

        // Simple PSD slope estimate (log-log regression of FFT magnitude)
        // Use Welch-like approach: split into 4 segments, average
        float psd_slope = 0;
        {
            int seg_len = STEPS / 4;
            std::vector<float> psd(seg_len/2, 0);
            for (int seg = 0; seg < 4; seg++) {
                for (int k = 1; k < seg_len/2; k++) {
                    float re = 0, im = 0;
                    for (int t = 0; t < seg_len; t++) {
                        float v = vals[seg*seg_len + t];
                        float angle = -2.0f * M_PI * k * t / seg_len;
                        re += v * cosf(angle);
                        im += v * sinf(angle);
                    }
                    psd[k] += (re*re + im*im) / (seg_len * 4);
                }
            }
            // Log-log linear regression
            float sx=0,sy=0,sxx=0,sxy=0;
            int cnt=0;
            for (int k = 1; k < seg_len/2; k++) {
                if (psd[k] > 1e-20) {
                    float lx = log2f((float)k);
                    float ly = log2f(psd[k]);
                    sx+=lx; sy+=ly; sxx+=lx*lx; sxy+=lx*ly; cnt++;
                }
            }
            if (cnt > 2) {
                psd_slope = (cnt*sxy - sx*sy) / (cnt*sxx - sx*sx);
            }
        }

        printf("%-35s | %6.3f   | %+6.3f   | %+6.3f   |   %d/%d    | %6.4f   | %+6.3f\n",
               source_names[src], entropy, acf1, acf10, n_unique, 8, stddev, psd_slope);
    }

    // ========================================================================
    // PART 2: RESERVOIR COMPUTING — each source drives LIF reservoir
    // ========================================================================
    printf("\n--- PART 2: LIF RESERVOIR PERFORMANCE PER SOURCE ---\n\n");

    // Best LIF params from v2/E/I experiments
    float sr = 1.0f, is_ = 1.0f;
    float alpha = 0.20f, thresh = 0.5f, reset = 0.0f, bias_frac = 0.95f;
    float ei_ratio = 0.90f, inh_gain = 5.0f;
    float noise_amp = 0.03f;

    // Generate input + targets
    float inp_cont[STEPS];
    for (int t = 0; t < STEPS; t++) {
        inp_cont[t] = sinf(2.0f * M_PI * t / 50.0f) * 0.3f +
                      sinf(2.0f * M_PI * t / 130.0f) * 0.2f +
                      sinf(2.0f * M_PI * t / 17.0f) * 0.15f;
    }

    // XOR-8 targets
    int xor8_class[STEPS];
    for (int t = 0; t < STEPS; t++) {
        int b = 0;
        for (int d = 0; d < 8; d++) {
            float th = -0.15f + d * 0.0375f;
            if (t-d >= 0 && inp_cont[t-d] > th) b ^= 1;
        }
        xor8_class[t] = b;
    }

    // Parity-3
    int par3_class[STEPS];
    for (int t = 0; t < STEPS; t++) {
        int b = 0;
        for (int d = 0; d < 3; d++) {
            if (t-d >= 0 && inp_cont[t-d] > 0) b ^= 1;
        }
        par3_class[t] = b;
    }

    // NARMA-5
    float narma[STEPS];
    narma[0] = 0;
    for (int t = 1; t < STEPS; t++) {
        float y = narma[t-1];
        float u = (inp_cont[t] + 1.0f) * 0.5f;
        float sum5 = 0;
        for (int d = 0; d < 5 && t-d >= 0; d++) sum5 += narma[t-d];
        narma[t] = 0.3f*y + 0.05f*y*sum5 + 1.5f*u*(t>0?u:0) + 0.1f;
        if (narma[t] != narma[t] || narma[t] > 10) narma[t] = 0.5f;
    }

    // Memory capacity
    // (measure how well reservoir can reconstruct delayed input)

    printf("%-35s | XOR-8    | Par3     | NARMA    | MC(d=1)  | SpkRate  | SpkVar\n", "Source");
    printf("------------------------------------+----------+----------+----------+----------+----------+--------\n");

    // First: NO NOISE baseline
    {
        hipMemcpy(d_input, inp_cont, STEPS*sizeof(float), hipMemcpyHostToDevice);
        hipMemset(d_state, 0, sz);
        hipMemset(d_spike, 0, sz);
        hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
        hipMemset((void*)d_contention, 0, 512*sizeof(unsigned int));

        lif_reservoir<<<N_CUS, THREADS>>>(
            d_state, d_spike, d_xchg, d_input, d_contention,
            STEPS, -1, 0, sr, is_, alpha, thresh, reset, bias_frac, ei_ratio, inh_gain);
        hipDeviceSynchronize();
        hipMemcpy(h_state, d_state, sz, hipMemcpyDeviceToHost);
        hipMemcpy(h_spike, d_spike, sz, hipMemcpyDeviceToHost);

        // Spike stats
        float total_spikes = 0;
        std::vector<float> per_neuron_rate(N_NEURONS);
        for (int n = 0; n < N_NEURONS; n++) {
            float sc = 0;
            for (int t = WASHOUT; t < STEPS; t++) sc += h_spike[n*STEPS+t];
            per_neuron_rate[n] = sc / (STEPS-WASHOUT);
            total_spikes += sc;
        }
        float avg_rate = total_spikes / (N_NEURONS * (STEPS-WASHOUT));
        float rate_var = 0;
        for (int n = 0; n < N_NEURONS; n++) rate_var += (per_neuron_rate[n]-avg_rate)*(per_neuron_rate[n]-avg_rate);
        rate_var /= N_NEURONS;

        // Build features and evaluate
        std::vector<std::vector<float>> feats;
        std::vector<int> labels_x, labels_p;
        std::vector<float> labels_n, labels_mc;
        for (int t = WASHOUT; t < STEPS; t++) {
            std::vector<float> f;
            extract_lif_features(h_state, h_spike, N_NEURONS, STEPS, t, f);
            feats.push_back(f);
            labels_x.push_back(xor8_class[t]);
            labels_p.push_back(par3_class[t]);
            labels_n.push_back(narma[t]);
            if (t >= 1) labels_mc.push_back(inp_cont[t-1]);
        }

        float xor_acc = ridge_acc(feats, labels_x, 2);
        float par_acc = ridge_acc(feats, labels_p, 2);
        float narma_r2 = ridge_narma(feats, labels_n);

        // MC at delay 1
        std::vector<std::vector<float>> feats_mc(feats.begin()+1, feats.end());
        std::vector<float> mc_labels(labels_mc.begin(), labels_mc.end()-1);
        float mc1 = ridge_narma(feats_mc, mc_labels);

        printf("%-35s | %5.1f%%   | %5.1f%%   | %+6.3f   | %+6.3f   | %5.3f    | %6.4f\n",
               "NO_NOISE (baseline)", xor_acc*100, par_acc*100, narma_r2, mc1, avg_rate, sqrtf(rate_var));
    }

    // Then: each noise source
    for (int src = 0; src < NOISE_SOURCES; src++) {
        hipMemcpy(d_input, inp_cont, STEPS*sizeof(float), hipMemcpyHostToDevice);
        hipMemset(d_state, 0, sz);
        hipMemset(d_spike, 0, sz);
        hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
        hipMemset((void*)d_contention, 0, 512*sizeof(unsigned int));

        lif_reservoir<<<N_CUS, THREADS>>>(
            d_state, d_spike, d_xchg, d_input, d_contention,
            STEPS, src, noise_amp, sr, is_, alpha, thresh, reset, bias_frac, ei_ratio, inh_gain);
        hipDeviceSynchronize();
        hipMemcpy(h_state, d_state, sz, hipMemcpyDeviceToHost);
        hipMemcpy(h_spike, d_spike, sz, hipMemcpyDeviceToHost);

        float total_spikes = 0;
        std::vector<float> per_neuron_rate(N_NEURONS);
        for (int n = 0; n < N_NEURONS; n++) {
            float sc = 0;
            for (int t = WASHOUT; t < STEPS; t++) sc += h_spike[n*STEPS+t];
            per_neuron_rate[n] = sc / (STEPS-WASHOUT);
            total_spikes += sc;
        }
        float avg_rate = total_spikes / (N_NEURONS * (STEPS-WASHOUT));
        float rate_var = 0;
        for (int n = 0; n < N_NEURONS; n++) rate_var += (per_neuron_rate[n]-avg_rate)*(per_neuron_rate[n]-avg_rate);
        rate_var /= N_NEURONS;

        std::vector<std::vector<float>> feats;
        std::vector<int> labels_x, labels_p;
        std::vector<float> labels_n, labels_mc;
        for (int t = WASHOUT; t < STEPS; t++) {
            std::vector<float> f;
            extract_lif_features(h_state, h_spike, N_NEURONS, STEPS, t, f);
            feats.push_back(f);
            labels_x.push_back(xor8_class[t]);
            labels_p.push_back(par3_class[t]);
            labels_n.push_back(narma[t]);
            if (t >= 1) labels_mc.push_back(inp_cont[t-1]);
        }

        float xor_acc = ridge_acc(feats, labels_x, 2);
        float par_acc = ridge_acc(feats, labels_p, 2);
        float narma_r2 = ridge_narma(feats, labels_n);

        std::vector<std::vector<float>> feats_mc(feats.begin()+1, feats.end());
        std::vector<float> mc_labels(labels_mc.begin(), labels_mc.end()-1);
        float mc1 = ridge_narma(feats_mc, mc_labels);

        printf("%-35s | %5.1f%%   | %5.1f%%   | %+6.3f   | %+6.3f   | %5.3f    | %6.4f\n",
               source_names[src], xor_acc*100, par_acc*100, narma_r2, mc1, avg_rate, sqrtf(rate_var));
    }

    // ========================================================================
    // PART 3: NOISE AMPLITUDE SWEEP — find stochastic resonance per source
    // ========================================================================
    printf("\n--- PART 3: STOCHASTIC RESONANCE SWEEP (XOR-8 accuracy vs noise amplitude) ---\n\n");

    float amps[] = {0.001f, 0.003f, 0.01f, 0.03f, 0.05f, 0.1f, 0.2f, 0.5f};
    int n_amps = 8;
    // Test top 3 most promising sources + combined
    int test_sources[] = {1, 3, 6, 7}; // PERF_SNAPSHOT, ATOMIC, CROSS_CU, COMBINED
    int n_test = 4;

    printf("Amplitude  ");
    for (int si = 0; si < n_test; si++) printf("| %-12s ", source_names[test_sources[si]]);
    printf("\n-----------");
    for (int si = 0; si < n_test; si++) printf("+---------------");
    printf("\n");

    for (int ai = 0; ai < n_amps; ai++) {
        printf("%8.3f   ", amps[ai]);
        for (int si = 0; si < n_test; si++) {
            int src = test_sources[si];
            hipMemcpy(d_input, inp_cont, STEPS*sizeof(float), hipMemcpyHostToDevice);
            hipMemset(d_state, 0, sz);
            hipMemset(d_spike, 0, sz);
            hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
            hipMemset((void*)d_contention, 0, 512*sizeof(unsigned int));

            lif_reservoir<<<N_CUS, THREADS>>>(
                d_state, d_spike, d_xchg, d_input, d_contention,
                STEPS, src, amps[ai], sr, is_, alpha, thresh, reset, bias_frac, ei_ratio, inh_gain);
            hipDeviceSynchronize();
            hipMemcpy(h_state, d_state, sz, hipMemcpyDeviceToHost);
            hipMemcpy(h_spike, d_spike, sz, hipMemcpyDeviceToHost);

            std::vector<std::vector<float>> feats;
            std::vector<int> labels;
            for (int t = WASHOUT; t < STEPS; t++) {
                std::vector<float> f;
                extract_lif_features(h_state, h_spike, N_NEURONS, STEPS, t, f);
                feats.push_back(f);
                labels.push_back(xor8_class[t]);
            }
            float acc = ridge_acc(feats, labels, 2);
            printf("| %5.1f%%       ", acc*100);
        }
        printf("\n");
    }

    // ========================================================================
    // PART 4: CROSS-SOURCE CORRELATION — which sources are independent?
    // ========================================================================
    printf("\n--- PART 4: CROSS-SOURCE INDEPENDENCE (Pearson r between sources, neuron 0) ---\n\n");

    // Collect noise from all sources
    float* all_noise[NOISE_SOURCES];
    for (int src = 0; src < NOISE_SOURCES; src++) {
        all_noise[src] = (float*)malloc(STEPS * sizeof(float));
        hipMemset((void*)d_contention, 0, 512*sizeof(unsigned int));
        sample_noise_raw<<<N_CUS, THREADS>>>(d_noise, d_contention, src, STEPS);
        hipDeviceSynchronize();
        hipMemcpy(h_noise, d_noise, sz, hipMemcpyDeviceToHost);
        memcpy(all_noise[src], h_noise, STEPS * sizeof(float)); // neuron 0
    }

    printf("%-20s", "");
    for (int j = 0; j < NOISE_SOURCES; j++) printf("| %3d ", j);
    printf("\n");
    for (int i = 0; i < NOISE_SOURCES; i++) {
        printf("%-20s", source_names[i]);
        float mi = 0, vi = 0;
        for (int t = 0; t < STEPS; t++) mi += all_noise[i][t];
        mi /= STEPS;
        for (int t = 0; t < STEPS; t++) vi += (all_noise[i][t]-mi)*(all_noise[i][t]-mi);
        vi /= STEPS;

        for (int j = 0; j < NOISE_SOURCES; j++) {
            float mj = 0, vj = 0;
            for (int t = 0; t < STEPS; t++) mj += all_noise[j][t];
            mj /= STEPS;
            for (int t = 0; t < STEPS; t++) vj += (all_noise[j][t]-mj)*(all_noise[j][t]-mj);
            vj /= STEPS;

            float cov = 0;
            for (int t = 0; t < STEPS; t++) cov += (all_noise[i][t]-mi)*(all_noise[j][t]-mj);
            cov /= STEPS;
            float r = (vi > 1e-12 && vj > 1e-12) ? cov / sqrtf(vi*vj) : 0;
            printf("|%+5.2f", r);
        }
        printf("\n");
    }

    for (int src = 0; src < NOISE_SOURCES; src++) free(all_noise[src]);

    printf("\nDone.\n");

    free(h_noise); free(h_state); free(h_spike);
    hipFree(d_noise); hipFree(d_state); hipFree(d_spike);
    hipFree((void*)d_xchg); hipFree(d_input); hipFree((void*)d_contention);
    return 0;
}
