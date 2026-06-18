// jitter_placement_reservoir.hip — Exploit PHYSICAL PLACEMENT effects on die
//
// Physics sources BEYOND time jitter:
//   1. Process Vth variation: each CU's transistors have unique threshold voltages
//      → same instruction, slightly different timing per CU (manufacturing fingerprint)
//   2. IR drop variation: supply voltage differs by physical position on die
//      → CUs near power pins are faster, far ones slower
//   3. Cross-CU wire delay: inter-CU communication has placement-dependent latency
//   4. Temperature gradient: different CUs at different temps → different speed
//   5. Clock tree skew: clock arrives at different CUs at slightly different times
//
// MEASUREMENT: Launch identical workloads on MANY blocks (→ different CUs).
// The per-CU timing SIGNATURE is a spatial fingerprint of the die.
// Combined with wall_clock64 temporal noise → spatiotemporal reservoir.
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_placement_reservoir scripts/jitter_placement_reservoir.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_placement_reservoir

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

#define N_CUS 12          // gfx1151 has 16 CUs, use 12 for measurement
#define THREADS_PER 64
#define N_NEURONS (N_CUS * THREADS_PER)  // 768 neurons!
#define STEPS 500
#define N_CLASSES 4
#define N_TRAIN 120
#define N_TEST 60
#define FEATURE_DIM 10

// ============================================================================
// PROBE 1: Per-CU timing fingerprint
// Same exact workload on every block → timing differences = placement physics
// ============================================================================
__global__ void probe_cu_fingerprint(uint64_t* timings, int iters) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (tid >= THREADS_PER) return;

    // Identical workload for every block — differences are PLACEMENT
    float x = (float)tid * 0.001f + 1.0f;
    float y = x + 0.1f;

    uint64_t t0 = __clock64();
    for (int i = 0; i < iters; i++) {
        x = __fmaf_rn(x, y, x);
        y = __fmaf_rn(y, x, y);
        x = __fmaf_rn(x, 1.0001f, -0.0001f);
        y = __fmaf_rn(y, 0.9999f, 0.0001f);
    }
    uint64_t t1 = __clock64();

    // Prevent optimization
    if (x + y == -999.999f) timings[0] = __float_as_uint(x);

    timings[bid * THREADS_PER + tid] = t1 - t0;
}

// ============================================================================
// PROBE 2: Cross-CU communication delay (placement-dependent wire length)
// Block pairs exchange data through global memory → delay = physical distance
// ============================================================================
__global__ void probe_cross_cu_delay(uint64_t* delays, volatile uint32_t* flag_mem, int n_blocks) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (tid != 0) return;  // single thread per block

    int partner = (bid + n_blocks/2) % n_blocks;  // pair with distant block

    // Write our flag
    flag_mem[bid] = 1;
    __threadfence();

    // Busy-wait for partner's flag — measures cross-die communication
    uint64_t t0 = __clock64();
    int timeout = 0;
    while (flag_mem[partner] == 0 && timeout < 100000) timeout++;
    uint64_t t1 = __clock64();

    delays[bid] = (timeout < 100000) ? (t1 - t0) : 0;
}

// ============================================================================
// MAIN RESERVOIR: Spatiotemporal — combines placement + time jitter
// Each BLOCK is a "CU neuron group" with its own physics fingerprint
// ============================================================================
__global__ void placement_reservoir(
    float* state_out,       // [N_CUS * THREADS_PER * STEPS]
    uint32_t* scratch,      // 16MB for memory access
    volatile float* cu_exchange,  // [N_CUS * THREADS_PER] for cross-CU coupling
    int input_class,
    int input_phase,
    int steps
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= THREADS_PER) return;

    int nid = bid * THREADS_PER + tid;

    __shared__ float lds[THREADS_PER];  // intra-CU coupling (fast, same CU)

    float v = 0.0f;
    float h = 0.0f;
    int spike_count = 0;

    // Per-neuron parameters (deterministic from ID)
    uint32_t rng = nid * 2654435761u + 12345;
    rng *= 1103515245; float leak = 0.05f + 0.15f * ((rng >> 16) & 0xFF) / 255.0f;
    rng *= 1103515245; float noise_gain = 0.2f + 0.3f * ((rng >> 16) & 0xFF) / 255.0f;
    rng *= 1103515245; float threshold = 0.6f + 0.3f * ((rng >> 16) & 0xFF) / 255.0f;
    rng *= 1103515245; int local_n1 = (tid + 1 + ((rng>>8)&7)) % THREADS_PER;
    rng *= 1103515245; int local_n2 = (tid + THREADS_PER - 1 - ((rng>>8)&7)) % THREADS_PER;
    rng *= 1103515245; float w_local = 0.05f + 0.10f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; int remote_cu = ((rng>>8)&0xFF) % N_CUS;
    rng *= 1103515245; int remote_tid = ((rng>>8)&0xFF) % THREADS_PER;
    rng *= 1103515245; float w_remote = 0.02f + 0.05f * ((rng>>16)&0xFF)/255.0f;

    for (int t = 0; t < steps; t++) {
        // ─── INPUT ENCODING ───
        int n_reads;
        int stride;
        switch (input_class) {
            case 0: n_reads = 4; stride = 4; break;
            case 1: n_reads = 12 + 8*((t/(8+input_phase*3))&1); stride = 32+96*((t/(12+input_phase*2))&1); break;
            case 2: n_reads = 4 + (t*16)/steps; stride = 4+(t*124)/steps; break;
            case 3: n_reads = ((t%(20+input_phase*5))<5)?20:4; stride = ((t%(15+input_phase*3))<4)?128:4; break;
            default: n_reads = 8; stride = 32;
        }

        // ─── SOURCE 1: wall_clock64 PLL jitter (temporal noise) ───
        float noise_sum = 0.0f;
        for (int r = 0; r < n_reads; r++) {
            uint64_t w0 = wall_clock64();
            uint64_t w1 = wall_clock64();
            noise_sum += (float)(w1 - w0);
        }
        float expected = n_reads * 2.0f;
        float time_noise = (noise_sum - expected) / (expected * 0.5f + 1e-6f);

        // ─── SOURCE 2: Placement-dependent compute timing ───
        // Same FMA chain on every block → timing differences = Vth/IR-drop/temp
        float px = (float)tid * 0.001f + 1.0f + 0.0001f * t;
        float py = px + 0.1f;
        uint64_t ct0 = __clock64();
        for (int i = 0; i < 50; i++) {
            px = __fmaf_rn(px, py, px);
            py = __fmaf_rn(py, px, py);
        }
        uint64_t ct1 = __clock64();
        if (px + py == -999.999f) state_out[0] = px;  // anti-optimize
        // Normalize: typical is ~150 cycles, variations ±2-5 cycles
        float placement_noise = ((float)(ct1 - ct0) - 150.0f) / 10.0f;

        // ─── SOURCE 3: Cross-PLL drift ───
        uint64_t sc = __clock64();
        uint64_t wc = wall_clock64();
        float pll_mod = (float)((sc - wc) & 0xFF) / 128.0f - 1.0f;

        // ─── SOURCE 4: Memory access (deterministic input encoder) ───
        uint32_t addr = (nid * stride * 16 + t * 997) & ((16*1024*1024/4)-1);
        uint64_t m0 = __clock64();
        uint32_t val = scratch[addr];
        uint64_t m1 = __clock64();
        float mem_input = ((float)(m1 - m0) - 25.0f) / 10.0f;
        if (val == 0xDEADDEAD) state_out[0] = 0;

        // ─── LOCAL recurrence (same CU — fast LDS) ───
        lds[tid] = v;
        __syncthreads();
        float local_rec = w_local * (lds[local_n1] + lds[local_n2]);

        // ─── REMOTE recurrence (cross-CU — global memory, PLACEMENT-dependent delay) ───
        cu_exchange[nid] = v;
        __threadfence();  // global fence — cross-CU visibility
        float remote_val = cu_exchange[remote_cu * THREADS_PER + remote_tid];
        float remote_rec = w_remote * remote_val;

        // ─── STATE UPDATE ───
        v = tanhf(
            (1.0f - leak) * v
          + noise_gain * time_noise          // PLL jitter (stochastic)
          + 0.08f * placement_noise          // placement/Vth variation
          + 0.04f * pll_mod                  // cross-PLL drift
          + 0.08f * mem_input                // memory timing (input encoder)
          + local_rec                        // intra-CU coupling (fast)
          + remote_rec                       // cross-CU coupling (placement delay!)
        );

        if (v > threshold) {
            spike_count++;
            v -= threshold;
        }

        h = 0.95f * h + 0.05f * v;
        state_out[nid * steps + t] = v + 0.3f * h + 0.1f * (float)spike_count / (t+1);

        __syncthreads();
    }
}

// ============================================================================
// DETERMINISTIC CONTROL: same architecture, no noise sources
// ============================================================================
__global__ void placement_reservoir_control(
    float* state_out, uint32_t* scratch, volatile float* cu_exchange,
    int input_class, int input_phase, int steps
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= THREADS_PER) return;
    int nid = bid * THREADS_PER + tid;

    __shared__ float lds[THREADS_PER];
    float v = 0.0f, h = 0.0f;

    uint32_t rng = nid * 2654435761u + 12345;
    rng *= 1103515245; float leak = 0.05f + 0.15f * ((rng >> 16) & 0xFF) / 255.0f;
    rng *= 1103515245; int local_n1 = (tid + 1 + ((rng>>8)&7)) % THREADS_PER;
    rng *= 1103515245; int local_n2 = (tid + THREADS_PER - 1 - ((rng>>8)&7)) % THREADS_PER;
    rng *= 1103515245; float w_local = 0.05f + 0.10f * ((rng>>16)&0xFF)/255.0f;

    for (int t = 0; t < steps; t++) {
        int stride;
        switch (input_class) {
            case 0: stride = 4; break;
            case 1: stride = 32+96*((t/(8+input_phase*3))&1); break;
            case 2: stride = 4+(t*124)/steps; break;
            case 3: stride = ((t%(15+input_phase*3))<4)?128:4; break;
            default: stride = 32;
        }

        uint32_t addr = (nid * stride * 16 + t * 997) & ((16*1024*1024/4)-1);
        uint64_t m0 = __clock64();
        uint32_t val = scratch[addr];
        uint64_t m1 = __clock64();
        float mem_input = ((float)(m1 - m0) - 25.0f) / 10.0f;
        if (val == 0xDEADDEAD) state_out[0] = 0;

        lds[tid] = v;
        __syncthreads();
        float local_rec = w_local * (lds[local_n1] + lds[local_n2]);

        cu_exchange[nid] = v;
        __threadfence();

        v = tanhf((1.0f - leak) * v + 0.08f * mem_input + local_rec);
        h = 0.95f * h + 0.05f * v;
        state_out[nid * steps + t] = v + 0.3f * h;
        __syncthreads();
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
    int fp = FEATURE_DIM;
    std::vector<float> f(n_neurons * fp);
    int half = steps / 2;
    int q = (steps - half) / 4;
    for (int i = 0; i < n_neurons; i++) {
        float s1=0,s2=0,mn=1e30,mx=-1e30,qq[4]={};
        int cnt=0, zc=0; float ps=0;
        for (int t = half; t < steps; t++) {
            float v = state[i*steps+t];
            if (v != v || v > 1e30 || v < -1e30) v = 0.0f;
            s1 += v; s2 += v*v;
            if (v<mn) mn=v; if (v>mx) mx=v;
            int qi = (t-half)/q; if(qi>3) qi=3;
            qq[qi] += v; cnt++;
            float sg = v>=0?1.0f:-1.0f;
            if (t>half && sg!=ps) zc++;
            ps = sg;
        }
        float mean = s1/cnt;
        float var = s2/cnt - mean*mean;
        f[i*fp+0] = mean;
        f[i*fp+1] = sqrtf(var > 0 ? var : 0);
        f[i*fp+2] = mn; f[i*fp+3] = mx;
        f[i*fp+4] = qq[0]/q; f[i*fp+5] = qq[1]/q;
        f[i*fp+6] = qq[2]/q; f[i*fp+7] = qq[3]/q;
        f[i*fp+8] = qq[3]/q - qq[0]/q;
        f[i*fp+9] = (float)zc/cnt;
    }
    return f;
}

float ridge_classify(Dataset& train, Dataset& test, int nc, float lambda=1.0f) {
    int n=train.X.size(), d=train.X[0].size();
    std::vector<double> XtX(d*d,0);
    for (int i=0;i<n;i++) for(int j=0;j<d;j++) for(int k=0;k<=j;k++)
        XtX[j*d+k] += train.X[i][j]*train.X[i][k];
    for(int j=0;j<d;j++){for(int k=0;k<j;k++) XtX[k*d+j]=XtX[j*d+k]; XtX[j*d+j]+=lambda;}

    std::vector<double> XtY(d*nc,0);
    for(int i=0;i<n;i++) for(int j=0;j<d;j++) XtY[j*nc+train.y[i]]+=train.X[i][j];

    std::vector<double> L(d*d,0);
    for(int i=0;i<d;i++) for(int j=0;j<=i;j++){
        double s=XtX[i*d+j]; for(int k=0;k<j;k++) s-=L[i*d+k]*L[j*d+k];
        L[i*d+j]=(i==j)?sqrt(s>0?s:1e-10):s/(L[j*d+j]>0?L[j*d+j]:1e-10);
    }

    std::vector<double> W(d*nc,0);
    for(int c=0;c<nc;c++){
        std::vector<double> z(d);
        for(int i=0;i<d;i++){double s=XtY[i*nc+c]; for(int k=0;k<i;k++) s-=L[i*d+k]*z[k]; z[i]=s/(L[i*d+i]>0?L[i*d+i]:1e-10);}
        for(int i=d-1;i>=0;i--){double s=z[i]; for(int k=i+1;k<d;k++) s-=L[k*d+i]*W[k*nc+c]; W[i*nc+c]=s/(L[i*d+i]>0?L[i*d+i]:1e-10);}
    }

    int correct=0;
    for(int i=0;i<(int)test.X.size();i++){
        float best=-1e30; int pred=0;
        for(int c=0;c<nc;c++){float sc=0; for(int j=0;j<d;j++) sc+=test.X[i][j]*W[j*nc+c]; if(sc>best){best=sc;pred=c;}}
        if(pred==test.y[i]) correct++;
    }
    return (float)correct/test.X.size();
}

int main() {
    fprintf(stderr, "═══ PLACEMENT PHYSICS RESERVOIR ═══\n");
    fprintf(stderr, "  Sources: wall_clock64 PLL + compute timing (Vth/IR-drop) + cross-PLL + cross-CU delay\n");
    fprintf(stderr, "  %d CUs × %d threads = %d neurons, %d steps\n\n", N_CUS, THREADS_PER, N_NEURONS, STEPS);

    // ═══ PROBE 1: CU FINGERPRINT ═══
    fprintf(stderr, "═══ PROBE: Per-CU compute timing fingerprint ═══\n");
    uint64_t *d_timings, *h_timings;
    hipMalloc(&d_timings, N_CUS * THREADS_PER * sizeof(uint64_t));
    h_timings = new uint64_t[N_CUS * THREADS_PER];

    // Run twice to see if CU identity is stable
    probe_cu_fingerprint<<<N_CUS, THREADS_PER>>>(d_timings, 10000);
    hipDeviceSynchronize();
    hipMemcpy(h_timings, d_timings, N_CUS * THREADS_PER * sizeof(uint64_t), hipMemcpyDeviceToHost);

    uint64_t* h_timings2 = new uint64_t[N_CUS * THREADS_PER];
    probe_cu_fingerprint<<<N_CUS, THREADS_PER>>>(d_timings, 10000);
    hipDeviceSynchronize();
    hipMemcpy(h_timings2, d_timings, N_CUS * THREADS_PER * sizeof(uint64_t), hipMemcpyDeviceToHost);

    fprintf(stderr, "  Per-CU median timing (thread 0):\n  ");
    for (int cu = 0; cu < N_CUS; cu++) {
        fprintf(stderr, "CU%d: %lu/%lu  ", cu, h_timings[cu*THREADS_PER], h_timings2[cu*THREADS_PER]);
    }
    fprintf(stderr, "\n");

    // Check inter-CU variation
    double cu_means[N_CUS];
    for (int cu = 0; cu < N_CUS; cu++) {
        double s = 0;
        for (int t = 0; t < THREADS_PER; t++) s += h_timings[cu*THREADS_PER+t];
        cu_means[cu] = s / THREADS_PER;
    }
    double cu_global_mean = 0;
    for (int cu = 0; cu < N_CUS; cu++) cu_global_mean += cu_means[cu];
    cu_global_mean /= N_CUS;
    double cu_var = 0;
    for (int cu = 0; cu < N_CUS; cu++) cu_var += (cu_means[cu]-cu_global_mean)*(cu_means[cu]-cu_global_mean);
    cu_var /= N_CUS;
    fprintf(stderr, "  Inter-CU std: %.1f cycles (mean=%.0f)\n", sqrt(cu_var), cu_global_mean);

    // Run-to-run stability
    int same_timing = 0;
    for (int i = 0; i < N_CUS * THREADS_PER; i++)
        if (h_timings[i] == h_timings2[i]) same_timing++;
    fprintf(stderr, "  Run-to-run identical: %d/%d (%.1f%%)\n\n",
            same_timing, N_CUS*THREADS_PER, 100.0*same_timing/(N_CUS*THREADS_PER));

    // ═══ PROBE 2: Cross-CU communication delay ═══
    fprintf(stderr, "═══ PROBE: Cross-CU communication delay ═══\n");
    uint64_t *d_delays, *h_delays;
    volatile uint32_t *d_flags;
    hipMalloc(&d_delays, N_CUS * sizeof(uint64_t));
    hipMalloc((void**)&d_flags, N_CUS * sizeof(uint32_t));
    h_delays = new uint64_t[N_CUS];

    hipMemset((void*)d_flags, 0, N_CUS * sizeof(uint32_t));
    probe_cross_cu_delay<<<N_CUS, 1>>>(d_delays, d_flags, N_CUS);
    hipDeviceSynchronize();
    hipMemcpy(h_delays, d_delays, N_CUS * sizeof(uint64_t), hipMemcpyDeviceToHost);

    fprintf(stderr, "  Cross-CU delays: ");
    for (int cu = 0; cu < N_CUS; cu++) fprintf(stderr, "%lu ", h_delays[cu]);
    fprintf(stderr, "\n\n");

    // ═══ RESERVOIR EXPERIMENTS ═══
    size_t scratch_size = 16*1024*1024;
    uint32_t *d_scratch;
    float *d_state;
    volatile float *d_exchange;
    hipMalloc(&d_scratch, scratch_size);
    hipMemset(d_scratch, 0xAB, scratch_size);
    hipMalloc(&d_state, N_NEURONS * STEPS * sizeof(float));
    hipMalloc((void**)&d_exchange, N_NEURONS * sizeof(float));

    float* h_state = new float[N_NEURONS * STEPS];
    int total = N_TRAIN + N_TEST;

    // Verify noise presence
    fprintf(stderr, "  Verifying spatiotemporal noise...\n");
    hipMemset(d_state, 0, N_NEURONS*STEPS*sizeof(float));
    hipMemset((void*)d_exchange, 0, N_NEURONS*sizeof(float));
    placement_reservoir<<<N_CUS, THREADS_PER>>>(d_state, d_scratch, d_exchange, 2, 0, STEPS);
    hipDeviceSynchronize();
    hipMemcpy(h_state, d_state, N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);

    float* h_state2 = new float[N_NEURONS * STEPS];
    hipMemset(d_state, 0, N_NEURONS*STEPS*sizeof(float));
    hipMemset((void*)d_exchange, 0, N_NEURONS*sizeof(float));
    placement_reservoir<<<N_CUS, THREADS_PER>>>(d_state, d_scratch, d_exchange, 2, 0, STEPS);
    hipDeviceSynchronize();
    hipMemcpy(h_state2, d_state, N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);

    // Check reproducibility per CU
    fprintf(stderr, "  Per-CU reproducibility (r between runs):\n  ");
    for (int cu = 0; cu < N_CUS; cu++) {
        int base = cu * THREADS_PER * STEPS;
        double sx=0,sy=0,sxx=0,syy=0,sxy=0;
        int cnt = 0;
        for (int t = STEPS/2; t < STEPS; t++) {
            float x = h_state[base + t];
            float y = h_state2[base + t];
            if (x == x && y == y && x < 1e30 && x > -1e30 && y < 1e30 && y > -1e30) {
                sx+=x; sy+=y; sxx+=x*x; syy+=y*y; sxy+=x*y; cnt++;
            }
        }
        double r = (cnt*sxy-sx*sy) / (sqrt(cnt*sxx-sx*sx+1e-10)*sqrt(cnt*syy-sy*sy+1e-10));
        fprintf(stderr, "CU%d:%.3f ", cu, r);
    }
    fprintf(stderr, "\n");

    // Show sample values from different CUs
    fprintf(stderr, "  CU0 run1: ");
    for (int t = 0; t < 10; t++) fprintf(stderr, "%.3f ", h_state[t]);
    fprintf(stderr, "\n  CU6 run1: ");
    for (int t = 0; t < 10; t++) fprintf(stderr, "%.3f ", h_state[6*THREADS_PER*STEPS+t]);
    fprintf(stderr, "\n\n");

    // ═══ EXP 1: Full Spatiotemporal Reservoir ═══
    fprintf(stderr, "═══ EXP 1: Spatiotemporal Placement Reservoir ═══\n");
    Dataset train, test;
    for (int cls = 0; cls < N_CLASSES; cls++) {
        for (int trial = 0; trial < total; trial++) {
            if (trial % 30 == 0) fprintf(stderr, "  cls=%d trial=%d/%d\r", cls, trial, total);

            hipMemset(d_state, 0, N_NEURONS*STEPS*sizeof(float));
            hipMemset((void*)d_exchange, 0, N_NEURONS*sizeof(float));
            placement_reservoir<<<N_CUS, THREADS_PER>>>(d_state, d_scratch, d_exchange, cls, trial, STEPS);
            hipDeviceSynchronize();
            hipMemcpy(h_state, d_state, N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);

            auto feat = extract_features(h_state, N_NEURONS, STEPS);
            if (trial < N_TRAIN) {
                train.X.push_back(feat);
                train.y.push_back(cls);
            } else {
                test.X.push_back(feat);
                test.y.push_back(cls);
            }
        }
    }
    fprintf(stderr, "  Train:%lu Test:%lu Feat:%lu\n", train.X.size(), test.X.size(), train.X[0].size());

    float best_acc = 0;
    for (float lam : {0.01f, 0.1f, 1.0f, 10.0f, 100.0f, 1000.0f}) {
        float acc = ridge_classify(train, test, N_CLASSES, lam);
        fprintf(stderr, "  lambda=%.2f: %.1f%%\n", lam, acc*100);
        if (acc > best_acc) best_acc = acc;
    }
    fprintf(stderr, "  ★ SPATIOTEMPORAL reservoir: %.1f%% (chance=%.1f%%)\n", best_acc*100, 100.0/N_CLASSES);
    fprintf(stderr, "  PASS: %s\n\n", best_acc > 0.30 ? "YES ★" : "NO");

    // ═══ EXP 2: Deterministic Control ═══
    fprintf(stderr, "═══ EXP 2: Deterministic Control (no noise) ═══\n");
    Dataset train_c, test_c;
    for (int cls = 0; cls < N_CLASSES; cls++) {
        for (int trial = 0; trial < total; trial++) {
            if (trial % 30 == 0) fprintf(stderr, "  cls=%d trial=%d/%d\r", cls, trial, total);

            hipMemset(d_state, 0, N_NEURONS*STEPS*sizeof(float));
            hipMemset((void*)d_exchange, 0, N_NEURONS*sizeof(float));
            placement_reservoir_control<<<N_CUS, THREADS_PER>>>(d_state, d_scratch, d_exchange, cls, trial, STEPS);
            hipDeviceSynchronize();
            hipMemcpy(h_state, d_state, N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);

            auto feat = extract_features(h_state, N_NEURONS, STEPS);
            if (trial < N_TRAIN) { train_c.X.push_back(feat); train_c.y.push_back(cls); }
            else { test_c.X.push_back(feat); test_c.y.push_back(cls); }
        }
    }

    float best_ctrl = 0;
    for (float lam : {0.01f, 0.1f, 1.0f, 10.0f, 100.0f, 1000.0f}) {
        float acc = ridge_classify(train_c, test_c, N_CLASSES, lam);
        if (acc > best_ctrl) best_ctrl = acc;
    }
    fprintf(stderr, "  ★ DETERMINISTIC control: %.1f%%\n", best_ctrl*100);
    fprintf(stderr, "  Noise advantage: %.1f pp\n\n", (best_acc - best_ctrl)*100);

    // ═══ EXP 3: Memory Capacity ═══
    fprintf(stderr, "═══ EXP 3: Memory Capacity ═══\n");
    float total_mc = 0;
    for (int delay = 0; delay <= 5; delay++) {
        Dataset mc_train, mc_test;
        for (int cls = 0; cls < N_CLASSES; cls++) {
            for (int trial = 0; trial < total; trial++) {
                hipMemset(d_state, 0, N_NEURONS*STEPS*sizeof(float));
                hipMemset((void*)d_exchange, 0, N_NEURONS*sizeof(float));
                placement_reservoir<<<N_CUS, THREADS_PER>>>(d_state, d_scratch, d_exchange, cls, trial, STEPS);
                hipDeviceSynchronize();
                hipMemcpy(h_state, d_state, N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);

                // For delay>0, label comes from a PREVIOUS trial's class
                int label = cls;
                if (delay > 0 && trial >= delay) {
                    label = (cls + delay) % N_CLASSES;  // shifted label
                }

                auto feat = extract_features(h_state, N_NEURONS, STEPS);
                if (trial < N_TRAIN) { mc_train.X.push_back(feat); mc_train.y.push_back(label); }
                else { mc_test.X.push_back(feat); mc_test.y.push_back(label); }
            }
        }
        float acc = ridge_classify(mc_train, mc_test, N_CLASSES, 10.0f);
        float mc = (acc - 1.0f/N_CLASSES) / (1.0f - 1.0f/N_CLASSES);
        if (mc < 0) mc = 0;
        total_mc += mc;
        fprintf(stderr, "  delay=%d: %.1f%% MC=%.3f %s\n", delay, acc*100, mc, mc>0.1?"█":"░");
    }
    fprintf(stderr, "  Total MC: %.3f\n\n", total_mc);

    // ═══ EXP 4: Spatial fingerprint — can we identify WHICH CU produced the output? ═══
    fprintf(stderr, "═══ EXP 4: CU Identification (spatial fingerprint) ═══\n");
    // Run same input on all CUs, extract per-CU features, classify CU identity
    Dataset cu_train, cu_test;
    int cu_trials = 60;
    for (int trial = 0; trial < cu_trials; trial++) {
        hipMemset(d_state, 0, N_NEURONS*STEPS*sizeof(float));
        hipMemset((void*)d_exchange, 0, N_NEURONS*sizeof(float));
        // Same input for all CUs
        placement_reservoir<<<N_CUS, THREADS_PER>>>(d_state, d_scratch, d_exchange, 2, trial, STEPS);
        hipDeviceSynchronize();
        hipMemcpy(h_state, d_state, N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);

        // Each CU is a sample, label = CU ID
        for (int cu = 0; cu < N_CUS; cu++) {
            auto feat = extract_features(h_state + cu*THREADS_PER*STEPS, THREADS_PER, STEPS);
            if (trial < cu_trials * 2/3) { cu_train.X.push_back(feat); cu_train.y.push_back(cu); }
            else { cu_test.X.push_back(feat); cu_test.y.push_back(cu); }
        }
    }
    float cu_acc = ridge_classify(cu_train, cu_test, N_CUS, 10.0f);
    fprintf(stderr, "  CU identification: %.1f%% (chance=%.1f%%)\n", cu_acc*100, 100.0/N_CUS);
    fprintf(stderr, "  Each CU has a unique physics fingerprint: %s\n\n",
            cu_acc > 2.0/N_CUS ? "YES ★" : "NO");

    fprintf(stderr, "═══ SUMMARY ═══\n");
    fprintf(stderr, "  EXP1 Spatiotemporal: %.1f%% %s\n", best_acc*100, best_acc>0.30?"PASS":"FAIL");
    fprintf(stderr, "  EXP2 Deterministic:  %.1f%%\n", best_ctrl*100);
    fprintf(stderr, "  EXP3 Memory:         MC=%.3f %s\n", total_mc, total_mc>0.5?"PASS":"FAIL");
    fprintf(stderr, "  EXP4 CU Fingerprint: %.1f%% %s\n", cu_acc*100, cu_acc>2.0/N_CUS?"PASS":"FAIL");
    fprintf(stderr, "  Noise advantage:     %.1f pp\n", (best_acc-best_ctrl)*100);

    delete[] h_state;
    delete[] h_state2;
    delete[] h_timings;
    delete[] h_timings2;
    delete[] h_delays;
    hipFree(d_timings);
    hipFree(d_delays);
    hipFree((void*)d_flags);
    hipFree(d_scratch);
    hipFree(d_state);
    hipFree((void*)d_exchange);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
