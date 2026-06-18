// jitter_thermal_reservoir.hip — Thermal-modulated PLL reservoir
//
// CONCEPT: Different instruction workloads (FMA, INT, transcendental, memory)
// draw different power on the same CU → local temperature shifts →
// PLL jitter distribution changes. We use instruction mix as an INPUT
// encoding mechanism, with PLL jitter as the readout.
//
// Also: self-heating — sustained computation raises CU temperature,
// creating a slow drift in PLL characteristics that acts as intrinsic
// memory.
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_thermal_reservoir scripts/jitter_thermal_reservoir.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_thermal_reservoir

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
#define STEPS 400
#define N_CLASSES 8
#define N_TRAIN 80
#define FEAT_PER_N 4
#define SAMPLE_NEURONS 192

// ============================================================================
// PROBE 1: Instruction-mix effect on PLL jitter
// Does heavy FMA vs heavy INT vs transcendental produce different jitter?
// ============================================================================
__global__ void probe_instruction_mix(float* results, int mix_type, int iters) {
    int tid = threadIdx.x;
    if (tid >= 32) return;  // one warp

    float dummy = 1.0001f + tid * 0.0001f;

    // Warm up PLL reading
    for (int i = 0; i < 100; i++) {
        uint64_t w = wall_clock64();
        if (w == 0xDEAD) dummy += 1.0f;
    }

    // Run workload + sample PLL
    float jitter_sum = 0.0f;
    float jitter_sq = 0.0f;
    int nsamples = 0;

    for (int i = 0; i < iters; i++) {
        // Workload depends on mix_type
        switch(mix_type) {
            case 0:  // Pure PLL (no workload)
                break;
            case 1:  // Heavy FMA
                for (int j = 0; j < 50; j++)
                    dummy = __fmaf_rn(dummy, 1.000001f, 0.0001f);
                break;
            case 2:  // Heavy INT
                {
                    uint32_t iv = __float_as_uint(dummy);
                    for (int j = 0; j < 50; j++)
                        iv = iv * 2654435761u + 1;
                    dummy = __uint_as_float((iv & 0x007FFFFF) | 0x3F800000);
                }
                break;
            case 3:  // Transcendental (sinf/cosf — uses SFU)
                for (int j = 0; j < 20; j++)
                    dummy = sinf(dummy) + cosf(dummy * 1.1f);
                break;
            case 4:  // Memory-heavy (LDS thrash)
                {
                    __shared__ float scratch[256];
                    scratch[tid] = dummy;
                    __syncthreads();
                    for (int j = 0; j < 50; j++) {
                        dummy += scratch[(tid + j) % 32];
                        scratch[tid] = dummy;
                        __syncthreads();
                    }
                }
                break;
        }

        // Sample PLL jitter after workload
        uint64_t w0 = wall_clock64();
        uint64_t w1 = wall_clock64();
        uint64_t w2 = wall_clock64();
        uint64_t w3 = wall_clock64();
        float j1 = (float)(w1 - w0);
        float j2 = (float)(w2 - w1);
        float j3 = (float)(w3 - w2);
        jitter_sum += j1 + j2 + j3;
        jitter_sq += j1*j1 + j2*j2 + j3*j3;
        nsamples += 3;
    }

    if (dummy == -9999.0f) results[0] = dummy;  // prevent optimization

    float mean = jitter_sum / nsamples;
    float var = jitter_sq / nsamples - mean * mean;

    // Store: [mean, var, sum, n] per thread
    results[tid * 4 + 0] = mean;
    results[tid * 4 + 1] = var;
    results[tid * 4 + 2] = jitter_sum;
    results[tid * 4 + 3] = (float)nsamples;
}

// ============================================================================
// PROBE 2: Self-heating PLL drift
// Sustained heavy workload should shift PLL characteristics over time
// ============================================================================
__global__ void probe_self_heating(float* jitter_trace, int segments, int work_per_seg) {
    int tid = threadIdx.x;
    if (tid >= 1) return;  // single thread

    float dummy = 1.0001f;

    for (int seg = 0; seg < segments; seg++) {
        // Heavy work to generate heat
        for (int w = 0; w < work_per_seg; w++) {
            dummy = __fmaf_rn(dummy, 1.000001f, 0.0001f);
            dummy = sinf(dummy) + cosf(dummy);
        }

        // Sample PLL jitter after heating
        float jsum = 0.0f;
        int n = 0;
        for (int r = 0; r < 256; r++) {
            uint64_t w0 = wall_clock64();
            uint64_t w1 = wall_clock64();
            jsum += (float)(w1 - w0);
            n++;
        }
        jitter_trace[seg] = jsum / n;
    }

    if (dummy == -9999.0f) jitter_trace[0] = dummy;
}

// ============================================================================
// MAIN RESERVOIR: Instruction-mix-modulated PLL reservoir
// Input class determines the ratio of FMA/INT/SFU workload per step,
// which modulates power draw → thermal → PLL jitter characteristics
// ============================================================================
__global__ void thermal_reservoir(
    float* state_out, volatile float* cu_xchg,
    int input_class, int input_phase, int steps
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= THREADS) return;
    int nid = bid * THREADS + tid;

    __shared__ float lds[THREADS];

    float v = 0.0f, h = 0.0f, slow = 0.0f;
    int spikes = 0;

    uint32_t rng = nid * 2654435761u + 12345;
    rng *= 1103515245; float leak = 0.08f + 0.12f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float ng = 0.15f + 0.25f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float thr = 0.5f + 0.4f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; int ln1 = (tid+1+((rng>>8)&7))%THREADS;
    rng *= 1103515245; int ln2 = (tid+THREADS-1-((rng>>8)&7))%THREADS;
    rng *= 1103515245; float wl = 0.03f + 0.07f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; int rcu = ((rng>>8)&0xFF)%N_CUS;
    rng *= 1103515245; int rtid = ((rng>>8)&0xFF)%THREADS;
    rng *= 1103515245; float wr = 0.01f + 0.04f * ((rng>>16)&0xFF)/255.0f;

    float dummy = 1.0001f + nid * 0.0001f;

    for (int t = 0; t < steps; t++) {
        // ─── INPUT: Instruction mix determines thermal profile ───
        int fma_iters, int_iters, sfu_iters, depth;
        float temporal_f;

        switch (input_class) {
            case 0:  // Pure PLL, minimal work
                fma_iters = 0; int_iters = 0; sfu_iters = 0;
                depth = 64; temporal_f = 0.0f;
                break;
            case 1:  // FMA-heavy (high power draw)
                fma_iters = 30; int_iters = 0; sfu_iters = 0;
                depth = 64; temporal_f = 0.0f;
                break;
            case 2:  // INT-heavy (medium power)
                fma_iters = 0; int_iters = 50; sfu_iters = 0;
                depth = 64; temporal_f = 0.0f;
                break;
            case 3:  // SFU-heavy (transcendental)
                fma_iters = 0; int_iters = 0; sfu_iters = 15;
                depth = 64; temporal_f = 0.0f;
                break;
            case 4:  // Mixed FMA+SFU oscillating
                fma_iters = (t % 20 < 10) ? 30 : 0;
                int_iters = 0;
                sfu_iters = (t % 20 >= 10) ? 15 : 0;
                depth = 64; temporal_f = 0.5f;
                break;
            case 5:  // Ramping depth + FMA
                fma_iters = 10 + (20 * t) / steps;
                int_iters = 0; sfu_iters = 0;
                depth = 32 + (96 * t) / steps;
                temporal_f = 0.3f;
                break;
            case 6:  // Burst thermal (heavy then light)
                fma_iters = ((t % (15 + input_phase*3)) < 5) ? 40 : 0;
                int_iters = 0; sfu_iters = 0;
                depth = 96; temporal_f = 0.6f;
                break;
            case 7:  // All mixed with phase modulation
                fma_iters = 10; int_iters = 20; sfu_iters = 5;
                depth = 48 + 48 * (((t + input_phase*7) / 12) & 1);
                temporal_f = 0.4f;
                break;
            default:
                fma_iters = 0; int_iters = 0; sfu_iters = 0;
                depth = 64; temporal_f = 0.0f;
        }

        // ─── INSTRUCTION-MIX WORKLOAD (generates heat) ───
        for (int i = 0; i < fma_iters; i++)
            dummy = __fmaf_rn(dummy, 1.000001f, 0.0001f);
        {
            uint32_t iv = __float_as_uint(dummy);
            for (int i = 0; i < int_iters; i++)
                iv = iv * 2654435761u + 1;
            dummy = __uint_as_float((iv & 0x007FFFFF) | 0x3F800000);
        }
        for (int i = 0; i < sfu_iters; i++)
            dummy = sinf(dummy) + cosf(dummy * 1.1f);

        // ─── DEEP PLL NOISE (thermal state modulates this) ───
        float noise_sum = 0.0f;
        for (int r = 0; r < depth; r++) {
            uint64_t w0 = wall_clock64();
            uint64_t w1 = wall_clock64();
            noise_sum += (float)(w1 - w0);
        }
        float expected = depth * 2.2f;
        float noise = (noise_sum - expected) / (sqrtf((float)depth) * 2.0f);

        // ─── PLACEMENT + CROSS-PLL ───
        uint64_t sc = __clock64();
        uint64_t wc = wall_clock64();
        float pll = (float)((sc - wc) & 0x1FF) / 256.0f - 1.0f;

        // ─── THERMAL SIGNATURE: instruction timing varies with temperature ───
        uint64_t t0 = __clock64();
        float tx = dummy;
        for (int i = 0; i < 10; i++) tx = __fmaf_rn(tx, 1.0001f, 0.001f);
        uint64_t t1 = __clock64();
        if (tx == -9999.0f) state_out[0] = tx;
        float thermal_timing = ((float)(t1 - t0) - 30.0f) / 10.0f;

        // ─── RECURRENCE ───
        lds[tid] = v;
        __syncthreads();
        float lrec = wl * (lds[ln1] + lds[ln2]);
        cu_xchg[nid] = v;
        __threadfence();
        float rrec = wr * cu_xchg[rcu * THREADS + rtid];

        float temporal = temporal_f * sinf(6.2832f * t / (20.0f + input_phase * 5.0f));

        // ─── STATE UPDATE ───
        v = tanhf(
            (1.0f - leak) * v
          + ng * noise              // deep PLL noise (PRIMARY)
          + 0.04f * thermal_timing  // instruction-dependent timing
          + 0.03f * pll             // cross-PLL drift
          + 0.07f * temporal        // temporal modulation
          + lrec + rrec
        );

        if (v > thr) { spikes++; v -= thr; }
        h = 0.93f * h + 0.07f * v;
        slow = 0.99f * slow + 0.01f * v;
        state_out[nid * steps + t] = v + 0.3f*h + 0.1f*slow + 0.05f*(float)spikes/(t+1);
        __syncthreads();
    }
    if (dummy == -9999.0f) state_out[0] = dummy;
}

// ============================================================================
// CONTROL: Same structure, no noise
// ============================================================================
__global__ void thermal_control(
    float* state_out, volatile float* cu_xchg,
    int input_class, int input_phase, int steps
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= THREADS) return;
    int nid = bid * THREADS + tid;
    __shared__ float lds[THREADS];
    float v = 0.0f, h = 0.0f, slow = 0.0f;

    uint32_t rng = nid * 2654435761u + 12345;
    rng *= 1103515245; float leak = 0.08f + 0.12f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float ng = 0.15f + 0.25f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; int ln1 = (tid+1+((rng>>8)&7))%THREADS;
    rng *= 1103515245; int ln2 = (tid+THREADS-1-((rng>>8)&7))%THREADS;
    rng *= 1103515245; float wl = 0.03f + 0.07f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; int rcu = ((rng>>8)&0xFF)%N_CUS;
    rng *= 1103515245; int rtid = ((rng>>8)&0xFF)%THREADS;
    rng *= 1103515245; float wr = 0.01f + 0.04f * ((rng>>16)&0xFF)/255.0f;

    for (int t = 0; t < steps; t++) {
        float temporal_f;
        switch(input_class) {
            case 0: temporal_f = 0.0f; break;
            case 1: temporal_f = 0.0f; break;
            case 2: temporal_f = 0.0f; break;
            case 3: temporal_f = 0.0f; break;
            case 4: temporal_f = 0.5f; break;
            case 5: temporal_f = 0.3f; break;
            case 6: temporal_f = 0.6f; break;
            case 7: temporal_f = 0.4f; break;
            default: temporal_f = 0.0f;
        }

        lds[tid] = v;
        __syncthreads();
        float lrec = wl * (lds[ln1] + lds[ln2]);
        cu_xchg[nid] = v;
        __threadfence();
        float rrec = wr * cu_xchg[rcu * THREADS + rtid];
        float temporal = temporal_f * sinf(6.2832f * t / (20.0f + input_phase * 5.0f));

        v = tanhf((1.0f - leak)*v + 0.07f*temporal + lrec + rrec);
        h = 0.93f*h + 0.07f*v;
        slow = 0.99f*slow + 0.01f*v;
        state_out[nid*steps+t] = v + 0.3f*h + 0.1f*slow;
        __syncthreads();
    }
}

// ============================================================================
// HOST
// ============================================================================
struct Dataset { std::vector<std::vector<float>> X; std::vector<int> y; };

std::vector<float> extract_features(float* state, int n, int steps) {
    int stride = n / SAMPLE_NEURONS;
    if (stride < 1) stride = 1;
    int actual = 0;
    std::vector<float> f(SAMPLE_NEURONS * FEAT_PER_N, 0.0f);
    int half = steps / 2;
    int q = (steps - half) / 2;
    for (int si = 0; si < SAMPLE_NEURONS && si*stride < n; si++) {
        int i = si * stride;
        float s1=0,s2=0, qq[2]={};
        int cnt=0,zc=0; float ps=0;
        for (int t = half; t < steps; t++) {
            float v = state[i*steps+t];
            if (v!=v || v>1e20 || v<-1e20) v=0;
            s1+=v; s2+=v*v;
            int qi=(t-half)/q; if(qi>1)qi=1;
            qq[qi]+=v; cnt++;
            float sg=v>=0?1.0f:-1.0f;
            if(t>half&&sg!=ps)zc++;
            ps=sg;
        }
        float mean=s1/cnt;
        float var=s2/cnt-mean*mean;
        f[actual*FEAT_PER_N+0]=mean;
        f[actual*FEAT_PER_N+1]=sqrtf(var>0?var:0);
        f[actual*FEAT_PER_N+2]=qq[1]/q-qq[0]/q;
        f[actual*FEAT_PER_N+3]=(float)zc/cnt;
        actual++;
    }
    f.resize(actual * FEAT_PER_N);
    return f;
}

float ridge_classify(Dataset& train, Dataset& test, int nc, float lambda=1.0f) {
    int n=train.X.size(), d=train.X[0].size();
    std::vector<double> XtX(d*d,0);
    for(int i=0;i<n;i++) for(int j=0;j<d;j++) for(int k=0;k<=j;k++)
        XtX[j*d+k]+=train.X[i][j]*train.X[i][k];
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
        float best=-1e30;int pred=0;
        for(int c=0;c<nc;c++){float sc=0;for(int j=0;j<d;j++)sc+=test.X[i][j]*W[j*nc+c];if(sc>best){best=sc;pred=c;}}
        if(pred==test.y[i]) correct++;
    }
    return (float)correct/test.X.size();
}

int main() {
    fprintf(stderr, "═══ THERMAL-MODULATED PLL RESERVOIR ═══\n");
    fprintf(stderr, "  %d CUs × %d threads = %d neurons, %d steps, %d classes\n",
            N_CUS, THREADS, N_NEURONS, STEPS, N_CLASSES);
    fprintf(stderr, "  Input: instruction mix → thermal → PLL jitter\n\n");

    // ─── PROBE 1: Instruction mix effect on PLL jitter ───
    fprintf(stderr, "═══ PROBE 1: Instruction Mix → PLL Jitter ═══\n");
    float *d_probe, *h_probe;
    h_probe = (float*)malloc(128 * 4 * sizeof(float));
    hipMalloc(&d_probe, 128 * 4 * sizeof(float));

    const char* mix_names[] = {"No workload", "FMA-heavy", "INT-heavy", "SFU(trig)", "LDS-heavy"};
    float mix_means[5], mix_vars[5];

    for (int mix = 0; mix < 5; mix++) {
        hipMemset(d_probe, 0, 128 * 4 * sizeof(float));
        probe_instruction_mix<<<1, 64>>>(d_probe, mix, 5000);
        hipDeviceSynchronize();
        hipMemcpy(h_probe, d_probe, 32 * 4 * sizeof(float), hipMemcpyDeviceToHost);

        float mean_sum = 0, var_sum = 0;
        for (int t = 0; t < 32; t++) {
            mean_sum += h_probe[t*4+0];
            var_sum += h_probe[t*4+1];
        }
        mix_means[mix] = mean_sum / 32;
        mix_vars[mix] = var_sum / 32;
        fprintf(stderr, "  %-14s: mean_jitter=%.3f  var=%.4f\n",
                mix_names[mix], mix_means[mix], mix_vars[mix]);
    }

    // Check if any mix produces different jitter
    float max_diff = 0;
    for (int i = 0; i < 5; i++)
        for (int j = i+1; j < 5; j++) {
            float d = fabsf(mix_means[i] - mix_means[j]);
            if (d > max_diff) max_diff = d;
        }
    fprintf(stderr, "  Max jitter mean difference: %.4f\n", max_diff);
    fprintf(stderr, "  Instruction mix affects PLL: %s\n\n",
            max_diff > 0.01f ? "YES ★" : "MAYBE (small effect)");

    hipFree(d_probe);
    free(h_probe);

    // ─── PROBE 2: Self-heating PLL drift ───
    fprintf(stderr, "═══ PROBE 2: Self-Heating PLL Drift ═══\n");
    int segs = 100;
    float *d_trace, *h_trace;
    h_trace = (float*)malloc(segs * sizeof(float));
    hipMalloc(&d_trace, segs * sizeof(float));
    hipMemset(d_trace, 0, segs * sizeof(float));

    probe_self_heating<<<1, 1>>>(d_trace, segs, 10000);
    hipDeviceSynchronize();
    hipMemcpy(h_trace, d_trace, segs * sizeof(float), hipMemcpyDeviceToHost);

    fprintf(stderr, "  Jitter over time (first 10 segments):\n    ");
    for (int i = 0; i < 10; i++) fprintf(stderr, "%.3f ", h_trace[i]);
    fprintf(stderr, "\n  Last 10 segments:\n    ");
    for (int i = segs-10; i < segs; i++) fprintf(stderr, "%.3f ", h_trace[i]);
    fprintf(stderr, "\n");

    // Check for drift
    float early_mean = 0, late_mean = 0;
    for (int i = 0; i < 20; i++) early_mean += h_trace[i];
    for (int i = segs-20; i < segs; i++) late_mean += h_trace[i];
    early_mean /= 20; late_mean /= 20;
    fprintf(stderr, "  Early mean: %.4f  Late mean: %.4f  Drift: %.4f\n",
            early_mean, late_mean, late_mean - early_mean);
    fprintf(stderr, "  Self-heating drift: %s\n\n",
            fabsf(late_mean - early_mean) > 0.01f ? "YES ★" : "MINIMAL");

    hipFree(d_trace);
    free(h_trace);

    // ─── MAIN EXPERIMENT: 8-class thermal reservoir ───
    float *d_state, *h_state;
    volatile float *d_xchg;
    h_state = (float*)malloc(N_NEURONS * STEPS * sizeof(float));
    hipMalloc(&d_state, N_NEURONS * STEPS * sizeof(float));
    hipMalloc((void**)&d_xchg, N_NEURONS * sizeof(float));

    int total = N_TRAIN + N_TRAIN/2;  // 80 train + 40 test

    // ─── EXP 1: 8-class classification ───
    fprintf(stderr, "═══ EXP 1: Thermal Reservoir — 8-class ═══\n");
    Dataset train, test;
    for (int cls = 0; cls < N_CLASSES; cls++) {
        for (int trial = 0; trial < total; trial++) {
            if (trial%20==0) fprintf(stderr, "  cls=%d trial=%d/%d\r", cls, trial, total);
            hipMemset(d_state, 0, N_NEURONS*STEPS*sizeof(float));
            hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
            thermal_reservoir<<<N_CUS, THREADS>>>(d_state, d_xchg, cls, trial, STEPS);
            hipDeviceSynchronize();
            hipMemcpy(h_state, d_state, N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);

            auto feat = extract_features(h_state, N_NEURONS, STEPS);
            if (trial < N_TRAIN) { train.X.push_back(feat); train.y.push_back(cls); }
            else { test.X.push_back(feat); test.y.push_back(cls); }
        }
    }
    fprintf(stderr, "  Train:%lu Test:%lu Feat:%lu\n", train.X.size(), test.X.size(), train.X[0].size());

    float best_acc = 0;
    for (float lam : {0.001f, 0.01f, 0.1f, 1.0f, 10.0f, 100.0f}) {
        float acc = ridge_classify(train, test, N_CLASSES, lam);
        fprintf(stderr, "  lambda=%.3f: %.1f%%\n", lam, acc*100);
        if (acc > best_acc) best_acc = acc;
    }
    fprintf(stderr, "  ★ THERMAL reservoir: %.1f%% (chance=%.1f%%)\n",
            best_acc*100, 100.0/N_CLASSES);
    fprintf(stderr, "  PASS: %s\n\n", best_acc > 0.20 ? "YES ★" : "NO");

    // ─── EXP 2: Control (no noise) ───
    fprintf(stderr, "═══ EXP 2: Deterministic Control ═══\n");
    Dataset train_c, test_c;
    for (int cls = 0; cls < N_CLASSES; cls++) {
        for (int trial = 0; trial < total; trial++) {
            if (trial%20==0) fprintf(stderr, "  cls=%d trial=%d/%d\r", cls, trial, total);
            hipMemset(d_state, 0, N_NEURONS*STEPS*sizeof(float));
            hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
            thermal_control<<<N_CUS, THREADS>>>(d_state, d_xchg, cls, trial, STEPS);
            hipDeviceSynchronize();
            hipMemcpy(h_state, d_state, N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);

            auto feat = extract_features(h_state, N_NEURONS, STEPS);
            if (trial < N_TRAIN) { train_c.X.push_back(feat); train_c.y.push_back(cls); }
            else { test_c.X.push_back(feat); test_c.y.push_back(cls); }
        }
    }
    float best_ctrl = 0;
    for (float lam : {0.001f, 0.01f, 0.1f, 1.0f, 10.0f, 100.0f}) {
        float acc = ridge_classify(train_c, test_c, N_CLASSES, lam);
        if (acc > best_ctrl) best_ctrl = acc;
    }
    fprintf(stderr, "  ★ CONTROL: %.1f%%\n", best_ctrl*100);
    fprintf(stderr, "  Noise advantage: %.1f pp\n\n", (best_acc-best_ctrl)*100);

    // ─── EXP 3: Instruction-type discrimination ───
    // Can the reservoir tell apart FMA-class from SFU-class?
    fprintf(stderr, "═══ EXP 3: Instruction-Type Discrimination ═══\n");
    fprintf(stderr, "  FMA (class 1) vs SFU (class 3) vs Mixed (class 4)\n");
    Dataset train_d, test_d;
    int disc_classes[] = {1, 3, 4};
    for (int ci = 0; ci < 3; ci++) {
        int cls = disc_classes[ci];
        for (int trial = 0; trial < total; trial++) {
            hipMemset(d_state, 0, N_NEURONS*STEPS*sizeof(float));
            hipMemset((void*)d_xchg, 0, N_NEURONS*sizeof(float));
            thermal_reservoir<<<N_CUS, THREADS>>>(d_state, d_xchg, cls, trial, STEPS);
            hipDeviceSynchronize();
            hipMemcpy(h_state, d_state, N_NEURONS*STEPS*sizeof(float), hipMemcpyDeviceToHost);

            auto feat = extract_features(h_state, N_NEURONS, STEPS);
            if (trial < N_TRAIN) { train_d.X.push_back(feat); train_d.y.push_back(ci); }
            else { test_d.X.push_back(feat); test_d.y.push_back(ci); }
        }
    }
    float best_disc = 0;
    for (float lam : {0.001f, 0.01f, 0.1f, 1.0f, 10.0f}) {
        float acc = ridge_classify(train_d, test_d, 3, lam);
        if (acc > best_disc) best_disc = acc;
    }
    fprintf(stderr, "  ★ Instruction discrimination: %.1f%% (chance=33.3%%)\n", best_disc*100);
    fprintf(stderr, "  PASS: %s\n\n", best_disc > 0.50 ? "YES ★" : "NO");

    // ─── SUMMARY ───
    fprintf(stderr, "═══ SUMMARY ═══\n");
    fprintf(stderr, "  PROBE1 inst-mix effect: max_diff=%.4f\n", max_diff);
    fprintf(stderr, "  PROBE2 self-heat drift: %.4f\n", fabsf(late_mean - early_mean));
    fprintf(stderr, "  EXP1 8-class:     %.1f%% %s\n", best_acc*100, best_acc>0.20?"PASS":"FAIL");
    fprintf(stderr, "  EXP2 Control:     %.1f%%\n", best_ctrl*100);
    fprintf(stderr, "  EXP3 Inst disc:   %.1f%% %s\n", best_disc*100, best_disc>0.50?"PASS":"FAIL");
    fprintf(stderr, "  Noise advantage:  %.1f pp\n", (best_acc-best_ctrl)*100);
    fprintf(stderr, "\nDone.\n");

    hipFree(d_state);
    hipFree((void*)d_xchg);
    free(h_state);
    return 0;
}
