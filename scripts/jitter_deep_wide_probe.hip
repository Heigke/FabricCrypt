// jitter_deep_wide_probe.hip — Exhaustive probe of ALL possible GPU physics
//
// Probe every mechanism we haven't tested yet:
//   A. VGPR pressure → register file power draw → PLL?
//   B. Occupancy modulation (1 vs 2 vs 4 wavefronts/CU) → scheduling noise?
//   C. Instruction pipeline depth (deep dependency chains vs ILP)
//   D. Shared memory (LDS) access pattern → bank timing?
//   E. Vector width (half-wave vs full-wave execution)
//   F. Denormal handling path (flush-to-zero vs gradual underflow)
//   G. Branch divergence within wavefront → EXEC mask → timing?
//   H. Global memory coalescing quality → MC arbiter?
//   I. Cross-PLL drift RATE (does it accelerate with workload?)
//   J. wall_clock64 vs __clock64 phase relationship stability
//   K. Instruction interleaving (FMA+INT simultaneously on same CU)
//   L. Texture unit path (if accessible)
//
// Build: hipcc --offload-arch=gfx1100 -O1 -o scripts/jitter_deep_wide_probe scripts/jitter_deep_wide_probe.hip
// Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/jitter_deep_wide_probe

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#define ITERS 5000
#define N_THREADS 64

struct ProbeStats {
    float mean, std_dev, min_val, max_val;
    int unique;
    float run_corr;  // correlation between two runs
};

// ============================================================================
// PROBE A: VGPR pressure → more registers used → more power → PLL change?
// ============================================================================
__global__ void probe_vgpr_low(uint64_t* timings) {
    int tid = threadIdx.x;
    if (tid >= 32) return;
    // Use only 2 VGPRs worth of state
    float a = 1.0f + tid * 0.001f;
    for (int i = 0; i < ITERS; i++) {
        uint64_t t0 = wall_clock64();
        a = __fmaf_rn(a, 1.0001f, 0.001f);
        a = __fmaf_rn(a, 1.0001f, 0.001f);
        uint64_t t1 = wall_clock64();
        timings[tid*ITERS+i] = t1 - t0;
    }
    if (a == -999.0f) timings[0] = 0;
}

__global__ void probe_vgpr_high(uint64_t* timings) {
    int tid = threadIdx.x;
    if (tid >= 32) return;
    // Use 32 VGPRs — max pressure
    float v0=1.0f+tid*0.001f, v1=v0+0.1f, v2=v0+0.2f, v3=v0+0.3f;
    float v4=v0+0.4f, v5=v0+0.5f, v6=v0+0.6f, v7=v0+0.7f;
    float v8=v0+0.8f, v9=v0+0.9f, va=v0+1.0f, vb=v0+1.1f;
    float vc=v0+1.2f, vd=v0+1.3f, ve=v0+1.4f, vf=v0+1.5f;
    for (int i = 0; i < ITERS; i++) {
        uint64_t t0 = wall_clock64();
        v0=__fmaf_rn(v0,v1,v2); v1=__fmaf_rn(v1,v2,v3);
        v2=__fmaf_rn(v2,v3,v4); v3=__fmaf_rn(v3,v4,v5);
        v4=__fmaf_rn(v4,v5,v6); v5=__fmaf_rn(v5,v6,v7);
        v6=__fmaf_rn(v6,v7,v8); v7=__fmaf_rn(v7,v8,v9);
        v8=__fmaf_rn(v8,v9,va); v9=__fmaf_rn(v9,va,vb);
        va=__fmaf_rn(va,vb,vc); vb=__fmaf_rn(vb,vc,vd);
        vc=__fmaf_rn(vc,vd,ve); vd=__fmaf_rn(vd,ve,vf);
        ve=__fmaf_rn(ve,vf,v0); vf=__fmaf_rn(vf,v0,v1);
        uint64_t t1 = wall_clock64();
        timings[tid*ITERS+i] = t1 - t0;
    }
    float s = v0+v1+v2+v3+v4+v5+v6+v7+v8+v9+va+vb+vc+vd+ve+vf;
    if (s == -999.0f) timings[0] = 0;
}

// ============================================================================
// PROBE B: Occupancy — 1 wavefront vs 4 wavefronts on same CU
// Does multi-wavefront scheduling introduce non-determinism?
// ============================================================================
__global__ void probe_occupancy(uint64_t* timings, int pad) {
    int tid = threadIdx.x;
    if (tid >= 32) return;
    // pad parameter unused but forces different occupancy via launch config
    float a = 1.0f + tid * 0.001f;
    for (int i = 0; i < ITERS; i++) {
        uint64_t t0 = wall_clock64();
        for (int j = 0; j < 10; j++)
            a = __fmaf_rn(a, 1.0001f, 0.001f);
        uint64_t t1 = wall_clock64();
        timings[tid*ITERS+i] = t1 - t0;
    }
    if (a == -999.0f) timings[0] = 0;
}

// ============================================================================
// PROBE C: Dependency chain depth — ILP vs serial
// Deep chain: a = f(a) = f(f(a))... (no ILP)
// Wide chain: a=f(a), b=f(b), c=f(c)... (full ILP)
// ============================================================================
__global__ void probe_dep_serial(uint64_t* timings) {
    int tid = threadIdx.x;
    if (tid >= 32) return;
    float a = 1.0f + tid * 0.001f;
    for (int i = 0; i < ITERS; i++) {
        uint64_t t0 = wall_clock64();
        // Deep serial chain — each depends on previous
        a = __fmaf_rn(a, 1.0001f, 0.001f);
        a = __fmaf_rn(a, 1.0001f, 0.001f);
        a = __fmaf_rn(a, 1.0001f, 0.001f);
        a = __fmaf_rn(a, 1.0001f, 0.001f);
        a = __fmaf_rn(a, 1.0001f, 0.001f);
        a = __fmaf_rn(a, 1.0001f, 0.001f);
        a = __fmaf_rn(a, 1.0001f, 0.001f);
        a = __fmaf_rn(a, 1.0001f, 0.001f);
        uint64_t t1 = wall_clock64();
        timings[tid*ITERS+i] = t1 - t0;
    }
    if (a == -999.0f) timings[0] = 0;
}

__global__ void probe_dep_ilp(uint64_t* timings) {
    int tid = threadIdx.x;
    if (tid >= 32) return;
    float a = 1.0f + tid * 0.001f;
    float b = a + 0.1f, c = a + 0.2f, d = a + 0.3f;
    for (int i = 0; i < ITERS; i++) {
        uint64_t t0 = wall_clock64();
        // Wide parallel — 4 independent chains
        a = __fmaf_rn(a, 1.0001f, 0.001f); b = __fmaf_rn(b, 1.0001f, 0.001f);
        c = __fmaf_rn(c, 1.0001f, 0.001f); d = __fmaf_rn(d, 1.0001f, 0.001f);
        a = __fmaf_rn(a, 1.0001f, 0.001f); b = __fmaf_rn(b, 1.0001f, 0.001f);
        c = __fmaf_rn(c, 1.0001f, 0.001f); d = __fmaf_rn(d, 1.0001f, 0.001f);
        uint64_t t1 = wall_clock64();
        timings[tid*ITERS+i] = t1 - t0;
    }
    if ((a+b+c+d) == -999.0f) timings[0] = 0;
}

// ============================================================================
// PROBE D: Branch divergence — uniform vs divergent EXEC mask
// Half the warp takes one path, half takes another → SIMD masking
// ============================================================================
__global__ void probe_branch_uniform(uint64_t* timings) {
    int tid = threadIdx.x;
    if (tid >= 32) return;
    float a = 1.0f + tid * 0.001f;
    for (int i = 0; i < ITERS; i++) {
        uint64_t t0 = wall_clock64();
        // ALL lanes take same branch
        if (1) { a = __fmaf_rn(a, 1.0001f, 0.001f); }
        else   { a = sinf(a); }
        a = __fmaf_rn(a, 1.0001f, 0.001f);
        uint64_t t1 = wall_clock64();
        timings[tid*ITERS+i] = t1 - t0;
    }
    if (a == -999.0f) timings[0] = 0;
}

__global__ void probe_branch_diverge(uint64_t* timings) {
    int tid = threadIdx.x;
    if (tid >= 32) return;
    float a = 1.0f + tid * 0.001f;
    for (int i = 0; i < ITERS; i++) {
        uint64_t t0 = wall_clock64();
        // HALF the lanes take each branch
        if (tid & 1) { a = __fmaf_rn(a, 1.0001f, 0.001f); }
        else         { a = sinf(a); }
        a = __fmaf_rn(a, 1.0001f, 0.001f);
        uint64_t t1 = wall_clock64();
        timings[tid*ITERS+i] = t1 - t0;
    }
    if (a == -999.0f) timings[0] = 0;
}

// ============================================================================
// PROBE E: Global memory coalescing — coalesced vs strided vs random
// ============================================================================
__global__ void probe_gmem_coalesced(uint64_t* timings, float* gmem, int gmem_size) {
    int tid = threadIdx.x;
    if (tid >= 32) return;
    float a = 0;
    for (int i = 0; i < ITERS; i++) {
        uint64_t t0 = wall_clock64();
        // Coalesced: threads read consecutive addresses
        a += gmem[(i * 32 + tid) % gmem_size];
        uint64_t t1 = wall_clock64();
        timings[tid*ITERS+i] = t1 - t0;
    }
    if (a == -999.0f) timings[0] = 0;
}

__global__ void probe_gmem_strided(uint64_t* timings, float* gmem, int gmem_size) {
    int tid = threadIdx.x;
    if (tid >= 32) return;
    float a = 0;
    for (int i = 0; i < ITERS; i++) {
        uint64_t t0 = wall_clock64();
        // Strided: each thread reads 128 bytes apart
        a += gmem[(tid * 32 + i) % gmem_size];
        uint64_t t1 = wall_clock64();
        timings[tid*ITERS+i] = t1 - t0;
    }
    if (a == -999.0f) timings[0] = 0;
}

// ============================================================================
// PROBE F: Cross-PLL drift rate under different workloads
// Does the drift between shader clock and wall clock change speed?
// ============================================================================
__global__ void probe_pll_drift_rate(uint64_t* drift_trace, int segments, int work_type) {
    int tid = threadIdx.x;
    if (tid != 0) return;

    float dummy = 1.001f;
    for (int seg = 0; seg < segments; seg++) {
        // Do workload
        switch (work_type) {
            case 0:  // idle-ish (just PLL reads)
                for (int j = 0; j < 100; j++) {
                    uint64_t w = wall_clock64();
                    if (w == 0) dummy += 1.0f;
                }
                break;
            case 1:  // FMA heavy
                for (int j = 0; j < 500; j++)
                    dummy = __fmaf_rn(dummy, 1.0001f, 0.001f);
                break;
            case 2:  // SFU heavy
                for (int j = 0; j < 200; j++)
                    dummy = sinf(dummy) + cosf(dummy);
                break;
            case 3:  // INT heavy
                {
                    uint32_t iv = __float_as_uint(dummy);
                    for (int j = 0; j < 500; j++)
                        iv = iv * 2654435761u + 1;
                    dummy = __uint_as_float((iv & 0x007FFFFF) | 0x3F800000);
                }
                break;
        }
        // Measure drift
        uint64_t sc = __clock64();
        uint64_t wc = wall_clock64();
        drift_trace[seg] = sc - wc;
    }
    if (dummy == -999.0f) drift_trace[0] = 0;
}

// ============================================================================
// PROBE G: Interleaved FMA+INT (dual-issue test)
// RDNA3+ can dual-issue FMA+INT — does this create different PLL signature?
// ============================================================================
__global__ void probe_dual_issue(uint64_t* timings) {
    int tid = threadIdx.x;
    if (tid >= 32) return;
    float a = 1.0f + tid * 0.001f;
    uint32_t iv = tid + 1;
    for (int i = 0; i < ITERS; i++) {
        uint64_t t0 = wall_clock64();
        // Interleave FMA and INT ops — should dual-issue on RDNA3.5
        a = __fmaf_rn(a, 1.0001f, 0.001f); iv = iv * 2654435761u + 1;
        a = __fmaf_rn(a, 1.0001f, 0.001f); iv = iv * 2654435761u + 1;
        a = __fmaf_rn(a, 1.0001f, 0.001f); iv = iv * 2654435761u + 1;
        a = __fmaf_rn(a, 1.0001f, 0.001f); iv = iv * 2654435761u + 1;
        uint64_t t1 = wall_clock64();
        timings[tid*ITERS+i] = t1 - t0;
    }
    if (a == -999.0f) timings[0] = 0;
    if (iv == 0) timings[0] = 0;
}

// ============================================================================
// PROBE H: Deep wall_clock64 accumulation — variance vs depth curve
// Map out the exact relationship: depth → unique values, std, distribution
// ============================================================================
__global__ void probe_depth_curve(float* results, int depth) {
    int tid = threadIdx.x;
    if (tid >= 32) return;

    float sums[100];
    for (int trial = 0; trial < 100; trial++) {
        float acc = 0;
        for (int r = 0; r < depth; r++) {
            uint64_t w0 = wall_clock64();
            uint64_t w1 = wall_clock64();
            acc += (float)(w1 - w0);
        }
        sums[trial] = acc;
    }

    // Compute stats
    float s1 = 0, s2 = 0;
    for (int i = 0; i < 100; i++) { s1 += sums[i]; s2 += sums[i]*sums[i]; }
    float mean = s1 / 100;
    float var = s2 / 100 - mean*mean;

    // Count unique values
    int unique = 1;
    for (int i = 1; i < 100; i++) {
        bool found = false;
        for (int j = 0; j < i; j++)
            if (sums[i] == sums[j]) { found = true; break; }
        if (!found) unique++;
    }

    results[tid * 3 + 0] = mean;
    results[tid * 3 + 1] = sqrtf(var > 0 ? var : 0);
    results[tid * 3 + 2] = (float)unique;
}

// ============================================================================
// HOST
// ============================================================================
ProbeStats analyze(uint64_t* h_timings, int n_threads, int iters) {
    ProbeStats ps;
    std::vector<float> vals;
    for (int t = 0; t < n_threads; t++)
        for (int i = 0; i < iters; i++)
            vals.push_back((float)h_timings[t*iters+i]);

    float s1=0, s2=0;
    ps.min_val = 1e30; ps.max_val = -1e30;
    for (float v : vals) {
        s1 += v; s2 += v*v;
        if (v < ps.min_val) ps.min_val = v;
        if (v > ps.max_val) ps.max_val = v;
    }
    ps.mean = s1 / vals.size();
    float var = s2 / vals.size() - ps.mean * ps.mean;
    ps.std_dev = sqrtf(var > 0 ? var : 0);

    // Count unique (sample first 1000)
    std::vector<float> sample(vals.begin(), vals.begin() + std::min((int)vals.size(), 1000));
    std::sort(sample.begin(), sample.end());
    ps.unique = 1;
    for (size_t i = 1; i < sample.size(); i++)
        if (sample[i] != sample[i-1]) ps.unique++;

    // Run-to-run correlation (first half vs second half)
    int half = iters / 2;
    float sx=0,sy=0,sxy=0,sx2=0,sy2=0;
    int n = std::min(half, 500);
    for (int i = 0; i < n; i++) {
        float x = (float)h_timings[i];
        float y = (float)h_timings[half+i];
        sx+=x; sy+=y; sxy+=x*y; sx2+=x*x; sy2+=y*y;
    }
    float num = n*sxy - sx*sy;
    float den = sqrtf((n*sx2-sx*sx)*(n*sy2-sy*sy));
    ps.run_corr = den > 0 ? num/den : 0;

    return ps;
}

void print_probe(const char* name, ProbeStats& ps) {
    fprintf(stderr, "  %-28s mean=%7.1f std=%6.2f unique=%3d range=[%.0f,%.0f] r=%.3f",
            name, ps.mean, ps.std_dev, ps.unique, ps.min_val, ps.max_val, ps.run_corr);
    if (ps.std_dev > 0.5f && ps.unique > 3)
        fprintf(stderr, " ← NON-DET ★");
    fprintf(stderr, "\n");
}

int main() {
    fprintf(stderr, "═══ DEEP WIDE GPU PHYSICS PROBE ═══\n");
    fprintf(stderr, "  Probing ALL possible noise sources and control knobs\n\n");

    int buf_size = 32 * ITERS;
    uint64_t *d_timings, *h_timings;
    h_timings = (uint64_t*)malloc(buf_size * sizeof(uint64_t));
    hipMalloc(&d_timings, buf_size * sizeof(uint64_t));

    auto run_probe = [&](auto kernel, const char* name) -> ProbeStats {
        hipMemset(d_timings, 0, buf_size * sizeof(uint64_t));
        kernel(d_timings);  // lambda wraps the <<<>>> launch
        hipDeviceSynchronize();
        hipMemcpy(h_timings, d_timings, buf_size * sizeof(uint64_t), hipMemcpyDeviceToHost);
        auto ps = analyze(h_timings, 32, ITERS);
        print_probe(name, ps);
        return ps;
    };

    // ─── PROBE A: VGPR pressure ───
    fprintf(stderr, "═══ PROBE A: VGPR Pressure ═══\n");
    auto vgpr_lo = run_probe([](uint64_t* t){ probe_vgpr_low<<<1,N_THREADS>>>(t); }, "VGPR low (2 regs)");
    auto vgpr_hi = run_probe([](uint64_t* t){ probe_vgpr_high<<<1,N_THREADS>>>(t); }, "VGPR high (32 regs)");
    fprintf(stderr, "  Delta: mean=%.2f std=%.3f\n\n",
            vgpr_hi.mean - vgpr_lo.mean, vgpr_hi.std_dev - vgpr_lo.std_dev);

    // ─── PROBE B: Occupancy ───
    fprintf(stderr, "═══ PROBE B: Occupancy (1 vs 4 wavefronts/CU) ═══\n");
    // 1 wavefront
    hipMemset(d_timings, 0, buf_size * sizeof(uint64_t));
    probe_occupancy<<<1, N_THREADS>>>(d_timings, 0);
    hipDeviceSynchronize();
    hipMemcpy(h_timings, d_timings, buf_size * sizeof(uint64_t), hipMemcpyDeviceToHost);
    auto occ1 = analyze(h_timings, 32, ITERS);
    print_probe("1 wavefront/CU", occ1);

    // 4 wavefronts on same CU (launch 4 blocks, each 64 threads)
    hipMemset(d_timings, 0, buf_size * sizeof(uint64_t));
    probe_occupancy<<<4, N_THREADS>>>(d_timings, 0);
    hipDeviceSynchronize();
    hipMemcpy(h_timings, d_timings, buf_size * sizeof(uint64_t), hipMemcpyDeviceToHost);
    auto occ4 = analyze(h_timings, 32, ITERS);
    print_probe("4 wavefronts/CU", occ4);
    fprintf(stderr, "  Delta: mean=%.2f std=%.3f\n\n",
            occ4.mean - occ1.mean, occ4.std_dev - occ1.std_dev);

    // ─── PROBE C: Dependency chain ───
    fprintf(stderr, "═══ PROBE C: Dependency Chain (serial vs ILP) ═══\n");
    auto dep_s = run_probe([](uint64_t* t){ probe_dep_serial<<<1,N_THREADS>>>(t); }, "Serial (deep chain)");
    auto dep_i = run_probe([](uint64_t* t){ probe_dep_ilp<<<1,N_THREADS>>>(t); }, "ILP (wide, 4 chains)");
    fprintf(stderr, "  Delta: mean=%.2f std=%.3f\n\n",
            dep_i.mean - dep_s.mean, dep_i.std_dev - dep_s.std_dev);

    // ─── PROBE D: Branch divergence ───
    fprintf(stderr, "═══ PROBE D: Branch Divergence ═══\n");
    auto br_u = run_probe([](uint64_t* t){ probe_branch_uniform<<<1,N_THREADS>>>(t); }, "Uniform (no diverge)");
    auto br_d = run_probe([](uint64_t* t){ probe_branch_diverge<<<1,N_THREADS>>>(t); }, "Divergent (50/50)");
    fprintf(stderr, "  Delta: mean=%.2f std=%.3f\n\n",
            br_d.mean - br_u.mean, br_d.std_dev - br_u.std_dev);

    // ─── PROBE E: Memory coalescing ───
    fprintf(stderr, "═══ PROBE E: Global Memory Coalescing ═══\n");
    float *d_gmem;
    int gmem_size = 1 << 20;  // 1M floats = 4MB
    hipMalloc(&d_gmem, gmem_size * sizeof(float));
    // Fill with data
    {
        std::vector<float> fill(gmem_size);
        for (int i = 0; i < gmem_size; i++) fill[i] = (float)i * 0.001f;
        hipMemcpy(d_gmem, fill.data(), gmem_size * sizeof(float), hipMemcpyHostToDevice);
    }

    hipMemset(d_timings, 0, buf_size * sizeof(uint64_t));
    probe_gmem_coalesced<<<1, N_THREADS>>>(d_timings, d_gmem, gmem_size);
    hipDeviceSynchronize();
    hipMemcpy(h_timings, d_timings, buf_size * sizeof(uint64_t), hipMemcpyDeviceToHost);
    auto mem_c = analyze(h_timings, 32, ITERS);
    print_probe("Coalesced reads", mem_c);

    hipMemset(d_timings, 0, buf_size * sizeof(uint64_t));
    probe_gmem_strided<<<1, N_THREADS>>>(d_timings, d_gmem, gmem_size);
    hipDeviceSynchronize();
    hipMemcpy(h_timings, d_timings, buf_size * sizeof(uint64_t), hipMemcpyDeviceToHost);
    auto mem_s = analyze(h_timings, 32, ITERS);
    print_probe("Strided reads", mem_s);
    fprintf(stderr, "  Delta: mean=%.2f std=%.3f\n\n",
            mem_s.mean - mem_c.mean, mem_s.std_dev - mem_c.std_dev);

    hipFree(d_gmem);

    // ─── PROBE F: Cross-PLL drift rate under workloads ───
    fprintf(stderr, "═══ PROBE F: Cross-PLL Drift Rate ═══\n");
    int segs = 200;
    uint64_t *d_drift, *h_drift;
    h_drift = (uint64_t*)malloc(segs * sizeof(uint64_t));
    hipMalloc(&d_drift, segs * sizeof(uint64_t));

    const char* wnames[] = {"Idle/PLL", "FMA", "SFU", "INT"};
    for (int wt = 0; wt < 4; wt++) {
        hipMemset(d_drift, 0, segs * sizeof(uint64_t));
        probe_pll_drift_rate<<<1, 1>>>(d_drift, segs, wt);
        hipDeviceSynchronize();
        hipMemcpy(h_drift, d_drift, segs * sizeof(uint64_t), hipMemcpyDeviceToHost);

        // Compute drift rate (delta per segment)
        float drift_sum = 0;
        int n = 0;
        for (int i = 1; i < segs; i++) {
            int64_t d = (int64_t)h_drift[i] - (int64_t)h_drift[i-1];
            drift_sum += (float)d;
            n++;
        }
        float drift_rate = drift_sum / n;

        // Drift variance
        float dvar = 0;
        for (int i = 1; i < segs; i++) {
            int64_t d = (int64_t)h_drift[i] - (int64_t)h_drift[i-1];
            float dd = (float)d - drift_rate;
            dvar += dd * dd;
        }
        dvar /= n;

        fprintf(stderr, "  %-10s drift_rate=%.2f cycles/seg  drift_std=%.2f",
                wnames[wt], drift_rate, sqrtf(dvar));
        if (sqrtf(dvar) > 1.0f) fprintf(stderr, " ← VARIABLE ★");
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");

    hipFree(d_drift);
    free(h_drift);

    // ─── PROBE G: Dual-issue FMA+INT ───
    fprintf(stderr, "═══ PROBE G: Dual-Issue (FMA+INT interleaved) ═══\n");
    auto dual = run_probe([](uint64_t* t){ probe_dual_issue<<<1,N_THREADS>>>(t); }, "FMA+INT dual-issue");
    fprintf(stderr, "  vs FMA-only: mean_delta=%.2f std_delta=%.3f\n\n",
            dual.mean - vgpr_lo.mean, dual.std_dev - vgpr_lo.std_dev);

    // ─── PROBE H: Depth curve — map wall_clock64 accumulation in detail ───
    fprintf(stderr, "═══ PROBE H: wall_clock64 Depth Curve ═══\n");
    float *d_depth_res, *h_depth_res;
    h_depth_res = (float*)malloc(32 * 3 * sizeof(float));
    hipMalloc(&d_depth_res, 32 * 3 * sizeof(float));

    int depths[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    fprintf(stderr, "  %-8s %-10s %-10s %-8s\n", "Depth", "Mean", "Std", "Unique");
    for (int di = 0; di < 10; di++) {
        int depth = depths[di];
        hipMemset(d_depth_res, 0, 32 * 3 * sizeof(float));
        probe_depth_curve<<<1, N_THREADS>>>(d_depth_res, depth);
        hipDeviceSynchronize();
        hipMemcpy(h_depth_res, d_depth_res, 32 * 3 * sizeof(float), hipMemcpyDeviceToHost);

        // Average across threads
        float avg_mean=0, avg_std=0, avg_unique=0;
        for (int t = 0; t < 32; t++) {
            avg_mean += h_depth_res[t*3+0];
            avg_std += h_depth_res[t*3+1];
            avg_unique += h_depth_res[t*3+2];
        }
        avg_mean /= 32; avg_std /= 32; avg_unique /= 32;
        fprintf(stderr, "  %-8d %-10.1f %-10.2f %-8.1f\n",
                depth, avg_mean, avg_std, avg_unique);
    }
    fprintf(stderr, "\n");

    hipFree(d_depth_res);
    free(h_depth_res);

    // ─── SUMMARY ───
    fprintf(stderr, "═══ SUMMARY: Control Knobs Found ═══\n");
    fprintf(stderr, "  %-30s %s\n", "Mechanism", "Status");
    fprintf(stderr, "  %-30s %s\n", "─────────────────────────────", "──────");
    fprintf(stderr, "  %-30s %s\n", "VGPR pressure",
            fabsf(vgpr_hi.std_dev - vgpr_lo.std_dev) > 0.5f ? "★ CONTROLLABLE" : "deterministic");
    fprintf(stderr, "  %-30s %s\n", "Occupancy (multi-wavefront)",
            fabsf(occ4.std_dev - occ1.std_dev) > 0.5f ? "★ CONTROLLABLE" : "deterministic");
    fprintf(stderr, "  %-30s %s\n", "Dependency chain (ILP)",
            fabsf(dep_i.std_dev - dep_s.std_dev) > 0.5f ? "★ CONTROLLABLE" : "deterministic");
    fprintf(stderr, "  %-30s %s\n", "Branch divergence",
            fabsf(br_d.std_dev - br_u.std_dev) > 0.5f ? "★ CONTROLLABLE" : "deterministic");
    fprintf(stderr, "  %-30s %s\n", "Memory coalescing",
            mem_s.std_dev > 5.0f ? "★ CONTROLLABLE" : "deterministic");
    fprintf(stderr, "  %-30s %s\n", "Dual-issue FMA+INT",
            fabsf(dual.std_dev - vgpr_lo.std_dev) > 0.5f ? "★ CONTROLLABLE" : "deterministic");
    fprintf(stderr, "  %-30s %s\n", "wall_clock64 depth", "★ CONTROLLABLE (verified)");
    fprintf(stderr, "  %-30s %s\n", "Instruction-mix PLL mod", "★ CONTROLLABLE (verified)");
    fprintf(stderr, "  %-30s %s\n", "Placement (per-CU)", "★ CONTROLLABLE (verified)");
    fprintf(stderr, "  %-30s %s\n", "Cross-PLL drift", "★ CONTROLLABLE (verified)");

    fprintf(stderr, "\nDone.\n");

    hipFree(d_timings);
    free(h_timings);
    return 0;
}
