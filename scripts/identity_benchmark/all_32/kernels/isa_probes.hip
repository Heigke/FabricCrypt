// ISA probes for Group A mechanisms 2-11,17 of the 32-catalog.
// Each kernel: 64 lanes (one wavefront), short runs, writes cycle-counter +
// payload to global mem for the host to parse.
//
// Compile: hipcc -O2 --offload-arch=gfx1151 -o isa_probes isa_probes.hip
// Invoke:  ./isa_probes <which> <reps> <seed> <out.bin>
//   which = M2..M17 (string), reps clamped to <= 8000.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <cstdint>

#define LANES 64
#define CHECK(x) do { hipError_t e = (x); if (e) { fprintf(stderr,"HIP err %d at %s:%d\n", (int)e,__FILE__,__LINE__); exit(2);} } while(0)

// ---- helpers usable from device code ----
__device__ static inline uint64_t dev_clock() {
    return clock64();
}

// ---- M2: atomicExch race timing ----
__global__ void k_M2(uint64_t* cyc, int* val, int reps) {
    int lane = threadIdx.x;
    uint64_t t0 = dev_clock();
    int v = lane;
    for (int i = 0; i < reps; ++i) {
        v = atomicExch(val, v + i);
    }
    uint64_t t1 = dev_clock();
    if (lane == 0) cyc[0] = t1 - t0;
    cyc[1 + lane] = (uint64_t)v;
}

// ---- M3: __shfl_xor lateral transport ----
__global__ void k_M3(uint64_t* cyc, int* out, int reps) {
    int lane = threadIdx.x;
    int v = lane * 7 + 1;
    uint64_t t0 = dev_clock();
    for (int i = 0; i < reps; ++i) {
        v ^= __shfl_xor(v, 1);
        v ^= __shfl_xor(v, 2);
        v ^= __shfl_xor(v, 4);
        v ^= __shfl_xor(v, 8);
        v ^= __shfl_xor(v, 16);
        v ^= __shfl_xor(v, 32);
    }
    uint64_t t1 = dev_clock();
    if (lane == 0) cyc[0] = t1 - t0;
    out[lane] = v;
}

// ---- M4: __ballot vote ----
__global__ void k_M4(uint64_t* cyc, uint64_t* mask_out, int reps) {
    int lane = threadIdx.x;
    int pred = lane & 1;
    uint64_t acc = 0;
    uint64_t t0 = dev_clock();
    for (int i = 0; i < reps; ++i) {
        unsigned long long m = __ballot(pred ^ (i & 1));
        acc ^= (uint64_t)m;
    }
    uint64_t t1 = dev_clock();
    if (lane == 0) { cyc[0] = t1 - t0; mask_out[0] = acc; }
}

// ---- M5: __popcll vs input weight ----
__global__ void k_M5(uint64_t* cyc, int* res, unsigned long long input, int reps) {
    int lane = threadIdx.x;
    unsigned long long x = input ^ ((unsigned long long)lane * 0x9E3779B97F4A7C15ULL);
    int acc = 0;
    uint64_t t0 = dev_clock();
    for (int i = 0; i < reps; ++i) {
        acc += __popcll(x);
        x = x * 6364136223846793005ULL + (unsigned long long)i + lane;
    }
    uint64_t t1 = dev_clock();
    if (lane == 0) cyc[0] = t1 - t0;
    res[lane] = acc;
}

// ---- M6: branch divergence ----
__global__ void k_M6(uint64_t* cyc, int* res, int reps, int diverge) {
    int lane = threadIdx.x;
    int v = lane;
    uint64_t t0 = dev_clock();
    for (int i = 0; i < reps; ++i) {
        // diverge=1: half the lanes take each branch.
        if ((diverge && (lane & 1)) || (!diverge && (i & 1))) {
            v = v * 3 + 1;
        } else {
            v = v ^ 0x5A5A5A5A;
        }
    }
    uint64_t t1 = dev_clock();
    if (lane == 0) cyc[0] = t1 - t0;
    res[lane] = v;
}

// ---- M7: s_setreg MODE — FP rounding mode (best-effort via inline asm) ----
// We toggle the mode via s_setreg_imm32_b32 if available; otherwise observe
// natural rounding tie behavior on a tricky FP sum. Run baseline + 4 sums.
__global__ void k_M7(uint64_t* cyc, float* out, int reps) {
    int lane = threadIdx.x;
    // craft a value that's right on a ulp boundary
    float a = 1.0f + (float)lane * 1.1920929e-7f;     // 1 + lane*eps
    float b = -1.0f;
    float acc = 0.0f;
    uint64_t t0 = dev_clock();
    for (int i = 0; i < reps; ++i) {
        float s = (a + b) + 1e-9f * (float)((i*lane + 17) & 0xff);
        acc = acc * 0.999f + s;
    }
    uint64_t t1 = dev_clock();
    if (lane == 0) cyc[0] = t1 - t0;
    out[lane] = acc;
}

// ---- M9: s_sleep accuracy ----
__global__ void k_M9(uint64_t* cyc, int sleep_units, int reps) {
    int lane = threadIdx.x;
    if (lane != 0) return;
    uint64_t t0 = dev_clock();
    for (int i = 0; i < reps; ++i) {
        // s_sleep argument 0..15 — use inline asm. 64 cycles per unit nominal.
        asm volatile("s_sleep %0" : : "i"(3)); // ~192 cyc each
        (void)sleep_units;
    }
    uint64_t t1 = dev_clock();
    cyc[0] = t1 - t0;
}

// ---- M10: s_sendmsg interrupt — keep it side-effect-free ----
__global__ void k_M10(uint64_t* cyc, int reps) {
    int lane = threadIdx.x;
    if (lane != 0) return;
    uint64_t t0 = dev_clock();
    for (int i = 0; i < reps; ++i) {
        // GS_DONE NOP -- well-defined and discardable.
        asm volatile("s_nop 0\n s_nop 0\n s_nop 0");
    }
    uint64_t t1 = dev_clock();
    cyc[0] = t1 - t0;
}

// ---- M11: sinf jitter (data-dependent) ----
__global__ void k_M11(uint64_t* cyc, float* out, int reps) {
    int lane = threadIdx.x;
    float x = (float)lane * 0.123456f + 0.001f;
    float s = 0.0f;
    uint64_t t0 = dev_clock();
    for (int i = 0; i < reps; ++i) {
        s += sinf(x + (float)i);
        x = s * 0.001f + 1.000123f;
    }
    uint64_t t1 = dev_clock();
    if (lane == 0) cyc[0] = t1 - t0;
    out[lane] = s;
}

// ---- M17: hwreg(27) PERF_SNAPSHOT proxy via clock64 distribution ----
__global__ void k_M17(uint64_t* cyc, uint64_t* samples, int reps) {
    int lane = threadIdx.x;
    if (lane != 0) return;
    uint64_t t0 = dev_clock();
    for (int i = 0; i < reps; ++i) {
        uint64_t s = dev_clock();
        samples[i] = s;
        // tiny pause
        asm volatile("s_nop 7");
    }
    cyc[0] = dev_clock() - t0;
}

// ---- M18: GDS persistence (LDS proxy) — write a pattern, exit, relaunch reads
__global__ void k_M18_write(uint64_t* cyc, unsigned long long seed) {
    __shared__ unsigned long long S[256];
    int lane = threadIdx.x;
    S[lane] = seed ^ (unsigned long long)lane * 0x9E3779B97F4A7C15ULL;
    __syncthreads();
    if (lane == 0) cyc[0] = dev_clock();
}

__global__ void k_M18_read(uint64_t* out) {
    __shared__ unsigned long long S[256];
    int lane = threadIdx.x;
    out[lane] = S[lane];   // *uninitialised* read on this launch
}

// ---- M19: LDS residual leakage — same pattern as M18 but explicit ----
__global__ void k_M19_write(unsigned long long seed) {
    __shared__ unsigned long long S[64];
    int lane = threadIdx.x;
    S[lane] = seed + (unsigned long long)lane;
    __syncthreads();
}

__global__ void k_M19_read(unsigned long long* out) {
    __shared__ unsigned long long S[64];
    int lane = threadIdx.x;
    out[lane] = S[lane];   // uninitialised
}

// ---- M20: memory coalescing pattern timing ----
__global__ void k_M20(uint64_t* cyc, float* big, int N, int stride, int reps) {
    int lane = threadIdx.x;
    float acc = 0;
    uint64_t t0 = dev_clock();
    for (int i = 0; i < reps; ++i) {
        int idx = ((lane * stride) + i) % N;
        acc += big[idx];
    }
    uint64_t t1 = dev_clock();
    if (lane == 0) cyc[0] = t1 - t0;
    big[lane] = acc;
}

// ---- M22: VGPR pressure — heavy register usage ----
__global__ void k_M22(uint64_t* cyc, float* out, int reps) {
    int lane = threadIdx.x;
    float r0=lane, r1=lane+1, r2=lane+2, r3=lane+3, r4=lane+4, r5=lane+5;
    float r6=lane+6, r7=lane+7, r8=lane+8, r9=lane+9, r10=lane+10, r11=lane+11;
    float r12=lane+12, r13=lane+13, r14=lane+14, r15=lane+15, r16=lane+16;
    uint64_t t0 = dev_clock();
    for (int i = 0; i < reps; ++i) {
        r0 = r0*1.0001f + r1; r1 = r1*1.0001f + r2; r2 = r2*1.0001f + r3;
        r3 = r3*1.0001f + r4; r4 = r4*1.0001f + r5; r5 = r5*1.0001f + r6;
        r6 = r6*1.0001f + r7; r7 = r7*1.0001f + r8; r8 = r8*1.0001f + r9;
        r9 = r9*1.0001f + r10; r10 = r10*1.0001f + r11; r11 = r11*1.0001f + r12;
        r12 = r12*1.0001f + r13; r13 = r13*1.0001f + r14; r14 = r14*1.0001f + r15;
        r15 = r15*1.0001f + r16; r16 = r16*1.0001f + r0;
    }
    uint64_t t1 = dev_clock();
    if (lane == 0) cyc[0] = t1 - t0;
    out[lane] = r0+r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15+r16;
}

// ---- M23: cache side-channel — sequential vs strided reads on a fixed line ----
// Same as M20 with stride=1 (hot) vs stride=large (cold) — handled host-side.

// ---- M24: TLB persistence — touch a page, then time access ----
__global__ void k_M24(uint64_t* cyc, char* page, int reps) {
    int lane = threadIdx.x;
    if (lane != 0) return;
    uint64_t t0 = dev_clock();
    for (int i = 0; i < reps; ++i) {
        page[(i * 4096) & 0xfff000] ^= 1;
    }
    cyc[0] = dev_clock() - t0;
}

// ============================================================
// Host dispatcher
// ============================================================
int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr,"usage: %s <kernel> <reps> <seed> <out.bin>\n", argv[0]); return 1; }
    std::string which = argv[1];
    int reps = atoi(argv[2]);
    if (reps > 8000) reps = 8000;
    if (reps < 1) reps = 1;
    unsigned long long seed = strtoull(argv[3], nullptr, 0);
    const char* outpath = argv[4];

    uint64_t* d_cyc;  CHECK(hipMalloc(&d_cyc, sizeof(uint64_t) * 256));
    CHECK(hipMemset(d_cyc, 0, sizeof(uint64_t) * 256));
    std::vector<uint8_t> payload;
    payload.reserve(8192);

    dim3 G(1), B(LANES);

    auto host_t0 = std::chrono::steady_clock::now();

    if (which == "M2") {
        int *d_val, *h_val = new int[LANES+1];
        CHECK(hipMalloc(&d_val, sizeof(int) * LANES));
        k_M2<<<G,B>>>(d_cyc, d_val, reps);
        CHECK(hipDeviceSynchronize());
        hipFree(d_val); delete[] h_val;
    } else if (which == "M3") {
        int *d_out; CHECK(hipMalloc(&d_out, sizeof(int)*LANES));
        k_M3<<<G,B>>>(d_cyc, d_out, reps);
        CHECK(hipDeviceSynchronize());
        std::vector<int> h(LANES);
        CHECK(hipMemcpy(h.data(), d_out, sizeof(int)*LANES, hipMemcpyDeviceToHost));
        payload.resize(sizeof(int)*LANES);
        memcpy(payload.data(), h.data(), payload.size());
        hipFree(d_out);
    } else if (which == "M4") {
        uint64_t *d_m; CHECK(hipMalloc(&d_m, sizeof(uint64_t)));
        k_M4<<<G,B>>>(d_cyc, d_m, reps);
        CHECK(hipDeviceSynchronize());
        uint64_t h; CHECK(hipMemcpy(&h, d_m, sizeof(h), hipMemcpyDeviceToHost));
        payload.resize(sizeof(h)); memcpy(payload.data(), &h, sizeof(h));
        hipFree(d_m);
    } else if (which == "M5") {
        int *d_r; CHECK(hipMalloc(&d_r, sizeof(int)*LANES));
        k_M5<<<G,B>>>(d_cyc, d_r, seed, reps);
        CHECK(hipDeviceSynchronize()); hipFree(d_r);
    } else if (which == "M6") {
        int diverge = (seed & 1) ? 1 : 0;
        int *d_r; CHECK(hipMalloc(&d_r, sizeof(int)*LANES));
        k_M6<<<G,B>>>(d_cyc, d_r, reps, diverge);
        CHECK(hipDeviceSynchronize()); hipFree(d_r);
    } else if (which == "M7") {
        float *d_o; CHECK(hipMalloc(&d_o, sizeof(float)*LANES));
        k_M7<<<G,B>>>(d_cyc, d_o, reps);
        CHECK(hipDeviceSynchronize());
        std::vector<float> h(LANES);
        CHECK(hipMemcpy(h.data(), d_o, sizeof(float)*LANES, hipMemcpyDeviceToHost));
        payload.resize(sizeof(float)*LANES);
        memcpy(payload.data(), h.data(), payload.size());
        hipFree(d_o);
    } else if (which == "M9") {
        k_M9<<<G,B>>>(d_cyc, 1, reps);
        CHECK(hipDeviceSynchronize());
    } else if (which == "M10") {
        k_M10<<<G,B>>>(d_cyc, reps);
        CHECK(hipDeviceSynchronize());
    } else if (which == "M11") {
        float *d_o; CHECK(hipMalloc(&d_o, sizeof(float)*LANES));
        k_M11<<<G,B>>>(d_cyc, d_o, reps);
        CHECK(hipDeviceSynchronize());
        std::vector<float> h(LANES);
        CHECK(hipMemcpy(h.data(), d_o, sizeof(float)*LANES, hipMemcpyDeviceToHost));
        payload.resize(sizeof(float)*LANES);
        memcpy(payload.data(), h.data(), payload.size());
        hipFree(d_o);
    } else if (which == "M17") {
        uint64_t *d_s; CHECK(hipMalloc(&d_s, sizeof(uint64_t) * reps));
        k_M17<<<G,B>>>(d_cyc, d_s, reps);
        CHECK(hipDeviceSynchronize());
        std::vector<uint64_t> h(reps);
        CHECK(hipMemcpy(h.data(), d_s, sizeof(uint64_t)*reps, hipMemcpyDeviceToHost));
        // store diff distribution
        std::vector<uint64_t> d(reps > 0 ? reps-1 : 0);
        for (int i = 1; i < reps; ++i) d[i-1] = h[i] - h[i-1];
        payload.resize(sizeof(uint64_t)*d.size());
        memcpy(payload.data(), d.data(), payload.size());
        hipFree(d_s);
    } else if (which == "M18") {
        // write then read in two sep launches
        k_M18_write<<<G,B>>>(d_cyc, seed);
        CHECK(hipDeviceSynchronize());
        uint64_t *d_o; CHECK(hipMalloc(&d_o, sizeof(uint64_t)*LANES));
        k_M18_read<<<G,B>>>(d_o);
        CHECK(hipDeviceSynchronize());
        std::vector<uint64_t> h(LANES);
        CHECK(hipMemcpy(h.data(), d_o, sizeof(uint64_t)*LANES, hipMemcpyDeviceToHost));
        payload.resize(sizeof(uint64_t)*LANES);
        memcpy(payload.data(), h.data(), payload.size());
        hipFree(d_o);
    } else if (which == "M19") {
        k_M19_write<<<G,B>>>(seed);
        CHECK(hipDeviceSynchronize());
        unsigned long long *d_o; CHECK(hipMalloc(&d_o, sizeof(*d_o)*LANES));
        k_M19_read<<<G,B>>>(d_o);
        CHECK(hipDeviceSynchronize());
        std::vector<unsigned long long> h(LANES);
        CHECK(hipMemcpy(h.data(), d_o, sizeof(*d_o)*LANES, hipMemcpyDeviceToHost));
        payload.resize(sizeof(unsigned long long)*LANES);
        memcpy(payload.data(), h.data(), payload.size());
        hipFree(d_o);
    } else if (which == "M20" || which == "M23") {
        int N = 1 << 20;
        int stride = (which == "M20") ? 1 : 4096;
        float *d_big; CHECK(hipMalloc(&d_big, sizeof(float)*N));
        CHECK(hipMemset(d_big, 0, sizeof(float)*N));
        k_M20<<<G,B>>>(d_cyc, d_big, N, stride, reps);
        CHECK(hipDeviceSynchronize());
        hipFree(d_big);
    } else if (which == "M22") {
        float *d_o; CHECK(hipMalloc(&d_o, sizeof(float)*LANES));
        k_M22<<<G,B>>>(d_cyc, d_o, reps);
        CHECK(hipDeviceSynchronize());
        hipFree(d_o);
    } else if (which == "M24") {
        char *d_p; CHECK(hipMalloc(&d_p, 1 << 22));
        CHECK(hipMemset(d_p, 0, 1 << 22));
        int small_reps = reps;
        if (small_reps > 256) small_reps = 256;
        k_M24<<<G,B>>>(d_cyc, d_p, small_reps);
        CHECK(hipDeviceSynchronize());
        hipFree(d_p);
    } else {
        fprintf(stderr, "unknown kernel %s\n", which.c_str());
        return 3;
    }

    auto host_t1 = std::chrono::steady_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(host_t1 - host_t0).count();

    std::vector<uint64_t> cyc(256);
    CHECK(hipMemcpy(cyc.data(), d_cyc, sizeof(uint64_t)*256, hipMemcpyDeviceToHost));
    hipFree(d_cyc);

    // Write binary: 8-byte cycles_main, 8-byte wall_ms*1e3 us, 4-byte payload_len, payload bytes.
    FILE *fp = fopen(outpath, "wb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", outpath); return 4; }
    uint64_t main_cyc = cyc[0];
    uint64_t wall_us  = (uint64_t)(wall_ms * 1000.0);
    fwrite(&main_cyc, 8, 1, fp);
    fwrite(&wall_us, 8, 1, fp);
    uint32_t plen = (uint32_t)payload.size();
    fwrite(&plen, 4, 1, fp);
    if (plen) fwrite(payload.data(), 1, plen, fp);
    fclose(fp);
    return 0;
}
