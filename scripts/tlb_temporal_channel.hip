/*  tlb_temporal_channel.hip — z2261: Deep TLB Temporal Channel Exploitation
 *
 *  TLB showed 56.9% at delay=1 (d=0.288, delta=332 clk) in z2260.
 *  Now we optimize: can we push this to 70%+ and use it as temporal memory?
 *
 *  Tests:
 *    1. TLB footprint sweep — how many pages makes strongest signal?
 *    2. TLB probe strategy — probe SAME vs DIFFERENT pages as encode
 *    3. No-sync TLB — encode+probe without hipDeviceSynchronize
 *    4. Multiple TLB sets — encode with pattern A vs B, probe both → differential
 *    5. TLB memory depth — how many steps back can we read?
 *    6. Multi-class TLB — can we encode >2 classes?
 *    7. Combined TLB + Thermal — use both channels together
 *    8. TLB reservoir — full reservoir computing with TLB as temporal channel
 *
 *  Compile: hipcc --offload-arch=gfx1100 -O1 -o scripts/tlb_temporal_channel scripts/tlb_temporal_channel.hip
 *  Run:     HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/tlb_temporal_channel
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ─── Utility ───
double vmean(const std::vector<double>& v) {
    return v.empty() ? 0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}
double vstd(const std::vector<double>& v) {
    double m = vmean(v); double s=0;
    for (auto x:v) s+=(x-m)*(x-m);
    return sqrt(s/std::max((size_t)1,v.size()));
}
double vcorr(const std::vector<double>& a, const std::vector<double>& b) {
    if(a.size()!=b.size()||a.size()<5) return 0;
    double ma=vmean(a),mb=vmean(b),sa=vstd(a),sb=vstd(b);
    if(sa<1e-12||sb<1e-12) return 0;
    double r=0; for(size_t i=0;i<a.size();i++) r+=(a[i]-ma)*(b[i]-mb);
    return r/(a.size()*sa*sb);
}
double classify(const std::vector<int>& cls, const std::vector<double>& v, int d) {
    std::vector<double> h,l;
    for(size_t i=d;i<cls.size()&&i<v.size();i++){
        if(cls[i-d]==1) h.push_back(v[i]); else l.push_back(v[i]);
    }
    if(h.empty()||l.empty()) return 0.5;
    double t=(vmean(h)+vmean(l))/2.0;
    int c1=0,c2=0,n=0;
    for(size_t i=d;i<cls.size()&&i<v.size();i++){
        int pc=cls[i-d]; double val=v[i];
        if((pc==1&&val>t)||(pc==0&&val<=t)) c1++;
        if((pc==1&&val<=t)||(pc==0&&val>t)) c2++;
        n++;
    }
    return std::max((double)c1/n,(double)c2/n);
}
double cohend(const std::vector<int>& cls, const std::vector<double>& v, int d) {
    std::vector<double> h,l;
    for(size_t i=d;i<cls.size()&&i<v.size();i++){
        if(cls[i-d]==1) h.push_back(v[i]); else l.push_back(v[i]);
    }
    if(h.size()<5||l.size()<5) return 0;
    double sp=sqrt((vstd(h)*vstd(h)+vstd(l)*vstd(l))/2.0);
    if(sp<1e-12) return 0;
    return fabs(vmean(h)-vmean(l))/sp;
}

// ─── TLB encode: touch pages in region A (high addresses) or B (low addresses) ───
__global__ void k_tlb_enc(int input_class, char* buf, long buf_size,
                           int n_pages, long region_offset) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float acc = 0;
    long base = (input_class == 1) ? region_offset : 0;
    // Touch n_pages pages, each 4KB apart (HugePages might be 64KB on AMD)
    for (int i = 0; i < n_pages; i++) {
        long offset = base + ((long)tid * 4096 + (long)i * 65536) % (buf_size / 2);
        acc += (float)buf[offset];
    }
    if (acc == -999.0f) buf[0] = (char)acc;
}

// TLB probe: time access to region A pages → faster if encode was class 1
__global__ void k_tlb_prb(char* buf, long buf_size, int n_pages,
                           long region_offset, unsigned long long* out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float acc = 0;

    // Probe region A (high addresses)
    unsigned long long t0 = wall_clock64();
    for (int i = 0; i < n_pages; i++) {
        long offset = region_offset + ((long)tid * 4096 + (long)i * 65536) % (buf_size / 2);
        acc += (float)buf[offset];
    }
    unsigned long long ta = wall_clock64() - t0;

    // Probe region B (low addresses)
    t0 = wall_clock64();
    for (int i = 0; i < n_pages; i++) {
        long offset = ((long)tid * 4096 + (long)i * 65536) % (buf_size / 2);
        acc += (float)buf[offset];
    }
    unsigned long long tb = wall_clock64() - t0;

    if (tid == 0) {
        out[0] = ta;
        out[1] = tb;
        out[2] = ta - tb; // differential: positive = A slower = encode was B
    }
    if (acc == -999.0f) buf[0] = (char)acc;
}

// TLB probe: time access with different page strides
__global__ void k_tlb_stride_probe(char* buf, long buf_size, int n_accesses,
                                    long stride, unsigned long long* out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float acc = 0;
    unsigned long long t0 = wall_clock64();
    for (int i = 0; i < n_accesses; i++) {
        long offset = ((long)tid * stride + (long)i * stride) % buf_size;
        acc += (float)buf[offset];
    }
    unsigned long long t1 = wall_clock64();
    if (tid == 0) out[0] = t1 - t0;
    if (acc == -999.0f) buf[0] = (char)acc;
}

// Combined thermal + TLB encode
__global__ void k_combined_encode(int input_class, char* tlb_buf, long buf_size,
                                   int n_pages, long region_offset,
                                   int fma_iters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float acc = 0;

    // TLB: touch pages in region based on input
    long base = (input_class == 1) ? region_offset : 0;
    for (int i = 0; i < n_pages; i++) {
        long offset = base + ((long)tid * 4096 + (long)i * 65536) % (buf_size / 2);
        acc += (float)tlb_buf[offset];
    }

    // Thermal: heavy FMA based on input
    float a = 1.0001f + (float)tid * 0.00001f;
    int iters = (input_class == 1) ? fma_iters : fma_iters / 50;
    for (int i = 0; i < iters; i++) a = __fmaf_rn(a, 0.9999f, 0.0001f);

    if (acc == -999.0f || a == -999.0f) tlb_buf[0] = (char)acc;
}

// Combined probe: TLB + FMA timing
__global__ void k_combined_probe(char* tlb_buf, long buf_size, int n_pages,
                                  long region_offset, unsigned long long* out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float acc = 0;

    // TLB probe A
    unsigned long long t0 = wall_clock64();
    for (int i = 0; i < n_pages; i++) {
        long offset = region_offset + ((long)tid * 4096 + (long)i * 65536) % (buf_size / 2);
        acc += (float)tlb_buf[offset];
    }
    unsigned long long ta = wall_clock64() - t0;

    // TLB probe B
    t0 = wall_clock64();
    for (int i = 0; i < n_pages; i++) {
        long offset = ((long)tid * 4096 + (long)i * 65536) % (buf_size / 2);
        acc += (float)tlb_buf[offset];
    }
    unsigned long long tb = wall_clock64() - t0;

    // FMA timing probe
    float a = 1.0001f + (float)tid * 0.00001f;
    t0 = wall_clock64();
    for (int i = 0; i < 200; i++) a = __fmaf_rn(a, 0.9999f, 0.0001f);
    unsigned long long fma_t = wall_clock64() - t0;

    // SFU timing probe
    float s = 1.0f + (float)tid * 0.001f;
    t0 = wall_clock64();
    for (int i = 0; i < 50; i++) s = __sinf(s) + 1.5f;
    unsigned long long sfu_t = wall_clock64() - t0;

    if (tid == 0) {
        out[0] = ta;
        out[1] = tb;
        out[2] = fma_t;
        out[3] = sfu_t;
    }
    if (acc == -999.0f || a == -999.0f || s == -999.0f) tlb_buf[0] = (char)acc;
}


int main() {
    printf("=== z2261: Deep TLB Temporal Channel ===\n\n");

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("GPU: %s, CUs: %d, Clock: %d MHz\n\n", prop.name, prop.multiProcessorCount, prop.clockRate/1000);

    const int N = 500;
    srand(42);
    std::vector<int> inp(N);
    for (int i=0;i<N;i++) inp[i]=i%2;
    for (int i=N-1;i>0;i--){int j=rand()%(i+1);std::swap(inp[i],inp[j]);}

    unsigned long long* d_out;
    HIP_CHECK(hipMalloc(&d_out, 64 * sizeof(unsigned long long)));

    // ═══════════════════════════════════════════════
    // TEST 1: TLB Footprint Sweep
    // How many pages in encode → strongest delay=1 signal?
    // ═══════════════════════════════════════════════
    printf("═══ TEST 1: TLB Footprint Sweep ═══\n\n");
    {
        long buf_size = 512L * 1024 * 1024; // 512MB
        char* d_buf;
        HIP_CHECK(hipMalloc(&d_buf, buf_size));
        HIP_CHECK(hipMemset(d_buf, 0, buf_size));
        long region_off = buf_size / 2;

        int page_counts[] = {16, 64, 128, 256, 512, 1024, 2048, 4096};
        printf("%-12s %8s %8s %8s %8s %8s\n", "N_pages", "acc_d0", "d_d0", "acc_d1", "d_d1", "delta");

        for (int pi = 0; pi < 8; pi++) {
            int np = page_counts[pi];
            std::vector<double> diff_vals;
            unsigned long long h_t[4];

            for (int step = 0; step < N; step++) {
                k_tlb_enc<<<4, 256>>>(inp[step], d_buf, buf_size, np, region_off);
                HIP_CHECK(hipDeviceSynchronize());

                k_tlb_prb<<<1, 256>>>(d_buf, buf_size, std::min(np, 256), region_off, d_out);
                HIP_CHECK(hipDeviceSynchronize());

                HIP_CHECK(hipMemcpy(h_t, d_out, 3*sizeof(unsigned long long), hipMemcpyDeviceToHost));
                diff_vals.push_back((double)h_t[0] - (double)h_t[1]);
            }

            double a0 = classify(inp, diff_vals, 0);
            double d0 = cohend(inp, diff_vals, 0);
            double a1 = classify(inp, diff_vals, 1);
            double d1 = cohend(inp, diff_vals, 1);

            std::vector<double> ah, al;
            for (size_t i=1;i<diff_vals.size();i++){
                if(inp[i-1]==1) ah.push_back(diff_vals[i]); else al.push_back(diff_vals[i]);
            }

            const char* tag = (a1 > 0.58) ? " ★★★" : (a1 > 0.55) ? " ★★" : (a1 > 0.52) ? " ★" : "";
            printf("%-12d %6.1f%% %8.3f %6.1f%% %8.3f %7.1f%s\n",
                   np, a0*100, d0, a1*100, d1, vmean(ah)-vmean(al), tag);
        }
        printf("\n");
        HIP_CHECK(hipFree(d_buf));
    }

    // ═══════════════════════════════════════════════
    // TEST 2: Page Size Sweep (4KB vs 64KB vs 2MB)
    // AMD GPU might use different page sizes
    // ═══════════════════════════════════════════════
    printf("═══ TEST 2: Page Size (Stride) Sweep ═══\n\n");
    {
        long buf_size = 512L * 1024 * 1024;
        char* d_buf;
        HIP_CHECK(hipMalloc(&d_buf, buf_size));
        HIP_CHECK(hipMemset(d_buf, 0, buf_size));

        long strides[] = {4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 2097152};
        const char* stride_names[] = {"4KB", "8KB", "16KB", "32KB", "64KB", "128KB", "256KB", "512KB", "2MB"};
        printf("%-12s %8s %8s %8s %8s\n", "Stride", "acc_d0", "d_d0", "acc_d1", "d_d1");

        for (int si = 0; si < 9; si++) {
            long stride = strides[si];
            std::vector<double> probe_vals;

            for (int step = 0; step < N; step++) {
                // Encode: many pages at given stride (class 1) vs few pages (class 0)
                if (inp[step] == 1) {
                    // Touch 256 pages at this stride
                    k_tlb_stride_probe<<<4, 256>>>(d_buf, buf_size, 256, stride, d_out);
                } else {
                    // Touch only 4 pages (tiny footprint)
                    k_tlb_stride_probe<<<1, 64>>>(d_buf, buf_size, 4, 4096, d_out);
                }
                HIP_CHECK(hipDeviceSynchronize());

                // Probe: fixed access at this stride
                k_tlb_stride_probe<<<1, 256>>>(d_buf, buf_size, 128, stride, d_out);
                HIP_CHECK(hipDeviceSynchronize());

                unsigned long long h_t;
                HIP_CHECK(hipMemcpy(&h_t, d_out, sizeof(unsigned long long), hipMemcpyDeviceToHost));
                probe_vals.push_back((double)h_t);
            }

            double a0 = classify(inp, probe_vals, 0);
            double d0 = cohend(inp, probe_vals, 0);
            double a1 = classify(inp, probe_vals, 1);
            double d1 = cohend(inp, probe_vals, 1);
            const char* tag = (a1 > 0.58) ? " ★★★" : (a1 > 0.55) ? " ★★" : (a1 > 0.52) ? " ★" : "";
            printf("%-12s %6.1f%% %8.3f %6.1f%% %8.3f%s\n",
                   stride_names[si], a0*100, d0, a1*100, d1, tag);
        }
        printf("\n");
        HIP_CHECK(hipFree(d_buf));
    }

    // ═══════════════════════════════════════════════
    // TEST 3: TLB Memory Depth
    // How many steps back can we read?
    // ═══════════════════════════════════════════════
    printf("═══ TEST 3: TLB Memory Depth ═══\n\n");
    {
        long buf_size = 512L * 1024 * 1024;
        char* d_buf;
        HIP_CHECK(hipMalloc(&d_buf, buf_size));
        HIP_CHECK(hipMemset(d_buf, 0, buf_size));
        long region_off = buf_size / 2;

        // Use best page count from sweep (try 512)
        int np = 512;
        std::vector<double> probe_a, probe_b, probe_diff;
        unsigned long long h_t[4];

        for (int step = 0; step < N; step++) {
            k_tlb_enc<<<4, 256>>>(inp[step], d_buf, buf_size, np, region_off);
            HIP_CHECK(hipDeviceSynchronize());

            k_tlb_prb<<<1, 256>>>(d_buf, buf_size, 256, region_off, d_out);
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(h_t, d_out, 3*sizeof(unsigned long long), hipMemcpyDeviceToHost));
            probe_a.push_back((double)h_t[0]);
            probe_b.push_back((double)h_t[1]);
            probe_diff.push_back((double)h_t[0] - (double)h_t[1]);
        }

        printf("Memory depth (512 pages, differential ta-tb):\n");
        printf("%-8s %8s %8s\n", "Delay", "acc", "d");
        for (int d = 0; d <= 10; d++) {
            double a = classify(inp, probe_diff, d);
            double cd = cohend(inp, probe_diff, d);
            const char* tag = (a > 0.55) ? " ★" : "";
            printf("d=%-6d %6.1f%% %8.3f%s\n", d, a*100, cd, tag);
        }

        // ACF of probe values
        double acf1 = vcorr(
            std::vector<double>(probe_diff.begin(), probe_diff.end()-1),
            std::vector<double>(probe_diff.begin()+1, probe_diff.end()));
        printf("\nProbe_A ACF(1)=%.3f, Probe_diff ACF(1)=%.3f\n",
            vcorr(std::vector<double>(probe_a.begin(), probe_a.end()-1),
                  std::vector<double>(probe_a.begin()+1, probe_a.end())),
            acf1);
        printf("\n");
        HIP_CHECK(hipFree(d_buf));
    }

    // ═══════════════════════════════════════════════
    // TEST 4: Multi-Class TLB
    // Can we encode 4 classes using different page regions?
    // ═══════════════════════════════════════════════
    printf("═══ TEST 4: Multi-Class TLB (4 regions) ═══\n\n");
    {
        long buf_size = 512L * 1024 * 1024;
        char* d_buf;
        HIP_CHECK(hipMalloc(&d_buf, buf_size));
        HIP_CHECK(hipMemset(d_buf, 0, buf_size));

        // 4 classes → 4 regions
        long region_size = buf_size / 4;
        std::vector<int> inp4(N);
        for (int i = 0; i < N; i++) inp4[i] = i % 4;
        for (int i = N-1; i > 0; i--) {int j=rand()%(i+1); std::swap(inp4[i],inp4[j]);}

        unsigned long long h_t[8];
        std::vector<std::vector<double>> probes(4); // timing for each region

        for (int step = 0; step < N; step++) {
            int cls = inp4[step];
            long base = cls * region_size;

            // Encode: touch 256 pages in region 'cls'
            k_tlb_stride_probe<<<4, 256>>>(d_buf + base, region_size, 256, 65536, d_out);
            HIP_CHECK(hipDeviceSynchronize());

            // Probe all 4 regions
            for (int r = 0; r < 4; r++) {
                long rbase = r * region_size;
                k_tlb_stride_probe<<<1, 256>>>(d_buf + rbase, region_size, 64, 65536, d_out);
                HIP_CHECK(hipDeviceSynchronize());
                unsigned long long ht;
                HIP_CHECK(hipMemcpy(&ht, d_out, sizeof(unsigned long long), hipMemcpyDeviceToHost));
                probes[r].push_back((double)ht);
            }
        }

        // Classification: which region was fastest? → that's the predicted class
        int correct_d0 = 0, correct_d1 = 0;
        for (int i = 0; i < N; i++) {
            // Same-step
            int pred = 0;
            double min_t = probes[0][i];
            for (int r = 1; r < 4; r++) {
                if (probes[r][i] < min_t) { min_t = probes[r][i]; pred = r; }
            }
            if (pred == inp4[i]) correct_d0++;

            // Delay=1
            if (i > 0) {
                pred = 0; min_t = probes[0][i];
                for (int r = 1; r < 4; r++) {
                    if (probes[r][i] < min_t) { min_t = probes[r][i]; pred = r; }
                }
                if (pred == inp4[i-1]) correct_d1++;
            }
        }

        printf("4-class TLB (chance=25%%):\n");
        printf("  Same-step: %.1f%%\n", (double)correct_d0/N*100);
        printf("  Delay=1:   %.1f%%\n", (double)correct_d1/(N-1)*100);

        // Per-region mean timings when that region was encoded
        printf("\nMean probe time by encode class:\n");
        printf("         ");
        for (int r=0;r<4;r++) printf("  Region%d", r);
        printf("\n");
        for (int c = 0; c < 4; c++) {
            printf("Class %d: ", c);
            for (int r = 0; r < 4; r++) {
                double sum = 0; int cnt = 0;
                for (int i = 0; i < N; i++) {
                    if (inp4[i] == c) { sum += probes[r][i]; cnt++; }
                }
                printf(" %7.0f", sum/cnt);
            }
            printf("\n");
        }
        printf("\n");
        HIP_CHECK(hipFree(d_buf));
    }

    // ═══════════════════════════════════════════════
    // TEST 5: Combined TLB + Thermal Channel
    // ═══════════════════════════════════════════════
    printf("═══ TEST 5: Combined TLB + Thermal ═══\n\n");
    {
        long buf_size = 512L * 1024 * 1024;
        char* d_buf;
        HIP_CHECK(hipMalloc(&d_buf, buf_size));
        HIP_CHECK(hipMemset(d_buf, 0, buf_size));
        long region_off = buf_size / 2;

        std::vector<double> tlb_diff, fma_time, sfu_time;
        unsigned long long h_t[8];

        for (int step = 0; step < N; step++) {
            // Combined encode: TLB pages + thermal FMA burst
            k_combined_encode<<<4, 256>>>(inp[step], d_buf, buf_size, 512, region_off, 10000);
            HIP_CHECK(hipDeviceSynchronize());

            // Combined probe: TLB differential + FMA + SFU timing
            k_combined_probe<<<1, 256>>>(d_buf, buf_size, 256, region_off, d_out);
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(h_t, d_out, 4*sizeof(unsigned long long), hipMemcpyDeviceToHost));
            tlb_diff.push_back((double)h_t[0] - (double)h_t[1]);
            fma_time.push_back((double)h_t[2]);
            sfu_time.push_back((double)h_t[3]);
        }

        // Individual channels
        const char* cn[] = {"TLB_diff", "FMA_time", "SFU_time"};
        std::vector<double>* cv[] = {&tlb_diff, &fma_time, &sfu_time};
        printf("Individual channels:\n");
        for (int c = 0; c < 3; c++) {
            printf("  %s: d=0 %.1f%% (d=%.3f) | d=1 %.1f%% (d=%.3f)\n",
                   cn[c], classify(inp,*cv[c],0)*100, cohend(inp,*cv[c],0),
                   classify(inp,*cv[c],1)*100, cohend(inp,*cv[c],1));
        }

        // Majority vote combination
        int correct = 0;
        for (int i = 1; i < N; i++) {
            // For each channel, vote based on comparison to running median
            int votes = 0;
            for (int c = 0; c < 3; c++) {
                auto& v = *cv[c];
                double rm = 0; int cnt = 0;
                for (int j = std::max(0,i-50); j < i; j++) { rm += v[j]; cnt++; }
                rm /= std::max(cnt,1);
                if (v[i] > rm) votes++;
            }
            int pred = (votes >= 2) ? 1 : 0;
            if (pred == inp[i-1]) correct++;
        }
        double vote_acc = std::max((double)correct/(N-1), 1.0-(double)correct/(N-1));
        printf("  Combined vote (d=1): %.1f%%\n\n", vote_acc*100);

        HIP_CHECK(hipFree(d_buf));
    }

    // ═══════════════════════════════════════════════
    // TEST 6: No-Sync Rapid Fire TLB
    // Launch encode+probe as fast as possible
    // ═══════════════════════════════════════════════
    printf("═══ TEST 6: Rapid-Fire TLB (minimal sync) ═══\n\n");
    {
        long buf_size = 512L * 1024 * 1024;
        char* d_buf;
        HIP_CHECK(hipMalloc(&d_buf, buf_size));
        HIP_CHECK(hipMemset(d_buf, 0, buf_size));
        long region_off = buf_size / 2;

        // Use streams for overlapping
        hipStream_t stream;
        HIP_CHECK(hipStreamCreate(&stream));

        std::vector<double> probe_vals;
        unsigned long long h_t[4];

        for (int step = 0; step < N; step++) {
            // Encode on stream (no host sync between encode and probe!)
            k_tlb_enc<<<4, 256, 0, stream>>>(inp[step], d_buf, buf_size, 512, region_off);
            // Probe immediately on SAME stream (GPU serializes within stream)
            k_tlb_prb<<<1, 256, 0, stream>>>(d_buf, buf_size, 256, region_off, d_out);
            // NOW sync to get result
            HIP_CHECK(hipStreamSynchronize(stream));

            HIP_CHECK(hipMemcpy(h_t, d_out, 3*sizeof(unsigned long long), hipMemcpyDeviceToHost));
            probe_vals.push_back((double)h_t[0] - (double)h_t[1]);
        }

        printf("Rapid-fire (same stream, no inter-kernel host sync):\n");
        printf("  d=0: acc=%.1f%%, d=%.3f\n", classify(inp,probe_vals,0)*100, cohend(inp,probe_vals,0));
        printf("  d=1: acc=%.1f%%, d=%.3f\n", classify(inp,probe_vals,1)*100, cohend(inp,probe_vals,1));

        double acf = vcorr(
            std::vector<double>(probe_vals.begin(), probe_vals.end()-1),
            std::vector<double>(probe_vals.begin()+1, probe_vals.end()));
        printf("  ACF(1)=%.3f\n\n", acf);

        HIP_CHECK(hipStreamDestroy(stream));
        HIP_CHECK(hipFree(d_buf));
    }

    // ═══════════════════════════════════════════════
    // TEST 7: TLB XOR Memory Test
    // Can TLB carry information for XOR(input[t], input[t-1])?
    // ═══════════════════════════════════════════════
    printf("═══ TEST 7: TLB XOR Memory ═══\n\n");
    {
        long buf_size = 512L * 1024 * 1024;
        char* d_buf;
        HIP_CHECK(hipMalloc(&d_buf, buf_size));
        HIP_CHECK(hipMemset(d_buf, 0, buf_size));
        long region_off = buf_size / 2;

        std::vector<double> features; // collect features for each step
        unsigned long long h_t[4];

        for (int step = 0; step < N; step++) {
            k_tlb_enc<<<4, 256>>>(inp[step], d_buf, buf_size, 512, region_off);
            HIP_CHECK(hipDeviceSynchronize());

            k_tlb_prb<<<1, 256>>>(d_buf, buf_size, 256, region_off, d_out);
            HIP_CHECK(hipDeviceSynchronize());

            HIP_CHECK(hipMemcpy(h_t, d_out, 3*sizeof(unsigned long long), hipMemcpyDeviceToHost));
            features.push_back((double)h_t[0] - (double)h_t[1]);
        }

        // XOR targets at different delays
        printf("TLB XOR(t, t-τ) accuracy:\n");
        for (int tau = 1; tau <= 5; tau++) {
            std::vector<int> xor_targets;
            std::vector<double> xor_features;
            for (int i = tau; i < N; i++) {
                xor_targets.push_back(inp[i] ^ inp[i-tau]);
                xor_features.push_back(features[i]);
            }
            double acc = classify(xor_targets, xor_features, 0);
            printf("  XOR τ=%d: %.1f%% (chance=50%%)\n", tau, acc*100);
        }
        printf("\n");

        // Memory capacity
        printf("Memory capacity R² at delays:\n");
        for (int d = 1; d <= 5; d++) {
            // R² of linear regression features[t] ~ inp[t-d]
            double sx=0,sy=0,sxy=0,sx2=0,sy2=0; int n=0;
            for (size_t i=d; i<features.size(); i++) {
                double x=(double)inp[i-d], y=features[i];
                sx+=x; sy+=y; sxy+=x*y; sx2+=x*x; sy2+=y*y; n++;
            }
            double den = (n*sx2-sx*sx)*(n*sy2-sy*sy);
            double r2 = (den > 0) ? pow(n*sxy-sx*sy,2)/den : 0;
            printf("  d=%d: R²=%.4f\n", d, r2);
        }

        HIP_CHECK(hipFree(d_buf));
    }

    printf("\n═══ DONE ═══\n");
    HIP_CHECK(hipFree(d_out));
    return 0;
}
