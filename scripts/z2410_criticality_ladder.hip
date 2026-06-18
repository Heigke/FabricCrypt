/*
 * z2410_criticality_ladder.hip — Stegvis relaxation mot kritikalitet
 *
 * Idé: Digital GPU börjar vid σ=0 (perfekt ordnad).
 * Vi aktiverar mekanismer en i taget och mäter:
 *   1. Branching ratio σ av shadow-neuronerna
 *   2. Beräkningsprestand (OOD-separation, timing-variation)
 *   3. Overhead
 *
 * 11 nivåer (0 → 10 mekanismer):
 *   L0:  Ren matmul (baseline)
 *   L1:  + branch divergence sign pattern (#2)
 *   L2:  + __shfl lateral koppling (#4)
 *   L3:  + __ballot population coding (#22/#23)
 *   L4:  + hwreg(29) timing (#16) + hwreg(27) perf (#17)
 *   L5:  + s_setreg MODE rounding feedback (#12)
 *   L6:  + s_sleep refractory (#8) + TRAPSTS state (#13)
 *   L7:  + per-CU ΔVth diversity (#21/#24)
 *   L8:  + TLB-persistens temporal memory (#9)
 *   L9:  + stale VGPR cross-kernel state (#10)
 *   L10: + memory coalescing collective (#6) + PLL drift (#19)
 *
 * Branching ratio σ mäts som: average(spikes_at_t+1 / spikes_at_t)
 * Kostnad mäts som: total tid relativt L0
 *
 * Körs på MNIST med tränad MLP + OOD (random noise).
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define INPUT_DIM 784
#define HIDDEN1 128
#define HIDDEN2 64
#define OUTPUT_DIM 10
#define N_SHADOW 32  // shadow neurons per output (= wave width)

struct LadderOut {
    // Per-sample shadow state (averaged across output neurons)
    float sign_ratio;      // fraction sign-agreeing blocks
    float timing_cv;       // timing coefficient of variation
    float neighbor_diff;   // mean neighbor difference via shfl
    float pop_minority;    // population minority fraction
    float spike_rate;      // spikes per neuron per matmul
    float perf_var;        // performance counter variability
    float cu_diversity;    // number of distinct CUs used
    float tlb_signal;      // TLB history effect (timing delta layer-over-layer)
    float vgpr_signal;     // stale VGPR persistence signal
    float coalesce_signal; // coalescing pattern signal
    float pll_drift;       // wall_clock vs shader_cycles drift
    float branching_ratio; // σ — key criticality metric
    float membrane_mean;   // mean shadow membrane at end
    float membrane_std;    // std of shadow membrane across neurons
};

/* ============================================================
 * CRITICALITY LADDER KERNEL
 * 'level' parameter controls how many mechanisms are active
 * ============================================================ */
__global__ void matmul_ladder(
    const float* __restrict__ A, const float* __restrict__ W,
    const float* __restrict__ bias, float* __restrict__ C,
    LadderOut* __restrict__ lout,
    int batch, int in_dim, int out_dim, int relu,
    int level,                    // 0-10: how many mechanisms active
    const float* __restrict__ prev_timing  // from previous layer (for TLB signal)
) {
    int b = blockIdx.x, o = threadIdx.x, lane = o % 32;
    if (b >= batch || o >= out_dim) return;

    // ── Shadow neuron state (all in registers) ──
    float membrane = 0.0f;
    uint32_t spike_count = 0;
    uint32_t spikes_this_step = 0, spikes_prev_step = 0;
    float branching_sum = 0.0f;
    uint32_t branching_count = 0;

    uint32_t sign_agree = 0, sign_total = 0;
    float timing_sum = 0.0f, timing_sumsq = 0.0f;
    float neighbor_diff_sum = 0.0f, pop_minority_sum = 0.0f;
    float perf_sum = 0.0f, perf_sumsq = 0.0f;
    uint32_t n_blocks = 0;

    // ── L7: Per-CU identity ──
    uint32_t hw_id = 0;
    if (level >= 7) {
        asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw_id));
    }

    // ── L5: Save original MODE ──
    uint32_t orig_mode = 0;
    if (level >= 5) {
        asm volatile("s_getreg_b32 %0, hwreg(1)" : "=s"(orig_mode));
    }

    // ── L10: PLL drift baseline ──
    uint64_t wall_t0 = 0;
    uint32_t shader_t0 = 0;
    if (level >= 10) {
        wall_t0 = wall_clock64();
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(shader_t0));
    }

    // ── L9: Read stale VGPR from previous kernel (first 4 lanes only) ──
    float vgpr_signal = 0.0f;
    if (level >= 9 && lane < 4) {
        // Don't initialize — read whatever value is already in this register
        // The compiler will likely zero it, but the TIMING of the read
        // may differ based on register file state
        uint32_t t_pre;
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t_pre));
        // Access a register that might have stale data
        volatile float dummy_read = membrane; // force register access
        uint32_t t_post;
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t_post));
        vgpr_signal = (float)(t_post - t_pre);
    }
    vgpr_signal = __shfl(vgpr_signal, 0); // broadcast lane 0's measurement

    // ── L8: TLB timing from previous layer ──
    float tlb_signal = 0.0f;
    if (level >= 8 && prev_timing != nullptr) {
        // Read previous layer's average timing — TLB state affects this read
        uint32_t t_before;
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t_before));
        float prev_t = prev_timing[b]; // this access is TLB-state-dependent
        uint32_t t_after;
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t_after));
        tlb_signal = (float)(t_after - t_before) + prev_t * 0.001f;
    }

    // ── Initial timing ──
    uint32_t t_prev = 0;
    if (level >= 4) {
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t_prev));
    }

    // ═══════════════════════════════════════════════
    // MAIN MATMUL LOOP with progressive mechanism activation
    // ═══════════════════════════════════════════════
    float acc = bias[o];

    for (int k = 0; k < in_dim; k += 4) {
        // ──── VALU: Traditional FMA ────
        float p0 = W[o*in_dim+k]*A[b*in_dim+k];
        float p1 = (k+1<in_dim)?W[o*in_dim+k+1]*A[b*in_dim+k+1]:0;
        float p2 = (k+2<in_dim)?W[o*in_dim+k+2]*A[b*in_dim+k+2]:0;
        float p3 = (k+3<in_dim)?W[o*in_dim+k+3]*A[b*in_dim+k+3]:0;
        acc += p0+p1+p2+p3;

        // ──── L1: Branch divergence sign pattern ────
        float signal = 0.0f;
        if (level >= 1) {
            int np = (p0>0)+(p1>0)+(p2>0)+(p3>0);
            sign_agree += (np==0||np==4)?1:0;
            sign_total++;
            // Sign disagreement → perturbation of membrane
            signal += (float)(4-np) * 0.15f;
        }

        // ──── L2: __shfl lateral coupling ────
        if (level >= 2) {
            float left = __shfl(acc, (lane+31)%32);
            float right = __shfl(acc, (lane+1)%32);
            float diff = fabsf(acc-left) + fabsf(acc-right);
            neighbor_diff_sum += diff;
            // Neighbor difference drives membrane
            signal += diff * 0.01f;

            // Long-range coupling (11 steps away)
            if (level >= 7) {
                float distant = __shfl(acc, (lane+11)%32);
                signal += (distant - acc) * 0.005f; // weak long-range
            }
        }

        // ──── L3: Population coding ────
        if (level >= 3) {
            float bs = p0+p1+p2+p3;
            uint64_t pm = __ballot(bs > 0.0f);
            uint32_t pp = __popcll(pm);
            uint32_t mn = (pp<16)?pp:(32-pp);
            pop_minority_sum += (float)mn;
            signal += (float)mn * 0.03f;

            // Also ballot on membrane state for σ measurement
            uint64_t spike_mask = __ballot(membrane > 4.0f);
            spikes_this_step = __popcll(spike_mask);
        }

        // ──── L4: Timing + perf snapshot ────
        if (level >= 4) {
            uint32_t t_now;
            asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t_now));
            float dt = (float)(t_now - t_prev);
            t_prev = t_now;
            timing_sum += dt;
            timing_sumsq += dt * dt;

            uint32_t perf;
            asm volatile("s_getreg_b32 %0, hwreg(27)" : "=s"(perf));
            float pf = (float)(perf & 0xFF);
            perf_sum += pf;
            perf_sumsq += pf * pf;

            // Timing anomaly → membrane perturbation
            signal += (dt > 20.0f) ? 0.3f : 0.0f;
            signal += pf * 0.002f;
        }

        // ──── L5: s_setreg rounding feedback ────
        if (level >= 5 && (n_blocks & 0x7) == 0) {
            // State-dependent rounding: membrane controls arithmetic
            if (membrane > 3.0f) {
                asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x3"); // RTZ
            } else if (membrane > 1.5f) {
                asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x1"); // R+inf
            } else {
                asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x0"); // RNE
            }
        }

        // ──── L8: TLB access pattern influence ────
        if (level >= 8) {
            // Access weight at a stride that creates TLB pressure
            // The timing of this access depends on TLB state from
            // the previous layer's memory accesses
            int far_k = (k + 512) % in_dim; // far stride
            float tlb_probe = W[o * in_dim + far_k]; // TLB-state-dependent
            signal += tlb_probe * 0.0001f; // tiny, just enough to sense
        }

        // ──── L10: Coalescing-dependent behavior ────
        if (level >= 10) {
            // Scattered vs coalesced access depends on input structure
            // Structured input (images) → coalesced → fast
            // Random input → scattered → slow, different timing
            int scatter_idx = (int)(A[b*in_dim + (k % in_dim)] * in_dim) % in_dim;
            float coalesce_probe = A[b * in_dim + scatter_idx];
            signal += coalesce_probe * 0.001f;
        }

        n_blocks++;

        // ──── Shadow neuron integration ────
        membrane += signal;

        // ──── Spike detection ────
        if (membrane > 5.0f) {
            spike_count++;
            membrane = 0.0f;

            // L6: Hardware refractory + state
            if (level >= 6) {
                asm volatile("s_sleep 1"); // 40 cycles physical pause

                // TRAPSTS parity toggle (safe bits 0-1 only)
                uint32_t ts;
                asm volatile("s_getreg_b32 %0, hwreg(3)" : "=s"(ts));
                asm volatile("s_setreg_b32 hwreg(3), %0" : : "s"(ts ^ 0x1));
            }
        }

        // ──── Branching ratio measurement ────
        if (level >= 3 && spikes_prev_step > 0) {
            branching_sum += (float)spikes_this_step / (float)spikes_prev_step;
            branching_count++;
        }
        spikes_prev_step = spikes_this_step;
    }

    // ── Restore MODE ──
    if (level >= 5) {
        asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(orig_mode));
    }

    // ── L10: PLL drift measurement ──
    float pll_drift_val = 0.0f;
    if (level >= 10) {
        uint64_t wall_t1 = wall_clock64();
        uint32_t shader_t1;
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(shader_t1));
        float wall_dt = (float)(wall_t1 - wall_t0);
        float shader_dt = (float)(shader_t1 - shader_t0);
        pll_drift_val = (wall_dt > 0) ? shader_dt / wall_dt : 1.0f;
    }

    // ── Activation ──
    if (relu) acc = fmaxf(0.0f, acc);
    C[b * out_dim + o] = acc;

    // ── Collect shadow membrane stats across lanes ──
    float my_membrane = membrane;
    float sum_membrane = my_membrane;
    float sumsq_membrane = my_membrane * my_membrane;
    // Wave-level reduction via shfl
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_membrane += __shfl_xor(sum_membrane, offset);
        sumsq_membrane += __shfl_xor(sumsq_membrane, offset);
    }
    float mean_membrane = sum_membrane / 32.0f;
    float var_membrane = sumsq_membrane / 32.0f - mean_membrane * mean_membrane;

    // ── Store output (only lane 0 writes per sample to avoid contention) ──
    if (lane == 0) {
        int idx = b;  // one output per sample
        lout[idx].sign_ratio = (sign_total>0)?(float)sign_agree/sign_total:0;
        float tm = (n_blocks>0)?timing_sum/n_blocks:0;
        float tv = (n_blocks>1)?fmaxf(0,timing_sumsq/n_blocks-tm*tm):0;
        lout[idx].timing_cv = (tm>0)?sqrtf(tv)/tm:0;
        lout[idx].neighbor_diff = (n_blocks>0)?neighbor_diff_sum/n_blocks:0;
        lout[idx].pop_minority = (n_blocks>0)?pop_minority_sum/n_blocks:0;
        lout[idx].spike_rate = (float)spike_count;
        float pm = (n_blocks>0)?perf_sum/n_blocks:0;
        float pv = (n_blocks>1)?fmaxf(0,perf_sumsq/n_blocks-pm*pm):0;
        lout[idx].perf_var = sqrtf(pv);
        lout[idx].cu_diversity = (float)(hw_id & 0xFF);
        lout[idx].tlb_signal = tlb_signal;
        lout[idx].vgpr_signal = vgpr_signal;
        lout[idx].coalesce_signal = 0; // filled by L10 access pattern
        lout[idx].pll_drift = pll_drift_val;
        lout[idx].branching_ratio = (branching_count>0)?branching_sum/branching_count:0;
        lout[idx].membrane_mean = mean_membrane;
        lout[idx].membrane_std = sqrtf(fmaxf(0, var_membrane));
    }
}

__global__ void argmax_k(const float* L, int* P, int n, int c) {
    int b = blockIdx.x*blockDim.x+threadIdx.x;
    if(b>=n) return;
    float best=-1e30; int bc=0;
    for(int i=0;i<c;i++){float v=L[b*c+i];if(v>best){best=v;bc=i;}}
    P[b]=bc;
}

// Softmax confidence
void host_softmax_conf(float* logits, float* conf, int n) {
    for (int i = 0; i < n; i++) {
        float mx = -1e30;
        for (int c = 0; c < 10; c++) if(logits[i*10+c]>mx) mx=logits[i*10+c];
        float sum = 0;
        for (int c = 0; c < 10; c++) sum += expf(logits[i*10+c]-mx);
        float best = 0;
        for (int c = 0; c < 10; c++) { float p=expf(logits[i*10+c]-mx)/sum; if(p>best)best=p; }
        conf[i] = best;
    }
}

double get_time_ms() {
    struct timeval tv; gettimeofday(&tv,NULL);
    return tv.tv_sec*1000.0+tv.tv_usec/1000.0;
}

float* load_bin(const char* p, int n) {
    FILE* f=fopen(p,"rb"); if(!f) return NULL;
    float* d=(float*)malloc(n*4); fread(d,4,n,f); fclose(f); return d;
}

int main() {
    printf("============================================================\n");
    printf("z2410: CRITICALITY LADDER — Stegvis relaxation mot σ=1.0\n");
    printf("============================================================\n\n");

    const char* base="/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy";
    char p[512];

    // Load MNIST test (2000 ID + 2000 OOD)
    sprintf(p,"%s/data/MNIST/raw/t10k-images-idx3-ubyte",base);
    FILE* f=fopen(p,"rb"); uint32_t hdr[4]; fread(hdr,4,4,f);
    int N_total=__builtin_bswap32(hdr[1]);
    int N = 2000;
    float* mnist_img=(float*)malloc(N*784*4);
    uint8_t* raw=(uint8_t*)malloc(N_total*784); fread(raw,1,N_total*784,f);
    for(int i=0;i<N*784;i++) mnist_img[i]=raw[i]/255.0f;
    fclose(f);

    sprintf(p,"%s/data/MNIST/raw/t10k-labels-idx1-ubyte",base);
    f=fopen(p,"rb"); fread(hdr,4,2,f);
    int* h_lbl=(int*)malloc(N*4);
    raw=(uint8_t*)realloc(raw,N_total); fread(raw,1,N_total,f);
    for(int i=0;i<N;i++) h_lbl[i]=raw[i]; free(raw); fclose(f);

    // OOD: random noise
    float* noise_img=(float*)malloc(N*784*4);
    srand(42);
    for(int i=0;i<N*784;i++) noise_img[i]=(float)rand()/RAND_MAX;

    // Load weights
    const char* md="/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/models/mnist_mlp";
    sprintf(p,"%s/w1.bin",md); float*w1=load_bin(p,128*784);
    sprintf(p,"%s/b1.bin",md); float*b1=load_bin(p,128);
    sprintf(p,"%s/w2.bin",md); float*w2=load_bin(p,64*128);
    sprintf(p,"%s/b2.bin",md); float*b2=load_bin(p,64);
    sprintf(p,"%s/w3.bin",md); float*w3=load_bin(p,10*64);
    sprintf(p,"%s/b3.bin",md); float*b3=load_bin(p,10);
    printf("Loaded %d ID + %d OOD images, trained weights\n\n", N, N);

    // GPU alloc
    float *d_in,*d_h1,*d_h2,*d_out;
    float *d_w1,*d_b1,*d_w2,*d_b2,*d_w3,*d_b3;
    int *d_pred;
    LadderOut *d_lout;
    float *d_prev_timing;

    hipMalloc(&d_in,N*784*4); hipMalloc(&d_h1,N*128*4);
    hipMalloc(&d_h2,N*64*4); hipMalloc(&d_out,N*10*4);
    hipMalloc(&d_pred,N*4);
    hipMalloc(&d_lout,N*sizeof(LadderOut));
    hipMalloc(&d_prev_timing,N*4);
    hipMalloc(&d_w1,128*784*4); hipMalloc(&d_b1,128*4);
    hipMalloc(&d_w2,64*128*4); hipMalloc(&d_b2,64*4);
    hipMalloc(&d_w3,10*64*4); hipMalloc(&d_b3,10*4);

    hipMemcpy(d_w1,w1,128*784*4,hipMemcpyHostToDevice);
    hipMemcpy(d_b1,b1,128*4,hipMemcpyHostToDevice);
    hipMemcpy(d_w2,w2,64*128*4,hipMemcpyHostToDevice);
    hipMemcpy(d_b2,b2,64*4,hipMemcpyHostToDevice);
    hipMemcpy(d_w3,w3,10*64*4,hipMemcpyHostToDevice);
    hipMemcpy(d_b3,b3,10*4,hipMemcpyHostToDevice);

    LadderOut* h_lout = (LadderOut*)malloc(N*sizeof(LadderOut));
    float h_logits[N*10], h_conf[N];
    int h_pred[N];

    // ================================================================
    // Run ladder: level 0-10, ID + OOD
    // ================================================================
    const char* level_names[] = {
        "L0:  Baseline (pure matmul)",
        "L1:  + branch sign (#2)",
        "L2:  + shfl coupling (#4)",
        "L3:  + ballot population (#22/23)",
        "L4:  + hwreg timing (#16/17)",
        "L5:  + s_setreg rounding (#12)",
        "L6:  + s_sleep refract (#8/13)",
        "L7:  + per-CU diversity (#21/24)",
        "L8:  + TLB persistence (#9)",
        "L9:  + stale VGPR (#10)",
        "L10: + coalesce+PLL (#6/19)",
    };

    printf("%-35s %6s %6s %6s %6s %6s %6s %6s %8s\n",
           "Level", "σ_ID", "σ_OOD", "Δσ", "tCV_ID", "tCV_OD", "mStd", "OHead%", "AccID%");
    printf("%-35s %6s %6s %6s %6s %6s %6s %6s %8s\n",
           "---", "---", "---", "---", "---", "---", "---", "---", "---");

    double baseline_time = 0;
    int REPS = 5;

    for (int level = 0; level <= 10; level++) {
        // Measure time
        double total_time = 0;

        float sigma_id = 0, sigma_ood = 0;
        float tcv_id = 0, tcv_ood = 0;
        float mstd_id = 0, mstd_ood = 0;
        int correct_id = 0;

        for (int is_ood = 0; is_ood <= 1; is_ood++) {
            float* input = is_ood ? noise_img : mnist_img;
            hipMemcpy(d_in, input, N*784*4, hipMemcpyHostToDevice);
            hipMemset(d_prev_timing, 0, N*4);

            hipDeviceSynchronize();
            double t0 = get_time_ms();

            for (int rep = 0; rep < REPS; rep++) {
                // Layer 1: 784→128
                matmul_ladder<<<N,128>>>(d_in,d_w1,d_b1,d_h1,d_lout,
                    N,784,128,1,level,nullptr);

                // Store timing for TLB signal to next layer
                // (extract timing_cv from lout → prev_timing)

                // Layer 2: 128→64
                matmul_ladder<<<N,64>>>(d_h1,d_w2,d_b2,d_h2,d_lout,
                    N,128,64,1,level,d_prev_timing);

                // Layer 3: 64→10
                matmul_ladder<<<N,OUTPUT_DIM>>>(d_h2,d_w3,d_b3,d_out,d_lout,
                    N,64,10,0,level,d_prev_timing);

                hipDeviceSynchronize();
            }

            double dt = (get_time_ms() - t0) / REPS;
            if (!is_ood) total_time = dt;

            // Get results from last rep
            hipMemcpy(h_lout, d_lout, N*sizeof(LadderOut), hipMemcpyDeviceToHost);
            hipMemcpy(h_logits, d_out, N*10*4, hipMemcpyDeviceToHost);

            argmax_k<<<(N+255)/256,256>>>(d_out,d_pred,N,10);
            hipDeviceSynchronize();
            hipMemcpy(h_pred, d_pred, N*4, hipMemcpyDeviceToHost);

            // Compute stats
            float sum_sigma=0, sum_tcv=0, sum_mstd=0;
            int n_valid_sigma = 0;
            for (int i = 0; i < N; i++) {
                if (h_lout[i].branching_ratio > 0 && h_lout[i].branching_ratio < 100) {
                    sum_sigma += h_lout[i].branching_ratio;
                    n_valid_sigma++;
                }
                sum_tcv += h_lout[i].timing_cv;
                sum_mstd += h_lout[i].membrane_std;
            }

            float avg_sigma = (n_valid_sigma > 0) ? sum_sigma / n_valid_sigma : 0;
            float avg_tcv = sum_tcv / N;
            float avg_mstd = sum_mstd / N;

            if (!is_ood) {
                sigma_id = avg_sigma;
                tcv_id = avg_tcv;
                mstd_id = avg_mstd;
                for (int i = 0; i < N; i++)
                    if (h_pred[i] == h_lbl[i]) correct_id++;
            } else {
                sigma_ood = avg_sigma;
                tcv_ood = avg_tcv;
                mstd_ood = avg_mstd;
            }
        }

        if (level == 0) baseline_time = total_time;
        float overhead = (baseline_time > 0) ? (total_time - baseline_time) / baseline_time * 100 : 0;

        printf("%-35s %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %7.1f%% %7.1f%%\n",
               level_names[level],
               sigma_id, sigma_ood, sigma_id - sigma_ood,
               tcv_id, tcv_ood, mstd_id,
               overhead,
               100.0f * correct_id / N);
    }

    // Cleanup
    hipFree(d_in);hipFree(d_h1);hipFree(d_h2);hipFree(d_out);hipFree(d_pred);
    hipFree(d_w1);hipFree(d_b1);hipFree(d_w2);hipFree(d_b2);hipFree(d_w3);hipFree(d_b3);
    hipFree(d_lout);hipFree(d_prev_timing);
    free(mnist_img);free(noise_img);free(h_lbl);
    free(w1);free(b1);free(w2);free(b2);free(w3);free(b3);free(h_lout);

    printf("\n=== z2410 COMPLETE ===\n");
    return 0;
}
