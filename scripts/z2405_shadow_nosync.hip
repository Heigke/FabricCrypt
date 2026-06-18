/*
 * z2405_shadow_nosync.hip — Shadow layer WITHOUT __syncthreads
 *
 * z2400: 0% overhead, 7% signal (5 mechanisms)
 * z2403: 130% overhead, 25.6% signal (16 mechanisms + syncthreads)
 * z2405: TARGET 0% overhead, >20% signal (12 mechanisms, NO syncthreads)
 *
 * Rules: ONLY use operations that don't block VALU pipeline:
 *   OK:  s_getreg, s_setreg, __ballot, __popcll, __shfl (SALU/comparison unit)
 *   OK:  branch divergence (already happens), sinf() timing (already called)
 *   NO:  __syncthreads, LDS store+load patterns, atomicAdd to shared
 *
 * Mechanisms used (12):
 *   #2  branch divergence sign pattern (free — just a comparison)
 *   #4  __shfl neighbor exchange (1 cycle, no sync needed)
 *   #8  s_sleep refractory (1 cycle SALU)
 *   #12 s_setreg MODE rounding feedback (1 cycle SALU)
 *   #13 TRAPSTS[0:1] 2-bit state (1 cycle SALU)
 *   #16 hwreg(29) pipeline timing (1 cycle SALU)
 *   #17 hwreg(27) perf snapshot (1 cycle SALU)
 *   #18 sinf() timing jitter (already computed if needed)
 *   #21 per-CU ΔVth via hwreg(23) (1 cycle SALU, once)
 *   #22 __ballot population coding (1 cycle comparison unit)
 *   #23 __popcll spike count (1 cycle SALU)
 *   #24 hwreg(23) CU identity (1 cycle SALU, once)
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
#define N_SHADOW_SIGNALS 8

// Compact shadow output — just the useful signals
struct Shadow {
    float sign_ratio;      // fraction of sign-agreeing blocks
    float timing_cv;       // coefficient of variation of timing
    float neighbor_diff;   // mean absolute difference with shfl neighbors
    float pop_minority;    // mean population minority fraction
    uint32_t spike_count;  // shadow neuron spike count
    uint32_t hw_id;        // CU identity
    float perf_entropy;    // entropy proxy from hwreg(27)
    uint32_t trapsts_parity; // parity of spike count (TRAPSTS bit 0)
};

/* ============================================================
 * Baseline: plain matmul
 * ============================================================ */
__global__ void matmul_plain(
    const float* __restrict__ A, const float* __restrict__ W,
    const float* __restrict__ bias, float* __restrict__ C,
    int batch, int in_dim, int out_dim, int relu)
{
    int b = blockIdx.x, o = threadIdx.x;
    if (b >= batch || o >= out_dim) return;
    float acc = bias[o];
    for (int k = 0; k < in_dim; k++)
        acc += W[o * in_dim + k] * A[b * in_dim + k];
    if (relu) acc = fmaxf(0.0f, acc);
    C[b * out_dim + o] = acc;
}

/* ============================================================
 * Shadow matmul — NO syncthreads, ONLY SALU + ballot + shfl
 * ============================================================ */
__global__ void matmul_shadow_nosync(
    const float* __restrict__ A, const float* __restrict__ W,
    const float* __restrict__ bias, float* __restrict__ C,
    Shadow* __restrict__ shadow,
    int batch, int in_dim, int out_dim, int relu,
    int enable_feedback)
{
    int b = blockIdx.x, o = threadIdx.x;
    int lane = o % 32;
    if (b >= batch || o >= out_dim) return;

    // #24/#21: CU identity (once, SALU)
    uint32_t hw_id;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw_id));

    // Save MODE for restore
    uint32_t orig_mode;
    if (enable_feedback) {
        asm volatile("s_getreg_b32 %0, hwreg(1)" : "=s"(orig_mode));
    }

    // Shadow neuron state — all in registers, no LDS
    float membrane = 0.0f;
    uint32_t spike_count = 0;
    uint32_t sign_agree_count = 0;
    uint32_t sign_total_blocks = 0;
    float timing_sum = 0.0f;
    float timing_sumsq = 0.0f;
    float neighbor_diff_sum = 0.0f;
    float pop_minority_sum = 0.0f;
    float perf_sum = 0.0f;
    uint32_t n_blocks = 0;

    // Initial timing reference (#16)
    uint32_t t_prev;
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t_prev));

    // MAIN MATMUL LOOP
    float acc = bias[o];

    for (int k = 0; k < in_dim; k += 4) {
        // ═══ VALU: matmul FMA (the actual work) ═══
        float p0 = W[o * in_dim + k] * A[b * in_dim + k];
        float p1 = (k+1 < in_dim) ? W[o*in_dim+k+1] * A[b*in_dim+k+1] : 0.0f;
        float p2 = (k+2 < in_dim) ? W[o*in_dim+k+2] * A[b*in_dim+k+2] : 0.0f;
        float p3 = (k+3 < in_dim) ? W[o*in_dim+k+3] * A[b*in_dim+k+3] : 0.0f;
        acc += p0 + p1 + p2 + p3;

        // ═══ SALU: shadow harvest (free during VALU memory stalls) ═══

        // #2: Sign pattern — how many of 4 products positive?
        int n_pos = (p0 > 0.0f) + (p1 > 0.0f) + (p2 > 0.0f) + (p3 > 0.0f);
        sign_agree_count += (n_pos == 0 || n_pos == 4) ? 1 : 0;
        sign_total_blocks++;

        // #22/#23: Population coding — what fraction of lanes had positive sum?
        float block_sum = p0 + p1 + p2 + p3;
        uint64_t pos_mask = __ballot(block_sum > 0.0f);
        uint32_t n_pop_pos = __popcll(pos_mask);
        uint32_t minority = (n_pop_pos < 16) ? n_pop_pos : (32 - n_pop_pos);
        pop_minority_sum += (float)minority;

        // #4: Neighbor difference via __shfl (no sync needed, wave-internal)
        float left = __shfl(acc, (lane + 31) % 32);
        float right = __shfl(acc, (lane + 1) % 32);
        neighbor_diff_sum += fabsf(acc - left) + fabsf(acc - right);

        // #16: Pipeline timing (SALU, reads cycle counter)
        uint32_t t_now;
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t_now));
        float dt = (float)(t_now - t_prev);
        t_prev = t_now;
        timing_sum += dt;
        timing_sumsq += dt * dt;

        // #17: Perf snapshot entropy
        uint32_t perf;
        asm volatile("s_getreg_b32 %0, hwreg(27)" : "=s"(perf));
        perf_sum += (float)(perf & 0xFF);

        n_blocks++;

        // ═══ Shadow neuron integration (pure register ops) ═══
        float signal = (float)(4 - n_pos) * 0.2f       // sign disagreement
                     + (dt > 25.0f ? 0.4f : 0.0f)       // timing anomaly
                     + (float)minority * 0.05f;           // population minority
        membrane += signal;

        // Spike detection + refractory
        if (membrane > 6.0f) {
            spike_count++;
            membrane = 0.0f;

            // #8: s_sleep refractory (SALU, 40 cycles, wavefront pauses)
            asm volatile("s_sleep 1");

            // #13: Toggle TRAPSTS bit 0 (safe, SALU)
            uint32_t ts;
            asm volatile("s_getreg_b32 %0, hwreg(3)" : "=s"(ts));
            asm volatile("s_setreg_b32 hwreg(3), %0" : : "s"(ts ^ 0x1));
        }

        // #12: s_setreg feedback — adapt rounding based on confidence
        if (enable_feedback && (n_blocks & 0xF) == 0) {
            // Every 16th block: check if computation looks confident
            // High sign agreement → stable → keep RNE
            // Low sign agreement → volatile → switch to RTZ (conservative)
            float agreement_so_far = (float)sign_agree_count / (float)sign_total_blocks;
            if (agreement_so_far < 0.3f) {
                asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x3"); // RTZ
            } else {
                asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x0"); // RNE
            }
        }
    }

    // Restore MODE
    if (enable_feedback) {
        asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(orig_mode));
    }

    // Activation
    if (relu) acc = fmaxf(0.0f, acc);

    // Store matmul result
    C[b * out_dim + o] = acc;

    // Store shadow output — compact, one write
    int idx = b * out_dim + o;
    shadow[idx].sign_ratio = (sign_total_blocks > 0) ?
        (float)sign_agree_count / sign_total_blocks : 0.0f;
    float t_mean = (n_blocks > 0) ? timing_sum / n_blocks : 0.0f;
    float t_var = (n_blocks > 1) ?
        fmaxf(0.0f, timing_sumsq / n_blocks - t_mean * t_mean) : 0.0f;
    shadow[idx].timing_cv = (t_mean > 0.0f) ? sqrtf(t_var) / t_mean : 0.0f;
    shadow[idx].neighbor_diff = (n_blocks > 0) ? neighbor_diff_sum / n_blocks : 0.0f;
    shadow[idx].pop_minority = (n_blocks > 0) ? pop_minority_sum / n_blocks : 0.0f;
    shadow[idx].spike_count = spike_count;
    shadow[idx].hw_id = hw_id;
    shadow[idx].perf_entropy = (n_blocks > 0) ? perf_sum / n_blocks : 0.0f;

    uint32_t final_ts;
    asm volatile("s_getreg_b32 %0, hwreg(3)" : "=s"(final_ts));
    shadow[idx].trapsts_parity = final_ts & 0x1;
}

/* Argmax */
__global__ void argmax_k(const float* L, int* P, int n, int c) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n) return;
    float best = -1e30f; int bc = 0;
    for (int i = 0; i < c; i++) {
        float v = L[b*c+i]; if (v > best) { best = v; bc = i; }
    }
    P[b] = bc;
}

/* ============================================================ */
double get_time_ms() {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

float read_power() {
    for (int h = 7; h <= 10; h++) {
        char p[64]; sprintf(p, "/sys/class/hwmon/hwmon%d/power1_average", h);
        FILE* f = fopen(p, "r");
        if (f) { long v; fscanf(f, "%ld", &v); fclose(f); return v / 1e6f; }
    }
    return -1.0f;
}

float* load_bin(const char* path, int n) {
    FILE* f = fopen(path, "rb"); if (!f) return NULL;
    float* d = (float*)malloc(n*4); fread(d, 4, n, f); fclose(f); return d;
}

int main() {
    printf("============================================================\n");
    printf("z2405: SHADOW NOSYNC (12 mechanisms, zero-sync target)\n");
    printf("============================================================\n\n");

    const char* base = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy";
    char p[512];

    // Load MNIST test
    sprintf(p, "%s/data/MNIST/raw/t10k-images-idx3-ubyte", base);
    FILE* f = fopen(p, "rb"); uint32_t hdr[4]; fread(hdr,4,4,f);
    int N = __builtin_bswap32(hdr[1]);
    float* h_img = (float*)malloc(N*784*4);
    uint8_t* raw = (uint8_t*)malloc(N*784);
    fread(raw, 1, N*784, f);
    for (int i = 0; i < N*784; i++) h_img[i] = raw[i]/255.0f;
    free(raw); fclose(f);

    sprintf(p, "%s/data/MNIST/raw/t10k-labels-idx1-ubyte", base);
    f = fopen(p, "rb"); fread(hdr, 4, 2, f);
    int* h_lbl = (int*)malloc(N*4);
    raw = (uint8_t*)malloc(N); fread(raw, 1, N, f);
    for (int i = 0; i < N; i++) h_lbl[i] = raw[i];
    free(raw); fclose(f);

    // Load weights
    const char* md = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/models/mnist_mlp";
    sprintf(p,"%s/w1.bin",md); float* w1 = load_bin(p, 128*784);
    sprintf(p,"%s/b1.bin",md); float* b1 = load_bin(p, 128);
    sprintf(p,"%s/w2.bin",md); float* w2 = load_bin(p, 64*128);
    sprintf(p,"%s/b2.bin",md); float* b2 = load_bin(p, 64);
    sprintf(p,"%s/w3.bin",md); float* w3 = load_bin(p, 10*64);
    sprintf(p,"%s/b3.bin",md); float* b3 = load_bin(p, 10);
    printf("MNIST: %d images, trained weights loaded\n\n", N);

    int BS = 1000;

    // GPU alloc
    float *d_in, *d_h1, *d_h2, *d_out;
    float *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3;
    int *d_pred; Shadow *d_sh1, *d_sh2, *d_sh3;

    hipMalloc(&d_in, BS*784*4); hipMalloc(&d_h1, BS*128*4);
    hipMalloc(&d_h2, BS*64*4);  hipMalloc(&d_out, BS*10*4);
    hipMalloc(&d_pred, BS*4);
    hipMalloc(&d_w1, 128*784*4); hipMalloc(&d_b1, 128*4);
    hipMalloc(&d_w2, 64*128*4);  hipMalloc(&d_b2, 64*4);
    hipMalloc(&d_w3, 10*64*4);   hipMalloc(&d_b3, 10*4);
    hipMalloc(&d_sh1, BS*128*sizeof(Shadow));
    hipMalloc(&d_sh2, BS*64*sizeof(Shadow));
    hipMalloc(&d_sh3, BS*10*sizeof(Shadow));

    hipMemcpy(d_w1,w1,128*784*4,hipMemcpyHostToDevice);
    hipMemcpy(d_b1,b1,128*4,hipMemcpyHostToDevice);
    hipMemcpy(d_w2,w2,64*128*4,hipMemcpyHostToDevice);
    hipMemcpy(d_b2,b2,64*4,hipMemcpyHostToDevice);
    hipMemcpy(d_w3,w3,10*64*4,hipMemcpyHostToDevice);
    hipMemcpy(d_b3,b3,10*4,hipMemcpyHostToDevice);

    int h_pred[BS];
    Shadow* h_sh3 = (Shadow*)malloc(BS*10*sizeof(Shadow));

    // ================================================================
    // Three modes × 10 repetitions
    // ================================================================
    struct { const char* nm; int sh; int fb; } modes[] = {
        {"A: Baseline", 0, 0},
        {"B: Shadow (no feedback)", 1, 0},
        {"C: Shadow + feedback", 1, 1},
    };

    int REPS = 10;

    for (int m = 0; m < 3; m++) {
        printf("--- %s ---\n", modes[m].nm);

        double t_total = 0; float p_total = 0;
        int correct = 0;

        // Shadow accumulators
        double s_sr_c=0, s_sr_i=0, s_tc_c=0, s_tc_i=0;
        double s_nd_c=0, s_nd_i=0, s_pm_c=0, s_pm_i=0;
        double s_sp_c=0, s_sp_i=0, s_pe_c=0, s_pe_i=0;
        int nc=0, ni=0;

        for (int rep = 0; rep < REPS; rep++) {
            float pw0 = read_power();
            double t0 = get_time_ms();

            int rep_correct = 0;
            for (int batch = 0; batch < N/BS; batch++) {
                hipMemcpy(d_in, h_img+batch*BS*784, BS*784*4, hipMemcpyHostToDevice);

                if (!modes[m].sh) {
                    matmul_plain<<<BS,128>>>(d_in,d_w1,d_b1,d_h1,BS,784,128,1);
                    matmul_plain<<<BS,64>>>(d_h1,d_w2,d_b2,d_h2,BS,128,64,1);
                    matmul_plain<<<BS,10>>>(d_h2,d_w3,d_b3,d_out,BS,64,10,0);
                } else {
                    matmul_shadow_nosync<<<BS,128>>>(d_in,d_w1,d_b1,d_h1,d_sh1,BS,784,128,1,modes[m].fb);
                    matmul_shadow_nosync<<<BS,64>>>(d_h1,d_w2,d_b2,d_h2,d_sh2,BS,128,64,1,modes[m].fb);
                    matmul_shadow_nosync<<<BS,10>>>(d_h2,d_w3,d_b3,d_out,d_sh3,BS,64,10,0,modes[m].fb);
                }

                argmax_k<<<(BS+255)/256,256>>>(d_out,d_pred,BS,10);
                hipDeviceSynchronize();
                hipMemcpy(h_pred, d_pred, BS*4, hipMemcpyDeviceToHost);

                for (int i = 0; i < BS; i++)
                    if (h_pred[i] == h_lbl[batch*BS+i]) rep_correct++;

                // Shadow analysis (last rep)
                if (modes[m].sh && rep == REPS-1) {
                    hipMemcpy(h_sh3, d_sh3, BS*10*sizeof(Shadow), hipMemcpyDeviceToHost);
                    for (int i = 0; i < BS; i++) {
                        int ok = (h_pred[i] == h_lbl[batch*BS+i]);
                        float sr=0,tc=0,nd=0,pm=0,sp=0,pe=0;
                        for (int c = 0; c < 10; c++) {
                            int x = i*10+c;
                            sr += h_sh3[x].sign_ratio;
                            tc += h_sh3[x].timing_cv;
                            nd += h_sh3[x].neighbor_diff;
                            pm += h_sh3[x].pop_minority;
                            sp += h_sh3[x].spike_count;
                            pe += h_sh3[x].perf_entropy;
                        }
                        sr/=10; tc/=10; nd/=10; pm/=10; sp/=10; pe/=10;
                        if (ok) { s_sr_c+=sr; s_tc_c+=tc; s_nd_c+=nd; s_pm_c+=pm; s_sp_c+=sp; s_pe_c+=pe; nc++; }
                        else    { s_sr_i+=sr; s_tc_i+=tc; s_nd_i+=nd; s_pm_i+=pm; s_sp_i+=sp; s_pe_i+=pe; ni++; }
                    }
                }
            }

            double t1 = get_time_ms();
            t_total += t1 - t0;
            p_total += (pw0 + read_power()) / 2.0f;
            if (rep == REPS-1) correct = rep_correct;
        }

        double t_avg = t_total / REPS;
        float p_avg = p_total / REPS;

        printf("  Accuracy:    %.2f%%\n", 100.0*correct/N);
        printf("  Time (avg):  %.2f ms  (%.0f samples/s)\n", t_avg, N/(t_avg/1000));
        printf("  Power:       %.1f W\n", p_avg);
        printf("  Energy:      %.4f J\n", p_avg * t_avg / 1000);

        if (modes[m].sh && nc > 0 && ni > 0) {
            printf("\n  Shadow signals (correct vs incorrect):\n");
            printf("  %-18s %9s %9s %7s  Direction\n", "Signal", "Correct", "Incorr.", "Ratio");

            #define PR(name, vc, vi) { \
                double rc = vc/nc, ri = vi/ni, ratio = rc/(ri+1e-30); \
                const char* dir = (ratio > 1.01) ? "correct HIGHER" : \
                                  (ratio < 0.99) ? "incorrect HIGHER" : "~equal"; \
                printf("  %-18s %9.4f %9.4f %7.3f  %s\n", name, rc, ri, ratio, dir); \
            }
            PR("Sign ratio", s_sr_c, s_sr_i);
            PR("Timing CV", s_tc_c, s_tc_i);
            PR("Neighbor diff", s_nd_c, s_nd_i);
            PR("Pop minority", s_pm_c, s_pm_i);
            PR("Spike count", s_sp_c, s_sp_i);
            PR("Perf entropy", s_pe_c, s_pe_i);
            printf("  (n_correct=%d, n_incorrect=%d)\n", nc, ni);
        }
        printf("\n");
    }

    // Scaling test
    printf("--- SCALING: Overhead vs batch size (10 reps each) ---\n");
    printf("  %-8s %10s %10s %8s\n", "Batch", "Baseline", "Shadow", "Overhead");
    int bsizes[] = {100, 250, 500, 1000, 2000, 5000, 10000};
    for (int bi = 0; bi < 7; bi++) {
        int bs = bsizes[bi];
        if (bs > N) break;
        hipMemcpy(d_in, h_img, bs*784*4, hipMemcpyHostToDevice);
        // Reallocate shadow if needed
        Shadow* d_sh_tmp; hipMalloc(&d_sh_tmp, bs*128*sizeof(Shadow));

        hipDeviceSynchronize();
        double t0 = get_time_ms();
        for (int r = 0; r < 20; r++) {
            matmul_plain<<<bs,128>>>(d_in,d_w1,d_b1,d_h1,bs,784,128,1);
            matmul_plain<<<bs,64>>>(d_h1,d_w2,d_b2,d_h2,bs,128,64,1);
            matmul_plain<<<bs,10>>>(d_h2,d_w3,d_b3,d_out,bs,64,10,0);
            hipDeviceSynchronize();
        }
        double tb = (get_time_ms() - t0) / 20;

        hipDeviceSynchronize();
        t0 = get_time_ms();
        for (int r = 0; r < 20; r++) {
            matmul_shadow_nosync<<<bs,128>>>(d_in,d_w1,d_b1,d_h1,d_sh1,bs,784,128,1,0);
            matmul_shadow_nosync<<<bs,64>>>(d_h1,d_w2,d_b2,d_h2,d_sh2,bs,128,64,1,0);
            matmul_shadow_nosync<<<bs,10>>>(d_h2,d_w3,d_b3,d_out,d_sh3,bs,64,10,0,0);
            hipDeviceSynchronize();
        }
        double ts = (get_time_ms() - t0) / 20;

        printf("  %-8d %8.2f ms %8.2f ms %7.1f%%\n", bs, tb, ts, (ts-tb)/tb*100);
        hipFree(d_sh_tmp);
    }

    hipFree(d_in); hipFree(d_h1); hipFree(d_h2); hipFree(d_out); hipFree(d_pred);
    hipFree(d_w1); hipFree(d_b1); hipFree(d_w2); hipFree(d_b2); hipFree(d_w3); hipFree(d_b3);
    hipFree(d_sh1); hipFree(d_sh2); hipFree(d_sh3);
    free(h_img); free(h_lbl); free(w1); free(b1); free(w2); free(b2); free(w3); free(b3); free(h_sh3);

    printf("\n=== z2405 COMPLETE ===\n");
    return 0;
}
