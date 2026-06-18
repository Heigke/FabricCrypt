/*  z2277_gpu_bridge_kern.hip — Slim HIP wrapper for real GPU→FPGA bridge
 *
 *  Runs the z2268 fourpop reservoir kernel and outputs raw sampled states
 *  for Python to use in the cross-substrate bridge.
 *
 *  Interface: reads binary input from file, writes binary states to file
 *    Input:  argv[1] = input file  (N_STEPS float32 values)
 *            argv[2] = output file (SAMPLE_N × N_STEPS float32 values, row-major)
 *            argv[3] = N_STEPS (int)
 *
 *  Compile: hipcc --offload-arch=gfx1100 -O1 -o scripts/z2277_gpu_bridge_kern scripts/z2277_gpu_bridge_kern.hip
 *  Run:     HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/z2277_gpu_bridge_kern input.bin output.bin 800
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define N_CUS 12
#define POP_SIZE 64
#define N_POPS 4
#define FOURPOP_THREADS (N_POPS * POP_SIZE)  // 256
#define FOURPOP_NEURONS (N_CUS * FOURPOP_THREADS)  // 3072
#define LDS_TOTAL 320

// Sample 128 per population = 512 total
#define SAMPLE_PER_POP 128
#define SAMPLE_N (SAMPLE_PER_POP * N_POPS)  // 512

#define MODE_ALL 0

// wall_clock64() is already provided by HIP runtime on ROCm 7.1+

// ═══════════════════════════════════════════════════════════════════════
// FOUR-POPULATION RESERVOIR KERNEL (from z2268, verbatim)
// ═══════════════════════════════════════════════════════════════════════
__global__ void fourpop_reservoir(
    float* state_out,
    volatile float* cu_xchg,
    const float* input_seq,
    int steps
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid >= N_CUS || tid >= FOURPOP_THREADS) return;
    int nid = bid * FOURPOP_THREADS + tid;

    int pop = tid / POP_SIZE;
    int pop_tid = tid % POP_SIZE;

    __shared__ float lds[LDS_TOTAL];
    lds[tid] = 0.0f;
    if (tid < LDS_TOTAL - FOURPOP_THREADS)
        lds[FOURPOP_THREADS + tid] = 0.0f;
    __syncthreads();

    float v = 0.0f, h = 0.0f, slow = 0.0f;
    uint32_t rng = nid * 2654435761u + 54321;

    rng *= 1103515245; float leak_base = 0.05f + 0.15f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float input_w = 0.05f + 0.20f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float thr = 0.4f + 0.5f * ((rng>>16)&0xFF)/255.0f;
    rng *= 1103515245; float bias = 0.02f * ((float)((rng>>16)&0xFF)/255.0f - 0.5f);

    int lds_base = pop * POP_SIZE;
    rng *= 1103515245; int ln1 = lds_base + (pop_tid + 1 + ((rng>>8)&7)) % POP_SIZE;
    rng *= 1103515245; int ln2 = lds_base + (pop_tid + POP_SIZE - 1 - ((rng>>8)&7)) % POP_SIZE;
    rng *= 1103515245;
    float wl = (pop == 2) ? (0.04f + 0.08f * ((rng>>16)&0xFF)/255.0f)
                           : (0.02f + 0.06f * ((rng>>16)&0xFF)/255.0f);

    rng *= 1103515245; int rcu = ((rng>>8)&0xFF) % N_CUS;
    rng *= 1103515245; int rtid = ((rng>>8)&0xFF) % FOURPOP_THREADS;
    rng *= 1103515245; float wr = 0.01f + 0.03f * ((rng>>16)&0xFF)/255.0f;

    int c_ln3 = lds_base + (pop_tid + 3 + ((rng>>8)&15)) % POP_SIZE; rng *= 1103515245;
    int c_ln4 = lds_base + (pop_tid + POP_SIZE - 3 - ((rng>>8)&15)) % POP_SIZE; rng *= 1103515245;
    int c_ln5 = lds_base + (pop_tid + 7 + ((rng>>8)&15)) % POP_SIZE; rng *= 1103515245;
    int c_ln6 = lds_base + (pop_tid + POP_SIZE - 7 - ((rng>>8)&15)) % POP_SIZE; rng *= 1103515245;
    float wl2 = 0.03f + 0.05f * ((rng>>16)&0xFF)/255.0f; rng *= 1103515245;

    int c_global1 = (0 * POP_SIZE) + (pop_tid + ((rng>>8)&0x3F)) % POP_SIZE; rng *= 1103515245;
    int c_global2 = (1 * POP_SIZE) + (pop_tid + ((rng>>8)&0x3F)) % POP_SIZE; rng *= 1103515245;
    int c_global3 = (3 * POP_SIZE) + (pop_tid + ((rng>>8)&0x3F)) % POP_SIZE; rng *= 1103515245;
    float wg = 0.015f + 0.025f * ((rng>>16)&0xFF)/255.0f; rng *= 1103515245;

    int xpop_tid1 = ((pop + 1) % N_POPS) * POP_SIZE + (pop_tid + ((rng>>8)&0x1F)) % POP_SIZE; rng *= 1103515245;
    int xpop_tid2 = ((pop + 2) % N_POPS) * POP_SIZE + (pop_tid + ((rng>>8)&0x1F)) % POP_SIZE; rng *= 1103515245;
    int xpop_tid3 = ((pop + 3) % N_POPS) * POP_SIZE + (pop_tid + ((rng>>8)&0x1F)) % POP_SIZE; rng *= 1103515245;

    float bthr = 0.5f + 0.3f * ((nid & 0xF) / 15.0f);
    int my_wavefront = tid / 32;
    int other_wf_tid = ((my_wavefront + 1) % (FOURPOP_THREADS / 32)) * 32 + (tid % 32);
    int d_other_lds = 192 + (other_wf_tid % POP_SIZE);

    float dummy = 1.0001f + nid * 0.0001f;

    float c_leak = leak_base;
    if (pop == 1) c_leak = 0.03f + 0.05f * ((nid & 0xFF) / 255.0f);
    if (pop == 2) c_leak = 0.05f + 0.05f * ((nid & 0xFF) / 255.0f);
    float c_sr = 1.05f;
    float c_temp = 0.65f;

    for (int t = 0; t < steps; t++) {
        float u = input_seq[t];
        float hw_signal = 0.0f;

        // Pop A: Branch divergence
        if (pop == 0) {
            float branch_val = 0.0f;
            if (v > bthr) {
                for (int i = 0; i < 20; i++)
                    dummy = __fmaf_rn(dummy, 1.000001f, 0.0001f);
                branch_val = 1.0f;
            } else {
                uint32_t iv = __float_as_uint(dummy);
                iv = iv * 2654435761u + 1;
                dummy = __uint_as_float((iv & 0x007FFFFF) | 0x3F800000);
                branch_val = -1.0f;
            }
            hw_signal = 0.04f * branch_val;
        }

        // Pop B: L1 bank conflicts + branch + product nonlinearity
        if (pop == 1) {
            float branch_val = 0.0f;
            if (u > 0.0f) {
                for (int i = 0; i < 20; i++)
                    dummy = __fmaf_rn(dummy, 0.99f, 0.01f);
                branch_val = 1.0f;
            } else {
                int iv = (int)(v * 1000.0f);
                dummy = (float)(iv) * 0.001f + dummy * 0.0001f;
                branch_val = -1.0f;
            }
            unsigned long long l1_t0 = __clock64();
            int l1_iters = 5 + (int)(10.0f * fabsf(u));
            int scratch_base = 256;
            for (int i = 0; i < l1_iters; i++)
                lds[scratch_base + (pop_tid + i * ((int)(u * 17.0f) & 0xF)) % POP_SIZE] += 0.0001f;
            unsigned long long l1_t1 = __clock64();
            float l1_signal = fminf(fmaxf(((float)(l1_t1 - l1_t0) - 30.0f) / 20.0f, -2.0f), 2.0f);
            float sign_u = (u > 0.0f ? 1.0f : -1.0f);
            float product_term = v * sign_u;
            float temporal_xor = h * sign_u;
            float deep_xor = slow * sign_u;
            hw_signal = 0.06f * branch_val + 0.005f * l1_signal
                      + 0.08f * product_term + 0.05f * temporal_xor + 0.02f * deep_xor;
        }

        // Pop D: Wavefront scheduling jitter
        float wf_coupling = 0.0f;
        if (pop == 3) {
            int work_amount = 10 + my_wavefront * 5 + (int)(10.0f * fabsf(u));
            unsigned long long wf_t0 = __clock64();
            for (int i = 0; i < work_amount; i++)
                dummy = __fmaf_rn(dummy, 1.000001f, 0.0001f);
            unsigned long long wf_t1 = __clock64();
            float wf_signal = fminf(fmaxf(((float)(wf_t1 - wf_t0) - 50.0f) / 30.0f, -2.0f), 2.0f);
            wf_coupling = lds[d_other_lds];
            hw_signal = 0.005f * wf_signal + 0.003f * wf_coupling;
        }

        __syncthreads();

        // Recurrence
        int my_lds_slot = pop * POP_SIZE + pop_tid;
        lds[my_lds_slot] = v;
        __syncthreads();

        float lrec = wl * (lds[ln1] + lds[ln2]);
        float dense_rec = 0.0f;
        if (pop == 2) {
            dense_rec = wl2 * c_sr * (lds[c_ln3] + lds[c_ln4] + lds[c_ln5] + lds[c_ln6]);
        }
        float global_rec = 0.0f;
        if (pop == 2) {
            global_rec = wg * (lds[c_global1] + lds[c_global2] + lds[c_global3]);
        }
        float xpop_w = (pop == 2) ? 0.015f : 0.003f;
        float xpop = xpop_w * (lds[xpop_tid1] + lds[xpop_tid2] + lds[xpop_tid3]);

        cu_xchg[nid] = v; __threadfence();
        float rrec = wr * cu_xchg[rcu * FOURPOP_THREADS + rtid];

        uint64_t sc = __clock64(); uint64_t wc = wall_clock64();
        float pll = (float)((sc - wc) & 0x1FF) / 256.0f - 1.0f;

        float c_input_boost = (pop == 2) ? 1.2f : 1.0f;
        float pre_act = (1.0f - c_leak) * v
                      + c_input_boost * input_w * u
                      + lrec + rrec + dense_rec + global_rec
                      + xpop + bias + hw_signal + 0.005f * pll;

        if (pop == 2) {
            v = tanhf(pre_act / c_temp);
        } else {
            v = tanhf(pre_act);
        }
        if (pop != 2 && v > thr) { v -= thr; }

        h = 0.93f * h + 0.07f * v;
        slow = 0.99f * slow + 0.01f * v;

        float output;
        if (pop == 1) {
            float sign_u_out = (u > 0.0f ? 1.0f : -1.0f);
            output = v + 0.3f*h + 0.1f*slow + 0.2f*(v * sign_u_out);
        } else if (pop == 2) {
            output = v + 0.3f*h + 0.1f*slow + 0.15f*v*v;
        } else {
            output = v + 0.3f*h + 0.1f*slow;
        }
        state_out[nid * steps + t] = output;
        __syncthreads();
    }
    if (dummy == -9999.0f) state_out[0] = dummy;
}


// ═══════════════════════════════════════════════════════════════════════
// MAIN: read input → run kernel → sample & write output
// ═══════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input.bin> <output.bin> <n_steps>\n", argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];
    int steps = atoi(argv[3]);
    if (steps < 10 || steps > 100000) {
        fprintf(stderr, "Invalid steps: %d\n", steps);
        return 1;
    }

    // Read input sequence
    FILE* fin = fopen(input_path, "rb");
    if (!fin) { fprintf(stderr, "Cannot open %s\n", input_path); return 1; }
    float* h_input = (float*)malloc(steps * sizeof(float));
    size_t nread = fread(h_input, sizeof(float), steps, fin);
    fclose(fin);
    if ((int)nread != steps) {
        fprintf(stderr, "Read %zu floats, expected %d\n", nread, steps);
        return 1;
    }

    // GPU setup
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    fprintf(stderr, "GPU: %s, CUs: %d, neurons: %d\n", prop.name, prop.multiProcessorCount, FOURPOP_NEURONS);

    float *d_state, *d_input;
    volatile float *d_xchg;
    long long state_bytes = (long long)FOURPOP_NEURONS * steps * sizeof(float);
    HIP_CHECK(hipMalloc(&d_state, state_bytes));
    HIP_CHECK(hipMalloc(&d_input, steps * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_xchg, FOURPOP_NEURONS * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_input, h_input, steps * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_state, 0, state_bytes));
    HIP_CHECK(hipMemset((void*)d_xchg, 0, FOURPOP_NEURONS * sizeof(float)));

    // Launch kernel
    fourpop_reservoir<<<N_CUS, FOURPOP_THREADS>>>(d_state, d_xchg, d_input, steps);
    HIP_CHECK(hipDeviceSynchronize());

    // Copy states back
    float* h_state = (float*)malloc(state_bytes);
    HIP_CHECK(hipMemcpy(h_state, d_state, state_bytes, hipMemcpyDeviceToHost));

    // Sample 512 neurons (128 per pop, strided across CUs)
    // Output layout: SAMPLE_N rows × steps cols, row-major
    // Each row = one sampled neuron's time series
    float* h_sampled = (float*)malloc((long long)SAMPLE_N * steps * sizeof(float));

    for (int p = 0; p < N_POPS; p++) {
        int stride = (N_CUS * POP_SIZE) / SAMPLE_PER_POP;
        if (stride < 1) stride = 1;
        for (int si = 0; si < SAMPLE_PER_POP; si++) {
            int pop_idx = si * stride;
            if (pop_idx >= N_CUS * POP_SIZE) pop_idx = N_CUS * POP_SIZE - 1;
            int cu = pop_idx / POP_SIZE;
            int ptid = pop_idx % POP_SIZE;
            int nid = cu * FOURPOP_THREADS + p * POP_SIZE + ptid;
            int out_row = p * SAMPLE_PER_POP + si;
            for (int t = 0; t < steps; t++) {
                float v = h_state[(long long)nid * steps + t];
                if (v != v || v > 1e20f || v < -1e20f) v = 0.0f;
                h_sampled[(long long)out_row * steps + t] = v;
            }
        }
    }

    // Write sampled states
    FILE* fout = fopen(output_path, "wb");
    if (!fout) { fprintf(stderr, "Cannot open %s for writing\n", output_path); return 1; }
    fwrite(h_sampled, sizeof(float), (long long)SAMPLE_N * steps, fout);
    fclose(fout);

    fprintf(stderr, "Wrote %d × %d states to %s\n", SAMPLE_N, steps, output_path);

    // Cleanup
    free(h_input); free(h_state); free(h_sampled);
    hipFree(d_state); hipFree(d_input); hipFree((void*)d_xchg);
    return 0;
}
