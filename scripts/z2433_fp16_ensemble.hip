/**
 * z2433: FP16 Rounding Mode Ensemble
 *
 * z2429b showed: FP32 rounding differences are ~20μ — too small to flip predictions.
 * FP16 has 10-bit mantissa (vs FP32's 23-bit) → rounding errors are ~8000× larger.
 * This should create real ensemble diversity at ZERO extra compute cost.
 *
 * Uses _Float16 (__half) arithmetic for the hidden layers,
 * with rounding mode changes via s_setreg affecting the VALU FP16 operations.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

#define N_MODES 4

// Layer kernel in FP16: Y = ReLU(X @ W^T + b) or linear
// Weights stored as FP32, converted to FP16 in kernel
__global__ void layer_fp16(
    const float* __restrict__ X,    // [N, in_dim] FP32
    const float* __restrict__ W,    // [out_dim, in_dim] FP32
    const float* __restrict__ bias, // [out_dim] FP32
    float* __restrict__ Y,          // [N, out_dim] FP32 output
    int N, int in_dim, int out_dim,
    int use_relu, int round_mode
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample = idx / out_dim;
    int neuron = idx % out_dim;
    if (sample >= N) return;

    // Set rounding mode
    if (round_mode == 1) {
        asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x1" ::: "memory");
    } else if (round_mode == 2) {
        asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x2" ::: "memory");
    } else if (round_mode == 3) {
        asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x3" ::: "memory");
    }

    // Accumulate in FP16
    __half sum = __float2half(bias[neuron]);
    const float* w_row = W + neuron * in_dim;
    const float* x_row = X + sample * in_dim;

    for (int k = 0; k < in_dim; k++) {
        __half w = __float2half(w_row[k]);
        __half x = __float2half(x_row[k]);
        sum = __hadd(sum, __hmul(w, x));  // FP16 multiply-add
    }

    float result = __half2float(sum);
    if (use_relu && result < 0.0f) result = 0.0f;
    Y[sample * out_dim + neuron] = result;

    // Reset
    if (round_mode > 0) {
        asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x0" ::: "memory");
    }
}

// FP32 baseline for comparison
__global__ void layer_fp32(
    const float* __restrict__ X,
    const float* __restrict__ W,
    const float* __restrict__ bias,
    float* __restrict__ Y,
    int N, int in_dim, int out_dim,
    int use_relu
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample = idx / out_dim;
    int neuron = idx % out_dim;
    if (sample >= N) return;

    float sum = bias[neuron];
    const float* w_row = W + neuron * in_dim;
    const float* x_row = X + sample * in_dim;
    for (int k = 0; k < in_dim; k++) {
        sum += w_row[k] * x_row[k];
    }
    if (use_relu && sum < 0.0f) sum = 0.0f;
    Y[sample * out_dim + neuron] = sum;
}

float* load_bin(const char* path, int count) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); exit(1); }
    float* data = (float*)malloc(count * sizeof(float));
    fread(data, sizeof(float), count, f);
    fclose(f);
    return data;
}

void run_mlp_fp16(float* d_img, float* d_w1, float* d_b1, float* d_w2, float* d_b2,
                   float* d_w3, float* d_b3, float* d_h1, float* d_h2, float* d_out,
                   int N, int mode) {
    int thr = 256;
    layer_fp16<<<(N*128+thr-1)/thr, thr>>>(d_img, d_w1, d_b1, d_h1, N, 784, 128, 1, mode);
    layer_fp16<<<(N*64+thr-1)/thr, thr>>>(d_h1, d_w2, d_b2, d_h2, N, 128, 64, 1, mode);
    layer_fp16<<<(N*10+thr-1)/thr, thr>>>(d_h2, d_w3, d_b3, d_out, N, 64, 10, 0, mode);
}

void run_mlp_fp32(float* d_img, float* d_w1, float* d_b1, float* d_w2, float* d_b2,
                   float* d_w3, float* d_b3, float* d_h1, float* d_h2, float* d_out, int N) {
    int thr = 256;
    layer_fp32<<<(N*128+thr-1)/thr, thr>>>(d_img, d_w1, d_b1, d_h1, N, 784, 128, 1);
    layer_fp32<<<(N*64+thr-1)/thr, thr>>>(d_h1, d_w2, d_b2, d_h2, N, 128, 64, 1);
    layer_fp32<<<(N*10+thr-1)/thr, thr>>>(d_h2, d_w3, d_b3, d_out, N, 64, 10, 0);
}

int main() {
    printf("============================================================\n");
    printf("z2433: FP16 ROUNDING MODE ENSEMBLE\n");
    printf("============================================================\n\n");

    const char* base = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy";
    char path[512];

    // Load MNIST
    sprintf(path, "%s/data/MNIST/raw/t10k-images-idx3-ubyte", base);
    FILE* fimg = fopen(path, "rb");
    unsigned char hdr[16]; fread(hdr, 1, 16, fimg);
    int N = 10000;
    unsigned char* raw = (unsigned char*)malloc(N * 784);
    fread(raw, 1, N * 784, fimg); fclose(fimg);
    float* h_images = (float*)malloc(N * 784 * sizeof(float));
    for (int i = 0; i < N * 784; i++) h_images[i] = raw[i] / 255.0f;
    free(raw);

    sprintf(path, "%s/data/MNIST/raw/t10k-labels-idx1-ubyte", base);
    FILE* flbl = fopen(path, "rb");
    unsigned char lhdr[8]; fread(lhdr, 1, 8, flbl);
    unsigned char* h_labels = (unsigned char*)malloc(N);
    fread(h_labels, 1, N, flbl); fclose(flbl);

    // Load weights
    sprintf(path, "%s/models/mnist_mlp/w1.bin", base); float* h_w1 = load_bin(path, 128*784);
    sprintf(path, "%s/models/mnist_mlp/b1.bin", base); float* h_b1 = load_bin(path, 128);
    sprintf(path, "%s/models/mnist_mlp/w2.bin", base); float* h_w2 = load_bin(path, 64*128);
    sprintf(path, "%s/models/mnist_mlp/b2.bin", base); float* h_b2 = load_bin(path, 64);
    sprintf(path, "%s/models/mnist_mlp/w3.bin", base); float* h_w3 = load_bin(path, 10*64);
    sprintf(path, "%s/models/mnist_mlp/b3.bin", base); float* h_b3 = load_bin(path, 10);

    float *d_img, *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3, *d_h1, *d_h2, *d_out;
    hipMalloc(&d_img, N*784*4); hipMalloc(&d_w1, 128*784*4); hipMalloc(&d_b1, 128*4);
    hipMalloc(&d_w2, 64*128*4); hipMalloc(&d_b2, 64*4);
    hipMalloc(&d_w3, 10*64*4); hipMalloc(&d_b3, 10*4);
    hipMalloc(&d_h1, N*128*4); hipMalloc(&d_h2, N*64*4); hipMalloc(&d_out, N*10*4);

    hipMemcpy(d_img, h_images, N*784*4, hipMemcpyHostToDevice);
    hipMemcpy(d_w1, h_w1, 128*784*4, hipMemcpyHostToDevice);
    hipMemcpy(d_b1, h_b1, 128*4, hipMemcpyHostToDevice);
    hipMemcpy(d_w2, h_w2, 64*128*4, hipMemcpyHostToDevice);
    hipMemcpy(d_b2, h_b2, 64*4, hipMemcpyHostToDevice);
    hipMemcpy(d_w3, h_w3, 10*64*4, hipMemcpyHostToDevice);
    hipMemcpy(d_b3, h_b3, 10*4, hipMemcpyHostToDevice);

    // FP32 baseline
    printf("--- FP32 Baseline ---\n");
    run_mlp_fp32(d_img, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_h1, d_h2, d_out, N);
    hipDeviceSynchronize();
    float* h_fp32 = (float*)malloc(N*10*4);
    hipMemcpy(h_fp32, d_out, N*10*4, hipMemcpyDeviceToHost);
    int correct_fp32 = 0;
    for (int i = 0; i < N; i++) {
        int pred = 0; float mx = h_fp32[i*10];
        for (int j = 1; j < 10; j++) if (h_fp32[i*10+j] > mx) { mx = h_fp32[i*10+j]; pred = j; }
        if (pred == h_labels[i]) correct_fp32++;
    }
    printf("  FP32: %.2f%%\n\n", 100.0*correct_fp32/N);

    // FP16 per mode
    printf("--- FP16 Per-Mode ---\n");
    float* h_modes[4];
    int correct_mode[4];
    const char* mode_names[] = {"RNE", "R+inf", "R-inf", "RTZ"};

    for (int m = 0; m < 4; m++) {
        h_modes[m] = (float*)malloc(N*10*4);
        run_mlp_fp16(d_img, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_h1, d_h2, d_out, N, m);
        hipDeviceSynchronize();
        hipMemcpy(h_modes[m], d_out, N*10*4, hipMemcpyDeviceToHost);

        correct_mode[m] = 0;
        for (int i = 0; i < N; i++) {
            int pred = 0; float mx = h_modes[m][i*10];
            for (int j = 1; j < 10; j++) if (h_modes[m][i*10+j] > mx) { mx = h_modes[m][i*10+j]; pred = j; }
            if (pred == h_labels[i]) correct_mode[m]++;
        }
        printf("  FP16 Mode %d (%s): %.2f%%\n", m, mode_names[m], 100.0*correct_mode[m]/N);
    }

    // Ensemble
    printf("\n--- FP16 Ensemble ---\n");
    int correct_avg = 0, correct_vote = 0;
    int disagreements = 0, ens_wins = 0, base_wins = 0;

    for (int i = 0; i < N; i++) {
        int preds[4];
        float avg[10] = {0};
        int votes[10] = {0};

        for (int m = 0; m < 4; m++) {
            float* l = h_modes[m] + i*10;
            int pred = 0; float mx = l[0];
            for (int j = 0; j < 10; j++) {
                avg[j] += l[j];
                if (l[j] > mx) { mx = l[j]; pred = j; }
            }
            preds[m] = pred;
            votes[pred]++;
        }

        // Average
        int avg_pred = 0; float avg_mx = avg[0];
        for (int j = 1; j < 10; j++) if (avg[j] > avg_mx) { avg_mx = avg[j]; avg_pred = j; }
        if (avg_pred == h_labels[i]) correct_avg++;

        // Vote
        int vpred = 0; int vmx = votes[0];
        for (int j = 1; j < 10; j++) if (votes[j] > vmx) { vmx = votes[j]; vpred = j; }
        if (vpred == h_labels[i]) correct_vote++;

        // Disagreement
        bool same = true;
        for (int m = 1; m < 4; m++) if (preds[m] != preds[0]) { same = false; break; }
        if (!same) {
            disagreements++;
            if (avg_pred == h_labels[i] && preds[0] != h_labels[i]) ens_wins++;
            if (preds[0] == h_labels[i] && avg_pred != h_labels[i]) base_wins++;
        }
    }

    printf("  Average logits: %.2f%% (Δ=%+.2fpp vs FP16-RNE)\n",
           100.0*correct_avg/N, 100.0*(correct_avg-correct_mode[0])/N);
    printf("  Majority vote:  %.2f%% (Δ=%+.2fpp vs FP16-RNE)\n",
           100.0*correct_vote/N, 100.0*(correct_vote-correct_mode[0])/N);
    printf("\n  Disagreements:  %d/%d (%.1f%%)\n", disagreements, N, 100.0*disagreements/N);
    printf("  Ensemble wins:  %d\n", ens_wins);
    printf("  Base wins:      %d\n", base_wins);
    printf("  Net gain:       %+d samples\n", ens_wins - base_wins);

    // Logit divergence
    double total_diff = 0, max_diff = 0;
    int n_differ = 0;
    for (int i = 0; i < N; i++) {
        for (int m = 1; m < 4; m++) {
            for (int j = 0; j < 10; j++) {
                double d = fabs(h_modes[0][i*10+j] - h_modes[m][i*10+j]);
                if (d > 0.001) n_differ++;
                total_diff += d;
                if (d > max_diff) max_diff = d;
            }
        }
    }
    printf("\n  Logits >0.001 diff: %d/%d (%.1f%%)\n", n_differ, N*10*3, 100.0*n_differ/(N*10*3));
    printf("  Mean |diff|: %.4f\n", total_diff/(N*10*3));
    printf("  Max  |diff|: %.4f\n", max_diff);

    printf("\n============================================================\n");
    double delta_avg = 100.0*(correct_avg - correct_mode[0])/N;
    double delta_fp32 = 100.0*(correct_avg - correct_fp32)/N;
    printf("  FP32 baseline:        %.2f%%\n", 100.0*correct_fp32/N);
    printf("  FP16 single (RNE):    %.2f%%\n", 100.0*correct_mode[0]/N);
    printf("  FP16 ensemble (avg):  %.2f%%\n", 100.0*correct_avg/N);
    printf("  Ensemble vs FP16-RNE: %+.2fpp\n", delta_avg);
    printf("  Ensemble vs FP32:     %+.2fpp\n", delta_fp32);

    if (correct_avg > correct_mode[0]) {
        printf("\n  >>> FP16 ENSEMBLE IMPROVES over single FP16 by +%d samples <<<\n",
               correct_avg - correct_mode[0]);
    }
    if (correct_avg > correct_fp32) {
        printf("  >>> FP16 ENSEMBLE BEATS FP32! <<<\n");
    }
    printf("============================================================\n");

    for (int m = 0; m < 4; m++) free(h_modes[m]);
    free(h_fp32); free(h_images); free(h_labels);
    free(h_w1); free(h_b1); free(h_w2); free(h_b2); free(h_w3); free(h_b3);
    hipFree(d_img); hipFree(d_w1); hipFree(d_b1); hipFree(d_w2); hipFree(d_b2);
    hipFree(d_w3); hipFree(d_b3); hipFree(d_h1); hipFree(d_h2); hipFree(d_out);
    return 0;
}
