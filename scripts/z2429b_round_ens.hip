/**
 * z2429b: Rounding Mode Ensemble — simplified version
 *
 * Process one LAYER at a time (not full MLP per thread) to avoid register spill.
 * Each kernel call processes one layer with one rounding mode.
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <chrono>

// Layer kernel: Y[n,out] = ReLU(X[n,in] @ W[out,in] + b[out])
// One thread computes one (sample, output_neuron) pair
__global__ void layer_kernel(
    const float* __restrict__ X,    // [N, in_dim]
    const float* __restrict__ W,    // [out_dim, in_dim]
    const float* __restrict__ bias, // [out_dim]
    float* __restrict__ Y,          // [N, out_dim]
    int N, int in_dim, int out_dim,
    int use_relu, int round_mode
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample = idx / out_dim;
    int neuron = idx % out_dim;
    if (sample >= N) return;

    // Set rounding mode via s_setreg_imm32_b32
    if (round_mode == 1) {
        asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x1" ::: "memory");
    } else if (round_mode == 2) {
        asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x2" ::: "memory");
    } else if (round_mode == 3) {
        asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x3" ::: "memory");
    }

    float sum = bias[neuron];
    const float* w_row = W + neuron * in_dim;
    const float* x_row = X + sample * in_dim;
    for (int k = 0; k < in_dim; k++) {
        sum += w_row[k] * x_row[k];
    }
    if (use_relu && sum < 0.0f) sum = 0.0f;
    Y[sample * out_dim + neuron] = sum;

    // Reset rounding to RNE
    if (round_mode > 0) {
        asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x0" ::: "memory");
    }
}

float* load_bin(const char* path, int count) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); exit(1); }
    float* data = (float*)malloc(count * sizeof(float));
    fread(data, sizeof(float), count, f);
    fclose(f);
    return data;
}

void run_mlp(float* d_images, float* d_w1, float* d_b1, float* d_w2, float* d_b2,
             float* d_w3, float* d_b3, float* d_h1, float* d_h2, float* d_logits,
             int N, int round_mode) {
    int threads = 256;
    // Layer 1: 784 -> 128 ReLU
    int total1 = N * 128;
    layer_kernel<<<(total1+threads-1)/threads, threads>>>(
        d_images, d_w1, d_b1, d_h1, N, 784, 128, 1, round_mode);
    // Layer 2: 128 -> 64 ReLU
    int total2 = N * 64;
    layer_kernel<<<(total2+threads-1)/threads, threads>>>(
        d_h1, d_w2, d_b2, d_h2, N, 128, 64, 1, round_mode);
    // Layer 3: 64 -> 10 linear
    int total3 = N * 10;
    layer_kernel<<<(total3+threads-1)/threads, threads>>>(
        d_h2, d_w3, d_b3, d_logits, N, 64, 10, 0, round_mode);
}

int main() {
    printf("============================================================\n");
    printf("z2429b: ROUNDING MODE ENSEMBLE\n");
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

    float *d_images, *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3;
    float *d_h1, *d_h2, *d_logits;

    hipMalloc(&d_images, N*784*sizeof(float));
    hipMalloc(&d_w1, 128*784*sizeof(float)); hipMalloc(&d_b1, 128*sizeof(float));
    hipMalloc(&d_w2, 64*128*sizeof(float));  hipMalloc(&d_b2, 64*sizeof(float));
    hipMalloc(&d_w3, 10*64*sizeof(float));   hipMalloc(&d_b3, 10*sizeof(float));
    hipMalloc(&d_h1, N*128*sizeof(float));
    hipMalloc(&d_h2, N*64*sizeof(float));
    hipMalloc(&d_logits, N*10*sizeof(float));

    hipMemcpy(d_images, h_images, N*784*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_w1, h_w1, 128*784*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b1, h_b1, 128*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_w2, h_w2, 64*128*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b2, h_b2, 64*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_w3, h_w3, 10*64*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b3, h_b3, 10*sizeof(float), hipMemcpyHostToDevice);

    float* h_all_logits = (float*)malloc(4 * N * 10 * sizeof(float));
    const char* mode_names[] = {"RNE", "R+inf", "R-inf", "RTZ"};

    // ================================================================
    // Run MLP with each rounding mode
    // ================================================================
    printf("Running MLP with 4 rounding modes on %d samples...\n\n", N);

    double mode_times[4];

    for (int m = 0; m < 4; m++) {
        // Warmup
        run_mlp(d_images, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_h1, d_h2, d_logits, N, m);
        hipDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < 20; rep++) {
            run_mlp(d_images, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_h1, d_h2, d_logits, N, m);
        }
        hipDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        mode_times[m] = std::chrono::duration<double, std::milli>(t1 - t0).count() / 20.0;

        hipMemcpy(h_all_logits + m * N * 10, d_logits, N*10*sizeof(float), hipMemcpyDeviceToHost);

        int correct = 0;
        for (int i = 0; i < N; i++) {
            float* l = h_all_logits + m * N * 10 + i * 10;
            int pred = 0; float mx = l[0];
            for (int j = 1; j < 10; j++) if (l[j] > mx) { mx = l[j]; pred = j; }
            if (pred == h_labels[i]) correct++;
        }
        printf("  Mode %d (%s): %.2f%% accuracy, %.2f ms\n",
               m, mode_names[m], 100.0*correct/N, mode_times[m]);
    }

    // ================================================================
    // Ensemble methods
    // ================================================================
    printf("\n--- Ensemble Results ---\n");

    int base_correct = 0;  // mode 0 = RNE
    int avg_correct = 0;
    int vote_correct = 0;
    int disagreements = 0;
    int ens_wins = 0, base_wins = 0;

    // Per-mode predictions for analysis
    for (int i = 0; i < N; i++) {
        // Baseline (mode 0)
        float* l0 = h_all_logits + i * 10;
        int base_pred = 0; float base_mx = l0[0];
        for (int j = 1; j < 10; j++) if (l0[j] > base_mx) { base_mx = l0[j]; base_pred = j; }
        if (base_pred == h_labels[i]) base_correct++;

        // Average across modes
        float avg[10] = {0};
        int mode_preds[4];
        for (int m = 0; m < 4; m++) {
            float* lm = h_all_logits + m * N * 10 + i * 10;
            int pred = 0; float mx = lm[0];
            for (int j = 0; j < 10; j++) {
                avg[j] += lm[j];
                if (lm[j] > mx) { mx = lm[j]; pred = j; }
            }
            mode_preds[m] = pred;
        }
        int avg_pred = 0; float avg_mx = avg[0];
        for (int j = 1; j < 10; j++) if (avg[j] > avg_mx) { avg_mx = avg[j]; avg_pred = j; }
        if (avg_pred == h_labels[i]) avg_correct++;

        // Majority vote
        int votes[10] = {0};
        for (int m = 0; m < 4; m++) votes[mode_preds[m]]++;
        int vpred = 0; int vmx = votes[0];
        for (int j = 1; j < 10; j++) if (votes[j] > vmx) { vmx = votes[j]; vpred = j; }
        if (vpred == h_labels[i]) vote_correct++;

        // Disagreement analysis
        bool same = true;
        for (int m = 1; m < 4; m++) if (mode_preds[m] != mode_preds[0]) { same = false; break; }
        if (!same) {
            disagreements++;
            if (avg_pred == h_labels[i] && base_pred != h_labels[i]) ens_wins++;
            if (base_pred == h_labels[i] && avg_pred != h_labels[i]) base_wins++;
        }
    }

    printf("  Baseline (RNE):   %.2f%%\n", 100.0*base_correct/N);
    printf("  Average logits:   %.2f%% (Δ=%+.2fpp)\n",
           100.0*avg_correct/N, 100.0*(avg_correct-base_correct)/N);
    printf("  Majority vote:    %.2f%% (Δ=%+.2fpp)\n",
           100.0*vote_correct/N, 100.0*(vote_correct-base_correct)/N);
    printf("\n  Disagreements:    %d/%d (%.2f%%)\n",
           disagreements, N, 100.0*disagreements/N);
    printf("  Ensemble wins:    %d (avg correct, base wrong)\n", ens_wins);
    printf("  Base wins:        %d (base correct, avg wrong)\n", base_wins);
    printf("  Net:              %+d samples\n", ens_wins - base_wins);
    printf("\n  Timing: baseline=%.2fms, per-mode-avg=%.2fms\n",
           mode_times[0], (mode_times[0]+mode_times[1]+mode_times[2]+mode_times[3])/4);

    // Logit difference analysis
    printf("\n--- Logit Divergence ---\n");
    double total_diff = 0;
    double max_diff = 0;
    int n_differ = 0;
    for (int i = 0; i < N; i++) {
        float* l0 = h_all_logits + i * 10;
        for (int m = 1; m < 4; m++) {
            float* lm = h_all_logits + m * N * 10 + i * 10;
            for (int j = 0; j < 10; j++) {
                double d = fabs(l0[j] - lm[j]);
                if (d > 1e-6) n_differ++;
                total_diff += d;
                if (d > max_diff) max_diff = d;
            }
        }
    }
    printf("  Logits that differ (>1e-6): %d/%d (%.1f%%)\n",
           n_differ, N*10*3, 100.0*n_differ/(N*10*3));
    printf("  Mean |diff|: %.6f\n", total_diff / (N*10*3));
    printf("  Max  |diff|: %.6f\n", max_diff);

    printf("\n============================================================\n");
    double delta = 100.0*(avg_correct - base_correct) / N;
    if (delta > 0.05) {
        printf("RESULT: ROUNDING ENSEMBLE IMPROVES by %+.2fpp (%d→%d correct)\n",
               delta, base_correct, avg_correct);
    } else if (delta < -0.05) {
        printf("RESULT: Rounding ensemble hurts by %.2fpp\n", delta);
    } else {
        printf("RESULT: Rounding ensemble neutral (%.3fpp, %d disagreements)\n",
               delta, disagreements);
    }
    printf("============================================================\n");

    hipFree(d_images); hipFree(d_w1); hipFree(d_b1);
    hipFree(d_w2); hipFree(d_b2); hipFree(d_w3); hipFree(d_b3);
    hipFree(d_h1); hipFree(d_h2); hipFree(d_logits);
    free(h_images); free(h_labels); free(h_all_logits);
    free(h_w1); free(h_b1); free(h_w2); free(h_b2); free(h_w3); free(h_b3);
    return 0;
}
