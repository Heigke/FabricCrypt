/**
 * z2434: GEMM Sidecar — Real matmul + mechanism expert families
 *
 * Architecture:
 *   Stream 0 (GEMM):    hipBLAS sgemm for real MLP inference
 *   Stream 1 (Sidecar): 4-family mechanism experts extract control signals
 *
 * The sidecar does NOT replace GEMM. It provides:
 *   - Confidence/uncertainty signal (for early exit or routing)
 *   - Memory trace (for temporal context)
 *   - Nonlinearity signal (for OOD detection)
 *   - Identity/diversity signal (for ensemble diversity)
 *
 * 4 Expert Families (from feedback, matched to our proven mechanisms):
 *
 *   MEMORY:       branch divergence state, stale register persistence
 *   NONLINEARITY: L1/LDS bank conflict patterns, atomic contention
 *   SLOW_CONTEXT: thermal/clock drift proxy (accumulation bias)
 *   IDENTITY:     per-CU variation via shfl pattern, ballot diversity
 *
 * Key design: per-layer sidecar, not per-op. One sidecar kernel after each GEMM.
 * Sidecar reads GEMM output tile summaries, emits a small control vector.
 */

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

#define SIDECAR_DIM 8  // 2 per family

// ============================================================
// SIDECAR KERNEL: 4-family mechanism experts
// Reads GEMM output, extracts control signals
// ============================================================
__global__ void sidecar_experts(
    const float* __restrict__ gemm_output,  // [batch, out_dim] — GEMM result
    float* __restrict__ control,             // [batch, SIDECAR_DIM] — control signals
    int batch, int out_dim
) {
    int sample = blockIdx.x;
    if (sample >= batch) return;
    int tid = threadIdx.x;
    int lane = tid % 64;

    const float* row = gemm_output + sample * out_dim;

    // Each thread scans a portion of the output
    float local_sum = 0.0f;
    float local_abs = 0.0f;
    float local_sq = 0.0f;
    float branch_state = 0.0f;
    int zero_count = 0;

    for (int j = tid; j < out_dim; j += blockDim.x) {
        float v = row[j];
        local_sum += v;
        local_abs += fabsf(v);
        local_sq += v * v;

        // MEMORY: branch divergence — track pattern of positive/negative transitions
        if (v > branch_state) {
            branch_state = branch_state * 0.8f + v * 0.2f;
        } else {
            branch_state = branch_state * 0.95f + v * 0.05f;
        }

        // Count zeros (ReLU dead neurons)
        if (v == 0.0f) zero_count++;
    }

    // === FAMILY 1: MEMORY (branch divergence + persistence) ===
    // Branch state preserves recent activation trend
    float mem_branch = branch_state;
    // Persistence: shuffle state across lanes — stable patterns survive
    float mem_persist = __shfl_xor(branch_state, 1) + __shfl_xor(branch_state, 2);

    // === FAMILY 2: NONLINEARITY (contention + activation shape) ===
    // Use atomicAdd to create contention-dependent accumulation
    __shared__ float s_nonlin[2];
    if (tid < 2) s_nonlin[tid] = 0.0f;
    __syncthreads();

    // Only high-activation threads contribute → contention varies with sparsity
    if (local_abs > 0.01f * out_dim) {
        atomicAdd(&s_nonlin[0], local_abs);
    }
    // Sparsity metric via dead neuron count
    atomicAdd(&s_nonlin[1], (float)zero_count);
    __syncthreads();

    float nonlin_contention = s_nonlin[0];
    float nonlin_sparsity = s_nonlin[1] / (float)(blockDim.x);

    // === FAMILY 3: SLOW CONTEXT (accumulation bias, drift proxy) ===
    // Different rounding modes create slightly different sums
    float ctx_sum_rne = local_sum;  // already computed in RNE
    // Approximate context: variance of outputs (high variance = uncertain)
    float mean_approx = local_sum / (out_dim / blockDim.x + 1e-10f);
    float ctx_variance = local_sq / (out_dim / blockDim.x + 1e-10f) - mean_approx * mean_approx;

    // === FAMILY 4: IDENTITY (per-lane diversity, ensemble signal) ===
    // Ballot: how many lanes agree on sign of their partial sum
    unsigned long long ballot_pos = __ballot(local_sum > 0.0f);
    float id_agreement = (float)__popcll(ballot_pos) / 64.0f;
    // Cross-lane diversity
    float neighbor_sum = __shfl_xor(local_sum, 1);
    float id_diversity = fabsf(local_sum - neighbor_sum);

    // Thread 0 writes the 8 control signals
    if (tid == 0) {
        float* out = control + sample * SIDECAR_DIM;
        // Memory family
        out[0] = mem_branch;
        out[1] = mem_persist / (out_dim + 1e-10f);
        // Nonlinearity family
        out[2] = nonlin_contention / (batch + 1e-10f);
        out[3] = nonlin_sparsity;
        // Slow context family
        out[4] = ctx_sum_rne;
        out[5] = ctx_variance;
        // Identity family
        out[6] = id_agreement;
        out[7] = id_diversity;
    }
}

// ============================================================
// DECISION KERNEL: use sidecar signals to modify GEMM output
// This is the "residual correction / gating" step
// ============================================================
__global__ void apply_sidecar(
    float* __restrict__ logits,           // [batch, 10] — GEMM output (modified in-place)
    const float* __restrict__ control,    // [batch, SIDECAR_DIM] — sidecar signals
    const float* __restrict__ weights,    // [SIDECAR_DIM, 10] — learned projection
    int batch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample = idx / 10;
    int cls = idx % 10;
    if (sample >= batch) return;

    // Small residual from sidecar → added to logits
    const float* ctrl = control + sample * SIDECAR_DIM;
    float residual = 0.0f;
    for (int k = 0; k < SIDECAR_DIM; k++) {
        residual += ctrl[k] * weights[k * 10 + cls];
    }

    // Scale residual to be small relative to logits
    logits[sample * 10 + cls] += residual * 0.01f;
}

float* load_bin(const char* path, int count) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); exit(1); }
    float* data = (float*)malloc(count * sizeof(float));
    fread(data, sizeof(float), count, f);
    fclose(f);
    return data;
}

int main() {
    printf("============================================================\n");
    printf("z2434: GEMM SIDECAR — Real matmul + mechanism experts\n");
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

    // Load MLP weights
    sprintf(path, "%s/models/mnist_mlp/w1.bin", base); float* h_w1 = load_bin(path, 128*784);
    sprintf(path, "%s/models/mnist_mlp/b1.bin", base); float* h_b1 = load_bin(path, 128);
    sprintf(path, "%s/models/mnist_mlp/w2.bin", base); float* h_w2 = load_bin(path, 64*128);
    sprintf(path, "%s/models/mnist_mlp/b2.bin", base); float* h_b2 = load_bin(path, 64);
    sprintf(path, "%s/models/mnist_mlp/w3.bin", base); float* h_w3 = load_bin(path, 10*64);
    sprintf(path, "%s/models/mnist_mlp/b3.bin", base); float* h_b3 = load_bin(path, 10);

    // GPU memory
    float *d_img, *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3;
    float *d_h1, *d_h2, *d_logits, *d_logits_base;
    float *d_control1, *d_control2, *d_control3;  // sidecar per layer
    float *d_sidecar_w;  // learned projection

    hipMalloc(&d_img, N*784*4);
    hipMalloc(&d_w1, 128*784*4); hipMalloc(&d_b1, 128*4);
    hipMalloc(&d_w2, 64*128*4);  hipMalloc(&d_b2, 64*4);
    hipMalloc(&d_w3, 10*64*4);   hipMalloc(&d_b3, 10*4);
    hipMalloc(&d_h1, N*128*4);
    hipMalloc(&d_h2, N*64*4);
    hipMalloc(&d_logits, N*10*4);
    hipMalloc(&d_logits_base, N*10*4);  // copy for baseline
    hipMalloc(&d_control1, N*SIDECAR_DIM*4);
    hipMalloc(&d_control2, N*SIDECAR_DIM*4);
    hipMalloc(&d_control3, N*SIDECAR_DIM*4);
    hipMalloc(&d_sidecar_w, SIDECAR_DIM*3*10*4);  // 3 layers × 8 × 10

    hipMemcpy(d_img, h_images, N*784*4, hipMemcpyHostToDevice);
    hipMemcpy(d_w1, h_w1, 128*784*4, hipMemcpyHostToDevice);
    hipMemcpy(d_b1, h_b1, 128*4, hipMemcpyHostToDevice);
    hipMemcpy(d_w2, h_w2, 64*128*4, hipMemcpyHostToDevice);
    hipMemcpy(d_b2, h_b2, 64*4, hipMemcpyHostToDevice);
    hipMemcpy(d_w3, h_w3, 10*64*4, hipMemcpyHostToDevice);
    hipMemcpy(d_b3, h_b3, 10*4, hipMemcpyHostToDevice);

    // Create hipBLAS handle and streams
    hipblasHandle_t blas_handle;
    hipblasCreate(&blas_handle);

    hipStream_t stream_gemm, stream_sidecar;
    hipStreamCreate(&stream_gemm);
    hipStreamCreate(&stream_sidecar);

    hipblasSetStream(blas_handle, stream_gemm);

    hipEvent_t event_layer1, event_layer2, event_layer3;
    hipEventCreate(&event_layer1);
    hipEventCreate(&event_layer2);
    hipEventCreate(&event_layer3);

    float alpha = 1.0f, beta = 0.0f;

    // ================================================================
    // BASELINE: Pure GEMM MLP (no sidecar)
    // ================================================================
    printf("--- Baseline: Pure hipBLAS GEMM MLP ---\n");

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < 20; rep++) {
        // Layer 1: h1 = images × W1^T + b1, then ReLU
        // hipblasSgemm: C = alpha * op(A) * op(B) + beta * C
        // A = W1 (128×784), B = images^T (784×N), C = h1 (128×N)
        // We want h1[N,128] = images[N,784] × W1^T[784,128]
        // In column-major: h1^T[128,N] = W1[128,784] × images^T[784,N]
        hipblasSgemm(blas_handle, HIPBLAS_OP_N, HIPBLAS_OP_T,
                     128, N, 784, &alpha, d_w1, 128, d_img, 784, &beta, d_h1, 128);

        // Add bias + ReLU (simple kernel)
        // For now, skip bias/ReLU for timing — we'll add them back
    }
    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double base_gemm_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 20.0;
    printf("  Layer 1 GEMM only: %.3f ms\n", base_gemm_ms);

    // Full MLP with bias+ReLU via simple kernels
    // Layer 1: h1 = ReLU(W1 × img + b1)
    hipblasSgemm(blas_handle, HIPBLAS_OP_N, HIPBLAS_OP_T,
                 128, N, 784, &alpha, d_w1, 128, d_img, 784, &beta, d_h1, 128);
    hipDeviceSynchronize();

    // Bias + ReLU on host (quick and dirty for now)
    float* h_h1 = (float*)malloc(N*128*4);
    hipMemcpy(h_h1, d_h1, N*128*4, hipMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < 128; j++) {
            h_h1[i*128+j] = h_h1[i*128+j] + h_b1[j]; // column-major to row-major issue
        }
    // Actually hipBLAS is column-major. Let me fix: h1[j,i] = sum(W1[j,k]*img[k,i])
    // In memory: h1 is [128, N] column-major, so h1[j + 128*i]
    for (int i = 0; i < N; i++)
        for (int j = 0; j < 128; j++) {
            float val = h_h1[j + 128*i] + h_b1[j];
            h_h1[j + 128*i] = (val > 0) ? val : 0;
        }
    hipMemcpy(d_h1, h_h1, N*128*4, hipMemcpyHostToDevice);

    // Layer 2: h2 = ReLU(W2 × h1 + b2)
    hipblasSgemm(blas_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                 64, N, 128, &alpha, d_w2, 64, d_h1, 128, &beta, d_h2, 64);
    hipDeviceSynchronize();
    float* h_h2 = (float*)malloc(N*64*4);
    hipMemcpy(h_h2, d_h2, N*64*4, hipMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < 64; j++) {
            float val = h_h2[j + 64*i] + h_b2[j];
            h_h2[j + 64*i] = (val > 0) ? val : 0;
        }
    hipMemcpy(d_h2, h_h2, N*64*4, hipMemcpyHostToDevice);

    // Layer 3: logits = W3 × h2 + b3
    hipblasSgemm(blas_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                 10, N, 64, &alpha, d_w3, 10, d_h2, 64, &beta, d_logits, 10);
    hipDeviceSynchronize();
    float* h_logits = (float*)malloc(N*10*4);
    hipMemcpy(h_logits, d_logits, N*10*4, hipMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < 10; j++)
            h_logits[j + 10*i] += h_b3[j];

    // Accuracy
    int correct_base = 0;
    for (int i = 0; i < N; i++) {
        int pred = 0; float mx = h_logits[10*i];
        for (int j = 1; j < 10; j++) if (h_logits[j+10*i] > mx) { mx = h_logits[j+10*i]; pred = j; }
        if (pred == h_labels[i]) correct_base++;
    }
    printf("  Baseline accuracy: %.2f%%\n\n", 100.0*correct_base/N);

    // Save baseline logits
    hipMemcpy(d_logits_base, d_logits, N*10*4, hipMemcpyDeviceToDevice);

    // ================================================================
    // SIDECAR: Run mechanism experts on GEMM hidden activations
    // ================================================================
    printf("--- Sidecar: Mechanism experts on GEMM activations ---\n");

    // Sidecar on layer 1 output (128-dim hidden)
    t0 = std::chrono::high_resolution_clock::now();
    sidecar_experts<<<N, 128, 0, stream_sidecar>>>(d_h1, d_control1, N, 128);
    // Sidecar on layer 2 output (64-dim hidden)
    sidecar_experts<<<N, 64, 0, stream_sidecar>>>(d_h2, d_control2, N, 64);
    // Sidecar on logits (10-dim output)
    sidecar_experts<<<N, 64, 0, stream_sidecar>>>(d_logits, d_control3, N, 10);
    hipStreamSynchronize(stream_sidecar);
    t1 = std::chrono::high_resolution_clock::now();
    double sidecar_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Sidecar time: %.3f ms (3 layers)\n", sidecar_ms);

    // Read sidecar features
    float* h_ctrl1 = (float*)malloc(N*SIDECAR_DIM*4);
    float* h_ctrl2 = (float*)malloc(N*SIDECAR_DIM*4);
    float* h_ctrl3 = (float*)malloc(N*SIDECAR_DIM*4);
    hipMemcpy(h_ctrl1, d_control1, N*SIDECAR_DIM*4, hipMemcpyDeviceToHost);
    hipMemcpy(h_ctrl2, d_control2, N*SIDECAR_DIM*4, hipMemcpyDeviceToHost);
    hipMemcpy(h_ctrl3, d_control3, N*SIDECAR_DIM*4, hipMemcpyDeviceToHost);

    // ================================================================
    // ANALYSIS: Do sidecar signals separate correct/incorrect?
    // ================================================================
    printf("\n--- Sidecar Signal Analysis ---\n");

    const char* family_names[] = {
        "Mem:branch", "Mem:persist", "NL:contention", "NL:sparsity",
        "Ctx:sum", "Ctx:variance", "ID:agreement", "ID:diversity"
    };

    // Compute per-family F-ratio for correct vs incorrect
    for (int layer = 0; layer < 3; layer++) {
        float* ctrl = (layer == 0) ? h_ctrl1 : (layer == 1) ? h_ctrl2 : h_ctrl3;
        printf("  Layer %d:\n", layer+1);

        for (int f = 0; f < SIDECAR_DIM; f++) {
            // Split by correct/incorrect
            double sum_c = 0, sum_i = 0, sq_c = 0, sq_i = 0;
            int n_c = 0, n_i = 0;

            for (int i = 0; i < N; i++) {
                float v = ctrl[i * SIDECAR_DIM + f];
                int pred = 0; float mx = h_logits[10*i];
                for (int j = 1; j < 10; j++) if (h_logits[j+10*i] > mx) { mx = h_logits[j+10*i]; pred = j; }

                if (pred == h_labels[i]) {
                    sum_c += v; sq_c += v*v; n_c++;
                } else {
                    sum_i += v; sq_i += v*v; n_i++;
                }
            }

            double mean_c = sum_c / (n_c + 1e-10);
            double mean_i = sum_i / (n_i + 1e-10);
            double var_c = sq_c / (n_c + 1e-10) - mean_c * mean_c;
            double var_i = sq_i / (n_i + 1e-10) - mean_i * mean_i;
            double pooled_var = (var_c * n_c + var_i * n_i) / (n_c + n_i + 1e-10);
            double cohens_d = fabs(mean_c - mean_i) / sqrt(pooled_var + 1e-10);

            if (cohens_d > 0.1) {
                printf("    %14s: correct=%.4f incorrect=%.4f d=%.3f %s\n",
                       family_names[f], mean_c, mean_i, cohens_d,
                       cohens_d > 0.5 ? "★★★" : cohens_d > 0.2 ? "★★" : "★");
            }
        }
    }

    // ================================================================
    // FUSION: Train simple projection and test accuracy improvement
    // ================================================================
    printf("\n--- Fusion: Sidecar residual correction ---\n");

    // Concatenate all sidecar features: 3 layers × 8 = 24 features
    int n_train = 5000;
    int n_test = N - n_train;
    int feat_dim = SIDECAR_DIM * 3;

    float* X_train = (float*)malloc(n_train * feat_dim * sizeof(float));
    float* X_test = (float*)malloc(n_test * feat_dim * sizeof(float));

    for (int i = 0; i < N; i++) {
        float* dst = (i < n_train) ? X_train + i * feat_dim : X_test + (i - n_train) * feat_dim;
        memcpy(dst, h_ctrl1 + i * SIDECAR_DIM, SIDECAR_DIM * sizeof(float));
        memcpy(dst + SIDECAR_DIM, h_ctrl2 + i * SIDECAR_DIM, SIDECAR_DIM * sizeof(float));
        memcpy(dst + SIDECAR_DIM * 2, h_ctrl3 + i * SIDECAR_DIM, SIDECAR_DIM * sizeof(float));
    }

    // Ridge regression: sidecar features → 10-class one-hot
    // W = (X^T X + αI)^{-1} X^T Y
    // Using logits + sidecar features combined
    int combo_dim = 10 + feat_dim;  // logits (10) + sidecar (24)
    float* combo_train = (float*)malloc(n_train * combo_dim * sizeof(float));
    float* combo_test = (float*)malloc(n_test * combo_dim * sizeof(float));

    for (int i = 0; i < n_train; i++) {
        for (int j = 0; j < 10; j++) combo_train[i * combo_dim + j] = h_logits[j + 10*i];
        for (int j = 0; j < feat_dim; j++) combo_train[i * combo_dim + 10 + j] = X_train[i * feat_dim + j];
    }
    for (int i = 0; i < n_test; i++) {
        int ii = i + n_train;
        for (int j = 0; j < 10; j++) combo_test[i * combo_dim + j] = h_logits[j + 10*ii];
        for (int j = 0; j < feat_dim; j++) combo_test[i * combo_dim + 10 + j] = X_test[i * feat_dim + j];
    }

    // Simple ridge in C (small dimensions, no need for LAPACK)
    // XtX = X^T X + alpha*I, XtY = X^T Y
    float alpha_ridge = 1.0f;
    int D = combo_dim;
    int C = 10;
    float* XtX = (float*)calloc(D * D, sizeof(float));
    float* XtY = (float*)calloc(D * C, sizeof(float));

    for (int i = 0; i < n_train; i++) {
        float* x = combo_train + i * D;
        // One-hot target
        int label = h_labels[i];
        for (int a = 0; a < D; a++) {
            for (int b = 0; b < D; b++) XtX[a*D+b] += x[a] * x[b];
            XtY[a*C+label] += x[a];
        }
    }
    for (int d = 0; d < D; d++) XtX[d*D+d] += alpha_ridge;

    // Solve via Cholesky (small D=34, fine for loops)
    // Use simple Gauss elimination
    float* A = (float*)malloc(D * (D+C) * sizeof(float));
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) A[i*(D+C)+j] = XtX[i*D+j];
        for (int j = 0; j < C; j++) A[i*(D+C)+D+j] = XtY[i*C+j];
    }
    // Gaussian elimination
    for (int i = 0; i < D; i++) {
        float pivot = A[i*(D+C)+i];
        if (fabsf(pivot) < 1e-10f) continue;
        for (int j = i; j < D+C; j++) A[i*(D+C)+j] /= pivot;
        for (int k = 0; k < D; k++) {
            if (k == i) continue;
            float factor = A[k*(D+C)+i];
            for (int j = i; j < D+C; j++) A[k*(D+C)+j] -= factor * A[i*(D+C)+j];
        }
    }
    float* W_ridge = (float*)malloc(D * C * sizeof(float));
    for (int i = 0; i < D; i++)
        for (int j = 0; j < C; j++)
            W_ridge[i*C+j] = A[i*(D+C)+D+j];

    // Test accuracy: logits only vs logits+sidecar
    int correct_logits_only = 0, correct_combo = 0;

    for (int i = 0; i < n_test; i++) {
        int ii = i + n_train;

        // Logits only (argmax)
        int pred_base = 0; float mx_base = h_logits[10*ii];
        for (int j = 1; j < 10; j++) if (h_logits[j+10*ii] > mx_base) { mx_base = h_logits[j+10*ii]; pred_base = j; }
        if (pred_base == h_labels[ii]) correct_logits_only++;

        // Combo (ridge)
        float scores[10] = {0};
        float* x = combo_test + i * D;
        for (int j = 0; j < C; j++)
            for (int k = 0; k < D; k++)
                scores[j] += x[k] * W_ridge[k*C+j];
        int pred_combo = 0; float mx_combo = scores[0];
        for (int j = 1; j < 10; j++) if (scores[j] > mx_combo) { mx_combo = scores[j]; pred_combo = j; }
        if (pred_combo == h_labels[ii]) correct_combo++;
    }

    float acc_base = 100.0f * correct_logits_only / n_test;
    float acc_combo = 100.0f * correct_combo / n_test;
    float delta = acc_combo - acc_base;

    printf("  Logits only (argmax): %.2f%% (%d/%d)\n", acc_base, correct_logits_only, n_test);
    printf("  Logits + sidecar:     %.2f%% (%d/%d)\n", acc_combo, correct_combo, n_test);
    printf("  Delta:                %+.2fpp (%+d samples)\n", delta, correct_combo - correct_logits_only);

    // ================================================================
    // TIMING: overlapped execution
    // ================================================================
    printf("\n--- Timing: GEMM + Sidecar Overlap ---\n");

    // GEMM alone
    t0 = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < 50; rep++) {
        hipblasSgemm(blas_handle, HIPBLAS_OP_N, HIPBLAS_OP_T,
                     128, N, 784, &alpha, d_w1, 128, d_img, 784, &beta, d_h1, 128);
        hipblasSgemm(blas_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                     64, N, 128, &alpha, d_w2, 64, d_h1, 128, &beta, d_h2, 64);
        hipblasSgemm(blas_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                     10, N, 64, &alpha, d_w3, 10, d_h2, 64, &beta, d_logits, 10);
    }
    hipDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    double gemm_only_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 50.0;

    // GEMM + sidecar overlapped
    t0 = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < 50; rep++) {
        // Layer 1 GEMM
        hipblasSgemm(blas_handle, HIPBLAS_OP_N, HIPBLAS_OP_T,
                     128, N, 784, &alpha, d_w1, 128, d_img, 784, &beta, d_h1, 128);
        hipEventRecord(event_layer1, stream_gemm);
        // Sidecar layer 1 (waits for GEMM)
        hipStreamWaitEvent(stream_sidecar, event_layer1, 0);
        sidecar_experts<<<N, 128, 0, stream_sidecar>>>(d_h1, d_control1, N, 128);

        // Layer 2 GEMM (doesn't wait for sidecar)
        hipblasSgemm(blas_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                     64, N, 128, &alpha, d_w2, 64, d_h1, 128, &beta, d_h2, 64);
        hipEventRecord(event_layer2, stream_gemm);
        hipStreamWaitEvent(stream_sidecar, event_layer2, 0);
        sidecar_experts<<<N, 64, 0, stream_sidecar>>>(d_h2, d_control2, N, 64);

        // Layer 3 GEMM
        hipblasSgemm(blas_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                     10, N, 64, &alpha, d_w3, 10, d_h2, 64, &beta, d_logits, 10);
        hipEventRecord(event_layer3, stream_gemm);
        hipStreamWaitEvent(stream_sidecar, event_layer3, 0);
        sidecar_experts<<<N, 64, 0, stream_sidecar>>>(d_logits, d_control3, N, 10);
    }
    hipDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    double overlap_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 50.0;

    printf("  GEMM only:          %.3f ms\n", gemm_only_ms);
    printf("  GEMM + sidecar:     %.3f ms\n", overlap_ms);
    printf("  Overhead:           %.1f%%\n", 100.0 * (overlap_ms - gemm_only_ms) / gemm_only_ms);

    printf("\n============================================================\n");
    printf("VERDICT\n");
    printf("============================================================\n");
    printf("  Accuracy:  %+.2fpp from sidecar\n", delta);
    printf("  Overhead:  %.1f%%\n", 100.0 * (overlap_ms - gemm_only_ms) / gemm_only_ms);
    if (delta > 0.05) {
        printf("  SIDECAR IMPROVES ACCURACY at %.1f%% overhead\n",
               100.0 * (overlap_ms - gemm_only_ms) / gemm_only_ms);
    } else if (delta > -0.05) {
        printf("  Sidecar neutral (signals exist but ridge can't exploit them yet)\n");
    } else {
        printf("  Sidecar hurts — feature noise > signal\n");
    }
    printf("============================================================\n");

    // Cleanup
    hipblasDestroy(blas_handle);
    hipStreamDestroy(stream_gemm); hipStreamDestroy(stream_sidecar);
    hipEventDestroy(event_layer1); hipEventDestroy(event_layer2); hipEventDestroy(event_layer3);
    hipFree(d_img); hipFree(d_w1); hipFree(d_b1); hipFree(d_w2); hipFree(d_b2);
    hipFree(d_w3); hipFree(d_b3); hipFree(d_h1); hipFree(d_h2);
    hipFree(d_logits); hipFree(d_logits_base);
    hipFree(d_control1); hipFree(d_control2); hipFree(d_control3); hipFree(d_sidecar_w);
    free(h_images); free(h_labels); free(h_logits); free(h_h1); free(h_h2);
    free(h_ctrl1); free(h_ctrl2); free(h_ctrl3);
    free(X_train); free(X_test); free(combo_train); free(combo_test);
    free(XtX); free(XtY); free(A); free(W_ridge);
    free(h_w1); free(h_b1); free(h_w2); free(h_b2); free(h_w3); free(h_b3);
    return 0;
}
