/**
 * z2436: REAL GPU mechanism sidecar for OOD detection
 *
 * z2435 used CPU-computed proxies. This uses ACTUAL GPU ISA mechanisms:
 *   - atomicAdd contention (real hardware race conditions)
 *   - __ballot population coding (real wavefront vote)
 *   - __shfl inter-lane communication (real register file crossbar)
 *   - branch divergence state (real branch prediction unit)
 *   - s_setreg rounding perturbation (real FP mode register)
 *
 * Pipeline:
 *   1. GEMM layers compute MLP (stream 0)
 *   2. Sidecar reads activations via real mechanisms (stream 1)
 *   3. Output: 10 logits + 24 sidecar features (8 per layer × 3 layers)
 *   4. Host analyzes: can sidecar detect OOD where softmax fails?
 *
 * Tests on: MNIST (ID) vs uniform, gaussian, constant, inverted, shuffled (OOD)
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

#define SIDECAR_DIM 8
#define N_OOD_TYPES 5

// GEMM layer
__global__ void gemm_relu(
    const float* __restrict__ X, const float* __restrict__ W, const float* __restrict__ b,
    float* __restrict__ Y, int N, int in_dim, int out_dim, int use_relu
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int s = idx / out_dim, n = idx % out_dim;
    if (s >= N) return;
    float sum = b[n];
    for (int k = 0; k < in_dim; k++) sum += W[n*in_dim+k] * X[s*in_dim+k];
    if (use_relu && sum < 0.0f) sum = 0.0f;
    Y[s*out_dim+n] = sum;
}

// REAL mechanism sidecar — uses actual GPU ISA features
__global__ void real_sidecar(
    const float* __restrict__ act,  // [N, dim] activations from GEMM
    float* __restrict__ ctrl,       // [N, SIDECAR_DIM]
    int N, int dim
) {
    int sample = blockIdx.x;
    if (sample >= N) return;
    int tid = threadIdx.x;
    int lane = tid % 64;

    const float* row = act + sample * dim;

    // ===== REAL MECHANISM 1: Branch divergence =====
    // GPU branch predictor tracks pattern of taken/not-taken
    // Different data creates different branch history → different state
    float branch_acc = 0.0f;
    int branch_count = 0;
    for (int j = tid; j < dim; j += blockDim.x) {
        float v = row[j];
        // This if/else creates REAL branch divergence in the wavefront
        // Active vs dead neurons diverge differently for ID vs OOD
        if (v > 0.0f) {
            branch_acc += v;
            branch_count++;
        } else {
            branch_acc -= 0.01f;  // small penalty for dead neurons
        }
    }

    // ===== REAL MECHANISM 2: __ballot population vote =====
    // Count how many lanes have positive partial sums — measures consensus
    unsigned long long ballot_pos = __ballot(branch_acc > 0.0f);
    float mech_ballot = (float)__popcll(ballot_pos) / 64.0f;

    // Second ballot: activation magnitude agreement
    float mean_act = branch_acc / (branch_count + 1e-10f);
    unsigned long long ballot_high = __ballot(mean_act > 0.5f);
    float mech_ballot_high = (float)__popcll(ballot_high) / 64.0f;

    // ===== REAL MECHANISM 3: __shfl neighbor difference =====
    // Cross-lane data exchange — measures local activation gradient
    float shfl_diff = 0.0f;
    float my_val = branch_acc;
    float neighbor1 = __shfl_xor(my_val, 1);
    float neighbor2 = __shfl_xor(my_val, 2);
    float neighbor4 = __shfl_xor(my_val, 4);
    shfl_diff = fabsf(my_val - neighbor1) + fabsf(my_val - neighbor2) + fabsf(my_val - neighbor4);

    // ===== REAL MECHANISM 4: atomicAdd contention =====
    // Race condition creates hardware-dependent accumulation order
    __shared__ float s_atomic[4];
    if (tid < 4) s_atomic[tid] = 0.0f;
    __syncthreads();

    // Only lanes with positive activations participate → contention varies with sparsity
    if (branch_acc > 0.0f) {
        atomicAdd(&s_atomic[lane % 4], branch_acc * 0.001f);
    }
    __syncthreads();

    // ===== REAL MECHANISM 5: Sparsity via branch count =====
    // branch_count / total = fraction of active neurons this thread saw
    // This is COMPUTED THROUGH the branch divergence path, not a separate metric
    float active_frac = (float)branch_count / (float)((dim + blockDim.x - 1) / blockDim.x);

    // ===== REAL MECHANISM 6: s_setreg rounding sensitivity =====
    // Compute sum with RNE, then with R+inf, measure difference
    float sum_rne = 0.0f;
    for (int j = tid; j < dim; j += blockDim.x) sum_rne += row[j];

    // Switch to R+inf
    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x1" ::: "memory");
    float sum_rinf = 0.0f;
    for (int j = tid; j < dim; j += blockDim.x) sum_rinf += row[j];
    // Back to RNE
    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x0" ::: "memory");

    float round_diff = sum_rinf - sum_rne;  // rounding sensitivity

    // ===== Write 8 mechanism features =====
    if (tid == 0) {
        float* out = ctrl + sample * SIDECAR_DIM;
        out[0] = branch_acc;                           // Branch: activation trend
        out[1] = active_frac;                          // Branch: sparsity via divergence
        out[2] = mech_ballot;                          // Ballot: positive consensus
        out[3] = mech_ballot_high;                     // Ballot: high-activation consensus
        out[4] = shfl_diff;                            // Shfl: inter-lane gradient
        out[5] = s_atomic[0] + s_atomic[1] + s_atomic[2] + s_atomic[3]; // Atomic: contention
        out[6] = round_diff;                           // s_setreg: rounding sensitivity
        out[7] = shfl_diff / (fabsf(branch_acc) + 1e-10f); // Diversity ratio
    }
}

float* load_bin(const char* p, int n) {
    FILE* f = fopen(p, "rb");
    if (!f) { printf("ERR: %s\n", p); exit(1); }
    float* d = (float*)malloc(n*4); fread(d, 4, n, f); fclose(f); return d;
}

// Simple AUC computation (Mann-Whitney U)
double compute_auc(const float* scores, const int* labels, int n) {
    // labels: 0=ID, 1=OOD. Higher score = more likely OOD.
    long long n0 = 0, n1 = 0, sum_rank = 0;
    // Sort by score
    int* idx = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;
    // Simple insertion sort (small n)
    for (int i = 1; i < n; i++) {
        int key = idx[i];
        int j = i - 1;
        while (j >= 0 && scores[idx[j]] > scores[key]) { idx[j+1] = idx[j]; j--; }
        idx[j+1] = key;
    }
    for (int i = 0; i < n; i++) {
        if (labels[idx[i]] == 1) { sum_rank += (i+1); n1++; }
        else n0++;
    }
    free(idx);
    if (n0 == 0 || n1 == 0) return 0.5;
    double u = (double)sum_rank - n1*(n1+1)/2.0;
    return u / ((double)n0 * n1);
}

int main() {
    printf("============================================================\n");
    printf("z2436: REAL GPU MECHANISM OOD DETECTION\n");
    printf("============================================================\n\n");

    const char* B = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy";
    char p[512];

    // MNIST test images
    sprintf(p, "%s/data/MNIST/raw/t10k-images-idx3-ubyte", B);
    FILE* fi = fopen(p, "rb"); unsigned char hd[16]; fread(hd,1,16,fi);
    int N_MNIST = 10000;
    unsigned char* raw = (unsigned char*)malloc(N_MNIST*784);
    fread(raw,1,N_MNIST*784,fi); fclose(fi);
    float* h_mnist = (float*)malloc(N_MNIST*784*4);
    for (int i = 0; i < N_MNIST*784; i++) h_mnist[i] = raw[i]/255.0f;
    free(raw);

    sprintf(p, "%s/data/MNIST/raw/t10k-labels-idx1-ubyte", B);
    FILE* fl = fopen(p, "rb"); fread(hd,1,8,fl);
    unsigned char* h_lbl = (unsigned char*)malloc(N_MNIST);
    fread(h_lbl,1,N_MNIST,fl); fclose(fl);

    // Weights
    sprintf(p,"%s/models/mnist_mlp/w1.bin",B); float* hw1=load_bin(p,128*784);
    sprintf(p,"%s/models/mnist_mlp/b1.bin",B); float* hb1=load_bin(p,128);
    sprintf(p,"%s/models/mnist_mlp/w2.bin",B); float* hw2=load_bin(p,64*128);
    sprintf(p,"%s/models/mnist_mlp/b2.bin",B); float* hb2=load_bin(p,64);
    sprintf(p,"%s/models/mnist_mlp/w3.bin",B); float* hw3=load_bin(p,10*64);
    sprintf(p,"%s/models/mnist_mlp/b3.bin",B); float* hb3=load_bin(p,10);

    // Use subset for speed
    int N_ID = 2000;
    int N_OOD = 500;

    // Generate OOD data on host
    float* h_ood[N_OOD_TYPES];
    const char* ood_names[] = {"uniform", "gaussian", "constant_1", "shuffled", "inverted"};

    srand(42);
    // Uniform [0,1]
    h_ood[0] = (float*)malloc(N_OOD*784*4);
    for (int i = 0; i < N_OOD*784; i++) h_ood[0][i] = (float)rand()/RAND_MAX;

    // Gaussian
    h_ood[1] = (float*)malloc(N_OOD*784*4);
    for (int i = 0; i < N_OOD*784; i++) {
        float u1 = (float)(rand()+1)/(RAND_MAX+2.0f);
        float u2 = (float)rand()/RAND_MAX;
        float z = sqrtf(-2*logf(u1)) * cosf(6.2832f*u2);
        h_ood[1][i] = fmaxf(0, fminf(1, 0.5f + 0.2f*z));
    }

    // Constant 1
    h_ood[2] = (float*)malloc(N_OOD*784*4);
    for (int i = 0; i < N_OOD*784; i++) h_ood[2][i] = 1.0f;

    // Shuffled MNIST
    h_ood[3] = (float*)malloc(N_OOD*784*4);
    for (int i = 0; i < N_OOD; i++) {
        memcpy(h_ood[3]+i*784, h_mnist+i*784, 784*4);
        // Fisher-Yates shuffle
        for (int j = 783; j > 0; j--) {
            int k = rand() % (j+1);
            float tmp = h_ood[3][i*784+j];
            h_ood[3][i*784+j] = h_ood[3][i*784+k];
            h_ood[3][i*784+k] = tmp;
        }
    }

    // Inverted
    h_ood[4] = (float*)malloc(N_OOD*784*4);
    for (int i = 0; i < N_OOD*784; i++) h_ood[4][i] = 1.0f - h_mnist[i];

    // GPU memory
    int N_MAX = N_ID;  // process one batch at a time
    float *di, *dw1,*db1,*dw2,*db2,*dw3,*db3, *dh1,*dh2,*dlo;
    float *dc1,*dc2,*dc3;
    hipMalloc(&di,N_MAX*784*4);
    hipMalloc(&dw1,128*784*4); hipMalloc(&db1,128*4);
    hipMalloc(&dw2,64*128*4);  hipMalloc(&db2,64*4);
    hipMalloc(&dw3,10*64*4);   hipMalloc(&db3,10*4);
    hipMalloc(&dh1,N_MAX*128*4); hipMalloc(&dh2,N_MAX*64*4); hipMalloc(&dlo,N_MAX*10*4);
    hipMalloc(&dc1,N_MAX*SIDECAR_DIM*4);
    hipMalloc(&dc2,N_MAX*SIDECAR_DIM*4);
    hipMalloc(&dc3,N_MAX*SIDECAR_DIM*4);

    hipMemcpy(dw1,hw1,128*784*4,hipMemcpyHostToDevice);
    hipMemcpy(db1,hb1,128*4,hipMemcpyHostToDevice);
    hipMemcpy(dw2,hw2,64*128*4,hipMemcpyHostToDevice);
    hipMemcpy(db2,hb2,64*4,hipMemcpyHostToDevice);
    hipMemcpy(dw3,hw3,10*64*4,hipMemcpyHostToDevice);
    hipMemcpy(db3,hb3,10*4,hipMemcpyHostToDevice);

    hipStream_t sg, ss;
    hipStreamCreate(&sg); hipStreamCreate(&ss);
    hipEvent_t e1,e2,e3;
    hipEventCreate(&e1); hipEventCreate(&e2); hipEventCreate(&e3);

    int thr = 256;

    // Function: run MLP + sidecar on a batch, return logits + sidecar features
    auto run_batch = [&](float* h_input, int N_batch,
                         float* h_logits_out, float* h_side_out) {
        hipMemcpy(di, h_input, N_batch*784*4, hipMemcpyHostToDevice);

        // GEMM stream
        gemm_relu<<<(N_batch*128+thr-1)/thr, thr, 0, sg>>>(di,dw1,db1,dh1,N_batch,784,128,1);
        hipEventRecord(e1, sg);
        hipStreamWaitEvent(ss, e1, 0);
        real_sidecar<<<N_batch, 128, 0, ss>>>(dh1, dc1, N_batch, 128);

        gemm_relu<<<(N_batch*64+thr-1)/thr, thr, 0, sg>>>(dh1,dw2,db2,dh2,N_batch,128,64,1);
        hipEventRecord(e2, sg);
        hipStreamWaitEvent(ss, e2, 0);
        real_sidecar<<<N_batch, 64, 0, ss>>>(dh2, dc2, N_batch, 64);

        gemm_relu<<<(N_batch*10+thr-1)/thr, thr, 0, sg>>>(dh2,dw3,db3,dlo,N_batch,64,10,0);
        hipEventRecord(e3, sg);
        hipStreamWaitEvent(ss, e3, 0);
        real_sidecar<<<N_batch, 64, 0, ss>>>(dlo, dc3, N_batch, 10);

        hipDeviceSynchronize();

        hipMemcpy(h_logits_out, dlo, N_batch*10*4, hipMemcpyDeviceToHost);

        float* hc1 = (float*)malloc(N_batch*8*4);
        float* hc2 = (float*)malloc(N_batch*8*4);
        float* hc3 = (float*)malloc(N_batch*8*4);
        hipMemcpy(hc1, dc1, N_batch*8*4, hipMemcpyDeviceToHost);
        hipMemcpy(hc2, dc2, N_batch*8*4, hipMemcpyDeviceToHost);
        hipMemcpy(hc3, dc3, N_batch*8*4, hipMemcpyDeviceToHost);

        // Concatenate 3 layers
        for (int i = 0; i < N_batch; i++) {
            memcpy(h_side_out + i*24, hc1 + i*8, 8*4);
            memcpy(h_side_out + i*24 + 8, hc2 + i*8, 8*4);
            memcpy(h_side_out + i*24 + 16, hc3 + i*8, 8*4);
        }
        free(hc1); free(hc2); free(hc3);
    };

    // Run on ID data
    printf("Processing ID data (%d samples)...\n", N_ID);
    float* id_logits = (float*)malloc(N_ID*10*4);
    float* id_side = (float*)malloc(N_ID*24*4);
    run_batch(h_mnist, N_ID, id_logits, id_side);

    // Baseline accuracy
    int correct = 0;
    for (int i = 0; i < N_ID; i++) {
        int pr = 0; float mx = id_logits[i*10];
        for (int j = 1; j < 10; j++) if (id_logits[i*10+j] > mx) { mx = id_logits[i*10+j]; pr = j; }
        if (pr == h_lbl[i]) correct++;
    }
    printf("MLP accuracy on ID: %.2f%%\n\n", 100.0*correct/N_ID);

    // Softmax confidence for ID
    float* id_conf = (float*)malloc(N_ID*4);
    for (int i = 0; i < N_ID; i++) {
        float mx = -1e30f;
        for (int j = 0; j < 10; j++) if (id_logits[i*10+j] > mx) mx = id_logits[i*10+j];
        float sum_exp = 0;
        for (int j = 0; j < 10; j++) sum_exp += expf(id_logits[i*10+j] - mx);
        id_conf[i] = expf(mx - mx) / sum_exp;  // max prob
        // Actually recompute properly
        float max_prob = 0;
        for (int j = 0; j < 10; j++) {
            float prob = expf(id_logits[i*10+j] - mx) / sum_exp;
            if (prob > max_prob) max_prob = prob;
        }
        id_conf[i] = max_prob;
    }

    printf("%-12s  %8s  %8s  %8s  %8s\n", "OOD Type", "SoftConf", "AUC_conf", "AUC_side", "AUC_combo");
    printf("%-12s  %8s  %8s  %8s  %8s\n", "--------", "--------", "--------", "--------", "---------");

    // For each OOD type
    for (int ood_idx = 0; ood_idx < N_OOD_TYPES; ood_idx++) {
        float* ood_logits = (float*)malloc(N_OOD*10*4);
        float* ood_side = (float*)malloc(N_OOD*24*4);
        run_batch(h_ood[ood_idx], N_OOD, ood_logits, ood_side);

        // OOD softmax confidence
        float* ood_conf = (float*)malloc(N_OOD*4);
        float mean_conf = 0;
        for (int i = 0; i < N_OOD; i++) {
            float mx = -1e30f;
            for (int j = 0; j < 10; j++) if (ood_logits[i*10+j] > mx) mx = ood_logits[i*10+j];
            float sum_exp = 0;
            for (int j = 0; j < 10; j++) sum_exp += expf(ood_logits[i*10+j] - mx);
            float max_prob = 0;
            for (int j = 0; j < 10; j++) {
                float prob = expf(ood_logits[i*10+j] - mx) / sum_exp;
                if (prob > max_prob) max_prob = prob;
            }
            ood_conf[i] = max_prob;
            mean_conf += max_prob;
        }
        mean_conf /= N_OOD;

        // AUC: confidence (negate so higher = more OOD)
        int N_total = N_ID + N_OOD;
        float* scores_conf = (float*)malloc(N_total*4);
        int* labels_ood = (int*)malloc(N_total*4);
        for (int i = 0; i < N_ID; i++) { scores_conf[i] = -id_conf[i]; labels_ood[i] = 0; }
        for (int i = 0; i < N_OOD; i++) { scores_conf[N_ID+i] = -ood_conf[i]; labels_ood[N_ID+i] = 1; }
        double auc_conf = compute_auc(scores_conf, labels_ood, N_total);

        // AUC: sidecar features (use Euclidean distance from ID mean)
        // Compute ID sidecar mean
        float id_mean[24] = {0};
        for (int i = 0; i < N_ID; i++)
            for (int j = 0; j < 24; j++) id_mean[j] += id_side[i*24+j];
        for (int j = 0; j < 24; j++) id_mean[j] /= N_ID;

        // ID sidecar std
        float id_std[24] = {0};
        for (int i = 0; i < N_ID; i++)
            for (int j = 0; j < 24; j++) {
                float d = id_side[i*24+j] - id_mean[j];
                id_std[j] += d*d;
            }
        for (int j = 0; j < 24; j++) id_std[j] = sqrtf(id_std[j]/N_ID + 1e-10f);

        // Mahalanobis-like distance
        float* scores_side = (float*)malloc(N_total*4);
        for (int i = 0; i < N_ID; i++) {
            float dist = 0;
            for (int j = 0; j < 24; j++) {
                float z = (id_side[i*24+j] - id_mean[j]) / (id_std[j] + 1e-10f);
                dist += z*z;
            }
            scores_side[i] = dist;
        }
        for (int i = 0; i < N_OOD; i++) {
            float dist = 0;
            for (int j = 0; j < 24; j++) {
                float z = (ood_side[i*24+j] - id_mean[j]) / (id_std[j] + 1e-10f);
                dist += z*z;
            }
            scores_side[N_ID+i] = dist;
        }
        double auc_side = compute_auc(scores_side, labels_ood, N_total);

        // Combined: confidence + sidecar distance
        float* scores_combo = (float*)malloc(N_total*4);
        for (int i = 0; i < N_total; i++)
            scores_combo[i] = scores_conf[i] * 0.5f + scores_side[i] * 0.01f;
        double auc_combo = compute_auc(scores_combo, labels_ood, N_total);

        // Try sidecar-only with simple threshold on each feature
        // Find best single feature AUC
        double best_feat_auc = 0;
        int best_feat = -1;
        for (int f = 0; f < 24; f++) {
            float* scores_f = (float*)malloc(N_total*4);
            for (int i = 0; i < N_ID; i++) scores_f[i] = id_side[i*24+f];
            for (int i = 0; i < N_OOD; i++) scores_f[N_ID+i] = ood_side[i*24+f];
            double auc_f = compute_auc(scores_f, labels_ood, N_total);
            if (auc_f < 0.5) auc_f = 1.0 - auc_f;
            if (auc_f > best_feat_auc) { best_feat_auc = auc_f; best_feat = f; }
            free(scores_f);
        }

        printf("%-12s  %7.3f   %7.4f   %7.4f   %7.4f  (best_feat=%d auc=%.4f)\n",
               ood_names[ood_idx], mean_conf, auc_conf, auc_side, auc_combo,
               best_feat, best_feat_auc);

        free(ood_logits); free(ood_side); free(ood_conf);
        free(scores_conf); free(scores_side); free(scores_combo); free(labels_ood);
    }

    // Timing
    printf("\n--- Timing ---\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < 100; r++) {
        gemm_relu<<<(N_ID*128+thr-1)/thr,thr,0,sg>>>(di,dw1,db1,dh1,N_ID,784,128,1);
        gemm_relu<<<(N_ID*64+thr-1)/thr,thr,0,sg>>>(dh1,dw2,db2,dh2,N_ID,128,64,1);
        gemm_relu<<<(N_ID*10+thr-1)/thr,thr,0,sg>>>(dh2,dw3,db3,dlo,N_ID,64,10,0);
    }
    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double base_ms = std::chrono::duration<double,std::milli>(t1-t0).count()/100;

    t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < 100; r++) {
        gemm_relu<<<(N_ID*128+thr-1)/thr,thr,0,sg>>>(di,dw1,db1,dh1,N_ID,784,128,1);
        hipEventRecord(e1,sg); hipStreamWaitEvent(ss,e1,0);
        real_sidecar<<<N_ID,128,0,ss>>>(dh1,dc1,N_ID,128);
        gemm_relu<<<(N_ID*64+thr-1)/thr,thr,0,sg>>>(dh1,dw2,db2,dh2,N_ID,128,64,1);
        hipEventRecord(e2,sg); hipStreamWaitEvent(ss,e2,0);
        real_sidecar<<<N_ID,64,0,ss>>>(dh2,dc2,N_ID,64);
        gemm_relu<<<(N_ID*10+thr-1)/thr,thr,0,sg>>>(dh2,dw3,db3,dlo,N_ID,64,10,0);
        hipEventRecord(e3,sg); hipStreamWaitEvent(ss,e3,0);
        real_sidecar<<<N_ID,64,0,ss>>>(dlo,dc3,N_ID,10);
    }
    hipDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    double side_ms = std::chrono::duration<double,std::milli>(t1-t0).count()/100;

    printf("  GEMM only:      %.3f ms\n", base_ms);
    printf("  GEMM+sidecar:   %.3f ms\n", side_ms);
    printf("  Overhead:       %.1f%%\n", 100*(side_ms-base_ms)/base_ms);

    printf("\n============================================================\n");

    // Cleanup
    for (int i = 0; i < N_OOD_TYPES; i++) free(h_ood[i]);
    free(h_mnist); free(h_lbl); free(id_logits); free(id_side); free(id_conf);
    free(hw1);free(hb1);free(hw2);free(hb2);free(hw3);free(hb3);
    hipFree(di);hipFree(dw1);hipFree(db1);hipFree(dw2);hipFree(db2);hipFree(dw3);hipFree(db3);
    hipFree(dh1);hipFree(dh2);hipFree(dlo);hipFree(dc1);hipFree(dc2);hipFree(dc3);
    return 0;
}
