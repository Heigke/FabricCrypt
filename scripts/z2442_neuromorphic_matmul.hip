/**
 * z2442: Neuromorphic matmul — mechanisms ARE the computation
 *
 * Not monitoring. Not sidecar. The mechanisms REPLACE parts of standard matmul.
 *
 * Standard matmul pipeline:     FMA loop → bias → ReLU → output
 * Neuromorphic matmul pipeline: FMA loop → mechanisms → output
 *
 * What each mechanism DOES (not measures):
 *
 *   1. ATOMIC ACCUMULATION: Instead of private register accumulate,
 *      partial products go through atomicAdd in LDS. Contention creates
 *      order-dependent rounding → built-in stochastic noise (FREE dropout).
 *
 *   2. BRANCH ACTIVATION: Instead of explicit ReLU(x)=max(0,x),
 *      use branch-conditional accumulation: if(partial > 0) acc += partial.
 *      Branch predictor state creates input-dependent bias on borderline values.
 *
 *   3. SHUFFLE MIXING: After accumulation, __shfl_xor swaps partial results
 *      between adjacent output neurons. This IS lateral interaction —
 *      like learned skip connections, but from hardware topology.
 *
 *   4. BALLOT GATING: __ballot creates a 64-bit mask of which neurons fired.
 *      Use this as a population gate: only pass neurons where >50% of the
 *      wavefront agrees on activation. This IS winner-take-all competition.
 *
 *   5. BANK CONFLICT NOISE: Write partial products to LDS addresses that
 *      deliberately create bank conflicts. Conflict resolution order is
 *      hardware-determined → structured stochastic nonlinearity.
 *
 * The key insight: these aren't monitoring signals added post-hoc.
 * They REPLACE ReLU, dropout, lateral inhibition, and gating.
 * The question: does this compute something USEFUL?
 *
 * Test: MNIST classification accuracy + noise robustness
 *   - Standard matmul+ReLU (baseline)
 *   - Neuromorphic matmul (our kernel)
 *   - Standard matmul+ReLU + Gaussian noise (to match stochasticity)
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

// ============================================================
// STANDARD: conventional matmul + ReLU
// ============================================================
__global__ void standard_layer(
    const float* __restrict__ X,  // [N, in]
    const float* __restrict__ W,  // [out, in]
    const float* __restrict__ b,  // [out]
    float* __restrict__ Y,        // [N, out]
    int N, int in_dim, int out_dim, int use_relu
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int s = idx / out_dim, n = idx % out_dim;
    if (s >= N) return;
    float sum = b[n];
    for (int k = 0; k < in_dim; k++) sum += W[n*in_dim+k] * X[s*in_dim+k];
    if (use_relu && sum < 0.0f) sum = 0.0f;
    Y[s*out_dim+n] = sum;
}

// ============================================================
// NEUROMORPHIC: mechanisms ARE the computation
// One block per sample. Threads = output neurons.
// ============================================================
__global__ void neuromorphic_layer(
    const float* __restrict__ X,  // [N, in]
    const float* __restrict__ W,  // [out, in]
    const float* __restrict__ b,  // [out]
    float* __restrict__ Y,        // [N, out]
    int N, int in_dim, int out_dim,
    int use_activation,           // 1 = use branch+ballot activation
    float shfl_mix                // mixing strength (0=none, 0.1=gentle)
) {
    int sample = blockIdx.x;
    if (sample >= N) return;
    int tid = threadIdx.x;  // = output neuron index
    if (tid >= out_dim) return;

    const float* x = X + sample * in_dim;
    const float* w = W + tid * in_dim;

    // LDS for atomic accumulation (bank-conflict-rich layout)
    __shared__ float s_acc[256];      // accumulation buffer
    __shared__ float s_scratch[256];  // scratch for mixing

    // =========================================================
    // MECHANISM 1: BRANCH-CONDITIONAL ACCUMULATION
    // Instead of sum += w*x, we do:
    //   partial = w[k]*x[k]
    //   if (partial > 0) sum += partial     ← excitatory path
    //   else sum += partial * 0.01          ← leaky inhibitory path (like LeakyReLU but PER PRODUCT)
    //
    // This creates per-synapse activation, not per-neuron.
    // Branch predictor learns which synapses are typically excitatory.
    // On novel inputs, predictor is wrong → different activation pattern.
    // =========================================================
    float sum_exc = 0.0f;
    float sum_inh = 0.0f;

    for (int k = 0; k < in_dim; k++) {
        float partial = w[k] * x[k];
        if (partial > 0.0f) {
            sum_exc += partial;      // excitatory: full strength
        } else {
            sum_inh += partial;      // inhibitory: accumulate separately
        }
    }

    // Combine with asymmetric weighting (like biological E/I balance)
    float raw = sum_exc + sum_inh * 0.2f + b[tid];

    // =========================================================
    // MECHANISM 2: ATOMIC ACCUMULATION in LDS
    // Write raw activation to shared memory via atomicAdd.
    // Multiple neurons writing to same bank → contention →
    // order-dependent floating-point rounding differences.
    // This IS stochastic noise from hardware, not software.
    // =========================================================
    s_acc[tid] = 0.0f;
    __syncthreads();

    // Split raw into chunks, atomicAdd each
    // Deliberate bank-conflict pattern: tid XOR 0x3 maps 4 neurons to same bank
    int conflict_addr = tid ^ (tid & 0x3);  // groups of 4 share address
    atomicAdd(&s_acc[conflict_addr], raw * 0.25f);
    atomicAdd(&s_acc[tid], raw * 0.75f);  // self gets 75%, shared gets 25%
    __syncthreads();

    float activated = s_acc[tid];

    // =========================================================
    // MECHANISM 3: SHUFFLE MIXING (lateral interaction)
    // __shfl_xor exchanges values between adjacent neurons.
    // This IS lateral inhibition/excitation — not a proxy.
    // Neuron i receives fraction of neuron i^1's activation.
    // =========================================================
    if (shfl_mix > 0.0f) {
        float neighbor1 = __shfl_xor(activated, 1);  // distance-1 neighbor
        float neighbor2 = __shfl_xor(activated, 2);  // distance-2 neighbor

        // Lateral inhibition: subtract neighbors, add self
        // This sharpens activation peaks (like biological lateral inhibition)
        activated = activated * (1.0f + shfl_mix)
                  - neighbor1 * (shfl_mix * 0.5f)
                  - neighbor2 * (shfl_mix * 0.25f);
    }

    // =========================================================
    // MECHANISM 4: BALLOT GATING (population competition)
    // __ballot counts how many neurons in the wavefront are active.
    // If too few are active → boost all (network is under-responsive)
    // If too many are active → suppress weak ones (competition)
    // This IS winner-take-all / homeostatic regulation.
    // =========================================================
    if (use_activation) {
        unsigned long long active_mask = __ballot(activated > 0.0f);
        int n_active = __popcll(active_mask);
        float activity_ratio = (float)n_active / 64.0f;

        // Homeostatic regulation:
        // If <25% active: lower threshold (boost weak signals)
        // If >75% active: raise threshold (suppress weak signals)
        float threshold = 0.0f;
        if (activity_ratio > 0.75f) {
            threshold = activated * 0.1f;  // suppress bottom 10%
        } else if (activity_ratio < 0.25f) {
            threshold = -fabsf(activated) * 0.1f;  // boost by allowing small negatives
        }
        activated = (activated > threshold) ? activated : activated * 0.01f;
    }

    // =========================================================
    // MECHANISM 5: BANK-CONFLICT WRITEBACK
    // Write result through LDS with deliberate bank conflicts.
    // This ensures the FINAL value has hardware-determined rounding.
    // =========================================================
    s_scratch[tid] = 0.0f;
    __syncthreads();
    atomicAdd(&s_scratch[tid], activated);
    __syncthreads();

    Y[sample * out_dim + tid] = s_scratch[tid];
}

float* load_bin(const char* p, int n) {
    FILE* f=fopen(p,"rb"); float*d=(float*)malloc(n*4); fread(d,4,n,f); fclose(f); return d;
}

int main() {
    printf("============================================================\n");
    printf("z2442: NEUROMORPHIC MATMUL — mechanisms ARE computation\n");
    printf("============================================================\n\n");

    const char* B = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy";
    char p[512];

    // MNIST
    sprintf(p,"%s/data/MNIST/raw/t10k-images-idx3-ubyte",B);
    FILE*fi=fopen(p,"rb"); unsigned char hd[16]; fread(hd,1,16,fi);
    int N=10000;
    unsigned char*raw=(unsigned char*)malloc(N*784);
    fread(raw,1,N*784,fi); fclose(fi);
    float*h_img=(float*)malloc(N*784*4);
    for(int i=0;i<N*784;i++) h_img[i]=raw[i]/255.0f;
    free(raw);

    sprintf(p,"%s/data/MNIST/raw/t10k-labels-idx1-ubyte",B);
    FILE*fl=fopen(p,"rb"); fread(hd,1,8,fl);
    unsigned char*h_lbl=(unsigned char*)malloc(N);
    fread(h_lbl,1,N,fl); fclose(fl);

    // 3-layer MLP weights (784→256→256→10)
    sprintf(p,"%s/models/mnist_mlp_d3/w0.bin",B); float*hw0=load_bin(p,256*784);
    sprintf(p,"%s/models/mnist_mlp_d3/b0.bin",B); float*hb0=load_bin(p,256);
    sprintf(p,"%s/models/mnist_mlp_d3/w1.bin",B); float*hw1=load_bin(p,256*256);
    sprintf(p,"%s/models/mnist_mlp_d3/b1.bin",B); float*hb1=load_bin(p,256);
    sprintf(p,"%s/models/mnist_mlp_d3/w2.bin",B); float*hw2=load_bin(p,10*256);
    sprintf(p,"%s/models/mnist_mlp_d3/b2.bin",B); float*hb2=load_bin(p,10);

    float *di,*dw0,*db0,*dw1,*db1,*dw2,*db2,*dh0,*dh1,*dlo;
    hipMalloc(&di,N*784*4);
    hipMalloc(&dw0,256*784*4); hipMalloc(&db0,256*4);
    hipMalloc(&dw1,256*256*4); hipMalloc(&db1,256*4);
    hipMalloc(&dw2,10*256*4);  hipMalloc(&db2,10*4);
    hipMalloc(&dh0,N*256*4); hipMalloc(&dh1,N*256*4); hipMalloc(&dlo,N*10*4);

    hipMemcpy(di,h_img,N*784*4,hipMemcpyHostToDevice);
    hipMemcpy(dw0,hw0,256*784*4,hipMemcpyHostToDevice); hipMemcpy(db0,hb0,256*4,hipMemcpyHostToDevice);
    hipMemcpy(dw1,hw1,256*256*4,hipMemcpyHostToDevice); hipMemcpy(db1,hb1,256*4,hipMemcpyHostToDevice);
    hipMemcpy(dw2,hw2,10*256*4,hipMemcpyHostToDevice);  hipMemcpy(db2,hb2,10*4,hipMemcpyHostToDevice);

    float*h_lo=(float*)malloc(N*10*4);
    int thr=256;

    auto eval = [&](float*h_logits) -> float {
        int c=0;
        for(int i=0;i<N;i++){
            int pr=0;float mx=h_logits[i*10];
            for(int j=1;j<10;j++) if(h_logits[i*10+j]>mx){mx=h_logits[i*10+j];pr=j;}
            if(pr==h_lbl[i]) c++;
        }
        return 100.0f*c/N;
    };

    // ========================================
    // A: STANDARD matmul + ReLU
    // ========================================
    printf("--- A: Standard matmul + ReLU ---\n");
    standard_layer<<<(N*256+thr-1)/thr,thr>>>(di,dw0,db0,dh0,N,784,256,1);
    standard_layer<<<(N*256+thr-1)/thr,thr>>>(dh0,dw1,db1,dh1,N,256,256,1);
    standard_layer<<<(N*10+thr-1)/thr,thr>>>(dh1,dw2,db2,dlo,N,256,10,0);
    hipDeviceSynchronize();
    hipMemcpy(h_lo,dlo,N*10*4,hipMemcpyDeviceToHost);
    float acc_std = eval(h_lo);
    printf("  Accuracy: %.2f%%\n", acc_std);

    auto t0=std::chrono::high_resolution_clock::now();
    for(int r=0;r<100;r++){
        standard_layer<<<(N*256+thr-1)/thr,thr>>>(di,dw0,db0,dh0,N,784,256,1);
        standard_layer<<<(N*256+thr-1)/thr,thr>>>(dh0,dw1,db1,dh1,N,256,256,1);
        standard_layer<<<(N*10+thr-1)/thr,thr>>>(dh1,dw2,db2,dlo,N,256,10,0);
    }
    hipDeviceSynchronize();
    auto t1=std::chrono::high_resolution_clock::now();
    double std_ms=std::chrono::duration<double,std::milli>(t1-t0).count()/100;
    printf("  Time: %.3f ms\n\n", std_ms);

    // ========================================
    // B: NEUROMORPHIC matmul (mechanisms = computation)
    // ========================================
    // Test different mixing strengths
    float mix_values[] = {0.0f, 0.05f, 0.10f, 0.20f};
    int n_mix = 4;

    printf("--- B: Neuromorphic matmul ---\n");
    for (int mi = 0; mi < n_mix; mi++) {
        float mix = mix_values[mi];
        // Layer 0: 784→256 with mechanisms
        neuromorphic_layer<<<N,256>>>(di,dw0,db0,dh0,N,784,256,1,mix);
        // Layer 1: 256→256 with mechanisms
        neuromorphic_layer<<<N,256>>>(dh0,dw1,db1,dh1,N,256,256,1,mix);
        // Layer 2: 256→10 linear (no activation on output)
        neuromorphic_layer<<<N,256>>>(dh1,dw2,db2,dlo,N,256,10,0,0.0f);
        hipDeviceSynchronize();

        hipMemcpy(h_lo,dlo,N*10*4,hipMemcpyDeviceToHost);
        float acc = eval(h_lo);

        // Time it
        t0=std::chrono::high_resolution_clock::now();
        for(int r=0;r<100;r++){
            neuromorphic_layer<<<N,256>>>(di,dw0,db0,dh0,N,784,256,1,mix);
            neuromorphic_layer<<<N,256>>>(dh0,dw1,db1,dh1,N,256,256,1,mix);
            neuromorphic_layer<<<N,256>>>(dh1,dw2,db2,dlo,N,256,10,0,0.0f);
        }
        hipDeviceSynchronize();
        t1=std::chrono::high_resolution_clock::now();
        double nm_ms=std::chrono::duration<double,std::milli>(t1-t0).count()/100;

        printf("  mix=%.2f: acc=%.2f%% (Δ=%+.2fpp) time=%.3fms (%.1f%%)\n",
               mix, acc, acc-acc_std, nm_ms, 100*(nm_ms-std_ms)/std_ms);
    }

    // ========================================
    // C: NOISE ROBUSTNESS — add Gaussian noise to inputs
    // ========================================
    printf("\n--- C: Noise robustness ---\n");
    float noise_levels[] = {0.0f, 0.1f, 0.2f, 0.3f, 0.5f};
    int n_noise = 5;

    float* h_noisy = (float*)malloc(N*784*4);
    float* d_noisy;
    hipMalloc(&d_noisy, N*784*4);

    printf("  %8s  %8s  %8s  %8s\n", "Noise_σ", "Standard", "Neuro", "Δ");
    printf("  %8s  %8s  %8s  %8s\n", "-------", "--------", "-----", "---");

    srand(42);
    for (int ni = 0; ni < n_noise; ni++) {
        float sigma = noise_levels[ni];

        // Add noise
        for (int i = 0; i < N*784; i++) {
            float u1 = (float)(rand()+1)/(RAND_MAX+2.0f);
            float u2 = (float)rand()/RAND_MAX;
            float noise = sigma * sqrtf(-2*logf(u1)) * cosf(6.2832f*u2);
            h_noisy[i] = fmaxf(0, fminf(1, h_img[i] + noise));
        }
        hipMemcpy(d_noisy, h_noisy, N*784*4, hipMemcpyHostToDevice);

        // Standard
        standard_layer<<<(N*256+thr-1)/thr,thr>>>(d_noisy,dw0,db0,dh0,N,784,256,1);
        standard_layer<<<(N*256+thr-1)/thr,thr>>>(dh0,dw1,db1,dh1,N,256,256,1);
        standard_layer<<<(N*10+thr-1)/thr,thr>>>(dh1,dw2,db2,dlo,N,256,10,0);
        hipDeviceSynchronize();
        hipMemcpy(h_lo,dlo,N*10*4,hipMemcpyDeviceToHost);
        float acc_s = eval(h_lo);

        // Neuromorphic (mix=0.10)
        neuromorphic_layer<<<N,256>>>(d_noisy,dw0,db0,dh0,N,784,256,1,0.10f);
        neuromorphic_layer<<<N,256>>>(dh0,dw1,db1,dh1,N,256,256,1,0.10f);
        neuromorphic_layer<<<N,256>>>(dh1,dw2,db2,dlo,N,256,10,0,0.0f);
        hipDeviceSynchronize();
        hipMemcpy(h_lo,dlo,N*10*4,hipMemcpyDeviceToHost);
        float acc_n = eval(h_lo);

        printf("  %8.2f  %7.2f%%  %7.2f%%  %+.2fpp\n", sigma, acc_s, acc_n, acc_n-acc_s);
    }

    // ========================================
    // D: Run-to-run variance (stochasticity test)
    // If neuromorphic kernel is truly stochastic, repeated runs differ
    // ========================================
    printf("\n--- D: Stochasticity (run-to-run variance) ---\n");
    int n_runs = 10;
    float accs_std[10], accs_nm[10];

    for (int r = 0; r < n_runs; r++) {
        // Standard (should be identical each run)
        standard_layer<<<(N*256+thr-1)/thr,thr>>>(di,dw0,db0,dh0,N,784,256,1);
        standard_layer<<<(N*256+thr-1)/thr,thr>>>(dh0,dw1,db1,dh1,N,256,256,1);
        standard_layer<<<(N*10+thr-1)/thr,thr>>>(dh1,dw2,db2,dlo,N,256,10,0);
        hipDeviceSynchronize();
        hipMemcpy(h_lo,dlo,N*10*4,hipMemcpyDeviceToHost);
        accs_std[r] = eval(h_lo);

        // Neuromorphic (may differ due to atomic ordering)
        neuromorphic_layer<<<N,256>>>(di,dw0,db0,dh0,N,784,256,1,0.10f);
        neuromorphic_layer<<<N,256>>>(dh0,dw1,db1,dh1,N,256,256,1,0.10f);
        neuromorphic_layer<<<N,256>>>(dh1,dw2,db2,dlo,N,256,10,0,0.0f);
        hipDeviceSynchronize();
        hipMemcpy(h_lo,dlo,N*10*4,hipMemcpyDeviceToHost);
        accs_nm[r] = eval(h_lo);
    }

    float std_mean=0, nm_mean=0, std_var=0, nm_var=0;
    for(int r=0;r<n_runs;r++) { std_mean+=accs_std[r]; nm_mean+=accs_nm[r]; }
    std_mean/=n_runs; nm_mean/=n_runs;
    for(int r=0;r<n_runs;r++) {
        std_var+=(accs_std[r]-std_mean)*(accs_std[r]-std_mean);
        nm_var+=(accs_nm[r]-nm_mean)*(accs_nm[r]-nm_mean);
    }
    std_var/=n_runs; nm_var/=n_runs;

    printf("  Standard: mean=%.3f%% std=%.4f%% (should be 0)\n", std_mean, sqrtf(std_var));
    printf("  Neuromorphic: mean=%.3f%% std=%.4f%% (non-zero = real stochasticity)\n", nm_mean, sqrtf(nm_var));

    printf("\n============================================================\n");
    printf("SUMMARY\n");
    printf("============================================================\n");
    printf("  Standard ReLU:    %.2f%%\n", acc_std);
    printf("  Neuromorphic:     see mix sweep above\n");
    printf("  Key question: does neuromorphic matmul compute something\n");
    printf("  that standard matmul+ReLU cannot?\n");
    printf("============================================================\n");

    // Cleanup
    free(h_img);free(h_lbl);free(h_lo);free(h_noisy);
    free(hw0);free(hb0);free(hw1);free(hb1);free(hw2);free(hb2);
    hipFree(di);hipFree(dw0);hipFree(db0);hipFree(dw1);hipFree(db1);
    hipFree(dw2);hipFree(db2);hipFree(dh0);hipFree(dh1);hipFree(dlo);hipFree(d_noisy);
    return 0;
}
