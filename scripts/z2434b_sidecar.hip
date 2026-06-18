/**
 * z2434b: GEMM Sidecar — clean version with correct matrix layout
 *
 * Uses our own GEMM kernel (proven working from z2429b) for correctness,
 * then adds sidecar mechanism experts on a separate stream.
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

#define SIDECAR_DIM 8

// Simple GEMM layer: Y[sample, out] = ReLU(X[sample, in] @ W[out, in]^T + b[out])
__global__ void gemm_layer(
    const float* __restrict__ X, const float* __restrict__ W, const float* __restrict__ b,
    float* __restrict__ Y, int N, int in_dim, int out_dim, int use_relu
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample = idx / out_dim;
    int neuron = idx % out_dim;
    if (sample >= N) return;

    float sum = b[neuron];
    for (int k = 0; k < in_dim; k++)
        sum += W[neuron * in_dim + k] * X[sample * in_dim + k];
    if (use_relu && sum < 0.0f) sum = 0.0f;
    Y[sample * out_dim + neuron] = sum;
}

// Sidecar: 4-family mechanism experts reading GEMM activations
__global__ void sidecar_experts(
    const float* __restrict__ activations, float* __restrict__ ctrl,
    int N, int dim
) {
    int sample = blockIdx.x;
    if (sample >= N) return;
    int tid = threadIdx.x;

    const float* row = activations + sample * dim;

    float local_sum = 0, local_abs = 0, local_sq = 0;
    float branch_st = 0;
    int zeros = 0;

    for (int j = tid; j < dim; j += blockDim.x) {
        float v = row[j];
        local_sum += v;
        local_abs += fabsf(v);
        local_sq += v * v;
        if (v > branch_st) branch_st = branch_st * 0.8f + v * 0.2f;
        else branch_st = branch_st * 0.95f + v * 0.05f;
        if (v == 0.0f) zeros++;
    }

    // MEMORY: branch state + cross-lane persistence
    float mem0 = branch_st;
    float mem1 = __shfl_xor(branch_st, 1) + __shfl_xor(branch_st, 2);

    // NONLINEARITY: atomic contention + sparsity
    __shared__ float s_nl[2];
    if (tid < 2) s_nl[tid] = 0.0f;
    __syncthreads();
    if (local_abs > 0.01f * dim) atomicAdd(&s_nl[0], local_abs);
    atomicAdd(&s_nl[1], (float)zeros);
    __syncthreads();

    // SLOW CONTEXT: sum + variance
    float mean = local_sum / (dim / (float)blockDim.x + 1e-10f);
    float var = local_sq / (dim / (float)blockDim.x + 1e-10f) - mean * mean;

    // IDENTITY: ballot agreement + diversity
    unsigned long long ballot = __ballot(local_sum > 0.0f);
    float id_agree = (float)__popcll(ballot) / 64.0f;
    float id_div = fabsf(local_sum - __shfl_xor(local_sum, 1));

    if (tid == 0) {
        float* out = ctrl + sample * SIDECAR_DIM;
        out[0] = mem0;
        out[1] = mem1 / (dim + 1e-10f);
        out[2] = s_nl[0] / (N + 1e-10f);
        out[3] = s_nl[1] / blockDim.x;
        out[4] = local_sum;
        out[5] = var;
        out[6] = id_agree;
        out[7] = id_div;
    }
}

float* load_bin(const char* p, int n) {
    FILE* f = fopen(p, "rb");
    if (!f) { printf("ERR: %s\n", p); exit(1); }
    float* d = (float*)malloc(n*4); fread(d, 4, n, f); fclose(f); return d;
}

int main() {
    printf("============================================================\n");
    printf("z2434b: GEMM SIDECAR — 4 expert families\n");
    printf("============================================================\n\n");

    const char* B = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy";
    char p[512];

    // MNIST
    sprintf(p, "%s/data/MNIST/raw/t10k-images-idx3-ubyte", B);
    FILE* fi = fopen(p, "rb"); unsigned char hd[16]; fread(hd,1,16,fi);
    int N = 10000;
    unsigned char* raw = (unsigned char*)malloc(N*784);
    fread(raw,1,N*784,fi); fclose(fi);
    float* h_img = (float*)malloc(N*784*4);
    for (int i = 0; i < N*784; i++) h_img[i] = raw[i]/255.0f;
    free(raw);

    sprintf(p, "%s/data/MNIST/raw/t10k-labels-idx1-ubyte", B);
    FILE* fl = fopen(p, "rb"); fread(hd,1,8,fl);
    unsigned char* h_lbl = (unsigned char*)malloc(N);
    fread(h_lbl,1,N,fl); fclose(fl);

    sprintf(p,"%s/models/mnist_mlp/w1.bin",B); float* hw1=load_bin(p,128*784);
    sprintf(p,"%s/models/mnist_mlp/b1.bin",B); float* hb1=load_bin(p,128);
    sprintf(p,"%s/models/mnist_mlp/w2.bin",B); float* hw2=load_bin(p,64*128);
    sprintf(p,"%s/models/mnist_mlp/b2.bin",B); float* hb2=load_bin(p,64);
    sprintf(p,"%s/models/mnist_mlp/w3.bin",B); float* hw3=load_bin(p,10*64);
    sprintf(p,"%s/models/mnist_mlp/b3.bin",B); float* hb3=load_bin(p,10);

    float *di,*dw1,*db1,*dw2,*db2,*dw3,*db3,*dh1,*dh2,*dlo;
    float *dc1,*dc2,*dc3;
    hipMalloc(&di,N*784*4);
    hipMalloc(&dw1,128*784*4); hipMalloc(&db1,128*4);
    hipMalloc(&dw2,64*128*4);  hipMalloc(&db2,64*4);
    hipMalloc(&dw3,10*64*4);   hipMalloc(&db3,10*4);
    hipMalloc(&dh1,N*128*4); hipMalloc(&dh2,N*64*4); hipMalloc(&dlo,N*10*4);
    hipMalloc(&dc1,N*SIDECAR_DIM*4); hipMalloc(&dc2,N*SIDECAR_DIM*4); hipMalloc(&dc3,N*SIDECAR_DIM*4);

    hipMemcpy(di,h_img,N*784*4,hipMemcpyHostToDevice);
    hipMemcpy(dw1,hw1,128*784*4,hipMemcpyHostToDevice); hipMemcpy(db1,hb1,128*4,hipMemcpyHostToDevice);
    hipMemcpy(dw2,hw2,64*128*4,hipMemcpyHostToDevice);  hipMemcpy(db2,hb2,64*4,hipMemcpyHostToDevice);
    hipMemcpy(dw3,hw3,10*64*4,hipMemcpyHostToDevice);   hipMemcpy(db3,hb3,10*4,hipMemcpyHostToDevice);

    hipStream_t sg, ss;
    hipStreamCreate(&sg); hipStreamCreate(&ss);
    hipEvent_t e1,e2,e3;
    hipEventCreate(&e1); hipEventCreate(&e2); hipEventCreate(&e3);

    int thr = 256;

    // === BASELINE: GEMM only ===
    printf("--- Baseline ---\n");
    gemm_layer<<<(N*128+thr-1)/thr, thr, 0, sg>>>(di,dw1,db1,dh1,N,784,128,1);
    gemm_layer<<<(N*64+thr-1)/thr, thr, 0, sg>>>(dh1,dw2,db2,dh2,N,128,64,1);
    gemm_layer<<<(N*10+thr-1)/thr, thr, 0, sg>>>(dh2,dw3,db3,dlo,N,64,10,0);
    hipDeviceSynchronize();

    float* hlo = (float*)malloc(N*10*4);
    hipMemcpy(hlo,dlo,N*10*4,hipMemcpyDeviceToHost);
    int corr_base = 0;
    for (int i=0;i<N;i++) {
        int pr=0; float mx=hlo[i*10];
        for (int j=1;j<10;j++) if(hlo[i*10+j]>mx){mx=hlo[i*10+j];pr=j;}
        if(pr==h_lbl[i]) corr_base++;
    }
    printf("  Accuracy: %.2f%%\n", 100.0*corr_base/N);

    // Timing baseline
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r=0;r<100;r++) {
        gemm_layer<<<(N*128+thr-1)/thr, thr, 0, sg>>>(di,dw1,db1,dh1,N,784,128,1);
        gemm_layer<<<(N*64+thr-1)/thr, thr, 0, sg>>>(dh1,dw2,db2,dh2,N,128,64,1);
        gemm_layer<<<(N*10+thr-1)/thr, thr, 0, sg>>>(dh2,dw3,db3,dlo,N,64,10,0);
    }
    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double base_ms = std::chrono::duration<double,std::milli>(t1-t0).count()/100;

    // === GEMM + SIDECAR overlapped ===
    printf("\n--- GEMM + Sidecar ---\n");
    t0 = std::chrono::high_resolution_clock::now();
    for (int r=0;r<100;r++) {
        gemm_layer<<<(N*128+thr-1)/thr, thr, 0, sg>>>(di,dw1,db1,dh1,N,784,128,1);
        hipEventRecord(e1, sg);
        hipStreamWaitEvent(ss, e1, 0);
        sidecar_experts<<<N, 128, 0, ss>>>(dh1, dc1, N, 128);

        gemm_layer<<<(N*64+thr-1)/thr, thr, 0, sg>>>(dh1,dw2,db2,dh2,N,128,64,1);
        hipEventRecord(e2, sg);
        hipStreamWaitEvent(ss, e2, 0);
        sidecar_experts<<<N, 64, 0, ss>>>(dh2, dc2, N, 64);

        gemm_layer<<<(N*10+thr-1)/thr, thr, 0, sg>>>(dh2,dw3,db3,dlo,N,64,10,0);
        hipEventRecord(e3, sg);
        hipStreamWaitEvent(ss, e3, 0);
        sidecar_experts<<<N, 64, 0, ss>>>(dlo, dc3, N, 10);
    }
    hipDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    double side_ms = std::chrono::duration<double,std::milli>(t1-t0).count()/100;

    printf("  GEMM only:      %.3f ms\n", base_ms);
    printf("  GEMM+sidecar:   %.3f ms\n", side_ms);
    printf("  Overhead:       %.1f%%\n", 100*(side_ms-base_ms)/base_ms);

    // Read sidecar features
    float *hc1=(float*)malloc(N*8*4), *hc2=(float*)malloc(N*8*4), *hc3=(float*)malloc(N*8*4);
    hipMemcpy(hc1,dc1,N*8*4,hipMemcpyDeviceToHost);
    hipMemcpy(hc2,dc2,N*8*4,hipMemcpyDeviceToHost);
    hipMemcpy(hc3,dc3,N*8*4,hipMemcpyDeviceToHost);

    // === Signal analysis ===
    printf("\n--- Signal Analysis: correct vs incorrect ---\n");
    const char* fn[]={"Mem:branch","Mem:persist","NL:contention","NL:sparsity",
                       "Ctx:sum","Ctx:var","ID:agree","ID:diverse"};

    for (int L=0;L<3;L++) {
        float* hc = (L==0)?hc1:(L==1)?hc2:hc3;
        int dim = (L==0)?128:(L==1)?64:10;
        printf("  Layer %d (dim=%d):\n", L+1, dim);
        for (int f=0;f<8;f++) {
            double sc=0,si=0,sqc=0,sqi=0; int nc=0,ni=0;
            for (int i=0;i<N;i++) {
                float v = hc[i*8+f];
                int pr=0; float mx=hlo[i*10];
                for (int j=1;j<10;j++) if(hlo[i*10+j]>mx){mx=hlo[i*10+j];pr=j;}
                if (pr==h_lbl[i]) { sc+=v; sqc+=v*v; nc++; }
                else { si+=v; sqi+=v*v; ni++; }
            }
            double mc=sc/(nc+1e-10), mi=si/(ni+1e-10);
            double vc=sqc/(nc+1e-10)-mc*mc, vi=sqi/(ni+1e-10)-mi*mi;
            double pv=(vc*nc+vi*ni)/(nc+ni+1e-10);
            double d=fabs(mc-mi)/sqrt(pv+1e-10);
            if (d>0.05)
                printf("    %14s: corr=%.4f incorr=%.4f d=%.3f %s\n",
                       fn[f],mc,mi,d, d>0.5?"★★★":d>0.2?"★★":"★");
        }
    }

    // === Ridge fusion ===
    printf("\n--- Ridge Fusion ---\n");
    int ntr=5000, nte=N-ntr, FD=8*3, CD=10+FD;
    float *Xtr=(float*)calloc(ntr*CD,4), *Xte=(float*)calloc(nte*CD,4);

    for (int i=0;i<N;i++) {
        float *dst = (i<ntr) ? Xtr+i*CD : Xte+(i-ntr)*CD;
        for (int j=0;j<10;j++) dst[j] = hlo[i*10+j];
        for (int j=0;j<8;j++) dst[10+j] = ((i<ntr)?hc1:hc1)[i*8+j];
        for (int j=0;j<8;j++) dst[18+j] = ((i<ntr)?hc2:hc2)[i*8+j];
        for (int j=0;j<8;j++) dst[26+j] = ((i<ntr)?hc3:hc3)[i*8+j];
    }

    // Normalize sidecar features
    for (int j=10;j<CD;j++) {
        double sum=0,sq=0;
        for (int i=0;i<ntr;i++) { sum+=Xtr[i*CD+j]; sq+=Xtr[i*CD+j]*Xtr[i*CD+j]; }
        double mn=sum/ntr, sd=sqrt(sq/ntr-mn*mn+1e-10);
        if (sd < 1e-6) { for(int i=0;i<ntr;i++) Xtr[i*CD+j]=0; for(int i=0;i<nte;i++) Xte[i*CD+j]=0; }
        else { for(int i=0;i<ntr;i++) Xtr[i*CD+j]=(Xtr[i*CD+j]-mn)/sd; for(int i=0;i<nte;i++) Xte[i*CD+j]=(Xte[i*CD+j]-mn)/sd; }
    }

    // Ridge: W = (XtX + αI)^{-1} XtY via Gauss elim
    float *XtX=(float*)calloc(CD*CD,4), *XtY=(float*)calloc(CD*10,4);
    for (int i=0;i<ntr;i++) {
        float *x=Xtr+i*CD; int lb=h_lbl[i];
        for (int a=0;a<CD;a++) { for(int b=0;b<CD;b++) XtX[a*CD+b]+=x[a]*x[b]; XtY[a*10+lb]+=x[a]; }
    }
    for (int d=0;d<CD;d++) XtX[d*CD+d]+=1.0f;

    float *A=(float*)malloc(CD*(CD+10)*4);
    for (int i=0;i<CD;i++) { for(int j=0;j<CD;j++) A[i*(CD+10)+j]=XtX[i*CD+j]; for(int j=0;j<10;j++) A[i*(CD+10)+CD+j]=XtY[i*10+j]; }
    for (int i=0;i<CD;i++) {
        float piv=A[i*(CD+10)+i]; if(fabsf(piv)<1e-10f) continue;
        for (int j=i;j<CD+10;j++) A[i*(CD+10)+j]/=piv;
        for (int k=0;k<CD;k++) { if(k==i) continue; float f=A[k*(CD+10)+i]; for(int j=i;j<CD+10;j++) A[k*(CD+10)+j]-=f*A[i*(CD+10)+j]; }
    }

    int corr_lo=0, corr_co=0;
    for (int i=0;i<nte;i++) {
        int ii=i+ntr;
        // Logits argmax
        int pb=0; float mb=hlo[ii*10]; for(int j=1;j<10;j++) if(hlo[ii*10+j]>mb){mb=hlo[ii*10+j];pb=j;}
        if(pb==h_lbl[ii]) corr_lo++;
        // Ridge combo
        float sc[10]={0}; float *x=Xte+i*CD;
        for(int j=0;j<10;j++) for(int k=0;k<CD;k++) sc[j]+=x[k]*A[k*(CD+10)+CD+j];
        int pc=0; float mc=sc[0]; for(int j=1;j<10;j++) if(sc[j]>mc){mc=sc[j];pc=j;}
        if(pc==h_lbl[ii]) corr_co++;
    }

    float acc_lo=100.0f*corr_lo/nte, acc_co=100.0f*corr_co/nte;
    printf("  Logits argmax:    %.2f%%\n", acc_lo);
    printf("  Logits+sidecar:   %.2f%%\n", acc_co);
    printf("  Delta:            %+.2fpp (%+d)\n", acc_co-acc_lo, corr_co-corr_lo);

    printf("\n============================================================\n");
    printf("  GEMM accuracy:    %.2f%%\n", 100.0*corr_base/N);
    printf("  +sidecar ridge:   %.2f%%\n", acc_co);
    printf("  Overhead:         %.1f%%\n", 100*(side_ms-base_ms)/base_ms);
    if (acc_co > acc_lo + 0.05) printf("  >>> SIDECAR IMPROVES by %+.2fpp at %.1f%% overhead <<<\n", acc_co-acc_lo, 100*(side_ms-base_ms)/base_ms);
    printf("============================================================\n");

    // Cleanup
    hipFree(di);hipFree(dw1);hipFree(db1);hipFree(dw2);hipFree(db2);hipFree(dw3);hipFree(db3);
    hipFree(dh1);hipFree(dh2);hipFree(dlo);hipFree(dc1);hipFree(dc2);hipFree(dc3);
    free(h_img);free(h_lbl);free(hlo);free(hc1);free(hc2);free(hc3);
    free(hw1);free(hb1);free(hw2);free(hb2);free(hw3);free(hb3);
    free(Xtr);free(Xte);free(XtX);free(XtY);free(A);
    return 0;
}
