/*
 * z2420_isa_reservoir_mnist.hip — ISA-level reservoir for MNIST
 *
 * PIVOT: Stop trying to use shadow signals as sensors.
 * Instead: USE the ISA mechanisms as ACTIVE computation.
 *
 * Compare:
 *   A: Traditional software ESN (tanh + matmul on GPU)
 *   B: ISA reservoir (atomicAdd neurons + __shfl + __ballot + s_setreg)
 *   C: ISA reservoir + fourpop mechanisms (branch div + LDS + timing)
 *
 * All three on same task: MNIST 10-class classification via reservoir readout.
 * Same number of neurons (256), same readout (ridge regression on host).
 *
 * Metrics: accuracy, wall-clock time, energy, memory capacity
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define N_NEURONS 256
#define N_STEPS 100      // reservoir runs per input image
#define N_CLASSES 10
#define N_READOUT (N_NEURONS * 2)  // mean + var per neuron

/* ============================================================
 * A: Traditional ESN — standard tanh reservoir on GPU
 * ============================================================ */
__global__ void esn_reservoir(
    const float* __restrict__ input,  // [batch, 784]
    float* __restrict__ states,       // [batch, N_NEURONS]
    const float* __restrict__ W_in,   // [N_NEURONS, 784] (sparse)
    const float* __restrict__ W_rec,  // [N_NEURONS, N_NEURONS]
    int batch, int in_dim, float leak, float spectral_radius)
{
    int b = blockIdx.x, n = threadIdx.x;
    if (b >= batch || n >= N_NEURONS) return;

    float state = 0.0f;

    // Run reservoir for N_STEPS with input injection
    for (int t = 0; t < N_STEPS; t++) {
        // Input: project input to neuron n (sparse: only use 10 input dims)
        float inp = 0.0f;
        for (int k = 0; k < 10; k++) {
            int idx = (n * 10 + k) % in_dim;
            inp += W_in[n * 10 + k] * input[b * in_dim + idx];
        }

        // Recurrence: connect to 4 neighbors
        int n1 = (n + 1) % N_NEURONS;
        int n2 = (n + N_NEURONS - 1) % N_NEURONS;
        int n3 = (n + 37) % N_NEURONS;
        int n4 = (n + 127) % N_NEURONS;

        // Read neighbor states from global memory
        float rec = W_rec[n * 4 + 0] * states[b * N_NEURONS + n1]
                  + W_rec[n * 4 + 1] * states[b * N_NEURONS + n2]
                  + W_rec[n * 4 + 2] * states[b * N_NEURONS + n3]
                  + W_rec[n * 4 + 3] * states[b * N_NEURONS + n4];

        // Update: leaky tanh
        state = (1 - leak) * state + leak * tanhf(inp + spectral_radius * rec);
    }

    states[b * N_NEURONS + n] = state;
}

/* ============================================================
 * B: ISA reservoir — uses hardware mechanisms as computation
 * ============================================================ */
__global__ void isa_reservoir(
    const float* __restrict__ input,
    float* __restrict__ states,
    int batch, int in_dim, float leak)
{
    int b = blockIdx.x, n = threadIdx.x;
    int lane = n % 32;
    if (b >= batch || n >= N_NEURONS) return;

    float membrane = 0.0f;
    float state_acc = 0.0f;
    uint32_t spike_count = 0;

    // Input weight (deterministic per neuron)
    uint32_t rng = n * 2654435761u + 12345;
    float input_w = ((float)((rng >> 16) & 0xFF) / 255.0f - 0.5f) * 0.2f;
    rng *= 1103515245;
    float threshold = 0.3f + 0.4f * ((rng >> 16) & 0xFF) / 255.0f;

    for (int t = 0; t < N_STEPS; t++) {
        // Input injection (sparse)
        int idx = (n * 7 + t * 3) % in_dim;
        float inp = input[b * in_dim + idx] * input_w;

        // === ISA mechanism #16: timing noise ===
        uint32_t cycles;
        asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(cycles));
        float noise = ((float)(cycles & 0x3F) - 32.0f) * 0.005f;

        // === ISA mechanism #4: __shfl lateral coupling ===
        float left = __shfl(membrane, (lane + 31) % 32);
        float right = __shfl(membrane, (lane + 1) % 32);
        float distant = __shfl(membrane, (lane + 11) % 32);

        // === Leaky integration ===
        membrane = membrane * leak + inp + noise
                 + left * 0.05f + right * 0.05f + distant * (-0.02f);

        // === ISA mechanism #22: population feedback ===
        uint64_t active_mask = __ballot(membrane > threshold * 0.5f);
        uint32_t pop_active = __popcll(active_mask);
        membrane -= (float)pop_active * 0.003f; // lateral inhibition

        // === Spike detection ===
        if (membrane > threshold) {
            spike_count++;
            membrane = 0.0f;

            // === ISA mechanism #8: s_sleep refractory ===
            asm volatile("s_sleep 1");
        }

        // === ISA mechanism #12: s_setreg adaptive rounding ===
        // Changes how FP accumulation works based on neuron state
        if ((t & 0xF) == 0) {
            if (membrane > threshold * 0.7f)
                asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x1"); // R+inf
            else
                asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x0"); // RNE
        }

        state_acc += membrane;
    }

    // Restore rounding
    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x0");

    states[b * N_NEURONS + n] = state_acc / N_STEPS; // mean state as readout
}

/* ============================================================
 * C: ISA + fourpop — adds branch divergence + LDS + timing mechanisms
 * ============================================================ */
__global__ void isa_fourpop_reservoir(
    const float* __restrict__ input,
    float* __restrict__ states,
    int batch, int in_dim, float leak)
{
    int b = blockIdx.x, n = threadIdx.x;
    int lane = n % 32;
    int pop = n / 64;  // 4 populations of 64
    if (b >= batch || n >= N_NEURONS) return;

    __shared__ float lds_state[N_NEURONS];
    lds_state[n] = 0.0f;
    __syncthreads();

    float membrane = 0.0f;
    float state_acc = 0.0f;
    uint32_t spike_count = 0;

    uint32_t rng = n * 2654435761u + 12345;
    float input_w = ((float)((rng >> 16) & 0xFF) / 255.0f - 0.5f) * 0.2f;
    rng *= 1103515245;
    float threshold = 0.3f + 0.4f * ((rng >> 16) & 0xFF) / 255.0f;

    float dummy = 1.0001f + n * 0.0001f;

    for (int t = 0; t < N_STEPS; t++) {
        int idx = (n * 7 + t * 3) % in_dim;
        float inp = input[b * in_dim + idx] * input_w;

        float hw_signal = 0.0f;

        // === Pop A (n 0-63): Branch divergence → memory ===
        if (pop == 0) {
            float bthr = 0.5f + 0.3f * ((n & 0xF) / 15.0f);
            if (membrane > bthr) {
                for (int i = 0; i < 10; i++)
                    dummy = __fmaf_rn(dummy, 1.000001f, 0.0001f);
                hw_signal = 0.04f;
            } else {
                uint32_t iv = __float_as_uint(dummy);
                iv = iv * 2654435761u + 1;
                dummy = __uint_as_float((iv & 0x007FFFFF) | 0x3F800000);
                hw_signal = -0.04f;
            }
        }

        // === Pop B (n 64-127): Nonlinear XOR via product terms ===
        if (pop == 1) {
            float sign_u = (inp > 0) ? 1.0f : -1.0f;
            hw_signal = membrane * sign_u * 0.08f;
        }

        // === Pop C (n 128-191): Dense ESN (strong recurrence) ===
        if (pop == 2) {
            // Read from LDS (all populations)
            int ln1 = (n + 1) % N_NEURONS;
            int ln2 = (n + N_NEURONS - 1) % N_NEURONS;
            int ln3 = (n + 37) % N_NEURONS;
            hw_signal = lds_state[ln1] * 0.04f + lds_state[ln2] * 0.04f
                      + lds_state[ln3] * (-0.02f);
        }

        // === Pop D (n 192-255): Wavefront timing ===
        if (pop == 3) {
            uint32_t cycles;
            asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(cycles));
            hw_signal = ((float)(cycles & 0xFF) - 128.0f) * 0.003f;
        }

        // === Shared: shfl coupling within wave ===
        float left = __shfl(membrane, (lane + 31) % 32);
        float right = __shfl(membrane, (lane + 1) % 32);
        membrane = membrane * leak + inp + hw_signal
                 + left * 0.03f + right * 0.03f;

        // === Population coding ===
        uint64_t active_mask = __ballot(membrane > threshold * 0.5f);
        uint32_t pop_active = __popcll(active_mask);
        membrane -= (float)pop_active * 0.002f;

        // === Spike ===
        if (membrane > threshold) {
            spike_count++;
            membrane = 0.0f;
            asm volatile("s_sleep 1");
        }

        // === s_setreg ===
        if ((t & 0xF) == 0) {
            if (membrane > threshold * 0.7f)
                asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x1");
            else
                asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x0");
        }

        state_acc += membrane;
        lds_state[n] = membrane;
        __syncthreads();
    }

    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 2), 0x0");
    states[b * N_NEURONS + n] = state_acc / N_STEPS;
}

/* ============================================================ */
__global__ void argmax_k(const float*L,int*P,int n,int c){
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=n)return;
    float best=-1e30;int bc=0;for(int i=0;i<c;i++){float v=L[b*c+i];if(v>best){best=v;bc=i;}}P[b]=bc;}

double get_ms(){struct timeval t;gettimeofday(&t,NULL);return t.tv_sec*1e3+t.tv_usec/1e3;}
float read_power(){for(int h=7;h<=10;h++){char p[64];sprintf(p,"/sys/class/hwmon/hwmon%d/power1_average",h);
    FILE*f=fopen(p,"r");if(f){long v;fscanf(f,"%ld",&v);fclose(f);return v/1e6;}}return -1;}

// Ridge regression readout (host)
void ridge_readout(float*X, int*y, int n_train, int n_test, int n_feat,
                   float alpha, float*accuracy, float*mc) {
    int n_total = n_train + n_test;
    // Simple: compute W = (XtX + αI)^-1 XtY via Gauss elimination
    // For speed, use only first 64 features
    int nf = (n_feat > 64) ? 64 : n_feat;

    float*XtX = (float*)calloc(nf*nf, sizeof(float));
    float*XtY = (float*)calloc(nf*N_CLASSES, sizeof(float));

    for(int i=0;i<nf;i++){
        for(int j=0;j<nf;j++){
            float s=0;for(int k=0;k<n_train;k++)s+=X[k*n_feat+i]*X[k*n_feat+j];
            XtX[i*nf+j]=s+(i==j?alpha:0);}
        for(int c=0;c<N_CLASSES;c++){
            float s=0;for(int k=0;k<n_train;k++)s+=X[k*n_feat+i]*(y[k]==c?1:0);
            XtY[i*N_CLASSES+c]=s;}}

    // Gauss elimination
    float*aug=(float*)calloc(nf*(nf+N_CLASSES),sizeof(float));
    int stride=nf+N_CLASSES;
    for(int i=0;i<nf;i++){
        for(int j=0;j<nf;j++)aug[i*stride+j]=XtX[i*nf+j];
        for(int c=0;c<N_CLASSES;c++)aug[i*stride+nf+c]=XtY[i*N_CLASSES+c];}

    for(int col=0;col<nf;col++){
        int piv=col;float mx=fabsf(aug[col*stride+col]);
        for(int r=col+1;r<nf;r++)if(fabsf(aug[r*stride+col])>mx){mx=fabsf(aug[r*stride+col]);piv=r;}
        if(piv!=col)for(int j=0;j<stride;j++){float t=aug[col*stride+j];aug[col*stride+j]=aug[piv*stride+j];aug[piv*stride+j]=t;}
        float d=aug[col*stride+col];if(fabsf(d)<1e-10)continue;
        for(int r=0;r<nf;r++){if(r==col)continue;float f=aug[r*stride+col]/d;
            for(int j=col;j<stride;j++)aug[r*stride+j]-=f*aug[col*stride+j];}}

    float*W=(float*)calloc(nf*N_CLASSES,sizeof(float));
    for(int i=0;i<nf;i++){float d=aug[i*stride+i];
        for(int c=0;c<N_CLASSES;c++)W[i*N_CLASSES+c]=aug[i*stride+nf+c]/d;}

    // Test accuracy
    int corr=0;
    for(int k=n_train;k<n_total;k++){
        int bc=0;float bv=-1e30;
        for(int c=0;c<N_CLASSES;c++){float s=0;
            for(int i=0;i<nf;i++)s+=X[k*n_feat+i]*W[i*N_CLASSES+c];
            if(s>bv){bv=s;bc=c;}}
        if(bc==y[k])corr++;}
    *accuracy=100.0f*corr/n_test;

    // Memory capacity (simple: R² of first feature predicting delayed input)
    *mc = 0; // placeholder

    free(XtX);free(XtY);free(aug);free(W);
}

int main(){
    printf("================================================================\n");
    printf("z2420: ISA RESERVOIR vs SOFTWARE ESN on MNIST\n");
    printf("================================================================\n\n");

    const char*base="/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy";
    char p[512];

    // Load MNIST
    sprintf(p,"%s/data/MNIST/raw/t10k-images-idx3-ubyte",base);
    FILE*f=fopen(p,"rb");uint32_t hdr[4];fread(hdr,4,4,f);
    int N=__builtin_bswap32(hdr[1]);
    float*h_img=(float*)malloc(N*784*4);
    uint8_t*raw=(uint8_t*)malloc(N*784);fread(raw,1,N*784,f);
    for(int i=0;i<N*784;i++)h_img[i]=raw[i]/255.0f;fclose(f);
    sprintf(p,"%s/data/MNIST/raw/t10k-labels-idx1-ubyte",base);
    f=fopen(p,"rb");fread(hdr,4,2,f);int*h_lbl=(int*)malloc(N*4);
    raw=(uint8_t*)realloc(raw,N);fread(raw,1,N,f);
    for(int i=0;i<N;i++)h_lbl[i]=raw[i];free(raw);fclose(f);

    int BS = 2000; // batch size
    int n_train = 1400, n_test = 600;
    printf("MNIST: %d samples, batch=%d, train=%d, test=%d\n", N, BS, n_train, n_test);
    printf("Reservoir: %d neurons × %d steps\n\n", N_NEURONS, N_STEPS);

    // Init ESN weights
    float*h_Win=(float*)malloc(N_NEURONS*10*4);
    float*h_Wrec=(float*)malloc(N_NEURONS*4*4);
    srand(42);
    for(int i=0;i<N_NEURONS*10;i++) h_Win[i]=((float)rand()/RAND_MAX-0.5f)*0.4f;
    for(int i=0;i<N_NEURONS*4;i++) h_Wrec[i]=((float)rand()/RAND_MAX-0.5f)*0.2f;

    float*d_input,*d_states,*d_Win,*d_Wrec;
    hipMalloc(&d_input,BS*784*4);
    hipMalloc(&d_states,BS*N_NEURONS*4);
    hipMalloc(&d_Win,N_NEURONS*10*4);
    hipMalloc(&d_Wrec,N_NEURONS*4*4);
    hipMemcpy(d_Win,h_Win,N_NEURONS*10*4,hipMemcpyHostToDevice);
    hipMemcpy(d_Wrec,h_Wrec,N_NEURONS*4*4,hipMemcpyHostToDevice);

    float*h_states=(float*)malloc(BS*N_NEURONS*4);
    hipMemcpy(d_input,h_img,BS*784*4,hipMemcpyHostToDevice);

    // Test three reservoir types
    struct {
        const char*name;
        int type; // 0=ESN, 1=ISA, 2=ISA+fourpop
    } reservoirs[] = {
        {"A: Software ESN (tanh)", 0},
        {"B: ISA reservoir", 1},
        {"C: ISA + fourpop", 2},
    };

    printf("%-25s %8s %8s %8s %10s\n", "Reservoir", "Acc%", "Time(ms)", "Power(W)", "Energy(mJ)");

    for(int ri=0;ri<3;ri++){
        // Run reservoir
        hipMemset(d_states,0,BS*N_NEURONS*4);
        hipDeviceSynchronize();

        float pw0=read_power();
        double t0=get_ms();
        int REPS=5;
        for(int r=0;r<REPS;r++){
            if(reservoirs[ri].type==0)
                esn_reservoir<<<BS,N_NEURONS>>>(d_input,d_states,d_Win,d_Wrec,BS,784,0.3f,0.9f);
            else if(reservoirs[ri].type==1)
                isa_reservoir<<<BS,N_NEURONS>>>(d_input,d_states,BS,784,0.85f);
            else
                isa_fourpop_reservoir<<<BS,N_NEURONS>>>(d_input,d_states,BS,784,0.85f);
            hipDeviceSynchronize();
        }
        double dt=(get_ms()-t0)/REPS;
        float pw1=read_power();
        float avg_pw=(pw0+pw1)/2;

        hipMemcpy(h_states,d_states,BS*N_NEURONS*4,hipMemcpyDeviceToHost);

        // Ridge regression readout
        float acc, mc;
        ridge_readout(h_states, h_lbl, n_train, n_test, N_NEURONS, 1.0f, &acc, &mc);

        printf("%-25s %7.1f%% %8.2f %8.1f %9.2f\n",
               reservoirs[ri].name, acc, dt, avg_pw, avg_pw*dt);
    }

    // Also run the pre-trained MLP for comparison
    printf("\n--- MLP comparison (trained, not reservoir) ---\n");
    {
        sprintf(p,"%s/models/mnist_mlp",base);
        float*w1=(float*)malloc(128*784*4);float*b1=(float*)malloc(128*4);
        float*w2=(float*)malloc(64*128*4);float*b2=(float*)malloc(64*4);
        float*w3=(float*)malloc(10*64*4);float*b3=(float*)malloc(10*4);
        sprintf(p,"%s/models/mnist_mlp/w1.bin",base);f=fopen(p,"rb");fread(w1,4,128*784,f);fclose(f);
        sprintf(p,"%s/models/mnist_mlp/b1.bin",base);f=fopen(p,"rb");fread(b1,4,128,f);fclose(f);
        sprintf(p,"%s/models/mnist_mlp/w2.bin",base);f=fopen(p,"rb");fread(w2,4,64*128,f);fclose(f);
        sprintf(p,"%s/models/mnist_mlp/b2.bin",base);f=fopen(p,"rb");fread(b2,4,64,f);fclose(f);
        sprintf(p,"%s/models/mnist_mlp/w3.bin",base);f=fopen(p,"rb");fread(w3,4,10*64,f);fclose(f);
        sprintf(p,"%s/models/mnist_mlp/b3.bin",base);f=fopen(p,"rb");fread(b3,4,10,f);fclose(f);

        // CPU inference on test set
        int corr=0;
        for(int i=n_train;i<n_train+n_test;i++){
            float h1[128],h2[64],out[10];
            for(int o=0;o<128;o++){float a=b1[o];for(int k=0;k<784;k++)a+=w1[o*784+k]*h_img[i*784+k];h1[o]=fmaxf(0,a);}
            for(int o=0;o<64;o++){float a=b2[o];for(int k=0;k<128;k++)a+=w2[o*128+k]*h1[k];h2[o]=fmaxf(0,a);}
            for(int o=0;o<10;o++){float a=b3[o];for(int k=0;k<64;k++)a+=w3[o*64+k]*h2[k];out[o]=a;}
            int bc=0;float bv=-1e30;for(int c=0;c<10;c++)if(out[c]>bv){bv=out[c];bc=c;}
            if(bc==h_lbl[i])corr++;
        }
        printf("  Trained MLP (3L):  %.1f%% (on same test split)\n",100.0f*corr/n_test);
        free(w1);free(b1);free(w2);free(b2);free(w3);free(b3);
    }

    hipFree(d_input);hipFree(d_states);hipFree(d_Win);hipFree(d_Wrec);
    free(h_img);free(h_lbl);free(h_Win);free(h_Wrec);free(h_states);

    printf("\n=== z2420 COMPLETE ===\n");
    return 0;
}
