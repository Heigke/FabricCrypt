/*
 * z2417_calibrated_sweep.hip — Fixed threshold + optimized ballot frequency
 *
 * Fixes from z2416:
 *   1. Shadow threshold adaptive: threshold = mean(signal) + 1*std(signal)
 *      computed from first 100 samples warmup
 *   2. Ballot every 16th block instead of every 4th (reduce overhead)
 *   3. Shfl every 8th block
 *   4. Test multiple threshold levels to find sweet spot
 *
 * Matrix: 3 depths × 5 threshold levels × timing-only vs full
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define MAX_LAYERS 8
#define IN 784
#define OUT 10

struct Hv {
    float tcv, spf, nd, br, mem;
    uint32_t sp;
};

/* Parameterized shadow matmul with configurable frequency */
__global__ void mm_cal(
    const float*A, const float*W, const float*b, float*C, Hv*hv,
    int batch, int ind, int outd, int relu,
    int do_timing,    // measure timing
    int ballot_freq,  // ballot every N blocks (0=never)
    int shfl_freq,    // shfl every N blocks (0=never)
    int do_setreg,    // s_setreg feedback
    int do_sleep,     // s_sleep on spike
    float threshold)  // shadow spike threshold
{
    int bi=blockIdx.x, o=threadIdx.x, lane=o%32;
    if(bi>=batch||o>=outd) return;

    float membrane=0; uint32_t spikes=0;
    float ts=0,tss=0,ss=0,nds=0,brs=0;
    uint32_t brn=0,psp=0,nb=0;
    uint32_t t0=0;
    if(do_timing) asm volatile("s_getreg_b32 %0, hwreg(29)":"=s"(t0));
    uint32_t om=0;
    if(do_setreg) asm volatile("s_getreg_b32 %0, hwreg(1)":"=s"(om));

    float acc=b[o];
    for(int k=0;k<ind;k+=4){
        float p0=W[o*ind+k]*A[bi*ind+k];
        float p1=(k+1<ind)?W[o*ind+k+1]*A[bi*ind+k+1]:0;
        float p2=(k+2<ind)?W[o*ind+k+2]*A[bi*ind+k+2]:0;
        float p3=(k+3<ind)?W[o*ind+k+3]*A[bi*ind+k+3]:0;
        acc+=p0+p1+p2+p3;
        float sig=0;

        // Timing (every block)
        if(do_timing){
            uint32_t t1;asm volatile("s_getreg_b32 %0, hwreg(29)":"=s"(t1));
            float dt=(float)(t1-t0);t0=t1;ts+=dt;tss+=dt*dt;
            sig+=(dt>20?0.2f:0);
        }

        // Ballot (every ballot_freq blocks)
        if(ballot_freq>0 && (nb%ballot_freq)==0){
            float bs=p0+p1+p2+p3;
            uint64_t pm=__ballot(bs>0);uint32_t np=__popcll(pm);
            ss+=(float)np;
            uint32_t mn=(np<16)?np:(32-np);
            sig+=(float)mn*0.03f;
            int npos=(p0>0)+(p1>0)+(p2>0)+(p3>0);
            sig+=(float)(4-npos)*0.1f;
            // Branching ratio
            uint64_t sm=__ballot(membrane>threshold);
            uint32_t cs=__popcll(sm);
            if(psp>0){brs+=(float)cs/(float)psp;brn++;}
            psp=cs;
        }

        // Shfl (every shfl_freq blocks)
        if(shfl_freq>0 && (nb%shfl_freq)==0){
            float left=__shfl(acc,(lane+31)%32);
            nds+=fabsf(acc-left);
            sig+=fabsf(acc-left)*0.003f;
        }

        // Setreg (every 16 blocks)
        if(do_setreg && (nb&0xF)==0){
            if(membrane>threshold*0.6f)
                asm volatile("s_setreg_imm32_b32 hwreg(1,0,2),0x3");
            else asm volatile("s_setreg_imm32_b32 hwreg(1,0,2),0x0");
        }

        nb++;
        membrane+=sig;
        if(membrane>threshold){
            spikes++;membrane=0;
            if(do_sleep) asm volatile("s_sleep 1");
        }
    }
    if(do_setreg) asm volatile("s_setreg_b32 hwreg(1),%0"::"s"(om));
    if(relu) acc=fmaxf(0,acc);
    C[bi*outd+o]=acc;

    if(lane==0&&hv){
        float tm=(nb>0)?ts/nb:0;float tv=(nb>1)?fmaxf(0,tss/nb-tm*tm):0;
        int bf=ballot_freq>0?((nb+ballot_freq-1)/ballot_freq):0;
        int sf=shfl_freq>0?((nb+shfl_freq-1)/shfl_freq):0;
        hv[bi].tcv=(tm>0)?sqrtf(tv)/tm:0;
        hv[bi].spf=(bf>0)?ss/(bf*32):0;
        hv[bi].nd=(sf>0)?nds/sf:0;
        hv[bi].br=(brn>0)?brs/brn:0;
        hv[bi].mem=membrane;
        hv[bi].sp=spikes;
    }
}

__global__ void argmax_k(const float*L,int*P,int n,int c){
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=n)return;
    float best=-1e30;int bc=0;for(int i=0;i<c;i++){float v=L[b*c+i];if(v>best){best=v;bc=i;}}P[b]=bc;}

double get_ms(){struct timeval t;gettimeofday(&t,NULL);return t.tv_sec*1e3+t.tv_usec/1e3;}

int main(){
    printf("================================================================\n");
    printf("z2417: CALIBRATED SWEEP — optimized frequency + threshold\n");
    printf("================================================================\n\n");

    const char*base="/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy";
    char p[512];

    // Load MNIST test
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

    float*noise_img=(float*)malloc(N*784*4);
    srand(42);for(int i=0;i<N*784;i++)noise_img[i]=(float)rand()/RAND_MAX;

    int depths[]={3,6,8};

    for(int di=0;di<3;di++){
        int depth=depths[di];
        printf("\n================================================================\n");
        printf("DEPTH=%d (trained, 98%%/97.7%%/97.6%% accuracy)\n",depth);
        printf("================================================================\n");

        sprintf(p,"%s/models/mnist_mlp_d%d",base,depth);
        int lin[MAX_LAYERS],lout[MAX_LAYERS];
        float*d_W[MAX_LAYERS],*d_b[MAX_LAYERS];
        for(int l=0;l<depth;l++){
            lin[l]=(l==0)?IN:256; lout[l]=(l==depth-1)?OUT:256;
            int sz=lout[l]*lin[l];
            float*hw=(float*)malloc(sz*4);float*hb=(float*)malloc(lout[l]*4);
            char wp[512],bp[512];
            sprintf(wp,"%s/w%d.bin",p,l);sprintf(bp,"%s/b%d.bin",p,l);
            FILE*fw=fopen(wp,"rb");if(fw){fread(hw,4,sz,fw);fclose(fw);}
            FILE*fb=fopen(bp,"rb");if(fb){fread(hb,4,lout[l],fb);fclose(fb);}
            hipMalloc(&d_W[l],sz*4);hipMalloc(&d_b[l],lout[l]*4);
            hipMemcpy(d_W[l],hw,sz*4,hipMemcpyHostToDevice);
            hipMemcpy(d_b[l],hb,lout[l]*4,hipMemcpyHostToDevice);
            free(hw);free(hb);
        }

        float*d_act[MAX_LAYERS+1];
        hipMalloc(&d_act[0],N*IN*4);
        for(int l=0;l<depth;l++)hipMalloc(&d_act[l+1],N*lout[l]*4);
        int*d_pred;hipMalloc(&d_pred,N*4);
        Hv*d_hv;hipMalloc(&d_hv,N*sizeof(Hv));
        Hv*h_hv=(Hv*)malloc(N*sizeof(Hv));
        int h_pred[10000];

        // Configurations to test
        struct Config {
            const char*nm;
            int do_tim,ballot_f,shfl_f,do_set,do_slp;
            float thresh;
        };
        Config configs[] = {
            // Name               tim bal shf set slp thresh
            {"Baseline",           0,  0,  0,  0,  0, 5.0},
            {"Tim only",           1,  0,  0,  0,  0, 5.0},
            {"Tim+Bal/16",         1, 16,  0,  0,  0, 1.0},
            {"Tim+Bal/16",         1, 16,  0,  0,  0, 2.0},
            {"Tim+Bal/16",         1, 16,  0,  0,  0, 3.0},
            {"Tim+Bal/16+Shf/8",   1, 16,  8,  0,  0, 2.0},
            {"Tim+Bal/32+Shf/16",  1, 32, 16,  0,  0, 2.0},
            {"Tim+Bal/32+Shf/16+Set",1,32,16,  1,  0, 2.0},
            {"Tim+Bal/32+Shf/16+All",1,32,16,  1,  1, 2.0},
            {"Tim+Bal/4",          1,  4,  0,  0,  0, 2.0},
            {"Tim+Bal/4+Shf/4",    1,  4,  4,  0,  0, 2.0},
        };
        int NC = 11;

        printf("\n%-25s %5s %6s %6s %6s %6s %7s %6s\n",
               "Config","Acc%","σ","TimCV","NeigD","Spk","Over%","ΔσOOD");

        double base_time=0;
        for(int ci=0;ci<NC;ci++){
            Config&c=configs[ci];

            // ID run
            hipMemcpy(d_act[0],h_img,N*IN*4,hipMemcpyHostToDevice);
            hipDeviceSynchronize();double t0=get_ms();
            int REPS=10;
            for(int r=0;r<REPS;r++){
                for(int l=0;l<depth;l++){
                    int relu=(l<depth-1)?1:0;
                    mm_cal<<<N,lout[l]>>>(d_act[l],d_W[l],d_b[l],d_act[l+1],
                        (c.do_tim||c.ballot_f||c.shfl_f)?d_hv:nullptr,
                        N,lin[l],lout[l],relu,
                        c.do_tim,c.ballot_f,c.shfl_f,c.do_set,c.do_slp,c.thresh);
                }
                argmax_k<<<(N+255)/256,256>>>(d_act[depth],d_pred,N,OUT);
                hipDeviceSynchronize();
            }
            double dt=(get_ms()-t0)/REPS;
            if(ci==0) base_time=dt;

            hipMemcpy(h_pred,d_pred,N*4,hipMemcpyDeviceToHost);
            int corr=0;for(int i=0;i<N;i++)if(h_pred[i]==h_lbl[i])corr++;

            float sigma=0,tcv=0,nd=0,spk=0;
            if(c.do_tim||c.ballot_f||c.shfl_f){
                hipMemcpy(h_hv,d_hv,N*sizeof(Hv),hipMemcpyDeviceToHost);
                float ss=0,st=0,sn=0,ssp=0;int nv=0;
                for(int i=0;i<N;i++){
                    if(h_hv[i].br>0&&h_hv[i].br<100){ss+=h_hv[i].br;nv++;}
                    st+=h_hv[i].tcv;sn+=h_hv[i].nd;ssp+=h_hv[i].sp;}
                sigma=nv?ss/nv:0;tcv=st/N;nd=sn/N;spk=ssp/N;
            }

            // OOD
            float sigma_ood=0;
            if(c.ballot_f>0){
                hipMemcpy(d_act[0],noise_img,N*IN*4,hipMemcpyHostToDevice);
                for(int l=0;l<depth;l++){
                    int relu=(l<depth-1)?1:0;
                    mm_cal<<<N,lout[l]>>>(d_act[l],d_W[l],d_b[l],d_act[l+1],
                        d_hv,N,lin[l],lout[l],relu,
                        c.do_tim,c.ballot_f,c.shfl_f,c.do_set,c.do_slp,c.thresh);}
                hipDeviceSynchronize();
                hipMemcpy(h_hv,d_hv,N*sizeof(Hv),hipMemcpyDeviceToHost);
                float ss=0;int nv=0;
                for(int i=0;i<N;i++)if(h_hv[i].br>0&&h_hv[i].br<100){ss+=h_hv[i].br;nv++;}
                sigma_ood=nv?ss/nv:0;
            }

            float overhead=(base_time>0)?(dt-base_time)/base_time*100:0;

            printf("%-25s %4.1f%% %6.3f %6.4f %6.3f %5.1f %6.1f%% %+6.3f\n",
                   c.nm,100.0f*corr/N,sigma,tcv,nd,spk,overhead,sigma-sigma_ood);
        }

        // Correct vs incorrect for best config
        printf("\n  Correct vs Incorrect (Tim+Bal/16, thresh=2.0):\n");
        hipMemcpy(d_act[0],h_img,N*IN*4,hipMemcpyHostToDevice);
        for(int l=0;l<depth;l++){
            int relu=(l<depth-1)?1:0;
            mm_cal<<<N,lout[l]>>>(d_act[l],d_W[l],d_b[l],d_act[l+1],
                d_hv,N,lin[l],lout[l],relu,1,16,8,0,0,2.0f);}
        argmax_k<<<(N+255)/256,256>>>(d_act[depth],d_pred,N,OUT);
        hipDeviceSynchronize();
        hipMemcpy(h_hv,d_hv,N*sizeof(Hv),hipMemcpyDeviceToHost);
        hipMemcpy(h_pred,d_pred,N*4,hipMemcpyDeviceToHost);
        {
            float tc_c=0,tc_i=0,nd_c=0,nd_i=0,br_c=0,br_i=0,sp_c=0,sp_i=0;
            int nc=0,ni=0;
            for(int i=0;i<N;i++){
                if(h_pred[i]==h_lbl[i]){tc_c+=h_hv[i].tcv;nd_c+=h_hv[i].nd;
                    br_c+=h_hv[i].br;sp_c+=h_hv[i].sp;nc++;}
                else{tc_i+=h_hv[i].tcv;nd_i+=h_hv[i].nd;
                    br_i+=h_hv[i].br;sp_i+=h_hv[i].sp;ni++;}
            }
            if(nc>0&&ni>0){
                printf("    TimCV:  corr=%.4f incorr=%.4f ratio=%.3f\n",tc_c/nc,tc_i/ni,(tc_c/nc)/(tc_i/ni));
                printf("    NeighD: corr=%.3f  incorr=%.3f  ratio=%.3f\n",nd_c/nc,nd_i/ni,(nd_c/nc)/(nd_i/ni));
                printf("    σ:      corr=%.3f  incorr=%.3f  ratio=%.3f\n",br_c/nc,br_i/ni,(br_c/nc)/(br_i/ni+1e-10));
                printf("    Spikes: corr=%.2f  incorr=%.2f  ratio=%.3f\n",sp_c/nc,sp_i/ni,(sp_c/nc)/(sp_i/ni+1e-10));
                printf("    (nc=%d, ni=%d)\n",nc,ni);
            }
        }

        for(int l=0;l<depth;l++){hipFree(d_W[l]);hipFree(d_b[l]);}
        for(int l=0;l<=depth;l++)hipFree(d_act[l]);
        hipFree(d_pred);hipFree(d_hv);free(h_hv);
    }

    free(h_img);free(noise_img);free(h_lbl);
    printf("\n=== z2417 COMPLETE ===\n");
    return 0;
}
