/*
 * z2416_trained_sweep.hip — Mechanism sweep on TRAINED deep networks
 *
 * Uses pre-trained weights from z2415_train_deep.py
 * Tests all mechanism groups × 3 depths with proper accuracy
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define MAX_LAYERS 8
#define W 256
#define IN 784
#define OUT 10

struct Hv { float tcv; float spf; float nd; float br; float mem; uint32_t sp; };

__global__ void mm_sw(const float*A, const float*Wm, const float*b,
    float*C, Hv*hv, int batch, int ind, int outd, int relu, int grp) {
    int bi=blockIdx.x, o=threadIdx.x, lane=o%32;
    if(bi>=batch||o>=outd) return;
    float membrane=0; uint32_t spikes=0;
    float ts=0,tss=0,ss=0,nds=0,brs=0; uint32_t brn=0,psp=0,nb=0;
    uint32_t t0=0;
    if(grp&1) asm volatile("s_getreg_b32 %0, hwreg(29)":"=s"(t0));
    uint32_t om=0;
    if(grp&8) asm volatile("s_getreg_b32 %0, hwreg(1)":"=s"(om));

    float acc=b[o];
    for(int k=0;k<ind;k+=4){
        float p0=Wm[o*ind+k]*A[bi*ind+k];
        float p1=(k+1<ind)?Wm[o*ind+k+1]*A[bi*ind+k+1]:0;
        float p2=(k+2<ind)?Wm[o*ind+k+2]*A[bi*ind+k+2]:0;
        float p3=(k+3<ind)?Wm[o*ind+k+3]*A[bi*ind+k+3]:0;
        acc+=p0+p1+p2+p3;
        float sig=0;
        if(grp&1){uint32_t t1;asm volatile("s_getreg_b32 %0, hwreg(29)":"=s"(t1));
            float dt=(float)(t1-t0);t0=t1;ts+=dt;tss+=dt*dt;sig+=(dt>20?0.3f:0);}
        if(grp&2){float bs=p0+p1+p2+p3;uint64_t pm=__ballot(bs>0);uint32_t np=__popcll(pm);
            ss+=(float)np;uint32_t mn=(np<16)?np:(32-np);sig+=(float)mn*0.04f;
            int npos=(p0>0)+(p1>0)+(p2>0)+(p3>0);sig+=(float)(4-npos)*0.15f;
            uint64_t sm=__ballot(membrane>5.0f);uint32_t cs=__popcll(sm);
            if(psp>0){brs+=(float)cs/(float)psp;brn++;}psp=cs;}
        if(grp&4){float l=__shfl(acc,(lane+31)%32);nds+=fabsf(acc-l);sig+=fabsf(acc-l)*0.005f;}
        if((grp&8)&&(nb&7)==0){
            if(membrane>3)asm volatile("s_setreg_imm32_b32 hwreg(1,0,2),0x3");
            else if(membrane>1.5f)asm volatile("s_setreg_imm32_b32 hwreg(1,0,2),0x1");
            else asm volatile("s_setreg_imm32_b32 hwreg(1,0,2),0x0");}
        nb++;membrane+=sig;
        if(membrane>5){spikes++;membrane=0;if(grp&0x10)asm volatile("s_sleep 1");}
    }
    if(grp&8)asm volatile("s_setreg_b32 hwreg(1),%0"::"s"(om));
    if(relu)acc=fmaxf(0,acc);
    C[bi*outd+o]=acc;
    if(lane==0&&hv){float tm=(nb>0)?ts/nb:0;float tv=(nb>1)?fmaxf(0,tss/nb-tm*tm):0;
        hv[bi].tcv=(tm>0)?sqrtf(tv)/tm:0;hv[bi].spf=(nb>0)?ss/(nb*32):0;
        hv[bi].nd=(nb>0)?nds/nb:0;hv[bi].br=(brn>0)?brs/brn:0;
        hv[bi].mem=membrane;hv[bi].sp=spikes;}
}

__global__ void argmax_k(const float*L,int*P,int n,int c){
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=n)return;
    float best=-1e30;int bc=0;for(int i=0;i<c;i++){float v=L[b*c+i];if(v>best){best=v;bc=i;}}P[b]=bc;}

double get_ms(){struct timeval t;gettimeofday(&t,NULL);return t.tv_sec*1e3+t.tv_usec/1e3;}
float read_power(){for(int h=7;h<=10;h++){char p[64];sprintf(p,"/sys/class/hwmon/hwmon%d/power1_average",h);
    FILE*f=fopen(p,"r");if(f){long v;fscanf(f,"%ld",&v);fclose(f);return v/1e6;}}return -1;}

int main(){
    printf("================================================================\n");
    printf("z2416: TRAINED DEEP MECHANISM SWEEP\n");
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
    struct{const char*nm;int mask;}groups[]={
        {"G0:Base",0},{"G1:Tim",1},{"G2:+Bal",3},{"G3:+Shf",7},
        {"G4:+Set",0xF},{"G5:+Slp",0x1F},{"G6:Full",0x3F}};
    int NG=7;

    for(int di=0;di<3;di++){
        int depth=depths[di];
        printf("\n================================================================\n");
        printf("DEPTH=%d (trained)\n",depth);
        printf("================================================================\n");

        // Load weights
        sprintf(p,"%s/models/mnist_mlp_d%d",base,depth);
        int lin[MAX_LAYERS],lout[MAX_LAYERS];
        float*d_W[MAX_LAYERS],*d_b[MAX_LAYERS];
        for(int l=0;l<depth;l++){
            lin[l]=(l==0)?IN:W; lout[l]=(l==depth-1)?OUT:W;
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

        printf("%-12s %6s %6s %6s %6s %7s %6s\n",
               "Group","Acc%","σ","TimCV","NeigD","Over%","ΔσOOD");

        double base_time=0;
        for(int gi=0;gi<NG;gi++){
            int mask=groups[gi].mask;

            // ID run
            hipMemcpy(d_act[0],h_img,N*IN*4,hipMemcpyHostToDevice);
            hipDeviceSynchronize();double t0=get_ms();
            int REPS=10;
            for(int r=0;r<REPS;r++){
                for(int l=0;l<depth;l++){
                    int relu=(l<depth-1)?1:0;
                    mm_sw<<<N,lout[l]>>>(d_act[l],d_W[l],d_b[l],d_act[l+1],
                        (mask?d_hv:nullptr),N,lin[l],lout[l],relu,mask);
                }
                argmax_k<<<(N+255)/256,256>>>(d_act[depth],d_pred,N,OUT);
                hipDeviceSynchronize();
            }
            double dt=(get_ms()-t0)/REPS;
            if(gi==0)base_time=dt;

            hipMemcpy(h_pred,d_pred,N*4,hipMemcpyDeviceToHost);
            int corr=0;for(int i=0;i<N;i++)if(h_pred[i]==h_lbl[i])corr++;
            float acc=100.0f*corr/N;

            float sigma_id=0,tcv_id=0,nd_id=0;
            if(mask){
                hipMemcpy(h_hv,d_hv,N*sizeof(Hv),hipMemcpyDeviceToHost);
                float ss=0,st=0,sn=0;int nv=0;
                for(int i=0;i<N;i++){
                    if(h_hv[i].br>0&&h_hv[i].br<100){ss+=h_hv[i].br;nv++;}
                    st+=h_hv[i].tcv;sn+=h_hv[i].nd;}
                sigma_id=nv?ss/nv:0;tcv_id=st/N;nd_id=sn/N;
            }

            // OOD run
            float sigma_ood=0;
            if(mask){
                hipMemcpy(d_act[0],noise_img,N*IN*4,hipMemcpyHostToDevice);
                for(int l=0;l<depth;l++){
                    int relu=(l<depth-1)?1:0;
                    mm_sw<<<N,lout[l]>>>(d_act[l],d_W[l],d_b[l],d_act[l+1],
                        d_hv,N,lin[l],lout[l],relu,mask);}
                hipDeviceSynchronize();
                hipMemcpy(h_hv,d_hv,N*sizeof(Hv),hipMemcpyDeviceToHost);
                float ss=0;int nv=0;
                for(int i=0;i<N;i++)
                    if(h_hv[i].br>0&&h_hv[i].br<100){ss+=h_hv[i].br;nv++;}
                sigma_ood=nv?ss/nv:0;
            }

            float overhead=(base_time>0)?(dt-base_time)/base_time*100:0;
            printf("%-12s %5.1f%% %6.3f %6.4f %6.3f %6.1f%% %+6.3f\n",
                   groups[gi].nm,acc,sigma_id,tcv_id,nd_id,overhead,sigma_id-sigma_ood);
        }

        // Per-layer σ analysis (G2 mask, per layer)
        printf("\n  Per-layer σ (G2 mask):\n  ");
        hipMemcpy(d_act[0],h_img,N*IN*4,hipMemcpyHostToDevice);
        for(int l=0;l<depth;l++){
            int relu=(l<depth-1)?1:0;
            mm_sw<<<N,lout[l]>>>(d_act[l],d_W[l],d_b[l],d_act[l+1],
                d_hv,N,lin[l],lout[l],relu,3);
            hipDeviceSynchronize();
            hipMemcpy(h_hv,d_hv,N*sizeof(Hv),hipMemcpyDeviceToHost);
            float ss=0;int nv=0;
            for(int i=0;i<N;i++)if(h_hv[i].br>0&&h_hv[i].br<100){ss+=h_hv[i].br;nv++;}
            printf("L%d:σ=%.3f ",l,nv?ss/nv:0);
        }
        printf("\n");

        // Confidence: correct vs incorrect (G3)
        printf("\n  Correct vs Incorrect (G3):\n");
        hipMemcpy(d_act[0],h_img,N*IN*4,hipMemcpyHostToDevice);
        for(int l=0;l<depth;l++){
            int relu=(l<depth-1)?1:0;
            mm_sw<<<N,lout[l]>>>(d_act[l],d_W[l],d_b[l],d_act[l+1],
                d_hv,N,lin[l],lout[l],relu,7);}
        argmax_k<<<(N+255)/256,256>>>(d_act[depth],d_pred,N,OUT);
        hipDeviceSynchronize();
        hipMemcpy(h_hv,d_hv,N*sizeof(Hv),hipMemcpyDeviceToHost);
        hipMemcpy(h_pred,d_pred,N*4,hipMemcpyDeviceToHost);
        {
            float tc_c=0,tc_i=0,nd_c=0,nd_i=0,br_c=0,br_i=0;
            int nc=0,ni=0;
            for(int i=0;i<N;i++){
                if(h_pred[i]==h_lbl[i]){tc_c+=h_hv[i].tcv;nd_c+=h_hv[i].nd;br_c+=h_hv[i].br;nc++;}
                else{tc_i+=h_hv[i].tcv;nd_i+=h_hv[i].nd;br_i+=h_hv[i].br;ni++;}
            }
            if(nc>0&&ni>0){
                printf("    TimCV:  corr=%.4f incorr=%.4f ratio=%.3f\n",tc_c/nc,tc_i/ni,(tc_c/nc)/(tc_i/ni));
                printf("    NeighD: corr=%.3f  incorr=%.3f  ratio=%.3f\n",nd_c/nc,nd_i/ni,(nd_c/nc)/(nd_i/ni));
                printf("    σ:      corr=%.3f  incorr=%.3f  ratio=%.3f\n",br_c/nc,br_i/ni,(br_c/nc)/(br_i/ni));
            }
        }

        for(int l=0;l<depth;l++){hipFree(d_W[l]);hipFree(d_b[l]);}
        for(int l=0;l<=depth;l++)hipFree(d_act[l]);
        hipFree(d_pred);hipFree(d_hv);free(h_hv);
    }

    free(h_img);free(noise_img);free(h_lbl);
    printf("\n=== z2416 COMPLETE ===\n");
    return 0;
}
