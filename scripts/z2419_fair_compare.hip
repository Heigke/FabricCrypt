/*
 * z2419 — Fair comparison: both kernels 4x unrolled + per-OOD-type AUC
 *
 * z2418 had unfair baseline (1x loop vs 4x harvest). Fix: both 4x.
 * Also: compute AUC per OOD type — NeighD may beat softmax on specific types
 * like "all ones" where softmax gives 99.99% confidence on garbage.
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define MAX_L 8
#define IN 784
#define OUT 10

// FAIR baseline: same 4x unrolling as harvest
__global__ void mm_fair(const float*A, const float*W, const float*b,
    float*C, int batch, int ind, int outd, int relu){
    int bi=blockIdx.x,o=threadIdx.x;
    if(bi>=batch||o>=outd)return;
    float acc=b[o];
    for(int k=0;k<ind;k+=4){
        acc+=W[o*ind+k]*A[bi*ind+k];
        if(k+1<ind)acc+=W[o*ind+k+1]*A[bi*ind+k+1];
        if(k+2<ind)acc+=W[o*ind+k+2]*A[bi*ind+k+2];
        if(k+3<ind)acc+=W[o*ind+k+3]*A[bi*ind+k+3];
    }
    if(relu)acc=fmaxf(0,acc);
    C[bi*outd+o]=acc;
}

// Harvest: 4x unrolled + timing + shfl
__global__ void mm_harv(const float*A, const float*W, const float*b,
    float*C, float*neighd, float*timcv,
    int batch, int ind, int outd, int relu){
    int bi=blockIdx.x,o=threadIdx.x,lane=o%32;
    if(bi>=batch||o>=outd)return;
    float nd_sum=0,ts=0,tss=0;uint32_t nb=0;
    uint32_t t0;asm volatile("s_getreg_b32 %0, hwreg(29)":"=s"(t0));
    float acc=b[o];
    for(int k=0;k<ind;k+=4){
        acc+=W[o*ind+k]*A[bi*ind+k];
        if(k+1<ind)acc+=W[o*ind+k+1]*A[bi*ind+k+1];
        if(k+2<ind)acc+=W[o*ind+k+2]*A[bi*ind+k+2];
        if(k+3<ind)acc+=W[o*ind+k+3]*A[bi*ind+k+3];
        uint32_t t1;asm volatile("s_getreg_b32 %0, hwreg(29)":"=s"(t1));
        float dt=(float)(t1-t0);t0=t1;ts+=dt;tss+=dt*dt;
        float left=__shfl(acc,(lane+31)%32);
        nd_sum+=fabsf(acc-left);
        nb++;
    }
    if(relu)acc=fmaxf(0,acc);
    C[bi*outd+o]=acc;
    float my_nd=(nb>0)?nd_sum/nb:0;
    for(int off=16;off>0;off>>=1)my_nd+=__shfl_xor(my_nd,off);
    my_nd/=32.0f;
    if(lane==0){
        neighd[bi]=my_nd;
        float tm=(nb>0)?ts/nb:0;float tv=(nb>1)?fmaxf(0,tss/nb-tm*tm):0;
        timcv[bi]=(tm>0)?sqrtf(tv)/tm:0;
    }
}

__global__ void argmax_k(const float*L,int*P,int n,int c){
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=n)return;
    float best=-1e30;int bc=0;for(int i=0;i<c;i++){float v=L[b*c+i];if(v>best){best=v;bc=i;}}P[b]=bc;}

double get_ms(){struct timeval t;gettimeofday(&t,NULL);return t.tv_sec*1e3+t.tv_usec/1e3;}
float read_power(){for(int h=7;h<=10;h++){char p[64];sprintf(p,"/sys/class/hwmon/hwmon%d/power1_average",h);
    FILE*f=fopen(p,"r");if(f){long v;fscanf(f,"%ld",&v);fclose(f);return v/1e6;}}return -1;}

float compute_auc(float*s,int*l,int n){
    long long c=0,d=0;srand(42);
    for(int t=0;t<2000000;t++){int i=rand()%n,j=rand()%n;
        if(l[i]==l[j])continue;
        if(l[i]==1&&s[i]>s[j])c++;else if(l[j]==1&&s[j]>s[i])c++;else d++;}
    return(float)c/(c+d+1);}

int main(){
    printf("================================================================\n");
    printf("z2419: FAIR comparison + per-OOD-type AUC\n");
    printf("================================================================\n\n");

    const char*base="/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy";
    char p[512];
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

    // 5 OOD types
    float*ood_data[4];
    const char*ood_names[]={"Noise","AllOnes","Shuffled","Inverted"};
    ood_data[0]=(float*)malloc(N*784*4); // noise
    ood_data[1]=(float*)malloc(N*784*4); // ones
    ood_data[2]=(float*)malloc(N*784*4); // shuffled
    ood_data[3]=(float*)malloc(N*784*4); // inverted
    srand(42);
    for(int i=0;i<N*784;i++){
        ood_data[0][i]=(float)rand()/RAND_MAX;
        ood_data[1][i]=1.0f;
        ood_data[3][i]=1.0f-h_img[i];
    }
    memcpy(ood_data[2],h_img,N*784*4);
    for(int i=0;i<N;i++)for(int j=783;j>0;j--){
        int k=rand()%(j+1);float t=ood_data[2][i*784+j];
        ood_data[2][i*784+j]=ood_data[2][i*784+k];ood_data[2][i*784+k]=t;}

    int depths[]={3,6,8};
    for(int di=0;di<3;di++){
        int depth=depths[di];
        printf("\n================================================================\n");
        printf("DEPTH=%d\n",depth);
        printf("================================================================\n");

        sprintf(p,"%s/models/mnist_mlp_d%d",base,depth);
        int lin[MAX_L],lout[MAX_L];
        float*d_W[MAX_L],*d_b[MAX_L];
        for(int l=0;l<depth;l++){
            lin[l]=(l==0)?IN:256;lout[l]=(l==depth-1)?OUT:256;
            int sz=lout[l]*lin[l];
            float*hw=(float*)malloc(sz*4);float*hb=(float*)malloc(lout[l]*4);
            sprintf(p,"%s/models/mnist_mlp_d%d/w%d.bin",base,depth,l);
            FILE*fw=fopen(p,"rb");if(fw){fread(hw,4,sz,fw);fclose(fw);}
            sprintf(p,"%s/models/mnist_mlp_d%d/b%d.bin",base,depth,l);
            FILE*fb=fopen(p,"rb");if(fb){fread(hb,4,lout[l],fb);fclose(fb);}
            hipMalloc(&d_W[l],sz*4);hipMalloc(&d_b[l],lout[l]*4);
            hipMemcpy(d_W[l],hw,sz*4,hipMemcpyHostToDevice);
            hipMemcpy(d_b[l],hb,lout[l]*4,hipMemcpyHostToDevice);
            free(hw);free(hb);
        }
        float*d_act[MAX_L+1];
        hipMalloc(&d_act[0],N*IN*4);
        for(int l=0;l<depth;l++)hipMalloc(&d_act[l+1],N*lout[l]*4);
        int*d_pred;hipMalloc(&d_pred,N*4);
        float*d_nd;hipMalloc(&d_nd,N*4);
        float*d_tc;hipMalloc(&d_tc,N*4);

        // FAIR overhead
        printf("\n--- FAIR Overhead (both 4x unrolled) ---\n");
        hipMemcpy(d_act[0],h_img,N*IN*4,hipMemcpyHostToDevice);
        hipDeviceSynchronize();double t0=get_ms();
        for(int r=0;r<30;r++){
            for(int l=0;l<depth;l++){int relu=(l<depth-1)?1:0;
                mm_fair<<<N,lout[l]>>>(d_act[l],d_W[l],d_b[l],d_act[l+1],N,lin[l],lout[l],relu);}
            hipDeviceSynchronize();}
        double tb=(get_ms()-t0)/30;
        hipDeviceSynchronize();t0=get_ms();
        for(int r=0;r<30;r++){
            for(int l=0;l<depth;l++){int relu=(l<depth-1)?1:0;
                mm_harv<<<N,lout[l]>>>(d_act[l],d_W[l],d_b[l],d_act[l+1],d_nd,d_tc,N,lin[l],lout[l],relu);}
            hipDeviceSynchronize();}
        double th=(get_ms()-t0)/30;
        printf("  Fair base: %.3f ms  Harvest: %.3f ms  Overhead: %.1f%%\n",tb,th,(th-tb)/tb*100);

        // Energy
        float pw0=read_power();hipDeviceSynchronize();t0=get_ms();
        for(int r=0;r<50;r++){
            for(int l=0;l<depth;l++){int relu=(l<depth-1)?1:0;
                mm_fair<<<N,lout[l]>>>(d_act[l],d_W[l],d_b[l],d_act[l+1],N,lin[l],lout[l],relu);}
            hipDeviceSynchronize();}
        double tbe=(get_ms()-t0)/50;float pw1=read_power();
        float bpw=(pw0+pw1)/2,bej=bpw*tbe/1000;

        pw0=read_power();hipDeviceSynchronize();t0=get_ms();
        for(int r=0;r<50;r++){
            for(int l=0;l<depth;l++){int relu=(l<depth-1)?1:0;
                mm_harv<<<N,lout[l]>>>(d_act[l],d_W[l],d_b[l],d_act[l+1],d_nd,d_tc,N,lin[l],lout[l],relu);}
            hipDeviceSynchronize();}
        double the=(get_ms()-t0)/50;pw1=read_power();
        float hpw=(pw0+pw1)/2,hej=hpw*the/1000;
        printf("  Base energy:  %.4fJ (%.1f μJ/samp)\n",bej,bej/N*1e6);
        printf("  Harv energy:  %.4fJ (%.1f μJ/samp)  overhead: %.1f%%\n",hej,hej/N*1e6,(hej-bej)/bej*100);

        // Accuracy
        hipMemcpy(d_act[0],h_img,N*IN*4,hipMemcpyHostToDevice);
        for(int l=0;l<depth;l++){int relu=(l<depth-1)?1:0;
            mm_harv<<<N,lout[l]>>>(d_act[l],d_W[l],d_b[l],d_act[l+1],d_nd,d_tc,N,lin[l],lout[l],relu);}
        argmax_k<<<(N+255)/256,256>>>(d_act[depth],d_pred,N,OUT);
        hipDeviceSynchronize();

        float*h_nd=(float*)malloc(N*4);
        float*h_tc=(float*)malloc(N*4);
        float*h_logits=(float*)malloc(N*OUT*4);
        int*h_pred=(int*)malloc(N*4);
        hipMemcpy(h_nd,d_nd,N*4,hipMemcpyDeviceToHost);
        hipMemcpy(h_tc,d_tc,N*4,hipMemcpyDeviceToHost);
        hipMemcpy(h_logits,d_act[depth],N*OUT*4,hipMemcpyDeviceToHost);
        hipMemcpy(h_pred,d_pred,N*4,hipMemcpyDeviceToHost);

        int corr=0;for(int i=0;i<N;i++)if(h_pred[i]==h_lbl[i])corr++;
        printf("  Accuracy: %.2f%%\n",100.0f*corr/N);

        // Softmax confidence
        float*h_sm=(float*)malloc(N*4);
        for(int i=0;i<N;i++){
            float mx=-1e30;for(int c=0;c<OUT;c++)if(h_logits[i*OUT+c]>mx)mx=h_logits[i*OUT+c];
            float sum=0;for(int c=0;c<OUT;c++)sum+=expf(h_logits[i*OUT+c]-mx);
            float best=0;for(int c=0;c<OUT;c++){float pp=expf(h_logits[i*OUT+c]-mx)/sum;if(pp>best)best=pp;}
            h_sm[i]=best;
        }

        // Per-OOD-type AUC
        printf("\n--- Per-OOD-type AUC (ID=MNIST vs each OOD separately) ---\n");
        printf("  %-12s %8s %8s %8s %8s\n","OOD type","AUC_ND","AUC_SM","AUC_TC","AUC_combo");

        for(int oi=0;oi<4;oi++){
            // Run OOD
            hipMemcpy(d_act[0],ood_data[oi],N*IN*4,hipMemcpyHostToDevice);
            for(int l=0;l<depth;l++){int relu=(l<depth-1)?1:0;
                mm_harv<<<N,lout[l]>>>(d_act[l],d_W[l],d_b[l],d_act[l+1],d_nd,d_tc,N,lin[l],lout[l],relu);}
            argmax_k<<<(N+255)/256,256>>>(d_act[depth],d_pred,N,OUT);
            hipDeviceSynchronize();

            float*ood_nd=(float*)malloc(N*4);
            float*ood_tc=(float*)malloc(N*4);
            float*ood_logits=(float*)malloc(N*OUT*4);
            hipMemcpy(ood_nd,d_nd,N*4,hipMemcpyDeviceToHost);
            hipMemcpy(ood_tc,d_tc,N*4,hipMemcpyDeviceToHost);
            hipMemcpy(ood_logits,d_act[depth],N*OUT*4,hipMemcpyDeviceToHost);

            float*ood_sm=(float*)malloc(N*4);
            for(int i=0;i<N;i++){
                float mx=-1e30;for(int c=0;c<OUT;c++)if(ood_logits[i*OUT+c]>mx)mx=ood_logits[i*OUT+c];
                float sum=0;for(int c=0;c<OUT;c++)sum+=expf(ood_logits[i*OUT+c]-mx);
                float best=0;for(int c=0;c<OUT;c++){float pp=expf(ood_logits[i*OUT+c]-mx)/sum;if(pp>best)best=pp;}
                ood_sm[i]=best;
            }

            // Combine: ID (label=1) + OOD (label=0)
            int total=N+N;
            float*all_nd=(float*)malloc(total*4);
            float*all_sm2=(float*)malloc(total*4);
            float*all_tc2=(float*)malloc(total*4);
            int*all_lbl=(int*)malloc(total*4);
            memcpy(all_nd,h_nd,N*4);memcpy(all_nd+N,ood_nd,N*4);
            memcpy(all_sm2,h_sm,N*4);memcpy(all_sm2+N,ood_sm,N*4);
            memcpy(all_tc2,h_tc,N*4);memcpy(all_tc2+N,ood_tc,N*4);
            for(int i=0;i<N;i++){all_lbl[i]=1;all_lbl[N+i]=0;}

            float auc_nd=compute_auc(all_nd,all_lbl,total);
            float auc_sm=compute_auc(all_sm2,all_lbl,total);
            float auc_tc=compute_auc(all_tc2,all_lbl,total);

            // Best combo
            float nd_min=1e30,nd_max=-1e30,sm_min=1e30,sm_max=-1e30;
            for(int i=0;i<total;i++){
                if(all_nd[i]<nd_min)nd_min=all_nd[i];if(all_nd[i]>nd_max)nd_max=all_nd[i];
                if(all_sm2[i]<sm_min)sm_min=all_sm2[i];if(all_sm2[i]>sm_max)sm_max=all_sm2[i];}
            float best_auc=0;
            float*combo=(float*)malloc(total*4);
            for(float a=0;a<=1.01;a+=0.1){
                for(int i=0;i<total;i++){
                    float n=(all_nd[i]-nd_min)/(nd_max-nd_min+1e-10);
                    float s=(all_sm2[i]-sm_min)/(sm_max-sm_min+1e-10);
                    combo[i]=a*s+(1-a)*n;}
                float auc=compute_auc(combo,all_lbl,total);
                if(auc>best_auc)best_auc=auc;
            }

            const char*winner=(auc_nd>auc_sm)?"ND★":(best_auc>auc_sm+0.01)?"COMBO★":"SM";
            printf("  %-12s %8.4f %8.4f %8.4f %8.4f  %s\n",
                   ood_names[oi],auc_nd,auc_sm,auc_tc,best_auc,winner);

            free(ood_nd);free(ood_tc);free(ood_logits);free(ood_sm);
            free(all_nd);free(all_sm2);free(all_tc2);free(all_lbl);free(combo);
        }

        // Correct vs incorrect with FAIR comparison
        printf("\n--- Correct vs Incorrect ---\n");
        {
            float nd_c=0,nd_i=0,tc_c=0,tc_i=0;int nc=0,ni=0;
            for(int i=0;i<N;i++){
                if(h_pred[i]==h_lbl[i]){nd_c+=h_nd[i];tc_c+=h_tc[i];nc++;}
                else{nd_i+=h_nd[i];tc_i+=h_tc[i];ni++;}
            }
            printf("  NeighD: corr=%.3f incorr=%.3f ratio=%.3f\n",nd_c/nc,nd_i/ni,(nd_c/nc)/(nd_i/ni));
            printf("  TimCV:  corr=%.4f incorr=%.4f ratio=%.3f\n",tc_c/nc,tc_i/ni,(tc_c/nc)/(tc_i/ni));
        }

        // Rejection curve
        printf("\n--- Rejection accuracy ---\n");
        printf("  %-8s %10s %10s %10s\n","Reject%","AccNeighD","AccSoftmax","AccCombo");
        for(int rp : {0,5,10,20,30}){
            int n_keep=N-N*rp/100;
            // Sort indices by score descending, keep top n_keep
            auto rejection_acc = [&](float*scores) -> float {
                int*idx=(int*)malloc(N*4);for(int i=0;i<N;i++)idx[i]=i;
                // Partial sort: find top n_keep
                for(int i=0;i<n_keep;i++){
                    int mx=i;for(int j=i+1;j<N;j++)if(scores[idx[j]]>scores[idx[mx]])mx=j;
                    int t=idx[i];idx[i]=idx[mx];idx[mx]=t;
                }
                int c=0;for(int i=0;i<n_keep;i++)if(h_pred[idx[i]]==h_lbl[idx[i]])c++;
                free(idx);return 100.0f*c/n_keep;
            };

            // Combined score
            float nd_min=1e30,nd_max=-1e30,sm_min=1e30,sm_max=-1e30;
            for(int i=0;i<N;i++){
                if(h_nd[i]<nd_min)nd_min=h_nd[i];if(h_nd[i]>nd_max)nd_max=h_nd[i];
                if(h_sm[i]<sm_min)sm_min=h_sm[i];if(h_sm[i]>sm_max)sm_max=h_sm[i];}
            float*combo_s=(float*)malloc(N*4);
            for(int i=0;i<N;i++){
                float n=(h_nd[i]-nd_min)/(nd_max-nd_min+1e-10);
                float s=(h_sm[i]-sm_min)/(sm_max-sm_min+1e-10);
                combo_s[i]=0.7f*s+0.3f*n;}

            printf("  %-8d %9.2f%% %9.2f%% %9.2f%%\n",rp,
                   rejection_acc(h_nd),rejection_acc(h_sm),rejection_acc(combo_s));
            free(combo_s);
        }

        free(h_nd);free(h_tc);free(h_logits);free(h_pred);free(h_sm);
        for(int l=0;l<depth;l++){hipFree(d_W[l]);hipFree(d_b[l]);}
        for(int l=0;l<=depth;l++)hipFree(d_act[l]);
        hipFree(d_pred);hipFree(d_nd);hipFree(d_tc);
    }

    free(h_img);free(h_lbl);
    for(int i=0;i<4;i++)free(ood_data[i]);
    printf("\n=== z2419 COMPLETE ===\n");
    return 0;
}
