/*
 * z2418 — NeighD as OOD detector + confidence calibrator
 *
 * NeighD scales with depth (2.17x at 8L). Test if it beats softmax for OOD.
 * Also: use NeighD to REJECT uncertain samples → effective accuracy boost.
 *
 * Key change: compute NeighD with ONLY __shfl (1 instruction, no ballot).
 * This should have near-zero overhead since shfl uses cross-lane unit.
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

// Ultra-minimal harvest: ONLY shfl neighbor diff + timing
__global__ void mm_neighd(
    const float*A, const float*W, const float*b, float*C,
    float* neighd_out,  // [batch] — just one float per sample
    float* timing_out,  // [batch] — timing CV
    int batch, int ind, int outd, int relu)
{
    int bi=blockIdx.x, o=threadIdx.x, lane=o%32;
    if(bi>=batch||o>=outd) return;

    float nd_sum=0;
    float ts=0, tss=0;
    uint32_t nb=0;
    uint32_t t0;
    asm volatile("s_getreg_b32 %0, hwreg(29)":"=s"(t0));

    float acc=b[o];
    for(int k=0;k<ind;k+=4){
        float p0=W[o*ind+k]*A[bi*ind+k];
        float p1=(k+1<ind)?W[o*ind+k+1]*A[bi*ind+k+1]:0;
        float p2=(k+2<ind)?W[o*ind+k+2]*A[bi*ind+k+2]:0;
        float p3=(k+3<ind)?W[o*ind+k+3]*A[bi*ind+k+3]:0;
        acc+=p0+p1+p2+p3;

        // Timing (1 SALU instruction)
        uint32_t t1;
        asm volatile("s_getreg_b32 %0, hwreg(29)":"=s"(t1));
        float dt=(float)(t1-t0); t0=t1;
        ts+=dt; tss+=dt*dt;

        // Shfl neighbor diff (1 cross-lane instruction)
        float left=__shfl(acc,(lane+31)%32);
        nd_sum+=fabsf(acc-left);
        nb++;
    }
    if(relu) acc=fmaxf(0,acc);
    C[bi*outd+o]=acc;

    // Reduce NeighD across lanes (wave-level)
    float my_nd = (nb>0)?nd_sum/nb:0;
    // Wave reduction via shfl
    for(int offset=16;offset>0;offset>>=1)
        my_nd += __shfl_xor(my_nd, offset);
    my_nd /= 32.0f; // average across lanes

    if(lane==0){
        neighd_out[bi] = my_nd;
        float tm=(nb>0)?ts/nb:0;
        float tv=(nb>1)?fmaxf(0,tss/nb-tm*tm):0;
        timing_out[bi] = (tm>0)?sqrtf(tv)/tm:0;
    }
}

__global__ void mm_base(const float*A, const float*W, const float*b,
    float*C, int batch, int ind, int outd, int relu){
    int bi=blockIdx.x,o=threadIdx.x;
    if(bi>=batch||o>=outd)return;
    float acc=b[o];
    for(int k=0;k<ind;k++)acc+=W[o*ind+k]*A[bi*ind+k];
    if(relu)acc=fmaxf(0,acc);
    C[bi*outd+o]=acc;
}

__global__ void argmax_k(const float*L,int*P,int n,int c){
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=n)return;
    float best=-1e30;int bc=0;for(int i=0;i<c;i++){float v=L[b*c+i];if(v>best){best=v;bc=i;}}P[b]=bc;}

double get_ms(){struct timeval t;gettimeofday(&t,NULL);return t.tv_sec*1e3+t.tv_usec/1e3;}
float read_power(){for(int h=7;h<=10;h++){char p[64];sprintf(p,"/sys/class/hwmon/hwmon%d/power1_average",h);
    FILE*f=fopen(p,"r");if(f){long v;fscanf(f,"%ld",&v);fclose(f);return v/1e6;}}return -1;}

// Simple AUC via sorted pairs
float compute_auc(float*scores, int*labels, int n){
    // labels: 1=positive, 0=negative. Higher score = more likely positive.
    long long conc=0,disc=0;
    // Sample 2M pairs for speed
    for(int t=0;t<2000000;t++){
        int i=rand()%n, j=rand()%n;
        if(labels[i]==labels[j]) continue;
        if(labels[i]==1 && scores[i]>scores[j]) conc++;
        else if(labels[j]==1 && scores[j]>scores[i]) conc++;
        else disc++;
    }
    return (float)conc/(conc+disc+1);
}

int main(){
    printf("================================================================\n");
    printf("z2418: NeighD/TimCV as OOD detector + rejection classifier\n");
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

    // OOD datasets
    float*noise_img=(float*)malloc(N*784*4);
    float*ones_img=(float*)malloc(N*784*4);
    float*shuf_img=(float*)malloc(N*784*4);
    float*inv_img=(float*)malloc(N*784*4);
    srand(42);
    for(int i=0;i<N*784;i++){
        noise_img[i]=(float)rand()/RAND_MAX;
        ones_img[i]=1.0f;
        inv_img[i]=1.0f-h_img[i];
    }
    memcpy(shuf_img,h_img,N*784*4);
    for(int i=0;i<N;i++)for(int j=783;j>0;j--){
        int k=rand()%(j+1);float t=shuf_img[i*784+j];
        shuf_img[i*784+j]=shuf_img[i*784+k];shuf_img[i*784+k]=t;}

    int depths[]={3,6,8};

    for(int di=0;di<3;di++){
        int depth=depths[di];
        printf("\n================================================================\n");
        printf("DEPTH=%d\n",depth);
        printf("================================================================\n");

        sprintf(p,"%s/models/mnist_mlp_d%d",base,depth);
        int lin[MAX_LAYERS],lout[MAX_LAYERS];
        float*d_W[MAX_LAYERS],*d_b[MAX_LAYERS];
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

        float*d_act[MAX_LAYERS+1];
        hipMalloc(&d_act[0],N*IN*4);
        for(int l=0;l<depth;l++)hipMalloc(&d_act[l+1],N*lout[l]*4);
        int*d_pred;hipMalloc(&d_pred,N*4);
        float*d_neighd;hipMalloc(&d_neighd,N*4);
        float*d_timing;hipMalloc(&d_timing,N*4);

        float*h_neighd=(float*)malloc(N*4);
        float*h_timing=(float*)malloc(N*4);
        float*h_logits=(float*)malloc(N*OUT*4);
        int*h_pred=(int*)malloc(N*4);

        // ── OVERHEAD TEST ──
        printf("\n--- Overhead ---\n");
        hipMemcpy(d_act[0],h_img,N*IN*4,hipMemcpyHostToDevice);
        hipDeviceSynchronize();double t0=get_ms();
        for(int r=0;r<20;r++){
            for(int l=0;l<depth;l++){int relu=(l<depth-1)?1:0;
                mm_base<<<N,lout[l]>>>(d_act[l],d_W[l],d_b[l],d_act[l+1],N,lin[l],lout[l],relu);}
            hipDeviceSynchronize();}
        double tb=(get_ms()-t0)/20;

        hipDeviceSynchronize();t0=get_ms();
        for(int r=0;r<20;r++){
            for(int l=0;l<depth;l++){int relu=(l<depth-1)?1:0;
                mm_neighd<<<N,lout[l]>>>(d_act[l],d_W[l],d_b[l],d_act[l+1],
                    d_neighd,d_timing,N,lin[l],lout[l],relu);}
            hipDeviceSynchronize();}
        double th=(get_ms()-t0)/20;
        printf("  Base: %.3f ms  NeighD: %.3f ms  Overhead: %.1f%%\n",tb,th,(th-tb)/tb*100);

        // ── ENERGY ──
        printf("\n--- Energy ---\n");
        float pw0=read_power();
        hipDeviceSynchronize();t0=get_ms();
        for(int r=0;r<50;r++){
            for(int l=0;l<depth;l++){int relu=(l<depth-1)?1:0;
                mm_base<<<N,lout[l]>>>(d_act[l],d_W[l],d_b[l],d_act[l+1],N,lin[l],lout[l],relu);}
            hipDeviceSynchronize();}
        double tbe=(get_ms()-t0)/50;float pw1=read_power();
        float base_pw=(pw0+pw1)/2, base_ej=base_pw*tbe/1000;

        pw0=read_power();
        hipDeviceSynchronize();t0=get_ms();
        for(int r=0;r<50;r++){
            for(int l=0;l<depth;l++){int relu=(l<depth-1)?1:0;
                mm_neighd<<<N,lout[l]>>>(d_act[l],d_W[l],d_b[l],d_act[l+1],
                    d_neighd,d_timing,N,lin[l],lout[l],relu);}
            hipDeviceSynchronize();}
        double the=(get_ms()-t0)/50;pw1=read_power();
        float harv_pw=(pw0+pw1)/2, harv_ej=harv_pw*the/1000;
        printf("  Base:  %.1fW × %.2fms = %.4fJ (%.1f μJ/sample)\n",base_pw,tbe,base_ej,base_ej/N*1e6);
        printf("  Harv:  %.1fW × %.2fms = %.4fJ (%.1f μJ/sample)\n",harv_pw,the,harv_ej,harv_ej/N*1e6);
        printf("  Energy overhead: %.1f%%\n",(harv_ej-base_ej)/base_ej*100);

        // ── OOD DETECTION ──
        printf("\n--- OOD Detection ---\n");
        struct{const char*nm;float*data;int is_id;}ood[]={
            {"MNIST (ID)",h_img,1},{"Noise",noise_img,0},
            {"All ones",ones_img,0},{"Shuffled",shuf_img,0},{"Inverted",inv_img,0}};

        // Collect NeighD + softmax for all conditions
        int total_ood = N*4; // 4 OOD conditions × N
        float*all_nd=(float*)malloc((N+total_ood)*4);
        float*all_sm=(float*)malloc((N+total_ood)*4);
        float*all_tc=(float*)malloc((N+total_ood)*4);
        int*all_is_id=(int*)malloc((N+total_ood)*4);

        printf("  %-12s %8s %8s %8s %8s\n","Condition","NeighD","TimCV","SmConf","SmMax");
        int offset=0;
        for(int ci=0;ci<5;ci++){
            hipMemcpy(d_act[0],ood[ci].data,N*IN*4,hipMemcpyHostToDevice);
            for(int l=0;l<depth;l++){int relu=(l<depth-1)?1:0;
                mm_neighd<<<N,lout[l]>>>(d_act[l],d_W[l],d_b[l],d_act[l+1],
                    d_neighd,d_timing,N,lin[l],lout[l],relu);}
            argmax_k<<<(N+255)/256,256>>>(d_act[depth],d_pred,N,OUT);
            hipDeviceSynchronize();
            hipMemcpy(h_neighd,d_neighd,N*4,hipMemcpyDeviceToHost);
            hipMemcpy(h_timing,d_timing,N*4,hipMemcpyDeviceToHost);
            hipMemcpy(h_logits,d_act[depth],N*OUT*4,hipMemcpyDeviceToHost);

            float nd_mean=0,tc_mean=0,sm_mean=0,sm_max_mean=0;
            for(int i=0;i<N;i++){
                nd_mean+=h_neighd[i]; tc_mean+=h_timing[i];
                // Softmax
                float mx=-1e30;for(int c=0;c<OUT;c++)if(h_logits[i*OUT+c]>mx)mx=h_logits[i*OUT+c];
                float sum=0;for(int c=0;c<OUT;c++)sum+=expf(h_logits[i*OUT+c]-mx);
                float best=0;for(int c=0;c<OUT;c++){float pp=expf(h_logits[i*OUT+c]-mx)/sum;if(pp>best)best=pp;}
                sm_mean+=best;
                // Softmax entropy
                float ent=0;for(int c=0;c<OUT;c++){float pp=expf(h_logits[i*OUT+c]-mx)/sum;ent-=pp*logf(pp+1e-10);}
                sm_max_mean+=ent;

                // Store for AUC
                all_nd[offset+i]=h_neighd[i];
                all_tc[offset+i]=h_timing[i];
                all_sm[offset+i]=best;
                all_is_id[offset+i]=ood[ci].is_id;
            }
            printf("  %-12s %8.3f %8.4f %8.4f %8.4f  %s\n",
                   ood[ci].nm,nd_mean/N,tc_mean/N,sm_mean/N,sm_max_mean/N,
                   ood[ci].is_id?"ID":"OOD");
            offset+=N;
        }

        // AUC: ID vs all OOD combined
        printf("\n--- AUC: ID vs OOD (all OOD combined) ---\n");
        int total=offset;
        float auc_nd=compute_auc(all_nd,all_is_id,total);
        float auc_sm=compute_auc(all_sm,all_is_id,total);
        float auc_tc=compute_auc(all_tc,all_is_id,total);
        // Combined: normalize and sum
        float nd_min=1e30,nd_max=-1e30,sm_min=1e30,sm_max=-1e30;
        for(int i=0;i<total;i++){
            if(all_nd[i]<nd_min)nd_min=all_nd[i];if(all_nd[i]>nd_max)nd_max=all_nd[i];
            if(all_sm[i]<sm_min)sm_min=all_sm[i];if(all_sm[i]>sm_max)sm_max=all_sm[i];}
        float*all_combo=(float*)malloc(total*4);
        float best_combo_auc=0;float best_alpha=0;
        for(float alpha=0;alpha<=1.01;alpha+=0.1){
            for(int i=0;i<total;i++){
                float nd_n=(all_nd[i]-nd_min)/(nd_max-nd_min+1e-10);
                float sm_n=(all_sm[i]-sm_min)/(sm_max-sm_min+1e-10);
                all_combo[i]=alpha*sm_n+(1-alpha)*nd_n;
            }
            float auc=compute_auc(all_combo,all_is_id,total);
            if(auc>best_combo_auc){best_combo_auc=auc;best_alpha=alpha;}
        }

        printf("  NeighD AUC:    %.4f\n",auc_nd);
        printf("  TimCV AUC:     %.4f\n",auc_tc);
        printf("  Softmax AUC:   %.4f\n",auc_sm);
        printf("  Best combo:    %.4f (α=%.1f sm + %.1f nd)\n",
               best_combo_auc,best_alpha,1-best_alpha);
        if(best_combo_auc>auc_sm+0.005)
            printf("  >>> SHADOW IMPROVES OOD DETECTION! (+%.4f) <<<\n",best_combo_auc-auc_sm);
        else if(auc_nd>auc_sm+0.005)
            printf("  >>> NeighD ALONE beats softmax! <<<\n");

        // ── REJECTION CURVE ──
        printf("\n--- Rejection: accuracy on remaining after rejecting lowest-NeighD ---\n");
        // Get accuracy + NeighD for ID only
        hipMemcpy(d_act[0],h_img,N*IN*4,hipMemcpyHostToDevice);
        for(int l=0;l<depth;l++){int relu=(l<depth-1)?1:0;
            mm_neighd<<<N,lout[l]>>>(d_act[l],d_W[l],d_b[l],d_act[l+1],
                d_neighd,d_timing,N,lin[l],lout[l],relu);}
        argmax_k<<<(N+255)/256,256>>>(d_act[depth],d_pred,N,OUT);
        hipDeviceSynchronize();
        hipMemcpy(h_neighd,d_neighd,N*4,hipMemcpyDeviceToHost);
        hipMemcpy(h_pred,d_pred,N*4,hipMemcpyDeviceToHost);
        hipMemcpy(h_logits,d_act[depth],N*OUT*4,hipMemcpyDeviceToHost);

        // Sort by NeighD descending (highest = most confident)
        int*sorted_idx=(int*)malloc(N*4);
        for(int i=0;i<N;i++)sorted_idx[i]=i;
        // Simple selection sort on first few percentiles
        // Actually just compute rejection accuracy directly
        printf("  %-8s %10s %10s\n","Reject%","AccNeighD","AccSoftmax");
        for(int rp : {0,5,10,20,30,50}){
            int n_reject=N*rp/100;
            int n_keep=N-n_reject;

            // NeighD: reject lowest
            // Find threshold
            float*nd_copy=(float*)malloc(N*4);
            memcpy(nd_copy,h_neighd,N*4);
            // Partial sort to find n_reject-th smallest
            for(int i=0;i<n_reject&&i<N;i++){
                int mi=i;for(int j=i+1;j<N;j++)if(nd_copy[j]<nd_copy[mi])mi=j;
                float t=nd_copy[i];nd_copy[i]=nd_copy[mi];nd_copy[mi]=t;
            }
            float nd_thresh=(n_reject>0&&n_reject<N)?nd_copy[n_reject-1]:-1e30;

            int corr_nd=0,kept_nd=0;
            for(int i=0;i<N;i++){
                if(h_neighd[i]>nd_thresh||kept_nd>=n_keep) continue;
                // Actually: keep highest NeighD
            }
            // Simpler: sort indices by NeighD descending, take top n_keep
            // Use reservoir sampling for simplicity
            int*keep_nd=(int*)malloc(N*4);
            for(int i=0;i<N;i++)keep_nd[i]=i;
            // Sort by NeighD descending
            for(int i=0;i<n_keep&&i<N;i++){
                int mx=i;for(int j=i+1;j<N;j++)if(h_neighd[keep_nd[j]]>h_neighd[keep_nd[mx]])mx=j;
                int t=keep_nd[i];keep_nd[i]=keep_nd[mx];keep_nd[mx]=t;
            }
            int cnd=0;for(int i=0;i<n_keep;i++)if(h_pred[keep_nd[i]]==h_lbl[keep_nd[i]])cnd++;

            // Softmax: reject lowest confidence
            float*sm_conf=(float*)malloc(N*4);
            for(int i=0;i<N;i++){
                float mx=-1e30;for(int c=0;c<OUT;c++)if(h_logits[i*OUT+c]>mx)mx=h_logits[i*OUT+c];
                float sum=0;for(int c=0;c<OUT;c++)sum+=expf(h_logits[i*OUT+c]-mx);
                float best=0;for(int c=0;c<OUT;c++){float pp=expf(h_logits[i*OUT+c]-mx)/sum;if(pp>best)best=pp;}
                sm_conf[i]=best;
            }
            int*keep_sm=(int*)malloc(N*4);
            for(int i=0;i<N;i++)keep_sm[i]=i;
            for(int i=0;i<n_keep&&i<N;i++){
                int mx=i;for(int j=i+1;j<N;j++)if(sm_conf[keep_sm[j]]>sm_conf[keep_sm[mx]])mx=j;
                int t=keep_sm[i];keep_sm[i]=keep_sm[mx];keep_sm[mx]=t;
            }
            int csm=0;for(int i=0;i<n_keep;i++)if(h_pred[keep_sm[i]]==h_lbl[keep_sm[i]])csm++;

            printf("  %-8d %9.2f%% %9.2f%%\n",rp,
                   100.0f*cnd/n_keep,100.0f*csm/n_keep);

            free(nd_copy);free(keep_nd);free(sm_conf);free(keep_sm);
        }

        free(all_nd);free(all_sm);free(all_tc);free(all_is_id);free(all_combo);
        free(h_neighd);free(h_timing);free(h_logits);free(h_pred);free(sorted_idx);
        for(int l=0;l<depth;l++){hipFree(d_W[l]);hipFree(d_b[l]);}
        for(int l=0;l<=depth;l++)hipFree(d_act[l]);
        hipFree(d_pred);hipFree(d_neighd);hipFree(d_timing);
    }

    free(h_img);free(noise_img);free(ones_img);free(shuf_img);free(inv_img);free(h_lbl);
    printf("\n=== z2418 COMPLETE ===\n");
    return 0;
}
