// Branch-predictor / stride-prefetcher state machine as a nonlinear-with-memory
// timing reservoir. The hardware branch predictor uses SATURATING up/down
// counters (a bounded nonlinearity) indexed by branch history (memory). We
// drive ONE branch with the input bit sequence u[t] and read the timing of a
// probe set of branches whose prediction state is coupled through the shared
// predictor tables (history register + pattern-history-table aliasing).
//
// Why this could do XOR where contention could not: the global history register
// shifts in the outcome of EVERY taken/not-taken decision, and the PHT entry is
// selected by an XOR/hash of (address, history). So the *latency* of a probe
// branch depends nonlinearly on the recent SEQUENCE of drive bits, not just
// their sum. Saturation makes it nonlinear; history shift makes it fading-memory.
//
// Build: gcc -O2 -o predictor_reservoir predictor_reservoir.c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sched.h>
#include <pthread.h>
#include <x86intrin.h>

static inline uint64_t rdtscp_now(void){ unsigned a; return __rdtscp(&a); }
static void pin(int cpu){ cpu_set_t s;CPU_ZERO(&s);CPU_SET(cpu,&s);
    pthread_setaffinity_np(pthread_self(),sizeof(s),&s); }

// Probe branches: an array of data-dependent branches. We time a fixed pass and
// also the drive branch. Misprediction shows up as added cycles.
#define PROBEN 64
static volatile int sink=0;

// drive a branch many times with a given direction pattern derived from history
static inline int drive_branch(int dir, int x){
    // a real conditional branch the predictor will learn
    if(dir){ return x*3+1; } else { return x^0x5a5a; }
}

int main(int argc,char**argv){
    int STEPS = argc>1?atoi(argv[1]):4000;
    int REPS  = argc>2?atoi(argv[2]):512;   // how many times we exercise the drive branch per step
    pin(0);

    // pseudo-random probe pattern (fixed) that the predictor must track alongside
    int probe_pat[PROBEN];
    srand(777);
    for(int i=0;i<PROBEN;i++) probe_pat[i]=rand()&1;

    uint32_t lfsr=0xACE1u;
    printf("step,u,f0,f1,f2,f3,f4\n");

    // warmup
    int acc=0;
    for(int w=0;w<2000;w++){ acc+=drive_branch(w&1,w); }

    for(int t=0;t<STEPS;t++){
        uint32_t bit=lfsr&1u; lfsr=(lfsr>>1)^(-(lfsr&1u)&0xB400u);

        // 1) train the drive branch with the CURRENT bit REPS times.
        //    This pushes the saturating counter + shifts global history.
        for(int r=0;r<REPS;r++){ acc+=drive_branch((int)bit, r); }

        // 2) probe: time a fixed pattern of branches whose predictor entries
        //    alias with the drive branch through shared history/PHT.
        uint64_t t0=rdtscp_now();
        for(int i=0;i<PROBEN;i++){ acc+=drive_branch(probe_pat[i], i); }
        uint64_t t1=rdtscp_now();
        double probe_lat=(double)(t1-t0)/PROBEN;

        // 3) measure mispredict cost on a SECOND probe that depends on bit-history:
        //    branch direction = XOR of two PRNG bits (forces predictor to mix)
        uint64_t t2=rdtscp_now();
        int hh=0;
        for(int i=0;i<PROBEN;i++){ int dd=probe_pat[i]^probe_pat[(i+1)&(PROBEN-1)];
            acc+=drive_branch(dd,i); hh+=dd; }
        uint64_t t3=rdtscp_now();
        double mix_lat=(double)(t3-t2)/PROBEN;

        // 4) latency to re-learn drive branch flipped (sensitivity to state)
        uint64_t t4=rdtscp_now();
        for(int r=0;r<32;r++){ acc+=drive_branch((int)(bit^1),r); }
        uint64_t t5=rdtscp_now();
        double flip_lat=(double)(t5-t4)/32;

        // 5) back-to-back tsc jitter while predictor settling
        uint64_t pa=rdtscp_now(),mx=0,prev=pa; double sum=0;int N=32;
        for(int j=0;j<N;j++){uint64_t a=rdtscp_now();uint64_t d=a-prev;prev=a;if(d>mx)mx=d;sum+=d;}
        double jit=(double)mx;

        printf("%d,%u,%.3f,%.3f,%.3f,%.3f,%.3f\n",t,bit,probe_lat,mix_lat,flip_lat,jit,sum/N);
    }
    sink=acc; if(sink==0x12345) fprintf(stderr," ");
    return 0;
}
