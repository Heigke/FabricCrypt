// Lens-2 timing reservoir probe for gfx1151 (Strix Halo).
// Hypothesis: an SMT-shared / cache-shared contention channel is a NONLINEAR
// dynamical system with fading memory. We drive it with a binary input u[t]
// and read a vector of timing features x[t]. A LINEAR readout of x[t] is then
// asked to compute delayed-XOR(u[t-d], u[t-d-1]). If it beats a linear readout
// of the raw drive history, the channel is doing genuine nonlinear-with-memory
// computation.
//
// Drive mechanism: the "load" thread, when u[t]==1, pounds a shared cache line
// region with atomic RMW (lock cmpxchg) which contends the L3 / coherence fabric
// and the SMT pipeline. When u[t]==0 it spins on a private line (no shared
// traffic). The probe thread measures pointer-chase latency through a buffer
// sized to live in L2/L3, plus a few timing-divergence features.
//
// Build: gcc -O2 -pthread -o contention_reservoir contention_reservoir.c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <x86intrin.h>
#include <stdatomic.h>

static inline uint64_t rdtscp_now(void){ unsigned a; return __rdtscp(&a); }

// ---- shared contention target ----
#define SHARED_LINES 4096            // 4096*64 = 256 KiB shared region
static _Atomic uint64_t *shared_region;
static volatile int drive_bit = 0;   // set by main loop each step
static volatile int run_load = 1;
static volatile uint64_t step_gen = 0; // generation counter to sync load thread

static void pin(int cpu){
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu,&s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

// load thread: when drive_bit==1 hammer shared region with atomics
static int LOAD_CPU = 1;
static void* load_fn(void* arg){
    (void)arg; pin(LOAD_CPU);
    uint64_t priv = 0; uint64_t i=0;
    while(run_load){
        if(drive_bit){
            // contended atomic RMW across shared region -> coherence traffic
            for(int k=0;k<256;k++){
                size_t idx = (i*2654435761u) & (SHARED_LINES-1);
                atomic_fetch_add_explicit(&shared_region[idx], 1, memory_order_seq_cst);
                i++;
            }
        } else {
            // private spin, no shared traffic
            for(int k=0;k<256;k++){ priv += k*i; i++; }
        }
    }
    // prevent optimizing away
    if(priv==0xdeadbeef) printf("%lu",(unsigned long)priv);
    return NULL;
}

// pointer-chase buffer (probe thread reads latency through it)
#define CHASE_NODES (1<<15)          // 32768 nodes * 64B = 2 MiB (spills L2->L3)
static uint64_t *chase;              // each slot holds next index*8 offset layout
static uint32_t *perm;

static void build_chase(void){
    chase = aligned_alloc(64, (size_t)CHASE_NODES*64);
    perm  = malloc(CHASE_NODES*sizeof(uint32_t));
    for(uint32_t i=0;i<CHASE_NODES;i++) perm[i]=i;
    // Fisher-Yates
    for(uint32_t i=CHASE_NODES-1;i>0;i--){ uint32_t j=rand()%(i+1); uint32_t t=perm[i];perm[i]=perm[j];perm[j]=t; }
    // build cycle: chase[node*8] = next node address offset
    for(uint32_t i=0;i<CHASE_NODES;i++){
        uint32_t cur=perm[i], nxt=perm[(i+1)%CHASE_NODES];
        chase[(size_t)cur*8] = (uint64_t)nxt;
    }
}

// one measurement step: returns a feature vector
#define NFEAT 5
static void probe_step(double *feat){
    // warm a pointer chase of N hops, time it
    const int HOPS=2048;
    uint64_t cur=0;
    uint64_t t0=rdtscp_now();
    for(int h=0;h<HOPS;h++){ cur = chase[(size_t)cur*8]; }
    uint64_t t1=rdtscp_now();
    double chase_lat = (double)(t1-t0)/HOPS;
    // sink
    if(cur==0xffffffff) printf(" ");

    // back-to-back rdtscp delta samples -> jitter / heavy tail
    uint64_t mn=~0ull, mx=0; double sum=0; double sum2=0; int N=64;
    uint64_t prev=rdtscp_now();
    for(int j=0;j<N;j++){
        uint64_t a=rdtscp_now();
        uint64_t d=a-prev; prev=a;
        if(d<mn)mn=d; if(d>mx)mx=d; sum+=d; sum2+=(double)d*d;
    }
    double mean=sum/N;
    double var=sum2/N-mean*mean; if(var<0)var=0;

    feat[0]=chase_lat;
    feat[1]=mean;          // rdtscp back-to-back mean
    feat[2]=(double)mx;    // max single delta (heavy tail)
    feat[3]=var;           // jitter variance
    feat[4]=(double)(mx-mn);// range
}

int main(int argc,char**argv){
    int PROBE_CPU = 0;     // sibling of LOAD_CPU=1 (SMT) by default
    int STEPS = 6000;
    int SETTLE = 400;      // micro-iterations of load applied per step before probe
    if(argc>1) STEPS=atoi(argv[1]);
    if(argc>2) LOAD_CPU=atoi(argv[2]);
    if(argc>3) PROBE_CPU=atoi(argv[3]);
    if(argc>4) SETTLE=atoi(argv[4]);

    srand(12345);
    shared_region = aligned_alloc(64,(size_t)SHARED_LINES*64);
    memset(shared_region,0,(size_t)SHARED_LINES*64);
    build_chase();

    pin(PROBE_CPU);
    pthread_t lt; pthread_create(&lt,NULL,load_fn,NULL);

    // deterministic PRBS-ish binary drive (reproducible). LFSR.
    uint32_t lfsr=0xACE1u;
    // header
    printf("step,u,f0,f1,f2,f3,f4\n");
    // warmup chase resident
    double f[NFEAT];
    for(int w=0;w<50;w++) probe_step(f);

    for(int t=0;t<STEPS;t++){
        // next drive bit
        uint32_t bit = lfsr & 1u;
        lfsr = (lfsr>>1) ^ (-(lfsr&1u) & 0xB400u);
        drive_bit = (int)bit;
        // let load thread apply this bit for SETTLE units (busy wait on probe cpu)
        // We simply spin a fixed amount so load thread runs concurrently.
        volatile uint64_t spin=0;
        for(int s=0;s<SETTLE;s++){ spin += s; }
        (void)spin;
        probe_step(f);
        printf("%d,%u,%.3f,%.3f,%.3f,%.3f,%.3f\n",t,bit,f[0],f[1],f[2],f[3],f[4]);
    }
    run_load=0; pthread_join(lt,NULL);
    return 0;
}
