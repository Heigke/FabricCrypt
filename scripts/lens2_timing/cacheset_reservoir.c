// Cache-set eviction reservoir. A cache set has bounded associativity W. Resident
// lines = a small state vector. The drive bit decides whether we touch a set of
// "aggressor" addresses that map to the SAME cache set as our probe lines. When
// enough aggressors are touched, probe lines get EVICTED (a hard threshold ->
// nonlinearity). The state (which probe lines are still resident) persists across
// steps with decay (other activity slowly evicts) -> fading memory. The readout
// is the probe reload latency vector (hit=fast, miss=slow).
//
// XOR rationale: eviction is set-associative LRU, which is order-and-count
// sensitive. Whether probe line P survives depends NONLINEARLY on the sequence
// of recent drive bits (how many aggressors hit this set, in what order). This
// is the classic "billiard / threshold" nonlinearity that can mix bits.
//
// Build: gcc -O2 -o cacheset_reservoir cacheset_reservoir.c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sched.h>
#include <pthread.h>
#include <x86intrin.h>

static inline uint64_t rdtscp_now(void){ unsigned a; return __rdtscp(&a); }
static inline void mfence(void){ _mm_mfence(); }
static void pin(int cpu){ cpu_set_t s;CPU_ZERO(&s);CPU_SET(cpu,&s);
    pthread_setaffinity_np(pthread_self(),sizeof(s),&s); }

// Big buffer; we build eviction sets by stride. L2 here is 16-way? we sweep.
#define BUF_MB 64
static uint8_t *buf;

static inline uint64_t load_lat(volatile uint8_t* p){
    mfence();
    uint64_t t0=rdtscp_now();
    volatile uint8_t v=*p; (void)v;
    uint64_t t1=rdtscp_now();
    return t1-t0;
}

int main(int argc,char**argv){
    int STEPS = argc>1?atoi(argv[1]):4000;
    long STRIDE = argc>2?atol(argv[2]):4096;   // page-stride => same set in many caches
    int NPROBE = argc>3?atoi(argv[3]):8;       // probe lines (state vector size)
    int NAGG   = argc>4?atoi(argv[4]):24;      // aggressors touched when drive=1
    pin(0);

    size_t SZ=(size_t)BUF_MB*1024*1024;
    buf=aligned_alloc(4096,SZ); memset(buf,1,SZ);

    // probe addresses: same congruence class (stride apart), base offset 0
    uint8_t* probe[64];
    for(int i=0;i<NPROBE;i++) probe[i]=buf + (size_t)i*STRIDE;
    // aggressors: further along, same stride (alias same set)
    uint8_t* agg[128];
    for(int i=0;i<NAGG;i++) agg[i]=buf + (size_t)(NPROBE+i)*STRIDE;

    uint32_t lfsr=0xACE1u;
    printf("step,u,f0,f1,f2,f3,f4,f5,f6,f7\n");

    for(int t=0;t<STEPS;t++){
        uint32_t bit=lfsr&1u; lfsr=(lfsr>>1)^(-(lfsr&1u)&0xB400u);

        // (a) (re)install probe lines into cache: touch them
        volatile uint64_t s=0;
        for(int i=0;i<NPROBE;i++) s+=probe[i][0];

        // (b) DRIVE: if bit==1, touch aggressors -> evict some probe lines
        if(bit){ for(int i=0;i<NAGG;i++) s+=agg[i][0]; }
        // a little "background" activity always, gives decay/fading memory
        for(int i=0;i<NAGG/3;i++) s+=agg[i][0];

        // (c) READ state: reload latency of each probe line (hit vs miss)
        double f[8];
        for(int i=0;i<NPROBE && i<8;i++){
            // measure latency WITHOUT first touching (so we see if it survived)
            f[i]=(double)load_lat(probe[i]);
        }
        printf("%d,%u",t,bit);
        for(int i=0;i<8;i++) printf(",%.0f", i<NPROBE? f[i]:0.0);
        printf("\n");
        if(s==0x123456789ull) fprintf(stderr," ");
    }
    return 0;
}
