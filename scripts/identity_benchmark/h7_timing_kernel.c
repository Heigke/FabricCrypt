/* H7 timing/jitter PUF kernel. Pins to a core, runs a FIXED dependent-ALU chain (no memory,
 * cannot be optimized away), times each block with rdtsc, and reads APERF/MPERF to get the
 * effective delivered frequency (delivered-vs-requested = die voltage/frequency margin = dopant-dependent).
 *
 * Args: core_id n_blocks ops_per_block
 * Output (stdout): one line "core med p10 p50 p90 std n_clip aperf_ratio" + raw deltas to stderr-free file is not used.
 * Compile: gcc -O2 -o h7_timing_kernel h7_timing_kernel.c
 * rdmsr of APERF/MPERF needs root (/dev/cpu/N/msr).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sched.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

static inline uint64_t rdtscp(void){ unsigned a,d,c; __asm__ volatile("rdtscp":"=a"(a),"=d"(d),"=c"(c)); return ((uint64_t)d<<32)|a; }

static uint64_t rdmsr(int core, uint32_t reg){
    char path[64]; snprintf(path,sizeof(path),"/dev/cpu/%d/msr",core);
    int fd=open(path,O_RDONLY); if(fd<0) return 0; uint64_t v=0;
    if(pread(fd,&v,8,reg)!=8) v=0; close(fd); return v;
}

static int cmp(const void*a,const void*b){ uint64_t x=*(const uint64_t*)a,y=*(const uint64_t*)b; return (x>y)-(x<y); }

int main(int argc,char**argv){
    int core=atoi(argv[1]); long nb=atol(argv[2]); long ops=atol(argv[3]);
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(core,&s);
    if(sched_setaffinity(0,sizeof(s),&s)!=0){ fprintf(stderr,"affinity fail\n"); return 1; }
    uint64_t *d=malloc(sizeof(uint64_t)*nb);
    volatile uint64_t sink=0;
    /* warm up */
    for(long w=0;w<5;w++){ uint64_t x=w+1; for(long i=0;i<ops;i++){ x=x*6364136223846793005ULL+1442695040888963407ULL; x^=x>>21; } sink+=x; }
    uint64_t ap0=rdmsr(core,0xE8), mp0=rdmsr(core,0xE7);
    for(long b=0;b<nb;b++){
        uint64_t x=(uint64_t)b*2654435761u+1;
        uint64_t t0=rdtscp();
        for(long i=0;i<ops;i++){ x=x*6364136223846793005ULL+1442695040888963407ULL; x^=x>>21; }  /* dependent chain */
        uint64_t t1=rdtscp();
        sink+=x; d[b]=t1-t0;
    }
    uint64_t ap1=rdmsr(core,0xE8), mp1=rdmsr(core,0xE7);
    qsort(d,nb,sizeof(uint64_t),cmp);
    double med=d[nb/2], p10=d[nb/10], p90=d[nb*9/10];
    double mean=0; for(long b=0;b<nb;b++) mean+=d[b]; mean/=nb;
    double var=0; for(long b=0;b<nb;b++){ double e=d[b]-mean; var+=e*e; } var/=nb;
    double aratio = (mp1>mp0)? (double)(ap1-ap0)/(double)(mp1-mp0) : 0.0;
    /* n_clip = count of blocks > 1.5*median (jitter tail = interruptions/marginality) */
    long nclip=0; for(long b=0;b<nb;b++) if(d[b]>1.5*med) nclip++;
    printf("%d %.0f %.0f %.0f %.3f %ld %.5f\n", core, p10, med, p90, var>0?var:0, nclip, aratio);
    (void)sink; free(d); return 0;
}
