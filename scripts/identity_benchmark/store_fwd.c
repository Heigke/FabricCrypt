/* store_fwd.c — store-to-load forwarding partial-overlap COINCIDENCE probe.
 * Per step two independent bits a,b. Store 8 bytes at buf+a*4, then load 8 bytes at buf+b*4
 * (same 16B window). If a==b -> full overlap -> store-buffer forwards (fast). If a!=b -> PARTIAL
 * overlap -> forwarding stalls (~10-15 cyc penalty). So the latency is HIGH iff a XOR b — a
 * nonlinearity the SILICON computes on two independent inputs (not a load aggregate).
 * Build: gcc -O2 -o store_fwd store_fwd.c ; Run: ./store_fwd a_file b_file out_file [core] */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sched.h>
static inline uint64_t rb(void){unsigned h,l;__asm__ __volatile__("lfence\n\trdtsc":"=a"(l),"=d"(h));return((uint64_t)h<<32)|l;}
static inline uint64_t re(void){unsigned h,l;__asm__ __volatile__("rdtscp\n\tlfence":"=a"(l),"=d"(h)::"rcx");return((uint64_t)h<<32)|l;}
int main(int c,char**v){
  if(c<4){fprintf(stderr,"usage: %s a b out [core]\n",v[0]);return 1;}
  int core=(c>4)?atoi(v[4]):4; cpu_set_t s;CPU_ZERO(&s);CPU_SET(core,&s);sched_setaffinity(0,sizeof(s),&s);
  FILE*af=fopen(v[1],"r"),*bf=fopen(v[2],"r");int*A=malloc(4<<22),*B=malloc(4<<22);long L=0,Lb=0;int x;
  while(fscanf(af,"%d",&x)==1)A[L++]=x; while(fscanf(bf,"%d",&x)==1)B[Lb++]=x; fclose(af);fclose(bf); if(Lb<L)L=Lb;
  FILE*of=fopen(v[3],"w");
  char*buf=aligned_alloc(64,256); for(int i=0;i<256;i++)buf[i]=0;
  volatile uint64_t sink=0;
  for(long t=0;t<L;t++){
    volatile uint64_t*sp=(volatile uint64_t*)(buf + A[t]*4);   /* store addr depends on a */
    volatile uint64_t*lp=(volatile uint64_t*)(buf + B[t]*4);   /* load addr depends on b */
    *sp = sink + 0x9e3779b9ULL;                                /* the store */
    uint64_t t0=rb(); uint64_t val=*lp; uint64_t t1=re();      /* the dependent load, timed */
    sink ^= val;
    fprintf(of,"%ld\n",(long)(t1-t0));
  }
  fclose(of); if(sink==0x1234)fprintf(stderr,"%llu\n",(unsigned long long)sink); return 0;
}
