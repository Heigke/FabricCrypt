
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>
static inline uint64_t rdtscp(void){unsigned aux,lo,hi;__asm__ __volatile__("rdtscp":"=a"(lo),"=d"(hi),"=c"(aux)::"memory");return ((uint64_t)hi<<32)|lo;}
int main(int argc,char**argv){int n=atoi(argv[1]);unsigned long long r;uint32_t *out=malloc(sizeof(uint32_t)*n);
for(int i=0;i<n;i++){uint64_t a=rdtscp();_rdrand64_step(&r);uint64_t b=rdtscp();uint64_t d=b-a;out[i]=(d>0xFFFFFFFFull)?0xFFFFFFFFu:(uint32_t)d;}
fwrite(out,4,n,stdout);free(out);return 0;}
