/* Task B — TSC offset between core 0 and core N, 10000 paired reads.
 * Usage: ./tsc_inter_core <other_core> <n_pairs> > out.bin
 * Output: binary, n_pairs * (int64 t0, int64 tN) pairs.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sched.h>
#include <pthread.h>
#include <unistd.h>

static inline uint64_t rdtscp(void) {
    unsigned aux;
    uint64_t t;
    unsigned lo, hi;
    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux) :: "memory");
    t = ((uint64_t)hi << 32) | lo;
    return t;
}

static void pin(int core) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(core, &s);
    if (sched_setaffinity(0, sizeof(s), &s) != 0) { perror("setaffinity"); exit(1); }
}

int main(int argc, char**argv) {
    if (argc < 3) { fprintf(stderr,"usage: %s other_core n\n", argv[0]); return 1; }
    int other = atoi(argv[1]);
    int n = atoi(argv[2]);
    int64_t *buf = malloc(sizeof(int64_t)*2*n);
    for (int i=0;i<n;i++) {
        pin(0);
        uint64_t t0 = rdtscp();
        pin(other);
        uint64_t tN = rdtscp();
        buf[2*i] = (int64_t)t0;
        buf[2*i+1] = (int64_t)tN;
    }
    fwrite(buf, sizeof(int64_t), 2*n, stdout);
    free(buf);
    return 0;
}
