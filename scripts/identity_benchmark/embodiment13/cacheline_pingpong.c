/* Task E (12B) — cache-line transfer latency between two cores via ping-pong.
 * Thread A writes generation g, thread B waits to see g then writes g+1.
 * Round-trip RTT/2 = one-way latency. Report cycles per transfer.
 * Usage: ./cacheline_pingpong <core_a> <core_b> <iters>
 * Prints: median_cycles_per_transfer mean std
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <string.h>

static struct { _Atomic uint64_t gen; char pad[64-sizeof(uint64_t)]; } cell __attribute__((aligned(64)));

static int ITERS = 100000;
static int CORE_A=0, CORE_B=1;
static uint64_t *samples;

static inline uint64_t rdtscp(void){unsigned aux,lo,hi;__asm__ __volatile__("rdtscp":"=a"(lo),"=d"(hi),"=c"(aux)::"memory");return ((uint64_t)hi<<32)|lo;}

static void pin(int c){cpu_set_t s;CPU_ZERO(&s);CPU_SET(c,&s);pthread_setaffinity_np(pthread_self(),sizeof(s),&s);}

static void* thread_a(void *u) {
    (void)u; pin(CORE_A);
    for (int i=0;i<ITERS;i++) {
        uint64_t expect = (uint64_t)(2*i);
        while (atomic_load_explicit(&cell.gen, memory_order_acquire) != expect) {}
        uint64_t t0 = rdtscp();
        atomic_store_explicit(&cell.gen, expect+1, memory_order_release);
        while (atomic_load_explicit(&cell.gen, memory_order_acquire) != expect+2) {}
        uint64_t t1 = rdtscp();
        samples[i] = t1 - t0;
    }
    return NULL;
}
static void* thread_b(void *u) {
    (void)u; pin(CORE_B);
    for (int i=0;i<ITERS;i++) {
        uint64_t expect = (uint64_t)(2*i+1);
        while (atomic_load_explicit(&cell.gen, memory_order_acquire) != expect) {}
        atomic_store_explicit(&cell.gen, expect+1, memory_order_release);
    }
    return NULL;
}

int main(int argc, char**argv) {
    CORE_A = atoi(argv[1]);
    CORE_B = atoi(argv[2]);
    ITERS  = atoi(argv[3]);
    samples = calloc(ITERS, sizeof(uint64_t));
    atomic_store(&cell.gen, 0);
    pthread_t ta, tb;
    pthread_create(&tb, NULL, thread_b, NULL);
    pthread_create(&ta, NULL, thread_a, NULL);
    pthread_join(ta, NULL);
    pthread_join(tb, NULL);
    /* dump binary uint64 */
    fwrite(samples, sizeof(uint64_t), ITERS, stdout);
    free(samples);
    return 0;
}
