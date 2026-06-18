/* S3: Per-CCX cache line wakeup (cross-CCX invalidation latency).
 * Two threads: writer pins to core A, waiter pins to core B (different CCX).
 * Waiter spins reading a shared cacheline, writer flips a sequence number,
 * waiter records cycles from flip to observation.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <x86intrin.h>

static inline uint64_t rdtscp_serial(void) {
    unsigned aux; return __rdtscp(&aux);
}

typedef struct {
    volatile uint64_t seq;
    char pad[64 - sizeof(uint64_t)];
} cl_t __attribute__((aligned(64)));

static cl_t shared;
static volatile uint64_t writer_tsc;
static volatile int go = 0;
static int writer_core, waiter_core;
static int n_samples;
static uint64_t* samples;

static void pin(int core) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(core, &s);
    sched_setaffinity(0, sizeof(s), &s);
}

static void* writer_fn(void* arg) {
    (void)arg;
    pin(writer_core);
    /* warm */
    for (volatile int i = 0; i < 100000; i++);
    for (int i = 0; i < n_samples; i++) {
        /* wait some random spin so waiter is ready */
        for (volatile int j = 0; j < 2000 + (i & 1023); j++);
        _mm_mfence();
        uint64_t t0 = rdtscp_serial();
        writer_tsc = t0;
        shared.seq = (uint64_t)(i + 1);
        _mm_mfence();
        /* wait until waiter ack via go */
        while (__atomic_load_n(&go, __ATOMIC_ACQUIRE) != i+1);
    }
    return NULL;
}

static void* waiter_fn(void* arg) {
    (void)arg;
    pin(waiter_core);
    for (int i = 0; i < n_samples; i++) {
        uint64_t expected = (uint64_t)(i + 1);
        while (shared.seq != expected) { _mm_pause(); }
        uint64_t t1 = rdtscp_serial();
        uint64_t t0 = writer_tsc;
        samples[i] = t1 - t0;
        __atomic_store_n(&go, i+1, __ATOMIC_RELEASE);
    }
    return NULL;
}

int main(int argc, char** argv) {
    writer_core = argc > 1 ? atoi(argv[1]) : 0;
    waiter_core = argc > 2 ? atoi(argv[2]) : 8;
    n_samples   = argc > 3 ? atoi(argv[3]) : 4000;
    samples = (uint64_t*)calloc(n_samples, sizeof(uint64_t));
    shared.seq = 0;
    go = 0;
    pthread_t tw, tr;
    pthread_create(&tw, NULL, writer_fn, NULL);
    pthread_create(&tr, NULL, waiter_fn, NULL);
    pthread_join(tw, NULL);
    pthread_join(tr, NULL);
    fwrite(samples, sizeof(uint64_t), n_samples, stdout);
    free(samples);
    return 0;
}
