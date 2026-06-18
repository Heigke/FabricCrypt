/* Task D (12B) — atomic contention throughput between two pinned threads.
 * Usage: ./atomic_contention <core_a> <core_b> <duration_ms>
 * Prints: increments_a increments_b total
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <stdatomic.h>

static atomic_uint_fast64_t counter = 0;
static atomic_int stop_flag = 0;

typedef struct { int core; uint64_t inc; } targ_t;

static void pin(int c) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(c, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

static void* worker(void *a) {
    targ_t *t = (targ_t*)a;
    pin(t->core);
    uint64_t n = 0;
    while (!atomic_load_explicit(&stop_flag, memory_order_relaxed)) {
        atomic_fetch_add_explicit(&counter, 1, memory_order_seq_cst);
        n++;
    }
    t->inc = n;
    return NULL;
}

int main(int argc, char**argv) {
    int ca = atoi(argv[1]);
    int cb = atoi(argv[2]);
    int dur_ms = atoi(argv[3]);
    pthread_t ta, tb;
    targ_t aa = {ca, 0}, bb = {cb, 0};
    pthread_create(&ta, NULL, worker, &aa);
    pthread_create(&tb, NULL, worker, &bb);
    struct timespec ts = { dur_ms/1000, (dur_ms%1000)*1000000L };
    nanosleep(&ts, NULL);
    atomic_store(&stop_flag, 1);
    pthread_join(ta, NULL);
    pthread_join(tb, NULL);
    printf("%lu %lu %lu\n", aa.inc, bb.inc, aa.inc+bb.inc);
    return 0;
}
