/* S1: Branch predictor warmup latency.
 * Fills BTB with pseudo-random branches, measures TSC cost per iteration.
 * Emit binary: 8-byte little-endian uint64 per measurement.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>

static inline uint64_t rdtscp_serial(void) {
    unsigned aux;
    return __rdtscp(&aux);
}

int main(int argc, char** argv) {
    int n_branches = argc > 1 ? atoi(argv[1]) : 4096;     /* BTB targets */
    int n_iter     = argc > 2 ? atoi(argv[2]) : 4000;     /* measurements */
    int reps_per_iter = 256;                              /* branches per sample */

    /* Generate random branch directions per slot (deterministic seed) */
    unsigned char* dirs = (unsigned char*)malloc(n_branches);
    if (!dirs) return 1;
    uint32_t s = 0x12345 + getpid();
    for (int i = 0; i < n_branches; i++) {
        s = s * 1664525 + 1013904223;
        dirs[i] = (s >> 23) & 1;
    }

    uint64_t* out = (uint64_t*)malloc(sizeof(uint64_t) * n_iter);
    if (!out) return 1;

    volatile int sink = 0;
    /* warm cache for dirs */
    for (int i = 0; i < n_branches; i++) sink += dirs[i];

    for (int it = 0; it < n_iter; it++) {
        uint32_t r = s = s * 1664525 + 1013904223;
        uint64_t t0 = rdtscp_serial();
        _mm_lfence();
        int acc = 0;
        for (int k = 0; k < reps_per_iter; k++) {
            int idx = (r + k * 31) & (n_branches - 1);
            if (dirs[idx]) acc += idx; else acc -= idx;
            r = r * 1103515245 + 12345;
        }
        _mm_lfence();
        uint64_t t1 = rdtscp_serial();
        sink += acc;
        out[it] = t1 - t0;
    }

    fwrite(out, sizeof(uint64_t), n_iter, stdout);
    free(out); free(dirs);
    return sink == 0xdeadbeef ? 1 : 0;
}
