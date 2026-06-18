/* S2: TLB miss latency.
 * Allocate many 4K pages, randomized access pattern guaranteed to miss TLB.
 * Emit raw timing samples (uint64 little-endian).
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <x86intrin.h>

static inline uint64_t rdtscp_serial(void) {
    unsigned aux;
    return __rdtscp(&aux);
}

int main(int argc, char** argv) {
    /* ~64MB → 16384 pages, exceeds dTLB by >>10x to force walks */
    size_t n_pages = argc > 1 ? (size_t)atol(argv[1]) : 16384;
    int n_samples  = argc > 2 ? atoi(argv[2]) : 4000;
    size_t bytes = n_pages * 4096UL;

    void* p = mmap(NULL, bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return 1;
    /* touch every page (faulting + warm cache lines) */
    volatile uint8_t* b = (uint8_t*)p;
    for (size_t i = 0; i < n_pages; i++) b[i*4096] = (uint8_t)i;

    /* deterministic shuffle of page indices */
    size_t* order = (size_t*)malloc(sizeof(size_t)*n_pages);
    for (size_t i = 0; i < n_pages; i++) order[i] = i;
    uint64_t s = 0xC0FFEE + getpid();
    for (size_t i = n_pages-1; i > 0; i--) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        size_t j = s % (i+1);
        size_t t = order[i]; order[i] = order[j]; order[j] = t;
    }

    uint64_t* out = (uint64_t*)malloc(sizeof(uint64_t)*n_samples);
    size_t mask = 1;
    while (mask < n_pages) mask <<= 1;
    mask >>= 1; mask -= 1;
    if (mask == 0) mask = n_pages - 1;

    volatile uint8_t sink = 0;
    size_t cursor = 0;
    for (int i = 0; i < n_samples; i++) {
        size_t pg = order[cursor & mask];
        cursor += 1013;
        _mm_lfence();
        uint64_t t0 = rdtscp_serial();
        sink ^= b[pg*4096 + (i & 63)];
        uint64_t t1 = rdtscp_serial();
        out[i] = t1 - t0;
    }
    fwrite(out, sizeof(uint64_t), n_samples, stdout);
    free(out); free(order); munmap(p, bytes);
    return sink == 0xDE ? 1 : 0;
}
