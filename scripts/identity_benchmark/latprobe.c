/* latprobe.c — driven cache-latency pointer-chase RESERVOIR probe.
 *
 * The memory hierarchy is a nonlinear dynamical system with state: access latency is a
 * THRESHOLD function (cache hit ~ few ns vs DRAM miss ~ 100ns) of the access HISTORY (which
 * lines are currently resident). We drive it with a binary stream u and read per-step latency.
 * If the die's cache dynamics do genuine nonlinear-temporal computation, a linear readout of
 * these latencies can compute delayed-XOR/parity of u that a linear model of u cannot.
 *
 * Two cache-conflicting regions A (low half) and B (high half) of a pointer-chase cycle.
 * Per step t: chase HOPS hops starting from the region selected by u_t; the persistent chase
 * pointer carries state across steps. u=1 -> region A, u=0 -> region B. Accessing one region
 * evicts the other (working set > cache) so latency at t depends nonlinearly on recent u.
 *
 * Build: gcc -O2 -o latprobe latprobe.c
 * Run:   ./latprobe <u_file> <out_file> [n_lines] [hops]
 *        u_file: one 0/1 per line; out_file: one latency (cycles/hop *100) per step.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static inline uint64_t rdtsc_begin(void) {
    unsigned hi, lo;
    __asm__ __volatile__("lfence\n\trdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}
static inline uint64_t rdtsc_end(void) {
    unsigned hi, lo;
    __asm__ __volatile__("rdtscp\n\tlfence" : "=a"(lo), "=d"(hi) :: "rcx");
    return ((uint64_t)hi << 32) | lo;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s u_file out_file [n_lines] [hops]\n", argv[0]); return 1; }
    /* Two SEPARATE chase buffers: a small HOT one (~ L2) and a large COLD one (~ multi-L3).
     * u=1 -> chase HOT (fast, unless evicted); u=0 -> chase COLD (slow, and EVICTS HOT).
     * We ALWAYS read a fixed small HOT readout-probe each step -> its latency depends on how
     * long since HOT was last refreshed = nonlinear fading memory of u history. */
    long NS = (argc > 3) ? atol(argv[3]) : (1L << 15);   /* hot: 32K * 8B = 256KB (~L2) */
    long NL = (argc > 4) ? atol(argv[4]) : (1L << 23);   /* cold: 8M * 8B = 64MB (DRAM) */
    long HOPS = 64, RDPROBE = 32;

    uint64_t *bs = aligned_alloc(64, (size_t)NS * sizeof(uint64_t));
    uint64_t *bl = aligned_alloc(64, (size_t)NL * sizeof(uint64_t));
    if (!bs || !bl) { fprintf(stderr, "alloc fail\n"); return 1; }
    srand(12345);
    long *perm = malloc((size_t)NL * sizeof(long));
    for (long i = 0; i < NS; i++) perm[i] = i;
    for (long i = NS - 1; i > 0; i--) { long j = rand() % (i + 1); long t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    for (long i = 0; i < NS; i++) bs[perm[i]] = perm[(i + 1) % NS];
    for (long i = 0; i < NL; i++) perm[i] = i;
    for (long i = NL - 1; i > 0; i--) { long j = rand() % (i + 1); long t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    for (long i = 0; i < NL; i++) bl[perm[i]] = perm[(i + 1) % NL];
    free(perm);

    FILE *uf = fopen(argv[1], "r");
    if (!uf) { fprintf(stderr, "no u_file\n"); return 1; }
    int *u = malloc(sizeof(int) * (1 << 22));
    long L = 0; int v;
    while (fscanf(uf, "%d", &v) == 1) u[L++] = v;
    fclose(uf);

    FILE *of = fopen(argv[2], "w");
    volatile uint64_t sink = 0;
    long ps = 0, pl = 0;
    for (long k = 0; k < NS; k++) ps = bs[ps];   /* warm hot */
    for (long t = 0; t < L; t++) {
        /* drive: u=1 refreshes HOT; u=0 thrashes COLD (evicts hot) */
        if (u[t]) { for (long h = 0; h < HOPS; h++) ps = bs[ps]; }
        else      { for (long h = 0; h < HOPS; h++) pl = bl[pl]; }
        /* fixed readout probe of HOT: its latency = nonlinear memory of "time since hot refresh" */
        uint64_t t0 = rdtsc_begin();
        for (long h = 0; h < RDPROBE; h++) ps = bs[ps];
        uint64_t t1 = rdtsc_end();
        sink ^= ps ^ pl;
        long lat = (long)((t1 - t0) * 100 / RDPROBE);
        fprintf(of, "%ld\n", lat);
    }
    fclose(of);
    free(bs); free(bl);
    if (sink == 0x1234) fprintf(stderr, "%llu\n", (unsigned long long)sink);  /* defeat opt */
    free(u);
    return 0;
}
