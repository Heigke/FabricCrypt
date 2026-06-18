/* latprobe_contention.c — SPATIAL contention reservoir probe (two simultaneous inputs).
 *
 * Temporal-memory probes (latprobe.c) failed: a single scalar low-pass latency is a 1-D linear
 * reservoir that can't do XOR. Here we test the OTHER physical nonlinearity: CAPACITY CONTENTION.
 * Three disjoint pointer-chase buffers share one L2:
 *   P  (small, ~64KB)  = readout target, want it resident
 *   A  (~half L2)      = driven by bit a
 *   B  (~half L2)      = driven by bit b
 * Sizes chosen so P+A fits, P+B fits, but P+A+B OVERFLOWS L2. Therefore probe P is evicted
 * (slow) ONLY when a AND b are both active this step -> lat_P encodes a genuine AND(a,b), a
 * physical threshold the silicon computes, not the readout.
 *
 * We emit THREE latencies per step: lat_P (~AND), lat_A (~NOT a), lat_B (~NOT b). A LINEAR readout
 * of {lat_A,lat_B,lat_P} can then form XOR = (1-a)+(1-b)-2*AND, which a linear model of (a,b) CANNOT.
 * That is the rigorous reservoir-necessity test: linear-on-state beats linear-on-input.
 *
 * Build: gcc -O2 -o latprobe_contention latprobe_contention.c
 * Run (pin to one core for a stable L2): taskset -c 3 ./latprobe_contention a_file b_file out_file [NP] [NA]
 *   a_file,b_file: one 0/1 per line; out_file: "lat_P lat_A lat_B" per step.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

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

static void make_ring(uint64_t *b, long n, long *perm) {
    for (long i = 0; i < n; i++) perm[i] = i;
    for (long i = n - 1; i > 0; i--) { long j = rand() % (i + 1); long t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    for (long i = 0; i < n; i++) b[perm[i]] = perm[(i + 1) % n];
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s a_file b_file out_file [NP] [NA]\n", argv[0]); return 1; }
    long NP = (argc > 4) ? atol(argv[4]) : (1L << 13);   /* P:  8K *8B =  64KB  (want resident) */
    long NA = (argc > 5) ? atol(argv[5]) : (3L << 20);   /* A:  3M *8B =  24MB  (~3/4 of 32MB L3) */
    long NB = NA;                                        /* B same; P+A or P+B < L3 < P+A+B (=>DRAM) */
    long RD = 256;                                       /* readout hops per probe (more => lower noise) */

    uint64_t *P = aligned_alloc(64, (size_t)NP * sizeof(uint64_t));
    uint64_t *A = aligned_alloc(64, (size_t)NA * sizeof(uint64_t));
    uint64_t *B = aligned_alloc(64, (size_t)NB * sizeof(uint64_t));
    long *perm = malloc((size_t)(NA > NP ? NA : NP) * sizeof(long));
    if (!P || !A || !B || !perm) { fprintf(stderr, "alloc fail\n"); return 1; }
    srand(12345);
    make_ring(P, NP, perm); make_ring(A, NA, perm); make_ring(B, NB, perm);
    free(perm);

    FILE *af = fopen(argv[1], "r"), *bf = fopen(argv[2], "r");
    if (!af || !bf) { fprintf(stderr, "no input\n"); return 1; }
    int *ua = malloc(sizeof(int) * (1 << 22)), *ub = malloc(sizeof(int) * (1 << 22));
    long L = 0, Lb = 0; int v;
    while (fscanf(af, "%d", &v) == 1) ua[L++] = v;
    while (fscanf(bf, "%d", &v) == 1) ub[Lb++] = v;
    fclose(af); fclose(bf);
    if (Lb < L) L = Lb;

    FILE *of = fopen(argv[3], "w");
    volatile uint64_t sink = 0;
    long pp = 0, pa = 0, pb = 0;
    volatile uint64_t acc = 0;
    for (long k = 0; k < NP; k++) pp = P[pp];   /* warm P resident */
    for (long t = 0; t < L; t++) {
        /* drive: each active bit STREAMS its whole 24MB buffer (1 access/line) at full bandwidth,
         * evicting by capacity. P (in L3) survives one buffer but BOTH together overflow L3 -> P to DRAM. */
        if (ua[t]) { uint64_t s = 0; for (long h = 0; h < NA; h += 8) s += A[h]; acc += s; }
        if (ub[t]) { uint64_t s = 0; for (long h = 0; h < NB; h += 8) s += B[h]; acc += s; }
        /* probe P first (the AND signal): slow iff BOTH a,b evicted it */
        uint64_t t0 = rdtsc_begin();
        for (long h = 0; h < RD; h++) pp = P[pp];
        uint64_t t1 = rdtsc_end();
        long latP = (long)((t1 - t0) * 100 / RD);
        /* probe A residency (~NOT a): touch a few A lines */
        uint64_t t2 = rdtsc_begin();
        for (long h = 0; h < RD; h++) pa = A[pa];
        uint64_t t3 = rdtsc_end();
        long latA = (long)((t3 - t2) * 100 / RD);
        /* probe B residency (~NOT b) */
        uint64_t t4 = rdtsc_begin();
        for (long h = 0; h < RD; h++) pb = B[pb];
        uint64_t t5 = rdtsc_end();
        long latB = (long)((t5 - t4) * 100 / RD);
        sink ^= pp ^ pa ^ pb ^ acc;
        fprintf(of, "%ld %ld %ld\n", latP, latA, latB);
    }
    fclose(of);
    free(P); free(A); free(B);
    if (sink == 0x1234) fprintf(stderr, "%llu\n", (unsigned long long)sink);
    free(ua); free(ub);
    return 0;
}
