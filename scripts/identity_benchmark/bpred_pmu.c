/* bpred_pmu.c — branch-predictor mispredict-count RESERVOIR (clean integer observable).
 *
 * The Zen5 conditional predictor is a TAGE-like FSM: mispredict count over a block of
 * data-dependent branches is a NON-ADDITIVE function of recent direction history (a run of
 * identical bits -> ~0 penalty; a FLIP after a run -> a mispredict). Prior timing probes failed
 * because the ~60-cycle rdtscp fence floor buried the ~15-cycle mispredict. Here we read the
 * CLEAN integer mispredict count from the PMU (PERF_COUNT_HW_BRANCH_MISSES), removing that noise.
 *
 * Per timestep t we replay the last-K-bit window of the drive as NREP*K data-dependent branches
 * (predictor state carries across steps, NOT reset) and emit the mispredict-count delta for the
 * block. If the die's predictor does genuine nonlinear-temporal compute, a LINEAR readout of these
 * counts can do delayed-XOR/parity of the drive that a linear model of the drive cannot.
 *
 * Build: gcc -O2 -fno-if-conversion -fno-if-conversion2 -fno-tree-loop-if-convert -o bpred_pmu bpred_pmu.c
 * Run:   ./bpred_pmu <u_file> <out_file> [core]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>

static int perf_open(void) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(pe);
    pe.config = PERF_COUNT_HW_BRANCH_MISSES;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    int fd = syscall(__NR_perf_event_open, &pe, 0 /*self*/, -1 /*any cpu*/, -1, 0);
    return fd;
}

#define K   24
#define NREP 8

/* two noinline sinks so each branch direction has a distinct, un-cmov-able body */
static long __attribute__((noinline)) up(long s, long x)   { return s + (x ^ 0x9e3779b9); }
static long __attribute__((noinline)) down(long s, long x) { return s - (x ^ 0x7f4a7c15); }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s u_file out_file [core]\n", argv[0]); return 1; }
    int core = (argc > 3) ? atoi(argv[3]) : 4;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) fprintf(stderr, "warn: affinity failed\n");

    int fd = perf_open();
    if (fd < 0) { fprintf(stderr, "perf_event_open failed (need perf_event_paranoid<=2)\n"); return 2; }

    FILE *uf = fopen(argv[1], "r");
    if (!uf) { fprintf(stderr, "no u_file\n"); return 1; }
    int *u = malloc(sizeof(int) * (1 << 22));
    long L = 0; int v;
    while (fscanf(uf, "%d", &v) == 1) u[L++] = v;
    fclose(uf);

    FILE *of = fopen(argv[2], "w");
    volatile long sink = 0;
    long tab[K];
    for (int j = 0; j < K; j++) tab[j] = (long)(j * 2654435761u);

    /* warm the predictor on the first window so step 0 isn't a cold-start outlier */
    for (long t = 0; t < L; t++) {
        long lo = t - K + 1; if (lo < 0) lo = 0;
        long cnt;
        ioctl(fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        long s = sink;
        for (int r = 0; r < NREP; r++) {
            for (long i = lo; i <= t; i++) {
                int bit = u[i];
                int j = (int)(i - lo);
                if (bit) s = up(s, tab[j]);     /* real data-dependent branch (if-conv disabled) */
                else     s = down(s, tab[j]);
            }
        }
        sink = s;
        ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
        if (read(fd, &cnt, sizeof(cnt)) != sizeof(cnt)) cnt = -1;
        fprintf(of, "%ld\n", cnt);
    }
    fclose(of);
    close(fd);
    if (sink == 0x123456) fprintf(stderr, "%ld\n", sink);  /* defeat DCE */
    free(u);
    return 0;
}
