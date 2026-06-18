/* H7 v2 region-encoded cache organ — fixes red-team hole #6 (the "00 idle-detector").
 * BOTH streamers ALWAYS run (never idle). The operand bit selects WHICH L3 region the streamer
 * hammers: flag 0 -> region A, flag 1 -> region B, flag 2 -> stop. So all four (a,b) cells are
 * "busy"; only SHARED-region destructive interference distinguishes them:
 *   same region (0,0)/(1,1) -> both hammer the same lines -> mutual eviction -> LOW throughput
 *   diff region (0,1)/(1,0) -> disjoint lines           -> less contention -> HIGH throughput
 * => throughput-sum is XOR-shaped with NO trivial idle cell. Each region sized to fit L3 alone but
 * two-in-same-region spills. Control/counters via mmap shm: int32 flag[2]@0, uint64 cnt[2]@8.
 * args: <cpu> <idx 0|1> <shmpath> <region_bytes>
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sched.h>
#include <time.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

int main(int argc, char **argv) {
    int cpu = atoi(argv[1]); int idx = atoi(argv[2]); const char *path = argv[3];
    size_t rbytes = (size_t)atoll(argv[4]);
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set); sched_setaffinity(0, sizeof(set), &set);
    int fd = open(path, O_RDWR); if (fd < 0) { perror("open"); return 1; }
    void *m = mmap(NULL, 64, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    volatile int32_t *flag = (volatile int32_t *)m;
    volatile uint64_t *cnt = (volatile uint64_t *)((char *)m + 8);
    /* two disjoint regions A and B, each rbytes; both streamers share the SAME buffer so that
     * "same region" means literally the same cache lines (true conflict). */
    size_t rn = rbytes / sizeof(double);
    double *buf = (double *)malloc(2 * rbytes);
    for (size_t i = 0; i < 2 * rn; i++) buf[i] = (double)(i & 1023) * 0.5 + 1.0;
    const size_t STRIDE = 8;
    struct timespec ts = {0, 300000};
    volatile double sink = 0;
    while (1) {
        int f = flag[idx];
        if (f == 2) break;
        if (f == 0 || f == 1) {
            double *a = buf + (f == 1 ? rn : 0);     /* region A or B */
            double s = 0;
            for (size_t i = 0; i < rn; i += STRIDE) { s += a[i]; a[i] = s * 0.5 + 1.0; }
            cnt[idx]++; sink += s;
        } else {
            nanosleep(&ts, NULL);
        }
    }
    if (sink == 4242.4242) printf("%f\n", sink);
    free(buf);
    return 0;
}
