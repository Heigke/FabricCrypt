/* H7 memory-contention stressor for DESTRUCTIVE interference (cache/bandwidth throttling).
 * Streams a working set sized so ONE instance fits in shared L3 but TWO together spill to DRAM ->
 * mutual cache-line eviction -> combined throughput drops BELOW single (sum inverts) -> the
 * micro-contention becomes linearly-separable XOR. Control/counters via mmap shm (like micro_div).
 * shm(64B): int32 flag[2]@0 (0 idle,1 run,2 stop); uint64 cnt[2]@8.
 * args: <cpu> <idx 0|1> <shmpath> <array_bytes>
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
    size_t bytes = (size_t)atoll(argv[4]);
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set); sched_setaffinity(0, sizeof(set), &set);
    int fd = open(path, O_RDWR); if (fd < 0) { perror("open"); return 1; }
    void *m = mmap(NULL, 64, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    volatile int32_t *flag = (volatile int32_t *)m;
    volatile uint64_t *cnt = (volatile uint64_t *)((char *)m + 8);
    size_t n = bytes / sizeof(double);
    double *a = (double *)malloc(bytes);
    for (size_t i = 0; i < n; i++) a[i] = (double)(i & 1023) * 0.5 + 1.0;
    const size_t STRIDE = 8;                 /* 64B cache-line stride: one access per line */
    struct timespec ts = {0, 300000};
    volatile double sink = 0;
    while (1) {
        int f = flag[idx];
        if (f == 2) break;
        if (f == 1) {
            double s = 0;
            for (size_t i = 0; i < n; i += STRIDE) { s += a[i]; a[i] = s * 0.5 + 1.0; } /* read+write */
            cnt[idx]++; sink += s;
        } else {
            nanosleep(&ts, NULL);
        }
    }
    if (sink == 4242.4242) printf("%f\n", sink);
    free(a);
    return 0;
}
