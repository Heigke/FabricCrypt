/* H7 micro divider-stressor: pin to a CPU, hammer the shared FP divider in a tight compiled loop.
 * Two instances on SMT siblings serialize on the single per-core divider -> strong sub-additive
 * throughput (the microarchitectural A*B nonlinearity). Control + counters via an mmap'd shm file.
 * shm layout (64B): int32 flag[2] @0 (0=idle,1=run,2=stop); uint64 cnt[2] @8.
 * args: <cpu> <idx 0|1> <shmpath>
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
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
    int fd = open(path, O_RDWR);
    if (fd < 0) { perror("open"); return 1; }
    void *m = mmap(NULL, 64, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    volatile int32_t *flag = (volatile int32_t *)m;
    volatile uint64_t *cnt = (volatile uint64_t *)((char *)m + 8);
    double x = 1.0103 + idx * 0.013;
    struct timespec ts = {0, 300000};
    volatile double sink = 0;
    while (1) {
        int f = flag[idx];
        if (f == 2) break;
        if (f == 1) {
            /* THROUGHPUT-bound: 8 independent FMA chains saturate the core's FP pipes (high ILP),
             * so an SMT sibling doing the same steals issue width -> strong sub-additive contention. */
            double a0=x,a1=x+0.1,a2=x+0.2,a3=x+0.3,a4=x+0.4,a5=x+0.5,a6=x+0.6,a7=x+0.7;
            for (int i = 0; i < 40000; i++) {
                a0=a0*1.0000001+0.1; a1=a1*1.0000002+0.1; a2=a2*1.0000003+0.1; a3=a3*1.0000004+0.1;
                a4=a4*1.0000005+0.1; a5=a5*1.0000006+0.1; a6=a6*1.0000007+0.1; a7=a7*1.0000008+0.1;
            }
            x = a0+a1+a2+a3+a4+a5+a6+a7;
            cnt[idx]++;
            sink += x;                               /* keep result live (no dead-code elim) */
        } else {
            nanosleep(&ts, NULL);
        }
    }
    if (sink == 4242.4242) printf("%f\n", sink);
    return 0;
}
