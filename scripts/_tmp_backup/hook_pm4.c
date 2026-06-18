#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <sys/ioctl.h>

typedef int (*ioctl_fn)(int, unsigned long, ...);

static int queue_count = 0;

int ioctl(int fd, unsigned long request, ...) {
    va_list ap;
    va_start(ap, request);
    void *arg = va_arg(ap, void*);
    va_end(ap);

    ioctl_fn real_ioctl = (ioctl_fn)dlsym(RTLD_NEXT, "ioctl");

    // Intercept CREATE_QUEUE (0xc0604b02)
    if (request == 0xc0604b02) {
        uint32_t *q32 = (uint32_t*)arg;
        uint64_t *q64 = (uint64_t*)arg;
        queue_count++;

        if (queue_count == 2) {
            uint32_t orig_type = q32[10];
            q32[10] = 0;  // Change queue_type from 2 (AQL) to 0 (PM4)
            fprintf(stderr, "[HOOK] CQ #%d: PATCHED type %u -> 0 (PM4)\n", queue_count, orig_type);
            fprintf(stderr, "[HOOK]   ring=0x%lx wptr=0x%lx rptr=0x%lx ring_sz=%u\n",
                    q64[0], q64[1], q64[2], q32[8]);

            // Save queue info for Python to read
            FILE *f = fopen("/tmp/pm4_queue_info", "w");
            if (f) {
                fprintf(f, "ring=0x%lx\n", q64[0]);
                fprintf(f, "wptr=0x%lx\n", q64[1]);
                fprintf(f, "rptr=0x%lx\n", q64[2]);
                fprintf(f, "ring_sz=%u\n", q32[8]);
                fclose(f);
            }
        } else {
            fprintf(stderr, "[HOOK] CQ #%d: pass-through type=%u\n", queue_count, q32[10]);
        }

        int ret = real_ioctl(fd, request, arg);
        int se = errno;

        if (ret == 0) {
            fprintf(stderr, "[HOOK] CQ #%d: OK! qid=%d doorbell=0x%lx\n",
                    queue_count, (int32_t)q32[13], q64[3]);
            // Append doorbell info
            if (queue_count == 2) {
                FILE *f = fopen("/tmp/pm4_queue_info", "a");
                if (f) {
                    fprintf(f, "qid=%d\n", (int32_t)q32[13]);
                    fprintf(f, "doorbell=0x%lx\n", q64[3]);
                    fclose(f);
                }
            }
        } else {
            fprintf(stderr, "[HOOK] CQ #%d: FAILED errno=%d\n", queue_count, se);
            if (queue_count == 2) {
                q32[10] = 2;
                fprintf(stderr, "[HOOK] CQ #%d: retrying with original type=2\n", queue_count);
                ret = real_ioctl(fd, request, arg);
                se = errno;
                if (ret == 0) {
                    fprintf(stderr, "[HOOK] CQ #%d: AQL fallback OK qid=%d\n",
                            queue_count, (int32_t)q32[13]);
                }
            }
        }
        errno = se;
        return ret;
    }

    return real_ioctl(fd, request, arg);
}
