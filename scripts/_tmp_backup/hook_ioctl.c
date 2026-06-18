#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/ioctl.h>

typedef int (*ioctl_fn)(int, unsigned long, ...);

int ioctl(int fd, unsigned long request, ...) {
    va_list ap;
    va_start(ap, request);
    void *arg = va_arg(ap, void*);
    va_end(ap);

    ioctl_fn real_ioctl = (ioctl_fn)dlsym(RTLD_NEXT, "ioctl");

    // Check for CREATE_QUEUE: _IOWR('K', 0x02, 96) = 0xc0604b02
    if (request == 0xc0604b02) {
        uint8_t *buf = (uint8_t*)arg;
        fprintf(stderr, "=== CREATE_QUEUE BEFORE (96 bytes) ===\n");
        for (int i = 0; i < 96; i++) {
            fprintf(stderr, "%02x ", buf[i]);
            if ((i+1) % 16 == 0) fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n");

        // Decode fields
        uint64_t *q64 = (uint64_t*)buf;
        uint32_t *q32 = (uint32_t*)buf;
        fprintf(stderr, "ring_base=0x%lx wptr=0x%lx rptr=0x%lx doorbell=0x%lx\n",
                q64[0], q64[1], q64[2], q64[3]);
        fprintf(stderr, "ring_size=%u gpu_id=0x%x queue_type=%u pct=%u prio=%u\n",
                q32[8], q32[9], q32[10], q32[11], q32[12]);
        fprintf(stderr, "queue_id=%d eop=0x%lx eop_sz=%lu ctx=0x%lx\n",
                (int32_t)q32[13], q64[7], q64[8], q64[9]);
        fprintf(stderr, "ctx_sz=%u ctl_stack=%u sdma_eng=%u pad=%u\n",
                q32[20], q32[21], q32[22], q32[23]);

        int ret = real_ioctl(fd, request, arg);

        fprintf(stderr, "=== CREATE_QUEUE AFTER (ret=%d) ===\n", ret);
        if (ret == 0) {
            fprintf(stderr, "queue_id=%d doorbell=0x%lx\n", (int32_t)q32[13], q64[3]);
        }
        return ret;
    }

    return real_ioctl(fd, request, arg);
}
