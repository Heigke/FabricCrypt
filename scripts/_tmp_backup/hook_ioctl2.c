#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <sys/ioctl.h>

typedef int (*ioctl_fn)(int, unsigned long, ...);

static void dump_hex(const uint8_t *buf, int len) {
    for (int i = 0; i < len; i++) {
        fprintf(stderr, "%02x ", buf[i]);
        if ((i+1) % 16 == 0) fprintf(stderr, "\n");
    }
    if (len % 16) fprintf(stderr, "\n");
}

int ioctl(int fd, unsigned long request, ...) {
    va_list ap;
    va_start(ap, request);
    void *arg = va_arg(ap, void*);
    va_end(ap);

    ioctl_fn real_ioctl = (ioctl_fn)dlsym(RTLD_NEXT, "ioctl");

    // Only intercept KFD ioctls (type='K'=0x4b)
    unsigned int type = (request >> 8) & 0xFF;
    unsigned int nr = request & 0xFF;
    unsigned int dir = (request >> 30) & 0x3;
    unsigned int sz = (request >> 16) & 0x3FFF;

    if (type == 0x4b) {
        int ret = real_ioctl(fd, request, arg);
        int saved_errno = errno;

        fprintf(stderr, "[KFD] ioctl nr=0x%02x dir=%u sz=%u fd=%d => ret=%d errno=%d\n",
                nr, dir, sz, fd, ret, ret < 0 ? saved_errno : 0);

        if (nr == 0x02) { // CREATE_QUEUE
            uint8_t *buf = (uint8_t*)arg;
            uint64_t *q64 = (uint64_t*)buf;
            uint32_t *q32 = (uint32_t*)buf;
            fprintf(stderr, "  CQ: ring_base=0x%lx wptr=0x%lx rptr=0x%lx doorbell=0x%lx\n",
                    q64[0], q64[1], q64[2], q64[3]);
            fprintf(stderr, "  CQ: ring_size=%u gpu_id=0x%x queue_type=%u pct=%u prio=%u qid=%d\n",
                    q32[8], q32[9], q32[10], q32[11], q32[12], (int32_t)q32[13]);
            fprintf(stderr, "  CQ: eop=0x%lx eop_sz=%lu ctx=0x%lx ctx_sz=%u ctl=%u sdma=%u\n",
                    q64[7], q64[8], q64[9], q32[20], q32[21], q32[22]);
        }
        else if (nr == 0x15) { // ACQUIRE_VM
            uint32_t *q32 = (uint32_t*)arg;
            fprintf(stderr, "  ACQUIRE_VM: gpu_id=0x%x drm_fd=%d\n", q32[0], q32[1]);
        }
        else if (nr == 0x16) { // ALLOC_MEMORY
            uint64_t *q64 = (uint64_t*)arg;
            uint32_t *q32 = (uint32_t*)arg;
            fprintf(stderr, "  ALLOC: va=0x%lx sz=0x%lx handle=0x%lx mmap_off=0x%lx gpu_id=0x%x flags=0x%x\n",
                    q64[0], q64[1], q64[2], q64[3], q32[8], q32[9]);
        }
        else if (nr == 0x18) { // MAP_MEMORY
            uint64_t *q64 = (uint64_t*)arg;
            uint32_t *q32 = (uint32_t*)arg;
            fprintf(stderr, "  MAP: handle=0x%lx dev_ids_ptr=0x%lx n_devs=%u n_succ=%d\n",
                    q64[0], q64[1], q32[4], (int32_t)q32[5]);
        }
        else if (nr == 0x04) { // SET_MEMORY_POLICY
            dump_hex((uint8_t*)arg, sz > 64 ? 64 : sz);
        }
        else if (nr == 0x25) { // RUNTIME_ENABLE
            uint64_t *q64 = (uint64_t*)arg;
            uint32_t *q32 = (uint32_t*)arg;
            fprintf(stderr, "  RUNTIME_ENABLE: r_debug=0x%lx mode=%u caps=%u\n",
                    q64[0], q32[2], q32[3]);
        }
        else if (nr == 0x06 || nr == 0x14) { // GET_APERTURES
            // Just note it happened
        }
        else if (sz <= 64 && arg) {
            dump_hex((uint8_t*)arg, sz);
        }

        errno = saved_errno;
        return ret;
    }

    return real_ioctl(fd, request, arg);
}
