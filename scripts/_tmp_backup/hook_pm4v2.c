#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

typedef int (*ioctl_fn)(int, unsigned long, ...);

static int queue_count = 0;
static volatile uint64_t *doorbell_ptr = NULL;  // Pointer to doorbell for PM4 queue
static int doorbell_offset = -1;
static int kfd_fd = -1;
static uint64_t doorbell_mmap_off = 0;

int ioctl(int fd, unsigned long request, ...) {
    va_list ap;
    va_start(ap, request);
    void *arg = va_arg(ap, void*);
    va_end(ap);

    ioctl_fn real_ioctl = (ioctl_fn)dlsym(RTLD_NEXT, "ioctl");

    unsigned int type = (request >> 8) & 0xFF;
    unsigned int nr = request & 0xFF;

    if (type == 0x4b) {
        // Track KFD fd
        if (kfd_fd < 0) kfd_fd = fd;

        // Capture doorbell alloc (ALLOC with MMIO_REMAP flag 0x10)
        if (nr == 0x16) {
            int ret = real_ioctl(fd, request, arg);
            int se = errno;
            uint32_t *q32 = (uint32_t*)arg;
            uint64_t *q64 = (uint64_t*)arg;
            uint32_t flags = q32[9];
            if ((flags & 0x10) && doorbell_mmap_off == 0) {
                doorbell_mmap_off = q64[3];  // mmap_offset
                fprintf(stderr, "[HOOK] Doorbell alloc: mmap_off=0x%lx va=0x%lx flags=0x%x\n",
                        doorbell_mmap_off, q64[0], flags);
            }
            errno = se;
            return ret;
        }

        // Intercept CREATE_QUEUE
        if (nr == 0x02) {
            uint32_t *q32 = (uint32_t*)arg;
            uint64_t *q64 = (uint64_t*)arg;
            queue_count++;

            if (queue_count == 2) {
                uint32_t orig_type = q32[10];
                q32[10] = 0;  // PM4
                fprintf(stderr, "[HOOK] CQ #%d: PATCHED type %u -> 0 (PM4)\n", queue_count, orig_type);
            }

            int ret = real_ioctl(fd, request, arg);
            int se = errno;

            if (ret == 0 && queue_count == 2) {
                // Capture doorbell offset for PM4 queue
                uint64_t db_val = q64[3];
                doorbell_offset = (int)(db_val & 0xFFFF);  // Low bits = byte offset in doorbell page
                fprintf(stderr, "[HOOK] CQ #%d: PM4 OK! qid=%d doorbell_raw=0x%lx offset=%d\n",
                        queue_count, (int32_t)q32[13], db_val, doorbell_offset);

                // Write doorbell info to shared file so Python can find it
                FILE *f = fopen("/tmp/pm4_doorbell_info", "w");
                if (f) {
                    fprintf(f, "%d %lu %d\n", kfd_fd, doorbell_mmap_off, doorbell_offset);
                    fclose(f);
                }
            } else if (ret != 0 && queue_count == 2) {
                fprintf(stderr, "[HOOK] CQ #%d: PM4 FAILED errno=%d, retrying AQL\n", queue_count, se);
                q32[10] = 2;
                ret = real_ioctl(fd, request, arg);
                se = errno;
            }

            errno = se;
            return ret;
        }

        // Intercept DESTROY_QUEUE for PM4 queue — skip it to avoid GPU reset
        if (nr == 0x03) {
            uint32_t *q32 = (uint32_t*)arg;
            int qid = (int32_t)q32[0];
            if (qid >= 1) {  // PM4 queue is typically id >= 1
                fprintf(stderr, "[HOOK] SKIP destroy queue %d (PM4)\n", qid);
                return 0;  // Pretend success
            }
        }
    }

    return real_ioctl(fd, request, arg);
}
