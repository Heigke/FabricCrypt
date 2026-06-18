/*
 * sdma_copy_tmr.c — Use amdgpu's SDMA copy to read TMR via GPU DMA
 *
 * amdgpu_copy_buffer(adev_ring, src_offset, dst_offset, num_bytes,
 *                    resv, fence, direct_submit, vm_needs_flush, tmz)
 *
 * SDMA is a GPU-internal DMA engine. It accesses VRAM through the
 * GPU's memory controller, NOT through CPU's Data Fabric.
 * If SDMA can read TMR, it's the DF bypass we need.
 */
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
MODULE_LICENSE("GPL");

/* amdgpu_copy_buffer signature (simplified):
 * int amdgpu_copy_buffer(struct amdgpu_ring *ring,
 *                        uint64_t src_offset, uint64_t dst_offset,
 *                        uint32_t byte_count,
 *                        struct dma_resv *resv,
 *                        struct dma_fence **fence,
 *                        bool direct_submit, bool vm_needs_flush, bool tmz)
 */
typedef int (*copy_buf_fn)(void *ring, u64 src, u64 dst, u32 count,
                           void *resv, void **fence,
                           int direct, int flush, int tmz);

static int __init sct_init(void) {
    struct pci_dev *p = pci_get_device(0x1002,0x1586,NULL);
    void *a, *psp;
    copy_buf_fn copy_buf;
    void *sdma_ring;
    u64 tmr_mc, fw_pri_mc;
    void *fw_pri;
    int ret;
    void *fence = NULL;

    if (!p) return -ENODEV;
    a=(u8*)pci_get_drvdata(p)-0x10;
    psp=(u8*)a+0x3B910;
    fw_pri=(void*)*(u64*)((u8*)psp+0x058);
    fw_pri_mc=*(u64*)((u8*)psp+0x050);
    tmr_mc=0x97E0000000ULL;
    copy_buf=(copy_buf_fn)0xFFFFFFFFC0E4A570ULL;

    /* Find SDMA ring. From adev scan: adev+0xFDA0 has SDMA GPU addr.
     * The ring structure is typically at adev->sdma.instance[0].ring.
     * struct amdgpu_sdma_instance { struct amdgpu_ring ring; ... }
     * amdgpu_ring starts at the instance base.
     *
     * From scan: adev+0xFD98 has CPU ptr 0xFFFFD26E00ACD000
     * and adev+0xFDA0 has GPU 0x7FFF00401000
     * The ring struct likely starts before these, at adev+0xFC00 or similar.
     *
     * Actually, struct amdgpu_ring has ring_mem_handle at ~+0x10,
     * gpu_addr at ~+0x18, ring_size at ~+0x20, etc.
     * If gpu_addr=0x7FFF00401000 at adev+0xFDA0:
     *   ring_start = adev+0xFDA0 - offset_of(gpu_addr in amdgpu_ring)
     *
     * For simplicity, let me search for the ring by looking for
     * the ring funcs pointer (a known kptr pattern before the ring data) */

    /* Actually, let's just use the SDMA ring from the adev structure.
     * In amdgpu: adev->sdma.instance[inst].ring
     * The SDMA instance is at a known offset in adev. */

    /* From adev+0x10888: gpu=0x7FFF00900000 (this might be SDMA page ring)
     * From adev+0x10AE0: gpu=0x7FFF00A00000 (another SDMA ring)
     * These are at ~0x108xx which could be sdma.instance[0].ring */

    /* The ring is a large struct (~0x200 bytes). The GPU addr is at
     * offset 0x1E0 or similar in the ring struct.
     * If gpu_addr at adev+0x10888 → ring starts at adev+0x10888 - 0x1E0 = 0x106A8 */

    /* Let me just try passing different ring pointers to amdgpu_copy_buffer
     * and see which one works */
    pr_info("sct: === SDMA TMR COPY TEST ===\n");
    pr_info("sct: tmr_mc=0x%llX fw_pri_mc=0x%llX\n", tmr_mc, fw_pri_mc);

    /* Write marker to fw_pri to detect if SDMA writes to it */
    { u32 marker = 0xAAAAAAAA;
      int i;
      for (i = 0; i < 16; i++)
          copy_to_kernel_nofault((u8*)fw_pri + i*4, &marker, 4);
      pr_info("sct: fw_pri filled with 0xAAAAAAAA\n");
    }

    /* Try to find the SDMA ring. The adev struct has sdma.instance[0].ring
     * which is a large embedded struct. Let me scan for the ring
     * by looking for known ring->funcs pointer patterns. */
    {
        int off;
        /* Ring struct has ring->funcs at offset ~0x148 on kernel 6.14.
         * Look for kptr followed by specific patterns. */
        for (off = 0xF000; off < 0x12000; off += 8) {
            u64 val = *(u64*)((u8*)a + off);
            /* Ring type identifier: look for ring_type field
             * AMDGPU_RING_TYPE_SDMA = 2 */
            if (val == 2) {
                /* Check if this looks like a ring struct */
                u64 before = *(u64*)((u8*)a + off - 8);
                u64 gpu_nearby = 0;
                int j;
                for (j = 0; j < 0x200; j += 8) {
                    u64 v = *(u64*)((u8*)a + off + j);
                    if (v >= 0x7FFF00000000ULL && v < 0x7FFF01000000ULL) {
                        gpu_nearby = v;
                        break;
                    }
                }
                if (gpu_nearby) {
                    sdma_ring = (void*)((u8*)a + off - 8);
                    /* The ring type is typically at offset ~0x08 in the ring struct.
                     * So ring starts at off - 8 if type is at +0x08 */
                    pr_info("sct: Candidate SDMA ring at adev+0x%X (type=2 gpu=0x%llX)\n",
                        off - 8, gpu_nearby);

                    /* Try the copy! */
                    pr_info("sct: Attempting SDMA copy TMR→fw_pri...\n");
                    ret = copy_buf(sdma_ring, tmr_mc, fw_pri_mc, 256,
                                   NULL, &fence, 1, 0, 0);
                    pr_info("sct: copy_buffer ret=%d fence=%px\n", ret, fence);

                    if (ret == 0) {
                        /* Wait for completion */
                        mdelay(100);

                        /* Read fw_pri to see if SDMA wrote TMR data */
                        { u32 d[8]; int i;
                          for (i = 0; i < 8; i++)
                            copy_from_kernel_nofault(&d[i], (u8*)fw_pri+i*4, 4);
                          pr_info("sct: fw_pri after copy: %08X %08X %08X %08X %08X %08X %08X %08X\n",
                              d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7]);

                          if (d[0] != 0xAAAAAAAA) {
                            pr_info("sct: *** SDMA WROTE DATA! TMR CONTENT: ***\n");
                            if (d[0] != 0xFFFFFFFF)
                                pr_info("sct: *** TMR READABLE VIA SDMA! DF BYPASSED! ***\n");
                            else
                                pr_info("sct: SDMA also gets 0xFFFFFFFF (DF blocks GPU DMA too)\n");
                          } else {
                            pr_info("sct: fw_pri unchanged (SDMA copy may have failed silently)\n");
                          }
                        }
                        break;
                    }
                }
            }
        }
    }

    pci_dev_put(p);
    return 0;
}
static void __exit sct_exit(void) { pr_info("sct: unloaded\n"); }
module_init(sct_init); module_exit(sct_exit);
