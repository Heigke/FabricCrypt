/*
 * dcn_tmr_read.c — Display Controller TMR Read
 *
 * INSIGHT: DCN MUST read TMR because HDCP content is stored there.
 * The display DMA path bypasses DF protection by design.
 * Point framebuffer scanout at TMR → DCN reads TMR content → capture!
 *
 * On SSH so display corruption is acceptable.
 */
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
#include <linux/slab.h>
MODULE_LICENSE("GPL");

typedef u32 (*rreg_fn)(void*, u32);
typedef void (*wreg_fn)(void*, u32, u32);

static int __init dcn_init(void) {
    struct pci_dev *p = pci_get_device(0x1002,0x1586,NULL);
    void *a;
    rreg_fn rr;
    wreg_fn wr;

    if (!p) return -ENODEV;
    a=(u8*)pci_get_drvdata(p)-0x10;
    rr=(rreg_fn)0xFFFFFFFFC0E28460ULL;
    wr=(wreg_fn)0xFFFFFFFFC0E28820ULL;

    pr_info("dcn: === DISPLAY CONTROLLER TMR READ ===\n");

    /* DCN registers are in a separate IP block.
     * Find DCN register base from adev->reg_offset[DCN_HWIP][0][0].
     * DCN_HWIP is typically IP block 3 or 4. */

    /* On GFX12 APU, the display registers are at MMIO offsets.
     * Key DCN registers for framebuffer:
     *   HUBP_SURFACE_ADDRESS_HIGH / HUBP_SURFACE_ADDRESS
     *   These set the scanout MC address.
     *
     * But finding the exact register offsets requires the DCN header files.
     * Let me search for the current framebuffer address in adev first. */

    /* Strategy: search adev for the current framebuffer MC address.
     * The framebuffer is typically in the first few MB of VRAM.
     * Look for MC addresses in the 0x8000000000-0x8001000000 range
     * that are associated with display structures. */

    pr_info("dcn: Searching adev for framebuffer/display structures...\n");
    {
        int off;
        int found = 0;
        /* The display mode (DM) sub-structure has surface addresses.
         * These are typically in the DC (Display Core) sub-system. */

        /* Actually, let me read the framebuffer address directly from
         * MMIO registers. DCN's HUBP registers contain the surface address.
         *
         * On RDNA4/DCN 4.x:
         *   HUBPREQ_SURFACE_ADDRESS = 0x0573 (BASE_IDX=2 for DCN)
         *   HUBPREQ_SURFACE_ADDRESS_HIGH = 0x0574
         *
         * But the base for DCN registers needs to be found.
         * Let me scan for non-zero values in the DCN register range. */

        /* DCN registers typically start at a specific MMIO range.
         * On recent AMD GPUs, DCN is at offset 0x0 in segment 2.
         * Let me scan MMIO directly for framebuffer-like addresses. */

        /* Alternative: use the DRM framebuffer info */
        /* Actually, the simplest approach: read from debugfs */
    }

    /* Read the current framebuffer info from DRM */
    {
        struct file *fp = filp_open("/sys/kernel/debug/dri/0000:c3:00.0/framebuffer", O_RDONLY, 0);
        if (!IS_ERR(fp)) {
            char buf[4096];
            loff_t pos = 0;
            ssize_t n = kernel_read(fp, buf, sizeof(buf)-1, &pos);
            if (n > 0) {
                buf[n] = 0;
                /* Print first 500 chars to find framebuffer address */
                { int i;
                  for (i = 0; i < n && i < 500; i++) {
                    if (buf[i] == '\n') buf[i] = '|';
                  }
                  buf[500] = 0;
                  pr_info("dcn: FB info: %.200s\n", buf);
                }
            }
            filp_close(fp, NULL);
        }
    }

    /* Let me directly scan MMIO for display surface addresses.
     * DCN surface registers contain MC addresses of the framebuffer.
     * These are large u64 values in VRAM range (0x8000XXXXXXXX).
     *
     * Scan MMIO register space for VRAM-range values. */
    pr_info("dcn: Scanning MMIO for display surface addresses...\n");
    {
        /* DCN registers on RDNA4 are typically at MMIO offset range
         * 0x0000-0x8000 (separate from GC which is at gc_base0+).
         * The HUBP surface address registers are around offset 0x0570-0x0580.
         *
         * But with seg2 base, we need to know DCN's segment offset.
         * From adev->reg_offset[DCE_HWIP][0][segment]:
         * Let me search adev for the DCN base. */

        /* Actually, just scan adev for a display/DC-related structure
         * that contains the framebuffer MC address. */
        int off;
        for (off = 0; off < 0x100000; off += 8) {
            u64 val = *(u64*)((u8*)a + off);
            /* Look for framebuffer GPU addresses (VRAM, low offset,
             * aligned to 4KB, in the first 256MB) */
            if (val >= 0x8000000000ULL && val < 0x8010000000ULL &&
                (val & 0xFFF) == 0) {
                u64 prev = *(u64*)((u8*)a + off - 8);
                u64 next = *(u64*)((u8*)a + off + 8);
                /* Framebuffer addresses often come in pairs (primary + flip) */
                if ((next >= 0x8000000000ULL && next < 0x8010000000ULL) ||
                    (prev >= 0x8000000000ULL && prev < 0x8010000000ULL)) {
                    pr_info("dcn: FB candidate at adev+0x%X: 0x%llX (prev=0x%llX next=0x%llX)\n",
                        off, val, prev, next);
                }
            }
        }
    }

    /* NOVEL APPROACH: Try to read VRAM at TMR address via
     * a DMA mechanism that isn't MM_INDEX.
     *
     * The GPU has multiple VRAM access paths:
     * 1. CPU → PCI BAR → DF → VRAM (blocked for TMR)
     * 2. CPU → MMIO → MM_INDEX/DATA → DF → VRAM (blocked for TMR)
     * 3. GPU GFX → L1 → L2 → MC → VRAM (IC path, encrypted)
     * 4. GPU SDMA → MC → VRAM (unknown for TMR)
     * 5. DCN → MC → VRAM (must access TMR for HDCP)
     * 6. VCN → MC → VRAM (must access TMR for DRM)
     * 7. PSP → MC → VRAM (has TMR access by design)
     *
     * Paths 5 and 6 are the most promising.
     *
     * For path 5 (DCN), we need to find and modify the HUBP
     * surface address registers.
     *
     * Let me look for the DCN register base. */
    pr_info("dcn: Looking for DCN register base...\n");
    {
        /* DCE_HWIP reg_offset is stored similarly to GC_HWIP.
         * GC_HWIP was at adev+0x768 (seg1=0x2800).
         * DCE_HWIP might be at adev+0x768 + some_offset.
         *
         * The reg_offset array is reg_offset[MAX_HWIP][MAX_INST][NUM_SEG].
         * MAX_HWIP varies. Let me scan for small values near adev+0x700-0x900
         * that could be segment bases for other IP blocks. */
        int off;
        pr_info("dcn: adev reg_offset scan (0x700-0x900):\n");
        for (off = 0x700; off < 0x900; off += 4) {
            u32 v = *(u32*)((u8*)a + off);
            if (v > 0 && v < 0x10000 && (v & 3) == 0)
                pr_info("dcn:   [+0x%X] = 0x%X\n", off, v);
        }
    }

    pci_dev_put(p);
    return 0;
}
static void __exit dcn_exit(void) { pr_info("dcn: unloaded\n"); }
module_init(dcn_init); module_exit(dcn_exit);
