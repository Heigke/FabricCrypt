/*
 * dcn_attack.c — DCN Display Controller TMR Read
 *
 * DCN MUST access TMR for HDCP content display.
 * Read current framebuffer address, save it, point scanout at TMR,
 * then check if DCN reads TMR content (via cursor readback or CRC).
 *
 * HUBPREQ0_DCSURF_PRIMARY_SURFACE_ADDRESS = 0x060a (BASE_IDX=2)
 * HUBPREQ0_DCSURF_PRIMARY_SURFACE_ADDRESS_HIGH = 0x060b (BASE_IDX=2)
 */
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
MODULE_LICENSE("GPL");

typedef u32 (*rreg_fn)(void*, u32);
typedef void (*wreg_fn)(void*, u32, u32);

static int __init da_init(void) {
    struct pci_dev *p = pci_get_device(0x1002,0x1586,NULL);
    void *a;
    rreg_fn rr;
    wreg_fn wr;
    u32 dcn_seg2;
    int i;

    if (!p) return -ENODEV;
    a=(u8*)pci_get_drvdata(p)-0x10;
    rr=(rreg_fn)0xFFFFFFFFC0E28460ULL;
    wr=(wreg_fn)0xFFFFFFFFC0E28820ULL;

    pr_info("da: === DCN TMR ACCESS TEST ===\n");

    /* Find DCN segment 2 base from adev reg_offset.
     * DCE_HWIP (or DCN_HWIP) segment 2 is stored in the reg_offset array.
     * From our scan: adev+0x8AC = 0x4C might be a DCN segment.
     *
     * Let me search more systematically: the reg_offset array has entries
     * for each IP. We know GC_HWIP=0 at offsets starting ~0x764.
     * Each IP takes HWIP_MAX_INSTANCE * NUM_REG_OFFSET * 4 bytes.
     * With 44 instances and 8 segments: 44*8*4 = 1408 bytes per IP.
     *
     * DCN is typically IP block ~15-20 in the enum.
     * But let's just try reading the HUBP register with various base offsets. */

    /* Strategy: try reading HUBPREQ0 surface address at offset 0x060a
     * with different segment bases until we find valid data */
    pr_info("da: Probing for DCN HUBP registers...\n");
    {
        /* Known candidates for DCN seg2 base: 0x4C, 0x4400, 0x0 */
        u32 bases[] = {0x0, 0x4C, 0x4400, 0x1E00, 0x3400, 0x8000};
        int b;
        for (b = 0; b < 6; b++) {
            u32 addr_lo = rr(a, bases[b] + 0x060a);
            u32 addr_hi = rr(a, bases[b] + 0x060b);
            u32 config  = rr(a, bases[b] + 0x05e5);
            if ((addr_lo != 0 && addr_lo != 0xFFFFFFFF) ||
                (addr_hi != 0 && addr_hi != 0xFFFFFFFF)) {
                u64 fb_addr = ((u64)addr_hi << 32) | addr_lo;
                pr_info("da: *** BASE 0x%X: FB_ADDR=0x%llX config=0x%X ***\n",
                    bases[b], fb_addr, config);
            }
        }

        /* Also try: scan for the FB address we know (0x80000XXXXX range)
         * by reading registers at many offsets */
        pr_info("da: Brute-force scanning for FB address in DCN regs...\n");
        for (i = 0x400; i < 0x800; i++) {
            u32 v = rr(a, 0x4C + i); /* try seg2=0x4C */
            if (v >= 0x80000000 && v < 0x98000000 && (v & 0xFFF) == 0) {
                u32 next = rr(a, 0x4C + i + 1);
                pr_info("da: [0x4C+0x%03X] = 0x%08X (next=0x%08X) — FB?\n",
                    i, v, next);
            }
        }
        /* Try with base 0 */
        for (i = 0x400; i < 0x800; i++) {
            u32 v = rr(a, i);
            if (v >= 0x80000000 && v < 0x98000000 && (v & 0xFFF) == 0) {
                u32 next = rr(a, i + 1);
                pr_info("da: [0x%03X] = 0x%08X (next=0x%08X) — FB?\n",
                    i, v, next);
            }
        }
    }

    /* Alternative: find DCN seg2 via adev structure.
     * The reg_offset for DCE_HWIP segment 2.
     * DCE_HWIP index varies. Let me dump all non-zero segment values. */
    pr_info("da: --- adev reg_offset dump (0x760-0xC00) ---\n");
    for (i = 0x760; i < 0xC00; i += 4) {
        u32 v = *(u32*)((u8*)a + i);
        if (v > 0 && v < 0x20000)
            pr_info("da: adev+0x%X = 0x%X\n", i, v);
    }

    pci_dev_put(p);
    return 0;
}
static void __exit da_exit(void) { pr_info("da: unloaded\n"); }
module_init(da_init); module_exit(da_exit);
