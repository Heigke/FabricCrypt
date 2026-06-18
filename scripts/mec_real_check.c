#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
MODULE_LICENSE("GPL");
typedef u32 (*rreg_fn)(void*, u32);
typedef void (*wreg_fn)(void*, u32, u32);
static int __init mrc_init(void) {
    struct pci_dev *p = pci_get_device(0x1002,0x1586,NULL);
    void *a; rreg_fn rr; wreg_fn wr;
    u32 gc=0x2800; /* seg1 for CP registers */
    int me, pipe;
    if (!p) return -ENODEV;
    a=(u8*)pci_get_drvdata(p)-0x10;
    rr=(rreg_fn)0xFFFFFFFFC0E28460ULL;
    wr=(wreg_fn)0xFFFFFFFFC0E28820ULL;

    pr_info("mrc: === CORRECT GRBM MEC CHECK ===\n");
    pr_info("mrc: GRBM_GFX_CNTL at 0x1900 (seg0)\n");
    pr_info("mrc: CP registers at gc_base=0x%X (seg1)\n", gc);

    /* Default state (no GRBM select) */
    pr_info("mrc: Default: PC=0x%04X IC=0x%08X_%08X CNTL=0x%X RS64=0x%X\n",
        rr(a,gc+0x2908), rr(a,gc+0x2813), rr(a,gc+0x2812),
        rr(a,gc+0x2808), rr(a,gc+0x2904));

    /* Scan ME=0,1,2,3 with all pipes */
    for (me = 0; me <= 3; me++) {
        for (pipe = 0; pipe < 4; pipe++) {
            u32 grbm = (me << 2) | pipe;
            wr(a, 0x1900, grbm); /* CORRECT GRBM offset! */
            udelay(100);
            { u32 pc = rr(a, gc+0x2908);
              u32 iclo = rr(a, gc+0x2812);
              u32 ichi = rr(a, gc+0x2813);
              u32 cntl = rr(a, gc+0x2808);
              u32 rs64 = rr(a, gc+0x2904);
              u32 pcs = rr(a, gc+0x2900);
              if (pc || iclo || ichi || cntl || rs64)
                pr_info("mrc: ME=%d P=%d: PC=0x%04X IC=0x%08X_%08X CNTL=0x%X RS64=0x%X PCS=0x%X\n",
                    me, pipe, pc, ichi, iclo, cntl, rs64, pcs);
            }
        }
    }
    /* Reset GRBM */
    wr(a, 0x1900, 0);

    pr_info("mrc: Done.\n");
    pci_dev_put(p);
    return 0;
}
static void __exit mrc_exit(void) {}
module_init(mrc_init); module_exit(mrc_exit);
