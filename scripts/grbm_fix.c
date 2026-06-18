#include <linux/module.h>
#include <linux/pci.h>
MODULE_LICENSE("GPL");
#include <linux/delay.h>
typedef u32 (*rreg_fn)(void*, u32);
typedef void (*wreg_fn)(void*, u32, u32);
static int __init gf_init(void) {
    struct pci_dev *p = pci_get_device(0x1002,0x1586,NULL);
    void *a; rreg_fn rr; wreg_fn wr;
    u32 gc1, gc0; int pipe;
    if (!p) return -ENODEV;
    a=(u8*)pci_get_drvdata(p)-0x10;
    rr=(rreg_fn)0xFFFFFFFFC0E28460ULL;
    wr=(wreg_fn)0xFFFFFFFFC0E28820ULL;

    /* Find segment 0 and segment 1 bases */
    gc1 = *(u32*)((u8*)a + 0x768); /* seg1=0x2800 confirmed */
    gc0 = *(u32*)((u8*)a + 0x764); /* seg0 should be nearby */
    pr_info("gf: seg0=0x%X seg1=0x%X\n", gc0, gc1);

    /* regGRBM_GFX_CNTL = 0x1900, BASE_IDX=0 → use seg0 */
    pr_info("gf: GRBM at offset: seg0+0x1900 = 0x%X\n", gc0 + 0x1900);
    pr_info("gf: GRBM current = 0x%08X\n", rr(a, gc0 + 0x1900));

    /* Try MEC with correct GRBM */
    pr_info("gf: === MEC state with correct GRBM ===\n");
    for (pipe = 0; pipe < 4; pipe++) {
        u32 grbm = (1 << 2) | pipe; /* ME=1, PIPE=pipe */
        wr(a, gc0 + 0x1900, grbm);
        udelay(100);
        { u32 pc = rr(a, gc1 + 0x2908);
          u32 iclo = rr(a, gc1 + 0x2812);
          u32 ichi = rr(a, gc1 + 0x2813);
          u32 cntl = rr(a, gc1 + 0x2808);
          u32 rs64 = rr(a, gc1 + 0x2904);
          pr_info("gf: pipe%d: PC=0x%04X IC=0x%08X_%08X CNTL=0x%X RS64=0x%X\n",
              pipe, pc, ichi, iclo, cntl, rs64);
        }
    }
    /* Reset GRBM */
    wr(a, gc0 + 0x1900, 0);

    /* Also check MES (ME=3) */
    pr_info("gf: === MES state ===\n");
    for (pipe = 0; pipe < 2; pipe++) {
        u32 grbm = (3 << 2) | pipe; /* ME=3, PIPE=pipe */
        wr(a, gc0 + 0x1900, grbm);
        udelay(100);
        { u32 pc = rr(a, gc1 + 0x2908);
          u32 iclo = rr(a, gc1 + 0x2812);
          u32 ichi = rr(a, gc1 + 0x2813);
          pr_info("gf: MES pipe%d: PC=0x%04X IC=0x%08X_%08X\n",
              pipe, pc, ichi, iclo);
        }
    }
    wr(a, gc0 + 0x1900, 0);

    /* Also check GFX (ME=0) */
    pr_info("gf: === GFX (ME/PFP) state ===\n");
    { wr(a, gc0 + 0x1900, 0); /* ME=0, PIPE=0 */
      udelay(100);
      pr_info("gf: GFX: PC=0x%04X IC=0x%08X_%08X\n",
          rr(a, gc1+0x2908), rr(a, gc1+0x2813), rr(a, gc1+0x2812));
    }

    pci_dev_put(p);
    return 0;
}
static void __exit gf_exit(void) {}
module_init(gf_init); module_exit(gf_exit);
