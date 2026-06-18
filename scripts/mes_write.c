#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
MODULE_LICENSE("GPL");
typedef u32 (*rreg_fn)(void*, u32);
typedef void (*wreg_fn)(void*, u32, u32);
static int __init mw_init(void) {
    struct pci_dev *p = pci_get_device(0x1002,0x1586,NULL);
    void *a; rreg_fn rr; wreg_fn wr; u32 gc=0x2800;
    if (!p) return -ENODEV;
    a=(u8*)pci_get_drvdata(p)-0x10;
    rr=(rreg_fn)0xFFFFFFFFC0E28460ULL;
    wr=(wreg_fn)0xFFFFFFFFC0E28820ULL;

    pr_info("mw: === MES REGISTER WRITE TEST ===\n");

    /* Read current MES state with GRBM ME=3 */
    wr(a, 0x1900, (3<<2)|0); /* GRBM: ME=3, PIPE=0 */
    udelay(200);

    pr_info("mw: MES_CNTL (0x2807) = 0x%08X\n", rr(a, gc+0x2807));
    pr_info("mw: PRGRM_CNTR_START (0x2800) = 0x%08X\n", rr(a, gc+0x2800));
    pr_info("mw: MES_INSTR_PNTR (0x2813) = 0x%08X\n", rr(a, gc+0x2813));
    pr_info("mw: MES_IC_OP_CNTL (0x2820) = 0x%08X\n", rr(a, gc+0x2820));

    /* Test write to MES_CNTL */
    {
        u32 orig = rr(a, gc+0x2807);
        wr(a, gc+0x2807, 0x00000001); /* try writing bit 0 */
        udelay(200);
        u32 after = rr(a, gc+0x2807);
        pr_info("mw: MES_CNTL write test: orig=0x%X wrote=0x1 read=0x%X %s\n",
            orig, after, after != orig ? "*** WRITABLE! ***" : "locked");
        wr(a, gc+0x2807, orig); /* restore */
    }

    /* Test write to PRGRM_CNTR_START */
    {
        u32 orig = rr(a, gc+0x2800);
        wr(a, gc+0x2800, 0x1234);
        udelay(200);
        u32 after = rr(a, gc+0x2800);
        pr_info("mw: PRGRM_CNTR_START write test: orig=0x%X wrote=0x1234 read=0x%X %s\n",
            orig, after, after == 0x1234 ? "*** WRITABLE! ***" :
            (after != orig ? "PARTIAL" : "locked"));
        wr(a, gc+0x2800, orig);
    }

    /* Test write to IC_OP_CNTL */
    {
        u32 orig = rr(a, gc+0x2820);
        wr(a, gc+0x2820, orig | 1); /* invalidate cache */
        udelay(200);
        u32 after = rr(a, gc+0x2820);
        pr_info("mw: IC_OP_CNTL write test: orig=0x%X after=0x%X\n", orig, after);
    }

    /* Also check: what are the MES scratch/status registers? */
    pr_info("mw: --- MES GP registers ---\n");
    { int i;
      for (i = 0; i < 8; i++) {
        u32 v = rr(a, gc + 0x2860 + i);
        pr_info("mw: GP%d (0x%X) = 0x%08X\n", i, 0x2860+i, v);
      }
    }

    /* Try MES pipe 1 (KIQ) */
    wr(a, 0x1900, (3<<2)|1); /* GRBM: ME=3, PIPE=1 */
    udelay(200);
    pr_info("mw: --- MES PIPE 1 (KIQ) ---\n");
    pr_info("mw: INSTR_PNTR = 0x%08X\n", rr(a, gc+0x2813));
    pr_info("mw: MES_CNTL = 0x%08X\n", rr(a, gc+0x2807));
    { int i;
      for (i = 0; i < 8; i++) {
        u32 v = rr(a, gc + 0x2860 + i);
        if (v) pr_info("mw: GP%d = 0x%08X\n", i, v);
      }
    }

    /* Reset GRBM */
    wr(a, 0x1900, 0);

    pci_dev_put(p);
    return 0;
}
static void __exit mw_exit(void) {}
module_init(mw_init); module_exit(mw_exit);
