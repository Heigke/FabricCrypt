#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
MODULE_LICENSE("GPL");
typedef u32 (*rreg_fn)(void*, u32);
typedef void (*wreg_fn)(void*, u32, u32);
static int __init ms_init(void) {
    struct pci_dev *p = pci_get_device(0x1002,0x1586,NULL);
    void *a; rreg_fn rr; wreg_fn wr; u32 gc=0x2800;
    int pipe;
    if (!p) return -ENODEV;
    a=(u8*)pci_get_drvdata(p)-0x10;
    rr=(rreg_fn)0xFFFFFFFFC0E28460ULL;
    wr=(wreg_fn)0xFFFFFFFFC0E28820ULL;

    pr_info("ms: === MES STATE (correct registers) ===\n");

    /* MES IC_BASE at separate offset (NOT banked via GRBM) */
    pr_info("ms: MES IC_BASE_LO (gc+0x5850) = 0x%08X\n", rr(a, gc+0x5850));
    pr_info("ms: MES IC_BASE_HI (gc+0x5851) = 0x%08X\n", rr(a, gc+0x5851));
    pr_info("ms: MES IC_BASE_CNTL (gc+0x5852) = 0x%08X\n", rr(a, gc+0x5852));

    /* MES registers that need GRBM for pipe selection
     * MES is ME=3 on GFX12 */
    for (pipe = 0; pipe < 2; pipe++) {
        u32 grbm = (3 << 2) | pipe; /* ME=3, PIPE=pipe */
        wr(a, 0x1900, grbm); /* CORRECT GRBM offset */
        udelay(200);

        pr_info("ms: --- MES PIPE %d (GRBM=0x%X) ---\n", pipe, grbm);
        pr_info("ms: INSTR_PNTR (0x2813) = 0x%08X\n", rr(a, gc+0x2813));
        pr_info("ms: CNTL (0x2807) = 0x%08X\n", rr(a, gc+0x2807));
        pr_info("ms: PRGRM_CNTR_START (0x2800) = 0x%08X\n", rr(a, gc+0x2800));
        pr_info("ms: PRGRM_CNTR_START_HI (0x289D) = 0x%08X\n", rr(a, gc+0x289d));
        pr_info("ms: IC_OP_CNTL (0x2820) = 0x%08X\n", rr(a, gc+0x2820));
        pr_info("ms: DC_OP_CNTL (0x2837) = 0x%08X\n", rr(a, gc+0x2837));
        pr_info("ms: DC_BASE_CNTL (0x2836) = 0x%08X\n", rr(a, gc+0x2836));
        pr_info("ms: PERFCOUNT_CNTL (0x2899) = 0x%08X\n", rr(a, gc+0x2899));

        /* MES scratch/GP registers */
        { int i;
          for (i = 0; i < 8; i++) {
            u32 v = rr(a, gc + 0x2860 + i); /* GP registers? */
            if (v) pr_info("ms: GP[%d] (0x%X) = 0x%08X\n", i, 0x2860+i, v);
          }
        }
    }

    /* Reset GRBM */
    wr(a, 0x1900, 0);

    /* Also try reading MES IC_BASE with GRBM ME=3 */
    wr(a, 0x1900, (3 << 2)); /* ME=3, PIPE=0 */
    udelay(200);
    pr_info("ms: With GRBM ME=3: IC_BASE_LO(0x5850) = 0x%08X\n", rr(a, gc+0x5850));
    pr_info("ms: With GRBM ME=3: IC_BASE_HI(0x5851) = 0x%08X\n", rr(a, gc+0x5851));
    wr(a, 0x1900, 0);

    /* Try writing to MES IC_BASE to see if it's writable! */
    pr_info("ms: === MES IC_BASE WRITE TEST ===\n");
    { u32 orig_lo = rr(a, gc+0x5850);
      u32 orig_hi = rr(a, gc+0x5851);
      wr(a, gc+0x5850, 0xDEADBEEF);
      wr(a, gc+0x5851, 0x00000042);
      udelay(200);
      { u32 lo = rr(a, gc+0x5850);
        u32 hi = rr(a, gc+0x5851);
        pr_info("ms: After write: IC_BASE = 0x%08X_%08X %s\n",
            hi, lo,
            lo == 0xDEADBEEF ? "*** WRITABLE! ***" :
            (lo == orig_lo ? "HW-LOCKED" : "PARTIAL"));
        /* Restore */
        wr(a, gc+0x5850, orig_lo);
        wr(a, gc+0x5851, orig_hi);
      }
    }

    pci_dev_put(p);
    return 0;
}
static void __exit ms_exit(void) {}
module_init(ms_init); module_exit(ms_exit);
