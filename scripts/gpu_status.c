#include <linux/module.h>
#include <linux/pci.h>
MODULE_LICENSE("GPL");
typedef u32 (*rreg_fn)(void*, u32);
static int __init gs_init(void) {
    struct pci_dev *p = pci_get_device(0x1002,0x1586,NULL);
    void *a; rreg_fn rr; u32 gc=0x2800;
    if (!p) return -ENODEV;
    a=(u8*)pci_get_drvdata(p)-0x10;
    rr=(rreg_fn)0xFFFFFFFFC0E28460ULL;

    /* GRBM_STATUS registers (seg0 base=0) */
    pr_info("gs: GRBM_STATUS = 0x%08X\n", rr(a, 0x1740));
    pr_info("gs: GRBM_STATUS2 = 0x%08X\n", rr(a, 0x1741));
    pr_info("gs: GRBM_STATUS_SE0 = 0x%08X\n", rr(a, 0x1744));
    pr_info("gs: CP_BUSY = 0x%08X\n", rr(a, 0x160A));
    pr_info("gs: CP_STAT = 0x%08X\n", rr(a, 0x1B40));

    /* Check MES-specific registers (ME=3) */
    /* regCP_MES_INSTR_PNTR = ? — let me scan for non-zero in MES range */
    pr_info("gs: === Scanning CP range 0x2800-0x2A00 for non-zero ===\n");
    { int r;
      for (r = 0x2800; r < 0x2A00; r++) {
        u32 v = rr(a, gc + r);
        if (v != 0)
          pr_info("gs: [gc+0x%04X=0x%04X] = 0x%08X\n", r, gc+r, v);
      }
    }

    /* MES registers at their own range */
    pr_info("gs: === MES registers (seg1) ===\n");
    { int regs[] = {0x2850, 0x2851, 0x2852, 0x2853, 0x2854, 0x2855,
                    0x2856, 0x2857, 0x2858, 0x2859, 0x285A};
      int i;
      for (i = 0; i < 11; i++) {
        u32 v = rr(a, gc + regs[i]);
        if (v) pr_info("gs: [gc+0x%04X] = 0x%08X\n", regs[i], v);
      }
    }

    /* Check if CP_CPC (compute) is idle */
    pr_info("gs: CP_CPC_STATUS = 0x%08X\n", rr(a, gc + 0x2814));
    pr_info("gs: CP_CPC_BUSY_STAT = 0x%08X\n", rr(a, gc + 0x281C));

    pci_dev_put(p);
    return 0;
}
static void __exit gs_exit(void) {}
module_init(gs_init); module_exit(gs_exit);
