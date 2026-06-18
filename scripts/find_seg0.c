#include <linux/module.h>
#include <linux/pci.h>
MODULE_LICENSE("GPL");
typedef u32 (*rreg_fn)(void*, u32);
static int __init fs_init(void) {
    struct pci_dev *p = pci_get_device(0x1002,0x1586,NULL);
    void *a; rreg_fn rr; int off;
    if (!p) return -ENODEV;
    a=(u8*)pci_get_drvdata(p)-0x10;
    rr=(rreg_fn)0xFFFFFFFFC0E28460ULL;

    /* Find all u32=0x2800 in adev and check neighbors */
    pr_info("fs: Searching for 0x2800 (gc_base seg1)...\n");
    for (off = 0; off < 0x2000; off += 4) {
        u32 v = *(u32*)((u8*)a + off);
        if (v == 0x2800) {
            u32 prev = *(u32*)((u8*)a + off - 4);
            u32 next = *(u32*)((u8*)a + off + 4);
            pr_info("fs: adev+0x%X = 0x2800 (prev=0x%X next=0x%X)\n",
                off, prev, next);
            /* If prev or next is a small value (0-0x4000), it could be seg0 */
            if (prev < 0x4000)
                pr_info("fs:   prev might be seg0 = 0x%X\n", prev);
            if (next < 0x4000)
                pr_info("fs:   next might be seg2 = 0x%X\n", next);

            /* Try reading GRBM_GFX_CNTL (0x1900) with each as base */
            { u32 grbm;
              if (prev < 0x4000) {
                grbm = rr(a, prev + 0x1900);
                pr_info("fs:   GRBM via prev+0x1900 = 0x%08X\n", grbm);
              }
              grbm = rr(a, 0x1900); /* raw offset, no base */
              pr_info("fs:   GRBM via raw 0x1900 = 0x%08X\n", grbm);
            }
        }
    }

    /* Also try: what does GRBM at various offsets read? */
    pr_info("fs: === GRBM probe ===\n");
    { int addrs[] = {0x1900, 0x2800+0x1900, 0x1900*4, 0xDE00};
      int i;
      for (i = 0; i < 4; i++) {
        u32 v = rr(a, addrs[i]);
        pr_info("fs: rreg(0x%X) = 0x%08X\n", addrs[i], v);
      }
    }

    pci_dev_put(p);
    return 0;
}
static void __exit fs_exit(void) {}
module_init(fs_init); module_exit(fs_exit);
