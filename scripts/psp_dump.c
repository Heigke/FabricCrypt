#include <linux/module.h>
#include <linux/pci.h>
MODULE_LICENSE("GPL");
static int __init pd_init(void) {
    struct pci_dev *p = pci_get_device(0x1002, 0x1586, NULL);
    void *drm, *adev, *psp;
    int i; u64 adev_val;
    if (!p) return -ENODEV;
    drm = pci_get_drvdata(p);
    adev = (u8*)drm - 0x10;
    psp = (u8*)adev + 0x3B910;
    adev_val = (u64)(unsigned long)adev;
    pr_info("pd: PSP context at %px (adev+0x3B910)\n", psp);
    for (i = 0; i < 128; i++) {
        u64 v = *(u64*)((u8*)psp + i*8);
        if (v != 0) {
            const char *t = "";
            if (v == adev_val) t = " <-ADEV";
            else if ((v>>48)==0xFFFF) t = " <-KPTR";
            else if (v>=0x7FFF00000000ULL && v<0x800000000000ULL) t = " <-GART_MC";
            else if (v>=0x8000000000ULL && v<=0x98000000000ULL) t = " <-VRAM_MC";
            else if (v>=0xFFFFFFFFC0000000ULL) t = " <-FUNC";
            pr_info("pd: [%03X] %016llX%s\n", i*8, v, t);
        }
    }
    pci_dev_put(p);
    return 0;
}
static void __exit pd_exit(void) {}
module_init(pd_init); module_exit(pd_exit);
