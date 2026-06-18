#include <linux/module.h>
#include <linux/pci.h>
MODULE_LICENSE("GPL");
static int __init td_init(void) {
    struct pci_dev *p = pci_get_device(0x1002,0x1586,NULL);
    void *a, *psp; u64 kptr;
    int off; u32 code[4];
    if (!p) return -ENODEV;
    a=(u8*)pci_get_drvdata(p)-0x10;
    psp=(u8*)a+0x3B910;
    kptr = *(u64*)((u8*)psp + 0x010); /* TMR ioremap kptr */

    pr_info("td: TMR ioremap kptr = 0x%llX\n", kptr);
    if (!kptr || (kptr >> 48) != 0xFFFF) { pci_dev_put(p); return -ENODEV; }

    /* Dump first 512 bytes — descriptor table */
    pr_info("td: === TMR DESCRIPTOR TABLE ===\n");
    for (off = 0; off < 512; off += 16) {
        u32 v[4]; int i;
        for (i = 0; i < 4; i++)
            copy_from_kernel_nofault(&v[i], (u8*)(unsigned long)(kptr + off + i*4), 4);
        if (v[0] || v[1] || v[2] || v[3])
            pr_info("td: [%03X]: %08X %08X %08X %08X\n", off, v[0], v[1], v[2], v[3]);
    }

    /* Dump at 0x1000 — might have more descriptors */
    pr_info("td: === TMR+0x1000 ===\n");
    for (off = 0x1000; off < 0x1100; off += 16) {
        u32 v[4]; int i;
        for (i = 0; i < 4; i++)
            copy_from_kernel_nofault(&v[i], (u8*)(unsigned long)(kptr + off + i*4), 4);
        if (v[0] || v[1] || v[2] || v[3])
            pr_info("td: [%04X]: %08X %08X %08X %08X\n", off, v[0], v[1], v[2], v[3]);
    }

    /* Dump at 0x2000 — data section start */
    pr_info("td: === TMR+0x2000 ===\n");
    for (off = 0x2000; off < 0x2100; off += 16) {
        u32 v[4]; int i;
        for (i = 0; i < 4; i++)
            copy_from_kernel_nofault(&v[i], (u8*)(unsigned long)(kptr + off + i*4), 4);
        if (v[0] || v[1] || v[2] || v[3])
            pr_info("td: [%04X]: %08X %08X %08X %08X\n", off, v[0], v[1], v[2], v[3]);
    }

    /* Search for firmware code pattern (0x04070663 = MEC PC=0) */
    pr_info("td: === Searching for firmware in TMR ioremap (8.4MB) ===\n");
    {
        u64 scan;
        int found = 0;
        for (scan = 0; scan < 0x85D000 - 16 && found < 5; scan += 4) {
            u32 v;
            copy_from_kernel_nofault(&v, (u8*)(unsigned long)(kptr + scan), 4);
            if (v == 0x04070663) {
                u32 v1;
                copy_from_kernel_nofault(&v1, (u8*)(unsigned long)(kptr + scan + 4), 4);
                pr_info("td: *** MEC FW pattern at TMR+0x%llX: %08X %08X ***\n",
                    scan, v, v1);
                found++;
            }
            /* Also search for MES firmware header */
            if (v == 0x0009D1D0 || v == 0x0009D0D0) {
                pr_info("td: MES FW header at TMR+0x%llX: 0x%08X\n", scan, v);
                found++;
            }
        }
        if (!found) pr_info("td: No firmware patterns found in TMR ioremap\n");
    }

    /* Check if TMR ioremap is still writable */
    {
        u32 canary = 0xCAFE1337, orig, rb;
        copy_from_kernel_nofault(&orig, (u8*)(unsigned long)(kptr + 0x1004), 4);
        copy_to_kernel_nofault((u8*)(unsigned long)(kptr + 0x1004), &canary, 4);
        copy_from_kernel_nofault(&rb, (u8*)(unsigned long)(kptr + 0x1004), 4);
        pr_info("td: Canary test: wrote 0x%X read 0x%X %s\n",
            canary, rb, rb == canary ? "WRITABLE" : "LOCKED");
        copy_to_kernel_nofault((u8*)(unsigned long)(kptr + 0x1004), &orig, 4);
    }

    pci_dev_put(p);
    return 0;
}
static void __exit td_exit(void) {}
module_init(td_init); module_exit(td_exit);
