/*
 * z2352_kmod_fw_inject.c — READ-ONLY register probe v3
 *
 * v3: READS ONLY — v2 crashed system proving corrected addresses hit real HW.
 *     regCP_CPC_IC_BASE_CNTL at 0xF84E is BEYOND 1MB BAR5 aperture (0x40000 dwords).
 *     That caused the crash — reading/writing unmapped MMIO.
 *
 * BAR5 = 1MB = 0x100000 bytes = 0x40000 DWORD offsets max.
 * GC_BASE[1] + CPC regs = 0xA000 + 0x584x = 0xF84x → byte offset 0x3E1xx
 *   → WITHIN 1MB, OK
 * But NOT all offsets may be valid. Must check carefully.
 *
 * BUILD: make -C /lib/modules/$(uname -r)/build M=$(pwd)/scripts modules
 * LOAD:  sudo insmod scripts/z2352_kmod_fw_inject.ko
 * CHECK: dmesg | grep "z2352"
 * UNLOAD: sudo rmmod z2352_kmod_fw_inject
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("FEEL Project");
MODULE_DESCRIPTION("READ-ONLY GC register probe v3");

#define AMD_VENDOR_ID  0x1002

/* GC IP block base offsets */
#define GC_BASE_0  0x1260   /* BASE_IDX=0 */
#define GC_BASE_1  0xA000   /* BASE_IDX=1 */

/* Max DWORD offset within 1MB BAR5 */
#define BAR5_MAX_DWORD  0x40000

static void __iomem *mmio_base;
static struct pci_dev *gpu_dev;
static resource_size_t bar5_len;

static u32 safe_read(const char *name, u32 dword_off)
{
    u32 val;
    if (dword_off >= BAR5_MAX_DWORD) {
        pr_warn("z2352: %s [0x%05X] SKIP — beyond BAR5 (max 0x%05X)\n",
                name, dword_off, BAR5_MAX_DWORD - 1);
        return 0xDEADBEEF;
    }
    val = readl(mmio_base + (u64)dword_off * 4);
    pr_info("z2352: %s [0x%05X] (byte 0x%06X) = 0x%08X\n",
            name, dword_off, dword_off * 4, val);
    return val;
}

static int __init fw_inject_init(void)
{
    struct pci_dev *pdev = NULL;
    int found = 0;

    pr_info("z2352: === Probe v3 READ-ONLY ===\n");

    while ((pdev = pci_get_device(AMD_VENDOR_ID, PCI_ANY_ID, pdev)) != NULL) {
        if ((pdev->class >> 8) == PCI_CLASS_DISPLAY_VGA ||
            (pdev->class >> 8) == PCI_CLASS_DISPLAY_OTHER) {
            pr_info("z2352: GPU %04x:%04x at %s\n",
                    pdev->vendor, pdev->device, pci_name(pdev));
            gpu_dev = pdev;
            found = 1;
            break;
        }
    }

    if (!found) {
        pr_err("z2352: No AMD GPU\n");
        return -ENODEV;
    }

    bar5_len = pci_resource_len(gpu_dev, 5);
    pr_info("z2352: BAR5 size = 0x%llX bytes (%llu KB)\n",
            (unsigned long long)bar5_len, (unsigned long long)bar5_len / 1024);

    mmio_base = pci_iomap(gpu_dev, 5, 0);
    if (!mmio_base) {
        pr_err("z2352: Failed to map BAR5\n");
        pci_dev_put(gpu_dev);
        return -ENOMEM;
    }

    pr_info("z2352: GC_BASE[0]=0x%04X, GC_BASE[1]=0x%04X\n", GC_BASE_0, GC_BASE_1);
    pr_info("z2352: --- BASE_IDX=0 registers ---\n");

    /* GRBM_STATUS: correct = GC_BASE_0 + 0x0DA4 = 0x2004 */
    safe_read("GRBM_STATUS_correct", GC_BASE_0 + 0x0DA4);
    safe_read("GRBM_STATUS2       ", GC_BASE_0 + 0x0DA2);

    pr_info("z2352: --- BASE_IDX=1 registers (small offsets first) ---\n");

    /* CP_MEC_CNTL: GC_BASE_1 + 0x0802 = 0xA802 */
    safe_read("CP_MEC_CNTL        ", GC_BASE_1 + 0x0802);

    /* CP_CPC_IC_OP_CNTL: GC_BASE_1 + 0x297A = 0xC97A */
    safe_read("CPC_IC_OP_CNTL     ", GC_BASE_1 + 0x297A);

    /* CP_CPC_STATUS: try at a few candidate offsets */
    safe_read("CP_CPC_STATUS?     ", GC_BASE_1 + 0x2180);

    pr_info("z2352: --- BASE_IDX=1 registers (large offsets) ---\n");

    /* These are 0xF8xx range — byte offset ~0x3E000, within 1MB */
    safe_read("MEC_UCODE_ADDR     ", GC_BASE_1 + 0x581A);
    safe_read("MEC_UCODE_DATA     ", GC_BASE_1 + 0x581B);
    safe_read("CPC_IC_BASE_LO     ", GC_BASE_1 + 0x584C);
    safe_read("CPC_IC_BASE_HI     ", GC_BASE_1 + 0x584D);
    safe_read("CPC_IC_BASE_CNTL   ", GC_BASE_1 + 0x584E);

    pr_info("z2352: --- OLD wrong offsets (no GC base) ---\n");
    safe_read("OLD_GRBM_STATUS    ", 0x2004);
    safe_read("OLD_CPC_IC_BASE_LO ", 0x584C);
    safe_read("OLD_CPC_IC_BASE_HI ", 0x584D);
    safe_read("OLD_CPC_IC_BASE_CNT", 0x584E);
    safe_read("OLD_CP_MEC_CNTL    ", 0x2188);

    pr_info("z2352: --- Scan for non-zero near GC_BASE_1 ---\n");
    /* Quick scan around 0xA000 to see if there's anything alive */
    {
        int i;
        int nz = 0;
        for (i = 0; i < 32; i++) {
            u32 off = GC_BASE_1 + i;
            u32 v = readl(mmio_base + (u64)off * 4);
            if (v != 0) {
                pr_info("z2352: scan[0x%05X] = 0x%08X\n", off, v);
                nz++;
            }
        }
        pr_info("z2352: %d/32 non-zero near GC_BASE_1\n", nz);
    }

    pr_info("z2352: === v3 complete — NO WRITES ===\n");
    return 0;
}

static void __exit fw_inject_exit(void)
{
    if (mmio_base)
        pci_iounmap(gpu_dev, mmio_base);
    if (gpu_dev)
        pci_dev_put(gpu_dev);
    pr_info("z2352: Module unloaded\n");
}

module_init(fw_inject_init);
module_exit(fw_inject_exit);
