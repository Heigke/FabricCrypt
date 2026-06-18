/*
 * smn_via_mmio.c — Access SMN via GPU MMIO NBIO registers
 * OR via PCI config space (NB function 0)
 *
 * SMN access method 1: PCI config 0:0.0 offset 0x60 (index) / 0x64 (data)
 * SMN access method 2: GPU NBIO registers (smn_index/smn_data in adev)
 */
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
MODULE_LICENSE("GPL");

static u32 smn_read_pci(struct pci_dev *nb, u32 addr) {
    u32 val;
    pci_write_config_dword(nb, 0x60, addr);
    pci_read_config_dword(nb, 0x64, &val);
    return val;
}

static void smn_write_pci(struct pci_dev *nb, u32 addr, u32 val) {
    pci_write_config_dword(nb, 0x60, addr);
    pci_write_config_dword(nb, 0x64, val);
}

static int __init svm_init(void) {
    struct pci_dev *nb;
    struct pci_dev *gpu;
    void *adev;
    int i;
    u32 val;

    /* Find the NB (Northbridge) PCI device for SMN access */
    nb = pci_get_device(0x1022, PCI_ANY_ID, NULL); /* AMD NB */
    while (nb && nb->devfn != 0) /* Function 0 */
        nb = pci_get_device(0x1022, PCI_ANY_ID, nb);

    if (!nb) {
        pr_info("svm: No AMD NB found, trying bus 0 dev 0 fn 0\n");
        nb = pci_get_domain_bus_and_slot(0, 0, PCI_DEVFN(0, 0));
    }
    if (!nb) { pr_info("svm: Cannot find NB\n"); return -ENODEV; }

    pr_info("svm: NB: %04X:%04X at %s\n", nb->vendor, nb->device, pci_name(nb));

    /* Verify SMN access works */
    val = smn_read_pci(nb, 0x00000000);
    pr_info("svm: SMN[0x0] = 0x%08X\n", val);

    /* Also find GPU for aper_base comparison */
    gpu = pci_get_device(0x1002, 0x1586, NULL);
    if (gpu) {
        adev = (u8*)pci_get_drvdata(gpu) - 0x10;
    }

    /* ===========================================
     * SCAN 1: Data Fabric protection registers
     * =========================================== */
    pr_info("svm: === DATA FABRIC REGISTERS ===\n");
    {
        /* DF instance 0 at SMN 0x18000000 */
        u32 df = 0x18000000;

        /* Core DF registers */
        for (i = 0; i < 0x400; i += 4) {
            val = smn_read_pci(nb, df + i);
            if (val != 0 && val != 0xFFFFFFFF) {
                /* Flag anything TMR-related */
                int tmr_flag = 0;
                /* TMR phys ≈ 0x2060000000, encoded as addr>>28 = 0x206 */
                if ((val & 0xFFF) == 0x206 || (val >> 20) == 0x206)
                    tmr_flag = 1;
                /* FB_BASE ≈ 0x8000000000, encoded as addr>>28 = 0x80000 */
                if (tmr_flag)
                    pr_info("svm: *** DF[0x%03X] = 0x%08X (TMR?) ***\n", i, val);
                else if (i < 0x50 || (i >= 0x100 && i < 0x200) || (i >= 0x240 && i < 0x300))
                    pr_info("svm: DF[0x%03X] = 0x%08X\n", i, val);
            }
        }

        /* DF instance 1-7 might have different protection configs */
        pr_info("svm: --- DF Instance scan ---\n");
        { u32 inst_bases[] = {0x18100000, 0x18200000, 0x18300000,
                              0x18400000, 0x18500000, 0x18600000};
          int b;
          for (b = 0; b < 6; b++) {
            val = smn_read_pci(nb, inst_bases[b]);
            if (val != 0 && val != 0xFFFFFFFF)
                pr_info("svm: DF_INST[%d][0] = 0x%08X\n", b+1, val);
          }
        }
    }

    /* ===========================================
     * SCAN 2: GPU MMHUB / GFXHUB SMN registers
     * =========================================== */
    pr_info("svm: === GPU MMHUB/GFXHUB SMN ===\n");
    {
        /* GPU internal registers via SMN
         * GC base: ~0x28C00000 (varies by SoC)
         * MMHUB: ~0x1A000000
         * GFXHUB: varies */
        u32 scan_bases[] = {
            0x1A000000, /* MMHUB */
            0x1A100000,
            0x28C00000, /* GC */
            0x3B10000,  /* PSP MP0 */
            0x3B10500,  /* PSP MP0 ext */
            0x3C00000,  /* PSP MP1 / SMU */
        };
        int b;
        for (b = 0; b < 6; b++) {
            int j;
            pr_info("svm: Scanning SMN 0x%08X...\n", scan_bases[b]);
            for (j = 0; j < 0x40; j += 4) {
                val = smn_read_pci(nb, scan_bases[b] + j);
                if (val != 0 && val != 0xFFFFFFFF)
                    pr_info("svm: [0x%08X] = 0x%08X\n",
                        scan_bases[b] + j, val);
            }
        }
    }

    /* ===========================================
     * SCAN 3: Look for TMR range in DF protection
     * =========================================== */
    pr_info("svm: === TMR PROTECTION SEARCH ===\n");
    {
        /* On Zen APUs, the DF has "System Fabric Address Map" registers
         * that define which address ranges map to which targets.
         * These are at DF offset 0x200-0x2FF.
         *
         * Also, the DRAM Controller has UMC registers for memory protection:
         * UMC at SMN 0x50000 + instance * 0x100000 */
        u32 umc_bases[] = {0x50000, 0x150000, 0x250000, 0x350000};
        int b;
        for (b = 0; b < 4; b++) {
            val = smn_read_pci(nb, umc_bases[b]);
            if (val != 0 && val != 0xFFFFFFFF) {
                pr_info("svm: UMC[%d][0x0] = 0x%08X\n", b, val);
                /* Read protection-related UMC registers */
                { int j;
                  for (j = 0; j < 0x100; j += 4) {
                    u32 v = smn_read_pci(nb, umc_bases[b] + j);
                    if (v != 0 && v != 0xFFFFFFFF) {
                        /* Check for TMR address patterns */
                        if ((v >> 16) == 0x2060 || (v >> 16) == 0x207F ||
                            (v & 0xFFFF0000) == 0x97E00000 ||
                            (v & 0xFFFF0000) == 0x88000000) {
                            pr_info("svm: *** UMC[%d][0x%02X] = 0x%08X (TMR?) ***\n",
                                b, j, v);
                        }
                    }
                  }
                }
            }
        }
    }

    if (gpu) pci_dev_put(gpu);
    pci_dev_put(nb);
    return 0;
}
static void __exit svm_exit(void) { pr_info("svm: unloaded\n"); }
module_init(svm_init); module_exit(svm_exit);
