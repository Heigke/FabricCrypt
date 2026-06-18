#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
MODULE_LICENSE("GPL");

static struct pci_dev *g_nb;

static u32 smn_read(u32 addr) {
    u32 val;
    pci_write_config_dword(g_nb, 0x60, addr);
    pci_read_config_dword(g_nb, 0x64, &val);
    return val;
}

static void smn_write(u32 addr, u32 val) {
    pci_write_config_dword(g_nb, 0x60, addr);
    pci_write_config_dword(g_nb, 0x64, val);
}

static int __init sds_init(void) {
    int i;
    u32 val;
    struct pci_dev *df;

    g_nb = pci_get_domain_bus_and_slot(0, 0, PCI_DEVFN(0, 0));
    if (!g_nb) return -ENODEV;

    pr_info("sds: === DEEP SMN + DF CONFIG SCAN ===\n");

    /* ============================================
     * PART 1: PSP SMN registers (0x3B10000 range)
     * ============================================ */
    pr_info("sds: --- PSP SMN Space (0x3B10000) ---\n");
    for (i = 0; i < 0x600; i += 4) {
        val = smn_read(0x3B10000 + i);
        if (val != 0 && val != 0xFFFFFFFF)
            pr_info("sds: PSP[0x%03X] = 0x%08X\n", i, val);
    }

    /* Also scan PSP extended range */
    pr_info("sds: --- PSP Extended (0x3B00000) ---\n");
    for (i = 0; i < 0x100; i += 4) {
        val = smn_read(0x3B00000 + i);
        if (val != 0 && val != 0xFFFFFFFF)
            pr_info("sds: PSP_EXT[0x%03X] = 0x%08X\n", i, val);
    }

    /* ============================================
     * PART 2: DF Indirect Config Access
     * via PCI device 0:18.0 (Data Fabric)
     * ============================================ */
    pr_info("sds: --- DF Indirect Config Access ---\n");
    df = pci_get_domain_bus_and_slot(0, 0x18, PCI_DEVFN(0, 0));
    if (!df) {
        pr_info("sds: No DF at 0:18.0, trying to find...\n");
        /* On some APUs it's at a different BDF */
        df = pci_get_device(0x1022, PCI_ANY_ID, NULL);
        while (df) {
            if (df->bus->number == 0 && PCI_SLOT(df->devfn) == 0x18) {
                pr_info("sds: Found DF at %s\n", pci_name(df));
                break;
            }
            df = pci_get_device(0x1022, PCI_ANY_ID, df);
        }
    }

    if (df) {
        /* FICAA at config offset 0x5C, FICAD at 0x98
         * Or: DF config via function-specific registers at offset 0x40-0x60 */
        u32 ficaa, ficad;

        /* Read some DF registers via config space */
        for (i = 0; i < 8; i++) {
            struct pci_dev *df_fn = pci_get_domain_bus_and_slot(0, 0x18, PCI_DEVFN(0, i));
            if (df_fn) {
                u32 v0, v4;
                pci_read_config_dword(df_fn, 0x00, &v0);
                pci_read_config_dword(df_fn, 0x04, &v4);
                if (v0 != 0xFFFFFFFF)
                    pr_info("sds: DF fn%d: vendor/dev=0x%08X\n", i, v0);
                pci_dev_put(df_fn);
            }
        }

        /* Use FICAA/FICAD for indirect register access
         * FICAA format: [31:16]=register, [10:8]=function, [3:0]=instance */
        pr_info("sds: DF FICAA/FICAD probe:\n");
        {
            /* Read DramHoleControl (DF reg 0x104) */
            u32 ficaa_val = (0x104 << 2) | (0 << 8) | 0; /* reg=0x104, fn=0, inst=0 */
            /* Actually FICAA format varies by generation. Let me try direct read. */

            /* On Zen4/5: use DF::FabricBlockInstanceInformation0
             * Read via PCI config at fn0 offset 0x44 */
            struct pci_dev *df0 = pci_get_domain_bus_and_slot(0, 0x18, PCI_DEVFN(0, 0));
            if (df0) {
                u32 regs[] = {0x44, 0x48, 0x50, 0x90, 0x94, 0x98, 0x9C,
                              0xA0, 0xA4, 0xA8, 0xB0, 0xB4, 0xB8, 0xBC,
                              0xC0, 0xC4, 0xC8, 0xCC, 0xD0, 0xD4};
                int r;
                for (r = 0; r < 20; r++) {
                    pci_read_config_dword(df0, regs[r], &val);
                    if (val != 0 && val != 0xFFFFFFFF)
                        pr_info("sds: DF0[0x%02X] = 0x%08X\n", regs[r], val);
                }

                /* Specifically look for memory protection/exclusion registers */
                /* DF::DramBaseAddress and DF::DramLimitAddress
                 * On Zen4: at fn0 offset 0x40+n*8 (old method)
                 * Or via indirect: FICAA/FICAD */

                /* Try reading ALL PCI config of DF fn0 */
                pr_info("sds: DF0 full config scan:\n");
                for (r = 0; r < 256; r += 4) {
                    pci_read_config_dword(df0, r, &val);
                    if (val != 0 && val != 0xFFFFFFFF &&
                        r >= 0x40) /* skip standard PCI header */
                        pr_info("sds: DF0[0x%02X] = 0x%08X\n", r, val);
                }

                pci_dev_put(df0);
            }

            /* Also check DF function 1-7 for memory map registers */
            { int fn;
              for (fn = 1; fn < 8; fn++) {
                struct pci_dev *dfn = pci_get_domain_bus_and_slot(0, 0x18, PCI_DEVFN(0, fn));
                if (dfn) {
                    u32 v;
                    int r;
                    int has_data = 0;
                    for (r = 0x40; r < 0x100; r += 4) {
                        pci_read_config_dword(dfn, r, &v);
                        if (v != 0 && v != 0xFFFFFFFF) {
                            if (!has_data) { pr_info("sds: DF fn%d:\n", fn); has_data = 1; }
                            pr_info("sds:   [0x%02X] = 0x%08X\n", r, v);
                        }
                    }
                    pci_dev_put(dfn);
                }
              }
            }
        }

        pci_dev_put(df);
    }

    /* ============================================
     * PART 3: Brute-force SMN scan for non-zero regions
     * ============================================ */
    pr_info("sds: --- SMN Brute Force (key ranges) ---\n");
    {
        /* Scan known AMD SoC SMN ranges */
        struct { u32 base; const char *name; } ranges[] = {
            {0x00050000, "UMC0"},
            {0x00150000, "UMC1"},
            {0x01400000, "IOHC"},
            {0x01500000, "NBIO"},
            {0x03800000, "GC_SMN"},
            {0x03B00000, "PSP_MP0"},
            {0x03C00000, "SMU_MP1"},
            {0x0A000000, "DCN"},
            {0x10000000, "SDMA"},
            {0x13000000, "VCN"},
        };
        int r;
        for (r = 0; r < 10; r++) {
            int j, found = 0;
            for (j = 0; j < 0x100 && found < 5; j += 4) {
                val = smn_read(ranges[r].base + j);
                if (val != 0 && val != 0xFFFFFFFF) {
                    pr_info("sds: %s[0x%02X] = 0x%08X\n", ranges[r].name, j, val);
                    found++;
                }
            }
        }
    }

    pci_dev_put(g_nb);
    return 0;
}
static void __exit sds_exit(void) { pr_info("sds: unloaded\n"); }
module_init(sds_init); module_exit(sds_exit);
