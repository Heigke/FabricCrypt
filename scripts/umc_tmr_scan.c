#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
MODULE_LICENSE("GPL");
static struct pci_dev *g_nb;
static u32 smn_read(u32 a) { u32 v; pci_write_config_dword(g_nb,0x60,a); pci_read_config_dword(g_nb,0x64,&v); return v; }
static void smn_write(u32 a, u32 v) { pci_write_config_dword(g_nb,0x60,a); pci_write_config_dword(g_nb,0x64,v); }

static int __init uts_init(void) {
    int i, ch;
    u32 val;
    void *adev;
    struct pci_dev *gpu;

    g_nb = pci_get_domain_bus_and_slot(0,0,PCI_DEVFN(0,0));
    if (!g_nb) return -ENODEV;
    gpu = pci_get_device(0x1002,0x1586,NULL);
    if (gpu) adev = (u8*)pci_get_drvdata(gpu) - 0x10;

    pr_info("uts: === UMC DEEP SCAN FOR TMR PROTECTION ===\n");

    /* Full UMC channel scan (4 channels on this APU) */
    for (ch = 0; ch < 4; ch++) {
        u32 umc_base = 0x50000 + ch * 0x100000;
        int found = 0;

        pr_info("uts: --- UMC Channel %d (SMN 0x%08X) ---\n", ch, umc_base);
        for (i = 0; i < 0x2000 && found < 30; i += 4) {
            val = smn_read(umc_base + i);
            if (val != 0 && val != 0xFFFFFFFF) {
                pr_info("uts: UMC%d[0x%04X] = 0x%08X\n", ch, i, val);
                found++;
            }
        }
    }

    /* Also scan for VPR (Video Protected Range) specific registers
     * These might be at UMC + 0x800-0xFFF or similar */
    pr_info("uts: --- VPR/TMR specific scan ---\n");
    for (ch = 0; ch < 2; ch++) {
        u32 umc_base = 0x50000 + ch * 0x100000;
        for (i = 0x800; i < 0x1000; i += 4) {
            val = smn_read(umc_base + i);
            if (val != 0 && val != 0xFFFFFFFF)
                pr_info("uts: UMC%d[0x%04X] = 0x%08X\n", ch, i, val);
        }
    }

    /* Scan PSP C2PMSG registers via SMN — these control PSP commands */
    pr_info("uts: --- PSP C2PMSG via SMN ---\n");
    for (i = 0; i < 0x100; i += 4) {
        val = smn_read(0x3B10500 + i);
        if (val != 0 && val != 0xFFFFFFFF)
            pr_info("uts: C2PMSG[0x%02X] = 0x%08X\n", i, val);
    }

    /* Scan PSP TMR config registers (around 0x3B10080-0x3B100FF) */
    pr_info("uts: --- PSP TMR Config ---\n");
    for (i = 0x60; i < 0x200; i += 4) {
        val = smn_read(0x3B10000 + i);
        if (val != 0 && val != 0xFFFFFFFF)
            pr_info("uts: PSP_CFG[0x%03X] = 0x%08X\n", i, val);
    }

    /* NOVEL: Try reading TMR phys addr via SMN
     * If SMN can access the DRAM at TMR physical address... */
    pr_info("uts: --- NOVEL: Direct DRAM read via SMN ---\n");
    {
        /* On some AMD SoCs, SMN address 0x0 maps to DRAM start.
         * TMR phys = 0x2060000000. This is way beyond SMN address space (32-bit).
         * But there might be an indirect DRAM access path via UMC. */

        /* UMC has data path registers for maintenance reads.
         * UMC::MaintRead or similar debug registers might allow
         * reading arbitrary DRAM addresses. */

        /* Check UMC maintenance/debug register range */
        for (ch = 0; ch < 2; ch++) {
            u32 umc = 0x50000 + ch * 0x100000;
            /* Common UMC debug/maintenance register ranges */
            u32 debug_ranges[] = {0x100, 0x200, 0x300, 0x400, 0x500, 0x600, 0x700};
            int r;
            for (r = 0; r < 7; r++) {
                for (i = 0; i < 0x40; i += 4) {
                    val = smn_read(umc + debug_ranges[r] + i);
                    if (val != 0 && val != 0xFFFFFFFF)
                        pr_info("uts: UMC%d[0x%04X] = 0x%08X\n",
                            ch, debug_ranges[r]+i, val);
                }
            }
        }
    }

    if (gpu) pci_dev_put(gpu);
    pci_dev_put(g_nb);
    return 0;
}
static void __exit uts_exit(void) { pr_info("uts: unloaded\n"); }
module_init(uts_init); module_exit(uts_exit);
