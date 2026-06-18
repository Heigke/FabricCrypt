/*
 * smn_df_probe.c — Scan Data Fabric registers via SMN for TMR protection
 *
 * NOVEL INSIGHT: TMR protection is configured in the Data Fabric (DF).
 * DF registers are accessible via SMN (System Management Network).
 * We have SMN access via ryzen_smu driver.
 * If we find the TMR protection range registers and disable them,
 * TMR becomes accessible without PSP involvement.
 *
 * AMD DF registers of interest:
 * - DramHoleControl (0x104)
 * - DramBaseAddress (0x110 + 8*n)
 * - DramLimitAddress (0x114 + 8*n)
 * - SystemCfg (contains memory protection bits)
 * - CS/UMC base/limit registers
 *
 * DF registers are at SMN base 0x18000000 + offset.
 */
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
#include <linux/fs.h>
MODULE_LICENSE("GPL");

/* Read SMN register via ryzen_smu driver */
static u32 smn_read(u32 addr) {
    struct file *fp;
    u32 val = 0xDEAD;

    /* Write address to smn file, then read value */
    fp = filp_open("/sys/kernel/ryzen_smu_drv/smn", O_RDWR, 0);
    if (IS_ERR(fp)) return 0xDEAD;
    {
        loff_t pos = 0;
        /* Write the SMN address (4 bytes) */
        kernel_write(fp, &addr, 4, &pos);
        /* Read the value (4 bytes) */
        pos = 0;
        kernel_read(fp, &val, 4, &pos);
    }
    filp_close(fp, NULL);
    return val;
}

static int __init sdf_init(void) {
    int i;
    u32 val;

    pr_info("sdf: === SMN DATA FABRIC TMR PROTECTION SCAN ===\n");

    /* Check if SMN is accessible */
    val = smn_read(0x00000000);
    pr_info("sdf: SMN[0x0] = 0x%08X (sanity check)\n", val);
    if (val == 0xDEAD) {
        pr_info("sdf: SMN not accessible via ryzen_smu\n");
        return -ENODEV;
    }

    /* DF Base: 0x18000000 on Zen4/Zen5 APUs
     * DF registers:
     *   0x044: FabricBlockInstanceCount
     *   0x104: DramHoleControl
     *   0x110+8*n: DramBaseAddress[n]
     *   0x114+8*n: DramLimitAddress[n]
     *   0x200: SystemCfgRegister
     *   0x240+8*n: MmioBaseAddress[n]
     *   0x244+8*n: MmioLimitAddress[n]
     */

    pr_info("sdf: --- DF Core Registers ---\n");
    {
        u32 df_base = 0x18000000;
        u32 regs[] = {
            0x000, 0x004, 0x008, 0x00C,  /* DF ID registers */
            0x040, 0x044, 0x048, 0x04C,  /* Instance info */
            0x100, 0x104, 0x108, 0x10C,  /* DRAM hole */
            0x200, 0x204, 0x208, 0x20C,  /* System config */
        };
        for (i = 0; i < 16; i++) {
            val = smn_read(df_base + regs[i]);
            if (val != 0 && val != 0xFFFFFFFF && val != 0xDEAD)
                pr_info("sdf: DF[0x%03X] = 0x%08X\n", regs[i], val);
        }
    }

    /* DRAM Base/Limit registers — define memory regions */
    pr_info("sdf: --- DRAM Base/Limit Regions ---\n");
    {
        u32 df_base = 0x18000000;
        for (i = 0; i < 16; i++) {
            u32 base_reg = 0x110 + i * 8;
            u32 limit_reg = 0x114 + i * 8;
            u32 base_val = smn_read(df_base + base_reg);
            u32 limit_val = smn_read(df_base + limit_reg);
            if ((base_val != 0 && base_val != 0xFFFFFFFF) ||
                (limit_val != 0 && limit_val != 0xFFFFFFFF)) {
                pr_info("sdf: DRAM[%d] base=0x%08X limit=0x%08X\n",
                    i, base_val, limit_val);
                /* Decode: base address = base_val[31:12] << 28
                 * RE/WE bits in lower bits */
            }
        }
    }

    /* MMIO Base/Limit — defines MMIO-mapped regions */
    pr_info("sdf: --- MMIO Base/Limit Regions ---\n");
    {
        u32 df_base = 0x18000000;
        for (i = 0; i < 16; i++) {
            u32 base_reg = 0x240 + i * 8;
            u32 limit_reg = 0x244 + i * 8;
            u32 base_val = smn_read(df_base + base_reg);
            u32 limit_val = smn_read(df_base + limit_reg);
            if ((base_val != 0 && base_val != 0xFFFFFFFF) ||
                (limit_val != 0 && limit_val != 0xFFFFFFFF)) {
                pr_info("sdf: MMIO[%d] base=0x%08X limit=0x%08X\n",
                    i, base_val, limit_val);
            }
        }
    }

    /* GPU-specific DF registers — TMR protection
     * On APUs, the GPU shares the DF with the CPU.
     * Look for protection/exclusion ranges that match TMR MC addr.
     * TMR at MC 0x97E0000000, phys 0x2060000000 */
    pr_info("sdf: --- Searching for TMR protection config ---\n");
    {
        u32 df_base = 0x18000000;
        /* Scan a wider range of DF registers */
        for (i = 0; i < 0x400; i += 4) {
            val = smn_read(df_base + i);
            if (val != 0 && val != 0xFFFFFFFF && val != 0xDEAD) {
                /* Check if this value relates to TMR addresses */
                /* TMR phys = 0x2060000000.
                 * In DF register encoding: addr >> 28 = 0x206
                 * Or addr >> 24 = 0x2060 */
                if ((val & 0xFFF00000) == 0x20600000 ||
                    (val & 0xFFFFF) == 0x00206 ||
                    (val >> 12) == 0x20600) {
                    pr_info("sdf: *** TMR-RELATED? DF[0x%03X] = 0x%08X ***\n",
                        i, val);
                }
                /* Also check for TMR MC address bits */
                if ((val & 0xFFF) == 0x97E ||
                    (val >> 20) == 0x97E ||
                    val == 0x97E00000 || val == 0x0097E000) {
                    pr_info("sdf: *** TMR MC? DF[0x%03X] = 0x%08X ***\n",
                        i, val);
                }
            }
        }
    }

    /* Also scan GPU-specific SMN ranges
     * GPU MMHUB registers might be at different SMN offsets */
    pr_info("sdf: --- GPU MMHUB SMN scan ---\n");
    {
        /* MMHUB registers at SMN 0x1A000000 (varies by SoC) */
        u32 mmhub_bases[] = {0x1A000000, 0x1B000000, 0x3400000, 0x3800000};
        int b;
        for (b = 0; b < 4; b++) {
            val = smn_read(mmhub_bases[b]);
            if (val != 0 && val != 0xFFFFFFFF && val != 0xDEAD)
                pr_info("sdf: SMN[0x%08X] = 0x%08X (MMHUB candidate)\n",
                    mmhub_bases[b], val);
        }
    }

    /* PSP-related SMN registers
     * PSP registers at SMN 0x3B10000 or 0x3C00000 (varies) */
    pr_info("sdf: --- PSP SMN registers ---\n");
    {
        u32 psp_bases[] = {0x3B10000, 0x3B10004, 0x3B10008, 0x3B1000C,
                           0x3B10500, 0x3B10504, 0x3B10508,
                           0x3C00000, 0x3C00004, 0x3C00008};
        for (i = 0; i < 10; i++) {
            val = smn_read(psp_bases[i]);
            if (val != 0 && val != 0xFFFFFFFF && val != 0xDEAD)
                pr_info("sdf: PSP[0x%07X] = 0x%08X\n", psp_bases[i], val);
        }
    }

    return 0;
}
static void __exit sdf_exit(void) { pr_info("sdf: unloaded\n"); }
module_init(sdf_init); module_exit(sdf_exit);
