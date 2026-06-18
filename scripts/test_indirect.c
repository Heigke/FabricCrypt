/*
 * test_indirect.c — Test different indirect register access paths
 * GFX11 may need different index/data pairs for different IP blocks
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/spinlock.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002

static void __iomem *mmio;
static resource_size_t bar5_size;

/* Multiple indirect register access pairs */
/* Standard MMIO */
#define PCIE_INDEX   0x000C  /* BIF_BX0_PCIE_INDEX */
#define PCIE_DATA    0x000D  /* BIF_BX0_PCIE_DATA */
/* Extended MMIO */ 
#define PCIE_INDEX2  0x000E  /* BIF_BX0_PCIE_INDEX2 */
#define PCIE_DATA2   0x000F  /* BIF_BX0_PCIE_DATA2 */
/* High address */
#define PCIE_INDEX_HI 0x0010 /* BIF_BX0_PCIE_INDEX_HI */

static u32 direct_read(u32 dw_off)
{
    return readl(mmio + (u64)dw_off * 4);
}

static void direct_write(u32 dw_off, u32 val)
{
    writel(val, mmio + (u64)dw_off * 4);
}

/* Read via PCIE_INDEX2/DATA2 (byte address) */
static u32 indirect2_read(u32 byte_addr)
{
    writel(byte_addr, mmio + (u64)PCIE_INDEX2 * 4);
    readl(mmio + (u64)PCIE_INDEX2 * 4); /* flush */
    return readl(mmio + (u64)PCIE_DATA2 * 4);
}

/* Read via PCIE_INDEX/DATA with HI for 64-bit addressing */
static u32 indirect_hi_read(u32 byte_addr)
{
    /* Set high address bits to 0 */
    writel(0, mmio + (u64)PCIE_INDEX_HI * 4);
    readl(mmio + (u64)PCIE_INDEX_HI * 4);
    writel(byte_addr, mmio + (u64)PCIE_INDEX * 4);
    readl(mmio + (u64)PCIE_INDEX * 4);
    return readl(mmio + (u64)PCIE_DATA * 4);
}

static void indirect_hi_write(u32 byte_addr, u32 val)
{
    writel(0, mmio + (u64)PCIE_INDEX_HI * 4);
    readl(mmio + (u64)PCIE_INDEX_HI * 4);
    writel(byte_addr, mmio + (u64)PCIE_INDEX * 4);
    readl(mmio + (u64)PCIE_INDEX * 4);
    writel(val, mmio + (u64)PCIE_DATA * 4);
}

#define GC_BASE_0  0x1260
#define GC_BASE_1  0xC00000

/* Test registers */
#define mmCP_MEC_CNTL           (GC_BASE_1 + 0x0802)
#define mmCP_CPC_IC_BASE_LO     (GC_BASE_1 + 0x584C)
#define mmCP_CPC_IC_BASE_HI     (GC_BASE_1 + 0x584D)
#define mmCP_CPC_IC_OP_CNTL     (GC_BASE_1 + 0x297A)
#define mmGRBM_STATUS           (GC_BASE_0 + 0x0DA4)
#define mmGRBM_GFX_CNTL         (GC_BASE_0 + 0x0013)

/* RLC registers (sometimes needed for accessing protected regs) */
#define mmRLC_SPM_MC_CNTL       (GC_BASE_1 + 0x3C00)
#define mmRLC_SAFE_MODE         (GC_BASE_1 + 0x4C00)

static int __init test_init(void)
{
    struct pci_dev *pdev = NULL;
    u32 val;

    while ((pdev = pci_get_device(AMD_VENDOR_ID, PCI_ANY_ID, pdev))) {
        if ((pdev->class >> 8) == PCI_CLASS_DISPLAY_OTHER)
            break;
    }
    if (!pdev) return -ENODEV;

    bar5_size = pci_resource_len(pdev, 5);
    mmio = pci_iomap(pdev, 5, 0);
    if (!mmio) { pci_dev_put(pdev); return -ENOMEM; }

    pr_info("tind: BAR5 size = %llu (0x%llX)\n", (u64)bar5_size, (u64)bar5_size);

    /* Test 1: Read GRBM_STATUS (known-working, should be non-zero) */
    /* GRBM_STATUS dw offset = GC_BASE_0 + 0x0DA4 = 0x1260 + 0x0DA4 = 0x2004 */
    /* byte addr = 0x2004 * 4 = 0x8010 — within BAR5 1MB range */
    val = direct_read(mmGRBM_STATUS);
    pr_info("tind: GRBM_STATUS (direct @0x%X) = 0x%08X\n", mmGRBM_STATUS, val);

    /* Test 2: Read CP_MEC_CNTL via different methods */
    /* dw offset = 0xC00802, byte addr = 0x3002008 */
    pr_info("tind: CP_MEC_CNTL dw=0x%X byte=0x%X\n", mmCP_MEC_CNTL, mmCP_MEC_CNTL * 4);
    
    val = indirect2_read(mmCP_MEC_CNTL * 4);
    pr_info("tind: CP_MEC_CNTL (INDEX2, byte 0x%X) = 0x%08X\n", mmCP_MEC_CNTL * 4, val);
    
    val = indirect_hi_read(mmCP_MEC_CNTL * 4);
    pr_info("tind: CP_MEC_CNTL (INDEX+HI, byte 0x%X) = 0x%08X\n", mmCP_MEC_CNTL * 4, val);

    /* Test 3: Read IC_BASE_LO via different methods */
    pr_info("tind: IC_BASE_LO dw=0x%X byte=0x%X\n", mmCP_CPC_IC_BASE_LO, mmCP_CPC_IC_BASE_LO * 4);
    
    val = indirect2_read(mmCP_CPC_IC_BASE_LO * 4);
    pr_info("tind: IC_BASE_LO (INDEX2) = 0x%08X\n", val);
    
    val = indirect_hi_read(mmCP_CPC_IC_BASE_LO * 4);
    pr_info("tind: IC_BASE_LO (INDEX+HI) = 0x%08X\n", val);

    /* Test 4: IC_OP_CNTL */
    val = indirect2_read(mmCP_CPC_IC_OP_CNTL * 4);
    pr_info("tind: IC_OP_CNTL (INDEX2) = 0x%08X\n", val);
    
    /* Test 5: Check if INDEX2 address range is limited
     * PCIE_INDEX2 is 32-bit, max byte addr = 0xFFFFFFFF
     * Our addresses: CP_MEC_CNTL = 0x3002008, IC_BASE_LO = 0x1416130
     * These should all fit */
    
    /* Test 6: Try reading via RLCG (RLC-guarded) path
     * On GFX11, some registers require going through RLC for access
     * The driver uses RREG32_SOC15 which can route through RLC */
    
    /* Test 7: Check PCIE_INDEX2 readback */
    direct_write(PCIE_INDEX2, mmCP_MEC_CNTL * 4);
    val = direct_read(PCIE_INDEX2);
    pr_info("tind: PCIE_INDEX2 readback = 0x%08X (wrote 0x%08X)\n", val, mmCP_MEC_CNTL * 4);
    
    direct_write(PCIE_INDEX2, mmCP_CPC_IC_BASE_LO * 4);
    val = direct_read(PCIE_INDEX2);
    pr_info("tind: PCIE_INDEX2 readback = 0x%08X (wrote 0x%08X)\n", val, mmCP_CPC_IC_BASE_LO * 4);

    /* Test 8: Read some registers with known non-zero values to verify indirect works */
    /* Try reading GRBM_STATUS via indirect (it's within BAR5 but let's test) */
    val = indirect2_read(mmGRBM_STATUS * 4);
    pr_info("tind: GRBM_STATUS (INDEX2) = 0x%08X\n", val);

    pci_iounmap(pdev, mmio);
    pci_dev_put(pdev);
    return -ENODEV;
}

static void __exit test_exit(void) {}
module_init(test_init);
module_exit(test_exit);
