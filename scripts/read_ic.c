#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/spinlock.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002
#define GC_BASE_0  0x1260
#define GC_BASE_1  0xC00000
#define PCIE_INDEX2  0x000e
#define PCIE_DATA2   0x000f

static void __iomem *mmio;
static resource_size_t bar5_size;
static DEFINE_SPINLOCK(indirect_lock);

static u32 rr(u32 off)
{
    if ((u64)off * 4 < bar5_size) {
        return readl(mmio + (u64)off * 4);
    } else {
        u32 r;
        unsigned long flags;
        spin_lock_irqsave(&indirect_lock, flags);
        writel(off * 4, mmio + (u64)PCIE_INDEX2 * 4);
        readl(mmio + (u64)PCIE_INDEX2 * 4);
        r = readl(mmio + (u64)PCIE_DATA2 * 4);
        spin_unlock_irqrestore(&indirect_lock, flags);
        return r;
    }
}

static void wr(u32 off, u32 val)
{
    if ((u64)off * 4 < bar5_size) {
        writel(val, mmio + (u64)off * 4);
    } else {
        unsigned long flags;
        spin_lock_irqsave(&indirect_lock, flags);
        writel(off * 4, mmio + (u64)PCIE_INDEX2 * 4);
        readl(mmio + (u64)PCIE_INDEX2 * 4);
        writel(val, mmio + (u64)PCIE_DATA2 * 4);
        spin_unlock_irqrestore(&indirect_lock, flags);
    }
}

#define mmGRBM_GFX_CNTL  (GC_BASE_0 + 0x0013)

static void select_me_pipe_q(u32 me, u32 pipe, u32 queue, u32 vmid)
{
    u32 val = (pipe & 3) | ((me & 3) << 2) | ((vmid & 0xF) << 4) | ((queue & 7) << 8);
    wr(mmGRBM_GFX_CNTL, val);
    rr(mmGRBM_GFX_CNTL);
}

/* CPC IC_BASE (banked) */
#define mmCP_CPC_IC_BASE_LO    (GC_BASE_1 + 0x584C)
#define mmCP_CPC_IC_BASE_HI    (GC_BASE_1 + 0x584D)
#define mmCP_CPC_IC_BASE_CNTL  (GC_BASE_1 + 0x584E)

/* MES IC_BASE (pipe-banked) */
#define mmCP_MES_IC_BASE_LO    (GC_BASE_1 + 0x5850)
#define mmCP_MES_IC_BASE_HI    (GC_BASE_1 + 0x5851)
#define mmCP_MES_IC_BASE_CNTL  (GC_BASE_1 + 0x5852)

/* MES program counter start */
#define mmCP_MES_PRGRM_CNTR_START    (GC_BASE_1 + 0x2800)
#define mmCP_MES_PRGRM_CNTR_START_HI (GC_BASE_1 + 0x289D)

/* HQD active */
#define mmCP_HQD_ACTIVE  (GC_BASE_1 + 0x3247)

/* CP_MEC_CNTL */
#define mmCP_MEC_CNTL    (GC_BASE_1 + 0x0802)

/* MEC instruction base */
#define mmCP_MEC_LOCAL_INSTR_BASE_LO  (GC_BASE_1 + 0x292C)
#define mmCP_MEC_LOCAL_INSTR_BASE_HI  (GC_BASE_1 + 0x292D)

static int __init read_ic_init(void)
{
    struct pci_dev *pdev = NULL;
    u32 orig_grbm;
    int me, pipe;

    while ((pdev = pci_get_device(AMD_VENDOR_ID, PCI_ANY_ID, pdev))) {
        if ((pdev->class >> 8) == PCI_CLASS_DISPLAY_OTHER)
            break;
    }
    if (!pdev) {
        pr_err("read_ic: no AMD GPU found\n");
        return -ENODEV;
    }

    bar5_size = pci_resource_len(pdev, 5);
    mmio = pci_iomap(pdev, 5, 0);
    if (!mmio) {
        pci_dev_put(pdev);
        return -ENOMEM;
    }

    orig_grbm = rr(mmGRBM_GFX_CNTL);
    
    pr_info("read_ic: CP_MEC_CNTL = 0x%08X\n", rr(mmCP_MEC_CNTL));
    
    /* Read IC_BASE for each ME/PIPE without checking HQD_ACTIVE */
    for (me = 0; me <= 2; me++) {
        for (pipe = 0; pipe < 4; pipe++) {
            u32 ic_lo, ic_hi, ic_cntl;
            u32 mes_ic_lo, mes_ic_hi, mes_ic_cntl;
            u32 pc_lo, pc_hi;
            u32 instr_lo, instr_hi;
            u32 hqd;
            
            select_me_pipe_q(me, pipe, 0, 0);
            
            ic_lo = rr(mmCP_CPC_IC_BASE_LO);
            ic_hi = rr(mmCP_CPC_IC_BASE_HI);
            ic_cntl = rr(mmCP_CPC_IC_BASE_CNTL);
            
            mes_ic_lo = rr(mmCP_MES_IC_BASE_LO);
            mes_ic_hi = rr(mmCP_MES_IC_BASE_HI);
            mes_ic_cntl = rr(mmCP_MES_IC_BASE_CNTL);
            
            pc_lo = rr(mmCP_MES_PRGRM_CNTR_START);
            pc_hi = rr(mmCP_MES_PRGRM_CNTR_START_HI);
            
            instr_lo = rr(mmCP_MEC_LOCAL_INSTR_BASE_LO);
            instr_hi = rr(mmCP_MEC_LOCAL_INSTR_BASE_HI);
            
            hqd = rr(mmCP_HQD_ACTIVE);
            
            if (ic_lo || ic_hi || mes_ic_lo || mes_ic_hi || pc_lo || pc_hi) {
                pr_info("read_ic: ME%d P%d: CPC_IC=0x%08X:%08X cntl=0x%X  MES_IC=0x%08X:%08X cntl=0x%X  PC=0x%08X:%08X  INSTR=0x%08X:%08X  HQD=%d\n",
                    me, pipe, ic_hi, ic_lo, ic_cntl,
                    mes_ic_hi, mes_ic_lo, mes_ic_cntl,
                    pc_hi, pc_lo,
                    instr_hi, instr_lo, hqd);
            }
        }
    }
    
    /* Restore */
    wr(mmGRBM_GFX_CNTL, orig_grbm);
    
    pr_info("read_ic: === DONE ===\n");
    pci_iounmap(pdev, mmio);
    pci_dev_put(pdev);
    return -ENODEV;
}

static void __exit read_ic_exit(void) {}
module_init(read_ic_init);
module_exit(read_ic_exit);
