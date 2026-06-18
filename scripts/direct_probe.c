/*
 * direct_probe.c — Probe GFX/MES/RLC registers after fw_load_type=0 failure
 * Maps MMIO BAR directly and reads key status registers to understand
 * where DIRECT firmware loading failed.
 */
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Probe GFX registers after DIRECT mode failure");

/* GFX11 register offsets (from IP discovery, BASE_IDX=0 → seg0 at 0x0) */
/* GRBM */
#define regGRBM_STATUS          0x0504  /* 4-byte reg index */
#define regGRBM_STATUS2         0x0502
#define regGRBM_STATUS3         0x0503
#define regGRBM_GFX_CNTL       0x0640  /* GRBM select */

/* CP */
#define regCP_STAT              0x0E40
#define regCP_MEC_CNTL          0x0E58
#define regCP_CPC_STATUS        0x0E54
#define regCP_CPF_STATUS        0x0E53
#define regCP_MES_CNTL          0x28C0
#define regCP_GFX_RS64_DC_BASE0_LO  0x2D14
#define regCP_GFX_RS64_DC_BASE0_HI  0x2D15
#define regCP_MEC_RS64_DC_BASE0_LO  0x2D25
#define regCP_MEC_RS64_DC_BASE0_HI  0x2D26

/* These are the IC_BASE registers we've been trying to write */
#define regCP_MEC_ME1_UCODE_ADDR   0x2D1C
#define regCP_MEC_ME1_UCODE_DATA   0x2D1D

/* RLC */
#define regRLC_CNTL             0x4C00
#define regRLC_STAT             0x4C04
#define regRLC_SAFE_MODE        0x4C20
#define regRLC_GPM_STAT         0x4C01

/* MES */
#define regCP_MES_PRGRM_CNTR_START  0x28C5
#define regCP_MES_INSTR_PNTR        0x28E1
#define regCP_MES_DOORBELL_CONTROL  0x28D2
#define regCP_MES_GP0_LO            0x28D9
#define regCP_MES_GP0_HI            0x28DA
#define regCP_MES_IC_BASE_LO        0x28C1
#define regCP_MES_IC_BASE_HI        0x28C2

/* SDMA */
#define regSDMA0_STATUS_REG     0x0D85

/* SMU/MP1 */
#define regMP1_SMN_C2PMSG_90    0x029A  /* SMU response */

/* PSP */
#define regMP0_SMN_C2PMSG_35    0x006B  /* PSP bootloader status */
#define regMP0_SMN_C2PMSG_64    0x0080
#define regMP0_SMN_C2PMSG_81    0x0091

/* Actually, MP0/MP1 are in different address space. Let me use proper MMIO offsets */
/* BAR5 contains all doorbell + MMIO registers at 4-byte granularity */

static void __iomem *mmio;

static u32 rreg(u32 reg_idx)
{
    return readl(mmio + (reg_idx * 4));
}

static int __init direct_probe_init(void)
{
    struct pci_dev *pdev;
    resource_size_t bar_start, bar_len;
    u32 val;

    pdev = pci_get_device(0x1002, 0x1586, NULL);
    if (!pdev) {
        pr_err("direct_probe: GPU 1002:1586 not found\n");
        return -ENODEV;
    }

    bar_start = pci_resource_start(pdev, 5);
    bar_len = pci_resource_len(pdev, 5);
    pr_info("direct_probe: BAR5 at 0x%llx len 0x%llx\n",
            (u64)bar_start, (u64)bar_len);

    if (!bar_start || !bar_len) {
        pr_err("direct_probe: BAR5 not available\n");
        pci_dev_put(pdev);
        return -ENODEV;
    }

    mmio = ioremap(bar_start, bar_len);
    if (!mmio) {
        pr_err("direct_probe: ioremap failed\n");
        pci_dev_put(pdev);
        return -ENOMEM;
    }

    pr_info("direct_probe: === GFX REGISTER STATE AFTER DIRECT MODE ===\n");

    /* GRBM status */
    val = rreg(regGRBM_STATUS);
    pr_info("direct_probe: GRBM_STATUS      = 0x%08x\n", val);
    val = rreg(regGRBM_STATUS2);
    pr_info("direct_probe: GRBM_STATUS2     = 0x%08x\n", val);
    val = rreg(regGRBM_STATUS3);
    pr_info("direct_probe: GRBM_STATUS3     = 0x%08x\n", val);

    /* CP status */
    val = rreg(regCP_STAT);
    pr_info("direct_probe: CP_STAT          = 0x%08x\n", val);
    val = rreg(regCP_MEC_CNTL);
    pr_info("direct_probe: CP_MEC_CNTL      = 0x%08x\n", val);
    val = rreg(regCP_CPC_STATUS);
    pr_info("direct_probe: CP_CPC_STATUS    = 0x%08x\n", val);
    val = rreg(regCP_CPF_STATUS);
    pr_info("direct_probe: CP_CPF_STATUS    = 0x%08x\n", val);

    /* RLC */
    val = rreg(regRLC_CNTL);
    pr_info("direct_probe: RLC_CNTL         = 0x%08x\n", val);
    val = rreg(regRLC_STAT);
    pr_info("direct_probe: RLC_STAT         = 0x%08x\n", val);
    val = rreg(regRLC_GPM_STAT);
    pr_info("direct_probe: RLC_GPM_STAT     = 0x%08x\n", val);
    val = rreg(regRLC_SAFE_MODE);
    pr_info("direct_probe: RLC_SAFE_MODE    = 0x%08x\n", val);

    /* MES registers */
    val = rreg(regCP_MES_CNTL);
    pr_info("direct_probe: CP_MES_CNTL      = 0x%08x\n", val);
    val = rreg(regCP_MES_PRGRM_CNTR_START);
    pr_info("direct_probe: MES_PRGRM_START  = 0x%08x\n", val);
    val = rreg(regCP_MES_INSTR_PNTR);
    pr_info("direct_probe: MES_INSTR_PNTR   = 0x%08x\n", val);
    val = rreg(regCP_MES_IC_BASE_LO);
    pr_info("direct_probe: MES_IC_BASE_LO   = 0x%08x\n", val);
    val = rreg(regCP_MES_IC_BASE_HI);
    pr_info("direct_probe: MES_IC_BASE_HI   = 0x%08x\n", val);
    val = rreg(regCP_MES_DOORBELL_CONTROL);
    pr_info("direct_probe: MES_DOORBELL_CTL = 0x%08x\n", val);

    /* MEC IC_BASE (RS64 data cache base) */
    val = rreg(regCP_GFX_RS64_DC_BASE0_LO);
    pr_info("direct_probe: GFX_RS64_DC_BASE_LO = 0x%08x\n", val);
    val = rreg(regCP_GFX_RS64_DC_BASE0_HI);
    pr_info("direct_probe: GFX_RS64_DC_BASE_HI = 0x%08x\n", val);
    val = rreg(regCP_MEC_RS64_DC_BASE0_LO);
    pr_info("direct_probe: MEC_RS64_DC_BASE_LO = 0x%08x\n", val);
    val = rreg(regCP_MEC_RS64_DC_BASE0_HI);
    pr_info("direct_probe: MEC_RS64_DC_BASE_HI = 0x%08x\n", val);

    /* SDMA */
    val = rreg(regSDMA0_STATUS_REG);
    pr_info("direct_probe: SDMA0_STATUS     = 0x%08x\n", val);

    /* Try to read some indirect registers via GRBM_GFX_INDEX */
    /* First, try PIPE0 ME0 (GFX pipe) */
    writel(0x00000000, mmio + (regGRBM_GFX_CNTL * 4)); /* ME0, PIPE0, QUEUE0 */
    mb();
    udelay(100);

    val = rreg(regCP_MES_CNTL);
    pr_info("direct_probe: MES_CNTL(P0)     = 0x%08x\n", val);

    /* Try PIPE1 (MEC pipe) */
    writel(0x00000002, mmio + (regGRBM_GFX_CNTL * 4)); /* ME0, PIPE1 */
    mb();
    udelay(100);

    val = rreg(regCP_MES_CNTL);
    pr_info("direct_probe: MES_CNTL(P1)     = 0x%08x\n", val);

    /* Reset GRBM select */
    writel(0x00000000, mmio + (regGRBM_GFX_CNTL * 4));
    mb();

    /* Check if MES firmware was actually written to VRAM */
    /* Read the GP registers which MES uses for status */
    val = rreg(regCP_MES_GP0_LO);
    pr_info("direct_probe: MES_GP0_LO       = 0x%08x\n", val);
    val = rreg(regCP_MES_GP0_HI);
    pr_info("direct_probe: MES_GP0_HI       = 0x%08x\n", val);

    /* Try reading some more registers that indicate if firmware was loaded */
    /* CP_MEC_ME1_UCODE_ADDR — writing non-zero to this was our original goal */
    val = rreg(regCP_MEC_ME1_UCODE_ADDR);
    pr_info("direct_probe: MEC_UCODE_ADDR   = 0x%08x\n", val);
    val = rreg(regCP_MEC_ME1_UCODE_DATA);
    pr_info("direct_probe: MEC_UCODE_DATA   = 0x%08x\n", val);

    /* Now try to READ IC_BASE via proper GFX11 register paths */
    /* On RS64, the instruction cache base is set via CP_MES_IC_BASE */
    /* Check all the instruction pointer registers */

    pr_info("direct_probe: === REGISTER SCAN 0x28C0-0x28FF (MES range) ===\n");
    {
        int i;
        for (i = 0x28C0; i <= 0x28FF; i++) {
            val = rreg(i);
            if (val != 0 && val != 0xFFFFFFFF)
                pr_info("direct_probe: reg[0x%04x] = 0x%08x\n", i, val);
        }
    }

    /* Check RLC SPM range */
    pr_info("direct_probe: === KEY RLC REGISTERS ===\n");
    {
        int regs[] = {0x4C00, 0x4C01, 0x4C02, 0x4C03, 0x4C04, 0x4C05,
                      0x4C10, 0x4C11, 0x4C20, 0x4C21, 0x4C40, 0x4C41};
        int i;
        for (i = 0; i < sizeof(regs)/sizeof(regs[0]); i++) {
            val = rreg(regs[i]);
            pr_info("direct_probe: reg[0x%04x] = 0x%08x\n", regs[i], val);
        }
    }

    /* NOW: the big question — can we WRITE to IC_BASE in this state? */
    pr_info("direct_probe: === WRITE TEST: MES_IC_BASE ===\n");
    val = rreg(regCP_MES_IC_BASE_LO);
    pr_info("direct_probe: MES_IC_BASE_LO BEFORE = 0x%08x\n", val);

    writel(0xDEADBEEF, mmio + (regCP_MES_IC_BASE_LO * 4));
    mb();
    udelay(100);

    val = rreg(regCP_MES_IC_BASE_LO);
    pr_info("direct_probe: MES_IC_BASE_LO AFTER  = 0x%08x\n", val);

    if (val == 0xDEADBEEF)
        pr_info("direct_probe: *** IC_BASE IS WRITABLE IN DIRECT MODE! ***\n");
    else
        pr_info("direct_probe: IC_BASE still locked (read back 0x%08x)\n", val);

    /* Also try writing to RLC_CNTL to see if we can start RLC */
    val = rreg(regRLC_CNTL);
    pr_info("direct_probe: RLC_CNTL BEFORE = 0x%08x\n", val);

    /* Try setting RLC_ENABLE bit (bit 0) */
    writel(val | 0x1, mmio + (regRLC_CNTL * 4));
    mb();
    udelay(1000);

    val = rreg(regRLC_CNTL);
    pr_info("direct_probe: RLC_CNTL AFTER  = 0x%08x (tried to set bit 0)\n", val);

    pr_info("direct_probe: === PROBE COMPLETE ===\n");

    iounmap(mmio);
    pci_dev_put(pdev);

    return -EAGAIN; /* don't stay loaded */
}

static void __exit direct_probe_exit(void) {}
module_init(direct_probe_init);
module_exit(direct_probe_exit);
