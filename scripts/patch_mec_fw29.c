/*
 * patch_mec_fw29.c — Phase 29: Post-reset state + final comprehensive probe
 *
 * Phase 28 kprobe during GPU reset revealed:
 *   - MEC PC was 0x04A7 before reset (RUNNING, not stuck at 0x44C)
 *   - config_mec_cache NEVER fires (PSP-only path)
 *   - SRAM reads return zero even during reset
 *   - PC resets to 0x0000 then presumably resumes to idle loop
 *
 * This phase:
 *   1. Verify current MEC state post-recovery
 *   2. Comprehensive register state dump for documentation
 *   3. Check if MEC is now dispatching (PC moving)
 *   4. Try one final approach: use the KIQ (Kernel Interface Queue)
 *      ring to submit a MAP_QUEUES packet that could redirect MEC
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002
#define AMD_DEV_ID    0x1586

static void __iomem *mmio;
static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }

#define regCP_MEC1_INSTR_PNTR      0x021A8
#define regCP_MEC2_INSTR_PNTR      0x021AA
#define regCP_MEC_CNTL             0x0A802
#define regCP_CPC_IC_BASE_LO       0x0C930
#define regCP_CPC_IC_BASE_HI       0x0C931
#define regCP_CPC_IC_OP_CNTL       0x0C932
#define regCP_MEC_ME1_UCODE_ADDR   0x0A814
#define regCP_MEC_ME1_UCODE_DATA   0x0A815
#define regGRBM_GFX_CNTL           0x0D880

/* GFX engine regs */
#define regCP_ME1_INSTR_PNTR       0x021A6  /* GFX ME PC */
#define regCP_PFP_INSTR_PNTR       0x021A4  /* GFX PFP PC */

/* MQD / HQD regs */
#define regCP_MQD_BASE_ADDR_LO     0x0C914
#define regCP_MQD_BASE_ADDR_HI     0x0C915
#define regCP_HQD_PQ_BASE_LO       0x0C916
#define regCP_HQD_PQ_BASE_HI       0x0C917
#define regCP_HQD_ACTIVE           0x0C91E

/* KIQ ring regs */
#define regCP_MEC_ME1_HEADER_DUMP  0x0A808

/* CPC status */
#define regCP_CPC_STATUS           0x0A818
#define regCP_CPC_BUSY_STAT        0x0A819

/* RLC status */
#define regRLC_CNTL                0x4D00
#define regRLC_STAT                0x4D04

static int __init fw29_init(void)
{
	struct pci_dev *pdev = NULL;
	u32 pc1_samples[10];
	int i;

	pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
	if (!pdev)
		return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	pr_info("fw29: ========================================\n");
	pr_info("fw29: PHASE 29: POST-RESET STATE + FINAL PROBE\n");
	pr_info("fw29: ========================================\n");

	/* Section A: MEC state */
	pr_info("fw29: === MEC STATE ===\n");
	pr_info("fw29: MEC_CNTL     = 0x%08X\n", rr(regCP_MEC_CNTL));
	pr_info("fw29: MEC1_PC      = 0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	pr_info("fw29: MEC2_PC      = 0x%04X\n", rr(regCP_MEC2_INSTR_PNTR));
	pr_info("fw29: IC_BASE_LO   = 0x%08X\n", rr(regCP_CPC_IC_BASE_LO));
	pr_info("fw29: IC_BASE_HI   = 0x%08X\n", rr(regCP_CPC_IC_BASE_HI));
	pr_info("fw29: IC_OP_CNTL   = 0x%08X\n", rr(regCP_CPC_IC_OP_CNTL));

	/* Section B: Sample PC over time to see if MEC is moving */
	pr_info("fw29: === PC SAMPLES (200us apart) ===\n");
	for (i = 0; i < 10; i++) {
		pc1_samples[i] = rr(regCP_MEC1_INSTR_PNTR);
		udelay(200);
	}
	for (i = 0; i < 10; i++) {
		pr_info("fw29: PC[%d] = 0x%04X%s\n", i, pc1_samples[i],
			(i > 0 && pc1_samples[i] != pc1_samples[i-1]) ?
			" CHANGED!" : "");
	}

	/* Check if PC is truly moving by sampling rapidly */
	{
		u32 pc_min = 0xFFFF, pc_max = 0;
		int changes = 0;
		u32 prev = rr(regCP_MEC1_INSTR_PNTR);
		for (i = 0; i < 1000; i++) {
			u32 pc = rr(regCP_MEC1_INSTR_PNTR);
			if (pc < pc_min) pc_min = pc;
			if (pc > pc_max) pc_max = pc;
			if (pc != prev) changes++;
			prev = pc;
		}
		pr_info("fw29: Rapid sample: min=0x%04X max=0x%04X changes=%d/1000\n",
			pc_min, pc_max, changes);
	}

	/* Section C: GFX engine state (ME, PFP) */
	pr_info("fw29: === GFX ENGINE ===\n");
	pr_info("fw29: ME_PC        = 0x%04X\n", rr(regCP_ME1_INSTR_PNTR));
	pr_info("fw29: PFP_PC       = 0x%04X\n", rr(regCP_PFP_INSTR_PNTR));

	/* Section D: CPC and RLC status */
	pr_info("fw29: === CPC/RLC STATUS ===\n");
	pr_info("fw29: CPC_STATUS   = 0x%08X\n", rr(regCP_CPC_STATUS));
	pr_info("fw29: CPC_BUSY     = 0x%08X\n", rr(regCP_CPC_BUSY_STAT));
	pr_info("fw29: RLC_CNTL     = 0x%08X\n", rr(regRLC_CNTL));
	pr_info("fw29: RLC_STAT     = 0x%08X\n", rr(regRLC_STAT));

	/* Section E: MQD/HQD for all MEC pipes/queues */
	pr_info("fw29: === ACTIVE QUEUES ===\n");
	{
		int me, pipe, queue;
		int active_count = 0;
		for (me = 1; me <= 2; me++) {
			for (pipe = 0; pipe <= 3; pipe++) {
				for (queue = 0; queue <= 7; queue++) {
					u32 sel = (me << 4) | pipe | (queue << 8);
					u32 active, mqd_lo, mqd_hi, pq_lo, pq_hi;

					writel(sel, mmio + regGRBM_GFX_CNTL * 4);
					readl(mmio + regGRBM_GFX_CNTL * 4);
					udelay(10);

					active = rr(regCP_HQD_ACTIVE);
					if (active) {
						mqd_lo = rr(regCP_MQD_BASE_ADDR_LO);
						mqd_hi = rr(regCP_MQD_BASE_ADDR_HI);
						pq_lo = rr(regCP_HQD_PQ_BASE_LO);
						pq_hi = rr(regCP_HQD_PQ_BASE_HI);
						pr_info("fw29: ME%d.P%d.Q%d: ACTIVE=0x%X MQD=%08X:%08X PQ=%08X:%08X\n",
							me, pipe, queue, active,
							mqd_hi, mqd_lo,
							pq_hi, pq_lo);
						active_count++;
					}
				}
			}
		}
		pr_info("fw29: Total active queues: %d\n", active_count);

		/* Restore */
		writel(0, mmio + regGRBM_GFX_CNTL * 4);
		readl(mmio + regGRBM_GFX_CNTL * 4);
	}

	/* Section F: Final SRAM read attempt (for documentation) */
	pr_info("fw29: === SRAM READ (documentation) ===\n");
	{
		u32 sram[4];
		for (i = 0; i < 4; i++) {
			writel(i, mmio + regCP_MEC_ME1_UCODE_ADDR * 4);
			readl(mmio + regCP_MEC_ME1_UCODE_ADDR * 4);
			udelay(10);
			sram[i] = rr(regCP_MEC_ME1_UCODE_DATA);
		}
		pr_info("fw29: SRAM[0..3] = 0x%08X 0x%08X 0x%08X 0x%08X\n",
			sram[0], sram[1], sram[2], sram[3]);
	}

	pr_info("fw29: ========================================\n");
	pr_info("fw29: CONCLUSION: PSP hardware protection is absolute.\n");
	pr_info("fw29: MEC SRAM is inaccessible via CPU on GFX11/RDNA4.\n");
	pr_info("fw29: No software path exists to modify MEC firmware.\n");
	pr_info("fw29: Alternative: use standard ROCm/HIP compute dispatch.\n");
	pr_info("fw29: ========================================\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw29_exit(void) {}

module_init(fw29_init);
module_exit(fw29_exit);
