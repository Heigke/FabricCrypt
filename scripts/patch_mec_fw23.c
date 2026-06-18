/*
 * patch_mec_fw23.c — Phase 23: MEC halt/unhalt + IC_BASE writability test
 *
 * Phase 22 crashed on BO traversal. This phase skips all BO walking and
 * focuses purely on register-level control:
 *
 *   1. Read MEC state (PC, MEC_CNTL, IC_BASE)
 *   2. Halt MEC (ME1+ME2)
 *   3. Verify halt stuck (PC frozen)
 *   4. Read IC_BASE while halted
 *   5. Try writing a DIFFERENT value to IC_BASE_LO, read back
 *   6. Restore original IC_BASE
 *   7. Unhalt MEC
 *   8. Verify PC resumes
 *
 * Also: try reading MEC SRAM via CP_MEC_ME1_UCODE_ADDR/DATA while halted.
 * If we can read SRAM, we have the decrypted firmware.
 * If IC_BASE is writable, we can point MEC at a custom firmware BO.
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
static void wr(u32 dw_off, u32 val) {
	writel(val, mmio + (u64)dw_off * 4);
	readl(mmio + (u64)dw_off * 4); /* flush */
}

/* MEC control */
#define regCP_MEC1_INSTR_PNTR      0x021A8
#define regCP_MEC2_INSTR_PNTR      0x021AA
#define regCP_MEC_CNTL             0x0A802
#define MEC_ME1_HALT               (1 << 30)
#define MEC_ME2_HALT               (1 << 28)

/* IC_BASE — instruction cache base address */
#define regCP_CPC_IC_BASE_LO       0x0C930
#define regCP_CPC_IC_BASE_HI       0x0C931

/* MEC SRAM read-back registers (GFX11) */
#define regCP_MEC_ME1_UCODE_ADDR   0x0A814
#define regCP_MEC_ME1_UCODE_DATA   0x0A815
#define regCP_MEC_ME2_UCODE_ADDR   0x0A816
#define regCP_MEC_ME2_UCODE_DATA   0x0A817

/* Alternative SRAM access (some GFX generations) */
#define regCP_MEC_ISA_CNTL         0x0A81E
#define regCP_MEC_ISA_ADDR         0x0A81F
#define regCP_MEC_ISA_DATA         0x0A820

/* CP_CPC_IC_OP_CNTL — cache operations */
#define regCP_CPC_IC_OP_CNTL       0x0C932

/* PRIME/INVALIDATE bits for IC_OP_CNTL */
#define IC_INVALIDATE_CACHE        (1 << 0)
#define IC_PRIME_ICACHE            (1 << 4)

static int __init fw23_init(void)
{
	struct pci_dev *pdev = NULL;
	u32 pc1, pc2, mec_cntl;
	u32 ic_lo, ic_hi;
	u32 ic_lo_new, ic_hi_new;
	int i;

	pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
	if (!pdev)
		return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	pr_info("fw23: ========================================\n");
	pr_info("fw23: PHASE 23: MEC HALT + IC_BASE + SRAM READ\n");
	pr_info("fw23: ========================================\n");

	/* Step 1: Read initial state */
	pc1 = rr(regCP_MEC1_INSTR_PNTR);
	pc2 = rr(regCP_MEC2_INSTR_PNTR);
	mec_cntl = rr(regCP_MEC_CNTL);
	ic_lo = rr(regCP_CPC_IC_BASE_LO);
	ic_hi = rr(regCP_CPC_IC_BASE_HI);

	pr_info("fw23: === INITIAL STATE ===\n");
	pr_info("fw23: MEC1_PC=0x%04X MEC2_PC=0x%04X\n", pc1, pc2);
	pr_info("fw23: MEC_CNTL=0x%08X\n", mec_cntl);
	pr_info("fw23: IC_BASE: LO=0x%08X HI=0x%08X\n", ic_lo, ic_hi);

	/* Step 2: Halt MEC */
	pr_info("fw23: === HALTING MEC ===\n");
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_ME2_HALT);
	udelay(100); /* let halt take effect */

	mec_cntl = rr(regCP_MEC_CNTL);
	pc1 = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("fw23: After halt: MEC_CNTL=0x%08X PC=0x%04X\n", mec_cntl, pc1);

	/* Verify halt by reading PC twice with delay */
	udelay(100);
	pc2 = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("fw23: Halt verify: PC1=0x%04X PC2=0x%04X %s\n",
		pc1, pc2, (pc1 == pc2) ? "FROZEN" : "STILL RUNNING!");

	/* Step 3: Read IC_BASE while halted */
	ic_lo = rr(regCP_CPC_IC_BASE_LO);
	ic_hi = rr(regCP_CPC_IC_BASE_HI);
	pr_info("fw23: Halted IC_BASE: LO=0x%08X HI=0x%08X\n", ic_lo, ic_hi);
	pr_info("fw23: IC_BASE full addr = 0x%02X_%08X\n", ic_hi, ic_lo);

	/* Step 4: Try writing IC_BASE_LO to a test value */
	pr_info("fw23: === IC_BASE WRITE TEST ===\n");

	/* Write a distinct test pattern */
	wr(regCP_CPC_IC_BASE_LO, 0xDEAD0000);
	ic_lo_new = rr(regCP_CPC_IC_BASE_LO);
	pr_info("fw23: Wrote 0xDEAD0000, read back: 0x%08X %s\n",
		ic_lo_new,
		(ic_lo_new == 0xDEAD0000) ? "WRITABLE!" :
		(ic_lo_new == ic_lo) ? "READ-ONLY" : "PARTIAL");

	/* Also try HI */
	wr(regCP_CPC_IC_BASE_HI, 0x42);
	ic_hi_new = rr(regCP_CPC_IC_BASE_HI);
	pr_info("fw23: Wrote HI=0x42, read back: 0x%08X %s\n",
		ic_hi_new,
		(ic_hi_new == 0x42) ? "WRITABLE!" :
		(ic_hi_new == ic_hi) ? "READ-ONLY" : "PARTIAL");

	/* Restore original IC_BASE */
	wr(regCP_CPC_IC_BASE_LO, ic_lo);
	wr(regCP_CPC_IC_BASE_HI, ic_hi);
	pr_info("fw23: Restored IC_BASE: LO=0x%08X HI=0x%08X\n",
		rr(regCP_CPC_IC_BASE_LO), rr(regCP_CPC_IC_BASE_HI));

	/* Step 5: Try reading MEC SRAM via ucode addr/data registers */
	pr_info("fw23: === MEC SRAM READ TEST ===\n");

	/* Method A: CP_MEC_ME1_UCODE_ADDR/DATA */
	wr(regCP_MEC_ME1_UCODE_ADDR, 0x00000000);
	udelay(10);
	pr_info("fw23: SRAM Method A (UCODE_ADDR/DATA):\n");
	for (i = 0; i < 16; i++) {
		u32 val;
		wr(regCP_MEC_ME1_UCODE_ADDR, i);
		udelay(10);
		val = rr(regCP_MEC_ME1_UCODE_DATA);
		pr_info("fw23:   SRAM[0x%03X] = 0x%08X%s%s\n", i, val,
			(val == 0xC424000BUL) ? " <<< FIRST_INSTR!" : "",
			(val == 0x88000000UL) ? " <<< BRANCH_SELF!" : "");
	}

	/* Read around the stuck PC (0x44C) */
	pr_info("fw23: SRAM around PC=0x44C:\n");
	for (i = 0; i < 8; i++) {
		u32 val;
		wr(regCP_MEC_ME1_UCODE_ADDR, 0x44A + i);
		udelay(10);
		val = rr(regCP_MEC_ME1_UCODE_DATA);
		pr_info("fw23:   SRAM[0x%03X] = 0x%08X%s%s%s\n",
			0x44A + i, val,
			(val == 0x88000000UL) ? " <<< BRANCH_SELF!" : "",
			(val == 0xBF800000UL) ? " <<< NOP" : "",
			(val == 0xD8000705UL) ? " <<< WAIT_MEM" : "");
	}

	/* Method B: ISA registers (may not exist on all gens) */
	pr_info("fw23: SRAM Method B (ISA_ADDR/DATA):\n");
	wr(regCP_MEC_ISA_ADDR, 0x00000000);
	udelay(10);
	for (i = 0; i < 4; i++) {
		u32 val;
		wr(regCP_MEC_ISA_ADDR, i);
		udelay(10);
		val = rr(regCP_MEC_ISA_DATA);
		pr_info("fw23:   ISA[0x%03X] = 0x%08X\n", i, val);
	}

	/* Step 6: Try IC_OP_CNTL — read current state */
	{
		u32 ic_op = rr(regCP_CPC_IC_OP_CNTL);
		pr_info("fw23: IC_OP_CNTL = 0x%08X\n", ic_op);
	}

	/* Step 7: Unhalt MEC */
	pr_info("fw23: === UNHALTING MEC ===\n");
	wr(regCP_MEC_CNTL, 0);
	udelay(100);

	mec_cntl = rr(regCP_MEC_CNTL);
	pc1 = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("fw23: After unhalt: MEC_CNTL=0x%08X PC=0x%04X\n", mec_cntl, pc1);

	/* Verify running by reading PC twice */
	udelay(100);
	pc2 = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("fw23: Resume verify: PC1=0x%04X PC2=0x%04X %s\n",
		pc1, pc2,
		(pc1 == pc2) ? "STILL_STUCK (branch-self)" : "RUNNING");

	pr_info("fw23: ========================================\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw23_exit(void) {}

module_init(fw23_init);
module_exit(fw23_exit);
