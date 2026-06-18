/*
 * patch_mec_fw25.c — Phase 25: Blind SRAM write + physical address write
 *
 * Phase 24 confirmed SRAM reads return all zeros (PSP-gated on GFX11).
 * But writes might still work. Plan:
 *
 *   A. Try writing NOP to SRAM[0x44C] via UCODE_ADDR/DATA while halted
 *   B. Try writing via physical address (from Phase 15: 0x115C8C100)
 *   C. Try writing via GPU VRAM BAR (BAR0/BAR2)
 *
 * After each write attempt, unhalt MEC and check if PC moves past 0x44C.
 * If PC remains at 0x44C, the write didn't work.
 * If PC advances, we have a SRAM write path.
 *
 * Safety: branch_self (0x88000000) is a benign instruction — the MEC is
 * already stuck there. Worst case: writing fails silently and MEC
 * continues branch_self. NOP (0xBF800000) just means "do nothing, advance
 * to next instruction" — we'll see what the next instruction is.
 *
 * Actually, to be safe, write another branch_self first (same value) to
 * test writability, THEN try NOP.
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
	readl(mmio + (u64)dw_off * 4);
}

#define regCP_MEC1_INSTR_PNTR      0x021A8
#define regCP_MEC_CNTL             0x0A802
#define MEC_ME1_HALT               (1 << 30)
#define MEC_ME2_HALT               (1 << 28)

#define regCP_MEC_ME1_UCODE_ADDR   0x0A814
#define regCP_MEC_ME1_UCODE_DATA   0x0A815

#define regGRBM_GFX_CNTL           0x0D880

/* GFX11 MEC instructions */
#define INST_BRANCH_SELF 0x88000000  /* branch to self (offset 0) */
#define INST_NOP         0xBF800000  /* s_nop 0 */

static int __init fw25_init(void)
{
	struct pci_dev *pdev = NULL;
	u32 pc;
	int i;

	pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
	if (!pdev)
		return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	pr_info("fw25: ========================================\n");
	pr_info("fw25: PHASE 25: BLIND SRAM WRITE ATTEMPT\n");
	pr_info("fw25: ========================================\n");

	pc = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("fw25: Initial PC=0x%04X MEC_CNTL=0x%08X\n",
		pc, rr(regCP_MEC_CNTL));

	/* ========================================
	 * METHOD A: UCODE_ADDR/DATA write
	 * ======================================== */
	pr_info("fw25: === METHOD A: UCODE_ADDR/DATA ===\n");

	/* Halt MEC */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_ME2_HALT);
	udelay(200);
	pc = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("fw25: Halted, PC=0x%04X\n", pc);

	/* Select MEC1 pipe0 */
	wr(regGRBM_GFX_CNTL, 0x10); /* ME=1, PIPE=0 */
	udelay(100);

	/* Step 1: Write BRANCH_SELF to 0x44C (same instruction — safe test) */
	pr_info("fw25: Writing BRANCH_SELF to SRAM[0x44C] (safe test)...\n");
	wr(regCP_MEC_ME1_UCODE_ADDR, 0x44C);
	udelay(10);
	wr(regCP_MEC_ME1_UCODE_DATA, INST_BRANCH_SELF);
	udelay(10);

	/* Try to read back */
	wr(regCP_MEC_ME1_UCODE_ADDR, 0x44C);
	udelay(10);
	{
		u32 rb = rr(regCP_MEC_ME1_UCODE_DATA);
		pr_info("fw25: Readback SRAM[0x44C] = 0x%08X %s\n", rb,
			(rb == INST_BRANCH_SELF) ? "MATCH!" :
			(rb == 0) ? "still zero" : "DIFFERENT");
	}

	/* Step 2: Write NOP to 0x44C */
	pr_info("fw25: Writing NOP (0xBF800000) to SRAM[0x44C]...\n");
	wr(regCP_MEC_ME1_UCODE_ADDR, 0x44C);
	udelay(10);
	wr(regCP_MEC_ME1_UCODE_DATA, INST_NOP);
	udelay(10);

	/* Read back */
	wr(regCP_MEC_ME1_UCODE_ADDR, 0x44C);
	udelay(10);
	{
		u32 rb = rr(regCP_MEC_ME1_UCODE_DATA);
		pr_info("fw25: Readback SRAM[0x44C] = 0x%08X %s\n", rb,
			(rb == INST_NOP) ? "NOP WRITTEN!" :
			(rb == INST_BRANCH_SELF) ? "old value" :
			(rb == 0) ? "still zero" : "DIFFERENT");
	}

	/* Clear pipe select */
	wr(regGRBM_GFX_CNTL, 0);
	udelay(100);

	/* Unhalt and check if PC moves */
	pr_info("fw25: Unhalting MEC...\n");
	wr(regCP_MEC_CNTL, 0);
	udelay(500);

	/* Sample PC several times */
	for (i = 0; i < 5; i++) {
		udelay(200);
		pc = rr(regCP_MEC1_INSTR_PNTR);
		pr_info("fw25: PC sample %d: 0x%04X%s\n", i, pc,
			(pc == 0x44C) ? " (still stuck)" : " MOVED!");
	}

	/* If PC still at 0x44C, the NOP write didn't work via this method.
	 * Re-halt for method B. */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_ME2_HALT);
	udelay(200);

	/* ========================================
	 * METHOD B: Physical address write via ioremap
	 *
	 * Phase 15 found MEC SRAM base at physical address:
	 *   IC_BASE = 0x115C8C100 (from MMIO reads in earlier phases)
	 * But IC_BASE was actually 0x03_00000007 (config flags), not a PA.
	 *
	 * Try the GPU VRAM BAR instead. BAR0 on RDNA4 maps VRAM.
	 * The firmware BO's GPU VA was 0x97FF943000 (from adev+0x16808).
	 * But that's GPU VA, not physical. We need the BAR base + offset.
	 * ======================================== */
	pr_info("fw25: === METHOD B: VRAM BAR exploration ===\n");
	{
		resource_size_t bar0_start, bar0_len;
		resource_size_t bar2_start, bar2_len;
		resource_size_t bar5_start, bar5_len;

		bar0_start = pci_resource_start(pdev, 0);
		bar0_len = pci_resource_len(pdev, 0);
		bar2_start = pci_resource_start(pdev, 2);
		bar2_len = pci_resource_len(pdev, 2);
		bar5_start = pci_resource_start(pdev, 5);
		bar5_len = pci_resource_len(pdev, 5);

		pr_info("fw25: BAR0: phys=0x%llX len=0x%llX (%llu MB)\n",
			(u64)bar0_start, (u64)bar0_len,
			(u64)bar0_len / (1024*1024));
		pr_info("fw25: BAR2: phys=0x%llX len=0x%llX (%llu MB)\n",
			(u64)bar2_start, (u64)bar2_len,
			(u64)bar2_len / (1024*1024));
		pr_info("fw25: BAR5: phys=0x%llX len=0x%llX (%llu MB) [MMIO]\n",
			(u64)bar5_start, (u64)bar5_len,
			(u64)bar5_len / (1024*1024));

		/* BAR0 is typically VRAM. If the firmware BO is at a known
		 * VRAM offset, we could map it. But we'd need to know the
		 * mapping from GPU VA 0x97FF943000 to VRAM offset.
		 *
		 * Try mapping the start of BAR0 and reading a few bytes to
		 * see if we can identify firmware content. */
		if (bar0_len > 0) {
			void __iomem *vram = ioremap(bar0_start, min_t(u64, bar0_len, 4096));
			if (vram) {
				u32 v0 = readl(vram);
				u32 v1 = readl(vram + 4);
				pr_info("fw25: VRAM[0x000] = 0x%08X\n", v0);
				pr_info("fw25: VRAM[0x004] = 0x%08X\n", v1);
				iounmap(vram);
			} else {
				pr_info("fw25: Failed to ioremap BAR0\n");
			}
		}

		/* BAR2 might be the "doorbell" BAR or extended VRAM */
		if (bar2_len > 0) {
			void __iomem *bar2 = ioremap(bar2_start, min_t(u64, bar2_len, 4096));
			if (bar2) {
				u32 v0 = readl(bar2);
				pr_info("fw25: BAR2[0x000] = 0x%08X\n", v0);
				iounmap(bar2);
			}
		}
	}

	/* ========================================
	 * METHOD C: Try writing via different register sequences
	 *
	 * In some AMD GPUs, firmware load uses:
	 *   1. Write start address to UCODE_ADDR
	 *   2. Write data words to UCODE_DATA (auto-increment)
	 * But the address might need a "load enable" bit.
	 *
	 * Check if CP_MEC_ME1_UCODE_ADDR has mode bits.
	 * ======================================== */
	pr_info("fw25: === METHOD C: UCODE_ADDR mode bits ===\n");
	{
		/* Try setting bit 30 or 31 in UCODE_ADDR (common "load" bits) */
		u32 addrs[] = {
			0x0000044C,            /* plain address */
			0x4000044C,            /* bit 30 set */
			0x8000044C,            /* bit 31 set */
			0xC000044C,            /* bits 30+31 set */
		};
		int a;

		wr(regGRBM_GFX_CNTL, 0x10); /* ME=1 PIPE=0 */
		udelay(100);

		for (a = 0; a < 4; a++) {
			u32 rb;
			pr_info("fw25: Trying UCODE_ADDR = 0x%08X\n", addrs[a]);
			wr(regCP_MEC_ME1_UCODE_ADDR, addrs[a]);
			udelay(10);
			wr(regCP_MEC_ME1_UCODE_DATA, INST_NOP);
			udelay(10);

			/* Read back with same address mode */
			wr(regCP_MEC_ME1_UCODE_ADDR, addrs[a]);
			udelay(10);
			rb = rr(regCP_MEC_ME1_UCODE_DATA);
			pr_info("fw25:   Readback = 0x%08X %s\n", rb,
				(rb == INST_NOP) ? "SUCCESS!" :
				(rb != 0) ? "NON-ZERO" : "zero");
		}

		wr(regGRBM_GFX_CNTL, 0);
	}

	/* Final: unhalt MEC and check PC one more time */
	pr_info("fw25: Final unhalt...\n");
	wr(regCP_MEC_CNTL, 0);
	udelay(1000);
	for (i = 0; i < 3; i++) {
		udelay(500);
		pc = rr(regCP_MEC1_INSTR_PNTR);
		pr_info("fw25: Final PC sample %d: 0x%04X%s\n", i, pc,
			(pc == 0x44C) ? " stuck" : " MOVED!");
	}

	pr_info("fw25: ========================================\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw25_exit(void) {}

module_init(fw25_init);
module_exit(fw25_exit);
