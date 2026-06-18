/*
 * patch_mec_fw7.c — Phase 7: Reproduce Phase 3 Test B two-step sequence
 *
 * Phase 3 Test B was the ONLY test that moved PC from 0x44C→0x44A.
 * It was preceded by Test A (bare IC invalidate while MEC running).
 * This module tests the theory that two-phase invalidation is key:
 *   Step 1: IC invalidate bit 27 while MEC is RUNNING (not halted)
 *   Step 2: Halt + pipe reset + IC invalidate
 *
 * Also tests: patching ALL 5 firmware copies (not just copy[0]),
 * and GART-based firmware location via DC_BASE.
 *
 * Auto-unloads (-ENODEV).
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/mm.h>
#include <linux/highmem.h>
#include <linux/delay.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002

/* Legacy registers */
#define regCP_MEC_CNTL              0x0A802
#define regCP_MEC1_INSTR_PNTR      0x021A8
#define regCP_CPC_IC_OP_CNTL       0x0C97A
#define regCP_MEC_DC_OP_CNTL       0x0C90C
#define regCP_MEC_DC_BASE_LO       0x0F870
#define regCP_MEC_DC_BASE_HI       0x0F871

/* CP_MEC_CNTL bits */
#define MEC_ME1_PIPE0_RESET   (1 << 16)
#define MEC_ME1_PIPE1_RESET   (1 << 17)
#define MEC_ME1_PIPE2_RESET   (1 << 18)
#define MEC_ME1_PIPE3_RESET   (1 << 19)
#define MEC_INVALIDATE_ICACHE (1 << 27)
#define MEC_ME1_HALT          (1 << 30)
#define ALL_PIPE_RESET (MEC_ME1_PIPE0_RESET | MEC_ME1_PIPE1_RESET | \
			MEC_ME1_PIPE2_RESET | MEC_ME1_PIPE3_RESET)

static void __iomem *mmio;

static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }
static void wr(u32 dw_off, u32 val) {
	writel(val, mmio + (u64)dw_off * 4);
	readl(mmio + (u64)dw_off * 4);
}

/* All 5 firmware copies found by find_mec_fw */
static const u64 fw_phys[] = {
	0x115c8c100ULL,
	0x13420c200ULL,
	0x28763f200ULL,
	0x28fc45200ULL,
	0x2b17c5200ULL,
};
#define N_COPIES 5

static u32 orig_44c[N_COPIES];
static u32 orig_44a[N_COPIES];

static u32 *map_fw_dword(u64 base_phys, int dw_off)
{
	u64 phys = base_phys + (u64)dw_off * 4;
	unsigned long pfn = phys >> PAGE_SHIFT;
	unsigned int page_off = phys & ~PAGE_MASK;
	struct page *page;
	u8 *vaddr;

	if (!pfn_valid(pfn))
		return NULL;
	page = pfn_to_page(pfn);
	vaddr = kmap_local_page(page);
	if (!vaddr)
		return NULL;
	return (u32 *)(vaddr + page_off);
}

static void unmap_fw(void *ptr)
{
	kunmap_local((void *)((unsigned long)ptr & PAGE_MASK));
}

static u32 read_fw(u64 base, int dw_off)
{
	u32 val = 0;
	u32 *ptr = map_fw_dword(base, dw_off);
	if (ptr) { val = *ptr; unmap_fw(ptr); }
	return val;
}

static void write_fw_flush(u64 base, int dw_off, u32 val)
{
	u32 *ptr = map_fw_dword(base, dw_off);
	if (ptr) {
		*ptr = val;
		wmb();
		clflush(ptr);
		wmb();
		unmap_fw(ptr);
	}
}

static void patch_all_copies(int dw_off, u32 val)
{
	int i;
	for (i = 0; i < N_COPIES; i++)
		write_fw_flush(fw_phys[i], dw_off, val);
}

static void sample_pc(const char *label, int count)
{
	int j;
	for (j = 0; j < count; j++) {
		mdelay(5);
		pr_info("fw7: %s PC[%d]=0x%04X\n",
			label, j, rr(regCP_MEC1_INSTR_PNTR));
	}
}

static int __init fw7_init(void)
{
	struct pci_dev *pdev = NULL;
	int i;

	while ((pdev = pci_get_device(AMD_VENDOR_ID, PCI_ANY_ID, pdev))) {
		if ((pdev->class >> 8) == PCI_CLASS_DISPLAY_OTHER)
			break;
	}
	if (!pdev) {
		pr_err("fw7: no AMD GPU\n");
		return -ENODEV;
	}
	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENOMEM;
	}

	pr_info("fw7: ========================================\n");
	pr_info("fw7: PHASE 7: TWO-PHASE IC INVALIDATION\n");
	pr_info("fw7: ========================================\n");

	pr_info("fw7: BASELINE: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));

	/* Save original instructions from all copies */
	for (i = 0; i < N_COPIES; i++) {
		orig_44c[i] = read_fw(fw_phys[i], 0x44C);
		orig_44a[i] = read_fw(fw_phys[i], 0x44A);
		pr_info("fw7: copy[%d] [0x44A]=0x%08X [0x44C]=0x%08X\n",
			i, orig_44a[i], orig_44c[i]);
	}

	/* ============================================================
	 * TEST A: Exact Phase 3 Test B replication
	 *
	 * Phase 3 ran A then B in sequence. A did bare IC invalidate
	 * (bit 27 only, no halt, no pipe reset). B then halted + pipe
	 * reset + bit 27. PC moved from 0x44C to 0x44A.
	 *
	 * We replicate this EXACTLY: bare IC invalidate first, then
	 * halt + patch + pipe reset + IC invalidate.
	 * ============================================================ */
	pr_info("fw7: --- TEST A: Phase 3 exact replication ---\n");

	/* A1: Bare IC invalidate while MEC is RUNNING (like P3 Test A) */
	pr_info("fw7: A1 MEC running, applying bare IC invalidate (bit 27)...\n");
	wr(regCP_MEC_CNTL, MEC_INVALIDATE_ICACHE);  /* ONLY bit 27, NO halt */
	udelay(1000);
	pr_info("fw7: A1 after: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));

	/* A2: Now halt */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	pr_info("fw7: A2 halted: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));

	/* A3: Patch 0x44C in ALL copies */
	patch_all_copies(0x44C, 0xBF80DEAD);
	pr_info("fw7: A3 patched ALL copies at 0x44C\n");

	/* Verify patches took */
	for (i = 0; i < N_COPIES; i++) {
		u32 rb = read_fw(fw_phys[i], 0x44C);
		pr_info("fw7: A3 copy[%d] readback=0x%08X %s\n",
			i, rb, rb == 0xBF80DEAD ? "OK" : "FAIL");
	}

	/* A4: Pipe reset + IC invalidate (like P3 Test B) */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);

	/* A5: Clear reset, keep halted */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	pr_info("fw7: A5 after reset: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* A6: Unhalt */
	wr(regCP_MEC_CNTL, 0);
	mdelay(20);
	sample_pc("A6", 8);

	/* RESTORE */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	for (i = 0; i < N_COPIES; i++)
		write_fw_flush(fw_phys[i], 0x44C, orig_44c[i]);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	wr(regCP_MEC_CNTL, 0);
	mdelay(50);
	pr_info("fw7: A RESTORED: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* ============================================================
	 * TEST B: Multiple IC invalidation pulses before pipe reset
	 * Maybe the hardware needs repeated invalidation signals
	 * ============================================================ */
	pr_info("fw7: --- TEST B: Multiple IC invalidation pulses ---\n");

	/* B1: Multiple IC invalidates while running */
	{
		int k;
		for (k = 0; k < 5; k++) {
			wr(regCP_MEC_CNTL, MEC_INVALIDATE_ICACHE);
			udelay(200);
		}
	}
	pr_info("fw7: B1 5x IC invalidate while running: PC=0x%04X\n",
		rr(regCP_MEC1_INSTR_PNTR));

	/* B2: Halt and patch ALL copies at BOTH 0x44A and 0x44C */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	patch_all_copies(0x44C, 0xBF80DEAD);
	patch_all_copies(0x44A, 0xBF80BEEF);
	pr_info("fw7: B2 patched ALL copies at 0x44A and 0x44C\n");

	/* B3: More IC invalidation pulses while halted */
	{
		int k;
		for (k = 0; k < 5; k++) {
			wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_INVALIDATE_ICACHE);
			udelay(200);
		}
	}

	/* B4: Pipe reset + IC invalidate */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	pr_info("fw7: B4 after reset: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* Also do CPC IC invalidation */
	wr(regCP_CPC_IC_OP_CNTL, 0x1); /* INVALIDATE_CACHE */
	udelay(2000);
	pr_info("fw7: B4 CPC_IC_OP=0x%08X\n", rr(regCP_CPC_IC_OP_CNTL));

	/* DC invalidation too */
	wr(regCP_MEC_DC_OP_CNTL, 0x1); /* INVALIDATE_DCACHE */
	udelay(2000);
	pr_info("fw7: B4 DC_OP=0x%08X\n", rr(regCP_MEC_DC_OP_CNTL));

	/* B5: Unhalt */
	wr(regCP_MEC_CNTL, 0);
	mdelay(20);
	sample_pc("B5", 8);

	/* RESTORE */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	for (i = 0; i < N_COPIES; i++) {
		write_fw_flush(fw_phys[i], 0x44C, orig_44c[i]);
		write_fw_flush(fw_phys[i], 0x44A, orig_44a[i]);
	}
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	wr(regCP_MEC_CNTL, 0);
	mdelay(50);
	pr_info("fw7: B RESTORED: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* ============================================================
	 * TEST C: wbinvd — flush entire CPU cache hierarchy
	 * The GPU might be snooping CPU caches. If our writes are stuck
	 * in L1/L2, the GPU never sees them. wbinvd forces all dirty
	 * lines to memory.
	 * ============================================================ */
	pr_info("fw7: --- TEST C: wbinvd + patch + invalidate ---\n");

	/* C1: IC invalidate while running */
	wr(regCP_MEC_CNTL, MEC_INVALIDATE_ICACHE);
	udelay(500);

	/* C2: Halt */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);

	/* C3: Patch all copies */
	patch_all_copies(0x44C, 0xBF80DEAD);

	/* C4: Full CPU cache flush */
	wbinvd();
	wmb();
	mb();
	pr_info("fw7: C4 wbinvd done\n");

	/* C5: IC + DC invalidation */
	wr(regCP_CPC_IC_OP_CNTL, 0x1);
	udelay(2000);
	wr(regCP_MEC_DC_OP_CNTL, 0x1);
	udelay(2000);

	/* C6: Pipe reset + IC invalidate */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	pr_info("fw7: C6 after reset: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* C7: Unhalt */
	wr(regCP_MEC_CNTL, 0);
	mdelay(20);
	sample_pc("C7", 8);

	/* RESTORE */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	for (i = 0; i < N_COPIES; i++)
		write_fw_flush(fw_phys[i], 0x44C, orig_44c[i]);
	wbinvd();
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	wr(regCP_MEC_CNTL, 0);
	mdelay(50);
	pr_info("fw7: C RESTORED: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* ============================================================
	 * TEST D: Probe GART/VRAM — find where GPU actually fetches FW
	 * Read DC_BASE address, try to find corresponding BAR mapping
	 * ============================================================ */
	pr_info("fw7: --- TEST D: GPU firmware address probe ---\n");
	{
		u32 dc_lo = rr(regCP_MEC_DC_BASE_LO);
		u32 dc_hi = rr(regCP_MEC_DC_BASE_HI);
		u64 dc_addr = ((u64)dc_hi << 32) | dc_lo;

		pr_info("fw7: D DC_BASE = 0x%016llX\n", dc_addr);
		pr_info("fw7: D Our FW_PHYS[0] = 0x%09llX\n", fw_phys[0]);

		/* Check if there are IC base registers nearby */
		/* Read some registers around DC_BASE area */
		pr_info("fw7: D MMIO[0x0F870]=0x%08X (DC_BASE_LO)\n", rr(0x0F870));
		pr_info("fw7: D MMIO[0x0F871]=0x%08X (DC_BASE_HI)\n", rr(0x0F871));
		pr_info("fw7: D MMIO[0x0F872]=0x%08X\n", rr(0x0F872));
		pr_info("fw7: D MMIO[0x0F873]=0x%08X\n", rr(0x0F873));
		pr_info("fw7: D MMIO[0x0F874]=0x%08X\n", rr(0x0F874));
		pr_info("fw7: D MMIO[0x0F875]=0x%08X\n", rr(0x0F875));

		/* Check IC base registers if they exist */
		/* gfx11: CP_MEC_MDBASE_LO/HI might be the instruction fetch base */
		pr_info("fw7: D MMIO[0x0C90D]=0x%08X\n", rr(0x0C90D));
		pr_info("fw7: D MMIO[0x0C90E]=0x%08X\n", rr(0x0C90E));
		pr_info("fw7: D MMIO[0x0C90F]=0x%08X\n", rr(0x0C90F));
		pr_info("fw7: D MMIO[0x0C910]=0x%08X\n", rr(0x0C910));

		/* Also check VRAM BAR */
		pr_info("fw7: D PCI BAR0 start=0x%llX size=0x%llX\n",
			(u64)pci_resource_start(pdev, 0),
			(u64)pci_resource_len(pdev, 0));
		pr_info("fw7: D PCI BAR2 start=0x%llX size=0x%llX\n",
			(u64)pci_resource_start(pdev, 2),
			(u64)pci_resource_len(pdev, 2));
		pr_info("fw7: D PCI BAR5 start=0x%llX size=0x%llX\n",
			(u64)pci_resource_start(pdev, 5),
			(u64)pci_resource_len(pdev, 5));
	}

	/* ============================================================
	 * TEST E: Try DC_BYPASS_ALL — disable data cache entirely
	 * If IC fetches through DC, bypassing DC forces direct mem reads
	 * ============================================================ */
	pr_info("fw7: --- TEST E: DC bypass + patch ---\n");

	/* E1: IC invalidate while running */
	wr(regCP_MEC_CNTL, MEC_INVALIDATE_ICACHE);
	udelay(500);

	/* E2: Halt */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);

	/* E3: Enable DC bypass */
	wr(regCP_MEC_DC_OP_CNTL, 0x4); /* BYPASS_ALL bit */
	udelay(100);
	pr_info("fw7: E3 DC bypass set: DC_OP=0x%08X\n", rr(regCP_MEC_DC_OP_CNTL));

	/* E4: Patch all copies */
	patch_all_copies(0x44C, 0xBF80DEAD);
	wbinvd();
	wmb();

	/* E5: IC invalidation */
	wr(regCP_CPC_IC_OP_CNTL, 0x1);
	udelay(2000);

	/* E6: Pipe reset + IC invalidate */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	pr_info("fw7: E6 after reset: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* E7: Unhalt (keep DC bypass active) */
	wr(regCP_MEC_CNTL, 0);
	mdelay(20);
	sample_pc("E7", 8);

	/* RESTORE — clear DC bypass first */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	wr(regCP_MEC_DC_OP_CNTL, 0x0); /* clear bypass */
	for (i = 0; i < N_COPIES; i++)
		write_fw_flush(fw_phys[i], 0x44C, orig_44c[i]);
	wbinvd();
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	wr(regCP_MEC_CNTL, 0);
	mdelay(50);
	pr_info("fw7: E RESTORED: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	pr_info("fw7: FINAL: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));
	pr_info("fw7: ========================================\n");
	pr_info("fw7: PHASE 7 COMPLETE\n");
	pr_info("fw7: ========================================\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw7_exit(void) {}
module_init(fw7_init);
module_exit(fw7_exit);
