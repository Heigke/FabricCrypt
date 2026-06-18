/*
 * patch_mec_fw5.c — Phase 5: Reproduce Phase 3 result & cache coherency tests
 *
 * Phase 3 Test B got PC=0x44A (IC invalidated). Phase 4 couldn't reproduce.
 * Hypothesis: Phase 3's Test A invalidated IC (bit 27), then Test B's pipe
 * reset forced re-fetch from RAM. Two-step sequence matters.
 *
 * Also tests: clflush for CPU→RAM coherency, separate IC inval + pipe reset.
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
#include <asm/cacheflush.h>
#include <asm/special_insns.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002

#define regCP_MEC_CNTL              0x0A802
#define regCP_CPC_IC_OP_CNTL       0x0C97A
#define regCP_MEC1_INSTR_PNTR      0x021A8

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

#define FW_PHYS  0x115c8c100ULL
#define PATCH_DW_OFF    0x44C
#define MARKER_INSTR    0xBF80DEADUL

static u32 orig_44a, orig_44c;

static u32 *map_fw_dword(int dw_off)
{
	u64 phys = FW_PHYS + (u64)dw_off * 4;
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

static void write_and_flush(int dw_off, u32 val)
{
	u32 *ptr = map_fw_dword(dw_off);
	if (ptr) {
		*ptr = val;
		/* Ensure write hits DRAM, not just CPU cache */
		clflush(ptr);
		wmb();
		mb();
		unmap_fw(ptr);
	}
}

static u32 read_fw(int dw_off)
{
	u32 val = 0;
	u32 *ptr = map_fw_dword(dw_off);
	if (ptr) {
		clflush(ptr);
		mb();
		val = *ptr;
		unmap_fw(ptr);
	}
	return val;
}

static void restore_and_flush(void)
{
	write_and_flush(0x44A, orig_44a);
	write_and_flush(0x44C, orig_44c);
}

static void sample_pc(const char *label, int count)
{
	int j;
	for (j = 0; j < count; j++) {
		mdelay(5);
		pr_info("p5: %s PC[%d]=0x%04X\n", label, j, rr(regCP_MEC1_INSTR_PNTR));
	}
}

static int __init patch5_init(void)
{
	struct pci_dev *pdev = NULL;

	while ((pdev = pci_get_device(AMD_VENDOR_ID, PCI_ANY_ID, pdev))) {
		if ((pdev->class >> 8) == PCI_CLASS_DISPLAY_OTHER)
			break;
	}
	if (!pdev) {
		pr_err("p5: no AMD GPU\n");
		return -ENODEV;
	}
	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENOMEM;
	}

	pr_info("p5: === PHASE 5: IC INVALIDATION + CACHE COHERENCY ===\n");
	pr_info("p5: BASELINE: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));

	/* Save originals */
	orig_44a = read_fw(0x44A);
	orig_44c = read_fw(0x44C);
	pr_info("p5: orig[0x44A]=0x%08X orig[0x44C]=0x%08X\n", orig_44a, orig_44c);

	/* ============================================================
	 * TEST 1: Exact Phase 3 reproduction
	 * Step A: Halt, patch, bit 27, unhalt (no pipe reset)
	 * Step B: Halt, patch, pipe reset + bit 27, unhalt
	 * The IC invalidation from step A should persist into step B
	 * ============================================================ */
	pr_info("p5: --- TEST 1: Phase 3 reproduction (2-step IC inval) ---\n");

	/* Step A: IC invalidate without pipe reset */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);

	write_and_flush(PATCH_DW_OFF, MARKER_INSTR);
	pr_info("p5: 1A patched [0x44C]=0x%08X\n", read_fw(PATCH_DW_OFF));

	/* Set IC invalidate while halted */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_INVALIDATE_ICACHE);
	udelay(1000);
	pr_info("p5: 1A MEC_CNTL after IC_INVAL=0x%08X\n", rr(regCP_MEC_CNTL));

	/* Clear IC invalidate, unhalt */
	wr(regCP_MEC_CNTL, 0);
	mdelay(10);
	pr_info("p5: 1A after unhalt: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	sample_pc("1A", 3);

	/* Restore firmware */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	restore_and_flush();
	wr(regCP_MEC_CNTL, 0);
	mdelay(10);

	/* Step B: NOW pipe reset (IC should already be invalidated from step A) */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);

	write_and_flush(PATCH_DW_OFF, MARKER_INSTR);
	pr_info("p5: 1B patched [0x44C]=0x%08X\n", read_fw(PATCH_DW_OFF));

	/* Pipe reset + IC invalidate */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	pr_info("p5: 1B after reset: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	wr(regCP_MEC_CNTL, 0);
	mdelay(20);
	pr_info("p5: 1B after unhalt: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	sample_pc("1B", 5);

	/* Restore */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	restore_and_flush();
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	wr(regCP_MEC_CNTL, 0);
	mdelay(50);
	pr_info("p5: 1 RESTORED: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* ============================================================
	 * TEST 2: wbinvd + clflush + IC invalidate + pipe reset
	 * Nuclear option: flush ALL CPU caches to RAM first
	 * ============================================================ */
	pr_info("p5: --- TEST 2: wbinvd + clflush nuclear flush ---\n");

	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);

	write_and_flush(PATCH_DW_OFF, MARKER_INSTR);

	/* Nuclear CPU cache flush */
	wbinvd();
	mb();

	pr_info("p5: 2 patched+flushed [0x44C]=0x%08X\n", read_fw(PATCH_DW_OFF));

	/* IC invalidate first (separate step) */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_INVALIDATE_ICACHE);
	mdelay(5);

	/* Then pipe reset */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET);
	mdelay(5);

	/* Clear reset */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	pr_info("p5: 2 after reset: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* Unhalt */
	wr(regCP_MEC_CNTL, 0);
	mdelay(20);
	pr_info("p5: 2 after unhalt: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	sample_pc("2", 5);

	/* Restore */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	restore_and_flush();
	wbinvd();
	mb();
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_INVALIDATE_ICACHE);
	mdelay(5);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET);
	mdelay(5);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	wr(regCP_MEC_CNTL, 0);
	mdelay(50);
	pr_info("p5: 2 RESTORED: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* ============================================================
	 * TEST 3: Multiple IC invalidation pulses + longer delays
	 * ============================================================ */
	pr_info("p5: --- TEST 3: Multiple IC invalidation pulses ---\n");

	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);

	write_and_flush(PATCH_DW_OFF, MARKER_INSTR);
	wbinvd();
	mb();

	/* Pulse IC invalidation 10 times */
	{
		int k;
		for (k = 0; k < 10; k++) {
			wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_INVALIDATE_ICACHE);
			udelay(100);
			wr(regCP_MEC_CNTL, MEC_ME1_HALT);
			udelay(100);
		}
	}

	/* Also try IC_OP_CNTL multiple times */
	{
		int k;
		for (k = 0; k < 5; k++) {
			wr(regCP_CPC_IC_OP_CNTL, 0x1);
			udelay(500);
		}
	}

	/* Pipe reset */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET);
	mdelay(10);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	pr_info("p5: 3 after reset: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	wr(regCP_MEC_CNTL, 0);
	mdelay(20);
	pr_info("p5: 3 after unhalt: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	sample_pc("3", 5);

	/* Restore */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	restore_and_flush();
	wbinvd();
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_INVALIDATE_ICACHE);
	mdelay(1);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET);
	mdelay(5);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	wr(regCP_MEC_CNTL, 0);
	mdelay(50);
	pr_info("p5: 3 RESTORED: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* ============================================================
	 * TEST 4: Control — pipe reset WITHOUT patching
	 * Verify that pipe reset alone doesn't change PC
	 * ============================================================ */
	pr_info("p5: --- TEST 4: Control — pipe reset without patch ---\n");

	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	mdelay(5);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	pr_info("p5: 4 after reset: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	wr(regCP_MEC_CNTL, 0);
	mdelay(20);
	pr_info("p5: 4 after unhalt: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	sample_pc("4", 3);

	/* ============================================================
	 * TEST 5: Patch via ioremap (bypass CPU cache entirely)
	 * Map the physical page as uncacheable and write through it
	 * ============================================================ */
	pr_info("p5: --- TEST 5: ioremap_wc (uncacheable write) ---\n");
	{
		u64 phys = FW_PHYS + (u64)PATCH_DW_OFF * 4;
		void __iomem *fw_mmio;

		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);

		fw_mmio = ioremap_wc(phys & PAGE_MASK, PAGE_SIZE);
		if (fw_mmio) {
			u32 page_off = phys & ~PAGE_MASK;
			u32 before = readl(fw_mmio + page_off);
			pr_info("p5: 5 ioremap read [0x44C]=0x%08X\n", before);

			writel(MARKER_INSTR, fw_mmio + page_off);
			wmb();
			{
				u32 rb = readl(fw_mmio + page_off);
				pr_info("p5: 5 ioremap wrote 0x%08lX readback=0x%08X %s\n",
					MARKER_INSTR, rb,
					rb == MARKER_INSTR ? "OK" : "FAIL");
			}

			/* IC invalidate then pipe reset */
			wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_INVALIDATE_ICACHE);
			mdelay(5);
			wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET);
			mdelay(5);
			wr(regCP_MEC_CNTL, MEC_ME1_HALT);
			udelay(500);

			wr(regCP_MEC_CNTL, 0);
			mdelay(20);
			pr_info("p5: 5 after unhalt: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
			sample_pc("5", 5);

			/* Restore */
			wr(regCP_MEC_CNTL, MEC_ME1_HALT);
			udelay(500);
			writel(before, fw_mmio + page_off);
			wmb();
			wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_INVALIDATE_ICACHE);
			mdelay(5);
			wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET);
			mdelay(5);
			wr(regCP_MEC_CNTL, MEC_ME1_HALT);
			udelay(500);
			wr(regCP_MEC_CNTL, 0);
			mdelay(50);
			pr_info("p5: 5 RESTORED: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

			iounmap(fw_mmio);
		} else {
			pr_err("p5: 5 ioremap_wc FAILED\n");
		}
	}

	pr_info("p5: FINAL: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));
	pr_info("p5: === PHASE 5 COMPLETE ===\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit patch5_exit(void) {}
module_init(patch5_init);
module_exit(patch5_exit);
