/*
 * patch_mec_fw12.c — Phase 12: PRIME_ICACHE — force firmware reload
 *
 * Discovery: CP_CPC_IC_OP_CNTL bit[4] = PRIME_ICACHE exists but driver
 * never uses it for MEC (only PFP/ME). PRIME forces the IC to reload
 * firmware from IC_BASE into internal instruction memory.
 *
 * We've been using INVALIDATE (bit 0) which only flushes the IC cache —
 * it re-reads from SRAM, NOT from external memory.
 *
 * PRIME_ICACHE should trigger: IC_BASE → DMA → internal SRAM.
 *
 * Tests:
 *   A: Patch all RAM copies + PRIME_ICACHE
 *   B: Patch all RAM copies + IC_BASE re-program + PRIME_ICACHE
 *   C: Patch all RAM copies + INVALIDATE + PRIME (both bits)
 *   D: Full driver sequence: DC_INVALIDATE + wait, IC_INVALIDATE + wait, PRIME + wait
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

#define regCP_MEC_CNTL              0x0A802
#define regCP_MEC1_INSTR_PNTR      0x021A8
#define regCP_CPC_IC_OP_CNTL       0x0C97A
#define regCP_MEC_DC_OP_CNTL       0x0C90C
#define regGRBM_GFX_CNTL           0x0A900
#define regCP_CPC_IC_BASE_LO       0x0F84C
#define regCP_CPC_IC_BASE_HI       0x0F84D
#define regCP_CPC_IC_BASE_CNTL     0x0F84E

/* CP_CPC_IC_OP_CNTL bits */
#define IC_INVALIDATE_CACHE          (1 << 0)
#define IC_INVALIDATE_CACHE_COMPLETE (1 << 1)
#define IC_PRIME_ICACHE              (1 << 4)
#define IC_ICACHE_PRIMED             (1 << 5)

/* CP_MEC_DC_OP_CNTL bits */
#define DC_INVALIDATE_DCACHE          (1 << 0)
#define DC_INVALIDATE_DCACHE_COMPLETE (1 << 1)

/* CP_MEC_CNTL bits */
#define MEC_ME1_PIPE0_RESET   (1 << 16)
#define MEC_ME1_PIPE1_RESET   (1 << 17)
#define MEC_ME1_PIPE2_RESET   (1 << 18)
#define MEC_ME1_PIPE3_RESET   (1 << 19)
#define MEC_INVALIDATE_ICACHE (1 << 27)
#define MEC_ME1_HALT          (1 << 30)
#define ALL_PIPE_RESET (MEC_ME1_PIPE0_RESET | MEC_ME1_PIPE1_RESET | \
			MEC_ME1_PIPE2_RESET | MEC_ME1_PIPE3_RESET)

#define RS64_NOP          0x7C408001UL
#define RS64_BRANCH_SELF  0x88000000UL

static void __iomem *mmio;

static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }
static void wr(u32 dw_off, u32 val) {
	writel(val, mmio + (u64)dw_off * 4);
	readl(mmio + (u64)dw_off * 4);
}

static void grbm_select(int me, int pipe, int queue, int vmid)
{
	u32 val = (pipe & 0xF) | ((me & 0x3) << 4) |
		  ((queue & 0x7) << 8) | ((vmid & 0xF) << 12);
	wr(regGRBM_GFX_CNTL, val);
	udelay(50);
}

/* All copies found by Phase 10 */
static u64 ram_copies[] = {
	0x115C4D878ULL,
	0x115C8C100ULL,  /* FW_PHYS */
	0x122D89978ULL,
	0x125FA7200ULL,
	0x1791C3200ULL,
	0x22E0C1200ULL,
	0x2A70AE200ULL,
	0x2AD7EA63CULL,
};
#define N_RAM_COPIES 8

static u32 saved_44a[N_RAM_COPIES];
static u32 saved_44c[N_RAM_COPIES];
static u32 saved_44e[N_RAM_COPIES];

static u32 *map_phys(u64 phys)
{
	unsigned long pfn = phys >> PAGE_SHIFT;
	unsigned int page_off = phys & ~PAGE_MASK;
	struct page *page;
	u8 *vaddr;
	if (!pfn_valid(pfn)) return NULL;
	page = pfn_to_page(pfn);
	vaddr = kmap_local_page(page);
	if (!vaddr) return NULL;
	return (u32 *)(vaddr + page_off);
}

static void unmap_phys(void *ptr)
{
	kunmap_local((void *)((unsigned long)ptr & PAGE_MASK));
}

static u32 read_phys(u64 phys)
{
	u32 val = 0;
	u32 *ptr = map_phys(phys);
	if (ptr) { val = *ptr; unmap_phys(ptr); }
	return val;
}

static void write_phys_flush(u64 phys, u32 val)
{
	u32 *ptr = map_phys(phys);
	if (ptr) {
		*ptr = val;
		clflush(ptr);
		wmb();
		unmap_phys(ptr);
	}
}

static void patch_all(int dw_off, u32 val)
{
	int i;
	for (i = 0; i < N_RAM_COPIES; i++)
		write_phys_flush(ram_copies[i] + (u64)dw_off * 4, val);
}

static void save_all(int dw_off, u32 *buf)
{
	int i;
	for (i = 0; i < N_RAM_COPIES; i++)
		buf[i] = read_phys(ram_copies[i] + (u64)dw_off * 4);
}

static void restore_all(int dw_off, u32 *buf)
{
	int i;
	for (i = 0; i < N_RAM_COPIES; i++)
		write_phys_flush(ram_copies[i] + (u64)dw_off * 4, buf[i]);
}

static void sample_pc(const char *label, int count)
{
	int j;
	for (j = 0; j < count; j++) {
		mdelay(5);
		pr_info("fw12: %s PC[%d]=0x%04X\n", label, j, rr(regCP_MEC1_INSTR_PNTR));
	}
}

static int wait_ic_complete(const char *op, int bit, int timeout_us)
{
	int j;
	u32 val;
	for (j = 0; j < timeout_us; j++) {
		val = rr(regCP_CPC_IC_OP_CNTL);
		if (val & bit) {
			pr_info("fw12: %s complete in %d us (IC_OP=0x%08X)\n", op, j, val);
			return 0;
		}
		udelay(1);
	}
	pr_info("fw12: %s TIMEOUT after %d us (IC_OP=0x%08X)\n", op, timeout_us, val);
	return -1;
}

static int wait_dc_complete(const char *op, int timeout_us)
{
	int j;
	u32 val;
	for (j = 0; j < timeout_us; j++) {
		val = rr(regCP_MEC_DC_OP_CNTL);
		if (val & DC_INVALIDATE_DCACHE_COMPLETE) {
			pr_info("fw12: %s complete in %d us (DC_OP=0x%08X)\n", op, j, val);
			return 0;
		}
		udelay(1);
	}
	pr_info("fw12: %s TIMEOUT after %d us (DC_OP=0x%08X)\n", op, timeout_us, val);
	return -1;
}

static void full_restore(void)
{
	restore_all(0x44A, saved_44a);
	restore_all(0x44C, saved_44c);
	restore_all(0x44E, saved_44e);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	wr(regCP_MEC_CNTL, 0);
	mdelay(50);
}

static int __init fw12_init(void)
{
	struct pci_dev *pdev = NULL;
	int i;
	u32 ic_op;

	while ((pdev = pci_get_device(AMD_VENDOR_ID, PCI_ANY_ID, pdev))) {
		if ((pdev->class >> 8) == PCI_CLASS_DISPLAY_OTHER)
			break;
	}
	if (!pdev) return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) { pci_dev_put(pdev); return -ENOMEM; }

	pr_info("fw12: ========================================\n");
	pr_info("fw12: PHASE 12: PRIME_ICACHE FIRMWARE RELOAD\n");
	pr_info("fw12: ========================================\n");
	pr_info("fw12: BASELINE: PC=0x%04X IC_OP=0x%08X DC_OP=0x%08X\n",
		rr(regCP_MEC1_INSTR_PNTR),
		rr(regCP_CPC_IC_OP_CNTL),
		rr(regCP_MEC_DC_OP_CNTL));

	/* Save originals */
	save_all(0x44A, saved_44a);
	save_all(0x44C, saved_44c);
	save_all(0x44E, saved_44e);

	/* ============================================================
	 * TEST A: Patch all copies + PRIME_ICACHE only
	 * ============================================================ */
	pr_info("fw12: --- TEST A: ALL copies NOP@44C + PRIME_ICACHE ---\n");

	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);

	patch_all(0x44C, RS64_NOP);

	/* Trigger PRIME_ICACHE (bit 4) */
	wr(regCP_CPC_IC_OP_CNTL, IC_PRIME_ICACHE);
	wait_ic_complete("A PRIME", IC_ICACHE_PRIMED, 10000);

	/* Pipe reset + unhalt */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	pr_info("fw12: A after reset: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	wr(regCP_MEC_CNTL, 0);
	mdelay(20);
	sample_pc("A", 8);

	/* Restore */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	full_restore();
	pr_info("fw12: A RESTORED: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* ============================================================
	 * TEST B: INVALIDATE + PRIME sequence (like PFP/ME driver does)
	 * ============================================================ */
	pr_info("fw12: --- TEST B: INVALIDATE + PRIME sequence ---\n");

	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);

	/* Patch: create spin at 0x44E, NOP at 0x44A and 0x44C */
	patch_all(0x44A, RS64_NOP);
	patch_all(0x44C, RS64_NOP);
	patch_all(0x44E, RS64_BRANCH_SELF);

	/* DC invalidate first (like driver) */
	wr(regCP_MEC_DC_OP_CNTL, DC_INVALIDATE_DCACHE);
	wait_dc_complete("B DC", 10000);

	/* IC invalidate */
	wr(regCP_CPC_IC_OP_CNTL, IC_INVALIDATE_CACHE);
	wait_ic_complete("B INVALIDATE", IC_INVALIDATE_CACHE_COMPLETE, 10000);

	/* Now PRIME */
	wr(regCP_CPC_IC_OP_CNTL, IC_PRIME_ICACHE);
	wait_ic_complete("B PRIME", IC_ICACHE_PRIMED, 10000);

	/* Pipe reset */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	pr_info("fw12: B after reset: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	wr(regCP_MEC_CNTL, 0);
	mdelay(20);
	sample_pc("B", 8);

	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	full_restore();
	pr_info("fw12: B RESTORED: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* ============================================================
	 * TEST C: Both bits simultaneously (INVALIDATE | PRIME)
	 * ============================================================ */
	pr_info("fw12: --- TEST C: INVALIDATE|PRIME simultaneous ---\n");

	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);

	patch_all(0x44A, RS64_NOP);
	patch_all(0x44C, RS64_NOP);
	patch_all(0x44E, RS64_BRANCH_SELF);

	/* DC + IC invalidate+prime in one shot */
	wr(regCP_MEC_DC_OP_CNTL, DC_INVALIDATE_DCACHE);
	wait_dc_complete("C DC", 10000);
	wr(regCP_CPC_IC_OP_CNTL, IC_INVALIDATE_CACHE | IC_PRIME_ICACHE);
	udelay(1000);
	ic_op = rr(regCP_CPC_IC_OP_CNTL);
	pr_info("fw12: C IC_OP after combined = 0x%08X\n", ic_op);
	pr_info("fw12: C  INVALIDATE_COMPLETE=%d PRIMED=%d\n",
		!!(ic_op & IC_INVALIDATE_CACHE_COMPLETE),
		!!(ic_op & IC_ICACHE_PRIMED));

	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	pr_info("fw12: C after reset: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	wr(regCP_MEC_CNTL, 0);
	mdelay(20);
	sample_pc("C", 8);

	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	full_restore();
	pr_info("fw12: C RESTORED: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* ============================================================
	 * TEST D: Re-write IC_BASE per pipe + PRIME (force full reload)
	 * Reading IC_BASE worked in Phase 8. Writing was ignored.
	 * Try re-writing the SAME value (not a new address) to trigger
	 * the hardware DMA.
	 * ============================================================ */
	pr_info("fw12: --- TEST D: Re-write IC_BASE + PRIME per pipe ---\n");
	{
		u32 ic_lo, ic_hi;

		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);

		patch_all(0x44A, RS64_NOP);
		patch_all(0x44C, RS64_NOP);
		patch_all(0x44E, RS64_BRANCH_SELF);

		/* Re-write IC_BASE for pipes 0 and 1 */
		for (i = 0; i < 2; i++) {
			grbm_select(1, i, 0, 0);
			ic_lo = rr(regCP_CPC_IC_BASE_LO);
			ic_hi = rr(regCP_CPC_IC_BASE_HI);
			pr_info("fw12: D pipe[%d] IC_BASE=0x%08X_%08X\n", i, ic_hi, ic_lo);

			/* Re-write same values to trigger DMA */
			wr(regCP_CPC_IC_BASE_LO, ic_lo);
			wr(regCP_CPC_IC_BASE_HI, ic_hi);
			udelay(100);

			/* Verify the write stuck */
			pr_info("fw12: D pipe[%d] readback=0x%08X_%08X\n",
				i, rr(regCP_CPC_IC_BASE_HI), rr(regCP_CPC_IC_BASE_LO));
		}
		grbm_select(0, 0, 0, 0);

		/* DC invalidate */
		wr(regCP_MEC_DC_OP_CNTL, DC_INVALIDATE_DCACHE);
		wait_dc_complete("D DC", 10000);

		/* IC invalidate + prime */
		wr(regCP_CPC_IC_OP_CNTL, IC_INVALIDATE_CACHE);
		wait_ic_complete("D INVALIDATE", IC_INVALIDATE_CACHE_COMPLETE, 10000);
		wr(regCP_CPC_IC_OP_CNTL, IC_PRIME_ICACHE);
		wait_ic_complete("D PRIME", IC_ICACHE_PRIMED, 10000);

		/* Pipe reset */
		wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET);
		udelay(2000);
		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);
		pr_info("fw12: D after reset: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

		wr(regCP_MEC_CNTL, 0);
		mdelay(20);
		sample_pc("D", 8);

		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);
		full_restore();
		pr_info("fw12: D RESTORED: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	}

	/* ============================================================
	 * TEST E: Per-pipe IC operations (GRBM context for IC_OP too)
	 * Maybe IC_OP needs to be done per-pipe like IC_BASE
	 * ============================================================ */
	pr_info("fw12: --- TEST E: Per-pipe IC invalidate + prime ---\n");
	{
		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);

		patch_all(0x44A, RS64_NOP);
		patch_all(0x44C, RS64_NOP);
		patch_all(0x44E, RS64_BRANCH_SELF);

		for (i = 0; i < 2; i++) {
			grbm_select(1, i, 0, 0);

			/* DC invalidate per pipe */
			wr(regCP_MEC_DC_OP_CNTL, DC_INVALIDATE_DCACHE);
			udelay(500);
			pr_info("fw12: E pipe[%d] DC_OP=0x%08X\n", i,
				rr(regCP_MEC_DC_OP_CNTL));

			/* IC invalidate per pipe */
			wr(regCP_CPC_IC_OP_CNTL, IC_INVALIDATE_CACHE);
			udelay(500);
			pr_info("fw12: E pipe[%d] IC_OP after inv=0x%08X\n", i,
				rr(regCP_CPC_IC_OP_CNTL));

			/* PRIME per pipe */
			wr(regCP_CPC_IC_OP_CNTL, IC_PRIME_ICACHE);
			udelay(500);
			ic_op = rr(regCP_CPC_IC_OP_CNTL);
			pr_info("fw12: E pipe[%d] IC_OP after prime=0x%08X (PRIMED=%d)\n",
				i, ic_op, !!(ic_op & IC_ICACHE_PRIMED));
		}
		grbm_select(0, 0, 0, 0);

		/* Pipe reset */
		wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET);
		udelay(2000);
		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);
		pr_info("fw12: E after reset: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

		wr(regCP_MEC_CNTL, 0);
		mdelay(20);
		sample_pc("E", 8);

		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);
		full_restore();
		pr_info("fw12: E RESTORED: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	}

	pr_info("fw12: FINAL: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	pr_info("fw12: ========================================\n");
	pr_info("fw12: PHASE 12 COMPLETE\n");
	pr_info("fw12: ========================================\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw12_exit(void) {}
module_init(fw12_init);
module_exit(fw12_exit);
