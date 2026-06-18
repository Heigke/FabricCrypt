/*
 * patch_mec_fw10.c — Phase 10: Find the REAL firmware copy
 *
 * Key insight: driver copies firmware into a GART BO (separate from the
 * request_firmware() buffer). We've been patching the wrong copy.
 *
 * This module:
 *   1. Reads the firmware signature (8 dwords at 0x448-0x44F) from FW_PHYS[0]
 *   2. Scans ALL physical memory for that exact signature
 *   3. Reports every match — one is FW_PHYS, others are the GART copy
 *   4. Patches the GART copy and tests
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
#include <linux/memblock.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002

/* GC registers */
#define regCP_MEC_CNTL              0x0A802
#define regCP_MEC1_INSTR_PNTR      0x021A8
#define regCP_CPC_IC_OP_CNTL       0x0C97A
#define regCP_MEC_DC_OP_CNTL       0x0C90C

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

/* Known firmware physical address (copy 0 = request_firmware buffer) */
#define FW_PHYS  0x115c8c100ULL

/* Signature: 4 dwords at offsets 0x449-0x44C in the firmware.
 * We use a 4-dword window that includes the two branch-self instructions
 * plus their neighbors for uniqueness. */
#define SIG_DW_START  0x449  /* start dword offset in firmware */
#define SIG_DW_COUNT  4      /* number of dwords in signature */
#define SIG_BYTE_OFF  (SIG_DW_START * 4)  /* byte offset from firmware base */

static u32 signature[SIG_DW_COUNT];

/* Max copies to find */
#define MAX_COPIES 16
static u64 copy_phys[MAX_COPIES];
static int num_copies;

static u32 *map_phys_dword(u64 phys)
{
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

static void unmap_phys(void *ptr)
{
	kunmap_local((void *)((unsigned long)ptr & PAGE_MASK));
}

static u32 read_phys_dword(u64 phys)
{
	u32 val = 0;
	u32 *ptr = map_phys_dword(phys);
	if (ptr) {
		val = *ptr;
		unmap_phys(ptr);
	}
	return val;
}

static void write_phys_dword(u64 phys, u32 val)
{
	u32 *ptr = map_phys_dword(phys);
	if (ptr) {
		*ptr = val;
		wmb();
		unmap_phys(ptr);
	}
}

/* Read the firmware signature from the known copy */
static void read_signature(void)
{
	int i;
	u64 base = FW_PHYS;
	for (i = 0; i < SIG_DW_COUNT; i++) {
		signature[i] = read_phys_dword(base + (u64)(SIG_DW_START + i) * 4);
	}
}

/* Check if a page contains the signature at the correct offset within the page.
 * The firmware start address could be at any page-aligned or non-aligned offset.
 * We search for the signature pattern anywhere in the page. */
static int check_page_for_sig(struct page *page, u64 page_phys)
{
	u8 *vaddr;
	u32 *dwords;
	int off, max_off;
	int found = 0;

	vaddr = kmap_local_page(page);
	if (!vaddr)
		return 0;

	dwords = (u32 *)vaddr;
	max_off = (PAGE_SIZE / 4) - SIG_DW_COUNT;

	for (off = 0; off <= max_off; off++) {
		if (dwords[off] == signature[0] &&
		    dwords[off+1] == signature[1] &&
		    dwords[off+2] == signature[2] &&
		    dwords[off+3] == signature[3]) {
			/* Found signature. The firmware base would be at:
			 * page_phys + off*4 - SIG_BYTE_OFF */
			u64 fw_base = page_phys + (u64)off * 4 - SIG_BYTE_OFF;
			if (num_copies < MAX_COPIES) {
				copy_phys[num_copies] = fw_base;
				pr_info("fw10: FOUND copy[%d] at phys=0x%llX (sig at page+0x%X)\n",
					num_copies, fw_base, off * 4);
				num_copies++;
				found++;
			}
		}
	}

	kunmap_local(vaddr);
	return found;
}

static void reset_and_sample(const char *label)
{
	int j;
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	pr_info("fw10: %s after reset: PC=0x%04X\n", label, rr(regCP_MEC1_INSTR_PNTR));
	wr(regCP_MEC_CNTL, 0);
	mdelay(20);
	for (j = 0; j < 8; j++) {
		mdelay(5);
		pr_info("fw10: %s PC[%d]=0x%04X\n", label, j, rr(regCP_MEC1_INSTR_PNTR));
	}
}

static int __init fw10_init(void)
{
	struct pci_dev *pdev = NULL;
	u64 phys;
	unsigned long pfn;
	unsigned long total_pages = 0, scanned_pages = 0;
	int i;

	while ((pdev = pci_get_device(AMD_VENDOR_ID, PCI_ANY_ID, pdev))) {
		if ((pdev->class >> 8) == PCI_CLASS_DISPLAY_OTHER)
			break;
	}
	if (!pdev) {
		pr_err("fw10: no AMD GPU\n");
		return -ENODEV;
	}

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENOMEM;
	}

	pr_info("fw10: ========================================\n");
	pr_info("fw10: PHASE 10: FIND ALL FIRMWARE COPIES\n");
	pr_info("fw10: ========================================\n");
	pr_info("fw10: BASELINE: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* Read signature */
	read_signature();
	pr_info("fw10: Signature (FW dwords 0x%X-0x%X):\n",
		SIG_DW_START, SIG_DW_START + SIG_DW_COUNT - 1);
	for (i = 0; i < SIG_DW_COUNT; i++)
		pr_info("fw10:   [0x%X] = 0x%08X\n", SIG_DW_START + i, signature[i]);

	/* Quick sanity: verify signature[2] or [3] is 0x88000000 */
	if (signature[1] != 0x88000000 && signature[2] != 0x88000000 &&
	    signature[3] != 0x88000000) {
		pr_warn("fw10: WARNING: No branch-self in signature, pattern may not be unique\n");
	}

	/* ============================================================
	 * STEP 1: Scan physical memory for firmware copies
	 * Only scan valid RAM pages (skip MMIO, reserved, etc.)
	 * Focus on first 32 GB to keep scan time reasonable.
	 * ============================================================ */
	pr_info("fw10: --- Scanning physical memory (0 - 32GB) ---\n");
	num_copies = 0;

	for (pfn = 0; pfn < (32ULL << 30) >> PAGE_SHIFT; pfn++) {
		struct page *page;
		u64 page_phys;

		if (!pfn_valid(pfn))
			continue;

		page = pfn_to_page(pfn);

		/* Skip pages that aren't regular RAM */
		if (PageReserved(page))
			continue;

		page_phys = (u64)pfn << PAGE_SHIFT;
		scanned_pages++;

		check_page_for_sig(page, page_phys);

		if (num_copies >= MAX_COPIES)
			break;

		/* Progress every 1GB */
		if ((pfn & ((1UL << (30 - PAGE_SHIFT)) - 1)) == 0 && pfn > 0)
			pr_info("fw10: ...scanned %lu GB (%lu pages, %d copies so far)\n",
				pfn >> (30 - PAGE_SHIFT), scanned_pages, num_copies);
	}

	/* Also scan reserved pages around the known FW_PHYS address
	 * (the request_firmware buffer might be in reserved memory) */
	{
		unsigned long fw_pfn = FW_PHYS >> PAGE_SHIFT;
		unsigned long start = (fw_pfn > 256) ? fw_pfn - 256 : 0;
		unsigned long end = fw_pfn + 256;
		pr_info("fw10: Also scanning reserved pages around FW_PHYS...\n");
		for (pfn = start; pfn <= end; pfn++) {
			struct page *page;
			if (!pfn_valid(pfn))
				continue;
			page = pfn_to_page(pfn);
			if (!PageReserved(page))
				continue;  /* already scanned above */
			check_page_for_sig(page, (u64)pfn << PAGE_SHIFT);
		}
	}

	pr_info("fw10: Scan complete: %d copies found, %lu pages scanned\n",
		num_copies, scanned_pages);

	/* ============================================================
	 * STEP 2: For each copy != FW_PHYS, try patching it
	 * ============================================================ */
	for (i = 0; i < num_copies; i++) {
		u64 base = copy_phys[i];
		u64 idle_phys = base + 0x44C * 4;
		u32 idle_val;
		int is_known = 0;

		/* Check if this is the known FW_PHYS copy */
		if (base >= FW_PHYS - 0x1000 && base <= FW_PHYS + 0x1000) {
			is_known = 1;
			pr_info("fw10: copy[%d] at 0x%llX — THIS IS FW_PHYS (known copy)\n",
				i, base);
			continue;
		}

		pr_info("fw10: copy[%d] at 0x%llX — UNKNOWN (potential GART copy!)\n",
			i, base);

		/* Verify idle loop */
		idle_val = read_phys_dword(idle_phys);
		pr_info("fw10: copy[%d] FW[0x44C] = 0x%08X\n", i, idle_val);

		if (idle_val != 0x88000000) {
			pr_info("fw10: copy[%d] no idle loop at 0x44C, skipping\n", i);
			continue;
		}

		/* Dump context */
		{
			int d;
			for (d = 0x448; d <= 0x450; d++)
				pr_info("fw10: copy[%d] [0x%X] = 0x%08X\n",
					i, d, read_phys_dword(base + (u64)d * 4));
		}

		/* ============================================================
		 * TEST: Patch THIS copy + IC invalidate + pipe reset
		 * ============================================================ */
		pr_info("fw10: --- TEST: Patching copy[%d] at 0x%llX ---\n", i, base);

		/* Save original */
		{
			u32 orig_44a = read_phys_dword(base + 0x44A * 4);
			u32 orig_44c = read_phys_dword(base + 0x44C * 4);

			/* Halt MEC */
			wr(regCP_MEC_CNTL, MEC_ME1_HALT);
			udelay(500);

			/* Patch: break both spins */
			write_phys_dword(base + 0x44C * 4, 0xBF80DEAD);
			/* Also flush cache line */
			{
				u32 *ptr = map_phys_dword(base + 0x44C * 4);
				if (ptr) {
					clflush(ptr);
					wmb();
					unmap_phys(ptr);
				}
			}

			pr_info("fw10: T patched [0x44C]=0x%08X (readback=0x%08X)\n",
				0xBF80DEAD, read_phys_dword(base + 0x44C * 4));

			/* DC + IC invalidate (driver style) */
			wr(regCP_MEC_DC_OP_CNTL, 0x00000001);
			udelay(500);
			pr_info("fw10: T DC_OP=0x%08X\n", rr(regCP_MEC_DC_OP_CNTL));
			wr(regCP_CPC_IC_OP_CNTL, 0x00000001);
			udelay(500);
			pr_info("fw10: T IC_OP=0x%08X\n", rr(regCP_CPC_IC_OP_CNTL));

			/* Pipe reset */
			reset_and_sample("T");

			/* Restore */
			wr(regCP_MEC_CNTL, MEC_ME1_HALT);
			udelay(500);
			write_phys_dword(base + 0x44C * 4, orig_44c);
			{
				u32 *ptr = map_phys_dword(base + 0x44C * 4);
				if (ptr) {
					clflush(ptr);
					wmb();
					unmap_phys(ptr);
				}
			}
			wr(regCP_MEC_DC_OP_CNTL, 0x00000001);
			udelay(500);
			wr(regCP_CPC_IC_OP_CNTL, 0x00000001);
			udelay(500);
			wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
			udelay(2000);
			wr(regCP_MEC_CNTL, MEC_ME1_HALT);
			udelay(500);
			wr(regCP_MEC_CNTL, 0);
			mdelay(50);
			pr_info("fw10: T RESTORED: PC=0x%04X\n",
				rr(regCP_MEC1_INSTR_PNTR));
		}
	}

	/* ============================================================
	 * STEP 3: If no new copies found, try patching BOTH FW_PHYS
	 * AND searching all pages (including reserved) more aggressively
	 * ============================================================ */
	if (num_copies <= 1) {
		pr_info("fw10: --- Only found known copy. Doing aggressive reserved scan ---\n");
		/* Scan ALL valid pfns including reserved, in 0-8GB range */
		num_copies = 0;
		for (pfn = 0; pfn < (8ULL << 30) >> PAGE_SHIFT; pfn++) {
			struct page *page;
			if (!pfn_valid(pfn))
				continue;
			page = pfn_to_page(pfn);
			check_page_for_sig(page, (u64)pfn << PAGE_SHIFT);
			if (num_copies >= MAX_COPIES)
				break;
			if ((pfn & ((1UL << (30 - PAGE_SHIFT)) - 1)) == 0 && pfn > 0)
				pr_info("fw10: ...aggressive: %lu GB, %d copies\n",
					pfn >> (30 - PAGE_SHIFT), num_copies);
		}
		pr_info("fw10: Aggressive scan: %d total copies\n", num_copies);

		for (i = 0; i < num_copies; i++) {
			if (copy_phys[i] >= FW_PHYS - 0x1000 &&
			    copy_phys[i] <= FW_PHYS + 0x1000)
				pr_info("fw10: copy[%d] 0x%llX = FW_PHYS (known)\n",
					i, copy_phys[i]);
			else
				pr_info("fw10: copy[%d] 0x%llX = NEW!\n",
					i, copy_phys[i]);
		}
	}

	pr_info("fw10: FINAL: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	pr_info("fw10: ========================================\n");
	pr_info("fw10: PHASE 10 COMPLETE\n");
	pr_info("fw10: ========================================\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw10_exit(void) {}
module_init(fw10_init);
module_exit(fw10_exit);
