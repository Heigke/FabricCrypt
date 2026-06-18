/*
 * patch_mec_fw22.c — Phase 22: Explore decrypted firmware BO and MEC reload
 *
 * From Phase 19/21:
 *   adev + 0x16800 = ffff8c9d562bc000 (kernel ptr — possible BO)
 *   adev + 0x16808 = 0x97FF943000 (GPU VA)
 *   adev + 0x16850 = ffffffffc1c55380 (checked by config_mec_cache)
 *
 * The encrypted firmware blob is PSP-decrypted and loaded to MEC SRAM.
 * The decrypted copy might live at the GPU VA pointed to by these BOs.
 *
 * Plan:
 *   1. Examine the BO at adev+0x16800 — read its TTM pages to get
 *      the decrypted firmware content
 *   2. Also examine adev+0x16850 as potential fw_obj
 *   3. Try to find a TTM kmap of the BO
 *   4. Check if MEC can be halted and IC_BASE rewritten
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/highmem.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002
#define AMD_DEV_ID    0x1586

static void __iomem *mmio;
static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }
static void wr(u32 dw_off, u32 val) {
	writel(val, mmio + (u64)dw_off * 4);
	readl(mmio + (u64)dw_off * 4);
}

/* Registers */
#define regCP_MEC1_INSTR_PNTR  0x021A8
#define regCP_MEC_CNTL         0x0A802
#define MEC_ME1_HALT           (1 << 30)
#define MEC_ME2_HALT           (1 << 28)

/* IC_BASE registers (from Phase 17d — BASE_IDX=1 => +0xA000 not needed for MMIO) */
#define regCP_CPC_IC_BASE_LO   0x0C930
#define regCP_CPC_IC_BASE_HI   0x0C931

static void dump_bo_like(void *ptr, const char *name)
{
	u64 dump[32]; /* 256 bytes */
	int i;

	if (copy_from_kernel_nofault(dump, ptr, sizeof(dump))) {
		pr_info("fw22: %s: cannot read\n", name);
		return;
	}

	pr_info("fw22: %s at %px:\n", name, ptr);
	for (i = 0; i < 32; i++) {
		const char *tag = "";
		if (dump[i] == 267904ULL) tag = " <<< FW_SIZE?";
		else if (dump[i] == 263168ULL) tag = " <<< UCODE_SIZE?";
		else if (dump[i] == 270336ULL) tag = " <<< PAGED_SIZE?";
		else if ((dump[i] >> 32) == 0x97ULL) tag = " <<< GPU_VA_97?";
		else if (dump[i] >= 0xffff800000000000ULL &&
			 dump[i] <= 0xffffffffffff0000ULL &&
			 dump[i] != 0xffffffffffffffffULL) tag = " KPTR";
		else if (dump[i] >= 0xffffea0000000000ULL) tag = " PAGE?";
		pr_info("fw22:   [0x%02X] = 0x%016llX%s\n", i * 8, dump[i], tag);
	}
}

/* Try to read firmware content from a BO-like pointer by navigating TTM structures.
 * amdgpu_bo -> tbo -> ttm (struct ttm_tt) -> pages[] -> kmap_local_page() */
static void try_read_bo_content(void *bo_ptr, const char *name)
{
	u64 fields[64]; /* 512 bytes of BO struct */
	int i;

	if (copy_from_kernel_nofault(fields, bo_ptr, sizeof(fields))) {
		pr_info("fw22: %s: cannot read BO struct\n", name);
		return;
	}

	/* Scan for a pointer that leads to a ttm_tt-like struct:
	 * ttm_tt has: {uint32_t num_pages; struct page **pages; ...}
	 * OR more specifically, scan for any pointer whose deref contains
	 * a struct page* array. */

	for (i = 0; i < 64; i++) {
		u64 sub_ptr = fields[i];

		if (sub_ptr < 0xffff800000000000ULL ||
		    sub_ptr > 0xffffffffffff0000ULL ||
		    sub_ptr == 0xffffffffffffffffULL)
			continue;

		/* Try reading this as a ttm_tt-like struct */
		{
			u64 sub[8];
			int j;

			if (copy_from_kernel_nofault(sub, (void *)sub_ptr, sizeof(sub)))
				continue;

			for (j = 0; j < 8; j++) {
				struct page *pg = NULL;
				struct page **pages;
				void *va;
				u32 *fw;

				if (sub[j] < 0xffff800000000000ULL)
					continue;
				if (sub[j] > 0xffffffffffff0000ULL)
					continue;

				/* Try reading as pages array */
				pages = (struct page **)sub[j];
				if (copy_from_kernel_nofault(&pg, &pages[0], sizeof(pg)))
					continue;

				if (!pg || (u64)pg < 0xffffea0000000000ULL)
					continue;

				/* This looks like a page! Try mapping it */
				va = kmap_local_page(pg);
				fw = (u32 *)va;

				if (fw[0] == 0xC424000BUL) {
					pr_info("fw22: *** FOUND DECRYPTED FW in %s! ***\n", name);
					pr_info("fw22:   BO[0x%02X] -> sub[%d] = pages at %px\n",
						i * 8, j, (void *)sub[j]);
					pr_info("fw22:   pages[0] = %px\n", pg);
					pr_info("fw22:   FW[0x000] = 0x%08X (FIRST INSTR)\n", fw[0]);
					pr_info("fw22:   FW[0x001] = 0x%08X\n", fw[1]);
					pr_info("fw22:   FW[0x002] = 0x%08X\n", fw[2]);
					pr_info("fw22:   FW[0x44A] = 0x%08X\n", fw[0x44A]);
					pr_info("fw22:   FW[0x44B] = 0x%08X\n", fw[0x44B]);
					pr_info("fw22:   FW[0x44C] = 0x%08X%s\n", fw[0x44C],
						fw[0x44C] == 0x88000000UL ? " BRANCH_SELF" : "");
					pr_info("fw22:   FW[0x44D] = 0x%08X\n", fw[0x44D]);
					kunmap_local(va);
					return;
				}

				kunmap_local(va);
			}
		}
	}

	pr_info("fw22: %s: no decrypted firmware found via TTM pages\n", name);
}

static int __init fw22_init(void)
{
	struct pci_dev *pdev = NULL;
	void *adev;
	u64 bo_ptr1, bo_ptr2;
	u32 mec_cntl, pc;

	pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
	if (!pdev)
		return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	pr_info("fw22: ========================================\n");
	pr_info("fw22: PHASE 22: DECRYPTED FW BO EXPLORATION\n");
	pr_info("fw22: ========================================\n");

	pc = rr(regCP_MEC1_INSTR_PNTR);
	mec_cntl = rr(regCP_MEC_CNTL);
	pr_info("fw22: PC=0x%04X MEC_CNTL=0x%08X\n", pc, mec_cntl);
	pr_info("fw22: IC_BASE_LO=0x%08X IC_BASE_HI=0x%08X\n",
		rr(regCP_CPC_IC_BASE_LO), rr(regCP_CPC_IC_BASE_HI));

	adev = (void *)pci_get_drvdata(pdev) - 16;

	/* Read BO pointers */
	copy_from_kernel_nofault(&bo_ptr1, adev + 0x16800, 8);
	copy_from_kernel_nofault(&bo_ptr2, adev + 0x16850, 8);

	pr_info("fw22: BO candidate 1 (0x16800): %px\n", (void *)bo_ptr1);
	pr_info("fw22: BO candidate 2 (0x16850): %px\n", (void *)bo_ptr2);

	/* Dump BO structs */
	if (bo_ptr1 >= 0xffff800000000000ULL &&
	    bo_ptr1 <= 0xffffffffffff0000ULL) {
		dump_bo_like((void *)bo_ptr1, "BO1(0x16800)");
		try_read_bo_content((void *)bo_ptr1, "BO1(0x16800)");
	}

	if (bo_ptr2 >= 0xffff800000000000ULL &&
	    bo_ptr2 <= 0xffffffffffff0000ULL &&
	    bo_ptr2 != bo_ptr1) {
		dump_bo_like((void *)bo_ptr2, "BO2(0x16850)");
		try_read_bo_content((void *)bo_ptr2, "BO2(0x16850)");
	}

	/* Try MEC halt + IC_BASE write test */
	pr_info("fw22: === MEC HALT/RELOAD TEST ===\n");

	/* First: try halting MEC */
	pr_info("fw22: Halting MEC...\n");
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_ME2_HALT);
	mec_cntl = rr(regCP_MEC_CNTL);
	pc = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("fw22: After halt: MEC_CNTL=0x%08X PC=0x%04X\n", mec_cntl, pc);

	/* Try reading IC_BASE while halted */
	{
		u32 ic_lo = rr(regCP_CPC_IC_BASE_LO);
		u32 ic_hi = rr(regCP_CPC_IC_BASE_HI);
		pr_info("fw22: Halted IC_BASE: LO=0x%08X HI=0x%08X\n", ic_lo, ic_hi);

		/* Try writing IC_BASE — just write same value back */
		wr(regCP_CPC_IC_BASE_LO, ic_lo);
		wr(regCP_CPC_IC_BASE_HI, ic_hi);

		{
			u32 new_lo = rr(regCP_CPC_IC_BASE_LO);
			u32 new_hi = rr(regCP_CPC_IC_BASE_HI);
			pr_info("fw22: After write-back: LO=0x%08X HI=0x%08X%s\n",
				new_lo, new_hi,
				(new_lo == ic_lo && new_hi == ic_hi) ?
				" (unchanged)" : " (CHANGED!)");
		}
	}

	/* Unhalt MEC */
	pr_info("fw22: Unhalting MEC...\n");
	wr(regCP_MEC_CNTL, 0);
	mec_cntl = rr(regCP_MEC_CNTL);
	pc = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("fw22: After unhalt: MEC_CNTL=0x%08X PC=0x%04X\n", mec_cntl, pc);

	pr_info("fw22: ========================================\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw22_exit(void) {}

module_init(fw22_init);
module_exit(fw22_exit);
