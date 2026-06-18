/*
 * patch_mec_fw3.c — Phase 3: IC invalidation via CP_MEC_CNTL bit 27
 *
 * Phase 2 showed IC_OP_CNTL writes are PSP-blocked (or no-op).
 * CP_MEC_CNTL IS writable (we proved halt via bit 30 works).
 * Bit 27 = MEC_INVALIDATE_ICACHE — separate HW invalidation path.
 *
 * Strategy:
 *   1. Record pre-halt PC and MEC_CNTL
 *   2. Halt MEC (bit 30)
 *   3. Patch ONLY copy[0] (the live firmware) at dword 0x44C
 *   4. Try multiple IC invalidation methods:
 *      a) MEC_INVALIDATE_ICACHE (bit 27) alone
 *      b) Bit 27 + pipe resets (bits 16-19)
 *      c) Full reset cycle: halt → invalidate → unhalt
 *   5. After each method, unhalt and sample PC
 *   6. If PC changes from 0x44C → invalidation worked
 *   7. RESTORE original instruction before exit
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

/* GFX 11.5.1 register dword offsets */
#define regCP_MEC_CNTL              0x0A802
#define regCP_CPC_IC_OP_CNTL       0x0C97A
#define regCP_MEC1_INSTR_PNTR      0x021A8

/* CP_MEC_CNTL bit definitions */
#define MEC_ME1_PIPE0_RESET   (1 << 16)
#define MEC_ME1_PIPE1_RESET   (1 << 17)
#define MEC_ME1_PIPE2_RESET   (1 << 18)
#define MEC_ME1_PIPE3_RESET   (1 << 19)
#define MEC_INVALIDATE_ICACHE (1 << 27)
#define MEC_ME2_HALT          (1 << 28)
#define MEC_ME1_HALT          (1 << 30)
#define MEC_ME1_STEP          (1 << 31)

#define ALL_PIPE_RESET (MEC_ME1_PIPE0_RESET | MEC_ME1_PIPE1_RESET | \
			MEC_ME1_PIPE2_RESET | MEC_ME1_PIPE3_RESET)

static void __iomem *mmio;

static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }
static void wr(u32 dw_off, u32 val) {
	writel(val, mmio + (u64)dw_off * 4);
	readl(mmio + (u64)dw_off * 4); /* flush */
}

/* Live firmware copy — the ONLY one with correct data at offset 0x44C */
#define FW_PHYS  0x115c8c100ULL  /* pfn 1137804, off 256 */

#define PATCH_DW_OFF    0x44C
#define MARKER_INSTR    0xBF80DEADUL  /* s_nop 0xDEAD */

static u32 orig_instr;

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

static void sample_pc(const char *label, int count)
{
	int j;
	for (j = 0; j < count; j++) {
		mdelay(5);
		pr_info("patch3: %s PC[%d]=0x%04X MEC_CNTL=0x%08X\n",
			label, j, rr(regCP_MEC1_INSTR_PNTR), rr(regCP_MEC_CNTL));
	}
}

static int __init patch3_init(void)
{
	struct pci_dev *pdev = NULL;
	u32 mec_cntl_orig, pc_orig;
	u32 *ptr;
	u32 readback;

	while ((pdev = pci_get_device(AMD_VENDOR_ID, PCI_ANY_ID, pdev))) {
		if ((pdev->class >> 8) == PCI_CLASS_DISPLAY_OTHER)
			break;
	}
	if (!pdev) {
		pr_err("patch3: no AMD GPU\n");
		return -ENODEV;
	}
	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENOMEM;
	}

	pr_info("patch3: === PHASE 3: MEC_INVALIDATE_ICACHE (bit 27) TEST ===\n");

	/* Step 0: Baseline */
	mec_cntl_orig = rr(regCP_MEC_CNTL);
	pc_orig = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("patch3: BASELINE: MEC_CNTL=0x%08X PC=0x%04X\n",
		mec_cntl_orig, pc_orig);

	/* Step 1: Map and read original instruction */
	ptr = map_fw_dword(PATCH_DW_OFF);
	if (!ptr) {
		pr_err("patch3: MAP FAILED for copy[0]\n");
		goto out;
	}
	orig_instr = *ptr;
	pr_info("patch3: copy[0] orig[0x%X] = 0x%08X\n", PATCH_DW_OFF, orig_instr);
	unmap_fw(ptr);

	/* ============================================================
	 * TEST A: Halt → Patch → bit 27 (MEC_INVALIDATE_ICACHE) → Unhalt
	 * ============================================================ */
	pr_info("patch3: --- TEST A: bit 27 MEC_INVALIDATE_ICACHE ---\n");

	/* Halt MEC */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	pr_info("patch3: A.1 HALTED: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));

	/* Patch firmware */
	ptr = map_fw_dword(PATCH_DW_OFF);
	if (ptr) {
		*ptr = MARKER_INSTR;
		wmb();
		readback = *ptr;
		pr_info("patch3: A.2 PATCHED: wrote 0x%08X readback=0x%08X %s\n",
			MARKER_INSTR, readback,
			readback == MARKER_INSTR ? "OK" : "FAIL");
		unmap_fw(ptr);
	}

	/* Set MEC_INVALIDATE_ICACHE while halted */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_INVALIDATE_ICACHE);
	udelay(1000);
	pr_info("patch3: A.3 IC_INVAL SET: MEC_CNTL=0x%08X\n", rr(regCP_MEC_CNTL));

	/* Clear invalidate bit, keep halted */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(100);
	pr_info("patch3: A.4 IC_INVAL CLR: MEC_CNTL=0x%08X\n", rr(regCP_MEC_CNTL));

	/* Unhalt */
	wr(regCP_MEC_CNTL, 0);
	mdelay(10);
	pr_info("patch3: A.5 UNHALTED: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));
	sample_pc("A", 5);

	/* Check: did PC move away from 0x44C? */
	{
		u32 pc_now = rr(regCP_MEC1_INSTR_PNTR);
		if (pc_now != pc_orig)
			pr_info("patch3: *** TEST A: PC CHANGED! 0x%04X → 0x%04X — IC MAY BE INVALIDATED ***\n",
				pc_orig, pc_now);
		else
			pr_info("patch3: TEST A: PC unchanged at 0x%04X — bit 27 did NOT invalidate IC\n",
				pc_now);
	}

	/* Restore before next test */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	ptr = map_fw_dword(PATCH_DW_OFF);
	if (ptr) {
		*ptr = orig_instr;
		wmb();
		unmap_fw(ptr);
	}
	wr(regCP_MEC_CNTL, 0);
	mdelay(10);

	/* ============================================================
	 * TEST B: Halt → Patch → Pipe Resets (bits 16-19) → bit 27 → Unhalt
	 * ============================================================ */
	pr_info("patch3: --- TEST B: PIPE_RESET + bit 27 ---\n");

	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	pr_info("patch3: B.1 HALTED: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));

	/* Patch */
	ptr = map_fw_dword(PATCH_DW_OFF);
	if (ptr) {
		*ptr = MARKER_INSTR;
		wmb();
		readback = *ptr;
		pr_info("patch3: B.2 PATCHED: 0x%08X readback=0x%08X %s\n",
			MARKER_INSTR, readback,
			readback == MARKER_INSTR ? "OK" : "FAIL");
		unmap_fw(ptr);
	}

	/* Assert all pipe resets + IC invalidate while halted */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);
	pr_info("patch3: B.3 RESET+INVAL: MEC_CNTL=0x%08X\n", rr(regCP_MEC_CNTL));

	/* De-assert resets but keep halted */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	pr_info("patch3: B.4 RESET CLR: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));

	/* Unhalt */
	wr(regCP_MEC_CNTL, 0);
	mdelay(10);
	pr_info("patch3: B.5 UNHALTED: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));
	sample_pc("B", 5);

	{
		u32 pc_now = rr(regCP_MEC1_INSTR_PNTR);
		if (pc_now != pc_orig)
			pr_info("patch3: *** TEST B: PC CHANGED! 0x%04X → 0x%04X — PIPE RESET + IC INVAL WORKED ***\n",
				pc_orig, pc_now);
		else
			pr_info("patch3: TEST B: PC unchanged at 0x%04X — pipe reset + bit 27 did NOT work\n",
				pc_now);
	}

	/* Restore before next test */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	ptr = map_fw_dword(PATCH_DW_OFF);
	if (ptr) {
		*ptr = orig_instr;
		wmb();
		unmap_fw(ptr);
	}
	wr(regCP_MEC_CNTL, 0);
	mdelay(10);

	/* ============================================================
	 * TEST C: Full nuke: Halt both MEs → Pipe reset → IC inval (bit 27)
	 *         → IC_OP_CNTL → Step once → Unhalt
	 * ============================================================ */
	pr_info("patch3: --- TEST C: FULL NUKE (halt both + reset + both IC methods + step) ---\n");

	/* Halt both ME1 and ME2 */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_ME2_HALT);
	udelay(500);
	pr_info("patch3: C.1 BOTH HALTED: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));

	/* Patch */
	ptr = map_fw_dword(PATCH_DW_OFF);
	if (ptr) {
		*ptr = MARKER_INSTR;
		wmb();
		readback = *ptr;
		pr_info("patch3: C.2 PATCHED: 0x%08X readback=0x%08X %s\n",
			MARKER_INSTR, readback,
			readback == MARKER_INSTR ? "OK" : "FAIL");
		unmap_fw(ptr);
	}

	/* Full reset + IC invalidate */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_ME2_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	mdelay(5);
	pr_info("patch3: C.3 FULL NUKE: MEC_CNTL=0x%08X\n", rr(regCP_MEC_CNTL));

	/* Also try IC_OP_CNTL while everything is halted+reset */
	wr(regCP_CPC_IC_OP_CNTL, 0x1);
	udelay(1000);
	pr_info("patch3: C.4 IC_OP_CNTL=0x%08X\n", rr(regCP_CPC_IC_OP_CNTL));

	/* De-assert resets and IC invalidate, keep halted */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_ME2_HALT);
	udelay(500);

	/* Try stepping once (bit 31) — forces one instruction fetch */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_ME1_STEP);
	udelay(500);
	pr_info("patch3: C.5 STEPPED: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));

	/* Clear step, unhalt all */
	wr(regCP_MEC_CNTL, 0);
	mdelay(10);
	pr_info("patch3: C.6 UNHALTED: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));
	sample_pc("C", 5);

	{
		u32 pc_now = rr(regCP_MEC1_INSTR_PNTR);
		if (pc_now != pc_orig)
			pr_info("patch3: *** TEST C: PC CHANGED! 0x%04X → 0x%04X — FULL NUKE WORKED ***\n",
				pc_orig, pc_now);
		else
			pr_info("patch3: TEST C: PC unchanged at 0x%04X — all methods exhausted\n",
				pc_now);
	}

	/* ============================================================
	 * FINAL RESTORE — always restore original instruction
	 * ============================================================ */
	pr_info("patch3: --- RESTORING ---\n");
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	ptr = map_fw_dword(PATCH_DW_OFF);
	if (ptr) {
		*ptr = orig_instr;
		wmb();
		readback = *ptr;
		pr_info("patch3: RESTORED: 0x%08X readback=0x%08X %s\n",
			orig_instr, readback,
			readback == orig_instr ? "OK" : "FAIL");
		unmap_fw(ptr);
	}

	/* IC invalidate after restore (try both methods) */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_INVALIDATE_ICACHE);
	udelay(1000);
	wr(regCP_CPC_IC_OP_CNTL, 0x1);
	udelay(1000);

	/* Unhalt */
	wr(regCP_MEC_CNTL, 0);
	mdelay(20);
	pr_info("patch3: FINAL: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));

	pr_info("patch3: === PHASE 3 COMPLETE ===\n");

out:
	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit patch3_exit(void) {}
module_init(patch3_init);
module_exit(patch3_exit);
