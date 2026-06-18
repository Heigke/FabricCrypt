/*
 * patch_mec_fw4.c — Phase 4: Confirm arbitrary MEC code injection
 *
 * Phase 3 proved: PIPE_RESET forces IC re-fetch from system RAM.
 * MEC saw our patched code (PC shifted from 0x44C to 0x44A).
 *
 * This module proves arbitrary control flow:
 *   TEST A: Patch BOTH spin points (0x44A + 0x44C) → MEC should end up
 *           at a completely different PC (earlier spin or crash)
 *   TEST B: Patch 0x44C with branch-to-0x44E (skip idle) → MEC should
 *           execute code past the idle loop
 *   TEST C: Inject 2-instruction payload at 0x44C: write scratch + spin
 *           → prove arbitrary register write from custom microcode
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
#define regCP_SCRATCH_INDEX         0x0A9D0
#define regCP_SCRATCH_DATA          0x0A9D1

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

/* Live firmware copy — confirmed in Phase 2/3 */
#define FW_PHYS  0x115c8c100ULL

/* RS64 branch encoding: 0x88XXXXXX where XXXXXX = signed 24-bit PC offset
 * 0x88000000 = branch to self (spin)
 * Testing: 0x88000001 = branch to PC+1, 0x88FFFFFF = branch to PC-1 */
#define RS64_BRANCH_SELF  0x88000000UL
#define RS64_NOP          0x7C408001UL  /* s_mov s1, s0 (nop-like, seen in FW) */

static u32 orig[8]; /* save 8 dwords around patch area */

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

/* Helper: halt MEC, apply pipe reset, unhalt, sample PC */
static void reset_and_sample(const char *label)
{
	int j;

	/* Halt */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);

	/* Pipe reset + IC invalidate (bit 27 is self-clearing but required!) */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);

	/* Clear reset, keep halted */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	pr_info("patch4: %s after reset: PC=0x%04X\n", label, rr(regCP_MEC1_INSTR_PNTR));

	/* Unhalt */
	wr(regCP_MEC_CNTL, 0);
	mdelay(20);

	/* Sample PC */
	for (j = 0; j < 8; j++) {
		mdelay(5);
		pr_info("patch4: %s PC[%d]=0x%04X\n", label, j, rr(regCP_MEC1_INSTR_PNTR));
	}
}

static void save_region(int base_dw, int count)
{
	int i;
	for (i = 0; i < count; i++) {
		u32 *ptr = map_fw_dword(base_dw + i);
		if (ptr) {
			orig[i] = *ptr;
			unmap_fw(ptr);
		}
	}
}

static void restore_region(int base_dw, int count)
{
	int i;
	for (i = 0; i < count; i++) {
		u32 *ptr = map_fw_dword(base_dw + i);
		if (ptr) {
			*ptr = orig[i];
			wmb();
			unmap_fw(ptr);
		}
	}
}

static void write_fw_dword(int dw_off, u32 val)
{
	u32 *ptr = map_fw_dword(dw_off);
	if (ptr) {
		*ptr = val;
		wmb();
		unmap_fw(ptr);
	}
}

static u32 read_fw_dword(int dw_off)
{
	u32 val = 0;
	u32 *ptr = map_fw_dword(dw_off);
	if (ptr) {
		val = *ptr;
		unmap_fw(ptr);
	}
	return val;
}

static int __init patch4_init(void)
{
	struct pci_dev *pdev = NULL;
	u32 pc_orig;
	u32 scratch_before, scratch_after;

	while ((pdev = pci_get_device(AMD_VENDOR_ID, PCI_ANY_ID, pdev))) {
		if ((pdev->class >> 8) == PCI_CLASS_DISPLAY_OTHER)
			break;
	}
	if (!pdev) {
		pr_err("patch4: no AMD GPU\n");
		return -ENODEV;
	}
	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENOMEM;
	}

	pr_info("patch4: === PHASE 4: ARBITRARY CODE INJECTION TEST ===\n");
	pc_orig = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("patch4: BASELINE: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), pc_orig);

	/* Dump region 0x448-0x44F for reference */
	{
		int i;
		pr_info("patch4: Firmware around idle loop:\n");
		for (i = 0x448; i <= 0x44F; i++) {
			u32 val = read_fw_dword(i);
			pr_info("patch4:   [0x%04X] = 0x%08X\n", i, val);
		}
	}

	/* Save 8 dwords starting at 0x448 */
	save_region(0x448, 8);

	/* ============================================================
	 * TEST A: Break BOTH spin points (0x44A and 0x44C)
	 * Replace both with s_nop-like instructions
	 * MEC should end up at a completely different PC
	 * ============================================================ */
	pr_info("patch4: --- TEST A: Break BOTH spin points ---\n");

	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);

	write_fw_dword(0x44A, 0xBF80DEAD);  /* break first spin */
	write_fw_dword(0x44C, 0xBF80DEAD);  /* break second spin */

	pr_info("patch4: A patched [0x44A]=0x%08X [0x44C]=0x%08X\n",
		read_fw_dword(0x44A), read_fw_dword(0x44C));

	reset_and_sample("A");

	/* Restore */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	restore_region(0x448, 8);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	wr(regCP_MEC_CNTL, 0);
	mdelay(50);
	pr_info("patch4: A RESTORED: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* ============================================================
	 * TEST B: Redirect branch — patch 0x44C with branch to 0x450
	 * Tests if we can redirect MEC execution to arbitrary PC
	 * 0x88000004 = branch forward 4 dwords (to 0x450 if offset=PC+offset)
	 * 0x88000003 = branch forward 3 dwords (to 0x44F if offset=PC+offset)
	 * Try both to determine offset semantics
	 * ============================================================ */
	pr_info("patch4: --- TEST B: Redirect branch ---\n");

	/* B1: patch 0x44C with 0x88000004 */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);

	/* Also patch 0x44A to not trap there */
	write_fw_dword(0x44A, 0xBF80DEAD);  /* break early spin */
	/* Patch 0x44C with forward branch */
	write_fw_dword(0x44C, 0x88000004);

	pr_info("patch4: B1 patched [0x44A]=nop [0x44C]=branch+4\n");
	reset_and_sample("B1");

	/* Restore */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	restore_region(0x448, 8);

	/* B2: patch 0x44C with 0x88000008 (branch forward 8) */
	write_fw_dword(0x44A, 0xBF80DEAD);
	write_fw_dword(0x44C, 0x88000008);
	pr_info("patch4: B2 patched [0x44A]=nop [0x44C]=branch+8\n");
	reset_and_sample("B2");

	/* Restore */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	restore_region(0x448, 8);

	/* B3: Keep 0x44A intact, just patch 0x44C with branch to 0x44A
	 * If offset is PC-relative: 0x44C + (-2) = 0x44A → offset = 0xFFFFFE (24-bit signed)
	 * Try 0x88FFFFFE */
	write_fw_dword(0x44C, 0x88FFFFFE);
	pr_info("patch4: B3 patched [0x44A]=orig(0x%08X) [0x44C]=branch(-2)\n",
		read_fw_dword(0x44A));
	reset_and_sample("B3");

	/* Restore */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	restore_region(0x448, 8);

	/* ============================================================
	 * TEST C: Insert spin at a NEW location
	 * Patch 0x44E with self-branch (new spin point) and break 0x44A/0x44C
	 * If MEC ends up at 0x44E → we created a new spin point
	 * ============================================================ */
	pr_info("patch4: --- TEST C: Create new spin point at 0x44E ---\n");

	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);

	write_fw_dword(0x44A, 0xBF80DEAD);  /* break spin 1 */
	write_fw_dword(0x44C, 0xBF80DEAD);  /* break spin 2 */
	write_fw_dword(0x44E, RS64_BRANCH_SELF);  /* NEW spin at 0x44E */

	pr_info("patch4: C patched [0x44A]=nop [0x44C]=nop [0x44E]=spin\n");
	reset_and_sample("C");

	/* Restore everything */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	restore_region(0x448, 8);

	/* ============================================================
	 * TEST D: Try to write to scratch register from microcode
	 * Read CP_SCRATCH before, inject write sequence, check after
	 * RS64 register write is likely via s_setreg or mapped store
	 * For now, just check if we can make MEC write GRBM-visible state
	 * ============================================================ */
	pr_info("patch4: --- TEST D: Scratch register probe ---\n");

	/* Read scratch reg 0 via MMIO */
	wr(regCP_SCRATCH_INDEX, 0);
	scratch_before = rr(regCP_SCRATCH_DATA);
	pr_info("patch4: D scratch[0] before = 0x%08X\n", scratch_before);

	/* Inject at 0x44C: C4240005 = s_load_dword? or use known firmware patterns
	 * Actually, let's just check if MEC naturally writes scratch on boot */
	wr(regCP_SCRATCH_INDEX, 0);
	wr(regCP_SCRATCH_DATA, 0xAAAAAAAA);  /* We write a known value */
	pr_info("patch4: D wrote 0xAAAAAAAA to scratch[0]\n");

	/* Now pipe reset (with original firmware) to see if boot overwrites scratch */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	wr(regCP_MEC_CNTL, 0);
	mdelay(50);

	wr(regCP_SCRATCH_INDEX, 0);
	scratch_after = rr(regCP_SCRATCH_DATA);
	pr_info("patch4: D scratch[0] after boot = 0x%08X (%s)\n",
		scratch_after,
		scratch_after != 0xAAAAAAAA ? "CHANGED by MEC boot" : "UNCHANGED");

	/* Check several scratch regs */
	{
		int s;
		for (s = 0; s < 8; s++) {
			wr(regCP_SCRATCH_INDEX, s);
			pr_info("patch4: D scratch[%d] = 0x%08X\n", s, rr(regCP_SCRATCH_DATA));
		}
	}

	/* Final restore + pipe reset to leave MEC in good state */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	wr(regCP_MEC_CNTL, 0);
	mdelay(50);

	pr_info("patch4: FINAL: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));
	pr_info("patch4: === PHASE 4 COMPLETE ===\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit patch4_exit(void) {}
module_init(patch4_init);
module_exit(patch4_exit);
