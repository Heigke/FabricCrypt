/*
 * patch_mec_fw11.c — Phase 11: VRAM full search + valid RS64 NOP patch
 *
 * Phase 10 found 8 RAM copies but patching with 0xBF80DEAD never moved PC.
 * Two theories:
 *   A) 0xBF80DEAD is invalid RS64 → exception → MEC restarts at 0x44C
 *   B) Real copy is in VRAM, not system RAM
 *
 * This module:
 *   TEST A: Search full 256MB VRAM BAR0 for firmware, patch in VRAM
 *   TEST B: Patch ALL system RAM copies simultaneously with valid NOP
 *   TEST C: Patch FW_PHYS with NOP + simple Phase-3-style pipe reset
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

#define MEC_ME1_PIPE0_RESET   (1 << 16)
#define MEC_ME1_PIPE1_RESET   (1 << 17)
#define MEC_ME1_PIPE2_RESET   (1 << 18)
#define MEC_ME1_PIPE3_RESET   (1 << 19)
#define MEC_INVALIDATE_ICACHE (1 << 27)
#define MEC_ME1_HALT          (1 << 30)
#define ALL_PIPE_RESET (MEC_ME1_PIPE0_RESET | MEC_ME1_PIPE1_RESET | \
			MEC_ME1_PIPE2_RESET | MEC_ME1_PIPE3_RESET)

/* Known valid RS64 instructions (from actual firmware) */
#define RS64_NOP          0x7C408001UL  /* s_mov s1, s0 — nop-like */
#define RS64_BRANCH_SELF  0x88000000UL

static void __iomem *mmio;

static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }
static void wr(u32 dw_off, u32 val) {
	writel(val, mmio + (u64)dw_off * 4);
	readl(mmio + (u64)dw_off * 4);
}

#define FW_PHYS  0x115c8c100ULL

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

static u32 saved_44c[N_RAM_COPIES];

static u32 *map_phys(u64 phys)
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

static void sample_pc(const char *label, int count)
{
	int j;
	for (j = 0; j < count; j++) {
		mdelay(5);
		pr_info("fw11: %s PC[%d]=0x%04X\n", label, j, rr(regCP_MEC1_INSTR_PNTR));
	}
}

static int __init fw11_init(void)
{
	struct pci_dev *pdev = NULL;
	resource_size_t bar0_start, bar0_size;
	void __iomem *vram = NULL;
	int i;

	while ((pdev = pci_get_device(AMD_VENDOR_ID, PCI_ANY_ID, pdev))) {
		if ((pdev->class >> 8) == PCI_CLASS_DISPLAY_OTHER)
			break;
	}
	if (!pdev) return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) { pci_dev_put(pdev); return -ENOMEM; }

	bar0_start = pci_resource_start(pdev, 0);
	bar0_size = pci_resource_len(pdev, 0);

	pr_info("fw11: ========================================\n");
	pr_info("fw11: PHASE 11: VRAM SEARCH + VALID NOP PATCH\n");
	pr_info("fw11: ========================================\n");
	pr_info("fw11: BASELINE: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	pr_info("fw11: BAR0: 0x%llX, size=%llu MB\n",
		(u64)bar0_start, (u64)bar0_size >> 20);

	/* ============================================================
	 * TEST A: Search ALL of VRAM BAR0 for firmware
	 * ============================================================ */
	pr_info("fw11: --- TEST A: Full VRAM search ---\n");
	vram = ioremap_wc(bar0_start, bar0_size);
	if (!vram) {
		pr_err("fw11: Failed to map full BAR0\n");
		goto test_b;
	}

	{
		u64 off;
		int found = 0;
		/* Search for the 4-dword signature: 7C408001 88000000 7C408001 88000000 */
		for (off = 0; off + 16 <= bar0_size; off += 4) {
			u32 d0 = readl(vram + off);
			if (d0 != 0x7C408001)
				continue;
			/* Potential match — check next 3 dwords */
			if (readl(vram + off + 4) == 0x88000000 &&
			    readl(vram + off + 8) == 0x7C408001 &&
			    readl(vram + off + 12) == 0x88000000) {
				u64 fw_base_off = off - 0x449 * 4;
				pr_info("fw11: A VRAM MATCH at BAR0+0x%llX (FW base ~BAR0+0x%llX)\n",
					off, fw_base_off);

				/* Dump context */
				{
					int d;
					u64 base = off - 4; /* 0x448 relative to sig start at 0x449 */
					for (d = 0; d < 9; d++) {
						pr_info("fw11: A VRAM[0x%X] = 0x%08X\n",
							0x448 + d, readl(vram + base + d * 4));
					}
				}

				found++;
				if (found >= 8) break;
			}

			if ((off & 0xFFFFFFF) == 0 && off > 0)
				pr_info("fw11: A ...searched %llu MB\n", off >> 20);
		}

		if (found == 0) {
			pr_info("fw11: A No firmware pattern in VRAM (%llu MB searched)\n",
				bar0_size >> 20);

			/* Also try searching for just 0x88000000 pairs */
			found = 0;
			for (off = 0; off + 8 <= bar0_size && found < 5; off += 4) {
				if (readl(vram + off) == 0x88000000 &&
				    readl(vram + off + 4) == 0x88000000) {
					pr_info("fw11: A branch-self PAIR at BAR0+0x%llX\n", off);
					found++;
				}
			}
			if (!found)
				pr_info("fw11: A No branch-self pairs in VRAM at all\n");
		} else {
			/* Found firmware in VRAM! Try patching it */
			/* Re-find the exact offset of 0x44C */
			for (off = 0; off + 16 <= bar0_size; off += 4) {
				if (readl(vram + off) == 0x7C408001 &&
				    readl(vram + off + 4) == 0x88000000 &&
				    readl(vram + off + 8) == 0x7C408001 &&
				    readl(vram + off + 12) == 0x88000000) {
					/* off = 0x449*4 relative to FW base */
					/* 0x44C*4 = off + (0x44C-0x449)*4 = off + 12 */
					u64 patch_off = off + 12;
					u32 orig = readl(vram + patch_off);

					pr_info("fw11: A Patching VRAM at BAR0+0x%llX (was 0x%08X)\n",
						patch_off, orig);

					/* Halt */
					wr(regCP_MEC_CNTL, MEC_ME1_HALT);
					udelay(500);

					/* Patch with valid NOP */
					writel(RS64_NOP, vram + patch_off);
					wmb();
					pr_info("fw11: A VRAM readback=0x%08X\n",
						readl(vram + patch_off));

					/* DC + IC invalidate */
					wr(regCP_MEC_DC_OP_CNTL, 0x00000001);
					udelay(500);
					wr(regCP_CPC_IC_OP_CNTL, 0x00000001);
					udelay(500);

					/* Pipe reset */
					wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
					udelay(2000);
					wr(regCP_MEC_CNTL, MEC_ME1_HALT);
					udelay(500);
					pr_info("fw11: A after reset: PC=0x%04X\n",
						rr(regCP_MEC1_INSTR_PNTR));

					wr(regCP_MEC_CNTL, 0);
					mdelay(20);
					sample_pc("A", 8);

					/* Restore */
					wr(regCP_MEC_CNTL, MEC_ME1_HALT);
					udelay(500);
					writel(orig, vram + patch_off);
					wmb();
					wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
					udelay(2000);
					wr(regCP_MEC_CNTL, MEC_ME1_HALT);
					udelay(500);
					wr(regCP_MEC_CNTL, 0);
					mdelay(50);
					pr_info("fw11: A RESTORED: PC=0x%04X\n",
						rr(regCP_MEC1_INSTR_PNTR));
					break;
				}
			}
		}
	}
	if (vram) { iounmap(vram); vram = NULL; }

test_b:
	/* ============================================================
	 * TEST B: Patch ALL system RAM copies with valid NOP
	 * ============================================================ */
	pr_info("fw11: --- TEST B: Patch ALL RAM copies with NOP ---\n");

	/* Save all originals */
	for (i = 0; i < N_RAM_COPIES; i++) {
		u64 addr = ram_copies[i] + 0x44C * 4;
		saved_44c[i] = read_phys(addr);
		pr_info("fw11: B copy[%d] 0x%llX: FW[0x44C]=0x%08X\n",
			i, ram_copies[i], saved_44c[i]);
	}

	/* Halt */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);

	/* Patch ALL copies with NOP */
	for (i = 0; i < N_RAM_COPIES; i++) {
		u64 addr = ram_copies[i] + 0x44C * 4;
		write_phys_flush(addr, RS64_NOP);
	}
	pr_info("fw11: B patched all %d copies with NOP 0x%08X\n",
		N_RAM_COPIES, RS64_NOP);

	/* Verify */
	for (i = 0; i < N_RAM_COPIES; i++) {
		u64 addr = ram_copies[i] + 0x44C * 4;
		pr_info("fw11: B verify copy[%d] = 0x%08X\n", i, read_phys(addr));
	}

	/* DC + IC invalidate */
	wr(regCP_MEC_DC_OP_CNTL, 0x00000001);
	udelay(500);
	wr(regCP_CPC_IC_OP_CNTL, 0x00000001);
	udelay(500);

	/* Pipe reset */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	pr_info("fw11: B after reset: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	wr(regCP_MEC_CNTL, 0);
	mdelay(20);
	sample_pc("B", 8);

	/* Restore all */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	for (i = 0; i < N_RAM_COPIES; i++) {
		u64 addr = ram_copies[i] + 0x44C * 4;
		write_phys_flush(addr, saved_44c[i]);
	}
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	wr(regCP_MEC_CNTL, 0);
	mdelay(50);
	pr_info("fw11: B RESTORED: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* ============================================================
	 * TEST C: Simple Phase-3-style — just FW_PHYS + NOP + pipe reset
	 * No DC/IC invalidation — just the basics that worked in Phase 3
	 * ============================================================ */
	pr_info("fw11: --- TEST C: Phase-3-style simple patch ---\n");
	{
		u64 addr = FW_PHYS + 0x44C * 4;
		u32 orig = read_phys(addr);

		/* Halt */
		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);

		/* Patch with NOP (MEC should fall through to 0x44D) */
		write_phys_flush(addr, RS64_NOP);
		pr_info("fw11: C patched FW_PHYS[0x44C] = 0x%08X (was 0x%08X)\n",
			read_phys(addr), orig);

		/* Just pipe reset + IC invalidate (bit 27) — Phase 3 style */
		wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
		udelay(2000);
		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);
		pr_info("fw11: C after reset: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

		wr(regCP_MEC_CNTL, 0);
		mdelay(20);
		sample_pc("C", 8);

		/* Restore */
		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);
		write_phys_flush(addr, orig);
		wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
		udelay(2000);
		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);
		wr(regCP_MEC_CNTL, 0);
		mdelay(50);
		pr_info("fw11: C RESTORED: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	}

	/* ============================================================
	 * TEST D: Patch ALL copies with branch-to-0x44A (0x88FFFFFE)
	 * If any copy is the real one, MEC should spin at 0x44A not 0x44C
	 * ============================================================ */
	pr_info("fw11: --- TEST D: ALL copies branch(-2) ---\n");
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);

	for (i = 0; i < N_RAM_COPIES; i++) {
		u64 addr = ram_copies[i] + 0x44C * 4;
		write_phys_flush(addr, 0x88FFFFFE);  /* branch to PC-2 = 0x44A */
	}

	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	pr_info("fw11: D after reset: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	wr(regCP_MEC_CNTL, 0);
	mdelay(20);
	sample_pc("D", 8);

	/* Restore */
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	for (i = 0; i < N_RAM_COPIES; i++) {
		u64 addr = ram_copies[i] + 0x44C * 4;
		write_phys_flush(addr, saved_44c[i]);
	}
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
	udelay(2000);
	wr(regCP_MEC_CNTL, MEC_ME1_HALT);
	udelay(500);
	wr(regCP_MEC_CNTL, 0);
	mdelay(50);
	pr_info("fw11: D RESTORED: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* ============================================================
	 * TEST E: Create NEW spin at 0x44E — patch ALL copies
	 * Replace 0x44A with NOP, 0x44C with NOP, 0x44E with branch-self
	 * If ANY copy is real, MEC should end up at 0x44E
	 * ============================================================ */
	pr_info("fw11: --- TEST E: ALL copies NOP+NOP+spin@0x44E ---\n");
	{
		u32 saved_44a[N_RAM_COPIES], saved_44e[N_RAM_COPIES];

		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);

		for (i = 0; i < N_RAM_COPIES; i++) {
			saved_44a[i] = read_phys(ram_copies[i] + 0x44A * 4);
			saved_44e[i] = read_phys(ram_copies[i] + 0x44E * 4);

			write_phys_flush(ram_copies[i] + 0x44A * 4, RS64_NOP);
			write_phys_flush(ram_copies[i] + 0x44C * 4, RS64_NOP);
			write_phys_flush(ram_copies[i] + 0x44E * 4, RS64_BRANCH_SELF);
		}

		wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
		udelay(2000);
		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);
		pr_info("fw11: E after reset: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

		wr(regCP_MEC_CNTL, 0);
		mdelay(20);
		sample_pc("E", 8);

		/* Restore */
		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);
		for (i = 0; i < N_RAM_COPIES; i++) {
			write_phys_flush(ram_copies[i] + 0x44A * 4, saved_44a[i]);
			write_phys_flush(ram_copies[i] + 0x44C * 4, saved_44c[i]);
			write_phys_flush(ram_copies[i] + 0x44E * 4, saved_44e[i]);
		}
		wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
		udelay(2000);
		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);
		wr(regCP_MEC_CNTL, 0);
		mdelay(50);
		pr_info("fw11: E RESTORED: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	}

	pr_info("fw11: FINAL: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	pr_info("fw11: ========================================\n");
	pr_info("fw11: PHASE 11 COMPLETE\n");
	pr_info("fw11: ========================================\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw11_exit(void) {}
module_init(fw11_init);
module_exit(fw11_exit);
