/*
 * patch_mec_fw9.c — Phase 9: GPU VA translation + correct-address patching
 *
 * Phase 8 discovered: IC_BASE = 0x20681D4000 (GPU VA).
 * This module:
 *   1. Reads MC_VM aperture registers to map GPU VA space
 *   2. Determines if IC_BASE is in VRAM, GART, or system aperture
 *   3. Translates IC_BASE to physical address
 *   4. Patches firmware at the CORRECT physical location
 *   5. If VRAM: patches through BAR0 at correct offset
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

/* GC registers (BAR5, dword offsets) */
#define regCP_MEC_CNTL              0x0A802
#define regCP_MEC1_INSTR_PNTR      0x021A8
#define regCP_CPC_IC_OP_CNTL       0x0C97A
#define regCP_MEC_DC_OP_CNTL       0x0C90C
#define regGRBM_GFX_CNTL           0x0A900
#define regCP_CPC_IC_BASE_LO       0x0F84C
#define regCP_CPC_IC_BASE_HI       0x0F84D
#define regCP_CPC_IC_BASE_CNTL     0x0F84E

/* MMHUB 3.0 registers (BASE_IDX=0, raw dword offsets) */
#define regMMMC_VM_FB_LOCATION_BASE       0x08EC
#define regMMMC_VM_FB_LOCATION_TOP        0x08ED
#define regMMMC_VM_AGP_TOP                0x08EE
#define regMMMC_VM_AGP_BOT                0x08EF
#define regMMMC_VM_AGP_BASE               0x08F0
#define regMMMC_VM_SYSTEM_APERTURE_LOW    0x08F1
#define regMMMC_VM_SYSTEM_APERTURE_HIGH   0x08F2

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
static void __iomem *vram;  /* BAR0 mapping */
static resource_size_t vram_base;
static resource_size_t vram_size;

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

/* Read GART page table entry for a given GPU VA.
 * GART table is in system memory. We need to find its base. */

static int __init fw9_init(void)
{
	struct pci_dev *pdev = NULL;
	u64 ic_base, fb_base, fb_top, sys_lo, sys_hi, agp_base, agp_bot, agp_top;
	u64 ic_offset_in_vram;
	u32 v;

	while ((pdev = pci_get_device(AMD_VENDOR_ID, PCI_ANY_ID, pdev))) {
		if ((pdev->class >> 8) == PCI_CLASS_DISPLAY_OTHER)
			break;
	}
	if (!pdev) {
		pr_err("fw9: no AMD GPU\n");
		return -ENODEV;
	}

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENOMEM;
	}

	/* Map BAR0 (VRAM) */
	vram_base = pci_resource_start(pdev, 0);
	vram_size = pci_resource_len(pdev, 0);

	pr_info("fw9: ========================================\n");
	pr_info("fw9: PHASE 9: GPU VA TRANSLATION\n");
	pr_info("fw9: ========================================\n");
	pr_info("fw9: BASELINE: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* ============================================================
	 * STEP 1: Read MC_VM aperture registers
	 * ============================================================ */
	pr_info("fw9: --- STEP 1: MC_VM Aperture Map ---\n");

	v = rr(regMMMC_VM_FB_LOCATION_BASE);
	fb_base = (u64)(v & 0x00FFFFFF) << 24;  /* bits [23:0] shifted left 24 */
	pr_info("fw9: FB_LOCATION_BASE raw=0x%08X → 0x%012llX\n", v, fb_base);

	v = rr(regMMMC_VM_FB_LOCATION_TOP);
	fb_top = (u64)(v & 0x00FFFFFF) << 24;
	pr_info("fw9: FB_LOCATION_TOP  raw=0x%08X → 0x%012llX\n", v, fb_top);

	v = rr(regMMMC_VM_SYSTEM_APERTURE_LOW);
	sys_lo = (u64)(v & 0x3FFFFFFF) << 18;  /* bits [29:0] shifted left 18 */
	pr_info("fw9: SYSTEM_APERTURE_LOW  raw=0x%08X → 0x%012llX\n", v, sys_lo);

	v = rr(regMMMC_VM_SYSTEM_APERTURE_HIGH);
	sys_hi = (u64)(v & 0x3FFFFFFF) << 18;
	pr_info("fw9: SYSTEM_APERTURE_HIGH raw=0x%08X → 0x%012llX\n", v, sys_hi);

	v = rr(regMMMC_VM_AGP_BASE);
	agp_base = (u64)(v & 0x00FFFFFF) << 24;
	pr_info("fw9: AGP_BASE raw=0x%08X → 0x%012llX\n", v, agp_base);

	v = rr(regMMMC_VM_AGP_BOT);
	agp_bot = (u64)(v & 0x00FFFFFF) << 24;
	pr_info("fw9: AGP_BOT  raw=0x%08X → 0x%012llX\n", v, agp_bot);

	v = rr(regMMMC_VM_AGP_TOP);
	agp_top = (u64)(v & 0x00FFFFFF) << 24;
	pr_info("fw9: AGP_TOP  raw=0x%08X → 0x%012llX\n", v, agp_top);

	/* Read IC_BASE from pipe 0 */
	grbm_select(1, 0, 0, 0);
	ic_base = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) | rr(regCP_CPC_IC_BASE_LO);
	grbm_select(0, 0, 0, 0);

	pr_info("fw9: IC_BASE = 0x%012llX\n", ic_base);

	/* ============================================================
	 * STEP 2: Determine which aperture IC_BASE falls in
	 * ============================================================ */
	pr_info("fw9: --- STEP 2: Aperture Classification ---\n");

	if (ic_base >= fb_base && ic_base <= fb_top) {
		ic_offset_in_vram = ic_base - fb_base;
		pr_info("fw9: IC_BASE is in FB (VRAM) aperture!\n");
		pr_info("fw9: Offset in VRAM: 0x%llX (%llu MB)\n",
			ic_offset_in_vram, ic_offset_in_vram >> 20);
		pr_info("fw9: BAR0 phys=0x%llX size=%llu MB\n",
			(u64)vram_base, (u64)vram_size >> 20);

		if (ic_offset_in_vram < vram_size) {
			pr_info("fw9: VRAM offset within BAR0 range — can map!\n");

			/* Map enough of BAR0 to reach our firmware */
			vram = ioremap_wc(vram_base + ic_offset_in_vram, 0x10000);
			if (vram) {
				u32 idle_val;
				/* Read FW dword at PC=0x44C offset from IC_BASE */
				/* PC is in dwords, so byte offset = 0x44C * 4 = 0x1130 */
				idle_val = readl(vram + 0x44C * 4);
				pr_info("fw9: VRAM[IC_BASE+0x1130] (PC=0x44C) = 0x%08X\n",
					idle_val);

				/* Also read around it */
				{
					int i;
					for (i = 0x448; i <= 0x450; i++) {
						pr_info("fw9: VRAM[IC_BASE+0x%X] (PC=0x%X) = 0x%08X\n",
							i * 4, i, readl(vram + i * 4));
					}
				}

				/* If we found the idle loop, try patching it */
				if (idle_val == 0x88000000) {
					pr_info("fw9: *** FOUND IDLE LOOP IN VRAM! ***\n");
					pr_info("fw9: --- TEST A: Patch VRAM + IC invalidate ---\n");

					/* Halt MEC */
					wr(regCP_MEC_CNTL, MEC_ME1_HALT);
					udelay(500);

					/* Patch in VRAM */
					writel(0xBF80DEAD, vram + 0x44C * 4);
					wmb();
					pr_info("fw9: A wrote 0xBF80DEAD, readback=0x%08X\n",
						readl(vram + 0x44C * 4));

					/* DC invalidate */
					wr(regCP_MEC_DC_OP_CNTL, 0x00000001);
					udelay(200);
					pr_info("fw9: A DC_OP=0x%08X\n", rr(regCP_MEC_DC_OP_CNTL));

					/* IC invalidate */
					wr(regCP_CPC_IC_OP_CNTL, 0x00000001);
					udelay(200);
					pr_info("fw9: A IC_OP=0x%08X\n", rr(regCP_CPC_IC_OP_CNTL));

					/* Pipe reset */
					wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
					udelay(2000);
					wr(regCP_MEC_CNTL, MEC_ME1_HALT);
					udelay(500);
					pr_info("fw9: A after reset: PC=0x%04X\n",
						rr(regCP_MEC1_INSTR_PNTR));

					/* Unhalt */
					wr(regCP_MEC_CNTL, 0);
					mdelay(20);

					{
						int j;
						for (j = 0; j < 8; j++) {
							mdelay(5);
							pr_info("fw9: A PC[%d]=0x%04X\n",
								j, rr(regCP_MEC1_INSTR_PNTR));
						}
					}

					/* Restore */
					wr(regCP_MEC_CNTL, MEC_ME1_HALT);
					udelay(500);
					writel(0x88000000, vram + 0x44C * 4);
					wmb();
					wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
					udelay(2000);
					wr(regCP_MEC_CNTL, MEC_ME1_HALT);
					udelay(500);
					wr(regCP_MEC_CNTL, 0);
					mdelay(50);
					pr_info("fw9: A RESTORED: PC=0x%04X\n",
						rr(regCP_MEC1_INSTR_PNTR));

					/* ============================================
					 * TEST B: Patch VRAM, NO halt, just IC invalidate
					 * (let MEC see it live)
					 * ============================================ */
					pr_info("fw9: --- TEST B: Live VRAM patch (no halt) ---\n");

					/* Write new spin at 0x44E, break 0x44C */
					writel(0x88000000, vram + 0x44E * 4);  /* spin at 0x44E */
					wmb();
					writel(0xBF800000, vram + 0x44C * 4);  /* s_nop at 0x44C */
					wmb();

					/* IC invalidate only */
					wr(regCP_CPC_IC_OP_CNTL, 0x00000001);
					udelay(500);
					/* Also bit-27 style */
					wr(regCP_MEC_CNTL, MEC_INVALIDATE_ICACHE);
					udelay(500);
					wr(regCP_MEC_CNTL, 0);
					mdelay(20);

					{
						int j;
						for (j = 0; j < 8; j++) {
							mdelay(10);
							pr_info("fw9: B PC[%d]=0x%04X\n",
								j, rr(regCP_MEC1_INSTR_PNTR));
						}
					}

					/* Restore */
					wr(regCP_MEC_CNTL, MEC_ME1_HALT);
					udelay(500);
					writel(0x88000000, vram + 0x44C * 4);
					writel(0x00000000, vram + 0x44E * 4); /* restore orig? */
					wmb();
					wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
					udelay(2000);
					wr(regCP_MEC_CNTL, MEC_ME1_HALT);
					udelay(500);
					wr(regCP_MEC_CNTL, 0);
					mdelay(50);
					pr_info("fw9: B RESTORED: PC=0x%04X\n",
						rr(regCP_MEC1_INSTR_PNTR));
				} else {
					pr_info("fw9: VRAM at idle offset has 0x%08X (not idle loop)\n",
						idle_val);

					/* Dump wider range looking for 0x88000000 pattern */
					{
						int i;
						int found = 0;
						pr_info("fw9: Scanning VRAM region for branch-self...\n");
						for (i = 0; i < 0x1000 && found < 10; i++) {
							u32 val = readl(vram + i * 4);
							if (val == 0x88000000) {
								pr_info("fw9: VRAM[+0x%X] PC=0x%X = 0x88000000\n",
									i * 4, i);
								found++;
							}
						}
						if (!found)
							pr_info("fw9: No branch-self in first 16KB\n");
					}
				}
				iounmap(vram);
			} else {
				pr_info("fw9: Failed to ioremap VRAM at offset\n");
			}
		} else {
			pr_info("fw9: VRAM offset 0x%llX > BAR0 size 0x%llX — unreachable!\n",
				ic_offset_in_vram, (u64)vram_size);

			/* Try mapping just beyond BAR0 using large BAR remap */
			pr_info("fw9: Attempting to map VRAM at full offset...\n");
			vram = ioremap_wc(vram_base, vram_size);
			if (vram) {
				pr_info("fw9: Full BAR0 mapped. Checking beginning for FW pattern...\n");
				/* Search entire mapped VRAM for 0x88000000 pairs */
				{
					unsigned long off;
					int found = 0;
					for (off = 0; off < vram_size && found < 5; off += 4) {
						if (readl(vram + off) == 0x88000000) {
							u32 prev = (off >= 8) ? readl(vram + off - 8) : 0;
							u32 next = (off + 4 < vram_size) ? readl(vram + off + 4) : 0;
							/* Look for paired branch-self (0x44A and 0x44C pattern) */
							if (next == 0x88000000 || prev == 0x88000000) {
								pr_info("fw9: VRAM pair at off=0x%lX: [0x%08X] 0x%08X [0x%08X]\n",
									off, prev, 0x88000000, next);
								found++;
							}
						}
						if ((off & 0xFFFFF) == 0 && off > 0)
							pr_info("fw9: ...scanned %lu MB\n", off >> 20);
					}
					if (!found)
						pr_info("fw9: No firmware pattern in full BAR0\n");
				}
				iounmap(vram);
			}
		}
	} else if (ic_base >= sys_lo && ic_base <= sys_hi) {
		pr_info("fw9: IC_BASE is in SYSTEM APERTURE (1:1 physical map)!\n");
		pr_info("fw9: Physical address = 0x%012llX\n", ic_base);
		/* IC_BASE IS the physical address — patch there directly */
	} else if (ic_base >= agp_bot && ic_base <= agp_top) {
		pr_info("fw9: IC_BASE is in AGP/GART aperture\n");
		pr_info("fw9: Need GART page table to translate\n");
	} else {
		pr_info("fw9: IC_BASE doesn't match any known aperture!\n");
		pr_info("fw9: May be using per-VMID page tables (full VM translation)\n");
	}

	/* ============================================================
	 * STEP 3: Also try reading GART-related registers
	 * ============================================================ */
	pr_info("fw9: --- STEP 3: Additional VM registers ---\n");
	{
		/* Try several MMHUB register ranges that might reveal the
		 * GART table base or page directory base */
		int reg;
		pr_info("fw9: MMHUB regs 0x08E0-0x0910:\n");
		for (reg = 0x08E0; reg <= 0x0910; reg++) {
			u32 val = rr(reg);
			if (val != 0 && val != 0xDEADBEEF)
				pr_info("fw9: [0x%04X] = 0x%08X\n", reg, val);
		}
	}

	/* ============================================================
	 * STEP 4: Check if GART is contiguous — read the GART table itself
	 * Look at mmVM_CONTEXT0_PAGE_TABLE_BASE_ADDR for VMID 0 (kernel)
	 * ============================================================ */
	pr_info("fw9: --- STEP 4: VMID 0 page table base ---\n");
	{
		/* GC VM registers - VMID 0 page table */
		/* For GFX11, these are in the GC block */
		/* regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32 and HI32 */
		/* From gc_11_0_0_offset.h, these should be around 0x5A00+ range */
		int i;
		u32 lo, hi;

		/* Try reading VM context registers in GC block */
		/* GCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32 = 0x5A1E + 0xA000 = 0xF1E for BASE_IDX=1 */
		/* Actually let's just probe a range */
		pr_info("fw9: Probing GC VM registers (0xF000-0xF060):\n");
		for (i = 0xF000; i <= 0xF060; i++) {
			u32 val = rr(i);
			if (val != 0 && val != 0xDEADBEEF)
				pr_info("fw9: GC[0x%04X] = 0x%08X\n", i, val);
		}
	}

	/* ============================================================
	 * STEP 5: Use IOMEM to find firmware — check /proc/iomem ranges
	 * and scan for the firmware via physical address probing
	 * ============================================================ */
	pr_info("fw9: --- STEP 5: Direct physical probing around IC_BASE ---\n");
	{
		/* IC_BASE in VRAM aperture means the firmware might be at
		 * vram_base + (ic_base - fb_base) in PCI space.
		 * Even if > BAR0 visible size, try mapping that PCI phys addr */
		u64 pci_phys;
		if (ic_base >= fb_base) {
			pci_phys = vram_base + (ic_base - fb_base);
			pr_info("fw9: Computed PCI physical: 0x%llX\n", pci_phys);
			pr_info("fw9: (vram_base=0x%llX + offset=0x%llX)\n",
				(u64)vram_base, ic_base - fb_base);

			/* Try ioremap at that address */
			if (pci_phys < vram_base + 0x100000000ULL) { /* sanity: within 4GB of BAR */
				void __iomem *probe = ioremap_wc(pci_phys, 0x10000);
				if (probe) {
					u32 idle_val = readl(probe + 0x44C * 4);
					pr_info("fw9: PCI[0x%llX+0x1130] = 0x%08X\n",
						pci_phys, idle_val);

					/* Dump region */
					{
						int i;
						for (i = 0x448; i <= 0x450; i++) {
							pr_info("fw9: PCI[+0x%X] PC=%X = 0x%08X\n",
								i * 4, i, readl(probe + i * 4));
						}
					}

					if (idle_val == 0x88000000) {
						pr_info("fw9: *** FOUND FIRMWARE VIA PCI PHYS! ***\n");
						pr_info("fw9: --- TEST C: Patch PCI + invalidate ---\n");

						wr(regCP_MEC_CNTL, MEC_ME1_HALT);
						udelay(500);

						writel(0xBF80DEAD, probe + 0x44C * 4);
						wmb();
						pr_info("fw9: C wrote 0xBF80DEAD, readback=0x%08X\n",
							readl(probe + 0x44C * 4));

						/* Full driver-style invalidation */
						wr(regCP_MEC_DC_OP_CNTL, 0x00000001);
						udelay(500);
						wr(regCP_CPC_IC_OP_CNTL, 0x00000001);
						udelay(500);

						/* Pipe reset */
						wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
						udelay(2000);
						wr(regCP_MEC_CNTL, MEC_ME1_HALT);
						udelay(500);
						pr_info("fw9: C after reset: PC=0x%04X\n",
							rr(regCP_MEC1_INSTR_PNTR));

						wr(regCP_MEC_CNTL, 0);
						mdelay(20);
						{
							int j;
							for (j = 0; j < 8; j++) {
								mdelay(5);
								pr_info("fw9: C PC[%d]=0x%04X\n",
									j, rr(regCP_MEC1_INSTR_PNTR));
							}
						}

						/* Restore */
						wr(regCP_MEC_CNTL, MEC_ME1_HALT);
						udelay(500);
						writel(0x88000000, probe + 0x44C * 4);
						wmb();
						wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
						udelay(2000);
						wr(regCP_MEC_CNTL, MEC_ME1_HALT);
						udelay(500);
						wr(regCP_MEC_CNTL, 0);
						mdelay(50);
						pr_info("fw9: C RESTORED: PC=0x%04X\n",
							rr(regCP_MEC1_INSTR_PNTR));
					}
					iounmap(probe);
				} else {
					pr_info("fw9: ioremap failed at 0x%llX\n", pci_phys);
				}
			}
		}
	}

	pr_info("fw9: FINAL: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	pr_info("fw9: ========================================\n");
	pr_info("fw9: PHASE 9 COMPLETE\n");
	pr_info("fw9: ========================================\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw9_exit(void) {}
module_init(fw9_init);
module_exit(fw9_exit);
