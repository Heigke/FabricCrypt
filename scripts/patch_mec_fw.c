/*
 * patch_mec_fw.c — Phase 2: Patch MEC RS64 firmware in-place in system RAM
 *
 * Strategy:
 *   1. Halt MEC via CP_MEC_CNTL
 *   2. Patch ALL 5 firmware copies at PC offset 0x44C (idle loop)
 *      Replace current instruction with s_nop 0xDEAD (identifiable marker)
 *   3. Attempt IC invalidation via IC_OP_CNTL
 *   4. Read back the instruction pointer and check behavior
 *   5. RESTORE original instruction before exiting
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
#define regGRBM_GFX_CNTL           0x0A900

static void __iomem *mmio;

static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }
static void wr(u32 dw_off, u32 val) {
	writel(val, mmio + (u64)dw_off * 4);
	readl(mmio + (u64)dw_off * 4);
}

/* 5 firmware code copies found by find_mec_fw in system RAM */
static const u64 fw_phys[] = {
	0x115c8c100ULL,  /* pfn 1137804, off 256 */
	0x13420c200ULL,  /* pfn 1262092, off 512 */
	0x28763f200ULL,  /* pfn 2651711, off 512 */
	0x28fc45200ULL,  /* pfn 2686021, off 512 */
	0x2b17c5200ULL,  /* pfn 2824133, off 512 */
};

/* Offset in dwords from firmware code start to the instruction we'll patch */
#define PATCH_DW_OFF    0x44C   /* MEC idle loop PC */
#define MARKER_INSTR    0xBF80DEADUL  /* s_nop with DEAD marker */

static u32 orig_instr[5];
static int patched_count;

static u32 *map_fw_dword(int idx, int dw_off)
{
	u64 phys = fw_phys[idx] + (u64)dw_off * 4;
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

static int __init patch_init(void)
{
	struct pci_dev *pdev = NULL;
	u32 mec_cntl, pc_before, pc_after;
	u32 ic_op;
	int i;

	while ((pdev = pci_get_device(AMD_VENDOR_ID, PCI_ANY_ID, pdev))) {
		if ((pdev->class >> 8) == PCI_CLASS_DISPLAY_OTHER)
			break;
	}
	if (!pdev) {
		pr_err("patch_mec: no AMD GPU\n");
		return -ENODEV;
	}
	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENOMEM;
	}

	pr_info("patch_mec: === PHASE 2: IN-PLACE FIRMWARE PATCH TEST ===\n");

	/* Step 0: Read MEC state before anything */
	mec_cntl = rr(regCP_MEC_CNTL);
	pc_before = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("patch_mec: PRE-HALT: MEC_CNTL=0x%08X PC=0x%04X\n", mec_cntl, pc_before);

	/* Step 1: Halt MEC */
	wr(regCP_MEC_CNTL, mec_cntl | (1 << 30));
	udelay(500);
	pr_info("patch_mec: MEC HALTED: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));

	/* Step 2: Read original instruction from all copies and patch */
	patched_count = 0;
	for (i = 0; i < 5; i++) {
		u32 *ptr = map_fw_dword(i, PATCH_DW_OFF);
		if (!ptr) {
			pr_info("patch_mec: copy[%d] MAP FAILED\n", i);
			continue;
		}
		orig_instr[i] = *ptr;
		pr_info("patch_mec: copy[%d] phys=0x%llX orig[0x%X]=0x%08X\n",
			i, fw_phys[i] + PATCH_DW_OFF * 4, PATCH_DW_OFF, orig_instr[i]);

		/* Write marker instruction */
		*ptr = MARKER_INSTR;
		wmb(); /* ensure write is visible */

		/* Verify write */
		{
			u32 readback = *ptr;
			pr_info("patch_mec: copy[%d] PATCHED → 0x%08X (readback=0x%08X %s)\n",
				i, MARKER_INSTR, readback,
				readback == MARKER_INSTR ? "OK" : "FAIL");
			if (readback == MARKER_INSTR)
				patched_count++;
		}
		unmap_fw(ptr);
	}
	pr_info("patch_mec: patched %d/5 copies\n", patched_count);

	/* Step 3: Try IC invalidation */
	ic_op = rr(regCP_CPC_IC_OP_CNTL);
	pr_info("patch_mec: IC_OP_CNTL before invalidate = 0x%08X\n", ic_op);

	/* Write INVALIDATE_CACHE bit */
	wr(regCP_CPC_IC_OP_CNTL, 0x1);
	udelay(1000);
	ic_op = rr(regCP_CPC_IC_OP_CNTL);
	pr_info("patch_mec: IC_OP_CNTL after invalidate = 0x%08X (COMPLETE=%d)\n",
		ic_op, (ic_op >> 1) & 1);

	/* Step 4: Restart MEC */
	wr(regCP_MEC_CNTL, mec_cntl & ~(1 << 30));
	mdelay(10);
	pc_after = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("patch_mec: POST-RESTART: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), pc_after);

	/* Sample PC a few times to see if MEC is running or stuck */
	{
		int j;
		for (j = 0; j < 5; j++) {
			mdelay(5);
			pr_info("patch_mec: PC sample[%d] = 0x%04X\n",
				j, rr(regCP_MEC1_INSTR_PNTR));
		}
	}

	/* Step 5: RESTORE original instruction in all copies */
	pr_info("patch_mec: RESTORING original instructions...\n");
	for (i = 0; i < 5; i++) {
		u32 *ptr = map_fw_dword(i, PATCH_DW_OFF);
		if (!ptr)
			continue;
		*ptr = orig_instr[i];
		wmb();
		{
			u32 rb = *ptr;
			pr_info("patch_mec: copy[%d] RESTORED → 0x%08X (readback=0x%08X %s)\n",
				i, orig_instr[i], rb,
				rb == orig_instr[i] ? "OK" : "FAIL");
		}
		unmap_fw(ptr);
	}

	/* Re-invalidate IC to load restored code */
	wr(regCP_CPC_IC_OP_CNTL, 0x1);
	udelay(1000);

	/* Check MEC state after restore */
	mdelay(10);
	pr_info("patch_mec: FINAL: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));

	pr_info("patch_mec: === PHASE 2 COMPLETE ===\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit patch_exit(void) {}
module_init(patch_init);
module_exit(patch_exit);
