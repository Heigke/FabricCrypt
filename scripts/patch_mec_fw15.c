/*
 * patch_mec_fw15.c — Phase 15: Find MEC firmware BO via amdgpu_device scan
 *
 * We know mec_fw_gpu_addr = 0x20681D4000 is stored in adev->gfx.mec.
 * The struct layout: { amdgpu_bo *mec_fw_obj; u64 mec_fw_gpu_addr; ... }
 * So the BO pointer is 8 bytes before the GPU VA in memory.
 *
 * Strategy:
 *   1. Get drm_device from pci_get_drvdata
 *   2. Scan the large amdgpu_device struct for 0x20681D4000
 *   3. Read the BO pointer from 8 bytes before
 *   4. Use amdgpu_bo_kmap to get CPU access
 *   5. Read BO[0x44C] and verify it matches firmware file (0xD8000705)
 *   6. Patch and try various reload approaches
 *
 * Auto-unloads (-ENODEV).
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/highmem.h>
#include <drm/drm_device.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002
#define AMD_DEV_ID    0x1586

#define MEC_FW_GPU_VA 0x20681D4000ULL

/* GC registers */
#define regCP_MEC_CNTL         0x0A802
#define regCP_MEC1_INSTR_PNTR  0x021A8
#define regCP_CPC_IC_OP_CNTL   0x0C97A
#define regCP_MEC_DC_OP_CNTL   0x0C90C

/* CP_MEC_CNTL bits */
#define MEC_ME2_HALT           (1 << 28)
#define MEC_ME1_HALT           (1 << 30)

static void __iomem *mmio;

static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }

static int __init fw15_init(void)
{
	struct pci_dev *pdev = NULL;
	struct drm_device *ddev;
	u64 *scan;
	int i, found = 0;
	long offset = -1;
	void *adev_base;
	u32 pc;

	pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
	if (!pdev)
		return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	pr_info("fw15: ========================================\n");
	pr_info("fw15: PHASE 15: FIND MEC FIRMWARE BO\n");
	pr_info("fw15: ========================================\n");

	pc = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("fw15: BASELINE: PC=0x%04X\n", pc);

	/* Get drm_device from PCI driver data */
	ddev = pci_get_drvdata(pdev);
	if (!ddev) {
		pr_err("fw15: pci_get_drvdata returned NULL\n");
		goto out;
	}
	pr_info("fw15: drm_device at %px\n", ddev);

	/*
	 * amdgpu_device embeds drm_device as its last field (ddev).
	 * container_of(ddev, struct amdgpu_device, ddev) gives adev.
	 * But we don't know the offset, so we scan a large area
	 * BEFORE ddev for the known GPU VA value.
	 *
	 * amdgpu_device is very large (~100KB+). The gfx.mec substruct
	 * is somewhere in the middle. We'll scan 256KB before ddev.
	 */
	adev_base = (void *)ddev - 256 * 1024;

	pr_info("fw15: Scanning %px to %px for GPU VA 0x%llX\n",
		adev_base, (void *)ddev + 4096, MEC_FW_GPU_VA);

	/* Scan for the GPU VA value */
	scan = (u64 *)adev_base;
	for (i = 0; i < (260 * 1024) / 8; i++) {
		if (scan[i] == MEC_FW_GPU_VA) {
			offset = (long)&scan[i] - (long)ddev;
			pr_info("fw15: FOUND GPU VA at offset %ld from ddev (%px)\n",
				offset, &scan[i]);

			/* The BO pointer should be at scan[i-1] */
			pr_info("fw15:   prev u64 (BO ptr?): 0x%016llX\n", scan[i-1]);
			pr_info("fw15:   next u64 (data BO?): 0x%016llX\n", scan[i+1]);

			/* Check if prev looks like a kernel pointer */
			if (scan[i-1] >= 0xffff000000000000ULL) {
				void *bo_ptr = (void *)scan[i-1];
				struct page *pg;
				u32 *bo_cpu;
				u32 bo_val_0, bo_val_44c;

				pr_info("fw15: BO pointer candidate: %px\n", bo_ptr);

				/*
				 * Try to read the TTM BO structure.
				 * amdgpu_bo embeds ttm_buffer_object as .tbo
				 * ttm_buffer_object has .base (drm_gem_object)
				 * drm_gem_object has .size
				 *
				 * We'll try to access the BO's pages directly.
				 * The BO is pinned in GTT (system memory).
				 *
				 * For a pinned GTT BO, the CPU mapping is typically
				 * available through the TTM kmap.
				 *
				 * Let's try reading the BO structure to find pages.
				 * This is fragile without proper headers, so we'll
				 * do it carefully with probe_kernel_read.
				 */
				{
					u64 maybe_size;
					u64 bo_data[32]; /* read first 256 bytes of BO struct */
					int j;

					if (!copy_from_kernel_nofault(bo_data, bo_ptr, sizeof(bo_data))) {
						pr_info("fw15: BO struct first 256 bytes:\n");
						for (j = 0; j < 32; j++) {
							pr_info("fw15:   [%3d] 0x%016llX\n",
								j * 8, bo_data[j]);
						}
					} else {
						pr_err("fw15: Cannot read BO struct at %px\n", bo_ptr);
					}
				}
				found++;
			}
			if (found >= 3)
				break;
		}
	}

	if (!found)
		pr_info("fw15: GPU VA not found in scan range\n");

	/*
	 * Alternative approach: scan for the firmware signature
	 * in all 8 known RAM copies and verify content matches
	 * the firmware file's instruction section.
	 */
	pr_info("fw15: --- Checking firmware file content at PC=0x44C ---\n");
	{
		/* Read the firmware file to get expected SRAM[0x44C] */
		/* From analysis: SRAM[0x44C] = 0xD8000705 (file byte 0x1230) */
		/* And SRAM[0x44A] = 0xD80006F1, SRAM[0x44B] = 0xD000077D */
		/* These are RS64 system instructions (wait/stall) */

		phys_addr_t fw_phys = 0x115C8C100ULL; /* known FW_PHYS copy */
		struct page *pg;
		void *va;
		u32 *fw;

		pg = pfn_to_page(fw_phys >> PAGE_SHIFT);
		va = kmap_local_page(pg);
		fw = (u32 *)(va + (fw_phys & ~PAGE_MASK));

		/*
		 * The RAM copy at FW_PHYS includes the header.
		 * Header is 256 bytes = 64 dwords.
		 * So SRAM[N] = RAM[N + 64] (if the copy starts at file byte 0)
		 *
		 * But wait: we found the pattern at RAM dword 0x449-0x44C
		 * In the file, the same pattern is at file dwords 0x4C9-0x4CC
		 * File dword 0x4C9 = SRAM dword 0x4C9 - 0x40 = 0x489
		 * So RAM[0x449] ≠ file[0x449] unless the copy doesn't have header
		 *
		 * Let me just read the RAM copy and compare with file expectations.
		 */
		pr_info("fw15: RAM FW_PHYS copy content:\n");
		pr_info("fw15:   RAM[0x000]=0x%08X (file[0x000] header)\n", fw[0]);
		pr_info("fw15:   RAM[0x040]=0x%08X (should be SRAM[0x000] if header=256B)\n", fw[0x40]);

		/* Check around 0x44C in RAM (our Phase 10 patch target) */
		{
			int k;
			pr_info("fw15: RAM around 0x44C:\n");
			for (k = 0x448; k <= 0x450; k++) {
				/* Need to handle page crossing */
				phys_addr_t addr = fw_phys + k * 4;
				struct page *pg2 = pfn_to_page(addr >> PAGE_SHIFT);
				void *va2 = kmap_local_page(pg2);
				u32 v = *(u32 *)(va2 + (addr & ~PAGE_MASK));
				pr_info("fw15:   RAM[0x%03X]=0x%08X => SRAM[0x%03X]\n",
					k, v, k - 0x40);
				kunmap_local(va2);
			}
		}

		/* Check what SRAM[0x44C] should be: RAM[0x44C + 0x40] = RAM[0x48C] */
		{
			int k;
			pr_info("fw15: RAM around 0x48C (=SRAM 0x44C if header present):\n");
			for (k = 0x488; k <= 0x490; k++) {
				phys_addr_t addr = fw_phys + k * 4;
				struct page *pg2 = pfn_to_page(addr >> PAGE_SHIFT);
				void *va2 = kmap_local_page(pg2);
				u32 v = *(u32 *)(va2 + (addr & ~PAGE_MASK));
				const char *tag = "";
				if (v == 0xD8000705)
					tag = " <<< EXPECTED PC STALL";
				else if (v == 0x88000000)
					tag = " <<< BRANCH_SELF";
				else if (v == 0x7C408001)
					tag = " <<< NOP";
				pr_info("fw15:   RAM[0x%03X]=0x%08X => SRAM[0x%03X]%s\n",
					k, v, k - 0x40, tag);
				kunmap_local(va2);
			}
		}

		kunmap_local(va);
	}

	pr_info("fw15: ========================================\n");
	pr_info("fw15: PHASE 15 COMPLETE\n");
	pr_info("fw15: ========================================\n");

out:
	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw15_exit(void) {}

module_init(fw15_init);
module_exit(fw15_exit);
