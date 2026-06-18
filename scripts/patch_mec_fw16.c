/*
 * patch_mec_fw16.c — Phase 16: Find MEC firmware BO (corrected scan direction)
 *
 * Phase 15 scanned BEFORE drm_device, but in kernel 6.14 amdgpu,
 * struct drm_device ddev is near the START of amdgpu_device.
 * The gfx.mec substruct (with mec_fw_gpu_addr=0x20681D4000) is
 * hundreds of KB AFTER ddev.
 *
 * Strategy:
 *   1. Get drm_device from pci_get_drvdata
 *   2. Scan 512KB AFTER ddev for 0x20681D4000
 *   3. Read the BO pointer from 8 bytes before the match
 *   4. Use copy_from_kernel_nofault to safely read BO struct
 *   5. If BO found, attempt kmap via function pointer call
 *
 * Also confirms: RAM copy at fw_phys has NO header, so
 * RAM[0x44C] = SRAM[0x44C] = 0x88000000 (branch-self).
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
#define regCP_MEC1_INSTR_PNTR  0x021A8

static void __iomem *mmio;

static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }

static int __init fw16_init(void)
{
	struct pci_dev *pdev = NULL;
	struct drm_device *ddev;
	u64 *scan;
	int i, found = 0;
	long offset;
	u32 pc;

	pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
	if (!pdev)
		return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	pr_info("fw16: ========================================\n");
	pr_info("fw16: PHASE 16: FIND MEC FW BO (SCAN AFTER DDEV)\n");
	pr_info("fw16: ========================================\n");

	pc = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("fw16: BASELINE: PC=0x%04X\n", pc);

	/* Get drm_device from PCI driver data */
	ddev = pci_get_drvdata(pdev);
	if (!ddev) {
		pr_err("fw16: pci_get_drvdata returned NULL\n");
		goto out;
	}
	pr_info("fw16: drm_device at %px\n", ddev);

	/*
	 * In kernel 6.14, struct amdgpu_device has ddev near the start.
	 * The gfx substruct (containing mec) is much later.
	 * Scan 512KB AFTER ddev for the GPU VA value.
	 *
	 * Use copy_from_kernel_nofault for safety in case we go
	 * past the allocation.
	 */
	pr_info("fw16: Scanning %px to %px for GPU VA 0x%llX\n",
		(void *)ddev, (void *)ddev + 512 * 1024, MEC_FW_GPU_VA);

	scan = (u64 *)ddev;
	for (i = 0; i < (512 * 1024) / 8; i++) {
		u64 val;

		if (copy_from_kernel_nofault(&val, &scan[i], sizeof(val)))
			continue; /* page fault — skip */

		if (val == MEC_FW_GPU_VA) {
			offset = (long)&scan[i] - (long)ddev;
			pr_info("fw16: FOUND GPU VA at offset %ld (0x%lX) from ddev (%px)\n",
				offset, (unsigned long)offset, &scan[i]);

			/* The BO pointer should be at scan[i-1] */
			{
				u64 bo_val = 0, data_bo = 0, data_addr = 0;

				copy_from_kernel_nofault(&bo_val, &scan[i-1], sizeof(bo_val));
				copy_from_kernel_nofault(&data_bo, &scan[i+1], sizeof(data_bo));
				copy_from_kernel_nofault(&data_addr, &scan[i+2], sizeof(data_addr));

				pr_info("fw16:   [i-1] mec_fw_obj?     = 0x%016llX\n", bo_val);
				pr_info("fw16:   [i+0] mec_fw_gpu_addr = 0x%016llX\n", val);
				pr_info("fw16:   [i+1] mec_fw_data_obj?= 0x%016llX\n", data_bo);
				pr_info("fw16:   [i+2] data_gpu_addr?  = 0x%016llX\n", data_addr);

				/* Check if bo_val looks like a kernel pointer */
				if (bo_val >= 0xffff000000000000ULL) {
					void *bo_ptr = (void *)bo_val;
					u64 bo_data[32]; /* first 256 bytes of BO struct */
					int j;

					pr_info("fw16: BO pointer candidate: %px\n", bo_ptr);

					if (!copy_from_kernel_nofault(bo_data, bo_ptr, sizeof(bo_data))) {
						pr_info("fw16: BO struct dump (first 256 bytes):\n");
						for (j = 0; j < 32; j++) {
							pr_info("fw16:   BO[%3d] = 0x%016llX\n",
								j * 8, bo_data[j]);
						}

						/*
						 * amdgpu_bo layout:
						 *   struct ttm_buffer_object tbo;
						 *     struct drm_gem_object base; (first field of tbo)
						 *       struct kref refcount; (4 bytes)
						 *       unsigned handle_count;
						 *       struct drm_device *dev;
						 *       ... (many fields)
						 *       size_t size; (offset ~72 in 6.14)
						 *
						 * Let's look for the size field.
						 * MEC firmware is 267904 bytes = 0x41600.
						 * So we should see 0x41600 somewhere in the first
						 * few qwords of the BO struct.
						 */
						for (j = 0; j < 32; j++) {
							if (bo_data[j] == 0x41600ULL ||
							    bo_data[j] == 0x42000ULL || /* page aligned */
							    (bo_data[j] & 0xFFFFFFFF) == 0x41600) {
								pr_info("fw16:   MATCH: BO[%d]=0x%llX looks like FW size\n",
									j * 8, bo_data[j]);
							}
						}
					} else {
						pr_err("fw16: Cannot read BO struct at %px\n", bo_ptr);
					}
				}

				/* Also check if data_bo is a valid BO */
				if (data_bo >= 0xffff000000000000ULL && data_addr > 0x200000000000ULL) {
					pr_info("fw16:   data BO (%px) also looks valid (addr=0x%llX)\n",
						(void *)data_bo, data_addr);
				}
			}
			found++;
			if (found >= 5)
				break;
		}
	}

	if (!found) {
		pr_info("fw16: GPU VA not found in 512KB after ddev\n");
		/* Try also scanning 64KB before ddev */
		pr_info("fw16: Trying 64KB before ddev...\n");
		scan = (u64 *)((void *)ddev - 64 * 1024);
		for (i = 0; i < (64 * 1024) / 8; i++) {
			u64 val;
			if (copy_from_kernel_nofault(&val, &scan[i], sizeof(val)))
				continue;
			if (val == MEC_FW_GPU_VA) {
				offset = (long)&scan[i] - (long)ddev;
				pr_info("fw16: FOUND GPU VA at offset %ld from ddev (BEFORE)\n", offset);
				found++;
				break;
			}
		}
	}

	if (!found)
		pr_info("fw16: GPU VA 0x%llX NOT FOUND in any scan range\n", MEC_FW_GPU_VA);

	pr_info("fw16: ========================================\n");
	pr_info("fw16: PHASE 16 COMPLETE\n");
	pr_info("fw16: ========================================\n");

out:
	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw16_exit(void) {}

module_init(fw16_init);
module_exit(fw16_exit);
