/*
 * patch_mec_fw17.c — Phase 17: Find MEC firmware BO via 2MB scan
 *
 * amdgpu_device is enormous (potentially 1MB+) due to display_manager
 * and other large embedded structs. Scan 2MB after ddev.
 *
 * Once found:
 *   - Read BO pointer from struct layout
 *   - Access BO pages via TTM (without calling unexported amdgpu_bo_kmap)
 *   - Verify firmware content
 *   - Patch and attempt reload
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
#define regCP_MEC_CNTL         0x0A802

#define MEC_ME2_HALT           (1 << 28)
#define MEC_ME1_HALT           (1 << 30)

static void __iomem *mmio;

static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }
static void wr(u32 dw_off, u32 val) {
	writel(val, mmio + (u64)dw_off * 4);
	readl(mmio + (u64)dw_off * 4);
}

static int __init fw17_init(void)
{
	struct pci_dev *pdev = NULL;
	struct drm_device *ddev;
	u64 *scan;
	int i, found = 0;
	long offset;
	u32 pc;
	#define SCAN_SIZE (2 * 1024 * 1024) /* 2MB */

	pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
	if (!pdev)
		return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	pr_info("fw17: ========================================\n");
	pr_info("fw17: PHASE 17: FIND MEC FW BO (2MB SCAN)\n");
	pr_info("fw17: ========================================\n");

	pc = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("fw17: BASELINE: PC=0x%04X\n", pc);

	ddev = pci_get_drvdata(pdev);
	if (!ddev) {
		pr_err("fw17: pci_get_drvdata returned NULL\n");
		goto out;
	}
	pr_info("fw17: drm_device at %px\n", ddev);

	/*
	 * Scan 2MB after ddev for the GPU VA 0x20681D4000.
	 * Use copy_from_kernel_nofault for safety.
	 */
	pr_info("fw17: Scanning 2MB after ddev for GPU VA 0x%llX...\n", MEC_FW_GPU_VA);

	scan = (u64 *)ddev;
	for (i = 0; i < SCAN_SIZE / 8; i++) {
		u64 val;

		if (copy_from_kernel_nofault(&val, &scan[i], sizeof(val)))
			continue;

		if (val == MEC_FW_GPU_VA) {
			u64 bo_val = 0, data_bo = 0, data_addr = 0;

			offset = (long)&scan[i] - (long)ddev;
			pr_info("fw17: FOUND GPU VA at offset %ld (0x%lX) from ddev\n",
				offset, (unsigned long)offset);

			copy_from_kernel_nofault(&bo_val, &scan[i-1], sizeof(bo_val));
			copy_from_kernel_nofault(&data_bo, &scan[i+1], sizeof(data_bo));
			copy_from_kernel_nofault(&data_addr, &scan[i+2], sizeof(data_addr));

			pr_info("fw17:   [i-1] mec_fw_obj?     = %px\n", (void *)bo_val);
			pr_info("fw17:   [i+0] mec_fw_gpu_addr = 0x%llX\n", val);
			pr_info("fw17:   [i+1] mec_fw_data_obj?= %px\n", (void *)data_bo);
			pr_info("fw17:   [i+2] data_gpu_addr?  = 0x%llX\n", data_addr);

			/*
			 * If we found the BO pointer, try to read the TTM BO
			 * struct to find the underlying pages.
			 *
			 * ttm_buffer_object layout (simplified):
			 *   struct drm_gem_object base; (offset 0)
			 *     ... size at known offset ...
			 *   struct ttm_resource *resource; (holds physical info)
			 *   struct ttm_tt *ttm; (holds page array)
			 *     struct page **pages;
			 *
			 * We'll dump the BO struct and look for recognizable values.
			 */
			if (bo_val >= 0xffff000000000000ULL) {
				u64 bo_data[64]; /* first 512 bytes */
				int j;

				if (!copy_from_kernel_nofault(bo_data, (void *)bo_val, sizeof(bo_data))) {
					pr_info("fw17: BO struct at %px (512 bytes):\n", (void *)bo_val);
					for (j = 0; j < 64; j++) {
						const char *tag = "";
						/* Look for FW size: 267904=0x41600, page-aligned=0x42000 */
						if (bo_data[j] == 0x41600ULL) tag = " <<< FW_SIZE_EXACT";
						else if (bo_data[j] == 0x42000ULL) tag = " <<< FW_SIZE_PAGED";
						else if ((bo_data[j] & 0xFFFFFFFF) == 0x41600) tag = " <<< LOW32=FW_SIZE";
						/* Look for ddev pointer */
						else if (bo_data[j] == (u64)ddev) tag = " <<< DRM_DEV_PTR";
						pr_info("fw17:   BO[0x%03X] = 0x%016llX%s\n",
							j * 8, bo_data[j], tag);
					}

					/*
					 * Try to find ttm->pages.
					 * If BO is in GTT, ttm_tt holds the pages.
					 * We look for a kernel pointer that leads to
					 * a page array, where we can then kmap and read FW.
					 */
					for (j = 0; j < 64; j++) {
						if (bo_data[j] >= 0xffff000000000000ULL &&
						    bo_data[j] != (u64)ddev) {
							/*
							 * This could be ttm_tt or ttm_resource.
							 * Try reading it as a struct with pages.
							 */
							u64 sub[16];
							if (!copy_from_kernel_nofault(sub, (void *)bo_data[j], sizeof(sub))) {
								int k;
								for (k = 0; k < 16; k++) {
									/* A pages array would contain
									 * struct page pointers (0xffffea...) */
									if (sub[k] >= 0xffffea0000000000ULL &&
									    sub[k] <= 0xfffffb0000000000ULL) {
										struct page **pages;
										struct page *pg;
										void *va;
										u32 *fw_data;

										pr_info("fw17: Found page array candidate at BO[0x%X]->sub[%d]=%px\n",
											j*8, k, (void *)sub[k]);

										/* Read page pointer array */
										pages = (struct page **)sub[k];
										if (!copy_from_kernel_nofault(&pg, &pages[0], sizeof(pg))) {
											pr_info("fw17:   pages[0] = %px\n", pg);

											if (pg && (u64)pg >= 0xffffea0000000000ULL) {
												va = kmap_local_page(pg);
												fw_data = (u32 *)va;

												pr_info("fw17:   FW page 0 content:\n");
												pr_info("fw17:     [0x000]=0x%08X\n", fw_data[0]);
												pr_info("fw17:     [0x040]=0x%08X\n", fw_data[0x40]);
												pr_info("fw17:     [0x44C]=0x%08X%s\n", fw_data[0x44C],
													fw_data[0x44C] == 0x88000000 ? " BRANCH_SELF" : "");

												kunmap_local(va);
											}
										}
										goto found_pages;
									}
								}
							}
						}
					}
found_pages:
					;
				}
			}

			found++;
			if (found >= 3)
				break;
		}
	}

	if (!found)
		pr_info("fw17: GPU VA NOT FOUND in 2MB scan!\n");

	pr_info("fw17: ========================================\n");
	pr_info("fw17: PHASE 17 COMPLETE\n");
	pr_info("fw17: ========================================\n");

out:
	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw17_exit(void) {}

module_init(fw17_init);
module_exit(fw17_exit);
