/*
 * patch_mec_fw18.c — Phase 18: Direct BO access via known struct offset
 *
 * From disassembly of gfx_v11_0_config_mec_cache:
 *   cmpq $0x0, 0x16850(%rdi)  => adev->gfx.mec.mec_fw_obj at offset 0x16850
 *   mec_fw_gpu_addr at 0x16858
 *   hpd_eop_obj at 0x16840, hpd_eop_gpu_addr at 0x16848
 *   mec_fw_data_obj at 0x16860, mec_fw_data_gpu_addr at 0x16868
 *
 * adev = pci_get_drvdata(pdev) - 16
 *   (verified: adev->dev and adev->pdev match)
 *
 * Plan:
 *   1. Read all mec struct fields at known offsets
 *   2. If mec_fw_obj is valid, access TTM BO pages
 *   3. Verify firmware content
 *   4. Patch and attempt MEC restart
 *
 * Auto-unloads (-ENODEV).
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/highmem.h>
#include <linux/mm.h>
#include <drm/drm_device.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002
#define AMD_DEV_ID    0x1586

/* Struct offsets from disassembly */
#define OFF_MEC_HPD_EOP_OBJ       0x16840
#define OFF_MEC_HPD_EOP_GPU_ADDR  0x16848
#define OFF_MEC_FW_OBJ            0x16850
#define OFF_MEC_FW_GPU_ADDR       0x16858
#define OFF_MEC_FW_DATA_OBJ       0x16860
#define OFF_MEC_FW_DATA_GPU_ADDR  0x16868
#define OFF_MEC_NUM_MEC           0x16870
#define OFF_MEC_NUM_PIPE          0x16874
#define OFF_MEC_NUM_QUEUE         0x16878

/* GC registers */
#define regCP_MEC1_INSTR_PNTR  0x021A8
#define regCP_MEC_CNTL         0x0A802
#define MEC_ME1_HALT           (1 << 30)
#define MEC_ME2_HALT           (1 << 28)

static void __iomem *mmio;

static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }
static void wr(u32 dw_off, u32 val) {
	writel(val, mmio + (u64)dw_off * 4);
	readl(mmio + (u64)dw_off * 4);
}

static int __init fw18_init(void)
{
	struct pci_dev *pdev = NULL;
	void *adev;
	u64 fw_obj_ptr, fw_gpu_addr;
	u64 hpd_obj_ptr, hpd_gpu_addr;
	u64 data_obj_ptr, data_gpu_addr;
	u32 num_mec, num_pipe, num_queue;
	u32 pc;

	pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
	if (!pdev)
		return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	pr_info("fw18: ========================================\n");
	pr_info("fw18: PHASE 18: DIRECT BO ACCESS VIA OFFSETS\n");
	pr_info("fw18: ========================================\n");

	pc = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("fw18: BASELINE: PC=0x%04X MEC_CNTL=0x%08X\n", pc, rr(regCP_MEC_CNTL));

	adev = (void *)pci_get_drvdata(pdev) - 16;
	pr_info("fw18: adev = %px\n", adev);

	/* Read all mec struct fields */
	copy_from_kernel_nofault(&hpd_obj_ptr, adev + OFF_MEC_HPD_EOP_OBJ, 8);
	copy_from_kernel_nofault(&hpd_gpu_addr, adev + OFF_MEC_HPD_EOP_GPU_ADDR, 8);
	copy_from_kernel_nofault(&fw_obj_ptr, adev + OFF_MEC_FW_OBJ, 8);
	copy_from_kernel_nofault(&fw_gpu_addr, adev + OFF_MEC_FW_GPU_ADDR, 8);
	copy_from_kernel_nofault(&data_obj_ptr, adev + OFF_MEC_FW_DATA_OBJ, 8);
	copy_from_kernel_nofault(&data_gpu_addr, adev + OFF_MEC_FW_DATA_GPU_ADDR, 8);
	copy_from_kernel_nofault(&num_mec, adev + OFF_MEC_NUM_MEC, 4);
	copy_from_kernel_nofault(&num_pipe, adev + OFF_MEC_NUM_PIPE, 4);
	copy_from_kernel_nofault(&num_queue, adev + OFF_MEC_NUM_QUEUE, 4);

	pr_info("fw18: hpd_eop_obj     = %px\n", (void *)hpd_obj_ptr);
	pr_info("fw18: hpd_eop_gpu_addr= 0x%llX\n", hpd_gpu_addr);
	pr_info("fw18: mec_fw_obj      = %px\n", (void *)fw_obj_ptr);
	pr_info("fw18: mec_fw_gpu_addr = 0x%llX\n", fw_gpu_addr);
	pr_info("fw18: mec_fw_data_obj = %px\n", (void *)data_obj_ptr);
	pr_info("fw18: mec_fw_data_gpu_addr = 0x%llX\n", data_gpu_addr);
	pr_info("fw18: num_mec=%u num_pipe=%u num_queue=%u\n",
		num_mec, num_pipe, num_queue);

	/* Verify: if fw_obj_ptr looks like a kernel pointer, read the BO struct */
	if (fw_obj_ptr >= 0xffff800000000000ULL &&
	    fw_obj_ptr != 0xffffffffffffffffULL) {
		u64 bo_dump[64]; /* 512 bytes of BO struct */
		int i;

		pr_info("fw18: Reading BO struct at %px...\n", (void *)fw_obj_ptr);

		if (!copy_from_kernel_nofault(bo_dump, (void *)fw_obj_ptr, sizeof(bo_dump))) {
			/* Dump key fields */
			for (i = 0; i < 20; i++) {
				const char *tag = "";
				/* MEC FW: 267904 (0x41600) bytes, 66 pages */
				if (bo_dump[i] == 267904ULL) tag = " <<< FW SIZE EXACT";
				else if (bo_dump[i] == 270336ULL) tag = " <<< FW SIZE PAGED";
				else if (bo_dump[i] == 66ULL) tag = " <<< PAGE COUNT?";
				pr_info("fw18:   BO[0x%02X] = 0x%016llX%s\n",
					i * 8, bo_dump[i], tag);
			}

			/*
			 * In the BO struct, we need to find the TTM resource
			 * and ttm_tt to access the actual pages.
			 *
			 * amdgpu_bo contains:
			 *   struct ttm_buffer_object tbo; (first member)
			 *     struct drm_gem_object base; (first member of tbo)
			 *       ... size at some offset
			 *     struct ttm_resource *resource;
			 *     struct ttm_tt *ttm;
			 *       struct page **pages; (first or early member of ttm_tt)
			 *
			 * Let's scan for any pointer that leads to a page array.
			 */
			for (i = 0; i < 64; i++) {
				if (bo_dump[i] >= 0xffff800000000000ULL &&
				    bo_dump[i] != 0xffffffffffffffffULL) {
					u64 sub[8];
					if (!copy_from_kernel_nofault(sub, (void *)bo_dump[i], sizeof(sub))) {
						int k;
						for (k = 0; k < 8; k++) {
							if (sub[k] >= 0xffffea0000000000ULL &&
							    sub[k] <= 0xfffffb0000000000ULL) {
								/* This could be a pages array! */
								struct page **pages = (struct page **)sub[k];
								struct page *pg = NULL;

								pr_info("fw18: PAGES candidate: BO[0x%02X]->sub[%d] = %px\n",
									i * 8, k, (void *)sub[k]);

								if (!copy_from_kernel_nofault(&pg, &pages[0], sizeof(pg))) {
									void *va;
									u32 *fw;

									pr_info("fw18:   pages[0] = %px\n", pg);

									if (pg && (u64)pg >= 0xffffea0000000000ULL) {
										va = kmap_local_page(pg);
										fw = (u32 *)va;

										pr_info("fw18:   FW CONTENT:\n");
										pr_info("fw18:     [0x000] = 0x%08X (should be first FW instr)\n", fw[0]);
										pr_info("fw18:     [0x001] = 0x%08X\n", fw[1]);
										pr_info("fw18:     [0x002] = 0x%08X\n", fw[2]);
										pr_info("fw18:     [0x003] = 0x%08X\n", fw[3]);

										/* Check if this looks like MEC firmware.
										 * RAM[0x000] from Phase 15 was 0xC424000B */
										if (fw[0] == 0xC424000BUL)
											pr_info("fw18:     *** CONFIRMED: MEC FW FIRST INSTR! ***\n");

										/* Read SRAM[0x44C] area (page 0 covers 0-1023 dwords) */
										pr_info("fw18:     [0x44A] = 0x%08X\n", fw[0x44A]);
										pr_info("fw18:     [0x44B] = 0x%08X\n", fw[0x44B]);
										pr_info("fw18:     [0x44C] = 0x%08X%s\n", fw[0x44C],
											fw[0x44C] == 0x88000000UL ? " <<< BRANCH_SELF" : "");

										kunmap_local(va);
									}
								}
								goto bo_done;
							}
						}
					}
				}
			}
bo_done:
			;
		} else {
			pr_err("fw18: Cannot read BO struct!\n");
		}
	} else {
		pr_info("fw18: mec_fw_obj is NULL or invalid\n");

		/* Dump raw bytes around the offset to diagnose */
		pr_info("fw18: Raw dump around 0x16840:\n");
		{
			u64 raw[8];
			int j;
			copy_from_kernel_nofault(raw, adev + OFF_MEC_HPD_EOP_OBJ, sizeof(raw));
			for (j = 0; j < 8; j++)
				pr_info("fw18:   [0x%05lX] = 0x%016llX\n",
					(long)(OFF_MEC_HPD_EOP_OBJ + j * 8), raw[j]);
		}
	}

	pr_info("fw18: ========================================\n");
	pr_info("fw18: PHASE 18 COMPLETE\n");
	pr_info("fw18: ========================================\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw18_exit(void) {}

module_init(fw18_init);
module_exit(fw18_exit);
