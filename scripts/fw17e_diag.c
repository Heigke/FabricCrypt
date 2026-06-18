/*
 * fw17e_diag.c — Find MEC BO via amdgpu_mec pattern matching
 *
 * amdgpu_mec layout:
 *   amdgpu_bo *hpd_eop_obj;        // kernel ptr
 *   u64        hpd_eop_gpu_addr;    // GPU VA (could be 0 or 0x20...)
 *   amdgpu_bo *mec_fw_obj;          // kernel ptr
 *   u64        mec_fw_gpu_addr;     // GPU VA
 *   amdgpu_bo *mec_fw_data_obj;     // kernel ptr or NULL
 *   u64        mec_fw_data_gpu_addr;// GPU VA or 0
 *
 * Pattern: 3 consecutive {kernel_ptr, u64_addr} pairs.
 * At least 2 of the 3 kernel ptrs should be non-NULL.
 * The mec_fw_gpu_addr won't necessarily start with 0x20.
 *
 * adev = drvdata - 16 (offsetof ddev = 16 bytes)
 * We scan the full struct from adev.
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/highmem.h>

MODULE_LICENSE("GPL");

static void __iomem *mmio;

static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }

static int __init fw17e_init(void)
{
	struct pci_dev *pdev;
	void *drvdata, *adev;
	u64 *scan;
	int i, hits = 0;

	pdev = pci_get_device(0x1002, 0x1586, NULL);
	if (!pdev)
		return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	drvdata = pci_get_drvdata(pdev);
	if (!drvdata)
		goto out;

	/* adev = container_of(ddev, struct amdgpu_device, ddev)
	 * ddev is at offset 16 (2 pointers: dev, pdev) */
	adev = (void *)drvdata - 16;
	pr_info("fw17e: drvdata(ddev) = %px, adev = %px\n", drvdata, adev);
	pr_info("fw17e: PC = 0x%04X\n", rr(0x021A8));

	/*
	 * Scan for 3 consecutive {ptr, addr} pairs pattern.
	 * Relaxed matching: ptr must look like kernel address,
	 * addr can be anything (even 0).
	 * We require at least 2 out of 3 ptrs to be valid kernel addresses.
	 */
	pr_info("fw17e: Scanning 4MB for amdgpu_mec pattern...\n");
	scan = (u64 *)adev;
	for (i = 0; i < (4 * 1024 * 1024) / 8 - 6; i++) {
		u64 v[6]; /* 3 pairs: ptr0,addr0, ptr1,addr1, ptr2,addr2 */
		int j, valid_ptrs = 0;

		for (j = 0; j < 6; j++) {
			if (copy_from_kernel_nofault(&v[j], &scan[i+j], sizeof(v[j])))
				goto next;
		}

		/* Count how many of the "ptr" positions look like kernel addresses */
		for (j = 0; j < 6; j += 2) {
			if (v[j] >= 0xffff800000000000ULL &&
			    v[j] <= 0xffffffffffff0000ULL &&
			    v[j] != 0xffffffffffffffffULL)
				valid_ptrs++;
		}

		/* Require at least 2 valid kernel pointers */
		if (valid_ptrs >= 2) {
			/* Additional check: the "addr" positions should NOT
			 * look like kernel addresses (they're GPU VAs or small values) */
			int addr_looks_kern = 0;
			for (j = 1; j < 6; j += 2) {
				if (v[j] >= 0xffff800000000000ULL)
					addr_looks_kern++;
			}
			if (addr_looks_kern >= 2)
				goto next; /* Too many kernel ptrs in addr positions */

			pr_info("fw17e: CANDIDATE at offset 0x%lX from adev (%d ptrs):\n",
				(long)i * 8, valid_ptrs);
			for (j = 0; j < 6; j++)
				pr_info("fw17e:   [%d] = 0x%016llX%s\n",
					j, v[j],
					(j % 2 == 0 && v[j] >= 0xffff800000000000ULL) ?
					" (kernel ptr)" : "");

			/* Try to verify: if v[2] is mec_fw_obj, read its contents
			 * to see if it looks like an amdgpu_bo (with size ~267904) */
			if (v[2] >= 0xffff800000000000ULL &&
			    v[2] != 0xffffffffffffffffULL) {
				u64 bo_dump[16];
				if (!copy_from_kernel_nofault(bo_dump, (void *)v[2], sizeof(bo_dump))) {
					int k;
					pr_info("fw17e:   Checking ptr[2] (%px) as mec_fw_obj:\n",
						(void *)v[2]);
					for (k = 0; k < 16; k++) {
						const char *tag = "";
						if (bo_dump[k] == 267904ULL) tag = " <<< MEC_FW_SIZE!";
						else if (bo_dump[k] == 0x42000ULL) tag = " <<< PAGE_ALIGNED_SIZE";
						else if ((u32)bo_dump[k] == 267904U) tag = " <<< LOW32=FW_SIZE";
						pr_info("fw17e:     [0x%02X] = 0x%016llX%s\n",
							k * 8, bo_dump[k], tag);
					}
				}
			}

			hits++;
			if (hits >= 10)
				break;
		}
next:
		;
	}

	pr_info("fw17e: Found %d candidates\n", hits);

	/* Also try direct approach: read amdgpu_gfx offset by checking
	 * what the struct size is from counting lines in the header */
	{
		/* Known fields from amdgpu.h between ddev (line 863) and gfx (line 1024):
		 * ~160 lines of struct members including huge embedded structs like
		 * amdgpu_display_manager (dm), amdgpu_mode_info, etc.
		 * The offset is unknowable without compilation.
		 *
		 * But we CAN find it by scanning for known values.
		 * We know:
		 *   adev->pdev == pdev
		 *   adev->dev == &pdev->dev
		 *
		 * Verify our adev computation is correct:
		 */
		u64 dev_ptr = 0, pdev_ptr = 0;
		copy_from_kernel_nofault(&dev_ptr, adev, sizeof(dev_ptr));
		copy_from_kernel_nofault(&pdev_ptr, (u64 *)adev + 1, sizeof(pdev_ptr));

		pr_info("fw17e: VERIFY: adev->dev = %px (expected %px)\n",
			(void *)dev_ptr, &pdev->dev);
		pr_info("fw17e: VERIFY: adev->pdev = %px (expected %px)\n",
			(void *)pdev_ptr, pdev);
	}

	pr_info("fw17e: ========================================\n");

out:
	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw17e_exit(void) {}
module_init(fw17e_init);
module_exit(fw17e_exit);
