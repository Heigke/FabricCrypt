/*
 * fw17c_diag.c — Find MEC BO by searching for {kernel_ptr, GPU_VA} pairs
 *
 * amdgpu_mec layout: { amdgpu_bo *mec_fw_obj; u64 mec_fw_gpu_addr;
 *                       amdgpu_bo *mec_fw_data_obj; u64 mec_fw_data_gpu_addr; }
 * Look for: [0xffff...., 0x20........] patterns
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>

MODULE_LICENSE("GPL");

static int __init fw17c_init(void)
{
	struct pci_dev *pdev;
	void *drvdata;
	u64 *scan;
	int i, hits = 0;

	pdev = pci_get_device(0x1002, 0x1586, NULL);
	if (!pdev)
		return -ENODEV;

	drvdata = pci_get_drvdata(pdev);
	if (!drvdata) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	pr_info("fw17c: drvdata = %px\n", drvdata);

	/*
	 * Search for {kernel_ptr, GPU_VA} pairs where:
	 *   kernel_ptr in [0xffff000000000000, 0xffffffffffff0000]
	 *   GPU_VA starts with 0x00000020 (upper 32 bits = 0x20)
	 *   GPU_VA is page-aligned (lower 12 bits = 0)
	 *
	 * These represent amdgpu_bo + gpu_addr pairs.
	 */
	pr_info("fw17c: Scanning 4MB for {kernel_ptr, 0x20xxxx000} pairs...\n");
	scan = (u64 *)drvdata;
	for (i = 1; i < (4 * 1024 * 1024) / 8; i++) {
		u64 ptr_val, addr_val;

		if (copy_from_kernel_nofault(&ptr_val, &scan[i-1], sizeof(ptr_val)))
			continue;
		if (copy_from_kernel_nofault(&addr_val, &scan[i], sizeof(addr_val)))
			continue;

		/* Check pattern: kernel pointer followed by GPU VA */
		if (ptr_val >= 0xffff800000000000ULL &&
		    ptr_val <= 0xffffffffffff0000ULL &&
		    (addr_val >> 32) == 0x20ULL &&
		    (addr_val & 0xFFF) == 0) {

			u64 next_ptr = 0, next_addr = 0;
			copy_from_kernel_nofault(&next_ptr, &scan[i+1], sizeof(next_ptr));
			copy_from_kernel_nofault(&next_addr, &scan[i+2], sizeof(next_addr));

			pr_info("fw17c: HIT[%d] offset=0x%lX:\n", hits, (long)(i-1) * 8);
			pr_info("fw17c:   obj     = %px\n", (void *)ptr_val);
			pr_info("fw17c:   gpu_va  = 0x%llX\n", addr_val);

			/* Check if next pair also matches (data BO) */
			if (next_ptr >= 0xffff800000000000ULL &&
			    next_ptr <= 0xffffffffffff0000ULL &&
			    (next_addr >> 32) == 0x20ULL) {
				pr_info("fw17c:   data_obj= %px  <<<< DOUBLE PAIR!\n",
					(void *)next_ptr);
				pr_info("fw17c:   data_va = 0x%llX\n", next_addr);
			}

			hits++;
			if (hits >= 30)
				break;
		}
	}
	pr_info("fw17c: Found %d BO-like pairs\n", hits);

	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw17c_exit(void) {}
module_init(fw17c_init);
module_exit(fw17c_exit);
