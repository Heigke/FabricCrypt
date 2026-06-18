/*
 * fw17f_diag.c — Focused MEC BO search at deep offsets
 *
 * Previous scans found GPU VA-like values (0x00000020XXXXXXXX) clustered
 * around offset 0xA0000 (640KB) from ddev. The gfx.mec struct should be
 * nearby. Dump the region in detail, looking for the mec pattern:
 *   {bo_ptr, gpu_va, bo_ptr, gpu_va, bo_ptr, gpu_va, u32, u32, u32, ...}
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>

MODULE_LICENSE("GPL");

static void __iomem *mmio;
static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }

static int __init fw17f_init(void)
{
	struct pci_dev *pdev;
	void *adev;
	u64 *scan;
	int i;

	pdev = pci_get_device(0x1002, 0x1586, NULL);
	if (!pdev)
		return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	adev = (void *)pci_get_drvdata(pdev) - 16;
	if (!pci_get_drvdata(pdev))
		goto out;

	pr_info("fw17f: adev = %px, PC = 0x%04X\n", adev, rr(0x021A8));

	/*
	 * Dump the region around 0x9F000-0xA5000 from adev.
	 * Look for patterns of {kernel_ptr, small_value} pairs.
	 */
	scan = (u64 *)adev;

	/* First: find all GPU VA values (upper 32 = 0x20) in 0x90000-0xB0000 range */
	pr_info("fw17f: GPU VA scan 0x90000-0xB0000:\n");
	for (i = 0x90000/8; i < 0xB0000/8; i++) {
		u64 val;
		if (copy_from_kernel_nofault(&val, &scan[i], sizeof(val)))
			continue;
		if ((val >> 32) == 0x20ULL && val != 0) {
			u64 prev = 0;
			copy_from_kernel_nofault(&prev, &scan[i-1], sizeof(prev));
			pr_info("fw17f:   [0x%06lX] = 0x%016llX (prev=0x%016llX%s)\n",
				(long)i * 8, val, prev,
				(prev >= 0xffff800000000000ULL &&
				 prev <= 0xffffffffffff0000ULL) ? " KPTR" : "");
		}
	}

	/* Also scan for value 267904 (0x41600) or 270336 (0x42000) in 0x00-0x200000 range */
	pr_info("fw17f: Scanning for FW size values (267904/270336)...\n");
	for (i = 0; i < 0x200000/8; i++) {
		u64 val;
		if (copy_from_kernel_nofault(&val, &scan[i], sizeof(val)))
			continue;
		if (val == 267904ULL || val == 270336ULL ||
		    val == 0x41600ULL || val == 0x42000ULL) {
			pr_info("fw17f:   FW_SIZE at [0x%06lX] = 0x%llX (%llu)\n",
				(long)i * 8, val, val);
		}
	}

	/* Dump detailed region around the GPU VA cluster */
	pr_info("fw17f: Detailed dump 0xA0100-0xA0300:\n");
	for (i = 0xA0100/8; i < 0xA0300/8; i++) {
		u64 val;
		if (copy_from_kernel_nofault(&val, &scan[i], sizeof(val)))
			continue;
		pr_info("fw17f:   [0x%06lX] = 0x%016llX\n", (long)i * 8, val);
	}

	/* Also look for num_mec/num_pipe pattern: {1-2, 4, 4-8} as u32 values
	 * after GPU VA values. Check around 0xA0000-0xA4000 */
	pr_info("fw17f: Looking for {bo,va,bo,va,bo,va, small_ints} pattern:\n");
	for (i = 0x9F000/8; i < 0xA5000/8; i++) {
		u64 v[8];
		int j;
		int has_kptr = 0;
		int has_small = 0;

		for (j = 0; j < 8; j++) {
			if (copy_from_kernel_nofault(&v[j], &scan[i+j], sizeof(v[j])))
				goto skip;
		}

		/* Check if positions 0,2,4 have kernel ptrs and
		 * position 6 has small values (num_mec etc) */
		for (j = 0; j < 6; j += 2) {
			if (v[j] >= 0xffff800000000000ULL &&
			    v[j] <= 0xffffffffffff0000ULL &&
			    v[j] != 0xffffffffffffffffULL)
				has_kptr++;
		}
		/* v[6] should contain two u32s: num_mec (1-2) and num_pipe (4) */
		{
			u32 lo = (u32)v[6];
			u32 hi = (u32)(v[6] >> 32);
			if (lo >= 1 && lo <= 4 && hi >= 1 && hi <= 8)
				has_small = 1;
		}

		if (has_kptr >= 2 && has_small) {
			u32 lo6 = (u32)v[6], hi6 = (u32)(v[6] >> 32);
			u32 lo7 = (u32)v[7];
			pr_info("fw17f: MEC PATTERN at 0x%lX: ptrs=%d\n",
				(long)i * 8, has_kptr);
			for (j = 0; j < 8; j++)
				pr_info("fw17f:   [%d] = 0x%016llX\n", j, v[j]);
			pr_info("fw17f:   num_mec=%u num_pipe=%u num_queue=%u\n",
				lo6, hi6, lo7);
		}
skip:
		;
	}

	pr_info("fw17f: ========================================\n");

out:
	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw17f_exit(void) {}
module_init(fw17f_init);
module_exit(fw17f_exit);
