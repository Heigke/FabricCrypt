/*
 * patch_mec_fw19.c — Phase 19: Scan around 0x16800-0x16A00 for real mec pattern
 *
 * The disassembly confirms 0x16850 but values look wrong.
 * Dump the entire region to find the actual mec struct fields:
 *   - num_mec=1, num_pipe=4, num_queue=4-8
 *   - kernel heap pointers (ffff888.../ffff8c9...)
 *   - GPU VAs (0x20XXXXXXXX or small values)
 *
 * Also try scanning wider range for {1, 4, 4-8} u32 triple.
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/highmem.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002
#define AMD_DEV_ID    0x1586

static void __iomem *mmio;
static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }

static int __init fw19_init(void)
{
	struct pci_dev *pdev = NULL;
	void *adev;
	int i;

	pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
	if (!pdev)
		return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	pr_info("fw19: ========================================\n");
	pr_info("fw19: PHASE 19: SCAN FOR MEC STRUCT PATTERN\n");
	pr_info("fw19: ========================================\n");

	pr_info("fw19: PC=0x%04X MEC_CNTL=0x%08X\n",
		rr(0x021A8), rr(0x0A802));

	adev = (void *)pci_get_drvdata(pdev) - 16;
	pr_info("fw19: adev = %px\n", adev);

	/* Dump raw 8-byte words from 0x16800 to 0x16A00 (512 bytes) */
	pr_info("fw19: RAW DUMP 0x16800-0x16A00:\n");
	for (i = 0; i < 64; i++) {
		u64 val;
		unsigned long off = 0x16800 + i * 8;
		if (copy_from_kernel_nofault(&val, adev + off, 8))
			continue;
		pr_info("fw19:   [0x%05lX] = 0x%016llX%s%s%s\n",
			off, val,
			(val >= 0xffff800000000000ULL &&
			 val <= 0xffffffffffff0000ULL &&
			 val != 0xffffffffffffffffULL) ? " KPTR" : "",
			(val >= 1 && val <= 16) ? " SMALL" : "",
			val == 0 ? " ZERO" : "");
	}

	/* Also dump 0x16A00-0x16C00 */
	pr_info("fw19: RAW DUMP 0x16A00-0x16C00:\n");
	for (i = 0; i < 64; i++) {
		u64 val;
		unsigned long off = 0x16A00 + i * 8;
		if (copy_from_kernel_nofault(&val, adev + off, 8))
			continue;
		pr_info("fw19:   [0x%05lX] = 0x%016llX%s%s%s\n",
			off, val,
			(val >= 0xffff800000000000ULL &&
			 val <= 0xffffffffffff0000ULL &&
			 val != 0xffffffffffffffffULL) ? " KPTR" : "",
			(val >= 1 && val <= 16) ? " SMALL" : "",
			val == 0 ? " ZERO" : "");
	}

	/* Scan wider region (0x15000-0x1A000) for the {num_mec, num_pipe} pattern.
	 * We're looking for two consecutive u32s where first is 1-2 and second is 2-8.
	 * Preceded by 3 pairs of {kernel_ptr, u64_value}. */
	pr_info("fw19: PATTERN SCAN 0x15000-0x1A000:\n");
	for (i = 0; i < (0x1A000 - 0x15000) / 4; i++) {
		u32 v32[2];
		unsigned long off = 0x15000 + i * 4;

		if (copy_from_kernel_nofault(v32, adev + off, 8))
			continue;

		/* Look for num_mec=1-2, num_pipe=2-8 */
		if (v32[0] >= 1 && v32[0] <= 2 &&
		    v32[1] >= 2 && v32[1] <= 8) {
			u64 prev[3]; /* 3 u64s before this u32 pair */
			int j, kptrs = 0;

			if (copy_from_kernel_nofault(prev, adev + off - 24, sizeof(prev)))
				continue;

			for (j = 0; j < 3; j++) {
				if (prev[j] >= 0xffff800000000000ULL &&
				    prev[j] <= 0xffffffffffff0000ULL &&
				    prev[j] != 0xffffffffffffffffULL)
					kptrs++;
			}

			/* Also check for next u32 (num_queue) */
			{
				u32 nq;
				if (!copy_from_kernel_nofault(&nq, adev + off + 8, 4) &&
				    nq >= 1 && nq <= 16 && kptrs >= 1) {
					pr_info("fw19:   MATCH at 0x%05lX: mec=%u pipe=%u queue=%u (kptrs=%d)\n",
						off, v32[0], v32[1], nq, kptrs);
					for (j = 0; j < 3; j++)
						pr_info("fw19:     prev[%d] = 0x%016llX\n", j, prev[j]);
				}
			}
		}
	}

	/* Alternative: scan for GPU VA pattern (0x20XXXXXXXX with preceding kernel pointer)
	 * in 0x16000-0x17000 range */
	pr_info("fw19: GPU_VA SCAN 0x16000-0x17000:\n");
	for (i = 0; i < (0x17000 - 0x16000) / 8; i++) {
		u64 val, prev;
		unsigned long off = 0x16000 + i * 8;

		if (copy_from_kernel_nofault(&val, adev + off, 8))
			continue;
		if ((val >> 32) == 0x20ULL && val != 0) {
			copy_from_kernel_nofault(&prev, adev + off - 8, 8);
			pr_info("fw19:   GPU_VA at 0x%05lX: 0x%016llX (prev=%px%s)\n",
				off, val, (void *)prev,
				(prev >= 0xffff800000000000ULL) ? " KPTR" : "");
		}
	}

	pr_info("fw19: ========================================\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw19_exit(void) {}

module_init(fw19_init);
module_exit(fw19_exit);
