#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>

MODULE_LICENSE("GPL");

static int __init fw17b_init(void)
{
	struct pci_dev *pdev;
	void *drvdata;
	u64 *scan;
	int i, hits = 0;

	pdev = pci_get_device(0x1002, 0x1586, NULL);
	if (!pdev)
		return -ENODEV;

	drvdata = pci_get_drvdata(pdev);
	pr_info("fw17b: pci_get_drvdata = %px\n", drvdata);

	if (!drvdata) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	/* Scan for any value starting with 0x2068 (upper 16 bits) */
	pr_info("fw17b: Scanning 4MB for 0x2068xxxx patterns...\n");
	scan = (u64 *)drvdata;
	for (i = 0; i < (4 * 1024 * 1024) / 8; i++) {
		u64 val;
		if (copy_from_kernel_nofault(&val, &scan[i], sizeof(val)))
			continue;

		/* Check upper 16 bits match 0x2068 */
		if ((val >> 32) == 0x2068ULL || (val >> 32) == 0x20ULL) {
			if (hits < 20) {
				pr_info("fw17b: [%d] offset 0x%lX: 0x%016llX\n",
					hits, (long)i * 8, val);
			}
			hits++;
		}
	}
	pr_info("fw17b: Found %d matches with 0x2068/0x20 upper bits\n", hits);

	/* Also scan for 0x1D4000 in lower 32 bits (partial GPU VA match) */
	hits = 0;
	for (i = 0; i < (4 * 1024 * 1024) / 8; i++) {
		u64 val;
		if (copy_from_kernel_nofault(&val, &scan[i], sizeof(val)))
			continue;

		if ((val & 0xFFFFFFFFULL) == 0x1D4000ULL) {
			if (hits < 10) {
				pr_info("fw17b: LOW32=0x1D4000 at offset 0x%lX: 0x%016llX\n",
					(long)i * 8, val);
			}
			hits++;
		}
	}
	pr_info("fw17b: Found %d matches with low32=0x1D4000\n", hits);

	/* Also check: maybe the GPU VA is stored in two 32-bit halves */
	hits = 0;
	for (i = 0; i < (4 * 1024 * 1024) / 4; i++) {
		u32 val;
		u32 *s = (u32 *)drvdata;
		if (copy_from_kernel_nofault(&val, &s[i], sizeof(val)))
			continue;

		if (val == 0x681D4000UL) {
			u32 next = 0;
			copy_from_kernel_nofault(&next, &s[i+1], sizeof(next));
			if (hits < 10)
				pr_info("fw17b: 0x681D4000 at dword offset 0x%X, next=0x%08X\n",
					i, next);
			hits++;
		}
	}
	pr_info("fw17b: Found %d matches for 0x681D4000\n", hits);

	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw17b_exit(void) {}
module_init(fw17b_init);
module_exit(fw17b_exit);
