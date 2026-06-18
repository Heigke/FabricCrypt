/*
 * find_mec_fw.c — Scan system RAM for MEC RS64 firmware signature
 *
 * The MEC firmware is loaded by PSP into GART-mapped system memory.
 * IC_BASE MC address 0x20681D4000 is not in VRAM. This module scans
 * physical RAM for the known RS64 code signature (first 8 bytes after
 * PSP header strip: 0xC424000B 0x800003B0).
 *
 * Reports physical address of each match. Auto-unloads (-ENODEV).
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/mm.h>
#include <linux/highmem.h>
#include <linux/io.h>
#include <linux/memblock.h>
#include <linux/pfn.h>

MODULE_LICENSE("GPL");

/* First 16 bytes of RS64 code (at file offset 0x200, after PSP header) */
static const u8 mec_sig[] = {
	0x0b, 0x00, 0x24, 0xc4,  /* 0xC424000B */
	0xb0, 0x03, 0x00, 0x80,  /* 0x800003B0 */
	0x8b, 0x00, 0x00, 0xd8,  /* 0xD800008B */
	0x01, 0x00, 0x80, 0x94,  /* 0x94800001 */
};

static int __init find_fw_init(void)
{
	unsigned long pfn;
	unsigned long max_pfn_val = get_num_physpages();
	int found = 0;
	unsigned long checked = 0;

	pr_info("find_mec: scanning %lu pages (%lu MB) for MEC RS64 signature...\n",
		max_pfn_val, max_pfn_val >> 8);

	for (pfn = 0; pfn < max_pfn_val; pfn++) {
		struct page *page;
		u8 *vaddr;
		int offset;

		if (!pfn_valid(pfn))
			continue;

		page = pfn_to_page(pfn);
		if (!page || PageReserved(page))
			continue;

		checked++;

		/* Map the page temporarily */
		vaddr = kmap_local_page(page);
		if (!vaddr)
			continue;

		/* Scan page for signature (firmware could start at any 256-byte boundary) */
		for (offset = 0; offset <= PAGE_SIZE - sizeof(mec_sig); offset += 4) {
			if (memcmp(vaddr + offset, mec_sig, sizeof(mec_sig)) == 0) {
				phys_addr_t phys = PFN_PHYS(pfn) + offset;
				/* Read more context to confirm */
				u32 *dw = (u32 *)(vaddr + offset);
				pr_info("find_mec: *** MATCH at phys 0x%llx (pfn=%lu off=%d) ***\n",
					(u64)phys, pfn, offset);
				pr_info("find_mec:   dwords: %08x %08x %08x %08x %08x %08x %08x %08x\n",
					dw[0], dw[1], dw[2], dw[3], dw[4], dw[5], dw[6], dw[7]);

				/* Check if we can see more of the firmware (next 32 bytes) */
				if (offset + 64 <= PAGE_SIZE) {
					pr_info("find_mec:   +32:    %08x %08x %08x %08x %08x %08x %08x %08x\n",
						dw[8], dw[9], dw[10], dw[11],
						dw[12], dw[13], dw[14], dw[15]);
				}
				found++;
			}
		}

		kunmap_local(vaddr);

		/* Progress every 1M pages (4GB) */
		if ((pfn & 0xFFFFF) == 0 && pfn > 0)
			pr_info("find_mec: scanned %lu/%lu pages (%d matches so far)\n",
				pfn, max_pfn_val, found);
	}

	pr_info("find_mec: done. checked %lu pages, found %d matches\n", checked, found);
	return -ENODEV;
}

static void __exit find_fw_exit(void) {}
module_init(find_fw_init);
module_exit(find_fw_exit);
