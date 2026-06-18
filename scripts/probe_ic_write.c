/*
 * probe_ic_write.c — v12: Find PDB0 and walk VMID0 page table to TMR
 *
 * v11 findings:
 *   - gart.ptr (0xFFFFD1B4BFB00000) contains garbage (not root PDB)
 *   - GART flat table has 131K entries, covers only 512MB — TMR is beyond this
 *   - ioremap 0x492EC000 fails (it's System RAM, need phys_to_virt)
 *   - BAR0 = 0x6800000000 (256MB VRAM aperture)
 *   - gart.table_addr = 0x100000 (GPU addr within VRAM)
 *   - TMR PTEs are in multi-level page table (PDB0), NOT flat GART table
 *
 * Strategy:
 *   1. Scan gmc struct +0xB00 to +0x1400 for pdb0 fields (bo ptr, gpu addr, cpu ptr)
 *   2. Look for BAR0 addr 0x6800000000 (aper_base) in gmc
 *   3. Scan first 4MB of VRAM (via BAR0) for page-table-like structures
 *   4. If pdb0 found, walk hierarchy to TMR PTE
 *
 * AUTO-UNLOADS (-ENODEV).
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/mm.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID  0x1002
#define TMR_VA         0x97FF943000ULL
#define GART_START_VA  0x8000000000ULL

/* Walk a 4-level page table from root PDB to TMR PTE */
static void walk_pt(void __iomem *vram_base, u64 pdb_offset, u64 target_va)
{
	u32 idx2, idx1, idx0, pte_idx;
	u64 pde, phys;
	void __iomem *level;

	idx2 = (u32)(target_va >> 39) & 0x1FF;
	idx1 = (u32)(target_va >> 30) & 0x1FF;
	idx0 = (u32)(target_va >> 21) & 0x1FF;
	pte_idx = (u32)(target_va >> 12) & 0x1FF;

	pr_info("z2356v12:   Walk VA=0x%llX: idx2=%u idx1=%u idx0=%u pte=%u\n",
		target_va, idx2, idx1, idx0, pte_idx);

	/* Level 2 (root PDB) */
	level = vram_base + pdb_offset;
	pde = readq(level + idx2 * 8);
	pr_info("z2356v12:   PDB2[%u] = 0x%016llX (V=%d)\n",
		idx2, pde, (u32)(pde & 1));
	if (!(pde & 1)) return;

	/* Check if it's a huge page (leaf PDE) or points to next level */
	phys = pde & 0x0000FFFFFFFFFC00ULL;  /* bits [47:6], 64-byte aligned for PDE */
	pr_info("z2356v12:   → L1 addr=0x%llX\n", phys);

	/* Level 1 */
	if (phys >= 0x10000000ULL) {
		pr_info("z2356v12:   L1 phys too large for BAR0, trying phys_to_virt\n");
		{
			u64 *l1 = (u64 *)phys_to_virt(phys);
			if (copy_from_kernel_nofault(&pde, l1 + idx1, 8) == 0) {
				pr_info("z2356v12:   PDB1[%u] = 0x%016llX\n", idx1, pde);
				if (pde & 1) {
					phys = pde & 0x0000FFFFFFFFFC00ULL;
					pr_info("z2356v12:   → L0 addr=0x%llX\n", phys);
					{
						u64 *l0 = (u64 *)phys_to_virt(phys);
						if (copy_from_kernel_nofault(&pde, l0 + idx0, 8) == 0) {
							pr_info("z2356v12:   PDB0[%u] = 0x%016llX\n", idx0, pde);
							if (pde & 1) {
								phys = pde & 0x0000FFFFFFFFFC00ULL;
								pr_info("z2356v12:   → PTB addr=0x%llX\n", phys);
								{
									u64 *ptb = (u64 *)phys_to_virt(phys);
									u64 pte_val;
									if (copy_from_kernel_nofault(&pte_val, ptb + pte_idx, 8) == 0) {
										pr_info("z2356v12:   *** TMR PTE = 0x%016llX ***\n", pte_val);
										pr_info("z2356v12:   *** phys = 0x%llX ***\n",
											(pte_val >> 12) << 12);
										pr_info("z2356v12:   *** V=%d S=%d TMZ=%d X=%d R=%d W=%d ***\n",
											(u32)(pte_val & 1),
											(u32)((pte_val >> 1) & 1),
											(u32)((pte_val >> 3) & 1),
											(u32)((pte_val >> 4) & 1),
											(u32)((pte_val >> 5) & 1),
											(u32)((pte_val >> 6) & 1));
									}
								}
							}
						}
					}
				}
			}
		}
		return;
	}

	/* If phys fits in BAR0 */
	pde = readq(vram_base + phys + idx1 * 8);
	pr_info("z2356v12:   PDB1[%u] = 0x%016llX\n", idx1, pde);
}

static int __init probe_init(void)
{
	struct pci_dev *pdev = NULL;
	void *drvdata;
	u64 *base;
	int i;
	resource_size_t bar0_start, bar0_len;
	void __iomem *vram = NULL;

	while ((pdev = pci_get_device(AMD_VENDOR_ID, PCI_ANY_ID, pdev))) {
		if ((pdev->class >> 8) == PCI_CLASS_DISPLAY_OTHER)
			break;
	}
	if (!pdev) {
		while ((pdev = pci_get_device(AMD_VENDOR_ID, PCI_ANY_ID, pdev))) {
			if ((pdev->class >> 8) == PCI_CLASS_DISPLAY_VGA)
				break;
		}
	}
	if (!pdev) return -ENODEV;

	drvdata = dev_get_drvdata(&pdev->dev);
	if (!drvdata) {
		pci_dev_put(pdev);
		return -ENODEV;
	}
	base = (u64 *)drvdata;

	bar0_start = pci_resource_start(pdev, 0);
	bar0_len = pci_resource_len(pdev, 0);
	pr_info("z2356v12: === PDB0 SEARCH v12 ===\n");
	pr_info("z2356v12: BAR0: phys=0x%llX len=%llu\n",
		(u64)bar0_start, (u64)bar0_len);

	/* Map first 4MB of VRAM via BAR0 */
	vram = ioremap_wc(bar0_start, 0x400000);
	if (!vram) {
		pr_err("z2356v12: BAR0 ioremap failed\n");
		pci_dev_put(pdev);
		return -ENODEV;
	}

	/* ===== Phase 1: Wide gmc struct scan ===== */
	pr_info("z2356v12: --- Phase 1: gmc struct scan +0xB00 to +0x1400 ---\n");
	{
		for (i = 0xB00/8; i < 0x1400/8; i++) {
			u64 v;
			if (copy_from_kernel_nofault(&v, base + i, 8))
				continue;
			if (v == 0) continue;

			/* Only print interesting values */
			if (v == 0x6800000000ULL) {
				pr_info("z2356v12:   +0x%04X = 0x%016llX  *** APER_BASE (BAR0) ***\n", i * 8, v);
			} else if ((v >> 48) == 0xFFFF) {
				pr_info("z2356v12:   +0x%04X = 0x%016llX KPTR\n", i * 8, v);
			} else if ((v >> 44) == 0xFFFFD) {
				pr_info("z2356v12:   +0x%04X = 0x%016llX VMALLOC\n", i * 8, v);
			} else if (v == 0x8000000000ULL || v == 0x97FFFFFFFFULL) {
				pr_info("z2356v12:   +0x%04X = 0x%016llX GART\n", i * 8, v);
			} else if (v >= 0x10000000ULL && v < 0x800000000ULL &&
				   (v & 0xFFF) == 0) {
				pr_info("z2356v12:   +0x%04X = 0x%016llX PHYS_ALIGNED\n", i * 8, v);
			} else if (v < 0x400000 && v > 0 && (v & 0xFFF) == 0) {
				pr_info("z2356v12:   +0x%04X = 0x%016llX SMALL_ALIGNED (VRAM offset?)\n", i * 8, v);
			}
		}
	}

	/* ===== Phase 2: Read GART table from VRAM ===== */
	pr_info("z2356v12: --- Phase 2: GART table at VRAM+0x100000 ---\n");
	{
		/* gart.table_addr = 0x100000 (from v11 +0x31D0) */
		u64 gart_off = 0x100000;
		pr_info("z2356v12:   Entries at VRAM+0x%llX:\n", gart_off);
		for (i = 0; i < 32; i++) {
			u64 entry = readq(vram + gart_off + i * 8);
			if (entry == 0) continue;
			pr_info("z2356v12:     GART[%d] = 0x%016llX V=%d S=%d phys=0x%llX\n",
				i, entry, (u32)(entry & 1),
				(u32)((entry >> 1) & 1),
				(entry >> 12) << 12);
		}
		/* Also check last entries */
		for (i = 131060; i < 131072; i++) {
			u64 entry = readq(vram + gart_off + i * 8);
			if (entry == 0) continue;
			pr_info("z2356v12:     GART[%d] = 0x%016llX\n", i, entry);
		}
	}

	/* ===== Phase 3: Scan VRAM for PDB-like structures ===== */
	pr_info("z2356v12: --- Phase 3: Scan VRAM for PDB structures ---\n");
	{
		/* PDB should be a page-aligned region where entries look like:
		 * - Some entries are 0 (unmapped)
		 * - Valid entries have bit 0 set
		 * - Physical addresses point to system RAM or VRAM
		 * Look at page boundaries in first 4MB of VRAM */
		int found = 0;
		u64 off;
		for (off = 0; off < 0x400000 && found < 5; off += 0x1000) {
			/* Read first 4 entries at this page */
			u64 e0 = readq(vram + off);
			u64 e1 = readq(vram + off + 8);
			u64 e2 = readq(vram + off + 16);
			u64 e3 = readq(vram + off + 24);

			/* Look for page table pattern:
			 * - Some entries valid (bit 0 = 1)
			 * - Physical addresses in reasonable range (< 0x200000000 = 8GB)
			 * - Not all same value
			 * - At least 2 of 4 entries are valid
			 */
			int valid_count = 0;
			u64 entries[4] = {e0, e1, e2, e3};
			int j;
			for (j = 0; j < 4; j++) {
				if ((entries[j] & 1) && entries[j] != 0xFFFFFFFFFFFFFFFFULL) {
					u64 addr = (entries[j] >> 12) << 12;
					/* For PDE, physical addr in bits [47:6] */
					u64 pde_addr = entries[j] & 0x0000FFFFFFFFFC00ULL;
					if (addr < 0x200000000ULL || pde_addr < 0x200000000ULL)
						valid_count++;
				}
			}

			if (valid_count >= 2 && e0 != e1) {
				found++;
				pr_info("z2356v12:   PDB candidate at VRAM+0x%llX:\n", off);
				for (j = 0; j < 8; j++) {
					u64 e = readq(vram + off + j * 8);
					pr_info("z2356v12:     [%d] = 0x%016llX\n", j, e);
				}
			}
		}
		if (found == 0)
			pr_info("z2356v12:   No PDB candidates found in first 4MB\n");
	}

	/* ===== Phase 4: Check system RAM at 0x492EC000 ===== */
	pr_info("z2356v12: --- Phase 4: System RAM at 0x492EC000 ---\n");
	{
		u64 *sysram = (u64 *)phys_to_virt(0x492EC000ULL);
		pr_info("z2356v12:   phys_to_virt(0x492EC000) = %px\n", sysram);
		for (i = 0; i < 32; i++) {
			u64 v;
			if (copy_from_kernel_nofault(&v, sysram + i, 8))
				continue;
			if (v == 0) continue;
			pr_info("z2356v12:     [%d] = 0x%016llX\n", i, v);
		}
	}

	/* ===== Phase 5: Follow gart.bo → ttm → dma_address ===== */
	pr_info("z2356v12: --- Phase 5: BO → TTM → DMA addr ---\n");
	{
		u64 bo_ptr;
		if (copy_from_kernel_nofault(&bo_ptr, base + 0x31B8/8, 8) == 0 &&
		    (bo_ptr >> 48) == 0xFFFF) {
			u64 *bo = (u64 *)bo_ptr;

			/* ttm_buffer_object starts at offset 0x28 in amdgpu_bo
			 * (after base_object, etc.)
			 * tbo.bdev at +0x00, tbo.type at +0x08, ...
			 * tbo.resource at +0x30 (pointer to ttm_resource)
			 * tbo.ttm at +0x68 or similar
			 *
			 * Let me just scan for pointers and check their targets
			 */
			for (i = 0; i < 48; i++) {
				u64 v;
				if (copy_from_kernel_nofault(&v, bo + i, 8))
					continue;
				if ((v >> 48) == 0xFFFF && v != bo_ptr) {
					/* Follow this pointer */
					u64 target[4];
					int j, ok = 1;
					for (j = 0; j < 4; j++) {
						if (copy_from_kernel_nofault(&target[j],
						    (u64 *)v + j, 8)) {
							ok = 0;
							break;
						}
					}
					if (ok) {
						/* Check if target[0] looks like a DMA addr */
						if (target[0] >= 0x10000000ULL &&
						    target[0] < 0x200000000ULL &&
						    (target[0] & 0xFFF) == 0) {
							pr_info("z2356v12:   bo+0x%X → 0x%llX → DMA? 0x%llX\n",
								i * 8, v, target[0]);
						}
						/* Check if any field is a physical addr */
						for (j = 0; j < 4; j++) {
							if (target[j] >= 0x40000000ULL &&
							    target[j] < 0x200000000ULL &&
							    (target[j] & 0xFFF) == 0) {
								pr_info("z2356v12:   bo+0x%X → [%d] = 0x%llX PHYS?\n",
									i * 8, j, target[j]);
							}
						}
					}
				}
			}
		}
	}

	/* ===== Phase 6: Search deeper in gmc for pdb0 fields ===== */
	pr_info("z2356v12: --- Phase 6: Deeper pdb0 search ---\n");
	{
		/* pdb0_bo, pdb0_gpu_addr, pdb0_ptr are typically together
		 * pdb0_gpu_addr would be a small-ish GPU VA (< 0x100000000)
		 * pdb0_ptr would be a KPTR or VMALLOC
		 * Scan +0xD00 to +0x1200 for (KPTR, small_aligned, KPTR/VMALLOC) triplets
		 */
		for (i = 0xD00/8; i < 0x1200/8; i++) {
			u64 v0, v1, v2;
			if (copy_from_kernel_nofault(&v0, base + i, 8)) continue;
			if (copy_from_kernel_nofault(&v1, base + i + 1, 8)) continue;
			if (copy_from_kernel_nofault(&v2, base + i + 2, 8)) continue;

			/* Pattern: KPTR, aligned_addr, KPTR/VMALLOC */
			if ((v0 >> 48) == 0xFFFF &&
			    v1 > 0 && v1 < 0x100000000ULL && (v1 & 0xFFF) == 0 &&
			    ((v2 >> 48) == 0xFFFF || (v2 >> 44) == 0xFFFFD)) {
				pr_info("z2356v12:   PDB0 candidate at +0x%04X:\n", i * 8);
				pr_info("z2356v12:     bo  = 0x%016llX\n", v0);
				pr_info("z2356v12:     addr= 0x%016llX\n", v1);
				pr_info("z2356v12:     ptr = 0x%016llX\n", v2);

				/* Try to use this as PDB and walk to TMR */
				if (v1 < 0x400000) {
					pr_info("z2356v12:   Attempting PT walk from VRAM+0x%llX...\n", v1);
					walk_pt(vram, v1, TMR_VA);
				}
			}
		}
	}

	iounmap(vram);
	pr_info("z2356v12: === v12 COMPLETE ===\n");
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit probe_exit(void) {}
module_init(probe_init);
module_exit(probe_exit);
