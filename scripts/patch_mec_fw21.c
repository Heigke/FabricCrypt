/*
 * patch_mec_fw21.c — Phase 21: Read actual MEC firmware from fw->data + 0x100
 *
 * From Phase 20:
 *   adev + 0x18890 = struct firmware *mec_fw
 *   fw->size = 263424, fw->data points to blob
 *   Common header: ucode_size=0x40400, ucode_array_offset=0x100
 *   So actual firmware = fw->data + 0x100, 263168 bytes (65792 dwords)
 *
 * This module:
 *   1. Reads the firmware from fw->data + 0x100
 *   2. Verifies first instruction matches SRAM[0] = 0xC424000B
 *   3. Reads around PC=0x44C to find the branch_self
 *   4. Attempts to patch the branch_self to a NOP/continue
 *   5. Reports whether the firmware blob is writable
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/mm.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002
#define AMD_DEV_ID    0x1586

static void __iomem *mmio;
static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }
static void wr(u32 dw_off, u32 val) {
	writel(val, mmio + (u64)dw_off * 4);
	readl(mmio + (u64)dw_off * 4);
}

/* MEC registers */
#define regCP_MEC1_INSTR_PNTR  0x021A8
#define regCP_MEC_CNTL         0x0A802
#define MEC_ME1_HALT           (1 << 30)
#define MEC_ME2_HALT           (1 << 28)

static int __init fw21_init(void)
{
	struct pci_dev *pdev = NULL;
	void *adev;
	u64 mec_fw_ptr, fw_data_ptr;
	u64 fw_size;
	u32 ucode_size, ucode_offset;
	u32 *fw_ucode;
	u32 first_instr, stuck_instr;
	u32 pc_before, pc_after;
	int i;

	pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
	if (!pdev)
		return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	pr_info("fw21: ========================================\n");
	pr_info("fw21: PHASE 21: FIRMWARE BLOB PATCH ATTEMPT\n");
	pr_info("fw21: ========================================\n");

	pc_before = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("fw21: PC=0x%04X MEC_CNTL=0x%08X\n", pc_before, rr(regCP_MEC_CNTL));

	adev = (void *)pci_get_drvdata(pdev) - 16;

	/* Get firmware pointer */
	copy_from_kernel_nofault(&mec_fw_ptr, adev + 0x18890, 8);
	if (mec_fw_ptr < 0xffff800000000000ULL) {
		pr_err("fw21: mec_fw invalid\n");
		goto out;
	}

	copy_from_kernel_nofault(&fw_size, (void *)mec_fw_ptr, 8);
	copy_from_kernel_nofault(&fw_data_ptr, (void *)(mec_fw_ptr + 8), 8);

	/* Read header fields */
	copy_from_kernel_nofault(&ucode_size, (void *)(fw_data_ptr + 0x14), 4);
	copy_from_kernel_nofault(&ucode_offset, (void *)(fw_data_ptr + 0x18), 4);

	pr_info("fw21: fw->data=%px size=%llu ucode_off=0x%X ucode_sz=%u\n",
		(void *)fw_data_ptr, fw_size, ucode_offset, ucode_size);

	fw_ucode = (u32 *)(fw_data_ptr + ucode_offset);

	/* Read and verify first instruction */
	if (copy_from_kernel_nofault(&first_instr, fw_ucode, 4)) {
		pr_err("fw21: Cannot read firmware ucode!\n");
		goto out;
	}
	pr_info("fw21: FW[0x000] = 0x%08X%s\n", first_instr,
		first_instr == 0xC424000BUL ? " <<< MATCHES SRAM[0]!" : " MISMATCH");

	/* Read first 32 dwords */
	{
		u32 code[32];
		if (!copy_from_kernel_nofault(code, fw_ucode, sizeof(code))) {
			for (i = 0; i < 32; i++) {
				const char *tag = "";
				if (code[i] == 0xC424000BUL) tag = " <<< FIRST_INSTR";
				if (code[i] == 0x88000000UL) tag = " <<< BRANCH_SELF";
				if (code[i] == 0xD8000705UL) tag = " <<< WAIT_MEM";
				pr_info("fw21:   [0x%03X] = 0x%08X%s\n", i, code[i], tag);
			}
		}
	}

	/* Read around stuck PC (0x44C) */
	if (ucode_size > 0x44C * 4 + 32) {
		u32 stuck[8];
		if (!copy_from_kernel_nofault(stuck,
			    (void *)((u64)fw_ucode + 0x44C * 4),
			    sizeof(stuck))) {
			pr_info("fw21: FW around PC=0x44C:\n");
			for (i = 0; i < 8; i++) {
				const char *tag = "";
				if (stuck[i] == 0x88000000UL) tag = " <<< BRANCH_SELF";
				if (stuck[i] == 0xD8000705UL) tag = " <<< WAIT_MEM";
				if (stuck[i] == 0xBF800000UL) tag = " <<< NOP";
				pr_info("fw21:   [0x%03X] = 0x%08X%s\n",
					0x44C + i, stuck[i], tag);
			}

			stuck_instr = stuck[0];
		}
	}

	/* Now the key question: can we WRITE to fw->data?
	 * The firmware blob is loaded by request_firmware() which uses
	 * vmalloc. The pages should be writable from kernel context. */
	pr_info("fw21: Attempting write test...\n");

	if (stuck_instr == 0x88000000UL) {
		u32 verify;
		/* Try to write a NOP (0xBF800000) over the branch_self.
		 * But first — this only changes the BLOB, not SRAM.
		 * We need to figure out how to reload from blob to SRAM. */

		/* For now, just verify writability without changing anything dangerous */
		u32 test_val;
		u64 test_addr = (u64)fw_ucode + 0x44C * 4;

		/* Read current value */
		copy_from_kernel_nofault(&test_val, (void *)test_addr, 4);
		pr_info("fw21: Current FW[0x44C] = 0x%08X\n", test_val);

		/* Try writing same value back (safe — no actual change) */
		if (copy_to_kernel_nofault((void *)test_addr, &test_val, 4)) {
			pr_info("fw21: WRITE FAILED — blob is read-only!\n");

			/* Try making it writable */
			{
				struct page *page;
				unsigned long addr = test_addr & PAGE_MASK;

				page = vmalloc_to_page((void *)addr);
				if (page) {
					pr_info("fw21: Page found: %px, flags=0x%lX\n",
						page, page->flags);
				} else {
					pr_info("fw21: vmalloc_to_page failed\n");
				}
			}
		} else {
			pr_info("fw21: WRITE SUCCEEDED — blob is writable!\n");

			/* Read back to verify */
			copy_from_kernel_nofault(&verify, (void *)test_addr, 4);
			pr_info("fw21: Verify FW[0x44C] = 0x%08X\n", verify);
		}
	}

	/* Check how firmware gets to SRAM. For non-RS64 GFX11:
	 * gfx_v11_0_config_mec_cache writes IC_BASE and PRIME_ICACHE registers.
	 * The GPU DMA's firmware from the BO GPU address into MEC SRAM.
	 *
	 * But mec_fw_obj at 0x16850 held 0xFFFFFFFFC1C55380.
	 * Let's also check what the gfx_v11_0_config_mec_cache function
	 * actually reads at that offset — maybe it IS the fw_obj, just
	 * allocated from module memory for some reason. */

	{
		u64 maybe_bo;
		copy_from_kernel_nofault(&maybe_bo, adev + 0x16850, 8);
		pr_info("fw21: adev[0x16850] = %px (config_mec_cache checks this)\n",
			(void *)maybe_bo);

		/* Look a bit before: check 0x16800 as possible fw_obj */
		copy_from_kernel_nofault(&maybe_bo, adev + 0x16800, 8);
		pr_info("fw21: adev[0x16800] = %px\n", (void *)maybe_bo);

		/* And the GPU VA that would follow */
		u64 gpu_va;
		copy_from_kernel_nofault(&gpu_va, adev + 0x16808, 8);
		pr_info("fw21: adev[0x16808] = 0x%llX (possible GPU VA)\n", gpu_va);
	}

	pr_info("fw21: ========================================\n");

out:
	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw21_exit(void) {}

module_init(fw21_init);
module_exit(fw21_exit);
