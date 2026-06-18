/*
 * patch_mec_fw20.c — Phase 20: Access MEC firmware via RS64 autoload path
 *
 * From disassembly of gfx_v11_0_rlc_backdoor_autoload_copy_gfx_ucode:
 *   adev + 0x18878: rs64_enable byte flag
 *   adev + 0x18890: struct firmware *mec_fw
 *   adev + 0x18880: struct firmware *pfp_fw
 *   adev + 0x16938: rlc_autoload_bo kptr (destination buffer)
 *
 * struct firmware: { size_t size; const u8 *data; ... }
 *   offset 0x00: size
 *   offset 0x08: data pointer
 *
 * RS64 firmware header (from fw->data):
 *   offset 0x24: ucode_offset (from start of data)
 *   offset 0x28: ucode_size_bytes
 *   offset 0x2C: data_offset
 *   offset 0x30: data_size_bytes
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002
#define AMD_DEV_ID    0x1586

static void __iomem *mmio;
static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }

static int __init fw20_init(void)
{
	struct pci_dev *pdev = NULL;
	void *adev;
	u8 rs64_enable;
	u64 mec_fw_ptr, pfp_fw_ptr, autoload_kptr;
	u64 fw_size, fw_data_ptr;
	u32 ucode_off, ucode_sz, data_off, data_sz;
	u32 *ucode;

	pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
	if (!pdev)
		return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	pr_info("fw20: ========================================\n");
	pr_info("fw20: PHASE 20: RS64 AUTOLOAD FIRMWARE ACCESS\n");
	pr_info("fw20: ========================================\n");
	pr_info("fw20: PC=0x%04X MEC_CNTL=0x%08X\n",
		rr(0x021A8), rr(0x0A802));

	adev = (void *)pci_get_drvdata(pdev) - 16;
	pr_info("fw20: adev = %px\n", adev);

	/* Read RS64 enable flag */
	if (copy_from_kernel_nofault(&rs64_enable, adev + 0x18878, 1))
		goto out;
	pr_info("fw20: rs64_enable = %u\n", rs64_enable);

	/* Read firmware pointers */
	copy_from_kernel_nofault(&mec_fw_ptr, adev + 0x18890, 8);
	copy_from_kernel_nofault(&pfp_fw_ptr, adev + 0x18880, 8);
	copy_from_kernel_nofault(&autoload_kptr, adev + 0x16938, 8);

	pr_info("fw20: mec_fw      = %px\n", (void *)mec_fw_ptr);
	pr_info("fw20: pfp_fw      = %px\n", (void *)pfp_fw_ptr);
	pr_info("fw20: autoload_bo = %px\n", (void *)autoload_kptr);

	if (mec_fw_ptr < 0xffff800000000000ULL ||
	    mec_fw_ptr == 0xffffffffffffffffULL) {
		pr_err("fw20: mec_fw pointer invalid!\n");
		goto out;
	}

	/* Read struct firmware fields: size at +0, data at +8 */
	if (copy_from_kernel_nofault(&fw_size, (void *)mec_fw_ptr, 8)) {
		pr_err("fw20: Cannot read fw->size\n");
		goto out;
	}
	if (copy_from_kernel_nofault(&fw_data_ptr, (void *)(mec_fw_ptr + 8), 8)) {
		pr_err("fw20: Cannot read fw->data\n");
		goto out;
	}

	pr_info("fw20: fw->size = %llu (0x%llX)\n", fw_size, fw_size);
	pr_info("fw20: fw->data = %px\n", (void *)fw_data_ptr);

	if (fw_data_ptr < 0xffff800000000000ULL) {
		pr_err("fw20: fw->data pointer invalid!\n");
		goto out;
	}

	/* Read RS64 firmware header fields */
	copy_from_kernel_nofault(&ucode_off, (void *)(fw_data_ptr + 0x24), 4);
	copy_from_kernel_nofault(&ucode_sz, (void *)(fw_data_ptr + 0x28), 4);
	copy_from_kernel_nofault(&data_off, (void *)(fw_data_ptr + 0x2C), 4);
	copy_from_kernel_nofault(&data_sz, (void *)(fw_data_ptr + 0x30), 4);

	pr_info("fw20: RS64 header:\n");
	pr_info("fw20:   ucode_offset = 0x%X\n", ucode_off);
	pr_info("fw20:   ucode_size   = %u (0x%X)\n", ucode_sz, ucode_sz);
	pr_info("fw20:   data_offset  = 0x%X\n", data_off);
	pr_info("fw20:   data_size    = %u (0x%X)\n", data_sz, data_sz);

	/* Dump first 64 bytes of firmware header */
	{
		u32 hdr[16];
		int i;
		if (!copy_from_kernel_nofault(hdr, (void *)fw_data_ptr, sizeof(hdr))) {
			pr_info("fw20: FW header dump:\n");
			for (i = 0; i < 16; i++)
				pr_info("fw20:   [0x%02X] = 0x%08X\n", i * 4, hdr[i]);
		}
	}

	/* Read actual ucode content at fw_data_ptr + ucode_off */
	if (ucode_off > 0 && ucode_off < fw_size && ucode_sz > 0) {
		u32 code[16];
		int i;

		ucode = (u32 *)(fw_data_ptr + ucode_off);
		if (!copy_from_kernel_nofault(code, ucode, sizeof(code))) {
			pr_info("fw20: MEC ucode first 16 dwords:\n");
			for (i = 0; i < 16; i++) {
				const char *tag = "";
				if (code[i] == 0xC424000BUL) tag = " <<< FIRST_INSTR";
				if (code[i] == 0x88000000UL) tag = " <<< BRANCH_SELF";
				pr_info("fw20:   [0x%03X] = 0x%08X%s\n", i, code[i], tag);
			}
		}

		/* Read around the stuck PC (0x44C) */
		if (ucode_sz > 0x44C * 4 + 16) {
			u32 stuck[8];
			if (!copy_from_kernel_nofault(stuck,
				    (void *)(fw_data_ptr + ucode_off + 0x44C * 4),
				    sizeof(stuck))) {
				pr_info("fw20: MEC ucode at PC=0x44C:\n");
				for (i = 0; i < 8; i++) {
					const char *tag = "";
					if (stuck[i] == 0x88000000UL) tag = " <<< BRANCH_SELF";
					if (stuck[i] == 0xD8000705UL) tag = " <<< WAIT";
					pr_info("fw20:   [0x%03X] = 0x%08X%s\n",
						0x44C + i, stuck[i], tag);
				}
			}
		}
	}

	/* Also check the autoload buffer */
	if (autoload_kptr >= 0xffff800000000000ULL &&
	    autoload_kptr != 0xffffffffffffffffULL) {
		pr_info("fw20: Autoload buffer accessible at %px\n",
			(void *)autoload_kptr);
		/* The MEC ucode position in autoload buffer depends on
		 * the TOC entries. Just dump first 64 bytes to confirm it's valid. */
		{
			u32 al[16];
			int i;
			if (!copy_from_kernel_nofault(al, (void *)autoload_kptr, sizeof(al))) {
				pr_info("fw20: Autoload buffer start:\n");
				for (i = 0; i < 16; i++)
					pr_info("fw20:   [0x%02X] = 0x%08X\n", i * 4, al[i]);
			}
		}
	} else {
		pr_info("fw20: autoload_kptr is NULL or invalid\n");
	}

	pr_info("fw20: ========================================\n");

out:
	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw20_exit(void) {}

module_init(fw20_init);
module_exit(fw20_exit);
