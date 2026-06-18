/*
 * patch_mec_fw27.c — Phase 27: Deep VRAM scan + RLCG register write
 *
 * Phase 26 found populated VRAM at 3MB and 6-7MB but only scanned
 * 0-1MB for firmware signatures. Fix: scan 0-8MB thoroughly.
 *
 * Also try RLC Gateway (RLCG) for privileged register writes.
 * On GFX11, many GC registers require RLCG indirect access.
 *
 * Also: find GC_HWIP SOC15 base offset from driver to use correct
 * register addresses.
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002
#define AMD_DEV_ID    0x1586

static void __iomem *mmio;
static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }
static void wr(u32 dw_off, u32 val) {
	writel(val, mmio + (u64)dw_off * 4);
	readl(mmio + (u64)dw_off * 4);
}

#define regCP_MEC1_INSTR_PNTR      0x021A8
#define regCP_MEC_CNTL             0x0A802
#define MEC_ME1_HALT               (1 << 30)
#define MEC_ME2_HALT               (1 << 28)
#define regCP_CPC_IC_BASE_LO       0x0C930
#define regCP_CPC_IC_BASE_HI       0x0C931
#define regCP_CPC_IC_OP_CNTL       0x0C932

/* RLCG (RLC Gateway) registers for GFX11 */
#define regRLC_SPM_MC_CNTL                0x4D87
#define regGRBM_GFX_CNTL                  0x0D880

/* RLCG interface registers (GFX11) */
/* The driver uses these for WREG32_SOC15_RLCG */
#define regSCRATCH_REG0                   0x0D840
#define regSCRATCH_REG1                   0x0D841
#define regSCRATCH_REG2                   0x0D842
#define regSCRATCH_REG3                   0x0D843

/* RLC_RLCG_DOORBELL_RANGE — for RLCG writes */
/* In GFX11, the RLCG write sequence is:
 *   1. SCRATCH_REG0 = data
 *   2. SCRATCH_REG1 = target_reg | (flag << 28)
 *   3. Read back SCRATCH_REG1 and poll for flag clear
 *
 * Or in newer implementations:
 *   RLCG uses SCRATCH_REG2 for address, SCRATCH_REG3 for data,
 *   then triggers via SCRATCH_REG1 command word.
 */

/* GFX11 RLCG command bits */
#define RLCG_FLAG_WRITE    0x10000000  /* bit 28 = write request */
#define RLCG_FLAG_READ     0x20000000  /* bit 29 = read request */

/* Common GFX11 GC register offsets from gc_11_0_0_offset.h:
 * regCP_CPC_IC_BASE_LO  = 0x584c (IP offset, before SOC15 base)
 * regCP_CPC_IC_BASE_HI  = 0x584d
 * regCP_CPC_IC_OP_CNTL  = 0x584e
 * regCP_MEC_CNTL        = 0x2818 (IP offset)
 * regCP_MEC_ME1_UCODE_ADDR = 0x282a
 * regCP_MEC_ME1_UCODE_DATA = 0x282b
 *
 * SOC15 address = base_offset + IP_offset
 * For direct MMIO, amdgpu translates this at driver init.
 * Our raw MMIO offsets (0xA802, 0xC930, etc.) are the translated values.
 *
 * Let's try to find the GC base by comparing known register values.
 * If our raw offset for MEC_CNTL = 0xA802 and IP offset = 0x2818,
 * then base = 0xA802 - 0x2818 = 0x7FEA
 */
#define GC_BASE_GUESS  0x7FEA

/* IP offsets from gc_11_0_0_offset.h (not the same as our raw MMIO) */
#define IP_regCP_CPC_IC_BASE_LO   0x584C
#define IP_regCP_CPC_IC_BASE_HI   0x584D
#define IP_regCP_CPC_IC_OP_CNTL   0x584E
#define IP_regCP_MEC_CNTL         0x2818
#define IP_regCP_MEC_ME1_UCODE_ADDR 0x282A
#define IP_regCP_MEC_ME1_UCODE_DATA 0x282B

/* MEC instructions */
#define INST_BRANCH_SELF 0x88000000
#define INST_NOP         0xBF800000

/* Try RLCG write to a register */
static int rlcg_write(u32 ip_offset, u32 data)
{
	u32 cmd;
	int timeout = 1000;

	/* RLCG write sequence for GFX11:
	 * Based on amdgpu_virt_rlcg_reg_rw() */
	wr(regSCRATCH_REG0, data);
	wr(regSCRATCH_REG1, ip_offset | RLCG_FLAG_WRITE);

	/* Poll for completion (flag bit clears) */
	while (timeout-- > 0) {
		cmd = rr(regSCRATCH_REG1);
		if (!(cmd & RLCG_FLAG_WRITE))
			return 0;
		udelay(1);
	}

	pr_info("fw27: RLCG write timeout! SCRATCH1=0x%08X\n", cmd);
	return -1;
}

/* Try RLCG read */
static u32 rlcg_read(u32 ip_offset)
{
	u32 cmd;
	int timeout = 1000;

	wr(regSCRATCH_REG1, ip_offset | RLCG_FLAG_READ);

	while (timeout-- > 0) {
		cmd = rr(regSCRATCH_REG1);
		if (!(cmd & RLCG_FLAG_READ))
			return rr(regSCRATCH_REG0);
		udelay(1);
	}

	pr_info("fw27: RLCG read timeout!\n");
	return 0xDEADDEAD;
}

static int __init fw27_init(void)
{
	struct pci_dev *pdev = NULL;
	resource_size_t bar0_start, bar0_len;
	void __iomem *vram;
	u64 map_size;
	u32 pc, val;

	pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
	if (!pdev)
		return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	bar0_start = pci_resource_start(pdev, 0);
	bar0_len = pci_resource_len(pdev, 0);

	pr_info("fw27: ========================================\n");
	pr_info("fw27: PHASE 27: DEEP VRAM SCAN + RLCG WRITE\n");
	pr_info("fw27: ========================================\n");

	pc = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("fw27: PC=0x%04X\n", pc);

	/* ========================================
	 * SECTION A: Verify GC base offset
	 * ======================================== */
	pr_info("fw27: === SECTION A: GC BASE OFFSET ===\n");
	{
		/* If base = 0x7FEA, then:
		 * IP_MEC_CNTL 0x2818 -> raw 0x7FEA+0x2818 = 0xA802 ✓
		 * IP_IC_BASE_LO 0x584C -> raw 0x7FEA+0x584C = 0xD836
		 * But we've been reading IC_BASE at 0xC930...
		 * So either the base guess is wrong or there are multiple bases.
		 *
		 * Let me verify by checking what's at 0xD836:
		 */
		u32 test1 = rr(GC_BASE_GUESS + IP_regCP_MEC_CNTL);
		u32 test2 = rr(0xA802);  /* our known raw offset */
		pr_info("fw27: GC_BASE_GUESS+MEC_CNTL (0x%X) = 0x%08X\n",
			GC_BASE_GUESS + IP_regCP_MEC_CNTL, test1);
		pr_info("fw27: Raw MEC_CNTL (0xA802) = 0x%08X\n", test2);

		/* Try the IC_BASE at the guessed offset */
		val = rr(GC_BASE_GUESS + IP_regCP_CPC_IC_BASE_LO);
		pr_info("fw27: GC_BASE+IC_BASE_LO (0x%X) = 0x%08X\n",
			GC_BASE_GUESS + IP_regCP_CPC_IC_BASE_LO, val);

		/* Also check: if IC_BASE raw is at 0xC930, what IP offset? */
		/* 0xC930 - 0x7FEA = 0x4946 -- doesn't match 0x584C */
		/* So the GC block might have multiple base segments */

		/* Try scanning for the IC_BASE value (0x07) at different offsets */
		pr_info("fw27: Scanning for IC_BASE value 0x07:\n");
		{
			u32 search_ranges[][2] = {
				{0xC900, 0xCA00},
				{0xD800, 0xD900},
				{0x5800, 0x5900},
			};
			int r, o;
			for (r = 0; r < 3; r++) {
				for (o = search_ranges[r][0];
				     o < search_ranges[r][1]; o++) {
					u32 v = rr(o);
					if (v == 0x07)
						pr_info("fw27:   [0x%05X]=0x07\n", o);
				}
			}
		}
	}

	/* ========================================
	 * SECTION B: Deep VRAM scan (3MB and 6-7MB regions)
	 * ======================================== */
	pr_info("fw27: === SECTION B: DEEP VRAM SCAN ===\n");
	map_size = min_t(u64, bar0_len, 8 * 1024 * 1024);
	vram = ioremap_wc(bar0_start, map_size);
	if (!vram) {
		pr_info("fw27: Failed to map VRAM\n");
		goto skip_vram;
	}

	/* Scan 3MB region thoroughly */
	{
		u64 base = 3 * 1024 * 1024;
		u64 end = 4 * 1024 * 1024;
		u64 off;
		int sig_count = 0;

		pr_info("fw27: --- VRAM 3MB-4MB signatures ---\n");
		for (off = base; off < end && sig_count < 20; off += 4) {
			u32 v = readl(vram + off);
			if (v == INST_BRANCH_SELF) {
				pr_info("fw27:   [0x%llX] BRANCH_SELF!\n", off);
				sig_count++;
			} else if (v == 0xC424000B) {
				pr_info("fw27:   [0x%llX] MEC_FIRST_INSTR!\n", off);
				sig_count++;
			} else if (v == 0x0E6F518F) {
				pr_info("fw27:   [0x%llX] ENCRYPTED_HEADER!\n", off);
				sig_count++;
			}
		}
		if (sig_count == 0)
			pr_info("fw27:   No known signatures\n");

		/* Dump first 16 dwords of 3MB region */
		pr_info("fw27: VRAM[3MB] first 16 dwords:\n");
		for (off = base; off < base + 64; off += 4) {
			pr_info("fw27:   [0x%llX] = 0x%08X\n", off,
				readl(vram + off));
		}
	}

	/* Scan 6MB region */
	{
		u64 base = 6 * 1024 * 1024;
		u64 end = 8 * 1024 * 1024;
		u64 off;
		int sig_count = 0;

		pr_info("fw27: --- VRAM 6MB-8MB signatures ---\n");
		for (off = base; off < end && sig_count < 20; off += 4) {
			u32 v = readl(vram + off);
			if (v == INST_BRANCH_SELF) {
				pr_info("fw27:   [0x%llX] BRANCH_SELF!\n", off);
				sig_count++;
			} else if (v == 0xC424000B) {
				pr_info("fw27:   [0x%llX] MEC_FIRST_INSTR!\n", off);
				sig_count++;
			} else if (v == 0x0E6F518F) {
				pr_info("fw27:   [0x%llX] ENCRYPTED_HEADER!\n", off);
				sig_count++;
			}
		}
		if (sig_count == 0)
			pr_info("fw27:   No known signatures\n");

		/* Dump first 16 dwords of 6MB region */
		pr_info("fw27: VRAM[6MB] first 16 dwords:\n");
		for (off = base; off < base + 64; off += 4) {
			pr_info("fw27:   [0x%llX] = 0x%08X\n", off,
				readl(vram + off));
		}
	}

	/* Find the FIRST non-zero content and dump around it */
	{
		u64 off;
		pr_info("fw27: --- First non-zero VRAM locations ---\n");
		for (off = 0; off < map_size; off += 4) {
			u32 v = readl(vram + off);
			if (v != 0) {
				int i;
				pr_info("fw27: First non-zero at 0x%llX:\n", off);
				for (i = 0; i < 8 && (off + i*4) < map_size; i++)
					pr_info("fw27:   [0x%llX] = 0x%08X\n",
						off + i*4,
						readl(vram + off + i*4));
				break;
			}
		}
	}

	iounmap(vram);

skip_vram:
	/* ========================================
	 * SECTION C: RLCG register access test
	 * ======================================== */
	pr_info("fw27: === SECTION C: RLCG ACCESS ===\n");

	/* First, check SCRATCH register current state */
	pr_info("fw27: SCRATCH[0..3] = 0x%08X 0x%08X 0x%08X 0x%08X\n",
		rr(regSCRATCH_REG0), rr(regSCRATCH_REG1),
		rr(regSCRATCH_REG2), rr(regSCRATCH_REG3));

	/* Try RLCG read of MEC_CNTL (should return current value) */
	pr_info("fw27: Trying RLCG read of MEC_CNTL (IP=0x%04X)...\n",
		IP_regCP_MEC_CNTL);
	val = rlcg_read(IP_regCP_MEC_CNTL);
	pr_info("fw27:   RLCG MEC_CNTL = 0x%08X (direct=0x%08X)\n",
		val, rr(regCP_MEC_CNTL));

	/* Try RLCG read of IC_BASE */
	val = rlcg_read(IP_regCP_CPC_IC_BASE_LO);
	pr_info("fw27:   RLCG IC_BASE_LO = 0x%08X (direct=0x%08X)\n",
		val, rr(regCP_CPC_IC_BASE_LO));

	/* Halt MEC for write test */
	pr_info("fw27: Halting MEC for RLCG write test...\n");
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_ME2_HALT);
	udelay(200);

	/* Try RLCG write to IC_BASE_LO */
	pr_info("fw27: RLCG write IC_BASE_LO = 0xDEAD0000...\n");
	{
		int rc = rlcg_write(IP_regCP_CPC_IC_BASE_LO, 0xDEAD0000);
		if (rc == 0) {
			val = rr(regCP_CPC_IC_BASE_LO);
			pr_info("fw27:   Direct readback = 0x%08X %s\n", val,
				(val == 0xDEAD0000) ? "RLCG WRITE WORKS!" :
				(val == 0x07) ? "no change" : "different");

			val = rlcg_read(IP_regCP_CPC_IC_BASE_LO);
			pr_info("fw27:   RLCG readback = 0x%08X\n", val);
		}
	}

	/* Try RLCG write to UCODE_ADDR then DATA (SRAM write via RLCG) */
	pr_info("fw27: RLCG SRAM write attempt...\n");
	{
		int rc;
		rc = rlcg_write(IP_regCP_MEC_ME1_UCODE_ADDR, 0x44C);
		if (rc == 0) {
			rc = rlcg_write(IP_regCP_MEC_ME1_UCODE_DATA, INST_NOP);
			if (rc == 0) {
				pr_info("fw27:   RLCG SRAM write completed\n");
				/* Read back */
				rlcg_write(IP_regCP_MEC_ME1_UCODE_ADDR, 0x44C);
				val = rlcg_read(IP_regCP_MEC_ME1_UCODE_DATA);
				pr_info("fw27:   RLCG SRAM readback = 0x%08X %s\n",
					val,
					(val == INST_NOP) ? "NOP WRITTEN!" :
					(val == 0) ? "still zero" : "different");
			}
		}
	}

	/* Restore and unhalt */
	pr_info("fw27: Restoring IC_BASE and unhalting...\n");
	wr(regCP_CPC_IC_BASE_LO, 0x07);
	wr(regCP_CPC_IC_BASE_HI, 0x03);
	wr(regCP_MEC_CNTL, 0);
	udelay(500);
	pr_info("fw27: PC=0x%04X MEC_CNTL=0x%08X\n",
		rr(regCP_MEC1_INSTR_PNTR), rr(regCP_MEC_CNTL));

	pr_info("fw27: ========================================\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw27_exit(void) {}

module_init(fw27_init);
module_exit(fw27_exit);
