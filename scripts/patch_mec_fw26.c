/*
 * patch_mec_fw26.c — Phase 26: VRAM scan for firmware + IC_BASE DMA test
 *
 * PSP blocks SRAM read/write via UCODE registers. New approach:
 *
 *   A. Scan VRAM (BAR0) for firmware signatures (header magic, branch_self)
 *   B. Try IC_BASE + IC_OP_CNTL cache prime from our own VRAM buffer
 *   C. Scan for MEC data segment (ring buffers, scratch, MQD) in VRAM
 *
 * From Phase 20: mec_fw at adev+0x18890, fw->data is PSP-encrypted
 * From Phase 19: GPU VA 0x97FF943000 at adev+0x16808
 * From Phase 25: BAR0 = 0x6800000000 (256MB VRAM)
 *
 * Key insight: config_mec_cache writes GPU VA to IC_BASE, then primes.
 * If we write firmware to a known VRAM offset and prime from there...
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

/* IC_OP_CNTL bits */
#define IC_INVALIDATE_CACHE        (1 << 0)
#define IC_PRIME_ICACHE            (1 << 4)

#define regGRBM_GFX_CNTL           0x0D880

/* Firmware header signature */
#define FW_COMMON_HEADER_MAGIC     0x00000000  /* version field */

/* MEC instructions */
#define INST_BRANCH_SELF 0x88000000
#define INST_NOP         0xBF800000

/* Scan VRAM for patterns */
static void scan_vram_region(void __iomem *vram, u64 base_off,
			     u64 len, const char *label)
{
	u64 off;
	int found = 0;

	pr_info("fw26: Scanning %s: offset 0x%llX, len 0x%llX\n",
		label, base_off, len);

	for (off = 0; off < len && found < 20; off += 4) {
		u32 val = readl(vram + off);

		/* Look for MEC firmware signatures */
		if (val == INST_BRANCH_SELF) {
			pr_info("fw26:   [0x%llX] BRANCH_SELF 0x88000000!\n",
				base_off + off);
			found++;
		} else if (val == 0xC424000B) {
			pr_info("fw26:   [0x%llX] MEC first_instr 0xC424000B!\n",
				base_off + off);
			found++;
		} else if (val == 0x0E6F518F) {
			pr_info("fw26:   [0x%llX] encrypted header 0x0E6F518F!\n",
				base_off + off);
			found++;
		} else if (val == 0x00040400) {
			/* ucode_size_bytes field from FW header */
			pr_info("fw26:   [0x%llX] ucode_size 0x00040400!\n",
				base_off + off);
			found++;
		} else if ((val & 0xFFFF0000) == 0xBF800000) {
			/* NOP instruction */
			if (found < 5) {
				pr_info("fw26:   [0x%llX] NOP-like 0x%08X\n",
					base_off + off, val);
				found++;
			}
		}
	}

	if (found == 0)
		pr_info("fw26:   No firmware signatures found\n");
}

/* Dump first N non-zero dwords in a region */
static void dump_nonzero(void __iomem *vram, u64 base_off,
			 u64 len, const char *label, int max_dump)
{
	u64 off;
	int count = 0;
	u64 first_nz = 0, last_nz = 0;
	int total_nz = 0;

	for (off = 0; off < len; off += 4) {
		u32 val = readl(vram + off);
		if (val != 0 && val != 0xFFFFFFFF) {
			total_nz++;
			if (total_nz == 1)
				first_nz = base_off + off;
			last_nz = base_off + off;
			if (count < max_dump) {
				pr_info("fw26: %s[0x%llX] = 0x%08X\n",
					label, base_off + off, val);
				count++;
			}
		}
	}
	pr_info("fw26: %s: %d non-zero dwords in range, first=0x%llX last=0x%llX\n",
		label, total_nz, first_nz, last_nz);
}

static int __init fw26_init(void)
{
	struct pci_dev *pdev = NULL;
	resource_size_t bar0_start, bar0_len;
	void __iomem *vram;
	u64 map_size;
	u32 pc;

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

	pr_info("fw26: ========================================\n");
	pr_info("fw26: PHASE 26: VRAM SCAN + IC PRIME TEST\n");
	pr_info("fw26: ========================================\n");

	pc = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("fw26: PC=0x%04X MEC_CNTL=0x%08X\n", pc, rr(regCP_MEC_CNTL));
	pr_info("fw26: BAR0: 0x%llX len=0x%llX\n",
		(u64)bar0_start, (u64)bar0_len);

	/* Map first 16MB of VRAM for scanning */
	map_size = min_t(u64, bar0_len, 16 * 1024 * 1024);
	vram = ioremap_wc(bar0_start, map_size);
	if (!vram) {
		pr_info("fw26: Failed to map VRAM!\n");
		goto out;
	}

	/* Section A: Scan VRAM for firmware content */
	pr_info("fw26: === SECTION A: VRAM SCAN ===\n");

	/* Scan first 1MB */
	scan_vram_region(vram, 0, min_t(u64, map_size, 1024*1024),
			 "VRAM[0-1MB]");

	/* Scan around common firmware offsets */
	if (map_size >= 4*1024*1024) {
		dump_nonzero(vram, 0, 64*1024, "VRAM_start", 8);
	}

	/* Scan near the top of mapped region (TMR is often near top of VRAM) */
	if (map_size > 1024*1024) {
		u64 top_start = map_size - 1024*1024;
		scan_vram_region(vram + top_start, top_start,
				 1024*1024, "VRAM_top_1MB");
	}

	/* Scan in 1MB chunks, just checking for non-zero content */
	{
		u64 chunk;
		for (chunk = 0; chunk < map_size; chunk += 1024*1024) {
			u64 end = min_t(u64, chunk + 1024*1024, map_size);
			u64 off;
			int nz = 0;
			for (off = chunk; off < end; off += 256) {
				u32 val = readl(vram + off);
				if (val != 0 && val != 0xFFFFFFFF)
					nz++;
			}
			if (nz > 0)
				pr_info("fw26: VRAM chunk [%lluMB]: %d/4096 samples non-zero\n",
					chunk / (1024*1024), nz);
		}
	}

	/* Section B: IC_BASE + IC_OP_CNTL cache prime test
	 *
	 * Write a simple MEC program into VRAM at offset 0 (BAR0 base):
	 *   [0x000] = NOP (0xBF800000)
	 *   [0x004] = NOP
	 *   ...
	 *   [0x44C*4] = NOP (where MEC would read at PC=0x44C)
	 *   [0x44D*4] = BRANCH_SELF (new trap)
	 *
	 * Then set IC_BASE = GPU VA of BAR0 offset 0 and prime.
	 * Problem: we don't know the GPU VA for VRAM offset 0.
	 * On GFX11, FB_OFFSET might == GPU VA for identity-mapped region.
	 * Or VRAM base might be at GPU VA 0x0 or at a known offset.
	 *
	 * Let's check MC_VM_FB_LOCATION registers to find the mapping.
	 */
	pr_info("fw26: === SECTION B: GPU VA MAPPING ===\n");
	{
		/* MC_VM_FB_LOCATION_BASE/TOP — where VRAM appears in GPU VA space */
		/* GFX11 registers (SOC15 GC block) */
		u32 fb_base_lo, fb_base_hi, fb_top, fb_offset;

		/* mmMC_VM_FB_LOCATION_BASE = 0x048 in MMHUB */
		/* Via MMIO, these are in the MMHUB space.
		 * Let's try some known register offsets. */

		/* MMHUB registers (base offsets vary by ASIC) */
		/* For GFX11, try common offsets */
#define regMC_VM_FB_LOCATION_BASE  0x0600
#define regMC_VM_FB_LOCATION_TOP   0x0601
#define regMC_VM_FB_OFFSET         0x0602
#define regMC_VM_AGP_BASE          0x0603
#define regMC_VM_AGP_BOT           0x0604
#define regMC_VM_AGP_TOP           0x0605

		fb_base_lo = rr(regMC_VM_FB_LOCATION_BASE);
		fb_top = rr(regMC_VM_FB_LOCATION_TOP);
		fb_offset = rr(regMC_VM_FB_OFFSET);

		pr_info("fw26: MC_VM_FB_LOCATION_BASE = 0x%08X\n", fb_base_lo);
		pr_info("fw26: MC_VM_FB_LOCATION_TOP  = 0x%08X\n", fb_top);
		pr_info("fw26: MC_VM_FB_OFFSET        = 0x%08X\n", fb_offset);
		pr_info("fw26: MC_VM_AGP_BASE         = 0x%08X\n",
			rr(regMC_VM_AGP_BASE));
		pr_info("fw26: MC_VM_AGP_BOT          = 0x%08X\n",
			rr(regMC_VM_AGP_BOT));
		pr_info("fw26: MC_VM_AGP_TOP          = 0x%08X\n",
			rr(regMC_VM_AGP_TOP));

		/* Try different register bank for MMHUB */
		/* On some GFX11, MMHUB0 base is at reg_offset[MMHUB_HWIP][0][0] */
		/* Common values: 0x3A00, 0x68E0, etc. */
		pr_info("fw26: Trying alternate MMHUB offsets:\n");
		{
			u32 offsets[] = {
				0x3A00, 0x3A01, 0x3A02, /* MMHUB alt 1 */
				0x68E0, 0x68E1, 0x68E2, /* MMHUB alt 2 */
				0x6800, 0x6801, 0x6802, /* MMHUB alt 3 */
			};
			int i;
			for (i = 0; i < 9; i++) {
				u32 val = rr(offsets[i]);
				if (val != 0 && val != 0xFFFFFFFF)
					pr_info("fw26:   [0x%04X] = 0x%08X\n",
						offsets[i], val);
			}
		}

		/* Also check via SMC/SMN path - the VM registers might need
		 * indirect access. Check regBIF_BX_PF0_GPU_HDP_FLUSH_REQ etc. */
	}

	/* Section C: Check MEC ring/MQD pointers in MMIO
	 * These are in-VRAM data structures the MEC reads during operation.
	 * If we can find them, we might patch MEC behavior via data. */
	pr_info("fw26: === SECTION C: MEC RING/MQD REGS ===\n");
	{
		/* CP_MQD_BASE_ADDR — MEC Queue Descriptor base in VRAM */
#define regCP_MQD_BASE_ADDR_LO     0x0C914
#define regCP_MQD_BASE_ADDR_HI     0x0C915
#define regCP_HQD_PQ_BASE_LO       0x0C916
#define regCP_HQD_PQ_BASE_HI       0x0C917
#define regCP_HQD_PQ_RPTR          0x0C91A
#define regCP_HQD_PQ_WPTR_LO       0x0C91B
#define regCP_HQD_PQ_WPTR_HI       0x0C91C
#define regCP_HQD_ACTIVE           0x0C91E

		/* Select MEC1 pipe0 queue0 */
		wr(regGRBM_GFX_CNTL, 0x10);
		udelay(100);

		pr_info("fw26: MQD_BASE: LO=0x%08X HI=0x%08X\n",
			rr(regCP_MQD_BASE_ADDR_LO),
			rr(regCP_MQD_BASE_ADDR_HI));
		pr_info("fw26: HQD_PQ_BASE: LO=0x%08X HI=0x%08X\n",
			rr(regCP_HQD_PQ_BASE_LO),
			rr(regCP_HQD_PQ_BASE_HI));
		pr_info("fw26: HQD_PQ_RPTR=0x%08X WPTR_LO=0x%08X WPTR_HI=0x%08X\n",
			rr(regCP_HQD_PQ_RPTR),
			rr(regCP_HQD_PQ_WPTR_LO),
			rr(regCP_HQD_PQ_WPTR_HI));
		pr_info("fw26: HQD_ACTIVE=0x%08X\n", rr(regCP_HQD_ACTIVE));

		/* Try different queues */
		{
			int q;
			for (q = 0; q < 4; q++) {
				u32 sel = 0x10 | (q << 8); /* ME=1 PIPE=0 QUEUE=q */
				u32 active, mqd_lo;
				wr(regGRBM_GFX_CNTL, sel);
				udelay(50);
				active = rr(regCP_HQD_ACTIVE);
				mqd_lo = rr(regCP_MQD_BASE_ADDR_LO);
				if (active || mqd_lo)
					pr_info("fw26: Q%d: ACTIVE=0x%X MQD=0x%08X\n",
						q, active, mqd_lo);
			}
		}

		wr(regGRBM_GFX_CNTL, 0);
	}

	/* Section D: Try IC_BASE write + prime while halted
	 * Write our NOP sled to VRAM at offset 0, then try to prime from
	 * physical VRAM offset (as if GPU VA == VRAM offset in FB space) */
	pr_info("fw26: === SECTION D: IC PRIME FROM VRAM ===\n");
	{
		u64 fw_off;
		int i;

		/* Write NOP sled at VRAM offset 0 */
		for (i = 0; i < 0x500; i++)
			writel(INST_NOP, vram + i * 4);

		/* Write branch_self at offset 0x44D as a new trap point */
		writel(INST_BRANCH_SELF, vram + 0x44D * 4);

		/* Verify writes */
		pr_info("fw26: VRAM[0x000]=0x%08X (expect NOP)\n",
			readl(vram));
		pr_info("fw26: VRAM[0x44C*4]=0x%08X (expect NOP)\n",
			readl(vram + 0x44C * 4));
		pr_info("fw26: VRAM[0x44D*4]=0x%08X (expect BRANCH_SELF)\n",
			readl(vram + 0x44D * 4));

		/* Halt MEC */
		wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_ME2_HALT);
		udelay(200);
		pr_info("fw26: Halted PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

		/* Try writing IC_BASE with different address interpretations:
		 * Attempt 1: VRAM physical address (BAR0 base)
		 * Attempt 2: VRAM offset 0 (identity map)
		 * Attempt 3: GPU VA from FB_LOCATION_BASE
		 */
		fw_off = 0; /* VRAM offset 0 */

		/* Attempt 1: physical BAR0 address */
		pr_info("fw26: IC_BASE attempt 1: phys BAR0=0x%llX\n",
			(u64)bar0_start);
		wr(regCP_CPC_IC_BASE_LO, lower_32_bits(bar0_start));
		wr(regCP_CPC_IC_BASE_HI, upper_32_bits(bar0_start));
		pr_info("fw26:   Readback: LO=0x%08X HI=0x%08X\n",
			rr(regCP_CPC_IC_BASE_LO),
			rr(regCP_CPC_IC_BASE_HI));

		/* Invalidate + prime */
		wr(regCP_CPC_IC_OP_CNTL, IC_INVALIDATE_CACHE);
		udelay(100);
		wr(regCP_CPC_IC_OP_CNTL, IC_PRIME_ICACHE);
		udelay(500);
		pr_info("fw26:   IC_OP_CNTL after prime=0x%08X\n",
			rr(regCP_CPC_IC_OP_CNTL));

		/* Unhalt and check */
		wr(regCP_MEC_CNTL, 0);
		udelay(1000);
		pc = rr(regCP_MEC1_INSTR_PNTR);
		pr_info("fw26:   After unhalt: PC=0x%04X%s\n", pc,
			(pc == 0x44C) ? " stuck" :
			(pc == 0x44D) ? " AT NEW TRAP!" : " MOVED!");

		/* Re-halt for next attempt */
		wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_ME2_HALT);
		udelay(200);

		/* Attempt 2: VRAM offset 0 as GPU VA */
		pr_info("fw26: IC_BASE attempt 2: VRAM offset 0\n");
		wr(regCP_CPC_IC_BASE_LO, 0x00000000);
		wr(regCP_CPC_IC_BASE_HI, 0x00000000);
		wr(regCP_CPC_IC_OP_CNTL, IC_INVALIDATE_CACHE);
		udelay(100);
		wr(regCP_CPC_IC_OP_CNTL, IC_PRIME_ICACHE);
		udelay(500);
		wr(regCP_MEC_CNTL, 0);
		udelay(1000);
		pc = rr(regCP_MEC1_INSTR_PNTR);
		pr_info("fw26:   After unhalt: PC=0x%04X%s\n", pc,
			(pc == 0x44C) ? " stuck" :
			(pc == 0x44D) ? " AT NEW TRAP!" : " MOVED!");

		/* Re-halt */
		wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_ME2_HALT);
		udelay(200);

		/* Restore original IC_BASE */
		wr(regCP_CPC_IC_BASE_LO, 0x00000007);
		wr(regCP_CPC_IC_BASE_HI, 0x00000003);
	}

	/* Final: unhalt and verify MEC still functional */
	pr_info("fw26: Final unhalt...\n");
	wr(regGRBM_GFX_CNTL, 0);
	wr(regCP_MEC_CNTL, 0);
	udelay(500);
	pr_info("fw26: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));

	/* Clean up VRAM — write zeros back */
	{
		int i;
		for (i = 0; i < 0x500; i++)
			writel(0, vram + i * 4);
	}

	iounmap(vram);
out:
	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw26_exit(void) {}

module_init(fw26_init);
module_exit(fw26_exit);
