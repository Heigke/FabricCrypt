/*
 * patch_mec_fw6.c — Phase 6: RS64-specific IC/DC invalidation
 *
 * Phases 2-4 used legacy CP_MEC_CNTL (0x0A802) — WRONG for GFX11 RS64.
 * This module uses the RS64 register interface:
 *   CP_MEC_RS64_CNTL          (0x0C904) — halt, pipe reset, IC invalidate
 *   CP_MEC_RS64_INSTR_PNTR    (0x0C908) — real RS64 program counter
 *   CP_MEC_DC_OP_CNTL         (0x0C90C) — data cache invalidation + poll
 *   CP_CPC_IC_OP_CNTL         (0x0C97A) — instruction cache ops + poll
 *   CP_MEC_RS64_PRGRM_CNTR_START (0x0C900) — boot PC
 *
 * Test plan:
 *   PROBE: Read both legacy + RS64 registers with GRBM context
 *   TEST A: RS64 IC invalidate (bit 4) + pipe reset
 *   TEST B: DC invalidate via DC_OP_CNTL with COMPLETE polling
 *   TEST C: IC invalidate via CPC_IC_OP_CNTL with COMPLETE polling
 *   TEST D: IC prime (force IC reload from memory)
 *   TEST E: Full sequence — DC flush + IC invalidate + IC prime + pipe reset
 *           with firmware patch at BOTH 0x44C and RS64 PC offset
 *
 * Auto-unloads (-ENODEV).
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/mm.h>
#include <linux/highmem.h>
#include <linux/delay.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002

/* === Legacy registers (for comparison) === */
#define regCP_MEC_CNTL              0x0A802
#define regCP_MEC1_INSTR_PNTR      0x021A8

/* === RS64-specific registers === */
#define regCP_MEC_RS64_CNTL             0x0C904
#define regCP_MEC_RS64_INSTR_PNTR       0x0C908
#define regCP_MEC_RS64_PRGRM_CNTR_START 0x0C900
#define regCP_MEC_RS64_PRGRM_CNTR_START_HI 0x0C938
#define regCP_MEC_RS64_INTERRUPT        0x0C907
#define regCP_MEC_DC_OP_CNTL            0x0C90C
#define regCP_MEC_DC_BASE_LO            0x0F870
#define regCP_MEC_DC_BASE_HI            0x0F871
#define regCP_MEC_DC_BASE_CNTL          0x0C90B

/* === Shared IC register === */
#define regCP_CPC_IC_OP_CNTL            0x0C97A

/* === GRBM === */
#define regGRBM_GFX_CNTL                0x0A900

/* === CP_MEC_RS64_CNTL bits === */
#define RS64_INVALIDATE_ICACHE  (1 << 4)
#define RS64_PIPE0_RESET        (1 << 16)
#define RS64_PIPE1_RESET        (1 << 17)
#define RS64_PIPE2_RESET        (1 << 18)
#define RS64_PIPE3_RESET        (1 << 19)
#define RS64_HALT               (1 << 30)
#define RS64_STEP               (1 << 31)
#define RS64_ALL_PIPE_RESET     (RS64_PIPE0_RESET | RS64_PIPE1_RESET | \
				 RS64_PIPE2_RESET | RS64_PIPE3_RESET)

/* === Legacy CP_MEC_CNTL bits (for comparison) === */
#define LEG_MEC_ME1_HALT        (1 << 30)
#define LEG_MEC_INVALIDATE_IC   (1 << 27)
#define LEG_ALL_PIPE_RESET      ((1<<16)|(1<<17)|(1<<18)|(1<<19))

/* === CP_MEC_DC_OP_CNTL bits === */
#define DC_INVALIDATE           (1 << 0)
#define DC_INVALIDATE_COMPLETE  (1 << 1)
#define DC_BYPASS_ALL           (1 << 2)

/* === CP_CPC_IC_OP_CNTL bits === */
#define IC_INVALIDATE           (1 << 0)
#define IC_INVALIDATE_COMPLETE  (1 << 1)
#define IC_PRIME_ICACHE         (1 << 4)
#define IC_ICACHE_PRIMED        (1 << 5)

static void __iomem *mmio;

static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }
static void wr(u32 dw_off, u32 val) {
	writel(val, mmio + (u64)dw_off * 4);
	readl(mmio + (u64)dw_off * 4); /* flush posted write */
}

/* Live firmware copy — confirmed in Phase 2/3 */
#define FW_PHYS  0x115c8c100ULL

static u32 orig_44c, orig_800;

static u32 *map_fw_dword(int dw_off)
{
	u64 phys = FW_PHYS + (u64)dw_off * 4;
	unsigned long pfn = phys >> PAGE_SHIFT;
	unsigned int page_off = phys & ~PAGE_MASK;
	struct page *page;
	u8 *vaddr;

	if (!pfn_valid(pfn))
		return NULL;
	page = pfn_to_page(pfn);
	vaddr = kmap_local_page(page);
	if (!vaddr)
		return NULL;
	return (u32 *)(vaddr + page_off);
}

static void unmap_fw(void *ptr)
{
	kunmap_local((void *)((unsigned long)ptr & PAGE_MASK));
}

static u32 read_fw(int dw_off)
{
	u32 val = 0;
	u32 *ptr = map_fw_dword(dw_off);
	if (ptr) { val = *ptr; unmap_fw(ptr); }
	return val;
}

static void write_fw(int dw_off, u32 val)
{
	u32 *ptr = map_fw_dword(dw_off);
	if (ptr) { *ptr = val; wmb(); unmap_fw(ptr); }
}

/* Poll a register bit with timeout */
static int poll_bit(u32 reg, u32 mask, int set, int timeout_us)
{
	int elapsed = 0;
	while (elapsed < timeout_us) {
		u32 val = rr(reg);
		if (set ? (val & mask) : !(val & mask))
			return 0; /* success */
		udelay(10);
		elapsed += 10;
	}
	return -1; /* timeout */
}

/* Select GRBM context for MEC1 pipe 0 */
static void select_mec1_pipe0(void)
{
	/* GRBM_GFX_CNTL: ME=1 (MEC1), PIPE=0, QUEUE=0, VMID=0 */
	/* Bits: [3:0]=PIPEID, [5:4]=MEID, [10:8]=QUEUEID, [15:12]=VMID */
	/* ME=1 → bits[5:4]=01, PIPE=0 → bits[3:0]=0000 */
	wr(regGRBM_GFX_CNTL, (1 << 4)); /* MEID=1 */
	udelay(100);
}

/* Clear GRBM context */
static void clear_grbm(void)
{
	wr(regGRBM_GFX_CNTL, 0);
	udelay(100);
}

static void dump_state(const char *label)
{
	pr_info("fw6: [%s] LEGACY: MEC_CNTL=0x%08X PC=0x%04X\n",
		label, rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));
	pr_info("fw6: [%s] RS64:   CNTL=0x%08X PC=0x%04X START=0x%04X\n",
		label, rr(regCP_MEC_RS64_CNTL),
		rr(regCP_MEC_RS64_INSTR_PNTR),
		rr(regCP_MEC_RS64_PRGRM_CNTR_START));
	pr_info("fw6: [%s] CACHE:  DC_OP=0x%08X IC_OP=0x%08X\n",
		label, rr(regCP_MEC_DC_OP_CNTL), rr(regCP_CPC_IC_OP_CNTL));
}

static int __init fw6_init(void)
{
	struct pci_dev *pdev = NULL;
	int ret;

	while ((pdev = pci_get_device(AMD_VENDOR_ID, PCI_ANY_ID, pdev))) {
		if ((pdev->class >> 8) == PCI_CLASS_DISPLAY_OTHER)
			break;
	}
	if (!pdev) {
		pr_err("fw6: no AMD GPU\n");
		return -ENODEV;
	}
	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENOMEM;
	}

	pr_info("fw6: ========================================\n");
	pr_info("fw6: PHASE 6: RS64 IC/DC INVALIDATION\n");
	pr_info("fw6: ========================================\n");

	/* ============================================================
	 * PROBE: Read all registers without GRBM context first,
	 * then with MEC1 PIPE0 context
	 * ============================================================ */
	pr_info("fw6: --- PROBE: Register state (no GRBM context) ---\n");
	dump_state("NO_CTX");

	pr_info("fw6: --- PROBE: Register state (GRBM: MEC1 PIPE0) ---\n");
	select_mec1_pipe0();
	dump_state("MEC1P0");

	/* Also read DC base address */
	pr_info("fw6: DC_BASE: LO=0x%08X HI=0x%08X CNTL=0x%08X\n",
		rr(regCP_MEC_DC_BASE_LO), rr(regCP_MEC_DC_BASE_HI),
		rr(regCP_MEC_DC_BASE_CNTL));
	pr_info("fw6: RS64_INTERRUPT=0x%08X\n", rr(regCP_MEC_RS64_INTERRUPT));
	pr_info("fw6: RS64_PRGRM_CNTR_START_HI=0x%08X\n",
		rr(regCP_MEC_RS64_PRGRM_CNTR_START_HI));

	clear_grbm();

	/* Show firmware at both potential patch points */
	pr_info("fw6: FW[0x44C]=0x%08X (legacy idle)\n", read_fw(0x44C));
	pr_info("fw6: FW[0x44A]=0x%08X (legacy idle2)\n", read_fw(0x44A));
	pr_info("fw6: FW[0x800]=0x%08X (RS64 PC_START)\n", read_fw(0x800));

	/* Save originals */
	orig_44c = read_fw(0x44C);
	orig_800 = read_fw(0x800);

	/* ============================================================
	 * TEST A: RS64 halt + IC invalidate (bit 4) + pipe reset
	 * Use RS64_CNTL exclusively. Patch 0x44C with marker.
	 * ============================================================ */
	pr_info("fw6: --- TEST A: RS64_CNTL IC invalidate + pipe reset ---\n");

	/* A1: Halt via RS64_CNTL */
	wr(regCP_MEC_RS64_CNTL, RS64_HALT);
	udelay(500);
	dump_state("A_HALT");

	/* A2: Patch firmware at 0x44C */
	write_fw(0x44C, 0xBF80DEAD);
	pr_info("fw6: A patched FW[0x44C]=0x%08X\n", read_fw(0x44C));

	/* A3: IC invalidate via RS64 bit 4 + pipe reset */
	wr(regCP_MEC_RS64_CNTL, RS64_HALT | RS64_ALL_PIPE_RESET | RS64_INVALIDATE_ICACHE);
	udelay(2000);

	/* A4: Clear reset, keep halted */
	wr(regCP_MEC_RS64_CNTL, RS64_HALT);
	udelay(500);
	dump_state("A_RST");

	/* A5: Unhalt */
	wr(regCP_MEC_RS64_CNTL, 0);
	mdelay(20);
	{
		int j;
		for (j = 0; j < 6; j++) {
			mdelay(5);
			pr_info("fw6: A PC[%d] legacy=0x%04X rs64=0x%04X\n",
				j, rr(regCP_MEC1_INSTR_PNTR),
				rr(regCP_MEC_RS64_INSTR_PNTR));
		}
	}

	/* Restore */
	wr(regCP_MEC_RS64_CNTL, RS64_HALT);
	udelay(500);
	write_fw(0x44C, orig_44c);
	wr(regCP_MEC_RS64_CNTL, RS64_HALT | RS64_ALL_PIPE_RESET | RS64_INVALIDATE_ICACHE);
	udelay(2000);
	wr(regCP_MEC_RS64_CNTL, RS64_HALT);
	udelay(500);
	wr(regCP_MEC_RS64_CNTL, 0);
	mdelay(50);
	dump_state("A_DONE");

	/* ============================================================
	 * TEST B: DC invalidation via CP_MEC_DC_OP_CNTL with polling
	 * Instructions might be fetched through data cache path
	 * ============================================================ */
	pr_info("fw6: --- TEST B: DC invalidation with polling ---\n");

	wr(regCP_MEC_RS64_CNTL, RS64_HALT);
	udelay(500);

	/* Patch 0x44C */
	write_fw(0x44C, 0xBF80DEAD);
	pr_info("fw6: B patched FW[0x44C]=0x%08X\n", read_fw(0x44C));

	/* DC invalidate */
	wr(regCP_MEC_DC_OP_CNTL, DC_INVALIDATE);
	ret = poll_bit(regCP_MEC_DC_OP_CNTL, DC_INVALIDATE_COMPLETE, 1, 10000);
	pr_info("fw6: B DC invalidate %s (DC_OP=0x%08X)\n",
		ret == 0 ? "COMPLETE" : "TIMEOUT", rr(regCP_MEC_DC_OP_CNTL));

	/* Pipe reset */
	wr(regCP_MEC_RS64_CNTL, RS64_HALT | RS64_ALL_PIPE_RESET);
	udelay(2000);
	wr(regCP_MEC_RS64_CNTL, RS64_HALT);
	udelay(500);

	/* Unhalt */
	wr(regCP_MEC_RS64_CNTL, 0);
	mdelay(20);
	{
		int j;
		for (j = 0; j < 6; j++) {
			mdelay(5);
			pr_info("fw6: B PC[%d] legacy=0x%04X rs64=0x%04X\n",
				j, rr(regCP_MEC1_INSTR_PNTR),
				rr(regCP_MEC_RS64_INSTR_PNTR));
		}
	}

	/* Restore */
	wr(regCP_MEC_RS64_CNTL, RS64_HALT);
	udelay(500);
	write_fw(0x44C, orig_44c);
	wr(regCP_MEC_DC_OP_CNTL, DC_INVALIDATE);
	poll_bit(regCP_MEC_DC_OP_CNTL, DC_INVALIDATE_COMPLETE, 1, 10000);
	wr(regCP_MEC_RS64_CNTL, RS64_HALT | RS64_ALL_PIPE_RESET);
	udelay(2000);
	wr(regCP_MEC_RS64_CNTL, RS64_HALT);
	udelay(500);
	wr(regCP_MEC_RS64_CNTL, 0);
	mdelay(50);

	/* ============================================================
	 * TEST C: IC invalidation via CP_CPC_IC_OP_CNTL with polling
	 * This is the CPC shared IC — may serve MEC instruction fetch
	 * ============================================================ */
	pr_info("fw6: --- TEST C: CPC IC invalidation with polling ---\n");

	wr(regCP_MEC_RS64_CNTL, RS64_HALT);
	udelay(500);

	/* Patch 0x44C */
	write_fw(0x44C, 0xBF80DEAD);

	/* IC invalidate via CPC_IC_OP_CNTL */
	wr(regCP_CPC_IC_OP_CNTL, IC_INVALIDATE);
	ret = poll_bit(regCP_CPC_IC_OP_CNTL, IC_INVALIDATE_COMPLETE, 1, 10000);
	pr_info("fw6: C IC invalidate %s (IC_OP=0x%08X)\n",
		ret == 0 ? "COMPLETE" : "TIMEOUT", rr(regCP_CPC_IC_OP_CNTL));

	/* Pipe reset */
	wr(regCP_MEC_RS64_CNTL, RS64_HALT | RS64_ALL_PIPE_RESET);
	udelay(2000);
	wr(regCP_MEC_RS64_CNTL, RS64_HALT);
	udelay(500);
	wr(regCP_MEC_RS64_CNTL, 0);
	mdelay(20);
	{
		int j;
		for (j = 0; j < 6; j++) {
			mdelay(5);
			pr_info("fw6: C PC[%d] legacy=0x%04X rs64=0x%04X\n",
				j, rr(regCP_MEC1_INSTR_PNTR),
				rr(regCP_MEC_RS64_INSTR_PNTR));
		}
	}

	/* Restore */
	wr(regCP_MEC_RS64_CNTL, RS64_HALT);
	udelay(500);
	write_fw(0x44C, orig_44c);
	wr(regCP_CPC_IC_OP_CNTL, IC_INVALIDATE);
	poll_bit(regCP_CPC_IC_OP_CNTL, IC_INVALIDATE_COMPLETE, 1, 10000);
	wr(regCP_MEC_RS64_CNTL, RS64_HALT | RS64_ALL_PIPE_RESET);
	udelay(2000);
	wr(regCP_MEC_RS64_CNTL, RS64_HALT);
	udelay(500);
	wr(regCP_MEC_RS64_CNTL, 0);
	mdelay(50);

	/* ============================================================
	 * TEST D: IC PRIME — force instruction cache to reload from mem
	 * Prime bit (4) triggers prefetch, poll Primed bit (5)
	 * ============================================================ */
	pr_info("fw6: --- TEST D: IC prime (force IC reload) ---\n");

	wr(regCP_MEC_RS64_CNTL, RS64_HALT);
	udelay(500);

	/* Patch 0x44C */
	write_fw(0x44C, 0xBF80DEAD);

	/* First invalidate IC */
	wr(regCP_CPC_IC_OP_CNTL, IC_INVALIDATE);
	poll_bit(regCP_CPC_IC_OP_CNTL, IC_INVALIDATE_COMPLETE, 1, 10000);
	pr_info("fw6: D IC invalidated: IC_OP=0x%08X\n", rr(regCP_CPC_IC_OP_CNTL));

	/* Now prime IC — force reload from memory */
	wr(regCP_CPC_IC_OP_CNTL, IC_PRIME_ICACHE);
	ret = poll_bit(regCP_CPC_IC_OP_CNTL, IC_ICACHE_PRIMED, 1, 50000);
	pr_info("fw6: D IC prime %s (IC_OP=0x%08X)\n",
		ret == 0 ? "COMPLETE" : "TIMEOUT", rr(regCP_CPC_IC_OP_CNTL));

	/* Pipe reset + unhalt */
	wr(regCP_MEC_RS64_CNTL, RS64_HALT | RS64_ALL_PIPE_RESET);
	udelay(2000);
	wr(regCP_MEC_RS64_CNTL, RS64_HALT);
	udelay(500);
	wr(regCP_MEC_RS64_CNTL, 0);
	mdelay(20);
	{
		int j;
		for (j = 0; j < 6; j++) {
			mdelay(5);
			pr_info("fw6: D PC[%d] legacy=0x%04X rs64=0x%04X\n",
				j, rr(regCP_MEC1_INSTR_PNTR),
				rr(regCP_MEC_RS64_INSTR_PNTR));
		}
	}

	/* Restore */
	wr(regCP_MEC_RS64_CNTL, RS64_HALT);
	udelay(500);
	write_fw(0x44C, orig_44c);

	/* ============================================================
	 * TEST E: FULL SEQUENCE — all invalidation + patch BOTH offsets
	 * This is the nuclear option: DC flush + IC invalidate +
	 * IC prime + RS64 IC invalidate + legacy IC invalidate +
	 * pipe reset on BOTH register sets
	 * Patch firmware at BOTH 0x44C (legacy PC) and 0x800 (RS64 PC)
	 * ============================================================ */
	pr_info("fw6: --- TEST E: Full nuclear invalidation ---\n");

	/* E1: Halt via BOTH register sets */
	wr(regCP_MEC_CNTL, LEG_MEC_ME1_HALT);
	wr(regCP_MEC_RS64_CNTL, RS64_HALT);
	udelay(1000);
	dump_state("E_HALT");

	/* E2: Patch BOTH offsets */
	write_fw(0x44C, 0xBF80DEAD);  /* legacy idle loop */
	write_fw(0x44A, 0xBF80BEEF);  /* legacy idle loop 2 */
	write_fw(0x800, 0xBF80CAFE);  /* RS64 PC start */
	pr_info("fw6: E patched: [0x44A]=0x%08X [0x44C]=0x%08X [0x800]=0x%08X\n",
		read_fw(0x44A), read_fw(0x44C), read_fw(0x800));

	/* E3: CPU cache flush — clflush the patched addresses */
	{
		u32 *ptr;
		ptr = map_fw_dword(0x44C);
		if (ptr) { clflush(ptr); wmb(); unmap_fw(ptr); }
		ptr = map_fw_dword(0x44A);
		if (ptr) { clflush(ptr); wmb(); unmap_fw(ptr); }
		ptr = map_fw_dword(0x800);
		if (ptr) { clflush(ptr); wmb(); unmap_fw(ptr); }
	}
	pr_info("fw6: E clflush done\n");

	/* E4: DC invalidate with polling */
	wr(regCP_MEC_DC_OP_CNTL, DC_INVALIDATE);
	ret = poll_bit(regCP_MEC_DC_OP_CNTL, DC_INVALIDATE_COMPLETE, 1, 10000);
	pr_info("fw6: E DC invalidate %s\n", ret == 0 ? "COMPLETE" : "TIMEOUT");

	/* E5: IC invalidate via CPC with polling */
	wr(regCP_CPC_IC_OP_CNTL, IC_INVALIDATE);
	ret = poll_bit(regCP_CPC_IC_OP_CNTL, IC_INVALIDATE_COMPLETE, 1, 10000);
	pr_info("fw6: E CPC IC invalidate %s\n", ret == 0 ? "COMPLETE" : "TIMEOUT");

	/* E6: IC prime to force reload */
	wr(regCP_CPC_IC_OP_CNTL, IC_PRIME_ICACHE);
	ret = poll_bit(regCP_CPC_IC_OP_CNTL, IC_ICACHE_PRIMED, 1, 50000);
	pr_info("fw6: E IC prime %s\n", ret == 0 ? "COMPLETE" : "TIMEOUT");

	/* E7: RS64 IC invalidate (bit 4) */
	wr(regCP_MEC_RS64_CNTL, RS64_HALT | RS64_INVALIDATE_ICACHE);
	udelay(1000);

	/* E8: Legacy IC invalidate (bit 27) — belt and suspenders */
	wr(regCP_MEC_CNTL, LEG_MEC_ME1_HALT | LEG_MEC_INVALIDATE_IC);
	udelay(1000);

	/* E9: Pipe reset via BOTH register sets */
	wr(regCP_MEC_RS64_CNTL, RS64_HALT | RS64_ALL_PIPE_RESET | RS64_INVALIDATE_ICACHE);
	wr(regCP_MEC_CNTL, LEG_MEC_ME1_HALT | LEG_ALL_PIPE_RESET | LEG_MEC_INVALIDATE_IC);
	udelay(3000);

	/* E10: Clear resets, keep halted */
	wr(regCP_MEC_RS64_CNTL, RS64_HALT);
	wr(regCP_MEC_CNTL, LEG_MEC_ME1_HALT);
	udelay(1000);
	dump_state("E_RST");

	/* E11: Unhalt BOTH */
	wr(regCP_MEC_RS64_CNTL, 0);
	wr(regCP_MEC_CNTL, 0);
	mdelay(30);

	/* E12: Sample PCs */
	{
		int j;
		for (j = 0; j < 10; j++) {
			mdelay(5);
			pr_info("fw6: E PC[%d] legacy=0x%04X rs64=0x%04X\n",
				j, rr(regCP_MEC1_INSTR_PNTR),
				rr(regCP_MEC_RS64_INSTR_PNTR));
		}
	}

	/* E13: Restore all patches */
	wr(regCP_MEC_RS64_CNTL, RS64_HALT);
	wr(regCP_MEC_CNTL, LEG_MEC_ME1_HALT);
	udelay(500);
	write_fw(0x44C, orig_44c);
	write_fw(0x44A, 0x88000000); /* original spin */
	write_fw(0x800, orig_800);
	{
		u32 *ptr;
		ptr = map_fw_dword(0x44C);
		if (ptr) { clflush(ptr); wmb(); unmap_fw(ptr); }
		ptr = map_fw_dword(0x44A);
		if (ptr) { clflush(ptr); wmb(); unmap_fw(ptr); }
		ptr = map_fw_dword(0x800);
		if (ptr) { clflush(ptr); wmb(); unmap_fw(ptr); }
	}

	/* Full restore sequence */
	wr(regCP_MEC_DC_OP_CNTL, DC_INVALIDATE);
	poll_bit(regCP_MEC_DC_OP_CNTL, DC_INVALIDATE_COMPLETE, 1, 10000);
	wr(regCP_CPC_IC_OP_CNTL, IC_INVALIDATE);
	poll_bit(regCP_CPC_IC_OP_CNTL, IC_INVALIDATE_COMPLETE, 1, 10000);
	wr(regCP_MEC_RS64_CNTL, RS64_HALT | RS64_ALL_PIPE_RESET | RS64_INVALIDATE_ICACHE);
	wr(regCP_MEC_CNTL, LEG_MEC_ME1_HALT | LEG_ALL_PIPE_RESET | LEG_MEC_INVALIDATE_IC);
	udelay(3000);
	wr(regCP_MEC_RS64_CNTL, RS64_HALT);
	wr(regCP_MEC_CNTL, LEG_MEC_ME1_HALT);
	udelay(500);
	wr(regCP_MEC_RS64_CNTL, 0);
	wr(regCP_MEC_CNTL, 0);
	mdelay(50);

	dump_state("FINAL");
	pr_info("fw6: ========================================\n");
	pr_info("fw6: PHASE 6 COMPLETE\n");
	pr_info("fw6: ========================================\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw6_exit(void) {}
module_init(fw6_init);
module_exit(fw6_exit);
