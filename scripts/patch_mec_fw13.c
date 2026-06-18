/*
 * patch_mec_fw13.c — Phase 13: HALT MEC → patch → PRIME → UN-HALT
 *
 * Key insight: Phases 8-12 all attempted PRIME/INVALIDATE while MEC was
 * running. The driver loads firmware while MEC is HALTED. PRIME_ICACHE
 * may only trigger a real DMA reload when the engine is halted.
 *
 * CP_MEC_CNTL (0x0802, BASE_IDX=1):
 *   bit[30] = MEC_ME1_HALT   (0x40000000)
 *   bit[28] = MEC_ME2_HALT   (0x10000000)
 *
 * Tests:
 *   A: HALT → patch all copies → PRIME → UN-HALT → check PC
 *   B: HALT → patch → clear IC_OP(0) → PRIME → wait PRIMED → UN-HALT
 *   C: HALT → patch → per-pipe IC_BASE re-write → PRIME → UN-HALT
 *   D: HALT → patch → full driver sequence (DC inv, IC inv, PRIME) → UN-HALT
 *   E: HALT → DON'T patch → PRIME → UN-HALT (control: does PRIME alone change behavior?)
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

/* GC registers (BASE_IDX=1: offset + 0xA000 in dwords) */
#define regCP_MEC_CNTL              0x0A802
#define regCP_MEC1_INSTR_PNTR      0x021A8
#define regCP_CPC_IC_OP_CNTL       0x0C97A
#define regCP_MEC_DC_OP_CNTL       0x0C90C
#define regGRBM_GFX_CNTL           0x0A900
#define regCP_CPC_IC_BASE_LO       0x0F84C
#define regCP_CPC_IC_BASE_HI       0x0F84D

/* CP_CPC_IC_OP_CNTL bits */
#define IC_INVALIDATE_CACHE          (1 << 0)
#define IC_INVALIDATE_CACHE_COMPLETE (1 << 1)
#define IC_PRIME_ICACHE              (1 << 4)
#define IC_ICACHE_PRIMED             (1 << 5)

/* CP_MEC_DC_OP_CNTL bits */
#define DC_INVALIDATE_DCACHE          (1 << 0)
#define DC_INVALIDATE_DCACHE_COMPLETE (1 << 1)

/* CP_MEC_CNTL bits */
#define MEC_ME1_PIPE0_RESET   (1 << 16)
#define MEC_ME1_PIPE1_RESET   (1 << 17)
#define MEC_ME1_PIPE2_RESET   (1 << 18)
#define MEC_ME1_PIPE3_RESET   (1 << 19)
#define MEC_ME2_HALT          (1 << 28)
#define MEC_ME1_HALT          (1 << 30)
#define ALL_PIPE_RESET (MEC_ME1_PIPE0_RESET | MEC_ME1_PIPE1_RESET | \
			MEC_ME1_PIPE2_RESET | MEC_ME1_PIPE3_RESET)

#define RS64_NOP          0x7C408001UL
#define RS64_BRANCH_SELF  0x88000000UL

static void __iomem *mmio;

static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }
static void wr(u32 dw_off, u32 val) {
	writel(val, mmio + (u64)dw_off * 4);
	readl(mmio + (u64)dw_off * 4); /* flush */
}

static void grbm_select(int me, int pipe, int queue, int vmid)
{
	u32 val = (pipe & 0xF) | ((me & 0x3) << 4) |
		  ((queue & 0x7) << 8) | ((vmid & 0xF) << 12);
	wr(regGRBM_GFX_CNTL, val);
	udelay(50);
}

/* 8 known firmware RAM copies from Phase 10 */
static phys_addr_t fw_copies[] = {
	0x115C4D878ULL,
	0x115C8C100ULL,  /* FW_PHYS original */
	0x122D89978ULL,
	0x125FA7200ULL,
	0x1791C3200ULL,
	0x22E0C1200ULL,
	0x2A70AE200ULL,
	0x2AD7EA63CULL,
};
#define N_COPIES 8

/* Firmware dword offset to patch: 0x44C = branch-self spin */
#define FW_PATCH_OFF 0x44C

static void patch_all_copies(u32 new_instr)
{
	int i;
	for (i = 0; i < N_COPIES; i++) {
		phys_addr_t pa = fw_copies[i] + (FW_PATCH_OFF * 4);
		struct page *pg;
		void *va;
		u32 *ptr;

		pg = pfn_to_page(pa >> PAGE_SHIFT);
		va = kmap_local_page(pg);
		ptr = (u32 *)(va + (pa & ~PAGE_MASK));

		/* Also patch 0x44A as NOP so if MEC slides back it still advances */
		if ((pa & ~PAGE_MASK) >= 8) {
			u32 *ptr_44a = ptr - 2; /* offset 0x44A */
			*ptr_44a = new_instr;
			clflush(ptr_44a);
		}
		*ptr = new_instr;
		clflush(ptr);
		kunmap_local(va);
	}
	mb();
}

static void restore_all_copies(void)
{
	int i;
	for (i = 0; i < N_COPIES; i++) {
		phys_addr_t pa = fw_copies[i] + (FW_PATCH_OFF * 4);
		struct page *pg;
		void *va;
		u32 *ptr;

		pg = pfn_to_page(pa >> PAGE_SHIFT);
		va = kmap_local_page(pg);
		ptr = (u32 *)(va + (pa & ~PAGE_MASK));

		if ((pa & ~PAGE_MASK) >= 8) {
			u32 *ptr_44a = ptr - 2;
			*ptr_44a = RS64_NOP; /* was already NOP at 0x44A originally */
			clflush(ptr_44a);
		}
		*ptr = RS64_BRANCH_SELF;
		clflush(ptr);
		kunmap_local(va);
	}
	mb();
}

static void halt_mec(void)
{
	u32 val = rr(regCP_MEC_CNTL);
	val |= MEC_ME1_HALT | MEC_ME2_HALT;
	wr(regCP_MEC_CNTL, val);
	udelay(100); /* driver uses 50us, we use 100 for safety */
}

static void unhalt_mec(void)
{
	u32 val = rr(regCP_MEC_CNTL);
	val &= ~(MEC_ME1_HALT | MEC_ME2_HALT);
	wr(regCP_MEC_CNTL, val);
	udelay(100);
}

static void dc_invalidate(void)
{
	int i;
	wr(regCP_MEC_DC_OP_CNTL, DC_INVALIDATE_DCACHE);
	for (i = 0; i < 1000; i++) {
		if (rr(regCP_MEC_DC_OP_CNTL) & DC_INVALIDATE_DCACHE_COMPLETE)
			break;
		udelay(10);
	}
}

static void ic_invalidate(void)
{
	int i;
	wr(regCP_CPC_IC_OP_CNTL, IC_INVALIDATE_CACHE);
	for (i = 0; i < 1000; i++) {
		if (rr(regCP_CPC_IC_OP_CNTL) & IC_INVALIDATE_CACHE_COMPLETE)
			break;
		udelay(10);
	}
}

static void ic_prime(void)
{
	int i;
	wr(regCP_CPC_IC_OP_CNTL, IC_PRIME_ICACHE);
	for (i = 0; i < 10000; i++) {
		if (rr(regCP_CPC_IC_OP_CNTL) & IC_ICACHE_PRIMED)
			break;
		udelay(10);
	}
}

static u32 read_pc(void)
{
	return rr(regCP_MEC1_INSTR_PNTR);
}

static void pipe_reset(void)
{
	u32 val = rr(regCP_MEC_CNTL);
	wr(regCP_MEC_CNTL, val | ALL_PIPE_RESET);
	udelay(50);
	wr(regCP_MEC_CNTL, val & ~ALL_PIPE_RESET);
	udelay(50);
}

static void check_pc_sequence(const char *tag)
{
	int i;
	for (i = 0; i < 8; i++) {
		msleep(5);
		pr_info("fw13: %s PC[%d]=0x%04X\n", tag, i, read_pc());
	}
}

static int __init fw13_init(void)
{
	struct pci_dev *pdev = NULL;
	u32 pc, mec_cntl, ic_op;
	int i;

	pdev = pci_get_device(AMD_VENDOR_ID, 0x1586, NULL);
	if (!pdev)
		return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	pr_info("fw13: ========================================\n");
	pr_info("fw13: PHASE 13: HALT MEC -> PRIME -> UN-HALT\n");
	pr_info("fw13: ========================================\n");

	pc = read_pc();
	mec_cntl = rr(regCP_MEC_CNTL);
	ic_op = rr(regCP_CPC_IC_OP_CNTL);
	pr_info("fw13: BASELINE: PC=0x%04X MEC_CNTL=0x%08X IC_OP=0x%08X\n",
		pc, mec_cntl, ic_op);
	pr_info("fw13: ME1_HALT=%d ME2_HALT=%d\n",
		!!(mec_cntl & MEC_ME1_HALT), !!(mec_cntl & MEC_ME2_HALT));

	/* ============================================================
	 * TEST A: HALT → patch all → PRIME → UN-HALT
	 * ============================================================ */
	pr_info("fw13: --- TEST A: HALT -> patch -> PRIME -> UN-HALT ---\n");

	halt_mec();
	pc = read_pc();
	mec_cntl = rr(regCP_MEC_CNTL);
	pr_info("fw13: A halted: PC=0x%04X MEC_CNTL=0x%08X\n", pc, mec_cntl);

	/* Verify MEC is actually halted — PC should not change */
	for (i = 0; i < 3; i++) {
		udelay(100);
		pr_info("fw13: A halted PC check[%d]=0x%04X\n", i, read_pc());
	}

	patch_all_copies(RS64_NOP);

	ic_prime();
	ic_op = rr(regCP_CPC_IC_OP_CNTL);
	pr_info("fw13: A after PRIME: IC_OP=0x%08X (PRIMED=%d)\n",
		ic_op, !!(ic_op & IC_ICACHE_PRIMED));

	unhalt_mec();
	pr_info("fw13: A un-halted\n");
	check_pc_sequence("A");

	/* Restore for next test */
	halt_mec();
	restore_all_copies();
	pipe_reset();
	unhalt_mec();
	msleep(50);
	pr_info("fw13: A RESTORED: PC=0x%04X\n", read_pc());

	/* ============================================================
	 * TEST B: HALT → patch → clear IC_OP → PRIME → UN-HALT
	 * (Force PRIMED bit to 0 first, so PRIME must actually work)
	 * ============================================================ */
	pr_info("fw13: --- TEST B: HALT -> patch -> clear IC_OP -> PRIME -> UN-HALT ---\n");

	halt_mec();
	patch_all_copies(RS64_NOP);

	/* Clear IC_OP completely — force PRIMED=0 */
	wr(regCP_CPC_IC_OP_CNTL, 0);
	udelay(50);
	ic_op = rr(regCP_CPC_IC_OP_CNTL);
	pr_info("fw13: B after clear IC_OP=0x%08X\n", ic_op);

	/* Now PRIME — PRIMED bit should go from 0→1 */
	ic_prime();
	ic_op = rr(regCP_CPC_IC_OP_CNTL);
	pr_info("fw13: B after PRIME: IC_OP=0x%08X (PRIMED=%d)\n",
		ic_op, !!(ic_op & IC_ICACHE_PRIMED));

	unhalt_mec();
	check_pc_sequence("B");

	halt_mec();
	restore_all_copies();
	pipe_reset();
	unhalt_mec();
	msleep(50);
	pr_info("fw13: B RESTORED: PC=0x%04X\n", read_pc());

	/* ============================================================
	 * TEST C: HALT → patch → per-pipe IC_BASE re-write → PRIME → UN-HALT
	 * (Re-writing IC_BASE while halted may trigger the DMA engine)
	 * ============================================================ */
	pr_info("fw13: --- TEST C: HALT -> patch -> IC_BASE rewrite -> PRIME -> UN-HALT ---\n");

	halt_mec();
	patch_all_copies(RS64_NOP);

	/* Re-write IC_BASE for each pipe while halted */
	for (i = 0; i < 2; i++) {
		u32 lo, hi;
		grbm_select(1, i, 0, 0);
		lo = rr(regCP_CPC_IC_BASE_LO);
		hi = rr(regCP_CPC_IC_BASE_HI);
		/* Write same value back — may trigger hardware DMA latch */
		wr(regCP_CPC_IC_BASE_LO, lo);
		wr(regCP_CPC_IC_BASE_HI, hi);
		pr_info("fw13: C pipe[%d] IC_BASE=0x%08X_%08X (rewritten)\n", i, hi, lo);
	}
	grbm_select(0, 0, 0, 0);

	wr(regCP_CPC_IC_OP_CNTL, 0); /* clear */
	udelay(50);
	dc_invalidate();
	ic_invalidate();
	ic_prime();
	ic_op = rr(regCP_CPC_IC_OP_CNTL);
	pr_info("fw13: C after full sequence: IC_OP=0x%08X\n", ic_op);

	unhalt_mec();
	check_pc_sequence("C");

	halt_mec();
	restore_all_copies();
	pipe_reset();
	unhalt_mec();
	msleep(50);
	pr_info("fw13: C RESTORED: PC=0x%04X\n", read_pc());

	/* ============================================================
	 * TEST D: HALT → patch → FULL driver-style sequence → UN-HALT
	 * (Exact same order as gfx_v11_0_config_mec_cache_rs64 but with PRIME)
	 * ============================================================ */
	pr_info("fw13: --- TEST D: HALT -> patch -> driver sequence + PRIME -> UN-HALT ---\n");

	halt_mec();
	patch_all_copies(RS64_NOP);

	/* Driver sequence: DC invalidate first */
	dc_invalidate();
	pr_info("fw13: D DC invalidated\n");

	/* Clear IC_OP, then invalidate IC */
	wr(regCP_CPC_IC_OP_CNTL, 0);
	udelay(10);
	ic_invalidate();
	pr_info("fw13: D IC invalidated\n");

	/* PRIME — this is the step the driver skips for MEC */
	ic_prime();
	ic_op = rr(regCP_CPC_IC_OP_CNTL);
	pr_info("fw13: D after PRIME: IC_OP=0x%08X (PRIMED=%d)\n",
		ic_op, !!(ic_op & IC_ICACHE_PRIMED));

	/* Pipe reset before un-halt (driver does this) */
	pipe_reset();

	unhalt_mec();
	check_pc_sequence("D");

	halt_mec();
	restore_all_copies();
	pipe_reset();
	unhalt_mec();
	msleep(50);
	pr_info("fw13: D RESTORED: PC=0x%04X\n", read_pc());

	/* ============================================================
	 * TEST E: HALT → NO patch → PRIME → UN-HALT (control)
	 * (If PC changes, PRIME itself disrupts MEC regardless of patch)
	 * ============================================================ */
	pr_info("fw13: --- TEST E: HALT -> PRIME (no patch) -> UN-HALT (control) ---\n");

	halt_mec();
	pc = read_pc();
	pr_info("fw13: E halted PC=0x%04X\n", pc);

	wr(regCP_CPC_IC_OP_CNTL, 0);
	udelay(50);
	dc_invalidate();
	ic_invalidate();
	ic_prime();
	ic_op = rr(regCP_CPC_IC_OP_CNTL);
	pr_info("fw13: E after PRIME: IC_OP=0x%08X\n", ic_op);

	pipe_reset();
	unhalt_mec();
	check_pc_sequence("E");

	pr_info("fw13: E FINAL: PC=0x%04X\n", read_pc());

	pr_info("fw13: ========================================\n");
	pr_info("fw13: PHASE 13 COMPLETE\n");
	pr_info("fw13: ========================================\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw13_exit(void) {}

module_init(fw13_init);
module_exit(fw13_exit);
