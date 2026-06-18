/*
 * patch_mec_fw14.c — Phase 14: Direct MEC SRAM access via UCODE_ADDR/DATA
 *
 * The traditional (non-RS64) firmware loading path writes MEC SRAM directly:
 *   WREG32(CP_MEC_ME1_UCODE_ADDR, 0)     // reset pointer
 *   for (i=0; i<size; i++)
 *       WREG32(CP_MEC_ME1_UCODE_DATA, fw[i])  // auto-increment
 *
 * RS64 uses RLC autoload + IC_BASE instead, but these registers still exist
 * in gc_11_5_0. If they work, we can directly read/write MEC instruction SRAM.
 *
 * Tests:
 *   A: Read SRAM[0x000..0x00F] — verify we get firmware instructions
 *   B: Read SRAM[0x448..0x44F] — verify we see the branch-self at 0x44C
 *   C: HALT → write NOP at 0x44C → un-halt → check PC
 *   D: HALT → write NOP at 0x44C + 0x44A → IC invalidate → un-halt
 *   E: HALT → write custom spin at NEW address (0x500) → patch 0x44C as
 *      branch to 0x500 → un-halt → check if PC moves to 0x500
 *
 * Auto-unloads (-ENODEV).
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002
#define AMD_DEV_ID    0x1586

/* GC registers (BASE_IDX=1: raw + 0xA000 for dword MMIO offset) */
#define regCP_MEC_CNTL              0x0A802
#define regCP_MEC1_INSTR_PNTR      0x021A8
#define regCP_CPC_IC_OP_CNTL       0x0C97A
#define regCP_MEC_DC_OP_CNTL       0x0C90C

/* Direct SRAM access (BASE_IDX=1) */
#define regCP_MEC_ME1_UCODE_ADDR   0x1581A
#define regCP_MEC_ME1_UCODE_DATA   0x1581B

/* CP_CPC_IC_OP_CNTL bits */
#define IC_INVALIDATE_CACHE          (1 << 0)
#define IC_INVALIDATE_CACHE_COMPLETE (1 << 1)

/* CP_MEC_CNTL bits */
#define MEC_ME1_PIPE0_RESET   (1 << 16)
#define MEC_ME1_PIPE1_RESET   (1 << 17)
#define MEC_ME1_PIPE2_RESET   (1 << 18)
#define MEC_ME1_PIPE3_RESET   (1 << 19)
#define MEC_ME2_HALT          (1 << 28)
#define MEC_ME1_HALT          (1 << 30)
#define ALL_PIPE_RESET (MEC_ME1_PIPE0_RESET | MEC_ME1_PIPE1_RESET | \
			MEC_ME1_PIPE2_RESET | MEC_ME1_PIPE3_RESET)

#define RS64_NOP          0x7C408001UL  /* s_mov s1, s0 */
#define RS64_BRANCH_SELF  0x88000000UL  /* branch to self (offset 0) */

static void __iomem *mmio;

static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }
static void wr(u32 dw_off, u32 val) {
	writel(val, mmio + (u64)dw_off * 4);
	readl(mmio + (u64)dw_off * 4); /* flush */
}

static void halt_mec(void)
{
	u32 val = rr(regCP_MEC_CNTL);
	val |= MEC_ME1_HALT | MEC_ME2_HALT;
	wr(regCP_MEC_CNTL, val);
	udelay(100);
}

static void unhalt_mec(void)
{
	u32 val = rr(regCP_MEC_CNTL);
	val &= ~(MEC_ME1_HALT | MEC_ME2_HALT);
	wr(regCP_MEC_CNTL, val);
	udelay(100);
}

/* Read SRAM at given dword address */
static u32 sram_read(u32 addr)
{
	wr(regCP_MEC_ME1_UCODE_ADDR, addr);
	udelay(10);
	return rr(regCP_MEC_ME1_UCODE_DATA);
}

/* Write SRAM at given dword address */
static void sram_write(u32 addr, u32 val)
{
	wr(regCP_MEC_ME1_UCODE_ADDR, addr);
	udelay(10);
	wr(regCP_MEC_ME1_UCODE_DATA, val);
	udelay(10);
}

static u32 read_pc(void)
{
	return rr(regCP_MEC1_INSTR_PNTR);
}

static void check_pc_sequence(const char *tag)
{
	int i;
	for (i = 0; i < 8; i++) {
		msleep(5);
		pr_info("fw14: %s PC[%d]=0x%04X\n", tag, i, read_pc());
	}
}

static int __init fw14_init(void)
{
	struct pci_dev *pdev = NULL;
	u32 pc, val, orig_44c, orig_44a;
	int i;

	pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
	if (!pdev)
		return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	pr_info("fw14: ========================================\n");
	pr_info("fw14: PHASE 14: DIRECT MEC SRAM VIA UCODE_ADDR/DATA\n");
	pr_info("fw14: ========================================\n");

	pc = read_pc();
	pr_info("fw14: BASELINE: PC=0x%04X\n", pc);

	/* ============================================================
	 * TEST A: Read SRAM[0x000..0x00F] — verify firmware access
	 * ============================================================ */
	pr_info("fw14: --- TEST A: Read SRAM[0x000..0x00F] ---\n");
	for (i = 0; i < 16; i++) {
		val = sram_read(i);
		pr_info("fw14: A SRAM[0x%03X] = 0x%08X\n", i, val);
	}

	/* ============================================================
	 * TEST B: Read SRAM around the spin point (0x448..0x450)
	 * ============================================================ */
	pr_info("fw14: --- TEST B: Read SRAM[0x448..0x450] ---\n");
	for (i = 0x448; i <= 0x450; i++) {
		val = sram_read(i);
		pr_info("fw14: B SRAM[0x%03X] = 0x%08X%s\n", i, val,
			(val == RS64_BRANCH_SELF) ? " <<< BRANCH_SELF" :
			(val == RS64_NOP) ? " <<< NOP" : "");
	}

	/* Save originals for restore */
	orig_44c = sram_read(0x44C);
	orig_44a = sram_read(0x44A);
	pr_info("fw14: B originals: [0x44A]=0x%08X [0x44C]=0x%08X\n",
		orig_44a, orig_44c);

	/* ============================================================
	 * TEST C: HALT → write NOP at 0x44C → UN-HALT
	 * ============================================================ */
	pr_info("fw14: --- TEST C: HALT -> SRAM write NOP@44C -> UN-HALT ---\n");

	halt_mec();
	pr_info("fw14: C halted, PC=0x%04X\n", read_pc());

	/* Write NOP at 0x44C */
	sram_write(0x44C, RS64_NOP);
	val = sram_read(0x44C);
	pr_info("fw14: C SRAM[0x44C] after write = 0x%08X (wanted 0x%08X)\n",
		val, RS64_NOP);

	unhalt_mec();
	check_pc_sequence("C");

	/* Restore */
	halt_mec();
	sram_write(0x44C, orig_44c);
	unhalt_mec();
	msleep(50);
	pr_info("fw14: C RESTORED: PC=0x%04X SRAM[0x44C]=0x%08X\n",
		read_pc(), sram_read(0x44C));

	/* ============================================================
	 * TEST D: HALT → write NOP at 0x44A + 0x44C → IC invalidate → UN-HALT
	 * ============================================================ */
	pr_info("fw14: --- TEST D: HALT -> NOP@44A+44C -> IC inv -> UN-HALT ---\n");

	halt_mec();

	sram_write(0x44A, RS64_NOP);
	sram_write(0x44C, RS64_NOP);
	pr_info("fw14: D SRAM[0x44A]=0x%08X SRAM[0x44C]=0x%08X\n",
		sram_read(0x44A), sram_read(0x44C));

	/* IC invalidate to flush any cached instructions */
	wr(regCP_CPC_IC_OP_CNTL, IC_INVALIDATE_CACHE);
	udelay(100);
	pr_info("fw14: D IC_OP=0x%08X\n", rr(regCP_CPC_IC_OP_CNTL));

	unhalt_mec();
	check_pc_sequence("D");

	/* Restore */
	halt_mec();
	sram_write(0x44A, orig_44a);
	sram_write(0x44C, orig_44c);
	unhalt_mec();
	msleep(50);
	pr_info("fw14: D RESTORED: PC=0x%04X\n", read_pc());

	/* ============================================================
	 * TEST E: Write spin at 0x500, branch from 0x44C to 0x500
	 * If PC moves to 0x500, we have confirmed SRAM write works!
	 * RS64 branch: 0x88000000 | (offset & 0xFFFFFF)
	 * branch target = PC + 1 + offset
	 * From 0x44C to 0x500: offset = 0x500 - 0x44C - 1 = 0xB3
	 * Instruction: 0x880000B3
	 * ============================================================ */
	pr_info("fw14: --- TEST E: Branch from 0x44C to spin@0x500 ---\n");

	halt_mec();

	/* Write a branch-self at 0x500 as landing target */
	sram_write(0x500, RS64_BRANCH_SELF);
	pr_info("fw14: E SRAM[0x500]=0x%08X\n", sram_read(0x500));

	/* Write branch to 0x500 at current spin location 0x44C */
	sram_write(0x44C, 0x880000B3UL); /* branch +0xB3: 0x44C+1+0xB3 = 0x500 */
	pr_info("fw14: E SRAM[0x44C]=0x%08X (branch to 0x500)\n", sram_read(0x44C));

	unhalt_mec();
	check_pc_sequence("E");

	/* Restore */
	halt_mec();
	sram_write(0x44C, orig_44c);
	sram_write(0x500, 0); /* clear our landing pad */
	unhalt_mec();
	msleep(50);
	pr_info("fw14: E RESTORED: PC=0x%04X\n", read_pc());

	pr_info("fw14: ========================================\n");
	pr_info("fw14: PHASE 14 COMPLETE\n");
	pr_info("fw14: ========================================\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw14_exit(void) {}

module_init(fw14_init);
module_exit(fw14_exit);
