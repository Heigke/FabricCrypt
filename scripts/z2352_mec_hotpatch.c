/*
 * z2352_mec_hotpatch.c — MEC firmware hot-patch module v6
 *
 * REQUIRES: amdgpu.cg_mask=0 (no clock gating)
 *
 * mode=0: READ-ONLY probe (verify register access)
 * mode=1: Test IC_BASE writability (direct, then with CPC soft reset)
 * mode=2: Full injection — dump UCODE_RAM → VRAM, redirect IC_BASE
 * mode=3: Verify state after injection
 * mode=4: GPA_OVERRIDE test — set CPC_PSP_DEBUG.GPA_OVERRIDE then test IC_BASE
 * mode=5: Full GPA injection — GPA_OVERRIDE + halt + invalidate + redirect IC_BASE
 * mode=6: PM4 KIQ injection — write CPC_PSP_DEBUG via GPU-internal PM4 path
 *
 * v6 ADDS:
 *   - Mode 6: PM4 KIQ register write test via amdgpu_kiq_wreg/rreg
 *     Uses kprobe trick to resolve unexported amdgpu symbols at runtime.
 *     Tests whether GPU-internal register bus bypasses PSP write locks.
 *
 * BUILD: make -C /lib/modules/$(uname -r)/build M=$(pwd)/scripts \
 *          obj-m=z2352_mec_hotpatch.o modules
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/spinlock.h>
#include <linux/kprobes.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("FEEL Project");
MODULE_DESCRIPTION("MEC firmware hot-patch v6 for gfx11.5.1");

static int mode = 0;
module_param(mode, int, 0444);
MODULE_PARM_DESC(mode, "0=probe, 1=test-writes+softreset, 2=inject, 3=verify, 4=gpa-test, 5=gpa-inject, 6=pm4-kiq");

#define AMD_VENDOR_ID  0x1002

/* GC base offsets (dword) — verified by register probe:
 * SEG0 = BASE_IDX=0 registers (direct MMIO, within 1MB BAR5)
 * SEG1 = BASE_IDX=1 registers (indirect PCIe access via INDEX2/DATA2)
 * Note: SEG0=0x1260 matches hardcoded Sienna Cichlid value and works.
 * SEG1=0xC00000 from IP discovery table — the old 0xA000 was wrong. */
#define GC_BASE_0  0x1260     /* SEG0 dword base (confirmed: GRBM_STATUS=0x302C) */
#define GC_BASE_1  0xC00000   /* SEG1 dword base (confirmed: CPC_PSP_DEBUG=0x0) */

/* PCIe indirect register access (NBIF)
 * For registers at dword offsets beyond BAR5 direct range.
 * Write byte address to INDEX2, read/write value at DATA2. */
#define PCIE_INDEX2  0x000e   /* dword offset in BAR5 */
#define PCIE_DATA2   0x000f   /* dword offset in BAR5 */

/* BASE_IDX=0 registers (direct MMIO via SEG0) */
#define mmGRBM_STATUS        (GC_BASE_0 + 0x0DA4)
#define mmGRBM_SOFT_RESET    (GC_BASE_0 + 0x0DA8)

/* BASE_IDX=1 registers (indirect via SEG1) */
#define mmCP_MEC_CNTL           (GC_BASE_1 + 0x0802)
#define mmCP_MEC_ME1_UCODE_ADDR (GC_BASE_1 + 0x581A)
#define mmCP_MEC_ME1_UCODE_DATA (GC_BASE_1 + 0x581B)
#define mmCP_CPC_IC_OP_CNTL     (GC_BASE_1 + 0x297A)
#define mmCP_CPC_IC_BASE_CNTL   (GC_BASE_1 + 0x584E)
#define mmCP_CPC_IC_BASE_LO     (GC_BASE_1 + 0x584C)
#define mmCP_CPC_IC_BASE_HI     (GC_BASE_1 + 0x584D)

/* CPC/CPG PSP debug — GPA_OVERRIDE is bit 3 */
#define mmCPC_PSP_DEBUG         (GC_BASE_1 + 0x5C11)
#define mmCPG_PSP_DEBUG         (GC_BASE_1 + 0x5C10)
#define GPA_OVERRIDE            (1 << 3)  /* 0x00000008 */

/* CP_MEC_CNTL bits (gc_11_5_0_sh_mask.h) */
#define MEC_ME1_HALT  (1 << 30) /* 0x40000000 */
#define MEC_ME2_HALT  (1 << 28) /* 0x10000000 */

/* GRBM_SOFT_RESET bits */
#define SOFT_RESET_CP   (1 << 0)   /* bit 0 */
#define SOFT_RESET_CPF  (1 << 17)  /* bit 17 */
#define SOFT_RESET_CPC  (1 << 18)  /* bit 18 */

/* IC invalidation bits */
#define IC_INVALIDATE       (1 << 0)
#define IC_INVALIDATE_DONE  (1 << 1)

/* MEC firmware size: ~67K dwords (268160 bytes / 4 = 67040 dwords) */
#define MEC_FW_DWORDS  67040

/* VRAM offset for firmware copy (1MB into VRAM) */
#define VRAM_FW_OFFSET 0x100000

/* FEEL marker */
#define FEEL_MARKER  0xFEE10001

static void __iomem *mmio;
static struct pci_dev *gpu_dev;
static resource_size_t bar5_size; /* bytes */
static DEFINE_SPINLOCK(indirect_lock);

/*
 * rr() — Read GPU register (dword offset)
 *
 * If dword_offset * 4 < BAR5 size → direct MMIO readl
 * Otherwise → indirect via PCIE_INDEX2/DATA2
 *
 * This matches the kernel's amdgpu_device_rreg() logic.
 */
static u32 rr(u32 off)
{
	if ((u64)off * 4 < bar5_size) {
		return readl(mmio + (u64)off * 4);
	} else {
		/* Indirect PCIe register access */
		u32 r;
		unsigned long flags;
		u32 byte_addr = off * 4;

		spin_lock_irqsave(&indirect_lock, flags);
		writel(byte_addr, mmio + (u64)PCIE_INDEX2 * 4);
		readl(mmio + (u64)PCIE_INDEX2 * 4); /* flush posted write */
		r = readl(mmio + (u64)PCIE_DATA2 * 4);
		spin_unlock_irqrestore(&indirect_lock, flags);
		return r;
	}
}

/*
 * wr() — Write GPU register (dword offset)
 */
static void wr(u32 off, u32 val)
{
	if ((u64)off * 4 < bar5_size) {
		writel(val, mmio + (u64)off * 4);
	} else {
		unsigned long flags;
		u32 byte_addr = off * 4;

		spin_lock_irqsave(&indirect_lock, flags);
		writel(byte_addr, mmio + (u64)PCIE_INDEX2 * 4);
		readl(mmio + (u64)PCIE_INDEX2 * 4); /* flush */
		writel(val, mmio + (u64)PCIE_DATA2 * 4);
		spin_unlock_irqrestore(&indirect_lock, flags);
	}
}

static void halt_mec(void)
{
	u32 v = rr(mmCP_MEC_CNTL);
	v |= MEC_ME1_HALT | MEC_ME2_HALT;
	wr(mmCP_MEC_CNTL, v);
	udelay(100);
}

static void unhalt_mec(void)
{
	u32 v = rr(mmCP_MEC_CNTL);
	v &= ~(MEC_ME1_HALT | MEC_ME2_HALT);
	wr(mmCP_MEC_CNTL, v);
	udelay(100);
}

static int invalidate_ic(void)
{
	u32 v;
	int i;

	v = rr(mmCP_CPC_IC_OP_CNTL);
	v |= IC_INVALIDATE;
	wr(mmCP_CPC_IC_OP_CNTL, v);

	for (i = 0; i < 50000; i++) {
		v = rr(mmCP_CPC_IC_OP_CNTL);
		if (v & IC_INVALIDATE_DONE)
			return 0;
		udelay(1);
	}
	return -ETIMEDOUT;
}

/* ================================================================ */
/* MODE 0: Read-only probe                                          */
/* ================================================================ */
static int do_probe(void)
{
	u32 grbm, mec, ic_lo, ic_hi, ic_cntl, ic_op, soft_rst;
	int i, nz = 0;

	pr_info("z2352: === PROBE v5 (correct GC bases) ===\n");
	pr_info("z2352: GC_BASE_0=0x%X (SEG0, direct), GC_BASE_1=0x%X (SEG1, indirect)\n",
		GC_BASE_0, GC_BASE_1);
	pr_info("z2352: BAR5 size=%llu bytes, indirect threshold=dword 0x%llX\n",
		(u64)bar5_size, (u64)bar5_size / 4);
	pr_info("z2352: GRBM_STATUS @ dword 0x%X (direct), CP_MEC_CNTL @ dword 0x%X (indirect)\n",
		mmGRBM_STATUS, mmCP_MEC_CNTL);

	grbm = rr(mmGRBM_STATUS);
	pr_info("z2352: GRBM_STATUS = 0x%08X %s\n",
		grbm, grbm ? "ACCESSIBLE" : "BLOCKED");
	if (!grbm) return -EACCES;

	soft_rst = rr(mmGRBM_SOFT_RESET);
	pr_info("z2352: GRBM_SOFT_RESET = 0x%08X\n", soft_rst);

	mec = rr(mmCP_MEC_CNTL);
	pr_info("z2352: CP_MEC_CNTL = 0x%08X (ME1_HALT=%d, ME2_HALT=%d)\n",
		mec, !!(mec & MEC_ME1_HALT), !!(mec & MEC_ME2_HALT));

	ic_lo   = rr(mmCP_CPC_IC_BASE_LO);
	ic_hi   = rr(mmCP_CPC_IC_BASE_HI);
	ic_cntl = rr(mmCP_CPC_IC_BASE_CNTL);
	ic_op   = rr(mmCP_CPC_IC_OP_CNTL);
	pr_info("z2352: IC_BASE = 0x%08X:%08X cntl=0x%08X op=0x%08X\n",
		ic_hi, ic_lo, ic_cntl, ic_op);

	/* CPC/CPG PSP debug registers */
	{
		u32 cpc_psp = rr(mmCPC_PSP_DEBUG);
		u32 cpg_psp = rr(mmCPG_PSP_DEBUG);
		pr_info("z2352: CPC_PSP_DEBUG = 0x%08X (GPA_OVERRIDE=%d)\n",
			cpc_psp, !!(cpc_psp & GPA_OVERRIDE));
		pr_info("z2352: CPG_PSP_DEBUG = 0x%08X (GPA_OVERRIDE=%d)\n",
			cpg_psp, !!(cpg_psp & GPA_OVERRIDE));
	}

	/* Read first 8 UCODE_RAM entries */
	wr(mmCP_MEC_ME1_UCODE_ADDR, 0);
	for (i = 0; i < 8; i++) {
		u32 v = rr(mmCP_MEC_ME1_UCODE_DATA);
		if (v) nz++;
		pr_info("z2352: UCODE[%d] = 0x%08X\n", i, v);
	}
	pr_info("z2352: %d/8 UCODE entries non-zero\n", nz);

	return 0;
}

/* ================================================================ */
/* MODE 1: Test CPC soft reset → IC_BASE writability                 */
/* ================================================================ */
static int do_test_writes(void)
{
	u32 orig_lo, orig_hi, orig_cntl;
	u32 mec, readback, soft_rst;

	pr_info("z2352: === TEST WRITES v3 (with CPC soft reset) ===\n");

	if (!rr(mmGRBM_STATUS)) {
		pr_err("z2352: registers not accessible\n");
		return -EACCES;
	}

	/* Save originals */
	orig_lo   = rr(mmCP_CPC_IC_BASE_LO);
	orig_hi   = rr(mmCP_CPC_IC_BASE_HI);
	orig_cntl = rr(mmCP_CPC_IC_BASE_CNTL);
	pr_info("z2352: Original IC_BASE = 0x%08X:%08X cntl=0x%08X\n",
		orig_hi, orig_lo, orig_cntl);

	/* Step 1: Halt MEC first */
	halt_mec();
	mec = rr(mmCP_MEC_CNTL);
	pr_info("z2352: Step 1 — MEC halted: 0x%08X (ME1=%d)\n",
		mec, !!(mec & MEC_ME1_HALT));

	/* Step 2: Try IC_BASE write WITHOUT soft reset (should fail) */
	wr(mmCP_CPC_IC_BASE_LO, 0xDEAD0000);
	readback = rr(mmCP_CPC_IC_BASE_LO);
	pr_info("z2352: Step 2 — Without reset: wrote 0xDEAD0000, read 0x%08X %s\n",
		readback,
		readback == 0xDEAD0000 ? "WRITABLE!" : "LOCKED");

	/* Step 3: Soft-reset CPC */
	soft_rst = rr(mmGRBM_SOFT_RESET);
	pr_info("z2352: Step 3 — GRBM_SOFT_RESET before = 0x%08X\n", soft_rst);

	soft_rst |= SOFT_RESET_CPC;
	wr(mmGRBM_SOFT_RESET, soft_rst);
	udelay(50);

	readback = rr(mmGRBM_SOFT_RESET);
	pr_info("z2352: Step 3 — GRBM_SOFT_RESET after set = 0x%08X (CPC=%d)\n",
		readback, !!(readback & SOFT_RESET_CPC));

	/* Step 4: Try IC_BASE write WITH CPC in soft reset */
	wr(mmCP_CPC_IC_BASE_LO, 0xDEAD0000);
	readback = rr(mmCP_CPC_IC_BASE_LO);
	pr_info("z2352: Step 4 — With CPC reset: wrote 0xDEAD0000, read 0x%08X %s\n",
		readback,
		readback == 0xDEAD0000 ? "WRITABLE!" : "STILL LOCKED");

	/* Step 5: Clear soft reset */
	soft_rst &= ~SOFT_RESET_CPC;
	wr(mmGRBM_SOFT_RESET, soft_rst);
	udelay(50);

	/* Step 6: Try IC_BASE write AFTER clearing soft reset */
	wr(mmCP_CPC_IC_BASE_LO, 0xDEAD0000);
	readback = rr(mmCP_CPC_IC_BASE_LO);
	pr_info("z2352: Step 6 — After reset clear: wrote 0xDEAD0000, read 0x%08X %s\n",
		readback,
		readback == 0xDEAD0000 ? "WRITABLE!" : "LOCKED AGAIN");

	/* Step 7: Also try soft-resetting CP + CPC + CPF together (like driver does) */
	if (readback != 0xDEAD0000) {
		pr_info("z2352: Step 7 — Trying full CP+CPC+CPF soft reset\n");
		soft_rst = rr(mmGRBM_SOFT_RESET);
		soft_rst |= SOFT_RESET_CP | SOFT_RESET_CPC | SOFT_RESET_CPF;
		wr(mmGRBM_SOFT_RESET, soft_rst);
		udelay(50);

		wr(mmCP_CPC_IC_BASE_LO, 0xDEAD0000);
		readback = rr(mmCP_CPC_IC_BASE_LO);
		pr_info("z2352: Step 7a — During full reset: wrote 0xDEAD0000, read 0x%08X %s\n",
			readback,
			readback == 0xDEAD0000 ? "WRITABLE!" : "STILL LOCKED");

		/* Clear full reset */
		soft_rst &= ~(SOFT_RESET_CP | SOFT_RESET_CPC | SOFT_RESET_CPF);
		wr(mmGRBM_SOFT_RESET, soft_rst);
		udelay(50);

		/* Try after clearing full reset */
		wr(mmCP_CPC_IC_BASE_LO, 0xDEAD0000);
		readback = rr(mmCP_CPC_IC_BASE_LO);
		pr_info("z2352: Step 7b — After full reset clear: wrote 0xDEAD0000, read 0x%08X %s\n",
			readback,
			readback == 0xDEAD0000 ? "WRITABLE!" : "LOCKED");
	}

	/* ALWAYS restore original IC_BASE */
	wr(mmCP_CPC_IC_BASE_LO, orig_lo);
	wr(mmCP_CPC_IC_BASE_HI, orig_hi);
	wr(mmCP_CPC_IC_BASE_CNTL, orig_cntl);

	/* Verify restoration */
	readback = rr(mmCP_CPC_IC_BASE_LO);
	pr_info("z2352: Restored IC_BASE_LO = 0x%08X (want 0x%08X) %s\n",
		readback, orig_lo,
		readback == orig_lo ? "OK" : "RESTORE FAILED");

	/* Unhalt MEC */
	unhalt_mec();
	mec = rr(mmCP_MEC_CNTL);
	pr_info("z2352: MEC unhalted: 0x%08X (ME1=%d)\n", mec, !!(mec & MEC_ME1_HALT));

	/* Verify IC invalidation still works */
	{
		int r = invalidate_ic();
		pr_info("z2352: IC invalidation: %s\n", r == 0 ? "OK" : "TIMEOUT");
	}

	pr_info("z2352: === TEST WRITES v3 COMPLETE ===\n");
	return 0;
}

/* ================================================================ */
/* MODE 2: Full injection via VRAM + IC_BASE redirect                */
/* ================================================================ */
static int do_inject(void)
{
	void __iomem *vram;
	resource_size_t bar0_start, bar0_len;
	u32 orig_lo, orig_hi, orig_cntl;
	u32 readback, soft_rst;
	int i, nz;

	pr_info("z2352: === FULL INJECTION v3 ===\n");

	if (!rr(mmGRBM_STATUS)) {
		pr_err("z2352: registers not accessible\n");
		return -EACCES;
	}

	/* Save original IC_BASE */
	orig_lo   = rr(mmCP_CPC_IC_BASE_LO);
	orig_hi   = rr(mmCP_CPC_IC_BASE_HI);
	orig_cntl = rr(mmCP_CPC_IC_BASE_CNTL);
	pr_info("z2352: Original IC_BASE = 0x%08X:%08X cntl=0x%08X\n",
		orig_hi, orig_lo, orig_cntl);

	/* Map BAR0 (VRAM) — need enough for firmware copy */
	bar0_start = pci_resource_start(gpu_dev, 0);
	bar0_len = pci_resource_len(gpu_dev, 0);
	pr_info("z2352: BAR0 start=0x%llX len=0x%llX (%llu MB)\n",
		(u64)bar0_start, (u64)bar0_len, (u64)bar0_len / (1024*1024));

	if (bar0_len == 0) {
		pr_err("z2352: BAR0 not available\n");
		return -ENODEV;
	}

	/* Map 2MB of VRAM for firmware copy area */
	vram = ioremap(bar0_start, min_t(u64, bar0_len, 0x200000));
	if (!vram) {
		pr_err("z2352: Failed to map BAR0\n");
		return -ENOMEM;
	}

	/* Step 1: Halt MEC */
	halt_mec();
	pr_info("z2352: Step 1 — MEC halted: 0x%08X\n", rr(mmCP_MEC_CNTL));

	/* Step 2: Read UCODE_RAM (decrypted firmware) and copy to VRAM */
	pr_info("z2352: Step 2 — Copying UCODE_RAM (%d dwords) to VRAM+0x%X\n",
		MEC_FW_DWORDS, VRAM_FW_OFFSET);

	wr(mmCP_MEC_ME1_UCODE_ADDR, 0);
	nz = 0;
	for (i = 0; i < MEC_FW_DWORDS; i++) {
		u32 v = rr(mmCP_MEC_ME1_UCODE_DATA);
		writel(v, vram + VRAM_FW_OFFSET + i * 4);
		if (v) nz++;
	}
	pr_info("z2352: Copied %d/%d non-zero dwords\n", nz, MEC_FW_DWORDS);

	/* Verify first 4 dwords */
	for (i = 0; i < 4; i++) {
		readback = readl(vram + VRAM_FW_OFFSET + i * 4);
		pr_info("z2352: VRAM_FW[%d] = 0x%08X\n", i, readback);
	}

	/* Step 3: Inject FEEL marker in zero region (dword 18871 = first zero region) */
	pr_info("z2352: Step 3 — Injecting FEEL marker at dword 18871\n");
	writel(FEEL_MARKER, vram + VRAM_FW_OFFSET + 18871 * 4);
	readback = readl(vram + VRAM_FW_OFFSET + 18871 * 4);
	pr_info("z2352: VRAM_FW[18871] = 0x%08X %s\n",
		readback, readback == FEEL_MARKER ? "INJECTED" : "FAILED");

	/* Step 4: Soft-reset CPC to unlock IC_BASE */
	pr_info("z2352: Step 4 — CPC soft reset to unlock IC_BASE\n");
	soft_rst = rr(mmGRBM_SOFT_RESET);
	soft_rst |= SOFT_RESET_CP | SOFT_RESET_CPC | SOFT_RESET_CPF;
	wr(mmGRBM_SOFT_RESET, soft_rst);
	udelay(50);

	/* Step 5: Write IC_BASE to point to VRAM firmware copy */
	/* VRAM MC address for offset 0x100000 is typically just 0x100000
	 * (VRAM base in GPU MC space is 0x0 for first VRAM segment) */
	pr_info("z2352: Step 5 — Redirecting IC_BASE to VRAM+0x%X\n", VRAM_FW_OFFSET);
	wr(mmCP_CPC_IC_BASE_LO, VRAM_FW_OFFSET);
	wr(mmCP_CPC_IC_BASE_HI, 0);

	/* Set IC_BASE_CNTL: VMID=0, CACHE_POLICY=0, EXE_DISABLE=0, ADDRESS_CLAMP=1 */
	wr(mmCP_CPC_IC_BASE_CNTL, (1 << 4)); /* ADDRESS_CLAMP bit 4 */

	readback = rr(mmCP_CPC_IC_BASE_LO);
	pr_info("z2352: IC_BASE_LO = 0x%08X (want 0x%08X) %s\n",
		readback, VRAM_FW_OFFSET,
		readback == VRAM_FW_OFFSET ? "REDIRECTED!" : "FAILED");

	/* Step 6: Clear soft reset */
	soft_rst &= ~(SOFT_RESET_CP | SOFT_RESET_CPC | SOFT_RESET_CPF);
	wr(mmGRBM_SOFT_RESET, soft_rst);
	udelay(50);

	/* Check if IC_BASE held after clearing reset */
	readback = rr(mmCP_CPC_IC_BASE_LO);
	pr_info("z2352: After reset clear: IC_BASE_LO = 0x%08X %s\n",
		readback,
		readback == VRAM_FW_OFFSET ? "HELD!" : "REVERTED");

	if (readback != VRAM_FW_OFFSET) {
		/* IC_BASE didn't hold — restore and bail */
		pr_warn("z2352: IC_BASE redirect failed — restoring original\n");
		wr(mmCP_CPC_IC_BASE_LO, orig_lo);
		wr(mmCP_CPC_IC_BASE_HI, orig_hi);
		wr(mmCP_CPC_IC_BASE_CNTL, orig_cntl);
		unhalt_mec();
		iounmap(vram);
		return -EPERM;
	}

	/* Step 7: Invalidate IC cache */
	pr_info("z2352: Step 7 — IC invalidation\n");
	{
		int r = invalidate_ic();
		pr_info("z2352: IC invalidation: %s\n", r == 0 ? "OK" : "TIMEOUT");
	}

	/* Step 8: Unhalt MEC with new firmware location */
	pr_info("z2352: Step 8 — Unhalting MEC with redirected IC_BASE\n");
	unhalt_mec();
	udelay(1000); /* 1ms for MEC to start fetching */

	pr_info("z2352: CP_MEC_CNTL = 0x%08X\n", rr(mmCP_MEC_CNTL));
	pr_info("z2352: GRBM_STATUS = 0x%08X\n", rr(mmGRBM_STATUS));
	pr_info("z2352: IC_BASE = 0x%08X:%08X\n",
		rr(mmCP_CPC_IC_BASE_HI), rr(mmCP_CPC_IC_BASE_LO));

	iounmap(vram);

	pr_info("z2352: === INJECTION v3 COMPLETE ===\n");
	return 0;
}

/* ================================================================ */
/* MODE 3: Verify                                                    */
/* ================================================================ */
static int do_verify(void)
{
	pr_info("z2352: === VERIFY v3 ===\n");
	pr_info("z2352: GRBM_STATUS = 0x%08X\n", rr(mmGRBM_STATUS));
	pr_info("z2352: CP_MEC_CNTL = 0x%08X\n", rr(mmCP_MEC_CNTL));
	pr_info("z2352: IC_BASE = 0x%08X:%08X cntl=0x%08X\n",
		rr(mmCP_CPC_IC_BASE_HI), rr(mmCP_CPC_IC_BASE_LO),
		rr(mmCP_CPC_IC_BASE_CNTL));
	pr_info("z2352: GRBM_SOFT_RESET = 0x%08X\n", rr(mmGRBM_SOFT_RESET));

	/* Read UCODE[220] to check for FEEL marker */
	wr(mmCP_MEC_ME1_UCODE_ADDR, 220);
	pr_info("z2352: UCODE[220] = 0x%08X\n", rr(mmCP_MEC_ME1_UCODE_DATA));

	return 0;
}

/* ================================================================ */
/* MODE 4: GPA_OVERRIDE test — does setting GPA unlock IC_BASE?      */
/* ================================================================ */
static int do_gpa_test(void)
{
	u32 orig_lo, orig_hi, orig_cntl;
	u32 orig_cpc_psp, orig_cpg_psp;
	u32 readback;

	pr_info("z2352: === GPA_OVERRIDE TEST v4 ===\n");

	if (!rr(mmGRBM_STATUS)) {
		pr_err("z2352: registers not accessible\n");
		return -EACCES;
	}

	/* Save originals */
	orig_lo      = rr(mmCP_CPC_IC_BASE_LO);
	orig_hi      = rr(mmCP_CPC_IC_BASE_HI);
	orig_cntl    = rr(mmCP_CPC_IC_BASE_CNTL);
	orig_cpc_psp = rr(mmCPC_PSP_DEBUG);
	orig_cpg_psp = rr(mmCPG_PSP_DEBUG);

	pr_info("z2352: Original IC_BASE = 0x%08X:%08X cntl=0x%08X\n",
		orig_hi, orig_lo, orig_cntl);
	pr_info("z2352: Original CPC_PSP_DEBUG = 0x%08X (GPA=%d)\n",
		orig_cpc_psp, !!(orig_cpc_psp & GPA_OVERRIDE));
	pr_info("z2352: Original CPG_PSP_DEBUG = 0x%08X (GPA=%d)\n",
		orig_cpg_psp, !!(orig_cpg_psp & GPA_OVERRIDE));

	/* Step 1: Test IC_BASE write WITHOUT GPA (control — expect LOCKED) */
	halt_mec();
	pr_info("z2352: Step 1 — MEC halted: 0x%08X\n", rr(mmCP_MEC_CNTL));

	wr(mmCP_CPC_IC_BASE_LO, 0xDEAD0000);
	readback = rr(mmCP_CPC_IC_BASE_LO);
	pr_info("z2352: Step 1 — Without GPA: wrote 0xDEAD0000, read 0x%08X %s\n",
		readback, readback == 0xDEAD0000 ? "WRITABLE!" : "LOCKED");

	/* Step 2: Set GPA_OVERRIDE on both CPC and CPG */
	pr_info("z2352: Step 2 — Setting GPA_OVERRIDE\n");
	wr(mmCPC_PSP_DEBUG, orig_cpc_psp | GPA_OVERRIDE);
	wr(mmCPG_PSP_DEBUG, orig_cpg_psp | GPA_OVERRIDE);
	udelay(100);

	readback = rr(mmCPC_PSP_DEBUG);
	pr_info("z2352: CPC_PSP_DEBUG after set = 0x%08X (GPA=%d) %s\n",
		readback, !!(readback & GPA_OVERRIDE),
		(readback & GPA_OVERRIDE) ? "SET OK" : "WRITE BLOCKED");

	readback = rr(mmCPG_PSP_DEBUG);
	pr_info("z2352: CPG_PSP_DEBUG after set = 0x%08X (GPA=%d) %s\n",
		readback, !!(readback & GPA_OVERRIDE),
		(readback & GPA_OVERRIDE) ? "SET OK" : "WRITE BLOCKED");

	/* Step 3: Test IC_BASE write WITH GPA_OVERRIDE set */
	wr(mmCP_CPC_IC_BASE_LO, 0xDEAD0000);
	readback = rr(mmCP_CPC_IC_BASE_LO);
	pr_info("z2352: Step 3 — With GPA: wrote 0xDEAD0000, read 0x%08X %s\n",
		readback, readback == 0xDEAD0000 ? "WRITABLE!" : "STILL LOCKED");

	/* Step 4: Try GPA + IC invalidation + IC_BASE write */
	if (readback != 0xDEAD0000) {
		int r;
		pr_info("z2352: Step 4 — Trying GPA + IC invalidation\n");
		r = invalidate_ic();
		pr_info("z2352: IC invalidation: %s\n", r == 0 ? "OK" : "TIMEOUT");

		wr(mmCP_CPC_IC_BASE_LO, 0xDEAD0000);
		readback = rr(mmCP_CPC_IC_BASE_LO);
		pr_info("z2352: Step 4 — GPA + invalidation: wrote 0xDEAD0000, read 0x%08X %s\n",
			readback, readback == 0xDEAD0000 ? "WRITABLE!" : "STILL LOCKED");
	}

	/* Step 5: Try GPA + CPC soft reset + IC_BASE write */
	if (readback != 0xDEAD0000) {
		u32 soft_rst;
		pr_info("z2352: Step 5 — Trying GPA + CPC soft reset\n");
		soft_rst = rr(mmGRBM_SOFT_RESET);
		soft_rst |= SOFT_RESET_CP | SOFT_RESET_CPC | SOFT_RESET_CPF;
		wr(mmGRBM_SOFT_RESET, soft_rst);
		udelay(50);

		wr(mmCP_CPC_IC_BASE_LO, 0xDEAD0000);
		readback = rr(mmCP_CPC_IC_BASE_LO);
		pr_info("z2352: Step 5a — GPA + reset during: wrote 0xDEAD0000, read 0x%08X %s\n",
			readback, readback == 0xDEAD0000 ? "WRITABLE!" : "STILL LOCKED");

		/* Clear soft reset */
		soft_rst &= ~(SOFT_RESET_CP | SOFT_RESET_CPC | SOFT_RESET_CPF);
		wr(mmGRBM_SOFT_RESET, soft_rst);
		udelay(50);

		/* Test after clearing reset */
		wr(mmCP_CPC_IC_BASE_LO, 0xDEAD0000);
		readback = rr(mmCP_CPC_IC_BASE_LO);
		pr_info("z2352: Step 5b — GPA + after reset: wrote 0xDEAD0000, read 0x%08X %s\n",
			readback, readback == 0xDEAD0000 ? "WRITABLE!" : "STILL LOCKED");
	}

	/* Step 6: Test UCODE_ADDR/DATA writability (these should work regardless) */
	{
		u32 test_val;
		wr(mmCP_MEC_ME1_UCODE_ADDR, 0);
		test_val = rr(mmCP_MEC_ME1_UCODE_DATA);
		pr_info("z2352: Step 6 — UCODE[0] read = 0x%08X\n", test_val);

		/* Try writing jump table (addr 0, data = test pattern) */
		wr(mmCP_MEC_ME1_UCODE_ADDR, 0);
		wr(mmCP_MEC_ME1_UCODE_DATA, 0xFEE10001);
		wr(mmCP_MEC_ME1_UCODE_ADDR, 0);
		readback = rr(mmCP_MEC_ME1_UCODE_DATA);
		pr_info("z2352: Step 6 — UCODE write test: wrote 0xFEE10001, read 0x%08X %s\n",
			readback, readback == 0xFEE10001 ? "WRITABLE!" : "LOCKED");

		/* Restore original value */
		wr(mmCP_MEC_ME1_UCODE_ADDR, 0);
		wr(mmCP_MEC_ME1_UCODE_DATA, test_val);
	}

	/* Restore everything */
	wr(mmCP_CPC_IC_BASE_LO, orig_lo);
	wr(mmCP_CPC_IC_BASE_HI, orig_hi);
	wr(mmCP_CPC_IC_BASE_CNTL, orig_cntl);
	wr(mmCPC_PSP_DEBUG, orig_cpc_psp);
	wr(mmCPG_PSP_DEBUG, orig_cpg_psp);
	unhalt_mec();

	pr_info("z2352: Restored all registers, MEC unhalted\n");
	pr_info("z2352: === GPA_OVERRIDE TEST v4 COMPLETE ===\n");
	return 0;
}

/* ================================================================ */
/* MODE 5: Full GPA injection — GPA + halt + IC redirect + restart  */
/* ================================================================ */
static int do_gpa_inject(void)
{
	void __iomem *vram;
	resource_size_t bar0_start, bar0_len;
	u32 orig_lo, orig_hi;
	u32 readback;
	int i, nz, r;

	pr_info("z2352: === GPA INJECTION v4 ===\n");

	if (!rr(mmGRBM_STATUS)) {
		pr_err("z2352: registers not accessible\n");
		return -EACCES;
	}

	/* Step 1: Halt MEC */
	halt_mec();
	pr_info("z2352: Step 1 — MEC halted: 0x%08X\n", rr(mmCP_MEC_CNTL));

	/* Step 2: Set GPA_OVERRIDE */
	wr(mmCPC_PSP_DEBUG, rr(mmCPC_PSP_DEBUG) | GPA_OVERRIDE);
	wr(mmCPG_PSP_DEBUG, rr(mmCPG_PSP_DEBUG) | GPA_OVERRIDE);
	udelay(100);
	readback = rr(mmCPC_PSP_DEBUG);
	pr_info("z2352: Step 2 — CPC_PSP_DEBUG = 0x%08X (GPA=%d)\n",
		readback, !!(readback & GPA_OVERRIDE));
	if (!(readback & GPA_OVERRIDE)) {
		pr_err("z2352: GPA_OVERRIDE write blocked — cannot proceed\n");
		unhalt_mec();
		return -EPERM;
	}

	/* Step 3: Save and read IC_BASE */
	orig_lo = rr(mmCP_CPC_IC_BASE_LO);
	orig_hi = rr(mmCP_CPC_IC_BASE_HI);
	pr_info("z2352: Step 3 — Original IC_BASE = 0x%08X:%08X\n", orig_hi, orig_lo);

	/* Step 4: Invalidate IC cache (driver does this before writing IC_BASE) */
	r = invalidate_ic();
	pr_info("z2352: Step 4 — IC invalidation: %s\n", r == 0 ? "OK" : "TIMEOUT");

	/* Step 5: Map BAR0 for VRAM access */
	bar0_start = pci_resource_start(gpu_dev, 0);
	bar0_len = pci_resource_len(gpu_dev, 0);
	if (bar0_len == 0) {
		pr_err("z2352: BAR0 not available\n");
		unhalt_mec();
		return -ENODEV;
	}
	vram = ioremap(bar0_start, min_t(u64, bar0_len, 0x200000));
	if (!vram) {
		pr_err("z2352: Failed to map BAR0\n");
		unhalt_mec();
		return -ENOMEM;
	}
	pr_info("z2352: Step 5 — BAR0 mapped at 0x%llX (%llu MB)\n",
		(u64)bar0_start, (u64)bar0_len / (1024*1024));

	/* Step 6: Copy decrypted UCODE_RAM to VRAM */
	wr(mmCP_MEC_ME1_UCODE_ADDR, 0);
	nz = 0;
	for (i = 0; i < MEC_FW_DWORDS; i++) {
		u32 v = rr(mmCP_MEC_ME1_UCODE_DATA);
		writel(v, vram + VRAM_FW_OFFSET + i * 4);
		if (v) nz++;
	}
	pr_info("z2352: Step 6 — Copied %d/%d non-zero dwords to VRAM+0x%X\n",
		nz, MEC_FW_DWORDS, VRAM_FW_OFFSET);

	/* Step 7: Inject FEEL marker */
	writel(FEEL_MARKER, vram + VRAM_FW_OFFSET + 18871 * 4);
	readback = readl(vram + VRAM_FW_OFFSET + 18871 * 4);
	pr_info("z2352: Step 7 — FEEL marker at dword 18871: 0x%08X %s\n",
		readback, readback == FEEL_MARKER ? "OK" : "FAILED");

	/* Step 8: Set IC_BASE_CNTL and write IC_BASE to VRAM copy */
	wr(mmCP_CPC_IC_BASE_CNTL, (1 << 4)); /* ADDRESS_CLAMP=1 */
	wr(mmCP_CPC_IC_BASE_LO, VRAM_FW_OFFSET & 0xFFFFF000);
	wr(mmCP_CPC_IC_BASE_HI, 0);

	readback = rr(mmCP_CPC_IC_BASE_LO);
	pr_info("z2352: Step 8 — IC_BASE_LO = 0x%08X (want 0x%08X) %s\n",
		readback, VRAM_FW_OFFSET & 0xFFFFF000,
		readback == (VRAM_FW_OFFSET & 0xFFFFF000) ? "REDIRECTED!" : "FAILED");

	if (readback != (VRAM_FW_OFFSET & 0xFFFFF000)) {
		pr_warn("z2352: IC_BASE redirect FAILED even with GPA_OVERRIDE\n");
		/* Restore */
		wr(mmCP_CPC_IC_BASE_LO, orig_lo);
		wr(mmCP_CPC_IC_BASE_HI, orig_hi);
		unhalt_mec();
		iounmap(vram);
		return -EPERM;
	}

	/* Step 9: Write jump table via UCODE_ADDR/DATA */
	pr_info("z2352: Step 9 — Writing jump table\n");
	wr(mmCP_MEC_ME1_UCODE_ADDR, 0);
	/* Read original jump table from VRAM copy and rewrite it */
	for (i = 0; i < 64; i++) {
		u32 jt_entry = readl(vram + VRAM_FW_OFFSET + i * 4);
		wr(mmCP_MEC_ME1_UCODE_DATA, jt_entry);
	}

	/* Set version in UCODE_ADDR (driver convention) */
	wr(mmCP_MEC_ME1_UCODE_ADDR, 0xFEE10004);

	/* Step 10: Invalidate IC again with new base */
	r = invalidate_ic();
	pr_info("z2352: Step 10 — IC re-invalidation: %s\n", r == 0 ? "OK" : "TIMEOUT");

	/* Step 11: Unhalt MEC */
	pr_info("z2352: Step 11 — Unhalting MEC\n");
	unhalt_mec();
	udelay(1000);

	pr_info("z2352: CP_MEC_CNTL = 0x%08X\n", rr(mmCP_MEC_CNTL));
	pr_info("z2352: GRBM_STATUS = 0x%08X\n", rr(mmGRBM_STATUS));
	pr_info("z2352: IC_BASE = 0x%08X:%08X cntl=0x%08X\n",
		rr(mmCP_CPC_IC_BASE_HI), rr(mmCP_CPC_IC_BASE_LO),
		rr(mmCP_CPC_IC_BASE_CNTL));

	iounmap(vram);
	pr_info("z2352: === GPA INJECTION v4 COMPLETE ===\n");
	return 0;
}

/*
 * klookup() — Resolve unexported kernel symbol via kprobe trick.
 * Since kallsyms_lookup_name was unexported in 5.7+, register a kprobe
 * on the symbol name, grab the address, then unregister.
 */
static unsigned long klookup(const char *name)
{
	struct kprobe kp = { .symbol_name = name };
	unsigned long addr;

	if (register_kprobe(&kp) < 0) {
		pr_warn("z2352: klookup(%s) FAILED\n", name);
		return 0;
	}
	addr = (unsigned long)kp.addr;
	unregister_kprobe(&kp);
	return addr;
}

/* Function pointer types for amdgpu internal API */
typedef u32 (*amdgpu_kiq_rreg_t)(void *adev, u32 reg, u32 xcc_id);
typedef void (*amdgpu_kiq_wreg_t)(void *adev, u32 reg, u32 val, u32 xcc_id);

/*
 * Mode 6: PM4 KIQ injection test
 *
 * Theory: PSP write-locks on CP/MEC registers may only protect the PCIe MMIO
 * path. The GPU's internal register bus (used by PM4 WRITE_DATA from ME/KIQ)
 * is a different path that may bypass PSP enforcement.
 *
 * Plan:
 *   1. Read CPC_PSP_DEBUG via MMIO (baseline, should be 0x0)
 *   2. Resolve amdgpu_kiq_wreg/rreg via kprobe
 *   3. Get amdgpu_device* from pci_get_drvdata (drm_device at offset +16)
 *   4. Read CPC_PSP_DEBUG via KIQ PM4 (COPY_DATA readback)
 *   5. Write GPA_OVERRIDE (0x8) to CPC_PSP_DEBUG via KIQ PM4 (WRITE_DATA)
 *   6. Read back via MMIO — if 0x8, PM4 bypasses PSP locks!
 *   7. If success, test IC_BASE, UCODE_ADDR/DATA writability
 */
static int do_pm4_kiq(void)
{
	amdgpu_kiq_rreg_t kiq_rreg;
	amdgpu_kiq_wreg_t kiq_wreg;
	void *drm_dev;
	void *adev;
	u32 mmio_before, mmio_after, kiq_read;
	u32 cpc_psp_dword = mmCPC_PSP_DEBUG; /* full SOC15 dword offset */

	pr_info("z2352: === MODE 6: PM4 KIQ INJECTION TEST ===\n");

	/* Step 1: Baseline MMIO reads */
	mmio_before = rr(mmCPC_PSP_DEBUG);
	pr_info("z2352: [MMIO] CPC_PSP_DEBUG = 0x%08X (baseline)\n", mmio_before);
	pr_info("z2352: [MMIO] GRBM_STATUS   = 0x%08X\n", rr(mmGRBM_STATUS));
	pr_info("z2352: [MMIO] CP_MEC_CNTL   = 0x%08X\n", rr(mmCP_MEC_CNTL));

	/* Step 2: Resolve amdgpu internal functions via kprobe */
	kiq_rreg = (amdgpu_kiq_rreg_t)klookup("amdgpu_kiq_rreg");
	kiq_wreg = (amdgpu_kiq_wreg_t)klookup("amdgpu_kiq_wreg");

	if (!kiq_rreg || !kiq_wreg) {
		pr_err("z2352: Failed to resolve KIQ symbols: rreg=%px wreg=%px\n",
		       kiq_rreg, kiq_wreg);
		return -ENOSYS;
	}
	pr_info("z2352: Resolved: kiq_rreg=%px kiq_wreg=%px\n",
		kiq_rreg, kiq_wreg);

	/* Step 3: Get amdgpu_device from PCI device
	 *
	 * pci_get_drvdata(pdev) returns &adev->ddev (struct drm_device)
	 * struct amdgpu_device layout:
	 *   offset 0:  struct device *dev;      (8 bytes)
	 *   offset 8:  struct pci_dev *pdev;    (8 bytes)
	 *   offset 16: struct drm_device ddev;  (embedded)
	 *
	 * So: adev = (char*)drm_dev - 16
	 */
	drm_dev = pci_get_drvdata(gpu_dev);
	if (!drm_dev) {
		pr_err("z2352: pci_get_drvdata returned NULL — no amdgpu driver?\n");
		return -ENODEV;
	}
	adev = (char *)drm_dev - 16;
	pr_info("z2352: drm_device=%px  adev=%px\n", drm_dev, adev);

	/* Sanity check: adev->pdev should be our gpu_dev (offset 8 in struct) */
	{
		struct pci_dev **pdev_ptr = (struct pci_dev **)((char *)adev + 8);
		if (*pdev_ptr != gpu_dev) {
			pr_err("z2352: SANITY FAIL: adev->pdev=%px != gpu_dev=%px\n",
			       *pdev_ptr, gpu_dev);
			pr_err("z2352: Struct offset assumption wrong! Aborting.\n");
			return -EINVAL;
		}
		pr_info("z2352: SANITY OK: adev->pdev matches gpu_dev\n");
	}

	/* Step 4: Read CPC_PSP_DEBUG via KIQ (PM4 COPY_DATA readback) */
	pr_info("z2352: Attempting KIQ read of CPC_PSP_DEBUG (reg=0x%06X)...\n",
		cpc_psp_dword);
	kiq_read = kiq_rreg(adev, cpc_psp_dword, 0);
	pr_info("z2352: [KIQ READ] CPC_PSP_DEBUG = 0x%08X\n", kiq_read);

	/* Step 5: Write GPA_OVERRIDE via KIQ (PM4 WRITE_DATA) */
	pr_info("z2352: Attempting KIQ write: CPC_PSP_DEBUG = 0x%08X...\n",
		GPA_OVERRIDE);
	kiq_wreg(adev, cpc_psp_dword, GPA_OVERRIDE, 0);
	udelay(100);

	/* Step 6: Read back via both paths */
	mmio_after = rr(mmCPC_PSP_DEBUG);
	pr_info("z2352: [MMIO] CPC_PSP_DEBUG = 0x%08X (after KIQ write)\n",
		mmio_after);

	kiq_read = kiq_rreg(adev, cpc_psp_dword, 0);
	pr_info("z2352: [KIQ READ] CPC_PSP_DEBUG = 0x%08X (after KIQ write)\n",
		kiq_read);

	if (mmio_after & GPA_OVERRIDE) {
		pr_info("z2352: *** PM4 WRITE BYPASSED PSP LOCK! ***\n");
		pr_info("z2352: GPA_OVERRIDE is SET — testing IC_BASE...\n");

		/* Quick test: try writing IC_BASE_LO */
		{
			u32 orig_lo = rr(mmCP_CPC_IC_BASE_LO);
			u32 test_val = 0x100000;

			pr_info("z2352: IC_BASE_LO before = 0x%08X\n", orig_lo);
			kiq_wreg(adev, mmCP_CPC_IC_BASE_LO, test_val, 0);
			udelay(100);
			pr_info("z2352: IC_BASE_LO after KIQ write = 0x%08X (want 0x%08X)\n",
				rr(mmCP_CPC_IC_BASE_LO), test_val);

			/* Restore */
			kiq_wreg(adev, mmCP_CPC_IC_BASE_LO, orig_lo, 0);
			udelay(100);
		}

		/* Test CP_MEC_CNTL halt/unhalt */
		{
			u32 orig = rr(mmCP_MEC_CNTL);
			pr_info("z2352: CP_MEC_CNTL before = 0x%08X\n", orig);
			kiq_wreg(adev, mmCP_MEC_CNTL,
				 orig | MEC_ME1_HALT | MEC_ME2_HALT, 0);
			udelay(100);
			pr_info("z2352: CP_MEC_CNTL after halt = 0x%08X\n",
				rr(mmCP_MEC_CNTL));

			/* Restore */
			kiq_wreg(adev, mmCP_MEC_CNTL, orig, 0);
			udelay(100);
			pr_info("z2352: CP_MEC_CNTL restored = 0x%08X\n",
				rr(mmCP_MEC_CNTL));
		}

		/* Clear GPA_OVERRIDE to leave clean state */
		kiq_wreg(adev, cpc_psp_dword, 0, 0);
		udelay(100);
		pr_info("z2352: CPC_PSP_DEBUG cleared = 0x%08X\n",
			rr(mmCPC_PSP_DEBUG));

		pr_info("z2352: === PM4 KIQ INJECTION: SUCCESS ===\n");
		return 0;
	}

	/* PM4 write didn't take effect via MMIO readback */
	if (kiq_read & GPA_OVERRIDE) {
		pr_info("z2352: KIQ read shows 0x8 but MMIO shows 0x0\n");
		pr_info("z2352: PSP may shadow register — different read/write paths\n");
	} else {
		pr_info("z2352: PM4 write BLOCKED — PSP enforces at register level, not bus level\n");
	}

	/* Also try writing CP_MEC_CNTL via PM4 as control test */
	{
		u32 orig = rr(mmCP_MEC_CNTL);
		pr_info("z2352: [CONTROL] Trying CP_MEC_CNTL via KIQ...\n");
		pr_info("z2352: CP_MEC_CNTL before = 0x%08X\n", orig);
		kiq_wreg(adev, mmCP_MEC_CNTL, orig | MEC_ME1_HALT, 0);
		udelay(100);
		mmio_after = rr(mmCP_MEC_CNTL);
		pr_info("z2352: CP_MEC_CNTL after = 0x%08X %s\n", mmio_after,
			(mmio_after & MEC_ME1_HALT) ? "HALT SET!" : "NO CHANGE");

		if (mmio_after & MEC_ME1_HALT) {
			pr_info("z2352: CP_MEC_CNTL writable via PM4 — partial bypass!\n");
			/* Unhalt */
			kiq_wreg(adev, mmCP_MEC_CNTL, orig, 0);
			udelay(100);
		}
	}

	pr_info("z2352: === PM4 KIQ INJECTION: FINISHED ===\n");
	return 0;
}

static int __init hotpatch_init(void)
{
	struct pci_dev *pdev = NULL;

	pr_info("z2352: === MEC hotpatch v5 (mode=%d) ===\n", mode);

	while ((pdev = pci_get_device(AMD_VENDOR_ID, PCI_ANY_ID, pdev))) {
		if ((pdev->class >> 8) == PCI_CLASS_DISPLAY_VGA ||
		    (pdev->class >> 8) == PCI_CLASS_DISPLAY_OTHER) {
			gpu_dev = pdev;
			break;
		}
	}
	if (!gpu_dev) { pr_err("z2352: No GPU\n"); return -ENODEV; }

	pr_info("z2352: GPU %04x:%04x at %s\n",
		gpu_dev->vendor, gpu_dev->device, pci_name(gpu_dev));

	bar5_size = pci_resource_len(gpu_dev, 5);
	mmio = pci_iomap(gpu_dev, 5, 0);
	if (!mmio) {
		pci_dev_put(gpu_dev);
		return -ENOMEM;
	}
	pr_info("z2352: BAR5 mapped (%llu KB), indirect for dwords >= 0x%llX\n",
		(u64)bar5_size / 1024, (u64)bar5_size / 4);

	switch (mode) {
	case 0: do_probe(); break;
	case 1: do_test_writes(); break;
	case 2: do_inject(); break;
	case 3: do_verify(); break;
	case 4: do_gpa_test(); break;
	case 5: do_gpa_inject(); break;
	case 6: do_pm4_kiq(); break;
	default: pr_err("z2352: bad mode\n");
	}

	return 0;
}

static void __exit hotpatch_exit(void)
{
	if (mmio) pci_iounmap(gpu_dev, mmio);
	if (gpu_dev) pci_dev_put(gpu_dev);
	pr_info("z2352: unloaded\n");
}

module_init(hotpatch_init);
module_exit(hotpatch_exit);
