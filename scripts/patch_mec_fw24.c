/*
 * patch_mec_fw24.c — Phase 24: GRBM pipe select + SRAM readback
 *
 * Phase 23 showed SRAM reads return all zeros. This is likely because
 * GRBM_GFX_INDEX must be set to select MEC1/pipe0 before accessing
 * per-pipe registers like UCODE_ADDR/DATA.
 *
 * The driver does:
 *   gfx_v11_0_select_me_pipe_q(adev, 1, 0, 0) before MEC register access
 * which writes GRBM_GFX_INDEX with ME=1, PIPE=0, QUEUE=0, INSTANCE_BROADCAST=0
 *
 * Also try:
 *   - Different SRAM read register offsets (auto-increment mode)
 *   - Reading CP_MEC_CNTL with pipe selected
 *   - Checking if IC_BASE changes with pipe selection
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

/* GRBM_GFX_INDEX — engine/pipe/queue selector */
#define regGRBM_GFX_INDEX          0x0D884

/* GRBM_GFX_INDEX bit layout for GFX11:
 *   [3:0]   SA_INDEX
 *   [7:4]   reserved
 *   [8]     SA_BROADCAST_WRITES
 *   [12:9]  INSTANCE_INDEX (SE)
 *   [15:13] reserved
 *   [16]    INSTANCE_BROADCAST_WRITES (SE broadcast)
 *   [20:17] reserved
 *   [21]    reserved
 *   [25:22] reserved
 *   [30:26] reserved
 *   [31]    reserved
 *
 * For ME/PIPE/QUEUE selection, gfx_v11_0_select_me_pipe_q uses:
 *   regGRBM_GFX_CNTL or similar.
 * Let me check the actual register used...
 */

/* Actually, for compute pipe/queue selection GFX11 uses: */
#define regGRBM_GFX_CNTL           0x0D880

/* CP_MEC_ME1_HEADER_DUMP — contains instruction data after UCODE load */
/* Alternative read method: some GFX gens expose loaded ucode via header dump */

/* MEC registers */
#define regCP_MEC1_INSTR_PNTR      0x021A8
#define regCP_MEC_CNTL             0x0A802
#define MEC_ME1_HALT               (1 << 30)
#define MEC_ME2_HALT               (1 << 28)

/* SRAM access */
#define regCP_MEC_ME1_UCODE_ADDR   0x0A814
#define regCP_MEC_ME1_UCODE_DATA   0x0A815

/* IC_BASE */
#define regCP_CPC_IC_BASE_LO       0x0C930
#define regCP_CPC_IC_BASE_HI       0x0C931
#define regCP_CPC_IC_OP_CNTL       0x0C932

/* GFX11 specific: try RLCG interface for privileged register access.
 * In GFX11, some registers need RLCG (RLC Gateway) indirect access.
 * The driver uses amdgpu_device_wreg() which may route through RLCG. */

/* RLCG registers */
#define regRLC_RLCG_DOORBELL_CNTL  0x4C8E
#define regSCRATCH_REG0            0x0D840
#define regSCRATCH_REG1            0x0D841

/* GFX11 CP_MEC specific */
#define regCP_HQD_ACTIVE           0x0C91C
#define regCP_HQD_PQ_CONTROL       0x0C923

/* Try different UCODE read registers that might work on GFX11 */
#define regCP_MEC_MDBASE_LO        0x0C924
#define regCP_MEC_MDBASE_HI        0x0C925

static void try_sram_read(const char *label)
{
	int i;
	u32 val;

	pr_info("fw24: SRAM read [%s]:\n", label);
	for (i = 0; i < 8; i++) {
		wr(regCP_MEC_ME1_UCODE_ADDR, i);
		udelay(10);
		val = rr(regCP_MEC_ME1_UCODE_DATA);
		if (i < 4 || val != 0)
			pr_info("fw24:   [0x%03X] = 0x%08X%s\n", i, val,
				val ? " NON-ZERO!" : "");
	}

	/* Also try auto-increment: write addr=0 then read DATA multiple times */
	pr_info("fw24: SRAM auto-inc [%s]:\n", label);
	wr(regCP_MEC_ME1_UCODE_ADDR, 0);
	udelay(10);
	for (i = 0; i < 4; i++) {
		val = rr(regCP_MEC_ME1_UCODE_DATA);
		if (i < 2 || val != 0)
			pr_info("fw24:   auto[%d] = 0x%08X%s\n", i, val,
				val ? " NON-ZERO!" : "");
	}

	/* Read around stuck PC */
	wr(regCP_MEC_ME1_UCODE_ADDR, 0x44C);
	udelay(10);
	val = rr(regCP_MEC_ME1_UCODE_DATA);
	pr_info("fw24:   [0x44C] = 0x%08X%s\n", val,
		(val == 0x88000000UL) ? " BRANCH_SELF!" : "");
}

static void dump_ic_base(const char *label)
{
	pr_info("fw24: IC_BASE [%s]: LO=0x%08X HI=0x%08X OP_CNTL=0x%08X\n",
		label,
		rr(regCP_CPC_IC_BASE_LO),
		rr(regCP_CPC_IC_BASE_HI),
		rr(regCP_CPC_IC_OP_CNTL));
}

static int __init fw24_init(void)
{
	struct pci_dev *pdev = NULL;
	u32 grbm_save;
	u32 pc;
	int me, pipe;

	pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
	if (!pdev)
		return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	pr_info("fw24: ========================================\n");
	pr_info("fw24: PHASE 24: GRBM PIPE SELECT + SRAM READ\n");
	pr_info("fw24: ========================================\n");

	pc = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("fw24: PC=0x%04X MEC_CNTL=0x%08X\n", pc, rr(regCP_MEC_CNTL));

	/* Save GRBM state */
	grbm_save = rr(regGRBM_GFX_INDEX);
	pr_info("fw24: GRBM_GFX_INDEX initial = 0x%08X\n", grbm_save);
	pr_info("fw24: GRBM_GFX_CNTL initial  = 0x%08X\n", rr(regGRBM_GFX_CNTL));

	/* Halt MEC first */
	pr_info("fw24: Halting MEC...\n");
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_ME2_HALT);
	udelay(200);
	pr_info("fw24: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));

	/* Try without pipe select first (baseline) */
	pr_info("fw24: --- No pipe select (baseline) ---\n");
	try_sram_read("no-select");
	dump_ic_base("no-select");

	/* Try selecting MEC1 pipe0 via GRBM_GFX_CNTL
	 *
	 * gfx_v11_0_select_me_pipe_q writes to regGRBM_GFX_CNTL:
	 *   val = (me << ME_SHIFT) | (pipe << PIPE_SHIFT) | (queue << QUEUE_SHIFT)
	 * For GFX11: ME=1 for MEC1, PIPE=0
	 *
	 * GRBM_GFX_CNTL bit layout:
	 *   [3:0]  PIPEID
	 *   [5:4]  MEID (0=GFX, 1=MEC1, 2=MEC2)
	 *   [10:8] QUEUEID
	 *   [12]   VMID broadcast (unused for this)
	 */
	for (me = 0; me <= 2; me++) {
		for (pipe = 0; pipe <= 1; pipe++) {
			u32 sel = (me << 4) | pipe;
			pr_info("fw24: --- GRBM_GFX_CNTL = ME=%d PIPE=%d (0x%02X) ---\n",
				me, pipe, sel);
			wr(regGRBM_GFX_CNTL, sel);
			udelay(100);

			try_sram_read("selected");
			dump_ic_base("selected");

			/* Also read HQD_ACTIVE to see which queues are active */
			pr_info("fw24: HQD_ACTIVE=0x%08X HQD_PQ_CTRL=0x%08X\n",
				rr(regCP_HQD_ACTIVE), rr(regCP_HQD_PQ_CONTROL));
		}
	}

	/* Restore GRBM and try GRBM_GFX_INDEX method */
	wr(regGRBM_GFX_CNTL, 0);
	udelay(100);

	/* Try GRBM_GFX_INDEX with instance broadcast disabled */
	pr_info("fw24: --- GRBM_GFX_INDEX methods ---\n");
	{
		u32 idx_vals[] = {
			0x00000000,  /* everything 0 */
			0x00010000,  /* SE_BROADCAST_WRITES=1 */
			0x00000100,  /* SA_BROADCAST_WRITES=1 */
			0x00010100,  /* both broadcasts */
		};
		int v;
		for (v = 0; v < 4; v++) {
			wr(regGRBM_GFX_INDEX, idx_vals[v]);
			udelay(100);
			pr_info("fw24: GRBM_GFX_INDEX=0x%08X:\n", idx_vals[v]);
			wr(regCP_MEC_ME1_UCODE_ADDR, 0);
			udelay(10);
			pr_info("fw24:   SRAM[0]=0x%08X SRAM[1]=0x%08X\n",
				rr(regCP_MEC_ME1_UCODE_DATA),
				rr(regCP_MEC_ME1_UCODE_DATA));
		}
	}

	/* Restore GRBM_GFX_INDEX */
	wr(regGRBM_GFX_INDEX, grbm_save);

	/* Try a brute-force scan of nearby registers for non-zero SRAM data.
	 * In GFX11, UCODE registers might be at different offsets. */
	pr_info("fw24: --- Register scan 0xA810-0xA830 ---\n");
	{
		u32 off;
		for (off = 0xA810; off <= 0xA830; off++) {
			u32 val = rr(off);
			if (val != 0)
				pr_info("fw24: reg[0x%05X] = 0x%08X NON-ZERO\n",
					off, val);
		}
	}

	/* Also scan around the IC_BASE area */
	pr_info("fw24: --- Register scan 0xC920-0xC940 ---\n");
	{
		u32 off;
		for (off = 0xC920; off <= 0xC940; off++) {
			u32 val = rr(off);
			if (val != 0)
				pr_info("fw24: reg[0x%05X] = 0x%08X\n", off, val);
		}
	}

	/* Scan for SRAM-like read ports: try many register ranges */
	pr_info("fw24: --- Wide register scan for non-zero near CP ---\n");
	{
		/* CP registers are 0xA800-0xAFFF and 0xC900-0xCFFF */
		u32 ranges[][2] = {
			{0xA800, 0xA820},
			{0xA830, 0xA850},
			{0xC930, 0xC940},
		};
		int r;
		for (r = 0; r < 3; r++) {
			u32 off;
			for (off = ranges[r][0]; off <= ranges[r][1]; off++) {
				u32 val = rr(off);
				if (val != 0)
					pr_info("fw24: reg[0x%05X] = 0x%08X\n",
						off, val);
			}
		}
	}

	/* Unhalt MEC */
	pr_info("fw24: Unhalting MEC...\n");
	wr(regGRBM_GFX_CNTL, 0);
	wr(regCP_MEC_CNTL, 0);
	udelay(200);
	pr_info("fw24: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));

	pr_info("fw24: ========================================\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw24_exit(void) {}

module_init(fw24_init);
module_exit(fw24_exit);
