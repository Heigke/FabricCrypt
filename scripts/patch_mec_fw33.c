/*
 * patch_mec_fw33.c — Phase 33: SPI ROM + Debug Interface + Safe TOCTOU
 *
 * Phase 32 results:
 *   - TOCTOU redirect: PSP loaded MEC from TMR cache (PC=0x04A7 unchanged)
 *     but VCN load failed (0xF) causing GPU reset failure
 *   - Ring doorbell: RPTR never advances (no doorbell mechanism found)
 *   - Mailbox: C2PMSG_69 IS writable but bootloader doesn't process new cmds
 *   - Security: C2PMSG_91=0x0E03003F — debug_en bit appears SET
 *
 * This phase:
 *   A. SPI ROM probe: Use psp_v13_0_exec_spi_cmd() path to read SPI flash
 *      (contains PSP firmware, keys, fuse mirror, VBIOS)
 *   B. Debug interface: Probe debug-related PSP registers and commands
 *   C. Safe TOCTOU: Only modify MEC1 (fw_type=0x31) cmd, leave all others
 *      intact. Try mode=4 (zero_size) which won't break other loads.
 *   D. PSP autoload: Try AUTOLOAD_RLC (cmd_id=14) which bulk-loads
 *      multiple firmware — maybe we can slip in a modified TOC
 *   E. Kfunc: Try calling the driver's own PSP functions directly
 *      via function pointer lookup
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/kprobes.h>
/* kallsyms_lookup_name not exported since 5.7, use kprobe trick */
static unsigned long klookup(const char *name)
{
	struct kprobe kp = { .symbol_name = name };
	unsigned long addr;
	if (register_kprobe(&kp) < 0) return 0;
	addr = (unsigned long)kp.addr;
	unregister_kprobe(&kp);
	return addr;
}

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002
#define AMD_DEV_ID    0x1586

static void __iomem *mmio;
static struct pci_dev *g_pdev;

static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }
static void wr(u32 dw_off, u32 val) {
	writel(val, mmio + (u64)dw_off * 4);
	readl(mmio + (u64)dw_off * 4);
}

/* Registers */
#define regCP_MEC1_INSTR_PNTR      0x021A8
#define regCP_MEC_CNTL             0x0A802
#define regCP_MEC_ME1_UCODE_ADDR   0x0A814
#define regCP_MEC_ME1_UCODE_DATA   0x0A815

/* MP0 C2PMSG */
#define C2PMSG_33   0x16061
#define C2PMSG_35   0x16063
#define C2PMSG_36   0x16064
#define C2PMSG_69   0x16085
#define C2PMSG_81   0x16091

/* PSP ring */
#define PSP_RING_RPTR  0x16083
#define PSP_RING_WPTR  0x16084

/* PSP SPI registers (from amdgpu driver psp_v13_0.c) */
/* These are MP0 private registers accessed via SMN or direct MMIO */
/* regMP0_SMN_C2PMSG_101..103 used for SPI commands */
#define C2PMSG_101  0x160A5  /* SPI cmd register? */
#define C2PMSG_102  0x160A6
#define C2PMSG_103  0x160A7

/* Attack mode */
static int attack_mode = 0;
module_param(attack_mode, int, 0444);
MODULE_PARM_DESC(attack_mode,
	"0=spi_probe, 1=debug_probe, 2=safe_toctou, 3=call_psp_fn");

/* Kprobe intercept count */
static int cmd_count = 0;
static int mec_count = 0;

/*
 * Strategy A: SPI ROM Probe
 *
 * AMD GPUs have SPI flash containing VBIOS, PSP firmware, and keys.
 * The PSP accesses it via SPI controller. The driver has psp_v13_0_exec_spi_cmd()
 * which uses C2PMSG registers to send SPI commands to PSP, which then accesses SPI.
 *
 * Protocol (from psp_v13_0_exec_spi_cmd):
 *   1. Write SPI cmd id to regMP0_SMN_C2PMSG_101
 *   2. Write offset to regMP0_SMN_C2PMSG_102
 *   3. Write data to regMP0_SMN_C2PMSG_103
 *   4. Write trigger to regMP0_SMN_C2PMSG_101 (set exec bit)
 *   5. Poll for completion
 *
 * SPI CMD IDs (from amdgpu):
 *   Read = 0, Write = 1, Erase = 2
 */
static void try_spi_read(void)
{
	u32 val;
	int i;

	pr_info("fw33: === SPI ROM PROBE ===\n");

	/* First check what's in the SPI-related C2PMSG registers */
	pr_info("fw33: SPI registers pre-state:\n");
	for (i = 0; i < 16; i++) {
		val = rr(0x160A0 + i);
		if (val)
			pr_info("fw33:   C2PMSG[%d] (0x%05X) = 0x%08X\n",
				0xA0 + i, 0x160A0 + i, val);
	}

	/* The actual SPI interface in psp_v13_0.c uses:
	 * - regMP0_SMN_C2PMSG_101 for command
	 * - regMP0_SMN_C2PMSG_102 for address/offset
	 * - regMP0_SMN_C2PMSG_103 for data/result
	 *
	 * But these register offsets depend on the exact IP version.
	 * For v13.0.4, they might be different. Let's try the approach
	 * from the driver: psp_v13_0_exec_spi_cmd/psp_v13_0_update_spirom
	 *
	 * The driver accesses these as SOC15 registers:
	 *   RREG32_SOC15(MP0, 0, regMP0_SMN_C2PMSG_XXX)
	 * which adds the IP base offset.
	 */

	/* Try reading SPI via a different approach:
	 * PSP exposes SPI ROM content through specific PSP mailbox commands.
	 * The psp_v13_0_read_spirom function exists but might not be exported.
	 * Let's probe for it. */

	/* Scan extended MP0 register space for SPI controller */
	pr_info("fw33: Extended MP0 scan (0x160A0-0x160C0):\n");
	for (i = 0; i < 32; i++) {
		val = rr(0x160A0 + i);
		if (val)
			pr_info("fw33:   [0x%05X] = 0x%08X\n",
				0x160A0 + i, val);
	}

	/* Try direct SPI read command */
	pr_info("fw33: Attempting SPI read (cmd=0, offset=0)...\n");

	/* Write offset 0 */
	wr(0x160A6, 0x00000000);  /* offset = 0 (start of SPI) */
	udelay(10);

	/* Write read command (0) with exec bit */
	wr(0x160A5, 0x00000001);  /* cmd=read, exec=1 */
	mdelay(10);

	/* Read result */
	val = rr(0x160A7);
	pr_info("fw33: SPI read result: 0x%08X\n", val);

	/* Check status */
	val = rr(0x160A5);
	pr_info("fw33: SPI status: 0x%08X\n", val);

	/* Try alternate SPI access via SMN (System Management Network)
	 * PSP SPI controller might be at SMN address 0x03B10000+
	 * We can try via direct MMIO if it's mapped */
}

/*
 * Strategy B: Debug Interface Probe
 *
 * C2PMSG_91 = 0x0E03003F has debug_en=1 (bit 0).
 * This MIGHT mean PSP debug mode is active, which could allow:
 * - Unsigned firmware loading
 * - Direct SRAM access
 * - Debug register access
 *
 * Also probe for:
 * - PSP TMR fence registers (might reveal TMR layout)
 * - PSP security fuse state
 * - VBIOS scratch registers
 */
static void try_debug_probe(void)
{
	u32 val;
	int i;

	pr_info("fw33: === DEBUG INTERFACE PROBE ===\n");

	/* C2PMSG_91 detailed analysis */
	val = rr(0x1609B);
	pr_info("fw33: C2PMSG_91 = 0x%08X\n", val);
	pr_info("fw33:   [0]  debug_en     = %d\n", (val >> 0) & 1);
	pr_info("fw33:   [1]  fuse_ready   = %d\n", (val >> 1) & 1);
	pr_info("fw33:   [2]  bit2         = %d\n", (val >> 2) & 1);
	pr_info("fw33:   [3]  bit3         = %d\n", (val >> 3) & 1);
	pr_info("fw33:   [4]  bit4         = %d\n", (val >> 4) & 1);
	pr_info("fw33:   [5]  bit5         = %d\n", (val >> 5) & 1);
	pr_info("fw33:   [8:15] sec_level  = 0x%02X\n", (val >> 8) & 0xFF);
	pr_info("fw33:   [16:23]           = 0x%02X\n", (val >> 16) & 0xFF);
	pr_info("fw33:   [24:31]           = 0x%02X\n", (val >> 24) & 0xFF);

	/* PSP debug unlock sequence attempt:
	 * Some PSP versions have a debug unlock via writing specific values
	 * to C2PMSG registers. This is chip-specific and usually fuse-locked.
	 */

	/* Check PSP version registers */
	pr_info("fw33: PSP version info:\n");
	pr_info("fw33:   C2PMSG_80 (0x16090) = 0x%08X\n", rr(0x16090));
	pr_info("fw33:   C2PMSG_81 (0x16091) = 0x%08X\n", rr(0x16091));

	/* Scan for VBIOS scratch registers (typically 0xDxxx range) */
	pr_info("fw33: VBIOS scratch regs:\n");
	for (i = 0; i < 8; i++) {
		val = rr(0x0D440 + i);
		if (val)
			pr_info("fw33:   BIOS_SCRATCH_%d = 0x%08X\n", i, val);
	}

	/* Security fuse registers (typically in FUSE block) */
	pr_info("fw33: Fuse block scan:\n");
	for (i = 0; i < 16; i++) {
		val = rr(0x16C00 + i);
		if (val)
			pr_info("fw33:   FUSE[%d] (0x%05X) = 0x%08X\n",
				i, 0x16C00 + i, val);
	}

	/* Additional fuse ranges */
	for (i = 0; i < 8; i++) {
		val = rr(0x17000 + i);
		if (val)
			pr_info("fw33:   FUSE2[%d] (0x%05X) = 0x%08X\n",
				i, 0x17000 + i, val);
	}

	/* Try writing to PSP debug register */
	pr_info("fw33: Attempting PSP debug write to C2PMSG_91...\n");
	{
		u32 old = rr(0x1609B);
		/* Try setting all bits (unlock) */
		wr(0x1609B, 0xFFFFFFFF);
		udelay(10);
		val = rr(0x1609B);
		pr_info("fw33: After write 0xFFFFFFFF: C2PMSG_91 = 0x%08X %s\n",
			val, (val != old) ? "CHANGED!" : "no change");
		/* Restore */
		wr(0x1609B, old);
	}

	/* PSP TMR (Trusted Memory Region) detailed probe */
	pr_info("fw33: TMR details:\n");
	{
		u64 tmr_base;
		u32 tmr_lo = rr(0x16086);
		u32 tmr_hi = rr(0x16087);
		tmr_base = ((u64)tmr_lo << 32) | ((u64)tmr_hi << 0);
		pr_info("fw33:   TMR regs: [0x16086]=0x%08X [0x16087]=0x%08X\n",
			tmr_lo, tmr_hi);

		/* Scan wider TMR register area */
		for (i = 0; i < 8; i++) {
			val = rr(0x16086 + i);
			pr_info("fw33:   TMR[%d] (0x%05X) = 0x%08X\n",
				i, 0x16086 + i, val);
		}
	}
}

/*
 * Strategy C: Safe TOCTOU kprobe
 * Only intercept MEC1 (fw_type=0x31), leave everything else alone.
 * Mode 2: set size=0 (safest — PSP may just skip the load)
 */
static int safe_cmd_pre(struct kprobe *p, struct pt_regs *regs)
{
	u32 *cmd = (u32 *)regs->dx;
	u32 cmd_id, fw_type;

	if (!cmd) return 0;
	cmd_count++;

	cmd_id = cmd[2];
	if (cmd_id != 6) return 0;  /* Only LOAD_IP_FW */

	fw_type = cmd[10];
	pr_info("fw33: >>> LOAD_IP_FW #%d: type=0x%04X addr=%08X:%08X size=0x%X\n",
		cmd_count, fw_type, cmd[8], cmd[7], cmd[9]);

	if (fw_type != 0x31) return 0;  /* Only MEC1 */
	mec_count++;

	if (attack_mode != 2) {
		pr_info("fw33: MEC1 observed (mode=%d, no modification)\n",
			attack_mode);
		return 0;
	}

	pr_info("fw33: *** SAFE TOCTOU: Setting MEC1 size to 0 ***\n");
	pr_info("fw33: Original size=0x%X\n", cmd[9]);
	cmd[9] = 0;
	pr_info("fw33: Size now=0\n");

	return 0;
}

static void safe_cmd_post(struct kprobe *p, struct pt_regs *regs,
			   unsigned long flags)
{
	if (mec_count == 0) return;

	/* Check MEC state after PSP processes the command */
	pr_info("fw33: POST: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));

	/* Try SRAM write during post-load window */
	{
		u32 test_addr = 0x000;
		u32 old_val, new_val;

		wr(regCP_MEC_ME1_UCODE_ADDR, test_addr);
		udelay(10);
		old_val = rr(regCP_MEC_ME1_UCODE_DATA);

		/* Try write */
		wr(regCP_MEC_ME1_UCODE_ADDR, test_addr);
		udelay(5);
		wr(regCP_MEC_ME1_UCODE_DATA, 0xBF800000);  /* NOP */
		udelay(10);

		/* Readback */
		wr(regCP_MEC_ME1_UCODE_ADDR, test_addr);
		udelay(10);
		new_val = rr(regCP_MEC_ME1_UCODE_DATA);

		pr_info("fw33: POST SRAM write attempt: old=0x%08X new=0x%08X %s\n",
			old_val, new_val,
			(new_val == 0xBF800000) ? "SRAM WRITE WORKS!!!" :
			(new_val != old_val) ? "changed!" : "no change");
	}
}

static struct kprobe kp_cmd = {
	.symbol_name = "psp_cmd_submit_buf",
};

/*
 * Strategy D: Try to find and call PSP functions directly
 * Use kallsyms_lookup_name to find driver functions
 */
static void try_psp_function_calls(void)
{
	unsigned long addr;

	pr_info("fw33: === PSP FUNCTION LOOKUP ===\n");

	/* Look up key PSP functions */
	addr = klookup("psp_v13_0_exec_spi_cmd");
	pr_info("fw33: psp_v13_0_exec_spi_cmd = 0x%lx\n", addr);

	addr = klookup("psp_v13_0_update_spirom");
	pr_info("fw33: psp_v13_0_update_spirom = 0x%lx\n", addr);

	addr = klookup("psp_ring_cmd_submit");
	pr_info("fw33: psp_ring_cmd_submit = 0x%lx\n", addr);

	addr = klookup("psp_cmd_submit_buf");
	pr_info("fw33: psp_cmd_submit_buf = 0x%lx\n", addr);

	/* Look for debug-specific functions */
	addr = klookup("psp_v13_0_4_bootloader_load_dbg_drv");
	pr_info("fw33: psp_v13_0_4_bootloader_load_dbg_drv = 0x%lx\n", addr);

	addr = klookup("psp_v13_0_mode1_reset");
	pr_info("fw33: psp_v13_0_mode1_reset = 0x%lx\n", addr);

	/* Look for PSP context pointer */
	addr = klookup("amdgpu_device_ip_get_ip_block");
	pr_info("fw33: amdgpu_device_ip_get_ip_block = 0x%lx\n", addr);

	/* Look for direct SRAM-related functions */
	addr = klookup("gfx_v11_0_config_mec_cache");
	pr_info("fw33: gfx_v11_0_config_mec_cache = 0x%lx\n", addr);

	addr = klookup("gfx_v11_0_cp_compute_enable");
	pr_info("fw33: gfx_v11_0_cp_compute_enable = 0x%lx\n", addr);

	/* MES (MicroEngine Scheduler) functions — MES manages MEC */
	addr = klookup("amdgpu_mes_init");
	pr_info("fw33: amdgpu_mes_init = 0x%lx\n", addr);

	addr = klookup("amdgpu_mes_add_hw_queue");
	pr_info("fw33: amdgpu_mes_add_hw_queue = 0x%lx\n", addr);
}

/*
 * Strategy E: PSP ring with proper wptr update
 * The driver's psp_v13_0_ring_set_wptr does:
 *   1. Read current RPTR
 *   2. Write new WPTR to C2PMSG_67 (which is at offset 0x16083)
 *
 * But maybe we need to also write to the ring's doorbell page.
 * Let's check if there's a doorbell BAR mapping.
 */
static void try_psp_ring_v2(void)
{
	u32 rptr, wptr;
	void *ring_va;
	struct page *ring_page;
	phys_addr_t ring_phys = 0x116F9F000ULL;
	u32 *ring;
	int i;

	pr_info("fw33: === PSP RING v2 ===\n");

	rptr = rr(PSP_RING_RPTR);
	wptr = rr(PSP_RING_WPTR);
	pr_info("fw33: Ring: RPTR=0x%04X WPTR=0x%04X\n", rptr, wptr);

	/* Map ring */
	ring_page = pfn_to_page(ring_phys >> PAGE_SHIFT);
	if (!ring_page) {
		pr_info("fw33: Cannot get ring page\n");
		return;
	}
	ring_va = page_address(ring_page);
	if (!ring_va) {
		pr_info("fw33: Cannot get ring page_address\n");
		return;
	}
	ring = (u32 *)((u8 *)ring_va + (ring_phys & ~PAGE_MASK));

	/* Dump current ring content around RPTR/WPTR */
	pr_info("fw33: Ring content at WPTR (0x%X):\n", wptr);
	{
		u32 off = wptr / 4;
		for (i = -4; i < 16; i++) {
			pr_info("fw33:   ring[WPTR%+d] = 0x%08X\n",
				i, ring[off + i]);
		}
	}

	/* Build a minimal PSP ring command frame
	 * PSP ring format (from psp_ring_cmd_submit):
	 *   ring[wptr++] = psp->cmd_buf_mc_addr;  // GPU addr of cmd buffer
	 *   ring[wptr++] = psp->cmd_buf_mc_addr >> 32;
	 *   ring[wptr++] = psp->fence_buf_mc_addr;  // GPU addr of fence
	 *   ring[wptr++] = psp->fence_buf_mc_addr >> 32;
	 *
	 * So each ring entry is 4 dwords (16 bytes).
	 * The PSP reads the cmd buffer from the GPU address.
	 */

	pr_info("fw33: Ring protocol: each entry = 4 dwords (cmd_addr_lo, cmd_addr_hi, fence_lo, fence_hi)\n");

	/* Read previous ring entries to understand format */
	pr_info("fw33: Previous ring entries:\n");
	for (i = 0; i < rptr / 4; i += 4) {
		u64 cmd_addr = ((u64)ring[i+1] << 32) | ring[i];
		u64 fence_addr = ((u64)ring[i+3] << 32) | ring[i+2];
		if (cmd_addr || fence_addr) {
			pr_info("fw33:   [%03X] cmd=0x%llX fence=0x%llX\n",
				i * 4, cmd_addr, fence_addr);
		}
	}
}

static int __init fw33_init(void)
{
	int ret;

	g_pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
	if (!g_pdev)
		return -ENODEV;

	mmio = pci_iomap(g_pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(g_pdev);
		return -ENODEV;
	}

	pr_info("fw33: ====================================================\n");
	pr_info("fw33: PHASE 33: SPI ROM + DEBUG + SAFE TOCTOU\n");
	pr_info("fw33: Attack mode: %d\n", attack_mode);
	pr_info("fw33: ====================================================\n");

	/* Initial state */
	pr_info("fw33: MEC: PC=0x%04X CNTL=0x%08X\n",
		rr(regCP_MEC1_INSTR_PNTR), rr(regCP_MEC_CNTL));

	/* Strategy A: SPI ROM */
	try_spi_read();

	/* Strategy B: Debug interface */
	try_debug_probe();

	/* Strategy D: Function lookup */
	try_psp_function_calls();

	/* Strategy E: Ring v2 */
	try_psp_ring_v2();

	/* Arm kprobe (Strategy C) */
	kp_cmd.pre_handler = safe_cmd_pre;
	kp_cmd.post_handler = safe_cmd_post;
	ret = register_kprobe(&kp_cmd);
	if (ret < 0) {
		pr_info("fw33: FAIL: kprobe: %d\n", ret);
	} else {
		pr_info("fw33: Kprobe armed on psp_cmd_submit_buf\n");
	}

	pr_info("fw33: ====================================================\n");
	pr_info("fw33: Mode %d active. Trigger GPU reset to test.\n",
		attack_mode);
	pr_info("fw33: Modes: 0=probe_only, 2=safe_toctou(zero_size)\n");
	pr_info("fw33: ====================================================\n");

	return 0;
}

static void __exit fw33_exit(void)
{
	unregister_kprobe(&kp_cmd);

	if (mmio)
		pci_iounmap(g_pdev, mmio);

	pci_dev_put(g_pdev);

	pr_info("fw33: Unloaded. %d cmds, %d MEC intercepts\n",
		cmd_count, mec_count);
}

module_init(fw33_init);
module_exit(fw33_exit);
