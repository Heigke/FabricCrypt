/*
 * patch_mec_fw30.c — Phase 30: PSP Attack Surface Mapping
 *
 * Findings so far:
 *   - MEC firmware on disk is SIGNED but NOT ENCRYPTED (RS64 plaintext at 0x200+)
 *   - PSP v13.0.4 IP block handles all firmware loading
 *   - PSP has debug driver loading path (psp_v13_0_4_bootloader_load_dbg_drv)
 *   - TMR at 0x97e0000000, 142MB reserved in VRAM
 *   - TEE + TSME enabled
 *
 * This phase:
 *   1. Probe MP0 C2PMSG mailbox registers (PSP status, version, debug mode)
 *   2. Kprobe psp_load_non_psp_fw to intercept MEC firmware DMA buffer
 *   3. Kprobe psp_ring_cmd_submit to see PSP command format
 *   4. Kprobe psp_cmd_submit_buf to find command buffer structure
 *   5. Read PSP ring buffer wptr/rptr to understand ring state
 *   6. Check for PSP secure debug unlock status
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/kprobes.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002
#define AMD_DEV_ID    0x1586

static void __iomem *mmio;

static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }

/*
 * MP0 C2PMSG registers — PSP mailbox
 * These are at SOC15 MP0 block offsets.
 * On GFX11, MP0 base in SOC15 varies. Common range: 0x16000-0x16100
 * We'll try multiple known base offsets.
 *
 * C2PMSG register layout (relative to MP0 base):
 *   C2PMSG_35 = base + 0x0063  (PSP response)
 *   C2PMSG_36 = base + 0x0064  (PSP status)
 *   C2PMSG_58 = base + 0x007A  (PSP version)
 *   C2PMSG_64 = base + 0x0080  (PSP command)
 *   C2PMSG_69 = base + 0x0085  (fence)
 *   C2PMSG_81 = base + 0x0091  (bootloader status)
 */

/* Try to find MP0 base by scanning for non-zero C2PMSG reads */
static u32 mp0_base = 0;

/* Known MP0 base candidates for various AMD GPUs */
static const u32 mp0_bases[] = {
	0x16000, /* Common for many AMD GPUs */
	0x16100, /* Alternative */
	0x3B100, /* Newer chips? */
	0x00380, /* Some GFX11 */
	0x00080, /* Direct low offsets */
};

/* PSP ring buffer registers (relative to MP0 base) */
#define MP0_C2PMSG_35   0x0063  /* Response */
#define MP0_C2PMSG_36   0x0064  /* Status */
#define MP0_C2PMSG_51   0x0073  /* Ring RPTR */
#define MP0_C2PMSG_52   0x0074  /* Ring WPTR */
#define MP0_C2PMSG_53   0x0075  /* Ring fence */
#define MP0_C2PMSG_58   0x007A  /* Version? */
#define MP0_C2PMSG_64   0x0080  /* Command */
#define MP0_C2PMSG_69   0x0085  /* Fence */
#define MP0_C2PMSG_81   0x0091  /* Bootloader status */

/* Probe counters */
static int probe_fired_nonpsp = 0;
static int probe_fired_ring = 0;
static int probe_fired_submit = 0;

/*
 * Kprobe on psp_load_non_psp_fw
 * This loads MEC/ME/PFP/RLC etc. firmware through PSP
 * Signature: int psp_load_non_psp_fw(struct psp_context *psp)
 *   rdi = psp_context pointer
 */
static int nonpsp_pre(struct kprobe *p, struct pt_regs *regs)
{
	u64 psp_ctx = regs->di;

	probe_fired_nonpsp++;
	pr_info("fw30: >>> psp_load_non_psp_fw FIRED #%d: psp_ctx=%llx\n",
		probe_fired_nonpsp, psp_ctx);

	/*
	 * psp_context structure (from amdgpu_psp.h):
	 * We can't safely dereference the full structure, but let's
	 * read some offsets that are relatively stable:
	 *
	 * psp->cmd_buf_mem = DMA buffer for PSP commands
	 * psp->fw_pri_buf = primary firmware buffer
	 * psp->fence_buf = fence buffer
	 *
	 * Let's try to read the first few pointers
	 */
	{
		u64 *p64 = (u64 *)psp_ctx;
		int i;
		pr_info("fw30:   psp_context first 8 qwords:\n");
		for (i = 0; i < 8; i++) {
			u64 val = 0;
			if (!copy_from_kernel_nofault(&val, &p64[i], 8))
				pr_info("fw30:     [%d] = 0x%016llx\n", i, val);
		}
	}

	/* Try to read MP0 status during firmware loading */
	if (mp0_base) {
		pr_info("fw30:   MP0 state during load:\n");
		pr_info("fw30:     C2PMSG_35  = 0x%08X\n", rr(mp0_base + MP0_C2PMSG_35));
		pr_info("fw30:     C2PMSG_36  = 0x%08X\n", rr(mp0_base + MP0_C2PMSG_36));
		pr_info("fw30:     C2PMSG_64  = 0x%08X\n", rr(mp0_base + MP0_C2PMSG_64));
		pr_info("fw30:     C2PMSG_81  = 0x%08X\n", rr(mp0_base + MP0_C2PMSG_81));
	}

	return 0;
}

/*
 * Kprobe on psp_ring_cmd_submit
 * Signature: int psp_ring_cmd_submit(struct psp_context *psp,
 *                                     u64 cmd_buf_mc_addr,
 *                                     u64 fence_mc_addr, int index)
 *   rdi = psp_context, rsi = cmd_buf physical addr,
 *   rdx = fence physical addr, rcx = index
 */
static int ring_submit_pre(struct kprobe *p, struct pt_regs *regs)
{
	u64 psp_ctx = regs->di;
	u64 cmd_addr = regs->si;
	u64 fence_addr = regs->dx;
	u64 index = regs->cx;

	probe_fired_ring++;
	pr_info("fw30: >>> psp_ring_cmd_submit FIRED #%d:\n", probe_fired_ring);
	pr_info("fw30:   psp_ctx=%llx cmd_addr=%llx fence_addr=%llx index=%lld\n",
		psp_ctx, cmd_addr, fence_addr, index);

	/* The cmd_addr is a GPU physical address pointing to the PSP command.
	 * This is the DMA buffer that PSP reads. If we can find the
	 * corresponding kernel virtual address, we could modify it (TOCTOU).
	 */

	return 0;
}

/*
 * Kprobe on psp_cmd_submit_buf
 * Signature: int psp_cmd_submit_buf(struct psp_context *psp,
 *                                    struct amdgpu_firmware_info *ucode,
 *                                    struct psp_gfx_cmd_resp *cmd,
 *                                    u64 fence_mc_addr)
 *   rdi = psp_context, rsi = firmware_info, rdx = cmd, rcx = fence_addr
 *
 * This is the HIGH-LEVEL function that prepares and submits commands.
 * The cmd structure contains the actual PSP command bytes!
 */
static int cmd_submit_pre(struct kprobe *p, struct pt_regs *regs)
{
	u64 psp_ctx = regs->di;
	u64 fw_info = regs->si;
	u64 cmd_ptr = regs->dx;
	u64 fence_addr = regs->cx;

	probe_fired_submit++;
	pr_info("fw30: >>> psp_cmd_submit_buf FIRED #%d:\n", probe_fired_submit);
	pr_info("fw30:   psp=%llx fw_info=%llx cmd=%llx fence=%llx\n",
		psp_ctx, fw_info, cmd_ptr, fence_addr);

	/* Read the command buffer structure */
	if (cmd_ptr) {
		u32 cmd_data[16];
		int i;
		if (!copy_from_kernel_nofault(cmd_data, (void *)cmd_ptr, 64)) {
			pr_info("fw30:   PSP CMD first 16 dwords:\n");
			for (i = 0; i < 16; i += 4) {
				pr_info("fw30:     [%02d] %08X %08X %08X %08X\n",
					i, cmd_data[i], cmd_data[i+1],
					cmd_data[i+2], cmd_data[i+3]);
			}
		}
	}

	/* Read firmware info structure */
	if (fw_info) {
		u64 fw_data[8];
		int i;
		if (!copy_from_kernel_nofault(fw_data, (void *)fw_info, 64)) {
			pr_info("fw30:   FW_INFO first 8 qwords:\n");
			for (i = 0; i < 8; i++) {
				pr_info("fw30:     [%d] = 0x%016llx\n", i, fw_data[i]);
			}
		}
	}

	return 0;
}

static struct kprobe kp_nonpsp = {
	.symbol_name = "psp_load_non_psp_fw",
};

static struct kprobe kp_ring = {
	.symbol_name = "psp_ring_cmd_submit",
};

static struct kprobe kp_submit = {
	.symbol_name = "psp_cmd_submit_buf",
};

static int __init fw30_init(void)
{
	struct pci_dev *pdev = NULL;
	int ret, i, j;
	u32 val;

	pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
	if (!pdev)
		return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	pr_info("fw30: ====================================================\n");
	pr_info("fw30: PHASE 30: PSP ATTACK SURFACE MAPPING\n");
	pr_info("fw30: ====================================================\n");

	/* Section A: Find MP0 base by scanning for non-zero C2PMSG reads */
	pr_info("fw30: === FINDING MP0 BASE ===\n");
	for (i = 0; i < ARRAY_SIZE(mp0_bases); i++) {
		u32 base = mp0_bases[i];
		u32 v35, v36, v64, v81;

		v35 = rr(base + MP0_C2PMSG_35);
		v36 = rr(base + MP0_C2PMSG_36);
		v64 = rr(base + MP0_C2PMSG_64);
		v81 = rr(base + MP0_C2PMSG_81);

		pr_info("fw30: base=0x%05X: C2P_35=0x%08X C2P_36=0x%08X C2P_64=0x%08X C2P_81=0x%08X\n",
			base, v35, v36, v64, v81);

		/* A valid MP0 base will show non-zero/non-FFFF responses */
		if (v35 != 0 && v35 != 0xFFFFFFFF && !mp0_base) {
			mp0_base = base;
			pr_info("fw30: *** Candidate MP0 base: 0x%05X ***\n", base);
		}
	}

	/* Also try brute-force scan of likely MP0 range */
	pr_info("fw30: === BRUTE-FORCE MP0 SCAN ===\n");
	for (i = 0x380; i < 0x3A0; i++) {
		val = rr(i);
		if (val != 0 && val != 0xFFFFFFFF)
			pr_info("fw30: [0x%04X] = 0x%08X\n", i, val);
	}
	/* Try direct low-offset MP0 */
	for (i = 0x80; i < 0xA0; i++) {
		val = rr(i);
		if (val != 0 && val != 0xFFFFFFFF)
			pr_info("fw30: [0x%04X] = 0x%08X\n", i, val);
	}

	/* Section B: Scan for PSP version in SMN space via MP0 */
	pr_info("fw30: === MP0 REGISTER SCAN (0x16000-0x160A0) ===\n");
	for (i = 0x16000; i < 0x160A0; i++) {
		val = rr(i);
		if (val != 0 && val != 0xFFFFFFFF)
			pr_info("fw30: [0x%05X] = 0x%08X\n", i, val);
	}

	/* Also scan MP0 at indirect offset range */
	pr_info("fw30: === MP0 SCAN (0x3B100-0x3B1A0) ===\n");
	for (i = 0x3B100; i < 0x3B1A0; i++) {
		val = rr(i);
		if (val != 0 && val != 0xFFFFFFFF)
			pr_info("fw30: [0x%05X] = 0x%08X\n", i, val);
	}

	/* Section C: Try reading PSP through known SOC15 IP discovery
	 * The IP discovery table is at VRAM offset 0xC000
	 * We can't easily read VRAM from BAR5, but let's try the
	 * MMIO-visible portion of the discovery table
	 */
	pr_info("fw30: === SOC15 DISCOVERY SCAN ===\n");
	/* The amdgpu driver reads IP discovery from BAR0 (VRAM) or MMIO */
	/* Let's check if there's a version register at known locations */
	pr_info("fw30: RCC_DEV0_EPF0_STRAP0 (0x%04X) = 0x%08X\n",
		0xCC30, rr(0xCC30));
	pr_info("fw30: RCC_STRAP0 (0x%04X) = 0x%08X\n",
		0xC500, rr(0xC500));

	/* Section D: Try to find PSP ring buffer info through MMIO */
	pr_info("fw30: === PSP RING BUFFER SEARCH ===\n");
	/* PSP ring wptr/rptr are typically in C2PMSG_51/52 or at fixed MMIO */
	/* Try reading some known PSP register ranges */
	for (i = 0x16060; i < 0x160A0; i++) {
		val = rr(i);
		if (val != 0 && val != 0xFFFFFFFF)
			pr_info("fw30: [0x%05X] = 0x%08X\n", i, val);
	}

	/* Section E: Fuse/security status scan */
	pr_info("fw30: === SECURITY / FUSE STATUS ===\n");
	/* CC_GC_SA_UNIT_DISABLE, CC_GC_PRIM_CONFIG, etc. */
	pr_info("fw30: FUSE_STATUS (0x%04X) = 0x%08X\n",
		0xAC00, rr(0xAC00));
	/* RLC_GPU_IOV_VF_ENABLE — check if SR-IOV active */
	pr_info("fw30: RLC_GPU_IOV_VF (0x%04X) = 0x%08X\n",
		0x4D9C, rr(0x4D9C));
	/* Security fuses / debug mode indicators */
	for (i = 0xAC00; i < 0xAC10; i++) {
		val = rr(i);
		pr_info("fw30: [0x%04X] = 0x%08X\n", i, val);
	}

	/* Section F: MES (Micro Engine Scheduler) — GFX11 uses MES for queue management */
	pr_info("fw30: === MES STATUS ===\n");
	/* MES has its own register block, separate from MEC */
	/* regCP_MES_CNTL, regCP_MES_INSTR_PNTR, etc. */
	/* On GFX11: MES regs around 0xA800-0xA830 area */
	for (i = 0xA800; i < 0xA830; i++) {
		val = rr(i);
		if (val != 0 && val != 0xFFFFFFFF)
			pr_info("fw30: [0x%04X] = 0x%08X\n", i, val);
	}

	/* Section G: GFX scratch registers — driver communication */
	pr_info("fw30: === SCRATCH REGISTERS ===\n");
	for (i = 0; i < 8; i++) {
		pr_info("fw30: SCRATCH[%d] = 0x%08X\n", i, rr(0x20C0 + i));
	}

	/* Section H: Register kprobes for GPU reset interception */
	pr_info("fw30: === ARMING KPROBES ===\n");

	kp_nonpsp.pre_handler = nonpsp_pre;
	ret = register_kprobe(&kp_nonpsp);
	if (ret < 0)
		pr_info("fw30: FAIL kprobe psp_load_non_psp_fw: %d\n", ret);
	else
		pr_info("fw30: Kprobe on psp_load_non_psp_fw at %pS\n", kp_nonpsp.addr);

	kp_ring.pre_handler = ring_submit_pre;
	ret = register_kprobe(&kp_ring);
	if (ret < 0)
		pr_info("fw30: FAIL kprobe psp_ring_cmd_submit: %d\n", ret);
	else
		pr_info("fw30: Kprobe on psp_ring_cmd_submit at %pS\n", kp_ring.addr);

	kp_submit.pre_handler = cmd_submit_pre;
	ret = register_kprobe(&kp_submit);
	if (ret < 0)
		pr_info("fw30: FAIL kprobe psp_cmd_submit_buf: %d\n", ret);
	else
		pr_info("fw30: Kprobe on psp_cmd_submit_buf at %pS\n", kp_submit.addr);

	pr_info("fw30: ====================================================\n");
	pr_info("fw30: Kprobes armed. Trigger GPU reset via:\n");
	pr_info("fw30:   cat /sys/kernel/debug/dri/1/amdgpu_gpu_recover\n");
	pr_info("fw30: Then check dmesg for 'fw30: >>>' messages.\n");
	pr_info("fw30: When done: sudo rmmod patch_mec_fw30\n");
	pr_info("fw30: ====================================================\n");

	pci_dev_put(pdev);
	return 0;  /* Stay loaded for kprobes */
}

static void __exit fw30_exit(void)
{
	unregister_kprobe(&kp_nonpsp);
	unregister_kprobe(&kp_ring);
	unregister_kprobe(&kp_submit);

	if (mmio) {
		struct pci_dev *pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
		if (pdev) {
			pci_iounmap(pdev, mmio);
			pci_dev_put(pdev);
		}
	}

	pr_info("fw30: Unloaded. nonpsp=%d ring=%d submit=%d fires\n",
		probe_fired_nonpsp, probe_fired_ring, probe_fired_submit);
}

module_init(fw30_init);
module_exit(fw30_exit);
