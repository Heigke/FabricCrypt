/*
 * psp_cmd_probe.c — Direct PSP command submission for RE
 *
 * mode=1: GET_FW_ATTESTATION — get firmware attestation DB address
 * mode=2: BOOT_CFG GET — read current boot configuration
 * mode=3: SAVE_RESTORE — save MEC firmware from TMR to accessible buffer
 * mode=4: PROG_REG — try to program IC_BASE via PSP
 * mode=5: DESTROY_TMR — destroy TMR and check if protection drops
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/slab.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("FEEL");
MODULE_DESCRIPTION("PSP command probe for RE");

static int mode = 1;
module_param(mode, int, 0644);

/* PSP command buffer structure (simplified) */
struct psp_cmd_buf {
	u32 buf_size;
	u32 buf_version;     /* must be 1 */
	u32 cmd_id;
	u32 resp_buf_addr_lo;
	u32 resp_buf_addr_hi;
	u32 resp_offset;
	u32 resp_buf_size;
	union {
		struct { /* SETUP_TMR / DESTROY_TMR */
			u32 buf_phy_addr_lo;
			u32 buf_phy_addr_hi;
			u32 buf_size_tmr;
			u32 tmr_flags;
			u32 system_phy_addr_lo;
			u32 system_phy_addr_hi;
		} tmr;
		struct { /* LOAD_IP_FW */
			u32 fw_phy_addr_lo;
			u32 fw_phy_addr_hi;
			u32 fw_size;
			u32 fw_type;
		} load_fw;
		struct { /* SAVE_RESTORE_IP_FW */
			u32 save_fw;          /* 1=save, 0=restore */
			u32 save_restore_addr_lo;
			u32 save_restore_addr_hi;
			u32 buf_size_sr;
			u32 fw_type;
		} save_restore;
		struct { /* PROG_REG */
			u32 reg_value;
			u32 reg_id;
		} reg_prog;
		struct { /* BOOT_CFG */
			u32 timestamp;
			u32 sub_cmd;    /* 1=SET, 2=GET, 3=INVALIDATE */
			u32 boot_config;
			u32 boot_config_valid;
		} boot_cfg;
		u32 raw[16];
	} cmd;
	u32 fence;           /* response fence value */
	u32 status;          /* response status */
	u32 pad[8];
};

/* Correct signature: psp_cmd_submit_buf(psp, ucode, cmd, fence_mc_addr) */
typedef int (*psp_submit_fn)(void *psp, void *ucode, struct psp_cmd_buf *cmd, u64 fence_mc_addr);

static int __init psp_cmd_init(void)
{
	struct pci_dev *pdev;
	void *drm_dev, *adev, *psp;
	psp_submit_fn submit_buf;
	struct psp_cmd_buf *cmd;
	int ret = 0;

	pr_info("psp_cmd: === PSP COMMAND PROBE mode=%d ===\n", mode);

	pdev = pci_get_device(0x1002, 0x1586, NULL);
	if (!pdev) { pdev = pci_get_device(0x1002, PCI_ANY_ID, NULL); }
	if (!pdev) { pr_info("psp_cmd: No AMD GPU\n"); return -ENODEV; }

	drm_dev = pci_get_drvdata(pdev);
	adev = drm_dev ? (void *)((u8 *)drm_dev - 0x10) : NULL;
	if (!adev) { pci_dev_put(pdev); return -ENODEV; }

	/* Find psp_cmd_submit_buf function address
	 * We know it's at 0xFFFFFFFFC0F2F840 from earlier probing */
	submit_buf = (psp_submit_fn)0xFFFFFFFFC0F2F840ULL;

	/* Find PSP context in adev — typically at a known offset.
	 * We found psp_context earlier. Let's search adev for the PSP
	 * ring structures. */
	{
		/* PSP context is embedded in adev. Search for it by looking
		 * for known PSP field patterns. The psp_context has:
		 * - fence_buf_mc_addr (GART range ~0x7FFF00...)
		 * - cmd_buf_mc_addr
		 * - tmr_mc_addr = 0x97E0000000
		 */
		int off;
		u64 psp_off = 0;
		for (off = 0; off < 0x80000; off += 8) {
			u64 val = *(u64 *)((u8 *)adev + off);
			if (val == 0x97E0000000ULL) {
				/* TMR MC addr found — PSP context is nearby */
				pr_info("psp_cmd: TMR MC at adev+0x%X\n", off);
				/* psp is typically at adev + some offset before TMR field */
				psp_off = off;
			}
		}

		/* Find PSP fence buf / cmd buf MC addresses nearby */
		if (psp_off) {
			int j;
			pr_info("psp_cmd: Context around TMR MC:\n");
			for (j = -16; j <= 16; j++) {
				u64 v = *(u64 *)((u8 *)adev + psp_off + j * 8);
				if (v != 0)
					pr_info("psp_cmd:   adev+0x%lX = 0x%016llX\n",
						psp_off + j * 8, v);
			}
		}
	}

	/* For now, get PSP as &adev->psp. We need to find the offset.
	 * psp_cmd_submit_buf takes (struct psp_context *psp, ...).
	 * struct psp_context is embedded in amdgpu_device.
	 * Let's search for it by looking for function pointers that
	 * match known PSP functions. */
	{
		/* Actually, psp_cmd_submit_buf is a high-level function.
		 * Let's look at how the driver calls it:
		 *   psp_cmd_submit_buf(psp, cmd_buf);
		 * where psp = &adev->psp
		 *
		 * We need to find the offset of psp within adev.
		 * The PSP ring functions store psp->fence_buf.mc_addr etc.
		 * We found TMR MC at some offset. The PSP struct has
		 * tmr_context.tmr_mc_addr or similar.
		 *
		 * For safety, let's just try calling submit_buf with
		 * adev + various offsets as "psp" until we find the right one.
		 * Actually that's dangerous.
		 *
		 * Better approach: find psp by looking for the psp_ring struct
		 * which has known patterns (ring_wptr, ring buffer GPU VA).
		 */

		/* Let's try a different approach: use the KNOWN psp_cmd_submit_buf
		 * address and call it with a cmd buffer, but we need the right
		 * psp pointer. The driver code shows psp = &adev->psp.
		 *
		 * From our earlier kernel modules, we successfully called PSP
		 * functions. Let me re-examine how we found psp there.
		 *
		 * In patch_mec_fw35.c, psp was found via:
		 *   psp = (void *)((u8 *)adev + psp_offset)
		 * where psp_offset was determined by symbol analysis.
		 *
		 * We know psp_ring_cmd_submit = 0xFFFFFFFFC0F2F6E0.
		 * And psp_cmd_submit_buf = 0xFFFFFFFFC0F2F840.
		 * These take struct psp_context * as first arg.
		 *
		 * Let's just scan adev for the TMR MC address and use
		 * the surrounding context to identify the PSP struct. */
	}

	/* Allocate command buffer */
	cmd = kzalloc(sizeof(*cmd), GFP_KERNEL);
	if (!cmd) { pci_dev_put(pdev); return -ENOMEM; }

	if (mode == 1) {
		/* GET_FW_ATTESTATION — just print the request, we can't
		 * safely call PSP without the correct psp pointer */
		pr_info("psp_cmd: GET_FW_ATTESTATION (cmd=0xF) — need PSP context\n");
	}

	if (mode == 2) {
		/* BOOT_CFG GET */
		pr_info("psp_cmd: BOOT_CFG GET — building command\n");
		cmd->buf_size = sizeof(*cmd);
		cmd->buf_version = 1;
		cmd->cmd_id = 0x22; /* BOOT_CFG */
		cmd->cmd.boot_cfg.sub_cmd = 2; /* GET */
		cmd->cmd.boot_cfg.timestamp = 0;
		cmd->cmd.boot_cfg.boot_config = 0;
		cmd->cmd.boot_cfg.boot_config_valid = 0xFFFFFFFF;

		pr_info("psp_cmd: cmd_id=0x%X sub_cmd=%d\n",
			cmd->cmd_id, cmd->cmd.boot_cfg.sub_cmd);
		pr_info("psp_cmd: (Cannot submit without PSP context pointer)\n");
	}

	/* Search for PSP context more aggressively */
	pr_info("psp_cmd: === PSP CONTEXT SEARCH ===\n");
	{
		/* The psp_context struct contains:
		 * - struct psp_ring km_ring (with ring_mem GPU VA)
		 * - struct amdgpu_bo *tmr_bo
		 * - u64 tmr_mc_addr
		 * - struct amdgpu_bo *fw_pri_bo
		 * - u64 fw_pri_mc_addr
		 *
		 * The km_ring has a GART-range GPU VA (0x7FFF...).
		 * tmr_mc_addr = 0x97E0000000.
		 *
		 * Let's find ALL occurrences of 0x97E0000000 and nearby
		 * GART addresses. */
		int off;
		int found_count = 0;

		for (off = 0; off < 0x100000 && found_count < 5; off += 8) {
			u64 val = *(u64 *)((u8 *)adev + off);
			if (val == 0x97E0000000ULL) {
				found_count++;
				pr_info("psp_cmd: TMR MC at adev+0x%X\n", off);

				/* Dump surrounding 256 bytes */
				{
					int j;
					for (j = -8; j <= 24; j++) {
						u64 v = *(u64 *)((u8 *)adev + off + j * 8);
						const char *tag = "";
						if (v == 0x97E0000000ULL) tag = " ← TMR_MC";
						else if ((v >> 44) == 0x7FFF0) tag = " ← GART?";
						else if ((v >> 48) == 0xFFFF) tag = " ← kptr";
						else if (v > 0x80000000ULL && v < 0x98000000ULL &&
						         (v & 0xFFF) == 0) tag = " ← VRAM?";
						if (v != 0 || j == 0)
							pr_info("psp_cmd:   [+0x%X] = 0x%016llX%s\n",
								off + j * 8, v, tag);
					}
				}
			}
		}
	}

	/* Find PSP context by searching backward from TMR MC for adev pointer */
	{
		u64 adev_val = (u64)(unsigned long)adev;
		int off;
		void *psp_ctx = NULL;

		pr_info("psp_cmd: Searching for adev ptr 0x%llX backward from TMR...\n",
			adev_val);

		for (off = 0x3BAA0; off > 0x3A000; off -= 8) {
			u64 val = *(u64 *)((u8 *)adev + off);
			if (val == adev_val) {
				pr_info("psp_cmd: *** adev ptr at adev+0x%X ***\n", off);
				pr_info("psp_cmd: PSP context likely starts at adev+0x%X\n", off);
				psp_ctx = (u8 *)adev + off;

				/* Dump first 64 bytes of PSP context */
				{
					int j;
					pr_info("psp_cmd: PSP context dump:\n");
					for (j = 0; j < 32; j++) {
						u64 v = *(u64 *)((u8 *)psp_ctx + j * 8);
						if (v != 0)
							pr_info("psp_cmd:   psp+0x%03X = 0x%016llX\n",
								j * 8, v);
					}
				}
				break;
			}
		}

		if (psp_ctx && mode >= 2) {
			/* Find fence_buf_mc_addr in PSP context.
			 * It's after fence_buf_bo (kptr) in the struct.
			 * Search for a GART-range MC address near the PSP context. */
			u64 fence_mc = 0;
			{
				int f;
				for (f = 0; f < 0x400; f += 8) {
					u64 v = *(u64 *)((u8 *)psp_ctx + f);
					/* Fence buf MC should be in GART range */
					if ((v >> 44) == 0x7FFF0 && v != 0x7FFF00700000ULL) {
						u64 prev = *(u64 *)((u8 *)psp_ctx + f - 8);
						if ((prev >> 48) == 0xFFFF) {
							/* prev is BO ptr, this is MC addr */
							fence_mc = v;
							pr_info("psp_cmd: fence_buf_mc at psp+0x%X = 0x%llX\n",
								f, v);
							break;
						}
					}
				}
				if (!fence_mc) {
					/* Try looking for any GART MC addr */
					for (f = 0; f < 0x400; f += 8) {
						u64 v = *(u64 *)((u8 *)psp_ctx + f);
						if (v >= 0x7FFF00000000ULL && v < 0x7FFF10000000ULL &&
						    v != 0x7FFF00700000ULL) {
							fence_mc = v;
							pr_info("psp_cmd: candidate fence_mc at psp+0x%X = 0x%llX\n",
								f, v);
							break;
						}
					}
				}
			}

			if (!fence_mc) {
				pr_info("psp_cmd: Cannot find fence_buf_mc_addr, aborting\n");
				goto done;
			}

			/* NOW we can submit PSP commands! */
			pr_info("psp_cmd: === SUBMITTING PSP COMMANDS ===\n");
			pr_info("psp_cmd: fence_mc = 0x%llX\n", fence_mc);

			memset(cmd, 0, sizeof(*cmd));
			cmd->buf_size = sizeof(*cmd);
			cmd->buf_version = 1;

			if (mode == 2) {
				/* BOOT_CFG GET */
				cmd->cmd_id = 0x22;
				cmd->cmd.boot_cfg.sub_cmd = 2; /* GET */
				cmd->cmd.boot_cfg.boot_config_valid = 0xFFFFFFFF;

				pr_info("psp_cmd: Submitting BOOT_CFG GET...\n");
				ret = submit_buf(psp_ctx, NULL, cmd, fence_mc);
				pr_info("psp_cmd: BOOT_CFG result: ret=%d status=0x%X\n",
					ret, cmd->status);
				pr_info("psp_cmd: boot_config = 0x%08X\n",
					cmd->cmd.boot_cfg.boot_config);
				pr_info("psp_cmd: boot_config_valid = 0x%08X\n",
					cmd->cmd.boot_cfg.boot_config_valid);
			}

			if (mode == 3) {
				/* SAVE_RESTORE — save MEC firmware */
				/* First we need a GPU-accessible buffer for the save.
				 * Use the GART BO we know: adev+0x3B958 has a BO
				 * with mc=0x7FFF00700000, cpu=writable.
				 * The GART BO is ~436KB (firmware blob size).
				 * We'll save MEC firmware there. */
				u64 gart_mc = *(u64 *)((u8 *)adev + 0x3B960);
				u64 gart_cpu = *(u64 *)((u8 *)adev + 0x3B968);

				pr_info("psp_cmd: SAVE MEC FW to GART BO mc=0x%llX\n",
					gart_mc);

				cmd->cmd_id = 0x08; /* SAVE_RESTORE */
				cmd->cmd.save_restore.save_fw = 1; /* SAVE */
				cmd->cmd.save_restore.save_restore_addr_lo =
					lower_32_bits(gart_mc);
				cmd->cmd.save_restore.save_restore_addr_hi =
					upper_32_bits(gart_mc);
				cmd->cmd.save_restore.buf_size_sr = 0x80000; /* 512KB */
				cmd->cmd.save_restore.fw_type = 4; /* CP_MEC */

				pr_info("psp_cmd: Submitting SAVE_RESTORE (save MEC)...\n");
				ret = submit_buf(psp_ctx, NULL, cmd, fence_mc);
				pr_info("psp_cmd: SAVE result: ret=%d status=0x%X\n",
					ret, cmd->status);

				/* If successful, read saved firmware from GART BO! */
				if (ret == 0 && cmd->status == 0 && gart_cpu) {
					u32 code[4];
					int i;
					for (i = 0; i < 4; i++)
						copy_from_kernel_nofault(&code[i],
							(void *)(unsigned long)(gart_cpu + i * 4), 4);
					pr_info("psp_cmd: *** SAVED FW: %08X %08X %08X %08X ***\n",
						code[0], code[1], code[2], code[3]);
				}
			}

			if (mode == 4) {
				/* PROG_REG — try to write IC_BASE_LO */
				cmd->cmd_id = 0x0B; /* PROG_REG */
				cmd->cmd.reg_prog.reg_value = 0xDEADBEEF;
				cmd->cmd.reg_prog.reg_id = 0; /* unknown reg ID scheme */

				pr_info("psp_cmd: Submitting PROG_REG...\n");
				ret = submit_buf(psp_ctx, NULL, cmd, fence_mc);
				pr_info("psp_cmd: PROG_REG result: ret=%d status=0x%X\n",
					ret, cmd->status);
			}

			if (mode == 5) {
				/* GET_FW_ATTESTATION */
				cmd->cmd_id = 0x0F;

				pr_info("psp_cmd: Submitting GET_FW_ATTESTATION...\n");
				ret = submit_buf(psp_ctx, NULL, cmd, fence_mc);
				pr_info("psp_cmd: FW_ATTEST result: ret=%d status=0x%X\n",
					ret, cmd->status);
				pr_info("psp_cmd: Response data: %08X %08X %08X %08X\n",
					cmd->cmd.raw[0], cmd->cmd.raw[1],
					cmd->cmd.raw[2], cmd->cmd.raw[3]);
			}
		}
	}

done:
	kfree(cmd);
	pci_dev_put(pdev);
	return 0;
}

static void __exit psp_cmd_exit(void)
{
	pr_info("psp_cmd: unloaded\n");
}

module_init(psp_cmd_init);
module_exit(psp_cmd_exit);
