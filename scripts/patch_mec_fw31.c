/*
 * patch_mec_fw31.c — Phase 31: PSP Command Injection (v2)
 *
 * Phase 30 gave us:
 *   - MP0 base = 0x16000
 *   - C2PMSG_35 (resp) at 0x16063 = 0xFFFFFFFF (bootloader ready!)
 *   - C2PMSG_64 (cmd)  at 0x16080 = 0x80C20000
 *   - C2PMSG_69 (blcmd) at 0x16085 = 0xFF7A3000
 *   - Ring RPTR/WPTR at 0x16083/0x16084 (currently 0x280)
 *   - Ring phys at 0x1_16F9F000 (system memory, not MMIO)
 *   - PSP command for MEC: cmd_id=6, fw_type=0x31
 *
 * Two attack strategies:
 *   A. Direct PSP ring injection via phys_to_virt() or page_address()
 *   B. C2PMSG mailbox commands (bootloader protocol still active?)
 *   C. Kprobe: modify MEC load command in-flight during GPU reset
 *
 * PSP bootloader protocol (psp_v13_0_4):
 *   1. Poll C2PMSG_35 for bit 31 set (ready)
 *   2. Clear C2PMSG_36
 *   3. Write fw_addr >> 20 to C2PMSG_35
 *   4. Write bl_cmd to C2PMSG_69
 *   5. Poll C2PMSG_35 for bit 31 set (done)
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/kprobes.h>
#include <linux/dma-mapping.h>
#include <linux/mm.h>
#include <linux/pfn.h>

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

/* MP0 C2PMSG registers (relative to MMIO, not MP0 base) */
#define C2PMSG_33   0x16061  /* Interrupt/status */
#define C2PMSG_35   0x16063  /* Response / address */
#define C2PMSG_36   0x16064  /* Status / clear */
#define C2PMSG_51   0x16073  /* Ring RPTR? */
#define C2PMSG_64   0x16080  /* Command */
#define C2PMSG_69   0x16085  /* Bootloader command trigger */
#define C2PMSG_81   0x16091  /* Bootloader status */

/* PSP ring RPTR/WPTR (these might be at different C2PMSG regs) */
#define REG_RING_RPTR   0x16083
#define REG_RING_WPTR   0x16084

/* Known PSP bootloader commands */
#define PSP_BL2_LOAD_SYSDRV     0x10000
#define PSP_BL2_LOAD_DBGDRV     0x80000   /* Debug driver! */
#define PSP_BL2_LOAD_INTFDRV    0xA0000
#define PSP_BL2_LOAD_SOCDRV     0xB0000
#define PSP_BL2_LOAD_SPL        0x120000
#define PSP_BL2_LOAD_KDB        0x140000

/* PSP ring entry */
struct psp_ring_entry {
	u32 cmd_addr_lo;
	u32 cmd_addr_hi;
	u32 fence_addr_lo;
	u32 fence_addr_hi;
};

/* DMA buffers for PSP commands */
static u32 *cmd_buf;
static dma_addr_t cmd_dma;
static u32 *fence_buf;
static dma_addr_t fence_dma;

/* Kprobe state */
static int probe_count = 0;
static int mec_intercepted = 0;

/*
 * Strategy A: Direct PSP ring injection
 * Use phys_to_virt() to access the ring buffer in system memory
 */
static void try_ring_injection(void)
{
	u64 ring_phys = 0x116F9F000ULL;
	void *ring_virt;
	u32 rptr, wptr, new_wptr;
	struct psp_ring_entry *entry;
	struct page *page;
	int timeout;

	pr_info("fw31: === STRATEGY A: RING INJECTION ===\n");

	rptr = rr(REG_RING_RPTR);
	wptr = rr(REG_RING_WPTR);
	pr_info("fw31: Ring RPTR=0x%04X WPTR=0x%04X\n", rptr, wptr);

	/* Try phys_to_virt first */
	if (pfn_valid(ring_phys >> PAGE_SHIFT)) {
		ring_virt = phys_to_virt(ring_phys);
		pr_info("fw31: Ring mapped via phys_to_virt: %px\n", ring_virt);
	} else {
		/* Try kmap of the page */
		page = pfn_to_page(ring_phys >> PAGE_SHIFT);
		if (page) {
			ring_virt = page_address(page);
			pr_info("fw31: Ring mapped via page_address: %px\n", ring_virt);
		} else {
			pr_info("fw31: Cannot map ring buffer phys 0x%llx\n", ring_phys);
			return;
		}
	}

	/* Dump current ring content at WPTR position */
	{
		u32 *rdata = (u32 *)(ring_virt + wptr);
		pr_info("fw31: Ring[WPTR]: %08X %08X %08X %08X\n",
			rdata[0], rdata[1], rdata[2], rdata[3]);
	}

	/* Also dump a few entries before WPTR (recent submissions) */
	if (wptr >= 16) {
		u32 *rdata = (u32 *)(ring_virt + wptr - 16);
		pr_info("fw31: Ring[WPTR-1]: %08X %08X %08X %08X\n",
			rdata[0], rdata[1], rdata[2], rdata[3]);
	}

	if (!cmd_buf || !fence_buf) {
		pr_info("fw31: No DMA buffers, skipping ring submit\n");
		return;
	}

	/* Prepare a BOOT_CFG query command (cmd_id=15) */
	memset(cmd_buf, 0, 1024);
	cmd_buf[0] = 0;         /* status */
	cmd_buf[1] = 0;         /* session_id */
	cmd_buf[2] = 15;        /* cmd_id = BOOT_CFG */
	cmd_buf[3] = 0;         /* resp */
	cmd_buf[4] = 0;         /* resp_size */
	cmd_buf[5] = 0;         /* cmd_flags */
	cmd_buf[6] = 0;         /* reserved */
	cmd_buf[7] = 0;         /* reserved */
	/* Boot config data starts at [8] */
	cmd_buf[8] = 0;         /* timestamp */
	cmd_buf[9] = 0;         /* QUERY command */
	cmd_buf[10] = 0;        /* config value */

	*fence_buf = 0;

	/* Write ring entry at WPTR */
	entry = (struct psp_ring_entry *)(ring_virt + wptr);
	entry->cmd_addr_lo = lower_32_bits(cmd_dma);
	entry->cmd_addr_hi = upper_32_bits(cmd_dma);
	entry->fence_addr_lo = lower_32_bits(fence_dma);
	entry->fence_addr_hi = upper_32_bits(fence_dma);

	/* Advance WPTR */
	new_wptr = (wptr + 16) % 4096;  /* 4KB ring */

	pr_info("fw31: Submitting BOOT_CFG query via ring:\n");
	pr_info("fw31:   cmd_dma=0x%llx fence_dma=0x%llx new_wptr=0x%04X\n",
		(u64)cmd_dma, (u64)fence_dma, new_wptr);

	/* Write memory barrier before WPTR update */
	wmb();

	/* Update WPTR to trigger PSP */
	wr(REG_RING_WPTR, new_wptr);

	/* Wait for response */
	timeout = 50000;  /* 50ms */
	while (timeout > 0) {
		if (*fence_buf != 0)
			break;
		udelay(1);
		timeout--;
	}

	pr_info("fw31: BOOT_CFG result: fence=0x%08X status=0x%08X resp=0x%08X\n",
		*fence_buf, cmd_buf[0], cmd_buf[3]);
	pr_info("fw31: Response data: [8]=0x%08X [9]=0x%08X [10]=0x%08X [11]=0x%08X\n",
		cmd_buf[8], cmd_buf[9], cmd_buf[10], cmd_buf[11]);

	/* Check new ring state */
	pr_info("fw31: Post-submit: RPTR=0x%04X WPTR=0x%04X\n",
		rr(REG_RING_RPTR), rr(REG_RING_WPTR));

	/* If that worked, try PROG_REG (cmd_id=11) */
	if (*fence_buf != 0) {
		pr_info("fw31: === BOOT_CFG SUCCEEDED! Trying PROG_REG ===\n");

		memset(cmd_buf, 0, 1024);
		cmd_buf[2] = 11;   /* PROG_REG */
		/* reg_prog data at [8]: reg_value, reg_id */
		cmd_buf[8] = 0;    /* value = 0 (just set address) */
		cmd_buf[9] = 0xA814; /* reg_id = CP_MEC_ME1_UCODE_ADDR */

		*fence_buf = 0;

		entry = (struct psp_ring_entry *)(ring_virt + new_wptr);
		entry->cmd_addr_lo = lower_32_bits(cmd_dma);
		entry->cmd_addr_hi = upper_32_bits(cmd_dma);
		entry->fence_addr_lo = lower_32_bits(fence_dma);
		entry->fence_addr_hi = upper_32_bits(fence_dma);

		new_wptr = (new_wptr + 16) % 4096;
		wmb();
		wr(REG_RING_WPTR, new_wptr);

		timeout = 50000;
		while (timeout > 0) {
			if (*fence_buf != 0) break;
			udelay(1);
			timeout--;
		}

		pr_info("fw31: PROG_REG result: fence=0x%08X status=0x%08X resp=0x%08X\n",
			*fence_buf, cmd_buf[0], cmd_buf[3]);
		pr_info("fw31: UCODE_DATA readback = 0x%08X\n",
			rr(0xA815));
	}
}

/*
 * Strategy B: C2PMSG mailbox — try bootloader protocol
 */
static void try_mailbox(void)
{
	u32 resp;
	int timeout;

	pr_info("fw31: === STRATEGY B: C2PMSG MAILBOX ===\n");

	/* Read current mailbox state */
	pr_info("fw31: C2PMSG_33 = 0x%08X\n", rr(C2PMSG_33));
	pr_info("fw31: C2PMSG_35 = 0x%08X\n", rr(C2PMSG_35));
	pr_info("fw31: C2PMSG_36 = 0x%08X\n", rr(C2PMSG_36));
	pr_info("fw31: C2PMSG_64 = 0x%08X\n", rr(C2PMSG_64));
	pr_info("fw31: C2PMSG_69 = 0x%08X\n", rr(C2PMSG_69));
	pr_info("fw31: C2PMSG_81 = 0x%08X\n", rr(C2PMSG_81));

	/* Check if bootloader is ready (C2PMSG_35 bit 31) */
	resp = rr(C2PMSG_35);
	if (!(resp & 0x80000000)) {
		pr_info("fw31: Bootloader NOT ready (bit 31 not set)\n");
		/* Try anyway */
	}

	/* Try sending a bootloader DBGDRV load command
	 * This won't actually load anything useful (no FW in DMA buf)
	 * but will tell us if the bootloader protocol is still active
	 */
	pr_info("fw31: Trying bootloader DBGDRV command (probe only)...\n");

	/* Clear response */
	wr(C2PMSG_36, 0);
	udelay(10);

	/* Write a dummy address (our cmd_buf) */
	if (cmd_buf) {
		/* Put some identifiable data in cmd_buf for debugging */
		memset(cmd_buf, 0, 1024);
		cmd_buf[0] = 0xDEAD0001;  /* marker */

		wr(C2PMSG_35, (u32)(cmd_dma >> 20));  /* addr in MB units */
		udelay(10);

		/* Trigger debug driver load */
		wr(C2PMSG_69, PSP_BL2_LOAD_DBGDRV);
		udelay(100);

		/* Poll for response */
		timeout = 100;  /* 10ms at 100us intervals */
		while (timeout > 0) {
			resp = rr(C2PMSG_35);
			if (resp & 0x80000000) break;
			udelay(100);
			timeout--;
		}

		pr_info("fw31: DBGDRV response: C2PMSG_35=0x%08X C2PMSG_36=0x%08X\n",
			rr(C2PMSG_35), rr(C2PMSG_36));
		pr_info("fw31: After DBGDRV: C2PMSG_69=0x%08X C2PMSG_81=0x%08X\n",
			rr(C2PMSG_69), rr(C2PMSG_81));
	}

	/* Scan additional C2PMSG registers for any response changes */
	pr_info("fw31: Post-mailbox C2PMSG scan:\n");
	{
		int i;
		for (i = 0x16060; i <= 0x160A0; i++) {
			u32 v = rr(i);
			if (v != 0 && v != 0xFFFFFFFF)
				pr_info("fw31:   [0x%05X] = 0x%08X\n", i, v);
		}
	}
}

/*
 * Strategy C: Read PSP context structure to find ring virtual address
 * and other internal state. We got psp_ctx=0xffff8b4d2c63b910 from Phase 30.
 */
static void try_read_psp_context(void)
{
	/* psp_context pointer from Phase 30 kprobe */
	u64 psp_ctx_addr = 0xffff8b4d2c63b910ULL;
	u64 qwords[64];
	int i;

	pr_info("fw31: === STRATEGY C: READ PSP CONTEXT ===\n");

	/* Read first 512 bytes of psp_context */
	if (copy_from_kernel_nofault(qwords, (void *)psp_ctx_addr, sizeof(qwords))) {
		pr_info("fw31: Cannot read psp_context at %llx\n", psp_ctx_addr);
		return;
	}

	/* Look for interesting pointers and values */
	for (i = 0; i < 64; i++) {
		if (qwords[i] == 0)
			continue;
		/* Look for DMA addresses (0x1xxxxxxxx range) */
		if ((qwords[i] >> 32) == 0x1 ||
		    (qwords[i] >> 32) == 0x97 ||
		    (qwords[i] >> 32) == 0xffff8b4d ||
		    (qwords[i] >> 32) == 0xffffcf87) {
			pr_info("fw31:   psp_ctx[%d] = 0x%016llx%s\n",
				i, qwords[i],
				(qwords[i] & 0xFFFFFF) == 0xF9F000 ? " ← RING?" :
				(qwords[i] >> 32) == 0x1 ? " (DMA)" :
				(qwords[i] >> 32) == 0x97 ? " (GPU)" : "");
		}
	}

	/* Also read at larger offsets (the ring info might be deeper) */
	if (!copy_from_kernel_nofault(qwords, (void *)(psp_ctx_addr + 512), sizeof(qwords))) {
		for (i = 0; i < 64; i++) {
			if (qwords[i] == 0) continue;
			if ((qwords[i] >> 32) == 0x1 ||
			    (qwords[i] >> 32) == 0xffff8b4d ||
			    (qwords[i] >> 32) == 0xffffcf87) {
				pr_info("fw31:   psp_ctx[%d+64] = 0x%016llx%s\n",
					i, qwords[i],
					(qwords[i] & 0xFFFFFF) == 0xF9F000 ? " ← RING?" : "");
			}
		}
	}
}

/*
 * Kprobe: intercept psp_cmd_submit_buf during GPU reset
 * When MEC1 loads, log full command and try modification
 */
static int cmd_submit_pre(struct kprobe *p, struct pt_regs *regs)
{
	u64 cmd_ptr = regs->dx;
	u32 cmd_data[64];

	probe_count++;

	if (!cmd_ptr) return 0;
	if (copy_from_kernel_nofault(cmd_data, (void *)cmd_ptr, 256))
		return 0;

	/* Only interested in LOAD_IP_FW */
	if (cmd_data[2] != 6) return 0;

	pr_info("fw31: >>> LOAD_IP_FW #%d: fw_type=0x%04X addr=%08X:%08X size=0x%X\n",
		probe_count, cmd_data[10], cmd_data[8], cmd_data[7], cmd_data[9]);

	if (cmd_data[10] == 0x31) { /* MEC1 */
		mec_intercepted++;
		pr_info("fw31: *** MEC1 INTERCEPT ***\n");

		/* Log full non-zero command content */
		{
			int i;
			for (i = 0; i < 64; i += 4) {
				if (cmd_data[i] || cmd_data[i+1] || cmd_data[i+2] || cmd_data[i+3])
					pr_info("fw31:   [%02d] %08X %08X %08X %08X\n",
						i, cmd_data[i], cmd_data[i+1],
						cmd_data[i+2], cmd_data[i+3]);
			}
		}

		/* Check cmd_flags field */
		pr_info("fw31:   cmd_flags = 0x%08X (at offset 20)\n", cmd_data[5]);

		/*
		 * MODIFICATION ATTEMPT:
		 * Try setting bit 0 of cmd_flags (speculation: "skip verify"?)
		 * This is a controlled experiment — if the flag does nothing,
		 * PSP processes normally. If it crashes, we learn the flag is checked.
		 */
		/*
		 * DISABLED FOR NOW — observe first, modify in next phase
		 * {
		 *     u32 new_flags = cmd_data[5] | 1;
		 *     copy_to_kernel_nofault((void *)(cmd_ptr + 5*4), &new_flags, 4);
		 * }
		 */
	}

	return 0;
}

static struct kprobe kp_submit = {
	.symbol_name = "psp_cmd_submit_buf",
};

static int __init fw31_init(void)
{
	int ret;

	g_pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
	if (!g_pdev) return -ENODEV;

	mmio = pci_iomap(g_pdev, 5, 0);
	if (!mmio) { pci_dev_put(g_pdev); return -ENODEV; }

	pr_info("fw31: ====================================================\n");
	pr_info("fw31: PHASE 31: PSP COMMAND INJECTION v2\n");
	pr_info("fw31: ====================================================\n");

	/* Allocate DMA buffers */
	cmd_buf = dma_alloc_coherent(&g_pdev->dev, 4096, &cmd_dma, GFP_KERNEL);
	fence_buf = dma_alloc_coherent(&g_pdev->dev, 4096, &fence_dma, GFP_KERNEL);
	if (cmd_buf && fence_buf) {
		pr_info("fw31: DMA: cmd=%px (0x%llx) fence=%px (0x%llx)\n",
			cmd_buf, (u64)cmd_dma, fence_buf, (u64)fence_dma);
	} else {
		pr_info("fw31: DMA allocation failed!\n");
	}

	/* Strategy A: Ring injection */
	try_ring_injection();

	/* Strategy B: C2PMSG mailbox */
	try_mailbox();

	/* Strategy C: Read PSP context */
	try_read_psp_context();

	/* Strategy D: Arm kprobe for GPU reset */
	pr_info("fw31: === ARMING KPROBE ===\n");
	kp_submit.pre_handler = cmd_submit_pre;
	ret = register_kprobe(&kp_submit);
	if (ret < 0)
		pr_info("fw31: FAIL kprobe: %d\n", ret);
	else
		pr_info("fw31: Kprobe armed on psp_cmd_submit_buf\n");

	pr_info("fw31: ====================================================\n");
	pr_info("fw31: Trigger GPU reset: cat /sys/kernel/debug/dri/1/amdgpu_gpu_recover\n");
	pr_info("fw31: ====================================================\n");

	return 0;
}

static void __exit fw31_exit(void)
{
	unregister_kprobe(&kp_submit);
	if (fence_buf) dma_free_coherent(&g_pdev->dev, 4096, fence_buf, fence_dma);
	if (cmd_buf)   dma_free_coherent(&g_pdev->dev, 4096, cmd_buf, cmd_dma);
	if (mmio)      pci_iounmap(g_pdev, mmio);
	pci_dev_put(g_pdev);
	pr_info("fw31: Unloaded. %d cmds, %d MEC intercepts\n", probe_count, mec_intercepted);
}

module_init(fw31_init);
module_exit(fw31_exit);
