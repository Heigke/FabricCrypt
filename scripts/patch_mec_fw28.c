/*
 * patch_mec_fw28.c — Phase 28: Kprobe on MEC init + GPU reset window
 *
 * Strategy:
 *   1. Register kprobes on key functions in the MEC init path:
 *      - gfx_v11_0_config_mec_cache (writes IC_BASE + primes cache)
 *      - gfx_v11_0_cp_compute_enable (halts/unhalts MEC)
 *   2. In the pre-handler, try SRAM write via UCODE_ADDR/DATA
 *      (hardware might be in writable state during init)
 *   3. Also try IC_BASE write to redirect firmware source
 *   4. Trigger GPU recovery via debugfs
 *
 * If SRAM writes work during the init window, we have our injection path.
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
static int probe_fired_compute = 0;
static int probe_fired_cache = 0;

static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }
static void wr(u32 dw_off, u32 val) {
	writel(val, mmio + (u64)dw_off * 4);
	readl(mmio + (u64)dw_off * 4);
}

#define regCP_MEC1_INSTR_PNTR      0x021A8
#define regCP_MEC_CNTL             0x0A802
#define MEC_ME1_HALT               (1 << 30)

#define regCP_MEC_ME1_UCODE_ADDR   0x0A814
#define regCP_MEC_ME1_UCODE_DATA   0x0A815

#define regCP_CPC_IC_BASE_LO       0x0C930
#define regCP_CPC_IC_BASE_HI       0x0C931
#define regCP_CPC_IC_OP_CNTL       0x0C932

#define INST_NOP         0xBF800000

/*
 * Kprobe on gfx_v11_0_cp_compute_enable
 * This is called with (adev, enable) during init/reset
 * When enable=true, it unhalts MEC; when false, it halts
 */
static int compute_enable_pre(struct kprobe *p, struct pt_regs *regs)
{
	/* rdi = adev, rsi = enable (bool) */
	u64 adev_ptr = regs->di;
	u64 enable = regs->si;
	u32 mec_cntl, pc, ic_lo, ic_hi;
	u32 sram_val;
	int i;

	probe_fired_compute++;

	pr_info("fw28: >>> cp_compute_enable FIRED #%d: adev=%llx enable=%lld\n",
		probe_fired_compute, adev_ptr, enable);

	mec_cntl = rr(regCP_MEC_CNTL);
	pc = rr(regCP_MEC1_INSTR_PNTR);
	ic_lo = rr(regCP_CPC_IC_BASE_LO);
	ic_hi = rr(regCP_CPC_IC_BASE_HI);

	pr_info("fw28:   MEC_CNTL=0x%08X PC=0x%04X IC_BASE=%08X:%08X\n",
		mec_cntl, pc, ic_hi, ic_lo);

	/* Try SRAM reads */
	wr(regCP_MEC_ME1_UCODE_ADDR, 0);
	udelay(10);
	sram_val = rr(regCP_MEC_ME1_UCODE_DATA);
	pr_info("fw28:   SRAM[0x000] = 0x%08X %s\n", sram_val,
		sram_val ? "NON-ZERO!" : "zero");

	wr(regCP_MEC_ME1_UCODE_ADDR, 0x44C);
	udelay(10);
	sram_val = rr(regCP_MEC_ME1_UCODE_DATA);
	pr_info("fw28:   SRAM[0x44C] = 0x%08X %s\n", sram_val,
		sram_val ? "NON-ZERO!" : "zero");

	/* Try SRAM write */
	pr_info("fw28:   Attempting SRAM write NOP to [0x000]...\n");
	wr(regCP_MEC_ME1_UCODE_ADDR, 0x000);
	udelay(10);
	wr(regCP_MEC_ME1_UCODE_DATA, INST_NOP);
	udelay(10);

	/* Read back */
	wr(regCP_MEC_ME1_UCODE_ADDR, 0x000);
	udelay(10);
	sram_val = rr(regCP_MEC_ME1_UCODE_DATA);
	pr_info("fw28:   SRAM[0x000] readback = 0x%08X %s\n", sram_val,
		(sram_val == INST_NOP) ? "SRAM WRITE WORKS!!!" :
		sram_val ? "different" : "still zero");

	/* Try IC_BASE write */
	pr_info("fw28:   Attempting IC_BASE write...\n");
	wr(regCP_CPC_IC_BASE_LO, 0xDEAD0000);
	udelay(10);
	{
		u32 rb = rr(regCP_CPC_IC_BASE_LO);
		pr_info("fw28:   IC_BASE_LO readback = 0x%08X %s\n", rb,
			(rb == 0xDEAD0000) ? "IC_BASE WRITABLE!!!" :
			(rb == ic_lo) ? "no change" : "different");
	}

	/* Restore IC_BASE */
	wr(regCP_CPC_IC_BASE_LO, ic_lo);

	/* Read first 8 SRAM locations */
	pr_info("fw28:   Full SRAM dump [0..7]:\n");
	for (i = 0; i < 8; i++) {
		wr(regCP_MEC_ME1_UCODE_ADDR, i);
		udelay(5);
		sram_val = rr(regCP_MEC_ME1_UCODE_DATA);
		pr_info("fw28:     [0x%03X] = 0x%08X\n", i, sram_val);
	}

	return 0; /* don't skip the original function */
}

/*
 * Kprobe on gfx_v11_0_config_mec_cache
 * This programs IC_BASE and primes the instruction cache
 */
static int config_cache_pre(struct kprobe *p, struct pt_regs *regs)
{
	u64 adev_ptr = regs->di;
	u32 mec_cntl, pc, ic_lo, ic_hi;

	probe_fired_cache++;

	pr_info("fw28: >>> config_mec_cache FIRED #%d: adev=%llx\n",
		probe_fired_cache, adev_ptr);

	mec_cntl = rr(regCP_MEC_CNTL);
	pc = rr(regCP_MEC1_INSTR_PNTR);
	ic_lo = rr(regCP_CPC_IC_BASE_LO);
	ic_hi = rr(regCP_CPC_IC_BASE_HI);

	pr_info("fw28:   MEC_CNTL=0x%08X PC=0x%04X IC_BASE=%08X:%08X\n",
		mec_cntl, pc, ic_hi, ic_lo);

	/* Try SRAM read here too */
	{
		u32 s0, s1;
		wr(regCP_MEC_ME1_UCODE_ADDR, 0);
		udelay(10);
		s0 = rr(regCP_MEC_ME1_UCODE_DATA);
		wr(regCP_MEC_ME1_UCODE_ADDR, 0x44C);
		udelay(10);
		s1 = rr(regCP_MEC_ME1_UCODE_DATA);
		pr_info("fw28:   SRAM[0]=0x%08X SRAM[0x44C]=0x%08X\n", s0, s1);
	}

	return 0;
}

static struct kprobe kp_compute = {
	.symbol_name = "gfx_v11_0_cp_compute_enable",
};

static struct kprobe kp_cache = {
	.symbol_name = "gfx_v11_0_config_mec_cache",
};

static int __init fw28_init(void)
{
	struct pci_dev *pdev = NULL;
	int ret;

	pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
	if (!pdev)
		return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	pr_info("fw28: ========================================\n");
	pr_info("fw28: PHASE 28: KPROBE MEC INIT + GPU RESET\n");
	pr_info("fw28: ========================================\n");

	pr_info("fw28: Initial state: PC=0x%04X MEC_CNTL=0x%08X\n",
		rr(regCP_MEC1_INSTR_PNTR), rr(regCP_MEC_CNTL));
	pr_info("fw28: IC_BASE: LO=0x%08X HI=0x%08X\n",
		rr(regCP_CPC_IC_BASE_LO), rr(regCP_CPC_IC_BASE_HI));

	/* Register kprobes */
	kp_compute.pre_handler = compute_enable_pre;
	ret = register_kprobe(&kp_compute);
	if (ret < 0) {
		pr_info("fw28: Failed to register kprobe on cp_compute_enable: %d\n", ret);
	} else {
		pr_info("fw28: Kprobe registered on cp_compute_enable at %pS\n",
			kp_compute.addr);
	}

	kp_cache.pre_handler = config_cache_pre;
	ret = register_kprobe(&kp_cache);
	if (ret < 0) {
		pr_info("fw28: Failed to register kprobe on config_mec_cache: %d\n", ret);
	} else {
		pr_info("fw28: Kprobe registered on config_mec_cache at %pS\n",
			kp_cache.addr);
	}

	pr_info("fw28: Kprobes armed. Trigger GPU reset via:\n");
	pr_info("fw28:   echo 1 > /sys/kernel/debug/dri/1/amdgpu_gpu_recover\n");
	pr_info("fw28: Then check dmesg for 'fw28: >>>' messages.\n");
	pr_info("fw28: When done: sudo rmmod patch_mec_fw28\n");
	pr_info("fw28: ========================================\n");

	/* Keep the module loaded (return 0) so kprobes stay active */
	pci_dev_put(pdev);
	/* Note: we keep mmio mapped for use in kprobe handlers */

	return 0;
}

static void __exit fw28_exit(void)
{
	unregister_kprobe(&kp_compute);
	unregister_kprobe(&kp_cache);

	if (mmio) {
		struct pci_dev *pdev = pci_get_device(AMD_VENDOR_ID, AMD_DEV_ID, NULL);
		if (pdev) {
			pci_iounmap(pdev, mmio);
			pci_dev_put(pdev);
		}
	}

	pr_info("fw28: Kprobes removed. compute_enable fired %d times, config_cache fired %d times\n",
		probe_fired_compute, probe_fired_cache);
}

module_init(fw28_init);
module_exit(fw28_exit);
