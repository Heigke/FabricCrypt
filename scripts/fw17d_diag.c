/*
 * fw17d_diag.c — Read IC_BASE registers + scan for actual GPU VA
 *
 * Read MEC IC_BASE_LO/HI to get the actual firmware GPU VA,
 * then scan for it in amdgpu_device.
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>

MODULE_LICENSE("GPL");

/* IC_BASE registers from TOC analysis (BASE_IDX=1, add 0xA000) */
#define regCP_MEC_ME1_IC_BASE_LO   0x0C930   /* from gc_11_0_0 */
#define regCP_MEC_ME1_IC_BASE_HI   0x0C931

/* Also try the register pairs from TOC file */
#define regR1_LO 0x15B60
#define regR1_HI 0x15B61
#define regR2_LO 0x15B6A
#define regR2_HI 0x15B6B
#define regR3_LO 0x15B68
#define regR3_HI 0x15B69
#define regR4_LO 0x15880
#define regR4_HI 0x15881

/* Standard MEC/CPC registers */
#define regCP_MEC1_INSTR_PNTR  0x021A8
#define regCP_MEC_CNTL         0x0A802

static void __iomem *mmio;

static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }

static int __init fw17d_init(void)
{
	struct pci_dev *pdev;
	void *drvdata;
	u64 gpu_va;
	u32 lo, hi;

	pdev = pci_get_device(0x1002, 0x1586, NULL);
	if (!pdev)
		return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENODEV;
	}

	drvdata = pci_get_drvdata(pdev);

	pr_info("fw17d: ========================================\n");
	pr_info("fw17d: READ IC_BASE REGISTERS\n");
	pr_info("fw17d: ========================================\n");
	pr_info("fw17d: PC = 0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	pr_info("fw17d: MEC_CNTL = 0x%08X\n", rr(regCP_MEC_CNTL));

	/* Read IC_BASE_LO/HI */
	lo = rr(regCP_MEC_ME1_IC_BASE_LO);
	hi = rr(regCP_MEC_ME1_IC_BASE_HI);
	gpu_va = ((u64)hi << 32) | lo;
	pr_info("fw17d: IC_BASE: LO=0x%08X HI=0x%08X => VA=0x%llX\n", lo, hi, gpu_va);

	/* Try all register pairs from TOC */
	lo = rr(regR1_LO); hi = rr(regR1_HI);
	pr_info("fw17d: R1(0x5B60/61): LO=0x%08X HI=0x%08X => 0x%llX\n", lo, hi, ((u64)hi << 32) | lo);

	lo = rr(regR2_LO); hi = rr(regR2_HI);
	pr_info("fw17d: R2(0x5B6A/6B): LO=0x%08X HI=0x%08X => 0x%llX\n", lo, hi, ((u64)hi << 32) | lo);

	lo = rr(regR3_LO); hi = rr(regR3_HI);
	pr_info("fw17d: R3(0x5B68/69): LO=0x%08X HI=0x%08X => 0x%llX\n", lo, hi, ((u64)hi << 32) | lo);

	lo = rr(regR4_LO); hi = rr(regR4_HI);
	pr_info("fw17d: R4(0x5880/81): LO=0x%08X HI=0x%08X => 0x%llX\n", lo, hi, ((u64)hi << 32) | lo);

	/* Try offset variants (without 0xA000 prefix, BASE_IDX=0) */
	lo = rr(0x0B60); hi = rr(0x0B61);
	pr_info("fw17d: 0x0B60/61: LO=0x%08X HI=0x%08X => 0x%llX\n", lo, hi, ((u64)hi << 32) | lo);

	lo = rr(0x5B60); hi = rr(0x5B61);
	pr_info("fw17d: 0x5B60/61: LO=0x%08X HI=0x%08X => 0x%llX\n", lo, hi, ((u64)hi << 32) | lo);

	lo = rr(0x5880); hi = rr(0x5881);
	pr_info("fw17d: 0x5880/81: LO=0x%08X HI=0x%08X => 0x%llX\n", lo, hi, ((u64)hi << 32) | lo);

	/* Also scan for interesting GPU VA patterns in all the BASE_IDX=1 range */
	{
		int offsets[] = {0x0C930, 0x0C931, 0x0C932, 0x0C933,
		                 0x0C934, 0x0C935, 0x0C936, 0x0C937};
		int j;
		pr_info("fw17d: IC_BASE region 0xC930-0xC937:\n");
		for (j = 0; j < 8; j++)
			pr_info("fw17d:   [0x%05X] = 0x%08X\n", offsets[j], rr(offsets[j]));
	}

	/* Now scan for the actual IC_BASE VA value in the amdgpu_device struct */
	if (gpu_va > 0x100000000ULL && drvdata) {
		u64 *scan = (u64 *)drvdata;
		int i;
		pr_info("fw17d: Scanning 4MB for IC_BASE VA 0x%llX...\n", gpu_va);
		for (i = 0; i < (4 * 1024 * 1024) / 8; i++) {
			u64 val;
			if (copy_from_kernel_nofault(&val, &scan[i], sizeof(val)))
				continue;
			if (val == gpu_va) {
				u64 prev = 0;
				copy_from_kernel_nofault(&prev, &scan[i-1], sizeof(prev));
				pr_info("fw17d: FOUND VA at offset 0x%lX, prev=0x%llX\n",
					(long)i * 8, prev);
			}
		}
	}

	pr_info("fw17d: ========================================\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit fw17d_exit(void) {}
module_init(fw17d_init);
module_exit(fw17d_exit);
