/*
 * mec_hotpatch.c — MEC IC_BASE redirect via direct MMIO
 * 
 * Register offsets from IP discovery table (UMR gfx1151):
 * All within BAR5 1MB — direct MMIO, no indirect needed.
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002

static void __iomem *mmio;

/* Correct GFX 11.5.1 register dword offsets from IP discovery */
#define regCP_MEC_CNTL              0x0A802
#define regCP_CPC_IC_BASE_LO        0x0F84C
#define regCP_CPC_IC_BASE_HI        0x0F84D
#define regCP_CPC_IC_BASE_CNTL      0x0F84E
#define regCP_CPC_IC_OP_CNTL        0x0C97A
#define regGRBM_GFX_CNTL            0x0A900
#define regCP_HQD_ACTIVE            0x0320B
#define regCP_MEC1_PRGRM_CNTR_START 0x030A6
#define regCP_MEC2_PRGRM_CNTR_START 0x030A7
#define regCP_MEC1_INSTR_PNTR       0x021A8
#define regCP_MEC2_INSTR_PNTR       0x021A9

/* CP_MEC_CNTL bits */
#define MEC_ME1_HALT            (1 << 30)
#define MEC_ME2_HALT            (1 << 28)
#define MEC_INVALIDATE_ICACHE   (1 << 27)

static u32 rr(u32 dw_off)
{
	return readl(mmio + (u64)dw_off * 4);
}

static void wr(u32 dw_off, u32 val)
{
	writel(val, mmio + (u64)dw_off * 4);
}

static void select_me_pipe_q(u32 me, u32 pipe, u32 queue, u32 vmid)
{
	u32 val = (pipe & 3) | ((me & 3) << 2) | ((vmid & 0xF) << 4) | ((queue & 7) << 8);
	wr(regGRBM_GFX_CNTL, val);
	rr(regGRBM_GFX_CNTL);
}

static int __init hotpatch_init(void)
{
	struct pci_dev *pdev = NULL;
	u32 mec_cntl, orig_grbm;
	void __iomem *vram_ptr;
	u64 vram_offset, gpu_va;
	int me, pipe;

	while ((pdev = pci_get_device(AMD_VENDOR_ID, PCI_ANY_ID, pdev))) {
		if ((pdev->class >> 8) == PCI_CLASS_DISPLAY_OTHER)
			break;
	}
	if (!pdev) return -ENODEV;

	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) { pci_dev_put(pdev); return -ENOMEM; }

	vram_ptr = pci_iomap(pdev, 0, 0);
	if (!vram_ptr) { pci_iounmap(pdev, mmio); pci_dev_put(pdev); return -ENOMEM; }

	/* === PHASE 1: Read current state === */
	mec_cntl = rr(regCP_MEC_CNTL);
	orig_grbm = rr(regGRBM_GFX_CNTL);
	
	pr_info("hp: CP_MEC_CNTL = 0x%08X\n", mec_cntl);
	pr_info("hp: MEC1_INSTR_PNTR = 0x%08X\n", rr(regCP_MEC1_INSTR_PNTR));
	pr_info("hp: MEC2_INSTR_PNTR = 0x%08X\n", rr(regCP_MEC2_INSTR_PNTR));
	
	/* Read IC_BASE for each ME/PIPE */
	for (me = 1; me <= 2; me++) {
		for (pipe = 0; pipe < 4; pipe++) {
			u32 lo, hi, cntl, op;
			select_me_pipe_q(me, pipe, 0, 0);
			lo = rr(regCP_CPC_IC_BASE_LO);
			hi = rr(regCP_CPC_IC_BASE_HI);
			cntl = rr(regCP_CPC_IC_BASE_CNTL);
			op = rr(regCP_CPC_IC_OP_CNTL);
			if (lo || hi)
				pr_info("hp: ME%d P%d: IC_BASE=0x%08X:%08X CNTL=0x%X OP=0x%X\n",
					me, pipe, hi, lo, cntl, op);
		}
	}
	wr(regGRBM_GFX_CNTL, orig_grbm);

	/* === PHASE 2: Write NOP loop to VRAM === */
	vram_offset = 0x10000;
	{
		u32 __iomem *code = (u32 __iomem *)(vram_ptr + vram_offset);
		int i;
		for (i = 0; i < 16; i++)
			writel(0xBF800000, &code[i]);  /* s_nop 0 */
		writel(0xBF82FFF0, &code[16]);     /* s_branch -16 */
		mb();
		pr_info("hp: VRAM[0x%llX] = 0x%08X (expect 0xBF800000)\n",
			vram_offset, readl(&code[0]));
	}

	/* VMID 0 GPU VA: system aperture base + VRAM offset */
	gpu_va = 0x8000000000ULL + vram_offset;
	pr_info("hp: Target GPU VA = 0x%016llX\n", gpu_va);
	pr_info("hp: IC_BASE_LO = 0x%08X, IC_BASE_HI = 0x%08X\n",
		(u32)(gpu_va & 0xFFFFF000), (u32)(gpu_va >> 32));

	/* === PHASE 3: Halt MEC === */
	pr_info("hp: === HALTING MEC ===\n");
	wr(regCP_MEC_CNTL, MEC_ME1_HALT | MEC_ME2_HALT);
	udelay(100);
	mec_cntl = rr(regCP_MEC_CNTL);
	pr_info("hp: CP_MEC_CNTL after halt = 0x%08X\n", mec_cntl);
	
	if (!(mec_cntl & MEC_ME1_HALT)) {
		pr_err("hp: HALT FAILED — register write blocked (PSP protected?)\n");
		goto out;
	}

	/* === PHASE 4: Invalidate IC + Write IC_BASE for all pipes === */
	pr_info("hp: === WRITING IC_BASE ===\n");
	for (me = 1; me <= 2; me++) {
		for (pipe = 0; pipe < 4; pipe++) {
			select_me_pipe_q(me, pipe, 0, 0);
			
			/* Invalidate IC */
			wr(regCP_CPC_IC_OP_CNTL, 0x01); /* INVALIDATE_ALL */
			udelay(50);
			
			/* Write new IC_BASE */
			wr(regCP_CPC_IC_BASE_LO, (u32)(gpu_va & 0xFFFFF000));
			wr(regCP_CPC_IC_BASE_HI, (u32)(gpu_va >> 32));
			wr(regCP_CPC_IC_BASE_CNTL, 0x10); /* VMID 0, same as original */
			
			/* Prime IC */
			wr(regCP_CPC_IC_OP_CNTL, 0x02); /* PRIME */
			udelay(50);
		}
	}
	wr(regGRBM_GFX_CNTL, orig_grbm);

	/* Verify IC_BASE was written */
	select_me_pipe_q(1, 0, 0, 0);
	pr_info("hp: Verify ME1P0: IC_BASE=0x%08X:%08X CNTL=0x%X OP=0x%X\n",
		rr(regCP_CPC_IC_BASE_HI), rr(regCP_CPC_IC_BASE_LO),
		rr(regCP_CPC_IC_BASE_CNTL), rr(regCP_CPC_IC_OP_CNTL));
	wr(regGRBM_GFX_CNTL, orig_grbm);

	/* === PHASE 5: Restart MEC === */
	pr_info("hp: === RESTARTING MEC ===\n");
	wr(regCP_MEC_CNTL, 0x00000000); /* Clear halt bits */
	udelay(100);
	pr_info("hp: CP_MEC_CNTL after restart = 0x%08X\n", rr(regCP_MEC_CNTL));
	pr_info("hp: MEC1_INSTR_PNTR = 0x%08X\n", rr(regCP_MEC1_INSTR_PNTR));
	pr_info("hp: MEC2_INSTR_PNTR = 0x%08X\n", rr(regCP_MEC2_INSTR_PNTR));

out:
	pr_info("hp: === DONE ===\n");
	pci_iounmap(pdev, vram_ptr);
	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit hotpatch_exit(void) {}
module_init(hotpatch_init);
module_exit(hotpatch_exit);
