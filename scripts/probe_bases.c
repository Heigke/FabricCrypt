/*
 * probe_bases.c — Read-only MEC register snapshot via PCIe indirect access
 *
 * Scans ME/PIPE/QUEUE instances via GRBM_GFX_CNTL banking to find
 * active queues and read banked MEC DC aperture / HQD registers.
 *
 * Module intentionally fails to load (-ENODEV) so it auto-unloads.
 *
 * BUILD: make -C /lib/modules/$(uname -r)/build M=$(pwd)/scripts \
 *          obj-m=probe_bases.o modules
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/spinlock.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002
#define GC_BASE_0  0x1260
#define GC_BASE_1  0xC00000
#define PCIE_INDEX2  0x000e
#define PCIE_DATA2   0x000f

static void __iomem *mmio;
static resource_size_t bar5_size;
static DEFINE_SPINLOCK(indirect_lock);

static u32 rr(u32 off)
{
	if ((u64)off * 4 < bar5_size) {
		return readl(mmio + (u64)off * 4);
	} else {
		u32 r;
		unsigned long flags;
		spin_lock_irqsave(&indirect_lock, flags);
		writel(off * 4, mmio + (u64)PCIE_INDEX2 * 4);
		readl(mmio + (u64)PCIE_INDEX2 * 4);
		r = readl(mmio + (u64)PCIE_DATA2 * 4);
		spin_unlock_irqrestore(&indirect_lock, flags);
		return r;
	}
}

static void wr(u32 off, u32 val)
{
	if ((u64)off * 4 < bar5_size) {
		writel(val, mmio + (u64)off * 4);
	} else {
		unsigned long flags;
		spin_lock_irqsave(&indirect_lock, flags);
		writel(off * 4, mmio + (u64)PCIE_INDEX2 * 4);
		readl(mmio + (u64)PCIE_INDEX2 * 4);
		writel(val, mmio + (u64)PCIE_DATA2 * 4);
		spin_unlock_irqrestore(&indirect_lock, flags);
	}
}

/* GRBM_GFX_CNTL (BASE_IDX=0, offset 0x0013) — instance banking select
 * bits [1:0] = PIPEID, bits [3:2] = MEID, bits [7:4] = VMID, bits [10:8] = QUEUEID */
#define mmGRBM_GFX_CNTL  (GC_BASE_0 + 0x0013)

static void select_me_pipe_q(u32 me, u32 pipe, u32 queue, u32 vmid)
{
	u32 val = (pipe & 3) | ((me & 3) << 2) | ((vmid & 0xF) << 4) | ((queue & 7) << 8);
	wr(mmGRBM_GFX_CNTL, val);
	/* Flush + settle */
	rr(mmGRBM_GFX_CNTL);
}

/* Key registers to read per instance */
#define mmGRBM_STATUS             (GC_BASE_0 + 0x0DA4)
#define mmCP_MEC_CNTL             (GC_BASE_1 + 0x0802)

/* HQD registers (banked by ME/PIPE/QUEUE) */
#define mmCP_HQD_ACTIVE           (GC_BASE_1 + 0x3247)
#define mmCP_HQD_VMID             (GC_BASE_1 + 0x3229)
#define mmCP_HQD_PQ_BASE_LO      (GC_BASE_1 + 0x320A)
#define mmCP_HQD_PQ_BASE_HI      (GC_BASE_1 + 0x3209)
#define mmCP_HQD_PQ_WPTR_LO      (GC_BASE_1 + 0x320C)
#define mmCP_HQD_PQ_CONTROL       (GC_BASE_1 + 0x322B)
#define mmCP_HQD_PQ_RPTR          (GC_BASE_1 + 0x320B)

/* DC Aperture registers (banked) */
#define mmCP_MEC_DC_APERTURE0_BASE  (GC_BASE_1 + 0x2948)
#define mmCP_MEC_DC_APERTURE0_MASK  (GC_BASE_1 + 0x2949)
#define mmCP_MEC_DC_APERTURE0_CNTL  (GC_BASE_1 + 0x294A)
#define mmCP_MEC_DC_APERTURE15_BASE (GC_BASE_1 + 0x2977)
#define mmCP_MEC_DC_APERTURE15_MASK (GC_BASE_1 + 0x2978)
#define mmCP_MEC_DC_APERTURE15_CNTL (GC_BASE_1 + 0x2979)

/* IC control */
#define mmCP_CPC_IC_OP_CNTL       (GC_BASE_1 + 0x297A)
#define mmCP_CPC_IC_BASE_LO       (GC_BASE_1 + 0x584C)
#define mmCP_CPC_IC_BASE_HI       (GC_BASE_1 + 0x584D)
#define mmCP_CPC_IC_BASE_CNTL     (GC_BASE_1 + 0x584E)

/* MEC instruction base */
#define mmCP_MEC_LOCAL_INSTR_BASE_LO  (GC_BASE_1 + 0x292C)
#define mmCP_MEC_LOCAL_INSTR_BASE_HI  (GC_BASE_1 + 0x292D)
#define mmCP_MEC_LOCAL_INSTR_APERTURE (GC_BASE_1 + 0x2930)

/* DC base control + set registers */
#define mmCP_MEC_DC_BASE_CNTL     (GC_BASE_1 + 0x290B)
#define mmCP_MEC_DC_AP_SET_ID     (GC_BASE_1 + 0x2997)
#define mmCP_MEC_DC_AP_SET_MASK   (GC_BASE_1 + 0x2998)

/* PSP debug */
#define mmCPC_PSP_DEBUG           (GC_BASE_1 + 0x5C11)

/* 0xA6 handler register addresses */
#define mmUNK_28EC                (GC_BASE_1 + 0x28EC)

/* Status registers (not banked) */
#define mmCP_CPC_STALLED_STAT1    (GC_BASE_0 + 0x0E26)
#define mmCP_CPF_STATUS           (GC_BASE_0 + 0x0E27)
#define mmCP_CPC_STATUS           (GC_BASE_0 + 0x0E24)

static int __init probe_init(void)
{
	struct pci_dev *pdev = NULL;
	u32 orig_grbm, grbm_status;
	int me, pipe, q;
	int active_count = 0;

	while ((pdev = pci_get_device(AMD_VENDOR_ID, PCI_ANY_ID, pdev))) {
		if ((pdev->class >> 8) == PCI_CLASS_DISPLAY_OTHER)
			break;
	}
	if (!pdev) {
		pr_err("z2355: no AMD GPU found\n");
		return -ENODEV;
	}

	bar5_size = pci_resource_len(pdev, 5);
	mmio = pci_iomap(pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(pdev);
		return -ENOMEM;
	}

	grbm_status = rr(mmGRBM_STATUS);
	pr_info("z2355: === MEC REGISTER SNAPSHOT v2 (banked) ===\n");
	pr_info("z2355: GRBM_STATUS=0x%08X BAR5=%llu\n", grbm_status, (u64)bar5_size);

	/* Save original GRBM_GFX_CNTL */
	orig_grbm = rr(mmGRBM_GFX_CNTL);
	pr_info("z2355: orig GRBM_GFX_CNTL=0x%08X\n", orig_grbm);

	/* Global (non-banked) registers */
	pr_info("z2355: --- GLOBAL (not banked) ---\n");
	pr_info("z2355: CP_MEC_CNTL         = 0x%08X\n", rr(mmCP_MEC_CNTL));
	pr_info("z2355: CP_CPC_STALLED      = 0x%08X\n", rr(mmCP_CPC_STALLED_STAT1));
	pr_info("z2355: CP_CPF_STATUS       = 0x%08X\n", rr(mmCP_CPF_STATUS));
	pr_info("z2355: CP_CPC_STATUS       = 0x%08X\n", rr(mmCP_CPC_STATUS));
	pr_info("z2355: CPC_PSP_DEBUG       = 0x%08X\n", rr(mmCPC_PSP_DEBUG));

	/* Scan ME 1-2, PIPE 0-3, QUEUE 0-7 for active HQDs */
	pr_info("z2355: --- SCANNING ME/PIPE/QUEUE ---\n");
	for (me = 1; me <= 2; me++) {
		for (pipe = 0; pipe < 4; pipe++) {
			for (q = 0; q < 8; q++) {
				u32 hqd_active, hqd_vmid, pq_base_lo, pq_base_hi, pq_ctrl;

				select_me_pipe_q(me, pipe, q, 0);

				hqd_active = rr(mmCP_HQD_ACTIVE);
				if (!hqd_active)
					continue;

				active_count++;
				hqd_vmid = rr(mmCP_HQD_VMID);
				pq_base_lo = rr(mmCP_HQD_PQ_BASE_LO);
				pq_base_hi = rr(mmCP_HQD_PQ_BASE_HI);
				pq_ctrl = rr(mmCP_HQD_PQ_CONTROL);

				pr_info("z2355: ACTIVE ME%d P%d Q%d: VMID=%d BASE=0x%08X:%08X CTRL=0x%08X\n",
					me, pipe, q, hqd_vmid,
					pq_base_hi, pq_base_lo, pq_ctrl);

				/* Read banked DC aperture for this instance */
				{
					u32 dc0_base, dc0_mask, dc0_cntl;
					u32 dc15_base, dc15_mask, dc15_cntl;
					u32 ic_op, ic_lo, ic_hi, ic_cntl;
					u32 instr_lo, instr_hi, instr_ap;
					u32 dc_base_cntl, set_id, set_mask;
					u32 unk_28ec;
					int ap;

					dc0_base = rr(mmCP_MEC_DC_APERTURE0_BASE);
					dc0_mask = rr(mmCP_MEC_DC_APERTURE0_MASK);
					dc0_cntl = rr(mmCP_MEC_DC_APERTURE0_CNTL);
					dc15_base = rr(mmCP_MEC_DC_APERTURE15_BASE);
					dc15_mask = rr(mmCP_MEC_DC_APERTURE15_MASK);
					dc15_cntl = rr(mmCP_MEC_DC_APERTURE15_CNTL);

					pr_info("z2355:   DC_AP0:  base=0x%08X mask=0x%08X cntl=0x%08X\n",
						dc0_base, dc0_mask, dc0_cntl);
					pr_info("z2355:   DC_AP15: base=0x%08X mask=0x%08X cntl=0x%08X\n",
						dc15_base, dc15_mask, dc15_cntl);

					/* Scan all 16 apertures for non-zero */
					for (ap = 1; ap < 15; ap++) {
						u32 b = rr(GC_BASE_1 + 0x2948 + ap * 3);
						u32 m = rr(GC_BASE_1 + 0x2949 + ap * 3);
						u32 c = rr(GC_BASE_1 + 0x294A + ap * 3);
						if (b || m || c) {
							pr_info("z2355:   DC_AP%d: base=0x%08X mask=0x%08X cntl=0x%08X\n",
								ap, b, m, c);
						}
					}

					ic_op = rr(mmCP_CPC_IC_OP_CNTL);
					ic_lo = rr(mmCP_CPC_IC_BASE_LO);
					ic_hi = rr(mmCP_CPC_IC_BASE_HI);
					ic_cntl = rr(mmCP_CPC_IC_BASE_CNTL);
					pr_info("z2355:   IC: op=0x%08X base=0x%08X:%08X cntl=0x%08X\n",
						ic_op, ic_hi, ic_lo, ic_cntl);

					instr_lo = rr(mmCP_MEC_LOCAL_INSTR_BASE_LO);
					instr_hi = rr(mmCP_MEC_LOCAL_INSTR_BASE_HI);
					instr_ap = rr(mmCP_MEC_LOCAL_INSTR_APERTURE);
					pr_info("z2355:   INSTR_BASE: 0x%08X:%08X aperture=0x%08X\n",
						instr_hi, instr_lo, instr_ap);

					dc_base_cntl = rr(mmCP_MEC_DC_BASE_CNTL);
					set_id = rr(mmCP_MEC_DC_AP_SET_ID);
					set_mask = rr(mmCP_MEC_DC_AP_SET_MASK);
					pr_info("z2355:   DC_BASE_CNTL=0x%08X SET_ID=0x%08X SET_MASK=0x%08X\n",
						dc_base_cntl, set_id, set_mask);

					unk_28ec = rr(mmUNK_28EC);
					pr_info("z2355:   UNK_28EC=0x%08X\n", unk_28ec);

					/* WPTR for queue activity tracking */
					pr_info("z2355:   PQ_WPTR=0x%08X PQ_RPTR=0x%08X\n",
						rr(mmCP_HQD_PQ_WPTR_LO), rr(mmCP_HQD_PQ_RPTR));
				}
			}
		}
	}

	/* Also check ME0 (GFX/PFP/ME) pipe 0 */
	select_me_pipe_q(0, 0, 0, 0);
	{
		u32 hqd_active = rr(mmCP_HQD_ACTIVE);
		if (hqd_active) {
			pr_info("z2355: ME0 P0 Q0: ACTIVE=%d VMID=%d BASE=0x%08X:%08X\n",
				hqd_active, rr(mmCP_HQD_VMID),
				rr(mmCP_HQD_PQ_BASE_HI), rr(mmCP_HQD_PQ_BASE_LO));
		}
	}

	pr_info("z2355: Found %d active queues\n", active_count);

	/* Restore original banking */
	wr(mmGRBM_GFX_CNTL, orig_grbm);

	pr_info("z2355: === SNAPSHOT v2 COMPLETE ===\n");

	pci_iounmap(pdev, mmio);
	pci_dev_put(pdev);
	return -ENODEV;
}

static void __exit probe_exit(void) {}
module_init(probe_init);
module_exit(probe_exit);
