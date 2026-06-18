/*
 * patch_mec_fw8.c — Phase 8: Read IC_BASE, find GART physical address,
 *                   patch at the correct location
 *
 * Discovery: The MEC fetches instructions from GPU VA stored in
 * CP_CPC_IC_BASE_LO/HI (0xF84C/0xF84D). Our previous patches went
 * to physical addresses found by pattern scanning system RAM, but
 * the GPU fetches through GART translation. We need to:
 *   1. Read IC_BASE to get GPU VA
 *   2. Find what physical RAM the GART VA maps to
 *   3. Patch at THAT physical address
 *   4. Invalidate IC and verify
 *
 * Auto-unloads (-ENODEV).
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/mm.h>
#include <linux/highmem.h>
#include <linux/delay.h>

MODULE_LICENSE("GPL");

#define AMD_VENDOR_ID 0x1002

/* Legacy registers */
#define regCP_MEC_CNTL              0x0A802
#define regCP_MEC1_INSTR_PNTR      0x021A8
#define regCP_CPC_IC_OP_CNTL       0x0C97A
#define regCP_MEC_DC_OP_CNTL       0x0C90C

/* GRBM */
#define regGRBM_GFX_CNTL           0x0A900

/* IC base registers (per-pipe, need GRBM context) */
#define regCP_CPC_IC_BASE_LO       0x0F84C
#define regCP_CPC_IC_BASE_HI       0x0F84D
#define regCP_CPC_IC_BASE_CNTL     0x0F84E

/* MD (data) base registers (per-pipe, need GRBM context) */
#define regCP_MEC_MDBASE_LO        0x0F870
#define regCP_MEC_MDBASE_HI        0x0F871

/* RS64 PC */
#define regCP_MEC_RS64_CNTL             0x0C904
#define regCP_MEC_RS64_INSTR_PNTR       0x0C908
#define regCP_MEC_RS64_PRGRM_CNTR_START 0x0C900

/* Legacy MEC_CNTL bits */
#define MEC_ME1_PIPE0_RESET   (1 << 16)
#define MEC_ME1_PIPE1_RESET   (1 << 17)
#define MEC_ME1_PIPE2_RESET   (1 << 18)
#define MEC_ME1_PIPE3_RESET   (1 << 19)
#define MEC_INVALIDATE_ICACHE (1 << 27)
#define MEC_ME1_HALT          (1 << 30)
#define ALL_PIPE_RESET (MEC_ME1_PIPE0_RESET | MEC_ME1_PIPE1_RESET | \
			MEC_ME1_PIPE2_RESET | MEC_ME1_PIPE3_RESET)

static void __iomem *mmio;
static struct pci_dev *gpu_pdev;

static u32 rr(u32 dw_off) { return readl(mmio + (u64)dw_off * 4); }
static void wr(u32 dw_off, u32 val) {
	writel(val, mmio + (u64)dw_off * 4);
	readl(mmio + (u64)dw_off * 4);
}

/*
 * GRBM context selection for SOC21 (GFX11)
 * soc21_grbm_select(adev, me, pipe, queue, vmid)
 * ME=1 for MEC1, PIPE=0..3, QUEUE=0, VMID=0
 *
 * GRBM_GFX_CNTL bit layout (GFX11):
 *   [3:0]   = PIPEID
 *   [5:4]   = MEID
 *   [10:8]  = QUEUEID
 *   [15:12] = VMID
 */
static void grbm_select(u32 me, u32 pipe, u32 queue, u32 vmid)
{
	u32 val = (pipe & 0xF) | ((me & 0x3) << 4) |
		  ((queue & 0x7) << 8) | ((vmid & 0xF) << 12);
	wr(regGRBM_GFX_CNTL, val);
	udelay(50);
}

static void grbm_clear(void)
{
	wr(regGRBM_GFX_CNTL, 0);
	udelay(50);
}

/*
 * Map a physical address and read/write firmware dwords.
 * This works for system RAM backed by struct page.
 */
static u32 *map_phys(u64 phys)
{
	unsigned long pfn = phys >> PAGE_SHIFT;
	unsigned int page_off = phys & ~PAGE_MASK;
	struct page *page;
	u8 *vaddr;

	if (!pfn_valid(pfn))
		return NULL;
	page = pfn_to_page(pfn);
	vaddr = kmap_local_page(page);
	if (!vaddr)
		return NULL;
	return (u32 *)(vaddr + page_off);
}

static void unmap_phys(void *ptr)
{
	kunmap_local((void *)((unsigned long)ptr & PAGE_MASK));
}

/*
 * Try to find the physical address backing a GPU GART VA.
 * Strategy: the GART table maps GPU VAs to physical pages.
 * For AMDGPU, the GART base is at BAR0 or in system RAM.
 * We can try to find the firmware by scanning the VRAM BAR
 * or by using the amdgpu GART page table.
 *
 * Simpler approach: scan ALL of system RAM for the firmware
 * pattern at the offset corresponding to the IC_BASE alignment.
 */

/* Known firmware copy in system RAM (from find_mec_fw Phase 1) */
#define FW_PHYS_COPY0  0x115c8c100ULL

static int __init fw8_init(void)
{
	u64 ic_base_va, md_base_va;
	u64 vram_start, vram_size;
	void __iomem *vram = NULL;
	int i;

	while ((gpu_pdev = pci_get_device(AMD_VENDOR_ID, PCI_ANY_ID, gpu_pdev))) {
		if ((gpu_pdev->class >> 8) == PCI_CLASS_DISPLAY_OTHER)
			break;
	}
	if (!gpu_pdev) {
		pr_err("fw8: no AMD GPU\n");
		return -ENODEV;
	}
	mmio = pci_iomap(gpu_pdev, 5, 0);
	if (!mmio) {
		pci_dev_put(gpu_pdev);
		return -ENOMEM;
	}

	pr_info("fw8: ========================================\n");
	pr_info("fw8: PHASE 8: IC_BASE DISCOVERY + GART PATCH\n");
	pr_info("fw8: ========================================\n");

	pr_info("fw8: BASELINE: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));

	/* ============================================================
	 * PROBE 1: Read IC_BASE and MDBASE for each pipe
	 * ============================================================ */
	pr_info("fw8: --- PROBE: IC_BASE per pipe ---\n");
	for (i = 0; i < 4; i++) {
		u32 ic_lo, ic_hi, ic_cntl, md_lo, md_hi;
		u32 rs64_cntl, rs64_pc, rs64_start;

		grbm_select(1, i, 0, 0); /* ME=1 (MEC1), PIPE=i */

		ic_lo = rr(regCP_CPC_IC_BASE_LO);
		ic_hi = rr(regCP_CPC_IC_BASE_HI);
		ic_cntl = rr(regCP_CPC_IC_BASE_CNTL);
		md_lo = rr(regCP_MEC_MDBASE_LO);
		md_hi = rr(regCP_MEC_MDBASE_HI);
		rs64_cntl = rr(regCP_MEC_RS64_CNTL);
		rs64_pc = rr(regCP_MEC_RS64_INSTR_PNTR);
		rs64_start = rr(regCP_MEC_RS64_PRGRM_CNTR_START);

		pr_info("fw8: PIPE[%d] IC_BASE=0x%08X_%08X CNTL=0x%08X\n",
			i, ic_hi, ic_lo, ic_cntl);
		pr_info("fw8: PIPE[%d] MDBASE=0x%08X_%08X\n",
			i, md_hi, md_lo);
		pr_info("fw8: PIPE[%d] RS64: CNTL=0x%08X PC=0x%04X START=0x%04X\n",
			i, rs64_cntl, rs64_pc, rs64_start);
	}
	grbm_clear();

	/* Read IC base for pipe 0 to use */
	grbm_select(1, 0, 0, 0);
	ic_base_va = ((u64)rr(regCP_CPC_IC_BASE_HI) << 32) | rr(regCP_CPC_IC_BASE_LO);
	md_base_va = ((u64)rr(regCP_MEC_MDBASE_HI) << 32) | rr(regCP_MEC_MDBASE_LO);
	grbm_clear();

	pr_info("fw8: IC_BASE GPU VA = 0x%016llX\n", ic_base_va);
	pr_info("fw8: MDBASE GPU VA  = 0x%016llX\n", md_base_va);
	pr_info("fw8: FW_PHYS[0]     = 0x%09llX\n", FW_PHYS_COPY0);

	/* ============================================================
	 * PROBE 2: Check VRAM BAR — firmware might be in VRAM
	 * ============================================================ */
	vram_start = pci_resource_start(gpu_pdev, 0);
	vram_size = pci_resource_len(gpu_pdev, 0);
	pr_info("fw8: VRAM BAR0: start=0x%llX size=0x%llX (%llu MB)\n",
		vram_start, vram_size, vram_size >> 20);

	/* Map first 4MB of VRAM to search for firmware signature */
	if (vram_size >= (4 << 20)) {
		u32 vram_search_size = 4 << 20; /* 4MB */
		vram = ioremap_wc(vram_start, vram_search_size);
		if (vram) {
			u32 offset;
			int found = 0;
			/* Search for the idle loop pattern: 0x88000000 at dword 0x44C */
			/* The firmware pattern at [0x44A..0x44C] is:
			 * 0x88000000 0x7C408001 0x88000000 */
			for (offset = 0; offset < vram_search_size - 0x2000; offset += 0x1000) {
				u32 v44a = readl(vram + offset + 0x44A * 4);
				u32 v44b = readl(vram + offset + 0x44B * 4);
				u32 v44c = readl(vram + offset + 0x44C * 4);
				if (v44a == 0x88000000 && v44b == 0x7C408001 && v44c == 0x88000000) {
					pr_info("fw8: VRAM MATCH at BAR0+0x%X (phys=0x%llX)\n",
						offset, vram_start + offset);
					found++;
					if (found >= 3) break;
				}
			}
			if (!found)
				pr_info("fw8: No firmware pattern in first 4MB of VRAM\n");
			iounmap(vram);
			vram = NULL;
		} else {
			pr_info("fw8: Could not map VRAM BAR0\n");
		}
	}

	/* ============================================================
	 * PROBE 3: Try to map the IC_BASE GPU VA to physical address
	 * On AMD GPUs, GART typically maps GPU VA = phys addr for
	 * system RAM (identity mapping in GTT domain).
	 * Check if IC_BASE VA is in system RAM range.
	 * ============================================================ */
	{
		/* GPU GART entries are typically at MC_VM_FB_OFFSET or similar.
		 * For GTT (system RAM), the GPU VA often has a high bit pattern.
		 * Let's check if IC_BASE points to something reasonable. */
		u64 ic_offset_44c;
		u32 test_val;

		pr_info("fw8: IC_BASE VA=0x%016llX\n", ic_base_va);

		/* The IC_BASE is the start of the firmware code buffer.
		 * PC=0x44C means the instruction is at IC_BASE + 0x44C * 4 bytes.
		 * But wait — the PC might be byte-addressed, not dword-addressed.
		 * RS64 might use byte offsets. Let me check both. */

		/* If dword-addressed: offset = 0x44C * 4 = 0x1130 bytes */
		ic_offset_44c = 0x44C * 4;
		pr_info("fw8: If PC is dword-addr: FW idle at GPU VA 0x%016llX\n",
			ic_base_va + ic_offset_44c);

		/* If byte-addressed: offset = 0x44C bytes */
		pr_info("fw8: If PC is byte-addr:  FW idle at GPU VA 0x%016llX\n",
			ic_base_va + 0x44C);

		/* Read from our known physical copy to verify it's still there */
		{
			u32 *ptr = map_phys(FW_PHYS_COPY0 + 0x44C * 4);
			if (ptr) {
				test_val = *ptr;
				unmap_phys(ptr);
				pr_info("fw8: FW_PHYS[0][0x44C] = 0x%08X %s\n",
					test_val,
					test_val == 0x88000000 ? "(idle loop OK)" : "(UNEXPECTED)");
			}
		}
	}

	/* ============================================================
	 * PROBE 4: Try to access firmware through VRAM BAR aperture
	 * On dGPU, GART pages may be accessible via BAR0 at certain offsets.
	 * IC_BASE VA minus the GART start VA gives the VRAM-relative offset.
	 *
	 * Also try: if IC_BASE is a GPU VA, the bottom 40-48 bits might
	 * be the physical address (identity GART mapping).
	 * ============================================================ */
	{
		/* Check if IC_BASE lower bits match our physical address */
		u64 ic_masked = ic_base_va & 0xFFFFFFFFFFULL; /* 40-bit mask */
		pr_info("fw8: IC_BASE & 0xFFFFFFFFFF = 0x%010llX\n", ic_masked);
		pr_info("fw8: FW_PHYS[0]             = 0x%010llX\n",
			FW_PHYS_COPY0 & 0xFFFFFFFFFFULL);

		/* Check if the difference is a GART base offset */
		if (ic_masked > FW_PHYS_COPY0)
			pr_info("fw8: IC_BASE - FW_PHYS = +0x%llX\n",
				ic_masked - FW_PHYS_COPY0);
		else
			pr_info("fw8: FW_PHYS - IC_BASE = +0x%llX\n",
				FW_PHYS_COPY0 - ic_masked);
	}

	/* ============================================================
	 * TEST A: Patch known physical copy + full invalidation
	 * Even if we don't know the exact GART mapping, the driver
	 * allocated via AMDGPU_GEM_DOMAIN_VRAM | AMDGPU_GEM_DOMAIN_GTT
	 * which means it could be in system RAM (GTT) or VRAM.
	 *
	 * If it's in system RAM (GTT domain), the physical address
	 * should be findable. Let's try patching at FW_PHYS[0] with
	 * proper GART awareness.
	 *
	 * Key insight from driver: after patching, the driver does:
	 *   1. DC invalidate + poll COMPLETE
	 *   2. CPC IC invalidate + poll COMPLETE
	 * We should do the SAME sequence.
	 * ============================================================ */
	pr_info("fw8: --- TEST A: Proper driver-style invalidation ---\n");
	{
		u32 orig_val;
		u32 *ptr;
		u32 tmp;
		int timeout;

		/* Read original */
		ptr = map_phys(FW_PHYS_COPY0 + 0x44C * 4);
		if (!ptr) {
			pr_err("fw8: Cannot map FW_PHYS\n");
			goto done;
		}
		orig_val = *ptr;
		pr_info("fw8: A orig FW[0x44C] = 0x%08X\n", orig_val);

		/* Halt MEC */
		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);

		/* Patch */
		*ptr = 0xBF80DEAD;
		wmb();
		clflush(ptr);
		wmb();
		unmap_phys(ptr);

		/* Verify */
		ptr = map_phys(FW_PHYS_COPY0 + 0x44C * 4);
		if (ptr) {
			pr_info("fw8: A readback = 0x%08X\n", *ptr);
			unmap_phys(ptr);
		}

		/* EXACT driver sequence: DC invalidate with polling */
		tmp = rr(regCP_MEC_DC_OP_CNTL);
		tmp |= 0x1; /* INVALIDATE_DCACHE */
		wr(regCP_MEC_DC_OP_CNTL, tmp);
		timeout = 50000;
		while (timeout > 0) {
			tmp = rr(regCP_MEC_DC_OP_CNTL);
			if (tmp & 0x2) /* INVALIDATE_DCACHE_COMPLETE */
				break;
			udelay(1);
			timeout--;
		}
		pr_info("fw8: A DC invalidate %s (0x%08X)\n",
			timeout > 0 ? "COMPLETE" : "TIMEOUT", rr(regCP_MEC_DC_OP_CNTL));

		/* EXACT driver sequence: CPC IC invalidate with polling */
		tmp = rr(regCP_CPC_IC_OP_CNTL);
		tmp |= 0x1; /* INVALIDATE_CACHE */
		wr(regCP_CPC_IC_OP_CNTL, tmp);
		timeout = 50000;
		while (timeout > 0) {
			tmp = rr(regCP_CPC_IC_OP_CNTL);
			if (tmp & 0x2) /* INVALIDATE_CACHE_COMPLETE */
				break;
			udelay(1);
			timeout--;
		}
		pr_info("fw8: A CPC IC invalidate %s (0x%08X)\n",
			timeout > 0 ? "COMPLETE" : "TIMEOUT", rr(regCP_CPC_IC_OP_CNTL));

		/* Pipe reset */
		wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
		udelay(2000);
		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);
		pr_info("fw8: A after reset: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));

		/* Unhalt */
		wr(regCP_MEC_CNTL, 0);
		mdelay(30);
		{
			int j;
			for (j = 0; j < 6; j++) {
				mdelay(5);
				pr_info("fw8: A PC[%d]=0x%04X\n",
					j, rr(regCP_MEC1_INSTR_PNTR));
			}
		}

		/* Restore */
		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);
		ptr = map_phys(FW_PHYS_COPY0 + 0x44C * 4);
		if (ptr) {
			*ptr = orig_val;
			wmb();
			clflush(ptr);
			wmb();
			unmap_phys(ptr);
		}
		wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
		udelay(2000);
		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);
		wr(regCP_MEC_CNTL, 0);
		mdelay(50);
		pr_info("fw8: A RESTORED: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	}

	/* ============================================================
	 * TEST B: Write firmware through VRAM BAR
	 * Map the full VRAM BAR and search for the firmware pattern.
	 * If found, patch through the VRAM BAR (GPU's view of memory)
	 * rather than through system RAM (CPU's view).
	 * ============================================================ */
	pr_info("fw8: --- TEST B: VRAM BAR firmware search + patch ---\n");
	{
		u32 search_size = min_t(u64, vram_size, 64ULL << 20); /* 64MB max */
		void __iomem *vbar = ioremap_wc(vram_start, search_size);
		u64 fw_vram_offset = 0;
		int found = 0;

		if (!vbar) {
			pr_info("fw8: B Cannot map VRAM BAR\n");
			goto test_c;
		}

		/* Search for firmware idle loop pattern in VRAM */
		pr_info("fw8: B Searching %u MB of VRAM for firmware...\n",
			search_size >> 20);
		{
			u32 offset;
			for (offset = 0; offset < search_size - 0x2000; offset += 0x100) {
				u32 v44a = readl(vbar + offset + 0x44A * 4);
				u32 v44b = readl(vbar + offset + 0x44B * 4);
				u32 v44c = readl(vbar + offset + 0x44C * 4);
				if (v44a == 0x88000000 && v44b == 0x7C408001 &&
				    v44c == 0x88000000) {
					u32 v449 = readl(vbar + offset + 0x449 * 4);
					pr_info("fw8: B VRAM MATCH at +0x%06X "
						"[0x449]=0x%08X [0x44A]=0x%08X "
						"[0x44C]=0x%08X\n",
						offset, v449, v44a, v44c);
					if (!found)
						fw_vram_offset = offset;
					found++;
					if (found >= 5) break;
				}
			}
		}

		if (found) {
			u32 orig_val, readback;
			pr_info("fw8: B Found %d VRAM copy(ies). Patching first at +0x%llX\n",
				found, fw_vram_offset);

			/* Read original through VRAM BAR */
			orig_val = readl(vbar + fw_vram_offset + 0x44C * 4);
			pr_info("fw8: B VRAM orig FW[0x44C] = 0x%08X\n", orig_val);

			/* Halt */
			wr(regCP_MEC_CNTL, MEC_ME1_HALT);
			udelay(500);

			/* Patch through VRAM BAR */
			writel(0xBF80DEAD, vbar + fw_vram_offset + 0x44C * 4);
			wmb();
			readback = readl(vbar + fw_vram_offset + 0x44C * 4);
			pr_info("fw8: B VRAM readback = 0x%08X %s\n",
				readback, readback == 0xBF80DEAD ? "OK" : "FAIL");

			/* Driver-style DC + IC invalidation */
			{
				u32 tmp;
				int timeout;

				tmp = rr(regCP_MEC_DC_OP_CNTL);
				wr(regCP_MEC_DC_OP_CNTL, tmp | 0x1);
				timeout = 50000;
				while (timeout-- > 0) {
					if (rr(regCP_MEC_DC_OP_CNTL) & 0x2)
						break;
					udelay(1);
				}

				tmp = rr(regCP_CPC_IC_OP_CNTL);
				wr(regCP_CPC_IC_OP_CNTL, tmp | 0x1);
				timeout = 50000;
				while (timeout-- > 0) {
					if (rr(regCP_CPC_IC_OP_CNTL) & 0x2)
						break;
					udelay(1);
				}
			}

			/* Pipe reset */
			wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
			udelay(2000);
			wr(regCP_MEC_CNTL, MEC_ME1_HALT);
			udelay(500);

			/* Unhalt */
			wr(regCP_MEC_CNTL, 0);
			mdelay(30);
			{
				int j;
				for (j = 0; j < 6; j++) {
					mdelay(5);
					pr_info("fw8: B PC[%d]=0x%04X\n",
						j, rr(regCP_MEC1_INSTR_PNTR));
				}
			}

			/* Restore */
			wr(regCP_MEC_CNTL, MEC_ME1_HALT);
			udelay(500);
			writel(orig_val, vbar + fw_vram_offset + 0x44C * 4);
			wmb();
			wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
			udelay(2000);
			wr(regCP_MEC_CNTL, MEC_ME1_HALT);
			udelay(500);
			wr(regCP_MEC_CNTL, 0);
			mdelay(50);
			pr_info("fw8: B RESTORED: PC=0x%04X\n",
				rr(regCP_MEC1_INSTR_PNTR));
		} else {
			pr_info("fw8: B No firmware found in VRAM\n");
		}

		iounmap(vbar);
	}

test_c:
	/* ============================================================
	 * TEST C: Reprogram IC_BASE to point to our system RAM copy
	 * Instead of finding where IC_BASE points, CHANGE IC_BASE
	 * to point to FW_PHYS[0]. Then invalidate and reset.
	 * This bypasses the GART entirely.
	 * ============================================================ */
	pr_info("fw8: --- TEST C: Reprogram IC_BASE to our phys copy ---\n");
	{
		u32 ic_lo_orig, ic_hi_orig;
		u64 our_base;
		u32 orig_val;
		u32 *ptr;

		/* Save original IC_BASE */
		grbm_select(1, 0, 0, 0);
		ic_lo_orig = rr(regCP_CPC_IC_BASE_LO);
		ic_hi_orig = rr(regCP_CPC_IC_BASE_HI);
		grbm_clear();
		pr_info("fw8: C orig IC_BASE = 0x%08X_%08X\n", ic_hi_orig, ic_lo_orig);

		/* Our firmware is at FW_PHYS_COPY0 (physical address).
		 * But IC_BASE expects a GPU VA, not physical address.
		 * On GPUs with GART, we'd need a GART mapping.
		 *
		 * However, for identity-mapped GART (common in some configs),
		 * phys == GPU VA in the GART region.
		 *
		 * Try it — worst case MEC crashes and we restore IC_BASE. */
		our_base = FW_PHYS_COPY0;
		pr_info("fw8: C setting IC_BASE to 0x%09llX (our phys addr)\n", our_base);

		/* Halt */
		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);

		/* Patch firmware */
		ptr = map_phys(FW_PHYS_COPY0 + 0x44C * 4);
		if (ptr) {
			orig_val = *ptr;
			*ptr = 0xBF80DEAD;
			wmb();
			clflush(ptr);
			wmb();
			unmap_phys(ptr);
		} else {
			pr_info("fw8: C cannot map FW\n");
			goto done;
		}

		/* Reprogram IC_BASE for all pipes */
		{
			int p;
			for (p = 0; p < 4; p++) {
				grbm_select(1, p, 0, 0);
				wr(regCP_CPC_IC_BASE_LO, (u32)(our_base & 0xFFFFF000ULL));
				wr(regCP_CPC_IC_BASE_HI, (u32)(our_base >> 32));
			}
			grbm_clear();
		}

		/* Verify IC_BASE change */
		grbm_select(1, 0, 0, 0);
		pr_info("fw8: C new IC_BASE = 0x%08X_%08X\n",
			rr(regCP_CPC_IC_BASE_HI), rr(regCP_CPC_IC_BASE_LO));
		grbm_clear();

		/* DC + IC invalidation */
		{
			u32 tmp;
			int timeout;

			tmp = rr(regCP_MEC_DC_OP_CNTL);
			wr(regCP_MEC_DC_OP_CNTL, tmp | 0x1);
			timeout = 50000;
			while (timeout-- > 0) {
				if (rr(regCP_MEC_DC_OP_CNTL) & 0x2) break;
				udelay(1);
			}
			pr_info("fw8: C DC invalidate %s\n",
				timeout > 0 ? "OK" : "TIMEOUT");

			tmp = rr(regCP_CPC_IC_OP_CNTL);
			wr(regCP_CPC_IC_OP_CNTL, tmp | 0x1);
			timeout = 50000;
			while (timeout-- > 0) {
				if (rr(regCP_CPC_IC_OP_CNTL) & 0x2) break;
				udelay(1);
			}
			pr_info("fw8: C IC invalidate %s\n",
				timeout > 0 ? "OK" : "TIMEOUT");
		}

		/* Pipe reset */
		wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
		udelay(2000);
		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);

		/* Unhalt */
		wr(regCP_MEC_CNTL, 0);
		mdelay(30);
		{
			int j;
			for (j = 0; j < 8; j++) {
				mdelay(5);
				pr_info("fw8: C PC[%d]=0x%04X\n",
					j, rr(regCP_MEC1_INSTR_PNTR));
			}
		}

		/* RESTORE: IC_BASE + firmware */
		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);

		ptr = map_phys(FW_PHYS_COPY0 + 0x44C * 4);
		if (ptr) {
			*ptr = orig_val;
			wmb();
			clflush(ptr);
			wmb();
			unmap_phys(ptr);
		}

		/* Restore IC_BASE */
		{
			int p;
			for (p = 0; p < 4; p++) {
				grbm_select(1, p, 0, 0);
				wr(regCP_CPC_IC_BASE_LO, ic_lo_orig);
				wr(regCP_CPC_IC_BASE_HI, ic_hi_orig);
			}
			grbm_clear();
		}

		/* Invalidate + reset */
		{
			u32 tmp;
			int timeout;
			tmp = rr(regCP_MEC_DC_OP_CNTL);
			wr(regCP_MEC_DC_OP_CNTL, tmp | 0x1);
			timeout = 50000;
			while (timeout-- > 0) {
				if (rr(regCP_MEC_DC_OP_CNTL) & 0x2) break;
				udelay(1);
			}
			tmp = rr(regCP_CPC_IC_OP_CNTL);
			wr(regCP_CPC_IC_OP_CNTL, tmp | 0x1);
			timeout = 50000;
			while (timeout-- > 0) {
				if (rr(regCP_CPC_IC_OP_CNTL) & 0x2) break;
				udelay(1);
			}
		}
		wr(regCP_MEC_CNTL, MEC_ME1_HALT | ALL_PIPE_RESET | MEC_INVALIDATE_ICACHE);
		udelay(2000);
		wr(regCP_MEC_CNTL, MEC_ME1_HALT);
		udelay(500);
		wr(regCP_MEC_CNTL, 0);
		mdelay(50);

		/* Verify restore */
		grbm_select(1, 0, 0, 0);
		pr_info("fw8: C restored IC_BASE = 0x%08X_%08X\n",
			rr(regCP_CPC_IC_BASE_HI), rr(regCP_CPC_IC_BASE_LO));
		grbm_clear();
		pr_info("fw8: C RESTORED: PC=0x%04X\n", rr(regCP_MEC1_INSTR_PNTR));
	}

done:
	pr_info("fw8: FINAL: MEC_CNTL=0x%08X PC=0x%04X\n",
		rr(regCP_MEC_CNTL), rr(regCP_MEC1_INSTR_PNTR));
	pr_info("fw8: ========================================\n");
	pr_info("fw8: PHASE 8 COMPLETE\n");
	pr_info("fw8: ========================================\n");

	pci_iounmap(gpu_pdev, mmio);
	pci_dev_put(gpu_pdev);
	return -ENODEV;
}

static void __exit fw8_exit(void) {}
module_init(fw8_init);
module_exit(fw8_exit);
