#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
#include <linux/slab.h>
MODULE_LICENSE("GPL");

struct psp_cmd { u32 buf_size, buf_version, cmd_id, r1,r2,r3,r4; u32 data[32]; u32 resp[16]; };
typedef int (*psp_fn)(void*, void*, struct psp_cmd*, u64);
typedef u32 (*rreg_fn)(void*, u32);
typedef void (*wreg_fn)(void*, u32, u32);

static int __init mcl_init(void) {
    struct pci_dev *p = pci_get_device(0x1002,0x1586,NULL);
    void *a, *psp; psp_fn submit; rreg_fn rr; wreg_fn wr;
    struct psp_cmd *cmd; u64 fence_mc, fw_pri_mc;
    void *fw_pri; u32 gc=0x2800; int ret;
    void *fw_buf = NULL; ssize_t fw_total;

    if (!p) return -ENODEV;
    a=(u8*)pci_get_drvdata(p)-0x10;
    psp=(u8*)a+0x3B910;
    fw_pri=(void*)*(u64*)((u8*)psp+0x058);
    fw_pri_mc=*(u64*)((u8*)psp+0x050);
    fence_mc=*(u64*)((u8*)psp+0x1B8);
    submit=(psp_fn)0xFFFFFFFFC0F55840ULL;
    rr=(rreg_fn)0xFFFFFFFFC0E28460ULL;
    wr=(wreg_fn)0xFFFFFFFFC0E28820ULL;
    cmd=kzalloc(sizeof(*cmd),GFP_KERNEL);
    if (!cmd) { pci_dev_put(p); return -ENOMEM; }

    pr_info("mcl: === CORRECT MES FIRMWARE LOAD ===\n");

    /* MES state before */
    wr(a, 0x1900, (3<<2)); udelay(200);
    pr_info("mcl: BEFORE: PC=0x%04X CNTL=0x%X\n",
        rr(a,gc+0x2813), rr(a,gc+0x2807));
    wr(a, 0x1900, 0);

    /* Load MES pipe0 blob */
    {
        struct file *fp = filp_open("/tmp/mes_re/mes_pipe0.bin", O_RDONLY, 0);
        loff_t pos = 0;
        if (IS_ERR(fp)) { pr_info("mcl: no fw\n"); goto done; }
        fw_buf = kmalloc(0xA0000, GFP_KERNEL);
        if (!fw_buf) { filp_close(fp, NULL); goto done; }
        fw_total = kernel_read(fp, fw_buf, 0xA0000, &pos);
        filp_close(fp, NULL);

        /* Parse header correctly:
         * common_firmware_header:
         *   [0] size_bytes
         *   [1] header_size_dw
         *   [2] header_version
         *   [3] ip_version
         *   [4] ucode_version
         *   [5] ucode_size_bytes
         *   [6] ucode_array_offset_bytes */
        {
            u32 *h = (u32*)fw_buf;
            u32 ucode_off = h[6];  /* offset to ucode array */
            u32 ucode_size = h[5]; /* ucode size */
            u32 *ucode;

            pr_info("mcl: total=%zd ucode_off=0x%X ucode_size=0x%X (%u KB)\n",
                fw_total, ucode_off, ucode_size, ucode_size >> 10);

            if (ucode_off >= fw_total || ucode_off + ucode_size > fw_total) {
                pr_info("mcl: Invalid ucode offset/size\n");
                goto done;
            }

            ucode = (u32*)((u8*)fw_buf + ucode_off);
            pr_info("mcl: ucode[0-3]: %08X %08X %08X %08X\n",
                ucode[0], ucode[1], ucode[2], ucode[3]);
            pr_info("mcl: ucode[4-7]: %08X %08X %08X %08X\n",
                ucode[4], ucode[5], ucode[6], ucode[7]);

            /* PATCH: NOP loop at entry */
            pr_info("mcl: Patching entry point...\n");
            ucode[0] = 0x00000013;
            ucode[1] = 0x00000013;
            ucode[2] = 0x00000013;
            ucode[3] = 0xFF5FF06F;

            /* Copy ONLY the ucode section to fw_pri */
            memcpy(fw_pri, ucode, ucode_size);
            pr_info("mcl: Copied %u bytes of ucode to fw_pri\n", ucode_size);

            /* Submit LOAD_IP_FW with CORRECT parameters */
            memset(cmd,0,sizeof(*cmd));
            cmd->buf_size=sizeof(*cmd); cmd->buf_version=1;
            cmd->cmd_id=0x06;
            cmd->data[0]=lower_32_bits(fw_pri_mc);
            cmd->data[1]=upper_32_bits(fw_pri_mc);
            cmd->data[2]=ucode_size;
            cmd->data[3]=33; /* GFX_FW_TYPE_CP_MES */

            pr_info("mcl: LOAD_IP_FW type=33 size=0x%X...\n", ucode_size);
            ret=submit(psp,NULL,cmd,fence_mc);
            pr_info("mcl: ret=%d resp=0x%X\n", ret, cmd->resp[0]);

            /* Check dmesg for PSP error messages */
        }
    }

    /* MES state after */
    mdelay(500);
    wr(a, 0x1900, (3<<2)); udelay(200);
    pr_info("mcl: AFTER: PC=0x%04X CNTL=0x%X\n",
        rr(a,gc+0x2813), rr(a,gc+0x2807));

    /* Try to activate MES via MES_CNTL */
    {
        u32 cntl = rr(a, gc+0x2807);
        pr_info("mcl: Attempting MES_CNTL write (activate pipe0)...\n");
        /* MES_PIPE0_ACTIVE bit */
        wr(a, gc+0x2807, cntl | (1 << 0)); /* bit 0 = PIPE0_ACTIVE? */
        udelay(200);
        pr_info("mcl: MES_CNTL after = 0x%X\n", rr(a, gc+0x2807));

        mdelay(100);
        pr_info("mcl: PC after activate = 0x%04X\n", rr(a, gc+0x2813));
    }
    wr(a, 0x1900, 0);

    /* Also check PSP error log */
    pr_info("mcl: Check dmesg for PSP errors\n");

done:
    kfree(fw_buf); kfree(cmd); pci_dev_put(p);
    return 0;
}
static void __exit mcl_exit(void) { pr_info("mcl: unloaded\n"); }
module_init(mcl_init); module_exit(mcl_exit);
