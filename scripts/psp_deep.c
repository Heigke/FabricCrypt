#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
#include <linux/slab.h>
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

static int __init pd_init(void) {
    struct pci_dev *p = pci_get_device(0x1002, 0x1586, NULL);
    void *adev, *psp;
    psp_fn submit;
    struct psp_cmd *cmd;
    u64 fence_mc, fw_pri_mc;
    void *fw_pri_buf;
    int ret;

    if (!p) return -ENODEV;
    adev = (u8*)pci_get_drvdata(p) - 0x10;
    psp = (u8*)adev + 0x3B910;
    fence_mc = *(u64*)((u8*)psp + 0x1B8);
    fw_pri_mc = *(u64*)((u8*)psp + 0x050);
    fw_pri_buf = (void*)*(u64*)((u8*)psp + 0x058);
    submit = (psp_fn)0xFFFFFFFFC0F2F840ULL;
    cmd = kzalloc(sizeof(*cmd), GFP_KERNEL);
    if (!cmd) { pci_dev_put(p); return -ENODEV; }

    pr_info("pd: mode=%d fence=0x%llX fw_pri_mc=0x%llX fw_pri_buf=%px\n",
        mode, fence_mc, fw_pri_mc, fw_pri_buf);

    if (mode == 1) {
        /* Verify fw_pri_buf readability first */
        u32 v[8]; int i;
        for (i = 0; i < 8; i++)
            copy_from_kernel_nofault(&v[i], (u8*)fw_pri_buf + i*4, 4);
        pr_info("pd: fw_pri_buf pre: %08X %08X %08X %08X %08X %08X %08X %08X\n",
            v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]);

        /* Write a canary to fw_pri_buf to test writability */
        {
            u32 canary = 0xDEADBEEF;
            copy_to_kernel_nofault((u8*)fw_pri_buf, &canary, 4);
            copy_from_kernel_nofault(&v[0], (u8*)fw_pri_buf, 4);
            pr_info("pd: canary readback: 0x%08X\n", v[0]);
        }

        /* Now submit SAVE and re-read */
        cmd->buf_size = sizeof(*cmd);
        cmd->buf_version = 1;
        cmd->cmd_id = 0x08;
        cmd->data[0] = 1; /* save_fw = 1 */
        cmd->data[1] = lower_32_bits(fw_pri_mc);
        cmd->data[2] = upper_32_bits(fw_pri_mc);
        cmd->data[3] = 0x80000; /* 512KB */
        cmd->data[4] = 4; /* CP_MEC */

        pr_info("pd: Submitting SAVE CP_MEC...\n");
        ret = submit(psp, NULL, cmd, fence_mc);
        pr_info("pd: ret=%d cmd_resp[0-3]: %08X %08X %08X %08X\n",
            ret, cmd->resp[0], cmd->resp[1], cmd->resp[2], cmd->resp[3]);

        /* Read buffer after save */
        for (i = 0; i < 8; i++)
            copy_from_kernel_nofault(&v[i], (u8*)fw_pri_buf + i*4, 4);
        pr_info("pd: fw_pri_buf post: %08X %08X %08X %08X %08X %08X %08X %08X\n",
            v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]);

        /* Try different fw_types */
        {
            int types[] = {1,2,3,4,5,6,8,9,24,33,34};
            int t;
            for (t = 0; t < 11; t++) {
                u32 canary2 = 0xCAFE0000 + types[t];
                copy_to_kernel_nofault((u8*)fw_pri_buf, &canary2, 4);

                memset(cmd, 0, sizeof(*cmd));
                cmd->buf_size = sizeof(*cmd);
                cmd->buf_version = 1;
                cmd->cmd_id = 0x08;
                cmd->data[0] = 1;
                cmd->data[1] = lower_32_bits(fw_pri_mc);
                cmd->data[2] = upper_32_bits(fw_pri_mc);
                cmd->data[3] = 0x80000;
                cmd->data[4] = types[t];

                ret = submit(psp, NULL, cmd, fence_mc);
                copy_from_kernel_nofault(&v[0], (u8*)fw_pri_buf, 4);
                copy_from_kernel_nofault(&v[1], (u8*)fw_pri_buf+4, 4);
                pr_info("pd: SAVE type=%d: ret=%d resp=0x%X buf[0]=0x%08X [1]=0x%08X%s\n",
                    types[t], ret, cmd->resp[0], v[0], v[1],
                    v[0] != canary2 ? " *** CHANGED ***" : " (unchanged)");
            }
        }
    }

    if (mode == 2) {
        /* BOOT_CFG SET — try to set boot config bits */
        /* First GET current config */
        cmd->buf_size = sizeof(*cmd); cmd->buf_version = 1;
        cmd->cmd_id = 0x22;
        cmd->data[1] = 2; /* GET */
        cmd->data[3] = 0xFFFFFFFF;
        ret = submit(psp, NULL, cmd, fence_mc);
        pr_info("pd: BOOT_CFG GET: ret=%d config=0x%08X valid=0x%08X\n",
            ret, cmd->data[2], cmd->data[3]);

        /* Try SET with each bit */
        {
            int bit;
            for (bit = 0; bit < 8; bit++) {
                memset(cmd, 0, sizeof(*cmd));
                cmd->buf_size = sizeof(*cmd); cmd->buf_version = 1;
                cmd->cmd_id = 0x22;
                cmd->data[1] = 1; /* SET */
                cmd->data[2] = (1 << bit); /* boot_config */
                cmd->data[3] = (1 << bit); /* valid mask */

                ret = submit(psp, NULL, cmd, fence_mc);
                pr_info("pd: SET bit%d: ret=%d resp=0x%X\n",
                    bit, ret, cmd->resp[0]);

                /* Read back */
                memset(cmd, 0, sizeof(*cmd));
                cmd->buf_size = sizeof(*cmd); cmd->buf_version = 1;
                cmd->cmd_id = 0x22;
                cmd->data[1] = 2; /* GET */
                cmd->data[3] = 0xFFFFFFFF;
                submit(psp, NULL, cmd, fence_mc);
                pr_info("pd:   config now = 0x%08X\n", cmd->data[2]);

                /* Clear it back */
                memset(cmd, 0, sizeof(*cmd));
                cmd->buf_size = sizeof(*cmd); cmd->buf_version = 1;
                cmd->cmd_id = 0x22;
                cmd->data[1] = 1; /* SET */
                cmd->data[2] = 0;
                cmd->data[3] = (1 << bit);
                submit(psp, NULL, cmd, fence_mc);
            }
        }
    }

    kfree(cmd);
    pci_dev_put(p);
    return 0;
}
static void __exit pd_exit(void) { pr_info("pd: unloaded\n"); }
module_init(pd_init); module_exit(pd_exit);
