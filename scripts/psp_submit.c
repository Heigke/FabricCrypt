#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
#include <linux/slab.h>
MODULE_LICENSE("GPL");
static int mode = 1;
module_param(mode, int, 0644);

/* PSP command response buffer */
struct psp_gfx_cmd_resp {
    u32 buf_size;
    u32 buf_version;
    u32 cmd_id;
    u32 resp_buf_addr_lo;
    u32 resp_buf_addr_hi;
    u32 resp_offset;
    u32 resp_buf_size;
    union {
        struct { u32 lo; u32 hi; u32 size; u32 flags; u32 sys_lo; u32 sys_hi; } tmr;
        struct { u32 save; u32 lo; u32 hi; u32 size; u32 fw_type; } save_restore;
        struct { u32 reg_value; u32 reg_id; } reg_prog;
        struct { u32 timestamp; u32 sub_cmd; u32 boot_config; u32 boot_config_valid; } boot_cfg;
        struct { u32 lo; u32 hi; u32 size; u32 fw_type; } load_fw;
        u32 raw[32];
    } cmd;
    u32 resp_data[16];
};

typedef int (*psp_submit_fn)(void *psp, void *ucode,
    struct psp_gfx_cmd_resp *cmd, u64 fence_mc_addr);

static int __init ps_init(void) {
    struct pci_dev *p = pci_get_device(0x1002, 0x1586, NULL);
    void *drm, *adev, *psp;
    psp_submit_fn submit;
    struct psp_gfx_cmd_resp *cmd;
    u64 fence_mc;
    int ret;

    if (!p) return -ENODEV;
    drm = pci_get_drvdata(p);
    adev = (u8*)drm - 0x10;
    psp = (u8*)adev + 0x3B910;  /* PSP context offset */
    fence_mc = *(u64*)((u8*)psp + 0x1B8);  /* fence_buf_mc_addr */
    submit = (psp_submit_fn)0xFFFFFFFFC0F2F840ULL;

    pr_info("ps: PSP at %px fence_mc=0x%llX mode=%d\n", psp, fence_mc, mode);

    cmd = kzalloc(sizeof(*cmd), GFP_KERNEL);
    if (!cmd) { pci_dev_put(p); return -ENOMEM; }

    if (mode == 1) {
        /* BOOT_CFG GET */
        cmd->buf_size = sizeof(*cmd);
        cmd->buf_version = 1;
        cmd->cmd_id = 0x22;
        cmd->cmd.boot_cfg.sub_cmd = 2; /* GET */
        cmd->cmd.boot_cfg.boot_config_valid = 0xFFFFFFFF;

        pr_info("ps: Submitting BOOT_CFG GET...\n");
        ret = submit(psp, NULL, cmd, fence_mc);
        pr_info("ps: ret=%d status=0x%X\n", ret, cmd->resp_data[0]);
        pr_info("ps: boot_config=0x%08X valid=0x%08X\n",
            cmd->cmd.boot_cfg.boot_config, cmd->cmd.boot_cfg.boot_config_valid);
        pr_info("ps: resp: %08X %08X %08X %08X\n",
            cmd->resp_data[0], cmd->resp_data[1],
            cmd->resp_data[2], cmd->resp_data[3]);
    }

    if (mode == 2) {
        /* GET_FW_ATTESTATION */
        cmd->buf_size = sizeof(*cmd);
        cmd->buf_version = 1;
        cmd->cmd_id = 0x0F;

        pr_info("ps: Submitting GET_FW_ATTESTATION...\n");
        ret = submit(psp, NULL, cmd, fence_mc);
        pr_info("ps: ret=%d\n", ret);
        pr_info("ps: resp: %08X %08X %08X %08X %08X %08X %08X %08X\n",
            cmd->cmd.raw[0], cmd->cmd.raw[1], cmd->cmd.raw[2], cmd->cmd.raw[3],
            cmd->cmd.raw[4], cmd->cmd.raw[5], cmd->cmd.raw[6], cmd->cmd.raw[7]);
    }

    if (mode == 3) {
        /* SAVE_RESTORE — save MEC firmware from TMR to GART buffer */
        u64 fw_pri_mc = *(u64*)((u8*)psp + 0x050);  /* fw_pri_mc_addr (GART) */
        pr_info("ps: SAVE MEC FW to fw_pri at mc=0x%llX\n", fw_pri_mc);

        cmd->buf_size = sizeof(*cmd);
        cmd->buf_version = 1;
        cmd->cmd_id = 0x08; /* SAVE_RESTORE */
        cmd->cmd.save_restore.save = 1; /* SAVE */
        cmd->cmd.save_restore.lo = lower_32_bits(fw_pri_mc);
        cmd->cmd.save_restore.hi = upper_32_bits(fw_pri_mc);
        cmd->cmd.save_restore.size = 0x80000; /* 512KB */
        cmd->cmd.save_restore.fw_type = 4; /* CP_MEC */

        pr_info("ps: Submitting SAVE_RESTORE (save CP_MEC)...\n");
        ret = submit(psp, NULL, cmd, fence_mc);
        pr_info("ps: ret=%d status=0x%X\n", ret, cmd->resp_data[0]);

        if (ret == 0) {
            /* Read saved firmware from fw_pri buffer! */
            void *fw_pri_buf = (void*)*(u64*)((u8*)psp + 0x058);
            if (fw_pri_buf) {
                u32 code[8]; int i;
                for (i = 0; i < 8; i++)
                    copy_from_kernel_nofault(&code[i],
                        (void*)(unsigned long)((u64)fw_pri_buf + i*4), 4);
                pr_info("ps: *** SAVED FW[0-7]: %08X %08X %08X %08X %08X %08X %08X %08X ***\n",
                    code[0], code[1], code[2], code[3],
                    code[4], code[5], code[6], code[7]);
            }
        }
    }

    if (mode == 4) {
        /* PROG_REG — program register via PSP */
        /* PSP_REG_IH_RB_CNTL = 0 — test with known safe reg */
        cmd->buf_size = sizeof(*cmd);
        cmd->buf_version = 1;
        cmd->cmd_id = 0x0B; /* PROG_REG */
        cmd->cmd.reg_prog.reg_value = 0;
        cmd->cmd.reg_prog.reg_id = 0; /* PSP_REG_IH_RB_CNTL */

        pr_info("ps: Submitting PROG_REG (reg_id=0, val=0)...\n");
        ret = submit(psp, NULL, cmd, fence_mc);
        pr_info("ps: ret=%d status=0x%X\n", ret, cmd->resp_data[0]);
    }

    kfree(cmd);
    pci_dev_put(p);
    return 0;
}
static void __exit ps_exit(void) { pr_info("ps: unloaded\n"); }
module_init(ps_init); module_exit(ps_exit);
