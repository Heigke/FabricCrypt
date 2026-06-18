/*
 * psp_load_reset.c — LOAD_IP_FW via PSP ring + GPU reset
 *
 * 1. Copy patched MEC firmware to fw_pri
 * 2. Submit LOAD_IP_FW via PSP ring (loads to TMR, no sig check)
 * 3. Now TMR has our patched firmware
 * 4. User triggers GPU reset (MODE2 preserves TMR)
 * 5. During reinit, driver initializes MEC from (patched) TMR
 */
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
#include <linux/slab.h>
MODULE_LICENSE("GPL");

struct psp_cmd {
    u32 buf_size, buf_version, cmd_id, r1,r2,r3,r4;
    u32 data[32]; u32 resp[16];
};
typedef int (*psp_fn)(void*, void*, struct psp_cmd*, u64);
typedef u32 (*rreg_fn)(void*, u32);

static int __init plr_init(void) {
    struct pci_dev *p = pci_get_device(0x1002,0x1586,NULL);
    void *a, *psp; psp_fn submit; rreg_fn rr;
    struct psp_cmd *cmd; u64 fence_mc, fw_pri_mc;
    void *fw_pri; u32 gc = 0x2800; int ret;

    if (!p) return -ENODEV;
    a = (u8*)pci_get_drvdata(p) - 0x10;
    psp = (u8*)a + 0x3B910;
    fw_pri = (void*)*(u64*)((u8*)psp + 0x058);
    fw_pri_mc = *(u64*)((u8*)psp + 0x050);
    fence_mc = *(u64*)((u8*)psp + 0x1B8);
    submit = (psp_fn)0xFFFFFFFFC0F55840ULL;
    rr = (rreg_fn)0xFFFFFFFFC0E28460ULL;
    cmd = kzalloc(sizeof(*cmd), GFP_KERNEL);
    if (!cmd) { pci_dev_put(p); return -ENOMEM; }

    pr_info("plr: === LOAD_IP_FW + GPU RESET ===\n");
    pr_info("plr: fw_pri=%px mc=0x%llX fence=0x%llX\n", fw_pri, fw_pri_mc, fence_mc);

    /* Step 1: Load + patch firmware to fw_pri */
    {
        struct file *fp = filp_open("/tmp/gc_12_0_1_mec.bin", O_RDONLY, 0);
        loff_t pos = 0; ssize_t n;
        if (IS_ERR(fp)) { pr_info("plr: no fw file\n"); goto done; }
        n = kernel_read(fp, fw_pri, 0x6E000, &pos);
        filp_close(fp, NULL);
        pr_info("plr: Loaded %zd bytes\n", n);

        /* Patch PC=0x800 with NOP loop */
        { u32 *c = (u32*)((u8*)fw_pri + 0x4000);
          pr_info("plr: Orig [0x800]: %08X %08X\n", c[0], c[1]);
          c[0]=0x00000013; c[1]=0x00000013; c[2]=0x00000013; c[3]=0xFF5FF06F;
          pr_info("plr: Patched [0x800]\n"); }

        /* Step 2: Submit LOAD_IP_FW for CP_MEC */
        memset(cmd,0,sizeof(*cmd));
        cmd->buf_size=sizeof(*cmd); cmd->buf_version=1;
        cmd->cmd_id = 0x06; /* LOAD_IP_FW */
        cmd->data[0] = lower_32_bits(fw_pri_mc);
        cmd->data[1] = upper_32_bits(fw_pri_mc);
        cmd->data[2] = n;
        cmd->data[3] = 4; /* GFX_FW_TYPE_CP_MEC */

        pr_info("plr: Submitting LOAD_IP_FW (patched MEC)...\n");
        ret = submit(psp, NULL, cmd, fence_mc);
        pr_info("plr: LOAD ret=%d resp=0x%X\n", ret, cmd->resp[0]);

        /* Also try MEC_ME1 (type 5) and MES (type 33) */
        { int types[] = {5, 33, 34}; int i;
          for (i = 0; i < 3; i++) {
            memset(cmd,0,sizeof(*cmd));
            cmd->buf_size=sizeof(*cmd); cmd->buf_version=1;
            cmd->cmd_id = 0x06;
            cmd->data[0] = lower_32_bits(fw_pri_mc);
            cmd->data[1] = upper_32_bits(fw_pri_mc);
            cmd->data[2] = n;
            cmd->data[3] = types[i];
            ret = submit(psp, NULL, cmd, fence_mc);
            pr_info("plr: LOAD type=%d ret=%d resp=0x%X\n", types[i], ret, cmd->resp[0]);
          }
        }

        /* Step 3: Check current state */
        { u32 pc = rr(a, gc+0x2908);
          u32 ic_lo = rr(a, gc+0x2812);
          pr_info("plr: After LOAD: PC=0x%04X IC_LO=0x%08X\n", pc, ic_lo); }

        /* Step 4: Try AUTOLOAD_RLC */
        memset(cmd,0,sizeof(*cmd));
        cmd->buf_size=sizeof(*cmd); cmd->buf_version=1;
        cmd->cmd_id = 0x21; /* AUTOLOAD_RLC */
        ret = submit(psp, NULL, cmd, fence_mc);
        pr_info("plr: AUTOLOAD ret=%d resp=0x%X\n", ret, cmd->resp[0]);

        mdelay(500);
        { u32 pc = rr(a, gc+0x2908);
          u32 ic_lo = rr(a, gc+0x2812);
          u32 ic_hi = rr(a, gc+0x2813);
          pr_info("plr: After AUTOLOAD: PC=0x%04X IC=0x%08X_%08X\n", pc, ic_hi, ic_lo);
          if (pc >= 0x800 && pc <= 0x803)
            pr_info("plr: *** MEC AT PATCHED PC! ***\n");
        }
    }

    pr_info("plr: Done. If IC_BASE still 0, try: sudo cat .../amdgpu_gpu_recover\n");
done:
    kfree(cmd); pci_dev_put(p);
    return 0;
}
static void __exit plr_exit(void) { pr_info("plr: unloaded\n"); }
module_init(plr_init); module_exit(plr_exit);
