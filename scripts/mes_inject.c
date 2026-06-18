#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
#include <linux/slab.h>
#include <linux/kthread.h>
MODULE_LICENSE("GPL");

typedef u32 (*rreg_fn)(void*, u32);
typedef void (*wreg_fn)(void*, u32, u32);

static void *g_fw_pri_buf;
static volatile int g_stop;
static volatile int g_patches;

/* High-speed fw_pri re-patcher thread */
static int patcher_thread(void *data)
{
    u32 nop = 0x00000013;
    u32 jal = 0xFF5FF06F;

    pr_info("mi: Patcher thread running (patching fw_pri every 1ms)\n");

    while (!g_stop && !kthread_should_stop()) {
        u32 v;
        /* Check if driver overwrote our patch at blob+0x4000 (PC=0x800) */
        copy_from_kernel_nofault(&v, (u8*)g_fw_pri_buf + 0x4000, 4);

        if (v == 0x00D67633) {
            /* Original firmware detected — RE-PATCH! */
            copy_to_kernel_nofault((u8*)g_fw_pri_buf + 0x4000, &nop, 4);
            copy_to_kernel_nofault((u8*)g_fw_pri_buf + 0x4004, &nop, 4);
            copy_to_kernel_nofault((u8*)g_fw_pri_buf + 0x4008, &nop, 4);
            copy_to_kernel_nofault((u8*)g_fw_pri_buf + 0x400C, &jal, 4);
            g_patches++;
            pr_info("mi: Re-patched fw_pri! (count=%d)\n", g_patches);
        }

        usleep_range(500, 1000); /* 0.5-1ms polling */
    }

    pr_info("mi: Patcher thread stopping (total patches=%d)\n", g_patches);
    return 0;
}

static int __init mi_init(void)
{
    struct pci_dev *p = pci_get_device(0x1002, 0x1586, NULL);
    void *a, *psp;
    struct task_struct *patcher;

    if (!p) return -ENODEV;
    a = (u8*)pci_get_drvdata(p) - 0x10;
    psp = (u8*)a + 0x3B910;
    g_fw_pri_buf = (void*)*(u64*)((u8*)psp + 0x058);
    g_stop = 0;
    g_patches = 0;

    pr_info("mi: === MEC FIRMWARE INJECTION VIA FW_PRI RACE ===\n");
    pr_info("mi: fw_pri_buf = %px\n", g_fw_pri_buf);

    /* Step 1: Load original firmware to fw_pri and patch it */
    {
        struct file *fp = filp_open("/tmp/gc_12_0_1_mec.bin", O_RDONLY, 0);
        loff_t pos = 0;
        ssize_t n;
        if (IS_ERR(fp)) {
            pr_info("mi: No firmware file\n");
            pci_dev_put(p);
            return -ENOENT;
        }
        n = kernel_read(fp, g_fw_pri_buf, 0x6E000, &pos);
        filp_close(fp, NULL);

        /* Patch PC=0x800 */
        {
            u32 *code = (u32*)((u8*)g_fw_pri_buf + 0x4000);
            pr_info("mi: Orig [0x800]: %08X %08X %08X %08X\n",
                code[0], code[1], code[2], code[3]);
            code[0] = 0x00000013;
            code[1] = 0x00000013;
            code[2] = 0x00000013;
            code[3] = 0xFF5FF06F;
            pr_info("mi: Patched. Loaded %zd bytes.\n", n);
        }
    }

    /* Step 2: Start patcher thread (continuously re-patches fw_pri) */
    patcher = kthread_run(patcher_thread, NULL, "mi_patcher");
    if (IS_ERR(patcher))
        pr_info("mi: Failed to start patcher thread\n");

    /* Step 3: The module stays loaded.
     * User triggers GPU reset from userspace:
     *   sudo cat /sys/kernel/debug/dri/0000:c3:00.0/amdgpu_gpu_recover
     *
     * During reset reinit, the driver copies firmware to fw_pri via
     * psp_copy_fw(). Our patcher thread detects the overwrite and
     * immediately re-patches fw_pri before PSP reads it. */
    pr_info("mi: Module loaded. Patcher running.\n");
    pr_info("mi: NOW RUN FROM USERSPACE:\n");
    pr_info("mi:   sudo cat /sys/kernel/debug/dri/0000:c3:00.0/amdgpu_gpu_recover\n");
    pr_info("mi: Then check dmesg for results.\n");

    pci_dev_put(p);
    return 0; /* Stay loaded! */
}

static void __exit mi_exit(void)
{
    g_stop = 1;
    msleep(100);
    pr_info("mi: Unloaded. Total patches applied: %d\n", g_patches);
}
module_init(mi_init);
module_exit(mi_exit);
