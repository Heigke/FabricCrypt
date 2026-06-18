#include <linux/module.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>
MODULE_LICENSE("GPL");

typedef u32 (*rreg_fn)(void*, u32);
typedef void (*wreg_fn)(void*, u32, u32);

static int __init pdw_init(void) {
    struct pci_dev *p = pci_get_device(0x1002, 0x1586, NULL);
    void *adev;
    rreg_fn rreg;
    wreg_fn wreg;
    u32 gc;
    void __iomem *tmr_map;

    if (!p) return -ENODEV;
    adev = (u8*)pci_get_drvdata(p) - 0x10;
    rreg = (rreg_fn)0xFFFFFFFFC0E02460ULL;
    wreg = (wreg_fn)0xFFFFFFFFC0E02820ULL;
    gc = 0x2800;

    pr_info("dw: === POST-DESTROY TMR STATE ===\n");

    /* Check 1: Is TMR still accessible? */
    tmr_map = ioremap_wc(0x2060000000ULL, 0x100000); /* TMR BO phys */
    if (tmr_map) {
        u32 v = readl(tmr_map);
        pr_info("dw: TMR BO[0] = 0x%08X%s\n", v,
            v == 0xFFFFFFFF ? " PROTECTED" :
            (v == 0 ? " ACCESSIBLE(zero)" : " ACCESSIBLE(data!)"));

        /* Try WRITING to TMR! */
        writel(0xCAFE1337, tmr_map);
        wmb();
        udelay(100);
        v = readl(tmr_map);
        pr_info("dw: TMR write test: 0x%08X%s\n", v,
            v == 0xCAFE1337 ? " *** TMR WRITABLE! ***" : " write failed");

        if (v == 0xCAFE1337) {
            pr_info("dw: *** TMR IS READ/WRITE ACCESSIBLE! ***\n");

            /* Write firmware directly to TMR! */
            /* Copy MEC firmware code section to TMR */
            {
                struct file *fp = filp_open("/tmp/gc_12_0_1_mec.bin", O_RDONLY, 0);
                if (!IS_ERR(fp)) {
                    void *fw_buf = kmalloc(0x6E000, GFP_KERNEL);
                    if (fw_buf) {
                        loff_t pos = 0;
                        ssize_t n = kernel_read(fp, fw_buf, 0x6E000, &pos);
                        pr_info("dw: Loaded %zd bytes of firmware\n", n);

                        /* Patch PC=0x800 */
                        {
                            u32 *code = (u32*)((u8*)fw_buf + 0x4000);
                            pr_info("dw: Orig [0x800]: %08X %08X %08X %08X\n",
                                code[0], code[1], code[2], code[3]);
                            code[0] = 0x00000013; /* NOP */
                            code[1] = 0x00000013;
                            code[2] = 0x00000013;
                            code[3] = 0xFF5FF06F; /* JAL loop */
                        }

                        /* Write code section (from blob+0x2000) to TMR+0 */
                        {
                            int i;
                            u32 *src = (u32*)((u8*)fw_buf + 0x2000);
                            int code_words = (n - 0x2000) / 4;
                            pr_info("dw: Writing %d words to TMR...\n", code_words);
                            for (i = 0; i < code_words && i < 0x40000; i++)
                                writel(src[i], tmr_map + i * 4);
                            wmb();

                            /* Verify */
                            {
                                u32 v0 = readl(tmr_map);
                                u32 v1 = readl(tmr_map + 4);
                                u32 v800 = readl(tmr_map + 0x800 * 4);
                                pr_info("dw: TMR[0]=0x%08X (expect 0x04070663)\n", v0);
                                pr_info("dw: TMR[1]=0x%08X (expect 0x00060663)\n", v1);
                                pr_info("dw: TMR[0x800]=0x%08X (expect 0x00000013)\n", v800);
                            }
                        }

                        /* Now set IC_BASE to point to TMR BO MC addr */
                        {
                            u64 tmr_mc = 0x97E0000000ULL;
                            u32 ic_lo_orig = rreg(adev, gc + 0x2812);
                            u32 ic_hi_orig = rreg(adev, gc + 0x2813);
                            pr_info("dw: IC_BASE before: 0x%08X_%08X\n",
                                ic_hi_orig, ic_lo_orig);

                            /* Try writing IC_BASE! */
                            wreg(adev, gc + 0x2812, lower_32_bits(tmr_mc));
                            wreg(adev, gc + 0x2813, upper_32_bits(tmr_mc));
                            udelay(200);

                            {
                                u32 new_lo = rreg(adev, gc + 0x2812);
                                u32 new_hi = rreg(adev, gc + 0x2813);
                                pr_info("dw: IC_BASE after write: 0x%08X_%08X\n",
                                    new_hi, new_lo);
                                if (new_lo == lower_32_bits(tmr_mc))
                                    pr_info("dw: *** IC_BASE IS WRITABLE NOW! ***\n");
                                else
                                    pr_info("dw: IC_BASE still locked\n");
                            }

                            /* IC invalidate + MEC restart */
                            {
                                u32 ic_op;
                                /* Halt */
                                wreg(adev, gc + 0x2808, (1 << 30));
                                udelay(2000);

                                /* IC invalidate */
                                ic_op = rreg(adev, gc + 0x2815);
                                wreg(adev, gc + 0x2815, ic_op | 1);
                                udelay(2000);

                                /* Set PRGRM_CNTR_START = 0x800 */
                                wreg(adev, gc + 0x2900, 0x800);
                                udelay(100);

                                /* Unhalt */
                                wreg(adev, gc + 0x2808, 0);
                                mdelay(100);

                                {
                                    u32 pc = rreg(adev, gc + 0x2908);
                                    u32 s0 = rreg(adev, gc + 0x3460);
                                    pr_info("dw: *** FINAL: PC=0x%04X SCRATCH0=0x%X ***\n",
                                        pc, s0);
                                    if (pc >= 0x800 && pc <= 0x803)
                                        pr_info("dw: *** MEC EXECUTING PATCHED CODE! ***\n");
                                }
                            }
                        }

                        kfree(fw_buf);
                    }
                    filp_close(fp, NULL);
                }
            }
        }

        /* Cleanup: zero TMR[0] */
        writel(0, tmr_map);
        iounmap(tmr_map);
    } else {
        pr_info("dw: TMR ioremap failed\n");
    }

    pci_dev_put(p);
    return 0;
}
static void __exit pdw_exit(void) { pr_info("dw: unloaded\n"); }
module_init(pdw_init); module_exit(pdw_exit);
