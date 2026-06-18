#include <linux/module.h>
#include <linux/pci.h>
#include <linux/delay.h>
#include <linux/slab.h>
MODULE_LICENSE("GPL");
struct psp_cmd { u32 buf_size, buf_version, cmd_id, r1,r2,r3,r4; u32 data[32]; u32 resp[16]; };
typedef int (*psp_fn)(void*, void*, struct psp_cmd*, u64);
static int __init cl_init(void) {
    struct pci_dev *p = pci_get_device(0x1002,0x1586,NULL);
    void *a, *psp; psp_fn submit;
    struct psp_cmd *cmd; u64 fence_mc, fw_pri_mc; void *fw_pri;
    int ret;
    if (!p) return -ENODEV;
    a=(u8*)pci_get_drvdata(p)-0x10;
    psp=(u8*)a+0x3B910;
    fw_pri=(void*)*(u64*)((u8*)psp+0x058);
    fw_pri_mc=*(u64*)((u8*)psp+0x050);
    fence_mc=*(u64*)((u8*)psp+0x1B8);
    submit=(psp_fn)0xFFFFFFFFC0E18840ULL;
    cmd=kzalloc(sizeof(*cmd),GFP_KERNEL);
    if (!cmd) { pci_dev_put(p); return -ENOMEM; }

    pr_info("cl: === CLEAN BOOT LOAD_IP_FW TEST ===\n");

    /* Test 1: Load UNMODIFIED MEC firmware (type=4) */
    { struct file *fp = filp_open("/tmp/gc_12_0_1_mec.bin", O_RDONLY, 0);
      loff_t pos = 0; ssize_t n;
      if (IS_ERR(fp)) goto done;
      n = kernel_read(fp, fw_pri, 0x6E000, &pos);
      filp_close(fp, NULL);
      pr_info("cl: Loaded %zd bytes UNMODIFIED MEC fw to fw_pri\n", n);

      memset(cmd,0,sizeof(*cmd));
      cmd->buf_size=sizeof(*cmd); cmd->buf_version=1;
      cmd->cmd_id=0x06;
      cmd->data[0]=lower_32_bits(fw_pri_mc);
      cmd->data[1]=upper_32_bits(fw_pri_mc);
      cmd->data[2]=n;
      cmd->data[3]=4; /* CP_MEC */
      pr_info("cl: LOAD_IP_FW type=4 UNMODIFIED...\n");
      ret = submit(psp, NULL, cmd, fence_mc);
      pr_info("cl: ret=%d\n", ret);
    }

    /* Check dmesg for PSP status 0x11 */
    mdelay(200);

    /* Test 2: Now try with ONE BYTE CHANGED */
    { u32 orig;
      copy_from_kernel_nofault(&orig, (u8*)fw_pri + 0x4000, 4);
      pr_info("cl: Original [0x800] = 0x%08X\n", orig);

      /* Patch ONE instruction */
      { u32 nop = 0x00000013;
        copy_to_kernel_nofault((u8*)fw_pri + 0x4000, &nop, 4); }

      memset(cmd,0,sizeof(*cmd));
      cmd->buf_size=sizeof(*cmd); cmd->buf_version=1;
      cmd->cmd_id=0x06;
      cmd->data[0]=lower_32_bits(fw_pri_mc);
      cmd->data[1]=upper_32_bits(fw_pri_mc);
      cmd->data[2]=447456;
      cmd->data[3]=4;
      pr_info("cl: LOAD_IP_FW type=4 PATCHED (1 byte diff)...\n");
      ret = submit(psp, NULL, cmd, fence_mc);
      pr_info("cl: ret=%d\n", ret);

      /* Restore */
      copy_to_kernel_nofault((u8*)fw_pri + 0x4000, &orig, 4);
    }

    /* Test 3: Also try MEC_ME1 (type=5) */
    { memset(cmd,0,sizeof(*cmd));
      cmd->buf_size=sizeof(*cmd); cmd->buf_version=1;
      cmd->cmd_id=0x06;
      cmd->data[0]=lower_32_bits(fw_pri_mc);
      cmd->data[1]=upper_32_bits(fw_pri_mc);
      cmd->data[2]=447456;
      cmd->data[3]=5; /* CP_MEC_ME1 */
      pr_info("cl: LOAD_IP_FW type=5 (MEC_ME1)...\n");
      ret = submit(psp, NULL, cmd, fence_mc);
      pr_info("cl: ret=%d\n", ret);
    }

done:
    kfree(cmd); pci_dev_put(p);
    return 0;
}
static void __exit cl_exit(void) { pr_info("cl: unloaded\n"); }
module_init(cl_init); module_exit(cl_exit);
