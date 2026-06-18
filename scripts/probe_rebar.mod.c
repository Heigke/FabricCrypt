#include <linux/module.h>
#include <linux/export-internal.h>
#include <linux/compiler.h>

MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};



static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0xe8213e80, "_printk" },
	{ 0xa59dd599, "pci_find_ext_capability" },
	{ 0x22029f10, "pci_read_config_dword" },
	{ 0xa9e0800a, "pci_read_config_word" },
	{ 0x418c4835, "pci_resize_resource" },
	{ 0x0a231cdc, "pci_write_config_word" },
	{ 0xcbae5412, "__const_udelay" },
	{ 0x89c39ff0, "pci_write_config_dword" },
	{ 0xd272d446, "__stack_chk_fail" },
	{ 0x8ac9537f, "pci_get_device" },
	{ 0x19bb1bcc, "pci_dev_put" },
	{ 0xc46670f1, "slow_virt_to_phys" },
	{ 0x97dd6ca9, "ioremap_wc" },
	{ 0xd94d5db9, "filp_open" },
	{ 0xbe001524, "kernel_write" },
	{ 0x02f9bbf0, "init_timer_key" },
	{ 0x12ad300e, "iounmap" },
	{ 0x1b60315e, "copy_from_kernel_nofault" },
	{ 0x1f8cc9e3, "filp_close" },
	{ 0x2352b148, "timer_delete_sync" },
	{ 0x635ab929, "param_ops_int" },
	{ 0xd272d446, "__fentry__" },
	{ 0x058c185a, "jiffies" },
	{ 0x32feeafc, "mod_timer" },
	{ 0xd272d446, "__x86_return_thunk" },
	{ 0xd268ca91, "module_layout" },
};

static const u32 ____version_ext_crcs[]
__used __section("__version_ext_crcs") = {
	0xe8213e80,
	0xa59dd599,
	0x22029f10,
	0xa9e0800a,
	0x418c4835,
	0x0a231cdc,
	0xcbae5412,
	0x89c39ff0,
	0xd272d446,
	0x8ac9537f,
	0x19bb1bcc,
	0xc46670f1,
	0x97dd6ca9,
	0xd94d5db9,
	0xbe001524,
	0x02f9bbf0,
	0x12ad300e,
	0x1b60315e,
	0x1f8cc9e3,
	0x2352b148,
	0x635ab929,
	0xd272d446,
	0x058c185a,
	0x32feeafc,
	0xd272d446,
	0xd268ca91,
};
static const char ____version_ext_names[]
__used __section("__version_ext_names") =
	"_printk\0"
	"pci_find_ext_capability\0"
	"pci_read_config_dword\0"
	"pci_read_config_word\0"
	"pci_resize_resource\0"
	"pci_write_config_word\0"
	"__const_udelay\0"
	"pci_write_config_dword\0"
	"__stack_chk_fail\0"
	"pci_get_device\0"
	"pci_dev_put\0"
	"slow_virt_to_phys\0"
	"ioremap_wc\0"
	"filp_open\0"
	"kernel_write\0"
	"init_timer_key\0"
	"iounmap\0"
	"copy_from_kernel_nofault\0"
	"filp_close\0"
	"timer_delete_sync\0"
	"param_ops_int\0"
	"__fentry__\0"
	"jiffies\0"
	"mod_timer\0"
	"__x86_return_thunk\0"
	"module_layout\0"
;

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "5ECB302EF8D322DDACAAD5B");
