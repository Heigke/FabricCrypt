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
	{ 0x19bb1bcc, "pci_dev_put" },
	{ 0xe1e1f979, "_raw_spin_lock_irqsave" },
	{ 0x81a1a811, "_raw_spin_unlock_irqrestore" },
	{ 0xcbae5412, "__const_udelay" },
	{ 0x8ac9537f, "pci_get_device" },
	{ 0x1a925fa4, "pci_iomap" },
	{ 0x5a844b26, "__x86_indirect_thunk_r13" },
	{ 0x5a844b26, "__x86_indirect_thunk_rbx" },
	{ 0x97dd6ca9, "ioremap" },
	{ 0x12ad300e, "iounmap" },
	{ 0x635ab929, "param_ops_int" },
	{ 0xd272d446, "__fentry__" },
	{ 0x1c489eb6, "register_kprobe" },
	{ 0x7a8e92c6, "unregister_kprobe" },
	{ 0xd272d446, "__x86_return_thunk" },
	{ 0xe8213e80, "_printk" },
	{ 0xd272d446, "__stack_chk_fail" },
	{ 0xdcf837ae, "pci_iounmap" },
	{ 0xd268ca91, "module_layout" },
};

static const u32 ____version_ext_crcs[]
__used __section("__version_ext_crcs") = {
	0x19bb1bcc,
	0xe1e1f979,
	0x81a1a811,
	0xcbae5412,
	0x8ac9537f,
	0x1a925fa4,
	0x5a844b26,
	0x5a844b26,
	0x97dd6ca9,
	0x12ad300e,
	0x635ab929,
	0xd272d446,
	0x1c489eb6,
	0x7a8e92c6,
	0xd272d446,
	0xe8213e80,
	0xd272d446,
	0xdcf837ae,
	0xd268ca91,
};
static const char ____version_ext_names[]
__used __section("__version_ext_names") =
	"pci_dev_put\0"
	"_raw_spin_lock_irqsave\0"
	"_raw_spin_unlock_irqrestore\0"
	"__const_udelay\0"
	"pci_get_device\0"
	"pci_iomap\0"
	"__x86_indirect_thunk_r13\0"
	"__x86_indirect_thunk_rbx\0"
	"ioremap\0"
	"iounmap\0"
	"param_ops_int\0"
	"__fentry__\0"
	"register_kprobe\0"
	"unregister_kprobe\0"
	"__x86_return_thunk\0"
	"_printk\0"
	"__stack_chk_fail\0"
	"pci_iounmap\0"
	"module_layout\0"
;

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "50CC49181148D6EB6B6BF65");
