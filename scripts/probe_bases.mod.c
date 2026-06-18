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
	{ 0x81a1a811, "_raw_spin_unlock_irqrestore" },
	{ 0x8ac9537f, "pci_get_device" },
	{ 0xe8213e80, "_printk" },
	{ 0x1a925fa4, "pci_iomap" },
	{ 0xdcf837ae, "pci_iounmap" },
	{ 0x19bb1bcc, "pci_dev_put" },
	{ 0xd272d446, "__x86_return_thunk" },
	{ 0xd272d446, "__fentry__" },
	{ 0xe1e1f979, "_raw_spin_lock_irqsave" },
	{ 0xd268ca91, "module_layout" },
};

static const u32 ____version_ext_crcs[]
__used __section("__version_ext_crcs") = {
	0x81a1a811,
	0x8ac9537f,
	0xe8213e80,
	0x1a925fa4,
	0xdcf837ae,
	0x19bb1bcc,
	0xd272d446,
	0xd272d446,
	0xe1e1f979,
	0xd268ca91,
};
static const char ____version_ext_names[]
__used __section("__version_ext_names") =
	"_raw_spin_unlock_irqrestore\0"
	"pci_get_device\0"
	"_printk\0"
	"pci_iomap\0"
	"pci_iounmap\0"
	"pci_dev_put\0"
	"__x86_return_thunk\0"
	"__fentry__\0"
	"_raw_spin_lock_irqsave\0"
	"module_layout\0"
;

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "03B91073BAD3470639A71B4");
