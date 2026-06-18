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
	{ 0x211f9d4e, "mem_section" },
	{ 0xb51225d6, "pcpu_hot" },
	{ 0xbd03ed67, "page_offset_base" },
	{ 0xd272d446, "__SCT__preempt_schedule" },
	{ 0xcbae5412, "__const_udelay" },
	{ 0xe8213e80, "_printk" },
	{ 0x8ac9537f, "pci_get_device" },
	{ 0x1a925fa4, "pci_iomap" },
	{ 0xdcf837ae, "pci_iounmap" },
	{ 0x19bb1bcc, "pci_dev_put" },
	{ 0x90a48d82, "__ubsan_handle_out_of_bounds" },
	{ 0xd272d446, "__x86_return_thunk" },
	{ 0xd272d446, "__fentry__" },
	{ 0xb1ad3f2f, "boot_cpu_data" },
	{ 0xd268ca91, "module_layout" },
};

static const u32 ____version_ext_crcs[]
__used __section("__version_ext_crcs") = {
	0x211f9d4e,
	0xb51225d6,
	0xbd03ed67,
	0xd272d446,
	0xcbae5412,
	0xe8213e80,
	0x8ac9537f,
	0x1a925fa4,
	0xdcf837ae,
	0x19bb1bcc,
	0x90a48d82,
	0xd272d446,
	0xd272d446,
	0xb1ad3f2f,
	0xd268ca91,
};
static const char ____version_ext_names[]
__used __section("__version_ext_names") =
	"mem_section\0"
	"pcpu_hot\0"
	"page_offset_base\0"
	"__SCT__preempt_schedule\0"
	"__const_udelay\0"
	"_printk\0"
	"pci_get_device\0"
	"pci_iomap\0"
	"pci_iounmap\0"
	"pci_dev_put\0"
	"__ubsan_handle_out_of_bounds\0"
	"__x86_return_thunk\0"
	"__fentry__\0"
	"boot_cpu_data\0"
	"module_layout\0"
;

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "930348EC85AE0C685AD521D");
